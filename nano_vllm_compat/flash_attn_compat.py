"""
Flash Attention 兼容层 for sm_75 (Turing, RTX 2080 Ti)

Flash Attention 2 不支持 sm_75，本模块提供等价的实现：
- flash_attn_varlen_func: 用于 prefill (变长序列)
- flash_attn_with_kvcache: 用于 decode (paged KV cache)

实现策略：
- prefill: 使用 PyTorch SDPA (自动选择 memory-efficient attention 或 math backend)
- decode: 手动从 paged KV cache 读取并计算 attention
"""
import torch
import torch.nn.functional as F


def flash_attn_varlen_func(
    q: torch.Tensor,           # [total_tokens, num_heads, head_dim]
    k: torch.Tensor,           # [total_tokens, num_kv_heads, head_dim]
    v: torch.Tensor,           # [total_tokens, num_kv_heads, head_dim]
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float = None,
    causal: bool = True,
    block_table: torch.Tensor = None,  # prefix cache (shape: [batch, max_blocks])
) -> torch.Tensor:
    """
    Variable-length multi-head attention with optional prefix cache.

    Args:
        q: [total_tokens_q, num_heads, head_dim]
        k: [total_tokens_k, num_kv_heads, head_dim]
        v: [total_tokens_k, num_kv_heads, head_dim]
        cu_seqlens_q: [batch+1] cumulative sequence lengths for queries
        cu_seqlens_k: [batch+1] cumulative sequence lengths for keys
        block_table: [batch, max_blocks] for prefix cache (K/V from k_cache/v_cache)

    Returns:
        [total_tokens_q, num_heads, head_dim]
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    batch_size = len(cu_seqlens_q) - 1
    gqa_ratio = num_heads // num_kv_heads

    # Handle prefix cache: read from block_table
    if block_table is not None:
        # k and v are actually the KV cache tensors
        # k_cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        # But in this context, k and v are already the cache views
        # block_table[i, j] = block_id for sequence i, block j
        # We need to gather K/V from the cache according to block_table
        k_cache = k
        v_cache = v

        # Build K/V for each sequence by gathering from cache
        k_list = []
        v_list = []
        for i in range(batch_size):
            seq_len_k = (cu_seqlens_k[i+1] - cu_seqlens_k[i]).item()
            block_size = k_cache.shape[1] if k_cache.dim() == 4 else k_cache.shape[2]

            # Gather tokens from blocks
            # For simplicity, use a loop over tokens
            k_seq = []
            v_seq = []
            for t in range(seq_len_k):
                block_idx = t // block_size
                token_idx = t % block_size
                block_id = block_table[i, block_idx].item()
                if k_cache.dim() == 4:
                    # k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
                    k_seq.append(k_cache[block_id, token_idx])
                    v_seq.append(v_cache[block_id, token_idx])
                else:
                    # k_cache: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
                    # This shouldn't happen in this context
                    k_seq.append(k_cache[0, block_id, token_idx])
                    v_seq.append(v_cache[0, block_id, token_idx])
            k_list.append(torch.stack(k_seq))
            v_list.append(torch.stack(v_seq))
        k = torch.cat(k_list, dim=0)  # [total_tokens_k, num_kv_heads, head_dim]
        v = torch.cat(v_list, dim=0)

    # Split into per-sequence tensors
    outputs = []
    for i in range(batch_size):
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i+1].item()
        k_start = cu_seqlens_k[i].item()
        k_end = cu_seqlens_k[i+1].item()

        q_seq = q[q_start:q_end]   # [seq_q, num_heads, head_dim]
        k_seq = k[k_start:k_end]   # [seq_k, num_kv_heads, head_dim]
        v_seq = v[k_start:k_end]   # [seq_k, num_kv_heads, head_dim]

        # Expand KV for GQA
        if gqa_ratio > 1:
            k_seq = k_seq.repeat_interleave(gqa_ratio, dim=1)  # [seq_k, num_heads, head_dim]
            v_seq = v_seq.repeat_interleave(gqa_ratio, dim=1)

        # Transpose to [1, num_heads, seq, head_dim] for SDPA
        q_sdpa = q_seq.transpose(0, 1).unsqueeze(0)
        k_sdpa = k_seq.transpose(0, 1).unsqueeze(0)
        v_sdpa = v_seq.transpose(0, 1).unsqueeze(0)

        # Use SDPA
        o = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale
        )

        # Transpose back to [seq_q, num_heads, head_dim]
        o = o.squeeze(0).transpose(0, 1)
        outputs.append(o)

    return torch.cat(outputs, dim=0)


def flash_attn_with_kvcache(
    q: torch.Tensor,           # [batch, 1, num_heads, head_dim]
    k_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_heads, head_dim]
    cache_seqlens: torch.Tensor,   # [batch]
    block_table: torch.Tensor,     # [batch, max_blocks]
    softmax_scale: float = None,
    causal: bool = True,
) -> torch.Tensor:
    """
    Paged attention for decode phase.

    Args:
        q: [batch, 1, num_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        cache_seqlens: [batch] - number of tokens in cache for each sequence
        block_table: [batch, max_blocks] - maps logical block to physical block

    Returns:
        [batch, 1, num_heads, head_dim]
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    batch_size = q.shape[0]
    num_heads = q.shape[2]
    num_kv_heads = k_cache.shape[2]
    head_dim = q.shape[3]
    block_size = k_cache.shape[1]
    gqa_ratio = num_heads // num_kv_heads

    outputs = []
    for i in range(batch_size):
        seq_len = cache_seqlens[i].item()

        # Gather K/V from paged cache
        k_list = []
        v_list = []
        for t in range(seq_len):
            block_idx = t // block_size
            token_idx = t % block_size
            block_id = block_table[i, block_idx].item()
            k_list.append(k_cache[block_id, token_idx])
            v_list.append(v_cache[block_id, token_idx])

        if len(k_list) == 0:
            # Empty sequence, return zeros
            outputs.append(torch.zeros(1, num_heads, head_dim, device=q.device, dtype=q.dtype))
            continue

        k_seq = torch.stack(k_list)  # [seq_len, num_kv_heads, head_dim]
        v_seq = torch.stack(v_list)

        # Expand for GQA
        if gqa_ratio > 1:
            k_seq = k_seq.repeat_interleave(gqa_ratio, dim=1)
            v_seq = v_seq.repeat_interleave(gqa_ratio, dim=1)

        # q: [batch, 1, num_heads, head_dim] -> [1, num_heads, 1, head_dim]
        q_seq = q[i:i+1]  # [1, 1, num_heads, head_dim]
        q_sdpa = q_seq.squeeze(1).transpose(0, 1).unsqueeze(2)  # [1, num_heads, 1, head_dim]
        k_sdpa = k_seq.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
        v_sdpa = v_seq.transpose(0, 1).unsqueeze(0)

        # SDPA
        o = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,  # Causal already handled by seq_len (only attending to past tokens)
            scale=softmax_scale
        )
        # o shape: [1, num_heads, 1, head_dim]
        # Need to reshape to [1, 1, num_heads, head_dim]
        o = o.squeeze(0).transpose(0, 1)  # [1, num_heads, head_dim]
        outputs.append(o)

    # Stack to [batch, 1, num_heads, head_dim]
    return torch.stack(outputs, dim=0)
