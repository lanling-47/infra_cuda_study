"""
Python wrapper for Paged Attention CUDA extension.

JIT-compiles on first import and caches the result.
Falls back to PyTorch SDPA (in flash_attn_compat.py) if compilation fails.

Usage:
    from paged_attn import paged_attention_decode

    # In Attention.forward (decode phase):
    #   q:            [B, 1, num_heads, D]       (from nano-vllm)
    #   k_cache:      [num_blocks, block_size, num_kv_heads, D]
    #   v_cache:      [num_blocks, block_size, num_kv_heads, D]
    #   block_table:  [B, max_blocks]
    #   cache_seqlens:[B]
    #
    # Returns: [B, 1, num_heads, D]
"""
import os
import torch
from torch.utils.cpp_extension import load

_EXT = None
_LOAD_ATTEMPTED = False

def _load_ext():
    global _EXT, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _EXT
    _LOAD_ATTEMPTED = True

    src_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [
        os.path.join(src_dir, "paged_attention_ext.cpp"),
        os.path.join(src_dir, "paged_attention.cu"),
    ]

    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
    else:
        arch = "sm_75"

    try:
        _EXT = load(
            name="paged_attention_cuda",
            sources=sources,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                f"-arch={arch}",
            ],
            verbose=False,
        )
        print(f"[paged_attn] CUDA extension loaded ({arch})")
    except Exception as e:
        print(f"[paged_attn] CUDA extension failed to load: {e}")
        _EXT = None
    return _EXT


def paged_attention_decode(
    q: torch.Tensor,          # [B, 1, num_heads, D]
    k_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, D]
    v_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, D]
    cache_seqlens: torch.Tensor,   # [B]
    block_table: torch.Tensor,     # [B, max_blocks]
    softmax_scale: float,
    causal: bool = True,
) -> torch.Tensor:
    """
    Paged attention for decode phase (one token per sequence).

    Returns: [B, 1, num_heads, D]
    """
    ext = _load_ext()
    if ext is None:
        # Fall back to SDPA-based implementation
        from flash_attn_compat import flash_attn_with_kvcache
        return flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens, block_table,
            softmax_scale=softmax_scale, causal=causal,
        )

    B = q.size(0)
    num_heads = q.size(2)
    D = q.size(3)

    # The kernel expects q: [B, num_heads, D] (squeeze the seq dim)
    q_squeezed = q.squeeze(1).contiguous()   # [B, num_heads, D]

    # Ensure inputs are contiguous
    k_cache_c = k_cache.contiguous()
    v_cache_c = v_cache.contiguous()
    block_table_c = block_table.contiguous()
    cache_seqlens_c = cache_seqlens.contiguous()

    # Detect block_size from k_cache shape
    block_size = k_cache_c.size(1)

    out = ext.paged_attention_forward(
        q_squeezed, k_cache_c, v_cache_c,
        block_table_c, cache_seqlens_c,
        softmax_scale, block_size,
    )
    # out: [B, num_heads, D] -> [B, 1, num_heads, D]
    return out.unsqueeze(1)


def is_available() -> bool:
    """Returns True if the CUDA extension loaded successfully."""
    return _load_ext() is not None
