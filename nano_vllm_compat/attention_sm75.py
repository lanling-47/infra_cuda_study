"""
nano-vllm attention.py 的 sm_75 兼容版本

优化 A：Triton KV store → CUDA float4 vectorized kernel
兼容层：flash_attn → PyTorch SDPA (sm_75 不支持 flash_attn)
"""
import torch
from torch import nn

# ── Attention backend ──────────────────────────────────────────────────────────
_CUDA_PAGED_ATTN = False
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
    print("[attention] Using flash_attn (sm_80+)")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("[attention] flash_attn not available, using sm_75 compatibility layer")
    import sys, os
    _compat_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "compat")
    if _compat_dir not in sys.path:
        sys.path.insert(0, _compat_dir)
    from flash_attn_compat import flash_attn_varlen_func, flash_attn_with_kvcache
    # Try to load our CUDA paged attention (higher priority than SDPA fallback)
    try:
        from paged_attn import paged_attention_decode, is_available as _paged_avail
        _CUDA_PAGED_ATTN = _paged_avail()
        if _CUDA_PAGED_ATTN:
            print("[attention] Using CUDA paged attention (FlashDecoding, sm_75)")
    except Exception as e:
        _CUDA_PAGED_ATTN = False
        print(f"[attention] CUDA paged attention unavailable: {e}")

# ── KV store backend ───────────────────────────────────────────────────────────
try:
    # Try loading our CUDA extension first
    from kv_store import store_kvcache as _store_kvcache_cuda
    # Verify it actually loaded (not just imported)
    from kv_store import _load_ext, _FALLBACK
    if _FALLBACK:
        raise ImportError("CUDA ext fell back to Triton")
    store_kvcache = _store_kvcache_cuda
    print("[attention] Using CUDA KV store (float4 vectorized)")
except Exception as e:
    print(f"[attention] CUDA KV store unavailable ({e}), using Triton fallback")
    import triton
    import triton.language as tl

    @triton.jit
    def _store_kvcache_triton_kernel(
        key_ptr, key_stride, value_ptr, value_stride,
        k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        _store_kvcache_triton_kernel[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping, D,
        )

from nanovllm.utils.context import get_context


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            if not FLASH_ATTN_AVAILABLE and _CUDA_PAGED_ATTN:
                o = paged_attention_decode(
                    q.unsqueeze(1), k_cache, v_cache,
                    context.context_lens, context.block_tables,
                    softmax_scale=self.scale, causal=True,
                )
                return o
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
