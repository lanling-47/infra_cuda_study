"""
Python wrapper for CUDA KV store kernel.

Usage:
    from kv_store import store_kvcache

The module JIT-compiles the CUDA extension on first import and caches it.
Falls back to the Triton kernel if compilation fails.
"""
import os
import torch
from torch.utils.cpp_extension import load

_EXT = None
_FALLBACK = False

def _load_ext():
    global _EXT, _FALLBACK
    if _EXT is not None:
        return _EXT

    src_dir = os.path.dirname(os.path.abspath(__file__))
    sources = [
        os.path.join(src_dir, "kv_store_cuda_ext.cpp"),
        os.path.join(src_dir, "kv_store_cuda.cu"),
    ]

    # Detect compute capability
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        arch = f"sm_{cap[0]}{cap[1]}"
    else:
        arch = "sm_75"

    try:
        _EXT = load(
            name="kv_store_cuda",
            sources=sources,
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                f"-arch={arch}",
                "-Xptxas=-v",
            ],
            verbose=False,
        )
        print(f"[kv_store] CUDA extension loaded ({arch})")
    except Exception as e:
        print(f"[kv_store] CUDA extension failed, falling back to Triton: {e}")
        _FALLBACK = True
    return _EXT


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    Store K/V into paged KV cache.

    Args:
        key:          [N, num_kv_heads, head_dim]  fp16 or fp32
        value:        [N, num_kv_heads, head_dim]
        k_cache:      [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache:      [num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: [N]  int32, slot = block_id * block_size + token_in_block
                       use -1 to skip (padding)
    """
    ext = _load_ext()
    if _FALLBACK:
        _store_kvcache_triton(key, value, k_cache, v_cache, slot_mapping)
        return
    ext.store_kvcache_cuda(key, value, k_cache, v_cache, slot_mapping)


# ── Triton fallback (original nano-vllm implementation) ──────────────────────

try:
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

    def _store_kvcache_triton(key, value, k_cache, v_cache, slot_mapping):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        _store_kvcache_triton_kernel[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping, D,
        )

except ImportError:
    def _store_kvcache_triton(*args, **kwargs):
        raise RuntimeError("Neither CUDA extension nor Triton is available")
