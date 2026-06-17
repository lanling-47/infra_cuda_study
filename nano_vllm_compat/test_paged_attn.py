"""
Micro-benchmark and correctness test for paged attention CUDA kernel.

Compares:
  1. PyTorch SDPA-based paged attention (flash_attn_compat.flash_attn_with_kvcache)
  2. Our CUDA paged attention (paged_attn.paged_attention_decode)

Usage:
    cd /root/cuda-lab
    python3 nano_vllm_compat/test_paged_attn.py
"""
import sys
import time
import torch

sys.path.insert(0, '/root/cuda-lab/nano_vllm_compat')
from flash_attn_compat import flash_attn_with_kvcache
from paged_attn import paged_attention_decode, is_available


def make_test_inputs(B, num_heads, num_kv_heads, D, seq_len, block_size, device="cuda"):
    """Build paged KV cache + random query for testing."""
    gqa_ratio = num_heads // num_kv_heads
    max_blocks = (seq_len + block_size - 1) // block_size

    # Random K/V in paged layout
    k_cache = torch.randn(max_blocks, block_size, num_kv_heads, D,
                          device=device, dtype=torch.float16)
    v_cache = torch.randn(max_blocks, block_size, num_kv_heads, D,
                          device=device, dtype=torch.float16)

    # Simple block table: block i maps to physical block i
    block_table = torch.arange(max_blocks, device=device, dtype=torch.int32).unsqueeze(0).expand(B, -1).contiguous()

    # All sequences have same length for simplicity
    cache_seqlens = torch.full((B,), seq_len, device=device, dtype=torch.int32)

    # Random queries
    q = torch.randn(B, 1, num_heads, D, device=device, dtype=torch.float16)

    return q, k_cache, v_cache, cache_seqlens, block_table


def correctness_test(B=2, num_heads=16, num_kv_heads=8, D=128, seq_len=64, block_size=16):
    """Verify CUDA kernel matches SDPA reference."""
    q, k_cache, v_cache, cache_seqlens, block_table = make_test_inputs(
        B, num_heads, num_kv_heads, D, seq_len, block_size)
    scale = D ** -0.5

    # Reference: SDPA-based
    ref = flash_attn_with_kvcache(
        q, k_cache, v_cache, cache_seqlens, block_table,
        softmax_scale=scale, causal=True)

    # CUDA kernel
    out = paged_attention_decode(
        q, k_cache, v_cache, cache_seqlens, block_table,
        softmax_scale=scale, causal=True)

    # Compare
    max_diff = (ref - out).abs().max().item()
    mean_diff = (ref - out).abs().mean().item()
    rel_diff = ((ref - out).abs() / (ref.abs() + 1e-6)).max().item()

    print(f"  max abs diff:  {max_diff:.6f}")
    print(f"  mean abs diff: {mean_diff:.6f}")
    print(f"  max rel diff:  {rel_diff:.6f}")
    print(f"  ref range:     [{ref.min().item():.3f}, {ref.max().item():.3f}]")
    print(f"  ✓ PASS" if max_diff < 0.05 else f"  ✗ FAIL (max_diff={max_diff:.4f})")
    return max_diff < 0.05


def latency_test(B=4, num_heads=16, num_kv_heads=8, D=128, seq_len=512, block_size=16,
                 warmup=10, trials=50):
    """Measure kernel latency."""
    q, k_cache, v_cache, cache_seqlens, block_table = make_test_inputs(
        B, num_heads, num_kv_heads, D, seq_len, block_size)
    scale = D ** -0.5

    print(f"\n  Config: B={B}, H={num_heads}, Hkv={num_kv_heads}, D={D}, "
          f"seq={seq_len}, blk={block_size}")

    # SDPA warmup
    for _ in range(warmup):
        _ = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table,
                                     softmax_scale=scale, causal=True)
    torch.cuda.synchronize()

    # SDPA benchmark
    times_sdpa = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table,
                                     softmax_scale=scale, causal=True)
        torch.cuda.synchronize()
        times_sdpa.append((time.perf_counter() - t0) * 1000)

    avg_sdpa = sum(times_sdpa) / len(times_sdpa)
    print(f"  SDPA:     {avg_sdpa:.3f} ms  (min={min(times_sdpa):.3f}, max={max(times_sdpa):.3f})")

    if not is_available():
        print("  CUDA paged attention not available, skipping")
        return

    # CUDA kernel warmup
    for _ in range(warmup):
        _ = paged_attention_decode(q, k_cache, v_cache, cache_seqlens, block_table,
                                    softmax_scale=scale, causal=True)
    torch.cuda.synchronize()

    # CUDA kernel benchmark
    times_cuda = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = paged_attention_decode(q, k_cache, v_cache, cache_seqlens, block_table,
                                    softmax_scale=scale, causal=True)
        torch.cuda.synchronize()
        times_cuda.append((time.perf_counter() - t0) * 1000)

    avg_cuda = sum(times_cuda) / len(times_cuda)
    speedup = avg_sdpa / avg_cuda
    print(f"  CUDA:     {avg_cuda:.3f} ms  (min={min(times_cuda):.3f}, max={max(times_cuda):.3f})")
    print(f"  Speedup:  {speedup:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Paged Attention: Correctness Test")
    print("=" * 60)
    correctness_test()

    print("\n" + "=" * 60)
    print("Paged Attention: Latency Benchmark")
    print("=" * 60)

    # Varying sequence lengths
    for seq_len in [64, 256, 1024, 2048]:
        latency_test(seq_len=seq_len)
