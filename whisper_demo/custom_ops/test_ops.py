"""
Test custom CUDA operations
Verify correctness before integrating into Whisper
"""

import torch
import sys

try:
    import custom_cuda_ops
    print("✓ custom_cuda_ops loaded successfully")
except ImportError as e:
    print(f"✗ Failed to load custom_cuda_ops: {e}")
    print("Run 'bash build.sh' to build the extension")
    sys.exit(1)

def test_flash_attention():
    print("\n" + "=" * 60)
    print("Testing Flash Attention")
    print("=" * 60)

    N, D = 64, 32
    Q = torch.randn(N, D, device='cuda', dtype=torch.float32)
    K = torch.randn(N, D, device='cuda', dtype=torch.float32)
    V = torch.randn(N, D, device='cuda', dtype=torch.float32)

    # Custom kernel
    O_custom = custom_cuda_ops.flash_attention_forward(Q, K, V, False)

    # PyTorch reference
    scale = 1.0 / (D ** 0.5)
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    O_ref = torch.matmul(attn, V)

    # Compare
    max_diff = (O_custom - O_ref).abs().max().item()
    mean_diff = (O_custom - O_ref).abs().mean().item()

    print(f"Shape: [{N}, {D}]")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("✓ Flash Attention PASSED")
        return True
    else:
        print("✗ Flash Attention FAILED")
        return False

def test_layernorm():
    print("\n" + "=" * 60)
    print("Testing LayerNorm")
    print("=" * 60)

    N, D = 512, 256
    x = torch.randn(N, D, device='cuda', dtype=torch.float32)
    gamma = torch.ones(D, device='cuda', dtype=torch.float32)
    beta = torch.zeros(D, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Custom kernel
    y_custom = custom_cuda_ops.layernorm_forward(x, gamma, beta, eps)

    # PyTorch reference
    y_ref = torch.nn.functional.layer_norm(x, (D,), gamma, beta, eps)

    # Compare
    max_diff = (y_custom - y_ref).abs().max().item()
    mean_diff = (y_custom - y_ref).abs().mean().item()

    print(f"Shape: [{N}, {D}]")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("✓ LayerNorm PASSED")
        return True
    else:
        print("✗ LayerNorm FAILED")
        return False

def test_softmax():
    print("\n" + "=" * 60)
    print("Testing Softmax")
    print("=" * 60)

    N, D = 512, 256
    x = torch.randn(N, D, device='cuda', dtype=torch.float32)

    # Custom kernel
    y_custom = custom_cuda_ops.softmax_forward(x)

    # PyTorch reference
    y_ref = torch.softmax(x, dim=-1)

    # Compare
    max_diff = (y_custom - y_ref).abs().max().item()
    mean_diff = (y_custom - y_ref).abs().mean().item()

    print(f"Shape: [{N}, {D}]")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("✓ Softmax PASSED")
        return True
    else:
        print("✗ Softmax FAILED")
        return False

def benchmark_flash_attention():
    print("\n" + "=" * 60)
    print("Benchmarking Flash Attention")
    print("=" * 60)

    N, D = 512, 64
    Q = torch.randn(N, D, device='cuda', dtype=torch.float32)
    K = torch.randn(N, D, device='cuda', dtype=torch.float32)
    V = torch.randn(N, D, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = custom_cuda_ops.flash_attention_forward(Q, K, V, False)
        torch.cuda.synchronize()

    # Benchmark
    import time
    iters = 100
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = custom_cuda_ops.flash_attention_forward(Q, K, V, False)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000 / iters

    print(f"Shape: [{N}, {D}]")
    print(f"Average time: {elapsed:.3f} ms")

def benchmark_layernorm():
    print("\n" + "=" * 60)
    print("Benchmarking LayerNorm")
    print("=" * 60)

    N, D = 512, 256
    x = torch.randn(N, D, device='cuda', dtype=torch.float32)
    gamma = torch.ones(D, device='cuda', dtype=torch.float32)
    beta = torch.zeros(D, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = custom_cuda_ops.layernorm_forward(x, gamma, beta, 1e-5)
        torch.cuda.synchronize()

    # Benchmark
    import time
    iters = 1000
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = custom_cuda_ops.layernorm_forward(x, gamma, beta, 1e-5)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000 / iters

    print(f"Shape: [{N}, {D}]")
    print(f"Average time: {elapsed:.3f} ms")

def benchmark_softmax():
    print("\n" + "=" * 60)
    print("Benchmarking Softmax")
    print("=" * 60)

    N, D = 512, 256
    x = torch.randn(N, D, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = custom_cuda_ops.softmax_forward(x)
        torch.cuda.synchronize()

    # Benchmark
    import time
    iters = 1000
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = custom_cuda_ops.softmax_forward(x)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000 / iters

    print(f"Shape: [{N}, {D}]")
    print(f"Average time: {elapsed:.3f} ms")

if __name__ == "__main__":
    print("Testing Custom CUDA Operations")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Correctness tests
    results = []
    results.append(test_flash_attention())
    results.append(test_layernorm())
    results.append(test_softmax())

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests PASSED!")

        # Run benchmarks
        benchmark_flash_attention()
        benchmark_layernorm()
        benchmark_softmax()
    else:
        print("\n✗ Some tests FAILED")
        sys.exit(1)
