"""
Test custom CUDA operations (V2 - Batched)
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

def test_flash_attention_batched():
    print("\n" + "=" * 60)
    print("Testing Batched Flash Attention")
    print("=" * 60)

    B, H, N, D = 2, 4, 64, 64
    Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

    scale = 1.0 / (D ** 0.5)

    # Custom kernel
    O_custom = custom_cuda_ops.flash_attention_batched_forward(Q, K, V, scale, False)

    # PyTorch reference
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    O_ref = torch.matmul(attn, V)

    # Compare
    max_diff = (O_custom - O_ref).abs().max().item()
    mean_diff = (O_custom - O_ref).abs().mean().item()

    print(f"Shape: [{B}, {H}, {N}, {D}]")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("✓ Batched Flash Attention PASSED")
        return True
    else:
        print("✗ Batched Flash Attention FAILED")
        return False

def test_flash_attention_causal():
    print("\n" + "=" * 60)
    print("Testing Causal Flash Attention")
    print("=" * 60)

    B, H, N, D = 1, 6, 32, 64
    Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)

    scale = 1.0 / (D ** 0.5)

    # Custom kernel (causal)
    O_custom = custom_cuda_ops.flash_attention_batched_forward(Q, K, V, scale, True)

    # PyTorch reference (causal)
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(N, N, device='cuda'), diagonal=1).bool()
    attn.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    O_ref = torch.matmul(attn, V)

    max_diff = (O_custom - O_ref).abs().max().item()
    mean_diff = (O_custom - O_ref).abs().mean().item()

    print(f"Shape: [{B}, {H}, {N}, {D}] (causal)")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("✓ Causal Flash Attention PASSED")
        return True
    else:
        print("✗ Causal Flash Attention FAILED")
        return False

def test_layernorm_3d():
    print("\n" + "=" * 60)
    print("Testing 3D LayerNorm")
    print("=" * 60)

    B, N, D = 2, 64, 384
    x = torch.randn(B, N, D, device='cuda', dtype=torch.float32)
    gamma = torch.ones(D, device='cuda', dtype=torch.float32)
    beta = torch.zeros(D, device='cuda', dtype=torch.float32)
    eps = 1e-5

    # Custom kernel
    y_custom = custom_cuda_ops.layernorm_3d_forward(x, gamma, beta, eps)

    # PyTorch reference
    y_ref = torch.nn.functional.layer_norm(x, (D,), gamma, beta, eps)

    max_diff = (y_custom - y_ref).abs().max().item()
    mean_diff = (y_custom - y_ref).abs().mean().item()

    print(f"Shape: [{B}, {N}, {D}]")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("✓ 3D LayerNorm PASSED")
        return True
    else:
        print("✗ 3D LayerNorm FAILED")
        return False

def test_softmax_3d():
    print("\n" + "=" * 60)
    print("Testing 4D Softmax")
    print("=" * 60)

    B, H, N, M = 2, 4, 64, 64
    x = torch.randn(B, H, N, M, device='cuda', dtype=torch.float32)

    # Custom kernel
    y_custom = custom_cuda_ops.softmax_3d_forward(x)

    # PyTorch reference
    y_ref = torch.softmax(x, dim=-1)

    max_diff = (y_custom - y_ref).abs().max().item()
    mean_diff = (y_custom - y_ref).abs().mean().item()

    print(f"Shape: [{B}, {H}, {N}, {M}]")
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-4:
        print("✓ 4D Softmax PASSED")
        return True
    else:
        print("✗ 4D Softmax FAILED")
        return False

def test_backward_compat():
    print("\n" + "=" * 60)
    print("Testing Backward Compatibility (2D APIs)")
    print("=" * 60)

    N, D = 64, 32
    Q = torch.randn(N, D, device='cuda', dtype=torch.float32)
    K = torch.randn(N, D, device='cuda', dtype=torch.float32)
    V = torch.randn(N, D, device='cuda', dtype=torch.float32)

    # Old API
    O = custom_cuda_ops.flash_attention_forward(Q, K, V, False)
    print(f"✓ flash_attention_forward [N, D] works: {O.shape}")

    x = torch.randn(64, 256, device='cuda', dtype=torch.float32)
    gamma = torch.ones(256, device='cuda', dtype=torch.float32)
    beta = torch.zeros(256, device='cuda', dtype=torch.float32)
    y = custom_cuda_ops.layernorm_forward(x, gamma, beta, 1e-5)
    print(f"✓ layernorm_forward [N, D] works: {y.shape}")

    z = custom_cuda_ops.softmax_forward(x)
    print(f"✓ softmax_forward [N, D] works: {z.shape}")

    return True

def benchmark_flash_attention_batched():
    print("\n" + "=" * 60)
    print("Benchmarking Batched Flash Attention")
    print("=" * 60)

    # Whisper-tiny shapes: B=1, H=6, N=variable, D=64
    configs = [
        (1, 6, 64, 64, "Encoder self-attn (short)"),
        (1, 6, 256, 64, "Encoder self-attn (medium)"),
        (1, 6, 1500, 64, "Encoder self-attn (long, ~30s audio)"),
    ]

    for B, H, N, D, desc in configs:
        Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float32)
        scale = 1.0 / (D ** 0.5)

        # Warmup
        for _ in range(5):
            _ = custom_cuda_ops.flash_attention_batched_forward(Q, K, V, scale, False)
            torch.cuda.synchronize()

        # Benchmark
        import time
        iters = 50
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = custom_cuda_ops.flash_attention_batched_forward(Q, K, V, scale, False)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000 / iters

        # PyTorch SDPA reference
        for _ in range(5):
            _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
        elapsed_sdpa = (time.time() - start) * 1000 / iters

        print(f"  {desc}: custom={elapsed:.3f}ms, SDPA={elapsed_sdpa:.3f}ms, ratio={elapsed/elapsed_sdpa:.2f}x")

def benchmark_layernorm_3d():
    print("\n" + "=" * 60)
    print("Benchmarking 3D LayerNorm")
    print("=" * 60)

    # Whisper-tiny shapes: B=1, N=variable, D=384
    configs = [
        (1, 64, 384, "Short sequence"),
        (1, 256, 384, "Medium sequence"),
        (1, 1500, 384, "Long sequence (~30s audio)"),
    ]

    for B, N, D, desc in configs:
        x = torch.randn(B, N, D, device='cuda', dtype=torch.float32)
        gamma = torch.ones(D, device='cuda', dtype=torch.float32)
        beta = torch.zeros(D, device='cuda', dtype=torch.float32)

        # Warmup
        for _ in range(10):
            _ = custom_cuda_ops.layernorm_3d_forward(x, gamma, beta, 1e-5)
            torch.cuda.synchronize()

        # Benchmark
        import time
        iters = 200
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = custom_cuda_ops.layernorm_3d_forward(x, gamma, beta, 1e-5)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000 / iters

        # PyTorch reference
        for _ in range(10):
            _ = torch.nn.functional.layer_norm(x, (D,), gamma, beta, 1e-5)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            _ = torch.nn.functional.layer_norm(x, (D,), gamma, beta, 1e-5)
        torch.cuda.synchronize()
        elapsed_pt = (time.time() - start) * 1000 / iters

        print(f"  {desc} [{B},{N},{D}]: custom={elapsed:.3f}ms, PyTorch={elapsed_pt:.3f}ms, ratio={elapsed/elapsed_pt:.2f}x")

if __name__ == "__main__":
    print("Testing Custom CUDA Operations (V2 - Batched)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Correctness tests
    results = []
    results.append(test_flash_attention_batched())
    results.append(test_flash_attention_causal())
    results.append(test_layernorm_3d())
    results.append(test_softmax_3d())
    results.append(test_backward_compat())

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests PASSED!")
        benchmark_flash_attention_batched()
        benchmark_layernorm_3d()
    else:
        print("\n✗ Some tests FAILED")
        sys.exit(1)
