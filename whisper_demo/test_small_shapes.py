"""
Test: Kernel performance on small shapes
"""
import torch
import time
import sys

sys.path.insert(0, '/root/cuda-lab/whisper_demo/custom_ops')

try:
    import custom_cuda_ops
    print("✓ Custom CUDA ops loaded")
except ImportError as e:
    print(f"✗ Failed to load: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("Test: Small Shape Performance")
print("=" * 70)

shapes = [
    (1, 1500, 384),  # Encoder
    (1, 3, 384),     # Decoder (3 tokens)
    (1, 1, 384),     # Decoder (1 token)
]

n_calls = 10000

for shape in shapes:
    print(f"\n--- Shape: {shape} ---")
    x = torch.randn(*shape, device='cuda', dtype=torch.float16)
    gamma = torch.ones(shape[-1], device='cuda', dtype=torch.float32)
    beta = torch.zeros(shape[-1], device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(100):
        _ = custom_cuda_ops.layernorm_3d_forward(x, gamma, beta, 1e-5)
    torch.cuda.synchronize()

    # Benchmark custom
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_calls):
        _ = custom_cuda_ops.layernorm_3d_forward(x, gamma, beta, 1e-5)
    torch.cuda.synchronize()
    elapsed_custom = time.time() - start
    avg_custom = elapsed_custom * 1000 / n_calls

    # Benchmark PyTorch
    x_fp32 = x.float()
    for _ in range(100):
        _ = torch.nn.functional.layer_norm(x_fp32, (shape[-1],), gamma, beta, 1e-5)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_calls):
        _ = torch.nn.functional.layer_norm(x_fp32, (shape[-1],), gamma, beta, 1e-5)
    torch.cuda.synchronize()
    elapsed_pytorch = time.time() - start
    avg_pytorch = elapsed_pytorch * 1000 / n_calls

    print(f"  Custom:  {avg_custom:.4f} ms ({n_calls / elapsed_custom:.0f} calls/sec)")
    print(f"  PyTorch: {avg_pytorch:.4f} ms ({n_calls / elapsed_pytorch:.0f} calls/sec)")
    print(f"  Ratio:   {avg_custom / avg_pytorch:.2f}x")

print("\n" + "=" * 70)
