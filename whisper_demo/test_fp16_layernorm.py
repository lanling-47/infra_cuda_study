"""
Test FP16 LayerNorm kernel in isolation
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

print("\n" + "=" * 60)
print("Test: FP16 LayerNorm Kernel")
print("=" * 60)

# Test shapes matching Whisper
batch, seq, hidden = 1, 1500, 384
x_fp16 = torch.randn(batch, seq, hidden, device='cuda', dtype=torch.float16)
x_fp32 = x_fp16.float()
gamma = torch.ones(hidden, device='cuda', dtype=torch.float32)
beta = torch.zeros(hidden, device='cuda', dtype=torch.float32)

print(f"\nInput shape: {x_fp16.shape}")
print(f"Input dtype: {x_fp16.dtype}")
print(f"Gamma dtype: {gamma.dtype}")

# Test FP32 kernel
print("\n--- FP32 Kernel ---")
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y_fp32 = custom_cuda_ops.layernorm_3d_forward(x_fp32, gamma, beta, 1e-5)
    torch.cuda.synchronize()
elapsed_fp32 = (time.time() - start) * 1000 / 100
print(f"  Time: {elapsed_fp32:.4f} ms")
print(f"  Output dtype: {y_fp32.dtype}")

# Test FP16 kernel
print("\n--- FP16 Kernel ---")
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y_fp16 = custom_cuda_ops.layernorm_3d_forward(x_fp16, gamma, beta, 1e-5)
    torch.cuda.synchronize()
elapsed_fp16 = (time.time() - start) * 1000 / 100
print(f"  Time: {elapsed_fp16:.4f} ms")
print(f"  Output dtype: {y_fp16.dtype}")

# Test PyTorch reference
print("\n--- PyTorch Reference ---")
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y_ref = torch.nn.functional.layer_norm(x_fp32, (hidden,), gamma, beta, 1e-5)
    torch.cuda.synchronize()
elapsed_ref = (time.time() - start) * 1000 / 100
print(f"  Time: {elapsed_ref:.4f} ms")

# Compare results
print("\n--- Correctness ---")
y_fp16_as_fp32 = y_fp16.float()
diff_fp32 = (y_fp32 - y_ref).abs().max().item()
diff_fp16 = (y_fp16_as_fp32 - y_ref).abs().max().item()
print(f"  FP32 kernel vs PyTorch: {diff_fp32:.6e}")
print(f"  FP16 kernel vs PyTorch: {diff_fp16:.6e}")

# Performance comparison
print("\n--- Performance ---")
print(f"  FP32 kernel: {elapsed_fp32:.4f} ms")
print(f"  FP16 kernel: {elapsed_fp16:.4f} ms")
print(f"  PyTorch ref: {elapsed_ref:.4f} ms")
print(f"  FP16 vs PyTorch: {elapsed_fp16 / elapsed_ref:.2f}x")

print("\n" + "=" * 60)
