"""
Test: Call LayerNorm kernel many times in a loop
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
print("Test: Repeated LayerNorm Calls")
print("=" * 70)

# Create test tensors
batch, seq, hidden = 1, 1500, 384
x_fp16 = torch.randn(batch, seq, hidden, device='cuda', dtype=torch.float16)
gamma = torch.ones(hidden, device='cuda', dtype=torch.float32)
beta = torch.zeros(hidden, device='cuda', dtype=torch.float32)

n_calls = 10000

print(f"\nCalling LayerNorm {n_calls} times...")
print(f"Input shape: {x_fp16.shape}")

# Warmup
for _ in range(100):
    _ = custom_cuda_ops.layernorm_3d_forward(x_fp16, gamma, beta, 1e-5)
torch.cuda.synchronize()

# Benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(n_calls):
    _ = custom_cuda_ops.layernorm_3d_forward(x_fp16, gamma, beta, 1e-5)
torch.cuda.synchronize()
elapsed = time.time() - start

avg_time = elapsed * 1000 / n_calls
print(f"\nTotal time: {elapsed:.2f} s")
print(f"Avg time per call: {avg_time:.4f} ms")
print(f"Throughput: {n_calls / elapsed:.0f} calls/sec")

# Compare with PyTorch
print(f"\n--- PyTorch Reference ---")
x_fp32 = x_fp16.float()

# Warmup
for _ in range(100):
    _ = torch.nn.functional.layer_norm(x_fp32, (hidden,), gamma, beta, 1e-5)
torch.cuda.synchronize()

# Benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(n_calls):
    _ = torch.nn.functional.layer_norm(x_fp32, (hidden,), gamma, beta, 1e-5)
torch.cuda.synchronize()
elapsed_pt = time.time() - start

avg_time_pt = elapsed_pt * 1000 / n_calls
print(f"Total time: {elapsed_pt:.2f} s")
print(f"Avg time per call: {avg_time_pt:.4f} ms")
print(f"Throughput: {n_calls / elapsed_pt:.0f} calls/sec")

print(f"\n--- Comparison ---")
print(f"Custom kernel: {avg_time:.4f} ms")
print(f"PyTorch ref:   {avg_time_pt:.4f} ms")
print(f"Ratio: {avg_time / avg_time_pt:.2f}x")

print("=" * 70)
