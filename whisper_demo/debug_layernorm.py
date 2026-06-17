"""
Debug script: Check what's happening with LayerNorm replacement
"""
import torch
import time
import sys
sys.path.insert(0, '/root/cuda-lab/whisper_demo')

from whisper.model import LayerNorm

try:
    import custom_cuda_ops
    print("✓ Custom CUDA ops loaded")
except ImportError as e:
    print(f"✗ Failed to load: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("Debug: LayerNorm Replacement")
print("=" * 60)

# Create a LayerNorm layer (in FP32 like Whisper does internally)
hidden_dim = 384
ln = LayerNorm(hidden_dim).cuda()  # Keep in FP32

print(f"\nLayerNorm created:")
print(f"  weight dtype: {ln.weight.dtype}")
print(f"  bias dtype: {ln.bias.dtype}")

# Create test input (FP16 to simulate Whisper's fp16=True mode)
batch, seq = 1, 1500
x = torch.randn(batch, seq, hidden_dim, device='cuda', dtype=torch.float16)
print(f"\nInput shape: {x.shape}, dtype: {x.dtype}")

# Test original forward
print("\n--- Original Forward ---")
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y_orig = ln(x)
    torch.cuda.synchronize()
elapsed_orig = (time.time() - start) * 1000 / 100
print(f"  Time: {elapsed_orig:.4f} ms")
print(f"  Output dtype: {y_orig.dtype}")

# Test patched forward
print("\n--- Patched Forward ---")
def patched_forward(self, x):
    x_fp32 = x.float()
    weight_fp32 = self.weight.float() if self.weight.dtype != torch.float32 else self.weight
    bias_fp32 = self.bias.float() if self.bias.dtype != torch.float32 else self.bias
    out = custom_cuda_ops.layernorm_3d_forward(x_fp32, weight_fp32, bias_fp32, self.eps)
    return out.to(x.dtype)

LayerNorm.forward = patched_forward

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y_patched = ln(x)
    torch.cuda.synchronize()
elapsed_patched = (time.time() - start) * 1000 / 100
print(f"  Time: {elapsed_patched:.4f} ms")
print(f"  Output dtype: {y_patched.dtype}")

# Compare
print("\n--- Comparison ---")
print(f"  Original: {elapsed_orig:.4f} ms")
print(f"  Patched:  {elapsed_patched:.4f} ms")
print(f"  Ratio:    {elapsed_patched / elapsed_orig:.2f}x")

# Check correctness
diff = (y_orig - y_patched).abs().max().item()
print(f"  Max diff: {diff:.6e}")

# Test without FP32 conversion
print("\n--- Without FP32 Conversion ---")
def patched_forward_no_convert(self, x):
    # Try calling kernel directly with FP16 (will fail)
    try:
        out = custom_cuda_ops.layernorm_3d_forward(x, self.weight, self.bias, self.eps)
        return out
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None

LayerNorm.forward = patched_forward_no_convert
result = ln(x)
if result is not None:
    print(f"  ✓ Kernel accepted FP16 input!")
    print(f"  Output dtype: {result.dtype}")

print("\n" + "=" * 60)
