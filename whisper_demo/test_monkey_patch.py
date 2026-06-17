"""
Test: Check if monkey-patching itself causes slowdown
"""
import whisper
import torch
import numpy as np
import time
import sys

sys.path.insert(0, '/root/cuda-lab/whisper_demo')

print("=" * 70)
print("Test: Monkey-patching Overhead")
print("=" * 70)

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Load model
print("\n[1/5] Loading model...")
model = whisper.load_model("tiny", device=device)
model.eval()

audio = np.random.randn(16000 * 30).astype(np.float32) * 0.1

# Test 1: Baseline
print("\n[2/5] Testing baseline (no patch)...")
for _ in range(5):
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()

latencies = []
for i in range(5):
    torch.cuda.synchronize()
    start = time.time()
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    latencies.append(elapsed)
    print(f"  Iter {i+1}: {elapsed:.2f} ms")
avg_baseline = sum(latencies) / len(latencies)

# Test 2: Monkey-patch with PyTorch's own layer_norm
print("\n[3/5] Testing monkey-patch with PyTorch layer_norm...")
from whisper.model import LayerNorm
original_forward = LayerNorm.forward

def patched_forward_pytorch(self, x):
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), self.weight, self.bias, self.eps).type(x.dtype)

LayerNorm.forward = patched_forward_pytorch

for _ in range(5):
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()

latencies = []
for i in range(5):
    torch.cuda.synchronize()
    start = time.time()
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    latencies.append(elapsed)
    print(f"  Iter {i+1}: {elapsed:.2f} ms")
avg_pytorch_patch = sum(latencies) / len(latencies)

# Test 3: Monkey-patch with custom kernel
print("\n[4/5] Testing monkey-patch with custom kernel...")
sys.path.insert(0, '/root/cuda-lab/whisper_demo/custom_ops')
import custom_cuda_ops

call_count = 0
shapes_seen = set()

def patched_forward_custom(self, x):
    global call_count, shapes_seen
    call_count += 1
    shapes_seen.add(tuple(x.shape))

    # Only print first 10 calls
    if call_count <= 10:
        print(f"    Call {call_count}: x.shape={x.shape}, x.dtype={x.dtype}")

    return custom_cuda_ops.layernorm_3d_forward(x, self.weight, self.bias, self.eps)

LayerNorm.forward = patched_forward_custom

for _ in range(5):
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()

latencies = []
for i in range(5):
    torch.cuda.synchronize()
    start = time.time()
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    latencies.append(elapsed)
    print(f"  Iter {i+1}: {elapsed:.2f} ms")
avg_custom_patch = sum(latencies) / len(latencies)

print(f"\n  Total calls: {call_count}")
print(f"  Unique shapes: {len(shapes_seen)}")
print(f"  Shapes: {shapes_seen}")

# Test 4: Monkey-patch with identity (no computation)
print("\n[4.5/5] Testing monkey-patch with identity (no-op)...")
call_count_id = 0

def patched_forward_identity(self, x):
    global call_count_id
    call_count_id += 1
    return x  # Just return input, no computation

LayerNorm.forward = patched_forward_identity

for _ in range(5):
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()

latencies = []
for i in range(5):
    torch.cuda.synchronize()
    start = time.time()
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    latencies.append(elapsed)
    print(f"  Iter {i+1}: {elapsed:.2f} ms")
avg_identity_patch = sum(latencies) / len(latencies)

print(f"\n  Total calls: {call_count_id}")

# Test 5: Monkey-patch with clone (new tensor, no computation)
print("\n[4.6/5] Testing monkey-patch with clone (new tensor, no-op)...")
call_count_clone = 0

def patched_forward_clone(self, x):
    global call_count_clone
    call_count_clone += 1
    return x.clone()  # Return a new tensor, but no computation

LayerNorm.forward = patched_forward_clone

for _ in range(5):
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()

latencies = []
for i in range(5):
    torch.cuda.synchronize()
    start = time.time()
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    latencies.append(elapsed)
    print(f"  Iter {i+1}: {elapsed:.2f} ms")
avg_clone_patch = sum(latencies) / len(latencies)

print(f"\n  Total calls: {call_count_clone}")

# Test 6: Custom kernel with dtype conversion (like PyTorch patch)
print("\n[4.7/5] Testing custom kernel with dtype conversion...")
call_count_dtype = 0

def patched_forward_dtype(self, x):
    global call_count_dtype
    call_count_dtype += 1
    # Mimic the PyTorch patch: convert to float32, compute, convert back
    x_fp32 = x.float()
    result = custom_cuda_ops.layernorm_3d_forward(x_fp32, self.weight, self.bias, self.eps)
    return result.type(x.dtype)

LayerNorm.forward = patched_forward_dtype

for _ in range(5):
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()

latencies = []
for i in range(5):
    torch.cuda.synchronize()
    start = time.time()
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    latencies.append(elapsed)
    print(f"  Iter {i+1}: {elapsed:.2f} ms")
avg_dtype_patch = sum(latencies) / len(latencies)

print(f"\n  Total calls: {call_count_dtype}")

# Restore
LayerNorm.forward = original_forward

# Compare
print("\n[5/5] Comparison")
print("-" * 70)
print(f"  Baseline:              {avg_baseline:.2f} ms")
print(f"  PyTorch patch:         {avg_pytorch_patch:.2f} ms ({(avg_pytorch_patch - avg_baseline) / avg_baseline * 100:+.1f}%)")
print(f"  Custom kernel patch:   {avg_custom_patch:.2f} ms ({(avg_custom_patch - avg_baseline) / avg_baseline * 100:+.1f}%)")
print(f"  Identity patch:        {avg_identity_patch:.2f} ms ({(avg_identity_patch - avg_baseline) / avg_baseline * 100:+.1f}%)")
print(f"  Clone patch:           {avg_clone_patch:.2f} ms ({(avg_clone_patch - avg_baseline) / avg_baseline * 100:+.1f}%)")
print(f"  Dtype conversion:      {avg_dtype_patch:.2f} ms ({(avg_dtype_patch - avg_baseline) / avg_baseline * 100:+.1f}%)")

print("=" * 70)
