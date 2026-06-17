"""
Simple test: Load model once, patch, and test
"""
import whisper
import torch
import numpy as np
import time
import sys

sys.path.insert(0, '/root/cuda-lab/whisper_demo')
from model_surgery import patch_whisper_layernorm, unpatch_whisper_model

print("=" * 70)
print("Simple Test: Single Model Load")
print("=" * 70)

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Load model once
print("\n[1/4] Loading model...")
model = whisper.load_model("tiny", device=device)
model.eval()

# Prepare audio
audio = np.random.randn(16000 * 30).astype(np.float32) * 0.1

# Test baseline
print("\n[2/4] Testing baseline...")
unpatch_whisper_model()

# Warmup
for _ in range(5):
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()

# Benchmark
latencies_baseline = []
for i in range(10):
    torch.cuda.synchronize()
    start = time.time()
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    latencies_baseline.append(elapsed)
    print(f"  Iter {i+1}: {elapsed:.2f} ms")

avg_baseline = sum(latencies_baseline) / len(latencies_baseline)
print(f"\n  Baseline avg: {avg_baseline:.2f} ms")

# Test custom
print("\n[3/4] Testing custom LayerNorm...")
patch_whisper_layernorm(model)

# Warmup
for _ in range(5):
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()

# Benchmark
latencies_custom = []
for i in range(10):
    torch.cuda.synchronize()
    start = time.time()
    _ = model.transcribe(audio, language="en", fp16=True)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    latencies_custom.append(elapsed)
    print(f"  Iter {i+1}: {elapsed:.2f} ms")

avg_custom = sum(latencies_custom) / len(latencies_custom)
print(f"\n  Custom avg: {avg_custom:.2f} ms")

# Compare
print("\n[4/4] Comparison")
print("-" * 70)
improvement = (avg_baseline - avg_custom) / avg_baseline * 100
print(f"  Baseline: {avg_baseline:.2f} ms")
print(f"  Custom:   {avg_custom:.2f} ms")
print(f"  Change:   {improvement:+.1f}%")

print("=" * 70)
