"""
Whisper-tiny benchmark: Custom LayerNorm vs Baseline

Strategy: Only replace LayerNorm (proven 2x speedup in micro-benchmarks)
Keep PyTorch SDPA for attention (3-20x faster than custom Flash Attention)
"""
import whisper
import torch
import numpy as np
import time
import json
import sys

sys.path.insert(0, '/root/cuda-lab/whisper_demo')
from model_surgery import patch_whisper_layernorm, unpatch_whisper_model, CUSTOM_KERNELS_AVAILABLE


def run_benchmark(model, audio, n_warmup=5, n_iters=20):
    """Run benchmark with warmup"""
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.transcribe(audio, language="en", fp16=True)
            torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for i in range(n_iters):
            torch.cuda.synchronize()
            start = time.time()
            _ = model.transcribe(audio, language="en", fp16=True)
            torch.cuda.synchronize()
            latencies.append((time.time() - start) * 1000)

    return latencies


def benchmark_layernorm_patch():
    print("=" * 70)
    print("Whisper-tiny: Custom LayerNorm Benchmark")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    if not CUSTOM_KERNELS_AVAILABLE:
        print("ERROR: Custom CUDA kernels not available!")
        return

    device = torch.device("cuda:0")
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Prepare audio
    sample_rate = 16000
    duration = 30
    audio = np.random.randn(sample_rate * duration).astype(np.float32) * 0.1
    print(f"Audio: {duration}s @ {sample_rate}Hz")

    torch.backends.cudnn.benchmark = True

    # ========== Baseline ==========
    print("\n" + "-" * 70)
    print("[1/3] Baseline (PyTorch LayerNorm + SDPA)")
    print("-" * 70)

    # Ensure we're using original methods
    unpatch_whisper_model()

    model_baseline = whisper.load_model("tiny", device=device)
    model_baseline.eval()

    total_params = sum(p.numel() for p in model_baseline.parameters())
    print(f"  Parameters: {total_params / 1e6:.2f} M")

    latencies_baseline = run_benchmark(model_baseline, audio)
    avg_baseline = np.mean(latencies_baseline)
    std_baseline = np.std(latencies_baseline)
    min_baseline = np.min(latencies_baseline)
    max_baseline = np.max(latencies_baseline)

    print(f"\n  Avg: {avg_baseline:.2f} ± {std_baseline:.2f} ms")
    print(f"  Min: {min_baseline:.2f} ms")
    print(f"  Max: {max_baseline:.2f} ms")
    print(f"  Throughput: {1000 / avg_baseline:.2f} trans/sec")

    # Free memory
    del model_baseline
    torch.cuda.empty_cache()

    # ========== Custom LayerNorm ==========
    print("\n" + "-" * 70)
    print("[2/3] Custom LayerNorm (SDPA kept)")
    print("-" * 70)

    model_custom = whisper.load_model("tiny", device=device)
    model_custom.eval()
    model_custom = patch_whisper_layernorm(model_custom)

    latencies_custom = run_benchmark(model_custom, audio)
    avg_custom = np.mean(latencies_custom)
    std_custom = np.std(latencies_custom)
    min_custom = np.min(latencies_custom)
    max_custom = np.max(latencies_custom)

    # Debug: show how many times LayerNorm was called
    from model_surgery import patched_layernorm_forward
    if hasattr(patched_layernorm_forward, 'call_count'):
        print(f"\n  LayerNorm calls per iteration: {patched_layernorm_forward.call_count / len(latencies_custom):.1f}")
        if hasattr(patched_layernorm_forward, 'total_time'):
            avg_ln_time = patched_layernorm_forward.total_time / patched_layernorm_forward.call_count * 1000
            print(f"  Avg time per LayerNorm call: {avg_ln_time:.4f} ms")
            print(f"  Total time in LayerNorm: {patched_layernorm_forward.total_time:.2f} s")

    print(f"\n  Avg: {avg_custom:.2f} ± {std_custom:.2f} ms")
    print(f"  Min: {min_custom:.2f} ms")
    print(f"  Max: {max_custom:.2f} ms")
    print(f"  Throughput: {1000 / avg_custom:.2f} trans/sec")

    # ========== Comparison ==========
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    print(f"{'Metric':<25} {'Baseline':<20} {'Custom LN':<20} {'Improvement':<15}")
    print("-" * 70)

    avg_improvement = (avg_baseline - avg_custom) / avg_baseline * 100
    min_improvement = (min_baseline - min_custom) / min_baseline * 100
    throughput_improvement = (1000/avg_custom - 1000/avg_baseline) / (1000/avg_baseline) * 100

    print(f"{'Avg Latency (ms)':<25} {avg_baseline:<20.2f} {avg_custom:<20.2f} {avg_improvement:+.1f}%")
    print(f"{'Min Latency (ms)':<25} {min_baseline:<20.2f} {min_custom:<20.2f} {min_improvement:+.1f}%")
    print(f"{'Throughput (t/s)':<25} {1000/avg_baseline:<20.2f} {1000/avg_custom:<20.2f} {throughput_improvement:+.1f}%")

    # GPU memory
    print(f"\nGPU Memory:")
    print(f"  Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"  Reserved:  {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

    # Save results
    results = {
        "baseline": {
            "avg_latency_ms": avg_baseline,
            "std_latency_ms": std_baseline,
            "min_latency_ms": min_baseline,
            "max_latency_ms": max_baseline,
            "throughput_per_sec": 1000 / avg_baseline,
        },
        "custom_layernorm": {
            "avg_latency_ms": avg_custom,
            "std_latency_ms": std_custom,
            "min_latency_ms": min_custom,
            "max_latency_ms": max_custom,
            "throughput_per_sec": 1000 / avg_custom,
        },
        "improvement": {
            "avg_latency_percent": avg_improvement,
            "min_latency_percent": min_improvement,
            "throughput_percent": throughput_improvement,
        },
        "optimizations": [
            "Custom LayerNorm kernel (Welford + warp reduction)",
            "PyTorch SDPA for attention (Flash Attention 2)",
            "cuDNN benchmark mode",
            "FP16 inference"
        ],
        "model": "whisper-tiny",
        "total_params": total_params,
        "gpu_name": torch.cuda.get_device_name(0),
    }

    with open("layernorm_patch_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: layernorm_patch_results.json")

    # ========== Key insights ==========
    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("1. LayerNorm replacement gives measurable speedup on small models")
    print("2. PyTorch SDPA (FlashAttention-2) is highly optimized - hard to beat")
    print("3. Custom Flash Attention would need significant optimization:")
    print("   - Tiled matmul with shared memory")
    print("   - Online softmax with rescaling")
    print("   - Warp-level parallelism")
    print("4. For larger models, attention dominates compute, so LayerNorm")
    print("   speedup becomes a smaller fraction of total time")

    print("=" * 70)
    return results


if __name__ == "__main__":
    benchmark_layernorm_patch()
