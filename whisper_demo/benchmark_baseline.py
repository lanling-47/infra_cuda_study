"""
Whisper-tiny Baseline Benchmark (PyTorch CPU)
"""
import whisper
import torch
import numpy as np
import time
import json

def benchmark_whisper_cpu():
    print("=" * 60)
    print("Whisper-tiny CPU Baseline Benchmark")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading Whisper-tiny model...")
    model = whisper.load_model("tiny")
    model.eval()

    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())

    print(f"  Total parameters: {total_params / 1e6:.2f} M")
    print(f"  Encoder: {encoder_params / 1e6:.2f} M")
    print(f"  Decoder: {decoder_params / 1e6:.2f} M")

    # Prepare synthetic audio (30 seconds, 16kHz)
    print("\n[2/4] Preparing synthetic audio input...")
    sample_rate = 16000
    duration = 30  # seconds
    audio = np.random.randn(sample_rate * duration).astype(np.float32) * 0.1
    print(f"  Audio shape: {audio.shape}")
    print(f"  Duration: {duration}s, Sample rate: {sample_rate}Hz")

    # Warmup
    print("\n[3/4] Warming up...")
    with torch.no_grad():
        _ = model.transcribe(audio, language="en", fp16=False)

    # Benchmark
    print("\n[4/4] Running benchmark (10 iterations)...")
    latencies = []
    for i in range(10):
        start = time.time()
        with torch.no_grad():
            result = model.transcribe(audio, language="en", fp16=False)
        end = time.time()
        latency = (end - start) * 1000  # ms
        latencies.append(latency)
        print(f"  Iteration {i+1:2d}: {latency:.2f} ms")

    # Statistics
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    print("\n" + "=" * 60)
    print("Benchmark Results (CPU)")
    print("=" * 60)
    print(f"Average latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
    print(f"Min latency:     {min_latency:.2f} ms")
    print(f"Max latency:     {max_latency:.2f} ms")
    print(f"Throughput:      {1000 / avg_latency:.2f} transcriptions/sec")

    # Save results
    results = {
        "model": "whisper-tiny",
        "device": "cpu",
        "total_params": total_params,
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "audio_duration": duration,
        "sample_rate": sample_rate,
        "avg_latency_ms": avg_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "throughput_per_sec": 1000 / avg_latency
    }

    with open("baseline_cpu_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: baseline_cpu_results.json")
    print("=" * 60)

    return results

if __name__ == "__main__":
    benchmark_whisper_cpu()
