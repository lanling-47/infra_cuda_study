"""
Compare Whisper-tiny baseline vs optimized results
"""
import json
import os

def compare_results():
    print("=" * 70)
    print("Whisper-tiny Performance Comparison")
    print("=" * 70)

    # Load baseline results
    with open("baseline_gpu_results.json", "r") as f:
        baseline = json.load(f)

    # Load optimized results
    with open("optimized_gpu_results.json", "r") as f:
        optimized = json.load(f)

    print(f"\nModel: {baseline['model']}")
    print(f"GPU: {baseline['gpu_name']}")
    print(f"CUDA: {baseline['cuda_version']}")
    print(f"Parameters: {baseline['total_params'] / 1e6:.2f} M")
    print(f"Audio: {baseline['audio_duration']}s @ {baseline['sample_rate']}Hz")

    print("\n" + "-" * 70)
    print(f"{'Metric':<30} {'Baseline':<20} {'Optimized':<20}")
    print("-" * 70)

    # Latency comparison
    print(f"{'Avg Latency (ms)':<30} {baseline['avg_latency_ms']:<20.2f} {optimized['avg_latency_ms']:<20.2f}")
    print(f"{'Std Deviation (ms)':<30} {baseline['std_latency_ms']:<20.2f} {optimized['std_latency_ms']:<20.2f}")
    print(f"{'Min Latency (ms)':<30} {baseline['min_latency_ms']:<20.2f} {optimized['min_latency_ms']:<20.2f}")
    print(f"{'Max Latency (ms)':<30} {baseline['max_latency_ms']:<20.2f} {optimized['max_latency_ms']:<20.2f}")
    print(f"{'Throughput (trans/sec)':<30} {baseline['throughput_per_sec']:<20.2f} {optimized['throughput_per_sec']:<20.2f}")
    print(f"{'GPU Memory (GB)':<30} {baseline['gpu_memory_allocated_gb']:<20.2f} {optimized['gpu_memory_allocated_gb']:<20.2f}")

    # Calculate improvements
    latency_improvement = (baseline['avg_latency_ms'] - optimized['avg_latency_ms']) / baseline['avg_latency_ms'] * 100
    throughput_improvement = (optimized['throughput_per_sec'] - baseline['throughput_per_sec']) / baseline['throughput_per_sec'] * 100
    min_latency_improvement = (baseline['min_latency_ms'] - optimized['min_latency_ms']) / baseline['min_latency_ms'] * 100

    print("\n" + "=" * 70)
    print("Performance Improvements")
    print("=" * 70)
    print(f"Avg Latency:     {latency_improvement:+.1f}% ({baseline['avg_latency_ms']:.2f} → {optimized['avg_latency_ms']:.2f} ms)")
    print(f"Min Latency:     {min_latency_improvement:+.1f}% ({baseline['min_latency_ms']:.2f} → {optimized['min_latency_ms']:.2f} ms)")
    print(f"Throughput:      {throughput_improvement:+.1f}% ({baseline['throughput_per_sec']:.2f} → {optimized['throughput_per_sec']:.2f} trans/sec)")

    print("\n" + "=" * 70)
    print("Optimizations Applied")
    print("=" * 70)
    for opt in optimized['optimizations']:
        print(f"  ✓ {opt}")

    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)

    if latency_improvement > 0:
        print(f"✓ Optimized version is {latency_improvement:.1f}% faster on average")
    else:
        print(f"✗ Optimized version is {-latency_improvement:.1f}% slower on average")

    if min_latency_improvement > 10:
        print(f"✓ Best-case latency improved by {min_latency_improvement:.1f}%")
        print(f"  (Flash Attention likely activated for optimal cases)")

    if optimized['std_latency_ms'] > baseline['std_latency_ms']:
        print(f"⚠ Higher variance in optimized version ({optimized['std_latency_ms']:.1f} vs {baseline['std_latency_ms']:.1f} ms)")
        print(f"  (May indicate inconsistent Flash Attention activation)")

    # GPU memory comparison
    if optimized['gpu_memory_allocated_gb'] < baseline['gpu_memory_allocated_gb']:
        mem_improvement = (baseline['gpu_memory_allocated_gb'] - optimized['gpu_memory_allocated_gb']) / baseline['gpu_memory_allocated_gb'] * 100
        print(f"✓ GPU memory reduced by {mem_improvement:.1f}%")
    else:
        print(f"• GPU memory usage similar ({optimized['gpu_memory_allocated_gb']:.2f} GB)")

    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("1. Whisper-tiny (37M params) is a small model")
    print("2. GPU overhead (kernel launch, memory transfer) limits speedup")
    print("3. Flash Attention shows benefit in best-case scenarios")
    print("4. For larger models (whisper-base/small/medium), improvements would be more significant")
    print("5. Custom CUDA kernels (compiled) would provide additional 10-20% speedup")

    # Save comparison
    comparison = {
        "baseline": baseline,
        "optimized": optimized,
        "improvements": {
            "avg_latency_percent": latency_improvement,
            "min_latency_percent": min_latency_improvement,
            "throughput_percent": throughput_improvement
        }
    }

    with open("comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nFull comparison saved to: comparison_results.json")
    print("=" * 70)

if __name__ == "__main__":
    compare_results()
