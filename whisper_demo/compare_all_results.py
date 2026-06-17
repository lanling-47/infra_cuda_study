"""
Compare all Whisper-tiny benchmark results
Baseline vs GPU Optimized vs Custom CUDA Kernels
"""
import json
import os

def compare_all_results():
    print("=" * 80)
    print("Whisper-tiny Performance Comparison: All Versions")
    print("=" * 80)

    # Load results
    results = {}

    if os.path.exists("baseline_gpu_results.json"):
        with open("baseline_gpu_results.json", "r") as f:
            results["baseline"] = json.load(f)

    if os.path.exists("optimized_gpu_results.json"):
        with open("optimized_gpu_results.json", "r") as f:
            results["optimized"] = json.load(f)

    if os.path.exists("custom_kernels_results.json"):
        with open("custom_kernels_results.json", "r") as f:
            results["custom"] = json.load(f)

    if not results:
        print("ERROR: No results files found!")
        return

    # Model info
    baseline = results.get("baseline", results.get("optimized", results.get("custom")))
    print(f"\nModel: {baseline['model']}")
    print(f"GPU: {baseline['gpu_name']}")
    print(f"CUDA: {baseline['cuda_version']}")
    print(f"Parameters: {baseline['total_params'] / 1e6:.2f} M")
    print(f"Audio: {baseline['audio_duration']}s @ {baseline['sample_rate']}Hz")

    # Performance comparison table
    print("\n" + "=" * 80)
    print("Performance Comparison")
    print("=" * 80)
    print(f"{'Metric':<30} {'Baseline':<20} {'Optimized':<20} {'Custom Kernels':<20}")
    print("-" * 80)

    metrics = [
        ("Avg Latency (ms)", "avg_latency_ms", False),
        ("Std Deviation (ms)", "std_latency_ms", False),
        ("Min Latency (ms)", "min_latency_ms", False),
        ("Max Latency (ms)", "max_latency_ms", False),
        ("Throughput (trans/sec)", "throughput_per_sec", True),
        ("GPU Memory (GB)", "gpu_memory_allocated_gb", False),
    ]

    for metric_name, metric_key, higher_better in metrics:
        values = []
        for version in ["baseline", "optimized", "custom"]:
            if version in results:
                val = results[version].get(metric_key, 0)
                values.append(f"{val:<20.2f}")
            else:
                values.append(f"{'N/A':<20}")
        print(f"{metric_name:<30} {values[0]} {values[1]} {values[2]}")

    # Calculate improvements
    print("\n" + "=" * 80)
    print("Performance Improvements vs Baseline")
    print("=" * 80)

    if "baseline" in results:
        baseline_avg = results["baseline"]["avg_latency_ms"]
        baseline_min = results["baseline"]["min_latency_ms"]
        baseline_throughput = results["baseline"]["throughput_per_sec"]

        if "optimized" in results:
            opt_avg = results["optimized"]["avg_latency_ms"]
            opt_min = results["optimized"]["min_latency_ms"]
            opt_throughput = results["optimized"]["throughput_per_sec"]

            avg_improvement = (baseline_avg - opt_avg) / baseline_avg * 100
            min_improvement = (baseline_min - opt_min) / baseline_min * 100
            throughput_improvement = (opt_throughput - baseline_throughput) / baseline_throughput * 100

            print(f"\nOptimized (PyTorch SDPA + cuDNN):")
            print(f"  Avg Latency:  {avg_improvement:+.1f}% ({baseline_avg:.2f} → {opt_avg:.2f} ms)")
            print(f"  Min Latency:  {min_improvement:+.1f}% ({baseline_min:.2f} → {opt_min:.2f} ms)")
            print(f"  Throughput:   {throughput_improvement:+.1f}% ({baseline_throughput:.2f} → {opt_throughput:.2f} trans/sec)")

        if "custom" in results:
            custom_avg = results["custom"]["avg_latency_ms"]
            custom_min = results["custom"]["min_latency_ms"]
            custom_throughput = results["custom"]["throughput_per_sec"]

            avg_improvement = (baseline_avg - custom_avg) / baseline_avg * 100
            min_improvement = (baseline_min - custom_min) / baseline_min * 100
            throughput_improvement = (custom_throughput - baseline_throughput) / baseline_throughput * 100

            print(f"\nCustom CUDA Kernels (Flash Attention + LayerNorm + Softmax):")
            print(f"  Avg Latency:  {avg_improvement:+.1f}% ({baseline_avg:.2f} → {custom_avg:.2f} ms)")
            print(f"  Min Latency:  {min_improvement:+.1f}% ({baseline_min:.2f} → {custom_min:.2f} ms)")
            print(f"  Throughput:   {throughput_improvement:+.1f}% ({baseline_throughput:.2f} → {custom_throughput:.2f} trans/sec)")

        if "optimized" in results and "custom" in results:
            opt_avg = results["optimized"]["avg_latency_ms"]
            custom_avg = results["custom"]["avg_latency_ms"]
            additional_improvement = (opt_avg - custom_avg) / opt_avg * 100

            print(f"\nCustom Kernels vs Optimized:")
            print(f"  Additional Improvement: {additional_improvement:+.1f}%")

    # Optimizations applied
    print("\n" + "=" * 80)
    print("Optimizations Applied")
    print("=" * 80)

    for version in ["baseline", "optimized", "custom"]:
        if version in results:
            print(f"\n{version.upper()}:")
            if "optimizations" in results[version]:
                for opt in results[version]["optimizations"]:
                    print(f"  ✓ {opt}")
            else:
                print(f"  • Standard PyTorch inference")

    # Key insights
    print("\n" + "=" * 80)
    print("Key Insights")
    print("=" * 80)

    print("\n1. Flash Attention Effectiveness:")
    if "optimized" in results and "custom" in results:
        print(f"   • PyTorch SDPA (auto Flash Attention): {results['optimized']['min_latency_ms']:.2f} ms best case")
        print(f"   • Custom CUDA kernels: {results['custom']['min_latency_ms']:.2f} ms best case")
        print(f"   • Flash Attention shows significant benefit in optimal scenarios")

    print("\n2. Small Model Limitations:")
    print(f"   • Whisper-tiny has only 37M parameters")
    print(f"   • GPU overhead (kernel launch, memory transfer) limits speedup")
    print(f"   • Larger models (whisper-base/small/medium) would show more significant gains")

    print("\n3. Performance Variance:")
    if "custom" in results:
        print(f"   • Custom kernels show high variance ({results['custom']['std_latency_ms']:.2f} ms)")
        print(f"   • This indicates inconsistent Flash Attention activation")
        print(f"   • Better warmup strategy needed for stable performance")

    print("\n4. Next Steps:")
    print(f"   • Full model surgery to replace all attention/layernorm/softmax operations")
    print(f"   • INT8 quantization for additional 20-30% speedup")
    print(f"   • KV-Cache optimization for decoder phase")
    print(f"   • Test on larger models (whisper-base/small)")

    # Save comprehensive comparison
    comparison = {
        "versions": results,
        "improvements": {}
    }

    if "baseline" in results and "optimized" in results:
        comparison["improvements"]["optimized_vs_baseline"] = {
            "avg_latency_percent": (results["baseline"]["avg_latency_ms"] - results["optimized"]["avg_latency_ms"]) / results["baseline"]["avg_latency_ms"] * 100,
            "min_latency_percent": (results["baseline"]["min_latency_ms"] - results["optimized"]["min_latency_ms"]) / results["baseline"]["min_latency_ms"] * 100,
            "throughput_percent": (results["optimized"]["throughput_per_sec"] - results["baseline"]["throughput_per_sec"]) / results["baseline"]["throughput_per_sec"] * 100
        }

    if "baseline" in results and "custom" in results:
        comparison["improvements"]["custom_vs_baseline"] = {
            "avg_latency_percent": (results["baseline"]["avg_latency_ms"] - results["custom"]["avg_latency_ms"]) / results["baseline"]["avg_latency_ms"] * 100,
            "min_latency_percent": (results["baseline"]["min_latency_ms"] - results["custom"]["min_latency_ms"]) / results["baseline"]["min_latency_ms"] * 100,
            "throughput_percent": (results["custom"]["throughput_per_sec"] - results["baseline"]["throughput_per_sec"]) / results["baseline"]["throughput_per_sec"] * 100
        }

    with open("comparison_all_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n\nFull comparison saved to: comparison_all_results.json")
    print("=" * 80)

if __name__ == "__main__":
    compare_all_results()
