"""
Test speculative decoding with nano-vllm.

Compares:
  - Standard autoregressive (target model only)
  - Speculative decoding (draft + target)

Expected: 2-3x speedup on decode latency for single request.

Usage:
    cd /root/cuda-lab
    python3 nano_vllm_compat/test_speculative.py
"""
import os
import sys
import time

sys.path.insert(0, '/root/cuda-lab/nano-vllm')
sys.path.insert(0, '/root/cuda-lab/nano_vllm_compat')

from transformers import AutoTokenizer


def test_standard():
    from nanovllm import LLM, SamplingParams

    model_path = "/root/cuda-lab/models/Qwen3-0.6B"
    print("[1/2] Loading standard model...")
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = [
        "Explain quantum computing in 3 sentences.",
        "Write a haiku about the ocean.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

    # Warmup
    _ = llm.generate(prompts, sampling_params)
    import torch
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    total_tokens = sum(len(o['token_ids']) for o in outputs)
    print(f"  Standard: {elapsed:.2f}s, {total_tokens} tokens, "
          f"{total_tokens/elapsed:.1f} tok/s")

    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return outputs, elapsed


def test_speculative():
    from speculative import SpeculativeLLM
    from nanovllm.sampling_params import SamplingParams

    target_path = "/root/cuda-lab/models/Qwen3-0.6B"
    draft_path = "/root/cuda-lab/models/Qwen3-0.6B"  # same model for demo

    print("\n[2/2] Loading speculative engine...")
    # Note: For a real speedup, target should be larger than draft.
    # Using same model here demonstrates correctness (acceptance rate ≈ 100%).
    llm = SpeculativeLLM(
        model=target_path,
        draft_model=draft_path,
        num_speculative_tokens=4,
        enforce_eager=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(target_path)
    prompts = [
        "Explain quantum computing in 3 sentences.",
        "Write a haiku about the ocean.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]

    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

    # Warmup
    _ = llm.generate(prompts, sampling_params)
    import torch
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    total_tokens = sum(len(o['token_ids']) for o in outputs)
    print(f"  Speculative: {elapsed:.2f}s, {total_tokens} tokens, "
          f"{total_tokens/elapsed:.1f} tok/s")

    llm.exit()
    return outputs, elapsed


if __name__ == "__main__":
    print("=" * 60)
    print("Speculative Decoding Test")
    print("=" * 60)

    std_outputs, std_time = test_standard()
    spec_outputs, spec_time = test_speculative()

    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)

    for i, (s, p) in enumerate(zip(std_outputs, spec_outputs)):
        match = "✓" if s['text'] == p['text'] else "✗"
        print(f"\n  Prompt {i+1}: {match}")
        print(f"    Standard:    {s['text'][:80]!r}")
        print(f"    Speculative: {p['text'][:80]!r}")

    speedup = std_time / spec_time if spec_time > 0 else 0
    print(f"\n  Standard time:    {std_time:.2f}s")
    print(f"  Speculative time: {spec_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print("\n  Note: With same-size draft/target, speedup ≈ 1x (no real gain).")
    print("  For actual speedup, use a smaller draft model (e.g., 0.6B draft + 4B target).")
