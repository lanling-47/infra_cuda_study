"""
Test INT8 quantization end-to-end with nano-vllm.

Compares FP16 baseline vs INT8 on:
  - Output quality (greedy match)
  - Generation speed (tokens/sec)
  - Memory usage

Usage:
    cd /root/cuda-lab
    python3 nano_vllm_compat/test_int8.py
"""
import os
import sys
import time

# Add nano-vllm to path
sys.path.insert(0, '/root/cuda-lab/nano-vllm')
sys.path.insert(0, '/root/cuda-lab/nano_vllm_compat')

from transformers import AutoTokenizer


def load_model_fp16():
    from nanovllm import LLM, SamplingParams
    path = "/root/cuda-lab/models/Qwen3-0.6B"
    print("[1/3] Loading FP16 model...")
    t0 = time.time()
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    print(f"  FP16 model loaded in {time.time()-t0:.2f}s")
    return llm, SamplingParams


def load_model_int8():
    # Enable INT8 patching BEFORE importing nanovllm
    os.environ['NANOVLLM_INT8'] = '1'
    import model_runner_int8  # applies the patch

    from nanovllm import LLM, SamplingParams
    path = "/root/cuda-lab/models/Qwen3-0.6B"
    print("[2/3] Loading INT8 model...")
    t0 = time.time()
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    print(f"  INT8 model loaded in {time.time()-t0:.2f}s")
    return llm, SamplingParams


def benchmark(llm, SamplingParams, prompts, max_tokens=128, label=""):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    # Warmup
    _ = llm.generate(prompts, sampling_params)
    import torch
    torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    total_tokens = sum(len(o['token_ids']) if 'token_ids' in o else len(o['text'].split()) for o in outputs)
    throughput = total_tokens / elapsed

    print(f"  [{label}] {elapsed:.2f}s, ~{throughput:.1f} tokens/sec")
    return outputs, elapsed


def main():
    path = "/root/cuda-lab/models/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(path)

    prompts = [
        "Hello, introduce yourself in one sentence.",
        "What is 2+2? Answer in one word.",
        "Explain what a GPU does in one sentence.",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    # FP16 baseline
    llm_fp16, SamplingParams = load_model_fp16()
    print("\n  Benchmarking FP16...")
    outputs_fp16, time_fp16 = benchmark(llm_fp16, SamplingParams, prompts, label="FP16")

    # Free FP16 model memory
    del llm_fp16
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    # INT8
    llm_int8, _ = load_model_int8()
    print("\n  Benchmarking INT8...")
    outputs_int8, time_int8 = benchmark(llm_int8, SamplingParams, prompts, label="INT8")

    # Compare outputs
    print("\n[3/3] Output Comparison")
    print("-" * 60)
    for i, (o_fp16, o_int8) in enumerate(zip(outputs_fp16, outputs_int8)):
        text_fp16 = o_fp16['text']
        text_int8 = o_int8['text']
        match = "✓" if text_fp16 == text_int8 else "✗"
        print(f"  Prompt {i+1}: {match}")
        print(f"    FP16: {text_fp16[:80]!r}")
        print(f"    INT8: {text_int8[:80]!r}")

    print(f"\n  Speedup: {time_fp16/time_int8:.2f}x")
    print(f"  FP16 time: {time_fp16:.2f}s")
    print(f"  INT8 time: {time_int8:.2f}s")


if __name__ == "__main__":
    main()
