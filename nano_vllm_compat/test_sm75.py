import os
import sys
import time

# Add nano-vllm to path
sys.path.insert(0, '/root/cuda-lab/nano-vllm')

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = "/root/cuda-lab/models/Qwen3-0.6B"
    print(f"Loading tokenizer from {path}...")
    tokenizer = AutoTokenizer.from_pretrained(path)

    print(f"Loading model...")
    t0 = time.time()
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    print(f"Model loaded in {time.time()-t0:.2f}s")

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    prompts = [
        "Hello, introduce yourself in one sentence.",
        "What is 2+2?",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    print("\n--- Generating ---")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print(f"Generated in {time.time()-t0:.2f}s")

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    print("\n✓ nano-vllm sm_75 test passed!")


if __name__ == "__main__":
    main()
