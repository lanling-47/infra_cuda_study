"""
Post-load INT8 quantization patch for nano-vllm models.

Call `quantize_model_int8(model)` after load_model() to replace all
FP16 linear weights with INT8 + per-row scales.

The forward method of each linear layer is monkey-patched to use
the fused Triton INT8 dequant+GEMM kernel.
"""
import torch
from torch import nn

from int8_linear import int8_linear, quantize_weight_int8


def _make_int8_forward(module: nn.Module):
    """Create a forward function that uses INT8 weights."""
    w_int8 = module._int8_weight
    scales = module._int8_scales
    bias = module.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return int8_linear(x, w_int8, scales, bias)

    return forward


def quantize_model_int8(model: nn.Module, verbose: bool = True) -> dict:
    """
    Quantize all linear layers in the model to INT8.

    Args:
        model: nn.Module (e.g. Qwen3ForCausalLM)
        verbose: print statistics

    Returns:
        dict with quantization statistics
    """
    from nanovllm.layers.linear import (
        LinearBase, ReplicatedLinear, ColumnParallelLinear,
        RowParallelLinear, MergedColumnParallelLinear, QKVParallelLinear,
    )

    stats = {"num_quantized": 0, "bytes_before": 0, "bytes_after": 0}

    for name, module in model.named_modules():
        if not isinstance(module, LinearBase):
            continue

        # Skip if already quantized
        if hasattr(module, "_int8_weight"):
            continue

        weight = module.weight
        if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            if verbose:
                print(f"  [int8] Skipping {name}: unsupported dtype {weight.dtype}")
            continue

        N, K = weight.shape
        bytes_before = N * K * weight.element_size()

        # Quantize
        w_int8, scales = quantize_weight_int8(weight.data)
        bytes_after = N * K * 1 + N * 2  # int8 weight + fp16 scales

        # Store quantized weights as buffers (not Parameters, to avoid optimizer issues)
        # Register on the same device
        module._int8_weight = w_int8.to(weight.device)
        module._int8_scales = scales.to(weight.device)

        # Replace forward method (bound to this specific module instance)
        import types
        module.forward = types.MethodType(_make_int8_forward(module), module)

        stats["num_quantized"] += 1
        stats["bytes_before"] += bytes_before
        stats["bytes_after"] += bytes_after

        if verbose:
            ratio = bytes_before / bytes_after if bytes_after > 0 else 0
            print(f"  [int8] {name}: {N}x{K}  "
                  f"{bytes_before//1024}KB -> {bytes_after//1024}KB  ({ratio:.2f}x)")

    if verbose:
        total_before = stats["bytes_before"] / (1024 * 1024)
        total_after = stats["bytes_after"] / (1024 * 1024)
        print(f"\n  Total: {total_before:.1f}MB -> {total_after:.1f}MB  "
              f"({stats['num_quantized']} layers quantized)")

    return stats
