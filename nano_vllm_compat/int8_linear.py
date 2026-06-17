"""
INT8 Weight-Only Quantization for nano-vllm linear layers.

Strategy:
  - At load time: quantize FP16 weights → INT8 + per-row scales (symmetric)
  - At inference: fused INT8 dequant + matmul via Triton kernel
  - Decode phase (batch=1) is memory-bandwidth-bound: 2x less weight data → ~2x speedup

Quantization formula:
  scale[r] = max(abs(W[r, :])) / 127
  W_int8[r, c] = clamp(round(W_fp16[r, c] / scale[r]), -128, 127)

Dequant + GEMM:
  Y = X @ W^T
    = X @ (W_int8 * scale)^T
    = sum_c X[..., c] * W_int8[r, c] * scale[r]    (fused in Triton)
"""
import torch
import triton
import triton.language as tl


# ── Triton fused INT8 dequant + GEMM kernel ───────────────────────────────────
# X: [M, K]  fp16   (activation, row-major)
# W: [N, K]  int8   (weight, row-major, stored as torch.int8)
# scales: [N] fp16  (per-output-row scale)
# Y: [M, N]  fp16   (output)
#
# Each program computes a (BLOCK_M, BLOCK_N) tile of Y.
# For decode (M=1), BLOCK_M=1, we have one row of X broadcast across N outputs.

@triton.jit
def int8_gemm_kernel(
    X_ptr, W_ptr, scales_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K-sized chunks
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0)

        # Load W tile: [BLOCK_N, BLOCK_K] (int8)
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=k_mask[None, :], other=0)

        # Dequantize W: int8 -> fp32 (multiply by scale later for precision)
        w_fp32 = w.to(tl.float32)

        # Accumulate
        acc += tl.dot(x, tl.trans(w_fp32))

    # Load scales for this N block: [BLOCK_N]
    scale_ptrs = scales_ptr + offs_n
    scales = tl.load(scale_ptrs, mask=offs_n < N, other=1.0)

    # Apply scales: acc[m, n] *= scales[n]
    acc = acc * scales[None, :]

    # Write output (fp16)
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.float16), mask=y_mask)


def int8_linear(x: torch.Tensor, w_int8: torch.Tensor, scales: torch.Tensor,
                bias: torch.Tensor = None) -> torch.Tensor:
    """
    Fused INT8 dequant + linear.

    Args:
        x:       [M, K] or [..., K]  fp16 activation
        w_int8:  [N, K]  int8 weight
        scales:  [N]     fp16 per-row scale
        bias:    [N]     optional fp16 bias

    Returns:
        [..., N]  fp16
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = w_int8.shape[0]
    assert w_int8.shape[1] == K
    assert scales.shape[0] == N

    # Heuristic block sizes
    BLOCK_M = min(32, triton.next_power_of_2(M))
    BLOCK_N = 64
    BLOCK_K = 64

    y = torch.empty(M, N, device=x.device, dtype=torch.float16)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    int8_gemm_kernel[grid](
        x_2d, w_int8, scales, y,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    if bias is not None:
        y = y + bias.unsqueeze(0)

    return y.reshape(*orig_shape[:-1], N)


def quantize_weight_int8(weight: torch.Tensor):
    """
    Symmetric per-row INT8 quantization.

    Args:
        weight: [N, K]  fp16 or fp32

    Returns:
        w_int8: [N, K]  torch.int8
        scales: [N]     fp16
    """
    assert weight.dim() == 2
    # Compute per-row max absolute value
    abs_max = weight.abs().amax(dim=1)  # [N]
    # Avoid division by zero
    scales = abs_max / 127.0
    scales = torch.clamp(scales, min=1e-8)

    # Quantize
    w_fp32 = weight.float()
    w_scaled = w_fp32 / scales.unsqueeze(1)
    w_int8 = w_scaled.round().clamp(-128, 127).to(torch.int8)

    return w_int8, scales.to(torch.float16)
