#pragma once
#include <cuda_runtime.h>
#include <math.h>

// RoPE (Rotary Position Embedding)
// Applies rotation to Q/K tensors for position encoding
// Each pair (x[2i], x[2i+1]) is rotated by angle = pos * base^(-2i/d)
// x shape: [batch, seq_len, n_heads, head_dim]
// We apply RoPE to head_dim dimensions (must be even)

__global__ void rope_kernel(
    float* __restrict__ x,        // [B, S, H, D] - in-place modification
    int B, int S, int H, int D,
    float base  // default 10000.0
) {
    // Each thread handles one (batch, seq, head) tuple
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * S * H;
    if (idx >= total) return;

    int b = idx / (S * H);
    int s = (idx / H) % S;
    int h = idx % H;

    float* x_ptr = x + (b * S * H + s * H + h) * D;

    // Apply rotation to each pair
    for (int i = 0; i < D / 2; i++) {
        float theta = s * powf(base, -2.0f * i / D);
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        float x0 = x_ptr[2 * i];
        float x1 = x_ptr[2 * i + 1];

        x_ptr[2 * i]     = x0 * cos_theta - x1 * sin_theta;
        x_ptr[2 * i + 1] = x1 * cos_theta + x0 * sin_theta;
    }
}

// Optimized version: each thread handles multiple elements
__global__ void rope_kernel_opt(
    float* __restrict__ x,        // [B*S*H, D]
    int total_rows, int D,
    int seq_len,  // to extract position from row index
    float base
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= total_rows) return;

    // Extract position (seq index) from row
    // row = b * S * H + s * H + h, position = s
    int pos = (row % (seq_len * (total_rows / (total_rows / seq_len)))) / (total_rows / (total_rows / seq_len));
    // Simplified: assume layout is [..., S, H, D], pos = (row / H) % S
    // For simplicity, pass position directly

    float* x_ptr = x + row * D;

    for (int i = tid; i < D / 2; i += blockDim.x) {
        float theta = pos * powf(base, -2.0f * i / D);
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        float x0 = x_ptr[2 * i];
        float x1 = x_ptr[2 * i + 1];

        x_ptr[2 * i]     = x0 * cos_theta - x1 * sin_theta;
        x_ptr[2 * i + 1] = x1 * cos_theta + x0 * sin_theta;
    }
}

// Simple version with position passed explicitly
__global__ void rope_simple_kernel(
    float* __restrict__ x,     // [N, D] where N = B*S*H
    const int* __restrict__ positions,  // [N] position for each row
    int N, int D, float base
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= N) return;

    int pos = positions[row];
    float* x_ptr = x + row * D;

    for (int i = tid; i < D / 2; i += blockDim.x) {
        float theta = pos * powf(base, -2.0f * i / D);
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);

        float x0 = x_ptr[2 * i];
        float x1 = x_ptr[2 * i + 1];

        x_ptr[2 * i]     = x0 * cos_theta - x1 * sin_theta;
        x_ptr[2 * i + 1] = x1 * cos_theta + x0 * sin_theta;
    }
}

void run_rope(float* d_x, const int* d_positions, int N, int D, float base = 10000.0f) {
    int threads = 256;
    int blocks = N;
    rope_simple_kernel<<<blocks, threads>>>(d_x, d_positions, N, D, base);
}
