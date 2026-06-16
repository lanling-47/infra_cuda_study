#pragma once
#include <cuda_runtime.h>

// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
// Single-pass Welford algorithm for mean and variance
// Each block processes one row (one sequence position)
// Block size: 256 threads (8 warps)

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ void warp_reduce_sum_2(float& val1, float& val2) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val1 += __shfl_xor_sync(0xffffffff, val1, offset);
        val2 += __shfl_xor_sync(0xffffffff, val2, offset);
    }
}

__global__ void layernorm_kernel(
    const float* __restrict__ x,      // [N, D]
    const float* __restrict__ gamma,  // [D]
    const float* __restrict__ beta,   // [D]
    float* __restrict__ y,            // [N, D]
    int N, int D, float eps
) {
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    const float* x_row = x + row * D;
    float* y_row = y + row * D;

    // Welford's online algorithm for mean and variance
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid; i < D; i += blockDim.x) {
        float val = x_row[i];
        sum += val;
        sum_sq += val * val;
    }

    // Warp-level reduction
    warp_reduce_sum_2(sum, sum_sq);

    // Block-level reduction using shared memory
    __shared__ float shared_sum[8];
    __shared__ float shared_sum_sq[8];

    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
        shared_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (lane_id < 8) ? shared_sum[lane_id] : 0.0f;
        sum_sq = (lane_id < 8) ? shared_sum_sq[lane_id] : 0.0f;
        warp_reduce_sum_2(sum, sum_sq);

        if (lane_id == 0) {
            shared_sum[0] = sum / D;
            shared_sum_sq[0] = sum_sq / D;
        }
    }
    __syncthreads();

    float mean = shared_sum[0];
    float var = shared_sum_sq[0] - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Normalize and apply affine transform
    for (int i = tid; i < D; i += blockDim.x) {
        float val = x_row[i];
        float normalized = (val - mean) * inv_std;
        y_row[i] = gamma[i] * normalized + beta[i];
    }
}

void run_layernorm(
    const float* d_x, const float* d_gamma, const float* d_beta,
    float* d_y, int N, int D, float eps = 1e-5f
) {
    int threads = 256;
    int blocks = N;
    layernorm_kernel<<<blocks, threads>>>(d_x, d_gamma, d_beta, d_y, N, D, eps);
}
