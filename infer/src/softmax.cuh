#pragma once
#include <cuda_runtime.h>
#include <math.h>

// Softmax: y = exp(x - max) / sum(exp(x - max))
// Online softmax algorithm: single pass, numerically stable
// Each block processes one row

__global__ void softmax_kernel(
    const float* __restrict__ x,  // [N, D]
    float* __restrict__ y,        // [N, D]
    int N, int D
) {
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    const float* x_row = x + row * D;
    float* y_row = y + row * D;

    // Online softmax: maintain running max and sum
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    // First pass: find max and compute sum of exp(x - max)
    for (int i = tid; i < D; i += blockDim.x) {
        float val = x_row[i];
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, val);
        // Rescale previous sum when max changes
        thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
    }

    // Warp-level reduction for max and sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_xor_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_xor_sync(0xffffffff, thread_sum, offset);
        float new_max = fmaxf(thread_max, other_max);
        thread_sum = thread_sum * expf(thread_max - new_max) + other_sum * expf(other_max - new_max);
        thread_max = new_max;
    }

    // Block-level reduction
    __shared__ float shared_max[8];
    __shared__ float shared_sum[8];

    if (lane_id == 0) {
        shared_max[warp_id] = thread_max;
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        thread_max = (lane_id < 8) ? shared_max[lane_id] : -INFINITY;
        thread_sum = (lane_id < 8) ? shared_sum[lane_id] : 0.0f;

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            float other_max = __shfl_xor_sync(0xffffffff, thread_max, offset);
            float other_sum = __shfl_xor_sync(0xffffffff, thread_sum, offset);
            float new_max = fmaxf(thread_max, other_max);
            thread_sum = thread_sum * expf(thread_max - new_max) + other_sum * expf(other_max - new_max);
            thread_max = new_max;
        }

        if (lane_id == 0) {
            shared_max[0] = thread_max;
            shared_sum[0] = thread_sum;
        }
    }
    __syncthreads();

    float global_max = shared_max[0];
    float global_sum = shared_sum[0];
    float inv_sum = 1.0f / global_sum;

    // Second pass: normalize
    for (int i = tid; i < D; i += blockDim.x) {
        y_row[i] = expf(x_row[i] - global_max) * inv_sum;
    }
}

void run_softmax(const float* d_x, float* d_y, int N, int D) {
    int threads = 256;
    int blocks = N;
    softmax_kernel<<<blocks, threads>>>(d_x, d_y, N, D);
}
