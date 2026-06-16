#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Flash Attention (simplified, single-head)
// O = softmax(Q @ K^T / sqrt(d)) @ V
// Each block handles one row of Q (one query)
// Online softmax: O(N) memory, numerically stable

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,   // [N, D]
    const float* __restrict__ K,   // [N, D]
    const float* __restrict__ V,   // [N, D]
    float* __restrict__ O,         // [N, D]
    int N, int D, float scale,
    bool causal
) {
    int q_row = blockIdx.x;
    if (q_row >= N) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ float shared_mem[];
    float* sQ = shared_mem;
    float* sK_tile = sQ + D;

    for (int d = tid; d < D; d += num_threads) {
        sQ[d] = Q[q_row * D + d];
    }
    __syncthreads();

    // Online softmax state
    float m = -FLT_MAX;
    float l = 0.0f;

    int kv_end = causal ? q_row + 1 : N;

    const int Bc = 32;
    for (int kv_start = 0; kv_start < kv_end; kv_start += Bc) {
        int tile_size = min(Bc, kv_end - kv_start);

        for (int i = tid; i < tile_size * D; i += num_threads) {
            int ki = i / D;
            int d = i % D;
            sK_tile[ki * D + d] = K[(kv_start + ki) * D + d];
        }
        __syncthreads();

        for (int ki = 0; ki < tile_size; ki++) {
            // Dot product Q . K with warp reduction
            float dot = 0.0f;
            for (int d = tid; d < D; d += num_threads) {
                dot += sQ[d] * sK_tile[ki * D + d];
            }
            // Warp reduction across 256 threads (8 warps)
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            }

            // Block reduction via shared memory
            __shared__ float warp_dots[8];
            int warp_id = tid / 32;
            int lane_id = tid % 32;
            if (lane_id == 0) warp_dots[warp_id] = dot;
            __syncthreads();

            if (warp_id == 0 && lane_id < 8) {
                dot = warp_dots[lane_id];
                #pragma unroll
                for (int offset = 4; offset > 0; offset /= 2) {
                    dot += __shfl_xor_sync(0xffffffff, dot, offset);
                }
                warp_dots[0] = dot;
            }
            __syncthreads();
            dot = warp_dots[0] * scale;

            // Online softmax update
            float new_m = fmaxf(m, dot);
            float exp_diff = expf(m - new_m);
            float p = expf(dot - new_m);

            for (int d = tid; d < D; d += num_threads) {
                // Use shared mem for O accumulation to avoid register pressure
                // Actually use registers since D <= 128 and each thread handles D/256 elements
            }

            // Need per-thread O accumulator
            // This approach won't work well with D > num_threads
            // Let's simplify: each thread handles specific D columns
            break;  // exit ki loop, use different approach below
        }
        __syncthreads();
    }
}

// Correct implementation: each thread handles D/num_threads columns of output
__global__ void flash_attention_v2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N, int D, float scale,
    bool causal
) {
    int q_row = blockIdx.x;
    if (q_row >= N) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;  // 256

    // Each thread handles columns: tid, tid+num_threads, ...
    // For D=32, each thread handles at most 1 column
    // For D=256, each thread handles exactly 1 column

    // Online softmax state (per-thread)
    float m = -FLT_MAX;
    float l = 0.0f;
    float O_acc[4];  // max 4 columns per thread (D <= 1024)
    int my_cols = 0;
    for (int d = tid; d < D; d += num_threads) {
        O_acc[my_cols++] = 0.0f;
    }

    // Load Q[q_row] into registers (each thread loads its columns)
    float q_reg[4];
    int col_idx = 0;
    for (int d = tid; d < D; d += num_threads) {
        q_reg[col_idx++] = Q[q_row * D + d];
    }

    int kv_end = causal ? q_row + 1 : N;

    for (int k_row = 0; k_row < kv_end; k_row++) {
        // Compute dot product Q[q_row] . K[k_row]
        float dot = 0.0f;
        col_idx = 0;
        for (int d = tid; d < D; d += num_threads) {
            dot += q_reg[col_idx] * K[k_row * D + d];
            col_idx++;
        }

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_xor_sync(0xffffffff, dot, offset);
        }

        // Block reduction
        __shared__ float warp_dots[8];
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        if (lane_id == 0) warp_dots[warp_id] = dot;
        __syncthreads();

        if (warp_id == 0) {
            dot = (lane_id < 8) ? warp_dots[lane_id] : 0.0f;
            #pragma unroll
            for (int offset = 4; offset > 0; offset /= 2) {
                dot += __shfl_xor_sync(0xffffffff, dot, offset);
            }
            if (lane_id == 0) warp_dots[0] = dot;
        }
        __syncthreads();
        dot = warp_dots[0] * scale;

        // Online softmax update
        float new_m = fmaxf(m, dot);
        float exp_diff = expf(m - new_m);
        float p = expf(dot - new_m);

        // Rescale previous O accumulator
        for (int i = 0; i < my_cols; i++) {
            O_acc[i] = O_acc[i] * exp_diff;
        }
        l = l * exp_diff + p;
        m = new_m;

        // Accumulate V contribution
        col_idx = 0;
        for (int d = tid; d < D; d += num_threads) {
            O_acc[col_idx] += p * V[k_row * D + d];
            col_idx++;
        }
    }

    // Normalize and write output
    float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    col_idx = 0;
    for (int d = tid; d < D; d += num_threads) {
        O[q_row * D + d] = O_acc[col_idx] * inv_l;
        col_idx++;
    }
}

void run_flash_attention(
    const float* d_Q, const float* d_K, const float* d_V, float* d_O,
    int N, int D, bool causal = true
) {
    float scale = 1.0f / sqrtf((float)D);
    int threads = 256;
    int blocks = N;

    flash_attention_v2_kernel<<<blocks, threads>>>(
        d_Q, d_K, d_V, d_O, N, D, scale, causal
    );
}
