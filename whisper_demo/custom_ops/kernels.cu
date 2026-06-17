/**
 * Custom CUDA kernels for PyTorch extension
 * Flash Attention, LayerNorm, Softmax
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

// ============================================================================
// Helper: Warp reduction for sum
// ============================================================================
__device__ __forceinline__ void warp_reduce_sum(float& val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
}

__device__ __forceinline__ void warp_reduce_sum_2(float& val1, float& val2) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val1 += __shfl_xor_sync(0xffffffff, val1, offset);
        val2 += __shfl_xor_sync(0xffffffff, val2, offset);
    }
}

// ============================================================================
// Flash Attention Kernel (Online Softmax)
// ============================================================================
__global__ void flash_attention_kernel(
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
    int num_threads = blockDim.x;

    // Initialize accumulators
    float m = -FLT_MAX;
    float l = 0.0f;

    // Each thread handles multiple columns of O
    float O_acc[4];
    int my_cols = 0;
    for (int d = tid; d < D; d += num_threads) {
        O_acc[my_cols++] = 0.0f;
    }

    // Load Q row into registers
    float q_reg[4];
    int col_idx = 0;
    for (int d = tid; d < D; d += num_threads) {
        q_reg[col_idx++] = Q[q_row * D + d];
    }

    int kv_end = causal ? q_row + 1 : N;

    // Iterate over K/V rows
    for (int k_row = 0; k_row < kv_end; k_row++) {
        // Compute dot product Q[q_row] · K[k_row]
        float dot = 0.0f;
        col_idx = 0;
        for (int d = tid; d < D; d += num_threads) {
            dot += q_reg[col_idx] * K[k_row * D + d];
            col_idx++;
        }

        // Warp reduction
        warp_reduce_sum(dot);

        // Block reduction via shared memory
        __shared__ float warp_dots[8];
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        if (lane_id == 0) warp_dots[warp_id] = dot;
        __syncthreads();

        if (warp_id == 0) {
            dot = (lane_id < 8) ? warp_dots[lane_id] : 0.0f;
            warp_reduce_sum(dot);
            if (lane_id == 0) warp_dots[0] = dot;
        }
        __syncthreads();
        dot = warp_dots[0] * scale;

        // Online softmax update
        float new_m = fmaxf(m, dot);
        float exp_diff = expf(m - new_m);
        float p = expf(dot - new_m);

        // Update O accumulator
        for (int i = 0; i < my_cols; i++) {
            O_acc[i] = O_acc[i] * exp_diff;
        }
        l = l * exp_diff + p;
        m = new_m;

        // Add contribution from V
        col_idx = 0;
        for (int d = tid; d < D; d += num_threads) {
            O_acc[col_idx] += p * V[k_row * D + d];
            col_idx++;
        }
    }

    // Normalize output
    float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    col_idx = 0;
    for (int d = tid; d < D; d += num_threads) {
        O[q_row * D + d] = O_acc[col_idx] * inv_l;
        col_idx++;
    }
}

// ============================================================================
// LayerNorm Kernel (Welford's algorithm)
// ============================================================================
__global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int N, int D, float eps
) {
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    const float* x_row = x + row * D;
    float* y_row = y + row * D;

    // Compute sum and sum of squares
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid; i < D; i += blockDim.x) {
        float val = x_row[i];
        sum += val;
        sum_sq += val * val;
    }

    // Warp reduction
    warp_reduce_sum_2(sum, sum_sq);

    // Block reduction via shared memory
    __shared__ float shared_sum[8];
    __shared__ float shared_sum_sq[8];

    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
        shared_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();

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

    // Normalize
    for (int i = tid; i < D; i += blockDim.x) {
        float val = x_row[i];
        float normalized = (val - mean) * inv_std;
        y_row[i] = gamma[i] * normalized + beta[i];
    }
}

// ============================================================================
// Softmax Kernel (Online algorithm)
// ============================================================================
__global__ void softmax_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int D
) {
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    const float* x_row = x + row * D;
    float* y_row = y + row * D;

    // Online max and sum
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    for (int i = tid; i < D; i += blockDim.x) {
        float val = x_row[i];
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, val);
        thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_xor_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_xor_sync(0xffffffff, thread_sum, offset);
        float new_max = fmaxf(thread_max, other_max);
        thread_sum = thread_sum * expf(thread_max - new_max) + other_sum * expf(other_max - new_max);
        thread_max = new_max;
    }

    // Block-level reduction via shared memory
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

    // Compute softmax
    for (int i = tid; i < D; i += blockDim.x) {
        y_row[i] = expf(x_row[i] - global_max) * inv_sum;
    }
}

// ============================================================================
// C++ Wrapper Functions
// ============================================================================

torch::Tensor flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");

    int N = Q.size(0);
    int D = Q.size(1);

    TORCH_CHECK(K.size(0) == N && K.size(1) == D, "K shape mismatch");
    TORCH_CHECK(V.size(0) == N && V.size(1) == D, "V shape mismatch");

    auto O = torch::empty_like(Q);

    float scale = 1.0f / sqrtf(static_cast<float>(D));

    int threads = 256;
    int blocks = N;

    flash_attention_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N, D, scale, causal
    );

    return O;
}

torch::Tensor layernorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    int N = x.size(0);
    int D = x.size(1);

    TORCH_CHECK(gamma.size(0) == D, "gamma shape mismatch");
    TORCH_CHECK(beta.size(0) == D, "beta shape mismatch");

    auto y = torch::empty_like(x);

    int threads = 256;
    int blocks = N;

    layernorm_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        N, D, eps
    );

    return y;
}

torch::Tensor softmax_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    int N = x.size(0);
    int D = x.size(1);

    auto y = torch::empty_like(x);

    int threads = 256;
    int blocks = N;

    softmax_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, D
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_forward", &flash_attention_forward, "Flash Attention forward (CUDA)");
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm forward (CUDA)");
    m.def("softmax_forward", &softmax_forward, "Softmax forward (CUDA)");
}
