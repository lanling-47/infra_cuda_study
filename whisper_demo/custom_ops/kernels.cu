/**
 * Custom CUDA kernels for PyTorch extension (V2 - Batched)
 * Supports Whisper's actual tensor shapes:
 *   - Flash Attention: [B, H, N, D]
 *   - LayerNorm: [B, N, D]  (Whisper uses [batch, seq, hidden])
 *   - Softmax: [B, H, N, N]  (attention weights)
 *
 * Both FP32 and FP16 variants
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>

// ============================================================================
// Helper: Warp reductions
// ============================================================================
__device__ __forceinline__ void warp_reduce_sum(float& val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
}

__device__ __forceinline__ void warp_reduce_max(float& val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
}

__device__ __forceinline__ void warp_reduce_sum_2(float& v1, float& v2) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v1 += __shfl_xor_sync(0xffffffff, v1, offset);
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    }
}

// ============================================================================
// Batched Flash Attention Kernel
// Q, K, V: [B, H, N, D]
// O: [B, H, N, D]
// Each block handles one (b, h, q_row)
// ============================================================================
__global__ void flash_attention_batched_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D,
    float scale,
    bool causal
) {
    int idx = blockIdx.x;
    int total_heads_rows = B * H * N;
    if (idx >= total_heads_rows) return;

    int q_row = idx % N;
    int bh = idx / N;
    int b = bh / H;
    int h = bh % H;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Base pointers for this (b, h) slice
    int bh_offset = (b * H + h) * N * D;
    const float* Q_bh = Q + bh_offset;
    const float* K_bh = K + bh_offset;
    const float* V_bh = V + bh_offset;
    float* O_bh = O + bh_offset;

    // Online softmax accumulators
    float m = -FLT_MAX;
    float l = 0.0f;

    // Each thread handles multiple columns of O_acc
    // Max D columns per thread: D / num_threads + 1
    float O_acc[8];
    int my_cols = 0;
    for (int d = tid; d < D; d += num_threads) {
        O_acc[my_cols++] = 0.0f;
    }

    // Load Q[q_row] into registers
    float q_reg[8];
    int col_idx = 0;
    for (int d = tid; d < D; d += num_threads) {
        q_reg[col_idx++] = Q_bh[q_row * D + d];
    }

    int kv_end = causal ? q_row + 1 : N;

    for (int k_row = 0; k_row < kv_end; k_row++) {
        // Compute Q[q_row] · K[k_row]
        float dot = 0.0f;
        col_idx = 0;
        for (int d = tid; d < D; d += num_threads) {
            dot += q_reg[col_idx] * K_bh[k_row * D + d];
            col_idx++;
        }

        // Warp reduction
        warp_reduce_sum(dot);

        // Block reduction via shared memory
        __shared__ float warp_sums[8];
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        int num_warps = num_threads / 32;
        if (lane_id == 0) warp_sums[warp_id] = dot;
        __syncthreads();

        if (warp_id == 0) {
            dot = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
            warp_reduce_sum(dot);
            if (lane_id == 0) warp_sums[0] = dot;
        }
        __syncthreads();
        dot = warp_sums[0] * scale;

        // Online softmax update
        float new_m = fmaxf(m, dot);
        float exp_diff = expf(m - new_m);
        float p = expf(dot - new_m);

        for (int i = 0; i < my_cols; i++) {
            O_acc[i] = O_acc[i] * exp_diff;
        }
        l = l * exp_diff + p;
        m = new_m;

        // Accumulate V[k_row] * p
        col_idx = 0;
        for (int d = tid; d < D; d += num_threads) {
            O_acc[col_idx] += p * V_bh[k_row * D + d];
            col_idx++;
        }
    }

    // Normalize
    float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    col_idx = 0;
    for (int d = tid; d < D; d += num_threads) {
        O_bh[q_row * D + d] = O_acc[col_idx] * inv_l;
        col_idx++;
    }
}

// ============================================================================
// 3D LayerNorm Kernel [B, N, D]
// Each block handles one (b, n) row
// ============================================================================
__global__ void layernorm_3d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    int BN, int D, float eps
) {
    int row = blockIdx.x;
    if (row >= BN) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    const float* x_row = x + row * D;
    float* y_row = y + row * D;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid; i < D; i += blockDim.x) {
        float val = x_row[i];
        sum += val;
        sum_sq += val * val;
    }

    warp_reduce_sum_2(sum, sum_sq);

    __shared__ float shared_sum[8];
    __shared__ float shared_sum_sq[8];

    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
        shared_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
        sum_sq = (lane_id < num_warps) ? shared_sum_sq[lane_id] : 0.0f;
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

    for (int i = tid; i < D; i += blockDim.x) {
        float val = x_row[i];
        float normalized = (val - mean) * inv_std;
        y_row[i] = gamma[i] * normalized + beta[i];
    }
}

// ============================================================================
// 3D LayerNorm Kernel FP16 [B, N, D]
// Input/output in FP16, compute in FP32 for numerical stability
// ============================================================================
__global__ void layernorm_3d_fp16_kernel(
    const half* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    half* __restrict__ y,
    int BN, int D, float eps
) {
    int row = blockIdx.x;
    if (row >= BN) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    const half* x_row = x + row * D;
    half* y_row = y + row * D;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid; i < D; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        sum += val;
        sum_sq += val * val;
    }

    warp_reduce_sum_2(sum, sum_sq);

    __shared__ float shared_sum[8];
    __shared__ float shared_sum_sq[8];

    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
        shared_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;
        sum_sq = (lane_id < num_warps) ? shared_sum_sq[lane_id] : 0.0f;
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

    for (int i = tid; i < D; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        float normalized = (val - mean) * inv_std;
        float result = gamma[i] * normalized + beta[i];
        y_row[i] = __float2half(result);
    }
}

// ============================================================================
// 3D Softmax Kernel [B, H, N, M] over last dim
// Each block handles one (b, h, n) row
// ============================================================================
__global__ void softmax_3d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int BHN, int M
) {
    int row = blockIdx.x;
    if (row >= BHN) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    const float* x_row = x + row * M;
    float* y_row = y + row * M;

    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    for (int i = tid; i < M; i += blockDim.x) {
        float val = x_row[i];
        float old_max = thread_max;
        thread_max = fmaxf(thread_max, val);
        thread_sum = thread_sum * expf(old_max - thread_max) + expf(val - thread_max);
    }

    // Warp reduction with online max/sum merge
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_xor_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_xor_sync(0xffffffff, thread_sum, offset);
        float new_max = fmaxf(thread_max, other_max);
        thread_sum = thread_sum * expf(thread_max - new_max) + other_sum * expf(other_max - new_max);
        thread_max = new_max;
    }

    __shared__ float shared_max[8];
    __shared__ float shared_sum[8];

    if (lane_id == 0) {
        shared_max[warp_id] = thread_max;
        shared_sum[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        thread_max = (lane_id < num_warps) ? shared_max[lane_id] : -INFINITY;
        thread_sum = (lane_id < num_warps) ? shared_sum[lane_id] : 0.0f;

        // Reduce across warps
        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            float other_max = __shfl_xor_sync(0xffffffff, thread_max, offset);
            float other_sum = __shfl_xor_sync(0xffffffff, thread_sum, offset);

            // Handle -INFINITY case to avoid NaN from -inf - (-inf)
            if (thread_max != -INFINITY || other_max != -INFINITY) {
                float new_max = fmaxf(thread_max, other_max);
                thread_sum = thread_sum * expf(thread_max - new_max) + other_sum * expf(other_max - new_max);
                thread_max = new_max;
            }
        }

        if (lane_id == 0) {
            shared_max[0] = thread_max;
            shared_sum[0] = thread_sum;
        }
    }
    __syncthreads();

    float global_max = shared_max[0];
    float inv_sum = 1.0f / shared_sum[0];

    for (int i = tid; i < M; i += blockDim.x) {
        y_row[i] = expf(x_row[i] - global_max) * inv_sum;
    }
}

// ============================================================================
// C++ Wrapper Functions
// ============================================================================

torch::Tensor flash_attention_batched_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale,
    bool causal
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, N, D]");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    auto O = torch::empty_like(Q);

    int total = B * H * N;
    int threads = min(256, max(32, D));
    int blocks = total;

    flash_attention_batched_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, H, N, D, scale, causal
    );

    return O;
}

torch::Tensor layernorm_3d_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [B, N, D]");

    int B = x.size(0);
    int N = x.size(1);
    int D = x.size(2);
    int BN = B * N;

    int threads = min(256, max(64, D));
    int blocks = BN;

    // Dispatch based on input dtype
    if (x.dtype() == torch::kFloat32) {
        TORCH_CHECK(gamma.dtype() == torch::kFloat32, "gamma must be float32");
        TORCH_CHECK(beta.dtype() == torch::kFloat32, "beta must be float32");

        auto y = torch::empty_like(x);
        layernorm_3d_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            y.data_ptr<float>(),
            BN, D, eps
        );
        return y;
    } else if (x.dtype() == torch::kFloat16) {
        // FP16 input, FP32 parameters (compute in FP32, output FP16)
        auto gamma_fp32 = gamma.dtype() == torch::kFloat32 ? gamma : gamma.to(torch::kFloat32);
        auto beta_fp32 = beta.dtype() == torch::kFloat32 ? beta : beta.to(torch::kFloat32);

        auto y = torch::empty_like(x);
        layernorm_3d_fp16_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            gamma_fp32.data_ptr<float>(),
            beta_fp32.data_ptr<float>(),
            reinterpret_cast<half*>(y.data_ptr<at::Half>()),
            BN, D, eps
        );
        return y;
    } else {
        TORCH_CHECK(false, "Unsupported dtype: must be float32 or float16");
        return x; // unreachable
    }
}

torch::Tensor softmax_3d_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, H, N, M]");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    int B = x.size(0);
    int H = x.size(1);
    int N = x.size(2);
    int M = x.size(3);
    int BHN = B * H * N;

    auto y = torch::empty_like(x);

    int threads = min(256, max(64, M));
    int blocks = BHN;

    softmax_3d_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        BHN, M
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_batched_forward", &flash_attention_batched_forward,
          "Batched Flash Attention [B, H, N, D] (CUDA)");
    m.def("layernorm_3d_forward", &layernorm_3d_forward,
          "3D LayerNorm [B, N, D] (CUDA)");
    m.def("softmax_3d_forward", &softmax_3d_forward,
          "4D Softmax [B, H, N, M] (CUDA)");

    // Keep original APIs for backward compat
    m.def("flash_attention_forward",
          [](torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool causal) -> torch::Tensor {
              // Reshape [N, D] -> [1, 1, N, D]
              auto Q4 = Q.unsqueeze(0).unsqueeze(0);
              auto K4 = K.unsqueeze(0).unsqueeze(0);
              auto V4 = V.unsqueeze(0).unsqueeze(0);
              int D = Q.size(-1);
              float scale = 1.0f / sqrtf(static_cast<float>(D));
              auto O4 = flash_attention_batched_forward(Q4, K4, V4, scale, causal);
              return O4.squeeze(0).squeeze(0);
          },
          "Flash Attention [N, D] (CUDA)");
    m.def("layernorm_forward",
          [](torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps) -> torch::Tensor {
              // Handle both 2D [N, D] and 3D [B, N, D]
              if (x.dim() == 2) {
                  auto x3 = x.unsqueeze(0);
                  auto y3 = layernorm_3d_forward(x3, gamma, beta, eps);
                  return y3.squeeze(0);
              } else {
                  return layernorm_3d_forward(x, gamma, beta, eps);
              }
          },
          "LayerNorm [N, D] or [B, N, D] (CUDA)");
    m.def("softmax_forward",
          [](torch::Tensor x) -> torch::Tensor {
              // Handle both 2D [N, D] and 4D [B, H, N, M]
              if (x.dim() == 2) {
                  auto x4 = x.unsqueeze(0).unsqueeze(0);
                  auto y4 = softmax_3d_forward(x4);
                  return y4.squeeze(0).squeeze(0);
              } else if (x.dim() == 4) {
                  return softmax_3d_forward(x);
              } else {
                  TORCH_CHECK(false, "softmax_forward expects 2D or 4D tensor");
              }
          },
          "Softmax [N, D] or [B, H, N, M] (CUDA)");
}
