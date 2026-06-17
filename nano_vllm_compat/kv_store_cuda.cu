/*
 * CUDA KV Cache Store Kernel
 *
 * Replaces Triton store_kvcache_kernel in nanovllm/layers/attention.py
 *
 * Optimizations vs Triton version:
 *   1. float4 vectorized loads/stores (4x memory coalescing)
 *   2. Grid-stride loop over tokens (handles variable batch sizes)
 *   3. Handles slot == -1 (padding) efficiently via early exit
 *
 * Layout:
 *   key/value:   [N, num_kv_heads, head_dim]  (row-major, last dim contiguous)
 *   k_cache/v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
 *                flattened to [num_slots, D] where D = num_kv_heads * head_dim
 *   slot_mapping: [N]  int32, slot = block_id * block_size + token_in_block
 *
 * Each thread handles 4 contiguous float elements via float4.
 * Grid x = tokens (grid-stride), Grid y = D/4 element groups.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// FP16 kernel: key/value are half, cache is half
__global__ void store_kvcache_fp16_kernel(
    const half* __restrict__ key,          // [N, D]
    const half* __restrict__ value,        // [N, D]
    half* __restrict__ k_cache,            // [num_slots, D]
    half* __restrict__ v_cache,            // [num_slots, D]
    const int* __restrict__ slot_mapping,  // [N]
    int N,
    int D,
    int key_stride,   // stride(0) of key in elements
    int value_stride  // stride(0) of value in elements
) {
    int D4 = D >> 2;  // D / 4
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem4_idx = blockIdx.y;

    if (elem4_idx >= D4) return;

    // Grid-stride loop over tokens
    for (int n = token_idx; n < N; n += blockDim.x * gridDim.x) {
        int slot = slot_mapping[n];
        if (slot == -1) continue;

        int src_offset = n * key_stride + elem4_idx * 4;
        int dst_offset = slot * D + elem4_idx * 4;

        // Vectorized load/store (4 halfs = 8 bytes = one 64-bit transaction)
        *reinterpret_cast<float2*>(k_cache + dst_offset) =
            *reinterpret_cast<const float2*>(key + src_offset);

        int src_v_offset = n * value_stride + elem4_idx * 4;
        *reinterpret_cast<float2*>(v_cache + dst_offset) =
            *reinterpret_cast<const float2*>(value + src_v_offset);
    }
}

// FP32 fallback kernel
__global__ void store_kvcache_fp32_kernel(
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const int* __restrict__ slot_mapping,
    int N,
    int D,
    int key_stride,
    int value_stride
) {
    int D4 = D >> 2;
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem4_idx = blockIdx.y;

    if (elem4_idx >= D4) return;

    for (int n = token_idx; n < N; n += blockDim.x * gridDim.x) {
        int slot = slot_mapping[n];
        if (slot == -1) continue;

        int src_offset = n * key_stride + elem4_idx * 4;
        int dst_offset = slot * D + elem4_idx * 4;

        *reinterpret_cast<float4*>(k_cache + dst_offset) =
            *reinterpret_cast<const float4*>(key + src_offset);

        int src_v_offset = n * value_stride + elem4_idx * 4;
        *reinterpret_cast<float4*>(v_cache + dst_offset) =
            *reinterpret_cast<const float4*>(value + src_v_offset);
    }
}

// Scalar fallback for D not divisible by 4
__global__ void store_kvcache_scalar_fp16_kernel(
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    const int* __restrict__ slot_mapping,
    int N,
    int D,
    int key_stride,
    int value_stride
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int elem_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (elem_idx >= D) return;

    for (int n = token_idx; n < N; n += blockDim.x * gridDim.x) {
        int slot = slot_mapping[n];
        if (slot == -1) continue;

        k_cache[slot * D + elem_idx] = key[n * key_stride + elem_idx];
        v_cache[slot * D + elem_idx] = value[n * value_stride + elem_idx];
    }
}

extern "C" {

void launch_store_kvcache(
    const void* key,
    const void* value,
    void* k_cache,
    void* v_cache,
    const int* slot_mapping,
    int N,
    int D,
    int key_stride,
    int value_stride,
    int is_fp16,
    cudaStream_t stream
) {
    if (D % 4 == 0) {
        int D4 = D / 4;
        dim3 block(128);
        dim3 grid((N + block.x - 1) / block.x, D4);
        // Cap grid x to avoid oversized grids for small N
        if (grid.x > 256) grid.x = 256;

        if (is_fp16) {
            store_kvcache_fp16_kernel<<<grid, block, 0, stream>>>(
                (const half*)key, (const half*)value,
                (half*)k_cache, (half*)v_cache,
                slot_mapping, N, D, key_stride, value_stride
            );
        } else {
            store_kvcache_fp32_kernel<<<grid, block, 0, stream>>>(
                (const float*)key, (const float*)value,
                (float*)k_cache, (float*)v_cache,
                slot_mapping, N, D, key_stride, value_stride
            );
        }
    } else {
        // Scalar fallback
        dim3 block(128, 4);
        dim3 grid((N + block.x - 1) / block.x, (D + block.y - 1) / block.y);
        if (grid.x > 256) grid.x = 256;

        store_kvcache_scalar_fp16_kernel<<<grid, block, 0, stream>>>(
            (const half*)key, (const half*)value,
            (half*)k_cache, (half*)v_cache,
            slot_mapping, N, D, key_stride, value_stride
        );
    }
}

} // extern "C"
