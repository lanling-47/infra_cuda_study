/*
 * PyBind11 wrapper for store_kvcache CUDA kernel
 *
 * Exposes: store_kvcache_cuda(key, value, k_cache, v_cache, slot_mapping)
 *
 * Compile with:
 *   python -c "from torch.utils.cpp_extension import load; \
 *     load(name='kv_store_cuda', sources=['kv_store_cuda_ext.cpp', 'kv_store_cuda.cu'], \
 *          extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_75'], verbose=True)"
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" void launch_store_kvcache(
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
);

void store_kvcache_cuda(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor slot_mapping
) {
    // Validate shapes
    TORCH_CHECK(key.dim() == 3, "key must be 3D [N, num_kv_heads, head_dim]");
    TORCH_CHECK(value.dim() == 3, "value must be 3D [N, num_kv_heads, head_dim]");
    TORCH_CHECK(key.sizes() == value.sizes(), "key and value must have same shape");
    TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D");

    int N = key.size(0);
    int num_kv_heads = key.size(1);
    int head_dim = key.size(2);
    int D = num_kv_heads * head_dim;

    TORCH_CHECK(slot_mapping.size(0) == N,
        "slot_mapping size mismatch: ", slot_mapping.size(0), " vs N=", N);

    // Contiguity checks (matching original Triton assertions)
    TORCH_CHECK(key.stride(2) == 1, "key last dim must be contiguous");
    TORCH_CHECK(value.stride(2) == 1, "value last dim must be contiguous");
    TORCH_CHECK(key.stride(1) == head_dim, "key stride(1) must equal head_dim");
    TORCH_CHECK(value.stride(1) == head_dim, "value stride(1) must equal head_dim");

    // Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
    // stride(2) must equal D so that (block, pos) within a slot are contiguous
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D");
    TORCH_CHECK(v_cache.dim() == 4, "v_cache must be 4D");
    TORCH_CHECK(k_cache.stride(2) == D,
        "k_cache stride(2) must equal D=", D, " got ", k_cache.stride(2));
    TORCH_CHECK(v_cache.stride(2) == D,
        "v_cache stride(2) must equal D=", D, " got ", v_cache.stride(2));

    int key_stride = key.stride(0);
    int value_stride = value.stride(0);

    int is_fp16 = (key.scalar_type() == at::ScalarType::Half);
    TORCH_CHECK(is_fp16 || key.scalar_type() == at::ScalarType::Float,
        "key must be fp16 or fp32, got: ", key.scalar_type());
    TORCH_CHECK(slot_mapping.scalar_type() == at::ScalarType::Int,
        "slot_mapping must be int32");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_store_kvcache(
        key.data_ptr(),
        value.data_ptr(),
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        (const int*)slot_mapping.data_ptr(),
        N,
        D,
        key_stride,
        value_stride,
        is_fp16,
        stream
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("store_kvcache_cuda", &store_kvcache_cuda,
          "CUDA KV cache store (float4 vectorized, sm_75 compatible)");
}
