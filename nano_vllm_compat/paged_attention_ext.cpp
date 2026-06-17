/*
 * PyBind11 wrapper for Paged Attention CUDA kernel
 *
 * Exposes: paged_attention_forward(q, k_cache, v_cache, block_table,
 *                                  cache_seqlens, scale, block_size)
 * Returns: output [B, num_heads, D]
 *
 * Workspace (partial_out, partial_lse) is allocated internally and cached.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" void launch_paged_attention(
    const half* q,
    const half* k_cache,
    const half* v_cache,
    const int*  block_table,
    const int*  cache_seqlens,
    half* output,
    half* partial_out,
    float* partial_lse,
    float scale,
    int B,
    int num_heads,
    int num_kv_heads,
    int D,
    int block_size,
    int max_blocks_table,
    cudaStream_t stream
);

// Cached workspace tensors (avoid re-allocation every call)
static torch::Tensor g_partial_out;
static torch::Tensor g_partial_lse;
static int64_t g_workspace_key = -1;  // encodes (B, H, max_blocks, D)

static void ensure_workspace(int B, int num_heads, int max_blocks, int D,
                              torch::Device device) {
    int64_t key = ((int64_t)B << 48) | ((int64_t)num_heads << 32)
                | ((int64_t)max_blocks << 16) | (int64_t)D;
    if (key == g_workspace_key) return;

    auto opts_fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(device);
    auto opts_fp32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    g_partial_out = torch::empty({B, num_heads, max_blocks, D}, opts_fp16);
    g_partial_lse = torch::empty({B, num_heads, max_blocks},    opts_fp32);
    g_workspace_key = key;
}

torch::Tensor paged_attention_forward(
    torch::Tensor q,              // [B, num_heads, D]  fp16
    torch::Tensor k_cache,        // [num_blocks, block_size, num_kv_heads, D]
    torch::Tensor v_cache,        // [num_blocks, block_size, num_kv_heads, D]
    torch::Tensor block_table,    // [B, max_blocks]  int32
    torch::Tensor cache_seqlens,  // [B]  int32
    float scale,
    int block_size
) {
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Half, "q must be fp16");
    TORCH_CHECK(k_cache.scalar_type() == at::ScalarType::Half, "k_cache must be fp16");
    TORCH_CHECK(block_table.scalar_type() == at::ScalarType::Int, "block_table must be int32");
    TORCH_CHECK(cache_seqlens.scalar_type() == at::ScalarType::Int, "cache_seqlens must be int32");
    TORCH_CHECK(q.dim() == 3, "q must be 3D [B, H, D]");
    TORCH_CHECK(k_cache.dim() == 4, "k_cache must be 4D");

    int B          = q.size(0);
    int num_heads  = q.size(1);
    int D          = q.size(2);
    int num_kv_heads = k_cache.size(2);
    int max_blocks = block_table.size(1);

    auto output = torch::empty({B, num_heads, D}, q.options());

    ensure_workspace(B, num_heads, max_blocks, D, q.device());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_paged_attention(
        (const half*)q.data_ptr(),
        (const half*)k_cache.data_ptr(),
        (const half*)v_cache.data_ptr(),
        (const int*)block_table.data_ptr(),
        (const int*)cache_seqlens.data_ptr(),
        (half*)output.data_ptr(),
        (half*)g_partial_out.data_ptr(),
        (float*)g_partial_lse.data_ptr(),
        scale,
        B, num_heads, num_kv_heads, D,
        block_size, max_blocks,
        stream
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention_forward", &paged_attention_forward,
          "FlashDecoding-style paged attention (sm_75 compatible)");
}
