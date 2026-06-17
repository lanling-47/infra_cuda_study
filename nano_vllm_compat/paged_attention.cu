/*
 * Paged Attention CUDA Kernel (sm_75 Turing compatible)
 *
 * Replaces flash_attn_with_kvcache for decode phase in nano-vllm.
 * FlashDecoding-style two-pass: parallelism across KV cache pages.
 *
 * Pass 1 (partial_attn_kernel):
 *   grid = (max_blocks_per_seq, B, H),  block = (D,)
 *   One thread block per (batch, head, kv_block).
 *   Each block computes attention over one KV page and writes:
 *     partial_out[B, H, block, D]   (unnormalized weighted V sum)
 *     partial_lse[B, H, block]      (log-sum-exp for combining)
 *
 * Pass 2 (reduce_kernel):
 *   grid = (H, B),  block = (D,)
 *   One thread block per (batch, head).
 *   Combines partial outputs using logsumexp weights.
 *
 * KV cache layout:  [num_blocks, block_size, num_kv_heads, D]  (half)
 * GQA: each Q head h uses KV head h / (num_heads / num_kv_heads)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

#define WARP_SIZE 32

// ── Block-wide sum reduction using warp shuffles + cross-warp reduce ─────────
// Requires shared memory of size (num_warps) floats.
// All D threads participate. Returns the sum in ALL threads (broadcast).
__device__ __forceinline__ float block_reduce_sum(float val, float* smem_warp) {
    // Intra-warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Lane 0 of each warp writes its sum to shared memory
    if (lane_id == 0) smem_warp[warp_id] = val;
    __syncthreads();

    // First warp reduces across warps
    int num_warps = blockDim.x / WARP_SIZE;
    float warp_val = (threadIdx.x < num_warps) ? smem_warp[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            warp_val += __shfl_xor_sync(0xffffffff, warp_val, offset);
    }
    // Broadcast result via shared memory
    if (threadIdx.x == 0) smem_warp[0] = warp_val;
    __syncthreads();
    return smem_warp[0];
}

// ── Pass 1: Partial attention per KV block ────────────────────────────────────

__global__ void paged_attn_partial_kernel(
    const half* __restrict__ q,             // [B, num_heads, D]
    const half* __restrict__ k_cache,       // [num_blocks, block_size, num_kv_heads, D]
    const half* __restrict__ v_cache,       // [num_blocks, block_size, num_kv_heads, D]
    const int*  __restrict__ block_table,   // [B, max_blocks_table]
    const int*  __restrict__ cache_seqlens, // [B]
    half*  __restrict__ partial_out,        // [B, H, max_blocks, D]
    float* __restrict__ partial_lse,        // [B, H, max_blocks]
    float scale,
    int num_heads,
    int num_kv_heads,
    int D,
    int block_size,
    int max_blocks_table,
    int max_blocks_per_seq
) {
    int kv_block_idx = blockIdx.x;
    int b            = blockIdx.y;
    int h            = blockIdx.z;
    int d            = threadIdx.x;

    // Shared memory layout:
    //   [0, block_size)         : attention scores for tokens in this block
    //   [block_size, +num_warps): warp reduction scratch
    extern __shared__ float smem[];
    float* scores     = smem;
    float* smem_warp  = smem + block_size;
    int num_warps     = blockDim.x / WARP_SIZE;

    int seq_len = cache_seqlens[b];
    int num_active_blocks = (seq_len + block_size - 1) / block_size;

    int lse_offset = b * num_heads * max_blocks_per_seq
                   + h * max_blocks_per_seq + kv_block_idx;
    int out_offset = lse_offset * D + d;

    // Out-of-range block: write sentinel and exit
    if (kv_block_idx >= num_active_blocks) {
        if (d == 0) partial_lse[lse_offset] = -FLT_MAX;
        partial_out[out_offset] = __float2half(0.0f);
        return;
    }

    int physical_block = block_table[b * max_blocks_table + kv_block_idx];
    int gqa_ratio = num_heads / num_kv_heads;
    int kv_h = h / gqa_ratio;

    int block_start   = kv_block_idx * block_size;
    int valid_tokens  = min(block_size, seq_len - block_start);

    // Load Q[d] for this thread (stays in register for all tokens)
    float q_val = __half2float(q[(b * num_heads + h) * D + d]);

    // Initialize scores to -inf (for softmax)
    if (d < block_size) scores[d] = -FLT_MAX;
    __syncthreads();

    // ── Compute attention scores via block-wide dot products ──────────────────
    // For each token t: score[t] = sum_d (Q[d] * K[t, d]) * scale
    // All D threads cooperate: thread d loads K[t, d] and contributes to sum.
    for (int t = 0; t < valid_tokens; t++) {
        int k_offset = (physical_block * block_size + t)
                       * num_kv_heads * D + kv_h * D + d;
        float prod = q_val * __half2float(k_cache[k_offset]) * scale;

        // Block-wide reduction to get score[t]
        float score = block_reduce_sum(prod, smem_warp);

        // Thread 0 writes score (all threads have the value via broadcast)
        if (d == 0) scores[t] = score;
        __syncthreads();
    }

    // ── Softmax over scores[0..valid_tokens-1] ───────────────────────────────
    // Compute max (single-thread, valid_tokens is small, typically 16)
    __shared__ float s_max, s_sum;
    if (d == 0) {
        float max_score = -FLT_MAX;
        float sum_exp   = 0.0f;
        for (int t = 0; t < valid_tokens; t++) {
            max_score = fmaxf(max_score, scores[t]);
        }
        for (int t = 0; t < valid_tokens; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum_exp  += scores[t];
        }
        s_max = max_score;
        s_sum = sum_exp;
    }
    __syncthreads();

    float max_score = s_max;
    float sum_exp   = s_sum;
    float logsumexp = (sum_exp > 0.0f) ? (max_score + logf(sum_exp)) : -FLT_MAX;

    // Normalize scores in shared memory
    if (d < valid_tokens && sum_exp > 0.0f) {
        scores[d] /= sum_exp;
    }
    __syncthreads();

    // ── Weighted V sum ────────────────────────────────────────────────────────
    // out[d] = sum_t attn_weight[t] * V[t, d]
    float out_val = 0.0f;
    for (int t = 0; t < valid_tokens; t++) {
        float attn_w = scores[t];
        int v_offset = (physical_block * block_size + t)
                       * num_kv_heads * D + kv_h * D + d;
        out_val += attn_w * __half2float(v_cache[v_offset]);
    }

    partial_out[out_offset] = __float2half(out_val);
    if (d == 0) partial_lse[lse_offset] = logsumexp;
}

// ── Pass 2: Reduction across KV blocks ────────────────────────────────────────

__global__ void paged_attn_reduce_kernel(
    const half*  __restrict__ partial_out,    // [B, H, max_blocks, D]
    const float* __restrict__ partial_lse,    // [B, H, max_blocks]
    const int*   __restrict__ cache_seqlens,  // [B]
    half*  __restrict__ output,               // [B, H, D]
    int num_heads,
    int D,
    int block_size,
    int max_blocks_per_seq
) {
    int b = blockIdx.y;
    int h = blockIdx.x;
    int d = threadIdx.x;

    int seq_len = cache_seqlens[b];
    int num_active_blocks = (seq_len + block_size - 1) / block_size;

    if (num_active_blocks == 0) {
        output[(b * num_heads + h) * D + d] = __float2half(0.0f);
        return;
    }

    int base = b * num_heads * max_blocks_per_seq + h * max_blocks_per_seq;

    // Find global max lse (single-thread loop, num_active_blocks is small)
    __shared__ float s_global_max;
    if (d == 0) {
        float global_max = -FLT_MAX;
        for (int i = 0; i < num_active_blocks; i++)
            global_max = fmaxf(global_max, partial_lse[base + i]);
        s_global_max = global_max;
    }
    __syncthreads();
    float global_max = s_global_max;

    // Accumulate: out = sum_i exp(lse_i - global_max) * partial_out_i
    //             denom = sum_i exp(lse_i - global_max)
    float acc   = 0.0f;
    float denom = 0.0f;

    for (int i = 0; i < num_active_blocks; i++) {
        float lse = partial_lse[base + i];
        if (lse <= -FLT_MAX + 1.0f) continue;
        float w = expf(lse - global_max);
        acc   += w * __half2float(partial_out[(base + i) * D + d]);
        denom += w;
    }

    float result = (denom > 0.0f) ? (acc / denom) : 0.0f;
    output[(b * num_heads + h) * D + d] = __float2half(result);
}

// ── C entry point ─────────────────────────────────────────────────────────────

extern "C" {

void launch_paged_attention(
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
) {
    int max_blocks_per_seq = max_blocks_table;

    // Pass 1: partial attention
    {
        dim3 grid(max_blocks_per_seq, B, num_heads);
        dim3 block(D);
        int num_warps  = D / WARP_SIZE;
        int smem_bytes = (block_size + num_warps) * sizeof(float);
        paged_attn_partial_kernel<<<grid, block, smem_bytes, stream>>>(
            q, k_cache, v_cache, block_table, cache_seqlens,
            partial_out, partial_lse,
            scale, num_heads, num_kv_heads, D,
            block_size, max_blocks_table, max_blocks_per_seq
        );
    }

    // Pass 2: reduction
    {
        dim3 grid(num_heads, B);
        dim3 block(D);
        paged_attn_reduce_kernel<<<grid, block, 0, stream>>>(
            partial_out, partial_lse, cache_seqlens, output,
            num_heads, D, block_size, max_blocks_per_seq
        );
    }
}

} // extern "C"
