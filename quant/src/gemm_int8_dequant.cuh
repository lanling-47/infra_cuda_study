#pragma once
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// INT8 Weight-Only Quantized GEMM: C(fp16) = A(fp16) × dequant(B_int8, scales_fp16)
// Weights B stored as INT8 → 2x less HBM bandwidth vs FP16
// Per-group quantization: every GROUP_SIZE=128 elements along K share one FP16 scale
// Dequant happens on-the-fly in shared memory before WMMA compute

#define I8_BM 64
#define I8_BN 64
#define I8_BK 16
#define GROUP_SIZE 128

__global__ void gemm_int8_dequant_kernel(
    const half* __restrict__ A,          // [M, K] FP16 activations
    const int8_t* __restrict__ B_int8,    // [K, N] INT8 weights
    const half* __restrict__ B_scales,    // [K/GROUP_SIZE, N] FP16 scales per group
    half* __restrict__ C,                 // [M, N] FP16 output
    int M, int N, int K
) {
    __shared__ half sA[I8_BM][I8_BK + 8];
    __shared__ half sB[I8_BK][I8_BN + 8];

    int bx = blockIdx.x, by = blockIdx.y;
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warpRow = warpId / 2;
    int warpCol = warpId % 2;

    int rowBase = by * I8_BM;
    int colBase = bx * I8_BN;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = 128;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(cFrag[i][j], 0.0f);

    for (int kBlock = 0; kBlock < K; kBlock += I8_BK) {
        // Load A tile: FP16 as-is
        for (int i = tid; i < I8_BM * I8_BK; i += numThreads) {
            int r = i / I8_BK, c = i % I8_BK;
            int gR = rowBase + r, gC = kBlock + c;
            sA[r][c] = (gR < M && gC < K) ? A[gR * K + gC] : __float2half(0.0f);
        }

        // Load B tile: INT8 → dequant to FP16 using group scale
        // Each element: fp16 = int8 * scale[group]
        for (int i = tid; i < I8_BK * I8_BN; i += numThreads) {
            int r = i / I8_BN, c = i % I8_BN;
            int gR = kBlock + r, gC = colBase + c;

            if (gR < K && gC < N) {
                int8_t val = B_int8[gR * N + gC];
                // Recompute group index for this specific row
                int grp = gR / GROUP_SIZE;
                half scale = B_scales[grp * N + gC];
                sB[r][c] = __int2half_rn((int)val) * scale;
            } else {
                sB[r][c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int wm = 0; wm < 2; wm++) {
            wmma::load_matrix_sync(aFrag, &sA[warpRow * 32 + wm * 16][0], I8_BK + 8);
            for (int wn = 0; wn < 2; wn++) {
                wmma::load_matrix_sync(bFrag, &sB[0][warpCol * 32 + wn * 16], I8_BN + 8);
                wmma::mma_sync(cFrag[wm][wn], aFrag, bFrag, cFrag[wm][wn]);
            }
        }
        __syncthreads();
    }

    __shared__ float sC[I8_BM][I8_BN];
    for (int wm = 0; wm < 2; wm++)
        for (int wn = 0; wn < 2; wn++)
            wmma::store_matrix_sync(&sC[warpRow * 32 + wm * 16][warpCol * 32 + wn * 16],
                                     cFrag[wm][wn], I8_BN, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < I8_BM * I8_BN; i += numThreads) {
        int r = i / I8_BN, c = i % I8_BN;
        int gR = rowBase + r, gC = colBase + c;
        if (gR < M && gC < N) C[gR * N + gC] = __float2half(sC[r][c]);
    }
}

void run_gemm_int8_dequant(const half* d_A, const int8_t* d_B,
                            const half* d_scales, half* d_C,
                            int M, int N, int K) {
    dim3 block(32, 4);
    dim3 grid((N + I8_BN - 1) / I8_BN, (M + I8_BM - 1) / I8_BM);
    gemm_int8_dequant_kernel<<<grid, block>>>(d_A, d_B, d_scales, d_C, M, N, K);
}
