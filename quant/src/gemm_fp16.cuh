#pragma once
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// FP16 GEMM using WMMA — C = A × B, all half precision
// Block output: 64x64, 4 warps (2x2 layout), each warp does 32x32 via 2x2 WMMA tiles
// K-dim tiled in chunks of 16 (WMMA_K)

#define BM 64
#define BN 64
#define BK 16

__global__ void gemm_fp16_kernel(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 half* __restrict__ C,
                                 int M, int N, int K) {
    __shared__ half sA[BM][BK + 8];
    __shared__ half sB[BK][BN + 8];

    int bx = blockIdx.x, by = blockIdx.y;
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warpRow = warpId / 2;  // 0 or 1
    int warpCol = warpId % 2;  // 0 or 1

    int rowBase = by * BM;
    int colBase = bx * BN;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = 128;

    // Accumulators: each warp does 2x2 WMMA tiles = 32x32 output
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(cFrag[i][j], 0.0f);

    for (int kBlock = 0; kBlock < K; kBlock += BK) {
        // Load sA: BM x BK = 64x16 = 1024 half
        for (int i = tid; i < BM * BK; i += numThreads) {
            int r = i / BK, c = i % BK;
            int gR = rowBase + r, gC = kBlock + c;
            sA[r][c] = (gR < M && gC < K) ? A[gR * K + gC] : __float2half(0.0f);
        }
        // Load sB: BK x BN = 16x64 = 1024 half
        for (int i = tid; i < BK * BN; i += numThreads) {
            int r = i / BN, c = i % BN;
            int gR = kBlock + r, gC = colBase + c;
            sB[r][c] = (gR < K && gC < N) ? B[gR * N + gC] : __float2half(0.0f);
        }
        __syncthreads();

        // WMMA compute: 2x2 tiles per warp
        for (int wm = 0; wm < 2; wm++) {
            wmma::load_matrix_sync(aFrag, &sA[warpRow * 32 + wm * 16][0], BK + 8);
            for (int wn = 0; wn < 2; wn++) {
                wmma::load_matrix_sync(bFrag, &sB[0][warpCol * 32 + wn * 16], BN + 8);
                wmma::mma_sync(cFrag[wm][wn], aFrag, bFrag, cFrag[wm][wn]);
            }
        }
        __syncthreads();
    }

    // Store: write accumulators to shared, then to global
    __shared__ float sC[BM][BN];
    for (int wm = 0; wm < 2; wm++)
        for (int wn = 0; wn < 2; wn++)
            wmma::store_matrix_sync(&sC[warpRow * 32 + wm * 16][warpCol * 32 + wn * 16],
                                     cFrag[wm][wn], BN, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < BM * BN; i += numThreads) {
        int r = i / BN, c = i % BN;
        int gR = rowBase + r, gC = colBase + c;
        if (gR < M && gC < N) C[gR * N + gC] = __float2half(sC[r][c]);
    }
}

void run_gemm_fp16(const half* d_A, const half* d_B, half* d_C, int M, int N, int K) {
    dim3 block(32, 4);  // 128 threads = 4 warps
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_fp16_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
