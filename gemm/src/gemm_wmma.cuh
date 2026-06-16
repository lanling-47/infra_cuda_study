#pragma once
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// V6: Tensor Core WMMA — 16x16x16 FP16 input, FP32 accumulate
// sm_75 (Turing) supports wmma<16,16,16, half, float>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void gemm_wmma_kernel(const float* __restrict__ A_fp32,
                                  const float* __restrict__ B_fp32,
                                  float* __restrict__ C_fp32,
                                  int M, int N, int K) {
    int bx = blockIdx.x, by = blockIdx.y;
    int warpId = (threadIdx.x + threadIdx.y * blockDim.x) / 32;

    // Grid of warps: each warp handles a 16x16 output tile
    // blockDim = (32, 4) = 128 threads = 4 warps
    // Each warp: 16x16 output tile
    // Block output: 4 warps along N -> 64 wide, 1 warp along M -> 16 tall
    // Actually let's use (32, 2) = 64 threads = 2 warps per block for simplicity
    // Block output: 16 x 32

    const int BM6 = 16;
    const int BN6 = 32;  // 2 warps along N

    int rowBase = by * BM6;
    int colBase = bx * BN6;

    // Shared memory for FP16 converted tiles
    __shared__ half sA[BM6][WMMA_K + 8];  // 16x16 per K-tile
    __shared__ half sB[BN6][WMMA_K + 8];  // BN6 x 16 per K-tile (transposed for col-major)

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;
    wmma::fill_fragment(cFrag, 0.0f);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = blockDim.x * blockDim.y;

    for (int kBlock = 0; kBlock < K; kBlock += WMMA_K) {
        // Load A tile: BM6 x WMMA_K = 16 x 16 = 256 floats
        for (int i = tid; i < BM6 * WMMA_K; i += numThreads) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int gRow = rowBase + r;
            int gCol = kBlock + c;
            sA[r][c] = __float2half((gRow < M && gCol < K) ? A_fp32[gRow * K + gCol] : 0.0f);
        }
        // Load B tile: WMMA_K x BN6 = 16 x 32 = 512 floats
        // Store in col-major: sB[col][row] for col-major WMMA load
        for (int i = tid; i < WMMA_K * BN6; i += numThreads) {
            int r = i / BN6;
            int c = i % BN6;
            int gRow = kBlock + r;
            int gCol = colBase + c;
            sB[c][r] = __float2half((gRow < K && gCol < N) ? B_fp32[gRow * N + gCol] : 0.0f);
        }
        __syncthreads();

        // Each warp loads its own 16x16 sub-tile
        wmma::load_matrix_sync(aFrag, &sA[0][0], WMMA_K + 8);
        wmma::load_matrix_sync(bFrag, &sB[warpId * WMMA_N][0], WMMA_K + 8);

        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
        __syncthreads();
    }

    // Store result
    __shared__ float sC[BM6][BN6];
    wmma::store_matrix_sync(&sC[0][warpId * WMMA_N], cFrag, BN6, wmma::mem_row_major);
    __syncthreads();

    // Write back to global memory
    for (int i = tid; i < BM6 * BN6; i += numThreads) {
        int r = i / BN6;
        int c = i % BN6;
        int gRow = rowBase + r;
        int gCol = colBase + c;
        if (gRow < M && gCol < N) C_fp32[gRow * N + gCol] = sC[r][c];
    }
}

void run_gemm_wmma(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    const int BM6 = 16;
    const int BN6 = 32;
    dim3 block(32, 2);  // 64 threads = 2 warps
    dim3 grid((N + BN6 - 1) / BN6, (M + BM6 - 1) / BM6);
    gemm_wmma_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
