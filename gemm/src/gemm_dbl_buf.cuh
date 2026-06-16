#pragma once
#include <cuda_runtime.h>

// V5: Double buffering with smaller tiles to fit in 48KB shared memory
#define BM5 64
#define BN5 64
#define BK5 16
#define TM5 4
#define TN5 4

// sA[2][64][17] = 8704B, sB[2][16][65] = 8320B, total ~17KB — well within 48KB

__global__ void gemm_dbl_buf(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
    __shared__ float sA[2][BM5][BK5 + 1];
    __shared__ float sB[2][BK5][BN5 + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * (BN5 / TN5) + tx;
    const int numThreads = (BM5 / TM5) * (BN5 / TN5);  // 16*16 = 256

    int rowBase = by * BM5;
    int colBase = bx * BN5;
    int tRow = ty * TM5;
    int tCol = tx * TN5;

    float acc[TM5][TN5] = {};
    int numK = (K + BK5 - 1) / BK5;

    // Helper lambda-style: load tile from global to shared
    #define LOAD_TILE(buf, kOff) \
        for (int i = tid; i < BM5 * BK5; i += numThreads) { \
            int r = i / BK5, c = i % BK5; \
            int gR = rowBase + r, gC = kOff + c; \
            sA[buf][r][c] = (gR < M && gC < K) ? A[gR * K + gC] : 0.0f; \
        } \
        for (int i = tid; i < BK5 * BN5; i += numThreads) { \
            int r = i / BN5, c = i % BN5; \
            int gR = kOff + r, gC = colBase + c; \
            sB[buf][r][c] = (gR < K && gC < N) ? B[gR * N + gC] : 0.0f; \
        }

    // Preload first tile
    LOAD_TILE(0, 0);
    __syncthreads();

    for (int t = 0; t < numK; t++) {
        int cur = t & 1;
        int nxt = 1 - cur;

        // Load next tile while computing current
        if (t + 1 < numK) {
            LOAD_TILE(nxt, (t + 1) * BK5);
        }

        // Compute on current buffer
        #pragma unroll
        for (int k = 0; k < BK5; k++) {
            float aReg[TM5];
            float bReg[TN5];
            #pragma unroll
            for (int i = 0; i < TM5; i++) aReg[i] = sA[cur][tRow + i][k];
            #pragma unroll
            for (int j = 0; j < TN5; j++) bReg[j] = sB[cur][k][tCol + j];
            #pragma unroll
            for (int i = 0; i < TM5; i++)
                #pragma unroll
                for (int j = 0; j < TN5; j++)
                    acc[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }

    #undef LOAD_TILE

    #pragma unroll
    for (int i = 0; i < TM5; i++) {
        int gRow = rowBase + tRow + i;
        if (gRow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN5; j++) {
            int gCol = colBase + tCol + j;
            if (gCol < N) C[gRow * N + gCol] = acc[i][j];
        }
    }
}

void run_gemm_dbl_buf(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 block(BN5 / TN5, BM5 / TM5);  // (16, 16) = 256 threads
    dim3 grid((N + BN5 - 1) / BN5, (M + BM5 - 1) / BM5);
    gemm_dbl_buf<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
