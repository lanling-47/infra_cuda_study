#pragma once
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8
// Each thread computes TM x TN = 64 output elements
// Block: (BM/TM) x (BN/TN) = 16 x 16 = 256 threads

__global__ void gemm_reg_tile(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    __shared__ float sA[BM][BK + 1];  // +1 to avoid bank conflicts
    __shared__ float sB[BK][BN + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * (BN / TN) + tx;  // linear thread id

    int rowBase = by * BM;
    int colBase = bx * BN;

    // Thread's output tile position
    int tRow = ty * TM;
    int tCol = tx * TN;

    // Accumulators in registers
    float acc[TM][TN] = {};

    // Number of threads loading shared memory
    const int numThreads = (BM / TM) * (BN / TN);  // 256

    for (int kBlock = 0; kBlock < K; kBlock += BK) {
        // Cooperatively load sA: BM x BK = 2048 elements, 256 threads -> 8 loads each
        for (int i = tid; i < BM * BK; i += numThreads) {
            int r = i / BK;
            int c = i % BK;
            int gRow = rowBase + r;
            int gCol = kBlock + c;
            sA[r][c] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
        }
        // Cooperatively load sB: BK x BN = 2048 elements
        for (int i = tid; i < BK * BN; i += numThreads) {
            int r = i / BN;
            int c = i % BN;
            int gRow = kBlock + r;
            int gCol = colBase + c;
            sB[r][c] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
        }
        __syncthreads();

        // Compute TM x TN output tile from shared memory
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float aReg[TM];
            float bReg[TN];

            #pragma unroll
            for (int i = 0; i < TM; i++) aReg[i] = sA[tRow + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) bReg[j] = sB[k][tCol + j];

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] += aReg[i] * bReg[j];
                }
            }
        }
        __syncthreads();
    }

    // Write back
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gRow = rowBase + tRow + i;
        if (gRow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gCol = colBase + tCol + j;
            if (gCol < N) {
                C[gRow * N + gCol] = acc[i][j];
            }
        }
    }
}

void run_gemm_reg_tile(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 block(BN / TN, BM / TM);  // (8, 16)
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_reg_tile<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
