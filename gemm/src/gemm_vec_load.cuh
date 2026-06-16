#pragma once
#include <cuda_runtime.h>

#define BM4 128
#define BN4 128
#define BK4 32
#define TM4 8
#define TN4 8
// V4: float4 vectorized global memory loads
// BK4=32 so each thread loads float4 (4 floats) per row

__global__ void gemm_vec_load(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int N, int K) {
    __shared__ float sA[BM4][BK4 + 1];
    __shared__ float sB[BK4][BN4 + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * (BN4 / TN4) + tx;
    const int numThreads = (BM4 / TM4) * (BN4 / TN4);  // 256

    int rowBase = by * BM4;
    int colBase = bx * BN4;
    int tRow = ty * TM4;
    int tCol = tx * TN4;

    float acc[TM4][TN4] = {};

    for (int kBlock = 0; kBlock < K; kBlock += BK4) {
        // Load sA: BM4 x BK4 = 4096 elements, 256 threads -> 16 loads each
        // Use float4: 4096 / 4 = 1024 float4 loads, 256 threads -> 4 float4 each
        for (int i = tid; i < (BM4 * BK4) / 4; i += numThreads) {
            int flatIdx = i * 4;
            int r = flatIdx / BK4;
            int c = flatIdx % BK4;
            int gRow = rowBase + r;
            int gCol = kBlock + c;
            float4 val;
            if (gRow < M && gCol + 3 < K && ((size_t)(&A[gRow * K + gCol]) % 16 == 0)) {
                val = *reinterpret_cast<const float4*>(&A[gRow * K + gCol]);
            } else {
                val.x = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
                val.y = (gRow < M && gCol+1 < K) ? A[gRow * K + gCol+1] : 0.0f;
                val.z = (gRow < M && gCol+2 < K) ? A[gRow * K + gCol+2] : 0.0f;
                val.w = (gRow < M && gCol+3 < K) ? A[gRow * K + gCol+3] : 0.0f;
            }
            sA[r][c] = val.x;
            sA[r][c+1] = val.y;
            sA[r][c+2] = val.z;
            sA[r][c+3] = val.w;
        }

        // Load sB: BK4 x BN4 = 4096 elements -> same approach
        for (int i = tid; i < (BK4 * BN4) / 4; i += numThreads) {
            int flatIdx = i * 4;
            int r = flatIdx / BN4;
            int c = flatIdx % BN4;
            int gRow = kBlock + r;
            int gCol = colBase + c;
            float4 val;
            if (gRow < K && gCol + 3 < N && ((size_t)(&B[gRow * N + gCol]) % 16 == 0)) {
                val = *reinterpret_cast<const float4*>(&B[gRow * N + gCol]);
            } else {
                val.x = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
                val.y = (gRow < K && gCol+1 < N) ? B[gRow * N + gCol+1] : 0.0f;
                val.z = (gRow < K && gCol+2 < N) ? B[gRow * N + gCol+2] : 0.0f;
                val.w = (gRow < K && gCol+3 < N) ? B[gRow * N + gCol+3] : 0.0f;
            }
            sB[r][c] = val.x;
            sB[r][c+1] = val.y;
            sB[r][c+2] = val.z;
            sB[r][c+3] = val.w;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK4; k++) {
            float aReg[TM4];
            float bReg[TN4];
            #pragma unroll
            for (int i = 0; i < TM4; i++) aReg[i] = sA[tRow + i][k];
            #pragma unroll
            for (int j = 0; j < TN4; j++) bReg[j] = sB[k][tCol + j];
            #pragma unroll
            for (int i = 0; i < TM4; i++)
                #pragma unroll
                for (int j = 0; j < TN4; j++)
                    acc[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM4; i++) {
        int gRow = rowBase + tRow + i;
        if (gRow >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN4; j++) {
            int gCol = colBase + tCol + j;
            if (gCol < N) C[gRow * N + gCol] = acc[i][j];
        }
    }
}

void run_gemm_vec_load(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 block(BN4 / TN4, BM4 / TM4);
    dim3 grid((N + BN4 - 1) / BN4, (M + BM4 - 1) / BM4);
    gemm_vec_load<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}
