#pragma once
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// INT4 Weight-Only Quantized GEMM: C(fp16) = A(fp16) × dequant(B_int4, scales_fp16)
// Weights B stored as INT4 (2 elements per byte) → 4x less HBM bandwidth vs FP16
// Per-group quantization: every GROUP_SIZE=128 elements along K share one FP16 scale
// Pack format: byte = (val_low & 0xF) | ((val_high & 0xF) << 4)
// INT4 range: [-8, 7] (signed 4-bit)

#define I4_BM 64
#define I4_BN 64
#define I4_BK 16
#define I4_GROUP_SIZE 128

// Unpack signed 4-bit: lower nibble of byte
__device__ __forceinline__ int unpack_int4_low(uint8_t byte) {
    int val = byte & 0xF;
    return (val >= 8) ? val - 16 : val;  // sign extend
}

// Unpack signed 4-bit: upper nibble of byte
__device__ __forceinline__ int unpack_int4_high(uint8_t byte) {
    int val = (byte >> 4) & 0xF;
    return (val >= 8) ? val - 16 : val;
}

__global__ void gemm_int4_dequant_kernel(
    const half* __restrict__ A,           // [M, K] FP16 activations
    const uint8_t* __restrict__ B_int4,   // [K, N/2] INT4 weights packed
    const half* __restrict__ B_scales,    // [K/GROUP_SIZE, N] FP16 scales
    half* __restrict__ C,                 // [M, N] FP16 output
    int M, int N, int K
) {
    __shared__ half sA[I4_BM][I4_BK + 8];
    __shared__ half sB[I4_BK][I4_BN + 8];

    int bx = blockIdx.x, by = blockIdx.y;
    int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warpRow = warpId / 2;
    int warpCol = warpId % 2;

    int rowBase = by * I4_BM;
    int colBase = bx * I4_BN;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = 128;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag[2][2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(cFrag[i][j], 0.0f);

    for (int kBlock = 0; kBlock < K; kBlock += I4_BK) {
        // Load A: FP16
        for (int i = tid; i < I4_BM * I4_BK; i += numThreads) {
            int r = i / I4_BK, c = i % I4_BK;
            int gR = rowBase + r, gC = kBlock + c;
            sA[r][c] = (gR < M && gC < K) ? A[gR * K + gC] : __float2half(0.0f);
        }

        // Load B: INT4 packed → dequant to FP16
        // B_int4 layout: row k, col c → byte at B_int4[k * (N/2) + c/2]
        //   if c is even: low nibble; if c is odd: high nibble
        for (int i = tid; i < I4_BK * I4_BN; i += numThreads) {
            int r = i / I4_BN, c = i % I4_BN;
            int gR = kBlock + r, gC = colBase + c;

            if (gR < K && gC < N) {
                uint8_t packed = B_int4[gR * (N / 2) + gC / 2];
                int val = (gC % 2 == 0) ? unpack_int4_low(packed) : unpack_int4_high(packed);

                int grp = gR / I4_GROUP_SIZE;
                half scale = B_scales[grp * N + gC];
                sB[r][c] = __int2half_rn(val) * scale;
            } else {
                sB[r][c] = __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int wm = 0; wm < 2; wm++) {
            wmma::load_matrix_sync(aFrag, &sA[warpRow * 32 + wm * 16][0], I4_BK + 8);
            for (int wn = 0; wn < 2; wn++) {
                wmma::load_matrix_sync(bFrag, &sB[0][warpCol * 32 + wn * 16], I4_BN + 8);
                wmma::mma_sync(cFrag[wm][wn], aFrag, bFrag, cFrag[wm][wn]);
            }
        }
        __syncthreads();
    }

    __shared__ float sC[I4_BM][I4_BN];
    for (int wm = 0; wm < 2; wm++)
        for (int wn = 0; wn < 2; wn++)
            wmma::store_matrix_sync(&sC[warpRow * 32 + wm * 16][warpCol * 32 + wn * 16],
                                     cFrag[wm][wn], I4_BN, wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < I4_BM * I4_BN; i += numThreads) {
        int r = i / I4_BN, c = i % I4_BN;
        int gR = rowBase + r, gC = colBase + c;
        if (gR < M && gC < N) C[gR * N + gC] = __float2half(sC[r][c]);
    }
}

void run_gemm_int4_dequant(const half* d_A, const uint8_t* d_B,
                            const half* d_scales, half* d_C,
                            int M, int N, int K) {
    dim3 block(32, 4);
    dim3 grid((N + I4_BN - 1) / I4_BN, (M + I4_BM - 1) / I4_BM);
    gemm_int4_dequant_kernel<<<grid, block>>>(d_A, d_B, d_scales, d_C, M, N, K);
}
