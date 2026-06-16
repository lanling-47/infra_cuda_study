#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include "gemm_fp16.cuh"
#include "gemm_int8_dequant.cuh"
#include "gemm_int4_dequant.cuh"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

// ---- Quantization helpers ----

// Quantize FP16 matrix to INT8 with per-group scales (group along K)
// For each group: scale = max(|val|) / 127, int8_val = round(val / scale)
void quantize_int8(const half* B_fp16, int8_t* B_int8, half* scales,
                   int K, int N, int group_size) {
    int num_groups = (K + group_size - 1) / group_size;
    for (int g = 0; g < num_groups; g++) {
        for (int n = 0; n < N; n++) {
            float max_abs = 0.0f;
            int k_start = g * group_size;
            int k_end = std::min(k_start + group_size, K);
            for (int k = k_start; k < k_end; k++) {
                float val = __half2float(B_fp16[k * N + n]);
                max_abs = std::max(max_abs, fabsf(val));
            }
            float scale = (max_abs > 0) ? max_abs / 127.0f : 1.0f;
            scales[g * N + n] = __float2half(scale);

            for (int k = k_start; k < k_end; k++) {
                float val = __half2float(B_fp16[k * N + n]);
                int q = (int)roundf(val / scale);
                q = std::max(-128, std::min(127, q));
                B_int8[k * N + n] = (int8_t)q;
            }
        }
    }
}

// Quantize FP16 matrix to INT4 packed (2 values per byte)
void quantize_int4(const half* B_fp16, uint8_t* B_int4, half* scales,
                   int K, int N, int group_size) {
    int num_groups = (K + group_size - 1) / group_size;
    // Allocate temp int4 array (unpack)
    std::vector<int> B_int4_unpacked(K * N);

    for (int g = 0; g < num_groups; g++) {
        for (int n = 0; n < N; n++) {
            float max_abs = 0.0f;
            int k_start = g * group_size;
            int k_end = std::min(k_start + group_size, K);
            for (int k = k_start; k < k_end; k++) {
                float val = __half2float(B_fp16[k * N + n]);
                max_abs = std::max(max_abs, fabsf(val));
            }
            float scale = (max_abs > 0) ? max_abs / 7.0f : 1.0f;
            scales[g * N + n] = __float2half(scale);

            for (int k = k_start; k < k_end; k++) {
                float val = __half2float(B_fp16[k * N + n]);
                int q = (int)roundf(val / scale);
                q = std::max(-8, std::min(7, q));
                B_int4_unpacked[k * N + n] = q;
            }
        }
    }

    // Pack: 2 values per byte
    // Layout: B_int4[k][n/2], low nibble = even col, high nibble = odd col
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n += 2) {
            int low = B_int4_unpacked[k * N + n] & 0xF;
            int high = (n + 1 < N) ? (B_int4_unpacked[k * N + n + 1] & 0xF) : 0;
            B_int4[k * (N / 2) + n / 2] = (uint8_t)(low | (high << 4));
        }
    }
}

// ---- Benchmark infrastructure ----

void random_init_fp16(half* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = __float2half((float)rand() / RAND_MAX * 2.0f - 1.0f);
    }
}

float max_diff_fp16(const half* a, const half* b, int size) {
    float maxd = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(__half2float(a[i]) - __half2float(b[i]));
        if (diff > maxd) maxd = diff;
    }
    return maxd;
}

typedef void (*KernelFn)(const half*, const half*, half*, int, int, int);

double benchmark_fp16(KernelFn kernel, const half* d_A, const half* d_B, half* d_C,
                      int M, int N, int K, int warmup, int repeats) {
    for (int i = 0; i < warmup; i++) kernel(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) kernel(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / repeats;
}

double benchmark_cublas_fp16(cublasHandle_t handle, const half* d_A, const half* d_B, half* d_C,
                              int M, int N, int K, int warmup, int repeats) {
    __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);
    for (int i = 0; i < warmup; i++)
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                   &alpha_h, d_B, CUDA_R_16F, N, d_A, CUDA_R_16F, K,
                                   &beta_h, d_C, CUDA_R_16F, N,
                                   CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++)
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                   &alpha_h, d_B, CUDA_R_16F, N, d_A, CUDA_R_16F, K,
                                   &beta_h, d_C, CUDA_R_16F, N,
                                   CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / repeats;
}

// Quantized kernel benchmark
typedef void (*QuantKernelFn)(const half*, const uint8_t*, const half*, half*, int, int, int);
typedef void (*I8KernelFn)(const half*, const int8_t*, const half*, half*, int, int, int);

double benchmark_quant_i8(I8KernelFn kernel, const half* d_A, const int8_t* d_B,
                           const half* d_scales, half* d_C, int M, int N, int K,
                           int warmup, int repeats) {
    for (int i = 0; i < warmup; i++) kernel(d_A, d_B, d_scales, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) kernel(d_A, d_B, d_scales, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / repeats;
}

double benchmark_quant_i4(QuantKernelFn kernel, const half* d_A, const uint8_t* d_B,
                           const half* d_scales, half* d_C, int M, int N, int K,
                           int warmup, int repeats) {
    for (int i = 0; i < warmup; i++) kernel(d_A, d_B, d_scales, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) kernel(d_A, d_B, d_scales, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / repeats;
}

int main(int argc, char** argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);

    int warmup = 5, repeats = 20;
    int group_size = 128;

    printf("=== Quantized GEMM Benchmark ===\n");
    printf("Matrix: M=%d, N=%d, K=%d (FP16)\n", M, N, K);
    printf("Quant group size: %d\n", group_size);
    printf("Warmup: %d, Repeats: %d\n\n", warmup, repeats);

    // Host allocations
    half *h_A = (half*)malloc((size_t)M * K * sizeof(half));
    half *h_B = (half*)malloc((size_t)K * N * sizeof(half));
    half *h_C_ref = (half*)malloc((size_t)M * N * sizeof(half));
    half *h_C_test = (half*)malloc((size_t)M * N * sizeof(half));

    srand(42);
    random_init_fp16(h_A, M * K);
    random_init_fp16(h_B, K * N);

    // Quantize B on host
    int num_groups = (K + group_size - 1) / group_size;
    int8_t *h_B_int8 = (int8_t*)malloc((size_t)K * N * sizeof(int8_t));
    half *h_scales_i8 = (half*)malloc((size_t)num_groups * N * sizeof(half));
    uint8_t *h_B_int4 = (uint8_t*)malloc((size_t)K * (N / 2) * sizeof(uint8_t));
    half *h_scales_i4 = (half*)malloc((size_t)num_groups * N * sizeof(half));

    printf("Quantizing B to INT8 and INT4...\n");
    quantize_int8(h_B, h_B_int8, h_scales_i8, K, N, group_size);
    quantize_int4(h_B, h_B_int4, h_scales_i4, K, N, group_size);

    // Device allocations
    half *d_A, *d_B, *d_C, *d_C_ref;
    int8_t *d_B_int8;
    uint8_t *d_B_int4;
    half *d_scales_i8, *d_scales_i4;

    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, (size_t)M * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C_ref, (size_t)M * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_int8, (size_t)K * N * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_B_int4, (size_t)K * (N / 2) * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_scales_i8, (size_t)num_groups * N * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_scales_i4, (size_t)num_groups * N * sizeof(half)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, (size_t)M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, (size_t)K * N * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_int8, h_B_int8, (size_t)K * N * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_int4, h_B_int4, (size_t)K * (N / 2) * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scales_i8, h_scales_i8, (size_t)num_groups * N * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scales_i4, h_scales_i4, (size_t)num_groups * N * sizeof(half), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    double gflops = 2.0 * M * N * K / 1e9;

    // Model size comparison
    double fp16_size = (double)K * N * 2 / 1e6;
    double int8_size = (double)K * N * 1 / 1e6;
    double int4_size = (double)K * N * 0.5 / 1e6;
    printf("Weight memory: FP16=%.1fMB  INT8=%.1fMB (2x)  INT4=%.1fMB (4x)\n\n",
           fp16_size, int8_size, int4_size);

    printf("%-20s %10s %10s %10s %10s\n", "Kernel", "Time(ms)", "GFLOPS", "vs cuBLAS", "MaxErr");
    printf("%-20s %10s %10s %10s %10s\n", "------", "------", "------", "---------", "------");

    // cuBLAS FP16 reference
    double ms_cublas = benchmark_cublas_fp16(handle, d_A, d_B, d_C_ref, M, N, K, warmup, repeats);
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C_ref, (size_t)M * N * sizeof(half), cudaMemcpyDeviceToHost));
    printf("%-20s %10.3f %10.1f %10s %10s\n", "cuBLAS_FP16", ms_cublas, gflops/ms_cublas, "1.00x", "-");

    // Our FP16 WMMA
    run_gemm_fp16(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C_test, d_C, (size_t)M * N * sizeof(half), cudaMemcpyDeviceToHost));
    float err = max_diff_fp16(h_C_test, h_C_ref, M * N);
    double ms_fp16 = benchmark_fp16(run_gemm_fp16, d_A, d_B, d_C, M, N, K, warmup, repeats);
    printf("%-20s %10.3f %10.1f %10.2fx %10.4f\n", "WMMA_FP16", ms_fp16, gflops/ms_fp16, ms_cublas/ms_fp16, err);

    // INT8 dequant
    run_gemm_int8_dequant(d_A, d_B_int8, d_scales_i8, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C_test, d_C, (size_t)M * N * sizeof(half), cudaMemcpyDeviceToHost));
    err = max_diff_fp16(h_C_test, h_C_ref, M * N);
    double ms_i8 = benchmark_quant_i8(
        (I8KernelFn)run_gemm_int8_dequant, d_A, d_B_int8, d_scales_i8, d_C, M, N, K, warmup, repeats);
    printf("%-20s %10.3f %10.1f %10.2fx %10.4f\n", "INT8_dequant", ms_i8, gflops/ms_i8, ms_cublas/ms_i8, err);

    // INT4 dequant
    run_gemm_int4_dequant(d_A, d_B_int4, d_scales_i4, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C_test, d_C, (size_t)M * N * sizeof(half), cudaMemcpyDeviceToHost));
    err = max_diff_fp16(h_C_test, h_C_ref, M * N);
    double ms_i4 = benchmark_quant_i4(
        (QuantKernelFn)run_gemm_int4_dequant, d_A, d_B_int4, d_scales_i4, d_C, M, N, K, warmup, repeats);
    printf("%-20s %10.3f %10.1f %10.2fx %10.4f\n", "INT4_dequant", ms_i4, gflops/ms_i4, ms_cublas/ms_i4, err);

    printf("\n");

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C)); CHECK_CUDA(cudaFree(d_C_ref));
    CHECK_CUDA(cudaFree(d_B_int8)); CHECK_CUDA(cudaFree(d_B_int4));
    CHECK_CUDA(cudaFree(d_scales_i8)); CHECK_CUDA(cudaFree(d_scales_i4));
    free(h_A); free(h_B); free(h_C_ref); free(h_C_test);
    free(h_B_int8); free(h_scales_i8);
    free(h_B_int4); free(h_scales_i4);

    return 0;
}
