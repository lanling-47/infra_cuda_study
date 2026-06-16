#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "gemm_naive.cuh"
#include "gemm_shared.cuh"

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

void random_init(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

float max_diff(const float* a, const float* b, int size) {
    float maxd = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > maxd) maxd = diff;
    }
    return maxd;
}

double benchmark(void (*kernel)(const float*, const float*, float*, int, int, int),
                 const float* d_A, const float* d_B, float* d_C,
                 int M, int N, int K, int warmup, int repeats) {
    for (int i = 0; i < warmup; i++) {
        kernel(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) {
        kernel(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / repeats;
}

double benchmark_cublas(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C,
                        int M, int N, int K, int warmup, int repeats) {
    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < warmup; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                  &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                  &alpha, d_B, N, d_A, K, &beta, d_C, N));
    }
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

    printf("=== GEMM Benchmark ===\n");
    printf("Matrix: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Warmup: %d, Repeats: %d\n\n", warmup, repeats);

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C_ref = (float*)malloc(sizeC);
    float *h_C_test = (float*)malloc(sizeC);

    srand(42);
    random_init(h_A, M * K);
    random_init(h_B, K * N);

    float *d_A, *d_B, *d_C, *d_C_ref;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));
    CHECK_CUDA(cudaMalloc(&d_C_ref, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // cuBLAS as reference
    double ms_cublas = benchmark_cublas(handle, d_A, d_B, d_C_ref, M, N, K, warmup, repeats);
    CHECK_CUDA(cudaMemcpy(h_C_ref, d_C_ref, sizeC, cudaMemcpyDeviceToHost));

    double gflops = 2.0 * M * N * K / 1e9;

    printf("%-20s %10s %10s %10s %10s\n", "Kernel", "Time(ms)", "GFLOPS", "vs cuBLAS", "MaxErr");
    printf("%-20s %10s %10s %10s %10s\n", "-------", "-------", "------", "---------", "------");
    printf("%-20s %10.3f %10.1f %10s %10s\n", "cuBLAS", ms_cublas, gflops/ms_cublas, "1.00x", "-");

    // Naive
    run_gemm_naive(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C_test, d_C, sizeC, cudaMemcpyDeviceToHost));
    float err = max_diff(h_C_test, h_C_ref, M * N);
    double ms_naive = benchmark(run_gemm_naive, d_A, d_B, d_C, M, N, K, warmup, repeats);
    printf("%-20s %10.3f %10.1f %10.2fx %10.4f\n", "naive", ms_naive, gflops/ms_naive, ms_cublas/ms_naive, err);

    // Shared memory tiling
    run_gemm_shared(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C_test, d_C, sizeC, cudaMemcpyDeviceToHost));
    err = max_diff(h_C_test, h_C_ref, M * N);
    double ms_shared = benchmark(run_gemm_shared, d_A, d_B, d_C, M, N, K, warmup, repeats);
    printf("%-20s %10.3f %10.1f %10.2fx %10.4f\n", "shared_mem_tile", ms_shared, gflops/ms_shared, ms_cublas/ms_shared, err);

    printf("\n");

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_C_ref));
    free(h_A); free(h_B); free(h_C_ref); free(h_C_test);

    return 0;
}
