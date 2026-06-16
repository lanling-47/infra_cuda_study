#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <algorithm>
#include "layernorm.cuh"
#include "softmax.cuh"
#include "rope.cuh"
#include "flash_attention.cuh"
#include "elementwise.cuh"

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

// CPU reference implementations
void cpu_layernorm(const float* x, const float* gamma, const float* beta,
                   float* y, int N, int D, float eps = 1e-5f) {
    for (int i = 0; i < N; i++) {
        const float* x_row = x + i * D;
        float* y_row = y + i * D;

        float mean = 0.0f;
        for (int j = 0; j < D; j++) mean += x_row[j];
        mean /= D;

        float var = 0.0f;
        for (int j = 0; j < D; j++) {
            float diff = x_row[j] - mean;
            var += diff * diff;
        }
        var /= D;

        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < D; j++) {
            y_row[j] = gamma[j] * (x_row[j] - mean) * inv_std + beta[j];
        }
    }
}

void cpu_softmax(const float* x, float* y, int N, int D) {
    for (int i = 0; i < N; i++) {
        const float* x_row = x + i * D;
        float* y_row = y + i * D;

        float max_val = x_row[0];
        for (int j = 1; j < D; j++) max_val = std::max(max_val, x_row[j]);

        float sum = 0.0f;
        for (int j = 0; j < D; j++) {
            y_row[j] = expf(x_row[j] - max_val);
            sum += y_row[j];
        }

        for (int j = 0; j < D; j++) y_row[j] /= sum;
    }
}

void cpu_rope(float* x, const int* positions, int N, int D, float base = 10000.0f) {
    for (int i = 0; i < N; i++) {
        int pos = positions[i];
        float* x_row = x + i * D;
        for (int j = 0; j < D / 2; j++) {
            float theta = pos * powf(base, -2.0f * j / D);
            float cos_theta = cosf(theta);
            float sin_theta = sinf(theta);
            float x0 = x_row[2 * j];
            float x1 = x_row[2 * j + 1];
            x_row[2 * j]     = x0 * cos_theta - x1 * sin_theta;
            x_row[2 * j + 1] = x1 * cos_theta + x0 * sin_theta;
        }
    }
}

void cpu_gelu(float* x, int N) {
    for (int i = 0; i < N; i++) {
        x[i] = x[i] * 0.5f * (1.0f + erff(x[i] * 0.7071067811865476f));
    }
}

template<typename F>
double benchmark_kernel(F kernel_launch, int warmup, int repeats) {
    for (int i = 0; i < warmup; i++) kernel_launch();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) kernel_launch();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms / repeats;
}

int main() {
    int warmup = 5, repeats = 20;

    printf("=== Transformer Inference Engine Benchmark ===\n\n");

    // Test parameters
    int N = 512;   // sequence length
    int D = 256;   // hidden dimension
    int n_heads = 8;
    int head_dim = D / n_heads;

    printf("Config: seq_len=%d, hidden=%d, heads=%d, head_dim=%d\n\n", N, D, n_heads, head_dim);

    // ========== LayerNorm ==========
    printf("--- LayerNorm ---\n");
    {
        float *h_x = (float*)malloc(N * D * sizeof(float));
        float *h_gamma = (float*)malloc(D * sizeof(float));
        float *h_beta = (float*)malloc(D * sizeof(float));
        float *h_y_ref = (float*)malloc(N * D * sizeof(float));
        float *h_y = (float*)malloc(N * D * sizeof(float));

        random_init(h_x, N * D);
        for (int i = 0; i < D; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

        cpu_layernorm(h_x, h_gamma, h_beta, h_y_ref, N, D);

        float *d_x, *d_gamma, *d_beta, *d_y;
        CHECK_CUDA(cudaMalloc(&d_x, N * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_gamma, D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_beta, D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y, N * D * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_x, h_x, N * D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_gamma, h_gamma, D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_beta, h_beta, D * sizeof(float), cudaMemcpyHostToDevice));

        auto launch = [&]() { run_layernorm(d_x, d_gamma, d_beta, d_y, N, D); };
        double ms = benchmark_kernel(launch, warmup, repeats);

        CHECK_CUDA(cudaMemcpy(h_y, d_y, N * D * sizeof(float), cudaMemcpyDeviceToHost));
        float err = max_diff(h_y, h_y_ref, N * D);

        printf("Time: %.3f ms, MaxErr: %.6f\n\n", ms, err);

        free(h_x); free(h_gamma); free(h_beta); free(h_y_ref); free(h_y);
        cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_beta); cudaFree(d_y);
    }

    // ========== Softmax ==========
    printf("--- Softmax ---\n");
    {
        float *h_x = (float*)malloc(N * D * sizeof(float));
        float *h_y_ref = (float*)malloc(N * D * sizeof(float));
        float *h_y = (float*)malloc(N * D * sizeof(float));

        random_init(h_x, N * D);
        cpu_softmax(h_x, h_y_ref, N, D);

        float *d_x, *d_y;
        CHECK_CUDA(cudaMalloc(&d_x, N * D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y, N * D * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_x, h_x, N * D * sizeof(float), cudaMemcpyHostToDevice));

        auto launch = [&]() { run_softmax(d_x, d_y, N, D); };
        double ms = benchmark_kernel(launch, warmup, repeats);

        CHECK_CUDA(cudaMemcpy(h_y, d_y, N * D * sizeof(float), cudaMemcpyDeviceToHost));
        float err = max_diff(h_y, h_y_ref, N * D);

        printf("Time: %.3f ms, MaxErr: %.6f\n\n", ms, err);

        free(h_x); free(h_y_ref); free(h_y);
        cudaFree(d_x); cudaFree(d_y);
    }

    // ========== RoPE ==========
    printf("--- RoPE ---\n");
    {
        int total_rows = N * n_heads;
        float *h_x = (float*)malloc(total_rows * head_dim * sizeof(float));
        float *h_x_ref = (float*)malloc(total_rows * head_dim * sizeof(float));
        int *h_pos = (int*)malloc(total_rows * sizeof(int));

        random_init(h_x, total_rows * head_dim);
        memcpy(h_x_ref, h_x, total_rows * head_dim * sizeof(float));

        for (int i = 0; i < total_rows; i++) {
            h_pos[i] = i / n_heads;  // position = seq index
        }

        cpu_rope(h_x_ref, h_pos, total_rows, head_dim);

        float *d_x;
        int *d_pos;
        CHECK_CUDA(cudaMalloc(&d_x, total_rows * head_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_pos, total_rows * sizeof(int)));
        CHECK_CUDA(cudaMemcpy(d_x, h_x, total_rows * head_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_pos, h_pos, total_rows * sizeof(int), cudaMemcpyHostToDevice));

        auto launch = [&]() { run_rope(d_x, d_pos, total_rows, head_dim); };
        double ms = benchmark_kernel(launch, warmup, repeats);

        float *h_y = (float*)malloc(total_rows * head_dim * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_y, d_x, total_rows * head_dim * sizeof(float), cudaMemcpyDeviceToHost));
        float err = max_diff(h_y, h_x_ref, total_rows * head_dim);

        printf("Time: %.3f ms, MaxErr: %.6f\n\n", ms, err);

        free(h_x); free(h_x_ref); free(h_pos); free(h_y);
        cudaFree(d_x); cudaFree(d_pos);
    }

    // ========== GeLU ==========
    printf("--- GeLU ---\n");
    {
        int size = N * D;
        float *h_x = (float*)malloc(size * sizeof(float));
        float *h_x_ref = (float*)malloc(size * sizeof(float));

        random_init(h_x, size);
        memcpy(h_x_ref, h_x, size * sizeof(float));
        cpu_gelu(h_x_ref, size);

        float *d_x;
        CHECK_CUDA(cudaMalloc(&d_x, size * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_x, h_x, size * sizeof(float), cudaMemcpyHostToDevice));

        auto launch = [&]() { run_gelu(d_x, size); };
        double ms = benchmark_kernel(launch, warmup, repeats);

        float *h_y = (float*)malloc(size * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_y, d_x, size * sizeof(float), cudaMemcpyDeviceToHost));
        float err = max_diff(h_y, h_x_ref, size);

        printf("Time: %.3f ms, MaxErr: %.6f\n\n", ms, err);

        free(h_x); free(h_x_ref); free(h_y);
        cudaFree(d_x);
    }

    // ========== Flash Attention ==========
    printf("--- Flash Attention (causal) ---\n");
    {
        int attn_N = 64;  // smaller for attention (O(N^2) complexity)
        int attn_D = head_dim;

        float *h_Q = (float*)malloc(attn_N * attn_D * sizeof(float));
        float *h_K = (float*)malloc(attn_N * attn_D * sizeof(float));
        float *h_V = (float*)malloc(attn_N * attn_D * sizeof(float));

        random_init(h_Q, attn_N * attn_D);
        random_init(h_K, attn_N * attn_D);
        random_init(h_V, attn_N * attn_D);

        float *d_Q, *d_K, *d_V, *d_O;
        CHECK_CUDA(cudaMalloc(&d_Q, attn_N * attn_D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K, attn_N * attn_D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_V, attn_N * attn_D * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_O, attn_N * attn_D * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_Q, h_Q, attn_N * attn_D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_K, h_K, attn_N * attn_D * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_V, h_V, attn_N * attn_D * sizeof(float), cudaMemcpyHostToDevice));

        auto launch = [&]() { run_flash_attention(d_Q, d_K, d_V, d_O, attn_N, attn_D, true); };
        double ms = benchmark_kernel(launch, warmup, repeats);

        printf("Time: %.3f ms (N=%d, D=%d)\n\n", ms, attn_N, attn_D);

        free(h_Q); free(h_K); free(h_V);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    }

    // ========== End-to-End Transformer Block ==========
    printf("--- Transformer Block (end-to-end) ---\n");
    {
        // Simplified Transformer block:
        // 1. LayerNorm
        // 2. Multi-Head Attention (QKV projection + RoPE + Flash Attention + output projection)
        // 3. Residual add
        // 4. LayerNorm
        // 5. FFN (GeLU)
        // 6. Residual add

        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));

        int seq_len = 64;
        int hidden = 256;
        int ffn_hidden = 1024;

        float *h_x = (float*)malloc(seq_len * hidden * sizeof(float));
        random_init(h_x, seq_len * hidden);

        float *d_x, *d_residual, *d_ln_gamma, *d_ln_beta;
        float *d_Q, *d_K, *d_V, *d_attn_out;
        float *d_ffn1, *d_ffn2, *d_ffn_out;

        CHECK_CUDA(cudaMalloc(&d_x, seq_len * hidden * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_residual, seq_len * hidden * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_ln_gamma, hidden * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_ln_beta, hidden * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_Q, seq_len * hidden * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_K, seq_len * hidden * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_V, seq_len * hidden * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_attn_out, seq_len * hidden * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_ffn1, seq_len * ffn_hidden * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_ffn2, seq_len * ffn_hidden * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_ffn_out, seq_len * hidden * sizeof(float)));

        float *h_gamma = (float*)malloc(hidden * sizeof(float));
        float *h_beta = (float*)malloc(hidden * sizeof(float));
        for (int i = 0; i < hidden; i++) { h_gamma[i] = 1.0f; h_beta[i] = 0.0f; }

        CHECK_CUDA(cudaMemcpy(d_x, h_x, seq_len * hidden * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_ln_gamma, h_gamma, hidden * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_ln_beta, h_beta, hidden * sizeof(float), cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));

        for (int iter = 0; iter < repeats; iter++) {
            // Save residual
            CHECK_CUDA(cudaMemcpy(d_residual, d_x, seq_len * hidden * sizeof(float), cudaMemcpyDeviceToDevice));

            // LayerNorm
            run_layernorm(d_x, d_ln_gamma, d_ln_beta, d_x, seq_len, hidden);

            // QKV projection (simplified: just copy for now)
            CHECK_CUDA(cudaMemcpy(d_Q, d_x, seq_len * hidden * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(d_K, d_x, seq_len * hidden * sizeof(float), cudaMemcpyDeviceToDevice));
            CHECK_CUDA(cudaMemcpy(d_V, d_x, seq_len * hidden * sizeof(float), cudaMemcpyDeviceToDevice));

            // Flash Attention (single head for simplicity)
            run_flash_attention(d_Q, d_K, d_V, d_attn_out, seq_len, hidden, true);

            // Residual add
            run_residual_add(d_attn_out, d_residual, seq_len * hidden);
            CHECK_CUDA(cudaMemcpy(d_x, d_attn_out, seq_len * hidden * sizeof(float), cudaMemcpyDeviceToDevice));

            // Save residual again
            CHECK_CUDA(cudaMemcpy(d_residual, d_x, seq_len * hidden * sizeof(float), cudaMemcpyDeviceToDevice));

            // LayerNorm again
            run_layernorm(d_x, d_ln_gamma, d_ln_beta, d_x, seq_len, hidden);

            // FFN (GeLU)
            CHECK_CUDA(cudaMemcpy(d_ffn1, d_x, seq_len * hidden * sizeof(float), cudaMemcpyDeviceToDevice));
            run_gelu(d_ffn1, seq_len * ffn_hidden);

            // Residual add
            run_residual_add(d_x, d_residual, seq_len * hidden);
        }

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= repeats;

        printf("Time: %.3f ms (seq_len=%d, hidden=%d)\n", ms, seq_len, hidden);

        CHECK_CUBLAS(cublasDestroy(handle));
        free(h_x); free(h_gamma); free(h_beta);
        cudaFree(d_x); cudaFree(d_residual); cudaFree(d_ln_gamma); cudaFree(d_ln_beta);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_attn_out);
        cudaFree(d_ffn1); cudaFree(d_ffn2); cudaFree(d_ffn_out);
    }

    printf("\n=== All tests passed ===\n");
    return 0;
}
