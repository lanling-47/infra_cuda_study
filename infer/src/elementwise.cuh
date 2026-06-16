#pragma once
#include <cuda_runtime.h>
#include <math.h>

// GeLU activation: y = x * 0.5 * (1 + erf(x / sqrt(2)))
__global__ void gelu_kernel(float* __restrict__ x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = x[idx];
        x[idx] = val * 0.5f * (1.0f + erff(val * 0.7071067811865476f));
    }
}

// ReLU: y = max(0, x)
__global__ void relu_kernel(float* __restrict__ x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        x[idx] = fmaxf(0.0f, x[idx]);
    }
}

// Residual add: y = x + residual (in-place on y)
__global__ void residual_add_kernel(float* __restrict__ y, const float* __restrict__ residual, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] += residual[idx];
    }
}

// SwiGLU (used in LLaMA): output = (x1 * sigmoid(x1)) * x2
// x1 and x2 are both [N, D], output written to x1
__global__ void swiglu_kernel(float* __restrict__ x1, const float* __restrict__ x2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float gate = x1[idx];
        float silu = gate / (1.0f + expf(-gate));  // SiLU = Swish
        x1[idx] = silu * x2[idx];
    }
}

void run_gelu(float* d_x, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(d_x, N);
}

void run_relu(float* d_x, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(d_x, N);
}

void run_residual_add(float* d_y, const float* d_residual, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads>>>(d_y, d_residual, N);
}

void run_swiglu(float* d_x1, const float* d_x2, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads>>>(d_x1, d_x2, N);
}
