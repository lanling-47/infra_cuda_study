# infra_cuda_study

CUDA kernel 优化与 GPU 推理能力建设，面向 AI Infra / 端侧推理方向。

## 项目结构

```
├── gem/          # GEMM kernel 分阶段优化 (V1-V6)
└── ...
```

## 一、GEMM Kernel 优化

目标：从 naive 实现逐步优化至接近 cuBLAS 性能，掌握 GPU 微架构与 kernel 调优方法。

**优化路线：**

| 版本 | 优化策略 | 状态 |
|------|---------|------|
| V1 naive | 一个线程一个输出元素 | done |
| V2 shared_mem_tile | Shared memory tiling，减少 global memory 访问 | done |
| V3 reg_tile | 寄存器分块，每个线程算多个输出 | TODO |
| V4 vectorized_load | float4 向量化加载 | TODO |
| V5 double_buffer | 异步加载 + 计算重叠 | TODO |
| V6 tensor_core | WMMA Tensor Core | TODO |

**Benchmark (RTX 2080 Ti, 1024×1024)：**

| Kernel | Time(ms) | GFLOPS | vs cuBLAS |
|--------|---------|--------|-----------|
| cuBLAS | 0.227 | 9.5 | 1.00x |
| naive | 2.392 | 0.9 | 0.09x |
| shared_mem_tile | 1.509 | 1.4 | 0.15x |

## 构建

```bash
make      # 编译
make bench  # 跑 benchmark
```
