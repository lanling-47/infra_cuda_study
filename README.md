# infra_cuda_study

CUDA kernel 优化与 GPU 推理能力建设，面向 AI Infra / 端侧推理方向。

## 项目结构

```
├── gemm/         # GEMM kernel 分阶段优化 (V1-V6)
│   ├── src/
│   ├── notes.md
│  └── README.md
├── quant/        # INT8/INT4 Weight-Only 量化 GEM
│   ├── src/
│   ├── notes.md
│   └── README.md
└── ...
```

## 一、GEMM Kernel 优化 (gemm/)

目标：从 naive 实现逐步优化至接近 cuBLAS 性能，掌握 GPU 微架构与 kernel 调优方法。

**优化路线与结果：**

| 版本 | 优化策略 | 状态 | Time(ms) | vs cuBLAS |
|------|---------|------|----------|-----------|
| V1 naive | 一个线程一个输出元素 | done | 2.392 | 0.09x |
| V2 shared_mem_tile | Shared memory tiling | done | 1.510 | 0.15x |
| V3 reg_tile | 寄存器分块 (8×8) | done | 0.457 | **0.50x** |
| V4 vec_load | float4 向量化加载 | done | 0.389 | 0.58x |
| V5 dbl_buf | 双缓冲 | done | 0.353 | **0.64x** |
| V6 wmma_tc | WMMA Tensor Core | done | 0.627 | 0.36x |

**Benchmark (RTX 2080 Ti, 1024×1024, FP32)：**
- cuBLAS baseline: **0.227ms (9.5 GFLOPS)**
- 最佳手写 (V5): **0.353ms (6.1 GFLOPS)** — 达到 cuBLAS 的 64%

## 二、量化 GEMM (quant/)

目标：实现 LLM 推理中的 Weight-Only 量化 GEMM，理解量化对 memory-bound 场景的收益。

**实现：**
- FP16 WMMA baseline（Tensor Core）
- INT8 dequant + WMMA（per-group scale, 2x 带宽节省）
- INT4 dequant + WMMA（2值/byte 打包, 4x 带宽节省）

**Benchmark (RTX 2080 Ti, 1024×1024, FP16)：**
- cuBLAS FP16 baseline: **0.039ms (54.6 GFLOPS)**
- 关键洞察：1024×1024 矩阵是 compute-bound，量化反而更慢。量化收益在 **memory-bound** 场景（LLM decode, batch=1）。

## 构建

```bash
cd gem    # 或 quant
make       # 编译
make bench   # 跑 benchmark
```

## 环境

- GPU: NVIDIA RTX 2080 Ti (sm_75, Turing)
- CUDA: 12.8
- OS: Ubuntu 22.04
