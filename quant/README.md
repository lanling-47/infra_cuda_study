# Quantized GEMM — INT8/INT4 Weight-Only 量化推理

## 目标
实现 LLM 推理中的 Weight-Only 量化 GEMM：
- FP16 激活 × INT8/INT4 权重 → FP16 输出
- Dequant 融合在 kernel 内（从 HBM 读 INT8/INT4 → 在 shared memory 中 dequant → WMMA 计算）

## 核心理念
LLM decode（batch=1）是 memory-bound：
- 每生成一个 token 需读所有权重
- 量化权重（INT8/INT4）→ 减少 HBM 读取量 → 加速推理

## 实现

| Kernel | 权重精度 | HBM 读取量 | 特点 |
|--------|---------|-----------|------|
| FP16 WMMA | FP16 | 2B/elem | baseline，Tensor Core |
| INT8 dequant | INT8 | 1B/elem | 2x 带宽节省，per-group scale |
| INT4 dequant | INT4 | 0.5B/elem | 4x 带宽节省，2值/byte 打包 |

## Benchmark (RTX 2080 Ti, 1024×1024)

| Kernel | Time(ms) | GFLOPS | vs cuBLAS | MaxErr | Weight Size |
|--------|----------|--------|-----------|--------|-------------|
| cuBLAS_FP16 | 0.039 | 54.6 | 1.00x | - | 2.1MB |
| WMMA_FP16 | 0.268 | 8.0 | 0.15x | 0.22 | 2.1MB |
| INT8_dequant | 0.376 | 5.7 | 0.10x | 0.27 | 1.0MB |
| INT4_dequant | 0.478 | 4.5 | 0.08x | 3.92 | 0.5MB |

**关键洞察**：1024×1024 矩阵是 compute-bound（算术强度 350 >> 22 FLOPs/byte），量化反而更慢。
量化的真正收益在 **memory-bound** 场景（如 LLM decode, batch=1）。

## 构建

```bash
make       # 编译
make bench  # 跑 benchmark
```
