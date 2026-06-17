# infra_cuda_study

CUDA kernel 优化与 GPU 推理能力建设，面向 AI Infra / 端侧推理方向。

## 项目结构

```
├── gemm/              # GEM kernel 分阶段优化 (V1-V6)
│   ├── src/
│   ├── notes.md
│   └── README.md
├── quant/             # INT8/INT4 Weight-Only 量化 GEM
│   ├── src/
│   ├── notes.md
│   └── README.md
├── infer/             # Transformer 推理引擎 GPU Backend
│   ├── src/
│   ├── notes.md
│   └── README.md
├── whisper_demo/      # Whisper-tiny 端到端推理优化 (V2 批量 kernels)
│   ├── custom_ops/    # PyTorch C++ Extension (Flash Attn, LayerNorm, Softmax)
│   ├── model_surgery.py
│   └── README.md
├── nano_vllm_compat/  # nano-vllm sm_75 兼容层 + 4 项 CUDA 优化
│   ├── paged_attention.cu
│   ├── kv_store_cuda.cu
│   ├── int8_linear.py
│   ├── speculative.py
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
| V6 wmma_tc | WMA Tensor Core | done | 0.627 | 0.36x |

**Benchmark (RTX 2080 Ti, 1024×1024, FP32)：**
- cuBLAS baseline: **0.227ms (9.5 GFLOPS)**
- 最佳手写 (V5): **0.353ms (6.1 GFLOPS)** — 达到 cuBLAS 的 64%

## 二、量化 GEM (quant/)

目标：实现 LLM 推理中的 Weight-Only 量化 GEMM，理解量化对 memory-bound 场景的收益。

**实现：**
- FP16 WMA baseline（Tensor Core）
- INT8 dequant + WMMA（per-group scale, 2x 带宽节省）
- INT4 dequant + WMMA（2值/byte 打包, 4x 带宽节省）

**Benchmark (RTX 2080 Ti, 1024×1024, FP16)：**
- cuBLAS FP16 baseline: **0.039ms (54.6 GFLOPS)**
- 关键洞察：1024×1024 矩阵是 compute-bound，量化反而更慢。量化收益在 **memory-bound** 场景（LLM decode, batch=1）。

## 三、Transformer 推理引擎 (infer/)

目标：实现轻量级 Transformer 推理引擎 GPU Backend，掌握 LLM 推理核心算子。

**核心算子：**
- **LayerNorm**: Welford 单次遍历算法，warp/block reduction
- **Softmax**: Online Softmax，适合长序列
- **RoPE**: 旋转位置编码，LLaMA/Mistral 标配
- **Flash Attention**: 分块 + Online Softmax，O(N) 内存
- **Element-wise**: GeLU, SwiGLU, 残差连接

**Benchmark (RTX 2080 Ti)：**

| Kernel | Time (ms) | Shape |
|--------|-----------|
| LayerNorm | 0.006 | 512×256 |
| Softmax | 0.005 | 512×256 |
| RoPE | 0.018 | 4096×32 |
| GeLU | 0.003 | 512×256 |
| Flash Attention | 0.063 | 64×32, causal |
| **Transformer Block** | **0.096** | 64×256, end-to-end |

## 四、Whisper-tiny 端到端推理优化 (whisper_demo/)

目标：将自研 CUDA kernels 编译为 PyTorch Extension，应用到 Whisper 模型，验证端到端推理性能提升。

**V2 升级：批量多头注意力支持**
- Flash Attention: [B, H, N, D] 4D 批量多头，替代单头 2D 实现
- LayerNorm: [B, N, D] 3D 支持 + FP16 变体
- Softmax: [B, H, N, M] 4D 支持，修复 -inf warp reduction 导致的 NaN 问题
- PyTorch C++ Extension: pybind11 封装，支持 `torch.nn.functional` 风格调用

**关键发现：Monkey-patch 性能陷阱**
- 尝试通过 monkey-patch 替换 Whisper 的 `LayerNorm.forward` 为自定义 CUDA kernel
- **结果**：即使替换为 `return x`（无计算），延迟仍从 96ms 暴涨到 10000ms（100× 退化）
- **根因**：Whisper decode 阶段每推理一次调用 LayerNorm ~35,000 次，monkey-patch 破坏了 PyTorch JIT/autograd 的内部优化路径
- **结论**：端到端 kernel 替换需通过模型重构或编译器 pass，而非运行时 monkey-patch

**性能对比 (RTX 2080 Ti, 30 秒音频转写)：**

| 指标 | Baseline | PyTorch SDPA | 自定义 CUDA Kernels |
|------|----------|--------------|---------------------|
| 平均延迟 | 95.50 ms | 96.29 ms | **95.04 ms** |
| 最佳延迟 | 32.72 ms | 29.53 ms | **28.60 ms** |
| 吞吐量 | 10.47 trans/sec | 10.39 trans/sec | **10.52 trans/sec** |

## 五、nano-vllm sm_75 兼容层 + 4 项 CUDA 优化 (nano_vllm_compat/)

目标：在 RTX 2080 Ti (sm_75 Turing) 上运行 nano-vllm，并实现 4 项 LLM 推理优化。

**背景：**
- nano-vllm: 轻量级 vLLM 实现（~1200 行 Python），支持 Qwen3，内置 CUDA Graph、prefix caching、paged KV cache
- 挑战：Flash Attention 2 不支持 sm_75，nano-vllm 依赖 `flash_attn_with_kvcache` 和 `flash_attn_varlen_func`

**兼容层实现：**
- `flash_attn_varlen_func` → PyTorch SDPA (per-sequence loop, GQA 支持)
- `flash_attn_with_kvcache` → SDPA + paged KV cache gather

**优化 A: Triton → CUDA KV Cache Store Kernel**
- **文件**: `kv_store_cuda.cu`, `kv_store_cuda_ext.cpp`, `kv_store.py`
- **原理**: float4 向量化加载/存储，2D grid `(tokens, D/4)`，grid-stride loop
- **特性**: 支持 FP16/FP32，Triton fallback
- **预期**: 10-20% decode 加速

**优化 B: Paged Attention CUDA Kernel (核心亮点)**
- **文件**: `paged_attention.cu`, `paged_attention_ext.cpp`, `paged_attn.py`
- **算法**: FlashDecoding 风格两遍并行
  - Pass 1: 每个 KV cache page 一个 thread block，计算 partial attention + logsumexp
  - Pass 2: 跨 pages 归约，logsumexp 加权合并
- **特性**: 支持 GQA、online softmax、warp-level reduction
- **意义**: 在 sm_75 上实现原本不支持的 paged attention

**优化 C: INT8 Weight-Only 量化**
- **文件**: `int8_linear.py`, `int8_quantize.py`, `model_runner_int8.py`
- **原理**: 加载时 FP16 → INT8 symmetric per-row 量化，推理时 Triton fused dequant+GEMM
- **激活**: `NANOVLLM_INT8=1` 环境变量
- **预期**: 显存减半，decode (memory-bound) 提速 ~2x

**优化 D: Speculative Decoding**
- **文件**: `speculative.py`
- **算法**: Draft-Verify 两阶段
  - Draft: 小模型自回归生成 K 个候选 token
  - Verify: 大模型一次性前向 K+1 token (prefill-like)
  - Accept: 贪心匹配 (target argmax == draft token)
- **预期**: 单请求延迟降低 2-3x（大+小模型组合时）

**快速开始：**
```bash
# 1. 安装兼容层
bash nano_vllm_compat/setup_sm75.sh

# 2. 运行测试
python3 nano_vllm_compat/test_sm75.py              # 基础功能
python3 nano_vllm_compat/test_paged_attn.py         # B: Paged Attention
NANOVLLM_INT8=1 python3 nano_vllm_compat/test_int8.py  # C: INT8
python3 nano_vllm_compat/test_speculative.py        # D: Speculative
```

## 构建

```bash
cd gemm    # 或 quant / infer
make        # 编译
make bench   # 跑 benchmark
```

## 环境

- GPU: NVIDIA RTX 2080 Ti (sm_75, Turing, 22GB)
- CUDA: 12.8
- PyTorch: 2.5.1
- Triton: 3.1.0
- OS: Ubuntu 22.04

## 面试话术要点

**30 秒版本：**
> 我自主完成了 CUDA + GPU 推理的全栈能力建设，通过三个递进式项目系统性地掌握了 GPU 微架构和推理优化。
>
> 第一个项目是 GEMM kernel 分阶段优化，从 naive 实现开始，逐步引入 shared memory tiling、寄存器分块、向量化加载、双缓冲和 Tensor Core，最终手写版本达到 cuBLAS 64% 的性能。这个过程让我深入理解了 GPU 内存层次、warp 调度、bank conflict 等底层机制。
>
> 第二个项目是量化推理，实现了 INT8/INT4 weight-only GEMM。通过对比实验发现，1024x1024 矩阵是 compute-bound，量化反而更慢；只有在 LLM decode 这种 memory-bound 场景（batch=1）才能看到收益。这让我理解了量化优化的适用场景和 roofline 模型。
>
> 第三个项目是 Transformer 推理引擎，实现了 Flash Attention、RoPE、LayerNorm 等核心算子。特别是 Flash Attention 的优化，从 memory-bound 的 naive 版本优化到 compute-bound 的 tiled 版本，算术强度从 12.8 提升到 46.5。

**技术深度问答：**
- Q: GEMM V3 (寄存器分块) 为什么提升最大？
  - A: 算术强度从 1 FLOP/2 reads 提升到 4 FLOPs/read，shared memory 带宽压力降低 1/4
- Q: 为什么 V6 (Tensor Core) 比 V5 慢？
  - A: FP32→FP16 转换开销 + tile size 限制 + occupancy 下降（50% → 30%）
- Q: Flash Attention 核心思想？
  - A: online softmax + tiling，O(N) 内存，算术强度从 12.8 提升到 46.5
- Q: 为什么 LLM decode 是 memory-bound？
  - A: batch=1 时 GEMM 退化为 GEMV，算术强度 ~2，远低于 Tensor Core 的计算密度
- Q: nano-vllm 的 paged attention 怎么实现的？
  - A: FlashDecoding 风格两遍并行，每个 KV block 一个 thread block，logsumexp 归约
