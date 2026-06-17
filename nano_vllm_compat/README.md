# nano-vllm sm_75 兼容层 + 四项 CUDA 优化

基于 nano-vllm 的 sm_75 (RTX 2080 Ti) 兼容层和四项面试级 CUDA 优化。

## 环境

- GPU: RTX 2080 Ti (sm_75 Turing, 22GB)
- CUDA 12.8, PyTorch 2.5.1, Triton 3.1.0
- 模型: Qwen3-0.6B (主), Qwen3-4B (用于 speculative decoding)

## 架构概览

```
┌──────────────────────────────────────────────────────────┐
│  nano-vllm (sm_75 compatible)                            │
├──────────────┬───────────────────────────────────────────┤
│  Prefill     │  flash_attn_varlen_func → PyTorch SDPA    │
│  Decode      │  flash_attn_with_kvcache → 两种后端:       │
│              │    1. CUDA Paged Attention (FlashDecoding) │
│              │    2. SDPA fallback (per-sequence loop)    │
│  KV Store    │  Triton kernel → CUDA float4 vectorized   │
│  Linear      │  FP16 matmul → INT8 dequant + GEMM fused  │
│  Engine      │  Autoregressive → Speculative Decoding    │
└──────────────┴───────────────────────────────────────────┘
```

## 四项优化

### A. Triton → CUDA KV Cache Store Kernel
- **文件**: `kv_store_cuda.cu`, `kv_store_cuda_ext.cpp`, `kv_store.py`
- **原理**: float4 向量化加载/存储, 每个线程处理 4 个连续元素
- **Grid**: `(tokens, D/4)` 2D grid, 支持 FP16/FP32
- **预期加速**: 10-20% decode 加速 (KV 写入是 decode 瓶颈之一)

### B. Paged Attention CUDA Kernel (核心亮点)
- **文件**: `paged_attention.cu`, `paged_attention_ext.cpp`, `paged_attn.py`
- **原理**: FlashDecoding 风格两遍算法
  - Pass 1: 每个 KV cache page 一个 thread block, 计算 partial attention + logsumexp
  - Pass 2: 跨 pages 归约, 用 logsumexp 加权合并
- **特性**: 支持 GQA, online softmax, warp-level reduction
- **预期加速**: 替代不支持的 flash_attn, sm_75 原生 paged attention

### C. INT8 Weight-Only 量化
- **文件**: `int8_linear.py`, `int8_quantize.py`, `model_runner_int8.py`
- **原理**:
  - 加载时: FP16 → INT8 symmetric per-row 量化 (scale = max_abs/127)
  - 推理时: Triton fused INT8 dequant + GEMM kernel
- **特性**: 显存减半, decode (memory-bound) 提速 ~2x
- **激活**: `NANOVLLM_INT8=1` 环境变量

### D. Speculative Decoding
- **文件**: `speculative.py`
- **原理**:
  - Draft: 小模型自回归生成 K 个候选 token
  - Verify: 大模型一次性前向 K+1 个 token (prefill-like)
  - Accept/Reject: 贪心匹配 (target argmax == draft token)
- **特性**: 单请求延迟降低 2-3x (大模型+小模型组合时)

## 快速开始

```bash
# 1. 克隆 nano-vllm
cd /root/cuda-lab
git clone https://github.com/GeeeekExplorer/nano-vllm

# 2. 下载模型
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
    Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

# 3. 安装兼容层
bash nano_vllm_compat/setup_sm75.sh

# 4. 运行测试
python3 nano_vllm_compat/test_sm75.py

# 5. 运行各优化测试
python3 nano_vllm_compat/test_paged_attn.py    # B: Paged Attention 正确性+性能
NANOVLLM_INT8=1 python3 nano_vllm_compat/test_int8.py  # C: INT8 量化
python3 nano_vllm_compat/test_speculative.py   # D: Speculative Decoding
```

## 文件清单

| 文件 | 用途 | 行数 |
|------|------|------|
| `attention_sm75.py` | 替换 `nanovllm/layers/attention.py` | 121 |
| `flash_attn_compat.py` | SDPA-based flash_attn 兼容层 | 207 |
| `kv_store_cuda.cu` | CUDA float4 KV store kernel | 169 |
| `kv_store_cuda_ext.cpp` | PyBind11 wrapper | 95 |
| `kv_store.py` | JIT load + Triton fallback | 113 |
| `paged_attention.cu` | FlashDecoding paged attention | 279 |
| `paged_attention_ext.cpp` | PyBind11 wrapper | 102 |
| `paged_attn.py` | Python API | 112 |
| `int8_linear.py` | Triton INT8 dequant+GEMM kernel | 152 |
| `int8_quantize.py` | 模型 INT8 量化 patcher | 91 |
| `model_runner_int8.py` | ModelRunner INT8 补丁 | 61 |
| `speculative.py` | 投机解码引擎 | 316 |
| `setup_sm75.sh` | 一键安装脚本 | ~50 |
| **Total** | | **~2270** |

## 面试亮点

1. **解决了 sm_75 不支持 flash_attn 的问题** — 不是简单 bypass, 而是自己实现了等价的 paged attention
2. **FlashDecoding 算法** — 跨 KV blocks 并行 + logsumexp 归约, 展示了 attention 的深入理解
3. **INT8 量化端到端** — 从量化理论到 Triton fused kernel, 展示全栈能力
4. **Speculative Decoding** — 系统设计能力, 理解 LLM 推理的系统级优化
5. **PyTorch C++ Extension** — 展示了 CUDA 工程化能力 (JIT compile, pybind11, stream handling)
