# Transformer 推理引擎 (GPU Backend)

轻量级 Transformer 推理引擎，实现核心算子的 CUDA kernel。

## 核心算子

| 算子 | 算法 | 特点 |
|------|------|------|
| **LayerNorm** | Welford's online algorithm | 单次遍历，数值稳定 |
| **Softmax** | Online Softmax | 单次遍历，适合长序列 |
| **RoPE** | 旋转位置编码 | 相对位置，外推性好 |
| **Flash Attention** | 分块 + Online Softmax | O(N) 内存，减少 HBM 访问 |
| **GeLU/SwiGLU** | 激活函数 | LLaMA 使用 SwiGLU |

## 性能 (RTX 2080 Ti)

| Kernel | Time (ms) | Shape |
|--------|-----------|-------|
| LayerNorm | 0.006 | 512×256 |
| Softmax | 0.005 | 512×256 |
| RoPE | 0.011 | 4096×32 |
| GeLU | 0.003 | 512×256 |
| Flash Attention | 0.061 | 64×32, causal |
| **Transformer Block** | **0.093** | 64×256, end-to-end |

## 文件结构

```
src/
├── layernorm.cuh       # LayerNorm kernel
├── softmax.cuh         # Online Softmax kernel
├── rope.cuh            # RoPE kernel
├── flash_attention.cuh # Flash Attention kernel
├── elementwise.cuh     # GeLU, ReLU, SwiGLU, residual add
└── infer_bench.cu      # Benchmark + 端到端测试
```

## 构建

```bash
make       # 编译
make bench  # 运行 benchmark
```

## 技术要点

### LayerNorm
- Welford 算法：单次遍历计算 mean/variance
- Warp reduction：`__shfl_xor_sync` 蝴蝶归约
- Block reduction：shared memory 跨 warp 通信

### Flash Attention
- 分块计算：Q/K/V 分成 tile，避免 O(N²) 内存
- Online Softmax：逐步更新 max/sum，数值稳定
- Causal mask：只计算下三角

### RoPE
- 旋转矩阵：`x_rot = x * cos(θ) + rotate(x) * sin(θ)`
- 相对位置：attention 分数自然包含位置信息
- 外推性：可以处理比训练更长的序列

## 面试价值

1. **LayerNorm/Softmax**: 展示 reduction 优化技巧
2. **Flash Attention**: LLM 推理的核心优化
3. **RoPE**: 现代 LLM 的位置编码标准
4. **端到端**: 展示系统级理解能力

## 参考

- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [RoPE 论文](https://arxiv.org/abs/2104.09864)
- [LLaMA 架构](https://arxiv.org/abs/2302.13971)
