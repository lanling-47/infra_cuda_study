# Transformer 推理引擎知识点

## 项目概述
实现轻量级 Transformer 推理引擎的 GPU Backend，包含核心算子：
- LayerNorm（单遍 Welford 算法）
- Softmax（Online Softmax）
- RoPE（旋转位置编码）
- Flash Attention（O(N) 内存的注意力机制）
- Element-wise 算子（GeLU, ReLU, SwiGLU, 残差连接）

## LayerNorm

### 算法：Welford's Online Algorithm
单次遍历计算均值和方差，数值稳定性更好：
```
mean = sum(x) / N
variance = sum((x - mean)^2) / N
```

### GPU 优化
- **Warp-level reduction**: 使用 `__shfl_xor_sync` 在 warp 内归约
- **Block-level reduction**: 使用 shared memory 在 block 内归约
- **融合计算**: mean/var/normalize/scale/shift 在一次 kernel 中完成

### 面试要点
- 为什么用 Welford？数值稳定性，避免大数减小数导致的精度损失
- Warp reduction 怎么实现？`__shfl_xor_sync` 蝴蝶归约
- Block reduction 为什么需要 shared memory？跨 warp 通信

## Softmax

### Online Softmax 算法
单次遍历同时计算 max 和 sum(exp)，避免两次遍历：
```
for each element x:
    new_max = max(old_max, x)
    sum = sum * exp(old_max - new_max) + exp(x - new_max)
    old_max = new_max
```

### 数值稳定性
减去 max 避免 exp 溢出：`exp(x - max)` 而非 `exp(x)`

### GPU 优化
- 第一遍：并行计算每个元素的 exp(x - max)
- Warp/Block reduction：归约 max 和 sum
- 第二遍：归一化 `y[i] = exp(x[i] - max) / sum`

### 面试要点
- 为什么要减 max？防止 exp 溢出
- Online Softmax vs 标准 Softmax？单次遍历，更适合长序列
- Flash Attention 如何使用 Online Softmax？分块计算 attention，逐步更新 max/sum

## RoPE (旋转位置编码)

### 原理
将位置信息编码为旋转矩阵，应用到 Q/K 向量：
```
θ = pos / (base^(2i/d))
x_rot[2i]   = x[2i]   * cos(θ) - x[2i+1] * sin(θ)
x_rot[2i+1] = x[2i+1] * cos(θ) + x[2i]   * sin(θ)
```

### 优点
- 相对位置编码：attention 分数自然包含相对位置信息
- 外推性：可以处理比训练时更长的序列

### GPU 实现
每个 block 处理一个 token 的所有 head，每个线程处理一部分维度对。

### 面试要点
- RoPE vs Sinusoidal PE？RoPE 是相对位置，Sinusoidal 是绝对位置
- 为什么 RoPE 能外推？旋转角度是连续的，可以计算任意位置的 cos/sin
- LLaMA/Mistral 为什么用 RoPE？相对位置编码更适合长文本

## Flash Attention

### 核心思想
标准 Attention 需要 O(N²) 内存存储 attention 矩阵。
Flash Attention 分块计算，使用 Online Softmax 逐步更新，只需 O(N) 内存。

### 算法
```
for each Q tile (block of queries):
    m = -inf, l = 0, O = 0
    for each K/V tile:
        S = Q_tile @ K_tile^T / sqrt(d)
        new_m = max(m, max(S))
        P = exp(S - new_m)
        l = l * exp(m - new_m) + sum(P)
        O = O * exp(m - new_m) + P @ V_tile
        m = new_m
    O = O / l
```

### GPU 优化
- **Shared memory**: 存储 Q/K/V tile，避免重复从 global memory 读取
- **Register accumulation**: O 累加器放在寄存器中
- **Causal mask**: 只计算下三角部分

### 面试要点
- Flash Attention 为什么快？减少 HBM 访问，利用 SRAM
- Online Softmax 在 Flash Attention 中的作用？分块计算时逐步更新 max/sum
- 内存复杂度？O(N) vs 标准 Attention 的 O(N²)
- 实际加速比？2-4x，取决于序列长度和硬件

## Element-wise 算子

### GeLU
```
GeLU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
```
近似版本：`GeLU(x) ≈ x * sigmoid(1.702 * x)`

### SwiGLU (LLaMA 使用)
```
SwiGLU(x, y) = Swish(x) * y
Swish(x) = x * sigmoid(x)
```

### 残差连接
```
y = x + residual
```
融合到 kernel 中避免额外的 memory access。

## Transformer Block 完整流程

```
Input x
├── LayerNorm
├── Multi-Head Attention
│   ├── QKV Projection (GEMM)
│   ├── RoPE on Q, K
│   ├── Flash Attention
│   └── Output Projection (GEMM)
├── Residual Add: x = x + attn_out
├── LayerNorm
├── FFN
│   ├── Up Projection (GEMM)
│   ├── GeLU/SwiGLU
│   └── Down Projection (GEMM)
└── Residual Add: x = x + ffn_out
```

## 性能 Benchmark

| Kernel | Time (ms) | Note |
|--------|-----------|------|
| LayerNorm | 0.006 | 512×256 |
| Softmax | 0.005 | 512×256 |
| RoPE | 0.011 | 4096×32 |
| GeLU | 0.003 | 512×256 |
| Flash Attention | 0.061 | 64×32, causal |
| Transformer Block | 0.093 | 64×256, end-to-end |

## 优化方向

1. **Kernel Fusion**: 将多个算子融合到一个 kernel 中
   - LayerNorm + QKV Projection
   - Attention + Output Projection
   - FFN 全部融合

2. **Quantization**: INT8/INT4 量化 GEMM
   - Weight-only 量化
   - Activation 量化需要更复杂的校准

3. **Speculative Decoding**: 用小模型预测，大模型验证
   - 减少大模型的推理次数
   - 适合 batch=1 的 decode 阶段

4. **PagedAttention (vLLM)**: 动态分配 KV cache
   - 避免内存碎片
   - 支持 continuous batching

## 面试高频问题

1. **Flash Attention 的原理？**
   - 分块计算 + Online Softmax，减少 HBM 访问

2. **LayerNorm 的 GPU 实现？**
   - Warp/Block reduction，融合计算

3. **RoPE vs 其他位置编码？**
   - 相对位置，外推性好，LLaMA 标配

4. **Transformer 推理的瓶颈？**
   - Decode 阶段是 memory-bound（batch=1）
   - Prefill 阶段是 compute-bound（长序列）

5. **如何优化 LLM 推理？**
   - 量化、KV cache、Flash Attention、Speculative Decoding
