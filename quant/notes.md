# 量化 GEMM Kernel 知识点

## 为什么需要量化？

### LLM 推理的核心瓶颈：Memory Bandwidth
LLM 推理（尤其是 decode 阶段，batch=1）是 **memory-bound**：
- 每次生成一个 token，需要把所有权重从 HBM 读一遍
- 计算量极小（矩阵向量乘），但数据搬运量巨大
- 例：7B 模型 FP16 权重 = 14GB，每生成一个 token 需读 14GB 数据

### 量化的收益
| 精度 | 每元素字节 | 模型大小(7B) | 带宽需求 |
|------|-----------|-------------|---------|
| FP16 | 2B | 14GB | 100% |
| INT8 | 1B | 7GB | 50% |
| INT4 | 0.5B | 3.5GB | 25% |

**核心思想**：用更少的字节存权重 → 读 HBM 更快 → 推理更快

---

## Weight-Only 量化 (W8A16 / W4A16)

### 架构
```
Activations (FP16) × Weights (INT8/INT4) → Output (FP16)
                      ↑
                    dequant on-the-fly
```
- **Weight-only**：只量化权重，激活保持 FP16
- **为什么可以？** 权重的分布比较集中（接近高斯），量化误差小
- **激活不量化**：因为激活的 outlier 很多（如 LLM.int8() 论文指出的），量化会导致严重误差

### 量化公式
```
量化：  q = round(x / scale)         # FP → INT
反量化：x_approx = q * scale          # INT → FP
scale = max(|x|) / max_int            # per-group 共享一个 scale
```

---

## Group-wise 量化

### 为什么不用 per-tensor？
整个矩阵共享一个 scale → 如果矩阵中有 outlier，scale 会被拉大 → 其他元素的量化误差变大。

### Group-wise 方案
沿 K 维度分组，每 GROUP_SIZE=128 个元素共享一个 scale：
```
B[K, N] 矩阵:
  group 0: B[0:128, :]   → scales[0, :]   (N 个 scale，每列一个)
  group 1: B[128:256, :] → scales[1, :]
  ...
```
- 每列独立有自己的 scale → 处理权重分布不均匀的情况
- Scale 开销：K/128 × N × 2 bytes（FP16）→ 对于 1024×1024，8 × 1024 × 2 = 16KB，可忽略

---

## INT8 Dequant Kernel 实现

### 核心流程
```
1. 从 HBM 加载 A tile (FP16) → shared memory sA
2. 从 HBM 加载 B tile (INT8) → 在 shared memory 中 dequant 成 FP16 → sB
3. WMMA 从 sA、sB 加载 → Tensor Core 做 FP16 MMA
```

### 关键代码
```c
// 在 shared memory 加载阶段做 dequant（融合访存与计算）
int8_t val = B_int8[gR * N + gC];      // 从 HBM 读 1 byte
half scale = B_scales[grp * N + gC];    // 从 HBM 读 2 bytes
sB[r][c] = __int2half_rn((int)val) * scale;  // dequant → FP16
```

### 为什么这样设计？
- HBM 带宽节省：INT8 只需读 1 byte/元素（FP16 需要 2 byte）
- Dequant 在 shared memory 中进行，不占用 HBM 带宽
- 代价：额外的 FP16 乘法（scale）和类型转换开销

---

## INT4 打包格式

### 存储方案
4-bit 无法直接寻址，必须打包：2 个 INT4 值塞进 1 个 byte
```c
// 打包格式
byte = (val_high << 4) | val_low   // low nibble + high nibble

// 解包
int low  = byte & 0xF;           // 低4位
int high = (byte >> 4) & 0xF;   // 高4位

// 符号扩展（signed 4-bit: [-8, 7]）
if (low >= 8) low -= 16;
if (high >= 8) high -= 16;
```

### 为什么 INT4 更慢？
1. **解包开销**：位操作 + 符号扩展
2. **量化误差大**：只有 16 个离散值 → 精度损失大
3. **Dequant 开销**：和 INT8 相同的 scale 乘法

---

## Benchmark 结果分析

| Kernel | Time(ms) | GFLOPS | vs cuBLAS | MaxErr | Weight Size |
|--------|----------|--------|-----------|--------|-------------|
| cuBLAS_FP16 | 0.039 | 54.6 | 1.00x | - | 2.1MB |
| WMMA_FP16 | 0.268 | 8.0 | 0.15x | 0.22 | 2.1MB |
| INT8_dequant | 0.376 | 5.7 | 0.10x | 0.27 | 1.0MB |
| INT4_dequant | 0.478 | 4.5 | 0.08x | 3.92 | 0.5MB |

### 为什么量化版本反而更慢？

**关键洞察**：1024×1024 矩阵太小，不是 memory-bound！

```
对于 M=N=K=1024:
- 计算量: 2 × 1024³ = 2.1 GFLOP
- 数据量: A + B + C = 2+2+2 = 6MB (FP16)
- 算术强度: 2.1G / 6M = 350 FLOPs/byte
- RTX 2080 Ti: 计算带宽比 ≈ 13.4 TFLOPS / 600 GB/s = 22 FLOPs/byte
- 350 >> 22 → 这是 compute-bound！不是 memory-bound！
```

**量化的收益只在 memory-bound 场景**：
- LLM decode（M=1, batch=1）→ 算术强度 ~2 FLOPs/byte → memory-bound
- 此时 INT8 可以提速 2x，INT4 可以提速 4x

### INT4 误差为什么大？
- INT4 只有 [-8, 7] 共 16 个值
- 对于随机初始化的权重，量化误差较大
- 实际 LLM 使用 AWQ/GPTQ 等算法优化量化点选择，误差更小

---

## 真正的生产级量化 GEMM (GPTQ/AWQ)

### GPTQ 的优化
1. **Column-wise 量化**：每列独立量化
2. **Hessian-based 补偿**：用二阶信息补偿量化误差
3. **Split-K 并行**：K 维度切分，多 SM 并行

### AWQ 的优化
1. **Activation-aware**：根据激活的 salient channel 保护重要权重
2. **Per-channel scaling**：每列独立的 scale
3. **Kernel fusion**：dequant + matmul + residual 全融合

### CUTLASS 的量化支持
- `cutlass::epilogue::thread::LinearCombination` 支持 scale/zero-point
- 提供 INT4/INT8 weight-only 的 optimized kernel
- 支持 group-wise 和 per-channel 量化

---

## 面试常问点

1. **LLM 推理为什么是 memory-bound？**
   - Decode 阶段 batch=1，计算是矩阵向量乘，算术强度 ~2 FLOPs/byte
   - 远低于 GPU 的计算/带宽比（~20+），所以瓶颈在带宽

2. **Weight-only vs Weight-Activation 量化？**
   - Weight-only (W8A16)：只量化权重，激活 FP16，简单稳定
   - W8A8：两者都量化，需要特殊硬件（如 INT8 Tensor Core），更复杂

3. **Per-tensor vs Per-group vs Per-channel？**
   - Per-tensor：全矩阵一个 scale，精度最差
   - Per-group：每 128 个元素一个 scale，平衡精度和开销
   - Per-channel：每列一个 scale，精度最好但 scale 开销大

4. **INT4 量化误差大怎么处理？**
   - 更好的量化算法：GPTQ（二阶补偿）、AWQ（保护重要通道）
   - 混合精度：salient channel 保持 FP16，其他量化
   - Calibration：用少量数据优化量化参数

5. **量化 kernel 在小矩阵上为什么没提速？**
   - 小矩阵是 compute-bound，dequant 的额外计算反而拖慢
   - 只有大模型 + decode（memory-bound）才能看到量化收益
