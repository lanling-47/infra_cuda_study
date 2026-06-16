# GEMM Kernel 优化知识点

## GPU 硬件基础

### 线程层次结构
```
Thread → Warp (32 threads) → Block → Grid
  线程：最小执行单元，拥有私有寄存器
  Warp：32个线程组成一个warp，SIMT同步执行
  Block：多个warp组成一个block，共享shared memory
  Grid：所有block的集合
```

### 内存层次（延迟递增，容量递增）
```
寄存器（~1 cycle）→ Shared Memory（~20-30 cycles）→ L1/L2 Cache → Global Memory (HBM, ~300-400 cycles)
```
- **寄存器**：每个SM有65536个（sm_75），线程私有
- **Shared Memory**：sm_75 每个SM最多64KB，block内线程共享
- **Bank Conflict**：shared memory分32个bank，若32个线程同时访问同一bank的不同地址，会串行化

---

## V1: Naive GEMM — 一个线程算一个输出元素

### 核心问题
每个线程独立计算C的一个元素，对A和B的每个元素重复从global memory读取。

```
对于 K=1024，每个线程读 2K 个float = 8KB from global memory
但相邻线程读的数据大量重叠 → 严重浪费带宽
```

### 知识点：Memory Coalescing
- GPU要求同一warp内的线程访问**连续内存地址**，才能合并为一次128字节事务
- V1的访问模式：`A[row * K + k]` 同行内连续，但 `B[k * N + col]` 列方向不连续 → B的访问无法coalesce

**性能**：0.09x cuBLAS — memory bound，带宽完全浪费在重复读取上。

---

## V2: Shared Memory Tiling — 用共享缓存减少global memory访问

### 核心思路
将A和B的tile加载到shared memory，block内所有线程共享这份数据，减少对global memory的重复访问。

```
原来：每个线程独立读 K 个 A 元素 + K 个 B 元素 → 2K 次 global memory 读
现在：整个block合作加载 tile，每个元素只从global memory读一次 → 复用 K/TILE_SIZE 次
```

### 知识点：Tiling 原理
```
for t in range(0, K, TILE_SIZE):
    load A[row_tile, t:t+TILE_SIZE] → shared memory
    load B[t:t+TILE_SIZE, col_tile] → shared memory
    __syncthreads()  ← 必须等所有线程加载完
    compute partial sum from shared memory
    __syncthreads()  ← 必须等所有线程算完再加载下一块
```

### 知识点：Bank Conflict 规避
```c
__shared__ float sA[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
```
- sA 是列连续的，若每行16列，线程0访问sA[0][0]，线程16访问sA[1][0] → 两者映射到同一bank
- 加1 padding 后，sA[1][0] 偏移到bank 17，避免conflict

**性能**：0.15x cuBLAS — 带宽利用率提升，但每个线程只算1个元素，计算/访存比仍低。

---

## V3: Register Tiling — 每个线程算多个输出元素

### 核心思路
V2中每个线程算1个输出，每次循环从shared memory读1次A和1次B，做1次乘加。
V3中每个线程算 TM×TN 个输出，从shared memory读TM次A和TN次B，但做 TM×TN 次乘加。

```
V2: 1次shared mem读 → 1次FMA  → 算术强度 = 1 FLOP / 2 reads
V3: (TM+TN)次shared mem读 → TM×TN次FMA → 算术强度 = TM×TN / (TM+TN) FLOPs / read
    当 TM=TN=8: 64/16 = 4 FLOPs/read → 4倍提升
```

### 知识点：寄存器压力
- sm_75：每SM 65536个寄存器，最多每线程255个
- TM=TN=8 时 acc[8][8]=64个寄存器 + aReg[8] + bReg[8] = 80个寄存器
- Block 256线程 × 80寄存器 = 20480 — 在限制内，但限制了occupancy
- **Occupancy** = 活跃warp / 最大warp — 寄存器越多，occupancy越低，隐藏延迟的能力越差

### 知识点：Block/Thread 配置
```c
// Block输出 = BM×BN = 128×128
// 每线程算 TM×TN = 8×8
// 线程数 = (BM/TM) × (BN/TN) = 16 × 16 = 256
dim3 block(16, 16);
dim3 grid((N + 127) / 128, (M + 127) / 128);
```

**性能**：0.50x cuBLAS — 寄存器复用是最大提升来源（V2→V3 提升 3.3x）。

---

## V4: Float4 向量化加载 — 提升 global memory 带宽利用率

### 核心思路
GPU的内存控制器按128字节（32个float）对齐处理事务。若线程访问对齐的16字节（4个float），可用float4指令一次加载。

```c
// 原来：每个线程加载1个float（4字节）
float val = A[row * K + col];

// V4：每个线程加载4个float（16字节）
float4 val = *reinterpret_cast<const float4*>(&A[row * K + col]);
// val.x, val.y, val.z, val.w
```

### 知识点：对齐要求
- float4 要求地址是16字节对齐
- 不对齐时用fallback标量加载，性能退化
- 矩阵按行存储，每行开头天然对齐（若K是4的倍数）

### 知识点：Memory Transaction 合并
```
32个线程各加载float4 → 32×16字节 = 512字节
GPU 合并为 4个128字节的内存事务（最优情况）
vs 32个float → 32×4字节 = 128字节 → 1个事务（也不错，但吞吐减半）
```

**性能**：0.58x cuBLAS — 从 V3 提升 16%，加载阶段带宽效率更高。

---

## V5: Double Buffering — 重叠数据加载与计算

### 核心思路
单buffer：`load_tile → compute → load_tile → compute`（串行）
双buffer：`load_tile_0 → [compute_0 + load_tile_1] → [compute_1 + load_tile_0] → ...`

```
单buffer：  |--load--|--compute--|--load--|--compute--|
双buffer：  |--load0--|--compute0+load1--|--compute1+load0--|
```

### 知识点：Shared Memory 限制
- sm_75 每 block 默认最多48KB shared memory
- V5 双buffer需要 2×(BM×BK + BK×BN)×sizeof(float)
- 解决方案：缩小tile到 BM=BN=64, BK=16 → ~17KB

### 知识点：真正的异步加载
- sm_80+（Ampere）支持 `cp.async` 指令，可以在计算时真正异步加载
- sm_75 没有 cp.async，double buffering 靠"计算当前tile的同时，其他线程加载下一tile"
- 实际效果取决于shared memory bandwidth vs compute 的平衡

**性能**：0.64x cuBLAS — 从 V4 提升 10%，部分重叠了访存与计算。

---

## V6: Tensor Core (WMMA) — 硬件矩阵乘法单元

### 核心思路
Turing（sm_75）内置 Tensor Core，可在一个时钟周期完成 16×16×16 的矩阵乘法（FP16输入，FP32累加）。

```c
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aFrag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bFrag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;

wmma::load_matrix_sync(aFrag, sA, stride);
wmma::load_matrix_sync(bFrag, sB, stride);
wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);  // 16×16×16 MMA
wmma::store_matrix_sync(output, cFrag, stride, wmma::mem_row_major);
```

### 知识点：Tensor Core 架构
```
sm_75 每SM 4个 Tensor Core
每个 Tensor Core 每周期：16×16×16 = 4096 次 FMA = 8192 FLOPs
vs CUDA Core 每周期：64 FMA = 128 FLOPs（sm_75 有64个 CUDA core/SM）
→ Tensor Core 吞吐量是 CUDA Core 的 ~64倍
```

### 知识点：V6 性能不如预期的原因
1. **FP32→FP16 转换开销**：输入是FP32，每次都要在kernel内转half
2. **Block太小**：只用2个warp（64线程），occupancy很低
3. **FP16 精度损失**：MaxErr=0.0133，精度低于FP32版本
4. **真正高性能的Tensor Core GEMM**（如cutlass/cuBLAS）用多级pipeline + cp.async + 更大的block

**性能**：0.36x cuBLAS — 精度换性能，适合量化推理场景。

---

## 总结：优化收益对比

| 版本 | 核心技术 | Time(ms) | GFLOPS | vs cuBLAS | 关键提升 |
|------|---------|----------|--------|-----------|---------|
| V1 | 一元素一线程 | 2.392 | 0.9 | 0.09x | baseline |
| V2 | Shared Memory Tiling | 1.510 | 1.4 | 0.15x | 减少global memory重复读 |
| V3 | 寄存器分块 (8×8) | 0.457 | 4.7 | 0.50x | **提升最大**，shared mem复用 |
| V4 | float4向量化加载 | 0.389 | 5.5 | 0.58x | 带宽效率提升 |
| V5 | 双缓冲 | 0.353 | 6.1 | 0.64x | 重叠访存与计算 |
| V6 | WMMA Tensor Core | 0.627 | 3.4 | 0.36x | 硬件MMA，精度有损 |
| cuBLAS | 工业级优化 | 0.227 | 9.5 | 1.00x | 参考基准 |

## 面试常问点

1. **为什么V3提升最大？** — shared memory读一次，寄存器复用TM×TN次，算术强度从1提升到4
2. **Bank Conflict是什么？** — 同warp内多个线程访问shared memory同一bank时串行化，用padding规避
3. **float4为什么能加速？** — 每线程16字节对齐加载，减少内存事务数
4. **Double buffering在sm_75上为何提升有限？** — 没有cp.async，不能真正异步，只是逻辑上的重叠
5. **Tensor Core为何不如CUDA Core快？** — 需要FP16输入、更大的tile/block、多级pipeline才能发挥全部性能
