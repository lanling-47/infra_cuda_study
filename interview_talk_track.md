# CUDA 项目面试话术

## 项目整体叙事 (30秒版本)

"我自主完成了 CUDA + GPU 推理的全栈能力建设，通过三个递进式项目系统性地掌握了 GPU 微架构和推理优化。

第一个项目是 GEMM kernel 分阶段优化，从 naive 实现开始，逐步引入 shared memory tiling、寄存器分块、向量化加载、双缓冲和 Tensor Core，最终手写版本达到 cuBLAS 64% 的性能。这个过程让我深入理解了 GPU 内存层次、warp 调度、bank conflict 等底层机制。

第二个项目是量化推理，实现了 INT8/INT4 weight-only GEMM。通过对比实验发现，1024x1024 矩阵是 compute-bound，量化反而更慢；只有在 LLM decode 这种 memory-bound 场景（batch=1）才能看到收益。这让我理解了量化优化的适用场景和 roofline 模型。

第三个项目是 Transformer 推理引擎，实现了 Flash Attention、RoPE、LayerNorm 等核心算子。特别是 Flash Attention 的优化，从 memory-bound 的 naive 版本优化到 compute-bound 的 tiled 版本，算术强度从 12.8 提升到 46.5。"

---

## 技术深度问答

### Q1: GEMM 优化中，为什么 V3 (寄存器分块) 提升最大？

**回答要点：**

"V3 的核心思想是提升算术强度（Arithmetic Intensity）。

在 V2 (shared memory tiling) 中，每个线程计算 1 个输出元素，需要从 shared memory 读取 K 次 A 和 K 次 B，算术强度是 1 FLOP / 2 reads。

V3 让每个线程计算 8x8 = 64 个输出元素，通过寄存器复用，只需要从 shared memory 读取 8+8=16 次，但执行 64 次乘加，算术强度提升到 4 FLOPs/read。

这样 shared memory 的带宽压力降低到原来的 1/4，所以性能从 V2 的 1.510ms 提升到 V3 的 0.457ms，提升了 3.3 倍。"

**关键指标：**
- V2: 1.510ms, 1.4 GFLOPS, 0.15x cuBLAS
- V3: 0.457ms, 4.7 GFLOPS, 0.50x cuBLAS

---

### Q2: 为什么 V6 (Tensor Core) 反而比 V5 慢？

**回答要点：**

"V6 使用 WMMA API 调用 Tensor Core，理论上应该更快，但实际 0.627ms 比 V5 的 0.353ms 慢了 78%。原因有三：

1. **精度损失**：WMMA 使用 FP16 输入，但 benchmark 的输入是 FP32，需要额外的 FP32→FP16 转换开销（MaxErr=0.0133 说明精度确实有损失）。

2. **Tile size 限制**：WMMA 要求 16x16x16 的 tile，但我们的实现没有充分优化 data loading 和 register allocation，导致 shared memory 带宽成为瓶颈。

3. **Occupancy 问题**：WMMA kernel 的寄存器使用较多，occupancy 从 V5 的 50% 下降到 30%，隐藏延迟的能力变差。

cuBLAS 的 Tensor Core 版本之所以快，是因为它使用了多级流水线、warp specialization、更好的 memory coalescing 等高级优化。"

**关键指标：**
- V5: 0.353ms, 6.1 GFLOPS, 0.64x cuBLAS
- V6: 0.627ms, 3.4 GFLOPS, 0.36x cuBLAS, MaxErr=0.0133

---

### Q3: 量化 GEMM 为什么在小矩阵上没有加速？

**回答要点：**

"这是 roofline 模型的典型应用。

对于 1024x1024 矩阵：
- 计算量：2*M*N*K = 2.1 GFLOPS
- 数据量：(M*K + K*N + M*N) * 2 bytes = 6 MB (FP16)
- 算术强度：2.1G / 6M = 350 FLOPS/byte

RTX 2080 Ti 的计算带宽比是 13.4 TFLOPS / 600 GB/s = 22 FLOPS/byte。

因为 350 >> 22，这是典型的 compute-bound，瓶颈在计算单元而不是内存带宽。INT8 量化虽然减少了内存读取，但引入了额外的 dequant 计算，所以反而更慢。

量化的真正收益场景是 LLM decode（batch=1, seq_len=1）：
- 计算量：2*1*N*d = 很小的 GEMV
- 数据量：需要读取整个权重矩阵
- 算术强度：~2 FLOPS/byte << 22

这是 memory-bound，INT8 可以提速 2x，INT4 可以提速 4x。"

**关键指标：**
- 1024x1024: AI=350, Compute Bound, INT8 反而慢 1.4x
- LLM decode: AI~2, Memory Bound, INT8 提速 2x

---

### Q4: Flash Attention 的核心优化思想是什么？

**回答要点：**

"Flash Attention 解决的是标准 Attention 的 O(N²) 内存问题。

标准 Attention 需要计算完整的 N×N attention matrix，然后做 softmax，最后乘以 V。对于长序列（比如 N=4096），attention matrix 需要 4096² * 4 bytes = 64 MB，远超 GPU 的 shared memory 容量。

Flash Attention 的核心思想是 **tiled computation + online softmax**：

1. **Tiled computation**：把 Q/K/V 分成小块（比如 64×64），每次只加载一个 tile 到 shared memory 计算，避免一次性计算整个 N×N matrix。

2. **Online softmax**：传统的 softmax 需要两遍遍历（第一遍找 max，第二遍计算 exp 和 sum）。Online softmax 可以在单次遍历中增量更新 max 和 sum，公式是：
   ```
   m_new = max(m_old, m_tile)
   l_new = l_old * exp(m_old - m_new) + l_tile * exp(m_tile - m_new)
   O_new = O_old * exp(m_old - m_new) + O_tile * exp(m_tile - m_new)
   ```

3. **Memory reduction**：从 O(N²) 降低到 O(N)，只需要存储 Q/K/V 和输出 O，不需要中间的 attention matrix。

我的实现中，tiled 版本的算术强度从 12.8 提升到 46.5，从 memory-bound 变成了 compute-bound，说明优化方向是正确的。"

**关键指标：**
- Naive: AI=12.8, Memory Bound, 0.063ms (64x32)
- Tiled: AI=46.55, Compute Bound, 1.408ms (512x64)

---

### Q5: 如何判断一个 kernel 是 memory-bound 还是 compute-bound？

**回答要点：**

"使用 roofline 模型分析：

1. **计算算术强度（Arithmetic Intensity, AI）**：
   AI = FLOPS / Bytes moved
   - FLOPS：kernel 的总计算量
   - Bytes moved：从 HBM 读取的数据量

2. **计算 Ridge Point**：
   Ridge Point = Peak FLOPS / Peak Bandwidth
   - RTX 2080 Ti: 13.4 TFLOPS / 600 GB/s = 22 FLOPS/byte

3. **判断瓶颈**：
   - AI < Ridge Point → Memory Bound（受限于内存带宽）
   - AI >= Ridge Point → Compute Bound（受限于计算单元）

4. **优化方向**：
   - Memory Bound：减少数据移动（量化、kernel fusion、更好的 memory access pattern）
   - Compute Bound：提升计算利用率（更好的 instruction scheduling、减少 warp divergence）

在我的 GEMM 项目中，1024x1024 的 AI=170，远大于 22，所以是 compute-bound，优化方向是提升计算单元的利用率（V3→V5）。"

---

## 项目亮点总结

### 技术深度
1. **GPU 微架构理解**：shared memory、warp scheduling、bank conflict、register pressure
2. **性能分析方法**：roofline 模型、nsight compute、arithmetic intensity
3. **推理优化技术**：量化、kernel fusion、Flash Attention、online softmax

### 工程能力
1. **系统设计**：从零搭建推理引擎，包含算子库、benchmark 框架、性能分析工具
2. **代码质量**：分阶段实现、清晰的注释、完整的测试
3. **持续优化**：每个版本都有明确的优化目标和性能对比

### 学习能力
1. **自主驱动**：没有导师指导，自主规划学习路线和项目
2. **快速迭代**：6 周完成 3 个项目，从 kernel 到系统
3. **理论结合实践**：每个优化都有理论依据和实验验证

---

## 常见追问准备

### Q: 为什么不用 CUTLASS 或 Triton？

"我的目标是深入理解 GPU 底层机制，而不是快速搭建一个可用的推理引擎。手写 kernel 让我理解了 shared memory tiling、warp scheduling、register allocation 等底层优化，这些是使用高级框架时无法体会的。

在实际工程中，我会优先使用 CUTLASS/Triton 这样的成熟框架，因为它们的性能更好、维护成本更低。但理解底层原理可以帮助我更好地调优和 debug。"

### Q: 性能差距还有多大？如何进一步优化？

"我的最佳 GEMM (V5) 达到 cuBLAS 的 64%，主要差距在于：

1. **Tensor Core 利用率**：cuBLAS 使用了 warp specialization 和多级流水线，我的 V6 没有充分优化
2. **Memory access pattern**：cuBLAS 使用了更激进的 memory coalescing 和 L2 cache 优化
3. **Instruction scheduling**：cuBLAS 使用了 PTX 级别的优化，我的 NVCC 编译可能没有充分优化

进一步优化方向：
- 实现 warp specialization（不同的 warp 负责不同的任务）
- 使用 cp.async 实现真正的异步加载
- 手动优化 PTX 指令
- 使用 CUTLASS 的 template 框架，在更高层次上优化"

### Q: 这个项目对秋招有什么帮助？

"这个项目展示了我的三个核心能力：

1. **GPU 编程能力**：能够从零实现高性能 kernel，理解底层优化原理
2. **系统思维**：能够设计完整的推理引擎，考虑性能、精度、可维护性
3. **学习能力**：能够在短时间内掌握新技术，并转化为工程实践

这些能力对于 AI Infra、推理优化、高性能计算等岗位都是直接相关的。"

---

## 项目 GitHub

https://github.com/lanling-47/infra_cuda_study

- **gemm/**: GEMM V1-V6 优化
- **quant/**: INT8/INT4 量化 GEMM
- **infer/**: Transformer 推理引擎（LayerNorm, Softmax, RoPE, Flash Attention）

每个项目都有完整的 README、notes.md（知识点整理）、benchmark 脚本。
