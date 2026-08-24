# AI Infra 学习路线 v2.0（2026 版）

> 在 AIInfraGuide 四大模块骨架之上，融合一线 AI Infra 工程师能力模型（OpenAI / Anthropic / NVIDIA / 字节大模型团队视角），补齐 **系统编程、网络、编译器、云原生平台、生产服务、Agent Infra、成本工程** 七个缺口。

---

## 0. 路线定位

### AI Infra 的本质

不是 CUDA + GPU，而是 **AI + HPC + 分布式系统 + 云基础设施** 的交叉领域：

```
              AI Application
                    │
        Agent / RAG / API Platform        ← 2026 新增层
                    │
                 Serving
        vLLM / TensorRT / Runtime
                    │
                Compiler
       Triton / MLIR / Graph Compiler
                    │
               Framework
       PyTorch / JAX / Runtime
                    │
          Distributed System
       NCCL / Megatron / FSDP
                    │
                Cluster
   Kubernetes / Scheduler / Storage
                    │
               Hardware
       GPU / Network / Memory
```

### 目标画像

| 目标 | 覆盖要求 | 本路线评级 |
|---|---|---|
| 普通 AI Infra 入门 | 四大模块即可 | ★★★★★ |
| 大模型推理工程师 | 模块 M1+M4+M5 | ★★★★★ |
| NVIDIA CUDA / 算子工程师 | 模块 M1 深度 | ★★★★★ |
| OpenAI / Anthropic / 顶级大模型团队 | 全栈 + 平台 + 系统深度 | ★★★★☆ |

---

## 1. 能力模型总览

达到顶级 AI Infra 工程师需要的能力清单与重要性：

| # | 能力 | 重要性 | 达到标准 |
|---|---|---|---|
| 1 | CUDA 编程 | ⭐⭐⭐⭐⭐ | 手写优化 GEMM，对标 cuBLAS 80% |
| 2 | GPU 架构 | ⭐⭐⭐⭐⭐ | 讲清 SM/Warp/Tensor Core/内存层次，能算 Roofline |
| 3 | 分布式训练 | ⭐⭐⭐⭐⭐ | 独立设计 70B 模型的 TP×PP×DP×CP 拓扑 |
| 4 | LLM Serving | ⭐⭐⭐⭐⭐ | 精读 vLLM 调度器，会做 $/token 成本核算 |
| 5 | C++ | ⭐⭐⭐⭐⭐ | 读得懂 NCCL/Megatron 核心代码，能写系统级模块 |
| 6 | Linux 系统编程 | ⭐⭐⭐⭐ | 懂 syscall/mmap/page cache/NUMA/scheduler |
| 7 | 编译器 | ⭐⭐⭐⭐ | 理解 Triton IR → LLVM → PTX 的完整链路 |
| 8 | 网络 | ⭐⭐⭐⭐ | 懂 RDMA/RoCE/拥塞控制，会算集合通信耗时 |
| 9 | Kubernetes | ⭐⭐⭐⭐ | 会写 GPU Device Plugin / Operator / CRD |
| 10 | 云架构 | ⭐⭐⭐⭐ | 能设计 GPU 集群的资源池化与调度方案 |
| 11 | Agent Runtime | ⭐⭐⭐ | 理解 workflow/state/memory/tool 执行架构 |
| 12 | 性能建模 | ⭐⭐⭐⭐ | 会算 6ND、MFU、通信 α-β 模型（原路线缺失） |
| 13 | 成本工程 | ⭐⭐⭐⭐ | 掌握 $/1M tokens、吞吐×延迟权衡（原路线缺失） |

---

## 2. 五阶段学习计划（总周期 24 个月）

> 阶段顺序经过重排：**先推理后训练**。原因：推理岗位需求大、上手快、能快速建立正反馈；训练 Infra 岗位少且集中在头部公司。

---

### Phase 0：GPU 软件工程师（0–3 个月）

**目标**：能写高效 CUDA kernel，具备系统编程底子

**学习清单**

- [ ] Linux 内核基础：syscall、mmap、page cache、NUMA、进程调度
- [ ] C++ 进阶：模板、智能指针、并发、内存模型
- [ ] CUDA 编程模型与内存模型
- [ ] GPU 架构：SM、Warp、Tensor Core、HBM、NVLink
- [ ] PyTorch 基础：Tensor、autograd、训练流程
- [ ] 网络入门：TCP/IP、Socket 编程（为 RDMA 打底）

**动手项目**

1. CUDA 向量加法 → GEMM（naive → tiling → shared memory）
2. LayerNorm、Softmax kernel
3. 用 PyTorch 手写 Transformer 训练一个小模型

**检验标准**：GEMM 达到 cuBLAS 50% 性能；能解释 GPU server 上 PCIe→GPU→HBM 各层延迟量级。

**必读**：《CSAPP》《OSTEP》CUDA Programming Guide、NVIDIA "How to Optimize GEMM" 博客。

---

### Phase 1：Kernel 工程师（3–6 个月）

**目标**：掌握工业级算子优化方法，进入推理 kernel 优化

**学习清单**

- [ ] Triton：block 级编程模型，手写 FlashAttention / LayerNorm / Softmax
- [ ] CUTLASS：GEMM template 设计、Tensor Core、WMMA（工业界事实标准）
- [ ] FlashAttention V1/V2/V3 论文 + CUDA 源码精读
- [ ] torch.compile：Dynamo → Inductor 链路
- [ ] Transformer Engine：FP8/FP4 精度机制（Hopper/Blackwell 必备）
- [ ] TensorRT：网络构建、层融合、量化工作流

**动手项目**

1. 优化 Llama 的 attention kernel（Triton → CUDA 两版）
2. 用 CUTLASS 拼一个端到端 GEMM 算子
3. 用 Nsight Compute 定位并消除 bank conflict / 低 occupancy

**检验标准**：自己写的 FlashAttention kernel 在 4090 上达到官方实现 80% 性能；能讲清 FP8 的缩放与回退机制。

**必读论文**：FlashAttention 1/2/3、CUTLASS 文档、FP8 Formats for Deep Learning。

---

### Phase 2：LLM Infra 工程师（6–12 个月）

**目标**：精通推理系统全链路，具备生产部署能力

**学习清单**

- [ ] LLM 推理基础：Prefill/Decode、KV Cache 显存账本、TTFT/TPOT
- [ ] PagedAttention、Continuous Batching、Prefix Cache、Chunked Prefill
- [ ] vLLM 源码精读：调度器、KV Cache 管理、V1 引擎
- [ ] SGLang、TensorRT-LLM 对比与选型
- [ ] 量化：SmoothQuant、GPTQ、AWQ、FP8、KV Cache 量化
- [ ] Speculative Decoding：Draft+Verify、Medusa、EAGLE
- [ ] PD 解耦：DistServe、Splitwise、KV Cache 传输
- [ ] **Serving 系统（原路线缺失）**：API Gateway（Envoy/Nginx/gRPC）、负载均衡、健康检查
- [ ] **成本工程（原路线缺失）**：$/1M tokens 核算、batching 策略、autoscaling、GPU 利用率优化

**动手项目**

1. 部署 Llama-3-8B → vLLM → K8s → Prometheus 监控的完整链路
2. 压测并画出吞吐-延迟曲线，算 $/token 成本
3. 给 vLLM 加一个 Prefix Cache 统计指标（读源码改代码）
4. 对比 FP16/INT8/INT4/FP8 的精度-吞吐-成本三维矩阵

**检验标准**：能回答"100 QPS 的 70B 服务需要多少张 A100、多少显存、月成本多少"；能画出 vLLM 调度器状态机。

**必读论文**：PagedAttention、Splitwise、SmoothQuant、AWQ、Speculative Decoding。

---

### Phase 3：Senior Infra——分布式训练（12–18 个月）

**目标**：掌握千卡/万卡训练技术，具备性能建模能力

**学习清单**

- [ ] 集合通信：Ring/Tree AllReduce、ReduceScatter、All-to-All、α-β 通信模型
- [ ] 数据并行：DDP → FSDP → ZeRO-1/2/3 的切分与通信量推导
- [ ] 张量并行：Megatron Column/Row Parallel Linear、通信插入位置
- [ ] 流水线并行：Bubble 分析、GPipe、1F1B、Interleaved
- [ ] 序列并行与上下文并行：Ring Attention、Ulysses
- [ ] MoE 并行：Expert Parallelism、All-to-All 代价、负载均衡
- [ ] 混合精度：BF16/FP8、Loss Scaling、Activation Checkpointing
- [ ] **性能建模（原路线缺失）**：
  - 训练 FLOPs：`6 × 参数量 × Tokens`
  - MFU：`实际 FLOPs / 理论峰值 FLOPs`
  - Roofline 判断瓶颈
  - Scaling Law（Chinchilla）估算数据与算力配比
  - 通信耗时公式：`α + β × message_size`，会手算 TP/PP 每步通信量

**动手项目**

1. 2 卡跑通 Megatron 风格 TP+PP 训练 GPT-2
2. 8 卡训练一个 1B–7B 模型，MFU 达到 45%+
3. 画出训练 timeline，量化 bubble 率，用 1F1B 消除
4. 手算并验证：70B 模型训练 1T tokens 需要多少 H100·天、多少网络带宽

**检验标准**：面试官给任意模型规模和卡数，能现场设计并行拓扑并估算 MFU 与训练时长。

**必读**：ZeRO 系列、Megatron-LM、Ring Attention、Ulysses、Chinchilla 论文。

---

### Phase 4：顶级 Infra 工程师（18–24 个月）

**目标**：补全平台层与前沿方向，形成全栈闭环

**学习清单**

- [ ] **编译器深入**：MLIR Dialect 设计、Triton IR → LLVM → PTX 链路、TVM/XLA 调度原语
- [ ] **集群与调度**：Kubernetes GPU Operator、Device Plugin、扩展调度器、Volcano、Slurm
- [ ] **分布式存储**：S3 对象存储、Lustre/并行文件系统、Ceph、Checkpoint 读写优化
- [ ] **网络深入**：RDMA/RoCE、InfiniBand、拥塞控制、TCP bypass、Fat-Tree/Dragonfly 拓扑
- [ ] **Agent Infra（2026 新增）**：
  - Agent Runtime：workflow、state、memory、tool execution
  - OpenAI Agents SDK、LangGraph、Temporal
  - LLM + Tools + Database + External API 的编排架构
- [ ] 分布式系统理论：MIT 6.824（Raft、容错）、《DDIA》

**里程碑项目：Mini OpenAI Platform**

```
      Training (1B 模型，4 卡训练)
            │
      Model Registry (版本管理)
            │
      Deployment (K8s GPU 池)
            │
      Serving (vLLM + API Gateway)
            │
      Monitoring (Prometheus + 成本看板)
            │
      Cost Control (autoscaling + 配额)
```

**检验标准**：能设计并实现一个带模型注册、自动扩缩容、成本核算的内部推理平台 MVP；能讲清 Agent Runtime 的 state 持久化与容错方案。

---

## 3. 知识模块地图（八大模块）

在原四大模块基础上增补为八大模块，与原路线的对照关系：

| 模块 | 状态 | 内容要点 |
|---|---|---|
| M0 系统与网络基础 | 🆕 增补 | Linux 内核、C++、网络/RDMA（原路线仅覆盖语言基础） |
| M1 CUDA 与算子 | 强化 | 原内容 + CUTLASS + Transformer Engine + FlashAttention 源码 |
| M2 框架与编译器 | 🆕 增补 | PyTorch 内部（autograd/dispatcher）、Triton/MLIR/torch.compile |
| M3 分布式训练 | 强化 | 原内容 + 性能建模（6ND/MFU/α-β/Scaling Law） |
| M4 推理优化与 Serving | 强化 | 原内容 + API Gateway + 成本工程 |
| M5 AI 平台工程 | 🆕 增补 | K8s GPU 平台、Volcano/Slurm、S3/Lustre/Ceph、可观测性 |
| M6 Agent Infra | 🆕 增补 | Agent Runtime、LangGraph、Temporal、工具编排 |
| M7 性能分析主线 | 保留 | Roofline、Nsight Systems/Compute、PyTorch Profiler |

### 各模块与阶段映射

| 阶段 | 覆盖模块 |
|---|---|
| Phase 0 | M0（基础）+ M1（入门） |
| Phase 1 | M1（深入）+ M2（编译器） |
| Phase 2 | M4 + M5（部署部分）+ M7 |
| Phase 3 | M3 全量 + M7 |
| Phase 4 | M5 全量 + M2（MLIR）+ M0（网络深入）+ M6 |

---

## 4. 源码阅读路线（按序精读，不要乱读）

| 顺序 | 项目 | 阅读重点 | 验收问题（自问） |
|---|---|---|---|
| 1 | PyTorch | Tensor → autograd → Dispatcher | 一次 `torch.add` 从 Python 到 kernel 经过哪些 dispatch key？ |
| 2 | Triton | Python DSL → IR → LLVM → PTX | 一个 Triton kernel 的 block 如何映射到 SM？ |
| 3 | vLLM | Scheduler → KV Cache → PagedAttention | 一个新请求如何被调度、KV block 如何分配？ |
| 4 | Megatron-LM | Tensor Parallel → Pipeline Parallel → 通信 | ColumnParallelLinear 的通信发生在 forward 还是 backward？ |
| 5 | NCCL | Ring/Tree 算法 → 拓扑感知 → 通道 | 8 卡 AllReduce 数据切分几块、几步完成？ |

**阅读方法**：每个项目先跑通最小 demo → 用 gdb/日志单步追一条数据流 → 画时序图 → 写笔记回答验收问题。

---

## 5. 岗位方向路径对照

| 岗位方向 | 核心模块组合 | 深度要求 | 目标梯队 |
|---|---|---|---|
| 算子/Kernel 工程师 | M1 + M2 | GEMM/Attention kernel 到极致 | T3 芯片厂、T0 高性能组 |
| 推理引擎工程师 | M1 + M4 | vLLM/TensorRT 源码级 | T0-T5 全梯队 |
| 训练框架工程师 | M3 + M1 | Megatron/FSDP 源码级 | T0 大模型团队（岗位少） |
| AI 平台工程师 | M5 + M4 | K8s + 调度 + 存储 | T0/T1 平台组 |
| 编译器工程师 | M2 深入 | MLIR/Triton/TVM | T3 芯片厂、T0 编译器组 |
| Agent Infra | M6 + M4 | Runtime + 编排 | T2 AI 独角兽（新兴） |

> 求职策略与 AIInfraGuide 面试宝典一致：**推理方向岗位最多，训练方向只在大厂**。但若目标顶级团队（OpenAI/Anthropic/NVIDIA），训练 + 平台 + 成本工程是差异化竞争力。

---

## 6. 全路线检查清单（Checklist）

**系统底子**
- [ ] 能解释 NUMA 对 GPU 训练的影响
- [ ] 能手写一个 mmap 文件加载器
- [ ] 懂 RDMA 与 TCP 的本质区别

**算子能力**
- [ ] GEMM 对标 cuBLAS 80%
- [ ] 手写 FlashAttention 达标
- [ ] 会用 CUTLASS 模板
- [ ] 懂 FP8/FP4 精度回退机制

**训练能力**
- [ ] 会算 6ND 与 MFU
- [ ] 会设计 TP×PP×DP×CP 拓扑
- [ ] 会算 α-β 通信耗时
- [ ] 8 卡训练 MFU 45%+

**推理能力**
- [ ] 会算 KV Cache 显存账本
- [ ] 精读 vLLM 调度器
- [ ] 会做量化三维对比（精度-吞吐-成本）
- [ ] 会算 $/1M tokens 与容量规划

**平台能力**
- [ ] 会写 GPU Device Plugin
- [ ] 懂 Volcano gang scheduling
- [ ] 会设计 Checkpoint 到 S3/Lustre 的存储方案
- [ ] 能搭带监控与成本核算的推理平台 MVP

**前沿（2026）**
- [ ] 懂 Agent Runtime 的 state/memory/tool 架构
- [ ] 能对比 LangGraph vs Temporal 的适用场景
- [ ] 跟踪 PD 解耦、推理时 Scaling 的工程影响

---

## 7. 资源清单

### 书籍
| 书 | 覆盖能力 |
|---|---|
| 《CSAPP》《OSTEP》 | Linux 系统、内存、并发 |
| 《DDIA》 | 分布式系统、存储 |
| 《Linux 高性能服务器编程》 | 网络编程（原路线缺失） |
| PMPP（Programming Massively Parallel Processors） | CUDA 入门 |
| CUDA C++ Programming Guide（官方） | CUDA 权威参考 |

### 课程
| 课程 | 覆盖能力 |
|---|---|
| MIT 6.824 | 分布式系统（原路线缺失） |
| MIT 6.S965 / CMU 15-418 | 并行计算 |
| CUDA MODE 社区讲座 | 算子优化实战 |
| MLIR 官方教程 | 编译器 |

### 论文（按阶段）
- **算子**：FlashAttention 1/2/3、FP8 Formats
- **训练**：ZeRO 系列、Megatron-LM、Ring Attention、Ulysses、Chinchilla
- **推理**：PagedAttention、Splitwise、SmoothQuant、AWQ、Speculative Decoding、Medusa
- **平台**：Ray、Volcano、Singularity（NVIDIA 推理服务）

### 跟踪渠道
arXiv（cs.DC/cs.LG）、SemiAnalysis、NVIDIA 官方博客、Meta/Google/字节/Anthropic Infra 博客。

---

## 8. v1.0 → v2.0 变化总结

| 维度 | v1.0（原 AIInfraGuide） | v2.0（本路线） |
|---|---|---|
| 定位 | CUDA → 训练 → 推理四大模块 | 八大模块全栈闭环 |
| 系统底子 | 语言基础 | + Linux 内核、网络/RDMA |
| 算子 | 经典算子 | + CUTLASS、Transformer Engine、FA 源码 |
| 训练 | 五大并行 | + 性能建模（6ND/MFU/α-β/Scaling Law） |
| 推理 | 引擎核心技术 | + Serving 网关、成本工程 |
| 平台 | 无 | + K8s/调度/存储/可观测性 |
| 前沿 | 无 | + Agent Infra、编译器深入 |
| 顺序 | 训练在前推理在后 | 推理在前训练在后（就业导向） |
| 周期 | 3–6 个月 | 24 个月五阶段（可按目标裁剪） |
