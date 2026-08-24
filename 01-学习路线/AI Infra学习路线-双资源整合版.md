# AI Infra 学习路线 v3.0（2026 · 双资源整合版）

> 在 v2.0（八大模块 + 五阶段）基础上，整合 ZOMI《AIInfra》课程（B站视频/PPT/Jupyter 代码），补齐 **大模型算法与数据、后训练与强化学习、Agent/RAG 应用** 三大块，形成「文章精读 + 视频搭框架 + 代码动手 + 面试检验」四轨学习法。

---

## 0. 双资源定位与使用原则

| 资源 | 形态 | 强项 | 弱项 |
|---|---|---|---|
| **AIInfraGuide**（caomaolufei） | Markdown 文章 + 面试宝典 | CUDA/算子/训练/推理的**原理深度**；180+ 面试题可自测 | 集群/算法/应用层内容少 |
| **ZOMI AIInfra**（chenzomi12） | B站视频 + PPT + Jupyter CODE | 集群/通信存储/云原生/**算法数据/后训练 RL/应用**的**广度**与代码实践 | 算子细节不如前者深 |

### 四轨学习法（每个主题的固定套路）

```
① 视频搭框架   → ZOMI 课程 1.25~1.5 倍速看，建立全局认知（2h/主题）
② 文章精原理   → AIInfraGuide 对应文章精读，搞懂数学推导与设计动机
③ CODE 动手    → ZOMI Jupyter 必做清单逐行跑通，改参数观察现象
④ 面试检验     → 对照 AIInfraGuide 面试宝典对应梯队真题自测
```

> 使用提示：ZOMI 仓库已超 10G（含全部制作过程和高清图），**不要 git clone，到 Releases 页按需下载模块**；视频托管于 B站「ZOMI酱」/ 油管「ZOMI6222」。

---

## 1. 十大模块知识地图（v2.0 八模块 → v3.0 十模块）

| 模块 | 状态 | 内容要点 | 主要资源 |
|---|---|---|---|
| M0 系统与网络基础 | 保留 | Linux 内核、C++、网络/RDMA | AIInfraGuide + ZOMI 02 |
| M1 CUDA 与算子 | 保留 | CUDA、GEMM/Attention、CUTLASS、Transformer Engine | AIInfraGuide 为主 |
| M2 框架与编译器 | 保留 | PyTorch 内部、Triton/MLIR/torch.compile | 双库 |
| M3 分布式训练 | 保留 | 五大并行 + 性能建模（6ND/MFU/α-β） | 双库强互补 |
| M4 后训练与强化学习 | 🆕 新增 | SFT、InstructGPT、DPO/PPO/GRPO、评估 | ZOMI 4.4/4.5/4.6 |
| M5 推理优化与 Serving | 保留 | vLLM、量化、投机解码、PD 解耦、成本工程 | 双库强互补 |
| M6 AI 平台工程 | 保留 | 集群基建、K8s/云原生、存储/Checkpoint、调度 | ZOMI 01/02/03 |
| M7 大模型算法与数据 | 🆕 新增 | Transformer 深挖、MoE 论文线、Tokenizer、多模态、数据工程 | ZOMI 06 |
| M8 Agent 与 RAG 应用 | 🆕 新增 | Agent Runtime、RAG/GraphRAG、应用热点 | ZOMI 07 |
| M9 性能分析主线 | 保留 | Roofline、Nsight、Profiler、MFU 评估 | 双库 |

---

## 2. 模块双资源对照表（核心章节）

> 每个主题 = 「AIInfraGuide 文章」+「ZOMI 编号」+「必做 CODE」，三者打勾才算学完。

### M3 分布式训练

| 主题 | AIInfraGuide 文章 | ZOMI 编号 | 必做 CODE |
|---|---|---|---|
| 并行总论/显存账本 | 分布式训练总论 1.1/1.2 | 4.1 分布式并行框架、DeepSpeed 介绍 | CODE 01: 从零构建 PyTorch DDP |
| 优化器 | 3.1 优化器演进 | 4.1（结合 4.2 ZeRO） | CODE 01: ZeRO 显存优化实践 |
| ZeRO 系列 | 4.1/4.2 数据并行详解 | 4.2 优化器并行 ZeRO1/2/3 原理 | CODE 01: ZeRO 显存优化实践 |
| 张量并行 | 模块三 TP 章节 | 4.2 Megatron-LM TP 原理 + 代码解析 | CODE 02: Megatron 张量并行复现 |
| 流水线并行 | 模块三 PP 章节 | 4.2 Megatron PP + 1F1B/Interleaved | CODE 03: Pipeline 并行实践 |
| 序列/上下文并行 | Ring Attention、Ulysses | 4.3 Ring Attention | CODE 04: Ring Attention 实践 |
| MoE 并行 | 模块三 MoE 章节 | 4.2 专家并行大规模训练 | CODE 04 + 06 MoE 分布式性能分析 |
| 训练加速 | FlashAttention/梯度检查点 | 4.3 大模型训练加速 | CODE 01/02: FA + 梯度检查点 |
| FP8 混合精度 | 模块三混合精度章节 | 4.3 FP8 混合精度训练 | CODE 03: FP8 混合精度训练 |
| 性能建模 | MFU/Roofline（v2.0 增补） | 1.4 集群性能分析 | CODE 03: MFU 模型利用率评估 |

### M4 后训练与强化学习（v3.0 新增）

| 主题 | 说明 | 必做 CODE |
|---|---|---|
| SFT 微调 | 算法原理、变体（LoRA/QLoRA）、多模态实践 | CODE 01: Qwen3-4B 模型微调；CODE 02: LoRA 微调 SD |
| RLHF 链路 | InstructGPT 全流程：SFT → RM → PPO | CODE 01: 经典 InstructGPT 复现 |
| PPO vs DPO | 两种对齐算法对比与工程差异 | CODE 02: DPO 与 PPO 在 LLM 对比 |
| GRPO（2026 热门） | 无 critic 的 RL 训练，成本更低 | CODE 03: LLM + GRPO 实践 |
| 模型评估 | 评估基准与统一框架 | CODE 01: OpenCompass 评估实践 |

> 背景：2026 年 RL 后训练是 Infra 岗位新增长点（RL 训练需要额外的 rollout 集群、reward 服务、KV 缓存复用，都是实打实的系统问题）。

### M5 推理优化

| 主题 | AIInfraGuide 文章 | ZOMI 编号 |
|---|---|---|
| 推理基础/指标 | 1.1 LLM 推理基础 | 5.1 基本概念（流程、框架对比、性能指标） |
| KV Cache/算子加速 | 2.1 PagedAttention | 5.2 KV 缓存优化、算子改进与高效引擎 |
| 调度/批处理 | 2.2~2.5 Continuous Batching、Chunked Prefill | 5.3 缓存优化、批处理与分布式调度 |
| 长序列推理 | — | 5.4 长序列推理算法、并行策略 |
| 采样解码 | — | 5.5 采样方法、加速策略、MoE 推理优化 |
| 量化/压缩 | 模块四量化章节 | 5.6 低精度量化、知识蒸馏 |

### M6 AI 平台工程

| 主题 | ZOMI 编号 | 说明 |
|---|---|---|
| 集群基建 | 01-1/2/3 | L0/L1 基建、万卡集群方案、散热与机房 |
| 集群性能分析 | 01-4 | 性能指标、建模与问题定位 |
| 组网与高速互联 | 02-1/2 | 组网架构、拓扑算法、拥塞控制 |
| 集合通信 | 02-3/4 | 通信域/原语/算法；MPI、NCCL、HCCL |
| 存储 | 02-5 | 数据存储、CheckPoint 梯度检查点 |
| 容器与云原生 | 03-1/2/3 | Docker、K8S 核心机制、AI 部署实践 |
| AI 云平台 | 03-4 | 平台演进、持续交付、智能化运维 |

### M7 大模型算法与数据（v3.0 新增）

| 主题 | ZOMI 编号 | 必做 CODE |
|---|---|---|
| Transformer 深挖 | 06-01~07（结构/Tokenizer/Embedding/Attention 变种/长序列/参数设置） | 01~04: 迷你 Transformer、从零训练、BPE、Sinusoidal；07: MHA/MQA/GQA/MLA |
| MoE 论文线 | 06 MoE 章节 01~13（GShard→Switch→GLaM/ST-MoE→DeepSeek-MoE） | 01~04: HF 推理、手撕 MoE、分布式实现、性能分析 |
| 创新架构 | SSM、Mamba、RWKV、Linear Transformer | — |
| 多模态 | 图文/视频/语音大模型 | — |
| 数据工程 | 数据质量、Prompt Engine | — |

### M8 Agent 与 RAG（v3.0 新增）

| 主题 | ZOMI 编号 | 说明 |
|---|---|---|
| Agent 概念与核心 | 07-01/02 | 智能体原理、架构、核心机制 |
| RAG 与 GraphRAG | 07-03 | 检索增强生成、知识增强落地 |
| 热点（选学） | 07-00/04~08 | 自动驾驶、具身智能、生成推荐、AI 安全、AI 十年史 |

---

## 3. 五阶段学习计划 v3.0（24 个月）

> 顺序沿用 v2.0：**先推理后训练**。每阶段标注双资源学习路径。

### Phase 0：GPU 软件工程师（0–3 个月）

**学习路径**

- 视频：ZOMI 06 Transformer 架构 01~07（搭框架）
- 文章：AIInfraGuide 模块一（编程/数学/PyTorch/GPU/集合通信）
- 系统：Linux 内核基础（syscall/mmap/NUMA）、C++、网络入门

**动手项目**

1. ZOMI CODE：搭建迷你 Transformer、从零实现 Transformer 训练
2. CUDA：向量加法 → GEMM → LayerNorm → Softmax
3. AIInfraGuide 4.1 手写优化版 GEMM 起步

**检验**：能解释 GPU server 全链路延迟量级；迷你 Transformer 训练 loss 正常下降。

### Phase 1：Kernel 工程师（3–6 个月）

**学习路径**

- 视频：ZOMI 4.3 训练加速（FA/梯度检查点/FP8/Ring Attention 原理）
- 文章：AIInfraGuide 模块二全量 + CUTLASS/Transformer Engine（v2.0 增补）
- 编译器：Triton、torch.compile

**动手项目**

1. ZOMI CODE：FlashAttention 实现（4.3-CODE 01）
2. 用 Triton 重写 Llama attention kernel，对照 FA2 性能
3. Nsight Compute 定位 bank conflict / occupancy 问题

**检验**：自写 FA kernel 达官方 80% 性能；能讲清 FP8 缩放与回退。

### Phase 2：LLM Infra 工程师（6–12 个月）

**学习路径**

- 视频：ZOMI 05 分布式推理全模块（5.1~5.6）
- 文章：AIInfraGuide 模块四 + v2.0 增补（Serving 网关、成本工程）
- 源码：vLLM 精读（scheduler → KV cache → PagedAttention）

**动手项目**

1. ZOMI 5.x 配套实践 + 部署 Llama → vLLM → K8s → Prometheus
2. 压测画吞吐-延迟曲线，算 $/1M tokens
3. 量化三维对比（FP16/INT8/INT4/FP8）

**检验**：能回答"100 QPS 的 70B 服务需要多少卡、多少显存、月成本"；能画 vLLM 调度器状态机。

### Phase 3：Senior Infra——训练 + 后训练（12–18 个月）

**学习路径**

- 视频：ZOMI 04 分布式训练全模块（4.1~4.6）
- 文章：AIInfraGuide 模块三 + 性能建模（6ND/MFU/α-β/Scaling Law）
- 后训练：ZOMI 4.4 RL + 4.5 SFT + 4.6 评估
- 源码：Megatron-LM → NCCL

**动手项目**

1. ZOMI CODE 全做：从零 DDP → Megatron TP 复现 → Pipeline → ZeRO → 专家并行
2. 8 卡训练 1B–7B 模型，MFU 45%+
3. InstructGPT 复现 / GRPO 实践（对齐 2026 岗位热点）
4. 手算验证：70B 训练 1T tokens 需要多少 H100·天、多少带宽

**检验**：任意模型规模能现场设计并行拓扑并估算 MFU/时长；能讲清 PPO 与 GRPO 的系统开销差异。

### Phase 4：顶级 Infra 工程师（18–24 个月）

**学习路径**

- 视频：ZOMI 01 集群 + 02 通信存储 + 03 云原生 + 07 应用
- 文章：v2.0 增补（编译器深入、分布式系统理论 MIT 6.824）
- 前沿：Agent Infra（LangGraph/Temporal）、MoE 架构深挖（ZOMI 06）

**里程碑项目：Mini OpenAI Platform**

```
Training(1B, 4卡) → Registry → Deployment(K8s GPU池)
→ Serving(vLLM + API Gateway) → Monitoring(成本看板) → Cost Control
```

进阶：在平台上叠加 Agent Runtime + RAG 服务，形成 Mini OpenAI 全栈。

**检验**：平台 MVP 带模型注册/自动扩缩容/成本核算；能讲清 Agent state 持久化与容错方案。

---

## 4. 必做 CODE 清单（ZOMI 课程，按优先级）

### P0：核心能力（面试硬通货）

| # | CODE | 对应能力 |
|---|---|---|
| 1 | 搭建迷你 Transformer + 从零训练 | Transformer 全栈理解 |
| 2 | 从零构建 PyTorch DDP | 分布式基本功 |
| 3 | Megatron 张量并行复现 | TP 原理落地 |
| 4 | FlashAttention 实现 | 算子核心 |
| 5 | MFU 模型利用率评估 | 性能建模 |
| 6 | ZeRO 显存优化实践 | 显存账本 |
| 7 | Pipeline 并行实践 | PP/Bubble 理解 |
| 8 | FP8 混合精度训练 | 新一代精度 |
| 9 | Ring Attention 实践 | 长序列 |
| 10 | InstructGPT 复现 / GRPO 实践 | RL 训练（2026 热点） |

### P1：进阶/差异化

MoE 手撕系列（01~04）、梯度检查点、OpenCompass 评估、Qwen3-4B 微调、LoRA 微调 SD、Transformer 核心机制手写（BPE/Sinusoidal/Embedding/MHA-GQA-MLA）。

---

## 5. MoE 论文精读线（ZOMI 06 专属路线）

```
GShard（分片门控，MoE 奠基）
  → Switch Transformer（稀疏激活，工程简化）
  → GLaM & ST-MoE（规模与稳定性）
  → DeepSeek-MoE（细粒度专家 + 共享专家，2024 代表作）
```

配合：MOE 实践 01~04（HF 推理 → 手撕 → 分布式 → 性能分析）。精读完这 4 篇 + 代码，MoE 相关的面试题基本全覆盖（路由、负载均衡、All-to-All 通信、EP×TP 组合）。

---

## 6. 检查清单 v3.0 增量（相对 v2.0）

**后训练/RL**
- [ ] 能讲清 InstructGPT 三阶段（SFT/RM/PPO）的系统开销
- [ ] 能对比 PPO vs DPO vs GRPO 的训练成本与工程复杂度
- [ ] 跑通过 GRPO 实践代码

**算法与数据**
- [ ] 讲清 GQA/MLA 对 KV Cache 的影响（推理 Infra 必考）
- [ ] 完成 MoE 论文线精读 + 至少 2 个 MoE CODE
- [ ] 懂 Tokenizer（BPE）与 Embedding 的实现细节

**集群/平台**
- [ ] 能讲万卡集群的组网方案与拥塞控制挑战
- [ ] 会设计 CheckPoint 存储方案（本地 SSD + Lustre/S3 分层）
- [ ] 懂 K8s GPU Device Plugin 与 gang scheduling

**应用层**
- [ ] 能画 Agent Runtime 架构（workflow/state/memory/tools）
- [ ] 能对比 RAG vs GraphRAG 的适用场景

---

## 7. v2.0 → v3.0 变化总结

| 维度 | v2.0 | v3.0 |
|---|---|---|
| 模块数 | 8 | 10（+后训练 RL、+算法与数据、+Agent/RAG 归并后为 10 个有效模块） |
| 学习资源 | 单一文章路线 | 文章 + 视频 + CODE 四轨法 |
| 分布式训练 | 原理为主 | 原理 + ZOMI 10 个必做 CODE 全覆盖 |
| 推理 | vLLM 为主 | + ZOMI 5.1~5.6 完整体系（采样/长序列/压缩） |
| 后训练 | 无 | SFT/RLHF/DPO/GRPO + 评估（2026 热点） |
| 算法深挖 | 无 | Transformer 深挖 + MoE 论文线 |
| 应用层 | Agent Runtime 概念 | Agent + RAG + 热点全览 |
| 集群 | 概念级 | ZOMI 01/02/03 完整课程（万卡/组网/存储/云原生） |

---

## 8. 使用注意事项

- **下载**：ZOMI 仓库 10G+，勿 clone，用 Releases 按模块下载
- **视频**：B站「ZOMI酱」/ 油管「ZOMI6222」，配合 PPT 使用
- **引用**：引用 ZOMI PPT 内容请规范转载并标明出处
- **节奏**：视频用 1.25~1.5 倍速，每主题控制在 2 天内完成四轨闭环，避免只刷视频不动手
- **裁剪**：目标明确时按 v2.0 第 5 节岗位方向表裁剪模块（如推理岗跳过 MoE 论文线深挖、平台岗跳过算子细节）
