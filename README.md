# AI Infra study

AI Infra 学习资料与实践仓库，按「基础 → CUDA/算子 → 分布式训练 → 推理部署 → 平台工程」组织。

## 目录

- `01-学习路线/`：系统工程版、双资源整合版和 AIInfraGuide 路线。
- `02-参考资料/`：论文、课程、官方文档和源码链接。
- `03-项目与实践/`：实验代码、Benchmark、学习笔记和项目记录。

## 互动学习站

`site/index.html` 是无需后端即可打开的本地学习页面，包含三个实验：

- Token 生成：用固定候选 logits 和稳定的 softmax 展示 temperature 对概率分布及实际采样结果的影响。
- KV Cache 账本：按层数、KV heads、head_dim、上下文长度、并发数与 KV 精度估算规模；页面同时展示逐项公式与精确字节数。
- 请求调度：模拟 continuous batching 中请求加入、decode 与完成的过程。

Token 实验使用可在页面中修改的固定随机种子，因此同一 seed、temperature 和采样顺序可以复现。它只用于解释采样机制，不代表真实模型推理或性能结果。

运行不依赖第三方包的测试：

```bash
node --test site/*.test.js
```

CI 会在每次推送和拉取请求时执行同一测试。

## 推荐学习顺序

1. Linux、C++、Python、线性代数、Transformer、PyTorch
2. CUDA 编程、GPU 架构、Reduce/GEMM/Softmax、Triton、Profiling
3. DDP、FSDP/ZeRO、TP/PP/CP/MoE、NCCL、MFU 与通信建模
4. vLLM/SGLang、KV Cache、量化、Speculative Decoding、Prefill/Decode 解耦
5. Kubernetes、Slurm/Volcano、存储、Checkpoint、监控与成本工程

## 学习原则

- 先跑通，再读源码，再做性能分析。
- 每个实验记录硬件、软件版本、配置、指标和结论。
- 始终分析计算、通信、显存、精度、延迟和成本之间的 trade-off。

## 核心资源

- AIInfraGuide: https://caomaolufei.github.io/AIInfraGuide/
- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- PyTorch Distributed: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM
- DeepSpeed: https://github.com/microsoft/DeepSpeed
- vLLM: https://github.com/vllm-project/vllm
- SGLang: https://github.com/sgl-project/sglang
- Triton: https://github.com/triton-lang/triton

本目录为个人学习资料整理，不包含第三方仓库源码和受版权保护的课程文件。
