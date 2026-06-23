# Vime

[English](./README.md) · [代码仓库](https://github.com/vllm-project/vime)

<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
[![文档](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.vllm.ai/projects/vime/zh-cn/latest/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/vllm-project/vime)

**Vime** 是基于 [slime](https://github.com/THUDM/slime) 的 RL scaling 用 LLM post-training 框架。在保留 slime 训练栈与数据生成设计的同时，默认以 [**vLLM**](https://github.com/vllm-project/vllm)（配合 [vllm-router](https://github.com/vllm-project/router)）作为 rollout 后端。Vime 提供两大核心能力：
=======
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/vime/)
[![CI](https://img.shields.io/github/actions/workflow/status/THUDM/vime/pr-test.yml?branch=zilin%2Fci-dont-merge&event=pull_request&label=CI&logo=github)](https://github.com/THUDM/vime/pull/2053/checks)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

**slime** 是为 RL scaling 设计的 LLM post‑training 框架，提供两大核心能力：

1. **高性能训练**：通过连接 Megatron 与 VLlm，支持各种模式的高效训练；
2. **灵活的数据生成**：通过自定义数据生成接口以及 server-based engine，实现任意训练数据生成流程。

slime 的设计目标，是让这两大能力彼此强化，同时避免把系统变成一组割裂的 trainer、rollout service 和 agent framework。Megatron training、VLlm rollout、custom data generation、reward computation、verifier feedback 和 environment interaction 都流经同一条 training / rollout / Data Buffer 路径。

这让 slime 成为最经受实战验证的开源 RL post-training 框架之一：它足够轻量、清晰、易扩展，同时也经过了 SOTA 级模型发布背后的完整训练闭环验证。

## 为什么这个设计重要

- **经过 frontier model 训练验证**：slime 是 [GLM-5.2](https://z.ai/blog/glm-5.2)、[GLM-5.1](https://z.ai/blog/glm-5.1)、[GLM-5](https://z.ai/blog/glm-5)、[GLM-4.7](https://z.ai/blog/glm-4.7)、[GLM-4.6](https://z.ai/blog/glm-4.6)、[GLM-4.5](https://z.ai/blog/glm-4.5) 背后的 RL 训练框架。这验证的是完整 post-training loop，而不是孤立 example。
- **以正确性为先的基础设施**：RL bug 往往不会立刻报错。slime 保持显式的数据流，支持 rollout-only 和 train-only 分离调试，并把可复现性、容错、trace、profiling 和 CI 作为一等工程问题来维护。
- **从设计开始就是 native**：slime 直接透传 Megatron 参数，并通过 `--vllm-` 前缀暴露当前安装版本 VLlm 支持的参数。新的上游训练和 serving 优化可以直接使用，不需要在 slime 里再加一层抽象。
- **最大化的数据生成自由度**：math、code、search、tool、sandbox、verifier、environment、multi-agent system 以及 long-horizon agentic workflow 都可以作为 data generation 或 reward workflow 接入，而不需要 fork training kernel。
- **轻量且有明确取舍**：slime 深度优化 Megatron + VLlm 这条大规模 RL 生产路径。选择单一 rollout backend 让 slime 可以直接发挥 VLlm 的特有能力，而不是为了同时兼容多个推理框架，被迫抽象成 lowest-common-denominator 的公共能力子集。
>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt

1. **高性能训练**：通过连接 Megatron 与 vLLM，支持各种模式的高效训练；
2. **灵活的数据生成**：通过自定义数据生成接口以及 server based engine，实现任意的训练数据生成流程。

Vime 继承了 slime 广泛的模型支持，包括：

- Qwen 系列（Qwen3.6、Qwen3.5、Qwen3Next、Qwen3MoE、Qwen3、Qwen2.5）；
- DeepSeek V3 系列（DeepSeek V3、V3.1、DeepSeek R1）；
- Llama 3。

<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
讨论渠道：
=======
## 原生 Engine 透传与 VLlm 部署

slime 不只是一个能调用推理后端的框架。它尽量保留 Megatron 和 VLlm 上游 engine 原生的控制面，同时在其外层加入 RL 所需的数据流：

- 原生 VLlm 参数透传：当前安装版本 VLlm 支持的每个参数，都可以通过增加 `--vllm-` 前缀使用，例如把 `--mem-fraction-static` 写成 `--vllm-mem-fraction-static`；
- 原生 Megatron 参数透传：slime 直接读取 Megatron 参数，因此 Megatron 侧的并行、优化器、checkpoint 和模型配置不需要 wrapper code 也能继续使用；
- [VLlm Config](docs/zh/advanced/vllm-config.md) 作为可选 YAML 扩展，用于 topology-specific control，例如为 prefill/decode/EPD-style 部署设置独立参数、配置 heterogeneous server group、multi-model serving 和 per-group VLlm override；
- 面向 multi-turn 和 agentic workload 的 [PD Disaggregation](docs/zh/advanced/pd-disaggregation.md)，用于处理 prefill/decode 资源需求不同的问题；
- 面向 multi-turn agent 的 session affinity 等 router policy；
- 面向 training/inference disaggregation 和大模型高效更新的 [Delta Weight Sync](docs/zh/advanced/delta-weight-sync.md)；
- [External Rollout Engines](docs/zh/advanced/external-rollout-engines.md)，用于 serving 由训练任务外部管理的部署形态；VLlm serving 侧可以使用独立环境，disk transport 下还可以使用不同型号或不同厂家的 GPU，并通过共享文件系统做 full checkpoint update from disk 或 delta update。

这种透传设计让 slime 从一开始就是 native 的。大多数上游 engine 的优化会随着 engine 升级自然可用，而 slime 可以把主要复杂度集中在 RL loop、dataflow、synchronization 和 correctness check 上。

选择 VLlm 作为单一 rollout backend 也是有意为之。多 backend 框架往往需要在多个 inference engine 的公共能力子集上做抽象，结果会遮住每个 backend 最强的特性。slime 则深度优化 VLlm，让 RL workload 可以直接使用 VLlm-specific 的 serving、routing、caching、disaggregation 和 weight-sync 能力。

## 正确性、稳定性与 CI
>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt

- [Slack](https://vllm-dev.slack.com/archives/C0B8W5QFL22/p1780899164831779)
- [微信群](./imgs/wechat_group.png)

## 定位

vLLM 社区横向支持许多 LLM post-training 框架，包括（按字母顺序）[NeMo RL](https://github.com/NVIDIA-NeMo/RL)、[OpenRLHF](https://github.com/openrlhf/openrlhf)、[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)、[SkyRL](https://github.com/NovaSky-AI/SkyRL)、[verl](https://github.com/verl-project/verl) 等。我们创建了 Vime 项目，旨在将 slime 经过验证的训练范式无缝引入 vLLM 生态系统，提供一个可用于生产的桥梁，对齐两个项目的快速迭代节奏。我们希望有不同需求的用户都能在 vLLM 生态中找到适合自己工作流的选择。vLLM 社区会一如既往地支持这些 post-training 框架中的 vLLM 集成。

## 目录

- [Vime](#vime)
  - [定位](#定位)
  - [目录](#目录)
  - [架构总览](#架构总览)
  - [快速开始](#快速开始)
  - [参数说明](#参数说明)
  - [开发指南](#开发指南)
  - [slime doc](#slime-doc)
  - [FAQ](#faq)
  - [致谢](#致谢)
  - [引用](#引用)

## 架构总览

![arch](./imgs/arch.png)

**模块说明**：

- **training (Megatron)**：负责主训练流程，从 Data Buffer 读取数据，训练完后将参数同步至 rollout 模块；
- **rollout (vLLM + router)**：启动 vLLM 推理引擎并路由生成请求，产出新数据（含 reward/verifier），存储至 Data Buffer；
- **data buffer**：桥梁模块，管理 prompt 初始化、自定义数据与 rollout 生成方法。

## 快速开始

有关环境配置、数据准备、训练启动和关键代码分析的完整快速开始指南，请参考：

- [快速开始指南](./docs/zh/get_started/quick_start.md)

我们还提供了一些未在快速开始中覆盖的使用示例，请查看 [examples](examples/)。

<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
=======
### Agentic RL 示例

下面这些 example 通过 customization 接口接入标准的 rollout / Data Buffer 闭环，而不是独立的 framework：

- [`examples/multi_agent`](examples/multi_agent/README.md)：通过自定义 `--rollout-function-path` 实现多 agent 的 rollout。
- [`examples/search-r1`](examples/search-r1/)：通过 `--custom-generate-function-path` 实现 search/RAG 风格的多轮生成。
- [`examples/fully_async`](examples/fully_async/README.md)：fully-async rollout，适合不同样本生成耗时差异较大的 long-tail agentic 场景。
- [`examples/coding_agent_rl`](examples/coding_agent_rl/README.md)：端到端 SWE coding-agent RL，包含 sandboxed tool use、test-based reward，以及通过 `--custom-generate-function-path` 导出的 token-correct trajectory segments。

如何为某种 agentic workflow 选择合适的接口，请参考 [自定义指南](docs/zh/get_started/customization.md)。

## 基于 slime 构建的生态

这些项目不只是 demo。它们是把 slime 作为可复用 RL substrate 的独立系统，覆盖生产级 post-training、agentic RL、domain RL 和 rollout-system research。

### ⛵ Miles：面向大规模模型训练的企业级强化学习框架

[Miles](https://github.com/radixark/miles) 是 [RadixArk](https://github.com/radixark) 基于 slime 构建的大模型 RL 后训练框架。它与 slime 上游开发保持紧密同步，同时在此基础上针对企业场景做了一系列扩展：更深度的 [VLlm](https://github.com/sgl-project/vllm) 集成、配套的运维与部署工具和服务，以及针对[新模型](https://www.radixark.com/miles/docs/models)和[新硬件](https://www.radixark.com/miles/docs/platforms)的优化。Miles 也在持续围绕真实生产环境需求迭代和进化，例如加入对 LoRA、TITO、低精度训练的支持。

### 🔷 vime: 基于 slime 的 vLLM-Native RL Post-Training 框架

[**vime**](https://github.com/vllm-project/vime) 是由 vLLM 项目维护的、基于 slime 的后训练框架。它保留 slime 的 Megatron 训练栈、Data Buffer 数据流与自定义 data generation 设计，主要特点是将 rollout 后端替换为 [**vLLM**](https://github.com/vllm-project/vllm)（配合 [vllm-router](https://github.com/vllm-project/router)）。在现有 slime 启动脚本基础上仅调整 rollout 相关参数，即可快速适配 vime 进行训练。

### 🌈 Relax: Asynchronous RL Engine for Omni-Modal Agentic Training

[**Relax**](https://github.com/redai-infra/Relax) (Reinforcement Engine Leveraging Agentic X-modality) 是 RedAI Infra 团队开源的 omni-modal agentic RL framework，构建在结合 Ray、Megatron-LM 和 VLlm 的 slime infrastructure stack 之上。Relax 采用 Ray Serve 上的 service-oriented architecture，以 Megatron-LM 和 VLlm 作为 training/inference backend。它使用 [TransferQueue](https://github.com/Ascend/TransferQueue) 将 Actor、Rollout、ActorFwd、Reference 和 Advantage computation 完全解耦到独立 GPU 集群，并引入 **DCS (Distributed Checkpoint Service)**，通过 NCCL-broadcast weight-sync engine 将更新后的 Actor 权重异步 stream 到 Rollout/ActorFwd/Reference，并与下一步训练重叠，从而在可配置 staleness 下实现 fully-async training。Relax 支持 text、vision、audio（包括 Qwen3-Omni）以及 agentic multi-turn rollout 的端到端 RL。

### 🦞 OpenClaw-RL: Train a Personalized Clawbot Simply by Talking to It

[**OpenClaw-RL**](https://github.com/Gen-Verse/OpenClaw-RL) 是面向 personalized OpenClaw agent 的 RL server。它托管 OpenClaw model，并从跨部署的历史对话中持续改进模型，同时 slime 的 asynchronous RL infrastructure 避免训练过程干扰 API serving。它支持两种自动优化方法：基于后续状态推断 binary feedback 的 GRPO，以及从后续反馈中提取 hindsight hint 的 on-policy distillation。

### ⚛️ P1: Mastering Physics Olympiads with Reinforcement Learning

[**P1**](https://prime-rl.github.io/P1/) 是一系列完全通过 reinforcement learning 训练的开源物理推理模型。P1 使用 slime 作为 RL post-training framework，并提出 multi-stage RL training algorithm，通过 adaptive learnability adjustment 和 stabilization mechanism 逐步增强推理能力。在这一训练范式下，P1 在开源物理推理上取得了突破性表现。

### 📈RLVE: Scaling LM RL with Adaptive Verifiable Environments

[**RLVE**](https://github.com/Zhiyuan-Zeng/RLVE) 提出使用 verifiable environments 来扩展语言模型 RL：环境以程序化方式生成问题，并提供可算法验证的 reward。通过在 400 个 verifiable environment 上联合训练，RLVE 能让每个 environment 随训练进展动态适配 problem difficulty distribution，使其匹配当前 policy model 的能力。

### ⚡ TritonForge: Agentic RL Training Framework for Kernel Generation

[**TritonForge**](https://github.com/RLsys-Foundation/TritonForge) 使用 slime 的 SFT 和 RL 能力训练能够自动生成优化 GPU kernel 的 LLM。通过 supervised fine-tuning 加 reinforcement learning with multi-turn compilation feedback 的两阶段训练，TritonForge 在将 PyTorch operation 转换为高性能 Triton kernel 上取得了显著结果。

### 🚀 APRIL: Accelerating RL Training with Active Partial Rollouts

[**APRIL**](https://github.com/RLsys-Foundation/APRIL) 提出一种可以无缝集成到 slime 的 system-level optimization，用于加速 RL 训练中的 rollout generation 阶段。它通过智能 over-provision request 并主动管理 partial completion，缓解 rollout 生成中常见的 long-tail bottleneck，而这一阶段通常会消耗 RL 训练 90% 以上的时间。

### 🏟️ qqr: Scaling Open-Ended Agents with ArenaRL & MCP

[**qqr**](https://github.com/Alibaba-NLP/qqr) (a.k.a. hilichurl) 是一个用于演化 open-ended agent 的 slime lightweight extension。它实现 **ArenaRL** algorithm，通过 tournament-based relative ranking（例如 Seeded Single-Elimination、Round-Robin）缓解 discriminative collapse，并无缝集成 **Model Context Protocol (MCP)**。qqr 利用 slime 的高吞吐训练能力，在标准化、解耦的 tool environment 中实现可扩展的分布式 agent evolution。

### ☁️ ART: Scalable and Sandboxed Agentic RL on AWS Bedrock AgentCore Runtime

[**ART (AgentCore RL Toolkit)**](https://github.com/awslabs/agentcore-rl-toolkit) 是一个能够将真实生产环境中的 agent 适配到 **AWS Bedrock AgentCore Runtime** 上进行 RL 训练的工具包。AgentCore Runtime 提供了能够自动扩展以及沙盒式封闭管理的智能体运行环境，这非常适合安全地并行运行大量 agent rollouts。利用 ART，用户只需在 agent 代码上使用一个 decorator（`@app.rollout_entrypoint`），即可在直接复用生产环境的 agent harness 基础上完成 RL 训练的适配，其中用于 RL 训练的 token capture 则在 model gateway layer 中完成。ART 将 slime 列为 RL 训练的后端选项之一，帮助用户能够轻松地使用 slime 中的 RL 训练算法优化生产环境上的 agent 模型。

这些项目共同体现了 slime 的核心思路：一个高性能 RL kernel 可以同时支撑 frontier model post-training、online agent optimization、verifiable environment、omni-modal rollout、kernel-generation agent 和 rollout-system research，而不需要改变核心 training loop。

>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt
## 参数说明

Vime 的参数分为三类：

1. **Megatron 参数**：Vime 会读取 Megatron 中的全部参数，可通过传入如 `--tensor-model-parallel-size 2` 的方式配置 Megatron；
2. **vLLM 参数**：vLLM server 与 engine 相关选项以 `--vllm-` 为前缀（例如 `--vllm-gpu-memory-utilization`）。路由相关选项分两类前缀：vllm-router 自身的选项以 `--router-` 传入（例如 `--router-policy round_robin`、`--router-request-timeout-secs`），Vime 侧用于告诉 Vime *router 在哪里* 的编排参数则以 `--vllm-router-` 为前缀（`--vllm-router-ip`、`--vllm-router-port`）。完整参数见 [vime/backends/vllm_utils/arguments.py](vime/backends/vllm_utils/arguments.py)。
3. **框架参数**：与 Vime 编排相关的开关（rollout GPU、数据路径、RL 算法等），见 [vime/utils/arguments.py](vime/utils/arguments.py)。

`--rollout-num-gpus-per-engine` 对应每个 vLLM engine 的 tensor parallel size。默认 rollout 入口为 `vime.rollout.vllm_rollout.generate_rollout`。

完整使用说明请查阅 [使用文档](docs/zh/get_started/usage.md)。

## 开发指南

- **欢迎贡献！** 若有功能建议、性能调优或使用体验反馈，欢迎提交 Issue / PR。

- 使用 [pre-commit](https://pre-commit.com/) 保证提交代码风格：

  ```bash
  apt install pre-commit -y
  pre-commit install

  # 运行 pre-commit 保证代码风格
  pre-commit run --all-files --show-diff-on-failure --color=always
  ```

- 调试技巧请参考 [debug 指南](docs/zh/developer_guide/debug.md)

## slime doc

Vime 由 slime 衍生而来。以下上游资源与本仓库文档仍沿用 slime 命名，可作为共享概念（Megatron 集成、定制化、高级主题）的参考：

[![Documentation](https://img.shields.io/badge/slime_文档-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

- 上游仓库：[THUDM/slime](https://github.com/THUDM/slime)
- 本仓库英文文档：[docs/en/](docs/en/)
- 本仓库中文文档：[docs/zh/](docs/zh/)

## FAQ

常见问题请见 [Q&A](docs/zh/get_started/qa.md)

## 致谢

Vime 构建于开源 RL 生态的想法与基础设施之上。特别感谢 [slime](https://github.com/THUDM/slime) 社区——Vime 直接构建于其出色工作之上；也感谢 [SkyRL](https://github.com/NovaSky-AI/SkyRL) 与 [verl](https://github.com/verl-project/verl)，我们参考了它们的优秀工作。Vime 由 vLLM 社区维护。

## 引用

```bibtex
@misc{vime,
  author       = {Vime Contributors},
  title        = {Vime: An LLM post-training framework with vLLM for RL Scaling},
  year         = {2026},
  howpublished = {\url{https://github.com/vllm-project/vime}},
  urldate      = {2026-06}
}
```
