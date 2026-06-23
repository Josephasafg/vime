# Vime

[中文版](./README_zh.md) · [Repository](https://github.com/vllm-project/vime)

<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.vllm.ai/projects/vime/en/latest/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/vllm-project/vime)

**Vime** is an LLM post-training framework for RL scaling, built on [slime](https://github.com/THUDM/slime). It keeps slime's training stack and data-generation design while using [**vLLM**](https://github.com/vllm-project/vllm) (with [vllm-router](https://github.com/vllm-project/router)) as the default rollout backend. Vime provides two core capabilities:
=======
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/vime/)
[![CI](https://img.shields.io/github/actions/workflow/status/THUDM/vime/pr-test.yml?branch=zilin%2Fci-dont-merge&event=pull_request&label=CI&logo=github)](https://github.com/THUDM/vime/pull/2053/checks)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

**slime** is an LLM post-training framework for RL scaling, providing two core capabilities:

1.  **High-Performance Training**: Supports efficient training in various modes by connecting Megatron with VLlm;
2.  **Flexible Data Generation**: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

slime's design goal is to make these two capabilities reinforce each other without turning the system into a heavy stack of disconnected trainers, rollout services, and agent frameworks. Megatron training, VLlm rollout, custom data generation, reward computation, verifier feedback, and environment interaction all flow through the same training / rollout / Data Buffer path.

This makes slime one of the most battle-tested open RL post-training frameworks: small enough to understand and extend, but validated through complete training loops behind SOTA-level model releases.

## Why This Design Matters

- **Battle-tested by frontier model training**: slime is the RL framework behind [GLM-5.2](https://z.ai/blog/glm-5.2), [GLM-5.1](https://z.ai/blog/glm-5.1), [GLM-5](https://z.ai/blog/glm-5), [GLM-4.7](https://z.ai/blog/glm-4.7), [GLM-4.6](https://z.ai/blog/glm-4.6), and [GLM-4.5](https://z.ai/blog/glm-4.5). This validates the full post-training loop, not only isolated examples.
- **Correctness-first infrastructure**: RL bugs are often silent. slime keeps the dataflow explicit, supports separate rollout-only and train-only debugging paths, and documents reproducibility, fault tolerance, tracing, profiling, and CI as first-class engineering concerns.
- **Native by design**: slime passes Megatron arguments through directly and exposes installed VLlm arguments with a `--vllm-` prefix. New upstream training and serving optimizations can be used without adding another abstraction layer inside slime.
- **Maximum data-generation freedom**: math, code, search, tools, sandboxes, verifiers, environments, multi-agent systems, and long-horizon agentic workflows plug in as data generation or reward workflows. They do not fork the training kernel.
- **Lightweight and opinionated**: slime focuses deeply on the Megatron + VLlm path used for large-scale RL. By choosing one rollout backend, slime can use VLlm-specific capabilities directly instead of flattening multiple inference engines into a lowest-common-denominator abstraction.
>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt

1. **High-performance training**: Efficient training in various modes by connecting Megatron with vLLM;
2. **Flexible data generation**: Arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

Vime inherits broad model support from slime, including:

- Qwen series (Qwen3.6, Qwen3.5, Qwen3Next, Qwen3MoE, Qwen3, Qwen2.5);
- DeepSeek V3 series (DeepSeek V3, V3.1, DeepSeek R1);
- Llama 3.

<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
Discussion channels:
=======
## Native Engine Pass-Through and VLlm Deployment

slime is not just a framework that can call an inference backend. It keeps the Megatron and VLlm control surfaces close to the upstream engines while adding the RL dataflow around them:

- native VLlm argument pass-through: every argument supported by the installed VLlm can be used by adding the `--vllm-` prefix, such as passing `--mem-fraction-static` as `--vllm-mem-fraction-static`;
- native Megatron argument pass-through: slime reads Megatron arguments directly, so Megatron-side parallelism, optimizer, checkpointing, and model options remain available without wrapper code;
- [VLlm Config](docs/en/advanced/vllm-config.md) as an optional YAML extension for topology-specific control, such as separate prefill/decode/EPD-style settings, heterogeneous server groups, multi-model serving, and per-group VLlm overrides;
- [PD Disaggregation](docs/en/advanced/pd-disaggregation.md) for multi-turn and agentic workloads with different prefill/decode resource needs;
- router policies such as session affinity for multi-turn agents;
- [Delta Weight Sync](docs/en/advanced/delta-weight-sync.md) for training/inference disaggregation and large-model update efficiency;
- [External Rollout Engines](docs/en/advanced/external-rollout-engines.md) for deployments where serving is managed outside the training job. The VLlm serving side can use an independent environment, and with disk transport can even run on different GPU models or vendors while using full-checkpoint update from disk or delta update over a shared filesystem.

This pass-through design makes slime native from the start. Most upstream engine improvements remain accessible as the engines evolve, while slime focuses on the RL loop, dataflow, synchronization, and correctness checks.

Choosing VLlm as the single rollout backend is also intentional. Multi-backend frameworks often have to abstract over the common subset of several inference engines, which can hide the strongest features of each backend. slime instead optimizes deeply for VLlm so RL workloads can use VLlm-specific serving, routing, caching, disaggregation, and weight-sync behavior directly.

## Correctness, Stability, and CI
>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt

- [slack](https://vllm-dev.slack.com/archives/C0B8W5QFL22/p1780899164831779)
- [wechat group](./imgs/wechat_group.png)

## Positioning

The vLLM community horizontally supports many LLM post-training frameworks, including (in alphabetical order) [NeMo RL](https://github.com/NVIDIA-NeMo/RL), [OpenRLHF](https://github.com/openrlhf/openrlhf), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), [SkyRL](https://github.com/NovaSky-AI/SkyRL), [verl](https://github.com/verl-project/verl), and so on. We built the Vime project to seamlessly bring slime's proven training paradigm into the vLLM ecosystem, offering a production-ready bridge that aligns both projects' rapid release cycles. We hope that users with different needs can find the right vLLM-ecosystem choice for their workflows. The vLLM community will continue to support the vLLM integration in these post-training frameworks.

## Table of Contents

- [Vime](#vime)
  - [Positioning](#positioning)
  - [Table of Contents](#table-of-contents)
  - [Architecture Overview](#architecture-overview)
  - [Quick Start](#quick-start)
  - [Arguments Walkthrough](#arguments-walkthrough)
  - [Developer Guide](#developer-guide)
  - [slime doc](#slime-doc)
  - [FAQ](#faq)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)

## Architecture Overview

![arch](./imgs/arch.png)

**Module Descriptions**:

- **training (Megatron)**: Responsible for the main training process, reads data from the Data Buffer, and synchronizes parameters to the rollout module after training.
- **rollout (vLLM + router)**: Launches vLLM inference engines and routes generation requests; produces new data (including rewards/verifier outputs) and stores it in the Data Buffer.
- **data buffer**: A bridge module that manages prompt initialization, custom data, and rollout generation methods.

## Quick Start

For a comprehensive quick start guide covering environment setup, data preparation, training startup, and key code analysis, please refer to:

- [Quick Start Guide](./docs/en/get_started/quick_start.md)

We also provide examples for some use cases not covered in the quick start guide; please check [examples](examples/).

<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
=======
### Agentic RL examples

For agentic RL workloads, the following examples plug into the standard rollout / Data Buffer loop through customization interfaces — they are not separate frameworks:

- [`examples/multi_agent`](examples/multi_agent/README.md): Multi-agent rollout via a custom `--rollout-function-path`.
- [`examples/search-r1`](examples/search-r1/): Search/RAG-style multi-turn generation via `--custom-generate-function-path`.
- [`examples/fully_async`](examples/fully_async/README.md): Fully-async rollout, useful for long-tail agentic generation where some samples take much longer than others.
- [`examples/coding_agent_rl`](examples/coding_agent_rl/README.md): End-to-end SWE coding-agent RL with sandboxed tool use, test-based rewards, and token-correct trajectory segments via `--custom-generate-function-path`.

See the [Customization Guide](docs/en/get_started/customization.md) for which interface to use for a given agentic workflow.

## Ecosystem Built on slime

These are not just demos. They are independent systems that use slime as a reusable RL substrate for production-scale post-training, agentic RL, domain RL, and rollout-system research.

### 🐎 Dressage: Scalable RL for Any Agent and Sandbox

[**Dressage**](https://github.com/Accio-Lab/Dressage) is an agentic RL training framework built on slime by [Alibaba Accio](https://www.accio.com/work?im_ref=1O8wgT3poxyZWCj31F1ZJ0fNUkuTK6x9ZTHw0Y0&sharedid=&im_pid=5619512&im_pname=AI%20INTRO%20COPORATE), centered on unified RL for blackbox agents (e.g., [OpenCode](https://github.com/anomalyco/opencode), [OpenClaw](https://github.com/openclaw/openclaw)) and white loops across any sandbox environment (e.g., [bwrap](https://github.com/containers/bubblewrap), [E2B](https://github.com/e2b-dev/e2b), Kubernetes). It decouples interaction semantics, execution placement, and token-level trajectory capture through Paddock, Sandbox, and Proxy layers, adapting agent workflows without rewriting their internal loops. Dressage records token-wise logprobs, loss masks, weight versions, and MoE routing, then uses TITO and segment-aware training to turn long-horizon tool interactions into stable RL samples.

### ⛵ Miles: Enterprise-Grade Reinforcement Learning for Large-Scale Model Training

[Miles](https://github.com/radixark/miles) is an RL post-training framework for large-scale models, built on slime by [RadixArk](https://github.com/radixark). It stays closely aligned with slime's upstream development while extending it with enterprise-oriented features: deeper [VLlm](https://github.com/sgl-project/vllm) integration, operational tooling, deployment support, and optimizations for new [models](https://www.radixark.com/miles/docs/models) and [hardware](https://www.radixark.com/miles/docs/platforms). Miles also adds a growing set of production features, including LoRA, TITO, and low-precision training.

### 🔷 vime: vLLM-Native RL Post-Training Built on slime

[**vime**](https://github.com/vllm-project/vime) is a post-training framework built on slime and maintained by the vLLM project. It keeps slime's Megatron training stack, Data Buffer dataflow, and custom data-generation design, with its main change being a rollout backend swapped to [**vLLM**](https://github.com/vllm-project/vllm) with [vllm-router](https://github.com/vllm-project/router). Starting from an existing slime launch script, adjusting only rollout-related parameters is enough to quickly run training with vime.

### 🌈 Relax: Asynchronous RL Engine for Omni-Modal Agentic Training

[**Relax**](https://github.com/redai-infra/Relax) (Reinforcement Engine Leveraging Agentic X-modality) is an omni-modal agentic RL framework open-sourced by the RedAI Infra team, built upon the slime infrastructure stack that combines Ray, Megatron-LM, and VLlm. Relax adopts a service-oriented architecture on Ray Serve with Megatron-LM and VLlm as training/inference backends. It uses [TransferQueue](https://github.com/Ascend/TransferQueue) to fully decouple Actor, Rollout, ActorFwd, Reference, and Advantage computation onto independent GPU clusters, and introduces **DCS (Distributed Checkpoint Service)** — an NCCL-broadcast weight-sync engine that streams updated Actor weights to Rollout/ActorFwd/Reference asynchronously and overlaps the transfer with the next training step, enabling fully-async training at configurable staleness. Relax supports end-to-end RL for text, vision, and audio (including Qwen3-Omni) and agentic multi-turn rollouts.

### 🦞 OpenClaw-RL: Train a Personalized Clawbot Simply by Talking to It

[**OpenClaw-RL**](https://github.com/Gen-Verse/OpenClaw-RL) is an RL server for personalized OpenClaw agents. It hosts the OpenClaw model and improves it from prior conversations across deployments, while slime's asynchronous RL infrastructure prevents training from interfering with API serving. It supports two automatic optimization methods: GRPO with binary feedback inferred from subsequent states, and on-policy distillation that extracts hindsight hints from later feedback for the current policy.

### ⚛️ P1: Mastering Physics Olympiads with Reinforcement Learning

[**P1**](https://prime-rl.github.io/P1/) is a family of open-source physics reasoning models trained entirely through reinforcement learning. P1 leverages slime as the RL post-training framework, and introduces a multi-stage RL training algorithm that progressively enhances reasoning ability through adaptive learnability adjustment and stabilization mechanisms. Empowered by this training paradigm, P1 delivers breakthrough performance in open-source physics reasoning.

### 📈RLVE: Scaling LM RL with Adaptive Verifiable Environments

[**RLVE**](https://github.com/Zhiyuan-Zeng/RLVE) introduces an approach using verifiable environments that procedurally generate problems and provide algorithmically verifiable rewards, to scale up RL for language models (LMs). With joint training across 400 verifiable environments, RLVE enables each environment to dynamically adapt its problem difficulty distribution to the policy model's capabilities as training progresses.

### ⚡ TritonForge: Agentic RL Training Framework for Kernel Generation

[**TritonForge**](https://github.com/RLsys-Foundation/TritonForge) leverages slime's SFT and RL capabilities to train LLMs that automatically generate optimized GPU kernels. By using a two-stage training approach—supervised fine-tuning followed by reinforcement learning with multi-turn compilation feedback—TritonForge achieves remarkable results in converting PyTorch operations into high-performance Triton kernels.

### 🚀 APRIL: Accelerating RL Training with Active Partial Rollouts

[**APRIL**](https://github.com/RLsys-Foundation/APRIL) introduces a system-level optimization that seamlessly integrates with slime to accelerate the rollout generation phase in RL training. By intelligently over-provisioning requests and actively managing partial completions, APRIL addresses the long-tail generation bottleneck that typically consumes over 90% of RL training time.

### 🏟️ qqr: Scaling Open-Ended Agents with ArenaRL & MCP

[**qqr**](https://github.com/Alibaba-NLP/qqr) (a.k.a. hilichurl) is a lightweight extension for slime designed to evolve open-ended agents. It implements the **ArenaRL** algorithm to tackle discriminative collapse through tournament-based relative ranking (**e.g., Seeded Single-Elimination, Round-Robin**) and seamlessly integrates the **Model Context Protocol (MCP)**. qqr leverages slime's high-throughput training capabilities to enable scalable, distributed evolution of agents in standardized, decoupled tool environments.

### ☁️ ART: Scalable and Sandboxed Agentic RL on AWS Bedrock AgentCore Runtime

[**ART (AgentCore RL Toolkit)**](https://github.com/awslabs/agentcore-rl-toolkit) is an SDK that adapts production agents for RL training on **AWS Bedrock AgentCore Runtime**. AgentCore Runtime provides auto-scaled and sandboxed agent execution environments well-suited for running many parallel agent rollouts securely. Using ART, user only needs to apply a decorator (`@app.rollout_entrypoint`) to their agent codes for RL adaption while the same production agent harness is reused directly, where token capture for RL is handled at model gateway layer. ART uses slime as one option of training backends, enabling users to easily optimizing the production agent model with RL training algorithms in slime.

Together, these projects show the main idea behind slime: one high-performance RL kernel can support frontier model post-training, online agent optimization, verifiable environments, omni-modal rollouts, kernel-generation agents, and rollout-system research without changing the core training loop.

>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt
## Arguments Walkthrough

Arguments in Vime are divided into three categories:

1. **Megatron arguments**: Vime reads all arguments in Megatron. You can configure Megatron by passing arguments like `--tensor-model-parallel-size 2`.
2. **vLLM arguments**: vLLM server and engine options are exposed with a `--vllm-` prefix (for example, `--vllm-gpu-memory-utilization`). Router options live under two prefixes: vllm-router's native options are passed with `--router-` (for example, `--router-policy round_robin`, `--router-request-timeout-secs`), while Vime-side orchestration knobs that tell Vime *where* the router lives use `--vllm-router-` (`--vllm-router-ip`, `--vllm-router-port`). See [vime/backends/vllm_utils/arguments.py](vime/backends/vllm_utils/arguments.py) for the full surface.
3. **Framework-specific arguments**: Shared Vime orchestration flags (rollout GPUs, data paths, RL algorithms, etc.). Please refer to [vime/utils/arguments.py](vime/utils/arguments.py).

`--rollout-num-gpus-per-engine` sets the tensor parallel size of each vLLM engine. The default rollout entry is `vime.rollout.vllm_rollout.generate_rollout`.

For complete usage instructions, please refer to the [Usage Documentation](docs/en/get_started/usage.md).

## Developer Guide

- **Contributions are welcome!** If you have suggestions for new features, performance tuning, or feedback on user experience, feel free to submit an Issue or PR.

- Use [pre-commit](https://pre-commit.com/) to ensure code style consistency for your commits:

```bash
apt install pre-commit -y
pre-commit install

# run pre-commit to ensure code style consistency
pre-commit run --all-files --show-diff-on-failure --color=always
```

- For debugging tips, please refer to the [Debugging Guide](docs/en/developer_guide/debug.md)

## slime doc

Vime is derived from slime. The following upstream resources and in-repo guides still use the slime naming and remain the reference for shared concepts (Megatron integration, customization, advanced topics):

[![Documentation](https://img.shields.io/badge/slime_docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

- Upstream repository: [THUDM/slime](https://github.com/THUDM/slime)
- English docs in this repo: [docs/en/](docs/en/)
- Chinese docs in this repo: [docs/zh/](docs/zh/)

## FAQ

For frequently asked questions, please see the [Q&A](docs/en/get_started/qa.md)

## Acknowledgements

Vime builds on ideas and infrastructure from the open-source RL ecosystem. We especially thank the [slime](https://github.com/THUDM/slime) community, whose great work Vime is directly built on. We also thank [SkyRL](https://github.com/NovaSky-AI/SkyRL) and [verl](https://github.com/verl-project/verl), whose excellent work we referenced. Vime is maintained by the vLLM community.

## Citation

```bibtex
@misc{vime,
  author       = {Vime Contributors},
  title        = {Vime: An LLM post-training framework with vLLM for RL Scaling},
  year         = {2026},
  howpublished = {\url{https://github.com/vllm-project/vime}},
  urldate      = {2026-06}
}
```
