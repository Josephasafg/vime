vime Documentation
====================

vime is an LLM post-training framework for RL scaling, providing two core capabilities:

- High-Performance Training: Supports efficient training in various modes by connecting Megatron with vLLM;
- Flexible Data Generation: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

vime is built on `slime <https://github.com/THUDM/slime>`_, the RL framework behind GLM-4.7, GLM-4.6 and GLM-4.5. vime keeps slime's training stack and data-generation design while using vLLM as the default rollout backend, and inherits broad model support from slime, including:

<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
- Qwen3 series (Qwen3Next, Qwen3MoE, Qwen3), Qwen2.5 series;
- DeepSeek V3 series (DeepSeek V3, V3.1, DeepSeek R1);
- Llama 3.

=======
This makes slime one of the most battle-tested open RL post-training frameworks: small enough to understand and extend, but validated through complete training loops behind SOTA-level model releases.

Why This Design Matters
-----------------------

- **Battle-tested by frontier model training**: slime is the RL framework behind `GLM-5.2 <https://z.ai/blog/glm-5.2>`_, `GLM-5.1 <https://z.ai/blog/glm-5.1>`_, `GLM-5 <https://z.ai/blog/glm-5>`_, `GLM-4.7 <https://z.ai/blog/glm-4.7>`_, `GLM-4.6 <https://z.ai/blog/glm-4.6>`_, and `GLM-4.5 <https://z.ai/blog/glm-4.5>`_.
- **Native by design**: slime passes Megatron arguments through directly and exposes installed VLlm arguments with a ``--vllm-`` prefix, so upstream training and serving optimizations remain available without adding another wrapper layer.
- **VLlm-focused rollout**: slime chooses one rollout backend intentionally. This avoids flattening multiple inference engines into a lowest-common-denominator abstraction and lets RL workloads use VLlm-specific serving, routing, caching, disaggregation, and weight-sync behavior directly.
- **Agentic workflows as data generation**: tool use, sandbox interaction, verifier rewards, environment feedback, multi-agent loops, and long-horizon agentic workflows plug into the same training / rollout / Data Buffer path instead of forking the training kernel.
- **BF16 training with FP8 rollout**: large MoE recipes use Megatron BF16 training state with VLlm FP8 rollout/inference; long-context rollout can also use ``--vllm-kv-cache-dtype fp8_e4m3`` to increase effective KV cache capacity.
- **Tested as RL infrastructure**: CPU correctness tests run automatically, while GPU e2e tests cover real Megatron + VLlm training/rollout paths, including dense/MoE recipes, async rollout, VLlm config, checkpointing, precision, and debug replay. See :doc:`developer_guide/ci`.

Production Validation
---------------------

Beyond the GLM family, slime also supports:

- Qwen series (Qwen3.6, Qwen3.5, Qwen3Next, Qwen3MoE, Qwen3, Qwen2.5);
- DeepSeek V3 series (DeepSeek V3, V3.1, DeepSeek R1);
- Llama 3.

Start by Use Case
-----------------

- New to slime: :doc:`get_started/quick_start`
- Configure training and rollout arguments: :doc:`get_started/usage`
- Add custom generation, reward, or rollout functions: :doc:`get_started/customization`
- Build agentic RL workflows: :doc:`get_started/agent`
- Configure production VLlm rollout topology: :doc:`advanced/vllm-config`
- Connect external rollout engines: :doc:`advanced/external-rollout-engines`
- Use PD disaggregation: :doc:`advanced/pd-disaggregation`
- Use BF16 training with FP8 rollout or FP8 KV cache: :doc:`advanced/low-precision`
- Use delta weight sync: :doc:`advanced/delta-weight-sync`
- Understand CI and reliability coverage: :doc:`developer_guide/ci`
- Debug, trace, and profile long-running jobs: :doc:`developer_guide/debug`, :doc:`developer_guide/trace`, :doc:`developer_guide/profiling`

>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt
.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/quick_start.md
   get_started/usage.md
   get_started/customization.md
   get_started/qa.md

.. toctree::
   :maxdepth: 1
   :caption: Dense

   examples/qwen3-4B.md

.. toctree::
   :maxdepth: 1
   :caption: MoE

   examples/qwen3-30B-A3B.md
<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
=======
   examples/glm5.2-744B-A40B.md
   examples/glm4.7-355B-A32B.md
   examples/deepseek-r1.md
>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   advanced/speculative-decoding.md
   advanced/reproducibility.md
   advanced/fault-tolerance.md
   advanced/observability.md
   advanced/pd-disaggregation.md
<<<<<<< /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_ours.txt
=======
   advanced/external-rollout-engines.md
   advanced/delta-weight-sync.md
>>>>>>> /home/aoshen/vime/projects/slime-sync-2118/agent_run/results/build_3way/tmp_theirs.txt
   advanced/vllm-config.md
   advanced/megatron-config.md
   advanced/arch-support-beyond-megatron.md

.. toctree::
   :maxdepth: 1
   :caption: Other Usage

   _examples_synced/fully_async/README.md
   _examples_synced/multi_agent/README.md

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/ci.md
   developer_guide/debug.md
   developer_guide/trace.md
   developer_guide/profiling.md
