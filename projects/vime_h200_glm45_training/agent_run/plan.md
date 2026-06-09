# Plan: vime_h200_glm45_training

## Goal

h200 2-node colocate L3 RL training of GLM-4.5-Air (106B-A12B): train EP8/PP2/TP4/CP2, rollout TP4 fp8; DAPO-Math-17k + deepscaler GRPO; figure out GLM tool/reasoning parser

## Durable Workspace

- Project root: `/home/aoshen/vime/projects/vime_h200_glm45_training`
- Agent run: `/home/aoshen/vime/projects/vime_h200_glm45_training/agent_run`
- Reusable scripts: `agent_run/scripts/`
- Reports / postmortems: `agent_run/reports/`
- Raw logs / metrics / artifacts: `agent_run/results/<phase-or-run>/`
- Debug good/bad anchors: append `state_anchor` events; generated view is
  `agent_run/bisect.md`

## Locked /solo session contract (2026-06-07)

- **Goal**: Train GLM-4.5-Air (106B-A12B MoE) with vime/vLLM on **h200, 2 nodes × 8 = 16 GPU, colocate (共卡)**.
- **Mode**: experiment / validation hybrid → ultimately **L3 convergence** (reward curve must climb and stay stable).
- **Parallelism (user-pinned, exact)**: train **TP4 / PP2 / CP2** (= world 16 → **DP1**), **EP8 / ETP1**, MoE dispatcher `alltoall`. Rollout **TP4** (`--rollout-num-gpus-per-engine 4` → 16/4 = 4 colocated engines). **FP8 rollout** (BF16 master weights, quantize→fp8 at engine load).
- **Dataset/algo**: reuse proven 30B recipe — **DAPO-Math-17k** prompt data + **deepscaler** RM, **GRPO** (entropy 0, eps-clip 0.2/0.28), KL off, precision-aware optimizer, lr 1e-6 constant.
- **GLM specifics to resolve**: tool-call parser = `glm45`, reasoning parser = `glm45` — confirm whether vime rollout needs them for a pure-math GRPO run (it reads raw text/token_ids; parsers change OpenAI-API response shape, likely NOT needed but must verify against source + GLM chat template / thinking toggle). Config `scripts/models/glm4.5-106B-A12B.sh` already exists (128 experts top-8, 46 layers, sigmoid router, no MTP).
- **Acceptance**: (smoke gate) load → 4 rollout engines up (fp8, MoE backend logged, routing capture non-zero) → colocate IPC weight-sync → train steps run → `VERIFY-D abs-diff` bounded (~0.01–0.05). (L3) sustained run, reward climbs from ~0.5 baseline and holds.
- **Repo/branch**: vime `main` @ 6011ae9f7f. Working tree policy: **all harness edits out-of-tree** (NFS / this `agent_run/`), **no in-tree vime edits** without flagging first.
- **Mutable surface**: out-of-tree harness scripts (run_in_container/launchers/env), weight staging, env vars, log/ckpt paths.
- **Frozen surface**: vime source, the model config `glm4.5-106B-A12B.sh`, DAPO-Math data shape, deepscaler reward, the user-pinned parallelism + fp8 + colocate decisions.
- **Run/check/metric**: launch via 2-node ray (head node0 `ray job submit train.py`, worker node1 joins); check `grep VERIFY-D abs-diff` + wandb reward; metric direction reward **higher better**, abs-diff **lower better / bounded**.
- **Timeout**: smoke ≤ 1 run; L3 long. **Stop conditions**: smoke fails irrecoverably after exhausting fixes → surface; L3 reward diverges/NaN → stop; user interrupt.
- **Template**: translate from proven gb200 colocate harness `projects/vime_gb200_training/run_in_container_gb200.sh` (same algo/optimizer/grpo args; swap platform plumbing arm→x86/h200, model 30B→glm45, +PP2, rollout TP2→TP4, +fp8, +GLM parser if needed).

### ⚠️ Active blocker (cannot self-fix)
h200 SSH tunnel **down**: nodes reachable only via `localhost:12222` (h200-1 direct, h200-0 via ProxyJump). Nothing listening on :12222, no helper/autossh process. **User must bring the port-forward up.** A background waiter polls `ssh h200-1 true`; once green, recon resumes autonomously.

### Long pole once tunnel is up
GLM-4.5-Air BF16 (~212 GB) download + **megatron-bridge → torch_dist conversion** (hours). Weight staging is *not* a model-selection input (user: "权重能自己下,忽略") but is real wall-clock for L3.

## Phases

## Event Projection

<!-- agent-run:projection:start -->

_Generated from `events.jsonl` by `scripts/agent_run.py project`._

### Recent Events
 - 2026-06-07T04:51:50Z | task_started | h200 2-node colocate L3 RL training of GLM-4.5-Air (106B-A12B): train EP8/PP2/TP4/CP2, rollout TP4 fp8; DAPO-Math-17k...
 - 2026-06-07T04:53:07Z | decision | plan-mode: locked /solo contract — h200 2node colocate GLM-4.5-Air L3, train TP4/PP2/CP2/EP8/ETP1, rollout TP4 fp8, D...
 - 2026-06-07T04:54:58Z | decision | GLM parser finding: vime RL rollout is token-only (/inference/v1/generate), never uses chat-completions reasoning/too...
 - 2026-06-07T05:04:40Z | decision | BLOCKER@recon: tunnel up; h200-0 FREE (8 GPU), h200-1 OCCUPIED by tiezhen vLLM dsv4-benchmark-ziming (8xTP worker ~13...
 - 2026-06-07T05:12:52Z | decision | PLATFORM PIVOT (user): h200->gb200 only (h200-1 had active tiezhen slurm job394, NOT killed). gb200 4 free nodes rack...
 - 2026-06-07T05:16:44Z | decision | FP8 MECHANISM (corrected via weight-sync KB + vime src): vime fp8 rollout = rollout --hf-checkpoint is a quantized ck...
 - 2026-06-07T05:24:10Z | phase_completed | Staging in progress: GLM-4.5-Air bf16 downloaded (47/47 shards, 206G). torch_dist convert (rack1-14, 4-GPU torchrun) ...

<!-- agent-run:projection:end -->

### Phase 1: <name>

- **Why**: <reason>
- **Inputs**: <files / prior phase output>
- **Outputs**: <concrete artifact path or fact>
- **How to verify**: <command, test, or observation>
- **Owner**: any
- **Status**: `[ ]`

  Subtasks:
  - [ ] <step 1>

## Open questions

- [ ] (none yet)
