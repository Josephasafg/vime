# Coding-Agent RL (Modal + Claude Code) — Training Handoff

**Date:** 2026-06-07
**Author:** opus-4-8 (handing off to next agent)
**Task:** Run end-to-end multi-turn coding-agent RL training on vime, using
Modal sandboxes + Claude Code, on the gb200 or h200 fleet.
**Issue tracker:** https://github.com/vllm-project/vime/issues/107 (slime 3.0 sync)

---

## 1. What This Is

SWE (software-engineering) coding-agent RL: a real coding agent (Claude Code CLI)
runs inside a per-sample Modal sandbox, produces a `git diff`, which is graded against
the dataset's test harness in a second clean sandbox. The model sees Anthropic Messages
API; under the hood an in-process `AnthropicAdapter` tokenizes each turn, calls the
vLLM rollout engine via `/inference/v1/generate`, records exact token ids + logprobs,
and emits `TokenSegment`s for training with proper `loss_mask` (model outputs = 1,
observations/template = 0).

### Architecture

```
   Dataset (SWE JSONL)
         │
   ┌─────▼──────────────────────────────────────────────┐
   │  generate.py  (--custom-generate-function-path)    │
   │    ├─ AnthropicAdapter (serves /v1/messages)       │
   │    │    └─ calls vLLM /inference/v1/generate       │
   │    │         (input_ids, return_logprob=True)       │
   │    ├─ sandbox.run_claude_code  (Modal sandbox)     │
   │    │    Claude Code ──HTTP──▶ adapter (via tunnel)  │
   │    ├─ sandbox.git_diff        (capture patch)      │
   │    ├─ sandbox.evaluate        (clean sandbox)      │
   │    └─ _merge_samples -> list[Sample] (fan-out)     │
   └────────────────────────────────────────────────────┘
         │
   train.py (Megatron GRPO/GSPO)
```

---

## 2. Current State (what's done vs what you need to do)

| Component | Status | Notes |
|-----------|--------|-------|
| `vime/agent/` subsystem (adapter + trajectory + sandbox protocol) | ✅ **merged** (PR #148, 2026-06-07) | On vime `main`. Fully vLLM-native: `VLLM_URL_KEY`, `/inference/v1/generate`, `vllm_tool_call_parser`/`vllm_reasoning_parser`. |
| `examples/coding_agent_rl/` (generate.py, sandbox.py, run script) | ✅ **merged** (PR #148) | On `main`. Already uses `args.vllm_router_ip`/`port`, `args.vllm_tool_call_parser`, `args.vllm_reasoning_parser`. |
| Modal sandbox backend (`ModalSandbox`) | ⏳ **PR #167 open** (retarget main, merge) | 36/36 unit + 13/13 real-Modal e2e. Adds `VIME_AGENT_SANDBOX_BACKEND=modal` + `ADAPTER_URL_OVERRIDE`. |
| slime 3.0 sync remaining | #143 (mega-B), #145 (mega-A) still open | B = models/placement; A = data-model with `group_ids`. A is needed for fan-out segment counting. **Check if merged by the time you start.** |
| vLLM `qwen3_coder` tool-call parser | ✅ registered | `vllm/tool_parsers/__init__.py` line 157 |
| vLLM `qwen3` reasoning parser | ✅ registered | `vllm/reasoning/__init__.py` line 107 |
| vLLM Anthropic-native endpoint (`/v1/messages`) | ✅ exists | `vllm/entrypoints/anthropic/` — but NOT used here; the adapter serves its own `/v1/messages` and calls vLLM's `/inference/v1/generate` internally |

---

## 3. Prerequisites to Prepare

### 3.1 Merge PR #167 (Modal sandbox)

```bash
gh pr edit 167 --repo vllm-project/vime --base main   # retarget from sync/slime-mega-D to main
# resolve any merge conflicts (likely none — purely additive files)
gh pr merge 167 --repo vllm-project/vime --squash
```

Then pull main on the run host:
```bash
cd /path/to/vime && git pull origin main
```

### 3.2 Model: Qwen3.6-35B-A3B

The slime reference script uses **Qwen3.6-35B-A3B** (MoE, 256 experts, top-8, 40 layers).
This is the model with matching `qwen3_coder` tool-call parser + `qwen3` reasoning parser.

**Download + convert:**

```bash
# 1. Download HF weights (~67 GB)
huggingface-cli download Qwen/Qwen3.6-35B-A3B --local-dir /path/to/Qwen3.6-35B-A3B

# 2. Convert to torch_dist (megatron checkpoint for --ref-load)
cd /path/to/vime
source scripts/models/qwen3.5-35B-A3B.sh   # <-- NOTE: qwen3.6 uses qwen3.5 bridge mapping
PYTHONPATH=/root/Megatron-LM/ torchrun --nproc-per-node 4 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /path/to/Qwen3.6-35B-A3B/ \
   --save /path/to/Qwen3.6-35B-A3B_torch_dist/
```

> **⚠️ qwen3.6 config:** vime's `scripts/models/` has NO `qwen3.6-35B-A3B.sh`. The
> test `tests/test_qwen3.6_35B_A3B_pd_mooncake.py` uses MODEL_TYPE `qwen3.5-35B-A3B`
> via the bridge mapping. miles has `qwen3.6-35B-A3B.sh` — copy it from
> `reference/miles/scripts/models/qwen3.6-35B-A3B.sh` into vime's `scripts/models/`
> if needed, OR use `scripts/models/qwen3.5-35B-A3B.sh` (architecturally identical;
> bridge handles naming).

**What's already staged on lustre (gb200):**
- `/mnt/lustre/aoshen/models/` may have Qwen3.6-35B-A3B (check `ls /mnt/lustre/hf-models/hub/models--Qwen--Qwen3.6*`)
- If not: download, convert inside a container (needs GPU for torchrun)

### 3.3 SWE Dataset

Standard vime JSONL with three keys per row:

```jsonc
{
  "prompt": "<fallback if metadata.problem_statement is missing>",
  "label": "<instance_id>",
  "metadata": {
    "image": "swedev/scaleswe.oh.34:<tag>",   // sandbox image reference
    "workdir": "/workspace/<repo>",            // repo path inside the sandbox
    "problem_statement": "<issue body>",
    // exactly one grader:
    "swepro": { /* SWE-bench Pro test harness — preferred */ },
    // OR:
    "eval_cmd": "pytest -x tests/..."          // exit 0 = solved
    // OR (sweb-style rows):
    // metadata.remote_env_info.f2p_script — auto-wrapped into eval_cmd
  }
}
```

Wire with: `--input-key prompt --label-key label --metadata-key metadata`

**Where to get it:**
- SWE-bench-Lite / SWE-bench-Verified / SWE-bench Pro — convert to the above JSONL format
- The `image` field must reference a docker image accessible from the sandbox runtime
- Set `SWE_SANDBOX_IMAGE_METADATA_KEY` to the metadata key holding the image name (default: `glm-platform/image` in slime; override if your JSONL uses just `image`)

### 3.4 Sandbox Image

The sandbox needs: the repo pre-cloned at `workdir`, test dependencies installed,
Python/conda env ready to run tests. Standard SWE-bench images work.

**For Modal:** the image must be pullable by Modal's infrastructure. Options:
- **Public DockerHub** image (simplest): `swedev/scaleswe.oh.34:tag`
- **Private registry**: set `DOCKER_USERNAME` + `DOCKER_PASSWORD` env vars; ModalSandbox
  creates a `modal.Secret` for registry auth

**For E2B** (if NOT using Modal): set `E2B_API_KEY` + `SWE_SANDBOX_METADATA_FILE`
(JSON dict of routing tags).

### 3.5 Host-Side Tarballs (REQUIRED)

Each sandbox at boot gets two tarballs uploaded from the head node:

| Env var | What | Where to get |
|---------|------|--------------|
| `SWE_HOST_NODE_TARBALL` | Node.js 22 (`node-v22.x-linux-x64.tar.xz`) | https://nodejs.org/dist/v22.20.0/node-v22.20.0-linux-x64.tar.xz |
| `SWE_HOST_CC_TARBALL` | Claude Code CLI npm tarball (`anthropic-ai-claude-code-local-linux-x64.tgz`) | `npm pack @anthropic-ai/claude-code-local-linux-x64` (requires npm auth / Anthropic access) |

Download both to the head node and export the env vars pointing to their paths.

### 3.6 Modal Credentials

```bash
export MODAL_TOKEN_ID="ak-QJwl8VavnZNhVsAlwms0Qz"
export MODAL_TOKEN_SECRET="as-QxiOjaMQcGGW0182XHKu6F"
export VIME_AGENT_SANDBOX_BACKEND=modal
```

Modal SDK version: **1.4.2** (test with `python -c "import modal; print(modal.__version__)"`)

### 3.7 Reverse-Network Tunnel (CRITICAL)

Claude Code runs INSIDE the sandbox and dials back to the head's Anthropic adapter.
The head must be reachable from inside Modal sandboxes. **Direct IP does NOT work**
(NAT egress-only, empirically verified — see HANDOFF.md §3).

**Solution: cloudflared tunnel on the head node:**

```bash
# Start BEFORE the training run:
cloudflared tunnel --url http://localhost:${SHIM_PORT:-18001} --no-autoupdate &
sleep 15   # wait for edge registration (important: edge warmup race!)
# Extract the public URL:
ADAPTER_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /path/to/cf.log)
export ADAPTER_URL_OVERRIDE="$ADAPTER_URL"
```

The adapter URL is passed into sandboxes as `ANTHROPIC_BASE_URL` so Claude Code
dials back through the tunnel.

**⚠️ Gotcha (empirically discovered):** cloudflared prints its URL before the edge route
is fully registered. A fast-booting cached sandbox may curl before the edge is live →
transient 502. The e2e harness polls `_wait_tunnel_ready()` before creating sandboxes.
For production: use a **named tunnel** (`cloudflared tunnel create ...`) with a stable
hostname instead of the quick-tunnel random URL.

---

## 4. Harness Configuration

### 4.1 Key vime CLI flags for coding_agent_rl

```bash
ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --num-rollout 100
   --rollout-batch-size 8          # concurrent sandboxes; lower if GPU OOM from fan-out
   --n-samples-per-prompt 8
   --rollout-max-context-len 96000 # multi-turn prompt+response budget
   --rollout-max-response-len 32768 # per-turn generation cap -> max_new_tokens
   --rollout-temperature 1.0
   --rollout-stop-token-ids 248046 248044   # Qwen3.6 stop tokens (NOT Qwen3/GLM)
   --num-steps-per-rollout 1
   --global-batch-size 64
   --micro-batch-size 1
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)
```

### 4.2 vLLM backend args (translated from slime's sglang args)

```bash
BACKEND_ARGS=(
   # Rollout engine: TP8 per engine (one engine = one node), high mem for long contexts
   --rollout-num-gpus-per-engine 8
   --vllm-gpu-memory-utilization 0.75

   # Parsers (REQUIRED for coding_agent_rl — adapter uses them to parse model output):
   --vllm-tool-call-parser qwen3_coder
   --vllm-reasoning-parser qwen3

   # MTP speculative decoding (if vLLM supports it for this model; else omit):
   # --vllm-speculative-config '{"method":"mtp","num_speculative_tokens":5}'

   # Routing (consistent_hash avoids the cache_aware degeneration bug):
   --vllm-router-policy consistent_hash
   --vllm-server-concurrency 128

   # MoE backend:
   --vllm-all2all-backend deepep_high_throughput
)
```

### 4.3 Parser translation table (sglang → vime/vLLM)

| slime/sglang flag | vime/vLLM equivalent | Notes |
|---|---|---|
| `--sglang-tool-call-parser qwen3_coder` | `--vllm-tool-call-parser qwen3_coder` | Both registered; same name |
| `--sglang-reasoning-parser qwen3` | `--vllm-reasoning-parser qwen3` | Both registered; same name |
| `--sglang-mem-fraction-static 0.75` | `--vllm-gpu-memory-utilization 0.75` | Different flag name |
| `--sglang-enable-dp-attention --sglang-dp-size N --sglang-ep-size M` | vLLM DP wiring via PR #173 (if merged) or omit | vLLM's DP attention is different |
| `--sglang-speculative-algorithm EAGLE --num-steps 3` | `--vllm-speculative-config '{"method":"mtp","num_speculative_tokens":5}'` | Check vLLM MTP support for Qwen3.6 |
| `--prefill-num-servers 1` | P/D disagg via PR #166 (if merged) or omit | Not required for MVP |

### 4.4 Model parallelism (train side)

The slime reference uses (for 8 nodes × 8 GPU = 64):
- TP2 / PP1 / CP8 / EP8 / ETP1

Scale to your fleet. Example for gb200 4 nodes × 4 GPU = 16:
- **TP2 / PP1 / CP2 / EP8 / ETP1** (DP1, world 4 per node, 16 total)
- or **TP4 / PP1 / CP2 / EP4 / ETP1** if EP8 cross-node is problematic (use `alltoall` dispatcher)

### 4.5 SWE / Claude Code env knobs

```bash
export VIME_AGENT_SANDBOX_BACKEND=modal   # or e2b
export VIME_HEAD_HOST="${MASTER_ADDR}"     # OR use ADAPTER_URL_OVERRIDE for tunneled
export SHIM_BIND_HOST="0.0.0.0"
export SHIM_PORT=18001

export SWE_HOST_NODE_TARBALL="/path/to/node-v22.20.0-linux-x64.tar.xz"
export SWE_HOST_CC_TARBALL="/path/to/anthropic-ai-claude-code-local-linux-x64.tgz"

export SWE_TIME_BUDGET_SEC=1800           # per agent run (30 min)
export SWE_EVAL_TIMEOUT_SEC=600           # per eval test execution
export SWE_BOOT_CONCURRENCY=6             # concurrent sandbox boots

# Claude Code CLI flags (register investigator sub-agent, disable web tools):
SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":80000}'
AGENTS_JSON='{"investigator":{"description":"Searches the repo for relevant files before any edit","prompt":"You are an investigator sub-agent. Use Grep/Read/Glob to find every file relevant to the user task, then return a short bulleted summary. Do NOT edit anything.","tools":["Grep","Read","Glob"]}}'
export SWE_CLAUDE_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands --agents '${AGENTS_JSON}' --disallowedTools WebFetch WebSearch"
```

### 4.6 Modal-specific env knobs

```bash
export MODAL_TOKEN_ID="ak-QJwl8VavnZNhVsAlwms0Qz"
export MODAL_TOKEN_SECRET="as-QxiOjaMQcGGW0182XHKu6F"
export VIME_AGENT_SANDBOX_MODAL_APP="vime-coding-agent-sandbox"
export VIME_AGENT_SANDBOX_MODAL_CPU=2.0
export VIME_AGENT_SANDBOX_MODAL_MEMORY_MB=8192
export VIME_AGENT_SANDBOX_MODAL_BLOCK_NETWORK=false   # MUST be false for reverse tunnel
export VIME_AGENT_SANDBOX_LIFETIME_SEC=3600
# Private registry (if sandbox image is not public):
# export DOCKER_USERNAME=xxx
# export DOCKER_PASSWORD=xxx
```

---

## 5. Step-by-Step Execution Plan

### Step 1: Pull latest main + verify the agent subsystem exists

```bash
cd /path/to/vime
git pull origin main
ls vime/agent/adapters/anthropic.py     # must exist (PR #148 merged)
ls examples/coding_agent_rl/generate.py # must exist
```

If `vime/agent/` doesn't exist, PR #148 hasn't merged yet — wait or cherry-pick.

### Step 2: Merge PR #167 (Modal sandbox) if not yet merged

```bash
gh pr view 167 --repo vllm-project/vime --json state
# If OPEN: retarget to main and merge
```

### Step 3: Stage model weights

Check what's available:
```bash
ls /mnt/lustre/hf-models/hub/models--Qwen--Qwen3.6*     # gb200
ls /mnt/lustre/aoshen/models/Qwen3.6*                     # gb200
```

If missing, download + convert (see §3.2).

### Step 4: Prepare SWE dataset

Place a JSONL file with the schema from §3.3 at an NFS-visible path.

### Step 5: Download Node.js + Claude Code tarballs

```bash
# Node.js 22
wget https://nodejs.org/dist/v22.20.0/node-v22.20.0-linux-x64.tar.xz -O /path/to/node-v22.tar.xz

# Claude Code CLI (requires access)
npm pack @anthropic-ai/claude-code-local-linux-x64
mv anthropic-ai-claude-code-local-linux-x64-*.tgz /path/to/cc.tgz
```

### Step 6: Start cloudflared tunnel on head node

```bash
cloudflared tunnel --url http://localhost:18001 --no-autoupdate > /tmp/cf.log 2>&1 &
sleep 15
export ADAPTER_URL_OVERRIDE=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cf.log)
echo "ADAPTER_URL_OVERRIDE=$ADAPTER_URL_OVERRIDE"
# Verify it's reachable:
curl -s "$ADAPTER_URL_OVERRIDE" | head    # should connect (may 404 — that's fine, adapter not up yet)
```

### Step 7: Write & run the launch script

Use the slime reference `run_qwen36_35b_a3b_swe_8nodes.sh` as a template.
Translate `SGLANG_ARGS` → `BACKEND_ARGS` per §4.2-§4.3. Key changes:

1. Replace all `--sglang-*` flags with `--vllm-*` equivalents
2. Remove sglang-only flags (MTP EAGLE, dp-attention, prefill-servers) initially
3. Add `--colocate` if using GPU time-sharing
4. Set `VIME_AGENT_SANDBOX_BACKEND=modal` + all Modal/tunnel env vars
5. The `--moe-token-dispatcher-type` in `MISC_ARGS` should be `alltoall` for
   cross-node EP (gb200) or `flex` with `--moe-enable-deepep` for intra-node EP

### Step 8: Smoke test (L1)

Run with `--num-rollout 2 --rollout-batch-size 2 --n-samples-per-prompt 1` to verify:
- [ ] vLLM rollout engines come up with `qwen3_coder` tool parser
- [ ] AnthropicAdapter starts on `SHIM_PORT`
- [ ] Modal sandbox boots, Claude Code runs, dials back via tunnel
- [ ] `git diff` captured, evaluator runs, reward computed
- [ ] Training step executes with TokenSegment samples
- [ ] `VERIFY-D abs-diff` is bounded

### Step 9: L3 convergence run

Scale up `NUM_ROLLOUT`, `rollout_batch_size`, `n_samples_per_prompt` to production
values. Monitor wandb reward curves + abs-diff.

---

## 6. Known Risks & Gotchas

### 6.1 Adapter is NOT the vLLM Anthropic endpoint

The adapter in `vime/agent/adapters/anthropic.py` serves its own `/v1/messages`
(string-in, token-out). It internally calls vLLM's `/inference/v1/generate` (token-only).
Do NOT confuse it with vLLM's built-in `vllm/entrypoints/anthropic/` which is a
different, end-user-facing Anthropic Messages endpoint.

### 6.2 Parser requirement is different from math-GRPO

For **math GRPO** (DAPO-Math, deepscaler), vime rollout is token-only — no parsers needed.
For **coding_agent_rl**, the adapter DOES use parsers (`tool_parser` + `reasoning_parser`)
to parse model output into thinking/visible/tool_uses blocks. Without them, tool calls
won't be parsed → Claude Code will break.

### 6.3 Fan-out & memory

`generate()` returns `list[Sample]` — one per trajectory **segment** (subagent/wipe/final).
Sub-agent dispatch increases K (each completed Agent turn block → its own segment).
The effective batch after flatten can be >> `rollout_batch_size * n_samples_per_prompt`.
If OOM: lower `rollout_batch_size` or `n_samples_per_prompt`, NOT `max-tokens-per-gpu`.

### 6.4 cloudflared quick-tunnel edge warmup race

cloudflared prints its URL before the edge route registers. A fast-booting cached sandbox
may curl before the edge is live → transient 502. Poll the URL before creating sandboxes.

### 6.5 Model config: qwen3.6 uses qwen3.5 bridge

There is no `scripts/models/qwen3.6-35B-A3B.sh` in vime. The bridge maps qwen3.6 → qwen3.5
architecture. Use `qwen3.5-35B-A3B.sh` or copy from
`reference/miles/scripts/models/qwen3.6-35B-A3B.sh`.

### 6.6 Lustre ENOSPC (gb200 fleet)

`/mnt/lustre` is at 97% (1.1T free). OST0-3 are nearly full. For large writes
(checkpoints, model staging), use `lfs setstripe -c 8 -o 8,9,10,11,12,13,14,15`
on the output dir to stripe across emptier OSTs.

### 6.7 Docker permissions on gb200 nodes

Only rack1-01/02/03/04/14/15/16/17 have `aoshen` in the `docker` group.
Nodes 05/06/07/08 need `sudo usermod -aG docker aoshen` (passwordless sudo works).

### 6.8 Stale ray sessions on gb200 nodes

Prior ray runs leave `/tmp/ray` on the host. The container uses `--network host` so
ray port 6379 collides. **Always** `rm -rf /tmp/ray` on the host before launching.

---

## 7. File References

| File | Purpose |
|------|---------|
| `vime/agent/adapters/anthropic.py` | AnthropicAdapter (serves /v1/messages, calls vLLM /inference/v1/generate) |
| `vime/agent/adapters/common.py` | `BaseAdapter`, `call_vllm_generate()`, `VLLM_URL_KEY` |
| `vime/agent/trajectory.py` | `TokenSegment`, `TurnRecord`, `merge_turn_segments`, `fan_out_sample_segments` |
| `vime/agent/sandbox.py` | `Sandbox` protocol, `E2BSandbox`, `ModalSandbox` (PR #167) |
| `vime/agent/parsing.py` | `parse_model_output` (uses vLLM's tool/reasoning parsers) |
| `examples/coding_agent_rl/generate.py` | Per-sample generate function (4-stage orchestrator) |
| `examples/coding_agent_rl/sandbox.py` | `run_claude_code`, `git_diff`, `evaluate`, sandbox helpers |
| `examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh` | Reference launcher (slime-native; translate sglang→vllm) |
| `projects/vime_modal_sandbox/HANDOFF.md` | Modal sandbox implementation handoff (credentials, Route A.2, e2e results) |
| `reference/slime/examples/coding_agent_rl/` | Upstream slime reference (sglang-native) |

---

## 8. Credentials Summary

| Credential | Value | Source |
|------------|-------|--------|
| Modal token ID | `ak-QJwl8VavnZNhVsAlwms0Qz` | HANDOFF.md §2 |
| Modal token secret | `as-QxiOjaMQcGGW0182XHKu6F` | HANDOFF.md §2 |
| WANDB API key | in `projects/vime_gb200_training/secrets.env` (chmod 600) | Guard with `set +x`/`set -x` |
| HF token | `~/.cache/huggingface/token` | Present on gb200-rack1-01 |
| cloudflared | `/usr/local/bin/cloudflared` | Pre-installed on gb200-rack1-01 |

---

## 9. Host / Fleet Info

| Host | Role | GPUs | Docker | Notes |
|------|------|------|--------|-------|
| gb200-rack1-01 | edit host / registry | 4× GB200 | ✅ | cluster-registry container; NFS `/mnt/lustre` shared |
| gb200-rack1-02..17 | compute nodes | 4× GB200 each | ✅ (01-04, 14-17) | ssh aliases `gb200-rack1-NN`; enp0s3 = mgmt net; NCCL_CUMEM_ENABLE=1 for MNNVL |
| h200-0 | compute | 8× H200 | ✅ | ProxyJump via h200-1; needs tunnel on localhost:12222 |
| h200-1 | compute | 8× H200 | ✅ | localhost:12222 tunnel; currently has tiezhen's slurm reservation (job 394, active) |

**Docker image:** `192.168.0.101:5000/aoshen/vime-vllm:flashqla-pb-arm` (arm64, gb200).
For h200 (x86): use `vime-vllm-cu129-sync1916` or equivalent x86 image.

---

## 10. What's NOT in This Handoff (Future Work)

1. **MTP/EAGLE speculative decoding** — slime uses sglang's EAGLE spec-decode for Qwen3.6.
   vLLM has MTP support (`--speculative-config '{"method":"mtp",...}'`); needs testing for Qwen3.6.
2. **P/D disaggregation** — slime uses `--prefill-num-servers 1`. vime has PR #166 (open)
   for vLLM PD disagg via vllm-router. Skip for MVP.
3. **DP attention** — slime uses sglang's `--enable-dp-attention --dp-size 8`. vime's vLLM
   DP wiring (PR #173) is different. Skip or adapt per PR #173's API.
4. **Production tunnel** — replace quick-tunnel (random URL, rotates on restart) with a
   named cloudflared tunnel for stability.
5. **SWE-bench Pro dataset** — if you don't have access, start with SWE-bench Lite
   (public, smaller, well-tested).
