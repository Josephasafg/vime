# CI Full-Sweep — Design Table (vime-only first, slime A/B on failure)

**Living ledger.** Target = current `origin/main` **`ae5d6b04`** (PR #178 merged: vllm_rollout aligned to slime sglang_rollout). Run host = **h200-0, x86_64, native 8-GPU** (NO 8→4 shrink). Image = `inferactinc/public:vime-latest` (70.9 GB, present). Worktree = `h200:/home/aoshen/vime/.worktree/ci-full-sweep` @ `ae5d6b04`. Harness = `vime-test-runs/_harness/ci_sweep/{run_ci_sweep.sh,ci_matrix.json}`.

## Why h200-x86 native changes expectations vs the gb200 baseline
The gb200 phase-1 baseline (`reports/ci/ci_baseline_gb200_phase1.md`) ran on **arm64 + 4-GPU/node (8→4 shrink)**. Two whole failure classes there **do not apply here**:
1. **arm64-vs-x86 FP artifacts** — the strict numeric gates (`#9` gsm8k `kl<1e-8`, `#20` parallel_check grad-norm) fail on gb200-arm but **PASS on h200-x86** (x86 is where the reference values were generated; it is the authoritative precision platform).
2. **8→4 shrink artifacts** — vllm_config family, parallel_check actor-num-gpus, moonlight_r3 EP8→EP4 split-mismatch were all caused by the gb200 4-GPU shrink. On h200 we run **native 8-GPU = exactly what CI does**, so these are moot.

⇒ On h200-x86 native, the expectation is **broadly green**; deviations are the signal worth chasing.

## Legend
✅ expect-pass · 🟢 historically-green (x86/native) · 🔶 watch (flaky / threshold / tunable) · 🧱 engine/env blocker possible · 🟥 known-issue · ⏭️ optional/excluded · ⛔ no slime A/B (vime-only)
A/B col = name of the equivalent **slime** test (sglang path) for `run_slime_ab.sh` if a vime test fails; ⛔ = vllm/vime-specific, no slime control.

---

## ▶ LIVE RESULTS — dual-host sweep 2026-06-08 (h200-0 + h200-1, NFS-shared `vime-test-runs/ci-full-sweep/<tag>/summary.tsv`)
rc from `summary.tsv` (id · exit_rc · duration). Dedup = latest/authoritative run wins (a pre-fix crash superseded by a post-fix green is shown as `1→0`).

**Tally: CPU 22/22 ✅ · GPU green 18 · fixed-and-verified 1 (#186 base) · fixes-in-flight 3 (#186-r3, #184, #166) · NEW failing group 1 (ckpt ×2).**

### ✅ GREEN (no action)
- **CPU contracts+agent (cpu-sweep-h0): 22/22 rc=0.**
- short: `short_gsm8k` 0/707s · `short_gsm8k_async` 0/683s · `short_ppo_critic` 0/246s · `short_fully_async` 0/292s
- vllm-config: `vllm_config` 0/305s · `vllm_config_dist` 0/270s · `vllm_mixed_offload` 0/361s · `vllm_mixed_offload_ft` 0/431s
- megatron dense/train: `mg_glm4_9B` 0/645s · `mg_qwen3_4B_ppo` 0/727s · `mg_qwen3_4B_ppo_disagg` 0/692s · `mg_qwen3_4B_ppo_critic` 0/643s · `mg_debug_rollout_train` 0/316s · `mg_opd_vllm` 0/279s
- megatron moe: `mg_qwen3_30B_r3` (bf16) 0/998s · `mg_moonlight_16B` 0/618s · `mg_moonlight_16B_r3` 0/671s
- precision: `prec_parallel_check` SUCC (in cleanup; rc pending write) — **green**

### 🔧 FIXED & VERIFIED — **PR #186 COMPLETE** (both FP8/deepep tests green)
- `mg_qwen3_30B_deepep_fp8` (D,F): **1→0** (238s crash → **0/1971s**).
- `mg_qwen3_30B_r3_deepep_fp8` (D,F,E0): **1→0** (236s crash → **0/1237s**).
- Fix = **PR #186** `--vllm-enable-expert-parallel` (FP8 block_n=96 reject). ✅ ready to merge.

### 🔄 FIX-IN-FLIGHT (h200-0 serial chain, after #186 r3 done; h200-1 returned to ziming)
- `mg_qwen3_4B_streaming`: pre-fix 1/315s (#178 ImportError) → **PR #184** RUNNING now (RUN_TAG vstream-184, ci-streaming-verify @16dfeb4e).
- `mg_glm47_30B_pd` + `mg_qwen36_35B_pd_deepep`: pre-fix 1/67s & 1/69s; stale-#166 also 1/410s → **main×#166** queued next (RUN_TAG vpd-166main, ci-pd166main @46ad5a8d). #173 DP/EP already in main.

### 🟥 NEW FAILING GROUP — checkpoint save (needs classify)
- `ckpt` **rc=1/534s** AND `ckpt_async` **rc=1/450s** (h200-0 sweepA chain, back-to-back).
- Error: `filesystem_async.py:326 … [enforce fail at inline_container.cc:672] . unexpected pos 704 vs 598` → `torch.distributed.checkpoint.api.CheckpointException`. Torch dist-ckpt **zip-writer corruption**, NOT ENOSPC (/home 19T free).
- **Both** sync+async failing weakens "flaky one-off"; leading hypothesis = **shared save-dir contamination** (both write `/root/models/Qwen3-4B_vime` back-to-back in the same chain; row note already flags "run isolated — writes shared dir"). Could also be a real dist-ckpt regression. **Action: isolated single re-run of `ckpt` after chains drain to classify contamination vs real.** No root-cause asserted yet (evidence-first).

> Superseded/contaminated earlier rows (ignore): `gpu-batch1-short short_gsm8k_async 1/233s` (early deadlock-era, superseded by sweepA 0/683s); `verify-pd166 … 1/410s` (stale #166 branch pre-#173, superseded by main×#166).

---

## Always-on CPU — plugin-contracts (19) + agent-adapter (3)
Run in `vime-latest` (CI runs these on ubuntu-latest); we install the CI CPU dep set (`pytest numpy … pylatexenc`) in-image per agreed option A. num_gpus=0, no model mount.

| id | test_file | what it checks | expect | A/B |
|---|---|---|---|---|
| pc_megatron_arg_validation | test_megatron_argument_validation.py | megatron arg validation | ✅ | ⛔ |
| pc_value_temperature | test_value_temperature.py | value head temperature (ported slime #1928) | ✅ (validated EXIT_RC=0) | ⛔ |
| pc_rollout_validation | test_rollout_validation.py | rollout arg validation | ✅ | ⛔ |
| pc_plugin_rollout | plugin_contracts/test_plugin_rollout_contracts.py | plugin rollout contract | ✅ | ⛔ |
| pc_plugin_runtime_hook | plugin_contracts/test_plugin_runtime_hook_contracts.py | plugin runtime-hook contract | ✅ | ⛔ |
| pc_plugin_path_loading | plugin_contracts/test_plugin_path_loading_contracts.py | plugin path-loading contract | ✅ | ⛔ |
| pc_plugin_generate | plugin_contracts/test_plugin_generate_contracts.py | plugin generate contract | ✅ | ⛔ |
| pc_rm_deepscaler | test_rm_deepscaler.py | deepscaler reward model | ✅ | ⛔ |
| pc_rm_f1 | test_rm_f1.py | f1 reward model | ✅ | ⛔ |
| pc_rm_gpqa | test_rm_gpqa.py | gpqa reward model | ✅ | ⛔ |
| pc_rm_math | test_rm_math.py | math reward model | ✅ | ⛔ |
| pc_rm_math_dapo | test_rm_math_dapo.py | dapo-math reward model | ✅ | ⛔ |
| pc_dp_schedule | test_dp_schedule.py | DP micro-batch schedule (#1926) | ✅ 8/8 | ⛔ |
| pc_cp_utils | test_cp_utils.py | CP utils reductions | ✅ | ⛔ |
| pc_metric_report | test_metric_report.py | metric report | ✅ | ⛔ |
| pc_metric_report_dist | test_metric_report_dist.py | metric report (multi-proc gloo) | ✅ | ⛔ |
| pc_loss_cp_invariance | test_loss_cp_invariance.py | CP-invariance + rollout==train one-step (#1930/#1933) | ✅ | ⛔ |
| pc_sample | test_sample.py | sampling (#1984 rename) | ✅ 16/16 | ⛔ |
| pc_hf_checkpoint_saver | utils/test_hf_checkpoint_saver.py | HF checkpoint saver | ✅ | ⛔ |
| agent_trajectory | test_agent_trajectory.py | agent trajectory (needs openai/anthropic) | ✅ | ⛔ |
| agent_adapters | test_agent_adapters.py | agent adapters | ✅ | ⛔ |
| agent_sdk_adapters | test_agent_sdk_adapters.py | agent SDK adapters | ✅ | ⛔ |

## Always-on CPU — unit_pytest (SPECIAL — runs as 1-GPU + PER-FILE isolation per user decision)
| id | how we run it | expect | note |
|---|---|---|---|
| unit_pytest | **1 GPU (gpu_lock_exec --count 1) + per-FILE pytest loop** (no model mount) | 🟢 target **192/192** | Per `reports/ci/ci_cpu_unit_gate_audit_cn.md` + verified on `vime-latest`@`ae5d6b04` this session. Single no-GPU invocation = **28 fail = 4 device-touching + 24 order-pollution**. **Root cause pinned (2026-06-08):** `tests/unit/backends/.../test_update_weight_from_tensor.py:81` runs `_install_stubs()` **at module import time**, doing `sys.modules.setdefault("megatron"/"vllm"/…)` → shadows real modules → sibling `tests/unit/backends/vllm_utils/test_vllm_engine.py` fails COLLECTION with `ModuleNotFoundError: No module named 'vllm.engine'` (verified: collected ALONE = 46 tests OK; whole tree even with `--forked` = still 1 collection error, because pytest collects all modules in ONE process before forking). ⇒ **`--forked` is insufficient; only true per-FILE invocation works** (each file = fresh process/sys.modules). slime has **no `tests/unit/` tree** → nothing to mirror; this is vime test-hygiene. **Optional follow-up fix (flag before edit):** move `_install_stubs()` out of module top-level into a fixture / `monkeypatch` so it can't pollute sibling collection — would make the single-invocation CI path green too. |

---

## GPU e2e matrix (25 runnable; native 8-GPU on h200-x86)
Columns: flags (D=deepep, F=fp8, E0=enable_eval=0) · type · what it tests · h200-x86 native expectation (+ history) · slime A/B test.

### short (4-GPU) — `run-ci-short`
| id | test | flags | type | tests | expect (history) | A/B |
|---|---|---|---|---|---|---|
| short_gsm8k_async | test_qwen3.5_0.8B_gsm8k_async_short | — | e2e-smoke | async gsm8k short train | 🟢 gb200 green | test_qwen3.5_0.8B_gsm8k_async_short.py |
| short_gsm8k | test_qwen3.5_0.8B_gsm8k_short | — | precision-gate | step-0 `kl_loss<1e-8` (rollout↔train logprob) | ✅ **x86 green** (gb200-arm fails = platform) | test_qwen3.5_0.8B_gsm8k_short.py |
| short_ppo_critic | test_qwen2.5_0.5B_ppo_critic_only_short | — | e2e-smoke | PPO critic-only short | 🟢 green | ⛔ (vllm-named) |
| short_fully_async | test_qwen2.5_0.5B_fully_async_short | — | e2e-smoke | fully-async short (#116) | 🟢 green | ⛔ |

### vllm-config (8-GPU) — `run-ci-vllm-config` (all vime/vllm-specific ⇒ no slime A/B)
| id | test | type | tests | expect (history) | A/B |
|---|---|---|---|---|---|
| vllm_config | test_qwen2.5_0.5B_vllm_config | engine-config | vLLM config matrix | 🟢 native (gb200 fail = shrink) | ⛔ |
| vllm_config_dist | test_qwen2.5_0.5B_vllm_config_distributed | engine-config | distributed vLLM config | 🟢 native (gb200 hang = 8→4 placement) | ⛔ |
| vllm_mixed_offload | test_vllm_config_mixed_offload | engine-config | mixed offload | 🟢 native | ⛔ |
| vllm_mixed_offload_ft | test_vllm_config_mixed_offload_ft | engine-config | mixed offload fault-tolerant | 🟢 native | ⛔ |

### megatron (8-GPU) — `run-ci-megatron`
| id | test | flags | type | tests | expect (history) | A/B |
|---|---|---|---|---|---|---|
| mg_glm4_9B | test_quick_start_glm4_9B | — | e2e-dense | GLM-4 9B quick start (model = GLM-Z1-9B-0414) | 🟢 green | test_quick_start_glm4_9B.py |
| mg_glm47_30B_pd | test_glm4.7_30B_A3B_pd_mooncake | — | heavy-PD | GLM-4.7-30B PD + mooncake | 🧱 PD/mooncake/NIXL env (gb200 not-run; h200 RoCE needs `UCX_NET_DEVICES=^mlx5_0:1`) | test_glm4.7_30B_A3B_pd_mooncake.py |
| mg_qwen3_30B_deepep_fp8 | test_qwen3_30B_A3B | D,F | heavy-moe | 30B deepep + FP8 rollout | 🟢 **FIXED PR#128+#133** (e2e green, logprobs agree) | test_qwen3_30B_A3B.py |
| mg_qwen36_35B_pd_deepep | test_qwen3.6_35B_A3B_pd_mooncake | D | heavy-PD | 35B PD + mooncake + deepep | 🧱 PD/mooncake (gb200 contract-excluded) | test_qwen3.6_35B_A3B_pd_mooncake.py |
| mg_qwen3_30B_r3_deepep_fp8 | test_qwen3_30B_A3B_r3 | D,F,E0 | heavy-moe-r3 | r3 routing-replay + DeepEP + FP8 | 🟢 **MEASURED GREEN** (in-tree per-engine-2 → 384%128 ok; Hopper auto≠TRTLLM so capture fires) | test_qwen3_30B_A3B_r3.py |
| mg_qwen3_30B_r3 | test_qwen3_30B_A3B_r3 | E0 | heavy-moe-r3 | r3 bf16 | 🟢 (PR#178 bf16 r3 passed on h200 earlier) | test_qwen3_30B_A3B_r3.py |
| mg_qwen3_4B_ppo | test_qwen3_4B_ppo | — | e2e-train | 4B PPO (variable global batch + per-token-loss path) | 🟢 green | test_qwen3_4B_ppo.py |
| mg_qwen3_4B_ppo_disagg | test_qwen3_4B_ppo_disaggregate | — | e2e-train | 4B PPO disaggregate | 🟢 green | test_qwen3_4B_ppo_disaggregate.py |
| mg_qwen3_4B_ppo_critic | test_qwen3_4B_ppo_train_critic_only | — | e2e-train | 4B PPO train-critic-only | 🟢 green | test_qwen3_4B_ppo_train_critic_only.py |
| mg_qwen3_4B_streaming | test_qwen3_4B_streaming_partial_rollout | — | e2e-train | streaming partial rollout (#117) | 🟢 green | test_qwen3_4B_streaming_partial_rollout.py |
| mg_moonlight_16B | test_moonlight_16B_A3B | — | heavy-moe | Moonlight-16B-A3B (MLA) | 🟢 green | test_moonlight_16B_A3B.py |
| mg_moonlight_16B_r3 | test_moonlight_16B_A3B_r3 | E0 | heavy-moe-r3 | Moonlight-16B r3 | 🔶 native EP8 (gb200 EP8→EP4 split = shrink); threshold-tight (missing slime #1975) → occasional -0.5 graze | test_moonlight_16B_A3B_r3.py |
| mg_debug_rollout_train | test_qwen2.5_0.5B_debug_rollout_then_train | — | e2e-smoke | debug rollout-then-train | 🟢 green | ⛔ |
| mg_opd_vllm | test_qwen2.5_0.5B_opd_vllm | — | e2e-train | OPD over vLLM (#141) | 🟢 **FIXED PR#141** (text + MM) | ⛔ (vllm-specific OPD) |

### precision (8-GPU) — `run-ci-precision`
| id | test | type | tests | expect (history) | A/B |
|---|---|---|---|---|---|
| prec_parallel_check | test_qwen3_0.6B_parallel_check | precision-gate | DP/TP/CP grad-norm invariance (tol 0.01) | ✅ **x86 authoritative** (gb200-arm + shrink = artifact; x86 ref). 🔶 historically data-dependent flake → run ≥2× if it trips | test_qwen3_0.6B_parallel_check.py |

### ckpt (8-GPU) — `run-ci-ckpt`
| id | test | flags | type | tests | expect | A/B |
|---|---|---|---|---|---|---|
| ckpt | test_qwen3_4B_ckpt | — | e2e-ckpt | save/load iter1/2 | 🟥 **rc=1/534s** `inline_container.cc:672 unexpected pos` (see LIVE RESULTS; hypothesis = shared-dir contamination, re-run isolated to classify) | test_qwen3_4B_ckpt.py |
| ckpt_async | test_qwen3_4B_ckpt | --async-save | e2e-ckpt | async-save flush | 🟥 **rc=1/450s** same `CheckpointException` as ckpt (ran back-to-back, same save dir) | test_qwen3_4B_ckpt.py --async-save |

---

## Optional / excluded
| set | why | action |
|---|---|---|
| e2e-test-image (13) | duplicate subset pinned to `inferactinc/public:vime-test-latest` (**absent on h200-0**) | ⏭️ skip unless image pulled; redundant with the megatron/short/ckpt rows above |
| PD rows (mg_glm47_30B_pd, mg_qwen36_35B_pd) | PD + mooncake/NIXL; gb200 baseline marked "contract-excluded / not-run" | run last; expect possible NIXL/RoCE env blocker (not a code regression) — see memory `project_pd_nixl_roce_blocker` |

## Run plan (after user confirm)
1. **CPU first** (cheap, no GPU contention): `run_ci_sweep.sh cpu` → 22 contract/agent entries green; unit_pytest per the decision above.
2. **GPU tiers** (gpu_lock_exec serializes on the shared host): short → vllm-config → precision → ckpt → megatron (4B before 30B/PD). `gpu` selector or per-group.
3. **On any failure** → `run_slime_ab.sh <slime_test> <gpus>` (A/B col) to classify vime-regression vs shared/env. Then `bisect_ci.sh <id> <good> <bad>` if vime-specific.
4. Faithfulness substitutions vs CI: models/datasets `overlay` (lower=models-shared ro, tmpfs upper) + HF-cache mount (no re-download); TORCHINDUCTOR cache → NFS; CPU dep install in-image. Everything else byte-identical to `pr-test.yml.j2 @ ae5d6b04`.
