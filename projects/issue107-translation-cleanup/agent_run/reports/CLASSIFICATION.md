# Issue #107 translation-cleanup — occurrence classification

Baseline (pre-#107 wave) = `6011ae9f` (#154, 2026-06-05). HEAD = `09450c1b`.

| term | baseline | HEAD | delta |
|---|---|---|---|
| sglang | 5 occ / 4 f | 22 / 8 | +17 |
| slime  | 29 occ / 7 f | 94 / 24 | +65 |

Spec: `agent_run/reports/sync/SGLANG_TO_VLLM_TRANSLATION.md`. Classify each delta occurrence: **FIX** (under/over/stale translation), **FLAG** (genuine §0.3 coupling — needs decision), **KEEP** (§5 provenance/attribution or accurate behavioral description).

## FIX — mechanical, spec-unambiguous (under-translation / stale)

| where | current | →fix | reason |
|---|---|---|---|
| `.github/workflows/pr-test.yml:559-562` + tests reading them | `SLIME_TEST_*` env vars | `VIME_TEST_*` | §2.6 env rename; must change code+CI together |
| `tests/test_qwen2.5_0.5B_fanout_short.py`, `_fanout_test_helpers.py`, `test_qwen3_4B_streaming_partial_rollout.py` | `SLIME_FANOUT_TEST_*`, `SLIME_TEST_TIGHT_*` | `VIME_*` | same |
| `scripts/run-minimax-m2.sh:37-38` | `MiniMax-M2.5_slime/` | `_vime/` | §2.1 path rename (matches run-qwen3-4B `_vime/`) |
| `tests/test_loss_cp_invariance.py:26` | `slime/backends/megatron_utils/loss.py` | `vime/backends/...` | §2.1 stale package path |
| `tests/test_cp_utils.py:18`, `test_metric_report.py:20`, `test_metric_report_dist.py:26,298` | "before the slime imports" | "vime imports" | stale package name |
| `vime/utils/dp_schedule.py:5` | "ray/sglang-importing modules" | "ray/vllm-importing" | under: vime imports vllm |
| `tests/test_agent_adapters.py:112,118,121` | `X-Slime-Session-Id` header | `x-session-id` | prod reads `x-session-id` (common.py:255); slime-ism |
| `tests/test_agent_adapters.py:141,159,188`, `test_agent_sdk_adapters.py:55,85,97,102` | fixture query `"slime"` / "find slime" | `"vime"` | mirror rename of test-fixture data |

## FLAG — genuine §0.3 coupling (decision needed)

- **`vime/agent/parsing.py:42,68,69,78` (+README:93)** — optionally delegates to `sglang.srt` reasoning + function-call parsers (lazy import; XML fallback by default; sglang not a declared dep). vLLM has no drop-in equivalent. Options: (a) keep as optional sglang path + document; (b) replace with vLLM parser if one exists; (c) drop the sglang branch, XML-only. **Recommend (a)** (graceful-degrade, matches "flag don't fake-translate") but needs sign-off.

## KEEP — §5 provenance / attribution / accurate behavioral description

- `docs/{en,zh}/index.rst` + `arch-support-beyond-megatron.md` (4× slime) — the #180 "built on slime" attribution (mandated).
- `README*.md` (20), `CONTRIBUTING.md` (2) — attribution (baseline, unchanged).
- `docker/Dockerfile:49,97,99` — "Mirrors slime's split", upstream `slime #1916/#1924` refs.
- `docs/*/reproducibility.md` — slime PR #370 link.
- `vime/rollout/vllm_streaming_rollout.py:20-21` — "counterpart of `slime.rollout.sglang_streaming_rollout`" provenance + behavioral contrast.
- `tests/test_qwen3_4B_streaming_partial_rollout.py:14` — "counterpart of slime's test" provenance.
- `vime/ray/rollout.py:820` — "in slime's convention" (describes inherited contract).
- `vime/agent/adapters/common.py:177,274` — "sglang-shaped" sampling dict (§3.2 deliberate normalization), `/abort_request` reference.
- `vime/utils/external_utils/command_utils.py:229` — "copied from SGLang" (baseline; attribution).
- `examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh:202-221` — didactic sglang→vLLM arg-map comment block (vime-authored translation note). **Borderline**: accurate & helpful, but 9 sglang tokens. Could trim to hit baseline count — decision.

## Net
Mechanical FIX clears most of the +65 slime (env vars + paths + fixtures dominate the test-file counts). sglang is harder: of +17, ~1 is mechanical (dp_schedule), 5 are FLAG (parsing.py), 9 are the didactic run-script comment block, rest KEEP. Hitting sglang baseline (5) requires resolving parsing.py and/or trimming the didactic block.
