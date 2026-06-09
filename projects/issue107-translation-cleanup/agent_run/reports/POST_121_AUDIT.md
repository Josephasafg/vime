# Post-#121 translation audit (9524dd52..main)

Scope: every change after the rename commit `9524dd52` (#121), 140 files / +12.5k lines.
Method: §6 bidirectional + the spec's documented bug cases. Audited on `issue107-cleanup`
HEAD (= main + PR #196).

## Result: translation-clean on every high-signal dimension

| dimension | check | verdict |
|---|---|---|
| Runtime sglang coupling | `from sglang` / `import sglang` in vime/tests/examples | **none** (only a comment "difference from sglang") |
| sglang-coupled examples (tau-bench/strands) | should be deleted per §5 | **absent** (already trimmed in #39) |
| §4.1 OVER (flush_cache gratuitous "vLLM") | log string | **fixed** (#137/#153) |
| §4.2 UNDER/genericization (agent) | `ENGINE_URL_KEY`/`call_engine_generate`/`engine_url`/`FakeEngine` | **none**; vime uses `VLLM_URL_KEY`/`call_vllm_generate`/`vllm_url`/`FakeVLLM` |
| OVER (vLLM bolted onto neutral slime strings) | wandb "vLLM metrics", common.py "vllm upstream", weight-update logs | **clean**: wandb←slime "SGLang metrics" ✓, common.py←slime "sglang upstream" ✓, weight-update logs are vime-authored (slime has no such strings) |
| Symbol genericization | `sglang_{url,args,router_ip,engine}`, `call_sglang_generate`, `add_sglang_arguments` | all have `vllm_*` counterparts; **no** `engine_url`/`engine_args`/`backend_url` leak |
| Under-translation stragglers (#107) | env vars, paths, fixtures, headers | **fixed in PR #196** |

## Remaining sglang/slime tokens are all intentional keeps
- sglang = 6: `common.py` sglang-shaped normalization (§3.2 ×2), `vllm_streaming_rollout` counterpart-of-`slime.rollout.sglang_streaming_rollout` provenance + behavioral contrast (×2), "copied from SGLang" (×1), streaming-test counterpart docstring (×1).
- slime = 36: README/CONTRIBUTING/docker attribution, upstream `slime #1916/#1924` links, reproducibility PR link, #180 "built on slime" attribution, "counterpart of slime…" provenance.

## Conclusion
After PR #196, the post-#121 diff has no detectable translation defects on the documented
bug classes or the symbol-level genericization sweep. The high-value finding of this audit —
`vime/agent/parsing.py` importing `sglang.srt` parsers — is fixed in #196 (now `vllm.reasoning`/
`vllm.tool_parsers`).
