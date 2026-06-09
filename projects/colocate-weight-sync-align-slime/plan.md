# Align colocate IPC weight-sync to slime

## PR-A — drop redundant sleep/resume  ✅ MERGED (#169)

**Status:** merged to main as `f6e49c7` (squash of `a0c4905`). Worktree
`projects/colocate-weight-sync-align-slime/vime` is now stale (can be removed).

Dropped `release_memory_occupation(level=0)`→`flush_cache()` and removed the paired
`resume_memory_occupation(...)` in the colocate branch; corrected the
`release_memory_occupation` docstring; +`strict=True` B905 fix. See §Why below.

## PR-B — full structural refactor to slime skeleton  🟡 PR #170 (integration pending)

**PR:** https://github.com/vllm-project/vime/pull/170 (`9eee779`, branch
`refactor/colocate-weight-sync-mirror-slime`, base merged main `f6e49c7`)
**Worktree:** `projects/colocate-weight-sync-align-slime/vime-refactor`
**Files:** `update_weight_from_tensor.py` (−60 net) + its unit test.
**Status:** pre-commit clean; **7/7 unit tests pass** (py312 env, isolated — the 19
`test_update_weight_from_distributed.py` failures are pre-existing on main: identical
`19 failed/7 passed` with PR-B stashed → cross-file sys.modules stub pollution + CUDA
env, not PR-B). User is running colocate integration testing.

Beyond the earlier plan, PR-B also made `_send_hf_params` return `(refs, long_lived_tensors)`
and `_send_to_colocated_engine` return `(refs, weight_refs)` with **no internal ray.get /
per-slot barrier** — synchronization is the all_gather_object collective deferral +
after-loop barrier (+ receiver-side accelerator.synchronize), exactly like slime. This
was NOT a vLLM requirement; the earlier "internal-sync" claim was a conservative
preservation of main's pattern, now corrected to mirror slime.

### What changed
- **`connect_rollout_engines`**: now structurally identical to slime — `use_distribute`
  flag; rename `_colocated_engines`→`rollout_engines`, `_distributed_engines`→
  `distributed_rollout_engines`; collapse the `_ipc_engine_coordinator`/`_slot_start`/
  `_slot_end`/`_ipc_slot_group` quartet into slime's `_ipc_gather_group`+`_ipc_gather_src`
  (leader == `rank == _ipc_gather_src`, slot_size == `get_world_size(group)`); guarded
  single-pass group creation; dropped the unused `rollout_engine_lock` store.
- **`update_weights`**: concise slime skeleton — pause/flush/continue act on
  `self.rollout_engines` only (distributed pause/flush/continue **removed**; the
  distributed *send* stays, folded into `_send_hf_params`). Added `torch.cuda.ipc_collect()`
  (per-chunk + after-loop), matching slime. **Removed the per-chunk global barrier**
  (slime has none) and restored slime's single after-loop barrier.
- Extracted **`_send_hf_params`** (colocated + distributed dispatch) and renamed the
  colocated sender to **`_send_to_colocated_engine`**; internals unchanged
  (all_gather_object + UUID merge + per-slot barrier — the genuine vLLM-IPC divergence).
- Comments trimmed to slime's density/wording; no `slime`/`sglang`/`mirror` phrasing in code.

### Irreducible vLLM divergences (kept, encapsulated)
1. `_send_to_colocated_engine` internals (reduce_tensor UUID handles + all_gather_object
   + merge) vs slime's FlattenedTensorBucket+gather_object — forced by vLLM IPC + the
   Megatron-reload monkey-patch (gather_object unsupported there).
2. vLLM #39212 state machine (`init`/`start`/`finish_weight_update`) — no sglang analog.
3. colocated send self-synchronizes (internal ray.get + per-slot barrier), so
   `_send_hf_params` returns only the distributed refs.

### ⚠️ Behavioral changes — MUST validate on real colocate before merge
- **Per-chunk barrier removed.** Concurrency change; unit tests can't catch a hang/race.
- **`pop_metrics` skipped** (dead in vime — not wired anywhere; adding it = dead code).
- Distributed pause/flush/continue removed (dead path in colocate: overflow-distributed
  is provably empty when `args.colocate` selects this class; see wiring checks).

### Gate
Run a colocate smoke (h200 or gb200, Qwen3-30B-A3B colocate dp2×tp2): verify weights
sync, version matches, no hang, no IPC leak. Only commit + push + PR + Codex-review after.

---

## PR-A details (merged)

## Why
Post-#168 (which added `pause_generation()`/`continue_generation()` quiesce to the
colocate weight-sync path), the colocate branch of `UpdateWeightFromTensor.update_weights`
still does:
- `release_memory_occupation(level=0)` = `flush_cache()` + `POST /sleep?level=0`. On
  upstream vLLM 0.22.0, `level=0` frees **no** GPU memory (just pauses the scheduler) —
  so the `/sleep level=0` pause is now **redundant** with `pause_generation()`; only its
  `flush_cache()` is doing real work.
- `resume_memory_occupation(tags=["weights","kv_cache"])` — a **no-op + "Executor is not
  sleeping" warning** (the executor never slept at level=0); `continue_generation()`
  already handles resume.

slime's `update_weights` (the thing vime mirrors) does exactly: `pause_generation()` +
`flush_cache()` + `continue_generation()` — **no sleep/release**. vime's `release(level=0)`
/`resume(...)` are a vestigial wrapper.

## Changes (slime-faithful)
1. `vime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
   - colocate release: `release_memory_occupation(level=0)` → `flush_cache()`
   - colocate resume: drop `resume_memory_occupation(tags=[...])` (keep `continue_generation()`)
   - update the flow docstring (colocated now == distributed: pause/flush + continue)
2. `vime/backends/vllm_utils/vllm_engine.py`
   - correct the `release_memory_occupation` docstring (level=0 = pause-only/no free;
     level=1 = offload weights→CPU + drop KV; level=2 = drop all). Method stays (still
     used by generic `offload()` at level=1).

## Verify
- diff structurally matches slime `update_weight_from_tensor.update_weights`.
- no remaining `level=0` caller; `release_memory_occupation` still used by `offload()` (level=1).
- behavior equivalent: pause_generation already quiesces; flush_cache preserved; resume via continue_generation.

## After
- commit + push + PR; then Codex-review and relay assessment.
