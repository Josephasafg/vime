# xnode-pd agent_run

**Goal**: Cross-node PD on H200 ×2 — Qwen3-30B-A3B, training PP2/EP8/TP4/CP2, rollout TP2 with 1-node prefill + 1-node decode (colocated).

**Worktree**: `projects/xnode-pd/vime-worktree/` (branch `feat/xnode-pd`)

**Harness**: out-of-tree under `vime-test-runs/xnode-pd*/`

## Status

Active — setting up harness + driver.

## Key paths

| Path | Role |
|------|------|
| `agent_run/plan.md` | phases + blockers |
| `agent_run/events.jsonl` | state history |
| `vime-test-runs/xnode-pd*/run.log` | experiment logs (on h200-0 NFS) |
| `_harness/ci_sbatch_h200_2node_xnpd.sh` | sbatch launch harness |
| `_harness/tests_2node/test_qwen3_30B_A3B_h200_xnode_pd.py` | test driver |
