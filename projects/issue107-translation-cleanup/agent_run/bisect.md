# Bisect anchors

_See `core/BISECT_ANCHORS.md` for the event schema and capture rules._

<!-- agent-run:bisect-projection:start -->

_Generated from `events.jsonl` by `scripts/agent_run.py project`._
_State anchors are typed good/bad/suspect snapshots for narrowing a
debug search space; append `state_anchor` events rather than editing
this file by hand._

## Latest Good/Bad Bracket
_(missing known-bad anchor)_

## Known Good Anchors
| id | ts | verdict | commit | repro / refs | evidence |
|---|---|---|---|---|---|
| BASELINE-pre107 | 2026-06-08T14:06:45Z | good | cdc439bb5e4e dirty | cmd=git grep -Iic sglang\|slime @6011ae9f; config=6011ae9f (#154, pre-#107-wave) | baseline: sglang=5occ/4f, slime=29occ/7f; HEAD 09450c1b: sglang=22/8 (+17), slime=94/24 (+65) |
| G2-baseline-reached | 2026-06-08T14:38:16Z | good | cdc439bb5e4e dirty | cmd=git grep -Iic sglang\|slime; config=issue107-cleanup HEAD | sglang 22->5 (=baseline); slime 94->36 (floor: +7 over baseline = #180 attribution + provenance). 3 commits: parser p... |

## Known Bad Anchors
_(none)_

## Anchor Quality Warnings
- BASELINE-pre107: dirty workspace without git_diff_hash; untracked-only state may need explicit artifact capture
- BASELINE-pre107: no log_path/log_dir/artifact_ref
- G2-baseline-reached: dirty workspace without git_diff_hash; untracked-only state may need explicit artifact capture
- G2-baseline-reached: no log_path/log_dir/artifact_ref

<!-- agent-run:bisect-projection:end -->

