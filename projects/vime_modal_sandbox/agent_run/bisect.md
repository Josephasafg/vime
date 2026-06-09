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
| E2E-GREEN | 2026-06-07T02:54:56Z | good | 6011ae9f7f78 dirty | cmd=MODAL_TOKEN_ID=*** MODAL_TOKEN_SECRET=*** /home/aoshen/code/uv_envs/py312/bin...; config=buildpack-deps:bookworm;... | unit 34/34; real-Modal e2e 13/13; reverse dial-back confirmed sandbox-stdout + head-log (xff=egress IP) |

## Known Bad Anchors
_(none)_

## Anchor Quality Warnings
- E2E-GREEN: dirty workspace without git_diff_hash; untracked-only state may need explicit artifact capture

<!-- agent-run:bisect-projection:end -->

