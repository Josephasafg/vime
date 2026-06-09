# Plan: ci-full-sweep

## Goal

comprehensive CI sweep + slime A/B + bisect — pull latest vime (origin/main), run ALL CI tests on h200 with vime-latest; on any failure run the same test with the slime image as A/B to confirm a real issue; git-bisect regressions. Setup first.

## Durable Workspace

- Project root: `/home/aoshen/vime/projects/ci-full-sweep`
- Agent run: `/home/aoshen/vime/projects/ci-full-sweep/agent_run`
- Reusable scripts: `agent_run/scripts/`
- Reports / postmortems: `agent_run/reports/`
- Raw logs / metrics / artifacts: `agent_run/results/<phase-or-run>/`
- Debug good/bad anchors: append `state_anchor` events; generated view is
  `agent_run/bisect.md`

## Out of scope

- (fill as the task is decomposed)

## Phases

## Event Projection

<!-- agent-run:projection:start -->

_Generated from `events.jsonl` by `scripts/agent_run.py project`._

### Recent Events
 - 2026-06-08T02:10:26Z | task_started | comprehensive CI sweep + slime A/B + bisect: pull latest vime (origin/main ae5d6b04), run ALL CI tests on h200 with v...

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
