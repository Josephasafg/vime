# Plan: vime-modal-sandbox

## Goal

Add a Modal sandbox backend for vime coding_agent_rl: implement ModalSandbox satisfying the Sandbox Protocol, add make_sandbox() factory + swap 2 E2BSandbox call sites, add ADAPTER_URL_OVERRIDE in generate.py. Reverse-network via cloudflared tunnel (empirically validated 2026-06-07). Then write extensive unit tests and run in-container e2e on real Modal. Route A.2.

## Durable Workspace

- Project root: `/home/aoshen/vime/projects/vime_modal_sandbox`
- Agent run: `/home/aoshen/vime/projects/vime_modal_sandbox/agent_run`
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
 - 2026-06-07T02:30:08Z | task_started | Add a Modal sandbox backend for vime coding_agent_rl: implement ModalSandbox satisfying the Sandbox Protocol, add mak...
 - 2026-06-07T02:54:56Z | state_anchor | anchor=E2E-GREEN | verdict=good | good
 - 2026-06-07T02:54:56Z | decision | ModalSandbox + make_sandbox + ADAPTER_URL_OVERRIDE implemented (additive, E2B untouched); Route A.2 validated; tunnel...
 - 2026-06-07T02:54:56Z | handoff | Code done+verified in projects/vime_modal_sandbox/workspace. NEXT: open vime PR against the branch carrying coding_ag...
 - 2026-06-07T03:28:03Z | handoff | PR #167 opened (base sync/slime-mega-D, branch aoshen/coding-agent-modal-sandbox). Codex review: 4 comments, fixed #1...

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
