# agent_run for `vime-modal-sandbox`

Initialized 2026-06-07T02:30:08Z on branch `main` @ `6011ae9f7f`.

This is the canonical entry point. Read these in order when picking up the task:

1. **`run_manifest.json`** — absolute project root, canonical agent_run path,
   and where scripts / reports / results must be written
2. **`handoff.json`** — last agent, next phase, blockers, one-line summary
3. **`plan.md`** — working plan plus generated event-projection status block
4. **`events.jsonl` (last 20 lines)** — recent activity, including dead_ends
5. **`user_brief.md`** — verbatim user ask, never paraphrased

Other files:

- `results/` — produced artifacts; each has a `.meta.json` sidecar with provenance
- `scripts/` — reusable task scripts, repro helpers, cluster probes, parsers
- `reports/` — durable summaries, audits, final reports, postmortems
- `bisect.md` — generated good/bad/suspect debug anchors after `state_anchor`
  events exist
- `errors.log` — append-only error tail (also captured as `error` events)

## Protocol reminder

- **After context compaction or a fresh session, read `run_manifest.json` before
  writing anything.** Use its `agent_run_dir` as the durable state root.
- **Reusable scripts belong in `agent_run/scripts/`.** If a script must also be
  convenient from project-level `scripts/`, add a symlink or tiny wrapper that
  points back to the canonical `agent_run/scripts/` file.
- **Reports and postmortems belong in `agent_run/reports/` unless the report is a
  raw experiment artifact, in which case put it under `agent_run/results/<run>/`.**
- **Never edit `handoff.json` or the generated projection block in `plan.md` by
  hand.** Also never hand-edit generated views like `bisect.md`,
  `candidates.md`, or `bugs/INDEX.md`. Append an event; regenerate the view with
  `<infra-clone>/scripts/agent_run.py project`.
- **Every artifact under `results/` needs a `.meta.json` sidecar** (cmd /
  git_commit / env_pins / ts / agent).
- **When debugging regressions, append `state_anchor` events** for known-good
  and known-bad states so `bisect.md` can bound the search space.
- **On exit, append a `handoff` event** with `next_phase` and summary so the
  next agent can pick up.

See `<infra-clone>/core/PRINCIPLES.md` for the full contract and
`<infra-clone>/core/DURABLE_WORKSPACE.md` for the compact-recovery write map.
