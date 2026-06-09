# vime Modal Sandbox — Handoff Document

**Project:** `projects/vime_modal_sandbox/`
**Status:** PR #167 open, unit 36/36, real-Modal e2e 13/13, Codex reviewed + fixed
**Last updated:** 2026-06-07
**Author:** opus-4-8

---

## 1. What This Is

A **Modal backend** (`ModalSandbox`) for vime's `coding_agent_rl` agent-RL sandbox,
alongside the existing E2BSandbox. Purely additive — E2B path is byte-unchanged.
Selected at runtime via `VIME_AGENT_SANDBOX_BACKEND=modal`.

**Route A.2** — empirically validated: Claude Code runs *inside* the Modal sandbox
and dials back to the head's Anthropic adapter through a **cloudflared reverse tunnel**.

---

## 2. Credentials & Tokens

### Modal

```bash
export MODAL_TOKEN_ID="ak-QJwl8VavnZNhVsAlwms0Qz"
export MODAL_TOKEN_SECRET="as-QxiOjaMQcGGW0182XHKu6F"
```

- Modal SDK version: **1.4.2** (`/home/aoshen/code/uv_envs/py312/bin/python`)
- No `~/.modal.toml` on this host — pass tokens via env vars
- Modal App names used: `vime-coding-agent-sandbox` (production), `vime-reverse-probe` / `vime-timeout-probe` (throwaway probes)

### Host Info

- **Edit host:** `gb200-rack1-01` (arm64, GB200)
- **Python env:** `/home/aoshen/code/uv_envs/py312/bin/python` (Python 3.12, has modal 1.4.2 + pytest 9.0.2)
- **cloudflared:** `/usr/local/bin/cloudflared` (pre-installed)
- **Public IP:** `136.111.112.12` (NAT egress-only — inbound blocked, hence the tunnel)
- **Git user:** `aoshen02`

---

## 3. Architecture Decision (empirically settled)

### Reverse-network requirement

`coding_agent_rl` runs Claude Code **inside** the sandbox. Claude Code dials back to
the head's in-process Anthropic adapter at `ANTHROPIC_BASE_URL`. This requires the
sandbox to reach the head — the "reverse network" problem.

### Measured on real Modal sandboxes

| Path | Result | Evidence |
|------|--------|----------|
| Sandbox outbound egress (`block_network=False`) | ✅ | Sandbox curl → ipify returns egress IP |
| Sandbox → **cloudflared tunnel** → head adapter | ✅ | Sandbox stdout: token received; head log: xff=sandbox IP |
| Sandbox → head **public IP directly** (`136.111.112.12:18901`) | ❌ | Empty response, no hit on head log — NAT egress-only |

### Conclusion

**Direct-IP is blocked → must use a reverse tunnel.** `cloudflared tunnel --url
http://localhost:{SHIM_PORT}` on the head exposes the adapter as a public URL. Set
`ADAPTER_URL_OVERRIDE=https://xxx.trycloudflare.com` (or equivalent) so
`generate.py` feeds it to the sandbox as `ANTHROPIC_BASE_URL`.

### Modal API surface verified (1.4.2)

Key findings from API introspection + empirical testing:
- `Sandbox.exec(*args, timeout=, env=, text=True)` → proc with `.stdout/.stderr` (`.read.aio()`), `.stdin` (`.write/.write_eof/.drain.aio()`), `.wait.aio()`
- **Per-exec timeout does NOT raise `SandboxTimeoutError`** — `wait()` returns `rc == -1` (verified: `sleep 20` + `timeout=3` → 3.0s, rc=-1, empty streams, no exception)
- `Image.from_registry(tag, secret=)` — `secret` is singular (not `secrets`)
- `App.lookup.aio(name, create_if_missing=True)`
- `Sandbox.create.aio(image=, app=, cpu=, memory=, timeout=, block_network=)`

---

## 4. PR Details

**PR #167:** https://github.com/vllm-project/vime/pull/167
- **Base:** `sync/slime-mega-D` (PR #148 — introduces `coding_agent_rl`)
- **Branch:** `aoshen/coding-agent-modal-sandbox`
- **Worktree:** `.worktree/modal-sandbox`
- **Commits:** 2
  1. `005d484` feat(agent): add Modal sandbox backend for coding_agent_rl
  2. `50c27f7` fix(agent): address Codex review on ModalSandbox exec/write_file

### Files changed (4, all additive)

| File | Change |
|------|--------|
| `vime/agent/sandbox.py` | +`ModalSandbox` class (~220 lines, lazy `import modal`) |
| `examples/coding_agent_rl/sandbox.py` | +`make_sandbox()` factory, swap 2 `E2BSandbox(image)` → `make_sandbox(image)` |
| `examples/coding_agent_rl/generate.py` | +`ADAPTER_URL_OVERRIDE` env (relaxes `VIME_HEAD_HOST` guard) |
| `tests/test_agent_modal_sandbox.py` | 36 unit tests (faked modal, CI-safe) |

### How `ModalSandbox` resolves E2B→Modal gaps

| Gap | Solution |
|-----|----------|
| Modal `exec` has no `user=` | `runuser -u <user> [--whitelist-environment=KEY1,KEY2] -- bash -lc <cmd>` |
| `write_file(str\|bytes\|Path)` | `mkdir -p && cat > path` via binary stdin; host Path streamed in 2 MiB chunks; `chown` after for non-root |
| Image | `modal.Image.from_registry(tag)` + optional `DOCKER_USERNAME/PASSWORD` → registry secret |
| Cleanup | `__aexit__` always reaches `terminate.aio()` (leaked sandboxes count against Modal account cap) |
| Timeout | `rc == -1` from `wait()` treated as timeout/kill (raise on `check=True`, sentinel stderr otherwise) |

---

## 5. Verification Results

### Unit tests (36/36, 0.09s)

```bash
cd /home/aoshen/vime/.worktree/modal-sandbox
/home/aoshen/code/uv_envs/py312/bin/python -m pytest tests/test_agent_modal_sandbox.py -v
```

Covers: Protocol conformance, env/kwarg config, SWE-fallback env, image/app/create
wiring, public vs private registry secret, exec root/runuser/env-whitelist,
check-raise, output cap, no-cap default, timeout (both raise + rc=-1 paths),
write_file str/bytes/host-Path streaming + chown, read_file success + missing→"",
always-terminate, async-context-manager, factory default/e2b/modal/case-insensitive/unknown.

### E2E on real Modal (13/13)

```bash
cd /home/aoshen/vime/projects/vime_modal_sandbox
MODAL_TOKEN_ID="ak-QJwl8VavnZNhVsAlwms0Qz" \
MODAL_TOKEN_SECRET="as-QxiOjaMQcGGW0182XHKu6F" \
/home/aoshen/code/uv_envs/py312/bin/python agent_run/scripts/e2e_modal_sandbox.py
```

Checks: aenter/sandbox_id, exec root, egress, write_file(str)+read roundtrip,
write_file(bytes) sha256, write_file(host Path 5MiB) sha256, exec user=agent
(runuser), write_file(user=agent) ownership, read_file(missing)→"", exec check=True
raises, **reverse dial-back (sandbox stdout + head-side log)**, aexit terminates.

Image: `buildpack-deps:bookworm`. Logs: `agent_run/results/e2e/e2e_clean.log`,
`agent_run/results/e2e/e2e_post_review.log`.

### Gotcha discovered + fixed

**cloudflared quick-tunnel edge-warmup race:** cloudflared prints its URL *before*
the edge route registers. A cached-image sandbox boots fast and curls before the edge
is live → transient reverse-dial fail. Fix in e2e harness: `_wait_tunnel_ready()` polls
the public URL from the head before creating the sandbox.

---

## 6. Codex Review Summary (PR #167)

| # | Issue | Severity | Verdict | Action |
|---|-------|----------|---------|--------|
| 1 | exec: gather stdout/stderr reads with wait() | major | **Valid (robustness)** | Fixed in commit 2 |
| 2 | Modal timeout returns rc=-1 (not raises), so except branch is dead | major | **Valid (real bug) — empirically verified** | Fixed: explicit rc==-1 handling |
| 3 | write_file: drain stderr alongside wait() | major | **Partly valid (diagnostics)** | Fixed: stderr captured into OSError |
| 4 | Non-root write_file only chowns file, not mkdir-created parents | minor | **Low-impact** — actual call sites use pre-chowned dirs | Not fixed (no over-engineering) |

---

## 7. File Layout

```
projects/vime_modal_sandbox/
├── HANDOFF.md                          ← this file
├── workspace/                          ← isolated dev copy (baseline from gb200 workspace)
│   ├── vime/agent/sandbox.py           ← ModalSandbox implementation
│   ├── examples/coding_agent_rl/
│   │   ├── sandbox.py                  ← make_sandbox() + swapped call sites
│   │   ├── generate.py                 ← ADAPTER_URL_OVERRIDE
│   │   └── aiohttp_threaded.py         ← unchanged baseline
│   └── tests/test_modal_sandbox.py     ← 36 unit tests (dev workspace copy)
└── agent_run/
    ├── INDEX.md / run_manifest.json / handoff.json / plan.md
    ├── events.jsonl                    ← 5 events (task_started, state_anchor, decision, 2× handoff)
    ├── scripts/
    │   ├── e2e_modal_sandbox.py        ← real-Modal e2e harness
    │   └── e2e_modal_sandbox.py.meta.json
    ├── results/e2e/
    │   ├── e2e_clean.log               ← 13/13 pass log (pre-review)
    │   ├── e2e_clean.log.meta.json
    │   └── e2e_post_review.log         ← 13/13 pass log (post Codex fixes)
    └── reports/
        └── MODAL_SANDBOX_IMPL.md       ← full implementation report
```

PR worktree: `.worktree/modal-sandbox` (branch `aoshen/coding-agent-modal-sandbox`).

---

## 8. How to Use (for future agent or developer)

### Run unit tests

```bash
cd /home/aoshen/vime/.worktree/modal-sandbox   # or any checkout with the PR
/home/aoshen/code/uv_envs/py312/bin/python -m pytest tests/test_agent_modal_sandbox.py -v
```

### Run e2e on real Modal

```bash
cd /home/aoshen/vime/projects/vime_modal_sandbox
export MODAL_TOKEN_ID="ak-QJwl8VavnZNhVsAlwms0Qz"
export MODAL_TOKEN_SECRET="as-QxiOjaMQcGGW0182XHKu6F"
/home/aoshen/code/uv_envs/py312/bin/python agent_run/scripts/e2e_modal_sandbox.py
```

Requires: cloudflared on PATH, outbound egress.

### Use in a training run

```bash
# In run.sh (or sbatch script):
export VIME_AGENT_SANDBOX_BACKEND=modal
export MODAL_TOKEN_ID="ak-QJwl8VavnZNhVsAlwms0Qz"
export MODAL_TOKEN_SECRET="as-QxiOjaMQcGGW0182XHKu6F"

# Start cloudflared sidecar (or use a named tunnel for production):
cloudflared tunnel --url http://localhost:${SHIM_PORT:-18001} --no-autoupdate &
sleep 10  # wait for edge registration
export ADAPTER_URL_OVERRIDE=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /path/to/cf.log)

# Rest of training launch unchanged
```

### Env knobs (all optional, sane defaults)

| Env var | Default | Purpose |
|---------|---------|---------|
| `VIME_AGENT_SANDBOX_BACKEND` | `e2b` | `e2b` or `modal` |
| `ADAPTER_URL_OVERRIDE` | (unset) | Full public adapter URL; replaces `http://{VIME_HEAD_HOST}:{port}` |
| `VIME_AGENT_SANDBOX_MODAL_APP` | `vime-coding-agent-sandbox` | Modal App name |
| `VIME_AGENT_SANDBOX_MODAL_CPU` | `2.0` | vCPU per sandbox |
| `VIME_AGENT_SANDBOX_MODAL_MEMORY_MB` | `8192` | Memory per sandbox |
| `VIME_AGENT_SANDBOX_MODAL_BLOCK_NETWORK` | `false` | Block outbound (must be false for reverse dial-back) |
| `VIME_AGENT_SANDBOX_MODAL_MAX_OUTPUT_BYTES` | `0` (unlimited) | Cap stdout/stderr per exec |
| `VIME_AGENT_SANDBOX_LIFETIME_SEC` | `3600` | Sandbox wall-clock timeout (shared with E2B) |
| `DOCKER_USERNAME` / `DOCKER_PASSWORD` | (unset) | Private registry credentials |

---

## 9. What's NOT Done (Next Steps)

1. **PR review cycle on #167** — stacked on #148 (`sync/slime-mega-D`); merge #148 first or retarget.
2. **Training-scale e2e** — real SWE-bench image + Node22/Claude-Code tarballs + live vLLM/adapter + diff→reward flow. Needs: cluster allocation, tarballs on host, SLIME_HEAD_HOST or ADAPTER_URL_OVERRIDE.
3. **Detached-poll launcher simplification** — Modal has no E2B 6.5-min HTTP/2 reset, so `_spawn_claude_code` could use a long foreground `exec` instead of the detached+5s-poll pattern. Deferred for parity in this PR.
4. **Production tunnel** — cloudflared quick-tunnels (trycloudflare.com) rotate URLs on restart. For production, use a named tunnel with a stable hostname (`cloudflared tunnel create ...`).
5. **Codex comment #4** — non-root `write_file` only chowns the file, not parent dirs created by `mkdir -p`. Not an issue for current call sites (all use pre-chowned dirs), but could matter if future code writes to novel paths as non-root.

---

## 10. References Studied

| Reference | Key takeaway |
|-----------|-------------|
| [tinker-cookbook `modal_sandbox.py`](https://github.com/thinking-machines-lab/tinker-cookbook/blob/dacb835/tinker_cookbook/sandbox/modal_sandbox.py) | Primary template: ModalSandbox + ModalSandboxPool; exec/write_file/read_file via stdin stream |
| [harbor `environments/modal.py`](https://github.com/harbor-framework/harbor/blob/8d40b8a/src/harbor/environments/modal.py) | Direct vs DinD strategies; `su` for user emulation; retry with tenacity; registry secret; host-networking |
| [uni-agent `deployment/modal/deployment.py`](https://github.com/aoshen02/uni-agent/tree/benchmark/semianalysis-verl) | `encrypted_ports`+`tunnels` for forward-dial; fleet limiter (`MODAL_MAX_STARTING`); wall-clock budget; sandbox-leak lesson (847 leaked) |
| [kimbochen/slime `examples/swe-bench/sandbox.py`](https://github.com/kimbochen/slime/tree/feat/swe-env-example) | Alternative "Route B": passive tool-executor (no reverse network), Modal-native `filesystem.read_text/write_text`, lazy create/reattach. **Not adopted** because vime uses Claude-Code-in-sandbox. |
| slime `agent/sandbox.py` (Protocol + E2BSandbox) | The contract: `exec→(rc,stdout,stderr)`, `write_file(str\|bytes\|Path)`, `read_file→str`, async context manager, `sandbox_id` |
| slime `coding_agent_rl/sandbox.py` + `generate.py` | Two swap points, `SLIME_HEAD_HOST`, detached-poll launcher, adapter singleton |
