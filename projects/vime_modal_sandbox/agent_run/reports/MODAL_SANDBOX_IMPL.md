# Modal sandbox backend for vime `coding_agent_rl` — implementation report

**Status:** implemented + verified (unit 34/34, real-Modal e2e 13/13). Route **A.2**
(Modal backend for the existing `coding_agent_rl` `Sandbox` Protocol + cloudflared
reverse tunnel). Not yet upstreamed as a vime PR.

## What was built (all additive; E2B path untouched)

1. **`vime/agent/sandbox.py` → new `ModalSandbox` class** — async context manager
   over `modal.Sandbox`, mirroring `E2BSandbox`'s public surface
   (`__aenter__/__aexit__/exec/write_file/read_file/sandbox_id`). `modal` is
   imported lazily so the module imports without the dependency.
   Four E2B→Modal gaps resolved:
   - `user=` → emulated with `runuser -u <user>` (env keys whitelisted through).
   - `write_file(str|bytes|host Path)` → `mkdir -p && cat >` with binary stdin
     streaming (2 MiB chunks), then `chown` to `user`.
   - image → `Image.from_registry(tag)` (+ `REGISTRY_USERNAME/PASSWORD` from
     `DOCKER_*` for private registries).
   - cleanup always reaches `terminate` (leaked sandboxes count against the
     account cap — uni-agent lesson).
2. **`examples/coding_agent_rl/sandbox.py` → `make_sandbox(image)` factory** +
   both `E2BSandbox(image)` call sites (work sandbox L77, eval sandbox L315)
   switched to `make_sandbox(image)`. Backend chosen by
   `VIME_AGENT_SANDBOX_BACKEND` (`e2b` default | `modal`), read per call.
3. **`examples/coding_agent_rl/generate.py` → `ADAPTER_URL_OVERRIDE`** — lets a
   reverse tunnel supply a ready-made public adapter URL when the head has no
   directly routable host:port (relaxes the `VIME_HEAD_HOST` guard; replaces the
   `http://{host}:{port}` construction when set).

Workspace (isolated dev copy, baseline read-only from
`projects/vime_gb200_training/workspace`): `../workspace/`.

## Reverse-network: the crux, empirically settled

`coding_agent_rl` runs Claude Code **inside** the sandbox, dialing back to the
head's in-process Anthropic adapter. Measured on real Modal sandboxes:

| Path | Result |
|------|--------|
| sandbox outbound egress (`block_network=False`) | ✅ |
| sandbox → **cloudflared tunnel** → head adapter | ✅ (Route A.2) |
| sandbox → head **public IP** directly (`136.111.112.12`) | ❌ NAT egress-only |

So on a private cluster the head exposes `SHIM_PORT` via `cloudflared tunnel`
and points `ADAPTER_URL_OVERRIDE` (or `VIME_HEAD_HOST`) at the public URL.

## Verification

- **Unit:** `workspace/tests/test_modal_sandbox.py` — 34 tests, faked `modal`
  (no network). Covers Protocol conformance, env/kwarg config, image/app/create
  wiring, registry secret, exec root/runuser/env-whitelist, check-raise, output
  cap, timeout contract, write_file str/bytes/host-Path streaming + chown,
  read_file swallow, always-terminate, and the factory. `34 passed in 0.07s`.
- **E2E (real Modal, `buildpack-deps:bookworm`):**
  `agent_run/scripts/e2e_modal_sandbox.py` — 13 checks through the real class:
  exec root/egress, write_file str+read roundtrip, bytes sha256, 5 MiB host-Path
  sha256 (multi-chunk), runuser user, chown ownership, read-missing, check-raise,
  **reverse dial-back (sandbox stdout + head-side log, xff = sandbox egress IP)**,
  terminate. `13 passed, 0 failed`. Log: `results/e2e/e2e_clean.log`.

### Gotcha found + fixed
cloudflared quick-tunnels print their URL **before** the edge route registers.
A cached-image sandbox boots fast and curls before the edge is live → transient
reverse-dial FAIL. Fix: `_wait_tunnel_ready()` polls the public URL from the head
until it round-trips before creating the sandbox. (First run masked this because
the first-time image pull bought ~60 s of warmup.)

## Not yet done (next phase)
- Open a vime PR against the branch that actually carries `coding_agent_rl`
  (it is **not on `main`**; lives in worktrees + `projects/vime_gb200_training`).
- Full training-scale e2e (real SWE image + Node22/Claude-Code tarballs + live
  vLLM + adapter producing a diff & reward) — needs cluster + tarballs.
- Decide whether to keep the detached-poll launcher (Modal has no E2B 6.5-min
  HTTP/2 reset, so a long foreground `exec` would work) or keep it for parity.
