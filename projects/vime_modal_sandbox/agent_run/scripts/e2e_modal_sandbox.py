"""End-to-end test of the real ModalSandbox backend against live Modal.

Exercises every Sandbox-Protocol method through the ACTUAL
``vime.agent.sandbox.ModalSandbox`` (not a mock) on a real Modal sandbox, plus
the decisive reverse-network path: from inside the sandbox, dial back to a
local mimic-adapter through a cloudflared tunnel (the coding_agent_rl
ANTHROPIC_BASE_URL flow).

Self-contained: starts its own mimic-adapter HTTP server + cloudflared tunnel,
runs the checks, then tears everything down (including the Modal sandbox).

Requires: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in env, cloudflared on PATH.

Usage:
    MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... \
      python e2e_modal_sandbox.py [--image buildpack-deps:bookworm]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Import the real ModalSandbox from the dev workspace.
WORKSPACE = Path(__file__).resolve().parents[2] / "workspace"
sys.path.insert(0, str(WORKSPACE))
from vime.agent.sandbox import ModalSandbox, Sandbox  # noqa: E402

TOKEN = "vime-e2e-reverse-ok"
PORT = 18917

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def _start_mimic_server(workdir: Path) -> subprocess.Popen:
    server_py = workdir / "server.py"
    server_py.write_text(
        "import http.server, socketserver, sys, datetime\n"
        f"PORT={PORT}\nTOKEN={TOKEN!r}\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        cl=self.client_address[0]; xff=self.headers.get('X-Forwarded-For','')\n"
        "        sys.stderr.write(f'HIT {self.path} from {cl} xff={xff}\\n'); sys.stderr.flush()\n"
        "        self.send_response(200); self.send_header('Content-Type','text/plain'); self.end_headers()\n"
        "        self.wfile.write(f'{TOKEN} path={self.path}\\n'.encode())\n"
        "    def log_message(self,*a): pass\n"
        "with socketserver.TCPServer(('0.0.0.0',PORT),H) as h:\n"
        "    sys.stderr.write('serving\\n'); sys.stderr.flush(); h.serve_forever()\n"
    )
    log = open(workdir / "server.log", "wb")
    return subprocess.Popen([sys.executable, str(server_py)], stdout=log, stderr=log)


def _start_tunnel(workdir: Path) -> tuple[subprocess.Popen, str]:
    log_path = workdir / "cf.log"
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{PORT}", "--no-autoupdate"],
        stdout=log,
        stderr=log,
    )
    url = ""
    deadline = time.time() + 40
    while time.time() < deadline:
        try:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log_path.read_text(errors="ignore"))
            if m:
                url = m.group(0)
                break
        except FileNotFoundError:
            pass
        time.sleep(0.5)
    return proc, url


def _wait_tunnel_ready(url: str, timeout: float = 45.0) -> bool:
    """Poll the public tunnel URL from the head until the edge route is live.

    cloudflared prints its URL before the edge has finished registering the
    tunnel; curling too early (from a fast cached-image sandbox) races and
    fails. Block here until the head itself can round-trip the token, so the
    in-sandbox dial-back below is deterministic."""
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/ping-warmup", timeout=5) as resp:
                if resp.status == 200 and TOKEN in resp.read().decode("utf-8", "replace"):
                    return True
        except Exception:
            pass
        time.sleep(1.5)
    return False


async def run_checks(image: str, tunnel_url: str, server_log: Path) -> None:
    ms = ModalSandbox(image, block_network=False, timeout=400)
    assert isinstance(ms, Sandbox)

    async with ms as sb:
        check("aenter sets sandbox_id", bool(sb.sandbox_id) and sb.sandbox_id.startswith("sb"), sb.sandbox_id)

        # 1. exec as root
        rc, out, err = await sb.exec("echo hi-from-root && id -un")
        check("exec root", rc == 0 and "hi-from-root" in out and "root" in out, f"rc={rc} out={out!r}")

        # 2. egress (block_network=False)
        rc, out, _ = await sb.exec("curl -s -m 20 https://api.ipify.org; echo")
        check("egress to public internet", rc == 0 and out.strip() != "", f"egress_ip={out.strip()!r}")

        # 3. write_file(str) + read_file roundtrip
        text = "Resolve the issue described here.\nLine2: ünïcode + symbols !@#\n"
        await sb.write_file("/workspace/PROBLEM_STATEMENT.md", text)
        got = await sb.read_file("/workspace/PROBLEM_STATEMENT.md")
        check("write_file(str) + read_file roundtrip", got == text, f"len got={len(got)} want={len(text)}")

        # 4. write_file(bytes) roundtrip via sha256
        blob = bytes(range(256)) * 32  # 8 KiB binary
        await sb.write_file("/tmp/blob.bin", blob)
        rc, out, _ = await sb.exec("sha256sum /tmp/blob.bin | awk '{print $1}'")
        want = hashlib.sha256(blob).hexdigest()
        check("write_file(bytes) sha256", rc == 0 and out.strip() == want, f"got={out.strip()[:16]} want={want[:16]}")

        # 5. write_file(host Path) streamed in chunks (>2MiB) sha256
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            payload = os.urandom(5 * 1024 * 1024)  # 5 MiB -> multiple 2 MiB chunks
            tf.write(payload)
            host_path = Path(tf.name)
        try:
            await sb.write_file("/tmp/big.tar", host_path)
            rc, out, _ = await sb.exec("sha256sum /tmp/big.tar | awk '{print $1}'")
            want = hashlib.sha256(payload).hexdigest()
            check("write_file(host Path) 5MiB sha256", rc == 0 and out.strip() == want, f"got={out.strip()[:16]} want={want[:16]}")
        finally:
            host_path.unlink(missing_ok=True)

        # 6. non-root user emulation via runuser
        await sb.exec("id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent", check=True, timeout=60)
        rc, out, _ = await sb.exec("whoami", user="agent")
        check("exec user=agent (runuser)", rc == 0 and out.strip() == "agent", f"whoami={out.strip()!r}")

        # 7. write_file as agent -> file owned by agent (chown path)
        await sb.exec("mkdir -p /home/agent/work && chown agent:agent /home/agent/work", check=True)
        await sb.write_file("/home/agent/work/run.sh", "#!/bin/bash\necho ok\n", user="agent")
        rc, out, _ = await sb.exec("stat -c '%U' /home/agent/work/run.sh")
        check("write_file(user=agent) ownership", rc == 0 and out.strip() == "agent", f"owner={out.strip()!r}")

        # 8. read_file missing -> "" (mirror E2B swallow)
        got = await sb.read_file("/does/not/exist")
        check("read_file(missing) -> ''", got == "", f"got={got!r}")

        # 9. exec check=True raises on nonzero
        raised = False
        try:
            await sb.exec("exit 3", check=True)
        except RuntimeError:
            raised = True
        check("exec check=True raises on nonzero", raised)

        # 10. THE reverse dial-back: sandbox -> cloudflared tunnel -> head adapter
        rc, out, _ = await sb.exec(
            f"curl -s -m 40 --retry 5 --retry-delay 3 --retry-all-errors {tunnel_url}/ping-from-modalsandbox; echo",
            timeout=90,
        )
        check("reverse dial-back reaches adapter (sandbox stdout)", rc == 0 and TOKEN in out, f"out={out.strip()!r}")

    # 11. after context exit, the sandbox is terminated (handle cleared)
    check("aexit terminates (handle cleared)", ms._sb is None)

    # 12. head-side proof: the mimic adapter logged the sandbox's request
    log_text = server_log.read_text(errors="ignore")
    check("reverse dial-back logged on head", "/ping-from-modalsandbox" in log_text,
          next((ln for ln in log_text.splitlines() if "ping-from-modalsandbox" in ln), ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="buildpack-deps:bookworm")
    args = ap.parse_args()

    if not (os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET")):
        print("ERROR: MODAL_TOKEN_ID / MODAL_TOKEN_SECRET not set", file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix="e2e-modal-"))
    server = tunnel = None
    try:
        server = _start_mimic_server(work)
        tunnel, url = _start_tunnel(work)
        if not url:
            print("ERROR: cloudflared tunnel URL not obtained", file=sys.stderr)
            return 3
        print(f"tunnel: {url}\nimage:  {args.image}", flush=True)
        if not _wait_tunnel_ready(url):
            print("ERROR: tunnel did not become edge-ready in time", file=sys.stderr)
            return 4
        print("tunnel edge-ready\n", flush=True)
        asyncio.run(run_checks(args.image, url, work / "server.log"))
    finally:
        for p in (tunnel, server):
            if p is not None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except Exception:
                    p.kill()

    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print(f"\n==== e2e summary: {n_pass} passed, {n_fail} failed (of {len(results)}) ====")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
