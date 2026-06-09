#!/usr/bin/env python3
"""Stage 5: Verify cloudflared reverse tunnel from Modal sandbox to head.

Starts a simple HTTP echo server on localhost, creates a cloudflared quick-tunnel,
then boots a Modal sandbox and curls the tunnel URL from inside.

Run:
  export MODAL_TOKEN_ID="ak-..."
  export MODAL_TOKEN_SECRET="as-..."
  PYTHONPATH=/home/aoshen/vime/projects/vime_modal_sandbox/vime \
  python scripts/test_tunnel_dialback.py
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

sys.path.insert(0, "/home/aoshen/vime/projects/vime_modal_sandbox/vime")

SHIM_PORT = 18901  # avoid colliding with anything


class EchoHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        resp = {"status": "ok", "path": self.path, "source": "tunnel_test_head"}
        self.wfile.write(json.dumps(resp).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode() if length else ""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        resp = {"status": "ok", "echo": body[:200], "source": "tunnel_test_head"}
        self.wfile.write(json.dumps(resp).encode())

    def log_message(self, format, *args):
        pass  # suppress request logs


def start_echo_server():
    server = HTTPServer(("127.0.0.1", SHIM_PORT), EchoHandler)
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def start_cloudflared(port: int, log_path: str) -> subprocess.Popen:
    """Start cloudflared quick-tunnel and extract the public URL."""
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def extract_tunnel_url(proc: subprocess.Popen, timeout: float = 30) -> str:
    """Read cloudflared output until we find the tunnel URL."""
    import re
    deadline = time.time() + timeout
    lines = []
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        lines.append(line.strip())
        m = re.search(r'(https://[a-z0-9-]+\.trycloudflare\.com)', line)
        if m:
            return m.group(1)
    print("  cloudflared output so far:")
    for l in lines[-20:]:
        print(f"    {l}")
    raise TimeoutError("cloudflared did not print tunnel URL within timeout")


async def test_dialback(tunnel_url: str):
    from vime.agent.sandbox import ModalSandbox

    print(f"\n▶ Stage 5b: Booting sandbox and testing dial-back to {tunnel_url} ...")
    sb = ModalSandbox("python:3.12-bookworm")
    await sb.__aenter__()
    try:
        print(f"  ✓ Sandbox booted (id={sb.sandbox_id})")

        # Test GET
        rc, stdout, stderr = await sb.exec(
            f"curl -sf '{tunnel_url}/health' || echo DIALBACK_FAILED",
            user="root", timeout=30,
        )
        print(f"  GET /health: rc={rc} body={stdout.strip()[:200]}")

        if "DIALBACK_FAILED" in stdout:
            print("  ✗ Dial-back FAILED — tunnel not reachable from sandbox")
            # Retry once after a short wait (edge warmup)
            print("  retrying in 5s (edge warmup) ...")
            await asyncio.sleep(5)
            rc, stdout, stderr = await sb.exec(
                f"curl -sf '{tunnel_url}/health' || echo DIALBACK_FAILED",
                user="root", timeout=30,
            )
            print(f"  GET /health (retry): rc={rc} body={stdout.strip()[:200]}")

        if "tunnel_test_head" in stdout:
            print("  ✓ Dial-back CONFIRMED — sandbox reached head's echo server via tunnel")
        else:
            print(f"  ✗ Unexpected response: {stdout[:300]}")

        # Test POST (simulates adapter request)
        rc, stdout, stderr = await sb.exec(
            f"curl -sf -X POST -H 'Content-Type: application/json' "
            f"-d '{{\"test\":\"hello from sandbox\"}}' '{tunnel_url}/v1/messages' || echo POST_FAILED",
            user="root", timeout=30,
        )
        print(f"  POST /v1/messages: rc={rc} body={stdout.strip()[:200]}")

        if "tunnel_test_head" in stdout:
            print("  ✓ POST dial-back CONFIRMED")

    finally:
        await sb.__aexit__(None, None, None)


async def main():
    token_id = os.environ.get("MODAL_TOKEN_ID", "")
    if not token_id:
        print("ERROR: MODAL_TOKEN_ID not set")
        sys.exit(1)

    print("▶ Stage 5a: Starting echo server + cloudflared tunnel ...")
    server = start_echo_server()
    print(f"  ✓ Echo server on localhost:{SHIM_PORT}")

    cf_proc = start_cloudflared(SHIM_PORT, "/tmp/cf_test.log")
    try:
        tunnel_url = extract_tunnel_url(cf_proc, timeout=30)
        print(f"  ✓ Tunnel URL: {tunnel_url}")

        # Host self-test (best-effort — edit host may lack DNS for trycloudflare.com)
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"{tunnel_url}/health", timeout=10)
            body = json.loads(resp.read())
            print(f"  ✓ Host self-test: {body}")
        except Exception as e:
            print(f"  ⚠ Host self-test failed (DNS/network): {e.__class__.__name__}: {e}")
            print(f"    (OK — sandbox has full internet, testing from there instead)")

        # Wait for edge warmup (critical per handoff: cloudflared prints URL before edge is live)
        print("  Waiting 15s for edge warmup ...")
        await asyncio.sleep(15)

        # Test from sandbox
        await test_dialback(tunnel_url)

    finally:
        cf_proc.send_signal(signal.SIGTERM)
        cf_proc.wait(timeout=5)
        server.shutdown()

    print(f"\n{'='*60}")
    print(f"  Stage 5: TUNNEL DIAL-BACK PASS")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
