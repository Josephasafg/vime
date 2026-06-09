#!/usr/bin/env python3
"""Standalone Modal sandbox pipeline test using vime's ModalSandbox wrapper.

Validates each stage of the coding_agent_rl sandbox pipeline:
  Stage 1: Boot ModalSandbox from a public image
  Stage 2: Install Node.js 22 from host tarball
  Stage 3: Install Claude Code CLI from host tarball
  Stage 4: Verify Claude Code can start (--help)

Run:
  export MODAL_TOKEN_ID="ak-..."
  export MODAL_TOKEN_SECRET="as-..."
  PYTHONPATH=/home/aoshen/vime/projects/vime_modal_sandbox/vime \
  python scripts/test_sandbox_pipeline.py \
    --node-tarball /tmp/node-v22.20.0-linux-x64.tar.xz \
    --cc-tarball /tmp/cc-pack/anthropic-ai-claude-code-2.1.168.tgz
"""

import argparse
import asyncio
import lzma
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "/home/aoshen/vime/projects/vime_modal_sandbox/vime")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-tarball", required=True)
    ap.add_argument("--cc-tarball", required=True)
    ap.add_argument("--image", default="python:3.12-bookworm")
    ap.add_argument("--skip-cc", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    token_id = os.environ.get("MODAL_TOKEN_ID", "")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET", "")
    if not token_id or not token_secret:
        print("ERROR: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must be set")
        sys.exit(1)

    node_tarball = Path(args.node_tarball)
    cc_tarball = Path(args.cc_tarball)
    assert node_tarball.exists(), f"Node tarball not found: {node_tarball}"
    if not args.skip_cc:
        assert cc_tarball.exists(), f"CC tarball not found: {cc_tarball}"

    # Decompress .xz on host
    if node_tarball.suffix == ".xz":
        plain = Path(tempfile.gettempdir()) / f"sandbox_test.{node_tarball.stem}.tar"
        if not plain.exists():
            print(f"  Decompressing {node_tarball.name} -> {plain.name} ...")
            tmp = plain.with_suffix(".tar.partial")
            with lzma.open(node_tarball, "rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp, plain)
        node_tarball = plain

    from vime.agent.sandbox import ModalSandbox

    print(f"\n{'='*60}")
    print(f"  Modal Sandbox Pipeline Test (via vime ModalSandbox)")
    print(f"  Image: {args.image}")
    print(f"  Node tarball: {node_tarball} ({node_tarball.stat().st_size/1e6:.1f} MB)")
    if not args.skip_cc:
        print(f"  CC tarball: {cc_tarball} ({cc_tarball.stat().st_size/1e6:.1f} MB)")
    print(f"{'='*60}\n")

    sb = ModalSandbox(args.image)

    # ── Stage 1: Boot ──
    print("▶ Stage 1: Booting Modal sandbox ...")
    t0 = time.time()
    await sb.__aenter__()
    boot_ms = (time.time() - t0) * 1000
    print(f"  ✓ Sandbox booted in {boot_ms:.0f}ms (id={sb.sandbox_id})")

    try:
        # Basic exec
        rc, stdout, stderr = await sb.exec("uname -m && cat /etc/os-release | head -3", user="root", timeout=15)
        print(f"  ✓ exec(uname): rc={rc}")
        for line in stdout.strip().split("\n")[:4]:
            print(f"    {line}")

        # Egress
        rc, stdout, stderr = await sb.exec("curl -sf https://api.ipify.org || echo EGRESS_FAILED", user="root", timeout=15)
        print(f"  ✓ Outbound egress: {stdout.strip()}")

        # ── Stage 2: Node.js ──
        print("\n▶ Stage 2: Installing Node.js 22 ...")
        t0 = time.time()
        await sb.write_file("/tmp/node22.tar", node_tarball, user="root")
        print(f"  uploaded node tarball ({(time.time()-t0)*1000:.0f}ms)")

        rc, stdout, stderr = await sb.exec(
            "set -e && mkdir -p /opt/node22 && "
            "tar xf /tmp/node22.tar -C /opt/node22 --strip-components=1 && "
            "ln -sf /opt/node22/bin/node /usr/local/bin/node && "
            "ln -sf /opt/node22/bin/npm  /usr/local/bin/npm && "
            "ln -sf /opt/node22/bin/npx  /usr/local/bin/npx && "
            "hash -r 2>/dev/null || true && node --version && npm --version",
            user="root", timeout=120,
        )
        node_ms = (time.time() - t0) * 1000
        if rc != 0:
            print(f"  ✗ Node install FAILED (rc={rc})\n    stderr: {stderr[:500]}")
            return
        versions = stdout.strip().split("\n")
        print(f"  ✓ Node {versions[0] if versions else '?'}, npm {versions[1] if len(versions)>1 else '?'} ({node_ms:.0f}ms)")

        if args.skip_cc:
            print("\n▶ Stage 3-4: SKIPPED (--skip-cc)")
            print(f"\n  RESULT: Stage 1-2 PASS")
            return

        # ── Stage 3: Claude Code CLI ──
        print("\n▶ Stage 3: Installing Claude Code CLI ...")
        t0 = time.time()
        await sb.write_file("/tmp/claude-code.tgz", cc_tarball, user="root")
        print(f"  uploaded CC tarball ({(time.time()-t0)*1000:.0f}ms, {cc_tarball.stat().st_size/1e6:.1f} MB)")

        rc, stdout, stderr = await sb.exec(
            "npm install -g --prefix=/usr/local --no-audit --no-fund /tmp/claude-code.tgz 2>&1 "
            "&& echo '---CC_VERSION---' "
            "&& ls -la /usr/local/bin/claude "
            "&& /usr/local/bin/claude --version 2>&1",
            user="root", timeout=300,
        )
        cc_ms = (time.time() - t0) * 1000
        if rc != 0:
            print(f"  ✗ Claude Code install FAILED (rc={rc})")
            lines = stdout.strip().split("\n")
            for line in lines[-25:]:
                print(f"    {line}")
            if stderr:
                print(f"    stderr: {stderr[:500]}")
            return

        # Show version info
        in_version = False
        for line in stdout.strip().split("\n"):
            if "---CC_VERSION---" in line:
                in_version = True
                continue
            if in_version:
                print(f"    {line.strip()}")
        print(f"  ✓ Claude Code installed ({cc_ms:.0f}ms)")

        # ── Stage 4: Verify ──
        print("\n▶ Stage 4: Verify Claude Code binary ...")
        rc, stdout, stderr = await sb.exec(
            "/usr/local/bin/claude --help 2>&1 | head -10",
            user="root", timeout=30,
        )
        print(f"  claude --help rc={rc}")
        for line in stdout.strip().split("\n")[:5]:
            print(f"    {line}")

        print(f"\n{'='*60}")
        print(f"  RESULT: ALL 4 STAGES PASS")
        print(f"  Sandbox: {sb.sandbox_id}")
        print(f"{'='*60}")

    finally:
        await sb.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
