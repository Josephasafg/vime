#!/bin/bash
set -ex
echo "=== INT4 convert START (SYMMETRIC) $(date -u +%FT%TZ) ==="
cd /root/vime
python tools/convert_hf_to_int4_direct.py \
  --model-dir /root/models/Qwen3-235B-A22B-Instruct-2507 \
  --save-dir  /root/models/Qwen3-235B-A22B-INT4 \
  --group-size 128 \
  --is-symmetric \
  --max-workers 32
echo "=== INT4 convert DONE $(date -u +%FT%TZ) ==="
