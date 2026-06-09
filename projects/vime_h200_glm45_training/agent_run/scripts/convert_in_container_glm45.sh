#!/bin/bash
set -ex
MODE="${1:?need MODE: torch_dist|fp8}"
export PYTHONPATH=/root/slime:/root/Megatron-LM/
cd /root/slime
source scripts/models/glm4.5-106B-A12B.sh
HF=/root/glm45/GLM-4.5-Air
if [ "$MODE" = "torch_dist" ]; then
  torchrun --nproc-per-node 4 tools/convert_hf_to_torch_dist.py \
     ${MODEL_ARGS[@]} \
     --hf-checkpoint "$HF/" \
     --save /root/glm45/GLM-4.5-Air_torch_dist/
elif [ "$MODE" = "fp8" ]; then
  python tools/convert_hf_to_fp8.py \
     --model-dir "$HF" \
     --save-dir /root/glm45/GLM-4.5-Air-FP8 \
     --strategy block --block-size 128 128 --max-workers 8
fi
echo "CONVERT_DONE mode=$MODE"
