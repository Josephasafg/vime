#!/usr/bin/env bash
# Produce the slime-blessed per-tensor fp8 ckpt for GLM-4.5-Air rollout weight-sync.
#
# WHY: vime/slime weight-sync requant only supports quant_method=fp8 (compressed-tensors
#   branch is "only int4 at the moment" -> fake_int4_quant_cuda crash). block-fp8 breaks on
#   GLM dense ffn 10944 (∤128). The ONLY working fp8 path is `--strategy tensor`
#   (tools/convert_hf_to_fp8.py:182) -> quant_method=fp8, NO weight_block_size -> per-tensor
#   (quantizer_fp8.py:106-111, no divisibility constraint). Router/gate/norm/embed/lm_head
#   auto-kept bf16 via modules_to_not_convert.
#
# Runs the conversion inside the arm container on one held gb200 node (needs 1 GPU).
# Input  : /mnt/lustre/aoshen/glm45-air/GLM-4.5-Air            (bf16, 206G, 47 shards)
# Output : /mnt/lustre/aoshen/glm45-air/GLM-4.5-Air-FP8-tensor (~105G, per-tensor fp8)
set -euo pipefail
NODE="${NODE:-gb200-rack1-02}"
IMAGE="192.168.0.101:5000/aoshen/vime-vllm:flashqla-pb-arm"
WS=/home/aoshen/vime/projects/vime_gb200_training/workspace
SRC=/mnt/lustre/aoshen/glm45-air/GLM-4.5-Air
OUT=/mnt/lustre/aoshen/glm45-air/GLM-4.5-Air-FP8-tensor

echo "=== convert GLM-4.5-Air -> per-tensor fp8 on $NODE ==="
# pre-create output with a wide stripe over emptier OSTs (avoid the ENOSPC seen on the
# torch_dist convert; aggregate-free != per-OST-free on this lustre).
ssh -n "$NODE" "mkdir -p '$OUT' && (lfs setstripe -c 8 -o 8,9,10,11,12,13,14,15 '$OUT' 2>/dev/null || lfs setstripe -c 8 '$OUT' 2>/dev/null || true)"

ssh -n "$NODE" "docker run --rm --gpus all --network host --ipc=host --shm-size=32g \
  --ulimit memlock=-1:-1 \
  -v '$WS:/root/slime' \
  -v '/mnt/lustre/aoshen/glm45-air:/root/glm45' \
  -w /root/slime '$IMAGE' \
  bash -lc 'python tools/convert_hf_to_fp8.py \
     --model-dir /root/glm45/GLM-4.5-Air \
     --save-dir  /root/glm45/GLM-4.5-Air-FP8-tensor \
     --strategy tensor --max-workers 4'"

echo "=== done. verify config quant_method=fp8 (no weight_block_size): ==="
ssh -n "$NODE" "python3 -c \"import json;c=json.load(open('$OUT/config.json'))['quantization_config'];print('quant_method=',c.get('quant_method'),'weight_block_size=',c.get('weight_block_size'),'fmt=',c.get('fmt'),'act=',c.get('activation_scheme'));print('skipped',len(c.get('modules_to_not_convert',[])),'modules')\""
echo "Next: set GLM_FP8_CKPT=/root/glm45/GLM-4.5-Air-FP8-tensor in common_env_glm45.sh, then re-run smoke."
