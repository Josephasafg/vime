#!/bin/bash
# Sourced inside the arm container on all 4 gb200 nodes by run_in_container_gb200_glm45.sh.
export NUM_GPUS="${NUM_GPUS:-4}"        # gb200 = 4 GPU/node
export NUM_NODES="${NUM_NODES:-4}"      # 4 nodes x 4 = 16 GPU (TP4xPP2xCP2)
export WANDB_PROJECT="vime-gb200-glm45"
export RUN_TAG="${RUN_TAG:-glm45air-colo}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"  # smoke=3; L3 raises
# container-mounted paths (/mnt/lustre/aoshen/glm45-air -> /root/glm45)
# per-tensor fp8 (quant_method=fp8, no weight_block_size) via slime convert_hf_to_fp8 --strategy tensor.
#   vime weight-sync requant only supports quant_method=fp8 (compressed-tensors branch is int4-only);
#   block-fp8 breaks on GLM dense ffn 10944 (∤128). per-tensor has no divisibility constraint.
#   prior dead ends: GLM-4.5-Air-FP8 (self block) + GLM-4.5-Air-FP8-official (zai compressed-tensors).
export GLM_FP8_CKPT=/root/glm45/GLM-4.5-Air-FP8-tensor
export GLM_REF_LOAD=/root/glm45/GLM-4.5-Air_torch_dist
export DAPO_DATA=/root/datasets/dapo-math-17k/dapo-math-17k.jsonl
# memory-fit knobs (tune at smoke)
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}"  # raised from 16384; train side had ~78 GiB free (106/184 GiB peak)
export RECOMPUTE_GRAN="${RECOMPUTE_GRAN:-selective}"
export ROLLOUT_MAX_RESP="${ROLLOUT_MAX_RESP:-8192}"
