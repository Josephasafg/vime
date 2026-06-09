#!/bin/bash
# Sourced inside the container on both h200 nodes. RECON fills the TODO values.
export NUM_GPUS=8            # per-node (h200 = 8 GPU/node)
export NUM_NODES=2
export WANDB_PROJECT=vime-h200-glm45
export RUN_TAG=glm45air-colo-l3
export NUM_ROLLOUT=${NUM_ROLLOUT:-3}     # smoke=3; L3 raises (e.g. 3000)

# ---- RECON-FILLED (confirm on h200 before launch) ----
export NCCL_IFACE=TODO_ETH_IFACE         # mgmt/eth iface for ray+NCCL bootstrap (e.g. eno1/bond0)
export NCCL_IB_HCA=mlx5_1                 # 400G NDR per inherited plan; confirm `ibstat`/`nvidia-smi topo -m`
export GLM_HF_CKPT=TODO/GLM-4.5-Air/snapshots/<hash>
export GLM_REF_LOAD=TODO/GLM-4.5-Air_torch_dist
export DAPO_DATA=TODO/dapo-math-17k/dapo-math-17k.jsonl

# ---- memory-fit knobs (tune at smoke if OOM) ----
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
export RECOMPUTE_GRAN=${RECOMPUTE_GRAN:-selective}
export ROLLOUT_MAX_RESP=${ROLLOUT_MAX_RESP:-8192}
