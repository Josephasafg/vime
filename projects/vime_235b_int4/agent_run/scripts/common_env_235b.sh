#!/bin/bash
# Common environment for Qwen3-235B-A22B INT4 training on GB200 4-node (16 GPU)
export NUM_GPUS="${NUM_GPUS:-4}"         # GB200 = 4 GPU/node
export NUM_NODES="${NUM_NODES:-4}"       # 4 nodes × 4 = 16 GPU
export WANDB_PROJECT="vime-gb200-235b"
export RUN_TAG="${RUN_TAG:-qwen3-235b-int4-colo}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"

# Checkpoints — container paths (lustre mounted at /root/models)
export Q235B_INT4_CKPT=/root/models/Qwen3-235B-A22B-INT4
# Q235B_REF_LOAD disabled: lustre has 385GB free, 235B torch_dist needs ~470GB.
# Re-enable when lustre space is freed. Smoke runs without ref model.
export Q235B_REF_LOAD=""
export DAPO_DATA=/root/datasets/dapo-math-17k/dapo-math-17k.jsonl

# INT4 fake QAT env (group-size 128 for Qwen3-235B)
export OPEN_TRAINING_INT4_FAKE_QAT_FLAG=1
export OPEN_TRAINING_INT4_GROUP_SIZE=128

# Performance knobs
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-4096}"
export RECOMPUTE_GRAN="${RECOMPUTE_GRAN:-full}"
export RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
# BF16 HF checkpoint for Megatron bridge loading (Megatron always trains in BF16)
export Q235B_BF16_CKPT=/root/models/Qwen3-235B-A22B-Instruct-2507
