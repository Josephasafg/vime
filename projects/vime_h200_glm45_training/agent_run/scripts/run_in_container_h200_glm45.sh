#!/bin/bash
# GLM-4.5-Air (106B-A12B) MoE  vime/vLLM RL training on H200 — 16 GPU = 2 nodes x 8, colocate.
# Runs INSIDE the x86 container on BOTH h200 nodes; behavior branches on NODE_RANK.
#   node0 (rank0): ray head -> wait for 16 GPUs -> ray job submit (train.py)
#   node1 (rank1): ray worker join -> stay alive until head exits
#
# User-pinned (LOCKED) parallelism:
#   train  TP4 / PP2 / CP2  (world 16 -> DP1) , EP8 / ETP1 , MoE dispatcher alltoall
#   rollout TP4 (--rollout-num-gpus-per-engine 4 -> 16/4 = 4 colocated engines) , FP8 (online quant)
# Recipe: reuse the proven 30B GRPO recipe (DAPO-Math-17k + deepscaler), KL off, precision-aware opt.
# GLM deltas vs the Qwen 30B harness: glm4.5 model.sh, +PP2, rollout TP2->TP4, +fp8 rollout,
#   +GLM stop-token-ids, NO reasoning/tool parser (vime rollout is token-only — verified).
set -ex

source /root/abcfg/common_env.sh                 # NUM_GPUS(per-node)=8, NUM_NODES=2, WANDB_PROJECT, RUN_TAG, NUM_ROLLOUT, paths, NCCL ifaces
set +x; [ -f /root/abcfg/secrets.env ] && source /root/abcfg/secrets.env; set -x   # WANDB_API_KEY (chmod600) — guarded, never logged

export PYTHONBUFFERED=16
NODE_RANK="${NODE_RANK:?NODE_RANK must be set (0=head, 1=worker)}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR must be set (head node ip on NCCL/ray iface)}"
NEED_GPUS=$(( NUM_NODES * NUM_GPUS ))

# --- torch_memory_saver .so name workaround (only if the image ships a suffixed variant) ---
TMS_DIR=$(python3 -c 'import os,torch_memory_saver as t;print(os.path.dirname(os.path.dirname(t.__file__)))' 2>/dev/null || echo /usr/local/lib/python3.12/dist-packages)
BASE=torch_memory_saver_hook_mode_preload
if [ ! -e "$TMS_DIR/${BASE}.abi3.so" ]; then
  for cand in "${BASE}_cu12.abi3.so" "${BASE}_cu13.abi3.so"; do
    [ -e "$TMS_DIR/$cand" ] && ln -sf "$TMS_DIR/$cand" "$TMS_DIR/${BASE}.abi3.so" && break
  done
fi

# --- cleanup any prior ray/engine state in THIS container ---
ray stop --force || true; pkill -9 ray || true; pkill -9 python || true; pkill -9 -f "vllm serve" || true; sleep 3

# --- H200 cross-node NCCL over IB (NDR 400G mlx5_1 preferred) ---
# IFACE + IB_HCA come from common_env.sh (recon-filled). Hopper: no MNNVL, no cuMem override needed.
export NCCL_SOCKET_IFNAME="${NCCL_IFACE:?set NCCL_IFACE in common_env}"
export GLOO_SOCKET_IFNAME="${NCCL_IFACE}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_1}"
export NCCL_DEBUG=WARN
WORKER_IP="${WORKER_IP:-$(ip -o -4 addr show "${NCCL_IFACE}" 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}"
echo "NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR WORKER_IP=$WORKER_IP NEED_GPUS=$NEED_GPUS IFACE=$NCCL_IFACE IB_HCA=$NCCL_IB_HCA RUN_TAG=$RUN_TAG NUM_ROLLOUT=$NUM_ROLLOUT"

# ============================ WORKER (node1) ============================
if [ "$NODE_RANK" != "0" ]; then
  echo "[worker] waiting for ray head at ${MASTER_ADDR}:6379 ..."
  until bash -c "echo > /dev/tcp/${MASTER_ADDR}/6379" 2>/dev/null; do sleep 3; done
  ray start --address="${MASTER_ADDR}:6379" --num-gpus ${NUM_GPUS} --node-ip-address "${WORKER_IP}" --disable-usage-stats
  echo "[worker] joined; staying alive until head exits."
  while ray status --address="${MASTER_ADDR}:6379" >/dev/null 2>&1; do sleep 30; done
  echo "[worker] head gone; exiting."
  exit 0
fi

# ============================ HEAD (node0) ============================
source /root/vime/scripts/models/glm4.5-106B-A12B.sh   # MODEL_ARGS (byte-identical to slime's)

# fresh-start: HF bf16 init weights + torch_dist ref-load; NO --load. BF16 master; rollout quantizes->fp8.
CKPT_ARGS=(
   --hf-checkpoint ${GLM_HF_CKPT:?set GLM_HF_CKPT (GLM-4.5-Air HF snapshot dir)}
   --ref-load ${GLM_REF_LOAD:?set GLM_REF_LOAD (GLM-4.5-Air torch_dist dir)}
)
ROLLOUT_ARGS=(
   --prompt-data ${DAPO_DATA:?set DAPO_DATA (dapo-math-17k.jsonl)}
   --input-key prompt --label-key label --apply-chat-template --rollout-shuffle
   --rm-type deepscaler
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32 --n-samples-per-prompt 8
   --rollout-max-response-len ${ROLLOUT_MAX_RESP:-8192} --rollout-temperature 1
   --global-batch-size 256 --balance-data
   # GLM family stop tokens (slime ref): <|user|> <|observation|> <|endoftext|>-ish. Verify vs GLM-4.5-Air tokenizer_config.
   --rollout-stop-token-ids 151329 151336 151338
)
# eval DISABLED for the config-bring-up smoke; re-enable for L3 (aime, max-resp 32768) once green.
EVAL_ARGS=()

# world 16: TP4 x PP2 x CP2 = 16 -> DP1. EP8 across each PP stage's 8 GPUs. ETP1.
PERF_ARGS=(
   --tensor-model-parallel-size 4 --sequence-parallel
   --pipeline-model-parallel-size 2 --context-parallel-size 2
   --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
   # recompute: start selective/moe (30B recipe); escalate to full/uniform/1 if 106B OOMs at smoke.
   --recompute-granularity ${RECOMPUTE_GRAN:-selective} --recompute-modules moe
   --use-dynamic-batch-size --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
)
GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2 --eps-clip-high 0.28
)
# KL off (kl-coef 0): drops the reference-model forward. --ref-load is the actor INIT ckpt (fresh-start).
OPTIMIZER_ARGS=(
   --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
   --adam-beta1 0.9 --adam-beta2 0.98
   --use-precision-aware-optimizer
   # add --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d ONLY if 106B optim states OOM (note open #18 ckpt bug).
)
MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
   --moe-token-dispatcher-type alltoall
   --train-memory-margin-bytes 2147483648
)
BACKEND_ARGS=(
   --rollout-num-gpus-per-engine 4 --vllm-gpu-memory-utilization 0.5
   --vllm-server-concurrency 128
   --vllm-quantization fp8                      # BF16 master -> FP8 rollout (online quant at engine load)
   --vllm-all2all-backend deepep_high_throughput
   --vllm-router-policy consistent_hash
)
WANDB_ARGS=(
   --use-wandb --wandb-project ${WANDB_PROJECT}
   --wandb-group ${RUN_TAG} --wandb-dir /root/runs/wandb
)

# --- ray head + wait for both nodes ---
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --port=6379
echo "[head] waiting for ${NEED_GPUS} GPUs across ${NUM_NODES} nodes ..."
until ray status 2>/dev/null | grep -qE "/${NEED_GPUS}\.0 GPU"; do sleep 5; done
echo "[head] cluster has ${NEED_GPUS} GPUs; submitting job."

set +x   # WANDB_API_KEY rides in RUNTIME_ENV_JSON — stop tracing so it never hits the log
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/vime:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"HF_HOME\": \"/root/hf\",
    \"VIME_ROLLOUT_PROF\": \"1\",
    \"NCCL_SOCKET_IFNAME\": \"${NCCL_IFACE}\",
    \"GLOO_SOCKET_IFNAME\": \"${NCCL_IFACE}\",
    \"NCCL_IB_HCA\": \"${NCCL_IB_HCA}\",
    \"NCCL_DEBUG\": \"WARN\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_PROJECT\": \"${WANDB_PROJECT}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --train-backend megatron \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} ${PERF_ARGS[@]} ${EVAL_ARGS[@]} ${BACKEND_ARGS[@]} ${MISC_ARGS[@]}
