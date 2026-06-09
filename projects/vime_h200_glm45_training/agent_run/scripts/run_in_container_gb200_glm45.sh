#!/bin/bash
# GLM-4.5-Air (106B-A12B) MoE  vime/vLLM RL training on GB200 — 16 GPU = 4 nodes x 4, colocate.
# Runs INSIDE the arm64 container on ALL 4 nodes; behavior branches on NODE_RANK.
#   node0 (rank0): ray head -> wait for 16 GPUs -> ray job submit (train.py)
#   node1-3      : ray worker join -> stay alive until head exits
#
# User-pinned (LOCKED) parallelism:
#   train  TP4 / PP2 / CP2  (world 16 -> DP1) , EP8 / ETP1 , MoE dispatcher alltoall (NCCL; cross-node EP)
#   rollout TP2 (--rollout-num-gpus-per-engine 2 -> 16/2 = 8 engines) , FP8 , moe-backend cutlass (benchmark winner)
# Topology: TP4 = innermost = 1 gb200 node (NVLink); CP2 x PP2 span the 4 nodes (MNNVL/NCCL);
#   EP8 spans 2 nodes per PP stage (same cross-node EP as the proven 2-node 30B run -> alltoall NCCL).
# FP8 rollout mechanism (vime): rollout --hf-checkpoint is a block-fp8 ckpt (config.json quant_method=fp8);
#   trainer master is bf16 (--ref-load torch_dist); weight-sync converts bf16->fp8 inline. NO --vllm-quantization.
# Parsers: NONE — vime rollout is token-only (/inference/v1/generate); glm45 reasoning/tool parsers not used.
set -ex

source /root/abcfg/common_env_glm45.sh            # NUM_GPUS=4, NUM_NODES=4, WANDB_PROJECT, RUN_TAG, NUM_ROLLOUT
set +x; [ -f /root/abcfg/secrets.env ] && source /root/abcfg/secrets.env; set -x   # WANDB_API_KEY — guarded, never logged

export PYTHONBUFFERED=16
NODE_RANK="${NODE_RANK:?NODE_RANK must be set (0=head .. 3=worker)}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR must be set (head enp0s3 ip)}"
NEED_GPUS=$(( NUM_NODES * NUM_GPUS ))

# --- torch_memory_saver .so name workaround (arm image ships _cu12 only) ---
TMS_DIR=$(python3 -c 'import os,torch_memory_saver as t;print(os.path.dirname(os.path.dirname(t.__file__)))' 2>/dev/null || echo /usr/local/lib/python3.12/dist-packages)
BASE=torch_memory_saver_hook_mode_preload
if [ ! -e "$TMS_DIR/${BASE}.abi3.so" ] && [ -e "$TMS_DIR/${BASE}_cu12.abi3.so" ]; then
    ln -sf "$TMS_DIR/${BASE}_cu12.abi3.so" "$TMS_DIR/${BASE}.abi3.so"
fi

ray stop --force || true; pkill -9 ray || true; pkill -9 python || true; pkill -9 -f "vllm serve" || true
rm -rf /tmp/ray /tmp/ray_sockets 2>/dev/null || true       # clear stale ray Redis session state
sleep 3

# --- GB200 cross-node NCCL/MNNVL (cuMem ON engages MNNVL/NVLink-fabric; enp0s3 = mgmt net) ---
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0
export NCCL_CUMEM_ENABLE=1
export NCCL_SOCKET_IFNAME=enp0s3
export GLOO_SOCKET_IFNAME=enp0s3
export NCCL_DEBUG=WARN
WORKER_IP="${WORKER_IP:-$(ip -o -4 addr show enp0s3 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}"
echo "NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR WORKER_IP=$WORKER_IP NEED_GPUS=$NEED_GPUS HAS_NVLINK=$HAS_NVLINK RUN_TAG=$RUN_TAG NUM_ROLLOUT=$NUM_ROLLOUT"

# ============================ WORKER (node1-3) ============================
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
source /root/slime/scripts/models/glm4.5-106B-A12B.sh   # MODEL_ARGS (byte-identical to slime's)

# FP8 rollout: --hf-checkpoint = block-fp8 ckpt (vLLM auto-detects fp8 from config). bf16 master via --ref-load.
CKPT_ARGS=(
   --hf-checkpoint ${GLM_FP8_CKPT:?set GLM_FP8_CKPT (GLM-4.5-Air-FP8 block-quant dir)}
   --ref-load ${GLM_REF_LOAD:?set GLM_REF_LOAD (GLM-4.5-Air_torch_dist bf16 dir)}
)
ROLLOUT_ARGS=(
   --prompt-data ${DAPO_DATA:?set DAPO_DATA}
   --input-key prompt --label-key label --apply-chat-template --rollout-shuffle
   --rm-type deepscaler
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 32 --n-samples-per-prompt 8
   --rollout-max-response-len ${ROLLOUT_MAX_RESP:-8192} --rollout-temperature 1
   --global-batch-size 256 --balance-data
   --rollout-stop-token-ids 151329 151336 151338     # GLM family stop tokens (verify vs ckpt tokenizer_config)
)
EVAL_ARGS=()   # eval disabled for bring-up; re-enable (aime, 32768) for L3 once green

# world 16: TP4 x PP2 x CP2 = 16 -> DP1. EP8/ETP1. recompute selective/moe (escalate to full if OOM).
PERF_ARGS=(
   --tensor-model-parallel-size 4 --sequence-parallel
   --pipeline-model-parallel-size 2 --context-parallel-size 2
   --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
   --recompute-granularity ${RECOMPUTE_GRAN:-selective} --recompute-modules moe
   --use-dynamic-batch-size --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
)
GRPO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2 --eps-clip-high 0.28
)
OPTIMIZER_ARGS=(
   --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
   --adam-beta1 0.9 --adam-beta2 0.98
   --use-precision-aware-optimizer
   # add --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d ONLY if 106B optim states OOM
)
MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
   --moe-token-dispatcher-type alltoall              # NCCL: required for gb200 cross-node EP8 (deepep crashes xnode)
   --train-memory-margin-bytes 2147483648
)
BACKEND_ARGS=(
   --rollout-num-gpus-per-engine 2 --vllm-gpu-memory-utilization 0.9   # rollout TP2 (user), 16/2 = 8 engines
   --vllm-server-concurrency 128
   --vllm-moe-backend triton      # per-tensor fp8: cutlass MoE DISABLED for per-tensor scale (oracle/fp8.py:302 ValueError); triton handles per-tensor. (cutlass benchmark win was on per-channel ckpt, now unused)
   --vllm-router-policy consistent_hash
)
WANDB_ARGS=(
   --use-wandb --wandb-project ${WANDB_PROJECT}
   --wandb-group ${RUN_TAG} --wandb-dir /root/runs/wandb
)

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --port=6379
echo "[head] waiting for ${NEED_GPUS} GPUs across ${NUM_NODES} nodes ..."
until ray status 2>/dev/null | grep -qE "/${NEED_GPUS}\.0 GPU"; do sleep 5; done
echo "[head] cluster has ${NEED_GPUS} GPUs; submitting job."

set +x
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/slime:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"HF_HOME\": \"/root/hf\",
    \"VIME_ROLLOUT_PROF\": \"1\",
    \"NCCL_CUMEM_ENABLE\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_SOCKET_IFNAME\": \"enp0s3\",
    \"GLOO_SOCKET_IFNAME\": \"enp0s3\",
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
