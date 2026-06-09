#!/bin/bash
# Run inside arm64 container on ALL 4 GB200 nodes for Qwen3-235B INT4 training
# node0 (rank0): ray head -> wait for 16 GPUs -> ray job submit
# node1-3      : ray worker join -> stay alive until head exits
set -ex
source /root/abcfg/common_env_235b.sh
set +x; [ -f /root/abcfg/secrets.env ] && source /root/abcfg/secrets.env; set -x

export PYTHONPATH=/root/vime:/root/Megatron-LM/
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_CUMEM_ENABLE=1
export NCCL_NVLS_ENABLE=1

cd /root/vime

NODE_RANK="${NODE_RANK:?NODE_RANK must be set}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR must be set}"
WORKER_IP="${WORKER_IP:-${MASTER_ADDR}}"
NEED_GPUS=$((NUM_NODES * NUM_GPUS))
HAS_NVLINK=$(nvidia-smi | grep -c "NVLink" || true)

echo "NODE_RANK=$NODE_RANK MASTER=$MASTER_ADDR WORKER_IP=$WORKER_IP NEED_GPUS=$NEED_GPUS"

# Worker path
if [ "$NODE_RANK" != "0" ]; then
  echo "[worker] waiting for ray head at ${MASTER_ADDR}:6379 ..."
  until bash -c "echo > /dev/tcp/${MASTER_ADDR}/6379" 2>/dev/null; do sleep 3; done
  ray start --address="${MASTER_ADDR}:6379" --num-gpus "${NUM_GPUS}" \
    --node-ip-address "${WORKER_IP}" --disable-usage-stats
  echo "[worker rank=$NODE_RANK] joined. Waiting for head to exit."
  while ray status --address="${MASTER_ADDR}:6379" >/dev/null 2>&1; do sleep 30; done
  exit 0
fi

# Head path (rank 0)
MODEL_ARGS=(
   --disable-bias-linear --qk-layernorm --group-query-attention
   --num-attention-heads 64 --num-query-groups 4 --kv-channels 128
   --num-layers 94 --hidden-size 4096 --ffn-hidden-size 12288
   --normalization RMSNorm --position-embedding-type rope
   --norm-epsilon 1e-6 --rotary-percent 1.0 --swiglu
   --untie-embeddings-and-output-weights --vocab-size 151936
   --rotary-base 5000000
   --moe-ffn-hidden-size 1536 --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall --moe-router-topk 8
   --num-experts 128 --moe-grouped-gemm --moe-token-drop-policy probs
   --moe-router-dtype fp32 --moe-permute-fusion --moe-aux-loss-coeff 0
)

CKPT_ARGS=(
   --hf-checkpoint "${Q235B_BF16_CKPT}" \
   --rollout-hf-checkpoint "${Q235B_INT4_CKPT}"
   ${Q235B_REF_LOAD:+--ref-load "${Q235B_REF_LOAD}"}
   --save /root/localckpt/Qwen3-235B-A22B-vime/
   --save-interval 20
)

OPTIMIZER_ARGS=(
   --optimizer adam --lr 1e-6 --lr-decay-style constant
   --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

ROLLOUT_ARGS=(
   --prompt-data "${DAPO_DATA}"
   --input-key prompt --label-key label --apply-chat-template
   --rollout-shuffle --rm-type deepscaler
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 16 --n-samples-per-prompt 8
   --rollout-max-response-len 8192 --rollout-temperature 0.8
   --global-batch-size 128 --balance-data
)

# Train: TP4/PP4/CP1/EP8 on 16 GPUs
# CPU optimizer: 235B/(TP4*PP4)=14.7B params/rank × 176GB/rank × 4 = 704GB/node < 882GB RAM
PERF_ARGS=(
   --tensor-model-parallel-size 4 --sequence-parallel
   --pipeline-model-parallel-size 4 --decoder-last-pipeline-num-layers 22
   --context-parallel-size 1
   --expert-model-parallel-size 4 --expert-tensor-parallel-size 1
   --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --recompute-granularity "${RECOMPUTE_GRAN}"
   --recompute-method "${RECOMPUTE_METHOD}"
   --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
   --attention-softmax-in-fp32
   --attention-dropout 0.0 --hidden-dropout 0.0
   --attention-backend flash --no-check-for-nan-in-loss-and-grad
   --moe-token-dispatcher-type alltoall
   --fp8-format e4m3
   --fp8-recipe blockwise
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss: disabled for smoke (ref model needs 470GB lustre, only 385GB free)
   --kl-loss-coef 0.00 --kl-loss-type low_var_kl
   --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 --use-tis
)

WANDB_ARGS=(
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group "${RUN_TAG}" --wandb-dir /root/runs/wandb
)

BACKEND_ARGS=(
   --rollout-num-gpus-per-engine 4
   --vllm-gpu-memory-utilization 0.9
   --vllm-server-concurrency 64
   --vllm-moe-backend triton
   --vllm-router-policy consistent_hash
   --vllm-enforce-eager
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/vime:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_CUMEM_ENABLE\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"1\",
    \"NCCL_TIMEOUT_MS\": \"360000000\",
    \"OPEN_TRAINING_INT4_FAKE_QAT_FLAG\": \"1\",
    \"OPEN_TRAINING_INT4_GROUP_SIZE\": \"128\",
    \"NVTE_FP8_BLOCK_SCALING_FP32_SCALES\": \"1\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_PROJECT\": \"${WANDB_PROJECT}\"
  }
}"

# Start Ray head
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --port=6379
echo "[head] waiting for ${NEED_GPUS} GPUs across ${NUM_NODES} nodes ..."
until ray status 2>/dev/null | grep -qE "/${NEED_GPUS}\.0 GPU"; do sleep 5; done
echo "[head] cluster has ${NEED_GPUS} GPUs; submitting job."

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes "${NUM_NODES}" \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${BACKEND_ARGS[@]}" \
   --megatron-to-hf-mode bridge \
   --train-memory-margin-bytes 0
