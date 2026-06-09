#!/bin/bash
# Coding-agent RL — Qwen3-30B-A3B-Instruct-2507 on GB200 (2 nodes × 4 GPU, colocate)
#
# Aligned with the 235B prod script (train_qwen3_235b_prod_verda.sh):
#   - Model: Qwen3-30B-A3B-Instruct-2507 (256K native context, same as 235B Instruct)
#   - Context: max_prompt=4K, max_response=128K (SWE trajectory mean ~70K)
#   - max-tokens-per-gpu = (4K + 128K) / CP (dynamic batch + sequence packing)
#   - Batch: rollout_batch=8, n_samples=8, global_batch=64
#   - Coding-agent pipeline: Modal sandbox + Claude Code + hermes tool parser
#
# Usage: run INSIDE the container on BOTH nodes.
#   NODE_RANK=0 (head): ray head → submit job
#   NODE_RANK=1 (worker): ray worker → join cluster
set -ex

source /root/abcfg/common_env.sh
set +x; [ -f /root/abcfg/secrets.env ] && source /root/abcfg/secrets.env; set -x

export PYTHONBUFFERED=16
NODE_RANK="${NODE_RANK:?NODE_RANK must be set (0=head, 1=worker)}"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR must be set}"
NEED_GPUS=$(( NUM_NODES * NUM_GPUS ))

# --- install modal SDK + sglang parser (not in the base image) ---
pip install -q modal==1.4.2 2>/dev/null || pip install -q modal 2>/dev/null
pip install -q sglang --no-deps 2>/dev/null
python3 -c "import modal; print(f'modal {modal.__version__}')"
python3 -c "from sglang.srt.function_call.function_call_parser import FunctionCallParser; print('sglang FunctionCallParser OK')"

# --- torch_memory_saver .so name workaround ---
TMS_DIR=$(python3 -c 'import os,torch_memory_saver as t;print(os.path.dirname(os.path.dirname(t.__file__)))' 2>/dev/null || echo /usr/local/lib/python3.12/dist-packages)
BASE=torch_memory_saver_hook_mode_preload
if [ ! -e "$TMS_DIR/${BASE}.abi3.so" ] && [ -e "$TMS_DIR/${BASE}_cu12.abi3.so" ]; then
    ln -sf "$TMS_DIR/${BASE}_cu12.abi3.so" "$TMS_DIR/${BASE}.abi3.so"
fi

# --- cleanup stale ray ---
ray stop --force || true; pkill -9 ray || true; pkill -9 python || true; sleep 3

# --- GB200 NCCL/MNNVL ---
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
[ "$NVLINK_COUNT" -gt 0 ] && HAS_NVLINK=1 || HAS_NVLINK=0
export NCCL_CUMEM_ENABLE=1
export NCCL_SOCKET_IFNAME=enp0s3
export GLOO_SOCKET_IFNAME=enp0s3
export NCCL_DEBUG=WARN
WORKER_IP="${WORKER_IP:-$(ip -o -4 addr show enp0s3 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}"

echo "NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR WORKER_IP=$WORKER_IP NEED_GPUS=$NEED_GPUS HAS_NVLINK=$HAS_NVLINK"

# ========================= WORKER =========================
if [ "$NODE_RANK" != "0" ]; then
  echo "[worker] waiting for ray head at ${MASTER_ADDR}:6379 ..."
  until bash -c "echo > /dev/tcp/${MASTER_ADDR}/6379" 2>/dev/null; do sleep 3; done
  ray start --address="${MASTER_ADDR}:6379" --num-gpus ${NUM_GPUS} --node-ip-address "${WORKER_IP}" --disable-usage-stats
  echo "[worker] joined; staying alive."
  while ray status --address="${MASTER_ADDR}:6379" >/dev/null 2>&1; do sleep 30; done
  exit 0
fi

# ========================= HEAD =========================
source /root/slime/scripts/models/qwen3-30B-A3B.sh   # MODEL_ARGS (bridge handles Instruct variant)

# --- Context lengths (aligned with 235B prod) ---
max_prompt_length=$((1024 * 4))          # 4K
max_response_length=$((1024 * 128))      # 128K
ACTOR_CP=2                               # 2 nodes, CP=2

# --- Paths ---
# Qwen3-30B-A3B-Instruct-2507 (256K native, no YaRN needed).
# Loaded via megatron bridge directly from HF — no torch_dist conversion needed.
HF_CKPT="/root/models/Qwen3-30B-A3B-Instruct-2507"
PROMPT_DATA="/root/data/swe_lite_smoke.jsonl"
RUN_ROOT="/root/runs/coding_agent_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_ROOT}/rollout_dumps"

# --- cloudflared tunnel URL (started on HOST by launch script; read from env) ---
ADAPTER_URL="${ADAPTER_URL_OVERRIDE:?ADAPTER_URL_OVERRIDE must be set by launch script}"
echo "[head] tunnel URL: ${ADAPTER_URL}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${HF_CKPT}"
)

# Coding-agent RL — aligned with 235B prod batch/context config.
ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt --label-key label --metadata-key metadata
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 8
   --n-samples-per-prompt 8
   --rollout-max-context-len $((max_prompt_length + max_response_length))
   --rollout-max-response-len ${max_response_length}
   --rollout-temperature 1.0
   --num-steps-per-rollout 1
   --global-batch-size 64
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)

PERF_ARGS=(
   --tensor-model-parallel-size 4 --sequence-parallel
   --pipeline-model-parallel-size 1 --context-parallel-size ${ACTOR_CP}
   --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
   --recompute-granularity selective --recompute-modules moe
   --use-dynamic-batch-size
   --max-tokens-per-gpu $(((max_prompt_length + max_response_length) / ACTOR_CP))
   --log-probs-chunk-size 1024
)

ALGO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2 --eps-clip-high 0.28
   --use-rollout-logprobs
)

OPTIMIZER_ARGS=(
   --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
   --adam-beta1 0.9 --adam-beta2 0.98
   --use-precision-aware-optimizer
   --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
)

MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash
   --moe-token-dispatcher-type alltoall
   --train-memory-margin-bytes 2147483648
)

BACKEND_ARGS=(
   --rollout-num-gpus-per-engine 2 --vllm-gpu-memory-utilization 0.5
   --vllm-server-concurrency 128
   --vllm-all2all-backend deepep_high_throughput
   --vllm-router-policy consistent_hash
   --vllm-tool-call-parser hermes
)

WANDB_ARGS=(
   --use-wandb --wandb-project ${WANDB_PROJECT}
   --wandb-group ${RUN_TAG} --wandb-dir /root/runs/wandb
)

# --- ray head ---
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus ${NUM_GPUS} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --port=6379
echo "[head] waiting for ${NEED_GPUS} GPUs ..."
until ray status 2>/dev/null | grep -qE "/${NEED_GPUS}\.0 GPU"; do sleep 5; done
echo "[head] cluster has ${NEED_GPUS} GPUs; submitting job."

# --- submit ---
set +x
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/slime:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"HF_HOME\": \"/root/hf\",
    \"NCCL_CUMEM_ENABLE\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_SOCKET_IFNAME\": \"enp0s3\",
    \"GLOO_SOCKET_IFNAME\": \"enp0s3\",
    \"NCCL_DEBUG\": \"WARN\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_PROJECT\": \"${WANDB_PROJECT}\",
    \"VIME_AGENT_SANDBOX_BACKEND\": \"modal\",
    \"MODAL_TOKEN_ID\": \"${MODAL_TOKEN_ID}\",
    \"MODAL_TOKEN_SECRET\": \"${MODAL_TOKEN_SECRET}\",
    \"ADAPTER_URL_OVERRIDE\": \"${ADAPTER_URL}\",
    \"SWE_HOST_NODE_TARBALL\": \"/root/data/node-v22.20.0-linux-x64.tar.xz\",
    \"SWE_HOST_CC_TARBALL\": \"/root/data/anthropic-ai-claude-code-2.1.168.tgz\",
    \"SWE_TIME_BUDGET_SEC\": \"1800\",
    \"SWE_EVAL_TIMEOUT_SEC\": \"600\",
    \"SWE_BOOT_CONCURRENCY\": \"8\",
    \"VIME_AGENT_SANDBOX_MODAL_BLOCK_NETWORK\": \"false\"
  }
}"
set -x

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --train-backend megatron \
   --actor-num-nodes ${NUM_NODES} \
   --actor-num-gpus-per-node ${NUM_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${OPTIMIZER_ARGS[@]} ${ALGO_ARGS[@]} \
   ${PERF_ARGS[@]} ${BACKEND_ARGS[@]} ${MISC_ARGS[@]} ${WANDB_ARGS[@]}
