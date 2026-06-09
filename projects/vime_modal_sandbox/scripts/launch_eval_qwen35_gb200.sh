#!/usr/bin/env bash
# Launch eval-only: Qwen3.5-35B-A3B on 4× GB200 nodes via Slurm.
#
# Uses salloc to reserve 4 nodes, then starts docker containers on each.
# Run from the edit host (aoshen's dev machine).
#
# Usage:
#   bash scripts/launch_eval_qwen35_gb200.sh          # allocate + launch
#   SLURM_JOB_ID=<id> bash scripts/launch_eval_qwen35_gb200.sh  # reuse existing allocation
set -euo pipefail

P=/home/aoshen/vime/projects/vime_modal_sandbox
IMAGE="192.168.0.101:5000/aoshen/vime-vllm:flashqla-pb-arm"
CONTAINER_PREFIX="eval-qwen35"

# Lustre paths
LUSTRE="/mnt/lustre"
VIME_SRC="${LUSTRE}/aoshen/vime-wt/eval-qwen35-swe"
DATA_DIR="${LUSTRE}/aoshen/cagent-eval-data"
HFHUB="${LUSTRE}/hf-models"
HF_MODEL_DIR="${HFHUB}/hub/Qwen3.5-35B-A3B"

NUM_NODES=4
NUM_GPUS=4
SHIM_PORT=18001

export RUN_TAG="${RUN_TAG:-qwen35-35b-eval-$(date +%Y%m%d_%H%M%S)}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-500}"

echo "=== Qwen3.5-35B-A3B SWE-bench-Verified Eval ==="
echo "  RUN_TAG=${RUN_TAG}  NUM_ROLLOUT=${NUM_ROLLOUT}  NODES=${NUM_NODES}"

# ===== Step 1: Slurm allocation =====
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Allocating ${NUM_NODES} nodes via Slurm ..."
  ALLOC_OUTPUT=$(salloc -N ${NUM_NODES} -p batch --exclusive --gpus-per-node=4 \
    --job-name="eval-qwen35" --time=24:00:00 \
    bash -c 'echo "SLURM_JOB_ID=$SLURM_JOB_ID SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"' 2>&1 | tail -1)
  echo "  $ALLOC_OUTPUT"
  eval "$ALLOC_OUTPUT"
fi

if [ -z "${SLURM_JOB_NODELIST:-}" ]; then
  SLURM_JOB_NODELIST=$(squeue -j ${SLURM_JOB_ID} -h -o "%N" 2>/dev/null)
fi

# Expand nodelist to array
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
if [ ${#NODES[@]} -lt ${NUM_NODES} ]; then
  echo "ERROR: Expected ${NUM_NODES} nodes, got ${#NODES[@]}: ${NODES[*]}"
  exit 1
fi
HEAD_HOST="${NODES[0]}"
HEAD_IP=$(ssh -o ConnectTimeout=3 ${HEAD_HOST} "ip -o -4 addr show enp0s3 | awk '{print \$4}' | cut -d/ -f1")
echo "  Nodes: ${NODES[*]}"
echo "  Head: ${HEAD_HOST} (${HEAD_IP})"

# ===== Step 2: Sync source + data =====
echo "Syncing vime source to ${VIME_SRC} ..."
ssh ${HEAD_HOST} "mkdir -p ${VIME_SRC}"
rsync -az --delete --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  "${P}/vime/" "${HEAD_HOST}:${VIME_SRC}/"

echo "Syncing data ..."
ssh ${HEAD_HOST} "mkdir -p ${DATA_DIR}"
rsync -az /tmp/node-v22.20.0-linux-x64.tar.xz "${HEAD_HOST}:${DATA_DIR}/"
rsync -az /tmp/cc-pack/anthropic-ai-claude-code-2.1.168.tgz "${HEAD_HOST}:${DATA_DIR}/"
rsync -az "${P}/data/swe_bench_verified_500.jsonl" "${HEAD_HOST}:${DATA_DIR}/"
rsync -az "${P}/data/swe_lite_smoke.jsonl" "${HEAD_HOST}:${DATA_DIR}/"
rsync -az "${P}/scripts/run_eval_qwen35_gb200.sh" "${HEAD_HOST}:${DATA_DIR}/"
rsync -az "${P}/secrets/secrets.env" "${HEAD_HOST}:${DATA_DIR}/secrets.env"

# common_env for this eval run
ssh ${HEAD_HOST} "cat > ${DATA_DIR}/common_env.sh" <<ENVEOF
export NUM_GPUS=${NUM_GPUS}
export NUM_NODES=${NUM_NODES}
export WANDB_PROJECT="vime-swe-eval"
export RUN_TAG="${RUN_TAG}"
export NUM_ROLLOUT=${NUM_ROLLOUT}
ENVEOF

# ===== Step 3: Stop old containers =====
for host in "${NODES[@]}"; do
  ssh ${host} "docker ps -q --filter 'name=${CONTAINER_PREFIX}' | xargs -r docker rm -f" 2>/dev/null || true
done

# ===== Step 4: Start cloudflared on HEAD HOST =====
echo "Starting cloudflared tunnel on ${HEAD_HOST} ..."
ssh ${HEAD_HOST} "pkill -f 'cloudflared tunnel' 2>/dev/null || true; sleep 1"
ssh ${HEAD_HOST} "nohup cloudflared tunnel --url http://localhost:${SHIM_PORT} --no-autoupdate > /tmp/eval_cloudflared.log 2>&1 &"
echo "  Waiting 15s for edge warmup ..."
sleep 15
ADAPTER_URL=$(ssh ${HEAD_HOST} "grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/eval_cloudflared.log | head -1")
if [ -z "$ADAPTER_URL" ]; then
  echo "ERROR: cloudflared did not provide tunnel URL"
  ssh ${HEAD_HOST} "cat /tmp/eval_cloudflared.log"
  exit 1
fi
echo "  Tunnel URL: ${ADAPTER_URL}"

# ===== Step 5: Launch containers =====
DOCKER_COMMON=(
  --gpus all --shm-size=128g --net=host --ipc=host
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --ulimit stack=8388608:8388608
  --device=/dev/nvidia-caps-imex-channels/channel0 --device=/dev/infiniband
  --cap-add=IPC_LOCK
  -v /mnt/lustre:/mnt/lustre
  -v /mnt/data:/mnt/data
)

OUT_DIR="${LUSTRE}/aoshen/runs/eval-qwen35-${RUN_TAG}"
ssh ${HEAD_HOST} "mkdir -p ${OUT_DIR}"

for i in "${!NODES[@]}"; do
  host="${NODES[$i]}"
  worker_ip=$(ssh -o ConnectTimeout=3 ${host} "ip -o -4 addr show enp0s3 | awk '{print \$4}' | cut -d/ -f1")
  cn="${CONTAINER_PREFIX}-${i}"

  echo "Launching node ${i} (${host}, ${worker_ip}) ..."
  ssh ${host} "docker run -d --name ${cn} ${DOCKER_COMMON[*]} \
    -e NODE_RANK=${i} \
    -e MASTER_ADDR=${HEAD_IP} \
    -e WORKER_IP=${worker_ip} \
    -e MODAL_TOKEN_ID=\$(grep MODAL_TOKEN_ID ${DATA_DIR}/secrets.env 2>/dev/null | cut -d= -f2- | tr -d '\"' || echo '') \
    -e MODAL_TOKEN_SECRET=\$(grep MODAL_TOKEN_SECRET ${DATA_DIR}/secrets.env 2>/dev/null | cut -d= -f2- | tr -d '\"' || echo '') \
    -e ADAPTER_URL_OVERRIDE=${ADAPTER_URL} \
    ${IMAGE} bash -c '
      ln -sfn ${VIME_SRC} /root/slime
      ln -sfn ${DATA_DIR} /root/data
      mkdir -p /root/abcfg /root/models
      cp ${DATA_DIR}/common_env.sh /root/abcfg/common_env.sh
      cp ${DATA_DIR}/secrets.env /root/abcfg/secrets.env 2>/dev/null || true
      ln -sfn ${HF_MODEL_DIR} /root/models/Qwen3.5-35B-A3B
      mkdir -p /root/hf/hub
      ln -sfn ${HFHUB}/hub /root/hf/hub
      cd /root/slime
      bash /root/data/run_eval_qwen35_gb200.sh
    '"
done

echo ""
echo "===================================="
echo "  Eval launched on ${NUM_NODES} nodes."
echo "  Slurm job: ${SLURM_JOB_ID}"
echo "  Nodes: ${NODES[*]}"
echo "  Head: ssh ${HEAD_HOST} && docker logs -f ${CONTAINER_PREFIX}-0"
for i in $(seq 1 $((NUM_NODES-1))); do
  echo "  Worker ${i}: ssh ${NODES[$i]} && docker logs -f ${CONTAINER_PREFIX}-${i}"
done
echo "  Ray dashboard: http://${HEAD_IP}:8265"
echo "  Tunnel: ${ADAPTER_URL}"
echo "  Output: ${OUT_DIR}"
echo "===================================="
