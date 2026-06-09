#!/bin/bash
# Launch coding_agent_rl containers on GB200 rack1-01 (head) + rack1-03 (worker).
# Run from the edit host (aoshen's dev machine, NOT inside a container).
set -ex

# ===== CONFIG =====
HEAD_HOST="gb200-rack1-01"
WORKER_HOST="gb200-rack1-03"
HEAD_IP="192.168.0.101"                                     # rack1-01 enp0s3
WORKER_IP="192.168.0.103"                                   # rack1-03 enp0s3
IMAGE="192.168.0.101:5000/aoshen/vime-vllm:flashqla-pb-arm" # arm64, cu129
CONTAINER_PREFIX="cagent"

# Paths inside the container (bind-mounted)
LUSTRE="/mnt/lustre"
VIME_SRC="${LUSTRE}/aoshen/vime-wt/coding-agent-swe"       # worktree will be rsynced here
DATA_DIR="${LUSTRE}/aoshen/cagent-data"                     # tarballs + dataset

# ===== Prep: sync worktree + data to lustre =====
WORKTREE_SRC="/home/aoshen/vime/projects/vime_modal_sandbox/vime"
echo "Syncing worktree to ${HEAD_HOST}:${VIME_SRC} ..."
ssh ${HEAD_HOST} "mkdir -p ${VIME_SRC}"
rsync -az --delete --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  "${WORKTREE_SRC}/" "${HEAD_HOST}:${VIME_SRC}/"

echo "Syncing data (tarballs + dataset) ..."
ssh ${HEAD_HOST} "mkdir -p ${DATA_DIR}"
rsync -az /tmp/node-v22.20.0-linux-x64.tar.xz "${HEAD_HOST}:${DATA_DIR}/"
rsync -az /tmp/cc-pack/anthropic-ai-claude-code-2.1.168.tgz "${HEAD_HOST}:${DATA_DIR}/"
rsync -az /home/aoshen/vime/projects/vime_modal_sandbox/data/swe_lite_smoke.jsonl "${HEAD_HOST}:${DATA_DIR}/"
rsync -az /home/aoshen/vime/projects/vime_modal_sandbox/scripts/run_coding_agent_gb200.sh "${HEAD_HOST}:${DATA_DIR}/"

# Secrets (WANDB + Modal)
rsync -az /home/aoshen/vime/projects/vime_modal_sandbox/secrets/secrets.env "${HEAD_HOST}:${DATA_DIR}/secrets.env"

# common_env override for coding_agent
ssh ${HEAD_HOST} "cat > ${DATA_DIR}/common_env.sh" <<'ENVEOF'
export NUM_GPUS="${NUM_GPUS:-4}"
export NUM_NODES="${NUM_NODES:-2}"
export WANDB_PROJECT="vime-coding-agent-swe"
export RUN_TAG="${RUN_TAG:-qwen3-30b-cagent-smoke}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
ENVEOF

# ===== Stop old containers =====
for host in ${HEAD_HOST} ${WORKER_HOST}; do
  ssh ${host} "docker rm -f ${CONTAINER_PREFIX}-0 ${CONTAINER_PREFIX}-1 2>/dev/null || true"
done

# ===== Start cloudflared on HEAD HOST (not in container; --net=host shares localhost) =====
echo "Starting cloudflared tunnel on ${HEAD_HOST} ..."
SHIM_PORT=18001
ssh ${HEAD_HOST} "pkill -f 'cloudflared tunnel' 2>/dev/null || true; sleep 1"
ssh ${HEAD_HOST} "nohup cloudflared tunnel --url http://localhost:${SHIM_PORT} --no-autoupdate > /tmp/cagent_cloudflared.log 2>&1 &"
echo "  Waiting 15s for edge warmup ..."
sleep 15
ADAPTER_URL=$(ssh ${HEAD_HOST} "grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cagent_cloudflared.log | head -1")
if [ -z "$ADAPTER_URL" ]; then
  echo "ERROR: cloudflared did not provide tunnel URL"
  ssh ${HEAD_HOST} "cat /tmp/cagent_cloudflared.log"
  exit 1
fi
echo "  Tunnel URL: ${ADAPTER_URL}"

# ===== Launch containers =====
DOCKER_COMMON=(
  --gpus all --shm-size=128g --net=host --ipc=host
  --ulimit memlock=-1 --ulimit nofile=1048576:1048576 --ulimit stack=8388608:8388608
  -v /mnt/lustre:/mnt/lustre
  -v /mnt/data:/mnt/data
)

# HEAD (rack1-01)
echo "Launching head container on ${HEAD_HOST} ..."
ssh ${HEAD_HOST} "docker run -d --name ${CONTAINER_PREFIX}-0 ${DOCKER_COMMON[*]} \
  -e NODE_RANK=0 \
  -e MASTER_ADDR=${HEAD_IP} \
  -e MODAL_TOKEN_ID=\$(grep MODAL_TOKEN_ID ${DATA_DIR}/secrets.env 2>/dev/null | cut -d= -f2- | tr -d '\"' || echo '') \
  -e MODAL_TOKEN_SECRET=\$(grep MODAL_TOKEN_SECRET ${DATA_DIR}/secrets.env 2>/dev/null | cut -d= -f2- | tr -d '\"' || echo '') \
  -e ADAPTER_URL_OVERRIDE=${ADAPTER_URL} \
  ${IMAGE} bash -c '
    # Link vime source + data
    ln -sfn ${VIME_SRC} /root/slime
    ln -sfn ${DATA_DIR} /root/data
    mkdir -p /root/abcfg
    cp ${DATA_DIR}/common_env.sh /root/abcfg/common_env.sh
    cp ${DATA_DIR}/secrets.env /root/abcfg/secrets.env 2>/dev/null || true
    # Link model checkpoints (Megatron-LM is already in the image at /root/Megatron-LM)
    mkdir -p /root/ref && ln -sfn ${LUSTRE}/aoshen/models/Qwen3-30B-A3B_torch_dist /root/ref/Qwen3-30B-A3B_torch_dist
    mkdir -p /root/hf/hub/models--Qwen--Qwen3-30B-A3B/snapshots
    ln -sfn ${LUSTRE}/aoshen/models/Qwen3-30B-A3B /root/hf/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
    cd /root/slime
    bash /root/data/run_coding_agent_gb200.sh
  '"

# WORKER (rack1-03)
echo "Launching worker container on ${WORKER_HOST} ..."
ssh ${WORKER_HOST} "docker run -d --name ${CONTAINER_PREFIX}-1 ${DOCKER_COMMON[*]} \
  -e NODE_RANK=1 \
  -e MASTER_ADDR=${HEAD_IP} \
  ${IMAGE} bash -c '
    ln -sfn ${VIME_SRC} /root/slime
    ln -sfn ${DATA_DIR} /root/data
    mkdir -p /root/abcfg
    cp ${DATA_DIR}/common_env.sh /root/abcfg/common_env.sh
    cd /root/slime
    bash /root/data/run_coding_agent_gb200.sh
  '"

echo "===================================="
echo "  Containers launched."
echo "  Head:   ssh ${HEAD_HOST} && docker logs -f ${CONTAINER_PREFIX}-0"
echo "  Worker: ssh ${WORKER_HOST} && docker logs -f ${CONTAINER_PREFIX}-1"
echo "  Ray dashboard: http://${HEAD_IP}:8265"
echo "===================================="
