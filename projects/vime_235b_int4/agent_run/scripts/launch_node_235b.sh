#!/bin/bash
# args: <node_rank 0..3> <worker_ip> <out_dir> <container_name>
# env : MASTER_ADDR, RUN_TAG, NUM_ROLLOUT, NUM_NODES=4
set -euo pipefail
RANK="${1:?node_rank}"; WIP="${2:?worker_ip}"; OUT="${3:?out_dir}"; CN="${4:?container}"
P=/home/aoshen/vime/projects/vime_235b_int4/agent_run
WS=/home/aoshen/vime/projects/vime_gb200_training/workspace
IMAGE="192.168.0.101:5000/aoshen/vime-vllm:flashqla-pb-arm"
MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR}"
NUM_GPUS=4; NUM_NODES="${NUM_NODES:-4}"
RUN_TAG="${RUN_TAG:-qwen3-235b-int4-colo}"
NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
mkdir -p "$OUT"
{ echo "=== launch $(date -u +%FT%TZ) node=$(hostname) rank=$RANK image=$IMAGE ==="
  echo "  MASTER=$MASTER_ADDR worker_ip=$WIP NUM_NODES=$NUM_NODES RUN_TAG=$RUN_TAG NUM_ROLLOUT=$NUM_ROLLOUT"; } > "$OUT/run.log"
docker rm -f "$CN" 2>/dev/null || true
exec docker run --rm --name "$CN" \
  --gpus all --network host --ipc=host --shm-size=32g \
  --device=/dev/nvidia-caps-imex-channels/channel0 --device=/dev/infiniband \
  --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  -e NODE_RANK="$RANK" -e MASTER_ADDR="$MASTER_ADDR" -e WORKER_IP="$WIP" \
  -e NUM_GPUS="$NUM_GPUS" -e NUM_NODES="$NUM_NODES" \
  -e RUN_TAG="$RUN_TAG" -e NUM_ROLLOUT="$NUM_ROLLOUT" \
  -e NCCL_CUMEM_ENABLE=1 -e NCCL_NVLS_ENABLE=1 \
  -e HF_HOME=/root/hf -e HF_HUB_ENABLE_HF_TRANSFER=1 \
  -v "$WS:/root/vime" \
  -v "$P/scripts:/root/abcfg:ro" \
  -v "/mnt/lustre/aoshen/models:/root/models:ro" \
  -v "/mnt/lustre/aoshen/datasets:/root/datasets:ro" \
  -v "/mnt/lustre/hf-models:/root/hf:ro" \
  -v "/mnt/data/235b-ckpt:/root/localckpt" \
  -v "$OUT:/root/runs" \
  -w /root/vime "$IMAGE" \
  bash -lc "bash /root/abcfg/run_in_container_235b.sh" >> "$OUT/run.log" 2>&1
