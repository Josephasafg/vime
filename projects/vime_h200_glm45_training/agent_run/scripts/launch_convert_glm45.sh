#!/usr/bin/env bash
# usage: launch_convert_glm45.sh <node-alias> <MODE torch_dist|fp8> <out-subdir>
set -euo pipefail
NODE="${1:?node}"; MODE="${2:?mode}"; OUT="${3:?out}"
P=/home/aoshen/vime/projects/vime_h200_glm45_training/agent_run
WS=/home/aoshen/vime/projects/vime_gb200_training/workspace
IMAGE="192.168.0.101:5000/aoshen/vime-vllm:flashqla-pb-arm"
CN="glm45-convert-$MODE"
mkdir -p "$P/results/$OUT"
ssh -n "$NODE" "docker rm -f $CN 2>/dev/null || true; \
  docker run --rm --name $CN --gpus all --network host --ipc=host --shm-size=32g \
   --device=/dev/nvidia-caps-imex-channels/channel0 --device=/dev/infiniband \
   --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --ulimit stack=67108864 \
   -e HF_HOME=/root/hf \
   -v $WS:/root/slime -v /mnt/lustre/aoshen/glm45-air:/root/glm45 \
   -v /mnt/lustre/hf-models:/root/hf:ro -v $P/scripts:/root/abcfg:ro \
   -w /root/slime $IMAGE \
   bash -lc 'bash /root/abcfg/convert_in_container_glm45.sh $MODE' \
   > $P/results/$OUT/convert.log 2>&1 &"
echo "launched $MODE convert on $NODE -> $P/results/$OUT/convert.log"
