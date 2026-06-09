#!/bin/bash
# Launch INT4 conversion in arm64 container on rack1-02
NODE="gb200-rack1-02"
IP="192.168.0.102"
IMAGE="radixark/miles:dev-20260515-cu13-arm64"

LOG=/home/aoshen/vime/projects/vime_235b_int4/agent_run/logs/int4_convert.log
mkdir -p "$(dirname "$LOG")"

ssh "$IP" "docker run --rm \
  -v /mnt/lustre/aoshen/models:/root/models \
  -v /home/aoshen/vime:/workspace/vime \
  --network host \
  $IMAGE \
  bash /workspace/vime/projects/vime_235b_int4/agent_run/scripts/run_int4_convert.sh" \
  > "$LOG" 2>&1 &

echo "INT4 conversion launched, PID=$!, log=$LOG"
echo $! > /home/aoshen/vime/projects/vime_235b_int4/agent_run/logs/int4_convert.pid
