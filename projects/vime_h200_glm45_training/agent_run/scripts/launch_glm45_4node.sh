#!/usr/bin/env bash
# GB200 4-node colocate GLM-4.5-Air L3 run.
#   head  rack1-02 (192.168.0.102)
#   workers rack1-08/16/17 (.108/.116/.117)
# All 4 nodes confirmed free + docker-capable (ran benchmark containers) at 2026-06-08T04Z.
set -euo pipefail
P=/home/aoshen/vime/projects/vime_h200_glm45_training/agent_run
LN="$P/scripts/launch_node_gb200_glm45.sh"
export RUN_TAG="${RUN_TAG:-glm45air-colo-$(date +%Y%m%d_%H%M%S)}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"
export NUM_NODES=4
M=192.168.0.102     # head = rack1-02

declare -A NODES
NODES[0]="gb200-rack1-02 192.168.0.102 head"
NODES[1]="gb200-rack1-01 192.168.0.101 w1"   # was 08 (= eval-qwen35 9396, other agent); 01 is free per user
NODES[2]="gb200-rack1-16 192.168.0.116 w2"
NODES[3]="gb200-rack1-17 192.168.0.117 w3"

echo "GLM45 4-node RUN_TAG=$RUN_TAG NUM_ROLLOUT=$NUM_ROLLOUT  nodes=02(head)/08/16/17"
for r in 0 1 2 3; do
  read -r host ip tag <<< "${NODES[$r]}"
  OUT="$P/results/run-$RUN_TAG/node$r-$tag"; mkdir -p "$OUT"
  ssh -n "$host" "MASTER_ADDR='$M' NUM_NODES=4 RUN_TAG='$RUN_TAG' NUM_ROLLOUT='$NUM_ROLLOUT' nohup bash '$LN' $r '$ip' '$OUT' glm45-$tag > '$OUT/nohup.out' 2>&1 &"
  echo "  rank$r -> $host ($tag)"
  sleep 2
done
echo "launched. head log: $P/results/run-$RUN_TAG/node0-head/run.log"
