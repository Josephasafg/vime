#!/bin/bash
# Launch Qwen3-235B INT4 training on 4 GB200 nodes
# Nodes: rack1-02 (head), rack1-14, rack1-15, rack1-16
set -euo pipefail
P=/home/aoshen/vime/projects/vime_235b_int4/agent_run
LN="$P/scripts/launch_node_235b.sh"

# Node layout: rank → (hostname, IP, role)
declare -A NODES
NODES[0]="gb200-rack1-02 192.168.0.102 head"
NODES[1]="gb200-rack1-14 192.168.0.114 w1"
NODES[2]="gb200-rack1-15 192.168.0.115 w2"
NODES[3]="gb200-rack1-16 192.168.0.116 w3"

M="192.168.0.102"   # head node IP
NUM_NODES=4
export RUN_TAG="${RUN_TAG:-qwen3-235b-int4-$(date +%H%M%S)}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-3}"

# Make sure /mnt/data/235b-ckpt exists on each node
for r in 0 1 2 3; do
  info="${NODES[$r]}"
  ip=$(echo "$info" | awk '{print $2}')
  ssh -n "$ip" "mkdir -p /mnt/data/235b-ckpt" || true
done

OUT="$P/results/run-${RUN_TAG}"
mkdir -p "$OUT"
echo "=== Launch 235B INT4 RUN_TAG=$RUN_TAG MASTER=$M ==="

for r in 0 1 2 3; do
  info="${NODES[$r]}"
  host=$(echo "$info" | awk '{print $1}')
  ip=$(echo "$info" | awk '{print $2}')
  tag=$(echo "$info" | awk '{print $3}')
  NODE_OUT="$OUT/node${r}-${tag}"
  mkdir -p "$NODE_OUT"
  echo "  rank=$r $host ($ip) → $NODE_OUT"
  ssh -n "$ip" \
    "MASTER_ADDR='$M' NUM_NODES=$NUM_NODES RUN_TAG='$RUN_TAG' NUM_ROLLOUT='$NUM_ROLLOUT' \
     nohup bash '$LN' $r '$ip' '$NODE_OUT' 235b-$tag \
     > '$NODE_OUT/nohup.out' 2>&1 &"
done

echo ""
echo "Logs:"
echo "  head: $OUT/node0-head/run.log"
echo "  tail -f $OUT/node0-head/run.log"
