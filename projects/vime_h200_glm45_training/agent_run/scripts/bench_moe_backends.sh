#!/usr/bin/env bash
# Benchmark 4 FP8 MoE backends for GLM-4.5-Air on GB200 (TP2, mem_util 0.9).
# Launches vllm serve on 4 nodes in parallel, waits for all healthy, benchmarks each.
set -euo pipefail

P=/home/aoshen/vime/projects/vime_h200_glm45_training/agent_run
WS=/home/aoshen/vime/projects/vime_gb200_training/workspace
IMAGE="192.168.0.101:5000/aoshen/vime-vllm:flashqla-pb-arm"
MODEL=/root/glm45/GLM-4.5-Air-FP8
TP=2
MEM_UTIL=0.9
PORT=8100

declare -A NODES
NODES[auto]="gb200-rack1-02"
NODES[deep_gemm]="gb200-rack1-08"
NODES[triton]="gb200-rack1-16"
NODES[cutlass]="gb200-rack1-17"

OUTDIR="$P/results/moe-backend-bench"
mkdir -p "$OUTDIR"

echo "=== Launching 4 vllm serve instances ($(date -u +%FT%TZ)) ==="

for backend in auto deep_gemm triton cutlass; do
  node=${NODES[$backend]}
  cn="glm45-bench-$backend"
  echo "  $node -> $cn (--moe-backend $backend)"

  ssh -n "$node" "docker rm -f $cn 2>/dev/null || true; \
    docker run -d --name $cn --gpus all --network host --ipc=host --shm-size=32g \
      --device=/dev/nvidia-caps-imex-channels/channel0 --device=/dev/infiniband \
      --cap-add=IPC_LOCK --ulimit memlock=-1:-1 --ulimit stack=67108864 \
      -v /mnt/lustre/aoshen/glm45-air:/root/glm45:ro \
      -v /mnt/lustre/hf-models:/root/hf:ro \
      -e HF_HOME=/root/hf \
      -w /root \
      $IMAGE \
      bash -lc 'vllm serve $MODEL --tensor-parallel-size $TP --gpu-memory-utilization $MEM_UTIL --moe-backend $backend --port $PORT --trust-remote-code --max-model-len 8192 --disable-log-requests 2>&1 | tee /tmp/serve.log'" \
    2>&1 | head -1
done

echo "=== Waiting for all 4 engines to be healthy ==="
# Each node's vllm serve listens on PORT; poll /health
declare -A IPS
IPS[auto]="192.168.0.102"
IPS[deep_gemm]="192.168.0.108"
IPS[triton]="192.168.0.116"
IPS[cutlass]="192.168.0.117"

MAX_WAIT=600
for backend in auto deep_gemm triton cutlass; do
  ip=${IPS[$backend]}
  echo -n "  waiting for $backend ($ip:$PORT) ..."
  elapsed=0
  until curl -sf "http://$ip:$PORT/health" >/dev/null 2>&1; do
    sleep 10; elapsed=$((elapsed+10))
    if [ $elapsed -ge $MAX_WAIT ]; then
      echo " TIMEOUT after ${MAX_WAIT}s"
      echo "  log tail:"; ssh -n "${NODES[$backend]}" "docker logs glm45-bench-$backend 2>&1 | tail -20" 2>/dev/null
      exit 1
    fi
  done
  echo " UP (${elapsed}s)"
done

echo "=== All 4 engines healthy. Running benchmark ==="

# Benchmark: send batches of requests, measure tokens/sec
# Use vLLM's /v1/completions with a fixed prompt, max_tokens, and concurrency
PROMPT="Solve the following math problem step by step. Show your work.\n\nProblem: Find all positive integers n such that n^2 + 1 is divisible by n + 1.\n\nSolution:"
MAX_TOKENS=512
N_REQUESTS=32
CONCURRENCY=8

for backend in auto deep_gemm triton cutlass; do
  ip=${IPS[$backend]}
  outf="$OUTDIR/bench_${backend}.jsonl"
  echo "--- benchmarking $backend ($ip) : $N_REQUESTS reqs, concurrency=$CONCURRENCY, max_tokens=$MAX_TOKENS ---"

  # Warmup (2 requests)
  for i in 1 2; do
    curl -sf "http://$ip:$PORT/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"Hello\",\"max_tokens\":16}" >/dev/null 2>&1 &
  done
  wait
  sleep 2

  # Timed benchmark
  START=$(date +%s%N)
  for i in $(seq 1 $N_REQUESTS); do
    (
      resp=$(curl -sf "http://$ip:$PORT/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0.7}" 2>&1)
      # Extract usage
      usage=$(echo "$resp" | python3 -c "import sys,json; u=json.load(sys.stdin).get('usage',{}); print(json.dumps(u))" 2>/dev/null || echo "{}")
      echo "{\"backend\":\"$backend\",\"req\":$i,\"usage\":$usage}" >> "$outf"
    ) &
    # Concurrency limiter
    if (( i % CONCURRENCY == 0 )); then wait; fi
  done
  wait
  END=$(date +%s%N)
  ELAPSED_MS=$(( (END - START) / 1000000 ))

  # Aggregate
  TOTAL_COMPLETION=$(python3 -c "
import json
toks=0
with open('$outf') as f:
    for line in f:
        u=json.loads(line).get('usage',{})
        toks+=u.get('completion_tokens',0)
print(toks)
" 2>/dev/null || echo 0)
  TPS=$(python3 -c "print(f'{$TOTAL_COMPLETION / ($ELAPSED_MS / 1000):.1f}')" 2>/dev/null || echo "?")
  echo "  $backend: ${ELAPSED_MS}ms total, ${TOTAL_COMPLETION} completion tokens, ${TPS} tok/s"
  echo "{\"backend\":\"$backend\",\"elapsed_ms\":$ELAPSED_MS,\"total_completion_tokens\":$TOTAL_COMPLETION,\"tokens_per_sec\":$TPS}" >> "$OUTDIR/summary.jsonl"
done

echo "=== Benchmark complete. Results in $OUTDIR/summary.jsonl ==="
cat "$OUTDIR/summary.jsonl"

echo "=== Cleaning up containers ==="
for backend in auto deep_gemm triton cutlass; do
  ssh -n "${NODES[$backend]}" "docker stop glm45-bench-$backend 2>/dev/null; docker rm -f glm45-bench-$backend 2>/dev/null" &
done
wait
echo "=== Done ==="
