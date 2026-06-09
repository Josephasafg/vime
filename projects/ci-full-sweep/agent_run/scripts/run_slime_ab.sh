#!/usr/bin/env bash
# run_slime_ab.sh — slime cross-framework A/B control for a vime CI test failure.
#
# When a vime test fails in run_ci_sweep.sh, run the SAME scenario against upstream slime
# (sglang rollout) in the slime baseline image. Interpretation:
#   slime PASSES, vime FAILS  -> vime-specific regression (most likely the vllm rollout path,
#                                since vime = slime modulo sglang->vllm).
#   slime FAILS too           -> shared/environment/upstream issue, not a vime regression.
#
# slime is pre-installed in the image at /root/slime (NO pip install -e). slime's own
# tests/<TEST_FILE> drives the scenario; it does its own convert_checkpoint / hf download.
# Env var names differ from vime: slime uses SLIME_TEST_* (vime uses VIME_TEST_*) — this
# script translates. Models/datasets/HF-cache mounted like the proven ci_h200_heavy runner.
#
# Usage:
#   run_slime_ab.sh <slime_test_file.py> [num_gpus]
# Env overrides:
#   IMAGE        slime baseline image   (default: inferactinc/public:slime-baseline-nightly-20260519a)
#   MODELS_DIR   model+dataset flat dir (default: /home/aoshen/models-shared)
#   HF_CACHE_DIR hf cache               (default: /home/aoshen/.cache/huggingface)
#   RUNS_ROOT    NFS results root       (default: /home/aoshen/vime-test-runs/ci-full-sweep)
#   USE_DEEPEP / USE_FP8_ROLLOUT / ENABLE_EVAL   -> SLIME_TEST_* (defaults 0/0/1)
#   TEST_ARGS    extra args to the test (e.g. --async-save)
#   PULL=1       pull the slime image if absent (~70GB); default 0 (fail fast if absent)
set -uo pipefail

TEST_FILE="${1:?usage: run_slime_ab.sh <slime_test_file.py> [num_gpus]}"
NUM_GPUS="${2:-8}"
IMAGE="${IMAGE:-inferactinc/public:slime-baseline-nightly-20260519a}"
MODELS_DIR="${MODELS_DIR:-/home/aoshen/models-shared}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/aoshen/.cache/huggingface}"
RUNS_ROOT="${RUNS_ROOT:-/home/aoshen/vime-test-runs/ci-full-sweep}"
USE_DEEPEP="${USE_DEEPEP:-0}"
USE_FP8_ROLLOUT="${USE_FP8_ROLLOUT:-0}"
ENABLE_EVAL="${ENABLE_EVAL:-1}"
TEST_ARGS="${TEST_ARGS:-}"
PULL="${PULL:-0}"

DK="docker"; docker ps >/dev/null 2>&1 || DK="sudo -n docker"
$DK ps >/dev/null 2>&1 || { echo "FATAL: cannot run docker" >&2; exit 2; }
[[ -d "$MODELS_DIR" ]] || { echo "FATAL: MODELS_DIR missing: $MODELS_DIR" >&2; exit 2; }

if ! $DK image inspect "$IMAGE" >/dev/null 2>&1; then
  if [[ "$PULL" == "1" ]]; then echo "[slime-ab] pulling $IMAGE (~70GB)..."; $DK pull "$IMAGE" || { echo "FATAL: pull failed" >&2; exit 2; }
  else echo "FATAL: slime image absent: $IMAGE  (re-run with PULL=1 to fetch ~70GB)" >&2; exit 2; fi
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
TAG="$(basename "$TEST_FILE" .py)"
OUT="$RUNS_ROOT/slime-ab/${TAG}-${TS}"; mkdir -p "$OUT/torchinductor"
LOG="$OUT/run.log"
CN="slime-ab-${TAG}-$$"

cleanup(){ $DK rm -f "$CN" >/dev/null 2>&1 || true; }
trap cleanup EXIT TERM INT

echo "== slime A/B == test=$TEST_FILE gpus=$NUM_GPUS deepep=$USE_DEEPEP fp8=$USE_FP8_ROLLOUT eval=$ENABLE_EVAL"
echo "  image=$IMAGE  out=$OUT"

$DK run -d --name "$CN" \
  --privileged --cap-add SYS_NICE --security-opt seccomp=unconfined \
  --network host --gpus all --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 --memory=0 --memory-swap=0 \
  -e PYTHONPATH=/root/Megatron-LM -e NCCL_CUMEM_ENABLE=1 \
  -e WANDB_MODE=offline -e WANDB_DISABLED=true \
  -e SLIME_TEST_ENABLE_INFINITE_RUN=false \
  -e SLIME_TEST_USE_DEEPEP="$USE_DEEPEP" \
  -e SLIME_TEST_USE_FP8_ROLLOUT="$USE_FP8_ROLLOUT" \
  -e SLIME_TEST_ENABLE_EVAL="$ENABLE_EVAL" \
  -e HF_HOME=/root/.cache/huggingface \
  -e TORCHINDUCTOR_CACHE_DIR=/root/runs/torchinductor \
  -v "$MODELS_DIR:/root/models" \
  -v "$MODELS_DIR:/root/datasets" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -v "$OUT:/root/runs" \
  -w /root/slime "$IMAGE" sleep infinity >>"$LOG" 2>&1 \
  || { echo "FATAL: container start failed (see $LOG)" >&2; exit 1; }

# slime baseline image ships torch_memory_saver_hook_mode_preload_cu12.abi3.so; slime expects
# the un-suffixed name (LD_PRELOAD assert). cu12 is correct on this cu12 image.
$DK exec "$CN" bash -lc 'P=$(python3 -c "import site,glob,os;print(next(iter(glob.glob(os.path.join(site.getsitepackages()[0],\"torch_memory_saver_hook_mode_preload*\"))),\"\"))" 2>/dev/null);
  B=/usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload;
  [ -f ${B}.abi3.so ] || ln -sf ${B}_cu12.abi3.so ${B}.abi3.so 2>/dev/null; ls -l ${B}*.abi3.so 2>/dev/null' >>"$LOG" 2>&1

# Resolve the slime test path (slime's in-tree tests/).
$DK exec "$CN" bash -lc "test -f tests/$TEST_FILE || { echo 'SLIME_TEST_MISSING: tests/$TEST_FILE not in slime image (vime-only test? no A/B possible)'; exit 3; }" >>"$LOG" 2>&1
rc_probe=$?
if [[ $rc_probe -eq 3 ]]; then
  echo "[slime-ab] tests/$TEST_FILE not present in slime image -> no A/B (vime-only scenario). See $LOG"
  exit 3
fi

t0=$(date +%s)
$DK exec \
  -e NCCL_CUMEM_ENABLE=1 -e WANDB_MODE=offline \
  -e SLIME_TEST_USE_DEEPEP="$USE_DEEPEP" -e SLIME_TEST_USE_FP8_ROLLOUT="$USE_FP8_ROLLOUT" -e SLIME_TEST_ENABLE_EVAL="$ENABLE_EVAL" \
  -e TEST_ARGS="$TEST_ARGS" \
  "$CN" bash -lc '
    set -uo pipefail
    if [[ -n "${TEST_ARGS:-}" ]]; then read -r -a A < <(printf "%s\n" "$TEST_ARGS"); else A=(); fi
    if [ "'"$NUM_GPUS"'" = "0" ]; then python tests/'"$TEST_FILE"' "${A[@]}";
    else python tests/ci/gpu_lock_exec.py --count '"$NUM_GPUS"' -- python tests/'"$TEST_FILE"' "${A[@]}"; fi
    echo "EXIT_RC=$?"' 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
t1=$(date +%s)

cat > "$OUT/meta.json" <<EOF
{
  "schema": "agent_run.artifact_meta.v1",
  "kind": "slime_ab", "test_file": "$TEST_FILE", "test_args": "$TEST_ARGS",
  "num_gpus": $NUM_GPUS, "use_deepep": "$USE_DEEPEP", "use_fp8_rollout": "$USE_FP8_ROLLOUT", "enable_eval": "$ENABLE_EVAL",
  "image": "$IMAGE", "exit_rc": $rc, "duration_s": $((t1-t0)),
  "host": "$(hostname)", "ts_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "agent": "claude-opus-4-8 ci-full-sweep"
}
EOF
echo "[slime-ab] $TEST_FILE EXIT_RC=$rc log=$LOG"
exit $rc
