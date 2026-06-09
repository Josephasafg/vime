#!/usr/bin/env bash
# run_ci_sweep.sh — Faithful local re-runner of the vime GitHub-Actions CI matrix.
#
# Runs ON an h200 run-host. Per ci_matrix.json entry it reproduces the CI docker
# invocation from .github/workflows/pr-test.yml.j2 (verified @ ae5d6b04), with two
# deliberate substitutions documented in ci_matrix.json:
#   1. model/dataset mount: CI's /mnt/nvme0n1/vime_ci/{models,datasets} (CI staging,
#      absent here) -> h200 $MODELS_DIR. CI binds these RW; the real tests `hf download
#      --local-dir /root/models/{X}` and hf_download_dataset() WRITE into them (at least
#      .cache/huggingface metadata), so a :ro bind would break them. We default to an
#      OVERLAY (ro lower = models-shared, per-container tmpfs upper) so writes are absorbed
#      per-run and the shared store stays pristine; MODELS_MODE=rw reproduces CI's raw RW
#      bind. CPU jobs mount no models at all (CI CPU runners have none).
#      convert_checkpoint writes torch_dist to /root/{X}_torch_dist (container layer, NOT
#      the mount), so that path is unaffected by the mount mode.
#   2. TORCHINDUCTOR_CACHE_DIR redirected to NFS (root disk is ~88% full).
# GPU jobs invoke `gpu_lock_exec.py --count $NUM_GPUS -- python tests/<test>.py`
# exactly as CI does; flock auto-serializes concurrent GPU jobs on the host.
#
# Usage:
#   run_ci_sweep.sh <selector> [selector ...]
#     selector := "all" | "cpu" | "gpu" | <group> | <entry-id> | <test_file.py>
#   "all" excludes optional/needs_image entries. Use explicit ids to force those.
#
# Env overrides:
#   MATRIX_JSON   path to ci_matrix.json           (default: alongside this script)
#   WORKTREE      vime checkout to mount as -w      (default: ../../vime-worktree resolved)
#   MODELS_DIR    h200 model+dataset flat dir       (default: /home/aoshen/models-shared)
#   RUNS_ROOT     NFS results root                  (default: /home/aoshen/vime-test-runs/ci-full-sweep)
#   IMAGE         default image                     (default: inferactinc/public:vime-latest)
#   RUN_TAG       run subdir name                   (default: UTC timestamp)
#   DRY_RUN=1     print docker cmd, do not execute
#   KEEP_GOING=1  continue after a failing entry    (default: 1; set 0 to stop on first fail)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATRIX_JSON="${MATRIX_JSON:-$SCRIPT_DIR/ci_matrix.json}"
MODELS_DIR="${MODELS_DIR:-/home/aoshen/models-shared}"
RUNS_ROOT="${RUNS_ROOT:-/home/aoshen/vime-test-runs/ci-full-sweep}"
IMAGE="${IMAGE:-inferactinc/public:vime-latest}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_GOING="${KEEP_GOING:-1}"
# Model/dataset mount policy for GPU jobs (CPU jobs never mount models, matching CI):
#   overlay (default) — models-shared bound read-only as the overlay LOWER; a per-container
#                       tmpfs absorbs ALL writes (hf-download metadata, freshly downloaded
#                       datasets). Faithful to CI's RW semantics, but models-shared stays
#                       pristine. Requires --privileged (GPU jobs have it).
#   rw              — bind models-shared directly RW at /root/{models,datasets}, exactly like
#                       CI. Simpler, but hf-download metadata / new datasets land in the SHARED
#                       store (pollution). Use only if overlay is unavailable.
MODELS_MODE="${MODELS_MODE:-overlay}"
OVL_TMPFS_SIZE="${OVL_TMPFS_SIZE:-96g}"
# HF cache: the pre-staged hf cache (195G) lets `hf download --local-dir` cache-hit instead
# of re-downloading from the network. Proven runners mount it at /root/.cache/huggingface
# (HF_HOME points there). Set HF_CACHE_DIR="" to disable. Mounted RW (cache is meant to grow);
# set HF_CACHE_RO=1 to mount it read-only.
HF_CACHE_DIR="${HF_CACHE_DIR:-/home/aoshen/.cache/huggingface}"
HF_CACHE_RO="${HF_CACHE_RO:-0}"

# Resolve WORKTREE: prefer env, else projects/ci-full-sweep/vime-worktree relative to script.
if [[ -z "${WORKTREE:-}" ]]; then
  CAND="$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd)/vime-worktree"
  WORKTREE="$CAND"
fi

command -v jq >/dev/null 2>&1 || { echo "FATAL: jq not found" >&2; exit 2; }
[[ -f "$MATRIX_JSON" ]] || { echo "FATAL: matrix not found: $MATRIX_JSON" >&2; exit 2; }
[[ -d "$WORKTREE/.git" || -f "$WORKTREE/.git" ]] || { echo "FATAL: WORKTREE not a git checkout: $WORKTREE" >&2; exit 2; }
[[ -d "$MODELS_DIR" ]] || { echo "FATAL: MODELS_DIR missing: $MODELS_DIR" >&2; exit 2; }

DK="docker"; docker ps >/dev/null 2>&1 || DK="sudo -n docker"
$DK ps >/dev/null 2>&1 || { echo "FATAL: cannot run docker (tried 'docker' and 'sudo -n docker')" >&2; exit 2; }

WORKTREE_COMMIT="$(git -C "$WORKTREE" rev-parse HEAD 2>/dev/null || echo unknown)"
RUN_DIR="$RUNS_ROOT/$RUN_TAG"
mkdir -p "$RUN_DIR"
SUMMARY_TSV="$RUN_DIR/summary.tsv"
[[ -f "$SUMMARY_TSV" ]] || printf 'id\tgroup\tnum_gpus\texit_rc\tduration_s\tlog\n' > "$SUMMARY_TSV"

echo "== ci-full-sweep runner =="
echo "  matrix     : $MATRIX_JSON"
echo "  worktree   : $WORKTREE ($WORKTREE_COMMIT)"
echo "  models_dir : $MODELS_DIR"
echo "  run_dir    : $RUN_DIR"
echo "  image      : $IMAGE"
echo "  models_mode: $MODELS_MODE (gpu jobs; cpu jobs mount no models)"
echo "  selectors  : $*"
echo "  dry_run    : $DRY_RUN | keep_going : $KEEP_GOING"
echo

# ---- preflight: sweep stray vime-* containers (do NOT touch others' jobs) ----
preflight_sweep() {
  local stray
  stray="$($DK ps --format '{{.Names}}' 2>/dev/null | grep -E '^vime-' || true)"
  if [[ -n "$stray" ]]; then
    echo "[preflight] killing stray vime-* containers:"; echo "$stray" | sed 's/^/    /'
    for c in $stray; do $DK kill "$c" >/dev/null 2>&1 || true; done
  else
    echo "[preflight] no stray vime-* containers"
  fi
}

# ---- selection ----
# Build the full entry-id list, then filter by selectors.
mapfile -t ALL_IDS < <(jq -r '.entries[].id' "$MATRIX_JSON")
declare -A SELECTED=()
want() {
  local sel="$1" id grp tf opt
  for id in "${ALL_IDS[@]}"; do
    grp="$(jq -r --arg i "$id" '.entries[]|select(.id==$i).group' "$MATRIX_JSON")"
    tf="$(jq -r --arg i "$id" '.entries[]|select(.id==$i).test_file' "$MATRIX_JSON")"
    opt="$(jq -r --arg i "$id" '.entries[]|select(.id==$i)|(.optional // false)|tostring' "$MATRIX_JSON")"
    cpu="$(jq -r --arg i "$id" '.entries[]|select(.id==$i)|(.cpu // false)|tostring' "$MATRIX_JSON")"
    case "$sel" in
      all) [[ "$opt" == "true" ]] || SELECTED[$id]=1 ;;
      cpu) [[ "$cpu" == "true" && "$opt" != "true" ]] && SELECTED[$id]=1 ;;
      gpu) [[ "$cpu" != "true" && "$opt" != "true" ]] && SELECTED[$id]=1 ;;
      "$id") SELECTED[$id]=1 ;;
      "$grp") [[ "$opt" != "true" ]] && SELECTED[$id]=1 ;;
      "$tf") SELECTED[$id]=1 ;;
    esac
  done
}
[[ $# -ge 1 ]] || { echo "usage: $0 <selector> [selector ...]   (see header)" >&2; exit 2; }
for s in "$@"; do want "$s"; done
[[ ${#SELECTED[@]} -ge 1 ]] || { echo "FATAL: no entries matched selectors: $*" >&2; exit 2; }

# preserve matrix order
mapfile -t RUN_IDS < <(for id in "${ALL_IDS[@]}"; do [[ -n "${SELECTED[$id]:-}" ]] && echo "$id"; done)
echo "[plan] ${#RUN_IDS[@]} entries to run:"; printf '    %s\n' "${RUN_IDS[@]}"; echo

run_one() {
  local id="$1"
  local j; j="$(jq -c --arg i "$id" '.entries[]|select(.id==$i)' "$MATRIX_JSON")"
  local test_file num_gpus cpu test_args use_deepep use_fp8 enable_eval ent_image extra_pip pytest group
  test_file="$(jq -r '.test_file' <<<"$j")"
  num_gpus="$(jq -r '.num_gpus' <<<"$j")"
  cpu="$(jq -r '(.cpu // false)|tostring' <<<"$j")"
  test_args="$(jq -r '(.test_args // "")' <<<"$j")"
  use_deepep="$(jq -r '(.use_deepep // "0")' <<<"$j")"
  use_fp8="$(jq -r '(.use_fp8_rollout // "0")' <<<"$j")"
  enable_eval="$(jq -r '(.enable_eval // "1")' <<<"$j")"
  ent_image="$(jq -r '(.image // "")' <<<"$j")"
  extra_pip="$(jq -r '(.extra_pip_deps // "")' <<<"$j")"
  pytest="$(jq -r '(.pytest // false)|tostring' <<<"$j")"
  group="$(jq -r '.group' <<<"$j")"
  local img="${ent_image:-$IMAGE}"

  local out="$RUN_DIR/$id"; mkdir -p "$out/torchinductor"
  local log="$out/run.log"
  local cname="vime-cisweep-${id}-$$"

  # Inner test command (shared by cpu/gpu paths). pytest entries override.
  local inner
  if [[ "$pytest" == "true" ]]; then
    # GPU + PER-FILE process isolation => 192/192 (per ci_cpu_unit_gate_audit). NOTE: --forked
    # is NOT enough — pytest collects ALL modules in one process before forking, and a
    # module-import-time sys.modules stub in test_update_weight_from_tensor.py (_install_stubs()
    # at top level) shadows `vllm`/`megatron`, breaking a sibling's collection
    # (test_vllm_engine.py -> ModuleNotFoundError vllm.engine). Only invoking pytest separately
    # per file gives each file a fresh process/sys.modules. gpu_lock_exec holds 1 GPU for the
    # whole loop so the 4 real device-touching tests run. set +e so one bad file doesn't abort.
    inner=$'pip install -q pytest --break-system-packages; python tests/ci/gpu_lock_exec.py --count 1 -- bash -c \'set +e; rc=0; p=0; t=0; ff=""; for f in $(find tests/unit tests/utils -name "test_*.py" | sort); do echo "::::: $f :::::"; python -m pytest -q -p no:cacheprovider "$f"; r=$?; t=$((t+1)); if [ $r -eq 0 ]; then p=$((p+1)); else rc=1; ff="$ff $f"; fi; done; echo "UNIT_PER_FILE passed_files=$p/$t"; [ -n "$ff" ] && echo "FAILED_FILES:$ff"; exit $rc\''
  else
    inner='TEST_PATH="$TEST_FILE"; [[ "$TEST_PATH" != tests/* ]] && TEST_PATH="tests/$TEST_PATH";
      if [[ -n "$TEST_ARGS" ]]; then read -r -a A < <(printf "%s\n" "$TEST_ARGS"); else A=(); fi;
      if [ "$NUM_GPUS" = "0" ]; then python "$TEST_PATH" "${A[@]}";
      else python tests/ci/gpu_lock_exec.py --count "$NUM_GPUS" -- python "$TEST_PATH" "${A[@]}"; fi'
  fi
  local extra_pip_line=""
  [[ -n "$extra_pip" ]] && extra_pip_line="pip install $extra_pip --break-system-packages;"
  # CPU entries: CI installs test deps on the ubuntu-latest runner (the test files import
  # pytest etc.), but we run CPU tests INSIDE vime-latest which ships runtime deps only.
  # Replicate CI's CPU dependency install so `import pytest` & friends resolve.
  local cpu_dep_line=""
  if [[ "$cpu" == "true" ]]; then
    cpu_dep_line='pip install -q pytest numpy packaging pyyaml omegaconf tqdm httpx pybase64 pylatexenc sympy aiohttp pillow --break-system-packages;'
  fi

  # docker flags: GPU jobs get the full CI flag set + --gpus all; CPU jobs omit --gpus.
  local -a gpu_flags=()
  if [[ "$cpu" != "true" ]]; then
    gpu_flags=(--privileged --cap-add SYS_NICE --security-opt seccomp=unconfined
               --gpus all --ulimit memlock=-1 --ulimit stack=67108864 --memory=0 --memory-swap=0)
  fi

  # ---- model/dataset mount policy ----
  # CPU jobs: NO model/dataset mount (CI CPU runners on ubuntu-latest have none).
  # GPU jobs: overlay (ro lower = models-shared, tmpfs upper absorbs writes) OR rw bind.
  local -a mnt_flags=()
  local ovl_pre=""
  # GPU jobs mount models (overlay/rw) + HF cache; CPU jobs and the unit_pytest entry (which
  # is a 1-GPU job but needs no models/datasets) skip all model mounts.
  if [[ "$cpu" != "true" && "$pytest" != "true" ]]; then
    if [[ "$MODELS_MODE" == "overlay" ]]; then
      mnt_flags=(
        -v "$MODELS_DIR:/root/models_lower:ro"
        -v "$MODELS_DIR:/root/datasets_lower:ro"
        --tmpfs "/root/.ovl:size=$OVL_TMPFS_SIZE"
      )
      ovl_pre='mkdir -p /root/models /root/datasets /root/.ovl/m_up /root/.ovl/m_wk /root/.ovl/d_up /root/.ovl/d_wk;
        mount -t overlay overlay -o lowerdir=/root/models_lower,upperdir=/root/.ovl/m_up,workdir=/root/.ovl/m_wk /root/models;
        mount -t overlay overlay -o lowerdir=/root/datasets_lower,upperdir=/root/.ovl/d_up,workdir=/root/.ovl/d_wk /root/datasets;'
    else
      # rw (faithful to CI; writes land in the shared store)
      mnt_flags=(
        -v "$MODELS_DIR:/root/models"
        -v "$MODELS_DIR:/root/datasets"
      )
    fi
    # HF cache mount (GPU jobs only) so hf download cache-hits instead of re-downloading.
    if [[ -n "$HF_CACHE_DIR" && -d "$HF_CACHE_DIR" ]]; then
      if [[ "$HF_CACHE_RO" == "1" ]]; then
        mnt_flags+=(-v "$HF_CACHE_DIR:/root/.cache/huggingface:ro")
      else
        mnt_flags+=(-v "$HF_CACHE_DIR:/root/.cache/huggingface")
      fi
    fi
  fi

  local -a cmd=(
    "$DK" run --rm --name "$cname"
    --network host --ipc=host --shm-size=16g
    "${gpu_flags[@]}"
    -e http_proxy -e https_proxy -e HTTP_PROXY -e HTTPS_PROXY
    -e GITHUB_COMMIT_NAME="${WORKTREE_COMMIT}_cisweep"
    -e WANDB_MODE=offline -e WANDB_DISABLED=true
    -e VIME_TEST_ENABLE_INFINITE_RUN=false
    -e VIME_TEST_USE_DEEPEP="$use_deepep"
    -e VIME_TEST_USE_FP8_ROLLOUT="$use_fp8"
    -e VIME_TEST_ENABLE_EVAL="$enable_eval"
    -e TEST_FILE="$test_file" -e TEST_ARGS="$test_args" -e NUM_GPUS="$num_gpus"
    -e HF_HOME=/root/.cache/huggingface
    -e TORCHINDUCTOR_CACHE_DIR=/root/runs/torchinductor
    -v "$WORKTREE:$WORKTREE"
    "${mnt_flags[@]}"
    -v "$out:/root/runs"
    -w "$WORKTREE"
    "$img"
    bash -lc "set -euo pipefail; pip install -e . --no-deps --break-system-packages; $cpu_dep_line $extra_pip_line $ovl_pre $inner"
  )

  echo "[$id] group=$group gpus=$num_gpus cpu=$cpu deepep=$use_deepep fp8=$use_fp8 eval=$enable_eval img=$img"
  if [[ "$DRY_RUN" == "1" ]]; then printf '    DRY: %q ' "${cmd[@]}"; echo; return 0; fi

  local t0 t1 rc
  t0=$(date +%s)
  "${cmd[@]}" >"$log" 2>&1; rc=$?
  t1=$(date +%s)
  local dur=$((t1-t0))
  echo "[$id] EXIT_RC=$rc duration=${dur}s log=$log"

  # provenance sidecar
  cat > "$out/meta.json" <<EOF
{
  "schema": "agent_run.artifact_meta.v1",
  "id": "$id", "group": "$group", "test_file": "$test_file", "test_args": "$test_args",
  "num_gpus": $num_gpus, "cpu": $cpu, "use_deepep": "$use_deepep", "use_fp8_rollout": "$use_fp8",
  "enable_eval": "$enable_eval", "image": "$img",
  "exit_rc": $rc, "duration_s": $dur,
  "git_commit": "$WORKTREE_COMMIT", "worktree": "$WORKTREE",
  "host": "$(hostname)", "ts_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "cmd": "docker run ... $img (see run.log)", "models_mode": "$([[ "$cpu" == "true" ]] && echo none || echo "$MODELS_MODE")", "agent": "claude-opus-4-8 ci-full-sweep"
}
EOF
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$group" "$num_gpus" "$rc" "$dur" "$log" >> "$SUMMARY_TSV"
  return $rc
}

preflight_sweep
echo
FAILED=()
for id in "${RUN_IDS[@]}"; do
  if ! run_one "$id"; then
    FAILED+=("$id")
    [[ "$KEEP_GOING" == "1" ]] || { echo "[stop] KEEP_GOING=0, halting after $id"; break; }
  fi
done

echo
echo "== sweep done: $RUN_TAG =="
echo "  summary: $SUMMARY_TSV"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  FAILED (${#FAILED[@]}): ${FAILED[*]}"
  echo "  -> A/B against slime image via run_slime_ab.sh, or bisect via bisect_ci.sh"
  exit 1
fi
echo "  ALL PASSED"
