#!/usr/bin/env bash
# bisect_ci.sh — git-bisect a single vime CI-matrix entry between a known-good and known-bad
# commit, running each candidate commit through run_ci_sweep.sh in the CI image.
#
# Runs ON h200 in the ci-full-sweep worktree. Each bisect step checks out a commit, then runs
# ONE matrix entry (KEEP_GOING=0). Exit 0 => good, non-0 => bad. git bisect converges to the
# first bad commit. A per-step anchor line is appended to <out>/anchors.jsonl (anchor_id,
# commit, verdict good|bad, exit_rc, log) so they can be replayed into the agent_run
# events.jsonl (state_anchor) on the gb200 side afterwards (this host has no infra clone).
#
# Usage:
#   bisect_ci.sh <entry-id> <good-commit> <bad-commit>
# Env overrides:
#   WORKTREE     vime worktree to bisect   (default: /home/aoshen/vime/.worktree/ci-full-sweep)
#   MATRIX_JSON  ci_matrix.json            (default: alongside this script)
#   RUNS_ROOT    NFS results root          (default: /home/aoshen/vime-test-runs/ci-full-sweep)
#   MODELS_MODE / IMAGE / MODELS_DIR / HF_CACHE_DIR  passed through to run_ci_sweep.sh
set -uo pipefail

ENTRY_ID="${1:?usage: bisect_ci.sh <entry-id> <good-commit> <bad-commit>}"
GOOD="${2:?need good commit}"
BAD="${3:?need bad commit}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKTREE="${WORKTREE:-/home/aoshen/vime/.worktree/ci-full-sweep}"
MATRIX_JSON="${MATRIX_JSON:-$SCRIPT_DIR/ci_matrix.json}"
RUNS_ROOT="${RUNS_ROOT:-/home/aoshen/vime-test-runs/ci-full-sweep}"
RUNNER="$SCRIPT_DIR/run_ci_sweep.sh"

[[ -x "$RUNNER" || -f "$RUNNER" ]] || { echo "FATAL: runner not found: $RUNNER" >&2; exit 2; }
[[ -d "$WORKTREE/.git" || -f "$WORKTREE/.git" ]] || { echo "FATAL: not a worktree: $WORKTREE" >&2; exit 2; }

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$RUNS_ROOT/bisect/${ENTRY_ID}-${TS}"; mkdir -p "$OUT"
ANCHORS="$OUT/anchors.jsonl"
LOG="$OUT/bisect.log"
: > "$ANCHORS"

log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

# The per-commit test runner that git-bisect calls. Exit code = verdict.
STEP="$OUT/bisect_step.sh"
cat > "$STEP" <<STEPEOF
#!/usr/bin/env bash
set -uo pipefail
C="\$(git -C "$WORKTREE" rev-parse --short HEAD)"
RT="bisect-${ENTRY_ID}-\$C"
MATRIX_JSON="$MATRIX_JSON" WORKTREE="$WORKTREE" RUNS_ROOT="$RUNS_ROOT" RUN_TAG="\$RT" \\
  KEEP_GOING=0 ${MODELS_MODE:+MODELS_MODE=$MODELS_MODE} ${IMAGE:+IMAGE=$IMAGE} \\
  ${MODELS_DIR:+MODELS_DIR=$MODELS_DIR} ${HF_CACHE_DIR:+HF_CACHE_DIR=$HF_CACHE_DIR} \\
  bash "$RUNNER" "$ENTRY_ID"
rc=\$?
verdict=good; [ \$rc -ne 0 ] && verdict=bad
LOGP="$RUNS_ROOT/\$RT/$ENTRY_ID/run.log"
printf '{"anchor_id":"%s","commit":"%s","verdict":"%s","exit_rc":%d,"entry":"%s","log":"%s","ts_utc":"%s"}\n' \\
  "B-\$C" "\$(git -C "$WORKTREE" rev-parse HEAD)" "\$verdict" "\$rc" "$ENTRY_ID" "\$LOGP" "\$(date -u +%FT%TZ)" >> "$ANCHORS"
exit \$rc
STEPEOF
chmod +x "$STEP"

log "bisect entry=$ENTRY_ID good=$GOOD bad=$BAD worktree=$WORKTREE out=$OUT"

# Save current HEAD to restore later.
ORIG="$(git -C "$WORKTREE" rev-parse HEAD)"
restore(){ git -C "$WORKTREE" bisect reset >/dev/null 2>&1 || true; git -C "$WORKTREE" checkout --quiet "$ORIG" 2>/dev/null || true; }
trap restore EXIT TERM INT

git -C "$WORKTREE" bisect start 2>&1 | tee -a "$LOG"
git -C "$WORKTREE" bisect good "$GOOD" 2>&1 | tee -a "$LOG"
git -C "$WORKTREE" bisect bad  "$BAD"  2>&1 | tee -a "$LOG"

log "=== git bisect run $STEP ==="
git -C "$WORKTREE" bisect run "$STEP" 2>&1 | tee -a "$LOG"

# Capture the first-bad commit git reports.
FIRSTBAD="$(git -C "$WORKTREE" bisect log 2>/dev/null | grep -E 'first bad commit' | tail -1)"
log "=== bisect done. first-bad: ${FIRSTBAD:-<see log>} ==="
log "anchors: $ANCHORS"
echo
echo "Replay anchors into agent_run on gb200 with, e.g.:"
echo "  while read -r j; do scripts/agent_run.py event state_anchor --repo-root <project> \\"
echo "    --anchor-id \$(jq -r .anchor_id <<<\"\$j\") --verdict \$(jq -r .verdict <<<\"\$j\") \\"
echo "    --repro-cmd 'run_ci_sweep.sh $ENTRY_ID' --config-ref '$ENTRY_ID' \\"
echo "    --log-path \$(jq -r .log <<<\"\$j\") --evidence \"exit=\$(jq -r .exit_rc <<<\"\$j\")\" --project; done < $ANCHORS"
