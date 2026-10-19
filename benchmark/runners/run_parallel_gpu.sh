#!/usr/bin/env bash
# Run up to 2 launch_libero_paired.py jobs of THIS checkout at once under one shared GPU lock (SL5h).
#
# The script takes /tmp/ag3s-vla-gpu-validation.lock itself (exec 9>>lock; flock -w N 9), checks once that no
# other GPU compute process runs and that the pod working set (memory.current - inactive_file) plus the estimate
# (--est-gib-per-run x concurrent jobs) stays <= --limit-gib, then runs the jobs with AG3S_GPU_LOCK_HELD=1 so each
# launcher verifies the inherited lock instead of taking it. Other checkouts keep waiting on the same lock, so
# runs of different checkouts never overlap. The lock is released when the script and all its launchers exit.
#
#   source scripts/setup/env.sh
#   benchmark/runners/run_parallel_gpu.sh --jobs jobs.txt [--max-parallel 2] [--est-gib-per-run 12] \
#       [--limit-gib 64] [--lock-wait 600] [--stagger-s 30] [--log PATH]
#
# jobs.txt: one launch_libero_paired.py argument line per job (shell quoting; blank lines and # comments skipped):
#   --task-ids 0 --episode-ids 0,1 --run-id sl5h-a --conversion B --ag3s-version V11AB-pb5
#   --task-ids 1 --episode-ids 0,1 --run-id sl5h-b --conversion B --ag3s-version V11AB-pb5
#
# Exit: 0 all jobs rc=0, 1 some job failed, 2 refused (lock / busy GPU / memory / bad arguments).
# Test-only overrides (tests/benchmark/test_parallel_gpu_launcher.py): AG3S_PARALLEL_TEST_LOCK,
# AG3S_PARALLEL_TEST_LAUNCHER, AG3S_PARALLEL_TEST_NVIDIA_SMI, AG3S_PARALLEL_TEST_CGROUP_DIR.
set -u -o pipefail
set -m  # each job in its own process group, SIGINT not ignored: forwarded SIGINT lets launchers stop their servers

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS="" MAX_PAR=2 EST_GIB=12 LIMIT_GIB=64 LOCK_WAIT=600 STAGGER=30 LOG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --max-parallel) MAX_PAR="$2"; shift 2 ;;
    --est-gib-per-run) EST_GIB="$2"; shift 2 ;;
    --limit-gib) LIMIT_GIB="$2"; shift 2 ;;
    --lock-wait) LOCK_WAIT="$2"; shift 2 ;;
    --stagger-s) STAGGER="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    -h|--help) sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

LOCK="${AG3S_PARALLEL_TEST_LOCK:-/tmp/ag3s-vla-gpu-validation.lock}"
NVSMI="${AG3S_PARALLEL_TEST_NVIDIA_SMI:-nvidia-smi}"
CGROUP="${AG3S_PARALLEL_TEST_CGROUP_DIR:-/sys/fs/cgroup}"
PY="$ROOT/.venv-ag3s/bin/python"
if [ -n "${AG3S_PARALLEL_TEST_LAUNCHER:-}" ]; then
  read -r -a LAUNCHER <<< "$AG3S_PARALLEL_TEST_LAUNCHER"
else
  LAUNCHER=("$PY" "$ROOT/benchmark/runners/launch_libero_paired.py")
fi

[ -n "$LOG" ] || LOG="$ROOT/outputs/parallel_gpu/$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
mkdir -p "$(dirname "$LOG")"
: >> "$LOG"

now() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) (KST $(TZ=KST-9 date +%H:%M:%S))"; }
log() { echo "$(now) $*" | tee -a "$LOG" >&2; }
refuse() { log "REFUSED: $*"; exit 2; }
ws_bytes() {  # pod working set = memory.current - inactive_file
  local cur inact
  cur=$(cat "$CGROUP/memory.current") || return 1
  inact=$(awk '$1 == "inactive_file" {print $2}' "$CGROUP/memory.stat") || return 1
  echo $((cur - inact))
}
gib() { awk -v b="$1" 'BEGIN {printf "%.2f", b / 1073741824}'; }

[ "${AG3S_PROJECT_ROOT:-}" = "$ROOT" ] || refuse "source $ROOT/scripts/setup/env.sh first (AG3S_PROJECT_ROOT=${AG3S_PROJECT_ROOT:-unset})"
[ -n "$JOBS" ] && [ -f "$JOBS" ] || refuse "--jobs FILE is required (got '${JOBS}')"
case "$MAX_PAR" in 1|2) ;; *) refuse "--max-parallel must be 1 or 2 (pod rule: servers <= 2), got $MAX_PAR" ;; esac
for v in "$EST_GIB" "$LIMIT_GIB" "$LOCK_WAIT" "$STAGGER"; do
  [[ "$v" =~ ^[0-9]+([.][0-9]+)?$ ]] || refuse "numeric option expected, got '$v'"
done

# parse every job line before taking the lock (shlex: no shell evaluation of the job text)
LINES=()
while IFS= read -r line || [ -n "$line" ]; do
  [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
  "$PY" -c 'import shlex, sys; shlex.split(sys.argv[1])' "$line" 2>/dev/null || refuse "cannot parse job line: $line"
  LINES+=("$line")
done < "$JOBS"
N=${#LINES[@]}
[ "$N" -gt 0 ] || refuse "no job lines in $JOBS"
CONC=$(( N < MAX_PAR ? N : MAX_PAR ))
log "batch jobs=$N max_parallel=$MAX_PAR concurrent=$CONC jobs_file=$JOBS lock=$LOCK commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)"

# 1. shared lock, held on fd 9 by this script; launchers inherit fd 9 (held_by_caller check)
exec 9>>"$LOCK" || refuse "cannot open $LOCK"
t0=$(date +%s)
flock -w "$LOCK_WAIT" 9 || refuse "lock $LOCK not acquired within ${LOCK_WAIT}s"
log "lock acquired after $(( $(date +%s) - t0 ))s"

# 2. no other GPU compute process (checked once; launchers do not wait in held_by_caller mode)
apps=$("$NVSMI" --query-compute-apps=pid,used_memory --format=csv,noheader) || refuse "$NVSMI failed"
apps=$(echo "$apps" | sed '/^[[:space:]]*$/d')
[ -z "$apps" ] || refuse "other GPU compute processes running: $(echo "$apps" | tr '\n' ';')"

# 3. memory: working set + estimate x concurrent <= limit
ws=$(ws_bytes) || refuse "cannot read $CGROUP/memory.current / memory.stat"
need=$(awk -v w="$ws" -v e="$EST_GIB" -v c="$CONC" 'BEGIN {printf "%.2f", w / 1073741824 + e * c}')
awk -v n="$need" -v l="$LIMIT_GIB" 'BEGIN {exit !(n <= l)}' \
  || refuse "working set $(gib "$ws") GiB + ${EST_GIB} GiB x $CONC = $need GiB > ${LIMIT_GIB} GiB"
log "preflight ok: gpu compute apps none; working set $(gib "$ws") GiB + ${EST_GIB} x $CONC = $need GiB <= ${LIMIT_GIB} GiB"

declare -A PID2JOB=()
on_signal() {
  log "signal: forwarding SIGINT to running launchers ${!PID2JOB[*]}"
  local p rc
  for p in "${!PID2JOB[@]}"; do kill -INT "$p" 2>/dev/null; done
  for p in "${!PID2JOB[@]}"; do
    wait "$p"; rc=$?
    log "job=${PID2JOB[$p]} event=end pid=$p rc=$rc (after SIGINT) ws_gib=$(gib "$(ws_bytes)")"
  done
  log "stopped by signal; lock released on exit"
  exit 130
}
trap on_signal INT TERM

start_job() {
  local i=$1 out="${LOG%.log}.job$1.out" argv=()
  mapfile -d '' -t argv < <("$PY" -c 'import shlex, sys; sys.stdout.write("\0".join(shlex.split(sys.argv[1])))' "${LINES[$i]}")
  AG3S_GPU_LOCK_HELD=1 "${LAUNCHER[@]}" "${argv[@]}" > "$out" 2>&1 &
  PID2JOB[$!]=$i
  log "job=$i event=start pid=$! ws_gib=$(gib "$(ws_bytes)") out=$out args=${LINES[$i]}"
}

next=0 failed=0
while [ "$next" -lt "$N" ] || [ "${#PID2JOB[@]}" -gt 0 ]; do
  if [ "$next" -lt "$N" ] && [ "${#PID2JOB[@]}" -lt "$MAX_PAR" ]; then
    [ "$next" -gt 0 ] && [ "${#PID2JOB[@]}" -gt 0 ] && sleep "$STAGGER"
    start_job "$next"; next=$((next + 1)); continue
  fi
  done_pid=""
  wait -n -p done_pid "${!PID2JOB[@]}"; rc=$?
  [ -n "$done_pid" ] || continue
  i=${PID2JOB[$done_pid]}; unset "PID2JOB[$done_pid]"
  [ "$rc" -eq 0 ] || failed=$((failed + 1))
  log "job=$i event=end pid=$done_pid rc=$rc ws_gib=$(gib "$(ws_bytes)")"
done

log "batch done jobs=$N failed=$failed; releasing lock"
flock -u 9; exec 9>&-
[ "$failed" -eq 0 ]
