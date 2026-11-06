#!/usr/bin/env bash
# nb1104 overnight GPU queue (2026-11-04). Holds the shared-pod GPU lock ONCE (fd 9, plain flock = exclusive holder)
# and runs the lines of a queue file, at most 2 at a time, with AG3S_GPU_LOCK_HELD=1 so each launcher verifies the
# inherited descriptor instead of taking the lock (same mechanism as benchmark/runners/run_parallel_gpu.sh).
# Lines can be appended while it runs. Line forms:
#   <shell command>          parallel slot (max 2 concurrent)
#   SOLO <shell command>     waits until nothing else runs, runs alone (TrajOpt closed loops: wall-clock budget)
#   STOP                     wait for running jobs, release the lock, exit
#   # comment / blank        skipped
# usage: nb1104_queue.sh QUEUE_FILE LOG_DIR
set -u
QUEUE=$1; LOGDIR=$2
cd "$(dirname "$0")/../.."
source scripts/setup/env.sh >/dev/null 2>&1
mkdir -p "$LOGDIR"
QLOG="$LOGDIR/queue.log"
log() { echo "$(TZ=KST-9 date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$QLOG"; }
exec 9>/tmp/ag3s-vla-gpu-validation.lock
log "waiting for GPU lock"
flock -w 7200 9 || { log "lock wait timeout"; exit 2; }
log "lock acquired"
export AG3S_GPU_LOCK_HELD=1
n=1
declare -A NAME=()
running() { jobs -rp | wc -l; }
reap() { for p in "${!NAME[@]}"; do if ! kill -0 "$p" 2>/dev/null; then wait "$p"; log "done job=${NAME[$p]} rc=$?"; unset "NAME[$p]"; fi; done; }
while true; do
  reap
  total=$(wc -l < "$QUEUE")
  if [ "$n" -gt "$total" ]; then sleep 15; continue; fi
  line=$(sed -n "${n}p" "$QUEUE")
  case "$line" in
    ""|\#*) n=$((n+1)); continue ;;
    STOP) log "STOP: waiting for ${#NAME[@]} running"; while [ "${#NAME[@]}" -gt 0 ]; do sleep 10; reap; done; log "exit"; exit 0 ;;
  esac
  solo=0; cmd="$line"
  if [[ "$line" == SOLO\ * ]]; then solo=1; cmd="${line#SOLO }"; fi
  if [ "$solo" = 1 ]; then
    while [ "${#NAME[@]}" -gt 0 ]; do sleep 10; reap; done
  else
    while [ "${#NAME[@]}" -ge 2 ]; do sleep 10; reap; done
  fi
  out="$LOGDIR/job$(printf %03d $n).out"
  log "start job=$n solo=$solo ws_gib=$(awk '$1=="inactive_file"{i=$2} END{}' /sys/fs/cgroup/memory.stat; echo $(( ($(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat)) / 1073741824 ))) cmd=$cmd"
  bash -c "$cmd" > "$out" 2>&1 &
  NAME[$!]=$n
  n=$((n+1))
  if [ "$solo" = 1 ]; then while [ "${#NAME[@]}" -gt 0 ]; do sleep 10; reap; done; else sleep 40; fi
done
