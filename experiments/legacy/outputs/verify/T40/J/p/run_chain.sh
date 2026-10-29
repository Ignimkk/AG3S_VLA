#!/bin/bash
# T40 J chain (copy of T40/p/run_chain.sh; snapshot ${AG3S_PROJECT_ROOT}-t40h): preflight -> server on PORT -> runs in order -> kill server.
# usage: EST_GB=<our per-server estimate> run_chain.sh COND PORT OUTDIR "ITEM ITEM ..."
#   ITEM = EP:SEED[:OUTSUBDIR[:OBST_CONFIG:OBST_PROFILE]]   (OUTSUBDIR relative to OUTDIR, default '.'; no OBST = no obstacle)
# - resumable: runs with OUTDIR/NAME.done are skipped; if every run is done no server is started.
# - preflight (logged to OUTDIR/chain_PORT.log, files OUTDIR/preflight_PORT_TS.*): nvidia-smi + free, GPU budget
#   (GPU memory.used now + EST_GB <= GPU_BUDGET_GB (130), re-checked every 60 s up to GPU_WAIT_MAX_S, then exit 3;
#   launchers stagger chains by 240 s so an earlier server has done its first-request JIT and grown before the next check), port free, code md5 == p/code_md5_j0.txt (else exit 4 unless
#   ALLOW_MD5_DIFF=1), import check server + client from ${AG3S_PROJECT_ROOT}-t40h (CPU only, else exit 5).
# - server log must say "XLA autotuning off" (else kill, exit 6).
# - a failed run is retried once if the server is still alive; if the server died the chain stops (exit 7).
# - T40FG.verify.partial.json is rebuilt (update_partial.py, flock) after every run.
source ${AG3S_PROJECT_ROOT}/outputs/verify/T40/J/p/env.sh
C=$1; PORT=$2; OUT=$3; LIST=$4; EST_GB=${EST_GB:-30}
port_ok $PORT || { echo "port $PORT not allowed (8240-8249 only, never 8123)"; exit 2; }
mkdir -p $OUT; CL=$OUT/chain_$PORT.log; TS=$(date +%Y%m%dT%H%M%S); PF=$OUT/preflight_${PORT}_$TS
log() { echo "$* $(date -Iseconds)" >> $CL; }
upd() { $PY ${AG3S_PROJECT_ROOT}/outputs/verify/T40/J/p/update_j.py >> $OUT/update_partial.log 2>&1; }
log "chain start cond=$C port=$PORT list=[$LIST] est_gb=$EST_GB budget_gb=$GPU_BUDGET_GB"
item() { IFS=: read -r EP SEED SUB OCFG OPROF <<< "$1"; SUB=${SUB:-.}; RO=$OUT/$SUB; NAME=${C}_ep${EP}_s${SEED}; }
todo=0; for RK in $LIST; do item $RK; [ -f $RO/$NAME.done ] || todo=$((todo+1)); done
if [ $todo = 0 ]; then log "all runs done, no server"; upd; echo DONE >> $CL; exit 0; fi
# --- preflight
{ nvidia-smi; echo; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; echo; free -m; } > $PF.resources.txt 2>&1
tw=$(date +%s)
while :; do
  USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  NEED=$(( (USED_MIB + 1023) / 1024 + EST_GB ))
  log "gpu used_mib=$USED_MIB used+est_gb=$NEED budget_gb=$GPU_BUDGET_GB"
  [ $NEED -le $GPU_BUDGET_GB ] && break
  if [ $(( $(date +%s) - tw )) -ge $GPU_WAIT_MAX_S ]; then log "REFUSED gpu budget (waited $(( $(date +%s) - tw )) s)"; exit 3; fi
  sleep 60
done
{ nvidia-smi; echo; free -m; } > $PF.resources_at_start.txt 2>&1
echo $USED_MIB > $PF.gpu_used_before_mib
if ! python3 -c "import socket; s=socket.socket(); s.bind(('0.0.0.0', $PORT)); s.close()" 2>/dev/null; then log "REFUSED port $PORT in use"; exit 3; fi
$P/code_md5.sh > $PF.code_md5.txt 2>&1
if ! cmp -s $PF.code_md5.txt $P/code_md5_j0.txt; then
  diff $P/code_md5_j0.txt $PF.code_md5.txt > $PF.code_md5.diff
  log "code md5 DIFFERS from J-0 ($(grep -c '^[<>]' $PF.code_md5.diff) lines, $PF.code_md5.diff)"
  [ "$ALLOW_MD5_DIFF" = 1 ] || { log "REFUSED code md5"; exit 4; }
else log "code md5 == J-0"; fi
( cd $O1 && JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=$O1 OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi \
  timeout 600 $PY -c "import sys; sys.argv=['-c','server']; exec(open('$P/import_check.py').read())" ) > $PF.import_server.json 2> $PF.import_server.err
( cd $O1/pi05_TO_hybrid && JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=$O1 \
  timeout 600 $PY $P/import_check.py client $O1/integration/rby1/bringup/pi05_infer.py ) > $PF.import_client.json 2> $PF.import_client.err
for s in server client; do
  grep -q '"all_benchmark_under_snapshot": true' $PF.import_$s.json || { log "REFUSED import check $s ($PF.import_$s.json)"; exit 5; }
done
log "import check server+client: all under ${AG3S_PROJECT_ROOT}-t40h"
# --- server
RECDIR=$OUT/server_${PORT}_$TS; mkdir -p $RECDIR; SL=$RECDIR/serve_safe.log
date -Iseconds > $RECDIR/server.start
echo $USED_MIB > $RECDIR/gpu_used_before_mib
nohup $P/start_server.sh $C $PORT $RECDIR > $SL 2>&1 &
SPID=$!; echo $SPID > $RECDIR/server.pid
# GPU total used every 2 s while the server lives (peak delta vs gpu_used_before_mib = this server's footprint
# when no other process changes its usage; other sessions' pids are not attributable)
( while kill -0 $SPID 2>/dev/null; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"; sleep 2; done ) > $RECDIR/gpu_samples.txt 2>/dev/null &
t0=$(date +%s)
until grep -q "server listening on 0.0.0.0:$PORT" $SL; do
  if ! kill -0 $SPID 2>/dev/null; then log "server died during start ($SL)"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt $SERVER_UP_TIMEOUT_S ]; then log "server not up after ${SERVER_UP_TIMEOUT_S}s"; kill $SPID; exit 1; fi
  sleep 5; done
log "server up pid=$SPID rec=$RECDIR startup_s=$(( $(date +%s) - t0 ))"
# RSS (lead rule 2026-10-03): every 5 s: epoch, server RSS kB, RSS kB of the client(s) talking to this port, pod memory.current
( while kill -0 $SPID 2>/dev/null; do cpid=$(pgrep -f "pi05_infer.py .*--remote localhost:$PORT" | tr '\n' ' ');
    crss=0; for q in $cpid; do r=$(ps -o rss= -p $q 2>/dev/null | tr -d ' '); crss=$(( crss + ${r:-0} )); done
    echo "$(date +%s) $(ps -o rss= -p $SPID | tr -d ' ') $crss $(cat /sys/fs/cgroup/memory.current)"; sleep 5; done ) > $RECDIR/rss_samples.txt 2>/dev/null &
{ echo "cwd=$(readlink /proc/$SPID/cwd)"; tr '\0' '\n' < /proc/$SPID/environ | grep -E '^(XLA_FLAGS|PYTHONPATH|XLA_PYTHON_CLIENT_PREALLOCATE|OPENPI_DATA_HOME|PYTHONUNBUFFERED)='; tr '\0' ' ' < /proc/$SPID/cmdline; echo; } > $RECDIR/server_proc.txt
if ! grep -q "XLA autotuning off" $SL; then log "REFUSED server log lacks 'XLA autotuning off'"; kill $SPID; sleep 5; pkill -f "serve_safe.*--port $PORT"; exit 6; fi
# nvidia-smi pids are not this container's pids ([Not Found]) -> log the GPU total and the per-app list as given
gpumem() { echo "$1 $(date -Iseconds) total_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ') apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >> $RECDIR/gpu_mem.txt; }
gpumem up
# --- runs
for RK in $LIST; do item $RK
  if [ -f $RO/$NAME.done ]; then log "skip $SUB/$NAME (done)"; continue; fi
  for attempt in 1 2; do
    wset() { echo $(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) )); }
    tm=$(date +%s); while [ $(( $(wset) + 4 * 1024 * 1024 * 1024 )) -gt $(( 64 * 1024 * 1024 * 1024 )) ]; do
      log "mem wait before client working_set=$(wset)"; [ $(( $(date +%s) - tm )) -ge 3600 ] && { log "REFUSED memory"; kill $SPID; exit 8; }; sleep 60; done
    log "start $SUB/$NAME attempt=$attempt obstacle=$OCFG:$OPROF working_set=$(wset)"
    $P/run_client.sh $C $PORT $RO $EP $SEED $OCFG $OPROF > /dev/null 2>&1
    log "end $SUB/$NAME attempt=$attempt $(tail -1 $RO/$NAME.client.log) done=$([ -f $RO/$NAME.done ] && echo 1 || echo 0)"
    gpumem "after_${SUB}_$NAME"
    upd
    [ -f $RO/$NAME.done ] && break
    if ! kill -0 $SPID 2>/dev/null; then log "server died, chain stops"; exit 7; fi
  done
done
kill $SPID; sleep 5; pkill -f "serve_safe.*--port $PORT"; sleep 3
log "server killed alive=$(pgrep -f "serve_safe.*--port $PORT" | wc -l)"
date -Iseconds > $RECDIR/server.end
upd
echo DONE >> $CL
