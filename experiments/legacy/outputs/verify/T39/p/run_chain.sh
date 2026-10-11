#!/bin/bash
# T39 S2 chain (from T37/p/run_chain_e3.sh): preflight -> server on PORT -> runs in order -> kill server.
# usage: EST_GB=<our per-server estimate> run_chain.sh COND PORT OUTDIR "EP:SEED EP:SEED ..."
# - resumable: runs with OUTDIR/NAME.done are skipped; if every run is done no server is started.
# - preflight (logged to OUTDIR/chain_PORT.log, files OUTDIR/preflight_PORT_TS.*): nvidia-smi + free, GPU budget
#   (GPU memory.used now + EST_GB <= GPU_BUDGET_GB (130), re-checked every 60 s up to GPU_WAIT_MAX_S, then exit 3;
#   launchers stagger chains by 240 s so an earlier server has done its first-request JIT and grown before the next check), port free, code md5 == p/code_md5_s20.txt (else exit 4 unless
#   ALLOW_MD5_DIFF=1), import check server + client from ${AG3S_PROJECT_ROOT}-o1 (CPU only, else exit 5).
# - server log must say "XLA autotuning off" (else kill, exit 6).
# - a failed run is retried once if the server is still alive; if the server died the chain stops (exit 7).
# - T39.verify.partial.json is rebuilt (update_partial.py, flock) after every run.
source ${AG3S_PROJECT_ROOT}/outputs/verify/T39/p/env.sh
C=$1; PORT=$2; OUT=$3; LIST=$4; EST_GB=${EST_GB:-30}
port_ok $PORT || { echo "port $PORT not allowed (8220-8229 only, never 8123)"; exit 2; }
mkdir -p $OUT; CL=$OUT/chain_$PORT.log; TS=$(date +%Y%m%dT%H%M%S); PF=$OUT/preflight_${PORT}_$TS
log() { echo "$* $(date -Iseconds)" >> $CL; }
upd() { $PY $P/update_partial.py >> $OUT/update_partial.log 2>&1; }
log "chain start cond=$C port=$PORT list=[$LIST] est_gb=$EST_GB budget_gb=$GPU_BUDGET_GB"
todo=0; for RK in $LIST; do [ -f $OUT/${C}_ep${RK%%:*}_s${RK##*:}.done ] || todo=$((todo+1)); done
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
if ! cmp -s $PF.code_md5.txt $P/code_md5_s20.txt; then
  diff $P/code_md5_s20.txt $PF.code_md5.txt > $PF.code_md5.diff
  log "code md5 DIFFERS from S2-0 ($(grep -c '^[<>]' $PF.code_md5.diff) lines, $PF.code_md5.diff)"
  [ "$ALLOW_MD5_DIFF" = 1 ] || { log "REFUSED code md5"; exit 4; }
else log "code md5 == S2-0"; fi
( cd $O1 && JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=$O1 OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi \
  timeout 600 $PY -c "import sys; sys.argv=['-c','server']; exec(open('$T39/s20/import_check.py').read())" ) > $PF.import_server.json 2> $PF.import_server.err
( cd $O1/pi05_TO_hybrid && JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=$O1 \
  timeout 600 $PY $T39/s20/import_check.py client $O1/integration/rby1/bringup/pi05_infer.py ) > $PF.import_client.json 2> $PF.import_client.err
for s in server client; do
  grep -q '"all_benchmark_under_work_o1": true' $PF.import_$s.json || { log "REFUSED import check $s ($PF.import_$s.json)"; exit 5; }
done
log "import check server+client: all under ${AG3S_PROJECT_ROOT}-o1"
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
{ echo "cwd=$(readlink /proc/$SPID/cwd)"; tr '\0' '\n' < /proc/$SPID/environ | grep -E '^(XLA_FLAGS|PYTHONPATH|XLA_PYTHON_CLIENT_PREALLOCATE|OPENPI_DATA_HOME|PYTHONUNBUFFERED)='; tr '\0' ' ' < /proc/$SPID/cmdline; echo; } > $RECDIR/server_proc.txt
if ! grep -q "XLA autotuning off" $SL; then log "REFUSED server log lacks 'XLA autotuning off'"; kill $SPID; sleep 5; pkill -f "serve_safe.*--port $PORT"; exit 6; fi
# nvidia-smi pids are not this container's pids ([Not Found]) -> log the GPU total and the per-app list as given
gpumem() { echo "$1 $(date -Iseconds) total_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ') apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >> $RECDIR/gpu_mem.txt; }
gpumem up
# --- runs
for RK in $LIST; do EP=${RK%%:*}; SEED=${RK##*:}; NAME=${C}_ep${EP}_s${SEED}
  if [ -f $OUT/$NAME.done ]; then log "skip $NAME (done)"; continue; fi
  for attempt in 1 2; do
    log "start $NAME attempt=$attempt"
    $P/run_client.sh $C $PORT $OUT $EP $SEED > /dev/null 2>&1
    log "end $NAME attempt=$attempt $(tail -1 $OUT/$NAME.client.log) done=$([ -f $OUT/$NAME.done ] && echo 1 || echo 0)"
    gpumem "after_$NAME"
    upd
    [ -f $OUT/$NAME.done ] && break
    if ! kill -0 $SPID 2>/dev/null; then log "server died, chain stops"; exit 7; fi
  done
done
kill $SPID; sleep 5; pkill -f "serve_safe.*--port $PORT"; sleep 3
log "server killed alive=$(pgrep -f "serve_safe.*--port $PORT" | wc -l)"
date -Iseconds > $RECDIR/server.end
upd
echo DONE >> $CL
