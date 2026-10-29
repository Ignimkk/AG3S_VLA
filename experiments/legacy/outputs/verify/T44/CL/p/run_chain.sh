#!/bin/bash
# T43V11 chain (copy of T43/V8/p/run_chain.sh; ARM = V11A | V11AB | V11ABC, per-ARM TA server-log guards; 120 s settle after our own server start): preflight -> server on PORT -> runs in order -> kill server.
# usage: EST_GB=<our per-server estimate> run_chain.sh ARM PORT OUTDIR "ITEM ITEM ..."
#   ITEM = EP:SEED[:OUTSUBDIR[:OBST_CONFIG:OBST_PROFILE]]   (OUTSUBDIR relative to OUTDIR, default '.'; no OBST = no obstacle)
# - resumable: runs with OUTDIR/NAME.done are skipped; if every run is done no server is started.
# - preflight (logged to OUTDIR/chain_PORT.log, files OUTDIR/preflight_PORT_TS.*): nvidia-smi + free, GPU budget
#   (GPU memory.used now + EST_GB <= GPU_BUDGET_GB (130), re-checked every 60 s up to GPU_WAIT_MAX_S, then exit 3;
#   launchers stagger chains by 240 s so an earlier server has done its first-request JIT and grown before the next check), port free, code md5 == p/code_md5_cl.txt (else exit 4 unless
#   ALLOW_MD5_DIFF=1), import check server + client from ${AG3S_PROJECT_ROOT}-t44cl (CPU only, else exit 5).
# - server log must say "XLA autotuning off" (else kill, exit 6).
# - a failed run is retried once if the server is still alive; if the server died the chain stops (exit 7).
# - T43W.verify.partial.json is rebuilt (update_partial.py, flock) after every run.
source ${AG3S_PROJECT_ROOT}/outputs/verify/T44/CL/p/env.sh
C=$1; PORT=$2; OUT=$3; LIST=$4; EST_GB=${EST_GB:-30}
is_arm $C || { echo "unknown ARM $C"; exit 2; }
port_ok $PORT || { echo "port $PORT not allowed (8280-8289 only, never 8123)"; exit 2; }
mkdir -p $OUT; CL=$OUT/chain_$PORT.log; TS=$(date +%Y%m%dT%H%M%S); PF=$OUT/preflight_${PORT}_$TS
log() { echo "$* $(date -Iseconds)" >> $CL; }
upd() { $PY $P/update_partial.py >> $OUT/update_partial.log 2>&1; }
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
if ! cmp -s $PF.code_md5.txt $P/code_md5_cl.txt; then
  diff $P/code_md5_cl.txt $PF.code_md5.txt > $PF.code_md5.diff
  log "code md5 DIFFERS from CL-0 ($(grep -c '^[<>]' $PF.code_md5.diff) lines, $PF.code_md5.diff)"
  [ "$ALLOW_MD5_DIFF" = 1 ] || { log "REFUSED code md5"; exit 4; }
else log "code md5 == CL-0"; fi
( cd $O1 && JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=$O1 OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi \
  timeout 600 $PY -c "import sys; sys.argv=['-c','server']; exec(open('$P/import_check.py').read())" ) > $PF.import_server.json 2> $PF.import_server.err
( cd $O1/pi05_TO_hybrid && JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=$O1 \
  timeout 600 $PY $P/import_check.py client $O1/integration/rby1/bringup/pi05_infer.py ) > $PF.import_client.json 2> $PF.import_client.err
for s in server client; do
  grep -q '"all_benchmark_under_snapshot": true' $PF.import_$s.json || { log "REFUSED import check $s ($PF.import_$s.json)"; exit 5; }
done
log "import check server+client: all under ${AG3S_PROJECT_ROOT}-t44cl"
# --- start guard (T43W): under an exclusive lock shared by our chains, re-check right before the start:
#     GPU used + EST_GB <= 130 GB, working set + EST_RSS_SERVER (8 GiB) <= 64 GiB, youngest serve_safe (anyone's) >= 120 s.
#     The lock is held until our server process exists, so our other chain then sees it as the youngest server.
EST_RSS_SERVER=${EST_RSS_SERVER:-8589934592}
exec 8>$P/.server_start.lock; flock 8
tw=$(date +%s)
while :; do
  USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  NEED=$(( (USED_MIB + 1023) / 1024 + EST_GB )); cur=$(wset); rs=$(recent_server_s)
  if [ $NEED -le $GPU_BUDGET_GB ] && [ $(( cur + EST_RSS_SERVER )) -le $(( 64 * 1024 * 1024 * 1024 )) ] && { [ -z "$rs" ] || [ "$rs" -ge 120 ]; }; then
    log "start guard ok gpu_used_mib=$USED_MIB used+est_gb=$NEED working_set=$cur youngest_server_s=${rs:-none}"; break; fi
  log "start guard wait gpu_used_mib=$USED_MIB used+est_gb=$NEED working_set=$cur youngest_server_s=${rs:-none}"
  if [ $(( $(date +%s) - tw )) -ge $GPU_WAIT_MAX_S ]; then log "REFUSED start guard (waited $(( $(date +%s) - tw )) s)"; exit 3; fi
  sleep 30
done
# --- server
RECDIR=$OUT/server_${PORT}_$TS; mkdir -p $RECDIR; SL=$RECDIR/serve_safe.log
date -Iseconds > $RECDIR/server.start
echo $USED_MIB > $RECDIR/gpu_used_before_mib
nohup $P/start_server.sh $C $PORT $RECDIR > $SL 2>&1 8>&- &
SPID=$!; echo $SPID > $RECDIR/server.pid
for i in $(seq 1 30); do pgrep -f "serve_safe.*--port $PORT" >/dev/null && break; sleep 1; done; flock -u 8; exec 8>&-
# GPU total used every 2 s while the server lives (peak delta vs gpu_used_before_mib = this server's footprint
# when no other process changes its usage; other sessions' pids are not attributable)
( while kill -0 $SPID 2>/dev/null; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"; sleep 2; done ) > $RECDIR/gpu_samples.txt 2>/dev/null &
t0=$(date +%s)
until grep -q "server listening on 0.0.0.0:$PORT" $SL; do
  if ! kill -0 $SPID 2>/dev/null; then log "server died during start ($SL)"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt $SERVER_UP_TIMEOUT_S ]; then log "server not up after ${SERVER_UP_TIMEOUT_S}s"; kill $SPID; exit 1; fi
  sleep 5; done
log "server up arm=$C pid=$SPID rec=$RECDIR startup_s=$(( $(date +%s) - t0 ))"
# RSS (lead rule 2026-10-03): every 5 s: epoch, server RSS kB, RSS kB of the client(s) talking to this port, pod memory.current
( while kill -0 $SPID 2>/dev/null; do cpid=$(pgrep -f "pi05_infer.py .*--remote localhost:$PORT" | tr '\n' ' ');
    crss=0; for q in $cpid; do r=$(ps -o rss= -p $q 2>/dev/null | tr -d ' '); crss=$(( crss + ${r:-0} )); done
    echo "$(date +%s) $(ps -o rss= -p $SPID | tr -d ' ') $crss $(cat /sys/fs/cgroup/memory.current)"; sleep 5; done ) > $RECDIR/rss_samples.txt 2>/dev/null &
{ echo "cwd=$(readlink /proc/$SPID/cwd)"; tr '\0' '\n' < /proc/$SPID/environ | grep -E '^(XLA_FLAGS|PYTHONPATH|XLA_PYTHON_CLIENT_PREALLOCATE|OPENPI_DATA_HOME|PYTHONUNBUFFERED)='; tr '\0' ' ' < /proc/$SPID/cmdline; echo; } > $RECDIR/server_proc.txt
if ! grep -q "XLA autotuning off" $SL; then log "REFUSED server log lacks 'XLA autotuning off'"; kill $SPID; sleep 5; pkill -f "serve_safe.*--port $PORT"; exit 6; fi
refuse() { log "REFUSED server log: $1"; kill $SPID; sleep 5; pkill -f "serve_safe.*--port $PORT"; exit 9; }
grep -q "TO obstacle_margin: 10.0 mm" $SL || refuse "lacks 'TO obstacle_margin: 10.0 mm'"
grep -qF "gripper cover (T43 R)" $SL && refuse "gripper cover line present"
grep -qF "body cover (T43 R2)" $SL || refuse "$C lacks 'body cover (T43 R2)'"
CMD=$(cat $RECDIR/server_proc.txt)
for f in --closing-keep-geometry --latched-identity-hold "--supported-max-bottom-mm 60" --held-slip-detach "--held-cover-cap-mm 10" "--hold-follow-to 3" "--finger-cover both" \
         "--plan-horizon 16" "--lookahead obstacles" "--lookahead-gate-mm 20" "--qp-eps 1e-4" "--w-deflection-rate 20" "--to-fast-eval" "--esdf-host-mirror"; do
  echo "$CMD" | grep -qF -- "$f" || refuse "$C cmdline lacks $f"; done
for f in --refused-identity-release --held-free-observed-mm --gripper-cover --rest-start --servo-model --servo-accel --held-obstacle-margin --target-volume-exempt --sweep-check --no-subtask-gate; do echo "$CMD" | grep -qF -- "$f" && refuse "$C cmdline has $f"; done
grep -qF "held_slip_detach=True" $SL || refuse "$C server log lacks held_slip_detach=True"
grep -qE "subtask label: .* gate ON" $SL || refuse "$C server log lacks 'subtask label: ... gate ON'"
grep -qF "held_cover_cap=0.01" $SL || refuse "$C server log lacks held_cover_cap=0.01"
grep -qF "hold_follow_to=3" $SL || refuse "$C server log lacks hold_follow_to=3"
grep -qF "TO path check (T43 Q)" $SL && refuse "$C has a T43 Q path-check line"
grep -qF "TO lookahead (T43 TA A): obstacles — plan 16 steps, certify the first 8" $SL || refuse "$C lacks TA A line"
grep -qF "gate 20 mm" $SL || refuse "$C lacks 'gate 20 mm'"
grep -qE "TO QP accuracy \(T43 TA\): OSQP eps_abs=0.0001 eps_rel=0.0001" $SL || refuse "$C lacks QP accuracy line"
grep -qF "TO deflection rate (T43 TA B): w_deflection_rate=20" $SL || refuse "$C lacks B line"
grep -qF "TO servo accel (T43 TA C)" $SL && refuse "$C has C line"
grep -q "TO rest_start" $SL && refuse "rest-start line present"
grep -qF "TO compute options (T43 TP): esdf_host_mirror, fast_eval (bit-identical)" $SL || refuse "$C lacks TP compute options line"
grep -qF "constraint finger cover (T43 FC)" $SL || refuse "$C lacks finger cover line"
case $C in
  C2) echo "$CMD" | grep -qF -- "--links arms" || refuse "C2 cmdline lacks --links arms"
      grep -qF "CONSTRAINT SCOPE IS ARMS" $SL || refuse "C2 log lacks 'CONSTRAINT SCOPE IS ARMS'"
      grep -qE "constraint finger cover \(T43 FC\): 구 274 → 422 " $SL || refuse "C2 log lacks finger cover 274 -> 422";;
  *)  echo "$CMD" | grep -qF -- "--links gripper" || refuse "$C cmdline lacks --links gripper"
      grep -qF "CONSTRAINT SCOPE IS ARMS" $SL && refuse "$C log has ARMS scope"
      grep -qE "constraint finger cover \(T43 FC\): 구 122 → 270 " $SL || refuse "$C log lacks finger cover 122 -> 270";;
esac
case $C in
  C0) for f in --target-exclusion --owned-max-dist-voxels --attention-split-radius-mm --attention-split-ratio-source --target-permission-strict --target-free-coarse; do echo "$CMD" | grep -qF -- "$f" && refuse "C0 cmdline has $f"; done
      grep -qF "AG3S target_exclusion = ball_owned" $SL && refuse "C0 log has ball_owned"
      grep -qF "target_permission_strict: ON" $SL && refuse "C0 log has permission strict"
      grep -qF "attention_split_ratio_source=raw" $SL && refuse "C0 log has split ratio_source=raw";;
  C1a|C1b|C2)
      for f in "--target-exclusion ball-owned" "--owned-max-dist-voxels 3" "--attention-split-radius-mm 12" "--attention-split-ratio-source raw" "--target-permission-strict"; do echo "$CMD" | grep -qF -- "$f" || refuse "$C cmdline lacks $f"; done
      grep -qF "AG3S target_exclusion = ball_owned (tie 1.00 · max_dist 3.00 voxels · unowned remove)" $SL || refuse "$C log lacks ball_owned max_dist 3.00"
      grep -qF "attention_split_radius=0.012, attention_split_ratio_source=raw" $SL || refuse "$C log lacks split 12 mm / raw"
      grep -qF "TO target_permission_strict: ON" $SL || refuse "$C log lacks target_permission_strict ON"
      echo "$CMD" | grep -qF -- "--target-free-coarse-sign-floor" && refuse "$C cmdline sets sign floor explicitly (default plane expected)";;
esac
case $C in
  C1b|C2) echo "$CMD" | grep -qF -- "--target-free-coarse" || refuse "$C cmdline lacks --target-free-coarse";;
  C0|C1a) echo "$CMD" | grep -qF -- "--target-free-coarse" && refuse "$C cmdline has --target-free-coarse";;
esac
log "server log: ta=[$(grep -E 'TO (lookahead|QP accuracy|deflection rate|compute options) \(T43 T' $SL | cut -c1-120 | tr '\n' '|')] excl=[$(grep -m1 'AG3S target_exclusion' $SL | cut -c1-140)] split=[$(grep -m1 'attention_split_radius' $SL | cut -c1-140)] strict=[$(grep -m1 'target_permission_strict' $SL | cut -c1-80)] scope=[$(grep -c 'CONSTRAINT SCOPE IS ARMS' $SL)] fc=[$(grep -m1 'constraint finger cover (T43 FC)' $SL | cut -c1-120)]"
# 120 s after our own server start before the first client (lead rule: wait 120 s after every server start)
while [ $(( $(date +%s) - $(date -d "$(cat $RECDIR/server.start)" +%s) )) -lt 120 ]; do sleep 10; done
# nvidia-smi pids are not this container's pids ([Not Found]) -> log the GPU total and the per-app list as given
gpumem() { echo "$1 $(date -Iseconds) total_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ') apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >> $RECDIR/gpu_mem.txt; }
gpumem up
# --- runs
for RK in $LIST; do item $RK
  if [ -f $RO/$NAME.done ]; then log "skip $SUB/$NAME (done)"; continue; fi
  for attempt in 1 2; do
    tm=$(date +%s); while [ $(( $(wset) + 4 * 1024 * 1024 * 1024 )) -gt $(( 64 * 1024 * 1024 * 1024 )) ]; do
      log "mem wait before client working_set=$(wset) current=$(cat /sys/fs/cgroup/memory.current)"; [ $(( $(date +%s) - tm )) -ge 3600 ] && { log "REFUSED memory"; kill $SPID; exit 8; }; sleep 60; done
    log "start $SUB/$NAME attempt=$attempt obstacle=$OCFG:$OPROF working_set=$(wset) mem_current=$(cat /sys/fs/cgroup/memory.current)"
    $P/run_client.sh $C $PORT $RO $EP $SEED $OCFG $OPROF > /dev/null 2>&1
    log "end $SUB/$NAME attempt=$attempt $(tail -1 $RO/$NAME.client.log) done=$([ -f $RO/$NAME.done ] && echo 1 || echo 0)"
    gpumem "after_${SUB}_$NAME"
    upd
    # T44d: coarse tier announcement appears at the first frame (not at startup): C1b / C2 must show it with sign_floor plane; C0 / C1a must not
    if [ -z "$COARSE_CHECKED" ] && [ -f $RO/$NAME.client.log ] && grep -q "exit=" $RO/$NAME.client.log; then
      case $C in
        C1b|C2) grep -qF "T44c TARGET-FREE COARSE TIER IS ON" $SL && grep "T44c TARGET-FREE COARSE TIER IS ON" $SL | grep -qF "sign_floor plane" && { COARSE_CHECKED=1; log "coarse check ok: $(grep -m1 'T44c TARGET-FREE COARSE' $SL | cut -c1-260)"; } || { log "REFUSED coarse tier line missing / not plane after first run"; kill $SPID; sleep 5; pkill -f "serve_safe.*--port $PORT"; exit 11; };;
        *) grep -qF "T44c TARGET-FREE COARSE" $SL && { log "REFUSED $C log has coarse tier"; kill $SPID; sleep 5; pkill -f "serve_safe.*--port $PORT"; exit 11; } || { COARSE_CHECKED=1; log "coarse check ok: no coarse line for $C"; };;
      esac
    fi
    # T43W (after the T2 collision_backend incident): the newest server chunk must carry an ESDF (esdf_stats), else stop the chain
    if ! $PY -c "import glob,json,sys,numpy as np; f=sorted(glob.glob('$RECDIR/server_constraints/run_*/chunk_*.npz'))[-1]; s=json.loads(str(np.load(f,allow_pickle=True)['summary_json'])); sys.exit(0 if s.get('esdf_stats') else 1)"; then
      log "REFUSED newest server chunk has no esdf_stats (AG3S without ESDF)"; kill $SPID; sleep 5; pkill -f "serve_safe.*--port $PORT"; exit 10; fi
    [ -f $RO/$NAME.done ] && break
    if ! kill -0 $SPID 2>/dev/null; then log "server died, chain stops"; exit 7; fi
  done
done
kill $SPID; sleep 5; pkill -f "serve_safe.*--port $PORT"; sleep 3
log "server killed alive=$(pgrep -f "serve_safe.*--port $PORT" | wc -l)"
date -Iseconds > $RECDIR/server.end
upd
echo DONE >> $CL
