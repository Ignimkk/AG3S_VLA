#!/bin/bash
# T42-1 launcher (verifier, 2026-10-05). Phases in order; each phase's lists come from plan.py (deterministic from
# candidates.json + finished runs), so re-running this script resumes (runs with NAME.done are skipped; a phase whose
# marker R/phase_<P>.done exists is skipped).
#   p1  E3b screening      2 chains (8250, 8251), staggered 240 s
#   p2a E3b rep 2          2 chains
#   p2b E3b rep 3          2 chains
#   p3  E0 on selected     1 chain  (8252)   (screening done -> 1 server, T41 needs 2 later)
# Memory guard (lead rule, revised 2026-10-05): before each chain working set (memory.current - inactive_file) + EST <= 64 GiB,
# no check within 120 s of any serve_safe start; re-check every 60 s up to 2 h; a second chain
# that never fits runs its list after the first chain, on the first chain's port. GPU guard is in run_chain.sh (used + EST_GB <= 130).
# EST = measured T40 G peaks (server + client RSS: E3b 6.9 + 3.0 GiB, E0 4.0 + 2.8 GiB) + about 1 GiB.
# usage: nohup launch_t42.sh [FIRST_PHASE] > ${AG3S_PROJECT_ROOT}/outputs/verify/T42/1/runs/launch.out 2>&1 &
source ${AG3S_PROJECT_ROOT}/outputs/verify/T42/p/env.sh
R=$T42/1/runs; mkdir -p $R
EST_RSS_E3B=$(( 11 * 1024 * 1024 * 1024 )); EST_RSS_E0=$(( 8 * 1024 * 1024 * 1024 ))
memok() { local need=$1 t0=$(date +%s); while :; do servers_settled; cur=$(mem_ws); \
  [ $(( cur + need )) -le $MEM_LIMIT_BYTES ] && { echo "mem ok working_set=$cur need=$need $(date -Iseconds)"; return 0; }; \
  echo "mem wait working_set=$cur need=$need $(date -Iseconds)"; [ $(( $(date +%s) - t0 )) -ge 7200 ] && return 1; sleep 60; done; }
upd() { $PY $P/update_partial.py >> $R/update_partial.log 2>&1; }

phase2() {  # COND EST_GB EST_RSS PHASE   two chains
  local C=$1 EG=$2 ER=$3 PH=$4
  ( memok $ER && EST_GB=$EG $P/run_chain.sh $C 8250 $R/$C "$(cat $P/lists/${PH}_0.txt)"; echo "$PH chain0 rc=$? $(date -Iseconds)" ) & local p1=$!
  sleep 240
  ( if memok $ER; then EST_GB=$EG $P/run_chain.sh $C 8251 $R/$C "$(cat $P/lists/${PH}_1.txt)"; echo "$PH chain1 rc=$? $(date -Iseconds)";
    else echo "$PH chain1 memory never fit; runs after chain0 on 8250 $(date -Iseconds)"; while kill -0 $p1 2>/dev/null; do sleep 30; done;
         memok $ER && EST_GB=$EG $P/run_chain.sh $C 8250 $R/$C "$(cat $P/lists/${PH}_1.txt)"; echo "$PH chain1(8250) rc=$? $(date -Iseconds)"; fi ) & local p2=$!
  wait $p1; wait $p2
}

phase_ok() {  # every item of the phase lists done
  local C=$1 PH=$2
  for f in $P/lists/${PH}_*.txt; do for it in $(cat $f); do IFS=: read -r EP SEED SUB <<< "$it"; [ -f $R/$C/$SUB/${C}_ep${EP}_s${SEED}.done ] || return 1; done; done
  return 0
}

FIRST=${1:-p1}; started=0
echo "t42-1 launch start $(date -Iseconds) first=$FIRST"
for PH in p1 p2a p2b p3; do
  [ $PH = $FIRST ] && started=1; [ $started = 1 ] || continue
  [ -f $R/phase_$PH.done ] && { echo "$PH already done"; continue; }
  if [ $PH = p3 ]; then C=E0; N=1; else C=E3b; N=2; fi
  rm -f $P/lists/${PH}_*.txt
  $PY $P/plan.py $PH $N > $R/plan_$PH.out 2>&1 || { echo "plan $PH failed"; exit 1; }
  echo "$PH plan: $(tr '\n' '|' < $R/plan_$PH.out) $(date -Iseconds)"
  if [ $PH = p3 ]; then
    memok $EST_RSS_E0 && EST_GB=10 $P/run_chain.sh E0 8252 $R/E0 "$(cat $P/lists/p3_0.txt)"; echo "p3 chain0 rc=$? $(date -Iseconds)"
  else
    phase2 E3b 22 $EST_RSS_E3B $PH
  fi
  upd
  if phase_ok $C $PH; then touch $R/phase_$PH.done; echo "$PH done $(date -Iseconds)"; else echo "$PH INCOMPLETE, stop $(date -Iseconds)"; exit 1; fi
done
echo "t42-1 launch end $(date -Iseconds)"
