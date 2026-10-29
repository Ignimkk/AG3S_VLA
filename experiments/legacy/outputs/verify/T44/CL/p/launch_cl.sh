#!/bin/bash
# T44d CL launcher. usage: nohup setsid launch_cl.sh BLOCK   (BLOCK = 1 2 3)  -- 2 rounds per block, 2 servers per round (ports 8280 / 8281), same pair list in all 4 arms.
source ${AG3S_PROJECT_ROOT}/outputs/verify/T44/CL/p/env.sh
B=$1; R=$V/runs
case $B in
  1) LIST="12:121:R 12:122:R 30:301:R 30:302:R 0:1:R 0:2:R"; RA="C0 C1b"; RB="C1a C2";;
  2) LIST="3:31:R 3:32:R 11:111:R 11:112:R 54:541:R 54:542:R"; RA="C1a C2"; RB="C0 C1b";;
  3) LIST="123:1231:R 123:1232:R 104:1041:R 104:1042:R 110:1101:R 110:1102:R"; RA="C0 C1b"; RB="C1a C2";;
  *) echo bad block; exit 2;; esac
echo "launch block $B $(date -Iseconds) loadavg=$(cat /proc/loadavg)"
for ROUND in "$RA" "$RB"; do
  set -- $ROUND; A1=$1; A2=$2
  mkdir -p $R/$A1 $R/$A2
  ( EST_GB=28 $P/run_chain.sh $A1 8280 $R/$A1 "$LIST"; echo "$A1 block$B rc=$? $(date -Iseconds)" ) > $R/chain_${A1}_b$B.out 2>&1 &
  ( sleep 20; EST_GB=28 $P/run_chain.sh $A2 8281 $R/$A2 "$LIST"; echo "$A2 block$B rc=$? $(date -Iseconds)" ) > $R/chain_${A2}_b$B.out 2>&1 &
  wait
  echo "round $ROUND end $(date -Iseconds) loadavg=$(cat /proc/loadavg)"
done
echo "all end block $B $(date -Iseconds)"
