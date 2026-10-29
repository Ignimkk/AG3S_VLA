#!/bin/bash
# T39 S2-1 acceptance: ep1807 seed 18071 on two different server processes per condition, ONE server at a time
# (GPU rule 2026-10-02: ~24 GB free next to another session's 105.5 GB pool). Order: E0_a 8220, E0_b 8221, E3b_a 8222, E3b_b 8223.
# Each server's GPU footprint = max(gpu_samples.txt) - gpu_used_before_mib. Then compare_pair.py per condition.
# usage: nohup launch_s21.sh > ${AG3S_PROJECT_ROOT}/outputs/verify/T39/s21/launch.out 2>&1 &
source ${AG3S_PROJECT_ROOT}/outputs/verify/T39/p/env.sh
S=$T39/s21; mkdir -p $S
echo "s21 start $(date -Iseconds)"
for x in E0:8220:a E0:8221:b E3b:8222:a E3b:8223:b; do C=${x%%:*}; r=${x#*:}; PORT=${r%%:*}; G=${r##*:}
  EST_GB=${EST_GB_S21:-20} $P/run_chain.sh $C $PORT $S/${C}_$G "1807:18071"; echo "${C}_$G rc=$? $(date -Iseconds)"
done
for C in E0 E3b; do
  if [ -f $S/${C}_a/${C}_ep1807_s18071.done ] && [ -f $S/${C}_b/${C}_ep1807_s18071.done ]; then
    $PY $P/compare_pair.py $S/compare_$C.json $S/${C}_a/${C}_ep1807_s18071 $S/${C}_b/${C}_ep1807_s18071 > $S/compare_$C.log 2>&1
    echo "compare $C rc=$? $(grep -m1 all_bit_identical $S/compare_$C.log)"
  else echo "compare $C skipped: run missing"; fi
done
echo "s21 end $(date -Iseconds)"
