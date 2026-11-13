#!/usr/bin/env bash
# (sdm-f6, 2026-11-13 22:3x; user: "네 뒤로 미룹시다") Defer the §9.37 stage 2 mechanism record (64 ep, not an adoption
# condition) until after Tab. 1. The detached gate (outputs/detached/gate37.pid, sota_gate37_full.sh) would start stage 2 right
# after a candidate passes stages 1 and 3. This watcher polls the gate log; when "stage 2 mechanism record" appears it stops the
# gate's process group (the stage-2 launcher is at most seconds old; its run dir, if any, is marked INTERRUPTED) and runs the
# L12H3 second candidate (E1a 30 ep + judge + guard + judge) itself. If no candidate passes (gate goes on to L12H3 by itself)
# the watcher just exits. Run detached:
#   bash experiments/segmentation/run_detached.sh gate37b 'bash /mnt/dev/AG3S_VLA-wt-hf/experiments/segmentation/gate37_defer_ms.sh'
set -u -o pipefail
cd /mnt/dev/AG3S_VLA
L=outputs/2026-11-11_SOTA_개선/gate37/queue.log
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$L"; }
P=$(cat outputs/detached/gate37.pid)
n0=$(wc -l < "$L")
log "defer-ms watcher: watching gate37 pid $P from log line $n0 (user: stage 2 after Tab. 1)"
while kill -0 "$P" 2>/dev/null; do
  new=$(tail -n +"$((n0 + 1))" "$L")
  if grep -q "stage 2 mechanism record" <<< "$new"; then
    kill -TERM -- "-$P" 2>/dev/null; sleep 20; kill -KILL -- "-$P" 2>/dev/null
    for d in $(ls -d /mnt/dev/AG3S_VLA-storage/results/benchmark/2*_ms-g37*sdm-* 2>/dev/null); do
      [ -f "$d/episodes.jsonl" ] || echo "$(date +%Y-%m-%dT%H:%M:%S) INTERRUPTED: stage 2 deferred by the user (stopped at start). Do NOT score." > "$d/INTERRUPTED.txt"
    done
    log "defer-ms watcher: stopped gate37 at the start of stage 2 (deferred until after Tab. 1)"
    break
  fi
  if grep -qE "candidate a12h3" <<< "$new"; then log "defer-ms watcher: gate went on to a12h3 by itself — nothing to do"; exit 0; fi
  sleep 2
done
if ! grep -q "defer-ms watcher: stopped gate37" "$L"; then log "defer-ms watcher: gate37 ended without stage 2"; exit 0; fi
J=".venv-ag3s/bin/python experiments/segmentation/gate37_judge.py"
V="AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-rs-a12h3-tro"
I="5,6,7,8,9,10,11,12,13,14"
log "candidate a12h3 (user 18:4x: SSR 7 -> 9): stage 1 E1a 40 = existing sub-a12h3sdm-e1a t1 (same layout) + t0, t2, t3"
V="$V" TAG=a12h3sdm SUB="0:$I 2:$I 3:$I" bash experiments/segmentation/sota_sub_queue.sh
if $J e1a a12h3sdm | tee -a "$L"; then
  log "candidate a12h3: stage 3 guard"
  V="$V" TAG=a12h3sdm OUT=outputs/2026-11-11_SOTA_개선/eval_a12h3sdm MIN27=21 bash experiments/segmentation/gtto16_queue.sh
  $J guard a12h3sdm | tee -a "$L" || log "candidate a12h3: stage 3 FAIL"
else
  log "candidate a12h3: stage 1 FAIL"
fi
log "gate37 full end (stage 2 deferred)"
