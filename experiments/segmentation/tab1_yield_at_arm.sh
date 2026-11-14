#!/usr/bin/env bash
# (sdm-f6 SL-AUD session, 2026-11-15 01:4x; user decision 01:3x relayed by dev-cf and confirmed in this session)
# Stop the Tab. 1 queue (tab1q) at the arm boundary: when the AG3S arm has finished and the GT oracle arm's first cell
# starts ("start gto ..."), stop tab1q's process group at once (the launcher is seconds old) and mark any run dir it created
# after that line INTERRUPTED. The GPU then goes to the RB-Y1 basket LoRA training (~1.5 h); the GT oracle arm is relaunched
# afterwards as its own queue (ARMS=gto). If tab1q ends some other way, the watcher just reports and exits.
# Run detached: bash experiments/segmentation/run_detached.sh tab1yield 'bash /mnt/dev/AG3S_VLA-wt-aud/experiments/segmentation/tab1_yield_at_arm.sh'
set -u
cd /mnt/dev/AG3S_VLA
LT=outputs/2026-11-13_paper_tab1_1-10/queue.log
B=/mnt/dev/AG3S_VLA-storage/results/benchmark
log() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LT"; }
P=$(cat outputs/detached/tab1q.pid)
n0=$(wc -l < "$LT")
log "tab1 yield watcher: watching tab1q ($P); stop at the first 'start gto' (user 01:3x: RB-Y1 basket training before the GT oracle arm)"
n0=$(wc -l < "$LT")
while kill -0 "$P" 2>/dev/null; do
  if tail -n +"$((n0 + 1))" "$LT" | grep -q "start gto "; then
    stamp=$(mktemp); touch -d "-3 seconds" "$stamp"
    kill -TERM -- "-$P" 2>/dev/null; sleep 20; kill -KILL -- "-$P" 2>/dev/null
    for d in $(find "$B" -maxdepth 1 -mindepth 1 -type d -newer "$stamp" -name "*tab1sdm-gto-*" 2>/dev/null); do
      [ -f "$d/episodes.jsonl" ] || echo "$(date +%Y-%m-%dT%H:%M:%S) INTERRUPTED: Tab. 1 queue stopped at the arm boundary for RB-Y1 training (user 01:3x). GT oracle arm reruns from its first cell. Do NOT score." > "$d/INTERRUPTED.txt"
    done
    log "tab1 yield watcher: STOPPED tab1q at the arm boundary (AG3S arm complete; GT oracle arm not run) -- GPU to RB-Y1 basket training"
    exit 0
  fi
  sleep 1
done
log "tab1 yield watcher: tab1q ended before the GT oracle arm started -- nothing stopped"
