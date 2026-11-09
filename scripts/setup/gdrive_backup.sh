#!/usr/bin/env bash
# Back up gitignored research records to Google Drive with rclone (user request 2026-11-09 14:4x KST, dev-4a).
# One tar.gz per folder (2026-11-09 15:5x: per-file upload ran at ~5-9 files/min through Drive, 12.4k files -> >1 day;
# ~400 archives instead):
#   checkout outputs/<entry>/                 -> <REMOTE>:<DEST>/archive/outputs/<entry>.tar.gz   (top-level files: _top.tar.gz)
#   AG3S_VLA-storage/results/benchmark/<run>/ -> <REMOTE>:<DEST>/archive/results_benchmark/<run>.tar.gz
# Excluded: every *.npz (user 2026-11-09), __pycache__, *.pyc, .manifest.lock, outputs/_backup_logs (uploaded as plain files).
# Incremental: a folder is re-uploaded only when its signature (relative path, size, mtime of every kept file) changed since
# the last successful upload (state in outputs/_backup_logs/state_*.tsv). Nothing on Drive is deleted.
# Restore one folder: rclone copy <REMOTE>:<DEST>/archive/outputs/<entry>.tar.gz . && tar xzf <entry>.tar.gz
# usage: bash scripts/setup/gdrive_backup.sh [--dry-run]     (--dry-run lists what would be uploaded, uploads nothing)
#   env: REMOTE (default gdrive), DEST (default AG3S_VLA_backup), RCLONE (default: rclone on PATH)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STORAGE=${AG3S_STORAGE_ROOT:-$(dirname "$ROOT")/AG3S_VLA-storage}
REMOTE=${REMOTE:-gdrive}
DEST=${DEST:-AG3S_VLA_backup}
RCLONE=${RCLONE:-$(command -v rclone || true)}
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
[ -x "$RCLONE" ] || { echo "rclone not found (set RCLONE=/path/to/rclone)"; exit 1; }
"$RCLONE" listremotes | grep -qx "$REMOTE:" || { echo "rclone remote '$REMOTE:' not configured (run: rclone config)"; exit 1; }

STAMP=$(date +%Y%m%dT%H%M%S)
LOGDIR=$ROOT/outputs/_backup_logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/backup_$STAMP.log
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }
EXCL=(--exclude='*.npz' --exclude='__pycache__' --exclude='*.pyc' --exclude='.manifest.lock' --exclude='_backup_logs')

# signature of a folder (or of the top-level files of outputs/ when $2 = top)
sig() {
  if [ "${2:-}" = top ]; then find "$1" -maxdepth 1 -type f ! -name '*.npz' -printf "%P %s %T@\n"
  else find "$1" -type f ! -name '*.npz' ! -name '*.pyc' ! -name '.manifest.lock' ! -path '*/__pycache__/*' -printf "%P %s %T@\n"
  fi | sort | md5sum | cut -c1-32
}

# backup_set <kind> <base dir>: one archive per entry of <base dir>
backup_set() {
  local kind=$1 base=$2 state=$LOGDIR/state_$1.tsv n_up=0 n_skip=0 n_fail=0
  touch "$state"
  local entries=()
  while IFS= read -r e; do entries+=("$e"); done < <(cd "$base" && find . -mindepth 1 -maxdepth 1 -type d ! -name _backup_logs ! -name __pycache__ -printf "%P\n" | sort)
  [ "$kind" = outputs ] && entries+=("_top")
  for e in "${entries[@]}"; do
    local s; if [ "$e" = _top ]; then s=$(sig "$base" top); else s=$(sig "$base/$e"); fi
    if grep -qxF "$e"$'\t'"$s" "$state"; then n_skip=$((n_skip + 1)); continue; fi
    if [ $DRY = 1 ]; then log "[dry] $kind/$e"; n_up=$((n_up + 1)); continue; fi
    local dst="$REMOTE:$DEST/archive/$kind/$e.tar.gz" rc
    if [ "$e" = _top ]; then
      (cd "$base" && find . -maxdepth 1 -type f ! -name '*.npz' -printf "%P\n" | tar czf - -T -) | "$RCLONE" rcat "$dst"
    else
      tar czf - -C "$base" "${EXCL[@]}" "$e" | "$RCLONE" rcat "$dst"
    fi
    rc=$?
    if [ $rc = 0 ]; then
      grep -vF "$e"$'\t' "$state" > "$state.tmp"; printf "%s\t%s\n" "$e" "$s" >> "$state.tmp"; mv "$state.tmp" "$state"
      n_up=$((n_up + 1)); log "up $kind/$e"
    else
      n_fail=$((n_fail + 1)); log "FAIL rc=$rc $kind/$e"
    fi
  done
  log "$kind: uploaded $n_up, unchanged $n_skip, failed $n_fail"
}

log "backup $STAMP -> $REMOTE:$DEST/archive  HEAD=$(git -C "$ROOT" rev-parse --short HEAD) dry=$DRY"
# two sets in parallel (CPU workers <= 2: one tar | gzip each)
backup_set outputs "$ROOT/outputs" & P1=$!
backup_set results_benchmark "$STORAGE/results/benchmark" & P2=$!
wait $P1; wait $P2
# manifest = what is in the archives (file list with sizes), kept as plain text next to them
MAN=$LOGDIR/manifest_$STAMP.txt
{
  echo "# backup $STAMP  HEAD=$(git -C "$ROOT" rev-parse --short HEAD)  dest=$REMOTE:$DEST/archive  excluded=*.npz,__pycache__,*.pyc,.manifest.lock"
  echo "## outputs/ (size path)"; (cd "$ROOT/outputs" && find . -type f ! -name '*.npz' ! -name '*.pyc' ! -path '*/__pycache__/*' ! -path './_backup_logs/*' -printf "%s %P\n" | sort -k2)
  echo "## results/benchmark/ (size path)"; (cd "$STORAGE/results/benchmark" && find . -type f ! -name '*.npz' ! -name '.manifest.lock' -printf "%s %P\n" | sort -k2)
} > "$MAN"
[ $DRY = 1 ] || "$RCLONE" copy "$LOGDIR" "$REMOTE:$DEST/logs" --log-level ERROR
log "backup $STAMP done (log $LOG, manifest $MAN)"
grep -q "FAIL" "$LOG" && exit 1 || exit 0
