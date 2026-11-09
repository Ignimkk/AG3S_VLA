#!/usr/bin/env bash
# Back up gitignored research records to Google Drive with rclone (user request 2026-11-09 14:4x KST, dev-4a).
#   (1) checkout outputs/                         -> <REMOTE>:<DEST>/outputs/
#   (2) AG3S_VLA-storage/results/benchmark/       -> <REMOTE>:<DEST>/results_benchmark/
# Excluded: every *.npz (episode records, server_constraints chunks; user 2026-11-09: not backed up), __pycache__, *.pyc,
#   .manifest.lock. Size at 2026-11-09 14:5x: ~2.4 GB / ~12.4k files (run records 1.48 GB + outputs 0.94 GB).
# Mode: `rclone copy` -- never deletes anything on Drive; a re-run uploads only new / changed files.
# A manifest (git HEAD, time, local file list with sizes) is written to outputs/_backup_logs/ and uploaded with (1).
# usage: bash scripts/setup/gdrive_backup.sh [--dry-run]   (extra args are passed to rclone)
#   env: REMOTE (default gdrive), DEST (default AG3S_VLA_backup), RCLONE (default: rclone on PATH)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
STORAGE=${AG3S_STORAGE_ROOT:-$(dirname "$ROOT")/AG3S_VLA-storage}
REMOTE=${REMOTE:-gdrive}
DEST=${DEST:-AG3S_VLA_backup}
RCLONE=${RCLONE:-$(command -v rclone || true)}
[ -x "$RCLONE" ] || { echo "rclone not found (set RCLONE=/path/to/rclone)"; exit 1; }
"$RCLONE" listremotes | grep -qx "$REMOTE:" || { echo "rclone remote '$REMOTE:' not configured (run: rclone config)"; exit 1; }

STAMP=$(date +%Y%m%dT%H%M%S)
LOGDIR=$ROOT/outputs/_backup_logs
mkdir -p "$LOGDIR"
MAN=$LOGDIR/manifest_$STAMP.txt
{
  echo "# backup $STAMP  host=$(hostname)  HEAD=$(git -C "$ROOT" rev-parse --short HEAD)  branch=$(git -C "$ROOT" branch --show-current)"
  echo "# dest=$REMOTE:$DEST  excluded=*.npz,__pycache__,*.pyc,.manifest.lock"
  echo "## outputs/ (size path)"
  (cd "$ROOT/outputs" && find . -type f ! -name '*.npz' ! -name '*.pyc' ! -path '*/__pycache__/*' -printf "%s %P\n" | sort -k2)
  echo "## results/benchmark/ (size path)"
  (cd "$STORAGE/results/benchmark" && find . -type f ! -name '*.npz' ! -name '.manifest.lock' -printf "%s %P\n" | sort -k2)
} > "$MAN"

COMMON=(--exclude '*.npz' --exclude '__pycache__/**' --exclude '*.pyc' --exclude '.manifest.lock'
        --transfers 4 --checkers 4 --drive-chunk-size 64M --stats 60s --stats-one-line --log-level INFO "$@")
"$RCLONE" copy "$ROOT/outputs" "$REMOTE:$DEST/outputs" "${COMMON[@]}" --exclude '_backup_logs/**' --log-file "$LOGDIR/rclone_${STAMP}_outputs.log"
"$RCLONE" copy "$STORAGE/results/benchmark" "$REMOTE:$DEST/results_benchmark" "${COMMON[@]}" \
  --log-file "$LOGDIR/rclone_${STAMP}_benchmark.log"
# the logs of this run go up last so the Drive copy has them too
"$RCLONE" copy "$LOGDIR" "$REMOTE:$DEST/outputs/_backup_logs" --log-level ERROR "$@"
echo "backup $STAMP done -> $REMOTE:$DEST  (manifest $MAN)"
