#!/usr/bin/env bash
# Run a GPU queue detached from the Claude session (sdm-f6, 2026-11-13; user: "세션이 끊기는 것을 대비합시다" — an explicit
# exception to the "no nohup/detach" rule for queues only: three Claude-session restarts on 11-13 killed running queues).
# The queue gets its own session (setsid) so a session teardown does not take it; stdout/stderr go to
# outputs/detached/<NAME>.log and the pid to outputs/detached/<NAME>.pid. The queue still takes the GPU lock itself.
# usage: bash experiments/segmentation/run_detached.sh NAME 'bash experiments/segmentation/<queue>.sh ...'
#   status: kill -0 "$(cat outputs/detached/NAME.pid)" && echo running
#   stop  : kill -TERM -- "-$(cat outputs/detached/NAME.pid)"   (the whole process group; the launchers stop their servers)
set -u
NAME=${1:?NAME}; CMD=${2:?CMD}
cd /mnt/dev/AG3S_VLA
D=outputs/detached; mkdir -p "$D"
if [ -f "$D/$NAME.pid" ] && kill -0 "$(cat "$D/$NAME.pid")" 2>/dev/null; then
  echo "already running: $NAME pid $(cat "$D/$NAME.pid")"; exit 1
fi
echo "$(date +%Y-%m-%dT%H:%M:%S) start $NAME: $CMD (HEAD $(git rev-parse --short HEAD))" >> "$D/$NAME.log"
setsid nohup bash -c "cd /mnt/dev/AG3S_VLA && $CMD; echo \"\$(date +%Y-%m-%dT%H:%M:%S) end $NAME rc=\$?\"" >> "$D/$NAME.log" 2>&1 < /dev/null &
echo $! > "$D/$NAME.pid"
echo "detached $NAME pid $(cat "$D/$NAME.pid") log $D/$NAME.log"
