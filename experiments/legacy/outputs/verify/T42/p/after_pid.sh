#!/bin/bash
# usage: after_pid.sh PID CMD...  -> waits for PID to exit, then execs CMD
P=$1; shift; while kill -0 $P 2>/dev/null; do sleep 20; done; exec "$@"
