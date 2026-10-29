#!/bin/bash
source ${AG3S_PROJECT_ROOT}/outputs/verify/T44/CL/p/env.sh
for b in 1 2 3; do $P/launch_cl.sh $b; sleep 5; done
