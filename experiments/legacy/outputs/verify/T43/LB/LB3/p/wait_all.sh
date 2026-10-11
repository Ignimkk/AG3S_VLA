#!/bin/bash
# usage: wait_all.sh TAG  -> blocks until launch_TAG.out has 'all end'
until grep -q "^all end $1" ${AG3S_PROJECT_ROOT}/outputs/verify/T43/LB/LB3/runs/launch_$1.out 2>/dev/null; do sleep 30; done
echo finished
