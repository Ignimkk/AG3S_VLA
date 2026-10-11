#!/bin/bash
cd ${AG3S_PROJECT_ROOT}/outputs/verify/T37/basel
until bash wait_idle.sh 590; do :; done
bash run_one.sh legacy legacy_shim
echo "legacy done $(date -Iseconds)"
until bash wait_idle.sh 590; do :; done
bash run_one.sh curobo curobo_r1
echo "curobo done $(date -Iseconds)"
