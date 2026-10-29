#!/bin/bash
until grep -q "F2 V11ABB" ${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/runs/chainF2_V11AB.out 2>/dev/null; do sleep 15; done
sleep 10
${AG3S_PROJECT_ROOT}/outputs/verify/T43/V11/AB/redo_bad.sh
