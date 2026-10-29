#!/bin/bash
# T37: run all per-condition analyses after a chain is DONE. usage: analyze_all.sh COND   (shim python, CPU only)
C=$1; S=/tmp/claude-0/-mnt-dev-work/769cea2f-2a69-4fbe-beed-8be40017af85/scratchpad/venv-ag3s-local/bin/python
V=${AG3S_PROJECT_ROOT}/outputs/verify/T37; cd ${AG3S_PROJECT_ROOT}
export MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}
echo "start $(date -Iseconds)"
$S $V/p/analyze_e3_base.py $C > $V/p/analyze_$C.log 2>&1; echo "analyze rc=$?"
$S $V/p/summarize_e3.py $C > $V/p/summarize_$C.log 2>&1; echo "summarize rc=$?"
$S $V/p/t37_extra.py $C > $V/p/t37_extra_$C.log 2>&1; echo "extra rc=$?"
$S $V/p/revoke_detail.py $C > $V/p/revoke_detail_$C.log 2>&1; echo "revoke rc=$?"
$S $V/p/t37_tipdev.py $C > $V/p/tipdev_$C.log 2>&1; echo "tipdev rc=$?"
cd $V/geom && $S a_runs.py a/a_runs.json $(ls -d $V/${C}_rec/${C}_ep*/ | sed 's#/$##') > a/a_runs_$C.log 2>&1; echo "a_runs rc=$?"
cd $V/geom && $S a/settle.py > a/settle.log 2>&1; echo "settle rc=$?"
echo "end $(date -Iseconds)"
