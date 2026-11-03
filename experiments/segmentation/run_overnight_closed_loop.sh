#!/usr/bin/env bash
# Sequential exclusive-lock closed loops: CBF arms first (priority), then LIBERO-Safety VLA-only rollouts with geometry capture.
set -u
cd "$(dirname "$0")/../.."
./experiments/segmentation/run_cbf_arms_safelibero.sh cbf20261103 multilink_cbf:gt multilink_cbf knows:gt knows aegis:gt aegis
./experiments/segmentation/run_safety_vla_rollouts.sh bmls20261103 0,1
echo ALL-DONE
