# T44d CL common settings (copy of T43/V11/p/env.sh). snapshot ${AG3S_PROJECT_ROOT}-t44cl = benchmark f891abe (git archive) + pi05_TO_hybrid rby1_* copied from ${AG3S_PROJECT_ROOT}-t43v11 (== work-t43w3 fce4af8, same as LB2 E0 runs); other pi05 entries are symlinks.
# arms C0 / C1a / C1b / C2 (T44d.task.md)
V=${AG3S_PROJECT_ROOT}/outputs/verify/T44/CL
P=$V/p
O1=${AG3S_PROJECT_ROOT}-t44cl
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
CKPT=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999
MODEL_XML=$O1/robots/rby1/description/models/rby1a/mujoco/model_transport.xml
XLA="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"
GPU_BUDGET_GB=${GPU_BUDGET_GB:-130}                   # lead rule here: GPU free >= 10 GB; EST per server 28 GB
GPU_WAIT_MAX_S=${GPU_WAIT_MAX_S:-7200}
CLIENT_TIMEOUT_S=${CLIENT_TIMEOUT_S:-3600}
# active arm per episode = LB2/passed.json used_arm
side_of() { case $1 in 12|0|104) echo left;; 30|3|11|54|123|110) echo right;; *) echo "BADEP" >&2; echo left;; esac; }
SERVER_UP_TIMEOUT_S=${SERVER_UP_TIMEOUT_S:-900}
OBST_STOP=0
port_ok() { [ "$1" != 8123 ] && [ "$1" -ge 8280 ] 2>/dev/null && [ "$1" -le 8289 ]; }
wset() { echo $(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) )); }
recent_server_s() { ps -eo etimes=,args= | awk '/benchmark\.trajopt\.serve_safe/ && !/awk/ {print $1}' | sort -n | head -1; }

# V11AB TA flags + TP (bit-identical speedups)
TA_A="--plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4"
TA_B="--w-deflection-rate 20"
TP="--to-fast-eval --esdf-host-mirror"
T44A="--target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 --attention-split-ratio-source raw --target-permission-strict"
arm_flags() { case $1 in
  C0)  echo "$TA_A $TA_B $TP";;
  C1a) echo "$TA_A $TA_B $TP $T44A";;
  C1b) echo "$TA_A $TA_B $TP $T44A --target-free-coarse";;
  C2)  echo "$TA_A $TA_B $TP $T44A --target-free-coarse";;
  *) echo "unknown ARM $1" >&2; return 1;; esac; }
arm_links() { case $1 in C2) echo arms;; *) echo gripper;; esac; }
is_arm() { case $1 in C0|C1a|C1b|C2) return 0;; *) return 1;; esac; }
