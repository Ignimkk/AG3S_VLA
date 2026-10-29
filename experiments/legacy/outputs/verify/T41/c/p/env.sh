# T41c — common settings (copy of T41/p/env.sh: snapshot work-t41c, ports 8270-8279, partial T41c), verifier 2026-10-05
T41=${AG3S_PROJECT_ROOT}/outputs/verify/T41
T40=$T41                                              # scripts copied from T40 refer to $T40 for their output root
P=$T41/c/p
O1=${AG3S_PROJECT_ROOT}-t41c                                 # immutable code snapshot (benchmark b7d5849 incl. 7717b9a; pi05_TO_hybrid rby1_description+rby1_manipulation bd6a002, rby1_bringup fce4af8)
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
CKPT=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999
MODEL_XML=$O1/robots/rby1/description/models/rby1a/mujoco/model_transport.xml   # server robot model, as T39/T40
XLA="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"   # as T39/T40 (autotune 0)
PARTIAL=AG3S/ag3s/docs/handoff/T41c.verify.partial.json
GPU_BUDGET_GB=${GPU_BUDGET_GB:-130}                   # used + EST(new server) <= 130 GB
GPU_WAIT_MAX_S=${GPU_WAIT_MAX_S:-3600}
CLIENT_TIMEOUT_S=${CLIENT_TIMEOUT_S:-2400}
SERVER_UP_TIMEOUT_S=${SERVER_UP_TIMEOUT_S:-900}
OBST_STOP=0                                           # --obstacle-stop-distance 0
OBSTACLE_MARGIN=${OBSTACLE_MARGIN:-0.010}             # control runs override with OBSTACLE_MARGIN=0; T41a: --obstacle-margin 0.010 (support plane excluded = default)
port_ok() { [ "$1" != 8123 ] && [ "$1" -ge 8270 ] 2>/dev/null && [ "$1" -le 8279 ]; }
# memory rule (lead, 2026-10-05 07:3x): working set = memory.current - inactive_file (memory.stat); working set + estimate <= 64 GiB
wset() { echo $(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) )); }
# stagger: no serve_safe process (anyone's) started less than 120 s ago (read-only ps)
recent_server_s() { ps -eo etimes=,args= | awk '/benchmark\.trajopt\.serve_safe/ && !/awk/ {print $1}' | sort -n | head -1; }
