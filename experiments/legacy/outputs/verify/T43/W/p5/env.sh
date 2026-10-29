# T43W V5 rerun — common settings (copy of T43/W/p56/env.sh with the third snapshot ${AG3S_PROJECT_ROOT}-t43w3 = benchmark 72ed927)
V=${AG3S_PROJECT_ROOT}/outputs/verify/T43/W
P=$V/p5
O1=${AG3S_PROJECT_ROOT}-t43w3                                # third snapshot (V5 rerun only): benchmark 72ed927 (with_overrides fix), pi05_TO_hybrid rby1_* fce4af8
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
CKPT=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999
MODEL_XML=$O1/robots/rby1/description/models/rby1a/mujoco/model_transport.xml   # server robot model, as T39/T40/T41
XLA="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"   # as T39-T41 (autotune 0)
PARTIAL=AG3S/ag3s/docs/handoff/T43W.verify.partial.json
GPU_BUDGET_GB=${GPU_BUDGET_GB:-130}                   # used + EST(new server) <= 130 GB
GPU_WAIT_MAX_S=${GPU_WAIT_MAX_S:-7200}
CLIENT_TIMEOUT_S=${CLIENT_TIMEOUT_S:-3600}
SWEEP="--sweep-max-mm 24 --sweep-max-samples 4 --sweep-rows-per-step 6"   # V3 sweep, chosen offline (outputs/verify/T43/W/sweep_tune)
side_of() { case $1 in 1834) echo right;; *) echo left;; esac; }   # active arm per episode (T43 P2: ep1834 right)
SERVER_UP_TIMEOUT_S=${SERVER_UP_TIMEOUT_S:-900}
OBST_STOP=0                                           # --obstacle-stop-distance 0
port_ok() { [ "$1" != 8123 ] && [ "$1" -ge 8280 ] 2>/dev/null && [ "$1" -le 8289 ]; }
# memory rule (lead): working set = memory.current - inactive_file (memory.stat); working set + estimate <= 64 GiB
wset() { echo $(( $(cat /sys/fs/cgroup/memory.current) - $(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat) )); }
# stagger: no serve_safe process (anyone's, incl. ours) started less than 120 s ago (read-only ps)
recent_server_s() { ps -eo etimes=,args= | awk '/benchmark\.trajopt\.serve_safe/ && !/awk/ {print $1}' | sort -n | head -1; }
