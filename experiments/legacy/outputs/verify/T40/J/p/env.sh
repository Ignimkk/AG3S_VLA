# T40 J — common settings (copy of T40/p/env.sh) (from T39/p/env.sh), sourced by start_server.sh / run_client.sh / run_chain.sh (verifier, 2026-10-03)
T40=${AG3S_PROJECT_ROOT}/outputs/verify/T40
P=$T40/J/p
O1=${AG3S_PROJECT_ROOT}-t40h                                  # immutable code snapshot (benchmark 150c28f, pi05_TO_hybrid bd6a002)
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
CKPT=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999
MODEL_XML=$O1/robots/rby1/description/models/rby1a/mujoco/model_transport.xml   # server robot model, as T39
XLA="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"   # as T39 (autotune 0, both conditions)
PARTIAL=AG3S/ag3s/docs/handoff/T40FG.verify.partial.json
GPU_BUDGET_GB=${GPU_BUDGET_GB:-130}                   # used + EST(new server) <= 130 GB
GPU_WAIT_MAX_S=${GPU_WAIT_MAX_S:-3600}
CLIENT_TIMEOUT_S=${CLIENT_TIMEOUT_S:-2400}
SERVER_UP_TIMEOUT_S=${SERVER_UP_TIMEOUT_S:-900}
OBST_STOP=0                                           # --obstacle-stop-distance 0 (task: E3b runs with esdf-margin 0)
port_ok() { [ "$1" != 8123 ] && [ "$1" -ge 8240 ] 2>/dev/null && [ "$1" -le 8249 ]; }
