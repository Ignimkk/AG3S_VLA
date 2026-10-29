# SUBTASK-g — common settings (copied from SUBTASK-f p/env.sh; snapshot work-sg = benchmark bf65756 + pi05_TO_hybrid 22d8dae archives)
SD_OUT=${AG3S_PROJECT_ROOT}/outputs/verify/SUBTASK-g
P=$SD_OUT/p
SD=${AG3S_PROJECT_ROOT}-sg                                    # immutable code snapshot (benchmark bf65756 + pi05_TO_hybrid 22d8dae rby1_bringup/rby1_manipulation/rby1_description)
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
CKPT=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999
MODEL_XML=$SD/robots/rby1/description/models/rby1a/mujoco/model_transport.xml
XLA="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"   # same as T39 S2
GPU_BUDGET_GB=${GPU_BUDGET_GB:-130}
GPU_WAIT_MAX_S=${GPU_WAIT_MAX_S:-3600}
CLIENT_TIMEOUT_S=${CLIENT_TIMEOUT_S:-2400}
SERVER_UP_TIMEOUT_S=${SERVER_UP_TIMEOUT_S:-900}
port_ok() { [ "$1" != 8123 ] && [ "$1" -ge 8230 ] 2>/dev/null && [ "$1" -le 8239 ]; }
