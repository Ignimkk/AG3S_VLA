# T39 S2 — common settings, sourced by start_server.sh / run_client.sh / run_chain.sh (verifier, 2026-10-02)
T39=${AG3S_PROJECT_ROOT}/outputs/verify/T39
P=$T39/p
O1=${AG3S_PROJECT_ROOT}-o1                                   # immutable code snapshot (S2-0)
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
CKPT=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999
MODEL_XML=$O1/robots/rby1/description/models/rby1a/mujoco/model_transport.xml
XLA="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"   # lead decision T39 S1 (autotune 0, both conditions)
PARTIAL=AG3S/ag3s/docs/handoff/T39.verify.partial.json
GPU_BUDGET_GB=${GPU_BUDGET_GB:-130}                   # GPU rule 2026-10-02: total used + EST(new server) <= 140 GB - 10 GB margin
GPU_WAIT_MAX_S=${GPU_WAIT_MAX_S:-3600}                # budget check fails -> re-check every 60 s up to this long, then refuse
CLIENT_TIMEOUT_S=${CLIENT_TIMEOUT_S:-2400}
SERVER_UP_TIMEOUT_S=${SERVER_UP_TIMEOUT_S:-900}
port_ok() { [ "$1" != 8123 ] && [ "$1" -ge 8220 ] 2>/dev/null && [ "$1" -le 8229 ]; }
