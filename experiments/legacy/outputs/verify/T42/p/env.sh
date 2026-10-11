# T42-1 — common settings (from T40/p/env.sh), sourced by start_server.sh / run_client.sh / run_chain.sh (verifier, 2026-10-05)
T42=${AG3S_PROJECT_ROOT}/outputs/verify/T42
P=$T42/p
O1=${AG3S_PROJECT_ROOT}-t42                                  # immutable code snapshot (benchmark e5debe5, pi05_TO_hybrid 98e07d6)
PY=${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python
CKPT=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999
MODEL_XML=$O1/robots/rby1/description/models/rby1a/mujoco/model_transport.xml   # server robot model, as T39/T40
XLA="--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"   # as T39/T40 (autotune 0, both conditions)
PARTIAL=AG3S/ag3s/docs/handoff/T42-1.verify.partial.json
DATASET_META=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1/meta/randomized_episodes.jsonl
MD5_REF=$P/code_md5_t42.txt
GPU_BUDGET_GB=${GPU_BUDGET_GB:-130}                   # used + EST(new server) <= 130 GB
GPU_WAIT_MAX_S=${GPU_WAIT_MAX_S:-3600}
CLIENT_TIMEOUT_S=${CLIENT_TIMEOUT_S:-2400}
SERVER_UP_TIMEOUT_S=${SERVER_UP_TIMEOUT_S:-900}
MEM_LIMIT_BYTES=$(( 64 * 1024 * 1024 * 1024 ))        # memory.current + estimate <= 64 GiB (lead rule)
OBST_STOP=0                                           # --obstacle-stop-distance 0 (no-op without an obstacle manager)
port_ok() { [ "$1" != 8123 ] && [ "$1" -ge 8250 ] 2>/dev/null && [ "$1" -le 8259 ]; }
# used_arm of an episode from the dataset metadata (the same file the client reads)
ep_arm() { python3 -c "import json,sys; e=int(sys.argv[1]); print(next(json.loads(l)['used_arm'] for l in open('$DATASET_META') if l.strip() and json.loads(l)['episode_index']==e))" $1; }
# lead rule 2026-10-05 07:2x UTC: working set = memory.current - inactive_file (memory.stat); working set + estimate <= 64 GiB
mem_ws() { local cur ina; cur=$(cat /sys/fs/cgroup/memory.current); ina=$(awk '$1=="inactive_file"{print $2}' /sys/fs/cgroup/memory.stat); echo $(( cur - ina )); }
# lead rule: no memory / GPU check within 120 s of ANY serve_safe start (any session; read-only ps). Waits until the youngest is >= 120 s old.
servers_settled() { local y; while :; do y=$(ps -eo etimes=,args= | awk '/AG3S.trajopt.serve_safe/ && !/awk/ {print $1}' | sort -n | head -1);
  { [ -z "$y" ] || [ "$y" -ge 120 ]; } && return 0; sleep $(( 121 - y )); done; }
