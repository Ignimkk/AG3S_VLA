#!/usr/bin/env bash
# 실물 RB-Y1 red_block 16D pi0.5 LoRA 학습 (config pi05_rby1_real_red_block_16d_lora), SDM 용.
#
# 이 branch(rby1-real/red-block)는 benchmark checkout(/mnt/dev/AG3S_VLA)의 HEAD를 바꾸지 않도록
# git worktree(/mnt/dev/AG3S_VLA-rby1real)로 받아서 쓴다. Python은 benchmark checkout의
# .venv-openpi-live를 빌려 쓰고, env.sh의 PYTHONPATH가 이 worktree의 third_party/openpi/src를 먼저 읽는다
# (`check`가 openpi import 경로를 확인한다).
#
# 순서 (모두 worktree root에서):
#   bash scripts/rby1/real_red_block.sh init     # run dir 생성 (results/benchmark/<KST>_<commit>_rby1real-red-block-v1-sdm)
#   bash scripts/rby1/real_red_block.sh check    # CPU: openpi 경로, dataset 64/13135, AV1 frame decode, config
#   bash scripts/rby1/real_red_block.sh fetch    # CPU: pi05_base (~12 GB) 를 GPU lock 밖에서 미리 받기
#   bash scripts/rby1/real_red_block.sh norm     # CPU: norm stats + 16D · 유한값 검사
#   bash scripts/rby1/real_red_block.sh smoke    # GPU(shared lock): 30 step + checkpoint 저장, GPU memory peak 기록
#   bash scripts/rby1/real_red_block.sh train    # GPU(shared lock): 20k step, tmux session rby1_real_red_block
#   bash scripts/rby1/real_red_block.sh resume   # 같은 run 에서 마지막 checkpoint 부터 이어서
#   bash scripts/rby1/real_red_block.sh status   # run dir, tmux, gpu_lock, checkpoint 목록, log 끝
#
# 환경 변수: EST_GIB (gpu_lock 추정 peak, 기본 60 — smoke 결과로 고친다), NUM_WORKERS (기본 2: pod 규칙
# 세션당 CPU worker <= 2), EXP (기본 red_block_v1), RBY1_MAIN_CHECKOUT (기본 /mnt/dev/AG3S_VLA).
# GPU 규칙: 학습 결과는 wall-clock 과 무관하므로 shared mode. 단, shared holder 가 있는 동안 exclusive
# (closed-loop) 요청은 기다린다 — 긴 학습을 시작하기 전에 같은 pod 의 다른 세션과 일정을 맞춘다.
set -euo pipefail

CONFIG=pi05_rby1_real_red_block_16d_lora
REPO_ID=local/rby1_red_block_16d_v1
TMUX_SESSION=rby1_real_red_block
EXP="${EXP:-red_block_v1}"
EST_GIB="${EST_GIB:-60}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAIN_CHECKOUT="${RBY1_MAIN_CHECKOUT:-/mnt/dev/AG3S_VLA}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ ! -x "$ROOT/.venv-openpi-live/bin/python" ] && [ -z "${AG3S_OPENPI_PYTHON:-}" ]; then
  export AG3S_OPENPI_PYTHON="$MAIN_CHECKOUT/.venv-openpi-live/bin/python"
fi
# shellcheck source=/dev/null
source "$ROOT/scripts/setup/env.sh"   # cd "$ROOT", HF_LEROBOT_HOME, OPENPI_DATA_HOME, XLA_FLAGS, PYTHONPATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled
PY="$AG3S_OPENPI_PYTHON"
[ -x "$PY" ] || { echo "OpenPI python 없음: $PY" >&2; exit 1; }

STATE_DIR="$ROOT/.runtime/rby1_real_red_block"
STATE_FILE="$STATE_DIR/run_dir"

die() { echo "ERROR: $*" >&2; exit 1; }

load_run() {
  [ -s "$STATE_FILE" ] || die "run dir 없음. 먼저: bash scripts/rby1/real_red_block.sh init"
  AG3S_RUN_DIR="$(cat "$STATE_FILE")"
  [ -d "$AG3S_RUN_DIR" ] || die "run dir 가 사라짐: $AG3S_RUN_DIR"
  export AG3S_RUN_DIR
  mkdir -p "$AG3S_RUN_DIR/logs"
}

norm_file() { echo "$AG3S_RUN_DIR/assets/$CONFIG/$REPO_ID/norm_stats.json"; }

cmd_init() {
  if [ -s "$STATE_FILE" ]; then
    echo "이미 있음: $(cat "$STATE_FILE")"
    echo "새 run 을 만들려면: rm $STATE_FILE"
    return 0
  fi
  mkdir -p "$STATE_DIR"
  local kst commit name run
  kst="$(date -u -d '+9 hours' +%Y%m%dT%H%M%S)+0900"
  commit="$(git -C "$ROOT" rev-parse --short=12 HEAD)"
  name="${kst}_${commit}_rby1real-red-block-v1-sdm"
  run="$(env -u AG3S_RUN_DIR "$PY" -m common.run_output --run-dir "$name" | tail -1)"
  echo "$run" > "$STATE_FILE"
  echo "run dir: $run"
}

cmd_check() {
  load_run
  df -h /mnt/dev | tail -1
  echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"
  echo "OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
  JAX_PLATFORMS=cpu RBY1_CONFIG="$CONFIG" RBY1_REPO_ID="$REPO_ID" "$PY" - <<'EOF'
import os, pathlib
root = pathlib.Path(os.environ["AG3S_PROJECT_ROOT"]).resolve()
import openpi
src = pathlib.Path(openpi.__file__).resolve()
assert root in src.parents, f"openpi 가 이 worktree 가 아닌 곳에서 import 됨: {src}"
print("openpi:", src)

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(os.environ["RBY1_REPO_ID"])
print("episodes", ds.num_episodes, "frames", ds.num_frames, "tasks", ds.meta.tasks, "fps", ds.fps)
assert (ds.num_episodes, ds.num_frames, ds.fps) == (64, 13135, 15)
item = ds[ds.num_frames // 2]  # AV1 decode 확인
for k in ("observation.state", "action", "observation.images.cam_high",
          "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"):
    print(f"  {k}: {tuple(item[k].shape)}")
print("  task:", item["task"])

from openpi.training import config as _config
cfg = _config.get_config(os.environ["RBY1_CONFIG"])
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
print("config", cfg.name, "repo", dc.repo_id, "episodes", len(dc.episodes),
      "action_horizon", cfg.model.action_horizon, "steps", cfg.num_train_steps,
      "save/keep", cfg.save_interval, cfg.keep_period, "quantile_norm", dc.use_quantile_norm)
print("CHECK OK")
EOF
}

cmd_fetch() {
  load_run
  JAX_PLATFORMS=cpu "$PY" - <<'EOF'
from openpi.shared import download
print("pi05_base:", download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params"))
EOF
}

cmd_norm() {
  load_run
  local f; f="$(norm_file)"
  if [ -s "$f" ]; then
    echo "이미 있음: $f (다시 만들려면 지운 뒤 실행)"
  else
    JAX_PLATFORMS=cpu "$PY" third_party/openpi/scripts/compute_norm_stats.py --config-name "$CONFIG" \
      2>&1 | tee "$AG3S_RUN_DIR/logs/norm_stats.console.log"
  fi
  NORM_FILE="$f" "$PY" - <<'EOF'
import json, os
import numpy as np
s = json.load(open(os.environ["NORM_FILE"]))["norm_stats"]
np.set_printoptions(precision=4, suppress=True, linewidth=160)
for key in ("state", "actions"):
    for stat in ("mean", "std", "q01", "q99"):
        v = np.asarray(s[key][stat], dtype=float)
        assert v.shape == (16,), (key, stat, v.shape)
        assert np.isfinite(v).all(), (key, stat, v)
    q01, q99 = np.asarray(s[key]["q01"]), np.asarray(s[key]["q99"])
    print(f"{key:7s} q01 {q01}\n{'':7s} q99 {q99}\n{'':7s} q99-q01 {q99 - q01}")
print("NORM OK (idx 0-7 = left arm + left gripper: 상수라 q99-q01 이 0 에 가까운 것이 정상)")
EOF
}

# gpu_lock 아래에서 학습. 1번째 인자 = tag, 나머지 = train.py 인자.
locked_train() {
  local tag="$1"; shift
  [ -s "$(norm_file)" ] || die "norm stats 없음. 먼저: bash scripts/rby1/real_red_block.sh norm"
  "$PY" scripts/setup/gpu_lock.py shared --est-gib "$EST_GIB" --tag "$tag" -- \
    "$PY" third_party/openpi/scripts/train.py "$CONFIG" \
      --assets-base-dir "$AG3S_RUN_DIR/assets" \
      --num-workers "$NUM_WORKERS" --no-wandb-enabled "$@"
}

cmd_smoke() {
  load_run
  local mem_log="$AG3S_RUN_DIR/logs/smoke_gpu_mem.csv" log="$AG3S_RUN_DIR/logs/smoke.console.log"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 \
    | awk -F', ' '{print "GPU 사용 중 (시작 전): " $1 " / " $2 " MiB"}'
  ( while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 >> "$mem_log" || true
      sleep 5
    done ) &
  SAMPLER=$!
  trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT
  set +e
  locked_train rby1-real-smoke --exp-name smoke --num-train-steps 30 --overwrite 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  kill "$SAMPLER" 2>/dev/null || true
  echo "train exit code: $rc"
  awk 'NR==1{min=$1} {if($1>max)max=$1; if($1<min)min=$1} END{printf "GPU memory.used: min %d MiB, max %d MiB (차이 ≈ 이 학습의 peak, 다른 job 이 함께 돌았으면 섞임)\n", min, max}' "$mem_log"
  ls -d "$AG3S_RUN_DIR/checkpoints/$CONFIG/smoke/"* 2>/dev/null || echo "checkpoint 없음"
  echo "확인 뒤 지우기: rm -rf $AG3S_RUN_DIR/checkpoints/$CONFIG/smoke"
  return "$rc"
}

start_tmux() {
  local extra="$1"
  command -v tmux >/dev/null || die "tmux 없음"
  tmux has-session -t "$TMUX_SESSION" 2>/dev/null && die "tmux session 이 이미 있음: $TMUX_SESSION"
  local log="$AG3S_RUN_DIR/logs/train_${EXP}.console.log"
  tmux new-session -d -s "$TMUX_SESSION" \
    "cd '$ROOT' && EXP='$EXP' EST_GIB='$EST_GIB' NUM_WORKERS='$NUM_WORKERS' bash scripts/rby1/real_red_block.sh _train_fg $extra 2>&1 | tee -a '$log'; echo '[끝남] Enter 로 닫기'; read"
  echo "started tmux '$TMUX_SESSION' (보기: tmux attach -t $TMUX_SESSION, 빠지기: Ctrl-b d)"
  echo "log: $log"
  echo "checkpoints: $AG3S_RUN_DIR/checkpoints/$CONFIG/$EXP/"
}

cmd_train() {
  load_run
  [ -e "$AG3S_RUN_DIR/checkpoints/$CONFIG/$EXP" ] && die "이미 있음: $AG3S_RUN_DIR/checkpoints/$CONFIG/$EXP (이어서는 resume, 새로는 EXP=다른이름)"
  start_tmux ""
}

cmd_resume() {
  load_run
  [ -d "$AG3S_RUN_DIR/checkpoints/$CONFIG/$EXP" ] || die "이어갈 checkpoint 없음: $AG3S_RUN_DIR/checkpoints/$CONFIG/$EXP"
  start_tmux "--resume"
}

cmd__train_fg() {
  load_run
  echo "$(date -Is) start $CONFIG exp=$EXP run=$AG3S_RUN_DIR est=${EST_GIB}GiB workers=$NUM_WORKERS $*"
  locked_train "rby1-real-$EXP" --exp-name "$EXP" "$@"
  echo "$(date -Is) done"
}

cmd_status() {
  load_run
  echo "run dir: $AG3S_RUN_DIR"
  tmux has-session -t "$TMUX_SESSION" 2>/dev/null && echo "tmux: $TMUX_SESSION 실행 중" || echo "tmux: 없음"
  "$PY" scripts/setup/gpu_lock.py status || true
  ls -1 "$AG3S_RUN_DIR/checkpoints/$CONFIG/$EXP" 2>/dev/null || echo "checkpoint 없음"
  local log="$AG3S_RUN_DIR/logs/train_${EXP}.console.log"
  if [ -f "$log" ]; then echo "--- $log (끝)"; tail -n 5 "$log"; fi
}

case "${1:-}" in
  init|check|fetch|norm|smoke|train|resume|status|_train_fg) sub="$1"; shift; "cmd_${sub}" "$@" ;;
  *) sed -n '2,22p' "$0"; exit 2 ;;
esac
