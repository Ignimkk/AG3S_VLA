# Shared storage and Python interpreter

현재 정본: **2026-10-12 KST**, `feature/ag3s-development`.
Development는 자기 source와 독립 환경을 사용한다. Benchmark checkout의 GPU 설치 상태는 별도다.
최신 설치·검증 근거와 legacy 삭제 조건은 [GPU_OUTPUT_IMPLEMENTATION_REPORT.md](migration/GPU_OUTPUT_IMPLEMENTATION_REPORT.md)에 있다.
이전 v1.4 관리 문서 원본은 [보존 manifest](migration/legacy-management-v1.4/manifest.json)에 기록한다.

## Shared input와 새 result

```text
/mnt/dev/AG3S_VLA-storage/
├── checkpoints/rby1/          # 기존 checkpoint: 읽기 전용 input
├── datasets/                  # 기존 dataset: 읽기 전용 input
├── recordings/                # 기존 video·policy_records: 읽기 전용 input
├── assets/                    # 기존 attention·OpenPI asset
├── results/
│   ├── workspace/             # 과거 evidence, 수정하지 않음
│   ├── rby1/                  # 과거 evidence, 수정하지 않음
│   ├── rby1-logs/             # 과거 log, 수정하지 않음
│   └── development/
│       └── <KST-date_commit_run-id>/
│           ├── run_manifest.json
│           ├── videos/
│           ├── policy_records/session_<unique>/step_N.npy
│           ├── logs/          # Python log와 W&B local files
│           └── checkpoints/   # 새 training checkpoint
├── cache/rby1/                # 이전 다운로드 input 보존
├── tools/{uv,tectonic}
└── legacy-inputs/             # input용 relative alias, source repository가 아님
```

`AG3S_STORAGE_ROOT`는 기본적으로 checkout 옆 `AG3S_VLA-storage`다.
`env.sh`는 `AG3S_LEGACY_STORAGE=$AG3S_STORAGE_ROOT/legacy-inputs`와
`AG3S_RESULTS_ROOT=$AG3S_STORAGE_ROOT/results/development`를 설정한다.
`common.run_output`은 결과 root를 독립 검증하고 KST 날짜·source commit·UUID로 run을 생성한다.
기존 `outputs/` 폴더는 현재 writer의 기본 저장 위치로 사용하지 않는다.
과거 Git source 비교는 자기 `.runtime/legacy-*` frozen fixture를 사용한다.
Checkpoint·dataset과 과거 결과의 절대경로·SHA는 provenance로 그대로 보존한다.

OpenPI의 기존 `third_party/openpi/policy_records` input link는 development에서 제거했다.
과거 기록을 읽을 때는 `$AG3S_STORAGE_ROOT/recordings/policy_records`를 명시한다.
`prepare_legacy_artifacts.py`는 이 link를 다시 만들지 않는다. `assets` input link는 유지한다.
Input inventory는 `experiments/manifests/shared-storage.json`, attention/tool SHA는 `legacy-artifacts.json`이다.
이관 전부터 없었던 atomic/transport dataset alias 두 개는 여전히 missing input이다.

## 실행별 output 설정

서버와 client를 같은 run으로 묶을 때 launching shell에서 다음을 한 번 실행한다.
새 실험을 시작할 때 기존 `AG3S_RUN_DIR`을 unset한다. 각 terminal에서 독립적으로 생성하면 서로 다른 run이 된다.

```bash
cd /mnt/dev/AG3S_VLA-development
source scripts/setup/env.sh
source .venv-openpi-live/bin/activate
unset AG3S_RUN_DIR
export AG3S_RUN_DIR="$(python -m common.run_output)"
echo "$AG3S_RUN_DIR"
```

OpenPI server는 `--run-dir`와 `--record-dir`를 지원한다.
`--record`의 기본값은 `$AG3S_RUN_DIR/policy_records/session_<unique>/step_N.npy`다.
같은 run으로 서버를 재시작하거나 둘 이상 시작해도 서로 다른 session에 기록하므로 `step_0`을 덮어쓰지 않는다.
각 파일은 exclusive creation으로 생성한다. 별도의 상대 `--record-dir`도 run 안에서 해석한다.

```bash
python third_party/openpi/scripts/serve_policy.py \
  --record --run-dir "$AG3S_RUN_DIR" --record-dir policy_records \
  policy:checkpoint \
  --policy.config pi05_rby1_randomized_pick_place_16d_lora \
  --policy.dir "$AG3S_STORAGE_ROOT/checkpoints/rby1/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999"
```

Bringup의 `pi05_infer.py`, `pi05_ex_infer.py`와 `AG3S.trajopt.serve_safe`도 `--run-dir`를 지원한다.
예: `--record videos/rollout.mp4 --record-inputs videos/policy_inputs --record-ag3s ag3s_records`.
활성화한 output 인자는 run 안으로 해석하며 run 밖 절대경로, `..` escape와 input을 향하는 symlink를 거부한다.
`--record-inputs`, `--record-ag3s` 등 기존 recording 옵션은 지정했을 때만 활성화한다.
Atomic/grid evaluation 기본 결과도 run 아래로 저장한다. 알고리즘·모델 입력·평가 판정은 바꾸지 않는다.

OpenPI training의 `checkpoint_base_dir` 기본값은 run의 `checkpoints`다.
명시적인 `--checkpoint-base-dir`도 같은 run 안이어야 한다. Resume할 때는 기존 run의 `AG3S_RUN_DIR`을 지정한다.
기존 공유 checkpoint는 policy loader/weight loader로 읽으며 training resume의 writable destination으로 사용하지 않는다.
Training launcher 두 개는 guard/train log와 새 checkpoint를 run 안에 둔다.
Norm stats 생성은 `--output-dir`로 별도 destination을 지정하고 기본값은 run/assets다.
Training에서 새 stats를 사용할 때는 해당 run/assets를 `--assets-base-dir`로 전달한다.
Python logging은 unique log file, W&B local files는 `logs/wandb`를 사용한다.
Bringup의 일반 `print`/native stderr까지 저장하려면 shell에서 run 아래 파일로 redirect한다.
예: `... > "$AG3S_RUN_DIR/logs/client-console.log" 2>&1`.
새 CLI는 결과 저장을 검증하지만 임의의 외부 script·shell redirect까지 강제하는 filesystem sandbox는 아니다.
과거 `experiments/legacy` launcher는 archive로 취급하며 새 실험을 위해 그대로 실행하지 않는다.

## Interpreter와 독립 재설치

| Development 환경 | 실제 설치 | 용도 |
|---|---|---|
| `.venv-openpi-live` | Python 3.11.16, OpenPI 0.1.0, JAX/JAXlib 0.5.3, Torch 2.7.1+cu126, numpy 1.26.4, MuJoCo 3.11.0 | VS Code 주 interpreter, OpenPI GPU inference |
| `.venv-curobo` | Python 3.11.16, cuRobo 0.8.0.post1.dev43, Torch 2.7.1+cu126, cuda-core 1.2.0, Warp 1.17.0, numpy 1.26.4 | 별도 TSDF/ESDF producer |
| `.venv-ag3s` | Python 3.11.16, numpy 2.4.6 | 기존 CPU 회귀·거리장 소비 |
| Benchmark `.venv-ag3s` | 기존 독립 CPU 환경 | 이 작업에서 변경하지 않음 |

주 interpreter는 tracked `.vscode/settings.json`의 `${workspaceFolder}/.venv-openpi-live/bin/python`이다.
이미 선택된 interpreter는 설정 변경만으로 전환되지 않을 수 있으므로 VS Code에서
`Python: Select Interpreter` → `/mnt/dev/AG3S_VLA-development/.venv-openpi-live/bin/python`을 선택한다.
Shell activation은 별도다. CPU 회귀에서는 `.venv-ag3s`, producer에서는 `.venv-curobo`를 activate한다.

기존 `/mnt/dev/work/.venv-openpi-live`를 metadata로 직접 점검하고 snapshot과 비교했다.
기존 Python 3.10 cuRobo system-site environment를 복사하지 않고 현재 pin의 `requires-python >=3.10`과
Torch/CUDA 요구사항에 따라 독립 Python 3.11 producer를 새로 만들었다.
모든 venv는 system-site-packages를 사용하지 않으며 Python executable은 자기 `.python`에 속한다.
OpenPI/client/workspace/cuRobo editable source는 자기 checkout에만 연결한다.
LeRobot은 기존 고정 Git commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`에서 새 wheel로 설치한다.
cuRobo package version은 기존 inventory와 맞추고 source SHA는 `78fd485fa82d9b9a063fb4985e371814587e666a`를 유지한다.
Torch transformer patch는 자기 vendored fork에서 자기 venv 안에만 적용한다.

```bash
# 새 장비 또는 새 checkout에서 재생성할 때 사용
source scripts/setup/env.sh
bash scripts/setup/create_gpu_env.sh --with-curobo
# CPU는 별도 유지
bash scripts/setup/create_cpu_env.sh
```

`environments/openpi-live/requirements.txt`는 관찰된 live distribution 버전의 재설치 명세다.
`--no-deps`로 기존 버전을 재현하고 source package를 별도로 설치하므로 새 dependency resolution lockfile로 주장하지 않는다.
`environments/curobo/requirements-live.txt`와 `openpi-live/constraints.txt`는 producer dependency resolution에 사용한다.
검증된 producer의 전체 snapshot도 `environments/curobo/resolved.txt`에 보존한다.
OpenPI `uv pip check`는 이전 환경과 같은 **gym-aloha의 MuJoCo <3 제약 1건**을 보고한다.
요청된 MuJoCo 3.11.0과 OpenPI/JAX/Torch 버전을 유지하며 실제 RBY1 checkpoint inference를 별도로 검증한다.
ALOHA 전체 호환성이나 신규 training 완료를 주장하지 않는다.

## Cache와 GPU 직렬 검증

`env.sh`는 자기 checkout의 `.cache/openpi`, `.cache/huggingface`, `.cache/cuda`, `.cache/warp`,
`.cache/jax`, `.cache/torch-extensions`를 설정한다. 이전 terminal의 `/mnt/dev/work` 기본값은 제거한다.
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_FLAGS=--xla_gpu_enable_command_buffer=`를 유지한다.

GPU 검증 전 `nvidia-smi` compute process를 확인하고 양쪽 checkout이 같은 advisory lock을 사용한다.
현재 H200 한 장에서는 OpenPI와 cuRobo 검증을 순서대로 실행한다.
Lock을 사용하지 않는 외부 실행을 통제하거나 기존 process를 종료하지 않는다.
검증 command·raw output·input identity는 최신 보고서와 verify manifest에 기록한다.

```bash
source scripts/setup/env.sh
.venv-ag3s/bin/python scripts/setup/check_workspace.py
.venv-openpi-live/bin/python scripts/setup/check_storage.py --hash-artifacts
.venv-ag3s/bin/python scripts/setup/run_without_legacy.py -m pytest tests/ag3s/test_record_timing.py -q
```

`run_without_legacy.py`는 Python audit hook으로 이전 checkout의 파일 접근을 거부한다.
Native library나 모든 subprocess syscall을 차단하는 완전한 filesystem sandbox는 아니다.
main·benchmark branch는 변경하지 않는다. 공통 경로 변경의 전달과 push는 사용자가 판단한다.

Compatibility references: [JAX installation](https://docs.jax.dev/en/latest/installation.html),
[cuRobo runtime configuration](https://nvlabs.github.io/curobo/latest/reference/runtime_configuration.html).
실제 source 요구사항은 pinned `third_party/curobo/pyproject.toml`, 실행 결과는 최신 validation artifact가 기준이다.
