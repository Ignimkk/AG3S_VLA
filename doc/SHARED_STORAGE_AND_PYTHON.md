# Shared storage and Python interpreter

현재 정본: 2026-10-12. Benchmark live 설치/검증 결과는
`doc/migration/LIVE_ENVIRONMENT_IMPLEMENTATION_REPORT.md`를 함께 읽는다. 현재 실행은 이전 `/mnt/dev/work` source·input·tool에 의존하지 않는다.
과거 log·recording의 절대경로와 Git source SHA는 provenance로 보존하며 현재 실행 주소가 아니다.

## Storage layout

```text
/mnt/dev/
├── AG3S_VLA-development/     # feature/ag3s-development: v2 이후 개선
├── AG3S_VLA-benchmark/       # feature/panda-benchmark: paired v2 평가
└── AG3S_VLA-storage/         # Git 외부의 실제 input 저장 공간
    ├── checkpoints/rby1/
    ├── datasets/{rby1,lerobot-atomic,lerobot-transport,workspace}/
    ├── recordings/{records,run_0004,run_0005,run_16d_ep1800,policy_records,states}/
    ├── assets/{attention,openpi}/
    ├── results/{workspace,rby1,rby1-logs}/
    ├── backups/rby1/
    ├── cache/rby1/           # 기존 다운로드 자료 보존
    ├── tools/{uv,tectonic}
    └── legacy-inputs/       # 위 자료로 연결하는 relative-path 호환 namespace
```

`AG3S_STORAGE_ROOT`의 기본값은 checkout 옆 `AG3S_VLA-storage`다. 다른 host에서는 이 변수로
실제 저장 위치를 지정한다. `AG3S_LEGACY_STORAGE`는 `source scripts/setup/env.sh`가 설정하는
`$AG3S_STORAGE_ROOT/legacy-inputs` 호환 alias다. 기존 변수만 설정해서 input 저장 위치를 바꾸지 않는다.
`legacy-inputs/pi05_TO_hybrid`와 `legacy-inputs/benchmark`는 input alias용 디렉터리이며 source repository가 아니다.
`workspace_path('legacy/pi05_TO_hybrid')`와 `workspace_path('legacy/benchmark')`처럼 역사적 코드
비교가 필요한 호출은 자기 checkout의 `.runtime/legacy-*` frozen Git fixture를 사용한다.
Historical client를 로드한 replay harness는 input 위치만 `bind_historical_client_inputs`로 재설정한다.
Frozen Git source와 algorithm은 수정하지 않는다.

Checkpoint·dataset·이전 결과는 읽기 전용 input으로 취급한다. 새 training checkpoint·rollout·log는
benchmark run은 `results/benchmark/<날짜_commit_run-id>/`에,
그 외 새 training output은 자기 checkout의 ignored `outputs/`에 쓴다.
`common/run_artifacts.py`가 benchmark run manifest와 고유 저장 위치를 만든다. cache·LIBERO config·venv도 각 checkout에서 독립적으로 관리한다.
이전 자료 경로에는 새 storage로 향하는 backlink를 남겼으므로 이전 도구의 data 접근은 유지된다.
새 checkout의 artifact link와 runtime 기본값은 이 backlink를 거치지 않는다.
OpenPI의 `assets`와 `policy_records`도 외부 storage를 직접 가리키는 ignored link로 연결한다.
이전 terminal에서 상속된 cache·dataset·interpreter 경로가 이전 checkout을 가리키면 `env.sh`가 새 기본값으로 전환한다.
새 Git clone만으로 storage의 대형 자료와 installed tool이 복원되지는 않는다. `experiments/manifests/shared-storage.json`을
기준으로 외부 storage를 준비한 뒤 checksum artifact를 연결한다.

```bash
cd /mnt/dev/AG3S_VLA-development  # 또는 /mnt/dev/AG3S_VLA-benchmark
source scripts/setup/env.sh
.venv-ag3s/bin/python scripts/setup/prepare_legacy_artifacts.py --relink
.venv-ag3s/bin/python scripts/setup/check_workspace.py
.venv-ag3s/bin/python scripts/setup/check_storage.py --hash-artifacts
```

전체 input 위치 목록은 `experiments/manifests/shared-storage.json`, attention/tool SHA-256은
`experiments/manifests/legacy-artifacts.json`에 있다. 기존 atomic/transport LeRobot alias의 dataset
두 개는 이관 전부터 없었다. 이동 작업으로 새로 생성하거나 복구한 dataset으로 간주하지 않는다.

## Interpreter selection in VS Code

두 checkout을 각각 다른 VS Code 창으로 연다. Python extension을 사용하고 `Ctrl+Shift+P` →
`Python: Select Interpreter`에서 해당 창의 `.venv-ag3s/bin/python`을 선택한다.
목록에 없으면 interpreter 경로를 직접 지정한다.

| 창 | 선택할 interpreter |
|---|---|
| development | `/mnt/dev/AG3S_VLA-development/.venv-ag3s/bin/python` |
| benchmark | `/mnt/dev/AG3S_VLA-benchmark/.venv-ag3s/bin/python` |

로컬 `.vscode/settings.json`은 `${workspaceFolder}/.venv-ag3s/bin/python`을 기본값으로 설정했다.
이미 선택된 interpreter가 있으면 위 명령으로 명시적으로 다시 선택한다.
이 설정은 shell의 `python`을 자동 변경하는 것과 별개다. 새 terminal에서 다음을 실행한다.

```bash
source scripts/setup/env.sh
source .venv-ag3s/bin/activate
python -c 'import sys; print(sys.executable); print(sys.version)'
python scripts/setup/check_workspace.py
```

User-level `/root/.local/bin/python3.11` shim은 아직 이전 workspace의 `.python`을 가리킨다(전역 파일이라 수정하지 않음).
`env.sh`는 bare `python3.11`이 이전 workspace로 해석될 때만 ignored `.runtime/bin/python3.11`(자기 `.python`)을 PATH 앞에 둔다.
이미 활성화한 venv의 `python3.11`은 그대로 우선한다.
두 새 CPU 환경의 Python은 3.11.16이다. 기존 editable install 또는 venv를 두 branch가 공유하지 않는다.
현재 확인한 이전 환경은 다음과 같다. 이 표는 재현 참고이며 새 실행 경로가 아니다.

| 이전 환경 | 확인 결과/역할 |
|---|---|
| `/mnt/dev/work/.venv-ag3s/bin/python` | Python 3.11 환경 명세; interpreter 확인 명령이 15초 timeout. 새 CPU 환경 사용 권장 |
| `/mnt/dev/work/.venv-openpi-live/bin/python` | Python 3.11.16; numpy 1.26.4, JAX/JAXlib 0.5.3, Torch 2.7.1, MuJoCo 3.11.0 |
| `/mnt/dev/work/.venv-curobo/bin/python` | Python 3.10.12; numpy 1.26.4, system-site-packages 사용 |
| `/mnt/dev/work/pi05_TO_hybrid/openpi/.venv/bin/python` | Python 3.11.16; 이전 OpenPI 전용 환경 |

이전 환경을 잠시 선택해야 한다면 동일한 `Python: Select Interpreter`에서 그 경로를 지정할 수 있다.
이는 이전 workspace를 계속 사용하는 경우의 방법이며 새 두 checkout의 권장 interpreter가 아니다.

## Recreate the CPU environment

현재 두 환경은 이미 준비되어 있다. 새 장비에서 재생성할 때만 아래를 실행한다.
`uv`가 PATH에 없으면 외부 storage의 `tools/uv`를 사용한다. `AG3S_UV`로 다른 설치를 지정할 수도 있다.

```bash
source scripts/setup/env.sh
bash scripts/setup/create_cpu_env.sh
source .venv-ag3s/bin/activate
```

최초 경로 전환에서는 GPU/OpenPI/cuRobo를 설치하지 않았다. 이후 benchmark checkout에
독립 `.venv-openpi-live`와 `.venv-curobo`(둘 다 Python 3.11.16)를 설치하고
Torch/JAX GPU, cuRobo ESDF kernel, randomized-pick-place 16D checkpoint loading을 검증했다.
Development checkout의 GPU 환경 설치 상태는 이번 작업에서 변경하지 않았다. CPU venv를 VLA inference 환경으로
간주하지 않는다. 이전 환경 package snapshot은 `environments/legacy-python/`에 보존한다.
그 파일은 inventory이며 GPU 환경의 검증된 재설치 lockfile이 아니다. 복사한 venv는 absolute path·
editable source·CUDA extension 경로를 계속 가리킬 수 있으므로 새 checkout에서 재생성해야 한다.
`bash scripts/setup/create_live_env.sh`와
`bash scripts/setup/create_live_env.sh .venv-curobo`로 자기 환경을 재생성한다. OpenPI 환경은 자기 `.venv-openpi-live`에 만들고 `third_party/openpi`와 `third_party/curobo`를
자기 checkout에서 설치한 뒤 import origin, CUDA, checkpoint loading을 검증한다.
Training launcher의 `AG3S_OPENPI_PYTHON` 기본값도 자기 `.venv-openpi-live/bin/python`이다.

## Verify without the previous checkout

```bash
source scripts/setup/env.sh
.venv-ag3s/bin/python scripts/setup/run_without_legacy.py -m pytest tests/ag3s/test_record_timing.py -q
```

이 launcher는 Python 파일 open·directory listing·chdir에서 이전 checkout과 그 alias 접근을
거부하며 Python subprocess에도 같은 audit hook을 주입한다. 첫 probe에서 차단 동작을 확인한다.
Native library나 Python 이외 프로세스의 모든 syscall을 차단하는 filesystem sandbox는 아니다.
Storage와 interpreter의 symlink·Git fixture 경로 검사는 `check_storage.py`가 수행한다.

공통 storage/path 변경만 양쪽 branch에 전달한다. main의 paired v2 source 기준과 두 원본 v2 tag는
유지하며 AG3S algorithm 개선을 benchmark branch에 섞지 않는다. Push는 사용자가 수행한다.

Sources: [VS Code Python environments](https://code.visualstudio.com/docs/python/environments),
[uv environment activation](https://docs.astral.sh/uv/pip/environments/),
[uv Python storage and existing environments](https://docs.astral.sh/uv/reference/storage/).

## Benchmark output protection

새 영상·log·policy record는 `$AG3S_STORAGE_ROOT/results/benchmark/<KST 날짜시간_commit_run-id>/`
아래 저장한다. `AG3S_RUN_DIR` 또는 `AG3S_V11_OUTPUT_DIR`로 선택할 때는 이 helper가 생성한
동일 checkout/commit의 run을 지정한다. Shared input·기존 recordings는 output으로 사용하지 않는다.
OpenPI `--record`는 새 run을 사용하며 기존 directory/step/log가 있으면 덮어쓰지 않고 거부한다.
상세 사용법·metadata 예외·archived helper 미완료 항목은 최신 live 보고서를 따른다.
