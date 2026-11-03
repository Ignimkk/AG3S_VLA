# 새 GPU 컨테이너에서 Panda 벤치마크 시작하기

기준: `feature/panda-benchmark`, 2026-11-03. LIBERO / SafeLIBERO의 π0.5 및 AG3S 실행에 필요한 준비를 다룬다. 새 이미지에서 전체 설치 및 rollout을 검증한 기록은 아니므로 아래 smoke test를 먼저 실행한다.

## 이미지 선택

제공된 이미지 중 **`cuda:12.8.1-devel-ubuntu24.04`**를 추천한다. CUDA 개발 도구를 포함하고, 이 저장소는 별도 가상환경에 Python 및 PyTorch/JAX를 설치하기 때문에 이미지에 선탑재된 PyTorch를 이용하지 않는다. 특히 선택적 SAM2 CUDA 확장 빌드에 `nvcc`가 필요하다. 크기는 9.21 GB다. CUDA 버전이 같다는 것만으로 호환이 보장되지는 않으며 GPU 모델, 호스트 NVIDIA 드라이버 및 Python wheel과의 조합을 아래에서 확인해야 한다.

`pytorch:2.4.1-cuda12.4-cudnn9-runtime`은 2.99 GB로 가볍지만 runtime 이미지이므로 CUDA 컴파일러를 확인해야 한다. `pytorch:24.10-py3` / `24.12-py3`는 크기가 약 18 GB이고 선탑재 패키지를 그대로 쓰지 않는다. Python, vLLM, Ollama, Open WebUI 이미지는 이 작업의 기본 선택으로 필요한 CUDA 개발 환경을 확인할 수 없다. 이미지에 표시된 Python 버전은 아래 환경의 Python 버전과 무관하다.

Ubuntu 24.04에서 선택 사항인 LIBERO-Safety 설치 스크립트가 요구하는 ImageMagick 6 패키지 이름을 apt 저장소가 제공하지 않을 수 있다. 해당 벤치마크가 우선이면 패키지 가용성을 확인하거나 호환되는 Ubuntu 이미지를 별도로 준비한다. 아래 LIBERO / SafeLIBERO 경로와는 별개다.

## 1. 클론과 시스템 준비

GPU가 컨테이너에 할당되어 있고 인터넷, 충분한 디스크 및 지속 저장 볼륨이 있다고 가정한다. `nvidia-smi`가 보이지 않으면 플랫폼의 GPU 할당/드라이버 설정부터 해결한다. `nvcc` 버전은 호스트 드라이버 버전과 다르다.

```bash
nvidia-smi
nvcc --version
# Ubuntu, root 기준. root가 아니면 관리자가 설치하거나 sudo 사용.
apt-get update
apt-get install -y git git-lfs curl ca-certificates build-essential cmake pkg-config libosmesa6 libgl1 libglib2.0-0 libegl1 libglfw3 ffmpeg

git clone --branch feature/panda-benchmark --recurse-submodules https://github.com/Ignimkk/AG3S_VLA.git
cd AG3S_VLA
git submodule update --init --recursive
git submodule status
git rev-parse HEAD
```

원격 브랜치에 필요한 커밋이 push되어 있어야 새 컨테이너에서 보인다. 재현하려면 실험에 사용할 커밋을 명시적으로 checkout한 뒤 submodule을 다시 갱신한다. OpenPI는 `third_party/openpi`에 수정본이 포함되어 있으므로 별도 upstream clone이나 `uv sync`로 교체하지 않는다. 모델, 데이터 및 기존 결과는 Git에 포함되지 않는다.

## 2. 저장 공간과 uv

컨테이너가 삭제되어도 남을 볼륨을 준비한다. 아래 경로는 예시이며 두 터미널에서 동일하게 지정한다. 캐시와 가상환경은 checkout 내부에 생기므로 컨테이너 재생성 후 유지하려면 checkout도 지속 볼륨에 둔다.

```bash
export AG3S_STORAGE_ROOT=/persistent/AG3S_VLA-storage
mkdir -p "$AG3S_STORAGE_ROOT"
curl -LsSf https://astral.sh/uv/install.sh | sh
# 새 셸에서는 uv 설치 경로를 PATH에 추가한다.
export PATH="$HOME/.local/bin:$PATH"
export AG3S_UV="$(command -v uv)"
test -x "$AG3S_UV"
source scripts/setup/env.sh
```

각 새 셸에서 `AG3S_STORAGE_ROOT`, `AG3S_UV`를 설정하고 `source scripts/setup/env.sh`를 실행한다. 결과 기본 위치는 `$AG3S_STORAGE_ROOT/results/benchmark`, 모델 위치는 `$AG3S_STORAGE_ROOT/checkpoints`다. `env.sh`는 `PYTHONPATH`, OpenPI 캐시, MuJoCo 렌더링(`osmesa`) 등을 설정한다.

## 3. Python 환경 설치

```bash
# 계획/launcher: Python 3.11.16
bash scripts/setup/create_cpu_env.sh
# OpenPI + JAX + Torch + cuRobo 서버: Python 3.11.16
bash scripts/setup/create_live_env.sh
# LIBERO/SafeLIBERO 클라이언트: Python 3.8.20
bash scripts/setup/create_libero_env.sh
bash scripts/setup/create_safelibero_config.sh
```

설치 시 uv가 Python과 wheel을 다운로드하고 LeRobot의 고정 커밋을 가져온다. 버전 기준은 각각 `environments/ag3s/requirements.txt`, `environments/testing/requirements.txt`, `environments/openpi-live/requirements.txt`와 그 inventory/validated constraints, OpenPI의 `examples/libero/requirements.txt` 및 `benchmark/LIBERO/requirements.txt`다. 서버는 Torch 2.7.1, JAX 0.5.3 기준이며 시스템 PyTorch를 공유하지 않는다. LIBERO 설치 스크립트는 현재 해석 결과를 `environments/libero/validated-freeze.txt`에 기록하므로 변경 여부를 확인한다. 원래 검증된 서버의 `gym-aloha`와 MuJoCo 메타데이터 충돌은 알려진 예외다. 설치 실패나 다른 `pip check` 충돌은 별도 확인한다.

LIBERO와 SafeLIBERO는 모두 `libero` 패키지 이름을 사용한다. SafeLIBERO runner는 클라이언트 실행 시 fork 경로와 별도 config를 선택한다. 두 fork를 한 환경에 각각 editable 설치하지 않는다. 개발 환경용 `create_gpu_env.sh`보다 이 브랜치에서 검증된 `create_live_env.sh`를 사용한다. 별도 cuRobo producer가 필요할 때만 `bash scripts/setup/create_live_env.sh .venv-curobo`를 추가한다.

## 4. 체크포인트

Panda 기본 체크포인트는 `$AG3S_STORAGE_ROOT/checkpoints/panda/pi05_libero`다. 기존에 검증한 동일 체크포인트와 norm stats가 있으면 해당 디렉터리 전체를 복사하거나 읽기 가능한 볼륨에 연결한다. 없으면 공개 체크포인트를 받는다. 다운로드는 수 GB이며 기존 목적지를 덮어쓰지 않는다. 중단 시 `.partial`을 재사용한다.

```bash
.venv-openpi-live/bin/python -I scripts/setup/fetch_pi05_libero.py \
  "$AG3S_STORAGE_ROOT/checkpoints/panda/pi05_libero"
```

스크립트는 크기와 CRC32C 등을 검사하고 `CHECKSUMS.json`을 남긴다. 다른 체크포인트는 launcher의 `--checkpoint-dir`로 지정한다. RB-Y1 전용 체크포인트나 실험별 probe/데이터는 별도로 확보해야 하며 공개 Panda 모델과 혼용하지 않는다.

## 5. 검증과 첫 실행

다른 GPU 작업이 없는지 `nvidia-smi`로 확인한다. 다음 검증의 GPU 모드는 유휴 GPU가 아니면 중단한다. 설치 결과와 CUDA 연산을 검사하며 첫 실행에는 컴파일 시간이 든다.

```bash
source scripts/setup/env.sh
.venv-openpi-live/bin/python scripts/setup/validate_live_env.py imports
.venv-openpi-live/bin/python scripts/setup/validate_live_env.py gpu
.venv-openpi-live/bin/python scripts/setup/validate_live_env.py curobo
.venv-libero/bin/python -c 'import libero.libero, openpi_client; print("LIBERO client OK")'
```

`validate_live_env.py checkpoint`는 Panda가 아니라 RB-Y1 전용 로컬 체크포인트를 요구하므로 위 Panda setup의 검증으로 사용하지 않는다. 우선 단일 LIBERO episode의 VLA-only를 실행한다. 실제 GPU와 체크포인트를 사용하고 결과를 저장한다.

```bash
.venv-ag3s/bin/python benchmark/runners/launch_libero_paired.py \
  --task-suite-name libero_spatial --task-ids 0 --episode-ids 0 \
  --run-id new-gpu-smoke --modes vla_only --conversion C
```

성공하면 SafeLIBERO의 짧은 paired smoke를 실행할 수 있다. 기존 실험과 비교할 때 suite, level, seed, step budget, conversion, 코드 및 checkpoint를 맞춘다.

```bash
.venv-ag3s/bin/python benchmark/runners/launch_libero_paired.py \
  --task-suite-name safelibero_spatial --safety-level I \
  --task-ids 0 --episode-ids 0 --data-split dev --run-id new-gpu-safe-smoke \
  --modes vla_only,vla_v11ab --ag3s-version V11AB-pb7 --conversion C
```

대규모 평가와 GPU 병렬 실행은 [평가 recipe](EVAL_RECIPE.md)를 따른다. GPU 메모리와 다른 프로세스를 먼저 확인한다. 실험 결과가 맞지 않으면 `nvidia-smi`, 서버/클라이언트 로그, checkpoint 경로, `LIBERO_CONFIG_PATH`, `MUJOCO_GL`, 드라이버와 라이브러리 버전을 확인한다. 새 이미지에서 smoke 통과 전에는 재현되었다고 가정하지 않는다.

## 선택 사항

- **LIBERO-Safety**: 별도 Python 3.10 환경 및 약 10.67 GB 압축/15.5 GB 압축 해제 asset이 필요하다. `scripts/setup/create_libero_safety_env.sh`와 `scripts/setup/libero_safety_env.sh`, `BM-SETUP-B` 문서를 확인한다. 위 paired launcher는 LIBERO-Safety용이 아니다.
- **논문 비교 baseline**: `Baseline/paper_baselines/setup/setup_venv.sh`는 별도 환경, 외부 소스 및 모델 가중치를 사용한다. SAM2 빌드의 기본 CUDA 경로/아키텍처는 기존 H200 환경 기준이므로 새 GPU에 맞게 `CUDA_HOME`, `TORCH_CUDA_ARCH_LIST`, `NVCC_APPEND_FLAGS`를 확인한다. 상세 내용은 `Baseline/paper_baselines/README.md` 참조.
- **추가 정보**: `environments/README.md`, `doc/active/migration/LIVE_ENVIRONMENT_IMPLEMENTATION_REPORT.md`, `third_party/openpi/README.md`.
