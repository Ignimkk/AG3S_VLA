# SDM GPU 서버 새로 세팅하기 — apt · git · venv · weight · 검증

작성: 2026-11-11 17:2x KST (dev-4a) · 상태: **작성만 함, SDM 에서 아직 실행 안 함** · 대상: 새 SDM 컨테이너, checkout `/mnt/dev/AG3S_VLA` (`feature/panda-benchmark`)

AMRC 데이터 전송 (PC 중계, 11-12 08:00 이후 SDM 업로드) 을 기다리지 않고, **GitHub 와 공개 출처만으로** SDM 에서 실험을 바로 시작하기 위한 절차다.
위에서부터 순서대로 실행한다. 각 단계 끝의 '확인' 이 통과해야 다음 단계로 간다.

## 0. 기존 문서와의 관계

| 문서 | 다루는 경로 | 이 문서와의 차이 |
|---|---|---|
| [`SDM_MIGRATION_RUNBOOK_20261110_KO.md`](SDM_MIGRATION_RUNBOOK_20261110_KO.md) (SDM이전방법 세션) + `scripts/setup/migration/bootstrap_sdm.sh` | AMRC 의 checkout · venv · storage 를 **통째로 복사**한 뒤 bootstrap | 이 문서는 복사 없이 **새로 설치**한다. apt 목록은 bootstrap 과 같다 |
| [`../NEXT_SERVER_HANDOFF_20261109_KO.md`](../NEXT_SERVER_HANDOFF_20261109_KO.md) §3 | 새 서버 체크리스트 (11-09, 요약) | 이 문서가 그 체크리스트의 실행 명령판이다 |
| [`../SHARED_STORAGE_AND_PYTHON.md`](../SHARED_STORAGE_AND_PYTHON.md), [`LIVE_ENVIRONMENT_IMPLEMENTATION_REPORT.md`](LIVE_ENVIRONMENT_IMPLEMENTATION_REPORT.md) | venv · storage 설계와 live 검증 정본 | 상세 근거는 그쪽을 따른다 |

**나중에 AMRC 복사본이 올라올 때 주의:** 이 문서로 만든 checkout 위에 AMRC checkout 을 그대로 rsync 하면 `.git` 과 venv site-packages 가 섞인다.
그래서 `pc_relay.sh upload` 에 guard 가 있다 (SDM이전방법 세션, 11-11 17:3x). SDM 에 `/mnt/dev/AG3S_VLA` 가 이미 있으면 업로드를 멈추고 `EXISTING` 값을 요구한다.
- `EXISTING=replace`: 이 문서로 만든 checkout 을 `AG3S_VLA.fresh-<stamp>` 로 옮기고 AMRC checkout 을 올린다.
- `EXISTING=archive`: 이 checkout 을 유지하고, AMRC 것은 `/mnt/dev/AG3S_VLA_amrc` 에 참고용으로만 둔다.

어느 쪽을 쓸지는 사용자가 정한다. storage 업로드에는 `--delete` 가 없어서, 여기서 받은 checkpoint · asset 은 지워지지 않는다.

## 1. 컨테이너 확인 (포털 설정은 runbook §5)

```bash
df -h /mnt/dev                     # NAS 볼륨이 정확히 /mnt/dev 에 mount (queue script · venv 가 이 절대경로를 씀)
nvidia-smi --query-gpu=name,memory.total --format=csv
cat /etc/os-release | head -2; echo "CUDA_VERSION=$CUDA_VERSION"; ls -d /usr/local/cuda*; which nvcc
nvidia-smi | head -4                # driver version · 지원 CUDA
# 외부 접속: 실제로 받는 URL 로 확인한다 (서버 root 주소는 정상이어도 400/403 이 나온다 — 11-11 SDM 실측)
for u in https://github.com/NVlabs/curobo \
         "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_libero/&maxResults=1" \
         https://download.pytorch.org/whl/cu113/torch/ \
         https://pypi.org/simple/numpy/ https://astral.sh/uv/install.sh \
         https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/cc594898137f460bfe9f0759e9844b3ce807cfb5/config.json \
         https://github.com/ultralytics/assets/releases; do
  curl -sL -m 20 -o /dev/null -w "%{http_code} $u\n" "$u"; done
```
확인: GPU 가 보이고, 위 URL 이 모두 200 이다.
`nvcc` 가 없으면 cuRobo · SAM2 kernel build 가 안 되므로 CUDA devel 이미지로 다시 만든다.

**실측 (2026-11-11 17:4x, 사용자가 SDM 에서 실행):**

| 항목 | AMRC | SDM | 영향 |
|---|---|---|---|
| OS | Ubuntu 24.04.1 | **Ubuntu 22.04.5** | apt 이름은 같다. MagickWand `.deb` 는 22.04 판 (6.9.11) 이 받아진다 — §5-5 에서 확인 |
| CUDA toolkit (`nvcc`) | 12.8.1 | **12.6.2** (`/usr/local/cuda` → `cuda-12.6`) | `setup_venv.sh` 의 기본 `CUDA_HOME=/usr/local/cuda-12.6` 이 그대로 맞는다. jax · torch wheel 은 CUDA runtime 을 자체로 가지므로 driver 버전만 중요하다 |
| GPU | H200 NVL 143 GB | **H200 NVL 70 GB** (공유 할당) | AMRC 의 GPU 한도 규칙 (≤ 130 GB) 은 여기서 70 GB 다. 동시에 두 server 를 띄우지 않는다 |
| `/mnt/dev` | NAS | NAS 18 TB (여유 5.9 TB) | 충분 |
| storage.googleapis.com · download.pytorch.org | – | root 주소 HEAD 가 400 · 403 | 정상 동작 (root 경로는 원래 이렇게 응답). 위의 실제 URL 로 다시 확인 |
| GPU driver | 570.211.01 (CUDA 12.8) | **570.211.01 (CUDA 12.8)** | 같다 → jax · torch GPU wheel 이 AMRC 와 같은 조건 |
| 외부 접속 (실제 URL 7 개) | – | **모두 200** (17:44) | GitHub · GCS · PyTorch index · PyPI · uv · HF · ultralytics release 모두 됨 |
| 시간대 | KST (`/etc/timezone` = Asia/Seoul) | **UTC** (`nvidia-smi` 08:44 = 17:44 KST) | 기록 · queue log 는 KST 기준 → §2 에서 Asia/Seoul 로 맞춘다 |

## 2. apt (AMRC 설치 기록과 같은 목록, `bootstrap_sdm.sh` 1/5 와 동일)

```bash
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y tzdata && ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && echo Asia/Seoul > /etc/timezone && date   # KST 로 (AMRC 와 같게)
apt-get install -y git git-lfs curl ca-certificates build-essential cmake pkg-config \
  libosmesa6 libgl1 libglib2.0-0 libegl1 libglfw3 ffmpeg rclone rsync openssh-server poppler-utils tmux unzip
# 논문 PDF 빌드가 필요할 때만 (~1 GB)
apt-get install -y texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra
```
- MuJoCo 는 `MUJOCO_GL=osmesa` 로 돈다 (`env.sh` 기본값). AMRC 에서 EGL 은 실패했다.
- LIBERO-Safety 의 MagickWand 는 apt 로 설치하지 않는다. §5-5 의 script 가 `.deb` 를 받아 `.runtime/magickwand` 에 푼다.

## 3. uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # → /root/.local/bin/uv
export PATH=/root/.local/bin:$PATH
echo 'export PATH=/root/.local/bin:$PATH; export AG3S_UV=/root/.local/bin/uv' >> /root/.bashrc
export AG3S_UV=/root/.local/bin/uv
```
setup script 의 기본 uv 경로는 `$AG3S_STORAGE_ROOT/tools/uv` 다. 이 경로는 새 서버에 없으므로 `AG3S_UV` 를 반드시 설정한다.

## 4. git clone · submodule

```bash
mkdir -p /mnt/dev/AG3S_VLA-storage && cd /mnt/dev
git clone -b feature/panda-benchmark https://github.com/Ignimkk/AG3S_VLA.git AG3S_VLA
cd AG3S_VLA && git submodule update --init --recursive
git log --oneline -1 && git submodule status
```
- submodule 5 개는 pin 된 commit 으로 받는다.

  | submodule | 경로 |
  |---|---|
  | LIBERO | `benchmark/LIBERO` |
  | SafeLIBERO (vlsa-aegis) | `benchmark/SafeLIBERO` |
  | LIBERO-Safety | `benchmark/LIBERO-Safety` |
  | aloha | `third_party/aloha` |
  | curobo | `third_party/curobo` |

  OpenPI fork 는 submodule 이 아니라 `third_party/openpi/` 로 직접 추적된다.
- GitHub 에는 **push 된 commit 까지만** 있다. 11-11 17:2x 기준 push 는 끝나 있다. AMRC 에서 그 뒤에 한 commit 은 다시 push 하거나, 나중에 AMRC 복사본으로 받는다.
- git identity 는 저장소 설정에 넣지 않는다. commit 할 때마다 `-c user.name=kmingi98 -c user.email=kmingi98@gmail.com` 을 붙인다 (`.claude/skills/git-commit/SKILL.md`).

## 5. venv 6 개 (모두 `/mnt/dev/AG3S_VLA/.venv-*`, Python 은 `.python/` 에 uv 가 설치)

`source scripts/setup/env.sh` 뒤 차례로 실행한다. 각 script 는 다시 실행해도 안전하다.

| # | venv | 명령 | 용도 · 비고 |
|---|---|---|---|
| 5-1 | `.venv-ag3s` (3.11.16) | `bash scripts/setup/create_cpu_env.sh` | CPU 분석 · test · launcher. 끝에 `check_workspace.py` |
| 5-2 | `.venv-openpi-live` (3.11.16) | `bash scripts/setup/create_live_env.sh` | π0.5 policy server (jax GPU) |
| 5-3 | `.venv-curobo` (3.11.16) | `bash scripts/setup/create_live_env.sh .venv-curobo` | cuRobo (첫 사용 때 kernel JIT build, `nvcc` 필요) |
| 5-4 | `.venv-libero` (3.8.20) | `bash scripts/setup/create_libero_env.sh && bash scripts/setup/create_safelibero_config.sh` | LIBERO · SafeLIBERO client. `.runtime/libero`, `.runtime/safelibero` config 생성 |
| 5-5 | `.venv-libero-safety` (3.10) | `bash scripts/setup/create_libero_safety_env.sh` | LIBERO-Safety client + MagickWand. **§6-3 asset 을 받은 뒤** 한 번 더 실행해야 asset symlink 가 생긴다 |
| 5-6 | `.venv-baselines` (3.11) | `CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 bash Baseline/paper_baselines/setup/setup_venv.sh` | 논문 baseline (AEGIS · multi-link CBF · KNOWS). Grounded-SAM-2 source 를 받고 SAM2 CUDA op 를 build 한다. SDM 은 CUDA 12.6 이라 script 기본값 `CUDA_HOME=/usr/local/cuda-12.6` 과 같다. 다른 이미지에서는 `/usr/local/cuda` 를 가리키게 한다 |

확인 (AMRC 와 버전 비교는 선택. AMRC manifest 가 SDM 에 있을 때만 가능):
```bash
for v in .venv-*; do printf '%-22s ' $v; $v/bin/python -V; done
.venv-openpi-live/bin/python -c 'import jax; print(jax.devices())'                         # cuda device
.venv-baselines/bin/python -c 'import torch; print(torch.cuda.get_device_name(0))'
MUJOCO_GL=osmesa .venv-libero/bin/python -c 'import mujoco; r=mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"),32,32); r.render(); print("osmesa OK")'
# (선택) diff <(uv pip freeze --python .venv-ag3s/bin/python) /mnt/dev/_migration/manifest/latest/freeze_.venv-ag3s.txt
```

## 6. checkpoint · asset · weight (공개 출처, revision 고정)

| # | 무엇 | 위치 | 크기 | 받는 법 |
|---|---|---|---|---|
| 6-1 | π0.5 `pi05_libero` | `$AG3S_STORAGE_ROOT/checkpoints/panda/pi05_libero` | ~12 GB | `.venv-openpi-live/bin/python -I scripts/setup/fetch_pi05_libero.py $AG3S_STORAGE_ROOT/checkpoints/panda/pi05_libero` (GCS `openpi-assets`, crc32c · md5 · SHA-256 검증) |
| 6-2 | π0.5 `pi05_libero_safety` | `…/checkpoints/panda/pi05_libero_safety` | 12.44 GB | HF `LIBERO-Safety/pi05_libero_safety` @`e66d84bfec17770384550593e39975f7e21e3463` (아래 명령) |
| 6-3 | LIBERO-Safety assets | `$AG3S_STORAGE_ROOT/benchmarks/libero-safety/assets_94ccca77/assets` | zip 10.7 GB → 15.5 GB | HF dataset `LIBERO-Safety/libero_safety_assets` @`94ccca773d1bab760d1aecec4f62c8d4b8797d40` `assets.zip` (sha256 `31a81123…7800`) → unzip → `create_libero_safety_env.sh` 재실행 |
| 6-4 | baseline weight (Grounding DINO · SAM · SAM2.1 · YOLOE · MobileCLIP) | `.cache/paper_baselines/` | ~6 GB | `Baseline/paper_baselines/weights.py` 의 `fetch()` (HF revision · release tag 고정, sha256 기록) |
| 6-5 | Qwen2.5-VL-7B-Instruct (AEGIS · multi-link 의 VLM naming) | `/root/.cache/ag3s_local/Qwen2.5-VL-7B-Instruct` | 16 GB | HF `Qwen/Qwen2.5-VL-7B-Instruct` @`cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| – | OpenPI tokenizer 등 | `.cache/openpi/big_vision` | 작음 | policy server 첫 실행 때 자동으로 받음 |

```bash
cd /mnt/dev/AG3S_VLA && source scripts/setup/env.sh
S=$AG3S_STORAGE_ROOT; HFPY=.venv-baselines/bin/python
# 6-2
$HFPY -c "from huggingface_hub import snapshot_download as d; d('LIBERO-Safety/pi05_libero_safety', revision='e66d84bfec17770384550593e39975f7e21e3463', local_dir='$S/checkpoints/panda/pi05_libero_safety')"
# 6-3
$HFPY -c "from huggingface_hub import hf_hub_download as d; print(d('LIBERO-Safety/libero_safety_assets', 'assets.zip', repo_type='dataset', revision='94ccca773d1bab760d1aecec4f62c8d4b8797d40', local_dir='$S/benchmarks/libero-safety/assets_zip_94ccca77'))"
sha256sum $S/benchmarks/libero-safety/assets_zip_94ccca77/assets.zip      # 31a81123c936f0ca3411a4e37c344e9b3dffe2800f1c1a2580841fda3aed7800
mkdir -p $S/benchmarks/libero-safety/assets_94ccca77 && unzip -q $S/benchmarks/libero-safety/assets_zip_94ccca77/assets.zip -d $S/benchmarks/libero-safety/assets_94ccca77
ls $S/benchmarks/libero-safety/assets_94ccca77/assets | head      # 'assets/' 아래에 풀렸는지 확인 (아니면 폴더 이름을 맞춘다)
bash scripts/setup/create_libero_safety_env.sh                    # asset symlink
# 6-4
source Baseline/paper_baselines/setup/paper_baselines_env.sh
for w in grounding-dino-base grounding-dino-tiny sam-vit-huge sam2.1-hiera-large yoloe-11l-seg mobileclip_blt; do
  $HFPY -c "import sys; from Baseline.paper_baselines.weights import fetch; print(fetch(sys.argv[1]))" $w; done
# 6-5
$HFPY -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-VL-7B-Instruct', revision='cc594898137f460bfe9f0759e9844b3ce807cfb5', local_dir='/root/.cache/ag3s_local/Qwen2.5-VL-7B-Instruct')"
```
- 6-2 · 6-3 의 원래 기록 (file 수 · aggregate sha256): `AG3S/ag3s/docs/handoff/BM-SETUP-B.impl.md` 119–121 행. 받은 뒤 같은 형식의 `CHECKSUMS.json` 을 남긴다.
- LIBERO · SafeLIBERO 평가에는 dataset 이 필요 없다. init state 와 BDDL 은 submodule 안에 있다.

## 7. 검증

```bash
cd /mnt/dev/AG3S_VLA && source scripts/setup/env.sh
nvidia-smi --query-compute-apps=pid,process_name --format=csv     # 비어 있어야 함
.venv-ag3s/bin/python scripts/setup/validate_live_env.py           # OpenPI loading · cuRobo kernel · GPU 연산
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-ag3s/bin/python -m pytest -q \
  tests/ag3s/test_sl_pe8_held_motion.py tests/ag3s/test_sl_pe9_attention_peak.py tests/ag3s/test_sl_pe11_target_cut.py \
  tests/ag3s/test_sl_pe_wiring.py tests/ag3s/test_sl_pe1_close_seed.py tests/trajopt/test_sl_qe2_translation_only.py
```

이어서 benchmark 마다 짧은 smoke 를 1 개씩 돌린다. 모든 GPU run 은 launcher 가 `/tmp/ag3s-vla-gpu-validation.lock` 을 직접 잡으므로 바깥에서 `flock` 으로 감싸지 않는다.

| smoke | 확인하는 것 | 예시 |
|---|---|---|
| LIBERO π0.5 단독 1 task | policy server + `.venv-libero` client | `benchmark/runners/launch_libero_paired.py` (지금 구성은 `doc/active/experiments/AG3S_FINAL_CONFIG_KO.md`) |
| SafeLIBERO baseline 1 block | `.venv-baselines` sidecar + Qwen + weight | `Baseline/paper_baselines/closed_loop/launch_method.py --method aegis --task-suite-name safelibero_spatial --safety-level I --task-ids 0 --episode-ids 15 --seed 7 --run-id sdm-smoke-aegis` |
| LIBERO-Safety 1 ep | MagickWand · asset · `pi05_libero_safety` | `INITS=19 DRY=1 bash experiments/segmentation/bl_tranche_libero_safety_queue.sh` 로 명령을 확인한 뒤 1 개만 실행 |

## 8. 그 밖

- **Claude 설정 · memory:** `/root/.claude` 는 새 컨테이너에 없다.
  - AMRC 의 `_migration/root_home/*.tar.gz` 가 오면 `bootstrap_sdm.sh` 2/5 단계로 복원한다.
  - 그 전에는 새 세션이 `CLAUDE.md` · 이 문서 · `doc/active/experiments/AG3S_PLAN_SCHEDULE_20261110_KO.md` 를 먼저 읽는다.
- **GPU 공유 규칙 · 단계 commit 규칙:** AMRC 와 같다 (`CLAUDE.md`, 계획서 §2 의 '규칙' 줄).
- **결과 위치:** 새 run 은 SDM 의 `/mnt/dev/AG3S_VLA-storage/results/benchmark/` 에 쌓인다.
  - AMRC 결과는 relay upload 로 **같은 `results/benchmark/` 에 그대로 합쳐진다** (`--delete` 없음). 계획서들이 가리키는 경로가 그대로 유지된다. 별도 `results/old_server/` 는 두지 않는다 (사용자 결정 11-12 01:5x, runbook §0-2).
  - 그래서 **SDM 에서 만드는 run-id 는 모두 `-sdm` 을 붙인다.** queue script 는 `SUFFIX=-sdm`, 직접 launch 는 `--run-id …-sdm`. 이유: `bl_tranche_*_queue.sh` 의 `done_run` 은 같은 run-id 의 끝난 run 이 있으면 건너뛰므로, AMRC run 과 run-id 가 같으면 SDM run 이 실행되지 않는다. LIBERO-Safety launcher 처럼 run-id 뒤에 `-<suite>-L<lvl>-…` 를 덧붙이는 경우에는 `-sdm` 이 이름 중간에 온다.
- **코드 · 문서는 GitHub 로만 옮긴다.** relay upload 는 `EXISTING=archive` → AMRC checkout 은 `/mnt/dev/AG3S_VLA_amrc` (참고용, 그 venv 는 경로 때문에 못 씀). SDM checkout 은 `git pull` 로만 갱신하고, AMRC 에만 있던 commit 은 `git fetch /mnt/dev/AG3S_VLA_amrc <branch>` 로 받을 수 있다.

## 9. 기록

| 시각 (KST) | 내용 |
|---|---|
| 2026-11-11 17:2x | 작성 (dev-4a). AMRC 의 실제 경로 · script · revision 을 읽어서 썼다: storage `CHECKSUMS.json`, `weights.py`, Qwen hub ref, `.gitmodules`, manifest `apt_history.txt`. SDM 에서는 아직 실행하지 않았다 |
| 2026-11-12 01:4x | 사용자가 SDM 에서 §1–§5 실행 완료 (dev-4a 가 출력 확인: HEAD af3cc897 = origin, submodule 5 개 pin, venv 6 개 Python 버전 AMRC 와 같음, jax `CudaDevice(id=0)` · torch `NVIDIA H200 NVL` · osmesa OK). §6 · §7 출력은 받지 않음 |
| 2026-11-12 02:0x | §8 결과 위치 수정: `results/old_server/` 대신 같은 `results/benchmark/` 에 합치고 SDM run-id 에 `-sdm` (사용자 결정 11-12 01:5x, SDM이전방법 전달) |
