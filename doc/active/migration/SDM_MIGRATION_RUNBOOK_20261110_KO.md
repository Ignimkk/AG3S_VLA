# AMRC → SDM 서버 이전 runbook

작성: 2026-11-10 23:5x KST · 상태: **준비 완료, 아직 실행 전** · 대상 checkout: `/mnt/dev/AG3S_VLA` (`feature/panda-benchmark`)

AI-Stack(INFINITIX, KETI) 에서 이 pod를 **AMRC 서버 → SDM 서버**로 옮기는 전체 절차다.
SDM 에서 `git clone` 하는 것부터 환경 검증까지 순서대로 따라가면 된다.

| 스크립트 (`scripts/setup/migration/`) | 실행 위치 | 하는 일 |
|---|---|---|
| `amrc_manifest.sh` | AMRC | git·branch·worktree·submodule·`uv pip freeze`·apt 기록 → `/mnt/dev/_migration/manifest/<stamp>/` |
| `pull_from_amrc.sh <mode>` | SDM | SSH 로 AMRC 에서 당겨옴. mode = `probe` · `git` · `kit` · `tree` · `storage` · `all` |
| `bootstrap_sdm.sh` | SDM | apt 재설치 · `/root` 설정 복원 · cache link · GPU/venv/freeze/render 검사 |

---

## 1. 제약 (2026-11-10 확인)

| 항목 | 상태 | 결과 |
|---|---|---|
| AMRC NAS PVC 를 SDM 에 mount/복사 | 관리자 답변: **불가** | 데이터를 직접 전송해야 함 |
| 사용자 지정 이미지 서버 간 공유 | **불가** | 컨테이너 내부 설치물은 스크립트로 재설치 |
| AMRC 컨테이너 → SDM (172.21.121.123, 168.254.248.76) | `No route to host` | **SDM 쪽에서 AMRC 로 당겨오는(pull) 방식** |
| GitHub `Ignimkk/AG3S_VLA` | public (인증 없이 clone 가능) | push 는 GitHub 로그인 필요 |

## 2. 무엇이 어디에 있고 어떻게 옮기나

| 항목 | 위치 | 크기 | 방법 |
|---|---|---|---|
| tracked code | `AG3S_VLA` (git) | – | GitHub clone |
| **push 안 된 commit · local branch** (`feature/panda-benchmark` ahead 3, `wip/stage-track`, `worktree-agent-*`) | AMRC `.git` | – | `pull … git`: AMRC repo 를 SSH remote(`amrc`)로 fetch |
| submodule 5개 (LIBERO, LIBERO-Safety, SafeLIBERO, aloha, curobo) | `benchmark/`, `third_party/` | – | `git submodule update --init` (pinned commit) |
| 미커밋 수정 · untracked 파일 (다른 세션 WIP 포함) | working tree | 작음 | `pull … tree` (.git 제외 rsync) |
| venv 5개 + uv Python (`.python/`) | `AG3S_VLA/.venv-*`, `.python` | 22 GB | `pull … tree` — **그대로 복사** (§8 참조) |
| `.runtime/` (MagickWand, LIBERO config) · `.cache/` (paper_baselines 21 GB 등) · `outputs/` · `Grounded-SAM-2` · `papers/` | gitignored | ~26 GB | `pull … tree` (`.cache/uv` 15 GB 는 제외) |
| 체크포인트 · 데이터셋 · 결과 | `/mnt/dev/AG3S_VLA-storage` | **380 GB** (results 327 · benchmarks 30 · checkpoints 24) | `pull … storage` |
| `/root` 설정: Claude memory/session, codex, rclone gdrive token, uv, `.bashrc` | 컨테이너 내부 | 0.3 GB | `_migration/root_home/*.tar.gz` → `pull … kit` → bootstrap |
| Qwen2.5-VL-7B (`/root/.cache/ag3s_local`) | 컨테이너 내부 | 16 GB | `_migration/cache/` → `pull … kit` → bootstrap 이 symlink |
| apt 패키지 (osmesa, ffmpeg, texlive, rclone, rsync …) | 이미지 | – | bootstrap 이 재설치 |
| `/root/.cache/huggingface/xet` (10 GB), `.vscode-server` | 컨테이너 내부 | – | 버림 (자동 재생성) |

전송 총량 ≈ **450 GB**. 1 Gbps 면 약 1–1.5 시간, 100 Mbps 면 10 시간 이상.

## 3. 전체 흐름

```text
AMRC                                   SDM (새 컨테이너, NAS 볼륨 = /mnt/dev)
────                                   ─────────────────────────────────────
A. manifest + /root 백업        ──┐
                                  │    B. 컨테이너 생성
                                  │    C. probe → 스크립트 가져오기
                                  └──► D. pull git      (GitHub clone + AMRC branch fetch)
                                       E. pull kit → tree → storage   (1차, AMRC 실험 중에도 가능)
A'. 실험 정지 · 최종 백업 ───────────► E'. 같은 mode 재실행 (차이분만)
                                       F. bootstrap_sdm.sh
                                       G. validate + smoke + 같은 서버 기준 재측정
```

## 4. Phase A — AMRC 준비

1. **manifest 기록** (읽기 전용, 몇 분):
   ```bash
   cd /mnt/dev/AG3S_VLA && export PATH=/root/.local/bin:$PATH
   bash scripts/setup/migration/amrc_manifest.sh        # → /mnt/dev/_migration/manifest/latest
   ```
   2026-11-10 23:50 1차 기록 완료 (`20261110T235059`).
2. **`/root` 백업** (2026-11-10 21:23 1차 완료, 최종 전환 직전에 다시):
   ```bash
   cd /root && tar czf /mnt/dev/_migration/root_home/root_home_$(date +%Y%m%dT%H%M).tar.gz \
     --exclude=.vscode-server --exclude=.cache --exclude=.bash_history \
     .claude .claude.json .codex .config .ssh .local .libero .nv .bashrc .profile .zshrc .copilot
   chmod 600 /mnt/dev/_migration/root_home/*.tar.gz
   ```
3. **실험 정리:** 최종 전환(A') 전에 실행 중인 queue 를 step 경계에서 멈추고 commit 한다. 1차 전송(E)은 실험 중에도 할 수 있다.
4. GitHub push 는 **필수가 아니다**. push 안 된 commit 은 D 단계에서 AMRC 에서 직접 fetch 한다.

## 5. Phase B — SDM 컨테이너 생성 (포털)

| 설정 | 값 |
|---|---|
| GPU 유형 | **`H200-NVL(140GB)(공유)`** ("(공유)" 없는 것은 분산학습 클러스터용 — 사용 금지) |
| 이미지 | CUDA 12.8 · Ubuntu 24.04 계열 (AMRC: `CUDA_VERSION=12.8.1`, 시스템 PyTorch 없음, venv 가 자체 torch/jax 보유) |
| 볼륨 | `NAS_Storage` 볼륨을 **반드시 `/mnt/dev`** 에 mount (venv shebang·`pyvenv.cfg`·egg-link 가 `/mnt/dev/AG3S_VLA` 절대경로) |
| End Time | 해당 없음 |
| vCPU / RAM | 쿼터 내 최대 (AMRC: 32 vCPU / 96 GiB, 병목은 CPU) |

## 6. Phase C — 연결 확인 · 스크립트 가져오기 (SDM 터미널)

AMRC 포털 → [컨테이너 목록] → AMRC 컨테이너 → [설명] 탭의 SSH IP/포트를 쓴다.

```bash
export AMRC_HOST=<AMRC_IP> AMRC_PORT=<AMRC_SSH_PORT>
timeout 5 bash -c "echo > /dev/tcp/$AMRC_HOST/$AMRC_PORT" && echo OK || echo FAIL
```

- **FAIL** → §9 "SSH 전송이 안 될 때".
- **OK** → 스크립트 3개를 AMRC 에서 가져온다 (AMRC 비밀번호 1회):
  ```bash
  apt-get update -qq && apt-get install -y -qq rsync git openssh-client
  mkdir -p /root/migration && scp -P $AMRC_PORT \
    "root@$AMRC_HOST:/mnt/dev/AG3S_VLA/scripts/setup/migration/*.sh" /root/migration/
  ```

## 7. Phase D·E·F — git, 데이터, bootstrap (SDM 터미널)

```bash
cd /root/migration
bash pull_from_amrc.sh git       # D: GitHub clone → /mnt/dev/AG3S_VLA, AMRC branch fetch, HEAD 일치 확인, submodule, worktree
bash pull_from_amrc.sh kit       # E: /mnt/dev/_migration (16 GB)
bash pull_from_amrc.sh tree      # E: venv · .python · .runtime · .cache · outputs · untracked/미커밋 파일 (~54 GB)
bash pull_from_amrc.sh storage   # E: AG3S_VLA-storage (380 GB) — 가장 오래 걸림, tmux/screen 안에서
bash /mnt/dev/AG3S_VLA/scripts/setup/migration/bootstrap_sdm.sh     # F
```

- 처음 SSH 접속 때 SDM key 를 AMRC `authorized_keys` 에 넣는다 (비밀번호 1회). 이후는 key 로 접속.
- 끊기면 **같은 mode 를 다시 실행**하면 이어받는다 (`rsync --partial`). SDM 쪽 파일은 지우지 않는다.
- `git` mode 를 `tree` 뒤에 다시 돌려도 수정된 파일을 덮지 않는다 (tracked 수정이 있으면 checkout/reset 생략).
- **최종 전환(E'):** AMRC 실험을 멈추고 §4-2 백업을 다시 한 뒤 `kit` → `tree` → `storage` 를 한 번 더 (차이분만 전송).
- `git` 단계의 clone 은 GitHub 에서 받으므로 SDM 의 외부 인터넷이 필요하다. 없으면 `git clone ssh://root@$AMRC_HOST:$AMRC_PORT/mnt/dev/AG3S_VLA /mnt/dev/AG3S_VLA` 로 AMRC 에서 직접 clone 한다
  (submodule 도 upstream GitHub 이므로 그 경우 `tree` 결과의 submodule 폴더를 그대로 쓴다).

bootstrap 이 검사하는 것: `nvidia-smi` · venv 5개 interpreter · jax GPU · torch CUDA · MuJoCo osmesa render ·
**venv 별 `uv pip freeze` 가 AMRC manifest 와 동일한지** · Claude memory 복원. 마지막 줄이 `BOOTSTRAP OK` 여야 한다.

## 8. Phase G — 검증

```bash
cd /mnt/dev/AG3S_VLA && source scripts/setup/env.sh
.venv-ag3s/bin/python scripts/setup/validate_live_env.py    # GPU 가 비어 있어야 실행됨
```

1. `validate_live_env.py` 통과 (OpenPI loading, cuRobo kernel, GPU arithmetic).
2. LIBERO smoke 1 episode — AMRC 에서 같은 task/seed 로 돌린 run 과 성공 여부·궤적을 비교.
3. **수치 비교 규칙:** 서버가 바뀌면 XLA kernel 차이로 π0.5 action chunk 가 달라질 수 있다 (2026-11-04 컨테이너 교체 때 확인).
   SDM 에서 나온 수치는 **SDM 에서 다시 측정한 same-machine reference** 와만 비교한다. AMRC 수치와 섞을 때는 표에 서버를 표시한다.
4. 기록: `AG3S/ag3s/docs/AG3S_BENCHMARK_LOG.md` · `AG3S_BASELINE_LOG.md` 에 이전 날짜·서버 전환을 남긴다.

**venv 를 복사하는 이유:** 같은 절대경로(`/mnt/dev/AG3S_VLA`)로 옮기므로 그대로 동작하고, 패키지 버전이 AMRC 와 바이트 단위로 같다
(재설치하면 lock 밖의 transitive 버전이 달라질 수 있음). CLAUDE.md 의 "기존 workspace venv 를 복사하지 않는다" 는 **legacy `/mnt/dev/work`
venv 에 대한 규칙**이고, 이 checkout 자신의 venv 를 같은 경로로 옮기는 것에는 해당하지 않는다. 복사가 불가능할 때만 재생성한다:

| venv | 재생성 스크립트 |
|---|---|
| `.venv-ag3s` | `scripts/setup/create_cpu_env.sh` |
| `.venv-openpi-live` | `scripts/setup/create_live_env.sh` (절차 정본: `doc/active/migration/LIVE_ENVIRONMENT_IMPLEMENTATION_REPORT.md`) |
| `.venv-baselines` | `Baseline/paper_baselines/setup/setup_venv.sh` (`requirements-baselines.txt` lock, `external/Grounded-SAM-2` 필요) |
| `.venv-libero` | `scripts/setup/create_libero_env.sh` → `create_safelibero_config.sh` |
| `.venv-libero-safety` | `scripts/setup/create_libero_safety_env.sh` (MagickWand 를 `.runtime/magickwand` 에 추출) |

재생성 뒤에는 `uv pip freeze` 를 `_migration/manifest/latest/freeze_*.txt` 와 diff 해서 차이를 기록한다.

## 9. SSH 전송이 안 될 때

위에서부터 시도한다. 어느 경로든 끝나면 F(bootstrap)·G(검증)는 같다.

| # | 경로 | 조건 | 속도 / 비고 |
|---|---|---|---|
| F1 | **PC 경유 스트리밍** | PC 가 AMRC·SDM 컨테이너 SSH 에 모두 접속 가능 | PC 네트워크 속도. 디스크 공간 불필요 |
| F2 | **PC/외장 디스크 2단계** | 동시 접속은 안 되지만 각각은 접속 가능 | PC 디스크 ~450 GB 필요. SDM 쪽은 포털 [저장소 관리] → [데이터 전송 정보] 컨테이너로 업로드 |
| F3 | **인터넷 경유 직접 전송** (`croc` 등 relay 방식) | AMRC·SDM 둘 다 외부 인터넷 outbound 가능 | 양쪽 inbound 불필요, end-to-end 암호화. **기관 보안 규정 확인 후** 사용 |
| F4 | **Google Drive (rclone)** | AMRC 에 `gdrive:` 이미 설정됨, SDM 에서 외부 인터넷 | 계정당 업로드 750 GB/일 · Drive 용량 필요 · 느림. tar 묶음 단위로 |
| F5 | **줄여서 옮기기** | 어느 경로든 느릴 때 같이 적용 | 아래 참조 |

**F1 예시** (PC 터미널, 두 컨테이너 SSH 설정이 `amrc`, `sdm` 로 되어 있을 때):
```bash
# checkout 을 .git·worktree·submodule 포함 통째로 옮긴다 (같은 /mnt/dev 경로라 worktree 링크도 유지).
# 디렉터리별로 나눠 실행하면 실패 시 재시도 단위가 작다. storage 는 하위 디렉터리별로 더 나눠도 된다.
for d in _migration AG3S_VLA AG3S_VLA-storage; do
  ssh amrc "tar cf - --exclude=AG3S_VLA/.cache/uv --exclude=__pycache__ -C /mnt/dev $d" | ssh sdm "tar xf - -C /mnt/dev"
done
# SDM 에서 확인: cd /mnt/dev/AG3S_VLA && git status && git worktree list && git log -1
```
F1·F2 는 `.git` 까지 옮기므로 D(clone/fetch) 단계가 필요 없다. 바로 F(bootstrap)로 간다.
tar stream 은 이어받기가 안 되므로, 끊기면 그 디렉터리만 다시 보낸다.

**F3 예시:** AMRC 에서 `croc send --zip /mnt/dev/_migration` → 출력되는 code 를 SDM 에서 `croc <code>`.
storage 는 하위 디렉터리별로 나눠 보낸다.

**F5 줄이기:**
- `.cache/uv` (15 GB), `/root/.cache/huggingface/xet` (10 GB) 는 이미 제외.
- 공개 가중치는 SDM 에서 다시 받는다: `checkpoints/panda/pi05_libero` 는 `scripts/setup/fetch_pi05_libero.py` (GCS crc32c/md5 + SHA-256 검증),
  Qwen2.5-VL-7B·baseline 가중치는 HuggingFace.
- `results/` 327 GB 는 먼저 내용을 분류한다 (최종 수치·표·figure 근거 run 우선, 재생성 가능한 중간 `*.npz`·`policy_records` 는 후순위).
  분류 결과에 따라 우선 전송 목록을 만든다.
- venv 는 §8 표로 재생성 (단, freeze diff 기록 필수).

## 10. 문제 해결

| 증상 | 원인 / 조치 |
|---|---|
| venv python 이 `No such file` | 볼륨이 `/mnt/dev` 가 아닌 곳에 mount 됨 → 컨테이너를 `/mnt/dev` mount 로 다시 생성 |
| `HEAD mismatch` (git mode) | AMRC 에서 그 사이 commit 됨 → `git` mode 재실행 |
| LIBERO-Safety client 가 MagickWand 오류 | `source scripts/setup/libero_safety_env.sh` 로 실행 (LD_LIBRARY_PATH 를 `.runtime/magickwand/lib` 로 교체) |
| MuJoCo EGL 오류 | 정상. 프로젝트 기본 `MUJOCO_GL=osmesa` (AMRC 에서도 EGL 은 실패) |
| `rsync` 가 NFS 에서 느림 | 정상 범위. storage 는 tmux 안에서 돌리고 같은 mode 재실행으로 이어받기 |
| baseline queue 가 Qwen 경로 못 찾음 | `/root/.cache/ag3s_local` symlink 확인 (bootstrap 3/5) |

## 11. 기록

| 시각 (KST) | 내용 |
|---|---|
| 2026-11-10 21:23 | `/root` 설정 백업 (`root_home_20261110T2123.tar.gz`, 280 MB), Qwen2.5-VL cache 16 GB 복사 → `/mnt/dev/_migration/` |
| 2026-11-10 21:3x | AMRC 에 `rsync` 설치. SDM 포털 주소 `No route to host` 확인 |
| 2026-11-10 23:50 | `amrc_manifest.sh` 1차 실행 (`manifest/20261110T235059`), venv 5개 freeze 기록 (현재 venv 와 동일 확인) |
| 2026-11-10 23:5x | 이 runbook 과 `scripts/setup/migration/` 3개 작성. 스크립트는 문법 검사만 했고 SDM 에서 실행 전 |
