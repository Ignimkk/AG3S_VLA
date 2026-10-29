# Development GPU environment와 output 분리 결과

기록: **2026-10-12 00:48 KST**. 목적: **layout / environment / execution output**.
작업 checkout `/mnt/dev/AG3S_VLA-development`, branch `feature/ag3s-development`.
시작 HEAD `4292882ef3054a16c67c03d3fd6261ef8644a7e7`, origin `https://github.com/Ignimkk/AG3S_VLA.git`.
최종 commit은 `git log -1`로 확인한다. Run의 commit은 검증 당시 base이며 변경 중인 code는
[validated-code-state.json](../../archive/migration/validation/gpu-environment/validated-code-state.json)의 SHA-256으로 식별한다.
main·benchmark·원본 legacy refs/submodule pin은 수정하지 않는다. AG3S algorithm과 기존 개선 이력을 유지한다.

## 완료된 변경

| 영역 | 변경 |
|---|---|
| OpenPI | 자기 Python 3.11.16으로 `.venv-openpi-live` 신규 생성, 기존 distribution 버전 재설치 |
| Editable source | workspace/OpenPI/client/cuRobo 모두 자기 checkout source; 이전 venv 복사 없음 |
| cuRobo | 독립 `.venv-curobo`, Python 3.11.16, system-site-packages 없이 별도 producer 구성 |
| VS Code / Claude | 주 interpreter `.venv-openpi-live`, CPU/producer 역할과 새 output 경로를 안내 |
| Run output | `common.run_output`: KST 날짜·commit·UUID run, `AG3S_RUN_DIR`로 server/client grouping |
| Policy recording | 상대 `policy_records` 제거; run 아래 unique session + exclusive `step_N.npy` creation |
| CLI output | Bringup/safe server의 video·record·trajectory·constraints·evaluation 경로를 run 안으로 제한 |
| Training | 새 checkpoint/log/W&B local output을 run 아래로 설정; tmux child에서도 own `env.sh` 재설정 |
| Norm stats | 별도 `--output-dir`, 기본 run/assets; 기존 shared assets를 output으로 쓰지 않음 |
| Input link | Development의 `third_party/openpi/policy_records` link 제거; 과거 입력의 실제 위치는 유지 |
| 문서 | 이전 `/mnt/dev/work/doc`의 최신 v1.4 관리 문서 원본·hash 보존, 현재 운영 guide 갱신 |
| Legacy 잔여 자료 | 필요한 잔여 script·PPTX·과거 build evidence와 원본 전체 Git bundle 보존 |

`common.run_output`은 run 밖의 절대경로, `..` 및 input을 향하는 기존 symlink를 거부한다.
Run 이름 예: `20261012T002906+0900_4292882_46c9b2ae`.
OpenPI 서버 재시작은 같은 run을 재사용해도 별도 `policy_records/session_*`를 생성한다.
한 recorder의 concurrent calls는 lock으로 step을 할당하고 파일을 exclusive create한다.
Training resume는 기존 development run을 명시한다. Shared checkpoint는 policy/weight input으로만 읽는다.
Norm stats를 새로 만들면 training에 해당 run/assets를 `--assets-base-dir`로 전달한다.
기존 launcher의 과거 norm log는 read-only readiness input이며 새로 append하지 않는다.

현재 경로·사용법 정본: [SHARED_STORAGE_AND_PYTHON.md](../SHARED_STORAGE_AND_PYTHON.md).

## 설치 정보와 재생성

| 환경 | Python | 핵심 package | 검증 역할 |
|---|---|---|---|
| `.venv-openpi-live` | 3.11.16 | OpenPI/client 0.1.0, JAX/JAXlib 0.5.3, Torch 2.7.1+cu126, numpy 1.26.4, MuJoCo 3.11.0 | H200 policy inference |
| `.venv-curobo` | 3.11.16 | cuRobo 0.8.0.post1.dev43, Torch 2.7.1+cu126, cuda-core 1.2.0, Warp 1.17.0, numpy 1.26.4 | Depth → TSDF/ESDF GPU producer |
| `.venv-ag3s` | 3.11.16 | numpy 2.4.6, 기존 CPU dependencies | 기존 CPU 회귀·producer 결과 소비 |

이전 live 환경을 metadata로 읽고 기존 `environments/legacy-python` snapshot과 함께 비교했다.
cuRobo를 별도 환경으로 옮기고 자체 workspace metadata를 제외한 **279 distribution 버전**이 일치한다.
[기존 직접 inventory](../../archive/migration/validation/gpu-environment/legacy-live-inventory.json),
[현재 OpenPI inventory](../../archive/migration/validation/gpu-environment/development-openpi-inventory.json),
[현재 cuRobo inventory](../../archive/migration/validation/gpu-environment/development-curobo-inventory.json)에 source URL까지 보존했다.
LeRobot Git pin `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`를 재설치했다.
OpenPI Torch transformer replacement 5개는 자기 vendored source와 installed 파일 SHA-256이 일치한다.

이전 cuRobo 환경은 Python 3.10.12 + system-site-packages였다. 새 producer는 현재 pinned
source `78fd485fa82d9b9a063fb4985e371814587e666a`의 `requires-python >=3.10`을 기준으로
Python 3.11.16과 CUDA 12.6 Torch를 사용해 실제 kernel 실행을 검증했다.
Metadata version은 기존 `0.8.0.post1.dev43`를 재현하며 SCM version override와 source pin을 함께 기록한다.
CUDA toolkit wheel 12.9 metadata와 Torch의 CUDA runtime 12.6을 같은 버전으로 표현하지 않는다.
Driver는 580.173.02, GPU는 NVIDIA H200 NVL이다.
Compatibility guidance: [JAX installation](https://docs.jax.dev/en/latest/installation.html),
[cuRobo runtime configuration](https://nvlabs.github.io/curobo/latest/reference/runtime_configuration.html).
최신 upstream 일반 설명보다 pinned source 및 실제 실행 결과를 설치 판정 근거로 사용한다.

```bash
source scripts/setup/env.sh
bash scripts/setup/create_gpu_env.sh --with-curobo
```

`create_gpu_env.sh`는 자기 `.python`의 Python을 쓰고 wheel을 새로 설치한다.
OpenPI 명세는 `environments/openpi-live/requirements.txt`이며 observed versions를 `--no-deps`로 재현한다.
Producer의 initial resolution은 `requirements-live.txt` + observed constraints로 수행했고,
재생성은 검증된 전체 `environments/curobo/resolved.txt`를 사용한다. Source는 자기 checkout에만 editable 설치한다.
공유 wheel/download cache는 source·venv 복사와 구분하며 installed files는 `--link-mode copy`로 독립적이다.
재생성 script의 기본 cache는 자기 `.cache/uv`다.

**남은 package metadata 제한:** OpenPI `uv pip check`에는 `gym-aloha`의 `mujoco <3`와
기존 `mujoco 3.11.0` 사이 제약 충돌 1건이 있다. 이전 live 환경에서도 동일하게 재현했다.
요청된 버전은 유지했고 RBY1 inference는 통과했다. ALOHA 전체 실행 호환성을 보장하지 않는다.
cuRobo `uv pip check`는 모든 installed packages compatible이다.

## 검증 결과

| 측정 | 결과 | 근거 |
|---|---|---|
| Own import origin | OpenPI/client/AG3S/common/cuRobo가 모두 자기 checkout | `execution-verification.json` |
| JAX / Torch GPU | H200에서 두 matrix 연산 sum 각각 3680 | 같은 JSON |
| 기존 checkpoint | 16D LoRA step 29999 실제 model 복원 | OpenPI GPU raw JSON |
| 실제 policy inference | 2회 모두 finite `(50,16)` action | OpenPI GPU JSON / action figure |
| Recorder | 3개 session; actual policy record 2개 + concurrent dummy record 8개; 첫 기록 보존 | OpenPI GPU JSON |
| cuRobo depth/ESDF | surface voxel 3,528개; `128³=2,097,152` ESDF 값 모두 finite | cuRobo GPU JSON |
| ESDF 재사용 buffer | `detach().clone()`한 첫 결과가 두 번째 호출 후 그대로 유지 | cuRobo GPU JSON |
| GPU 직렬 실행 | OpenPI process 종료 → cuRobo process 시작, 각각 공용 flock + empty compute process precheck | raw GPU log / verify manifest |
| CPU output/record timing | 독립 최종 suite 43 passed | `final-final-cpu-junit.xml` |
| 선택한 기존 SafePolicy 회귀 | 구현 담당 suite 88 passed (일부 output test 중복) | handoff implementation |
| 이전 source 접근 차단 | CPU `run_without_legacy` record timing 23 passed | `no-legacy-record-timing.log` |
| Input artifact | 35 input root, 8 artifact SHA check 정상; 이전 checkout symlink 0 | `check-storage-openpi.json` |
| 과거 policy record | 114개, 52,225,743 B의 SHA/inode/size/mtime 동일; 변경·추가 0 | historical record manifests |
| 기존 checkpoint/input | checkpoint 28개 파일 metadata 및 observation/meta SHA 동일 | OpenPI GPU JSON |
| Claude / shell | 실제 interpreter/output 안내 정상, shell syntax와 whitespace 검사 정상 | `claude-context.json` |

처음 checkpoint restore는 약 38.4초, 첫 inference는 compile을 포함한다. 이 값으로 steady latency를 주장하지 않는다.
두 stochastic inference의 action 값이 완전히 동일하다는 수치 재현성도 주장하지 않는다.
Producer 검증은 synthetic 16×16 depth plane의 compatibility smoke test다.
실제 정책 입력 image는 과거 recording이며 새 closed-loop rollout이나 physical robot/camera 실험을 실행한 것은 아니다.
Training 자체와 신규 학습 checkpoint 생성은 실행하지 않았다. Checkpoint의 writable destination과 침범 차단은 unit test로 검증했다.

검증 run:
`/mnt/dev/AG3S_VLA-storage/results/development/20261012T002906+0900_4292882_46c9b2ae/`.
Raw GPU scripts/log/JSON과 action·ESDF 배열은 `verification/`, 새 MP4는 `videos/`, 기록은 `policy_records/`에 있다.
새 영상은 기존 input frame 15개의 시각화이며 source recording은 변경하지 않았다.
반복 검증에서는 새 `AG3S_RUN_DIR`을 생성하고 `verification/openpi_gpu.py` 다음에 `verification/curobo_gpu.py`를
각 venv에서 순서대로 실행한다. Raw measurement scripts는 당시 checkout/input 경로를 명시한 보존 artifact다.
GPU가 사용 중이거나 공용 lock을 얻지 못하면 시작하지 않는다.

Tracked 요약: [ENV_OUTPUT.verify.json](../../../AG3S/ag3s/docs/handoff/ENV_OUTPUT.verify.json),
[execution-verification.json](../../archive/migration/validation/gpu-environment/execution-verification.json),
[legacy-audit.json](../../archive/migration/validation/gpu-environment/legacy-audit.json).
Figure: [실제 policy 입력](../../../AG3S/ag3s/docs/figures/env-output/recorded-policy-input.png),
[action chunk](../../../AG3S/ag3s/docs/figures/env-output/checkpoint-action-chunks.png),
[ESDF](../../../AG3S/ag3s/docs/figures/env-output/curobo-synthetic-esdf.png),
[검증 표](../../../AG3S/ag3s/docs/figures/env-output/gpu-validation-table.png).
연구 로그와 review plan에는 factual infrastructure 결과만 append했다.

## `/mnt/dev/work` 삭제 가능 여부

아래는 이 보고서 작성 시점의 판정이다. 이후 승인된 archive·link·임시 worktree 정리와
최신 live process 상태는 [LEGACY_RETIREMENT_REPORT.md](LEGACY_RETIREMENT_REPORT.md)를 따른다.

**현재 삭제 불가. 새 development 실행은 독립적으로 검증됐지만 외부 사용 흔적이 남아 있다.**
삭제·process 종료·기존 input 변경·다른 checkout 수정은 수행하지 않았다.

| 점검 대상 | 현재 결과 / 필요한 후속 조건 |
|---|---|
| 최신 관리 문서 | v1.4 정본을 canonical `doc/archive/PROJECT_FOLDER_AND_GIT_MANAGEMENT_FINAL.md`에 보존; 원본 3개와 hashes는 `legacy-management-v1.4`에도 보존 |
| 과거 source archive | storage `archive/source-snapshot-20261011`의 8,972개 SHA 정상; 변경된 운영 문서 2개는 추가 보존 |
| 이전 폴더에만 있던 필요한 자료 | patch_dataset_timestamps.py, PPTX 2개, notices/config, 과거 paper build, cuRobo metadata 등 40개를 run/legacy-preservation에 보존; source/destination SHA 오류 0 |
| 원본 Git object | 새 checkout에 없던 auxiliary object 2개를 `work-all-refs.bundle`로 보존, complete bundle verify 정상 |
| Live legacy dependency | VS Code/Pylance/Python process 4개가 이전 cwd 또는 executable 사용; 해당 사용자가 종료/새 창으로 전환한 뒤 재점검 필요 |
| Historical sibling link | `/mnt/dev/work-*` 22개 폴더의 symlink 202개가 이전 source/assets 참조; 각각 frozen source identity를 보존해 이관하거나 사용자 판단으로 해당 archive를 retire한 뒤 재점검 필요 |
| 등록된 Git worktree | 이전 benchmark repository의 detached worktree 1개가 `/mnt/dev/work/benchmark/.git`에 의존; 먼저 별도 보존/정리 필요 |
| 기존 venv | 새 환경 재설치/실행 검증 완료; 기존 환경 자체를 복사해 runtime dependency로 유지하지 않음 |
| 과거 installer/cache | reproducible wheels·ruff cache·generated egg-info는 새 runtime에 필요 없음; 필요한 metadata는 보존 |
| 외부 namespace | 다른 사용자·container의 mount/open file 전체까지 보장한 audit는 아님; 실제 삭제 직전에 현재 사용자 범위에서 재점검 필요 |

등록된 worktree 경로:
`/tmp/claude-0/-mnt-dev-work/cafbdbf4-2a7d-43e4-9303-6ce2f6880a96/scratchpad/pristine8f/benchmark`.
Process와 link 상세는 run/legacy-audit의 `live-process-paths.json`와 `symlinks.json`, `sibling-work-paths.json`,
`legacy-git-refs.json` 및 tracked legacy audit에 있다. Process command-line의 인증 token은 저장하지 않았다.
Codex auxiliary ref 이름은 작업 중 자동으로 갱신될 수 있으므로 이름의 정지 snapshot보다
누락된 object SHA 두 개가 bundle에 보존된 사실을 판정 근거로 사용했다.

보존 manifest: [legacy-preserved-files.json](../../archive/migration/validation/gpu-environment/legacy-preserved-files.json).
Bundle 위치: 위 run의 `legacy-preservation/work-all-refs.bundle` (146,605,138 B).
이 보존 자료와 shared input storage를 유지한 채 위 dependency 조건을 해소한 후 삭제 여부를 다시 판단한다.
실제 삭제는 이번 작업 범위에서 수행하지 않는다. Commit은 현재 branch에만 생성하며 push는 사용자가 수행한다.
