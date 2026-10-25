# AG3S_VLA — migration 완료 후 시작 규칙

현재 작업 경로는 이 파일이 속한 독립 checkout이다. `git rev-parse --show-toplevel`,
`git branch --show-current`, `git rev-parse HEAD`, `git remote -v`를 먼저 확인한다.
`doc/active/BRANCHES.md`와 `doc/active/migration/source-manifest.json`에서 branch 목적·source 기준을 읽는다.
main은 paired v2 기반 V11AB, feature/panda-benchmark는 그 benchmark 평가,
feature/ag3s-development는 v2 이후 전체 개선 이력과 현재 WIP를 포함한다.
개선 branch 전체를 benchmark branch에 merge하면 V11AB 평가 대상이 바뀐다.
기존 /mnt/dev/work의 Claude session은 새 코드 작업에 resume하지 말고 새 세션을 시작한다.
공유 input은 별도 AG3S_VLA-storage에 있다. 기존 /mnt/dev/work는 현재 실행에 필요하지 않은 이전 checkout이다.
새 코드는 자기 checkout에서 수정한다. 새 output은 별도 storage의 results/development/<날짜_commit_run-id>/에 생성한다. source scripts/setup/env.sh로 경로를 설정한다.
아래에 남은 과거 진행 기록은 source-manifest와 실제 branch를 기준으로 해석한다.
Panda adapter와 benchmark rollout 구현·평가는 아직 시작하지 않았다.

# AG3S / trajopt 검토 — 작업 규칙

## 프로젝트 목적

본 프로젝트의 목적은 **AG3S에서 VLA의 visual attention을 활용하여 작업 대상(Target)과 장애물(Obstacle)을 구분하고, Target을 충돌 대상에서 제외한 3D 환경 정보를 cuRobo의 TSDF/ESDF 파이프라인으로 전달하는 전체 시스템을 구축하는 것**이다.

> **규칙 본문의 정본은 `.claude/ag3s-rules.md` 다.** UserPromptSubmit hook이 매 턴 주입하므로
> 규칙은 대화 내내 살아 있다.
> 이 파일은 세션 시작에 필요한 프로젝트 목적, 진행 내용, 필수 규칙, sub-agent 역할 및 환경 정보를 담는다.

## 진행내용 파악

기록은 두 파일로 나뉜다. **최신 → 오래된 순**으로 읽는다 (2026-10-12 사용자 지시로 개편).

| 순서 | 파일 | 상태 | 범위 · 언제 읽나 |
|---|---|---|---|
| 1 | `AG3S/ag3s/docs/AG3S_T47_LOG.md` | **최신 · 이어 쓰는 정본** | **T47 이후 모든 STEP** (attention head 재선택 T47·T47b, HSR counterfactual T48a·T48b, 명사 token query T49 …). 맨 위 표 (문서 상태 · 용어 · 진행 현황 · 물려받은 결정 · 재측정 대기) 를 먼저 읽는다 |
| 2 | `AG3S/ag3s/docs/AG3S_T0T6_LOG.md` | **이전 정본 · T46 까지만** | 16D 모델 위의 T0→T46 실행 기록. 이미 진행 중이던 T46 (V-T46 · CL3) 은 여기서 마무리하고, **T47 번호부터는 새로 쓰지 않는다.** T46 이전 결정 · 발견 ID 의 출처를 찾을 때 읽는다 |

- T46 (V-T46 · CL3) 를 제외한 2026-10-12 이후의 모든 새 STEP — T46 의 후속 접미사 STEP (`T46b`, `T46c` …) 포함 — 은 `AG3S_T47_LOG.md` 에 쓴다. 새 STEP **번호** 는 쓰기 전에 다른 세션에 알려 예약한다 (번호 충돌 방지).
- 새 STEP 의 판단 근거가 옛 로그에 있으면 그 줄을 **링크**하고 내용을 새 로그에 다시 옮겨 적지 않는다.
- 옛 로그의 수치와 새 로그의 수치가 다르면 **새 로그가 우선**한다 (모델 · 배선 · 기준이 바뀌었을 수 있다 — 그 차이를 새 로그에 적는다).
- `AG3S/ag3s/docs/AG3S_CUROBO_LIVE_TEST_PLAN.md` — T0~T6 실행 계획과 gate별 판정 항목 (이전 국면 계획).
- `doc/active/reviews/AG3S_TWO_PAPER_REVIEW_20261012_KO.md` — T47 이후 연구 방향 (Kang et al. localization head · Bhardwaj et al. progress probe 와의 비교, 실험 순서와 채택 기준).

## 반드시 지켜야 할 규칙

- **규칙 A — 시각화:** 구현 및 검증 등 수행한 작업은 실제 scene 또는 시각화할 수 있는 범위에서
  figure로 생성한다. 작업 내용을 기록할 때는 표와 graph를 사용한다.
- **규칙 B — 로그:** 한 번의 구현 및 검증 등 작업한 내용에 대해 날짜와 시간을 포함하여
  핵심 내용을 설명하듯 기록한다.
- **규칙 C — 용어:** 기술적인 용어는 English로 작성하여 문장의 자연스러움을 유지한다.

## Sub-agents

Sub-agent의 상세 역할과 작업 범위는 `.claude/agents/`의 정의를 따른다.

- **`ag3s-implementer`** — AG3S/trajopt code 구현과 unit test를 담당하고 구현 결과를 `impl.md`로 전달한다.
- **`ag3s-verifier`** — 구현 결과를 독립적으로 검증하고 raw output, 수치, figure 및 `verify.json`을 생성한다.
- **`ag3s-scribe`** — 검증된 결과를 바탕으로 log, 계획서 및 paper를 작성한다.

새로운 session을 시작하면 `.claude/agents/`를 확인하고, 작업 성격에 맞는 sub-agent를 사용하여
구현, 검증, 기록을 수행한다. 각 sub-agent는 자신에게 정의된 file ownership과 작업 범위를 지킨다.

## 현재 환경

이 development checkout의 VS Code 주 interpreter는 `.venv-openpi-live/bin/python`이다.
Python 3.11.16 / OpenPI 0.1.0 / JAX·JAXlib 0.5.3 / Torch 2.7.1을 독립 재설치했다.
OpenPI와 openpi-client는 자기 `third_party/openpi` source에 editable install한다.
CPU 회귀는 기존 독립 `.venv-ag3s`(Python 3.11.16, numpy 2.4.6)를 유지한다.
cuRobo producer는 별도 `.venv-curobo`(Python 3.11.16, numpy 1.26.4, Torch 2.7.1,
CUDA 12.6, cuda-core 1.2.0, cuRobo 0.8.0.post1.dev43)를 사용하며 system packages를 공유하지 않는다.
Benchmark checkout의 GPU 환경 설치 상태는 이 문서의 development 결과로 추정하지 않는다.

`source scripts/setup/env.sh` 후 목적에 맞는 venv를 activate한다. Checkpoint·dataset·과거 기록은
`AG3S_STORAGE_ROOT`에서 읽고 기존 input은 수정하지 않는다. `AG3S_LEGACY_STORAGE`는 파생 input alias다.
새 video·policy record·log·training checkpoint는 `$AG3S_STORAGE_ROOT/results/development`의
KST 날짜·commit·UUID run 아래로 저장한다. 서버와 client가 같은 run을 사용할 때는
`export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"`를 한 번 실행한다.
OpenPI `--record`는 run의 `policy_records/session_*/step_N.npy`에 쓰며 과거 recording link를 사용하지 않는다.

GPU 작업은 모든 checkout이 `/tmp/ag3s-vla-gpu-validation.lock`을 `scripts/setup/gpu_lock.py`로 잡고 시작한다
(사용자 판정 2026-10-25, 두 단계). **독점**(`exclusive`): 결과가 벽시계 시간에 의존하거나 시간을 재는 작업 —
TrajOpt가 들어가는 closed-loop(serve_safe safe mode, `time_budget_ms`가 강제 상한), 지연 시간·runtime 측정.
**공유**(`shared`): 결과가 벽시계와 무관한 작업 — VLA 단독(`--no-safe`), offline 추출·replay, 시간을 보고하지 않는 smoke.
공유 작업은 GPU 사용량 + 예상치 ≤ 120 GiB이면 함께 돈다. 줄 서 있는 독점 요청 뒤의 공유 요청은 기다린다.
예: `python3 scripts/setup/gpu_lock.py exclusive --est-gib 30 --tag 실험이름 -- bash launch.sh`, 상태는 `gpu_lock.py status`.
`OPENPI_DATA_HOME`·CUDA/Warp/JAX/Torch cache는 자기 checkout 아래다.
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_FLAGS=--xla_gpu_enable_command_buffer=`를 유지한다.

현재 storage·interpreter 설정의 정본은 `doc/active/SHARED_STORAGE_AND_PYTHON.md`이며 설치·검증·삭제 조건은
`doc/active/migration/GPU_OUTPUT_IMPLEMENTATION_REPORT.md`를 읽는다. Legacy 정리의 최신 판정은
`doc/active/migration/LEGACY_RETIREMENT_REPORT.md`를 따른다. 이전 보고서는 그 시점 기록으로 보존한다.
AG3S algorithm과 기존 개선 이력은 유지한다. `/mnt/dev/work`는 삭제하지 않았고 process를 종료하지 않는다.
원본 Git·잔여 자료의 permanent archive, 역사적 sibling link 전환 및 clean 임시 worktree 정리는
retirement 보고서의 검증 결과를 따른다. 실제 삭제 직전 live process 참조를 다시 확인한다.
