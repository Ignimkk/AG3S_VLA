# AG3S_VLA — migration 완료 후 시작 규칙

현재 작업 경로는 이 파일이 속한 독립 checkout이다. `git rev-parse --show-toplevel`,
`git branch --show-current`, `git rev-parse HEAD`, `git remote -v`를 먼저 확인한다.
`doc/BRANCHES.md`와 `doc/migration/source-manifest.json`에서 branch 목적·source 기준을 읽는다.
main은 paired v2 기반 V11AB, feature/panda-benchmark는 그 benchmark 평가,
feature/ag3s-development는 v2 이후 전체 개선 이력과 현재 WIP를 포함한다.
개선 branch 전체를 benchmark branch에 merge하면 V11AB 평가 대상이 바뀐다.
기존 /mnt/dev/work의 Claude session은 새 코드 작업에 resume하지 말고 새 세션을 시작한다.
공유 input은 별도 AG3S_VLA-storage에 있다. 기존 /mnt/dev/work는 현재 실행에 필요하지 않은 이전 checkout이다.
새 코드·새 output은 자기 checkout에서만 수정/생성한다. source scripts/setup/env.sh로 경로를 설정한다.
아래에 남은 과거 진행 기록은 source-manifest와 실제 branch를 기준으로 해석한다.
Panda adapter와 benchmark rollout 구현·평가는 아직 시작하지 않았다.

# AG3S / trajopt 검토 — 작업 규칙

## 프로젝트 목적

본 프로젝트의 목적은 **AG3S에서 VLA의 visual attention을 활용하여 작업 대상(Target)과 장애물(Obstacle)을 구분하고, Target을 충돌 대상에서 제외한 3D 환경 정보를 cuRobo의 TSDF/ESDF 파이프라인으로 전달하는 전체 시스템을 구축하는 것**이다.

> **규칙 본문의 정본은 `.claude/ag3s-rules.md` 다.** UserPromptSubmit hook이 매 턴 주입하므로
> 규칙은 대화 내내 살아 있다.
> 이 파일은 세션 시작에 필요한 프로젝트 목적, 진행 내용, 필수 규칙, sub-agent 역할 및 환경 정보를 담는다.

## 진행내용 파악

1. `AG3S/ag3s/docs/AG3S_T0T6_LOG.md` — **정본.** 16D 모델 위의 T0→T6 실행 기록.
   맨 위의 **용어 · 진행 현황 · 물려받은 결정 · 재측정 대기** 네 표를 먼저 읽으면 현재 위치를 알 수 있다.
2. `AG3S/ag3s/docs/AG3S_CUROBO_LIVE_TEST_PLAN.md` — T0~T6 실행 계획과 gate별 판정 항목.

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

각 checkout은 별도 `.python/`·`.venv-ag3s`·output·cache를 사용한다.
`source scripts/setup/env.sh` 후 `.venv-ag3s/bin/python`으로 CPU 검증을 실행한다.
GPU/OpenPI·cuRobo 환경과 로컬 PC 환경은 아직 이번 구조로 설치하지 않았다.
기존 workspace의 venv를 복사하거나 새 Python code의 import 경로로 사용하지 않는다.
Checkpoint·dataset·녹화 자료는 `AG3S_STORAGE_ROOT`(기본: checkout 옆 `AG3S_VLA-storage`)에서 읽는다.
`AG3S_LEGACY_STORAGE`는 그 아래 `legacy-inputs`를 가리키는 호환 alias다.
`doc/SHARED_STORAGE_AND_PYTHON.md`가 현재 storage·interpreter 설정의 정본이다.
기존 “OpenPI submodule 내부의 venv를 쓴다”는 지침은 현재 통합 구조에 적용하지 않는다.
OpenPI fork는 `third_party/openpi/`에 직접 추적된다.

GPU 실행 시 `OPENPI_DATA_HOME`은 자기 checkout의 `.cache/openpi`,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_FLAGS=--xla_gpu_enable_command_buffer=`를 사용한다.
실제 VLA 환경·체크포인트·robot/camera adapter 검증은 다음 단계다.
Migration 검증과 인수인계는 `doc/migration/IMPLEMENTATION_REPORT.md`에 기록한다.
기존 단계 로그는 그 로그의 source 기준과 실제 branch가 일치할 때만 이어 쓴다.
