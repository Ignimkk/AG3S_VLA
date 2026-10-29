---
paths:
  - "paper/**"
---

# CVPR 2027 논문 작업 범위

## 먼저 읽을 문서

1. `${AG3S_PROJECT_ROOT}/paper/README.md`
2. `${AG3S_PROJECT_ROOT}/paper/SETUP_PLAN.md`
3. `${AG3S_PROJECT_ROOT}/paper/guidelines/CVPR_2027_STATUS.md`와 `SOURCES.md`
4. `paper/evidence/CLAIMS.md`, `EXPERIMENTS.md`, `RESULTS.md`, `METRICS.md`
5. `paper/notes/PAPER_STORY.md`와 `paper/experiments/manifests/README.md`
6. `${AG3S_PROJECT_ROOT}/paper/paper_structure_setup.md`
7. `paper/notes/CLAUDE_WORKFLOW.md`: skill 선택·Evidence tracking·검토 역할 소유권
8. 해당 Step의 나머지 정책·Evidence·연구 계획 문서

한 번에 1 Step만 수행하고, 완료 후 수정 이유와 검증 결과를 보고한다.
Root `${AG3S_PROJECT_ROOT}/CLAUDE.md`는 사용자가 복원한 기존 내용을 수정하지 않는다.
논문 작업의 진입점은 `paper/CLAUDE.md`이며 이 파일만 공백·줄바꿈을 포함해 200자 이내로 유지한다.
상세 지침은 이 rule과 논문 문서에 둔다. Claude 설정은 `${AG3S_PROJECT_ROOT}/.claude/`에서
관리하며 `paper/.claude/`를 중복 생성하지 않는다.

## 작업 경계

- 실제 논문 작성은 Claude Code가 담당한다. 준비 담당 assistant는 규칙·환경·입력 안내를 구성한다.
  사용자 별도 요청 없이 후속 연구 분석·원고 작성으로 준비 범위를 확대하지 않는다.
- 필수 환경 준비 Step 1–6은 완료됐다. 인계는 `paper/notes/CLAUDE_HANDOFF.md`를 따른다.
  미완료 연구·실험·최종 제출 항목을 추가 환경 준비 Step으로 자동 전환하지 않는다.
- 선행연구는 사용자 후보 목록을 기다린다. 기존 R1과 원고 citation은 탐색적 참고 후보·임시 초안이다.

- 기존 연구 소스와 `outputs/`, `tests/`는 읽기 전용이다. 기존 코드를 복사하거나 수정하지 않는다.
- 논문 분석·plotting·LaTeX용 코드는 `paper/tools/`에 작성한다.
- 논문 준비 기록은 `paper/SETUP_PLAN.md`, `paper/notes/SETUP_CHANGES.md`에서 관리한다.
  `paper/evidence/PAPER_LOG.md`가 생성되면 논문 작성·evidence 처리 기록을 그곳에 남긴다.
- 기존 AG3S 실행 로그, figure 저장 규칙, 구현·검증 역할은 연구 구현 작업에 적용한다.
  논문 환경 구성만을 위해 연구 실행 로그를 변경하지 않는다.
- 계획·가설·측정·파생·추정·placeholder를 서로 구분해 표시한다.

## Evidence와 정책

- 논문에는 등록·검증 여부와 관계없이 측정값, 파생값, 추정값, 계획값과 placeholder를 사용할 수 있다.
  각 항목에 자료 유형, provenance, 검증 상태, dimension과 scope를 표시하고 파일명만으로 설정을 추정하지 않는다.
- `HYPOTHESIS`와 `CANDIDATE`는 그 상태를 표시한다. Missing metric은 0이 아니다.
  Task outcome과 collision·safe success를 구분하고, 유의하지 않은 차이로 성능 유지를 주장하지 않는다.
- Source identity나 metric/config가 달라지면 그 차이를 표시하고 필요하면 re-audit한다.
  기존 결과를 다른 snapshot·task·geometry 또는 real robot 범위로 해석할 때 그 전환을 명시한다.
- 수치와 주장은 research log, verify.json, 계산식, 가정 또는 작성 근거까지 추적할 수 있어야 한다.
  `evidence/` 문서는 출처와 자료 유형을 보존하며, 표시 없이 자료 성격을 바꾸지 않는다.
- citation, DOI, baseline, 통계적 유의성, 구현 세부 사항을 확인 없이 생성하지 않는다.
- CVPR 규정과 일정은 공식 출처로 확인하고 `CONFIRMED_2027`, `PROVISIONAL_2026`,
  `UNKNOWN`으로 구분한다. 원문에 적힌 일정과 Author Kit 상태도 검증 대상이다.
- 공식 규정의 현재 적용 상태는 `paper/guidelines/`가 기준이다. 준비 원문의 예시보다
  최신 검증 자료를 우선한다. `PROJECT_POLICY`는 conference 규정과 구분한다.
- AoE→KST 계획용 계산을 공식 시각으로 보고하지 않는다. 실제 제출 timestamp를 재확인한다.
- LLM의 author assistance와 학회 공식 peer review를 구분한다.
- 공식 template 원본을 보존하고 `cvpr.sty`나 margin·font size를 임의로 수정하지 않는다.
- 공개 제출용 자료는 확인된 anonymity 정책을 따르며 hidden reviewer instruction을 삽입하지 않는다.
- 아직 생성되지 않은 문서·skill·audit 결과를 존재하거나 완료된 것처럼 보고하지 않는다.

영문 manuscript를 작성하되 사용자 설명과 작업 기록은 자연스러운 한국어로 작성한다.
Technical term은 English로 쓰고, 내부 발견 번호나 약어는 의미를 함께 설명한다.

## 원고 용어 규칙 (사용자 지시, 2026-10-06)

- 범용적으로 쓰이는 기술 용어를 사용한다. 우리가 새로 만드는 용어는 최소화한다.
- 예: querier-dependent exemption → link-specific allowed collision, payload → attached object,
  authorised links → contact links, admissibility → candidate filtering, fail-closed → fail-safe,
  phase probe → sub-task probe, Target(고유명사처럼 쓰기) → target object.
- 새 용어가 꼭 필요하면 처음 나올 때 범용 용어로 정의하고, 같은 대상에는 한 용어만 쓴다.

## Claude 작성·검토 운영

CVPR skill 7개와 검토 역할 3개는 `paper/notes/CLAUDE_WORKFLOW.md`를 따른다.
검토 역할은 읽기 전용으로 보고하고 parent writer가 필요한 논문 파일 변경을 반영한다.
Source 결과 사용 전에는 manifest validator를 현재 실행한다.
최종 제출 준비 판정 전에 `/cvpr-evidence full`, `/cvpr-review full`,
`/cvpr-compliance full`의 실제 검사 범위·미해결 사항을 확인한다.
정의 생성이나 준비 문서 구성만으로 원고·PDF 검토 완료를 선언하지 않는다.
