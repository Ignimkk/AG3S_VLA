# CVPR 2027 VLA Paper Workspace

이곳은 Research → Evidence → Experiment → Figure/Table → Manuscript → Review →
Submission 과정을 관리하는 논문 작업 공간이다. 현재 **Step 1–6: 논문 작성 workspace 구성**을 모두 마쳤다.

실제 원고 작성은 **Claude Code**가 담당한다. 준비 담당 assistant는 규칙·환경·사전 작업만 맡는다.
남은 필수 환경 준비 항목은 **0개**이며, 작성 시작 방법과 미정 입력은
[CLAUDE_HANDOFF](notes/CLAUDE_HANDOFF.md)에 정리했다.

논문 작업은 이 폴더에서 [CLAUDE.md](CLAUDE.md)와 [SETUP_PLAN.md](SETUP_PLAN.md)를 읽고,
한 번에 1 Step씩 진행한다. 상위 `/mnt/dev/work/CLAUDE.md`는 사용자 복원본을 수정하지 않는다.
준비 원문은 [paper_structure_setup.md](paper_structure_setup.md), 수정 내역은
[SETUP_CHANGES.md](notes/SETUP_CHANGES.md)에 있다.

## 위치와 역할

| 경로 | 역할 |
|---|---|
| `../.claude/` | 기존 설정 및 CVPR skill 7개·검토 역할 3개; Step 4 구성 완료 |
| `guidelines/` | 공식 규정·일정·출처; Step 2 작성 완료 |
| `evidence/`, `experiments/` | Claim·실험·결과 registry와 검증된 16-D manifest; Step 3 구성 완료 |
| `notes/` | 연구 story, novelty, contribution, 작업 계획 |
| `template/` | Pinned 공식 2026 fallback 원본·commit·파일 hash; Step 5 도입 완료 |
| `latex/`, `supplementary/` | English draft scaffold·별도 PDF; Step 5 build 검증 완료 |
| `figures/`, `tables/`, `algorithms/` | Conceptual F02/A01·descriptive T00·미측정 placeholder·sidecar |
| `references/` | 선행연구 자료; 기존 `../papers/`도 자료 위치로 유지 |
| `tools/` | 논문 통계·plotting·LaTeX 작업용 코드 |
| `submission/` | 제출 checklist 5개·최종 audit·machine-readable 검증 결과; Step 6 구성 완료 |

기존 연구 소스는 `../benchmark/ag3s/`, `../benchmark/trajopt/`, `../src/`,
`../pi05_TO_hybrid/`, `../outputs/`, `../tests/`에서 읽기만 한다.
연구 코드와 대용량 결과를 이 폴더로 복사하지 않는다.

논문 작성 규칙은 [cvpr-workspace.md](../.claude/rules/cvpr-workspace.md)를 따른다.
현재 폴더 구조와 준비 문서는 논문 결과의 검증이나 submission readiness를 의미하지 않는다.

규정 검토는 [CVPR_2027_STATUS.md](guidelines/CVPR_2027_STATUS.md)에서 시작한다.
확정된 2027 규정, provisional 2026 fallback, 미확인 항목을 구분한다.
[일정](guidelines/CVPR_2027_DEADLINES.md)과 [출처](guidelines/SOURCES.md)를 함께 확인한다.

Evidence 작업은 [CLAIMS](evidence/CLAIMS.md), [EXPERIMENTS](evidence/EXPERIMENTS.md),
[RESULTS](evidence/RESULTS.md), [PAPER_STORY](notes/PAPER_STORY.md)에서 시작한다.
현재 등록된 EXP001의 범위는 기존 T39 snapshot의 descriptive task outcomes뿐이다.
Safety 개선과 task-preservation 등 핵심 연구 목표는 아직 가설이다.
[Manifest 운영과 검증](experiments/manifests/README.md)을 함께 따른다.

Claude Code 사용은 [CLAUDE_WORKFLOW](notes/CLAUDE_WORKFLOW.md)에서 시작한다.
Skill의 호출 예시, Evidence gate, 검토 역할의 파일 소유권을 설명한다.
Skill·역할 정의와 workspace/scaffold 검증을 완료했다. 실제 Claude 모델 호출과 세 full audit
명령은 실행하지 않았다. 현재 최종 구성 결과는 [FINAL_AUDIT](submission/FINAL_AUDIT.md),
실제 제출 준비 항목은 [SUBMISSION_CHECKLIST](submission/SUBMISSION_CHECKLIST.md)를 따른다.

원고 작업은 [MANUSCRIPT_MAP](notes/MANUSCRIPT_MAP.md), 빌드는
[LaTeX README](latex/README.md), 원본 정보는 [template README](template/README.md)를 따른다.
Main 3페이지·supplementary 1페이지의 scaffold를 빌드했으며 확인 범위는
[STEP5_BUILD_AUDIT](notes/STEP5_BUILD_AUDIT.json)에 기록한다.
미측정 safety·grounding·runtime·generalization을 완료된 결과로 표시하지 않는다.

Workspace 구성 상태는 `COMPLETE`, 현재 원고는 `DRAFT_SCAFFOLD`, 실제 제출 준비 상태는
`NOT_READY`다. 핵심 실험·bibliography/novelty·구현/protocol 검증과 OpenReview 준비가 남아 있다.
후속 논문 작업은 [연구 TODO](notes/TODO.md)와 [최종 audit의 미완료 항목](submission/FINAL_AUDIT.md)을 따른다.
재검사 명령과 의존성은 [검증 안내](tools/analysis/README.md)에 있다.

후속 연구 R1에서 기존 PDF 3편의 primary 서지·핵심 내용·novelty overlap을 분석하고
초기 Related Work와 인용을 반영했다. [R1 분석](notes/REFERENCE_ANALYSIS_R1.md)과
[문헌 map](notes/RELATED_WORK_MAP.md)을 따른다. 전체 novelty 검증은 미완료이며
R2의 구현 snapshot·method audit는 Claude Code·연구 담당자의 후속 업무이며,
준비 담당자의 다음 작업으로 지정하지 않는다.
선행연구는 아직 최종 선정하지 않았다. 현재 3편과 원고 인용은 탐색적 참고 후보·임시 초안이며,
추후 사용자 후보 목록을 받은 뒤 비교 대상과 최종 인용 범위를 정리한다.
