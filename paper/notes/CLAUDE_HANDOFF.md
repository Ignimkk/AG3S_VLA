# Claude Code 논문 작성 인계

## 역할과 준비 상태

실제 논문 작성은 Claude Code가 담당한다. 준비 담당 assistant는 규칙·설정·폴더·template·
검증 도구·입력 안내만 구성한다. 사용자 별도 요청 없이 후속 연구 분석이나 원고 작성을 계속하지 않는다.

| 준비 항목 | 상태 |
|---|---|
| Step 1: 작업 공간·경로·지침 | 완료 |
| Step 2: 규정·출처·미확인 정책 구분 | 완료 |
| Step 3: Evidence registry·manifest·검증 도구 | 완료 |
| Step 4: Claude 작성 skill·검토 역할·workflow | 완료 |
| Step 5: Template·LaTeX scaffold·artifact·build 기반 | 완료 |
| Step 6: 제출 checklist·workspace 검증 | 완료 |

**필수 환경 준비의 남은 항목: 0개.** 이 문서는 작성 시작과 업무 범위를 정리한 인계 안내다.
설정의 정적 검사와 scaffold build는 수행했다. 실제 Claude 모델 호출은 아직 수행하지 않았다.
환경 준비 완료는 최종 과학적 원고나 제출 준비 완료를 의미하지 않는다.

## Claude Code 시작

작업 폴더는 `/mnt/dev/work/paper`다. [CLAUDE.md](../CLAUDE.md),
[작성 규칙](../../.claude/rules/cvpr-workspace.md), [workflow](CLAUDE_WORKFLOW.md)를 읽는다.
Root `/mnt/dev/work/CLAUDE.md`는 사용자 복원본을 수정하지 않는다.
200자 제한은 `paper/CLAUDE.md`에만 적용한다. 공통 설정은 Root `.claude/`에 유지한다.
상위 지침의 자동 로딩을 차단한 구성이라고 가정하지 않는다.

Claude Code에 전달할 첫 요청 예시:

```text
현재 폴더의 CLAUDE.md와 notes/CLAUDE_HANDOFF.md를 읽고 논문 작성 업무를 시작하세요.
첫 Step에서는 기존 자료와 미확인 항목을 정리하고, 근거를 연결한 논문 outline을 제시하세요.
선행연구 최종 선정은 사용자 목록을 기다립니다. 현재 R1 문헌과 원고 인용은 참고 후보입니다.
가설·미측정 결과·conceptual 도식을 성과로 쓰지 말고 근거가 없는 부분은 pending으로 표시하세요.
기존 연구 코드와 Root CLAUDE.md는 수정하지 마세요. 한 번에 1 Step만 진행하세요.
```

이 예시는 실행 기록이 아니다. 구체적인 원고 작성·수정은 Claude Code가 맡는다.

## 입력과 후속 업무

| 입력 | 자료 | Claude Code·사용자·연구 담당자의 후속 업무 |
|---|---|---|
| 연구 목표 | [PAPER_STORY](PAPER_STORY.md), [CONTRIBUTIONS](CONTRIBUTIONS.md), [CLAIMS](../evidence/CLAIMS.md) | 근거 확보 후 가설과 최종 기여를 구분해 문장 확정 |
| 선행연구 | [RELATED_WORK_MAP](RELATED_WORK_MAP.md), [R1 분석](REFERENCE_ANALYSIS_R1.md) | 사용자 목록 대기; 선정 전 자동 확대·최종 비교 대상 확정 금지 |
| 구현 설명 | [BASELINES](../evidence/BASELINES.md), [원고 map](MANUSCRIPT_MAP.md), read-only research source | Version·입출력·좌표계·contact·solver·실행 범위를 확인한 뒤 method 작성 |
| 결과·그림·표 | [RESULTS](../evidence/RESULTS.md), [EXPERIMENTS](../evidence/EXPERIMENTS.md), [FIGURES_TABLES](../evidence/FIGURES_TABLES.md) | 자료 유형·출처·검증 상태를 표시; 필요한 실험은 연구 담당 workflow로 요청 |
| 규정·template | [2027 STATUS](../guidelines/CVPR_2027_STATUS.md), [template 안내](../template/README.md) | 저장된 확인 시각·provisional/UNKNOWN 확인; 필요한 최신 규정 재확인 |
| 제출 준비 | [FINAL_AUDIT](../submission/FINAL_AUDIT.md), [제출 checklist](../submission/SUBMISSION_CHECKLIST.md) | 최종 자료 확보 후 full audit; 초안 시작 전 환경 준비 항목과 구분 |

현재 원고와 R1 citation은 준비 과정의 임시 scaffold/탐색적 자료다.
사용자가 최종 내용이나 선행연구를 선정·승인한 것으로 간주하지 않는다.
EXP001은 historical T39 snapshot의 descriptive task outcomes뿐이다.
Safety·task preservation·grounding·runtime·generalization·최종 novelty는 입증되지 않았다.
이 후속 업무들은 준비 담당자가 자동으로 계속 수행할 계획이 아니다.

## 작성 중 검증

Paper 폴더에서 기존 결과를 사용하기 전에 source 검사를 실행한다.

```bash
python3 tools/analysis/validate_manifests.py
python3 tools/latex/build.py --check-only
```

의존성·검사 범위는 [검증 도구 README](../tools/analysis/README.md),
빌드는 [LaTeX README](../latex/README.md)를 따른다.
Source 불일치가 발생한 결과는 재확인 전 사용하지 않는다. 해당 결과를 쓰지 않는 outline 작업은 가능하다.
문헌·실험 입력이 미정이어도 outline과 pending 표시가 있는 초안을 시작할 수 있다.
근거가 확보되기 전에는 확정된 method·성과·novelty 문장이나 최종 제출 준비 판정을 작성하지 않는다.
