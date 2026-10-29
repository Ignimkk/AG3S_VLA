# 논문 준비 상태와 Claude Code 후속 TODO

Updated: 2026-10-06 KST

## Workspace 구축

- [x] Step 1: 경로·기존 지침·기본 구조.
- [x] Step 2: 공식 규정·출처·일정 분류.
- [x] Step 3: Evidence 문서·연구 계획·16-D manifest schema·제한된 source 등록.
- [x] Step 4: CVPR skill 7개·읽기 전용 역할 3개·공통 workflow.
- [x] Step 5: Pinned 2026 fallback·English scaffold·artifact 계획·source/build 검증.
- [x] Step 6: 제출 checklist·최종 audit·workspace/PDF 검증.

필수 환경 준비는 모두 완료했다. **준비 담당 assistant의 남은 필수 준비 항목: 0개.**
실제 원고는 Claude Code가 작성한다. 역할·시작 방법·입력은 [CLAUDE_HANDOFF](CLAUDE_HANDOFF.md)를 따른다.
아래 연구·작성·제출 항목은 Claude Code·사용자·연구 담당자의 후속 업무이며,
준비 담당 assistant가 자동으로 계속 수행할 계획이 아니다.

## 탐색적 추가 작업 이력

- [x] R1: 기존 PDF 3편의 핵심 내용·primary 서지·citation·novelty overlap을 기록한다.
  [분석](REFERENCE_ANALYSIS_R1.md); 전체 survey와 novelty 확정은 미완료다.

## Claude Code·연구 담당자의 후속 업무

- [ ] Claude Code에서 outline과 근거/pending을 연결한 작성 계획을 시작한다.
- [ ] 구현 snapshot과 논문 method의 범위·변경 이력을 확인한 뒤 method를 작성한다.
- [ ] 선행연구 선정: 사용자가 추후 제공할 후보 목록을 기다린다. R1의 3편은 탐색적 참고 후보다.
- [ ] 후속 문헌 Step (후보 목록 수령 후): 비교 대상·분석 우선순위·인용 범위를 정하고 novelty를 검증한다.
- [ ] Train/tuning/test와 scene provenance를 audit한다.
- [ ] Allowed contact, robot/payload coverage, independent contact GT와 task metric을 고정한다.
- [ ] Non-inferiority margin·sample size·uncertainty·retry/exclusion 규칙을 사전 고정한다.
- [ ] P001–P005의 protocol을 정하고 연구 담당 workflow로 실행한다.
- [ ] T40/T41/T42 candidate를 log·dimension·metric·source별로 검토한다.
- [ ] Real-robot P006을 포함할지와 가능한 범위를 결정한다.
- [ ] 등록 result를 확보한 뒤 final contribution과 abstract/conclusion 문장을 작성한다.

## 작성·최종 제출 시 재확인

이 미완료 항목은 환경 준비 완료와 별개이며 outline 시작을 차단하지 않는다.
결과·method·novelty에 관한 확정 문장과 최종 제출 판정은 관련 근거 확보 후 작성한다.
실제 제출 준비는 [FINAL_AUDIT](../submission/FINAL_AUDIT.md)를 따른다.

- [ ] 원고 작성·제출·rebuttal 전에 2027 정책과 kit을 다시 확인한다.
- [ ] Evidence 사용 전 [manifest validator](../tools/analysis/validate_manifests.py)를 실행한다.
- [ ] Source가 달라지면 해당 result의 사용을 중단하고 변경 원인을 확인한다.
