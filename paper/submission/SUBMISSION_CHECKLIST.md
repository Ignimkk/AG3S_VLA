# CVPR 2027 Submission Checklist

Updated: 2026-10-05 23:52 KST

Current status: `NOT_READY`. 아래 checked 항목은 현재 workspace/scaffold 범위다.

## 규정·일정

- [x] 공식 CFP·Dates·profile 안내와 kit 상태를 재확인하고 출처·시각을 기록했다.
- [ ] 최신 2027 상세 Guidelines/Author Kit·LLM author-side 규정을 확인하고 UNKNOWN을 처리한다.
- [ ] 실제 OpenReview timestamp·file size/type 제한을 확인한다. AoE→KST 참고값을 확정 시각으로 쓰지 않는다.
- [ ] Dual submission·contemporaneous/prior-work 조건을 실제 저자/원고 이력과 대조한다.

등록 2026-11-10, main 11-16, supplementary 11-23 AoE는
[2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)와
[일정 문서](../guidelines/CVPR_2027_DEADLINES.md)를 따른다.

## OpenReview — 상태 미확인

- [ ] Paper 등록·실제 paper ID·title·abstract·subject area를 확인한다.
- [ ] 모든 저자의 profile·confirmed email·career·conflict·expertise·publication 정보를 확인한다.
- [ ] Abstract registration 마감 전 저자 목록을 확정한다. 이후 추가·삭제 제한을 지킨다.
- [ ] 초청되는 reviewer/AC/SAC service와 필요한 예외 처리 여부를 확인한다.
- [ ] 저자 순서·별도 profile deadline 등 2027 세부 안내를 확인한다.

[공식 profile 안내](https://cvpr.thecvf.com/Conferences/2027/CompleteYourORProfile),
[SUBMISSION_POLICY](../guidelines/CVPR_SUBMISSION_POLICY.md). 개인 profile 완료를 이 workspace에서 추정하지 않는다.

## 원고·근거

- [x] English section 7개·review style·원본 hash·분리된 supplementary 기반을 구성했다.
- [x] EXP001의 실제 16-D·source/array·R001–R003 연결을 재검증했고 task outcome scope를 표시했다.
- [ ] Novelty·관련 연구·baseline citation과 구현 snapshot을 검증한다.
- [ ] 주요 claim의 grounding·독립 contact/safety·ablation·generalization·runtime 결과를 확보한다.
- [ ] Train/tuning/test 독립성·allowed contact·geometry/payload coverage를 검증한다.
- [ ] Paired/episode-cluster 분석과 non-inferiority margin·uncertainty·exclusion 규칙을 확정한다.
- [ ] 최종 manuscript·figure/table/caption의 모든 수치를 해당 registry·source key로 추적한다.
- [ ] DRAFT·placeholder·미등록 paper ID·pending bibliography의 상태와 출처 표시를 최종 점검한다.
- [ ] 최종 [page](PAGE_LIMIT_CHECKLIST.md)·[anonymity](ANONYMITY_CHECKLIST.md)·
  [references](REFERENCES_CHECKLIST.md)·[supplementary](SUPPLEMENTARY_CHECKLIST.md) 검사를 완료한다.

## 최종 package

- [ ] `/cvpr-evidence full`의 실제 보고서와 미해결 사항을 확인한다.
- [ ] `/cvpr-review full`의 실제 내부 critique와 수정 결과를 확인한다.
- [ ] `/cvpr-compliance full`의 실제 보고서·정책 상태·미검사 항목을 확인한다.
- [ ] 최종 PDF/package의 hash·버전·source revision과 업로드 후 다운로드 파일을 대조한다.
- [ ] 실제 제출 완료 표시·receipt·마감 전 상태를 확인한다.

Skill 정의와 이번 Step의 수동 scaffold audit를 위 세 명령의 실행 이력으로 표시하지 않는다.
