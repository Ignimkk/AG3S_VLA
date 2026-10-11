# CVPR 2027 Status

Target venue: CVPR 2027

Last verified: 2026-10-05 19:21 KST

## 상태의 의미

| 분류 | 의미 |
|---|---|
| `CONFIRMED_2027` | 2027 공식 자료에서 확인한 사항 |
| `PROVISIONAL_2026` | 확인한 2026 규정을 작업 기준으로 임시 적용; 2027 확정 규정이 아님 |
| `UNKNOWN` | 2027 요구사항 또는 공개 여부를 현재 자료로 확인하지 못함 |

`PROJECT_POLICY`는 이 workspace의 작성 원칙이며 conference 규정의 상태가 아니다.
현재의 확인 결과는 영구적인 정책 확정이나 submission readiness를 의미하지 않는다.

## 확인 결과

| 항목 | 분류 | 확인 결과·근거 |
|---|---|---|
| CFP | `CONFIRMED_2027` | 공개됨; [S01](SOURCES.md#s01) |
| 개최 | `CONFIRMED_2027` | Seattle, 2027-06-20–25; [S02](SOURCES.md#s02), [S03](SOURCES.md#s03) |
| 제출 일정 | `CONFIRMED_2027` | [DEADLINES](CVPR_2027_DEADLINES.md) 참고 |
| OpenReview·저자 목록·reviewer 참여·dual submission·contemporaneous work | `CONFIRMED_2027` | [SUBMISSION_POLICY](CVPR_SUBMISSION_POLICY.md) 참고 |
| 2027 Author Guidelines | `UNKNOWN` | CFP의 연결 URL과 직접 확인 모두 404; [S05](SOURCES.md#s05) |
| 2027 Author Kit | `UNKNOWN` | 확인한 공식 repository의 README·main.tex·tag는 2026을 가리킴; 별도 2027 kit은 확인하지 못함 |
| 2027 format·anonymity·supplementary·rebuttal 세부 규정 | `UNKNOWN` | 각 문서에서 `PROVISIONAL_2026` 기준을 따로 기록 |
| 2027 LLM 세부 정책 | `UNKNOWN` | CFP에서 conference/review 활용 지침을 확정 중이라고 안내 |

2027 Author Guidelines/Kit이 존재하지 않는다고 단정하지 않는다. 확인한 URL과
repository의 관찰 결과를 기록하고, 공개 경로가 바뀌었는지 다시 확인한다.

## 적용 순서

1. 해당 항목의 확인된 2027 규정을 우선한다.
2. 비어 있는 항목만 확인된 2026 규정을 provisional하게 적용한다.
3. 관련 공개 자료가 서로 다르면 차이를 기록하고, 해당 항목을 임의로 확정하지 않는다.
4. 새로운 2027 지침 공개 시 각 문서의 상태·출처·확인 시각을 함께 갱신한다.
5. 원고·rebuttal 작성 및 제출 전 최신 자료를 다시 확인한다.

특히 2026의 arXiv prior-work 설명을 2027 contemporaneous-work 규정보다 우선하지 않는다.
Step 5에서 pin한 2026 fallback template과 영문 원고 scaffold를 도입했다.
Original·working metadata·build 범위는 [template README](../template/README.md)와
[LaTeX README](../latex/README.md)를 따른다. 2027 kit 상태는 여전히 UNKNOWN이다.

## 문서 안내

- [일정](CVPR_2027_DEADLINES.md)
- [Format](CVPR_FORMAT_RULES.md)
- [Submission](CVPR_SUBMISSION_POLICY.md)
- [Anonymity](CVPR_ANONYMITY_POLICY.md)
- [LLM](CVPR_LLM_POLICY.md)
- [Supplementary](CVPR_SUPPLEMENTARY_POLICY.md)
- [Rebuttal](CVPR_REBUTTAL_POLICY.md)
- [출처와 재확인 기록](SOURCES.md)

출처 기록: [S01](SOURCES.md#s01), [S02](SOURCES.md#s02), [S03](SOURCES.md#s03), [S05](SOURCES.md#s05), [S07](SOURCES.md#s07), [S08](SOURCES.md#s08), [S13](SOURCES.md#s13).

## Step 5 부분 재확인

시각: 2026-10-05 21:01 KST. 2027 CFP·Author Guidelines URL과 공식 kit repository·main.tex·tags를 다시 확인했다.
해당 Guidelines URL은 404이며 확인한 README/main.tex/tag는 2026을 가리킨다.
별도 2027 kit의 전체 공개 여부를 단정하지 않는다. Adopted fallback은 tag
`CVPR2026-v1(latex)`와 commit `12909ae437f6dbc7435069cfdb4ca44c18e6a02f`다.
이는 template 관련 부분 재확인이며 모든 정책을 새 시각에 검증했다는 뜻이 아니다.
기록: [SOURCES](SOURCES.md#s14), [PROVENANCE](../template/PROVENANCE.json).

## Step 6 구성 검증 완료

시각: 2026-10-06 00:01 KST. 2027 공개 자료와 2026 fallback의 관련 정책을 부분 재확인했다.
기존 CONFIRMED_2027/PROVISIONAL_2026/UNKNOWN 분류를 유지한다.
이번 확인의 source별 범위는 [SOURCES](SOURCES.md)의 Step 6 기록을 따른다.

Workspace 구성을 완료했으며 실제 원고의 제출 준비 상태는
[FINAL_AUDIT](../submission/FINAL_AUDIT.md)의 NOT_READY다.
새 정책 공개, 실제 원고/package 완성 또는 source 변경 시 해당 검사를 다시 수행한다.

## 2026-10-07 부분 재확인 — 학회 scope·Appendix·프로젝트 사이트

시각: 2026-10-07 19:18 KST. 2027 CFP의 Embodied vision and robotics, Physical AI/VLA, 3D geometry 주제를 확인했다. Author Guidelines의 내용을 이번 도구 접근에서는 확보하지 못했으며, 과거 404 결과를 이번 시각의 HTTP 상태로 재확인한 것은 아니다.

2026 지침과 공식 kit를 근거로 부록·프로젝트 사이트 계획을 작성했다. 2027 확정 정보와 2026 임시 기준을 구분한다. 전체 정책이나 실제 제출물의 재검증을 의미하지 않는다. 출처·확인 범위·후속 조치: [Appendix·프로젝트 페이지 계획](../notes/CVPR_APPENDIX_PROJECT_PAGE_PLAN.md).
