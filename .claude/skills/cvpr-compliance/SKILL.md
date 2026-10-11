---
name: cvpr-compliance
description: Audit this CVPR manuscript package against confirmed 2027 policies, provisional 2026 rules, and documented unknowns.
---

# CVPR Compliance Audit

[공통 workflow](../../../paper/notes/CLAUDE_WORKFLOW.md)의 정책 분류와 최종 audit gate를 읽는다.
요청: `$ARGUMENTS`. `full`, `format`, `anonymity`, `dual-submission`, `llm`,
`supplementary`, `rebuttal`, `references`, `deadlines`를 지원한다. 기본은 `full`이다.

1. [2027 STATUS](../../../paper/guidelines/CVPR_2027_STATUS.md)와
   [SOURCES](../../../paper/guidelines/SOURCES.md)를 읽고 요청 범위의 policy 문서를 확인한다.
   실제 제출·rebuttal 직전이면 최신 공식 자료를 재확인해 출처·확인 시각을 갱신한다.
2. 항목마다 `CONFIRMED_2027`, `PROVISIONAL_2026`, `UNKNOWN`을 표시한다.
   Workspace의 `PROJECT_POLICY`를 conference 의무와 구분한다.
3. 실제 main PDF·LaTeX·kit provenance가 있으면 page count, English, review mode,
   style/margin/font, bibliography 위치와 정합성을 확인한다. Artifact가 없으면 `NOT_CHECKED`다.
4. 본문·supplementary·영상·metadata·URL의 anonymity, 저자 식별 요소와 hidden reviewer
   instruction을 점검한다. 외부 링크는 content expansion·tracking·identity 등 정책의 범위로
   판단하고 citation/third-party dataset 링크를 일괄 금지하지 않는다.
5. Author-list 마감, profile·conflict·reviewer-service, dual submission, deadlines와
   OpenReview 확인 자료를 점검한다. 파일 구성만으로 실제 등록·업로드 완료를 추정하지 않는다.
6. Supplementary의 main-paper 의존성·개선 method/결과·수정 main PDF 제한과 rebuttal의
   총 page count·template·links를 해당 정책 상태로 점검한다.
7. 인용 실재성·bibliography·unsupported claim·LLM author assistance·prompt injection을 확인한다.
   Evidence audit와 내부 원고 critique는 별도 gate이며 정책 audit로 대체하지 않는다.

보고 표: 항목, 적용 규정·source URL, 상태, 검사 범위(`CHECKED`/`NOT_CHECKED`),
판정(`BLOCKING`/`WARNING`/`OK`), 근거 파일/위치, 필요한 조치.
없는 검사 자료를 `OK`로 처리하지 않는다. 필수 검사 미완료·불확정 정책은 제출 준비
판정의 unresolved item으로 남기며, `UNKNOWN` 자체를 학회 위반이라고 단정하지 않는다.
본 skill은 audit 보고서를 반환하며 OpenReview 등록·업로드·제출을 실행하지 않는다.
