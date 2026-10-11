---
name: cvpr-figure
description: Plan or render paper figures and tables with traceable registered results, explicit units, and reproducible sidecars.
---

# CVPR Figure and Table

[공통 workflow](../../../paper/notes/CLAUDE_WORKFLOW.md)의 Evidence gate와 파일 소유권을 읽는다.
요청: `$ARGUMENTS`. `plan Fxx 또는 Txx`, `render Fxx 또는 Txx`를 지원한다.
기본은 [FIGURES_TABLES](../../../paper/evidence/FIGURES_TABLES.md)의 계획 검토다.

1. 목적·claim·result·metric·manifest·verify pointer를 확인한다. 정량 렌더링 전에는
   manifest validator를 실행한다. 등록된 범위에 값이 없으면 필요한 근거를 보고한다.
2. Method diagram은 conceptual/hypothesis와 구현된 단계를 구분한다. Teaser의 safe/unsafe,
   collision 판정과 qualitative 선택은 실제 측정과 selection 기준으로 뒷받침한다.
3. Unit·denominator·aggregation·paired structure를 명시한다. Error bar 종류와 표본 단위를
   caption에 쓰고, 미측정 cell은 `not measured`로 표시한다. 임의의 보간·수치 생성은 하지 않는다.
4. 분석·plotting 코드는 `paper/tools/analysis/` 또는 `paper/tools/plotting/`에 둔다.
   결과는 `paper/figures/` 또는 `paper/tables/`의 해당 폴더에 저장한다.
   기존 research figure는 읽기만 하며, 기존 artifact를 수정할 때 provenance를 유지한다.
5. PDF/SVG 등 적절한 출력과 JSON sidecar를 생성한다. Sidecar에는 artifact ID,
   claim/result/metric ID, source 경로·pointer·input hash, script/version, 단위·aggregation,
   caption scope를 기록한다. Conceptual diagram에는 실측 결과가 없음을 명시한다.
6. 실제 출력의 legibility·clipping·색 구분·font embedding·anonymity와 재생성 가능성을 확인한다.
   최종 page-size 점검은 도입한 공식 template을 기준으로 한다.
7. Evidence map과 PAPER_LOG에 파일·검증·남은 한계를 기록한다.

`plan`은 그림 생성 완료를 뜻하지 않는다. 내부 sidecar와 절대 경로는 provenance용이며
공개 제출물에 그대로 포함하지 않는다. 필요 시 기존 figure style을 읽되 research 경로로 쓰지 않는다.
