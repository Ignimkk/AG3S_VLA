---
paths:
  - "references/legacy-papers/**"
---

# 논문 규칙

로그 (`AG3S_T47_LOG.md` 최신 · `AG3S_T0T6_LOG.md` 이전) 는 주요 원자료 중 하나다.
로그 밖의 자료도 논문에 사용할 수 있으며, 측정·파생·추정·계획·placeholder 여부와 출처를 함께 표시한다.

- 모든 수치는 `verify.json`, 계산식, 가정 또는 작성 근거까지 거슬러 갈 수 있어야 한다. 표를 만들 때
  각 셀의 출처와 자료 유형을 주석으로 남긴다.
- figure 는 새로 만들지 않고 `AG3S/ag3s/docs/figures/` 의 것을 쓴다. 논문용 재편집이 필요하면
  verifier 에게 요청한다 — 스타일이 갈라지면 한 문서로 읽히지 않는다.
- 선행연구는 `references/legacy-papers/` 의 PDF 부터 읽는다 (cuRoboV2 등). 웹에서 찾은 것은 출처 URL 을 남긴다.
- **작성 대상 독자를 먼저 정한다.** 이 프로젝트의 기본 독자는 robot motion planning 을 알지만
  이 파이프라인은 처음 보는 연구자다. AG3S 내부 약칭(`T3`, `E3`, `D2`)을 풀지 않고 쓰지 않는다.
