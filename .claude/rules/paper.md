---
paths:
  - "references/legacy-papers/**"
---

# 논문 규칙

**로그가 원자료다.** `AG3S_T0T6_LOG.md` 에 없는 주장은 논문에도 없다. 로그에 없는 것이
필요하면 논문에 쓰지 말고 lead 에게 STEP 을 요청한다.

- 모든 수치는 `AG3S/ag3s/docs/handoff/<STEP>.verify.json` 까지 거슬러 갈 수 있어야 한다. 표를 만들 때
  그 셀이 어느 verify.json 의 어느 키에서 왔는지 주석으로 남긴다.
- figure 는 새로 만들지 않고 `AG3S/ag3s/docs/figures/` 의 것을 쓴다. 논문용 재편집이 필요하면
  verifier 에게 요청한다 — 스타일이 갈라지면 한 문서로 읽히지 않는다.
- 선행연구는 `references/legacy-papers/` 의 PDF 부터 읽는다 (cuRoboV2 등). 웹에서 찾은 것은 출처 URL 을 남긴다.
- **작성 대상 독자를 먼저 정한다.** 이 프로젝트의 기본 독자는 robot motion planning 을 알지만
  이 파이프라인은 처음 보는 연구자다. AG3S 내부 약칭(`T3`, `E3`, `D2`)을 풀지 않고 쓰지 않는다.
