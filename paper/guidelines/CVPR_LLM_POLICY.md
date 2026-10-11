# CVPR LLM Usage Policy

Target venue: CVPR 2027

Last verified: 2026-10-05 19:21 KST

## 2027 안내의 범위

`CONFIRMED_2027`: CFP는 **conference와 reviewing process의 일부에서 LLM을
활용할 예정이며 scope·허용 용도·지침을 확정 중**이라고 안내한다.
이 문장만으로 저자의 모든 LLM 사용이 허용되거나 금지되었다고 추론하지 않는다.
2027 author-side 세부 정책은 현재 `UNKNOWN`이다.

[2027 CFP: LLM Usage](https://cvpr.thecvf.com/Conferences/2027/CallForPapers).

## Author assistance — `PROVISIONAL_2026`

2026은 저자의 productive tools 사용을 허용하지만, 사실·인용·독창성의 책임은 저자에게
있다. 허위 citation과 명백한 factual error는 rejection 사유가 될 수 있다.
Reviewer나 automated tool을 조작하려는 hidden instruction은 금지된다.

[2026 Author Guidelines: About LLMs / Prompt injection](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines).

## Workspace 운영 — `PROJECT_POLICY`

- Claude가 citation, DOI, 결과값, statistical significance, 구현 사실을 만들지 않는다.
- 과학적 주장은 `evidence/CLAIMS.md`, 수치는 `evidence/RESULTS.md` 및 원자료로 추적한다.
  이 문서들은 Step 3에서 구성했다. 등록된 EXP001의 제한된 task-outcome scope만
  사용하며, 미등록 결과나 미측정 safety·runtime 성과를 가정하지 않는다.
- 본인 원고의 내부 critique인 `/cvpr-review`는 author assistance다.
  학회에서 배정받은 타인 원고의 공식 peer review와 구분한다.
- 저자의 draft 점검 가능 여부를 공식 reviewer service의 LLM 사용 허가로 해석하지 않는다.
- 최신 2027 정책을 제출 전에 재확인한다. 현재 파일은 별도의 disclosure 의무가 없다는 확인이 아니다.

출처 기록: [S01](SOURCES.md#s01), [S06](SOURCES.md#s06).
