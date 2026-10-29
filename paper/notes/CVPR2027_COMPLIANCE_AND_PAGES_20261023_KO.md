# CVPR 2027 정책 반영과 8 page 감축안 — 2026-10-23 KST

근거: [CVPR 2027 Author Guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines), [CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers), author kit v1 (`paper/template/PROVENANCE.json`). 정책 요약은 `paper/guidelines/` 각 문서 상단의 2026-10-23 절 (`CONFIRMED_2027`).

## 1. 이번에 반영한 것

| 항목 | 전 | 후 |
|---|---|---|
| Template | 2026 kit (`PROVISIONAL_2026`), review copy 에 Paper ID 표기 | 2027 kit v1 원본을 `paper/template/cvpr2027_official/` 에 보존, working `cvpr.sty` (main · supplementary) 를 2027 bytes 로 교체. 2026 기록은 `cvpr2026_official/` · `PROVENANCE_2026.json` 에 보존. 차이는 Paper ID 제거뿐, margin · font · 줄 번호 동일 |
| `metadata.tex` · `main.tex` | `\paperID{*****}`, "PROVISIONAL_2026 kit" 주석 | `\paperID` 제거, 2027 kit 주석. PDF 머리글 "CVPR 2027 Submission. CONFIDENTIAL REVIEW COPY." 확인 |
| LLM Usage Statement | 없음 (desk reject 사유) | `sections/07_llm_usage.tex` 를 main text 뒤 · references 앞에 추가. 도구 (Anthropic Claude, Claude Code) · 원고 · figure code · 연구 code 와 실험 scripting 범위. **저자 확인 필요** (`\tbd`) |
| WBAG | Related Work · SafeLIBERO 표 · bib | 사용자 지시로 전부 제거 (bib entry `zhen2026wbag`, Related Work 의 "and WBAG", SafeLIBERO "reported" 행). 남은 참조 0 |
| `paper/guidelines/` | 2027 세부 `UNKNOWN` | STATUS · FORMAT · LLM · SUBMISSION · SUPPLEMENTARY · REBUTTAL · ANONYMITY · DEADLINES 에 `CONFIRMED_2027` 절, SOURCES 에 S16 (Guidelines) · S17 (kit) |

빌드: `python3 paper/tools/latex/build_cvpr_v1.py` → BUILD_PASS, overfull 0, 미정의 참조 0. `python3 paper/tools/latex/build.py --check-only` 는 kit hash 검사를 통과한 뒤, 이번 작업과 무관한 기존 문제 (EXP001 manifest 가 옛 경로 `benchmark/ag3s/docs/handoff/T39.verify.json` 을 가리킴) 로 실패한다.

## 2. 제출 전에 남은 정책 항목

- **OpenReview profile**: 모든 공저자가 2026-11-10 AoE 까지 완성. 기관 이메일이 없으면 승인에 최대 2 주 — 지금 확인.
- **Fabricated content = desk reject**: 참고문헌 전부를 원문과 대조 (`/cvpr-reference-analysis`), `\tbd` · `\pend` 표시 전부 해소.
- **Contemporaneous (2026-09-15 이후)**: `agarwal2026multilink` (2609.40007) · `guo2026nudge` (2609.35231) 의 정확한 공개일 확인. 이후면 인용 · 논의로 충분.
- **외부 링크**: 본문 · 참고문헌에 내용을 확장하는 링크 금지. 현재 본문에 URL 없음.

## 3. 8 page 감축안 (보류)

> **사용자 판정 (2026-10-23):** Fig 1 (teaser: 연구 방향과 성과), Fig 2 (전체 overview), Fig 3 (method pipeline) 은 모두 유지한다. 내용 감축은 추후 진행한다. 아래 표의 Fig 2 행은 철회하며, 감축은 2–6 행 (본문 압축 · supplementary 이동) 으로만 한다.

현재 본문은 9 쪽 오른쪽 열 하단에서 끝난다 (약 8.9 page). SafeLIBERO · 실물 절이 결과로 채워지면 약 0.3 page 가 더 든다. 목표 감축 **약 1.3 page**.

| 순서 | 항목 | 현재 위치 · 분량 | 조치 | 절감 (page, 추정) |
|---|---|---|---|---|
| 1 | Fig 2 overview | p.2 상단, 3.25 in + caption | 삭제하고 Fig 1 · Fig 3 로 대체 (reviewer 지적: Fig 3 과 내용 중복). 또는 상단 strip 만 남겨 1.2 in | 0.42 (삭제) / 0.25 (축소) |
| 2 | Method 본문 | p.4 L – p.6 R, 약 1.9 page (Fig 3 제외) | 절마다 "naive → 문제" 도입 1 문장으로 압축, 3.3 consistency 문단과 3.5 grasp/carry 문단 축약, CCR 의 수식 설명 줄이기 | 0.25 |
| 3 | Experiments Setup · Q1 · Q4 | p.6 R – p.7 R | 정의 문단을 표 caption 으로, 진단 재실행 문장을 supplementary 로, Table 4 를 본문 2 문장으로 | 0.25 |
| 4 | Introduction | p.1 R – p.3 L | 성과 문단과 기여 목록의 중복 제거 | 0.15 |
| 5 | Related Work | p.3 L – p.4 L, 약 0.5 page | 네 문단을 세 문단으로 (attention · feature graph 를 grounding 문단에 합침) | 0.15 |
| 6 | Limitations · Conclusion | p.9 L, 약 0.25 page | 결론 문단 2 문장으로 | 0.05 |
| | 합계 | | | 약 1.27 (Fig 2 삭제 시) |

Fig 2 를 지울지, 축소할지는 사용자 판정이 필요하다 (사용자가 Intro overview 로 요청한 그림). 나머지 2–6 은 내용 삭제가 아니라 압축 · supplementary 이동이다.
