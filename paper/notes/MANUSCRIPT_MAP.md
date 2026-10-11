# Manuscript and Artifact Map

Updated: 2026-10-06 KST (R1)

Status: `DRAFT_SCAFFOLD`. 원고는 English이며 완성된 contribution·결론이 아니다.

| Section | 파일 | Evidence / 필요한 작업 |
|---|---|---|
| Abstract | [00_abstract](../latex/sections/00_abstract.tex) | C001–C004 연구 질문; 성과 문장 미작성 |
| Introduction | [01_introduction](../latex/sections/01_introduction.tex) | Novelty·citation 검증, candidate contribution |
| Related Work | [02_related_work](../latex/sections/02_related_work.tex) | R1 3편 검증·초기 비교; 전체 grounding/contact/safety 문헌 확장 필요 |
| Method | [03_method](../latex/sections/03_method.tex) | C002–C004; F02/A01 conceptual, source correspondence audit 필요 |
| Experiments | [04_experiments](../latex/sections/04_experiments.tex) | C006/EXP001/R001–R003 descriptive; P001–P005 미실행 |
| Conclusion | [05_conclusion](../latex/sections/05_conclusion.tex) | 가설과 제한된 관찰만; final conclusion 보류 |
| Limitations | [06_limitations](../latex/sections/06_limitations.tex) | Safety·split·runtime·generalization·hardware 미검증 |

## 포함할 자료

- F02: conceptual native-LaTeX overview; measured architecture 성과가 아니다.
- A01: proposed pseudocode; source 구현·안전성 인증이 아니다.
- T00: EXP001의 R001/R002 task counts만 연결한 별도 descriptive table.
- T01: P001 main comparison layout; 모든 측정값 `n/m`.
- F01/F03–F05, T02–T04: 계획·source·sidecar를 구성했지만 default main에는 포함하지 않는다.
- [FIGURES_TABLES](../evidence/FIGURES_TABLES.md)와 artifact별 PLAN.md를 현재 상태의 기준으로 읽는다.

## Page budget — 작성 계획이며 통과 판정이 아님

2026 fallback의 main 8-page working limit을 기준으로 초안의 공간을 배분한다.
Abstract/teaser/introduction 약 1.5, related work 약 0.75, method 약 2.0,
experiments 약 2.75, conclusion/limitations 약 0.5 page를 첫 계획으로 삼는다.
실측 evidence 확보 후 actual PDF에서 조정하며 margin·font·style은 바꾸지 않는다.
현재 짧은 scaffold PDF의 page count는 최종 원고의 page-limit 준수를 증명하지 않는다.

## Citation 및 공개 자료

`references.bib`는 [R1 분석](REFERENCE_ANALYSIS_R1.md)의 검증된 preprint/report 3개를 포함한다.
Related Work에서 모두 인용한다. 전체 survey·novelty 확정은 미완료이며
허위 citation이나 official kit의 예시 bibliography로 보완하지 않는다.
`metadata.tex`의 paperID는 미등록 placeholder다. 내부 source path/ID는 주석·sidecar에 보존하고
공개 PDF에는 raw 경로·기관·저자 정보가 나타나지 않게 점검한다.
