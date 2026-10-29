# References Checklist

Updated: 2026-10-06 KST (R1)

R1에서 기존 PDF 3편의 primary 서지와 관련 citation을 등록했다.
전체 related-work/novelty 검증은 미완료다. 상태: `BLOCKING`, `PARTIAL`.
확인한 페이지와 범위는 [R1 분석](../notes/REFERENCE_ANALYSIS_R1.md)을 따른다.
[RELATED_WORK_MAP](../notes/RELATED_WORK_MAP.md), [NOVELTY](../notes/NOVELTY.md),
[LLM_POLICY](../guidelines/CVPR_LLM_POLICY.md)를 따른다.

## 현재 구성

- [x] Official kit의 예시 bibliography를 연구 인용으로 복사하지 않았다.
- [x] 3편의 제한된 분석을 전체 verified literature review로 표시하지 않는다.
- [x] Citation 확인용 skill과 literature-analyst의 읽기 전용 역할을 구성했다.

## 검증할 내용

- [x] R1 기존 PDF 3편의 핵심 section과 primary source 확인; 전체 문헌 queue는 남아 있다.
- [x] R1 title·authors·year·identifier 대조; cuRobo local version UNKNOWN·venue 미확인 범위를 명시했다.
- [x] R1 BibTeX key·실제 cite·rendered bibitem 3개 일치; 누락·중복·미사용 entry·unresolved citation 없음.
- [ ] Baseline과 핵심 method가 올바른 원 논문/버전을 인용하는지 확인한다.
- [x] R1 3편 내용 비교의 근거 page/section과 적용 scope를 기록했다.
- [ ] Novelty/first/SOTA 문장을 prior-art 분석과 실측 비교로 뒷받침한다.
- [ ] 영향을 준 work는 공개 날짜와 무관하게 prior work로 cite/discuss한다.
- [ ] Contemporaneous work도 cite/discuss하고 가능한 범위에서 비교한다.
- [ ] Self-citation·concurrent submission의 anonymity 처리를 해당 규정으로 확인한다.
- [x] R1 verified entry 3개와 실제 citation을 추가한 뒤 bibliography 출력을 활성화했다.

2027 contemporaneous cutoff는 2026-09-15이며 arXiv와 다른 publication에 같은 원칙을 적용한다.
[2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers).
Citation build 성공은 문헌의 충분성·최종 novelty를 검증한 것이 아니다.
