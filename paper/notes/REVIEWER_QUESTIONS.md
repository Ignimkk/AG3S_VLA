# Reviewer Questions and Evidence Gaps

Updated: 2026-10-05 19:45 KST

| 질문 | 현재 답변 범위 | 필요한 작업 |
|---|---|---|
| Attention이 실제 task relevance를 담는가? | 아직 검증하지 않음 | GT·phase별 P002와 대조 attention ablation |
| Learned semantics와 geometry module의 기여를 분리했는가? | 통합 task outcome만 등록됨 | P003의 단일 요소·matched protocol |
| Target을 제거하면 unsafe target/payload contact를 놓치지 않는가? | Contact policy와 검사 범위에 한계가 있음 | Allowed contact·held-object·full robot coverage |
| Collision은 실제 실행 기하로 측정했는가? | EXP001의 등록 scope에 collision metric 없음 | Independent sub-step contact GT와 unknown coverage |
| Task 성능을 유지했다는 근거는 무엇인가? | Descriptive paired outcome만 있음 | 사전 margin·표본 수·non-inferiority 검토 |
| Obstacle 시연의 성공 episode selection이 결과를 왜곡하지 않는가? | T40는 미등록 candidate | Selection 규칙·모든 trial outcome·representative evaluation |
| Unseen 조건과 held-out split은 입증됐는가? | 기록된 test label만 확인 | Train/tuning/test overlap·scene provenance audit |
| 전체 시간이 control budget 안에 들어오는가? | Timing 결과는 미등록 | E0/E3b 같은 경계, cold/warm와 GPU 간섭 구분 |
| 최신 default 설정에서도 재현되는가? | 등록값은 T39 snapshot만 해당 | 별도 current-config experiment; 기존 결과 덮어쓰기 금지 |
| Novelty가 기존 safety·grounding 방법과 어떻게 다른가? | PDF 내용과 선행연구 차이 미검증 | Related-work 분석 후 N001–N004 갱신 |
| 실제 robot에 대한 evidence가 있는가? | 이 registry에는 없음 | P006의 범위 결정; simulation 결과를 hardware로 표현하지 않기 |

이 문서는 본인 원고의 내부 critique 준비다. 공식 peer review 결과 또는 실제 reviewer 의견이 아니다.
[CLAIMS](../evidence/CLAIMS.md), [METRICS](../evidence/METRICS.md), [TODO](TODO.md)를 함께 읽는다.
