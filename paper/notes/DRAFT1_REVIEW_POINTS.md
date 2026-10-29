# Draft 1 — 사용자 결정이 필요한 점

작성: 2026-10-06 KST. 본문은 Abstract–Experimental Setup까지이며 결과는 비어 있다(빨간 `\tbd{}`).

| # | 결정할 것 | 지금 초안의 선택 | 이유 |
|---|---|---|---|
| 1 | 기여 문장 | 'attention으로 Target 구분'을 첫 기여로 쓰지 않고 querier-dependent exemption · payload attach · chunk refinement 로 좁힘 | KNOWS(2606.09749)가 pi0.5 attention으로 Target을 골라 obstacle에서 제외, MoManipVLA가 실기 ESDF에서 target mask 제외 |
| 2 | 방법 이름 | 임시로 AG3S | 사내 모듈명(Attention-Guided 3D Safety Scene) |
| 3 | 제목 | 임시: Target-Aware Distance Fields from VLA Attention for Refining Action Chunks | 첫 안 |
| 4 | cuRobo 역할 | TSDF/ESDF(`Mapper`)만 cuRobo, 정제는 자체 SQP로 명시 | 코드에 MotionGen·cuRobo optimiser 없음 |
| 5 | 평가 구성 서술 | T39 구성(gripper 링크만, margin 0, phase 고정 approach, 왼팔) 그대로 | 코드로 확인된 사실. Limitations에 명시 |
| 6 | 실험 설계 | 4개 configuration · 지표 · 통계는 제안일 뿐 | 새 실험 전 확정 필요(독립 contact 정의, non-inferiority margin, episode 수) |
| 7 | 인용 | 19개, PDF에 인쇄된 서지만. 사용자 추가 후보 목록 대기 | 규칙: venue를 기억으로 채우지 않음 |
| 8 | 제출 패키지 | `../figures`·`../algorithms` 상대경로 입력 | OpenReview 제출 전 평탄화 필요 |


## Draft 2 추가 (2026-10-06)

| # | 결정할 것 | 지금 초안의 선택 |
|---|---|---|
| 9 | 기여 2 (phase probe) | 'VLA가 sub-task를 출력'이 아니라 'frozen KV cache의 linear probe가 닫힌 3-class 단계를 읽는다'. V5는 gate를 끈 채 돌았으므로 Method에는 optional gate로 쓰고 결과 자리는 TBD |
| 10 | probe 근거 | val 0.981 / closed-loop 0.959이지만 단순 규칙 baseline 0.942, layer 0도 비슷한 정확도. 규칙 baseline과의 비교·layer sweep을 등록해야 함 |
| 11 | V5 'best' | 12/14 vs V4 10/14는 McNemar p=0.5(불일치 쌍 2개). 같은 episode에서 개발됨(held-out 없음). HOLD 163 vs 56, 무장애물 4/5 동일. 본문에 수치를 넣지 않음 |
| 12 | WBAG(2610.01083) | 기여 3과 가장 가까움(frozen pi0.5, target 제외, grasp 후 attached geometry). 차이는 인지된 Target, dense ESDF, chunk SQP. 단 WBAG는 whole-body, 우리는 gripper 링크만 |
| 13 | 로그 오류 | AG3S_T0T6_LOG의 probe label 구간(0-5/6-9/10-13)은 코드(0-3/4-8/9-13)와 다름 — 코드가 맞음 |
