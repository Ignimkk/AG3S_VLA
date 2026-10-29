# AG3S 채택 · 미채택 결정 모음

여러 세션에서 수행한 실험을 근거로, 무엇을 채택했고 무엇을 채택하지 않았는지 주제별로 모아 둔다.

- 원시 기록은 storage run, 진행 기록은 `AG3S/ag3s/docs/handoff/*.task.md` 에 있다.
- 이 폴더는 결정과 그 이유 · 한계만 적는다.

## 규칙

- 상태는 셋 중 하나다.
  - **채택**: 사용자가 확정한 것.
  - **미채택**: 실험했지만 쓰지 않기로 한 것.
  - **판정 대기**: 실험은 끝났고 사용자 판정을 기다리는 것.
- 결정마다 이유와 한계를 함께 적는다. 수치는 출처 기록의 값만 옮기고 출처를 단다.
- 채택 항목은 [ADOPTED_KO.md](ADOPTED_KO.md), 미채택 항목은 [NOT_ADOPTED_KO.md](NOT_ADOPTED_KO.md), 남은 문제 · 필요한 실험은 [OPEN_ISSUES_KO.md](OPEN_ISSUES_KO.md) 에 주제별로 이어 쓴다.
- 같은 주제의 결정이 바뀌면 새 문서를 만들고 옛 문서는 남긴다. 옛 문서 상태 줄에는 "대체됨 → 새 문서" 를 적는다.

## 목록

| 주제 | 상태 | 문서 | 날짜 |
|---|---|---|---|
| 로봇 충돌 구 모델: 손 (손가락 덮개 + 손바닥 껍질 구 5 mm) | 채택 | [AG3S_COLLISION_MODEL_DECISION_20261028_KO.md](AG3S_COLLISION_MODEL_DECISION_20261028_KO.md) | 2026-10-28 |
| 로봇 충돌 구 모델: 팔 구 모델링 방법 (내접 덮개) | 채택 | 같은 문서 §3 | 2026-10-28 |
| 로봇 충돌 구 모델: 팔을 serving 에서 켤지 | 판정 대기 (lead 추천: 켬) | 같은 문서 §4 | 2026-10-28 |
| 로봇 충돌 구 모델: 미채택 6 건 (MorphIt · capsule 1.0 배 · 팔 5 mm 넘침 · palm cover · 손목 범위 · 껍질 구 10 mm 여유) | 미채택 | [NOT_ADOPTED_KO.md](NOT_ADOPTED_KO.md) §1 | 2026-10-28 |
| 핵심 평가 전 채택 7 건 (BLAS thread 1 · GPU 여러 초기값 · 손바닥 껍질 구 · 쥐기 목표 항 닫는 행만 · 쥐기 때 꼭지 덮개 · 쥐기 때 가벼운 이웃 접촉 허용 · 로봇 지우기는 구 mask 만, 렌더 mask 끔) | 채택 (꼭지 덮개 검증 전, 이웃 접촉 구현 전) | [ADOPTED_KO.md](ADOPTED_KO.md) | 2026-10-28 |
| TrajOpt 비용 · 경로, target 제외, 렌더 로봇 mask: 미채택 9 건 | 미채택 | [NOT_ADOPTED_KO.md](NOT_ADOPTED_KO.md) §2 | 2026-10-28 |
| 미해결 · 추가 실험 9 건 (꼭지 덮개 검증 · 이웃 접촉 구현 · 목표 항 조건 · 회귀 119 · 렌더 mask 문서 불일치 등) | 판정 대기 · 실험 필요 | [OPEN_ISSUES_KO.md](OPEN_ISSUES_KO.md) | 2026-10-28 |
| 논문 방법론 채택: 지금 AG3S (A1–A4 + 구 mask), 핵심 평가 1차 근거 | **채택** (사용자 2026-10-29 01:05) | [ADOPTED_KO.md](ADOPTED_KO.md) A8 (최종 구성 · flag · 코드 상태) · [AG3S-CORE-EVALUATION task](../../../../AG3S/ag3s/docs/handoff/AG3S-CORE-EVALUATION.task.md) §0.1 · `outputs/2026-10-28_핵심평가_모듈별_가림기둥3_회귀6_x2/` | 2026-10-29 |
| target 찾기 · 쥐기 상태 판단 (세션 B): 결과 전에 고정한 attention 선택 절차 · attach reach 기준점 결함 수정 대상 · 종료 구간 보조 평가 · HSR 6 단계 사용 | 채택 (HSR 사용 방식은 판정 대기) | [AG3S_IDENTITY_PHASE_SOURCE_DECISION_20261029_KO.md](AG3S_IDENTITY_PHASE_SOURCE_DECISION_20261029_KO.md) | 2026-10-29 |
| 기하 grasp 보정 (AG3S-GEO-GRASP 세션): VLA 닫기 근처 보정 채택 1 건 · 미채택 6 건 (접근 목표 · 닫기 gate · 중심 통과 검사 · 장애물 검사 · TO 거리장 검사 · 타겟 쪽 유도) | 채택 1 · 미채택 6 | [AG3S_GEO_GRASP_DECISION_20261029_KO.md](AG3S_GEO_GRASP_DECISION_20261029_KO.md) | 2026-10-29 |
