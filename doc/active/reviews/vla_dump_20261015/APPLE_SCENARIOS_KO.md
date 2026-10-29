# 사과 target 시나리오 정리 (기록 기반)

작성: 2026-10-19 KST. 사용자 지시: "다른 과일은 성공 1회만 쓰고 사과 위주로. 기록에 성공 시나리오 정리가 있을 것."
기존 기록에서 사과가 target인 closed-loop 결과를 episode별로 모았다. **새 rollout은 하지 않았다.**
수치 원본: [apple_scenarios.json](apple_scenarios.json), 집계 script: `AG3S/ag3s/experiments/studies/ifg_apple_scenarios.py`.

## 출처

| 출처 | 무엇 | 조건 |
|---|---|---|
| T39 · T42-1 | test split(1800–1999) 장면, VLA 단독(E0)과 당시 AG3S(E3b), seed 3개 | 장애물 없음 |
| T43 LB | train split의 덜 붐비는 장면, VLA 단독 선별 | 장애물 없음 |
| T44 LB2 | train split의 붐비는 장면(이웃 과일 < 10 cm), VLA 단독 두 seed 모두 성공 + 단단히 쥠 | 장애물 없음 |
| T43 V11 · IFG-HURDLE | 사과 운반 경로 위 허들(T43 P2 배치), 7 episode × 2 seed, 여러 AG3S 구성 | 허들 |
| IFG-CL1 | 현재 serving(C1a)이 성공한 held-out 장면에서 C1a · 기존 방식 · 표면 영역 방식 | 없음 / 옆 장애물 / place 장애물 |
| IFG-CL3 | 선별 없는 재평가의 사과 장면, 기존 방식 · 표면 영역 방식 | 같음 |

- 칸의 값은 "성공 run / run 수"다. 출처마다 성공 정의가 조금 다르다. T42-1 · LB는 잡고 놓기 성공, IFG는 접촉 없는 작업 성공이다
- 허들 칸은 여러 AG3S 구성(T43 V8 · V11AB, IFG의 C1a · 기존 방식 · 표면 영역 방식)을 합친 값이다

## tier (기록 성공으로 고른 것 — 시연 · figure · 사슬 탐색용이며 성공률 평가용이 아니다)

| tier | 기준 | episode |
|---|---|---|
| A | VLA 단독 ≥ 2/3 이고 AG3S(장애물 없음) 전부 성공 | 1828 · 1834 · 1925 · 1939 · 1968 · 1977 (모두 test split) |
| B | 허들에서 ≥ 80 % 성공 | 1807 · 1982 · 1995 |
| C | VLA 단독 전부 성공 (AG3S 기록 없음) | train: 12 · 30 · 40 · 54 · 228 · 274 · 394 · 745 · 1031 |
| D | IFG-CL1에서 세 구성 모두 성공 (VLA 단독 기록 없음) | 19 · 64 · 204 · 311 · 583 · 729 · 1245 · 1419 |

- 1925 · 1968은 VLA 단독, AG3S, 허들까지 모두 높다. 최종 사슬(① VLA 단독 → ② AG3S → ③ 허들 → ④ 허들 + pick 장애물)의 1순위 후보다
- 1834는 허들 성공이 6/10으로 낮다

## 전체 표

| tier | episode | 사용 팔 | split | VLA 단독 | AG3S (장애물 없음 또는 옆 장애물) | AG3S + 허들 | 출처 |
|---|---|---|---|---|---|---|---|
| - | 8 | left | train | 2/3 | - | - | T43-LB |
| - | 39 | left | train | 0/1 | - | - | T43-LB |
| - | 165 | right | train | 0/1 | - | - | T43-LB |
| - | 194 | left | train | - | 1/2 | - | IFG-CL3 |
| - | 219 | left | train | - | 2/2 | - | IFG-CL3 |
| - | 232 | left | train | - | 1/2 | - | IFG-CL3 |
| - | 321 | right | train | - | 2/2 | - | IFG-CL3 |
| - | 423 | left | train | - | 0/2 | - | IFG-CL3 |
| - | 434 | left | train | 1/2 | - | - | T43-LB |
| - | 444 | left | train | - | 2/2 | - | IFG-CL3 |
| - | 447 | left | train | - | 0/2 | - | IFG-CL3 |
| - | 526 | left | train | 0/1 | - | - | T43-LB |
| - | 622 | left | train | 1/2 | - | - | T43-LB |
| - | 623 | left | train | - | 2/3 | - | IFG-CL1 |
| - | 694 | left | train | 1/2 | - | - | T43-LB |
| - | 831 | left | train | - | 1/2 | - | IFG-CL3 |
| - | 1222 | left | train | 0/1 | - | - | T43-LB |
| - | 1286 | right | train | - | 1/2 | - | IFG-CL3 |
| - | 1407 | left | train | - | 0/2 | - | IFG-CL3 |
| - | 1512 | left | train | 1/2 | - | - | T43-LB |
| - | 1854 | right | test | - | 3/3 | - | T42-1 |
| - | 1863 | left | test | - | 2/2 | - | IFG-CL3 |
| - | 1967 | left | test | - | - | 5/10 | IFG-HURDLE, T43-V11 |
| - | 1972 | right | test | - | 0/1 | - | T42-1 |
| A | 1828 | left | test | 2/2 | 3/3 | - | T39, T42-1 |
| A | 1834 | right | test | 2/3 | 3/3 | 6/10 | IFG-HURDLE, T42-1, T43-V11 |
| A | 1925 | left | test | 3/3 | 3/3 | 9/10 | IFG-HURDLE, T39, T42-1, T43-V11 |
| A | 1939 | left | test | 3/3 | 3/3 | - | T39, T42-1 |
| A | 1968 | left | test | 3/3 | 3/3 | 10/10 | IFG-HURDLE, T39, T42-1, T43-V11 |
| A | 1977 | right | test | 3/3 | 3/3 | - | T42-1 |
| B | 1807 | left | test | - | - | 8/10 | IFG-HURDLE, T43-V11 |
| B | 1982 | left | test | - | - | 8/10 | IFG-HURDLE, T43-V11 |
| B | 1995 | left | test | - | - | 9/10 | IFG-HURDLE, T43-V11 |
| C | 12 | left | train | 2/2 | - | - | T44-LB2 |
| C | 30 | right | train | 2/2 | - | - | T44-LB2 |
| C | 40 | right | train | 3/3 | - | - | T43-LB |
| C | 54 | right | train | 2/2 | - | - | T44-LB2 |
| C | 228 | right | train | 3/3 | - | - | T43-LB |
| C | 274 | right | train | 3/3 | - | - | T43-LB |
| C | 394 | right | train | 3/3 | - | - | T43-LB |
| C | 745 | left | train | 3/3 | - | - | T43-LB |
| C | 1031 | left | train | 3/3 | - | - | T43-LB |
| D | 19 | right | train | - | 3/3 | - | IFG-CL1 |
| D | 64 | right | train | - | 3/3 | - | IFG-CL1 |
| D | 204 | left | train | - | 3/3 | - | IFG-CL1 |
| D | 311 | right | train | - | 3/3 | - | IFG-CL1 |
| D | 583 | right | train | - | 3/3 | - | IFG-CL1 |
| D | 729 | right | train | - | 3/3 | - | IFG-CL1 |
| D | 1245 | left | train | - | 3/3 | - | IFG-CL1 |
| D | 1419 | right | train | - | 3/3 | - | IFG-CL1 |
