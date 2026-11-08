# 원격 SL9–SL16 와 SL-GTTO (F1–F7) 비교 — 어떤 방식으로 갈 것인가 (2026-11-07 KST)

> 읽는 사람: 사용자 (방향 판정). writer: lead. 두 작업은 같은 base `89b42459` 에서 갈라져 2026-11-06 merge `ff21caa5` 로 합쳤다.
> 출처: 원격 = `doc/active/experiments/SL_AUTONOMOUS_DECISIONS_20261105_KO.md` · `_20261106_KO.md` (E## · F## · G##), handoff SL9–SL16;
> 우리 = `doc/active/experiments/E1_GTTO_DECISIONS_20261104_KO.md` (G##), `E1_GTTO_REPORT_20261105_KO.md`.
> 원격 run 은 다른 기계에서 돌아 그 run 폴더가 이 기계에 없다 — 원격 수치는 문서에 적힌 값을 옮긴 것이다.

## 1. 같은 목표, 다른 고정 조건

| | 원격 (SL9–SL16, 11-04 10:33 – 11-06 14:41) | 우리 (SL-GTTO, 11-04 – 11-05 21:00) |
|---|---|---|
| 질문 | AG3S 전체 (추정 인식 + TO) 를 benchmark 에서 기준 이상으로 | **인식을 GT 로 고정**하고 공용 TO 가 §5 를 통과하는가 |
| 기준 판 | pc판 `AG3S-final-panda-v3-e0to-pc` (attention ball-owned 추정, conversion C) · 표면 판 | `AG3S-commonto-gt*` (GT code image grounding + oracle extent) |
| 고친 층 | 거리장 (target · 쥔 물체 · 손 voxel), TO 항 · 문턱, action 한계 제약, conversion B/D/E, latch | admissibility · TSDF 손안 제외, 손 행 해제 (zr), pass-through, **정책 action 공간 QP (F7)** |
| 평가 | SL14 부터 LIBERO 27 (seed 7·8·9), LS 10, SafeLIBERO 11; 기제 시험 먼저 | seed 7 장면당 1 episode (LIBERO 9 · SafeLIBERO 11 · LS 10) |
| 규칙 | 결과 전 고정, 후보 = LIBERO ≥ 17/27 ∧ LS ≥ 6 · 위반 0 ∧ SafeLIBERO 충돌 ≤ 5 · 성공 ≥ 1 | 결과 전 고정, §5 (LIBERO ≥ 8 · SafeLIBERO 충돌 ≤ 6 ∧ 안전 성공 ≥ 5 · LS ≥ 5 ∧ 위반 ≤ 1) |

## 2. 결과 (각자 문서의 값)

| 판 | LIBERO | SafeLIBERO 안전 성공 (충돌) | LIBERO-Safety (위반) | 출처 |
|---|---|---|---|---|
| VLA 단독 (원격) | 9/9 · 27/27 | 2 (9) | **10/10 (0)** | E27 · E44 |
| VLA 단독 (우리) | 9/9 | 2 (9) | **7/10 (1)** | gt10_ee 결과표 |
| 원격 pc판 | 6/9 · 10/27 (6·2·2) | 1 (5) | 1 (0) | E27 · E42 |
| 원격 최고 LIBERO: conversion B | 19/27 | 충돌 9 (보호 상실) | 8 (0) | E46 · E51 · F2 |
| 원격 SL16 HSC | 17/27 (6·6·5) | 2/8 | 2 (0) | G5–G7 |
| 원격 GT target 진단 SL12 / SL13 | 3/9 · 7/9 | – | 3 · 5 (0) / – | E38 · E40 |
| 우리 zr (F4) | 6/9 | 4 (5) | 4 (1) | G44 |
| **우리 zr-ee (F7)** | **8/9** | **6 (3)** | 3 (0) | G49 · G50 |

**LIBERO-Safety 의 VLA 단독이 10/10 과 7/10 으로 다르다.** 장면 구성 · checkpoint · 응답 모델 중 무엇이 다른지 문서만으로는 확정 못 했다 —
LS 수치는 같은 기계 · 같은 장면으로 다시 맞추기 전에는 두 쪽을 직접 비교하지 않는다.

## 3. 두 쪽이 독립적으로 같은 결론에 닿은 것

1. **conversion C 증폭.** 원격 E45: β 0.0706 → 경로 1 mm = action 0.28, 3.5 mm 에서 포화, 쥔 call 포화 0.91. 우리 G47: 3.6 mm 당 action 1.
2. **action 한계를 hard 로 걸면 실현 불가능한 call 이 HOLD 가 된다.** 원격 `--to-action-box` (joint QP 에 action 행 추가): LIBERO HOLD 16 → 143, pc판 6 → 3/9 (E8 · E13, "HOLD ≠ 보호").
   우리 F7 (action 공간 QP): LIBERO · SafeLIBERO 는 좋아졌지만 LS 쥔 후 HOLD 0.03 → 0.356, 충돌 행의 step 0 (= 지금 자세가 이미 위반) 188 (G50).
3. **GT 인식만으로는 원래 TO 가 통과하지 못한다.** 원격 SL12: target 을 다 지워도 (남은 비율 0.02) 3/9 (E40). 우리 v1 (GT + 원래 공용 TO): LIBERO 2/9 (G21).
   → TO 층 수정이 먼저 필요하다는 것을 양쪽이 보였다.
4. **쥔 물체 · 손 기하가 쥔 후 HOLD 의 주인.** 원격: 부착 구 ↔ bowl 자기 voxel (E16), 손 voxel 거짓 기하 (G3). 우리: `attached_point` · `nearest_is_not_manipulated` (G44 · G50).

## 4. 다른 점과 각 방식의 위험

| | 원격 방식 | 우리 방식 |
|---|---|---|
| 강점 | 실제 AG3S (추정 인식) 그대로 평가; **seed 흔들림을 재고 (pc판 6·2·2, R = 4, E42) 27 episode 로 바꿈**; 기제 시험 먼저; offline 재풀이 · geom 분류 도구 | 인식 오류를 빼서 TO 층만 가를 수 있음; F7 로 conversion 포화를 구조적으로 없앰 (포화 0.9 → 0); SafeLIBERO 보호와 성공을 함께 얻음 (6 (3)) |
| 위험 | 여러 층 (거리장 · latch · conversion · TO) 을 같은 판 위에서 바꿔 원인 층 분리가 어려움; 후보 판 없음 | **seed 1 개 · 장면당 1 episode** — 원격이 잰 R = 4 안에서 6 → 8 은 확정 아님; GT 전제라 실제 AG3S 성능이 아님; OSQP 미수렴 · LS deadlock 미해결 |

## 5. 추천 — 안정적으로 갈 수 있는 순서

1. **평가 규약은 원격 것을 쓴다:** LIBERO 27 (seed 7·8·9), 기제 시험 먼저, 후보 규칙 (a)–(c). 우리 SL-GTTO 결과도 이 규약으로 다시 잰다.
2. **층을 나눠 한 번에 하나씩:** (i) GT 인식으로 TO 층 확정 → (ii) 그 TO 위에서 추정 인식 (원격 HVC · HSC 계열) 을 바꾼다.
   원격 SL12 · 우리 v1 이 보였듯 TO 가 먼저 막히면 인식 개선의 효과를 읽을 수 없다.
3. **TO 후보 = F7 (`zr-ee`), conversion 은 C 유지.** 원격 F6: 변환만 바꾸면 성공 ↔ 보호를 맞바꾼다 (C 10 / 충돌 5, D 15 / 7, B 19 / 9).
   F7 은 같은 C 위에서 action 공간으로 풀어 SafeLIBERO 보호를 지켰다. 단 채택 전에:
   (a) OSQP scaling 수정 (offline 150 회 · 5 ms, G50), (b) 지금 자세가 이미 위반일 때의 회복 규칙 (이산 CBF `h_k ≥ (1 − γ) h_{k−1}`, 원격 · 우리 공통의 "hard 한계 → HOLD" 문제),
   (c) 세 벤치를 27 / 10 / 11 규약으로 재측정.
4. **먼저 맞출 것:** LIBERO-Safety VLA 단독 기준 (10/10 vs 7/10) 을 같은 기계 · 같은 장면 · 같은 checkpoint 로 맞춘다.

## 6. 사용자 판정이 필요한 것

1. 위 순서 (평가 규약 통일 → GT 로 TO 확정 → 추정 인식) 로 갈지.
2. F7 + (a) + (b) 를 새 판으로 만들어 27 episode 로 잴지 (GPU 독점 대략 LIBERO 27 + SafeLIBERO 11 + LS 10 = 48 episode, 판당).
3. 원격 SL16 §6 남은 일 (HVC seed 9 · LS · SafeLIBERO, HSCREC) 을 TO 확정 전에 마칠지, 뒤로 미룰지.
