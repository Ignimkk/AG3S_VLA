# AG3S-CORE-EVAL-A6 — 쥐는 순간의 손바닥 이웃 접촉 허용 (A6) + HSR 규칙 ② 수정

> writer: lead · 구현: ag3s-implementer · 작성: 2026-10-29 11:40 KST. **허용 범위 · 검증 기준은 결과 전에 고정한다.**
> 상위: [ADOPTED_KO.md](../../../../doc/active/reviews/AG3S_decision/ADOPTED_KO.md) "A8 최종 구성 확정", [AG3S-CORE-EVAL-HSR6](AG3S-CORE-EVAL-HSR6.task.md).

## 0. 사용자 판정 (2026-10-29 ~11:30 KST)

1. 최종 구성 = A8 + 기하 grasp 보정 (G1) + HSR 6 단계 (①–④) + A6.
2. HSR 규칙 ② 는 세션 B (T66) 제안대로 고친다.
3. A6 를 켠다. **쥐는 순간만** 허용한다. 허용 범위를 고정해 구현한다.

## 1. A6 — 근거

- 장면 119 (`outputs/2026-10-29_장면119_손바닥이웃오렌지_쥐기실패_진단/`):
  - 쥐기 직전 요청 11–14 (손바닥 – 사과 30–55 mm) 에서 TrajOpt 가 손바닥 (`ee_left`, 껍질 구) 과 가장 가깝게 본 장애물은 오렌지였다. 오렌지는 사과에서 22 mm 떨어져 있었고, 여유 5 mm 가 걸렸다.
  - TrajOpt 가 손바닥을 VLA 보다 6–10 mm 비켜 옮겼다. 사과가 손가락 사이 중심에서 21–28 mm (VLA 15 mm) 벗어나 쥐지 못했다.
  - 실제 오렌지 접촉은 0 이었다.
- 지금 켜진 손가락 이웃 접촉 허용 (`--neighbour-contact-mm 20`, IFG-NBC) 은 손가락 구만 풀어 준다. 걸린 것은 손바닥이다.

## 2. A6 규칙 (결과 전 고정)

| 항목 | 고정 값 |
|---|---|
| flag | `--neighbour-palm-contact-mm MM` (기본 꺼짐). 최종 구성에서 MM = **10** |
| 언제 ("쥐는 순간") | 쥐는 손이 아직 쥐지 않았고 (grasp latch ∈ {LATCHED, CLOSING}, HELD 아님), 손바닥 (hold link site) 이 target 중심 (manipulated centroid) 에서 **60 mm 안**. 쥔 뒤 (HELD) · 운반 · 놓기 · 놓은 뒤에는 꺼진다 |
| 어느 구 | 쥐는 손의 손바닥 구 (`ee_<side>` link 의 손 몸체 구 + 손바닥 껍질 구). 손가락 구는 지금 IFG-NBC 가 따로 다룬다 |
| 무엇과 | 이웃 장애물: target 이 아닌 과일 · 장애물 (기둥 · 막대). 그 표면이 target 중심 120 mm 안 (IFG-NBC 와 같은 범위). **탁자 · 바구니 · target 은 대상이 아니다** (지금 규칙 그대로) |
| 얼마나 | 그 행의 요구 여유를 "≥ 껍질 구 여유 5 mm" 에서 **"≥ −10 mm" (10 mm 겹침까지)** 로 푼다 |
| HSR6 ④ 와의 관계 | A6 는 자기 "쥐는 순간" 창 (latch + 60 mm) 을 쓴다. ④ (HSR6 = grasp 일 때만) 로 막지 않는다. 이유: 결합 시험에서 닫기 전 HSR6 는 대부분 approach 였다 (approach 438 · grasp 32 요청). ④ 로 막으면 장면 119 의 요청 11–14 에서 A6 가 작동하지 않는다 |
| 평가 | 금지 접촉 정의는 바꾸지 않는다. 실제 이웃 접촉은 그대로 실패로 센다 |
| 기록 | 요청마다 `summary_json` 에 A6 창 여부 · 풀린 행 수 · 그 행의 최소 여유를 남긴다 |

- 10 mm 근거: 장면 119 에서 TrajOpt 가 손바닥을 옮긴 6–10 mm 를 덮는 최소 값이다. 손가락 허용 20 mm 의 절반이고, 껍질 구는 mesh 밖으로 5 mm 넘쳐 있다.
- 60 mm 근거: 장면 119 의 제약 요청 (손바닥 – 사과 30–55 mm) 을 포함하는 최소 값이다. IFG-NBC 의 150 mm 보다 좁게 해 "쥐는 순간" 으로 제한한다.

## 3. HSR 규칙 ② 수정 (세션 B T66 제안, 결과 전 고정)

- 지금: HSR6 ∉ {lift, transport, placement} 이면 쥔 물체 들어올림 허용을 거둔다. 그래서 쥔 직후 HSR6 = grasp 인 1–3 요청에서도 거둔다.
- 수정: **HSR6 ∈ {approach, retreat} 일 때만 거둔다.** HSR6 = grasp 이고 latch 가 HELD 면 허용을 유지한다. 나머지 (lift · transport · placement) 는 지금 규칙 그대로다.
- 근거 (`outputs/2026-10-29_HSR6_쥔직후_궤적급변_원인/`):
  - 거둔 요청 14 개 수정 중앙 6.2° (5° 초과 8, multi-seed 6). 허용 156 개 0.76°.
  - 장면 121 에서 쥔 직후 팔이 꺾였다.

## 4. 검증 (결과 전 고정)

| # | 검사 | 통과 조건 |
|---|---|---|
| A1 | 단위 test | A6 창 참 · 거짓, 대상 구 · 대상 물체 (탁자 · 바구니 · target 제외), −10 mm 경계, ② 수정 (grasp ∧ HELD 유지, approach · retreat 거둠) |
| A2 | 꺼짐 동등성 | A6 flag 를 끄고 ② 수정 전 코드와 같은 flag 로 core · 결합 시험 기록 요청 ≥ 20 재생 → refined bit 동일. ② 수정은 `--hsr6-held-gate` 가 켜질 때만 동작이 바뀐다 |
| A3 | 장면 119 재생 (offline) | 결합 시험 HSR k0 기록의 요청 10–14 를 A6 on 으로 재풀이한다. A6 창이 켜지고, 손바닥 이동 (refined − VLA) 이 A6 off 보다 줄어드는지 숫자로 보고한다 (통과 기준 아님, 기전 확인) |

## 5. 다음 (lead)

- 최종 구성 확인 closed-loop: 같은 9 장면 × 2 = 18 run. 비교는 A8 1 차 · 결합 시험.
- 놓는 곳 허들 장면 AG3S 14 run (VLA 단독 0 / 14 와 비교).
- 주 지표 · 읽는 규칙은 §0.1 · §0.2 와 같다.
