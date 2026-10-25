# AG3S-PALM-COVER-1345 — 손바닥 mesh 에 맞춘 구 (palm cover) 로 1345 오렌지 접촉을 막으면서 쥐기를 지키는가

> writer: lead · 실행: lead 직접 (순차 작업 — 단계 1 결과가 단계 2 진행을 정한다)

작성: 2026-10-25 18:1x KST (`date -u` 09:1x UTC + 9 h), lead. **조건 · 지표 · 판정은 결과를 보기 전에 고정한다.**
앞선 STEP: `AG3S-NBC-1345` (원인 = 구 모델이 손바닥을 덮지 못함, URDF 굵기 손바닥은 쥐기를 잃음). 사용자 판정 2026-10-25: "권장하는 대로 수행" =
손바닥 mesh 에 맞춘 구, 기본 꺼짐 flag, offline 덮임 확인 → 1345 → 18 장면.

## 0. 이미 있는 구현을 쓴다 (새 코드 없음)

| 사실 | 출처 |
|---|---|
| `--palm-cover on` (T45 A10, 기본 off): 손바닥 (`ee_<side>`) 의 구를 MJCF 손바닥 collision mesh 겉면에 다시 만든다 (overflow 5 mm). 손바닥 겉면 표본의 구 밖 비율 0.946 → 0.00028 (T46f A14 audit) | `grounding_report.build_constraint_robot_model` docstring, `AG3S_T47_LOG.md` 4651 줄 |
| 위험 (이미 기록됨): CL4 (palm cover + 손목 구 + 다른 변경 동시) 에서 손바닥 구가 target-free 층의 최악 행이 된 chunk 가 CL3 21 → 109 / 95 로 늘고 성공이 낮았다. 형상만의 효과를 가르는 X2G 의 결과는 기록에 없다 | `AG3S_T47_LOG.md` 5256–5257 줄, `handoff/CL4.task.md` |

## 1. 단계 1 — offline 덮임 확인과 1345 closed-loop

- offline (`sphere_coverage_check.py`, rollout 아님): 요청 12–16 에서 접촉 부위가 구 안에 드는지, 손바닥 구 → 사과 거리. 변형 = 지금 / URDF 굵기 / palm cover / `--links gripper_wrist` / `gripper_wrist` + palm cover.
- closed-loop: **palm cover 구성** = E4 serving 구성 (GPU 판정) + `--palm-cover on`, 1345 · seed 13453, 1 episode.
- **단계 1 통과 = 손–오렌지 접촉 0 row 그리고 성공.** 둘 중 하나라도 아니면 단계 2 로 가지 않고 결과를 보고한다.

## 2. 단계 2 — 18 장면 (단계 1 통과 시)

- 장면 · seed: E2 · E3 · E4 와 같은 VLA 단독 성공 사과 18 장면. 기준 = E4 기록 (GPU 판정과 bit 동일, `AG3S-ROI-CARVE-GPU` G3).
- 판정 (모두 지켜야 채택 제안):

| 기준 | 내용 |
|---|---|
| P1 | 성공 수 ≥ 기준 (15 / 18) |
| P2 | 금지 접촉 episode ≤ 기준 (1) |
| P3 | 손–이웃 과일 접촉 row 합이 기준보다 줄어든다 (기준 = E4 의 18 장면 합) |
| 기록만 | TrajOpt 풀이 시간 (구 수가 는다), 서버 시작 시간, 손바닥 구가 최악 행이 된 chunk 수 |

통과해도 serving 기본값은 사용자 판정으로 바꾼다. 실패한 장면은 ATS · HSR · LTE · CCR 중 원인 위치를 적는다.

## 3. 산출물

- 원시: `$RUN/verify/AG3S-PALM-COVER-1345/` (run `20261025T035831+0900_8da37119afd3_f082f20e9463`)
- `AG3S/ag3s/docs/handoff/AG3S-PALM-COVER-1345.verify.json`, 그림 `AG3S/ag3s/docs/figures/palm-cover/`
- 로그: `AG3S/ag3s/docs/AG3S_TSDF_DYNAMIC_LOG.md` §15
- 보기: `outputs/2026-10-25_손바닥구_palmcover/`

## 4. 사용자 판정 (2026-10-25, 결과 뒤)

"권장 방안으로 넘어갑시다" → **1345 의 손바닥–오렌지 접촉은 알려진 한계로 기록하고 동적 거리장 축을 마무리한다. 다음은 핵심 평가
(`AG3S-CORE-EVALUATION`, 핵심 평가 세션 담당).** 핵심 평가 보고에는 "palm cover 없는 구성 (`gripper` · `gripper_wrist` 모두) 은 손바닥
mesh 의 접촉을 제약에서 보지 못한다 — 1345 에서 손바닥 몸체가 이웃 과일에 닿는 것은 VLA 단독에도 있는 접촉이다" 를 명시한다.

### 추후에 해 볼 방법 (사용자 지시로 정리, 지금은 실행하지 않는다)

| # | 방법 | 묻는 것 | 설계 초안 (실행 전에 다시 고정) | 비용 · 위험 |
|---|---|---|---|---|
| F1 | **palm cover 를 18 장면에 돌린다** | 1345 만 예외인가 — 다른 장면에서는 손바닥 접촉을 줄이면서 쥐기를 지키는가 | E4 serving 구성 + `--palm-cover on`, E2–E4 와 같은 VLA 단독 성공 사과 18 장면 · 같은 seed, 기준 = E4 기록. 주 지표: 성공 수 · 금지 접촉 episode · 손–이웃 과일 접촉 row 합 · 손바닥 접촉 row. 기록: TrajOpt 수정량 · 풀이 시간 (구 270 → 466) · 서버 시작 시간 | 18 episode · 서버 2 대 ≈ 1 h. 이번 단계 1 의 "실패면 중단" 규칙을 바꾸는 것이므로 사용자 판정 필요. CL4 에서 palm cover 를 포함한 구성의 성공이 낮았던 기록 (`AG3S_T47_LOG.md` 5256) 이 있다 |
| F2 | **palm cover + 손바닥에도 작은 이웃 접촉 허용** | 손바닥을 정확히 보되, 쥐기 직전 손바닥이 이웃 물체를 수 mm 스치는 것을 허용하면 쥐기를 지키면서 겹침 깊이를 제한할 수 있는가 | 지금 IFG-NBC 허용 (`--neighbour-contact-mm`, 손가락 구만, target 120 mm 안, latched · closing) 을 손바닥 구로 넓히는 새 flag (기본 꺼짐), 깊이 후보 2 · 5 mm. 1345 1 episode → 통과하면 18 장면. 주 지표: 손–오렌지 최대 겹침 · 접촉 row, 성공 | 구현 필요 (TrajOpt `linearize` 의 neighbour-contact 행 선택에 손바닥 link 추가). 허용이므로 접촉 자체는 남을 수 있다 — "막기" 가 아니라 "깊이 제한" 이다 |
