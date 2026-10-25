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
