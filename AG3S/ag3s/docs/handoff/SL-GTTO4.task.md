# SL-GTTO4 — 쥔 물체가 아래 표면에 닿는 것을 허용 (F3 `--held-beneath-contact`)

> writer: lead, 2026-11-05 07:2x KST. 사용자 위임(2026-11-05 ~01:00 KST, "오전 11:30까지 스스로 판정하며 수행")에 따른 lead 자율 판정 G29 · G30.
> **결과를 보기 전에 고정.** 선행: [SL-GTTO3.task.md](SL-GTTO3.task.md), [SL6.task.md](SL6.task.md) §11 (SL6-H).

## 0. 근거 — gt2 30 장면 본 실행 (2026-11-05 05:31–07:00) 기록 + sim 진실

| 관찰 | 수치 | 출처 |
|---|---|---|
| held call에서 VLA reference가 attached 질의 구 바닥을 지지면 평면 15 mm 안으로 가져감 | gt2 LIBERO: t0e0 11/20, t1e0 4/7, t2e0 10/15, t6e0 3/5, t8e0 3/6 call | `calls.npz` `joint_reference` FK + `grasp.attached.held.spheres_parent_frame` + `support.plane` (scratchpad `held_floor.py`) |
| 그때의 TO 경로 수정 | `reference_deviation` 중앙값 0.18–0.29 (그 밖의 held call 0.02–0.06) | 같은 표 |
| 모델 바닥 − 실제 물체 바닥 (row 0 자세) | 중앙값 −15 … −100 mm (v1 · gt2 대부분 episode) | `geom.npz` `sim_state` 물체 위치 (scratchpad `model_err.py`) |
| `attached_point` HOLD | LIBERO t1e0 14 · t3e0 5, SafeLIBERO t2e3 14 call | `gtto_aggregate.py` 표 |

원인 후보 "attach snapshot 시점" 은 반사실 계산(finger 정지 frame 자세로 snapshot)에서 오차가 줄지 않아 기각 (G29).
같은 증상(놓기 하강을 TO가 위로 뒤집음)은 SL6 E0판 진단에서 실패 10 중 5였고, SL6-H 가 placement-contact 행 규칙을 넣었으나 HSR6 분류기가 plate 위 call을 대부분 `transport` 로 판정해 거의 켜지지 않았다 (로그 SL6-H 절).

## 1. 변경 (기본 꺼짐)

- `--held-beneath-contact` (serve_safe → `SafePolicy.held_beneath_contact`): latch HELD ∧ attach 된 쥔 물체 ∧ 거리장 → `SceneSnapshot.placement_contact = hsr6.placement_contact_spec()`.
  그 뒤는 SL6-H 그대로 (`CollisionLinearizer._placement_contact_margin`): 쥔 물체 자신의 질의 행 중 **최근접 표면이 아래**(기울기 위로 45° 안)이고 **쥔 물체 xy 발자국 + 1 voxel 안**인 행만 margin −1 m. 손 · 손가락 · 팔 행, 옆 · 위 표면은 그대로.
- 판정(verdict)도 같은 snapshot으로 분류하므로 풀린 행은 HOLD 사유가 되지 않는다.
- 기록: `to.held_beneath_contact {latch, attached, applied, n_held_rows, rows_released, held_rows_released, min_clearance_released_m}`.
- 판: `AG3S-commonto-gt2-hbc` = gt2 + F3, `AG3S-commonto-gt2-termsoff-hbc` = gt2-termsoff + F3.
- **추가 (07:4x, F3 결과 전, G31):** `AG3S-commonto-gt2-termsoff-pass-hbc` = gt2-termsoff-pass + F3 (SL-GTTO3 LIBERO 최고 판 위).
- test: `tests/trajopt/test_sl_gtto4_held_beneath.py`.

## 2. 장면 · 지표

SL-GTTO3 와 같다: LIBERO spatial task 0–4, 6–9 × init 0, seed 7, conversion C. 주 지표: 공식 성공 (9 중). 진단: held call 중 `applied` · 행을 푼 call 수, held 구간 `reference_deviation`, `attached_point` HOLD, C 포화 call 비율.

## 3. 판정 (결과 전 고정)

1. **비교 짝:** hbc ↔ gt2 (2/9), termsoff-hbc ↔ termsoff (3/9), termsoff-pass-hbc ↔ termsoff-pass (4/9) (같은 기계, 같은 commit + patch).
2. **넓히기:** 세 hbc 판 중 LIBERO 성공이 높은 판(같으면 포화 비율이 낮은 판)이 자기 짝보다 **2 episode 이상** 높으면, 그 판으로 SafeLIBERO 11 · LIBERO-Safety 10 을 돌려 SL-GTTO §5 전체를 판정한다.
   SafeLIBERO 에서는 F3 의 위험(쥔 물체 아래 장애물도 풀림)을 충돌 수로 본다.
3. 그 미만이면 F3 는 GT 판의 실패를 설명하지 못한다고 기록하고, 남은 실패를 층별로 분류만 한다.
4. 공용 TO 정의에 F3 를 넣는 것은 **사용자 판정 대상**으로 남긴다.

## 4. lead 예측 (판정 규칙 아님, 사후 대조용)

- 되찾을 것: 놓기 단계에서 막힌 episode (gt2 t1e0 · t3e0 · t6e0 의 `attached_point` HOLD, held 구간 위쪽 수정).
- 못 바꿀 것: ④a 쥐기 전 (t4e0 · t9e0 서랍 · 캐비닛, attach 없음), t7e0 (attach 가 seq 43 으로 늦음 — 쥐기 이전 층).
