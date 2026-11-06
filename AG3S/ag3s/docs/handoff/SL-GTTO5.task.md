# SL-GTTO5 — 최소 개입 공용 TO + 쥐기 직전 접근 구간 손 행 해제 (F4 `--grasp-zone-release`)

> writer: lead, 2026-11-05 09:4x KST. 사용자 지시 (09:3x KST, Panda TO · 충돌회피 논문 조사 뒤): "니가 생각했을 때 최선의 방향으로 바꾸세요" → lead 판정 G34.
> **결과를 보기 전에 고정.**

## 0. 근거

| 근거 | 내용 |
|---|---|
| 문헌 (Panda safety filter · planner 13 편, 조사 09:0x–09:2x) | VLA safety filter(AEGIS · KNOWS · multi-link · VLPSA · OSCBF)는 모두 `min ‖u − u_nom‖²` 만 풀고 제약이 비활성이면 nominal 그대로 (AEGIS 명시). 평활 항이 있는 곳은 LIBERO-Safety 부록 chunk QP 하나. cuRobo `plan_grasp`: 마지막 직선 접근 구간에서 finger · hand ↔ world 충돌을 끈다. MoveIt MTC: hand–object 충돌 허용 + 직선 접근 |
| SL-GTTO3 | termsoff-pass (연속성 · 꺾임 변화율 항 끔 + 위반 없으면 통과) 가 LIBERO 4/9 로 최고. HOLD 82 → 14, C 포화 비율 0.48 → 0.22 |
| SL-GTTO3 남은 실패 | t4e0 · t9e0 (서랍 · 캐비닛 그릇): 쥐기 전 손 ↔ 용기 벽 `answer_excludes_target` −33 … −40 mm 와 link5–7 행 → attach 없음. t0e0 · t6e0: latched · closing 중 "changed" call (latched 32/114, closing 31/104) 뒤 빈손 닫힘 반복 → attach 없음 |
| 기존 grasp zone (pb5, `--grasp-zone-mm 10`) | target · 지지면 class 행에만 10 mm 허용. 용기 벽 · 이웃 물체 (obstacle class) 는 그대로 |

## 1. 변경 (기본 꺼짐)

- `--grasp-zone-release` (`SafePolicy.grasp_zone_release` → `SceneSnapshot.grasp_contact["zone_release"]` → `CollisionLinearizer._grasp_contact_margin`):
  latch LATCHED · CLOSING 에서, 쥐는 손의 **권한 link (palm + fingers)** 행 중 구 중심이 조작 대상 centroid 에서 **수평 `grasp_zone_radius` (100 mm, base argv 값)** 안인 행은 최근접 표면이 무엇이든 margin −1 m (묶이지 않음).
  팔 link (link1–7) 행, 영역 밖 행, 쥔 물체 행, 다른 latch 상태는 그대로. 판정(verdict)도 같은 snapshot 이라 그 행은 HOLD 사유가 되지 않는다.
- 기록: `to.grasp_contact.zone_release_rows`.
- 판: `AG3S-commonto-gt2-termsoff-pass-zr` = `AG3S-commonto-gt2-termsoff-pass` + F4.
- test: `tests/trajopt/test_sl_gtto5_zone_release.py` (6) + 관련 suite 293 passed / 5 skipped.

## 2. 실행 · 판정 (결과 전 고정)

1. queue: LIBERO 9 (SOLO) → SafeLIBERO 11 → LIBERO-Safety 10 (SOLO). 
2. **LIBERO 9 공식 성공 ≥ 5** (termsoff-pass 4 보다 높음) 이면 SafeLIBERO · LIBERO-Safety 줄을 그대로 돌려 SL-GTTO §5 세 규칙을 판정한다. 아니면 그 6 줄을 시작 전에 지우고 "F4 는 쥐기 전 실패를 설명하지 못함" 으로 기록한다.
3. F4 의 위험: 손 행이 target 근처의 **실제 장애물**도 무시한다 (SafeLIBERO 장애물이 target 100 mm 안이면). SafeLIBERO 충돌 수와 충돌 시점의 latch 상태로 본다.
4. 공용 TO 정의 교체는 §5 통과 여부와 함께 사용자에게 보고한다 (사용자 위임 "최선의 방향" 으로 lead 가 후보를 정하되, 채택 확정은 결과를 보고 사용자가 한다).

## 3. lead 예측 (판정 규칙 아님)

- 되찾을 것: t4e0 · t9e0 중 손 행이 막던 쪽 (단 link5–7 행이 남으면 못 함), t0e0 · t6e0 의 접근 수정.
- SafeLIBERO: 충돌이 termsoff-pass 보다 늘 수 있다 (장애물이 target 가까이 놓인 장면).
