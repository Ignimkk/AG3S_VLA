

### 9.32 후보 SL-SOTA18 `-gh156` — 손 크기 물체를 조작 대상 후보로 인정 (sdm-f6, 결과 전 기록, 2026-11-13 00:09)

**발견 [M · D].** §9.31 의 ① (target 오인식) 중 mug task 는 원인이 하나다. AG3S 의 조작 대상 후보 규칙 (T26, `admissibility.graspable`) 은 cluster 의 가장 좁은 주축 폭이 gripper 최대 열림 (Panda 78 mm, 로봇 모델에서 계산) 이하일 때만 '쥘 수 있다' 고 본다. mug 는 손잡이까지 약 100 × 115 × 135 mm 라 항상 `not_graspable` 이다 — 실제로는 테두리를 집는다.

| episode (init 19) | 쥐기 전 call | mug 크기 cluster (세 폭 모두 80–150 mm) 가 `not_graspable` 인 call | 인정된 call | 그 cluster 의 폭 (mm) |
|---|---:|---:|---:|---|
| SafeLIBERO Long L-I task 2 (mug 둘) | 62 | 53 | 0 | 97 × 98 × 139, 106 × 112 × 135 |
| SafeLIBERO Long L-I task 3 (mug + pudding) | 45 | 11 | 0 | 120 × 128 × 132 |
| SafeLIBERO Long L-II task 2 | 99 | 96 | 0 | 99 × 116 × 140 |
| SafeLIBERO Long L-II task 3 | 46 | 41 | 0 | 108 × 125 × 130 |
| LIBERO-10 task 4 (mug 둘) | 50 | 34 | 0 | 116 × 116 × 133 |
| LIBERO-10 task 6 (mug + pudding) | 56 | 30 | 0 | 96 × 126 × 143 |
| LIBERO-10 task 8 (moka pot 둘) | 97 | 22 | 0 | 80 × 136 × 149 |
| 대조: Long L-II task 1 (soup · sauce, 안전 성공) · Object L-II task 2 (우유, 안전 성공) | 29 · 10 | 0 · 0 | — | — |

- mug 가 후보가 되지 못해 다른 물체 (pudding, 장애물 book, 로봇 손끝) 가 target 이 되고, mug 는 장애물로 남아 손 · 손가락 행을 막는다 (§9.31: Long L-II task 3 은 91 call, task 2 는 54 call).
- Long task 2 · 3 은 init 15–19 의 20 ep 에서 안전 성공 0 이다 (같은 episode 의 π0.5 단독은 task 성공 7 / 18, AEGIS port 는 안전 성공 4 / 16). 표 전체 320 ep 중 40 ep 이 이 두 task 다. LIBERO 20 ep 에서도 mug task 2 ep 을 잃었다 (AG3S 14 / 20).

**후보.** `--graspable-hand-sized-mm 156` (`clustering.graspable_hand_sized`, 새 code — `admissibility.graspable` · `assess`, `TargetConfirm`, `AG3S._exclusion_gate` 의 재검사, 설정 · server 인자). 가장 좁은 폭이 최대 열림을 넘어도 **가장 긴 폭이 156 mm (최대 열림의 2 배) 이하**이면 쥘 수 있다고 본다. 기본은 꺼짐 (T26 규칙 그대로, bit 동일). 판 `…-hb-qs-rs-gh156-tro`. crate · basket (185 mm 이상) · book (233 mm) · cabinet 은 계속 탈락한다. **사용자 승인 (2026-11-13 00:0x): 2026-09-28 판정 ("crate 는 어떤 상황에서도 조작 대상이 될 수 없다") 으로 넣은 규칙의 변경이라 승인을 받고 진행.** 단위 test `tests/benchmark/test_sl_sota18_hand_sized.py` + 기존 admissibility · grounding test 포함 160 passed.

**위험.** mug 크기의 SafeLIBERO 장애물 (red coffee mug, moka pot) 도 후보가 된다 — attention 이 그것을 가리키면 회피 대상에서 빠진다 (지금도 milk 는 후보다: Goal L-I task 0 에서 target 의 57 %). 주축 폭은 관측된 점으로 재므로 한계 (156 mm) 근처의 물체는 frame 마다 갈릴 수 있다 (moka pot 149 mm).

**순서 · 판정 (결과 전 고정).**

| 단계 | 셋 | 통과 |
|---|---|---|
| 1 기제 (in-sample, 6 ep) | SafeLIBERO Long task 2 · 3 × L-I · L-II × init 19 (동결 판: TSR 0 · SSR 0 / 4) + LIBERO-10 task 4 · 6 × init 19 (0 / 2) | 6 ep 모두에서 쥐기 전 call 의 과반에 mug 가 target 으로 인식되고, task 성공이 6 ep 중 2 이상, > 50 mm 충돌이 늘지 않음 |
| 2 나머지 (1 단계 통과 때만) | 1/10 셋의 나머지 — SafeLIBERO 28 ep + LIBERO 18 ep (flag 는 mug 크기 cluster 가 있을 때만 판정을 바꾸지만, 궤적이 갈릴 수 있어 전부 다시 잰다) | 나머지에서 안전 성공 · LIBERO 성공이 동결 판보다 2 넘게 줄지 않음 |
| 판정 | 1 · 2 를 모두 채우면 동결 판 교체 후보로 보고한다 (교체는 사용자 판단) | |

GPU 약 40 분 (1 단계), 약 3 h (2 단계).
