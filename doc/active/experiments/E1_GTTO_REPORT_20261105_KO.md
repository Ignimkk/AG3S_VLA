# GT 인식 + 공용 TO 검증 — 자율 판정 구간 보고 (2026-11-05 01:00 – 21:00 KST)

> 읽는 사람: 사용자 (lead 의 자율 판정 검토). writer: lead. 판정 한 줄씩의 정본은
> [E1_GTTO_DECISIONS_20261104_KO.md](E1_GTTO_DECISIONS_20261104_KO.md) G18–G50, 실행 기록 정본은
> `AG3S/ag3s/docs/AG3S_BENCHMARK_LOG.md` "SL-GTTO" 절, 표 · 그림은 `outputs/2026-11-05_GT인식_공용TO_3벤치/`.
> 모든 run: 같은 기계 (H200 container), commit `89b42459` + 미commit patch, seed 7, conversion C, 장면당 1 episode.
> **commit 하지 않았다.**

## 1. 결론

1. **공용 TO 는 아직 SL-GTTO §5 전체를 통과하지 못했다 — 21 시 기준 가장 좋은 판 `zr-ee` (F7) 가 세 벤치 중 두 벤치 통과.**
   LIBERO 8/9 ✓ · SafeLIBERO 안전 성공 6 / 충돌 3 ✓ (두 벤치 모두 처음 통과) · LIBERO-Safety 3 ✗ (기준 5).
   LIBERO-Safety 의 차이는 human_safety (0/4, VLA 3/4) 에서 나며, 쥔 후 접촉 자세의 HOLD deadlock 이 원인이다 (§8, G50).
   16 시 기준 판은 `AG3S-commonto-gt2-termsoff-pass-zr` (세 벤치 모두 1–2 episode 부족) 였다.
2. 실패 위치는 판을 바꿀 때마다 앞에서 뒤로 옮겨 왔다: 인식 전달 (F1 · F2) → TO 목적함수 (termsoff · pass)
   → 쥐는 중 (F4) → **지금은 쥔 후 (놓기) 가 남은 실패의 대부분**이다.
3. **14 시에 찾은 공통 원인 후보:** TO 가 실제로 개입하는 call 은 모든 판에서 SQP 반복 cap (3) 에 100 % 걸리고,
   위반의 2–3 배만큼 궤적을 바꾸며, conversion C 가 포화해 실행 action 이 ±1 범위에서 1.2–1.6 씩 바뀐다 (§3b).
   F3 · F3b · F5 처럼 "어느 행을 푸나" 를 고친 판은 거절 수는 줄였지만 성공을 못 바꿨다.
   SQP 반복을 8 로 늘린 F6 도 6/9, 변경/위반 3.0 → 2.4 로 조금만 줄었다 (G46) → **남은 문제는 joint 공간 TO + conversion C
   구조 쪽이며, 다음 단계는 EE-delta 공간 QP (문헌과 같은 구조) 로 보는 것이 lead 의 판단이다 (§6-2, 사용자 판정).**
4. 남은 실패는 장면이 정해져 있다: LIBERO t4 · t9 (서랍 · 캐비닛), SafeLIBERO t3 (캐비닛 위 그릇), LIBERO-Safety 일부.
   zr 계열 판 사이 1 episode 차이 (t6 · t7) 는 장면당 1 episode 의 흔들림이다 (SL9 "2 이하 구분 못 함").

## 2. 판 사다리 (세 벤치, §5 규칙)

§5: LIBERO 성공 ≥ 8 ∧ VLA − GT ≤ 1 · SafeLIBERO 충돌 ≤ VLA − 3 (= 6) ∧ 안전 성공 ≥ VLA + 3 (= 5) ·
LIBERO-Safety 성공∧무위반 ≥ VLA − 2 (= 5) ∧ 위반 ≤ VLA (= 1).

| 판 (판정 ID) | 바꾼 것 | LIBERO 9 | SafeLIBERO 11 성공 / 충돌 / 안전 성공 | LIBERO-Safety 10 성공∧무위반 / 위반 |
|---|---|---:|---|---|
| VLA 단독 | — | 9 | 11 / 9 / 2 | 7 / 1 |
| gt (v1, G21) | 공용 TO + GT 인식 | 2 | 0 / 2 / 0 | 2 / 1 |
| gt2 (G28) | F1 GT target admissibility 통과 · F2 손안 GT pixel TSDF 제외 | 2 | 4 / 6 / 2 | 2 / 0 |
| termsoff · pass (G31) | 연속성 · 꺾임 항 끔 / 위반 없으면 VLA 그대로 | 3 · 3 | — | — |
| termsoff-pass (G31 · G36) | 둘 다 (= 문헌의 최소 개입) | 4 | 1 / 3 / 0 | **5 / 0 ✓** |
| + F3 hbc 계열 (G32) | 쥔 물체 아래 표면 행 해제 | 2 · 2 · 3 | — | — |
| **zr (G34 · G40 · G44)** | termsoff-pass + F4 쥐기 직전 · 쥐는 중 손 행 해제 (target 수평 100 mm) | **6** | **5 / 5 ✓ / 4** | 4 / 1 |
| zr-oc (G42) | + F5 가려진 쥔 물체 frame 을 TO 에도 인증 | 5 | — | — |
| zr-oc-hbc (G44) | + F3 | 6 | — | — |
| zr-oc-hbch (G44 · G45) | + F3b 쥔 동안 손 행도 아래 표면 해제 | 6 | gate 닫힘 (< 7) | gate 닫힘 |
| zr-it8 (G45 · G46) | zr + SQP 반복 3 → 8 (F6) | 6 | — (정보용) | — |
| **zr-ee (G47 · G49 · G50)** | zr + F7 정책 action 공간 min-norm QP (conversion C 우회) | **8 ✓** | **6 / 3 ✓ / 6 ✓** | 3 / 0 |

## 3. 단계 분석 (사용자 지시 09:5x)

단계는 sim 진실로 모든 판에 같게 정했다: 쥐기 전 (첫 gripper 닫힘 명령 전) · 쥐는 중 (닫힘 ~ target 3 cm 들림) ·
쥔 후. 표 `outputs/.../단계분석/단계표.md`, 그림 `그래프_단계별.png`, script `experiments/segmentation/gtto_phase.py`.

| 판 | 실패 단계 (쥐기 전 / 쥐는 중 / 쥔 후), 세 벤치 합 | 쥐는 중 걸린 step (LIBERO 중앙값, VLA 18) |
|---|---|---:|
| gt2 | 1 / 5 / 16 | 20.5 |
| termsoff-pass | 0 / 8 / 12 | 22 |
| zr | 0 / 3 / 12 | 20 |
| zr-oc-hbch (LIBERO 만) | 0 / 1 / 2 | 20 |
| zr-it8 (LIBERO 만) | 0 / 1 / 2 | 19 |

- 쥐기 전 실패는 거의 없다. 서랍 · 캐비닛도 VLA 는 닫았다.
- F4 가 쥐는 중 실패를 줄였다 (LIBERO 2 → 0, SafeLIBERO 4 → 2).
- TO 개입 크기 (실행 − 정책 action) 는 모든 판에서 쥐기 전 < 쥐는 중 < 쥔 후.
- 쥔 후에 TO 가 VLA 를 그대로 통과시키지 못한 이유 (새 기록 `to.passthrough_clear`): zr 에서 기하 미인증 67 · 충돌 6 · 한계 6.
  F5 로 미인증을 없애자 충돌 32 · 한계 14 가 드러났다. HOLD 행은 쥔 물체 (`attached_point`) 와 손 · 손가락 ↔ 놓는 면.

## 3b. TO 가 개입할 때의 증폭 (G45)

충돌로 pass-through 가 거절된 call 만 (`outputs/.../증폭진단/증폭표.md`, `그래프_증폭.png`,
script `experiments/segmentation/gtto_amplification.py`):

| 판 | 벤치 | call | 위반 중앙값 mm | 궤적 변경 중앙값 mm (p90) | 변경/위반 | C 포화 행 비율 | \|Δa\| (±1 범위) | 반복 cap 도달 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| zr | LIBERO | 30 | 15.8 | 35.9 (101) | 3.0 | 0.9 | 1.6 | 100 % |
| zr | SafeLIBERO | 99 | 13.4 | 28.3 (145) | 2.7 | 0.9 | 1.34 | 100 % |
| zr | LIBERO-Safety | 119 | 10.2 | 23.9 (79) | 2.3 | 0.6 | 1.24 | 100 % |
| zr-oc-hbch | LIBERO | 42 | 13.4 | 34.4 (171) | 3.2 | 0.9 | 1.52 | 100 % |
| zr-it8 (반복 8) | LIBERO | 37 | 11.0 | 25.5 (63) | 2.4 | 0.7 | 1.3 | 97 % |

- 실패 episode 는 이런 call 이 episode 당 평균 7.1 (zr) · 10.3 (hbch), 성공 episode 는 1.5 · 1.8. 상관이며, 실패 episode 가 더 길다.
- 예: LIBERO t4 (서랍 속 그릇) 는 latched ↔ closing 을 끝까지 오가고, 쥐기 직전 개입 call 5 개 연속 \|Δa\| 1.5–1.9.
- 가설 "3 회 cap 에서의 미수렴" 은 F6 (반복 8) 으로 확인했다: 변경/위반 3.0 → 2.4, 성공 6 → 6. 결과 전 규칙 (절반 이하) 불성립
  → 반복 수로는 풀리지 않는다. 증폭은 conversion C · trust region 쪽, 즉 §6-2 의 EE-delta QP 구조 변경 문제로 넘긴다 (G46).

## 4. 문헌 조사 (사용자 지시 09:0x) 와 그 반영

Panda 에서 TO · 충돌회피를 한 13 편 (AEGIS · KNOWS · multi-link CBF · VLPSA · OSCBF · cuRobo · MoveIt MTC · STORM 등).
VLA safety filter 는 모두 `min ‖u − u_nom‖²` 에 위반 없으면 그대로 통과; cuRobo `plan_grasp` · MTC 는 마지막 접근에서
손–world 충돌을 끄고, 놓기에서는 물체–지지면 접촉을 허용한다. 우리 TO 와의 가장 큰 차이는 joint 공간 최적화 → EE-delta
역변환 (변환 C) 이다. 반영: termsoff-pass (최소 개입), F4 (cuRobo 접근), F3b (MTC 놓기). **EE-delta 공간 QP 로의 구조 변경은
하지 않았다** (사용자 판정 대상).

## 5. 바꾼 code (모두 기본 꺼짐 flag, commit 안 함)

| flag (판) | 위치 | 되돌리기 |
|---|---|---|
| `--oracle-admissible` · `--oracle-held-target` (F1 · F2, gt2) | `ag3s/runtime/pipeline.py`, `stages/target_grounding.py`, `config.py` | `outputs/.../gtto2.patch` `git apply -R` |
| `--held-beneath-contact` (F3) | `trajopt/safe_policy.py` `_held_beneath_contact` | scratchpad `*.pre-gtto4` |
| `--grasp-zone-release` (F4) | `trajopt/linearize.py` `_grasp_contact_margin`, `safe_policy._grasp_contact` | `*.pre-gtto5` |
| `--occluded-certify` (F5) | `safe_policy._occluded_certified` + `_scene_fn` 1 줄 | `*.pre-gtto6` |
| `--held-beneath-hand` (F3b) | `linearize._placement_contact_margin` hand 분기 | `*.pre-gtto7` |
| F6 판 `zr-it8` (`--sqp-min/max-iterations 8`, 기존 flag) | profile 등록만 | `*.pre-gtto8` |
| 기록만: `to.passthrough_clear`, `limit_overshoot` | `safe_policy._to_block`, `sqp._clear_passthrough` | G43 |

공통: serve_safe flag · `apply_grasp_contact_flags`, Panda server `PORTED_FINAL_CONFIG_DESTS`, profile 판 등록.
test: `tests/trajopt/test_sl_gtto{4,5,6,7,8}_*.py`, `tests/benchmark/test_sl_gtto*_versions.py` 등 — 최근 실행 232 passed / 5 skipped.

## 6. 사용자 판정이 필요한 것

1. 공용 TO 정의를 무엇으로 할지. **21 시 lead 추천: zr-ee (F7)** — LIBERO · SafeLIBERO 를 통과한 유일한 판. LIBERO-Safety 만 남았고 원인 층 (쥔 후 HOLD deadlock) 이 좁혀졌다. 대안: termsoff-pass (LIBERO-Safety 만 통과), zr (세 벤치 1–2 부족).
2. EE-delta 공간 QP 로 옮기는 구조 변경 — **17:4x 에 lead 추천대로 기본 꺼짐 flag 로 실행함 (F7, G47).** 채택 여부와 다음 두 수정의 승인:
   (a) OSQP scaling 수정 (행 정규화 + quadratic slack; offline 150 회 · 5 ms 에 엄밀해, G50) — 동작이 바뀌므로 새 판으로;
   (b) step 0 · 1 위반을 hard 제약 대신 이산 CBF 조건 `h_k ≥ (1 − γ) h_{k−1}` (악화만 금지) 으로 — LIBERO-Safety deadlock 을 겨냥.
3. 장면당 1 episode 로는 1 episode 차이를 가를 수 없다 — 다음 판정 전에 episode 수를 늘릴지.
4. G31 규칙 이탈 (글자대로면 pass 판, 포화 0.0003 차이로 termsoff-pass 를 정보용으로 실행) 을 인정할지.

## 7. E1a baseline 진행

SafeLIBERO spatial Level I · II 7 arm 완료 (`outputs/e1a/agg_1235/`). SSR / 40: Level I π0.5 6 · AEGIS 17 (privileged 17) ·
multi-link 11 (GT 23) · KNOWS 1 (GT 21); Level II π0.5 7 · AEGIS 11 (13) · multi-link 12 (23) · KNOWS 3 (11).
**object suite Level I · II 7 arm 완료 (`outputs/e1a/agg_2030/`):** Level I π0.5 3 · AEGIS 2 (privileged 15) · multi-link 2 (GT 5) · KNOWS 2 (GT 6);
Level II π0.5 10 · AEGIS 12 (23) · multi-link 15 (11) · KNOWS 1 (17). goal suite 시작 (20:23, 진행 중). 4-suite 평균은 네 suite 가 다 끝나야 나온다.

## 8. 연장 구간 (17:30 – 21:00, 사용자 지시 "오후 9시까지 연장")

§6 의 판정 요청에 답이 오기 전이므로 lead 추천 (§6-2 "EE-delta 공간 QP 를 한다") 을 **기본 꺼짐 flag 로** 먼저 실행했다 (G47).

- **원인 (code 로 확인):** conversion C 의 역변환 `Dx = (Dy − α·Dy₋₁)/β`, 위치 β = 0.071 · scale 0.05 → 한 step 에서 경로 3.6 mm 를
  바꾸는 데 action 1 이 든다. joint 공간 TO 는 이 OSC 지연과 \|a\| ≤ 1 을 모른 채 경로를 고쳐 실현 불가능한 수정을 낸다.
- **F7 `--ee-action-qp`:** pass-through 가 거절된 chunk 에서 `min ‖Δa‖²` (정책 action 공간), 제약 = TO 와 같은 선형화 충돌 행
  (`Δq = M Δa`, M = OSC 응답 모델 × EE Jacobian pseudo-inverse) + action box (hard). 최종 안전 검사 · 판정은 기존 그대로,
  SafePolicy 가 그 궤적을 반환했을 때만 `a_ref + Δa` 를 보낸다. 판 `AG3S-commonto-gt2-termsoff-pass-zr-ee`.
- **LIBERO 9: 8/9 — 처음으로 LIBERO §5 통과** (zr 6, F6 6). 되찾은 장면 t7 · t9, 남은 실패 t4 (쥐는 중). F7 이 돈 call 42 개 모두
  실제로 Δa 가 보내졌고, conversion C 포화 0 (zr 0.9), 변경/위반 3.0 → 2.15, 변경 p90 101 → 41 mm (G49).
- 한계: OSQP 가 대부분 iteration 상한 (20000) 에서 끝남 → 엄밀한 min-norm 이 아닐 수 있음, solve 1.46 s (sim 동기식이라 결과 영향 없음).
- **SafeLIBERO 11: 공식 6 · 충돌 3 · 안전 성공 6 → §5 통과 (처음).** zr (5 · 5 · 4) 대비 t3 (캐비닛 위 그릇) 0/4 → 2/4.
  conversion C 포화 행 중앙값 0.9 → 0.2, \|Δa\| 1.34 → 0.95.
- **LIBERO-Safety 10: 성공∧무위반 3 · 위반 0 → 실패 (기준 5).** suite 별 F7 / VLA / zr: human_safety 0 / 3 / 1 (4 중),
  obstacle_avoidance 3 / 3 / 3 (4 중), affordance 0 / 1 / 0 (2 중). 실패 7 개 모두 쥔 후, 쥔 후 HOLD 비율 0.356 (zr 0.03).
- **원인 (G50):** HOLD 를 만든 충돌 행의 step 이 0 (= 지금 자세가 이미 위반) 188 · 1 132 · 2–4 69. step 0 은 action 으로 못 고치고,
  step 1 은 OSC 지연 때문에 action 1 당 3.6 mm. F7 은 box 가 hard 라 slack 을 쓰고 최종 검사가 HOLD → 쥔 물체 · 손이 표면에 닿은
  자세에서 멈춘 채 반복 (deadlock). 행 사유는 손 ↔ 쥔 물체 아닌 표면 (`nearest_is_not_manipulated`) 189 · 쥔 물체 (`attached_point`) 150.
  **= grasp handling · 판정 규칙 층의 문제이며, solver 층이 아니다.**
- 가설 (미검증): zr 의 낮은 HOLD 비율은 conversion C 가 실현 못 하는 joint 경로를 최종 검사가 통과시킨 결과일 수 있다
  (zr LIBERO-Safety 개입 call 의 C 포화 0.6). F7 은 실현 가능한 경로만 내므로 "못 피함" 이 HOLD 로 드러난다.
- OSQP 미수렴 (G49 한계) offline 확인: 기록된 call 하나에서 지금 정식화는 20000 회에 norm 0.1712 (엄밀해 0.1689, 1.4 % 차이),
  행 정규화 + quadratic slack 벌점은 150 회 · 5 ms 에 엄밀해. code 는 고치지 않았다 (§6-2a).
- 그림 · 표: `outputs/2026-11-05_GT인식_공용TO_3벤치/gt10_ee/` (결과표 · 성공 grid · HOLD 비율), `단계분석/`, `증폭진단/`.
