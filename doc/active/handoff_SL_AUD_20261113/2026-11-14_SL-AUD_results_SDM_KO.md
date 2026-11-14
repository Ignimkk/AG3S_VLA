# SL-AUD 결과 (SDM → 로컬 PC, 2026-11-14 02:4x 시작 · 18:5x 마감)

받는 쪽: SL-AUD code 를 만든 로컬 PC 세션. 작성: SDM 세션 (sdm-f6, audq). 요청 문서 `2026-11-13_SL-AUD_test_plan_KO.md` §8 의 형식을 따른다.
**마감 (18:5x).** §A = 동결 판 확인 · 거르기 1, §B = 거르기 2, §C = `-md5` · 거르기 1 재실행 · 채택 gate · 최종 동결. 결론: 채택 gate 탈락 → 최종 동결 판 = `-gh156a`.

**가장 먼저 볼 것: `-rf` (SL-AUD1) 가 closed loop 에서 한 번도 작동하지 않았다 (§A.4).** 거르기 1 은 사실상 `-lp-mr8-af` 를 잰 것이다. → `0f1d90b4` 로 고친 재실행은 §C.3 (628 / 628 call 적용, 결과 FAIL).

---

## 0. 넣은 방법 (§8.5 질문 1 · 2)

| 항목 | 결과 |
|---|---|
| base | `wip/sota-hf` (= `e5f0254d` + FWDATT merge `656a98f6` + `-fwd` 판 등록 `328f601e` + gate 기록) |
| `282e22d9` | 깨끗이 들어감 |
| `ac62c2bc` | 충돌 2 파일 (예고대로): `test_sl_gtto11_scaled_qp.py`, `test_sl_sota11_opening_rule_reach_sticky.py` 의 허용 목록 → `+ pf.SOTA_FWD_VERSIONS + pf.SOTA_AUDIT_VERSIONS` (양쪽 남김). `profile.py` 와 `test_sl_sota18_hand_sized.py` 는 자동 병합 |
| 손으로 더한 것 | (1) `-gh156a-fwd` 를 SOTA_AUDIT base 에 추가 (판 등록 영역 끝, 13 개 → 합 39, 이름 `…-gh156a-fwd<segment>-tro`), `test_sl_aud_versions.py` 를 base 셋 · 39 개로. (2) `test_sl_sota18_hand_sized.py` 판 목록에 `and v not in pf.SOTA_FWD_VERSIONS` (FWDATT 등록 뒤 `wip/sota-hf` 에서 이미 실패하던 것) |
| branch | `wip/sl-aud` (main 이 01:09:56 에 fast-forward, HEAD `09dc923b`) |
| unit test (§3 의 14 파일 + FWDATT 2 파일) | **233 통과 · 4 skip · 실패 0** (Linux, Python 3.11, CPU) |
| `tests/benchmark tests/trajopt -k "version or flag"` | 433 통과 · 42 skip · 10 실패. 10 개 모두 변경과 무관: 7 = storage 의 `sl6b-hsr6-panda-labels/probe/hsr6_panda_kv_L4.npz` 없음, 1 = `test_sl_gtto_gt_perception` 판 목록에 `…-tro-gto` 없음 (둘 다 변경 전 `e5f0254d` 에서도 실패), 2 = worktree 에 LIBERO submodule 이 없어서 (본 checkout 에서 20 / 20 통과). `test_t41b` · `t41c` · `t43q` 는 legacy fixture 가 없어 수집 안 됨 (SDM 환경) |

## A. 동결 판 확인과 거르기 1

### A.1 동결 판 1 ep (§8.5 질문 3)

`ms-audchksdm-spatial-LI-i15` (동결 판, Spatial L-I init 15–18 run 의 첫 episode = task 0 init 15, 같은 인자), 01:09:58–01:18. **같음**: call 17 / 17 의 `max_abs_action_change_executed` · `to.passthrough_clear.reason` 이 기준 run (`20261112T154530`) 과 모두 같고, episode 결과 (성공 · 충돌 없음 · 변위 0 · 85 step · 17 call) 도 같다. joint SQP call 은 없었다.

### A.2 거르기 1 run (§8.1)

판 `…-hb-qs-rs-rf-lp-mr8-af-tro`, HEAD `09dc923b`, TAG `aud1sdm`, 01:18:15–02:37:03. run: `20261114T011817_…_ms-aud1sdm-spatial-LI-i15_16_17_18`, `20261114T015416_…_ms-aud1sdm-goal-LI-i17_18`, `20261114T021640_…_ms-aud1sdm-goal-LII-i17_18` (기준 run 과 같은 구성, `sota_ms_queue.sh`).

| S32 | 동결 판 TSR · CAR · SSR, > 50 mm | `-rf-lp-mr8-af` |
|---|---|---|
| Spatial L-I (16) | 7 · 12 · 5, 2 | 8 · 11 · 7, 2 |
| Goal L-I (8) | 2 · 6 · 1, 1 | 2 · 6 · 1, 0 |
| Goal L-II (8) | 4 · 5 · 3, 0 | 4 · 5 · 2, 1 |
| **합 (32)** | **13 · 23 · 9, 3** | **14 · 22 · 10, 3** |

| 기준 (§5) | 값 | 판정 |
|---|---|---|
| SSR ≥ 12 | 10 | 불통과 |
| > 50 mm ≤ 4 | 3 | 통과 |
| 동결 판 안전 성공 9 중 잃는 것 ≤ 1 | 4 (spatial t3 e16 → F, goal L-I t1 e17 → S+C 15 mm, goal L-II t0 e17 → S+C 33 mm, goal L-II t3 e17 → F) | 불통과 |

**→ FAIL.** 잃은 13 ep 중 큰 보정 5 call 이상이 남은 ep 9 (≥ 7) → §5 셋째 줄: `<frozen>-md5-tro` 를 S32 로 (거르기 2 뒤). 안전 성공 얻음 5 (spatial t0 e16 · t1 e17 · t2 e18, goal L-I t0 e18, goal L-II t3 e18), 잃은 13 중 돌아옴 2 (spatial t1 e17, goal L-I t0 e18).

S32 32 ep 의 짝 (outcome: SS 안전 성공 · S+C 성공 + 충돌 · F 실패 · F+C 실패 + 충돌; π0.5 = baseline tranche t1):

| cell | task | init | π0.5 | 동결 판 → 거르기 1 | 변위 mm | 잃은 13 |
|---|---:|---:|---|---|---|---|
| spatial L-I | 0 | 15 | F+C | SS → **SS** | 0.0 → 0.0 |  |
| spatial L-I | 0 | 16 | F+C | S+C → **SS** | 56.9 → 0.0 |  |
| spatial L-I | 0 | 17 | S+C | SS → **SS** | 0.0 → 0.0 |  |
| spatial L-I | 0 | 18 | F+C | S+C → **F+C** | 5.0 → 5.8 |  |
| spatial L-I | 1 | 15 | S+C | F → **F+C** | 0.0 → 87.0 | ✓ |
| spatial L-I | 1 | 16 | S+C | F+C → **S+C** | 7.3 → 111.1 | ✓ |
| spatial L-I | 1 | 17 | SS | F → **SS** | 0.0 → 0.0 | ✓ |
| spatial L-I | 1 | 18 | S+C | F → **F+C** | 0.0 → 29.3 | ✓ |
| spatial L-I | 2 | 15 | S+C | SS → **SS** | 0.0 → 0.0 |  |
| spatial L-I | 2 | 16 | S+C | F → **F** | 0.0 → 0.0 | ✓ |
| spatial L-I | 2 | 17 | S+C | SS → **SS** | 0.0 → 0.0 |  |
| spatial L-I | 2 | 18 | F+C | F → **SS** | 0.0 → 0.0 |  |
| spatial L-I | 3 | 15 | F+C | F → **F** | 0.4 → 0.6 |  |
| spatial L-I | 3 | 16 | S+C | SS → **F** | 0.0 → 0.0 |  |
| spatial L-I | 3 | 17 | S+C | F → **F** | 0.0 → 0.0 | ✓ |
| spatial L-I | 3 | 18 | F+C | F+C → **F+C** | 294.5 → 4.9 |  |
| goal L-I | 0 | 17 | S+C | F+C → **F** | 2.2 → 0.0 | ✓ |
| goal L-I | 0 | 18 | S+C | F → **SS** | 0.0 → 0.0 | ✓ |
| goal L-I | 1 | 17 | S+C | SS → **S+C** | 0.0 → 15.2 |  |
| goal L-I | 1 | 18 | S+C | S+C → **F+C** | 83.6 → 2.4 |  |
| goal L-I | 2 | 17 | F+C | F → **F** | 0.0 → 0.0 |  |
| goal L-I | 2 | 18 | S+C | F → **F** | 0.0 → 0.0 | ✓ |
| goal L-I | 3 | 17 | F+C | F → **F** | 0.0 → 0.0 |  |
| goal L-I | 3 | 18 | S+C | F → **F** | 0.0 → 0.0 | ✓ |
| goal L-II | 0 | 17 | F+C | SS → **S+C** | 0.0 → 33.4 |  |
| goal L-II | 0 | 18 | F | SS → **SS** | 0.0 → 0.0 |  |
| goal L-II | 1 | 17 | S+C | F+C → **F** | 10.5 → 0.0 | ✓ |
| goal L-II | 1 | 18 | S+C | F → **F** | 0.0 → 0.0 | ✓ |
| goal L-II | 2 | 17 | S+C | F+C → **F+C** | 6.7 → 53.8 | ✓ |
| goal L-II | 2 | 18 | S+C | S+C → **S+C** | 2.2 → 2.2 |  |
| goal L-II | 3 | 17 | S+C | SS → **F** | 0.9 → 0.0 |  |
| goal L-II | 3 | 18 | F+C | F → **SS** | 0.0 → 0.3 |  |

### A.3 `calls.jsonl` 에서 센 값 (§8.2, episode 마다 동결 판 → 거르기 1)

합 (32 ep): 큰 보정 call 318 → 167 (쥐기 전 58 → 43, 쥔 동안 205 → 107, 놓은 뒤 55 → 17; 최악 row step 0–4 61 → 32, 5–9 178 → 131), ±1 포화 성분 3735 → 1847, slack > 15 mm 177 → 57, QP 에서 뺀 row (마지막 iteration) 930 → 30,953, 사유 `limit` 5 → 11 (그중 `limit_passthrough` 9), IK 한도 316 → 290, OSQP 반복 한도 148 → 41, QP call 의 solve ms 중앙값 합 8110 → 4078. `reference_floor.n_rows_below_backoff` 는 기록되지 않음 (§A.4).

| cell t e | call | QP call | 보정≥1 | 포화 합 | slack>15mm | 뺀 row | 사유 limit | limit 통과 | IK 한도 | IK 오차 max mm | OSQP 한도 | solve ms 중앙 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial LI t0 e15 | 17 → 17 | 9 → 8 | 0 → 0 | 13 → 7 | 0 → 0 | 30 → 414 | 0 → 0 | – → – | 0 → 0 | – → – | 0 → 0 | 69.6 → 39 |
| spatial LI t0 e16 | 41 → 34 | 19 → 15 | 8 → 4 | 48 → 24 | 3 → 1 | 29 → 736 | 0 → 0 | – → – | 0 → 0 | – → – | 3 → 1 | 14.5 → 27.5 |
| spatial LI t0 e17 | 16 → 16 | 5 → 5 | 0 → 0 | 0 → 0 | 0 → 0 | 10 → 255 | 0 → 0 | – → – | 0 → 0 | – → – | 0 → 0 | 10 → 10.1 |
| spatial LI t0 e18 | 43 → 60 | 22 → 35 | 9 → 5 | 108 → 33 | 6 → 0 | 89 → 1802 | 0 → 0 | – → – | 3 → 0 | 102 → – | 5 → 0 | 42.2 → 52.2 |
| spatial LI t1 e15 | 60 → 60 | 40 → 25 | 26 → 9 | 424 → 102 | 24 → 5 | 14 → 1164 | 1 → 4 | – → 4 | 20 → 25 | 138 → 276 | 22 → 3 | 107 → 31.2 |
| spatial LI t1 e16 | 60 → 40 | 33 → 14 | 12 → 3 | 88 → 28 | 5 → 0 | 0 → 672 | 0 → 0 | – → – | 0 → 0 | – → – | 3 → 1 | 110 → 31.5 |
| spatial LI t1 e17 | 60 → 31 | 25 → 10 | 12 → 1 | 52 → 5 | 1 → 0 | 14 → 480 | 0 → 0 | – → – | 0 → 0 | – → – | 4 → 0 | 21.3 → 20.2 |
| spatial LI t1 e18 | 60 → 60 | 44 → 26 | 24 → 7 | 391 → 14 | 21 → 0 | 0 → 1248 | 0 → 0 | – → – | 23 → 3 | 153 → 9.4 | 7 → 0 | 124 → 32.8 |
| spatial LI t2 e15 | 28 → 35 | 5 → 14 | 1 → 2 | 8 → 34 | 0 → 1 | 5 → 677 | 1 → 1 | – → – | 7 → 14 | 123 → 209 | 0 → 0 | 15.6 → 28.6 |
| spatial LI t2 e16 | 60 → 60 | 24 → 12 | 19 → 7 | 368 → 79 | 14 → 3 | 9 → 565 | 0 → 0 | – → – | 27 → 21 | 264 → 112 | 6 → 4 | 14.9 → 14.3 |
| spatial LI t2 e17 | 33 → 52 | 7 → 14 | 2 → 3 | 17 → 62 | 1 → 2 | 37 → 828 | 0 → 0 | – → – | 11 → 9 | 210 → 223 | 0 → 3 | 11.3 → 13.2 |
| spatial LI t2 e18 | 60 → 32 | 10 → 7 | 1 → 1 | 28 → 21 | 1 → 1 | 0 → 400 | 0 → 0 | – → – | 30 → 7 | 137 → 81.5 | 0 → 1 | 13.7 → 11.7 |
| spatial LI t3 e15 | 60 → 60 | 25 → 24 | 6 → 6 | 87 → 79 | 4 → 3 | 34 → 1205 | 1 → 2 | – → 2 | 33 → 36 | 336 → 542 | 5 → 1 | 25.2 → 14.7 |
| spatial LI t3 e16 | 27 → 60 | 16 → 26 | 2 → 6 | 27 → 39 | 3 → 0 | 0 → 1287 | 0 → 0 | – → – | 0 → 20 | – → 267 | 3 → 1 | 51.2 → 13.9 |
| spatial LI t3 e17 | 60 → 60 | 24 → 13 | 6 → 1 | 172 → 14 | 7 → 0 | 17 → 624 | 0 → 0 | – → – | 24 → 0 | 260 → – | 3 → 0 | 15.3 → 11.7 |
| spatial LI t3 e18 | 60 → 60 | 38 → 36 | 5 → 12 | 109 → 160 | 4 → 4 | 34 → 1812 | 1 → 2 | – → 1 | 26 → 41 | 426 → 602 | 3 → 2 | 50.7 → 50.2 |
| goal LI t0 e17 | 60 → 60 | 53 → 48 | 22 → 20 | 151 → 87 | 0 → 0 | 0 → 2307 | 0 → 0 | – → – | 0 → 0 | – → – | 13 → 1 | 184 → 115 |
| goal LI t0 e18 | 60 → 18 | 35 → 10 | 16 → 0 | 130 → 11 | 5 → 0 | 43 → 541 | 0 → 0 | – → – | 13 → 2 | 329 → 3.8 | 2 → 0 | 68.2 → 69.1 |
| goal LI t1 e17 | 21 → 31 | 6 → 14 | 3 → 4 | 29 → 58 | 0 → 1 | 31 → 717 | 0 → 0 | – → – | 0 → 0 | – → – | 0 → 3 | 14.9 → 30.3 |
| goal LI t1 e18 | 46 → 60 | 19 → 17 | 7 → 3 | 70 → 29 | 3 → 0 | 91 → 1069 | 0 → 0 | – → – | 2 → 0 | 15.2 → – | 3 → 1 | 21.4 → 14.6 |
| goal LI t2 e17 | 60 → 60 | 32 → 13 | 26 → 5 | 254 → 74 | 22 → 5 | 28 → 650 | 0 → 0 | – → – | 0 → 4 | – → 11.4 | 25 → 2 | 86.5 → 24.8 |
| goal LI t2 e18 | 60 → 60 | 35 → 19 | 29 → 10 | 369 → 120 | 24 → 2 | 44 → 1007 | 0 → 1 | – → 1 | 0 → 23 | – → 72.5 | 27 → 6 | 166 → 33.9 |
| goal LI t3 e17 | 60 → 60 | 39 → 24 | 8 → 3 | 50 → 9 | 2 → 0 | 0 → 1152 | 0 → 0 | – → – | 0 → 0 | – → – | 3 → 0 | 63.5 → 14.4 |
| goal LI t3 e18 | 60 → 60 | 17 → 14 | 7 → 5 | 110 → 105 | 5 → 4 | 30 → 676 | 0 → 0 | – → – | 23 → 21 | 263 → 259 | 1 → 1 | 10.2 → 16.5 |
| goal LII t0 e17 | 26 → 27 | 16 → 17 | 4 → 4 | 54 → 75 | 2 → 2 | 36 → 1007 | 1 → 0 | – → – | 6 → 8 | 79.7 → 117 | 0 → 0 | 81.7 → 81.2 |
| goal LII t0 e18 | 29 → 25 | 3 → 2 | 1 → 1 | 8 → 11 | 1 → 0 | 24 → 103 | 0 → 1 | – → 1 | 5 → 5 | 45.6 → 76.9 | 1 → 0 | 15.8 → 14.4 |
| goal LII t1 e17 | 60 → 60 | 30 → 22 | 17 → 18 | 133 → 289 | 2 → 13 | 34 → 1205 | 0 → 0 | – → – | 27 → 25 | 244 → 364 | 1 → 5 | 34.1 → 15.1 |
| goal LII t1 e18 | 60 → 60 | 27 → 22 | 16 → 8 | 170 → 129 | 7 → 5 | 0 → 1173 | 0 → 0 | – → – | 36 → 26 | 328 → 61.8 | 2 → 1 | 24.8 → 23.6 |
| goal LII t2 e17 | 60 → 60 | 37 → 42 | 8 → 8 | 57 → 69 | 3 → 3 | 220 → 2274 | 0 → 0 | – → – | 0 → 0 | – → – | 4 → 4 | 64 → 65.3 |
| goal LII t2 e18 | 25 → 25 | 4 → 4 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 192 | 0 → 0 | – → – | 0 → 0 | – → – | 0 → 0 | 18.4 → 17.4 |
| goal LII t3 e17 | 55 → 60 | 37 → 46 | 11 → 7 | 33 → 12 | 1 → 0 | 0 → 2212 | 0 → 0 | – → – | 0 → 0 | – → – | 1 → 0 | 46.5 → 41.4 |
| goal LII t3 e18 | 60 → 28 | 18 → 10 | 10 → 4 | 177 → 68 | 6 → 2 | 27 → 499 | 0 → 0 | – → – | 0 → 0 | – → – | 1 → 0 | 12.7 → 14.8 |

(뺀 row = `n_rows_unreachable` 의 마지막 iteration 합. 동결 판은 `-mr` 없이도 0 이 아닌 값을 기록한다 — 그 의미는 code 쪽에서 확인 바람.)

### A.4 `-rf` 가 작동하지 않음 [M]

- S32 의 action QP call **608 개 모두** `to.ee_action_qp.reference_floor = {"why": "no ESDF row"}`. `error` key 는 없다 (예외 아님).
- code: `sqp.py` `_solve_ee_action` (wip/sl-aud 기준 1581–1594 줄) — `c_ref = linearizer._esdf_clearance(linearizer.sphere_states(reference, q_now)[0], scene)` 가 `ndim == 2 and shape[0] == H and shape[1] > 0` 을 채우지 못해 floor 를 건너뜀.
- 같은 call 에서 QP 의 장애물 row 는 정상 (`rows` 240, `n_obstacle_rows` 236), chunk 기록 (`server_constraints/run_0000/chunk_*.npz`) 에 curobo ESDF 가 있다 (`esdf_stats.backend = curobo`, main 층 voxel 5 mm). `-mr8` 의 `n_rows_unreachable` 는 매 call 0 이 아니다 → ESDF row 는 linearize 에 있었다.
- 그러므로 live 경로에서만 `_esdf_clearance` 의 결과 모양이 다르다. 후보 (확인하지 못함): `scene.esdf is None` 이어서 `_esdf_clearance_core` 가 3 차원 `zeros((H, Q, 0))` 을 돌려줌 (ESDF row 가 다른 경로로 들어오는 경우), 또는 `H` (`nq, H = reference.shape`) 와 `sphere_states` 의 첫 축이 다름. 단위 test 는 CPU 의 실제 linearizer 로 통과했으나 서버의 `--esdf-backend curobo --to-fast-eval` 경로를 지나지 않는다.
- 요청: 원인과 고친 code. 고친 판을 S32 로 다시 잴지는 SDM 사용자가 정한다 (H1 은 `-rf` + `-mr8` 이 함께여야 한다고 했으므로 지금 결과로는 H1 을 닫을 수 없다).

### A.5 가설 (§4) 에 대한 읽기 (D)

| 가설 | 거르기 1 에서 본 것 |
|---|---|
| H1 | 큰 보정은 절반 (318 → 167) 이 됐으나 잃은 ep 은 2 / 13 만 돌아옴. 단 `-rf` 미작동이라 판정 보류가 맞다 |
| H2 | CAR 23 → 22, > 50 mm 3 → 3. spatial t1 e15 · e16 이 0 · 7 → 87 · 111 mm 로 새로 밂, spatial t3 e18 은 294 → 5 mm |
| H3 | 반대: Goal L-I t0 e18 은 F → SS (IK 한도 call 13 → 2). e17 은 F 그대로 (IK 한도 0 → 0, 큰 보정 22 → 20) |

## §8.3 raw 기록

SDM `/mnt/dev/AG3S_VLA/outputs/2026-11-14_SL-AUD/raw6/{aud1sdm,pkgFrssdm}/<stem>_<level>/` 에 6 ep × 두 판의 `*_calls.jsonl` · `*_calls.npz` · `*_geom.npz` (+ `RUN.txt` = run dir), 107 MB. `chunk_summary.jsonl` 은 이 서버의 run 에 없다 — chunk 별 기록은 run 의 `server_constraints/run_0000/chunk_*.npz` (`summary_json` + ESDF · 구 · cloud). 거르기 2 의 같은 6 ep 은 끝난 뒤 더한다.

## §8.4 CPU 집계 (동결 판 144 ep)

`outputs/2026-11-14_SL-AUD/cpu_agg/frozen144.csv` (episode 한 줄) · `frozen144_summary.md` · `agg_calls.py` (run glob 을 받아 같은 열을 냄; `aud1sdm.csv` 도 같은 script). 큰 보정 (≥ 1) 수는 `qp_delta_latch.csv` 의 `ge1_total` 과 144 / 144 ep 일치.

| 112 ep (init 15–18) | 안전 성공 48 | 잃음 29 | 둘 다 실패 31 | 성공 + 충돌 4 |
|---|---:|---:|---:|---:|
| 큰 보정 call | 129 | 388 | 327 | 24 |
| · 쥐기 전 / 쥔 동안 / 놓은 뒤 | 21 / 92 / 16 | 92 / 178 / 118 | 178 / 87 / 62 | 13 / 11 / 0 |
| 큰 보정 5 call 이상인 ep | 8 | 26 | – | – |

- 잃은 29 ep 의 큰 보정 388 call: 최악 row step 0–4 = 77, 5–9 = 247, 기록 없음 64 (`to.passthrough_clear.obstacle_rows.worst.step`). **init 19 의 3 ep 에서 본 'step 0–4 쪽이 나쁘다' 와 달리 112 ep 에서는 5–9 쪽이 훨씬 많다.**
- 막은 link (잃은 ep 의 큰 보정, 최악 row): `panda_hand` 95 · `leftfinger` 81 · `link5` 52 · 쥔 물체 구 47.
- 사유 `limit` call: 144 ep 에서 24 개뿐 (`-lp` 의 효과 범위가 작다).
- 놓은 뒤 (잃은 ep): 530 call 중 close seed 가 다시 나온 call 206 (14 ep). 그때의 target 이 task 물체인지는 init 19 만 (`tt_tab10`); init 15–18 은 `target_truth.py` replay 필요.
- 막은 표면의 물체 (전달 문서 §6.1 의 2): 기존 `owner_held` (쥔 동안만) · `owner_tab10` (init 19) 를 seq 로 붙여 388 call 중 114 call 만. 나머지는 `surface_owner_task.py` replay (MuJoCo, 약 1 h) 가 필요해 미룸.

## §8.5 질문

1. **cherry-pick:** §0.
2. **unit test:** §0.
3. **동결 판 1 ep:** 같음 (§A.1).
4. **`-gh156a` gate 1 단계에서 > 50 mm 가 1 → 2 가 된 episode:** E1a **task 3 init 14** — 268.5 mm (`yellow_book_obstacle_1`), 결과 성공 + 충돌 (동결 판은 실패, 충돌 없음). 나머지 하나 (task 3 init 9, 238.8 mm wine bottle) 는 두 판이 같다.
5. **'동결 판 충돌 38 ep 의 원인' 네 줄 (계획서 §9.39 보충 / sdm-ahs §11.7):** 로봇 · 쥔 물체가 닿지 않음 (table 등) 12 · gripper 10 · 팔 link5/6 10 (6 ep 은 계획 clearance 가 이미 음수 −0.8 – −3.4 cm 인데 `budget_only` → 실행, 4 ep 은 양수) · 쥔 물체 5 · replay 실패 1. **정정 (§9.40):** gripper 10 ep 은 '손 구가 field 전체에서 빠짐' 이 아니다 — `exclude_authorized` 는 손 · 손가락 구를 대상 주변을 파낸 층 (table · 장애물은 남음) 에 묻는다. 10 ep = 장애물 오인식 5 + 두 층에 있는데 계획이 들어감 5.
6. **'gripper 가 빠지는 경로' 와 `--grasp-zone-release`:** 위 정정대로 gripper 는 field 전체에서 빠지지 않는다. 별도 분석 (L12H3 충돌 원인, 아래) 에서 팔 link 충돌의 다수가 **T88 통과 판정이 장애물 물체 row 가 아닌 row 의 위반을 100 mm 까지 받아 `clear` 로 실행한 것** 이었다 — 대상 옆 장애물의 표면이 그 call 에서 장애물 물체 집합에 들지 않았다. `--grasp-zone-release` 와의 관계는 재지 않았다.

## 덧붙임: L12H3 충돌 원인 (사용자 질문, CPU replay)

같은 56 ep 구성에서 충돌: 동결 판 9 · L12H3 16. L12H3 에만 생긴 충돌 10 = A (닿은 팔 · gripper link 의 계획 clearance 가 충돌 전 20 step 안의 chunk 에서 음수) 7 (실행된 앞 5 step 만 보면 5) · B (양수) 1 · E (로봇이 닿지 않음) 2. A 의 팔 link 는 대부분 TO `optimal` + T88 passthrough `clear` (예: spatial t1 e11, 접촉 step 134 직전 call 은 장애물 row 위반 0, 그 밖의 row 위반 15.2 mm, 허용 100 mm). SDM `outputs/2026-11-14_SL-AUD/l12h3_collision/`.

## 남은 것 (이 문서에 덧붙일 예정)

- §B 거르기 2 `-ohr-dlo-pro` (02:37–약 03:57): §8.1 · §8.2 (`history_refilter.n_removed`, 목적지 등록 call, `progress.mode`, 쥔 구간 거절), §8.3 6 ep.
- §C `-md5` (약 03:57–05:17).
- 통과가 있으면 채택 gate (X = `-gh156a`; L12H3 는 gate 1 단계 탈락: E1a 40 TSR 23 · CAR 28 · SSR 21 · > 50 mm 3).

---

## B. 거르기 2 `-ohr-dlo-pro` (2026-11-14 04:0x 덧붙임)

판 `…-hb-qs-rs-ohr-dlo-pro-tro`, HEAD `09dc923b`, TAG `aud2sdm`, 02:37:03–03:55:03. run: `20261114T023705_…_ms-aud2sdm-spatial-LI-i15_16_17_18`, `20261114T031255_…_goal-LI-i17_18`, `20261114T033440_…_goal-LII-i17_18`.

| S32 | 동결 판 | 거르기 1 (`-rf` 미작동) | 거르기 2 |
|---|---|---|---|
| Spatial L-I (16) | 7 · 12 · 5, 2 | 8 · 11 · 7, 2 | 8 · 13 · 7, 1 |
| Goal L-I (8) | 2 · 6 · 1, 1 | 2 · 6 · 1, 0 | 3 · 6 · 2, 1 |
| Goal L-II (8) | 4 · 5 · 3, 0 | 4 · 5 · 2, 1 | 5 · 5 · 4, 0 |
| **합** | **13 · 23 · 9, 3** | **14 · 22 · 10, 3** | **16 · 24 · 13, 2** |

(TSR · CAR · SSR, > 50 mm.) 기준 §6.3: SSR ≥ 11 ✓ (13), > 50 mm ≤ 4 ✓ (2), 동결 판 안전 성공 잃음 ≤ 1 ✓ (0) → **PASS** → 채택 gate 에 넣는다. 안전 성공 얻음 4 · 잃음 0; 잃은 13 중 돌아온 것 1 (goal L-I t2 e18).

| cell | task | init | π0.5 | 동결 판 | 거르기 1 | 거르기 2 | 변위 mm (동결 → 1 → 2) | 큰 보정 call (동결 → 1 → 2) | 잃은 13 |
|---|---:|---:|---|---|---|---|---|---|---|
| spatial L-I | 0 | 15 | F+C | SS | SS | **SS** | 0.0 → 0.0 → 0.0 | 0 → 0 → 0 |  |
| spatial L-I | 0 | 16 | F+C | S+C | SS | **SS** | 56.9 → 0.0 → 0.0 | 8 → 4 → 4 |  |
| spatial L-I | 0 | 17 | S+C | SS | SS | **SS** | 0.0 → 0.0 → 0.0 | 0 → 0 → 0 |  |
| spatial L-I | 0 | 18 | F+C | S+C | F+C | **S+C** | 5.0 → 5.8 → 5.0 | 9 → 5 → 9 |  |
| spatial L-I | 1 | 15 | S+C | F | F+C | **F** | 0.0 → 87.0 → 0.0 | 26 → 9 → 26 | ✓ |
| spatial L-I | 1 | 16 | S+C | F+C | S+C | **F+C** | 7.3 → 111.1 → 14.8 | 12 → 3 → 5 | ✓ |
| spatial L-I | 1 | 17 | SS | F | SS | **F** | 0.0 → 0.0 → 0.0 | 12 → 1 → 12 | ✓ |
| spatial L-I | 1 | 18 | S+C | F | F+C | **F** | 0.0 → 29.3 → 0.0 | 24 → 7 → 24 | ✓ |
| spatial L-I | 2 | 15 | S+C | SS | SS | **SS** | 0.0 → 0.0 → 0.0 | 1 → 2 → 1 |  |
| spatial L-I | 2 | 16 | S+C | F | F | **F** | 0.0 → 0.0 → 0.0 | 19 → 7 → 17 | ✓ |
| spatial L-I | 2 | 17 | S+C | SS | SS | **SS** | 0.0 → 0.0 → 0.0 | 2 → 3 → 2 |  |
| spatial L-I | 2 | 18 | F+C | F | SS | **SS** | 0.0 → 0.0 → 0.0 | 1 → 1 → 1 |  |
| spatial L-I | 3 | 15 | F+C | F | F | **F** | 0.4 → 0.6 → 0.4 | 6 → 6 → 6 |  |
| spatial L-I | 3 | 16 | S+C | SS | F | **SS** | 0.0 → 0.0 → 0.0 | 2 → 6 → 2 |  |
| spatial L-I | 3 | 17 | S+C | F | F | **F** | 0.0 → 0.0 → 0.0 | 6 → 1 → 8 | ✓ |
| spatial L-I | 3 | 18 | F+C | F+C | F+C | **F+C** | 294.5 → 4.9 → 294.5 | 5 → 12 → 5 |  |
| goal L-I | 0 | 17 | S+C | F+C | F | **F+C** | 2.2 → 0.0 → 2.2 | 22 → 20 → 22 | ✓ |
| goal L-I | 0 | 18 | S+C | F | SS | **F** | 0.0 → 0.0 → 0.0 | 16 → 0 → 14 | ✓ |
| goal L-I | 1 | 17 | S+C | SS | S+C | **SS** | 0.0 → 15.2 → 0.0 | 3 → 4 → 3 |  |
| goal L-I | 1 | 18 | S+C | S+C | F+C | **S+C** | 83.6 → 2.4 → 83.6 | 7 → 3 → 11 |  |
| goal L-I | 2 | 17 | F+C | F | F | **F** | 0.0 → 0.0 → 0.0 | 26 → 5 → 7 |  |
| goal L-I | 2 | 18 | S+C | F | F | **SS** | 0.0 → 0.0 → 0.0 | 29 → 10 → 3 | ✓ |
| goal L-I | 3 | 17 | F+C | F | F | **F** | 0.0 → 0.0 → 0.0 | 8 → 3 → 8 |  |
| goal L-I | 3 | 18 | S+C | F | F | **F** | 0.0 → 0.0 → 0.0 | 7 → 5 → 10 | ✓ |
| goal L-II | 0 | 17 | F+C | SS | S+C | **SS** | 0.0 → 33.4 → 0.0 | 4 → 4 → 4 |  |
| goal L-II | 0 | 18 | F | SS | SS | **SS** | 0.0 → 0.0 → 0.0 | 1 → 1 → 1 |  |
| goal L-II | 1 | 17 | S+C | F+C | F | **F+C** | 10.5 → 0.0 → 31.0 | 17 → 18 → 19 | ✓ |
| goal L-II | 1 | 18 | S+C | F | F | **F** | 0.0 → 0.0 → 0.0 | 16 → 8 → 15 | ✓ |
| goal L-II | 2 | 17 | S+C | F+C | F+C | **F+C** | 6.7 → 53.8 → 6.7 | 8 → 8 → 8 | ✓ |
| goal L-II | 2 | 18 | S+C | S+C | S+C | **S+C** | 2.2 → 2.2 → 2.2 | 0 → 0 → 0 |  |
| goal L-II | 3 | 17 | S+C | SS | F | **SS** | 0.9 → 0.0 → 0.9 | 11 → 7 → 11 |  |
| goal L-II | 3 | 18 | F+C | F | SS | **SS** | 0.0 → 0.3 → 0.0 | 10 → 4 → 2 |  |

**flag 가 실제로 바꾼 episode.** call 단위 (`max_abs_action_change_executed` · `passthrough_clear.reason`) 로 동결 판과 같은 ep 19 / 32, 다른 ep 13 — 처음 달라지는 call 은 대부분 20–46 번째 (쥔 뒤 · 놓는 구간), spatial t0 e16 은 7 번째, goal L-II t3 e18 은 8 번째. 얻은 4 ep 은 모두 다른 13 ep 안에 있다. (거르기 1 은 30 / 32 가 다르고 대부분 0–13 번째에서 갈림.)

**§8.2 의 놓는 구간 값 (32 ep 합, 동결 판 → 거르기 1 → 거르기 2).**

| 값 | 동결 판 | 거르기 1 | 거르기 2 |
|---|---:|---:|---:|
| 쥔 구간 거절 call (`grasp.state == held`, `applied == false`) | 353 | 292 | 296 |
| 놓은 뒤 거절 call | 101 | 51 | 94 |
| 진행 선호가 꺼진 call (`progress.mode = after the release`) | 0 | 0 | 74 |
| 놓은 뒤 'before the grasp: toward the target' call | 83 | 18 | 8 |
| 놓은 뒤 close seed call | 37 | 83 | 41 |
| 큰 보정 call (쥔 동안 / 놓은 뒤) | 318 (205 / 55) | 167 (107 / 17) | 260 (159 / 52) |

- **`-dlo`:** 기록 key 가 없어 간접 지표로 봤다 — 쥔 뒤 목적지가 처음 기록된 chunk 까지의 지연 (chunk 요약 `destination`) 분포가 동결 판과 같다 (중앙값 2–3, 최대 24), 등록된 ep 18 → 19. **목적지가 한 번도 등록되지 않는 ep 이 13 / 32** (동결 판 14).
- **`-ohr`:** `history_refilter.n_removed` 는 셀 수 없다 — `pipeline._scene_obstacle_points` 의 기록은 `constraint_set.metrics["obstacle_objects"]` 로 가는데 server 가 이 metrics 를 call 기록 (`calls.jsonl`) 에도 chunk 요약 (`summary_json`) 에도 저장하지 않는다 (동결 판도 없음). **요청:** 다음 판에서 셀 수 있도록 기록 경로를 더해 주세요 (기본 꺼짐 flag 와 같은 규칙).
- **H4 읽기 (D):** 쥔 구간 거절 −16 %, 안전 성공 +4 / −0. 그러나 목적지 등록 시점은 그대로이고 '아예 등록되지 않는' ep 이 그대로다 — 이득은 `-pro` (놓은 뒤 옛 target 쪽 끌림 83 → 8) 와 `-ohr` 쪽으로 보인다 (세 flag 를 따로 재지 않음, §6.3).

§8.3: 거르기 2 의 같은 6 ep 을 `raw6/aud2sdm/` 에 더했다.

## 일시 정지와 남은 순서 (04:00)

사용자 결정 (03:5x): 거르기 2 판정 직후 queue 를 멈추고 GPU 를 실물 RB-Y1 학습 (약 8 h) · 13:00 실물 시험에 먼저 쓴다. 03:55:03 에 시작한 `-md5` run (episode 0) 은 INTERRUPTED 표시. 재개 순서: `-md5` (§5 셋째 줄) → **거르기 1 재실행** (`0f1d90b4` 넣음, §10; 첫 ep 의 `reference_floor` 확인 → S32, TAG `aud1bsdm`) → 채택 gate (거르기 2 가 통과했으므로 반드시: X = `-gh156a` + `-ohr-dlo-pro`, 재실행도 통과면 + `-rf-lp-mr8-af`) → Tab. 1. §10 의 '떨어지면 action QP 쪽을 닫는다' 는 `-md5` 도 포함하는 것으로 읽었다 (`-md5` 는 `-rf` 미작동 거르기 1 의 갈림이므로 결과는 기록만).

---

## C. 재개 뒤: `-md5` · 거르기 1 재실행 · 채택 gate · 최종 동결 (2026-11-14 18:5x 덧붙임)

실물 RB-Y1 학습 · 시험 (04:05–14:09) 동안 queue 를 멈췄다가 14:16 에 재개 (`aud_queue_c.sh`). 순서: `-md5` → main 을 `0f1d90b4` 포함 commit (`93478890`) 으로 → `-rf` 첫 ep 확인 → 거르기 1 재실행 → 채택 gate → Tab. 1.

### C.1 `-md5` (§5 셋째 줄; §10 에 따라 기록만)

TAG `aud1md5sdm`, HEAD `09dc923b`, 14:16–15:35 (03:55 의 run 은 INTERRUPTED, 처음부터). S32 합 **TSR 16 · CAR 19 · SSR 12, > 50 mm 8** (Spatial L-I 7 · 7 · 5, 6 / Goal L-I 5 · 7 · 5, 1 / Goal L-II 4 · 5 · 2, 1). 기준: SSR ≥ 12 ✓, > 50 mm ≤ 4 ✗ (8), 동결 판 안전 성공 잃음 ≤ 1 ✗ (4) → FAIL. > 50 mm 8 ep: spatial t0 e16 151 · t0 e18 253 · t1 e15 214 · t1 e16 54 · t1 e18 251 · t3 e16 262, goal L-I t1 e18 722, goal L-II t1 e17 78 mm. 읽기 (D): 상한은 task 성공을 늘리지만 장애물을 크게 민다 — H2 쪽 근거. (판정 줄의 '잃은 13 중 큰 보정 5 call 이상 0 / 13' 은 상한 0.5 때문에 정의상 0.)

### C.2 `-rf` 첫 ep 확인 (§10)

`ms-aud1bchksdm-spatial-LI-i15` (고친 판, Spatial L-I task 0 init 15 = S32 spatial run 의 첫 episode 를 같은 인자로 따로; §A.1 의 동결 판 확인이 이 방식으로 call 단위 같음을 보였다), 15:35–15:44: **OK** — action-QP call 8 / 8 에 `reference_floor.{n_rows_below_backoff, min_floor_m}`, why / error / shape_mismatch 없음. `n_rows_below_backoff` 합 293 (call 당 중앙값 39), `min_floor_m` 중앙값 −5.2 mm · 최소 −33.3 mm.

### C.3 거르기 1 재실행 (§10)

판 `…-hb-qs-rs-rf-lp-mr8-af-tro` (거르기 1 과 같은 이름), HEAD `93478890`, TAG `aud1bsdm`, 15:44:51–17:02:45. run: `20261114T154453_…_ms-aud1bsdm-spatial-LI-i15_16_17_18`, `20261114T162105_…_goal-LI-i17_18`, `20261114T164306_…_goal-LII-i17_18`.

| S32 (TSR · CAR · SSR, > 50 mm) | 동결 판 | 거르기 1 (`-rf` 미작동) | **재실행** (`-rf` 작동) |
|---|---|---|---|
| Spatial L-I (16) | 7 · 12 · 5, 2 | 8 · 11 · 7, 2 | 7 · 9 · 7, 4 |
| Goal L-I (8) | 2 · 6 · 1, 1 | 2 · 6 · 1, 0 | 3 · 7 · 3, 0 |
| Goal L-II (8) | 4 · 5 · 3, 0 | 4 · 5 · 2, 1 | 5 · 5 · 4, 0 |
| **합** | **13 · 23 · 9, 3** | **14 · 22 · 10, 3** | **15 · 21 · 14, 4** |

기준 §5: SSR ≥ 12 ✓ (14), > 50 mm ≤ 4 ✓ (4, 경계), **동결 판 안전 성공 잃음 ≤ 1 ✗ (2: spatial t3 e16 → F+C 291 mm, goal L-II t3 e17 → F)** → **FAIL → action QP 쪽 후보를 닫음** (§10). 잃은 두 ep 은 거르기 1 · `-md5` 에서도 잃었다. 안전 성공 얻음 7 (spatial t0 e16 · t1 e17 · t2 e18, goal L-I t0 e17 · t0 e18, goal L-II t2 e17 · t3 e18), 잃은 13 중 돌아옴 4. > 50 mm: spatial t1 e15 88 · t1 e18 77 · t3 e15 **749** · t3 e16 291 mm.

**§10 의 더할 값.**

| 값 | 재실행 |
|---|---|
| `-rf` 가 적용된 action-QP call / 전체 | **628 / 628** (세 run 모두 `aud_rf_check.py` OK) |
| 접촉 row 중 5 mm 아래였던 것의 합 · call 당 중앙값 | 45,852 · 16–23 (run 별) |
| 가장 깊은 floor (`min_floor_m`) 중앙값 · 최소 | −18 – −25 mm (run 별 중앙값) · −91.2 mm |
| 큰 보정 call 의 최악 row step 0–4 · 5–9 | 31 · 120 (동결 판 61 · 178, 거르기 1 32 · 131) |

S32 32 ep 의 짝 (다섯 판):

| cell | task | init | π0.5 | 동결 판 | 거르기 1 | 재실행 | -md5 | 거르기 2 | 변위 mm (동결 → 재실행) | 큰 보정 (동결 → 1 → 재실행) | 잃은 13 |
|---|---:|---:|---|---|---|---|---|---|---|---|---|
| spatial L-I | 0 | 15 | F+C | SS | SS | **SS** | S+C | SS | 0.0 → 0.0 | 0 → 0 → 0 |  |
| spatial L-I | 0 | 16 | F+C | S+C | SS | **SS** | F+C | SS | 56.9 → 0.0 | 8 → 4 → 3 |  |
| spatial L-I | 0 | 17 | S+C | SS | SS | **SS** | SS | SS | 0.0 → 0.0 | 0 → 0 → 0 |  |
| spatial L-I | 0 | 18 | F+C | S+C | F+C | **F+C** | F+C | S+C | 5.0 → 5.8 | 9 → 5 → 3 |  |
| spatial L-I | 1 | 15 | S+C | F | F+C | **F+C** | F+C | F | 0.0 → 88.2 | 26 → 9 → 6 | ✓ |
| spatial L-I | 1 | 16 | S+C | F+C | S+C | **F+C** | F+C | F+C | 7.3 → 16.5 | 12 → 3 → 4 | ✓ |
| spatial L-I | 1 | 17 | SS | F | SS | **SS** | SS | F | 0.0 → 0.0 | 12 → 1 → 1 | ✓ |
| spatial L-I | 1 | 18 | S+C | F | F+C | **F+C** | F+C | F | 0.0 → 76.5 | 24 → 7 → 1 | ✓ |
| spatial L-I | 2 | 15 | S+C | SS | SS | **SS** | F | SS | 0.0 → 0.0 | 1 → 2 → 2 |  |
| spatial L-I | 2 | 16 | S+C | F | F | **F** | SS | F | 0.0 → 0.0 | 19 → 7 → 22 | ✓ |
| spatial L-I | 2 | 17 | S+C | SS | SS | **SS** | SS | SS | 0.0 → 0.0 | 2 → 3 → 1 |  |
| spatial L-I | 2 | 18 | F+C | F | SS | **SS** | SS | SS | 0.0 → 0.0 | 1 → 1 → 1 |  |
| spatial L-I | 3 | 15 | F+C | F | F | **F+C** | F+C | F | 0.4 → 749.1 | 6 → 6 → 16 |  |
| spatial L-I | 3 | 16 | S+C | SS | F | **F+C** | S+C | SS | 0.0 → 291.3 | 2 → 6 → 7 |  |
| spatial L-I | 3 | 17 | S+C | F | F | **F** | F | F | 0.0 → 0.0 | 6 → 1 → 1 | ✓ |
| spatial L-I | 3 | 18 | F+C | F+C | F+C | **F+C** | F+C | F+C | 294.5 → 1.5 | 5 → 12 → 13 |  |
| goal L-I | 0 | 17 | S+C | F+C | F | **SS** | SS | F+C | 2.2 → 0.0 | 22 → 20 → 10 | ✓ |
| goal L-I | 0 | 18 | S+C | F | SS | **SS** | SS | F | 0.0 → 0.0 | 16 → 0 → 0 | ✓ |
| goal L-I | 1 | 17 | S+C | SS | S+C | **SS** | SS | SS | 0.0 → 0.0 | 3 → 4 → 2 |  |
| goal L-I | 1 | 18 | S+C | S+C | F+C | **F+C** | F+C | S+C | 83.6 → 2.4 | 7 → 3 → 3 |  |
| goal L-I | 2 | 17 | F+C | F | F | **F** | SS | F | 0.0 → 0.0 | 26 → 5 → 6 |  |
| goal L-I | 2 | 18 | S+C | F | F | **F** | SS | SS | 0.0 → 0.0 | 29 → 10 → 11 | ✓ |
| goal L-I | 3 | 17 | F+C | F | F | **F** | F | F | 0.0 → 0.0 | 8 → 3 → 1 |  |
| goal L-I | 3 | 18 | S+C | F | F | **F** | F | F | 0.0 → 0.0 | 7 → 5 → 12 | ✓ |
| goal L-II | 0 | 17 | F+C | SS | S+C | **SS** | SS | SS | 0.0 → 0.0 | 4 → 4 → 0 |  |
| goal L-II | 0 | 18 | F | SS | SS | **SS** | SS | SS | 0.0 → 0.0 | 1 → 1 → 3 |  |
| goal L-II | 1 | 17 | S+C | F+C | F | **F+C** | S+C | F+C | 10.5 → 30.9 | 17 → 18 → 12 | ✓ |
| goal L-II | 1 | 18 | S+C | F | F | **F+C** | F | F | 0.0 → 7.7 | 16 → 8 → 7 | ✓ |
| goal L-II | 2 | 17 | S+C | F+C | F+C | **SS** | F+C | F+C | 6.7 → 0.0 | 8 → 8 → 1 | ✓ |
| goal L-II | 2 | 18 | S+C | S+C | S+C | **S+C** | S+C | S+C | 2.2 → 2.2 | 0 → 0 → 0 |  |
| goal L-II | 3 | 17 | S+C | SS | F | **F** | F | SS | 0.9 → 0.9 | 11 → 7 → 2 |  |
| goal L-II | 3 | 18 | F+C | F | SS | **SS** | F | SS | 0.0 → 0.0 | 10 → 4 → 0 |  |

§8.2 의 값 (다섯 판 나란히):

| 값 (32 ep 합) | 동결 판 | 거르기 1 | 재실행 | -md5 | 거르기 2 |
|---|---:|---:|---:|---:|---:|
| action-QP call | 754 | 608 | 628 | 613 | 671 |
| 큰 보정 (≥ 1) | 318 | 167 | 151 | 0 | 260 |
| · 쥔 동안 | 205 | 107 | 66 | 0 | 159 |
| · 놓은 뒤 | 55 | 17 | 39 | 0 | 52 |
| · 최악 row step 0–4 | 61 | 32 | 31 | 0 | 59 |
| · 최악 row step 5–9 | 178 | 131 | 120 | 0 | 161 |
| ±1 포화 성분 합 | 3735 | 1847 | 1867 | 805 | 3100 |
| slack > 15 mm call | 177 | 57 | 69 | 183 | 130 |
| QP 에서 뺀 row (마지막 iteration) | 930 | 30953 | 32935 | 812 | 869 |
| 접촉 row 5 mm 아래 (`-rf`) | 0 | 0 | 45852 | 0 | 0 |
| `limit` 통과 call | 0 | 9 | 6 | 0 | 0 |
| 쥔 구간 거절 call | 353 | 292 | 256 | 197 | 296 |
| IK 한도 call | 316 | 290 | 286 | 253 | 286 |
| OSQP 한도 call | 148 | 41 | 28 | 190 | 110 |

**읽기 (§10 에 결과 전에 적은 대로).**
- `-rf` 는 쥔 동안의 큰 보정을 107 → 66 으로 더 줄였고, `-mr8` 이 건드리지 않는 step 5–9 쪽도 131 → 120 으로 줄였다.
- 큰 보정이 더 줄면서 SSR 은 10 → 14 로 올랐다 → 'SSR 이 그대로면 H1 을 닫는다' 는 해당하지 않음. 그러나 같은 두 ep 을 늘 잃고 > 50 mm 가 경계 (4, 그중 749 mm 1) 에 닿아 정해 둔 기준을 넘지 못했다.
- 거르기 1 의 새 충돌 Spatial L-I t1 e15 · e16 중 e15 는 재실행에서도 F+C 88 mm, e16 은 F+C 16.5 mm.

### C.4 채택 gate (§7)

거르기 2 만 통과 → `…-rs-gh156a-ohr-dlo-pro-tro` (TAG `audgsdm`, HEAD `93478890`, 17:02–18:40), `-gh156a` 와 같은 기준 (계획서 §9.2 / §9.37, 기준 판 = 동결 판).

| E1a 40 (TSR · CAR · SSR, > 50 mm) | 값 |
|---|---|
| 동결 판 | 23 · 34 · 21, 1 |
| X = `-gh156a` | 25 · 33 · 22, 2 |
| X + `-ohr-dlo-pro` | **22 · 33 · 20, 3** |

1 단계 **FAIL** (SSR ≥ 22 또는 21 · CAR ≥ 36 를 못 채움) → guard 는 돌리지 않음. X 대비 짝: 안전 성공 얻음 0 · 잃음 2 (task 1 init 13 → F, task 3 init 8 → F); task 별 SSR 동결 판 7 · 2 · 7 · 5 / X 7 · 3 · 7 · 5 / 이 판 7 · 2 · 7 · 4. 읽기 (D): S32 의 +4 / −0 이 E1a (Spatial L-I 한 cell, init 5–14) 로 옮겨가지 않았다 — S32 는 놓는 구간 문제가 몰린 Goal cell 을 담는다.

### C.5 최종 동결과 Tab. 1

**최종 동결 판 = X = `…-hb-qs-rs-gh156a-tro`** (§7: 채택 gate 탈락 → X). SL-AUD flag 는 판에 들어가지 않는다. Tab. 1: 18:42 시작, AG3S arm = `…-rs-gh156a-fwd-tro` (`--attention-inpass` · `--graspable-hand-sized-mm 156` 확인), GT oracle arm = `…-rs-tro-gto`; 예상 끝 AG3S ~11-15 05:00, GT oracle ~11-15 13:00.

### C.6 로컬 PC 에 남기는 것

- **`-ohr` 기록:** `metrics["obstacle_objects"]` (`history_refilter.n_removed` 포함) 가 call · chunk 기록에 저장되지 않는다 (§B). 다음에 잴 때 셀 수 있게 기록 경로가 필요하다.
- **단위 test 와 live 경로:** `-rf` 의 shape 문제는 CPU 의 stub · monkeypatch 가 2 차원을 돌려줘 숨었다. `0f1d90b4` 의 실제 `EsdfField` test 처럼, 새 flag 는 서버 경로 (`--esdf-backend curobo --to-fast-eval`) 에서 기록 key 가 나오는지 첫 ep 으로 먼저 보는 것이 안전하다 (이번 §10 의 첫 ep 확인이 그 역할).
- **S32 와 E1a 의 차이:** 놓는 구간 후보는 S32 에서 통과하고 E1a 에서 떨어졌다. 다음 후보를 거를 때 셋이 그 후보가 고치는 문제를 담는지 먼저 보는 것이 좋다.
- §8.3 raw: `raw6/{pkgFrssdm, aud1sdm, aud1bsdm, aud1md5sdm, aud2sdm}/` (다섯 판 × 6 ep).
