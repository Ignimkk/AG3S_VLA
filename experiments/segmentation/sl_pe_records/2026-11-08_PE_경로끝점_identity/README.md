# SL-PE — target identity = VLA 가 손을 닫는 곳 (2026-11-08 19:0x–, dev-30)

- 기준 판: B5 GT 없음 (`AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100`, tag gt24att).
- 판정 기록: `doc/active/experiments/E1_GTTO_DECISIONS_20261104_KO.md` G150 부터. 분담: dev-30 = identity · 장애물 집합, dev-38 (구 dev-e5) = 정량 · SOTA · B5 (GT) 한계 (G102).
- **규약 표시:** offline 설계는 평가 seed (gt24att LIBERO seed 7 · 8 · 9, SafeLIBERO seed 7) 기록 위에서 했다 = **in-sample**. 아래 closed-loop step 1 은 그 실패 task 만 다시 돈 **기제 확인 (mechanism check)** 이고 평가가 아니다. 최종 평가는 보정 seed 10 · 11 확인 → seed 7 · 8 · 9 전체 27 (성공 task 회귀 포함) → SafeLIBERO 11 · LS 10 순서로 한다.

## 왜 (dev-38 정량 `../2026-11-08_GT없는_인식_정량/`)

- GT 없는 판의 실패 16 중 12 는 첫 identity 가 다른 물체 (distractor · 목적지 · 서랍 cluster), 2 는 target 을 못 냄, 2 (LIBERO t7) 는 identity 는 맞고 쥔 뒤 들린 그릇이 장애물로 남음.
- attention 은 대부분 맞다 (물체 중 target mass 1 위 60–87 %). 틀린 것은 **3D 분할**: cabinet 윗면 · 서랍 위 / 안의 그릇이 fixture 와 한 성분이 되어 잡을 수 없는 크기 → 후보에서 사라진다.

## 신호 — VLA chunk 의 close 지점 (`pe_signal.py`, `signal_*.json`)

- `plan_close` = 들어오는 chunk 에서 gripper 명령이 latch 문턱 아래인 첫 row 의 손끝 중점 (T43 부터 `grasp.tip_alignment.plan_close` 로 기록, 측정 전용).
- close 2 call 전 그 점에 가장 가까운 실제 물체 = **target 34 / 36 episode** (예외 SafeLIBERO t2e0 · t3e3).
- `plan_close` 는 첫 closing 보다 2–4 call 먼저 나온다. gt24att 의 LATCHED 는 seq 2–5 (GT 없는 판은 `latched_identity_hold` 꺼짐 — LATCHED 가 identity 를 고정하지 않는다. 고정한 것은 TargetConfirm 의 3 frame streak 규칙과 분할).

## 방법 — SL-PE1 · 2 · 3 (모두 기본 꺼짐)

| flag (clustering key) | 무엇 | 근거 |
|---|---|---|
| `--close-seed-grounding` (`close_seed_grounding`) | chunk 에 close 명령이 있고 attach 전이면, grounding 입력 = `plan_close` 손끝 아래에서 잘라낸 물체 (`AG3S/ag3s/stages/close_seed.py`). `TargetConfirm` 은 그 물체를 바로 채택 (`close_seed_switch`, 3 frame streak 없이). 없으면 attention grounding 그대로 | 위 신호 · 분할 실패 |
| `--destination-after-held` (`destination_after_held`) | 인식 목적지 등록 · 갱신은 쥔 동안만 | dev-38: 쥐기 전 cabinet+그릇 cluster 가 목적지로 등록 (L t9 · S t3) |
| `--scene-obstacle-held-exclude` (`scene_obstacle_held_exclude`) | attach 중이면 지금 손 안 쥔 물체 점 15 mm 안 scene 성분도 장애물 물체에서 뺌 | dev-38 held_m4: 실패 t7 HELD chunk 65 / 67 에서 쥔 그릇이 장애물 |

`split_at_seed` (학습 없음, 결정적):
1. seed 15 cm 안 비-받침 점;
2. 국소 받침 = 위를 향한 법선 (|n_z| ≥ 0.9) 띠 (15 mm) 중 윗면이 seed + 2 cm 아래이고 x-y 1 cm 칸을 가장 많이 덮는 것 (≥ 40 칸, 폭 ≥ 10 cm) — cabinet 윗면 · 서랍 바닥 · 상자 뚜껑; 그 윗면 + 4 mm 아래 점 제거;
3. 10 mm 격자 26-연결 성분 중 seed 에 가장 가까운 것 (≤ 6 cm);
4. 그 성분이 15 cm 를 넘으면 평평한 수직면 (RANSAC, 폭 ≥ 10 cm, 면 칸 채움 ≥ 40 %) = 서랍 · cabinet 벽을 빼고 다시; 20 cm 를 넘으면 seed 8 cm 안만.

판: `…-t100-pe1` (= gt24att + PE1), `…-t100-pe123` (+ PE2 + PE3) — `AG3S/adapters/panda/profile.py` `PE_VERSIONS`.

## offline replay (`pe_replay.py`, `replay_*.json`, `replay_table.md`, `pe_summary.py`)

기록된 점군 위에서 규칙만 바꿔 다시 판정 (팔 궤적은 기록 그대로 = 인식만의 반사실). GT 는 채점에만 (dev-38 `quantify.py` 의 장면 · 역할).

![graph](pe_graph.png)

| 묶음 (close 명령이 있는 쥐기 전 call, R0 기록이 있는 call) | call | R0 정답 | PE 정답 | target 이 장애물 집합에 (R0 → PE) |
|---|---|---|---|---|
| LIBERO 성공 episode | 102 | 99 % | 98 % | 0 → 5 % |
| LIBERO 실패 episode | 187 | 6 % | 89 % | 80 → 7 % |
| SafeLIBERO 성공 | 41 | 32 % | 100 % | 44 → 0 % |
| SafeLIBERO 실패 | 179 | 6 % | 44 % | 45 → 16 % |

- PE 가 자른 성분의 순도 중앙값 1.00, recall 0.75–1.00 (`replay_fair.txt`).
- 늦은 잠금 흉내 (PEL: plan_close 성분이 2 call 연속 같은 곳일 때 잠금) 의 첫 잠금 identity = target 33 / 36 (잠금 없음 2, cabinet 1 = SafeLIBERO t3e3).
- 한계: SafeLIBERO 실패 episode 는 44 % — 그 기록의 손 경로가 이미 개입으로 바뀌어 다른 물체 쪽으로 갔다 (t2e0: close 지점 최근접 = 다른 물체). LIBERO 성공 episode 의 장애물 섞임 0 → 5 % (자른 성분 recall 0.75, 받침 띠 아래 그릇 바닥이 남음).

## 실제 장면

![scene](pe_scene.png)

`pe_scene_figure.py` (`pe_scene.json`): seed 8 t9 · t4 · t3 의 같은 call. 별 = `plan_close` 손끝, 파랑 = PE1 이 자른 물체, 주황 = 기록된 identity R0, 빈 원 = GT target 표면 (채점용).
t9: R0 = 탁자 위 다른 검은 그릇 (crop 밖), PE1 = cabinet 위 그릇 (순도 1.00 · recall 0.84). t4: 서랍 안 그릇 + 서랍 앞판 윗모서리 한 줄 (순도 0.91). t3: R0 = 목적지 접시, PE1 = 상자 위 그릇.

## closed loop step 1 — 기제 확인 (`closed_loop/`, `experiments/segmentation/pe1_queue.sh` → `pe1b_queue.sh`)

비교 도구 `cl_compare.py <tag>` (같은 task · seed · init 의 gt24att 와 episode 별).

### v1 (pe123, 20:14–20:56, LIBERO seed 8 만 — 그 뒤 멈춤)

| episode | 성공 gt24att → v1 | 개입 call 쥐기 전 / 쥔 뒤 / HOLD | seed 전환 / 첫 closing | attach |
|---|---|---|---|---|
| t3 | FAIL → FAIL | 23 / 0 / 0 → 2 / 1 / 0 | 7 / 9 | 없음 → 있음 |
| t4 | FAIL → FAIL | 29 / 0 / 3 → 15 / 0 / 3 | 20 / 24 | 없음 → 없음 |
| t7 | FAIL → **ok** | 0 / 19 / 0 → 0 / 4 / 0 | 5 / 7 | 있음 → 있음 |
| t9 | FAIL → FAIL | 20 / 0 / 0 → 5 / 1 / 0 | 11 / 12 | 없음 → 있음 |

v1 에서 찾은 두 가지 (둘 다 PE1 안에서 고침, 21:01 · 21:02 적용, 이후 run = pe123b):
1. **seed 로 자른 그릇이 `not_supported` 로 거절** (t9 call 7: 받침 test 의 plane 은 탁자뿐, 그릇은 그 위 240 mm). 전환은 손이 18.7 mm 까지 온 call 11 에야 (손 안 예외). → 자른 물체의 국소 받침면을 그 call 의 받침 test plane 에 더함 (`pipeline._close_seed_grounding_input`).
2. **closing 중 identity 가 오감** (t4 call 24–41: 그릇 ↔ 서랍 조각, t9 call 12). seed 가 움직일 때마다 다른 조각을 자름. → seed 로 고른 identity 는 closing 부터 잠금 (`TargetConfirm._seed_adopted_id`; attention 으로 고른 identity 만 closing 중에도 seed 로 바꿀 수 있음). 사용자 제안 2 ('close 직전, 손이 향하는 물체로 잠그기') 를 이 형태로 넣었다.

### v2 이후 (pe123b · pe12345 · pe125cls · pe1234567 · pe1234567b)

결과 표와 다음 할 일은 `다음세션_프롬프트.md` §2–§4, 판정은 G155–G161, 현재 구성은 `doc/active/experiments/AG3S_FINAL_CONFIG_KO.md`.

## test

- `tests/ag3s/test_sl_pe1_close_seed.py` (5): cabinet 위 그릇 · 벽 붙은 서랍 그릇 · 탁자 받침 제외 · seed 가 멀 때 None · 보조 함수.
- `tests/ag3s/test_sl_pe_wiring.py` (8): 기본 꺼짐 · 검증, 꺼지면 3 frame 규칙 그대로, 켜면 즉시 switch (attach 뒤에는 안 함), 목적지 등록은 쥔 동안만, pipeline 입력 자르기, scene 장애물에서 쥔 물체 빼기, 판 argv.
- 회귀 216 통과 (target_grounding · t43t_identity · manipulated_identity · sl_gtto16 · sl_gtto2_versions · sl_gtto_gt_perception · sl_qe1 · merge_dev_main_20261103_panda · sl6h), 적용 전 같은 내용의 overlay 에서.
