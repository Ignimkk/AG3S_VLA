# AG3S-CORE-EVAL-HSR6 — 구현 (6 단계 HSR 을 서빙에 넣기, 기본 꺼짐 4 flag)

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe · 작성 2026-10-29 04:20 KST (`date -u` 19:20 + 9 h)
> task: [AG3S-CORE-EVAL-HSR6.task.md](AG3S-CORE-EVAL-HSR6.task.md) §1–3 (§4 closed-loop 은 lead) · 바탕: [AG3S-CORE-EVAL-G1.impl.md](AG3S-CORE-EVAL-G1.impl.md) 의 메인 작업 트리 (G1 합치기 끝난 상태, md5 대조로 확인)
> 원시 기록 · 도구: `$AG3S_STORAGE_ROOT/results/development/20261029T032907+0900_461231a432a2_2a8bef6076d6/verify/AG3S-CORE-EVAL-HSR6/` (아래 `$R`)

## 무엇을 했나 (평이한 요약 먼저)

- 세션 B 의 6 단계 probe (T64 Part A `probe_6.npz`) 를 서버에 넣었다. 지금 HSR (pick / place / home) 이 이미 쓰는 같은 `kv_L4` feature 에 붙인다. 추가 forward 는 없다.
  - 요청마다 6 단계 확률과 raw argmax 를 `result["hsr6"]` 와 `summary_json.hsr6` 에 싣는다 (flag 가 하나라도 켜졌을 때만).
  - 지금 서빙 HSR (3 단계, subtask gate · B3 · placed_no_seed_certify) 은 그대로다.
- task §2 의 4 사용처를 각각 **기본 꺼짐 flag** 로 넣었다: `--hsr6-attach-assist` ① · `--hsr6-held-gate` ② · `--hsr6-target-exempt` ③ · `--hsr6-objective` ④.
  - ④ 는 G1 이 켜져 있으면 G1 slot (기하 grasp 목표) 도 같은 규칙으로 막는다.
- **꺼짐 = 이전과 같다.** 핵심 평가 기록 106 요청 재생에서 refined chunk 106/106 bit 동일 (H2). TrajOpt 재생이 닿지 않는 grasp latch · 기록 · AG3S held 구 경로도 합성 순서열 6 개에서 전과 같다 (H2b).
- **서버 단계 = 세션 B 단계.** 기록 feature 1,200 frame (partB.json 대조) 과 7,500 frame (t64_partB 계산 대조) 모두 100 % 일치 (H3).
- 4 규칙마다 "지금 규칙과 결정이 달라졌나" 를 `summary_json.hsr6.changed` 에 요청마다 남긴다 (H4).

## 0. 시작할 때 확인한 것

- 메인 작업 트리의 G1 관련 7 파일 md5 가 G1 impl §2 와 같다 (config · grasp_anchor · refiner · safe_policy · serve_safe · sqp · test_core_eval_g1_merge). G1 위에서 시작했고, G1 코드는 지우거나 바꾸지 않았다 (`geo_grasp.py` 에는 메서드 2 개만 더함).
- 변경 전 트리 사본: scratch `pre_hsr6/` (.py · npz · json) 와 재생용 `pre_tree/` (변경 6 파일만 이전 판). H2 · H2b 의 "변경 전" 쪽이다.
- **SubtaskProbe 가 classes 를 npz 에서 읽는가:** 읽는다 (`AG3S/ag3s/stages/subtask_probe.py` `SubtaskProbe.load` — `classes=[str(c) for c in z["classes"]]`). `probe_6.npz` 에는 `name` 키가 없어서 이름은 파일 stem 이 된다. 그래서 서빙용 사본 이름을 `hsr6_t64_kv_L4` 로 정했다. 적재할 때 class 순서가 `approach · grasp · lift · transport · placement · retreat` 이고 dim 이 512 인지 확인하고, 아니면 거절한다 (`hsr6.load_hsr6_probe`).
- `SubtaskProbe.proba` 와 offline 의 `t50_core.LogReg.proba` 는 같은 식이다. 무작위 200 feature 에서 `np.array_equal` 로 같다 (test).

## 1. 지금 서빙에서 단계 · latch 를 읽는 곳 (task 요청: 의존 지도)

client 는 `--safe-phase approach` 를 고정으로 보낸다 (`integration/rby1/bringup/pi05_infer.py:899`). 그래서 AG3S `process_multi_debug(phase=...)` 의 단계는 늘 approach 다. 단계에 따라 바뀌는 기능은 대부분 grasp latch 상태를 읽는다.

| 기능 | 읽는 것 | 코드 (`AG3S/trajopt/safe_policy.py`) | HSR6 사용처 |
|---|---|---|---|
| attach / detach (파지 확인 · 놓기 · 회수 · slip) | latch `observe_grasp` 의 checks (`closed` · `blocked` · `settled` …) | `_advance_grasp` `:1541`, latch `grasp_latch.py:506` | **①** |
| AG3S target 동결 · challenger 무시 | CLOSING ∨ HELD | `:992` `set_grasp_active` | 없음 |
| latched_identity_hold | LATCHED ∨ CLOSING | `:997` `set_grasp_latched` | 없음 |
| subtask gate (3 단계 HSR) · B3 release · placed_no_seed_certify | 3 단계 HSR 확률 + PLACED | `:1004` `set_subtask`, `:1007` `set_placed` → `target_grounding.py:979,987` | 없음 (그대로) |
| held-body cover 의 지지면 들어올림 (쥔 몸통 – 탁자 · 바구니 바닥) | 쥔 물체가 있음 (HELD 뒤 attach) | AG3S `pipeline.py` `_frame_held_attached` | **②** |
| target 권한 (target 없는 거리장 · 권한 link target 마진 · target 판정) | AG3S 의 manipulated (쥐지 않았을 때) | `scene_from_constraint_set` → `SceneSnapshot` (`linearize.py:2134`) | **③** |
| 이웃 접촉 허용 (IFG-NBC) | LATCHED ∨ CLOSING + 손바닥 거리 | `_neighbour_contact_gate` `:1916` | **④** |
| 닫는 행 목표 A4 (IFG-TOA / E1-C) | LATCHED ∨ CLOSING + 닫기 계획 | `_grasp_anchor_gate` `:2007` | **④** |
| G1 기하 grasp 보정 | geo task state (latch phase) + 닫기 계획 | `_geo_request` `:1105`, `geo_grasp.py` | **④** |
| 손목 추종 배율 (IFG-TOW) | LATCHED ∨ CLOSING | `_wrist_track_gate` `:1966` | 없음 |
| 연속성 항 · 꺾임률 항 끄기 | latch phase | `:1750` · `:1771` · `:1792` | 없음 |
| 회복 (IFG-REC) | latch phase | `_recovery_observe` `:1808` | 없음 (A8 에서 꺼짐) |
| HELD 중 task gate (T57-A) | HELD | `_task_gate` `:2382` | 없음 (A8 에서 꺼짐) |

## 2. 4 사용처 — 규칙과 코드 위치

규칙 함수는 모두 `AG3S/trajopt/hsr6.py` 에 있다 (순수 함수, test 로 참 · 거짓을 고정). 요청에 HSR6 label 이 없으면 (probe 실패) phase = None 이다. 그때 ① 보완 없음, ② 허용 없음 (엄격한 쪽), ③ 지금 규칙, ④ 목표 항 없음.

| # | flag | 규칙 (코드) | 걸리는 곳 | "결정이 달라졌다" (H4) |
|---|---|---|---|---|
| ① | `--hsr6-attach-assist` | CLOSING 에서 strong 파지 확인이 실패해도 phase ∈ {lift, transport, placement} ∧ `closed` ∧ `blocked` 이면 attach (`hsr6.attach_assist_ok`). T64 Part C 와 같은 식 | `GraspLatch.observe_grasp(attach_assist={"phase"})` (`grasp_latch.py:606`), `_advance_grasp` `:1635` 에서만 kwarg 를 준다 | 보완 attach 가 실제로 걸린 요청 (`AG3S.attach` 의 마지막 관문을 통과). 거절되면 `detail.attach_assist.rejected_by_ag3s` |
| ② | `--hsr6-held-gate` | 쥔 물체 허용 = phase ∈ {lift, transport, placement} ∧ (`blocked` ∨ latch HELD) (`hsr6.held_allowance`). 거짓이면 그 frame 의 held-body cover 를 **들어 올리지 않은 몸통 구**로 둔다 (탁자 · 바구니 바닥 위 허용을 거둠) | `SafePolicy._hsr6_held_gate` `:1212` (AG3S frame 전, `:1009`) → `AG3S.set_held_allowance` (`pipeline.py:541`) → `_frame_held_attached` (`pipeline.py:1551`) | 이 frame 에서 들어올림이 거두어졌다 (`held_record.body_cover.frame.hsr6_lift_withdrawn`, `:1227` 이 읽음) |
| ③ | `--hsr6-target-exempt` | phase = retreat ∧ 쥔 것 없음 → target 권한을 모두 거둔다 (`hsr6.strip_target_permission`): target 없는 거리장 mask · 권한 link target 마진 · manipulated 점 · target 판정 행 · target 부피 → None. 놓은 사과는 장애물 행 (obstacle_margin 적용), 판정에서도 `no_manipulated` 충돌. approach · grasp · lift · transport · placement 는 지금 규칙 | `SafePolicy._hsr6_target_gate` `:1201` (`_scene_fn` `:1085`, snapshot 직후 · 판정 snapshot 과 같음). GPU seed 도 같은 snapshot 을 읽는다 | 거둘 권한이 실제로 있었다 (`detail.target_exempt.stripped` 에 필드 이름) |
| ④ | `--hsr6-objective` | 지금 규칙이 켠 이웃 접촉 허용 · A4 닫는 행 목표 · G1 slot 을 phase = grasp 일 때만 둔다 (`hsr6.objective_allowed`) | NBC: `_neighbour_contact_gate` `:1958` (기존 조건을 다 통과한 뒤 마지막). A4: `_hsr6_a4_gate` `:1192` (`:1061`). G1: `_geo_request` `:1121` → `GeoGraspController.withdraw_anchor` (`geo_grasp.py:390`) | 지금 규칙이 켰는데 HSR6 가 막았다 (`detail.objective`: `nbc_withdrawn` · `a4_withdrawn` · `g1_withdrawn`) |

규칙을 코드로 옮기며 정한 것 (task 문장을 좁힌 곳, 결과 전):

- **② 의 "held-body cover, 쥔 사과–탁자 · 사과–바구니 접촉 허용"** 을 held-body cover 의 frame 별 지지면 들어올림 (T43 R2 + T34 J1 규칙) 으로 옮겼다.
  - 바구니 바닥도 이 규칙으로 덮인다 (`_frame_held_attached` docstring: 바닥은 탁자 평면 8 mm 위, 실행 경로로 판정).
  - 쥔 행의 목적지 마진 (`destination_margin` 20 mm) 은 A8 의 `esdf_margin 0` 보다 **엄격**하다. 허용이 아니라서 건드리지 않았다.
  - 그래서 `--hsr6-held-gate` 는 `--held-body-cover` 없이 주면 거절한다.
  - attach 그 자체 · 쥔 손의 접촉 권한 (T44b) · `--held-cover-cap-mm` · look-ahead 꼬리의 support / 잔상 행 제외는 그대로다.
- **③ 은 retreat 에서만 바뀐다.** 쥔 동안 (attach 있음) 에는 지금 규칙이다 ("쥔 물체 처리 그대로").
- **④ A4:** 목표를 얼리는 일 (`_grasp_goal`) 은 지금 규칙대로 하고, TO 에 항을 넣는 것만 막는다.
- **④ G1:** G1 이 목표를 낸 chunk 를 막으면 그 chunk 의 G1 전부가 꺼진다 (목표 항, 쥐는 팔 추종 0.25, 연속성 끔, 보정 닫기 gate). controller 의 `started` · `n_goal_chunks` 는 부르기 전 값으로 되돌린다. 그래서 G1 의 쥔 뒤 추종 복원도 G1 이 실제로 쓰인 시도에만 걸린다.

## 3. 바뀐 파일 · md5 (메인 작업 트리, 이 STEP 끝)

줄 수는 변경 전 사본 (`pre_hsr6/`) 대비 `diff` (+ 더한 줄 / − 지운·바꾼 줄).

| 파일 | 변경 | md5 | 무엇이 |
|---|---|---|---|
| `AG3S/trajopt/hsr6.py` (**새**) | 153 줄 | `c97e6c1719876100026b37b9d51daf5b` | 단계 상수, `Hsr6Flags`, `load_hsr6_probe`, `phase_of`, 규칙 함수 4 개, `strip_target_permission` |
| `AG3S/ag3s/asset/subtask_probe/hsr6_t64_kv_L4.npz` (**새**) | 사본 | `f0bb92aaf6869276c37786d6d2351917` (= storage `T64/partA/probe_6.npz`) | 6 단계 probe |
| `AG3S/ag3s/asset/subtask_probe/hsr6_t64_kv_L4.json` (**새**) | | `82e8f1031647aae2126e88cc5261d2bd` | 출처 · md5 · class 묶음 · 적용 식 |
| `AG3S/trajopt/attention_policy.py` | +30 −11 | `8f16644266b34cbcb33d76009dbd7689` | `hsr6_probe=None` 인자, `_want_feature`, `_label_feature` (`result["subtask"]` 그대로 + `result["hsr6"]`) |
| `AG3S/trajopt/safe_policy.py` | +132 −2 | `647f2f790afbef37d9b108fe6569f167` | `hsr6=None` 인자 (+ class 기본값), §2 의 4 gate, `_hsr6_*` 메서드, `summary_json.hsr6`, metadata `hsr6` (켰을 때만) |
| `AG3S/trajopt/grasp_latch.py` | +24 −1 | `40bd92f9694a56ea439d516e7ffced76` | `observe_grasp(attach_assist=None)` |
| `AG3S/ag3s/runtime/pipeline.py` | +18 −1 | `6c4563953a15578365a4233ee36fbd01` | `set_held_allowance`, `_frame_held_attached` 의 거둠 + 기록 키 2 개 (켰을 때만) |
| `AG3S/trajopt/geo_grasp.py` | +16 −0 | `3197f05847e3a85b11aa3a6003f4a6f2` | `anchor_state` · `withdraw_anchor` |
| `AG3S/trajopt/serve_safe.py` | +73 −1 | `d7d916352366bae066a08ed1b2e360ae` | `--hsr6-*` 5 flag, `hsr6_flags_from_args` · `hsr6_record_meta` · `reject_hsr6_flags`, AttentionPolicy / SafePolicy / 기록 meta 배선 (켰을 때만) |
| `tests/trajopt/test_core_eval_hsr6.py` (**새**) | 55 test | `bdb36df4b55d2f1dc905d672d2c69f14` | §4 H1 |

손대지 않은 것: `config.py` · `grasp_anchor.py` · `refiner.py` · `sqp.py` · `linearize.py` (G1 md5 그대로), 문서 · figure.

## 4. 검증 (task §3)

모두 CPU · 1 process · `nice -n 10` · `.venv-ag3s` · 메인 checkout. 로그 첫 줄 `trees:` 로 module 위치를 확인했다.

| # | 검사 | 결과 | 판정 |
|---|---|---|---|
| H1 | 단위 test | 새 test 55/55. 전체 `tests/trajopt` + geo 2 + subtask gate: 아래 | **통과** (실패 1 = 알려진 기존 실패) |
| H2 | 꺼짐 동등성 (G1 V2 와 같은 방식) | 106 요청 (REG ep211 75 + OCC 121 31): 변경 전 트리 = 변경 후 **106/106 bit 동일**, 기록 (시간 필드 제외) 106/106 같음. G1 V2 의 `merged` 결과와도 106/106 bit 동일 | **통과** |
| H2b | 꺼짐 동등성 (재생이 닿지 않는 경로) | 합성 순서열 6 개 (attach→놓기 · 이른 닫힘 · 빈손 · AttentionPolicy · held 몸통 근처 / 먼 곳): chunk 기록 17 개 포함 6/6 같음 | **통과** |
| H3 | 단계 parity | (a) partB.json 의 `hsr` 대비 16 episode 1,200 frame **1,200/1,200**. (b) t64_partB 계산 대비 100 파일 7,500 frame **7,500/7,500**, 확률 최대 차 0.0 | **통과** (≥ 99 %, ≥ 50) |
| H4 | 발동 기록 | `summary_json.hsr6 = {phase, p, probe, flags, changed{켠 flag 마다 bool}, detail}`. test 로 각 규칙의 참 · 거짓 표시를 고정 | **구현** (lead 가 closed-loop 에서 센다) |

### H1 — 단위 test

```bash
cd /mnt/dev/AG3S_VLA-development
MUJOCO_GL=osmesa PYTHONPATH=$PWD OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 10 \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt tests/ag3s/test_geo_grasp.py \
  tests/ag3s/test_geo_grasp_obstacles.py tests/ag3s/test_subtask_gate.py
```

| 실행 | 결과 | 로그 |
|---|---|---|
| 1 차 (최종 직전 코드) | 2176 passed, 34 skipped, **3 failed**: 알려진 `test_t43t_replay::test_off_is_bit_identical_to_the_pre_fix_commit` + 2 개 (`test_ifg_nbc_neighbour_contact` · `test_scm_palm_shell_margin`) — `__init__` 없이 만든 SafePolicy test double 이 `self.hsr6` 를 못 찾음 | `$R/H1/h1_pytest.log` |
| 고침 | `SafePolicy` 에 class 기본값 `hsr6 = None` · `_last_hsr6 = {}` | |
| 최종 코드 전체 | **2178 passed, 34 skipped, 1 failed** = 알려진 기존 실패 `test_t43t_replay::test_off_is_bit_identical_to_the_pre_fix_commit` (`.venv-ag3s` numpy golden; G1 · PORT · GGW impl 에 같은 기록) — 23 분 | `$R/H1/h1_final_pytest.log` |

새 test (`test_core_eval_hsr6.py`) 가 고정하는 것:

1. probe — 동봉 사본 md5 = T64, classes 를 npz 에서 읽음, 잘못된 class 순서 거절, 서빙 `proba` = offline `LogReg.proba` (bit), `phase_of` 의 없는 · NaN · 모르는 class.
2. AttentionPolicy — probe 없으면 `hsr6` 키 없음 · `subtask` 그대로, 있으면 키가 생김, 3 단계 probe 를 꺼도 HSR6 feature 는 계산.
3. ① — lift · transport · placement 에서 한 요청 일찍 attach (strong 실패 = `settled` 만), approach · grasp · retreat · None 에서는 그대로, `blocked` 거짓 (빈손) 이면 보완 없음, kwarg 없음 = 이전 event (asdict 같음), SafePolicy 경로와 H4 표시.
4. ② — 규칙 8 경우, SafePolicy 가 AG3S 에 알리는 값, `_frame_held_attached`: 허용 None (꺼짐) = 들어올림 + 키 없음, True = 들어올림, False = 몸통 구 + `hsr6_lift_withdrawn`, 지지면에서 멀면 거둠 없음, reset 이 지움.
5. ③ — retreat 에서만, 쥔 동안은 아님, strip 이 모든 권한 필드를 비우고 strict 에서 어느 행도 target 접촉 불가, 권한 없으면 같은 객체.
6. ④ — grasp 에서만. A4 · NBC · G1 slot 각각 grasp 에서 유지 · 다른 단계에서 거둠, G1 거둠 뒤 추종 배율 · 연속성 끔 · controller 계수 원상. 기존 조건이 먼저 막으면 "달라짐" 아님.
7. serve_safe — A8 + A4 + G1 에 flag 없으면 meta 없음, 4 flag 켜면 meta, 할 일 없는 조합 5 가지 거절. summary · metadata 는 켰을 때만.

### H2 — 꺼짐 동등성 (핵심 평가 기록 재생)

- 도구: `$R/tools/h2_replay.py` = G1 `$G1/tools/v2_replay.py` 사본 (입력 · 방식 같음, 출력 이름만 다름). `tp_timing.run_timing` server mode, 기록 명령줄 그대로 (hsr6 · geo flag 없음), CPU 라서 GPU seed 만 끔, CPU 여러 초기값 16 은 켬.
- 입력: run `20261028T224856+0900_b3b69311f948_8784a1265958` REG_k0 ep211 첫 75 요청 + OCC case1_L2_s13453 31 요청 (G1 V2 와 같음). 저장소에는 쓰지 않았다.
- 두 트리: `pre_tree` (변경 6 파일 = G1 끝 판) 대 메인.

| episode | 요청 | refined bit 동일 (전 = 후) | 기록 같음 | 다른 기록 키 | A4 켜진 요청 | 여러 초기값 발동 |
|---|---|---|---|---|---|---|
| REG ep211 | 75 | **75** | 73 | seq 15 · 16 의 `multi_seed.*_ms` · `solve_ms` (시간) 만 | 27 | 2 |
| OCC 121 | 31 | **31** | 31 | — | 6 | 0 |

- G1 V2 `merged` npz 와도 106/106 bit 동일 (`$R/H2/h2_vs_g1_v2_merged.txt`).
- 한계 (G1 V2 와 같음): 이 재생은 TrajOpt 만 다시 푼다. SafePolicy 의 latch · AG3S 지각은 지나지 않는다. 그 경로는 H2b 와 H1 의 기존 test 2,000 여 개로 덮었다.
- 로그: `$R/H2/h2_pre.log`, `h2_main.log`, `h2_compare.json`.

### H2b — 꺼짐 동등성 (재생이 닿지 않는 경로)

- 도구: `$R/tools/h2b_off_records.py`. 두 트리에서 같은 합성 순서열을 돌려 JSON 을 비교했다.
  - SafePolicy grasp latch 경로: 실행 피드백 순서열 3 개, 요청마다 `_exec_summary()` 전체 (시각 `attached_at` 만 뺌) + metadata + latch 상태.
  - AttentionPolicy 결과 키와 `subtask`.
  - AG3S `_frame_held_attached` 의 구 중심 · held 기록 (지지면 근처 2 mm · 먼 곳 200 mm).
- 결과: 6/6 같음, chunk 기록 17 개 (`$R/H2b/h2b_compare.log`).

### H3 — 단계 parity

- 도구: `$R/tools/h3_parity.py`.
- 서버 쪽은 서빙 코드 경로를 그대로 쓴다. JAX prefix pass 만 기록 feature 를 내는 stub 으로 바꿨다: `AttentionPolicy(hsr6_probe=load_hsr6_probe())` → `result["hsr6"]` → `SafePolicy._hsr6_start` → `_hsr6_phase()`.
- feature: T55 test `feat/test/*.npz` `kv_l4`.
  - 서빙 축약 함수 `AttentionSampler._jit_kv_feature` 로 뽑은 것이다 (`t53_extract.py:14-15`).
  - T55 check 에서 T53 feature 와 최대 차 0 (`T55/feat/test/check.json` `vs_t53_feat`).
  - 서빙의 지금 HSR 도 같은 함수 · 같은 feature 다. 그래서 이 feature 가 서버가 만드는 feature 이고, 추가로 바꾼 전처리는 없다.

| 비교 | 범위 | 일치 |
|---|---|---|
| (a) partB.json 의 frame 별 `hsr` (세션 B 실행 결과) | natural 16 episode, t64_partB 와 같은 trace loader | **1,200 / 1,200** |
| (b) t64_partB 계산 (`LogReg.proba` argmax, storage `probe_6.npz`) | T55 test feature 100 파일 (instruction · natural · empty · slip) | **7,500 / 7,500**, 확률 max \|Δ\| 0.0 |

- 참고, (a) 의 서버 단계 분포: approach 454 · grasp 61 · lift 64 · transport 89 · placement 89 · retreat 443.
- 결과: `$R/H3/h3_parity.json`, 로그 `h3_parity.log`.

### 서버 argv CPU 검사

- `$R/tools/argv_check.py` 가 하는 일 (G1 V4 와 같은 방식, GPU 가용성 probe 는 통과로 둠):
  - parse → 조합 검사 → TrajOptConfig → HSR6 flag · 기록 meta
  - probe 적재 → AttentionPolicy 의 `hsr6` → SafePolicy metadata
- (가) · (나) 모두 ok. `.venv-ag3s` (`$R/argv/argv_check_ag3s.log`) 와 서버 venv `.venv-openpi-live` (`CUDA_VISIBLE_DEVICES=` · `JAX_PLATFORMS=cpu`, `$R/argv/argv_check_openpi.log`) 둘 다 통과.
- GPU 서버 기동은 lead 몫이다.

## 5. 서버 argv (한 줄씩)

- A8 flag 는 ADOPTED_KO.md §A8 그대로이고, G1 flag 는 G1 impl §5 를 따른다.
- `<…>` 는 lead 가 채운다: checkpoint, snapshot, port, run dir, `--record-constraints`.

(가) A8 + G1 (판정 (a), A4 대체):

```text
python -m AG3S.trajopt.serve_safe --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint <16D 30k ckpt> --model-xml <snapshot>/robots/rby1/description/models/rby1a/mujoco/model_transport.xml --esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 --attention-split-ratio-source raw --target-permission-strict --attention-source noun_target:L10H7 --attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1 --neighbour-contact-mm 20 --nontarget-evidence surface --placed-no-seed-certify --kept-geometry-follow --roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda --blas-threads 1 --to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass --palm-shell on --palm-shell-margin 5 --grasp-anchor 2000 --grasp-anchor-path-weight 0 --geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off
```

(나) (가) + HSR6 4 사용처:

```text
python -m AG3S.trajopt.serve_safe --config pi05_rby1_randomized_pick_place_16d_lora --checkpoint <16D 30k ckpt> --model-xml <snapshot>/robots/rby1/description/models/rby1a/mujoco/model_transport.xml --esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 --attention-split-ratio-source raw --target-permission-strict --attention-source noun_target:L10H7 --attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1 --neighbour-contact-mm 20 --nontarget-evidence surface --placed-no-seed-certify --kept-geometry-follow --roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda --blas-threads 1 --to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass --palm-shell on --palm-shell-margin 5 --grasp-anchor 2000 --grasp-anchor-path-weight 0 --geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off --hsr6-attach-assist --hsr6-held-gate --hsr6-target-exempt --hsr6-objective
```

- client flag 는 A8 과 같다 (HSR6 는 client 를 바꾸지 않는다).
- **closed-loop 에서 셀 기록** (`summary_json`, 요청마다):
  - `hsr6.phase` (raw argmax), `hsr6.p` (6 단계 확률)
  - `hsr6.changed.{attach_assist, held_gate, target_exempt, objective}`: 지금 규칙과 결정이 달라진 요청
  - `hsr6.detail`: 무엇이 달라졌나 · 허용 값
  - 함께 볼 곳: `grasp.evidence.attach_assist`, `grasp.attached.held.body_cover.frame.hsr6_lift_withdrawn`, `grasp.neighbour_contact.hsr6_withdrawn`, `grasp.anchor.hsr6_withdrawn`, `grasp.geo.anchor.hsr6_withdrawn`
  - 기록 meta 의 `hsr6` (flag · probe)

## verifier · lead 가 알아야 할 것

- **새 flag (기본 꺼짐):** `--hsr6-attach-assist` · `--hsr6-held-gate` · `--hsr6-target-exempt` · `--hsr6-objective` · `--hsr6-probe NPZ`.
  - 거절하는 조합:
    - `--no-safe` · `--no-attention` · `--no-perception` 과 함께
    - `--hsr6-held-gate` 를 `--held-body-cover` 없이
    - `--hsr6-objective` 에 거둘 것 (NBC · A4 · E1-B 상자 · G1) 이 없을 때
    - `--hsr6-probe` 만 줄 때
- **기록 키:** 꺼짐이면 새 키가 없다. 다음 키는 켰을 때만 생긴다: 응답 · `summary_json` 의 `hsr6`, metadata `hsr6`, 기록 meta `hsr6`, latch `evidence.attach_assist`, body cover frame 의 `held_allowance` · `hsr6_lift_withdrawn`, gate 기록의 `hsr6_withdrawn`.
- **재생산할 산출물:** 없음. 옛 기록과의 호환도 깨지지 않는다.
- **위험 — 결과 전에 미리 적어 둔다. 규칙은 바꾸지 않았다:**
  1. **② 와 쥔 직후 grasp 단계.** 쥔 직후 사과가 아직 탁자 위인 동안 HSR6 의 정답은 grasp 다 (T64 정의: 2 cm 오를 때까지 grasp).
     - 그 frame 들에서 ② 는 지지면 들어올림을 거둔다.
     - J1 기록에서 attach 순간의 fit 구는 탁자 아래 1.8–10.1 mm 였다. 거리장 band 도 약 8.8 mm 다.
     - 그래서 쥔 몸통 구와 탁자 사이 위반 → HOLD 가 생길 수 있다. HOLD 면 손이 들리지 않는다. 그러면 HSR6 가 lift 로 넘어가지 못한 채 멈출 수 있다.
     - `--hold-follow-to` 는 실제 침투 행을 받지 않으므로 이 경우를 풀지 못한다.
     - closed-loop 에서 `hsr6.changed.held_gate` 와 HOLD 사유 (`collision`, 쥔 행 `attached_point`) 를 함께 보기를 권한다.
  2. **① 은 손가락이 움직이는 중에도 걸린다.** `closed ∧ blocked` 는 닫히는 도중의 frame (예: 개도 0.80, 명령 0.0) 에서도 참이다. T64 Part C 식에는 `settled` 가 없다.
     - 그래서 보완 attach 가 strong 보다 먼저 걸리고, 다음 요청에서 T34 회수 (`opening_dropped`) 가 풀 수 있다 (attach → 회수 → strong attach).
     - 이 동작을 test 로 고정해 두었다 (`test_assist_while_the_fingers_still_close_meets_the_t34_revoke`).
  3. **① 서빙에는 `AG3S.attach` 의 마지막 관문이 남는다.** 붙일 기하가 있어야 하고 (usable), reach ≤ 120 mm 여야 한다.
     - 그래서 strong 이 `manipulated` · `reach` 때문에 실패한 frame 은 보완해도 거절된다.
     - T64 Part C 의 offline 회복 (107 / 117) 은 이 관문이 없는 frame 단위 계산이다. 서빙의 회복 수는 그보다 작거나 같다.
  4. **④ 는 A4 를 거의 닫기 순간으로 줄인다.** A4 는 지금 latched / closing 동안 손바닥이 목표 150 mm 안이면 켜진다. HSR6 는 닫기 전까지 대부분 approach 다 (T64 정의: phase 0–2).
     - 그래서 ④ 를 켜면 A4 · G1 · NBC 는 HSR6 = grasp (phase 3–4, 손가락 닫기 · 확인) 인 요청에서만 남는다.
     - G1 보정은 VLA 가 닫기를 계획한 latched chunk 에서만 생긴다. 그 chunk 의 HSR6 가 아직 approach 면 G1 은 거의 발동하지 않을 수 있다.
  5. **③ 은 놓은 뒤에도 target 권한이 남아 있을 때만 무엇인가를 바꾼다.** subtask gate (home) · placed release 가 이미 target 을 놓았다면 거둘 것이 없다 (`changed = False`).
- **worktree 는 쓰지 않았다.** 모든 test 는 메인 checkout 에서 돌렸다. commit 은 하지 않았다.

## 내가 기대하는 결과

(verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.)

- ③ 의 발동은 적거나 0 일 것이다. 놓은 뒤 home label 로 target 이 이미 풀리는 경로 (A12 / B3) 가 먼저 작동하기 때문이다.
- ④ 의 발동 (`a4_withdrawn` · `nbc_withdrawn`) 은 쥐는 장면마다 여러 요청 있을 것이다.
- ② 의 `lift_withdrawn` 은 attach 직후 몇 요청에서 생길 것이다 (위 위험 1).
- ① 의 보완 attach 는 드물 것이다 (strong 이 대부분 먼저 잡는다).
