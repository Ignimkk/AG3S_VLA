# SL6-H — 구현 (placement 단계 쥔 물체 접촉 허용, `--hsr6-placement-contact`)

> writer: ag3s-implementer (SL6-H) · 읽는 쪽: lead, verifier, scribe · 작성 2026-10-30 (KST)
> task: [SL6.task.md](SL6.task.md) §0 마지막 행 · §11 (H1–H4) · 진단 `outputs/2026-10-30_작업성공률저하_원인진단_Panda/README.md` §2
> 바탕: checkout `feature/panda-benchmark` HEAD `d53745f5` (작업 트리, commit 없음). CPU 만, GPU · closed-loop 실행 없음.

## 무엇을 했나 (평이한 요약 먼저)

- 새 flag `--hsr6-placement-contact` (기본 꺼짐) 를 HSR6 의 다섯째 사용처 (⑤) 로 넣었다.
  - 켜지는 call: HSR6 argmax = `placement` ∧ (latch HELD ∨ (latch CLOSING ∧ evidence `closed` ∧ `blocked`)).
  - 그 call 에서 **쥔 물체 (attach 된 held object) 의 질의 구 행만**, 최근접 표면이 그 구 **아래** 에 있으면 막지 않는다.
  - 손 · 손가락 · 팔 행은 전혀 바뀌지 않는다. 쥔 물체 행이라도 옆 · 위의 표면은 그대로 막는다.
- 바뀌는 곳은 TrajOpt 의 행 여유 (`CollisionLinearizer`) 하나다. 궤적 쪽 원인 (TrajOpt 가 하강을 위로 뒤집음) 이므로 TrajOpt 층에서 고쳤다. AG3S 기하 (held-body cover · support lift) 는 손대지 않았다.
- call 기록: Panda 서버 응답에 `panda/hsr6` (HSR6 phase · 확률 · 규칙별 변경 · placement 허용 기록) 를 싣고, client 가 `*_calls.jsonl` 행에 `hsr6` 로 남긴다. HSR6 가 꺼진 서버 (V11AB · pb) 의 행은 키가 그대로다.
- profile `AG3S-final-panda-v3-e0to-pc` = `AG3S-final-panda-v3-e0to` + `--hsr6-placement-contact`.
- Panda 서버는 이 flag 를 이식된 option 으로 받는다. 다른 HSR6 flag 처럼 `--hsr6-probe` 없이는 거절한다.
- 결과: 새 test 45 개 통과, `tests/benchmark` 전체 **284 passed · 3 skipped**, `tests/trajopt` 전체 **2,267 passed · 38 skipped · 1 failed** (실패 1 개는 HEAD 에서도 같은 실패, §4.2). 기존 판의 argv 는 HEAD 와 같고, 2판 · 3판 · E0판의 Panda CPU 기동 (SafePolicy 생성 인자 · `info`) 은 HEAD 서버 module 과 같다.

## 1. 규칙 (구현한 그대로)

### 1.1 켜지는 call (`hsr6.placement_contact_ok`, `SafePolicy._hsr6_placement_contact`)

| HSR6 argmax | latch (이번 call 최종) | evidence | ⑤ `on` |
|---|---|---|---|
| placement | held | 무관 | 켬 |
| placement | closing | `closed` ∧ `blocked` | 켬 |
| placement | closing | 둘 중 하나라도 아님 | 끔 |
| placement | searching · latched · placed | 무관 | 끔 |
| 그 밖 (approach · grasp · lift · transport · retreat · label 없음) | 무관 | 무관 | 끔 |

- latch 는 `_run_latch` 뒤 (snapshot 을 만들 때) 의 최종 상태다. evidence 는 pb7 (`closing_exempt_now`) 과 같은 곳, `GraspLatch.last_grasp.evidence.checks` 에서 읽는다. pb7 과 달리 `not_empty` 는 보지 않는다 (task 문장: `closed` ∧ `blocked`).
- `on` 이어도 snapshot 에 쥔 물체 질의점 (`attached_points`) 이 없으면 바꿀 행이 없다. 이때는 아무것도 바꾸지 않고 `applied = false`, `reason = "no attached held object: no held rows to release"` 로 기록한다 (진단의 CLOSING 미인식 episode 가 이 경우다).
- `applied = on ∧ attached ∧ 거리장 있음` 일 때만 `SceneSnapshot.placement_contact` 가 생긴다. 끄면 (flag 없음) `_hsr6_placement_contact` 는 불리지 않는다.

### 1.2 풀리는 행 — "아래 표면" 판정 (`CollisionLinearizer._placement_contact_margin`)

질의 행 마다 (계획 지평의 모든 step, 실행 창 · look-ahead 꼬리 모두):

| 행 | 조건 | 여유 (margin) |
|---|---|---|
| 로봇 행 (`< n_spheres`: 손 몸통 · 손바닥 껍질 · 손가락 · 팔 link1–7) | — | **그대로** |
| 쥔 물체 행 (`≥ n_spheres`) | (a) 그 구 중심 `p` 의 거리장 단위 기울기 `ĝ = ∇d/|∇d|` 가 `ĝ_z ≥ cos 45° (0.7071)` **그리고** (b) 최근접 표면점 `s = p − d·ĝ` 가 같은 step 의 쥔 물체 xy 발자국 안 (`min_j (‖s_xy − c_j,xy‖ − r_j) ≤ tol`, 발자국 = 쥔 물체 구들의 xy 원판 합집합, `tol` = 거리장 voxel 1 개 = 5 mm) | **−1 m** (그 행은 묶이지 않는다 — pb4 held exemption 과 같은 값) |
| 쥔 물체 행 | 그 밖 (옆 · 위 표면, 발자국 밖, `|∇d| ≈ 0` 인 평평한 장) | 그대로 |

- (a) 의 뜻: 최근접 표면이 아래 45° 원뿔 안에 있다. 중심이 이미 물체 안 (`d < 0`) 이면 기울기는 빠져나가는 방향이므로 "위로 빠져나간다 = 아래 물체 안에 박혔다" 로 같은 판정이 된다.
- (b) 는 (a) 만으로 부족한 경우 (구에서 멀리 떨어진 표면, 비스듬한 모서리) 를 막는다. 구 하나만 보면 `d < r·√2` 인 아래 표면은 자동으로 발자국 안이다.
- 판정은 기하만 쓴다. Panda 거리장에는 목적지 label 이 없다 (진단 §2). 그래서 탁자 · plate · 그 밖의 아래 물체를 구분하지 않는다 (§5 한계).
- 기존 구조와의 관계:
  - held-body cover · support lift (`pipeline._frame_held_attached`) 는 쥔 몸통 구를 **탁자 평면** + band 위로 올린다. plate 는 지지 평면이 아니어서 그 lift 가 닿지 않는다. 이번 규칙은 그 위의 TrajOpt 행 여유만 바꾼다. AG3S 쪽 lift · HSR6 ② 는 그대로다.
  - support band (`_held_rows_class` 의 `|d − s| ≤ 2 voxel`) 는 평면만 안다. 그래서 plate 를 "아래" 로 잡으려고 band 대신 기울기 · 발자국을 썼다.
  - 같은 기울기 · 최근접 표면점 계산은 `_held_rows_class` (held remnant 판정) 와 같다 (`scene.esdf.gradient`).

### 1.3 순서 · 다른 규칙과의 관계

- `_esdf_clearance_core` 안에서 `grasp_contact` (pb4 · pb5 · pb7) 다음, IFG-NBC · A6 앞에서 적용한다. 앞 규칙들은 로봇 행만, 이 규칙은 쥔 물체 행만 바꾸므로 겹치지 않는다.
- 판정 (`_verdict` 의 `max_violation`) 도 같은 snapshot 으로 다시 재므로, 풀린 행은 위반으로 세지 않는다 (SQP `full_violation` / `worst_row` 가 같은 `_esdf_clearance` 를 쓴다).
- `to-fast-eval` memo 가 부작용 속성을 되살릴 수 있게 `_last_placement_contact` 를 `_CLEARANCE_SIDE_ATTRS` 에 넣었다.

## 2. call 기록이 어떻게 생기나 (H2)

`*_calls.jsonl` 행에 `hsr6` 키 (서버 응답 `panda/hsr6`, HSR6 use flag 가 하나라도 켜진 서버에서만):

```json
"hsr6": {
  "phase": "placement",                      // raw argmax (None = label 없음)
  "p": {"approach": 0.01, "grasp": 0.02, "lift": 0.03, "transport": 0.14, "placement": 0.80, "retreat": 0.0},
  "probe": "hsr6_panda_kv_L4",
  "flags": ["attach_assist", "held_gate", "target_exempt", "objective", "placement_contact"],
  "objective_scope": "nbc",
  "changed": {"attach_assist": false, "held_gate": false, "target_exempt": false, "objective": false,
              "placement_contact": true},      // ⑤: 이번 call 반환 궤적의 최종 검사에서 풀린 행이 1 개 이상
  "detail": {
    "placement_contact": {
      "on": true, "phase": "placement", "latch": "held", "closed": true, "blocked": true,
      "attached": true, "applied": true, "n_held_rows": 4,
      "rows_released": 12,                   // (step, 쥔 물체 행) 쌍 수, 반환 궤적 최종 검사
      "held_rows_released": 2,               // 한 step 이라도 풀린 쥔 물체 행 수
      "min_clearance_released_m": -0.010,    // 풀리기 전 그 행들의 최소 여유 (음수 = 아래 표면에 박힘)
      "final_check": {"n_held_rows": 4, "steps": 10, "rows_released": 12, "held_rows_released": 2,
                      "min_clearance_released_m": -0.010, "cos_min": 0.707107, "footprint_tol_m": 0.005}
    },
    "held_gate": {"allowed": true, "latch_held": true, "blocked": true}
  }
}
```

- `on = true · attached = false` 는 "규칙은 켜졌는데 바꿀 행이 없었다" 다 (`reason` 이 붙는다). `applied = true · rows_released = 0` 은 "켜졌지만 아래 표면이 없었다" 다.
- ⑤ 가 꺼진 HSR6 서버 (2판 · 3판 · E0판) 에도 `hsr6` 키가 생긴다 (phase · p · 기존 네 규칙의 changed / detail). `detail.placement_contact` 와 `changed.placement_contact` 는 없다.
- 같은 내용이 TrajOpt metrics `to.metrics.placement_contact` (`sqp._finish`, 반환 궤적의 최종 검사) 와 constraint record 의 `summary_json.hsr6` 에도 들어간다 (`--record-constraints` 를 켠 run 만).
- 서버 응답에 키를 더할 뿐 actions 는 같다. client 는 응답에 `panda/hsr6` 가 있을 때만 행에 `hsr6` 를 쓴다 (py3.8 test: actions 동일, 행 키 차이 = `hsr6` 하나).

## 3. 바뀐 파일 · md5 (작업 트리, commit 하지 않음)

| 파일 | md5 (작업 트리) | HEAD `d53745f5` md5 |
|---|---|---|
| `AG3S/trajopt/hsr6.py` | `87d74101f61bf424d66ecde25cc743a4` | `fd27eb17817768979456c9c07a09d49f` |
| `AG3S/trajopt/linearize.py` | `6cac4cf39c1f1555436dcd38aea4037c` | `2de47067560887ed72b350aa976469dd` |
| `AG3S/trajopt/sqp.py` | `644b7203ded53a7856037606a88c6c19` | `4947b903323cfdf00a910b4d991f1913` |
| `AG3S/trajopt/safe_policy.py` | `6f95ba73741386894ed592d42cfcebc7` | `f0f95471eb7acf3bbb0b2c26099d6e65` |
| `AG3S/trajopt/serve_safe.py` | `fb414675ea0e77cc681b4112ef581541` | `4b124394304db32a05ef1c9c99fe6a31` |
| `AG3S/adapters/panda/server.py` | `d763ffb4b8d1312f63938897046ada7f` | `3c174f6984fc7a59d837306841ae6a49` |
| `AG3S/adapters/panda/profile.py` | `7dae0873475d3d9978747ccee0572ea3` | `abd6ef3c03efc37b3754ae68baca38e8` |
| `benchmark/runners/libero_safe_client.py` | `5c59c5c1517eafe300f1aa4df7291679` | `bd79c31f5ca81bab5dfaaef4dbfc736d` |
| `tests/trajopt/_sl5a_golden.py` | `4d42c7002d6df06835d72088c62264a3` | `8709f6a91b6b5226739d09649413e47d` |
| `tests/trajopt/test_sl6h_placement_contact.py` (새) | `f8fcfa68f4da1b91f943cc88f38288b2` | — |
| `tests/benchmark/test_sl6h_panda_placement_contact.py` (새) | `fe4055b7d3856df8db62145e68ee09a3` | — |

| 파일:줄 | 무엇이 |
|---|---|
| `AG3S/trajopt/hsr6.py:18` · `:56-61` · `:77-83` · `:146-152` · `:203-230` | ⑤ 표 행, `PLACEMENT` · `PLACEMENT_CONTACT_CHECKS` · `PLACEMENT_BENEATH_COS`, `Hsr6Flags.placement_contact` (마지막 field — 기존 네 이름 · 순서 그대로), `placement_contact_ok`, `placement_contact_spec` |
| `AG3S/trajopt/linearize.py:121` · `:311-319` · `:1061-1064` · `:1145-1197` | memo 부작용 속성, `SceneSnapshot.placement_contact` (기본 None), `_esdf_clearance_core` 호출, `_placement_contact_margin` |
| `AG3S/trajopt/sqp.py:1508-1512` · `:1615` | `to.metrics.placement_contact` (scene 에 키가 있을 때만) |
| `AG3S/trajopt/safe_policy.py:185-186` · `:471` · `:617` · `:732-733` · `:1229-1230` · `:1310` · `:1392-1442` | 기록 기본값 · reset, refine 뒤 결과 읽기, `_scene_fn` 에서 ⑤ (flag 켜졌을 때만), 요청마다 기록 비우기, `_hsr6_placement_contact` · `_hsr6_placement_contact_result` |
| `AG3S/trajopt/serve_safe.py:1977-1981` · `:2767-2774` | `--hsr6-placement-contact`, `hsr6_flags_from_args` |
| `AG3S/adapters/panda/server.py:295-311` · `:344-350` | `panda/hsr6` 응답 키 (`hsr6_call_record`), `PORTED_FINAL_CONFIG_DESTS` 에 `hsr6_placement_contact` |
| `AG3S/adapters/panda/profile.py:316-331` | `FINAL_CONFIG_V3_E0TO_PC_FLAGS`, `FINAL_CONFIG_VERSIONS` 끝에 pc판, `AG3S_VERSIONS["AG3S-final-panda-v3-e0to-pc"]` |
| `benchmark/runners/libero_safe_client.py:19-20` · `:339-342` | 행에 `hsr6` (응답에 `panda/hsr6` 가 있을 때만) |
| `tests/trajopt/_sl5a_golden.py:406-408` | RB-Y1 golden 비교의 "꺼진 새 option" 목록에 `hsr6_placement_contact: False` (SL6-A A9 와 같은 방식 — 켜지면 차이로 잡힌다) |
| `tests/trajopt/test_sl6h_placement_contact.py` (새) | §4 |
| `tests/benchmark/test_sl6h_panda_placement_contact.py` (새) | §4 |

`benchmark/runners/run_parallel_gpu.sh` 는 손대지 않았다.

## 4. 단위 검증

```bash
source scripts/setup/env.sh
MUJOCO_GL=osmesa nice -n 10 .venv-ag3s/bin/python -m pytest -q tests/trajopt/test_sl6h_placement_contact.py tests/benchmark/test_sl6h_panda_placement_contact.py
MUJOCO_GL=osmesa nice -n 10 .venv-ag3s/bin/python -m pytest -q tests/benchmark
MUJOCO_GL=osmesa nice -n 10 .venv-ag3s/bin/python -m pytest -q tests/trajopt
```

- `PYTHONPATH` 는 env.sh 가 놓은 값 그대로 썼다. `PYTHONPATH=${AG3S_PROJECT_ROOT}` 로 덮어쓰면 `openpi_client` 가 빠져 `test_panda_sl3` · `test_sl6_b5_noun_rows` 등 9 개가 실패 / error 로, 18 개가 더 skip 된다 (이번 변경과 무관, 처음 한 번 그렇게 돌렸다).

| 묶음 | 결과 |
|---|---|
| 새 test (trajopt 35 + benchmark 10) | **45 passed** |
| `tests/benchmark` 전체 | **284 passed · 3 skipped** (HEAD 274 passed · 3 skipped + 새 10) |
| `tests/trajopt` 전체 | **2,267 passed · 38 skipped · 1 failed** (실패 1 개는 HEAD 에서도 같은 실패, §4.2) |

H4 항목별:

| H4 | test | 확인한 것 |
|---|---|---|
| 합성 장면: plate 로 내려가는 쥔 구의 위반이 on + placement + HELD 에서 사라짐 | `test_on_releases_only_held_rows_on_the_surface_beneath`, `test_sqp_plate_beneath_moves_the_plan_only_without_the_rule` | 행 수준: plate 에 10 mm 박힌 쥔 구 · 탁자에 5 mm 박힌 쥔 구만 여유 +1 m, 나머지 6 행 bit 동일. SQP (RB-Y1 fixture 팔, `TrajOptChunkRefiner`): 끄면 reference 위반 10 mm · 계획이 움직임 (> 1e-3 rad), 켜면 계획 = plate 가 없는 장면의 계획 (bit 동일), `to.metrics.placement_contact.rows_released > 0` |
| flag off · phase ≠ placement · latch 미충족에서는 그대로 | `test_off_is_bit_identical`, `test_placement_contact_rule` (14 경우), `test_safe_policy_gate` (7 경우: transport+HELD · placement+LATCHED · CLOSING 에 blocked 없음 → snapshot 그대로), `test_record_resets_per_request_and_is_absent_when_off` | `placement_contact=None` 과 키 없음이 bit 동일 |
| 같은 쥔 구 옆 장애물은 on 에서도 제약 | `test_held_sphere_beside_a_wall_is_constrained_while_lowered_onto_the_plate` (높이 5 개), `test_sqp_wall_beside_the_held_sphere_still_moves_the_plan_with_the_rule`, 위 장애물 (`held_under_shelf`), 발자국 밖 아래 모서리 (`held_step_off_footprint`) | SQP: 켜도 reference 위반 10 mm, 계획이 움직임, 풀린 행 0 |
| 손 · 손가락 행은 on 에서도 불변 | `test_hand_rows_never_change_even_on_the_plate`, 위 행 수준 test | plate 에 2 mm 박힌 손가락 행 그대로 위반 |
| flag off 회귀: 기존 판 argv · CPU 기동 불변 | `test_every_head_version_keeps_its_argv` (HEAD `d53745f5` 의 profile 을 git 에서 읽어 모든 판의 flags · base argv · 서버 flags 비교), `test_existing_final_builds_match_the_head_server` (2판 · 3판 · E0판: SafePolicy 생성 인자 · `info` 가 HEAD 서버 module 과 같음), 기존 `test_v11ab_family_build_is_the_pre_sl6_build` (V11AB · pb7), `test_rby1_golden_identity` | |
| Panda 서버: probe 와 함께 받고, 없으면 거절, CPU 기동에서 켜짐 | `test_server_accepts_the_flag_with_the_probe_only`, `test_pc_builds_on_cpu` | `safe.hsr6.placement_contact = True`, `final_config.hsr6.flags` 끝 = `placement_contact`, `options_on` 에 포함; E0판은 False |
| H2 기록 | `test_response_carries_the_hsr6_record_only_with_hsr6_on`, `test_client_row_records_the_hsr6_record` (`.venv-libero` py3.8) | V11AB-pb7 서버는 `panda/hsr6` 없음 |

### 4.1 HEAD 대비 CPU 기동 dump (한 번, pytest 밖)

HEAD `d53745f5` 를 `git archive` 로 scratch 에 풀고 (third_party/curobo 의 `content/configs` 만 복사, 비교 뒤 지움), 같은 script 로 두 tree 에서 Panda 서버를 CPU 로 기동해 dump 했다.
- 비교 항목: argv, 파싱된 args, `info` (= `panda_server_config.json` 내용), SafePolicy 생성 인자, `to_config` 전체, `version_flag_values`, `match_ag3s_version`.
- 판: V11AB · V11AB-pb7 · 1판 · 2판 · 3판 · E0판.

| 판 | 차이 수 | 차이 (전부 새 option 의 꺼진 기본값) |
|---|---|---|
| V11AB · V11AB-pb7 | 1 | `args.hsr6_placement_contact = False` (namespace 에 새 dest) |
| 1판 · 2판 · 3판 · E0판 | 3 | 위 + `Hsr6Flags.placement_contact = False` (생성 인자 `hsr6` 의 새 field) + `version_flag_values.hsr6_placement_contact = false` |

- argv · `info` · `to_config` · 그 밖의 생성 인자는 6 판 모두 같다.
- script: `/tmp/claude-0/-mnt-dev-AG3S-VLA-benchmark/f2c505ed-ce02-4de8-97c6-a12570d6f876/scratchpad/sl6h/dump_builds.py` · `diff_builds.py`, dump `build_head.json` · `build_work.json` (같은 폴더, scratch).

### 4.2 `tests/trajopt` 의 실패 1 개는 HEAD 에서도 실패한다

- `tests/trajopt/test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit` (`to_T42_L1992_16` 배열 불일치).
- 같은 test 를 HEAD tree (위 4.1 의 scratch 풀이) 에서 돌려도 같은 key 로 실패한다 → 이번 변경과 무관한 기존 실패다.
- 이번 변경은 `placement_contact = None` 일 때 수치 경로가 없다 (`_esdf_clearance_core` 의 `if pc is not None`, `sqp._finish` 의 키 조건).

## 5. launcher dry-run — `AG3S-final-panda-v3-e0to-pc` 서버 argv

```bash
source scripts/setup/env.sh
.venv-ag3s/bin/python -m benchmark.runners.launch_libero_paired --task-suite-name safelibero_spatial --safety-level I \
  --data-split dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14 --task-ids 1 --episode-ids 1,4 --run-id sl6h-dryrun \
  --conversion C --modes vla_v11ab --ag3s-version AG3S-final-panda-v3-e0to-pc --dry-run
```

서버 argv (`vla_v11ab`, port 0, 한 줄로):

```
--config pi05_libero --checkpoint /mnt/dev/AG3S_VLA-storage/checkpoints/panda/pi05_libero --model-xml /mnt/dev/AG3S_VLA-benchmark/robots/rby1/description/models/rby1a/mujoco/model_transport.xml --port 0 --esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --robot panda --to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 --attention-split-ratio-source raw --target-permission-strict --attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1 --neighbour-contact-mm 20 --nontarget-evidence surface --placed-no-seed-certify --kept-geometry-follow --roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda --blas-threads 1 --palm-shell on --palm-shell-margin 5 --hsr6-attach-assist --hsr6-held-gate --hsr6-target-exempt --hsr6-objective --neighbour-palm-contact-mm 10 --hsr6-objective-scope nbc --reach-at-capture --links arms --held-hand-exempt --latch-phase --grasp-zone-mm 10 --grasp-reach-sticky --grasp-drop-from-open --closing-hand-exempt --continuity-after-grasp --deflection-rate-after-grasp --hsr6-placement-contact --hsr6-probe /mnt/dev/AG3S_VLA-storage/results/benchmark/20261029T192223_5ea56af44722_sl6b-hsr6-panda-labels/probe/hsr6_panda_kv_L4.npz
```

- E0판 smoke run 의 기록된 서버 argv (`20261030T101944_d53745f50539_sl6f-smoke-AG3S-final-panda-v3-e0to-t1-LI-vla_v11ab-AG3S-final-panda-v3-e0to-C/run_manifest.json` `settings.server_argv`, port 만 0 으로) 와 비교했다. 차이는 `--hsr6-probe` 바로 앞의 `--hsr6-placement-contact` 한 token 뿐이다.
- client argv 는 E0판과 같다 (`libero_safe_client.py --task-suite-name safelibero_spatial … --safety-level I`).

## 6. 한계 · 경계 경우

1. **아래 물체를 가리지 않는다.** Panda 거리장에는 목적지 label 이 없다. 그래서 "쥔 물체 발자국 아래" 에 있는 것은 plate 가 아니어도 풀린다.
   - 예: HSR6 가 일찍 `placement` 를 내는 동안 쥔 bowl 이 SafeLIBERO 장애물 · 다른 bowl · cabinet 위를 지나면, 쥔 물체 행은 그 윗면에 막히지 않는다. 손 · 손가락 · 팔 행은 그대로 막힌다.
   - SafeLIBERO 충돌은 장애물 변위로 센다. 그래서 이 경우는 충돌 수로 드러날 수 있다. smoke 에서 "placement 허용이 켜진 call 의 쥔 bowl 위치가 plate 위였는가" 를 함께 보면 가를 수 있다.
2. **최근접 표면 하나만 본다.** ESDF 는 행마다 가장 가까운 표면 하나의 거리만 준다.
   - 아래 표면이 가장 가까운 동안에는, 같은 구 옆의 더 먼 장애물이 그 행에 보이지 않는다. 이 행은 −1 m 로 풀렸기 때문이다.
   - 그 장애물이 더 가까워져 최근접이 되면 기울기가 옆을 향해 다시 막힌다. 구가 아래 표면에 깊이 박힐수록 (d 작음) 옆 장애물이 보이기 전에 겹치는 깊이가 커진다 (최대 반지름 − d 정도).
   - 쥔 물체의 다른 구 (옆쪽 구) 는 따로 그 장애물을 본다.
3. **비스듬한 아래 모서리.** 발자국 안이고 45° 원뿔 안이면 장애물 모서리도 "아래" 다. 예: bowl 을 cabinet 가장자리 위로 내리면 그 모서리에 막히지 않는다.
4. **CLOSING 미인식 call 에는 효과가 없다.** 진단의 t1e4 · t2e0 · t2e4 (CLOSING 29–38 call) 는 attach 가 없어 쥔 물체 행이 없다. 이 call 들은 `on = true · attached = false` 로 기록만 된다. 그때 bowl 은 AG3S 에서 target / 장애물로 남아 있고, 그것을 바꾸는 것은 이 task 의 범위 밖이다 (latch · attach 층).
5. **HSR6 raw argmax 그대로.** debounce 가 없다. 한 call 만 `placement` 여도 그 call 은 풀린다. Panda probe 는 retreat 가 없는 5 class 다.
6. **"풀린 행" 기록은 반환 궤적의 최종 검사 한 번이다.** SQP 반복 중 다른 iterate 에서 풀린 수와는 다를 수 있다. SQP 를 거치지 않은 call (solver 실패 · 장면 없음) 은 `final_check = null · rows_released = 0` 이다. E0판에는 minimal intervention · multi-seed 가 없어 그 경로는 없다.
7. **RB-Y1 서버도 같은 flag 를 받는다** (`serve_safe` parser 는 공유). 기본 꺼짐이고 RB-Y1 argv · golden 은 바뀌지 않았다. RB-Y1 에서 켜는 것은 시험하지 않았다 (RB-Y1 에는 목적지 label 이 있으므로, 쓴다면 목적지 label 로 좁히는 것이 맞을 수 있다).
8. **`version_flag_values` 기록 키.** 최종 구성 판 (2판 · 3판 · E0판 등) 서버의 constraint-record meta `version_flags` 에 `hsr6_placement_contact: false` 가 새로 생긴다 (이 함수는 최종 판들의 dest 합집합을 쓴다). SL6-E · F 가 dest 를 더할 때와 같은 동작이다. V11AB 계열의 키는 그대로다.
9. **HSR6 서버 응답에 `panda/hsr6` 가 생긴다.** ⑤ 를 켜지 않은 2판 · 3판 · E0판의 call 기록에도 `hsr6` 키가 생긴다 (H2 의 "HSR6 phase 를 call 기록에" 를 모든 HSR6 서버에 적용). actions · 서버 기동 인자는 같다.

## 7. verifier 가 알아야 할 것

- 새로 생긴 플래그: `--hsr6-placement-contact` (기본 off). profile `AG3S-final-panda-v3-e0to-pc`.
- 재생산이 필요한 산출물: 없음.
- 옛 기록과 호환: `*_calls.jsonl` 에 `hsr6` 키가 더해질 뿐이다 (HSR6 서버만). 기존 키는 그대로다.
- smoke 에서 셀 것 (task §11 "함께 보고"): `hsr6.detail.placement_contact` 의 `applied` call 수 · `rows_released`, `on ∧ ¬attached` call 수 (CLOSING 미인식), `changed.placement_contact`.

## 8. 내가 기대하는 결과 (verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다)

이 절은 측정 전의 예상이다. 판정에 쓰지 않는다.

- 진단의 놓기 실패 5 개 중 latch 가 HELD 인 동안 막힌 episode (t1e1 · t2e2) 에서는, HSR6 가 placement 를 내는 call 에서 쥔 bowl 행이 plate · 탁자에 막히지 않는다. 그래서 하강 명령이 위로 뒤집히는 row 가 줄 것으로 본다.
  - 단, 손가락 · 손 행이 plate 나 탁자에 먼저 닿으면 그 행이 하강을 막는다. 이번 규칙은 그 행을 풀지 않는다. 그런 call 은 `rows_released > 0` 인데도 실행 z 가 VLA 보다 위인 call 로 보일 것이다.
- CLOSING 미인식 episode (t1e4 · t2e0 · t2e4) 에는 효과가 거의 없을 것으로 본다. attach 가 없어 풀 행이 없다 (`on ∧ ¬attached` 로 기록된다).
- HSR6 가 placement 를 늦게 (plate 바로 위에서만) 내면 효과가 작고, 일찍 (운반 중) 내면 장애물 위를 지날 때 쥔 bowl 이 덜 막힌다. 그래서 충돌이 늘 위험이 있다 (§6 한계 1).
- task 3 (target 오류 4 개) 에는 영향이 없을 것으로 본다.
