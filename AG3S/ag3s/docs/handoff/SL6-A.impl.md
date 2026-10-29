# SL6-A — 구현 (AG3S 최종 구성 Panda 이식: 서버 · 기하)

> writer: ag3s-implementer (SL6-A) · 읽는 쪽: lead, verifier, scribe · 작성 2026-10-29 (KST, 아래 시각은 각 절에)
> task: [SL6.task.md](SL6.task.md) §0 · §1 · §2 · §2.1 (§3 은 SL6-B) · 바탕: HEAD `553ad13e` (코드) = `5ea56af4` (task 문서 commit 만 더함)
> 원시 기록 · 도구: `$AG3S_STORAGE_ROOT/results/benchmark/20261029T193450_5ea56af44722_sl6a-verify/verify/SL6-A/` (아래 `$R`)

## 무엇을 했나 (평이한 요약 먼저)

- RB-Y1 최종 서버 argv (A6 impl §3) 를 `serve_safe --robot panda` 가 그대로 받아 돌 수 있게 했다. profile key 둘:
  `AG3S-final-panda` (G1 · A4 제거판), `AG3S-final-panda-g1a4` (작동판). 둘 다 `--hsr6-objective-scope nbc` 를 갖고,
  pb1–pb7 flag 와 `V11AB_PRE_MERGE_DEFAULTS` 는 없다. Panda HSR6 probe (`--hsr6-probe`) 는 run 마다 주어야 하고 없으면 거절한다.
- RB-Y1 이름이 박혀 있던 곳 (손목 관절, 손가락 이웃 접촉 손가락, A6 기준점, multi-seed 손바닥, GPU seed 의 URDF · base · tool,
  G1 의 gripper spec · IK 관절) 을 **주입 지점** 으로 바꿨다. 주입하지 않으면 지금 RB-Y1 값이다.
- Panda 손바닥 껍질 구 (A5) 는 RB-Y1 규칙 (`gripper_cover.palm_shell_spheres`) 을 그대로 Panda 손 collision mesh 에 적용해
  **111 개** 를 만들었고, 파일로 commit 할 수 있게 `AG3S/adapters/panda/panda_palm_shell.json` 에 고정했다.
- G1 은 Panda 손의 frame 차이 (y = 닫는 방향, +z = 접근) 를 spec 의 `frame` 회전으로 맞춘다. G1 이 닫기를 미룬 chunk 에서만
  역변환이 refined gripper 열을 보낸다.
- 새 flag `--hsr6-objective-scope {all,nbc}` (기본 all = 지금 규칙).
- 결과: VA1 단위 test **2,505 통과 · 실패 0** (lead 실행), VA2 RB-Y1 재풀이 **103/103 bit 동일**, VA3 Panda V11AB · pb7 config dump **차이 = 새 option 기본값 칸뿐**, VA4 최종 구성 두 판 CPU 기동 통과.

## 0. 규칙을 코드로 옮기며 정한 것 (lead 확인 필요 표시)

| 항목 | task 문장 | 코드 | 이유 |
|---|---|---|---|
| **G1 최대 개도 (확인 필요)** | "최대 개도 0.08" | `PANDA_LEFT.max_opening = 0.0790` | RB-Y1 spec 의 정의가 "두 pad 면 사이 최대 거리" 다. Panda pad box 가 손가락 mesh 보다 0.5 mm 안쪽으로 나와 있어 열림 끝 (관절 0.04 m × 2) 에서 pad 면 거리 = 79.0 mm (export 측정). 0.08 은 관절 범위 합. 0.08 로 바꾸려면 상수 1 줄 + test 1 줄 |
| G1 pad 범위 | "손가락 pad" | pad z_H 85.4–106.1 mm (pad box ∪ 손가락 mesh 의 평평한 안쪽 면, 손끝까지), 폭 ±8.0 mm | pad box 만 쓰면 (85.4–101.4) 손끝 4.7 mm 가 손가락 box 밖으로 빠진다. 안쪽 면은 y ≈ 0 으로 pad 와 같은 면이다 |
| G1 frame | "spec 의 frame 변환" | spec 에 `frame = R_HG` (원점 같음, 순수 회전). Panda `x_G = y_H, y_G = x_H, z_G = −z_H` | controller 안에서 손 자세를 gripper frame 으로 읽고, IK 목표와 `GraspAnchor.goal_R` 만 hold-link frame 으로 되돌린다. 후보 생성 (`candidates.py`) · planner 는 손대지 않음 |
| A5 껍질 = 실행 때 계산 대신 파일 | "RB-Y1 과 같은 생성 규칙" | export (`outputs/verify/SL2/raw/geo_libero.npz`, git 제외) 로 한 번 계산해 JSON 으로 고정. test 가 export 가 있으면 다시 계산해 digest 같음을 본다 | 서버가 export 없이 뜬다. self-filter cover 와 같은 방식 |
| A5 contype=0 문제 | "껍질 생성에서만 mesh 를 읽도록" | `cover.build_mujoco_model(collidable=True, joints=True)` 를 껍질 생성에서만 쓴다 (기본 False = self-filter cover 그대로) | joint 를 넣은 이유: 없으면 로봇 전체가 한 강체라 손가락 · 팔이 "강체 이웃" 으로 잡혀 숨은 면 판정이 틀린다. 넣으면 강체 묶음 = 손 · `panda_link7` · eef (RB-Y1 의 FT sensor 와 같은 역할) |
| A6 허용 구 | "panda_hand franka.yml 구 + A5 껍질 구" | `palm_contact_rows` 그대로 (link 이름 = hold link) → 18 + 111 = **129** | 껍질 구가 `panda_hand` link 에 붙으므로 코드 변경 없음 |
| A9 scope nbc 의 기록 | "A4 · G1 닫는 행 목표는 막지 않는다" | `all` 이었으면 막혔을 요청을 `hsr6.detail.objective.{a4,g1}_kept_by_scope = true` 로 남기고 `changed.objective` 는 바꾸지 않는다 | §6 "함께 보고: scope all 이면 막혔을 요청 수" 를 기록에서 바로 센다 |
| 주입을 넘기는 조건 | A1 "flag 가 꺼지면 아무것도 넘기지 않는다" | Panda 주입 (`safe_policy_injections`) · `hsr6=` · scope 는 **이식한 최종 구성 option 중 하나라도 켜졌을 때만** SafePolicy 에 넘긴다 | V11AB · pb 서버의 SafePolicy 생성 인자가 SL6 전과 같다 (기존 SL5j test 가 그것을 고정한다) |
| `--plan-horizon 16` | "10 으로 대체" | argv 에는 V11AB 와 같이 16 을 두고 Panda 서버가 H 10 · K 5 · plan 10 으로 바꾼다 (`PANDA_HORIZON`, 지금 V11AB 와 같은 경로) | V11AB argv 와 다른 곳을 줄인다. VA4 test 가 planned 10 · K 5 를 확인 |
| GPU warm-up 실패 | "경고 + config 기록" | warm-up 예외를 잡아 서버는 뜨고, `!!!` 경고 + `final_config.gpu_seeds.warm_up_error` | RB-Y1 `main` 은 warm-up 예외로 죽는다. task 문장대로 Panda 는 뜨고 크게 말한다 |

## 1. 바뀐 파일 · md5 (작업 트리, commit 하지 않음)

<FILES>

## 2. 항목별 (file:line)

| # | 무엇을 | 어디서 |
|---|---|---|
| A1 | version 2 개 · 최종 flag · base argv (pre-merge 없음) · probe 필수 · attention cell 교체 | `profile.py:254-298` (`FINAL_CONFIG_FLAGS`, `G1A4_FLAGS`, `version_server_flags`), `:300-305` `version_server_argv`, `:357-363` `v11ab_server_argv(pre_merge=)`, 기록 `version_flag_values` `:313-328` (V11AB 계열은 키가 그대로), `match_ag3s_version` `:330-347` |
| A1 | 서버 wiring: attention source (T57-A kwargs) · HSR6 probe → `AttentionPolicy`, `hsr6=` · scope → `SafePolicy`, GPU warm-up | `server.py:521-553` `attention_policy_kwargs`, `:583-657` `build_safe_server` (주입 조건 `:617-620`), `:555-580` `gpu_seeds_setup` |
| A1 | launcher `--hsr6-probe` · `--attention-source`, 시작 전 검사 | `launch_libero_paired.py:64-90`, `:392-397`, `:425-427` |
| A2 | target exclusion · ROI carve · evidence 의 config 병합 · 생존 검사 | `profile.py:455-457` (`v11ab_config_dict`), `:630-636` (`build_panda_ag3s`), 기록 `info.ag3s.target_exclusion` `:652-653` |
| A3 | SafePolicy 주입 (기본 = RB-Y1) | `safe_policy.py:84-98` (`hold_site_position(offset)`, `rby1_arm_joint_name`), class 기본값 `:183-191`, 생성자 `:216-221` · `:226-245`, multi-seed 손바닥 `:496-501`, NBC 손가락 `:2197-2205`, A6 site `:2250`, 손목 관절 `:2311-2313`; `sqp.py:653-656` (`palm_links`) |
| A3 | Panda 값 | `server.py:474-476` (`panda_joint1..7`, 손가락 둘), `:494-505` `safe_policy_injections` |
| A4 | GPU seed robot | `gpu_seeds.py:71-109` (`GpuSeedRobot`, `RBY1_ROBOT`, `robot_of`, `set_robot`), `robot_spec(gpu_robot=)` `:160-187`, batch `:374-378` · `:389` · `:518` · `:619` · `:802`; Panda `server.py:485-491` |
| A5 | Panda 껍질 구 | `hand_geometry.py` (새), `panda_palm_shell.json` (새, 생성물), `model.py:249-300` · `:405-409` (`palm_shell=` 맨 뒤에 붙임), `cover.py:144-206` (`collidable` · `joints`), `profile.py:617-621` |
| A6 | 기준점 `panda_grip_site` | `hand_geometry.py` `PANDA_HOLD_SITE_OFFSET_M = (0, 0, 0.097)`, 주입 `server.py:503` |
| A7 | A4 거절만 풂 | `server.py:312-338` (`UNPORTED_FINAL_CONFIG_DESTS` 에서 뺌) |
| A8 | Panda gripper spec · frame · IK 관절 · 역변환 gripper 열 | `gripper.py:67-71` (`frame`), `:113-131` (`PANDA_FRAME`, `PANDA_LEFT`); `geo_grasp.py:62-172` (`gripper_specs` · `arm_joints` · `_hand_pose` · `_to_hold`), IK `:123-133` · `:260-268`, goal_R `:373` · `:434`, 추적 `:473-475`; `action.py:331-347` `apply_deferred_close`, `server.py:252-274` |
| A9 | `--hsr6-objective-scope` | `serve_safe.py:1977-1980` (flag), `:2779-2801` (meta · 거절), RB-Y1 main 전달 `:4100-4102` (기본이면 인자 없음); `safe_policy.py:1323-1340` (`_hsr6_objective_ok(term)`), `:1264` (G1), `:1349` (A4), 기록 `:1312-1313` · `:563-565` |
| A10 | 거절 목록 | `server.py:312-338` (103 → 68 dest, 이식한 35 + `hsr6_probe` 를 `PORTED_FINAL_CONFIG_DESTS` 로), `:340-362` (`--hsr6-*` 에 probe 필수, `--palm-shell right` 거절), `profile.py:89-91` (`UNPORTED_BUILD_KEYWORDS` 에서 `palm_shell` · `target_exclusion` 뺌) |
| A11 | `placed_no_seed_certify` inert | `server.py:478-481` · `:628-631` (시작 log `!!!` + `final_config.inert_on_panda`), recorder meta `:413-416` |

### A5 — Panda 손바닥 껍질 구 (panda_hand frame, mm)

`hand_geometry.fit_palm_shell` (RB-Y1 상수 그대로: overflow 5 mm, 표본 1 mm, 격자 2 mm, union_eps 1 mm, 파지 면 cos 0.5).
입력 = LIBERO export `outputs/verify/SL2/raw/geo_libero.npz` (geometry digest `ea38872f…`, SL2 · SL3 self-filter cover 와 같은 파일).
shell digest `8ce53131619d7fb75fb2727a4844c826376a2a9bed96b3a706c0f1673bc486a3`.

| 항목 | 값 |
|---|---|
| 구 수 | **111** (link `panda_hand`, role `palm_shell`, 제약 모델 맨 뒤: `--links gripper` 22 → 133, `arms` 59 → 170) |
| 반지름 | 1.89 – 34.19 mm (중앙 13.89, 10 % 5.0, 90 % 27.5) |
| 중심 범위 (panda_hand frame) | x −27.6 … 28.4, y −100.0 … 94.0, z −23.9 … 64.1 |
| 구가 덮는 범위 (중심 ± r) | x −34.6 … 34.6, y −108.0 … 105.0, z −30.1 … 71.0 |
| 손 collision mesh (비교) | x ±31.6, y −104.0 … 100.4, z −25.9 … 66.0 (y = 닫는 방향, z = 접근 방향) |
| 손가락 통로 | 반폭 39.0 mm (열린 손가락 안쪽, franka.yml 손가락 구), 손 바닥면 z 65.96 mm |
| 덮을 표본 / 못 덮은 표본 | 41489 / 1 (최대 간격 0.07 mm), 넘침 실측 5.04 mm |
| 빠진 표본 | 파지 면 2729 (통로 안 · 법선 손끝 쪽), 숨은 면 8115 (`panda_link7` flange 아래 윗면) |
| 파지 공간 침범 | 0.000 mm (통로 안 구는 손 바닥면 z 65.96 mm 보다 손끝 쪽으로 나오지 않는다, test 확인) |

A6 허용 구 = `panda_hand` 의 franka.yml 구 18 + 껍질 111 = **129**. 기준점 `panda_grip_site` = `panda_hand` + (0, 0, 0.097) m
(URDF FK 로 확인, 차 < 1e-9 m). RB-Y1 은 hold link + (0, 0, −0.100) m.

### A8 — Panda gripper spec (gripper frame, m) — `gripper.PANDA_LEFT`

| 항목 | 값 | 측정 (`hand_geometry.measure_gripper_spec`, export) |
|---|---|---|
| frame `R_HG` | ((0,1,0),(1,0,0),(0,0,−1)) | x_G = y_H (닫는 방향, `panda_finger_joint1` 축 +y), z_G = −z_H (접근 = +z_H) |
| max_opening | 0.0790 | 열림 끝 pad 면 거리 79.0 mm (§0 확인 필요) |
| pad_half_y | 0.0080 | pad box x_H ±8.0 mm |
| pad_z | (−0.1061, −0.0854) | pad box z_H 85.4–101.4 ∪ 손가락 안쪽 면 ~106.1 |
| finger_thickness | 0.0213 | pad 면 뒤 손가락 mesh (pad z 범위) |
| palm box | (−0.1040, −0.0316, −0.0660) – (0.1004, 0.0316, 0.0259) | 손 collision mesh |

## 3. 검증 (task §2.1, 결과 전 고정)

모두 CPU · 한 번에 1 process · `nice -n 10` · `.venv-ag3s` (OMP/OpenBLAS 2 thread). GPU 는 쓰지 않았다.
기준 tree = `git worktree add --detach <scratch>/head_tree 553ad13e` (끝나고 지움). worktree 에는 submodule
`third_party/curobo` 가 없어 메인 checkout 것을 symlink 했다 (franka.yml 읽기만, VA3 에서만 필요). 로그 첫 줄 `trees:` / dump 의
`tree` 키로 module 위치를 확인했다.

| # | 검사 | 결과 | 판정 |
|---|---|---|---|
| VA1 | `tests/trajopt` + `tests/benchmark` (새 test 2 파일 포함) | 2,505 passed · 41 skipped · 실패 0 (알려진 numpy 기존 실패 `test_t43t_replay::test_off_is_bit_identical_to_the_pre_fix_commit` 1 개 deselect) | **통과** |
| VA2 (a) | SL5a golden `--check` · `--check-serve-safe` | 둘 다 `identical: true, n_differences 0` | **통과** |
| VA2 (b) | RB-Y1 최종 구성 기록 FIN_k0 재풀이 (장면 211 chunk 0–74, 장면 119 chunk 209–236), HEAD `553ad13e` worktree 대 SL6-A tree | refined **103/103 bit 동일** (211: 75/75, 119: 28/28). 요청별 기록 102/103 같음 — 다른 1 개 (211 seq 16, 여러 초기값 발동 요청) 는 `multi_seed.*_ms` 시간 필드만. 두 tree 모두 merge 검증 (2) 의 merge tree 출력과도 103/103 bit 동일 | **통과** |
| VA3 | Panda V11AB · V11AB-pb7 config dump (launcher argv, parse namespace, TrajOpt config, build kwargs, version flags · 읽어낸 version, server `info` = panda_server_config.json, SafePolicy 생성 인자, recorder meta, profile manifest), HEAD 대 SL6-A | 차이 **2 개 = 새 option `hsr6_objective_scope` 가 namespace 에 기본값 `all` 로 생긴 것** (두 version 각 1). 나머지 전부 같음 | **통과** |
| VA4 | `AG3S-final-panda` · `-g1a4` argv: parse · `reject_bad_flag_combinations` · `check_unported_flags` · config · 서버 build (stub policy, checkpoint 미로딩, GPU 가용성 probe 는 통과로 둠, warm-up 생략) | 둘 다 통과. config dump `info.final_config` 에 A5 껍질 (111 구, digest, margin 5 mm), A6 기준점 (`panda_grip_site`, (0, 0, 0.097), 허용 행 129), gpu seed robot (panda / `panda_libero.urdf` / `panda_link0` / `panda_hand`), ④ scope `nbc`, inert 목록 (`--placed-no-seed-certify`), 손목 관절 `panda_joint5..7`, multi-seed 손바닥 `panda_hand`, G1 spec (g1a4 만) | **통과** |

### VA1

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
MUJOCO_GL=osmesa PYTHONPATH=$PWD OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 10 \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider -rfE tests/trajopt tests/benchmark
```

**lead 기입 (2026-10-29 23:35 KST):** implementer 세션이 끝나 VA1 은 lead 가 같은 작업 트리로 돌렸다 (A12 · L8H6 · probe 반영 뒤). `nice -n 10 .venv-ag3s/bin/python -m pytest -v -p no:cacheprovider tests/trajopt tests/benchmark tests/ag3s/test_sl6_b6_head_select.py --deselect tests/trajopt/test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit` → `2505 passed, 41 skipped, 1 deselected in 3010 s`. 로그: scratchpad `sl6a_va1v.log` (요약 줄을 이 문서에 옮겼다).

새 test 가 고정하는 것:

| 파일 | 경우 |
|---|---|
| `tests/trajopt/test_sl6a_injections.py` (14) | A3 기본값 = RB-Y1 (생성자 · class 기본값, `HOLD_LINK_SITE_OFFSET_M`, `{hand}_arm_{j}`, `MS.PALM_LINKS`, `gpu_seeds.RBY1_ROBOT` 상수), `robot_spec` 기본 = RB-Y1 spec · Panda robot 은 3 키만 다름, `set_robot` 두 optimizer · batch 뒤 거절; Panda: A6 site = `panda_grip_site` FK · 창 30 mm 켜짐 · 원점 (97 mm) 꺼짐 · 허용 행 = `panda_hand` 129 (껍질 111), NBC 손가락 둘 · 손목 `panda_joint5..7` (주입 없으면 손가락 하나 · 손목 0 — 고친 결함), A9 `all`/`nbc` 두 값 · 기본 호출 = 예전 규칙 · flag parse · 거절 · meta 키, A8 frame: 열린 손에서 손가락 원점 잇는 축 = x_G, palm → grip site = −z_G, `_hand_pose`/`_to_hold` 왕복, RB-Y1 spec 은 변환 없음, G1 추적 관절 주입 |
| `tests/benchmark/test_sl6a_panda_final.py` (15) | A5 commit 파일 (111, 상수, keep-out), 껍질 없는 모델 = 껍질 모델의 앞부분 (gripper · arms), `hands right` · 다른 overflow 거절, export 로 다시 계산 = 파일 digest (export 있을 때), self-filter cover model 기본 불변; A8 spec = export 측정 (±0.1 mm), 닫기 미룬 chunk 의 a6 덮어쓰기 · 안 미룬 chunk bit 동일; A1 최종 flag = RB-Y1 최종 argv − V11AB flag − G1A4 + scope, pb flag 없음, pre-merge 없음, probe 필수, launcher argv, 읽어낸 version; VA4 두 판 build + dump; 거절 (probe 없음, palm-shell right, 미이식 flag, pre-merge 와 ROI carve 동시); recorder meta 키; VA3 in-process (HEAD server.py 로 V11AB · pb7 build = 지금 build) |
| 기존 test 수정 | `test_panda_sl4.py` (`UNPORTED_BUILD_KEYWORDS` 에서 뺀 두 keyword 를 따로 확인), `test_panda_sl5j_recorder.py` (version 읽기 test 는 V11AB 계열만, 최종 판은 새 파일), `_sl5a_golden.py` (`PB_DEFAULT_OFF` 에 `hsr6_objective_scope: all` — 기본값일 때만 비교에서 뺀다) |

로그 · 원시: `$R/VA1/full_pytest.log`, `$R/VA2/{head_553ad13e,sl6a_tree}/`, `$R/VA2/fin_compare.json`, `$R/VA2/compare_stdout.txt`,
`$R/VA2/sl5a_*_check.log`, `$R/VA3/{head_553ad13e,sl6a_tree}.json`, `$R/VA3/compare.json`, `$R/VA4/*.json`,
도구 `$R/tools/va2_replay.py` (merge 검증 `fin_replay.py` 무수정 사본), `$R/tools/va3_dump.py`, `$R/tools/va4_dump.py`.

## 4. 서버 argv 두 줄

`<…>` 는 lead 가 채운다. `--port` · `--record-constraints` 는 launcher 가 붙인다.
launcher 로는 `--ag3s-version AG3S-final-panda[-g1a4] --hsr6-probe <npz> [--attention-source <SPEC>]`.

```text
# AG3S-final-panda  (G1 · A4 제거판)
python -m AG3S.trajopt.serve_safe --config pi05_libero --checkpoint <pi05_libero checkpoint> --model-xml <checkout>/robots/rby1/description/models/rby1a/mujoco/model_transport.xml --port <port> --esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --robot panda --to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 --attention-split-ratio-source raw --target-permission-strict --attention-source noun_target:L10H7 --attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1 --neighbour-contact-mm 20 --nontarget-evidence surface --placed-no-seed-certify --kept-geometry-follow --roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda --blas-threads 1 --to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass --palm-shell on --palm-shell-margin 5 --hsr6-attach-assist --hsr6-held-gate --hsr6-target-exempt --hsr6-objective --neighbour-palm-contact-mm 10 --hsr6-objective-scope nbc --hsr6-probe <Panda probe_6.npz (SL6-B B4)>

# AG3S-final-panda-g1a4  (작동판 = 위 + G1 · A4)
python -m AG3S.trajopt.serve_safe --config pi05_libero --checkpoint <pi05_libero checkpoint> --model-xml <checkout>/robots/rby1/description/models/rby1a/mujoco/model_transport.xml --port <port> --esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --robot panda --to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 --attention-split-ratio-source raw --target-permission-strict --attention-source noun_target:L10H7 --attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1 --neighbour-contact-mm 20 --nontarget-evidence surface --placed-no-seed-certify --kept-geometry-follow --roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda --blas-threads 1 --to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass --palm-shell on --palm-shell-margin 5 --hsr6-attach-assist --hsr6-held-gate --hsr6-target-exempt --hsr6-objective --neighbour-palm-contact-mm 10 --hsr6-objective-scope nbc --grasp-anchor 2000 --grasp-anchor-path-weight 0 --geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off --hsr6-probe <Panda probe_6.npz (SL6-B B4)>
```

- `--model-xml` 는 V11AB 와 같이 RB-Y1 MJCF 가 argv 에 남지만 Panda 서버는 읽지 않는다 (adapter URDF, `recorder_meta.model_xml = None`).
- `--finger-cover both` · `--capsule-radius-scale` 등 RB-Y1 구 옵션은 V11AB 와 같이 Panda 모델에 적용하지 않는다 (`v11ab_build_kwargs_not_applied`).
- head 재선택 (§4) 결과가 L10H7 이 아니면 launcher `--attention-source noun_target:L<l>H<h>` 로 바꾼다 (version 이름은 그대로 읽힌다).

## 5. verifier · lead 가 알아야 할 것

- **새 flag (기본 꺼짐):** `serve_safe --hsr6-objective-scope {all,nbc}` (기본 all, `--hsr6-objective` 없이 주면 거절).
  launcher `--hsr6-probe NPZ` (최종 판 필수, 다른 version 은 거절) · `--attention-source SPEC` (최종 판만).
- **새 profile key:** `AG3S-final-panda`, `AG3S-final-panda-g1a4` (`profile.FINAL_CONFIG_VERSIONS`). V11AB · pb1–pb7 의 argv ·
  기록 키는 그대로 (VA3).
- **기록 키 (켤 때만 생김):** `panda_server_config.json` 의 `final_config` (위 VA4 항목) · `ag3s.target_exclusion` · `ag3s.palm_shell`;
  recorder meta `final_config` · `palm_shell` · `attention_source` · `hsr6` (+ `objective_scope`) · `subtask.placed_no_seed_certify(_inert_on_panda)`;
  요청마다 `summary_json.hsr6.objective_scope` 와 `hsr6.detail.objective.{a4,g1}_kept_by_scope` (scope nbc 에서 all 이었으면 막혔을 요청),
  응답 `panda/conversion.g1_deferred_close` (G1 이 닫기를 미룬 chunk 에서 덮어쓴 행 · a6 전후). §6 "G1 / A4 가 목표를 맡은 요청 수" 는
  기존 `grasp.geo.anchor` · `grasp.anchor` · `to.metrics.grasp_anchor.variant` 로 센다.
- **GPU 단계 (lead) 에서 처음 확인되는 것 — CPU 로는 볼 수 없었다:**
  1. cuRobo 가 `panda_libero.urdf` (mesh · inertial 없는 link) 로 seed batch 를 짓는가 (`final_config.gpu_seeds.warm_up` 기록,
     실패하면 `warm_up_error` + 시작 log `!!! TO GPU seeds (panda)` — 서버는 뜨고 발동 chunk 마다 CPU 16 seed 로 푼다 = 느림).
  2. ROI carve (`--roi-carve-device cuda`, footprint) 가 Panda 카메라 · Panda 자기 필터와 도는가. CPU test 는 config 만 확인했다.
  3. noun attention 이 LIBERO 문장에서 target 을 내는가 (SL6-B B1; 서버 시작 log 의 `noun_rule` 이 `libero` 여야 한다,
     `fruit` 이면 B 의 규칙이 없는 것 — §6).
  4. 요청 시간: 제약 구가 22 → 133 (껍질 111). RB-Y1 은 손마다 껍질 168 개.
- **열린 위험:**
  - G1 이 LIBERO bowl 에 후보를 낼지 모른다: 후보 생성의 target 구 반지름 상한 = 0.5 × 79 mm + 10 mm = 49.5 mm, LIBERO 검은 bowl 은
    지름 ~130 mm 라 테두리 쥐기 후보가 적거나 없을 수 있다 (그러면 correct 모드는 보정하지 않고 A4 가 슬롯을 맡는다). 측정하지 않았다.
  - A6 창 (grip site 60 mm) 과 bowl 중심: 테두리를 쥘 때 grip site–bowl 중심이 60 mm 를 넘으면 창이 켜지지 않는다.
  - 요청 수로 정한 값 (`--hold-follow-to 3`, G1 `geo_close_timeout` 8 · `geo_carry_requests` 3, HSR6 raw argmax) 은 그대로라 Panda 에서는
    시간으로 절반 (0.25 s / 요청). §1 대로 바꾸지 않았다.
  - `--placed-no-seed-certify` 는 Panda 에서 동작하지 않는다 (A11, 기록만).
  - G1 `max_opening` 0.079 (§0) — lead 확인 필요.
- 재생산할 산출물: 없음. `panda_palm_shell.json` 은 새 생성물 (commit 대상, 생성 명령은 `hand_geometry.py` 머리말).
- commit · GPU · 다른 세션 process: 하지 않음 / 쓰지 않음 / 건드리지 않음. 새 출력은 위 `$R` run 과 scratch 에만 썼다. worktree 는 지웠다.

## 6. SL6-B 에 요청

- `NounTokens.from_config(config_name, rule="libero")` 의 인자 이름 `rule` · 값 `"libero"` 를 유지해 달라. Panda 서버는 signature 에
  `rule` 이 있으면 `rule="libero"` 로 부르고, 없으면 fruit 규칙 + 경고로 뜬다 (`server.PANDA_NOUN_RULE`, `server._noun_tokens`).
- `load_hsr6_probe(path)` 가 돌려주는 객체의 `.name` · `.classes` 를 서버 기록에 쓴다 (`final_config.attention.hsr6_probe`). 5 class
  부분집합 probe 도 이 둘을 가지면 된다.
- `hsr6.objective_allowed(phase)` 의 뜻 (grasp 일 때만 True) 을 그대로 쓴다. A9 scope 는 SafePolicy 쪽에서만 바꿨다 (`hsr6.py` 무수정).

## 내가 기대하는 결과

(verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.)

- G2 (1 episode 기동): 서버가 뜨고 시작 log 에 `HSR6 (panda)`, `noun_rule libero`, palm shell 111, `TO GPU seeds warm-up (panda)` 가 보일 것.
  cuRobo 가 Panda URDF 로 batch 를 못 지으면 (가능성 낮지 않다고 본다: link 에 geometry 가 없는 URDF 를 cuRobo 가 처음 읽는다)
  `warm_up_error` 가 남고 서버는 CPU seed 로 돈다. 요청 시간은 V11AB-pb7 보다 길 것 (구 6 배, 여러 초기값 · ROI carve).
- G0 (V11AB-pb7 · C, LIBERO 10 init 0): merge 검증 (3) 의 run 과 npz array bit 동일일 것 (VA3 에서 config · SafePolicy 인자가 같고,
  주입은 최종 구성 option 이 꺼지면 넘기지 않는다).
- §6 smoke: G1 은 bowl 에 후보를 거의 못 내서 작동판에서 목표를 맡는 것은 대부분 A4 일 것으로 본다. 두 판의 safe success 차가 2
  episode 이상 벌어질 것으로는 보지 않는다 (→ "제거" 쪽). HSR6 Panda probe 가 approach 를 길게 내면 scope nbc 의 `*_kept_by_scope`
  기록이 A4 활성 요청의 상당 부분에 붙을 것이다.
