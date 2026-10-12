# SL4 — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-12 22:55 KST
> branch `feature/panda-benchmark`, 시작 HEAD `e152f658` (commit 안 함 — 새 파일 6개 모두 untracked)

## 무엇을 했나 (평이한 요약 먼저)

V11AB server 가 쓰는 AG3S 를 **AG3S core 를 한 줄도 바꾸지 않고** Panda 용으로 만드는 profile(`AG3S/adapters/panda/profile.py`)을 만들었다.
config 는 손으로 다시 적지 않고, `start_server.sh`/`env.sh` 에서 V11AB flag 를 읽어 **serve_safe 자신의 parser·검사·`main` 의 `build_ag3s(...)` 호출식(AST)** 으로
keyword 를 만들고, `build_ag3s` 의 config 조립(유일한 복사본)으로 config 를 만든다. 이 복사본은 test 가 **실제 `build_ag3s`** 를 같은 keyword 로 돌려
AG3S 생성자에 넘어가는 config 와 field 별로 대조한다(일치). Panda 차이는 config 두 줄(`contact.contact_links`, `esdf.bounds_lower` z −0.10)과
생성자 인자 셋(self-filter model, constraint model, attached parent link)뿐이고, `diff_against_v11ab()` 가 정확히 그 두 config 줄만 돌려준다.

그 위에 SL3 dev rollout 의 policy-call frame(221)을 **V11AB 의 실제 perception 경로**(`wire_pack` → `wire.strip_request` →
`wire.unpack_camera_observations` → robot_state 9→q7 split → `set_finger_joints` → `process_multi_debug(phase="approach", ["left"])`)로 다시 돌리는
offline harness(`benchmark/runners/perception_replay.py`)와, MuJoCo 정답(py3.8 `scene_export.py` → 3.11 `libero_scene_gt.py`: 물체 표면 sample, 정확한
점–삼각형 거리, collision primitive 포함 판정, depth 가시성 잔차)을 만들었다. Harness 는 grounding·cloud·ESDF(main / target-free)·sphere 정확 거리·
stage 시간·GPU memory 를 새 run 에 저장한다. GPU smoke(task 0 의 21 call)는 끝까지 돌았다.

**절차 문제 하나 발견**: task 에 적힌 `flock -w 1800 9 CMD … 9>/tmp/ag3s-vla-gpu-validation.lock` 은 공유 lock 을 잡지 않는다(아래 "verifier 가 알아야 할 것").

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/adapters/panda/profile.py` (새, 3.11) | 모듈 docstring 표 = override 목록과 근거 (RB-Y1 ↔ Panda) | 요청 1 |
| `profile.py:146-196` | `ta_flags`, `v11ab_server_flags`, `v11ab_required_args`, `v11ab_client_phase`: launcher 스크립트를 **텍스트로** 읽음(실행하지 않음 — `env.sh` 는 source 될 때 run dir 을 만든다). backslash-newline·주석 처리, `$VAR`/`$(ta_flags $C)` 치환, 미해결 변수는 예외 | 요청 1 "flag 를 serve_safe argparse 로" |
| `profile.py:204-245` | `parse_v11ab_args` (= `serve_safe.build_parser` + `reject_bad_flag_combinations` + `resolve_subtask_gate`), `v11ab_build_kwargs`: `serve_safe.main` 의 유일한 `build_ag3s(...)` 호출을 AST 로 찾아 **각 keyword 식을 serve_safe namespace 에서 eval** (log 전용 `plan_horizon_steps`/`rows_per_step` 제외 — `announce_row_budget` 에만 쓰임, test 가 확인) | 값 재기술 없음 |
| `profile.py:248-296` | `v11ab_config_dict` / `v11ab_ag3s_config`: `build_ag3s` 의 config 조립 복사 (esdf·pointcloud·constraint·timing·subtask + identity `with_overrides`) + `assert_ag3s_config` | 유일하게 피할 수 없는 복사 (모델 생성 중간에 있음) |
| `profile.py:298-380` | `panda_esdf_bounds` (`default_bounds()` 에서 z 만 −0.10), `panda_overrides`(`esdf_bounds="panda"|"v11ab"`), `panda_ag3s_config`, `effective_config_dict`(bounds `None` → `default_bounds()` 로 해석, `curobo_builder._resolve_bounds` 와 같은 규칙), `diff_against_v11ab` (선언 안 된 차이는 `reason: "UNDECLARED"`) | 요청 1 diff |
| `profile.py:382-415` | `PANDA_GRASP_LINKS`/`PANDA_HOLD_LINKS` → `safe_policy.grasp_parent_links(...)` = `("panda_hand",)`; `constructor_differences` (models, parents, 적용 안 한 RB-Y1 build option) | 요청 1 |
| `profile.py:417-451` | `build_panda_ag3s(esdf_bounds=, extra_overrides=)`: `AG3S(config, robot_model=build_panda_self_filter_model(), constraint_robot_model=…gripper, attached_parent_links=…)` + `build_ag3s` 와 같은 생성 후 설정(`held_body_cover=True`, `held_free_observed_pad`) + `assert_ag3s_config`. `extra_overrides` 는 test/진단 전용 | 요청 1 |
| `profile.py:453` | `manifest_entry`: 스크립트·serve_safe SHA-256, flag, 호출식, kwargs, effective bounds, config diff, constructor 차이, Panda config 전체 | manifest |
| `benchmark/adapters/libero_panda/scene_export.py` (새, py3.8) | episode 별 LIBERO env 재생성 → call frame 마다 `sim_state_image` 복원·forward → geom/body world pose, `T_world_base`(기록과 대조), 정적 geom 표, **비로봇 geom 의 mesh**(mount 포함), MuJoCo contact, 로봇 collision geom × 비로봇 collision geom `mj_geomDistance`(< 0.4 m, fromto 포함), BDDL role(`task_target`). `client_env` re-exec, 덮어쓰기 거부 | 요청 3 (SL3 GT export 일반화) |
| `benchmark/evaluators/libero_scene_gt.py` (새, 3.11) + `__init__.py` | `SceneGT`: instance(target/destination/object/fixture + `mount`), visual(group 1) 표면 sample(면적 가중·seed 고정·instance 당 cap), frame 별 base 좌표, `exact_distance`(점–삼각형 Ericson, AABB 가지치기; mesh 그대로, primitive 는 96 분할), `inside`(collision primitive 해석적), `target_position_base`; `depth_residual`(기록 depth − 점의 camera z) | 요청 3 |
| `benchmark/runners/perception_replay.py` (새, `.venv-openpi-live`) | harness. `index_sources`(dev run ↔ audit run: audit 이 읽은 run 경로·episode npz/geom SHA-256·`obs_index == chunk_t` 검사), `index_scene_gt`(GT 파일의 source SHA·call frame 포함 검사), `build_observations`, `apply_frame_state`, `run_frame`, `front_end_probe`, `run_episode`, 출력·summary, `--dry-run` | 요청 2 |
| `tests/benchmark/test_panda_sl4.py` (새, 13 함수 / 14 case) | 아래 표 | 요청 4 |

손대지 않은 것: `AG3S/ag3s/**`, `AG3S/trajopt/**` (`git diff --stat HEAD -- AG3S/ag3s AG3S/trajopt` = 0, untracked 0), 기존 adapter·runner·test 파일, submodule, storage input, 실제 results 디렉터리.

### profile 이 유도한 V11AB 값과 차이 목록

V11AB flag (`v11ab_server_flags("V11AB")`): `--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20`. client phase = `approach`.

`diff_against_v11ab()` (config, effective 값):

| key | V11AB | Panda | 근거 |
|---|---|---|---|
| `contact.contact_links` | `{left: (ee_left, ee_finger_l1, ee_finger_l2), right: (ee_right, ee_finger_r1, ee_finger_r2)}` | `{left: (panda_hand, panda_leftfinger, panda_rightfinger)}` | RB-Y1 link 이름은 Panda 에 없음; 한 팔 = left manipulator (SL2) |
| `esdf.bounds_lower` | `(-0.3, -0.9, 0.0)` (= `default_bounds()`) | `(-0.3, -0.9, -0.10)` — `axes_changed: ["z"]` | LIBERO table top = base z −0.012 (SL1) |

`esdf.bounds_upper` 는 AG3S 가 두 bound 를 같이 요구해서 명시하지만 값은 `default_bounds()[1]` = `(1.2, 0.9, 1.6)` 그대로 (effective diff 없음).
`--esdf-bounds v11ab` 로 만들면 diff 는 `contact.contact_links` 한 줄.

`constructor_differences()` (config 밖):

| 무엇 | V11AB | Panda |
|---|---|---|
| self-filter model | RB-Y1 `build_robot_model` | `build_panda_self_filter_model()` (confined_hand_mount, 4271 구) |
| constraint model | `build_constraint_robot_model(links="gripper", sphere_options, finger_cover)` | `build_panda_constraint_model(PANDA_GRIPPER_LINKS)` (22 구) |
| attached parent links | `("ee_left", "ee_right")` | `("panda_hand",)` |
| constraint sphere options `{capsule_radius_scale 0.05, sphere_spacing 0.4, max_spheres_per_capsule 32}` | 적용 | **적용 안 함** (franka 구는 길이 0 role-collision capsule → 0.05 면 반지름 5 %, SL2) |
| `finger_cover: both` | 적용 | **적용 안 함** (RB-Y1 손가락 덮개) |

TO·기록 flag(`esdf-margin`, `obstacle-margin`, `plan-horizon`, `lookahead*`, `qp-eps`, `w-deflection-rate`, `record-constraints`)는 AG3S 설정이 아니라 parse 만 하고 쓰지 않는다 (SL-5).

### harness 설계

| 단계 | 무엇 | V11AB 대응 |
|---|---|---|
| 입력 join | dev run 의 `episodes.jsonl`·`policy_records`(SL3 `index_source_run` 재사용) ↔ audit `attention_index.jsonl`: audit 의 `source_run_dir` = dev run, episode npz/geom SHA-256 일치, audit `obs_index` = episode `chunk_t`. 다르면 시작 안 함 | — |
| 관측 | `wire_pack.wire_from_geom(geom, row)` → `wire.strip_request` → `wire.unpack_camera_observations(scene, robot_model, attention={head, left_wrist})` | `SafePolicy._scene_fn` 와 같은 함수 |
| attention | audit `cell_grid_policy[k]` 를 `attention.unmirror_grid(…, unmirror=mapping=="unmirror")` 해서 `attention_map` 에 부착 (16×16 grid 그대로 — V11AB 도 grid 를 넘기고 `lift` 가 resampling) | `attention_extractor` 의 alias 자리 |
| robot_state | 두 camera 의 9-D 가 같은 순간인지 검사 → `split_robot_state` → 각 observation `robot_state = q7`, `set_finger_joints({finger1: +rs[7], finger2: −rs[8]})` | `_apply_finger_state` |
| latch 상태 | 매 frame `set_grasp_active(False)`, `set_grasp_latched(False)`, `set_subtask(None)`, `set_placed(False)` | grasp 전 SafePolicy 값 (Panda subtask label 없음 → gate on 이지만 ungated) |
| 처리 | `process_multi_debug(obs, phase="approach", active_manipulators=["left"], static_geometry=None)`; `execution_path` 없음 | `_scene_fn` |
| episode 경계 | AG3S 하나, episode 시작마다 `ag3s.reset()` | `SafePolicy.reset` |

저장 (`create_run`, settings 에 dev/audit run id·경로·manifest SHA-256, episode 별 npz/geom/attn SHA-256, GT 파일 SHA-256, profile manifest, 옵션):

- `perception/<stem>_perception.npz` — call 별: `grounding_status`, `has_target`, `target_centroid/confidence/attention_score/seed_camera/supporting`, `target_points`(+`_offsets`), `seed_indices/points`, `seed_camera_mask`(관측한 camera bit: head 1, left_wrist 2), `seed_argmax_camera`(core `_with_camera_provenance` 와 같은 규칙: 그 점에서 per-camera attention 이 가장 큰 camera; 0 head, 1 wrist), `manipulated_centroid/radius/state`, `has_target_free`, `cloud_points/camera_mask/attention/raw_attention/support/seed/target`(+offsets, `--no-clouds` 로 끔), `robot_state9`, `q7`, `fingers`, `sph_center (N,22,3)`·`sph_radius`·`sph_link`·`sph_authorized`, `sph_esdf_main/tf`·`sph_tier_main/tf`, `sph_all_center (N,61,3)`(franka 61 구), GT: `gt_points_local`/`gt_point_geom`/`gt_point_instance`·`instance_names/roles`, `gt_esdf_main/tf`·`gt_tier_main/tf (N,P)`, `gt_depth_residual_head/left_wrist (N,P) f16`, `sph_exact (N,22,I)`(구 중심 → instance visual 표면 거리, 0.45 m 넘으면 inf), `sph_inside (N,22,I)`, `gt_target_position`, `slice_{xy,xz}_{main,tf} (N,101,101) f16`·`slice_centre`, `t_process_s`, `t_record_s`, `stage_ms`(JSON), `n_points_before_fusion`, `n_self_filtered`, `n_filtered_points`, `probe_self_filter_ms_{cam}`, `gpu_max_allocated`, `gpu_reserved`. tier code = `tiers` 배열 index (`outside, coarse, fine, target_free, window_edge, static`; −1 = 답 없음).
- `frames.jsonl` — call 별 요약 + `manipulated`, `exclusion`, `notes`, `fusion` metrics, `self_filter`, `finger_joints`, `esdf` stats 전체(window·target_free·tiers), `stage_ms`, `front_end_probe`, `gpu_memory`.
- `perception_index.jsonl`, `summary.json`(첫 call 따로, 나머지 p50/p95, stage 별, probe, GPU, 끝날 때 `nvidia-smi` compute apps).

`front_end_probe`: `profile` 의 `scene_reconstruction` 은 back-projection·downsample·4271 구 depth mask·fusion 을 한 stage 로 잰다(fused path 의 `robot_self_filter` 는 core 가 0.0 으로 적는다). self-filter 만의 시간을 위해 frame **뒤에** `device_recon.camera_front_end` 의 단계를 camera 별로 그 frame 의 filter model/inflation/guard 로 다시 돌려 synchronize 하며 잰다. 다시 만든 mask 가 frame 의 `CameraResult.depth_mask` 와 pixel 단위로 같은지 `mask_equals_frame` 에 적는다. frame 의 `t_process_s` 에는 포함되지 않는다. `--no-stage-probe` 로 끈다.

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
OMP_NUM_THREADS=2 .venv-ag3s/bin/python scripts/setup/run_without_legacy.py -m pytest -q -p no:cacheprovider -rs tests/benchmark/test_panda_sl4.py
# 14 passed in 36.83s
OMP_NUM_THREADS=2 .venv-openpi-live/bin/python scripts/setup/run_without_legacy.py -m pytest -q -p no:cacheprovider -rs tests/benchmark/test_panda_sl4.py
# 14 passed in 38.80s
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu bash scripts/testing/run_cpu_regression.sh
# == A: 73 passed, 77 warnings in 41.37s          (SL3 와 같은 73)
# == B: tests/benchmark (.venv-openpi-live): 92 passed in 185.08s   (SL3 78 + SL4 14)
# == C: tests/benchmark (.venv-ag3s): 90 passed, 2 skipped in 122.07s (SL3 76 + SL4 14; skip 2 = SL3 와 같음)
# == cpu regression exit status: 0
```

| test (`test_panda_sl4.py`) | 무엇을 |
|---|---|
| `test_v11ab_flags_are_read_from_the_launcher_scripts` | V11AB flag 17 개 포함, `--servo-accel`(V11ABC 전용)·`--record-constraints`($REC) 없음, 모르는 arm 거부, `--config`·model xml, client phase `approach`, parse 후 gate True |
| `test_build_kwargs_follow_the_serve_safe_main_call` | kwargs 이름 = `main` 의 호출 keyword − log 전용 2 개; 그 둘이 `build_ag3s` 안에서 `announce_row_budget` 에만 쓰임(AST); 값 (curobo, 0.02, gripper, exclude-authorized, self_collision False, identity 6 키, sphere options, gate) |
| `test_profile_config_equals_the_real_build_ag3s_config` | **실제 `serve_safe.build_ag3s`**(RB-Y1 MJCF·모델 생성 포함, ~16 s)를 같은 kwargs 로 돌리고 `pipeline.AG3S` 만 capture 로 바꿔, 생성자에 넘어간 config 와 profile config 의 effective dict·`to_dict()` 가 완전히 같음; parents = `("ee_left","ee_right")`, `held_body_cover` True |
| `test_diff_against_v11ab_is_exactly_the_declared_overrides` | diff key = `OVERRIDE_REASONS` 두 개, bounds 는 z 만, upper 동일, contact links 값; `v11ab` box 면 contact 한 줄; 선언 안 한 override 는 `UNDECLARED` |
| `test_build_panda_ag3s_injects_only_the_panda_geometry` | config 값, `CuroboFieldBuilder._resolve_bounds` 가 실제로 쓸 box (panda: z −0.1, v11ab: 0.0), 구 4271/22, constraint link, parents `("panda_hand",)`, held_body_cover, max opening 78 mm, manifest |
| `test_robot_state_split_and_finger_parameters_reach_both_models` | 합성 geom: 두 camera, `robot_state` (7,) = rs[:7], depth mm→m, finger 부호(joint2 반전), `apply_frame_state` 뒤 두 model 의 `joint_parameters`, record; camera 간 state 불일치 거부 |
| `test_attention_is_attached_per_camera_in_the_chosen_orientation` ×2 | one-hot policy patch → unmirror 는 열 `15−c`, mirror 는 그대로; 모르는 mapping 거부 |
| `test_point_triangle_distance_is_exact`, `test_primitive_meshes_samples_and_containment`, `test_depth_residual_sign_and_bounds` | Ericson 거리 ≤ 조밀 sample 거리 (차 < 3 cm @ 4 만 점), 꼭짓점·변·면 영역 해석값; box 면적, 원기둥 반지름, sphere sample 오차 한계, seed 결정성; depth 잔차 부호·화면 밖·depth 0 |
| `test_scene_export_is_python38` | `ast.parse(feature_version=(3,8))` |
| `test_harness_dry_run_output` (storage) | dry-run: 2 episode × 2 call, 두 camera, q7, attention = audit `cell_grid_opencv`, settings(step, phase, mapping, diff, run id, SHA), GT 없는 dir 거부 |
| `test_scene_export_and_one_cpu_frame` (storage + `.venv-libero`) | py3.8 export 1 frame (기록 대비 `T_world_base`·robot qpos 차 0.0, 덮어쓰기 거부) → role 표, sample 이 자기 표면 위(< 1e-9 m), table top base z −0.012 (±2 mm) → **CPU 에서 한 frame** (legacy ESDF + `relax`: test 전용 `extra_overrides`; device switch 3 개 `cpu`): npz shape, q7, cloud offsets, field grid origin z ≈ −0.10 |

**GPU smoke** (`AG3S_STORAGE_ROOT` = scratchpad 사본: dev·audit run 복사, audit index 의 경로만 사본으로 바꿈; 실제 results 에 쓰지 않음):

| 실행 | 결과 |
|---|---|
| `scene_export.py --task-ids 0 --limit-frames 3` / 전체 21 frame | 11 s / 12.4 s, 기록 대비 `T_world_base`·qpos 차 0.0, 21 frame contact 1745 행·geomdist 26365 행, 4.5 MB |
| harness `--task-ids 0 --limit-calls 3` (13:35:06–13:35:36 UTC) | exit 0, 첫 call 16.1 s(JIT), 다음 118 / 100 ms |
| harness `--task-ids 0` 21 call (13:37:02–13:37:48 UTC, 46 s) | exit 0; 첫 call 7.93 s; 이후 20 call `t_process` p50 89.6 ms, p95 122.9 ms; stage p50: scene_reconstruction 40.4, esdf 41.2 (constraint_generation 42.3 에 포함), support 2.9, grounding 3.2 ms; probe self-filter mask head p50 16.9 / wrist 19.0 ms, mask 일치 42/42; torch peak 815 MB, `nvidia-smi` process 1700 MiB; call 당 기록 오버헤드 ~1.15 s; episode npz 6.2 MB |

ATS·LTE·field 정확도 숫자는 재지 않았다 (verifier 몫).

## verifier 가 알아야 할 것

- **GPU lock 명령 형식**: `flock -w 1800 9 CMD … 9>/tmp/ag3s-vla-gpu-validation.lock` 은 명령이 뒤에 붙으면 `9` 를 **파일 이름**으로 받아 cwd 에 `./9` 를 만들어 잠그고, 공유 lock 은 fd 9 로 열려 있기만 하다 (scratchpad 에서 확인: 바깥 flock 아래에서 `flock -n x.lock` 이 성공). 쓸 형식: `( flock -w 1800 9 || exit 1; CMD ) 9>/tmp/ag3s-vla-gpu-validation.lock` 또는 `flock -w 1800 /tmp/ag3s-vla-gpu-validation.lock CMD`. 내 두 smoke 는 그 잘못된 형식으로 돌아 **공유 lock 을 잡지 않았다** (시작 직전 compute apps 비어 있음, 30 s·46 s, 1.7 GiB). main 에 알렸다. 생긴 빈 `./9` 는 지웠다.
- **실행 명령** (각 1 process, foreground):
  ```bash
  cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
  DEV=20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0
  AUD=20261012T202459_309e4c6da477_sl3-attention-audit-dev
  # 0. profile / diff (CPU)
  .venv-ag3s/bin/python -c "import json; from AG3S.adapters.panda import profile as pf; print(json.dumps(pf.manifest_entry(), indent=1, default=str))" > outputs/verify/SL4/profile_manifest.json
  # 1. GT export (CPU, .venv-libero, 10 episode ≈ 2–3 min, ~1 GB)
  .venv-libero/bin/python benchmark/adapters/libero_panda/scene_export.py --run $AG3S_STORAGE_ROOT/results/benchmark/$DEV --out outputs/verify/SL4/raw/scene
  # 2. dry-run (CPU, 아무것도 만들지 않음)
  .venv-openpi-live/bin/python benchmark/runners/perception_replay.py --source-run $DEV --attention-run $AUD --scene-gt outputs/verify/SL4/raw/scene --dry-run > outputs/verify/SL4/dryrun.json
  # 3. 본 측정 (GPU; 221 call ≈ 5–7 min → 시작·끝을 main 에 알림)
  ( flock -w 1800 9 || exit 1; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    .venv-openpi-live/bin/python benchmark/runners/perception_replay.py --source-run $DEV --attention-run $AUD \
        --scene-gt outputs/verify/SL4/raw/scene --run-id sl4-perception-dev ) 9>/tmp/ag3s-vla-gpu-validation.lock
  # 4. #4 반전 진단: 위 3 에 --attention-mapping mirror --run-id sl4-perception-dev-mirror
  # 5. #7 box 전/후: 위 3 에 --esdf-bounds v11ab --run-id sl4-perception-dev-v11abbox
  # 6. #10 회귀
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 JAX_PLATFORMS=cpu bash scripts/testing/run_cpu_regression.sh
  ```
  stdout 첫 줄이 새 run 경로. Harness 는 `nvidia-smi` 에 compute process 가 있으면 시작하지 않는다 (`--allow-busy-gpu`). `AG3S_PROJECT_ROOT` 가 이 checkout 이 아니면 거부.
- **새 플래그** (harness): `--attention-mapping {unmirror,mirror}`(기본 unmirror), `--esdf-bounds {panda,v11ab}`(기본 panda), `--scene-gt DIR` / `--no-gt`, `--task-ids`, `--limit-calls`, `--no-clouds`, `--gt-spacing 0.004 --gt-cap 3000 --gt-seed 0`, `--slice-half 0.25 --slice-spacing 0.005`, `--no-stage-probe`, `--allow-busy-gpu`, `--dry-run`. `scene_export.py`: `--run`, `--out`, `--task-ids`, `--frames {calls,all}`, `--limit-frames`, `--distmax 0.40`. 기존 기본값 변경 없음.
- **GT 의 정의** (scoring 때):
  - 표면 sample = instance 의 **visual geom (group 1, camera 가 render 하는 것)** 표면, 면적 가중, 4 mm, instance 당 최대 3000 점 (table·cabinet·stove·mount 는 cap 에 걸린다 — 밀도가 낮아짐). 바닥면·가려진 면도 들어 있으므로 "보이는 점" 은 `gt_depth_residual_<cam>` 의 |잔차| 로 고른다 (허용치는 verifier 선택; 음수 큰 값 = 가려짐).
  - `sph_exact` = 구 **중심** → instance visual 표면의 무부호 거리 (반지름 빼지 않음, 0.45 m 넘으면 inf). `sph_inside` = 중심이 그 instance 의 collision primitive 안 (LIBERO 물체 collision 은 box). MuJoCo 자체 수치는 scene npz 의 `geomdist`(로봇 collision geom ↔ 물체 collision geom, mesh 는 convex hull) 와 `contacts`.
  - instance: BDDL object·fixture(`main_table` = body `table`, 다리 포함) + `mount`(robot pedestal, `mount0_*`). role: target / destination / object / fixture / mount. 바닥·벽(world body)은 instance 가 아니다.
  - slice 중심 = GT target body 위치 (그림 배치용 privileged 값, AG3S 에는 안 들어감). `xy` = 그 z 의 수평면, `xz` = 그 y 의 수직면, ±0.25 m, 5 mm.
  - `gt_esdf_tf`/`sph_esdf_tf` = `target_free_distance` — target-free 계층이 없거나 창 밖이면 main 과 같은 값이 나온다 (그것이 권한 link 가 읽는 값). `has_target_free` 로 구분.
- **harness 가 V11AB 와 다르게 하는 것 (offline 이라서)**: (a) `execution_path` 없음 → fine window 는 `q_now` 로만 놓인다 (`esdf.window.hand_source = "q_now"`); (b) grasp latch·attach 없음 → grasp 이후 frame 에서도 bowl 은 쥐지 않은 target 으로 처리된다 (V11AB 는 attach 했을 것) — 운반 구간 숫자는 그렇게 읽어야 한다; (c) subtask label None (Panda probe 없음); (d) RB-Y1 constraint sphere option·finger cover 미적용 (위 표).
- **AG3S 내부 속성 읽기** (쓰기 없음): `front_end_probe` 가 `_frame_filter_config/_frame_filter_model/_frame_filter_inflation/_frame_guard_centre`, test 가 `_last_esdf_field` 를 읽는다.
- 재생산이 필요한 산출물: 없음 (dev·audit run 은 읽기만). GT export 는 새로 만들어야 한다 (위 1).
- 옛 기록과 호환이 깨지는가: 없음.
- core 변경이 필요한 것: **SL4 에는 없음.** SL-5 (online) 에는 `serve_safe.build_ag3s` 에 모델·contact links·bounds 주입 길이 없다 — core 를 바꾸거나 이 profile 을 쓰는 wrapper server 가 필요 (SL2·SL3 impl 의 표 그대로: attention alias·flip, chunk layout, finger state 입력원).

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->
- #1: core diff 0 줄, diff 목록 = 두 줄 (contact_links, bounds_lower z). baseline 둘 다 그대로 (core 무변경).
- #2: 접근 구간 대부분 frame 에서 `has_target` (smoke t0: 21 call 중 19, `no_cluster` 2 는 grasp 이후). target 점은 bowl 위에 있되 bowl 옆 table/plate 점이 섞여 precision 이 recall 보다 낮을 것; centroid 오차 수 cm 이하 (smoke t0 call 5: ATS centroid z 가 GT body 원점보다 ~4 cm 위 — body 원점이 bowl 바닥이라 표면 centroid 와 다르다; 비교는 표면 sample 로 해야 한다).
- #3: seed 의 다수는 head camera 가 만든다 (smoke t0 call 5: 142 seed 중 head 101, wrist 41) — wrist 비율은 접근 후반에 오를 것.
- #4: mirror mapping 은 wrist 에서 target 질량을 줄이고 ATS 를 더 자주 실패시킬 것. head 는 차이가 작을 것.
- #5: 비 target 물체 보존율(main) ≈ 0.94–1.0, target-free 에서 plate·bowl_2 는 거의 그대로, target bowl 은 target-free 에서 대부분 지워짐 (smoke t0 call 5 눈대중: 보이는 bowl 점 occupied main 0.99 → tf 0.22, plate 0.99 유지). target ball(반지름 ~74 mm)이 인접 plate 가장자리를 일부 지울 가능성은 남는다.
- #6: 구 중심 ESDF − 정확 거리는 coarse 20 mm 가 답하는 먼 구에서 양수 수 mm–1 cm, fine 이 답하는 가까운 구에서 수 mm 이내.
- #7: v11ab box 에서는 table top(z −0.012)이 box 밖(z < 0)이라 table 표면 sample 의 ESDF 가 크게 나오거나 바깥 답이 될 것; panda box 에서는 occupied.
- #8: mount 점은 self-filter 뒤 거의 남지 않고 (smoke: 보이는 mount 점의 ESDF 는 0.4 = 장 밖), 남아도 gripper 구에서 수십 cm.
- #9: JIT 후 frame p50 ~90 ms, p95 ~120 ms; self-filter mask camera 당 17–19 ms; GPU ~1.7 GiB (process), torch peak ~0.8 GB.
- #10: 회귀 A 73 / B 92 / C 90 + 2 skip, SL0 경로 보존.
