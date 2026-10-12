# SL2 — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-12 14:05 KST
> branch `feature/panda-benchmark`, HEAD `205e7337` (commit 안 함 — lead 가 commit)

## 무엇을 했나 (평이한 요약 먼저)

**SL1 후속 2건.** adapter 가 쓰는 K 의 principal point 를 pixel-centre `((W−1)/2, (H−1)/2)` 로 바꿨고(규약 이름 `pixel_centre`,
`conventions()`·manifest 기록), `robot_state` 를 image 시점 q 로 바꿨다. capture observable 이 같은 substep 에서
`mj_integratePos(qpos, qvel, −dt)` 와 현재 `qpos` 두 후보를 scratch `MjData` 에서 `mj_kinematics` 로 다시 풀어 `data.xpos`(= render 가
보여 주는 kinematics)를 재현하는 쪽을 고른다 (`env.step` 뒤 = 적분 전 qpos, `reset`/`set_init_state` 강제 sample = 현재 qpos).
어느 쪽도 1e-9 m 안에 재현 못 하면 capture 가 예외를 낸다. 기록된 `sim_state` 는 원래 의미(그 substep 의 simulator 상태) 그대로 두고
`sim_state_image` 를 추가했다.

**Panda geometry (V11AB 와 같은 구조).** (1) kinematics: cuRobo `franka_panda.urdf` 의 팔 관절은 LIBERO MuJoCo 와 정확히 같고(0.000 mm)
손·손가락만 robosuite 조립 offset 때문에 어긋나서(hand 0.5 mm, finger 6.5 mm) 그 세 값만 robosuite 값으로 고친 adapter URDF
`robots/panda/description/panda_libero.urdf` 를 만들었다 (mesh 는 두 쪽이 link frame 에서 같으므로 frame offset 문제다). (2) constraint
model = 그 URDF 위에 cuRobo `franka.yml` collision sphere 61 개를 zero-length `UrdfCapsule` 로. (3) self-filter model = 같은 61 개 +
**V11AB 의 `self_filter_covering_capsules` 를 그대로 호출**해서 robosuite collision mesh 에서 맞춘 덮개 capsule 51 개(role `self_filter_cover`,
JSON 으로 commit, provenance·digest 포함). robosuite 모델을 3.11 에서 다시 짓는 방법: py3.8 exporter 가 robot subtree 의 body·joint·geom·mesh
(body frame)를 npz 로 내고, 3.11 쪽이 그것으로 `MjSpec` 모델을 지어 body 이름을 URDF link 이름으로 바꿔 둔다 → V11AB 함수의
`static_urdf_ancestor` 가 RB-Y1 에서처럼 link 를 찾는다. 손가락 개도는 T29 parameter joint. Panda 한 팔 = `left`.

AG3S core(`AG3S/ag3s/**`, `AG3S/trajopt/**`)는 한 줄도 바꾸지 않았다. SL-4/5 에 필요한 주입 지점은 아래 표에 적었다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/adapters/libero_panda/camera.py:81-92` | `K_CONVENTION="pixel_centre"`, `ROBOT_STATE_INSTANT`, `QPOS_IMAGE_*` 코드·허용치 1e-9 m, `MANIPULATOR_SIDE="left"` | 0.1, 0.2, A.5 |
| `camera.py:113,118-123` | `conventions()` 에 `K` 문장 교체 + `K_convention`, `robot_state_instant`, `manipulator_side`(+이유) | manifest 기록 (0.1, A.5) |
| `camera.py:142` | `pixel_centre_intrinsic_matrix` (robosuite `intrinsic_matrix` 는 비교 기준으로 남김) | 0.1 |
| `camera.py:263-336` | observable 이 `[..., qpos_image(nq), source, residual_current, residual_back]` 를 덧붙임; `image_instant_qpos(m, d, scratch)` | 0.2 |
| `camera.py:339-358` | `unpack_capture_state`: `qpos_image`, `qpos_image_source`, `qpos_image_residual_m`, `sim_state_image` | 0.2 |
| `camera.py:386-392, 407-414` | `capture`: 허용치 검사, `robot_state = qpos_image[robot idx]`, `K = pixel_centre`, frame 에 `sim_state_image` 등 | 0.1, 0.2 |
| `benchmark/runners/libero_pi05_client.py:244,260-262,273-275` | `_geom.npz` 에 `sim_state_image` (N,92), `qpos_image_source` (N,), `qpos_image_residual_m` (N,2) | 0.2 (verifier 재현용) |
| `benchmark/runners/launch_libero_pi05_only.py:347-356` | geometry on 이면 manifest `geometry_capture.robot_model = panda.manifest_entry()` (URDF/franka.yml SHA-256, sphere·cover 수, joint map, side, link sets) | A.5 manifest |
| `benchmark/adapters/libero_panda/robot_geometry.py` (새, py3.8) | `export_tables` (robot0_/gripper0_/mount0_ subtree: body pose, joint, camera, site, geom + mesh in body frame), `tables_digest` (base 배치 제외), `fk_poses`/`fk_fixture` (4 고정 q, robot0_base frame), source `libero`(실제 env) / `robosuite`(로봇만, ~1 s), CLI `export`·`fk` | A.3, A.4, B.7 |
| `robots/panda/description/panda_libero.urdf` (새) | kinematics-only URDF, header 에 provenance·보정표·joint map | A.3 |
| `robots/panda/description/libero_panda_self_filter_cover.json` (새, 생성물) | 덮개 capsule 51 개 + provenance (geometry/kinematic digest, export 출처 libero_spatial t0, fitter 상수, URDF·franka.yml SHA-256) | A.4 |
| `robots/panda/description/README.md` (새) | 자산 표, 재생성 명령, joint map | A.3 |
| `AG3S/adapters/panda/__init__.py`, `model.py` (새) | 경로, `JOINT_MAP`, `BODY/SITE/CAMERA_TO_LINK`, link sets + RB-Y1 역할표(docstring + `LINK_SET_TABLE`), `franka_sphere_capsules`, `check_link_set`(오타 → error), `panda_constraint_link_filter`, `build_panda_bare_model`/`build_panda_self_filter_model`/`build_panda_constraint_model`, `split_robot_state`/`set_finger_parameters`, `panda_depth_robot_mask` (V11AB `backproject`+`depth_robot_mask`), `manifest_entry` | A.4–A.6 |
| `AG3S/adapters/panda/cover.py` (새) | export npz → `MjSpec` 모델 (body 이름 = URDF link, primitive 는 삼각화), `fit_self_filter_cover` (body 별로 V11AB `self_filter_covering_capsules(existing=bare)`), 선택적 confined variant (`--confine`), JSON 읽기/쓰기, CLI | A.4 |
| `tests/benchmark/test_panda_sl2.py` (새, 13 test) | 아래 표 | B |
| `tests/benchmark/libero_py38_checks.py:113-131, 169-206, 255-271` | K 검사 = robosuite K − 0.5 px; 새 check `robot_state_is_image_instant`; client episode check 를 image 시점으로 | 0.1, 0.2 |
| `tests/benchmark/test_libero_sl1.py:175-185, 203, 405-406` | pixel-centre K·side 검사, 새 py38 check 등록, manifest 에 `robot_model`/`K_convention` | 0.1, 0.2 |

손대지 않은 것: `AG3S/ag3s/**`, `AG3S/trajopt/**`, `robots/rby1/**`, submodule(cuRobo, LIBERO), `.venv-*`, storage input, 실제 results 디렉터리.

### 결정과 근거

| 항목 | 결정 | 근거 |
|---|---|---|
| URDF | cuRobo URDF 의 팔 관절 그대로 + 3 값 보정 (`panda_joint8` z 0.107→0.1065, `panda_hand_joint` yaw −0.785398→`2·atan2(−0.383, 0.924)` = −0.7858908, finger joint z 0.0584→0.0524). root = `panda_link0` (= `robot0_base` = `robot0_link0`, 같은 pose 확인). `panda_grip_site`·`panda_eye_in_hand`(OpenCV 축) fixed link 추가 | 4 고정 q 에서 수정 전 cuRobo: link0–7 0.000 mm, hand 0.500 mm·0.028°, finger 6.500 mm (hand z 방향). 손·손가락 collision mesh bbox 가 두 쪽 link frame 에서 같음(finger `[-0.0105,-0.0001,0.0001]–[0.0105,0.0264,0.0538]`, hand 동일) → frame offset |
| finger joint 대응 | `robot_state[7]` → `panda_finger_joint1` (+), `robot_state[8]` → `panda_finger_joint2` (**부호 반전**) | MuJoCo `finger_joint2` 는 +y 축·range [−0.04,0], URDF 는 −y 축·[0,0.04]. test 가 FK 로 확인 |
| constraint sphere | `franka.yml` 61 개 (link0 2, link1 4, link2 4, link3 4, link4 4, link5 13, link6 3, link7 5, hand 18, finger 2+2), 반지름 11–60 mm, role `collision`, length 0 | task A.4 |
| 덮개 surface | robosuite **collision** geoms (group 0; collision mesh + primitive pad/mount box·cylinder) | task 문언. test 가 "collision 으로 맞춘 덮개가 visual surface 4 mm 표본 전부를 담는다" 를 모든 body 에 대해 확인 → visual 을 더해도 바뀌는 것이 없다 |
| 덮개 대상 body | Panda link0–7, hand(`gripper0_right_gripper`), 손가락 + pad body, mount 3 개(`controller_box`, `pedestal_feet`, `pedestal`; `torso` 는 visual 뿐) | SL1 segmentation 에서 head 가 `mount0_pedestal_vis`·`mount0_torso_vis` 를 본다. RB-Y1 이 base 를 덮은 것과 같은 이유 |
| confined 규칙 | **기본 꺼짐** (V11AB 함수 그대로). `--confine mount` / body 이름으로 variant 생성 가능 | V11AB 의 `COVER_CONFINED_BODIES=("base",)` 는 module 상수라 Panda body 에 안 걸린다. 아래 "알려진 위험" 의 숫자 |
| 손가락 개도 | T29 parameter joint (`UrdfSphereChain.param_joint_names` = `panda_finger_joint1/2`, `gripper_joint_pairs` 가 `panda_hand` 위 한 쌍으로 인식) | RB-Y1 과 같은 장치 |
| link sets | `PANDA_PALM_LINKS=(panda_hand,)`, `PANDA_FINGER_LINKS=(panda_leftfinger, panda_rightfinger)`, `PANDA_GRIPPER_LINKS`=palm+finger, `PANDA_ARM_LINKS`=link1–7+gripper (link0 제외), contact `{left: gripper}`, `--links` 그룹 `gripper/arms/all` | 아래 표 |
| side | `left` (camera.py `MANIPULATOR_SIDE`, model.py `MANIPULATOR_SIDE`, manifest 두 곳) | task A.5 |

### link 집합 대응 (RB-Y1 ↔ Panda)

| 역할 (RB-Y1 이름, `grounding_report`/`config`) | RB-Y1 | Panda |
|---|---|---|
| `PALM_LINKS` | `ee_left`, `ee_right` | `panda_hand` |
| `FINGER_LINKS` | `ee_finger_{l1,l2,r1,r2}` | `panda_leftfinger`, `panda_rightfinger` |
| `GRIPPER_LINKS` (`--links gripper`, V11AB) | PALM + FINGER (6) | PALM + FINGER (3, 22 sphere) |
| `ARM_LINKS` (`--links arms`) | `link_{left,right}_arm_0..6` + GRIPPER | `panda_link1..7` + GRIPPER (10, 59 sphere) |
| 결정 변수 없는 link (filter 만) | base, wheels, torso, head | `panda_link0`, mount (덮개만) |
| `DEFAULT_CONTACT_LINKS[side]` | left: `ee_left, ee_finger_l1, l2` / right: … | left: `panda_hand, panda_leftfinger, panda_rightfinger` |

### 모델별 sphere 수

| model | 출처 | sphere 수 |
|---|---|---:|
| constraint `gripper` | franka.yml (hand 18 + finger 2 + 2) | 22 |
| constraint `arms` (builder 기본) | franka.yml link1–7 + gripper | 59 |
| constraint `all` | franka.yml 전부 | 61 |
| self-filter | franka.yml 61 + 덮개 51 capsule → 184 sphere (link0 46 (mount 포함), link1 26, link2 10, link3 9, link4 9, link5 9, link6 11, link7 12, hand 11, leftfinger 20, rightfinger 21) | 245 |

### 알려진 위험 (smoke 에서 본 기하 사실 — 측정은 verifier)

- **손바닥 덮개가 손가락 사이로 내려온다.** V11AB fitter 가 hand body 의 덮이지 않은 표면(양 옆면 x=±31.6 mm 등, 10 mm eps 로 한 덩어리)을
  y 방향 slab 3 개로 나눠 capsule 을 맞추고, 각 capsule 축이 hand z 이고 반지름 44–48 mm 라 끝 반구가 손바닥 바닥(hand z 0.066 m) 아래
  hand z 0.112–0.120 m 까지 나온다 — 손가락(z 0.052–0.106) 사이의 파지 자리다. RB-Y1 T17(`link_*_arm_5` capsule 이 사과를 삼킨 건)과
  같은 종류의 기하. confined 규칙으로 hand 를 다시 맞추면 246 capsule / 893 sphere 에 손바닥 아래 13 mm 까지로 줄어든다 (variant, 기본 아님).
- **mount pedestal 덮개가 table 위로 나온다.** pedestal collision cylinder(r 0.18 m) 의 덮개 3 capsule 중 위쪽이 base frame z +0.173 m,
  x +0.394 m 까지 닿는다 (table top = base z −0.012 m). confined 로 맞추면 239 capsule / 1039 sphere, 55 s, 그래도 x +0.219 m.
- 두 variant 모두 sphere 수가 커서(self-filter 245 → 1100+) 기본으로 두지 않았다. 판정은 verifier #4 숫자를 보고 lead·사용자.

### SL-4/5 가 필요로 할 AG3S core 의 RB-Y1 hardcode 지점 (구현하지 않음)

| 위치 | 무엇이 RB-Y1 고정인가 | Panda 에 필요한 것 |
|---|---|---|
| `AG3S/trajopt/serve_safe.py:833-849` (`build_ag3s`) | `MjModel.from_xml_path(model_xml)` → `TransportScene.attach` → `build_robot_model(scene)` / `build_constraint_robot_model(scene, …)` (RB-Y1 URDF + MJCF gap/cover, `HEAD_JOINTS`) | 두 model 을 주입받는 길 (`build_panda_self_filter_model()`, `build_panda_constraint_model(panda_constraint_link_filter(links))`) |
| `serve_safe.py:2074-2087` (`build_to_only_robot_model`) | 같은 builder | 같은 주입 |
| `serve_safe.py:128-145` (`constraint_link_filter`) | `grounding_report.GRIPPER_LINKS/ARM_LINKS` | `panda_constraint_link_filter` |
| `serve_safe.py:864-867` + `AG3S/ag3s/config.py:826-832, 1193` (`DEFAULT_CONTACT_LINKS`, `ContactConfig.contact_links` 기본) | `ee_*` 이름 | config `contact.contact_links = PANDA_CONTACT_LINKS` |
| `AG3S/trajopt/safe_policy.py:46-54` (`DEFAULT_GRASP_LINKS`, `DEFAULT_HOLD_LINKS`) + `serve_safe.py:967-969` (`grasp_parent_links`) | `ee_finger_l1`, `ee_left` … (attached slot 예약) | `{left: ("panda_leftfinger", (panda_leftfinger, panda_rightfinger))}`, hold `{left: panda_hand}` |
| `AG3S/ag3s/runtime/multiview.py:142-186` → `robot_filter.depth_robot_mask` | 관측의 `robot_state` 를 **그대로 model q** 로 씀 (RB-Y1 nq=20) | wire `robot_state` 는 9-D (7 arm + 2 MuJoCo finger) — `split_robot_state` 로 q7 + finger parameter (`set_finger_joints`) 로 나누는 주입, 또는 wire 에 q7 을 싣고 finger 는 따로 |
| `AG3S/ag3s/runtime/pipeline.py:492` `set_finger_joints` 의 입력원: `safe_policy.py:421, 912, 946` (16D state 열 (7,15)), `robot_models/gripper_state.py:177, 260` (`FingerJointMap`, `openings_from_mujoco` RB-Y1 규약) | RB-Y1 gripper 개도 규약 | Panda: `robot_state[7:9]` → `{panda_finger_joint1: +, panda_finger_joint2: −}` |
| `AG3S/trajopt/wire.py:381` `DEFAULT_CAMERAS=("zed_left","wrist_cam_l","wrist_cam_r")`, `:389-408` `ARM_JOINT_DIM=7`, `ACTION_WIDTH=16`, `GRIPPER_COLUMNS=(7,15)` | 3 카메라·16-D joint action | Panda wire 카메라 `head/left_wrist`; Panda action 은 7-D EE delta (OSC_POSE) → SL-5 변환 |
| `AG3S/trajopt/types.py:162` (`ChunkLayout.rby1`) | `[L7, Lg, R7, Rg]` + `DEFAULT_RBY1_JOINTS` 이름 | Panda 를 L 슬롯으로 (`left`) |
| `AG3S/trajopt/config.py:299` (`DEFAULT_LIMITS_MODEL_XML`) | RB-Y1 MJCF 의 관절 한계 | `panda_libero.urdf` limits |
| `AG3S/ag3s/stages/robot_filter.py:33-47` (`inflation_link_groups`) | `self_filter_inflation_by_link` 의 `arms`/`gripper` 그룹이 RB-Y1 link | 그 override 를 쓸 때만 Panda 집합 필요 |
| V11AB server flag `--capsule-radius-scale 0.05 --sphere-spacing 0.4` (`scripts/benchmark/v11ab/start_server.sh:10`) | RB-Y1 gap capsule 을 얇게 하는 손잡이 | franka sphere 는 role `collision` 의 zero-length capsule 이라 **scale 0.05 를 그대로 주면 반지름이 5 % 가 된다** — Panda 에서는 넘기지 않거나 exact role 로 둘지 결정 필요 |
| `--finger-cover both` (`experiments/sources/gripper_cover.py:852` `finger_body_capsules`) | RB-Y1 손가락 inward·palm channel 전제 | Panda 미이식 (franka finger sphere 그대로) |
| `experiments/sources/mujoco_source.py:487` `MJCF_BODY_ALIASES`, `:674` `COVER_CONFINED_BODIES` | RB-Y1 이름 module 상수 | SL2 는 body 이름을 URDF link 로 바꿔 우회 (위 결정표) |

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
OMP_NUM_THREADS=4 .venv-ag3s/bin/python scripts/setup/run_without_legacy.py -m pytest -q -p no:cacheprovider -rs tests/benchmark/test_panda_sl2.py
# 13 passed in 16.65s
OMP_NUM_THREADS=4 bash scripts/testing/run_cpu_regression.sh
# == A: 73 passed, 77 warnings in 37.25s           (SL0/SL1 와 같은 73)
# == B: tests/benchmark (.venv-openpi-live): 58 passed in 67.68s   (SL1 44 + 새 py38 check 1 + SL2 13)
# == C: tests/benchmark (.venv-ag3s): 57 passed, 1 skipped in 41.33s (skip = serve_policy test, flax 없음 — SL1 과 같음)
# == cpu regression exit status: 0
```

| test (`test_panda_sl2.py`) | 무엇을 |
|---|---|
| `test_fk_matches_libero_mujoco_bodies_sites_camera` | `.venv-libero` subprocess(`robot_geometry.py fk --source robosuite`, 4 고정 q, 손가락 개도 포함) vs adapter URDF FK: 11 body + `grip_site` + `eye_in_hand`(OpenCV) = 14 frame × 4 q. 위치 < 1e-9 m, 회전 < 1e-8 rad (실측 libero fixture 로 5.1e-12 m / 1.5e-11 rad) |
| `test_curobo_urdf_differs_only_by_the_documented_hand_offsets` | 팔 관절 7 개 origin·axis·limit 이 cuRobo 와 bit 동일, 보정 3 값, 보정 전 cuRobo 오차 hand 0.5 mm / finger 6.5 mm |
| `test_hand_yaw_is_the_robosuite_quaternion` | hand yaw = robot.xml quat |
| `test_geometry_export_is_the_one_the_committed_cover_was_fitted_from` | robosuite-only export 의 geometry/kinematic digest = cover JSON 에 기록된 libero export digest, URDF·franka.yml SHA-256 일치 |
| `test_cover_refits_to_the_committed_capsules` | 다시 맞추면 commit 된 51 capsule 과 1e-12 안에서 같음 |
| `test_cover_contains_every_collision_and_visual_surface_sample` | self-filter model 이 모든 body 의 collision·visual 표면 표본(4 mm)을 담음 |
| `test_franka_spheres_source_counts_and_radii` | link 별 수, 합 61, 반지름 > 0, 모르는 link 거부 |
| `test_constraint_and_self_filter_models_have_the_v11ab_structure` | `gripper` 22 / `arms` 59 / `all` 61, constraint 에 cover 없음, filter 의 non-cover sphere = constraint sphere (임의 q·개도), finger parameter joint 쌍 |
| `test_robot_state_maps_to_q_and_finger_parameters` | 부호 반전 포함 대응, 손가락 간격 = `rs[7] − rs[8]` (FK) |
| `test_link_sets_are_consistent_and_typos_raise` | contact ⊂ gripper ⊂ arms ⊂ all, link0 ∉ arms, 관절 i 의 child = `panda_link{i}`, RB-Y1 열 = `grounding_report`/`config` 정본, 오타·빈 집합·모르는 그룹 → ValueError |
| `test_panda_depth_robot_mask_removes_robot_points_only` | 합성 depth (손 앞 0.6 m 카메라): 손 pixel 16 개만 mask |
| `test_manifest_entry_and_camera_conventions_agree` | side `left`, joint map 순서 = `ROBOT_STATE_LAYOUT`, 61 / 51 |
| `test_py38_sources_parse_as_python38` | py3.8 파일 3 개 문법 |

`libero_py38_checks.py` (`.venv-libero`): 기존 6 + 새 `robot_state_is_image_instant` 모두 ok. 새 check: 움직이는 action 10 step 에서
FK(기록 `robot_state`) 의 wrist camera pose − 기록 `T_world_cam` = **0.0** (max abs), FK(obs `robot0_joint_pos`) 는 1.9e-3; source 코드
`[0, 1, 1, …]` (초기 sample 은 현재 qpos, step 뒤는 적분 전). client episode check: 14 frame source `[0, 1×13]`, 고른 후보 잔차 0.0, 다른 후보 최소 2.2e-6 m.

**Smoke (policy·GPU 없음, 임시 storage root `scratchpad/storage`, SL0 run 복사본을 source 로)**:

| 실행 | 결과 |
|---|---|
| `launch_libero_replay.py --source-run <SL0> --task-ids 0 --episode-ids 0 --capture-geometry --capture-float-depth --capture-segmentation --run-id sl2-smoke-geom` | exit 0, 46 s; episode npz SHA-256 = SL0; `_geom.npz` 에 `sim_state_image`·`qpos_image_source`(1 × 0, 113 × 1)·`qpos_image_residual_m`; K cx=cy=127.5; 113 frame 에서 `robot_state ≠ sim_state` qpos |
| 위 run 114 frame, adapter URDF FK(`robot_state`) 의 `panda_eye_in_hand` vs 기록 `T_base_cam/left_wrist` | 위치 max 4.4e-12 m, 회전 max 9.9e-12 rad (FK(`sim_state` qpos) 는 5.7e-4 m) |
| `robot_geometry.py export/fk --source libero` vs `--source robosuite` | kinematic·geometry digest 동일, FK fixture 차 3.9e-16 |
| `python -m AG3S.adapters.panda.cover --geometry <libero export>` | 51 capsule, 5 s |

결과는 scratchpad 에만 있다. self-filter 제거율·오제거·coverage·contact 는 verifier 몫이라 여기 적지 않는다.

## verifier 가 알아야 할 것

- **실행 명령** (GPU·server 불필요, 각 1 process):
  ```bash
  cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
  SRC=20261012T045129_6ba81024d243_sl0-pi05only-t0e0
  # 새 replay (image 시점 robot_state, pixel-centre K, sim_state_image 포함)
  .venv-ag3s/bin/python benchmark/runners/launch_libero_replay.py --source-run $SRC --task-ids 0 --episode-ids 0 \
      --capture-geometry --capture-float-depth --capture-segmentation --run-id sl2-replay-seg-t0e0
  .venv-ag3s/bin/python benchmark/runners/launch_libero_replay.py --source-run $SRC --task-ids 0 --episode-ids 0 \
      --run-id sl2-replay-off-t0e0                                   # #6 SL0 경로 보존
  # robot geometry ground truth (.venv-libero; MUJOCO_GL=osmesa, PYTHONPATH 제거)
  env -u PYTHONPATH MUJOCO_GL=osmesa .venv-libero/bin/python benchmark/adapters/libero_panda/robot_geometry.py \
      export --source libero --out outputs/verify/SL2/raw/geo_libero.npz
  env -u PYTHONPATH MUJOCO_GL=osmesa .venv-libero/bin/python benchmark/adapters/libero_panda/robot_geometry.py \
      fk --source libero --out outputs/verify/SL2/raw/fk_libero.npz
  bash scripts/testing/run_cpu_regression.sh
  ```
- **Python 쪽 API** (`.venv-ag3s`, `PYTHONPATH=$AG3S_PROJECT_ROOT`):
  `from AG3S.adapters.panda import model as pm, cover` →
  `pm.build_panda_self_filter_model()`, `pm.build_panda_constraint_model(pm.panda_constraint_link_filter("gripper"|"arms"|"all"))`,
  `q7 = pm.set_finger_parameters(model, robot_state9)` 후 `model.sphere_centers_numeric(q7)` / `model.link_pose(q7, link)`,
  `pm.panda_depth_robot_mask(depth_m, K, T_base_cam, robot_state9, model, PointCloudConfig())` → V11AB `DepthRobotMask`.
  sphere 의 link·role: `model.sphere_link_names`, `model.sphere_roles` (`collision` / `self_filter_cover`).
  mesh 표면: `cover.build_mujoco_model(cover.load_geometry(geo_libero.npz), ("collision",)|("visual",))` → MuJoCo model (body 이름 = URDF link
  또는 robosuite 이름) + `AG3S.ag3s.experiments.sources.mujoco_source._body_surface_points` / `static_urdf_ancestor` (test 가 쓰는 방식).
  MuJoCo body ↔ URDF link 표: `pm.BODY_TO_LINK`, `pm.SITE_TO_LINK`, `pm.CAMERA_TO_LINK`.
- **FK 비교 frame**: 모두 `robot0_base` (= `panda_link0`) frame. 기록 `T_world_base` 로 world 와 오간다. 손가락은 parameter joint 라
  `link_pose(q7, link)` 전에 반드시 `set_finger_parameters` (또는 `params=`).
- **image 시점 상태**: 물체·로봇 pose 를 image 와 맞추려면 `sim_state_image` 로 `set_state_from_flattened` + `forward` (SL1 의 `sim_state` 는
  1 substep 늦은 kinematics 였다). segmentation 은 여전히 `env.step` return 시점 render 다 (SL1 차이 #4 그대로).
- 새 플래그·기본값: `cover.py --confine BODY…|mount` (variant 생성용, 기본 없음). 기존 플래그 기본값 변경 없음.
- 바뀐 기록 형식: `_geom.npz` 에 key 3 개 추가; capture observable 벡터 길이 +nq+3 (`unpack_capture_state` 가 새 길이만 받는다 — 옛 run 의
  observable 원 벡터는 저장된 적이 없으므로 영향 없음); `K/<cam>` 값이 −0.5 px; `robot_state/<cam>` 값이 image 시점 (SL1 run 들과 다르다);
  manifest `geometry_capture.conventions` 에 `K_convention`·`robot_state_instant`·`manipulator_side`, `geometry_capture.robot_model` 추가.
  settings `step` 은 geometry on 이면 여전히 `"SL1"` (flag 를 만든 step 이름, 바꾸지 않음).
- 재생산이 필요한 산출물: SL1 run 의 `K`/`robot_state` 를 쓰는 측정은 새 replay 로. cover JSON 은 commit 된 것 그대로 (재생성 명령은 README).
- 옛 기록과 호환: SL0/SL1 episode npz 형식 그대로. 옛 `_geom.npz` 는 새 key 가 없을 뿐 읽힌다.
- 자원: 측정 중 다른 session process 를 건드리지 않았다. cover fitting 은 scipy 가 thread 를 많이 쓰므로 `OMP_NUM_THREADS=4` 권장.

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->
- #1 (a) image 시점 robot_state 로 forward 한 wrist pose − `T_world_cam` ≈ 1e-12 이하. (b) pixel-centre K 로 SL1 #2·#4 의 head table 오차가
  SL1 `K_pixel_centre` 열(median 0.9 mm 대)과 같을 것 — 새 run 은 K 만 바뀌고 depth 는 같다.
- #2 FK: 114 frame 전부 1e-9 mm 수준 (float). 초과 link 없음.
- #3 coverage: franka.yml sphere 만으로는 robosuite collision mesh 를 잘 못 덮는다 — 구현 중 본 bare 모델 값(4 mm 표본, 안쪽 비율):
  link0 3.5 %, link1 76 %, link2 23 %, link3 33 %, link4 24 %, link5 14 %, link6 7 %, link7 38 %, hand 8.7 %, finger 14.5 %, 밖 최대 90 mm(link6),
  손가락 14 mm. self-filter model(덮개 포함)은 모든 body 100 %. 덮개가 mesh 밖으로 나간 두께는 hand 에서 수 cm(손바닥 아래 54 mm).
  → "franka.yml sphere coverage 가 크게 다르면" 의 되돌아올 지점에 걸릴 가능성이 높다.
- #4 self-filter: robot pixel 제거율 head ≈ 99 %, wrist ≈ 99.9 %. 오제거: head 는 table(mount pedestal 덮개, base 근처) 프레임당 1–2 k px,
  wrist 는 **grasp 닫히는 구간(대략 frame 49–56)에 target bowl 의 1/4–1/3** 이 hand 덮개(가운데 capsule) 에 지워질 것 — T17 과 같은 원인(덮개
  capsule 이 손가락 사이를 삼킴)이고 finger parameter 문제가 아니다.
- #5 contact: 실제 접촉 frame 에서 finger/hand sphere 가 bowl 에 침투로 나타날 것. 비접촉 frame clearance 는 finger sphere 가 mesh 보다
  작은 곳(손가락 끝)에서 실제보다 크게(낙관적으로) 나올 수 있다.
- #6 회귀: off replay = SL0 bit 동일 (flag off 경로는 바뀐 코드가 없다), legacy 16D 기준선은 SL1 과 같은 값 (AG3S core 무변경).
