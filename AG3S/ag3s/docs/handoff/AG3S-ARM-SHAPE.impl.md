# AG3S-ARM-SHAPE — 팔 전체 구 모델 (내접 덮개 · MorphIt) 구현 · 기하 지표 · 서버 flag

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe
> 작성 2026-10-27 18:15 KST (`date -u` 09:15 UTC + 9 h). Task: [`AG3S-ARM-SHAPE.task.md`](AG3S-ARM-SHAPE.task.md) §5 첫 줄 (구현).
> 코드는 사본 `$RUN/snapshot_arm` (= `snapshot_scpt` 의 rsync 사본, `__pycache__` 제외) 에서만 고쳤다. main checkout 코드는 건드리지 않음.
> patch (`snapshot_scpt` 대비): `$RUN/arm_shape.patch`.
> `RUN = /mnt/dev/AG3S_VLA-storage/results/development/20261027T165001+0900_b3b69311f948_430762a2a2fc`. 원시 출력: `$RUN/verify/AG3S-ARM-SHAPE/fit/` · `…/tests/`.

## 0. 요약 (평이하게)

- **무엇을 했나:** 양팔 `link_*_arm_0` … `link_*_arm_6` 과 `FT_sensor_*` (16 link) 의 구 모델을 MuJoCo 가 실제로 충돌시키는 MJCF collision geom 에 두 방법으로 맞춰 파일로 썼다 — **내접 덮개** (gripper_cover 의 후보 · set cover 를 팔에 확장, 구가 mesh 밖으로 나가지 않음) 와 **MorphIt** (cuRobo `fit_spheres_to_mesh(MORPHIT)`). 서버는 `--arm-spheres PATH` (기본 꺼짐) 로 그 파일의 구를 제약 모델 **맨 뒤에** 붙인다. 손 (손가락 · 손바닥 · 껍질 구) 606 구는 그대로이고 비트로 같다.
- **매개변수는 기하로만 골랐다:** link 마다 "0.5 mm 표면 표본이 구 밖에 남는 거리 max ≤ 5 mm, 손 keep-out 겹침 0" 을 만족하는 가장 적은 구. GT · 자세 평가는 보지 않았다.
- **꺼짐은 그대로:** flag 를 안 주면 모델 (이름 · 중심 · 반지름 · role · FK) · config dump · 기록 키 · client metadata 가 `snapshot_scpt` 와 같고, 기록 chunk 다시 풀기 SHA 가 SC golden (`dcc931c0ed740d85` · `e6f6986f5730bfc8`) 과 같다 (두 venv).
- **결과 (내접 덮개):** 16 link 모두 통과. 팔 구 **2,607** 개 (서버 모델 606 → 3,213). 표면이 구 밖에 남는 거리 max 4.62–4.99 mm, mesh 밖 돌출 ≤ 0.42 mm, 손 겹침 0 (FT sensor 는 keep-out 전 26 · 22 구가 손 껍질에 최대 4.9 mm 들어갔다 → 0).
- **MorphIt:** §5.2 (GPU lock 대기 — 채움).
- **알아둘 것 (gap):** (1) SCMR flag 에서 팔 행의 최근접 표면이 target 이면 여유는 `--obstacle-margin` 10 mm 가 아니라 `geometry.safety_margin` **50 mm** 다 (기존 비권한 행 규칙). (2) cuRobo GPU 초기값 world 에는 그 per-link target 여유가 없다 (기존 `WORLD_GAPS`). (3) 쥔 뒤에는 `held_permission_parent_links = 1` 규칙으로 `link_*_arm_6` 행이 쥔 물체에 닿아도 되는 행이 된다 (기존 규칙, 지금까지는 arm_6 구가 없어 작동 안 함). §4.

## 1. 설계

### 1.1 형상 — MJCF collision geom 을 URDF link frame 으로

- 대상 = 각 팔 body 의 collision mesh geom (contype/conaffinity ≠ 0, 전부 mesh, MuJoCo 는 조각마다 볼록 껍질로 충돌). arm link 은 MJCF body 이름 = URDF link 이름, FT sensor 는 `FT_SENSOR_<S>` → URDF `FT_sensor_<S>` (`gripper_cover.WRIST_MJCF_BODY`).
- 변환 = `gripper_cover._mjcf_body_in_link` (손목 덮개와 같은 것; arm link 은 항등, FT 는 URDF fixed `tool_*`). **검증:** 무작위 관절값 8 개 (0 자세 포함) × 16 link 에서 `URDF FK(link) @ T_link_body` 대 MuJoCo body pose (둘 다 `link_torso_5` 기준) 차이 최대 **0.00056 mm · 0.00005°** (`fit/frames_check.json`, `fit/check_frames.py`).
- 겉면 = `gripper_cover.body_collision_surface` (같은 body 다른 조각에 묻힌 표본 제거, 강체 이웃 body 에 묻힌 면은 `hidden` 표시). 덮을 대상 = 같은 body 조각에 맞붙은 면 (`finger_body_targets`) 도 `hidden` 도 아닌 것. `hidden` 은 `arm_6` (FT · 손바닥 아래) 과 FT (손바닥 위) 에만 있다 (7,614 · 13,414 표본 @1 mm, 왼쪽).

### 1.2 방법 A — 내접 덮개 (`arm_cover.inscribed_spheres`)

- 후보 = mesh 안 2 mm 격자점 (`gripper_cover._finger_candidates`). 반지름 = 그 점의 **깊이** — `max(볼록 조각 정확 깊이, 겉면 표본 거리 − 1 mm)`, 둘 다 진짜 깊이의 하한이라 구는 mesh 안이다 (안쪽에서 겉면에 접하는 가장 큰 구; overflow 0). 손목 덮개 (T46f A14) 와 같은 격자 · union_eps · 표본 간격 (1 mm).
- 덮음 = 겉면 표본이 구 표면에서 `max_gap` 안이면 덮인 것 (`pad`). 대상 전부를 덮는 구를 **lazy greedy set cover** 로 고른다 (`_greedy_exact_cover` 와 같은 규칙 · 동률이면 작은 반지름 · 끝에 중복 제거; 팔 link 은 후보 3–20 만 개라 후보마다 덮는 집합을 들고 있을 수 없어 lazy 로). ILP 는 쓰지 않았다 (후보 수가 손가락의 수십 배).
- `max_gap` 고르기: link 마다 5.00 → 4.75 → 4.50 → 4.25 … mm 순서로 시험해 **0.5 mm 판정 표본** 에서 기준을 처음 넘는 값 (= 가장 적은 구). 1 mm 표본에서 맞춘 것이 0.5 mm 표본에서는 0.2–0.4 mm 더 나와 5.00 은 늘 떨어졌다.

### 1.3 방법 B — MorphIt (`arm_cover.morphit_spheres`)

- 입력 mesh = 그 link 의 collision 조각 볼록 껍질들의 합집합 (`trimesh.boolean.union`, manifold; 닫힌 mesh 확인). `fit_spheres_to_mesh(mesh, num_spheres=n, fit_type=MORPHIT, iterations=200, compute_metrics=True)`, 손실 가중치 기본 (coverage 1000 · protrusion 10), seed 0 (torch · numpy). cuRobo 자체 지표 (`metrics`) 는 파일에 같이 싣는다.
- 구 수 고르기: 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512 순서로 시험해 기준을 처음 넘는 수를 찾고, 바로 앞 (떨어진) 수와 사이를 이분 탐색해 가장 적은 수. 512 에서도 못 넘으면 그 link 은 FAILED 로 적고 서버용 이름으로 쓰지 않는다 (판정을 느슨하게 하지 않음).
- CPU 실행은 불가: cuRobo `WarpMeshQuery` 가 CUDA 전용이라 CPU device 에서 segfault (`fit/morphit_trial.py`). GPU 는 `gpu_lock.py shared --est-gib 4 --tag arm-shape-fit` 로 (`fit/run_gpu.sh`).

### 1.4 같은 손 keep-out 규칙 (두 방법 공통, `arm_cover.arm_keep_out` · `apply_keep_out`)

| link | 손과의 관계 | keep-out 영역 |
|---|---|---|
| `link_*_arm_6`, `FT_sensor_*` | 손바닥과 사이 joint 가 fixed 뿐 | 권한 손 (`EE_BODY_*` + `ee_finger_*`, 손가락은 prismatic 이동 전 구간 sweep) 의 볼록 껍질들 (`gripper_cover.hand_keep_out`, T46f A14 와 같은 영역) |
| `link_*_arm_5` | 맨 위 revolute 하나 (`*_arm_6`) | 그 축 방향으로 손의 최대값 아래 반공간 (A14 와 같음) |
| `link_*_arm_0` … `arm_4` | revolute 2 개 이상 — 손 자리가 자세마다 다름 | 없음. 보고만: 사이 revolute 를 URDF 한계 안 9^k 격자로 돌린 구–손 껍질 거리 하한의 최소 = arm_4 **90.0 · 90.8 mm**, arm_3 **78.8 · 79.4 mm** (왼 · 오) — 손이 이 link 근처로 오지 않는다 |

- 규칙: 반지름 ≤ `keep_out_clearance(c)` (지지 평면 거리 — 표본이 아닌 정확한 보장), 잘린 반지름 < 0.5 mm 인 구는 버린다. 내접 덮개는 이 상한을 **후보에** 걸고 (greedy 가 알고 고름), MorphIt 는 끝난 구에 같은 상한을 건다. 두 방법 모두 "keep-out 없이 맞춘 결과" 의 겹침 (전) 과 적용 뒤 (후) 를 파일 `geometry_metrics[link].keep_out_rule` 에 싣는다.

### 1.5 기하 지표 (`arm_cover.link_geometry_metrics`, 자세 무관)

- 표면 → 구 거리: 0.5 mm 표면 표본 (맞출 때 쓴 1 mm 와 다른 표본) 에서 `max(0, min_s |p − c_s| − r_s)`. **판정 = `reachable`** (같은 body 조각끼리 맞붙은 면 · 강체 이웃에 묻힌 면 제외), `all` (숨은 면 포함) 은 보고.
- 돌출: 구마다 Fibonacci 2,000 방향 표면 점 중 mesh (볼록 조각 반공간으로 정확 판정) 밖인 점의 가장 가까운 0.5 mm 표면 표본까지 거리 (오차 ≤ 약 0.4 mm).
- 손 겹침: keep-out 이 있는 link 의 `r − keep_out_clearance(c) > 0` 구 수 · 최대 (보수적).

### 1.6 서버 — 제약 모델 맨 뒤에 붙인다

- `build_constraint_robot_model(..., arm_spheres=None)`: 꺼지면 예전 함수 (`_build_constraint_robot_model`, 이름만 바꿈) 의 모델을 **그대로** 돌려준다. 켜면 `_append_arm_spheres` 가 파일을 읽고 (`arm_cover.load_arm_spheres`: 최상위 키 · method · link 이름 · 길이 · 유한 · r > 0 검사, sha256) 구를 길이 0 `UrdfCapsule(role="arm_cover")` 로 만들어, URDF capsule 은 link_filter 안의 것만 남긴 URDF 위에 `기존 extra capsule + 팔 구` 를 넓힌 link_filter 로 다시 짓는다 — 팔 link 의 URDF capsule 은 들어가지 않는다. 지은 뒤 앞 606 구의 이름 · 중심 · 반지름 · role 이 꺼진 모델과 비트로 같은지 확인하고 다르면 RuntimeError.
- role `arm_cover` 는 `EXACT_SPHERE_ROLES` (반지름 그대로, capsule_radius_scale 0.05 를 받지 않음).
- 비권한: 팔 link 은 `contact.contact_links` 에 없다 → `target_field_exclude` (target-free 층) False, `target_contact_rows` False (strict), `manipulated_link_margin` = full. 즉 target 은 장애물이다.
- 소비자마다:

| 소비자 | 팔 구가 보이나 | 근거 |
|---|---|---|
| `CollisionLinearizer` (CPU TrajOpt) | 예 — 같은 `UrdfSphereChain` 하나 (`safe_policy` 는 `ag3s.constraint_robot_model` 을 씀) | readiness: linearizer 구 3,213; E4 기록 다시 계산에서 앞 606 열 행 값 · 분류 비트 동일 (test) |
| 기록 `sphere_centres/sphere_radii/sphere_link_names` · `clearance` | 예 — `linearizer.robot_model` 에서 씀 | 코드 (`safe_policy.py:1962-1963`) |
| `--record-constraints` meta | 켰을 때만 `arm_spheres` 키 (path · sha256 · method · link 별 수) | `serve_safe.arm_spheres_file_meta` |
| client metadata | 켰을 때만 `arm_spheres` 키 (sha256 · method · 구 수); `constraint_spheres` 값은 구 수 | test · readiness (차이 키 = 이 둘) |
| cuRobo GPU 초기값 (`--to-gpu-seeds curobo`) | 예 — `gpu_seeds.robot_spec` 이 live 모델의 `_local_spheres` 를 그대로 복사, B0m world 는 구 수로 짓는다 | §5.3 GPU FK 검사 (채움) |
| `curobo_v2/export_inputs` (T63 offline export) | 아니오 — 명령줄에서 모델을 다시 짓는 도구라 `--arm-spheres` 를 지원 안 하는 flag 목록에 넣어 거절 | 조용히 다른 모델로 export 하지 않게 |
| `ta_offline.build_robot` (offline 다시 풀기) | 예 — 명령줄의 `--arm-spheres` 를 읽는다 | test |
| AG3S `pipeline` 의 "구가 coarse 격자 밖이면 DEGRADED" | 밖으로 안 나감 — E4 1345 기록 32 chunk × (현재 + 계획 16 step) 에서 팔 구의 coarse 격자 여유 최소 0.645 m | `fit/readiness_inscribed.json` |

## 2. 바뀐 파일 (줄 = `snapshot_arm` 기준)

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/experiments/sources/arm_cover.py` (새, 642 줄) | 팔 link 표 · 겉면 · keep-out 규칙 (`arm_keep_out:127` · `apply_keep_out:140`), 내접 덮개 (`_lazy_greedy_cover:162` · `inscribed_spheres:203`), MorphIt (`union_mesh:239` · `morphit_spheres:252`), 기하 지표 (`link_geometry_metrics:295` · `hand_sweep_clearance:353`), 파일 (`load_arm_spheres:400` · `arm_sphere_capsules:434`), offline 맞춤 CLI (`fit_inscribed:467` · `fit_morphit:530` · `main:612`) | §1 |
| `AG3S/ag3s/robot_models/urdf_sphere_chain.py:247-255` | `ARM_COVER_ROLE = "arm_cover"`, `EXACT_SPHERE_ROLES` 에 포함 | 파일 반지름 그대로 (배율 · inflation 없음) |
| `AG3S/ag3s/experiments/reports/grounding_report.py:233-315` | `build_constraint_robot_model(arm_spheres=None)` = 예전 함수 + `_append_arm_spheres` (`:258`), `_MODEL_REPORTS` 옮김 | §1.6 |
| `AG3S/ag3s/experiments/reports/grounding_report.py:318` | 예전 본문의 이름을 `_build_constraint_robot_model` 로 (본문 그대로) | 꺼짐 = 예전 호출 그대로 |
| `AG3S/trajopt/serve_safe.py:759-831` | `arm_spheres_options` · `check_arm_spheres` (시작 거절) · `arm_spheres_file_meta` · `arm_spheres_meta` · `announce_arm_spheres` | CLI → 모델 · 기록 |
| `AG3S/trajopt/serve_safe.py:1253-1254, 1293, 1368-1370, 1457` | `ag3s_from_args` → `build_ag3s(arm_spheres=)` (꺼지면 키 없음) → `build_constraint_robot_model` · 시작 WARNING | 배선 |
| `AG3S/trajopt/serve_safe.py:2240-2246` | CLI `--arm-spheres PATH` (기본 None) | |
| `AG3S/trajopt/serve_safe.py:2996-3000` | `reject_bad_flag_combinations` 에서 `check_arm_spheres`: `--links` 가 `gripper` 가 아님 · `--no-safe` · `--no-perception` · `--gripper-cover` · `--exclude-links <팔 link>` · 읽을 수 없는 / 형식이 틀린 파일 → 시작 거절 | 체크포인트를 올리기 전에 |
| `AG3S/trajopt/serve_safe.py:3723-3725` | `--record-constraints` meta 에 `arm_spheres` (켰을 때만) | |
| `AG3S/trajopt/safe_policy.py:390-394` | client metadata `arm_spheres` (켰을 때만) | |
| `AG3S/trajopt/experiments/ta_offline.py:273-275` | `build_robot` 이 명령줄의 `--arm-spheres` 를 읽는다 | offline 다시 풀기가 서버와 같은 모델 |
| `AG3S/trajopt/curobo_v2/export_inputs.py:61` | `--arm-spheres` 를 재구성 못 하는 flag 목록에 | 조용히 다른 모델로 export 하지 않게 |
| `tests/trajopt/test_arm_shape.py` (새, 17 test) | §3 | |

## 3. 단위 검증

```bash
# cwd = $RUN/snapshot_arm, env = $RUN/verify/AG3S-ARM-SHAPE/fit/env.sh (PYTHONPATH = snapshot_arm 먼저)
# AG3S.trajopt.serve_safe.__file__ = $RUN/snapshot_arm/AG3S/trajopt/serve_safe.py 확인 (두 venv; test_module_is_the_snapshot_code 도 확인)
.venv-openpi-live/bin/python -m pytest -q tests/trajopt/test_arm_shape.py                    # (채움)
.venv-ag3s/bin/python -m pytest -q tests/trajopt/test_arm_shape.py tests/trajopt/test_scm_palm_shell_margin.py   # 27 passed, 2 skipped (MorphIt 파일 전)
bash $RUN/verify/AG3S-ARM-SHAPE/tests/run_files.sh "" "" 1   # tests/trajopt + tests/ag3s, 파일마다 별도 프로세스 (채움)
```

test 가 고정하는 것: (1) 꺼짐 — `arm_spheres=None/False/""` 이 인자 없는 모델과 비트 동일 (606 구), CLI 기본 · SCMR 명령줄 → `build_ag3s` · `ta_offline.build_robot` 에 키 없음, config dict · `sqp_overrides` 동일, client metadata 키 없음, 기록 chunk (E4 1345 seq 12 · 13) 다시 풀기 SHA = SC golden. (2) 켜짐 — 앞 606 구 비트 동일, 구 수 = 606 + 파일 구 수, 이름 · role `arm_cover` · 반지름 그대로, FK = link pose @ 파일 중심 (1e-12), 팔 link URDF capsule 은 안 들어옴, `{"path":…}` = 경로, E4 기록 TrajOpt 행 값 · 분류의 앞 606 열 비트 동일 · 팔 열 유한 · 팔 행 target-free 층 안 읽음. (3) 기하 — 합성 상자에서 `inscribed_spheres` 가 max_gap · mesh 안 (정확) 을 지킴, `apply_keep_out` 의 상한 · 버림, link 별 keep-out 규칙, 두 파일 (있으면) 의 형식 · 16 link · gap ≤ 5 mm · 겹침 0 (정확한 평면 거리로 다시) · 내접은 돌출 ≤ 1 mm · 두 link 다시 재면 파일 값과 같음, 파일로 지은 서버 모델의 구 수 · 앞부분. (4) 거절 — 파일 형식 6 경우 + 키 누락, `link_filter=None` · `ARM_LINKS` (URDF 팔 capsule 있음) · 모르는 키, 서버: `--links arms/all/gripper_wrist` · `--no-safe` · `--gripper-cover` · `--exclude-links link_left_arm_3` · 없는 파일 · 형식 틀린 파일.

## 4. verifier · lead 가 알아야 할 것

- **새 flag:** `--arm-spheres PATH` (기본 None = 예전 모델). 기본값 변경 없음. 기록 키는 켰을 때만 (`--record-constraints` meta `arm_spheres`, client metadata `arm_spheres`). 시작 WARNING 한 줄 (구 수 · link 별 수 · sha256 앞 16).
- **구 순서:** 팔 구는 모델 **맨 뒤** (index 606 부터), link 은 `left arm_0…6, FT_sensor_L, right arm_0…6, FT_sensor_R` 순, link 안은 파일 순서. npz 의 `sphere_link_names` 로 열을 고를 것.
- **target 여유 (gap 1):** SCMR flag 에는 `--nonauthorized-target-margin` 이 없다 (= AG3S `contact.nonauthorized_target_margin = full`). 그래서 팔 행의 최근접 표면이 target 으로 판정되면 (`|d − d_object| ≤ voxel`) 그 행 여유 = `manipulated_link_margin` = `geometry.safety_margin` **50 mm** 이고, strict 분류에서는 obstacle 이라 `max(50, 10)` = 50 mm 다 (`linearize.py:991-1023`). lead 가 적은 "normal --obstacle-margin applies" 와 다르다. 반대손 (오른손, 왼손 조작 때) 행이 이미 이 규칙이다. E4 1345 기록 (32 chunk × 16 step) 에서 이 50 mm 행 때문에만 생긴 팔 위반은 **0** (팔 행 최소 값: FT_L +15.1 mm, arm_6_L +39.4 mm; 50 mm 행 수 FT_L 7,740 / 74,752) (`fit/target_margin_rows_inscribed.json`). 10 mm 로 하려면 `--nonauthorized-target-margin obstacle` 이 있지만 반대손 행도 같이 바뀐다 — 판단은 lead.
- **GPU 초기값 (gap 2):** cuRobo B0m world 에는 per-link target 여유 (`manipulated_link_margin`) 가 없다 (`gpu_seeds.WORLD_GAPS` 기존 항목) — 팔 행이 target 옆이면 GPU 는 10 mm, CPU 검사는 50 mm. 고치지 않았다 (SCM 의 껍질 여유와 같은 종류의 차이).
- **쥔 뒤 (gap 3):** `held_contact_depth` 는 쥔 손의 hold 사슬에서 **구가 있는** link 을 센다. 지금 (`--links gripper`) 은 팔 link 에 구가 없어 깊이 1 이 비어 있었는데, 팔 구가 있으면 `link_<side>_arm_6` 이 깊이 1 이 되어 `held_permission_parent_links = 1` (사용자 판정 2026-10-10) 에 따라 **쥔 물체에 대해서만** 권한 행이 된다. FT sensor 는 `PROTECTED_LINKS` 라 제외. `arm_5` 이상은 권한 없음. 이 규칙은 바꾸지 않았다.
- **구 수:** 내접 덮개 2,607 (서버 3,213) — TrajOpt 질의 행 · FK 가 5.3 배. **풀이 시간은 재지 않았다** (독점 lock 필요, task 에서 보고만).
- **거리장:** 팔 구 대부분은 손 둘레의 5 mm 미세 창 밖이라 coarse 20 mm 층을 읽는다 (거리장 오차는 verifier 의 측정 항목).
- **내접 덮개 파일을 만든 프로세스:** fit 도중에 `arm_cover.py` 의 실패 처리 (판정을 못 넘은 link 을 FAILED 로 적고 계속, 서버용 이름으로 쓰지 않음) 와 MorphIt 구 수 목록을 고쳤다. 내접 덮개는 모든 link 이 통과해 그 분기를 타지 않았으므로 파일 내용은 지금 코드와 같다.
- 재생산할 산출물 없음. 옛 기록과의 호환 깨짐 없음.

## 5. 결과 — 파일과 기하 지표

### 5.1 내접 덮개 — `$RUN/arm_spheres/inscribed.json`

- sha256 `07841afcebb22a11c5477a9aa5f9ac465b8277c873a2ccb61a7def85c4da2342`. 매개변수: 격자 2 mm · union_eps 1 mm · 맞춤 표본 1 mm · 판정 표본 0.5 mm · link 별 `max_gap` (아래). 실행 `fit/run_inscribed.sh` (CPU 1 프로세스, 44 분), log `fit/inscribed_fit.log`.

| link | 구 수 | max_gap (mm) | 표면→구 max / p95 (판정 면) | max (숨은 면 포함) | 돌출 max / p95 | 손 겹침 (keep-out 전 → 후) | 반지름 (mm) |
|---|---|---|---|---|---|---|---|
| link_left_arm_0 | 271 | 4.50 | 4.85 / 3.76 | 4.85 | 0.42 / 0.36 | keep-out 없음 | 0.5–16.3 |
| link_left_arm_1 | 165 | 4.50 | 4.76 / 3.56 | 4.76 | 0.22 / 0.22 | keep-out 없음 | 0.5–44.3 |
| link_left_arm_2 | 269 | 4.50 | 4.77 / 3.71 | 4.77 | 0.36 / 0.36 | keep-out 없음 | 0.5–45.3 |
| link_left_arm_3 | 95 | 4.75 | 4.95 / 3.99 | 4.95 | 0.28 / 0.27 | keep-out 없음 | 0.5–38.8 |
| link_left_arm_4 | 149 | 4.50 | 4.80 / 3.72 | 4.80 | 0.34 / 0.32 | keep-out 없음 | 0.5–14.7 |
| link_left_arm_5 | 137 | 4.50 | 4.62 / 3.69 | 4.62 | 0.34 / 0.33 | 0 → 0 | 0.5–32.1 |
| link_left_arm_6 | 101 | 4.50 | 4.79 / 3.72 | 8.89 | 0.31 / 0.29 | 0 → 0 | 0.5–13.0 |
| FT_sensor_L | 146 | 4.25 | 4.83 / 3.60 | 8.46 | 0.41 / 0.31 | 26 구 (최대 4.89 mm, 137 구 중) → 0 | 0.5–16.6 |
| link_right_arm_0 | 275 | 4.50 | 4.82 / 3.77 | 4.82 | 0.40 / 0.40 | keep-out 없음 | 0.5–15.2 |
| link_right_arm_1 | 172 | 4.50 | 4.80 / 3.60 | 4.80 | 0.40 / 0.36 | keep-out 없음 | 0.5–44.3 |
| link_right_arm_2 | 233 | 4.50 | 4.77 / 3.69 | 4.77 | 0.33 / 0.33 | keep-out 없음 | 0.5–45.1 |
| link_right_arm_3 | 96 | 4.75 | 4.89 / 4.01 | 4.89 | 0.39 / 0.37 | keep-out 없음 | 0.5–39.4 |
| link_right_arm_4 | 136 | 4.75 | 4.99 / 3.91 | 4.99 | 0.13 / 0.13 | keep-out 없음 | 0.5–15.0 |
| link_right_arm_5 | 131 | 4.75 | 4.91 / 3.90 | 4.91 | 0.00 / 0.00 | 0 → 0 | 0.5–31.7 |
| link_right_arm_6 | 98 | 4.50 | 4.97 / 3.72 | 9.81 | 0.39 / 0.34 | 0 → 0 | 0.5–13.0 |
| FT_sensor_R | 133 | 4.50 | 4.77 / 3.73 | 9.12 | 0.41 / 0.37 | 22 구 (최대 3.82 mm, 128 구 중) → 0 | 0.5–16.6 |
| **합** | **2,607** | | **max 4.99** | | **max 0.42** | **0** | |

- 숨은 면 포함 max 가 5 mm 를 넘는 곳은 `arm_6` · FT 의 강체 이웃 (FT sensor · 손바닥) 에 묻힌 면뿐이다 (아무것도 닿을 수 없음, T46f A14 와 같은 정의).
- 구가 많은 이유: 볼록 껍질 조각의 날카로운 꼭짓점 · 모서리는 안쪽에서 접하는 구로 5 mm 안에 덮으려면 작은 구가 필요하다 (최소 반지름 0.5 mm 구가 link 마다 있다). 팔 몸통 (arm_1 · 2 · 3) 의 가장 큰 구는 38–45 mm.

### 5.2 MorphIt — `$RUN/arm_spheres/morphit.json`

(GPU lock 대기 — 채움)

### 5.3 cuRobo GPU 초기값 경로 검사

(GPU lock 대기 — 채움)

## 6. closed-loop 준비 확인 (서버 없이) — `fit/readiness.py` → `fit/readiness_inscribed.json`

SCMR 서버 명령줄 (`launch_occ_SCMR.sh` 의 BASE + SAFE + `--record-constraints`, `--model-xml` = snapshot_arm) 과 + `--arm-spheres $RUN/arm_spheres/inscribed.json`:

| 항목 | SCMR | SCMR + 내접 덮개 |
|---|---|---|
| parse · `reject_bad_flag_combinations` | 통과 | 통과 |
| `sqp_overrides` · config dict | — | SCMR 과 같음 |
| 제약 로봇 구 (서버 builder 옵션 그대로, `ta_offline.build_robot`) | 606 (collision 46 · finger_body 224 · palm_shell 336) | **3,213** (+ arm_cover 2,607) |
| 앞 606 구 | — | 비트 동일 |
| linearizer 구 | 606 | 3,213 |
| client metadata 차이 | — | `arm_spheres` · `constraint_spheres` 둘만 |
| record meta `arm_spheres` | 없음 | path · sha256 · method · link 별 수 |
| 시작 log | 없음 | `constraint arm spheres (AG3S-ARM-SHAPE): 구 606 → 3213 …` |
| coarse 격자 밖 구 (E4 1345, 544 상태) | 0 | 0 (여유 최소 0.645 m) |

link 별 구 수는 §5.1 표와 같다. `--to-gpu-seeds curobo` 의 CUDA 확인은 패치 (실제 확인은 서버 시작 때). AG3S 자체 (`build_ag3s`, checkpoint · CUDA) 는 짓지 않았다 — 그 길로 키가 들어가는 것은 test 가 고정한다.

### 띄우는 법

`launch_occ_SCMR.sh` 의 `SN` 을 `$RUN/snapshot_arm` 으로, `SAFE` 끝에 `--arm-spheres $RUN/arm_spheres/<inscribed|morphit>.json` 을 붙인다 (`--links gripper` 그대로). 독점 GPU lock.

## 7. 프로세스 · 캐시

- CPU worker ≤ 2 (내접 fit 1 + test 1), GPU 는 shared lock 1 건. 서버 · closed-loop · git commit 없음.
- (채움: reclaim 줄)

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->
- 내접 덮개는 구가 mesh 안이라 형상만의 거리가 GT 보다 늘 크다 (실제보다 멀다고 봄, 최대 약 5 mm) — 깊이 5 mm 미만의 실제 침투는 놓칠 수 있다. 거짓 충돌은 거의 없을 것.
- MorphIt 는 mesh 밖으로 나오는 구가 있어 거짓 충돌 쪽이 더 많고, 표면 덮음 기준을 맞추려면 구 수가 많이 필요할 것.
