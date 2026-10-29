# IFG-CL1 — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 2026-10-16 03:45 KST (UTC 2026-10-15 18:45)
> task: `AG3S/ag3s/docs/handoff/IFG-CL1.task.md` · branch `feature/ag3s-development` @ `f15e5872` (commit 안 함)

## 무엇을 했나 (평이한 요약 먼저)

서버가 이제 **attention + depth 로 직접 code 영상 (0 없음/로봇 · 1 target · 2 static) 을 만들어** 이미 있는 `oracle_visible`
경로 (memory · carving · split · 계층 · 권한) 에 넣을 수 있다 (`--target-extent attention-surface`). target 범위를 고르는 계산은
IFG LOCK 의 `top1-only-ns+rm:N` 을 **잠금 코드 (`ifg_graph`, `ifg_round_ns`) 를 import 해서** 그대로 돌린다 — 다시 쓰지 않았다.
로봇 pixel 은 `--robot-mask render` 로 서버 `--model-xml` 의 로봇만 요청 관절 상태로 렌더해 depth 일치 (≤ 10 mm) 로 정할 수 있다
(IFG R6 의 서버판). 둘 다 기본 off 이고 off 에서는 호출 자체가 없다.

dev 180 frame 전부에서 서버 모듈이 고른 node 집합 = offline LOCK arm (IoU 1.0, 180/180), render mask 는 offline R6 mask 와
pixel 일치율 ≥ 0.99974 (30 frame × 3 camera). 대가는 **요청당 약 1.0–1.1 s 추가 (이 CPU, median)** 이고 대부분은 잠금
graph 생성 (`build_geometry`, median 678 ms) 이다 — 아래 "남은 위험" 1.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/ag3s/runtime/attention_surface.py` (신규) | `SurfaceView` · `select_target_surface` · `attention_vector` · `_pixel_nodes`. LOCK 의 `GraphConfig()` · `eligible(geo, cfg, True)` · `g_and_top1` 을 import 해 호출, 선택 node → pixel 로 되돌려 code 영상 (원 depth 해상도, uint8). 영역이 없으면 `why` (`missing_camera:*` · `no_attention:*` · `no_seed` · `no_region` · `no_depth_points`) 와 함께 `codes=None`. 단계별 ms | task 설계 2 |
| `AG3S/ag3s/runtime/robot_render_mask.py` (신규) | `RobotRenderMask`: 자기 MjModel 사본에서 로봇 geom 만 (R6 규칙: gripper root 아래 ∪ `is_robot_body`, mocap 제외, 나머지 group 5) 남기고, 요청의 20-joint `robot_state` + 측정 손가락 개도 (client 규약 `gripper_state`) + MJCF joint equality (둘째 손가락) 로 자세, 나머지 관절 (head · 바퀴 · base) 은 keyframe `teleop`. 카메라 0 의 `cam_xpos/xmat` · `fovy` 를 요청 K · T_base_cam 으로 덮어쓰고 depth 렌더 → `valid_r & |depth − render| ≤ TOL` (TOL 은 잠금 `ifg_render_mask.TOL` import). 중심 · 정사각 pixel 이 아닌 K 는 그 카메라 `None` + 이유. `probe()` = 시작 시 1회 렌더 | task 설계 3 |
| `AG3S/ag3s/config.py:1141` | `TARGET_EXTENTS` 에 `"attention_surface"` | 설계 1 |
| `AG3S/ag3s/config.py:1148-1149` · `:1298` | `esdf.robot_mask = "spheres"` (기본) · `ROBOT_MASKS = ("spheres", "render")` · validation | 설계 3 의 flag 를 config 로 |
| `AG3S/ag3s/config.py:1301-1307` | F1 · F2 거절 규칙을 `attention_surface` 에도 (oracle 경로가 그 분할을 대신하므로) | oracle 과 같은 이유 |
| `AG3S/ag3s/runtime/pipeline.py:145-154` | `AG3S(robot_render=...)` kwarg. `robot_mask == "render"` 인데 없으면 생성에서 `ValueError` | fail loudly |
| `AG3S/ag3s/runtime/pipeline.py:530-582` | `_oracle_update` 를 `_oracle_memory_or_new` · `_oracle_fold` 로 쪼갬 (oracle 경로의 동작 · 기록 키 그대로) | 같은 memory 를 두 출처가 쓰도록 |
| `AG3S/ag3s/runtime/pipeline.py:605-655` | `_apply_render_mask`: TSDF mask = **구 mask ∪ render mask**, 바뀐 카메라는 `device_depth=None` (구 mask 로만 GPU 에 올라간 depth 를 쓰지 않게). 렌더 실패 → 구 mask 유지 + `why: render_failed` 기록 + warning | 설계 3 (b) |
| `AG3S/ag3s/runtime/pipeline.py:657-705` | `_attention_surface_update`: CAMERA_ORDER (head · left_wrist · right_wrist = attention token 순) 로 view 를 짓고, graph robot mask = render mask (render on) / 구 mask (off) → `select_target_surface` → code 영상을 `_oracle_fold` 로 memory 에. abstain 이면 memory 갱신 없음 | 설계 2 |
| `AG3S/ag3s/runtime/pipeline.py:1644-1647` | frame 시작에 `_surface_frame` · `_surface_codes` · `_render_frame` 초기화 | 지난 frame 기록이 남지 않게 |
| `AG3S/ag3s/runtime/pipeline.py:1937-1953` | ESDF 단계: `robot_mask == "render"` → `_apply_render_mask` (profiler `robot_render_mask`), `target_extent == "attention_surface"` → `_attention_surface_update` (profiler `attention_surface`) | 배선 |
| `AG3S/ag3s/runtime/pipeline.py:2940-2952` | `_build_esdf`: attention_surface 이고 이번 frame 이 abstain 이면 evidence 를 만들지 않고 (`estimated`) `why: attention_surface_abstained (<이유>)` | 설계 2 "영역이 없으면 estimated" |
| `AG3S/ag3s/runtime/pipeline.py:3005-3014` | `stats["target_free"]["attention_surface"]` (surface 기록), `stats["robot_mask_render"]` (render 기록) | 설계 4 |
| `AG3S/ag3s/runtime/pipeline.py:2161-2164` | `debug["attention_surface_codes"]` (attention_surface 일 때만) | 기록기가 code 영상을 싣게 |
| `AG3S/ag3s/experiments/sources/constraint_record.py:200-203` | npz `attention_surface_codes_<camera>` (uint8, 있을 때만) | 설계 4 |
| `AG3S/ag3s/experiments/sources/constraint_record.py:245-248` | summary `robot_mask_render` (JSON, 있을 때만) | 설계 4 |
| `AG3S/ag3s/experiments/sources/constraint_record.py:532-539` | summary `target_exclusion` 에 `attention_surface` (+ 그때만 `oracle`) | 설계 4 — oracle_visible 기록은 그대로 |
| `AG3S/trajopt/serve_safe.py:1681-1702` | `--target-extent` 에 `attention-surface`, 새 `--robot-mask {spheres,render}` | 새 flag |
| `AG3S/trajopt/serve_safe.py:915-916` | `target_exclusion_section` 이 `robot_mask` 를 준 경우만 esdf 키로 | 준 flag 만 config 에 |
| `AG3S/trajopt/serve_safe.py:2142-2166` | 시작 거절: attention-surface 인데 `--attention-source` 가 `noun_target:L10H7` 이 아님 · `--no-attention`; `--robot-mask render` + `--no-safe`/`--no-perception`; oracle 거절 규칙 (relax policy · F1/F2 · `--oracle-*` 단독) 을 attention-surface 에도 | fail loudly |
| `AG3S/trajopt/serve_safe.py:1196-1213` | `build_ag3s`: render 이면 `RobotRenderMask(model_xml)` 생성 + `probe()` (GL 없으면 여기서 죽는다) + 큰 warning, `AG3S(..., robot_render=)`. off 이면 AG3S 호출 그대로 | 시작 시 확인 |
| `AG3S/trajopt/serve_safe.py:1240-1250` · `:2010` · `:2950-2951` | 시작 warning, record meta `target_extent` · `robot_mask` (켰을 때만) | 기록 구별 |
| `AG3S/trajopt/safe_policy.py:383-390` | metadata `target_extent: attention_surface` · `oracle_extent_identity` · `robot_mask: render` (켰을 때만). `ORACLE_SEG_REQUIRED` 는 싣지 않는다 (client segmentation 불필요) | 서버 설정 노출 |
| `AG3S/ag3s/experiments/studies/cl1_parity.py` (신규, 잠금 아님) | dev 전 frame parity (extent · render) 를 JSONL 로 쓰는 도구 — verifier 재측정용 | 재현 |
| `tests/ag3s/test_ifg_cl1_attention_surface.py` (신규) | 23 test (아래) | 수락 조건 |

잠금 `ifg_*.py` 는 손대지 않았다 (`sha256sum -c` LOCK 목록: 내가 import 하는 `ifg_graph` · `ifg_round_ns` · `ifg_round` ·
`ifg_run` · `ifg_s1` · `ifg_reader` · `ifg_render_mask` 모두 OK). 참고: LOCK 목록 중 `ifg_phase.py` 는 이미 sha 가 다르다 — commit
`17ae6ad3` (내 세션 전) 에서 바뀌었고 이 작업과 무관하다. lead 확인 필요.

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-development
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv-ag3s/bin/python -m pytest -q tests/ag3s/test_ifg_cl1_attention_surface.py
#   → 23 passed (52 s)
MUJOCO_GL=osmesa PYTHONPATH=. .venv-openpi-live/bin/python -m pytest -q tests/ag3s/test_ifg_cl1_attention_surface.py tests/ag3s/test_t46e_oracle_extent.py
#   → 53 passed
MUJOCO_GL=osmesa PYTHONPATH=. .venv-ag3s/bin/python -m pytest -q tests/ag3s/test_t46e_oracle_extent.py \
    tests/trajopt/test_t46f_a14c_oracle_combination.py tests/trajopt/test_t57a_attention_source.py
#   → 112 passed, 3 skipped
MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv-ag3s/bin/python -m pytest -q tests/      # 기존 전부 (CPU)
#   → 3739 passed, 129 skipped, 1 failed (48 min). 실패 1 개 = tests/trajopt/test_t43t_replay.py::
#     test_off_is_bit_identical_to_the_pre_fix_commit (key to_T42_L1992_16, TO chunk) — `git archive HEAD` 의 깨끗한
#     사본에서 같은 key 로 똑같이 실패한다 (이 변경 전부터의 실패, TO 수치 golden). 이 변경과 무관
```

| 수락 조건 | test | 결과 |
|---|---|---|
| parity ≥ 5 dev frame, node IoU ≥ 0.99 | `test_parity_with_the_offline_locked_arm` (6 frame: M1 · M2 · F1 · M7 ×2 · M1) — offline 은 잠금 chain + **실제 GT**, node id 는 위치로 대조; 서버 모듈은 **더미 GT** · 카메라별 16×16 map · R6 render mask | IoU **1.0** (6/6), seed patch 동일, 선택 node 수 = 저장된 R6 row `pred_voxels` |
| (보충) dev 전체 | `cl1_parity.py` 와 같은 계산, 180 frame | IoU 1.0 **180/180**, seed 일치 180/180, `pred_voxels` 일치 180/180. 1 frame (`T54c_ep1831_f032` M7) 은 offline 도 선택 0 → 서버는 `no_region` abstain |
| 더미 GT 가 선택을 바꾸지 않는다 | 같은 test: `class_map/classes` = 실제 GT 로 다시 돌려 선택 node · seed 동일 | 6/6 |
| code-1 pixel 을 다시 올린 점이 선택 node voxel 에 | `test_code_one_pixels_lift_into_the_selected_voxels` (3 frame): `backproject_labels` 로 올린 code-1 voxel 집합 **= 선택 node voxel 집합**, code-2 와 교집합 0, robot pixel 은 code 0 | 3/3 (180 frame 보충: 1.0) |
| 기본 설정 비트 동일 | `test_pipeline_default_never_calls_the_new_steps` (select 를 터지게 바꿔도 기본 · 명시 `estimated`+`spheres` 둘 다 호출 없음, stats 키 없음, builder kwargs 키 동일), `test_server_metadata_only_when_on`, `test_cli_flags_map_to_esdf_keys_only_when_given`, 기록 test, + 기존 test 전부 | 통과 |
| attention source 없으면 시작 에러 | `test_cli_refuses_attention_surface_without_the_locked_source` (없음 · `action:last:L8H2` · `noun_target:L10H6` → exit 2) | 통과 |
| render mask vs offline `ifg_render_mask` | `test_render_mask_matches_the_offline_r6_masks` (6 frame, 저장된 R6 mask) · `test_render_mask_on_a_synthetic_state_matches_the_r6_procedure` (합성 자세: 팔 · 몸통 5 관절 ±0.3 rad 무작위 + 왼손 개도 0.03, R6 절차 = MJCF 카메라 + 전체 qpos 로 렌더, 관측 depth = 같은 자세의 전체 씬 렌더) | 6 frame 최소 일치율 ≥ 0.9999; 합성 ≥ 0.999 (3 camera). 보충 30 frame × 3: 최소 **0.999743** (79 px, 오른손 wrist cam), 평균 0.999984, 총 437 px 차 |

다른 test: abstain 이유 (`test_abstain_reasons`), pipeline 배선 — 3 카메라 raycast rig 에서 **실제 선택**이 사과 (GT code 1 순도 > 0.95)
를 고르고 oracle evidence 의 target 표본이 사과 표면 (median |d| < 3 mm), 이웃은 code 2 → other 표본
(`test_pipeline_attention_surface_feeds_the_oracle_path`); abstain frame 은 이전 memory 가 있어도 evidence 없음
(`..._abstain_builds_the_estimated_extent`); render ∪ 구 mask · device_depth 해제 · 측정 개도 전달 · 렌더 실패 fallback
(`test_pipeline_render_mask_is_unioned_with_the_sphere_mask`); 기록 npz · JSON (`test_record_carries_...`); K guard.

## 서버 1 요청당 추가 ms 추정 (이 machine, CPU, osmesa, load ≈ 37 — 다른 세션 서버 · client 가 같은 CPU 에서 돌 때)

| 단계 | 언제 | median | p90 | 근거 |
|---|---|---:|---:|---|
| render mask (3 × 640×480, osmesa) | `--robot-mask render` | 105–130 ms | ≈ 200 ms | 30 frame (`cl1_parity --render` 와 같은 계산). 첫 렌더 (context) ≈ 0.9–1.2 s → `probe()` 가 시작 때 흡수 |
| `build_geometry` (잠금, 3 camera 전 pixel ~52 만 → ~9 만 node, 지지면 RANSAC 포함) | attention-surface | 678 ms | 1037 ms | 180 frame |
| `g_and_top1` (seed · ROI · solve · region) | 〃 | 38 ms | 101 ms | 〃 |
| code 영상 (pixel → node 재도출 + 영상) | 〃 | 109 ms | 208 ms | 〃 |
| OracleMemory 갱신 (carving · ROI · 띠) | 〃 (oracle_visible 와 같은 비용) | 50–130 ms | — | 5 연속 frame |
| TSDF host depth 재업로드 (union 카메라는 B4 GPU depth 를 못 씀) | render | 측정 안 함 (3 × 1.2 MB 업로드, 수 ms 예상) | | |
| **합계** | 둘 다 켰을 때 | **≈ 1.0–1.15 s** | ≈ 1.7 s | attention-surface 단독 ≈ 0.9–1.0 s |

`build_geometry` 가 비싼 이유: 3 camera 전 pixel `np.unique` (sort) 2 회 · node 마다 3×3 `eigh` · 전 scene RANSAC. ROI 로 입력을
줄이면 빨라지지만 **지지면 RANSAC 이 전 scene 에 의존해 선택이 달라질 수 있다** — 잠금 방법이 바뀌므로 하지 않았다 (lead 결정 사항).
GPU 서버가 EGL (`MUJOCO_GL=egl`) 로 뜨면 render 는 훨씬 짧을 것으로 보지만 여기서 재지 않았다 (env.sh 기본은 osmesa).

## self-filter 처리 제안 (지금 구현된 것 = 제안 1)

지금 serving 의 self-filter 경로 (읽은 것): `fuse_observations` 가 카메라마다 구 모델 (URDF capsule + gap · cover capsule, T29) 에
inflation (`pointcloud.self_filter_inflation[_by_link]`), **쥔 물체 구** (T32 H1, attach 동안), **target guard** (T19/T21: manipulated
물체 중심 근처 pixel 은 손가락 inflation 이 지우지 못함) 를 적용해 (1) fused cloud (grounding · clustering · attention lifting ·
지지면) 에서 로봇 점을 빼고 (2) 같은 결정을 pixel mask 로 한 번 정해 (T38 B2) TSDF 의 `CameraDepth.robot_mask` (+ B4 device depth)
로 넘긴다.

| 안 | TSDF | graph (attention-surface) | cloud (grounding) | 장 | 단 |
|---|---|---|---|---|---|
| **1 (구현, 권장)** | 구 ∪ render | render 만 (R6 와 같다) | 구 그대로 | 쥔 물체 구 · guard · inflation 이 계속 TSDF 를 지킨다; render 가 구가 못 덮은 mesh pixel (`n_render_only_px` 기록) 을 더 뺀다; 로봇 pixel 을 **덜** 빼는 경우가 없다 | TSDF 쪽에서는 R6 의 효과 (손가락 사이 과일 보존) 를 얻지 못한다 — 그 효과는 graph 에만 |
| 2 | render 만 | render 만 | 구 그대로 | offline R6 와 완전히 같은 mask | **쥔 과일이 TSDF 에 들어간다** (render 는 로봇만 그린다 → 손에 붙은 장애물 · 자기 충돌 위반), guard · inflation 없음. 지금은 비권장 — 하려면 쥔 물체 render (attach 구) 를 더해야 한다 |
| 3 | 구 만 | render 만 | 구 그대로 | TSDF 는 C1a 와 비트 동일 → extent 출처만 다른 순수 비교 | render 의 추가 로봇 pixel 이 TSDF 에 남는다 |

closed-loop 비교를 "target 범위 출처만 다르게" 하려면 안 3 이 가장 깨끗하다. 안 3 은 구현에서 한 줄 차이 (`_apply_render_mask` 의
union 을 끄는 sub-option) 이지만 **task 에 없는 flag 라 넣지 않았다** — lead 가 고르면 추가한다. cloud self-filter (grounding /
identity) 는 세 안 모두 그대로다: M2–M3 에서 손가락 사이 과일 점이 cloud 에서 지워지면 AG3S identity (`oracle_extent_identity
= pipeline` 이 쓰는 manipulated 기하) 가 code-1 표본을 못 찾을 수 있다 (위험 6).

## verifier 가 알아야 할 것

- 새 flag (기본 = 지금 그대로): `--target-extent attention-surface` (esdf `target_extent = "attention_surface"`; `--attention-source
  noun_target:L10H7` + `--target-field-policy exclude-*` 필수, `--oracle-extent-identity` · `--oracle-max-dist-voxels` 도 읽힘),
  `--robot-mask {spheres,render}` (esdf `robot_mask`). 기본값 변경 없음.
- 새 기록: chunk npz `attention_surface_codes_{head,left_wrist,right_wrist}` (uint8, 선택된 frame 만), summary
  `target_exclusion.attention_surface` · `target_exclusion.oracle` (attention_surface 일 때만), summary `robot_mask_render`;
  `esdf_stats` 에 같은 내용 (repr). profiler stage `robot_render_mask` · `attention_surface` (켰을 때만). record meta
  `target_extent` · `robot_mask`, server metadata `target_extent` · `robot_mask` (켰을 때만).
- attention_surface 가 적용된 frame 의 `stats["target_free"]["mode"]` 는 builder 가 출처를 모르므로 **`oracle_visible`** 로 찍힌다.
  출처는 `target_free.attention_surface` · record meta 로 구별한다. `oracle.object` = `"attention_surface"` (memory legend).
- 재생산이 필요한 산출물: 없음. parity 재측정: `python -m AG3S.ag3s.experiments.studies.cl1_parity --run $RUN --out $AG3S_RUN_DIR/cl1_parity --render`.
- 옛 기록과 호환: oracle_visible 기록 · 기본 기록의 키 집합은 그대로 (test). `_oracle_update` 는 리팩터만 (oracle test 30 개 통과).
- **실수 보고**: `.venv-openpi-live` 에서 `tests/ag3s/test_t46e_oracle_extent.py` 를 돌릴 때 그 venv 는 `curobo` 를 import 할 수 있고
  CUDA 가 보여 기존 GPU test 1 개 (`test_curobo_default_is_bit_identical_...`) 가 **GPU 에서 수 초 돌았다** (lock 없이, collector
  가 GPU 를 쓰는 중). 다른 process 는 건드리지 않았다. 이후 GPU 가능 venv 로는 test 를 돌리지 않았다 (`.venv-ag3s` 에는 torch 없음).

## 남은 위험

1. **지연**: 요청당 ≈ 1.0–1.15 s (median) 추가. 잠금 `build_geometry` 가 전 scene graph 를 매 요청 새로 짓기 때문 — 잠금을 지키는 한
   줄일 방법이 code 영상 쪽 (~0.1 s) 밖에 없다. closed loop 에서 TO · 실행 timing 에 미치는 영향은 lead 판단.
2. render 는 osmesa CPU 로 ~0.1–0.2 s. 서버 환경의 `MUJOCO_GL` 이 없거나 깨지면 `probe()` 가 시작에서 죽인다 (의도). 실행 중 실패는
   그 frame 만 구 mask 로 되돌아가고 `robot_mask_render.why = render_failed` + warning.
3. 둘째 손가락은 MJCF equality 로 계산 (시뮬레이터 실제값과 ≤ 0.2 mm 차 → wrist cam 손가락 가장자리 최대 79 px 불일치). head ·
   바퀴 · base 는 keyframe `teleop` (head_1 = 0.7): 요청에 없는 관절이라 head 가 움직이는 과제에서는 틀린다.
4. 중심 · 정사각 pixel 이 아닌 K (실 카메라) 는 render 하지 않는다 (그 카메라는 구 mask, 기록에 이유).
5. memory 의미: 잘못 고른 영역 (이웃 과일) 의 code-1 표본은 같은 칸이 code 2 로 다시 보이거나 carving 될 때까지 남는다. abstain frame
   은 memory 를 갱신하지 않고 `estimated` 로 짓는다 (oracle 경로는 segmentation 없는 frame 에서도 옛 memory 를 쓴다 — 여기서는
   task 의 "영역이 없으면 estimated" 를 따랐다).
6. identity 결합: 기본 `oracle_extent_identity = pipeline` 은 manipulated 기하 (구 self-filter 를 거친 cloud) 의 점이 code-1 표본 위에
   다수 (≥ ½, fusion voxel 반경) 로 있어야 적용된다. attention 영역이 과일 일부만 덮으면 같은 과일의 나머지가 code 2 라 `identity_unresolved`
   → estimated 가 날 수 있다. `gt` (= 늘 code 1 = attention 영역) 와 무엇을 쓸지는 lead 결정.
7. attention 입력: parity 는 dump `derived.npz` 의 noun map (float16) 으로 했다. 서버의 float32 map 과 top-5 순서가 동점 근처에서 다를 수
   있다 (T57-A parity 가 map 자체는 다룸). 서버 map 이 dump 와 같은 이미지에서 계산된다는 전제. depth 도 parity 는 replay float32,
   서버는 wire 의 uint16 mm (1 mm 양자화) 라 node 경계 pixel 이 조금 다를 수 있다 (jump gate 10 mm · voxel 5 mm 에 비해 작다).
8. 이 checkout 을 다른 세션 process 도 import 한다 — 새로 뜨는 그들의 서버는 이 변경을 읽는다 (기본 경로는 test 로 비트 동일).

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->
closed loop 에서 attention-surface (+ render) arm 은 C1a (`ball-owned`) 대비 미세 target-free 계층이 지우는 seed 가 이웃 과일 ·
table 에서 줄고 target 표면에서 유지될 것으로 기대한다 (oracle 상한 쪽으로). abstain 은 dev 기준 1/180 frame 이라 드물 것이다.
지연 ≈ 1 s 가 실행 timing 을 바꿔 (TO 가 기다림) 성공률 비교를 흐릴 수 있다 — arm 간 timing 을 같이 볼 것.
