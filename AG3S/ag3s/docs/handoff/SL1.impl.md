# SL1 — 구현

> writer: ag3s-implementer (A1) · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-12 12:30 KST
> branch `feature/panda-benchmark`, HEAD `14ca3048` (commit 안 함 — lead 가 commit)

## 무엇을 했나 (평이한 요약 먼저)

LIBERO/robosuite 관측을 V11AB 가 읽는 형식으로 옮기는 Python 3.8 adapter(`benchmark/adapters/libero_panda/`)를 만들었다.
camera 별로 metric z-depth(OpenCV row order), K(fovy), `T_base_cam`(robot0_base 기준, OpenCV 축), 9D robot_state, stamp 를 내고,
이것을 `AG3S.trajopt.wire` 와 key·dtype 이 같은 wire dict/npz 로 바꾼다. Client 에 `--capture-geometry`(기본 off)를 붙여
step 마다 `episodes/<stem>_geom.npz` 를 남기게 했고, 기존 run 의 기록 action 으로 policy 없이 같은 episode 를 다시 돌리는
replay 도구(3.11 launcher + 3.8 replay)를 만들었다. D 항목(manifest 의 해당 없음 항목·runtime schedule, `serve_policy` 의 `env="aloha_sim"` 기록, 고정 CPU 회귀 명령)도 처리했다.

구현 중 **계획에 없던 사실 하나**를 찾았다: robosuite 는 관측(카메라 render 포함)을 `env.step` 이 끝날 때가 아니라
physics substep loop 안에서 observable timer 가 정한 substep 에 찍는다. 그래서 `env.step` 이 돌아온 순간의 `env.sim`
(관절, wrist camera pose, 물체 위치)은 image 보다 늦다 (smoke: 1 substep = 2 ms, joint 차 최대 3e-4 rad).
`env.sim` 에서 그대로 pose 를 읽으면 wrist `T_base_cam`·robot_state 가 image 와 다른 순간의 것이 된다. 그래서 camera 와
같은 sampling rate 의 **읽기 전용 observable `ag3s_capture_state`** 를 env 에 추가해, image 와 같은 substep 의
`[time, qpos, qvel, T_world_base, T_world_cam(camera 별)]` 를 obs 에 같이 싣고 adapter 는 그 값을 쓴다
(`camera.py:218`). 이 observable 은 geometry capture 가 켜졌을 때만 설치되고, sim 을 읽기만 한다.

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `benchmark/adapters/libero_panda/__init__.py` (새) | package 설명 | A |
| `benchmark/adapters/libero_panda/camera.py` (새, py3.8, numpy) | camera table `:58` (`agentview→head`, `robot0_eye_in_hand→left_wrist`), `conventions()` `:72` (manifest 용 규약 문장), robosuite 공식 재서술 `intrinsic_matrix` `:100`·`real_depth` `:106`·`world_camera_pose_opencv` `:131` (`cam_xmat @ diag(1,-1,-1)`), `rigid_inverse` `:123`, OpenGL→OpenCV row flip `:136` | A.1 |
| `camera.py:142-189` | policy image ↔ OpenCV pixel: `raw_to_policy_array` (client 의 `[::-1, ::-1]` 그대로), `resize_with_pad_geometry` (openpi_client `_resize_with_pad_pil` 의 크기·pad 산술 복제), `opencv_to_policy_pixel` / `policy_to_opencv_pixel` (연속 좌표, pixel 중심 = 정수 index; PIL 사상 `x_out=(x_in+0.5)s-0.5+pad`), `opencv_to_raw_pixel`, `project_base_points` `:191` (AG3S `backproject` 와 같은 pinhole) | A.4 |
| `camera.py:204-326` | `install_capture_observable` `:218` (위 요약의 snapshot observable), `unpack_capture_state` `:254`, `capture(env, obs, segmentation=False)` `:269` → camera 별 `{depth_m float32 (H,W), rgb, K, T_world_cam, T_world_base, T_base_cam, robot_state(9), stamp, sim_state, fovy, near/far, [segmentation_geom int16]}`; `model_tables` `:329` (geom/body/joint 이름, robot qpos index) | A.1 |
| `benchmark/adapters/libero_panda/wire_pack.py` (새, py3.8, numpy) | `pack_cameras` `:49` = `pack_request` 의 camera 부분(`ag3s/stamp_mode`, `cameras`, `depth/<cam>` uint16 `clip(rint(d·1000),0,65535)`, `K`, `T_base_cam`, `robot_state` float64, `stamp` float). npz 변환 `:78`/`:93`, exclusive 저장 `:109`, geom npz 의 frame i → wire `:139`, CLI `export` `:150` (frame 별 `<stem>_wire_<i>.npz`, 덮어쓰기 거부) | A.2 |
| `benchmark/runners/libero_pi05_client.py` | `_get_libero_env_with_depth` `:154` (공식 `_get_libero_env` + `camera_depths=True` 만), `_geometry_adapter` `:222` (checkout root 를 `sys.path` 끝에 **append** — 설치 package 를 가리지 않음, flag on 일 때만 import), `_GeometryRecorder` `:235`, `run_episode(..., observer=None)` `:285` (`rec.observe` 직후 3곳 `:307/:322/:379`), flag `:453`·검사 `:463`, `eval_libero` `:477` → task/episode loop 를 `run_tasks` `:511` 로 분리 (내용 그대로, `policy_for` 만 주입), on 일 때 env 생성 직후 observable 설치 `:529`, `_geom.npz` 저장·jsonl `geometry` key `:564` | B.5 |
| `benchmark/runners/libero_replay.py` (새, py3.8) | source npz 검사 `check_replayable` `:57` (기록 `action` == `action_chunks` 에 replan 규칙을 적용한 값, bit 단위), `ReplayPolicy` `:71` (i 번째 호출에 i 번째 기록 chunk, 다 쓰면 예외 → `execution_error "replay exhausted"`, action 을 만들어내지 않음), protocol(suite/seed/num_steps_wait) 대조 `:109`, `replay` `:130` (client 의 `run_tasks` 를 그대로 씀 → seed 순서·env 생성·`set_init_state`·대기·replan·done 처리 동일), jsonl `replay` key (source run id/경로/npz SHA-256/done step) | B.6 |
| `benchmark/runners/launch_libero_replay.py` (새, py3.11) | source run 해석 `:32` (이름 또는 절대경로, `results/benchmark` 아래만), source 기록 `:40` (manifest·jsonl·npz SHA-256, commit, protocol), settings `:76` (seed/suite 불일치 거부), `create_run` → `record_execution(role=replay_client)` → `.venv-libero` 실행(SL0 와 같은 `client_environment`: PYTHONPATH 제거, foreign torch `LD_LIBRARY_PATH` 제거, thread 4) → `launcher_result` `:108`, `--dry-run` `:158` | B.6 |
| `benchmark/runners/launch_libero_pi05_only.py` | `geometry_flags` `:207` (client argv 로 전달), `libero_control_freq` `:273` (LIBERO `ControlEnv.__init__` 기본값을 AST 로 읽음 = 20), `runtime_schedule` `:289` (20 Hz, replan 5, policy 호출 주기 5 step = 0.25 s sim, 대기 10, 동기 sim), `not_applicable_items` `:306` (protected/authorized links, active_manipulators, phase, level, ag3s seq, exec_feedback, policy_seed = `n/a` + 이유), `geometry_settings` `:346`, settings `step` = flag 에 따라 `SL0`/`SL1` `:317`, parser `:464`, 검사 `:479` | D.9, B.5 |
| `third_party/openpi/scripts/serve_policy.py:107-112` | manifest 자기 항목: checkpoint 경로이면 `env=None`, `policy={"source","config","dir"}` (아래 diff) | D.9 |
| `scripts/testing/run_cpu_regression.sh` (새) | A: SL0 의 5 파일(`.venv-openpi-live` + `run_without_legacy.py`), B: `tests/benchmark`(같은 venv·guard), C: `tests/benchmark`(`.venv-ag3s`). 하나라도 실패면 exit 1 | D.10 |
| `tests/benchmark/test_libero_sl1.py` (새, 21 test 함수 / 28 case) | 아래 표 | C |
| `tests/benchmark/libero_py38_checks.py` (새, py3.8 script) | `.venv-libero` 에는 pytest 가 없어(공식 pin, freeze 고정) 검사를 script 로 두고 pytest 가 subprocess 로 부름. 실제 LIBERO env 1개 생성 (~20 s, 1 GB 이하) | C.8 |

손대지 않은 것: `AG3S/**` 코드, `robots/**`, `benchmark/LIBERO`(git status clean), `.venv-libero`(설치 변경 없음), storage 기존 input, 실제 results 디렉터리(smoke 는 scratchpad 의 임시 storage root).

### `serve_policy.py` diff (D.9)

```diff
@@ -104,6 +104,12 @@ def create_policy(args: Args) -> _policy.Policy:
 def main(args: Args) -> None:
     settings = json.loads(json.dumps(dataclasses.asdict(args), default=lambda value: value.value))
     checkpoint = args.policy if isinstance(args.policy, Checkpoint) else DEFAULT_CHECKPOINT[args.env]
+    # Record the policy actually served. `env` only selects a Default policy; with policy:checkpoint
+    # its Args default ("aloha_sim") is unused and must not appear as if it were the environment.
+    if isinstance(args.policy, Checkpoint):
+        settings["env"] = None
+    settings["policy"] = {"source": "checkpoint" if isinstance(args.policy, Checkpoint) else "default",
+                          "config": checkpoint.config, "dir": checkpoint.dir}
     run = ensure_run(settings=settings, checkpoint=checkpoint.dir)
```
Default policy 경로(`--env libero`)는 `env` 를 유지하고 `policy` 에 실제 default checkpoint(config/dir)를 적는다. 동작(load·serve) 변경 없음.

### 계획·공식 경로와 다른 점

| # | 차이 | 이유 / 영향 |
|---|---|---|
| 1 | robosuite `camera_utils` 를 import 하지 않고 세 공식을 `camera.py` 에 다시 썼다 | 그 module 이 `h5py` 를 import 하는데 `.venv-libero`(공식 requirements pin)에 없다. 설치하면 freeze 재현이 깨진다. 대신 robosuite source 의 세 함수를 AST 로 꺼내 exec 해서 값 비교(`libero_py38_checks.py`) |
| 2 | depth 를 float64 로 계산 후 float32 저장 (robosuite 는 numpy 1.22 casting 으로 float32 계산) | 두 값 차이 최대 3.2e-5 m(head) / 7.7e-7 m(wrist) — wire 1 mm 양자화보다 작다. test 허용 1e-4 m |
| 3 | geometry on 이면 env 에 observable `ag3s_capture_state` 추가 (obs key `ag3s_capture_state`, `ag3s_capture-state` 가 생김) | 위 요약. policy 입력 key·recorder key 는 그대로. **#1 에서 RGB 가 다르면 원인 후보는 `camera_depths=True` 와 이 observable 두 가지**다 (둘 다 render 를 바꾸지 않는다고 보지만 측정 전) |
| 4 | segmentation(`--capture-segmentation`)은 `env.step` 이 돌아온 시점에 따로 render 한다 | image 순간이 아니라 `sim_state_at_return` 순간 (1 substep 뒤). image 순간의 id 가 필요하면 `sim_state` 로 별도 env 에서 재생성. image 순간으로 sim 을 되돌렸다 복원하는 방식은 물리 상태를 건드릴 수 있어 쓰지 않았다 |
| 5 | K 는 robosuite/AG3S RB-Y1 과 같은 `cx=W/2, cy=H/2` | AG3S `backproject` 는 `u = column index`(pixel 중심이 정수)를 쓰므로 그 규약에서 광축은 `(W-1)/2=127.5`. 즉 +0.5 px principal point 오프셋이 있다. 고치지 않고 manifest·`conventions()` 에 적었다 — 크기는 측정 대상 |
| 6 | replay 는 별도 step loop 가 아니라 client 의 `run_tasks`/`run_episode` 를 그대로 쓰고 policy 만 `ReplayPolicy` 로 바꿨다 | 같은 seed·env·대기·replan·done 처리를 보장. 실행 action 은 기록 chunk 의 앞 5 행(float64 로 기록된 값 → `tolist()` 가 원래 float32/64 와 같은 python float). 시작 전 기록 `action` 과 bit 일치를 확인 |
| 7 | 회귀 script 는 3 suite 로 나눴다 | SL0 의 73 개를 그대로 비교할 수 있게 A 를 따로 둠. B/C 는 같은 `tests/benchmark` 를 두 interpreter 로 |
| 8 | SL0 launcher 의 manifest settings 에 새 key(`runtime_schedule`, `not_applicable`, `geometry_capture`)가 생겼다 | D.9. flag off 이면 client argv·client 출력은 SL0 와 같다 (`step` 은 `SL0` 그대로) |
| 9 | live client + `--capture-geometry` 를 policy server(stub 포함)와 함께 돌려 보지는 않았다 | client 와 replay 가 같은 `run_tasks` 를 쓰므로 geometry 경로는 replay smoke 로 실행됐다. `make_policy_client` 경로는 SL0 그대로 |

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-benchmark && bash scripts/testing/run_cpu_regression.sh
# == A: 73 passed, 77 warnings in 32.44s        (SL0 verifier 의 73 과 같음)
# == B: tests/benchmark (.venv-openpi-live): 44 passed in 48.30s
# == C: tests/benchmark (.venv-ag3s): 43 passed, 1 skipped in 21.16s   (skip = serve_policy test, flax 없음)
# == cpu regression exit status: 0
```
`tests/benchmark` = SL0 16 + SL1 28 (B 기준). 3.8 문법: 새/바뀐 3.8 파일 6개 모두 `.venv-libero` 에서 `ast.parse(feature_version=(3,8))` + `py_compile` 통과.

| test (`test_libero_sl1.py`) | 무엇을 |
|---|---|
| `test_wire_pack_matches_ag3s_pack_request` | 같은 입력으로 3.8 replica 와 `AG3S.trajopt.wire.pack_request` 의 camera key 집합·type·dtype·shape·값 동일, 경계값(음수→0, 0.4999 mm→0, 0.5 mm→0 (rint 짝수), 70 m→65535), 잘못된 stamp_mode 거부 |
| `test_wire_npz_roundtrip_unpacks_to_camera_observations` | npz exclusive 저장 → load → `strip_request` → `unpack_camera_observations` → `CameraID.HEAD/LEFT_WRIST`, depth = mm/1000, K·T·robot_state·stamp 보존, `unpack_stamps().mode == sim_frozen` |
| `test_wire_from_geometry_npz_layout` | geom npz → frame 별 wire npz export, 덮어쓰기 거부 |
| `test_policy_pixel_map_is_the_client_array_operation` | 각 pixel 에 자기 (u,v) 를 넣은 영상으로: policy 배열 = client 의 `raw[::-1,::-1]`, 사상·역사상이 정확, policy256 = OpenCV 의 열 반전 (배열 대수) |
| `test_policy_pixel_map_matches_resize_with_pad` ×3 | 256→224, 256×320→224, 240×180→224 (pad 포함): 좌표 ramp 를 openpi_client 의 resize 코드에 통과시켜 사상과 0.1 px 이내 (PIL antialias bilinear 의 ramp 오차 ≤ 0.064 px; 다른 규약 `x·s` 는 0.10–0.23 px 어긋나므로 구별됨) |
| `test_projection_inverts_backprojection` | `project_base_points` 가 AG3S `backproject` 의 역 (uv 1e-9 px) |
| `test_conventions_and_camera_table` | table 이 AG3S `CameraID` 가 받는 이름, K 공식 |
| `test_py38_adapter_checks` ×6 + `test_py38_fixture_unpacks_in_ag3s` | `.venv-libero` subprocess 결과(아래) + 실제 frame 의 wire npz 를 3.11 에서 unpack: 두 camera, 양자화 오차 ≤ 0.5 mm, K/T/robot_state bit 일치, stamp 하나 |
| `test_get_libero_env_with_depth_differs_only_by_camera_depths` | 공식 `_get_libero_env` 와 AST 비교, 차이는 `camera_depths: True` 하나 |
| `test_geometry_flags_default_off_and_dependencies` | 세 flag 기본 off, seg/float 은 geometry 없이 거부 |
| `test_run_episode_observer_sees_every_observation` | observer 가 `rec.observe` 와 같은 순간·같은 횟수 |
| `test_run_tasks_geometry_off_writes_exactly_the_sl0_outputs` | off: `_get_libero_env` 사용, observable 설치 안 함, episodes 에 `<stem>.npz` 만, jsonl key 집합 = SL0 의 25개 그대로 |
| `test_run_tasks_geometry_on_writes_geom_npz` | on: depth env → observable 설치 순서, `_geom.npz`, jsonl `geometry` (frames = step+1) |
| `test_replay_policy_reproduces_the_recorded_episode` | 가짜 env 에서 기록 → replay: `env.step` 에 들어간 python float 동일, recorder 배열 전부 동일 |
| `test_replay_refuses_inconsistent_records_and_never_invents_actions` | action 1e-12 변조 거부, replan 불일치 거부, 더 길게 가면 `execution_error "replay exhausted"` |
| `test_sl0_record_is_replayable` | 실제 SL0 npz(읽기만) 가 replay 규칙과 bit 일치 (storage 없으면 skip) |
| `test_replay_launcher_dry_run`, `test_replay_validates_source_and_protocol` | settings 의 source run id·경로·SHA-256, argv 가 replay parser 를 통과, run 미생성, seed/없는 episode/source 자신에 쓰기 거부 |
| `test_launcher_manifest_records_schedule_and_not_applicable` | flag 전달, `runtime_schedule.control_freq_hz == 20`(LIBERO source), 0.25 s, n/a 항목, off 이면 argv 끝이 `--seed 7`·`step == SL0`, AST 파서 |
| `test_serve_policy_records_the_served_checkpoint` | checkpoint: `env None`, `policy={checkpoint, pi05_libero, dir}`; default(`--env libero`): `env libero`, gs:// default (`.venv-openpi-live` 에서만) |

`libero_py38_checks.py` (`.venv-libero`, Python 3.8.20, numpy 1.22.4) 6 항목 모두 ok: camera table/key·shape; K = fovy 식 (`f=0.5H/tan(fovy/2)`, head fovy 45° f 309.019 px, wrist fovy 75° f 166.813 px) 이고 robosuite 와 bit 일치; `T_world_cam` 이 robosuite extrinsic 과 차 0, depth 차 ≤ 3.2e-5 m; `T_base_cam` 직교성 (8 step, 움직이는 action 포함: `RᵀR−I` ≤ 3.6e-15, `det−1` ≤ 5.3e-15, `inv(T_world_base)·T_world_cam` 과 차 0, 마지막 행 `[0,0,0,1]`); depth row 순서(table plane, 아래 기대 절에 수치); client `run_episode` + `_GeometryRecorder` 14 frame: OpenCV RGB = 기록 raw `[::-1]` bit 일치, depth uint16, robot_state = obs `robot0_joint_pos`/`gripper_qpos` bit 일치, 초기 frame 의 capture 시각 = return 시각.

**Smoke (policy·GPU 없음, 임시 storage root `scratchpad/storage`, SL0 run 의 manifest·jsonl·npz 를 복사해 source 로 씀)**:

| 실행 | 결과 |
|---|---|
| `launch_libero_replay.py --source-run <SL0> --capture-geometry --capture-segmentation --capture-float-depth --run-id smoke-geom` | exit 0, 46 s, jsonl `status=success`, env step 113, 호출 21, geometry frames 114, 영상 1. `episodes/…_geom.npz` 29.7 MB (seg+float 포함), key 36개 |
| 같은 명령, flag 없음 (`smoke-off`) | exit 0, 36 s, `success`, 113, 21, jsonl 에 `geometry` 없음, episodes 에 `<stem>.npz` 만 |
| `.venv-libero` 로 `wire_pack.py --geom … --frames 0,113` → 3.11 unpack | 2 frame × 두 camera `CameraObservation` 생성 (24–26 KB/frame) |

결과는 scratchpad 에만 있고 실제 `AG3S_VLA-storage/results` 에는 쓰지 않았다. 두 replay 의 출력과 SL0 원 기록의 비교(#1, #7)는 하지 않았다 — verifier 몫.

## verifier 가 알아야 할 것

- **실행 명령** (GPU·server 불필요, 각 1 process, thread 4):
  ```bash
  cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
  SRC=20261012T045129_6ba81024d243_sl0-pi05only-t0e0
  # (a) #1·#4·#5·#6: depth on, segmentation 없음 (render 추가 없음)
  .venv-ag3s/bin/python benchmark/runners/launch_libero_replay.py --source-run $SRC --task-ids 0 --episode-ids 0 \
      --capture-geometry --capture-float-depth --run-id sl1-replay-geom-t0e0
  # (b) #2·#3: segmentation 포함 (step 마다 camera 당 render 1회 추가)
  .venv-ag3s/bin/python benchmark/runners/launch_libero_replay.py --source-run $SRC --task-ids 0 --episode-ids 0 \
      --capture-geometry --capture-float-depth --capture-segmentation --run-id sl1-replay-seg-t0e0
  # (c) #7 SL0 경로 보존: flag 없음
  .venv-ag3s/bin/python benchmark/runners/launch_libero_replay.py --source-run $SRC --task-ids 0 --episode-ids 0 \
      --run-id sl1-replay-off-t0e0
  # #5: 3.8 pack (frame 전부) → 3.11 unpack
  env -u PYTHONPATH .venv-libero/bin/python benchmark/adapters/libero_panda/wire_pack.py \
      --geom <run(a)>/episodes/libero_spatial_t0_e0_geom.npz --out-dir outputs/verify/SL1/wire --frames all
  # #7 회귀
  bash scripts/testing/run_cpu_regression.sh
  ```
  stdout 첫 줄이 새 run 경로. `--dry-run` 은 아무것도 만들지 않고 settings·argv 만 출력. 각 replay ≈ 35–50 s.
  그 run 아래: `run_manifest.json`(`settings.replay_source`: source run id·경로·commit·manifest/jsonl/npz SHA-256; `executions` 에 `replay_client`·`launcher_result`), `logs/libero_replay.log`, `episodes.jsonl`(`replay`, `geometry` key), `episodes/libero_spatial_t0_e0.npz`(SL0 와 같은 형식), `…_geom.npz`, `videos/…mp4`, `source.patch`.
- **`_geom.npz` 내용**: `cameras`=[head,left_wrist], `source_cameras`, `obs_index`(=episode npz 의 obs 순서, 114), camera 별 `depth_mm/<cam>` (N,256,256) uint16 · `K/<cam>` · `T_base_cam/<cam>` · `T_world_cam/<cam>` · `robot_state/<cam>` (N,9) · `stamp/<cam>`, 선택 `depth_m/<cam>` float32 · `segmentation_geom/<cam>` int16(geom id, −1 = geom 아님). 공통 `T_world_base`, **`sim_state` = image 순간**의 robosuite flattened `[time,qpos,qvel]` (92), `sim_state_at_return` = `env.step` 이 돌아온 순간, `depth_near_m/far_m`, `fovy_deg/<cam>`, `model_geom_names/bodyid`, `model_body_names`, `model_joint_names`, `robot_qpos_indexes`, `conventions_json`. 모든 이미지·depth 는 OpenCV row order.
- **capture 순간**: 물체 표면 대조(#2)에서 MuJoCo geom pose 는 `sim_state`(image 순간)로 복원해야 한다 (`env.sim.set_state_from_flattened(sim_state); env.sim.forward()`). `sim_state_at_return` 을 쓰면 1 substep 늦은 pose 다. segmentation 은 `sim_state_at_return` 순간에 render 된 것이다 (차이 #4).
- **K 규약**: `cx=cy=128` (위 차이 #5). pixel 중심이 정수 index 인 back-projection 에서는 광축이 127.5 다.
- **policy image**: 배열 대수상 `policy256[i,j] = opencv[i, 255−j]` (열 반전) 다음 `resize_with_pad` 256→224 (pad 0, scale 0.875). 사상 함수 `camera.opencv_to_policy_pixel` / `policy_to_opencv_pixel`. 그것이 물리적으로 180° 회전인지 수평 반전인지는 #3 의 측정으로 판정.
- base frame = MuJoCo body `robot0_base` (`robot0_link0` 과 같은 pose), 매 frame `data.xpos/xmat` 에서 읽음 (`T_world_base` 로 저장).
- 새로 생긴 플래그·기본값 변경: client·SL0 launcher·replay launcher 에 `--capture-geometry`, `--capture-segmentation`, `--capture-float-depth` (모두 기본 off). 기본값 변경 없음. SL0 launcher manifest settings 에 새 key 3개 (위 차이 #8). `serve_policy` manifest 항목의 `env`/`policy` 값 (위 diff).
- 재생산이 필요한 산출물: 없음 (SL0 run 은 읽기만).
- 옛 기록과 호환이 깨지는가: 없음. SL0 run 의 jsonl·npz 형식 그대로 읽힘(`test_sl0_record_is_replayable`). 이전 `serve_policy` manifest 의 `env:"aloha_sim"` 기록은 그대로 남아 있다 (고치지 않음).
- 자원: 측정 중 다른 session 의 process 는 건드리지 않았다. `libero_py38_checks.py` 와 replay 는 각각 LIBERO env 1개.

## 내가 기대하는 결과

<!-- verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다 -->
- #1: depth on replay 의 RGB·state 는 SL0 원 기록과 bit 단위 같고 done step 112 도 같을 것 (`camera_depths=True` 는 같은 render 의 depth buffer 를 같이 읽을 뿐이고, 추가 observable 은 읽기만 한다). smoke 의 episode npz 크기가 SL0 와 같은 17,599,260 bytes 였지만 내용 비교는 하지 않았다. segmentation 포함 run(b)도 같을 것으로 보지만 덜 확실하다.
- #2: table plane 은 unit check 에서 head camera 29,055 pixel 의 world z − table top(0.9 m) 이 median +1.94 mm, |p95| 2.87 mm 였다 (row 를 뒤집으면 median −454 mm, p95 1700 mm). +0.5 px principal point 오프셋이 이 수 mm 의 일부일 수 있다. 물체는 수 mm 이내를 기대.
- #3: 투영이 OpenCV 영상에서 물체에 떨어지고, policy image 는 OpenCV 의 수평 반전(배열 대수)이므로 "180° 회전"이 아니라 상하는 OpenCV 와 같고 좌우만 바뀐 영상일 것으로 본다 — 측정이 판정.
- #4: static 표면 wrist→agentview p50 수 mm. #5: 모든 frame 성공, 양자화 ≤ 0.5 mm. #6: base 가 table top 보다 약간 위(수 cm 이내)일 것. #7: off replay 출력 = SL0, 회귀 A 73 passed.
