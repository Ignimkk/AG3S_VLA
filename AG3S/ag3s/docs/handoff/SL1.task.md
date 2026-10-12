# SL1 — LIBERO Panda camera adapter: RGB/depth · K · T_base_cam → V11AB wire 관측

> writer: lead (A0) · 읽는 쪽: implementer, verifier · 작성 2026-10-12 KST
> branch `feature/panda-benchmark`, 시작 HEAD `13926d4d`. 계획: `doc/develope_PLAN_SUMMARY.md` §2.1 SL-1.
> 사용자 판정 2026-10-12: SL0 확정, SL-1 진행. depth 활성화가 policy RGB 입력을 바꾸지 않는지 bit 단위 확인을 통과 조건에 포함.

## 왜 지금 이것인가

SL0(pi0.5-only가 공식 LIBERO Spatial 1 episode를 완주·기록)으로 기준 경로가 섰다. V11AB safety layer는
RGB가 아니라 **depth + camera 기하**로 scene을 만든다. LIBERO/robosuite 관측을 V11AB가 읽는 형식으로
정확히 옮기는 것이 SL-2(Panda geometry/self-filter), SL-3(attention), SL-4(ATS/LTE/ESDF)의 전제다.

```
robosuite render (OpenGL row order, depth = normalized z-buffer [0,1])
  → [SL1 adapter] OpenCV row order · metric z-depth(m) · K(fovy) · T_base_cam = inv(T_world_base)·T_world_cam(OpenCV axes)
  → V11AB wire keys: ag3s/cameras, ag3s/depth/<cam>(uint16 mm), ag3s/K/<cam>, ag3s/T_base_cam/<cam>,
                     ag3s/robot_state/<cam>, ag3s/stamp/<cam>, ag3s/stamp_mode=sim_frozen
  → AG3S.trajopt.wire.unpack_camera_observations → CameraObservation (OpenCV, robot base frame)
```

## 확인된 사실 (audit, read-only)

| 항목 | 근거 |
|---|---|
| wire 형식 | `AG3S/trajopt/wire.py:8-31`, depth mm 변환 `:413`, `:452-457`, unpack `:529-558` |
| camera 이름 | `AG3S/ag3s/types.py:216-224` (`head`/`left_wrist`/`right_wrist`만 허용), `mujoco_source.py:55-60` |
| frame 규약 | OpenCV(+z forward, +y down), robot base frame; RB-Y1 기준식 `mujoco_source.py:229-250` |
| depth 범위 | `AG3S/ag3s/config.py:534-542` (0.05–3.0 m); ESDF box 기본 base-frame z∈[0,1.6] `fields/esdf.py:777-790` |
| robosuite depth | `camera_depths=True` → `<cam>_depth` (H,W,1) float32 normalized, OpenGL bottom-up (`macros.IMAGE_CONVENTION`) |
| robosuite 기하 | `robosuite/utils/camera_utils.py`: `get_camera_intrinsic_matrix`, `get_camera_extrinsic_matrix`(camera→world, OpenCV 축), `get_real_depth_map`, `get_camera_segmentation` |
| client 삽입점 | `benchmark/runners/libero_pi05_client.py:135-145` env 생성, `:175` recorder, `:237-262` 관측/요청 |

## 구현에 요청하는 것 (→ implementer)

### A. Camera adapter (`benchmark/adapters/libero_panda/`, **Python 3.8 호환**, `.venv-libero`에서 실행)
1. `camera.py`: obs + `env.sim` → camera별 `{depth_m (H,W) float32 OpenCV row order, K (3,3), T_base_cam (4,4), T_world_cam, T_world_base, robot_state (q7 + gripper qpos2), stamp}`.
   - depth: `get_real_depth_map` 후 OpenGL→OpenCV row flip. RGB도 같은 OpenCV 배열을 별도로 낸다.
   - `T_world_base`는 `robot0_base` body pose(MuJoCo `data.xpos/xmat`). base frame 정의를 코드 주석·manifest에 명시.
2. `wire_pack.py`: 위 결과를 V11AB wire key/dtype 그대로 담은 dict(npz 저장 가능)로 변환. AG3S 모듈을 import하지 않는 3.8 복제이므로
   **cross-env test**로 동치성을 보장한다(아래 C).
3. Camera 이름 table (adapter 내부, AG3S core 수정 금지): `agentview → head`, `robot0_eye_in_hand → left_wrist`.
   근거: pi05_libero transform이 wrist image를 `left_wrist_0_rgb` 슬롯에 넣으므로 SL-3 attention token 순서와 맞춘다. 표를 manifest에 기록.
4. Policy image ↔ depth pixel 대응을 **명시적 함수**로: raw(OpenGL) → policy image(`[::-1, ::-1]` + resize_with_pad 256→224)와
   OpenCV depth pixel (u,v) 사이의 변환과 역변환. 회전/좌우 반전이 실제로 무엇인지는 verifier가 측정으로 판정한다(아래 #3).

### B. Client/launcher 연결 (기존 SL0 경로 보존)
5. `libero_pi05_client.py`에 `--capture-geometry`(default off) 추가: 켜면 `camera_depths=True`로 env를 만들고 step마다 A의 결과를
   `episodes/<...>_geom.npz`에 기록(depth는 uint16 mm로 저장 + K/T/robot_state). **off일 때 SL0과 동작·출력이 동일**해야 한다.
6. **Action replay 도구** `benchmark/runners/libero_replay.py`(3.8): 기존 run의 episode npz에서 실행 action을 읽어 policy 없이 같은
   task/init state/seed로 재생하고, 새 run에 geometry를 기록. 새 run은 launcher와 같은 `create_run` 경로로 만든다(3.11 쪽 wrapper가
   run을 만들고 3.8 replay를 실행). Replay 대상 run의 run_id/경로를 manifest에 기록.

### C. Test (CPU)
7. `.venv-ag3s`: 3.8 adapter가 저장한 wire npz fixture를 `AG3S.trajopt.wire.unpack_camera_observations`로 읽어
   `CameraObservation`이 생성되는지, depth mm↔m, K, T 값이 보존되는지. Fixture는 test가 `.venv-libero`를 subprocess로 불러 생성하거나,
   작은 고정 fixture를 tests에 둔다(대용량 금지).
8. `.venv-libero`: K가 fovy 식과 일치, `T_base_cam` 직교성(R^T R=I, det=+1), depth flip 방향(알려진 높이의 table plane 픽셀 행 순서) 단위 test.

### D. SL0 후속 정리 (사용자 확정 항목)
9. Run manifest에 pi0.5-only 조건의 **해당 없음 항목을 명시**: protected/authorized links, phase, level, runtime schedule(control 20 Hz 여부는
   LIBERO source에서 읽은 값으로, replan 5 step, policy 호출 주기). `serve_policy`가 manifest에 남기는 자기 항목이 `env="aloha_sim"`(Args default)로
   기록되는 문제를 실제 사용 경로(`policy:checkpoint`, config/dir)로 바로잡는다. `third_party/openpi/scripts/serve_policy.py` 최소 변경, impl.md에 diff.
10. CPU 회귀 명령을 고정: `scripts/testing/run_cpu_regression.sh`(또는 기존 위치 관례) — SL0 verifier가 재구성한 5개 파일
    (`tests/test_run_artifacts.py`, `tests/test_policy_recorder_destination.py`, `tests/ag3s/test_record_timing.py`, `tests/ag3s/test_frame_record.py`,
    `tests/trajopt/test_t43ta_path_shape.py`; `.venv-openpi-live` + `run_without_legacy.py`) + `tests/benchmark`.

**손대지 않을 것:** `AG3S/**` algorithm·config(V11AB), `robots/**`, `benchmark/LIBERO` 등 submodule, storage 기존 input.
Panda geometry/self-filter(SL-2), attention(SL-3), ESDF(SL-4)는 이번 범위 아님. GPU 불필요.

## 측정에 요청하는 것 (→ verifier)

Replay 대상: SL0 run `20261012T045129_6ba81024d243_sl0-pi05only-t0e0`의 task 0 / episode 0.

| # | 무엇을 재나 | 어떤 숫자로 | 통과 기준 |
|---|---|---|---|
| 1 | Replay 재현성 + depth 활성화 RGB 불변 | depth on replay vs SL0 원 기록: agentview/wrist RGB, eef/joint/gripper state, done step의 최대 차이 | **RGB bit-identical**, state 차이 0, done step 동일. 불일치 시 첫 불일치 step·key |
| 2 | depth 정합 | segmentation으로 고른 물체(target bowl, plate 등) pixel을 back-project한 base-frame 점 vs MuJoCo geom 표면/중심; table 면 z | 물체별 median 거리(mm), table plane 잔차 RMS(mm). 기준은 측정 후 lead·사용자 판정 |
| 3 | policy image ↔ OpenCV pixel 대응 | 물체 중심을 K·T로 투영한 (u,v)가 OpenCV RGB/segmentation에서 그 물체에 떨어지는지; 같은 점이 policy image(`[::-1,::-1]`+resize)에서 어디인지 | 투영 hit 여부 + pixel 오차; 회전/반전 판정(180° 회전인지, 수평 반전 포함인지) |
| 4 | 두 camera 일관성 | static 표면(table, plate)에서 wrist 점 → agentview 점 nearest 거리 분포 | p50/p95 (mm) |
| 5 | wire 왕복 | 3.8 pack → 3.11 `unpack_camera_observations` | 모든 camera·frame 성공, depth 양자화 오차 ≤ 0.5 mm |
| 6 | V11AB 범위 정보(측정만) | base frame에서 table 면 z, 물체 bbox, robot base 높이; depth 0.05–3.0 m 밖 pixel 비율 | 숫자만 (SL-4 ESDF box 판정 자료) |
| 7 | SL0 경로 보존·회귀 | `--capture-geometry` off로 SL0 replay 시 출력 동일; 고정된 CPU 회귀 명령 | pass 수 기존과 동일 |

- figure 3종 (`sl1-*`, 기존 이름과 겹치지 않게): (1) 실제 scene — 두 camera RGB 위 depth/back-projected 점·물체 투영 overlay와
  base frame 3D point cloud(두 camera 색 구분) + camera frustum, (2) graph — #2/#4 거리 분포, replay 차이 0 확인, (3) table — 변환 규약·K·T·측정 결과.
- raw: `outputs/verify/SL1/`. Replay run 경로를 verify.json에 기록.

## 이 STEP 이 답하는 질문

LIBERO Panda의 두 camera 관측을, policy 입력을 바꾸지 않고, V11AB가 읽는 OpenCV/robot-base 규약의 metric depth·K·T로
정확하게(물체·table에 mm 수준으로) 옮길 수 있는가?

## 되돌아올 지점

- #1 RGB 불일치 → depth 활성화가 render를 바꾸는 것이다. 그때는 geometry를 policy와 **별도 render pass**(같은 state에서 depth 전용)로
  얻는 방식으로 전환하고 사용자에게 보고한다.
- #2 오차가 voxel(V11AB fine 5 mm, coarse 20 mm) 이상 → near/far·fovy·base pose 정의를 재점검. 원인 확정 전 SL-2로 넘어가지 않는다.
- Camera 이름(`left_wrist`) 선택은 SL-3에서 token 순서 audit 결과와 다르면 adapter table만 바꾼다.
