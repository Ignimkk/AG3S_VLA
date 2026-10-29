# AG3S-PRE-CORE-TO E3-b — 구현 (GPU batched multi-seed, cuRobo L-BFGS)

> writer: ag3s-implementer · 읽는 쪽: lead, verifier, scribe
> 작성: 2026-10-26 12:40 KST (`date -u` 03:40 UTC + 9 h). 기준 코드: `feature/ag3s-development` HEAD `b3b69311` + E1 · E2 작업 트리 + E3-a 작업 트리(병렬, 다른 implementer) + 이 변경 (commit 안 함).
> 명세: [`AG3S-PRE-CORE-TO.task.md`](AG3S-PRE-CORE-TO.task.md) §6 "E3 구현 명세" E3-b 줄, §2 T63 · T63b 줄 (VLA 추종 항 없는 cuRobo = 0/10, 다시 돌리지 않음). E2 방식은 [`AG3S-PRE-CORE-TO-E2.impl.md`](AG3S-PRE-CORE-TO-E2.impl.md).

## 무엇을 했나 (평이한 요약 먼저)

기본 꺼짐 flag `--to-gpu-seeds {off,curobo}` (config `sqp.gpu_seeds`) 를 넣었다. `--to-multi-seed N` 과 함께 켜면, E2 가 발동하는 chunk (표준 풀이가 0.1 mm 를 넘는 침범을 남긴 chunk) 에서 E2 와 **같은 N 개 seed** (`multi_seed.seed_specs` → `_build_seeds`) 를 CPU SQP 로 하나씩 푸는 대신, **cuRobo 의 L-BFGS 한 번으로 한 batch 에서 함께** 최적화한다. 비용은 우리 `w_track` 과 같은 꼴의 관절 VLA 추종 항이 첫째이고 (T63b 의 0/10 원인으로 본 "추종 항 없음" 은 없다), smoothness · continuity · deflection rate · 관절 한계 hinge · 충돌 (우리 거리장을 T63 `composed_world` 의 `b0m` world 로, 10 mm 여유 포함) · 지지면 · (앵커가 목표 행을 가질 때만) 목표 항이다. GPU 결과 N 개는 위치 box 와 첫 행 anchor 로 clip 한 뒤 **우리 `_finish` (모든 반환 궤적이 거치는 전체 해상도 검사)** 로 하나하나 다시 재고, E2 의 `select_moving_first` 가 표준 풀이(후보 0, 손대지 않음)와 N 개 중 하나를 고른다.
꺼져 있으면 아무것도 바뀌지 않는다: E2 를 켠 fixture 5 경우의 궤적 SHA 가 E3-b 이전 `sqp.py` (이 파일에서 E3-b 두 hunk 를 지운 사본) 의 값과 같고, 기록에 새 키가 없고, `gpu_seeds` 모듈을 import 조차 하지 않는다 (두 venv 에서 test). GPU 경로가 실패하면 (CUDA 없음 · cuRobo 없음 · 모델이 c-space 로 안 됨 · 거리장 없음 등) E2 의 CPU seed 로 그대로 풀고 기록에 이유를 남긴다 (결과 SHA 는 꺼짐과 같음).
S-OCC smoke 1 episode (case3 L3 s13453, E0 침범 35 chunk): `E3b_16` 이 35 / 35 를 침범 0 으로 풀었다 (`E2_16` 도 35 / 35). 시간은 판정이 아니다 (공유 lock, 아래 표).

## 바뀐 파일

| 파일:줄 | 무엇이 | 왜 |
|---|---|---|
| `AG3S/trajopt/gpu_seeds.py` (새 파일, ~750 줄) | 상수 (`LBFGS_SETTINGS` :70, `CUDA_GRAPH` :79, `COLLISION_WEIGHT_SCALE` :81, `ACTIVATION_M` :83, `LIMIT_WEIGHT` :85, `LIMIT_SHRINK` :87, `POSITION_INSET_RAD` :89, `WORLD_GAPS` :98), `robot_spec` (:111, 살아 있는 `UrdfSphereChain` + TO `JointLimits` → T63 spec), `available` (:154), `_act` (:174, cuRobo 충돌 activation), `_Rollout` (:185, cuRobo rollout protocol; `terms` :248 = 모든 비용 항), `CuroboSeedBatch` (:299; `_build` :314, `_load` :422 = chunk 마다 buffer · world 복사, `solve` :529), `batch_for` (:589, optimizer 마다 (N, H) 한 번 build, 실패도 cache), `anchor_terms_for` (:603), `solve_seeds` (:614, GPU → clip → `check` → 기록), `check` (:674, `_finish` 그대로), `warm_up` (:691, 합성 world 로 build + CUDA graph capture) | E3-b 본체. CUDA 가 필요한 부분은 함수 안에서만 import (모듈 import 는 가벼움) |
| `AG3S/trajopt/sqp.py:139–140` | `self.gpu_seeds` | flag 읽기 |
| `AG3S/trajopt/sqp.py:460–481` | `_run_seeds` 안: `gpu_seeds != "off"` 이면 `GS.solve_seeds(...)` 로 후보를 만들고 E2 와 같은 `describe` 로 기록 · 자격 · key 를 만든다. 그 앞뒤로 `_anchor_state` · `_track_scale` 를 seed 풀이와 같게 둔다 (`_finish` 의 `cost` 가 같은 식). CPU seed loop 는 `gpu is None or gpu["results"] is None` 일 때만 (`:481`, 꺼짐이면 예전 loop 그대로) | E2 의 seed 를 푸는 자리 하나만 바꿈. 선택 · 상태 복원 (`_previous`, `frame_index`) · notes · `solve_time_ms` 는 E2 코드 그대로 |
| `AG3S/trajopt/config.py:712–717, 722–725` | `SqpConfig.gpu_seeds: str = "off"` + 검증 (`off`/`curobo` 만, `curobo` 는 `multi_seed > 0` 필요) | config 자리. 창 optimizer 는 `with_overrides` 로 물려받는다 |
| `AG3S/trajopt/serve_safe.py:1833–1838` | `--to-gpu-seeds {off,curobo}` (기본 off, help 한국어) | |
| `AG3S/trajopt/serve_safe.py:2827–2835` | `curobo` 인데 `--no-safe` 이거나 `--to-multi-seed` 가 0 → `ap.error`; `gpu_seeds.available()` 가 아니면 (torch · CUDA · cuRobo) checkpoint 를 올리기 전에 `ap.error` | E3-a 의 `osqp_cuda_available` 과 같은 방식 |
| `AG3S/trajopt/serve_safe.py:2893–2894` | `sqp_overrides`: 켰을 때만 `{"gpu_seeds": "curobo"}` | off 이면 키 없음 |
| `AG3S/trajopt/serve_safe.py:3202–3206` | 시작 log (WARNING, 켰을 때만) | |
| `AG3S/trajopt/serve_safe.py:3621–3625` | `SafePolicy` 를 만든 뒤 `gpu_seeds.warm_up(served.refiner.optimizer)` (켰을 때만) | build (~6.5 s) 와 CUDA graph capture (~3.6 s + 1.2 s) 를 첫 chunk 전에 끝낸다. 첫 발동 chunk 가 수 초 멈추지 않게 |
| `AG3S/trajopt/experiments/tp_timing.py:315–321` | GPU setting 이 있으면 refiner 를 만든 뒤 `warm_up`, 출력에 `gpu_seeds_warm_up` | 서버와 같은 조건에서 첫 chunk 를 잰다. GPU setting 이 없으면 출력 그대로 |
| `AG3S/trajopt/experiments/e3b_gpu_seeds.py` (새 파일) | `SETTINGS` = `e2_multi_seed.SETTINGS` 전부 + `E3b_16/32/64`, `E1C_E3b_16`, smoke 전용 `SMOKE_IT1_E3b_16`; `summarize [--base NAME]` = `e2_multi_seed.summarize` 그대로 + `per_setting[v]["gpu"]` (backend 개수, `gpu_ms` · `optimize_ms` · `load_ms` · `check_ms` · `total_ms` 의 median · p95 · max) | offline 비교. E2 표의 키 · 값은 바뀌지 않는다 (test) |
| `tests/trajopt/test_e3b_gpu_seeds.py` (새 파일) | 6 test (아래) | |

`multi_seed.py`, `qp.py` (E3-a 소유), `refiner.py`, `curobo_v2/**` 는 건드리지 않았다.

## GPU 문제 (seed 마다 하나, 변수 = 계획 H 행 × 최적화 관절 14, 우리 SQP 와 같은 변수)

| 항 | 값 (`D = Q − R`, `R` = 정책 reference) | 가중치 |
|---|---|---|
| tracking | `Σ_k decay^k Σ_j s_j D_kj²` — `problem.objective` 의 항, IFG-TOW scale `s` 포함 | `cost.w_track` (V11AB 1.0) |
| smoothness | `Σ_k ‖Q_{k+2} − 2Q_{k+1} + Q_k‖²` | `cost.w_smooth` (0.05) |
| continuity | `Σ_{k<L} ‖Q_k − P_k‖²` (이전 refined chunk, `_solve` 처럼 look-ahead 창으로 자름) | `cost.w_continuity` (0.5) |
| deflection rate (T43 TA B) | `Σ_k ‖D_{k+1} − D_k‖² + ‖D_0 − D_prev‖²` (history 있을 때만 경계 항) | `cost.w_deflection_rate` (V11AB 20) |
| 관절 한계 | 제곱 hinge: 위치 box (안쪽 1e-3 rad), `|Q_k − q_now|` ≤ 첫 행 anchor (`max_step`) · rest envelope, 속도 · 가속도 ≤ 0.97 × 한계 (`velocity_rows` 가 아니면 속도 없음) | `LIMIT_WEIGHT` = 1e6 / rad² |
| 충돌 | cuRobo 구 비용 `w · act(r + e + m + η − sdf)` (구 × 행, 아래 world). `act` = 0 → η 안에서 2 차 → 선형 | `cost.w_slack × 10` = 1e4 / m, η = 5 mm |
| 지지면 | 같은 activation 을 우리 plane 행 `n·p − o − r` 에 (look-ahead 꼬리 행 제외, `_clearances_from` 과 같음) | 같은 1e4 |
| 목표 (앵커에 이 horizon 의 목표 행이 있을 때만) | 앵커 자신의 항 (`grasp_anchor.anchor_targets`): 목표 행 `w_goal ‖p_k − g‖²` (+ IFG-TOA 경로 항이 켜져 있으면 `w_tcp ‖p_k − p_ref,k‖²`), `p` = 앵커 link 원점 (cuRobo tool frame `ee_left` · `ee_right`) | 앵커의 `w_goal` (E1-C 2000) / `w_tcp` |

- **cuRobo 에서 쓰는 것:** kinematics (T63 parity, 우리 구 순서는 `sphere_permutation`), 충돌 kernel (`composed_world` 경유), `curobo.optim.LBFGSOpt` (line search · two-loop 가 GPU kernel, CUDA graph). **쓰지 않는 것:** cuRobo 의 c-space 목표 · regularization · B-spline · 재시간화 (T63/T63b 의 `TrajectoryOptimizer`). 그래서 계획은 우리 15 Hz 행 위에 그대로 있다.
- **L-BFGS:** `trajopt/lbfgs_bspline_trajopt.yml` 의 line search (`[0, 0.1, 0.5, 1.0]`, approx Wolfe), `step_scale 0.98`, `epsilon 0.01`, `initial_step_scale 0.001` + 이 크기에 맞춘 `history 15`, `num_iters 100` (`inner_iters 25`, `fixed_iters`). MPPI 는 넣지 않았다 (선택 사항).
- **TF32:** cuRobo 는 import 때 TF32 matmul 을 전역으로 켠다. build · load · solve · `available` 마다 앞뒤 값을 저장 · 복원한다. (이 pod 의 `.venv-openpi-live` 는 처음부터 `True` 였다 — 복원은 "원래 값" 으로.) rollout 의 좌표 변환은 원소별 곱 (matmul 아님).

## World (T63 `composed_world`, mode `b0m`, chunk 마다 살아 있는 거리장에서)

target 제외 층(`esdf_free`)의 lattice 안에서는 그 층, 밖에서는 coarse 층; target 제외 층이 없는 chunk 는 fine lattice 안에서 `min(coarse, fine)` (우리 주 거리장). 구마다 반지름 `r + esdf_margin + m` 로 묻는다: `m` = `obstacle_margin − esdf_margin` (≥ 0) 을 우리 `_obstacle_rows` 가 `obstacle` 이라 부르는 행에만 (`b0m.rules_from_scene` 이 같은 `SceneSnapshot` 에서 읽음; target · support 행 0). 그래서 obstacle 행은 `max(esdf_margin, obstacle_margin)` = 10 mm, 나머지 행은 `esdf_margin` — 우리 규칙과 같다. 손가락 opening 은 TO parameter path 의 0 행 (`serving.finger_values`), URDF 범위로 clip.

**우리 거리장 · 권한과 다른 점 (`WORLD_GAPS`, 모든 GPU 기록 `gpu.settings.gaps` 에 들어감):**

1. target 제외 층을 lattice 안의 **모든 로봇 행**이 본다. 우리 TO 는 `target_free_mask` 가 허락한 행(권한 link)만 그 층을 본다 → GPU 에서는 권한 없는 link 가 target 에 밀려나지 않는다. (`composed_world` 의 `lte` mode 가 행별 제외를 갖지만 obstacle 여유가 없다; 둘을 합친 mode 는 만들지 않았다.)
2. `manipulated_link_margin` (link 별 target 여유) 없음.
3. 손가락 opening 은 chunk 당 하나 (행마다의 envelope 아님).
4. destination label · neighbour contact (IFG-NBC) 완화 없음, 쥔 물체의 질의점(held / attached points) 없음.
5. servo-accel 비용 (T43 TA C), E1-B 목표 상자, 경로 행 (T43 Q path rows) 없음.

이 차이는 GPU 결과가 **선택에서 지게** 만들 수만 있다: 모든 GPU 결과는 `_finish` 로 다시 재고, 표준 풀이가 후보 0 이다. S-OCC smoke 의 최악 행은 모두 왼손 손가락 (`ee_finger_l1` / `l2`, URDF 이름) 이고 held 아님 (E2 offline 기록) — 1 · 4 가 걸리지 않는 경우다. `rules_from_scene` 이 지원하지 않는 scene (`scene.target_volume`, T43 T) 은 GPU 경로가 실패 → CPU seed.

## 기록 `to.metrics.multi_seed` (E2 키 + 켰을 때만 둘)

- `backend`: `"curobo"` 또는 `"sqp (cuRobo failed)"` (그때 CPU seed 로 풀었고 `gpu.error` · `gpu.traceback`).
- `gpu`: `build_ms` (이번 호출에서 build 했으면), `load_ms` (buffer · world 복사), `optimize_ms` (L-BFGS, CUDA sync 포함), `gpu_ms` (= load + optimize + 비용 평가), `cold`, `cost_seed` · `cost_out` · `collision_out` · `limits_out` (seed 별 GPU 비용), `finite`, `finger_values` · `finger_source`, `obstacle_margin_m` · `esdf_margin_m`, `goal_term` (+ `goal_note`), `world` (`box_layer` 등), `settings` (L-BFGS · 가중치 · gaps · tool frames), `check_ms` (N 개 `_finish`), `clip_rad` (seed 별 clip 최대 변화), `total_ms`.
- `candidates[i]` (i ≥ 1) 는 E2 와 같은 필드: `max_violation_mm` · `status` 는 **우리 `_finish`** 의 값, `sqp_iterations` 0, `solve_ms` = 그 후보의 검사 시간. `total_ms` 는 E2 처럼 seed 단계 전체 (GPU + 검사).
- 주의: `_finish` 를 `iterations = 0` 으로 부르므로 GPU 결과의 상태는 침범이 없으면 `optimal` 이다 (SQP 결과는 반복 상한에 닿아 `feasible`). 둘 다 `status.safe`.

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-development
# CPU (GPU test 2 개는 skip)
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=. \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt/test_e3b_gpu_seeds.py
# GPU (공유 lock)
python3 scripts/setup/gpu_lock.py shared --est-gib 16 --tag e3b-impl -- bash -c 'source scripts/setup/env.sh >/dev/null; \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu .venv-openpi-live/bin/python -m pytest -q -p no:cacheprovider \
  tests/trajopt/test_e3b_gpu_seeds.py'
# trajopt 전체
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYTHONPATH=. \
  .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/trajopt \
  --deselect tests/trajopt/test_t43t_replay.py::test_off_is_bit_identical_to_the_pre_fix_commit
```

- `.venv-ag3s`: **4 passed, 2 skipped** (GPU 2 개). `.venv-openpi-live` + GPU: **6 passed**. 모듈 `__file__` = 이 checkout (worktree 아님).
- trajopt 전체 (`.venv-ag3s`, 최종 코드): **1966 passed, 33 skipped, 1 deselected, 0 failed** (20 min 31 s; E3-a 의 작업 트리 변경도 포함된 상태). skip 33 중 2 개가 이 파일의 GPU test.

| test | 내용 |
|---|---|
| `test_off_is_bit_identical_to_the_pre_e3b_code` | E2 를 켠 5 경우 (candidate 막힌 fixture, ESDF 막힌 fixture, E1-C 앵커, refiner 2 chunk) 의 SHA 가 E3-b 이전 `sqp.py` 사본에서 잰 값과 같다 (numpy 주 버전별로 고정: `.venv-ag3s` numpy 2, `.venv-openpi-live` numpy 1 — 두 venv 는 같은 풀이의 마지막 bit 가 다르다). `gpu_seeds` import 를 막아도 돈다. 기록에 `backend` · `gpu` 없음 |
| `test_config_and_cli` | 기본 off, `cuda` · `True` · `multi_seed 0` 과 함께는 거절, 창 optimizer 상속, `--to-gpu-seeds` 는 켰을 때만 키, `--to-multi-seed` 없이 / `--no-safe` 와 함께 `ap.error`, (GPU 없는 venv) checkpoint 전에 `ap.error` |
| `test_a_gpu_path_that_cannot_run_falls_back_to_the_cpu_seeds` | 거리장 없는 candidate fixture · 전체 RB-Y1 모델 (base · torso 구 = `link_torso_5` 밖, c-space 불가) 에서 `backend "sqp (cuRobo failed)"`, 결과 · 기록이 꺼짐과 같음, 실패한 build 는 cache 되어 다음 chunk 에서 다시 build 하지 않음 |
| `test_gpu_seeds_resolve_the_stuck_fixture_and_the_choice_passes_our_check` (GPU) | ESDF 막힌 fixture (경로 정면 2 mm 옆 반지름 10 cm 구, SQP 1 반복: 기준 187 mm → 표준 47 mm 남음): GPU 후보 16 개 중 ≥ 9 해소 (측정 16 / 16), 고른 후보가 `stay` 아닌 해소 후보, **반환 궤적을 test 안에서 다시 잼** (`full_violation` ≤ 0.1 mm, `limit_report` ≤ 1e-3, 첫 행 anchor), 상태 복원, GPU 비용이 seed 보다 낮아짐, cuRobo FK 의 구 중심 = 우리 FK (< 0.01 mm), TF32 값 복원 |
| `test_gpu_seeds_are_deterministic_and_carry_the_goal` (GPU) | warm-up 한 optimizer 둘 · 같은 optimizer 의 reset 뒤 두 번: 같은 SHA · 같은 기록 · 같은 GPU 비용. E1-C 앵커 (목표 행 5–7) 가 GPU 문제에 들어감 (`goal_term`), 가중치 2000 이 가중치 0 보다 해소 후보의 최소 목표 오차를 줄임 (측정 159 → 105 mm; 이 sweep 은 세 행 동안 손바닥을 한 점에 둘 수 없을 만큼 빠르다) |
| `test_offline_settings_and_the_summary` | settings 등록 · config (`E3b_n` = `E2_n` + `gpu_seeds`), `summarize` 가 E2 표를 바꾸지 않고 `gpu` 통계를 더함 |

- 처음 build 에서 찾은 것 (고침): 손가락 구가 없는 모델에서 cuRobo `lock_joints` 가 실패 → 구가 있는 손가락 관절만 spec 에 넣음. 손바닥에 구가 없는 모델에서 `arm_6` 관절이 tree 에 없음 → tool frame 을 언제나 `ee_left` · `ee_right` 로 (URDF 에 있으면). 실패한 build 를 chunk 마다 다시 시도해 chunk 당 ~8 s 가 걸림 → 실패를 cache. 첫 실제 chunk 의 world load 305 ms (target channel 첫 사용) → warm-up 에 합성 target · 지지면을 넣어 25 ms.
- 처음 실행의 cold solve 는 warm 결과와 다르다 (CUDA graph capture 중 반복이 더 돈다). 서버 · `tp_timing` 은 warm-up 을 먼저 하므로 실제 chunk 는 모두 warm (`n_cold` 0).

## offline smoke (1 episode, 판정 아님)

S-OCC `OCC/AG3S_case3_L3_s13453` (46 chunk, E0 침범 35 chunk). **link 는 storage run 안에 만들지 않았다:** E3-a implementer 의 smoke 가 같은 시각에 `make_links.sh` 로 만든 link 를 쓰고 있어서 (pid 334825), 끝나는 순서에 따라 서로의 run 이 깨질 수 있었다. 대신 session scratchpad 에 같은 구조 (`<name>/server_proc.txt` 복사 + `sc/run_0000` → 실제 server_split) 와 그 한 entry 의 index (`occ_index_e3b.json`) 를 만들었다 — storage run 에는 link 를 하나도 만들지 않았고 지우지도 않았다. 인계 시점 (12:45 KST) 에 `E1/occ_index` 아래 link 는 0 개 (E3-a 가 지움; `find $RUN -type l` 에 남은 6 개는 원래 있던 `snapshot_*/third_party/openpi/third_party/{aloha,libero}`).

```bash
RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581
O=$RUN/verify/AG3S-PRE-CORE-TO/E3/b/offline_smoke
IDX=<scratchpad>/occ_links/occ_index_e3b.json     # 위 설명의 한 entry index (smoke 뒤 scratchpad link 도 지웠다; 다시 돌릴 때는 E1 의 occ_index.json + make_links.sh)
python3 scripts/setup/gpu_lock.py shared --est-gib 16 --tag e3b-impl -- bash -c 'source scripts/setup/env.sh >/dev/null; \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu MUJOCO_GL=osmesa .venv-openpi-live/bin/python \
  -m AG3S.trajopt.experiments.tp_timing --index '$IDX' --key OCC/AG3S_case3_L3_s13453 \
  --settings-module AG3S.trajopt.experiments.e3b_gpu_seeds --variants E0 E2_16 E3b_16 --field cuda --mode server \
  --replay-wrist-track --progress --out '$O'/AG3S_case3_L3_s13453.json'
.venv-ag3s/bin/python -m AG3S.trajopt.experiments.e3b_gpu_seeds summarize $O/AG3S_case3_L3_s13453.json > $O/summary.json
# 최종 코드로 다시 (E0 E3b_16): AG3S_case3_L3_s13453_rerun.json / summary_rerun.json
```

산출 (`$O`): `AG3S_case3_L3_s13453.json` · `.log` (E0 · E2_16 · E3b_16, 185 s), `summary.json`, `AG3S_case3_L3_s13453_rerun.json` · `.log` (최종 코드, E0 · E3b_16), `summary_rerun.json`.

| 지표 (E0 침범 35 chunk) | E0 | E2_16 (CPU) | E3b_16 (첫 실행) | E3b_16 (최종 코드) |
|---|---|---|---|---|
| 해소 (≤ 0.1 mm) | 0 | 35 | 35 | 35 |
| 해소 ∧ 목표 ≤ 10 mm | 0 | 35 | 35 | 35 |
| 목표가 있는 chunk | 0 | 0 | 0 | 0 |
| 고른 후보 종류 | — | detour 35 | noise 23 · detour 11 · policy 1 | 같음 |
| seed 단계 ms 중앙값 / p95 | — | 3519 / — | 443 / — | 428 / 445 |
| chunk `solve_ms` 중앙값 / p95 | 67.6 / 124.9 | 3591 / 4258 | 522 / 610 | 492 / 546 |
| `gpu.optimize_ms` 중앙값 | — | — | 154.6 | 151.3 |
| `gpu.load_ms` 중앙값 (최대) | — | — | 22.5 (305, 첫 chunk) | 21.8 (25.3) |
| `gpu.check_ms` 중앙값 (N = 16 `_finish`) | — | — | 253.9 | 241.4 |
| backend | — | — | curobo 35 | curobo 35 |

- **시간은 판정이 아니다:** 공유 lock, 같은 시각에 E3-a smoke (GPU) 와 내 pytest (CPU 1 process) 가 돌았다. 독점 lock 측정은 lead 몫.
- 결정성: 최종 코드 재실행의 E3b_16 궤적 SHA 가 첫 실행과 46 / 46 같다 (그 사이 바꾼 것은 warm-up 의 합성 target 과 손가락 clip 뿐; 첫 실행엔 E2_16 도 같은 pass 에 있었다). 발동하지 않은 11 chunk 는 E0 와 SHA 같음.
- warm-up (최종): H 16 build 6.4 s + capture 3.6 s, H 8 (창 optimizer) build 0.5 s + capture 1.2 s, 합 16.5 s (서버 시작에 한 번).
- seed 단계 ~430 ms 중 **검사(`_finish` × 16)가 ~240 ms 로 가장 크다** (GPU 는 ~180 ms). 아래 "확실하지 않은 것" 3.

## full 비교 명령 (lead 가 실행, 독점 lock)

`E2_16` 을 같은 pass 에 넣으면 같은 부하에서 나란히 잴 수 있지만 S-OCC 에서 pass 가 ~2 배 길어진다 (E2 기록이 이미 있으면 `E0 E3b_16` 만으로 충분하다 — `summarize` 는 같은 파일 안에서만 비교하므로 E2 와의 대조는 파일을 합쳐 본다). 아래는 E2 와 같은 꼴 (`E0 E2_16 E3b_16`). `--field cuda` · `.venv-openpi-live` 필수 (cuRobo + torch). `CUDA_VISIBLE_DEVICES` 를 비우지 않는다.

**S-OCC (OCC round 4, 18 run) — link 를 만들고, 끝나면 반드시 지운다 (다른 reader 가 없을 때, `pgrep -f occ_index.json`):**

```bash
RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581
IDX=$RUN/verify/AG3S-PRE-CORE-TO/E1/occ_index/occ_index.json
O=$RUN/verify/AG3S-PRE-CORE-TO/E3/b/offline_occ
mkdir -p $O
cd /mnt/dev/AG3S_VLA-development
bash $RUN/verify/AG3S-PRE-CORE-TO/E1/occ_index/make_links.sh
cat > $O/run.sh <<EOS
source /mnt/dev/AG3S_VLA-development/scripts/setup/env.sh >/dev/null
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu MUJOCO_GL=osmesa
for key in \$(python3 -c "import json;print(' '.join(json.load(open('$IDX'))['runs']))"); do
  n=\$(basename \$key)
  .venv-openpi-live/bin/python -m AG3S.trajopt.experiments.tp_timing --index $IDX --key \$key \
    --settings-module AG3S.trajopt.experiments.e3b_gpu_seeds --variants E0 E2_16 E3b_16 \
    --field cuda --mode server --replay-wrist-track --out $O/\$n.json > $O/\$n.log 2>&1 || echo "FAIL \$key" >> $O/fail.txt
done
EOS
python3 scripts/setup/gpu_lock.py exclusive --est-gib 16 --tag e3b-offline-occ -- bash $O/run.sh
bash $RUN/verify/AG3S-PRE-CORE-TO/E1/occ_index/make_links.sh --remove   # pgrep -f occ_index.json 가 비었을 때만
find $RUN -type l | head    # 비어 있어야 한다
.venv-ag3s/bin/python -m AG3S.trajopt.experiments.e3b_gpu_seeds summarize $O/*.json > $O/summary.json
touch $O/DONE
```

**S-REG (E4 18 장면, link 없음):**

```bash
RUN=/mnt/dev/AG3S_VLA-storage/results/development/20261025T191319+0900_35b909cf096a_cc215238f581
IDX=$RUN/verify/AG3S-GPU-STAGES/M2_cpu/e4_index.json
O=$RUN/verify/AG3S-PRE-CORE-TO/E3/b/offline_e4
mkdir -p $O
cd /mnt/dev/AG3S_VLA-development
cat > $O/run.sh <<EOS
source /mnt/dev/AG3S_VLA-development/scripts/setup/env.sh >/dev/null
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu MUJOCO_GL=osmesa
for key in \$(python3 -c "import json;print(' '.join(json.load(open('$IDX'))['runs']))"); do
  n=\$(basename \$key)
  .venv-openpi-live/bin/python -m AG3S.trajopt.experiments.tp_timing --index $IDX --key \$key \
    --settings-module AG3S.trajopt.experiments.e3b_gpu_seeds --variants E0 E2_16 E3b_16 \
    --field cuda --mode server --replay-wrist-track --out $O/\$n.json > $O/\$n.log 2>&1 || echo "FAIL \$key" >> $O/fail.txt
done
EOS
python3 scripts/setup/gpu_lock.py exclusive --est-gib 16 --tag e3b-offline-reg -- bash $O/run.sh
.venv-ag3s/bin/python -m AG3S.trajopt.experiments.e3b_gpu_seeds summarize $O/*.json > $O/summary.json
touch $O/DONE
```

- 추정 시간 (공유 lock smoke 기준, 판정 아님): episode 마다 warm-up ~17 s + chunk 당 E0 ~70 ms + E3b 발동 chunk ~0.45 s + E2_16 발동 chunk ~3.6 s. S-OCC (E0 침범 68 chunk / 967) ≈ 18 × 17 s + 967 × 0.15 s + 68 × 4 s ≈ 15 분, S-REG (침범 2 chunk) ≈ episode 당 ~60–90 s.
- `summary.json`: `per_setting["E3b_16"]` 에 E2 와 같은 키 (`n_resolved`, `share_resolved_goal_le_10mm`, `chosen_kind` …, `multi_seed_ms_median/p95`) + `gpu` (backend 개수 — `sqp (cuRobo failed)` 가 0 인지 먼저 본다 — 와 `gpu_ms` · `optimize_ms` · `load_ms` · `check_ms` · `total_ms` 의 median · p95 · max). §7 E3 판정의 "발동 chunk 시간" 은 E2 와 같은 `solve_ms` (표준 + seed 단계) 로 보는 것이 맞다고 생각한다 (`solve_ms_median` / `_p95`, 그 chunk 들).

## verifier 가 알아야 할 것

- 새 flag: `--to-gpu-seeds {off,curobo}` (기본 off). 새 config 키 `sqp.gpu_seeds` (기본 `"off"`) — `to_dict()` 에 키가 하나 는다 (E2 의 `sqp.multi_seed` 와 같은 방식). 서버 기록 meta 의 `sqp` 는 그대로.
- 꺼짐이면 응답 · 기록 · `to.metrics` 키가 같다 (E2 를 켠 경우도 같음). 켜면 발동 chunk 의 `to.metrics.multi_seed` 에 `backend` · `gpu` 가 생긴다. 서버 시작 log 에 warm-up 한 줄.
- `tp_timing` 출력: GPU setting 이 있으면 top-level `gpu_seeds_warm_up` 이 생긴다.
- 재생산이 필요한 산출물: 없음. 옛 기록과 호환: 깨지지 않음.
- 서버는 `.venv-openpi-live` 에서 cuRobo 를 import 할 수 있다 (`third_party/curobo` editable, `nvidia-curobo 0.8.0.post1.dev43`, numpy 1.26.4, torch 2.7.1+cu126; `gpu_seeds.available()` → `(True, "")` 확인). 실제 서버 (`serve_safe --to-gpu-seeds curobo`) 를 띄워 closed-loop 은 돌리지 않았다 — `SafePolicy` 뒤 warm-up 줄과 인자 검사만 test 했다.

## 확인하지 못한 것 · 확실하지 않은 것

1. **World 차이 1 (target 제외 층을 모든 행이 봄)** 은 권한 없는 link 가 target 쪽으로 가는 GPU 결과를 만들 수 있다. 우리 검사가 그 결과를 침범으로 재므로 고르지 않지만, 그런 결과가 많으면 해소 후보 수가 준다. S-OCC smoke 에서는 발동 chunk 당 해소 후보 13–16 / 16 이었다.
2. **`stay` seed 의 GPU 결과도 kind `stay` 로 기록된다.** GPU 의 추종 항 때문에 실제로는 움직일 수 있지만, E2 의 `select_moving_first` 를 같은 규칙으로 쓰려고 이름을 그대로 두었다 (첫 순위에서 빠짐). 그 후보를 잃는 것 외의 영향은 없다.
3. **시간의 가장 큰 몫은 우리 검사 16 번 (~240 ms) 이다.** 같은 선택을 더 적은 검사로 낼 수 있다: `select_moving_first` 의 첫 순위는 (목표 오차, reference 편차) 이고 둘 다 검사 없이 계산되므로, 그 순서로 검사해 처음 통과한 후보에서 멈추면 고르는 후보가 같다 (해소 후보가 없을 때만 전부 검사). 명세가 "GPU 결과 N 개를 모두 다시 잰다" 이므로 넣지 않았다 — 시간 기준을 못 넘으면 lead · 사용자 판단으로 넣을 수 있다 (기록에는 검사하지 않은 후보가 빈다).
4. 가중치 (`COLLISION_WEIGHT_SCALE` 10, η 5 mm, `LIMIT_WEIGHT` 1e6, 0.97, L-BFGS 100 반복) 는 fixture 와 smoke 1 episode 로만 보았다. 튜닝하지 않았다 (명세 밖).
5. 목표 항은 fixture 에서만 확인했다 (smoke episode 에는 목표가 있는 발동 chunk 가 없었다: `n_with_goal` 0). `E1C_E3b_16` setting 은 만들었지만 돌리지 않았다.
6. 손가락은 chunk 당 하나의 opening. 손이 닫히는 chunk (closing) 에서는 envelope 와 다를 수 있다 (검사가 잡음).
7. `rules_from_scene` 의 `NotImplementedError` (`target_volume`) 이나 지지면 > 8 개는 GPU 경로 실패 → CPU seed (느림). S-OCC · S-REG 의 서버 설정에서 이것이 생기는지는 full run 의 `gpu.backend` 개수로 본다.
8. 그림은 만들지 않았다 (verifier 몫).

## 내가 기대하는 결과

> verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.

- S-OCC 68 chunk 에서 `E3b_16` 의 해소 비율은 `E2_16` 과 같거나 약간 낮을 것이다 (world 차이 1 · 3 때문에 일부 chunk 에서 해소 후보가 적어질 수 있음). 고른 후보는 E2 보다 reference 에 가까운 noise · detour 가 많을 것이다 (GPU 결과가 reference 에 더 가깝게 수렴).
- 독점 lock 에서 발동 chunk `solve_ms` 중앙값은 ~400–500 ms 로 533 ms 경계 근처일 것이다 (검사가 절반). 3 번의 검사 줄이기를 넣으면 ~250 ms 아래로 내려갈 것으로 본다.
