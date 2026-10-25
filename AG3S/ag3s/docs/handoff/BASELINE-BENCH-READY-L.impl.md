# BASELINE-BENCH-READY — 구현 L (검출 상자 crop 의 depth trim · head/wrist association · CFM action 범위)

> writer: ag3s-implementer (L) · 읽는 쪽: verifier V6, scribe, lead · 작성 2026-10-25 (UTC 09:30, KST 18:30)
> base: `feature/panda-benchmark` 의 `3412861c` 에서 시작했다. 작업 중에 M 의 `a63f2864` 가 HEAD 가 됐다. 내 수정은 working tree 에만 있고 commit 하지 않았다. GPU 는 쓰지 않았고 CPU worker 는 1 개였다.
> task: `BASELINE-BENCH-READY.task.md` 의 "Implementer L". lead 의 추가 결정 (`gt_obstacle_set`) 도 반영했다.

## 무엇을 했나 (평이한 요약 먼저)

- **검출 상자 crop**: V5 의 filter arm (aegis · knows · multilink) 은 Grounding DINO 상자 안의 모든 depth 점을 장애물로 썼다. 그래서 탁자 · 배경 · 로봇 받침이 섞였고, KNOWS 의 call 0 ellipsoid 는 반축이 0.77 m 였다.
  - 이제 기본값 `crop_mode="depth_trim"` 은 상자 **중심에 있는 물체 하나** 만 남긴다.
  - 여러 상자 · 카메라에서 나온 cloud 는 3D 위치로 같은 물체인지 판단한 뒤에 합친다.
  - 옛 동작은 `crop_mode="box"` 로 bit 단위 그대로 남겼다.
- **CFM 의 \|xyz\| 7.28**: CPU replay 로 재현했다 (V5 call 24-26 → replay 11.4 · 7.1 · 6.2). 원인은 두 가지가 겹친 것이다.
  - CFM 자체: 원 방법의 QP 에 action 상한이 없다. trajectory model 의 Δt 가 0.01 이라 경로 1 cm 를 되돌리려면 normalized action 이 1 바뀌어야 한다.
  - 입력 쪽: grasp 하는 동안 EE ellipsoid 가 이미 점 구름과 겹쳐 있었다 (`h_0` = −0.07 ~ −0.08 m). 그 점의 80-94 % 는 target (bowl) 자리였다. head 카메라의 target 상자가 틀린 곳 (화면 왼쪽 끝) 에 잡혀서 bowl 이 장애물에서 빠지지 않았다.
  - 원 repo 에도 범위 제약이 없다. 그래서 lan 과 같은 방식의 `action_box=True` (기본) 를 넣었다. 원 방법과의 차이로 README 에 적었다.
- **다른 arm**: none · nudge · omniguide · lan 의 served chunk 는 V5 replay 입력에서 **bit 동일** 하다. 고친 4 arm 도 V5 option (`crop_mode="box"` · `action_box=False`) 을 주면 bit 동일하다.
- **lead 결정 반영**: `gt_seg` 에 `gt_obstacle_set` option 을 추가했다.
  - `"all_non_target"` 이 기본값이고 지금 동작과 같다.
  - `"obstacle_names"` 는 `privileged/obstacle_names` 에 있는 물체만 장애물로 본다.

## 1. 검출 상자 crop

### 원인 (V5 의 실제 입력으로 확인)

V5 aegis pilot 은 t0e0 call 0 의 head 상자 4 개를 기록했다 (`method_calls/calls.jsonl` 첫 행). 그 상자와 run 의 depth · K · T 로 call 0 관측을 다시 만들었다. `crop_mode="box"` 가 V5 KNOWS ellipsoid 를 그대로 재현하는 것도 확인했다: object#2 는 3441 점이고 반축은 (0.135, 0.264, 0.772), 오차는 6e-4 이내다 (test `test_v5_call0_box_mode_reproduces_the_077m_ellipsoid_and_depth_trim_removes_it`).

| 상자 (head, px) | V5 box crop 의 cloud | 무엇이 섞였나 |
|---|---|---|
| object#2 `[81.9, 0, 168.4, 101.3]` (화면 위 중앙) | centroid (−0.08, 0.01, −0.09), KNOWS 반축 **0.772 m** | 탁자 아래와 먼 배경, Panda 받침. 받침은 sphere model 의 가장 낮은 구가 z 0.085 m 라서 robot filter 가 덮지 못했다 |
| object#1 `[0, 84.9, 82.5, 214.8]` | 반축 0.531 m | cabinet 과 그 주변 탁자 |
| object#0 `[87.5, 94.4, 169.4, 174.2]` | 반축 0.358 m | 물체와 그 뒤 탁자 |

- **AEGIS 원 code 는 이 cleanup 을 `filtering_points` 에서 한다** (`benchmark/SafeLIBERO/main/utils.py:257-316`).
  - world workspace box 로 탁자면과 배경을 자른다. table scene 은 z 0.92-1.5 m, \|x\|, \|y\| < 0.3 m 이다 (`:265`).
  - 그다음 centroid 에서 가까운 80 % 를 남긴다.
  - DBSCAN 은 `eps=0.0001` m (`:304`) 라 실제로는 아무것도 지우지 않는다.
  - workspace box 는 suite 이름이 있어야 고를 수 있다. 그런데 V5 wire 에는 suite 이름이 없었다. 그래서 V5 기록은 `workspace: null` 이고, 80 % trim 만 적용됐다.
- **KNOWS · multilink 의 경로**:
  - KNOWS 는 `filtering_points` 를 거치지 않고 crop cloud 에 바로 MVEE 를 맞춘다. 그래서 0.77 m 가 그대로 나왔다.
  - multilink 는 AEGIS 의 `perceive_obstacles` 를 쓴다. 그래서 aegis 와 같은 영향을 받았다.
- **head/wrist 합침에 대한 정정**: K 는 "head · wrist 의 다른 물체가 한 cloud 로 합쳐진다" 고 적었다. 하지만 V5 closed-loop 의 `cameras` 는 `null` 이었다. `null` 이면 `CAMERAS_EXTERNAL = ("head",)` 이므로 **head 만** 썼다. aegis 기록의 상자도 head 것뿐이다.
  - 그 결함 자체는 실제로 있다: `all_objects` 이름 `object#j` 의 j 를 카메라마다 다시 센다. 다만 cameras 에 wrist 를 넣을 때만 생긴다.
  - 아래 association 으로 막았다. test 로 재현한 것은 `test_head_and_wrist_views_of_one_object_are_associated` 다.

### 수정 — `ObstacleCloudSource(crop_mode=...)` (`methods/safety_common.py`)

| 내용 | 위치 |
|---|---|
| `CROP_MODES = ("box", "depth_trim")`, 기본 `"depth_trim"` | `:261`, `:273` |
| `"box"` = V5 의 code path 를 그대로 둔 것 (상자 전체 − robot sphere, 같은 이름은 concat) | `_detect` `:530` 의 `else` 경로 |
| depth_trim (1) robot sphere + **Panda 받침 envelope** (`base_envelope_m` = r_xy 0.18 m, z < 0.16 m, base frame) | `trim_box` `:454`, `:466` |
| (2) **support plane**: 외부 카메라 (head) 전체 lift 에서 가장 밀한 수평면 (`guidance_common.scene.estimate_table_z`, guidance arm 과 같은 규칙, non-privileged) 위 1.5 cm 이하와 0.58 m 초과를 지운다. AEGIS workspace box 의 z 범위 (탁자면 위 ~0.6 m) 를 suite 이름 없이 대신한다 | `support_z` `:440` |
| (3) 1 cm voxel 26-연결 성분 중 **상자 중앙 40 % 창** 의 점을 가장 많이 가진 것 (창이 비면 상자 중심에 가장 가까운 성분) | `center_component` `:175` |
| (4) KNOWS `_trim_by_depth` (카메라 거리 2-98 percentile) | `depth_trim` (기존 함수) |
| **3D association**: 상자 · 카메라마다 나온 cloud 를 centroid 거리 ≤ 5 cm 이거나, 작은 cloud 의 50 % 이상이 1 cm 안에 겹치면 한 물체로 합친다. `obstacle_names` 는 같은 이름일 때만 합친다. 이름이 같은데 위치가 다르면 `<name>@<cam>` 으로 따로 둔다. `all_objects` 는 group 마다 `object#k` 로 다시 번호를 매긴다 | `associate` `:486` |
| 점이 `min_points` 보다 적은 crop (탁자 · 빈 곳 · 로봇) 은 장애물에서 뺀다. info 에 `trim` 단계별 점 수를 남긴다 | `_detect` `:572` |
| `remove_robot` 을 `robot_keep_mask` (mask 반환) 로 나눴다. 결과는 같다 (box 경로 bit 동일 test) | `:410` |
| provenance 에 `crop_mode` 와 `depth_trim` parameter 를 남긴다 | `provenance` |
| **`gt_obstacle_set`** (lead 결정): `"all_non_target"` (기본, 지금 동작) · `"obstacle_names"` (`privileged/obstacle_names` 에 있는 것만. 원 논문 조건과 비교하는 sensitivity 변형). 그 key 가 없으면 KeyError 를 낸다. provenance 의 `privileged_inputs` 에 key 가 더해진다 | `:265`, `_gt_seg` `:390` |

- 원 방법과 남는 차이 (aegis README item 3 에 적었다):
  - workspace box 대신 탁자 높이를 추정하고 연결 성분을 쓴다.
  - AEGIS 는 VLM 이 이름을 준 장애물 하나를 agentview + backview 1024 px 로 본다. 여기는 head 256 px 에서 generic prompt 로 찾은 여러 물체다.
  - 원 DBSCAN 은 사실상 아무것도 하지 않으므로 그대로 이식해 두었다.
- M 이 server 에 `extras["suite"]` 를 넣었다 (`a63f2864`). 그래서 aegis · multilink 에서는 이제 공식 workspace box 도 함께 켜진다. 두 처리는 서로 독립이다 (아래 표의 "+suite" 열).

### 전후 증거

**V5 t0e0 call 0, 기록된 GDINO 상자, KNOWS MVEE 최대 반축 (m)**. 출처: `outputs/paper_baselines_smoke/bench_ready_L/summary.json` 의 `filters_v5smoke…` 와 위 test.

| object | box (V5) | depth_trim | depth_trim 이 남긴 것 |
|---|---|---|---|
| #0 | 0.358 | **0.148** | c (0.62, 0.01, 0.14). 탁자 위 물체 하나 |
| #1 | 0.531 | **0.200** | cabinet (큰 물체 자체) |
| #2 | **0.772** | **0.054** | 받침 · 탁자 아래가 빠지고, 상자 중앙 창에는 점이 남지 않았다. 가장 가까운 작은 성분 270 점만 남았다 |
| #3 | 0.119 | 0.084 | 납작한 plate (destination) |

**AEGIS 공식 fit (`perceive_obstacles`) 의 반축 (m)** (`aegis_call0_ellipsoids.json`, 그리고 같은 입력에 `extras["suite"]` 를 더한 결과):

| | box | depth_trim | box + suite (M) | depth_trim + suite |
|---|---|---|---|---|
| 로봇 받침 상자 (#2) | **(0.216, 0.210, 0.101)**, c (0.10, 0.00, 0.00) = Panda base | 장애물 아님 (0.044 의 작은 성분만 남음) | (0.047, 0.040, 0.010) **z 0.21 의 팔 잔여 115 점** | 없음 (workspace 0 점) |
| #0 | (0.173, 0.162, 0.151) | (0.108, 0.099, 0.062) | (0.120, 0.109, 0.066) | (0.107, 0.099, 0.062) |
| #1 | (0.181, 0.171, 0.122) | (0.150, 0.143, 0.097) | (0.156, 0.140, 0.087) | (0.150, 0.134, 0.078) |

**CPU replay — V5 knows smoke 의 관측 12 call** (call 0-11, `--fake-source-key nominal`, GDINO cache 로 before / after 가 같은 상자를 본다):

| arm | 지표 | before (= V5 code) | after (depth_trim) |
|---|---|---|---|
| knows | `h_min` (h(n_stored)) | −0.1795 (V5 GPU 와 같은 값) | **−0.0423** |
| knows | `h_star_min` · h*<0 행 | −0.0381 · 56 | **−0.0256 · 10** |
| knows | served max \|xyz\| · 실행 행 \|xyz\|>1 | 1.434 · 8 (V5 GPU 와 같은 값) | **0.938 · 0** |
| knows | 개입 행 비율 | 0.99 | 0.82 |
| aegis | `h_min` · 개입 행 비율 | −0.0092 · 0.81 | +0.0169 · 0.91 |
| multilink_cbf | `h_min` · 개입 행 비율 | −0.0253 · 0.76 | −0.0005 · 0.85 |
| 셋 모두 | fallback | 0 | 0 |

- aegis 와 multilink 의 개입 비율은 오히려 올랐다. 장애물이 실제 물체 위치로 옮겨져 EE 경로에 더 가까워졌기 때문으로 보인다. 다만 이것은 open-loop replay 의 수치다. closed-loop 판정은 V6 에서 한다.

## 2. CFM 의 action 범위

### 재현 (CPU replay, V5 CFM pilot t0e2 의 관측과 nominal chunk)

`--fake-source-key nominal --fake-kind tracking` 으로 돌렸다. surrogate sampler 를 쓰므로 수치가 GPU 와 똑같지는 않다. 대신 같은 call 들이 같은 크기로 튄다.

| call (seq) | V5 GPU served max \|xyz\| | replay before | replay after |
|---|---|---|---|
| 17 | 2.51 | 2.42 | **1.00** |
| 18 | 1.60 | 1.62 | **1.00** |
| 24 | 7.11 | **11.38** | **1.00** |
| 25 | **7.28** | 7.05 | **1.00** |
| 26 | 6.40 | 6.16 | **1.00** |
| call 0-30 전체 | — | 31 call 중 served>1 은 7 call, 실행 행>1 은 17 | **0 · 0** (최대 1.0000000185) |
| `--fake-kind constant` (call 14-28) | — | 13 / 15 call, 최대 7.07 | **1 / 15 call, 최대 1.254** (아래 "남는 차이") |

### 원인 (`outputs/paper_baselines_smoke/bench_ready_L/cfm_diag/cfm_diag.jsonl`)

V5 run 의 관측으로 CFM 이 본 scene 을 다시 만들어 측정했다.

| call | `h_0` (측정 EE 에서의 clearance − d_safe) | EE ellipsoid 안의 장애물 voxel | 그중 target 점 3 cm 안의 비율 | 해당 call 의 head target 상자 |
|---|---|---|---|---|
| 5, 10 | +0.009, +0.007 | 0 | — | 없음 / `[0, 85, 82.6, 215.2]` |
| 15, 17 | −0.027, −0.020 | 13, 8 | 1.00 | `[0, 85, 82.7, 215.2]` (화면 왼쪽 끝. bowl 이 아니다) |
| **24, 25, 26** | **−0.074, −0.079, −0.066** | 31, 44, 32 | **0.81, 0.80, 0.94** | 같다 |

- **원 방법의 성질**:
  - trajectory model 은 `p_j = p_{j-1} + a_j[:3]·dt` 이고 dt = 0.01 이다 (normalized action). repo `openpi.patch:1012-1017` 에 있다.
  - 제약은 `h_j ≥ (1−γ) h_{j-1}` 이고 γ = 0.9 다.
  - 그래서 `h_0 < 0` 이면 첫 행부터 `~|h_0|/dt` 만큼 회복해야 한다. call 24 에서는 normalized 약 7 이다 (diag `delta_norm_max_abs` 11.3).
  - repo 의 SLSQP 호출에는 `bounds` 가 없다 (`openpi.patch:1143`). 갱신 `x_t = x_t + safety_correction` (`:1364`) 뒤에도 clip 이 없다.
  - 그러므로 원 방법에는 action 범위 제약이 없다. 실행할 때는 LIBERO OSC 가 dim 마다 [-1, 1] 로 자른다.
  - 원 논문은 SafeLIBERO GT sphere 를 쓴다. 그래서 EE 가 target 을 쥘 때 "장애물 안" 에 있지 않다.
- **입력 쪽 원인** (내 수정 범위 밖, `closed_loop/perception.py` · target 검출):
  - head 의 "black bowl" 상자가 화면 왼쪽 끝 (cabinet) 에 잡혔다. 그래서 head 에서 본 bowl 이 장애물 cloud 에 남았다.
  - wrist target 상자의 점과 겹치는 것으로 확인했다.
  - grasp 접근 중에 EE ellipsoid (6/12/11 cm) 가 그 점들을 감싸서 `h_0 < 0` 이 됐다.
  - privileged 변형 (`constrained_flow_matching:gt`, M) 은 이 원인이 없다.

### 수정 (`methods/constrained_flow_matching/guidance.py`)

| 내용 | 위치 |
|---|---|
| `action_box=True` (기본, default_params) | `:87` |
| box = LIBERO OSC 입력 범위 raw [-1, 1] 을 policy stats 로 normalized 한 것. lan 과 같은 함수 `action_box_normalized` 를 쓴다 | `_prepare` `:96-99` |
| denoising 중간 step: `clip(x_t + δ, min(lo, x_t), max(hi, x_t))`. 이미 box 밖인 dim 은 더 밖으로 밀지 않는다 (lan `clip_sample` 규칙) | `_guide_one` `:162-187` |
| 마지막 step (served chunk, `t_next < 0.5/N`): 수정된 (δ ≠ 0) 원소는 box 안으로 자른다. 수정되지 않은 원소는 bit 그대로 둔다 | `:182` |
| `x0_mode="noise"` 를 쓸 때는 lan 과 같은 k 보정을 한다 (원 CFM 은 identity) | `:185-187` |
| `posthoc` schedule 에도 같은 box 를 적용한다 | `posthoc_correct` `:188` |
| `action_box=False` = V5 동작 | — |

**원 방법과 남는 차이** (README 표에 한 줄 추가):

- 범위 제약은 원에 없는 것이다.
- 마지막 step 에서 δ = 0 이면 앞선 step 의 push 가 network 를 통해 남은 원소를 자르지 않는다. 그래서 constant surrogate (모든 push 가 끝까지 남는 최악의 경우) 에서는 1 / 15 call 이 1.254 다. 이것을 자르려면 unguided 출력이 원래 box 밖인 경우까지 건드려야 한다. 그러면 "피할 것이 없으면 bit 동일" 이 깨지므로 그렇게 하지 않았다.
- tracking replay 에서 원래 범위 안이던 24 call 은 최대 3.5e-6 (float 반올림) 만 다르다. 중간 step 의 clip 이 다음 Euler step 의 입력을 바꾸기 때문이다.

## 3. 다른 arm 은 그대로인가 (V5 replay 입력)

입력은 SL5c VLA-only `safelibero_spatial_t0_e0` call 0-5 이다 (K 의 replay 입력 A). before 는 `3412861c` 의 Baseline 을 고정 복사한 tree 로, after 는 현재 checkout 으로 돌렸다.

| arm | after vs before | after + V5 option vs before |
|---|---|---|
| `none` | **bit 동일** | — |
| nudge | **bit 동일** | — |
| omniguide | **bit 동일** | — |
| lan_o3dp_guidance_adapted | **bit 동일** | — |
| aegis | 바뀜 (max 0.54, 의도) | `{"crop_mode":"box"}` → **bit 동일** |
| knows | 바뀜 (max 0.91, 의도) | `{"crop_mode":"box"}` → **bit 동일** |
| multilink_cbf | 바뀜 (max 0.73, 의도) | `{"crop_mode":"box"}` → **bit 동일** |
| constrained_flow_matching | 바뀜 (max 5.47, 의도) | `{"action_box":false}` → **bit 동일** (t0e2 31 call 도 bit 동일) |

- after tree 에는 M 의 `a63f2864` (closed_loop) 도 들어 있다. replay 는 `--suite` 없이 돌렸으므로 `extras["suite"]` 는 꺼져 있다.

## 바뀐 파일

| 파일 | 무엇이 | 왜 |
|---|---|---|
| `Baseline/paper_baselines/methods/safety_common.py` | docstring (crop · gt_obstacle_set), `backproject_uv` · `voxel_components` · `center_component` · `cloud_overlap` (`:149-215`), `ObstacleCloudSource` 의 `crop_mode` 와 그 parameter · `gt_obstacle_set` · `robot_keep_mask` · `support_z` · `trim_box` · `associate` · `_detect` (`:261-598`) | 1, lead 결정 |
| `Baseline/paper_baselines/methods/constrained_flow_matching/guidance.py` | docstring, `action_box` (`:87`, `:96`, `:162-205`) | 2 |
| `Baseline/paper_baselines/methods/{aegis,knows,multilink_cbf,constrained_flow_matching}/README.md` | crop_mode · gt_obstacle_set · action_box 와 원 방법과의 차이 | 기록 |
| `Baseline/paper_baselines/methods/tests/test_depth_trim_crop.py` (신규, 10 test) | 아래 | |
| `Baseline/paper_baselines/methods/constrained_flow_matching/tests/test_cfm_action_box.py` (신규, 7 test) | 아래 | |

closed_loop · client · benchmark · AG3S 는 건드리지 않았다. 다만 작업 중 실수로 repo 의 디렉터리 14 곳 (`AG3S/AG3S`, `benchmark/benchmark`, `.venv-*/.venv-*` 등) 과 `Baseline/paper_baselines/external/external` 에 자기 자신을 가리키는 symlink 를 만들었다. scratchpad 의 before-tree 를 다시 만들 때 생긴 것이다. 모두 바로 지웠고, `git status` 로 남은 것이 없음을 확인했다.

## 단위 검증

```bash
cd /mnt/dev/AG3S_VLA-benchmark; export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
PYTHONPATH=$PWD .venv-baselines/bin/python -m pytest -q Baseline/paper_baselines/methods/tests \
  Baseline/paper_baselines/methods/aegis/tests Baseline/paper_baselines/methods/multilink_cbf/tests Baseline/paper_baselines/methods/knows/tests
#   -> 88 passed, 1 skipped
PYTHONPATH=$PWD .venv-baselines/bin/python -m pytest -q Baseline/paper_baselines/tests/test_evaluate_offline.py Baseline/paper_baselines/tests/test_interface.py
#   -> 18 passed, 1 skipped
PYTHONPATH=$PWD:$PWD/third_party/openpi/src .venv-openpi-live/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests \
  --ignore=Baseline/paper_baselines/closed_loop/tests/test_metrics_launcher.py Baseline/paper_baselines/tests/test_methods_guidance.py \
  Baseline/paper_baselines/methods/constrained_flow_matching/tests
#   -> 94 passed, 1 skipped (M 의 closed_loop test 포함)
PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests/test_metrics_launcher.py
#   -> 6 passed
```

| 신규 test | 확인하는 것 |
|---|---|
| `test_render_sanity` | 합성 장면 (ray cast: 탁자 z=0 + 앞 정육면체 + 그 뒤의 높은 상자) |
| `test_depth_trim_keeps_only_the_front_object_of_a_box` | 앞 물체 상자 (25 % 여유): box crop 에는 뒤 상자와 탁자가 섞인다. depth_trim 은 점이 모두 앞 정육면체 안이고, MVEE 최대 반축 < 0.07 m < box crop |
| `…_picks_the_centre_object_when_the_box_is_on_the_back_one` | 상자 중심이 뒤 물체면 뒤 물체만 남는다 |
| `test_table_and_empty_boxes_give_no_object` | 탁자만 있는 상자는 장애물이 아니다 (V5 box 는 장애물로 만든다). `support_z ≈ 0` |
| `test_head_and_wrist_views_of_one_object_are_associated` | head 와 wrist 의 같은 물체는 하나로 합쳐진다. 옛 순위-이름 합침은 다른 물체를 합친다 (재현) |
| `test_same_name_far_apart_is_not_merged_obstacle_names` | 같은 이름이라도 다른 위치면 `name@left_wrist` 로 따로 둔다 |
| `test_box_mode_is_the_v5_crop` | `crop_mode="box"` = V5 code path (점 배열이 bit 동일), 잘못된 mode 는 ValueError |
| `test_crop_mode_reaches_the_filters` | aegis · knows · multilink 에 기본 `depth_trim` 이 들어가고, `crop_mode` kwarg 가 전달된다 |
| `test_v5_call0_box_mode_reproduces_the_077m_ellipsoid_and_depth_trim_removes_it` | V5 실제 입력: box = V5 의 0.772 m (오차 6e-4), depth_trim 최대 반축 < box 의 절반, 모든 점이 탁자면 위 |
| `test_gt_obstacle_set_restricts_gt_seg_to_the_named_obstacles` | `obstacle_names` 는 이름이 있는 것만 (target 은 이름이 있어도 제외), cloud 는 기본과 같다. provenance 에 남고, key 가 없으면 KeyError |
| CFM `test_overlap_at_start_blows_up_without_the_box_and_stays_inside_with_it[tracking, constant]` | `h_0 < 0` 장면: V5 (`action_box=False`) 는 raw > 1.5, 기본은 ≤ 1 + 1e-5. guidance 는 여전히 작동하고, 회전 · gripper 는 그대로다 |
| CFM `test_correction_inside_the_box_is_bit_identical_to_v5[tracking, constant]` | 수정이 box 안에서 끝나는 장면은 V5 와 bit 동일하다 |
| CFM `test_nothing_to_avoid_is_bit_identical` | 피할 것이 없으면 두 설정 모두 unguided 와 bit 동일하다 |
| CFM `test_network_outside_the_box_is_not_pushed_further_out_during_denoising` | 중간 step 규칙과 마지막 step 규칙 |
| CFM `test_provenance_records_the_box` | provenance 의 params |

## CPU replay 산출물

| 무엇 | 경로 (`outputs/paper_baselines_smoke/bench_ready_L/`) |
|---|---|
| 요약 (bit 동일성, CFM, filter) | `summary.json` |
| before / after / V5 option 의 served chunk · method_calls | `{before,after,after_v5opt}/<arm>-{sl5c,v5smoke}/`, `…/cfm-e2-{tracking,constant}/` (`served.npz`, `method_calls/`, `replay.log`) |
| CFM 원인 진단 | `cfm_diag/cfm_diag.jsonl` |
| AEGIS call 0 ellipsoid | `aegis_call0_ellipsoids.json` |
| 도구 (scratchpad, in-process server + 기록된 request) | `/tmp/claude-0/-mnt-dev-AG3S-VLA-benchmark/91b137ec-607f-41ff-9542-8a1b0025e74a/scratchpad/L/{inproc_replay.py,run_inproc.sh,run_set.sh,cfm_diag.py,summarize.py}` |
| GDINO cache (replay 전용) | `outputs/paper_baselines_smoke/closed_loop_fix/gdino_cache` (`AG3S_SIDECAR_DETECTION_CACHE`) |

## verifier 가 알아야 할 것

- **새 option · 기본값 변경**

| 대상 | 내용 | V5 로 되돌리기 |
|---|---|---|
| aegis · knows · multilink_cbf (`ObstacleCloudSource`) | `crop_mode="depth_trim"` (기본). parameter: `center_frac` 0.4, `cluster_voxel_m` 0.01, `support_margin_m` 0.015, `max_height_m` 0.58, `trim_percentiles` (2, 98), `assoc_gate_m` 0.05, `assoc_overlap` 0.5, `base_envelope_m` (0.18, 0.16) | `--method-kwargs '{"crop_mode":"box"}'` |
| gt_seg | `gt_obstacle_set="all_non_target"` (기본, 변화 없음) · `"obstacle_names"` | — |
| constrained_flow_matching | `action_box=True` (기본) | `--method-kwargs '{"action_box":false}'` |

- M 의 `<arm>@v5` preset (`jobs.PRESETS["v5"]`) 이 위 두 kwarg 와 `--no-suite-extras` 를 묶는다.
- **새 기본에서 aegis · multilink 는 두 변화를 함께 받는다**: L 의 depth_trim 과 M 의 `extras["suite"]` (공식 workspace box). V5 대비 차이를 나누려면 `aegis@v5` · `aegis --no-suite-extras` · `aegis` 를 비교한다.
- **재생산이 필요한 산출물**: 고친 4 arm 의 GPU smoke (아래 명령).
- **옛 기록과의 호환**:
  - `filter_perception.detector` 에 `crop_mode`, `support_z`, `association`, 상자마다 `trim` 이 더해진다.
  - depth_trim 에서는 상자 수와 장애물 수가 같지 않을 수 있다. 상자에 `name: null` 인 것은 장애물이 되지 않았다는 뜻이다.
  - V5 기록의 key 는 그대로다.
- **depth_trim 의 처리 시간**: 상자마다 cKDTree 와 voxel label 을 한 번 더 한다. CPU replay 에서 filter 시간은 GDINO 에 비해 작았다 (after call 당 filter 0.03-0.16 s, cache 사용). GPU 측정은 V6 에서 한다.
- **확인하지 않은 것**:
  - GPU closed loop.
  - LIBERO object · goal · long 장면에서의 탁자 추정. floor scene 이나 living room 에서 가장 밀한 수평면이 탁자인지 확인하지 않았다. `support_z` 가 info 에 남으니 smoke 에서 본다.
  - wrist 카메라를 포함한 closed-loop 설정. 기본은 계속 head 만이다.

## V6 명령 (GPU, verifier)

공통 준비는 M 의 impl.md "V6 명령" 과 같다 (`L=…launch_method.py`, `O=outputs/verify/BASELINE-BENCH-READY`, GPU lock 과 다른 session 알림).

1. **CPU 재확인**: 위 "단위 검증" 의 네 명령과 replay 요약.
   ```bash
   .venv-ag3s/bin/python /tmp/claude-0/-mnt-dev-AG3S-VLA-benchmark/91b137ec-607f-41ff-9542-8a1b0025e74a/scratchpad/L/summarize.py
   # bit 동일 표만 다시 만들려면 (약 15 min, CPU 1):
   /tmp/claude-0/-mnt-dev-AG3S-VLA-benchmark/91b137ec-607f-41ff-9542-8a1b0025e74a/scratchpad/L/run_set.sh sl5c
   ```
   scratchpad 가 사라졌으면 outputs 의 `replay.log` 와 `served.npz` 만으로 표를 다시 셀 수 있다.

2. **고친 4 arm 의 smoke** (SafeLIBERO spatial L I t0e0). 새 기본과 V5 option 을 같은 seed 로 비교한다.
   ```bash
   .venv-ag3s/bin/python -m Baseline.paper_baselines.closed_loop.jobs plan \
     --arms none,aegis,aegis@v5,knows,knows@v5,multilink_cbf,multilink_cbf@v5,constrained_flow_matching,constrained_flow_matching@v5 \
     --suites safelibero_spatial:I --split dev --tasks 0 --episodes 0 --run-prefix blbr-L-smoke --step BASELINE-BENCH-READY-V6 \
     --out-dir $O/L_smoke
   AG3S_PARALLEL_TEST_LAUNCHER="$L" benchmark/runners/run_parallel_gpu.sh --jobs $O/L_smoke/jobs.txt --max-parallel 2 --est-gib-per-run 16
   .venv-ag3s/bin/python -m Baseline.paper_baselines.closed_loop.jobs aggregate --plan $O/L_smoke/plan.json
   ```
   - plan 은 CPU 에서 확인했다: 9 job, 2 병렬 약 0.25 h (jobs.py 의 V5 시간 모델). jobs.py 는 `none` 이 없는 plan 을 거부한다.
   - jobs.py 를 쓰지 않을 때의 단일 job 형식:
     `$L --method knows --safety-level I --task-ids 0 --episode-ids 0 --run-id blbr-L-smoke --step BASELINE-BENCH-READY-V6 --sidecar-device cuda --ref-run $AG3S_STORAGE_ROOT/results/benchmark/20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only`
     V5 option 은 여기에 `--no-suite-extras --tag v5 --method-kwargs '{"crop_mode":"box"}'` (CFM 은 `'{"action_box":false}'`) 를 더한다.
   - 볼 key (`method_calls/calls.jsonl`, `closed_loop_metrics.json`):

| arm | key |
|---|---|
| knows | `filter_obstacles.*.semi_axes` (call 0 최대), `h_min`, `h_star_min`, `h_star_neg_steps`, `served_max_abs_xyz` |
| aegis · multilink | `filter_perception.detector.boxes.head[*].trim`, `support_z`, `clouds` (workspace 점 수), `h_min` |
| CFM | `served_max_abs_xyz`, `served_rows_out_of_range` (기대 0 에 가깝다), `guidance.ctx`, `h_served` |
| 공통 | fallback, `server_ms_p50` |

3. **(선택) 20 ep pilot 재측정**: V5 pilot 과 같은 형식으로 고친 4 arm 만 돌린다 (`--tasks 0,1,2,3 --episodes 0,1,2,3,4`). benchmark 전체 실험은 하지 않는다.

4. **figure 입력** (task 의 V6 4 항목 중 앞의 둘):
   - crop 전후 ellipsoid 장면: `test_depth_trim_crop.v5_call0()` 이 V5 call 0 의 관측과 상자를 돌려준다. `ObstacleCloudSource(..., crop_mode="box"|"depth_trim")` 의 cloud, `info["boxes"]["head"][*]["trim"]`, `aegis_call0_ellipsoids.json` 을 쓴다.
   - CFM 범위 전후: `summary.json` 의 `cfm_v5pilot_t0e2.*.per_call_max`, 그리고 GPU smoke 의 `served_max_abs_xyz` 를 쓴다.

## 내가 기대하는 결과

(verifier 는 측정을 마친 뒤 읽는다)

- knows: call 0 의 최대 반축은 0.2 m 안팎 (cabinet) 이고, 0.77 m 는 없다. `h_star_neg_steps` 는 V5 보다 적다. served \|xyz\|>1 행은 줄지만 0 은 아닐 수 있다 (Eq. 12 에 상한이 없다).
- aegis · multilink: 로봇 받침 ellipsoid 가 없다. 성공률 · 충돌의 방향은 예측하지 않는다 (개입이 실제 물체 쪽으로 옮겨진다).
- CFM: served \|xyz\| 최대 ≤ 1 + 1e-5 가 대부분이다. 남을 수 있는 경우는 마지막 step 에 수정이 없는 call 이다. intervention 은 그대로 있고, 충돌은 V5 와 비슷하다 (target 이 장애물로 남는 입력 쪽 원인은 그대로다).
- `@v5` 의 4 arm 은 V5 smoke 의 같은 seed 결과와 같다 (GPU kernel 이 같다면).
