# CLOSED-LOOP-FIX — 구현 (lan_o3dp action 크기 · knows 제약과 h 정의 · omniguide skip / 재compile)

> writer: ag3s-implementer (K) · 읽는 쪽: verifier V5, scribe, lead · 작성 2026-10-24 (UTC 10:10, KST 19:10)
> base: `feature/panda-benchmark` HEAD `7a89ac2b` + working tree 수정 (commit 하지 않았다). GPU 는 쓰지 않았다.
> task: `CLOSED-LOOP-FIX.task.md`. 근거: `CLOSED-LOOP-METHODS-V3.verify.json` issues.

## 무엇을 했나 (평이한 요약 먼저)

세 arm 의 V3 문제를 CPU replay 로 그대로 재현한 뒤 원인을 찾고 고쳤다.

- **lan_o3dp**: pi0.5 sampler 에는 원 방법(DDIM)의 `clip_sample` 이 없어서 push 가 쌓였다. 원 방법처럼 action 범위 안으로 제한했다. served |xyz| 최대는 46.8 에서 1.0 이 됐다.
- **knows**: 두 가지가 섞여 있었다.
  - 우리 adapter 의 결함: detector 가 물체 이름을 call 마다 다시 매겨 얼어 있는 ellipsoid 가 다른 물체 위치로 순간이동했다. 논문 §3.2 처럼 track 을 유지하도록 고쳤다.
  - 논문 자체의 성질: relaxed normal 이 `h<0` 행을 normal 기울이기만으로 만족시킨다. 이것은 고치지 않고 step 마다 측정해 남겼다.
- **omniguide**: target 을 call 마다 다시 검출해서 freeze zone 이 거의 계속 켜져 있었다. 원 방법처럼 target 을 episode 에 한 번 저장하고, freeze zone 을 나가는 판정도 원 규칙으로 바꿨다. skip 은 38 에서 19 call 이 됐다.
- **재compile**: jaxpr 의 shape 와 구조를 고정해 lan · omniguide 모두 server 당 1 회가 됐다 (전에는 7 · 10 회).
- **latency key**: `ms.total` (client 가 기다리는 시간) 과 `ms.method` (= total − diagnostics) 로 정했다. 진단 시간도 850 ms 에서 12 ms 로 줄였다.

다른 다섯 arm (`none` · aegis · multilink_cbf · CFM · nudge) 의 served chunk 는 전후 bit 동일하다.

## 증거를 만든 방법 — CPU replay

| 항목 | 내용 |
|---|---|
| server | `serve_method --fake-policy recorded --sidecar-device cpu`, JAX CPU, `OMP_NUM_THREADS=1` (worker 1 개, 순차) |
| 입력 A | V3 replay 입력: SL5c VLA-only `safelibero_spatial_t0_e0` call 0-5 (8 arm 전부). fake policy 의 chunk = SL5c 가 실제로 낸 chunk |
| 입력 B | V3 closed-loop smoke run 자체의 관측 56 call (`--fake-source-key nominal`): 그 run 의 method 가 받은 VLA nominal chunk 를 그대로 다시 준다. knows · omniguide · lan |
| sampler | guidance arm 은 surrogate. B 와 lan 의 A-constant 는 `--fake-kind constant` 로 돌렸다. 모든 push 가 끝까지 남는 경우라 누적의 최악이다 |
| 검출 | GDINO 결과는 `AG3S_SIDECAR_DETECTION_CACHE` 로 저장했다. 그래서 before 와 after 가 같은 box 를 본다 |
| before | 수정 전 code 를 고정 복사한 tree (scratchpad `before_tree`, Baseline 만 복사, 나머지는 symlink) 로 돌렸다 |
| after | 수정된 checkout 으로 돌렸다 |
| 출력 | `outputs/paper_baselines_smoke/closed_loop_fix/{before,after}/<arm>-{sl5c,sl5c-constant,smoke}/`. 비교표는 `outputs/paper_baselines_smoke/closed_loop_fix/summary.json` |

재현 확인 (replay 가 V3 를 그대로 다시 만드는가):

| arm | V3 (GPU) | before CPU replay |
|---|---|---|
| knows, SL5c call 0-5 `h_min` | −0.1795 −0.1453 −0.1184 −0.4169 −0.0498 −0.1250 | 같은 6 값 |
| omniguide, smoke 56 call | skip 38, compile 10, skip 순서 `gggSgggSSSSSgSSg…` | 같은 수, 같은 순서 (문자열 일치) |
| lan, smoke 56 call | compile 7, served \|a\| 27.28 | compile 7, served \|a\| 46.8 (constant surrogate) |
| knows, smoke 56 call | `h_min` −0.447, h<0 419 | call 0-10 은 bit 동일, call 11 부터 갈라짐 (GDINO CPU / CUDA box 차이). `h_min` −0.609, h<0 437 |

## 문제 1 — lan_o3dp_guidance_adapted: action 크기

**원인**

- `methods/lan_o3dp_guidance_adapted/guidance.py` (수정 전 `_guide_one`, 현재 :158-182): push `da / max(1 − t_next, 0.1)` 를 매 denoising step `x_next` 에 더한다. 상한이 없다.
- t = 0.9 step 의 계수는 10 배다. V3 smoke call 5 의 `denoise_steps.dx_l2` 는 33.5 → 16.0 → 11.0 → … 였다.
- pi0.5 sampler (`pi0.sample_actions`) 는 x0 를 자르지 않는다. 그래서 network 가 push 를 되돌리지 않는 만큼 그대로 쌓였다.
- normalized ↔ raw 변환과 `step_m` 자체는 맞다. 1 cm waypoint push 를 pseudo-inverse 로 바꾸면 1-4 normalized 이다.

**원 방법의 정의** (`qqianfeng/lan-o3dp@b89c2898`)

- `scheduling_ddim.py:296-297`: `clip_sample` 이면 `pred_original_sample` 을 [-1, 1] 로 자른다.
- `scheduling_ddim.py:313-316`: `prev_sample[..., :2] += ngrad · 0.03 / sqrt(ᾱ_prev)`.
- Lan-o3dp 의 모든 workspace config 가 `clip_sample: True` 다.
- gradient 는 잘린 x0 에서 계산한다 (`diffusion_unet_pcd_policy.py:151-166`).
- 따라서 원 출력은 [-1, 1] + 마지막 push 한 번 안에 있다.

**수정** (`guidance.py`)

| 내용 | 위치 |
|---|---|
| `clip_sample=True` (기본). field 는 box 로 자른 `x0_hat` 위에서 계산한다 | :163-166 |
| push 는 `x0_hat + push/k` (k = x_next 의 x0 계수 `1−t`) 가 box 안에 남도록 줄인다 | :176-181 |
| network 가 이미 box 밖에 둔 차원은 더 밖으로 밀지 않는다 | :176-181 |
| push 0 → `x_next` bit 그대로 | — |
| box = LIBERO OSC 입력 범위 raw [-1, 1] → `pi05_libero` quantile stats 로 normalized: x [−1.300, 1.075], y [−1.246, 1.170], z [−1.067, 1.067] | `action_box_normalized` :57 |
| 마지막 step (t=0, `x0_hat = x_next`) 에서 served xyz 가 box 안에 들어간다 | — |
| `guidance_common/core.py:136` `ctx["x0_mode"]` 추가 (identity mode 에서 k=1). 다른 method 의 수치는 바뀌지 않는다 | core.py:136 |

**원 방법과 남는 차이** (README 에도 적었다)

- 원 sampler 는 guidance 와 무관하게 network 의 x0 도 자른다. 여기서는 pi0.5 를 고치지 않으므로 guidance 가 더한 부분만 제한한다.
- 범위는 작업공간이 아니라 action (EE delta) 의 범위다.

**전후 증거** (CPU replay)

| 입력 | served max \|xyz\| | 실행 행 \|xyz\|>1 | guidance compile | `da_xyz_l2_max` |
|---|---|---|---|---|
| SL5c 0-5, tracking | 2.469 → **1.000** | 16 → **0** | 4 → **1** | 2.05 → 2.05 |
| SL5c 0-5, constant | 34.63 → **1.000** | 28 → **0** | 4 → **1** | 35.9 → 2.36 |
| smoke 56 call, constant | 46.84 → **1.000** | 247 → **0** | 7 → **1** | 48.1 → 2.40 |

- 개입 행 비율은 1.0 그대로다 (guidance 는 계속 작동한다).
- `h_min` (hand SDF) 은 −0.0299 → −0.0270 이다.

## 문제 2 — knows: h_min −0.447 · h<0 419 step · qp_fail 0

### 확인한 세 질문

**1) 제약이 slack 이나 relaxed normal 로 풀렸는가**

- slack: 아니다. closed-loop arm 은 `CbfParams()` (논문 기본값, `slack_weight=None`) 이다.
- relaxed normal: **그렇다.** Eq. 12 의 ε box (ε=0.05) 안에서 normal 을 기울이면 `δc` 를 건드리지 않고 행을 만족시킬 수 있다.
  - after smoke 에서 `h<0` 인 348 행 중 293 행이 `relax_m ≥ need_m` 이다. relax_m = `ε‖∇ₙh‖₁`, need_m = `γ_h·|h|`.
  - 그 결과 `h<0` 인데 개입 0 인 행이 299 개다.
  - 이것은 KNOWS `docs/11 §4` 에 이미 기록된 논문 Eq. 12 의 성질이다 ("ε ≳ γ_h|h| 영역에서 필터가 사실상 비활성").
- qp_fail 0 도 논문 그대로다. Eq. 12 에는 `δc` 상한이 없어서 QP 가 infeasible 이 되지 않는다 (docs/16 D3: 5,879 step 중 0.0%).
- → **의도된 (논문) 동작이므로 고치지 않았다.** 대신 step 마다 측정해 남긴다.

**2) h 기록 시점과 QP 시점이 맞는가**

- 맞다. 기록되는 `min_h` 는 QP 가 쓰는 행 값 그 자체다 (`cbf/filter.py` `_rows` 의 `h_vals`).
- 다만 그 값은 **저장된 normal 에서의 h(n)** 이다. 진짜 분리 거리 `h* = maxₙ h(n)` 보다 항상 작거나 같다.
- after smoke 에서 `h(n)<0` 이지만 `h*≥0` 인 행이 308 / 348 이다. 즉 대부분의 "h<0" 은 겹침이 아니라 normal 이 최적값을 따라가지 못한 것이다.
- 기록 정의 (adapter docstring · README 에 명시):

| key | 정의 |
|---|---|
| `min_h` | Eq. 6 h(n_stored). row k 앞의 EE pose 에서 잰다. pose 는 **filtered** chunk 를 따라간다 (row 0 = 측정 pose, row k>0 = filtered row 0..k−1 의 `propagate_ee`) |
| `h_star` | 같은 pose 에서의 h* |
| `h_star_next` | QP step (δc, δθ) 뒤의 h*. QP 의 단위 모델 (0.05 m / unit) 로 잰다 |
| `h_star_next_nominal` | nominal step 뒤의 h* |
| `cbf_ok` | docs/16 §1 조건: 모든 obstacle 에서 h*_next ≥ (1−γ)h* |
| `relax_m` · `need_m` | 위 1) 의 두 값 |

**3) 시작부터 h<0 (이미 안쪽) 일 때의 처리가 논문과 같은가**

- 수식은 같다. Eq. 7 은 h<0 에서 Δh ≥ −γh > 0 (회복) 을 요구하고, 코드도 그 행을 그대로 넣는다.
- 진짜로 겹친 (h*<0) 행은 smoke 의 call 0-3 (40 행) 뿐이다.
- 원인은 perception 이다. call 0 의 obstacle ellipsoid 반축이 최대 0.77 m 다. GDINO box crop 에 배경 depth 가 섞였고, 같은 `object#j` 이름의 head · wrist crop 이 한 cloud 로 합쳐졌다 (`safety_common.ObstacleCloudSource._detect`).
- 이것은 `safety_common.py` (수정 범위 밖, aegis · multilink 도 같이 쓴다) 의 문제라 **고치지 않았다.** 아래 "남은 문제" 에 적었다.

### 원인으로 확인하고 고친 것 (논문 §3.2 와 다른 adapter 동작)

**원인** — `methods/knows/adapter.py` (수정 전 `_obstacles`):

- shape 를 첫 call 의 이름 (`object#0` …) 에 고정하고, 이후 같은 **이름** 의 cloud 로 centroid 를 옮겼다.
- detector 이름은 call 마다 score 순으로 다시 매겨진다. 그래서 track 이 다른 물체로 바뀐다. after 기록의 call 3 association: `object#0 ↔ 현재 object#1`, `object#1 ↔ 현재 object#0`.
- 그 결과 0.77 m ellipsoid 가 다른 물체 위치로 순간이동했다 (V3 call 3: h −0.417).
- 안 보이는 track 은 t=0 centroid 로 되돌아갔다.

**논문의 정의** (§3.2, `docs/03-math.md:277-339`, `perception/ellipsoid_fit.ObjectTracker`):

- 물체마다 track 하나를 유지한다. shape 는 t=0 에 고정한다.
- 가려지면 마지막 위치를 유지한다 (freeze).
- identity swap 은 HSV histogram Bhattacharyya 로 다시 잇는다.

**수정** (`adapter.py`)

| 내용 | 위치 |
|---|---|
| `associate="centroid"`: detector · given source 의 기본값 | :67-73 |
| Hungarian 으로 track (마지막 centroid) ↔ 현재 cloud centroid 를 잇는다. gate 0.10 m | `_match` :112 |
| 안 맞은 track 은 마지막 centroid 를 유지한다 | :151-160 |
| `gt_seg` · `ellipsoids` 는 이름이 안정적이므로 이전 동작 (`"name"`) 그대로다 | — |
| HSV 대신 centroid 거리를 쓴다. box crop 으로는 물체 색 histogram 을 만들 수 없다 (README 에 차이로 적음) | — |
| per-step 진단 (`_diagnose` :171, `diagnostics=True`). 읽기 전용이다. on / off 의 served chunk 와 filter normal state 가 같음을 test 로 확인했다 | :171 |

**전후 증거** (CPU replay, smoke 56 call · 560 행)

| | before | after: `associate=name` (진단만 켬) | after (수정) |
|---|---|---|---|
| served chunk vs before | — | **bit 동일** (진단이 출력을 안 바꿈) | 다름 |
| `h_min` (h(n_stored)) | −0.6086 | −0.6086 | **−0.2515** |
| h<0 행 | 437 | 437 | **348** |
| `h_star_min` (진짜 분리) | — | −0.1655 | **−0.0381** |
| h*<0 행 | — | 80 | **40** (call 0-3 뿐) |
| h(n)<0 이지만 h*≥0 | — | 357 | 308 |
| relax ≥ need 행 (normal 기울이기로 충족) | — | 303 | 293 |
| `cbf_ok` 위반 행 | — | 90 | 83 |
| qp_fail · emergency stop | 0 · 0 | 0 · 0 | 0 · 0 |
| served max \|xyz\| | 4.48 | 4.48 | **1.43** |
| 실행 행 \|xyz\|>1 | 41 | 41 | **8** |
| 개입 행 비율 | 0.386 | 0.386 | 0.275 |

- SL5c 0-5 입력에서 `h_min` 은 −0.4169 → −0.1795, served max \|xyz\| 는 2.71 → 1.43 이다.
- served \|xyz\| > 1 은 아직 남는다. 논문 Eq. 12 에 `δc` 상한이 없기 때문이다 (`max_delta_pos=None`, docs/16). 실행 때는 OSC 가 자른다.

## 문제 3 — omniguide: skip 38/56, jit 재compile 10 (lan 7)

### skip

**원인** — `methods/omniguide/guidance.py` (수정 전 `_prepare`):

- freeze zone 판정에 **그 call 의** target 점 (GDINO target box 를 lift 한 것) 을 썼다.
- 나가는 판정도 그 call 의 점으로 했다.
- 그런데 head camera box 가 자주 검정 cabinet 을 "black bowl" 로 잡는다 (점 1,400 개 이상). 쥔 bowl 은 gripper 와 함께 움직인다.
- 그래서 EE 가 늘 0.1 m 안에 들어가 38 call 이 identity 가 됐다.

**원 방법의 정의** (`omniguide-openpi@663f1cb2`, `pi0_pytorch.py`)

| 규칙 | 위치 |
|---|---|
| target 점 = VLM pointer + SAM2 mask. `cache_stale` (subtask index 가 바뀔 때) 에만 다시 구한다. sim config (`vlm_pointer_collision_sim.yaml`) 는 subtask monitor 를 끄므로 episode 에 한 번이다. cache 가 비어 있으면 다시 구한다 | `_apply_vlm_sam2_filter` :462-495 |
| freeze zone 은 들어갈 때 점을 `cached_task_relevant_pcd` 로 저장한다. 나가는 판정은 그 저장된 점으로부터의 거리 > exit_threshold 로 한다 | `_update_freeze_zone` :956-1002 |
| attractor 도 cache 된 점의 centroid 를 쓴다 | :1097-1104 |
| skip 은 freeze zone 으로만 일어난다 | :1440-1450 |

→ V3 의 skip 사유 문구 ("freeze zone / 피할 것 없음") 자체는 원 규칙과 같다. 다른 것은 판정에 쓰는 **점의 출처** 였다.

**수정**

- `target_cache="episode"` (기본): episode 의 첫 비어 있지 않은 검출을 저장해 freeze zone 과 attractor 에 쓴다 (`_target_points` :82).
- freeze zone 은 원 규칙 그대로다: 들어갈 때 점 저장, 나갈 때 그 점과의 거리로 판정 (`_update_freeze_zone` :92).
- `reset()` 이 둘 다 지운다.
- `target_cache="call"` 은 이전 출처 (매 call 검출) 다. 비교용이다.

### 재compile

**원인** — D 의 지적처럼 closure 가 장면을 상수로 잡으므로, shape 나 구조가 바뀌면 jaxpr key 가 바뀐다.

| arm | 바뀌던 것 |
|---|---|
| omniguide | EE clearance 안의 점을 **지워서** 점 수가 바뀌었다 (`o.padded()` 를 가변 길이로 호출). `ctx["collision"]` · `ctx["attract"]` 를 Python `if` 로 trace 했다 (target 이 있다 / 없다 에 따라 구조가 바뀜) |
| lan | `NO_PADDING` 으로 obstacle cloud 를 padding 하지 않았다. circle 수 (n) 도 가변이었다 |

**수정**

- omniguide (`guidance.py`):
  - clearance 안의 점은 FAR 로 옮긴다 (:119).
  - 두 항은 늘 trace 하고 (1,) 배열 0/1 가중치를 곱한다 (:134, :151). jaxpr 상수라서 literal 이 아니다. 값만 바뀌고 구조는 같다는 것을 확인했다.
- lan:
  - circle 을 16 개로 padding 한다 (:132).
  - `obstacle_circles` 는 FAR padding 점 · primitive 를 무시한다.
  - `serve_method.NO_PADDING` 을 비웠다 (`serve_method.py:48`).

### 전후 증거

| | before | after | after `target_cache=call` |
|---|---|---|---|
| smoke 56 call skip | **38** | **19** | 41 |
| skip 순서 | `gggSgggSSSSSgSSgSSSgggSgSgSgSSSSSgggSSSSSSSSSSSSSSSgSSSS` | `ggSSSgggggggggSggggggggggggggggggggggggggSSSSSSSSSSSSSSS` | `gggSSSSgSgSggSSgSSSgggSgSSSSSSSSSSSgSSSgSSSSgSSSSSSSSSSS` |
| guidance compile (56 call) | **10** | **1** | 1 |
| SL5c 0-5 compile | 1 | 1 | — |
| served max \|xyz\| | 1.032 | 1.056 | 1.032 |
| 개입 행 비율 | 0.321 | 0.661 | 0.268 |

- after 의 skip 은 두 구간이다.
  - call 2-4: 첫 bowl 위치에 다가간 grasp 접근 구간이다 (원 규칙 그대로).
  - call 41-55: 이 open-loop 녹화에서 EE 가 첫 bowl 위치 0.03-0.08 m 안으로 돌아온 구간이다 (`guidance.ctx.freeze_min_dist_m`).
- lan compile 은 문제 1 표에 있다 (7 → 1).
- 원 방법과 남는 차이 (README):
  - freeze 판정 점은 현재 EE 다. 원 방법은 unguided chunk 의 첫 waypoint tip 이다. 차이는 action 한 step 이다.
  - target 은 SAM2 mask 가 아니라 GDINO box lift 이며, 첫 검출이 틀리면 episode 내내 틀린 위치를 기억한다.

## 추가 — latency key (task 의 "diagnostics 시간")

**결정**: 진단은 `ms.serve` 에서 빼지 않고 별도 key 로 둔다.

| key | 정의 |
|---|---|
| `ms.total` | `MethodServer.infer` 의 wall time = client 가 기다리는 시간 (전송 제외) |
| `ms.diagnostics` | chunk 가 확정된 뒤의 기록용 계산 |
| `ms.method` | `total − diagnostics` |
| `ms.serve` | `total` 과 같다. **V2 / V3 기록에서는 `total − diagnostics` 였다** |

- 정의는 `serve_method.LATENCY_KEYS` (:58) 에 있다.
- `metrics.py` 는 옛 기록과 새 기록 모두에서 `total` 과 `total − diagnostics` 를 읽는다 (`server_ms_*`, `method_ms_*`, `diagnostics_ms_*`).
- 진단 계산은 jitted fixed-shape 거리 (`path_metrics_fast` :76) 로 바꿨다. CPU replay p50 이 lan 848 → 12 ms, CFM 137 → 11 ms, nudge 136 → 12 ms 다.
- client − `ms.total` 차이는 모든 arm 에서 p50 3-6 ms (전송) 다. 진단 값 (`h_nominal` / `h_served`) 은 jit 과 op-by-op 의 float32 반올림만큼 다를 수 있다. served chunk 와는 무관하다.

## 다른 arm 은 그대로인가

SL5c call 0-5 CPU replay, before ↔ after 의 served chunk:

| arm | 결과 |
|---|---|
| `none` | **bit 동일** (max diff 0.0) |
| aegis | **bit 동일** |
| multilink_cbf | **bit 동일** |
| constrained_flow_matching | **bit 동일** |
| nudge | **bit 동일** |
| knows · lan · omniguide | 바뀜 (의도) |

`none` 과 filter identity 의 bit 동일성 test 도 통과한다 (아래).

## 바뀐 파일

| 파일 | 무엇이 | 왜 |
|---|---|---|
| `methods/lan_o3dp_guidance_adapted/guidance.py` | `clip_sample` (:158-182), `action_box_normalized` (:57), circle 16 padding (:132), FAR 무시 (`obstacle_circles` :65) | 문제 1, 재compile |
| `methods/omniguide/guidance.py` | `target_cache` (:82), 원 freeze 규칙 (:92), FAR 이동 (:119), 0/1 가중치 (:134, :151) | 문제 3 |
| `methods/knows/adapter.py` | `associate` · `_match` (:112), last-centroid 유지 (:151-160), `_diagnose` (:171) | 문제 2 |
| `methods/guidance_common/core.py:136` | `ctx["x0_mode"]` | lan 의 k |
| `closed_loop/serve_method.py` | `NO_PADDING` 비움 (:48), `LATENCY_KEYS` (:58), `path_metrics_fast` (:76), ms 정의 (:243-250), KNOWS `*_steps` · `filter_obstacles` (:286-305), `guidance.ctx` (:357), `served_max_abs_xyz` · `served_rows_out_of_range`, `--fake-source-key` · `--fake-kind` (:483-488) | 기록 · 측정 |
| `closed_loop/fake_policy.py` | `source_key="nominal"` (method run 의 `method_calls` nominal 재생), `kind` | CPU replay |
| `closed_loop/sidecar.py` | `CachedDetector` (:101), `AG3S_SIDECAR_DETECTION_CACHE` 가 설정됐을 때만 켜진다 (:149) | replay 의 before / after 가 같은 box 를 보게 함 |
| `closed_loop/metrics.py` | latency 정의, `h_star_*` · `cbf_violated_steps` · `relaxation_covers_need_steps` · `guidance_skipped_calls` · `guidance_compiles` · `served_*` | 요약 |
| `closed_loop/tests/test_closed_loop_fix.py` (신규, 13 test) | 아래 | |
| `methods/knows/tests/test_knows_adapter.py` (+4 test) | 아래 | |
| README: `closed_loop/README.md`, `methods/{lan_o3dp_guidance_adapted,omniguide,knows}/README.md` | 원 방법과의 차이, h 정의, latency key | 수용 기준 |

`benchmark/**`, `AG3S/**`, `third_party/**`, SafeLIBERO, 다른 methods 는 건드리지 않았다.

## 단위 검증

```bash
source scripts/setup/env.sh; export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
PYTHONPATH=$PWD:$PWD/third_party/openpi/src .venv-openpi-live/bin/python -m pytest -q \
  Baseline/paper_baselines/closed_loop/tests --ignore=Baseline/paper_baselines/closed_loop/tests/test_metrics_launcher.py \
  Baseline/paper_baselines/tests/test_methods_guidance.py
#   -> 68 passed, 1 skipped (수정 전 55 passed, 1 skipped + 신규 13)
PYTHONPATH=$PWD .venv-baselines/bin/python -m pytest -q Baseline/paper_baselines/methods/knows/tests
#   -> 51 passed, 1 skipped (수정 전 47 + 신규 4)
PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests/test_metrics_launcher.py
#   -> 6 passed
PYTHONPATH=$PWD .venv-baselines/bin/python -m pytest -q Baseline/paper_baselines/tests/test_evaluate_offline.py Baseline/paper_baselines/tests/test_interface.py
#   -> 18 passed, 1 skipped
```

- 그대로 통과하는 bit 동일성 test:
  - `test_method_none_is_bit_identical_to_the_vla_only_server`
  - `test_wire_none_replays_the_recorded_vla_only_chunks`
  - `test_guidance_off_matches_policy_infer`
  - `test_disabled_is_bit_identical` · `test_no_obstacle_far_target_is_identity` (4 guidance)
  - `test_safe_chunk_is_identity_bitwise` (knows)
  - `test_other_filters_identity_without_obstacles_and_run_with_one`
- 신규 test:

| test | 확인하는 것 |
|---|---|
| `test_lan_guided_chunk_stays_in_the_libero_action_box` (tracking · constant × points · prim) | raw \|xyz\| ≤ 1 + 1e-5. guidance 가 여전히 작동한다. `clip_sample=False` 는 > 1.5 |
| `test_lan_obstacle_off_the_path_is_bit_identical` | circle 은 있으나 안에 든 waypoint 가 없으면 bit 동일 |
| `test_lan_no_obstacle_is_identity` | 장애물이 없으면 identity |
| `test_lan_circles_ignore_the_far_padding` | FAR padding 점 · primitive 를 circle 로 만들지 않는다 |
| `test_guidance_compiles_once_over_consecutive_calls[lan, omniguide]` | 점 수 · circle 수 · target 유무 · clearance 점 · gripper 가 다른 4 call 에서 `GuidanceSlot.stats == {compiles: 1, hits: 3}` |
| `test_omniguide_target_is_cached_per_episode` | target 이 episode 에 한 번 저장되고 `reset` 이 지운다 |
| `test_omniguide_freeze_zone_exits_when_the_held_target_leaves_its_cached_place` | episode `[F, ·, ·, ·]` vs call `[F, ·, F, ·]` |
| `test_omniguide_clearance_points_moved_to_far_not_dropped` | clearance 점이 지워지지 않고 FAR 로 옮겨진다 |
| `test_latency_keys_are_consistent` | latency key 정의 |
| knows `test_track_identity_survives_a_name_swap` | 이름이 바뀌어도 track 이 유지된다. `name` 모드는 > 0.3 m 순간이동 |
| knows `test_unmatched_track_holds_its_last_centroid` | 안 보이는 track 은 마지막 centroid 를 유지한다 |
| knows `test_diagnostics_are_read_only_and_complete[relaxed, fixed]` | 진단이 출력과 state 를 바꾸지 않고, key 가 모두 있다 |

## verifier 가 알아야 할 것

**새 플래그 · 기본값 변경**

| 대상 | 내용 |
|---|---|
| lan | `clip_sample=True` (기본) |
| omniguide | `target_cache="episode"` (기본) |
| knows | `associate="auto"` (detector / given → centroid), `assoc_gate_m=0.10`, `diagnostics=True` |
| serve_method | `--fake-source-key`, `--fake-kind` (CPU 전용) |
| sidecar | `AG3S_SIDECAR_DETECTION_CACHE` (replay 전용, **GPU run 에서는 설정하지 않는다**) |

**옛 기록과의 호환**

- `ms.serve` 의 뜻이 바뀌었다 (V3: total − diagnostics → 지금: total). 비교는 `ms.total` · `ms.method` 로 한다. `metrics.py` 는 둘 다 처리한다.
- `guidance_compiles` 는 **server process 누적** 값이다 (pilot 에서는 20 episode 합).

**재생산이 필요한 산출물**: 고친 3 arm 의 V3 smoke · replay 결과 (아래 V5 명령).

**git 주의 (lead 확인 필요)**: `Baseline/paper_baselines/methods/guidance_common/core.py` 는 `.gitignore:39` 의 `core.*` (core dump pattern) 에 걸려 **git 에 한 번도 올라간 적이 없다**.

- 7a89ac2b 를 새로 checkout 하면 모든 guidance arm 이 import 에 실패한다.
- 이번 수정 (:136 한 줄) 도 diff 에 안 보인다.
- commit 할 때 `git add -f` 를 쓰거나 `.gitignore` 에 예외를 넣어야 한다. `.gitignore` 는 내 수정 범위 밖이다.

## 남은 문제 (이번 범위 밖)

- **filter detector 의 cloud 품질** (`safety_common.ObstacleCloudSource._detect`, 수정 금지 범위):
  - box crop 에 depth trim 이 없다 (gt_seg 경로에는 있다).
  - `all_objects` 이름 `object#j` 가 camera 마다 j 를 다시 세서 head · wrist 의 다른 물체가 한 cloud 로 합쳐진다.
  - 그래서 call 0 의 ellipsoid 반축이 0.77 m 이고, 시작부터 EE 가 그 안에 있다 (h*<0).
  - knows · aegis · multilink 가 모두 이 경로를 쓴다.
- **guidance arm 의 target box**: head camera 가 cabinet 을 "black bowl" 로 잡는 경우가 있다. target 제외 (perception) 와 OmniGuide 의 첫 target cache 가 틀릴 수 있다.
- **knows 의 \|xyz\| > 1** (1.43): 논문 Eq. 12 에 `δc` 상한이 없어서 생긴다. 논문 정의라 그대로 뒀다.
- **omniguide served 1.056**: OmniGuide 자체의 `factor·guide` 로 생긴다. 원 방법에도 action clamp 가 없다. 이번 범위 (lan) 와 다른 문제로 남긴다.

## V5 명령 (GPU, verifier)

공통 준비:

```bash
cd /mnt/dev/AG3S_VLA-benchmark && source scripts/setup/env.sh
export OPENPI_DATA_HOME=$PWD/.cache/openpi XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer=
unset AG3S_SIDECAR_DETECTION_CACHE
SL5C=20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only
V=outputs/verify/CLOSED-LOOP-FIX-V5; mkdir -p $V
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # 비어 있어야 한다
```

1. **고친 3 arm 의 GPU smoke (t0e0, 1 episode 씩)**. launcher 가 GPU lock 을 스스로 잡는다.

   ```bash
   for M in knows omniguide lan_o3dp_guidance_adapted; do
     .venv-ag3s/bin/python Baseline/paper_baselines/closed_loop/launch_method.py --method $M --safety-level I \
       --task-ids 0 --episode-ids 0 --run-id clm-v5smoke --step CLOSED-LOOP-FIX --sidecar-device cuda \
       --ref-run $AG3S_STORAGE_ROOT/results/benchmark/$SL5C > $V/smoke-$M.launcher.log 2>&1
   done
   ```

   확인할 key (`<run>/closed_loop_metrics.json` 의 `method_calls`):
   - lan: `served_max_abs_xyz`, `served_rows_out_of_range`, `guidance_compiles`
   - omniguide: `guidance_skipped_calls`, `guidance_compiles`
   - knows: `h_min`, `h_star_min`, `relaxation_covers_need_steps`, `cbf_violated_steps`
   - 공통: `server_ms_p50` · `method_ms_p50`

2. **pilot: SafeLIBERO spatial Level I dev 20 ep (task 0-3 × ep 0-4, SL5c 와 같은 seed 7) × 8 arm, 최대 2 job**
   - 시작 전에 `ag3s-vla-benchmark-1a`, `ag3s-vla-development-ed` 에 알린다.
   - `git status --short benchmark/ AG3S/adapters third_party/` 가 비어 있는지 확인한다.

   ```bash
   REF=$AG3S_STORAGE_ROOT/results/benchmark/$SL5C
   : > $V/jobs_pilot.txt
   for M in none aegis knows multilink_cbf constrained_flow_matching nudge omniguide lan_o3dp_guidance_adapted; do
     X=""; [ $M != none ] && X="--sidecar-device cuda"
     echo "--method $M --safety-level I --task-ids 0,1,2,3 --episode-ids 0,1,2,3,4 --run-id clm-v5pilot --step CLOSED-LOOP-FIX $X --ref-run $REF" >> $V/jobs_pilot.txt
   done
   AG3S_PARALLEL_TEST_LAUNCHER="$PWD/.venv-ag3s/bin/python $PWD/Baseline/paper_baselines/closed_loop/launch_method.py" \
     benchmark/runners/run_parallel_gpu.sh --jobs $V/jobs_pilot.txt --max-parallel 2 --est-gib-per-run 16 \
     --log $V/pilot_parallel.log
   ```

   지표 재계산:
   ```bash
   PYTHONPATH=$PWD .venv-ag3s/bin/python -m Baseline.paper_baselines.closed_loop.metrics <run> --ref $REF
   ```

   - `run_parallel_gpu.sh` 의 launcher override 는 CLOSED-LOOP-METHODS-I 와 같은 방식이다 (`--launcher` H3 는 아직 없다).
   - 이번 수정으로 guidance arm 의 진단 시간이 줄었으므로 V3 의 pilot 추정 (4-5 h) 보다 짧을 수 있다.

## 내가 기대하는 결과

(verifier 는 측정을 마친 뒤 읽는다)

- GPU smoke lan: served \|xyz\| ≤ 1 (+1e-5), out-of-range 행 0, compile 1.
- omniguide: compile 1, skip 은 56 call 중 V3 의 38 보다 적다. 단 closed loop 라 궤적이 바뀌므로 수치는 replay 와 다를 것이다.
- knows: qp_fail 0 그대로, `h_min` 은 V3 보다 덜 음수. `relaxation_covers_need_steps` 가 h<0 행의 대부분이다 (논문 성질).
- 다른 5 arm 은 같은 seed 에서 V3 smoke 와 같은 결과다 (served chunk 경로 불변).
