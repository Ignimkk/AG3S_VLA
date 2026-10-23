# PAPER-METHODS (D) — 구현: sampling guidance 4종 + 외부 repo 6개 fetch

> writer: ag3s-implementer (Implementer D) · 읽는 쪽: verifier, scribe, lead · 작성 2026-10-23 KST
> checkout `/mnt/dev/AG3S_VLA-benchmark`, branch `feature/panda-benchmark`, 선행 commit `bf1403d5`. commit 하지 않았다.

## 무엇을 했나 (평이한 요약 먼저)

논문 비교 방법 중 **flow-matching denoising 안에서 action chunk 를 미는 방법 4개** 를 openpi `pi0.sample_actions` 의 `guidance_fn(x_next, t_next)` hook 에 맞춰 pure JAX 로 만들었다. openpi 는 고치지 않았다.

- OmniGuide [19]: 공식 code (Apache-2.0) 에서 포팅했다.
- constrained flow matching [5]: repo 에 license 가 없어서 **재구현** 했다.
- Lan-o3dp [15]: guidance 만 이식했다 (**adapted**).
- NUDGE [9]: code 가 없어서 논문을 읽고 구현했다.

네 방법 모두 `get_method(name)` 로 부르고 같은 `SceneObservation` 을 받는다. obstacle 은 depth lift, privileged primitive, 주어진 점 구름 중에서 고른다.

외부 repo 7개를 full SHA 로 pin 하고 fetch 했다. 요청은 6개였고, OmniGuide 의 실제 code 가 submodule `omniguide-openpi` 에 있어서 그것을 하나 더 넣었다. RoboGround, GuidedVLA, OBEYED-VLA 는 README 만 썼다.

**작동 근거:**
- unit test 35개가 통과했다 (1개 skip).
- 기록된 SL3 dev call 2개로 CPU smoke 를 돌렸다. 네 방법 모두 jit loop 안에서 끝까지 실행되어 수정된 chunk 를 냈다. 장애물을 뚫던 path 의 clearance 가 늘었다.
- pi0.5 를 붙인 GPU smoke 는 명령만 준비했다 (verifier 몫).

## 바뀐 파일

| 파일 | 무엇이 | 왜 |
|---|---|---|
| `Baseline/paper_baselines/external/sources.json` | 7개 source 추가: OmniGuide `d649d698…`, omniguide-openpi `663f1cb2…`, VLA-Safety-ConstrainedFlowMatching `b91153ad…`, lan-o3dp `b89c2898…`, RoboGround `25a86263…`, GuidedVLA `04be059e…`, DRAGON_VLA `bdf25556…`. 모두 full SHA. `license` 필드는 pinned checkout 의 LICENSE 를 보고 채웠다 | task D-1 |
| `Baseline/paper_baselines/external/fetch_sources.py` | `--used-by` filter 와 docstring (license 가 없는 clone 은 읽기 전용) | task D-1. fetch 동작 자체는 그대로다 |
| `Baseline/paper_baselines/methods/guidance_common/core.py` | `GuidanceBase` (`make_guidance_fn`: batch `vmap`, `x0_mode` noise / identity, `enabled=False` → 입력을 그대로 반환), `surrogate_sample` (pi0 와 같은 Euler `while_loop`, velocity 는 `tracking` 또는 `constant`), `load_norm_stats`, `chunk_to_model_space` | 네 방법이 함께 쓴다 |
| `.../guidance_common/kinematics.py` | `EEPathModel`: EE-delta → EE path. OSC response model (`robots/panda/description/osc_response_fit.json` 을 읽기만 한다) / `integrator` / `commanded` | NUDGE 의 decoder 와 OmniGuide 의 action→trajectory 단계 |
| `.../guidance_common/geometry.py` | `Obstacles` (점, sphere, box, cylinder. 고정 shape 로 padding), JAX signed distance 와 closest point | |
| `.../guidance_common/scene.py` | `build_scene`: obstacle 우선순위 `privileged/obstacle_geoms` → `obstacle_points` → depth lift. depth lift 는 target, robot, ignore mask 를 빼고 table 을 추정하고 voxel downsample 한다. hand / finger sphere 는 cuRobo `franka.yml` + URDF grip-site 를 읽기만 한다. 읽은 privileged key 를 `info["privileged_inputs"]` 에 남긴다 | |
| `.../guidance_common/recorded.py` | 녹화 call → `SceneObservation` + 기록된 raw chunk. A 의 `eval/offline_grounding.Episode` 를 재사용한다. GT mask 는 `privileged/` key 로만 넣는다 | smoke 입력 |
| `.../guidance_common/evaluation.py` | `path_metrics` (EE, hand min SDF), `injected_obs`, `SCENE_KW` | CPU / GPU smoke 공용 |
| `.../guidance_common/gpu_smoke.py` | 실제 pi05_libero 로 같은 noise 를 주고 guidance 유무를 비교한다. **실행하지 않았다** | task D-7 (verifier 에게 넘김) |
| `.../omniguide/{__init__,guidance}.py` | `OmniGuideGuidance`: log-barrier repeller (EE + wrist probe), target attractor, row 별 norm clip, factor, freeze zone hysteresis, EE clearance filter | task D-2 |
| `.../constrained_flow_matching/{__init__,guidance}.py` | `ConstrainedFlowMatchingGuidance`: ellipsoid EE clearance, discrete CBF 제약, min-norm δ (SQP: `jacfwd` 선형화 4회 × Hildreth 60 sweep), flat / bellcurve / posthoc schedule | task D-3 |
| `.../lan_o3dp_guidance_adapted/{__init__,guidance}.py` | `LanO3dpGuidanceAdapted`: 물체별 원 (xy bbox) → 원 안 waypoint 를 radial unit 방향으로 민다 → OSC Jacobian 의 damped pseudo-inverse 로 action 변화를 구한다 → `÷ max(1−t, 0.1)` | task D-4 |
| `.../nudge/{__init__,guidance}.py` | `NudgeGuidance`: hand sphere 의 squared hinge SDF cost, `w = ρ/(κ(1−ρ))`, 논문 flow 식 `x_next += (Δ/t_k)(clip(x0−wg) − x0)`, `calibrate_kappa`, `KAPPA_SL3_DEV = 0.0086` | task D-5 (sampling guidance 로 분류) |
| `.../{omniguide,constrained_flow_matching,lan_o3dp_guidance_adapted,nudge,guidance_common}/README.md` | 원 논문, repo@SHA, license, 원 방법과의 차이, 실행 명령, 상태 | task 공통 |
| `.../{roboground,guidedvla,obeyed_vla}/README.md` | 입력, checkpoint, 우리 protocol 에 넣는 데 필요한 것 (실행하지 않음) | task D-6 |
| `Baseline/paper_baselines/tests/test_methods_guidance.py` | 36개 test (아래) | task D-7 |
| `Baseline/paper_baselines/scripts/smoke_guidance_methods.py` | recorded-frame CPU smoke, figure, `--calibrate-kappa` | 작동 증거 |

**쓰지 않은 것:** `methods/interface.py` (C 의 것. 그대로 import 한다), `AG3S/**`, `benchmark/**`, `third_party/**` (openpi 포함), 루트 `.gitignore`, venv 설치. `.venv-openpi-live` 에 이미 있는 JAX 0.5.3, numpy 1.26.4, scipy 1.15.3 만 썼다.

## 단위 검증

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
  .venv-openpi-live/bin/python -m pytest -q Baseline/paper_baselines/tests/test_methods_guidance.py -p no:cacheprovider
# -> 35 passed, 1 skipped in 26.06s
```

skip 1개는 `omniguide` × `points` repeller 다. OmniGuide 는 EE 에서 5 cm 안의 점을 일부러 지운다 (원 설계). 그래서 이 합성 장면에서는 앞쪽 반구가 사라진다. prim 경우로 대신 확인했다.

| test | 무엇을 보나 | 결과 |
|---|---|---|
| `test_registry_and_protocol` ×4 | `get_method(name)` 이 `SamplingGuidance` 이고 provenance 가 있다 | pass |
| `test_disabled_is_bit_identical` ×4 | `enabled=False` → hook 이 입력 객체를 그대로 반환. 표본 chunk 가 `guidance_fn=None` 과 `tobytes()` 단위로 같다 | pass |
| `test_no_obstacle_far_target_is_identity` ×4 | 피할 것이 없으면 bit 단위로 같다 | pass |
| `test_surrogate_reproduces_target` | 대체 sampler 가 목표 chunk 를 재현한다 (atol 1e-5) | pass |
| `test_repeller_increases_clearance` ×8 (1 skip) | 3 cm sphere 를 뚫는 직진 path 의 min clearance 가 5 mm 넘게 늘어난다 (prim, points) | pass 7 |
| `test_repeller_pushes_waypoints_away` ×3 | 가장 가까운 waypoint 가 장애물에서 3 mm 넘게 멀어지고, 끝점이 −y (장애물 반대쪽) 로 간다 | pass |
| `test_batch_rows_independent_and_equal` ×4 | b=2 batch, 유한값 | pass |
| `test_omniguide_attractor_moves_toward_target` | attractor 방향 (factor 20 으로 확대해서 본다) | pass |
| `test_omniguide_freeze_zone_is_identity_near_target` | target 5 cm 안이면 identity | pass |
| `test_cfm_correction_satisfies_cbf_constraints` | 수정 전 `min g < 0` 이었던 것이 수정 뒤 `≥ −1e-3` | pass |
| `test_cfm_no_violation_no_correction` | 위반이 없으면 δ = 0 | pass |
| `test_nudge_update_rule_matches_paper_flow_form` | hook 출력 = 논문 Ex. III.6 식 (rtol 1e-5) | pass |
| `test_lan_circles_from_point_cloud` | 점 구름 → 원 1개 (중심 오차 < 5 mm, 반경 3-5 cm) | pass |
| `test_signed_distance_primitives` | sphere, box (안과 밖), cylinder 의 해석해 | pass |
| `test_path_model_matches_response_recursion` | `EEPathModel` = `AG3S/adapters/panda/action.py:predict_ee_path` 재귀식 (회전 base 포함) | pass |

합성 장면 수치 (`EE (0.45, 0, 0.30)`, raw dx = 1.0 × 10 row, sphere r 3 cm @ (0.52, 0.02, 0.30), tracking 대체 sampler):

| method | clearance 전 → 후 (prim) | (points) |
|---|---|---|
| omniguide (collision 만) | −10.0 → +7.2 mm | (EE clearance filter 때문에 −9.7) |
| constrained_flow_matching | −10.0 → +47.2 mm | −10.0 → +47.2 mm |
| lan_o3dp_guidance_adapted | −10.0 → −2.3 mm | −10.0 → −1.2 mm |
| nudge | −10.0 → +36.8 mm | −10.0 → +36.8 mm |

## recorded-frame smoke (CPU)

```bash
source scripts/setup/env.sh
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
  .venv-openpi-live/bin/python Baseline/paper_baselines/scripts/smoke_guidance_methods.py
```

- 입력:
  - SL3 dev `20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0`, `libero_spatial_t0_e0`, call 1 (obs 15) 과 call 2 (obs 20).
  - 기록된 raw chunk (10×7) 를 pi05_libero quantile norm 으로 바꿔 대체 sampler 의 목표로 쓴다.
  - 재현 오차는 raw 1.2e-7 이다.
  - openpi 의 실제 Normalize → Unnormalize 와 우리 normalize 의 왕복 오차는 2.0e-7 이다. openpi transform 으로 따로 확인했다.
- obstacle (scene):
  - head + wrist depth lift 에서 privileged mask (GT target, robot, support / mount / destination) 를 뺐다. 873 / 850 점 (2 cm voxel).
  - injected 는 기록된 path 6번째 waypoint 옆 1 cm 에 둔 privileged 3 cm sphere 다.
- 출력: `outputs/paper_baselines_smoke/methods/guidance/summary.json`, `guidance_smoke_libero_spatial_t0_e0_c{1,2}.png` (EE path 위에서 본 그림, unguided / tracking / constant).
- 비용: 최대 RSS 989 MB, CPU 1 process, 전체 약 1분. method 마다 jit + 실행에 0.1-2.2 s.

call 1 (mm. path 는 clip 한 action 을 OSC response 로 decode 했다. tracking / constant 는 대체 sampler 의 양 끝 경우다):

| variant | method | EE min SDF 전 → tracking / constant | hand min SDF 전 → tracking / constant | max \|Δa\| raw (trk / const) |
|---|---|---|---|---|
| scene | omniguide | 44.8 → 44.9 / 45.6 | 8.4 → 8.7 / 13.3 | 0.03 / 0.25 |
| scene | constrained_flow_matching | 44.8 → 47.8 / 48.1 | 8.4 → 6.8 / 3.3 | 5.56 / 4.56 |
| scene | lan_o3dp_guidance_adapted | 44.8 → 47.2 / 22.5 | 8.4 → 10.1 / 25.1 | 1.44 / 14.39 |
| scene | nudge | 44.8 → 43.8 / 44.3 | 8.4 → 33.9 / 20.0 | 0.29 / 0.12 |
| injected | omniguide | −20.0 → −18.7 / +13.2 | −30.9 → −31.3 / −29.9 | 0.03 / 0.93 |
| injected | constrained_flow_matching | −20.0 → +39.3 / +38.1 | −30.9 → −18.5 / −18.2 | 2.98 / 3.12 |
| injected | lan_o3dp_guidance_adapted | −20.0 → −5.6 / +9.1 | −30.9 → −33.2 / −34.2 | 1.23 / 12.34 |
| injected | nudge | −20.0 → +8.4 / +20.2 | −30.9 → −28.8 / −16.5 | 0.97 / 1.33 |

call 2 도 같은 경향이다 (`summary.json`). 모든 경우 출력이 유한했고, 같은 입력을 두 번 돌리면 같은 값이 나왔다 (`deterministic` True).

**NUDGE κ calibration:**
```bash
.venv-openpi-live/bin/python Baseline/paper_baselines/scripts/smoke_guidance_methods.py --calibrate-kappa
```
- 결과: `kappa_median` 0.00859 (n = 135. scene 중앙값 0.0044, injected 중앙값 0.0102).
- 처음에는 call 마다 ℓ∞ 로 정규화했다 (`kappa=None`). 그때 scene 의 hand SDF 가 +8.4 에서 −21.2 mm 로 **나빠졌다**. 작은 gradient (5e-4) 에도 5.67 normalized step 이 걸려 overshoot 했기 때문이다.
- 그래서 논문 정의대로 calibration set 의 중앙값을 기본값으로 바꿨다.

## 원 방법과의 차이 (요약. 자세한 것은 각 README)

| method | 원과 같은 것 | 다른 것 |
|---|---|---|
| OmniGuide | cost 식, 상수 (factor 0.4, 0.2 / 0.1 m, 0.1 / 0.01, max_guide_length 1.0, num_guided 10), row clip, freeze zone, EE clearance | DROID joint velocity + FK → LIBERO EE-delta + OSC response model. network 를 거치는 backprop 을 하지 않는다 (`x0_hat` gradient). ESDF Chebyshev 근사 → 정확한 거리. best-of-N noise 선택과 subtask monitor 는 생략. LIBERO 에서는 action → 변위 gain 이 약 10배 작아서 attractor 가 약하다 ([9] Table III 의 "OmniGuide ≈ unguided" 와 맞는다) |
| constrained flow matching | 논문 Eq. 7-11 과 repo 기본값 (d_safe 0.01, γ 0.9, 반축 (0.06, 0.12, 0.11), Δt 0.01, scale 1.0, flat), post-Euler 적용 위치, normalized action 적분 | SLSQP → jit SQP (Hildreth). greedy fallback 은 없다. ellipsoid primitive 는 없다. obstacle 을 주지 않으면 depth 점 구름 전체가 제약이 된다 (원은 SafeLIBERO GT sphere 만 쓴다) |
| Lan-o3dp (adapted) | 물체별 원 (bbox 반대각), xy radial unit field, `÷ (x0 계수)` | DDIM → flow. 절대 위치 → EE-delta (pseudo-inverse). 0.03 normalized → `step_m` 0.01 m. 원 policy 와 결과가 아니다 |
| NUDGE | squared hinge SDF cost, ρ 0.85, κ 정의, 논문 flow update 식 (test 로 확인), H = K = 10 | Panda sphere 41개 + IK → hand / finger sphere 22개와 OSC path (orientation 고정). 팔 link 는 보지 않는다. d_safe 0.02 (논문에 값 없음). κ 는 SL3 dev 로 calibrate. cuRobo ESDF → 점 구름까지의 정확한 거리 |

## license 상태

| repo | SHA | license (pinned checkout) | 처리 |
|---|---|---|---|
| daniilidis-group/OmniGuide | d649d69849bfc0c56bc822cdbe1f5f8a9fad2706 | LICENSE 파일 없음 (code 는 submodule 에 있다) | 참조만 |
| OmniGuide-Code/omniguide-openpi | 663f1cb2021282cf673fdfd737800b04ba80c893 | Apache-2.0 | 식과 상수를 포팅하고 출처 줄을 주석에 적었다 |
| Dubascudes/VLA-Safety-ConstrainedFlowMatching | b91153adc2f40d0ecfd2ae9ece4b6a2aed31dc7e | **없음** ("not yet been selected") | code 를 복사하지 않았다. 논문과 repo 의 규칙을 다시 적어 재구현했다 |
| qqianfeng/lan-o3dp | b89c2898d8548f7dfde63c44eeb733a59f0c5222 | MIT | guidance 만 이식 |
| ZzZZCHS/RoboGround | 25a862634398bc0407bc7b36b5d8b7c61d7a502b | MIT | fetch + README |
| GuidedVLA/GuidedVLA | 04be059e0d6bd448be5cb45fdbafc775f7eb5e38 | Apache-2.0 (+ Gemma) | fetch + README |
| UARK-AICV/DRAGON_VLA | bdf25556b65728588080cd5a7245b76d39b24591 | **없음** | fetch + README. code 를 복사하지 않았다 |

clone 본체는 `external/.gitignore` (`*`) 로 추적하지 않는다. `git status` 에는 `sources.json` 과 `fetch_sources.py` 만 나온다.
`fetch_sources.py --check --used-by D` 결과 7개 모두 `ok` 다.

## verifier 가 알아야 할 것

- 새 플래그와 기본값: 모두 새 module 이다. 기존 동작은 바뀌지 않았다.
  - `fetch_sources.py --used-by` 를 추가했다.
  - NUDGE `kappa` 기본값은 0.0086 이다 (calibrated).
  - CFM 기본값은 원과 같이 `x0_mode="identity"`, `path_model="integrator"` 이다.
  - 나머지 세 방법의 기본값은 `x0_mode="auto"` (noise 를 주면 noise 기반) 와 `path_model="osc_response"` 다.
- 재생산이 필요한 산출물: 없음. smoke 출력은 `outputs/` 아래에 있고 git 이 추적하지 않는다.
- 옛 기록과 호환이 깨지는가: 없음.
- **GPU smoke (다음 STEP):**
  - 명령은 `methods/guidance_common/README.md` 또는 `gpu_smoke.py` docstring 에 있다 (GPU lock 과 `flock -w 3600 9` 형식).
  - CPU 에서는 다음까지만 확인했다: import, `--help`, recorded 관측 → openpi input transform → `Observation.from_dict` (image 224×224 ×3, state 32), output transform 왕복.
  - model load 와 `sample_actions` jit 은 실행하지 않았다.
  - guidance closure 가 scene 을 상수로 잡고 있어서 call 마다 다시 trace 된다. smoke 에서는 괜찮지만 closed-loop 에서는 compile 비용이 든다. 다음 STEP 에서 scene 을 traced 인자로 바꾸는 편이 낫다.
- `recorded.py` 는 A/E 의 `eval/offline_grounding.Episode` 를 import 한다. 작업 중에 E 가 그 파일을 수정했고, 마지막 smoke 는 수정된 판으로 돌았다.
- `.venv-ag3s` 와 `.venv-baselines` 에는 JAX 가 없어서 이 test 파일은 skip 된다 (`importorskip`). 반드시 `.venv-openpi-live` 로 돌려야 한다.

## 확인하지 못한 것

- 실제 pi0.5 network 위의 효과. CPU 에서는 대체 velocity 로 양 끝 경우 (tracking, constant) 만 봤다.
  - tracking 에서는 `t_next = 0` 의 마지막 call 만 남는다.
  - constant 에서는 모든 수정이 누적된다. 그래서 Lan-o3dp 의 `1/(1−t)` 가 max |Δa| 12-16 raw 까지 커진다.
- closed-loop 성공률과 충돌률 (SafeLIBERO, LIBERO). 다음 STEP 이다.
- grounding provider (Grounding DINO 등) 를 붙인 non-privileged smoke. 이번 smoke 는 GT mask (privileged) 만 썼다. `extras["target_mask"]` 경로는 코드에 있지만 provider 와 실제로 연결해 보지는 않았다.
- CFM 의 ellipsoid obstacle primitive 와 greedy fallback. NUDGE 의 팔 link sphere (IK 필요).

## 내가 기대하는 결과

(verifier 는 측정이 끝나기 전에 이 절을 읽지 않는다.)

- 실제 pi0.5 에서도 guidance off 는 baseline 과 bit 단위로 같아야 한다. hook 이 입력을 그대로 반환하기 때문이다.
- injected sphere 에서 CFM 과 NUDGE 는 EE clearance 를 양수로 만들 것으로 기대한다. OmniGuide 는 효과가 작고, Lan-o3dp 는 중간일 것으로 기대한다.
- CFM 은 depth 점 구름 전체를 obstacle 로 주면 수정량이 커서 task 성공을 해칠 수 있다. closed-loop 에서는 SafeLIBERO obstacle (privileged primitive) 로 주는 것이 원 논문 조건이다.
