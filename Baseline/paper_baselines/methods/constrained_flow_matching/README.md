# constrained_flow_matching — Neuro-symbolic safety guidance via constrained flow matching [5]

| 항목 | 내용 |
|---|---|
| 원 논문 | English, Zheng, Ewetz, *Neuro-Symbolic Safety Guidance for Vision-Language-Action Models via Constrained Flow Matching*, arXiv:2607.01378 v1 |
| 출처 repo | `Dubascudes/VLA-Safety-ConstrainedFlowMatching@b91153adc2f40d0ecfd2ae9ece4b6a2aed31dc7e` (1저자 계정. 논문과 project page 는 이 repo 를 링크하지 않는다) |
| license | **license unknown.** repo 의 LICENSE 파일에 "A license for this project's original contributions has not yet been selected" 라고 적혀 있다 → **code 를 복사하지 않았다.** clone 은 `external/VLA-Safety-ConstrainedFlowMatching` 에서 읽기만 한다 (untracked) |
| 이 폴더 | `guidance.py` (`ConstrainedFlowMatchingGuidance`). 논문 식과 repo 의 갱신 규칙을 다시 적어 **재구현** 했다 |
| 상태 | unit test 와 recorded-frame CPU smoke 를 통과했다. pi0.5 GPU smoke 와 SafeLIBERO closed-loop 는 아직 하지 않았다 |

## 갱신 규칙 (논문 Eq. 7-11, repo `software_module/patches/openpi.patch` 의 `SafetyGuidance`)

```
p_0 = p_ee ,  p_j = p_{j-1} + Δt · a_j[:3]                    (a = post-Euler 상태 x_t 그대로, normalized)
h_j = clearance(ellipsoid(p_j; r=(0.06,0.12,0.11)), obstacle) − d_safe
제약  h_j − (1−γ) h_{j−1} ≥ 0 , j=1..H ;  위반이 없으면 수정하지 않는다
δ*  = argmin ½‖δ‖²  s.t. 제약(x + δ) ≥ 0 , δ ∈ R^{H×3} (위치 3 dim)
x_t ← x_t + guidance_scale · time_scale(τ) · δ*               (Euler step 뒤 = openpi hook 위치와 같다)
```

## 논문 · repo 의 SafeLIBERO 설정 (재현할 때의 기준)

| 항목 | 값 | 출처 |
|---|---|---|
| policy | `pi05_libero` (repo 는 PyTorch 변환본 `pi05_libero_pytorch`), H = 10, N = 10 Euler step | 논문 Sec. 5.1, repo `serve_policy.py` |
| obstacle | **simulator GT** sphere (위치와 반경). "isolating the control contribution from perception" | 논문 Sec. 5 |
| EE 모양 | 축 정렬 ellipsoid, 반축 (0.06, 0.12, 0.11) m | 논문, `SafetyConfig.eef_ellipsoid_radii` |
| d_safe | 0.01 m | 논문, `--safety-margin 0.01` |
| γ (CBF decay) | 0.9 | 논문, `cbf_gamma` |
| Δt | 0.01. 논문에는 값이 없고, repo 는 normalized action 에 바로 곱한다 | `SafetyConfig.dt` |
| guidance_scale | 1.0 (min-norm 해를 그대로 쓴다) | release-fixes.patch, README |
| schedule | `flat`, `min_guidance_time` 0.0 → 모든 denoising step 에 적용 | release-fixes.patch |
| solver | SciPy SLSQP (maxiter 50, ftol 1e-8). 실패하면 greedy per-step fallback | `_compute_cbf_qp_guidance` |
| 보고 결과 (Table 1, 4 suite × Level I/II 평균) | CAR 82.81 %, TSR 81.62 %. 비교: π0.5 18.69 / 50.88, AEGIS 77.85 / 68.13 | 논문 (Table 2 overall 84.00 / 84.62 와 다름. 원문 불일치) |

CAR 은 충돌이 없는 episode 비율이고 TSR 은 task 성공률이다. episode 수는 논문에 없다.

## 원 방법과 다른 점

| 원 | 여기 |
|---|---|
| SciPy SLSQP (CPU, numpy, GPU→CPU 복사) | jit 가능한 SQP 다. `n_outer=4` 번 `jax.jacfwd` 로 선형화하고, 매번 Hildreth dual coordinate ascent (`n_sweeps=60`) 로 min-norm 점을 정확히 구한다. greedy fallback 은 포팅하지 않았다 (마지막 iterate 를 반환한다) |
| obstacle: SafeLIBERO env 의 GT primitive (sphere, box, cylinder, ellipsoid) | `privileged/obstacle_geoms` 를 주면 같은 GT 경로를 쓴다 (sphere, box, cylinder 지원. ellipsoid 는 아직 없다). 주지 않으면 depth lift 점 구름을 쓴다. **이 경우 장면의 비-target 물체가 모두 제약이 된다** (원 논문에는 없는 설정이다) |
| clearance: 각 primitive 의 closest surface point → ellipsoid 거리. 논문 Eq. 6 은 중심 거리 − r_obs | repo 방식을 따랐다 (primitive 마다 closest point → ellipsoid clearance → min) |
| `eef_pos` 는 robosuite world frame | base frame 의 `ee_pose` 를 쓴다 (LIBERO 는 회전이 같아 결과가 같다) |
| 기본 trajectory model `integrator` (Δt · normalized action) | 기본값은 원과 같다. `path_model="osc_response"` 로 OSC response model 을 고를 수 있다 |
| `bellcurve`, `posthoc` schedule | 둘 다 구현했다 (`posthoc` 은 `posthoc_correct()` 로 최종 chunk 에 한 번 적용한다) |

## 작동 증거 (CPU)

- unit test:
  - 수정 뒤 제약 `min g ≥ −1e-3` 를 만족한다.
  - 위반이 없으면 δ = 0 이다.
  - repeller 가 clearance 를 늘린다 (prim 과 points 둘 다).
  - guidance off 와 obstacle 없음은 bit 단위 identity 다.
- recorded smoke, call 1:
  - injected sphere: EE min SDF −20.0 mm 가 +39.3 mm (tracking), +38.1 mm (constant) 로 바뀌었다.
  - scene (depth 점 구름): 44.8 mm 가 47.8 mm 로 바뀌었다. 다만 ellipsoid 가 커서 (y 반축 12 cm) 주변 물체 때문에 수정량이 크다 (max |Δa| 5.6 raw, 실행할 때 [-1, 1] 로 clip 된다).

## 실행

```python
g = get_method("constrained_flow_matching")        # safety_margin=0.01, cbf_gamma=0.9, guidance_scale=1.0 ...
fn = g.make_guidance_fn(obs, norm)                  # obs.extras["privileged/obstacle_geoms"] = SafeLIBERO obstacle (권장)
actions = model.sample_actions(rng, observation, guidance_fn=fn)
```

GPU smoke 는 `methods/guidance_common/README.md` 를 본다.
