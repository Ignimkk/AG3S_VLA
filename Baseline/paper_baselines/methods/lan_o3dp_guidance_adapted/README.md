# lan_o3dp_guidance_adapted — Lan-o3dp [15] obstacle guidance, pi0.5 이식판

> **원 Lan-o3dp 결과가 아니다.**
> Lan-o3dp 는 RLBench 용 task 별 DP3 계열 diffusion policy 이고, checkpoint 가 공개되지 않았다.
> 여기서는 학습 없이 쓰는 **obstacle-cost guidance 만** pi0.5 의 `guidance_fn` 으로 옮겼다.
> 이 결과를 "Lan-o3dp" 수치로 보고하지 않는다 (develope_PLAN_SUMMARY §3.1).

| 항목 | 내용 |
|---|---|
| 원 논문 | Li et al., *Lan-o3dp* (ICRA 2025), arXiv:2407.00451 v3 |
| 출처 repo | `qqianfeng/lan-o3dp@b89c2898d8548f7dfde63c44eeb733a59f0c5222` |
| license | **MIT** (LICENSE 확인. diffusion_policy 의 Columbia AI & Robotics Lab copyright) |
| 가져온 것 | `diffusion_policy/policy/diffusion_unet_pcd_policy.py:155-166` (guidance 호출), `:395-417` (`gradient_field`). `scheduling_ddim.py:313-316` (적용식). `diffusion_policy/real_world/real_data_conversion.py:184-215` (`preprocess_obstacle_pcd`) |
| 이 폴더 | `guidance.py` (`LanO3dpGuidanceAdapted`, `obstacle_circles`) |
| 상태 | unit test 와 recorded-frame CPU smoke 를 통과했다. GPU smoke 는 아직 하지 않았다 |

## 원 guidance

- obstacle: 물체 별 point cloud 하나를 원 하나로 만든다.
  - 중심 = xy 평균. 반경 = xy bbox 대각선의 절반.
  - 실물 robot 에서는 `real_env.py:633-640` 이 반경을 0.15 m 로 덮어쓴다.
- field: 예측 clean trajectory (`pred_original_sample`, unnormalized) 의 waypoint 가 원 안에 있으면 중심에서 바깥을 향하는 unit vector 를 준다.
  - xy 만 쓰고 z 는 0 이다. obstacle 들에 대해 합한다.
- 적용: `prev_sample[..., :2] += normalize(field)[..., :2] · 0.03 / sqrt(ᾱ_prev)` (DDIM).

## 이식에서 바꾼 것

| 원 | 여기 |
|---|---|
| DDIM, `pred_original_sample` | flow matching. `x0_hat` 는 `guidance_common.core` 의 noise 기반 또는 identity 추정이다 |
| `1/sqrt(ᾱ_prev)` | `1/max(1 − t_next, 0.1)` (다음 sample 에서 x0 의 계수) |
| action = 절대 EE 위치. field 를 위치에 바로 더한다 | action = EE delta. waypoint 를 `step_m · u_k` 만큼 밀고, 그 변위를 OSC response model Jacobian 의 damped pseudo-inverse (`damping 1e-2`) 로 action 변화로 되돌린 뒤 normalized 단위로 바꾼다 |
| 상수 0.03 (normalized. RLBench 작업공간에서 약 3 cm) | `step_m = 0.01` m 로 새로 정했다. LIBERO 한 chunk 는 최대 약 10 cm 만 움직인다 |
| object point cloud: GPT-4 로 이름 → GroundingDINO → SAM → Cutie | scene 의 obstacle 점 구름을 2 cm xy grid 의 연결 성분으로 나눈다. primitive 를 주면 그 외접원을 쓴다 (sphere, cylinder 는 r·√2, box 는 footprint bbox 대각선/2) |
| DDIM `clip_sample: True`: 매 step `pred_original_sample` 을 normalized 범위 [-1, 1] 로 자른다 (`scheduling_ddim.py:296-297`) | `clip_sample=True` (기본, CLOSED-LOOP-FIX): field 는 잘린 `x0_hat` 위에서 계산한다. push 는 그 뒤의 clean chunk `x0_hat + push/(1−t)` 가 LIBERO action box (raw [-1, 1], `pi05_libero` quantile norm stats 로 normalized 단위로 바꾼 값) 안에 남도록 줄인다. network 가 이미 box 밖에 둔 값은 더 밖으로 밀지 않는다. push 가 0 이면 `x_next` 는 bit 단위로 그대로다 |
| circle 수 = 물체 수 (가변) | `MAX_CIRCLES=16` 으로 padding (중심 FAR, 반경 0). obstacle 점 구름의 FAR padding 점과 padding primitive 는 무시한다. 한 jaxpr → server 당 compile 1 회 |

### 원 정의와 남는 차이 (CLOSED-LOOP-FIX)

- 원 방법의 action 은 절대 위치라서 `clip_sample` 범위 = 작업공간 범위다. 여기서는 EE delta 라서 범위 = OSC 입력 범위 [-1, 1] 이다. pi0.5 의 quantile 정규화에서 이 범위는 normalized 로 x: [-1.300, 1.075], y: [-1.246, 1.170], z: [-1.067, 1.067] 이다.
- 원 sampler 는 network 의 x0 예측도 매 step 자른다 (guidance 와 무관하게). 여기서는 pi0.5 sampler 를 고치지 않으므로 guidance 가 더한 부분만 제한한다. 그래서 `none` arm 과 같은 call 에서 push 가 0 이면 결과가 bit 동일하다.
- V3 GPU smoke 에서 |a| 최대 27.28 (nominal 0.96), 실행된 183 행이 |xyz| > 1 이었다. 원인은 `1/max(1−t, 0.1)` (최대 10 배) push 가 pi0.5 의 clamp 없는 sampler 에서 누적된 것이다 (t=0.9 step 의 `dx_l2` 33.5).

## 작동 증거 (CPU)

- unit test:
  - repeller 가 clearance 를 늘린다 (3 cm sphere: −10.0 mm 가 −2.3 mm (prim), −1.2 mm (points) 로 바뀐다).
  - 점 구름에서 원 1개가 복원된다 (중심 오차 < 5 mm, 반경 3-5 cm).
  - guidance off 와 obstacle 없음은 bit 단위 identity 다.
- recorded smoke, call 1:
  - injected sphere: EE min SDF −20.0 mm 가 −5.6 mm (tracking), +9.1 mm (constant) 로 바뀌었다.
  - xy 만 밀기 때문에 장애물 앞 waypoint 는 뒤로 밀린다 (진행이 늦어진다).
  - constant 대체 sampler 에서는 `1/(1−t)` 가 누적되어 max |Δa| 가 12-16 raw 까지 커졌다. CLOSED-LOOP-FIX 의 `clip_sample` 이후 served xyz 는 [-1, 1] 안이다 (`closed_loop/tests/test_closed_loop_fix.py`).
  - 실제 network 에서 얼마나 되는지는 GPU smoke 로 확인해야 한다.

## 실행

```python
g = get_method("lan_o3dp_guidance_adapted", step_m=0.01)   # radius_override=0.15 -> 원 실물 설정
fn = g.make_guidance_fn(obs, norm, noise=noise)
```
