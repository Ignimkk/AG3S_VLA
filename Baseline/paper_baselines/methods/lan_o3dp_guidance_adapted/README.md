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

## 작동 증거 (CPU)

- unit test:
  - repeller 가 clearance 를 늘린다 (3 cm sphere: −10.0 mm 가 −2.3 mm (prim), −1.2 mm (points) 로 바뀐다).
  - 점 구름에서 원 1개가 복원된다 (중심 오차 < 5 mm, 반경 3-5 cm).
  - guidance off 와 obstacle 없음은 bit 단위 identity 다.
- recorded smoke, call 1:
  - injected sphere: EE min SDF −20.0 mm 가 −5.6 mm (tracking), +9.1 mm (constant) 로 바뀌었다.
  - xy 만 밀기 때문에 장애물 앞 waypoint 는 뒤로 밀린다 (진행이 늦어진다).
  - constant 대체 sampler 에서는 `1/(1−t)` 가 누적되어 max |Δa| 가 12-16 raw 까지 커진다. 실행할 때 clip 된다.
  - 실제 network 에서 얼마나 되는지는 GPU smoke 로 확인해야 한다.

## 실행

```python
g = get_method("lan_o3dp_guidance_adapted", step_m=0.01)   # radius_override=0.15 -> 원 실물 설정
fn = g.make_guidance_fn(obs, norm, noise=noise)
```
