# nudge — NUDGE / zero-shot reactive obstacle avoidance [9] (sampling guidance)

| 항목 | 내용 |
|---|---|
| 원 논문 | Guo & Kavraki, *Zero-Shot Reactive Obstacle Avoidance for Generative Robot Policies* (NUDGE = Nudge Update via Differentiable GEometry), arXiv:2609.35231 v2 (2026-09-29) |
| 출처 repo | **없음.** 논문과 abstract 에 URL 이 없다. GitHub 검색 0건 (`docs/survey.md` [9], 2026-10-23) |
| license | 해당 없음. 논문 Sec. III (Example III.2, III.6, Algorithm 1) 과 Sec. V-B 를 읽고 구현했다 |
| 분류 | flow-matching sampling 안에서 작동하므로 `SamplingGuidance` 로 둔다 (`SafetyFilter` 가 아니다) |
| 이 폴더 | `guidance.py` (`NudgeGuidance`, `calibrate_kappa`, `KAPPA_SL3_DEV`) |
| 상태 | unit test 와 recorded-frame CPU smoke 를 통과했다. GPU smoke 는 아직 하지 않았다 |

## 논문 방법 → hook

```
A0_hat  = A^k − t_k v_θ                                   (flow matching clean-chunk 추정)
C_SDF   = Σ_i Σ_{l,j} [ d_safe − Φ(T_l(q_{t+i}) c^{l,j}) + r^{l,j} ]_+²       (robot sphere vs workspace SDF)
A0_til  = A0_hat − w_k ∇C_SDF(A0_hat) ,  w_k = ρ / (κ (1−ρ))
A^{k−1} = (t_{k−1}/t_k) A^k + ((t_k − t_{k−1})/t_k) A0_til
```

이 update 와 unguided Euler step 의 차이는 `((t_k−t_{k−1})/t_k)(A0_til − A0_hat)` 뿐이다. 그래서 post-Euler hook 에서는 다음과 같이 **정확히 같은 식** 이 된다.

```
x_next ← x_next + (Δ/t_k)·(clip(x0_hat − w g) − x0_hat) ,  t_k = t_next + Δ
```

unit test `test_nudge_update_rule_matches_paper_flow_form` 이 이 식을 확인한다.

## 논문의 LIBERO-Object 설정과 이 구현

| 항목 | 논문 (LIBERO-Object, π0.5) | 여기 |
|---|---|---|
| policy | fine-tuned π0.5 (이름 미기재), EE-delta, H = 10, K = 10 Euler | pi05_libero, H = 10, K = 10 |
| ρ | 0.85 (모든 step) | 0.85 (`guided_steps` 로 마지막 n step 만 켤 수 있다. UR5 설정은 2) |
| κ | calibration set 의 non-zero ℓ∞ gradient 중앙값 (값 미기재) | `KAPPA_SL3_DEV = 0.0086`. 같은 정의로 SL3 dev 135 call 에서 구했다 (episode 0-9, call 1, 4, …, 19. scene 과 injected sphere. `--calibrate-kappa` 로 재현한다) |
| d_safe | 미기재 | 0.02 m |
| SDF | cuRoboV2 ESDF, metric depth 1 view, 2 cm voxel, robot / target / basket / table mask | depth lift 점 구름 (2 cm voxel, 점 반경 1 cm) 까지의 정확한 거리. target, robot, ignore mask 를 쓴다. `privileged/obstacle_geoms` 도 받는다 |
| robot sphere | Panda sphere 41개 + 7 cm payload sphere (`panda_hand` 고정 offset) | cuRobo `franka.yml` 의 **panda_hand 와 finger sphere 22개만** 쓴다 (EE 와 함께 강체로 움직인다). `extras["holding"]` 이면 7 cm payload 를 더한다 |
| decoder | EEF delta → IK 로 joint trajectory → FK 로 sphere 위치 | OSC response model 로 EE path 를 만든다. orientation 은 고정하고 hand sphere 를 평행이동한다. **IK 와 팔 link sphere 는 생략했다** (팔이 장애물에 닿는 경우는 보지 못한다) |
| clipping | "mapping preserves the update and subsequent clipping" (규칙 미기재) | nudge 가 움직인 원소만 raw [-1, 1] 범위로 clip 한다 (`clip_raw`) |

`kappa=None` 이면 call 마다 ℓ∞ 로 정규화한다. 이 경우 스치는 정도의 접촉에도 ρ/(1−ρ)=5.67 normalized step 이 걸려 overshoot 한다.
recorded smoke 에서 hand SDF 가 +8.4 mm 에서 −21 mm 로 나빠지는 것을 보고 calibrated κ 를 기본값으로 바꿨다.

## 작동 증거 (CPU)

- unit test:
  - 논문 flow 식과 수치가 일치한다.
  - 3 cm sphere 에서 repeller 가 clearance 를 −10.0 mm 에서 +36.8 mm 로 늘린다 (prim 과 points 둘 다). 옆과 위로 피한다.
  - guidance off 와 obstacle 없음은 bit 단위 identity 다.
- recorded smoke, call 1:
  - scene: hand min SDF 가 +8.4 mm 에서 +33.9 mm (tracking), +20.0 mm (constant) 로 늘었다.
  - injected: EE min SDF 가 −20.0 mm 에서 +8.4 mm (tracking), +20.2 mm (constant) 로 늘었다.

## 실행

```python
g = get_method("nudge")                    # rho=0.85, kappa=0.0086, d_safe=0.02
fn = g.make_guidance_fn(obs, norm, noise=noise)   # noise = sample_actions 에 준 것 (x0_hat 추정)
```

GPU smoke 는 `methods/guidance_common/README.md` 를 본다.
