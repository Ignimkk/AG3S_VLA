# guidance_common — sampling guidance 공통 부품

OmniGuide · constrained flow matching · Lan-o3dp (adapted) · NUDGE 가 함께 쓰는 부품이다.
모두 pure JAX 이고 `pi0.sample_actions(..., guidance_fn=...)` 의 jit 안에서 돈다.
openpi 는 고치지 않는다.

| 파일 | 내용 |
|---|---|
| `core.py` | `GuidanceBase` (`make_guidance_fn`, batch `vmap`, `x0_hat` 추정, provenance), `surrogate_sample` (CPU 대체 sampler), normalize helper |
| `kinematics.py` | `EEPathModel`: LIBERO EE-delta action → EE path (base frame). OSC response model (`y_k = α y_{k-1} + β·0.05·a_k`, `robots/panda/description/osc_response_fit.json`), `integrator`, `commanded` |
| `geometry.py` | `Obstacles` (point · sphere · box · cylinder, 고정 shape padding), signed distance 와 closest point |
| `scene.py` | `SceneObservation` → `GuidanceScene` (obstacle · target · EE pose · hand sphere). 우선순위: `privileged/obstacle_geoms` → `obstacle_points` → depth lift |
| `recorded.py` | 녹화된 SL3 dev call → `SceneObservation` + 기록된 action chunk (A 의 `eval/offline_grounding.Episode` 사용) |
| `evaluation.py` | smoke 공용 helper (`path_metrics`, `injected_obs`, `SCENE_KW`) |
| `gpu_smoke.py` | 실제 pi05_libero 를 붙이는 GPU smoke. **준비만 했고 실행하지 않았다** (verifier 몫) |

## hook 과 원 방법 사이의 간격

openpi hook 은 **post-Euler** 이고 `guidance_fn(x_next, t_next)` 만 받는다. `v_t` 와 `x_t` 는 받지 않는다.
네 방법 모두 clean-chunk 추정 `x0_hat` 이 필요해서 `x0_mode` 로 다음 둘 중 하나를 쓴다.

- `noise`: caller 가 `sample_actions(noise=...)` 에 준 noise 를 `make_guidance_fn(..., noise=noise)` 로도 넘긴다.
  그러면 `x0_hat = (x_next - t·ε)/(1 - t)` 로 추정한다.
- `identity`: noise 가 없을 때 `x0_hat = x_next` 로 둔다.

gradient 는 `x0_hat` 에 대해 구한다. `∂x0_hat/∂x_t = I` 로 놓는다. frozen network 를 거치는 backprop 은 hook 에서 할 수 없다.

## CPU 대체 sampler (`surrogate_sample`)

pi0 와 같은 Euler `while_loop` 를 쓴다. network 자리에는 기록된 chunk 를 목표로 하는 직선 flow velocity 를 넣는다.
guidance 효과가 다음 step 에 얼마나 남는지를 양 끝 두 경우로 본다.

- `tracking`: `v=(x_t-x*)/t`. 최악의 경우다. 중간 수정이 줄어들어 마지막 call 만 그대로 남는다.
- `constant`: `v=ε-x*`. 최선의 경우다. 모든 수정이 그대로 누적된다.

실제 network 는 이 둘 사이에 있다. GPU smoke 로 확인해야 한다.

## 실행

```bash
# unit test (CPU)
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
  .venv-openpi-live/bin/python -m pytest -q Baseline/paper_baselines/tests/test_methods_guidance.py
# recorded-frame smoke (CPU, ~1 GB RSS, ~1 min)
source scripts/setup/env.sh
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD \
  .venv-openpi-live/bin/python Baseline/paper_baselines/scripts/smoke_guidance_methods.py
# GPU smoke (verifier, GPU lock)
export OPENPI_DATA_HOME=$PWD/.cache/openpi XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer=
( flock -w 3600 9 || exit 1
  PYTHONPATH=$PWD .venv-openpi-live/bin/python -m Baseline.paper_baselines.methods.guidance_common.gpu_smoke \
      --methods omniguide constrained_flow_matching lan_o3dp_guidance_adapted nudge --episode 0 --calls 1 2 --inject-sphere
) 9>/tmp/ag3s-vla-gpu-validation.lock
```

`.venv-openpi-live` 에는 아무것도 설치하지 않았다. 이미 있는 JAX 0.5.3, numpy 1.26.4, scipy 만 쓴다.
