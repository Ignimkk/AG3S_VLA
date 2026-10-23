# omniguide — OmniGuide [19] (sampling guidance)

| 항목 | 내용 |
|---|---|
| 원 논문 | Song et al., *OmniGuide*, arXiv:2603.10052 v1 |
| 출처 repo | `daniilidis-group/OmniGuide@d649d69849bfc0c56bc822cdbe1f5f8a9fad2706` (umbrella). 실제 code 는 submodule `OmniGuide-Code/omniguide-openpi@663f1cb2021282cf673fdfd737800b04ba80c893` |
| license | `omniguide-openpi`: **Apache-2.0** (LICENSE 파일 확인). umbrella repo 에는 LICENSE 파일이 없지만 code 를 담고 있지 않다 |
| 가져온 것 | `src/openpi/models_pytorch/pi0_pytorch.py` 의 collision cost (:1008-1028), VLM-pointer attraction (:1090-1104), guidance update (:1557-1583), freeze zone (:956-1002), EE clearance filter (:346-400), wrist probe (:193-196). 기본값은 `config/guidance/vlm_pointer_collision_sim.yaml`, `collision.yaml` 에서 가져왔다 |
| 이 폴더 | `guidance.py` (`OmniGuideGuidance`), `get_method("omniguide")` |
| 상태 | unit test 와 recorded-frame CPU smoke 를 통과했다. pi0.5 GPU smoke 는 아직 돌리지 않았다 (명령은 아래에 있다) |

## 무엇을 하나

cost 를 최대화하는 방향으로 action chunk 를 민다 (repeller + attractor).

```
cost  = Σ_k 0.1·1[sdf(p_k)≤0.2]·log sdf(p_k) + Σ_k Σ_w 0.01·1[sdf(q_kw)≤0.1]·log sdf(q_kw)   (collision)
      − Σ_{k<10} ‖p_k − centroid(target)‖²                                              (attract)
guide = ∂cost/∂x0_hat ; row 별 norm 을 ≤ 1.0 으로 clip ;  x_next ← x_next + 0.4·guide
```

- `p_k`: decode 된 EE (grip site) path. `q_kw`: hand frame 의 wrist probe 4점.
- obstacle (repeller): 비-target 기하. 기본은 depth lift 다. 전체 pixel 에서 target, robot, ignore mask 를 빼고, EE 에서 5 cm (gripper open) 또는 10 cm (closed) 안의 점도 뺀다.
- target (attractor): `extras["target_points"]` 또는 target mask lift 를 쓴다. A 의 grounding provider 결과, ATS, GT (privileged) 중 하나다.
- freeze zone: EE 가 target 점에서 0.1 m 안에 들어오면 guidance 를 끈다. hysteresis 상태는 call 사이에 유지하고, `reset()` 으로 지운다.
- `guidance_types=("collision",)` 로 collision 만 켤 수 있다. 원 `collision*.yaml` 과 같은 설정이다.

## 원 방법과 다른 점

| 원 OmniGuide | 여기 |
|---|---|
| π0.5 **DROID** joint-velocity action → FK 로 tip path 를 만든다 | pi05_libero **EE-delta** action → OSC response model 로 EE path 를 만든다 (orientation 은 현재 값으로 고정) |
| `x_t` 에서 network 를 거쳐 backprop 한다 (`x_t + (−t)v_t`) | post-Euler hook 이라 network 를 거칠 수 없다. `x0_hat` (noise 기반 또는 identity) 에 대한 gradient 를 쓰고 `∂x0/∂x_t = I` 로 둔다 |
| ESDF: 1 cm voxel, 20회 min-pool (Chebyshev 근사, 약 0.2 m 에서 cap) | 점 / primitive 까지의 정확한 Euclidean 거리. 점 반경은 voxel/2 |
| point cloud: VGGT 또는 depth + SAM2 / CLIP 으로 target 과 arm 을 걸러낸다 | 우리 wire depth + target mask (provider 또는 GT) + robot mask |
| attractor: VLM pointer + SAM2 로 찾은 target centroid | grounding provider 또는 GT target 점의 centroid |
| best-of-N 초기 noise 선택 (`num_init_candidates`) | 생략한다 (model 을 N 번 돌려야 함) |
| subtask monitor / VLM progress tracker | 생략한다 |

**크기 차이 주의.** LIBERO 에서 action 1 단위가 만드는 EE 이동은 steady state 에서 약 0.0125 m/step 이다. DROID joint velocity 의 tip 이동보다 약 10배 작다.
그래서 원래 factor 0.4 로는 attractor 효과가 작다. 합성 test 에서 약 0.3 mm 였다.
[9] (NUDGE) Table III 도 LIBERO-Object 에서 OmniGuide 가 unguided 와 비슷하다고 보고했다 (success 30 vs 30, contact 12.52 vs 13.15).
repeller 는 log barrier 라 접촉 근처에서 gradient 가 커지므로 작동한다.
`factor` 는 parameter 로 바꿀 수 있다.

## 작동 증거 (CPU)

- unit test `tests/test_methods_guidance.py`: identity (bit 단위), repeller clearance 증가, waypoint 가 밀려남, attractor 방향, freeze zone identity, batch.
- recorded smoke `outputs/paper_baselines_smoke/methods/guidance/summary.json`, `libero_spatial_t0_e0` call 1:
  - injected 3 cm sphere: EE min SDF 가 −20.0 mm 에서 tracking −18.7 mm, constant +13.2 mm 로 바뀌었다.
  - scene: 이미 44.8 mm 떨어져 있어 거의 그대로다.

## 실행

```python
from Baseline.paper_baselines.methods.interface import get_method
g = get_method("omniguide")                       # 또는 guidance_types=("collision",), factor=...
fn = g.make_guidance_fn(obs, norm, noise=noise)    # obs: SceneObservation, norm: pi05_libero norm_stats["actions"]
actions = model.sample_actions(rng, observation, noise=noise, guidance_fn=fn)
```

GPU smoke 는 `methods/guidance_common/README.md` 를 본다.
