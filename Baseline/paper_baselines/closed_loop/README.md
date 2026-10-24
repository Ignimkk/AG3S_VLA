# closed_loop — comparison methods in the LIBERO / SafeLIBERO closed loop (CLOSED-LOOP-METHODS)

The methods of `methods/` (filters and sampling guidance) are attached to the **same wire protocol** as the
V11AB / VLA-only Panda server, so the existing client `benchmark/runners/libero_safe_client.py` runs them unchanged.
No file outside this folder is modified.

```
launch_method.py (.venv-ag3s)  create_run · GPU lock · suite registry · lp.client_argv · metrics
  ├─ serve_method.py (.venv-openpi-live, openpi WebsocketPolicyServer)
  │    request ──> MethodServer.infer
  │      none     : PandaVLAOnly(SeededPolicy(policy))                         (= serve_safe --robot panda --no-safe)
  │      filter   : same call → nominal 10×7 chunk ─┐
  │      guidance : PandaVLAOnly(SeededPolicy(GuidedPolicy(policy)))         │  pi0.sample_actions(guidance_fn=hook)
  │      wire_obs.scene_observation (depth · K · T · q · EE, no privileged key)
  │      perception.scene_points (depth lift − robot spheres − table − target box)
  │      method_calls/calls.jsonl + ep<k>_seed<s>.npz (nominal / served chunk, Δa, h, timings)
  │                                                 │ pipe (framing: JSON + .npy, numpy 1 ↔ 2)
  │                                                 └─> sidecar.py (.venv-baselines): SafetyFilter.filter,
  │                                                     Grounding DINO (filter obstacles, target box)
  └─ libero_safe_client.py (.venv-libero, unchanged): official loop, rows 0-4 of each chunk, collision monitor
metrics.py: TSR / CAR / ETS (official), SL5c safe success, KNOWS SR / CR / SSR, multi-link 11 mm, per-call stats
```

## Arms (`arms.py`) — all non-privileged

| arm | family | obstacles | target exclusion | main deviations (all arms: no VLM / API; perception from the wire only) |
|---|---|---|---|---|
| `none` | — | — | — | identical server path to VLA-only |
| `knows` | filter | GDINO "object" boxes − target box (`--obstacles detector`) or depth components (`depth`) | GDINO target box | attention target exclusion (Eq. 4) → detector box |
| `aegis` | filter | same | same | GLM-4.5V naming → generic GDINO prompt (= offline `aegis:detector_all`) |
| `multilink_cbf` | filter | same | same | moving-hazard flow omitted |
| `omniguide`, `constrained_flow_matching`, `lan_o3dp_guidance_adapted`, `nudge` | guidance | depth lift (− robot, − table, − target box), padded to 1024 points (Lan-o3dp unpadded) | GDINO target box (`--target detector`) or none | post-Euler hook, `x0_hat` from the sampling noise; CFM paper uses GT spheres |

Filters edit the whole 10-row chunk at the policy call (later rows on the method's EE prediction); the client
executes rows 0-4 (replan 5), as for VLA-only and V11AB. The original filters run every env step.

## Guidance execution

`guided.py`: the pi0.5 sampler is compiled once; the call's `guidance_fn` (a new closure per observation) runs
through a fixed `jax.pure_callback` hook on the JAX CPU backend (`--guidance-exec callback`, default), compiled
once per jaxpr structure. `--guidance-exec inline` instead passes the closure constants into the jitted sampler
(recompiles the sampler when the structure changes). Both give the same chunk (`tests/test_server_identity.py`).

## Commands

```bash
source scripts/setup/env.sh
# unit tests (CPU)
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/third_party/openpi/src \
  .venv-openpi-live/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests \
  --ignore=Baseline/paper_baselines/closed_loop/tests/test_metrics_launcher.py
PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests/test_metrics_launcher.py
# CPU smoke, real SafeLIBERO client, recorded SL5c VLA-only chunks as the policy
.venv-ag3s/bin/python Baseline/paper_baselines/closed_loop/launch_method.py --cpu-smoke --fake-policy recorded \
  --fake-source-run 20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only --method none \
  --safety-level I --task-ids 0 --episode-ids 0 --run-id clm-cpu-smoke
# GPU (verifier): one arm, SafeLIBERO spatial Level I dev
.venv-ag3s/bin/python Baseline/paper_baselines/closed_loop/launch_method.py --method aegis --safety-level I \
  --task-ids 0,1,2,3 --episode-ids 0,1,2,3,4 --run-id clm-pilot --sidecar-device cuda \
  --ref-run 20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only
# metrics of any run
PYTHONPATH=$PWD .venv-ag3s/bin/python -m Baseline.paper_baselines.closed_loop.metrics <run> [--ref <run>]
```

Handoff with results and V3 commands: `AG3S/ag3s/docs/handoff/CLOSED-LOOP-METHODS-I.impl.md`.
