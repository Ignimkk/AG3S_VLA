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

## Arms (`arms.py`) — non-privileged unless `--gt`

| arm | family | obstacles | target exclusion | main deviations (all arms: no VLM / API; perception from the wire only) |
|---|---|---|---|---|
| `none` | — | — | — | identical server path to VLA-only |
| `knows` | filter | GDINO "object" boxes (`--obstacles detector`) or depth components (`depth`) | pi0.5 attention, Eq. 2-4 (`--target attention`, default since F4a 2026-11-10); `--target detector` = GDINO target box (before F4a) | YOLOE → GDINO "object"; K, delta from P0b |
| `aegis` | filter | same | same | GLM-4.5V naming → generic GDINO prompt (= offline `aegis:detector_all`) |
| `multilink_cbf` | filter | same | same | moving-hazard flow omitted |
| `omniguide`, `constrained_flow_matching`, `lan_o3dp_guidance_adapted`, `nudge` | guidance | depth lift (− robot, − table, − target box), padded to 1024 points | GDINO target box (`--target detector`) or none | post-Euler hook, `x0_hat` from the sampling noise; CFM paper uses GT spheres |

Filters edit the whole 10-row chunk at the policy call (later rows on the method's EE prediction); the client
executes rows 0-4 (replan 5), as for VLA-only and V11AB. The original filters run every env step.

## Filter rate (F2, 2026-11-10) and KNOWS attention target (F4a)

`launch_method.py --filter-rate step` (filter arms; default `chunk` = the paragraph above, command lines unchanged):
server `serve_method --filter-rate step`, client `libero_safe_client.py --filter-every-step`. The official loop asks
the client every env step; every 5th step is the same policy request as in chunk mode (same env step, `ag3s/seq`,
reset, `policy_seed`, + `ag3s/filter_step = 0`), the server calls the policy, caches the nominal chunk and filters row 0
on the measured EE pose; the 4 steps in between send `ag3s/filter_step = k` with the current observation and the
server filters row `k` of the cached chunk without touching the policy (RNG unchanged). One filter step per env step
(AEGIS `z`, KNOWS normals / tracks advance once per step). Records: `calls.jsonl` one row per env step (`call_kind`,
`k`), `ep*_seed*.npz` `seq` / `nominal` / `served` per policy call + `step_*` per step; client `_calls.npz` adds
`step_actions` / `step_obs_index` / `step_k`; `metrics.method_calls` counts policy calls once and adds `steps`.
Note: `episodes.jsonl` `policy_calls` / `policy_latency_ms` (the official loop's own counters) count client requests =
env steps in this mode; the policy calls are `safe_client.calls`.

KNOWS detector arm (`--target attention`, default): `attention_tap.AttentionTap` between `SeededPolicy` and the policy
recomputes denoising step 0 of the sample that produced the chunk (same obs, the policy's own noise, same weights,
`return_attn_probs`) and hands `extras["knows/attention"]` (layer 12, head 3, agent-view 16 x 16, OpenCV orientation)
to the adapter; the served chunk is the policy's, untouched (`tests/test_filter_rate_attention.py`). KNOWS F3 (track
freeze while the arm intervenes) is an adapter default for both KNOWS arms (`--method-kwargs '{"track_freeze": "off"}'`
= before). Pre-fix runs: `--filter-rate chunk --target detector --method-kwargs '{"track_freeze": "off"}'`.

## Privileged variants (`--gt`) and benchmark jobs (BASELINE-BENCH-READY, 2026-10-25)

`launch_method.py --method <arm> --gt [SOURCE]` = the original papers' perception condition: SafeLIBERO ground-truth
obstacles. The client sends them only with `libero_safe_client.py --emit-privileged-obstacles` (default off; off = byte
identical, `tests/test_client_privileged.py`), the launcher adds the flag (and `--capture-segmentation` for `seg`).
Wire `ag3s/privileged/*` (world frame) -> `privileged.extras` -> `SceneObservation.extras["privileged/..."]` (base frame).
Manifest: `settings.arm.privileged = true`, `privileged_inputs`, `client_flags`.

| source | arms (default) | obstacles |
|---|---|---|
| `names` | `aegis` | Grounding DINO prompted with the active obstacle names (AEGIS's VLM output) |
| `seg` | `knows`, `multilink_cbf` | GT segmentation of the active obstacle objects (`obstacle_source="gt_seg"`) |
| `spheres` | `constrained_flow_matching` | one enclosing sphere per active obstacle (CFM paper) |
| `geoms` | `nudge`, `omniguide`, `lan_o3dp_guidance_adapted` | collision geoms as primitives; `repo` = CFM repo rule (mesh -> box) |

Every arm also gets `extras["suite"]` (`--suite`, AEGIS's official workspace box; `--no-suite-extras` = V5).

`jobs.py plan --arms none,aegis,aegis:gt,... --suites all --split dev|test --run-prefix P --out-dir D` writes
`run_parallel_gpu.sh` lines of this launcher (registry split labels, checked by the launcher's parser);
`jobs.py smoke` = every suite x level, t0 e0; `jobs.py aggregate --plan D/plan.json` = per suite x level x arm metrics
and pairing with the plan's own (re-run) `none`. `<arm>@v5` reproduces the V5 pilot behaviour (box crop, CFM without
action box, no suite extras). Until `run_parallel_gpu.sh` has `--launcher`, run with
`AG3S_PARALLEL_TEST_LAUNCHER="$PWD/.venv-ag3s/bin/python $PWD/Baseline/paper_baselines/closed_loop/launch_method.py"`.

## Guidance execution

`guided.py`: the pi0.5 sampler is compiled once; the call's `guidance_fn` (a new closure per observation) runs
through a fixed `jax.pure_callback` hook on the JAX CPU backend (`--guidance-exec callback`, default), compiled
once per jaxpr structure. `--guidance-exec inline` instead passes the closure constants into the jitted sampler
(recompiles the sampler when the structure changes). Both give the same chunk (`tests/test_server_identity.py`).

## CLOSED-LOOP-FIX (2026-10-24) — what changed before the pilot

| arm | V3 symptom | cause | change (original method's rule) |
|---|---|---|---|
| `lan_o3dp_guidance_adapted` | served `|a|` up to 27.28, 183 executed rows `|xyz| > 1` | `1/max(1-t, 0.1)` push accumulates in pi0.5's unclamped sampler | port of DDIM `clip_sample`: field on the clamped `x0_hat`, push limited so the implied clean chunk stays in the LIBERO action box (raw `[-1, 1]`); circles padded to 16, FAR padding ignored (one compile) |
| `knows` | `h_min` -0.447, 419 `h<0` steps, `qp_fail` 0 | (a) detector names re-ranked per call -> frozen shapes on the wrong centroids; unseen tracks reset to t=0; (b) paper Eq. 12 relaxed normals (eps box) can meet a violated row by tilting the normal (docs/11 §4) and has no infeasible case (docs/16 D3) | (a) tracks matched by nearest centroid, held at the last centroid while unseen (paper §3.2); (b) kept (paper behaviour), measured per step: `h_star*`, `cbf_ok`, `relax_m` vs `need_m` |
| `omniguide` | 38 / 56 calls skipped, 10 jit compiles | per-call target box (moves with the held bowl, wrong head-camera boxes) kept the freeze zone on; EE-clearance drop and conditional terms changed the jaxpr | episode-cached target (original: VLM + SAM2 once per subtask), exit test on the cloud cached at entry; FAR instead of drop, 0/1 weights (one compile) |

Latency keys (`serve_method.LATENCY_KEYS`): `ms.total` = wall time in `MethodServer.infer` (what the client waits
for, minus transport), `ms.diagnostics` = logging-only work after the chunk is final, `ms.method` = `total -
diagnostics`. `ms.serve` now equals `total` (in V2 / V3 records it was `total - diagnostics`); `metrics.py` reads
`total` and `total - diagnostics` for both. Guidance diagnostics use a jitted fixed-shape distance
(`path_metrics_fast`). New per-call keys: `served_max_abs_xyz`, `served_rows_out_of_range`, `guidance.ctx`
(OmniGuide freeze / target, Lan-o3dp circles), KNOWS `*_steps` diagnostics and `filter_obstacles`.

CPU replay harness: `serve_method --fake-source-key nominal` replays a method run's own observations with the VLA
chunks that run's method saw (`method_calls/ep*_seed*.npz`), `--fake-kind constant` is the surrogate sampler that
keeps every push (worst case for accumulation), `AG3S_SIDECAR_DETECTION_CACHE=<dir>` memoises Grounding DINO on disk
(replays only). Details and before / after numbers: `AG3S/ag3s/docs/handoff/CLOSED-LOOP-FIX-K.impl.md`.

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
