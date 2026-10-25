# Multi-link safety filter [1] — `Baseline/paper_baselines/methods/multilink_cbf`

| item | value |
|---|---|
| paper | Agarwal & Raghunathan, *Multi-Link Safety Filtering for VLA Policies Around Moving Hazards*, arXiv:2609.40007 (main.pdf [1]) |
| official code | **none** (arXiv page links only a project page, yathag.github.io/multilink-safety-filter). Implemented from the arXiv HTML v1 |
| license | project code; the barrier helpers are the AEGIS official `utils.py` (MIT), imported unchanged |
| interface | `adapter.py` → `MultiLinkCBFFilter` (`get_method("multilink_cbf")`), `methods/interface.py` `SafetyFilter` |
| state | runs end to end on recorded SL3 LIBERO frames (CPU smoke); unit tests pass. Closed-loop rollout: next STEP |

## Method (paper Eq. 2) and what is implemented

```
min_{u, zdot}  ||u - u_nom||^2_W + ||zdot - zdot_nom||^2_Wz
s.t.  a^u_k G_k(x) u + a^z_k zdot_k + alpha h_k(x, z_k) >= 0      for every guarded body k (and every hazard)
G_k = J_k J_ee^+,   W = I_6/25,  W_z = I_3,  alpha = 10,  OSQP via CVXPY
```

| paper | here |
|---|---|
| 5 ellipsoids: gripper, wrist, 3 overlapping forearm ellipsoids (SO-101); elbow / upper arm unguarded | Panda: `gripper` (hand + fingers), `wrist` (link6 + link7), `forearm_0..2` (link5 split in 3 along its axis, one sphere overlap). Each = MVEE of the surface samples of the **adapter sphere model** (`AG3S/adapters/panda/model.py` → cuRobo `franka.yml` spheres, read only), refit at the first action of each call, then rigid; coverage of every sample point checked |
| `h_k`, `a^u_k`, `a^z_k` = AEGIS rotating-supporting-hyperplane barrier | official AEGIS `compute_h_coeffs_3d`; body-frame coefficients mapped to the base frame with `@ R_k^T` (finite-difference test) |
| `G_k = J_k J_ee^+` | Jacobians of each ellipsoid centre and the grip site by central finite differences of the adapter URDF FK; damped pseudo-inverse (lambda = 1e-3) |
| `u_nom` = policy delta-pose × fixed scale (not given) | `u = 5 a[:6]` (AEGIS scale), output `a[:6] = u / 5`; base-frame twist (robosuite OSC rotation delta is world-frame) |
| `zdot_nom` "inherited from AEGIS" | `10 dh/dz`, `z` integrated with `dt = 0.05` and renormalised |
| infeasible QP → nominal action, logged | same |
| hazard: Qwen3-VL names it, Grounding DINO boxes, depth crops, workspace / outlier / density filters, MVEE, at reset | AEGIS perception of `methods/aegis` (`perceive_obstacles`): `gt_seg` (privileged, default), `detector` (Grounding DINO + privileged names or generic prompt), `ellipsoids` (given). No VLM. Detector crop `crop_mode="depth_trim"` (default) / `"box"` (V5) and `gt_obstacle_set` as in `methods/aegis/README.md` item 3 |
| **moving hazards**: centre carried by sparse LK optical flow every 5 steps, with gates and template re-search | **omitted.** Option `track_centers=True` re-measures each hazard's point-cloud centroid at every call and shifts the frozen ellipsoid (centre-only model, re-perception instead of flow). Default off = the paper's "frozen at reset" ablation |
| edge deployment (prefix trimming, 5 flow steps) | not applicable (filter only) |

Chunks: for action k > 0 of a chunk, `q` is predicted with `q += J_ee^+ [0.0115 a_xyz, 0.115 a_rot]` (measured EE
gain, `safety_common`); T = 1 per env step needs no prediction.

## Run

```bash
source Baseline/paper_baselines/setup/paper_baselines_env.sh
CUDA_VISIBLE_DEVICES="" .venv-baselines/bin/python -m pytest Baseline/paper_baselines/methods/multilink_cbf/tests -q
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 .venv-baselines/bin/python -m \
    Baseline.paper_baselines.methods.smoke_safety_filters --calls 0 12 --arms multilink_cbf aegis
```

Timing (CPU, 1 thread, 5 bodies x 5 hazards): about 7 ms per action after the first call (cvxpy import and the
per-call ellipsoid fits make the first call about 1 s).
