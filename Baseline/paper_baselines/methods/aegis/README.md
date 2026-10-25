# VLSA / AEGIS [10] — `Baseline/paper_baselines/methods/aegis`

| item | value |
|---|---|
| paper | *VLSA: Vision-Language-Action Models with Plug-and-Play Safety Constraint Layer* (AEGIS), arXiv:2512.11891 (main.pdf [10]) |
| official code | `benchmark/SafeLIBERO` = THU-RCSCT/vlsa-aegis @ `2457feed5968ae803926e178c8ce8243b9ecdcf9` (submodule, **not modified**): `main/main_aegis.py`, `main/utils.py` |
| license | MIT (repo `LICENSE`, LIBERO copyright inherited) |
| interface | `adapter.py` → `AegisSafetyFilter` (`get_method("aegis")`), `methods/interface.py` `SafetyFilter` |
| state | runs end to end on recorded SL3 LIBERO frames (CPU smoke, GT-seg and Grounding DINO obstacle sources); unit tests pass. Closed-loop SafeLIBERO rollout: next STEP |

## How the official code is used

| official | here |
|---|---|
| `utils.py` `compute_h_ij`, `compute_h_coeffs_3d`, `vector_hat`, `project_matrix`, `mvee_cvxpy`, `fit_ellipse` | **imported unchanged** (`official.py` loads the file; robosuite is stubbed during the import only, since its one robosuite name is used by `get_point_cloud`, which is not called) |
| `main_aegis.py:149-164` EE ellipsoid (`eef + R [0,0,-0.08]`, semi-axes 0.06/0.12/0.11, 0.2 for tall objects) | ported, `adapter.ee_ellipsoid` |
| `main_aegis.py:184-233` per-episode obstacle fit, `z = (p2 - p1)/|.|` | ported, fit at the first `filter` call after `reset` |
| `main_aegis.py:301-354` per-step CBF-QP (`u_ref = [5 R1^T a_xyz, 5 a_rot, 10 mu]`, `W = diag(1/25 x6, 1 x3)`, `0.2 a_v u_v + 0.2 a_w u_w + a_z u_z + 10 h >= 0`, cvxpy OSQP; `z` update `dt = 0.05`) | ported with line numbers in comments |
| `utils.py:257-316` `filtering_points` (world workspace box, 80 % nearest centroid, open3d DBSCAN eps 1e-4 / 50) | ported (`perception.py`): box applied after base→world with `T_world_base`; DBSCAN re-implemented with a scipy KD-tree (open3d not installed) |
| `utils.py:155-255` `get_point_cloud` (Grounding DINO box → depth crop → back-projection) | ported as `ObstacleCloudSource("detector")` with A's HF Grounding DINO (`providers/grounding_dino.py`, `grounding-dino-tiny` = SwinT, thresholds 0.35 / 0.25) |
| `utils.py:447-499` `obstacle_detection` (GLM-4.5V via zhipuai API names the obstacle) | **not used — external API forbidden.** Replaced, see below |

## Differences from the official layer

1. **VLM replaced** (the obstacle *name* the GLM returned):
   - `obstacle_source="detector", detector_prompt="obstacle_names"` (a): names from scene metadata
     `extras["privileged/obstacle_names"]` (SafeLIBERO obstacle objects; in LIBERO: every non-target,
     non-destination object + non-support fixture) → Grounding DINO → crop → fit. **Privileged.**
   - `detector_prompt="all_objects"` (b): one generic prompt (`"object"`), every box except the one overlapping the
     detected target phrase (IoU > 0.5) and table-sized boxes (> 50 % of the image). Not privileged.
   - `obstacle_source="gt_seg"` (default for tests/smoke): instance masks from the simulator segmentation instead
     of detector boxes, target + destination excluded. **Privileged geometry** (upper bound of the perception).
2. **Several obstacles**: one barrier row and one `z` per obstacle (official: exactly one, the VLM's choice).
   With SafeLIBERO metadata this is normally one obstacle.
3. **Box crops: robot points removed** (`robot_filter=True`): crop points within sphere radius + 2 cm of the Panda
   collision spheres at `q` are dropped (kinematics, not GT). Without it Grounding DINO boxes that cover the arm
   (seen on SL3 call 0) put an obstacle ellipsoid on the gripper.
   **Crop (`crop_mode`, BASELINE-BENCH-READY):** `"depth_trim"` (default) keeps only the object at the box centre:
   robot spheres + Panda base envelope removed, support plane (estimated table height) and points > 0.58 m above it
   removed, the 26-connected component (1 cm grid) under the central 40 % of the box kept, range percentiles 2-98.
   This stands in for the official `filtering_points` workspace box, which needs the suite name (not on the closed-loop
   wire: V5 records `workspace: null`, so in V5 nothing but the 80 % trim was applied). The official DBSCAN
   (`eps = 1e-4` m) is a no-op in practice and stays as ported. Per-box clouds are associated in 3D across boxes and
   cameras before merging. `"box"` = the whole box (CLOSED-LOOP-FIX V5, ellipsoid semi-axis up to 0.77 m at V5 call 0).
   `gt_seg` obstacle set (`gt_obstacle_set`): `"all_non_target"` (default) or `"obstacle_names"` (only
   `privileged/obstacle_names`: SafeLIBERO's designated obstacle, as the official layer sees one VLM-named obstacle).
4. **Chunks**: the official layer filters one action per env step with the measured EE pose. A chunk (T > 1) is
   filtered sequentially with the EE pose predicted by the measured gain 0.0115 m / action unit / step
   (`safety_common.POS_GAIN_M`); call with T = 1 per env step to reproduce the official timing.
5. **QP failure**: official prints "No feasible solution" and then raises `NameError` (`main_aegis.py:337-339`,
   ending the episode); here the nominal action is returned and the status is logged.
6. **Inactive rows**: when the nominal action already satisfies every row it is returned bit-exactly (it is the QP
   optimum) instead of OSQP's approximate solution (`skip_inactive=True`); `z` is still updated with `10 mu`.
7. **Cameras**: official = agentview + backview at 1024 px; our recorded frames have `head` (agentview) + wrist at
   256 px. `gt_seg` uses all cameras, `detector` the external `head` camera.
8. Faithful, kept as is: the official code feeds `action[3:6]` (robosuite OSC world-frame rotation delta) directly as
   the body-frame angular velocity of the EE ellipsoid. `methods/multilink_cbf` uses consistent base-frame twists.

## Run

```bash
source Baseline/paper_baselines/setup/paper_baselines_env.sh
CUDA_VISIBLE_DEVICES="" .venv-baselines/bin/python -m pytest Baseline/paper_baselines/methods/aegis/tests -q
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 .venv-baselines/bin/python -m \
    Baseline.paper_baselines.methods.smoke_safety_filters --calls 0 12 --arms aegis aegis:detector_names aegis:detector_all
```

```python
from Baseline.paper_baselines.methods.interface import get_method
f = get_method("aegis", obstacle_source="detector", detector_prompt="obstacle_names")  # or "gt_seg" / "ellipsoids"
f.reset()                       # at episode start: obstacles are fitted at the first filter() call
res = f.filter(obs, chunk)      # obs: SceneObservation, chunk: (T, 7) LIBERO actions -> res.actions, res.intervened
```

Smoke output: `outputs/paper_baselines_smoke/methods/` (`*_aegis*.json|npz`, `libero_spatial_t0_e0_c<call>.png`).
