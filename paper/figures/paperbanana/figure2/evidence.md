# Figure 2 evidence — conceptual overview

Prepared 2026-10-13 KST on `feature/ag3s-development` @ `cf38c9db`.


## Current version (2026-10-13): overview with pipeline comparison + one concrete scene

`reference_montage.png` and `brief_en.txt` now hold the **overview** layout. The previous conceptual version is kept as `*_v1_conceptual.*`.

| Panel | Source |
|---|---|
| Scene for all of (b) | T43 `V11/runs/V11AB/H/V11AB_ep1982_s19822` (AG3S main configuration, overbar `t43p2_overbar`). MuJoCo replay of the recorded `qpos`; the server chunk dumps are paired by `refined_chunk == traj.npz predicted_chunks` (segment 4) |
| ① attention | chunk 8, `attention_left_wrist` over a re-rendered `wrist_cam_l` |
| ② grounding | chunk 5, `filtered_cloud` / `target_points`. The pear is coloured by its replayed body position (display only) |
| ③ field | chunk 5, recorded `esdf_distance` vs **recorded `esdf_free_distance`** (target-free layer). Contact spheres = `exclusion.authorized_links` (`ee_left`, `ee_finger_l1/l2`) |
| ④ timeline | per-chunk `grasp.state` and `subtask.p` of the 75 chunks |
| ⑤ CCR | the carry chunk with the largest executed-row tool-centre shift. FK of `reference_chunk` / `refined_chunk` (site `left_ee`). The attached sphere is the chunk-16 attach record, carried with `EE_BODY_L` |
| (a) cards | left: Figure 1 `vla_place_fail` (E0 seed 19823). Middle: ③'s voxels with the target kept. Right: Figure 1 `ag3s_place_ok` (V11AB seed 19823) |

Generator: `paper/tools/plotting/paperbanana_fig2_overview.py`. Each asset has a sidecar in `assets/ex_*.json`.

## The three cases and their implementation (v1 conceptual)

| Panel | Mechanism in code | Flag / default | Status |
|---|---|---|---|
| (a) All surfaces as obstacles | Scene field = cuRobo TSDF → ESDF over all self-filtered depth (`AG3S/ag3s/runtime/pipeline.py:2571 _build_esdf`, `fields/curobo_builder.py:987 CuroboFieldBuilder`). With field policy `relax` the target stays in the field and contact links only get a relaxed margin | `--target-field-policy relax` is the **code default** | Implemented |
| (b) Global target exclusion (ball) | `pipeline.py:3151 _target_ball`, `curobo_builder.py:2038 _ball_exclusion`: a ball (surface-fit centre, quantile-1.0 radius + 1 voxel, cut above the support plane) whose TSDF seeds are deleted from the fine target-free layer. Global exclusion for all links = policy `exclude_all` | `--target-exclusion ball` (code default) | Implemented. Also the ball baseline used in T44/T46 |
| (c) Boundary-preserving extent | `curobo_builder.py:416 _ball_owned_split` (T44 I1). Inside the ball, a seed is removed only if it is nearer the target points than non-target observed points (tie 1 voxel, max distance 3 voxels in the evaluated arm) | `--target-exclusion ball-owned --owned-max-dist-voxels 3`: **off by code default, on in the evaluated arms** C1a (CL2, CL3, T53) | Implemented, offline-verified (below); closed-loop benefit **not established** |
| (c) Link-conditioned queries | `config.py:854` authorized links; `to_adapter.py:227-241` mask; `linearize.py:927-950`. Contact links of the active hand query the target-free layer, all other queried links query the scene field | `--target-field-policy exclude-authorized` (evaluated arms) | Implemented. **Scope:** `--links gripper`, so "other links" = the other hand's gripper. Wrist links (T46f A14) exist only in an uncommitted worktree; no result |
| F1 variant (not drawn) | `curobo_builder.py:469` `--nontarget-evidence surface` (T46) | Off by default; on in CL3 arm C1aF1; adopted by user decision T46c | Implemented; not used in the drawn run (T53 C1a uses `proximity`) |

## Verified evidence for the (b) → (c) difference

| Source | Key / location | Value | Reading |
|---|---|---|---|
| T44 I1 demo (64 frames, `gt_label`) | `AG3S/ag3s/docs/AG3S_T0T6_LOG.md:7056-7058` | distractor seeds removed inside the ball: `ball` 833 → `ball_owned` d=3 344; pooled reduction 0.587 (pre-set criterion ≥ 0.5, pass); target recall median 0.936 → 0.9345 | Offline. The ownership split keeps most neighbour surface that the ball erases |
| T46 held-out, 162 chunks | `T46.verify.json` `numbers.3_4_5_heldout.variants.{off,F1,F2}.lost_neighbour_seeds_all_chunks` | off 0 · F1 0 · F2 2463 | Relative to C1a (ball-owned). Removing ambiguous cells (F2) costs neighbour surface |
| T46 held-out | `…variants.{off,F1,F2}.collision_rows_all_chunks` | 887 · 401 · 136 | Target-surface residue that still blocks the hand |
| T44e CL2 / T46 CL3 closed loop | `T46-CL3.verify.json` `numbers.arms.{C1a,C1aF1,C1aF1F2}.t1_success.task_success` (n_runs 18) | 14 · 13 · 13 | No closed-loop difference established between extent variants |
| T46d | `AG3S/ag3s/docs/AG3S_T47_LOG.md:2486`, table at `:2595` | in the gripper scope, 0 of 349,920 non-authorized rows lie within 50 mm of the target, so LTE vs global exclusion is degenerate | Panel (c)'s link split is implemented, but its **benefit over (b) is not measured** |

No number appears in the figure. If the paper cites these, it records their source, scope, data type, and verification status; registration is optional.

## Assets

All renders are **MuJoCo 3.11 replays of the recorded `qpos`** from the client step records (record step j = control step 8j), re-rendered at 1600x1200 from chosen viewpoints with the CPU renderer. Nothing is re-simulated. Overlays come from the server chunk dumps, or from FK on the recorded chunks. Generator: `paper/tools/plotting/paperbanana_render.py` (renders) and `paperbanana_compose.py` (montage, F3 charts). Each PNG has a `.json` sidecar with source paths, SHA-256 hashes and the values used. The earlier chart-style assets are kept in `assets/diagnostic/`.

All four panels show the same recorded chunk (T53 C1a ep1842, chunk 17, the last pre-grasp chunk). The observed surface near the target is voxelised at 1 cm.

| File | What is real | What is approximate |
|---|---|---|
| `assets/f2a_scene_render.png` | observed surface (fused cloud of the chunk) as voxels; replayed scene | 1 cm voxels for display (the server's TSDF is 5 mm) |
| `assets/f2b_ball_render.png` | recorded ball (surface-fit centre, radius 60.2 mm). Voxels inside it are red ghosts: 156 inside, of which 43 are not near the target points | same voxelisation |
| `assets/f2c_ownership_render.png` | serving `_ball_owned_split` (T44 I1) with non-target evidence first filtered by the serving T46 F1 function `nontarget_surface_evidence` (adopted method, T46c). Counts: target-owned removed 122, neighbour-owned kept 32, ambiguous kept 2. F1 dropped 55 target-surface evidence points | **offline application** to this recorded chunk; the T53 run itself used `proximity` evidence |
| `assets/f2c_ownership_asrun_proximity.png` | the same split with the as-run `proximity` evidence: target 49, neighbour 32, ambiguous 75. Target-surface residue stays ambiguous, which is the T45 D-TF finding that motivated F1 | — |
| `assets/f2c_link_render.png` | recorded query spheres coloured by `target_field_exclude` (135 of 270: ee_finger_r1, ee_finger_r2, ee_right); gripper meshes made translucent so the spheres show | — |
| `reference_montage.png` | layout draft | — |

Existing research figures with the same phenomenon (read-only, Korean labels, not copied): `AG3S/ag3s/docs/figures/t45/t45tf-scene.png`, `t46/t46-scene.png`, `t46/t46e-d-scene.png`, `t43/t43dx-4-finger-scene.png`.

## Limits

- The figure is conceptual. Its renders show one recorded chunk, not a measured comparison.
- Panel (c) shows the adopted F1 evidence applied offline. Say so in the paper, or replace it with a run that has F1 on (CL3 arm C1aF1).
- The visible-target oracle extent (T46e, `--target-extent oracle-visible`) is a diagnostic upper bound that uses GT. It is not our method and is not drawn.
