# Figure 1 variants (draft, 2026-10-13)

| File | Variant |
|---|---|
| `variant_A_performance.png` + `brief_variantA_en.txt` | **A. Performance teaser.** Top: VLA-only failures (pick next to an obstacle, place over a hurdle). Middle: the same tasks succeed with AG3S. Bottom: how (attention, 3D grounding, HSR, link-specific contact field). Right: small performance bars |
| `variant_B_limitation.png` + `brief_variantB_en.txt` | **B. Limitation first.** Top: what each existing approach gets wrong (VLA only / geometry-only filter / global target exclusion). Bottom: the AG3S mechanism and its two successes |

## Sources
- Pick and place scenes come from the T43 paired runs, all in **one environment**: episode 1982, the red overbar `t43p2_overbar` (config `T43/P2/configs/t43p2_ep1982_s19822.json`, bar height 0.23 m). Each frame is a MuJoCo replay of the recorded 66-D `qpos` (`frames.jsonl`), with the hurdle placed by `PickPlaceObstacleManager`. Path lines are FK of the recorded `qpos` (site `left_ee`). The red glow marks the closest gripper–hurdle point (`mj_geomDistance`).
  - `vla_pick_fail`: E0 (VLA only) seed 19822, step 197; gripper–bar −0.1 mm, grasp fails
  - `ag3s_pick_ok`: V11AB (AG3S main) seed 19822, step 140; grasped and lifted
  - `vla_place_fail`: E0 seed 19823, step 245; gripper–bar 0.6 mm, place fails
  - `ag3s_place_ok`: V11AB seed 19823, step 212; carried over the bar into the basket
- Mechanism panels: T53 C1a ep1842 renders from `../../figure*/assets/`. The HSR curve is the recorded sub-task probe output of that run.
- **Performance bars are placeholder values** for layout only (`assets/perf_placeholder.json`). Keep the placeholder label or replace them with other sourced values whose status is shown.

## Regenerate
```bash
MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/paperbanana_fig1_scenes.py
MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/paperbanana_fig1_variants.py
```
