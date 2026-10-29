# CVPR 2027 main figures, version 1 (2026-10-18 KST)

Built on `feature/ag3s-development` @ `f15e5872` (uncommitted work tree). CPU only (OSMesa MuJoCo replay +
matplotlib). Nothing under `AG3S/`, `tests/` or `paper/latex/` was changed; the `ifg_overview` and `paperbanana`
assets were read, never rewritten.

## Files

| File | Size (in) | Aspect | Min font | Content |
|---|---|---|---|---|
| `fig01_teaser.{pdf,png,json}` | 6.875 x 2.86 | 2.40 | 7.0 pt | 2x2 hurdle scenes (VLA only / VLA + AG3S x pick / carry) + "Measured" box + 4-panel "how" strip |
| `fig02_overview.{pdf,png,json}` | 6.875 x 3.35 | 2.05 | 7.0 pt | framework strip + three contribution cards (identity / extent / permission), prior and AG3S side by side |
| `fig03_pipeline.{pdf,png,json}` | 6.875 x 3.30 | 2.08 | 7.0 pt | five stages, contact-interface stage with Eq. (linkquery) notation |
| `fig06_episode_strip.{pdf,png,json}` | 6.875 x 1.80 | 3.82 | 7.0 pt | the "one episode, moment by moment" strip removed from Fig 3, with a vector re-plot of the phase inputs |
| `fig04_heldout.{pdf,png,json}` | 3.25 x 2.30 | 1.41 | 7.0 pt | single column: paired per-episode IoU (G:N vs B) + IoU by grasp state for the four locked arms |
| `fig07_failure.{pdf,png,json}` | 3.25 x 2.00 | 1.62 | 7.0 pt | single column: T58 ep1830 adjacent-post contact with the hand's query spheres + cover-row distances |
| `assets/t58_ep1830_contact.{png,json}` | 1400 x 1400 px | | | T58 replay render (sources with sha256, reconstructed distances) |
| `assets/{vla_pick,vla_carry_fail,ag3s_pick,ag3s_carry}.{png,json}` | 1600 x 1200 px | | | hurdle replay frames + sidecars (sources with sha256) |

PDFs are vector; raster panels are embedded at 400 dpi (`PDF_DPI`), fonts are embedded TrueType (DejaVu Sans,
`pdf.fonttype = 42`, math in STIX). Every font size is audited at save time (`fonts.min_pt` in each sidecar); the
floor is 7 pt at print size, because the figures are authored at the CVPR text width (6.875 in), not scaled.

## Scripts

- `paper/tools/plotting/cvpr_v1_fig1_scenes.py` renders the four hurdle frames (MuJoCo replay of the recorded
  66-D qpos in `frames.jsonl`, obstacle from the run's `t43p2_ep1968_s19681.json` config, profile `t43p2_overbar`,
  model XML from the 2026-10-17 IFG-HURDLE snapshot). Derived from `paperbanana_fig1_scenes.py`; the camera is the
  same for both rows of a column (paired view). `explore OUTDIR` writes contact sheets used to pick frames.
- `paper/tools/plotting/cvpr_v1_fig7_scene.py` reconstructs the T58 ep1830 contact scene exactly as the verifier's `verify/T58/p/diag_cover_t58.py` (captured `.mjb` + reset state, post from `configs/t58_ep1830.json`, qpos of control row 55, left-hand spheres of server `chunk_00016` carried rigidly with `EE_BODY_L` from t = 48) and asserts the recomputed distances against `diag/cover_ON_ep1830.json` (0.44 mm hand-post, 32.12 mm outside) before writing.
- `paper/tools/plotting/cvpr_v1_compose.py` composes all six figures from the IFG-era assets
  (`paper/figures/ifg_overview/assets`), the paperbanana figure-2 assets (card 3 only) and the new scene frames.
  Layout follows `ifg_teaser_pipeline_compose.py` (teaser_fig1 / pipeline) and `ifg_overview_compose.py`.

## Hurdle episode for Fig 1 and why

Episode 1968 ("Put the apple in the basket."), seed 19681, for both rows. The VLA-only row is the T43 W E0 run
(2026-10-05); the AG3S row is the IFG-HURDLE SURF run (2026-10-17) with the same episode, seed, obstacle config and
profile. Each row shows two moments of one run (t = 200 / 234 for E0, t = 156 / 196 for SURF).

- SURF ep1968 is `success_ct` for both seeds with no hurdle or neighbour contact rows (agg.json items), so the
  AG3S row is a clean success of the paper's method; the earlier draft used V11AB (coarse extent), which is now an
  ablation.
- The E0 run grasps the apple and is stopped when a finger contacts the bar during the carry. This is the
  representative VLA-only failure (11/14 E0 runs grasp, 13/14 end in an obstacle stop); only 3/14 E0 runs fail
  before grasping, and ep1982 s19822 (the draft's pick-failure frame) is not a SURF success. The VLA-only pick
  panel therefore carries a neutral badge ("grasps the apple beside the bar"), and the failure is shown in the
  carry panel.
- Frame choice: E0 t = 200 = `grasp_t`; t = 234 = last recorded control step (contact, obstacle stop, closest
  gripper/bar point 0.5 mm, marked in red). SURF t = 156 = apple just lifted (`attach_t` 144, `grasp_t` 160);
  t = 196 = tool centre directly above the bar line (y = 0.104 m vs bar y = 0.106 m).

## Numbers printed (all repeated in the sidecars with path and JSON pointer)

- Fig 1: held-out locked IoU 0.773 (G:N) vs 0.891 (B); neighbour FP mass 176,070 vs 13,582 (-92 %); 34 episodes,
  2,523 frames (`heldout_locked/summary.json /main_success/...`); hurdle success 0/14 (T43W.verify.json
  `numbers.summary_per_arm["E0/H"].success`) vs 7/14 (IFG-HURDLE agg.json `arms.SURF.S_contact`). The coarse-extent
  11/14 and the A-arm values are in the sidecar only. Dev-set 0.837 / 0.893 are no longer printed.
- Panel labels inside the identity assets (`target 0.61`, `fruit 0.11`, `fruit 0.42`, `fruit 0.967`, `target 0.980`)
  are illustration values from the asset sidecars (`p_memory.json`, `c3_prior_raw_cosine.json` `/scores`), not
  results. The original raster labels (~3 pt at print size) are located by template matching, inpainted and
  redrawn as 7 pt vector text ("other fruit" shortened to "fruit").
- Fig 6: per-moment surface IoU of the development episode (`tp_episode.json`). Fig 2 and Fig 3 print no result
  numbers; the switch margin / alert threshold constants are described in words.

Registration status: the held-out and hurdle numbers are copied from raw `summary.json` / `verify.json` / `agg.json`
and are not yet registered in `paper/evidence/RESULTS.md` (EXP003 / EXP004 in the plan). Each sidecar says so.

## Choices

- Fig 1 right panel uses bars for IoU and leakage and a 14-square row per arm for the closed loop; "n = 14 paired
  runs, simulation" and "task success without a forbidden contact" are printed next to it. Note for the caption:
  E0 was recorded on 2026-10-05 (T43 W) and SURF on 2026-10-17, same episodes and seeds.
- Fig 2 card 3 reuses paperbanana figure-2 assets (T53 C1a ep1842 chunk 17 and T43 V11AB ep1982 chunks 22 / 34):
  an illustration of the contact interface from recorded runs of the coarse-extent configuration (sidecar
  `card3_provenance`). No "not yet evaluated" badges remain in any figure.
- Fig 3 keeps the five stages and writes the link rule as
  delta_{i,t}(x) = d_t^{\setminus R}(x) if i in L and e_t = 1, d_t(x) otherwise (Eq. linkquery of 03_method.tex).
  Matplotlib mathtext has no `cases`, so it is two lines.
- Card captions and badges were shortened until they fit at 7 pt; the LaTeX captions should carry the full
  sentences.

## Not done / limits

- The IoU baseline is labelled "radius + clustering" as in the brief; its source arm is `G:N` (rounds/R6 reference),
  which the sidecar records. Check the label against the final Table 2 wording.
- The noun-attention / surface / permission panels of the "how" strip and all of Fig 3 come from one development
  episode (T53 ep1842), as in the drafts.
- No SafeLIBERO or real-robot panels (not available).

## Regenerate

```bash
cd /mnt/dev/AG3S_VLA-development
MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_fig1_scenes.py      # 4 hurdle frames + sidecars
.venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_compose.py                           # fig01 fig02 fig03 fig06
.venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_compose.py fig01                     # one figure
MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_fig1_scenes.py explore /tmp/sheets  # frame picking
```

## Follow-up 2026-10-18 (second pass)

- Fig 3: the membership equation now reads (Λ + L) m = Λ y (manuscript Eq. 5); nothing else changed.
- Fig 4 (`fig04_heldout`): per-episode IoU means are recomputed from the locked per-frame rows
  (`heldout_locked/frames.json` for the success/episode labels, `run/rounds/R6/rows/*.jsonl` arm `G:N` and `top1-only-ns:N`,
  `run/rounds/R10b/rows/*.jsonl` arm `R6` (= A) and `Mem-first:prefix_L8:d0.05` (= B); valid frames = the G:N row's `valid` flag,
  2523 frames, 34 episodes). The script asserts that the recomputed episode means equal `summary.json` `/main_success/*/iou/episode_mean`
  to 1e-9 and that the pair counts match (B − G:N 32 better / 2 worse; A − G:N 34 / 0; B − A 14 / 6). The connectors therefore show
  **32 of 34 improved for B** — the "34 of 34" statement in the brief holds for A (ours without reassociation), not for B; the figure
  prints no count. Right panel bar heights = `/by_grasp_state/<state>/<arm>/iou/episode_mean` (no value labels printed); legend labels
  as requested; colours grey / blue-grey / light gold / gold.
- Fig 7 (`fig07_failure`): built from the recorded data without approximation (see `cvpr_v1_fig7_scene.py`). Printed numbers: "32 mm"
  (= `T58.verify.json` `numbers.on_diagnosis.1830.cover_rows[t=55].closest_point_outside_spheres_mm` = 32.12) and the cover-row curves
  (hand-mesh and sphere-union distance to the post, t = 48–55). The ON arm of T58 is the coarse-extent serving configuration; the same
  episodes fail with the VLA alone (sidecar `provenance`).
- Regenerate: `MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_fig7_scene.py` then
  `.venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_compose.py` (all six) or `... fig04 fig07`.

## Third pass 2026-10-18 (naming aligned with the revised manuscript)

- Names used verbatim: "VLA only", "Coarse" (previous grounding: radius growth + clustering + ball exemption + sphere/capsule
  robot filter = arm G:N), "Surface (vol. mask)" (A-fk), "Surface" (A), "Surface + ID (ours)" (B) or "AG3S" where space is short;
  module acronyms ATS / HSR / LTE / CCR; "accepted target surface"; attention and features come from "an instrumented copy" of the
  frozen checkpoint.
- Fig 1: left badges are now parallel ("grasps the apple" grey / green); baseline bar label "Coarse". New variant
  `fig01_teaser_noCL.{pdf,png,json}` (6.875 x 2.62 in) without the closed-loop rows, bar charts enlarged; the original
  `fig01_teaser` (2.86 in) is kept. Both are produced by `fig01(cl=...)` in the compose script.
- Fig 2: "Prior" → "Baseline" in all three cards (cards 1–2 compare against our previous configuration); card titles carry the
  acronyms ("1 What: identity (ATS)", "2 Where it ends: extent (ATS)", "3 Who and when: permission (HSR + LTE)", 7.5 pt); the
  top-strip box reads "LTE: link-specific field δ_{i,t}".
- Fig 3: stage titles "Frozen VLA", "Surface graph (ATS)", "Seeds → region (ATS)", "Identity and state (ATS · HSR)",
  "Contact interface (LTE · CCR)" (two lines each); stage 1 says "an instrumented copy exposes: …"; stage 5 uses the exclusion set
  𝒳_t: d_t^{∖𝒳} in the link rule and "𝒳_t: accepted target surface only". Equation (Λ + L) m = Λ y unchanged.
- Fig 4: legend labels "Coarse", "Surface (vol. mask)", "Surface", "Surface + ID (ours)"; the left panel prints
  "32 improved · 2 worse" (summary.json /main_success/paired_iou/B - G:N better / worse; registered in the sidecar).
- Regenerate: `.venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_compose.py` (now fig01, fig01_noCL, fig02, fig03, fig04, fig06, fig07).

## Fourth pass 2026-10-18 (user feedback; finished by the lead after the figure agent hit its session limit)

- Tidiness: `Fig.layout_audit()` (compose script) checks text–text overlap, text–arrow/line crossing, text-to-panel-border
  distance (≥ 0.05 in) and badge heights at save time; `CVPR_V1_STRICT=1` (default) refuses to save with violations.
  All of fig01, fig01_noCL, fig02, fig03, fig04 save with 0 violations. Fixes this pass: fig03 stage captions shortened so
  they stay inside their columns ("robot mask by rendering / whole target kept (orange)", "seeds → membership m",
  "cosine to memory", "contact links only"); link-rule equation lines re-spaced; fig02 card-3 title shortened
  ("3 Who and when: permission", acronyms moved to the subtitle), extent badge "✓ kept apart", footer lines shortened to
  the figure width, segmenter-mask prior panel cropped to show the mask; fig01 instruction line moved down (was clipped).
- VLA input is RGB: fig02 top strip reads "RGB + instruction" → frozen VLA; depth enters only the geometric layer.
- fig02 prior rows are prior practice, not our previous configuration: external segmenter (illustration, simulator label as
  stand-in), volume exemption (recorded ball), fixed object role (recorded carry frame). Header "Prior".
- fig03 stage 2 top panel (`assets/p_robot_mask_v2`, re-rendered by `cvpr_v1_fig3_assets.py`): the whole observed target is
  orange and the rendered robot mask purple; the red partial overlay / outline was removed. The other IFG panels used in
  fig01 / fig02 / fig03 (`t_surface_v2`, `p_region_v2`, `t_permission_v2`, `c2_ours_*_v2`) tint the target mesh orange and
  the touching neighbour light blue under the unchanged 5 mm nodes so the target reads as one object.
- No closed-loop success number is printed on any figure; the manuscript uses `fig01_teaser_noCL`.
- Regenerate: `MUJOCO_GL=osmesa PYTHONPATH=. .venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_fig3_assets.py` then
  `.venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_compose.py fig01 fig01_noCL fig02 fig03 fig04`.

## Fig 3 v2 (2026-10-18, user request: less text, clearer method, include the missing scene field)

- `fig03()` in the compose script is rewritten as a two-row story (old version kept as `fig03_legacy()`): row 1 ATS =
  1 noun attention (frozen VLA) → 2 rendered robot mask → 3 depth-continuous surface graph → 4 target surface → 5 same object?;
  row 2 = 6 cuRobo ESDF scene field (new, `paperbanana/figure3/assets/f3_scene_field`) → 7 contact links → 8 grasp state
  (recorded probe p(place) + annotated grasp state, vector) → 9 carried object (new, `f3_attached_object`) → 10 refine + check
  (schematic). One label per panel; all explanations moved to the LaTeX caption. 6.875 × 2.66 in, min font 7 pt, layout audit 0.
- Panels 6 and 9 come from the recorded coarse-extent serving run of the same episode (T53 C1a ep1842); the caption says so.

## Fig 3 v3 (2026-10-18): architecture diagram replaces the storyboard

User: the ten-frame storyboard (now `fig03_storyboard()`) is not a CVPR method figure. `fig03()` now draws
inputs → frozen VLA (token strip, taps: noun attention / features / KV) → ATS (seed patches [real] → depth-continuous graph with
depth break and seeds [schematic] → target surface 𝒯_t [real]; identity check raw vs centered cosine of the R10 example) →
HSR (phase and grasp chips) → LTE (cuRobo TSDF→ESDF; two exact 2-D distance-transform slices d_t vs d_t^{∖𝒳} with wrist / finger
query spheres; attached object after grasp [recorded]) → CCR (refine A_t s.t. h ≥ 0 [schematic], check prefix → execute / hold).
Action lane A_t runs from the VLA to CCR. 6.875 × 3.05 in, min font 7 pt, layout audit 0.
