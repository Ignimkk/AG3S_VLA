# Figure 3 v2: panels for Figure Labs (2026-10-26 KST)

Use `mockup_v2.png` as the layout reference, together with `FIGURELABS_PROMPT.md` and the images in `panels/`.
The mockup is `../action-hidden-state.png` with these regions replaced: the ATS middle panel, the identity strip under it, the LTE top box, the two LTE field slices and the HSR body.
CPU only. Recorded runs were read and never modified.

| Panel | File | Source (read only) | New? |
|---|---|---|---|
| inputs | `01a/b/c_input_*.png` | `ifg_overview/assets/p_in_{0,1,2}.png`, T53 C1a ep1842 frame f016 | copy |
| ATS seed patches | `02_ats_seed_patches.png` | `ifg_overview/assets/t_noun_attention.png`, ep1842 f016 | copy |
| ATS membership | `03_ats_membership.png` | `ifg_overview/assets/p_membership.png`, ep1842 f016 (graph membership m: red; seed-mass nodes: gold) | replaces the schematic |
| ATS surface | `04_ats_target_surface.png` | `cvpr_v1/assets/t_surface_v2.png`, ep1842 f016 | copy |
| HSR | `05_hsr_timeline.png` | **new**: `paper/tools/plotting/fig3_v2_hsr_timeline.py`. Recorded `summary.subtask.p` and grasp state of ep1842 (75 chunks, held = chunks 19–32) | replaces the text rows |
| ATS identity memory | `11a_ats_identity_memory.png`, `11b_ats_identity_now.png`, `11c_ats_identity_codes.png` | **new**: `paper/tools/plotting/fig3_v2_identity.py`. ep1842. Memory = target region at the approach frame f010; now = candidate regions of f016, each tinted grey→orange by its centered cosine to the memory (target 0.870, touching orange 0.578, next 0.145). Codes = the 32 dimensions with the largest \|memory\| of the centered prefix-L8 descriptors: row 1 memory, rows 2–4 the three most similar candidates. Details in `f3v2_identity.json` | replaces the raw/centered bar chart |
| LTE d_t | `06_lte_field_scene_dt.png` | **new**: `paper/tools/plotting/fig3_v2_lte_slices.py`. Recorded main field of the apple split-surface closed-loop run (IFG-APPLE-SPLIT), ep572, request 8, latched; the target apple touches a banana (gap 1.3 mm). Picked by `fig3_v2_lte_scan.py` from 669 recorded latched/closing requests of 24 episodes: on this slice, 1.0 % of the target interior is still an obstacle in the target-free field and 98.7 % of the banana interior stays an obstacle | replaces the old slice |
| LTE d_t^X | `07_lte_field_contact_dtX.png` | same request, recorded target-free field | replaces the old slice |
| LTE attached | `08_lte_attached_object.png` | `paperbanana/figure3/assets/f3_attached_object.png`, ep1842 chunks 19–26 | copy |
| LTE free-space carving | `10a_lte_carving_off.png`, `10b_lte_carving_on.png` | **new**: `paper/tools/plotting/fig3_v2_roi_carve.py`. Recorded closed-loop run with carving off (dynamic-map study, configuration B), ep211, request 47, latched. The apple was knocked away (857 mm from its first pose), so 614 surface voxels remain stale at its earlier poses (red). 10b applies the serving carving rule (`curobo_builder.seen_through`; footprint, pad 0 px, robot pixels ignored, margin 35 mm) to the same voxels and this frame's recorded depth from 3 cameras: 386 of the 614 stale voxels are cleared, and 5 head-camera rays through cleared voxels are drawn (last 30 cm). Details in `f3v2_roi_carve.json` | replaces nothing; new in the "cuRobo TSDF → ESDF" box |
| CCR | `09_ccr_refined_path.png` | `paperbanana/figure2/assets/ex_ccr.png`, T43 V11AB ep1982 chunk 20 | copy |

Run for the new LTE panels: `/mnt/dev/AG3S_VLA-storage/results/development/20261019T132719+0900_f15e587290f6_11408fd6d27f`.
The slice is the vertical plane through the target and nearest-neighbour centres.
- Background: field distance from 0 to 40 mm.
- Fill: field ≤ 0, coloured by the nearest ground-truth geom (orange target, blue neighbour, tan table). Fruit-coloured pixels within 10 mm above the table top are drawn as table (the support plane is protected scene geometry). This recolours 91 of the 112 target-coloured pixels left in d_t^X (the contact seeds where the apple meets the table) and 482 of 3,704 in d_t (the bottom of the apple). The view is cropped 2 mm below the table top.
- Dashed orange: ground-truth target outline. It appears only in d_t^X.
- Query spheres: no sphere crosses this plane at this request (the hand is still far away), so none are drawn.

Details are in `f3v2_lte_field.json`.

Limits to keep in mind:
- **The panels come from four recorded episodes.** ep1842 is a pear touching an orange, ep572 an apple touching a banana, ep211 an apple knocked off its place, and ep1982 an apple carried over a bar. The caption should not imply that all panels show one episode.
- **The table under the removed apple shows a gap in d_t^X.** The camera never saw that table top, because the apple hid it, so it stays unknown rather than becoming an obstacle.
- **Not every recorded request looks this clean.** At a later approach request (ep1345, request 15, the earlier version), an unobserved pocket inside the target stayed in d_t^X. The selected request shows the case where the observed surface covers the target.
- **The carving panels re-apply the rule offline to one frame.** Serving carves the TSDF voxels inside the region of interest on every frame. This figure instead applies the same rule to the recorded surface voxels of one frame, using the same input. That is why some red voxels remain in 10b: no camera saw past them in this frame. The rule also cleared 756 voxels that are not stale but lie within 15 mm of a ground-truth surface. Inside the drawn area, 64 voxels disappeared besides the 386 stale ones. Robot pixels come from MuJoCo segmentation; serving uses the FK-sphere depth mask.
- **A paired comparison was tried and not used.** At the same request, the carving-on closed-loop run of the same scene and seed has only 19 stale voxels against 614 with carving off. But the apple sits in very different places in the two runs (35 mm vs 857 mm from its first pose), so the same-input re-application was used instead (`--paired` mode keeps that option).
- **Robot removal is a short label, not an image.** Serving uses the FK sphere/capsule self-filter (`esdf.robot_mask = "spheres"`), and the rendered robot mask is on hold. Robot removal is standard preprocessing, so it appears only as the input label "depth (robot removed)".
- **No performance number is printed in any panel.**

Regenerate:
```bash
MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python paper/tools/plotting/fig3_v2_lte_slices.py extract \
  --run <RUN above> --ep 572 --request 8 --out <scratch>/lte_572_8.npz
.venv-ag3s/bin/python paper/tools/plotting/fig3_v2_lte_slices.py plot --npz <scratch>/lte_572_8.npz --outdir paper/figures/paperbanana/figure3/assets_v2 --fill support --band 10 --zmin -32
MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python paper/tools/plotting/fig3_v2_identity.py --outdir paper/figures/paperbanana/figure3/assets_v2
MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/fig3_v2_hsr_timeline.py
```
