# paper_baselines — comparison baselines for the AG3S paper

One folder for the paper's comparison rows (PAPER-BASELINES, 2026-10-23). Every target-grounding baseline here
implements the same interface (`interface.py`): RGB (+ instruction) -> per-camera **target mask** in the RGB/depth
pixel frame. All of them are then lifted with the **same depth, the same back-projection and the same robot-removal
option** (`lift.py`) and scored with the **same 5 mm soft-voxel metrics** as the development branch (`eval/`).

```
RGB(+instruction) --[provider]--> per-camera target mask
depth + K + T_base_cam --[lift.py]--+--> base-frame target points / 5 mm voxels --> eval/metrics.py vs GT
```

The paper claim this serves: grounding the observed target surface from frozen-VLA attention **without an external
segmenter** (main.pdf Tab. 2 "External segmenter" row, Sec. 4.1 TBD "Grounded-SAM / YOLOE masks lifted with the same
depth", Fig. 2 "Prior: external segmenter").

## Catalog

Citation numbers are those of `main.pdf` (CVPR 2027 draft, 2026-10-23). The VLA-internal grounding rows
(`localization_heads`, `tokencut`) are detailed in `docs/catalog_vla_internal.md`, including where they deviate
from the original papers.

| baseline | cite | role in the paper | source repo @ SHA | license | form | status (2026-10-23) | run |
|---|---|---|---|---|---|---|---|
| GT oracle (`gt_oracle`) | — | upper bound / lift check for Tab. 2 (privileged: simulator segmentation of the BDDL target) | this repo (`task_target` + `segmentation_geom`) | — | `re-impl` | CPU smoke + tests pass | `--providers gt_oracle` |
| Grounded-SAM (`grounded_sam`) | not in bib yet (text cites it in Sec. 4.1) | Tab. 2 "External segmenter", Fig. 2 prior column | IDEA-Research/Grounded-Segment-Anything @ `126abe6` (recipe/thresholds; weights via HF) | Apache-2.0 | `wrapper` (HF Grounding DINO + HF SAM v1) | CPU smoke (tiny/base weights) | `--providers grounded_sam` |
| Grounded-SAM-2 per frame (`grounded_sam2:per_frame`) | not in bib | Tab. 2 "External segmenter" (current recipe) | IDEA-Research/Grounded-SAM-2 @ `b7a9c29` | Apache-2.0 | `vendored-run` (pinned clone installed as `sam2`) + HF Grounding DINO | CPU smoke | `--providers grounded_sam2:per_frame` |
| Grounded-SAM-2 causal track (`grounded_sam2:track`) | not in bib | same, temporal variant (detect once, SAM 2 streaming tracker) | same @ `b7a9c29` (`IncrementalObjectTracker` streaming path) | Apache-2.0 | `vendored-run` | CPU smoke + causality check | `--providers grounded_sam2:track` |
| YOLOE (`yoloe`) | [25] | Tab. 2 "External segmenter" (real-time open-vocabulary seg) | THU-MIG/yoloe @ `40cd606` (recipe); runtime `ultralytics==8.4.184` | **AGPL-3.0** | `wrapper` | CPU smoke (wiring; 11s weights found nothing at the default conf on the LIBERO frame) | `--providers yoloe` |
| Localization heads (`localization_heads`) | [14] | VLA-internal grounding prior (Sec. 2) | seilk/LocalizationHeads @ `9ffe219` | no LICENSE at SHA (README: MIT) | `partial` re-impl (B) | B: see `docs/catalog_vla_internal.md`; `sam_point` CPU smoke by A | B |
| TokenCut (`tokencut`) | [26] | attention/feature-graph object discovery prior (Sec. 2) | YangtaoWANG95/TokenCut @ `fed52cd` | MIT | `re-impl` + parity (B) | B: tests; real DINO ViT-S/16 CPU smoke by A | B |
| ATS V11AB (`ats_v11ab`, Coarse) | — (previous AG3S) | Tab. 2 "Coarse" row, same frames | this repo, `benchmark/runners/perception_replay.py` runs | — | `existing` (read from a perception run) | wired (`--ats-run`) | `--ats-run <perception run>` |
| π0.5 only | [18] | closed-loop "VLA only" arm (no grounding) | `benchmark/runners/launch_libero_pi05_only.py` | — | `existing` | existing runs (SL0/SL3/SL5) | see AG3S_BENCHMARK_LOG.md |
| SEAM | [31] | related work (chunk smoothing); not a comparison arm (user 2026-10-23) | `Baseline/seam_vla` | see its README | `existing` | not moved, not used | `Baseline/seam_vla/README.md` |

## Methods — safety filters and sampling guidance (`methods/`, PAPER-METHODS 2026-10-23)

All implement `methods/interface.py` (`SafetyFilter`: action chunk -> modified chunk; `SamplingGuidance`: openpi
`pi0.sample_actions(guidance_fn=...)` hook, pure JAX, openpi unchanged) and are reachable through `get_method(name)`.
"Works" = runs end to end on recorded SL3 dev frames and returns a modified chunk (CPU smoke + unit tests).
Closed-loop rollout and pi0.5-attached GPU smoke are the next step. Details: `PAPER-METHODS-C.impl.md`, `-D.impl.md`.

| method | cite | kind | source @ SHA | license | form | main deviations from the paper |
|---|---|---|---|---|---|---|
| KNOWS (`knows`) | [17] | SafetyFilter (ellipsoid EE CBF-QP) | moved from `Baseline/knows_vla` (history kept; old import path kept as an alias because V11AB `stages/geometry.py` uses its `mvee`) | — | `existing` + adapter | earlier P0–P3 re-implementation; obstacle shapes fixed at first call |
| VLSA / AEGIS (`aegis`) | [10] | SafetyFilter (CBF-QP) | official `benchmark/SafeLIBERO/main/{utils,main_aegis}.py` @ `2457fee` (helpers imported, QP ported with line refs) | see SafeLIBERO LICENSE | `wrapper` | **no VLM / no external API** (GLM step replaced by privileged names + Grounding DINO, generic Grounding DINO, or GT seg); one constraint per obstacle |
| Multi-link safety filter (`multilink_cbf`) | [1] | SafetyFilter (per-link CBF-QP) | no public code — implemented from the paper | — | `re-impl` (`partial`) | moving-hazard optical flow omitted; Panda links as ellipsoids fitted to the adapter sphere model |
| OmniGuide (`omniguide`) | [19] | SamplingGuidance (attractor/repeller energy) | `daniilidis-group/OmniGuide` @ `d649d698` + `omniguide-openpi` @ `663f1cb2` | Apache-2.0 (openpi part) | `port` | EE-delta instead of DROID joints; no best-of-N, no subtask monitor |
| Constrained flow matching (`constrained_flow_matching`) | [5] | SamplingGuidance (CBF-constrained) | `VLA-Safety-ConstrainedFlowMatching` @ `b91153ad` (reference only) | **none** — not copied | `re-impl` | jit SQP instead of SciPy SLSQP; paper uses GT spheres on SafeLIBERO |
| Lan-o3dp guidance (`lan_o3dp_guidance_adapted`) | [15] | SamplingGuidance (obstacle cost) | `qqianfeng/lan-o3dp` @ `b89c2898` | MIT | `adapted` | guidance only, moved onto pi0.5; **not the original policy's result** |
| NUDGE (`nudge`) | [9] | SamplingGuidance | no public code — implemented from the paper | — | `re-impl` (`partial`) | hand + finger spheres only, no IK; d_safe 0.02 chosen (not given in paper) |
| RoboGround / GuidedVLA / OBEYED-VLA | [11] [13] [24] | trained grounding policies | fetched @ `25a86263` / `04be059e` / `bdf25556` | MIT / Apache-2.0 / none | `fetch + README` | need training or own checkpoints; not run |

### Code availability (`docs/survey.md`, checked 2026-10-23)

| cite | method | public code | note |
|---|---|---|---|
| [1] | Multi-link safety filter | no ("coming soon") | reports SafeLIBERO with `pi05_libero` |
| [5] | Constrained flow matching | first-author repo, no license | reports SafeLIBERO with `pi05_libero`; not linked from paper |
| [9] | Zero-shot reactive avoidance (NUDGE) | none found | — |
| [11] | RoboGround | yes, MIT | no checkpoint; RoboCasa only |
| [13] | GuidedVLA | yes, Apache-2.0 | π0 checkpoint; LIBERO-Plus |
| [15] | Lan-o3dp | yes, MIT | no checkpoint; RLBench only |
| [19] | OmniGuide | yes | π0.5 DROID / RoboCasa |
| [24] | OBEYED-VLA | perception module only, no LICENSE | real robot only |

No exemption / All-link exemption ablations map to existing V11AB flags (`--target-field-policy relax` /
`exclude-all`); see `docs/exemption_ablations.md` (All-link is only meaningful on `--links arms` profiles).

`Baseline/seam_vla` and `benchmark/SafeLIBERO` keep their paths (existing runs and imports depend on them).
`Baseline/knows_vla` is now only an import alias for `methods/knows`.

## Layout

```
Baseline/paper_baselines/
├── interface.py       TargetQuery · TargetMaskResult · TargetMaskProvider · registry (lazy; no torch at import)
├── lift.py            pixel frames (raw / OpenCV / policy-224 mirror) · robot removal · V11AB backproject · voxel keys
├── weights.py         pinned weights (HF revision / release URL), sha256, presets default (large) / smoke (small)
├── providers/         gt_oracle · grounded_sam · grounded_sam2 · yoloe · grounding_dino · sam_backend (A)
│                      localization_heads · tokencut · frozen/ (B)
├── eval/              metrics.py (2D, soft voxel IoU, bootstrap) · offline_grounding.py (harness, create_run) ·
│                      native_metrics.py (each paper's own metrics) · cloud_labeling.py (fair point-level ATS comparison)
├── external/          fetch_sources.py · sources.json · .gitignore (clones untracked)
├── setup/             requirements-baselines.in (top pins) · requirements-baselines.txt (lock) · setup_venv.sh ·
│                      paper_baselines_env.sh
├── methods/           interface.py · knows · aegis · multilink_cbf · omniguide · constrained_flow_matching ·
│                      lan_o3dp_guidance_adapted · nudge · roboground/guidedvla/obeyed_vla (README) · evaluate_offline.py
├── scripts/           smoke_providers.py (A) · b_smoke_*.py (B) · check_track_memory.py (G)
├── docs/              metrics.md (metric registry per baseline/benchmark) · providers.md · survey.md ·
│                      exemption_ablations.md · catalog_vla_internal.md
└── tests/
```

## Setup

```bash
python3 -I Baseline/paper_baselines/external/fetch_sources.py          # pinned clones -> external/<name>/
bash Baseline/paper_baselines/setup/setup_venv.sh                      # .venv-baselines (py3.11, uv, lock file)
source Baseline/paper_baselines/setup/paper_baselines_env.sh           # HF_HOME/TORCH_HOME/YOLO_CONFIG_DIR -> .cache/paper_baselines
```

Weights download on first use into `.cache/paper_baselines/` (ids, revisions and sha256 in `weights.py` and each
provider's `provenance()`).

## Run

CPU wiring smoke (one recorded frame, smallest weights; output `outputs/paper_baselines_smoke/`):

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 .venv-baselines/bin/python Baseline/paper_baselines/scripts/smoke_providers.py --provider grounded_sam
```

Offline grounding evaluation (GPU only under the shared lock, `flock -w <s> /tmp/ag3s-vla-gpu-validation.lock ...`,
after `nvidia-smi --query-compute-apps` is empty):

```bash
.venv-baselines/bin/python -m Baseline.paper_baselines.eval.offline_grounding \
  --source-run 20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0 --scene-gt outputs/verify/SL4/raw/scene \
  --providers gt_oracle grounded_sam grounded_sam2:per_frame grounded_sam2:track yoloe \
  --prompt-modes phrase instruction --robot-removal none \
  --ats-run 20261012T225820_46a08d6ff644_sl4-perception-dev --run-id paper-baselines-sl3dev
```

Recorded inputs with segmentation (2026-10-23): SL3 dev `20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0`
(10 ep, scene GT `outputs/verify/SL4/raw/scene`, ATS `20261012T225820_46a08d6ff644_sl4-perception-dev`) and SL4b dev
`20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4` (40 ep, scene GT `outputs/verify/SL4b/raw/scene`).
The SL5* closed-loop runs carry no segmentation.
