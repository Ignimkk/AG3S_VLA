# External-segmenter providers and the GT oracle (Implementer A)

All providers follow `interface.py`: input RGB in the **OpenCV frame** of the recorded depth (`raw[::-1]` of the
robosuite observation), output one bool mask per camera in that frame, `found=False` + empty mask on failure,
highest-score candidate when several pass the thresholds (GT never used for selection). Thresholds are the official
demo defaults; none was tuned on held-out data.

## Grounded-SAM (v1 recipe) — `providers/grounded_sam.py`

| item | official (`Grounded-Segment-Anything` @ 126abe6, `grounded_sam_demo.py`) | here |
|---|---|---|
| detector | GroundingDINO Swin-T OGC (`groundingdino_swint_ogc.pth`), repo model code + CUDA op | `IDEA-Research/grounding-dino-tiny` (smoke) / `-base` (default), HF `GroundingDinoForObjectDetection` |
| box threshold | `--box_threshold` default **0.3** (l.165) | 0.3 |
| text threshold | `--text_threshold` default **0.25** (l.166) | 0.25 |
| caption | lower-case, trailing "." | same (`grounding_dino.caption`) |
| filtering | `logits.max(dim=1) > box_threshold`; phrase tokens `> text_threshold` | HF `post_process_grounded_object_detection(threshold, text_threshold)` = the same two rules |
| SAM | `SamPredictor`, vit_h, `predict_torch(boxes, multimask_output=False)` | HF `SamModel` `facebook/sam-vit-huge` (default) / `sam-vit-base` (smoke), box prompt, `multimask_output=False` |
| boxes kept | all boxes drawn | top-1 detector score (target selection rule) |

Equivalence argument: the HF checkpoints are ports of the same official weights (model cards of
`IDEA-Research/grounding-dino-*` and `facebook/sam-vit-*`), the thresholds and caption rule are identical, and the
filtering rule is the same. The task's example "box 0.35 / text 0.25" is not the code default; the code default
0.3 / 0.25 is used. `grounding-dino-tiny` is the Swin-T model and `-base` the Swin-B model according to the HF model cards (the card claim was not re-verified against the original `.pth` here).
Not measured here: numerical parity between the HF port and the original repo output (would need the CUDA op build).

## Grounded-SAM-2 — `providers/grounded_sam2.py`

| item | official (`Grounded-SAM-2` @ b7a9c29) | here |
|---|---|---|
| detector | `grounded_sam2_hf_model_demo.py`: HF `grounding-dino-tiny` | HF `grounding-dino-tiny` (smoke) / `-base` (default) |
| thresholds | `threshold=0.4`, `text_threshold=0.3` (l.91-97) | 0.4 / 0.3 |
| image seg | `SAM2ImagePredictor.predict(box=..., multimask_output=False)`, `sam2.1_hiera_large.pt` | same class from the pinned clone; `sam2.1-hiera-tiny` (smoke) / `-large` (default), HF hub copies of the official `.pt` |
| tracking | `grounded_sam2_tracking_camera_with_continuous_id.py`: `init_state()` (streaming), `add_new_frame`, `reset_state` + `add_new_mask` on a detection frame, `infer_single_frame` | same calls; detection only until the first found frame (no periodic re-detection, `detection_interval` not used); one object |
| CUDA ext | built by default | **not built** (`SAM2_BUILD_CUDA=0`): `fill_hole_area=8` post-processing is skipped with a warning (`provenance()["sam2_cuda_ext"]`) |

Causality (`track`): `CausalTrack` only appends frames that `predict` has been given and only infers the newest one.
Checked by `tests/test_grounded_sam2_track.py` (fake predictor, results for frames < t unchanged when later frames
change) and by `scripts/smoke_providers.py --provider grounded_sam2:track` with the real tiny weights
(`causality_check.past_masks_identical`).

## YOLOE — `providers/yoloe.py`

`predict_text_prompt.py` @ 40cd606: `set_classes(names, get_text_pe(names))`, `predict(PIL RGB, verbose=False)` with
ultralytics defaults conf 0.25 / NMS IoU 0.7. Runtime is the pinned `ultralytics==8.4.184` wheel (the repo's vendored
ultralytics fork is not installed); weights `yoloe-11s-seg` (smoke) / `yoloe-11l-seg` (default) from the ultralytics
v8.4.0 release; text encoder `mobileclip_blt.ts` (same release). Deviation: `retina_masks=True` (full-resolution masks).
License **AGPL-3.0** (repo and ultralytics) — using it in a released system has copyleft consequences.

## GT oracle — `providers/gt_oracle.py`

Mask = `segmentation_geom/<cam>` in the geom ids of the BDDL target (`benchmark.adapters.libero_panda.task_target`),
or the scene export's `target_json` when `--scene-gt` is given. Caveat: segmentation is rendered at `env.step`
return, one physics substep after the RGB/depth instant (geom npz `conventions_json["segmentation"]`).

## Shared SAM wrapper (for B)

`from Baseline.paper_baselines.providers.sam_backend import SamV1Backend, Sam2ImageBackend`

```python
sam = SamV1Backend("sam-vit-base", device="cpu")      # or "sam-vit-huge" (default)
mask, score = sam.predict_points(rgb, points_xy, labels, multimask=True)   # best predicted-IoU mask
mask, score = sam.predict_box(rgb, box_xyxy)                               # multimask_output=False
sam.provenance()
```
