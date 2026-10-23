"""Grounded-SAM-2: Grounding DINO (HF) box -> SAM 2.1 mask; optional causal video tracking.

Recipe source: ``external/Grounded-SAM-2`` @ b7a9c29 --
* ``grounded_sam2_hf_model_demo.py``: HF Grounding DINO, ``threshold=0.4`` / ``text_threshold=0.3`` (l.91-97),
  ``SAM2ImagePredictor.predict(box=..., multimask_output=False)``;
* ``grounded_sam2_tracking_camera_with_continuous_id.py`` (``IncrementalObjectTracker``): streaming video predictor
  ``init_state()`` -> ``add_new_frame`` -> ``reset_state`` + ``add_new_mask`` on a detection frame ->
  ``infer_single_frame`` on every later frame.

Modes:
* ``per_frame`` -- every frame is detected and segmented independently (no state).
* ``track``     -- per camera: detect each frame until the first ``found`` frame, prompt the SAM 2 video predictor
  with that mask, then only track (``infer_single_frame``) on every later frame. **Causal by construction**: the
  state only ever receives frames already given to ``predict`` (``add_new_frame`` appends; nothing is propagated
  backwards). ``tests/test_grounded_sam2_track.py`` checks that results up to frame t do not change when the
  frames after t change. ``reset()`` (episode boundary) drops the state. A lost track (empty mask) is reported as
  ``found=False``; re-detection after loss is off by default (``redetect_on_loss``) to keep the recipe.

The harness must call ``predict`` on consecutive frames in time order for ``track`` (``eval/offline_grounding.py``
feeds every recorded frame up to the last evaluated one).
"""
from __future__ import annotations

import numpy as np

from Baseline.paper_baselines import weights as W
from Baseline.paper_baselines.interface import (SELECTION_RULE, TargetMaskResult, TargetQuery, Timer, check_rgb,
                                                select_top1)

SOURCE = {"repo": "https://github.com/IDEA-Research/Grounded-SAM-2", "sha": "b7a9c29f196edff0eb54dbe14588d7ae5e3dde28",
          "thresholds_from": "grounded_sam2_hf_model_demo.py:91-97 (post_process threshold=0.4, text_threshold=0.3)",
          "tracking_from": "grounded_sam2_tracking_camera_with_continuous_id.py IncrementalObjectTracker.add_image"}
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.3
MODES = ("per_frame", "track")
OBJ_ID = 1


class CausalTrack:
    """One camera's streaming SAM 2 video state (only past + current frames ever enter it)."""

    def __init__(self, video_predictor):
        self.vp = video_predictor
        self.state = None
        self.n_frames = 0
        self.prompt_frame = None

    def start(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.state = self.vp.init_state()
        self.state["video_height"], self.state["video_width"] = image.shape[:2]
        idx = self.vp.add_new_frame(self.state, image)
        self.vp.reset_state(self.state)
        self.vp.add_new_mask(self.state, idx, OBJ_ID, mask)
        self.n_frames, self.prompt_frame = 1, idx
        return self._infer(idx)

    def step(self, image: np.ndarray) -> np.ndarray:
        idx = self.vp.add_new_frame(self.state, image)
        self.n_frames += 1
        return self._infer(idx)

    def _infer(self, idx: int) -> np.ndarray:
        _, obj_ids, masks = self.vp.infer_single_frame(inference_state=self.state, frame_idx=idx)
        k = list(obj_ids).index(OBJ_ID)
        m = masks[k]
        m = m.detach().float().cpu().numpy() if hasattr(m, "detach") else np.asarray(m, np.float32)
        return np.asarray(m).reshape(m.shape[-2:]) > 0.0


class GroundedSAM2Provider:
    name = "grounded_sam2"

    def __init__(self, mode: str = "per_frame", size: str = "default", device: str | None = None,
                 box_threshold: float = BOX_THRESHOLD, text_threshold: float = TEXT_THRESHOLD,
                 detector_weight: str | None = None, sam2_weight: str | None = None, redetect_on_loss: bool = False,
                 _video_predictor=None, _detector=None, _sam=None):
        # _video_predictor / _detector / _sam: test injection only (tests/test_grounded_sam2_track.py)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        from Baseline.paper_baselines.providers.grounding_dino import GroundingDinoDetector
        from Baseline.paper_baselines.providers.sam_backend import Sam2ImageBackend

        p = W.preset("grounded_sam2", size)
        self.mode, self.size, self.redetect_on_loss = mode, size, bool(redetect_on_loss)
        self.box_threshold, self.text_threshold = float(box_threshold), float(text_threshold)
        self.sam2_weight = sam2_weight or p["sam2"]
        self.detector = _detector or GroundingDinoDetector(detector_weight or p["detector"], device)
        self.sam = _sam or Sam2ImageBackend(self.sam2_weight, self.detector.device)
        self.vp = _video_predictor
        if mode == "track" and self.vp is None:
            from sam2.build_sam import build_sam2_video_predictor

            self.vp = build_sam2_video_predictor(W.WEIGHTS[self.sam2_weight]["config"], str(W.weight_file(self.sam2_weight)),
                                                 device=self.detector.device)
        self.tracks: dict[str, CausalTrack] = {}

    @property
    def name_with_mode(self) -> str:
        return f"{self.name}:{self.mode}"

    def provenance(self) -> dict:
        return {"provider": self.name, "mode": self.mode, "source": SOURCE, "size": self.size,
                "box_threshold": self.box_threshold, "text_threshold": self.text_threshold,
                "selection_rule": SELECTION_RULE, "redetect_on_loss": self.redetect_on_loss,
                "track_rule": "detect until first found frame, prompt video predictor with that SAM2 mask (add_new_mask), "
                              "then infer_single_frame on each new frame; causal (no future frames)" if self.mode == "track" else None,
                "sam2_cuda_ext": _sam2_cuda_ext(),
                "detector": self.detector.provenance(), "sam2": self.sam.provenance()}

    def reset(self) -> None:
        self.tracks = {}

    def _detect_segment(self, cam, img, query, t_extra=None):
        boxes, scores, phrases = self.detector.detect(img, query.prompt, self.box_threshold, self.text_threshold)
        k = select_top1(scores)
        mask = sam_score = None
        if k is not None:
            mask, sam_score = self.sam.predict_box(img, boxes[k])
        return boxes, scores, phrases, k, mask, sam_score

    def predict(self, rgb: dict[str, np.ndarray], query: TargetQuery, *, aux: dict | None = None
                ) -> dict[str, TargetMaskResult]:
        out = {}
        for cam, img in rgb.items():
            img = check_rgb(img)
            tr = self.tracks.get(cam)
            with Timer() as t:
                if self.mode == "track" and tr is not None and tr.state is not None:
                    mask = tr.step(img)
                    det = None
                else:
                    det = self._detect_segment(cam, img, query)
                    mask = det[4]
                    if self.mode == "track" and det[3] is not None and mask is not None and mask.any():
                        tr = self.tracks.setdefault(cam, CausalTrack(self.vp))
                        mask = tr.start(img, mask)
            if det is None:  # tracked frame
                extras = {"prompt": query.prompt, "source": "track", "track_frames": tr.n_frames}
                if mask.any():
                    out[cam] = TargetMaskResult(cam, mask, True, None, None, [], t.ms, extras)
                else:
                    out[cam] = TargetMaskResult.not_found(cam, img.shape, t.ms, reason="track_lost", extras=extras)
                    if self.redetect_on_loss:
                        self.tracks.pop(cam, None)
                continue
            boxes, scores, phrases, k, _, sam_score = det
            extras = {"prompt": query.prompt, "source": "detect", "candidate_scores": scores.tolist()}
            if k is None:
                out[cam] = TargetMaskResult.not_found(cam, img.shape, t.ms, boxes=boxes, phrases=phrases,
                                                      reason="no_box_above_threshold", extras=extras)
            elif mask is None or not mask.any():
                out[cam] = TargetMaskResult.not_found(cam, img.shape, t.ms, boxes=boxes, phrases=phrases,
                                                      reason="empty_sam_mask", extras=extras | {"selected": k})
            else:
                out[cam] = TargetMaskResult(cam, mask, True, float(scores[k]), boxes, phrases, t.ms,
                                            extras | {"selected": k, "selected_phrase": phrases[k] if k < len(phrases) else None,
                                                      "sam_iou_score": sam_score,
                                                      "track_started": self.mode == "track"})
        return out


def _sam2_cuda_ext() -> str:
    try:
        from sam2 import _C  # noqa: F401

        return "built"
    except Exception:  # noqa: BLE001
        return ("not built (SAM2_BUILD_CUDA=0): sam2 skips the fill_hole_area=8 / connected-component post-processing "
                "with a warning; mask decoder outputs are otherwise unchanged")


__all__ = ["BOX_THRESHOLD", "CausalTrack", "GroundedSAM2Provider", "MODES", "TEXT_THRESHOLD"]
