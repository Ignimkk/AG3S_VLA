"""Grounded-SAM (v1 recipe): Grounding DINO box -> SAM v1 mask, via HF transformers (no GroundingDINO CUDA op build).

Recipe source: ``external/Grounded-Segment-Anything/grounded_sam_demo.py`` @ 126abe6 --
``--box_threshold`` default 0.3 (l.165), ``--text_threshold`` default 0.25 (l.166), SAM box prompt with
``multimask_output=False``. Differences (docs/providers.md): HF ports of the same official weights instead of the
original repo's model code; the demo keeps every box and colours them, here the highest-score box is the target
(``interface.SELECTION_RULE``).
"""
from __future__ import annotations

import numpy as np

from Baseline.paper_baselines import weights as W
from Baseline.paper_baselines.interface import (SELECTION_RULE, TargetMaskResult, TargetQuery, Timer, check_rgb,
                                                select_top1)

SOURCE = {"repo": "https://github.com/IDEA-Research/Grounded-Segment-Anything",
          "sha": "126abe633ffe333e16e4a0a4e946bc1003caf757",
          "thresholds_from": "grounded_sam_demo.py:165-166 (argparse defaults)"}
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25


class GroundedSAMProvider:
    name = "grounded_sam"

    def __init__(self, size: str = "default", device: str | None = None, box_threshold: float = BOX_THRESHOLD,
                 text_threshold: float = TEXT_THRESHOLD, detector_weight: str | None = None, sam_weight: str | None = None):
        from Baseline.paper_baselines.providers.grounding_dino import GroundingDinoDetector
        from Baseline.paper_baselines.providers.sam_backend import SamV1Backend

        p = W.preset("grounded_sam", size)
        self.size = size
        self.box_threshold, self.text_threshold = float(box_threshold), float(text_threshold)
        self.detector = GroundingDinoDetector(detector_weight or p["detector"], device)
        self.sam = SamV1Backend(sam_weight or p["sam"], self.detector.device)

    def provenance(self) -> dict:
        return {"provider": self.name, "source": SOURCE, "size": self.size,
                "box_threshold": self.box_threshold, "text_threshold": self.text_threshold,
                "selection_rule": SELECTION_RULE, "detector": self.detector.provenance(), "sam": self.sam.provenance()}

    def reset(self) -> None:
        pass

    def predict(self, rgb: dict[str, np.ndarray], query: TargetQuery, *, aux: dict | None = None
                ) -> dict[str, TargetMaskResult]:
        out = {}
        for cam, img in rgb.items():
            img = check_rgb(img)
            with Timer() as t:
                boxes, scores, phrases = self.detector.detect(img, query.prompt, self.box_threshold, self.text_threshold)
                k = select_top1(scores)
                mask = score = None
                if k is not None:
                    mask, sam_score = self.sam.predict_box(img, boxes[k])
            extras = {"prompt": query.prompt, "candidate_scores": scores.tolist()}
            if k is None:
                out[cam] = TargetMaskResult.not_found(cam, img.shape, t.ms, boxes=boxes, phrases=phrases,
                                                      reason="no_box_above_threshold", extras=extras)
            elif not mask.any():
                out[cam] = TargetMaskResult.not_found(cam, img.shape, t.ms, boxes=boxes, phrases=phrases,
                                                      reason="empty_sam_mask", extras=extras | {"selected": k})
            else:
                out[cam] = TargetMaskResult(cam, mask, True, float(scores[k]), boxes, phrases, t.ms,
                                            extras | {"selected": k, "selected_phrase": phrases[k] if k < len(phrases) else None,
                                                      "sam_iou_score": sam_score})
        return out


__all__ = ["BOX_THRESHOLD", "GroundedSAMProvider", "TEXT_THRESHOLD"]
