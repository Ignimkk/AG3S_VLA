"""Grounding DINO through HF ``transformers`` (``GroundingDinoForObjectDetection``), shared by grounded_sam(2).

The caption rule is the official one (``grounded_sam_demo.py`` / ``grounded_sam2_hf_model_demo.py``: "text queries
need to be lowercased + end with a dot"). ``post_process_grounded_object_detection`` applies the same two
thresholds as the original ``get_grounding_output``: a query is kept if its max token logit > ``box_threshold``, and
its phrase is the tokens whose logit > ``text_threshold``.
"""
from __future__ import annotations

import numpy as np

from Baseline.paper_baselines import weights as W
from Baseline.paper_baselines.interface import check_rgb
from Baseline.paper_baselines.providers.sam_backend import default_device


def caption(text: str) -> str:
    c = " ".join(str(text).lower().strip().split())
    return c if c.endswith(".") else c + "."


class GroundingDinoDetector:
    def __init__(self, weight: str = "grounding-dino-base", device: str | None = None):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.weight, self.device = weight, device or default_device()
        path = W.fetch(weight)
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path).to(self.device).eval()
        self._torch = torch

    def provenance(self) -> dict:
        return {"detector": "grounding_dino_hf", "weights": W.provenance(self.weight), "device": self.device,
                "caption_rule": "lower-case, single spaces, trailing '.'",
                "implementation": "transformers.GroundingDinoForObjectDetection + post_process_grounded_object_detection"}

    def detect(self, rgb: np.ndarray, text: str, box_threshold: float, text_threshold: float):
        """``(boxes (N, 4) xyxy pixels, scores (N,), phrases [N])`` in the pixel frame of ``rgb``."""
        from PIL import Image

        rgb = check_rgb(rgb)
        H, W_ = rgb.shape[:2]
        inputs = self.processor(images=Image.fromarray(rgb), text=caption(text), return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(H, W_)])[0]
        boxes = res["boxes"].float().cpu().numpy().reshape(-1, 4)
        scores = res["scores"].float().cpu().numpy().reshape(-1)
        phrases = [str(p) for p in res.get("text_labels", res.get("labels", []))]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W_)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        return boxes, scores, phrases


__all__ = ["GroundingDinoDetector", "caption"]
