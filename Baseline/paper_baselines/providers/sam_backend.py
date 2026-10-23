"""SAM wrappers shared by the providers (A) and by ``localization_heads`` ``sam_point`` mode (B).

Both backends have one API, all in the pixel frame of the RGB they are given:

    backend.set_image(rgb_uint8_HxWx3)                         # image embedding, cached until the next set_image
    mask, score = backend.predict_box(rgb, box_xyxy)           # bool (H, W), predicted IoU score
    mask, score = backend.predict_points(rgb, points_xy, labels, multimask=True)

* :class:`SamV1Backend` -- SAM v1 through HF ``transformers.SamModel`` (official ViT-B / ViT-H weights). Box prompt
  uses ``multimask_output=False`` like ``Grounded-Segment-Anything/grounded_sam_demo.py`` (``predict_torch(...,
  multimask_output=False)``); point prompts default to ``multimask_output=True`` and keep the highest predicted-IoU
  mask (SAM's recommendation for ambiguous single clicks).
* :class:`Sam2ImageBackend` -- SAM 2.1 ``SAM2ImagePredictor`` from the pinned Grounded-SAM-2 clone (editable ``sam2``),
  same prompt semantics as ``grounded_sam2_hf_model_demo.py`` (``multimask_output=False`` for boxes).

``rgb`` arguments of ``predict_*`` re-run ``set_image`` only when the array differs from the cached one.
"""
from __future__ import annotations

import pathlib

import numpy as np

from Baseline.paper_baselines import weights as W
from Baseline.paper_baselines.interface import check_rgb

GSAM2_ROOT = pathlib.Path(__file__).resolve().parents[1] / "external" / "Grounded-SAM-2"


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class _CachedImage:
    def __init__(self):
        self._key = None

    @staticmethod
    def key(rgb: np.ndarray):
        a = np.ascontiguousarray(rgb)
        return (a.shape, hash(a.tobytes()))

    def is_current(self, rgb) -> bool:
        return self._key is not None and self._key == self.key(rgb)


class SamV1Backend(_CachedImage):
    name = "sam_v1_hf"

    def __init__(self, weight: str = "sam-vit-huge", device: str | None = None):
        super().__init__()
        import torch
        from transformers import SamModel, SamProcessor

        self.weight, self.device = weight, device or default_device()
        path = W.fetch(weight)
        self.processor = SamProcessor.from_pretrained(path)
        self.model = SamModel.from_pretrained(path).to(self.device).eval()
        self._torch = torch
        self._emb = None
        self._sizes = None

    def provenance(self) -> dict:
        return {"backend": self.name, "weights": W.provenance(self.weight), "device": self.device,
                "box_multimask_output": False, "point_multimask_output_default": True,
                "mask_threshold": 0.0, "implementation": "transformers.SamModel"}

    def set_image(self, rgb: np.ndarray) -> None:
        rgb = check_rgb(rgb)
        if self.is_current(rgb):
            return
        torch = self._torch
        inputs = self.processor(images=rgb, return_tensors="pt")
        with torch.inference_mode():
            self._emb = self.model.get_image_embeddings(inputs["pixel_values"].to(self.device))
        self._sizes = (inputs["original_sizes"], inputs["reshaped_input_sizes"])
        self._key = self.key(rgb)

    def _decode(self, rgb, multimask: bool, **prompt):
        self.set_image(rgb)
        torch = self._torch
        inputs = self.processor(images=rgb, return_tensors="pt", **prompt)
        inputs.pop("pixel_values", None)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}
        inputs.pop("original_sizes", None)
        inputs.pop("reshaped_input_sizes", None)
        with torch.inference_mode():
            out = self.model(image_embeddings=self._emb, multimask_output=multimask, **inputs)
        masks = self.processor.image_processor.post_process_masks(out.pred_masks.cpu(), self._sizes[0], self._sizes[1])[0]
        scores = out.iou_scores.float().cpu().numpy()[0, 0]  # (n_masks,)
        masks = masks.numpy()[0].astype(bool)  # (n_masks, H, W)
        best = int(np.argmax(scores))
        return masks[best], float(scores[best])

    def predict_box(self, rgb: np.ndarray, box_xyxy) -> tuple[np.ndarray, float]:
        box = [float(x) for x in np.asarray(box_xyxy, np.float64).reshape(4)]
        return self._decode(rgb, False, input_boxes=[[box]])

    def predict_points(self, rgb: np.ndarray, points_xy, labels, multimask: bool = True) -> tuple[np.ndarray, float]:
        pts = np.asarray(points_xy, np.float64).reshape(-1, 2).tolist()
        lab = [int(x) for x in np.asarray(labels).reshape(-1)]
        if len(pts) != len(lab) or not pts:
            raise ValueError("points_xy and labels must be non-empty and the same length")
        return self._decode(rgb, multimask, input_points=[[pts]], input_labels=[[lab]])


class Sam2ImageBackend(_CachedImage):
    name = "sam2_image"

    def __init__(self, weight: str = "sam2.1-hiera-large", device: str | None = None):
        super().__init__()
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.weight, self.device = weight, device or default_device()
        self.ckpt = W.weight_file(weight)
        self.config = W.WEIGHTS[weight]["config"]
        self.model = build_sam2(self.config, str(self.ckpt), device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self._torch = torch

    def provenance(self) -> dict:
        return {"backend": self.name, "weights": W.provenance(self.weight), "config": self.config, "device": self.device,
                "box_multimask_output": False, "point_multimask_output_default": True,
                "implementation": "sam2.sam2_image_predictor.SAM2ImagePredictor (Grounded-SAM-2 pinned clone)"}

    def set_image(self, rgb: np.ndarray) -> None:
        rgb = check_rgb(rgb)
        if self.is_current(rgb):
            return
        with self._torch.inference_mode():
            self.predictor.set_image(rgb)
        self._key = self.key(rgb)

    def _predict(self, rgb, multimask, **prompt):
        self.set_image(rgb)
        with self._torch.inference_mode():
            masks, scores, _ = self.predictor.predict(multimask_output=multimask, **prompt)
        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)
        if masks.ndim == 2:
            masks = masks[None]
        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])

    def predict_box(self, rgb, box_xyxy):
        return self._predict(rgb, False, point_coords=None, point_labels=None,
                             box=np.asarray(box_xyxy, np.float32).reshape(1, 4))

    def predict_points(self, rgb, points_xy, labels, multimask: bool = True):
        pts = np.asarray(points_xy, np.float32).reshape(-1, 2)
        lab = np.asarray(labels, np.int32).reshape(-1)
        if len(pts) != len(lab) or not len(pts):
            raise ValueError("points_xy and labels must be non-empty and the same length")
        return self._predict(rgb, multimask, point_coords=pts, point_labels=lab, box=None)


__all__ = ["Sam2ImageBackend", "SamV1Backend", "default_device"]
