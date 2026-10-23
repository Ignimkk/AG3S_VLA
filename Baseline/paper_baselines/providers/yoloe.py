"""YOLOE text-prompt instance segmentation (pinned ultralytics wheel), prompt = ``query.prompt`` as the single class.

Recipe source: ``external/yoloe/predict_text_prompt.py`` @ 40cd606 -- ``model.set_classes(names,
model.get_text_pe(names))`` then ``model.predict(PIL RGB image, verbose=False)`` with ultralytics' default
thresholds (``conf=0.25``, ``iou=0.7``; not overridden by the demo). Deviation: ``retina_masks=True`` so masks
come back at the input resolution (the demo's mask resolution only affects drawing). The highest-confidence
instance is the target (``interface.SELECTION_RULE``). License: AGPL-3.0 (YOLOE repo and ultralytics).

ultralytics side effects are kept in the checkout: ``YOLO_CONFIG_DIR`` -> ``.cache/paper_baselines/ultralytics_config``,
``weights_dir`` -> ``.cache/paper_baselines/weights`` (MobileCLIP text encoder ``mobileclip_blt.ts`` is fetched
there by :mod:`weights` before the model asks for it), ``YOLO_AUTOINSTALL=False``.
"""
from __future__ import annotations

import os

import numpy as np

from Baseline.paper_baselines import weights as W
from Baseline.paper_baselines.interface import SELECTION_RULE, TargetMaskResult, TargetQuery, Timer, check_rgb, select_top1

SOURCE = {"repo": "https://github.com/THU-MIG/yoloe", "sha": "40cd606cabdbe2b566d6f14a6b162c89206e9a1b",
          "recipe_from": "predict_text_prompt.py:54-55 (set_classes + get_text_pe, predict defaults)",
          "runtime": "ultralytics==8.4.184 (pip)", "license": "AGPL-3.0"}
CONF = 0.25  # ultralytics default (cfg/default.yaml), used by the demo
IOU = 0.7  # ultralytics default NMS IoU


def _prepare_ultralytics_env():
    os.environ.setdefault("YOLO_CONFIG_DIR", str(W.CACHE / "ultralytics_config"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("YOLO_OFFLINE", "True")
    os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)


class YOLOEProvider:
    name = "yoloe"

    def __init__(self, size: str = "default", device: str | None = None, conf: float = CONF, iou: float = IOU,
                 model_weight: str | None = None):
        _prepare_ultralytics_env()
        from Baseline.paper_baselines.providers.sam_backend import default_device

        self.size, self.conf, self.iou = size, float(conf), float(iou)
        self.model_weight = model_weight or W.preset("yoloe", size)["model"]
        self.device = device or default_device()
        path = W.fetch(self.model_weight)
        W.fetch("mobileclip_blt")
        from ultralytics import YOLOE
        from ultralytics.utils import SETTINGS

        SETTINGS.update({"weights_dir": str(W.WEIGHTS_DIR)})
        self.model = YOLOE(str(path))
        self.model.to(self.device)
        self._classes = None

    def provenance(self) -> dict:
        import ultralytics

        return {"provider": self.name, "source": SOURCE, "size": self.size, "conf": self.conf, "iou": self.iou,
                "retina_masks": True, "selection_rule": SELECTION_RULE, "device": self.device,
                "ultralytics": ultralytics.__version__,
                "weights": W.provenance(self.model_weight), "text_encoder": W.provenance("mobileclip_blt")}

    def reset(self) -> None:
        pass

    def _set_prompt(self, text: str):
        names = [text]
        if self._classes != names:
            self.model.set_classes(names, self.model.get_text_pe(names))
            self._classes = names

    def predict(self, rgb: dict[str, np.ndarray], query: TargetQuery, *, aux: dict | None = None
                ) -> dict[str, TargetMaskResult]:
        from PIL import Image

        self._set_prompt(query.prompt)
        out = {}
        for cam, img in rgb.items():
            img = check_rgb(img)
            H, W_ = img.shape[:2]
            with Timer() as t:
                res = self.model.predict(Image.fromarray(img), conf=self.conf, iou=self.iou, retina_masks=True,
                                         device=self.device, verbose=False)[0]
                n = 0 if res.boxes is None else len(res.boxes)
                boxes = res.boxes.xyxy.float().cpu().numpy().reshape(-1, 4) if n else np.zeros((0, 4))
                scores = res.boxes.conf.float().cpu().numpy().reshape(-1) if n else np.zeros(0)
                k = select_top1(scores)
                mask = None
                if k is not None and res.masks is not None:
                    mask = res.masks.data[k].float().cpu().numpy() > 0.5
            phrases = [query.prompt] * n
            extras = {"prompt": query.prompt, "candidate_scores": scores.tolist()}
            if k is None:
                out[cam] = TargetMaskResult.not_found(cam, img.shape, t.ms, boxes=boxes, phrases=phrases,
                                                      reason="no_instance_above_conf", extras=extras)
            elif mask is None or mask.shape != (H, W_) or not mask.any():
                out[cam] = TargetMaskResult.not_found(
                    cam, img.shape, t.ms, boxes=boxes, phrases=phrases, reason="empty_or_misshaped_mask",
                    extras=extras | {"selected": k, "mask_shape": None if mask is None else list(mask.shape)})
            else:
                out[cam] = TargetMaskResult(cam, mask, True, float(scores[k]), boxes, phrases, t.ms, extras | {"selected": k})
        return out


__all__ = ["CONF", "IOU", "YOLOEProvider"]
