"""ABL-M6 test stub: a ``TargetMaskProvider`` with a known mask, loaded by the real sidecar (``segmenter_init`` with
``provider="tests.benchmark.abl_m6_stub_segmenter:StubSegmenter"``). numpy only (no torch, no CUDA)."""
from __future__ import annotations

import numpy as np

from Baseline.paper_baselines.interface import TargetMaskResult


class StubSegmenter:
    name = "stub"

    def __init__(self, boxes=None, mode="per_frame", size="default", device=None):
        #: {camera: [row0, row1, col0, col1]} -> found with that box as the mask; a camera not listed -> not found
        self.boxes = dict(boxes or {"head": [96, 160, 100, 170]})
        self.mode, self.size = mode, size
        self.prompts, self.n_reset = [], 0

    def provenance(self):
        return {"provider": "stub", "boxes": self.boxes, "mode": self.mode}

    def reset(self):
        self.n_reset += 1

    def predict(self, rgb, query, *, aux=None):
        self.prompts.append(query.prompt)
        out = {}
        for cam, img in rgb.items():
            img = np.asarray(img)
            assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, (img.dtype, img.shape)
            b = self.boxes.get(cam)
            if b is None:
                out[cam] = TargetMaskResult.not_found(cam, img.shape, 1.5, reason="no_box_above_threshold",
                                                      extras={"prompt": query.prompt, "source": "detect"})
                continue
            m = np.zeros(img.shape[:2], bool)
            m[b[0]:b[1], b[2]:b[3]] = True
            out[cam] = TargetMaskResult(cam, m, True, 0.77, np.array([[b[2], b[0], b[3], b[1]]], float), [query.prompt],
                                        2.5, {"prompt": query.prompt, "source": "detect", "n_reset": self.n_reset,
                                              "n_calls": len(self.prompts)})
        return out
