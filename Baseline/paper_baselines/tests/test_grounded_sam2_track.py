"""grounded_sam2 track mode: causal (result at t independent of frames after t), reset, not-found path.

Uses a fake detector / SAM / streaming video predictor with the same calls as the pinned SAM 2 fork
(init_state, add_new_frame, reset_state, add_new_mask, infer_single_frame) so no weights are needed. The real
models are checked by scripts/smoke_providers.py --provider grounded_sam2:track (causality_check).
"""
import numpy as np

from Baseline.paper_baselines.interface import TargetQuery, validate_results
from Baseline.paper_baselines.providers.grounded_sam2 import GroundedSAM2Provider


class FakeDetector:
    device = "cpu"

    def provenance(self):
        return {"fake": True}

    def detect(self, img, text, bt, tt):
        if img.mean() < 1:  # black frame -> nothing
            return np.zeros((0, 4)), np.zeros(0), []
        return np.array([[1, 1, 4, 4], [0, 0, 2, 2.0]]), np.array([0.5, 0.9]), ["a", "b"]


class FakeSam:
    def provenance(self):
        return {"fake": True}

    def predict_box(self, img, box):
        m = np.zeros(img.shape[:2], bool)
        x0, y0, x1, y1 = [int(v) for v in box]
        m[y0:y1, x0:x1] = True
        return m, 0.8


class FakeVP:
    """Mask at frame i = prompt mask shifted by the running sum of frame brightness (depends on past + current)."""

    def init_state(self):
        return {"images": None, "video_height": None, "video_width": None, "frames": []}

    def add_new_frame(self, state, img):
        state["frames"].append(float(img.mean()))
        return len(state["frames"]) - 1

    def reset_state(self, state):
        state.pop("prompt", None)

    def add_new_mask(self, state, idx, obj_id, mask):
        state["prompt"] = (idx, np.asarray(mask, bool))
        return idx, [obj_id], None

    def infer_single_frame(self, inference_state, frame_idx):
        assert frame_idx == len(inference_state["frames"]) - 1, "must only infer the newest frame"
        idx0, m = inference_state["prompt"]
        shift = int(sum(inference_state["frames"][idx0:frame_idx + 1])) % 3
        out = np.roll(m, shift, axis=1).astype(np.float32)[None, None] * 2 - 1
        return frame_idx, [1], out


def _prov():
    return GroundedSAM2Provider(mode="track", _video_predictor=FakeVP(), _detector=FakeDetector(), _sam=FakeSam())


def _frame(v):
    return {"head": np.full((6, 6, 3), v, np.uint8)}


def _run(values):
    p = _prov()
    q = TargetQuery("x", "bowl")
    out = []
    for v in values:
        rgb = _frame(v)
        r = p.predict(rgb, q)
        validate_results(r, rgb)
        out.append(r["head"])
    return out


def test_track_is_causal():
    a = _run([0, 10, 11, 12, 13])
    b = _run([0, 10, 11, 200, 7])
    for i in range(3):
        assert a[i].found == b[i].found
        np.testing.assert_array_equal(a[i].mask, b[i].mask)
    assert not a[0].found and a[0].extras["not_found_reason"] == "no_box_above_threshold"
    assert a[1].extras["source"] == "detect" and a[1].extras["selected"] == 1  # top-1 score, not first box
    assert a[2].extras["source"] == "track"


def test_reset_drops_state():
    p = _prov()
    q = TargetQuery("x", "bowl")
    p.predict(_frame(10), q)
    assert "head" in p.tracks
    p.reset()
    assert p.tracks == {}
    r = p.predict(_frame(10), q)["head"]
    assert r.extras["source"] == "detect"


def test_per_frame_mode_has_no_state():
    p = GroundedSAM2Provider(mode="per_frame", _detector=FakeDetector(), _sam=FakeSam())
    q = TargetQuery("x", "bowl")
    r1 = p.predict(_frame(10), q)["head"]
    r2 = p.predict(_frame(10), q)["head"]
    assert r1.extras["source"] == r2.extras["source"] == "detect" and p.tracks == {}
    np.testing.assert_array_equal(r1.mask, r2.mask)
