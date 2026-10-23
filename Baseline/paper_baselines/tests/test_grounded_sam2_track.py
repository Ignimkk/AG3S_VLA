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


# --- bounded track memory (BASELINE-EVAL) ---------------------------------------------------------------------
class StreamingVP:
    """Stores like the pinned ``SAM2VideoPredictor`` streaming path and reads like ``sam2_base.py:534-620``.

    ``add_new_frame``: concatenates the frame onto ``state["images"]`` (``num_frames`` = its length) and caches a
    large per-frame feature in ``cached_features``; ``infer_single_frame`` reads its own feature, the prompt (cond)
    output, the mask memories of frames ``f-1`` and every ``stride``-th frame back to ``f-1-(num_maskmem-2)*stride``,
    and the object pointers of frames ``f-1 .. f-(max_obj_ptrs-1)``, and stores a large ``maskmem_features`` plus a
    small ``obj_ptr`` per frame (and a per-object slice of it). A read of a dropped entry inside the window raises.
    """

    num_maskmem, memory_temporal_stride_for_eval, max_obj_ptrs_in_encoder, use_obj_ptrs_in_encoder = 7, 1, 16, True
    FEAT = 4096  # floats per cached feature (stands in for ~125 MB)
    MEM = 1024  # floats per maskmem_features

    def init_state(self):
        return {"images": None, "num_frames": 0, "cached_features": {}, "constants": {},
                "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                "output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
                "frames_already_tracked": {}, "video_height": None, "video_width": None}

    def add_new_frame(self, state, img):
        x = np.asarray(img, np.float32)[None]
        state["images"] = x if state["images"] is None else np.concatenate([state["images"], x])
        state["num_frames"] = len(state["images"])
        idx = state["num_frames"] - 1
        state["cached_features"][idx] = np.full(self.FEAT, float(np.asarray(img).mean()), np.float32)
        return idx

    def reset_state(self, state):
        state["output_dict"]["cond_frame_outputs"].clear()
        state["output_dict"]["non_cond_frame_outputs"].clear()

    def _out(self, value):
        return {"maskmem_features": np.full(self.MEM, value, np.float32), "maskmem_pos_enc": [np.zeros(8, np.float32)],
                "pred_masks": np.full(16, value, np.float32), "obj_ptr": np.full(4, value, np.float32),
                "object_score_logits": np.array([value], np.float32)}

    def add_new_mask(self, state, idx, obj_id, mask):
        v = float(state["cached_features"][idx][0])
        out = self._out(v)
        state["output_dict"]["cond_frame_outputs"][idx] = out
        state["output_dict_per_obj"][0]["cond_frame_outputs"][idx] = {k: out[k] for k in out}
        state["prompt"] = (idx, np.asarray(mask, bool))
        return idx, [obj_id], None

    def infer_single_frame(self, inference_state, frame_idx):
        st = inference_state
        assert frame_idx == st["num_frames"] - 1, "must only infer the newest frame"
        idx0, m = st["prompt"]
        od = st["output_dict"]
        if frame_idx == idx0:
            v = float(od["cond_frame_outputs"][idx0]["maskmem_features"][0])
        else:
            feat = float(st["cached_features"][frame_idx][0])  # KeyError if the new frame's feature is missing
            acc = feat + sum(float(o["maskmem_features"][0]) for o in od["cond_frame_outputs"].values())
            s = self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos
                prev = frame_idx - 1 if t_rel == 1 else ((frame_idx - 2) // s) * s - (t_rel - 2) * s
                if idx0 < prev < frame_idx:
                    acc += 0.5 * float(od["non_cond_frame_outputs"][prev]["maskmem_features"][0])
            n_ptr = min(st["num_frames"], self.max_obj_ptrs_in_encoder)
            for d in range(1, n_ptr):
                t = frame_idx - d
                if t < 0:
                    break
                o = od["non_cond_frame_outputs"].get(t, od["cond_frame_outputs"].get(t))
                if t > idx0:
                    assert o is not None, f"obj_ptr of frame {t} needed by frame {frame_idx} was dropped"
                if o is not None:
                    acc += 0.25 * d * float(o["obj_ptr"][0])
            v = acc % 7.0
            out = self._out(v)
            od["non_cond_frame_outputs"][frame_idx] = out
            st["output_dict_per_obj"][0]["non_cond_frame_outputs"][frame_idx] = {k: out[k] for k in out}
        st["frames_already_tracked"][frame_idx] = {"reverse": False}
        shift = int(v * 1000) % 5
        return frame_idx, [1], np.roll(m, shift, axis=1).astype(np.float32)[None, None] * 2 - 1


def _long_run(bounded, values):
    from Baseline.paper_baselines.providers.grounded_sam2 import state_footprint

    p = GroundedSAM2Provider(mode="track", bounded_memory=bounded, _video_predictor=StreamingVP(),
                             _detector=FakeDetector(), _sam=FakeSam())
    q = TargetQuery("x", "bowl")
    masks, foot = [], []
    for v in values:
        r = p.predict(_frame(v), q)["head"]
        masks.append(r.mask)
        tr = p.tracks.get("head")
        foot.append(state_footprint(tr.state) if tr is not None and tr.state is not None else None)
    return masks, foot, p


def test_bounded_memory_same_masks_and_flat_footprint():
    values = [0, 0] + [int(v) for v in np.random.default_rng(3).integers(5, 250, 118)]  # prompt at frame 2
    mb, fb, pb = _long_run(True, values)
    mu, fu, _ = _long_run(False, values)
    assert len(mb) == 120
    for a, b in zip(mb, mu):  # bounded == unbounded (the pinned code's state), frame by frame
        np.testing.assert_array_equal(a, b)
    assert len({m.tobytes() for m in mb[3:]}) > 1  # the masks do depend on the memory
    tr = pb.tracks["head"]
    w = tr.window
    assert (w["mask_memory_frames"], w["obj_ptr_frames"], w["frames"]) == (6, 15, 15)
    # bounded: footprint stops growing once the read window is full; unbounded grows every frame
    sizes = [f["bytes"] for f in fb[2:]]
    counts = [f["n_tensors"] for f in fb[2:]]
    assert sizes[w["frames"] + 1:] == [sizes[-1]] * len(sizes[w["frames"] + 1:])
    assert counts[w["frames"] + 1:] == [counts[-1]] * len(counts[w["frames"] + 1:])
    assert max(sizes) == sizes[-1] < fu[-1]["bytes"] / 20
    u = [f["bytes"] for f in fu[2:]]
    assert all(b > a for a, b in zip(u, u[1:]))
    st = tr.state
    assert len(st["images"]) == 1 and st["num_frames"] == tr.n_frames == 118 and not st["cached_features"]
    nc = st["output_dict"]["non_cond_frame_outputs"]
    assert sorted(nc) == list(range(117 - 15 + 1, 118))  # the last 15 frames (object-pointer window)
    full = [f for f in nc if "maskmem_features" in nc[f]]
    assert sorted(full) == list(range(117 - 6 + 1, 118))  # mask memories only for the last 6 frames
    assert 0 in st["output_dict"]["cond_frame_outputs"]  # prompt frame kept


def test_bounded_memory_is_causal():
    def run(values):
        return _long_run(True, values)[0]

    a = run([0, 10] + list(range(20, 60)))
    b = run([0, 10] + list(range(20, 50)) + [200] * 10)
    for i in range(32):
        np.testing.assert_array_equal(a[i], b[i])
