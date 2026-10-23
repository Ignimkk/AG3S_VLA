"""cloud_labeling (BASELINE-EVAL F): point-level labels of the V11AB cloud for GT, ATS and mask arms."""
import json

import numpy as np
import pytest

from Baseline.paper_baselines.eval import cloud_labeling as C

TGT, NB, SUP = (C.POINT_CLASSES.index(k) for k in ("target", "other_object", "support"))


def _camera():
    """Camera at the origin looking down +z (OpenCV axes = base axes), f = 100, 64 x 64."""
    K = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1]])
    return K, np.eye(4)


def test_role_class_and_arm_dir():
    assert C.role_class("akita_black_bowl_1", "target") == "target"
    assert C.role_class("plate_1", "destination") == "destination"
    assert C.role_class("cookies_1", "object") == "other_object"
    assert C.role_class("main_table", "fixture") == "support"
    assert C.role_class("flat_stove_1", "fixture") == "fixture"
    assert C.role_class("mount", "mount") == "mount"
    assert C.arm_from_dir("grounded_sam2-per_frame__phrase") == "grounded_sam2:per_frame|phrase"
    assert C.arm_from_dir("localization_heads-sam_point__no_text") == "localization_heads:sam_point|no_text"
    assert C.arm_from_dir("gt_oracle__instruction") == "gt_oracle|instruction"
    from Baseline.paper_baselines.eval.native_metrics import arm_dir_name

    for a in ("tokencut:attention_overlap|no_text", "yoloe|phrase", "grounded_sam|instruction"):
        assert C.arm_from_dir(arm_dir_name(a)) == a


def test_classes_from_labels():
    codes = C.classes_from_labels(["bowl_1", "robot", "none", "table"], {"bowl_1": "target", "table": "support"})
    assert [C.POINT_CLASSES[c] for c in codes] == ["target", "robot", "none", "support"]


def test_mask_positive_projection_and_visibility():
    K, T = _camera()
    # three points at z = 1 m: pixel (32, 32), (42, 32), (32, 42)
    pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0], [0.0, 0.0, -1.0]])
    m = np.zeros((64, 64), bool)
    m[32, 32] = m[32, 42] = True  # rows = v, cols = u
    bits = np.array([1, 2, 1, 1], np.uint8)  # point 1 seen only by left_wrist (bit 2)
    pos = C.mask_positive(pts, bits, {"head": m}, {"head": K}, {"head": T}, visibility="observed")
    np.testing.assert_array_equal(pos, [True, False, False, False])  # behind-camera point never positive
    pos = C.mask_positive(pts, bits, {"head": m}, {"head": K}, {"head": T}, visibility="any")
    np.testing.assert_array_equal(pos, [True, True, False, False])
    depth = np.full((64, 64), 1.0)
    depth[32, 42] = 1.5  # occluded -> not visible in the depth test
    pos = C.mask_positive(pts, None, {"head": m}, {"head": K}, {"head": T}, visibility="depth", depth_m={"head": depth})
    np.testing.assert_array_equal(pos, [True, False, False, False])
    with pytest.raises(ValueError):
        C.mask_positive(pts, None, {"head": m}, {"head": K}, {"head": T}, visibility="observed")


def test_point_metrics():
    cls = np.array([TGT, TGT, TGT, NB, SUP, NB], np.int8)
    pred = np.array([1, 1, 0, 1, 0, 0], bool)
    m = C.point_metrics(pred, cls)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert m["iou"] == pytest.approx(2 / 4) and m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["neighbour_frac"] == pytest.approx(1 / 3) and m["support_frac"] == 0.0
    e = C.point_metrics(np.zeros(6, bool), cls)
    assert e["precision"] is None and e["recall"] == 0.0 and e["iou"] == 0.0 and not e["found"]
    nv = C.point_metrics(pred, np.full(6, SUP, np.int8))
    assert nv["iou"] is None and nv["missing_reason"] == "target_not_visible"


def test_robot_surface_distance():
    d = C.robot_surface_distance(np.array([[0.0, 0, 0], [1.0, 0, 0]]), np.array([[0.0, 0, 0], [0.9, 0, 0]]),
                                 np.array([0.05, 0.02]))
    np.testing.assert_allclose(d, [-0.05, 0.08])


def test_summarize_paired_vs_ats():
    rows = []
    for e, (ai, si) in enumerate([(0.2, 0.6), (0.3, 0.5), (0.1, 0.7)]):
        for arm, v in ((C.ATS_ARM, ai), ("gsam|phrase", si)):
            rows.append({"arm": arm, "episode_id": f"e{e}", "iou": v, "precision": v, "recall": v, "neighbour_frac": 0.1,
                         "support_frac": 0.0, "robot_frac": 0.0, "none_frac": 0.0, "tp": 1, "fp": 1, "fn": 1,
                         "n_pred": 2, "n_gt": 2, "pred_neighbour": 0, "found": True})
    s = C.summarize(rows)
    p = s["paired_vs_ats"]["gsam|phrase"]["iou"]
    assert p["diff"] == pytest.approx(0.4) and p["better"] == 3
    assert s["arms"][C.ATS_ARM]["iou"]["episode_mean"] == pytest.approx(0.2)


# --- recorded data (skipped when the runs are not reachable) ----------------------------------------------------------
def test_recorded_calls_gt_oracle_agrees(tmp_path, monkeypatch):
    from Baseline.paper_baselines.eval import native_metrics as N

    try:
        ats = N.resolve_run("20261012T225820_46a08d6ff644_sl4-perception-dev")
        gr = N.resolve_run("20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack")
    except Exception:
        pytest.skip("SL3 / SL4 runs not reachable")
    monkeypatch.setattr(N, "PROJECT_ROOT", tmp_path)
    out = C.run(C.build_parser().parse_args([
        "--ats-run", str(ats), "--grounding-run", str(gr), "--episodes", "libero_spatial_t0_e0", "--limit-calls", "2",
        "--arms", "gt_oracle|phrase", "--out-dir", str(tmp_path / "outputs/cloud")]))
    s = json.loads((out / "summary.json").read_text())
    g = s["arms"]["gt_oracle|phrase"]["pooled"]
    # the GT mask projected onto the cloud recovers the GT-labelled target points (boundary pixels aside)
    assert g["recall"] > 0.9 and g["precision"] > 0.8
    assert C.ATS_ARM in s["arms"] and "gt_oracle|phrase" in s["paired_vs_ats"]
