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


# --- ATS representation (ATS-CLOUD-DIAG J) ----------------------------------------------------------------------------
def test_flags_near():
    pts = np.array([[0.0, 0, 0], [0.004, 0, 0], [0.006, 0, 0], [1.0, 0, 0]])
    np.testing.assert_array_equal(C.flags_near(pts, np.zeros((1, 3)), 0.005), [True, True, False, False])
    np.testing.assert_array_equal(C.flags_near(pts, np.zeros((1, 3)), 1e-6), [True, False, False, False])
    assert not C.flags_near(pts, None, 0.005).any() and not C.flags_near(pts, np.zeros((0, 3)), 0.005).any()
    assert C.flags_near(np.zeros((0, 3)), np.zeros((1, 3)), 0.005).shape == (0,)


def _synthetic_episode():
    """fresh -> anchor stand-in -> occluded stand-in -> excluded (carried) -> no target, nothing excluded."""
    g0 = np.array([[0.0, 0, 0], [0.01, 0, 0]])
    g2 = np.array([[0.5, 0, 0], [0.52, 0, 0]])
    has = [True, True, True, False, False]
    n_ti = [2, 0, 0, 0, 0]
    state = ["visible", "visible", "occluded", "occluded", ""]
    tps = [g0, g0, g2, np.zeros((0, 3)), np.zeros((0, 3))]
    cen = [g0.mean(0), g0.mean(0), g2.mean(0), g2.mean(0), np.zeros(3)]
    return has, n_ti, state, tps, cen, g0, g2


def test_ats_episode_geometry_categories_and_carry_forward():
    has, n_ti, state, tps, cen, g0, g2 = _synthetic_episode()
    geo = C.ats_episode_geometry(has, n_ti, state, tps, cen)
    assert [g["cat"] for g in geo] == ["fresh", "anchor", "occluded", "excluded_no_target", "no_admissible"]
    assert geo[3]["tp"] is None and geo[3]["carried_from"] == 2 and geo[3]["carry_centroid_err_m"] == pytest.approx(0)
    np.testing.assert_array_equal(geo[3]["excl"], g2)
    assert geo[4]["excl"] is None and geo[4]["tp"] is None
    np.testing.assert_array_equal(geo[1]["excl"], g0)
    # flags per representation on a cloud with one point at each geometry point and one far away
    cloud = np.vstack([g0, g2 + [0.003, 0, 0], [[2.0, 0, 0]]])
    n = [[int(C.flags_near(cloud, g[key], tol).sum()) for g in geo] for key, tol in
         (("tp", 0.005), ("excl", 0.005), ("excl", 1e-6))]
    assert n == [[2, 2, 2, 0, 0], [2, 2, 2, 2, 0], [2, 2, 0, 0, 0]]
    # carried geometry that is not the ESDF's (centroid moved) is reported, a has_target=False call before any
    # target is no_admissible even with a manipulated state
    cen[3] = cen[3] + np.array([0.01, 0, 0])
    assert C.ats_episode_geometry(has, n_ti, state, tps, cen)[3]["carry_centroid_err_m"] == pytest.approx(0.01)
    geo = C.ats_episode_geometry([False, True], [0, 1], ["occluded", "visible"], [np.zeros((0, 3)), g0])
    assert [g["cat"] for g in geo] == ["no_admissible", "fresh"]
    geo = C.ats_episode_geometry([True], [0], ["held"], [g0])
    assert geo[0]["cat"] == "other"


def test_parser_defaults_keep_f():
    a = C.build_parser().parse_args(["--ats-run", "x"])
    assert a.ats_repr == "cloud_target" and a.ats_tol == pytest.approx(0.005)
    with pytest.raises(SystemExit):
        C.build_parser().parse_args(["--ats-run", "x", "--ats-repr", "points"])


# recorded data: V1 (BASELINE-EVAL F) and H's ATS-CLOUD-DIAG recompute
V1_RUN = "20261024T035953_5752c64c7006_baseline-eval-cloud-sl3dev"
SL4_RUN = "20261012T225820_46a08d6ff644_sl4-perception-dev"
GR_RUN = "20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack"
H_ROWS = C.PROJECT_ROOT / "outputs/ats_cloud_diag/recompute/calls_ats_arms.jsonl"
H_CATS = C.PROJECT_ROOT / "outputs/ats_cloud_diag/calls_diag.jsonl"
H_CAT_NAME = {"fresh": "fresh", "anchor_standin": "anchor", "occl_standin": "occluded",
              "none_excl": "excluded_no_target", "none": "no_admissible"}
# t0: fresh / anchor / occluded / excluded; t4: + no_admissible -> every category
EPISODES = ("libero_spatial_t0_e0", "libero_spatial_t4_e0")


def _runs():
    from Baseline.paper_baselines.eval import native_metrics as N

    try:
        return N, N.resolve_run(V1_RUN), N.resolve_run(SL4_RUN), N.resolve_run(GR_RUN)
    except Exception:
        pytest.skip("V1 / SL4 / SL3 runs not reachable")


def _run(tmp_path, monkeypatch, name, extra):
    N, v1, ats, gr = _runs()
    monkeypatch.setattr(N, "PROJECT_ROOT", tmp_path)
    return C.run(C.build_parser().parse_args(["--ats-run", str(ats), "--grounding-run", str(gr), "--episodes", *EPISODES,
                                              "--out-dir", str(tmp_path / "outputs" / name), *extra]))


def test_default_repr_byte_identical_to_v1(tmp_path, monkeypatch):
    """Default options reproduce V1's calls.jsonl lines byte for byte (all 14 arms, 2 episodes) and its figure dumps."""
    N, v1, _, _ = _runs()
    out = _run(tmp_path, monkeypatch, "default", ["--dump-figure-data", "--dump-per-episode", "2"])
    ref = [ln for ln in (v1 / "calls.jsonl").read_bytes().splitlines(keepends=True)
           if json.loads(ln)["episode_id"] in EPISODES]
    mine = (out / "calls.jsonl").read_bytes().splitlines(keepends=True)
    assert len(ref) == len(mine) > 0 and mine == ref
    dumps = sorted(p.name for p in (out / "figure_dump").iterdir())
    assert dumps == sorted(p.name for p in (v1 / "figure_dump").iterdir() if any(e in p.name for e in EPISODES))
    for name in dumps:
        with np.load(out / "figure_dump" / name) as a, np.load(v1 / "figure_dump" / name) as b:
            assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
    # summary: every V1 key but provenance (settings: argv / episodes / new ats_* keys, definitions: docstring)
    s = json.loads((out / "summary.json").read_text())
    assert set(s) == {"arms", "paired_vs_ats", "settings", "definitions"}
    assert s["settings"]["ats_repr"] == "cloud_target"
    cats = [json.loads(ln) for ln in (out / "ats_calls.jsonl").read_text().splitlines()]
    assert len(cats) == len(mine) // 14 and all(r["n_ats_pred"] == r["n_cloud_target"] for r in cats)


@pytest.mark.parametrize("repr_, tol, h_arm", [("exclusion", 0.005, "ats_ex_5mm"), ("exclusion", 1e-6, "ats_ex_exact"),
                                               ("exclusion", 0.010, "ats_ex_10mm"), ("target_points", 0.005, "ats_tp_5mm")])
def test_repr_equals_h_recompute(tmp_path, monkeypatch, repr_, tol, h_arm):
    """--ats-repr exclusion / target_points == H's recompute_ats_arms.py rows (metrics and call category)."""
    if not H_ROWS.exists() or not H_CATS.exists():
        pytest.skip("ATS-CLOUD-DIAG H outputs not present")
    out = _run(tmp_path, monkeypatch, f"{repr_}_{tol}", ["--ats-repr", repr_, "--ats-tol", str(tol),
                                                       "--arms", "gt_oracle|phrase"])
    h = {}
    for ln in H_ROWS.read_text().splitlines():
        r = json.loads(ln)
        if r["episode_id"] in EPISODES and r["arm"] in (h_arm, "ats_ct"):
            h[(r["arm"], r["episode_id"], r["call_k"])] = r
    mine = [json.loads(ln) for ln in (out / "calls.jsonl").read_text().splitlines()]
    keys = [k for k in mine[0] if k not in ("arm", "ats_call_cat")]
    n = 0
    for r in mine:
        hk = {C.ATS_ARM: h_arm, C.ATS_CT_ARM: "ats_ct"}.get(r["arm"])
        if hk is None:
            continue
        ref = h[(hk, r["episode_id"], r["call_k"])]
        assert {k: r[k] for k in keys} == {k: ref[k] for k in keys}, (r["arm"], r["episode_id"], r["call_k"])
        assert r["ats_call_cat"] == H_CAT_NAME[ref["cat"]]
        n += 1
    assert n == 2 * len({(r["episode_id"], r["call_k"]) for r in mine}) == 2 * sum(
        1 for k in h if k[0] == h_arm)
    cats = {r["ats_call_cat"] for r in mine}
    assert cats == set(H_CAT_NAME.values())
    s = json.loads((out / "summary.json").read_text())
    assert s["settings"]["ats_repr"] == repr_ and s["settings"]["ats_tol_m"] == tol
    assert set(s["ats_by_category"]) == cats and C.ATS_CT_ARM in s["paired_vs_ats"]
