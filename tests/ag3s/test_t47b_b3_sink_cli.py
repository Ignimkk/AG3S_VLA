"""T47b B3 (raw-amplitude audit + x k numeric check), #5 sink diagnostic and the CLI, on synthetic data.

Pinned (no MuJoCo, no storage writes):
  - every audit point's code still exists in its file (`locate`), and the table covers lifting,
    fusion, grounding, split and TargetConfirm;
  - the real `lift` + `ground_target` / `fuse` on a synthetic cloud: x4 (exact in float) is bit
    identical, x3 keeps the decision, a per-camera scale change moves the fused peak and the chosen
    object (the variance the audit predicts for M2), a different map is caught by `first_difference`;
  - sink statistics on hand-made maps (one hot patch, constant map, two blobs, modal fraction);
  - `build_sources` against T47's `source_maps` (sigma 0 and 1: the same maps bit for bit), head sets
    read from T47-shaped json, the parity comparison, argument validation and the AG3S_RUN_DIR refusal.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t47_head_selection as hs
from AG3S.ag3s.experiments.studies import t47b_b3 as b3
from AG3S.ag3s.experiments.studies import t47b_sink as sk
from AG3S.ag3s.experiments.studies import t47b_study as st

# ---------------------------------------------------------------------------------- audit


def test_audit_points_exist_in_code():
    table = b3.audit_table()
    missing = [r["id"] for r in table if not r["found"]]
    assert not missing, f"audit points whose code is gone: {missing}"
    files = {r["file"] for r in table}
    for f in ("AG3S/ag3s/stages/attention_lifting.py", "AG3S/ag3s/runtime/multiview.py",
              "AG3S/ag3s/stages/target_grounding.py", "AG3S/ag3s/stages/attention_split.py"):
        assert f in files
    ids = [r["id"] for r in table]
    assert len(ids) == len(set(ids))
    for r in table:
        assert r["x_k"].split(" ")[0] in ("invariant", "variant")
        assert r["head_reweight"].split(" ")[0] in ("invariant", "variant")


def test_audit_absolute_constants_on_raw_are_degenerate_only():
    """The only absolute comparisons on raw amplitude in the default path are numerical guards."""
    raw_abs = [p for p in b3.AUDIT if p.reads == "raw" and not p.absolute.startswith("none")
               and p.default_active]
    assert {p.id for p in raw_abs} == {"L3"}
    assert all(p.x_k.startswith("variant (degenerate only)") for p in raw_abs)
    nondefault = {p.id for p in b3.AUDIT if not p.default_active}
    assert {"L5", "L6", "G3b"} <= nondefault


# ------------------------------------------------------------------------ numeric check


def two_blobs(spacing=0.01, n=5, gap=0.5):
    """Two 5x5 point patches (25 points each) at x = 0 and x = gap; uv puts blob 1 in u < 8."""
    from AG3S.ag3s.types import PointCloud

    g = np.arange(n) * spacing
    xs, ys = np.meshgrid(g, g, indexing="ij")
    p1 = np.column_stack([xs.ravel(), ys.ravel(), np.full(n * n, 0.3)])
    p2 = p1 + [gap, 0.0, 0.0]
    uv1 = np.column_stack([np.arange(n * n) % 8, 2 + np.arange(n * n) // 8])
    uv2 = uv1 + [8, 0]
    pts = np.vstack([p1, p2])
    uv = np.vstack([uv1, uv2]).astype(np.int32)
    return PointCloud(pts, uv), n * n


def realistic_map(seed=0, hot_left=3.0):
    rng = np.random.default_rng(seed)
    m = rng.uniform(1e-3, 3e-3, (16, 16))
    m[:, :8] *= hot_left
    return m


def test_x4_is_bit_identical_and_x3_keeps_the_decision():
    cloud, n1 = two_blobs()
    m = realistic_map()
    a = b3.snapshot(*b3.lift_and_ground(cloud, m, image_hw=(16, 16)))
    assert a["has_target"] and set(a["chosen"].tolist()) <= set(range(n1))
    for k in (4.0, 0.25):
        b = b3.snapshot(*b3.lift_and_ground(cloud, (m * k).astype(np.float32).astype(np.float64),
                                            image_hw=(16, 16)))
        c = b3.compare_snapshots(a, b, k)
        assert c["bit_identical"] and c["first_difference"] is None, c
    b = b3.snapshot(*b3.lift_and_ground(cloud, m * 3.0, image_hw=(16, 16)))
    c = b3.compare_snapshots(a, b, 3.0)
    assert c["has_target_equal"] and c["chosen_equal"] and c["status_equal"]
    assert c["raw_max_rel_err"] < 1e-6


def test_a_different_map_is_caught():
    cloud, n1 = two_blobs()
    a = b3.snapshot(*b3.lift_and_ground(cloud, realistic_map(hot_left=3.0), image_hw=(16, 16)))
    b = b3.snapshot(*b3.lift_and_ground(cloud, realistic_map(hot_left=1 / 3.0), image_hw=(16, 16)))
    c = b3.compare_snapshots(a, b, 1.0)
    assert not c["decision_identical"] and not c["chosen_equal"]
    assert c["first_difference"] == "raw_order"
    assert set(b["chosen"].tolist()) <= set(range(n1, 2 * n1))


def test_fusion_common_scale_invariant_per_camera_scale_variant():
    cloud, n1 = two_blobs()
    m_a = np.ones((16, 16))
    m_a[:, :8] = 2.0  # camera A: blob 1 hottest (raw 2)
    m_b = np.ones((16, 16))
    m_b[:, 8:] = 1.5  # camera B: blob 2 (raw 1.5)
    ac, g, _ = b3.fuse_and_ground([cloud, cloud], [m_a, m_b], image_hw=(16, 16), voxel_size=0.005)
    base = b3.snapshot(ac, g)
    assert base["has_target"] and set(base["chosen"].tolist()) == set(range(n1))
    ac4, g4, _ = b3.fuse_and_ground([cloud, cloud], [m_a * 4, m_b * 4], image_hw=(16, 16),
                                    voxel_size=0.005)
    assert b3.compare_snapshots(base, b3.snapshot(ac4, g4), 4.0)["bit_identical"]
    # camera B x2 only: its raw 3.0 now wins the max on blob 2 -> the target moves (audit M2)
    acc, gc, _ = b3.fuse_and_ground([cloud, cloud], [m_a, m_b * 2], image_hw=(16, 16),
                                    voxel_size=0.005)
    moved = b3.snapshot(acc, gc)
    c = b3.compare_snapshots(base, moved, 1.0)
    assert not c["decision_identical"] and c["first_difference"] == "raw_order"
    assert set(moved["chosen"].tolist()) == set(range(n1, 2 * n1))


def test_summarize_b3():
    rows = [{"source": "l8h2", "factor": 3.0, "decision_identical": True, "bit_identical": False,
             "first_difference": "normalized", "norm_bit_equal": False,
             "raw_scaled_bit_equal": False, "raw_max_rel_err": 1e-8, "norm_max_abs_diff": 6e-8},
            {"source": "l8h2", "factor": 3.0, "decision_identical": False, "bit_identical": False,
             "first_difference": "seeds", "record": "ep1850", "frame": 4, "status_a": "ok",
             "status_b": "ok", "norm_bit_equal": False, "raw_scaled_bit_equal": False,
             "raw_max_rel_err": 1e-8, "norm_max_abs_diff": 6e-8},
            {"source": "l8h2", "factor": 4.0, "decision_identical": True, "bit_identical": True,
             "first_difference": None, "norm_bit_equal": True, "raw_scaled_bit_equal": True,
             "raw_max_rel_err": 0.0, "norm_max_abs_diff": 0.0}]
    s = b3.summarize_b3(rows)
    assert s["l8h2@x3"]["n_frames"] == 2 and s["l8h2@x3"]["decision_identical"] == 1
    assert s["l8h2@x3"]["first_difference_counts"] == {"normalized": 1, "seeds": 1}
    assert s["l8h2@x3"]["frames_not_decision_identical"][0]["frame"] == 4
    assert s["l8h2@x4"]["bit_identical"] == 1


# ----------------------------------------------------------------------------------- sink


def test_sink_map_stats():
    hot = np.full((16, 16), 0.001)
    hot[1, 11] = 1.0  # a fixed-patch sink
    const = np.full((16, 16), 0.5)
    two = np.zeros((16, 16))
    two[0:2, 0:2] = 1.0
    two[10:14, 10:14] = 1.0
    s = sk.map_stats(np.stack([hot, const, two]))
    assert s["argmax_patch"][0] == 1 * 16 + 11
    assert s["top1_share"][0] == pytest.approx(1.0 / (1.0 + 255 * 0.001))
    assert s["n_components"].tolist() == [1, 0, 2]
    assert s["largest_component_px"].tolist() == [1, 0, 16]
    assert s["largest_share_of_binarized"][2] == pytest.approx(16 / 20)
    assert s["largest_share_of_grid"][2] == pytest.approx(16 / 256)
    assert math.isnan(s["largest_share_of_grid"][1]) and math.isnan(s["entropy"][1])
    assert s["entropy"][0] == 0.0
    assert s["entropy"][2] == pytest.approx(-(0.2 * math.log(0.2) + 0.8 * math.log(0.8)))
    np.testing.assert_allclose(s["S_cam"], [hot.sum(), 128.0, 20.0])


def test_modal_fraction_and_summary():
    p = np.array([[5, 1], [5, 2], [7, 2], [5, 1]])  # frames x 2 maps
    frac, mode = sk.modal_fraction(p)
    assert frac.tolist() == [0.75, 0.5] and mode.tolist() == [5, 1]  # tie 1 vs 2 -> lowest
    stats = sk.map_stats(np.random.default_rng(0).random((6, 2, 3, 16, 16)))
    summ = sk.summarize_cells(stats)
    assert summ["modal_fraction"].shape == (2, 3) and summ["S_cam_mean"].shape == (2, 3)
    assert summ["modal_patch_rc"].shape == (2, 3, 2)
    assert sk.mark_cells([(2, 0), (8, 2), (17, 7)]) == [0, 50, 127]
    assert sk.classify_cell(0.96, 0.67, 0.004) == "fixed_patch_sink"
    assert sk.classify_cell(0.2, 0.02, 0.4) == "broad_single_blob"
    assert sk.classify_cell(0.2, 0.02, 0.05) == "other"


# -------------------------------------------------------------------------------- the CLI


def fake_att(seed=0, f=3):
    rng = np.random.default_rng(seed)
    att = rng.random((f, 3, 3, 18, 8, 3, 16, 16)).astype(np.float16)
    return att, [0, 4, 9], ["mean", "first", "last"]


def fake_selection():
    cam = {"common": {"topk": {"3": [[4, 1], [2, 0], [14, 5]]}},
           "per_camera": {c: {"topk": {"3": cells}} for c, cells in zip(
               hs.POLICY_CAMERAS, ([[4, 7], [3, 3], [4, 1]], [[15, 2], [9, 6], [9, 2]],
                                   [[4, 1], [4, 6], [2, 3]]))}}
    return {"settings": {"d0_last": {"denoise": 0, "agg": "last", "tau_source": {"cam": cam}}},
            "policy": {"n_lowest": 10}}


def fake_oracle():
    return {"settings": {"d0_last": {"top1": [[8, 3]], "top3": [[8, 3], [10, 7], [12, 4]]}},
            "records": ["long", "t2", "ep1807"]}


def write_t47(tmp_path):
    (tmp_path / "select").mkdir()
    (tmp_path / "validate").mkdir()
    (tmp_path / "select" / "selection.json").write_text(json.dumps(fake_selection()))
    (tmp_path / "validate" / "oracle.json").write_text(json.dumps(fake_oracle()))
    return tmp_path


def test_load_head_sets(tmp_path):
    heads = st.load_head_sets(write_t47(tmp_path))
    assert heads["common3"] == [(4, 1), (2, 0), (14, 5)]
    assert heads["oracle3"] == [(8, 3), (10, 7), (12, 4)]
    assert heads["camera3"][1] == [(15, 2), (9, 6), (9, 2)]
    assert heads["l8h2"] == [(8, 2)]
    assert len(heads["files"]["oracle.json"]["sha256"]) == 64


@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_build_sources_matches_t47_source_maps(tmp_path, sigma):
    att, dn, agl = fake_att()
    heads = st.load_head_sets(write_t47(tmp_path))
    ours = st.build_sources(att, dn, agl, heads, list(st.SOURCE_BASES), sigma)
    t = f"s{sigma:g}"
    assert set(ours) == {"l8h2", f"common3_raw_sum_{t}"} | {
        f"{h}_{m}_{t}" for h in st.HEAD_SETS for m in
        ("raw_sum", "mass_norm_joint", "mass_norm_per_camera")}
    theirs = hs.source_maps(att, dn, agl, fake_selection(), setting="d0_last",
                            sources=["l8h2", f"topk_common_k3_{t}_cam", f"topk_camera_k3_{t}_cam",
                                     f"oracle_top3_{t}", "head_avg"], oracle=fake_oracle())
    assert np.array_equal(ours["l8h2"], theirs["l8h2"])
    assert np.array_equal(ours[f"common3_raw_sum_{t}"], theirs[f"topk_common_k3_{t}_cam"])
    assert np.array_equal(ours[f"camera3_raw_sum_{t}"], theirs[f"topk_camera_k3_{t}_cam"])
    assert np.array_equal(ours[f"oracle3_raw_sum_{t}"], theirs[f"oracle_top3_{t}"])
    if sigma == 0.0:
        assert np.array_equal(ours["headavg_raw_sum_s0"], theirs["head_avg"])
    for name, m in ours.items():
        assert m.shape == (3, 3, 16, 16) and np.all(np.isfinite(m)), name
    np.testing.assert_allclose(ours[f"oracle3_mass_norm_joint_{t}"].sum(axis=(1, 2, 3)), 3.0,
                               rtol=1e-6)
    np.testing.assert_allclose(ours[f"headavg_mass_norm_per_camera_{t}"].sum(axis=(2, 3)), 1.0,
                               rtol=1e-6)


def test_source_names_and_oracle_refusal():
    assert st.source_names(["l8h2", "oracle3_mass_norm_joint"], 1.0) == {
        "l8h2": "l8h2", "oracle3_mass_norm_joint": "oracle3_mass_norm_joint_s1"}
    with pytest.raises(SystemExit):
        st.source_names(["oracle3_mean"], 1.0)
    with pytest.raises(SystemExit, match="oracle"):
        st._refuse_oracle_on_selection(["oracle3_raw_sum"], ["long", "ep1850"])
    st._refuse_oracle_on_selection(["oracle3_raw_sum"], ["ep1850", "headfix"])
    st._refuse_oracle_on_selection(["l8h2"], ["long"])


def test_parity_compare_rows():
    t47 = [{"record": "ep1850", "frame": 0, "source": "topk_common", "has_target": True,
            "on_target": False, "silent_failure": True, "centroid_error_m": 0.2,
            "grounding_status": "ok", "n_clusters": 3, "n_cloud": 10, "n_cloud_target": 2,
            "n_seeds": 5, "n_seeds_target": 0, "seed_camera": "head", "seed_precision": 0.0,
            "seed_recall": 0.0, "chosen_target_fraction": 0.0},
           {"record": "ep1850", "frame": 1, "source": "topk_common", "has_target": False,
            "on_target": False, "silent_failure": False, "centroid_error_m": float("nan"),
            "grounding_status": "no_seed", "n_clusters": 0, "n_cloud": 10, "n_cloud_target": 2,
            "n_seeds": 0, "n_seeds_target": 0, "seed_camera": None,
            "seed_precision": float("nan"), "seed_recall": 0.0,
            "chosen_target_fraction": float("nan")}]
    t47b = [dict(r, source="common3_raw_sum_s0") for r in t47]
    t47b[1] = dict(t47b[1], n_cloud=11)
    out = st.compare_rows(t47, t47b)["topk_common"]
    assert out["matched"] == 2 and out["identical"] == 1
    assert out["differing"] == [{"record": "ep1850", "frame": 1, "fields": ["n_cloud"]}]


def test_cli_argument_checks(monkeypatch):
    ap = st.build_parser()
    for argv in (["--stage", "sink", "--workers", "3"], ["--stage", "e2e", "--workers", "5"],
                 ["--stage", "b3", "--factors", "0"], ["--stage", "e2e", "--sources", "nope"],
                 ["--stage", "e2e", "--esdf-backend", "curobo"], ["--stage", "b2d", "--sigmas", "-1"],
                 ["--stage", "e2e", "--records", "nope"]):
        with pytest.raises(SystemExit):
            st.validate_args(ap, ap.parse_args(argv))
    st.validate_args(ap, ap.parse_args(["--stage", "sink", "--workers", "2"]))
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="AG3S_RUN_DIR"):
        st.out_root()
    with pytest.raises(SystemExit, match="AG3S_RUN_DIR"):
        st.main(["--stage", "audit"])


# ------------------------------------------------------------ replay glue (no rendering)


def test_t47_row_and_b2_fields_on_a_fake_frame():
    from types import SimpleNamespace

    from AG3S.ag3s.experiments.studies import t47b_metrics as bm
    from AG3S.ag3s.experiments.studies import t47b_replay as rp
    from AG3S.ag3s.fields.curobo_builder import TargetBall

    T, D, S = (bm.POINT_LABELS.index(n) for n in ("target", "distractor", "support"))
    pts = np.array([[0, 0, 0.30], [0.01, 0, 0.30], [0.02, 0, 0.30], [0.06, 0, 0.30],
                    [0.5, 0, 0.0]], float)
    lab = np.array([T, T, T, D, S])
    lut = {tuple(p): l for p, l in zip(pts.round(6), lab)}

    def label_of(p):
        return np.array([lut.get(tuple(q), bm.POINT_LABELS.index("unlabelled"))
                         for q in np.asarray(p, float).reshape(-1, 3).round(6)], np.int16)

    chosen = np.array([0, 1, 3])  # two target points + the neighbour
    target = SimpleNamespace(centroid=pts[chosen].mean(0), points=pts[chosen],
                             point_indices=chosen, seed_camera=None)
    cs = SimpleNamespace(has_target=True, target=target, support_surfaces=[])
    debug = {"filtered_cloud": SimpleNamespace(points=pts),
             "support_mask": np.array([False, False, False, False, True]),
             "seed_indices": np.array([0, 3]),
             "grounding": SimpleNamespace(status=SimpleNamespace(value="ok"), clusters=[1, 2])}
    row, arrays = rp.t47_row(cs, debug, label_of=label_of, gt_centre=np.array([0.01, 0, 0.3]),
                             on_target_radius=0.10)
    assert row["on_target"] and row["n_cloud_target"] == 3 and row["n_seeds_target"] == 1
    assert row["seed_precision"] == 0.5 and row["chosen_target_fraction"] == pytest.approx(2 / 3)
    fake_ag = SimpleNamespace(_target_ball=lambda p, s, voxel: TargetBall(
        centre=np.array([0.01, 0, 0.3]), radius=0.03, z_min=0.1))
    m = rp.b2_fields(fake_ag, cs, arrays, ball_voxel=0.005, membership_radius=0.005)
    assert m["chosen_membership"] == "indices"
    assert m["identity_correct"] and m["chosen_cluster_purity"] == pytest.approx(2 / 3)
    assert m["nontarget_seed_loss__other_object"] == 1 and m["target_residual_n"] == 1
    assert m["ball_built"] and m["ball_n"] == 3 and m["ball_target_residual_n"] == 0
    assert m["ball"]["radius_m"] == pytest.approx(0.03)
    none_ag = SimpleNamespace(_target_ball=lambda p, s, voxel: None)
    m2 = rp.b2_fields(none_ag, cs, arrays, ball_voxel=0.005, membership_radius=0.005)
    assert not m2["ball_built"] and "error" in m2["ball"]
