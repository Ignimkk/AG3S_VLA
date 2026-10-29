"""T47 — localization-head selection (Kang et al., arXiv 2503.06287) on synthetic data.

Pinned here (no MuJoCo rendering, no storage reads):
  - spatial entropy (appendix B.1 Eq.4–7): one blob -> 0, two equal blobs -> ln 2, a diagonal pair is
    one 8-connected component, constant maps -> NaN (not selectable), no joining across stacked maps;
  - the knee threshold tau on curves with a known knee (kneedle primary, discrete curvature variant);
  - selection frequency on a hand-computed example; ineligible / NaN cells are never selected;
  - combine_maps: sigma 0 = plain sum, Gaussian smoothing preserves each map's mass (reflect boundary);
  - 2D scoring (mass / lift / peak / IoU / class decomposition), selection end to end on a synthetic
    attention block (L0/L1 excluded even when perfect), source map construction, e2e summaries;
  - body classes on a tiny in-memory MJCF (gripper = bodies below link_*_arm_6, mocap -> background);
  - CLI argument validation and the AG3S_RUN_DIR refusal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t47_head_metrics as hm
from AG3S.ag3s.experiments.studies import t47_head_selection as hs


def blob(*pixels, shape=(16, 16), value=1.0):
    m = np.zeros(shape)
    for r, c in pixels:
        m[r, c] = value
    return m


# ------------------------------------------------------------------------- spatial entropy


def test_single_blob_has_zero_entropy():
    m = blob((3, 3), (3, 4), (4, 3), (4, 4))
    assert hm.spatial_entropy(m) == 0.0
    h, n = hm.spatial_entropy_batch(m[None])
    assert n[0] == 1 and h[0] == 0.0


def test_two_equal_blobs_give_ln2():
    m = blob((1, 1), (1, 2), (10, 10), (10, 11))
    assert hm.spatial_entropy(m) == pytest.approx(math.log(2))


def test_component_size_weights_not_attention_mass():
    # sizes 1 and 3 -> P = (1/4, 3/4); the bigger value on the small blob must not matter
    m = blob((0, 0), value=50.0) + blob((8, 8), (8, 9), (9, 8))
    want = -(0.25 * math.log(0.25) + 0.75 * math.log(0.75))
    assert hm.spatial_entropy(m) == pytest.approx(want)


def test_diagonal_pixels_are_one_component_under_8_connectivity():
    m = blob((5, 5), (6, 6))
    h, n = hm.spatial_entropy_batch(m[None])
    assert n[0] == 1
    assert h[0] == 0.0
    # and an anti-diagonal chain too
    m2 = blob((2, 9), (3, 8), (4, 7))
    assert hm.spatial_entropy_batch(m2[None])[1][0] == 1


def test_only_pixels_above_the_mean_count():
    base = np.full((16, 16), 0.5)
    base[2, 2] = 3.0
    base[12, 12] = 0.6
    # mean = (254*0.5 + 3.0 + 0.6)/256 ≈ 0.5101 -> both are above it: two components
    assert hm.spatial_entropy_batch(base[None])[1][0] == 2
    base[12, 12] = 0.505  # mean ≈ 0.5098 -> now below it: one component
    assert hm.spatial_entropy_batch(base[None])[1][0] == 1


@pytest.mark.parametrize("value", [0.0, 0.1, 1.0 / 3.0, 7.25])
def test_constant_map_has_no_component_and_nan_entropy(value):
    m = np.full((16, 16), value)
    h, n = hm.spatial_entropy_batch(m[None])
    assert n[0] == 0
    assert np.isnan(h[0])
    assert np.isnan(hm.spatial_entropy(m))


def test_stacked_maps_never_join_and_shape_is_kept():
    a = blob((15, 0), (15, 1))  # bottom row
    b = blob((0, 0), (0, 1))  # top row: would touch `a` if the stack axis were connected
    c = blob((0, 0), (15, 15))
    maps = np.stack([a, b, c, np.full((16, 16), 2.0)]).reshape(2, 2, 16, 16)
    h, n = hm.spatial_entropy_batch(maps, chunk=3)  # chunk boundary inside the stack
    assert h.shape == (2, 2) and n.shape == (2, 2)
    np.testing.assert_array_equal(n, [[1, 1], [2, 0]])
    assert h[0, 0] == 0.0 and h[0, 1] == 0.0
    assert h[1, 0] == pytest.approx(math.log(2))
    assert np.isnan(h[1, 1])


def test_nan_maps_are_rejected():
    m = np.zeros((16, 16))
    m[0, 0] = np.nan
    with pytest.raises(ValueError):
        hm.spatial_entropy(m)


# ------------------------------------------------------------------------------ tau (knee)


def _knee_curve(n_flat=40, n_steep=10, rng=None):
    flat = np.linspace(0.0, 0.05, n_flat)
    steep = np.linspace(0.05, 1.0, n_steep + 1)[1:]
    v = np.concatenate([flat, steep])
    if rng is not None:
        v = rng.permutation(v)
    return v, n_flat - 1  # knee = last flat point (ascending index)


def test_kneedle_finds_the_known_knee():
    rng = np.random.default_rng(0)
    v, knee = _knee_curve(rng=rng)
    k = hm.max_curvature_threshold(v, method="kneedle")
    assert k.knee_found
    assert k.index == knee
    assert k.tau == pytest.approx(0.05)
    elig = v >= k.tau
    assert elig.sum() == 11  # the knee point itself + the 10 steep values


def test_discrete_curvature_finds_the_known_knee_within_one():
    v, knee = _knee_curve()
    k = hm.max_curvature_threshold(v, method="curvature")
    assert k.knee_found
    assert abs(k.index - knee) <= 1
    assert k.index not in (0, len(v) - 1)


def test_degenerate_curves_make_everything_eligible():
    for v in (np.full(10, 0.3), np.array([1.0, 2.0]), np.linspace(0, 1, 5) ** 0.5):
        k = hm.max_curvature_threshold(v, method="kneedle")
        if np.ptp(v) == 0 or len(v) < 3:
            assert not k.knee_found and k.tau == pytest.approx(v.min())
    # concave (rises then plateaus): nothing below the chord
    k = hm.max_curvature_threshold(np.sqrt(np.linspace(0, 1, 30)), method="kneedle")
    assert not k.knee_found and k.tau == 0.0
    with pytest.raises(ValueError):
        hm.max_curvature_threshold([1.0, np.nan, 2.0])
    with pytest.raises(ValueError):
        hm.max_curvature_threshold([1.0, 2.0, 3.0], method="spline")


def test_eligible_cells_uses_a_tau_per_sample():
    v1, _ = _knee_curve()
    v2 = v1 * 10.0
    s = np.stack([v1, v2])
    elig, tau, found = hm.eligible_cells(s)
    assert found.all()
    assert tau[1] == pytest.approx(10 * tau[0])
    np.testing.assert_array_equal(elig[0], elig[1])


# --------------------------------------------------------------------- selection frequency


def test_selection_frequency_matches_a_hand_computed_example():
    # 3 samples x 4 cells, choose the 2 lowest-entropy eligible cells per sample
    ent = np.array([[0.1, 0.5, 0.2, 0.9],
                    [0.3, 0.1, np.nan, 0.2],
                    [0.0, 0.0, 0.0, 0.0]])
    elig = np.array([[True, True, True, True],
                     [False, True, True, True],
                     [True, False, True, True]])
    sf = hm.selection_frequency(ent, elig, n_lowest=2)
    # s0: cells 0 (0.1), 2 (0.2)
    # s1: cell 0 ineligible, cell 2 NaN -> candidates 1 (0.1), 3 (0.2) -> both
    # s2: cell 1 ineligible; ties at 0.0 broken by index -> 0, 2
    np.testing.assert_array_equal(sf.counts, [2, 1, 2, 1])
    np.testing.assert_allclose(sf.frequency, np.array([2, 1, 2, 1]) / 3)
    np.testing.assert_array_equal(sf.n_selected, [2, 2, 2])
    assert sf.n_samples == 3


def test_ineligible_cell_is_never_selected_even_with_the_lowest_entropy():
    ent = np.array([[0.0, 1.0, 2.0]] * 5)
    elig = np.array([[False, True, True]] * 5)
    sf = hm.selection_frequency(ent, elig, n_lowest=1)
    np.testing.assert_array_equal(sf.counts, [0, 5, 0])


def test_fewer_candidates_than_n_lowest_takes_all_and_counts_empty_samples():
    ent = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    elig = np.array([[True, False, False], [False, False, False]])
    sf = hm.selection_frequency(ent, elig, n_lowest=10)
    np.testing.assert_array_equal(sf.counts, [1, 0, 0])
    np.testing.assert_array_equal(sf.n_selected, [1, 0])
    assert sf.frequency[0] == 0.5  # the empty sample stays in the denominator


def test_rank_cells_orders_by_frequency_then_tiebreaks():
    freq = np.array([0.5, 0.9, 0.5, 0.0])
    me = np.array([1.0, 2.0, 0.5, 0.0])
    order = hm.rank_cells(freq, mean_entropy=me)
    np.testing.assert_array_equal(order, [1, 2, 0, 3])


# --------------------------------------------------------------------------- combine maps


def _cell_block(rng, n_l=4, n_h=3):
    return rng.random((2, n_l, n_h, 16, 16))


def test_combine_maps_sigma0_is_the_plain_sum():
    rng = np.random.default_rng(1)
    maps = _cell_block(rng)
    cells = [(1, 2), (3, 0), (0, 1)]
    out = hm.combine_maps(maps, cells, sigma=0.0)
    want = maps[:, 1, 2] + maps[:, 3, 0] + maps[:, 0, 1]
    np.testing.assert_allclose(out, want)
    assert out.shape == (2, 16, 16)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_smoothing_preserves_mass_including_corner_deltas(sigma):
    rng = np.random.default_rng(2)
    maps = _cell_block(rng)
    maps[0, 0, 0] = 0.0
    maps[0, 0, 0, 0, 0] = 5.0  # a delta in the corner — the worst case for leakage
    sm = hm.gaussian_smooth(maps, sigma)
    np.testing.assert_allclose(sm.sum(axis=(-1, -2)), maps.sum(axis=(-1, -2)), rtol=1e-12)
    assert not np.allclose(sm, maps)
    out = hm.combine_maps(maps, [(0, 0), (2, 1)], sigma)
    np.testing.assert_allclose(out.sum(axis=(-1, -2)),
                               maps[:, 0, 0].sum(axis=(-1, -2)) + maps[:, 2, 1].sum(axis=(-1, -2)),
                               rtol=1e-12)


def test_combine_maps_rejects_bad_input():
    with pytest.raises(ValueError):
        hm.combine_maps(np.zeros((16, 16)), [(0, 0)])
    with pytest.raises(ValueError):
        hm.combine_maps(np.zeros((2, 2, 16, 16)), [])
    with pytest.raises(ValueError):
        hm.gaussian_smooth(np.zeros((16, 16)), -1.0)


def test_head_average_skips_layers_0_and_1():
    maps = np.zeros((1, 4, 2, 16, 16))
    maps[:, 0] = 100.0
    maps[:, 1] = 100.0
    maps[:, 2:] = 1.0
    np.testing.assert_allclose(hm.head_average(maps), 1.0)
    assert hm.cell_list(4, 2) == [(2, 0), (2, 1), (3, 0), (3, 1)]


def test_image_attention_sum():
    maps = np.ones((5, 3, 16, 16))
    maps[:, 1] *= 2.0
    s_cam, s_img = hm.image_attention_sum(maps)
    np.testing.assert_allclose(s_cam, [[256, 512, 256]] * 5)
    np.testing.assert_allclose(s_img, [1024] * 5)
    with pytest.raises(ValueError):
        hm.image_attention_sum(np.ones((3, 8, 8)))


# ------------------------------------------------------------------------------ 2D scores


def _coverage(target_patches, n=1):
    cov = np.zeros((n, len(hm.CLASS_NAMES), 16, 16))
    cov[:, -1] = 1.0  # background everywhere
    for r, c, share in target_patches:
        cov[:, 0, r, c] = share
        cov[:, -1, r, c] = 1.0 - share
    return cov


def test_map_scores_on_a_perfect_and_a_uniform_map():
    cov = _coverage([(4, 4, 1.0), (4, 5, 1.0), (5, 4, 1.0), (5, 5, 1.0)], n=2)
    perfect = blob((4, 4), (4, 5), (5, 4), (5, 5))
    uniform = np.ones((16, 16))
    sc = hm.map_scores(np.stack([perfect, uniform]), cov)
    np.testing.assert_allclose(sc["class_mass"].sum(axis=1), 1.0)
    assert sc["target_mass"][0] == pytest.approx(1.0)
    assert sc["lift"][0] == pytest.approx(64.0)
    assert sc["lift"][1] == pytest.approx(1.0)
    assert sc["peak_on_target"][0] == 1.0
    assert sc["iou@0.5"][0] == pytest.approx(1.0)
    assert sc["iou@0.5"][1] == 0.0  # nothing above the mean of a constant map


def test_iou_is_nan_when_the_gt_mask_is_empty_not_zero():
    cov = _coverage([(4, 4, 0.3)])  # target never reaches 0.5 of a patch
    sc = hm.map_scores(blob((4, 4))[None], cov)
    assert np.isnan(sc["iou@0.5"][0])
    assert sc["iou@0.25"][0] == pytest.approx(1.0)
    assert sc["peak_on_target"][0] == 1.0


def test_spearman_handles_degenerate_input():
    assert np.isnan(hm.spearman([1, 1, 1], [1, 2, 3])["rho"])
    r = hm.spearman([1, 2, 3, 4], [10, 20, 30, 40])
    assert r["rho"] == pytest.approx(1.0) and r["n"] == 4
    assert np.isnan(hm.spearman([1, np.nan], [1, 2])["rho"])


# ------------------------------------------------------------- selection on a synthetic block


def _synthetic_stats(n_frames=6, n_l=5, n_h=3, good=((3, 1), (4, 0)), perfect_l0=True, seed=0):
    """attention (F, D=1, A=1, L, H, C=3, 16, 16): `good` cells = one bright compact blob (high S,
    entropy 0), others = low-mass scattered noise. L0 is made 'perfect' to check its exclusion."""
    rng = np.random.default_rng(seed)
    att = rng.random((n_frames, 1, 1, n_l, n_h, 3, 16, 16)) * 1e-3
    for (l, h) in good:
        att[:, 0, 0, l, h, :, 6:8, 6:8] += 0.5
    if perfect_l0:
        att[:, 0, 0, 0, :, :, 2:4, 2:4] += 5.0
    return att


def test_select_picks_the_compact_high_mass_heads_and_never_layer_0():
    att = _synthetic_stats()
    s_cam, ent, ncomp = hs.record_statistics(att)
    stats = {"rec": {"S_cam": s_cam, "entropy": ent}}
    cells = hm.cell_list(att.shape[3], att.shape[4])
    res = hs.select_from_stats(stats, ["rec"], 0, 0, cells, tau_source="cam",
                               tau_method="kneedle", n_lowest=2, kmax=3)
    assert sorted(map(tuple, res["common"]["topk"]["2"])) == [(3, 1), (4, 0)]
    for cam in hs.POLICY_CAMERAS:
        assert sorted(map(tuple, res["per_camera"][cam]["topk"]["2"])) == [(3, 1), (4, 0)]
    assert all(c[0] not in hm.EXCLUDED_LAYERS for c in cells)
    assert res["common"]["n_samples"] == att.shape[0] * 3
    res_img = hs.select_from_stats(stats, ["rec"], 0, 0, cells, tau_source="img",
                                   tau_method="curvature", n_lowest=2, kmax=3)
    assert sorted(map(tuple, res_img["common"]["topk"]["2"])) == [(3, 1), (4, 0)]


def _fake_selection(cells_common, cells_cam):
    block = lambda cells: {"topk": {str(k): [list(c) for c in cells[:k]] for k in range(1, 4)}}
    return {"settings": {"d0_last": {"denoise": 0, "agg": "last", "tau_source": {
        "cam": {"common": block(cells_common),
                "per_camera": {c: block(cells_cam[c]) for c in hs.POLICY_CAMERAS}}}}}}


def test_source_maps_build_each_source_from_the_right_cells():
    rng = np.random.default_rng(3)
    att = rng.random((2, 3, 3, 18, 8, 3, 16, 16))
    denoise, aggs = [0, 4, 9], ["mean", "first", "last"]
    cam_cells = {"cam_high": [(5, 1), (6, 2), (7, 3)],
                 "cam_left_wrist": [(9, 0), (10, 1), (11, 2)],
                 "cam_right_wrist": [(12, 4), (13, 5), (14, 6)]}
    sel = _fake_selection([(3, 3), (4, 4), (5, 5)], cam_cells)
    oracle = {"settings": {"d0_last": {"top1": [[2, 7]], "top3": [[2, 7], [3, 6], [4, 5]]}}}
    names = ["l8h2", "head_avg", "uniform", "topk_common_k3_s0_cam", "topk_camera_k2_s0_cam",
             "oracle_top3_s0", "cell_L8H2"]
    maps = hs.source_maps(att, denoise, aggs, sel, setting="d0_last", sources=names, oracle=oracle)
    last = att[:, 0, 2]  # d0, agg last: (F, L, H, C, 16, 16)
    np.testing.assert_allclose(maps["l8h2"], last[:, 8, 2])
    np.testing.assert_allclose(maps["cell_L8H2"], maps["l8h2"])
    np.testing.assert_allclose(maps["topk_common_k3_s0_cam"],
                               last[:, 3, 3] + last[:, 4, 4] + last[:, 5, 5])
    for ci, cam in enumerate(hs.POLICY_CAMERAS):
        (l1, h1), (l2, h2) = cam_cells[cam][:2]
        np.testing.assert_allclose(maps["topk_camera_k2_s0_cam"][:, ci],
                                   last[:, l1, h1, ci] + last[:, l2, h2, ci])
    np.testing.assert_allclose(maps["oracle_top3_s0"],
                               last[:, 2, 7] + last[:, 3, 6] + last[:, 4, 5])
    np.testing.assert_allclose(maps["head_avg"], last[:, 2:].mean(axis=(1, 2)))
    assert maps["uniform"].shape == (2, 3, 16, 16)
    with pytest.raises(SystemExit):
        hs.source_maps(att, denoise, aggs, sel, setting="d0_last", sources=["oracle_top1_s0"])
    with pytest.raises(SystemExit):
        hs.source_maps(att, denoise, aggs, sel, setting="d0_last", sources=["nonsense"])


# ------------------------------------------------------------------------------ e2e helpers


def test_e2e_source_names_and_summary():
    names = hs.e2e_source_names(["l8h2", "topk_common", "topk_camera", "head_avg", "oracle"],
                                k=3, sigma=0.0, tau_source="cam")
    assert names["topk_common"] == "topk_common_k3_s0_cam"
    assert names["oracle"] == "oracle_top3_s0"
    with pytest.raises(SystemExit):
        hs.e2e_source_names(["sam"], k=3, sigma=0.0, tau_source="cam")
    from AG3S.ag3s.experiments.studies.t47_replay import summarize_e2e

    def row(src, has, on, prec=0.5, n_t=10):
        return {"source": src, "has_target": has, "on_target": on,
                "silent_failure": has and not on, "n_cloud_target": n_t,
                "seed_precision": prec, "seed_recall": 0.2, "chosen_target_fraction": 0.9,
                "centroid_error_m": 0.05 if on else 0.3, "ag3s_ms": 100.0}
    rows = [row("a", True, True), row("a", True, False), row("a", False, False, prec=float("nan"),
                                                             n_t=0)]
    s = summarize_e2e(rows)["a"]
    assert s["n_frames"] == 3 and s["has_target"] == 2 and s["on_target"] == 1
    assert s["silent_failure"] == 1
    assert s["on_target_rate"] == pytest.approx(1 / 3)
    assert s["n_frames_target_in_cloud"] == 2
    assert s["seed_precision_mean"] == pytest.approx(0.5)


# ------------------------------------------------------------------------- body classes


_MJCF = """
<mujoco>
  <worldbody>
    <body name="table" pos="0 0 0"><geom type="box" size="0.5 0.5 0.02"/></body>
    <body name="crate" pos="0.3 0 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.05"/></body>
    <body name="apple" pos="0 0.2 0.1"><freejoint/><geom type="sphere" size="0.04"/></body>
    <body name="pear" pos="0 -0.2 0.1"><freejoint/><geom type="sphere" size="0.04"/></body>
    <body name="office"><geom type="plane" size="3 3 0.1"/></body>
    <body name="base" pos="-0.5 0 0">
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="link_left_arm_6" pos="0 0.3 0.5">
        <joint name="j6" type="hinge"/><geom type="sphere" size="0.03"/>
        <body name="EE_BODY_L"><geom type="sphere" size="0.02"/>
          <body name="ee_finger_l1"><joint name="f1" type="slide"/><geom type="box" size="0.01 0.01 0.02"/></body>
          <body name="ee_finger_l2"><joint name="f2" type="slide"/><geom type="box" size="0.01 0.01 0.02"/></body>
        </body>
      </body>
      <body name="link_right_arm_6" pos="0 -0.3 0.5">
        <joint name="j6r" type="hinge"/><geom type="sphere" size="0.03"/>
        <body name="ee_finger_r1"><joint name="f1r" type="slide"/><geom type="box" size="0.01 0.01 0.02"/></body>
      </body>
    </body>
    <body name="left_ee_target" mocap="true"><geom type="sphere" size="0.01" contype="0" conaffinity="0"/></body>
  </worldbody>
</mujoco>
"""


def test_body_class_table_on_a_tiny_model():
    mujoco = pytest.importorskip("mujoco")
    from AG3S.ag3s.experiments.studies.t47_replay import body_class_table, class_map

    model = mujoco.MjModel.from_xml_string(_MJCF)
    lut, members = body_class_table(model, "apple")
    name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    by_name = {name(i): hm.CLASS_NAMES[int(lut[i])] for i in range(model.nbody)}
    assert by_name["apple"] == "target"
    assert by_name["pear"] == "other_object"
    assert by_name["crate"] == "destination"
    assert by_name["table"] == "support"
    assert by_name["office"] == "background"
    assert by_name["left_ee_target"] == "background"  # mocap marker
    for g in ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2", "ee_finger_r1"):
        assert by_name[g] == "robot_gripper", g
    for r in ("base", "link_left_arm_6", "link_right_arm_6"):
        assert by_name[r] == "robot", r
    ids = np.array([[-1, 1], [2, 3]])
    cm = class_map(ids, lut)
    assert cm[0, 0] == hm.CLASS_NAMES.index("background")
    with pytest.raises(KeyError):
        body_class_table(model, "banana")


# ------------------------------------------------------------------------------- the CLI


def test_record_table_roles_and_ep1800_location():
    assert hs.SELECTION_KEYS == ("long", "t2", "ep1807")
    assert set(hs.HELDOUT_KEYS) == {"headfix", "ep1850", "ep1900", "ep1800"}
    assert hs.RECORDS["ep1800"].record == "recordings/run_16d_ep1800"
    assert hs.SERVING_CELL == {"denoise": 0, "agg": "last", "layer": 8, "head": 2}


@pytest.mark.parametrize("argv", [
    ["--stage", "gt", "--workers", "5"],
    ["--stage", "gt", "--workers", "0"],
    ["--stage", "select", "--n-lowest", "0"],
    ["--stage", "e2e", "--k", "6"],
    ["--stage", "e2e", "--sigma", "-1"],
    ["--stage", "e2e", "--esdf-backend", "curobo"],
    ["--stage", "select", "--max-frames", "-2"],
    ["--stage", "nope"],
])
def test_bad_arguments_are_refused(argv):
    with pytest.raises(SystemExit):
        hs.validate_args(ap := hs.build_parser(), ap.parse_args(argv))


def test_unknown_record_key_is_refused():
    ap = hs.build_parser()
    with pytest.raises(SystemExit):
        hs.validate_args(ap, ap.parse_args(["--stage", "gt", "--records", "ep9999"]))


def test_cli_refuses_without_run_dir(monkeypatch):
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="AG3S_RUN_DIR"):
        hs.main(["--stage", "select"])


def test_replay_module_imports_without_rendering():
    from AG3S.ag3s.experiments.studies import t47_replay

    assert callable(t47_replay.extract_gt) and callable(t47_replay.run_e2e_record)
    assert t47_replay.GRIPPER_ROOT_LINKS == ("link_left_arm_6", "link_right_arm_6")
