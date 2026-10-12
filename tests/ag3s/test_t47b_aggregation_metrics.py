"""T47b B1 (head aggregation variants) and B2 (label-based 3D metrics) on synthetic data.

Pinned (no MuJoCo, no storage):
  - ``raw_sum`` is T47's `combine_maps` bit for bit; ``mass_norm_joint`` makes every head weigh 1 and
    keeps its camera ratios; ``mass_norm_per_camera`` makes every camera of every head weigh 1;
    zero-mass heads contribute zeros, negative attention is refused; the per-camera head sets and the
    mass-equalised head average;
  - B2 frame metrics against a hand-labelled 12-point cloud (identity plurality, purity, target
    residual and non-target loss at the cluster / seed / ball levels), the membership fallback, and
    record-level aggregation with scene / phase strata (frames never pooled across records).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t47_head_metrics as hm
from AG3S.ag3s.experiments.studies import t47b_aggregation as ag
from AG3S.ag3s.experiments.studies import t47b_metrics as bm


def block(seed=0, f=4, c=3, l=5, h=4, scale=None):
    rng = np.random.default_rng(seed)
    b = rng.random((f, c, l, h, 16, 16))
    if scale is not None:
        b = b * scale
    return b


# ---------------------------------------------------------------------------------- B1


@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_raw_sum_is_t47_combine_maps(sigma):
    b = block()
    cells = [(2, 1), (4, 3), (3, 0)]
    assert np.array_equal(ag.combine_heads(b, cells, sigma, mode="raw_sum"),
                          hm.combine_maps(b, cells, sigma))


@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_mass_norm_joint_equalises_heads_and_keeps_camera_ratios(sigma):
    b = block(1)
    b[:, :, 2, 1] *= 50.0  # one heavy head
    b[:, 0, 3, 2] *= 7.0  # a head whose camera split is uneven
    cells = [(2, 1), (3, 2)]
    # every head contributes total mass 1 over the three cameras -> the sum has mass k
    out = ag.combine_heads(b, cells, sigma, mode="mass_norm_joint")
    np.testing.assert_allclose(out.sum(axis=(-3, -2, -1)), len(cells), rtol=1e-12)
    for cell in cells:
        one = ag.combine_heads(b, [cell], sigma, mode="mass_norm_joint")
        np.testing.assert_allclose(one.sum(axis=(-3, -2, -1)), 1.0, rtol=1e-12)
        raw_cam = b[:, :, cell[0], cell[1]].sum(axis=(-2, -1))
        np.testing.assert_allclose(one.sum(axis=(-2, -1)), raw_cam / raw_cam.sum(-1, keepdims=True),
                                   rtol=1e-12)
    # raw sum is dominated by the heavy head; the joint-normalised sum is not
    raw = ag.combine_heads(b, cells, 0.0, mode="raw_sum")
    assert ag.head_masses(b, cells)["dominance"].min() > 0.85
    assert not np.allclose(raw / raw.sum(axis=(-3, -2, -1), keepdims=True),
                           out / out.sum(axis=(-3, -2, -1), keepdims=True))


@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_mass_norm_per_camera_equalises_cameras(sigma):
    b = block(2)
    b[:, 1] *= 30.0
    cells = [(2, 1), (4, 3), (1, 2)]
    out = ag.combine_heads(b, cells, sigma, mode="mass_norm_per_camera")
    np.testing.assert_allclose(out.sum(axis=(-2, -1)), len(cells), rtol=1e-12)
    np.testing.assert_allclose(ag.camera_mass_share(out), 1.0 / 3.0, rtol=1e-12)


def test_zero_mass_head_contributes_nothing_and_negative_is_refused():
    b = block(3)
    b[:, 2, 2, 1] = 0.0  # cell (2, 1) is blank on camera 2
    cells = [(2, 1), (3, 3)]
    assert ag.zero_mass_counts(b, cells) == {"joint": 0, "per_camera": b.shape[0]}
    out = ag.combine_heads(b, cells, 0.0, mode="mass_norm_per_camera")
    np.testing.assert_allclose(out[:, 2].sum(axis=(-2, -1)), 1.0)  # only (3, 3) on camera 2
    b2 = b.copy()
    b2[:, :, 2, 1] = 0.0
    assert ag.zero_mass_counts(b2, cells)["joint"] == b.shape[0]
    assert np.all(np.isfinite(ag.combine_heads(b2, cells, 1.0, mode="mass_norm_joint")))
    b2[0, 0, 3, 3, 0, 0] = -1e-3
    with pytest.raises(ValueError, match="negative"):
        ag.combine_heads(b2, cells, 0.0, mode="mass_norm_joint")
    with pytest.raises(ValueError):
        ag.combine_heads(b, cells, 0.0, mode="mean")
    with pytest.raises(ValueError):
        ag.combine_heads(b, [], 0.0)


@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_per_camera_head_sets(sigma):
    b = block(4)
    sets = [[(2, 1), (3, 0)], [(4, 3), (1, 1)], [(0, 0), (2, 2)]]
    raw = ag.combine_heads_per_camera(b, sets, sigma, mode="raw_sum")
    for c in range(3):  # T47 `topk_camera`: combine_maps on that camera only
        assert np.array_equal(raw[:, c], hm.combine_maps(b[:, c], sets[c], sigma))
    joint = ag.combine_heads_per_camera(b, sets, sigma, mode="mass_norm_joint")
    for c in range(3):  # each head divided by its mass over all three cameras
        want = sum(hm.gaussian_smooth(b[:, c, l, h] / b[:, :, l, h].sum(axis=(1, 2, 3))[:, None, None],
                                      sigma) for l, h in sets[c])
        np.testing.assert_allclose(joint[:, c], want, rtol=1e-12)
    with pytest.raises(ValueError):
        ag.combine_heads_per_camera(b, sets[:2], sigma)


def test_head_average_variants():
    b = block(5, l=4, h=3)
    assert np.array_equal(ag.head_average_variant(b, 0.0, mode="raw_sum"),
                          hm.head_average(b))  # T47 head_avg, bit for bit
    b[:, :, 3, 2] *= 1000.0  # one head swamps the raw mean
    eq = ag.head_average_variant(b, 0.0, mode="mass_norm_joint")
    n = 2 * 3  # layers 2, 3 x 3 heads (L0, L1 excluded)
    np.testing.assert_allclose(eq.sum(axis=(-3, -2, -1)), 1.0, rtol=1e-12)  # mean of unit masses
    heavy = b[:, :, 3, 2] / b[:, :, 3, 2].sum(axis=(1, 2, 3), keepdims=True)
    raw = ag.head_average_variant(b, 0.0, mode="raw_sum")
    share_raw = (b[:, :, 3, 2].sum(axis=(1, 2, 3)) / n) / raw.sum(axis=(1, 2, 3))
    assert share_raw.min() > 0.95  # raw: the heavy head is ~all of the mass
    want = sum(b[:, :, l, hh] / b[:, :, l, hh].sum(axis=(1, 2, 3), keepdims=True)
               for l in (2, 3) for hh in range(3)) / n
    np.testing.assert_allclose(eq, want, rtol=1e-12)  # every head weighs 1/n, the heavy one too
    np.testing.assert_allclose((heavy / n).sum(axis=(1, 2, 3)), 1.0 / n)
    pc = ag.head_average_variant(b, 1.0, mode="mass_norm_per_camera")
    np.testing.assert_allclose(pc.sum(axis=(-2, -1)), 1.0, rtol=1e-9)


def test_head_masses_and_top_camera():
    b = np.zeros((1, 3, 3, 2, 16, 16))
    b[0, 0, 2, 0, 5, 5] = 3.0
    b[0, 1, 2, 0, 1, 1] = 1.0
    b[0, 2, 2, 1, 0, 0] = 4.0
    m = ag.head_masses(b, [(2, 0), (2, 1)])
    np.testing.assert_allclose(m["joint"], [[4.0, 4.0]])
    np.testing.assert_allclose(m["share_of_sum"], [[0.5, 0.5]])
    np.testing.assert_allclose(m["camera_share"][0, 0], [0.75, 0.25, 0.0])
    np.testing.assert_allclose(m["dominance"], [0.5])
    maps = ag.combine_heads(b, [(2, 0), (2, 1)], 0.0)
    assert ag.top_camera(maps).tolist() == [2]
    norm = ag.combine_heads(b, [(2, 0), (2, 1)], 0.0, mode="mass_norm_per_camera")
    assert ag.top_camera(norm).tolist() == [0]  # every camera's single patch -> 1, tie -> first


# ---------------------------------------------------------------------------------- B2

T, D, S, X, R, O, U = (bm.POINT_LABELS.index(n) for n in
                       ("target", "distractor", "support", "destination", "robot", "other",
                        "unlabelled"))


def test_labels_match_exclusion_e2():
    from AG3S.ag3s.experiments.live.exclusion_e2 import LABELS

    assert tuple(LABELS) == bm.POINT_LABELS
    assert bm.B2_CLASSES == ("target", "other_object", "table", "destination", "robot", "other",
                             "shared", "unlabelled")


def frame_fixture():
    #            0  1  2  3  4  5  6  7  8  9 10 11
    labels = np.array([T, T, T, T, T, D, D, S, X, R, U, T])
    support = np.zeros(12, bool)
    support[[7, 11]] = True  # point 11 is a target point inside the support mask
    chosen = np.zeros(12, bool)
    chosen[[0, 1, 2, 5, 6, 10]] = True  # 3 target, 2 other_object, 1 unlabelled
    seeds = np.array([0, 1, 5])
    ball = np.zeros(12, bool)
    ball[[0, 1, 2, 3, 5, 8]] = True
    return labels, support, chosen, seeds, ball


def test_frame_b2_hand_computed():
    labels, support, chosen, seeds, ball = frame_fixture()
    m = bm.frame_b2(labels, support_mask=support, seed_indices=seeds, chosen_mask=chosen,
                    has_target=True, ball=ball)
    assert m["n_target_obs"] == 5  # 0-4 (11 is support)
    assert m["identity_label"] == "target" and m["identity_correct"] is True
    assert m["chosen_n"] == 6
    assert m["chosen_cluster_purity"] == pytest.approx(3 / 6)
    assert m["chosen__other_object"] == 2 and m["chosen__unlabelled"] == 1
    assert m["target_residual_n"] == 2 and m["target_residual_rate"] == pytest.approx(2 / 5)
    assert m["nontarget_seed_loss"] == 2
    assert m["nontarget_seed_loss__other_object"] == 2
    assert m["nontarget_seed_loss__destination"] == 0
    assert m["seed_n"] == 3 and m["seed_nontarget"] == 1
    assert m["seed_target_residual_n"] == 3
    assert m["seed_target_residual_rate"] == pytest.approx(3 / 5)
    assert m["ball_built"] and m["ball_n"] == 6
    assert m["ball_nontarget"] == 2 and m["ball__destination"] == 1
    assert m["ball__other_object"] == 1
    assert m["ball_target_residual_n"] == 1


def test_identity_plurality_rules():
    labels, support, chosen, seeds, _ = frame_fixture()
    ch = np.zeros(12, bool)
    ch[[0, 5, 6]] = True  # other_object leads 2:1 -> wrong identity even though a target is in it
    m = bm.frame_b2(labels, support_mask=support, seed_indices=seeds, chosen_mask=ch,
                    has_target=True)
    assert m["identity_label"] == "other_object" and not m["identity_correct"]
    ch2 = np.zeros(12, bool)
    ch2[[0, 1, 5, 6, 10]] = True  # 2 target : 2 other (unlabelled ignored) -> tie -> not correct
    m2 = bm.frame_b2(labels, support_mask=support, seed_indices=seeds, chosen_mask=ch2,
                     has_target=True)
    assert m2["identity_label"] is None and not m2["identity_correct"]
    ch3 = np.zeros(12, bool)
    ch3[[0, 5, 8, 9, 10]] = True  # target 1, other 1, dest 1, robot 1 -> tie
    assert bm.frame_b2(labels, support_mask=support, seed_indices=seeds, chosen_mask=ch3,
                       has_target=True)["identity_correct"] is False
    assert bm.plurality({"target": 0, "unlabelled": 9}) is None


def test_no_target_means_everything_stays():
    labels, support, chosen, seeds, _ = frame_fixture()
    m = bm.frame_b2(labels, support_mask=support, seed_indices=seeds, chosen_mask=chosen,
                    has_target=False)
    assert not m["identity_correct"] and math.isnan(m["chosen_cluster_purity"])
    assert m["target_residual_n"] == 5 and m["target_residual_rate"] == 1.0
    assert m["nontarget_seed_loss"] == 0 and not m["ball_built"]
    empty = bm.frame_b2(np.array([D, S]), support_mask=None, seed_indices=None, chosen_mask=None,
                        has_target=False)
    assert empty["n_target_obs"] == 0 and math.isnan(empty["target_residual_rate"])


def test_membership_mask_indices_and_proximity():
    cloud = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]], float)
    m, src = bm.membership_mask(cloud, cloud[[1, 2]], np.array([1, 2]))
    assert src == "indices" and m.tolist() == [False, True, True, False]
    # stale indices (an occluded geometry from another frame) -> proximity
    m, src = bm.membership_mask(cloud, cloud[[1, 2]] + 0.001, np.array([1, 2]), radius=0.005)
    assert src == "proximity" and m.tolist() == [False, True, True, False]
    m, src = bm.membership_mask(cloud, np.zeros((0, 3)), np.zeros(0, int))
    assert src == "empty" and not m.any()
    b = bm.ball_mask(np.array([[0, 0, 0.0], [0, 0, 0.05], [0.2, 0, 0.05]]), [0, 0, 0.04], 0.05, 0.01)
    assert b.tolist() == [False, True, False]


def _row(record, frame, phase, source="s", **kw):
    base = {"record": record, "frame": frame, "phase": phase, "source": source,
            "has_target": True, "on_target": True, "identity_correct": True,
            "chosen_cluster_purity": 1.0, "n_target_obs": 10, "target_residual_n": 0,
            "target_residual_rate": 0.0, "nontarget_seed_loss": 0, "seed_target_residual_n": 0,
            "seed_target_residual_rate": 0.0, "seed_nontarget": 0, "ball_built": False}
    base.update(kw)
    return base


def test_record_level_and_strata():
    rows = [
        # headfix (same scene): 2 frames, one on_target but wrong identity (neighbour absorbed)
        _row("headfix", 0, "approach"),
        _row("headfix", 1, "grasp", identity_correct=False, chosen_cluster_purity=0.3,
             nontarget_seed_loss=7, **{"nontarget_seed_loss__other_object": 7},
             target_residual_n=4, target_residual_rate=0.4),
        # ep1850 (new scene): 4 frames, all fine
        *[_row("ep1850", i, "pre_grasp") for i in range(4)],
    ]
    lvl = bm.record_level(rows[:2])
    assert lvl["on_target_rate"] == 1.0 and lvl["identity_correct_rate"] == 0.5
    assert lvl["on_target_not_identity"] == 1 and lvl["identity_not_on_target"] == 0
    assert lvl["cluster_target_residual_ratio"] == pytest.approx(4 / 20)
    assert lvl["cluster_nontarget_loss__other_object_sum"] == 7
    assert lvl["cluster_frames_with_other_object_loss"] == 1
    assert lvl["identity_failure_rate"] == 0.5 and lvl["silent_failure_rate"] == 0.0
    s = bm.summarize(rows)["s"]
    # record mean, not frame pooling: (0.5 + 1.0) / 2, not 5/6
    assert s["all"]["record_mean"]["identity_correct_rate"] == pytest.approx(0.75)
    assert s["all"]["n_records"] == 2
    assert s["scene=same_scene"]["record_mean"]["identity_correct_rate"] == 0.5
    assert s["scene=new_scene"]["record_mean"]["identity_correct_rate"] == 1.0
    assert s["phase=post_grasp"]["per_record"]["headfix"]["n_frames"] == 1
    assert s["phase=post_grasp"]["record_mean"]["identity_correct_rate"] == 0.0
    assert "ep1850" not in s["phase=post_grasp"]["per_record"]
    assert s["scene=same_scene|phase=pre_grasp"]["record_mean"]["identity_correct_rate"] == 1.0
    assert bm.phase_group("grasp") == "post_grasp" and bm.phase_group("approach") == "pre_grasp"
    assert bm.scene_group("ep1900") == "new_scene" and bm.scene_group("t2") == "selection"


def test_jsonable():
    out = bm.jsonable({"a": np.float32(1.5), "b": [np.int64(2)], "c": float("inf"),
                       "d": np.arange(2)})
    assert out == {"a": 1.5, "b": [2], "c": "inf", "d": [0, 1]}
