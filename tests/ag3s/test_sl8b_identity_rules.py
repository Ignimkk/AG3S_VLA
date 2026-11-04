"""SL8b — loosening the lock on the first-chosen object (task `AG3S/ag3s/docs/handoff/SL8b.task.md` §2), both rules
default off:

  1. window vote (`clustering.target_switch_window` / `target_switch_votes`): a challenger that alternates with the
     held object with other objects takes over; calls without a vote age out instead of resetting; off = the
     consecutive-streak rule of before, with the same decisions and records;
  2. attention release (`clustering.identity_attention_release` …): K pre-grasp calls whose seed attention is off
     the held object release it — to the cluster with >= 50% of the seed mass, else to nothing; not while frozen or
     grasp-latched; a refused `first` until a cluster holds the seeds; off = seeds are not even stored;
  3. config validation, `ground_target` passing the seeds, the serve_safe flags.
"""
from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfigError, ClusteringConfig
from AG3S.ag3s.stages.geometry import fit_sphere
from AG3S.ag3s.stages.target_grounding import (
    ATTENTION_RELEASE,
    ATTENTION_RELEASE_NONE,
    NO_ADMISSIBLE,
    WINDOW_VOTE,
    ClusterInfo,
    TargetConfirm,
)
from AG3S.ag3s.types import TargetGeometry

OPENING = 0.0715
R = 0.031
A = np.array([0.50, 0.00, 0.80])     # the first-chosen object
B = np.array([0.50, -0.40, 0.80])    # the objects the policy actually goes for
C = np.array([0.50, 0.40, 0.80])


def _cap(centre, n=60, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n * 3, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.asarray(centre)[None, :] + R * v[v[:, 2] > 0.2][:n]


PTS = {"A": _cap(A, seed=1), "B": _cap(B, seed=2), "C": _cap(C, seed=3)}


def _frame(scores, seeds=None):
    """`(clusters sorted by score, cloud)` for `{name: score}`; `seeds` = `{name or 'stray': mass}` → the seed
    `(points, attention)` — ten points per name, the stray ones in no cluster."""
    clouds, infos, start = [], [], 0
    for i, (name, score) in enumerate(scores.items()):
        pts = PTS[name]
        clouds.append(pts)
        infos.append(ClusterInfo(
            id=i, point_indices=np.arange(start, start + len(pts)), centroid=pts.mean(axis=0), mean_attention=score,
            max_attention=score, attention_mass=score, point_count=len(pts), spatial_compactness=1.0,
            distance_from_attention_peak=0.0, target_score=score))
        start += len(pts)
    infos.sort(key=lambda c: -c.target_score)
    sp = sa = None
    if seeds is not None:
        sp = np.concatenate([PTS[k][:10] if k != "stray" else np.full((10, 3), 5.0) + np.arange(10)[:, None] * 0.1
                             for k in seeds])
        sa = np.concatenate([np.full(10, m / 10.0) for m in seeds.values()])
    return infos, np.concatenate(clouds), sp, sa


def _confirm(**kw):
    return TargetConfirm(ClusteringConfig(min_points=10, **kw), max_opening=OPENING, overlap_distance=0.005)


def _step(conf, scores, seeds=None, *, observe=True):
    clusters, cloud, sp, sa = _frame(scores, seeds)
    if sp is not None:
        conf.note_seeds(sp, sa)
    d = conf.select(clusters, points=cloud)
    if observe and d.cluster is not None:     # what `ground_target` does after the decision
        pts = cloud[d.cluster.point_indices]
        conf.observe(TargetGeometry(id=0, points=pts, point_indices=d.cluster.point_indices, centroid=pts.mean(axis=0),
                                    bounding_geometry=fit_sphere(pts), attention_score=d.cluster.target_score,
                                    confidence=d.cluster.target_score))
    return d


HELD_LEADS = {"A": 0.9, "B": 0.4}
B_LEADS = {"A": 0.3, "B": 0.8}
LOW = {"A": 0.03, "B": 0.04}      # the challenger leads below target_switch_min_score: no vote


# --------------------------------------------------------------------------------------- 1. window vote
def test_alternating_challenger_switches_by_window_vote():
    conf = _confirm(target_switch_window=5, target_switch_votes=3)
    assert _step(conf, HELD_LEADS).mode == "first"
    modes = [_step(conf, s).mode for s in (B_LEADS, HELD_LEADS, B_LEADS, HELD_LEADS)]
    assert modes == ["hold", "keep", "hold", "keep"]
    d = _step(conf, B_LEADS)                       # B led calls 1, 3, 5 of the last five
    assert d.mode == "switch" and d.reason == WINDOW_VOTE and d.manipulated_id == 1
    assert d.streak == 0 and d.frames == 3 and d.cluster.centroid == pytest.approx(PTS["B"].mean(axis=0))


def test_the_consecutive_rule_never_switches_on_the_same_alternation():
    conf = _confirm()                              # off
    _step(conf, HELD_LEADS)
    for s in (B_LEADS, HELD_LEADS, B_LEADS, HELD_LEADS, B_LEADS, HELD_LEADS, B_LEADS):
        assert _step(conf, s).mode in ("hold", "keep")
    assert conf.manipulated.id == 0
    # … and three in a row switch, with no reason and the old record
    modes = [_step(conf, B_LEADS) for _ in range(2)]      # the last alternation call was already the first of the run
    assert [d.mode for d in modes] == ["hold", "switch"] and modes[-1].reason is None
    assert conf.identity_record is None and conf._votes == []


def test_calls_without_a_vote_age_out_and_do_not_reset():
    conf = _confirm(target_switch_window=5, target_switch_votes=3)
    _step(conf, HELD_LEADS)
    assert _step(conf, B_LEADS).mode == "hold"
    assert _step(conf, LOW).mode == "hold"          # leader below the minimum score: a call without a vote …
    assert _step(conf, B_LEADS).mode == "hold"      # … the vote of call 1 still counts: 2 votes
    assert _step(conf, HELD_LEADS).mode == "keep"
    assert _step(conf, B_LEADS).mode == "switch"    # calls 1, 3, 5 (+ the aged ones in between)
    # votes older than W calls are gone: B, A, A, A, A, B, B → the first vote has left the window
    conf = _confirm(target_switch_window=5, target_switch_votes=3)
    _step(conf, HELD_LEADS)
    assert [_step(conf, s).mode for s in (B_LEADS, HELD_LEADS, HELD_LEADS, HELD_LEADS, HELD_LEADS, B_LEADS, B_LEADS)
            ] == ["hold", "keep", "keep", "keep", "keep", "hold", "hold"]


def test_votes_for_different_places_do_not_add_up_and_a_switch_clears_the_window():
    conf = _confirm(target_switch_window=5, target_switch_votes=3)
    _step(conf, {"A": 0.9, "B": 0.4, "C": 0.4})
    for s in ({"A": 0.3, "B": 0.8, "C": 0.2}, {"A": 0.3, "B": 0.2, "C": 0.8}, {"A": 0.3, "B": 0.8, "C": 0.2},
              {"A": 0.3, "B": 0.2, "C": 0.8}):
        assert _step(conf, s).mode == "hold"        # two votes each
    assert _step(conf, {"A": 0.3, "B": 0.8, "C": 0.2}).mode == "switch"
    assert conf._votes == [] and conf.manipulated.id == 1
    assert _step(conf, {"A": 0.3, "B": 0.9, "C": 0.2}).mode == "keep"


def test_window_vote_config_checks():
    assert ClusteringConfig().target_switch_window is None and ClusteringConfig().target_switch_votes == 3
    ClusteringConfig(target_switch_window=5, target_switch_votes=3).validate()
    for bad in ({"target_switch_window": 0}, {"target_switch_window": 2.5}, {"target_switch_window": True},
                {"target_switch_window": 2}, {"target_switch_window": 5, "target_switch_votes": 6},
                {"target_switch_window": 5, "target_switch_votes": 0}):
        with pytest.raises(AG3SConfigError):
            ClusteringConfig(**bad).validate()


# ---------------------------------------------------------------------------- 2. attention release
RELEASE = dict(identity_attention_release=4, identity_attention_off_frac=0.7, identity_attention_radius=0.015)
ON_A = {"A": 1.0}
ON_B = {"B": 1.0}


def _adopted(**kw):
    conf = _confirm(**{**RELEASE, **kw})
    assert _step(conf, HELD_LEADS, ON_A).mode == "first"
    return conf


def test_seeds_off_the_held_object_for_k_calls_release_to_the_seed_cluster():
    conf = _adopted()
    for _ in range(3):
        d = _step(conf, HELD_LEADS, ON_B)           # A still leads the ranking: the consecutive rule would hold
        assert d.mode == "keep" and conf.identity_record["off_share"] == pytest.approx(1.0)
    d = _step(conf, HELD_LEADS, ON_B)
    assert d.mode == "switch" and d.reason == ATTENTION_RELEASE and d.manipulated_id == 1 and d.rank == 1
    assert d.cluster.centroid == pytest.approx(PTS["B"].mean(axis=0))
    assert conf.identity_record["decision"] == "switch" and conf.identity_record["best_share"] == pytest.approx(1.0)
    assert _step(conf, HELD_LEADS, ON_A).mode in ("hold", "keep")   # the new object holds; a fresh count starts


def test_off_does_not_store_the_seeds_nor_release():
    conf = _confirm()                                   # attention release off
    _step(conf, HELD_LEADS, ON_A)
    for _ in range(8):
        assert _step(conf, HELD_LEADS, ON_B).mode == "keep"
    assert conf._seeds is None and conf.identity_record is None and conf.manipulated.id == 0
    assert not conf.identity_rules_on


def test_the_mean_of_the_last_k_shares_decides_not_a_single_call():
    conf = _adopted()
    on_a = {"A": 0.5, "B": 0.5}                      # half of the mass on the held object: off share 0.5
    for s in (ON_B, ON_B, on_a, ON_B):               # mean 0.875 ≥ 0.7 at call 4 → release
        d = _step(conf, HELD_LEADS, s)
    assert d.mode == "switch"
    conf = _adopted()
    for s in (ON_B, on_a, on_a, ON_B):               # mean 0.75 → still release; use 0.6 below
        d = _step(conf, HELD_LEADS, s)
    assert d.mode == "switch"
    conf = _adopted()
    for s in (ON_B, on_a, ON_A, on_a, ON_B):         # window [0.5, 0, 0.5, 1]: mean 0.5 < 0.7
        d = _step(conf, HELD_LEADS, s)
    assert d.mode == "keep"


@pytest.mark.parametrize("how", ["frozen", "latched"])
def test_no_release_while_frozen_or_latched_and_the_count_restarts_after(how):
    conf = _adopted()
    for _ in range(3):
        _step(conf, HELD_LEADS, ON_B)
    for _ in range(6):                                # a grasp is in progress: nothing counts, old evidence is dropped
        conf.freeze(how == "frozen")
        conf.note_latched(how == "latched")
        d = _step(conf, HELD_LEADS, ON_B)
        assert d.mode == "keep" and conf.identity_record["why"] == how
    conf.freeze(False)
    conf.note_latched(False)
    assert [_step(conf, HELD_LEADS, ON_B).mode for _ in range(3)] == ["keep"] * 3   # 3 < K again
    assert _step(conf, HELD_LEADS, ON_B).mode == "switch"


def test_less_than_half_of_the_seed_mass_in_one_cluster_holds_nothing_and_refuses_a_weak_first():
    conf = _adopted()
    scattered = {"B": 0.4, "C": 0.3, "stray": 0.3}    # largest admissible share 0.4 < 0.5
    three = {"A": 0.9, "B": 0.4, "C": 0.3}
    for _ in range(3):
        assert _step(conf, three, scattered).mode == "keep"
    d = _step(conf, three, scattered)
    assert d.mode == NO_ADMISSIBLE and d.reason == ATTENTION_RELEASE_NONE and d.cluster is None
    assert conf.manipulated is None and conf.identity_record["decision"] == "none"
    # until the next adoption a `first` needs a cluster with half of the seed mass
    d = _step(conf, three, scattered)
    assert d.mode == NO_ADMISSIBLE and d.reason == ATTENTION_RELEASE_NONE and conf.manipulated is None
    d = _step(conf, three, None)                       # no seeds at all: nothing to hold on to
    assert d.mode == NO_ADMISSIBLE and conf.manipulated is None
    d = _step(conf, three, {"C": 0.6, "stray": 0.4})   # the seeds are on C, though A leads the ranking
    assert d.mode == "first" and d.reason == ATTENTION_RELEASE and d.manipulated_id == 1
    assert d.cluster.centroid == pytest.approx(PTS["C"].mean(axis=0))
    # the condition is gone with the adoption: the next release starts from a normal state
    assert conf.manipulated is not None and not conf._attn_pending


def test_a_release_whose_seed_cluster_is_the_held_object_changes_nothing():
    conf = _adopted()
    for _ in range(3):
        _step(conf, HELD_LEADS, ON_B, observe=True)
    # the held object's stored geometry is stale (it was pushed): the seeds are on A's own, moved, cluster
    PTS_A = PTS["A"]
    PTS["A"] = _cap(A + np.array([0.0, 0.03, 0.0]), seed=1)
    try:
        conf._attn_shares = []
        d = None
        for _ in range(4):
            d = _step(conf, HELD_LEADS, ON_A, observe=False)
        assert d.mode == "keep" and conf.manipulated.id == 0
        assert conf.identity_record["decision"] == "same_object"
    finally:
        PTS["A"] = PTS_A


def test_the_seed_membership_by_cloud_index_matches_the_coordinate_match():
    for use_index in (False, True):
        conf = _adopted()
        for _ in range(4):
            clusters, cloud, sp, sa = _frame(HELD_LEADS, ON_B)
            idx = None
            if use_index:
                idx = np.arange(len(PTS["A"]), len(PTS["A"]) + 10)     # the first ten points of B in the cloud
            conf.note_seeds(sp, sa, idx)
            d = conf.select(clusters, points=cloud)
            if d.cluster is not None and d.mode != "switch":
                pts = cloud[d.cluster.point_indices]
                conf.observe(TargetGeometry(id=0, points=pts, point_indices=d.cluster.point_indices,
                                            centroid=pts.mean(axis=0), bounding_geometry=fit_sphere(pts),
                                            attention_score=0.9, confidence=0.9))
        assert d.mode == "switch" and d.reason == ATTENTION_RELEASE


def test_attention_release_config_checks():
    c = ClusteringConfig()
    assert (c.identity_attention_release, c.identity_attention_off_frac, c.identity_attention_radius) == (None, 0.7, 0.015)
    ClusteringConfig(**RELEASE).validate()
    for bad in ({"identity_attention_release": 0}, {"identity_attention_release": 1.5},
                {"identity_attention_off_frac": 0.0}, {"identity_attention_off_frac": 1.2},
                {"identity_attention_radius": 0.0}, {"identity_attention_radius": float("nan")}):
        with pytest.raises(AG3SConfigError):
            ClusteringConfig(**bad).validate()


# ------------------------------------------------------------------------------------ 3. ground_target
def test_ground_target_passes_the_seeds_and_the_release_ends_in_a_target():
    """`ground_target` hands this frame's seed points / attention to the confirm: the attention peak sits on B while
    A (held first) is the only object the old score would keep."""
    from AG3S.ag3s.stages.target_grounding import ground_target
    from AG3S.ag3s.types import AttentionPointCloud, PointCloud

    def cloud_for(att_a, att_b):
        pts = np.concatenate([PTS["A"], PTS["B"]])
        att = np.concatenate([np.full(len(PTS["A"]), att_a), np.full(len(PTS["B"]), att_b)])
        return AttentionPointCloud(PointCloud(points=pts, frame_id="base"), att.astype(np.float64))

    # the consecutive rule is out of the way (50 frames), so only the release can move the target
    cfg = ClusteringConfig(min_points=10, target_confirm_frames=50, **RELEASE)
    conf = TargetConfirm(cfg, max_opening=OPENING, overlap_distance=0.005)
    first = ground_target(cloud_for(1.0, 0.1), cfg, seed_percentile=50.0, confirm=conf)
    assert first.target is not None and conf.manipulated.id == 0
    results = [ground_target(cloud_for(0.05, 1.0), cfg, seed_percentile=50.0, confirm=conf) for _ in range(4)]
    assert conf.manipulated.id == 1
    final = results[-1]
    assert [r.target.metrics["identity_attention_release"] for r in results] == [0.0, 0.0, 0.0, 1.0]
    assert final.target is not None and final.target.centroid == pytest.approx(PTS["B"].mean(axis=0), abs=1e-6)
    # default off: no new metric keys on the target
    off = TargetConfirm(ClusteringConfig(min_points=10), max_opening=OPENING, overlap_distance=0.005)
    r = ground_target(cloud_for(1.0, 0.1), ClusteringConfig(min_points=10), seed_percentile=50.0, confirm=off)
    assert not any(k.startswith("identity_") for k in r.target.metrics)


# ---------------------------------------------------------------------------------- 4. serve_safe flags
BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _parse(*a):
    from AG3S.trajopt import serve_safe

    return serve_safe.build_parser().parse_args([*BASE_ARGV, *a])


def test_flags_map_to_the_clustering_keys_and_default_to_nothing():
    from AG3S.trajopt import serve_safe as ss
    from AG3S.ag3s.config import AG3SConfig

    assert ss.identity_rules_section(_parse()) == {}
    assert not any(k.startswith(("target_switch", "identity_")) for k in ss.t43t_identity_section(_parse()))
    sec = ss.t43t_identity_section(_parse("--identity-switch-window", "5", "--identity-switch-votes", "3"))
    assert sec["target_switch_window"] == 5 and sec["target_switch_votes"] == 3
    assert "identity_attention_release" not in sec
    sec = ss.t43t_identity_section(_parse(
        "--identity-attention-release", "4", "--identity-attention-off-frac", "0.7", "--identity-attention-radius-mm", "15"))
    assert sec["identity_attention_release"] == 4 and sec["identity_attention_off_frac"] == 0.7
    assert sec["identity_attention_radius"] == pytest.approx(0.015) and "target_switch_window" not in sec
    cfg = AG3SConfig.from_dict({"clustering": sec}).clustering
    assert all(getattr(cfg, k) == v for k, v in sec.items())


def test_a_threshold_without_its_rule_and_bad_values_are_refused():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    for argv in (["--identity-switch-votes", "3"], ["--identity-attention-off-frac", "0.5"],
                 ["--identity-attention-radius-mm", "10"], ["--identity-switch-window", "2", "--identity-switch-votes", "3"],
                 ["--identity-attention-release", "0"], ["--identity-attention-release", "4", "--identity-attention-off-frac", "0"]):
        with pytest.raises(SystemExit):
            ss.reject_identity_rules_flags(ap, ap.parse_args([*BASE_ARGV, *argv]))
    ss.reject_identity_rules_flags(ap, ap.parse_args(BASE_ARGV))
    ss.reject_identity_rules_flags(ap, ap.parse_args([*BASE_ARGV, "--identity-switch-window", "5"]))


def test_help_renders_with_the_new_flags():
    from AG3S.trajopt import serve_safe as ss

    text = ss.build_parser().format_help()
    assert "--identity-switch-window" in text and "--identity-attention-release" in text
