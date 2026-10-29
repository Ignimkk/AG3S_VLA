"""T56 self scores S0 / S1 / S2 and the label GT check on a synthetic geometry (no MuJoCo)."""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t56_held as hd
from AG3S.ag3s.experiments.studies import t56_replay as r56
from AG3S.ag3s.experiments.studies import t56_selfscore as ss

TARGET, OTHER, SUPPORT = 0, 1, 2  # `t47b_metrics.POINT_LABELS` indices


def _ball(centre, n, r, rng):
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.asarray(centre) + d * r * rng.uniform(0.2, 1.0, size=(n, 1))


def make_geom(n_f=10, *, held_pos=6, empty=False, hand_path=None, carry=True, seed=0):
    """A record: target fruit at T, neighbour at N, table points; the left hand closes on T at
    `held_pos` and then (if `hand_path`) moves; the fruit moves with it when `carry`.
    Candidates: ``good`` = target points, ``big`` = target + neighbour + more, ``nbr`` = neighbour,
    ``none`` = no output."""
    rng = np.random.default_rng(seed)
    T = np.array([0.5, 0.3, 1.0])
    N = np.array([0.5, 0.45, 1.0])
    hand0 = T if not empty else T + np.array([0.0, -0.12, 0.0])  # empty grasp: closes beside the fruit
    path = hand_path or [np.zeros(3)] * n_f
    t_pts = _ball(T, 40, 0.03, rng)
    n_pts = _ball(N, 40, 0.03, rng)
    table = np.c_[rng.uniform(0.3, 0.7, 200), rng.uniform(0.1, 0.6, 200), np.full(200, 0.96)]
    z: dict = {k: [] for k in ("cloud_pts", "cloud_support", "cloud_labels")}
    off, dig, cfr = [0], [], []
    idx_parts, idx_off = [], [0]
    pads, Rs = [], []
    names = ["big", "good", "nbr", "none"]
    calls = {nm: [] for nm in names}
    for f in range(n_f):
        shift = path[f] if f >= held_pos else np.zeros(3)
        tp = t_pts + (shift if carry else 0)
        pts = np.r_[tp, n_pts, table]
        lab = np.r_[np.full(40, TARGET), np.full(40, OTHER), np.full(200, SUPPORT)].astype(np.int8)
        sup = np.r_[np.zeros(80, bool), np.ones(200, bool)]
        z["cloud_pts"].append(pts.astype(np.float32))
        z["cloud_support"].append(sup)
        z["cloud_labels"].append(lab)
        off.append(off[-1] + len(pts))
        dig.append(f"d{f:03d}")
        cfr.append(f)
        hc = hand0 + shift
        pads.append([[hc + [0, 0.04, 0], hc - [0, 0.04, 0]], [hc + [0, 0.5, 0], hc + [0, 0.42, 0]]])
        Rs.append([np.eye(3), np.eye(3)])
        sel = {"good": np.arange(40), "big": np.arange(80), "nbr": np.arange(40, 80),
               "none": np.zeros(0, int)}
        for nm in names:
            calls[nm].append(sel[nm])
    cc = np.zeros((4, n_f), np.int32)
    ch = np.zeros((4, n_f), bool)
    cm = np.zeros((4, n_f), np.int8)
    for a, nm in enumerate(names):
        for f in range(n_f):
            cc[a, f] = f
            ch[a, f] = nm != "none"
            cm[a, f] = 1 if nm != "none" else 0
            idx_parts.append(calls[nm][f].astype(np.int32))
            idx_off.append(idx_off[-1] + len(calls[nm][f]))
    both = np.full(n_f, -1, np.int8)
    held = np.full(n_f, -1, np.int8)
    if not empty:
        both[held_pos - 1:] = 0
        held[held_pos:] = 0
    arr = {
        "frames": np.arange(n_f), "t_step": np.arange(n_f) * 8, "cand": np.asarray(names),
        "cloud_pts": np.concatenate(z["cloud_pts"]), "cloud_support": np.concatenate(z["cloud_support"]),
        "cloud_labels": np.concatenate(z["cloud_labels"]), "cloud_off": np.asarray(off),
        "cloud_digest": np.asarray(dig), "cloud_frame": np.asarray(cfr),
        "call_cloud": cc, "call_has": ch, "call_member": cm,
        "chosen_idx": np.concatenate(idx_parts), "chosen_off": np.asarray(idx_off),
        "tpts": np.zeros((0, 3), np.float32), "tpts_off": np.zeros(4 * n_f + 1, np.int64),
        "T_world_base": np.tile(np.eye(4), (n_f, 1, 1)), "pads_world": np.asarray(pads),
        "finger_R_world": np.asarray(Rs), "parent_world": np.asarray(pads)[:, :, 0],
        "both_hand": both, "held_hand": held, "target_world": np.tile(T, (n_f, 1)),
        "pad_half": np.array([0.005, 0.005]), "state": np.zeros((n_f, 16)),
        "actions": np.zeros((n_f, 8, 16)),
    }
    return r56.Geometry(arr), names


def held_event(pos, n_f, *, states=None):
    st = states or (["latched"] * pos + ["held"] * (n_f - pos))
    return hd.HeldEvent("server_chunk", "left", pos, st, frames=list(range(n_f)))


def test_combine_penalises_a_large_cluster_and_a_tiny_one():
    inc = np.array([1.0, 0.5, 1.0, 0.0])
    cov = np.array([1.0, 1.0, 0.1, 0.0])
    h = ss.combine(inc, cov, "harmonic")
    assert h[0] == pytest.approx(1.0) and h[1] == pytest.approx(2 / 3) and h[3] == 0.0
    assert ss.combine(inc, cov, "min")[2] == pytest.approx(0.1)
    assert ss.normalize(np.array([[0.5, np.nan], [0.25, np.nan]]), "frame_max")[1, 0] == pytest.approx(0.5)


def test_s1_labels_frames_before_held_and_ranks_by_inclusion_and_coverage():
    g, names = make_geom()
    ev = held_event(6, 10)
    p1, p2 = ss.S1Params(), ss.S2Params()
    fs = ss.frame_scores(g, names, ev, p1, p2)
    l1 = fs["labels"]["S1"]
    assert l1["event_frame"] == 6 and l1["labeled"] == list(range(6))
    assert np.allclose(l1["centre_world"], [0.5, 0.3, 1.0])
    ep = ss.episode_scores(fs)
    s1 = dict(zip(names, ep["S1"]))
    assert s1["good"] > s1["big"] > s1["nbr"] == 0.0
    assert s1["none"] == 0.0
    cov = dict(zip(names, ep["S1_coverage"]))
    assert cov["big"] == pytest.approx(cov["good"]) == pytest.approx(1.0)
    inc = dict(zip(names, ep["S1_inclusion"]))
    assert inc["big"] == pytest.approx(0.5)
    assert np.isnan(fs["frame"]["S1"][:, 6:]).all()


def test_s0_is_t51_both_contact_self_identity():
    g, names = make_geom()
    fs = ss.frame_scores(g, names, held_event(6, 10), ss.S1Params(), ss.S2Params())
    l0 = fs["labels"]["S0"]
    assert l0["event_frame"] == 5 and l0["labeled"] == list(range(5))
    ep = ss.episode_scores(fs)
    s0 = dict(zip(names, ep["S0"]))
    # big: half inside the ball → not a strict majority → 0 (the T51 saturation / binary rule)
    assert s0["good"] == 1.0 and s0["big"] == 0.0 and s0["none"] == 0.0


def test_no_held_event_gives_no_s1_s2_score():
    g, names = make_geom()
    ev = hd.HeldEvent("offline_latch", None, None, ["latched"] * 10, frames=list(range(10)))
    fs = ss.frame_scores(g, names, ev, ss.S1Params(), ss.S2Params())
    ep = ss.episode_scores(fs)
    assert np.isnan(ep["S1"]).all() and np.isnan(ep["S2"]).all()
    assert np.isfinite(ep["S0"]).all()  # S0 has its own (sim) event


def _moving(n_f=10, held_pos=6, step=0.03):
    return [np.array([0.0, 0.0, step * max(0, f - held_pos)]) for f in range(n_f)]


def test_s2_extends_only_with_hand_motion_and_comoving_points():
    g, names = make_geom(hand_path=_moving())
    p2 = ss.S2Params(min_hand_motion=0.05)
    ext = ss.s2_extension(g, held_event(6, 10), ss.s1_label(g, held_event(6, 10), ss.S1Params()), p2)
    # motion 0.03 at frame 7 (< 0.05) → skipped; 0.06+ from frame 8 on → labeled
    assert sorted(ext["frames"]) == [8, 9]
    assert ext["trace"][0]["skip"] == "no_hand_motion"
    fs = ss.frame_scores(g, names, held_event(6, 10), ss.S1Params(), p2)
    assert fs["s2_frames"] == [8, 9]
    assert np.isfinite(fs["frame"]["S2"][:, 8]).all() and np.isnan(fs["frame"]["S1"][:, 8]).all()


def test_s2_stops_after_release_or_slip_and_does_not_restart():
    g, names = make_geom(hand_path=_moving(step=0.05))
    states = ["latched"] * 6 + ["held", "held", "placed", "held"]  # slip / release at 8, HELD again at 9
    ev = held_event(6, 10, states=states)
    ext = ss.s2_extension(g, ev, ss.s1_label(g, ev, ss.S1Params()), ss.S2Params())
    assert sorted(ext["frames"]) == [7]
    assert ext["stopped"] == {"pos": 8, "reason": "latch_placed"}


def test_s2_hand_proximity_alone_is_not_a_positive_label():
    # the hand moves away but the fruit stays on the table (not carried): nothing co-moves
    g, names = make_geom(hand_path=_moving(step=0.1), carry=False)
    ev = held_event(6, 10)
    ext = ss.s2_extension(g, ev, ss.s1_label(g, ev, ss.S1Params()), ss.S2Params())
    assert ext["frames"] == {}
    assert ext["stopped"]["reason"] == "not_following"


def test_held_is_not_certification_empty_grasp_label_is_counted_wrong():
    from AG3S.ag3s.experiments.studies import t56_study as st

    g, names = make_geom(empty=True)
    ev = held_event(6, 10)  # HELD reported although the hand closed beside the fruit
    p1 = ss.S1Params()
    lab = ss.s1_label(g, ev, p1)
    chk = ss.label_check(g, lab)
    assert chk["has_label"] and not chk["event_label_correct"]
    assert chk["event_verdict"] in ("empty_or_tied", "wrong_object")
    # a correct grasp in another record → error rate 1 / 2 in the aggregate
    g2, _ = make_geom()
    rd1 = st.RecordData("empty", ev, {("current", "cell"): g}, {}, None, {"cell": names}, {}, {})
    rd2 = st.RecordData("ok", held_event(6, 10), {("current", "cell"): g2}, {}, None, {"cell": names}, {}, {})
    fs = {rd.key: st.score_record(rd, "current", "cell", p1, ss.S2Params()) for rd in (rd1, rd2)}

    class A:
        set = "dev"

    agg = st.label_checks([rd1, rd2], "current", "cell", p1, ss.S2Params(), fs, A())
    assert agg["n_held_events"] == 2 and agg["n_label_wrong"] == 1
    assert agg["held_label_error_rate"] == pytest.approx(0.5) and agg["wrong_records"] == ["empty"]


def test_unscorable_frame_when_the_ball_holds_no_point():
    g, names = make_geom()
    c = np.array([5.0, 5.0, 5.0])
    inc, cov = ss.ball_components(g, "good", 0, c, 0.05, 0.005)
    assert np.isnan(inc) and np.isnan(cov)
