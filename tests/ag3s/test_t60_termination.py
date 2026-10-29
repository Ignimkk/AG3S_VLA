import numpy as np

from AG3S.ag3s.experiments.studies import t60_termination as T


def _series(n=40, place_at=10, home_at=25, contact_until=12, start_ready=True):
    inc = np.zeros(n + 1, bool); inc[place_at:] = True
    ht = np.zeros(n, bool); ht[:contact_until + 1] = True
    speed = np.zeros(n); speed[place_at:place_at + 2] = 0.2
    q_err = np.full(n, 1.0); q_err[home_at:] = 0.05
    if start_ready:
        q_err[:3] = 0.0
    tcp_err = np.where(q_err < 0.35, 0.01, 0.3)
    return {"in_crate": inc, "hand_target": ht, "speed": speed, "q_err": q_err, "tcp_err": tcp_err}


def test_first_dwell_counts_consecutive_rows():
    ok = np.array([1, 1, 0, 1, 1, 1, 0], bool)
    assert T.first_dwell(ok, 1) == 0
    assert T.first_dwell(ok, 2) == 1
    assert T.first_dwell(ok, 3) == 5
    assert T.first_dwell(ok, 4) is None
    assert T.first_dwell(ok, 2, start=2) == 4


def test_rule_p_needs_release_and_settle():
    s = _series()
    # in crate from 10, contact until 12, speed high at 10-11 -> first ok row 13, dwell 8 -> 20
    e = T.endpoints(s, T.Params(dwell_p=8))
    assert e["t_P"] == 20
    assert T.endpoints(s, T.Params(dwell_p=1))["t_P"] == 13


def test_rule_h_never_ends_before_p_even_if_start_is_ready():
    s = _series(start_ready=True)
    e = T.endpoints(s, T.Params(dwell_p=8, dwell_h=8))
    assert e["t_H"] == 32  # q_err small from 25, dwell 8 -> 32; rows 0-2 (ready at start) ignored
    assert e["t_H"] > e["t_P"]


def test_no_placement_keeps_full_horizon():
    s = _series(place_at=41)
    e = T.endpoints(s, T.Params())
    assert e == {"t_P": None, "t_H": None}
    assert T.keep_rows(40, None).all()
    assert T.keep_frames(np.arange(0, 40, 8), None).all()


def test_keep_masks_cut_after_end():
    keep = T.keep_rows(10, 4)
    assert keep.tolist() == [True] * 5 + [False] * 5
    frames = T.keep_frames(np.array([0, 8, 16, 24]), 16)
    assert frames.tolist() == [True, True, True, False]


def test_in_crate_matches_centre_rule():
    crate = np.array([[0.5, 0.0, 0.9]]); q = np.array([[1.0, 0, 0, 0]])
    assert T.in_crate(np.array([[0.5, 0.0, 0.9]]), crate, q).tolist() == [True]
    assert T.in_crate(np.array([[0.5 + 0.08, 0.0, 0.9]]), crate, q).tolist() == [False]
    assert T.in_crate(np.array([[0.5, 0.0, 0.9 + 0.08]]), crate, q).tolist() == [False]


def test_sampled_endpoints_need_two_consecutive_samples():
    t = np.arange(0, 80, 8)
    inc = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], bool)
    ht = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    sp = np.zeros(10)
    q = np.array([0, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1])
    tcp = np.where(q < 0.35, 0.01, 0.3)
    e = T.endpoints_sampled(t, inc, ht, sp, q, tcp, T.Params())
    assert e["t_P"] == 24 + 7  # samples 3 and 4 (t 24, 32) both ok
    assert e["t_H"] == 40 + 7  # first sample after t_P with the next one also ok
    assert T.keep_frames(t, e["t_P"]).sum() == 4
