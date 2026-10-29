"""IFG-HSR2 — the hand-relative carry starts at the closing start, not at an older pin.

`clustering.latch_carry_from_closing` / `serve_safe --latch-carry-from-closing` (needs `--latch-reach-hand-relative`), default off.
IFG-10EP 1404 (hand-relative reach + kept-unseen): the seen-state pin stayed at request 44 (hand still approaching) while the gripper
closed at 54; the carry moved the on-table centroid with the approach too — 56 mm off, attach on a wrong place, revoked next request.
Before the fingers close the object does not ride the hand, so the carry (and the attach snapshot pose) starts at the request the latch
entered CLOSING when the pin is older.

What this file pins:
1. Flag on, pin older than the closing start: the carry starts at the closing-start pose (record `carry_start = closing_start`,
   `closing_start_frame`, `carry_frame`); flag off: from the pin (T46g as is, no new keys).
2. A pin newer than the closing start (an in-hand observation after closing began) is kept.
3. The closing start is forgotten when the latch leaves CLOSING / HELD.
4. CLI: flag → clustering key; refused without `--latch-reach-hand-relative` and on a `--no-safe` server; config type check.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _lock
from tests.trajopt.test_t46g_reach_hand_relative import HOLD_LINK, LEFT_ARM_1, _ident, _run

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _sp(*, carry=None):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.trajopt.safe_policy import SafePolicy, grasp_parent_links
    from tests.trajopt.test_safe_policy import _robot_model, _StubPolicy

    model = _robot_model()
    clustering = {"gripper_max_opening": 0.0715, "latch_reach_hand_relative": True}
    if carry is not None:
        clustering["latch_carry_from_closing"] = carry
    ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf", "clustering": clustering}),
                robot_model=model, constraint_robot_model=model, attached_parent_links=grasp_parent_links())
    return SafePolicy(_StubPolicy(), ag3s=ag3s)


def _poses(sp):
    """q_far (approach, pin), q_close (fingers at the apple, closing), q_now (lifted)."""
    n = len(sp.constraint_robot_model.joint_names)
    q_close = np.zeros(n)
    q_far, q_now = q_close.copy(), q_close.copy()
    q_far[LEFT_ARM_1] = -0.25
    q_now[LEFT_ARM_1] = 0.35
    return q_far, q_close, q_now


def _episode(sp, *, pin_frame=4, first_frame=6):
    """Pin at q_far (frame `pin_frame`), the apple at the closing hand; open → closing at q_close, then settled after the lift."""
    from tests.trajopt.test_t46g_reach_hand_relative import _ident as ident

    q_far, q_close, q_now = _poses(sp)
    apple = _finger_origin(sp, q_close) + np.array([0.03, 0.0, 0.0])
    seen = ident(apple, state="visible", frame=pin_frame, last_seen_frame=pin_frame)
    sp.ag3s._note_manipulated_seen(seen, q_far)                                       # noqa: SLF001
    sp.ag3s._manipulated = ident(apple, state="occluded", frame=first_frame, last_seen_frame=pin_frame,  # noqa: SLF001
                                 geometry=seen.geometry)
    _lock(sp)
    frames = [(q_close, _fb(1, [1.0] * 8, 1.0)), (q_close, _fb(2, [0.0] * 8, 0.715)),
              (q_now, _fb(3, [0.0] * 8, 0.715)), (q_now, _fb(4, [0.0] * 8, 0.716)), (q_now, _fb(5, [0.0] * 8, 0.716))]
    out = []
    for i, fr in enumerate(frames):
        sp.ag3s.frame_index = first_frame + i
        out += _run(sp, [fr])
    return out, apple, (q_far, q_close, q_now)


def test_carry_starts_at_the_closing_start_with_the_flag():
    from AG3S.ag3s.constraints.attached import carried_point

    res = {c: _episode(_sp(carry=c)) for c in (False, True)}
    off, apple, (q_far, q_close, q_now) = res[False]
    on, _, _ = res[True]
    model = _sp().constraint_robot_model
    want = carried_point(model, q_close, q_now, HOLD_LINK, apple)
    wrong = carried_point(model, q_far, q_now, HOLD_LINK, apple)
    assert np.linalg.norm(want - wrong) > 0.03                       # the approach would move the centre
    assert [r["state"] for r in on][:2] == ["latched", "closing"]
    rc_on = on[2]["evidence"]["reach_carry"]
    assert rc_on["carry_start"] == "closing_start" and rc_on["closing_start_frame"] == 7
    assert rc_on["seen_frame"] == 4 and rc_on["carry_frame"] == 7
    np.testing.assert_allclose(rc_on["centroid_carried_m"], want, atol=1e-5)
    rc_off = off[2]["evidence"]["reach_carry"]
    assert "carry_start" not in rc_off and "closing_start_frame" not in rc_off   # flag off: T46g record
    np.testing.assert_allclose(rc_off["centroid_carried_m"], wrong, atol=1e-5)
    assert any(r["attach"] for r in on)
    att = next(i for i, r in enumerate(on) if r["attach"])
    assert on[att]["evidence"]["reach_carry"]["carry_start"] == "closing_start"


def test_a_pin_newer_than_the_closing_start_is_kept():
    sp = _sp(carry=True)
    out, _, _ = _episode(sp, pin_frame=4, first_frame=2)   # closing entered at frame 3 < pin 4
    rc = out[2]["evidence"]["reach_carry"]
    assert rc["closing_start_frame"] == 3 and rc["carry_start"] == "seen" and rc["carry_frame"] == 4


def test_closing_start_is_forgotten_when_the_latch_leaves_closing():
    from AG3S.trajopt.grasp_latch import GraspPhase

    sp = _sp(carry=True)
    _episode(sp)
    assert sp._closing_start is not None                                              # noqa: SLF001
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    for k in range(6, 9):   # the hand opens for good
        sp.ag3s.frame_index = 20 + k
        _run(sp, [(q, _fb(k, [1.0] * 8, 1.0))])
        if sp._latch.phase not in (GraspPhase.CLOSING, GraspPhase.HELD):                # noqa: SLF001
            break
    assert sp._latch.phase not in (GraspPhase.CLOSING, GraspPhase.HELD)               # noqa: SLF001
    assert sp._closing_start is None                                                  # noqa: SLF001
    sp.reset()
    assert sp._closing_start is None                                                  # noqa: SLF001


def test_cli_flag_maps_needs_hand_relative_and_the_safe_layer(monkeypatch, capsys):
    from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
    from AG3S.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args([*BASE_ARGV, "--latch-reach-hand-relative", "--latch-carry-from-closing"])
    sec = serve_safe.t43t_identity_section(args)
    assert sec["latch_carry_from_closing"] is True
    assert AG3SConfig.from_dict({"clustering": sec}).clustering.latch_carry_from_closing is True
    assert "latch_carry_from_closing" not in serve_safe.t43t_identity_section(serve_safe.build_parser().parse_args(BASE_ARGV))
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"clustering": {"latch_carry_from_closing": "yes"}})
    for extra in (["--latch-carry-from-closing"], ["--no-safe", "--latch-reach-hand-relative", "--latch-carry-from-closing"]):
        monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, *extra])
        with pytest.raises(SystemExit) as e:
            serve_safe.main()
        assert e.value.code == 2
        assert "--latch-carry-from-closing" in capsys.readouterr().err
