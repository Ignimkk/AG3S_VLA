"""IFG-HSR — a closing-kept frame is not an observation for the grasp latch's hand-relative reach.

`clustering.latch_reach_kept_unseen` / `serve_safe --latch-reach-kept-unseen` (needs `--latch-reach-hand-relative`), default off.
IFG-CL3 423 · 831 · 936 (closed loop, hand-relative reach on): while the gripper closes, `closing_keep_geometry` keeps the
pre-closing (on-table) geometry and counts the frame as seen (`visible`, `last_seen_frame == frame`). On the first request after the
policy lifted the fruit, the reach is then measured to the table (123–159 mm) and the attach comes one request late (17–19 cm up).

What this file pins:
1. AG3S: with the flag a closing-kept frame neither re-pins the seen state nor loses it — the pin of the frame that really observed the
   object stands through the kept frames; without the flag the pin moves to the kept frame (T46g as is).
2. `manipulated_kept_this_frame` is False with the flag off, True only for a `closing_kept_geometry` decision with it on.
3. SafePolicy: a visible-but-kept stale centroid after the lift is carried with the hand (attach) with the flag; refused without it.
4. CLI: flag → clustering key; refused without `--latch-reach-hand-relative` and on a `--no-safe` server.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from tests.trajopt.test_grasp_evidence import _finger_origin, _lock
from tests.trajopt.test_t46g_reach_hand_relative import _ident, _poses, _run, _sl5b_frames


def _sp(*, hand=True, kept=None):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.trajopt.safe_policy import SafePolicy, grasp_parent_links
    from tests.trajopt.test_safe_policy import _robot_model, _StubPolicy

    model = _robot_model()
    clustering = {"gripper_max_opening": 0.0715, "latch_reach_hand_relative": hand}
    if kept is not None:
        clustering["latch_reach_kept_unseen"] = kept
    ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf", "clustering": clustering}),
                robot_model=model, constraint_robot_model=model, attached_parent_links=grasp_parent_links())
    return SafePolicy(_StubPolicy(), ag3s=ag3s)


def _set_decision(ag, reason):
    ag._target_confirm.last = types.SimpleNamespace(reason=reason, mode="keep")   # noqa: SLF001


def test_kept_frames_do_not_move_the_pin_with_the_flag():
    from AG3S.ag3s.stages.target_grounding import CLOSING_KEPT_GEOMETRY

    out = {}
    for kept in (False, True):
        sp = _sp(kept=kept); ag = sp.ag3s
        q = [np.full(len(sp.constraint_robot_model.joint_names), 0.01 * k) for k in range(10)]
        c = np.array([0.4, 0.2, 0.8])
        _set_decision(ag, None)
        ag._manipulated = _ident(c, state="visible", frame=4, last_seen_frame=4)   # noqa: SLF001
        ag._note_manipulated_seen(ag._manipulated, q[4])                            # noqa: SLF001
        _set_decision(ag, CLOSING_KEPT_GEOMETRY)
        for k in (5, 6):
            ag._manipulated = _ident(c, state="visible", frame=k, last_seen_frame=k)  # noqa: SLF001
            ag._note_manipulated_seen(ag._manipulated, q[k])                          # noqa: SLF001
        out[kept] = ag.manipulated_seen_state()
    f_off, q_off = out[False]
    f_on, q_on = out[True]
    assert f_off == 6 and np.allclose(q_off, 0.06)          # T46g as is: the kept frame re-pins
    assert f_on == 4 and np.allclose(q_on, 0.04)            # the real observation stands through frames 5–6


def test_kept_this_frame_needs_the_flag_and_the_reason():
    from AG3S.ag3s.stages.target_grounding import CLOSING_KEPT_GEOMETRY, SUBSET_KEPT_ANCHOR

    off, on = _sp(kept=False).ag3s, _sp(kept=True).ag3s
    for ag in (off, on):
        _set_decision(ag, CLOSING_KEPT_GEOMETRY)
    assert off.manipulated_kept_this_frame() is False and on.manipulated_kept_this_frame() is True
    _set_decision(on, SUBSET_KEPT_ANCHOR)
    assert on.manipulated_kept_this_frame() is False
    on._target_confirm.last = None                                                # noqa: SLF001
    assert on.manipulated_kept_this_frame() is False


def test_lifted_visible_kept_stale_centroid_attaches_with_the_flag_only():
    from AG3S.ag3s.stages.target_grounding import CLOSING_KEPT_GEOMETRY

    res = {}
    for kept in (False, True):
        sp = _sp(kept=kept); ag = sp.ag3s
        q_seen, q_now = _poses(sp)
        apple = _finger_origin(sp, q_seen) + np.array([0.05, 0.0, 0.0])
        _set_decision(ag, None)
        seen = _ident(apple, state="visible", frame=4, last_seen_frame=4)
        ag._note_manipulated_seen(seen, q_seen)                                     # noqa: SLF001
        # the closing-kept frame 7, captured after the lift: the stale on-table centroid counts as seen
        _set_decision(ag, CLOSING_KEPT_GEOMETRY)
        kept_ident = _ident(apple, state="visible", frame=7, last_seen_frame=7, geometry=seen.geometry)
        ag._note_manipulated_seen(kept_ident, q_now)                                # noqa: SLF001
        ag._manipulated = kept_ident                                                 # noqa: SLF001
        _lock(sp)
        res[kept] = (sp, _run(sp, _sl5b_frames(q_seen, q_now)))
    sp_off, off = res[False]
    sp_on, on = res[True]
    assert sp_off.ag3s.attached is None
    assert all(r["evidence"]["reach_carry"]["reason"] == "visible" for r in off)
    assert off[2]["evidence"]["checks"]["reach"] is False and off[2]["evidence"]["reach_mm"] > 120.0
    assert [r["attach"] for r in on] == [False, False, True, False, False]
    ev = on[2]["evidence"]
    assert ev["reach_source"] == "hand_relative" and ev["reach_mm"] <= 120.0
    assert ev["reach_carry"]["kept_geometry"] is True and ev["reach_carry"]["seen_frame"] == 4


BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def test_cli_flag_maps_needs_hand_relative_and_the_safe_layer(monkeypatch, capsys):
    from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
    from AG3S.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args([*BASE_ARGV, "--latch-reach-hand-relative", "--latch-reach-kept-unseen"])
    sec = serve_safe.t43t_identity_section(args)
    assert sec["latch_reach_kept_unseen"] is True and sec["latch_reach_hand_relative"] is True
    assert AG3SConfig.from_dict({"clustering": sec}).clustering.latch_reach_kept_unseen is True
    assert "latch_reach_kept_unseen" not in serve_safe.t43t_identity_section(serve_safe.build_parser().parse_args(BASE_ARGV))
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"clustering": {"latch_reach_kept_unseen": "yes"}})
    for extra in (["--latch-reach-kept-unseen"], ["--no-safe", "--latch-reach-hand-relative", "--latch-reach-kept-unseen"]):
        monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, *extra])
        with pytest.raises(SystemExit) as e:
            serve_safe.main()
        assert e.value.code == 2
        assert "--latch-reach-kept-unseen" in capsys.readouterr().err
