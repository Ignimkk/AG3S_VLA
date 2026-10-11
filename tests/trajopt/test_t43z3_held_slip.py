"""T43 Z3 — a held object that slips out of the **closed** hand is detached (`LatchConfig.slip_detach`,
`clustering.held_slip_detach`, `serve_safe --held-slip-detach`; default off).

Rule (`GraspLatch._slip_check`): while HELD and nothing else answers (T34 revoke window · open release),
`attach opening − opening ≥ slip_drop (0.05)` and `opening − last executed command < slip_gap (0.05)` on
`slip_frames (2)` consecutive requests → detach, PLACED, `LatchEvent.slipped`. Synthetic sequences pin
each piece; the T43 W record replay is at the bottom (skipped when the records are absent).

Shapes are T43 W V5 1925 s19253's (server records): attach at opening 0.731 with the closing command
0.587, carry at 0.729 under ~0.59, HOLD from t 216 with the opening creeping 0.727 → 0.708 while the
apple slid, then 0.596 · 0.593 (the apple left at t 369; fingers at the held command 0.593).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.trajopt.grasp_latch import GraspPhase, LatchConfig
from tests.trajopt import _t43z3_replay as R
from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _grasp_frames, _latched, _lock, \
    _manipulated, _signal

HOLD = [False] * 8


def _attach(latch, opening=0.731, command=0.587):
    for seq, applied, meas in ((1, [1.0] * 8, 0.993), (2, [0.002] * 8, opening - 0.007),
                               (3, [command] * 8, opening)):
        e = latch.observe_grasp(_signal(seq, applied, meas), reach_m=0.047, manipulated_id=0,
                                manipulated_state="visible")
    assert e.attach and latch.phase is GraspPhase.HELD
    return e


def _hold(latch, seq, applied, meas, executed=None, placed=False):
    return latch.observe_grasp(_signal(seq, applied, meas, executed=executed), reach_m=0.25,
                               manipulated_id=0, manipulated_state="occluded", placed=placed)


def _carry(latch, n=6, start=4):
    """Executed carry chunks past the T34 window (4 executed chunks), opening steady at 0.729."""
    for k in range(start, start + n):
        e = _hold(latch, k, [0.59] * 8, 0.729)
        assert not e.detach
    return start + n


def _v5_1925(latch):
    """attach → carry → HOLD creeping 0.727 → 0.708 → slip 0.596 → 0.593 (HOLD applied 0.593)."""
    _attach(latch)
    k = _carry(latch)
    out = []
    for meas in (0.727, 0.722, 0.716, 0.712, 0.708, 0.596, 0.593, 0.593):
        out.append(_hold(latch, k, [0.593] * 8, meas, executed=HOLD))
        k += 1
    return out


# --- 1. default off: Z3 changes nothing -----------------------------------------------------


def test_off_by_default_and_the_slip_never_detaches():
    assert LatchConfig().slip_detach is False
    latch = _latched()
    evs = _v5_1925(latch)
    assert not any(e.detach for e in evs) and latch.phase is GraspPhase.HELD
    assert all("slip" not in (e.evidence or {}) for e in evs)        # the record is unchanged
    assert all(e.slipped is False for e in evs)


def test_off_records_equal_the_pre_z3_records_field_for_field():
    """Same inputs, flag off vs the slip thresholds changed: identical events (the thresholds are inert)."""
    a, b = _latched(), _latched(slip_drop=0.0, slip_gap=9.0, slip_frames=1)
    assert [dataclasses.asdict(e) for e in _v5_1925(a)] == [dataclasses.asdict(e) for e in _v5_1925(b)]


# --- 2. the rule ------------------------------------------------------------------------------


def test_v5_1925_slip_detaches_on_the_second_closed_request_and_goes_placed():
    latch = _latched(slip_detach=True)
    evs = _v5_1925(latch)
    det = [i for i, e in enumerate(evs) if e.detach]
    assert det == [6]                                   # 0.596 (streak 1) → 0.593 (streak 2)
    e = evs[6]
    assert e.slipped and not e.revoked and not e.attach
    assert latch.phase is GraspPhase.PLACED and e.phase is GraspPhase.PLACED
    assert latch.manipulated == "apple"                 # the identity lock stays (as an open release)
    s = e.evidence["slip"]
    assert s["attach_opening"] == pytest.approx(0.731)
    assert s["hold_command"] == pytest.approx(0.59)    # last *executed* command (carry), not HOLD's
    assert s["drop"] == pytest.approx(0.138, abs=1e-3) and s["gap"] == pytest.approx(0.003, abs=1e-3)
    assert s["streak"] == 2 and s["frames"] == 2
    assert e.note.startswith("slip")
    # the creep before the slip (drop ≤ 0.023) never counted
    assert [x.evidence["slip"]["streak"] for x in evs[:5]] == [0] * 5
    assert evs[5].evidence["slip"]["now"] and not evs[5].detach
    # PLACED is terminal for the episode: nothing re-attaches, nothing detaches again
    after = _hold(latch, 99, [0.0] * 8, 0.593)
    assert not after.attach and not after.detach and latch.phase is GraspPhase.PLACED


def test_slip_frames_one_detaches_on_the_first_closed_request():
    latch = _latched(slip_detach=True, slip_frames=1)
    evs = _v5_1925(latch)
    assert [i for i, e in enumerate(evs) if e.detach] == [5]


def test_a_held_object_under_a_tighter_command_is_blocked_and_kept():
    """A real hold: the policy squeezes (command 0.30) — the fingers stop on the object (gap 0.42)."""
    latch = _latched(slip_detach=True)
    _attach(latch)
    k = _carry(latch)
    for j in range(10):
        e = _hold(latch, k + j, [0.30] * 8, 0.72)
        assert not e.detach and not e.evidence["slip"]["now"]
    assert latch.phase is GraspPhase.HELD


def test_a_command_loosened_to_the_opening_alone_is_not_a_slip():
    """Gap ~0 but the fingers did not close past the object (drop 0.01): still held."""
    latch = _latched(slip_detach=True)
    _attach(latch)
    k = _carry(latch)
    for j in range(5):
        e = _hold(latch, k + j, [0.72] * 8, 0.721)
        assert not e.detach
    assert e.evidence["slip"]["gap"] == pytest.approx(0.001, abs=1e-3)


def test_a_drop_still_blocked_by_the_command_is_not_a_slip():
    """Drop 0.10 (object squeezed / re-seated) but the fingers stop 0.33 above the command: held."""
    latch = _latched(slip_detach=True)
    _attach(latch)
    k = _carry(latch)
    for j in range(5):
        e = _hold(latch, k + j, [0.30] * 8, 0.63)
        assert not e.detach
    assert latch.phase is GraspPhase.HELD


def test_one_glitched_request_resets_and_does_not_detach():
    latch = _latched(slip_detach=True)
    _attach(latch)
    k = _carry(latch)
    _hold(latch, k, [0.59] * 8, 0.592)                   # one request at the command
    e = _hold(latch, k + 1, [0.59] * 8, 0.728)           # back on the object
    assert not e.detach and e.evidence["slip"]["streak"] == 0
    e = _hold(latch, k + 2, [0.59] * 8, 0.592)
    assert not e.detach and e.evidence["slip"]["streak"] == 1


def test_an_open_release_is_still_an_open_release():
    latch = _latched(slip_detach=True)
    _attach(latch)
    k = _carry(latch)
    _hold(latch, k, [1.0] * 8, 0.95)
    e = _hold(latch, k + 1, [1.0] * 8, 1.0)
    assert e.detach and not e.slipped and latch.phase is GraspPhase.PLACED
    assert e.note.startswith("개도가 2 프레임 열려")


def test_inside_the_t34_window_the_revoke_answers_first():
    """E3b ep1808 r1 shape (attach 0.684 → 0.555 at the command 0.552 on the first executed chunk):
    with the flag on it is still a revoke (LATCHED), not a slip (PLACED)."""
    latch = _latched(slip_detach=True)
    _attach(latch, opening=0.684, command=0.576)
    e = _hold(latch, 4, [0.552] * 8, 0.555)
    assert e.detach and e.revoked and not e.slipped and latch.phase is GraspPhase.LATCHED


def test_the_held_command_rides_through_hold_chunks_whatever_their_applied_value_says():
    """HOLD chunks' applied values are not commands (T22) — even if a client wrote the current state
    (0.72) there, the slip is judged against the last executed command (0.59)."""
    latch = _latched(slip_detach=True)
    _attach(latch)
    k = _carry(latch)
    evs = [_hold(latch, k + j, [0.72] * 8, m, executed=HOLD) for j, m in enumerate((0.60, 0.592))]
    assert evs[-1].detach and evs[-1].slipped
    assert evs[-1].evidence["slip"]["hold_command"] == pytest.approx(0.59)


def test_hold_only_after_attach_uses_the_closing_command():
    """Nothing executed after the attach: the command is the closing attempt's last one."""
    latch = _latched(slip_detach=True, revoke_use_opening=False)   # window would revoke otherwise
    _attach(latch, command=0.587)
    evs = [_hold(latch, 4 + j, [0.9] * 8, 0.59, executed=HOLD) for j in range(2)]
    assert evs[-1].slipped and evs[-1].evidence["slip"]["hold_command"] == pytest.approx(0.587)


def test_unknown_opening_is_never_a_slip():
    latch = _latched(slip_detach=True)
    _attach(latch)
    k = _carry(latch)
    evs = [latch.observe_grasp(_signal(k + j, [0.59] * 8, 0.59, executed=HOLD),
                               reach_m=0.25, manipulated_id=0, manipulated_state="occluded")
           for j in range(3)]
    assert any(e.slipped for e in evs)          # sanity: with the opening known it slips
    latch = _latched(slip_detach=True)
    _attach(latch)
    k = _carry(latch)
    latch._attach_opening = None                                     # noqa: SLF001 — unknown
    for j in range(3):
        e = _hold(latch, k + j, [0.59] * 8, 0.59)
        assert not e.detach and e.evidence["slip"]["drop"] is None


def test_reset_clears_the_slip_state():
    latch = _latched(slip_detach=True)
    _attach(latch)
    k = _carry(latch)
    _hold(latch, k, [0.59] * 8, 0.592)
    latch.reset()
    assert latch._slip_streak == 0 and latch._hold_command is None   # noqa: SLF001


# --- 3. wiring: AG3SConfig → SafePolicy → record; serve_safe → AG3SConfig ----------------------


def _sp(held_slip_detach=None, **kw):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.trajopt.safe_policy import SafePolicy, grasp_parent_links
    from tests.trajopt.test_safe_policy import _robot_model, _StubPolicy

    model = _robot_model()
    clustering = {"gripper_max_opening": 0.0715}
    if held_slip_detach is not None:
        clustering["held_slip_detach"] = held_slip_detach
    ag3s = AG3S(AG3SConfig.from_dict({"collision_backend": "esdf", "clustering": clustering}),
                robot_model=model, constraint_robot_model=model,
                attached_parent_links=grasp_parent_links())
    return SafePolicy(_StubPolicy(), ag3s=ag3s, **kw)


def test_config_field_default_validation_and_round_trip():
    from AG3S.ag3s.config import AG3SConfig, AG3SConfigError

    assert AG3SConfig().clustering.held_slip_detach is False
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf"}).with_overrides(
        {"clustering": {"held_slip_detach": True}})
    assert cfg.clustering.held_slip_detach is True and cfg.collision_backend == "esdf"
    assert AG3SConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.to_dict()["clustering"]["held_slip_detach"] is True
    with pytest.raises(AG3SConfigError, match="held_slip_detach"):
        AG3SConfig.from_dict({"clustering": {"held_slip_detach": 1}})


def test_safe_policy_turns_the_latch_rule_on_from_the_ag3s_config_only():
    assert _sp()._latch.config.slip_detach is False                             # noqa: SLF001
    assert _sp(False)._latch.config.slip_detach is False                        # noqa: SLF001
    assert _sp(True)._latch.config.slip_detach is True                          # noqa: SLF001
    # an explicit latch config keeps its other fields
    sp = _sp(True, latch=LatchConfig(release_frames=3))
    assert sp._latch.config.slip_detach is True and sp._latch.config.release_frames == 3  # noqa: SLF001
    assert _sp(latch=LatchConfig(slip_detach=True))._latch.config.slip_detach is True     # noqa: SLF001


def _slip_feedbacks():
    fbs = [_fb(1, [1.0] * 8, 0.993), _fb(2, [0.002] * 8, 0.724), _fb(3, [0.587] * 8, 0.731)]
    fbs += [_fb(k, [0.59] * 8, 0.729) for k in range(4, 10)]
    fbs += [_fb(10, [0.593] * 8, 0.708, executed=HOLD), _fb(11, [0.593] * 8, 0.596, executed=HOLD),
            _fb(12, [0.593] * 8, 0.593, executed=HOLD)]
    return fbs


@pytest.mark.parametrize("on", [False, True])
def test_safe_policy_detaches_the_slipped_object_before_the_constraints(on):
    sp = _sp(on)
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(_finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))  # noqa: SLF001
    _lock(sp)
    recs = list(_grasp_frames(sp, q, _slip_feedbacks()))
    assert [r["attach"] for r in recs].index(True) == 2
    if not on:
        assert not any(r["detach"] for r in recs) and sp.ag3s.attached is not None
        assert all("held_slip" not in r for r in recs)
        assert all("slip" not in r["evidence"] for r in recs)
        return
    assert [r["detach"] for r in recs][-3:] == [False, False, True]
    assert sp.ag3s.attached is None and recs[-1]["state"] == "placed"
    assert recs[-1]["held_slip"] is True and recs[-2]["held_slip"] is False
    assert recs[-1]["attach_revoked"] is None and recs[-1]["state_version"] == 2
    assert recs[-1]["evidence"]["slip"]["streak"] == 2


def test_cli_flag_maps_to_the_clustering_section_and_needs_the_safe_layer():
    from AG3S.trajopt import serve_safe
    from tests.trajopt.test_t41a_obstacle_margin import BASE_ARGV

    ap = serve_safe.build_parser()
    assert "held_slip_detach" not in serve_safe.t43t_identity_section(ap.parse_args(BASE_ARGV))
    args = ap.parse_args(BASE_ARGV + ["--held-slip-detach"])
    sec = serve_safe.t43t_identity_section(args)
    assert sec == {"held_slip_detach": True}
    from AG3S.ag3s.config import AG3SConfig

    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": {"backend": "curobo"}}) \
        .with_overrides({"clustering": sec})
    serve_safe.assert_ag3s_config(cfg, collision_backend="esdf", esdf_backend="curobo", clustering=sec)
    with pytest.raises(RuntimeError, match="clustering.held_slip_detach"):
        serve_safe.assert_ag3s_config(AG3SConfig.from_dict({"collision_backend": "esdf",
                                                            "esdf": {"backend": "curobo"}}),
                                      collision_backend="esdf", esdf_backend="curobo", clustering=sec)
    with pytest.raises(SystemExit):
        bad = ap.parse_args(BASE_ARGV + ["--no-safe", "--held-slip-detach"])
        serve_safe.reject_bad_flag_combinations(ap, bad)


# --- 4. T43 W record replay ------------------------------------------------------------------

needs_records = pytest.mark.skipif(not R.available(), reason="T43 W / Y records not present")

#: (run, MuJoCo release t, slip detach t with slip_frames 2) — the 4 closed-command slips Y named.
SLIPS = [("V5/H/V5_ep1925_s19253", 369, 384), ("V5/H/V5_ep1807_s18071", 165, 176),
         ("V5/H/V5_ep1807_s18073", 215, 232), ("V5/B/V5_ep1807_s18071", 345, 360)]


@needs_records
@pytest.mark.parametrize("key,release_t,slip_t", SLIPS)
def test_replay_the_four_y_slips(key, release_t, slip_t):
    off = R.replay(key, slip_detach=False)
    assert off["mismatch"] == []                         # the replay reproduces the live latch
    assert not any(r["detach"] for r in off["rows"] if r["t"] > 136)      # stale to the end today
    on = R.replay(key, slip_detach=True)
    assert on["first_slip_t"] == slip_t and slip_t > release_t


@needs_records
def test_replay_no_slip_detach_before_the_release_in_any_scored_run():
    """All 38 V4 · V5 runs: off reproduces the record; on, no slip detach before the MuJoCo release
    (and none at all in a run that never released)."""
    y = R.truth()
    for key in R.scored_keys():
        assert R.replay(key, slip_detach=False)["mismatch"] == [], key
        t = R.replay(key, slip_detach=True)["first_slip_t"]
        rel = y[key].get("release_t")
        if t is not None:
            assert rel is not None and t > rel, (key, t, rel)
        if y[key].get("release_mode") == "open_cmd":
            assert t is None, key                       # open releases stay with the open path
