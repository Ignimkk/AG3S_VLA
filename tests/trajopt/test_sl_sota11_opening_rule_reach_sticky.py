"""SL-SOTA11 (2026-11-12, dev-57) — `--grasp-opening-rule-reach-sticky` (versions ``-rs``): inside one closing attempt of the
opening rule, a reach that passed while the closed command went on stays passed (`LatchConfig.opening_rule_reach_sticky`).

E1a ...-pt6-pg4-hr150-hb-qs-tro task 0 init 11 (the same on goal L-I task 3 init 16 and goal L-II task 0 init 16): the reach
passes while the fingers close (52-78 mm), the opening stops moving one call later -- by then the hand has lifted the bowl and
the reach against the occluded bowl's last observed geometry is 128 mm (> 120). `settled` and `reach` are never true in the
same call, the latch stays CLOSING for 41 calls and the bowl in the hand is an obstacle. V11AB-pb6's `reach_sticky` fixed
this in the evidence state machine; the opening rule replaces that state machine. Off by default.
"""
from __future__ import annotations

import pytest

from AG3S.trajopt.grasp_latch import GraspPhase, LatchConfig
from tests.benchmark.test_sl14_opening_rule import _latch, _sig

#: (last executed command, measured opening, reach m) of task 0 init 11, calls 14-19
T0E11 = [(1.0, 0.83, 0.108), (0.0, 0.63, 0.052), (0.0, 0.29, 0.053), (0.0, 0.07, 0.057), (0.0, 0.045, 0.078),
         (0.0, 0.045, 0.128)]


def _run(latch, rows, mid=0):
    ev = None
    for k, (cmd, opening, reach) in enumerate(rows):
        ev = latch.observe_grasp(_sig(cmd, opening, seq=k + 1), reach_m=reach, manipulated_id=mid,
                                 manipulated_state="occluded", reach_context={"geometry_frame": k})
    return ev


def _cfg(**kw):
    return dict(opening_rule_moved_min=0.05, reach_sticky=True, **kw)


def test_default_off_and_the_recorded_failure_is_reproduced():
    assert LatchConfig().opening_rule_reach_sticky is False
    latch = _latch(**_cfg())
    ev = _run(latch, T0E11)
    assert not ev.attach and latch.phase is GraspPhase.CLOSING
    assert ev.evidence["checks"] == {"closed_command": True, "settled": True, "opening_min": True, "reach": False,
                                     "moved": True}
    assert "reach_sticky" not in ev.evidence          # pb6's flag alone does not reach the opening rule


def test_the_pass_of_the_closing_calls_attaches_when_the_opening_settles():
    latch = _latch(**_cfg(opening_rule_reach_sticky=True))
    ev = _run(latch, T0E11[:-1])
    assert not ev.attach and ev.evidence["checks"]["reach"] is True and ev.evidence["reach_sticky"]["used"] is False
    ev = _run(latch, T0E11[-1:])
    assert ev.attach and latch.phase is GraspPhase.HELD
    st = ev.evidence["reach_sticky"]
    assert st["used"] is True and st["now"] is False and st["now_mm"] == 128.0 and st["same_object"] is True
    assert st["passed"]["reach_mm"] == 52.0 and st["passed"]["manipulated_id"] == 0
    assert ev.evidence["checks"] == {"closed_command": True, "settled": True, "opening_min": True, "reach": True,
                                     "moved": True}
    assert ev.evidence["opening_rule"]["decision"] == "attach" and ev.evidence["opening_rule"]["reach_mm"] == 128.0


def test_an_empty_close_does_not_attach():
    latch = _latch(**_cfg(opening_rule_reach_sticky=True))
    ev = _run(latch, [(1.0, 0.83, 0.108), (0.0, 0.40, 0.06), (0.0, 0.02, 0.07), (0.0, 0.02, 0.13), (0.0, 0.02, 0.13)])
    assert not ev.attach and latch.phase is GraspPhase.CLOSING
    assert ev.evidence["checks"]["reach"] is True and ev.evidence["checks"]["opening_min"] is False


def test_an_executed_open_command_ends_the_pass():
    """Closed near the target, opened again without leaving CLOSING (the opening stays under the closed threshold), then
    closed on something 130 mm away: the earlier pass does not count."""
    latch = _latch(**_cfg(opening_rule_reach_sticky=True))
    ev = _run(latch, [(1.0, 0.83, 0.108), (0.0, 0.40, 0.06), (1.0, 0.30, 0.09), (0.0, 0.06, 0.13), (0.0, 0.06, 0.13)])
    assert latch.phase is GraspPhase.CLOSING and not ev.attach
    assert ev.evidence["checks"]["reach"] is False and ev.evidence["reach_sticky"]["passed"] is None


def test_never_within_reach_never_attaches_and_another_object_does_not_inherit_the_pass():
    latch = _latch(**_cfg(opening_rule_reach_sticky=True))
    ev = _run(latch, [(1.0, 0.83, 0.20), (0.0, 0.40, 0.19), (0.0, 0.06, 0.19), (0.0, 0.06, 0.19)])
    assert not ev.attach and ev.evidence["reach_sticky"]["passed"] is None
    latch = _latch(**_cfg(opening_rule_reach_sticky=True))
    _run(latch, T0E11[:-1], mid=0)
    ev = _run(latch, T0E11[-1:], mid=1)               # the manipulated object changed: the pass is the other object's
    assert not ev.attach and ev.evidence["reach_sticky"]["same_object"] is False


def test_a_new_attempt_starts_without_the_pass():
    latch = _latch(**_cfg(opening_rule_reach_sticky=True))
    _run(latch, [(1.0, 0.83, 0.108), (0.0, 0.40, 0.06)])                 # passed, then given up (fully reopened)
    ev = _run(latch, [(1.0, 0.95, 0.20)])
    assert latch.phase is GraspPhase.LATCHED and latch._reach_pass is None
    ev = _run(latch, [(0.0, 0.50, 0.15), (0.0, 0.06, 0.15), (0.0, 0.06, 0.15)])
    assert not ev.attach and ev.evidence["reach_sticky"]["passed"] is None


def test_flag_plumbing_and_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss
    from tests.trajopt.test_t88_passthrough_when_clear import BASE as ARGV

    ap = ss.build_parser()
    assert ap.parse_args(ARGV).grasp_opening_rule_reach_sticky is False
    for bad in (["--grasp-opening-rule-reach-sticky"],
                ["--grasp-opening-rule", "--grasp-opening-rule-reach-sticky"],
                ["--grasp-reach-sticky", "--grasp-opening-rule-reach-sticky"]):
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, ap.parse_args(ARGV + bad))
    assert srv.panda_latch_config().opening_rule_reach_sticky is False
    assert srv.panda_latch_config(opening_rule=True, opening_rule_reach_sticky=True).opening_rule_reach_sticky is True
    assert "grasp_opening_rule_reach_sticky" in srv.PORTED_FINAL_CONFIG_DESTS
    base = "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150"
    for mid in ("-qs", "-hb-qs"):
        name = base + mid + "-rs-tro"
        assert name in pf.SOTA_REACHSTICKY_VERSIONS and name in pf.GT15_VERSIONS
        v = tuple(pf.AG3S_VERSIONS[name])
        assert v[-1] == "--grasp-opening-rule-reach-sticky" and v[:-1] == tuple(pf.AG3S_VERSIONS[base + mid + "-tro"])
        args = pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(v))
        assert args.grasp_opening_rule_reach_sticky is True and args.grasp_opening_rule is True
        assert args.grasp_reach_sticky is True
    assert all(n.endswith("-qs-rs-tro") and "-pt6-" in n and "-hr150" in n for n in pf.SOTA_REACHSTICKY_VERSIONS)
    for n in pf.SOTA_REACHSTICKY_VERSIONS:
        assert pf.version_server_flags(n)
    for name, flags in pf.AG3S_VERSIONS.items():
        if name not in pf.SOTA_REACHSTICKY_VERSIONS:
            assert "--grasp-opening-rule-reach-sticky" not in flags, name
    assert "grasp_opening_rule_reach_sticky" not in pf.version_flag_values(pf.parse_v11ab_args())


# ------------------------------------------------------------------ the real SafePolicy._advance_grasp / AG3S.attach
def _policy_run(**latch):
    import numpy as np

    from tests.benchmark.test_sl14_opening_rule import _frames, _sp
    from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _manipulated

    sp, q = _sp(distance=0.05, latch=LatchConfig(opening_rule=True, reach_sticky=True, **latch))
    recs = _frames(sp, q, [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.30)])      # closing, 50 mm: passes, not settled
    # the hand lifted and the close seed re-grounded the (same id) target on something 200 mm away
    sp.ag3s._manipulated = _manipulated(_finger_origin(sp, q) + np.array([0.20, 0.0, 0.0]))  # noqa: SLF001
    recs += _frames(sp, q, [_fb(3, [0.0] * 8, 0.06), _fb(4, [0.0] * 8, 0.06)])     # the opening settles on the last one
    return sp, recs


def test_the_policy_attaches_the_geometry_whose_reach_passed():
    sp, recs = _policy_run()
    assert recs[-1]["state"] == "closing" and sp.ag3s.attached is None             # off: the recorded failure
    assert "reach_sticky" not in recs[-1]["evidence"] and getattr(sp, "_rs_pass_geometry", None) is None
    sp, recs = _policy_run(opening_rule_reach_sticky=True)
    assert [r["state"] for r in recs] == ["latched", "closing", "closing", "held"] and sp.ag3s.attached is not None
    st = recs[-1]["evidence"]["reach_sticky"]
    assert st["used"] is True and st["now_mm"] > 120.0 and st["attach_geometry_frame"] == 4
    gate = sp.ag3s._held_record["reach_gate"]                                      # noqa: SLF001
    # the snapshot is the geometry that was within reach (about 50 mm), not the latest one (about 200 mm)
    assert gate["sticky"] is True and gate["over_limit"] is False and gate["dist_mm"] < 120.0
