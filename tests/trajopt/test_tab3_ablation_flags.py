"""TAB3 (paper Tab. 3, 2026-11-16) — row 7 (`--no-attached-query`) and its version on X-fwd.

Row 7: with `collision.attached_query` False the real `scene_from_constraint_set` drops the held object's query spheres
(TO rows and final check share the snapshot); the holding-state rules (held capture volume, strict held permission) stay.
Version `-ab7`: argv = X-fwd + the one switch, parsed with the server's own checks, config validated,
record keys only with the flag. Off: the config dump, the snapshot and every earlier version's record keys are as before.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig
from AG3S.trajopt.linearize import scene_from_constraint_set
from tests.trajopt.test_t32_held_query import _constraint_set, _CrateWall, _held

SEGMENTS = ("-ab7",)
ADDED = {"-ab7": ("--no-attached-query",)}


def _cfg(**collision):
    return TrajOptConfig.from_dict({"collision": {"backend": "esdf", "esdf_margin": 0.05, "use_support_planes": False,
                                                  **collision},
                                    "horizon": {"lookahead": "obstacles", "plan_horizon": 16}})


def _parse(v):
    from AG3S.adapters.panda import profile as pf

    return pf.parse_v11ab_args(argv=pf.version_server_argv(v) + list(pf.version_server_flags(v)))


# ============================================================================ row 7 — the snapshot

def test_config_default_and_dump():
    assert TrajOptConfig().collision.attached_query is True
    assert "attached_query" not in TrajOptConfig().to_dict()["collision"]          # off: the dump keeps its keys
    c = _cfg(attached_query=False)
    c.validate()
    assert c.to_dict()["collision"]["attached_query"] is False
    assert TrajOptConfig.from_dict(c.to_dict()).collision.attached_query is False
    with pytest.raises(ValueError):
        _cfg(attached_query=0).validate()


def test_off_the_held_spheres_are_queried_on_they_are_not():
    cap = np.array([[0.5, 0.0, 0.9, 0.04]])
    cs = _constraint_set(_held(), _CrateWall())
    off = scene_from_constraint_set(cs, np.array([0.01]), _cfg(), held_capture=cap)
    assert off.attached_parent_link == "ee_left" and len(off.attached_points) == 2 and off.attached_radii is not None
    on = scene_from_constraint_set(cs, np.array([0.01]), _cfg(attached_query=False), held_capture=cap)
    assert on.attached_points is None and on.attached_radii is None and on.attached_parent_link is None
    # the holding state is still seen: the held capture volume of the look-ahead tail stays
    assert np.allclose(off.held_capture_spheres, cap) and np.allclose(on.held_capture_spheres, cap)
    # nothing held: the flag changes nothing
    free = _constraint_set(None, _CrateWall())
    a = scene_from_constraint_set(free, np.array([0.01]), _cfg())
    b = scene_from_constraint_set(free, np.array([0.01]), _cfg(attached_query=False))
    assert a.attached_points is None and b.attached_points is None
    assert a.held_capture_spheres is None and b.held_capture_spheres is None


def test_the_held_point_model_is_dropped_too():
    cs = _constraint_set(_held(), _CrateWall())
    cs.metrics = {"held_point_model": {"parent_link": "ee_left", "spheres": [[0.0, 0.0, 0.1, 0.01]] * 3}}
    assert len(scene_from_constraint_set(cs, np.array([0.01]), _cfg()).attached_points) == 3
    assert scene_from_constraint_set(cs, np.array([0.01]), _cfg(attached_query=False)).attached_points is None


def test_grasp_record_says_queried_false_only_with_the_flag():
    from AG3S.trajopt.safe_policy import SafePolicy

    held = _held()
    for flag, want in ((True, None), (False, False)):
        fake = types.SimpleNamespace(
            _last_grasp_record={"mode": "evidence"}, _latch=types.SimpleNamespace(phase=types.SimpleNamespace(value="held")),
            _grasp_state_version=1, _grasp_constraint_version=1,
            ag3s=types.SimpleNamespace(attached=held, held_record=None), to_config=_cfg(attached_query=flag))
        SafePolicy._finish_grasp_record(fake, types.SimpleNamespace(attached=held))
        rec = fake._last_grasp_record["attached"]
        assert rec["parent_link"] == "ee_left" and rec.get("queried") == want and (("queried" in rec) is (want is not None))


# ============================================================================ argv -> config -> snapshot

def test_ab7_argv_reaches_the_snapshot():
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    xf = pf.SOTA_FINAL_X_FWD
    cs = _constraint_set(_held(), _CrateWall())
    for v, queried in ((xf, True), (xf[:-len("-tro")] + "-ab7-tro", False)):
        cfg = ss.trajopt_config_from_args(_parse(v))
        cfg.validate()
        assert cfg.collision.attached_query is queried
        scene = scene_from_constraint_set(cs, np.array([0.01]), cfg)
        assert (scene.attached_points is not None) is queried


# ============================================================================ versions and server checks

def test_tab3_code_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss

    xf = pf.SOTA_FINAL_X_FWD
    names = tuple(xf[:-len("-tro")] + s + "-tro" for s in SEGMENTS)
    assert pf.SOTA_TAB3_CODE_VERSIONS == names and set(names) <= set(pf.GT15_VERSIONS)
    assert list(pf.AG3S_VERSIONS)[-len(names):] == list(names)                       # appended at the end of the registry
    base = tuple(pf.AG3S_VERSIONS[xf])
    xa = _parse(xf)
    xvals = pf.version_flag_values(xa)
    assert not set(pf.TAB3_CODE_DESTS) & set(xvals)
    assert srv.tab3_record(xa) == {} and ss.tab3_inert(xa) == []
    for seg, v in zip(SEGMENTS, names):
        assert pf.AG3S_VERSIONS[v] == base + ADDED[seg]                        # X-fwd + the one switch
        a = _parse(v)                                                           # reject_bad_flag_combinations inside
        ap = ss.build_parser()
        ss.reject_bad_flag_combinations(ap, a)
        srv.check_unported_flags(a, None)
        srv.check_inpass_flags(a, None)
        ss.trajopt_config_from_args(a).validate()
        assert a.attention_inpass is True and a.latched_identity_hold is True   # base argv untouched
        vals = pf.version_flag_values(a)
        new = set(vals) - set(xvals)
        assert set(xvals) <= set(vals) and all(vals[k] == xvals[k] for k in xvals)
        rec = srv.tab3_record(a)["tab3_ablation"]
        assert new == {"no_attached_query"} and vals["no_attached_query"] is True and rec == {"attached_query": False}
        assert set(srv.final_options_on(a)) - set(srv.final_options_on(xa)) == {"no_attached_query"}


def test_earlier_versions_keep_their_record_keys():
    """No version registered before TAB3 code rows records one of their dests (spot check over the GT15 family, which
    is where they were registered; parsing every version is slow)."""
    from AG3S.adapters.panda import profile as pf

    for v in pf.SOTA_TAB3_VERSIONS + (pf.SOTA_FINAL_X_FWD,):
        assert not set(pf.TAB3_CODE_DESTS) & set(pf.version_flag_values(_parse(v)))


@pytest.mark.parametrize("extra, msg", [
    (["--no-attached-query", "--no-safe"], "no-attached-query"),
])
def test_bad_combinations_are_refused(extra, msg, capsys):
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    xf = pf.SOTA_FINAL_X_FWD
    ap = ss.build_parser()
    args = ap.parse_args(pf.version_server_argv(xf) + list(pf.version_server_flags(xf)) + extra)
    with pytest.raises(SystemExit):
        ss.reject_tab3_flags(ap, args)
    assert msg in capsys.readouterr().err
