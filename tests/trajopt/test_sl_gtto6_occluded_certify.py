"""SL-GTTO6 F5 — `--occluded-certify` (lead self-ruling G40, 2026-11-05), CPU.

1. `SafePolicy._occluded_certified` = the verdict's `occluded_target` condition (`_certification_reason`): true exactly
   when that reason would be `occluded_target`, false for ok / lost / degraded / no constraint set.
2. `_scene_fn` hands the TO `certified=True` on such a frame only with the switch on (source check of the one line).
3. serve_safe flag + `apply_grasp_contact_flags`; profile `AG3S-commonto-gt2-termsoff-pass-zr-oc` = zr + the flag.
"""
from __future__ import annotations

import inspect
import types

import pytest

from tests.trajopt.test_sl6h_placement_contact import _policy


def _cs(status="no_target", certified=True, state="occluded", manipulated=True):
    return types.SimpleNamespace(
        status=types.SimpleNamespace(value=status), grounding_status=types.SimpleNamespace(value="no_cluster"),
        validity=types.SimpleNamespace(value="valid"), geometry_certified=certified,
        metrics={"manipulated": {"id": 0, "state": state, "age_frames": 3}},
        manipulated=object() if manipulated else None)


@pytest.mark.parametrize("kw,want", [
    ({}, True),
    ({"status": "ok"}, False),                 # certified anyway by `geometry_certified`, not this rule
    ({"status": "degraded"}, False),
    ({"certified": False}, False),
    ({"state": "lost"}, False),
    ({"state": "visible"}, False),
    ({"manipulated": False}, False)])
def test_same_condition_as_the_verdict(kw, want):
    sp = _policy()
    cs = _cs(**kw)
    assert sp._occluded_certified(cs) is want
    assert (sp._certification_reason(cs)["kind"] == "occluded_target") is (want or kw == {"status": "ok"} and False)
    assert sp._occluded_certified(None) is False


def test_scene_fn_uses_it_only_when_on():
    from AG3S.trajopt import safe_policy as m

    src = inspect.getsource(m.SafePolicy)
    assert "if not certified and self.occluded_certify and self._occluded_certified(constraint_set):" in src
    assert _policy().occluded_certify is False


def test_flag():
    from AG3S.trajopt import serve_safe as ss

    act = ss.build_parser()._option_string_actions["--occluded-certify"]
    assert act.dest == "occluded_certify" and act.default is False
    sp = types.SimpleNamespace()
    assert "occluded_certify" not in ss.apply_grasp_contact_flags(sp, types.SimpleNamespace(occluded_certify=False))
    out = ss.apply_grasp_contact_flags(sp, types.SimpleNamespace(occluded_certify=True))
    assert out["occluded_certify"] is True and sp.occluded_certify is True


def test_profile_version(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from tests.benchmark.test_sl_gtto_gt_perception import _args

    v = "AG3S-commonto-gt2-termsoff-pass-zr-oc"
    assert pf.GT6_VERSIONS == (v, v + "-hbc")
    assert list(pf.AG3S_VERSIONS[v + "-hbc"]) == list(pf.AG3S_VERSIONS[v]) + ["--held-beneath-contact"]
    assert list(pf.AG3S_VERSIONS[v]) == list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr"]) + ["--occluded-certify"]
    assert "occluded_certify" in srv.PORTED_FINAL_CONFIG_DESTS
    args = _args(v)
    assert pf.match_ag3s_version(args) == v and args.occluded_certify is True
    assert pf.version_flag_values(args)["occluded_certify"] is True
    assert "occluded_certify" not in pf.version_flag_values(_args("AG3S-commonto-gt2-termsoff-pass-zr"))
    for name, flags in pf.AG3S_VERSIONS.items():
        if name not in pf.GT6_VERSIONS + pf.GT7_VERSIONS:
            assert "--occluded-certify" not in flags, name


from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: E402,F401 (fixture)
