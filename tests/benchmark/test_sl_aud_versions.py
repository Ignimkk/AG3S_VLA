"""SL-AUD (code audit 2026-11-13): the versions that carry the action-QP switches of `tests/trajopt/test_sl_aud_action_qp.py`
(`profile.SOTA_AUDIT_VERSIONS`) -- on the frozen version and on its ``-gh156a`` version."""
from __future__ import annotations

import pytest

from AG3S.adapters.panda import profile as pf
from AG3S.adapters.panda import server as srv
from AG3S.trajopt import serve_safe as ss


def test_audit_versions():
    ap = ss.build_parser()
    frozen = pf.FROZEN_20261112
    stem = frozen[:-len("-tro")]
    gh = stem + "-gh156a-tro"                                             # SL-SOTA18b, the version at the gate
    suffixes = ("-gr", "-rf", "-lp", "-mr8", "-mr12", "-af", "-md5", "-rf-lp-mr8-af",
                "-ohr", "-dlo", "-pro", "-ohr-dlo-pro", "-rf-lp-mr8-af-ohr-dlo-pro")
    assert pf.SOTA_AUDIT_VERSIONS == tuple(b[:-len("-tro")] + s + "-tro" for b in (frozen, gh) for s in suffixes)
    assert len(set(pf.SOTA_AUDIT_VERSIONS)) == 26 and not set(pf.SOTA_AUDIT_VERSIONS) & {frozen, gh}
    assert set(pf.SOTA_AUDIT_VERSIONS) <= set(pf.GT15_VERSIONS) and all(pf.is_gtto_version(n) for n in pf.SOTA_AUDIT_VERSIONS)
    assert not any(pf.version_requires_oracle_segmentation(n) for n in pf.SOTA_AUDIT_VERSIONS)            # no GT
    for name in pf.SOTA_AUDIT_VERSIONS:
        b = gh if "-gh156a" in name else frozen
        n = len(pf.AG3S_VERSIONS[b])
        v = tuple(pf.AG3S_VERSIONS[name])
        assert v[:n] == tuple(pf.AG3S_VERSIONS[b]) and len(v) > n        # the base argv + the new flags only
        a = ap.parse_args(pf.version_server_argv(name) + list(pf.version_server_flags(name)))
        ss.reject_bad_flag_combinations(ap, a)
        srv.check_unported_flags(a, None)
        ss.trajopt_config_from_args(a).validate()
    a = ap.parse_args(pf.version_server_argv(frozen) + list(pf.version_server_flags(stem + "-rf-lp-mr8-af-tro")))
    sqp = ss.trajopt_config_from_args(a).sqp
    assert sqp.ee_action_reference_floor and sqp.ee_action_limit_passthrough and sqp.ee_action_audit_fixes
    assert sqp.ee_action_min_reach == pytest.approx(0.008) and not sqp.ee_action_gate_rows
    m = ss.trajopt_config_from_args(ap.parse_args(pf.version_server_argv(gh) + list(pf.version_server_flags(
        gh[:-len("-tro")] + "-md5-tro")))).sqp
    assert m.ee_action_max_delta == pytest.approx(0.5) and not m.ee_action_reference_floor
    # the place-phase switches (SL-AUD6-8) and everything together
    a = ap.parse_args(pf.version_server_argv(gh) + list(pf.version_server_flags(gh[:-len("-tro")] + "-ohr-dlo-pro-tro")))
    sec = ss.t43t_identity_section(a)
    assert sec["scene_obstacle_history_refilter"] is True and sec["destination_leader_outside_hand"] is True
    assert a.ee_action_qp_progress_release_off is True and not ss.trajopt_config_from_args(a).sqp.ee_action_reference_floor
    a = ap.parse_args(pf.version_server_argv(frozen) + list(pf.version_server_flags(stem + "-rf-lp-mr8-af-ohr-dlo-pro-tro")))
    assert ss.trajopt_config_from_args(a).sqp.ee_action_reference_floor and a.scene_obstacle_history_refilter
    assert a.destination_leader_outside_hand and a.ee_action_qp_progress_release_off
