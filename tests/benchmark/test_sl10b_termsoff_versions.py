"""SL10b -- "쥔 뒤 변화율 항 끔": pc판 and pc판 + surface with the continuity / deflection-rate terms off in every phase
(task `AG3S/ag3s/docs/handoff/SL10b.task.md` §1). Pins: (a) each version is its base minus the two after-grasp flags plus
`--w-continuity 0 --w-deflection-rate 0`, nothing else; (b) the server's TrajOpt config has both terms off in every phase
(weights 0, no `*_after_grasp` key) while the bases keep them from HELD; (c) the bases did not move and their records keep
their keys."""
from __future__ import annotations

import pathlib

import pytest

from tests.benchmark.test_sl6a_panda_final import _Stub, gpu_ok  # noqa: F401 (fixture)

PC = "AG3S-final-panda-v3-e0to-pc"
PCS = "AG3S-final-panda-v3-e0to-pc-surface"
VERSIONS = {f"{PC}-termsoff": PC, f"{PCS}-termsoff": PCS}
AFTER = ("--continuity-after-grasp", "--deflection-rate-after-grasp")
OFF = ["--w-continuity", "0", "--w-deflection-rate", "0"]


def _probe_or_skip(v):
    from AG3S.adapters.panda import profile as pf

    if v.startswith(PCS) and not pathlib.Path(pf.PANDA_HSR6_PROBE).exists():
        pytest.skip("Panda HSR6 probe not in this storage root")


def _args(v, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(v)
    argv[argv.index("--config") + 1] = "pi05_libero"
    return pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(v))


def test_versions_are_their_bases_minus_the_after_grasp_flags_plus_terms_off(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    assert set(VERSIONS) == set(pf.SL10B_VERSIONS)
    for v, base in VERSIONS.items():
        b, f = list(pf.AG3S_VERSIONS[base]), list(pf.AG3S_VERSIONS[v])
        assert all(t in b for t in AFTER) and not any(t in f for t in AFTER)
        assert f == [t for t in b if t not in AFTER] + OFF
        # the whole diff: two flags gone, four tokens added
        assert [t for t in b if t not in f] == list(AFTER)
        assert f[-4:] == OFF and not set(OFF[::2]) & set(b)
        assert pf.is_final_version(v) and v in pf.PC_LATCH_VERSIONS
        assert pf.version_server_argv(v) == pf.version_server_argv(base)
        assert pf.version_server_flags(v)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_server_config_has_both_terms_off_in_every_phase(version, gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss

    _probe_or_skip(version)
    base = VERSIONS[version]
    a, b = _args(version, monkeypatch), _args(base, monkeypatch)
    # parsed flags
    assert a.w_continuity == 0.0 and a.w_deflection_rate == 0.0
    assert not a.continuity_after_grasp and not a.deflection_rate_after_grasp
    assert b.continuity_after_grasp and b.deflection_rate_after_grasp and b.w_continuity is None
    # cost overrides handed to TrajOptConfig: weights 0, no after_grasp key
    ov = ss.cost_overrides(a)
    assert ov["w_continuity"] == 0.0 and ov["w_deflection_rate"] == 0.0
    assert "continuity_after_grasp" not in ov and "deflection_rate_after_grasp" not in ov
    ob = ss.cost_overrides(b)
    assert ob["continuity_after_grasp"] is True and ob["deflection_rate_after_grasp"] is True
    # the built server
    ps.check_unported_flags(a, None)
    served, _info = ps.build_safe_server(_Stub(), a, gpu_warm_up=False)
    cost = served._policy.to_config.cost
    assert cost.w_continuity == 0.0 and cost.w_deflection_rate == 0.0
    assert not cost.continuity_after_grasp and not cost.deflection_rate_after_grasp
    base_served, _ = ps.build_safe_server(_Stub(), b, gpu_warm_up=False)
    bcost = base_served._policy.to_config.cost
    assert bcost.continuity_after_grasp and bcost.deflection_rate_after_grasp
    assert bcost.w_continuity > 0.0 and bcost.w_deflection_rate > 0.0
    # every other cost / limit field is the base's
    for f in ("w_track", "w_smooth", "w_slack"):
        assert getattr(cost, f) == getattr(bcost, f)
    # the other flag values: only the four terms-related dests differ
    from AG3S.adapters.panda import profile as pf

    va, vb = vars(a), vars(b)
    assert {k for k in va if va[k] != vb[k]} == {"w_continuity", "w_deflection_rate", "continuity_after_grasp",
                                                  "deflection_rate_after_grasp"}
    assert pf.match_ag3s_version(a) == version and pf.match_ag3s_version(b) == base


def test_base_versions_unchanged_and_keep_their_record_keys(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    assert pf.AG3S_VERSIONS[PC] == pf.FINAL_CONFIG_V3_E0TO_PC_FLAGS and pf.AG3S_VERSIONS[PCS] == pf.PC_SURFACE_FLAGS
    for base in VERSIONS.values():
        assert base not in pf.SL10B_VERSIONS and all(t in pf.AG3S_VERSIONS[base] for t in AFTER)
        assert not any(t in pf.AG3S_VERSIONS[base] for t in ("--w-continuity", "--w-deflection-rate"))
        _probe_or_skip(base)
        vals = pf.version_flag_values(_args(base, monkeypatch))
        assert vals["continuity_after_grasp"] is True and vals["deflection_rate_after_grasp"] is True
    # the SL10b dests are ones the earlier versions already record (no new key on any record)
    for v in pf.SL10B_VERSIONS:
        _probe_or_skip(v)
        va = pf.version_flag_values(_args(v, monkeypatch))
        vb = pf.version_flag_values(_args(VERSIONS[v], monkeypatch))
        assert set(va) == set(vb)
