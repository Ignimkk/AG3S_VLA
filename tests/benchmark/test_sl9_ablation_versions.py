"""SL9 -- the development proposals A (target as a closed solid, T96 / T96c / T96d) and B (removed-seed sign fix, T96e) as
Panda ablation versions (task `AG3S/ag3s/docs/handoff/SL9.task.md` §1-2, §4-2): flag classification of the merge's options,
the seven new versions (flag diff, build, esdf config), the base versions unchanged, and the `panda/surface` record keys."""
from __future__ import annotations

import pathlib
import types

import pytest

from tests.benchmark.test_sl6a_panda_final import _Stub, gpu_ok  # noqa: F401 (fixture)

P0M = "AG3S-final-panda-p0m"
PCS = "AG3S-final-panda-v3-e0to-pc-surface"
FB = "AG3S-final-panda-v3-e0to-pc-surface-fb"

#: version -> (base, A, B)
SL9 = {
    f"{P0M}-B": (P0M, False, True), f"{P0M}-A": (P0M, True, False), f"{P0M}-AB": (P0M, True, True),
    f"{FB}-B": (FB, False, True),
    f"{PCS}-B": (PCS, False, True), f"{PCS}-A": (PCS, True, False), f"{PCS}-AB": (PCS, True, True),
}
A_ADDED = ["--surface-solid-owned", "--surface-solid-protect-mm", "5", "--surface-solid-dilate-mm", "0",
           "--surface-solid-dn-ever-target", "--surface-solid-shadow-mm", "80"]
SOLID_DESTS = ("surface_solid_owned", "surface_solid_protect_mm", "surface_solid_dilate_mm", "surface_solid_shadow_mm",
               "surface_solid_dn_ever_target", "surface_solid_dn_support", "surface_removed_sign_free")
UNPORTED_DESTS = ("surface_crease_protrusion", "surface_crease_protrusion_area", "surface_crease_hand_freeze_mm",
                  "palm_cover_margin")


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


def _drop_pairs(flags, dropped):
    out, i = [], 0
    while i < len(flags):
        if flags[i] in dropped:
            i += 1 + dropped[flags[i]]
        else:
            out.append(flags[i]); i += 1
    return out


# --- classification of the merge's options --------------------------------------------------------------------------
def test_merge_options_are_classified():
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss

    dests = {a.dest for a in ss.build_parser()._actions}
    for d in SOLID_DESTS:
        assert d in dests and d in ps.PORTED_FINAL_CONFIG_DESTS and d not in ps.UNPORTED_FINAL_CONFIG_DESTS, d
    for d in UNPORTED_DESTS:
        assert d in dests and d in ps.UNPORTED_FINAL_CONFIG_DESTS and d not in ps.PORTED_FINAL_CONFIG_DESTS, d
    assert not set(ps.UNPORTED_FINAL_CONFIG_DESTS) & set(ps.PORTED_FINAL_CONFIG_DESTS)


@pytest.mark.parametrize("dest", UNPORTED_DESTS)
def test_unported_options_are_refused(dest, gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps

    args = _args(P0M, monkeypatch)
    ps.check_unported_flags(args, None)
    setattr(args, dest, "set-by-test")
    with pytest.raises(ValueError, match="not ported"):
        ps.check_unported_flags(args, None)


# --- the seven versions ---------------------------------------------------------------------------------------------
def test_flag_diff_against_the_base(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    assert set(SL9) == set(pf.SL9_VERSIONS) and len(SL9) == 7
    for v, (base, a, b) in SL9.items():
        flags, bflags = list(pf.AG3S_VERSIONS[v]), list(pf.AG3S_VERSIONS[base])
        expected = bflags
        if a:
            expected = _drop_pairs(bflags, {"--surface-hull-fill": 0, "--surface-contact-allowance-mm": 1}) + A_ADDED
        if b:
            expected = expected + ["--surface-removed-sign-free"]
        assert flags == expected, v
        assert ("--surface-hull-fill" in flags) is (not a and base != FB)             # A removes it, fb never had it
        assert ("--surface-contact-allowance-mm" in flags) is (not a and base != FB)
        assert ("--surface-removed-sign-free" in flags) is b
        # T97 / palm cover stay off
        assert not any(t.startswith(("--surface-crease-protrusion", "--surface-crease-hand-freeze", "--palm-cover"))
                       for t in flags)
    # fb has neither hull fill nor allowance (no keep-last / no-ball) -- A cannot sit on it, so only fb-B exists
    assert f"{FB}-A" not in pf.AG3S_VERSIONS and f"{FB}-AB" not in pf.AG3S_VERSIONS


def test_version_handling_matches_the_base(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    for v, (base, _a, _b) in SL9.items():
        assert pf.is_final_version(v) and v not in pf.FINAL_CONFIG_VERSIONS
        assert pf.version_server_argv(v) == pf.version_server_argv(base)             # final base argv (no pre-merge pins)
        if base == P0M:
            assert v in pf.P0M_SL9_VERSIONS and "--hsr6-probe" not in pf.version_server_flags(v)
            with pytest.raises(ValueError, match="no HSR6 rule"):
                pf.version_server_flags(v, hsr6_probe="/p.npz")
        else:
            assert v in pf.PC_LATCH_VERSIONS and pf.version_server_flags(v)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]


@pytest.mark.parametrize("v", sorted(SL9))
def test_each_version_builds_with_the_esdf_keys(v, gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps

    _probe_or_skip(v)
    base, a, b = SL9[v]
    args = _args(v, monkeypatch)
    ps.check_unported_flags(args, None)                                              # no serve_safe refusal
    served, info = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
    e = served._policy.ag3s.config.esdf
    assert e.target_extent == "attention_surface" and e.surface_removed_sign_free is b
    assert e.surface_solid_owned is a
    if a:   # the values reach the AG3S config of the Panda server (build survival check) and the server record
        assert (e.surface_solid_protect_mm, e.surface_solid_dilate_mm, e.surface_solid_shadow_mm) == (5.0, 0.0, 80.0)
        assert e.surface_solid_dn_ever_target is True and e.surface_solid_dn_support == "exclude"
        assert not e.surface_hull_fill and e.surface_contact_allowance_mm is None
        te = info["ag3s"]["target_exclusion"]
        assert te["surface_solid_owned"] is True and te["surface_solid_shadow_mm"] == 80.0
    else:
        assert e.surface_solid_shadow_mm is None and not e.surface_solid_dn_ever_target
        assert e.surface_hull_fill is (base == P0M or base == PCS)
        assert (e.surface_contact_allowance_mm == 10.0) is (base == P0M or base == PCS)
    assert info["ag3s"].get("target_exclusion", {}).get("surface_removed_sign_free", False) is b
    # the version name is read back from the flags
    from AG3S.adapters.panda import profile as pf

    assert pf.match_ag3s_version(args) == v


# --- the base versions are unchanged -------------------------------------------------------------------------------
def test_base_versions_unchanged(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    assert pf.AG3S_VERSIONS[P0M] == pf.P0M_FLAGS and pf.AG3S_VERSIONS[PCS] == pf.PC_SURFACE_FLAGS
    assert pf.AG3S_VERSIONS[FB] == pf.PC_SURFACE_FALLBACK_FLAGS
    for base in (P0M, PCS, FB):
        assert not any("solid" in t or "sign-free" in t for t in pf.AG3S_VERSIONS[base])
        assert base not in pf.SL9_VERSIONS
    assert pf.P0M_VERSIONS.count(P0M) == 1 and not set(pf.SL9_VERSIONS) & set(pf.P0M_VERSIONS)
    # recorded flag values: the SL9 dests appear only on an SL9 version, the older records keep their keys
    sl9_keys = {"surface_solid_owned", "surface_solid_protect_mm", "surface_solid_dilate_mm", "surface_solid_shadow_mm",
                "surface_solid_dn_ever_target", "surface_removed_sign_free"}   # (dn_support is set by no version: default)
    for base in (P0M, PCS, FB):
        _probe_or_skip(base)
        assert not sl9_keys & set(pf.version_flag_values(_args(base, monkeypatch)))
    for v in SL9:
        _probe_or_skip(v)
        vals = pf.version_flag_values(_args(v, monkeypatch))
        assert sl9_keys <= set(vals)
        assert vals["surface_removed_sign_free"] is SL9[v][2] and vals["surface_solid_owned"] is SL9[v][1]


# --- the `panda/surface` record -------------------------------------------------------------------------------------
def _safe(frame, tf=None, extent="attention_surface"):
    esdf = types.SimpleNamespace(target_extent=extent)
    field = None if tf is None else types.SimpleNamespace(stats={"target_free": tf})
    ag3s = types.SimpleNamespace(config=types.SimpleNamespace(esdf=esdf), _surface_frame=frame, _last_esdf_field=field)
    return types.SimpleNamespace(ag3s=ag3s)


SOLID = {"applied": True, "volume_cm3": 12.5, "n_projected": 40, "n_vertices": 9, "protect_mm": 5.0, "dilate_mm": 0.0,
         "region_source": "this_request", "why": None, "ms": 1.2,
         "shadow": {"applied": True, "shadow_mm": 80.0, "cameras": {"head": 30, "left_wrist": 12}, "box_lo_m": [0, 0, 0]},
         "dn_ever_target": {"applied": True, "n_ever_target_cells": 3},
         "voxels": {"n_removed_surface": 4, "n_removed_band": 5, "n_removed_solid": 100, "n_removed_dilate": 0,
                    "n_removed_shadow": 7, "n_removed_total": 116, "n_protected_tau": 6,
                    "n_removed_shadow_by_camera": {"head": 7, "left_wrist": 0}, "dn_support_plane": True}}
FRAME = {"applied": True, "t78": {}}


def test_record_is_unchanged_without_the_new_records():
    from AG3S.adapters.panda.server import surface_call_record

    # flags off: the field's stats carry no solid / sign-free record -> exactly the SL7i record
    tf = {"tiers": [{"name": "fine", "voxel_size": 0.005}], "attention_surface": {"applied": True},
          "oracle_split": {"n_candidates": 3}, "geometry_version": {"request_seq": 4}}
    assert surface_call_record(_safe(FRAME, tf)) == {"applied": True, "state": "surface"}
    assert surface_call_record(_safe(FRAME, None)) == {"applied": True, "state": "surface"}
    assert surface_call_record(_safe(None, tf)) == {"state": "none"}
    assert surface_call_record(_safe(FRAME, tf, extent="estimated")) is None


def test_record_has_the_solid_summary():
    from AG3S.adapters.panda.server import surface_call_record

    tf = {"tiers": [{"name": "fine", "voxel_size": 0.004}], "attention_surface": {"applied": True, "solid": SOLID},
          "geometry_version": {"request_seq": 9}}
    rec = surface_call_record(_safe(FRAME, tf))
    assert "removed_sign_free" not in rec and rec["hand_voxel_m"] == 0.004 and rec["field_request_seq"] == 9
    s = rec["solid"]
    assert s["applied"] is True and s["volume_cm3"] == 12.5 and s["n_projected"] == 40
    assert s["shadow"]["target_px_by_camera"] == {"head": 30, "left_wrist": 12} and s["shadow"]["shadow_mm"] == 80.0
    assert "cameras" not in s["shadow"] and "box_lo_m" not in s["shadow"]
    assert s["dn_ever_target"] == {"applied": True, "n_ever_target_cells": 3}
    v = s["voxels"]
    assert v["n_removed_solid"] == 100 and v["n_removed_shadow"] == 7 and v["n_protected_tau"] == 6
    assert v["n_removed_shadow_by_camera"] == {"head": 7, "left_wrist": 0}


def test_record_has_the_sign_free_summary_and_is_scalar_only():
    import json

    import numpy as np
    from AG3S.adapters.panda.server import surface_call_record

    rsf = {"applied": True, "n_removed_listed": 50, "n_flipped": 11, "n_flipped_within_purity": 9, "n_zero_left": 0,
           "purity_radius_m": 0.0075, "junk": np.zeros(3)}
    tf = {"tiers": [{"name": "fine", "voxel_size": 0.005}], "oracle_split": {"removed_sign_free": rsf}}
    rec = surface_call_record(_safe(FRAME, tf))
    assert rec["removed_sign_free"] == {"applied": True, "n_removed_listed": 50, "n_flipped": 11,
                                        "n_flipped_within_purity": 9, "n_zero_left": 0, "purity_radius_m": 0.0075}
    assert "solid" not in rec and rec["hand_voxel_m"] == 0.005 and "field_request_seq" not in rec
    json.dumps(rec)
    # a stats dict that is not a dict / a solid that is not a dict never raises
    assert surface_call_record(_safe(FRAME, {"attention_surface": {"solid": 3}, "oracle_split": 5, "tiers": "x"})) == {
        "applied": True, "state": "surface"}
    assert surface_call_record(types.SimpleNamespace(ag3s=types.SimpleNamespace(
        config=types.SimpleNamespace(esdf=types.SimpleNamespace(target_extent="attention_surface")),
        _surface_frame=FRAME, _last_esdf_field=types.SimpleNamespace(stats=None)))) == {"applied": True, "state": "surface"}
