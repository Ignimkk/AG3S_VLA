"""SL16 HVC -- the version ``-pc-openrule-hvc`` (``--hand-voxel-carve current --hand-voxel-carve-mm 20``) and the measurement behind its value. CPU only.

(a) Off is unchanged: the server argv of every version registered before it is bit-identical (path-independent sha256, measured at the
    base commit f16e4b38 and here with the new version left out); the new dests join a record only when set; the AG3S config dict has no
    new key (the switch is a pipeline attribute, like SL10d's).
(b) The version is the ``-pc-openrule`` flags + the two options; the options reach ``build_ag3s``'s keyword and the Panda AG3S.
(c) ``serve_safe``'s parser and checks and the Panda server's ``check_unported_flags`` / ``check_record_flags`` accept the argv; the
    launchers and ``eval_matrix`` accept the name; the RB-Y1 server refuses the flags.
(d) The measurement tool (``panda_hvc_origin``): its pure functions, and the committed data file reproduces the mode and the value by the
    fixed rule.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: F401 (fixture)
from tests.benchmark.test_sl12_oracle_panda import PC, _args
from tests.benchmark.test_sl13_oracle_volume import path_independent_argv

OR = f"{PC}-openrule"
HVC = f"{OR}-hvc"
FLAGS = ["--hand-voxel-carve", "current", "--hand-voxel-carve-mm", "20"]
#: sha256 of ``path_independent_argv`` over the 69 versions registered at f16e4b38 (before ``-hvc``), measured there.
EARLIER_ARGV_SHA256 = "568968bef41a8685b0982429b3ae6afa6d1544d0a18cbfd2242a8bae8dafcd3b"
DATA = pathlib.Path(__file__).resolve().parents[2] / "AG3S" / "ag3s" / "docs" / "handoff" / "SL16.hvc.data.json"


# ---------------------------------------------------------------------------------------------------- (a) off is unchanged
def test_every_earlier_versions_server_argv_is_bit_identical():
    from AG3S.adapters.panda import profile as pf

    old = {v: a for v, a in path_independent_argv(pf).items() if v not in pf.SL16_HVC_VERSIONS and v not in pf.SL16_HSC_VERSIONS and not pf.is_gtto_version(v)}   # 2026-11-06 merge: SL-GTTO versions came later
    assert len(old) == 69
    assert hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest() == EARLIER_ARGV_SHA256
    assert not any("--hand-voxel-carve" in a for a in old.values())
    assert pf.SL16_HVC_VERSIONS == (HVC,) and not set(pf.SL16_HVC_VERSIONS) & set(pf.PC_LATCH_VERSIONS)
    assert not any(v.endswith("-lsfork") for v in pf.SL16_HVC_VERSIONS) and f"{HVC}-lsfork" not in pf.AG3S_VERSIONS
    assert not set(pf.SL16_HVC_VERSIONS) & (set(pf.SL14_VERSIONS) | set(pf.SL15_NTM_VERSIONS) | set(pf.SL15_SFHAND_VERSIONS))


def test_off_by_default_in_parser_pipeline_and_config():
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    B = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]
    off, on = ap.parse_args(B), ap.parse_args(B + FLAGS)
    assert off.hand_voxel_carve == "off" and off.hand_voxel_carve_mm is None
    assert on.hand_voxel_carve == "current" and on.hand_voxel_carve_mm == 20.0
    assert ss.hand_voxel_carve_section(off) == {} and ss.hand_voxel_carve_section(on) == {"mode": "current", "radius_m": 0.020}
    assert AG3S(AG3SConfig()).hand_voxel_carve is None
    # no key in the AG3S config dict (the manifest's panda_config / the effective config): the switch is a pipeline attribute
    assert not any("hand_voxel" in k for k in AG3SConfig().to_dict()["esdf"])
    assert ss.dynamic_field_section(on) == {} and ss.target_exclusion_section(on) == {}


def test_the_ag3s_config_is_the_same_with_the_flags(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    a = pf.panda_ag3s_config(pf.v11ab_ag3s_config(pf.v11ab_build_kwargs(_args(HVC, monkeypatch)))).to_dict()
    b = pf.panda_ag3s_config(pf.v11ab_ag3s_config(pf.v11ab_build_kwargs(_args(OR, monkeypatch)))).to_dict()
    assert a == b


# ---------------------------------------------------------------------------------------------------- (b) the version
def test_the_version_is_the_openrule_flags_plus_the_two_options(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf, server as ps

    reg = pf.AG3S_VERSIONS
    assert list(reg[HVC]) == list(reg[OR]) + FLAGS and (pf.SL16_HVC_MODE, pf.SL16_HVC_MM) == ("current", 20.0)
    assert "--grasp-opening-rule" in reg[HVC] and "--hand-voxel-carve" not in reg[OR]
    assert pf.is_final_version(HVC) and pf.version_server_argv(HVC) == pf.version_server_argv(OR)
    assert pf.version_server_flags(HVC)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]
    assert not pf.needs_oracle_grasp(HVC) and not pf.needs_oracle_segmentation(HVC)
    a, b = _args(HVC, monkeypatch), _args(OR, monkeypatch)
    ps.check_unported_flags(a, None)
    assert (a.hand_voxel_carve, a.hand_voxel_carve_mm) == ("current", 20.0) and (b.hand_voxel_carve, b.hand_voxel_carve_mm) == ("off", None)
    assert pf.match_ag3s_version(a) == HVC and pf.match_ag3s_version(b) == OR
    vals, bvals = pf.version_flag_values(a), pf.version_flag_values(b)
    assert vals["hand_voxel_carve"] == "current" and vals["hand_voxel_carve_mm"] == 20.0
    assert {k: x for k, x in vals.items() if k not in ("hand_voxel_carve", "hand_voxel_carve_mm")} == bvals      # nothing else joined
    assert "hand_voxel_carve" not in bvals and "hand_voxel_carve_mm" not in bvals
    for v in pf.AG3S_VERSIONS:                                                       # no earlier version's record gains a key
        if v != HVC:
            got = pf.version_flag_values(_args(v, monkeypatch))
            assert "hand_voxel_carve" not in got and "hand_voxel_carve_mm" not in got, v


def test_the_flags_reach_the_build_keyword_and_the_panda_ag3s_only_when_on(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    kw_on, kw_off = pf.v11ab_build_kwargs(_args(HVC, monkeypatch)), pf.v11ab_build_kwargs(_args(OR, monkeypatch))
    assert kw_on["hand_voxel_carve"] == {"mode": "current", "radius_m": 0.020} and kw_off["hand_voxel_carve"] is None
    assert {k: v for k, v in kw_on.items() if k != "hand_voxel_carve"} == {k: v for k, v in kw_off.items() if k != "hand_voxel_carve"}
    on, info_on = pf.build_panda_ag3s(kw=kw_on)
    off, info_off = pf.build_panda_ag3s(kw=kw_off)
    assert on.hand_voxel_carve == {"mode": "current", "radius_m": 0.020} and off.hand_voxel_carve is None
    assert info_on["hand_voxel_carve"] == {"mode": "current", "radius_m": 0.020} and "hand_voxel_carve" not in info_off
    assert {k: v for k, v in info_on.items() if k != "hand_voxel_carve"} == info_off
    assert on.config.to_dict() == off.config.to_dict()
    # the Panda self-filter model has the three hand links the carve selects
    from AG3S.adapters.panda import model as pm

    links = set(pm.build_panda_self_filter_model().sphere_link_names)
    assert set(on.HAND_CARVE_LINKS) <= links


# ---------------------------------------------------------------------------------------------------- (c) validation without a server
def test_serve_safe_parser_and_checks_accept_the_argv_and_rby1_refuses(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf, server as ps
    from AG3S.trajopt import serve_safe as ss

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "D")
    argv = pf.version_server_argv(HVC)
    argv[argv.index("--config") + 1] = "pi05_libero"
    argv += ["--robot", "panda"] + pf.version_server_flags(HVC)
    ap = ss.build_parser()
    args = ap.parse_args(argv)
    assert args.links == "arms" and args.esdf_backend == "curobo" and args.hand_voxel_carve == "current"
    ss.reject_bad_flag_combinations(ap, args)
    ss.reject_panda_only_flags(ap, args)
    ps.check_unported_flags(args, ap)
    ps.check_record_flags(args)
    args.record_constraints = "/tmp/rec"            # the recording run (--record-constraints full) takes the version too
    ps.check_record_flags(args)
    assert not {"hand_voxel_carve", "hand_voxel_carve_mm"} & set(ps.UNPORTED_FINAL_CONFIG_DESTS)
    args.robot = "rby1"                              # the same flags on an RB-Y1 server
    with pytest.raises(SystemExit):
        ss.reject_panda_only_flags(ap, args)


def test_launchers_and_eval_matrix_accept_the_name(gpu_ok):
    from benchmark.runners import eval_matrix as em
    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_safety_paired as lsp

    got = lp.server_argv("vla_v11ab", "/ck", 9, HVC, hsr6_probe="/p.npz")
    assert got[-2:] == ["--hsr6-probe", "/p.npz"] and got[got.index("--hand-voxel-carve") + 1] == "current"
    assert got[got.index("--hand-voxel-carve-mm") + 1] == "20" and got[got.index("--robot") + 1] == "panda"
    assert "--grasp-opening-rule" in got and "--oracle-latch" not in got
    a = lp.build_parser().parse_args(["--task-suite-name", "libero_spatial", "--task-ids", "0", "--episode-ids", "0",
                                      "--run-id", "x", "--conversion", "D", "--seed", "7", "--ag3s-version", HVC])
    assert lp.oracle_client_argv("vla_v11ab", a) == [] and "oracle_grasp" not in lp.oracle_settings("vla_v11ab", a)
    wrapped = lsp.make_server_argv("pi05_libero_safety")("vla_v11ab", "/ck", 9, HVC)
    assert "--hand-voxel-carve" in wrapped and wrapped[wrapped.index("--robot") + 1] == "panda"
    from AG3S.adapters.panda import profile

    profile.version_server_flags(HVC, **lp.version_options(a))
    jobs = em.build_jobs(suites=em.parse_suites("libero_goal,libero_spatial"), version=HVC, conversion="D", split="dev",
                         run_prefix="t", step="S")
    assert jobs and all(j["version"] == HVC and HVC in j["argv"] for j in jobs)
    with pytest.raises(ValueError):
        em.build_jobs(suites=[("libero_goal", None)], version=f"{HVC}2", conversion="D", split="dev", run_prefix="t", step="S")


# ---------------------------------------------------------------------------------------------------- (d) the measurement tool
def test_classify_and_decide_pure_functions():
    from AG3S.trajopt.experiments import panda_hvc_origin as T

    tol = 0.004
    sd_cur = np.array([0.001, 0.0039, 0.0041, 0.010, 0.030, -0.005])
    sd_trail = np.array([np.inf, 0.001, 0.002, 0.020, 0.0045, np.inf])
    assert T.classify_hand_voxels(sd_cur, sd_trail, tol).tolist() == [0, 0, 1, 2, 2, 0]      # current wins over trail; neither = smear
    assert T.decide_mode(0.49, 0.0236)["mode"] == "current" and T.decide_mode(0.49, 0.0236)["r_mm"] == 20.0     # capped
    assert T.decide_mode(0.49, 0.0123)["r_mm"] == pytest.approx(12.3)
    t = T.decide_mode(0.5, 0.0236)
    assert t["mode"] == "trail" and t["r_mm"] is None and T.decide_mode(0.4999, 0.01)["mode"] == "current"


def test_the_committed_data_gives_the_registered_mode_and_value():
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt.experiments import panda_hvc_origin as T

    d = json.loads(DATA.read_text())
    main = d["tables"]["closing_and_held"]
    shares = {k: main[k]["share"] for k in ("current", "trail", "smear")}
    assert sum(shares.values()) == pytest.approx(1.0) and sum(main[k]["n"] for k in shares) == main["n_hand_voxels"]
    assert d["tol_mm"] == 4.0 and d["n_prev"] == 8 and d["reach_mm"] == 220.0
    # the fixed rule, recomputed from the table
    rule = T.decide_mode(shares["trail"], main["sd_cur_all_mm"]["p90"] / 1000.0)
    assert rule == d["decision"]
    assert (rule["mode"], rule["r_mm"]) == (pf.SL16_HVC_MODE, pf.SL16_HVC_MM)
    assert shares["trail"] < 0.5 and rule["hand_voxel_sd_p90_mm"] > 20.0                   # the cap binds
    assert d["rby1_comparison"].startswith("기록 없음")
    assert d["n_calls_measured"] <= d["n_calls_closing_held_with_chunk"]
