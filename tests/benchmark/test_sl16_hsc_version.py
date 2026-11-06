"""SL16 HSC -- the version ``-pc-openrule-hsc`` (``--held-support-contact``). CPU only.

(a) Off is unchanged: the server argv of every version registered before it is bit-identical (path-independent sha256 measured at
    f16e4b38, before the registration); the new dest joins a record only when set.
(b) The version is the ``-pc-openrule`` flags + the one flag; the Panda server's checks accept it; the launchers and ``eval_matrix``
    accept the name; no earlier version's record gains the key.
(c) The RB-Y1 server refuses the flag (``reject_panda_only_flags``).
"""
from __future__ import annotations

import hashlib
import json

import pytest

from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: F401 (fixture)
from tests.benchmark.test_sl12_oracle_panda import PC, _args
from tests.benchmark.test_sl13_oracle_volume import path_independent_argv

OR = f"{PC}-openrule"
HSC = f"{OR}-hsc"
FLAG = ["--held-support-contact"]
#: sha256 of ``path_independent_argv`` over the 69 versions registered at f16e4b38 (before ``-hsc``), measured there.
EARLIER_ARGV_SHA256 = "568968bef41a8685b0982429b3ae6afa6d1544d0a18cbfd2242a8bae8dafcd3b"


# ---------------------------------------------------------------------------------------------------- (a) off is unchanged
def test_every_earlier_versions_server_argv_is_bit_identical():
    from AG3S.adapters.panda import profile as pf

    old = {v: a for v, a in path_independent_argv(pf).items() if v not in pf.SL16_HSC_VERSIONS}
    assert len(old) == 69
    assert hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest() == EARLIER_ARGV_SHA256
    assert not any("--held-support-contact" in a for a in old.values())
    assert pf.SL16_HSC_VERSIONS == (HSC,) and not set(pf.SL16_HSC_VERSIONS) & set(pf.PC_LATCH_VERSIONS)
    assert not any(v.endswith("-lsfork") for v in pf.SL16_HSC_VERSIONS) and f"{HSC}-lsfork" not in pf.AG3S_VERSIONS
    assert not set(pf.SL16_HSC_VERSIONS) & (set(pf.SL14_VERSIONS) | set(pf.SL15_NTM_VERSIONS) | set(pf.SL15_SFHAND_VERSIONS))


def test_off_by_default_in_the_parser():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    B = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]
    off, on = ap.parse_args(B), ap.parse_args(B + FLAG)
    assert off.held_support_contact is False and on.held_support_contact is True
    assert off.held_support_contact_finger_mm is None and on.held_support_contact_finger_mm is None      # the default 10 is applied, not parsed


# ---------------------------------------------------------------------------------------------------- (b) the version
def test_the_version_is_the_openrule_flags_plus_the_one_flag(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf, server as ps

    reg = pf.AG3S_VERSIONS
    assert list(reg[HSC]) == list(reg[OR]) + FLAG and "--grasp-opening-rule" in reg[HSC]
    assert pf.is_final_version(HSC) and pf.version_server_argv(HSC) == pf.version_server_argv(OR)
    assert pf.version_server_flags(HSC)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]
    assert pf.version_server_flags(HSC, hsr6_probe="/p.npz")[-2:] == ["--hsr6-probe", "/p.npz"]
    assert not pf.needs_oracle_grasp(HSC) and not pf.needs_oracle_segmentation(HSC)
    a, b = _args(HSC, monkeypatch), _args(OR, monkeypatch)
    ps.check_unported_flags(a, None)
    assert a.held_support_contact is True and b.held_support_contact is False and a.grasp_opening_rule is True
    assert pf.match_ag3s_version(a) == HSC and pf.match_ag3s_version(b) == OR
    vals, bvals = pf.version_flag_values(a), pf.version_flag_values(b)
    assert vals["held_support_contact"] is True
    assert {k: x for k, x in vals.items() if k != "held_support_contact"} == bvals          # nothing else joined
    assert "held_support_contact" not in bvals
    for v in pf.AG3S_VERSIONS:                                                              # no earlier record gains the key
        if v != HSC:
            assert "held_support_contact" not in pf.version_flag_values(_args(v, monkeypatch)), v


def test_serve_safe_parser_and_checks_accept_the_argv(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf, server as ps
    from AG3S.trajopt import serve_safe as ss

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(HSC)
    argv[argv.index("--config") + 1] = "pi05_libero"
    argv += ["--robot", "panda"] + pf.version_server_flags(HSC)
    ap = ss.build_parser()
    args = ap.parse_args(argv)
    assert args.held_support_contact and args.grasp_opening_rule and args.links == "arms" and args.palm_shell == "on"
    ss.reject_bad_flag_combinations(ap, args)
    ss.reject_panda_only_flags(ap, args)
    ps.check_unported_flags(args, ap)
    ps.check_record_flags(args)
    args.record_constraints = "/tmp/rec"
    ps.check_record_flags(args)
    assert "held_support_contact" not in ps.UNPORTED_FINAL_CONFIG_DESTS and "held_support_contact_finger_mm" not in ps.UNPORTED_FINAL_CONFIG_DESTS


def test_the_panda_server_applies_the_flag_through_apply_grasp_contact_flags(gpu_ok, monkeypatch):
    """``build_safe_server`` calls ``apply_grasp_contact_flags(safe, args)`` (server.py); this is the step that arms the SafePolicy."""
    from AG3S.adapters.panda import server as ps
    import inspect

    src = inspect.getsource(ps.build_safe_server)
    assert "ss.apply_grasp_contact_flags(safe, args)" in src


def test_launchers_and_eval_matrix_accept_the_name(gpu_ok):
    from benchmark.runners import eval_matrix as em
    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_safety_paired as lsp

    got = lp.server_argv("vla_v11ab", "/ck", 9, HSC, hsr6_probe="/p.npz")
    assert got[-2:] == ["--hsr6-probe", "/p.npz"] and "--held-support-contact" in got and "--grasp-opening-rule" in got
    assert got[got.index("--robot") + 1] == "panda" and "--oracle-latch" not in got
    a = lp.build_parser().parse_args(["--task-suite-name", "libero_spatial", "--task-ids", "0", "--episode-ids", "0",
                                      "--run-id", "x", "--conversion", "C", "--seed", "7", "--ag3s-version", HSC])
    assert lp.oracle_client_argv("vla_v11ab", a) == [] and "oracle_grasp" not in lp.oracle_settings("vla_v11ab", a)
    wrapped = lsp.make_server_argv("pi05_libero_safety")("vla_v11ab", "/ck", 9, HSC)
    assert "--held-support-contact" in wrapped and wrapped[wrapped.index("--robot") + 1] == "panda"
    from AG3S.adapters.panda import profile

    profile.version_server_flags(HSC, **lp.version_options(a))
    jobs = em.build_jobs(suites=em.parse_suites("libero_goal,libero_spatial"), version=HSC, conversion="C", split="dev",
                         run_prefix="t", step="S")
    assert jobs and all(j["version"] == HSC and HSC in j["argv"] for j in jobs)
    with pytest.raises(ValueError):
        em.build_jobs(suites=[("libero_goal", None)], version=f"{HSC}2", conversion="C", split="dev", run_prefix="t", step="S")


# ---------------------------------------------------------------------------------------------------- (c) RB-Y1 refuses
def test_the_rby1_server_refuses_the_flag_even_with_a_final_version_argv(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(HSC)
    argv[argv.index("--config") + 1] = "pi05_libero"
    argv += pf.version_server_flags(HSC)                           # no --robot panda: the RB-Y1 server's view of the same argv
    ap = ss.build_parser()
    args = ap.parse_args(argv)
    assert getattr(args, "robot", "rby1") == "rby1"
    with pytest.raises(SystemExit):
        ss.reject_panda_only_flags(ap, args)
