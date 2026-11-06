"""SL15 -- the versions ``-pc-ntm`` and ``-pc-openrule-ntm`` (``--nonauthorized-target-margin obstacle``, T44e A9). CPU only.

(a) Off is unchanged: the server argv of every version registered before these two is bit-identical (path-independent sha256 measured at
    816f7aa4, before the registration), the new dest joins a record only when set, the server config keeps its keys.
(b) The versions are the pc / openrule flags + the one option; the flag reaches AG3S as ``contact.nonauthorized_target_margin``.
(c) ``serve_safe``'s parser and its checks (``reject_bad_flag_combinations``, ``reject_panda_only_flags``) and the Panda server's
    ``check_unported_flags`` / ``check_record_flags`` accept the argv; the launchers and ``eval_matrix`` accept the names.
(d) The semantics on the real code: which robot rows are "nonauthorized" and the margins they get (``full`` 50 mm / ``obstacle`` 10 mm).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: F401 (fixture)
from tests.benchmark.test_sl12_oracle_panda import MI, PC, _args
from tests.benchmark.test_sl13_oracle_volume import path_independent_argv

NTM = f"{PC}-ntm"
ORN = f"{PC}-openrule-ntm"
OR = f"{PC}-openrule"
FLAG = ["--nonauthorized-target-margin", "obstacle"]
#: sha256 of ``path_independent_argv`` over the 66 versions registered at 816f7aa4 (before ``-ntm`` / ``-openrule-ntm``), measured there.
EARLIER_ARGV_SHA256 = "213654d423fcc59750a4f592a926e3a2c657798c0e0b077cd92fe8dbf123163d"


# ---------------------------------------------------------------------------------------------------- (a) off is unchanged
def test_every_earlier_versions_server_argv_is_bit_identical():
    from AG3S.adapters.panda import profile as pf

    old = {v: a for v, a in path_independent_argv(pf).items() if v not in pf.SL15_NTM_VERSIONS and v not in pf.SL15_SFHAND_VERSIONS and v not in pf.SL16_HSC_VERSIONS and v not in pf.SL16_HVC_VERSIONS and not pf.is_gtto_version(v)}   # 2026-11-06 merge: SL-GTTO versions came later
    assert len(old) == 66
    assert hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest() == EARLIER_ARGV_SHA256
    assert not any("--nonauthorized-target-margin" in a for a in old.values())
    assert pf.SL15_NTM_VERSIONS == (NTM, ORN) and not set(pf.SL15_NTM_VERSIONS) & set(pf.PC_LATCH_VERSIONS)
    assert not any(v.endswith("-lsfork") for v in pf.SL15_NTM_VERSIONS)
    assert not set(pf.SL15_NTM_VERSIONS) & set(pf.SL14_VERSIONS)


def test_off_by_default_in_parser_section_and_config():
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    B = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]
    off, on = ap.parse_args(B), ap.parse_args(B + FLAG)
    assert off.nonauthorized_target_margin is None and on.nonauthorized_target_margin == "obstacle"
    assert ss.nonauthorized_target_margin_section(off) == {}
    assert ss.nonauthorized_target_margin_section(on) == {"nonauthorized_target_margin": "obstacle"}
    assert AG3SConfig().contact.nonauthorized_target_margin == "full"


# ---------------------------------------------------------------------------------------------------- (b) the versions
def test_the_versions_are_the_base_flags_plus_the_one_option(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf, server as ps

    reg = pf.AG3S_VERSIONS
    assert list(reg[NTM]) == list(reg[PC]) + FLAG and list(reg[ORN]) == list(reg[OR]) + FLAG
    assert list(reg[ORN]) == list(reg[PC]) + ["--grasp-opening-rule"] + FLAG
    for v, base in ((NTM, PC), (ORN, OR)):
        assert pf.is_final_version(v) and pf.version_server_argv(v) == pf.version_server_argv(PC)
        assert pf.version_server_flags(v)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]          # a probe-taking pc variant
        assert pf.version_server_flags(v, hsr6_probe="/p.npz")[-2:] == ["--hsr6-probe", "/p.npz"]
        assert not pf.needs_oracle_grasp(v) and not pf.needs_oracle_segmentation(v) and f"{v}-lsfork" not in reg
        a, b = _args(v, monkeypatch), _args(base, monkeypatch)
        ps.check_unported_flags(a, None)
        assert a.nonauthorized_target_margin == "obstacle" and b.nonauthorized_target_margin is None
        assert pf.match_ag3s_version(a) == v and pf.match_ag3s_version(b) == base
        vals, bvals = pf.version_flag_values(a), pf.version_flag_values(b)
        assert vals["nonauthorized_target_margin"] == "obstacle"
        assert {k: x for k, x in vals.items() if k != "nonauthorized_target_margin"} == bvals     # nothing else joined
        assert "nonauthorized_target_margin" not in bvals
    # no earlier version's record gains the key (the pc / openrule / rearm / oraclelatch / mi10 records keep theirs)
    for v in (PC, OR, f"{PC}-rearm", f"{PC}-oraclelatch", MI, f"{PC}-gap", f"{PC}-oraclegt", f"{PC}-oraclevol"):
        assert "nonauthorized_target_margin" not in pf.version_flag_values(_args(v, monkeypatch))


def test_the_flag_is_ported_on_the_panda_server_and_nothing_else_moved():
    from AG3S.adapters.panda import server as ps

    assert "nonauthorized_target_margin" in ps.PORTED_FINAL_CONFIG_DESTS
    assert "nonauthorized_target_margin" not in ps.UNPORTED_FINAL_CONFIG_DESTS
    assert not set(ps.PORTED_FINAL_CONFIG_DESTS) & set(ps.UNPORTED_FINAL_CONFIG_DESTS)
    # the Panda server still refuses the neighbouring T44e options it has not ported
    assert {"target_free_coarse_support", "target_free_coarse"} <= set(ps.UNPORTED_FINAL_CONFIG_DESTS)


# ---------------------------------------------------------------------------------------------------- (c) validation without a server
@pytest.mark.parametrize("version", [NTM, ORN])
def test_serve_safe_parser_and_checks_accept_the_argv(version, gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf, server as ps
    from AG3S.trajopt import serve_safe as ss

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(version)
    argv[argv.index("--config") + 1] = "pi05_libero"
    argv += ["--robot", "panda"] + pf.version_server_flags(version)
    ap = ss.build_parser()
    args = ap.parse_args(argv)
    # the flags the margin could clash with are all on this argv
    assert args.target_permission_strict is True and args.obstacle_margin == pytest.approx(0.010)
    assert args.esdf_backend == "curobo" and args.esdf_margin == 0.0 and args.target_field_policy == "exclude-authorized"
    assert args.links == "arms" and args.nonauthorized_target_margin == "obstacle"
    # main()'s checks before the checkpoint is loaded
    ss.reject_bad_flag_combinations(ap, args)
    ss.reject_panda_only_flags(ap, args)
    ps.check_unported_flags(args, ap)
    ps.check_record_flags(args)
    # the RB-Y1 server (default --robot) takes the flag too: it is an RB-Y1 option, only the Panda server did not port it
    rby1 = ap.parse_args(["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"] + FLAG)
    ss.reject_panda_only_flags(ap, rby1)


def test_the_config_receives_obstacle_and_the_survival_check_holds(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    for v, want in ((NTM, "obstacle"), (ORN, "obstacle"), (PC, "full"), (OR, "full")):
        args = _args(v, monkeypatch)
        kw = pf.v11ab_build_kwargs(args)
        assert kw["contact"] == ({"nonauthorized_target_margin": "obstacle"} if want == "obstacle" else None)
        data, _ = pf.v11ab_config_dict(kw)
        assert ("contact" in data) is (want == "obstacle")
        cfg = pf.panda_ag3s_config(pf.v11ab_ag3s_config(kw))
        assert cfg.contact.nonauthorized_target_margin == want
        # the Panda contact links (palm + fingers) are what `panda_overrides` set, in both cases
        assert tuple(cfg.contact.contact_links["left"]) == ("panda_hand", "panda_leftfinger", "panda_rightfinger")
    # the real serve_safe section and the profile's copy agree
    a = _args(NTM, monkeypatch)
    assert ss.nonauthorized_target_margin_section(a) == pf.v11ab_build_kwargs(a)["contact"]


def test_build_panda_ag3s_keeps_the_key_and_the_info_only_when_on(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    on, info_on = pf.build_panda_ag3s(kw=pf.v11ab_build_kwargs(_args(NTM, monkeypatch)))
    off, info_off = pf.build_panda_ag3s(kw=pf.v11ab_build_kwargs(_args(PC, monkeypatch)))
    assert on.config.contact.nonauthorized_target_margin == "obstacle" and info_on["contact"] == {"nonauthorized_target_margin": "obstacle"}
    assert off.config.contact.nonauthorized_target_margin == "full" and "contact" not in info_off
    # everything else about the two servers' AG3S config is the same
    a, b = on.config.to_dict(), off.config.to_dict()
    a["contact"] = dict(a["contact"], nonauthorized_target_margin="full")
    assert a == b


def test_launchers_and_eval_matrix_accept_the_names(gpu_ok):
    from benchmark.runners import eval_matrix as em
    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_safety_paired as lsp

    for v in (NTM, ORN):
        got = lp.server_argv("vla_v11ab", "/ck", 9, v, hsr6_probe="/p.npz")
        assert got[-2:] == ["--hsr6-probe", "/p.npz"] and "--nonauthorized-target-margin" in got
        assert got[got.index("--nonauthorized-target-margin") + 1] == "obstacle" and got[got.index("--robot") + 1] == "panda"
        assert "--oracle-latch" not in got and "--placed-rearm" not in got and ("--grasp-opening-rule" in got) is (v == ORN)
        a = lp.build_parser().parse_args(["--task-suite-name", "libero_spatial", "--task-ids", "0", "--episode-ids", "0",
                                          "--run-id", "x", "--conversion", "C", "--seed", "7", "--ag3s-version", v])
        assert lp.oracle_client_argv("vla_v11ab", a) == [] and "oracle_grasp" not in lp.oracle_settings("vla_v11ab", a)
        wrapped = lsp.make_server_argv("pi05_libero_safety")("vla_v11ab", "/ck", 9, v)
        assert "--nonauthorized-target-margin" in wrapped and wrapped[wrapped.index("--robot") + 1] == "panda"
        # the launcher's own version check (main() before any GPU work) accepts it
        from AG3S.adapters.panda import profile

        profile.version_server_flags(v, **lp.version_options(a))
        jobs = em.build_jobs(suites=em.parse_suites("libero_goal,libero_spatial"), version=v, conversion="C", split="dev",
                             run_prefix="t", step="S")
        assert jobs and all(j["version"] == v and v in j["argv"] for j in jobs)
    # a name that is not registered is still refused
    with pytest.raises(ValueError):
        em.build_jobs(suites=[("libero_goal", None)], version=f"{PC}-ntm2", conversion="C", split="dev", run_prefix="t", step="S")


# ---------------------------------------------------------------------------------------------------- (d) the semantics
def _pc_constraint_model(monkeypatch):
    from AG3S.adapters.panda import model as pm, profile as pf

    kw = pf.v11ab_build_kwargs(_args(PC, monkeypatch))
    shell = {"palm_shell": dict(kw["palm_shell"])} if kw.get("palm_shell") else {}
    return pm.build_panda_models(kw["links"], **shell).constraint


def test_nonauthorized_rows_are_the_arm_links_one_to_seven(gpu_ok, monkeypatch):
    """With the pc flags (``--links arms``, palm shell on) the rows of ``panda_link1..7`` are nonauthorized; palm, shell and fingers are not."""
    from AG3S.adapters.panda import model as pm
    from AG3S.ag3s.constraints.clearance import ClearancePolicy
    from AG3S.ag3s.types import ContactPolicyContext
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.adapters.panda import profile as pf

    model = _pc_constraint_model(monkeypatch)
    names = [str(n) for n in model.sphere_link_names]
    cfg = pf.panda_ag3s_config(pf.v11ab_ag3s_config(pf.v11ab_build_kwargs(_args(NTM, monkeypatch))))
    policy = ClearancePolicy.from_config(cfg.contact, cfg.geometry, cfg.support_surface)
    authorized = policy.authorized_links(ContactPolicyContext.make("grasp", ["left"]))
    assert authorized == frozenset(pm.PANDA_CONTACT_LINKS["left"])
    rows = np.asarray([n not in authorized for n in names], bool)           # to_adapter.py: `str(n) not in authorized`
    assert {n for n, r in zip(names, rows) if r} == {f"panda_link{i}" for i in range(1, 8)}
    assert int(rows.sum()) == 37 and len(names) == 170                      # 4+4+4+4+13+3+5 of 170 rows
    assert not any(r for n, r in zip(names, rows) if n in authorized)
    # the same set under every phase that has an active left manipulator
    for phase in ("approach", "grasp", "transit"):
        assert policy.authorized_links(ContactPolicyContext.make(phase, ["left"])) == authorized


def test_full_margin_is_50_mm_and_obstacle_is_10_mm_for_those_rows(gpu_ok, monkeypatch):
    """The two margins a nonauthorized row whose nearest surface is the target gets in the ESDF path (``--esdf-margin 0``)."""
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.constraints.clearance import ClearancePolicy
    from AG3S.ag3s.types import ContactPolicyContext
    from AG3S.ag3s.types import SourceType
    from AG3S.adapters.panda import profile as pf

    args = _args(NTM, monkeypatch)
    cfg = pf.panda_ag3s_config(pf.v11ab_ag3s_config(pf.v11ab_build_kwargs(args)))
    policy = ClearancePolicy.from_config(cfg.contact, cfg.geometry, cfg.support_surface)
    model = _pc_constraint_model(monkeypatch)
    names = [str(n) for n in model.sphere_link_names]
    ctx = ContactPolicyContext.make("grasp", ["left"])
    # `full`: `manipulated_link_margin` = the TARGET column of margin_matrix -> `geometry.safety_margin` for a nonauthorized link
    col = policy.margin_matrix(names, [SourceType.TARGET], context=ctx, target_grounded=True)[:, 0]
    nonauth = np.asarray([n.startswith("panda_link") for n in names], bool)
    assert cfg.geometry.safety_margin == pytest.approx(0.05)
    assert np.allclose(col[nonauth], 0.05)
    # `obstacle`: not a target row any more -> the obstacle rows' rule: max(esdf_margin, obstacle_margin) = max(0, 0.010)
    assert args.esdf_margin == 0.0 and args.obstacle_margin == pytest.approx(0.010)
    assert max(args.esdf_margin, args.obstacle_margin) == pytest.approx(0.010)
    # the authorized rows are not touched by the option (their target margin is the phase rule, 0 in GRASP)
    assert np.all(col[~nonauth] < 0.05)


def test_build_constraint_set_sends_the_arm_link_rows_for_the_pc_config(gpu_ok, monkeypatch):
    """The real ``to_adapter.build_constraint_set`` with the Panda link names and the ClearancePolicy of the ``-pc-ntm`` config:
    ``nonauthorized_target_rows`` marks the 37 ``panda_link1..7`` rows; the ``full`` config sends none (margin numbers of the TrajOpt side:
    ``tests/trajopt/test_t44e_a9_nonauthorized_target_margin.py``, which pins 50 mm -> 10 mm with ``esdf_margin 0`` / ``obstacle_margin 0.010``)."""
    import types

    from AG3S.ag3s.config import ConstraintConfig
    from AG3S.ag3s.constraints.clearance import ClearancePolicy
    from AG3S.ag3s.constraints.to_adapter import build_constraint_set
    from AG3S.ag3s.types import ContactPolicyContext
    from AG3S.adapters.panda import profile as pf
    from tests.trajopt.test_t44e_a9_nonauthorized_target_margin import _target

    names = [str(n) for n in _pc_constraint_model(monkeypatch).sphere_link_names]

    def builder(version):
        cfg = pf.panda_ag3s_config(pf.v11ab_ag3s_config(pf.v11ab_build_kwargs(_args(version, monkeypatch))))
        return types.SimpleNamespace(
            build=None, max_candidates=32, max_support_surfaces=2, constraint_config=ConstraintConfig(target_field_policy="exclude_authorized"),
            robot_model=None, n_robot_spheres=len(names), clearance_policy=ClearancePolicy.from_config(cfg.contact, cfg.geometry, cfg.support_surface), sphere_link_names=names)

    kw = dict(candidates=[], support_surfaces=[], target=_target(), robot_state=np.zeros(2), phase="approach",
              contact_context=ContactPolicyContext.make("approach", ["left"]), build_spec=False)
    on = build_constraint_set(builder(NTM), **kw)
    assert {n for n, r in zip(names, on.nonauthorized_target_rows) if r} == {f"panda_link{i}" for i in range(1, 8)}
    assert on.metrics["target_permission"]["nonauthorized_target_margin"] == {"mode": "obstacle", "n_rows": 37, "applied": True}
    off = build_constraint_set(builder(PC), **kw)
    assert off.nonauthorized_target_rows is None and "nonauthorized_target_margin" not in off.metrics["target_permission"]
    assert np.array_equal(on.manipulated_link_margin, off.manipulated_link_margin)     # the permission rows are not touched
