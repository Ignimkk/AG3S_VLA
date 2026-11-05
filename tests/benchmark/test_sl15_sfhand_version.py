"""SL15 sfhand -- the version ``-pc-sfhand`` (``--self-filter-inflation-link panda_hand=<m>``) and the leak measurement behind its value. CPU only.

(a) Off is unchanged: the server argv of every version registered before it is bit-identical (path-independent sha256 measured at
    5bccd363, before the registration); the new dest joins a record only when set; the server config keeps its keys.
(b) The version is the ``-pc`` flags + the one option; the option reaches the AG3S config and the self-filter's per-sphere table.
(c) ``serve_safe``'s parser and its checks and the Panda server's ``check_unported_flags`` / ``check_record_flags`` accept the argv; the
    launchers and ``eval_matrix`` accept the name.
(d) The Panda self-filter model has the ``panda_hand`` link (the flag would otherwise raise) and does **not** contain the palm shell.
(e) The measurement tool (``panda_sfhand_leak``): its pure functions, and the committed data file reproduces the value by the fixed rule.
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

SF = f"{PC}-sfhand"
VALUE_M = 0.004
FLAG = ["--self-filter-inflation-link", "panda_hand=0.004"]
#: sha256 of ``path_independent_argv`` over the 68 versions registered at 5bccd363 (before ``-sfhand``), measured there.
EARLIER_ARGV_SHA256 = "91fe46e49b6b8cb4c98ee7e8752efff438a9625bc49281a97d587d4e71a56069"
DATA = pathlib.Path(__file__).resolve().parents[2] / "AG3S" / "ag3s" / "docs" / "handoff" / "SL15.sfhand.data.json"


# ---------------------------------------------------------------------------------------------------- (a) off is unchanged
def test_every_earlier_versions_server_argv_is_bit_identical():
    from AG3S.adapters.panda import profile as pf

    old = {v: a for v, a in path_independent_argv(pf).items() if v not in pf.SL15_SFHAND_VERSIONS}
    assert len(old) == 68
    assert hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest() == EARLIER_ARGV_SHA256
    assert not any("--self-filter-inflation-link" in a for a in old.values())
    assert pf.SL15_SFHAND_VERSIONS == (SF,) and not set(pf.SL15_SFHAND_VERSIONS) & set(pf.PC_LATCH_VERSIONS)
    assert not any(v.endswith("-lsfork") for v in pf.SL15_SFHAND_VERSIONS) and f"{SF}-lsfork" not in pf.AG3S_VERSIONS
    assert not set(pf.SL15_SFHAND_VERSIONS) & (set(pf.SL14_VERSIONS) | set(pf.SL15_NTM_VERSIONS))


def test_off_by_default_in_parser_options_and_config():
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    B = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]
    off, on = ap.parse_args(B), ap.parse_args(B + FLAG)
    assert tuple(off.self_filter_inflation_link) == () and list(on.self_filter_inflation_link) == ["panda_hand=0.004"]
    assert ss.self_filter_options(off) == {}
    assert ss.self_filter_options(on) == {"self_filter_inflation_by_link": {"panda_hand": VALUE_M}}
    cfg = AG3SConfig()
    assert cfg.pointcloud.self_filter_inflation == 0.0 and cfg.pointcloud.self_filter_inflation_by_link == {}


# ---------------------------------------------------------------------------------------------------- (b) the version
def test_the_version_is_the_pc_flags_plus_the_one_option(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf, server as ps

    reg = pf.AG3S_VERSIONS
    assert list(reg[SF]) == list(reg[PC]) + FLAG and pf.SL15_SFHAND_PANDA_HAND_M == VALUE_M
    assert pf.is_final_version(SF) and pf.version_server_argv(SF) == pf.version_server_argv(PC)
    assert pf.version_server_flags(SF)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]          # a probe-taking pc variant
    assert pf.version_server_flags(SF, hsr6_probe="/p.npz")[-2:] == ["--hsr6-probe", "/p.npz"]
    assert not pf.needs_oracle_grasp(SF) and not pf.needs_oracle_segmentation(SF)
    a, b = _args(SF, monkeypatch), _args(PC, monkeypatch)
    ps.check_unported_flags(a, None)
    assert list(a.self_filter_inflation_link) == ["panda_hand=0.004"] and tuple(b.self_filter_inflation_link) == ()
    assert pf.match_ag3s_version(a) == SF and pf.match_ag3s_version(b) == PC
    vals, bvals = pf.version_flag_values(a), pf.version_flag_values(b)
    assert vals["self_filter_inflation_link"] == ["panda_hand=0.004"]
    assert {k: x for k, x in vals.items() if k != "self_filter_inflation_link"} == bvals       # nothing else joined
    assert "self_filter_inflation_link" not in bvals
    # no earlier version's record gains the key
    for v in pf.AG3S_VERSIONS:
        if v != SF:
            assert "self_filter_inflation_link" not in pf.version_flag_values(_args(v, monkeypatch)), v


def test_the_config_receives_the_per_link_table_and_the_survival_check_holds(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import model as pm, profile as pf
    from AG3S.ag3s.stages.robot_filter import resolve_self_filter_inflation

    kw = pf.v11ab_build_kwargs(_args(SF, monkeypatch))
    assert kw["self_filter"] == {"self_filter_inflation_by_link": {"panda_hand": VALUE_M}}
    assert pf.v11ab_build_kwargs(_args(PC, monkeypatch))["self_filter"] == {}
    data, _ = pf.v11ab_config_dict(kw)
    assert data["pointcloud"]["self_filter_inflation_by_link"] == {"panda_hand": VALUE_M}
    cfg = pf.panda_ag3s_config(pf.v11ab_ag3s_config(kw))
    assert cfg.pointcloud.self_filter_inflation_by_link == {"panda_hand": VALUE_M} and cfg.pointcloud.self_filter_inflation == 0.0
    # the per-sphere table that reaches `robot_sphere_mask`: panda_hand spheres only, everything else (fingers included) stays 0
    model = pm.build_panda_self_filter_model()
    table = np.asarray(resolve_self_filter_inflation(model.sphere_link_names, cfg.pointcloud))
    links = np.asarray(model.sphere_link_names)
    assert table.shape == (model.n_spheres,) and int((links == "panda_hand").sum()) > 0
    assert np.all(table[links == "panda_hand"] == VALUE_M) and np.all(table[links != "panda_hand"] == 0.0)
    assert np.isscalar(resolve_self_filter_inflation(model.sphere_link_names, pf.panda_ag3s_config(pf.v11ab_ag3s_config(
        pf.v11ab_build_kwargs(_args(PC, monkeypatch)))).pointcloud))                  # off: the old scalar, same arithmetic


def test_build_panda_ag3s_keeps_the_key_and_the_info_only_when_on(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    on, info_on = pf.build_panda_ag3s(kw=pf.v11ab_build_kwargs(_args(SF, monkeypatch)))
    off, info_off = pf.build_panda_ag3s(kw=pf.v11ab_build_kwargs(_args(PC, monkeypatch)))
    assert on.config.pointcloud.self_filter_inflation_by_link == {"panda_hand": VALUE_M}
    assert info_on["self_filter"]["asked"] == {"self_filter_inflation_by_link": {"panda_hand": VALUE_M}}
    eff = info_on["self_filter"]["applied"]["inflation_effective"]
    assert eff["panda_hand"] == VALUE_M and eff["panda_leftfinger"] == 0.0 and eff["panda_rightfinger"] == 0.0 and eff["panda_link7"] == 0.0
    assert off.config.pointcloud.self_filter_inflation_by_link == {} and "self_filter" not in info_off
    # everything else about the two servers' AG3S config is the same
    a, b = on.config.to_dict(), off.config.to_dict()
    a["pointcloud"] = dict(a["pointcloud"], self_filter_inflation_by_link=b["pointcloud"]["self_filter_inflation_by_link"])
    assert a == b


# ---------------------------------------------------------------------------------------------------- (c) validation without a server
def test_serve_safe_parser_and_checks_accept_the_argv(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf, server as ps
    from AG3S.trajopt import serve_safe as ss

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(SF)
    argv[argv.index("--config") + 1] = "pi05_libero"
    argv += ["--robot", "panda"] + pf.version_server_flags(SF)
    ap = ss.build_parser()
    args = ap.parse_args(argv)
    assert args.links == "arms" and args.palm_shell == "on" and list(args.self_filter_inflation_link) == ["panda_hand=0.004"]
    ss.reject_bad_flag_combinations(ap, args)
    ss.reject_panda_only_flags(ap, args)
    ps.check_unported_flags(args, ap)
    ps.check_record_flags(args)
    args.record_constraints = "/tmp/rec"            # the DREC-style recording run (--record-constraints full) takes the version too
    ps.check_record_flags(args)
    assert "self_filter_inflation_link" not in ps.UNPORTED_FINAL_CONFIG_DESTS and "self_filter_target_guard" not in ps.UNPORTED_FINAL_CONFIG_DESTS


def test_launchers_and_eval_matrix_accept_the_name(gpu_ok):
    from benchmark.runners import eval_matrix as em
    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_safety_paired as lsp

    got = lp.server_argv("vla_v11ab", "/ck", 9, SF, hsr6_probe="/p.npz")
    assert got[-2:] == ["--hsr6-probe", "/p.npz"] and got[got.index("--self-filter-inflation-link") + 1] == "panda_hand=0.004"
    assert got[got.index("--robot") + 1] == "panda" and "--oracle-latch" not in got and "--nonauthorized-target-margin" not in got
    a = lp.build_parser().parse_args(["--task-suite-name", "libero_spatial", "--task-ids", "0", "--episode-ids", "0",
                                      "--run-id", "x", "--conversion", "C", "--seed", "7", "--ag3s-version", SF])
    assert lp.oracle_client_argv("vla_v11ab", a) == [] and "oracle_grasp" not in lp.oracle_settings("vla_v11ab", a)
    wrapped = lsp.make_server_argv("pi05_libero_safety")("vla_v11ab", "/ck", 9, SF)
    assert "--self-filter-inflation-link" in wrapped and wrapped[wrapped.index("--robot") + 1] == "panda"
    from AG3S.adapters.panda import profile

    profile.version_server_flags(SF, **lp.version_options(a))
    jobs = em.build_jobs(suites=em.parse_suites("libero_goal,libero_spatial"), version=SF, conversion="C", split="dev",
                         run_prefix="t", step="S")
    assert jobs and all(j["version"] == SF and SF in j["argv"] for j in jobs)
    with pytest.raises(ValueError):
        em.build_jobs(suites=[("libero_goal", None)], version=f"{SF}2", conversion="C", split="dev", run_prefix="t", step="S")


# ---------------------------------------------------------------------------------------------------- (d) the models
def test_the_self_filter_model_has_the_hand_link_and_no_palm_shell(gpu_ok, monkeypatch):
    """The palm shell (``--palm-shell on``, 111 spheres under ``-pc``) is appended to the **constraint** model only; the self-filter
    model is ``build_panda_self_filter_model()`` and carries franka.yml + the covering capsules (roles collision / self_filter_cover)."""
    from AG3S.adapters.panda import model as pm, profile as pf

    kw = pf.v11ab_build_kwargs(_args(SF, monkeypatch))
    assert kw["palm_shell"]
    models = pm.build_panda_models(kw["links"], palm_shell=dict(kw["palm_shell"]))
    sf, tc = models.self_filter, models.constraint
    assert "palm_shell" not in set(sf.sphere_roles) and set(sf.sphere_roles) == {"collision", "self_filter_cover"}
    assert "palm_shell" in set(tc.sphere_roles) and sum(r == "palm_shell" for r in tc.sphere_roles) == 111
    names = set(sf.sphere_link_names)
    assert {"panda_hand", "panda_leftfinger", "panda_rightfinger"} <= names                  # the link the flag names exists
    assert sf.n_spheres == 4271 and sum(n == "panda_hand" for n in sf.sphere_link_names) == 911


def test_an_unknown_link_name_is_refused_not_silently_ignored():
    from AG3S.adapters.panda import model as pm
    from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
    from AG3S.ag3s.stages.robot_filter import resolve_self_filter_inflation

    model = pm.build_panda_self_filter_model()
    cfg = AG3SConfig.from_dict({"pointcloud": {"self_filter_inflation_by_link": {"panda_hands": 0.009}}})
    with pytest.raises(AG3SConfigError):
        resolve_self_filter_inflation(model.sphere_link_names, cfg.pointcloud)


# ---------------------------------------------------------------------------------------------------- (e) the measurement
def test_rule_helpers():
    from AG3S.trajopt.experiments import panda_sfhand_leak as L

    assert L.ceil_mm(0.00894, 15.0) == 9.0 and L.ceil_mm(0.009, 15.0) == 9.0 and L.ceil_mm(0.0090001, 15.0) == 10.0
    assert L.ceil_mm(0.0295, 15.0) == 15.0                                            # the cap
    s = L.stats_mm(np.arange(1, 101) / 1000.0)
    assert s["n"] == 100 and s["median"] == pytest.approx(50.5) and s["p95"] == pytest.approx(95.05) and s["max"] == pytest.approx(100.0)
    assert L.stats_mm([]) == {"n": 0}
    # sd = |p - c| - r, the min over spheres
    c, r = np.asarray([[0.0, 0, 0], [1.0, 0, 0]]), np.asarray([0.1, 0.2])
    assert L.sphere_signed(np.asarray([[0.5, 0, 0], [0.15, 0, 0], [0.05, 0, 0]]), c, r) == pytest.approx([0.3, 0.05, -0.05])


def test_decide_follows_the_fixed_rule():
    from AG3S.trajopt.experiments import panda_sfhand_leak as L

    def blk(hand_p95, fing_p95, tgt_median):
        return {"hand_points_to_panda_hand_spheres_mm": {"p95": hand_p95}, "finger_points_to_finger_spheres_mm": {"p95": fing_p95, "n": 5},
                "target_nearest_to_finger_per_call_mm": {"median": tgt_median, "n": 9}}

    d = L.decide(blk(8.94, 2.1, 0.67))
    assert d["panda_hand_mm"] == 9.0 and d["fingers_included"] is False and d["finger_mm"] is None    # finger p95 above the target distance
    d = L.decide(blk(8.94, 2.1, 5.0))
    assert d["fingers_included"] is True and d["finger_mm"] == 3.0
    assert L.decide(blk(31.0, 2.1, 0.67))["panda_hand_mm"] == 15.0                                     # cap


def test_measure_points_classifies_by_nearest_geom_and_measures_to_the_filter_spheres():
    """A synthetic world: a hand box at the origin, a bowl box 0.3 m away, a hand sphere and a finger sphere. No MuJoCo needed."""
    from AG3S.trajopt.experiments import panda_sfhand_leak as L

    eye = np.eye(3).reshape(1, 9)
    snap = {"geom_contype": np.asarray([1, 1]), "geom_conaffinity": np.asarray([1, 1]), "body_names": ["gripper0_right_gripper", "bowl_main"],
            "geom_bodyid": np.asarray([0, 1]), "geom_type": np.asarray([6, 6]), "geom_size": np.asarray([[0.05] * 3, [0.05] * 3]),
            "geom_xpos": np.asarray([[0.0, 0, 0], [0.3, 0, 0]]), "geom_xmat": np.repeat(eye, 2, axis=0), "target_bodies": ["bowl_main"]}
    centres = np.asarray([[0.0, 0, 0], [0.0, 0.0, -0.4]])        # a hand sphere r 0.055 (covers the hand box) and a finger sphere
    radii = np.asarray([0.055, 0.02])
    links = np.asarray(["panda_hand", "panda_leftfinger"])
    pts = np.asarray([[0.06, 0, 0],       # 10 mm from the hand box, 5 mm outside the hand sphere -> a leak of 5 mm
                      [0.03, 0, 0],       # inside the hand sphere -> deleted by the filter at inflation 0
                      [0.26, 0, 0.0],     # on the bowl box face (inside the box): target class
                      [0.0, 0.0, -0.43]])  # 10 mm outside the finger sphere, no geom within ghost -> nothing
    out = L.measure_points(pts, centres, radii, links, np.eye(4), snap, ghost_m=0.03, near_m=0.5, survivors_only=True, reach_m=1.0,
                           extra={"palm_shell": (centres[:1], radii[:1] * 0.9)})
    assert out["n_masked_by_filter"] == 1 and out["n_points"] == 3
    assert out["hand"]["sd_hand"] == pytest.approx([0.005]) and out["hand"]["geom"] == pytest.approx([0.01])
    assert out["hand"]["extra"]["palm_shell"] == pytest.approx([0.06 - 0.055 * 0.9])
    assert len(out["finger"]["sd_finger"]) == 0
    assert len(out["target"]["sd_hand"]) == 1 and out["target"]["sd_hand"][0] == pytest.approx(0.26 - 0.055)
    # without the survivors-only cut the inside point stays a hand point, with a negative distance
    out2 = L.measure_points(pts, centres, radii, links, np.eye(4), snap, ghost_m=0.03, near_m=0.5, survivors_only=False, reach_m=1.0)
    assert sorted(out2["hand"]["sd_hand"]) == pytest.approx([-0.025, 0.005])
    # block(): the inflation grid counts the hand point at 5 mm and not at 4 mm
    blk = L.block([{**out, "phase": "closing"}])
    assert blk["inflation_grid_mm"]["4"]["hand_points_caught_by_panda_hand"] == 0.0
    assert blk["inflation_grid_mm"]["5"]["hand_points_caught_by_panda_hand"] == 1.0


def test_committed_data_reproduces_the_registered_value():
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt.experiments import panda_sfhand_leak as L

    d = json.loads(DATA.read_text())
    assert d["self_filter_model"]["palm_shell_in_model"] is False
    primary = d["sources"][d["primary_source"]]["closing_and_held"]
    dec = L.decide(primary)
    assert dec == d["decision"]
    assert dec["panda_hand_mm"] / 1000.0 == pf.SL15_SFHAND_PANDA_HAND_M
    assert dec["fingers_included"] is False and f"panda_hand={pf.SL15_SFHAND_PANDA_HAND_M:g}" in pf.AG3S_VERSIONS[SF]
    assert not any("panda_leftfinger" in t or "panda_rightfinger" in t for t in pf.AG3S_VERSIONS[SF])
