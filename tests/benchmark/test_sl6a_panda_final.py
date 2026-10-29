"""SL6-A — the AG3S final configuration on the LIBERO Panda server (task `AG3S/ag3s/docs/handoff/SL6.task.md` §2,
VA1 / VA3 / VA4), CPU only.

1. A5 palm shell: the committed file (count, digest, frame, keep-out), the shell-less model is the exact prefix of the
   model with it, a refit from the LIBERO export (when present) reproduces the file; the self-filter cover model
   builder is unchanged by default.
2. A8 gripper spec = the export measured in the gripper frame (when present); the deferred-close gripper override.
3. A1 / A10 versions: the final flags = the RB-Y1 final argv (A6 impl §3) minus the V11AB flags (+ scope nbc), G1 / A4
   only in `-g1a4`, no V11AB_PRE_MERGE_DEFAULTS, the Panda probe required; launcher argv; read-back.
4. VA4: both final versions parse, pass the combination checks and build the server (stub policy, no checkpoint, GPU
   probe and warm-up skipped); the config dump names the shell, the A6 site, the GPU seed robot, the ④ scope and the
   inert feature.
5. VA3 (in-process part): V11AB / pb7 build the same SafePolicy arguments and `info` as the HEAD 553ad13e server.
"""
from __future__ import annotations

import json
import pathlib
import shlex
import subprocess
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPORT = ROOT / "outputs/verify/SL2/raw/geo_libero.npz"
PRE_SL6 = "553ad13e2ac6a82bc010001b95d2939c3cf23a35"
#: the Panda probe (SL6-B B4, storage); the RB-Y1 6-class probe stands in (format only) where storage is absent
PANDA_PROBE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark/20261029T192223_5ea56af44722_sl6b-hsr6-panda-labels/"
                           "probe/hsr6_panda_kv_L4.npz")
PROBE = PANDA_PROBE if PANDA_PROBE.exists() else ROOT / "AG3S/ag3s/asset/subtask_probe/hsr6_t64_kv_L4.npz"

#: AG3S-CORE-EVAL-A6.impl.md §3 (RB-Y1 final server argv), flags after the required ones.
RBY1_FINAL = (
    "--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy "
    "exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 "
    "--max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry "
    "--latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 "
    "--finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 "
    "--w-deflection-rate 20 --to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 "
    "--attention-split-radius-mm 12 --attention-split-ratio-source raw --target-permission-strict --attention-source "
    "noun_target:L10H7 --attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1 "
    "--neighbour-contact-mm 20 --nontarget-evidence surface --placed-no-seed-certify --kept-geometry-follow --roi-carve "
    "--roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda "
    "--blas-threads 1 --to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass --palm-shell on "
    "--palm-shell-margin 5 --grasp-anchor 2000 --grasp-anchor-path-weight 0 --geo-grasp --geo-goal-mode correct "
    "--geo-track-scale 0.25 --geo-continuity-off --hsr6-attach-assist --hsr6-held-gate --hsr6-target-exempt "
    "--hsr6-objective --neighbour-palm-contact-mm 10")


def pf_sha():
    from AG3S.adapters.panda import profile as pf

    return pf.PANDA_HSR6_PROBE_SHA256


@pytest.fixture
def gpu_ok(monkeypatch):
    """CPU host: `--to-gpu-seeds curobo` is parsed as on a GPU host (the availability probe passes)."""
    from AG3S.trajopt import gpu_seeds

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, ""))


def _final_args(version, probe=PROBE, extra=()):
    from AG3S.adapters.panda import profile as pf

    argv = pf.version_server_argv(version)
    argv[argv.index("--config") + 1] = "pi05_libero"
    argv += ["--robot", "panda"] + pf.version_server_flags(version, hsr6_probe=str(probe)) + list(extra)
    return pf.parse_v11ab_args(argv=argv)


# ------------------------------------------------------------------------------------------------ 1. A5 palm shell
def test_committed_palm_shell():
    from AG3S.adapters.panda import hand_geometry as hg
    from AG3S.ag3s.experiments.sources import gripper_cover as gc
    from AG3S.ag3s.robot_models.urdf_sphere_chain import PALM_SHELL_ROLE

    shell, doc = hg.load_palm_shell()
    assert len(shell) == 111 and {c.link for c in shell} == {"panda_hand"}
    assert {c.role for c in shell} == {PALM_SHELL_ROLE} and all(c.length == 0.0 for c in shell)
    assert doc["provenance"]["constants"] == {"overflow": gc.PALM_SHELL_OVERFLOW, "spacing": gc.PALM_COVER_SPACING,
                                              "grid": gc.PALM_COVER_GRID, "union_eps": gc.PALM_COVER_UNION_EPS,
                                              "face_cos": gc.PALM_SHELL_FACE_COS}
    rep = doc["report"]["panda_hand"]
    assert rep["keep_out_max_entry_mm"] <= 0.0 and rep["n_uncovered"] <= 1 and rep["overflow_mm"] == 5.0
    assert rep["channel_half_width_mm"] == pytest.approx(39.0) and rep["bottom_mm"] == pytest.approx(65.96, abs=0.01)
    table = hg.palm_shell_table(shell)
    assert table["radius_mm"]["min"] > 1.0 and table["radius_mm"]["max"] < 35.0
    # nothing below the hand mesh bottom inside the open-finger channel (keep-out), 5 mm overflow elsewhere
    C = np.asarray([c.origin[:3, 3] for c in shell])
    R = np.asarray([c.radius for c in shell])
    in_channel = np.abs(C[:, 1]) + R <= 0.039
    assert np.all(C[in_channel, 2] + R[in_channel] <= rep["bottom_mm"] / 1e3 + 1e-9)


def test_shell_is_appended_after_the_v11ab_model():
    from AG3S.adapters.panda import model as pm

    for links in (pm.PANDA_GRIPPER_LINKS, pm.PANDA_ARM_LINKS):
        bare = pm.build_panda_constraint_model(links)
        shelled = pm.build_panda_constraint_model(links, palm_shell={"hands": "both"})
        n = bare.n_spheres
        assert shelled.n_spheres == n + 111 and not hasattr(bare, "palm_shell_report")
        assert list(shelled.sphere_link_names[:n]) == list(bare.sphere_link_names)
        np.testing.assert_array_equal(np.asarray(shelled.radii[:n]), np.asarray(bare.radii))
        assert shelled.palm_shell_report["n_spheres_before"] == n
    with pytest.raises(ValueError, match="one hand"):
        pm.build_panda_constraint_model(pm.PANDA_GRIPPER_LINKS, palm_shell={"hands": "right"})
    with pytest.raises(ValueError, match="refit"):
        pm.build_panda_constraint_model(pm.PANDA_GRIPPER_LINKS, palm_shell={"hands": "both", "overflow": 0.0025})


@pytest.mark.skipif(not EXPORT.exists(), reason="LIBERO geometry export (outputs/, not in git) absent")
def test_palm_shell_refit_reproduces_the_committed_file():
    pytest.importorskip("mujoco")
    from AG3S.adapters.panda import cover as pc
    from AG3S.adapters.panda import hand_geometry as hg

    tables = pc.load_geometry(EXPORT)
    shell, _ = hg.fit_palm_shell(tables)
    committed, doc = hg.load_palm_shell()
    assert hg.shell_digest(shell) == doc["provenance"]["shell_digest"] == hg.shell_digest(committed)
    assert doc["provenance"]["geometry_digest"] == str(tables["geometry_digest"])
    # default builder = the self-filter cover model (no joints, contype 0), unchanged
    m, _ = pc.build_mujoco_model(tables)
    assert m.njnt == 0 and int(np.max(m.geom_contype)) == 0 and int(np.max(m.geom_conaffinity)) == 0
    c = hg.collision_model(tables)
    assert c.njnt == 9 and int(np.min(c.geom_contype)) == 1


# ------------------------------------------------------------------------------------------------ 2. A8
@pytest.mark.skipif(not EXPORT.exists(), reason="LIBERO geometry export (outputs/, not in git) absent")
def test_panda_gripper_spec_matches_export():
    pytest.importorskip("mujoco")
    from AG3S.adapters.panda import cover as pc
    from AG3S.adapters.panda import hand_geometry as hg
    from AG3S.ag3s.grasp.gripper import PANDA_LEFT

    m = hg.measure_gripper_spec(pc.load_geometry(EXPORT))
    tol = 1e-4
    assert m["max_opening"] == pytest.approx(PANDA_LEFT.max_opening, abs=tol)
    assert m["pad_half_y"] == pytest.approx(PANDA_LEFT.pad_half_y, abs=tol)
    assert m["pad_z"] == pytest.approx(PANDA_LEFT.pad_z, abs=tol)
    assert m["finger_thickness"] == pytest.approx(PANDA_LEFT.finger_thickness, abs=tol)
    for got, want in zip(m["palm"], (PANDA_LEFT.palm.lo, PANDA_LEFT.palm.hi)):
        assert got == pytest.approx(want, abs=tol)


def test_deferred_close_override():
    from AG3S.adapters.panda import action as pa

    actions = np.zeros((10, 7), np.float32)
    actions[:, 6] = [-1, -1, -1, 1, 1, 1, 1, 1, 1, 1]                               # the policy closes from row 3
    ref = np.zeros((10, 8))
    ref[:, 7] = (1.0 - actions[:, 6]) / 2.0
    refined = ref.copy()
    refined[3:, 7] = 1.0                                                             # G1 kept it open
    out, rec = pa.apply_deferred_close(actions, refined, ref)
    assert out.dtype == actions.dtype and rec["rows"] == list(range(3, 10))
    assert np.all(out[:, 6] == -1.0) and np.array_equal(out[:, :6], actions[:, :6])
    same, rec2 = pa.apply_deferred_close(actions, ref, ref)
    assert np.array_equal(same, actions) and rec2["rows"] == []


def test_g1_deferred_reads_the_close_gate_record():
    from AG3S.adapters.panda import server as ps

    srv = object.__new__(ps.PandaSafeServer)
    srv._policy = types.SimpleNamespace(_geo=None)
    assert srv._g1_deferred() is False                                               # G1 off
    srv._policy._geo = types.SimpleNamespace(last={"close_gate": {"applied": True, "decision": "defer"}})
    assert srv._g1_deferred() is True
    srv._policy._geo.last = {"close_gate": {"applied": False}}
    assert srv._g1_deferred() is False


# ------------------------------------------------------------------------------------------------ 3. versions
def test_final_flags_are_the_rby1_final_argv(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    base = pf.v11ab_server_flags()
    rby1 = shlex.split(RBY1_FINAL)
    assert rby1[:len(base)] == base                                                  # the RB-Y1 final starts as V11AB
    added = rby1[len(base):]
    g1a4 = list(pf.G1A4_FLAGS)
    i = added.index(g1a4[0])
    assert added[i:i + len(g1a4)] == g1a4
    expect = added[:i] + added[i + len(g1a4):] + ["--hsr6-objective-scope", "nbc",                    # A9
                                                  "--continuity-after-grasp", "--deflection-rate-after-grasp"]  # A12
    expect[expect.index("--attention-source") + 1] = "noun_target:L8H6"                    # SL6-B head re-selection
    assert list(pf.FINAL_CONFIG_FLAGS) == expect
    assert pf.AG3S_VERSIONS["AG3S-final-panda-g1a4"] == pf.FINAL_CONFIG_FLAGS + pf.G1A4_FLAGS
    for flag in ("--reach-at-capture", "--latch-phase", "--closing-hand-exempt", "--held-hand-exempt"):
        assert flag not in pf.AG3S_VERSIONS["AG3S-final-panda-g1a4"]                # no pb flag
    for v in pf.FINAL_CONFIG_VERSIONS:
        argv = pf.version_server_argv(v)
        assert "--no-roi-carve" not in argv and argv[-len(base):] == base
        assert pf.version_server_flags(v)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]       # default: Panda probe
        with pytest.raises(ValueError, match="Panda HSR6 probe"):
            pf.version_server_flags(v, hsr6_probe="")
        flags = pf.version_server_flags(v, hsr6_probe="/p.npz", attention_source="noun_target:L8H2")
        assert flags[-2:] == ["--hsr6-probe", "/p.npz"] and flags[flags.index("--attention-source") + 1] == \
            "noun_target:L8H2"
    assert pf.version_server_argv("V11AB-pb7")[-3:] == ["--no-roi-carve", "--blas-threads", "0"]
    assert pf.v11ab_server_argv() == pf.version_server_argv("V11AB")
    with pytest.raises(ValueError):
        pf.version_server_flags("V11AB", hsr6_probe="/p.npz")


def test_launcher_argv(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from benchmark.runners import launch_libero_paired as lp

    old = lp.server_argv("vla_v11ab", "/ck", 9, "V11AB-pb7")
    assert old[-len(pf.AG3S_VERSIONS["V11AB-pb7"]) - 2:-len(pf.AG3S_VERSIONS["V11AB-pb7"])] == ["--robot", "panda"]
    assert "--no-roi-carve" in old
    args = types.SimpleNamespace(hsr6_probe=None, attention_source=None)
    assert lp.version_options(args) == {}
    args.hsr6_probe = "/p.npz"
    new = lp.server_argv("vla_v11ab", "/ck", 9, "AG3S-final-panda-g1a4", **lp.version_options(args))
    assert "--no-roi-carve" not in new and new[new.index("--config") + 1] == "pi05_libero"
    assert new[-2:] == ["--hsr6-probe", "/p.npz"] and "--geo-grasp" in new and "--roi-carve" in new
    parser = lp.build_parser()
    a = parser.parse_args(["--task-ids", "1", "--episode-ids", "1", "--run-id", "x", "--conversion", "C",
                           "--ag3s-version", "AG3S-final-panda", "--hsr6-probe", "/p.npz"])
    assert lp.version_options(a) == {"hsr6_probe": "/p.npz"}


def test_final_versions_read_back(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    for v in pf.FINAL_CONFIG_VERSIONS:
        assert pf.match_ag3s_version(_final_args(v)) == v
    # the per-run attention cell does not change the version name
    from AG3S.adapters.panda import profile as pf2

    argv = pf2.version_server_argv("AG3S-final-panda")
    argv += ["--robot", "panda"] + pf2.version_server_flags("AG3S-final-panda", hsr6_probe=str(PROBE),
                                                            attention_source="noun_target:L8H2")
    assert pf2.match_ag3s_version(pf2.parse_v11ab_args(argv=argv)) == "AG3S-final-panda"
    # V11AB family values keep their keys (no final-only key appears)
    fam = pf.version_flag_values(pf.parse_v11ab_args())
    assert "palm_shell" not in fam and "hsr6_objective" not in fam and "blas_threads" not in fam
    assert "palm_shell" in pf.version_flag_values(_final_args("AG3S-final-panda"))


# ------------------------------------------------------------------------------------------------ 4. VA4
class _Stub:
    def infer(self, obs, **kw):  # pragma: no cover
        raise AssertionError


@pytest.mark.parametrize("version", ["AG3S-final-panda", "AG3S-final-panda-g1a4"])
def test_final_config_builds_on_cpu(version, gpu_ok, monkeypatch, tmp_path):
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    args = _final_args(version)
    ps.check_unported_flags(args, None)                                              # every final dest is ported
    served, info = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
    safe = served._policy
    fc = info["final_config"]
    assert fc["version"] == version
    assert fc["palm_shell"]["n_spheres"] == 111 and fc["palm_shell"]["margin_m"] == pytest.approx(0.005)
    assert fc["neighbour_palm_contact"]["site"] == "panda_grip_site"
    assert fc["neighbour_palm_contact"]["site_offset_m"] == [0.0, 0.0, 0.097]
    assert fc["neighbour_palm_contact"]["palm_rows"] == 129
    assert fc["gpu_seeds"]["robot"] == {"name": "panda", "urdf": str(ps.pm.PANDA_URDF), "base_link": "panda_link0",
                                        "tool_frames": ["panda_hand"], "fk_root_link": "panda_link0"}
    assert fc["hsr6"] == {"flags": ["attach_assist", "held_gate", "target_exempt", "objective"],
                          "objective_scope": "nbc", "probe": str(PROBE)}
    # A12: both before-grasp switches in the config (and the TrajOpt config validated with them)
    assert fc["before_grasp_terms_off"] == {"continuity_after_grasp": True, "deflection_rate_after_grasp": True,
                                            "w_continuity": safe.to_config.cost.w_continuity,
                                            "w_deflection_rate": 20.0,
                                            "geo_continuity_off": version.endswith("g1a4")}
    safe.to_config.cost.validate()                                                       # CostConfig.validate
    assert safe.to_config.cost.w_continuity > 0.0
    # the Panda probe through hsr6.load_hsr6_probe, the attention cell L8H6
    from AG3S.trajopt.attention_source import parse_attention_source

    src = parse_attention_source(args.attention_source)
    assert (src.label, src.layer, src.head) == ("noun_target:L8H6", 8, 6)
    hp = fc["attention"]["hsr6_probe"]
    if PROBE == PANDA_PROBE:
        assert hp["sha256"] == pf_sha() and hp["classes"] == ["approach", "grasp", "lift", "transport", "placement"]
        assert hp["type"] != "SubtaskProbe"                                                     # B's subset wrapper
    assert [r["flag"] for r in fc["inert_on_panda"]] == ["--placed-no-seed-certify"]
    assert fc["wrist_track"]["joints"] == ["panda_joint5", "panda_joint6", "panda_joint7"]
    assert fc["multi_seed"]["palm_links"] == ["panda_hand"]
    assert fc["attention"]["attention_source"] == "noun_target:L8H6"
    assert info["ag3s"]["target_exclusion"]["target_exclusion"] == "ball_owned"
    assert info["ag3s"]["target_exclusion"]["roi_carve"] is True
    assert safe.hsr6 is not None and safe.hsr6_objective_scope == "nbc"
    assert safe.hold_site_offset_m == (0.0, 0.0, 0.097)
    from AG3S.trajopt import gpu_seeds as gs

    assert gs.robot_of(safe.refiner.optimizer).name == "panda"
    win = getattr(safe.refiner.optimizer, "window_optimizer", None)
    assert win is None or gs.robot_of(win).name == "panda"
    if version.endswith("g1a4"):
        assert safe._geo is not None and fc["geo_grasp"]["spec"] == "panda_hand"
        assert fc["grasp_anchor"]["gripper_column"] == 7
        assert safe._geo._ik("left")[1] == list(range(7)) and safe._geo._ik("left")[2] == "urdf"
    else:
        assert safe._geo is None and fc["geo_grasp"] is None and fc["grasp_anchor"] is None
    assert safe.to_config.horizon.planned == 10 and safe.to_config.horizon.execution_length == 5
    json.dumps(info, default=str)


def test_final_config_refusals(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps

    argv = pf.version_server_argv("AG3S-final-panda") + ["--robot", "panda"] + list(pf.AG3S_VERSIONS["AG3S-final-panda"])
    with pytest.raises(ValueError, match="needs --hsr6-probe"):
        ps.check_unported_flags(pf.parse_v11ab_args(argv=argv), None)
    with pytest.raises(ValueError, match="single left hand"):
        ps.check_unported_flags(_final_args("AG3S-final-panda", extra=["--palm-shell", "right"]), None)
    with pytest.raises(ValueError, match="not ported"):
        ps.check_unported_flags(_final_args("AG3S-final-panda", extra=["--geo-wrist-check"]), None)
    # the pre-merge pin cannot be combined with the explicit ROI carve (serve_safe refuses it)
    with pytest.raises(SystemExit):
        pf.parse_v11ab_args(argv=pf.v11ab_server_argv() + list(pf.FINAL_CONFIG_FLAGS) + ["--hsr6-probe", "/p"])


def test_recorder_meta_final_keys(gpu_ok, tmp_path, monkeypatch):
    from AG3S.adapters.panda import server as ps

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    args = _final_args("AG3S-final-panda", extra=["--record-constraints", str(tmp_path / "rc"),
                                                  "--record-constraints-esdf", "occupancy"])
    served, info = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
    meta = served.recorder.meta
    assert meta["palm_shell"] == {"hands": "both"} and meta["attention_source"]["spec"] == "noun_target:L8H6"
    assert meta["hsr6"]["objective_scope"] == "nbc" and meta["hsr6"]["probe"] == str(PROBE)
    assert meta["subtask"]["placed_no_seed_certify_inert_on_panda"] is True
    assert meta["final_config"]["palm_shell"]["n_spheres"] == 111
    assert meta["ag3s_version"] == "AG3S-final-panda"
    ps.close_recorder(served)


# ------------------------------------------------------------------------------------------------ 5. VA3 (in process)
def _git_module(path: str, name: str):
    try:
        src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PRE_SL6}:{path}"], text=True,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git object {PRE_SL6} not available")
    mod = types.ModuleType(name)
    mod.__file__ = str(ROOT / path)
    exec(compile(src, f"{PRE_SL6}:{path}", "exec"), mod.__dict__)  # noqa: S102
    return mod


@pytest.mark.parametrize("version", ["V11AB", "V11AB-pb7"])
def test_v11ab_family_build_is_the_pre_sl6_build(version, monkeypatch):
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import safe_policy
    from tests.benchmark.test_panda_sl5j_recorder import _args, _describe

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    seen = []

    class Capturing(safe_policy.SafePolicy):
        def __init__(self, *a, **kw):
            seen.append((a, kw))
            super().__init__(*a, **kw)

    monkeypatch.setattr(safe_policy, "SafePolicy", Capturing)
    old = _git_module("AG3S/adapters/panda/server.py", "_panda_server_pre_sl6")
    args = _args(version)

    def build(module):
        served, info = module.build_safe_server(_Stub(), args)
        a, kw = seen[-1]
        return info, {"args": [_describe(x) for x in a], "kwargs": {k: _describe(v) for k, v in sorted(kw.items())}}

    info_old, call_old = build(old)
    info_new, call_new = build(ps)
    norm = lambda d: json.loads(json.dumps(d, default=str).replace("_panda_server_pre_sl6", "panda.server"))  # noqa: E731
    assert json.dumps(norm(call_new), sort_keys=True) == json.dumps(norm(call_old), sort_keys=True)
    assert json.dumps(info_new, sort_keys=True, default=str) == json.dumps(info_old, sort_keys=True, default=str)
    assert "final_config" not in info_new
    safe = seen[-1]
    assert "palm_links" not in safe[1] and "hsr6" not in safe[1]


# ------------------------------------------------------------------------------------------------ A12
@pytest.mark.parametrize("version", ["AG3S-final-panda", "AG3S-final-panda-g1a4"])
def test_before_grasp_terms_drop_only_while_not_held(version, gpu_ok, monkeypatch):
    """A12 — on the Panda server built from the final argv, `--continuity-after-grasp` / `--deflection-rate-after-grasp`
    drop the two terms on latch searching / latched / closing and keep them from held on (and after a place)."""
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt.grasp_latch import GraspPhase

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    served, _ = ps.build_safe_server(_Stub(), _final_args(version), gpu_warm_up=False)
    safe = served._policy
    assert safe.to_config.cost.continuity_after_grasp and safe.to_config.cost.deflection_rate_after_grasp
    for phase in GraspPhase:
        safe._latch = types.SimpleNamespace(phase=phase)
        cont, rate = safe._continuity_after_grasp_gate(), safe._deflection_rate_gate()
        before = phase in (GraspPhase.SEARCHING, GraspPhase.LATCHED, GraspPhase.CLOSING)
        assert (cont is not None) is before and (rate is not None) is before, phase
        if before:
            assert cont.startswith("before_grasp") and "deflection_rate_after_grasp" in rate


def test_v11ab_family_keeps_both_terms(monkeypatch):
    from AG3S.adapters.panda import server as ps
    from tests.benchmark.test_panda_sl5j_recorder import _args

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    for version in ("V11AB", "V11AB-pb7"):
        served, _ = ps.build_safe_server(_Stub(), _args(version))
        cost = served._policy.to_config.cost
        assert not cost.continuity_after_grasp and not cost.deflection_rate_after_grasp


# ------------------------------------------------------------------------------------------------ SL6-C 2판
def test_v2_flags_are_final_without_noun_cell_plus_pb167(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    v2 = list(pf.AG3S_VERSIONS["AG3S-final-panda-v2"])
    base = list(pf.FINAL_CONFIG_FLAGS)
    i = base.index("--attention-source")
    assert base[i:i + 2] == ["--attention-source", "noun_target:L8H6"]
    assert v2 == base[:i] + base[i + 2:] + ["--reach-at-capture", "--grasp-reach-sticky", "--grasp-drop-from-open",
                                            "--closing-hand-exempt"]
    names = lambda flags: {t for t in flags if t.startswith("--")}
    assert "--attention-source" not in v2 and not names(pf.G1A4_FLAGS) & names(v2)
    # the earlier versions are unchanged
    assert pf.AG3S_VERSIONS["AG3S-final-panda"] == pf.FINAL_CONFIG_FLAGS
    assert pf.AG3S_VERSIONS["AG3S-final-panda-g1a4"] == pf.FINAL_CONFIG_FLAGS + pf.G1A4_FLAGS
    assert pf.AG3S_VERSIONS["V11AB-pb7"][-1] == "--closing-hand-exempt" and "--hsr6-objective" not in \
        pf.AG3S_VERSIONS["V11AB-pb7"]
    # a run's attention cell is appended (v2 has none of its own)
    assert pf.version_server_flags("AG3S-final-panda-v2", hsr6_probe="/p.npz")[-2:] == ["--hsr6-probe", "/p.npz"]
    f = pf.version_server_flags("AG3S-final-panda-v2", hsr6_probe="/p.npz", attention_source="noun_target:L8H6")
    assert f[f.index("--attention-source") + 1] == "noun_target:L8H6"
    assert pf.match_ag3s_version(_final_args("AG3S-final-panda-v2")) == "AG3S-final-panda-v2"


def test_v2_builds_on_cpu(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    args = _final_args("AG3S-final-panda-v2")
    assert args.attention_source is None                                     # default cell = action:last:L8H2
    ps.check_unported_flags(args, None)
    served, info = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
    safe = served._policy
    fc = info["final_config"]
    assert fc["version"] == "AG3S-final-panda-v2"
    assert fc["hsr6"]["objective_scope"] == "nbc" and fc["palm_shell"]["n_spheres"] == 111
    assert fc["before_grasp_terms_off"]["continuity_after_grasp"] is True
    assert safe._geo is None and fc["grasp_anchor"] is None                  # G1 · A4 removed
    # pb1: AG3S reach gate at the capture state; pb6: latch options; pb7: closing exemption
    assert safe.ag3s.reach_at_capture is True
    assert safe._latch.config.reach_sticky is True and safe._latch.config.drop_from_open is True
    assert safe.closing_hand_exempt is True
    assert safe.latch_phase is False and safe.held_hand_exempt is False and safe.grasp_zone_allowance is None  # no pb4/5
    assert safe.to_config.horizon.planned == 10 and safe.to_config.horizon.execution_length == 5
    json.dumps(info, default=str)
