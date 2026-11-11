"""SL-GTTO — the common TO with GT perception on the Panda (task `AG3S/ag3s/docs/handoff/SL-GTTO.task.md` §3), CPU only.

1. Versions (`AG3S/adapters/panda/profile.py`): `AG3S-commonto-gt` / `-pass` = the classified final flags (impl.md table)
   + grasp handling + GT perception; no probe, no attention cell; read back by `match_ag3s_version`; every other version's
   flags, argv and recorded flag keys are the base commit's.
2. Panda server: the oracle extent / identity / max-dist and the grounding source are ported, the hybrid is still refused;
   the GT versions build on CPU (config, SafePolicy metadata `oracle_segmentation_required`, config dump, per-call record);
   no GT key on any other version.
3. Launchers: `--oracle-seg --capture-segmentation` reach the client of the GT server only (run_mode, dry run, the
   LIBERO-Safety wrapper); every other run's argv / settings / dry-run output is byte-identical to the base commit.
4. Client (`benchmark/adapters/libero_panda/oracle_codes.py`, `libero_safe_client.SafeClient`): the code table, the
   request keys through the AG3S wire, the BDDL target rule (+ the affordance fallback), the connect check; a request
   without `--oracle-seg` is unchanged.
5. Real scenes (subprocess in the benchmark's interpreter, `tests/benchmark/libero_gtto_scene_checks.py`): LIBERO,
   SafeLIBERO, LIBERO-Safety -- code 1 == the gt_oracle mask, the code image is on the depth grid (projection of the
   target lands on code 1, back-projection lies at the target, the row-flipped image does not).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.benchmark.test_launcher_record_constraints import BASE_ARGV, _module_at, _run_mode
from tests.benchmark.test_sl6a_panda_final import _Stub, gpu_ok  # noqa: F401 (fixture)

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_COMMIT = "89b42459"   # feature/panda-benchmark before SL-GTTO (task base)
GT, GT_PASS = "AG3S-commonto-gt", "AG3S-commonto-gt-pass"
#: the classified common-TO + GT argv (SL-GTTO.impl.md flag table), spelled out
EXPECTED_GT = [
    "--to-fast-eval", "--esdf-host-mirror", "--target-exclusion", "ball-owned", "--owned-max-dist-voxels", "3",
    "--target-permission-strict", "--wrist-track-scale", "0.1", "--placed-no-seed-certify", "--kept-geometry-follow",
    "--roi-carve", "--roi-carve-footprint", "--roi-carve-footprint-pad-px", "0", "--roi-carve-footprint-robot",
    "ignore", "--roi-carve-device", "cuda", "--blas-threads", "1", "--palm-shell", "on", "--palm-shell-margin", "5",
    "--continuity-after-grasp", "--deflection-rate-after-grasp",
    "--reach-at-capture", "--links", "arms", "--held-hand-exempt", "--latch-phase", "--grasp-zone-mm", "10",
    "--grasp-reach-sticky", "--grasp-drop-from-open", "--closing-hand-exempt", "--subtask-from-latch",
    "--target-extent", "oracle-visible", "--oracle-extent-identity", "gt", "--grounding-source", "oracle"]
FORBIDDEN = ("--attention-source", "--attention-split-radius-mm", "--attention-split-ratio-source",
             "--attention-split-partition", "--split-supersedes-anchor", "--nontarget-evidence", "--hsr6-attach-assist",
             "--hsr6-held-gate", "--hsr6-target-exempt", "--hsr6-objective", "--hsr6-objective-scope",
             "--hsr6-placement-contact", "--hsr6-probe", "--to-multi-seed", "--to-gpu-seeds", "--to-gpu-seeds-check",
             "--neighbour-contact-mm", "--neighbour-palm-contact-mm", "--minimal-intervention-mm",
             "--target-extent attention-surface", "--surface-no-ball")
GT_ONLY_DESTS = ("grounding_source", "oracle_extent_identity", "oracle_max_dist_voxels")


def _args(version, extra=()):
    from AG3S.adapters.panda import profile as pf

    argv = pf.version_server_argv(version)
    argv[argv.index("--config") + 1] = "pi05_libero"
    return pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(version) + list(extra))


def _head_profile():
    try:
        src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:AG3S/adapters/panda/profile.py"],
                                      text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git object {BASE_COMMIT} not available")
    import types

    mod = types.ModuleType("_panda_profile_base")
    mod.__file__ = str(ROOT / "AG3S/adapters/panda/profile.py")
    exec(compile(src, "base:profile.py", "exec"), mod.__dict__)  # noqa: S102
    return mod


# ================================================================================================ 1 versions
def test_versions_are_the_classified_final_flags():
    from AG3S.adapters.panda import profile as pf

    assert pf.GT_VERSIONS == (GT, GT_PASS)
    assert list(pf.AG3S_VERSIONS[GT]) == EXPECTED_GT
    assert list(pf.AG3S_VERSIONS[GT_PASS]) == EXPECTED_GT + ["--passthrough-when-clear"]
    text = " ".join(pf.AG3S_VERSIONS[GT_PASS])
    for flag in FORBIDDEN:
        assert flag not in pf.AG3S_VERSIONS[GT_PASS] and f" {flag} " not in f" {text} ", flag
    for flag in pf.P0M_PANDA_FIX_FLAGS + pf.P0M_SUBTASK_FLAGS + pf.GT_PERCEPTION_FLAGS:
        assert flag in pf.AG3S_VERSIONS[GT]
    # every final flag is either kept or removed by the table, nothing else (GT_REMOVED_FLAGS lists the removed ones)
    kept = [t for t in pf.FINAL_CONFIG_FLAGS if t.startswith("--") and t not in pf.GT_REMOVED_FLAGS]
    assert all(t in pf.AG3S_VERSIONS[GT] for t in kept)
    assert all(t in pf.FINAL_CONFIG_FLAGS for t in pf.GT_REMOVED_FLAGS)


def test_gt_versions_take_no_probe_and_no_attention_cell():
    from AG3S.adapters.panda import profile as pf

    for v in pf.GT_VERSIONS:
        assert pf.is_final_version(v)
        assert pf.version_server_flags(v) == list(pf.AG3S_VERSIONS[v])
        for kw in ({"hsr6_probe": "/p.npz"}, {"attention_source": "action:last:L8H2"}):
            with pytest.raises(ValueError, match="nothing to feed"):
                pf.version_server_flags(v, **kw)
        argv = pf.version_server_argv(v)
        assert "--no-roi-carve" not in argv and argv == pf.v11ab_server_argv(pre_merge=False)


def test_only_the_gt_versions_require_the_segmentation():
    from AG3S.adapters.panda import profile as pf

    assert [v for v in pf.AG3S_VERSIONS if pf.version_requires_oracle_segmentation(v)] == \
        list(pf.GT_VERSIONS) + list(getattr(pf, "GT2_VERSIONS", ()))                    # + SL-GTTO2
    assert not pf.version_requires_oracle_segmentation("no-such-version")


@pytest.mark.parametrize("version", [GT, GT_PASS])
def test_versions_parse_build_their_config_and_read_back(version, gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    args = _args(version)
    kw = pf.v11ab_build_kwargs(args)
    assert kw["identity"]["grounding_source"] == "oracle"
    te = kw["target_exclusion"]
    assert (te["target_extent"], te["oracle_extent_identity"]) == ("oracle_visible", "gt")
    cfg = pf.v11ab_ag3s_config(kw)
    assert cfg.clustering.grounding_source == "oracle" and cfg.esdf.target_extent == "oracle_visible"
    assert cfg.esdf.oracle_extent_identity == "gt" and cfg.esdf.oracle_max_dist_voxels == 6.0
    assert cfg.clustering.attention_split_partition is False and cfg.esdf.nontarget_evidence == "proximity"
    assert pf.match_ag3s_version(args) == version
    values = pf.version_flag_values(args)
    assert values["grounding_source"] == "oracle" and values["oracle_extent_identity"] == "gt"
    assert values["passthrough_when_clear"] is (version == GT_PASS)
    assert ss.hsr6_flags_from_args(args) is None and args.attention_source is None and args.links == "arms"
    to = ss.trajopt_config_from_args(args)
    assert int(to.sqp.multi_seed or 0) == 0 and to.cost.wrist_track_scale == 0.1
    assert to.cost.continuity_after_grasp and to.cost.deflection_rate_after_grasp
    assert bool(to.sqp.passthrough_when_clear) is (version == GT_PASS)


def test_every_other_version_is_the_base_commits(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    old = _head_profile()
    assert set(pf.AG3S_VERSIONS) - set(old.AG3S_VERSIONS) == set(pf.GT_VERSIONS) | set(getattr(pf, "GT2_VERSIONS", ()))
    for name, flags in old.AG3S_VERSIONS.items():
        assert pf.AG3S_VERSIONS[name] == flags, name
        assert pf.version_server_argv(name) == old.version_server_argv(name), name
        assert pf.is_final_version(name) == old.is_final_version(name), name
        assert not pf.version_requires_oracle_segmentation(name), name
    for name in ("V11AB", "V11AB-pb7", "AG3S-final-panda-v3-e0to-pc", "AG3S-final-panda-p0m-pandafix",
                 "AG3S-final-panda-p0m-pandafix-pass", "AG3S-final-panda-v3-e0to-pc-idboth"):
        argv = pf.version_server_argv(name) + list(pf.version_server_flags(name))
        args = pf.parse_v11ab_args(argv=argv)
        assert pf.version_flag_values(args) == old.version_flag_values(args), name
        assert not set(GT_ONLY_DESTS) & set(pf.version_flag_values(args)), name
        assert "grounding_source" not in (pf.v11ab_build_kwargs(args)["identity"] or {}), name


# ================================================================================================ 2 Panda server
def test_server_ports_the_gt_flags_and_still_refuses_the_hybrid(gpu_ok):
    from AG3S.adapters.panda import server as ps

    for d in GT_ONLY_DESTS:
        assert d in ps.PORTED_FINAL_CONFIG_DESTS and d not in ps.UNPORTED_FINAL_CONFIG_DESTS, d
    assert ps.PORTED_TARGET_EXTENTS == ("estimated", "attention-surface", "oracle-visible")
    for v in (GT, GT_PASS):
        ps.check_unported_flags(_args(v), None)
    args = _args(GT)
    args.target_extent = "attention-surface-hybrid"
    with pytest.raises(ValueError, match="not ported"):
        ps.check_unported_flags(args, None)


@pytest.mark.parametrize("version", [GT, GT_PASS])
def test_gt_versions_build_on_cpu(version, gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import wire

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    served, info = ps.build_safe_server(_Stub(), _args(version), gpu_warm_up=False)
    safe = served._policy
    c = safe.ag3s.config
    assert (c.clustering.grounding_source, c.esdf.target_extent, c.esdf.oracle_extent_identity) == \
        ("oracle", "oracle_visible", "gt")
    md = served.metadata
    assert md[wire.ORACLE_SEG_REQUIRED] is True and md["target_extent"] == "oracle_visible"
    assert md["oracle_extent_identity"] == "gt" and "hsr6" not in md
    fc = info["final_config"]
    assert fc["version"] == version and fc["hsr6"] is None and fc["multi_seed"] is None
    assert fc["gt_perception"]["grounding_source"] == "oracle" and fc["gt_perception"]["segmentation_required"]
    assert "neighbour_contact" in fc and fc["neighbour_contact"] is None and fc["neighbour_palm_contact"] is None
    assert "panda_link4" in set(safe.constraint_robot_model.sphere_link_names)          # --links arms
    rec = ps.gt_perception_call_record(safe)
    assert rec["grounding_source"] == "oracle" and rec["target_extent"] == "oracle_visible"
    assert rec["grounding"] is None and rec["oracle"] is None                            # before the first call
    json.dumps(rec)


def test_other_versions_carry_no_gt_key(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import wire

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    served, info = ps.build_safe_server(_Stub(), _args("AG3S-final-panda-p0m-pandafix-pass"), gpu_warm_up=False)
    assert wire.ORACLE_SEG_REQUIRED not in served.metadata
    assert "gt_perception" not in info["final_config"]
    assert ps.gt_perception_call_record(served._policy) is None
    assert served._policy.ag3s.config.clustering.grounding_source == "attention"


def test_serve_safe_refuses_the_gt_grounding_without_the_oracle_extent(gpu_ok):
    with pytest.raises(SystemExit):
        _args("AG3S-final-panda-p0m-pandafix", extra=["--grounding-source", "oracle"])


# ================================================================================================ 3 launchers
@pytest.fixture
def lp():
    from benchmark.runners import launch_libero_paired as lp

    return lp


def _largs(lp, *extra):
    a = lp.build_parser().parse_args(BASE_ARGV + list(extra))
    for name, value in (("resize_size", 224), ("replan_steps", 5), ("num_steps_wait", 10)):
        setattr(a, name, getattr(a, name, value))
    return a


def test_oracle_client_argv_only_for_the_gt_server(lp):
    gt, other = _largs(lp, "--ag3s-version", GT), _largs(lp, "--ag3s-version", "AG3S-final-panda-p0m-pandafix")
    base = lp.client_argv("/run", 7, gt)
    assert lp.oracle_client_argv("vla_v11ab", gt, base) == base + ["--capture-segmentation", "--oracle-seg"]
    assert lp.oracle_client_argv("vla_only", gt, base) == base
    assert lp.oracle_client_argv("vla_v11ab", other, base) == base
    seg = _largs(lp, "--ag3s-version", GT, "--capture-segmentation")
    assert lp.oracle_client_argv("vla_v11ab", seg, lp.client_argv("/r", 1, seg)).count("--capture-segmentation") == 1
    with pytest.raises(RuntimeError, match="needs the GT segmentation"):
        lp.check_oracle_client_argv("vla_v11ab", gt, base)
    lp.check_oracle_client_argv("vla_only", gt, base)


def test_run_mode_hands_the_gt_client_its_flags(monkeypatch, tmp_path, lp):
    args = _largs(lp, "--ag3s-version", GT, "--modes", "vla_v11ab")
    seen, _ = _run_mode(monkeypatch, lp, tmp_path, "vla_v11ab", args)
    client = [e for e in seen["executions"] if e.get("role") == "client"][0]["argv"]
    assert client[-2:] == ["--capture-segmentation", "--oracle-seg"]
    assert seen["create"]["settings"]["oracle_segmentation"]["client_flags"] == ["--capture-segmentation", "--oracle-seg"]
    assert seen["popen"][seen["popen"].index("--target-extent") + 1] == "oracle-visible"
    seen, _ = _run_mode(monkeypatch, lp, tmp_path / "o", "vla_only", _largs(lp, "--ag3s-version", GT))
    client = [e for e in seen["executions"] if e.get("role") == "client"][0]["argv"]
    assert "--oracle-seg" not in client and "oracle_segmentation" not in seen["create"]["settings"]


@pytest.mark.parametrize("mode", ["vla_only", "vla_v11ab"])
@pytest.mark.parametrize("version", ["V11AB-pb7", "AG3S-final-panda-p0m-pandafix", "AG3S-final-panda-v3-e0to-pc"])
def test_every_other_run_is_byte_identical_to_the_base_launcher(monkeypatch, tmp_path, lp, mode, version, gpu_ok):
    old = _module_at(BASE_COMMIT, "_launch_libero_paired_base")
    opts = ["--ag3s-version", version] + (["--hsr6-probe", "/p.npz"] if version.endswith("-pc") else [])
    new, nd = _run_mode(monkeypatch, lp, tmp_path / "n", mode, _largs(lp, *opts))
    ref, od = _run_mode(monkeypatch, old, tmp_path / "o", mode, _largs(old, *opts))

    def dump(x, run_dir):
        return json.dumps(x, sort_keys=False).replace(str(run_dir), "<R>").encode()

    for key in ("popen", "create", "executions"):
        assert dump(new[key], nd) == dump(ref[key], od), key


def _dry(monkeypatch, capsys, mod, argv):
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(ROOT))
    monkeypatch.setattr(mod.l0, "checkpoint_identity", lambda path, require=True: {"path": "/ckpt"})
    monkeypatch.setattr(mod.l0, "libero_python", lambda: "/fake/libero/python")
    assert mod.main(BASE_ARGV + argv + ["--dry-run"]) == 0
    return capsys.readouterr().out


def test_dry_run(monkeypatch, capsys, lp, gpu_ok):
    old = _module_at(BASE_COMMIT, "_launch_libero_paired_base")
    for version in ("V11AB-pb7", "AG3S-final-panda-p0m-pandafix"):
        argv = ["--ag3s-version", version]
        assert _dry(monkeypatch, capsys, lp, argv) == _dry(monkeypatch, capsys, old, argv), version
    out = _dry(monkeypatch, capsys, lp, ["--ag3s-version", GT])
    line = [json.loads(x) for x in out.splitlines() if x.startswith('{"client_argv_by_mode"')]
    assert len(line) == 1
    by_mode = line[0]["client_argv_by_mode"]
    assert by_mode["vla_v11ab"][-2:] == ["--capture-segmentation", "--oracle-seg"]
    assert "--oracle-seg" not in by_mode["vla_only"]
    assert '"--grounding-source",\n  "oracle"' in out


def test_libero_safety_launcher_dry_run_and_run_mode(monkeypatch, capsys, tmp_path, lp, gpu_ok):
    from benchmark.runners import launch_libero_safety_paired as lsp

    for name in ("CLIENT_SCRIPT", "client_argv", "protocol_settings", "runtime_schedule", "server_argv"):
        monkeypatch.setattr(lp, name, getattr(lp, name))     # install_hooks swaps them: restored after the test
    monkeypatch.setattr(lsp.l0, "libero_python", lsp.l0.libero_python)
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(ROOT))
    monkeypatch.setattr(lsp, "checkpoint_identity", lambda name, require=True: {"path": "/ckpt"})
    argv = ["--task-suite-name", "obstacle_avoidance", "--level", "0", "--task-ids", "0", "--episode-ids", "0",
            "--run-id", "x", "--policy-checkpoint", "pi05_libero", "--conversion", "C"]
    assert lsp.main(argv + ["--ag3s-version", GT, "--dry-run"]) == 0
    out = capsys.readouterr().out
    line = [json.loads(x) for x in out.splitlines() if x.startswith('{"client_argv_by_mode"')][0]["client_argv_by_mode"]
    assert line["vla_v11ab"][-2:] == ["--capture-segmentation", "--oracle-seg"] and "--oracle-seg" not in line["vla_only"]
    assert str(lsp.CLIENT_SCRIPT) in line["vla_v11ab"]
    assert lsp.main(argv + ["--ag3s-version", "V11AB-pb7", "--dry-run"]) == 0
    assert "client_argv_by_mode" not in capsys.readouterr().out
    # the wrapper's run goes through lp.run_mode: the LIBERO-Safety client gets the flags too
    args = lsp.build_parser().parse_args(argv + ["--ag3s-version", GT])
    for name, value in (("resize_size", 224), ("replan_steps", 5), ("num_steps_wait", 10)):
        setattr(args, name, value)
    args.safety_level, args.record_constraints, args.data_split = None, "none", "dev"
    lsp.install_hooks(args)
    monkeypatch.setattr(lp, "protocol_settings", lambda a: {"p": 1})
    seen, _ = _run_mode(monkeypatch, lp, tmp_path, "vla_v11ab", args)
    client = [e for e in seen["executions"] if e.get("role") == "client"][0]["argv"]
    assert client[1] == str(lsp.CLIENT_SCRIPT) and client[-2:] == ["--capture-segmentation", "--oracle-seg"]


# ================================================================================================ 4 client
def _oc():
    from benchmark.adapters.libero_panda import oracle_codes

    return oracle_codes


def test_code_constants_are_ag3s_and_wire():
    from AG3S.ag3s.runtime import oracle_segmentation as osg
    from AG3S.trajopt import wire

    oc = _oc()
    assert (oc.CODE_NONE, oc.CODE_TARGET, oc.CODE_STATIC, oc.CODE_OBJECT_BASE, oc.CODE_VERSION) == \
        (osg.CODE_NONE, osg.CODE_TARGET, osg.CODE_STATIC, osg.CODE_OBJECT_BASE, osg.CODE_VERSION)
    assert (oc.ORACLE_SEG, oc.ORACLE_SEG_META, oc.ORACLE_SEG_REQUIRED) == \
        (wire.ORACLE_SEG, wire.ORACLE_SEG_META, wire.ORACLE_SEG_REQUIRED)


#: a LIBERO-like body table: world, table, robot, target bowl, a second bowl, plate, a cabinet fixture with parts, a
#: SafeLIBERO obstacle not in the BDDL (free joint), the robot's eef marker
BODIES = ["world", "table", "eef_marker", "robot0_base", "robot0_link1", "gripper0_leftfinger", "mount0_base",
          "akita_black_bowl_1_main", "akita_black_bowl_2_main", "plate_1_main", "wooden_cabinet_1_main",
          "wooden_cabinet_1_base", "milk_obstacle_1_main"]
ROOTS = [0, 1, 2, 3, 3, 3, 3, 7, 8, 9, 10, 10, 12]
FREE = [7, 8, 9, 12]


def test_body_code_table():
    oc = _oc()
    codes, legend, info = oc.body_code_table(BODIES, ROOTS, FREE, "akita_black_bowl_1",
                                             ["akita_black_bowl_1", "akita_black_bowl_2", "plate_1"])
    want = [2, 2, 0, 0, 0, 0, 0, 1, 3, 4, 2, 2, 5]
    assert codes.tolist() == want
    assert legend == {"akita_black_bowl_1": 1, "akita_black_bowl_2": 3, "plate_1": 4, "milk_obstacle_1_main": 5}
    assert info["free_joint_objects_not_in_bddl"] == ["milk_obstacle_1_main"] and info["n_robot_bodies"] == 5
    assert oc.body_code_table(BODIES, ROOTS, FREE, "akita_black_bowl_1", ["ghost_1"])[2]["unresolved_objects"] == \
        ["ghost_1"]
    with pytest.raises(ValueError, match="resolves to no MuJoCo body"):
        oc.body_code_table(BODIES, ROOTS, FREE, "akita_black_bowl_10", [])
    lut = oc.geom_lut(codes, [1, 7, 8, 4, 12])            # geoms: table, target, bowl 2, robot, obstacle
    seg = np.array([[-1, 0, 1], [2, 3, 4], [5, 99, -7]], np.int16)
    assert oc.code_image(seg, lut).tolist() == [[0, 2, 1], [3, 0, 5], [0, 0, 0]]
    assert oc.code_image(seg, lut).dtype == np.uint8


def _frames(oc, shape=(6, 8)):
    seg = np.full(shape, -1, np.int16)
    seg[1:3, 2:5] = 1                                           # target geom
    seg[4, :] = 0                                               # table geom
    frame = lambda: {"depth_m": np.full(shape, 0.8, np.float32), "K": np.eye(3), "T_base_cam": np.eye(4),
                     "robot_state": np.zeros(9), "stamp": 1.0, "T_world_base": np.eye(4),
                     "segmentation_geom": seg.copy()}
    return {"head": frame(), "left_wrist": frame()}


def test_request_keys_go_through_the_ag3s_wire():
    from AG3S.trajopt import wire

    oc = _oc()
    coder = oc.OracleCoder("akita_black_bowl_1", ["akita_black_bowl_1"])
    with pytest.raises(RuntimeError):
        coder.request_keys(_frames(oc))
    codes, coder.legend, coder.info = oc.body_code_table(BODIES, ROOTS, FREE, "akita_black_bowl_1",
                                                         ["akita_black_bowl_1"])
    coder.lut = oc.geom_lut(codes, [1, 7])
    frames = _frames(oc)
    keys = coder.request_keys(frames)
    assert sorted(keys) == ["ag3s/oracle_seg/head", "ag3s/oracle_seg/left_wrist", "ag3s/oracle_seg_meta"]
    assert keys["ag3s/oracle_seg_meta"]["target"] == "akita_black_bowl_1" and keys["ag3s/oracle_seg_meta"]["version"] == 1
    # the same arrays through the AG3S request (wire.pack_request's checks) and the server's unpack
    scene = {k[len(wire.PREFIX):]: v for k, v in keys.items()}
    scene["cameras"] = ["head", "left_wrist"]
    seg = wire.unpack_oracle_segmentation(scene)
    assert sorted(seg["labels"]) == ["head", "left_wrist"]
    lab = seg["labels"]["head"]
    assert lab.shape == frames["head"]["depth_m"].shape and lab.dtype == np.uint8
    assert (lab == 1).sum() == 6 and (lab[1:3, 2:5] == 1).all() and (lab[4] == 2).all()
    bad = _frames(oc)
    bad["head"]["segmentation_geom"] = bad["head"]["segmentation_geom"][:-1]
    with pytest.raises(ValueError, match="!= depth"):
        coder.request_keys(bad)
    del bad["left_wrist"]["segmentation_geom"]
    bad["head"] = _frames(oc)["head"]
    with pytest.raises(KeyError, match="--capture-segmentation"):
        coder.request_keys(bad)


BDDL = """(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language Pick up the black bowl and place it on the plate)
  (:objects akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl plate_1 - plate)
  (:fixtures main_table - table)
  (:obj_of_interest akita_black_bowl_1 plate_1)
  (:goal (And (On akita_black_bowl_1 plate_1))))"""
AFFORDANCE = BDDL.replace("(And (On akita_black_bowl_1 plate_1))",
                          "(And (CheckGripperContactPart akita_black_bowl_1 (2 3 4)))")


def test_bddl_target_rule_and_the_affordance_fallback(tmp_path):
    from benchmark.adapters.libero_panda import task_target as tt

    oc = _oc()
    for text, rule in ((BDDL, "task_target.goal_roles"), (AFFORDANCE, "first argument")):
        path = tmp_path / "t.bddl"
        path.write_text(text)
        target, objects, rec = oc.bddl_oracle_target(path, tt)
        assert target == "akita_black_bowl_1" and objects == ["akita_black_bowl_1", "akita_black_bowl_2", "plate_1"]
        assert rec["rule"].startswith(rule)
    path.write_text(BDDL.replace("(:obj_of_interest akita_black_bowl_1 plate_1)", "(:obj_of_interest plate_9)"))
    with pytest.raises(ValueError):
        oc.bddl_oracle_target(path, tt)


def _clients():
    sys.path.insert(0, str(ROOT / "benchmark/runners"))
    import libero_safe_client as sc
    import libero_safety_client as lsc

    return sc, lsc


def test_client_flags_and_the_connect_check():
    sc, lsc = _clients()
    a = sc.build_parser().parse_args(["--run-dir", "r", "--port", "1"])
    assert a.oracle_seg is False
    sc.check_oracle_arguments(a)
    a = sc.build_parser().parse_args(["--run-dir", "r", "--port", "1", "--oracle-seg"])
    with pytest.raises(ValueError, match="--capture-segmentation"):
        sc.check_oracle_arguments(a)
    a = lsc.build_parser().parse_args(["--run-dir", "r", "--port", "1", "--task-suite-name", "affordance", "--level", "0",
                                       "--policy-checkpoint", "pi05_libero", "--oracle-seg", "--capture-segmentation"])
    sc.check_oracle_arguments(a)
    assert a.oracle_seg and a.capture_segmentation
    assert sc.check_server_oracle({}, False) == {"server_requires": False, "client_sends": False}
    assert sc.check_server_oracle({"oracle_segmentation_required": True}, True)["server_requires"]
    with pytest.raises(RuntimeError, match="--oracle-seg --capture-segmentation"):
        sc.check_server_oracle({"oracle_segmentation_required": True}, False)


def _client(sc, oracle=None):
    from benchmark.adapters.libero_panda import camera, wire_pack

    c = sc.SafeClient(SimpleNamespace(), 7, 0, 0, 5, oracle=oracle)
    c.geometry = SimpleNamespace(frames=[_frames(_oc())], cameras=["head", "left_wrist"])
    c.camera, c.wire_pack = camera, wire_pack
    c.eef = [(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))]
    return c


def test_safe_client_request_is_unchanged_without_the_flag():
    sc, _ = _clients()
    oc = _oc()
    off, _ = _client(sc)._request({"prompt": "x"})
    coder = oc.OracleCoder("akita_black_bowl_1", ["akita_black_bowl_1"])
    codes, coder.legend, coder.info = oc.body_code_table(BODIES, ROOTS, FREE, "akita_black_bowl_1",
                                                         ["akita_black_bowl_1"])
    coder.lut = oc.geom_lut(codes, [1, 7])
    on, _ = _client(sc, coder)._request({"prompt": "x"})
    assert set(on) - set(off) == {"ag3s/oracle_seg/head", "ag3s/oracle_seg/left_wrist", "ag3s/oracle_seg_meta"}
    assert set(off) <= set(on)
    for k in off:
        a, b = off[k], on[k]
        assert (np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b), k
    assert not any(k.startswith("ag3s/oracle") for k in off)


# ================================================================================================ 5 real scenes
def _venv_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("GTTO_VENV_ROOT", str(ROOT)))


SCENES = {
    "libero_spatial_t0": ("libero", [".venv-libero", "--bench", "libero", "--task", "0"], "LIBERO_CONFIG_PATH",
                          ".runtime/libero"),
    "safelibero_spatial_LI_t0": ("safelibero", [".venv-libero", "--bench", "safelibero", "--task", "0"],
                                 "SAFELIBERO_CONFIG_PATH", ".runtime/safelibero"),
    "libero_safety_affordance_L0_t0": ("libero_safety", [".venv-libero-safety", "--bench", "libero_safety", "--suite",
                                                         "affordance", "--level", "0", "--task", "0"],
                                       "LIBERO_SAFETY_CONFIG_PATH", ".runtime/libero-safety"),
    "libero_safety_human_safety_L0_t0": ("libero_safety", [".venv-libero-safety", "--bench", "libero_safety", "--suite",
                                                           "human_safety", "--level", "0", "--task", "0"],
                                         "LIBERO_SAFETY_CONFIG_PATH", ".runtime/libero-safety"),
    "libero_safety_obstacle_avoidance_L0_t0": ("libero_safety", [".venv-libero-safety", "--bench", "libero_safety",
                                                                 "--suite", "obstacle_avoidance", "--level", "0",
                                                                 "--task", "0"],
                                               "LIBERO_SAFETY_CONFIG_PATH", ".runtime/libero-safety"),
}


@pytest.mark.parametrize("case", sorted(SCENES))
def test_code_image_on_a_real_scene_is_the_depth_grid(case):
    bench, (venv, *argv), cfg_env, cfg_rel = SCENES[case]
    py = _venv_root() / venv / "bin/python"
    cfg = pathlib.Path(os.environ.get(cfg_env, str(_venv_root() / cfg_rel)))
    if not py.exists() or not (cfg / "config.yaml").is_file():
        pytest.skip(f"{py} / {cfg}/config.yaml not on this host")
    env = {**os.environ, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONDONTWRITEBYTECODE": "1", cfg_env: str(cfg)}
    if bench == "libero_safety":
        magick = _venv_root() / ".runtime/magickwand"
        env.setdefault("MAGICK_HOME", str(magick))
        env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "") + ":" + str(magick / "lib")
    out = subprocess.run([str(py), str(ROOT / "tests/benchmark/libero_gtto_scene_checks.py")] + argv, cwd=str(ROOT),
                         env=env, capture_output=True, text=True, timeout=900)
    last = [x for x in out.stdout.splitlines() if x.startswith("{")]
    if not last and "resolved to" in out.stderr:
        pytest.skip("the benchmark package of this checkout is not the interpreter's (git worktree without it)")
    assert last, out.stderr[-3000:]
    rec = json.loads(last[-1])
    legend = rec["legend"]
    assert legend[rec["target"]["target"]] == 1 and sorted(legend.values()) == [1] + list(range(3, len(legend) + 2))
    seen_target = False
    for cam, c in rec["cameras"].items():
        assert c["shape"] == c["depth_shape"] and c["dtype"] == "uint8", cam
        assert c["equals_gt_oracle_mask"] and c["robot_pixels_code0"] and c["table_pixels_code2"] in (True, None), cam
        if c["n_code1"] < 50:
            continue
        seen_target = True
        tp = c["target_projection"]
        assert tp["inside"] and tp["code_at_pixel"] == 1, (cam, tp)
        bp, fl = c["backprojected"], c["backprojected_flipped"]
        # (the affordance knife is ~0.2 m long: 81 % of its pixels lie within 0.10 m of its body origin)
        assert bp["median_dist_to_target_body_m"] < 0.08 and bp["frac_within_0.10m"] > 0.75, (cam, bp)
        assert fl["median_dist_to_target_body_m"] > bp["median_dist_to_target_body_m"], (cam, fl)
    assert seen_target
