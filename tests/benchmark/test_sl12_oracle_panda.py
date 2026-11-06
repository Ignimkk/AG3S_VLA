"""SL12 -- the visible-target oracle (RB-Y1 T46e A13) on the Panda LIBERO / LIBERO-Safety clients and server. CPU only.

What this file pins:

(a) Off is unchanged. A client without ``--oracle-segmentation`` builds the same request (no oracle key), its call rows and
    episode record keep their keys; the Panda server adds ``panda/oracle`` only for ``--target-extent oracle-visible``; every
    earlier version's registry entry is the one of main (sha256), the new flags are refused for the Panda server exactly as
    before unless the extent is oracle-visible.
(b) The code image. The geom -> code table on a LIBERO Spatial body list (target 1, other movable objects 3.., table / fixtures 2,
    robot and no-hit 0) agrees with the AG3S ``oracle_segmentation.label_image`` table; the request keys unpack on the server
    (``wire.unpack_oracle_segmentation``) and feed the server's own ``OracleMemory``.
(c) The target choice per suite (BDDL goal) -- LIBERO Spatial / Object agree with SL3's ``task_target``; LIBERO-Safety examples;
    a fixture-only goal is refused.
(d) Versions ``-pc-oraclegt`` / ``-pc-mi10-oraclegt``; the launchers pass the client flag for them and for nothing else; the
    client and the server metadata must agree.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess

import numpy as np
import pytest

from tests.benchmark.test_sl6a_panda_final import ROOT, gpu_ok  # noqa: F401 (fixture)

PC = "AG3S-final-panda-v3-e0to-pc"
MI = f"{PC}-mi10"
NEW = (f"{PC}-oraclegt", f"{MI}-oraclegt")
#: sha256 of ``json.dumps({name: list(flags)}, sort_keys=True)`` over the 59 AG3S_VERSIONS entries of main 5f473482 (measured on
#: this registry with the SL12 entries left out; SL12 only adds two entries and a helper)
OLD_REGISTRY_SHA256 = "388005f7eba418cc460fa912651a9ab04b3fa85c82422485375c1b3efb2ff57b"
BDDL_LIBERO = ROOT / "benchmark/LIBERO/libero/libero/bddl_files"
BDDL_SAFETY = ROOT / "benchmark/LIBERO-Safety/libero/libero/bddl_files"

#: body list of a LIBERO Spatial scene (model_body_names of the recorded run) and one geom per body
BODIES = ["world", "table", "robot0_base", "robot0_link0", "robot0_link7", "robot0_right_hand", "gripper0_right_gripper",
          "gripper0_leftfinger", "mount0_base", "akita_black_bowl_1_main", "akita_black_bowl_2_main", "cookies_1_main",
          "glazed_rim_porcelain_ramekin_1_main", "plate_1_main", "wooden_cabinet_1_main", "wooden_cabinet_1_cabinet_top",
          "flat_stove_1_main", "flat_stove_1_burner"]
OBJECTS = ["akita_black_bowl_1", "akita_black_bowl_2", "cookies_1", "glazed_rim_porcelain_ramekin_1", "plate_1"]
#: geom g belongs to body g (geom 18 = a second geom of bowl 1)
GEOM_BODY = list(range(len(BODIES))) + [BODIES.index("akita_black_bowl_1_main")]


def _libero_python():
    for base in [ROOT, *ROOT.parents]:
        cand = base / ".venv-libero/bin/python"
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------------------------------- (b) the code image
def test_code_table_target_other_static_robot():
    from benchmark.adapters.libero_panda import oracle_codes as oc

    lut, codes = oc.geom_code_table(BODIES, GEOM_BODY, "akita_black_bowl_1", OBJECTS)
    assert codes == {"akita_black_bowl_1": 1, "akita_black_bowl_2": 3, "cookies_1": 4,
                     "glazed_rim_porcelain_ramekin_1": 5, "plate_1": 6}
    code_of = lambda body: int(lut[GEOM_BODY.index(BODIES.index(body)) + 1])   # noqa: E731
    assert code_of("akita_black_bowl_1_main") == oc.CODE_TARGET and int(lut[len(BODIES) + 1]) == oc.CODE_TARGET
    assert [code_of(b) for b in ("akita_black_bowl_2_main", "cookies_1_main", "plate_1_main")] == [3, 4, 6]
    assert [code_of(b) for b in ("world", "table", "wooden_cabinet_1_main", "wooden_cabinet_1_cabinet_top",
                                 "flat_stove_1_burner")] == [oc.CODE_STATIC] * 5
    assert [code_of(b) for b in ("robot0_base", "robot0_link7", "gripper0_leftfinger", "mount0_base")] == [oc.CODE_NONE] * 4
    seg = np.array([[-1, 9, 18, 1], [0, 10, 17, 3]], np.int16)       # no hit, bowl1, bowl1 (2nd geom), table / world, bowl2, burner, link0
    out = oc.label_geom_image(seg, lut)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 1, 1, 2], [2, 3, 2, 0]]
    assert oc.label_geom_image(np.array([[-5, 400]]), lut).tolist() == [[0, 0]]          # outside the table never raises
    with pytest.raises(ValueError, match="not a movable object"):
        oc.geom_code_table(BODIES, GEOM_BODY, "mug_1", OBJECTS)
    with pytest.raises(ValueError, match="no MuJoCo body"):
        oc.geom_code_table(BODIES, GEOM_BODY, "plate_1", OBJECTS + ["mug_1"])
    with pytest.raises(ValueError, match="exceed"):
        oc.geom_code_table(BODIES, GEOM_BODY, "plate_1", ["plate_1"] + [f"o{i}" for i in range(40)])


def test_code_table_agrees_with_the_ag3s_label_image():
    """The py3.8 replica gives the code image ``oracle_segmentation.label_image(code_table(...))`` gives for the same scene."""
    from AG3S.ag3s.runtime.oracle_segmentation import code_table, label_image
    from benchmark.adapters.libero_panda import oracle_codes as oc

    lut, codes = oc.geom_code_table(BODIES, GEOM_BODY, "akita_black_bowl_2", OBJECTS)
    # AG3S table: bodies named by object (the root-name rule), robot predicate = ours
    names = {i: (b[:-len("_main")] if b.endswith("_main") and b[:-5] in OBJECTS else b) for i, b in enumerate(BODIES)}
    blut, bcodes = code_table(names, "akita_black_bowl_2", objects=OBJECTS, is_robot=oc.is_robot_body)
    assert bcodes == codes
    geom_img = np.random.RandomState(0).randint(-1, len(GEOM_BODY), (32, 32)).astype(np.int16)
    body_img = np.where(geom_img >= 0, np.asarray(GEOM_BODY)[np.clip(geom_img, 0, None)], -1)
    # the replica folds the part-of-object rule into the table; compare on the bodies AG3S names exactly (cabinet / stove parts
    # are static in both)
    assert np.array_equal(oc.label_geom_image(geom_img, lut), label_image(body_img, blut))


def test_request_keys_unpack_on_the_server_and_feed_the_memory():
    from AG3S.ag3s.runtime.oracle_segmentation import CODE_TARGET, OracleMemory, OracleView
    from AG3S.trajopt import wire
    from benchmark.adapters.libero_panda import oracle_codes as oc

    assert oc.REQUEST_PREFIX == wire.PREFIX + wire.ORACLE_SEG + "/" and oc.REQUEST_META_KEY == wire.PREFIX + wire.ORACLE_SEG_META
    assert oc.SERVER_REQUIRES_KEY == wire.ORACLE_SEG_REQUIRED and oc.CODE_VERSION == 1
    from AG3S.ag3s.runtime import oracle_segmentation as osg
    assert (oc.CODE_NONE, oc.CODE_TARGET, oc.CODE_STATIC, oc.CODE_OBJECT_BASE) == (
        osg.CODE_NONE, osg.CODE_TARGET, osg.CODE_STATIC, osg.CODE_OBJECT_BASE)

    problem = {"goal": [["on", "akita_black_bowl_1", "plate_1"]], "obj_of_interest": ["akita_black_bowl_1", "plate_1"],
               "objects": OBJECTS}
    lab = oc.OracleLabeler(problem, BODIES, GEOM_BODY, suite="libero_spatial", task_id=1)
    seg = np.full((16, 16), -1, np.int16)
    seg[4:8, 4:8] = 9                      # bowl 1
    seg[10:14, 2:6] = 1                    # table
    frames = {"head": {"segmentation_geom": seg}, "left_wrist": {"segmentation_geom": seg}}
    keys = lab.request_keys(frames)
    assert set(keys) == {"ag3s/oracle_seg/head", "ag3s/oracle_seg/left_wrist", "ag3s/oracle_seg_meta"}
    assert keys["ag3s/oracle_seg/head"].dtype == np.uint8 and keys["ag3s/oracle_seg/head"].shape == (16, 16)
    assert keys["ag3s/oracle_seg_meta"]["target"] == "akita_black_bowl_1" and keys["ag3s/oracle_seg_meta"]["version"] == 1
    assert lab.last_pixels == {"head": 16, "left_wrist": 16} and lab.record()["rule"] == "first_goal_predicate_first_argument"
    scene = {"cameras": ["head", "left_wrist"], **{k[len("ag3s/"):]: v for k, v in keys.items()}}
    seg_in = wire.unpack_oracle_segmentation(scene)
    assert set(seg_in["labels"]) == {"head", "left_wrist"} and seg_in["meta"]["codes"]["plate_1"] == 6
    # the server's own memory labels the target samples from a flat plane of depth (camera looking along +z)
    depth = np.full((16, 16), 0.6)
    K = np.array([[20.0, 0, 7.5], [0, 20.0, 7.5], [0, 0, 1.0]])
    mem = OracleMemory(cell=0.0025, roi_margin=0.065, carve_tol=0.01, depth_min=0.0, depth_max=2.0)
    rec = mem.update([OracleView(depth, K, np.eye(4), seg_in["labels"]["head"], "head")], legend=seg_in["meta"]["codes"])
    assert rec["n_target_samples"] == 16 * 0 + int((mem.codes == CODE_TARGET).sum()) > 0
    assert mem.legend["akita_black_bowl_1"] == 1


def test_frame_without_segmentation_is_refused():
    from benchmark.adapters.libero_panda import oracle_codes as oc

    problem = {"goal": [["on", "plate_1", "x"]], "obj_of_interest": [], "objects": OBJECTS}
    lab = oc.OracleLabeler(problem, BODIES, GEOM_BODY)
    with pytest.raises(KeyError, match="segmentation_geom"):
        lab.request_keys({"head": {}})


# ---------------------------------------------------------------------------------------------------- (c) target per suite
def test_target_choice_agrees_with_task_target_on_libero_spatial_and_object():
    from benchmark.adapters.libero_panda import oracle_codes as oc, task_target as tt

    seen = 0
    for suite in ("libero_spatial", "libero_object"):
        for f in sorted((BDDL_LIBERO / suite).glob("*.bddl")):
            problem = tt.parse_bddl(f)
            want, _dest = tt.goal_roles(problem)
            got, rule = oc.choose_target(oc.problem_from_parsed(problem))
            assert got == want, (f.name, got, want)
            assert rule == "first_goal_predicate_first_argument"
            seen += 1
    assert seen >= 20


def test_target_choice_libero_safety_examples():
    from benchmark.adapters.libero_panda import oracle_codes as oc, task_target as tt

    if not BDDL_SAFETY.exists():
        pytest.skip("LIBERO-Safety not in this checkout")
    want = {"affordance/L1/grab_the_knife_and_cut_the_lemon_on_the_cutting_board": "knife_1",
            "affordance/L1/pass_me_the_hammer": "hammer_1",
            "affordance/L1/pour_the_water_in_the_classic_blue_mug_to_the_bowl": "classic_blue_mug_1",
            "affordance/L1/put_the_frypan_on_the_stove": "frypan_1",
            "human_safety/L1/pick_the_akita_black_bowl_next_to_the_cookies_box_and_bring_it_for_me": "akita_black_bowl_1",
            "obstacle_avoidance/L1/put_the_white_yellow_mug_in_the_microwave_and_close_it": "white_yellow_mug_1",
            "obstacle_avoidance/L1/put_both_moka_pots_on_the_stove": "moka_pot_1"}
    for rel, target in want.items():
        problem = oc.problem_from_parsed(tt.parse_bddl(BDDL_SAFETY / (rel + ".bddl")))
        assert oc.choose_target(problem)[0] == target, rel
    # every LIBERO-Safety task of the three levels of the four suites has a target (or is refused loudly, never silently)
    n = 0
    for f in sorted(BDDL_SAFETY.glob("*/L*/*.bddl")):
        problem = oc.problem_from_parsed(tt.parse_bddl(f))
        try:
            assert oc.choose_target(problem)[0] in problem["objects"]
        except ValueError:
            pass
        n += 1
    assert n > 30


def test_fixture_only_goal_and_obj_of_interest_fallback():
    from benchmark.adapters.libero_panda import oracle_codes as oc, task_target as tt

    f = BDDL_LIBERO / "libero_goal/open_the_middle_drawer_of_the_cabinet.bddl"
    with pytest.raises(ValueError, match="no movable pick target"):
        oc.choose_target(oc.problem_from_parsed(tt.parse_bddl(f)))
    assert oc.choose_target({"goal": [["open", "cabinet_1_top"]], "obj_of_interest": ["plate_1"], "objects": ["plate_1"]}) == (
        "plate_1", "first_movable_obj_of_interest")
    # the env form (robosuite_parse_problem): goal_state + objects as {category: [names]}
    env_form = {"goal_state": [["on", "b_1", "p_1"]], "obj_of_interest": ["b_1"], "objects": {"b": ["b_1"], "p": ["p_1"]}}
    assert oc.choose_target(oc.problem_from_parsed(env_form))[0] == "b_1"


# ---------------------------------------------------------------------------------------------------- (a) off is unchanged
def test_every_earlier_version_is_unchanged():
    from AG3S.adapters.panda import profile as pf

    old = {k: list(v) for k, v in pf.AG3S_VERSIONS.items() if k not in NEW and k not in pf.SL13_VERSIONS and k not in pf.SL14_VERSIONS and k not in pf.SL15_NTM_VERSIONS and k not in pf.SL15_SFHAND_VERSIONS and k not in pf.SL16_HSC_VERSIONS}   # SL13 adds the -oraclevol pair
    assert len(old) == 59
    assert hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest() == OLD_REGISTRY_SHA256
    assert pf.SL12_VERSIONS == NEW
    assert not any(pf.needs_oracle_segmentation(k) for k in old)
    # the new entries are not in the lists the earlier tests slice (PC_LATCH_VERSIONS tail, LSFORK twins)
    assert not set(NEW) & set(pf.PC_LATCH_VERSIONS) and not any(v.endswith("-lsfork") for v in NEW)


def test_flags_off_client_request_and_rows_keep_their_keys():
    """The py3.8 client: no ``--oracle-segmentation`` -> no oracle key in the request, no labeller, no ``oracle`` in the row or
    the episode record; with it exactly the three oracle keys are added and the server's ``panda/oracle`` lands in the row."""
    py = _libero_python()
    if py is None:
        pytest.skip("no .venv-libero in this checkout or above it")
    code = r'''
import sys, json, types, logging, numpy as np
sys.path.insert(0, "benchmark/runners")
import libero_safe_client as c
from benchmark.adapters.libero_panda import camera, wire_pack, oracle_codes as oc

BODIES = %s
GEOM_BODY = %s
OBJECTS = %s
seg = np.full((16, 16), -1, np.int16); seg[4:8, 4:8] = 9

def frame():
    return {"depth_m": np.full((16, 16), 0.6, np.float32), "K": np.eye(3), "T_base_cam": np.eye(4), "T_world_base": np.eye(4),
            "robot_state": np.zeros(9), "stamp": 1.0, "segmentation_geom": seg}

class WS:
    def __init__(self, extra): self.extra = extra
    def infer(self, req):
        out = {"actions": np.zeros((10, 7)), "safe": True, "panda/mode": "vla_v11ab"}
        out.update(self.extra)
        return out

res = {}
for name, oracle, extra in (("off", False, {}), ("on", True, {"panda/oracle": {"field": {"applied": True}}})):
    cl = c.SafeClient(WS(extra), 7, 0, 0, 5, oracle=oracle)
    frames = {"head": frame(), "left_wrist": frame()}
    cl.geometry = types.SimpleNamespace(frames=[frames], cameras=["head", "left_wrist"])
    cl.camera, cl.wire_pack = camera, wire_pack
    cl.eef = [(np.zeros(3), np.array([0, 0, 0, 1.0]))]
    if oracle:
        problem = {"goal": [["on", "akita_black_bowl_1", "plate_1"]], "obj_of_interest": [], "objects": OBJECTS}
        cl.labeler = oc.OracleLabeler(problem, BODIES, GEOM_BODY)
    req, _ = cl._request({"observation/image": 1})
    cl.infer({"observation/image": 1})
    cl.eef.append(cl.eef[0])
    row = cl.calls[-1]
    res[name] = {"req_keys": sorted(req), "row_keys": sorted(row), "oracle_row": row.get("oracle"),
                 "dtype": str(req["ag3s/oracle_seg/head"].dtype) if oracle else None,
                 "has_oracle_save": None}
import tempfile, pathlib
for name, oracle in (("off", False), ("on", True)):
    cl = c.SafeClient(None, 7, 0, 0, 5, oracle=oracle)
    cl.calls = [{"mode": "x", "hold": False}]
    if oracle:
        problem = {"goal": [["on", "akita_black_bowl_1", "plate_1"]], "obj_of_interest": [], "objects": OBJECTS}
        cl.labeler = oc.OracleLabeler(problem, BODIES, GEOM_BODY, suite="s", task_id=3)
    d = pathlib.Path(tempfile.mkdtemp())
    res[name]["save_keys"] = sorted(cl.save(d, "stem"))
    res[name]["oracle_record"] = cl.labeler.record() if oracle else None
print(json.dumps(res))
''' % (json.dumps(BODIES), json.dumps(GEOM_BODY), json.dumps(OBJECTS))
    r = subprocess.run([str(py), "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0 and "No module named" in r.stderr:
        pytest.skip(r.stderr.splitlines()[-1])
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    off, on = res["off"], res["on"]
    assert set(on["req_keys"]) - set(off["req_keys"]) == {"ag3s/oracle_seg/head", "ag3s/oracle_seg/left_wrist",
                                                          "ag3s/oracle_seg_meta"}
    assert set(off["req_keys"]) - set(on["req_keys"]) == set() and on["dtype"] == "uint8"
    assert set(on["row_keys"]) - set(off["row_keys"]) == {"oracle"} and off["oracle_row"] is None
    assert on["oracle_row"]["field"]["applied"] is True
    assert set(on["save_keys"]) - set(off["save_keys"]) == {"oracle"} and off["oracle_record"] is None
    assert on["oracle_record"]["target"] == "akita_black_bowl_1" and on["oracle_record"]["task_id"] == 3


def test_client_pairing_with_the_server_metadata():
    py = _libero_python()
    if py is None:
        pytest.skip("no .venv-libero in this checkout or above it")
    code = r'''
import sys, json, types
sys.path.insert(0, "benchmark/runners")
import libero_safe_client as c
res = {}
for flag in (False, True):
    for required in (False, True):
        ws = types.SimpleNamespace(get_server_metadata=lambda r=required: {"oracle_segmentation_required": True} if r else {})
        try:
            c.check_oracle_pairing(types.SimpleNamespace(oracle_segmentation=flag), ws)
            res["%d%d" % (flag, required)] = "ok"
        except RuntimeError as exc:
            res["%d%d" % (flag, required)] = str(exc)
p = c.build_parser().parse_args(["--run-dir", "r", "--port", "1"])
res["default"] = [p.oracle_segmentation, p.capture_segmentation]
p = c.build_parser().parse_args(["--run-dir", "r", "--port", "1", "--oracle-segmentation"])
res["flag"] = p.oracle_segmentation
print(json.dumps(res))
'''
    r = subprocess.run([str(py), "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0 and "No module named" in r.stderr:
        pytest.skip(r.stderr.splitlines()[-1])
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res["00"] == "ok" and res["11"] == "ok"
    assert "needs the GT code" in res["01"] and "does not run the visible-target oracle" in res["10"]
    assert res["default"] == [False, False] and res["flag"] is True


# ---------------------------------------------------------------------------------------------------- (d) versions, server, launchers
def _args(v, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(v)
    argv[argv.index("--config") + 1] = "pi05_libero"
    return pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(v))


def test_versions_are_the_base_minus_nontarget_evidence_plus_the_oracle_flags():
    from AG3S.adapters.panda import profile as pf

    reg = pf.AG3S_VERSIONS
    for v, base in zip(NEW, (PC, MI)):
        want = []
        it = iter(reg[base])
        for tok in it:
            if tok == "--nontarget-evidence":
                next(it)
            else:
                want.append(tok)
        assert list(reg[v]) == want + ["--target-extent", "oracle-visible", "--oracle-extent-identity", "gt"]
        assert pf.is_final_version(v) and pf.needs_oracle_segmentation(v)
        assert pf.version_server_argv(v) == pf.version_server_argv(base)
        assert pf.version_server_flags(v)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]   # the HSR6 rules stay: the probe is taken
    assert "--minimal-intervention-mm" in reg[NEW[1]] and "--minimal-intervention-mm" not in reg[NEW[0]]
    assert not pf.needs_oracle_segmentation(PC) and not pf.needs_oracle_segmentation("unknown")


def test_versions_parse_pass_the_panda_flag_check_and_read_back(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf, server as ps

    for v, base in zip(NEW, (PC, MI)):
        a = _args(v, monkeypatch)
        ps.check_unported_flags(a, None)
        assert (a.target_extent, a.oracle_extent_identity, a.nontarget_evidence) == ("oracle-visible", "gt", None)
        assert pf.match_ag3s_version(a) == v
        b = _args(base, monkeypatch)
        assert pf.match_ag3s_version(b) == base
        assert "oracle_extent_identity" not in pf.version_flag_values(b) and "oracle_extent_identity" in pf.version_flag_values(a)
        assert a.target_field_policy in ("exclude-authorized", "exclude-all")


def test_other_extents_and_the_oracle_dests_stay_refused(gpu_ok):
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss

    B = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml", "--robot", "panda"]
    ap = ss.build_parser()
    ps.check_unported_flags(ap.parse_args(B + ["--target-extent", "oracle-visible", "--oracle-extent-identity", "gt",
                                               "--oracle-max-dist-voxels", "5"]), None)
    with pytest.raises(ValueError, match="oracle_extent_identity|oracle-extent-identity"):
        ps.check_unported_flags(ap.parse_args(B + ["--target-extent", "attention-surface", "--oracle-extent-identity", "gt"]), None)
    with pytest.raises(ValueError, match="attention-surface-hybrid"):
        ps.check_unported_flags(ap.parse_args(B + ["--target-extent", "attention-surface-hybrid"]), None)
    assert "oracle_extent_identity" in ps.UNPORTED_FINAL_CONFIG_DESTS and "oracle_extent_identity" in ps.SL12_ORACLE_DESTS


def test_server_record_exists_only_for_the_oracle_extent():
    from AG3S.adapters.panda import server as ps

    mk = lambda ext, **kw: type("S", (), {"ag3s": type("A", (), {   # noqa: E731
        "config": type("C", (), {"esdf": type("E", (), {"target_extent": ext})()})(), **kw})()})()
    assert ps.oracle_call_record(mk("estimated")) is None and ps.oracle_call_record(mk("attention_surface")) is None
    assert ps.oracle_call_record(type("S", (), {})()) is None
    stats = {"target_free": {"oracle": {"applied": True, "identity_matches_gt": False, "pipeline_code": 3, "n_target_samples": 40,
                                        "pipeline_instance": {"votes": {3: 5}, "why": "x"}}}}
    field = type("F", (), {"stats": stats})()
    rec = ps.oracle_call_record(mk("oracle_visible", _oracle_frame={"segmentation": True, "cameras": ["head"],
                                                                     "memory": {"n_samples": 9}}, _last_esdf_field=field))
    assert rec["segmentation"] is True and rec["cameras"] == ["head"] and rec["memory"]["n_samples"] == 9
    assert rec["field"]["applied"] is True and rec["field"]["identity_matches_gt"] is False
    assert rec["field"]["pipeline_instance"]["why"] == "x"
    json.dumps(rec)


def test_launchers_pass_the_client_flag_only_for_the_oracle_versions(gpu_ok):
    import argparse

    from AG3S.adapters.panda import profile as pf
    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_safety_paired as lsp

    for v, base in zip(NEW, (PC, MI)):
        got = lp.server_argv("vla_v11ab", "/ck", 9, v, hsr6_probe="/p.npz")
        assert got[got.index("--target-extent") + 1] == "oracle-visible" and "--nontarget-evidence" not in got
        assert got[-2:] == ["--hsr6-probe", "/p.npz"]
        a = lp.build_parser().parse_args(["--task-ids", "1", "--episode-ids", "1", "--run-id", "x", "--conversion", "C",
                                          "--ag3s-version", v])
        assert a.ag3s_version == v and pf.version_server_flags(v, **lp.version_options(a))
        assert lp.oracle_client_argv("vla_v11ab", a) == ["--oracle-segmentation"]
        assert lp.oracle_client_argv("vla_only", a) == []           # the VLA-only client renders nothing extra
        assert lp.oracle_settings("vla_v11ab", a)["oracle_segmentation"]["target_rule"]
        assert lp.oracle_settings("vla_only", a) == {}
        wrapped = lsp.make_server_argv("pi05_libero_safety")("vla_v11ab", "/ck", 9, v)
        assert wrapped[wrapped.index("--target-extent") + 1] == "oracle-visible" and wrapped[wrapped.index("--robot") + 1] == "panda"
    for v in (PC, MI, "V11AB", f"{PC}-surface", f"{PC}-gap"):
        a = argparse.Namespace(ag3s_version=v)
        assert lp.oracle_client_argv("vla_v11ab", a) == [] and lp.oracle_settings("vla_v11ab", a) == {}
    assert lp.oracle_client_argv("vla_v11ab", argparse.Namespace()) == []        # a namespace built before SL12


def test_oracle_versions_are_refused_on_safelibero(monkeypatch, gpu_ok):
    from benchmark.runners import launch_libero_paired as lp

    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(lp.PROJECT_ROOT))
    with pytest.raises(ValueError, match="LIBERO and LIBERO-Safety only"):
        lp.main(["--task-suite-name", "safelibero_spatial", "--safety-level", "I", "--task-ids", "0", "--episode-ids", "0",
                 "--run-id", "x", "--conversion", "C", "--ag3s-version", NEW[0], "--data-split", "x", "--dry-run"])


@pytest.mark.parametrize("version", NEW)
def test_oracle_versions_build_on_cpu_and_the_server_asks_for_the_code_image(version, gpu_ok, monkeypatch):
    """The Panda server (stub policy, no checkpoint): oracle-visible in the AG3S config, identity gt, the metadata key the client
    checks, the config dump entry; the base version has none of them."""
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import wire
    from tests.benchmark.test_sl6a_panda_final import _Stub

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    served, info = ps.build_safe_server(_Stub(), _args(version, monkeypatch), gpu_warm_up=False)
    e = served._policy.ag3s.config.esdf
    assert e.target_extent == "oracle_visible" and e.oracle_extent_identity == "gt" and e.nontarget_evidence == "proximity"
    assert e.target_exclusion == "ball_owned" and not e.target_free_coarse
    meta = served.metadata
    assert meta[wire.ORACLE_SEG_REQUIRED] is True and meta["target_extent"] == "oracle_visible"
    assert meta["oracle_extent_identity"] == "gt"
    fc = info["final_config"]
    assert fc["version"] == version and fc["oracle_visible"]["identity"] == "gt" and "attention_surface" not in fc
    json.dumps(info, default=str)
    base = PC if version == NEW[0] else MI
    served0, info0 = ps.build_safe_server(_Stub(), _args(base, monkeypatch), gpu_warm_up=False)
    assert wire.ORACLE_SEG_REQUIRED not in served0.metadata and "oracle_visible" not in info0["final_config"]
    assert served0._policy.ag3s.config.esdf.target_extent == "estimated"
