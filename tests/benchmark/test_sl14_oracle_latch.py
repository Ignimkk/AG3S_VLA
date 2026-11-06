"""SL14 -- ``--oracle-latch`` (the grasp latch's attach / detach follow the MuJoCo ground truth) and ``--placed-rearm`` (PLACED is not
absorbing). CPU only.

What this file pins:

(a) Off is unchanged. The server argv (``version_server_argv`` + the version's flags) of every one of the 61 versions that existed
    before SL13 is bit-identical (path-independent sha256, see ``test_sl13_oracle_volume.path_independent_argv``), the new dests join a
    record only when set, the server metadata, the call record and the latch config keep their keys, and a client without
    ``--oracle-grasp`` sends nothing new.
(b) The ground-truth grasp of the client: the target object touches the left **and** the right finger geoms (``data.contact``).
(c) The override on the real ``SafePolicy._advance_grasp`` (the T22 test harness): held and not HELD -> attach (also when the evidence
    would refuse: far object, no settled opening); not held and HELD -> detach and SEARCHING (and a new grasp is allowed); while HELD the
    evidence's release / revoke is void when the oracle still says held; the flag off ignores the request key.
(d) ``--placed-rearm``: PLACED -> SEARCHING once the measured opening is open, then the normal evidence path attaches the second grasp.
(e) The versions ``-pc-oraclelatch`` / ``-pc-rearm`` and the launchers.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import types

import numpy as np
import pytest

from tests.benchmark.test_sl6a_panda_final import ROOT, gpu_ok  # noqa: F401 (fixture)
from tests.benchmark.test_sl12_oracle_panda import MI, PC, _args, _libero_python
from tests.benchmark.test_sl13_oracle_volume import OLD_ARGV_SHA256, path_independent_argv
from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _lock, _manipulated, _policy
from AG3S.trajopt import wire
from AG3S.trajopt.grasp_latch import GraspLatch, GraspPhase, GraspSignal, LatchConfig

OL = f"{PC}-oraclelatch"
RE = f"{PC}-rearm"


# ---------------------------------------------------------------------------------------------------- (a) off is unchanged
def test_every_earlier_versions_server_argv_is_bit_identical():
    import hashlib

    from AG3S.adapters.panda import profile as pf

    old = {v: a for v, a in path_independent_argv(pf).items() if v not in pf.SL13_VERSIONS and v not in pf.SL14_VERSIONS and v not in pf.SL15_NTM_VERSIONS and v not in pf.SL15_SFHAND_VERSIONS and v not in pf.SL16_HSC_VERSIONS and v not in pf.SL16_HVC_VERSIONS}
    assert len(old) == 61
    assert hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest() == OLD_ARGV_SHA256
    assert pf.SL14_VERSIONS[:2] == (OL, RE)
    assert not any("--oracle-latch" in a or "--placed-rearm" in a for a in old.values())
    assert not set(pf.SL14_VERSIONS) & set(pf.PC_LATCH_VERSIONS) and not any(v.endswith("-lsfork") for v in pf.SL14_VERSIONS)


def test_the_path_independent_hash_does_not_see_the_checkout_path(monkeypatch):
    """The hashed argv has the root tokens in place of absolute paths: no ``/mnt`` path of a checkout is left in it."""
    from AG3S.adapters.panda import profile as pf

    text = json.dumps(path_independent_argv(pf))
    assert str(ROOT) not in text and "/mnt/dev" not in text


def test_off_by_default_in_parser_latch_config_and_class():
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.safe_policy import SafePolicy

    ap = ss.build_parser()
    B = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]
    off = ap.parse_args(B)
    on = ap.parse_args(B + ["--oracle-latch", "--placed-rearm"])
    assert off.oracle_latch is False and off.placed_rearm is False and on.oracle_latch is True and on.placed_rearm is True
    assert SafePolicy.oracle_latch is False and LatchConfig().placed_rearm is False
    # the RB-Y1 server (default --robot) refuses both
    for flag in ("--oracle-latch", "--placed-rearm"):
        with pytest.raises(SystemExit):
            ss.reject_panda_only_flags(ap, ap.parse_args(B + [flag]))
    ss.reject_panda_only_flags(ap, off)


def test_server_metadata_and_call_record_have_the_key_only_when_on():
    off = _policy()
    assert wire.ORACLE_GRASP_REQUIRED not in off.metadata
    on = _policy()
    on.oracle_latch = True
    assert on.metadata[wire.ORACLE_GRASP_REQUIRED] is True
    assert {k: v for k, v in on.metadata.items() if k != wire.ORACLE_GRASP_REQUIRED} == off.metadata
    # the call record: no oracle / rearm key unless the flag is on (evidence only, as before)
    q = np.zeros(len(off.constraint_robot_model.joint_names))
    off.ag3s._manipulated = _manipulated(_finger_origin(off, q) + np.array([0.05, 0.0, 0.0]))  # noqa: SLF001
    _lock(off)
    rec = _run(off, q, [(_fb(1, [1.0] * 8, 1.0), {"held": True, "target_body": "b_main"})])[0]
    assert "oracle" not in rec and "rearm" not in rec


def test_wire_constants_agree_with_the_client_module():
    from benchmark.adapters.libero_panda import oracle_codes as oc

    assert oc.REQUEST_GRASP_KEY == wire.PREFIX + wire.ORACLE_GRASP
    assert oc.SERVER_GRASP_REQUIRES_KEY == wire.ORACLE_GRASP_REQUIRED
    assert oc.CLIENT_GRASP_FLAG == "--oracle-grasp"
    from benchmark.adapters.libero_panda import phase_export

    assert oc.FINGER_BODIES == phase_export.FINGER_BODIES


# ---------------------------------------------------------------------------------------------------- (b) the ground-truth grasp
BODIES = ["world", "table", "robot0_base", "robot0_link7", "gripper0_right_gripper", "gripper0_leftfinger",
          "gripper0_finger_joint1_tip", "gripper0_rightfinger", "gripper0_finger_joint2_tip", "mount0_base",
          "akita_black_bowl_1_main", "akita_black_bowl_2_main", "plate_1_main"]
OBJECTS = ["akita_black_bowl_1", "akita_black_bowl_2", "plate_1"]
GEOM_BODY = list(range(len(BODIES))) + [BODIES.index("akita_black_bowl_1_main")]     # last geom = a second geom of bowl 1
G = {b: i for i, b in enumerate(BODIES)}
BOWL2 = len(BODIES)


def _oracle():
    from benchmark.adapters.libero_panda import oracle_codes as oc

    problem = {"goal": [["on", "akita_black_bowl_1", "plate_1"]], "obj_of_interest": [], "objects": OBJECTS}
    return oc.GraspOracle(problem, BODIES, GEOM_BODY, suite="libero_spatial", task_id=0)


def _data(*pairs):
    return types.SimpleNamespace(ncon=len(pairs), contact=[types.SimpleNamespace(geom1=a, geom2=b) for a, b in pairs] + [
        types.SimpleNamespace(geom1=0, geom2=0)])    # a stale slot beyond ncon is never read


def test_held_means_the_target_touches_both_fingers():
    go = _oracle()
    assert go.target == "akita_black_bowl_1"
    bowl, bowl_b = G["akita_black_bowl_1_main"], BOWL2
    lf, rf, lt, rt = (G["gripper0_leftfinger"], G["gripper0_rightfinger"], G["gripper0_finger_joint1_tip"], G["gripper0_finger_joint2_tip"])
    assert go.held(_data((bowl, lf), (rf, bowl))) is True                  # geom order does not matter
    assert go.held(_data((bowl_b, lt), (bowl, rt))) is True                # the tip bodies are fingers, the 2nd bowl geom is the target
    assert go.held(_data((bowl, lf))) is False and go.held(_data((rf, bowl))) is False       # one finger only
    assert go.held(_data((G["akita_black_bowl_2_main"], lf), (G["akita_black_bowl_2_main"], rf))) is False   # another object
    assert go.held(_data((bowl, G["table"]), (bowl, G["gripper0_right_gripper"]))) is False      # table / palm are not fingers
    assert go.held(_data((lf, rf), (bowl, bowl_b))) is False               # fingers touch each other, bowl touches itself
    assert go.held(_data()) is False
    keys = go.request_keys(_data((bowl, lf), (bowl, rf)))
    assert keys == {"ag3s/oracle_grasp": {"held": True, "target_body": "akita_black_bowl_1_main"}}
    assert go.record()["frames"] == 1 and go.record()["frames_held"] == 1 and go.record()["target"] == "akita_black_bowl_1"


def test_the_target_is_the_one_of_the_code_image():
    """Same ``choose_target`` as SL12 (the BDDL goal's first object, else the first movable ``obj_of_interest``)."""
    from benchmark.adapters.libero_panda import oracle_codes as oc

    problem = {"goal": [], "obj_of_interest": ["plate_1"], "objects": OBJECTS}
    assert oc.GraspOracle(problem, BODIES, GEOM_BODY).target == oc.choose_target(problem)[0] == "plate_1"


def test_flags_off_client_request_and_record_keep_their_keys():
    """The py3.8 client: without ``--oracle-grasp`` no ``ag3s/oracle_grasp`` in the request and no ``oracle_grasp`` in the episode
    record; with it exactly that key is added. The flag default is off in the parser, and the pairing with the server metadata holds."""
    py = _libero_python()
    if py is None:
        pytest.skip("no .venv-libero in this checkout or above it")
    code = r'''
import sys, json, types, tempfile, pathlib, numpy as np
sys.path.insert(0, "benchmark/runners")
import libero_safe_client as c
from benchmark.adapters.libero_panda import camera, wire_pack, oracle_codes as oc

BODIES = %s
GEOM_BODY = %s
OBJECTS = %s
seg = np.full((16, 16), -1, np.int16)

def frame():
    return {"depth_m": np.full((16, 16), 0.6, np.float32), "K": np.eye(3), "T_base_cam": np.eye(4), "T_world_base": np.eye(4),
            "robot_state": np.zeros(9), "stamp": 1.0, "segmentation_geom": seg}

class WS:
    def infer(self, req):
        return {"actions": np.zeros((10, 7)), "safe": True, "panda/mode": "vla_v11ab"}

res = {}
for name, on in (("off", False), ("on", True)):
    cl = c.SafeClient(WS(), 7, 0, 0, 5, oracle_grasp=on)
    frames = {"head": frame(), "left_wrist": frame()}
    cl.geometry = types.SimpleNamespace(frames=[frames], cameras=["head", "left_wrist"])
    cl.camera, cl.wire_pack = camera, wire_pack
    cl.eef = [(np.zeros(3), np.array([0, 0, 0, 1.0]))]
    if on:
        problem = {"goal": [["on", "akita_black_bowl_1", "plate_1"]], "obj_of_interest": [], "objects": OBJECTS}
        cl.grasp_oracle = oc.GraspOracle(problem, BODIES, GEOM_BODY, suite="s", task_id=3)
        data = types.SimpleNamespace(ncon=0, contact=[])
        cl._grasp_keys = cl.grasp_oracle.request_keys(data)
    req, _ = cl._request({"observation/image": 1})
    cl.infer({"observation/image": 1})
    cl.calls = [{"mode": "x", "hold": False}]
    d = pathlib.Path(tempfile.mkdtemp())
    res[name] = {"req_keys": sorted(req), "save_keys": sorted(cl.save(d, "stem")), "grasp": req.get("ag3s/oracle_grasp")}
for flag in (False, True):
    for required in (False, True):
        ws = types.SimpleNamespace(get_server_metadata=lambda r=required: {"oracle_grasp_required": True} if r else {})
        try:
            c.check_oracle_pairing(types.SimpleNamespace(oracle_segmentation=False, oracle_grasp=flag), ws)
            res["%%d%%d" %% (flag, required)] = "ok"
        except RuntimeError as exc:
            res["%%d%%d" %% (flag, required)] = str(exc)
# an old namespace without the dest (built before SL14) and a server without the key stay as before
c.check_oracle_pairing(types.SimpleNamespace(oracle_segmentation=False), types.SimpleNamespace(get_server_metadata=lambda: {}))
p = c.build_parser().parse_args(["--run-dir", "r", "--port", "1"])
res["default"] = [p.oracle_segmentation, p.oracle_grasp, p.capture_segmentation]
p = c.build_parser().parse_args(["--run-dir", "r", "--port", "1", "--oracle-grasp"])
res["flag"] = [p.oracle_grasp, p.capture_segmentation]
print(json.dumps(res))
''' % (json.dumps(BODIES), json.dumps(GEOM_BODY), json.dumps(OBJECTS))
    r = subprocess.run([str(py), "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0 and "No module named" in r.stderr:
        pytest.skip(r.stderr.splitlines()[-1])
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    off, on = res["off"], res["on"]
    assert "ag3s/oracle_grasp" not in off["req_keys"] and off["grasp"] is None
    assert set(on["req_keys"]) - set(off["req_keys"]) == {"ag3s/oracle_grasp"} and not set(off["req_keys"]) - set(on["req_keys"])
    assert on["grasp"] == {"held": False, "target_body": "akita_black_bowl_1_main"}
    assert set(on["save_keys"]) - set(off["save_keys"]) == {"oracle_grasp"} and not set(off["save_keys"]) - set(on["save_keys"])
    assert res["00"] == "ok" and res["11"] == "ok"
    assert "needs ag3s/oracle_grasp" in res["01"] and "does not run --oracle-latch" in res["10"]
    assert res["default"] == [False, False, False] and res["flag"] == [True, False]


# ---------------------------------------------------------------------------------------------------- (c) the override
def _run(sp, q, items):
    """Run ``_advance_grasp`` once per ``(feedback, oracle_grasp or None)``; yield the call records."""
    out = []
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q)]
    for fb, og in items:
        scene = {} if og is None else {wire.ORACLE_GRASP: og}
        sp._pending = {"scene": scene, "exec_feedback": fb}                    # noqa: SLF001
        sp._advance_grasp(obs, ["left"])                                       # noqa: SLF001
        sp._grasp_constraint_version = sp._grasp_state_version                 # noqa: SLF001
        sp._finish_grasp_record(types.SimpleNamespace(attached=sp.ag3s.attached))  # noqa: SLF001
        out.append(dict(sp._last_grasp_record))                                # noqa: SLF001
    return out


H = {"held": True, "target_body": "obj0_main"}
N = {"held": False, "target_body": "obj0_main"}
#: the T22 sequence: open, closing (CLOSING), same opening again (attach by evidence), hold chunk, two open requests (release)
SEQ = [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.0] * 8, 0.715),
       _fb(4, [0.72] * 8, 0.72, executed=[False] * 8), _fb(5, [1.0] * 8, 1.0), _fb(6, [1.0] * 8, 1.0)]


def _sp(distance=0.05, **kw):
    sp = _policy(**kw)
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(_finger_origin(sp, q) + np.array([distance, 0.0, 0.0]))  # noqa: SLF001
    _lock(sp)
    return sp, q


def test_oracle_held_attaches_before_the_evidence_would():
    sp, q = _sp()
    sp.oracle_latch = True
    recs = _run(sp, q, [(SEQ[0], N), (SEQ[1], H)])            # the evidence is still CLOSING (the opening has not settled)
    assert recs[0]["oracle"] == {"held": False, "overrode": None, "evidence_state": "latched", "evidence_attach": False,
                                 "evidence_detach": False, "target_body": "obj0_main"}
    assert recs[1]["oracle"]["evidence_state"] == "closing" and recs[1]["oracle"]["overrode"] == "attach"
    assert recs[1]["oracle"]["held_source"] == "manipulated"
    assert sp._latch.phase is GraspPhase.HELD and sp.ag3s.attached is not None      # noqa: SLF001
    assert recs[1]["state"] == "held" and recs[1]["transition"] and recs[1]["state_version"] == 1
    assert recs[1]["constraints_match_state"] and recs[1]["attached_in_constraints"]
    # the evidence alone would not have attached on this request
    ev, qe = _sp()
    assert _run(ev, qe, [(SEQ[0], None), (SEQ[1], None)])[1]["state"] == "closing" and ev.ag3s.attached is None


def test_oracle_held_attaches_a_far_object_the_reach_gate_would_refuse():
    sp, q = _sp(distance=0.636)
    sp.oracle_latch = True
    recs = _run(sp, q, [(SEQ[0], N), (SEQ[1], H)])
    assert recs[1]["oracle"]["overrode"] == "attach" and sp.ag3s.attached is not None
    assert recs[1]["oracle"]["reach_mm"] == pytest.approx(636.0, abs=1.0)


def test_oracle_not_held_detaches_to_searching_and_a_new_grasp_is_allowed():
    sp, q = _sp()
    sp.oracle_latch = True
    recs = _run(sp, q, [(SEQ[0], N), (SEQ[1], N), (SEQ[2], H)])          # the evidence attaches on the 3rd request, the oracle agrees
    assert recs[2]["attach"] is True and recs[2]["oracle"]["overrode"] is None and sp._latch.phase is GraspPhase.HELD  # noqa: SLF001
    rec = _run(sp, q, [(SEQ[3], N)])[0]                                  # the hand still holds on the evidence side; the oracle says no
    assert rec["oracle"]["overrode"] == "detach" and rec["oracle"]["evidence_state"] == "held"
    assert sp.ag3s.attached is None and sp._latch.phase is GraspPhase.SEARCHING   # noqa: SLF001
    assert rec["state"] == "searching" and rec["transition"] and rec["state_version"] == 2
    # re-grasp: the lock stays, the next latch update goes LATCHED, the oracle attaches on the next close
    sp._latch.update(label="obj0", score=0.9, runner_up=0.1, gripper=1.0)       # noqa: SLF001
    assert sp._latch.phase is GraspPhase.LATCHED                                  # noqa: SLF001
    again = _run(sp, q, [(_fb(7, [0.0] * 8, 0.715), H)])[0]
    assert again["oracle"]["overrode"] == "attach" and sp.ag3s.attached is not None and again["state_version"] == 3


def test_oracle_not_held_wins_over_an_evidence_attach():
    sp, q = _sp()
    sp.oracle_latch = True
    recs = _run(sp, q, [(SEQ[0], N), (SEQ[1], N), (SEQ[2], N)])
    assert recs[2]["oracle"]["evidence_attach"] is True and recs[2]["oracle"]["overrode"] == "detach"
    assert sp.ag3s.attached is None and sp._latch.phase is GraspPhase.SEARCHING   # noqa: SLF001


def test_while_held_the_evidence_release_is_void_when_the_oracle_still_says_held():
    sp, q = _sp()
    sp.oracle_latch = True
    items = [(fb, H if i >= 1 else N) for i, fb in enumerate(SEQ)]
    recs = _run(sp, q, items)
    assert [r["state"] for r in recs] == ["latched", "held", "held", "held", "held", "held"]
    assert recs[5]["oracle"]["evidence_detach"] is True and recs[5]["oracle"]["suppressed_detach"] is True
    assert recs[5]["detach"] is False and sp.ag3s.attached is not None
    assert recs[5]["state_version"] == recs[1]["state_version"] == 1             # ag3s was never told to detach
    # the oracle lets go: now it detaches
    rec = _run(sp, q, [(_fb(7, [1.0] * 8, 1.0), N)])[0]
    assert rec["oracle"]["overrode"] == "detach" and sp.ag3s.attached is None


def test_without_the_flag_the_request_key_is_ignored():
    sp, q = _sp()
    ref, qr = _sp()
    assert sp.oracle_latch is False
    recs = _run(sp, q, [(fb, H) for fb in SEQ])
    refs = _run(ref, qr, [(fb, None) for fb in SEQ])
    assert [r["attach"] for r in recs] == [False, False, True, False, False, False]
    assert [r["detach"] for r in recs] == [False, False, False, False, False, True]
    assert all("oracle" not in r for r in recs)
    assert [(r["state"], r["attach"], r["detach"], r["state_version"]) for r in recs] == \
           [(r["state"], r["attach"], r["detach"], r["state_version"]) for r in refs]
    assert [sorted(r) for r in recs] == [sorted(r) for r in refs]


def test_a_flagged_request_without_the_key_runs_the_evidence_alone_and_says_so():
    sp, q = _sp()
    sp.oracle_latch = True
    recs = _run(sp, q, [(fb, None) for fb in SEQ[:3]])
    assert recs[2]["attach"] is True and recs[2]["oracle"]["held"] is None and "ag3s/oracle_grasp" in recs[2]["oracle"]["error"]


def test_force_hold_and_release_on_the_latch():
    latch = GraspLatch(LatchConfig(), evidence=True)
    for _ in range(3):
        latch.update(label="apple", score=0.9, runner_up=0.1, gripper=1.0)
    latch.force_hold(0.7)
    assert latch.phase is GraspPhase.HELD and latch.holding
    latch.force_release()
    assert latch.phase is GraspPhase.SEARCHING and latch.manipulated == "apple"
    latch.update(label="apple", score=0.9, runner_up=0.1, gripper=1.0)
    assert latch.phase is GraspPhase.LATCHED


# ---------------------------------------------------------------------------------------------------- (d) --placed-rearm
def _open_signal(opening=1.0):
    return GraspSignal(available=True, feedback_seq=1, n_steps=8, n_exec=8, closing_steps=0, last_command=1.0, opening=opening)


def test_rearm_on_the_latch():
    for flag in (False, True):
        latch = GraspLatch(LatchConfig(placed_rearm=flag), evidence=True)
        latch.phase = GraspPhase.PLACED
        ev = latch.observe_grasp(_open_signal(0.3)).evidence       # a closed hand: PLACED stays either way
        assert latch.phase is GraspPhase.PLACED and "rearm" not in ev
        ev = latch.observe_grasp(_open_signal(1.0)).evidence
        assert ("rearm" in ev) is flag and latch.phase is (GraspPhase.SEARCHING if flag else GraspPhase.PLACED)


def test_the_second_grasp_attaches_after_a_release_with_the_flag_and_not_without():
    q = None
    results = {}
    for flag in (False, True):
        sp, q = _sp(latch=LatchConfig(placed_rearm=flag))
        recs = _run(sp, q, [(fb, None) for fb in SEQ])                     # grasp, release -> PLACED
        assert recs[5]["detach"] is True and sp._latch.phase is GraspPhase.PLACED   # noqa: SLF001
        assert ("rearm" in recs[5]) is flag and (recs[5].get("rearm") is not True)
        r7 = _run(sp, q, [(_fb(7, [1.0] * 8, 1.0), None)])[0]
        assert r7.get("rearm") is (True if flag else None)
        assert sp._latch.phase is (GraspPhase.SEARCHING if flag else GraspPhase.PLACED)   # noqa: SLF001
        sp._latch.update(label="obj0", score=0.9, runner_up=0.1, gripper=1.0)            # noqa: SLF001  (what _run_latch does)
        second = _run(sp, q, [(_fb(8, [0.0] * 8, 0.715), None), (_fb(9, [0.0] * 8, 0.715), None),
                              (_fb(10, [0.0] * 8, 0.715), None)])
        results[flag] = (sp.ag3s.attached is not None, [r["state"] for r in second], second[-1].get("rearm"))
    assert results[False] == (False, ["placed", "placed", "placed"], None)
    assert results[True][0] is True and results[True][1][-1] == "held" and results[True][2] is False


# ---------------------------------------------------------------------------------------------------- (e) versions and launchers
def test_versions_are_pc_plus_one_flag(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf, server as ps

    reg = pf.AG3S_VERSIONS
    assert list(reg[OL]) == list(reg[PC]) + ["--oracle-latch"] and list(reg[RE]) == list(reg[PC]) + ["--placed-rearm"]
    base = _args(PC, monkeypatch)
    for v, dest, other in ((OL, "oracle_latch", "placed_rearm"), (RE, "placed_rearm", "oracle_latch")):
        assert pf.is_final_version(v) and pf.version_server_argv(v) == pf.version_server_argv(PC)
        assert pf.version_server_flags(v)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]          # a probe-taking pc variant
        assert pf.version_server_flags(v, hsr6_probe="/p.npz")[-2:] == ["--hsr6-probe", "/p.npz"]
        a = _args(v, monkeypatch)
        ps.check_unported_flags(a, None)
        assert getattr(a, dest) is True and getattr(a, other) is False
        assert pf.match_ag3s_version(a) == v
        vals = pf.version_flag_values(a)
        assert vals[dest] is True and vals[other] is False       # the two SL14 dests join together, only when one is set
        assert {k: x for k, x in vals.items() if k not in (dest, other)} == pf.version_flag_values(base)   # nothing else joined
    bv = pf.version_flag_values(base)
    assert "oracle_latch" not in bv and "placed_rearm" not in bv and pf.match_ag3s_version(base) == PC
    for v in (MI, f"{PC}-gap", f"{PC}-oraclegt", f"{PC}-oraclevol"):
        vals = pf.version_flag_values(_args(v, monkeypatch))
        assert "oracle_latch" not in vals and "placed_rearm" not in vals
    assert pf.needs_oracle_grasp(OL) and not pf.needs_oracle_grasp(RE) and not pf.needs_oracle_grasp(PC)
    assert not pf.needs_oracle_segmentation(OL) and not pf.needs_oracle_segmentation(RE)


def test_launchers_pass_the_client_flag_only_for_the_oracle_latch_version(gpu_ok):
    import argparse

    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_safety_paired as lsp

    for v in (OL, RE):
        got = lp.server_argv("vla_v11ab", "/ck", 9, v, hsr6_probe="/p.npz")
        assert got[-2:] == ["--hsr6-probe", "/p.npz"] and ("--oracle-latch" in got) is (v == OL) and ("--placed-rearm" in got) is (v == RE)
        assert "--target-extent" not in got or got[got.index("--target-extent") + 1] != "oracle-visible"
        a = lp.build_parser().parse_args(["--task-suite-name", "libero_spatial", "--task-ids", "0,1,2,3,4,6,7,8,9", "--episode-ids", "0",
                                          "--run-id", "x", "--conversion", "C", "--seed", "7", "--ag3s-version", v])
        want = ["--oracle-grasp"] if v == OL else []
        assert lp.oracle_client_argv("vla_v11ab", a) == want and lp.oracle_client_argv("vla_only", a) == []
        settings = lp.oracle_settings("vla_v11ab", a)
        assert ("oracle_grasp" in settings) is (v == OL) and "oracle_segmentation" not in settings
        assert lp.oracle_settings("vla_only", a) == {}
        wrapped = lsp.make_server_argv("pi05_libero_safety")("vla_v11ab", "/ck", 9, v)
        assert ("--oracle-latch" in wrapped) is (v == OL) and wrapped[wrapped.index("--robot") + 1] == "panda"
    # every other version: the client argv and the manifest are what they were
    for v in (PC, MI, "V11AB", f"{PC}-gap", f"{PC}-oraclegt", f"{PC}-oraclevol"):
        a = argparse.Namespace(ag3s_version=v)
        assert lp.oracle_client_argv("vla_v11ab", a) == ([] if "oracle" not in v else ["--oracle-segmentation"])
        assert ("oracle_grasp" in lp.oracle_settings("vla_v11ab", a)) is False
    assert lp.oracle_client_argv("vla_v11ab", argparse.Namespace()) == []


def test_the_oracle_latch_version_is_refused_on_safelibero(monkeypatch, gpu_ok):
    from benchmark.runners import launch_libero_paired as lp

    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(lp.PROJECT_ROOT))
    with pytest.raises(ValueError, match="LIBERO and LIBERO-Safety only"):
        lp.main(["--task-suite-name", "safelibero_spatial", "--safety-level", "I", "--task-ids", "0", "--episode-ids", "0",
                 "--run-id", "x", "--conversion", "C", "--ag3s-version", OL, "--data-split", "x", "--dry-run"])
