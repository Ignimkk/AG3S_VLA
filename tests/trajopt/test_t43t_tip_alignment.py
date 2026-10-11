"""T43 T (측정만) — `fingertip_alignment`: 손끝 중점 ↔ AG3S target centroid.

Recorded inputs (T42-1 E3b ep1908 s19082, right arm) where available; skipped otherwise.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib

import numpy as np
import pytest

from AG3S.trajopt.safe_policy import TIP_ALIGNMENT_DEFINITION, fingertip_alignment

RUN = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify/T42/1/runs/E3b/noobs/E3b_ep1908_s19082'))
SERVER = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify/T42/1/runs/E3b/server_8250_20261005T083818'))
OFFSET = 449          # chunk index = seq + 449 (raw/chunk_index_t42.json: seq 1 -> chunk_00450)

pytestmark = pytest.mark.skipif(not (RUN / "frames.jsonl").exists(), reason="T42-1 records not here")


@pytest.fixture(scope="module")
def rec():
    from AG3S.ag3s.robot_models.gripper_state import FingerJointMap
    from AG3S.trajopt.experiments import fixed_input_ablation as FIA
    from AG3S.trajopt.types import ChunkLayout

    class _A:
        links = sphere_spacing = max_spheres_per_capsule = capsule_radius_scale = None
        max_sphere_radius = esdf_margin = target_field_policy = phase = manipulators = None
        model_xml = hand = rows_per_step = None
        allow_uncertified = False

    man = json.loads((RUN / "manifest.json").read_text())
    man["scene"]["model_xml"] = (_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml'))
    st = FIA.resolve_settings(_A, FIA.parse_server_log(str(SERVER / "serve_safe.log")), man)
    robot, qadr, _ = FIA.build_robot(st)
    plan = {r["seq"]: r for r in (json.loads(x) for x in (RUN / "frames.jsonl").read_text()
                                  .splitlines() if x.strip()) if r.get("kind") == "planning"}
    return dict(robot=robot, qadr=qadr, layout=ChunkLayout.rby1(robot.joint_names), plan=plan,
                fmap=FingerJointMap(robot), FIA=FIA)


def _chunk(seq):
    z = np.load(SERVER / f"server_constraints/run_0000/chunk_{seq + OFFSET:05d}.npz")
    assert int(z["t_step"]) == seq
    return json.loads(str(z["summary_json"])), np.asarray(z["reference_chunk"], float)


def _at(rec, seq):
    s, ref = _chunk(seq)
    q = rec["FIA"].q_now_from_qpos(np.asarray(rec["plan"][seq]["qpos"], float), rec["qadr"])
    rec["robot"].set_joint_parameters(rec["fmap"].joint_values(s["finger_joints"]["opening_norm"]))
    return s, ref, q


def test_definition_and_decomposition(rec):
    s, ref, q = _at(rec, 14)
    c = np.asarray(s["manipulated"]["centroid"], float)
    out = fingertip_alignment(rec["robot"], q, ("ee_finger_r1", "ee_finger_r2"), "ee_right", c,
                              chunk=ref, layout=rec["layout"], gripper_column=15)
    now = out["now"]
    d = np.asarray(now["tip_m"]) - c
    assert now["d3_mm"] == pytest.approx(np.linalg.norm(d) * 1e3, abs=0.02)
    assert now["dxy_mm"] == pytest.approx(np.linalg.norm(d[:2]) * 1e3, abs=0.02)
    assert now["dz_mm"] == pytest.approx(d[2] * 1e3, abs=0.02)
    assert now["d3_mm"] ** 2 == pytest.approx(now["dxy_mm"] ** 2 + now["dz_mm"] ** 2, rel=1e-3)
    assert "midpoint of the two finger tips" in TIP_ALIGNMENT_DEFINITION


def test_tip_is_between_the_fingers_and_beyond_the_palm(rec):
    s, ref, q = _at(rec, 14)
    robot = rec["robot"]
    out = fingertip_alignment(robot, q, ("ee_finger_r1", "ee_finger_r2"), "ee_right", np.zeros(3))
    tip = np.asarray(out["now"]["tip_m"])
    o1 = robot.link_pose(q, "ee_finger_r1")[:3, 3]
    o2 = robot.link_pose(q, "ee_finger_r2")[:3, 3]
    palm = robot.link_pose(q, "ee_right")[:3, 3]
    # the midpoint lies (nearly) on the symmetry plane between the two finger links
    n = (o2 - o1) / np.linalg.norm(o2 - o1)
    assert abs(float((tip - 0.5 * (o1 + o2)) @ n)) < 0.005
    # and farther from the palm origin than the finger link origins
    assert np.linalg.norm(tip - palm) > np.linalg.norm(0.5 * (o1 + o2) - palm)


def test_zero_when_centroid_is_the_tip(rec):
    _, _, q = _at(rec, 14)
    a = fingertip_alignment(rec["robot"], q, ("ee_finger_r1", "ee_finger_r2"), "ee_right",
                            np.zeros(3))
    b = fingertip_alignment(rec["robot"], q, ("ee_finger_r1", "ee_finger_r2"), "ee_right",
                            a["now"]["tip_m"])
    assert b["now"]["d3_mm"] < 0.01


def test_plan_close_row_is_the_first_close_command(rec):
    s, ref, q = _at(rec, 14)
    chunk = np.array(ref, float)
    chunk[:, 15] = 1.0
    out = fingertip_alignment(rec["robot"], q, ("ee_finger_r1", "ee_finger_r2"), "ee_right",
                              np.zeros(3), chunk=chunk, layout=rec["layout"], gripper_column=15)
    assert out["plan_close"] is None                       # no close command, nothing planned
    chunk[3:, 15] = 0.0
    chunk[5, 15] = 1.0
    out = fingertip_alignment(rec["robot"], q, ("ee_finger_r1", "ee_finger_r2"), "ee_right",
                              np.zeros(3), chunk=chunk, layout=rec["layout"], gripper_column=15)
    assert out["plan_close"]["row"] == 3 and out["plan_close"]["command"] == 0.0
    # the planned tip is the FK of that row's arm joints (row 0 of a constant chunk = same pose)
    const = np.repeat(chunk[3:4], 8, axis=0)
    o2 = fingertip_alignment(rec["robot"], q, ("ee_finger_r1", "ee_finger_r2"), "ee_right",
                             np.zeros(3), chunk=const, layout=rec["layout"], gripper_column=15)
    assert o2["plan_close"]["row"] == 0
    assert o2["plan_close"]["tip_m"] == out["plan_close"]["tip_m"]


def test_does_not_touch_the_model_parameters(rec):
    _, _, q = _at(rec, 14)
    before = rec["robot"].param_vector().copy()
    fingertip_alignment(rec["robot"], q, ("ee_finger_r1", "ee_finger_r2"), "ee_right", np.zeros(3))
    assert np.array_equal(before, rec["robot"].param_vector())


def test_closing_chunk_of_the_right_arm_run(rec):
    """seq 15 = the chunk the latch went latched -> closing (record). Value printed for the impl note."""
    s, ref, q = _at(rec, 15)
    assert s["grasp"]["state"] == "closing"
    out = fingertip_alignment(rec["robot"], q, ("ee_finger_r1", "ee_finger_r2"), "ee_right",
                              s["manipulated"]["centroid"], chunk=ref, layout=rec["layout"],
                              gripper_column=15)
    assert np.isfinite(out["now"]["d3_mm"])
    print("ep1908 s19082 seq 15 tip alignment:", json.dumps(out))


# --- SafePolicy wiring (synthetic, no records needed) ----------------------------------------

def test_safe_policy_records_tip_alignment_per_chunk_and_changes_nothing_else():
    from tests.trajopt.test_grasp_evidence import (
        _fb, _finger_origin, _grasp_frames, _lock, _manipulated, _policy)

    def run(patch_out):
        import AG3S.trajopt.safe_policy as SP

        sp = _policy()
        q = np.zeros(len(sp.constraint_robot_model.joint_names))
        apple = _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0])
        sp.ag3s._manipulated = _manipulated(apple)                                 # noqa: SLF001
        _lock(sp)
        orig = SP.fingertip_alignment
        if patch_out:
            SP.fingertip_alignment = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        try:
            recs = list(_grasp_frames(sp, q, [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715),
                                              _fb(3, [0.0] * 8, 0.715)]))
        finally:
            SP.fingertip_alignment = orig
        return sp, recs, apple

    sp, recs, apple = run(False)
    for r in recs:
        tip = r["tip_alignment"]
        assert tip["definition"] == TIP_ALIGNMENT_DEFINITION
        assert tip["fingers"] == ["ee_finger_l1", "ee_finger_l2"] and tip["hold_link"] == "ee_left"
        assert tip["centroid_m"] == pytest.approx(list(apple), abs=1e-5)
        assert np.isfinite(tip["now"]["d3_mm"])
        assert tip["plan_close"] is None                     # no chunk in `_pending`
    # a failing measurement is recorded, and the rest of the record is the same as without it
    sp2, recs2, _ = run(True)
    assert all("error" in r["tip_alignment"] for r in recs2)
    def strip(rs):      # the measurement key, and the wall-clock attach stamp (differs run to run)
        out = []
        for r in rs:
            r = {k: v for k, v in r.items() if k != "tip_alignment"}
            if isinstance(r.get("attached"), dict):
                r["attached"] = {k: v for k, v in r["attached"].items() if k != "attached_at"}
            out.append(r)
        return out
    assert json.dumps(strip(recs), sort_keys=True, default=str) == json.dumps(
        strip(recs2), sort_keys=True, default=str)
    assert recs[-1]["state"] == "held" and recs[-1]["attach"]
