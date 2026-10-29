"""T48b — MuJoCo side: predicates, state reconstruction, render pose, edits (no GPU).

Synthetic (always): physical-state rules by hand, the crate volume test, host / empty-closure
selection on hand-built predicate sequences, and `classify` on a tiny MJCF scene (sphere fruit,
two finger pairs, table, crate) whose contacts are worked out by hand.
Real RB-Y1 model + shared storage (skipped when absent):
  - free-joint qpos addresses 31/38/45/52/59 read from the model;
  - state reconstruction from recorded qpos == recorded state (tolerance 0 / 1e-12);
  - render-pose rule (`observation_follows_step`) and a bit-identical re-render of one record;
  - edits: E1s / E2 change only the target fruit's free joint; E1 / E1z / E3 change the fruit and
    the used hand's two coupled finger joints, and the state changes only in the gripper dim, to
    |q(finger_x1)| / 0.045.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from AG3S.ag3s.experiments.studies import t48b_scene as sc  # noqa: E402

mujoco = pytest.importorskip("mujoco")

# ------------------------------------------------------------------------------- pure rules

F = {"left": (10, 11), "right": (20, 21)}
TABLE, CRATE = 1, 2


@pytest.mark.parametrize("touching,inside,want", [
    ({10, 11}, False, ("held", "left")),
    ({20, 21}, False, ("held", "right")),
    ({10, 11, TABLE}, False, ("grasping", "left")),
    ({20, 21, CRATE}, True, ("grasping", "right")),
    ({10, CRATE}, True, ("in_crate", None)),
    ({10, TABLE}, False, ("on_table", None)),
    ({TABLE}, False, ("on_table", None)),
    ({10, 20}, False, ("other", None)),
    (set(), False, ("other", None)),
    (set(), True, ("in_crate", None)),
])
def test_classify_contacts_rules(touching, inside, want):
    assert sc.classify_contacts(touching, F, TABLE, CRATE, inside) == want


def test_point_in_crate_rotated_frame():
    vol = (0.08, 0.14, -0.05, 0.06)
    yaw90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)   # crate x axis = world y
    c = np.array([1.0, 2.0, 0.5])
    assert sc.point_in_crate(c + [0.0, 0.07, 0.0], c, yaw90, vol)          # local x = 0.07 < 0.08
    assert not sc.point_in_crate(c + [0.0, 0.09, 0.0], c, yaw90, vol)      # local x = 0.09
    assert sc.point_in_crate(c + [-0.13, 0.0, 0.0], c, yaw90, vol)         # local y = 0.13 < 0.14
    assert not sc.point_in_crate(c + [0.0, 0.0, 0.07], c, yaw90, vol)      # above the rim
    assert not sc.point_in_crate(c + [0.0, 0.0, -0.06], c, yaw90, vol)     # below the floor top


def _ps(state, hand=None, fc=0):
    return sc.PhysState(state=state, hand=hand, touching=(), penetration=0.0, fruit_pos=(0, 0, 0),
                        finger_contact={"left": fc, "right": 0})


def test_held_and_reach_rows_by_hand():
    phys = [_ps("on_table"), _ps("on_table"), _ps("on_table", fc=1), _ps("grasping", "left", 2),
            _ps("held", "left", 2), _ps("held", "right"), _ps("held", "left", 2), _ps("in_crate")]
    g = np.array([1.0, 0.95, 1.02, 0.7, 0.72, 0.72, 0.72, 1.0])
    assert sc.held_rows(phys, "left").tolist() == [4, 6]
    assert sc.held_rows(phys, "right").tolist() == [5]
    # first closure (< 0.9) at row 3 -> rows 0..2, all on_table and open
    assert sc.reach_rows(phys, g).tolist() == [0, 1, 2]
    g2 = g.copy()
    g2[1] = 0.85                                     # a closing attempt at row 1
    assert sc.reach_rows(phys, g2).tolist() == [0]
    assert sc.reach_rows(phys, np.ones(8)).tolist() == [0, 1, 2]   # never closes: every on_table row


def test_empty_closure_values_hand_computed():
    phys = [_ps("on_table"), _ps("on_table"), _ps("on_table"), _ps("on_table"), _ps("on_table", fc=1),
            _ps("held", "left", 2), _ps("on_table")]
    g = np.array([1.0, 0.60, 0.45, 0.44, 0.44, 0.70, 0.43])
    # row1: |0.60-1.00| >= 0.02 (not settled); row2: |0.45-0.60| >= 0.02; row3: settled, no finger -> keep
    # row4: a finger touches -> drop; row5: held -> drop; row6: |0.43-0.70| not settled -> drop
    assert sc.empty_closure_values(phys, g, "left").tolist() == [0.44]


# ------------------------------------------------------------------------------- tiny MJCF scene

TINY = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <body name="table" pos="0 0 0.4"><geom name="table_top" type="box" size="0.5 0.5 0.02"/></body>
    <body name="crate" pos="0.3 0 0.48">
      <freejoint name="crate_free"/>
      <geom name="crate_floor" type="box" size="0.09 0.15 0.004" pos="0 0 -0.056"/>
      <geom name="crate_wall_px" type="box" size="0.004 0.15 0.056" pos="0.086 0 0.004"/>
      <geom name="crate_wall_nx" type="box" size="0.004 0.15 0.056" pos="-0.086 0 0.004"/>
      <geom name="crate_wall_py" type="box" size="0.09 0.004 0.056" pos="0 0.146 0.004"/>
      <geom name="crate_wall_ny" type="box" size="0.09 0.004 0.056" pos="0 -0.146 0.004"/>
    </body>
    <body name="apple" pos="0 -0.3 0.8"><freejoint name="apple_free"/><geom type="sphere" size="0.03"/></body>
    <body name="hand_l" pos="0 0 0.7">
      <body name="fl1" pos="0 -0.0345 0"><geom type="box" size="0.005 0.005 0.02"/></body>
      <body name="fl2" pos="0 0.0345 0"><geom type="box" size="0.005 0.005 0.02"/></body>
    </body>
    <body name="hand_r" pos="-0.3 0.3 0.449">
      <body name="fr1" pos="0 -0.0345 0"><geom type="box" size="0.005 0.005 0.02"/></body>
      <body name="fr2" pos="0 0.0345 0"><geom type="box" size="0.005 0.005 0.02"/></body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def tiny():
    m = mujoco.MjModel.from_xml_string(TINY)
    d = mujoco.MjData(m)

    def bid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

    def jadr(n):
        return int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)])

    idx = sc.SceneIndex(left_q=(), right_q=(), finger_q={}, free_q={"crate": jadr("crate_free"),
                                                                       "apple": jadr("apple_free")},
                        finger_body={"left": (bid("fl1"), bid("fl2")), "right": (bid("fr1"), bid("fr2"))},
                        hand_body={"left": bid("hand_l"), "right": bid("hand_r")}, fruit_body={"apple": bid("apple")},
                        table_body=bid("table"), crate_body=bid("crate"), crate_volume=sc.crate_inner_volume(m),
                        timestep=float(m.opt.timestep))
    return m, d, idx


def _place(m, d, idx, xyz):
    mujoco.mj_resetData(m, d)
    a = idx.free_q["apple"]
    d.qpos[a:a + 7] = [*xyz, 1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    return sc.classify(m, d, idx, "apple")


def test_crate_inner_volume_from_boxes(tiny):
    m, _, idx = tiny
    # floor half 0.09 x 0.15 minus 2 x wall half-thickness 0.004; floor top -0.052; wall top 0.06
    np.testing.assert_allclose(idx.crate_volume, (0.082, 0.142, -0.052, 0.06), atol=1e-12)


@pytest.mark.parametrize("xyz,state,hand", [
    ((0.0, 0.0, 0.7), "held", "left"),          # between the left pads, 0.5 mm into each
    ((-0.3, 0.3, 0.449), "grasping", "right"),  # between the right pads and 1 mm into the table top
    ((0.0, -0.2, 0.449), "on_table", None),     # 1 mm into the table top only
    ((0.3, 0.0, 0.459), "in_crate", None),      # centre 21 mm above the crate floor top
    ((0.0, -0.3, 0.8), "other", None),          # in the air
])
def test_classify_on_tiny_scene(tiny, xyz, state, hand):
    m, d, idx = tiny
    ph = _place(m, d, idx, xyz)
    assert (ph.state, ph.hand) == (state, hand)


def test_penetration_hand_computed(tiny):
    m, d, idx = tiny
    ph = _place(m, d, idx, (0.0, 0.0, 0.7))     # pad inner faces at +-0.0295, radius 0.03
    assert ph.penetration == pytest.approx(0.0005, abs=2e-5)
    assert ph.finger_contact == {"left": 2, "right": 0}
    ph = _place(m, d, idx, (0.0, 0.003, 0.7))    # 3 mm towards fl2: 3.5 mm into fl2, 2.5 mm gap to fl1
    assert ph.state == "other" and ph.penetration == pytest.approx(0.0035, abs=2e-5)


def test_pose_algebra_round_trip():
    hand_p, hand_q = np.array([0.1, -0.2, 0.9]), np.array([0.9, 0.1, -0.3, 0.2])
    hand_q /= np.linalg.norm(hand_q)
    obj = np.array([0.15, -0.18, 0.86, 0.7, 0.1, 0.6, -0.2])
    obj[3:] /= np.linalg.norm(obj[3:])
    rel = sc.relative_pose(hand_p, hand_q, obj)
    back = sc.compose_pose(hand_p, hand_q, rel)
    np.testing.assert_allclose(back[:3], obj[:3], atol=1e-12)
    assert abs(abs(np.dot(back[3:], obj[3:])) - 1.0) < 1e-12


def test_observation_follows_step_from_argv():
    base = ["pi05_infer.py", "--episode-index", "1807"]
    assert sc.observation_follows_step({"argv": base + ["--start-delay", "0"]}, 0) is False
    assert sc.observation_follows_step({"argv": base + ["--start-delay", "0"]}, 1) is True
    assert sc.observation_follows_step({"argv": base}, 0) is True                    # default 2.0 s
    assert sc.observation_follows_step({"argv": base + ["--start-delay", "0", "--warmup-steps", "1000"]}, 0) is True
    assert sc.observation_follows_step({"argv": base + ["--start-delay", "3", "--warmup-steps", "0"]}, 0) is False


# ------------------------------------------------------------------------------- real model + storage

STORAGE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage"))
EP1807 = STORAGE / "results/workspace/live_test/20260925_ep1807/run_0000"
needs_record = pytest.mark.skipif(not (EP1807 / "meta.json").exists(), reason="shared storage not mounted")


@pytest.fixture(scope="module")
def real():
    meta = json.loads((EP1807 / "meta.json").read_text())
    model, _ = sc.load_record_model(meta)
    idx = sc.SceneIndex.from_model(model)
    data = mujoco.MjData(model)
    recs = {}
    for i in (0, 5, 22, 25):
        with np.load(EP1807 / f"step_{i:05d}.npz") as z:
            recs[i] = {k: np.asarray(z[k]) for k in z.files}
    return meta, model, idx, data, recs


@needs_record
def test_free_joint_addresses_from_model(real):
    _, model, idx, _, _ = real
    assert idx.free_q == {"crate": 31, "apple": 38, "banana": 45, "orange": 52, "pear": 59}
    assert idx.finger_q == {"left": (27, 28), "right": (18, 19)}
    assert idx.left_q == tuple(range(20, 27)) and idx.right_q == tuple(range(11, 18))


@needs_record
def test_state_reconstruction_exact(real):
    _, _, idx, _, recs = real
    for r in recs.values():
        s = sc.reconstruct_state(idx, r["qpos"])
        np.testing.assert_allclose(s, r["state"], rtol=0, atol=1e-12)


@needs_record
def test_rerender_bit_identical_one_record(real):
    meta, model, idx, _, recs = real
    r = recs[22]
    ren = sc.PolicyRenderer(model)
    try:
        q = sc.render_qpos(model, r["qpos"], r["qvel"], after_step=sc.observation_follows_step(meta, 22))
        out = ren.render(q)
    finally:
        ren.close()
    par = sc.image_parity({c: r[f"image_{c}"] for c in sc.CAMS}, out)
    assert all(par[c]["max"] == 0 for c in sc.CAMS), par


def _ctx(idx, recs, model, data, **kw):
    q0 = sc.render_qpos(model, recs[0]["qpos"], recs[0]["qvel"], after_step=False)
    return sc.EditContext(idx=idx, fruit="apple", hand="left", initial_pose=sc.free_pose(q0, idx, "apple"),
                          g_empty=0.457, g_hold=0.7231, **kw)


def _changed(a, b):
    return set(np.flatnonzero(np.abs(np.asarray(a) - np.asarray(b)) > 0).tolist())


@needs_record
def test_e1s_touches_only_the_fruit(real):
    meta, model, idx, data, recs = real
    r = recs[25]                                  # held record (lifted apple)
    q_ren = sc.render_qpos(model, r["qpos"], r["qvel"])
    res = sc.edit_frame(model, data, _ctx(idx, recs, model, data), "E1s", r["qpos"], q_ren)
    assert _changed(res.q_render, q_ren) and _changed(res.q_render, q_ren) <= set(range(38, 45))
    np.testing.assert_array_equal(res.q_state, r["qpos"])
    np.testing.assert_array_equal(res.state, r["state"])


@needs_record
def test_e2_touches_only_the_fruit_and_the_used_fingers(real):
    """Task: E2 keeps every other object and robot joint; its fingers / state follow E1 (task
    table), so the used hand's two finger joints are the only robot qpos that change."""
    meta, model, idx, data, recs = real
    r = recs[25]
    q_ren = sc.render_qpos(model, r["qpos"], r["qvel"])
    res = sc.edit_frame(model, data, _ctx(idx, recs, model, data), "E2", r["qpos"], q_ren)
    assert _changed(res.q_render, q_ren) <= set(range(38, 45)) | {27, 28}
    assert _changed(res.q_state, r["qpos"]) == {27, 28}
    assert _changed(res.state, r["state"]) == {7} and res.state[7] == pytest.approx(0.457, abs=1e-12)
    # straight below: same xy as the held apple, z / orientation of the initial resting pose
    np.testing.assert_allclose(res.q_render[38:40], q_ren[38:40], atol=0)
    ctx = _ctx(idx, recs, model, data)
    np.testing.assert_allclose(res.q_render[40:45], ctx.initial_pose[2:], atol=1e-12)


@needs_record
@pytest.mark.parametrize("family,host,g", [("E1", 25, 0.457), ("E1z", 25, 0.0), ("E3", 5, None)])
def test_finger_edits_consistent_with_state(real, family, host, g):
    meta, model, idx, data, recs = real
    r = recs[host]
    q_ren = sc.render_qpos(model, r["qpos"], r["qvel"])
    res = sc.edit_frame(model, data, _ctx(idx, recs, model, data), family, r["qpos"], q_ren)
    allowed = set(range(38, 45)) | {27, 28}
    assert _changed(res.q_render, q_ren) <= allowed
    assert _changed(res.q_state, r["qpos"]) <= {27, 28}
    gv = res.info["g"]
    if g is not None:
        assert gv == g
    for q in (res.q_render, res.q_state):        # coupled fingers: x2 = -x1, x1 = -g * 0.045
        assert q[27] == pytest.approx(-gv * 0.045, abs=1e-15) and q[28] == pytest.approx(gv * 0.045, abs=1e-15)
    assert _changed(res.state, r["state"]) <= {7}
    assert res.state[7] == pytest.approx(abs(res.q_state[27]) / 0.045, abs=1e-15)
    assert res.state[7] == pytest.approx(gv, abs=1e-12)


@needs_record
def test_edit_outcomes_physical(real):
    meta, model, idx, data, recs = real
    ctx = _ctx(idx, recs, model, data)
    r = recs[25]
    q_ren = sc.render_qpos(model, r["qpos"], r["qvel"])
    e1 = sc.edit_frame(model, data, ctx, "E1", r["qpos"], q_ren)
    assert e1.included and e1.phys.state == "on_table" and e1.phys.finger_contact["left"] == 0
    r = recs[5]                                   # approach record, apple on the table
    q_ren = sc.render_qpos(model, r["qpos"], r["qvel"])
    e3 = sc.edit_frame(model, data, ctx, "E3", r["qpos"], q_ren)
    assert e3.included, e3.reason
    assert e3.phys.state == "held" and e3.phys.hand == "left"
    assert e3.phys.penetration <= sc.PENETRATION_MAX_M
    assert e3.info["placement"].startswith("pad_midpoint")   # no template given in this context
    assert 0.57 <= e3.info["g"] <= 0.83


@needs_record
def test_penetrating_edit_is_excluded(real):
    meta, model, idx, data, recs = real
    ctx = _ctx(idx, recs, model, data)
    r = recs[25]
    q_ren = sc.render_qpos(model, r["qpos"], r["qvel"])
    # put the "initial" pose straight into the closed hand: the apple overlaps the fingers
    bad = dataclass_replace(ctx, initial_pose=sc.free_pose(q_ren, idx, "apple"))
    res = sc.edit_frame(model, data, bad, "E1z", r["qpos"], q_ren)
    assert not res.included and res.reason == "penetration"


def dataclass_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)
