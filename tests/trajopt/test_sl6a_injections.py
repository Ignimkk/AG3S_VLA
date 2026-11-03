"""SL6-A — robot injections of the AG3S final configuration (task `AG3S/ag3s/docs/handoff/SL6.task.md` §2 A3 · A4 · A6 ·
A8 · A9, VA1). Every injection's default is the RB-Y1 behaviour; the Panda values are tested on the Panda models.

1. Defaults = RB-Y1: `SafePolicy` class / constructor defaults, the `{hand}_arm_{j}` wrist rule, the NBC finger prefix
   rule, the A6 site (0, 0, -0.1), multi-seed `PALM_LINKS`, `gpu_seeds.RBY1_ROBOT` (constants unchanged).
2. Panda values: wrist joints `panda_joint5..7`, NBC fingers = both Panda fingers, A6 site = `panda_grip_site`
   (FK of the adapter URDF), multi-seed palm `panda_hand`, GPU seed robot (URDF / base / tool).
3. A9: `--hsr6-objective-scope` `all` (default) withholds NBC · A4 · G1 off-grasp; `nbc` withholds NBC only and records
   what `all` would have withheld; the flag's parse / refusals; RB-Y1 records gain no key by default.
4. A8: the Panda gripper spec's frame — in the gripper frame the Panda hand closes along x and approaches along -z.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt import hsr6 as H6
from AG3S.trajopt.config import TrajOptConfig
from AG3S.trajopt.grasp_latch import GraspPhase


# ------------------------------------------------------------------------------------------------ 1. defaults = RB-Y1
def test_safe_policy_injection_defaults_are_rby1():
    import inspect

    from AG3S.trajopt import safe_policy as sp

    sig = inspect.signature(sp.SafePolicy.__init__).parameters
    for name, default in (("hsr6_objective_scope", "all"), ("arm_joint_names", None), ("neighbour_finger_links", None),
                          ("hold_site_offset_m", None), ("palm_links", None), ("geo_gripper", None)):
        assert sig[name].default == default, name
    for name, default in (("hsr6_objective_scope", "all"), ("arm_joint_names", None), ("neighbour_finger_links", None),
                          ("hold_site_offset_m", None)):
        assert getattr(sp.SafePolicy, name) == default, name                       # test doubles without __init__
    assert sp.HOLD_LINK_SITE_OFFSET_M == (0.0, 0.0, -0.100)
    assert sp.rby1_arm_joint_name("left", 4) == "left_arm_4" and sp.HSR6_OBJECTIVE_SCOPES == ("all", "nbc")


def test_multi_seed_and_gpu_seed_constants_are_rby1():
    from AG3S.trajopt import gpu_seeds as gs
    from AG3S.trajopt import multi_seed as MS

    assert MS.PALM_LINKS == ("ee_left", "ee_right")
    assert gs.CUROBO_BASE_LINK == "link_torso_5" and gs.TOOL_FRAMES == ("ee_left", "ee_right")
    assert gs.RBY1_ROBOT == gs.GpuSeedRobot(name="rby1", urdf=None, base_link="link_torso_5",
                                            tool_frames=("ee_left", "ee_right"), fk_root_link="base")
    opt = types.SimpleNamespace(window_optimizer=None)
    assert gs.robot_of(opt) is gs.RBY1_ROBOT


def test_gpu_seed_robot_set_on_both_optimizers_and_refused_after_a_build():
    from AG3S.trajopt import gpu_seeds as gs

    window = types.SimpleNamespace()
    opt = types.SimpleNamespace(window_optimizer=window)
    robot = gs.GpuSeedRobot(name="x", urdf="/u.urdf", base_link="b", tool_frames=("t",))
    gs.set_robot(opt, robot)
    assert gs.robot_of(opt) is robot and gs.robot_of(window) is robot
    built = types.SimpleNamespace(window_optimizer=None, _gpu_batches={(16, 10): object()})
    with pytest.raises(RuntimeError, match="set_robot after"):
        gs.set_robot(built, robot)


class _Model:
    """A minimal robot model for `robot_spec` (one sphere, no fingers)."""

    def __init__(self, path=None):
        self.model = types.SimpleNamespace(path=path, joints=(), links=("root", "a"))
        self.joint_names = ("j0",)
        self._local_spheres = [("a", np.zeros(3), 0.01)]
        self.n_spheres = 1
        self.sphere_link_names = ("a",)
        self.param_joint_names = ()


def test_robot_spec_default_is_the_rby1_spec_and_the_panda_robot_changes_only_its_keys():
    from AG3S.trajopt import gpu_seeds as gs

    layout = types.SimpleNamespace(q_indices=(0,))
    limits = types.SimpleNamespace(lower=[-1.0], upper=[1.0], max_step=[0.1], max_step_change=[0.01])
    base = gs.robot_spec(_Model("/rby1.urdf"), layout, limits, 0.05)
    assert base["fk_root_link"] == "base" and base["curobo_base_link"] == "link_torso_5"
    assert base == gs.robot_spec(_Model("/rby1.urdf"), layout, limits, 0.05, gs.RBY1_ROBOT)
    panda = gs.GpuSeedRobot(name="panda", urdf="/panda.urdf", base_link="panda_link0", tool_frames=("panda_hand",),
                            fk_root_link="panda_link0")
    other = gs.robot_spec(_Model(None), layout, limits, 0.05, panda)
    assert other["urdf_path"].endswith("panda.urdf") and other["curobo_base_link"] == "panda_link0"
    assert other["fk_root_link"] == "panda_link0"
    assert {k: v for k, v in other.items() if k not in ("urdf_path", "curobo_base_link", "fk_root_link")} == \
           {k: v for k, v in base.items() if k not in ("urdf_path", "curobo_base_link", "fk_root_link")}


# ------------------------------------------------------------------------------------------------ 2. Panda values
@pytest.fixture(scope="module")
def panda_model():
    pytest.importorskip("yaml")
    from AG3S.adapters.panda import model as pm

    return pm.build_panda_constraint_model(pm.PANDA_GRIPPER_LINKS, palm_shell={"hands": "both"})


def _panda_policy(model, *, nbc=0.0, depth=0.0, wrist=1.0, hsr6=None, scope="all"):
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt.safe_policy import SafePolicy

    p = object.__new__(SafePolicy)
    p.to_config = TrajOptConfig.from_dict({"collision": {"neighbour_palm_contact_depth": depth,
                                                         "neighbour_contact_depth": nbc},
                                           "cost": {"wrist_track_scale": wrist}})
    p._latch = types.SimpleNamespace(phase=GraspPhase.LATCHED)
    p._last_grasp_record = {"state": "latched"}
    p._last_neighbour_contact, p._last_neighbour_palm_contact, p._last_wrist_track = {}, {}, {}
    p.constraint_robot_model = model
    p.grasp_links = {"left": ("panda_leftfinger", ("panda_leftfinger", "panda_rightfinger"))}
    p.hold_links = {"left": "panda_hand"}
    p.ag3s = types.SimpleNamespace(manipulated=None)
    p.hsr6, p.hsr6_objective_scope, p._last_hsr6, p._pending = hsr6, scope, {}, {}
    inj = ps.safe_policy_injections()
    p.arm_joint_names = inj["arm_joint_names"]
    p.neighbour_finger_links = inj["neighbour_finger_links"]
    p.hold_site_offset_m = inj["hold_site_offset_m"]
    from AG3S.adapters.panda import action as pa

    p.layout = pa.panda_chunk_layout(model.joint_names)
    return p


def _q0():
    return np.array([0.0, -0.161, 0.0, -2.44, 0.0, 2.23, 0.785])


def test_panda_site_is_the_grip_site_and_a6_rows_are_the_hand_spheres(panda_model):
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from AG3S.ag3s.robot_models.urdf_sphere_chain import PALM_SHELL_ROLE

    p = _panda_policy(panda_model, depth=0.010)
    q = _q0()
    site = np.asarray(_link_pose_numeric(panda_model, q, "panda_grip_site")[:3, 3])
    p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=site + np.array([0.0, 0.0, -0.03]))
    npc = p._neighbour_palm_contact_gate(q, ["left"])
    assert npc is not None and p._last_neighbour_palm_contact["site_to_target_mm"] == pytest.approx(30.0, abs=0.1)
    names = [str(n) for n in panda_model.sphere_link_names]
    roles = [str(r) for r in panda_model.sphere_roles]
    mask = npc["links_mask"]
    assert {n for n, m in zip(names, mask) if m} == {"panda_hand"}
    assert sum(1 for n, r, m in zip(names, roles, mask) if m and r == PALM_SHELL_ROLE) == 111
    assert int(mask.sum()) == 18 + 111                                                   # franka.yml hand + shell
    # at the hold link origin (97 mm behind the site): outside the 60 mm window
    origin = np.asarray(_link_pose_numeric(panda_model, q, "panda_hand")[:3, 3])
    assert np.linalg.norm(site - origin) == pytest.approx(0.097, abs=1e-9)
    p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=origin)
    assert p._neighbour_palm_contact_gate(q, ["left"]) is None


def test_panda_nbc_fingers_and_wrist_joints(panda_model):
    p = _panda_policy(panda_model, nbc=0.020, wrist=0.1)
    q = _q0()
    from AG3S.ag3s.constraints.attached import _link_pose_numeric

    palm = np.asarray(_link_pose_numeric(panda_model, q, "panda_hand")[:3, 3])
    p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=palm + np.array([0.0, 0.0, -0.08]))
    nbc = p._neighbour_contact_gate(q, ["left"])
    names = [str(n) for n in panda_model.sphere_link_names]
    assert nbc is not None and {n for n, m in zip(names, nbc["links_mask"]) if m} == {"panda_leftfinger",
                                                                                      "panda_rightfinger"}
    scale = p._wrist_track_gate(q, ["left"])
    assert p._last_wrist_track["joints"] == ["panda_joint5", "panda_joint6", "panda_joint7"]
    np.testing.assert_array_equal(scale, [1, 1, 1, 1, 0.1, 0.1, 0.1])
    # without the injection the RB-Y1 rules miss the Panda (what the port fixes): the finger prefix rule keeps one
    # finger ("panda_leftfinger" has no trailing digit), the `left_arm_j` rule names no Panda joint
    p.neighbour_finger_links, p.arm_joint_names = None, None
    rb = p._neighbour_contact_gate(q, ["left"])
    assert {n for n, m in zip(names, rb["links_mask"]) if m} == {"panda_leftfinger"}
    np.testing.assert_array_equal(p._wrist_track_gate(q, ["left"]), np.ones(7))


def test_multi_seed_palm_links_attribute(panda_model):
    from AG3S.trajopt import sqp

    opt = object.__new__(sqp.TrajectoryOptimizer)
    assert "palm_links" not in opt.__dict__                                              # default: MS.PALM_LINKS
    opt.palm_links = ("panda_hand",)
    assert opt.__dict__["palm_links"] == ("panda_hand",)


# ------------------------------------------------------------------------------------------------ 3. A9 scope
def _phase(p, phase):
    p._hsr6_start({"p": {c: (0.9 if c == phase else 0.02) for c in H6.PHASES6}, "argmax": phase})


@pytest.mark.parametrize("scope", ["all", "nbc"])
def test_objective_scope(scope, panda_model):
    p = _panda_policy(panda_model, hsr6=H6.Hsr6Flags(objective=True), scope=scope)
    _phase(p, "approach")
    assert p._hsr6_objective_ok(term="nbc") is False                                          # NBC: both scopes
    assert p._hsr6_objective_ok(term="a4") is (scope == "nbc")
    assert p._hsr6_objective_ok(term="g1") is (scope == "nbc")
    rec = p._last_hsr6
    if scope == "nbc":
        assert rec["objective_scope"] == "nbc"
        assert rec["detail"]["objective"] == {"a4_kept_by_scope": True, "g1_kept_by_scope": True}
        assert rec["changed"]["objective"] is False
    else:
        assert "objective_scope" not in rec and "objective" not in rec["detail"]
    _phase(p, "grasp")
    assert all(p._hsr6_objective_ok(term=t) for t in ("nbc", "a4", "g1"))
    anchor = object()
    _phase(p, "lift")
    p._last_grasp_anchor = {"active": True}
    assert (p._hsr6_a4_gate(anchor) is anchor) is (scope == "nbc")


def test_objective_scope_default_call_is_the_old_rule(panda_model):
    p = _panda_policy(panda_model, hsr6=H6.Hsr6Flags(objective=True))
    _phase(p, "approach")
    assert p._hsr6_objective_ok() is False and p._last_hsr6["detail"] == {}


def test_scope_flag_parse_and_refusals():
    from AG3S.trajopt import serve_safe as ss

    base = ["--config", "c", "--checkpoint", "k", "--model-xml", "m.xml"]
    ap = ss.build_parser()
    assert ap.parse_args(base).hsr6_objective_scope == "all"
    assert ss.hsr6_record_meta(ap.parse_args(base + ["--hsr6-objective", "--neighbour-contact-mm", "20"])) == {
        "hsr6": {"flags": ["objective"], "probe": "shipped:hsr6_t64_kv_L4", "decision": "raw argmax per request"}}
    on = ap.parse_args(base + ["--hsr6-objective", "--neighbour-contact-mm", "20", "--hsr6-objective-scope", "nbc"])
    assert ss.hsr6_record_meta(on)["hsr6"]["objective_scope"] == "nbc"
    for extra in (["--hsr6-objective-scope", "nbc"],
                  ["--hsr6-held-gate", "--held-body-cover", "--hsr6-objective-scope", "nbc"]):
        a = ap.parse_args(base + extra)
        with pytest.raises(SystemExit):
            ss.reject_hsr6_flags(ap, a)


def test_safe_policy_rejects_a_bad_scope():
    from AG3S.trajopt.safe_policy import SafePolicy

    with pytest.raises(ValueError, match="hsr6_objective_scope"):
        SafePolicy(object(), ag3s=types.SimpleNamespace(), hsr6_objective_scope="g1")


# ------------------------------------------------------------------------------------------------ 4. A8 frame
def test_panda_gripper_frame_closing_and_approach_directions(panda_model):
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from AG3S.ag3s.grasp.gripper import PANDA_LEFT, RBY1_LEFT, spec_for
    from AG3S.trajopt.geo_grasp import GeoGraspController

    assert RBY1_LEFT.frame is None and spec_for("left") is RBY1_LEFT
    F = np.asarray(PANDA_LEFT.frame)
    assert np.allclose(F @ F.T, np.eye(3)) and np.linalg.det(F) == pytest.approx(1.0)
    q = _q0()
    for opening in (0.0, 0.04):
        panda_model.set_joint_parameters({"panda_finger_joint1": opening, "panda_finger_joint2": opening})
        T_hand = np.asarray(_link_pose_numeric(panda_model, q, "panda_hand"))
        lf = np.asarray(_link_pose_numeric(panda_model, q, "panda_leftfinger")[:3, 3])
        rf = np.asarray(_link_pose_numeric(panda_model, q, "panda_rightfinger")[:3, 3])
        site = np.asarray(_link_pose_numeric(panda_model, q, "panda_grip_site")[:3, 3])
        R_G = T_hand[:3, :3] @ F
        if opening > 0:                                                                   # closing axis = x_G
            u = (lf - rf) / np.linalg.norm(lf - rf)
            assert abs(float(u @ R_G[:, 0])) == pytest.approx(1.0, abs=1e-9)
        approach = (site - T_hand[:3, 3]) / np.linalg.norm(site - T_hand[:3, 3])         # palm -> fingertips
        assert float(approach @ -R_G[:, 2]) == pytest.approx(1.0, abs=1e-9)              # = -z_G
        # the pad centre sits on the approach axis at the pad depth (gripper frame)
        assert PANDA_LEFT.pad_centre_depth == pytest.approx(0.5 * (0.1061 + 0.0854))
    panda_model.set_joint_parameters({"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04})
    # controller: hand pose in the gripper frame, goal rotation back to the hold link
    ctl = object.__new__(GeoGraspController)
    ctl.gripper_specs, ctl.model, ctl.hold_links = {"left": PANDA_LEFT}, panda_model, {"left": "panda_hand"}
    ctl.arm_joints = None
    T = ctl._hand_pose(q, "left")
    T_hand = np.asarray(_link_pose_numeric(panda_model, q, "panda_hand"))
    np.testing.assert_allclose(T[:3, 3], T_hand[:3, 3])
    np.testing.assert_allclose(ctl._to_hold(T[:3, :3], "left"), T_hand[:3, :3], atol=1e-12)
    rb = object.__new__(GeoGraspController)
    rb.gripper_specs, rb.model, rb.hold_links, rb.arm_joints = None, panda_model, {"left": "panda_hand"}, None
    np.testing.assert_array_equal(rb._hand_pose(q, "left"), T_hand)                      # RB-Y1 spec: no frame
    R = np.eye(3)
    assert rb._to_hold(R, "left") is R


def test_geo_arm_joint_injection_tracks_only_those_joints():
    from AG3S.trajopt.geo_grasp import GeoGraspController

    ctl = object.__new__(GeoGraspController)
    ctl.cost = types.SimpleNamespace(geo_track_scale=0.25, geo_carry_requests=3)
    ctl.last, ctl._anchor_active, ctl._hand, ctl._phase = {}, True, "left", "latched"
    ctl.arm_joints = {"left": tuple(f"panda_joint{i}" for i in range(1, 8))}
    names = [f"panda_joint{i}" for i in range(1, 8)]
    np.testing.assert_array_equal(ctl.track_scale(names), np.full(7, 0.25))
    ctl.arm_joints = None                                                                 # RB-Y1 rule
    np.testing.assert_array_equal(ctl.track_scale(names), np.ones(7))
    np.testing.assert_array_equal(ctl.track_scale(["left_arm_0", "right_arm_0"]), [0.25, 1.0])
