"""AG3S-CORE-EVAL-A6 — light contact of the grasping palm with a neighbour object at the grasp moment
(`collision.neighbour_palm_contact_depth`, `serve_safe --neighbour-palm-contact-mm`, default off) and the HSR6 ②
fix (held-object allowance withdrawn only in approach · retreat).

Task: `AG3S/ag3s/docs/handoff/AG3S-CORE-EVAL-A6.task.md` §2 (A6 rule) · §3 (② fix) · §4 A1 (this file).

What this file pins:
1. Linearizer: off (None / depth 0) is bit-identical; on, only robot rows of the masked (palm) links whose centre is
   within the radius of the target centre and whose nearest surface is an obstacle get the margin `-depth`; target,
   support (table), held rows, other links, far rows and the destination label (crate / basket) keep theirs. The
   palm-shell 5 mm margin is replaced. The −10 mm boundary: 10 mm of overlap is feasible, 11 mm is not. Record.
2. `SafePolicy._neighbour_palm_contact_gate`: the "grasp moment" window (latch latched / closing, palm site within
   60 mm of the target) true / false, the rows = the hold link's spheres (not fingers, not the other hand), HSR6 ④
   does not withdraw it, the solve's rows are merged into the record.
3. The rows on the real server model (A8 `--palm-shell on`): hand-body + palm-shell spheres of `ee_<side>`; the
   hold link site is the MJCF `left_ee` / `right_ee` site.
4. ② fix: grasp ∧ HELD keeps the allowance, approach · retreat withdraw it, the rest unchanged.
5. Config / CLI: default off, keys absent from `to_dict`, the final argv parses to 10 mm / 60 mm / 120 mm.
6. A recorded chunk (scene 119, combination test HSR draw 0): the solver's final check and the record see the same
   relaxed rows; off has no key.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import shlex
import types

import numpy as np
import pytest

from AG3S.trajopt import hsr6 as H6
from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.grasp_latch import GraspPhase
from tests.trajopt.test_t41a_obstacle_margin import BAR_R, M, R_FINGER, ROWS, TABLE_PLANE, _clear, _rig, _scene

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
DEPTH = 0.010


def _npc(centre, mask=(True, True, True, True), depth=DEPTH, radius=0.05):
    return {"links_mask": np.asarray(mask, bool), "depth": depth, "centre": np.asarray(centre, np.float64),
            "radius": radius}


# ------------------------------------------------------------------------------------------------ 1. linearizer
def test_off_is_bit_identical_and_sets_no_record():
    lin, centres, field, points = _rig()
    base = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE))
    assert "_last_neighbour_palm_contact" not in lin.__dict__
    zero = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE,
                                       neighbour_palm_contact=_npc(field.bar, depth=0.0)))
    np.testing.assert_array_equal(zero, base)
    assert "_last_neighbour_palm_contact" not in lin.__dict__
    from AG3S.trajopt.linearize import SceneSnapshot

    assert SceneSnapshot.__dataclass_fields__["neighbour_palm_contact"].default is None


def test_only_masked_obstacle_rows_near_the_target_are_relaxed():
    lin, centres, field, points = _rig()
    base = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE))
    on = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE,
                                     neighbour_palm_contact=_npc(field.bar)))
    bar = ROWS["bar"]
    assert base[bar] == pytest.approx(0.008 - R_FINGER - M, abs=1e-12)
    assert on[bar] == pytest.approx(0.008 - R_FINGER + DEPTH, abs=1e-12)              # margin -10 mm
    others = [i for i in range(len(on)) if i != bar]
    np.testing.assert_array_equal(on[others], base[others])                          # apple / table / box / held
    rec = lin._last_neighbour_palm_contact
    assert rec["n_rows"] == 1 and rec["n_spheres"] == 1 and rec["n_rows_below_old_margin"] == 1
    assert rec["min_clearance_mm"] == pytest.approx((0.008 - R_FINGER) * 1000.0, abs=1e-3)
    assert rec["depth_mm"] == pytest.approx(10.0) and "n_destination_excluded" not in rec
    # the bar row's link not in the mask -> unchanged (fingers / the other hand / arm)
    off_mask = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE,
                                           neighbour_palm_contact=_npc(field.bar, mask=(False, True, True, True))))
    np.testing.assert_array_equal(off_mask, base)
    assert lin._last_neighbour_palm_contact["n_rows"] == 0
    assert lin._last_neighbour_palm_contact["min_clearance_mm"] is None
    # outside the radius -> unchanged
    far = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE,
                                      neighbour_palm_contact=_npc(field.bar + np.array([0.0, 0.0, 0.2]))))
    np.testing.assert_array_equal(far, base)


@pytest.mark.parametrize("row", ["table", "apple", "held"])
def test_table_target_and_held_rows_are_not_neighbours(row):
    lin, centres, field, points = _rig()
    base = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE))
    i = ROWS[row]
    on = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE,
                                     neighbour_palm_contact=_npc(centres[0, i], radius=0.02)))
    assert on[i] == base[i]


def test_destination_label_is_not_a_neighbour_and_an_unreadable_label_relaxes_nothing():
    lin, centres, field, points = _rig()

    class _Labelled(type(field)):
        has_labels = True

        def is_label(self, pts, label):
            p = np.asarray(pts, np.float64).reshape(-1, 3)
            return np.linalg.norm(p - self.bar, axis=1) < 0.05                       # the "crate" is the bar

    crate = _Labelled()
    kw = dict(margin=M, support_planes=TABLE_PLANE, destination_label="crate", destination_margin=0.02)
    base = _clear(lin, centres, _scene(crate, points, **kw))
    on = _clear(lin, centres, _scene(crate, points, **kw, neighbour_palm_contact=_npc(crate.bar)))
    np.testing.assert_array_equal(on, base)
    assert lin._last_neighbour_palm_contact["n_destination_excluded"] == 1
    assert lin._last_neighbour_palm_contact["n_rows"] == 0
    # a destination the field cannot name: fail closed (no row relaxed)
    base2 = _clear(lin, centres, _scene(field, points, **kw))
    on2 = _clear(lin, centres, _scene(field, points, **kw, neighbour_palm_contact=_npc(field.bar)))
    np.testing.assert_array_equal(on2, base2)
    assert lin._last_neighbour_palm_contact["reason"] == "destination label unreadable"


def test_palm_shell_margin_is_replaced_by_minus_depth():
    lin, centres, field, points = _rig()
    lin.robot_model = types.SimpleNamespace(sphere_roles=("palm_shell", "collision", "collision", "collision"))
    kw = dict(margin=M, support_planes=TABLE_PLANE, palm_shell_margin=0.005)
    base = _clear(lin, centres, _scene(field, points, **kw))
    on = _clear(lin, centres, _scene(field, points, **kw, neighbour_palm_contact=_npc(field.bar)))
    bar = ROWS["bar"]
    assert base[bar] == pytest.approx(0.008 - R_FINGER - 0.005, abs=1e-12)            # palm shell: >= 5 mm
    assert on[bar] - base[bar] == pytest.approx(0.005 + DEPTH, abs=1e-12)              # now >= -10 mm
    assert lin._last_neighbour_palm_contact["n_rows_below_old_margin"] == 1            # 4.7 mm < 5 mm


@pytest.mark.parametrize("overlap,feasible", [(0.0095, True), (0.010, True), (0.0105, False), (0.011, False)])
def test_minus_10_mm_boundary(overlap, feasible):
    lin, centres, field, points = _rig()
    c = centres.copy()
    bar = ROWS["bar"]
    c[0, bar] = field.bar + np.array([0.0, 0.0, -(BAR_R + R_FINGER - overlap)])     # sphere surface `overlap` inside
    on = _clear(lin, c, _scene(field, points, margin=M, support_planes=TABLE_PLANE,
                               neighbour_palm_contact=_npc(field.bar)))
    assert on[bar] == pytest.approx(DEPTH - overlap, abs=1e-12)
    assert bool(on[bar] >= -1e-12) is feasible
    assert lin._last_neighbour_palm_contact["min_clearance_mm"] == pytest.approx(-overlap * 1000.0, abs=1e-6)


def test_finger_and_palm_allowances_are_independent():
    lin, centres, field, points = _rig()
    nb = {"links_mask": np.array([False, True, True, True]), "depth": 0.020, "centre": field.bar, "radius": 0.05}
    kw = dict(margin=M, support_planes=TABLE_PLANE)
    nbc_only = _clear(lin, centres, _scene(field, points, **kw, neighbour_contact=nb))
    both = _clear(lin, centres, _scene(field, points, **kw, neighbour_contact=nb,
                                       neighbour_palm_contact=_npc(field.bar, mask=(True, False, False, False))))
    bar = ROWS["bar"]
    assert both[bar] == pytest.approx(0.008 - R_FINGER + DEPTH, abs=1e-12)
    others = [i for i in range(len(both)) if i != bar]
    np.testing.assert_array_equal(both[others], nbc_only[others])


# ------------------------------------------------------------------------------------------------ 2. SafePolicy gate
class _HandNames:
    """The test arm model with palm and finger sphere names appended (the server model has them)."""

    EXTRA = ("ee_left", "ee_left", "ee_finger_l1", "ee_finger_l2", "ee_right", "ee_finger_r1")

    def __init__(self, model):
        self._m = model
        self.sphere_link_names = [*map(str, model.sphere_link_names), *self.EXTRA]

    def __getattr__(self, name):
        return getattr(self._m, name)


def _gate_policy(depth=DEPTH, nbc=0.0, hsr6=None):
    from AG3S.trajopt.safe_policy import SafePolicy
    from tests.trajopt.test_safe_policy import _robot_model

    p = object.__new__(SafePolicy)
    p.to_config = TrajOptConfig.from_dict({"collision": {"neighbour_palm_contact_depth": depth,
                                                         "neighbour_contact_depth": nbc}})
    p._latch = types.SimpleNamespace(phase=GraspPhase.LATCHED)
    p._last_grasp_record = {"state": "latched"}
    p._last_neighbour_contact = {}
    p._last_neighbour_palm_contact = {}
    model = _HandNames(_robot_model())
    p.constraint_robot_model = model
    p.grasp_links = {"left": ("ee_finger_l1", ()), "right": ("ee_finger_r1", ())}
    p.hold_links = {"left": "ee_left", "right": "ee_right"}
    p.ag3s = types.SimpleNamespace(manipulated=None)
    p.hsr6 = hsr6
    p._last_hsr6 = {}
    p._pending = {}
    return p, model


def _site(model, q, link="ee_left"):
    from AG3S.trajopt.safe_policy import hold_site_position

    return hold_site_position(model, q, link)


def _target(p, centroid):
    p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=np.asarray(centroid, np.float64))


def test_window_on_selects_the_grasping_palm_only():
    p, model = _gate_policy()
    q = np.zeros(len(model.joint_names))
    assert p._neighbour_palm_contact_gate(q, ["left"]) is None
    assert p._last_neighbour_palm_contact["reason"] == "no target"
    _target(p, _site(model, q) + np.array([0.0, 0.0, -0.03]))
    for phase in (GraspPhase.LATCHED, GraspPhase.CLOSING):
        p._latch.phase = phase
        npc = p._neighbour_palm_contact_gate(q, ["left"])
        assert npc is not None and npc["depth"] == DEPTH and npc["radius"] == pytest.approx(0.12)
        names = [str(n) for n in model.sphere_link_names]
        assert {n for n, m in zip(names, npc["links_mask"]) if m} == {"ee_left"}
        assert int(npc["links_mask"].sum()) == 2
        np.testing.assert_allclose(npc["centre"], p.ag3s.manipulated.centroid)
        rec = p._last_neighbour_palm_contact
        assert rec["active"] is True and rec["site_to_target_mm"] == pytest.approx(30.0, abs=0.1)
        assert p._last_grasp_record["neighbour_palm_contact"] is rec                 # chunk record `grasp.*`
    assert p._neighbour_palm_contact_gate(q, ["right"]) is None                      # the right palm is far


@pytest.mark.parametrize("dist,on", [(0.059, True), (0.0599, True), (0.0601, False), (0.10, False)])
def test_window_boundary_is_60_mm_from_the_hold_link_site(dist, on):
    p, model = _gate_policy()
    q = np.zeros(len(model.joint_names))
    _target(p, _site(model, q) + np.array([0.0, dist, 0.0]))
    assert (p._neighbour_palm_contact_gate(q, ["left"]) is not None) is on
    if not on:
        assert p._last_neighbour_palm_contact["reason"] == "palm outside the gate"


def test_window_measures_the_site_not_the_link_origin():
    from AG3S.ag3s.constraints.attached import _link_pose_numeric

    p, model = _gate_policy()
    q = np.zeros(len(model.joint_names))
    origin = np.asarray(_link_pose_numeric(model, q, "ee_left")[:3, 3])
    assert np.linalg.norm(_site(model, q) - origin) == pytest.approx(0.100, abs=1e-9)
    _target(p, origin)                                                               # at the origin: 100 mm off the site
    assert p._neighbour_palm_contact_gate(q, ["left"]) is None


def test_window_off_when_held_searching_placed_revoked_or_flag_off():
    p, model = _gate_policy()
    q = np.zeros(len(model.joint_names))
    _target(p, _site(model, q))
    for phase in (GraspPhase.SEARCHING, GraspPhase.HELD, GraspPhase.PLACED):
        p._latch.phase = phase
        assert p._neighbour_palm_contact_gate(q, ["left"]) is None
        assert p._last_neighbour_palm_contact["reason"] == "phase " + phase.value
    p._latch.phase = GraspPhase.CLOSING
    p._last_grasp_record = {"attach_revoked": True}
    assert p._neighbour_palm_contact_gate(q, ["left"]) is None
    assert p._last_neighbour_palm_contact["reason"] == "revoked"
    off, _ = _gate_policy(depth=0.0)
    _target(off, _site(model, q))
    assert off._neighbour_palm_contact_gate(q, ["left"]) is None and off._last_neighbour_palm_contact == {}
    assert "neighbour_palm_contact" not in off._last_grasp_record


def test_hsr6_objective_does_not_withdraw_a6_but_withdraws_nbc():
    p, model = _gate_policy(nbc=0.020, hsr6=H6.Hsr6Flags(objective=True))
    p._hsr6_start({"p": {c: (0.9 if c == "approach" else 0.02) for c in H6.PHASES6}, "argmax": "approach"})
    q = np.zeros(len(model.joint_names))
    _target(p, _site(model, q) + np.array([0.0, 0.0, -0.03]))
    assert p._neighbour_contact_gate(q, ["left"]) is None                              # ④: NBC withheld
    assert p._last_neighbour_contact.get("hsr6_withdrawn") is True
    npc = p._neighbour_palm_contact_gate(q, ["left"])                                  # A6: its own window
    assert npc is not None and "hsr6_withdrawn" not in p._last_neighbour_palm_contact


def test_solve_rows_are_merged_into_the_record():
    p, model = _gate_policy()
    q = np.zeros(len(model.joint_names))
    _target(p, _site(model, q))
    rows = {"n_rows": 3, "min_clearance_mm": -2.5}
    p.refiner = types.SimpleNamespace(last_result=types.SimpleNamespace(metrics={"neighbour_palm_contact": rows}))
    assert p._neighbour_palm_contact_gate(q, ["left"]) is not None
    p._neighbour_palm_contact_result()
    assert p._last_grasp_record["neighbour_palm_contact"]["rows"] == rows
    p.refiner = types.SimpleNamespace(last_result=types.SimpleNamespace(metrics={}))   # no relaxed scene in the solve
    p._neighbour_palm_contact_gate(q, ["left"])
    p._neighbour_palm_contact_result()
    assert p._last_neighbour_palm_contact["rows"] is None


# ------------------------------------------------------------------------------------------------ 3. real model
@pytest.fixture(scope="module")
def server_model():
    mujoco = pytest.importorskip("mujoco")
    from common.workspace import workspace_path

    xml = pathlib.Path(workspace_path("robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
    if not xml.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {xml}")
    from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    from AG3S.trajopt.serve_safe import constraint_link_filter

    m = mujoco.MjModel.from_xml_path(str(xml))
    scene = TransportScene.attach(m, mujoco.MjData(m))
    model = build_constraint_robot_model(
        scene, link_filter=constraint_link_filter("gripper"),
        sphere_options=dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05),
        finger_cover={"hands": "both"}, palm_shell=True)
    return mujoco, m, model


def test_rows_on_the_server_model_are_hand_body_and_palm_shell(server_model):
    from AG3S.ag3s.robot_models.urdf_sphere_chain import PALM_SHELL_ROLE
    from AG3S.trajopt.safe_policy import palm_contact_rows

    _, _, model = server_model
    names = [str(n) for n in model.sphere_link_names]
    roles = [str(r) for r in model.sphere_roles]
    for side, f in (("left", "l"), ("right", "r")):
        mask = palm_contact_rows(model, f"ee_{side}")
        sel_roles = {r for r, m in zip(roles, mask) if m}
        assert PALM_SHELL_ROLE in sel_roles and len(sel_roles) >= 2                    # shell + hand body
        assert sum(1 for r, m in zip(roles, mask) if m and r == PALM_SHELL_ROLE) == 168
        assert not any(m and n.startswith("ee_finger") for n, m in zip(names, mask))
        shell_side = [n for n, r in zip(names, roles) if r == PALM_SHELL_ROLE and n == f"ee_{side}"]
        assert len(shell_side) == 168


def test_hold_link_site_is_the_mjcf_ee_site(server_model):
    from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
    from AG3S.trajopt.safe_policy import hold_site_position

    mujoco, m, _ = server_model
    rm = UrdfSphereChain(parse_urdf(RBY1_URDF),
                         link_filter=[f"link_{s}_arm_{i}" for s in ("left", "right") for i in range(6)])
    d = mujoco.MjData(m)
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    for seed in range(3):
        q = np.random.default_rng(seed).uniform(-0.8, 0.8, len(rm.joint_names))
        for i, n in enumerate(rm.joint_names):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            assert jid >= 0, n
            d.qpos[m.jnt_qposadr[jid]] = q[i]
        mujoco.mj_kinematics(m, d)
        R, o = d.xmat[base].reshape(3, 3), d.xpos[base]
        for side in ("left", "right"):
            s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee")
            np.testing.assert_allclose(hold_site_position(rm, q, f"ee_{side}"), (d.site_xpos[s] - o) @ R, atol=1e-6)


# ------------------------------------------------------------------------------------------------ 4. ② fix
@pytest.mark.parametrize("phase,held,blocked,want", [
    ("grasp", True, False, True), ("grasp", True, True, True),        # the fix: kept right after the grasp
    ("grasp", False, True, True),                                     # blocked fingers count as before (rule kept)
    ("grasp", False, False, False),
    ("approach", True, True, False), ("approach", False, True, False),
    ("retreat", True, True, False), ("retreat", True, False, False),
    ("lift", True, False, True), ("transport", False, True, True), ("placement", False, False, False),
    (None, True, True, False), ("pick", True, True, False)])
def test_held_allowance_fixed_rule(phase, held, blocked, want):
    assert H6.held_allowance(phase, latch_held=held, blocked=blocked) is want
    assert H6.HELD_WITHDRAW_PHASES == frozenset({"approach", "retreat"})


def test_safe_policy_keeps_the_allowance_for_grasp_held():
    from tests.trajopt.test_grasp_evidence import _policy

    sp = _policy(hsr6=H6.Hsr6Flags(held_gate=True))
    got = []
    sp.ag3s.set_held_allowance = got.append
    for phase, latch_phase, want in (("grasp", GraspPhase.HELD, True), ("approach", GraspPhase.HELD, False),
                                     ("retreat", GraspPhase.HELD, False), ("lift", GraspPhase.HELD, True)):
        sp._pending = {}
        sp._hsr6_start({"p": {c: (0.9 if c == phase else 0.02) for c in H6.PHASES6}, "argmax": phase})
        sp._latch.phase = latch_phase
        sp._last_grasp_record = {"evidence": {"checks": {}}}
        sp._hsr6_held_gate()
        assert got[-1] is want
    off = _policy()
    assert off.hsr6 is None                                                          # flag off: never asked


# ------------------------------------------------------------------------------------------------ 5. config / CLI
#: The final server argv (A8 §A8 + G1 + four HSR6 flags + A6), `<…>` placeholders replaced.
FINAL = (
    "--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper "
    "--target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 "
    "--sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry "
    "--latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 "
    "--finger-cover both --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 "
    "--w-deflection-rate 20 --to-fast-eval --esdf-host-mirror --target-exclusion ball-owned --owned-max-dist-voxels 3 "
    "--attention-split-radius-mm 12 --attention-split-ratio-source raw --target-permission-strict "
    "--attention-source noun_target:L10H7 --attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1 "
    "--neighbour-contact-mm 20 --nontarget-evidence surface --placed-no-seed-certify --kept-geometry-follow --roi-carve "
    "--roi-carve-footprint --roi-carve-footprint-pad-px 0 --roi-carve-footprint-robot ignore --roi-carve-device cuda "
    "--blas-threads 1 --to-multi-seed 16 --to-gpu-seeds curobo --to-gpu-seeds-check first-pass "
    "--palm-shell on --palm-shell-margin 5 --grasp-anchor 2000 --grasp-anchor-path-weight 0 "
    "--geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off "
    "--hsr6-attach-assist --hsr6-held-gate --hsr6-target-exempt --hsr6-objective --neighbour-palm-contact-mm 10")


def test_config_default_off_keys_absent_and_validated():
    c = TrajOptConfig()
    assert c.collision.neighbour_palm_contact_depth == 0.0
    assert c.collision.neighbour_palm_contact_gate == pytest.approx(0.06)
    assert c.collision.neighbour_palm_contact_radius == pytest.approx(0.12)
    d = c.to_dict()["collision"]
    assert not any(k.startswith("neighbour_palm") for k in d)
    on = TrajOptConfig.from_dict({"collision": {"neighbour_palm_contact_depth": 0.01}})
    assert on.to_dict()["collision"]["neighbour_palm_contact_depth"] == 0.01
    assert TrajOptConfig.from_dict(on.to_dict()).collision.neighbour_palm_contact_depth == 0.01
    for bad in ({"neighbour_palm_contact_depth": -0.01}, {"neighbour_palm_contact_radius": 0.0},
                {"neighbour_palm_contact_gate": 0.0}, {"neighbour_palm_contact_depth": float("nan")}):
        with pytest.raises(TrajOptConfigError):
            TrajOptConfig.from_dict({"collision": bad})


def test_cli_off_and_on():
    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    off = serve_safe.trajopt_config_from_args(p.parse_args(BASE_ARGV))
    assert off.collision.neighbour_palm_contact_depth == 0.0
    assert "neighbour_palm_contact_depth" not in serve_safe.obstacle_overrides(p.parse_args(BASE_ARGV))
    on = serve_safe.trajopt_config_from_args(p.parse_args([*BASE_ARGV, "--neighbour-palm-contact-mm", "10"]))
    assert on.collision.neighbour_palm_contact_depth == pytest.approx(0.010)
    with pytest.raises(SystemExit):
        serve_safe.trajopt_config_from_args(p.parse_args([*BASE_ARGV, "--neighbour-palm-contact-mm", "0"]))


def test_final_argv_parses_to_the_fixed_values(monkeypatch):
    from AG3S.trajopt import gpu_seeds, serve_safe

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, "patched for a CPU test"))
    p = serve_safe.build_parser()
    a = p.parse_args(BASE_ARGV + shlex.split(FINAL))
    serve_safe.reject_bad_flag_combinations(p, a)
    c = serve_safe.trajopt_config_from_args(a).collision
    assert c.neighbour_palm_contact_depth == pytest.approx(0.010)
    assert c.neighbour_palm_contact_gate == pytest.approx(0.060)
    assert c.neighbour_palm_contact_radius == pytest.approx(0.120)
    assert c.neighbour_contact_depth == pytest.approx(0.020) and c.palm_shell_margin == pytest.approx(0.005)
    assert serve_safe.hsr6_flags_from_args(a).names() == ["attach_assist", "held_gate", "target_exempt", "objective"]


def test_announce_line_only_when_on():
    import inspect

    from AG3S.trajopt import serve_safe

    assert "AG3S-CORE-EVAL-A6" in inspect.getsource(serve_safe)                        # the server start line exists


# ------------------------------------------------------------------------------------------------ 6. recorded chunk
CB = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261028T224856+0900_b3b69311f948_8784a1265958/"
                  "verify/AG3S-CORE-EVAL")


def _spec119():
    path = CB / "module" / "spec_cb_HSR.json"
    if not path.exists():
        return None
    sp = [x for x in json.load(open(path)) if x["label"] == "회귀119_k0"]
    return sp[0] if sp and pathlib.Path(sp[0]["chunks"][0]).exists() else None


needs_119 = pytest.mark.skipif(_spec119() is None, reason="scene 119 combination-test record not present")
_RUN: dict = {}


def _run119():
    from AG3S.trajopt.experiments import ta_offline as TA

    if "run" not in _RUN:
        sp = _spec119()
        _RUN["run"] = TA.Run("119", {"run_dir": sp["client_dir"],
                                     "server_segment": {"first": sp["chunks"][0], "n": 16}},
                             types.SimpleNamespace(hm=False, no_plane_rows=False))
    return _RUN["run"]


@needs_119
def test_recorded_chunk_final_check_sees_the_relaxed_rows():
    pytest.importorskip("casadi")
    pytest.importorskip("osqp")
    from AG3S.trajopt.refiner import TrajOptChunkRefiner
    from AG3S.trajopt.safe_policy import hold_site_position, palm_contact_rows

    run = _run119()
    seq = sorted(run.chunks)[13]                                   # request 13: palm–apple ~30 mm, orange binding
    inp, why = run.inputs(seq)
    assert inp is not None, why
    cfg = run.base_cfg.with_overrides({"sqp": {"gpu_seeds": "off", "gpu_seeds_check": "all", "multi_seed": 0},
                                       "collision": {"neighbour_palm_contact_depth": DEPTH}})
    centre = np.asarray(inp["target_centroid"], np.float64)
    assert np.linalg.norm(hold_site_position(run.robot, inp["q_now"], "ee_left") - centre) < 0.06
    npc = {"links_mask": palm_contact_rows(run.robot, "ee_left"), "depth": DEPTH, "centre": centre, "radius": 0.12}
    out = {}
    for name, scene in (("off", inp["scene"]), ("on", dataclasses.replace(inp["scene"], neighbour_palm_contact=npc))):
        holder = {"v": (scene, inp["q_now"], True)}
        ref = TrajOptChunkRefiner(run.robot, run.layout, lambda _c: holder["v"], cfg)
        lin = ref.optimizer.linearizer
        lin.set_joint_parameter_path(run.finger_path(inp["summary"], inp["ref_chunk"], cfg.horizon.planned))
        ref.refine(inp["ref_chunk"], {"t_step": seq, "previous_physical_chunk": None})
        res = ref.last_result
        out[name] = res
        if name == "off":
            assert "neighbour_palm_contact" not in res.metrics
            continue
        rows = res.metrics["neighbour_palm_contact"]
        assert rows["n_rows"] > 0 and rows["depth_mm"] == pytest.approx(10.0)
        # the record is the returned trajectory's final check: re-measured on the same scene it is the same. A closed
        # look-ahead gate solves (and certifies) the executed window only, with the window optimizer.
        opt = ref.optimizer
        if (res.metrics.get("lookahead") or {}).get("gate") == "closed":
            k = int(res.metrics["lookahead"]["from"])
            opt, traj, steps = opt.window_optimizer, res.trajectory[:, :k], None
        else:
            traj, steps = res.trajectory, opt.lookahead_from
        clearance, _ = opt.linearizer.worst_row(traj, inp["q_now"], scene, steps=steps)
        assert opt.linearizer._last_neighbour_palm_contact == rows
        assert res.metrics["clearance_m"] == pytest.approx(clearance, abs=1e-12)
        assert res.max_violation == pytest.approx(max(0.0, -clearance), abs=1e-12)
