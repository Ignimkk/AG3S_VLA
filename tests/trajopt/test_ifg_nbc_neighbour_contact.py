"""IFG-NBC — light contact of the grasping fingers with a neighbour object (`collision.neighbour_contact_depth`,
`serve_safe --neighbour-contact-mm`), default off.

IFG ep211 / ep954 (2026-10-21): VLA alone touches the banana next to the apple while it grasps and succeeds; AG3S keeps
the obstacle margin and either pushes the hand (954) or holds (211). Option B (only a wrist roll at the VLA-alone grasp
point) cannot meet the margin in either scene (`ifg_roll_feasibility`); user ruling: then A.

What this file pins:
1. Linearizer: off = bit-identical; on, only obstacle rows of the masked links within the radius of the centre get the
   margin `-depth`; target, support and held rows, other links and far rows keep theirs.
2. `SafePolicy._neighbour_contact_gate`: off by default; on only latched / closing with the palm within the gate; the
   mask is the grasping hand's finger spheres.
3. Config / CLI.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from tests.trajopt.test_t41a_obstacle_margin import M, R_FINGER, ROWS, TABLE_PLANE, _clear, _rig, _scene

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
DEPTH = 0.020


# ------------------------------------------------------------------------------------------------ 1. linearizer
def test_only_masked_obstacle_rows_near_the_centre_are_relaxed():
    lin, centres, field, points = _rig()
    base = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE))
    nb = {"links_mask": np.ones(4, bool), "depth": DEPTH, "centre": field.bar, "radius": 0.05}
    on = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE, neighbour_contact=nb))
    bar = ROWS["bar"]
    assert on[bar] == pytest.approx(0.008 - R_FINGER + DEPTH, abs=1e-12)            # was 0.008 - r - M
    assert base[bar] == pytest.approx(0.008 - R_FINGER - M, abs=1e-12)
    others = [i for i in range(len(on)) if i != bar]
    np.testing.assert_array_equal(on[others], base[others])                         # apple / table / box / held
    assert lin._last_neighbour_contact == {"n_rows": 1}
    # the bar row's link not in the mask -> unchanged
    nb_other = {**nb, "links_mask": np.array([False, True, True, True])}
    np.testing.assert_array_equal(
        _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE, neighbour_contact=nb_other)), base)
    # a support row near the centre is not a neighbour
    table = ROWS["table"]
    nb_table = {**nb, "centre": centres[0, table]}
    t = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE, neighbour_contact=nb_table))
    assert t[table] == base[table]
    # off (None / depth 0) is bit-identical, also without an obstacle margin
    np.testing.assert_array_equal(
        _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE,
                                    neighbour_contact={**nb, "depth": 0.0})), base)
    plain = _clear(lin, centres, _scene(field, points, support_planes=TABLE_PLANE))
    no_margin = _clear(lin, centres, _scene(field, points, support_planes=TABLE_PLANE, neighbour_contact=nb))
    assert no_margin[bar] == pytest.approx(plain[bar] + DEPTH, abs=1e-12)


# ------------------------------------------------------------------------------------------------ 2. SafePolicy gate
class _FingerNames:
    """The test arm model (no finger spheres) with finger sphere names appended (the server model has them)."""

    FINGERS = ("ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")

    def __init__(self, model, fingers=FINGERS):
        self._m = model
        self.sphere_link_names = [*map(str, model.sphere_link_names), *fingers]

    def __getattr__(self, name):
        return getattr(self._m, name)


def _gate_policy(depth=DEPTH):
    from AG3S.trajopt.grasp_latch import GraspPhase
    from AG3S.trajopt.safe_policy import SafePolicy
    from tests.trajopt.test_safe_policy import _robot_model

    p = object.__new__(SafePolicy)
    p.to_config = TrajOptConfig.from_dict({"collision": {"neighbour_contact_depth": depth}})
    p._latch = types.SimpleNamespace(phase=GraspPhase.LATCHED)
    p._last_grasp_record = {}
    p._last_neighbour_contact = {}
    model = _FingerNames(_robot_model())
    p.constraint_robot_model = model
    p.grasp_links = {"left": ("ee_finger_l1", ()), "right": ("ee_finger_r1", ())}
    p.hold_links = {"left": "ee_left", "right": "ee_right"}
    p.ag3s = types.SimpleNamespace(manipulated=None)
    return p, model


def test_gate_selects_the_grasping_fingers_near_the_target_only():
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from AG3S.trajopt.grasp_latch import GraspPhase

    p, model = _gate_policy()
    q = np.zeros(len(model.joint_names))
    palm = np.asarray(_link_pose_numeric(model, q, "ee_left")[:3, 3])
    assert p._neighbour_contact_gate(q, ["left"]) is None and p._last_neighbour_contact["reason"] == "no target"
    p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=palm + np.array([0.0, 0.0, -0.05]))
    nb = p._neighbour_contact_gate(q, ["left"])
    names = [str(n) for n in model.sphere_link_names]
    assert nb is not None and nb["depth"] == DEPTH and nb["radius"] == pytest.approx(0.12)
    sel = {n for n, m in zip(names, nb["links_mask"]) if m}
    assert sel and all(n.startswith("ee_finger_l") for n in sel)
    assert not any(n.startswith("ee_finger_r") for n in sel)
    np.testing.assert_allclose(nb["centre"], p.ag3s.manipulated.centroid)
    assert p._neighbour_contact_gate(q, ["right"]) is None                         # right palm far from the target
    p.ag3s.manipulated.centroid = palm + np.array([0.0, 0.0, -0.30])
    assert p._neighbour_contact_gate(q, ["left"]) is None
    p.ag3s.manipulated.centroid = palm
    for phase in (GraspPhase.SEARCHING, GraspPhase.HELD, GraspPhase.PLACED):
        p._latch.phase = phase
        assert p._neighbour_contact_gate(q, ["left"]) is None
    p._latch.phase = GraspPhase.CLOSING
    assert p._neighbour_contact_gate(q, ["left"]) is not None
    bare, _ = _gate_policy()
    bare.constraint_robot_model = _FingerNames(bare.constraint_robot_model._m, fingers=())
    bare.ag3s.manipulated = p.ag3s.manipulated
    assert bare._neighbour_contact_gate(q, ["left"]) is None                        # no finger spheres: fail closed
    assert bare._last_neighbour_contact["reason"] == "no sphere of a link ee_finger_l*"
    off, _ = _gate_policy(depth=0.0)
    off.ag3s.manipulated = p.ag3s.manipulated
    assert off._neighbour_contact_gate(q, ["left"]) is None and off._last_neighbour_contact == {}


# ------------------------------------------------------------------------------------------------ 3. config / CLI
def test_config_and_cli():
    from AG3S.trajopt import serve_safe

    assert TrajOptConfig().collision.neighbour_contact_depth == 0.0
    for bad in ({"neighbour_contact_depth": -0.01}, {"neighbour_contact_radius": 0.0}, {"neighbour_contact_gate": 0.0}):
        with pytest.raises(TrajOptConfigError):
            TrajOptConfig.from_dict({"collision": bad})
    args = serve_safe.build_parser().parse_args([*BASE_ARGV, "--neighbour-contact-mm", "20"])
    assert serve_safe.trajopt_config_from_args(args).collision.neighbour_contact_depth == pytest.approx(0.020)
    assert serve_safe.trajopt_config_from_args(serve_safe.build_parser().parse_args(BASE_ARGV)).collision.neighbour_contact_depth == 0.0
    with pytest.raises(SystemExit):
        serve_safe.trajopt_config_from_args(serve_safe.build_parser().parse_args([*BASE_ARGV, "--neighbour-contact-mm", "0"]))
