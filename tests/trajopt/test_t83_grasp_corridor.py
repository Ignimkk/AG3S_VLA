"""T83 — grasp corridor carve (`collision.grasp_corridor_allow`, `serve_safe --grasp-corridor-allow`, default off).

Task: `AG3S/ag3s/docs/handoff/T83.task.md` §1 (corridor, k) · §1b (carve the fine target-free tier, fixed before
results) · lead message 2026-11-01 (tests).

What this file pins:
1. Corridor geometry (`AG3S/ag3s/grasp/corridor.py`): the box (`lo + 3 mm <= x <= hi - 3 mm`, `|y| <= pad + 5 mm`,
   `z_tip - 60 mm <= z <= z_palm`), the union over the poses (q_now + planned rows), `corridor_voxels` = brute force,
   closed fingers = empty. On the server model the pose is the URDF hold-link pose and the pads' faces follow the
   opening (sides by position).
2. Builder carve (`CuroboFieldBuilder._corridor_carve`): seeds inside the union are cleared; seeds outside the fingers
   stay; seeds within 3 mm of the support plane or below stay; the sign-correction set is the corridor's non-support
   voxels; `_build_tiers` without a corridor is the call as before.
3. AG3S (`AG3S._grasp_corridor_carve`): on with >= k of this frame's target points in the q_now corridor; off below k,
   with no target-free tier (held / no target), with no corridor given; the request is consumed by one frame.
4. SafePolicy (`_grasp_corridor_request`): off → AG3S is never told, no record, no `grasp.corridor` key; held / placed
   → AG3S is told None; pre-grasp → q_now + the planned rows (finger joints per row from the TrajOpt finger path).
5. Config / CLI: default off, keys absent from `to_dict`, validation; the T83 base argv (T78 S3 server argv without
   `--surface-split`) with and without the flag parses.
"""

from __future__ import annotations

import inspect
import pathlib
import shlex
import sys
import types

import numpy as np
import pytest

from AG3S.ag3s.grasp.corridor import (
    CorridorCarve,
    GraspCorridorRequest,
    corridor_voxels,
    near_or_below_support,
    support_planes,
)
from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.grasp_latch import GraspPhase

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
PAD_HALF_Y, Z_TIP, Z_PALM = 0.0163, -0.1338, -0.0737


def _yaw(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


R0 = _yaw(30.0)
T0 = np.array([0.50, 0.10, 0.40])          # hold-link origin; the hand points down (-z local = -z base)


def _req(poses=((R0, T0, -0.030, 0.030),), **kw):
    R, t, lo, hi = (np.asarray(v) for v in zip(*poses))
    return GraspCorridorRequest(R=R, t=t, lo=lo, hi=hi, pad_half_y=PAD_HALF_Y, z_tip=Z_TIP, z_palm=Z_PALM, **kw)


def _world(local, R=R0, t=T0):
    return np.asarray(local, np.float64) @ R.T + t


# ------------------------------------------------------------------------------------------------ 1. geometry
@pytest.mark.parametrize("pt,inside", [
    ((0.0, 0.0, -0.10), True), ((0.0269, 0.0, -0.10), True), ((0.0271, 0.0, -0.10), False),     # x: 30 - 3 mm
    ((-0.0271, 0.0, -0.10), False), ((0.0, 0.0213, -0.10), True), ((0.0, 0.0214, -0.10), False),  # y: 16.3 + 5
    ((0.0, 0.0, -0.0736), False), ((0.0, 0.0, -0.0738), True),                                      # palm face
    ((0.0, 0.0, -0.1937), True), ((0.0, 0.0, -0.1939), False),                                      # tip - 60 mm
])
def test_corridor_box(pt, inside):
    assert bool(_req().contains(_world([pt]))[0]) is inside


def test_union_over_poses_and_q_now_only():
    lifted = T0 + np.array([0.0, 0.0, 0.10])
    req = _req(poses=((R0, T0, -0.03, 0.03), (R0, lifted, -0.03, 0.03)))
    p_low = _world([(0.0, 0.0, -0.17)])                           # in pose 0 only
    p_high = _world([(0.0, 0.0, -0.10)], t=lifted)                # in pose 1 only (90 mm above pose 0's palm face)
    assert req.contains(p_low)[0] and req.contains(p_high)[0]
    assert req.contains(p_low, poses=[0])[0] and not req.contains(p_high, poses=[0])[0]
    assert req.n_poses == 2 and req.summary()["opening_mm"] == [60.0, 60.0]


def test_closed_fingers_give_an_empty_corridor():
    req = _req(poses=((R0, T0, 0.002, -0.002),))
    assert not req.contains(_world([(0.0, 0.0, -0.10)])).any()
    assert corridor_voxels(req, T0 - 0.2, 0.005, (80, 80, 80)).shape == (0, 3)


def test_corridor_voxels_equal_brute_force():
    lifted = T0 + np.array([0.02, -0.01, 0.03])
    req = _req(poses=((R0, T0, -0.03, 0.03), (_yaw(50.0), lifted, -0.025, 0.035)))
    origin, vs, shape = T0 - np.array([0.12, 0.12, 0.25]), 0.005, (48, 48, 60)
    got = corridor_voxels(req, origin, vs, shape)
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    want = ijk[req.contains(origin + ijk * vs)]
    assert len(got) == len(want) > 500
    assert set(map(tuple, got)) == set(map(tuple, want))
    # clipped to the grid
    small = corridor_voxels(req, origin, vs, (10, 10, 10))
    assert small.size == 0 or (small.max(axis=0) < 10).all()


def test_support_band():
    planes = np.array([[0.0, 0.0, 1.0, 0.20]])
    p = np.array([[0, 0, 0.2029], [0, 0, 0.2031], [0, 0, 0.19], [0, 0, 0.25]])
    assert near_or_below_support(p, planes, 0.003).tolist() == [True, False, True, False]
    assert not near_or_below_support(p, np.zeros((0, 4)), 0.003).any()
    surf = types.SimpleNamespace(normal=np.array([0.0, 0.0, 1.0]), offset=0.2)
    np.testing.assert_array_equal(support_planes([surf]), planes)


# ------------------------------------------------------------------------------------------------ server model
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
    return build_constraint_robot_model(
        scene, link_filter=constraint_link_filter("gripper"),
        sphere_options=dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05),
        finger_cover={"hands": "both"}, palm_shell=True)


@pytest.mark.parametrize("hand", ["left", "right"])
def test_server_model_pose_is_the_hold_link_and_faces_follow_the_opening(server_model, hand):
    from AG3S.ag3s.grasp.corridor import corridor_pose, hand_corridor_spec
    from AG3S.ag3s.robot_models.gripper_state import FingerJointMap

    s = hand[0]
    spec, why = hand_corridor_spec(server_model, f"ee_{hand}", f"ee_finger_{s}")
    assert spec is not None, why
    assert spec["fingers"] == [f"ee_finger_{s}1", f"ee_finger_{s}2"]
    assert [len(c) for c in spec["local"]] == [56, 56]
    fm = FingerJointMap(server_model)
    rng = np.random.default_rng(1)
    widths = []
    for op in (0.2, 0.6, 1.0):
        vals = fm.joint_values({"left": op, "right": op})
        P = np.array([vals[n] for n in server_model.param_joint_names])
        q = rng.uniform(-0.6, 0.6, server_model.nq)
        R, t, lo, hi = corridor_pose(server_model, q, P, spec)
        T = server_model.link_pose(q, f"ee_{hand}", P)
        np.testing.assert_allclose(R, T[:3, :3], atol=1e-12)
        np.testing.assert_allclose(t, T[:3, 3], atol=1e-12)
        # faces = the finger spheres' inner extents in the hold frame (brute force over every finger sphere)
        cen, rad = server_model.sphere_centers_numeric(q, P)
        names = server_model.sphere_link_names
        for f, sign in ((spec["fingers"][0], None), (spec["fingers"][1], None)):
            idx = [i for i, n in enumerate(names) if n == f]
            x = ((cen[idx] - t) @ R)[:, 0]
            if x.mean() < 0:
                assert lo == pytest.approx(float((x + rad[idx]).max()), abs=1e-12)
            else:
                assert hi == pytest.approx(float((x - rad[idx]).min()), abs=1e-12)
        assert lo == pytest.approx(-hi, abs=1e-6)
        widths.append(hi - lo)
    assert widths[0] < widths[1] < widths[2]
    assert widths[2] == pytest.approx(0.0844, abs=0.001)          # sphere-model inner gap, open


# ------------------------------------------------------------------------------------------------ 2. builder carve
VS = 0.005


class _Site:
    def __init__(self, a):
        self.array = a
        self.device = None

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, k):
        return self.array[k]

    def __setitem__(self, k, v):
        self.array[k] = v


@pytest.fixture
def _torch_stub(monkeypatch):
    if "torch" in sys.modules:  # pragma: no cover - the real one where installed
        return
    stub = types.SimpleNamespace(as_tensor=lambda x, device=None, dtype=None: np.asarray(x, np.int64), long=np.int64)
    monkeypatch.setitem(sys.modules, "torch", stub)


def _grid_scene():
    """A 5 mm grid around the hand: a remnant blob between the pads, a neighbour outside finger +x, a table plane
    under the tips. Seeds = surface voxels of each."""
    origin = T0 - np.array([0.10, 0.10, 0.22])
    shape = (41, 41, 41)
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
    c = origin + ijk * VS
    loc = (c - T0) @ R0                                           # hold frame
    table_z = T0[2] - 0.16                                        # 26 mm under the tips (inside the ahead region)
    remnant = np.linalg.norm(loc - [0.0, 0.0, -0.13], axis=-1) < 0.012
    neighbour = np.linalg.norm(loc - [0.06, 0.0, -0.12], axis=-1) < 0.012
    table = np.abs(c[..., 2] - table_z) <= VS * 0.6
    under = (c[..., 2] < table_z) & (c[..., 2] > table_z - 0.015)
    site = np.where(remnant | neighbour | table | under, 0, -1).astype(np.int64)
    planes = np.array([[0.0, 0.0, 1.0, table_z]])
    return origin, shape, site, c, loc, dict(remnant=remnant, neighbour=neighbour, table=table, under=under), planes


def test_builder_carve_removes_corridor_seeds_only(_torch_stub):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder

    origin, shape, site, c, loc, parts, planes = _grid_scene()
    before = site.copy()
    itg = types.SimpleNamespace(_site_index=_Site(site))
    b = CuroboFieldBuilder(AG3SConfig().esdf)
    rec, sign = b._corridor_carve(itg, CorridorCarve(request=_req(), planes=planes), origin, VS, shape)
    cleared = (before >= 0) & (site < 0)
    inside = _req().contains(c.reshape(-1, 3)).reshape(shape)
    assert parts["remnant"][inside].sum() > 0 and cleared[parts["remnant"] & inside].all()
    assert not cleared[parts["neighbour"]].any()                  # outside the fingers: stays
    assert not cleared[parts["table"] | parts["under"]].any()     # support (within 3 mm / below): stays
    assert parts["table"][inside].any()                           # ... although the corridor reaches the table
    assert np.array_equal(cleared, (before >= 0) & inside & ~near_or_below_support(
        c.reshape(-1, 3), planes, 0.003).reshape(shape))
    assert rec["applied"] and rec["n_seeds_carved"] == int(cleared.sum()) and rec["n_support_seeds_kept"] > 0
    assert rec["n_corridor_voxels"] == int(inside.sum()) and rec["ms"] >= 0.0
    assert len(rec["carved_ijk"]) == rec["n_seeds_carved"] and not rec["carved_ijk_truncated"]
    sup = near_or_below_support(origin + sign * VS, planes, 0.003)
    assert not sup.any() and len(sign) == int((inside & ~near_or_below_support(
        c.reshape(-1, 3), planes, 0.003).reshape(shape)).sum())


def test_builder_carve_seed_above_the_band_in_the_corridor_is_carved(_torch_stub):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder

    origin, shape, site, c, loc, parts, planes = _grid_scene()
    site[:] = -1
    inside = _req().contains(c.reshape(-1, 3)).reshape(shape)
    h = c[..., 2] - planes[0, 3]
    a = inside & (np.abs(h - 0.005) < 1e-6)                       # 5 mm above the plane: carved
    k = inside & (np.abs(h - 0.0) < 1e-6)                         # on the plane: kept
    assert a.any() and k.any()
    site[a | k] = 0
    itg = types.SimpleNamespace(_site_index=_Site(site))
    b = CuroboFieldBuilder(AG3SConfig().esdf)
    b._corridor_carve(itg, CorridorCarve(request=_req(), planes=planes), origin, VS, shape)
    assert (site[a] < 0).all() and (site[k] >= 0).all()


def test_build_tiers_without_a_corridor_is_the_call_as_before():
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder

    sig = inspect.signature(CuroboFieldBuilder._build_tiers)
    assert sig.parameters["corridor"].default is None
    assert inspect.signature(CuroboFieldBuilder.update).parameters["grasp_corridor"].default is None
    src = inspect.getsource(CuroboFieldBuilder._build_tiers)
    assert "if corridor is not None:" in src
    up = inspect.getsource(CuroboFieldBuilder.update)
    assert '**({} if grasp_corridor is None else {"corridor": grasp_corridor})' in up
    assert "if grasp_corridor is not None:   # T83" in up


# ------------------------------------------------------------------------------------------------ 3. AG3S
def _ag3s_stub(req):
    return types.SimpleNamespace(_grasp_corridor_request=req, _grasp_corridor_record={})


def _pts(n_in, n_out=20):
    rng = np.random.default_rng(0)
    inside = np.c_[rng.uniform(-0.02, 0.02, n_in), rng.uniform(-0.01, 0.01, n_in), rng.uniform(-0.15, -0.09, n_in)]
    outside = np.c_[rng.uniform(0.05, 0.08, n_out), np.zeros(n_out), rng.uniform(-0.15, -0.09, n_out)]
    return _world(np.vstack([inside, outside]))


def test_ag3s_on_with_k_target_points_in_the_q_now_corridor():
    from AG3S.ag3s.runtime.pipeline import AG3S

    cfg = types.SimpleNamespace(backend="curobo")
    surf = types.SimpleNamespace(normal=np.array([0.0, 0.0, 1.0]), offset=0.2)
    lifted = T0 + np.array([0.0, 0.0, 0.3])
    for poses, want in ((((R0, T0, -0.03, 0.03),), True),
                        (((R0, lifted, -0.03, 0.03), (R0, T0, -0.03, 0.03)), False)):   # only pose 0 counts
        st = _ag3s_stub(_req(poses=poses))
        pts = _pts(5)
        out = AG3S._grasp_corridor_carve(st, pts, pts, [surf], cfg)
        rec = st._grasp_corridor_record
        assert (out is not None) is want and st._grasp_corridor_request is None          # consumed
        assert set(rec) >= {"active", "reason", "n_target_in", "n_rows_swept", "n_seeds_carved"}
        assert rec["n_rows_swept"] == len(poses) - 1
        if want:
            assert rec["active"] and rec["reason"] == "on" and rec["n_target_in"] == 5
            np.testing.assert_array_equal(out.planes, [[0.0, 0.0, 1.0, 0.2]])
        else:
            assert rec["n_target_in"] == 0 and not rec["active"]


def test_ag3s_off_below_k_no_target_free_tier_or_no_request():
    from AG3S.ag3s.runtime.pipeline import AG3S

    cfg = types.SimpleNamespace(backend="curobo")
    st = _ag3s_stub(_req())
    pts = _pts(4)
    assert AG3S._grasp_corridor_carve(st, pts, pts, [], cfg) is None
    assert st._grasp_corridor_record["reason"] == "target points in the q_now corridor 4 < 5"
    st = _ag3s_stub(_req())
    assert AG3S._grasp_corridor_carve(st, _pts(30), None, [], cfg) is None            # held: no target-free tier
    assert st._grasp_corridor_record["reason"].startswith("no fine target-free tier")
    st = _ag3s_stub(_req())
    assert AG3S._grasp_corridor_carve(st, _pts(30), _pts(30), [], types.SimpleNamespace(backend="legacy")) is None
    st = _ag3s_stub(None)
    assert AG3S._grasp_corridor_carve(st, _pts(30), _pts(30), [], cfg) is None and st._grasp_corridor_record == {}
    src = inspect.getsource(AG3S._build_esdf)
    assert "if self._grasp_corridor_request is not None else None" in src
    assert '**({} if corridor is None else {"grasp_corridor": corridor})' in src


def test_ag3s_setter_and_reset():
    from AG3S.ag3s.runtime.pipeline import AG3S

    st = types.SimpleNamespace(_grasp_corridor_request=None, _grasp_corridor_record={"old": 1})
    AG3S.set_grasp_corridor(st, _req())
    assert st._grasp_corridor_request is not None and st._grasp_corridor_record == {}
    assert "self._grasp_corridor_request = None   # T83" in inspect.getsource(AG3S.reset)


# ------------------------------------------------------------------------------------------------ 4. SafePolicy
class _Ag3sSpy:
    def __init__(self):
        self.calls = []
        self.grasp_corridor_record = {}

    def set_grasp_corridor(self, req):
        self.calls.append(req)


def _policy(server_model, allow=True, phase=GraspPhase.LATCHED, horizon=4):
    from AG3S.ag3s.robot_models.gripper_state import FingerJointMap
    from AG3S.trajopt.safe_policy import SafePolicy
    from AG3S.trajopt.types import ChunkLayout

    p = object.__new__(SafePolicy)
    p.to_config = TrajOptConfig.from_dict({"collision": {"grasp_corridor_allow": allow}})
    p._latch = types.SimpleNamespace(phase=phase)
    p._last_grasp_record = {"state": phase.value}
    p._last_grasp_corridor = {}
    p.constraint_robot_model = server_model
    p.layout = ChunkLayout.rby1(server_model.joint_names)
    p.grasp_links = {"left": ("ee_finger_l1", ()), "right": ("ee_finger_r1", ())}
    p.hold_links = {"left": "ee_left", "right": "ee_right"}
    p.ag3s = _Ag3sSpy()
    fm = FingerJointMap(server_model)
    cols = []
    for op in np.linspace(0.4, 1.0, horizon):
        v = fm.joint_values({"left": op, "right": op})
        cols.append([v[n] for n in server_model.param_joint_names])
    P = np.asarray(cols).T
    p.linearizer = types.SimpleNamespace(horizon=horizon, param_names=tuple(server_model.param_joint_names),
                                         param_path=lambda: P)
    p.refiner = types.SimpleNamespace(optimizer=None)
    p._pending = {"chunk": np.zeros((10, 16))}
    return p, P


def test_safe_policy_off_never_tells_ag3s(server_model):
    p, _ = _policy(server_model, allow=False)
    p._grasp_corridor_request(np.zeros(p.layout.nq_model), ["left"])
    assert p.ag3s.calls == [] and p._last_grasp_corridor == {}
    if p._last_grasp_corridor:
        p._grasp_corridor_finish()
    assert "corridor" not in p._last_grasp_record


@pytest.mark.parametrize("phase", [GraspPhase.HELD, GraspPhase.PLACED])
def test_safe_policy_held_placed_tell_none(server_model, phase):
    p, _ = _policy(server_model, phase=phase)
    p._grasp_corridor_request(np.zeros(p.layout.nq_model), ["left"])
    assert p.ag3s.calls == [None]
    p._grasp_corridor_finish()
    assert p._last_grasp_record["corridor"]["reason"] == "phase " + phase.value
    assert p._last_grasp_record["corridor"]["active"] is False


def test_safe_policy_pre_grasp_request_has_q_now_and_the_planned_rows(server_model):
    from AG3S.ag3s.grasp.corridor import corridor_pose, hand_corridor_spec

    for phase in (GraspPhase.SEARCHING, GraspPhase.LATCHED, GraspPhase.CLOSING):
        p, P = _policy(server_model, phase=phase)
        q = np.zeros(p.layout.nq_model)
        p._grasp_corridor_request(q, ["left"])
        (req,) = p.ag3s.calls
        assert isinstance(req, GraspCorridorRequest) and req.n_poses == 1 + 4
        assert (req.ahead, req.margin_in, req.margin_y, req.support_band, req.min_target_points) == \
            (0.060, 0.003, 0.005, 0.003, 5)
        assert req.info["hand"] == "left" and req.info["hold_link"] == "ee_left"
        spec, _ = hand_corridor_spec(server_model, "ee_left", "ee_finger_l")
        R, t, lo, hi = corridor_pose(server_model, q, P[:, 2], spec)
        np.testing.assert_allclose(req.R[3], R)
        assert req.lo[3] == pytest.approx(lo) and req.hi[3] == pytest.approx(hi)
        assert req.hi[4] - req.lo[4] > req.hi[1] - req.lo[1]            # the opening path is per row
    # AG3S's record replaces the request record and goes to `grasp.corridor`
    p.ag3s.grasp_corridor_record = {"active": True, "reason": "on", "n_target_in": 9, "n_rows_swept": 4,
                                    "n_seeds_carved": 31}
    p._grasp_corridor_finish()
    rec = p._last_grasp_record["corridor"]
    assert rec["n_seeds_carved"] == 31 and rec["active"] is True and "request_ms" in rec


def test_safe_policy_bad_pose_tells_none_and_records_why(server_model):
    p, _ = _policy(server_model)
    p._grasp_corridor_request(np.zeros(3), ["left"])                    # q of the wrong size
    assert p.ag3s.calls == [None]
    assert p._last_grasp_corridor["reason"].startswith("corridor not built: ")


def test_safe_policy_call_sites():
    from AG3S.trajopt.safe_policy import SafePolicy

    src = inspect.getsource(SafePolicy._scene_fn)
    assert 'if bool(getattr(self.to_config.collision, "grasp_corridor_allow", False)):' in src
    assert src.index("self._grasp_corridor_request(") < src.index("self.ag3s.process_multi_debug(")
    assert "self._grasp_corridor_finish()" in src


# ------------------------------------------------------------------------------------------------ 5. config / CLI
def test_config_default_off_keys_absent_and_validated():
    c = TrajOptConfig().collision
    assert c.grasp_corridor_allow is False and c.grasp_corridor_ahead == pytest.approx(0.060)
    assert c.grasp_corridor_min_target_points == 5
    assert (c.grasp_corridor_margin_in, c.grasp_corridor_margin_y, c.grasp_corridor_support_band) == \
        (0.003, 0.005, 0.003)
    assert not any(k.startswith("grasp_corridor") for k in TrajOptConfig().to_dict()["collision"])
    on = TrajOptConfig.from_dict({"collision": {"grasp_corridor_allow": True}})
    d = on.to_dict()["collision"]
    assert d["grasp_corridor_allow"] is True and "grasp_corridor_ahead" not in d
    assert TrajOptConfig.from_dict(on.to_dict()).collision.grasp_corridor_allow is True
    for bad in ({"grasp_corridor_allow": 1}, {"grasp_corridor_ahead": -0.01}, {"grasp_corridor_ahead": float("nan")},
                {"grasp_corridor_min_target_points": 0}, {"grasp_corridor_min_target_points": 2.5},
                {"grasp_corridor_support_band": -1.0}):
        with pytest.raises((TrajOptConfigError, ValueError, TypeError)):
            TrajOptConfig.from_dict({"collision": bad})


def test_cli_off_on_and_dependent_flags():
    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    off = p.parse_args(BASE_ARGV)
    assert not any(k.startswith("grasp_corridor") for k in serve_safe.obstacle_overrides(off))
    assert serve_safe.trajopt_config_from_args(off).collision.grasp_corridor_allow is False
    on = serve_safe.trajopt_config_from_args(p.parse_args([*BASE_ARGV, "--grasp-corridor-allow"])).collision
    assert on.grasp_corridor_allow is True and on.grasp_corridor_ahead == pytest.approx(0.060)
    assert on.grasp_corridor_min_target_points == 5
    on2 = serve_safe.trajopt_config_from_args(p.parse_args(
        [*BASE_ARGV, "--grasp-corridor-allow", "--grasp-corridor-ahead-mm", "40",
         "--grasp-corridor-min-target-points", "8"])).collision
    assert on2.grasp_corridor_ahead == pytest.approx(0.040) and on2.grasp_corridor_min_target_points == 8
    for argv in (["--grasp-corridor-ahead-mm", "40"], ["--grasp-corridor-min-target-points", "3"],
                 ["--grasp-corridor-allow", "--grasp-corridor-min-target-points", "0"]):
        with pytest.raises(SystemExit):
            serve_safe.trajopt_config_from_args(p.parse_args([*BASE_ARGV, *argv]))


#: T78 S3 server argv (`verify/T78/S3/launch_occ_S3.sh` SAFE) without `--surface-split` = the T83 base.
T83_BASE = (
    "--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper "
    "--target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 "
    "--sphere-spacing 0.4 --max-spheres-per-capsule 32 --obstacle-margin 0.010 --held-body-cover "
    "--closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach "
    "--held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both --plan-horizon 16 --lookahead obstacles "
    "--lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --to-fast-eval --esdf-host-mirror "
    "--target-exclusion ball-owned --owned-max-dist-voxels 3 --attention-split-radius-mm 12 "
    "--attention-split-ratio-source raw --target-permission-strict --attention-source noun_target:L10H7 "
    "--attention-split-partition --split-supersedes-anchor --wrist-track-scale 0.1 --neighbour-contact-mm 20 "
    "--placed-no-seed-certify --kept-geometry-follow --roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0 "
    "--roi-carve-footprint-robot ignore --roi-carve-device cuda --blas-threads 1 --palm-shell on --palm-shell-margin 5 "
    "--geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off --hsr6-attach-assist "
    "--hsr6-held-gate --hsr6-target-exempt --hsr6-objective --neighbour-palm-contact-mm 10 "
    "--deflection-rate-after-grasp --continuity-after-grasp --target-extent attention-surface "
    "--surface-target-check overlap --surface-keep-last --surface-grasp-hold --surface-no-ball "
    "--attention-surface-device cuda")


@pytest.mark.parametrize("extra,on", [("", False), (" --grasp-corridor-allow", True)])
def test_t83_base_argv_parses(monkeypatch, extra, on):
    from AG3S.trajopt import gpu_seeds, serve_safe

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, "patched for a CPU test"))
    p = serve_safe.build_parser()
    a = p.parse_args(BASE_ARGV + shlex.split(T83_BASE + extra))
    serve_safe.reject_bad_flag_combinations(p, a)
    c = serve_safe.trajopt_config_from_args(a).collision
    assert c.grasp_corridor_allow is on
    assert c.neighbour_palm_contact_depth == pytest.approx(0.010) and c.neighbour_contact_depth == pytest.approx(0.020)
    assert not getattr(a, "surface_split", False)
