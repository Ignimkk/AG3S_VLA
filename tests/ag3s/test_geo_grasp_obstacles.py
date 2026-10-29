"""AG3S-GEO-GRASP §3g-1 — wrist / arm occupancy shapes, obstacle memory, and their use in the grasp checks."""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from AG3S.ag3s.grasp.arm_shape import ArmChecker, OccupancyShape, shapes_hits_many, voxelize_hulls
from AG3S.ag3s.grasp.candidates import GeoGraspConfig, close_pose_check, generate_candidates, pose_checks
from AG3S.ag3s.grasp.gripper import RBY1_LEFT, with_wrist
from AG3S.ag3s.grasp.memory import ObstacleMemory

from tests.ag3s.test_geo_grasp import MJCF, sphere_cloud, table


def cube(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


# a block above the palm that sticks out sideways beyond it (y 40 .. 55 mm; the palm box ends at 33.2 mm) — like the
# RB-Y1 force-torque sensor (80 mm across): outside the palm's approach sweep, so the hand boxes never see it
WRIST = voxelize_hulls([cube((-0.02, 0.040, 0.005), (0.02, 0.055, 0.035))], 0.003, "wrist", "ee_left")


def test_occupancy_union_contains_and_dilates():
    a = voxelize_hulls([cube((0, 0, 0), (0.03, 0.03, 0.03)), cube((0.05, 0, 0), (0.06, 0.01, 0.01))], 0.003, "s", "f")
    P = np.array([[0.015, 0.015, 0.015], [0.055, 0.005, 0.005], [0.04, 0.015, 0.015], [0.1, 0.1, 0.1]])
    assert a.contains(P).tolist() == [True, True, False, False]       # union of two hulls, the gap between stays free
    g = a.dilated(0.012)
    assert g.contains(P).tolist() == [True, True, True, False]        # the 10 mm gap is within 12 mm of both
    assert a.dilated(0.0) is a and g.n_occupied > a.n_occupied


def test_shapes_hits_follow_the_pose():
    T = np.eye(4); T[:3, 3] = (0.5, 0.2, 1.0)
    p_in = T[:3, 3] + np.array([0.0, 0.047, 0.02])
    assert shapes_hits_many(np.array([p_in]), T[None], (WRIST,)).tolist() == [1]
    R = np.diag([1.0, -1.0, -1.0]); T2 = T.copy(); T2[:3, :3] = R     # flipped: the block is now below the origin
    assert shapes_hits_many(np.array([p_in]), T2[None], (WRIST,)).tolist() == [0]


def test_wrist_rejects_a_pose_the_hand_boxes_accept():
    cfg = GeoGraspConfig(time_budget_ms=0.0)
    P = sphere_cloud()
    free = generate_candidates(P, table(), RBY1_LEFT, cfg)
    g = free.candidates[0]
    # an obstacle point inside the wrist block of the best pose, outside its hand boxes
    pt = g.T[:3, :3] @ np.array([0.0, 0.047, 0.02]) + g.T[:3, 3]
    assert not any(b.contains(((pt - g.T[:3, 3]) @ g.T[:3, :3])[None], cfg.scene_margin)[0]
                   for b in RBY1_LEFT.body_boxes(g.open_width))
    scene = np.vstack([table(), pt[None]])
    spec = with_wrist(RBY1_LEFT, (WRIST,), cfg.scene_margin)
    off = generate_candidates(P, scene, spec, cfg)                                   # flag off: the spec's wrist is ignored
    on = generate_candidates(P, scene, spec, dataclasses.replace(cfg, wrist_check=True))
    assert any(np.allclose(c.T, g.T) for c in off.candidates)
    assert not any(np.allclose(c.T, g.T) for c in on.candidates)
    assert on.stats["rejected"]["wrist_collision"] > 0 and "wrist_collision" not in off.stats["rejected"]
    ok, checks, why = pose_checks(g.T, g.T_pre, RBY1_LEFT.body_boxes(g.open_width), scene, on.shell,
                                  cfg, wrist=spec.wrist)
    assert not ok and why == "wrist_collision" and checks["wrist_hits"] == 1
    ok, rec = close_pose_check(g.T, spec, dataclasses.replace(cfg, wrist_check=True), on.surface, on.scene, on.shell)
    assert ok is False and rec["reason"] == "wrist_collision"


def test_arm_fn_rejects_ik_solutions_and_is_ignored_when_off():
    cfg = GeoGraspConfig(time_budget_ms=0.0)
    ik = lambda T, seed: (np.zeros(7), {"converged": True})  # noqa: E731
    calls = []

    def blocked(q, pts):
        calls.append(len(pts))
        return {"hits": 3, "links": ["link_left_arm_2"]}

    P, S = sphere_cloud(), table()
    on = generate_candidates(P, S, RBY1_LEFT, dataclasses.replace(cfg, arm_check=True), ik_fn=ik, arm_fn=blocked)
    assert on.candidates == [] and on.stats["n_arm_rejected"] > 0
    assert on.stats["reason"] == "every IK solution puts the arm in an obstacle"
    assert calls and calls[0] == on.stats["n_scene_points_arm"] > on.stats["n_scene_points_near"] - 1
    off = generate_candidates(P, S, RBY1_LEFT, cfg, ik_fn=ik, arm_fn=blocked)
    assert off.candidates and "n_arm_rejected" not in off.stats
    clear = generate_candidates(P, S, RBY1_LEFT, dataclasses.replace(cfg, arm_check=True), ik_fn=ik,
                                arm_fn=lambda q, pts: {"hits": 0, "links": []})
    assert [c.T.tolist() for c in clear.candidates] == [c.T.tolist() for c in off.candidates]


def test_arm_checker_places_link_shapes_by_fk():
    shape = voxelize_hulls([cube((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01))], 0.003, "link_a", "link_a")
    pose = lambda q, link: np.block([[np.eye(3), np.array([[q[0]], [0.0], [0.0]])], [np.zeros((1, 3)), np.ones((1, 1))]])  # noqa: E731
    chk = ArmChecker((shape,), pose, lambda qs: np.asarray(qs, float), margin=0.005)
    pts = np.array([[0.30, 0.0, 0.0]])
    assert chk([0.30], pts) == {"hits": 1, "links": ["link_a"]}
    assert chk([0.312], pts)["hits"] == 1           # 12 mm away: inside the 10 mm half-size + 5 mm margin
    assert chk([0.32], pts) == {"hits": 0, "links": []}   # 20 mm: beyond 15 mm + the grid error (<= 2.6 mm)


def test_obstacle_memory_accumulates_excludes_target_and_resets():
    mem = ObstacleMemory(voxel=0.005)
    a, b = np.array([[0.0, 0.0, 0.0]]), np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    mem.update(a); mem.update(b); mem.update(None)
    assert len(mem.points) == 3 and mem.n_frames == 2
    assert len(mem.without(np.array([[0.101, 0.0, 0.0]]))) == 2
    mem.reset()
    assert len(mem.points) == 0


def test_planner_obstacle_memory_keeps_an_obstacle_seen_once():
    from AG3S.ag3s.grasp.planner import GeoGraspPlanner

    cfg = GeoGraspConfig(time_budget_ms=0.0, refresh_every=1)
    P, S = sphere_cloud(), table()
    wall = np.array([[0.5 + 0.06, y, z] for y in np.arange(0.1, 0.3, 0.005) for z in np.arange(0.82, 1.05, 0.005)])
    res = {}
    for mem in (False, True):
        pl = GeoGraspPlanner(RBY1_LEFT, dataclasses.replace(cfg, obstacle_memory=mem))
        for scene in (np.vstack([S, wall]), S):          # the wall is seen in the first request only
            r = pl.step(phase="latched", target_id=1, target_state="visible", target_points=P,
                        target_centroid=P.mean(axis=0), scene_points=scene, vla_pose=None, ik_seed=None)
        res[mem] = (r, pl.last_set)
    r_off, cs_off = res[False]
    r_on, cs_on = res[True]
    assert "obstacles" not in r_off and r_on["obstacles"]["n_frames"] == 2
    assert cs_on.stats["rejected"]["scene_collision"] > cs_off.stats["rejected"]["scene_collision"]
    for g in cs_on.candidates:
        ok, _, _ = pose_checks(g.T, g.T_pre, RBY1_LEFT.body_boxes(g.open_width), wall, np.zeros((0, 3)), cfg)
        assert ok
    pl.reset()
    assert len(pl.obstacles.points) == 0


def test_config_flags_need_geo_grasp_and_map_from_the_server():
    from AG3S.trajopt import serve_safe
    from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError

    with pytest.raises(TrajOptConfigError):
        TrajOptConfig().with_overrides({"cost": {"geo_arm_check": True}})
    base_args = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml", "--geo-grasp"]
    args = serve_safe.build_parser().parse_args(base_args + ["--geo-wrist-check", "--geo-arm-check",
                                                             "--geo-obstacle-memory"])
    over = serve_safe.cost_overrides(args)
    assert over["geo_wrist_check"] and over["geo_arm_check"] and over["geo_obstacle_memory"]
    TrajOptConfig().with_overrides({"cost": over})                    # valid together with geo_grasp
    base = serve_safe.cost_overrides(serve_safe.build_parser().parse_args(base_args))
    assert not any(k in base for k in ("geo_wrist_check", "geo_arm_check", "geo_obstacle_memory"))


# --- the real RB-Y1 shapes against MuJoCo ------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(MJCF), reason="MJCF not available")
def test_rby1_shapes_contain_the_mujoco_meshes_at_random_poses():
    mujoco = pytest.importorskip("mujoco")
    from AG3S.ag3s.grasp.arm_shape import _body_meshes, measure_arm_from_mjcf
    from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

    shapes = measure_arm_from_mjcf(MJCF, "left")
    m = mujoco.MjModel.from_xml_path(os.path.abspath(MJCF))
    d = mujoco.MjData(m)
    robot = UrdfSphereChain(parse_urdf(RBY1_URDF))
    rng = np.random.default_rng(0)
    for _ in range(3):
        q = rng.uniform(-0.6, 0.6, robot.nq)
        for i, nm in enumerate(robot.joint_names):
            d.qpos[m.jnt_qposadr[m.joint(nm).id]] = q[i]
        mujoco.mj_kinematics(m, d)
        world = lambda body: np.vstack(_body_meshes(m, d, body, "world"))  # noqa: E731
        # wrist parts: rigid in the hold-link frame (URDF ee_left == MJCF EE_BODY_L)
        T = robot.link_pose(q, "ee_left")
        for s in shapes["wrist"]:
            V = (world(s.name) - T[:3, 3]) @ T[:3, :3]
            assert s.dilated(0.003).contains(V).mean() > 0.995, s.name
        # arm links 0..5: each in its own URDF link frame
        for s in shapes["arm"]:
            Tl = robot.link_pose(q, s.frame)
            V = (world(s.name) - Tl[:3, 3]) @ Tl[:3, :3]
            assert s.dilated(0.003).contains(V).mean() > 0.995, s.name


@pytest.mark.skipif(not os.path.exists(MJCF), reason="MJCF not available")
def test_controller_builds_the_checks_only_when_asked():
    pytest.importorskip("mujoco")
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.trajopt.geo_grasp import GeoGraspController
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)

    def ctl(**kw):
        cfg = TrajOptConfig().with_overrides({"cost": {"geo_grasp": True, **kw}})
        return GeoGraspController(cfg.cost, sc.robot, sc.layout, {"left": "ee_left"}, (7, 15), 0.95, cfg.limits,
                                  horizon=8, execution_length=8)

    off = ctl()._planner("left")
    assert off.gripper.wrist == () and off.arm_fn is None and off.obstacles is None
    assert ctl().gcfg.time_budget_ms == 150.0 and ctl(geo_time_budget_ms=0.0).gcfg.time_budget_ms == 0.0
    g = ctl(geo_wrist_check=True, geo_arm_check=True, geo_obstacle_memory=True)
    pl = g._planner("left")
    assert len(pl.gripper.wrist) == 2 and pl.arm_fn is not None and pl.obstacles is not None
    _, idx, _ = g._ik("left")
    g._q_now = np.asarray(sc.q_now, float)
    q_sub = g._q_now[idx]
    elbow = g._pose(g._q_now, "link_left_arm_3")[:3, 3]
    assert pl.arm_fn(q_sub, np.array([elbow + 5.0]))["hits"] == 0
    hit = pl.arm_fn(q_sub, np.array([elbow]))
    assert hit["hits"] >= 1 and "link_left_arm_3" in hit["links"]


def test_generation_time_budget_flag():
    from AG3S.trajopt import serve_safe
    from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
    base = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml", "--geo-grasp"]
    assert "geo_time_budget_ms" not in serve_safe.cost_overrides(serve_safe.build_parser().parse_args(base))
    over = serve_safe.cost_overrides(serve_safe.build_parser().parse_args(base + ["--geo-time-budget-ms", "0"]))
    assert over["geo_time_budget_ms"] == 0.0
    cost = TrajOptConfig().with_overrides({"cost": over}).cost
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig().with_overrides({"cost": {"geo_time_budget_ms": 0.0}})
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig().with_overrides({"cost": {"geo_grasp": True, "geo_time_budget_ms": -1.0}})
    from AG3S.ag3s.grasp.candidates import GeoGraspConfig
    assert GeoGraspConfig().time_budget_ms == 150.0
    cfg_default = TrajOptConfig().with_overrides({"cost": {"geo_grasp": True}}).cost
    assert cfg_default.geo_time_budget_ms is None and cost.geo_time_budget_ms == 0.0


# --- §3j: the trajectory optimizer's distance field as the obstacle source ---------------------------------------
def wall_field(x0=0.56):
    """Signed distance to the half-space x >= x0 (an analytic 'field')."""
    return lambda P: x0 - np.asarray(P, float)[:, 0]


def test_surface_points_of_boxes_and_shapes():
    from AG3S.ag3s.grasp.gripper import Box

    b = Box((0.0, 0.0, 0.0), (0.02, 0.01, 0.03))
    S = b.surface_points(0.005)
    on_face = np.isclose(S, b.lo).any(axis=1) | np.isclose(S, b.hi).any(axis=1)
    assert on_face.all() and len(S) > 50
    shape = voxelize_hulls([cube((0, 0, 0), (0.03, 0.03, 0.03))], 0.003, "s", "f")
    B = shape.surface_points(0.003)
    assert len(B) < shape.n_occupied and shape.contains(B).all()
    centre = np.array([[0.015, 0.015, 0.015]])
    assert np.linalg.norm(B - centre, axis=1).min() > 0.010          # boundary only, not the interior


def test_field_hits_count_boundary_points_near_a_surface():
    from AG3S.ag3s.grasp.arm_shape import field_hits_many

    S = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    T = np.eye(4); T2 = np.eye(4); T2[0, 3] = 0.54
    # at x 0.54 / 0.55 / 0.56 the wall is 20 / 10 / 0 mm away: one point within 5 mm
    assert field_hits_many(wall_field(0.56), np.stack([T, T2]), S, 0.005).tolist() == [0, 1]


def test_generator_field_source_avoids_the_wall_and_is_off_by_default():
    cfg = GeoGraspConfig(time_budget_ms=0.0)
    P = sphere_cloud()
    wall = np.array([[0.5 + 0.06, y, z] for y in np.arange(0.1, 0.3, 0.004) for z in np.arange(0.82, 1.05, 0.004)])
    table_field = lambda X: np.minimum(wall_field(0.56)(X), np.asarray(X, float)[:, 2] - 0.815)  # noqa: E731
    on = generate_candidates(P, table(), RBY1_LEFT, dataclasses.replace(cfg, obstacle_source="field"), field_fn=table_field)
    pts = generate_candidates(P, np.vstack([table(), wall]), RBY1_LEFT, cfg)
    assert on.stats["obstacle_source"] == "field" and on.field_fn is table_field and on.candidates
    obst = np.vstack([wall, table()])
    for g in on.candidates:      # every returned grasp keeps the wall and the table at the margin (true distance to
        q = (obst - g.T[:3, 3]) @ g.T[:3, :3]                  # the boxes), up to the 5 mm boundary sampling
        for b in RBY1_LEFT.body_boxes(g.open_width):
            gap = np.linalg.norm(np.maximum(np.maximum(np.asarray(b.lo) - q, 0.0), q - np.asarray(b.hi)), axis=1)
            assert gap.min() > cfg.scene_margin - 0.0036
    assert abs(on.stats["rejected"]["scene_collision"] - pts.stats["rejected"]["scene_collision"]) \
        < 0.35 * pts.stats["rejected"]["scene_collision"]
    off = generate_candidates(P, table(), RBY1_LEFT, cfg, field_fn=table_field)      # source "points": field ignored
    assert off.stats["obstacle_source"] == "points" and off.field_fn is None
    g = on.candidates[0]
    ok, checks, _ = pose_checks(g.T, g.T_pre, RBY1_LEFT.body_boxes(g.open_width), np.zeros((0, 3)), on.shell, cfg,
                                field_fn=table_field, gripper=RBY1_LEFT, open_width=g.open_width)
    assert ok and checks["obstacle_source"] == "field"
    T = g.T.copy(); T[0, 3] = 0.56                                           # the hand moved into the wall
    ok, _, why = pose_checks(T, T, RBY1_LEFT.body_boxes(g.open_width), np.zeros((0, 3)), on.shell, cfg,
                             field_fn=table_field, gripper=RBY1_LEFT, open_width=g.open_width)
    assert not ok and why == "scene_collision"


def test_arm_checker_reads_a_field():
    shape = voxelize_hulls([cube((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01))], 0.003, "link_a", "link_a")
    pose = lambda q, link: np.block([[np.eye(3), np.array([[q[0]], [0.0], [0.0]])], [np.zeros((1, 3)), np.ones((1, 1))]])  # noqa: E731
    chk = ArmChecker((shape,), pose, lambda qs: np.asarray(qs, float), margin=0.005)
    assert chk([0.50], wall_field(0.56))["hits"] == 0           # far side 0.51 + margin < 0.56
    assert chk([0.548], wall_field(0.56))["hits"] > 0          # near face within 5 mm of the wall
    assert chk([0.548], wall_field(0.56))["links"] == ["link_a"]


def test_obstacle_source_config_and_cli():
    from AG3S.trajopt import serve_safe
    from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError

    base = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml", "--geo-grasp"]
    over = serve_safe.cost_overrides(serve_safe.build_parser().parse_args(base + ["--geo-obstacle-source", "field"]))
    assert TrajOptConfig().with_overrides({"cost": over}).cost.geo_obstacle_source == "field"
    assert "geo_obstacle_source" not in serve_safe.cost_overrides(serve_safe.build_parser().parse_args(base))
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig().with_overrides({"cost": {"geo_obstacle_source": "field"}})            # needs geo_grasp
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig().with_overrides({"cost": {"geo_grasp": True, "geo_obstacle_source": "tsdf"}})
