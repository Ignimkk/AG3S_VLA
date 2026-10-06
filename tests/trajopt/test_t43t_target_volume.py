"""T43 T step 2 (ii) — `collision.target_volume_exempt`: a robot row whose nearest field surface point
lies inside the manipulated object's carve-out ball is a **target** row for the obstacle margin (T41 a)
and for the sweep path rows (T43 Q), not only rows that pass E1's `|d − d_object| ≤ voxel`.

Off (default): `SceneSnapshot.target_volume` is None and nothing reads it — bit-identical (the T41 a /
T43 Q suites pin those paths). The analytic scene is T41 a's (`test_t41a_obstacle_margin._rig`).
"""

from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.linearize import OBSTACLE_CLASSES, SceneSnapshot, scene_from_constraint_set
from tests.trajopt import test_t41a_obstacle_margin as A


class _GradTable(A._Table):
    def gradient(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        h = 1e-5
        g = np.stack([(self.distance(p + h * e) - self.distance(p - h * e)) / (2 * h)
                      for e in np.eye(3)], axis=1)
        return g


def _rig():
    lin, centres, _, points = A._rig()
    return lin, centres, _GradTable(), points


def _ball(field, extra=0.005):
    return np.array([*field.apple, A.APPLE_R + extra, -np.inf])


def _no_e1(scene):
    """The apple is not named by E1 (no observed points reach it) — the U case: apple surfaces
    outside the observed points."""
    return dataclasses.replace(scene, manipulated_points=None, manipulated_link_margin=None)


def test_default_is_none_and_off_is_unchanged():
    assert SceneSnapshot.__dataclass_fields__["target_volume"].default is None
    assert TrajOptConfig().collision.target_volume_exempt is False
    lin, centres, field, points = _rig()
    s = A._scene(field, points, margin=A.M, support_planes=A.TABLE_PLANE)
    assert np.array_equal(A._clear(lin, centres, s),
                          A._clear(lin, centres, dataclasses.replace(s, target_volume=None)))


def test_a_row_nearest_the_apple_but_not_named_by_e1_gets_the_margin_off_and_not_on():
    lin, centres, field, points = _rig()
    off = _no_e1(A._scene(field, points, margin=A.M, support_planes=A.TABLE_PLANE))
    on = dataclasses.replace(off, target_volume=_ball(field))
    c_off, c_on = A._clear(lin, centres, off), A._clear(lin, centres, on)
    apple = A.ROWS["apple"]
    assert c_on[apple] - c_off[apple] == pytest.approx(A.M, abs=1e-9)          # margin lifted
    for name in ("bar", "box", "table", "held"):
        assert c_on[A.ROWS[name]] == c_off[A.ROWS[name]], name                  # nothing else moves
    lin._esdf_clearance(centres, on)
    got = [OBSTACLE_CLASSES[k] for k in lin._last_obstacle_class[0]]
    assert got == ["obstacle", "target", "support", "obstacle", "held"]


def test_the_ball_is_about_the_nearest_surface_point_not_the_query_point():
    """A bar row whose query point is inside a (too) large ball keeps the margin: its nearest surface
    (the bar) is outside the ball."""
    lin, centres, field, points = _rig()
    bar_q = centres[0, A.ROWS["bar"]]
    big = np.array([*bar_q, 0.006, -np.inf])           # contains the query point, not the bar surface (8 mm)
    on = dataclasses.replace(A._scene(field, points, margin=A.M, support_planes=A.TABLE_PLANE),
                             target_volume=big)
    off = A._scene(field, points, margin=A.M, support_planes=A.TABLE_PLANE)
    assert A._clear(lin, centres, on)[A.ROWS["bar"]] == A._clear(lin, centres, off)[A.ROWS["bar"]]


def test_z_min_keeps_the_table_out():
    lin, centres, field, points = _rig()
    table_q = centres[0, A.ROWS["table"]]
    ball = np.array([*table_q, 0.02, 0.001])           # the table point (z = 0) is below z_min
    s = A._scene(field, points, margin=A.M)            # no support planes: the table is an obstacle
    on = dataclasses.replace(s, target_volume=ball)
    assert A._clear(lin, centres, on)[A.ROWS["table"]] == A._clear(lin, centres, s)[A.ROWS["table"]]


def test_held_points_are_never_in_the_volume():
    lin, centres, field, points = _rig()
    s = dataclasses.replace(A._scene(field, points), target_volume=np.array([0, 0, 0, 10.0, -np.inf]))
    d = np.asarray(field.distance(centres.reshape(-1, 3))).reshape(centres.shape[:2])
    inside = lin.in_target_volume(d, centres.reshape(-1, 3), s)
    assert not inside[:, lin.n_spheres:].any() and inside[:, :lin.n_spheres].all()


def test_sweep_rows_exempt_the_volume():
    from benchmark.trajopt.sweep import PathChecker

    lin, centres, field, points = _rig()
    pc = object.__new__(PathChecker)
    pc.lin = lin
    pc._floor = np.zeros(centres.shape[1])
    base = _no_e1(A._scene(field, points))
    base = dataclasses.replace(base, manipulated_points=points)   # the E1 test runs, finds nothing
    # move the apple row's query just off E1's reach: E1 compares to the observed points only
    far_pts = points[points[:, 0] < 0]                            # the far half of the apple
    base = dataclasses.replace(base, manipulated_points=far_pts)
    off = pc._target_rows(centres, base)
    on = pc._target_rows(centres, dataclasses.replace(base, target_volume=_ball(field)))
    apple = A.ROWS["apple"]
    assert off is None or not off[0, apple]
    assert on is not None and on[0, apple]
    assert not on[0, A.ROWS["bar"]] and not on[0, A.ROWS["held"]]


def test_scene_from_constraint_set_carries_the_ball_only_when_on_and_nothing_held():
    _, _, field, points = _rig()
    field.stats = {"target_free": {"ball": {"centre_m": [0.0, 0.0, 0.033], "radius_mm": 38.0,
                                            "z_min_m": 0.005}}}
    cs = A._constraint_set(field, points)
    radii = np.full(4, A.R_FINGER)
    off = scene_from_constraint_set(cs, radii, A._config())
    assert off.target_volume is None
    on = scene_from_constraint_set(cs, radii, A._config(target_volume_exempt=True))
    assert np.allclose(on.target_volume, [0.0, 0.0, 0.033, 0.038, 0.005])
    holding = types.SimpleNamespace(**{**vars(cs), "attached": types.SimpleNamespace(
        points=np.zeros((1, 3)), parent_link="ee_left", primitives=[])})
    held = scene_from_constraint_set(holding, radii, A._config(target_volume_exempt=True))
    assert held.target_volume is None
    field.stats = {"target_free": {"ball": None}}
    assert scene_from_constraint_set(cs, radii, A._config(target_volume_exempt=True)).target_volume is None


def test_cli_flag():
    from benchmark.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(A.BASE_ARGV)
    assert "target_volume_exempt" not in serve_safe.obstacle_overrides(args)
    args = serve_safe.build_parser().parse_args(A.BASE_ARGV + ["--target-volume-exempt"])
    assert serve_safe.trajopt_config_from_args(args).collision.target_volume_exempt is True


def test_cli_t43t_ag3s_flags_map_to_the_clustering_section_and_are_off_by_default():
    from benchmark.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(A.BASE_ARGV)
    assert serve_safe.t43t_identity_section(args) == {}
    assert args.held_free_observed_mm is None
    args = serve_safe.build_parser().parse_args(
        A.BASE_ARGV + ["--closing-keep-geometry", "--latched-identity-hold",
                       "--refused-identity-release", "3", "--supported-max-bottom-mm", "60"])
    assert serve_safe.t43t_identity_section(args) == {
        "closing_keep_geometry": True, "latched_identity_hold": True,
        "refused_identity_release_frames": 3, "supported_max_bottom": 0.06}
    from benchmark.ag3s.config import AG3SConfig
    cfg = AG3SConfig.from_dict({"clustering": {"subtask_gate": True}}).with_overrides(
        {"clustering": serve_safe.t43t_identity_section(args)})
    assert cfg.clustering.subtask_gate is True and cfg.clustering.latched_identity_hold is True
    assert cfg.clustering.closing_keep_geometry is True and cfg.clustering.supported_max_bottom == 0.06


@pytest.mark.parametrize("extra", [["--no-safe", "--closing-keep-geometry"],
                                   ["--no-safe", "--target-volume-exempt"],
                                   ["--held-free-observed-mm", "15", "--esdf-backend", "legacy"]])
def test_cli_rejects_t43t_flags_without_their_layer(extra):
    from benchmark.trajopt import serve_safe

    ap = serve_safe.build_parser()
    with pytest.raises(SystemExit):
        args = ap.parse_args(A.BASE_ARGV + extra)
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_server_asserts_the_ag3s_config_it_asked_for():
    """T43 T2 bug guard: `assert_ag3s_config` raises when the backend (or a flag's clustering key)
    is not what the server asked for — `build_ag3s` calls it before and after building AG3S."""
    import inspect

    from benchmark.ag3s.config import AG3SConfig
    from benchmark.trajopt import serve_safe

    ok = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": {"backend": "curobo"}})
    serve_safe.assert_ag3s_config(ok, collision_backend="esdf", esdf_backend="curobo")
    ok2 = ok.with_overrides({"clustering": {"latched_identity_hold": True}})
    serve_safe.assert_ag3s_config(ok2, collision_backend="esdf", esdf_backend="curobo",
                                  clustering={"latched_identity_hold": True})
    with pytest.raises(RuntimeError, match="collision_backend"):
        serve_safe.assert_ag3s_config(AG3SConfig(), collision_backend="esdf", esdf_backend="legacy")
    with pytest.raises(RuntimeError, match="esdf.backend"):
        serve_safe.assert_ag3s_config(ok, collision_backend="esdf", esdf_backend="legacy")
    with pytest.raises(RuntimeError, match="clustering.closing_keep_geometry"):
        serve_safe.assert_ag3s_config(ok, collision_backend="esdf", esdf_backend="curobo",
                                      clustering={"closing_keep_geometry": True})
    src = inspect.getsource(serve_safe.build_ag3s)
    assert src.count("assert_ag3s_config(") == 2
