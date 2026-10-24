"""AG3S-DYNAMIC-FIELD — the field follows moving obstacles (`esdf.free_space_carve` D1, `esdf.robot_occupied_free` D2).

  - D1 `seen_through`: a point a depth ray passed (observed depth beyond it by more than the margin) is seen
    through; a point at or behind the surface, under the robot mask, at an invalid / out-of-range pixel or
    outside the image decides nothing.
  - D2 `AG3S._robot_free_spheres` + `inside_robot_spheres`: inside the shrunk robot spheres, nothing below the
    support plane under each sphere, nothing near the destination, None without a model.
  - config defaults off, validation, the curobo-only rule; the CLI flags map to the esdf keys and need AG3S.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, EsdfConfig
from AG3S.ag3s.fields.curobo_builder import inside_robot_spheres, seen_through

K = np.array([[200.0, 0.0, 32.0], [0.0, 200.0, 24.0], [0.0, 0.0, 1.0]])


def _cam(depth, mask=None):
    """Camera at the base origin looking along +z (T_base_cam = identity)."""
    return types.SimpleNamespace(name="cam", depth=np.asarray(depth, np.float64), camera_intrinsics=K,
                                 T_base_cam=np.eye(4), robot_mask=mask)


def test_seen_through_rules():
    depth = np.full((48, 64), 1.0)              # a wall 1 m ahead fills the image
    depth[0:10, 0:10] = 0.0                     # no return there
    mask = np.zeros((48, 64), bool)
    mask[30:48, 40:64] = True                   # the robot hides the lower right
    pts = np.array([[0.0, 0.0, 0.5],            # in front of the wall: the ray passed it
                    [0.0, 0.0, 0.99],           # on the wall's surface band: not by more than the margin
                    [0.0, 0.0, 1.20],           # behind the wall: never seen
                    [0.05, 0.05, 0.5],          # in front, but behind the robot mask (u≈52, v≈44)
                    [-0.08, -0.06, 0.5],        # in front, at a pixel with no return (u≈0, v≈0)
                    [2.0, 0.0, 0.5],            # outside the image
                    [0.0, 0.0, -0.5]])          # behind the camera
    hit, per = seen_through(pts, [_cam(depth, mask)], margin=0.03, depth_min=0.05, depth_max=3.0)
    assert hit.tolist() == [True, False, False, False, False, False, False]
    assert per == {"cam": 1}


def test_seen_through_depth_range_and_union():
    near = np.full((48, 64), 0.4)               # this camera sees only 0.4 m: nothing at 0.5 m is seen through
    far = np.full((48, 64), 2.5)
    p = np.array([[0.0, 0.0, 0.5]])
    assert not seen_through(p, [_cam(near)], 0.03)[0][0]
    assert not seen_through(p, [_cam(far)], 0.03, depth_max=2.0)[0][0]   # out of range decides nothing
    assert seen_through(p, [_cam(near), _cam(far)], 0.03)[0][0]           # any camera suffices


def _fake_pipeline(centers, radii, *, shrink=0.005, plane_z=None):
    from AG3S.ag3s.runtime.pipeline import AG3S
    model = types.SimpleNamespace(sphere_centers_numeric=lambda q: (np.asarray(centers, float), np.asarray(radii, float)))
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", "tsdf_voxel_size": 0.005,
                                         "robot_occupied_free": True, "robot_free_shrink": shrink}})
    fake = types.SimpleNamespace(_frame_filter_model=model, config=cfg, _guard_pad=lambda: 0.005,
                                 _support_plane_z=AG3S._support_plane_z)
    surfaces = () if plane_z is None else (types.SimpleNamespace(normal=np.array([0.0, 0.0, 1.0]), offset=plane_z),)
    return AG3S._robot_free_spheres.__get__(fake), surfaces


def test_robot_free_spheres_shrink_floor_destination():
    fn, surfaces = _fake_pipeline([[0.0, 0.0, 0.80], [0.3, 0.0, 1.0]], [0.02, 0.003], plane_z=0.79)
    sph = fn(np.zeros(3), surfaces, np.array([[0.0, 0.0, 0.805]]))
    assert len(sph["centres"]) == 1 and abs(sph["radii"][0] - 0.015) < 1e-12   # shrunk by 5 mm; the 3 mm sphere vanishes
    assert abs(sph["z_min"][0] - (0.79 + 0.005)) < 1e-12 and sph["dest_pad"] == 0.010
    pts = np.array([[0.0, 0.0, 0.80],     # inside, above the floor, but 5 mm from the destination point
                    [0.0, 0.010, 0.80],   # inside, 11.2 mm from the destination point: freed
                    [0.0, 0.0, 0.790],    # inside the sphere, below the floor (table + pad)
                    [0.0, 0.016, 0.80],   # 1 mm outside the shrunk sphere
                    [0.0, -0.012, 0.80]]) # inside, above the floor, far from the destination
    assert inside_robot_spheres(pts, sph).tolist() == [False, True, False, False, True]


def test_robot_free_spheres_without_model_or_after_shrink():
    fn, surfaces = _fake_pipeline([[0, 0, 1.0]], [0.004])
    assert fn(np.zeros(3), surfaces, None) is None             # every sphere smaller than the shrink
    fn.__self__._frame_filter_model = None
    assert fn(np.zeros(3), surfaces, None) is None


def test_config_defaults_and_validation():
    e = EsdfConfig()
    assert e.free_space_carve is False and e.robot_occupied_free is False and e.free_space_carve_margin is None
    for bad in ({"free_space_carve": "yes"}, {"free_space_carve_margin": -0.01}, {"robot_free_shrink": -1.0}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"esdf": {"backend": "curobo", **bad}})
    with pytest.raises(AG3SConfigError):   # the legacy numpy field has no TSDF blocks to free
        AG3SConfig.from_dict({"esdf": {"backend": "legacy", "free_space_carve": True}})


BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def test_cli_flags_map_to_esdf_keys():
    from AG3S.trajopt import serve_safe

    parse = lambda *a: serve_safe.build_parser().parse_args([*BASE_ARGV, *a])   # noqa: E731
    assert serve_safe.dynamic_field_section(parse()) == {}
    sec = serve_safe.dynamic_field_section(parse("--free-space-carve", "--free-space-carve-margin-mm", "35",
                                                 "--robot-occupied-free", "--robot-free-shrink-mm", "4",
                                                 "--tsdf-time-decay", "0.99", "--tsdf-frustum-decay", "0.8"))
    assert sec == {"free_space_carve": True, "free_space_carve_margin": 0.035, "robot_occupied_free": True,
                   "robot_free_shrink": 0.004, "time_decay": 0.99, "frustum_decay": 0.8}
    cfg = AG3SConfig.from_dict({"esdf": {"backend": "curobo", **sec}})
    assert cfg.esdf.free_space_carve and cfg.esdf.robot_occupied_free


def test_cli_refuses_the_flags_without_ag3s(monkeypatch, capsys):
    from AG3S.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, "--no-safe", "--free-space-carve"])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    assert e.value.code == 2
    assert "--free-space-carve" in capsys.readouterr().err
