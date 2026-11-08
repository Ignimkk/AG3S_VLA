"""SL-GTTO17 (B, user ruling 2026-11-08) — `serve_safe --held-point-model-mm MM`: the held object's TO query spheres are a
shell of small spheres on its observed points (`held_object.point_model_spheres`) instead of the body-cover spheres.
Pinned: the shell covers every observed point and stays close to them; the support fill; the CLI checks; the scene adapter
reads the point model from the constraint set's metrics (and only for the held parent link); the F14 version."""
from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.ag3s.robot_models.held_object import point_model_spheres


def _bowl(n=400, seed=0):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    rad = rng.uniform(0.02, 0.054, n)
    z = 0.30 + 0.055 * (rad - 0.02) / 0.034          # a bowl-ish cone, rim at 55 mm
    return np.c_[0.7 + rad * np.cos(th), -0.1 + rad * np.sin(th), z]


def test_shell_covers_points_and_hugs_them():
    p = _bowl()
    sph, rec = point_model_spheres(p, 0.02)
    c = np.asarray([s[0] for s in sph])
    r = sph[0][1]
    rim = np.linalg.norm(p[:, :2] - [0.7, -0.1], axis=1) > 0.05   # boundary points (inner-surface points sit in the
    d = np.linalg.norm(p[rim][:, None, :] - c[None], axis=2)      # filled solid's interior voxels, enclosed by the shell)
    assert (d.min(axis=1) <= r + 1e-9).all()
    assert rec["applied"] and rec["n_spheres"] == len(sph) and r == pytest.approx(0.02 * np.sqrt(3) / 2)
    below = p[:, 2].min() - (c[:, 2] - r).min()
    assert 0.0 <= below <= 0.02                                     # bottom within one voxel of the lowest point


def test_fills_down_to_the_support_it_rested_on():
    p = _bowl()
    plane = (np.array([0.0, 0.0, 1.0]), 0.28)                       # support 20 mm under the lowest point
    sph, rec = point_model_spheres(p, 0.02, support=plane)
    c = np.asarray([s[0] for s in sph])
    assert rec["filled_to_support"] is True and c[:, 2].min() < 0.30
    far = (np.array([0.0, 0.0, 1.0]), 0.0)                          # 300 mm below: not where it rested
    _, rec2 = point_model_spheres(p, 0.02, support=far)
    assert rec2["filled_to_support"] is False


def test_empty_input():
    assert point_model_spheres(np.zeros((0, 3)), 0.02)[0] == []


def test_cli_checks_and_port():
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss
    from tests.trajopt.test_t88_passthrough_when_clear import BASE

    ap = ss.build_parser()
    for bad in (["--held-point-model-mm", "20"], ["--held-body-cover", "--held-point-model-mm", "2"]):
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, ap.parse_args(BASE + bad))
    ss.reject_bad_flag_combinations(ap, ap.parse_args(BASE + ["--held-body-cover", "--held-point-model-mm", "20"]))
    assert "held_point_model_mm" in srv.PORTED_FINAL_CONFIG_DESTS


def test_build_kwargs_carry_the_point_model():
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    v = "AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst-pts"
    assert tuple(pf.AG3S_VERSIONS[v]) == tuple(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst"]) + (
        "--held-point-model-mm", "20")
    args = ss.build_parser().parse_args(pf.version_server_argv(v) + pf.version_server_flags(v))
    kw = pf.v11ab_build_kwargs(args)
    assert kw["held_point_model"] == pytest.approx(0.02)
    off = ss.build_parser().parse_args(pf.version_server_argv(v) + pf.version_server_flags(
        "AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp-loose2-obst"))
    assert "held_point_model" not in pf.v11ab_build_kwargs(off)


def test_scene_reads_the_point_model_for_the_held_link_only(monkeypatch):
    import inspect

    from AG3S.trajopt import linearize as lz

    src = inspect.getsource(lz)
    assert 'get("held_point_model")' in src and 'str(hp.get("parent_link")) == str(attached_link)' in src


# ---- pipeline: attach with the point model (the RB-Y1 apple fixture of T43 R2) ------------------------------------
def test_attach_builds_the_point_shell_and_the_frame_lifts_it():
    from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
    from tests.ag3s import test_t43r2_held_body_cover as r2

    model = UrdfSphereChain(parse_urdf(RBY1_URDF), link_filter=["ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2",
                                                                 "ee_finger_r1", "ee_finger_r2"])
    orig = r2._ag3s

    def with_points(m, body_cover):
        a = orig(m, body_cover)
        a.held_point_model_m = 0.01
        return a

    r2._ag3s = with_points
    try:
        ag3s, held, q, table, apple = r2._grasp(model, "left", True)
    finally:
        r2._ag3s = orig
    pm = ag3s.held_record["body_cover"]["point_model"]
    assert pm["applied"] and pm["n_spheres"] > 4 and ag3s._held_query["points"]           # noqa: SLF001
    surfaces = [r2._plane([0, 0, 1], table, 1)]
    ag3s._frame_held_attached(q, surfaces)                                                  # noqa: SLF001
    link, spheres = ag3s._held_point_frame                                                  # noqa: SLF001
    assert link == held.parent_link and len(spheres) == pm["n_spheres"]
    T = model.link_pose(q, link)
    bottoms = [float(T[2, :3] @ c + T[2, 3]) - r for c, r in spheres]
    band = ag3s.held_record["body_cover"]["band_mm"] / 1000.0
    assert min(bottoms) >= table + band - 1e-6                                              # lifted like the body
    # off: no point frame
    plain, *_ = r2._grasp(model, "left", True)
    plain._frame_held_attached(q, surfaces)                                                 # noqa: SLF001
    assert plain._held_point_frame is None and "point_model" not in plain.held_record["body_cover"]  # noqa: SLF001
