"""--gt exact (2026-11-10): multi-link CBF privileged hazard = MVEE of the obstacle's collision geoms (paper Table III)."""
import numpy as np

from Baseline.paper_baselines.closed_loop import arms as A
from Baseline.paper_baselines.closed_loop import privileged as P


def _obst(active=True):
    R = np.eye(3)
    tall = {"id": 1, "name": "wine_bottle_g0", "type": P.MJ_BOX, "size": np.array([0.03, 0.03, 0.127]),
            "pos": np.array([0.5, 0.1, 0.13]), "rot": R, "collision": True, "group": 0}
    vis = dict(tall, id=2, collision=False, size=np.array([0.3, 0.3, 0.3]))
    return {"name": "wine_bottle_obstacle_1", "display_name": "wine_bottle_1", "active": active, "pos": tall["pos"],
            "geoms": [tall, vis]}


def test_exact_ellipsoid_encloses_the_collision_geom_only():
    (E,) = P.exact_ellipsoids([_obst(), _obst(active=False)])
    c, R, s = np.asarray(E["center"]), np.asarray(E["R"]), np.asarray(E["semi_axes"])
    corners = P._box_corners(_obst()["geoms"][0])
    d = (corners - c) @ R / s
    assert np.all(np.einsum("ij,ij->i", d, d) <= 1.0 + 1e-3)  # every corner inside (MVEE of the corners)
    assert max(s) > 0.2 and max(s) < 0.25  # long axis ~ sqrt(3) * 0.127 (the full bottle, not a trimmed cloud)
    assert E["name"] == "wine_bottle_1" and E["n_points"] == 8  # the visual geom is not used


def test_arm_default_and_record():
    rec = A.resolve("multilink_cbf", gt="default")
    assert rec["gt_source"] == "exact" and rec["privileged"] and rec["client_flags"] == ["--emit-privileged-obstacles"]
    assert A.resolve("knows", gt="default")["gt_source"] == "seg" and A.resolve("aegis", gt="default")["gt_source"] == "names"
    assert A.resolve("multilink_cbf", gt="seg")["gt_source"] == "seg"  # the previous condition stays reachable
