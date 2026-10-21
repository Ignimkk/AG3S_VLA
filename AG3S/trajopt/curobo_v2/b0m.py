"""T63b B0m — our TO's obstacle-margin rule, carried into the cuRobo B0 world (pure numpy).

User decision 2 of T63b (2026-10-21): B0m = B0 + the server's `obstacle_margin` subtracted from the
**non-target** world distances, by our TO's rule — information only, no objective term.

Our TO's rule (snapshot `f0d8046`, `linearize.CollisionLinearizer._esdf_clearance_core` +
`_obstacle_rows`, `config.py:781-812`): every robot row is classified by what its nearest surface
is, from its distance `d` alone, and only the `obstacle` class gets `max(margin, obstacle_margin)`:

| order | class | condition (`tol` = the finest field voxel, `scene.esdf.grid.voxel_size`) |
|---|---|---|
| 1 | `target` | `|d − d_obj| ≤ tol` **and** the row's link may touch the target (`target_permission_strict`: `scene.target_contact_rows`; minus `nonauthorized_target_rows`). `d_obj` = distance to the manipulated object (`_manipulated_distance`: held spheres analytic, else the target points) |
| — | (`target_volume` ball / `layer_diff`) | only with `collision.target_volume_exempt` — **refused here** (not on in any served config of T63/T63b) |
| 2 | `support` | `|d − (n_k·p − o_k)| ≤ 2 tol` for some support plane `k` (skipped with `obstacle_margin_support`) |
| 3 | `obstacle` | the rest |

B0m applies the same table to the **B0 world distance** `d_B0` of each sphere and step: margin =
`obstacle_margin` on `obstacle`, 0 otherwise — the B0m world distance is `d_B0 − margin`.
(`composed_world.ComposedSceneCollision` mode `b0m` does it on the GPU per optimizer query;
`classify` / `margin` here are the numpy reference.)

What B0m does **not** carry (stated in `T63b.impl.md`): the contact margins of `target` rows
(`manipulated_link_margin`, 20 / 50 mm — B4's addition), the target for non-authorized rows inside
the box (the B0 world removes it for every sphere there), held query points (not in cuRobo).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

#: same codes as `linearize.OBSTACLE_CLASSES` (`_OBSTACLE, _TARGET, _SUPPORT = 0, 1, 2`)
OBSTACLE, TARGET, SUPPORT = 0, 1, 2
CLASS_NAMES = ("obstacle", "target", "support")
#: `linearize._SUPPORT_TOL_VOXELS`
SUPPORT_TOL_VOXELS = 2.0
#: plane slots of the GPU rule (fixed for CUDA graphs); AG3S `support_surface.max_surfaces` is 2
MAX_PLANES = 8


def _robot_bool(rows, n: int, fill: bool) -> np.ndarray:
    out = np.full(n, fill, bool)
    if rows is not None:
        r = np.asarray(rows, bool).reshape(-1)[:n]
        out[:r.shape[0]] = r
    return out


def rules_from_scene(scene, n_spheres: int) -> dict[str, Any]:
    """What our TO's `_obstacle_rows` reads, from the `SceneSnapshot` it certifies against (our
    sphere order). Raises when the scene asks for a rule B0m does not implement."""
    n = int(n_spheres)
    m = float(getattr(scene, "obstacle_margin", 0.0) or 0.0)
    if getattr(scene, "target_volume", None) is not None:
        raise NotImplementedError(
            "B0m: scene.target_volume (collision.target_volume_exempt) is set — the ball / "
            "layer_diff target test is not implemented in the B0m world")
    esdf = getattr(scene, "esdf", None)
    tol = float(esdf.grid.voxel_size) if esdf is not None else float("nan")
    # `CollisionLinearizer.target_contact` + `_nonauthorized_target_rows`, as `_obstacle_rows`
    strict = bool(getattr(scene, "target_permission_strict", False))
    contact = (_robot_bool(getattr(scene, "target_contact_rows", None), n, False) if strict
               else np.ones(n, bool))
    nonauth = getattr(scene, "nonauthorized_target_rows", None)
    if nonauth is not None:
        contact &= ~_robot_bool(nonauth, n, False)
    pts = getattr(scene, "manipulated_points", None)
    sph = getattr(scene, "manipulated_spheres", None)
    sph_r = getattr(scene, "manipulated_sphere_radii", None)
    target: dict[str, Any] = {"kind": None}
    if sph is not None and len(sph):
        target = {"kind": "spheres", "centres": np.asarray(sph, np.float64).reshape(-1, 3),
                  "radii": np.asarray(sph_r, np.float64).reshape(-1)}
    elif pts is not None and len(pts):
        target = {"kind": "points", "points": np.asarray(pts, np.float64).reshape(-1, 3)}
    support_exempt = not bool(getattr(scene, "obstacle_margin_support", False))
    planes = getattr(scene, "support_planes", None)
    planes = (np.zeros((0, 4)) if planes is None or not support_exempt
              else np.asarray(planes, np.float64).reshape(-1, 4))
    if planes.shape[0] > MAX_PLANES:
        raise ValueError(f"B0m: {planes.shape[0]} support planes > MAX_PLANES {MAX_PLANES}")
    return {"obstacle_margin": m, "target_tol": tol, "support_tol": SUPPORT_TOL_VOXELS * tol,
            "contact": contact, "target": target, "planes": planes,
            "support_exempt": support_exempt, "strict": strict,
            "nonauthorized_rows": nonauth is not None}


def target_distance(p_base: np.ndarray, target: Mapping[str, Any]) -> Optional[np.ndarray]:
    """`linearize._manipulated_distance` (`(N,)`, `None` = nothing manipulated)."""
    kind = target.get("kind")
    p = np.asarray(p_base, np.float64).reshape(-1, 3)
    if kind == "spheres":
        c, r = np.asarray(target["centres"]), np.asarray(target["radii"])
        return (np.linalg.norm(p[:, None, :] - c[None], axis=2) - r[None]).min(axis=1)
    if kind == "points":
        from scipy.spatial import cKDTree

        return cKDTree(np.asarray(target["points"], np.float64)).query(p)[0]
    return None


def classify(d: np.ndarray, p_base: np.ndarray, d_obj: Optional[np.ndarray],
             rules: Mapping[str, Any]) -> np.ndarray:
    """`(H, S)` int8 classes of rows with distance `d (H, S)` at centres `p_base (H, S, 3)` (robot
    base frame, our sphere order) — `_obstacle_rows` on robot rows."""
    d = np.asarray(d, np.float64)
    cls = np.full(d.shape, OBSTACLE, np.int8)
    if d_obj is not None:
        hit = np.abs(d - np.asarray(d_obj, np.float64).reshape(d.shape)) <= rules["target_tol"]
        hit &= np.asarray(rules["contact"], bool)[None, :]
        cls[hit] = TARGET
    planes = np.asarray(rules["planes"], np.float64).reshape(-1, 4)
    if planes.shape[0]:
        p = np.asarray(p_base, np.float64).reshape(-1, 3)
        s = p @ planes[:, :3].T - planes[None, :, 3]
        on = (np.abs(s - d.reshape(-1, 1)) <= rules["support_tol"]).any(axis=1).reshape(d.shape)
        cls[on & (cls == OBSTACLE)] = SUPPORT
    return cls


def margin(cls: np.ndarray, obstacle_margin: float) -> np.ndarray:
    """B0m's margin per row: `obstacle_margin` on `obstacle` rows, 0 elsewhere (B0 has no other
    margin: no `esdf_margin` (0 on the server), no contact margins)."""
    return np.where(np.asarray(cls) == OBSTACLE, float(obstacle_margin), 0.0)


def to_curobo(rules: Mapping[str, Any], perm: np.ndarray) -> dict[str, Any]:
    """`rules` with the per-sphere vector in cuRobo's sphere order (`perm[i_curobo] = i_ours`)."""
    out = dict(rules)
    out["contact"] = np.asarray(rules["contact"], bool)[np.asarray(perm)]
    return out


def counts(cls: np.ndarray) -> dict[str, int]:
    c = np.asarray(cls).reshape(-1)
    return {name: int((c == k).sum()) for k, name in enumerate(CLASS_NAMES)}


__all__ = ["CLASS_NAMES", "MAX_PLANES", "OBSTACLE", "SUPPORT", "SUPPORT_TOL_VOXELS", "TARGET",
           "classify", "counts", "margin", "rules_from_scene", "target_distance", "to_curobo"]
