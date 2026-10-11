"""Deterministic synthetic scenes with full ground truth.

No real captures exist for this workspace, so accuracy claims have to come from somewhere the truth
is known exactly. These fixtures analytically ray-cast a tabletop scene — a plane plus a sphere, a
box and a cylinder — through a real pinhole camera, so the depth image is *exact* (no mesh, no
rasterizer, no sampling noise) and every pixel carries the id of the surface it hit.

That gives grounding and clustering a numerical target rather than a visual one: a cluster can be
scored against the GT label mask by IoU, and a fitted centroid against a centroid computed from the
same pixels.

Two subtleties the tests must respect, both properties of single-view depth rather than of AG3S:

* **A fitted centroid is not the object's centre.** One camera sees the near hemisphere only, so the
  observed centroid sits toward the camera — for a 4 cm sphere, about 1.6 cm. `visible_centroid`
  records where the truth actually is for a single view; `center` records the object's real centre.
  Tests compare against the former and report the gap to the latter.
* **The attention map deliberately leaks.** `attention_leak` bleeds mass from the target onto a
  neighbouring object, which is exactly the 2D failure mode §3.3 says 3D geometric connectivity is
  there to fix. A fixture without leakage could not tell a working grounding stage from a broken one.

Everything is a pure function of its arguments. No RNG unless `noise_std > 0`, and then it is a
seeded `default_rng`.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional, Sequence

import numpy as np

# ------------------------------------------------------------------------------ scene objects

TABLE_Z = 0.75
TABLE_XY = ((0.28, 1.15), (-0.55, 0.55))

LABEL_NONE = 0
LABEL_TABLE = 1
LABEL_SPHERE = 2  # the target
LABEL_BOX = 3
LABEL_CYLINDER = 4
LABEL_ROBOT = 5

LABEL_NAMES = {
    LABEL_NONE: "none",
    LABEL_TABLE: "table",
    LABEL_SPHERE: "sphere",
    LABEL_BOX: "box",
    LABEL_CYLINDER: "cylinder",
    LABEL_ROBOT: "robot",
}


@dataclasses.dataclass(frozen=True)
class GTObject:
    """Ground truth for one scene object."""

    label: int
    name: str
    kind: str  # sphere | box | cylinder
    center: np.ndarray  # (3,) true centre
    extent: np.ndarray  # sphere: (r, r, r); box: half extents; cylinder: (r, r, half_height)

    @property
    def bounding_radius(self) -> float:
        if self.kind == "sphere":
            return float(self.extent[0])
        return float(np.linalg.norm(self.extent))


@dataclasses.dataclass(frozen=True)
class SyntheticScene:
    """A rendered scene plus everything needed to check what a stage did with it."""

    depth: np.ndarray  # (H, W) float64 metres, 0.0 where nothing was hit
    labels: np.ndarray  # (H, W) int32, LABEL_*
    camera_intrinsics: np.ndarray  # (3, 3)
    T_base_cam: np.ndarray  # (4, 4)
    attention_grid: np.ndarray  # (G, G) float32 raw patch attention
    objects: tuple[GTObject, ...]
    target_label: int
    table_z: float
    robot_state: Optional[np.ndarray] = None
    robot_model: Any = None

    @property
    def hw(self) -> tuple[int, int]:
        return int(self.depth.shape[0]), int(self.depth.shape[1])

    def object_by_label(self, label: int) -> GTObject:
        for o in self.objects:
            if o.label == label:
                return o
        raise KeyError(f"no object with label {label}")

    @property
    def target(self) -> GTObject:
        return self.object_by_label(self.target_label)

    def points_of(self, label: int) -> np.ndarray:
        """Base-frame points of every pixel labelled `label`, straight from the depth image."""
        rows, cols = np.nonzero(self.labels == label)
        if rows.size == 0:
            return np.zeros((0, 3), np.float64)
        return _unproject(self.depth[rows, cols], cols, rows, self.camera_intrinsics, self.T_base_cam)

    def visible_centroid(self, label: int) -> np.ndarray:
        """Centroid of the *visible surface* — the honest target for a single-view fit."""
        pts = self.points_of(label)
        return pts.mean(axis=0) if pts.size else np.full(3, np.nan)

    def pixel_count(self, label: int) -> int:
        return int((self.labels == label).sum())


# ------------------------------------------------------------------------------ ray casting


def _unproject(z: np.ndarray, cols: np.ndarray, rows: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    cam = np.stack([(cols - cx) * z / fx, (rows - cy) * z / fy, z], axis=1)
    return cam @ T[:3, :3].T + T[:3, 3]


def _look_at(eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """`T_base_cam` for an OpenCV-convention camera (+x right, +y down, +z forward)."""
    eye = np.asarray(eye, np.float64)
    forward = np.asarray(target, np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, np.float64))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, forward, eye
    return T


def _hit_sphere(o: np.ndarray, d: np.ndarray, c: np.ndarray, r: float) -> np.ndarray:
    """Smallest positive ray parameter hitting the sphere; +inf where there is no hit."""
    oc = o - c
    a = (d * d).sum(axis=1)
    b = 2.0 * (d * oc).sum(axis=1)
    cc = (oc * oc).sum() - r * r
    disc = b * b - 4.0 * a * cc
    t = np.full(d.shape[0], np.inf)
    ok = disc >= 0.0
    if not np.any(ok):
        return t
    sq = np.sqrt(disc[ok])
    t0 = (-b[ok] - sq) / (2.0 * a[ok])
    t1 = (-b[ok] + sq) / (2.0 * a[ok])
    near = np.where(t0 > 1e-9, t0, t1)
    t[ok] = np.where(near > 1e-9, near, np.inf)
    return t


def _hit_box(o: np.ndarray, d: np.ndarray, c: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Axis-aligned slab method."""
    lo, hi = c - half, c + half
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t_lo = (lo - o) * inv
        t_hi = (hi - o) * inv
    t_near = np.nanmax(np.minimum(t_lo, t_hi), axis=1)
    t_far = np.nanmin(np.maximum(t_lo, t_hi), axis=1)
    hit = (t_far >= np.maximum(t_near, 0.0)) & (t_far > 1e-9)
    return np.where(hit & (t_near > 1e-9), t_near, np.inf)


def _hit_cylinder(o: np.ndarray, d: np.ndarray, c: np.ndarray, r: float, half_h: float) -> np.ndarray:
    """Z-axis-aligned finite cylinder, including its end caps."""
    n = d.shape[0]
    oc = o - c
    a = (d[:, :2] ** 2).sum(axis=1)
    b = 2.0 * (d[:, :2] * oc[:2]).sum(axis=1)
    cc = float((oc[:2] ** 2).sum()) - r * r
    best = np.full(n, np.inf)

    disc = b * b - 4.0 * a * cc
    side = (disc >= 0.0) & (a > 1e-12)
    if np.any(side):
        sq = np.sqrt(disc[side])
        for t_cand in ((-b[side] - sq) / (2.0 * a[side]), (-b[side] + sq) / (2.0 * a[side])):
            z = o[2] + t_cand * d[side, 2]
            ok = (t_cand > 1e-9) & (np.abs(z - c[2]) <= half_h)
            idx = np.nonzero(side)[0][ok]
            best[idx] = np.minimum(best[idx], t_cand[ok])

    for z_cap in (c[2] - half_h, c[2] + half_h):
        with np.errstate(divide="ignore", invalid="ignore"):
            t_cand = (z_cap - o[2]) / d[:, 2]
        p = o[:2] + t_cand[:, None] * d[:, :2]
        ok = np.isfinite(t_cand) & (t_cand > 1e-9) & (((p - c[:2]) ** 2).sum(axis=1) <= r * r)
        best[ok] = np.minimum(best[ok], t_cand[ok])
    return best


def _hit_plane_rect(o: np.ndarray, d: np.ndarray, z: float, xy: tuple) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z - o[2]) / d[:, 2]
    p = o[None, :] + t[:, None] * d
    inside = (
        np.isfinite(t)
        & (t > 1e-9)
        & (p[:, 0] >= xy[0][0]) & (p[:, 0] <= xy[0][1])
        & (p[:, 1] >= xy[1][0]) & (p[:, 1] <= xy[1][1])
    )
    return np.where(inside, t, np.inf)


# --------------------------------------------------------------------------- attention grid


def _project(points: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Base-frame points -> pixel (u, v). Inverse of `_unproject`."""
    R, t = T[:3, :3], T[:3, 3]
    cam = (np.asarray(points, np.float64).reshape(-1, 3) - t) @ R
    uv = np.empty((cam.shape[0], 2))
    uv[:, 0] = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    uv[:, 1] = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    return uv


def make_attention_grid(
    scene_hw: tuple[int, int],
    K: np.ndarray,
    T_base_cam: np.ndarray,
    target_center: np.ndarray,
    *,
    grid: int = 16,
    sigma_patches: float = 1.1,
    leak_center: Optional[np.ndarray] = None,
    leak_weight: float = 0.0,
    leak_sigma_patches: float = 1.4,
    floor: float = 0.02,
) -> np.ndarray:
    """A `grid x grid` patch attention map peaked on the target's projection.

    `leak_weight > 0` adds a second lobe on `leak_center`. This is the whole reason the fixture
    exists in this shape: real VLA attention smears across nearby objects, and a grounding stage that
    simply thresholds attention will merge them. The `floor` keeps every patch slightly non-zero, so
    a stage that (wrongly) used attention to *select geometry* would still see the whole scene and
    could not accidentally look correct.

    Matches π0.5's 16x16 agent-view patch layout by default (`Baseline/knows_vla/probe_p0b.py`).
    """
    h, w = scene_hw
    uv = _project(np.atleast_2d(target_center), K, T_base_cam)[0]
    px, py = uv[0] / w * grid, uv[1] / h * grid

    gy, gx = np.mgrid[0:grid, 0:grid]
    gx = gx + 0.5
    gy = gy + 0.5
    a = np.exp(-((gx - px) ** 2 + (gy - py) ** 2) / (2.0 * sigma_patches**2))

    if leak_center is not None and leak_weight > 0.0:
        luv = _project(np.atleast_2d(leak_center), K, T_base_cam)[0]
        lx, ly = luv[0] / w * grid, luv[1] / h * grid
        a = a + leak_weight * np.exp(
            -((gx - lx) ** 2 + (gy - ly) ** 2) / (2.0 * leak_sigma_patches**2)
        )

    a = a + floor
    return (a / a.max()).astype(np.float32)


# ------------------------------------------------------------------------------ scene builder


DEFAULT_OBJECTS: tuple[GTObject, ...] = (
    GTObject(LABEL_SPHERE, "target_sphere", "sphere",
             np.array([0.62, 0.00, TABLE_Z + 0.040]), np.array([0.040, 0.040, 0.040])),
    GTObject(LABEL_BOX, "neighbour_box", "box",
             np.array([0.62, 0.135, TABLE_Z + 0.050]), np.array([0.045, 0.045, 0.050])),
    GTObject(LABEL_CYLINDER, "far_cylinder", "cylinder",
             np.array([0.86, -0.190, TABLE_Z + 0.060]), np.array([0.035, 0.035, 0.060])),
)
"""Sphere at the centre, box 13.5 cm to its left, cylinder well away.

The box sits close enough that a 2D attention blob covers both — the gap between their surfaces is
5.0 cm, comfortably more than the 3 cm DBSCAN `eps`, so 3D connectivity can separate what 2D cannot.
That specific arrangement is what `test_target_grounding.py` measures.
"""


def make_scene(
    *,
    hw: tuple[int, int] = (480, 640),
    focal: float = 600.0,
    eye: Sequence[float] = (0.05, 0.0, 1.42),
    look_at: Sequence[float] = (0.66, 0.0, TABLE_Z + 0.02),
    objects: Sequence[GTObject] = DEFAULT_OBJECTS,
    target_label: int = LABEL_SPHERE,
    table_z: float = TABLE_Z,
    table_xy: tuple = TABLE_XY,
    attention_leak: float = 0.55,
    attention_grid_size: int = 16,
    robot_model: Any = None,
    robot_state: Optional[np.ndarray] = None,
    noise_std: float = 0.0,
    seed: int = 0,
) -> SyntheticScene:
    """Render the scene exactly, by ray casting. Deterministic unless `noise_std > 0`.

    When `robot_model` and `robot_state` are given, its collision spheres are rendered into the same
    depth image and labelled `LABEL_ROBOT`, so the self-filter has something real to remove and the
    test can assert it removed exactly that.
    """
    h, w = hw
    K = np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], np.float64)
    T = _look_at(eye, look_at)

    rows, cols = np.mgrid[0:h, 0:w]
    rows, cols = rows.ravel().astype(np.float64), cols.ravel().astype(np.float64)
    d_cam = np.stack(
        [(cols - K[0, 2]) / K[0, 0], (rows - K[1, 2]) / K[1, 1], np.ones_like(cols)], axis=1
    )
    d_base = d_cam @ T[:3, :3].T  # unnormalized on purpose: then t == camera-frame depth
    o = T[:3, 3]

    best_t = np.full(rows.size, np.inf)
    best_label = np.zeros(rows.size, np.int32)

    def _accumulate(t: np.ndarray, label: int) -> None:
        closer = t < best_t
        best_t[closer] = t[closer]
        best_label[closer] = label

    _accumulate(_hit_plane_rect(o, d_base, table_z, table_xy), LABEL_TABLE)
    for obj in objects:
        if obj.kind == "sphere":
            t = _hit_sphere(o, d_base, obj.center, float(obj.extent[0]))
        elif obj.kind == "box":
            t = _hit_box(o, d_base, obj.center, obj.extent)
        elif obj.kind == "cylinder":
            t = _hit_cylinder(o, d_base, obj.center, float(obj.extent[0]), float(obj.extent[2]))
        else:
            raise ValueError(f"unknown object kind {obj.kind!r}")
        _accumulate(t, obj.label)

    if robot_model is not None and robot_state is not None:
        centers, radii = robot_model.sphere_centers_numeric(np.asarray(robot_state, np.float64))
        for c, r in zip(centers, radii):
            _accumulate(_hit_sphere(o, d_base, c, float(r)), LABEL_ROBOT)

    depth = np.where(np.isfinite(best_t), best_t, 0.0).reshape(h, w)
    labels = best_label.reshape(h, w)
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        depth = np.where(depth > 0.0, depth + rng.normal(0.0, noise_std, depth.shape), 0.0)

    target = next(o_ for o_ in objects if o_.label == target_label)
    leak_source = next((o_ for o_ in objects if o_.label != target_label), None)
    attention = make_attention_grid(
        (h, w), K, T, target.center,
        grid=attention_grid_size,
        leak_center=None if leak_source is None else leak_source.center,
        leak_weight=attention_leak,
    )

    return SyntheticScene(
        depth=depth,
        labels=labels,
        camera_intrinsics=K,
        T_base_cam=T,
        attention_grid=attention,
        objects=tuple(objects),
        target_label=target_label,
        table_z=table_z,
        robot_state=None if robot_state is None else np.asarray(robot_state, np.float64),
        robot_model=robot_model,
    )


# ------------------------------------------------------------------- analytic robot mock


FINGERTIP_LINK = "mock_fingertip"
"""The mock arm's contact-authorized link — its distal-most sphere.

Separate from `link_1` on purpose. A clearance test that relaxed a whole link could not tell a
correct per-sphere margin matrix from one that relaxes too much, because every sphere on the link
would agree. With one sphere carrying its own name, "only the fingertip was relaxed" is a checkable
statement about three specific neighbours.
"""


class PlanarTwoLinkArm:
    """A two-link planar arm in the x-z plane — the analytic `RobotCollisionModel` mock.

    §3.5 says a real Pinocchio implementation is out of scope and the constraint builder should be
    tested against an analytic mock. This is that mock, and it is deliberately trivial: closed-form
    FK, a handful of spheres, and an exactly-known answer for every distance query, so a failing
    `test_constraint_builder` or `test_to_contract` is unambiguously AG3S's fault and not the robot
    model's.

    It also stands in for a robot inside `make_scene`, where a real RB-Y1 configuration would put
    most of its geometry outside the camera frustum.

    Joint 0 rotates link 0 about +y from the +x axis; joint 1 is relative to link 0. The last sphere
    of the last link is named `FINGERTIP_LINK` so the clearance tests can relax exactly one sphere
    and check that its three neighbours on the same link stay at full margin.
    """

    def __init__(
        self,
        base: Sequence[float] = (0.30, -0.34, 1.30),
        link_lengths: Sequence[float] = (0.30, 0.28),
        sphere_radius: float = 0.045,
        spheres_per_link: int = 4,
    ):
        self.base = np.asarray(base, np.float64).reshape(3)
        self.link_lengths = tuple(float(v) for v in link_lengths)
        self.sphere_radius = float(sphere_radius)
        self.spheres_per_link = int(spheres_per_link)

    @property
    def nq(self) -> int:
        return len(self.link_lengths)

    #: T26: the mock's "gripper" inner opening (m). A planar two-link arm has no fingers, so the
    #: number is the fixture's own statement — wide enough that every `DEFAULT_OBJECTS` member
    #: (80 mm sphere, 90 mm box, 70 mm cylinder) is graspable, as the pre-T26 tests assumed.
    MOCK_MAX_OPENING = 0.12

    def gripper_openings(self) -> list[dict]:
        """`RobotCollisionModel` T26 extension (`UrdfSphereChain.gripper_openings` is the real one)."""
        return [{"parent_link": "link_1", "fingers": [FINGERTIP_LINK],
                 "joints": {}, "origin_gap_m": self.MOCK_MAX_OPENING,
                 "inner_gap_m": self.MOCK_MAX_OPENING, "geometry": "PlanarTwoLinkArm mock"}]

    @property
    def n_spheres(self) -> int:
        return self.nq * self.spheres_per_link

    @property
    def radii(self) -> np.ndarray:
        return np.full(self.n_spheres, self.sphere_radius)

    @property
    def sphere_link_names(self) -> tuple[str, ...]:
        """`link_0` for the upper arm, `link_1` for the forearm, in sphere order.

        The mock needs these for the same reason the real robot does: a clearance policy that says
        "only the fingertips may touch the target" has to be able to point at a specific row. The
        last sphere of the last link doubles as the fingertip in `tests/ag3s/test_clearance.py`,
        which is why `FINGERTIP_LINK` is a separate name rather than the link's own.
        """
        names: list[str] = []
        for j in range(self.nq):
            link = f"link_{j}"
            for i in range(self.spheres_per_link):
                last = j == self.nq - 1 and i == self.spheres_per_link - 1
                names.append(FINGERTIP_LINK if last else link)
        return tuple(names)

    def link_pose_numeric(self, q: np.ndarray, link: str) -> np.ndarray:
        """Pose of a link frame: origin at the link's distal end, +x along the link.

        Attached-object tests need a parent frame that actually rotates with `q`, so this returns a
        full 4x4 rather than a position — a translation-only "pose" would let a wrong
        `T_parent_object` convention pass unnoticed.
        """
        q = np.asarray(q, np.float64).reshape(-1)
        index = self._link_index(link)
        p = self.base.copy()
        angle = 0.0
        R = np.eye(3)
        for j, length in enumerate(self.link_lengths):
            angle += float(q[j])
            c, s = math.cos(angle), math.sin(angle)
            direction = np.array([c, 0.0, s])
            R = np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
            p = p + direction * length
            if j == index:
                break
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T

    def link_pose_symbolic(self, q: Any, link: str) -> Any:
        """The same chain in CasADi, so a held object's motion is differentiable in `q`."""
        import casadi as ca

        index = self._link_index(link)
        p = ca.DM(self.base.reshape(3, 1))
        angle = 0.0
        R = ca.DM(np.eye(3))
        for j, length in enumerate(self.link_lengths):
            angle = angle + q[j]
            c, s = ca.cos(angle), ca.sin(angle)
            R = ca.vertcat(
                ca.horzcat(c, 0.0, -s), ca.horzcat(0.0, 1.0, 0.0), ca.horzcat(s, 0.0, c)
            )
            p = p + ca.vertcat(c, 0.0, s) * length
            if j == index:
                break
        return ca.vertcat(ca.horzcat(R, p), ca.horzcat(ca.DM.zeros(1, 3), ca.DM(1.0)))

    def _link_index(self, link: str) -> int:
        if link == FINGERTIP_LINK:
            return self.nq - 1
        if link.startswith("link_"):
            try:
                return int(link.split("_")[1])
            except (IndexError, ValueError):
                pass
        raise KeyError(f"no link {link!r} on the mock arm; have {sorted(set(self.sphere_link_names))}")

    def _fractions(self) -> np.ndarray:
        n = self.spheres_per_link
        return np.linspace(1.0 / n, 1.0, n)

    def sphere_centers_numeric(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, np.float64).reshape(-1)
        if q.shape[0] != self.nq:
            raise ValueError(f"q must have {self.nq} entries, got {q.shape[0]}")
        centres = []
        p = self.base.copy()
        angle = 0.0
        for j, length in enumerate(self.link_lengths):
            angle += float(q[j])
            direction = np.array([math.cos(angle), 0.0, math.sin(angle)])
            for f in self._fractions():
                centres.append(p + direction * (length * f))
            p = p + direction * length
        return np.asarray(centres, np.float64), self.radii

    def sphere_centers_symbolic(self, q: Any) -> list[tuple[Any, float]]:
        import casadi as ca

        out: list[tuple[Any, float]] = []
        p = ca.DM(self.base.reshape(3, 1))
        angle = 0.0
        for j, length in enumerate(self.link_lengths):
            angle = angle + q[j]
            direction = ca.vertcat(ca.cos(angle), 0.0, ca.sin(angle))
            for f in self._fractions():
                out.append((p + direction * (length * float(f)), self.sphere_radius))
            p = p + direction * length
        return out

    def end_effector_numeric(self, q: np.ndarray) -> np.ndarray:
        centres, _ = self.sphere_centers_numeric(q)
        return centres[-1]


@dataclasses.dataclass(frozen=True)
class ThreeCameraScene:
    """The same analytic world seen from three poses, in the form `AG3S.process_multi` consumes.

    Built by ray-casting `make_scene` from three viewpoints rather than by perturbing one render, so
    the overlap between views is genuine geometry rather than copied points. That matters for the
    fusion tests: two cameras landing in the same voxel has to mean they actually saw the same
    surface, or "provenance from both cameras" would be trivially true.

    The head observation carries `T_base_cam` directly, as a fixed mount would. The two wrist
    observations carry `mount_link` and `T_link_cam` instead, so `resolve_T_base_cam` has to run
    forward kinematics at each one's own `robot_state` — which is the property that separates this
    fixture from three unrelated cameras.
    """

    observations: tuple  # (CameraObservation, ...)
    views: dict  # CameraID -> SyntheticScene, for ground truth
    robot_model: Any
    robot_state: np.ndarray

    @property
    def head(self):
        return self.observations[0]


def make_three_camera_scene(
    *,
    robot_model: Any = None,
    robot_state: Optional[np.ndarray] = None,
    timestamps: Sequence[float] = (1.00, 1.02, 1.04),
    wrist_states: Optional[Sequence[np.ndarray]] = None,
    with_attention: Sequence[bool] = (True, True, True),
    hw: tuple[int, int] = (120, 160),
    **scene_kwargs: Any,
) -> "ThreeCameraScene":
    """Three views of the default scene: a head camera and two wrist cameras.

    Args:
        wrist_states: the `q` each wrist camera was captured at. Defaults to `robot_state` for all
            three, which is the *degenerate* case; pass different values to exercise the reason
            capture-time kinematics exist at all.
        with_attention: per camera. `False` leaves that camera's `attention_map` as None, which must
            cost it nothing geometrically.

    The resolution defaults low (120x160) because these tests are about plumbing — provenance,
    transforms, timestamps — and a full-size render would make the suite slow for no extra coverage.
    """
    from AG3S.ag3s.types import CameraID, CameraObservation

    robot_state = DEFAULT_ARM_Q if robot_state is None else np.asarray(robot_state, np.float64)
    states = list(wrist_states) if wrist_states is not None else [robot_state] * 3

    # Three genuinely different vantage points on the same table: over the top, and from either side
    # closer in, which is roughly how a head camera and two wrist cameras are placed.
    poses = [
        ((0.05, 0.00, 1.42), (0.66, 0.00, TABLE_Z + 0.02)),
        ((0.40, 0.28, 1.12), (0.62, 0.02, TABLE_Z + 0.03)),
        ((0.40, -0.28, 1.12), (0.70, -0.02, TABLE_Z + 0.03)),
    ]
    mounts = [None, "link_0", "link_1"]
    ids = [CameraID.HEAD, CameraID.LEFT_WRIST, CameraID.RIGHT_WRIST]

    observations = []
    views = {}
    for i, (camera_id, (eye, look_at), mount) in enumerate(zip(ids, poses, mounts)):
        view = make_scene(
            hw=hw, eye=eye, look_at=look_at,
            robot_model=robot_model, robot_state=states[i], **scene_kwargs,
        )
        views[camera_id] = view
        kwargs = dict(
            camera_id=camera_id,
            depth=view.depth,
            camera_intrinsics=view.camera_intrinsics,
            robot_state=states[i],
            timestamp=float(timestamps[i]),
            attention_map=view.attention_grid if with_attention[i] else None,
            image_hw=view.hw,
        )
        if mount is None:
            kwargs["T_base_cam"] = view.T_base_cam
        else:
            # The hand-eye calibration, derived so that FK at this camera's own capture state
            # reproduces the pose the view was rendered from. Solving for it rather than inventing
            # one is what lets the test assert an exact match instead of a loose tolerance.
            T_base_link = robot_model.link_pose_numeric(states[i], mount)
            kwargs["mount_link"] = mount
            kwargs["T_link_cam"] = np.linalg.inv(T_base_link) @ view.T_base_cam
        observations.append(CameraObservation(**kwargs))

    return ThreeCameraScene(
        observations=tuple(observations),
        views=views,
        robot_model=robot_model,
        robot_state=robot_state,
    )


DEFAULT_ARM_Q = np.array([-0.60, -0.60])
"""A configuration that puts the mock arm inside the default camera frustum without occluding
anything.

Measured on the default scene: 9,714 robot pixels, while the sphere / box / cylinder pixel counts
are *identical* to the robot-free render (2,519 / 6,855 / 3,375). That equality is the property the
fixture needs — the self-filter has real work to do, and any change in object pixel counts after
filtering is the filter's doing rather than an occlusion artefact.
"""


__all__ = [
    "DEFAULT_ARM_Q",
    "DEFAULT_OBJECTS",
    "FINGERTIP_LINK",
    "GTObject",
    "LABEL_BOX",
    "LABEL_CYLINDER",
    "LABEL_NAMES",
    "LABEL_NONE",
    "LABEL_ROBOT",
    "LABEL_SPHERE",
    "LABEL_TABLE",
    "PlanarTwoLinkArm",
    "SyntheticScene",
    "TABLE_Z",
    "ThreeCameraScene",
    "make_attention_grid",
    "make_scene",
    "make_three_camera_scene",
]
