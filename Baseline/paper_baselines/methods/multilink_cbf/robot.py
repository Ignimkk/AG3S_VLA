"""Panda guarded bodies for the multi-link filter [1]: ellipsoids fitted to the adapter's sphere model + Jacobians.

The paper guards five ellipsoids on an SO-101 (gripper, wrist, three overlapping forearm ellipsoids; elbow / upper
arm unguarded). The Panda analogue, read from ``AG3S/adapters/panda/model.py`` (cuRobo ``franka.yml`` collision
spheres on ``robots/panda/description/panda_libero.urdf``; read only):

=========  ========================================================  ============  =====
group      sphere links                                              frame         split
=========  ========================================================  ============  =====
gripper    panda_hand, panda_leftfinger, panda_rightfinger            panda_hand    1
wrist      panda_link6, panda_link7                                  panda_link6   1
forearm    panda_link5                                               panda_link5   3 (overlapping)
=========  ========================================================  ============  =====

Each ellipsoid = minimum-volume ellipsoid enclosing sampled surface points of its spheres (KNOWS Khachiyan
``mvee``), refitted at every ``filter`` call at the current ``q`` / finger opening (so the link6-link7 relative pose
and the finger opening are exact at the call, rigid within the chunk), then slightly inflated (``inflate``).

Kinematics: ``UrdfSphereChain.link_pose`` (numpy FK of the adapter URDF; its grip site matches the LIBERO
``robot0_eef`` pose, ``tests/benchmark/test_panda_sl2.py``). Jacobians at each ellipsoid centre and at the grip site
by central finite differences (6 x 7, world linear + world angular).
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial import ConvexHull

from Baseline.paper_baselines.methods.knows.perception.ellipsoid_fit import mvee
from Baseline.paper_baselines.methods.safety_common import mat_to_axisangle

EE_FRAME = "panda_grip_site"
#: name -> (sphere links, frame link, number of overlapping pieces)
DEFAULT_GROUPS: dict[str, tuple[tuple[str, ...], str, int]] = {
    "gripper": (("panda_hand", "panda_leftfinger", "panda_rightfinger"), "panda_hand", 1),
    "wrist": (("panda_link6", "panda_link7"), "panda_link6", 1),
    "forearm": (("panda_link5",), "panda_link5", 3),
}


def fibonacci_sphere(n: int = 32) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


@dataclasses.dataclass
class GuardedBody:
    """One guarded ellipsoid at the current q (AEGIS convention: semi-axes + rotation)."""

    name: str
    frame: str
    center_local: np.ndarray  # in the frame link
    R_local: np.ndarray
    semi_axes: np.ndarray
    center: np.ndarray  # base frame at q
    R: np.ndarray
    J: np.ndarray  # (6, 7) [linear; angular] of the centre, base frame

    def summary(self) -> dict:
        return {"name": self.name, "frame": self.frame, "center": self.center.round(4).tolist(),
                "semi_axes": self.semi_axes.round(4).tolist()}


class PandaGuardedBodies:
    """Builds :class:`GuardedBody` list + EE Jacobian for a Panda configuration."""

    def __init__(self, groups: dict | None = None, *, inflate: float = 1.03, n_surface: int = 32, fd_eps: float = 1e-6,
                 overlap: int = 1):
        from AG3S.adapters.panda import model as panda

        self._panda = panda
        self.model = panda.build_panda_bare_model()
        self.groups = dict(groups or DEFAULT_GROUPS)
        known = set(self.model.sphere_link_names)
        for g, (links, frame, _) in self.groups.items():
            panda.check_link_set(links, sphere_links=known, what=f"multilink group {g}")
        self.inflate, self.fd_eps, self.overlap = float(inflate), float(fd_eps), int(overlap)
        self.dirs = fibonacci_sphere(n_surface)
        self.mvee_tol = 1e-3
        self._cache: dict = {}
        self._last_local: list = []

    def provenance(self) -> dict:
        p = self._panda
        return {"sphere_source": str(p.CUROBO_FRANKA_YML.relative_to(p.PROJECT_ROOT)),
                "sphere_sha256": p.file_sha256(p.CUROBO_FRANKA_YML),
                "urdf": str(p.PANDA_URDF.relative_to(p.PROJECT_ROOT)), "urdf_sha256": p.file_sha256(p.PANDA_URDF),
                "groups": {k: {"links": list(v[0]), "frame": v[1], "split": v[2]} for k, v in self.groups.items()},
                "fit": "MVEE (KNOWS Khachiyan) of sphere surface samples, refit at the first action of each call, rigid within the chunk", "inflate": self.inflate,
                "fit_cache": "per (group, piece, relative link poses @1e-4)", "jacobian": f"central finite differences, eps={self.fd_eps}", "ee_frame": EE_FRAME}

    # --- kinematics ---------------------------------------------------------------------------------------------
    def set_fingers(self, robot_state=None, gripper_width: float | None = None) -> None:
        if robot_state is not None and len(np.asarray(robot_state).reshape(-1)) == 9:
            self._panda.set_finger_parameters(self.model, robot_state)
        elif gripper_width is not None:
            w = float(np.clip(gripper_width / 2.0, 0.0, 0.04))
            self.model.set_joint_parameters({"panda_finger_joint1": w, "panda_finger_joint2": w})

    def pose(self, q, frame: str) -> np.ndarray:
        return self.model.link_pose(np.asarray(q, np.float64), frame)

    def jacobian(self, q, frame: str, offset_local: np.ndarray | None = None) -> np.ndarray:
        """(6, 7) Jacobian of the point ``offset_local`` rigidly attached to ``frame`` (base frame)."""
        q = np.asarray(q, np.float64)
        off = np.zeros(3) if offset_local is None else np.asarray(offset_local, np.float64)
        J = np.zeros((6, len(q)))
        for i in range(len(q)):
            dq = np.zeros(len(q))
            dq[i] = self.fd_eps
            Tp, Tm = self.pose(q + dq, frame), self.pose(q - dq, frame)
            J[:3, i] = ((Tp[:3, :3] @ off + Tp[:3, 3]) - (Tm[:3, :3] @ off + Tm[:3, 3])) / (2 * self.fd_eps)
            J[3:, i] = mat_to_axisangle(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * self.fd_eps)
        return J

    # --- bodies -------------------------------------------------------------------------------------------------
    def _group_points(self, q, links, frame) -> list[np.ndarray]:
        """Per sphere: surface sample points in the ``frame`` link frame (each sphere a (n, 3) array)."""
        T_f_inv = np.linalg.inv(self.pose(q, frame))
        out, key = [], []
        for link in links:
            c, r = self.model.local_spheres(link)
            if not len(c):
                continue
            T = T_f_inv @ self.pose(q, link)
            key.append(np.round(T[:3], 4).tobytes())
            for ci, ri in zip(c, r):
                cf = T[:3, :3] @ ci + T[:3, 3]
                out.append(cf + ri * self.dirs)
        return out, b"".join(key)

    def bodies(self, q, *, refit: bool = True) -> list[GuardedBody]:
        """Guarded bodies at ``q``. ``refit=False`` reuses the local fits of the last refit (rigid bodies)."""
        q = np.asarray(q, np.float64)
        if not refit and self._last_local:
            out = []
            for name, frame, c, V, semi in self._last_local:
                T_f = self.pose(q, frame)
                out.append(GuardedBody(name, frame, c, V, semi, T_f[:3, :3] @ c + T_f[:3, 3], T_f[:3, :3] @ V,
                                       self.jacobian(q, frame, c)))
            return out
        out = []
        for g, (links, frame, split) in self.groups.items():
            spheres, key = self._group_points(q, links, frame)
            centres = np.array([s.mean(0) for s in spheres])
            if split > 1:  # overlapping pieces along the principal axis of the sphere centres
                ax = np.linalg.svd(centres - centres.mean(0))[2][0]
                order = np.argsort(centres @ ax)
                n = len(order)
                pieces = [order[max(0, int(pos[0]) - self.overlap): min(n, int(pos[-1]) + 1 + self.overlap)]
                          for pos in np.array_split(np.arange(n), split)]
            else:
                pieces = [np.arange(len(spheres))]
            T_f = self.pose(q, frame)
            for k, idx in enumerate(pieces):
                ck = (g, k, key)
                if ck not in self._cache:  # the local fit depends only on the relative link poses in the group
                    pts = np.concatenate([spheres[i] for i in idx])
                    pts = pts[ConvexHull(pts).vertices]  # MVEE depends only on the hull; Khachiyan cost ~ n points
                    c, Q = mvee(pts, tol=self.mvee_tol)
                    w, V = np.linalg.eigh(0.5 * (Q + Q.T))
                    if np.linalg.det(V) < 0:
                        V[:, 0] *= -1
                    semi = np.sqrt(np.clip(w, 1e-10, None))
                    semi = semi * self.inflate * coverage_scale(pts, c, V, semi)
                    if len(self._cache) > 256:
                        self._cache.clear()
                    self._cache[ck] = (c, V, semi)
                c, V, semi = self._cache[ck]
                name = g if len(pieces) == 1 else f"{g}_{k}"
                out.append(GuardedBody(name, frame, c, V, semi, T_f[:3, :3] @ c + T_f[:3, 3], T_f[:3, :3] @ V,
                                       self.jacobian(q, frame, c)))
        self._last_local = [(b.name, b.frame, b.center_local, b.R_local, b.semi_axes) for b in out]
        return out

    def ee_jacobian(self, q) -> np.ndarray:
        return self.jacobian(q, EE_FRAME)

    def ee_pose(self, q) -> np.ndarray:
        return self.pose(q, EE_FRAME)


def coverage_scale(points: np.ndarray, c: np.ndarray, V: np.ndarray, semi: np.ndarray) -> float:
    """Factor >= 1 so that every point lies inside the ellipsoid (Khachiyan stops at a tolerance, not exactly)."""
    y = ((points - c) @ V) / semi
    return float(max(1.0, np.sqrt(np.max(np.sum(y * y, axis=1)))))


def damped_pinv(J: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """``J^T (J J^T + lambda^2 I)^-1`` (the paper's ``J_ee^+``; damping keeps it finite near singularities)."""
    return J.T @ np.linalg.inv(J @ J.T + damping ** 2 * np.eye(J.shape[0]))


__all__ = ["DEFAULT_GROUPS", "EE_FRAME", "GuardedBody", "PandaGuardedBodies", "damped_pinv", "fibonacci_sphere"]
