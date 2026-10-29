"""AG3S-GEO-GRASP — the parallel two-finger gripper as the grasp-candidate generator sees it.

All coordinates are in the **hold-link frame** ``H`` (RB-Y1: ``ee_left`` / ``ee_right``; MuJoCo ``EE_BODY_L`` /
``EE_BODY_R`` has the same origin and axes): ``+x`` is the closing axis (finger 1 on ``-x``, finger 2 on ``+x``),
``-z`` points from the palm to the finger tips (the approach direction), ``y = z x x``.

The numbers are the collision meshes of ``rby1a/mujoco/rby1.xml`` measured in that frame (`measure_from_mjcf`, test
``test_gripper_spec_matches_mjcf``): finger collision part 2 (the pad) spans ``y`` +-16.3 mm, ``z`` -133.8 .. -75.1 mm,
and its inner face is at ``x = -+(opening / 2)``; the palm block (EE_BODY collision 0-6) spans ``x`` +-63.7 mm,
``y`` +-33.2 mm, ``z`` -73.7 .. 0.7 mm. Boxes are conservative (they contain the meshes they stand for).
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class Box:
    """Axis-aligned box in the hold-link frame (m)."""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def contains(self, p: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """`(N,)` bool: points `p[N, 3]` (hold-link frame) inside the box grown by `margin`."""
        p = np.asarray(p, np.float64).reshape(-1, 3)
        lo = np.asarray(self.lo) - margin
        hi = np.asarray(self.hi) + margin
        return np.all((p >= lo) & (p <= hi), axis=1)

    def surface_points(self, step: float = 0.005) -> np.ndarray:
        """Points on the six faces, no farther than `step` apart (hold-link frame)."""
        lo, hi = np.asarray(self.lo, np.float64), np.asarray(self.hi, np.float64)
        ax = [np.linspace(lo[i], hi[i], max(int(np.ceil((hi[i] - lo[i]) / step)), 1) + 1) for i in range(3)]
        out = []
        for i in range(3):
            j, k = [a for a in range(3) if a != i]
            A, B = np.meshgrid(ax[j], ax[k], indexing="ij")
            for v in (lo[i], hi[i]):
                P = np.zeros((A.size, 3)); P[:, i] = v; P[:, j] = A.ravel(); P[:, k] = B.ravel()
                out.append(P)
        return np.unique(np.round(np.vstack(out), 9), axis=0)


@dataclasses.dataclass(frozen=True)
class ParallelGripperSpec:
    """Pad and body geometry of one parallel gripper in its hold-link frame (see the module docstring)."""

    hold_link: str
    #: largest distance between the two pad faces (m) — AG3S `gripper_max_opening` (robot model, 71.46 mm on RB-Y1).
    max_opening: float
    #: pad face extent: half-width along y and the z interval (m).
    pad_half_y: float
    pad_z: tuple[float, float]
    #: finger thickness behind the pad face along x (m).
    finger_thickness: float
    #: palm block (everything that is not a finger).
    palm: Box
    #: the wrist parts rigid with the hold link (`arm_shape.OccupancyShape`, already grown by the scene margin);
    #: empty = not checked (default; `with_wrist`, task §3g-1).
    wrist: tuple = ()
    #: boundary points (hold-link frame) of the undilated wrist shapes — where a distance field is queried (task §3j).
    wrist_surface: Optional[np.ndarray] = None

    def surface_points(self, opening: float, step: float = 0.005, wrist: bool = False) -> np.ndarray:
        """Boundary points of the palm and both fingers at `opening` (and the wrist with `wrist`), hold-link frame."""
        P = [b.surface_points(step) for b in self.body_boxes(opening)]
        if wrist and self.wrist_surface is not None:
            P.append(self.wrist_surface)
        return np.vstack(P)

    @property
    def pad_centre_depth(self) -> float:
        """Depth (m, > 0) of the pad face centre below the hold-link origin along the approach direction."""
        return -0.5 * (self.pad_z[0] + self.pad_z[1])

    def finger_boxes(self, opening: float) -> tuple[Box, Box]:
        """The two finger boxes at pad-face distance `opening` (m)."""
        h = 0.5 * float(opening)
        t = self.finger_thickness
        y = self.pad_half_y
        z0, z1 = self.pad_z
        return (Box((-h - t, -y, z0), (-h, y, z1)), Box((h, -y, z0), (h + t, y, z1)))

    def body_boxes(self, opening: float) -> tuple[Box, ...]:
        """Palm and both fingers at `opening`."""
        return (self.palm,) + self.finger_boxes(opening)


#: RB-Y1 (rby1a MJCF) left gripper, measured in `EE_BODY_L` (= URDF `ee_left`) — see the module docstring.
RBY1_LEFT = ParallelGripperSpec(
    hold_link="ee_left",
    max_opening=0.07146,
    pad_half_y=0.0163,
    pad_z=(-0.1338, -0.0751),
    finger_thickness=0.0097,
    palm=Box((-0.0637, -0.0332, -0.0737), (0.0637, 0.0332, 0.0007)),
)

#: Same hand on the right arm (mirror-identical in its own frame).
RBY1_RIGHT = dataclasses.replace(RBY1_LEFT, hold_link="ee_right")


def spec_for(hand: str) -> ParallelGripperSpec:
    """The RB-Y1 gripper of `hand` (``"left"`` / ``"right"``)."""
    return RBY1_LEFT if str(hand) == "left" else RBY1_RIGHT


def with_wrist(spec: ParallelGripperSpec, shapes, margin: float) -> ParallelGripperSpec:
    """`spec` with the wrist occupancy shapes (hold-link frame) grown by `margin` (m) — checked wherever the hand boxes
    are, with the same scene points."""
    surf = np.vstack([s.surface_points(0.005) for s in shapes]) if shapes else None
    return dataclasses.replace(spec, wrist=tuple(s.dilated(margin) for s in shapes), wrist_surface=surf)


def measure_from_mjcf(model_xml: str, hand: str = "left") -> dict:
    """Collision-mesh bounds of the hand in its hold-link frame, fingers closed (joint 0) — for the spec test.

    Returns ``{"pad": (lo, hi), "palm": (lo, hi)}`` (m): the union of finger collision part 2 of both fingers (the pad
    part; at joint 0 its inner faces meet at x ~ 0) and the union of the EE_BODY collision parts 0-6 (the palm block)."""
    import mujoco

    side = "L" if hand == "left" else "R"
    s = "l" if hand == "left" else "r"
    m = mujoco.MjModel.from_xml_path(str(model_xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ee = m.body(f"EE_BODY_{side}").id
    R = d.xmat[ee].reshape(3, 3)
    p0 = d.xpos[ee]

    def bounds(body: str, names: Optional[set]) -> tuple[np.ndarray, np.ndarray]:
        b = m.body(body).id
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] == 0:
                continue
            mid = int(m.geom_dataid[g])
            if names is not None and m.mesh(mid).name not in names:
                continue
            a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            V = (d.geom_xmat[g].reshape(3, 3) @ m.mesh_vert[a:a + n].T).T + d.geom_xpos[g]
            V = (R.T @ (V - p0).T).T
            lo, hi = np.minimum(lo, V.min(0)), np.maximum(hi, V.max(0))
        return lo, hi

    pads = [bounds(f"ee_finger_{s}{i}", {"EE_FINGER_collision_2"}) for i in (1, 2)]
    pad = (np.minimum(pads[0][0], pads[1][0]), np.maximum(pads[0][1], pads[1][1]))
    palm = bounds(f"EE_BODY_{side}", {f"EE_BODY_collision_{i}" for i in range(7)})
    return {"pad": pad, "palm": palm}


__all__ = ["Box", "ParallelGripperSpec", "RBY1_LEFT", "RBY1_RIGHT", "spec_for", "with_wrist", "measure_from_mjcf"]
