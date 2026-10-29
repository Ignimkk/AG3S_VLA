"""Which clusters may become the **manipulated object**, and which one is the **destination** (T26).

## Why this exists

T14 seq 12–21: the apple was behind the fingers, the crate led the attention ranking at score 0.015,
grounding named it the target, and `_target_ball` built a 246 mm ball around it and deleted 15.8k
crate seeds from the field the fingers were checked against. T20 stopped the *hand-over* (a hidden
object is `occluded`, not replaced), but nothing said the crate could not be the manipulated object
in the first place — `TargetConfirm`'s `first` adopted whatever led.

A radius cap would be the wrong fix (user ruling 2026-09-28: "crate 는 어떤 상황에서도 공이 될 수
없다"; "반지름 상한은 답이 아니다"). A cap would still let the crate *be* the manipulated object and
merely limit the damage. The rule here makes that state unreachable:

    admissible  =  graspable                 narrowest principal extent  ≤  max gripper opening
                   AND not the destination   fewer than `destination_overlap_fraction` of the
                                             cluster's points lie within one voxel of the
                                             registered destination's points

`max_opening` is **computed from the robot model** (the fingers' collision geometry at the finger
joints' open limit — `robot_models.UrdfSphereChain.gripper_openings`), never written as a number.
When nobody can say what the gripper's opening is, *nothing* is graspable (fail-closed): the only
alternative is to admit anything, which is the T14 failure.

## The destination is registered from perception

The crate is a **free body** in `model_transport.xml` (`crate`, joint 31, `free`), so the static
geometry channel (`--static-geometry auto`, which drops free bodies) cannot supply it. It is
therefore registered from what the cameras see: a cluster that is **known not graspable** and leads
the attention ranking at `target_switch_min_score` or above for `target_confirm_frames` frames running
becomes the destination (`DestinationRegistry`). A caller that knows the destination injects its
points instead (`AG3S.reset(destination_points=...)`), and the injection takes precedence.

The destination is **never removed from any field** (F18's thin margin for the *held* object's
points against it is unchanged). It only takes part here, as the thing a manipulated object must
not be.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any, Optional

import numpy as np

#: Sources of `MaxOpening.value_m`.
FROM_ROBOT_MODEL, FROM_CONFIG, UNAVAILABLE = "robot_model", "config", "unavailable"
#: Sources of a `DestinationGeometry`.
INJECTED, REGISTERED = "injected", "registered"

#: `Admissibility.reason` vocabulary. `admissible` is the only one that admits.
ADMISSIBLE = "admissible"
NOT_GRASPABLE = "not_graspable"
MAX_OPENING_UNKNOWN = "max_opening_unknown"
NO_EXTENT = "no_extent"
DESTINATION_OVERLAP = "destination_overlap"


class ExclusionInvariantViolation(RuntimeError):
    """Exclusion geometry was about to be built from something with no admissibility evidence.

    Raised only in strict mode (`AG3S(strict_invariants=True)` or `AG3S_STRICT_INVARIANTS=1`, which
    the unit tests set). In production the pipeline refuses the geometry and records
    `invariant_violation` instead — admissible-only is not a limit to be enforced at the consumer,
    it is a property of what reaches it; reaching it anyway is a bug to be found, not a case to
    handle.
    """

    def __init__(self, detail: str, record: Optional[dict] = None):
        super().__init__(detail)
        self.detail = str(detail)
        self.record = dict(record or {})


# ------------------------------------------------------------------------------ geometry


def principal_extents(points: np.ndarray) -> np.ndarray:
    """`(3,)` extents of `points` along their principal axes, **ascending** (m).

    PCA axes, so the answer does not depend on how the object sits in the base frame: a banana
    lying diagonally on the table measures its real thickness, not the diagonal of its AABB. Fewer
    than two points have no extent (zeros); a degenerate set (collinear/coplanar) has zeros in the
    missing directions, which is the truth about that set.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if len(pts) < 2:
        return np.zeros(3, np.float64)
    centred = pts - pts.mean(axis=0)
    cov = centred.T @ centred / max(len(pts) - 1, 1)
    _, axes = np.linalg.eigh(cov)
    proj = centred @ axes
    return np.sort(proj.max(axis=0) - proj.min(axis=0))


@dataclasses.dataclass(frozen=True)
class MaxOpening:
    """The widest object the gripper can close around, and where that number came from.

    `value_m` is the **inner-face gap** of the finger collision geometry at the finger joints' open
    limit (not the finger-origin gap, which is larger by the fingers' own thickness). `None` means
    no source could say — every cluster is then inadmissible (`max_opening_unknown`).
    """

    value_m: Optional[float]
    source: str
    detail: dict = dataclasses.field(default_factory=dict)

    @property
    def known(self) -> bool:
        return self.value_m is not None and math.isfinite(float(self.value_m)) \
            and float(self.value_m) > 0.0

    def record(self) -> dict:
        return {
            "value_mm": None if self.value_m is None else round(float(self.value_m) * 1000.0, 2),
            "source": self.source,
            **self.detail,
        }


def max_opening_from_models(*models: Any) -> MaxOpening:
    """Ask each model, in order, for its grippers' openings; the first that answers wins.

    A model answers through `gripper_openings()` (`UrdfSphereChain` implements it from the finger
    links' spheres and the finger joints' limits). The smallest opening over its grippers is used —
    "graspable" must hold for whichever hand closes.
    """
    tried = []
    for model in models:
        if model is None:
            continue
        fn = getattr(model, "gripper_openings", None)
        name = type(model).__name__
        if not callable(fn):
            tried.append(f"{name}: no gripper_openings()")
            continue
        openings = list(fn() or ())
        if not openings:
            tried.append(f"{name}: no parallel gripper found")
            continue
        best = min(openings, key=lambda o: float(o["inner_gap_m"]))
        return MaxOpening(
            value_m=float(best["inner_gap_m"]),
            source=FROM_ROBOT_MODEL,
            detail={
                "model": name,
                "rule": "min over grippers of the finger collision geometry's inner-face gap "
                        "along the closing axis, finger joints at the open limit",
                "grippers": [_jsonable(o) for o in openings],
            })
    return MaxOpening(None, UNAVAILABLE, {"tried": tried or ["no robot model"]})


def resolve_max_opening(config_value: Optional[float], *models: Any) -> MaxOpening:
    """`clustering.gripper_max_opening` when set (offline tools with no robot model), else the
    robot model's own number, else unavailable."""
    if config_value is not None:
        return MaxOpening(float(config_value), FROM_CONFIG,
                          {"rule": "clustering.gripper_max_opening"})
    return max_opening_from_models(*models)


# ---------------------------------------------------------------------------- destination


@dataclasses.dataclass(frozen=True)
class DestinationGeometry:
    """Where the manipulated object is being taken — the crate (guide §4.1's third role)."""

    id: int
    source: str                      # `injected` | `registered`
    points: np.ndarray               # (N, 3) base frame, last observation
    registered_frame: int
    last_seen_frame: int
    n_observations: int
    score: Optional[float] = None
    extents_m: Optional[np.ndarray] = None
    #: KD-tree over `points`, built on first use (the overlap test runs per cluster per frame).
    _kdtree: Any = dataclasses.field(default=None, compare=False, repr=False)

    def tree(self):
        if self._kdtree is None:
            from scipy.spatial import cKDTree

            object.__setattr__(self, "_kdtree",
                               cKDTree(np.asarray(self.points, np.float64).reshape(-1, 3)))
        return self._kdtree

    @property
    def centroid(self) -> np.ndarray:
        return np.asarray(self.points, np.float64).reshape(-1, 3).mean(axis=0)

    @property
    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(self.points, np.float64).reshape(-1, 3)
        return pts.min(axis=0), pts.max(axis=0)

    @property
    def label(self) -> str:
        """Name for `GraspLatch.set_destination` and logs."""
        return f"destination{self.id}"

    def record(self) -> dict:
        lo, hi = self.aabb
        return {
            "id": int(self.id),
            "label": self.label,
            "source": self.source,
            "n_points": int(len(self.points)),
            "centroid": [round(float(v), 5) for v in self.centroid],
            "aabb_min": [round(float(v), 5) for v in lo],
            "aabb_max": [round(float(v), 5) for v in hi],
            "extents_mm": (None if self.extents_m is None
                           else [round(float(v) * 1000.0, 1) for v in self.extents_m]),
            "registered_frame": int(self.registered_frame),
            "last_seen_frame": int(self.last_seen_frame),
            "n_observations": int(self.n_observations),
            "score": None if self.score is None else float(self.score),
        }


def destination_overlap(points: np.ndarray, destination: Optional[DestinationGeometry],
                        distance: float) -> float:
    """Fraction of `points` within `distance` of any destination point (0.0 without one)."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if destination is None or not len(pts) or not len(destination.points):
        return 0.0
    d, _ = destination.tree().query(pts, k=1,
                                    distance_upper_bound=float(distance) * (1.0 + 1e-9))
    return float(np.mean(np.isfinite(d)))


class DestinationRegistry:
    """The destination, registered from perception or injected by the caller.

    Registration: the **attention leader** (rank 0) is *known not graspable* and scores at least
    `min_score`, for `frames` frames running on the same object. "Same object" is a centroid within
    `tolerance` of the running candidate's centroid **or inside its AABB grown by `tolerance`** — the
    crate is 0.18 × 0.30 m, and which part of it the cameras see moves its partial-view centroid by
    more than the 60 mm that separates two small objects.

    Once registered, a non-graspable cluster matching it by the same rule refreshes its geometry
    (last observation, not a union: a registration that grew forever would swallow what is put into
    it). A graspable-looking fragment of it (one crate wall seen edge-on) does not refresh it; it is
    kept out of the manipulated role by the overlap rule instead.

    An injected destination is never replaced or refreshed by perception.
    """

    def __init__(self, *, frames: int, min_score: float, tolerance: float):
        self.frames = int(frames)
        self.min_score = float(min_score)
        self.tolerance = float(tolerance)
        self.current: Optional[DestinationGeometry] = None
        self._next_id = 0
        self._candidate: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._streak = 0
        #: What the last `observe()` did — `registered` / `updated` / None.
        self.last_event: Optional[str] = None

    def reset(self, injected_points: Optional[np.ndarray] = None) -> None:
        self.current = None
        self._next_id = 0
        self._candidate = None
        self._streak = 0
        self.last_event = None
        if injected_points is not None:
            self.inject(injected_points, frame=-1)

    def inject(self, points: np.ndarray, *, frame: int) -> None:
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        if not len(pts):
            return
        prev = self.current
        self.current = DestinationGeometry(
            id=prev.id if prev is not None and prev.source == INJECTED else self._take_id(),
            source=INJECTED, points=pts, registered_frame=int(frame), last_seen_frame=int(frame),
            n_observations=1, extents_m=principal_extents(pts))
        self.last_event = "injected"

    @property
    def streak(self) -> int:
        return self._streak

    def observe(self, clusters: Sequence[Any], cluster_points: Sequence[np.ndarray],
                graspable: Sequence[Optional[bool]], *, frame: int) -> Optional[str]:
        """One frame. `clusters` score-sorted; `graspable[i]` True/False/None (unknown)."""
        self.last_event = None
        if not clusters:
            return None
        if self.current is not None:
            if self.current.source == REGISTERED:
                i = self._match(clusters, graspable, self.current)
                if i is not None:
                    self.current = dataclasses.replace(
                        self.current, points=np.asarray(cluster_points[i], np.float64),
                        _kdtree=None,
                        last_seen_frame=int(frame),
                        n_observations=self.current.n_observations + 1,
                        score=float(clusters[i].target_score),
                        extents_m=_extents_of(clusters[i], cluster_points[i]))
                    self.last_event = "updated"
            return self.last_event

        leader = clusters[0]
        if graspable[0] is False and float(leader.target_score) >= self.min_score:
            pts = np.asarray(cluster_points[0], np.float64).reshape(-1, 3)
            box = (np.asarray(leader.centroid, np.float64).reshape(3),
                   pts.min(axis=0), pts.max(axis=0))
            if self._candidate is not None and self._same(box[0], self._candidate):
                self._streak += 1
            else:
                self._streak = 1
            self._candidate = box
            if self._streak >= self.frames:
                self.current = DestinationGeometry(
                    id=self._take_id(), source=REGISTERED, points=pts,
                    registered_frame=int(frame), last_seen_frame=int(frame), n_observations=1,
                    score=float(leader.target_score), extents_m=_extents_of(leader, pts))
                self._candidate, self._streak = None, 0
                self.last_event = "registered"
        else:
            self._candidate, self._streak = None, 0
        return self.last_event

    # --- internals ---------------------------------------------------------------------
    def _take_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _same(self, centroid, box) -> bool:
        c0, lo, hi = box
        c = np.asarray(centroid, np.float64).reshape(3)
        if float(np.linalg.norm(c - c0)) <= self.tolerance:
            return True
        return bool(np.all(c >= lo - self.tolerance) and np.all(c <= hi + self.tolerance))

    def _match(self, clusters, graspable, dest: DestinationGeometry) -> Optional[int]:
        lo, hi = dest.aabb
        box = (dest.centroid, lo, hi)
        best, best_d = None, float("inf")
        for i, c in enumerate(clusters):
            if graspable[i] is not False:
                continue
            if not self._same(c.centroid, box):
                continue
            d = float(np.linalg.norm(np.asarray(c.centroid, np.float64).reshape(3) - box[0]))
            if d < best_d:
                best, best_d = i, d
        return best


def _extents_of(cluster: Any, points: np.ndarray) -> np.ndarray:
    ext = getattr(cluster, "principal_extents", None)
    return principal_extents(points) if ext is None else np.asarray(ext, np.float64)


# ---------------------------------------------------------------------------- the verdict


@dataclasses.dataclass(frozen=True)
class Admissibility:
    """Why a cluster may or may not become the manipulated object — the evidence, kept.

    `graspable` is tri-state: None when it could not be decided (no max opening, no extent), and
    only `True` admits. The pipeline re-checks `narrowest_extent_m ≤ max_opening_m` and the overlap
    before it builds any exclusion geometry (`AG3S._exclusion_gate`).
    """

    graspable: Optional[bool]
    narrowest_extent_m: Optional[float]
    extents_m: tuple[float, ...]
    max_opening_m: Optional[float]
    destination_overlap: float
    overlaps_destination: bool
    destination_id: Optional[int]
    reason: str

    @property
    def admissible(self) -> bool:
        return self.reason == ADMISSIBLE

    def record(self) -> dict:
        return {
            "admissible": bool(self.admissible),
            "reason": self.reason,
            "graspable": self.graspable,
            "narrowest_extent_mm": (None if self.narrowest_extent_m is None
                                    else round(float(self.narrowest_extent_m) * 1000.0, 2)),
            "extents_mm": [round(float(v) * 1000.0, 1) for v in self.extents_m],
            "max_opening_mm": (None if self.max_opening_m is None
                               else round(float(self.max_opening_m) * 1000.0, 2)),
            "destination_overlap": round(float(self.destination_overlap), 4),
            "overlaps_destination": bool(self.overlaps_destination),
            "destination_id": self.destination_id,
        }


def graspable(extents: Optional[np.ndarray], max_opening_m: Optional[float]) -> Optional[bool]:
    """True/False, or None when either side is unknown."""
    if max_opening_m is None or extents is None:
        return None
    ext = np.asarray(extents, np.float64).reshape(-1)
    if not ext.size or not np.all(np.isfinite(ext)):
        return None
    return bool(float(ext.min()) <= float(max_opening_m))


def assess(points: np.ndarray, *, extents: Optional[np.ndarray], max_opening_m: Optional[float],
           destination: Optional[DestinationGeometry], overlap_distance: float,
           overlap_fraction: float) -> Admissibility:
    """The admissibility of one cluster (or one observed geometry)."""
    ext = principal_extents(points) if extents is None else np.asarray(extents, np.float64)
    g = graspable(ext, max_opening_m)
    narrow = float(ext.min()) if ext.size and len(np.asarray(points).reshape(-1, 3)) >= 2 else None
    overlap = destination_overlap(points, destination, overlap_distance)
    overlaps = bool(destination is not None and overlap >= float(overlap_fraction))
    if max_opening_m is None:
        reason = MAX_OPENING_UNKNOWN
    elif narrow is None or g is None:
        reason = NO_EXTENT
    elif not g:
        reason = NOT_GRASPABLE
    elif overlaps:
        reason = DESTINATION_OVERLAP
    else:
        reason = ADMISSIBLE
    return Admissibility(
        graspable=g, narrowest_extent_m=narrow,
        extents_m=tuple(float(v) for v in ext.reshape(-1)),
        max_opening_m=None if max_opening_m is None else float(max_opening_m),
        destination_overlap=float(overlap), overlaps_destination=overlaps,
        destination_id=None if destination is None else int(destination.id), reason=reason)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


__all__ = [
    "ADMISSIBLE",
    "Admissibility",
    "DESTINATION_OVERLAP",
    "DestinationGeometry",
    "DestinationRegistry",
    "ExclusionInvariantViolation",
    "FROM_CONFIG",
    "FROM_ROBOT_MODEL",
    "INJECTED",
    "MAX_OPENING_UNKNOWN",
    "MaxOpening",
    "NOT_GRASPABLE",
    "NO_EXTENT",
    "REGISTERED",
    "UNAVAILABLE",
    "assess",
    "destination_overlap",
    "graspable",
    "max_opening_from_models",
    "principal_extents",
    "resolve_max_opening",
]
