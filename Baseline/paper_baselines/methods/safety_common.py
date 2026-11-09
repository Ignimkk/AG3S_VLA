"""Shared pieces of the safety-filter methods (PAPER-METHODS C): frames, chunk roll-out, obstacle point clouds,
and the recorded-frame loader used by the smoke.

Frames: every 3D quantity is in the **robot base frame** (``SceneObservation.T_base_cam`` / ``ee_pose``). LIBERO
actions are robosuite ``OSC_POSE`` deltas (``control_delta=True``): ``a[:3]`` translation and ``a[3:6]`` axis-angle,
both in the base (= world, rotation-wise) frame, ``a[6]`` gripper (-1 open, +1 close), in controller units ``[-1, 1]``.

Chunk roll-out: a filter that receives a whole chunk (T > 1) has to predict the EE pose before each later action.
:func:`propagate_ee` does this with the **measured** EE displacement per action unit per env step
(:data:`POS_GAIN_M`, 5 SL3-dev episodes: 0.0112-0.0121 m, R^2 0.89-0.95; robosuite's nominal ``output_max`` 0.05 m is
4x larger because the OSC does not reach its set-point within one step). Called with T = 1 per env step (closed
loop), no prediction is involved and the filter sees the measured pose every step, as the original methods do.

Obstacle point clouds (:class:`ObjectClouds`) come from one of three sources, chosen per method:

* ``given``     -- ``extras["obstacle_points"] = {name: (N, 3) base points}`` from an upstream module (not privileged);
* ``gt_seg``    -- simulator instance segmentation (privileged): ``extras["privileged/segmentation_geom"]`` +
  ``extras["privileged/instances_geom_ids"]``; obstacle set (``gt_obstacle_set``) = every instance except the excluded
  target / destination / support (``"all_non_target"``, default) or only those in ``extras["privileged/obstacle_names"]``
  (``"obstacle_names"``: SafeLIBERO's designated obstacle, the original papers' condition; sensitivity variant);
* ``detector``  -- Grounding DINO boxes (A's ``providers/grounding_dino.py``) cropped on the depth image, the
  AEGIS perception (``benchmark/SafeLIBERO/main/utils.py:155-255``). Prompts: SafeLIBERO obstacle names
  (privileged metadata, replacing the VLM) or a generic object prompt with the target box removed.

Detector crop (``crop_mode``, BASELINE-BENCH-READY):

* ``"box"`` -- every valid-depth pixel of the box (``get_point_cloud`` utils.py:188-255), robot spheres removed;
  one cloud per name, the clouds of every camera with the same name concatenated. This is the CLOSED-LOOP-FIX V5
  behaviour (``crop_mode="box"`` reproduces it bit for bit).
* ``"depth_trim"`` (default) -- the box crop keeps only the object at the box centre, i.e. what the original
  pipeline leaves after its own cleanup. In AEGIS that cleanup is ``filtering_points`` (utils.py:257-316): a fixed
  world workspace box (table scenes: z 0.92-1.5 m, |x|, |y| < 0.3 m) that cuts the table top and the background, the
  80 % nearest-centroid trim, and DBSCAN largest-cluster (``eps=0.0001`` m, in practice a no-op). The workspace box
  needs the suite name (not on the wire), so here, per box: (1) robot spheres removed (as ``box``) and the Panda base (link0 + mount, ``base_envelope_m`` = r_xy 0.18 m,
  z < 0.16 m in the base frame; the sphere model starts at z 0.085 m), (2) support
  plane removed -- points below the densest horizontal plane of the external camera's lift + ``support_margin_m``
  (non-privileged estimate, ``guidance_common.scene.estimate_table_z``, the same rule as the guidance arms) and
  points higher than ``max_height_m`` above it (the box's 0.58 m above the table top), (3) 26-connected components on
  a ``cluster_voxel_m`` grid, the component holding most pixels of the central ``center_frac`` window of the box is
  kept (ties / empty centre: nearest the box centre), (4) KNOWS ``_trim_by_depth`` (camera-range percentiles 2-98).
  Then the per-box clouds of all cameras are **associated in 3D** before merging: two clouds are one object when
  their centroids are within ``assoc_gate_m`` or ``assoc_overlap`` of the smaller cloud lies within
  ``cluster_voxel_m`` of the other (``obstacle_names``: same name required; a same-name cloud that does not match is
  kept as ``<name>@<camera>``). ``all_objects`` names are per-camera ranks (``object#j``), so the old name merge
  joined unrelated head / wrist objects; here they are re-numbered per associated group.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any

import numpy as np

from Baseline.paper_baselines.methods.interface import PRIVILEGED_PREFIX, SceneObservation

#: robosuite OSC_POSE ``output_max`` (controller units -> metres / radians of the *set-point* delta).
OSC_POS_SCALE = 0.05
OSC_ROT_SCALE = 0.5
#: Measured EE displacement per action unit per env step (SL3 dev 20261012T163617, 5 episodes, least squares).
POS_GAIN_M = 0.0115
#: Rotation analogue, assumed with the same set-point ratio (0.0115 / 0.05) x 0.5 rad -- not measured.
ROT_GAIN_RAD = 0.115
#: LIBERO fixtures that are support surfaces, never an obstacle ellipsoid (KNOWS docs/11: a table ellipsoid is metres).
SUPPORT_NAMES = ("main_table", "kitchen_table", "study_table", "living_room_table", "coffee_table", "floor", "table")
#: Base-frame z below which points are support (LIBERO tables sit at base z ~ 0); used only by ``detector`` crops.
CAMERAS_EXTERNAL = ("head",)


# --- small SO(3) helpers ---------------------------------------------------------------------------------------
def skew(v) -> np.ndarray:
    v = np.asarray(v, np.float64).reshape(3)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def axisangle_to_mat(v) -> np.ndarray:
    v = np.asarray(v, np.float64).reshape(3)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    K = skew(v / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def mat_to_axisangle(R) -> np.ndarray:
    R = np.asarray(R, np.float64)
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-9:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - th < 1e-6:
        w, V = np.linalg.eigh(0.5 * (R + R.T))
        return V[:, int(np.argmax(w))] * th
    return th / (2.0 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def quat_xyzw_to_mat(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, np.float64).reshape(4) / max(np.linalg.norm(q), 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def propagate_ee(ee_pose: np.ndarray, action: np.ndarray, pos_gain: float = POS_GAIN_M,
                 rot_gain: float = ROT_GAIN_RAD) -> np.ndarray:
    """Predicted EE pose (4x4, base) one env step after ``action`` (LIBERO OSC delta, world-frame rotation)."""
    T = np.array(ee_pose, np.float64, copy=True)
    a = np.asarray(action, np.float64).reshape(-1)
    T[:3, 3] = T[:3, 3] + pos_gain * a[:3]
    T[:3, :3] = axisangle_to_mat(rot_gain * a[3:6]) @ T[:3, :3]
    return T


def check_actions(actions: np.ndarray) -> np.ndarray:
    a = np.asarray(actions, np.float64)
    if a.ndim != 2 or a.shape[1] < 7:
        raise ValueError(f"actions must be (T, >=7) LIBERO EE-delta actions, got {a.shape}")
    return a.copy()


# --- point clouds ----------------------------------------------------------------------------------------------
def backproject(depth_m: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, pixel_mask: np.ndarray | None = None,
                *, range_max: float = 2.0) -> np.ndarray:
    """Pinhole back-projection of the (masked) valid-depth pixels -> base points (N, 3). OpenCV frame, no flip."""
    d = np.asarray(depth_m, np.float64)
    m = np.isfinite(d) & (d > 1e-6) & (d < range_max)
    if pixel_mask is not None:
        m &= np.asarray(pixel_mask, bool)
    v, u = np.nonzero(m)
    z = d[v, u]
    K = np.asarray(K, np.float64)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    T = np.asarray(T_base_cam, np.float64)
    return pc @ T[:3, :3].T + T[:3, 3]


def depth_trim(points: np.ndarray, cam_pos: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Drop range outliers of one object's pixels (KNOWS ``_trim_by_depth``: silhouette bleed stretches MVEE)."""
    if len(points) < 10:
        return points
    r = np.linalg.norm(points - cam_pos, axis=1)
    a, b = np.percentile(r, [lo, hi])
    return points[(r >= a) & (r <= b)]


def backproject_uv(depth_m: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray, pixel_mask: np.ndarray | None = None,
                   *, range_max: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`backproject` that also returns the pixel coordinates ``(points (N, 3), u (N,), v (N,))``."""
    d = np.asarray(depth_m, np.float64)
    m = np.isfinite(d) & (d > 1e-6) & (d < range_max)
    if pixel_mask is not None:
        m &= np.asarray(pixel_mask, bool)
    v, u = np.nonzero(m)
    return backproject(d, K, T_base_cam, m, range_max=range_max), u, v


def voxel_components(points: np.ndarray, voxel: float) -> tuple[np.ndarray, int]:
    """26-connected components of the occupied ``voxel`` grid -> ``(label per point (N,) in 0..n-1, n)``."""
    from scipy import ndimage

    p = np.asarray(points, np.float64).reshape(-1, 3)
    if not len(p):
        return np.zeros(0, np.int64), 0
    k = np.floor(p / float(voxel)).astype(np.int64)
    k -= k.min(0)
    grid = np.zeros(tuple(k.max(0) + 1), bool)
    grid[tuple(k.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
    return lab[tuple(k.T)].astype(np.int64) - 1, int(n)


def center_component(points: np.ndarray, u: np.ndarray, v: np.ndarray, box, *, voxel: float = 0.01,
                     center_frac: float = 0.4) -> tuple[np.ndarray, dict]:
    """Keep the connected component (``voxel`` grid) of the object at the centre of ``box`` (x0, y0, x1, y1).

    The component with the most points among the pixels of the central ``center_frac`` window wins; when no point
    lies in that window, the component whose nearest pixel is closest to the box centre. Returns ``(keep (N,) bool,
    info)``.
    """
    n_pts = len(points)
    info = {"n_components": 0, "chosen": None, "n_center": 0}
    if not n_pts:
        return np.zeros(0, bool), info
    lab, n = voxel_components(points, voxel)
    info["n_components"] = n
    x0, y0, x1, y1 = (float(b) for b in box)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    hw, hh = 0.5 * center_frac * (x1 - x0), 0.5 * center_frac * (y1 - y0)
    win = (np.abs(u + 0.5 - cx) <= max(hw, 0.5)) & (np.abs(v + 0.5 - cy) <= max(hh, 0.5))
    info["n_center"] = int(win.sum())
    if win.any():
        votes = np.bincount(lab[win], minlength=n)
        best = int(np.argmax(votes))
        info["center_votes"] = int(votes[best])
    else:
        dpx = np.hypot(u + 0.5 - cx, v + 0.5 - cy)
        best = int(lab[int(np.argmin(dpx))])
    info["chosen"] = best
    keep = lab == best
    info["n_kept"] = int(keep.sum())
    return keep, info


def cloud_overlap(a: np.ndarray, b: np.ndarray, radius: float) -> float:
    """Fraction of the smaller cloud's points within ``radius`` of the other cloud."""
    from scipy.spatial import cKDTree

    small, big = (a, b) if len(a) <= len(b) else (b, a)
    if not len(small):
        return 0.0
    d, _ = cKDTree(big).query(small, k=1, distance_upper_bound=radius)
    return float(np.mean(np.isfinite(d)))


#: AEGIS official obstacle wording (2026-11-09, user-approved fidelity fix): the official code asks GLM-4.5V to name the
#: obstacle "preferably from this list" (``benchmark/SafeLIBERO/main/utils.py:466`` ``prefer_list``), so a perfect VLM
#: answer for a SafeLIBERO obstacle is the list entry, not the display name. Keyed by ``phrase_from_object`` of the display
#: name, ``small`` variants share the entry. Audit ``outputs/2026-11-09_AEGIS_port_점검`` (Grounding DINO top-1 hits on the
#: recorded call-0 frames 45 -> 89 / 111 with this wording).
AEGIS_PREFER_LIST_WORDING = {"yellow book": "yellow rectangular book", "moka pot": "blue moka pot",
                             "red coffee mug": "red mug", "white storage box": "white storage box",
                             "wine bottle": "black wine bottle", "milk": "red milk carton",
                             "binder": "gray rectangular binder"}
NAME_WORDINGS = ("display", "aegis_prefer_list")
#: the official GLM-4.5V candidate list in its order (utils.py:466-468; the binder only when "long" is in the suite name)
AEGIS_PREFER_LIST = ("yellow rectangular book", "blue moka pot", "red mug", "white storage box", "black wine bottle",
                     "red milk carton")
AEGIS_PREFER_LIST_LONG = "gray rectangular binder"


def aegis_prefer_list(suite: str | None) -> list[str]:
    """``prefer_list`` of the official ``obstacle_detection`` for a task suite name (non-privileged: fixed per suite)."""
    return list(AEGIS_PREFER_LIST) + ([AEGIS_PREFER_LIST_LONG] if "long" in str(suite or "") else [])


def obstacle_query_phrase(name: str, wording: str = "display") -> str:
    """Detector prompt for a privileged obstacle name: ``display`` = :func:`phrase_from_object`; ``aegis_prefer_list``
    = the official AEGIS ``prefer_list`` entry for that object (falls back to the display phrase if not listed)."""
    p = phrase_from_object(name)
    if wording == "aegis_prefer_list":
        base = p[: -len(" small")] if p.endswith(" small") else p
        return AEGIS_PREFER_LIST_WORDING.get(base, p)
    if wording != "display":
        raise ValueError(f"wording must be one of {NAME_WORDINGS}, got {wording!r}")
    return p


def phrase_from_object(name: str) -> str:
    """``akita_black_bowl_1`` -> ``akita black bowl`` (same rule as ``providers/gt_oracle.phrase_from_object``)."""
    parts = str(name).split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    return " ".join(parts)


def box_iou(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


@dataclasses.dataclass
class ObjectClouds:
    """Obstacle point sets for one observation: ``{name: (N, 3) base points}`` + how they were obtained."""

    points: dict[str, np.ndarray]
    source: str
    privileged: bool
    excluded: list[str]
    info: dict


class ObstacleCloudSource:
    """Produces :class:`ObjectClouds` from a :class:`SceneObservation` (see module docstring for the sources).

    ``exclude`` (``gt_seg``; candidates = every scene instance): ``"task_objects"`` (default) excludes the BDDL target
    and destination, ``"phase_target"`` the target before grasp and the destination after (``extras["gripper_closed"]``;
    KNOWS offline_filter rule -- the held target then becomes an obstacle), ``"target"`` only the target, ``"none"``
    nothing. Support surfaces (:data:`SUPPORT_NAMES`) are never obstacles; ``include_fixtures`` keeps cabinets / stoves.

    ``detector`` prompts (``detector_prompt``): ``"obstacle_names"`` = ``privileged/obstacle_names`` (SafeLIBERO
    metadata; the VLM replacement (a)), ``"all_objects"`` = ``generic_prompt`` with every box overlapping the
    target box (IoU > ``target_iou``) removed (replacement (b); not privileged), ``"candidate_list"`` = replacement (c),
    2026-11-10: the official GLM-4.5V step ("identify exactly one non-robot obstacle ... preferably from prefer_list") done
    by the detector -- every :func:`aegis_prefer_list` phrase is prompted, the top-1 valid box of each (table-sized and
    target-box overlaps removed as in (b)), and the single phrase with the highest score is the episode's one obstacle,
    then cropped exactly like (a) (not privileged: the list is the official prompt's, fixed per suite);
    ``"vlm_name"`` = replacement (d), 2026-11-10: the official GLM-4.5V prompt answered by a local open VLM
    (``providers.vlm_namer.QwenVLNamer``) on the first external camera's image; its answer is the one Grounding DINO
    phrase, cropped like (a) -- the official pipeline with the API model swapped (not privileged).
    """

    SOURCES = ("given", "gt_seg", "detector")
    #: detector crop: ``box`` = CLOSED-LOOP-FIX V5 (whole box), ``depth_trim`` = box-centre object only (default)
    CROP_MODES = ("box", "depth_trim")
    EXCLUDE = ("task_objects", "phase_target", "target", "none")
    #: ``gt_seg`` obstacle set: every non-target instance (default) | only ``privileged/obstacle_names`` (SafeLIBERO's
    #: designated obstacle(s): the condition of the original papers; a sensitivity variant here)
    GT_OBSTACLE_SETS = ("all_non_target", "obstacle_names")

    def __init__(self, source: str = "gt_seg", *, exclude: str = "task_objects", include_fixtures: bool = True,
                 cameras: tuple[str, ...] | None = None, min_points: int = 30, max_points: int = 4000,
                 detector: Any = None, detector_weight: str = "grounding-dino-tiny", detector_device: str = "cpu",
                 detector_prompt: str = "obstacle_names", generic_prompt: str = "object",
                 box_threshold: float = 0.35, text_threshold: float = 0.25, target_iou: float = 0.5,
                 max_box_frac: float = 0.5, max_obstacles: int = 8, robot_filter: bool = True,
                 robot_margin: float = 0.02, seed: int = 0, crop_mode: str = "depth_trim",
                 center_frac: float = 0.4, cluster_voxel_m: float = 0.01, support_margin_m: float = 0.015,
                 max_height_m: float = 0.58, trim_percentiles: tuple[float, float] = (2.0, 98.0),
                 assoc_gate_m: float = 0.05, assoc_overlap: float = 0.5,
                 base_envelope_m: tuple[float, float] | None = (0.18, 0.16),
                 gt_obstacle_set: str = "all_non_target", name_wording: str = "display", target_box: bool = True,
                 vlm: Any = None, vlm_weight: str | None = None, vlm_device: str = "cuda"):
        if name_wording not in NAME_WORDINGS:
            raise ValueError(f"name_wording must be one of {NAME_WORDINGS}, got {name_wording!r}")
        self.name_wording = name_wording
        #: ``all_objects`` prompt: drop boxes overlapping the target phrase's top-1 box (default, replacement (b)).
        #: ``False`` (KNOWS F4a, 2026-11-10): keep every box -- the target is excluded upstream by attention (Eq. 2-4).
        self.target_box = bool(target_box)
        self._vlm, self.vlm_weight, self.vlm_device = vlm, vlm_weight, vlm_device  # detector_prompt="vlm_name" only
        if source not in self.SOURCES:
            raise ValueError(f"source must be one of {self.SOURCES}, got {source!r}")
        if exclude not in self.EXCLUDE:
            raise ValueError(f"exclude must be one of {self.EXCLUDE}, got {exclude!r}")
        if detector_prompt not in ("obstacle_names", "all_objects", "candidate_list", "vlm_name"):
            raise ValueError("detector_prompt must be obstacle_names|all_objects|candidate_list|vlm_name, "
                             f"got {detector_prompt!r}")
        if gt_obstacle_set not in self.GT_OBSTACLE_SETS:
            raise ValueError(f"gt_obstacle_set must be one of {self.GT_OBSTACLE_SETS}, got {gt_obstacle_set!r}")
        self.gt_obstacle_set = gt_obstacle_set
        if crop_mode not in self.CROP_MODES:
            raise ValueError(f"crop_mode must be one of {self.CROP_MODES}, got {crop_mode!r}")
        self.source, self.exclude, self.include_fixtures = source, exclude, include_fixtures
        self.cameras, self.min_points, self.max_points = cameras, int(min_points), int(max_points)
        self._detector, self.detector_weight, self.detector_device = detector, detector_weight, detector_device
        self.detector_prompt, self.generic_prompt = detector_prompt, generic_prompt
        self.box_threshold, self.text_threshold, self.target_iou = box_threshold, text_threshold, target_iou
        self.max_box_frac, self.max_obstacles = max_box_frac, int(max_obstacles)
        self.robot_filter, self.robot_margin = bool(robot_filter), float(robot_margin)
        self._robot_model = None
        self.rng = np.random.default_rng(seed)
        self.crop_mode, self.center_frac, self.cluster_voxel_m = crop_mode, float(center_frac), float(cluster_voxel_m)
        self.support_margin_m, self.max_height_m = float(support_margin_m), float(max_height_m)
        self.trim_percentiles = (float(trim_percentiles[0]), float(trim_percentiles[1]))
        self.assoc_gate_m, self.assoc_overlap = float(assoc_gate_m), float(assoc_overlap)
        self.base_envelope_m = None if base_envelope_m is None else (float(base_envelope_m[0]), float(base_envelope_m[1]))

    # provenance --------------------------------------------------------------------------------------------
    @property
    def privileged(self) -> bool:
        return self.source == "gt_seg" or (self.source == "detector" and self.detector_prompt == "obstacle_names")

    def provenance(self) -> dict:
        p = {"obstacle_source": self.source, "privileged": self.privileged, "cameras": self.cameras,
             "min_points": self.min_points, "max_points": self.max_points}
        if self.source == "gt_seg":
            p.update(exclude=self.exclude, include_fixtures=self.include_fixtures, support_names=list(SUPPORT_NAMES),
                     gt_obstacle_set=self.gt_obstacle_set,
                     privileged_inputs=["segmentation_geom", "instances_geom_ids", "target", "destination"]
                     + (["obstacle_names"] if self.gt_obstacle_set == "obstacle_names" else []))
        if self.source == "detector":
            det = self._detector
            p.update(detector=(det.provenance() if det is not None and hasattr(det, "provenance")
                               else {"weight": self.detector_weight, "device": self.detector_device, "loaded": False}),
                     detector_prompt=self.detector_prompt, generic_prompt=self.generic_prompt,
                     name_wording=self.name_wording,
                     **({"vlm": (self._vlm.provenance() if self._vlm is not None and hasattr(self._vlm, "provenance")
                                 else {"weight": self.vlm_weight, "device": self.vlm_device, "loaded": False})}
                        if self.detector_prompt == "vlm_name" else {}),
                     box_threshold=self.box_threshold, text_threshold=self.text_threshold, target_iou=self.target_iou,
                     crop_mode=self.crop_mode,
                     crop=("axis-aligned box on depth (AEGIS utils.py:188-255), not a mask; = CLOSED-LOOP-FIX V5"
                           if self.crop_mode == "box" else
                           "box crop -> robot spheres -> support plane / height cap (estimated table, stands in for "
                           "AEGIS filtering_points' world workspace box) -> box-centre connected component -> range "
                           "percentile trim; per-box clouds associated in 3D across boxes / cameras"),
                     depth_trim=(None if self.crop_mode == "box" else
                                 {"center_frac": self.center_frac, "cluster_voxel_m": self.cluster_voxel_m,
                                  "support_margin_m": self.support_margin_m, "max_height_m": self.max_height_m,
                                  "support_estimate": "guidance_common.scene.estimate_table_z of the external camera lift",
                                  "trim_percentiles": list(self.trim_percentiles), "assoc_gate_m": self.assoc_gate_m,
                                  "assoc_overlap": self.assoc_overlap,
                                  "robot_base_envelope_m": (None if self.base_envelope_m is None else
                                                            {"r_xy": self.base_envelope_m[0],
                                                             "z_max": self.base_envelope_m[1]})}),
                     robot_filter=(f"drop crop points within sphere radius + {self.robot_margin} m of the Panda "
                                   "collision spheres at q (AG3S/adapters/panda model; not in AEGIS)"
                                   if self.robot_filter else None),
                     privileged_inputs=["obstacle_names"] if self.detector_prompt == "obstacle_names" else [])
            if not self.target_box:  # key only off the default (default provenance unchanged)
                p["target_box"] = False
        return p

    @property
    def vlm(self):
        if self._vlm is None:
            from Baseline.paper_baselines.providers.vlm_namer import DEFAULT_VLM, QwenVLNamer

            self._vlm = QwenVLNamer(self.vlm_weight or DEFAULT_VLM, device=self.vlm_device)
        return self._vlm

    @property
    def detector(self):
        if self._detector is None:
            from Baseline.paper_baselines.providers.grounding_dino import GroundingDinoDetector

            self._detector = GroundingDinoDetector(self.detector_weight, device=self.detector_device)
        return self._detector

    # main ----------------------------------------------------------------------------------------------------
    def __call__(self, obs: SceneObservation) -> ObjectClouds:
        if self.source == "given":
            pts = {str(k): np.asarray(v, np.float64).reshape(-1, 3) for k, v in obs.extras.get("obstacle_points", {}).items()}
            return ObjectClouds(pts, "given", False, [], {})
        if self.source == "gt_seg":
            return self._gt_seg(obs)
        return self._detect(obs)

    def _cams(self, obs: SceneObservation, default) -> list[str]:
        cams = self.cameras if self.cameras is not None else default
        return [c for c in cams if c in obs.depth]

    def _subsample(self, p: np.ndarray) -> np.ndarray:
        if len(p) > self.max_points:
            p = p[self.rng.choice(len(p), self.max_points, replace=False)]
        return p

    def excluded_names(self, obs: SceneObservation) -> list[str]:
        tgt, dst = obs.privileged("target"), obs.privileged("destination")
        if self.exclude == "none":
            return []
        if self.exclude == "task_objects":
            return [n for n in (tgt, dst) if n]
        if self.exclude == "target" or not bool(obs.extras.get("gripper_closed", False)):
            return [tgt] if tgt else []
        return [dst] if dst else ([tgt] if tgt else [])

    def _gt_seg(self, obs: SceneObservation) -> ObjectClouds:
        seg = obs.privileged("segmentation_geom")
        inst = obs.privileged("instances_geom_ids")
        if seg is None or inst is None:
            raise KeyError("gt_seg needs extras['privileged/segmentation_geom'] and ['privileged/instances_geom_ids']")
        excluded = self.excluded_names(obs)
        fixtures = set(obs.privileged("fixture_names", []) or [])
        names = [n for n in inst if n not in excluded and n not in SUPPORT_NAMES
                 and (self.include_fixtures or n not in fixtures)]
        if self.gt_obstacle_set == "obstacle_names":
            allowed = obs.privileged("obstacle_names")
            if allowed is None:
                raise KeyError("gt_obstacle_set='obstacle_names' needs extras['privileged/obstacle_names']")
            allowed = {str(a) for a in allowed}
            names = [n for n in names if n in allowed]
        out: dict[str, np.ndarray] = {}
        for n in names:
            ids = np.asarray(list(inst[n]), np.int64)
            chunks = []
            for cam in self._cams(obs, list(obs.depth)):
                m = np.isin(np.asarray(seg[cam]), ids)
                if m.any():
                    p = backproject(obs.depth[cam], obs.intrinsics[cam], obs.T_base_cam[cam], m)
                    chunks.append(depth_trim(p, np.asarray(obs.T_base_cam[cam])[:3, 3]))
            p = np.concatenate(chunks) if chunks else np.zeros((0, 3))
            if len(p) >= self.min_points:
                out[n] = self._subsample(p)
        return ObjectClouds(out, "gt_seg", True, excluded, {"candidates": names})

    def robot_keep_mask(self, obs: SceneObservation, pts: np.ndarray) -> np.ndarray:
        """``(N,)`` bool: points outside the Panda collision spheres (+ margin) at ``obs.q`` -- kinematics, not GT."""
        if not len(pts):
            return np.ones(0, bool)
        if self._robot_model is None:
            from AG3S.adapters.panda import model as panda

            self._panda = panda
            self._robot_model = panda.build_panda_bare_model()
        rs = obs.extras.get("robot_state")
        if rs is not None and len(np.asarray(rs).reshape(-1)) == 9:
            q = self._panda.set_finger_parameters(self._robot_model, rs)
        else:
            q = np.asarray(obs.q, np.float64).reshape(-1)[:7]
        C, Rr = self._robot_model.sphere_centers_numeric(q)
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        drop = np.zeros(len(pts), bool)
        for c, r in zip(C, Rr):
            drop[tree.query_ball_point(c, r + self.robot_margin)] = True
        return ~drop

    def remove_robot(self, obs: SceneObservation, pts: np.ndarray) -> np.ndarray:
        """Drop points inside the Panda collision spheres (+ margin) at ``obs.q`` -- kinematics, not GT."""
        if not len(pts):
            return pts
        return pts[self.robot_keep_mask(obs, pts)]

    # --- depth_trim crop ----------------------------------------------------------------------------------------
    def support_z(self, obs: SceneObservation, cams: list[str]) -> tuple[float | None, str | None]:
        """Non-privileged support-plane height: densest horizontal plane of the first external camera's lift."""
        from Baseline.paper_baselines.methods.guidance_common.scene import estimate_table_z

        order = [c for c in CAMERAS_EXTERNAL if c in obs.depth] + [c for c in cams if c not in CAMERAS_EXTERNAL]
        for cam in order:
            if cam not in obs.intrinsics or cam not in obs.T_base_cam:
                continue
            p = backproject(obs.depth[cam], obs.intrinsics[cam], obs.T_base_cam[cam])
            z = estimate_table_z(p)
            if z is not None:
                return float(z), cam
        return None, None

    def trim_box(self, obs: SceneObservation, cam: str, box, table_z: float | None) -> tuple[np.ndarray, dict]:
        """``depth_trim`` crop of one box (module docstring steps 1-4) -> ``(points (M, 3), info)``."""
        d = obs.depth[cam]
        H, W = np.asarray(d).shape[:2]
        x0, y0, x1, y1 = np.round(np.asarray(box, float)).astype(int)
        m = np.zeros((H, W), bool)
        m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
        p, u, v = backproject_uv(d, obs.intrinsics[cam], obs.T_base_cam[cam], m)
        info = {"n_box": int(len(p))}
        keep = np.ones(len(p), bool)
        if self.robot_filter and len(p):
            keep &= self.robot_keep_mask(obs, p)
        if self.robot_filter and self.base_envelope_m is not None and len(p):
            # Panda link0 + mount: the sphere model starts at z 0.085 (r 0.03) and leaves the base plate visible
            r0, z0 = self.base_envelope_m
            keep &= ~((np.hypot(p[:, 0], p[:, 1]) < r0) & (p[:, 2] < z0))
        info["n_robot"] = int(len(p) - keep.sum())
        if table_z is not None and len(p):
            sup = (p[:, 2] <= table_z + self.support_margin_m) | (p[:, 2] > table_z + self.max_height_m)
            info["n_support_or_high"] = int((sup & keep).sum())
            keep &= ~sup
        p, u, v = p[keep], u[keep], v[keep]
        if not len(p):
            info["n_out"] = 0
            return p, info
        sel, cinfo = center_component(p, u, v, box, voxel=self.cluster_voxel_m, center_frac=self.center_frac)
        info.update(cinfo)
        p = p[sel]
        p = depth_trim(p, np.asarray(obs.T_base_cam[cam], np.float64)[:3, 3], *self.trim_percentiles)
        info["n_out"] = int(len(p))
        return p, info

    def associate(self, entries: list[dict]) -> tuple[dict[str, np.ndarray], list[dict]]:
        """Group per-box clouds that are one object in 3D (module docstring); ``entries`` in camera / score order."""
        n = len(entries)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        cents = [e["points"].mean(0) for e in entries]
        links = []
        for i in range(n):
            for j in range(i + 1, n):
                if self.detector_prompt in ("obstacle_names", "candidate_list", "vlm_name") and entries[i]["key"] != entries[j]["key"]:
                    continue
                if find(i) == find(j):
                    continue
                dc = float(np.linalg.norm(cents[i] - cents[j]))
                ov = cloud_overlap(entries[i]["points"], entries[j]["points"], self.cluster_voxel_m)
                if dc <= self.assoc_gate_m or ov >= self.assoc_overlap:
                    parent[find(j)] = find(i)
                    links.append({"a": entries[i]["label"], "b": entries[j]["label"], "centroid_dist_m": round(dc, 4),
                                  "overlap": round(ov, 3)})
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        out: dict[str, np.ndarray] = {}
        used_keys: set = set()
        k_all = 0
        for g in sorted(groups.values(), key=lambda m: m[0]):  # order of the first member (camera, score)
            e0 = entries[g[0]]
            if self.detector_prompt in ("obstacle_names", "candidate_list", "vlm_name"):
                name = e0["key"] if e0["key"] not in used_keys else f"{e0['key']}@{e0['cam']}"
                used_keys.add(e0["key"])
            else:
                name = f"{e0['phrase'] or 'object'}#{k_all}"
                k_all += 1
            out[name] = np.concatenate([entries[i]["points"] for i in g])
            for i in g:
                entries[i]["name"] = name
        return out, links

    def _choose_candidate(self, obs: SceneObservation, cams) -> tuple[str | None, dict]:
        """``candidate_list``: the one prefer_list phrase GLM would answer -> highest top-1 valid box score over cameras."""
        det = self.detector
        cands = aegis_prefer_list(obs.extras.get("suite"))
        best: dict[str, float] = {c: 0.0 for c in cands}
        for cam in cams:
            rgb = obs.rgb[cam]
            H, W = rgb.shape[:2]
            tbox = None
            if obs.target_phrase and self.target_box:
                b, s, _ = det.detect(rgb, obs.target_phrase, self.box_threshold, self.text_threshold)
                if len(s):
                    tbox = b[int(np.argmax(s))]
            for c in cands:
                boxes, scores, _ = det.detect(rgb, c, self.box_threshold, self.text_threshold)
                for i in (np.argsort(-np.asarray(scores)) if len(scores) else []):
                    bx = boxes[i]
                    if (bx[2] - bx[0]) * (bx[3] - bx[1]) > self.max_box_frac * H * W:
                        continue
                    if tbox is not None and box_iou(bx, tbox) > self.target_iou:
                        continue
                    best[c] = max(best[c], float(scores[i]))
                    break
        chosen = max(cands, key=lambda c: best[c]) if any(v > 0 for v in best.values()) else None
        return chosen, {"candidates": cands, "scores": {c: round(v, 4) for c, v in best.items()}, "chosen": chosen}

    def _detect(self, obs: SceneObservation) -> ObjectClouds:
        det = self.detector
        info: dict = {"boxes": {}, "crop_mode": self.crop_mode}
        out: dict[str, np.ndarray] = {}
        trim = self.crop_mode == "depth_trim"
        entries: list[dict] = []
        if self.detector_prompt == "obstacle_names":
            names = obs.privileged("obstacle_names")
            if names is None:
                raise KeyError("detector_prompt='obstacle_names' needs extras['privileged/obstacle_names']")
            queries = [(str(n), obstacle_query_phrase(n, self.name_wording)) for n in names]
        else:
            queries = [("object", self.generic_prompt)]
        cams = self._cams(obs, CAMERAS_EXTERNAL)
        if self.detector_prompt == "candidate_list":
            phrase, info["candidate_list"] = self._choose_candidate(obs, cams)
            queries = [(phrase, phrase)] if phrase is not None else []
        elif self.detector_prompt == "vlm_name":
            name, vinfo = self.vlm.name(obs.rgb[cams[0]], obs.instruction, aegis_prefer_list(obs.extras.get("suite")))
            info["vlm_name"] = {**vinfo, "camera": cams[0]}
            queries = [(name, name)] if name else []
        table_z = None
        if trim:
            table_z, zcam = self.support_z(obs, cams)
            info["support_z"] = None if table_z is None else round(table_z, 4)
            info["support_cam"] = zcam
        for cam in cams:
            rgb = obs.rgb[cam]
            H, W = rgb.shape[:2]
            tbox = None
            if self.detector_prompt in ("all_objects", "candidate_list") and obs.target_phrase and self.target_box:
                b, s, _ = det.detect(rgb, obs.target_phrase, self.box_threshold, self.text_threshold)
                if len(s):
                    tbox = b[int(np.argmax(s))]
            for key, phrase in queries:
                boxes, scores, phrases = det.detect(rgb, phrase, self.box_threshold, self.text_threshold)
                order = np.argsort(-np.asarray(scores)) if len(scores) else []
                kept = []
                for i in order:
                    bx = boxes[i]
                    if (bx[2] - bx[0]) * (bx[3] - bx[1]) > self.max_box_frac * H * W:
                        continue  # table-sized boxes
                    if tbox is not None and box_iou(bx, tbox) > self.target_iou:
                        continue  # the target is not an obstacle
                    kept.append(int(i))
                    if self.detector_prompt in ("obstacle_names", "candidate_list", "vlm_name"):
                        break  # AEGIS keeps one box per obstacle phrase (utils.py:197 ``pixel_boxes[0]``; here top-1 score)
                for j, i in enumerate(kept[: self.max_obstacles]):
                    rec = {"box": np.asarray(boxes[i], float).round(1).tolist(), "score": float(scores[i])}
                    if trim:
                        p, tinfo = self.trim_box(obs, cam, boxes[i], table_z)
                        rec["trim"] = tinfo
                        rec["name"] = None
                        if len(p) >= self.min_points:  # empty / tiny crops (e.g. the table, the robot) are no object
                            entries.append({"key": key, "phrase": phrases[i], "cam": cam, "points": p, "rec": rec,
                                            "label": f"{cam}:{key if self.detector_prompt in ('obstacle_names', 'candidate_list', 'vlm_name') else j}"})
                        info["boxes"].setdefault(cam, []).append(rec)
                        continue
                    x0, y0, x1, y1 = np.round(boxes[i]).astype(int)
                    m = np.zeros((H, W), bool)
                    m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
                    p = backproject(obs.depth[cam], obs.intrinsics[cam], obs.T_base_cam[cam], m)
                    if self.robot_filter:
                        p = self.remove_robot(obs, p)
                    name = key if self.detector_prompt in ("obstacle_names", "candidate_list", "vlm_name") else f"{phrases[i] or 'object'}#{j}"
                    out[name] = np.concatenate([out[name], p]) if name in out else p
                    info["boxes"].setdefault(cam, []).append({"name": name, **rec})
        if trim:
            out, info["association"] = self.associate(entries)
            for e in entries:
                e["rec"]["name"] = e["name"]
        out = {k: self._subsample(v) for k, v in out.items() if len(v) >= self.min_points}
        return ObjectClouds(out, "detector", self.privileged, [], info)


# --- image-plane silhouette of an ellipsoid / robot pixels (KNOWS F3 track freeze, F4a attention masks; 2026-11-10) ---
def ellipsoid_silhouette(center, Q, K, T_base_cam, hw) -> np.ndarray:
    """``(H, W)`` bool: pixels whose viewing ray meets the ellipsoid ``{x : (x - c)^T Q^-1 (x - c) <= 1}`` in front of
    the camera (exact projection; the paper's "ellipsoid projected onto the image plane", KNOWS Sec. 3.3). Pixel
    convention = :func:`backproject` (column ``u`` / row ``v`` indices with ``K`` as given). Camera inside -> all True."""
    H, W = int(hw[0]), int(hw[1])
    T = np.asarray(T_base_cam, np.float64)
    K = np.asarray(K, np.float64)
    P = np.linalg.inv(np.asarray(Q, np.float64))
    oc = T[:3, 3] - np.asarray(center, np.float64).reshape(3)
    cc = float(oc @ P @ oc) - 1.0
    if cc <= 0.0:
        return np.ones((H, W), bool)
    v, u = np.mgrid[0:H, 0:W]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((H, W))], -1).reshape(-1, 3)
    d = rays @ T[:3, :3].T  # base-frame ray directions (t > 0 = in front)
    a = np.einsum("ni,ij,nj->n", d, P, d)
    b = 2.0 * (d @ (P @ oc))
    disc = b * b - 4.0 * a * cc
    hit = (disc >= 0.0) & (b < 0.0)  # cc > 0: both roots share a sign; in front <=> -b / 2a > 0
    return hit.reshape(H, W)


def robot_spheres(obs: SceneObservation, model_cache: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Panda collision spheres ``(centres (S, 3), radii (S,))`` at ``obs.q`` (fingers from ``extras['robot_state']``),
    the model :meth:`ObstacleCloudSource.robot_keep_mask` uses (``AG3S/adapters/panda``; kinematics, not GT)."""
    cache = {} if model_cache is None else model_cache
    if "model" not in cache:
        from AG3S.adapters.panda import model as panda

        cache["panda"], cache["model"] = panda, panda.build_panda_bare_model()
    rs = obs.extras.get("robot_state")
    if rs is not None and len(np.asarray(rs).reshape(-1)) == 9:
        q = cache["panda"].set_finger_parameters(cache["model"], rs)
    else:
        q = np.asarray(obs.q, np.float64).reshape(-1)[:7]
    C, R = cache["model"].sphere_centers_numeric(q)
    return np.asarray(C, np.float64).reshape(-1, 3), np.asarray(R, np.float64).reshape(-1)


def robot_pixels_in(obs: SceneObservation, cam: str, region: np.ndarray, spheres, margin: float) -> int:
    """Number of valid-depth pixels of ``region`` (camera ``cam``) whose back-projected point lies inside a robot
    sphere + ``margin`` -- the robot seen inside that image region (the same test as the detector crop's robot cut)."""
    p = backproject(obs.depth[cam], obs.intrinsics[cam], obs.T_base_cam[cam], region)
    if not len(p):
        return 0
    C, R = spheres
    inside = np.zeros(len(p), bool)
    for c, r in zip(C, R):
        inside |= np.einsum("ij,ij->i", p - c, p - c) <= (r + margin) ** 2
    return int(np.count_nonzero(inside))


# --- recorded LIBERO frames (smoke input) ----------------------------------------------------------------------
RAW_RGB_KEYS = {"head": "obs_agentview_image", "left_wrist": "obs_robot0_eye_in_hand_image"}


def resolve_run(run: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(run)
    if p.is_dir():
        return p
    from common.run_artifacts import results_root

    return results_root() / str(run)


def load_recorded_call(run: str | pathlib.Path, stem: str, call: int, *, cameras=("head", "left_wrist")
                       ) -> tuple[SceneObservation, np.ndarray, dict]:
    """One policy call of a ``--capture-geometry --capture-segmentation`` LIBERO run.

    Returns ``(obs, recorded action chunk (T, 7), meta)``. Same reading rules as ``eval/offline_grounding.py``
    (RGB OpenCV = raw[::-1]; depth uint16 mm / 1000; K / T_base_cam of the geom row whose ``obs_index`` = chunk_t).
    ``ee_pose`` = robosuite ``robot0_eef_pos`` / ``robot0_eef_quat`` (the frame AEGIS reads) in the base frame; the
    URDF ``panda_grip_site`` frame of ``AG3S/adapters/panda`` is this frame turned by -90 deg about z (same origin).
    """
    from Baseline.paper_baselines.providers.gt_oracle import libero_target

    run = resolve_run(run)
    rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    row = next(r for r in rows if f"{r['suite']}_t{int(r['task_id'])}_e{int(r['episode_idx'])}" == stem)
    with np.load(run / "episodes" / f"{stem}.npz", allow_pickle=False) as z:
        chunk_t = z["chunk_t"].astype(np.int64)
        t = int(chunk_t[call])
        actions = z["action_chunks"][call].astype(np.float64)
        rgb_raw = {c: z[RAW_RGB_KEYS[c]][t] for c in cameras}
        eef_pos, eef_quat = z["obs_robot0_eef_pos"][t], z["obs_robot0_eef_quat"][t]
        q = z["obs_robot0_joint_pos"][t].astype(np.float64)
        gq = z["obs_robot0_gripper_qpos"][t].astype(np.float64)
        executed = z["action"]
        last_grip = float(executed[t - 1, 6]) if t >= 1 else -1.0
    with np.load(run / "episodes" / f"{stem}_geom.npz", allow_pickle=False) as g:
        r = int(np.flatnonzero(g["obs_index"] == t)[0])
        tables = {"model_geom_bodyid": g["model_geom_bodyid"], "model_body_names": g["model_body_names"]}
        depth = {c: g[f"depth_mm/{c}"][r].astype(np.float64) / 1000.0 for c in cameras}
        K = {c: g[f"K/{c}"][r].astype(np.float64) for c in cameras}
        T = {c: g[f"T_base_cam/{c}"][r].astype(np.float64) for c in cameras}
        seg = {c: g[f"segmentation_geom/{c}"][r] for c in cameras}
        T_world_base = g["T_world_base"][r].astype(np.float64)
        robot_state = g[f"robot_state/{cameras[0]}"][r].astype(np.float64)
    target = libero_target(row["suite"], int(row["task_id"]), tables)
    T_world_ee = np.eye(4)
    T_world_ee[:3, :3], T_world_ee[:3, 3] = quat_xyzw_to_mat(eef_quat), eef_pos
    ee_pose = np.linalg.inv(T_world_base) @ T_world_ee
    others = [n for n in target["objects"] if n not in (target["target"], target.get("destination"))]
    fixtures = [n for n in target["fixtures"] if n not in SUPPORT_NAMES]
    obs = SceneObservation(
        rgb={c: np.ascontiguousarray(rgb_raw[c][::-1]) for c in cameras}, depth=depth, intrinsics=K, T_base_cam=T,
        q=q, ee_pose=ee_pose, gripper=float(gq[0] - gq[1]), instruction=row.get("task_description") or target["language"],
        target_phrase=phrase_from_object(target["target"]),
        extras={"suite": row["suite"], "task_id": int(row["task_id"]), "obs_index": t, "call": int(call),
                "robot_state": robot_state, "T_world_base": T_world_base, "gripper_closed": last_grip > 0.0,
                PRIVILEGED_PREFIX + "segmentation_geom": seg,
                PRIVILEGED_PREFIX + "instances_geom_ids": {k: list(map(int, v)) for k, v in target["instances_geom_ids"].items()},
                PRIVILEGED_PREFIX + "target": target["target"],
                PRIVILEGED_PREFIX + "destination": target.get("destination"),
                PRIVILEGED_PREFIX + "fixture_names": list(target["fixtures"]),
                # LIBERO has no designated obstacle: every non-target, non-destination object + non-support fixture
                PRIVILEGED_PREFIX + "obstacle_names": others + fixtures})
    meta = {"run": str(run), "stem": stem, "call": int(call), "obs_index": t, "suite": row["suite"],
            "task_id": int(row["task_id"]), "target": target["target"], "destination": target.get("destination")}
    return obs, actions, meta


def rollout_positions(ee_pose: np.ndarray, actions: np.ndarray, pos_gain: float = POS_GAIN_M) -> np.ndarray:
    """Predicted EE positions (T + 1, 3) of a chunk (for figures / tests)."""
    p = [np.asarray(ee_pose)[:3, 3]]
    for a in np.asarray(actions):
        p.append(p[-1] + pos_gain * a[:3])
    return np.asarray(p)


__all__ = ["AEGIS_PREFER_LIST", "AEGIS_PREFER_LIST_LONG", "aegis_prefer_list", "AEGIS_PREFER_LIST_WORDING", "NAME_WORDINGS", "obstacle_query_phrase", "CAMERAS_EXTERNAL", "OSC_POS_SCALE", "OSC_ROT_SCALE", "POS_GAIN_M", "ROT_GAIN_RAD", "SUPPORT_NAMES", "ObjectClouds",
           "ObstacleCloudSource", "axisangle_to_mat", "backproject", "backproject_uv", "box_iou", "center_component",
           "check_actions", "cloud_overlap", "depth_trim", "ellipsoid_silhouette", "robot_pixels_in", "robot_spheres",
           "load_recorded_call", "mat_to_axisangle", "phrase_from_object", "propagate_ee", "quat_xyzw_to_mat",
           "rollout_positions", "skew", "voxel_components"]
