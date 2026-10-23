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
  ``extras["privileged/instances_geom_ids"]``; obstacle set = ``extras["privileged/obstacle_names"]`` or every
  instance except the excluded target / support;
* ``detector``  -- Grounding DINO boxes (A's ``providers/grounding_dino.py``) cropped on the depth image, the
  AEGIS perception (``benchmark/SafeLIBERO/main/utils.py:155-255``). Prompts: SafeLIBERO obstacle names
  (privileged metadata, replacing the VLM) or a generic object prompt with the target box removed.
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
    target box (IoU > ``target_iou``) removed (replacement (b); not privileged).
    """

    SOURCES = ("given", "gt_seg", "detector")
    EXCLUDE = ("task_objects", "phase_target", "target", "none")

    def __init__(self, source: str = "gt_seg", *, exclude: str = "task_objects", include_fixtures: bool = True,
                 cameras: tuple[str, ...] | None = None, min_points: int = 30, max_points: int = 4000,
                 detector: Any = None, detector_weight: str = "grounding-dino-tiny", detector_device: str = "cpu",
                 detector_prompt: str = "obstacle_names", generic_prompt: str = "object",
                 box_threshold: float = 0.35, text_threshold: float = 0.25, target_iou: float = 0.5,
                 max_box_frac: float = 0.5, max_obstacles: int = 8, robot_filter: bool = True,
                 robot_margin: float = 0.02, seed: int = 0):
        if source not in self.SOURCES:
            raise ValueError(f"source must be one of {self.SOURCES}, got {source!r}")
        if exclude not in self.EXCLUDE:
            raise ValueError(f"exclude must be one of {self.EXCLUDE}, got {exclude!r}")
        if detector_prompt not in ("obstacle_names", "all_objects"):
            raise ValueError(f"detector_prompt must be obstacle_names|all_objects, got {detector_prompt!r}")
        self.source, self.exclude, self.include_fixtures = source, exclude, include_fixtures
        self.cameras, self.min_points, self.max_points = cameras, int(min_points), int(max_points)
        self._detector, self.detector_weight, self.detector_device = detector, detector_weight, detector_device
        self.detector_prompt, self.generic_prompt = detector_prompt, generic_prompt
        self.box_threshold, self.text_threshold, self.target_iou = box_threshold, text_threshold, target_iou
        self.max_box_frac, self.max_obstacles = max_box_frac, int(max_obstacles)
        self.robot_filter, self.robot_margin = bool(robot_filter), float(robot_margin)
        self._robot_model = None
        self.rng = np.random.default_rng(seed)

    # provenance --------------------------------------------------------------------------------------------
    @property
    def privileged(self) -> bool:
        return self.source == "gt_seg" or (self.source == "detector" and self.detector_prompt == "obstacle_names")

    def provenance(self) -> dict:
        p = {"obstacle_source": self.source, "privileged": self.privileged, "cameras": self.cameras,
             "min_points": self.min_points, "max_points": self.max_points}
        if self.source == "gt_seg":
            p.update(exclude=self.exclude, include_fixtures=self.include_fixtures, support_names=list(SUPPORT_NAMES),
                     privileged_inputs=["segmentation_geom", "instances_geom_ids", "target", "destination"])
        if self.source == "detector":
            det = self._detector
            p.update(detector=(det.provenance() if det is not None and hasattr(det, "provenance")
                               else {"weight": self.detector_weight, "device": self.detector_device, "loaded": False}),
                     detector_prompt=self.detector_prompt, generic_prompt=self.generic_prompt,
                     box_threshold=self.box_threshold, text_threshold=self.text_threshold, target_iou=self.target_iou,
                     crop="axis-aligned box on depth (AEGIS utils.py:188-255), not a mask",
                     robot_filter=(f"drop crop points within sphere radius + {self.robot_margin} m of the Panda "
                                   "collision spheres at q (AG3S/adapters/panda model; not in AEGIS)"
                                   if self.robot_filter else None),
                     privileged_inputs=["obstacle_names"] if self.detector_prompt == "obstacle_names" else [])
        return p

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

    def remove_robot(self, obs: SceneObservation, pts: np.ndarray) -> np.ndarray:
        """Drop points inside the Panda collision spheres (+ margin) at ``obs.q`` -- kinematics, not GT."""
        if not len(pts):
            return pts
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
        return pts[~drop]

    def _detect(self, obs: SceneObservation) -> ObjectClouds:
        det = self.detector
        info: dict = {"boxes": {}}
        out: dict[str, np.ndarray] = {}
        if self.detector_prompt == "obstacle_names":
            names = obs.privileged("obstacle_names")
            if names is None:
                raise KeyError("detector_prompt='obstacle_names' needs extras['privileged/obstacle_names']")
            queries = [(str(n), phrase_from_object(n)) for n in names]
        else:
            queries = [("object", self.generic_prompt)]
        for cam in self._cams(obs, CAMERAS_EXTERNAL):
            rgb = obs.rgb[cam]
            H, W = rgb.shape[:2]
            tbox = None
            if self.detector_prompt == "all_objects" and obs.target_phrase:
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
                    if self.detector_prompt == "obstacle_names":
                        break  # AEGIS keeps one box per obstacle phrase (utils.py:197 ``pixel_boxes[0]``; here top-1 score)
                for j, i in enumerate(kept[: self.max_obstacles]):
                    x0, y0, x1, y1 = np.round(boxes[i]).astype(int)
                    m = np.zeros((H, W), bool)
                    m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
                    p = backproject(obs.depth[cam], obs.intrinsics[cam], obs.T_base_cam[cam], m)
                    if self.robot_filter:
                        p = self.remove_robot(obs, p)
                    name = key if self.detector_prompt == "obstacle_names" else f"{phrases[i] or 'object'}#{j}"
                    out[name] = np.concatenate([out[name], p]) if name in out else p
                    info["boxes"].setdefault(cam, []).append(
                        {"name": name, "box": np.asarray(boxes[i], float).round(1).tolist(), "score": float(scores[i])})
        out = {k: self._subsample(v) for k, v in out.items() if len(v) >= self.min_points}
        return ObjectClouds(out, "detector", self.privileged, [], info)


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


__all__ = ["CAMERAS_EXTERNAL", "OSC_POS_SCALE", "OSC_ROT_SCALE", "POS_GAIN_M", "ROT_GAIN_RAD", "SUPPORT_NAMES", "ObjectClouds",
           "ObstacleCloudSource", "axisangle_to_mat", "backproject", "box_iou", "check_actions", "depth_trim",
           "load_recorded_call", "mat_to_axisangle", "phrase_from_object", "propagate_ee", "quat_xyzw_to_mat",
           "rollout_positions", "skew"]
