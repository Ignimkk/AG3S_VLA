"""KNOWS [17] CBF-QP (``cbf/filter.py``) behind the common :class:`~Baseline.paper_baselines.methods.interface.SafetyFilter`.

Nothing in ``cbf/`` changes; this adapter only maps one policy call onto it, following ``offline_filter.py``:

* robot = one ellipsoid rigidly attached to the EE, shape rotated with the EE (``R Q_local R^T``). Default
  ``ee_model="hand_mvee"`` (F1, 2026-11-09): the MVEE of the Panda hand + open-finger collision-mesh vertices, centred
  ``HAND_MVEE_CENTER`` (grip-site frame) behind the grip site -- the paper's "calibrated offline" EE ellipsoid (Sec. 3.1;
  Sec. 5 leaves only links upstream of the wrist unprotected). The offset centre moves with wrist rotation through the
  ``cbf.filter.RobotBody`` lever-arm term (one part, the paper's single ellipsoid). ``ee_model="grip_site_v0"`` is the
  port's original model (``offline_filter.DEFAULT_GRIPPER_SEMI_AXES`` (0.04, 0.04, 0.07) m centred on the grip site),
  which encloses none of the hand / finger vertices (audit ``outputs/2026-11-09_KNOWS_port_점검``);
* obstacles = one MVEE per object point cloud (KNOWS ``perception.ellipsoid_fit.mvee``, Khachiyan), shapes frozen at
  the first call after ``reset`` and centroids re-measured on later calls (``ObjectTracker`` rule, Sec. 3.2);
* target exclusion (paper Eq. 4: attention top-1 object) -- by default the BDDL target and destination from
  privileged GT (``ObstacleCloudSource(exclude="task_objects")``); ``exclude="phase_target"`` is the
  ``offline_filter --target gt`` rule (target before grasp, destination after); an upstream attention / grounding
  module can instead hand over target-free clouds with ``obstacle_source="given"``;
* units: ``delta_c = a[:3] * 0.05 m``, ``delta_theta = a[3:6] * 0.5 rad`` (robosuite OSC ``output_max``, Appendix 7.1
  "scaled to physical units") and back; emergency stop -> zero motion (Appendix 7.1); gripper untouched;
* chunk (T > 1): the EE pose before action k is predicted with :func:`propagate_ee` (measured gain); T = 1 per env
  step reproduces the per-step filter (closed loop ``--filter-rate step``, F2 2026-11-10).

F3 (2026-11-10, paper Sec. 3.2 "freezing a track at its last position when the arm intervenes"), ``track_freeze="arm"``
(default; ``"off"`` = the port before F3): before a track's centroid is re-measured, its object's pixels in every camera
the tracker reads (``gt_seg``: the GT mask in head + wrist; ``detector``: the boxes of the matched cloud in head) are
dilated by one pixel; if at least ``freeze_min_pixels`` = 1 valid-depth pixel there lies within a Panda collision sphere
+ ``freeze_margin_m`` (the robot touching / occluding the object, :func:`safety_common.robot_pixels_in`; default = the
source's ``robot_margin`` 0.02 m, the detector crop's robot cut), the track keeps its last centre this step. Same rule
for ``gt_seg`` and ``detector``; a track not seen this step follows the existing rule; nothing else changes (no other
hold, no smoothing). ``given`` (depth-lift arm, no pixel masks): the track's projected ellipsoid stands in for the mask.

F4a (2026-11-10, paper Eq. 2-4), ``target_exclusion="attention"`` (detector source only; default ``"source"`` = the
cloud source's own rule): every tracked object is a candidate; at each policy call the pi0.5 attention grid handed in
``extras["knows/attention"]`` (layer 12, head 3, action-query mean x agent-view keys, 16 x 16, OpenCV orientation,
``closed_loop.attention_tap``) gives each track ``m = sum A c`` over the track's projected silhouette in the head camera
(Eq. 2, ``c`` = patch coverage) and ``alpha`` = silhouette area in 224-px policy pixels; Eq. 3 ``d = sum_K m (sum_K
alpha)^beta`` over the last ``attention_k`` policy calls, Eq. 4 target = top-1 if ``d1 - d2 >= attention_delta`` else
none; the target is dropped from the CBF obstacle set until the next policy call. ``K`` = 5 and ``delta`` = 7.3e-6 are
not in the paper: P0b (``docs/08-p0b-results.md``: K = 5, gap p10 = 7.30e-6 at layer 12 / head 3).

``CbfParams()`` defaults reproduce the paper (Eq. 12 as written); pass ``cbf_params={...}`` for the D1-D7 variants.

Track identity (CLOSED-LOOP-FIX, ``associate``): the paper keeps one track per object (shape frozen at t=0, centroid
re-measured, **held at the last position while occluded**, identity swaps re-associated by an HSV histogram match;
§3.2, ``perception/ellipsoid_fit.ObjectTracker``). Source names are stable for ``gt_seg`` (instance names) but not for
``detector`` (``object#j`` = j-th box by score, re-ranked every call) or ``given`` (``depth#j`` = j-th largest
component). ``associate="centroid"`` (default for those sources) matches each frozen track to the nearest current
cloud centroid (Hungarian, gate ``assoc_gate_m``; centroid distance replaces the colour histogram, which the box
crops cannot provide) and holds unmatched tracks at their last centroid. ``associate="name"`` (default for
``gt_seg`` / ``ellipsoids``) is the previous behaviour (match by name, unmatched -> t=0 centroid).

Per-step diagnostics (``info["steps"][k]``, ``diagnostics=True``; nothing here feeds back into the QP):

* ``min_h`` -- Eq. (6) ``h(n)`` at the **stored** separating normal (the QP's own row value, before the step's
  ``delta_n``), at the EE pose before row ``k`` along the **filtered** chunk (row 0 = measured pose, rows k > 0 =
  ``propagate_ee`` of the filtered rows 0..k-1). This is the quantity the closed-loop ``h_min`` / ``h_neg_steps`` count;
* ``h_star`` / ``h_star_next`` -- the true separation ``max_n h(n)`` (``optimal_normals_batch``) before the row and
  after the QP's step ``(delta_c, delta_theta)``; ``h_star_next_nominal`` -- after the nominal step instead;
* ``cbf_ok`` -- docs/16 §1 condition on the true barrier, every obstacle: ``h*_next >= (1 - gamma_h) h*``;
* ``relax_m`` / ``need_m`` (binding row = min ``h(n)``): ``eps * ||grad_n h||_1`` (what tilting the normal inside the
  Eq. 12 box can supply, ``cbf.filter.effective_margin``) vs ``gamma_h * max(-h, 0)`` (what Eq. 7 demands of an
  overlapping pair). ``relax_m >= need_m`` = the row can be met by ``delta_n`` alone (docs/11 §4).
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from Baseline.paper_baselines.methods.interface import FilterResult, SceneObservation, register_method
from Baseline.paper_baselines.methods.knows.cbf.ellipsoid import (Ellipsoid, barrier, grad_normal, initial_normal,
                                                                  optimal_normals_batch, rotate_shape)
from Baseline.paper_baselines.methods.knows.cbf.filter import CbfParams, RobotBody, SafetyFilter as KnowsCbfFilter
from Baseline.paper_baselines.methods.knows.perception.ellipsoid_fit import mvee
from Baseline.paper_baselines.methods.safety_common import (CAMERAS_EXTERNAL, OSC_POS_SCALE, OSC_ROT_SCALE, POS_GAIN_M,
                                                             ROT_GAIN_RAD, ObstacleCloudSource, check_actions,
                                                             ellipsoid_silhouette, propagate_ee, robot_pixels_in,
                                                             robot_spheres)

#: ``offline_filter.DEFAULT_GRIPPER_SEMI_AXES`` (not imported: offline_filter pulls the LIBERO camera model) -- the
#: port's original grip-site-centred EE ellipsoid (``ee_model="grip_site_v0"``).
GRIPPER_SEMI_AXES = (0.04, 0.04, 0.07)
#: F1 (2026-11-09, user-approved fidelity fix): MVEE (Khachiyan, margin 0) of the Panda hand + open-finger collision-mesh
#: vertices in the grip-site frame, ``outputs/2026-11-09_KNOWS_port_점검/gripper_coverage.json`` ``calibration``.
#: ``HAND_MVEE_AXES`` columns = principal axes (grip-site frame), semi-axes in the same order.
HAND_MVEE_CENTER = (-0.0075, 0.0, -0.0699)
HAND_MVEE_SEMI_AXES = (0.0332, 0.0884, 0.1287)
HAND_MVEE_AXES = ((0.0, -0.085, -0.996), (-1.0, -0.001, -0.0), (-0.001, 0.996, -0.085))
#: Frame fix (2026-11-10, KNOWS collapse diagnosis outputs/2026-11-10_KNOWS_collapse_진단): ``obs.ee_pose`` orientation is the
#: robosuite ``robot0_right_hand`` body = the grip-site frame rotated by Rz(+90 deg) (``panda_gripper.xml:14`` quat
#: (0.707, 0, 0, -0.707); measured 90.0 deg on 98 recorded frames). A grip-site-frame vector v is ``GRIP_SITE_TO_EE @ v``
#: in the EE frame. ``hand_mvee`` (F1) applies it; ``hand_mvee_v1`` = the 2026-11-09 F1 without it (axes 90 deg off:
#: 4.7 % of the hand + finger vertices inside, vs 94.2 % with it), kept only to reproduce those runs.
GRIP_SITE_TO_EE = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
EE_MODELS = ("hand_mvee", "hand_mvee_v1", "grip_site_v0")
#: F3 track freeze: ``arm`` = paper Sec. 3.2 (default), ``off`` = the port before 2026-11-10
TRACK_FREEZE = ("arm", "off")
#: F4a target exclusion: ``source`` = the cloud source's rule (gt_seg: BDDL task objects; detector: GDINO target box),
#: ``attention`` = paper Eq. 2-4 on the pi0.5 attention grid (detector source only)
TARGET_EXCLUSION = ("source", "attention")
#: F4a constants. Paper: layer 12 / head 3 (Sec. 3.4), agent view, beta = -1 (Sec. 3.3). Not in the paper (P0b,
#: docs/08-p0b-results.md): K = 5 frames, delta = gap p10 at the paper cell (~10 % of steps with top-1 ~ top-2).
ATTENTION_LAYER, ATTENTION_HEAD, ATTENTION_BETA = 12, 3, -1.0
ATTENTION_K = 5
ATTENTION_DELTA = 7.30e-6
ATTENTION_KEY = "knows/attention"
#: Eq. 2 grid / pixel units: 16 x 16 SigLIP patches over the 224-px policy image (P0b alpha is in 224-px pixels)
ATTENTION_GRID, POLICY_PX = 16, 224


class AttentionTarget:
    """Paper Eq. 2-4 over the tracked ellipsoids (see module docstring, F4a). One frame per policy call."""

    def __init__(self, k: int = ATTENTION_K, delta: float = ATTENTION_DELTA, beta: float = ATTENTION_BETA,
                 camera: str = "head"):
        self.k, self.delta, self.beta, self.camera = int(k), float(delta), float(beta), camera
        self.reset()

    def reset(self) -> None:
        self.frames: list[dict[str, tuple[float, float]]] = []
        self.last_seq = None
        self.target: str | None = None
        self.info: dict = {"target": None, "reason": "no attention yet"}

    def masses(self, obs: SceneObservation, grid: np.ndarray, tracks: dict[str, Ellipsoid]) -> dict:
        """Eq. 2 per track: ``(m, alpha)``; ``grid`` (16, 16) in the camera's OpenCV orientation."""
        cam = self.camera
        d = np.asarray(obs.depth[cam])
        H, W = d.shape[:2]
        g = np.asarray(grid, np.float64).reshape(ATTENTION_GRID, ATTENTION_GRID)
        ph, pw = H // ATTENTION_GRID, W // ATTENTION_GRID
        px_scale = (POLICY_PX / H) * (POLICY_PX / W)
        out = {}
        for name, E in tracks.items():
            m = ellipsoid_silhouette(E.c, E.Q, obs.intrinsics[cam], obs.T_base_cam[cam], (H, W))
            cov = m[: ph * ATTENTION_GRID, : pw * ATTENTION_GRID].reshape(ATTENTION_GRID, ph, ATTENTION_GRID, pw).mean((1, 3))
            out[name] = (float((g * cov).sum()), float(m.sum()) * px_scale)
        return out

    def update(self, obs: SceneObservation, tracks: dict[str, Ellipsoid]) -> str | None:
        att = obs.extras.get(ATTENTION_KEY)
        if not isinstance(att, dict) or att.get("grid") is None or self.camera not in obs.depth:
            if self.last_seq is None:
                self.info = {"target": None, "reason": "no attention grid in this call"}
            return self.target  # filter step between policy calls: the call's target is kept
        seq = att.get("seq")
        if seq is not None and seq == self.last_seq:
            return self.target
        self.last_seq = seq
        self.frames.append(self.masses(obs, att["grid"], tracks))
        self.frames = self.frames[-self.k:]
        names = sorted(tracks)
        msum = {n: sum(f.get(n, (0.0, 0.0))[0] for f in self.frames) for n in names}
        asum = {n: sum(f.get(n, (0.0, 0.0))[1] for f in self.frames) for n in names}
        dens = {n: (msum[n] * max(asum[n], 1e-6) ** self.beta) for n in names}  # Eq. 3 (P0b: area floor 1e-6)
        order = sorted(names, key=lambda n: -dens[n])
        gap = (dens[order[0]] - dens[order[1]]) if len(order) >= 2 else None
        # Eq. 4; fewer than two candidates: no lead to measure -> no target (the conservative branch of Eq. 4)
        self.target = order[0] if gap is not None and gap >= self.delta else None
        self.info = {"target": self.target, "seq": seq, "frames": len(self.frames), "gap": gap,
                     "density": {n: dens[n] for n in order}, "mass": {n: self.frames[-1].get(n, (0, 0))[0] for n in order},
                     "alpha": {n: self.frames[-1].get(n, (0, 0))[1] for n in order},
                     "reason": None if self.target else ("fewer than 2 tracks" if gap is None else "gap < delta")}
        return self.target


class KnowsSafetyFilter:
    name = "knows"

    ASSOCIATE = ("auto", "name", "centroid")

    def __init__(self, obstacle_source: str = "gt_seg", *, cbf_params: dict | CbfParams | None = None,
                 ee_model: str = "hand_mvee", gripper_semi_axes=None, exclude: str = "task_objects",
                 track_centroids: bool = True,
                 pos_gain: float = POS_GAIN_M, rot_gain: float = ROT_GAIN_RAD, intervene_tol: float = 1e-6,
                 associate: str = "auto", assoc_gate_m: float = 0.10, diagnostics: bool = True,
                 track_freeze: str = "arm", freeze_margin_m: float | None = None, freeze_min_pixels: int = 1,
                 target_exclusion: str = "source", attention_k: int = ATTENTION_K,
                 attention_delta: float = ATTENTION_DELTA, **source_kwargs):
        if track_freeze not in TRACK_FREEZE:
            raise ValueError(f"track_freeze must be one of {TRACK_FREEZE}")
        if target_exclusion not in TARGET_EXCLUSION:
            raise ValueError(f"target_exclusion must be one of {TARGET_EXCLUSION}")
        if target_exclusion == "attention":
            if obstacle_source != "detector":
                raise ValueError("target_exclusion='attention' (F4a) applies to the detector source only")
            source_kwargs["target_box"] = False  # every GDINO 'object' box is a candidate; attention picks the target
        self.track_freeze, self.freeze_min_pixels = track_freeze, int(freeze_min_pixels)
        self.target_exclusion = target_exclusion
        self.attention = AttentionTarget(attention_k, attention_delta) if target_exclusion == "attention" else None
        if associate not in self.ASSOCIATE:
            raise ValueError(f"associate must be one of {self.ASSOCIATE}")
        if associate == "auto":
            associate = "name" if obstacle_source in ("gt_seg", "ellipsoids") else "centroid"
        self.associate, self.assoc_gate_m, self.diagnostics = associate, float(assoc_gate_m), bool(diagnostics)
        if isinstance(cbf_params, CbfParams):
            self.params = cbf_params
        else:
            self.params = CbfParams(**(cbf_params or {}))
        self.obstacle_source = obstacle_source
        self.source = (None if obstacle_source == "ellipsoids"
                       else ObstacleCloudSource(obstacle_source, exclude=exclude, **source_kwargs))
        self.freeze_margin_m = (float(freeze_margin_m) if freeze_margin_m is not None
                                else (self.source.robot_margin if self.source is not None else 0.02))
        self._robot_cache: dict = {}
        if gripper_semi_axes is not None:  # explicit grip-site-centred axis-aligned ellipsoid (old keyword)
            ee_model = "custom_grip_site"
            self.Q_local = np.diag(np.asarray(gripper_semi_axes, np.float64) ** 2)
            self.ee_offset = np.zeros(3)
        elif ee_model in ("hand_mvee", "hand_mvee_v1"):
            V = np.asarray(HAND_MVEE_AXES, np.float64)
            Q_site = V @ np.diag(np.asarray(HAND_MVEE_SEMI_AXES, np.float64) ** 2) @ V.T
            c_site = np.asarray(HAND_MVEE_CENTER, np.float64)
            M = np.asarray(GRIP_SITE_TO_EE, np.float64) if ee_model == "hand_mvee" else np.eye(3)
            self.Q_local = M @ Q_site @ M.T
            self.ee_offset = M @ c_site
        elif ee_model == "grip_site_v0":
            self.Q_local = np.diag(np.asarray(GRIPPER_SEMI_AXES, np.float64) ** 2)
            self.ee_offset = np.zeros(3)
        else:
            raise ValueError(f"ee_model must be one of {EE_MODELS}")
        self.ee_model = ee_model
        self.track_centroids, self.pos_gain, self.rot_gain = bool(track_centroids), float(pos_gain), float(rot_gain)
        self.intervene_tol = float(intervene_tol)
        self.cbf = KnowsCbfFilter(self.params)
        self.reset()

    def provenance(self) -> dict:
        return {"method": self.name,
                "cite": "[17] Your Model Already Knows: Attention-Guided Safety Filter for VLA Models, arXiv:2606.09749",
                "official_code": "none (paper-only reimplementation, Baseline/paper_baselines/methods/knows, moved from "
                                 "Baseline/knows_vla)",
                "cbf_params": dataclasses.asdict(self.params) | {"joint_weights": None},
                "robot": {"ee_model": self.ee_model,
                          "semi_axes": np.sqrt(np.clip(np.linalg.eigvalsh(self.Q_local), 0, None)).tolist(),
                          "center_offset_ee_frame_m": self.ee_offset.tolist(),
                          "frame": "obs.ee_pose (robot0_right_hand body = grip site * Rz(+90 deg))",
                          "calibration": {"hand_mvee": "hand + open-finger collision-mesh MVEE, margin 0 (F1 2026-11-09), "
                                                       "grip-site axes mapped to the EE frame (frame fix 2026-11-10)",
                                          "hand_mvee_v1": "F1 2026-11-09 WITHOUT the frame fix (axes 90 deg off; "
                                                          "reproduction only)"}.get(self.ee_model,
                                                                                    "port original (grip-site centred)")},
                "obstacle": (self.source.provenance() if self.source is not None
                             else {"obstacle_source": "ellipsoids", "privileged": False}),
                "obstacle_fit": "KNOWS mvee (Khachiyan), shapes frozen at first call, centroids tracked",
                "track_identity": {"associate": self.associate, "gate_m": self.assoc_gate_m,
                                   "unmatched_track": "hold last centroid" if self.associate == "centroid"
                                   else "t=0 centroid"},
                "diagnostics": self.diagnostics,
                "track_freeze": ({"rule": "arm", "test": "robot pixels (valid depth within a Panda collision sphere + "
                                  "margin) touching the object's pixels (gt mask / detector boxes, dilated 1 px) in a "
                                  "tracker camera -> keep the last centre (paper Sec. 3.2, F3 2026-11-10)",
                                  "dilation_px": 1, "margin_m": self.freeze_margin_m,
                                  "min_pixels": self.freeze_min_pixels, "cameras": self._tracker_cameras_rule()}
                                 if self.track_freeze == "arm" else {"rule": "off (port before F3)"}),
                "target_exclusion": ({"rule": "attention (paper Eq. 2-4, F4a 2026-11-10)", "layer": ATTENTION_LAYER,
                                      "head": ATTENTION_HEAD, "camera": "head (agent view)", "query_agg": "mean",
                                      "denoise_step": 0, "beta": ATTENTION_BETA, "K_policy_calls": self.attention.k,
                                      "delta": self.attention.delta, "mask": "projected tracked ellipsoid (exact "
                                      "silhouette), alpha in 224-px pixels", "source": "K, delta: P0b docs/08"}
                                     if self.attention is not None else {"rule": "source"}),
                "units": {"pos_scale": OSC_POS_SCALE, "rot_scale": OSC_ROT_SCALE},
                "chunk": {"pos_gain_m": self.pos_gain, "rot_gain_rad": self.rot_gain}}

    def reset(self) -> None:
        self.cbf.reset()
        self.shapes: dict[str, Ellipsoid] | None = None
        self.mean0: dict[str, np.ndarray] = {}
        self.last_c: dict[str, np.ndarray] = {}
        self.cur_c: dict[str, np.ndarray] = {}  # F3: the centre each track had at the last call
        self.perception_info: dict = {}
        if getattr(self, "attention", None) is not None:
            self.attention.reset()

    def _tracker_cameras_rule(self) -> str:
        return {"gt_seg": "every camera of the observation (head + wrist, as the gt_seg clouds)",
                "detector": "external cameras (head), as the detector clouds"}.get(self.obstacle_source,
                                                                                 "every camera of the observation")

    def _tracker_cameras(self, obs: SceneObservation) -> list[str]:
        cams = list(CAMERAS_EXTERNAL) if self.obstacle_source == "detector" else list(obs.depth)
        if self.source is not None and self.source.cameras is not None:
            cams = list(self.source.cameras)
        return [c for c in cams if c in obs.depth and c in obs.intrinsics and c in obs.T_base_cam]

    def _regions(self, obs: SceneObservation, clouds, cloud: str, cams: list[str], E: Ellipsoid) -> dict:
        """The object's pixels per tracker camera this step: ``gt_seg`` = its GT mask, ``detector`` = the boxes whose
        crops formed the matched cloud, ``given`` (no pixels on that path) = the last ellipsoid's projection."""
        out = {}
        for cam in cams:
            hw = np.asarray(obs.depth[cam]).shape[:2]
            if self.obstacle_source == "gt_seg":
                ids = (obs.privileged("instances_geom_ids") or {}).get(cloud)
                seg = (obs.privileged("segmentation_geom") or {}).get(cam)
                m = (np.isin(np.asarray(seg), np.asarray(list(ids), np.int64)) if ids is not None and seg is not None
                     else np.zeros(hw, bool))
            elif self.obstacle_source == "detector":
                m = np.zeros(hw, bool)
                for rec in (clouds.info.get("boxes") or {}).get(cam, []):
                    if rec.get("name") == cloud:
                        x0, y0, x1, y1 = np.round(np.asarray(rec["box"], float)).astype(int)
                        m[max(0, y0):min(hw[0], y1), max(0, x0):min(hw[1], x1)] = True
            else:
                m = ellipsoid_silhouette(E.c, E.Q, obs.intrinsics[cam], obs.T_base_cam[cam], hw)
            if m.any():
                out[cam] = m
        return out

    def _arm_intervenes(self, obs: SceneObservation, regions: dict, spheres) -> dict:
        """F3 test of one track: robot pixels touching its region (region dilated by one pixel, 8-neighbourhood)."""
        from scipy.ndimage import binary_dilation

        hits = {}
        for cam, m in regions.items():
            n = robot_pixels_in(obs, cam, binary_dilation(m, structure=np.ones((3, 3), bool)), spheres,
                                self.freeze_margin_m)
            if n >= self.freeze_min_pixels:
                hits[cam] = n
        return hits

    def _match(self, clouds: dict[str, np.ndarray]) -> tuple[dict[str, str], dict]:
        """Track -> current cloud name: Hungarian on centroid distance (last known centroid), gated."""
        from scipy.optimize import linear_sum_assignment

        tracks = list(self.shapes)
        names = list(clouds)
        if not tracks or not names:
            return {}, {"matched": 0, "unmatched_tracks": len(tracks), "unmatched_clouds": len(names)}
        ref = np.stack([self.last_c[t] for t in tracks])
        cur = np.stack([clouds[n].mean(0) for n in names])
        D = np.linalg.norm(ref[:, None, :] - cur[None, :, :], axis=-1)
        r, c = linear_sum_assignment(D)
        m = {tracks[i]: names[j] for i, j in zip(r, c) if D[i, j] <= self.assoc_gate_m}
        return m, {"matched": len(m), "unmatched_tracks": len(tracks) - len(m), "unmatched_clouds": len(names) - len(m),
                   "pairs": {t: [n, round(float(D[tracks.index(t), names.index(n)]), 4)] for t, n in m.items()}}

    # --- obstacles ---------------------------------------------------------------------------------------------------
    def _obstacles(self, obs: SceneObservation) -> dict[str, Ellipsoid]:
        if self.obstacle_source == "ellipsoids":
            out = {}
            for d in obs.extras.get("obstacle_ellipsoids", []):
                R = np.asarray(d.get("R", np.eye(3)), float)
                out[str(d["name"])] = Ellipsoid.from_semi_axes(d["center"], d["semi_axes"], R)
            return out
        t0 = time.perf_counter()
        clouds = self.source(obs)
        first = self.shapes is None
        if first:
            self.shapes = {}
            for name, pts in clouds.points.items():
                try:
                    c, Q = mvee(pts)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                self.shapes[name] = Ellipsoid(c, Q)
                self.mean0[name] = pts.mean(0)
                self.last_c[name] = pts.mean(0)
        cur = {}
        assoc = None
        if self.associate == "centroid":
            m, assoc = (({n: n for n in self.shapes}, {"matched": len(self.shapes), "first_call": True}) if first
                        else self._match(clouds.points))
        else:
            m = {n: n for n in self.shapes if n in clouds.points}
        frozen: dict[str, dict] = {}
        t_f = time.perf_counter()
        if self.track_freeze == "arm" and not first and self.track_centroids and self.shapes:
            # F3 (paper Sec. 3.2): a track whose object the arm touches in a tracker camera keeps its last position
            cams = self._tracker_cameras(obs)
            spheres = robot_spheres(obs, self._robot_cache)
            for name, E in self.shapes.items():
                if name not in m:  # not seen this step: the existing unmatched rule applies (no measurement)
                    continue
                regions = self._regions(obs, clouds, m[name], cams, Ellipsoid(self.cur_c.get(name, E.c), E.Q))
                hits = self._arm_intervenes(obs, regions, spheres)
                if hits:
                    frozen[name] = hits
        freeze_ms = (time.perf_counter() - t_f) * 1e3
        if self.associate == "centroid":
            for name, E in self.shapes.items():
                if self.track_centroids and name in m and name not in frozen:  # shape frozen, centroid moves
                    self.last_c[name] = clouds.points[m[name]].mean(0)
                # unmatched (occluded / not detected): hold the last known centroid (paper §3.2)
                cur[name] = Ellipsoid(E.c + (self.last_c[name] - self.mean0[name]), E.Q)
        else:
            for name, E in self.shapes.items():
                c = E.c
                if name in frozen:  # F3: the last centre (as used at the previous call)
                    c = self.cur_c[name]
                elif self.track_centroids and name in clouds.points:  # ObjectTracker: shape frozen, centroid moves
                    c = E.c + (clouds.points[name].mean(0) - self.mean0[name])
                cur[name] = Ellipsoid(c, E.Q)
        self.cur_c = {name: E.c.copy() for name, E in cur.items()}
        self.perception_info = {"source": clouds.source, "privileged": clouds.privileged, "excluded": clouds.excluded,
                                "n_points": {k: int(len(v)) for k, v in clouds.points.items()},
                                "associate": self.associate, "association": assoc,
                                "ms": (time.perf_counter() - t0) * 1e3}
        if self.track_freeze == "arm":
            self.perception_info["frozen"] = {k: {c: int(n) for c, n in v.items()} for k, v in frozen.items()}
            self.perception_info["freeze_ms"] = freeze_ms
        return cur

    def _diagnose(self, robot: Ellipsoid, obstacles: dict[str, Ellipsoid], res, dc_nom, dth_nom, stored: dict,
                  lever=None) -> dict:
        """True-barrier view of one step (see module docstring); read-only w.r.t. the filter state."""
        p = self.params
        keys = sorted(obstacles, key=repr)
        if not keys:
            return {}
        obs_l = [obstacles[k] for k in keys]

        def h_star(R):
            N = optimal_normals_batch([R] * len(keys), obs_l, inits=[None] * len(keys), iters=400)
            return np.array([barrier(n, R, o) for n, o in zip(N, obs_l)])

        def moved(dc, dth):
            dth = np.asarray(dth, np.float64)
            shift = np.asarray(dc, np.float64) + (np.cross(dth, lever) if lever is not None else 0.0)
            return Ellipsoid(robot.c + shift, rotate_shape(robot.Q, dth))

        hb = h_star(robot)
        ha = h_star(moved(res.delta_c, res.delta_theta))
        hn = h_star(moved(dc_nom, dth_nom))
        ok = ha >= (1.0 - p.gamma_h) * hb - 1e-9
        out = {"h_star": float(hb.min()), "h_star_next": float(ha.min()), "h_star_next_nominal": float(hn.min()),
               "cbf_ok": bool(ok.all()), "n_cbf_violated": int(np.count_nonzero(~ok))}
        if res.h.size:  # binding row = min h(n): what delta_n alone could supply vs what Eq. 7 asks
            i = int(np.argmin(res.h))
            key = res.rows[i] if i < len(res.rows) else None
            okey = key[1] if key is not None else None
            n = stored.get(key) if key is not None else None
            if okey in obstacles and n is None:
                n = initial_normal(robot, obstacles[okey])  # first sight: the row used the centre-to-centre normal
            if okey in obstacles and n is not None:
                g = grad_normal(n, robot, obstacles[okey])
                out.update(relax_m=float(p.eps_normal * np.abs(g).sum()) if p.normals == "relaxed" else 0.0,
                           need_m=float(p.gamma_h * max(-float(res.h[i]), 0.0)), binding_obstacle=str(okey))
        return out

    # --- filter --------------------------------------------------------------------------------------------------------
    def filter(self, obs: SceneObservation, actions: np.ndarray) -> FilterResult:
        A = check_actions(actions)
        out = A.copy()
        intervened = np.zeros(len(A), bool)
        obstacles = self._obstacles(obs)
        target_info = None
        if self.attention is not None:  # F4a: Eq. 2-4 over every track, the target leaves the obstacle set
            tau = self.attention.update(obs, obstacles)
            target_info = dict(self.attention.info)
            obstacles = {k: v for k, v in obstacles.items() if k != tau}
        info = {"obstacles": {k: {"c": E.c.round(4).tolist(), "Q": E.Q.round(7).tolist(),
                                  "semi_axes": np.sqrt(np.clip(np.linalg.eigvalsh(E.Q), 0, None)).round(4).tolist()}
                              for k, E in obstacles.items()}, "perception": self.perception_info, "steps": []}
        if target_info is not None:
            info["target"] = target_info
        T_ee = np.asarray(obs.ee_pose, np.float64)
        for k, a in enumerate(A):
            R = T_ee[:3, :3]
            Qw = R @ self.Q_local @ R.T
            lever = R @ self.ee_offset
            robot_w = Ellipsoid(T_ee[:3, 3] + lever, Qw)  # the EE ellipsoid in the world (diagnostics)
            robot = (RobotBody(T_ee[:3, 3], {"eef": Ellipsoid(lever, Qw)}) if np.any(self.ee_offset) else robot_w)
            dc_nom, dth_nom = a[:3] * OSC_POS_SCALE, a[3:6] * OSC_ROT_SCALE
            stored = self.cbf.normals() if self.diagnostics else None  # normals the QP rows use (before delta_n)
            res = self.cbf(robot, obstacles, dc_nom, dth_nom)
            new = a.copy()
            if res.status != "no_obstacles":
                new[:3], new[3:6] = res.delta_c / OSC_POS_SCALE, res.delta_theta / OSC_ROT_SCALE
            diff = float(np.max(np.abs(new[:6] - a[:6])))
            if diff <= self.intervene_tol:
                new[:6] = a[:6]  # solver round-off on an inactive QP is not an intervention
            out[k] = new
            intervened[k] = diff > self.intervene_tol
            step = {"status": res.status, "feasible": bool(res.feasible), "emergency_stop": bool(res.emergency_stop),
                    "min_h": float(res.h.min()) if res.h.size else None,
                    "binding": [str(r[1]) for r, h in zip(res.rows, res.h) if h < 0.05][:4]}
            if self.diagnostics and res.status != "no_obstacles":
                step.update(self._diagnose(robot_w, obstacles, res, dc_nom, dth_nom, stored,
                                           lever=lever if np.any(self.ee_offset) else None))
            info["steps"].append(step)
            T_ee = propagate_ee(T_ee, new, self.pos_gain, self.rot_gain)
        return FilterResult(out, intervened, info)


register_method("knows")(KnowsSafetyFilter)


def make_method(**kwargs) -> KnowsSafetyFilter:
    return KnowsSafetyFilter(**kwargs)


__all__ = ["ATTENTION_DELTA", "ATTENTION_K", "AttentionTarget", "TARGET_EXCLUSION", "TRACK_FREEZE", "EE_MODELS", "GRIPPER_SEMI_AXES", "HAND_MVEE_AXES", "HAND_MVEE_CENTER", "HAND_MVEE_SEMI_AXES",
           "KnowsSafetyFilter", "make_method"]
