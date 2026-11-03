"""T79 — ATS time tracking: per-request ground truth, reference request and event type (`handoff/T79.task.md` §1, §2).

Built on the T77 spec (`t77_labels spec`: run list, request pairing **by order** — i-th server chunk <-> i-th client
step sorted by ``t_step``, scene split test = {211, 954, 122, 119}) and the T77 labels (`gt.held` per request).

For every AG3S run and every **pre-grasp** request (before the first GT-held request; all requests if the apple is
never held), with the MuJoCo state of that request (client step qpos in the scene XML, `task_dynamic_map_eval
.scene_truth`, the `module_eval.py` path):

  GT        apple body centre (base frame); the apple GT surface samples visible in the **head camera** (depth agrees
            within 10 mm in ``zed_left``; `task_dynamic_map_eval._visible` restricted to that camera, T75 ``ats_extra``
            ``head_visible``) -> ``vis_n`` and their mean ``vis_c`` (the "visible surface centre").
  ATS       recorded ``manipulated.state`` / ``manipulated.centroid`` and target points: precision = share of target
            points within 5 mm of a GT apple sample, majority object = most common GT object within 10 mm (T75
            ``module_eval``); ``ats_ok`` = precision >= 0.8 ∧ majority = apple (T75 verdict).
  reference ``ref`` = the last earlier pre-grasp request of the run with ``ats_ok`` (-1: none yet).
  event     (a) ``moved_mm`` = |apple centre now − apple centre at ``ref``| >= 20 mm, (b) recorded state ``occluded``
            ∧ ``vis_n`` > 0. ``event`` ∈ {"a", "b", "ab", "normal"}; requests without a reference get ``"no_ref"``
            (the hidden re-search has no appearance vector there; counted, not scored).

For the requests the re-search is scored on (every event — or up to ``--event-cap`` per run if given — + up to
``--normal-per-run`` sampled normal requests per run, seeded) the head camera geometry is tabulated **per 16 × 16 patch** so `t79_track` needs no MuJoCo:

  * depth pixels (every ``PIXEL_STRIDE``-th row / column, 0.1–2.0 m) back-projected to the base frame
    (``X = (u − cx) z / fx``, ``T_base_cam_zed_left``), patch = ``(v · 16 // 480) · 16 + (u · 16 // 640)``
    (`ifg_graph._cam_pixels`; see "patch geometry" below);
  * dropped (serving-side information only, recorded in the same chunk): points inside a recorded robot sphere
    (``sphere_centres`` / ``sphere_radii`` + ``ROBOT_MARGIN_M``) and points within ``SUPPORT_M`` of a recorded support
    point (``raw_cloud[support_mask]``, table / floor);
  * per patch: point count ``N``, coordinate sum ``S`` (centroid of any patch set = ΣS / ΣN) and GT class counts
    ``C`` (nearest GT surface sample within 8 mm: apple / other fruit / crate / table / obstacle / robot, else none).

Also stored for those requests: the recorded noun attention ``attention_head`` (16 × 16, L10H7 — the reference
method). For every ``ats_ok`` request: the appearance patches = head patches its target points project to (only the
points the head depth confirms within 10 mm; all in-frame projections if none is confirmed).

Patch geometry. ``image_cam_high`` (224 × 224) is the ``zed_left`` render at 4:3 (`POLICY_SOURCE_WIDTH` ×
`POLICY_SOURCE_HEIGHT` = 299 × 224) bilinearly **stretched** to 224 × 224 (`rby1_manipulation.data.recording
.resize_policy_image`: no crop, no padding); the recorded depth / K are the same camera at 640 × 480 (fx = fy = 240,
cx = 320, cy = 240). SigLIP So400m/14 cuts the 224 image into 14 × 14 px patches, row-major (token ``k`` = row ``k //
16``, column ``k % 16``). So head token ``k`` covers the depth pixels ``u ∈ [40 c, 40 c + 40)``, ``v ∈ [30 r, 30 r +
30)``: 1 patch = 14 × 14 px at 224² = 40 × 30 px at 640 × 480.

usage (.venv-ag3s, CPU, MuJoCo):
  SNAP=<snapshot_t67> PYTHONPATH=$PWD:$SNAP/integration/rby1/manipulation/src MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=2 \\
    python -m AG3S.ag3s.experiments.studies.t79_events OUT_events.json --spec t77_spec.json --labels t77_labels.json \\
        [--only LABEL ...] [--normal-per-run 4] [--event-cap 0] [--jobs 4]
Writes ``OUT_events.json`` (rows, counts, ``patch_requests`` = {label: requests whose hidden is needed}) and
``OUT_events.geom.npz`` (per-patch tables).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import zlib
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies.t77_labels import TARGET_BODY, default_xml

GRID = 16
IMG_W, IMG_H = 640, 480           # zed_left depth / K resolution
POLICY_PX = 224                   # image_cam_high side
PATCH_PX = POLICY_PX // GRID      # 14 (SigLIP So400m/14)
HEAD_CAM = "zed_left"
PREC_MM = 5.0
MAJ_MM = 10.0
ATS_PREC = 0.80
MOVED_MM = 20.0
VISIBLE_MM = 10.0
CLASS_MM = 8.0
PIXEL_STRIDE = 2
DEPTH_RANGE_M = (0.10, 2.0)
ROBOT_MARGIN_M = 0.005
SUPPORT_M = 0.010
CLASSES = ("apple", "fruit", "crate", "table", "obstacle", "robot", "none")
EVENT_TYPES = ("a", "b", "ab")


# ================================================================================ pure geometry (tested)


def patch_of_pixel(u, v, width: int = IMG_W, height: int = IMG_H):
    """Head token index (0..255) of full-frame pixel (u = column, v = row) — `ifg_graph._cam_pixels`."""
    u = np.asarray(u, np.int64)
    v = np.asarray(v, np.int64)
    return (v * GRID // height) * GRID + (u * GRID // width)


def patch_of_policy_pixel(u224, v224):
    """Head token index of a pixel of the 224 × 224 policy image (SigLIP 14-px patches, row-major)."""
    return (np.asarray(v224, np.int64) // PATCH_PX) * GRID + np.asarray(u224, np.int64) // PATCH_PX


def patch_box(k: int, width: int = IMG_W, height: int = IMG_H) -> tuple[int, int, int, int]:
    """``(u0, u1, v0, v1)`` (half-open) of head token k in the full frame."""
    r, c = divmod(int(k), GRID)
    return c * width // GRID, (c + 1) * width // GRID, r * height // GRID, (r + 1) * height // GRID


def backproject(depth_m: np.ndarray, K: np.ndarray, T: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Base-frame points of pixels (u, v) with metric depth (pinhole, `stages.reconstruction.backproject`)."""
    z = np.asarray(depth_m, np.float64)[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z], 1)
    return pc @ np.asarray(T, float)[:3, :3].T + np.asarray(T, float)[:3, 3]


def project(P: np.ndarray, K: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(u, v, z)`` (rounded pixel, camera depth) of base-frame points; z <= 0 -> u = v = -1."""
    P = np.asarray(P, float).reshape(-1, 3)
    T = np.asarray(T, float)
    pc = (P - T[:3, 3]) @ T[:3, :3]
    z = pc[:, 2]
    ok = z > 1e-6
    u = np.full(len(P), -1)
    v = np.full(len(P), -1)
    u[ok] = np.rint(K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]).astype(int)
    v[ok] = np.rint(K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]).astype(int)
    return u, v, z


def visible_in(P: np.ndarray, depth_m: np.ndarray, K: np.ndarray, T: np.ndarray, tol_m: float = VISIBLE_MM / 1e3):
    """Points whose projection lands in the frame and whose depth agrees within `tol_m` (`tde._visible`, 1 camera).
    Returns ``(visible, in_frame, u, v)``."""
    u, v, z = project(P, K, T)
    h, w = depth_m.shape
    inside = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    vis = np.zeros(len(u), bool)
    ins = np.flatnonzero(inside)
    dd = depth_m[v[ins], u[ins]]
    vis[ins[(dd > 0) & (np.abs(dd - z[ins]) <= tol_m)]] = True
    return vis, inside, u, v


def prev_reference(ok: Sequence[bool]) -> list[int]:
    """For each position, the last earlier position with ``ok`` (strictly before), else -1."""
    out, last = [], -1
    for i, o in enumerate(ok):
        out.append(last)
        if o:
            last = i
    return out


def ats_correct(precision: Optional[float], majority: Optional[str]) -> bool:
    return bool(precision is not None and precision >= ATS_PREC and majority == TARGET_BODY)


def event_type(has_ref: bool, moved_mm: Optional[float], state: Optional[str], vis_n: int) -> str:
    """§2 event of one pre-grasp request: "a" (moved >= 20 mm since the reference), "b" (recorded occluded while the
    head camera sees the apple), "ab" (both), "normal", or "no_ref" (no reference request yet)."""
    if not has_ref:
        return "no_ref"
    a = moved_mm is not None and moved_mm >= MOVED_MM
    b = state == "occluded" and vis_n > 0
    return "ab" if (a and b) else "a" if a else "b" if b else "normal"


def sample_normals(requests: Sequence[int], k: int, label: str, seed: int = 0) -> list[int]:
    """Up to k requests, deterministic per (label, seed)."""
    requests = list(requests)
    if k <= 0 or not requests:
        return []
    if len(requests) <= k:
        return sorted(requests)
    rng = np.random.default_rng([seed, zlib.crc32(label.encode())])
    return sorted(int(x) for x in rng.choice(requests, k, replace=False))


def patch_table(P: np.ndarray, patch: np.ndarray, cls: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(N [256] int32, S [256, 3] float64, C [256, n_class] int32)`` of points P with patch ids and class ids."""
    N = np.bincount(patch, minlength=GRID * GRID).astype(np.int32)
    S = np.stack([np.bincount(patch, weights=P[:, i], minlength=GRID * GRID) for i in range(3)], 1)
    C = np.zeros((GRID * GRID, len(CLASSES)), np.int32)
    np.add.at(C, (patch, cls), 1)
    return N, S, C


# ================================================================================ MuJoCo / record side


class SceneGT:
    """GT surface samples of one scene (`tde.scene_truth`) with class ids, plus the robot surface.

    Static geoms (no joint) are transformed and put in a KD-tree once (rebuilt only if the robot base moved more than
    1 mm / 0.1° — the samples are in the base frame); per request only the movable bodies (fruits, crate) and the robot
    surface samples within ``ROBOT_CROP_M`` of the apple are rebuilt. Robot points farther than that from the apple
    classify as ``none`` (only matters for a centre already > 50 mm off)."""

    ROBOT_CROP_M = 0.35

    def __init__(self, model_xml: str, obstacle: Sequence):
        import mujoco

        from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde

        self.gt = tde.scene_truth(model_xml, (obstacle[0], obstacle[1]))
        self.rob = tde.RobotSurface(self.gt)
        self.names = np.asarray([k[1] for k in self.gt.kind])
        self.cls = np.asarray([CLASSES.index("apple") if k[1] == TARGET_BODY else CLASSES.index(k[0])
                               for k in self.gt.kind], np.int64)
        self.movable = sorted(self.gt.movable)
        self.static = [i for i in range(len(self.gt.kind)) if i not in set(self.movable)]
        self.apple_i = [i for i, k in enumerate(self.gt.kind) if k[1] == TARGET_BODY]
        self.apple_body = mujoco.mj_name2id(self.gt.m, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
        self._static = None  # (base pose, kd, cls, names)

    @staticmethod
    def _tree(samples, idx, cls, names):
        from scipy.spatial import cKDTree

        keep = [i for i in idx if len(samples[i])]
        if not keep:
            return None, np.zeros(0, np.int64), np.zeros(0, str)
        P = np.vstack([samples[i] for i in keep])
        c = np.concatenate([np.full(len(samples[i]), cls[i]) for i in keep]).astype(np.int64)
        n = np.concatenate([np.full(len(samples[i]), names[i]) for i in keep])
        return cKDTree(P), c, n

    def _static_ok(self, pose) -> bool:
        if self._static is None:
            return False
        (R0, o0), (R1, o1) = self._static[0], pose
        ang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
        return bool(np.linalg.norm(o1 - o0) <= 1e-3 and ang <= 0.1)

    def at(self, qpos) -> dict:
        from scipy.spatial import cKDTree

        samples, _ = self.gt.at(np.asarray(qpos, float), only=self.movable)
        pose = (self.gt.base_pose[0].copy(), self.gt.base_pose[1].copy())
        if not self._static_ok(pose):
            full, _ = self.gt.at(np.asarray(qpos, float))
            self._static = (pose, *self._tree(full, self.static, self.cls, self.names))
            samples = full
        R, o = pose
        apple_c = (self.gt.d.xpos[self.apple_body] - o) @ R
        apple = np.vstack([samples[i] for i in self.apple_i])
        kd_m, cls_m, name_m = self._tree(samples, self.movable, self.cls, self.names)
        rob = self.rob.at()
        rob = rob[np.all(np.abs(rob - apple_c) <= self.ROBOT_CROP_M, axis=1)]
        return {"apple_c": apple_c, "apple": apple, "kd_apple": cKDTree(apple),
                "parts": [(self._static[1], self._static[2], self._static[3]), (kd_m, cls_m, name_m),
                          (cKDTree(rob) if len(rob) else None, np.full(len(rob), CLASSES.index("robot")),
                           np.full(len(rob), "robot"))]}

    @staticmethod
    def nearest(g: dict, P: np.ndarray, robot: bool = True,
                upper: float = np.inf) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(distance m, class id, object name)`` of the nearest GT sample (robot surface optional); farther than
        `upper` (m) -> ``(inf, none, "none")``."""
        P = np.asarray(P, float).reshape(-1, 3)
        best = np.full(len(P), np.inf)
        cls = np.full(len(P), CLASSES.index("none"), np.int64)
        name = np.full(len(P), "none", dtype=object)
        for k, (kd, c, n) in enumerate(g["parts"]):
            if kd is None or (k == 2 and not robot) or not len(P):
                continue
            d, i = kd.query(P, distance_upper_bound=upper)
            b = d < best
            best[b], cls[b], name[b] = d[b], c[i[b]], n[i[b]]
        return best, cls, name

    @staticmethod
    def classify(g: dict, P: np.ndarray) -> np.ndarray:
        """GT class id per point: nearest GT sample (objects + robot) within 8 mm, else none."""
        d, c, _ = SceneGT.nearest(g, P, upper=CLASS_MM / 1e3 + 1e-9)
        return np.where(d * 1e3 <= CLASS_MM, c, CLASSES.index("none"))


def ats_of(g: dict, tp: np.ndarray) -> dict:
    """T75 precision / majority (10 mm) and the 8 mm class majority (robot included) of target points."""
    out = {"n_target_points": int(len(tp))}
    if not len(tp):
        return out
    da = g["kd_apple"].query(tp)[0] * 1e3
    out["precision_5mm"] = round(float(np.mean(da <= PREC_MM)), 4)
    dd, _, nm = SceneGT.nearest(g, tp, robot=False, upper=MAJ_MM / 1e3 + 1e-9)
    names = np.where(dd * 1e3 <= MAJ_MM, nm.astype(str), "none")
    out["majority_object"] = collections.Counter(names.tolist()).most_common(1)[0][0]
    cls = SceneGT.classify(g, tp)
    out["class_counts_8mm"] = np.bincount(cls, minlength=len(CLASSES)).astype(int).tolist()
    out["tp_centroid"] = tp.mean(0).tolist()
    return out


def head_view(step) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(step[f"depth_{HEAD_CAM}"], np.float64) / 1000.0  # uint16 mm
    return depth, np.asarray(step[f"K_{HEAD_CAM}"], float), np.asarray(step[f"T_base_cam_{HEAD_CAM}"], float)


def appearance_patches(tp: np.ndarray, depth_m, K, T) -> tuple[list[int], str]:
    """Head patches of a reference request's target points (head-confirmed points first)."""
    if not len(tp):
        return [], "none"
    vis, inside, u, v = visible_in(tp, depth_m, K, T)
    sel, src = (vis, "visible") if vis.any() else (inside, "projected")
    if not sel.any():
        return [], "none"
    return sorted(set(patch_of_pixel(u[sel], v[sel]).tolist())), src


def patch_geometry(g: dict, depth_m, K, T, chunk) -> tuple[tuple, dict]:
    """Per-patch (N, S, C) of the head depth points left after the robot-sphere and support filters."""
    from scipy.spatial import cKDTree

    H, W = depth_m.shape
    vv, uu = np.mgrid[0:H:PIXEL_STRIDE, 0:W:PIXEL_STRIDE]
    u, v = uu.ravel(), vv.ravel()
    z = depth_m[v, u]
    ok = np.isfinite(z) & (z >= DEPTH_RANGE_M[0]) & (z <= DEPTH_RANGE_M[1])
    u, v = u[ok], v[ok]
    P = backproject(depth_m, K, T, u, v)
    info = {"n_pixels": int(len(P))}
    keep = np.ones(len(P), bool)
    if "sphere_centres" in chunk.files and len(chunk["sphere_centres"]):
        C, r = np.asarray(chunk["sphere_centres"], float), np.asarray(chunk["sphere_radii"], float)
        hit = cKDTree(P).query_ball_point(C, r + ROBOT_MARGIN_M)
        idx = np.unique(np.concatenate([np.asarray(h, np.int64) for h in hit])) if len(hit) else np.zeros(0, np.int64)
        keep[idx] = False
    info["n_robot_dropped"] = int((~keep).sum())
    if "support_mask" in chunk.files and chunk["support_mask"].any():
        sup = np.asarray(chunk["raw_cloud"], float)[np.asarray(chunk["support_mask"], bool)]
        d = cKDTree(sup).query(P[keep], distance_upper_bound=SUPPORT_M)[0]
        idx = np.flatnonzero(keep)
        keep[idx[np.isfinite(d)]] = False
        info["n_support_dropped"] = int(np.isfinite(d).sum())
    P, u, v = P[keep], u[keep], v[keep]
    info["n_kept"] = int(len(P))
    cls = SceneGT.classify(g, P)
    return patch_table(P, patch_of_pixel(u, v, W, H), cls), info


def _sorted_steps(client_dir: str) -> list[str]:
    from AG3S.ag3s.experiments.studies.t77_labels import _step_tsteps

    return [f for _, f in _step_tsteps(pathlib.Path(client_dir, "ag3s", "run_0000"))]


def events_run(scene: SceneGT, entry: dict, lab: dict, normal_per_run: int, seed: int = 0,
               event_cap: int = 0) -> tuple[dict, list]:
    """Rows of one run + the per-patch tables ``[(request, N, S, C, attention)]`` of its scored requests."""
    steps = _sorted_steps(entry["client_dir"])
    held = {r["request"]: bool(r["gt"]["held"]) for r in lab["rows"] if "gt" in r}
    first_held = min((j for j, h in held.items() if h), default=None)
    n_req = min(len(entry["chunks"]), len(steps))
    pre = [j for j in range(n_req) if j in held and (first_held is None or j < first_held)]
    rows = []
    for j in pre:
        with np.load(steps[j]) as st:
            qpos = np.asarray(st["qpos"], float)
            depth, K, T = head_view(st)
            t_step = int(st["t_step"])
        with np.load(entry["chunks"][j], allow_pickle=True) as z:
            summ = json.loads(str(z["summary_json"]))
            tp = np.asarray(z["target_points"], float) if "target_points" in z.files else np.zeros((0, 3))
        g = scene.at(qpos)
        vis, _, _, _ = visible_in(g["apple"], depth, K, T)
        man = summ.get("manipulated") or {}
        row = {"request": j, "t_step": t_step, "apple_c": g["apple_c"].tolist(), "vis_n": int(vis.sum()),
               "vis_c": g["apple"][vis].mean(0).tolist() if vis.any() else None,
               "state": man.get("state"), "centroid": man.get("centroid"), "has_target": bool(summ.get("has_target"))}
        row.update(ats_of(g, tp))
        row["ats_ok"] = ats_correct(row.get("precision_5mm"), row.get("majority_object"))
        if row["ats_ok"]:
            row["appearance_patches"], row["appearance_source"] = appearance_patches(tp, depth, K, T)
        rows.append(row)
    refs = prev_reference([r["ats_ok"] for r in rows])
    for r, ri in zip(rows, refs):
        r["ref"] = rows[ri]["request"] if ri >= 0 else -1
        r["moved_mm"] = (round(float(np.linalg.norm(np.subtract(r["apple_c"], rows[ri]["apple_c"])) * 1e3), 2)
                         if ri >= 0 else None)
        r["event"] = event_type(ri >= 0, r["moved_mm"], r["state"], r["vis_n"])
    normals = sample_normals([r["request"] for r in rows if r["event"] == "normal"], normal_per_run, entry["label"], seed)
    ev_all = [r["request"] for r in rows if r["event"] in EVENT_TYPES]
    events = set(ev_all) if event_cap <= 0 else set(sample_normals(ev_all, event_cap, "event:" + entry["label"], seed))
    tables = []
    for r in rows:
        r["sampled"] = r["request"] in events or r["request"] in normals
        if not r["sampled"]:
            continue
        with np.load(steps[r["request"]]) as st:  # second pass: the GT KD-trees are not kept per request
            g = scene.at(np.asarray(st["qpos"], float))
            depth, K, T = head_view(st)
        with np.load(entry["chunks"][r["request"]], allow_pickle=True) as z:
            (N, S, C), info = patch_geometry(g, depth, K, T, z)
            att = (np.asarray(z["attention_head"], np.float32).reshape(-1) if "attention_head" in z.files
                   else np.full(GRID * GRID, np.nan, np.float32))
        r["geom"] = info
        tables.append((r["request"], N, S, C, att))
    need = sorted({r["request"] for r in rows if r["sampled"]} | {r["ref"] for r in rows if r["sampled"] and r["ref"] >= 0})
    out = {k: entry.get(k) for k in ("label", "client_dir", "kit", "config", "scene", "scene_number", "split",
                                     "stratum", "episode", "seed", "side", "obstacle")}
    out.update({"n_requests": n_req, "first_held": first_held, "n_pre": len(pre), "rows": rows,
                "patch_requests": need, "event_counts": dict(collections.Counter(r["event"] for r in rows))})
    return out, tables


# ================================================================================ driver


def _worker(args) -> tuple[dict, list]:
    entries, labels, model_xml, normal_per_run, seed, event_cap = args
    scene, scene_key = None, None
    res = []
    for e in entries:  # entries come grouped by obstacle configuration: keep one scene GT at a time (~1.5 GiB each)
        key = tuple(e["obstacle"])
        if key != scene_key:
            scene, scene_key = None, None
            scene, scene_key = SceneGT(model_xml, key), key
        run, tables = events_run(scene, e, labels[e["label"]], normal_per_run, seed, event_cap)
        print(f"{e['label']}: pre {run['n_pre']}, events {run['event_counts']}", flush=True)
        res.append((run, tables))
    return res


def summarise(runs: Sequence[dict]) -> dict:
    out: dict[str, Any] = {}
    for split in ("train", "test"):
        rs = [r for r in runs if r["split"] == split]
        rows = [x for r in rs for x in r["rows"]]
        ev = collections.Counter(x["event"] for x in rows)
        out[split] = {"runs": len(rs), "pre_grasp_requests": len(rows), "events": dict(ev),
                      "events_a_or_ab": ev.get("a", 0) + ev.get("ab", 0), "events_b_or_ab": ev.get("b", 0) + ev.get("ab", 0),
                      "events_total": sum(ev.get(k, 0) for k in EVENT_TYPES),
                      "events_without_head_visible_gt": sum(1 for x in rows if x["event"] in EVENT_TYPES and not x["vis_n"]),
                      "sampled_normals": sum(1 for x in rows if x["event"] == "normal" and x["sampled"]),
                      "sampled_events": sum(1 for x in rows if x["event"] in EVENT_TYPES and x["sampled"]),
                      "recorded_states": dict(collections.Counter(str(x["state"]) for x in rows)),
                      "patch_requests": sum(len(r["patch_requests"]) for r in rs)}
    return out


def build_events(out: pathlib.Path, spec_path: pathlib.Path, labels_path: pathlib.Path, model_xml: str,
                 only: Optional[list] = None, normal_per_run: int = 4, seed: int = 0, jobs: int = 1,
                 event_cap: int = 0) -> dict:
    spec = json.loads(spec_path.read_text())
    entries = spec["runs"] if isinstance(spec, dict) else spec
    if only:
        entries = [e for e in entries if e["label"] in only]
    labels = {r["label"]: r for r in json.loads(labels_path.read_text())["runs"]}
    missing = [e["label"] for e in entries if e["label"] not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} spec runs have no T77 labels, e.g. {missing[:3]}")
    labs = {e["label"]: labels[e["label"]] for e in entries}
    # one work list per obstacle configuration, spread over the workers (scene GT built once per worker and scene)
    by_scene = collections.defaultdict(list)
    for e in entries:
        by_scene[tuple(e["obstacle"])].append(e)
    jobs = max(1, min(int(jobs), 4))
    buckets = [[] for _ in range(jobs)]
    for i, (_, es) in enumerate(sorted(by_scene.items(), key=lambda kv: -len(kv[1]))):
        min(buckets, key=len).extend(es)
    work = [(b, {e["label"]: labs[e["label"]] for e in b}, model_xml, normal_per_run, seed, event_cap)
            for b in buckets if b]
    if jobs == 1:
        parts = [_worker(w) for w in work]
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(len(work)) as pool:
            parts = pool.map(_worker, work)
    order = {e["label"]: i for i, e in enumerate(entries)}
    results = sorted((x for p in parts for x in p), key=lambda x: order[x[0]["label"]])
    runs = [r for r, _ in results]
    G_run, G_req, Ns, Ss, Cs, As = [], [], [], [], [], []
    for ri, (run, tables) in enumerate(results):
        gidx = {}
        for req, N, S, C, att in tables:
            gidx[req] = len(G_req)
            G_run.append(ri)
            G_req.append(req)
            Ns.append(N)
            Ss.append(S)
            Cs.append(C)
            As.append(att)
        for x in run["rows"]:
            if x["request"] in gidx:
                x["geom_index"] = gidx[x["request"]]
    geom = out.with_suffix(".geom.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(geom, run=np.asarray(G_run, np.int32), request=np.asarray(G_req, np.int32),
                        N=np.asarray(Ns, np.int32).reshape(-1, GRID * GRID),
                        S=np.asarray(Ss, np.float64).reshape(-1, GRID * GRID, 3),
                        C=np.asarray(Cs, np.int32).reshape(-1, GRID * GRID, len(CLASSES)),
                        attention=np.asarray(As, np.float32).reshape(-1, GRID * GRID),
                        labels=np.asarray([r["label"] for r in runs]), classes=np.asarray(CLASSES))
    res = {"step": "T79", "spec": str(spec_path), "labels": str(labels_path), "model_xml": model_xml,
           "geom": str(geom), "definitions": __doc__,
           "params": {"prec_mm": PREC_MM, "maj_mm": MAJ_MM, "ats_prec": ATS_PREC, "moved_mm": MOVED_MM,
                      "visible_mm": VISIBLE_MM, "class_mm": CLASS_MM, "pixel_stride": PIXEL_STRIDE,
                      "depth_range_m": DEPTH_RANGE_M, "robot_margin_m": ROBOT_MARGIN_M, "support_m": SUPPORT_M,
                      "normal_per_run": normal_per_run, "event_cap": event_cap, "seed": seed, "classes": CLASSES, "grid": GRID,
                      "image_wh": [IMG_W, IMG_H], "policy_px": POLICY_PX},
           "summary": summarise(runs),
           "patch_requests": {r["label"]: r["patch_requests"] for r in runs},
           "runs": runs}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=0, default=float))
    return res


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model-xml", default=None)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--normal-per-run", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event-cap", type=int, default=0,
                    help="scored event requests per run, sampled (seeded); 0 = all (default, task §2)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (<= 4)")
    a = ap.parse_args(argv)
    res = build_events(pathlib.Path(a.out), pathlib.Path(a.spec), pathlib.Path(a.labels), a.model_xml or default_xml(),
                       a.only, a.normal_per_run, a.seed, a.jobs, a.event_cap)
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    main()
