"""T51 — grasp-event self label: a target label for head selection without human annotation.

Idea (`T51.task.md` "플랫폼별 label"): when the gripper closes on the object, the fingertip centre *is*
the object's position. If the object did not move before the grasp, projecting that 3D point into the
cameras of the **earlier** frames gives the target's pixel location there — on a real robot from
forward kinematics + the `grasp_latch` HELD event, in sim from a physical predicate. This module builds
that label from a recorded 16D rollout and measures it against the MuJoCo GT.

Steps (`build_selflabel`, MuJoCo `mj_forward` only — nothing is rendered):

1. **Grasp event.** Per recorded frame (posed with `policy_record.pose_scene`, the T47 replay pose),
   the bodies in contact with the target. ``both_contact`` = both finger bodies of one hand touch it;
   ``held`` = both fingers and neither table nor crate (the T48b predicate, `t48b_scene.py:40-44,
   330-343`, re-implemented here so T51 does not depend on a module that is still being written).
   Default event (``both_contact``): the **first frame where both fingers of a hand touch the target,
   provided the target is later ``held`` by that hand** (a grasp that lifted it). Variant ``held``:
   the first ``held`` frame (later, after the lift; closer to what a HELD latch reports).
2. **Fingertip centre** at the event frame: the midpoint of the two pads of that hand — each pad is
   the finger body's largest collision geom (the `t48b_scene.pad_midpoint` definition, `:571`). The
   two fingers of a hand are used symmetrically, so the MJCF/URDF left-finger name swap (T43 OC) does
   not matter. **Object radius** from the same pads: half the pad-centre separation minus the pad half
   thickness, clipped to [1.5, 6] cm (a real gripper reports this as its opening width).
3. **Projection** of the world point into every frame *before* the event (``labeled``): world -> base
   (the base may drive) -> each camera with `TransportScene.intrinsics` / `T_base_cam` (OpenCV, 480x640,
   the frame the GT cache and the attention lifting use). The label is a disc of the projected object
   radius (``f r / z``) rasterised at stride 2 and pooled to the 16x16 attention grid (``self_cov``);
   ``self_px`` = its in-image area at 480x640.
4. **3D label** for the 3D score: the grasp point in each labeled frame's base frame
   (``grasp_point_base``) with ball radius ``r_obj + SELF_BALL_MARGIN`` (`self_identity`).
5. **GT check** (sim only): projection error to the GT target pixel centroid (T47 GT cache
   ``label_small``), whether the projected pixel / patch lies on the target, and how far the target
   body moved between each labeled frame and the event (``target_disp_m``) — the "stationary before the
   grasp" assumption measured.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

#: Finger bodies per hand (MJCF names; used as an unordered pair).
FINGER_BODIES = {"left": ("ee_finger_l1", "ee_finger_l2"), "right": ("ee_finger_r1", "ee_finger_r2")}
HANDS = ("left", "right")
SUPPORT_BODIES = ("table", "crate")
GRASP_EVENTS = ("both_contact", "held")
DEFAULT_EVENT = "both_contact"
#: Object radius clip (m): a parallel gripper with 2 x 5 cm stroke cannot hold more than ~5 cm radius.
R_OBJ_MIN, R_OBJ_MAX = 0.015, 0.06
#: 3D self-label ball = object radius + this margin (m).
SELF_BALL_MARGIN = 0.02
#: The frame the label is drawn in (= GT cache render, `t47_replay.extract_gt` 480x640).
RENDER_HW = (480, 640)
#: Raster stride (2 -> 240x320, i.e. 15x20 samples per 30x40-pixel patch; the GT cache `label_small`
#: uses the same stride).
LABEL_STRIDE = 2
GRID = 16
#: The 2D minimum area, as T47 (`--min-target-px 200` at 480x640).
MIN_SELF_PX = 200
#: Points closer than this to the camera plane are not projected (m).
MIN_DEPTH_M = 0.02
#: Target displacement (m) above which a labeled frame counts as "moved before the grasp".
MOTION_TOL_M = 0.01
#: `t47_head_metrics.PATCH_HIT_COV`: a patch "is" the target when the target covers > 5 % of it.
PATCH_HIT_COV = 0.05
#: Class index of the target in the GT cache (`t47_head_metrics.CLASS_NAMES[0]`).
GT_TARGET_CLASS = 0


# ----------------------------------------------------------------------------- pure pieces


def contact_state(touching: set, fingers: Mapping[str, tuple], support: set
                  ) -> tuple[Optional[str], Optional[str]]:
    """``(both_hand, held_hand)`` from the set of body ids touching the target.

    ``both_hand`` = the hand whose two finger bodies both touch it (left checked first); ``held_hand``
    = that hand when no support body (table, crate) touches it, else None.
    """
    for h in HANDS:
        if all(b in touching for b in fingers[h]):
            return h, (None if touching & set(support) else h)
    return None, None


def detect_grasp(both: Sequence[Optional[str]], held: Sequence[Optional[str]], *,
                 event: str = DEFAULT_EVENT) -> Optional[tuple[int, str]]:
    """The grasp event ``(frame index, hand)`` or None (module docstring, step 1)."""
    if event not in GRASP_EVENTS:
        raise ValueError(f"event must be one of {GRASP_EVENTS}")
    n = len(both)
    if len(held) != n:
        raise ValueError("both / held lengths differ")
    if event == "held":
        for i in range(n):
            if held[i] is not None:
                return i, held[i]
        return None
    for i in range(n):
        h = both[i]
        if h is not None and any(held[j] == h for j in range(i, n)):
            return i, h
    return None


def object_radius(p1: np.ndarray, p2: np.ndarray, pad_half_thickness: float) -> float:
    """Half the pad-centre separation minus the pad half thickness, clipped (module docstring)."""
    d = float(np.linalg.norm(np.asarray(p1, float) - np.asarray(p2, float)))
    return float(np.clip(0.5 * d - float(pad_half_thickness), R_OBJ_MIN, R_OBJ_MAX))


def to_frame(T_world_frame: np.ndarray, p_world: np.ndarray) -> np.ndarray:
    """A world point in the frame whose world pose is ``T_world_frame`` (4x4)."""
    T = np.asarray(T_world_frame, float)
    return T[:3, :3].T @ (np.asarray(p_world, float) - T[:3, 3])


def project(p_base: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> tuple[float, float, float]:
    """``(u, v, z)``: pixel (OpenCV: u right, v down, origin top-left corner) and depth of a base-frame
    point for a camera with intrinsics `K` and pose `T_base_cam` (OpenCV camera, +z forward)."""
    T = np.asarray(T_base_cam, float)
    pc = T[:3, :3].T @ (np.asarray(p_base, float) - T[:3, 3])
    z = float(pc[2])
    if z <= MIN_DEPTH_M:
        return float("nan"), float("nan"), z
    Km = np.asarray(K, float)
    return (float(Km[0, 0] * pc[0] / z + Km[0, 2]), float(Km[1, 1] * pc[1] / z + Km[1, 2]), z)


def disc_coverage(u: float, v: float, r_px: float, *, hw: tuple[int, int] = RENDER_HW,
                  stride: int = LABEL_STRIDE, grid: int = GRID) -> tuple[np.ndarray, int]:
    """``(coverage (grid, grid), area px)`` of a disc at pixel ``(u, v)`` with radius `r_px` in an
    ``hw`` image: pixel centres sampled every `stride` px, the in-disc share of each patch's samples,
    and the in-image disc area (samples x stride^2). Non-finite input -> zeros."""
    h, w = int(hw[0]), int(hw[1])
    if h % (grid * stride) or w % (grid * stride):
        raise ValueError(f"{hw} is not divisible into {grid}x{grid} patches at stride {stride}")
    cov = np.zeros((grid, grid), np.float64)
    if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(r_px)) or r_px <= 0:
        return cov, 0
    ys = (np.arange(h // stride) + 0.5) * stride
    xs = (np.arange(w // stride) + 0.5) * stride
    inside = ((xs[None, :] - u) ** 2 + (ys[:, None] - v) ** 2) <= float(r_px) ** 2
    ph, pw = h // stride // grid, w // stride // grid
    cov = inside.reshape(grid, ph, grid, pw).mean(axis=(1, 3))
    return cov, int(inside.sum()) * stride * stride


def gt_centroid(label_small: np.ndarray, *, stride: int = LABEL_STRIDE,
                cls: int = GT_TARGET_CLASS) -> tuple[float, float, int]:
    """``(u, v, area px at full resolution)`` of the GT class pixels in a strided class image."""
    lab = np.asarray(label_small)
    ys, xs = np.nonzero(lab == cls)
    if not len(xs):
        return float("nan"), float("nan"), 0
    return (float(((xs + 0.5) * stride).mean()), float(((ys + 0.5) * stride).mean()),
            int(len(xs)) * stride * stride)


def patch_of(u: float, v: float, *, hw: tuple[int, int] = RENDER_HW, grid: int = GRID
             ) -> Optional[tuple[int, int]]:
    """The 16x16 patch ``(row, col)`` of an in-image pixel, else None."""
    h, w = hw
    if not (np.isfinite(u) and np.isfinite(v)) or not (0 <= u < w and 0 <= v < h):
        return None
    return int(v / h * grid), int(u / w * grid)


def self_identity(points: np.ndarray, centre: np.ndarray, radius: float) -> dict[str, Any]:
    """3D self-label identity of a chosen cluster: points inside the self ball vs outside.

    ``self_identity_correct`` = strictly more points inside than outside (the binary analogue of the
    GT plurality rule — the self label has only "target" and "not target"); empty cluster -> False.
    """
    p = np.asarray(points, np.float64).reshape(-1, 3)
    c = np.asarray(centre, np.float64).reshape(3)
    if not len(p) or not np.all(np.isfinite(c)):
        return {"self_in_n": 0, "self_out_n": int(len(p)), "self_identity_correct": False,
                "self_centroid_dist_m": float("nan")}
    d = np.linalg.norm(p - c, axis=1)
    n_in = int((d <= float(radius)).sum())
    return {"self_in_n": n_in, "self_out_n": int(len(p) - n_in),
            "self_identity_correct": bool(n_in > len(p) - n_in),
            "self_centroid_dist_m": float(np.linalg.norm(p.mean(axis=0) - c))}


# ----------------------------------------------------------------------------- MuJoCo pieces


def _bid(model, name: str) -> int:
    import mujoco

    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if b < 0:
        raise KeyError(f"body {name!r} missing from the model")
    return int(b)


def touching_bodies(model, data, body: int) -> set:
    """Body ids in contact with `body` (any contact listed by `mj_forward`)."""
    out = set()
    gb = model.geom_bodyid
    for i in range(int(data.ncon)):
        c = data.contact[i]
        b1, b2 = int(gb[c.geom1]), int(gb[c.geom2])
        if b1 == body:
            out.add(b2)
        elif b2 == body:
            out.add(b1)
    return out


def pad_geom(model, body: int) -> int:
    """The finger body's largest collision geom (by bounding-box volume) — the pad."""
    gs = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body
          and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not gs:
        raise ValueError(f"body {body} has no collision geom")
    return max(gs, key=lambda k: float(np.prod(model.geom_size[k])))


def build_selflabel(record_path, *, target: str, event: str = DEFAULT_EVENT,
                    gt: Optional[Mapping[str, np.ndarray]] = None,
                    log: Callable[[str], None] = print) -> dict[str, Any]:
    """The self label of one record (module docstring) as arrays for `np.savez_compressed`.

    `gt` = the record's T47 GT cache (``label_small``, ``class_cov``) for the GT check; None skips it.
    """
    from AG3S.ag3s.experiments.sources.policy_record import (
        CAMERA_BINDINGS, POLICY_CAMERA_NAMES, load_run, pose_scene, provenance_arrays, replay_scene)

    record = load_run(record_path)
    scene = replay_scene(record, height=RENDER_HW[0], width=RENDER_HW[1])
    model, data = scene.model, scene.data
    cams = [CAMERA_BINDINGS[c][0] for c in POLICY_CAMERA_NAMES]
    tb = _bid(model, target)
    fingers = {h: tuple(_bid(model, n) for n in FINGER_BODIES[h]) for h in HANDS}
    support = {_bid(model, n) for n in SUPPORT_BODIES}
    pads = {h: tuple(pad_geom(model, b) for b in fingers[h]) for h in HANDS}
    pad_half = {h: float(min(np.min(model.geom_size[g]) for g in pads[h])) for h in HANDS}
    n_f, n_c = len(record), len(cams)
    both, held = [], []
    supp = np.zeros(n_f, bool)
    T_wb = np.zeros((n_f, 4, 4))
    t_world = np.zeros((n_f, 3))
    pad_pts = {h: np.zeros((n_f, 2, 3)) for h in HANDS}
    K = np.zeros((n_f, n_c, 3, 3))
    T_bc = np.zeros((n_f, n_c, 4, 4))
    try:
        for i, step in enumerate(record):
            pose_scene(scene, step)
            touch = touching_bodies(model, data, tb)
            bh, hh = contact_state(touch, fingers, support)
            both.append(bh)
            held.append(hh)
            supp[i] = bool(touch & support)
            T_wb[i] = scene.body_pose("base")
            t_world[i] = data.xpos[tb]
            for h in HANDS:
                pad_pts[h][i] = [data.geom_xpos[g] for g in pads[h]]
            for ci, cam in enumerate(cams):
                K[i, ci] = scene.intrinsics(cam)
                T_bc[i, ci] = scene.T_base_cam(cam)
    finally:
        scene.close()

    ev = detect_grasp(both, held, event=event)
    labeled = np.zeros(n_f, bool)
    gp_world = np.full(3, np.nan)
    gp_base = np.full((n_f, 3), np.nan)
    r_obj = float("nan")
    uvz = np.full((n_f, n_c, 3), np.nan)
    r_px = np.full((n_f, n_c), np.nan)
    self_cov = np.zeros((n_f, n_c, GRID, GRID), np.float32)
    self_px = np.zeros((n_f, n_c), np.int64)
    g_idx, g_hand, g_t = -1, "", -1
    if ev is not None:
        g_idx, g_hand = ev
        g_t = int(record.steps[g_idx].t_step)
        p1, p2 = pad_pts[g_hand][g_idx]
        gp_world = 0.5 * (p1 + p2)
        r_obj = object_radius(p1, p2, pad_half[g_hand])
        labeled[:g_idx] = True
        for i in range(n_f):
            gp_base[i] = to_frame(T_wb[i], gp_world)
            if not labeled[i]:
                continue
            for ci in range(n_c):
                u, v, z = project(gp_base[i], K[i, ci], T_bc[i, ci])
                uvz[i, ci] = (u, v, z)
                if np.isfinite(u):
                    r_px[i, ci] = K[i, ci, 0, 0] * r_obj / z
                    cov, px = disc_coverage(u, v, r_px[i, ci])
                    self_cov[i, ci], self_px[i, ci] = cov, px

    # --- GT check
    err = np.full((n_f, n_c), np.nan)
    gt_px = np.zeros((n_f, n_c), np.int64)
    hit_pixel = np.full((n_f, n_c), np.nan)
    hit_patch = np.full((n_f, n_c), np.nan)
    if gt is not None:
        lab = np.asarray(gt["label_small"])
        cov_t = np.asarray(gt["class_cov"])[:, :, GT_TARGET_CLASS]
        stride = int(np.asarray(gt.get("label_stride", LABEL_STRIDE)))
        if lab.shape[:2] != (n_f, n_c):
            raise ValueError(f"GT cache {lab.shape[:2]} does not match the record ({n_f}, {n_c})")
        for i in range(n_f):
            for ci in range(n_c):
                cu, cv, px = gt_centroid(lab[i, ci], stride=stride)
                gt_px[i, ci] = px
                if not labeled[i]:
                    continue
                u, v = uvz[i, ci, 0], uvz[i, ci, 1]
                pc = patch_of(u, v)
                if pc is None:
                    continue
                if px >= MIN_SELF_PX:
                    err[i, ci] = float(np.hypot(u - cu, v - cv))
                hit_pixel[i, ci] = float(lab[i, ci, int(v // stride), int(u // stride)]
                                         == GT_TARGET_CLASS)
                hit_patch[i, ci] = float(cov_t[i, ci, pc[0], pc[1]] > PATCH_HIT_COV)
    disp = (np.linalg.norm(t_world - t_world[g_idx], axis=1) if g_idx >= 0
            else np.full(n_f, np.nan))
    offset = float(np.linalg.norm(gp_world - t_world[g_idx])) if g_idx >= 0 else float("nan")
    log(f"  selflabel: event={event} frame={g_idx} hand={g_hand or '-'} r_obj={r_obj:.3f} "
        f"labeled={int(labeled.sum())}/{n_f}")
    code = {None: -1, "left": 0, "right": 1}
    return {
        "t_step": np.asarray([s.t_step for s in record], np.int64),
        "target": np.asarray(target), "event": np.asarray(event),
        "cameras": np.asarray(list(POLICY_CAMERA_NAMES)), "mujoco_cameras": np.asarray(cams),
        "render_hw": np.asarray(RENDER_HW), "label_stride": np.asarray(LABEL_STRIDE),
        "both_contact_hand": np.asarray([code[b] for b in both], np.int8),
        "held_hand": np.asarray([code[h] for h in held], np.int8),
        "support_contact": supp,
        "grasp_index": np.asarray(g_idx), "grasp_hand": np.asarray(g_hand),
        "grasp_t_step": np.asarray(g_t),
        "grasp_point_world": gp_world, "grasp_point_base": gp_base,
        "r_obj": np.asarray(r_obj), "r_ball": np.asarray(r_obj + SELF_BALL_MARGIN),
        "labeled": labeled, "uvz": uvz, "r_px": r_px,
        "self_cov": self_cov, "self_px": self_px,
        "target_world": t_world, "target_disp_m": disp, "grasp_offset_m": np.asarray(offset),
        "T_world_base": T_wb, "K": K, "T_base_cam": T_bc,
        "gt_checked": np.asarray(gt is not None), "proj_err_px": err, "gt_target_px": gt_px,
        "hit_pixel": hit_pixel, "hit_patch": hit_patch,
        **provenance_arrays(record, n_f),
    }


def summarize_selflabel(arr: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Per-record numbers of one `build_selflabel` output (no averaging across records)."""
    lab = np.asarray(arr["labeled"], bool)
    cams = [str(c) for c in arr["cameras"]]
    g = int(arr["grasp_index"])
    out: dict[str, Any] = {
        "event": str(arr["event"]), "grasp_index": g, "grasp_hand": str(arr["grasp_hand"]),
        "grasp_t_step": int(arr["grasp_t_step"]), "n_frames": int(lab.size),
        "n_labeled": int(lab.sum()), "r_obj_m": float(arr["r_obj"]),
        "r_ball_m": float(arr["r_ball"]), "grasp_offset_m": float(arr["grasp_offset_m"]),
    }
    if g < 0:
        out["note"] = "no grasp event in this record -> no self label"
        return out
    disp = np.asarray(arr["target_disp_m"], float)[lab]
    out["target_disp_m"] = {"max": float(disp.max()) if disp.size else float("nan"),
                            "median": float(np.median(disp)) if disp.size else float("nan"),
                            "n_frames_gt_tol": int((disp > MOTION_TOL_M).sum()),
                            "tol_m": MOTION_TOL_M}
    per_cam = {}
    for ci, c in enumerate(cams):
        e = np.asarray(arr["proj_err_px"], float)[lab, ci]
        e = e[np.isfinite(e)]
        hp = np.asarray(arr["hit_pixel"], float)[lab, ci]
        hq = np.asarray(arr["hit_patch"], float)[lab, ci]
        px = np.asarray(arr["self_px"])[lab, ci]
        per_cam[c] = {
            "n_labeled_in_image": int(np.isfinite(hp).sum()),
            "n_self_px_ge_min": int((px >= MIN_SELF_PX).sum()),
            "n_err": int(e.size),
            "err_px_median": float(np.median(e)) if e.size else float("nan"),
            "err_px_p90": float(np.percentile(e, 90)) if e.size else float("nan"),
            "err_px_max": float(e.max()) if e.size else float("nan"),
            "hit_pixel_rate": float(np.nanmean(hp)) if np.isfinite(hp).any() else float("nan"),
            "hit_patch_rate": float(np.nanmean(hq)) if np.isfinite(hq).any() else float("nan"),
        }
    out["per_camera"] = per_cam
    return out


__all__ = ["DEFAULT_EVENT", "FINGER_BODIES", "GRASP_EVENTS", "MIN_SELF_PX", "MOTION_TOL_M",
           "RENDER_HW", "R_OBJ_MAX", "R_OBJ_MIN", "SELF_BALL_MARGIN", "build_selflabel",
           "contact_state", "detect_grasp", "disc_coverage", "gt_centroid", "object_radius",
           "pad_geom", "patch_of", "project", "self_identity", "summarize_selflabel", "to_frame",
           "touching_bodies"]
