"""T47 — the MuJoCo replay half: GT patch coverage (I2) and the 3D end-to-end grounding run (I4).

Nothing here is imported by the pure 2D analysis (`t47_head_metrics`), so `--stage select` and
`--stage validate` never touch MuJoCo. Both functions below **render**; they are run only by the
verifier after user approval (T47.task.md "실행 승인").

GT (`extract_gt`)
    `policy_record.load_run · replay_scene · pose_scene · patch_coverage` — the same calls as
    `attention_report.ground_truth` (`AG3S/ag3s/experiments/reports/attention_report.py:94-112`),
    but for **all three** policy cameras (`policy_record.CAMERA_BINDINGS`) and for body *classes*
    (`body_class_table`) instead of a fixed body list. One replay per record; the caller caches it.

E2E (`run_e2e_record`)
    The live serving path, call for call:
      client  `wire.pack_request`  (depth as uint16 mm, K, T_base_cam, robot_state, stamp per camera)
      server  `wire.strip_request` -> `wire.unpack_camera_observations(attention={camera: map})`
              (`serve_safe.attention_extractor` maps cam_high/cam_left_wrist/cam_right_wrist to
              zed_left/wrist_cam_l/wrist_cam_r) -> `AG3S.process_multi_debug` (fusion -> lifting
              -> grounding), with the AG3S built by `serve_safe.build_ag3s` at the server's defaults.
    The same sequence `exclusion_e2.run_frame` uses (`AG3S/ag3s/experiments/live/exclusion_e2.py:
    665-678`). Only the attention source changes between runs; each source has its own AG3S
    instance so cross-frame state (TargetConfirm) stays per source.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies.t47_head_metrics import CLASS_NAMES

#: Fruits in the transport scene (the prompt names one; the rest are `other_object`).
FRUITS = ("apple", "banana", "orange", "pear")
DESTINATION_BODIES = ("crate",)
SUPPORT_BODIES = ("table", "shelf")
#: The gripper is every body **kinematically below** these links (wrist bracket, wrist camera,
#: FT sensor, EE body, both fingers) — found from `body_parentid`, not from name prefixes, so the
#: MJCF/URDF left-finger name swap (T43 OC) cannot matter: both fingers land in the same class.
GRIPPER_ROOT_LINKS = ("link_left_arm_6", "link_right_arm_6")
#: Named bodies whose own coverage is kept next to the class coverage (figures / sanity).
NAMED_BODIES = ("crate", "apple", "banana", "orange", "pear", "table", "shelf")
#: `CLASS_NAMES` index of each class.
CLS = {name: i for i, name in enumerate(CLASS_NAMES)}
#: Policy render: 299x224 (4:3), reflections off, bilinear resize to 224x224
#: (`integration/rby1/manipulation/src/rby1_manipulation/data/recording.py:27-55`).
POLICY_SOURCE_HW = (224, 299)
POLICY_IMAGE_SIZE = 224


def _body_names(model) -> list[str]:
    import mujoco

    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(model.nbody)]


def gripper_body_ids(model) -> set[int]:
    """Bodies strictly below `GRIPPER_ROOT_LINKS` in the kinematic tree."""
    import mujoco

    roots = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in GRIPPER_ROOT_LINKS}
    if any(r < 0 for r in roots):
        raise KeyError(f"model lacks one of {GRIPPER_ROOT_LINKS}")
    out = set()
    for bid in range(model.nbody):
        j = bid
        while j > 0:
            j = int(model.body_parentid[j])
            if j in roots:
                out.add(bid)
                break
    return out


def body_class_table(model, target: str) -> tuple[np.ndarray, dict[str, list[str]]]:
    """``(lut, members)``: class index per body id (`CLASS_NAMES`), and which bodies each class got.

    Rules, first match wins:
      mocap body (IK target markers)        -> background
      the prompt's target                   -> target
      crate                                 -> destination
      another fruit                         -> other_object
      below link_{left,right}_arm_6         -> robot_gripper
      `mujoco_source.is_robot_body`         -> robot
      table, shelf                          -> support
      anything else (world floor, office)   -> background
    Pixels that hit nothing (body id -1) are background too.
    """
    from AG3S.ag3s.experiments.sources.mujoco_source import is_robot_body

    names = _body_names(model)
    if target not in names:
        raise KeyError(f"target body {target!r} not in the model")
    grip = gripper_body_ids(model)
    lut = np.full(model.nbody, CLS["background"], np.int16)
    members: dict[str, list[str]] = {c: [] for c in CLASS_NAMES}
    for bid, name in enumerate(names):
        if int(model.body_mocapid[bid]) >= 0:
            cls = "background"
        elif name == target:
            cls = "target"
        elif name in DESTINATION_BODIES:
            cls = "destination"
        elif name in FRUITS:
            cls = "other_object"
        elif bid in grip:
            cls = "robot_gripper"
        elif is_robot_body(name):
            cls = "robot"
        elif name in SUPPORT_BODIES:
            cls = "support"
        else:
            cls = "background"
        lut[bid] = CLS[cls]
        members[cls].append(name)
    return lut, members


def class_map(body_ids: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Body-id image -> class-index image (``-1`` -> background)."""
    ids = np.asarray(body_ids)
    out = np.full(ids.shape, CLS["background"], np.int16)
    hit = ids >= 0
    out[hit] = lut[ids[hit]]
    return out


def render_policy_rgb(model, data, renderer, camera: str) -> np.ndarray:
    """What the policy camera rendered: 299x224, no reflections, bilinear to 224x224."""
    import mujoco
    from PIL import Image

    renderer.update_scene(data, camera=camera)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    img = renderer.render()
    return np.asarray(Image.fromarray(img).resize((POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE),
                                                  resample=Image.Resampling.BILINEAR))


# ---------------------------------------------------------------------------------- I2: GT


def extract_gt(record_path, *, target: str, height: int = 480, width: int = 640,
               rgb_check_frames: int = 3, label_stride: int = 2,
               log: Callable[[str], None] = print) -> dict[str, Any]:
    """GT for every (frame, policy camera) of one record, as arrays ready for `np.savez_compressed`.

    Keys: ``class_cov`` (F, 3, K, 16, 16) per-patch class share (sums to 1 over K), ``class_px``
    (F, 3, K) unoccluded pixels per class at ``height x width``, ``body_cov`` / ``body_px`` for
    `NAMED_BODIES`, ``label_small`` (F, 3, H/stride, W/stride) class image for figures,
    ``target_pos_base`` (F, 3), ``fruit_pos_base`` (F, 4, 3), the RGB self-check
    (``rgb_check_index``, ``rgb_mad``, ``rgb_frac_gt16``, ``rgb_replay``) and provenance.
    """
    import mujoco

    from AG3S.ag3s.experiments.sources.policy_record import (
        ATTENTION_GRID, CAMERA_BINDINGS, POLICY_CAMERA_NAMES, load_run, patch_coverage,
        pose_scene, provenance_arrays, replay_scene)

    record = load_run(record_path)
    scene = replay_scene(record, height=height, width=width)
    lut, members = body_class_table(scene.model, target)
    names = _body_names(scene.model)
    named_ids = [names.index(n) for n in NAMED_BODIES if n in names]
    named = [n for n in NAMED_BODIES if n in names]
    cams = list(POLICY_CAMERA_NAMES)
    mj_cams = [CAMERA_BINDINGS[c][0] for c in cams]
    n_f, n_c, n_k = len(record), len(cams), len(CLASS_NAMES)
    class_cov = np.zeros((n_f, n_c, n_k, ATTENTION_GRID, ATTENTION_GRID), np.float32)
    class_px = np.zeros((n_f, n_c, n_k), np.int64)
    body_cov = np.zeros((n_f, n_c, len(named), ATTENTION_GRID, ATTENTION_GRID), np.float32)
    body_px = np.zeros((n_f, n_c, len(named)), np.int64)
    marker_px = np.zeros((n_f, n_c), np.int64)
    label_small = np.zeros((n_f, n_c, height // label_stride, width // label_stride), np.uint8)
    target_pos = np.zeros((n_f, 3), np.float64)
    fruit_pos = np.zeros((n_f, len(FRUITS), 3), np.float64)
    mocap_ids = [b for b in range(scene.model.nbody) if int(scene.model.body_mocapid[b]) >= 0]

    n_check = max(0, min(int(rgb_check_frames), n_f))
    check_idx = (np.unique(np.linspace(0, n_f - 1, n_check).round().astype(int))
                 if n_check else np.zeros(0, int))
    rgb_mad = np.zeros((len(check_idx), n_c), np.float64)
    rgb_frac = np.zeros((len(check_idx), n_c), np.float64)
    rgb_replay = np.zeros((len(check_idx), n_c, POLICY_IMAGE_SIZE, POLICY_IMAGE_SIZE, 3), np.uint8)
    rgb_renderer = mujoco.Renderer(scene.model, *POLICY_SOURCE_HW) if n_check else None
    t0 = time.time()
    try:
        for i, step in enumerate(record):
            pose_scene(scene, step)
            target_pos[i] = scene.body_position_in_base(target)
            for j, fruit in enumerate(FRUITS):
                fruit_pos[i, j] = scene.body_position_in_base(fruit)
            for ci, mj in enumerate(mj_cams):
                frame = scene.capture(mj)
                cls = class_map(frame.body_ids, lut)
                cov, px = patch_coverage(cls, list(range(n_k)))
                class_cov[i, ci], class_px[i, ci] = cov, px.astype(np.int64)
                bcov, bpx = patch_coverage(frame.body_ids, named_ids)
                body_cov[i, ci], body_px[i, ci] = bcov, bpx.astype(np.int64)
                marker_px[i, ci] = int(np.isin(frame.body_ids, mocap_ids).sum()) if mocap_ids else 0
                label_small[i, ci] = cls[::label_stride, ::label_stride].astype(np.uint8)
            if rgb_renderer is not None and i in set(check_idx.tolist()):
                r = int(np.flatnonzero(check_idx == i)[0])
                for ci, (pol, mj) in enumerate(zip(cams, mj_cams)):
                    img = render_policy_rgb(scene.model, scene.data, rgb_renderer, mj)
                    rec = np.asarray(step.images[pol], np.uint8)
                    diff = np.abs(img.astype(np.int16) - rec.astype(np.int16))
                    rgb_mad[r, ci] = float(diff.mean())
                    rgb_frac[r, ci] = float((diff.max(axis=-1) > 16).mean())
                    rgb_replay[r, ci] = img
            if i % 10 == 0 or i == n_f - 1:
                log(f"  gt frame {i + 1}/{n_f}  ({time.time() - t0:.1f}s)")
    finally:
        if rgb_renderer is not None:
            rgb_renderer.close()
        scene.close()
    return {
        "t_step": np.asarray([s.t_step for s in record], np.int64),
        "classes": np.asarray(CLASS_NAMES),
        "class_members": np.asarray([f"{c}: {', '.join(members[c])}" for c in CLASS_NAMES]),
        "class_cov": class_cov, "class_px": class_px,
        "named_bodies": np.asarray(named), "body_cov": body_cov.astype(np.float16),
        "body_px": body_px, "marker_px": marker_px,
        "label_small": label_small, "label_stride": np.asarray(label_stride),
        "target": np.asarray(target), "prompt": np.asarray(record.prompt),
        "target_pos_base": target_pos, "fruits": np.asarray(FRUITS), "fruit_pos_base": fruit_pos,
        "cameras": np.asarray(cams), "mujoco_cameras": np.asarray(mj_cams),
        "render_hw": np.asarray([height, width]),
        "rgb_check_index": check_idx.astype(np.int64), "rgb_mad": rgb_mad,
        "rgb_frac_gt16": rgb_frac, "rgb_replay": rgb_replay,
        "model_xml": np.asarray(record.model_xml),
        **provenance_arrays(record, n_f),
    }


# ---------------------------------------------------------------------------------- I4: e2e


def _nanmean(values) -> float:
    v = np.asarray([x for x in values if x is not None], np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def summarize_e2e(rows: Sequence[dict]) -> dict:
    """Per-source aggregates of `run_e2e_record` rows (one row = one frame x one source)."""
    out: dict[str, dict] = {}
    for src in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == src]
        n = len(rs)
        has = [r for r in rs if r["has_target"]]
        visible = [r for r in rs if r["n_cloud_target"] > 0]
        out[src] = {
            "n_frames": n,
            "has_target": sum(r["has_target"] for r in rs),
            "on_target": sum(r["on_target"] for r in rs),
            "silent_failure": sum(r["silent_failure"] for r in rs),
            "has_target_rate": len(has) / n if n else float("nan"),
            "on_target_rate": sum(r["on_target"] for r in rs) / n if n else float("nan"),
            "silent_failure_rate": sum(r["silent_failure"] for r in rs) / n if n else float("nan"),
            "n_frames_target_in_cloud": len(visible),
            "on_target_given_target_in_cloud": (sum(r["on_target"] for r in visible) / len(visible)
                                                if visible else float("nan")),
            "seed_precision_mean": _nanmean(r["seed_precision"] for r in rs),
            "seed_recall_mean": _nanmean(r["seed_recall"] for r in rs),
            "chosen_target_fraction_mean": _nanmean(r["chosen_target_fraction"] for r in has),
            "centroid_error_m_median": (float(np.median([r["centroid_error_m"] for r in has]))
                                        if has else float("nan")),
            "ms_mean": _nanmean(r["ag3s_ms"] for r in rs),
            "ms_median": float(np.median([r["ag3s_ms"] for r in rs])) if rs else float("nan"),
        }
    return out


def run_e2e_record(record_path, *, target: str, sources: dict[str, np.ndarray],
                   frames: Optional[Sequence[int]] = None, voxel: float = 0.020,
                   range_max: float = 2.0, links: str = "gripper", esdf_backend: str = "legacy",
                   on_target_radius: float = 0.10, label_radius: float = 0.003,
                   log: Callable[[str], None] = print) -> list[dict]:
    """Ground the target on each frame, once per attention source, on the live serving path.

    `sources` maps a source name to ``(F, 3, 16, 16)`` raw maps in `POLICY_CAMERA_NAMES` order
    (cam_high, cam_left_wrist, cam_right_wrist). Returns one row per (frame, source):

    ``has_target`` (a cluster was named), ``on_target`` (its centroid within `on_target_radius` of
    the GT target body origin in the base frame — T1-a's `on_apple`), ``silent_failure``
    (has ∧ ¬on), seed precision / recall against GT point labels (`exclusion_e2.point_labeler`:
    nearest GT depth pixel of any camera within `label_radius`; recall's denominator = GT-target
    points of the grounding cloud outside the support mask), the chosen cluster's GT-target point
    fraction, and the wall time of `process_multi_debug`.
    """
    from AG3S.ag3s.experiments.live.exclusion_e2 import (
        LABELS, gt_points, label_classes, point_labeler)
    from AG3S.ag3s.experiments.sources.mujoco_source import set_finger_joints_from_scene
    from AG3S.ag3s.experiments.sources.policy_record import (
        CAMERA_BINDINGS, POLICY_CAMERA_NAMES, control_step_ms, load_run,
        phase_boundaries_for_path, phase_for, pose_scene, replay_scene)
    from AG3S.ag3s.runtime.asset_path import resolve_asset
    from AG3S.trajopt import wire
    from AG3S.trajopt.serve_safe import build_ag3s

    record = load_run(record_path)
    n_f = len(record)
    for name, maps in sources.items():
        if np.asarray(maps).shape[:2] != (n_f, len(POLICY_CAMERA_NAMES)):
            raise ValueError(f"source {name!r} maps {np.asarray(maps).shape} do not match "
                             f"({n_f}, {len(POLICY_CAMERA_NAMES)}, 16, 16)")
    mj_cams = [CAMERA_BINDINGS[c][0] for c in POLICY_CAMERA_NAMES]
    model_xml = str(resolve_asset(record.model_xml, what="transport model"))
    scene = replay_scene(record)
    boundaries, _evidence = phase_boundaries_for_path(record_path)
    dt = control_step_ms(record) / 1000.0
    # The server's factory at its own defaults (`serve_safe` argparse: voxel 0.02, range 2.0,
    # links gripper, esdf legacy). Subtask gate off: no subtask label exists offline, which is
    # exactly when `serve_safe.resolve_subtask_gate` turns it off too.
    ags = {name: build_ag3s(model_xml, voxel=voxel, range_max=range_max, links=links,
                            backend=esdf_backend, subtask_gate=False) for name in sources}
    classes = label_classes(scene, target)
    t_lab = LABELS.index("target")
    idx = range(n_f) if frames is None else [int(i) for i in frames]
    rows: list[dict] = []
    t_start = time.time()
    try:
        for i in idx:
            step = record.steps[i]
            pose_scene(scene, step)
            captured = {c: scene.capture(c) for c in mj_cams}
            gt_centre = scene.body_position_in_base(target)
            gpts, glab = gt_points(captured.values(), classes)
            label_of = point_labeler(gpts, glab, radius=label_radius)
            phase = phase_for(step.t_step, boundaries)
            stamp = float(step.t_step) * dt
            req = wire.pack_request(
                {}, cameras=mj_cams, depth={c: f.depth for c, f in captured.items()},
                intrinsics={c: f.camera_intrinsics for c, f in captured.items()},
                extrinsics={c: f.T_base_cam for c, f in captured.items()},
                robot_state={c: f.robot_state for c, f in captured.items()},
                stamps={c: stamp for c in mj_cams}, phase=phase, seq=int(i))
            _, sc = wire.strip_request(req)
            for name, ag in ags.items():
                att = {mj: np.asarray(sources[name][i, ci], np.float32)
                       for ci, mj in enumerate(mj_cams)}
                set_finger_joints_from_scene(scene, ag.robot_model)
                if ag.constraint_robot_model is not None:
                    set_finger_joints_from_scene(scene, ag.constraint_robot_model)
                obs = wire.unpack_camera_observations(sc, robot_model=ag.robot_model,
                                                      attention=att)
                t0 = time.perf_counter()
                cs, debug = ag.process_multi_debug(obs, phase=phase)
                ms = (time.perf_counter() - t0) * 1000.0
                cloud = debug.get("filtered_cloud")
                pts = np.zeros((0, 3)) if cloud is None else np.asarray(cloud.points)
                labels = label_of(pts)
                support = debug.get("support_mask")
                support = (np.zeros(len(pts), bool) if support is None
                           else np.asarray(support, bool))
                seeds = np.asarray(debug.get("seed_indices", np.zeros(0, np.int64)), np.int64)
                seed_is_t = labels[seeds] == t_lab if seeds.size else np.zeros(0, bool)
                n_cloud_t = int(((labels == t_lab) & ~support).sum())
                has = bool(cs.has_target)
                err = (float(np.linalg.norm(np.asarray(cs.target.centroid) - gt_centre))
                       if has else float("nan"))
                on = bool(has and err <= on_target_radius)
                chosen = (float((label_of(cs.target.points) == t_lab).mean())
                          if has and len(cs.target.points) else float("nan"))
                grounding = debug.get("grounding")
                rows.append({
                    "frame": int(i), "t_step": int(step.t_step), "phase": phase,
                    "source": name, "has_target": has, "on_target": on,
                    "silent_failure": bool(has and not on),
                    "centroid_error_m": err,
                    "grounding_status": (None if grounding is None
                                         else getattr(grounding.status, "value",
                                                      str(grounding.status))),
                    "n_clusters": None if grounding is None else len(grounding.clusters),
                    "n_cloud": int(len(pts)), "n_cloud_target": n_cloud_t,
                    "n_seeds": int(seeds.size), "n_seeds_target": int(seed_is_t.sum()),
                    "seed_precision": (float(seed_is_t.mean()) if seeds.size else float("nan")),
                    "seed_recall": (float(seed_is_t.sum() / n_cloud_t) if n_cloud_t
                                    else float("nan")),
                    "chosen_target_fraction": chosen,
                    "seed_camera": (None if not has or cs.target.seed_camera is None
                                    else str(getattr(cs.target.seed_camera, "value",
                                                     cs.target.seed_camera))),
                    "ag3s_ms": ms,
                })
            log(f"  e2e frame {i + 1}/{n_f} t={step.t_step} {phase}  "
                + "  ".join(f"{r['source']}:{'T' if r['on_target'] else ('s' if r['has_target'] else '-')}"
                            for r in rows[-len(ags):])
                + f"  ({time.time() - t_start:.0f}s)")
    finally:
        scene.close()
    return rows


__all__ = ["CLS", "FRUITS", "GRIPPER_ROOT_LINKS", "NAMED_BODIES", "body_class_table",
           "class_map", "extract_gt", "gripper_body_ids", "render_policy_rgb", "run_e2e_record",
           "summarize_e2e"]
