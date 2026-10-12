"""T47b — the MuJoCo replay half: B2 (label-based 3D metrics) and B3 (x k numeric check).

Both functions **render** (osmesa, CPU) and are run only by the verifier (T47 approval covers it).

The frame loop is T47's `t47_replay.run_e2e_record` call for call (the live serving path:
`wire.pack_request -> strip_request -> unpack_camera_observations(attention=...) ->
AG3S.process_multi_debug`, AG3S from `serve_safe.build_ag3s` at the server defaults, subtask gate off,
one AG3S per variant so TargetConfirm state never crosses variants). `t47_replay.py` itself is not
touched (T47's verifier was running it); the loop is re-assembled here, and every T47 row field is
computed with the same expressions so the two row files can be compared (`t47b_study --stage parity`).

B2 adds per frame (`t47b_metrics.frame_b2`): the chosen cluster's GT class counts, purity,
``identity_correct``, target residual and non-target loss at three levels (chosen cluster, attention
seeds, exclusion ball — the ball is `AG3S._target_ball(target points, support surfaces, voxel)`,
the call `pipeline.py:2598` makes, evaluated post hoc: the offline AG3S runs the legacy ESDF backend
and does not carve).

B3 runs, per source, one AG3S on the map and one per factor on ``float32(factor * map64)`` (the
combined map is float64 in the study; T47 casts it to float32 the same way) and compares the two
frames' grounding (`t47b_b3.compare_snapshots`).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t47b_b3 as b3
from AG3S.ag3s.experiments.studies import t47b_metrics as bm

#: The live fine voxel (V11AB cuRobo ESDF, 5 mm) — the voxel the exclusion ball is padded with live.
DEFAULT_BALL_VOXEL = 0.005


def _frames(record_path, *, target: str, frames: Optional[Sequence[int]], label_radius: float,
            log: Callable[[str], None]) -> Iterator[dict]:
    """T47's per-frame preparation (`t47_replay.run_e2e_record`), yielded frame by frame."""
    from AG3S.ag3s.experiments.live.exclusion_e2 import gt_points, label_classes, point_labeler
    from AG3S.ag3s.experiments.sources.policy_record import (
        CAMERA_BINDINGS, POLICY_CAMERA_NAMES, control_step_ms, load_run,
        phase_boundaries_for_path, phase_for, pose_scene, replay_scene)
    from AG3S.trajopt import wire

    record = load_run(record_path)
    n_f = len(record)
    mj_cams = [CAMERA_BINDINGS[c][0] for c in POLICY_CAMERA_NAMES]
    scene = replay_scene(record)
    boundaries, _ = phase_boundaries_for_path(record_path)
    dt = control_step_ms(record) / 1000.0
    classes = label_classes(scene, target)
    idx = range(n_f) if frames is None else [int(i) for i in frames]
    try:
        for i in idx:
            step = record.steps[i]
            pose_scene(scene, step)
            captured = {c: scene.capture(c) for c in mj_cams}
            gt_centre = scene.body_position_in_base(target)
            gpts, glab = gt_points(captured.values(), classes)
            phase = phase_for(step.t_step, boundaries)
            stamp = float(step.t_step) * dt
            req = wire.pack_request(
                {}, cameras=mj_cams, depth={c: f.depth for c, f in captured.items()},
                intrinsics={c: f.camera_intrinsics for c, f in captured.items()},
                extrinsics={c: f.T_base_cam for c, f in captured.items()},
                robot_state={c: f.robot_state for c, f in captured.items()},
                stamps={c: stamp for c in mj_cams}, phase=phase, seq=int(i))
            _, sc = wire.strip_request(req)
            yield {"i": int(i), "n_f": n_f, "step": step, "phase": phase, "sc": sc,
                   "scene": scene, "mj_cams": mj_cams, "gt_centre": gt_centre,
                   "label_of": point_labeler(gpts, glab, radius=label_radius)}
    finally:
        scene.close()


def _build(record_path, names: Sequence, *, voxel, range_max, links, esdf_backend) -> dict:
    from AG3S.ag3s.experiments.sources.policy_record import load_run
    from AG3S.ag3s.runtime.asset_path import resolve_asset
    from AG3S.trajopt.serve_safe import build_ag3s

    record = load_run(record_path)
    model_xml = str(resolve_asset(record.model_xml, what="transport model"))
    return {n: build_ag3s(model_xml, voxel=voxel, range_max=range_max, links=links,
                          backend=esdf_backend, subtask_gate=False) for n in names}


def _ground(ag, fr: dict, att_maps: dict):
    from AG3S.ag3s.experiments.sources.mujoco_source import set_finger_joints_from_scene
    from AG3S.trajopt import wire

    set_finger_joints_from_scene(fr["scene"], ag.robot_model)
    if ag.constraint_robot_model is not None:
        set_finger_joints_from_scene(fr["scene"], ag.constraint_robot_model)
    obs = wire.unpack_camera_observations(fr["sc"], robot_model=ag.robot_model, attention=att_maps)
    t0 = time.perf_counter()
    cs, debug = ag.process_multi_debug(obs, phase=fr["phase"])
    return cs, debug, (time.perf_counter() - t0) * 1000.0


def _check_sources(sources: dict, n_f: int, n_c: int) -> None:
    for name, maps in sources.items():
        if np.asarray(maps).shape[:2] != (n_f, n_c):
            raise ValueError(f"source {name!r} maps {np.asarray(maps).shape} do not match "
                             f"({n_f}, {n_c}, 16, 16)")


def t47_row(cs, debug, *, label_of, gt_centre, on_target_radius: float) -> tuple[dict, dict]:
    """T47's row fields (`t47_replay.run_e2e_record`, same expressions) + the arrays B2 needs."""
    from AG3S.ag3s.experiments.live.exclusion_e2 import LABELS

    t_lab = LABELS.index("target")
    cloud = debug.get("filtered_cloud")
    pts = np.zeros((0, 3)) if cloud is None else np.asarray(cloud.points)
    labels = label_of(pts)
    support = debug.get("support_mask")
    support = np.zeros(len(pts), bool) if support is None else np.asarray(support, bool)
    seeds = np.asarray(debug.get("seed_indices", np.zeros(0, np.int64)), np.int64)
    seed_is_t = labels[seeds] == t_lab if seeds.size else np.zeros(0, bool)
    n_cloud_t = int(((labels == t_lab) & ~support).sum())
    has = bool(cs.has_target)
    err = (float(np.linalg.norm(np.asarray(cs.target.centroid) - gt_centre)) if has
           else float("nan"))
    on = bool(has and err <= on_target_radius)
    chosen = (float((label_of(cs.target.points) == t_lab).mean())
              if has and len(cs.target.points) else float("nan"))
    grounding = debug.get("grounding")
    row = {
        "has_target": has, "on_target": on, "silent_failure": bool(has and not on),
        "centroid_error_m": err,
        "grounding_status": (None if grounding is None
                             else getattr(grounding.status, "value", str(grounding.status))),
        "n_clusters": None if grounding is None else len(grounding.clusters),
        "n_cloud": int(len(pts)), "n_cloud_target": n_cloud_t,
        "n_seeds": int(seeds.size), "n_seeds_target": int(seed_is_t.sum()),
        "seed_precision": float(seed_is_t.mean()) if seeds.size else float("nan"),
        "seed_recall": float(seed_is_t.sum() / n_cloud_t) if n_cloud_t else float("nan"),
        "chosen_target_fraction": chosen,
        "seed_camera": (None if not has or cs.target.seed_camera is None
                        else str(getattr(cs.target.seed_camera, "value", cs.target.seed_camera))),
    }
    return row, {"pts": pts, "labels": labels, "support": support, "seeds": seeds}


def b2_fields(ag, cs, arrays: dict, *, ball_voxel: float, membership_radius: float) -> dict:
    """`t47b_metrics.frame_b2` on one frame + how the chosen cluster and the ball were found."""
    pts = arrays["pts"]
    has = bool(cs.has_target)
    chosen, source = (np.zeros(len(pts), bool), "none")
    ball, ball_rec = None, None
    if has:
        chosen, source = bm.membership_mask(pts, np.asarray(cs.target.points),
                                            np.asarray(cs.target.point_indices),
                                            radius=membership_radius)
        try:
            tb = ag._target_ball(np.asarray(cs.target.points), list(cs.support_surfaces),
                                 voxel=float(ball_voxel))
        except Exception as exc:  # noqa: BLE001 — recorded, the frame still counts
            tb, ball_rec = None, {"error": f"{type(exc).__name__}: {exc}"}
        if tb is not None:
            ball = bm.ball_mask(pts, tb.centre, tb.radius, tb.z_min)
            ball_rec = {"radius_m": float(tb.radius), "z_min_m": float(tb.z_min),
                        "centre_m": np.asarray(tb.centre, np.float64).round(5).tolist(),
                        "centre_method": tb.provenance.get("centre_method")}
        elif ball_rec is None:
            ball_rec = {"error": "no ball (no support plane under the target, or none above it)"}
    m = bm.frame_b2(arrays["labels"], support_mask=arrays["support"], seed_indices=arrays["seeds"],
                    chosen_mask=chosen, has_target=has, ball=ball)
    m["chosen_membership"] = source
    m["ball"] = ball_rec
    return m


def run_b2_record(record_path, *, target: str, sources: dict[str, np.ndarray],
                  frames: Optional[Sequence[int]] = None, voxel: float = 0.020,
                  range_max: float = 2.0, links: str = "gripper", esdf_backend: str = "legacy",
                  on_target_radius: float = 0.10, label_radius: float = 0.003,
                  ball_voxel: float = DEFAULT_BALL_VOXEL, membership_radius: float = 0.005,
                  log: Callable[[str], None] = print) -> list[dict]:
    """One row per (frame, source): T47's fields + B2 (`b2_fields`). `sources` as in T47:
    ``{name: (F, 3, 16, 16)}`` raw maps in `POLICY_CAMERA_NAMES` order."""
    from AG3S.ag3s.experiments.sources.policy_record import POLICY_CAMERA_NAMES, load_run

    _check_sources(sources, len(load_run(record_path)), len(POLICY_CAMERA_NAMES))
    ags = _build(record_path, list(sources), voxel=voxel, range_max=range_max, links=links,
                 esdf_backend=esdf_backend)
    rows: list[dict] = []
    t_start = time.time()
    for fr in _frames(record_path, target=target, frames=frames, label_radius=label_radius,
                      log=log):
        i, step = fr["i"], fr["step"]
        for name, ag in ags.items():
            att = {mj: np.asarray(sources[name][i, ci], np.float32)
                   for ci, mj in enumerate(fr["mj_cams"])}
            cs, debug, ms = _ground(ag, fr, att)
            row, arrays = t47_row(cs, debug, label_of=fr["label_of"], gt_centre=fr["gt_centre"],
                                  on_target_radius=on_target_radius)
            rows.append({"frame": i, "t_step": int(step.t_step), "phase": fr["phase"],
                         "phase_group": bm.phase_group(fr["phase"]), "source": name, **row,
                         "ag3s_ms": ms,
                         **b2_fields(ag, cs, arrays, ball_voxel=ball_voxel,
                                     membership_radius=membership_radius)})
        recent = rows[-len(ags):]
        log(f"  b2 frame {i + 1}/{fr['n_f']} t={step.t_step} {fr['phase']}  "
            + "  ".join(f"{r['source']}:{'I' if r['identity_correct'] else ('T' if r['on_target'] else ('s' if r['has_target'] else '-'))}"
                        for r in recent) + f"  ({time.time() - t_start:.0f}s)")
    return rows


def run_b3_record(record_path, *, target: str, sources: dict[str, np.ndarray],
                  factors: Sequence[float] = (3.0, 4.0), frames: Optional[Sequence[int]] = None,
                  voxel: float = 0.020, range_max: float = 2.0, links: str = "gripper",
                  esdf_backend: str = "legacy", log: Callable[[str], None] = print) -> list[dict]:
    """One row per (frame, source, factor): `compare_snapshots(map, factor x map)` scalars.

    `sources` hold float64 combined maps ``(F, 3, 16, 16)``; the base run sees ``float32(map)`` and
    the scaled run ``float32(factor * map)``. Each (source, factor) pair has its own AG3S, and the
    base AG3S of a source is shared by its factors (it is the same sequence).
    """
    from AG3S.ag3s.experiments.sources.policy_record import POLICY_CAMERA_NAMES, load_run

    _check_sources(sources, len(load_run(record_path)), len(POLICY_CAMERA_NAMES))
    if any(float(f) <= 0 for f in factors):
        raise ValueError("factors must be > 0")
    keys = [(s, 1.0) for s in sources] + [(s, float(f)) for s in sources for f in factors]
    ags = _build(record_path, keys, voxel=voxel, range_max=range_max, links=links,
                 esdf_backend=esdf_backend)
    rows: list[dict] = []
    t_start = time.time()
    for fr in _frames(record_path, target=target, frames=frames, label_radius=0.003, log=log):
        i = fr["i"]
        snaps = {}
        for (name, f), ag in ags.items():
            m64 = np.asarray(sources[name][i], np.float64)
            att = {mj: (m64[ci] * f).astype(np.float32) for ci, mj in enumerate(fr["mj_cams"])}
            cs, debug, _ = _ground(ag, fr, att)
            snaps[(name, f)] = b3.snapshot_from_debug(cs, debug)
        for name in sources:
            for f in factors:
                cmp = b3.compare_snapshots(snaps[(name, 1.0)], snaps[(name, float(f))], f)
                rows.append({"frame": i, "t_step": int(fr["step"].t_step), "phase": fr["phase"],
                             "source": name, **cmp})
        recent = rows[-len(sources) * len(factors):]
        log(f"  b3 frame {i + 1}/{fr['n_f']}  "
            + "  ".join(f"{r['source']}x{r['factor']:g}:{'=' if r['bit_identical'] else ('~' if r['decision_identical'] else 'X')}"
                        for r in recent) + f"  ({time.time() - t_start:.0f}s)")
    return rows


__all__ = ["DEFAULT_BALL_VOXEL", "b2_fields", "run_b2_record", "run_b3_record", "t47_row"]
