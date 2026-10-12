"""SL4 offline perception replay: recorded Panda frames -> V11AB AG3S (Panda profile) -> ATS / LTE / ESDF.

``.venv-openpi-live`` (torch + cuRobo + mujoco in one process), GPU. For every policy-call frame of a
geometry-capture rollout (SL3 dev run) this runs **V11AB's own perception code**
(`AG3S.process_multi_debug`) on the recorded observation, with the SL3 attention read-out of the same
call, and stores what it produced plus the ESDF queried at ground-truth places. AG3S core is not
modified; the Panda differences come from :mod:`AG3S.adapters.panda.profile` (V11AB flags + Panda
models / contact links / ESDF floor).

Per call ``k`` of an episode (``obs_index = chunk_t[k]``):

1. wire camera observation of the geometry npz row (``wire_pack.wire_from_geom`` -> ``wire.strip_request``
   -> ``wire.unpack_camera_observations``: the V11AB server path) with the attention grids of the audit run
   attached per camera (``--attention-mapping unmirror`` = ``attention.unmirror_grid``, ``mirror`` = the
   policy-orientation grid as it is; diagnosis);
2. ``robot_state`` (9) -> ``q7`` + finger parameters (``model.split_robot_state``); every camera's
   observation gets its ``q7``; ``AG3S.set_finger_joints(fingers)`` (what ``SafePolicy._apply_finger_state``
   does for RB-Y1); grasp / latch / subtask / placed are set to the no-latch values ``SafePolicy`` gives
   before a grasp (``False`` / ``None``: no subtask label on Panda -> ungated);
3. ``process_multi_debug(observations, phase=<V11AB client phase: approach>, active_manipulators=["left"])``;
   no ``execution_path`` (the Panda chunk is EE deltas; SL-5) and no static geometry (V11AB default);
4. records: grounding (status, target points / centroid, seed points + camera provenance), manipulated
   object and exclusion record, fused / filtered cloud (points, camera mask, attention, seed / support /
   target flags; ``--no-clouds`` drops them), ESDF at the GT surface samples and at the constraint sphere
   centres -- main layer (``field.distance``) and target-free composition (``field.target_free_distance``,
   what authorized links read) with the answering tier --, exact sphere-to-object distances
   (``libero_scene_gt``), ESDF slices through the GT target (main and target-free), stage times
   (``constraint_set.profile``), wall times and GPU memory.

Episodes share one AG3S and are separated by ``AG3S.reset()`` (``SafePolicy.reset`` at the V11AB episode
boundary). The run is new (``create_run``); source runs and GT files are only read and their SHA-256 are
recorded.

Run layout::

    <run>/perception/<stem>_perception.npz   per-episode arrays (EPISODE_ARRAYS below)
    <run>/frames.jsonl                       one row per call: grounding / esdf / timing summary
    <run>/perception_index.jsonl             one row per episode: sources, GT file, counts
    <run>/summary.json                       totals, timing p50/p95 (first call excluded), GPU memory

Example (verifier; GPU, inside the shared lock)::

    source scripts/setup/env.sh
    ( flock -w 1800 9 || exit 1
      .venv-openpi-live/bin/python benchmark/runners/perception_replay.py \\
          --source-run <dev rollout run> --attention-run <SL3 audit run> --scene-gt <scene_export dir> \\
          --run-id sl4-perception-dev ) 9>/tmp/ag3s-vla-gpu-validation.lock

(``flock -w 1800 9 CMD 9>LOCK`` would lock a file named ``9`` in the cwd, not LOCK: use the subshell form or
``flock -w 1800 LOCK CMD``.) The run refuses to start when ``nvidia-smi`` lists compute processes
(``--allow-busy-gpu`` overrides).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Optional

import numpy as np

from common.workspace import PROJECT_ROOT

STEP = "SL4"
CONDITION = ("perception replay (offline): V11AB AG3S with the Panda profile on recorded LIBERO frames + SL3 "
             "V11AB-cell attention; no policy, no TO")
CAMERAS = ("head", "left_wrist")
CAMERA_BIT = {"head": 1, "left_wrist": 2}
MAPPINGS = ("unmirror", "mirror")
#: ESDF answer tiers as small ints (index into this tuple; -1 = no answer).
TIERS = ("outside", "coarse", "fine", "target_free", "window_edge", "static")
EXACT_MAX_DISTANCE = 0.45


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _jsonable(value.value)
    return value


# --- sources ---------------------------------------------------------------------------------------------
def index_sources(dev_value: str, audit_value: str, *, task_ids=None) -> dict:
    """Dev rollout run + SL3 attention audit run, joined per episode and **checked**."""
    from benchmark.runners import attention_audit as aa

    dev = aa.index_source_run(aa.resolve_source_run(dev_value))
    audit_dir = aa.resolve_source_run(audit_value)
    rows = [json.loads(l) for l in (audit_dir / "attention_index.jsonl").read_text().splitlines() if l.strip()]
    by_stem = {}
    for row in rows:
        by_stem[aa.episode_stem(row["suite"], row["task_id"], row["episode_idx"])] = row
    episodes = []
    for e in dev["episodes"]:
        if task_ids is not None and e["task_id"] not in set(task_ids):
            continue
        stem = aa.episode_stem(e["suite"], e["task_id"], e["episode_idx"])
        row = by_stem.get(stem)
        if row is None:
            raise ValueError(f"attention run {audit_dir.name} has no episode {stem}")
        if Path(row["source_run_dir"]).resolve() != Path(dev["run_dir"]).resolve():
            raise ValueError(f"{stem}: attention run read {row['source_run_dir']}, not {dev['run_dir']}")
        if not e.get("geom_npz"):
            raise ValueError(f"{stem}: the dev run has no geometry npz")
        dev_dir = Path(dev["run_dir"])
        npz_sha, geom_sha = sha256_file(dev_dir / e["npz"]), sha256_file(dev_dir / e["geom_npz"])
        if row.get("npz_sha256") != npz_sha or row.get("geom_npz_sha256") != geom_sha:
            raise ValueError(f"{stem}: the attention run read different episode files (SHA-256 mismatch)")
        attn = audit_dir / row["attn_npz"]
        with np.load(attn) as a, np.load(dev_dir / e["npz"]) as ep:
            cams = tuple(str(c) for c in a["cameras"])
            if cams != CAMERAS:
                raise ValueError(f"{stem}: attention cameras {cams}, expected {CAMERAS}")
            chunk_t = np.asarray(ep["chunk_t"]).astype(np.int64)
            if not np.array_equal(np.asarray(a["obs_index"]).astype(np.int64), chunk_t):
                raise ValueError(f"{stem}: attention obs_index differs from the episode chunk_t")
        episodes.append({**e, "stem": stem, "attn_npz": str(attn), "attn_npz_sha256": sha256_file(attn),
                         "npz_sha256": npz_sha, "geom_npz_sha256": geom_sha, "target": row.get("target"),
                         "calls": int(len(chunk_t))})
    if not episodes:
        raise ValueError("no episode selected")
    audit_manifest = audit_dir / "run_manifest.json"
    return {"dev": {k: v for k, v in dev.items() if k != "episodes"},
            "attention": {"run_dir": str(audit_dir), "run_id": audit_dir.name,
                          "manifest_sha256": sha256_file(audit_manifest),
                          "attention_index_sha256": sha256_file(audit_dir / "attention_index.jsonl"),
                          "settings_v11ab_cell": json.loads(audit_manifest.read_text()).get("settings", {}).get("v11ab_cell")},
            "episodes": episodes}


def index_scene_gt(scene_dir: Optional[str], sources: dict, *, limit_calls: Optional[int] = None) -> Optional[dict]:
    """``scene_export.py`` output for every selected episode, checked against the dev run files."""
    if scene_dir is None:
        return None
    root = Path(scene_dir).resolve()
    files = {}
    for e in sources["episodes"]:
        path = root / f"scene_{e['stem']}.npz"
        if not path.exists():
            raise ValueError(f"no scene GT for {e['stem']} in {root}")
        with np.load(path) as z:
            prov = json.loads(str(z["provenance_json"]))
            frames = set(int(x) for x in z["frames"])
        if prov.get("geom_npz_sha256") != e["geom_npz_sha256"] or prov.get("npz_sha256") != e["npz_sha256"]:
            raise ValueError(f"{path.name} was exported from different episode files")
        with np.load(Path(sources["dev"]["run_dir"]) / e["npz"]) as ep:
            calls = [int(t) for t in ep["chunk_t"]][: limit_calls or None]
            missing = sorted(set(calls) - frames)
        if missing:
            raise ValueError(f"{path.name} lacks call frames {missing[:5]}...")
        files[e["stem"]] = {"path": str(path), "sha256": sha256_file(path), "checks": prov.get("checks")}
    summary = root / "scene_export.json"
    return {"dir": str(root), "export_json_sha256": sha256_file(summary) if summary.exists() else None, "files": files}


# --- one frame -----------------------------------------------------------------------------------------
def build_observations(geom: dict, row: int, grids_policy: np.ndarray, mapping: str, robot_model):
    """Camera observations of geometry-npz ``row`` with attention and the Panda ``q7``.

    ``grids_policy``: (2, 16, 16) V11AB cell grids [head, left_wrist] in policy orientation (audit
    ``cell_grid_policy``). Returns ``(observations, fingers, robot_state9, scene)``.
    """
    from AG3S.adapters.panda import attention as pa
    from AG3S.adapters.panda import model as pm
    from AG3S.trajopt import wire
    from benchmark.adapters.libero_panda import wire_pack

    if mapping not in MAPPINGS:
        raise ValueError(f"mapping must be one of {MAPPINGS}, got {mapping!r}")
    _, scene = wire.strip_request(dict(wire_pack.wire_from_geom(geom, row)))
    attention = {cam: np.ascontiguousarray(pa.unmirror_grid(np.asarray(grids_policy[i], np.float32),
                                                            unmirror=mapping == "unmirror"))
                 for i, cam in enumerate(CAMERAS)}
    observations = wire.unpack_camera_observations(scene, robot_model=robot_model, attention=attention)
    states = {o.camera_id.value: np.asarray(o.robot_state, np.float64) for o in observations}
    rs9 = states[CAMERAS[0]]
    for cam, rs in states.items():
        if not np.array_equal(rs, rs9):
            raise ValueError(f"row {row}: robot_state of {cam} differs from {CAMERAS[0]} (one instant expected)")
    q7, fingers = pm.split_robot_state(rs9)
    observations = [dataclasses.replace(o, robot_state=q7.copy()) for o in observations]
    return observations, fingers, rs9, scene


def apply_frame_state(ag3s, fingers: dict, rs9: np.ndarray) -> dict:
    """What ``SafePolicy._scene_fn`` sets before ``process_multi_debug`` when no grasp is latched."""
    record = ag3s.set_finger_joints(fingers, record={
        "source": "robot_state[7:9] (Panda adapter, image instant)", "robot_state_fingers": rs9[7:9].tolist(),
        "convention": "JOINT_MAP: finger_joint1 +, finger_joint2 - (AG3S.adapters.panda.model)"})
    ag3s.set_grasp_active(False)
    ag3s.set_grasp_latched(False)
    ag3s.set_subtask(None)
    ag3s.set_placed(False)
    return record


def _tier_codes(tiers) -> np.ndarray:
    lookup = {t: i for i, t in enumerate(TIERS)}
    return np.array([lookup.get(str(t), -1) for t in np.asarray(tiers).reshape(-1)], np.int8)


def query_field(field, points: np.ndarray) -> dict:
    """Main layer + target-free composition at ``points`` (base frame) and the answering tiers."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    n = len(pts)
    out = {"main": np.full(n, np.nan, np.float32), "tf": np.full(n, np.nan, np.float32),
           "tier_main": np.full(n, -1, np.int8), "tier_tf": np.full(n, -1, np.int8)}
    if field is None or not n:
        return out
    out["main"] = np.asarray(field.distance(pts), np.float32)
    tf = getattr(field, "target_free_distance", None)
    if callable(tf):
        out["tf"] = np.asarray(tf(pts), np.float32)
    tier = getattr(field, "answer_tier", None)
    if callable(tier):
        out["tier_main"] = _tier_codes(tier(pts))
        out["tier_tf"] = _tier_codes(tier(pts, target_free=True))
    return out


def slice_points(centre: np.ndarray, plane: str, half: float, spacing: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Grid of a horizontal (``xy``, z = centre z) or vertical (``xz``, y = centre y) plane through ``centre``."""
    ticks = np.arange(-half, half + 1e-9, spacing)
    a, b = np.meshgrid(ticks, ticks, indexing="ij")
    pts = np.zeros(a.shape + (3,))
    if plane == "xy":
        pts[..., 0], pts[..., 1], pts[..., 2] = centre[0] + a, centre[1] + b, centre[2]
    elif plane == "xz":
        pts[..., 0], pts[..., 1], pts[..., 2] = centre[0] + a, centre[1], centre[2] + b
    else:
        raise ValueError(plane)
    return pts.reshape(-1, 3), a.shape


def cloud_camera_mask(fused) -> np.ndarray:
    """(N,) uint8 bitmask of the cameras that observed each fused point (head 1, left_wrist 2)."""
    bits = np.array([CAMERA_BIT.get(c.value, 0) for c in fused.cameras], np.uint8)
    obs_bits = bits[np.asarray(fused.obs_camera, np.int64)] if len(fused.obs_camera) else np.zeros(0, np.uint8)
    starts = np.asarray(fused.obs_offset[:-1], np.int64)
    if not len(starts):
        return np.zeros(0, np.uint8)
    return np.bitwise_or.reduceat(obs_bits, starts).astype(np.uint8)


def seed_argmax_camera(fused, indices: np.ndarray) -> np.ndarray:
    """Per point: the camera whose observation carried the largest per-camera attention (the rule
    ``AG3S._with_camera_provenance`` uses for the target peak). 0 head, 1 left_wrist, -1 none."""
    out = np.full(len(indices), -1, np.int8)
    for j, i in enumerate(np.asarray(indices, np.int64)):
        block = fused.observation_slice(int(i))
        local = fused.obs_attention[block]
        if local.size:
            cam = fused.cameras[int(fused.obs_camera[block][int(np.argmax(local))])].value
            out[j] = CAMERAS.index(cam) if cam in CAMERAS else -1
    return out


def front_end_probe(ag3s, observations, fusion) -> Optional[dict]:
    """Per-camera timing of the GPU front end's steps, re-run **after** the frame (not inside it).

    `constraint_set.profile` times the whole multi-camera front end as one stage (``scene_reconstruction``:
    back-projection, voxel downsample, the 4271-sphere depth robot mask, cloud cut, fusion). This re-runs
    ``device_recon.camera_front_end``'s steps one by one with the frame's own filter model / inflation /
    guard (read from the AG3S instance as the frame left them) and synchronizes around each, so the
    self-filter can be reported on its own. The mask is checked pixel-equal to the frame's
    (``CameraResult.depth_mask``); ``None`` when the GPU path is not in use.
    """
    torch = _torch()
    if torch is None:
        return None
    from AG3S.ag3s.stages import device_recon as dr
    from AG3S.ag3s.stages.robot_filter import depth_robot_mask_device

    dev = dr.recon_device()
    if dev is None:
        return None
    cfg = ag3s._frame_filter_config.pointcloud  # noqa: SLF001 -- the frame's effective config (guard radius)
    model = ag3s._frame_filter_model  # noqa: SLF001
    out = {}
    per_camera = {r.camera_id.value: r for r in fusion.per_camera}
    for o in observations:
        q = np.asarray(o.robot_state, np.float64)
        ms = {}
        torch.cuda.synchronize()
        t = time.perf_counter()
        depth_dev = dr.upload_depth(o.depth, dev)
        torch.cuda.synchronize()
        ms["upload"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        full = dr.backproject_device(depth_dev, o.camera_intrinsics, o.T_base_cam, cfg)
        torch.cuda.synchronize()
        ms["backproject"] = (time.perf_counter() - t) * 1e3
        if full is None:
            out[o.camera_id.value] = ms
            continue
        t = time.perf_counter()
        dr.downsample_and_cap_device(full.points, cfg)
        torch.cuda.synchronize()
        ms["downsample"] = (time.perf_counter() - t) * 1e3
        t = time.perf_counter()
        mask = depth_robot_mask_device(full, model, q, cfg, inflation=ag3s._frame_filter_inflation,  # noqa: SLF001
                                       guard_centre=ag3s._frame_guard_centre, fk_cache=None)  # noqa: SLF001
        torch.cuda.synchronize()
        ms["self_filter_mask"] = (time.perf_counter() - t) * 1e3
        ref = getattr(per_camera.get(o.camera_id.value), "depth_mask", None)
        ms["mask_equals_frame"] = (None if ref is None else bool(np.array_equal(np.asarray(mask.mask), np.asarray(ref.mask))))
        ms["n_spheres"] = int(getattr(model, "n_spheres", -1))
        out[o.camera_id.value] = ms
    return out


@dataclasses.dataclass
class FrameOptions:
    phase: str = "approach"
    manipulators: tuple = ("left",)
    save_clouds: bool = True
    slice_half: float = 0.25
    slice_spacing: float = 0.005
    exact_max_distance: float = EXACT_MAX_DISTANCE
    stage_probe: bool = True


class EpisodeRecorder:
    """Collects per-call arrays of one episode (ragged arrays as concatenation + offsets)."""

    def __init__(self):
        self.rows: dict[str, list] = {}
        self.ragged: dict[str, list] = {}

    def add(self, **values):
        for k, v in values.items():
            self.rows.setdefault(k, []).append(v)

    def add_ragged(self, **values):
        for k, v in values.items():
            self.ragged.setdefault(k, []).append(np.asarray(v))

    def arrays(self) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v) for k, v in self.rows.items()}
        for k, parts in self.ragged.items():
            out[k] = np.concatenate(parts) if parts else np.zeros(0)
            out[k + "_offsets"] = np.r_[0, np.cumsum([len(p) for p in parts])].astype(np.int64)
        return out


def run_frame(ag3s, observations, fingers, rs9, *, opts: FrameOptions, gt=None, gt_row=None, samples=None,
              spheres_all=None, recorder: EpisodeRecorder) -> dict:
    """Process one frame and record it. Returns the ``frames.jsonl`` row (without ids)."""
    torch = _torch()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    finger_record = apply_frame_state(ag3s, fingers, rs9)
    cs, debug = ag3s.process_multi_debug(observations, phase=opts.phase,
                                         active_manipulators=list(opts.manipulators), static_geometry=None)
    if torch is not None:
        torch.cuda.synchronize()
    t_process = time.perf_counter() - t0
    probe = front_end_probe(ag3s, observations, debug["fusion"]) if opts.stage_probe else None
    t1 = time.perf_counter()

    grounding, fusion = debug["grounding"], debug["fusion"]
    fused = fusion.cloud
    cloud = debug["filtered_cloud"]
    attention_cloud = debug["attention_cloud"]
    target = grounding.target
    seeds = np.asarray(grounding.seed_indices, np.int64)
    cam_mask = cloud_camera_mask(fused)
    seed_cam = seed_argmax_camera(fused, seeds)
    field = cs.esdf
    q7 = np.asarray(observations[0].robot_state, np.float64)

    # grounding / manipulated
    recorder.add(grounding_status=grounding.status.value, has_target=target is not None,
                 target_centroid=(np.asarray(target.centroid, np.float64) if target is not None else np.full(3, np.nan)),
                 target_confidence=(float(target.confidence) if target is not None else np.nan),
                 target_attention_score=(float(target.attention_score) if target is not None else np.nan),
                 target_seed_camera=(target.seed_camera.value if target is not None and target.seed_camera else ""),
                 target_supporting=(",".join(c.value for c in target.supporting_cameras) if target is not None else ""),
                 best_score=float(grounding.best_score), n_clusters=len(grounding.clusters),
                 robot_state9=rs9, q7=q7, fingers=np.array([fingers[k] for k in sorted(fingers)]))
    recorder.add_ragged(target_points=(np.asarray(target.points, np.float32) if target is not None else np.zeros((0, 3), np.float32)),
                        target_indices=(np.asarray(target.point_indices, np.int64) if target is not None else np.zeros(0, np.int64)),
                        seed_indices=seeds, seed_points=np.asarray(cloud.points[seeds], np.float32) if len(seeds) else np.zeros((0, 3), np.float32),
                        seed_camera_mask=cam_mask[seeds] if len(seeds) else np.zeros(0, np.uint8),
                        seed_argmax_camera=seed_cam)
    manipulated = cs.metrics.get("manipulated")
    recorder.add(manipulated_centroid=(np.asarray(manipulated["centroid"], np.float64) if manipulated else np.full(3, np.nan)),
                 manipulated_radius=(float(manipulated.get("radius") or np.nan) if manipulated else np.nan),
                 manipulated_state=(str(manipulated.get("state")) if manipulated else ""),
                 has_target_free=bool(getattr(field, "has_target_free", False)) if field is not None else False)

    # clouds
    n_points = len(cloud)
    if opts.save_clouds:
        support = np.asarray(debug["support_mask"], bool) if debug["support_mask"] is not None else np.zeros(n_points, bool)
        seed_flag = np.zeros(n_points, bool)
        seed_flag[seeds] = True
        target_flag = np.zeros(n_points, bool)
        if target is not None:
            target_flag[np.asarray(target.point_indices, np.int64)] = True
        recorder.add_ragged(cloud_points=np.asarray(cloud.points, np.float32), cloud_camera_mask=cam_mask,
                            cloud_attention=np.asarray(attention_cloud.attention, np.float16),
                            cloud_raw_attention=np.asarray(attention_cloud.peak_signal, np.float32),
                            cloud_support=support, cloud_seed=seed_flag, cloud_target=target_flag)

    # robot spheres (constraint model as AG3S used it: fingers already set) + all franka spheres
    model = ag3s.constraint_robot_model
    centres, radii = model.sphere_centers_numeric(q7)
    centres = np.asarray(centres, np.float64).reshape(-1, 3)
    sph = query_field(field, centres)
    recorder.add(sph_center=centres, sph_esdf_main=sph["main"], sph_esdf_tf=sph["tf"],
                 sph_tier_main=sph["tier_main"], sph_tier_tf=sph["tier_tf"])
    if spheres_all is not None:
        spheres_all.set_joint_parameters(fingers)
        c_all, _ = spheres_all.sphere_centers_numeric(q7)
        recorder.add(sph_all_center=np.asarray(c_all, np.float64).reshape(-1, 3))

    row: dict[str, Any] = {}
    # ground truth
    if gt is not None:
        P = gt.points_base(gt_row, samples)
        q = query_field(field, P)
        recorder.add(gt_esdf_main=q["main"], gt_esdf_tf=q["tf"], gt_tier_main=q["tier_main"], gt_tier_tf=q["tier_tf"])
        from benchmark.evaluators.libero_scene_gt import depth_residual

        for o in observations:
            recorder.add(**{f"gt_depth_residual_{o.camera_id.value}":
                            depth_residual(P, np.asarray(o.depth), o.camera_intrinsics, o.T_base_cam).astype(np.float16)})
        exact = gt.exact_distance(centres, gt_row, max_distance=opts.exact_max_distance)
        recorder.add(sph_exact=exact.astype(np.float32), sph_inside=gt.inside(centres, gt_row),
                     gt_target_position=gt.target_position_base(gt_row))
        centre = gt.target_position_base(gt_row)
        for plane in ("xy", "xz"):
            pts, shape = slice_points(centre, plane, opts.slice_half, opts.slice_spacing)
            sq = query_field(field, pts)
            recorder.add(**{f"slice_{plane}_main": sq["main"].reshape(shape).astype(np.float16),
                            f"slice_{plane}_tf": sq["tf"].reshape(shape).astype(np.float16)})
        recorder.add(slice_centre=centre)
        dist = exact.min(1)
        row["gt"] = {"sphere_exact_min_m": float(np.min(dist)) if np.isfinite(dist).any() else None}
    t_record = time.perf_counter() - t1

    esdf_stats = cs.metrics.get("esdf") or {}
    profile = {k: float(v) for k, v in (cs.profile or {}).items()}
    fm = fusion.metrics
    recorder.add(t_process_s=t_process, t_record_s=t_record, n_filtered_points=n_points,
                 n_points_before_fusion=int(fm.get("n_points_before_fusion", -1)),
                 n_self_filtered=int(fm.get("n_self_filtered", -1)), stage_ms=json.dumps(profile))
    for cam in CAMERAS:
        recorder.add(**{f"probe_self_filter_ms_{cam}": float((probe or {}).get(cam, {}).get("self_filter_mask", np.nan))})
    gpu = None
    if torch is not None:
        gpu = {"max_allocated": int(torch.cuda.max_memory_allocated()), "allocated": int(torch.cuda.memory_allocated()),
               "reserved": int(torch.cuda.memory_reserved())}
        recorder.add(gpu_max_allocated=gpu["max_allocated"], gpu_reserved=gpu["reserved"])
    row.update({
        "grounding_status": grounding.status.value, "has_target": target is not None,
        "target_centroid": None if target is None else _jsonable(target.centroid),
        "target_n_points": 0 if target is None else int(len(target.points)),
        "target_seed_camera": None if target is None or not target.seed_camera else target.seed_camera.value,
        "n_seeds": int(len(seeds)), "seed_argmax_camera_counts": {c: int((seed_cam == i).sum()) for i, c in enumerate(CAMERAS)},
        "manipulated": _jsonable(manipulated), "exclusion": _jsonable(cs.metrics.get("exclusion")),
        "status": _jsonable(cs.status), "validity": _jsonable(cs.validity), "notes": list(cs.notes),
        "n_filtered_points": int(n_points), "front_end_probe": probe,
        "fusion": _jsonable({k: v for k, v in fusion.metrics.items() if not isinstance(v, np.ndarray)}),
        "self_filter": _jsonable(cs.metrics.get("self_filter")),
        "finger_joints": _jsonable(finger_record),
        "esdf": _jsonable(esdf_stats), "has_target_free": bool(getattr(field, "has_target_free", False)) if field else False,
        "sphere_esdf_main_min_m": float(np.nanmin(sph["main"])) if len(sph["main"]) else None,
        "stage_ms": profile, "t_process_s": t_process, "t_record_s": t_record, "gpu_memory": gpu,
    })
    return row


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch if torch.cuda.is_available() else None


# --- episodes ------------------------------------------------------------------------------------------
def run_episode(ag3s, episode: dict, dev_dir: Path, *, mapping: str, opts: FrameOptions, gt_path: Optional[str],
                samples_cfg: dict, out_dir: Path, frames_out, limit_calls: Optional[int] = None,
                spheres_all=None) -> dict:
    from benchmark.evaluators.libero_scene_gt import SceneGT

    ag3s.reset()
    with np.load(dev_dir / episode["geom_npz"]) as g:
        geom = {k: g[k] for k in g.files}
    with np.load(episode["attn_npz"]) as a:
        grids = np.asarray(a["cell_grid_policy"], np.float32)
        obs_index = np.asarray(a["obs_index"]).astype(np.int64)
        global_step = np.asarray(a["global_step"]).astype(np.int64)
    gt = SceneGT(gt_path) if gt_path else None
    samples = gt.samples(**samples_cfg) if gt is not None else None
    row_of = {int(o): i for i, o in enumerate(np.asarray(geom["obs_index"]).reshape(-1))}
    recorder = EpisodeRecorder()
    calls = range(len(obs_index)) if not limit_calls else range(min(limit_calls, len(obs_index)))
    for k in calls:
        t = int(obs_index[k])
        row = row_of[t]
        observations, fingers, rs9, _ = build_observations(geom, row, grids[k], mapping, ag3s.robot_model)
        recorder.add(call_k=k, obs_index=t, global_step=int(global_step[k]), geom_row=row)
        info = run_frame(ag3s, observations, fingers, rs9, opts=opts, gt=gt,
                         gt_row=(gt.frame_row(t) if gt is not None else None), samples=samples,
                         spheres_all=spheres_all, recorder=recorder)
        info = {"stem": episode["stem"], "task_id": episode["task_id"], "call_k": k, "obs_index": t,
                "global_step": int(global_step[k]), **info}
        frames_out.write(json.dumps(_jsonable(info)) + "\n")
        frames_out.flush()
        print(f"  {episode['stem']} call {k + 1}/{len(calls)} {info['grounding_status']} "
              f"target={'y' if info['has_target'] else 'n'} {info['t_process_s'] * 1e3:.0f} ms", flush=True)
    arrays = recorder.arrays()
    if gt is not None:
        arrays.update({"gt_points_local": samples.points_local.astype(np.float32), "gt_point_geom": samples.geom,
                       "gt_point_instance": samples.instance,
                       "instance_names": np.array([x.name for x in gt.instances()]),
                       "instance_roles": np.array([x.role for x in gt.instances()])})
    model = ag3s.constraint_robot_model
    arrays.update({"sph_radius": np.asarray(model.sphere_centers_numeric(np.zeros(model.nq))[1], np.float64),
                   "sph_link": np.array(model.sphere_link_names),
                   "sph_authorized": np.array([l in set(ag3s.config.contact.links_for("left")) for l in model.sphere_link_names]),
                   "cameras": np.array(CAMERAS), "tiers": np.array(TIERS), "mapping": np.array(mapping)})
    if spheres_all is not None:
        arrays.update({"sph_all_radius": np.asarray(spheres_all.sphere_centers_numeric(np.zeros(spheres_all.nq))[1]),
                       "sph_all_link": np.array(spheres_all.sphere_link_names)})
    target = out_dir / "perception" / f"{episode['stem']}_perception.npz"
    with open(target, "xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"perception_npz": str(target.relative_to(out_dir)), "calls": len(calls),
            "gt": None if gt is None else {"path": gt_path, "sha256": gt.sha256, "samples": samples.per_instance(),
                                           "instances": gt.instance_table()}}


# --- settings / run ---------------------------------------------------------------------------------------
def gpu_compute_apps() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        return [f"nvidia-smi failed: {error!r}"]
    return [l.strip() for l in out.stdout.splitlines() if l.strip()]


def build_settings(args, sources: dict, scene_gt: Optional[dict], profile_entry: dict) -> dict:
    return {
        "step": STEP, "condition": CONDITION,
        "sources": sources, "scene_gt": scene_gt,
        "profile": profile_entry,
        "options": {"attention_mapping": args.attention_mapping, "esdf_bounds": args.esdf_bounds,
                    "phase": profile_entry["client_phase"], "active_manipulators": ["left"],
                    "save_clouds": not args.no_clouds, "limit_calls": args.limit_calls, "task_ids": args.task_ids,
                    "gt_samples": {"spacing": args.gt_spacing, "cap": args.gt_cap, "seed": args.gt_seed,
                                   "surfaces": "visual geoms (group 1)"},
                    "slices": {"planes": ["xy (z = GT target z)", "xz (y = GT target y)"], "half_m": args.slice_half,
                               "spacing_m": args.slice_spacing, "centre": "GT target body position (visualisation only)"},
                    "exact_max_distance_m": EXACT_MAX_DISTANCE,
                    "stage_probe": (None if args.no_stage_probe else
                                    "front_end_probe: per-camera GPU front-end steps re-run after each frame (timing only)"),
                    "execution_path": "none (Panda chunk = EE deltas; SL-5)", "static_geometry": "none (V11AB default)",
                    "grasp_latch": "none: set_grasp_active/latched False, set_subtask None, set_placed False every frame"},
        "privileged_inputs": "AG3S sees only the wire observation + attention; scene GT is used only for the "
                             "recorded ESDF queries / exact distances / slice placement, after process_multi_debug",
        "data_split": "dev (SL3 dev run, LIBERO init 0)",
        "interpreter": sys.executable,
        "environment": {key: os.environ.get(key) for key in
                        ("AG3S_STORAGE_ROOT", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS")},
        "commands": {"perception_replay": [sys.executable, *sys.argv]},
    }


def percentile_summary(values) -> dict:
    v = np.asarray(values, np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "mean": float(v.mean()), "max": float(v.max())}


def summarize(frames: list[dict]) -> dict:
    out: dict[str, Any] = {"calls": len(frames)}
    if not frames:
        return out
    out["first_call_t_process_s"] = frames[0]["t_process_s"]
    rest = frames[1:]
    out["t_process_s_after_first"] = percentile_summary([f["t_process_s"] for f in rest])
    firsts = {f["stem"] for f in frames if f["call_k"] == 0}
    out["t_process_s_excluding_episode_first"] = percentile_summary([f["t_process_s"] for f in frames if f["call_k"] != 0])
    stages = sorted({k for f in rest for k in f["stage_ms"]})
    out["stage_ms_after_first"] = {s: percentile_summary([f["stage_ms"][s] for f in rest if s in f["stage_ms"]]) for s in stages}
    for cam in CAMERAS:
        vals = [f["front_end_probe"][cam][step] for f in rest if f.get("front_end_probe") and cam in f["front_end_probe"]
                for step in ("self_filter_mask",) if step in f["front_end_probe"][cam]]
        out[f"probe_self_filter_mask_ms_{cam}_after_first"] = percentile_summary(vals)
    eq = [v.get("mask_equals_frame") for f in frames if f.get("front_end_probe") for v in f["front_end_probe"].values()]
    out["probe_mask_equals_frame"] = {"n": len(eq), "true": int(sum(e is True for e in eq))}
    gpu = [f["gpu_memory"]["max_allocated"] for f in frames if f.get("gpu_memory")]
    out["gpu_max_allocated_bytes"] = max(gpu) if gpu else None
    out["has_target_calls"] = int(sum(bool(f["has_target"]) for f in frames))
    out["has_target_free_calls"] = int(sum(bool(f["has_target_free"]) for f in frames))
    out["episodes"] = len(firsts)
    return out


def run(args) -> int:
    from AG3S.adapters.panda import model as pm
    from AG3S.adapters.panda import profile as pf
    from common.run_artifacts import create_run, record_execution

    if os.environ.get("AG3S_PROJECT_ROOT") != str(PROJECT_ROOT):
        raise RuntimeError("source scripts/setup/env.sh of this checkout first")
    sources = index_sources(args.source_run, args.attention_run, task_ids=args.task_ids)
    scene_gt = index_scene_gt(args.scene_gt, sources, limit_calls=args.limit_calls)
    if scene_gt is None and not args.no_gt:
        raise ValueError("--scene-gt DIR is required (or --no-gt for a run without GT queries)")
    profile_entry = pf.manifest_entry(esdf_bounds=args.esdf_bounds)
    undeclared = [r for r in profile_entry["config_diff_against_v11ab"] if r["reason"] == "UNDECLARED"]
    if undeclared:
        raise RuntimeError(f"Panda config differs from V11AB outside the declared overrides: {undeclared}")
    settings = build_settings(args, sources, scene_gt, profile_entry)
    if args.dry_run:
        # one observation per episode is built (CPU) to check the join and the attention attachment
        checks = []
        dev_dir = Path(sources["dev"]["run_dir"])
        filter_model = pm.build_panda_self_filter_model()
        for e in sources["episodes"]:
            with np.load(dev_dir / e["geom_npz"]) as g, np.load(e["attn_npz"]) as a:
                geom = {k: g[k] for k in g.files}
                row = int(np.flatnonzero(np.asarray(geom["obs_index"]) == int(a["obs_index"][0]))[0])
                obs, fingers, rs9, _ = build_observations(geom, row, np.asarray(a["cell_grid_policy"][0]),
                                                          args.attention_mapping, filter_model)
                checks.append({"stem": e["stem"], "calls": e["calls"], "cameras": [o.camera_id.value for o in obs],
                               "q7": obs[0].robot_state.tolist(), "fingers": fingers,
                               "attention_equals_audit_opencv": bool(np.array_equal(
                                   np.stack([o.attention_map for o in obs]), np.asarray(a["cell_grid_opencv"][0])))})
        print(json.dumps(_jsonable({"settings": settings, "checks": checks,
                                    "calls": sum(min(e["calls"], args.limit_calls or e["calls"]) for e in sources["episodes"])}),
                         indent=1, default=str))
        return 0

    apps = gpu_compute_apps()
    if apps and not args.allow_busy_gpu:
        raise RuntimeError(f"GPU has compute processes {apps}; run inside the shared lock when it is free "
                           "(or --allow-busy-gpu)")
    settings["gpu_compute_apps_at_start"] = apps
    run_dir = create_run(run_id=args.run_id, settings=settings)
    (run_dir / "perception").mkdir()
    print(run_dir, flush=True)
    record_execution(run_dir, settings={"role": "perception_replay", "argv": sys.argv, "pid": os.getpid()})
    result: dict[str, Any] = {"role": "perception_replay_result"}
    started = time.monotonic()
    try:
        t0 = time.monotonic()
        ag3s, info = pf.build_panda_ag3s(esdf_bounds=args.esdf_bounds)
        result["build_seconds"] = time.monotonic() - t0
        result["ag3s"] = info
        spheres_all = pm.build_panda_constraint_model(None)
        opts = FrameOptions(phase=profile_entry["client_phase"], save_clouds=not args.no_clouds,
                            slice_half=args.slice_half, slice_spacing=args.slice_spacing,
                            stage_probe=not args.no_stage_probe)
        frames: list[dict] = []
        with open(run_dir / "frames.jsonl", "x") as frames_out, open(run_dir / "perception_index.jsonl", "x") as index:
            for e in sources["episodes"]:
                gt_path = scene_gt["files"][e["stem"]]["path"] if scene_gt else None
                start = len(frames)
                out = run_episode(ag3s, e, Path(sources["dev"]["run_dir"]), mapping=args.attention_mapping, opts=opts,
                                  gt_path=gt_path, samples_cfg={"spacing": args.gt_spacing, "cap": args.gt_cap,
                                                                "seed": args.gt_seed},
                                  out_dir=run_dir, frames_out=frames_out, limit_calls=args.limit_calls,
                                  spheres_all=spheres_all)
                frames_out.flush()
                frames.extend(json.loads(l) for l in (run_dir / "frames.jsonl").read_text().splitlines()[start:])
                index.write(json.dumps(_jsonable({**{k: e[k] for k in ("stem", "suite", "task_id", "episode_idx", "status",
                                                                        "npz", "geom_npz", "npz_sha256", "geom_npz_sha256",
                                                                        "attn_npz", "attn_npz_sha256", "target")},
                                                   **out})) + "\n")
                index.flush()
        result.update(summarize(frames))
        result["gpu_compute_apps_at_end"] = gpu_compute_apps()
        result["pid"] = os.getpid()
        (run_dir / "summary.json").write_text(json.dumps(_jsonable(result), indent=1) + "\n")
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        record_execution(run_dir, settings=_jsonable(result))
        print(json.dumps({k: v for k, v in _jsonable(result).items() if k not in ("role", "ag3s")}, default=str), flush=True)
    return 0


def _int_list(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    from benchmark.evaluators.libero_scene_gt import SAMPLE_CAP, SAMPLE_SPACING

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-run", required=True, help="geometry-capture rollout run (name or path below results/benchmark)")
    parser.add_argument("--attention-run", required=True, help="SL3 attention audit run of that rollout")
    parser.add_argument("--scene-gt", default=None, help="scene_export.py output directory")
    parser.add_argument("--no-gt", action="store_true", help="run without GT queries (no --scene-gt)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--attention-mapping", choices=MAPPINGS, default="unmirror",
                        help="unmirror = SL3 mapping (default); mirror = policy-orientation grid as is (diagnosis)")
    parser.add_argument("--esdf-bounds", choices=("panda", "v11ab"), default="panda",
                        help="panda = V11AB box with floor z -0.10 (profile default); v11ab = default_bounds() (diagnosis)")
    parser.add_argument("--task-ids", type=_int_list, default=None)
    parser.add_argument("--limit-calls", type=int, default=None, help="first N calls of each episode")
    parser.add_argument("--no-clouds", action="store_true", help="do not store the filtered cloud per call")
    parser.add_argument("--gt-spacing", type=float, default=SAMPLE_SPACING)
    parser.add_argument("--gt-cap", type=int, default=SAMPLE_CAP)
    parser.add_argument("--gt-seed", type=int, default=0)
    parser.add_argument("--slice-half", type=float, default=0.25)
    parser.add_argument("--slice-spacing", type=float, default=0.005)
    parser.add_argument("--no-stage-probe", action="store_true",
                        help="skip the per-camera front-end timing re-run after each frame (front_end_probe)")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="index + checks + settings only; no run, no GPU")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.scene_gt and args.no_gt:
        raise SystemExit("--scene-gt and --no-gt are exclusive")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
