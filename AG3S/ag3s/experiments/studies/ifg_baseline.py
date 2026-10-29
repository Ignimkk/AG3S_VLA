"""IFG baseline arms — the serving grounding (T54 arms) on the dump frames, `current` condition.

User ruling 2026-10-15: baselines run in T54's ``current`` condition first (one AG3S, `reset()` before
every (frame, arm) call: no TargetConfirm history), the same single-frame setting as G / FG.

Reuses T54 unchanged: `t54_study.record_inputs` (record / action npz / noun extract / server_proc
pairing), `t54_study.build_maps`, `t54_replay.make_builder("c1a", ...)` (the rollout server's own
AG3S flags, legacy ESDF), `t54_replay.ground_arm` with `t47b_replay._frames` (MuJoCo osmesa replay)
and `t47b_replay._ground` (the serving call). The only addition is a `ground_fn` wrapper that keeps
what the serving call chose: the target cluster points (`TargetGeometry.points`, "region") and its
`bounding_geometry` primitive ("ball"), plus the fusion voxel size.

Stage `replay` writes `<out>/raw/<frame_id>.npz` + `<out>/raw/<frame_id>.json` per (frame) with every
arm. Stage `eval` maps them onto the IFG evaluation voxels (same `ifg_graph.build_geometry` grid as
G / FG) and appends rows `<out>/rows/<frame_id>.jsonl`:
  region  eval voxel is target iff its fusion cell floor(p / fusion_voxel) holds a chosen point
  ball    eval voxel centre inside the chosen primitive (sphere / box / capsule / ellipsoid)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import time

import numpy as np

ARMS = ("L8H2", "N", "B")
RECORDS_RUN = {"T53": "/mnt/dev/AG3S_VLA-storage/results/development/20261013T100210+0900_4a540f1a18d7_cb037c425d2f",
               "T54c": "/mnt/dev/AG3S_VLA-storage/results/development/20261014T133115+0900_cf38c9dbca7b_5697bbdb5190"}
REGISTRY = {"T53": "t53", "T54c": "t54c"}


def _prim_dict(p):
    if p is None:
        return None
    return dict(type=str(getattr(p.type, "value", p.type)), center=np.asarray(p.center, float).tolist(),
                dimensions=np.asarray(p.dimensions, float).tolist(),
                orientation=np.asarray(p.orientation, float).tolist())


def inside_primitive(pts: np.ndarray, prim: dict) -> np.ndarray:
    c = np.asarray(prim["center"]); d = np.asarray(prim["dimensions"]); Rm = np.asarray(prim["orientation"])
    loc = (pts - c) @ Rm  # world -> local
    t = prim["type"]
    if t == "sphere":
        return np.linalg.norm(pts - c, axis=1) <= d[0]
    if t == "box":
        return np.all(np.abs(loc) <= d, axis=1)
    if t == "ellipsoid":
        return np.sum((loc / np.maximum(d, 1e-9)) ** 2, axis=1) <= 1.0
    if t == "capsule":  # dimensions = (radius, radius, half-length) along local z
        z = np.clip(loc[:, 2], -d[2], d[2])
        return np.linalg.norm(loc - np.stack([np.zeros_like(z), np.zeros_like(z), z], 1), axis=1) <= d[0]
    raise ValueError(f"primitive {t}")


def _frames_masked(record_path, *, target, frames, mask_dir, frame_prefix):
    """`t47b_replay._frames` with the IFG render robot mask applied to the captured depth (masked pixels -> 0 m,
    dropped by the serving back-projection). Serving's sphere self-filter is switched off by the caller."""
    from AG3S.ag3s.experiments.live.exclusion_e2 import gt_points, label_classes, point_labeler
    from AG3S.ag3s.experiments.sources.policy_record import (CAMERA_BINDINGS, POLICY_CAMERA_NAMES, control_step_ms,
                                                              load_run, phase_boundaries_for_path, phase_for,
                                                              pose_scene, replay_scene)
    from AG3S.trajopt import wire

    record = load_run(record_path)
    n_f = len(record)
    mj_cams = [CAMERA_BINDINGS[c][0] for c in POLICY_CAMERA_NAMES]
    scene = replay_scene(record)
    boundaries, _ = phase_boundaries_for_path(record_path)
    dt = control_step_ms(record) / 1000.0
    classes = label_classes(scene, target)
    try:
        for i in frames:
            step = record.steps[i]
            pose_scene(scene, step)
            captured = {c: scene.capture(c) for c in mj_cams}
            with np.load(pathlib.Path(mask_dir) / f"{frame_prefix}_f{i:03d}.npz") as z:
                rm = z["robot_mask"]
            depth = {c: np.where(rm[ci], 0.0, f.depth) for ci, (c, f) in enumerate(captured.items())}
            gpts, glab = gt_points(captured.values(), classes)
            phase = phase_for(step.t_step, boundaries)
            stamp = float(step.t_step) * dt
            req = wire.pack_request(
                {}, cameras=mj_cams, depth=depth,
                intrinsics={c: f.camera_intrinsics for c, f in captured.items()},
                extrinsics={c: f.T_base_cam for c, f in captured.items()},
                robot_state={c: f.robot_state for c, f in captured.items()},
                stamps={c: stamp for c in mj_cams}, phase=phase, seq=int(i))
            _, sc = wire.strip_request(req)
            yield {"i": int(i), "n_f": n_f, "step": step, "phase": phase, "sc": sc, "scene": scene, "mj_cams": mj_cams,
                   "gt_centre": scene.body_position_in_base(target), "label_of": point_labeler(gpts, glab, radius=0.003)}
    finally:
        scene.close()


def _replay_episode(spec):
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from AG3S.ag3s.experiments.studies import t47b_replay as br
    from AG3S.ag3s.experiments.studies import t54_arms as arms_mod
    from AG3S.ag3s.experiments.studies import t54_replay as rp
    from AG3S.ag3s.experiments.studies import t54_study as st
    from AG3S.ag3s.experiments.studies.t49_noun_query_extract import load_extract

    ds, key, frames, out = spec["ds"], spec["key"], spec["frames"], pathlib.Path(spec["out"])
    args = st.build_parser().parse_args(["--run-dir", str(out), "--set", "test", "--stage", "inputs",
                                         "--registry", REGISTRY[ds], "--records-run", RECORDS_RUN[ds],
                                         "--arms", *ARMS, "--conditions", "current"])
    args._t53_cfg = st.configure_t53(args.t53_run, args.registry)
    info = st.record_inputs("test", key, args)
    if info["problems"]:
        return dict(key=key, error=info["problems"])
    arms = arms_mod.resolve_arms(list(ARMS))
    att, dn, agl, _ = st.load_action(pathlib.Path(info["npz"]))
    maps = st.build_maps(att, dn, agl, load_extract(pathlib.Path(info["noun"])), arms_mod.sources_needed(arms))
    del att
    if spec.get("mask_dir"):
        from AG3S.ag3s.runtime.asset_path import resolve_asset
        from AG3S.ag3s.experiments.sources.policy_record import load_run
        from AG3S.trajopt.serve_safe import build_ag3s

        _, kw, rec = rp.c1a_build_kwargs(rp.server_argv(pathlib.Path(info["server_proc"])))
        kw["self_filter"] = {**dict(kw.get("self_filter") or {}), "self_filter": False}
        mx = str(resolve_asset(load_run(pathlib.Path(info["record"])).model_xml, what="transport model"))
        build, prov = (lambda: build_ag3s(mx, **kw)), {"profile": "c1a+render_mask", **rec}
    else:
        build, prov = rp.make_builder("c1a", pathlib.Path(info["record"]), server_proc=pathlib.Path(info["server_proc"]))
    ag = build()
    fusion_voxel = float(ag.config.timing.fusion_voxel_size)
    captured = []

    def gfn(ag_, fr, att):
        cs, debug, ms = br._ground(ag_, fr, att)
        t = cs.target if bool(cs.has_target) else None
        captured.append(dict(points=None if t is None else np.asarray(t.points, np.float32),
                             prim=None if t is None else _prim_dict(t.bounding_geometry), ms=ms))
        return cs, debug, ms

    done = []
    frame_iter = (_frames_masked(pathlib.Path(info["record"]), target=info["target"], frames=frames,
                                 mask_dir=spec["mask_dir"], frame_prefix=f"{ds}_{key}") if spec.get("mask_dir") else
                  br._frames(pathlib.Path(info["record"]), target=info["target"], frames=frames, label_radius=0.003,
                             log=lambda m: None))
    for fr in frame_iter:
        fid = f"{ds}_{key}_f{fr['i']:03d}"
        arrays, meta = {}, dict(frame_id=fid, t_step=int(fr["step"].t_step), fusion_voxel=fusion_voxel,
                               profile=prov.get("profile"), server_proc=info["server_proc"], arms={})
        for a in arms:
            ag.reset()
            r = rp.ground_arm(ag, fr, a, maps, label_of=fr["label_of"], gt_centre=fr["gt_centre"], ground_fn=gfn)
            c = captured.pop()
            if c["points"] is not None:
                arrays[f"{a.name}:points"] = c["points"]
            meta["arms"][a.name] = dict(has_target=bool(r["has_target"]), identity_correct=r.get("identity_correct"),
                                        prim=c["prim"], ms=c["ms"], n_points=0 if c["points"] is None else len(c["points"]))
        np.savez(out / "raw" / f"{fid}.npz", **arrays)
        (out / "raw" / f"{fid}.json").write_text(json.dumps(meta, default=float))
        done.append(fid)
    return dict(key=key, done=done)


def stage_replay(a):
    from AG3S.ag3s.experiments.studies import ifg_reader as R

    (a.out / "raw").mkdir(parents=True, exist_ok=True)
    eps = {}
    for r in R.list_frames():
        if not (a.out / "raw" / f"{r.frame_id}.json").exists():
            eps.setdefault((r.ds, r.key), []).append(r.frame)
    jobs = [dict(ds=ds, key=k, frames=sorted(f), out=str(a.out), mask_dir=(str(a.render_mask) if a.render_mask else None))
            for (ds, k), f in sorted(eps.items())]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(a.workers, 4)) as pool:
        for res in pool.imap_unordered(_replay_episode, jobs):
            print(res, f"({time.time() - t0:.0f}s)", flush=True)


def stage_eval(a):
    from AG3S.ag3s.experiments.studies import ifg_graph as G
    from AG3S.ag3s.experiments.studies import ifg_reader as R

    (a.out / "rows").mkdir(parents=True, exist_ok=True)
    cfg = G.GraphConfig()
    refs = {r.frame_id: r for r in R.list_frames()}
    for jf in sorted((a.out / "raw").glob("*.json")):
        fid = jf.stem
        dst = a.out / "rows" / f"{fid}.jsonl"
        if dst.exists() or fid not in refs:
            continue
        meta = json.loads(jf.read_text())
        ref = refs[fid]
        with np.load(a.s0 / "replay" / f"{fid}.npz") as z:
            depth, rmask, K, T = z["depth"], z["robot_mask"], z["K"], z["T_base_cam"]
        if a.mask_dir_eval is not None:  # eval grid is mask-independent except robot_removed bookkeeping
            with np.load(a.mask_dir_eval / f"{fid}.npz") as z:
                rmask = z["robot_mask"]
        with np.load(a.s0 / "labels" / f"{fid}.npz") as z:
            class_map, classes = z["class_map"], [str(c) for c in z["classes"]]
        geo = G.build_geometry(depth, K, T, rmask, class_map, classes, cfg)
        arrs = np.load(a.out / "raw" / f"{fid}.npz")
        vx = meta["fusion_voxel"]
        ev_cell = np.floor(geo.eval_pos / vx).astype(np.int64)
        rows = []
        for arm, am in meta["arms"].items():
            for kind in ("region", "ball"):
                m = np.full(len(geo.eval_pos), -1.0)
                if am["has_target"]:
                    if kind == "region" and f"{arm}:points" in arrs.files:
                        cells = {tuple(c) for c in np.floor(arrs[f"{arm}:points"] / vx).astype(np.int64)}
                        hit = np.fromiter((tuple(c) in cells for c in ev_cell), bool, len(ev_cell))
                    elif kind == "ball" and am["prim"] is not None:
                        hit = inside_primitive(geo.eval_pos, am["prim"])
                    else:
                        hit = np.zeros(len(geo.eval_pos), bool)
                    m[hit] = 1.0
                # evaluate on eval voxels directly: identity mapping node_eval = arange(V)
                ge = G.dataclasses.replace(geo, node_eval=np.arange(len(geo.eval_pos)))
                ev = G.evaluate(ge, m, 0.5, cfg)
                rows.append(dict(frame_id=fid, episode_id=ref.episode_id, ds=ref.ds, moment=ref.moment, kind=ref.kind,
                                 split="dev", level="voxel", arm_id=f"T54{kind}{'+rm' if a.mask_dir_eval else ''}:{arm}", method=f"T54{kind}",
                                 seed_source=arm, threshold=0.5, voxel_size=cfg.voxel,
                                 valid=ev.get("missing_reason") is None, missing_reason=ev.get("missing_reason"),
                                 has_target=am["has_target"], identity_correct=am["identity_correct"],
                                 ms_solve=am["ms"], fusion_voxel=vx, prim_type=(am["prim"] or {}).get("type"), **ev))
        with open(dst, "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        print(fid, [(r["arm_id"], round(r["iou"] or 0, 3)) for r in rows], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["replay", "eval"], required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--s0", type=pathlib.Path, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--render-mask", type=pathlib.Path, default=None,
                    help="IFG R6 render robot mask dir: masked depth + serving self-filter off (filter-matched B0)")
    ap.add_argument("--mask-dir-eval", type=pathlib.Path, default=None, help="eval: same mask for the IFG eval grid")
    a = ap.parse_args()
    {"replay": stage_replay, "eval": stage_eval}[a.stage](a)


if __name__ == "__main__":
    main()
