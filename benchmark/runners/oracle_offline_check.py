"""SL12 offline check of the visible-target oracle on a recorded LIBERO geometry run (no server, no GPU).

Two stages (two interpreters, the project's usual split):

``--stage render`` (``.venv-libero``, Python 3.8; ``LIBERO_CONFIG_PATH=.runtime/libero MUJOCO_GL=osmesa``): rebuilds the env of
each wanted episode with the recorded history (``env_history.episode_envs``), and for every policy-call frame sets the stored
simulator state and renders the geom-id segmentation exactly as the live client does (``camera.capture(segmentation=True)``:
state at ``env.step`` return = ``sim_state_at_return``; the ``sim_state_image`` render is kept next to it to measure the
instant difference), turns it into the code image with ``oracle_codes.OracleLabeler`` and writes the target object's mesh
vertices (world frame) of the same frame. Output: ``<out>/oracle_<stem>.npz``.

``--stage memory`` (``.venv-ag3s``): feeds the code images and the *recorded* depth / K / T_base_cam to the AG3S
``OracleMemory`` (the server's own class, the pipeline's cell / ROI / carving / band settings for a 5 mm fine voxel) and
compares the memory's target-code samples with the true target mesh: per call, the distance of each target sample to the
nearest mesh vertex of the true target object (and the same for the samples of every other code, to show they do not lie on
the target). Output: ``<out>/oracle_memory_<stem>.json``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "benchmark/runners")):
    if _p not in sys.path:
        sys.path.append(_p)


def stage_render(args):
    from benchmark.adapters.libero_panda import camera, env_history, oracle_codes as oc

    run = pathlib.Path(args.run).resolve()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stems = set(args.stems)
    for row, env, history in env_history.episode_envs(run, stems):
        stem = env_history.stem_of(row)
        g = np.load(str(run / "episodes/{}_geom.npz".format(stem)))
        e = np.load(str(run / row["npz"]))
        obs_index = np.asarray(g["obs_index"]).reshape(-1)
        row_of = {int(o): i for i, o in enumerate(obs_index)}
        calls = [int(t) for t in np.asarray(e["chunk_t"]).reshape(-1)]
        sim = env.sim
        m = sim.model._model
        tables = camera.model_tables(env)
        labeler = oc.OracleLabeler.from_env(env, tables, suite=row["suite"], task_id=int(row["task_id"]))
        inner = getattr(env, "env", env)
        bodies = oc.instance_body_ids(labeler.target, [str(b) for b in tables["model_body_names"]])
        geoms = [k for k in range(m.ngeom) if int(m.geom_bodyid[k]) in bodies and int(m.geom_type[k]) == 7]
        res = {c: [] for c in ("at_return", "image")}
        verts_world = []
        for t in calls:
            i = row_of[t]
            for key, state in (("at_return", g["sim_state_at_return"][i]), ("image", g["sim_state_image"][i])):
                sim.set_state_from_flattened(np.asarray(state).copy())
                sim.forward()
                frames = {}
                for source, wire in camera.CAMERA_TABLE:
                    seg = camera.opengl_rows_to_opencv(sim.render(camera_name=source, height=256, width=256,
                                                                  segmentation=True))
                    frames[wire] = {"segmentation_geom": np.where(seg[..., 0] == camera.MJOBJ_GEOM, seg[..., 1], -1)
                                    .astype(np.int16)}
                lab = labeler.labels(frames)
                res[key].append(np.stack([lab[w] for _, w in camera.CAMERA_TABLE]))
                if key == "at_return":
                    d = sim.data._data
                    pts = []
                    for k in geoms:
                        mid = int(m.geom_dataid[k])
                        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                        v = np.asarray(m.mesh_vert[va:va + vn], np.float64)
                        pts.append(v @ np.asarray(d.geom_xmat[k]).reshape(3, 3).T + np.asarray(d.geom_xpos[k]))
                    verts_world.append(np.concatenate(pts))
        n = min(len(v) for v in verts_world)
        np.savez_compressed(str(out / "oracle_{}.npz".format(stem)), calls=np.array(calls),
                            geom_rows=np.array([row_of[t] for t in calls]),
                            labels_at_return=np.stack(res["at_return"]), labels_image=np.stack(res["image"]),
                            target_vertices_world=np.stack([v[:n] for v in verts_world]),
                            record=np.array(json.dumps(labeler.record())), cameras=np.array(["head", "left_wrist"]))
        print(json.dumps({"stem": stem, "calls": len(calls), **labeler.record(),
                          "target_px_at_return": [int((a == 1).sum()) for a in res["at_return"][:3]],
                          "history": history.get("resets_before")}), flush=True)


def stage_memory(args):
    from AG3S.ag3s.runtime.oracle_segmentation import CODE_TARGET, OracleMemory, OracleView
    from scipy.spatial import cKDTree

    run = pathlib.Path(args.run).resolve()
    fine = 0.005
    for stem in args.stems:
        o = np.load(str(pathlib.Path(args.out) / "oracle_{}.npz".format(stem)))
        g = np.load(str(run / "episodes/{}_geom.npz".format(stem)))
        key = "labels_at_return" if args.labels == "at_return" else "labels_image"
        mem = OracleMemory(cell=0.5 * fine, roi_margin=2.0 * 6.0 * fine + fine, carve_tol=2.0 * fine, depth_min=0.0,
                           depth_max=2.0, band_depth=0.030)
        scale = float(np.asarray(g["depth_scale_mm"])) / 1000.0
        T_world_base = np.asarray(g["T_world_base"])
        rows_out = []
        for k, i in enumerate(o["geom_rows"]):
            views = []
            for c, cam in enumerate(["head", "left_wrist"]):
                depth = np.asarray(g["depth_mm/" + cam][i], np.float64) / 1000.0
                views.append(OracleView(depth_m=depth, K=g["K/" + cam][i], T_base_cam=g["T_base_cam/" + cam][i],
                                        labels=o[key][k, c], name=cam))
            rec = mem.update(views)
            Tw = T_world_base[i]
            tgt = mem.points[mem.codes == CODE_TARGET]
            oth = mem.points[(mem.codes != CODE_TARGET)]
            to_world = lambda p: p @ Tw[:3, :3].T + Tw[:3, 3]
            tree = cKDTree(o["target_vertices_world"][k])
            row = {"call": int(o["calls"][k]), "n_target_samples": int(len(tgt)), "n_other_samples": int(len(oth)),
                   "n_target_samples_this_frame": rec["n_target_samples_this_frame"]}
            if len(tgt):
                d = tree.query(to_world(tgt))[0] * 1e3
                row.update({"target_to_true_mesh_mm": {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
                                                        "max": float(d.max())}})
            if len(oth):
                d = tree.query(to_world(oth))[0] * 1e3
                row["other_within_3mm_of_true_mesh"] = int((d <= 3.0).sum())
            rows_out.append(row)
        last = rows_out[-1]
        summary = {"stem": stem, "labels": args.labels, "calls": len(rows_out),
                   "median_of_per_call_median_mm": float(np.median([r["target_to_true_mesh_mm"]["median"]
                                                                    for r in rows_out if "target_to_true_mesh_mm" in r])),
                   "worst_per_call_p90_mm": float(max(r["target_to_true_mesh_mm"]["p90"]
                                                      for r in rows_out if "target_to_true_mesh_mm" in r)),
                   "calls_without_target_sample": int(sum(r["n_target_samples"] == 0 for r in rows_out)),
                   "other_samples_within_3mm_of_target_max": int(max(r.get("other_within_3mm_of_true_mesh", 0)
                                                                     for r in rows_out)),
                   "final": last, "record": json.loads(str(o["record"])), "per_call": rows_out}
        (pathlib.Path(args.out) / "oracle_memory_{}_{}.json".format(stem, args.labels)).write_text(
            json.dumps(summary, indent=1))
        print(json.dumps({k: v for k, v in summary.items() if k != "per_call"}), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("render", "memory"), required=True)
    ap.add_argument("--run", required=True, help="geometry-capture run directory (read only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stems", nargs="+", required=True, help="e.g. libero_spatial_t1_e0")
    ap.add_argument("--labels", choices=("at_return", "image"), default="at_return")
    args = ap.parse_args(argv)
    (stage_render if args.stage == "render" else stage_memory)(args)
    sys.stdout.flush()
    if args.stage == "render":
        import os
        os._exit(0)


if __name__ == "__main__":
    main()
