"""IFG-CL1 diagnostic — are the seeds the fine target-free tier KEPT as non-target actually the target (GT)?

Per run (closed-loop record + its server segment from analysis/<run>.json), per chunk k (client planning step k):
  replay the recorded qpos in the run's snapshot scene, take the GT target fruit's geoms (mesh vertices / primitive) in the base
  frame, and classify every `target_free_nontarget_kept_ijk` / `target_free_removed_ijk` voxel centre as GT-target when it lies
  inside the convex hull of the target geometry or within TOL of it (plane distance bound).
Voxel centre convention (origin + ijk·vs vs origin + (ijk+½)·vs) is picked per run by the smaller median hull distance of the
REMOVED seeds (the ones the extent attributed to the target) and recorded.

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_cl1_keptseeds --cl1 <run>/verify/IFG-CL1 --out <json> [--runs NAME ...]
"""
import argparse, glob, json, os, pathlib

import numpy as np
from scipy.spatial import ConvexHull

TOL = 0.005


def target_points(m, d, body):
    import mujoco
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid:
            continue
        R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]; a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n]
        else:  # primitive: sample its bounding box corners scaled by size (sphere / ellipsoid / capsule rough hull)
            s = m.geom_size[g]
            v = np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1] if s[1] else -s[0], s[1] if s[1] else s[0])
                          for z in (-s[2] if s[2] else -s[0], s[2] if s[2] else s[0])])
        pts.append(v @ R.T + p)
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def hull_dist(hull, x):
    """max plane distance (>0 outside, <=0 inside) — a lower bound of the true outside distance."""
    eq = hull.equations
    return (x @ eq[:, :3].T + eq[:, 3]).max(1)


def run_one(cl1, name):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene
    a = json.loads((cl1 / "analysis" / f"{name}.json").read_text())
    arm = a["arm"]; rd = pathlib.Path(a["run_dir"])
    rec = load_run(rd / "ag3s/run_0000")
    man = json.loads((rd / "manifest.json").read_text())
    target = man["scene"]["episode_target_fruit"]
    seg = a["server_segment"]; d0 = pathlib.Path(seg["first"]).parent
    i0 = int(pathlib.Path(seg["first"]).stem.split("_")[1]); n = int(seg["n"])
    scene = TransportScene(rec.meta["model_xml"], height=64, width=64, settle_steps=0)
    rows = []
    try:
        for k in range(min(n, len(rec.steps))):
            f = d0 / f"chunk_{i0 + k:05d}.npz"
            z = np.load(f, allow_pickle=True)
            if "target_free_nontarget_kept_ijk" not in z.files:
                continue
            kept, rem = z["target_free_nontarget_kept_ijk"], z["target_free_removed_ijk"]
            if len(kept) + len(rem) == 0:
                continue
            o, vs = np.asarray(z["target_free_trace_origin"], float), float(z["target_free_trace_voxel_size"])
            pose_scene(scene, rec.steps[k])
            Tb = np.linalg.inv(scene.body_pose("base"))
            tp = target_points(scene.model, scene.data, target)
            tp = tp @ Tb[:3, :3].T + Tb[:3, 3]
            hull = ConvexHull(tp)
            res = {}
            for off in (0.0, 0.5):
                dr = hull_dist(hull, o + (rem + off) * vs) if len(rem) else np.zeros(0)
                res[off] = float(np.median(dr)) if len(dr) else np.inf
            off = min(res, key=res.get)
            dk = hull_dist(hull, o + (kept + off) * vs) if len(kept) else np.zeros(0)
            dr = hull_dist(hull, o + (rem + off) * vs) if len(rem) else np.zeros(0)
            rows.append(dict(k=k, t_step=int(rec.steps[k].t_step), n_kept=int(len(kept)), n_removed=int(len(rem)),
                             kept_target=int((dk <= TOL).sum()), removed_target=int((dr <= TOL).sum()),
                             removed_median_dist_mm=float(np.median(dr) * 1e3) if len(dr) else None, offset=off))
    finally:
        scene.close()
    pre = [r for r in rows if a.get("grasp_t") is None or r["t_step"] < a["grasp_t"]]
    agg = lambda rs, key: int(sum(r[key] for r in rs))
    return dict(name=name, arm=arm, episode=a["episode"], success=a["success"], failure_mode=a["failure_mode"], grasp_t=a.get("grasp_t"),
                target=target, n_chunks=len(rows), n_pre=len(pre),
                pre=dict(kept=agg(pre, "n_kept"), kept_target=agg(pre, "kept_target"), removed=agg(pre, "n_removed"),
                         removed_target=agg(pre, "removed_target")),
                all=dict(kept=agg(rows, "n_kept"), kept_target=agg(rows, "kept_target"), removed=agg(rows, "n_removed"),
                         removed_target=agg(rows, "removed_target")),
                removed_median_dist_mm=float(np.median([r["removed_median_dist_mm"] for r in rows if r["removed_median_dist_mm"] is not None]))
                if rows else None, chunks=rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cl1", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--runs", nargs="*", default=None); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    names = a.runs or sorted(pathlib.Path(p).stem for p in glob.glob(str(a.cl1 / "analysis" / "*.json")))
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(a.workers, 4)) as p:
        out = p.starmap(run_one, [(a.cl1, n) for n in names])
    a.out.write_text(json.dumps(out, indent=1, default=float))
    for r in out:
        print(r["name"], r["failure_mode"], "pre kept", r["pre"]["kept"], "kept_GTtarget", r["pre"]["kept_target"],
              "removed", r["pre"]["removed"], "removed_GTtarget", r["pre"]["removed_target"], "rem_med_mm", round(r["removed_median_dist_mm"] or -1, 1))


if __name__ == "__main__":
    main()
