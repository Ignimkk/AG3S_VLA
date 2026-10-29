"""AG3S-TASK-DYNAMIC-MAP §4 — the **served** fields of a closed-loop kit against MuJoCo ground truth.

Unlike `dynamic_field_replay` (which re-runs only the builder and so misses the server's held-object handling), this
reads the fields the server actually built and recorded (`--record-constraints` chunk npz: `esdf_distance` = the
main fine tier, `esdf_free_distance` = the target-free fine tier the contact-authorised hand reads) and compares them,
chunk by chunk, with the MuJoCo state of the same request (policy record `step_<j>.npz` qpos; depth / K / T for the
visibility split). Nothing is re-computed by AG3S.

Region of interest (fixed before results, identical for every configuration — task §4): the box of the target's
first observed GT surface ∪ the crate, ± `ROI_PAD`; ∪ the target's current GT surface ± `ROI_PAD`.

Per chunk and layer (`main` / `hand`), inside the region and the recorded fine grid:
  residue          surface voxels (|value| ≤ half a voxel: the cuRobo seed sites, not object interiors) farther than
                   `STALE_MM` from every current GT surface (non-robot geoms, floor planes) and robot surface sample
  target_old       residue within `STALE_MM` of a pose the target's surface had at an earlier chunk and has since left
                   by more than `MOVED_MM` (centroid)
  moved_old        the same for any other movable body (and not target_old) — a body that never moved leaves no memory
  missed           GT surface samples (non-robot, never the target) whose field value is > `MISSED_MM`;
                   split visible now (depth agrees within `VISIBLE_MM` in a camera) / occluded now but seen before

  MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 .venv-openpi-live/bin/python -m \\
      AG3S.ag3s.experiments.studies.task_dynamic_map_eval --kit <V> --model-xml <xml> --out <dir> \\
      --configurations B B_TL B_ROI B_TL_ROI [--episodes 211 …]
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re

import numpy as np

from AG3S.ag3s.experiments.studies.dynamic_field_replay import CAMS, GroundTruth

STALE_MM = 15.0
MISSED_MM = 10.0
VISIBLE_MM = 10.0
ROI_PAD = 0.10
MOVED_MM = 10.0
TARGET = "apple"


def obstacle_profile(run_dir: pathlib.Path) -> tuple:
    """`(config path, profile)` from the run's `.start` file (`obstacle=<path>:<profile>`), or `(None, "clear")`."""
    start = pathlib.Path(str(run_dir) + ".start")
    if start.exists():
        for line in start.read_text().splitlines():
            if line.startswith("obstacle=") and ":" in line[len("obstacle="):]:
                path, prof = line[len("obstacle="):].rsplit(":", 1)
                return (path or None), (prof or "clear")   # a clear scene is written as "obstacle=:"
    return None, "clear"


def scene_truth(model_xml: str, obstacle: tuple) -> GroundTruth:
    """`GroundTruth` of the client's scene: the obstacle slots placed as the client placed them (`activate`), or
    parked for a clear scene, then the geoms re-sampled (activation resizes slot geoms)."""
    from rby1_manipulation.simulation.pick_place_obstacles import (PickPlaceObstacleManager,
                                                                   load_pick_place_obstacle_config)

    gt = GroundTruth(model_xml)
    path, prof = obstacle
    if path is not None:
        mgr = PickPlaceObstacleManager(gt.m, gt.d, load_pick_place_obstacle_config(path))
        mgr.activate(prof) if prof != "clear" else mgr.park_all()
    else:
        try:
            from rby1_manipulation.simulation.pick_place_obstacles import DEFAULT_CONFIG
            PickPlaceObstacleManager(gt.m, gt.d, load_pick_place_obstacle_config(DEFAULT_CONFIG)).park_all()
        except RuntimeError:   # a model without obstacle slots
            pass
    gt.local = [gt._sample_local(g) for g in gt.geoms]
    gt.local_full = [p.copy() for p in gt.local]
    return gt


def episodes_of(kit: pathlib.Path, conf: str) -> dict:
    """{episode: (policy-record dir, [(request index, chunk file, grasp state), …])} for one configuration.

    Server directories are attributed by their `arm` file (several configurations share the policy seeds)."""
    runs = {}
    for r in sorted(glob.glob(str(kit / "runs" / conf / f"{conf}_ep*_s*"))):
        mo = re.search(r"_ep(\d+)_s(\d+)$", r)
        if mo and pathlib.Path(r).is_dir():
            runs[int(mo.group(2))] = (int(mo.group(1)), pathlib.Path(r) / "ag3s" / "run_0000")
    out = {}
    for sdir in sorted(glob.glob(str(kit / "runs" / "server_*"))):
        arm = pathlib.Path(sdir) / "arm"
        if not arm.exists() or arm.read_text().strip() != conf:
            continue
        for rdir in sorted(glob.glob(sdir + "/server_constraints/run_*")):
            seg, j = None, -1
            for f in sorted(glob.glob(rdir + "/chunk_*.npz")):
                with np.load(f) as z:
                    s = json.loads(str(z["summary_json"]))
                seed = (s.get("policy_seed") or {}).get("episode")
                j = j + 1 if seed == seg else 0
                seg = seed
                if seed in runs:
                    ep, pdir = runs[seed]
                    out.setdefault(ep, (pdir, []))[1].append((j, f, (s.get("grasp") or {}).get("state")))
    return out


class RobotSurface:
    """Surface samples of the robot's collision geoms (the `GroundTruth` sampler, robot side)."""

    def __init__(self, gt: GroundTruth):
        m, mj = gt.m, gt.mj
        self.gt = gt
        self.geoms, self.local = [], []
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            b = int(m.geom_bodyid[g])
            while b > 0 and b != gt.base:
                b = int(m.body_parentid[b])
            if b != gt.base or m.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE:
                continue
            p = gt._sample_local(g)
            if len(p):
                self.geoms.append(g)
                self.local.append(p)

    def at(self) -> np.ndarray:
        """Base-frame samples at the qpos of the last `gt.at` call."""
        d = self.gt.d
        R, o = self.gt.base_pose
        out = [((p @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) - o) @ R for g, p in zip(self.geoms, self.local)]
        return np.vstack(out) if out else np.zeros((0, 3))


def _grid(z, prefix):
    if f"{prefix}_distance" not in z.files:
        return None
    return (np.asarray(z[f"{prefix}_distance"], np.float32), np.asarray(z[f"{prefix}_origin"], np.float64).reshape(3),
            float(z[f"{prefix}_voxel_size"]))


def _lookup(grid, p):
    """Nearest-voxel value at `p` (N, 3); NaN outside the grid."""
    v, o, vs = grid
    idx = np.rint((p - o) / vs).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.asarray(v.shape)), axis=1)
    out = np.full(len(p), np.nan)
    out[ok] = v[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    return out


def _surface(grid):
    """The field's surface voxels (the cuRobo seed sites, |value| ≤ half a voxel) — not object interiors."""
    v, o, vs = grid
    ijk = np.argwhere(np.abs(v) <= 0.5 * vs)
    return o + ijk * vs


def _visible(p, step):
    vis = np.zeros(len(p), bool)
    for cam in CAMS:
        if f"depth_{cam}" not in step.files or not len(p):
            continue
        d = np.asarray(step[f"depth_{cam}"], np.float64) / 1000.0
        K = np.asarray(step[f"K_{cam}"], np.float64)
        T = np.asarray(step[f"T_base_cam_{cam}"], np.float64)
        pc = (p - T[:3, 3]) @ T[:3, :3]
        zc = pc[:, 2]
        ok = zc > 1e-6
        u = np.full(len(p), -1)
        v = np.full(len(p), -1)
        u[ok] = np.rint(K[0, 0] * pc[ok, 0] / zc[ok] + K[0, 2]).astype(int)
        v[ok] = np.rint(K[1, 1] * pc[ok, 1] / zc[ok] + K[1, 2]).astype(int)
        h, w = d.shape
        ins = np.flatnonzero(ok & (u >= 0) & (u < w) & (v >= 0) & (v < h))
        dd = d[v[ins], u[ins]]
        vis[ins[(dd > 0) & (np.abs(dd - zc[ins]) <= VISIBLE_MM / 1000.0)]] = True
    return vis


def run_episode(pdir: pathlib.Path, chunks, gt: GroundTruth, robot: RobotSurface) -> list[dict]:
    from scipy.spatial import cKDTree

    t_idx = [i for i, (k, n, b) in enumerate(gt.kind) if n == TARGET]
    crate_idx = [i for i, (k, n, b) in enumerate(gt.kind) if k == "crate"]
    bodies = sorted({gt.kind[i][2] for i in gt.movable})
    t_body = {gt.kind[i][2] for i in t_idx}
    hist: dict = {b: [] for b in bodies}   # body → [(centroid, samples)] of poses it has left
    rows, roi0 = [], None
    seen = None
    for j, chunk_file, state in chunks:
        sp = pdir / f"step_{j:05d}.npz"
        if not sp.exists():
            continue
        step = np.load(sp)
        samples, planes = gt.at(np.asarray(step["qpos"], np.float64))
        rob = robot.at()
        tgt = np.vstack([samples[i] for i in t_idx]) if t_idx else np.zeros((0, 3))
        if roi0 is None:
            base = [tgt] + [samples[i] for i in crate_idx]
            allp = np.vstack([b for b in base if len(b)])
            roi0 = (allp.min(0) - ROI_PAD, allp.max(0) + ROI_PAD)
            seen = [np.zeros(len(s), bool) for s in samples]
        boxes = [roi0] + ([(tgt.min(0) - ROI_PAD, tgt.max(0) + ROI_PAD)] if len(tgt) else [])

        def in_roi(p, grow=0.0):
            k = np.zeros(len(p), bool)
            for lo, hi in boxes:
                k |= np.all((p >= lo - grow) & (p <= hi + grow), axis=1)
            return k

        # distances are judged within STALE_MM, so samples beyond the region + 5 cm cannot matter (speed)
        near = [in_roi(s, 0.05) for s in samples]
        gt_all = np.vstack([s[k] for s, k in zip(samples, near) if k.any()] or [np.zeros((0, 3))])
        rob = rob[in_roi(rob, 0.05)]
        gt_tree = cKDTree(gt_all if len(gt_all) else np.full((1, 3), 1e6))
        rob_tree = cKDTree(rob if len(rob) else np.full((1, 3), 1e6))
        # where bodies were: only poses a body has since left by more than MOVED_MM (a body that never moved —
        # the crate, a fruit at rest — leaves no memory)
        cur = {b: np.vstack([samples[i] for i in gt.movable if gt.kind[i][2] == b and len(samples[i])]
                            or [np.zeros((0, 3))]) for b in bodies}
        cen = {b: (p.mean(0) if len(p) else None) for b, p in cur.items()}
        t_old_pts, o_old_pts = [], []
        for b in bodies:
            for c, p in hist[b]:
                if cen[b] is not None and np.linalg.norm(c - cen[b]) > MOVED_MM / 1000.0:
                    (t_old_pts if b in t_body else o_old_pts).append(p)
        tm_tree = cKDTree(np.vstack(t_old_pts)) if t_old_pts else None
        om_tree = cKDTree(np.vstack(o_old_pts)) if o_old_pts else None
        vis = [np.zeros(len(s), bool) for s in samples]
        for i, s in enumerate(samples):
            if near[i].any():
                vis[i][near[i]] = _visible(s[near[i]], step)
        with np.load(chunk_file) as z:
            layers = {"main": _grid(z, "esdf"), "hand": _grid(z, "esdf_free")}
            summary = json.loads(str(z["summary_json"]))
        row = {"request": j, "state": state, "has_hand_layer": layers["hand"] is not None}
        for name, grid in layers.items():
            if grid is None:
                continue
            occ = _surface(grid)
            occ = occ[in_roi(occ)]
            r = {"n_occupied": int(len(occ))}
            if len(occ):
                d_gt = gt_tree.query(occ)[0]
                for n, off in planes:
                    d_gt = np.minimum(d_gt, np.abs(occ @ n - off))
                res = (d_gt > STALE_MM / 1000.0) & (rob_tree.query(occ)[0] > STALE_MM / 1000.0)
                t_old = np.zeros(len(occ), bool)
                if tm_tree is not None and res.any():
                    t_old[res] = tm_tree.query(occ[res])[0] <= STALE_MM / 1000.0
                m_old = np.zeros(len(occ), bool)
                rest = res & ~t_old
                if om_tree is not None and rest.any():
                    m_old[rest] = om_tree.query(occ[rest])[0] <= STALE_MM / 1000.0
                r.update(residue=int(res.sum()), target_old=int(t_old.sum()), moved_old=int(m_old.sum()))
            missed = {"visible": [0, 0], "occluded_seen_before": [0, 0]}
            for i, s in enumerate(samples):
                if i in t_idx or not len(s):
                    continue
                k = in_roi(s)
                if not k.any():
                    continue
                val = _lookup(grid, s[k])
                inside = np.isfinite(val)
                far = inside & (val > MISSED_MM / 1000.0)
                vn, sb = vis[i][k], seen[i][k]
                for key, sel in (("visible", vn & inside), ("occluded_seen_before", ~vn & sb & inside)):
                    missed[key][0] += int((far & sel).sum())
                    missed[key][1] += int(sel.sum())
            r["missed"] = missed
            row[name] = r
        st = (summary.get("esdf_stats") or {}) if isinstance(summary, dict) else {}
        for key in ("target_layer", "roi_carve"):
            if key in st:
                row[key] = st[key]
        rows.append(row)
        for i in range(len(samples)):
            seen[i] |= vis[i]
        for b in bodies:   # a new pose is remembered when it differs from every remembered one by > 5 mm
            if cen[b] is not None and all(np.linalg.norm(c - cen[b]) > 0.005 for c, _ in hist[b]):
                hist[b].append((cen[b], cur[b]))
    return rows


def summarise(rows_by_ep: dict) -> dict:
    out = {}
    for layer in ("main", "hand"):
        agg = {"chunks": 0, "n_occupied": 0, "residue": 0, "target_old": 0, "moved_old": 0,
               "missed": {"visible": [0, 0], "occluded_seen_before": [0, 0]}}
        for rows in rows_by_ep.values():
            for r in rows:
                x = r.get(layer)
                if x is None:
                    continue
                agg["chunks"] += 1
                for k in ("n_occupied", "residue", "target_old", "moved_old"):
                    agg[k] += int(x.get(k, 0))
                for k in agg["missed"]:
                    agg["missed"][k][0] += x["missed"][k][0]
                    agg["missed"][k][1] += x["missed"][k][1]
        agg["missed_frac"] = {k: (round(v[0] / v[1], 6) if v[1] else None) for k, v in agg["missed"].items()}
        out[layer] = agg
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", required=True, type=pathlib.Path)
    ap.add_argument("--model-xml", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--configurations", nargs="+", required=True)
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"definitions": {"stale_mm": STALE_MM, "missed_mm": MISSED_MM, "visible_mm": VISIBLE_MM,
                               "roi_pad_m": ROI_PAD, "target": TARGET,
                               "surface_voxel": "|recorded value| <= 0.5 voxel", "moved_mm": MOVED_MM, "layers": {"main": "esdf_distance",
                                                                             "hand": "esdf_free_distance"}},
               "configurations": {}}
    for conf in args.configurations:
        eps = episodes_of(args.kit, conf)
        rows_by_ep = {}
        for ep, (pdir, chunks) in sorted(eps.items()):
            if args.episodes and ep not in args.episodes:
                continue
            obstacle = obstacle_profile(pdir.parent.parent)
            gt = scene_truth(args.model_xml, obstacle)
            rows = run_episode(pdir, chunks, gt, RobotSurface(gt))
            rows.insert(0, {"obstacle_profile": obstacle[1]})
            rows_by_ep[ep] = rows
            (args.out / f"rows_{conf}_ep{ep}.json").write_text(json.dumps(rows))
            print(f"{conf} ep{ep}: {len(rows)} chunks", flush=True)
        summary["configurations"][conf] = {"episodes": sorted(rows_by_ep), **summarise(rows_by_ep)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["configurations"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
