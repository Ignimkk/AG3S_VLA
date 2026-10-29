"""T31-diag D4 — occupied voxels coloured by GT nearest body (the label layer is not recorded).

usage: MUJOCO_GL=osmesa PYTHONPATH=outputs/verify/T31diag/snap .venv-ag3s/bin/python d4_voxels.py RUN T_FIG [--all]
Occupied voxel := recorded grid value d <= voxel/2 (surface or inside), per layer (fine 5 mm: whole window; coarse 20 mm: within the
fine window box + 0.1 m). Voxel centre (base = world here) -> min mj_geomDistance to every collision geom (robot included; geoms
pre-filtered by |centre - p| - rbound < 0.06 m). Class = body of the nearest geom: crate / table / apple / other_fruit / robot /
other; if that distance > 30 mm the class is 'no_gt_surface_30mm'. With --all: the same counts for every window chunk (fine layer).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T31diag/d4'))
import fig_common as FC
from AG3S.ag3s.experiments.common import figstyle as fs
KO = fs.use_korean(); C = fs.CATEGORICAL
CLASSES = ["crate", "table", "apple", "other_fruit", "robot", "other", "no_gt_surface_30mm"]
COL = {k: C[i] for i, k in enumerate(CLASSES)}
FRUIT = {"banana", "orange", "pear"}


def setup_probe(R):
    spec = mujoco.MjSpec.from_file(FC.XML)
    b = spec.worldbody.add_body(name="probe", mocap=True)
    b.add_geom(name="probe_g", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1e-4, 0, 0], contype=0, conaffinity=0)
    m = spec.compile(); d = mujoco.MjData(m)
    pg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_g")
    mid = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "probe")])
    geoms = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and g != pg]
    root = {g: m.body(m.body_rootid[m.geom_bodyid[g]]).name for g in geoms}
    body = {g: m.body(m.geom_bodyid[g]).name for g in geoms}
    return m, d, pg, mid, geoms, root, body


def classify(P, pts):
    m, d, pg, mid, geoms, root, body = P
    gx = d.geom_xpos[geoms]; rb = m.geom_rbound[geoms]
    out = []; dist = []
    ft = np.zeros(6)
    for p in pts:
        near = np.linalg.norm(gx - p, axis=1) - rb < 0.06
        best, bg = np.inf, None
        d.mocap_pos[mid] = p; mujoco.mj_kinematics(m, d)
        for gi in np.flatnonzero(near):
            g = geoms[gi]
            v = float(mujoco.mj_geomDistance(m, d, pg, g, 0.06, ft))
            if abs(abs(v) - np.linalg.norm(ft[:3] - ft[3:])) > 1e-3 and v < 0.06:
                continue
            if v < best:
                best, bg = v, g
        dist.append(best)
        if bg is None or best > 0.03:
            out.append("no_gt_surface_30mm")
        else:
            b = body[bg]
            out.append(b if b in ("crate", "table", "apple") else "other_fruit" if b in FRUIT else
                       "other" if root[bg] in ("office", "shelf", "world") or b in ("office", "shelf", "world") else "robot")
    return out, np.array(dist)


def occupied(layer, box_lo=None, box_hi=None):
    V = np.asarray(layer.distance_grid, np.float32); vs = layer.grid.voxel_size
    idx = np.argwhere(V <= vs / 2)
    pts = np.asarray(layer.grid.origin) + idx * vs
    if box_lo is not None:
        k = np.all((pts >= box_lo) & (pts <= box_hi), axis=1); pts = pts[k]; idx = idx[k]
    return pts, V[tuple(idx.T)]


def main():
    run, tfig = sys.argv[1], int(sys.argv[2]); do_all = "--all" in sys.argv
    R = FC.Run(run)
    P = setup_probe(R)
    res = {"definitions": __doc__, "run": run}
    D1 = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T31diag/d1/d1_oscillation.json')).read_text())["runs"][run]
    ts = [c["t"] for c in R.chunks if D1["grasp_t"] <= c["t"] <= D1["release_t"]] if do_all else [tfig]
    per = []
    for t in ts:
        ch = R.load(t)
        P[1].qpos[:len(ch.row["qpos"])] = ch.row["qpos"]; mujoco.mj_kinematics(P[0], P[1])
        fine, coarse = ch.field.layers[1], ch.field.layers[0]
        fp, fv = occupied(fine)
        fcls, fd = classify(P, fp)
        e = {"t": t, "fine_n_occupied": len(fp), "fine_class_counts": dict(collections.Counter(fcls))}
        if t == tfig:
            g = fine.grid; lo = np.asarray(g.origin) - 0.1; hi = np.asarray(g.origin) + np.asarray(g.shape) * g.voxel_size + 0.1
            cp, cv = occupied(coarse, lo, hi)
            ccls, cd = classify(P, cp)
            e.update({"coarse_n_occupied_in_box": len(cp), "coarse_class_counts": dict(collections.Counter(ccls))})
            fig = plt.figure(figsize=(17, 8))
            for j, (pts, cls, name) in enumerate(((fp, fcls, "fine 5 mm (window)"), (cp, ccls, "coarse 20 mm (window box + 0.1 m)"))):
                ax = fig.add_subplot(1, 2, j + 1, projection="3d")
                cls = np.array(cls)
                for k in CLASSES:
                    s = cls == k
                    if s.any():
                        ax.scatter(pts[s, 0], pts[s, 1], pts[s, 2], s=2 if j == 0 else 14, color=COL[k], label=f"{k} ({int(s.sum())})", depthshade=False)
                cen = ch.traj["refined"]["centres"]
                ax.scatter(cen[:, R.finger_idx, 0].ravel(), cen[:, R.finger_idx, 1].ravel(), cen[:, R.finger_idx, 2].ravel(), s=4, color=fs.INK, label="refined finger spheres (8 steps)")
                ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
                ax.set_title(name, fontsize=10); ax.legend(fontsize=7, loc="upper left", markerscale=3)
                ax.view_init(elev=22, azim=-60)
                try:
                    ax.set_box_aspect(np.ptp(np.vstack([pts, cen[:, R.finger_idx].reshape(-1, 3)]), axis=0))
                except Exception:
                    pass
            fig.suptitle(f"{run} t={t} occupied voxels (d <= voxel/2), colour = GT nearest body (label layer not recorded)" if not KO else
                         f"{run} t={t} 점유 복셀 (d ≤ 복셀/2), 색 = GT 최근접 물체 (라벨 층은 기록에 없음)", fontsize=12)
            fig.patch.set_facecolor(fs.SURFACE)
            out = FC.FIG / f"t31-d4-voxels3d-{run}-t{t:03d}.png"
            fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
            e["figure"] = str(out)
            (out.with_suffix(".json")).write_text(json.dumps({"definitions": __doc__, **e}, indent=1, default=float))
        per.append(e)
        print(t, e["fine_n_occupied"], e["fine_class_counts"], e.get("coarse_class_counts", ""), flush=True)
    res["chunks"] = per
    pathlib.Path(f"{_ag3s_workspace_path()}/outputs/verify/T31diag/d4/voxels_{run}{'_all' if do_all else ''}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
