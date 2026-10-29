"""T34 verifier — thickness of the table's zero-distance band in the recorded FINE ESDF layer (esdf_distance, 5 mm).

Per chunk npz of a run (every STRIDE-th chunk):
  plane    : TLS plane of support_mask points with base z > 0.5 m (normal up).
  columns  : xy grid, 20 mm spacing, over the fine layer footprint; kept when
             (a) >= 3 support points within 10 mm (xy) of the column (table under it),
             (b) the column is > 90 mm (xy) from every GT fruit centre (apple/banana/orange/pear, planning-row
                 object_poses -> base frame) and outside the GT crate footprint + 60 mm (crate half-extents from the
                 recorded destination extents are not used; a 0.25 m radius around the crate centre is excluded),
             (c) eikonal above the band: d(h) for h = +20..+45 mm (plane normal) fits d = s h + c with s in [0.9, 1.1],
                 max residual < 1.0 mm (table is the nearest surface).
  sets     : 'table' = columns passing (a)-(c); 'ring' = the subset 60-120 mm (xy) from the GT apple centre (fruit
             exclusion (b) not applied to the apple for this set, other fruits still > 90 mm).
  layers   : 'fine' = recorded esdf_distance (5 mm) alone; 'coarse' = esdf_coarse_distance (20 mm) alone;
             'combined' = fixed_input_ablation.build_field (min over layers where covered = what the TO reads).
             (c) is evaluated per layer.
  profile  : trilinear d(h) at h = -60..+45 mm, step 0.5 mm, along the plane normal.
  band     : the contiguous run of h with d(h) <= 0.1 mm that contains the h closest to 0 among such h;
             above_mm = its top h, below_mm = -(its bottom h); bias_mm = 20 - d(+20 mm) (= h - d above the band).
Per run: pooled columns over the sampled chunks -> median / p90 of above_mm, below_mm, total (above+below), bias_mm.
usage: band.py REC_DIR PORT RUNS_JSON(analysis json with runs->chunks->server_chunk, or 'scan') OUT.json [STRIDE]
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P
from AG3S.ag3s.fields.curobo_field import layer_from_arrays
import AG3S.trajopt.experiments.fixed_input_ablation as T

REC, PORT, ANA, OUTP = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
STRIDE = int(sys.argv[5]) if len(sys.argv) > 5 else 1
m, d, probe, obj, _ = P.build()
H = np.round(np.arange(-0.060, 0.0451, 0.0005), 6)
FRUIT = ("apple", "banana", "orange", "pear")


def world_to_base(pw):
    import mujoco
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    R = d.xmat[b].reshape(3, 3); t = d.xpos[b]
    return R.T @ (np.asarray(pw, float) - t)


def profile_band(field, base, n):
    Pq = base[:, None, :] + H[None, :, None] * n[None, None, :]
    D = np.asarray(field.distance(Pq.reshape(-1, 3)), float).reshape(len(base), len(H))
    sel = (H >= 0.020) & (H <= 0.045)
    A = np.column_stack([H[sel], np.ones(sel.sum())])
    coef, *_ = np.linalg.lstsq(A, D[:, sel].T, rcond=None)
    res = np.abs(A @ coef - D[:, sel].T).max(0)
    ok = (coef[0] > 0.9) & (coef[0] < 1.1) & (res < 0.001)
    out = []
    i20 = int(np.argmin(np.abs(H - 0.020)))
    for k in np.where(ok)[0]:
        z0 = D[k] <= 1e-4
        if not z0.any():
            out.append((0.0, 0.0, 1000 * (0.020 - D[k, i20]), False)); continue
        idx = np.where(z0)[0]; c = idx[np.argmin(np.abs(H[idx]))]
        a = c
        while a + 1 < len(H) and z0[a + 1]:
            a += 1
        b_ = c
        while b_ - 1 >= 0 and z0[b_ - 1]:
            b_ -= 1
        out.append((1000 * H[a], -1000 * H[b_], 1000 * (0.020 - D[k, i20]), b_ == 0))
    return out


def chunk_band(f, row):
    z = np.load(f, allow_pickle=False)
    pts = np.asarray(z["raw_cloud"], float)[np.asarray(z["support_mask"], bool)]
    tab = pts[pts[:, 2] > 0.5]
    if len(tab) < 100:
        return None
    c0 = tab.mean(0); n = np.linalg.svd(tab - c0, full_matrices=False)[2][-1]; n = n if n[2] > 0 else -n; off = float(n @ c0)
    layers = {"fine": layer_from_arrays(z["esdf_distance"], z["esdf_origin"], float(z["esdf_voxel_size"])),
              "coarse": layer_from_arrays(z["esdf_coarse_distance"], z["esdf_coarse_origin"], float(z["esdf_coarse_voxel_size"])),
              "combined": T.build_field(z, outside_distance=1.0)}
    o = np.asarray(z["esdf_origin"], float); vs = float(z["esdf_voxel_size"]); shp = np.asarray(z["esdf_distance"].shape)
    lo, hi = o[:2] + 0.03, o[:2] + (shp[:2] - 1) * vs - 0.03
    xs = np.arange(lo[0], hi[0], 0.02); ys = np.arange(lo[1], hi[1], 0.02)
    X, Y = np.meshgrid(xs, ys, indexing="ij"); cols = np.column_stack([X.ravel(), Y.ravel()])
    from scipy.spatial import cKDTree
    tr = cKDTree(tab[:, :2])
    cnt = np.array([len(v) for v in tr.query_ball_point(cols, 0.010)])
    base_keep = cnt >= 3
    P.set_q(m, d, row["qpos"])
    op = row["object_poses"]
    others = np.ones(len(cols), bool)
    for b in FRUIT:
        if b in op and b != "apple":
            pb = world_to_base(op[b]["pos"])
            others &= np.linalg.norm(cols - pb[:2], axis=1) > 0.090
    if "crate" in op:
        pb = world_to_base(op["crate"]["pos"])
        others &= np.linalg.norm(cols - pb[:2], axis=1) > 0.25
    pa = world_to_base(op["apple"]["pos"])
    ra = np.linalg.norm(cols - pa[:2], axis=1)
    sets = {"table": base_keep & others & (ra > 0.090), "ring": base_keep & others & (ra >= 0.060) & (ra <= 0.120)}
    out = {"plane_offset_m": off, "n_cols": {k: int(v.sum()) for k, v in sets.items()}, "vals": {}}
    for sk, sm in sets.items():
        cs = cols[sm]
        if not len(cs):
            continue
        zc = (off - n[0] * cs[:, 0] - n[1] * cs[:, 1]) / n[2]
        base = np.column_stack([cs, zc])
        for lk, fld in layers.items():
            out["vals"][f"{sk}/{lk}"] = profile_band(fld, base, n)
    return out


def main():
    res = {"doc": __doc__, "runs": {}}
    A = json.loads(pathlib.Path(ANA).read_text())["runs"]
    q = lambda x: {"median": round(float(np.median(x)), 2), "p90": round(float(np.percentile(x, 90)), 2),
                   "p10": round(float(np.percentile(x, 10)), 2)}
    for run in sorted(A):
        rd = REC / run
        rows = {int(x["seq"]): x for x in (json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip())
                if x.get("kind") == "planning"}
        ch = [c for c in A[run]["chunks"] if c.get("server_chunk")][::STRIDE]
        pooled = {}; per = []
        for c in ch:
            r = chunk_band(REC / c["server_chunk"], rows[int(c["seq"])])
            if not r:
                continue
            pc = {"t": c["t"], "n_cols": r["n_cols"]}
            for k, vals in r["vals"].items():
                if not vals:
                    continue
                v = np.array([x[:3] for x in vals]); pooled.setdefault(k, []).append(v)
                pc[k] = {"n": len(v), "above_med": float(np.median(v[:, 0])), "below_med": float(np.median(v[:, 1])),
                         "bias20_med": float(np.median(v[:, 2])), "n_below_at_limit": int(sum(x[3] for x in vals))}
            per.append(pc)
        rr = {"n_chunks": len(per), "per_chunk": per}
        for k, lst in pooled.items():
            V_ = np.vstack(lst)
            rr[k] = {"n_columns": int(len(V_)), "above_mm": q(V_[:, 0]), "below_mm": q(V_[:, 1]),
                     "total_mm": q(V_[:, 0] + V_[:, 1]), "bias_at_20mm_mm": q(V_[:, 2])}
        res["runs"][run] = rr
        print(run, json.dumps({k: rr[k] for k in rr if "/" in k}), flush=True)
        pathlib.Path(OUTP).write_text(json.dumps(res, indent=1, default=float))
    pathlib.Path(OUTP).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
