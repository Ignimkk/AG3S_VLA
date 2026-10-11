"""T44dA verifier (CPU only): per chunk, re-query the RECORDED field (float16 grids, field.py) at the authorized rows
(target_permission.target_field_exclude == True; = active... both hands' palm+fingers under exclude_authorized? -> mask as recorded)
along reference_chunk and refined_chunk rows 0..15, coarse ON (served formula) vs OFF (same fine layers, no c term = C1a formula).
Per chunk -> CLA/field/<NAME>.jsonl:
  for traj in (ref, fin): min clearance (d - r, no margin) ON / OFF over authorized ACTIVE-hand rows (rows 0..7 = certified window, 0..15 = plan),
  n rows with ON < OFF - 1 mm, n 'cross0' (OFF >= 0 > ON), n 'cross10' (OFF >= 10 mm > ON), tier counts ON,
  location of the rows with ON < OFF - 1 mm: height of sphere bottom above the support plane, inside ball (|c - ball centre| <= ball r),
  GT-hull signed gap (mm) to target / nearest other fruit (fruit poses of the planning row).
Applies to arms whose chunk npz has esdf_free_coarse (C1b, C2). usage: field_scan.py NAME ... (run from ${AG3S_PROJECT_ROOT}-t44cl)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import os, sys, json, pathlib
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/CLA/p'))
import numpy as np
import fkmodel as F, field as FD
from extract_chunks import hull_sd, FRUITS

CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL')); OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CLA/field'))

def run(name):
    arm = name.split("_ep")[0]
    a = json.load(open(CL / "analysis" / f"{name}.json"))
    rows = [json.loads(l) for l in open((pathlib.Path(_ag3s_workspace_path('')) / a["run_dir"]) / "frames.jsonl")]
    P = {r["seq"]: r for r in rows if r["kind"] == "planning"}
    side = a["used_arm"]; s1 = "l" if side == "left" else "r"
    links_model = "arms" if arm == "C2" else "gripper"
    sc, qm = F.model(links_model)
    links = list(qm.sphere_link_names); rad = np.asarray(qm.radii)
    act = np.array([l in (f"ee_finger_{s1}1", f"ee_finger_{s1}2", f"ee_{side}") for l in links])
    tgt = a["target_fruit"]; others = [f for f in FRUITS if f != tgt]
    OUT.mkdir(parents=True, exist_ok=True); fo = open(OUT / f"{name}.jsonl", "w")
    for c in a.get("server_chunks") or []:
        z = np.load(c["server_file"], allow_pickle=True)
        if "esdf_free_coarse_distance" not in z.files or "esdf_free_distance" not in z.files:
            continue
        s = json.loads(str(z["summary_json"])); p = P.get(c["seq"])
        if p is None:
            continue
        mask = np.asarray((s.get("target_permission") or {}).get("target_field_exclude") or [], bool)
        if mask.size != len(links):
            fo.write(json.dumps({"seq": c["seq"], "error": f"mask {mask.size} != spheres {len(links)}"}) + "\n"); continue
        sel = mask & act
        tf = s.get("esdf_stats", {}).get("target_free"); tf = eval(tf) if isinstance(tf, str) else (tf or {})
        ball = (tf or {}).get("ball") or {}
        bc = np.asarray(ball.get("centre_m") or [np.nan] * 3); br = (ball.get("radius_mm") or np.nan) / 1e3; sz = ball.get("support_plane_z_m")
        qpos = p["qpos"]; pr = F.params_from_qpos(sc, qm, qpos); op = p["object_poses"]
        rec = {"name": name, "seq": c["seq"], "t": c.get("t"), "hold": bool(c.get("hold")), "n_auth_active": int(sel.sum()), "n_mask": int(mask.sum()),
               "mask_links": sorted(set(np.asarray(links)[mask].tolist())), "support_z": sz}
        for key, ch in (("ref", z["reference_chunk"]), ("fin", z["refined_chunk"])):
            Q = F.chunk_q(sc, qm, qpos, np.asarray(ch, np.float64)[:16])
            S = np.stack([F.spheres(qm, q, pr) for q in Q])[:, sel]          # (16, k, 3)
            R = np.tile(rad[sel], (16, 1))
            X = S.reshape(-1, 3); r = R.reshape(-1)
            res = FD.query(z, X, True)
            con = (res["d_on"] - r).reshape(16, -1) * 1e3; coff = (res["d_off"] - r).reshape(16, -1) * 1e3
            ton = res["tier_on"].reshape(16, -1)
            low = con < coff - 1.0
            o = {"min_on_0_7": float(con[:8].min()), "min_off_0_7": float(coff[:8].min()), "min_on_0_15": float(con.min()), "min_off_0_15": float(coff.min()),
                 "n_lower": int(low.sum()), "n_lower_0_7": int(low[:8].sum()),
                 "cross0": int(((coff >= 0) & (con < 0)).sum()), "cross10": int(((coff >= 10) & (con < 10)).sum()),
                 "cross0_0_7": int(((coff[:8] >= 0) & (con[:8] < 0)).sum()), "cross10_0_7": int(((coff[:8] >= 10) & (con[:8] < 10)).sum()),
                 "tier_on": {t: int((ton == t).sum()) for t in np.unique(ton)},
                 "tier_on_lt10": {t: int(((ton == t) & (con < 10)).sum()) for t in np.unique(ton)}}
            if low.any():
                L = X[low.reshape(-1)]; rl = r[low.reshape(-1)]
                hb = (L[:, 2] - rl - sz) * 1e3 if sz is not None else np.full(len(L), np.nan)
                inball = np.linalg.norm(L - bc, axis=1) <= br
                gT = (hull_sd(op[tgt], tgt, L) - rl) * 1e3
                gO = np.min([(hull_sd(op[b], b, L) - rl) for b in others], axis=0) * 1e3
                dd = (coff - con)[low]
                o["lower_loc"] = {"n": int(len(L)), "drop_mm_median": float(np.median(dd)), "drop_mm_max": float(dd.max()),
                                  "h_bottom_above_support_mm_median": float(np.nanmedian(hb)), "frac_h_lt_20mm": float(np.mean(hb < 20)),
                                  "frac_inball": float(inball.mean()), "gap_target_mm_median": float(np.median(gT)), "frac_gap_target_lt_15": float(np.mean(gT < 15)),
                                  "gap_other_mm_median": float(np.median(gO)), "frac_gap_other_lt_15": float(np.mean(gO < 15))}
            # location of the single worst ON row (rows 0..7)
            k = np.unravel_index(np.argmin(con[:8]), con[:8].shape); w = S[:8][k]; rw = R[:8][k]
            o["worst_on"] = {"row": int(k[0]), "clr_on": float(con[:8][k]), "clr_off": float(coff[:8][k]), "tier_on": str(ton[:8][k]),
                             "h_bottom_mm": float((w[2] - rw - sz) * 1e3) if sz is not None else None, "inball": bool(np.linalg.norm(w - bc) <= br),
                             "gap_target_mm": float((hull_sd(op[tgt], tgt, w[None]) - rw)[0] * 1e3),
                             "gap_other_mm": float(min((hull_sd(op[b], b, w[None]) - rw)[0] for b in others) * 1e3)}
            rec[key] = o
        fo.write(json.dumps(rec) + "\n")
    fo.close()

if __name__ == "__main__":
    for n in sys.argv[1:]:
        if (OUT / f"{n}.jsonl").exists():
            continue
        run(n); print("done", n, flush=True)
