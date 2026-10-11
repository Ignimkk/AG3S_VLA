"""T44b D-merge: image-space attention on target vs the nearest other fruit, GT Gaussian vs real VLA (16x16 per camera).
Per camera that sees both objects (>= 20 px each): mean attention per pixel on the target / on the neighbour (the 16x16 map
nearest-upsampled to 640x480), and whether the argmax patch lies on the target.
usage: att_2d.py --out F.json"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse, json, pathlib
import numpy as np

M = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/merge'))
CAMS = {"cam_high": "zed_left", "cam_left_wrist": "wrist_cam_l", "cam_right_wrist": "wrist_cam_r"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    V = np.load(M / "vla_attention.npz", allow_pickle=True)
    vkeys = {str(k): i for i, k in enumerate(V["keys"])}
    vcams = [str(c) for c in V["cameras"]]
    E2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2/raw_demo'))
    rows = []
    for fj in sorted((M / "raw_gt").glob("demo_*.json")):
        rec = json.loads(fj.read_text()); z = np.load(fj.with_suffix(".npz"))
        names = {int(k): v for k, v in json.loads(str(z["body_names"])).items()}
        inv = {v: k for k, v in names.items()}
        t = rec["target"]; o = min(rec["mesh_gap_mm"], key=rec["mesh_gap_mm"].get)
        e2 = json.loads((E2 / f"{rec['key']}.json").read_text())["identities"]["gt"]["merge"]["target_cluster"]
        row = {"key": rec["key"], "target": t, "neighbour": o, "e2_merged": bool(e2["target_fraction"] < .8 and e2["distractor_fraction"] > 0),
               "cams": {}}
        for pc, cam in CAMS.items():
            ids = z[f"bodyids_{cam}"]
            tm, om = ids == inv[t], ids == inv[o]
            r = {"n_px_target": int(tm.sum()), "n_px_neigh": int(om.sum())}
            for src in ("gt", "vla"):
                if src == "gt":
                    if f"attmap_{cam}" not in z.files:
                        r[src] = None; continue
                    g = z[f"attmap_{cam}"]
                else:
                    g = V["attention"][vkeys[rec["key"]], vcams.index(pc)]
                up = np.kron(g, np.ones((30, 40)))
                d = {"argmax_on_target": bool(tm[np.unravel_index(np.argmax(up), up.shape)]) if tm.any() else None,
                     "frac_mass_on_target_px": round(float(up[tm].sum() / up.sum()), 4),
                     "frac_mass_on_neigh_px": round(float(up[om].sum() / up.sum()), 4)}
                if tm.sum() >= 20 and om.sum() >= 20:
                    d["per_px_ratio_t_over_n"] = round(float(up[tm].mean() / max(up[om].mean(), 1e-12)), 3)
                r[src] = d
            row["cams"][cam] = r
        rows.append(row)
    summ = {}
    for src in ("gt", "vla"):
        for cam in CAMS.values():
            rat = [r["cams"][cam][src]["per_px_ratio_t_over_n"] for r in rows
                   if r["cams"][cam].get(src) and "per_px_ratio_t_over_n" in r["cams"][cam][src]]
            arg = [r["cams"][cam][src]["argmax_on_target"] for r in rows if r["cams"][cam].get(src)
                   and r["cams"][cam][src]["argmax_on_target"] is not None and r["cams"][cam]["n_px_target"] >= 20]
            ft = [r["cams"][cam][src]["frac_mass_on_target_px"] for r in rows if r["cams"][cam].get(src) and r["cams"][cam]["n_px_target"] >= 20]
            summ[f"{src}|{cam}"] = {"n_both_visible": len(rat), "per_px_ratio_median": None if not rat else round(float(np.median(rat)), 3),
                                    "n_ratio_ge_2": sum(x >= 2 for x in rat), "n_ratio_gt_1": sum(x > 1 for x in rat),
                                    "n_target_visible": len(arg), "n_argmax_on_target": sum(arg),
                                    "frac_mass_on_target_median": None if not ft else round(float(np.median(ft)), 4)}
    pathlib.Path(a.out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
