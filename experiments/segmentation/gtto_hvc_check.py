#!/usr/bin/env python3
"""G61 HVC hypothesis check: does F8's hand-voxel carve (SL16 HVC, current, 20 mm) erase obstacle voxels near the hand?

Same SafeLIBERO episode recorded with F7 and F8 (`--record-constraints full`); at the call where the two first act differently
(t2e4 seq 3: F7 intervened, F8 read 'clear') the observations must be identical (checked: the capture depth images and the
executed actions before the call). The obstacle's GT surface comes from the recorded segmentation (geom ids -> body name) and
depth (both cameras, OpenCV rows, K pixel-centre, `T_base_cam`), so no MuJoCo model is needed.
Rule (fixed before results, G61): F8 has >= 30 % fewer occupied fine voxels within 3 cm of the obstacle's GT surface than F7.
usage: gtto_hvc_check.py --f7 RUN --f8 RUN [--stem safelibero_spatial_t2_e4] [--seq 3] [--obj red_coffee_mug_obstacle_1] --out DIR
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def gt_points(geom, oi, obj):
    """Base-frame points of every pixel (both cameras) whose geom belongs to a body named `obj*`, at capture frame `oi`."""
    names = [str(b) for b in geom["model_body_names"]]
    gb = np.asarray(geom["model_geom_bodyid"])
    bodies = {i for i, n in enumerate(names) if n.startswith(obj)}
    geoms = np.flatnonzero(np.isin(gb, list(bodies)))
    idx = list(np.asarray(geom["obs_index"]).astype(int)).index(int(oi))
    pts = []
    for cam in [str(c) for c in geom["cameras"]]:
        seg = np.asarray(geom[f"segmentation_geom/{cam}"][idx])
        d = np.asarray(geom[f"depth_mm/{cam}"][idx], np.float64) / float(geom["depth_scale_mm"])
        K = np.asarray(geom[f"K/{cam}"][idx], np.float64)
        T = np.asarray(geom[f"T_base_cam/{cam}"][idx], np.float64)
        m = np.isin(seg, geoms) & (d > 0)
        v, u = np.nonzero(m)
        z = d[v, u]
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        p = np.stack([x, y, z, np.ones_like(z)], 1) @ T.T
        pts.append(p[:, :3])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def occupied(z, prefix="esdf"):
    d = np.asarray(z[f"{prefix}_distance"], np.float32)
    o, vs = np.asarray(z[f"{prefix}_origin"], np.float64), float(z[f"{prefix}_voxel_size"])
    ijk = np.argwhere(d < 0.5 * vs)
    return o + ijk * vs, vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f7", required=True, type=pathlib.Path)
    ap.add_argument("--f8", required=True, type=pathlib.Path)
    ap.add_argument("--f7-scored", type=pathlib.Path, default=None)
    ap.add_argument("--f8-scored", type=pathlib.Path, default=None)
    ap.add_argument("--stem", default="safelibero_spatial_t2_e4")
    ap.add_argument("--seq", type=int, default=3)
    ap.add_argument("--obj", default="red_coffee_mug_obstacle_1")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from scipy.spatial import cKDTree
    from AG3S.trajopt.experiments import panda_fixed_input_ablation as fa

    res = {"stem": a.stem, "seq": a.seq, "obj": a.obj}
    # J0: the recorded runs act exactly as the scored ones
    for k, rec, sc in (("F7", a.f7, a.f7_scored), ("F8", a.f8, a.f8_scored)):
        if sc is not None:
            x, y = np.load(rec / "episodes" / f"{a.stem}_calls.npz"), np.load(sc / "episodes" / f"{a.stem}_calls.npz")
            res[f"J0_{k}"] = all(np.array_equal(x[f], y[f]) for f in x.files if f in y.files)
    per = {}
    for k, run in (("F7", a.f7), ("F8", a.f8)):
        calls = fa.load_episode_calls(run / "episodes", a.stem)
        fa.attach_chunks(calls, run / "server_constraints" / "run_0000")
        c = next(c for c in calls if c.seq == a.seq)
        geom = np.load(run / "episodes" / f"{a.stem}_geom.npz", allow_pickle=True)
        z = np.load(c.chunk_path, allow_pickle=False)
        summ = json.loads(str(z["summary_json"]))
        per[k] = {"call": c, "geom": geom, "z": z, "summary": summ}
    c7, c8 = per["F7"]["call"], per["F8"]["call"]
    oi = c7.obs_index
    same_obs = all(np.array_equal(np.asarray(per["F7"]["geom"][f"depth_mm/{cam}"][list(per["F7"]["geom"]["obs_index"]).index(oi)]),
                                  np.asarray(per["F8"]["geom"][f"depth_mm/{cam}"][list(per["F8"]["geom"]["obs_index"]).index(oi)]))
                   for cam in [str(x) for x in per["F7"]["geom"]["cameras"]])
    res["same_observation_depth"] = bool(same_obs) and c7.obs_index == c8.obs_index
    res["same_executed_before"] = all(np.array_equal(p.actions_executed, q.actions_executed)
                                      for p, q in zip(sorted(c7.episode.values(), key=lambda c: c.seq),
                                                      sorted(c8.episode.values(), key=lambda c: c.seq)) if p.seq < a.seq)
    pts = gt_points(per["F7"]["geom"], oi, a.obj)
    res["n_gt_points"] = int(len(pts))
    tree = cKDTree(pts) if len(pts) else None
    for k in ("F7", "F8"):
        z = per[k]["z"]
        vox, vs = occupied(z)
        dd = tree.query(vox, distance_upper_bound=0.03)[0] if tree is not None else np.full(len(vox), np.inf)
        near = np.isfinite(dd)
        cent = np.asarray(z["sphere_centres"], np.float64)
        names = per[k]["summary"].get("sphere_link_names") or []
        hand = np.array([n in ("panda_hand", "panda_leftfinger", "panda_rightfinger") for n in names[:len(cent)]])
        dh = np.min(np.linalg.norm(vox[:, None, :] - cent[hand][None], axis=2), axis=1) if hand.any() and len(vox) else np.full(len(vox), np.inf)
        res[k] = {"obstacle_voxels_3cm": int(near.sum()), "obstacle_voxels_3cm_within_10cm_of_hand": int((near & (dh < 0.10)).sum()),
                  "occupied_within_10cm_of_hand": int((dh < 0.10).sum()),
                  "hand_voxel_carve": per[k]["summary"].get("hand_voxel_carve"),
                  "clearance_min_mm": per[k]["summary"].get("clearance_min_mm"),
                  "passthrough": ((c7 if k == "F7" else c8).row.get("to") or {}).get("passthrough_clear")}
        per[k]["vox"], per[k]["near"], per[k]["hand_c"] = vox, near, cent[hand]
    n7, n8 = res["F7"]["obstacle_voxels_3cm"], res["F8"]["obstacle_voxels_3cm"]
    res["drop"] = None if n7 == 0 else 1.0 - n8 / n7
    res["verdict"] = ("HVC erases obstacle voxels" if res["drop"] is not None and res["drop"] >= 0.30 else "hypothesis rejected")
    (a.out / "hvc_check.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    L = [f"# HVC 가설 검증 (G61) — {a.stem} seq {a.seq}, 장애물 {a.obj}", "",
         f"- J0 (기록 run = 채점 run): F7 {res.get('J0_F7')} · F8 {res.get('J0_F8')}",
         f"- 같은 관측 (depth) {res['same_observation_depth']} · 앞 call 실행 action 동일 {res['same_executed_before']} · GT 표면 점 {res['n_gt_points']}", "",
         "| | F7 | F8 |", "|---|---|---|"]
    for key in ("obstacle_voxels_3cm", "obstacle_voxels_3cm_within_10cm_of_hand", "occupied_within_10cm_of_hand", "clearance_min_mm"):
        L.append(f"| {key} | {res['F7'][key]} | {res['F8'][key]} |")
    L += ["", f"감소율 {res['drop']} → **{res['verdict']}** (규칙: ≥ 0.30 이면 HVC 가 지운다)"]
    (a.out / "HVC검증표.md").write_text("\n".join(L) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 5.5))
    for i, k in enumerate(("F7", "F8")):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        v, nr, hc = per[k]["vox"], per[k]["near"], per[k]["hand_c"]
        sel = np.linalg.norm(v - pts.mean(0), axis=1) < 0.2 if len(pts) else np.ones(len(v), bool)
        ax.scatter(*v[sel & ~nr].T, s=1, c="#bbbbbb", label="occupied (other)")
        ax.scatter(*v[nr].T, s=4, c="#e6550d", label=f"within 3 cm of {a.obj} ({int(nr.sum())})")
        if len(pts):
            ax.scatter(*pts[::5].T, s=1, c="#3182bd", alpha=0.3, label="GT obstacle surface")
        ax.scatter(*hc.T, s=30, marker="x", c="k", label="hand spheres")
        ax.set_title(f"{k}: field at {a.stem} seq {a.seq}", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.out / "그래프_HVC검증.png", dpi=120)
    print("\n".join(L))


if __name__ == "__main__":
    main()
