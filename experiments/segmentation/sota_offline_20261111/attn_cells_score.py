#!/usr/bin/env python3
"""(2026-11-13 sdm-f6) CPU: score every [layer, head] cell re-read by attn_cells_replay.py with the attn_vs_target.py rule.

Per pre-grasp call and cell: the cell's head-camera grid (tokens 0..255) is resampled to the 256 x 256 OpenCV map by the
server's own `attention_pixel_map` (a linear operator, built once from the 256 basis grids), read at the head-camera
projection of the call's recorded attention_cloud points (the SafeLIBERO cloud is head-camera only), normalised by
`normalize_attention` (default percentile 5-99, as served), and A = body with the highest mean value over non-support
points within 6 cm of its GT position (>= 15 points). R (the server's recognised target) is the recorded one.
NOTE (found while validating, 2026-11-13 13:5x): the served cloud is the per-voxel MAX over the head and wrist cameras of the
RAW values, and the wrist raw is ~200x the head raw (median; wrist > head in 91 % of 939 sampled calls), so the served A is
wrist-dominated. The cells here are scored HEAD-ONLY (agent view, as KNOWS uses L12H3): `tops` = head-only A per cell;
`a_recorded` = the served (fused) A, kept for comparison, not as a validation target.

usage: .venv-ag3s/bin/python attn_cells_score.py REPLAY_DIR OUT.json"""
import glob, json, os, sys, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_vs_target as avt  # noqa: E402
from AG3S.adapters.panda import attention as pa  # noqa: E402
from AG3S.ag3s.config import AttentionConfig  # noqa: E402
from AG3S.ag3s.stages.attention_lifting import normalize_attention  # noqa: E402

B = avt.B
L, H = 18, 8
HEAD = next(b for b in pa.PI05_LIBERO_TOKEN_BLOCKS if b.wire_camera == "head").tokens


def operator():
    """(65536, 256): flattened OpenCV pixel map = OP @ policy-orientation grid (flattened)."""
    cols = []
    for j in range(256):
        g = np.zeros(256, np.float32); g[j] = 1.0
        cols.append(pa.attention_pixel_map(g.reshape(16, 16), (256, 256)).reshape(-1))
    return np.stack(cols, 1)


def body_masks(P, sup, gt):
    out = {}
    for n, q in gt.items():
        m = (np.linalg.norm(P - q, axis=1) < 0.06) & ~sup
        if m.sum() >= 15:
            out[n] = m
    return out


def top_body(vals, masks):
    means = {n: float(vals[m].mean()) for n, m in masks.items()}
    if not means or max(means.values()) <= 0.0:
        return None
    return max(means, key=means.get)


def main():
    rep, outp = sys.argv[1], sys.argv[2]
    OP = operator(); cfg = AttentionConfig(); G = avt.goals(); res = []
    runs = {os.path.basename(D)[:15] + "_" + os.path.basename(D).split("_", 2)[2][:40]: D
            for pat in avt_patterns() for D in glob.glob(B + pat)}
    for sub in sorted(os.listdir(rep)):
        D = runs.get(sub)
        if D is None:
            continue
        m = __import__("re").search(r"_ms-[^-]+-(\w+)-(LI+)-i", D)
        suite, lvl = (m.group(1), m.group(2)) if m else ("spatial", "LI")
        E = {(int(e["task_id"]), int(e["init_state_index"])): e for e in map(json.loads, open(D + "/episodes.jsonl")) if e}
        for f in sorted(glob.glob(os.path.join(rep, sub, "*.npz"))):
            r = np.load(f)
            if len(r["seq"]) == 0:
                continue
            stem = os.path.basename(f)[:-4]; t = int(stem.split("_t")[1].split("_")[0]); i = int(stem.split("_e")[-1])
            goal = G.get((suite, t)); e = E[(t, i)]; s = e.get("safelibero") or {}
            lab = ("SS" if e["official_success"] and not s.get("collided") else "S+C" if e["official_success"]
                   else "F+C" if s.get("collided") else "F")
            cf = glob.glob(D + f"/episodes/*_t{t}_e{i}_calls.jsonl")[0]; cs = [json.loads(x) for x in open(cf)]
            g = np.load(cf.replace("_calls.jsonl", "_geom.npz"), allow_pickle=True); O = avt.objs(g)
            oi = [int(x) for x in g["obs_index"]]; T = np.asarray(g["T_world_base"], np.float64).reshape(-1, 4, 4)
            calls = []
            for c, (sq, ch) in enumerate(zip(r["seq"], r["chunk"])):
                z = np.load(os.path.join(D, "server_constraints/run_0000", str(ch)), allow_pickle=True)
                k = oi.index(cs[int(sq)]["obs_index"]); Tinv = np.linalg.inv(T[min(k, len(T) - 1)])
                P = np.asarray(z["attention_cloud"], np.float64); sup = np.asarray(z["support_mask"], bool)
                gt = {n: (Tinv @ np.r_[p[k], 1.0])[:3] for n, p in O.items()
                      if p[k, 2] < 2 and np.linalg.norm((Tinv @ np.r_[p[k], 1.0])[:3]) < 1.0}
                tp = z["target_points"] if "target_points" in z.files else np.zeros((0, 3))
                rec = min(gt, key=lambda n: np.linalg.norm(gt[n] - np.asarray(tp).mean(0))) if len(tp) and gt else None
                Tc = np.linalg.inv(g["T_base_cam/head"][k]); K = g["K/head"][k]
                q = (Tc[:3, :3] @ P.T + Tc[:3, 3:]).T
                u = np.rint(K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2]).astype(int)
                v = np.rint(K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]).astype(int)
                ok = (q[:, 2] > 0) & (u >= 0) & (u < 256) & (v >= 0) & (v < 256)
                Pk, supk = P[ok], sup[ok]
                grids = r["pooled_f16"][c][:, :, HEAD[0]:HEAD[1]].astype(np.float32).reshape(L * H, 256)
                raw = OP[v[ok] * 256 + u[ok]] @ grids.T                     # (npts, 144)
                mk = body_masks(Pk, supk, gt)
                tops = [top_body(normalize_attention(raw[:, cell], cfg), mk) for cell in range(L * H)]
                a_rec = top_body(np.asarray(z["attention_values"], np.float64), body_masks(P, sup, gt))
                calls.append({"seq": int(sq), "rec": rec, "a_recorded": a_rec, "tops": tops})
            res.append({"suite": suite, "level": lvl, "task": t, "init": i, "label": lab, "goal": goal, "calls": calls})
            print(sub[:50], stem, len(calls), flush=True)
    json.dump(res, open(outp, "w"))
    # validation + summary for the served cell and the KNOWS cell
    agree = sum(c["a_recorded"] == c["tops"][8 * H + 2] for e in res for c in e["calls"])
    n = sum(len(e["calls"]) for e in res)
    print(json.dumps({"calls": n, "l8h2_reproduced_vs_recorded_top_agree": agree}), flush=True)


def avt_patterns():
    return ("2*_ms-pkgFrssdm-*", "2*_ms-tab10sdm-*-i19-*", "2*_sub-pkgFrssdm-e1a-spatial-LI-t*")


if __name__ == "__main__":
    main()
