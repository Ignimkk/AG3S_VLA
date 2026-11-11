#!/usr/bin/env python3
"""(2026-11-11 dev-57, offline CPU, no GT) Replay SL-PE8's HeldMotionTracker on a run's server records with other parameters.
Hand frame = rigid fit (Kabsch) of the panda_hand constraint spheres to their first-request positions (any frame fixed to the
hand gives the same masks); tip = grasp.tip_alignment.now.tip_m of the call record; pre-held geometry = target_points.
Per held call: how many of the non-support cloud points within 60 mm of the worst obstacle-object row point are removed.
usage: held_motion_replay.py <run tag> <task> <init> [reach_pad_mm reach_cap_mm drift_mm match_mm]"""
import ast, glob, json, sys
import numpy as np
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.stages.held_motion import HeldMotionTracker

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    return x or {}


def kabsch(A, Bp):
    """T (4x4) with Bp ~ R A + t."""
    ca, cb = A.mean(0), Bp.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (Bp - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cb - R @ ca
    return T


def episode(tag, t, init):
    d = sorted(glob.glob(B + f"2*{tag}-e1a-spatial-LI-t{t}-*"))[-1]
    eps = [json.loads(l) for l in open(d + "/episodes.jsonl")]
    per, cur, last = [], [], None
    for c in sorted(glob.glob(d + "/server_constraints/run_*/chunk_*.npz")):
        ts = int(np.load(c)["t_step"])
        if last is not None and ts <= last:
            per.append(cur)
            cur = []
        cur.append((ts, c))
        last = ts
    per.append(cur)
    k = [i for i, e in enumerate(eps) if e["init_state_index"] == init][0]
    e = eps[k]
    cs = [json.loads(x) for x in open(f"{d}/episodes/safelibero_spatial_t{t}_e{e['episode_idx']}_calls.jsonl")]
    return e, dict(per[k]), cs


def replay(tag, t, init, **kw):
    e, ch, cs = episode(tag, t, init)
    tr = HeldMotionTracker(**kw)
    ref = None
    rows = []
    for c in cs:
        q = c["seq"]
        if q not in ch:
            continue
        z = np.load(ch[q])
        names = [str(n) for n in json.loads(str(z["summary_json"])).get("sphere_link_names", [])]
        sc = z["sphere_centres"].astype(float)
        hi = [i for i, n in enumerate(names) if n == "panda_hand"]
        if len(hi) < 3:
            continue
        if ref is None:
            ref = sc[hi].copy()
        T = kabsch(ref, sc[hi])
        g = c.get("grasp") or {}
        tip = lit(lit(g.get("tip_alignment")).get("now")).get("tip_m")
        held = g.get("state") == "held"
        cloud = z["filtered_cloud"].astype(float)[~z["support_mask"].astype(bool)]
        tgt = z["target_points"].astype(float) if "target_points" in z.files and len(z["target_points"]) else None
        m = tr.update(cloud, T, None if tip is None else np.asarray(tip, float), held, None if held else tgt)
        if not held:
            continue
        pc = lit((c.get("to") or {}).get("passthrough_clear"))
        w = lit(lit(pc.get("obstacle_rows")).get("worst"))
        p = np.asarray(w.get("point_m") or [np.nan] * 3, float)
        near = np.linalg.norm(cloud - p, axis=1) <= 0.06 if np.isfinite(p).all() else np.zeros(len(cloud), bool)
        rows.append(dict(seq=q, applied=bool(pc.get("applied")), cl=round(1000 * float(w.get("clearance_m") or 0), 1),
                         n_near=int(near.sum()), removed_near=int((near & m).sum()), removed=int(m.sum()), **tr.last))
    return rows


if __name__ == "__main__":
    tag, t, init = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    kw = {}
    if len(sys.argv) > 4:
        kw = dict(reach_pad_m=float(sys.argv[4]) / 1e3, reach_cap_m=float(sys.argv[5]) / 1e3, drift_m=float(sys.argv[6]) / 1e3,
                  match_m=float(sys.argv[7]) / 1e3)
    for r in replay(tag, t, init, **kw):
        print(f"seq {r['seq']:2d} {'P' if r['applied'] else 'Q'} cl {r['cl']:6.1f} near-row {r['removed_near']:3d}/{r['n_near']:3d} removed | all removed {r['removed']:4d} verified {r.get('n_verified')} new {r.get('n_new')} drift {r.get('median_drift_mm')} reach {r.get('reach_mm')}")
