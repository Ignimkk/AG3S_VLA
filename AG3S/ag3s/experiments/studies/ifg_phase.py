"""IFG R12 — phase (pick/place/home) and held judgment from HSR + gripper + 3D target relations (+ feature identity).

Stage `features` (CPU, MuJoCo for FK only): per frame, the R6 target region (ifg_memory.r6_choice) and
  hand_d   min distance from its centroid to the two gripper TCP proxies (mean position of the bodies
           below link_{left,right}_arm_6 at the recorded qpos, base frame)
  height   centroid z - median z of support nodes within 0.25 m (xy) of the centroid
  lognodes log(1 + region node count)
  grip     min of the two gripper state columns (7, 15) of the recorded raw state
  hsr      log of derived.npz hsr_p (pick, place, home)
  q_mem    q_mem_first:prefix_L8 from rounds/R11/conf.jsonl (first frame: 1.0, has_mem 0)
Stage `eval`: P1 HSR argmax, P0 gripper rule, and leave-one-episode-out L2 logistic regressions
(supervised ablation; numpy, standardised inputs, L2 = 1.0 fixed in the registry).
"""
import argparse, json, multiprocessing as mp, pathlib

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_graph as G
from AG3S.ag3s.experiments.studies import ifg_memory as M
from AG3S.ag3s.experiments.studies import ifg_reader as R

PHASES = ("pick", "place", "home")
PHASE_OF = {"pre": "pick", "post": "pick", "held": "place", "placed": "home"}
L2 = 1.0


def episode_features(args):
    refs, run = args
    import mujoco
    from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
    from AG3S.ag3s.experiments.studies.t47_replay import GRIPPER_ROOT_LINKS

    rec = load_run(R.record_dir(refs[0]))
    scene = replay_scene(rec)
    m = scene.model
    roots = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in GRIPPER_ROOT_LINKS}
    below = {n: [] for n in roots}
    for b in range(m.nbody):
        j = b
        while j > 0:
            j = int(m.body_parentid[j])
            for n, r in roots.items():
                if j == r:
                    below[n].append(b)
    cfg = G.GraphConfig()
    out = []
    try:
        for ref in sorted(refs, key=lambda r: r.frame):
            pose_scene(scene, rec.steps[ref.frame])
            Tb = np.linalg.inv(scene.body_pose("base"))
            tcp = [Tb[:3, :3] @ scene.data.xpos[below[n]].mean(0) + Tb[:3, 3] for n in GRIPPER_ROOT_LINKS]
            fin = R.load_frame(ref, features=())
            with np.load(run / "s0/replay" / f"{ref.frame_id}.npz") as z:
                depth, K, T = z["depth"], z["K"], z["T_base_cam"]
            with np.load(run / "rounds/R6/robot_mask" / f"{ref.frame_id}.npz") as z:
                rmask = z["robot_mask"]
            with np.load(run / "s0/labels" / f"{ref.frame_id}.npz") as z:
                class_map, classes = z["class_map"], [str(c) for c in z["classes"]]
            geo = G.build_geometry(depth, K, T, rmask, class_map, classes, cfg)
            c = M.r6_choice(geo, cfg, fin.maps["N"])
            ch = None if c is None else c["chosen"]
            if ch is not None and ch.any():
                cen = geo.pos[ch].mean(0)
                sp_ = geo.pos[geo.support]
                near = sp_[np.linalg.norm(sp_[:, :2] - cen[:2], axis=1) <= 0.25] if len(sp_) else sp_
                zs = np.median(near[:, 2]) if len(near) else (np.median(sp_[:, 2]) if len(sp_) else np.nan)
                hand_d = float(min(np.linalg.norm(cen - t) for t in tcp)); height = float(cen[2] - zs); nn = int(ch.sum())
            else:
                hand_d, height, nn = np.nan, np.nan, 0
            with np.load(ref.path / "derived.npz") as z:
                hsr = np.asarray(z["hsr_p"], float)
            out.append(dict(frame_id=ref.frame_id, episode_id=ref.episode_id, moment=ref.moment, kind=ref.kind,
                            grasp_state=ref.done["grasp_state"], hand_d=hand_d, height=height, lognodes=float(np.log1p(nn)),
                            grip=float(min(fin.raw_state[7], fin.raw_state[15])), hsr=hsr.tolist()))
    finally:
        scene.close()
    return out


def fit_lr(X, y, k, l2=L2, iters=500):
    """Multinomial logistic regression, full-batch gradient descent with backtracking-free fixed step on standardised X."""
    n, d = X.shape
    W = np.zeros((d + 1, k))
    Xb = np.hstack([X, np.ones((n, 1))])
    Y = np.eye(k)[y]
    lr = 0.5
    for _ in range(iters):
        Z = Xb @ W
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        g = Xb.T @ (P - Y) / n
        g[:-1] += l2 * W[:-1] / n
        W -= lr * g
    return W


def predict(W, X):
    Z = np.hstack([X, np.ones((len(X), 1))]) @ W
    return Z.argmax(1)


def loeo(F, cols, y, eps, k):
    X = np.column_stack([F[c] for c in cols]).astype(float)
    pred = np.zeros(len(y), int)
    for e in np.unique(eps):
        tr, te = eps != e, eps == e
        mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0) + 1e-9
        Xtr = np.nan_to_num((X[tr] - mu) / sd); Xte = np.nan_to_num((X[te] - mu) / sd)
        pred[te] = predict(fit_lr(Xtr, y[tr], k), Xte)
    return pred


def metrics(y, p, k, eps):
    acc = float((y == p).mean())
    f1 = []
    rec = {}
    for c in range(k):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f1.append(2 * tp / max(2 * tp + fp + fn, 1)); rec[c] = float(tp / max((y == c).sum(), 1))
    ep_acc = {e: float((y[eps == e] == p[eps == e]).mean()) for e in np.unique(eps)}
    return dict(accuracy=acc, macro_f1=float(np.mean(f1)), recall=rec, episode_acc=ep_acc)


def paired(a, b):
    es = sorted(a); d = np.array([a[e] - b[e] for e in es])
    rng = np.random.default_rng(20261015)
    boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    return dict(diff=float(d.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                better=int((d > 0).sum()), worse=int((d < 0).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--stage", choices=["features", "eval"], required=True)
    a = ap.parse_args()
    out = a.run / "rounds/R12"
    if a.stage == "features":
        eps = {}
        for r in R.list_frames():
            eps.setdefault(r.episode_id, []).append(r)
        rows = []
        with mp.get_context("spawn").Pool(4) as p:
            for res in p.imap_unordered(episode_features, [(v, a.run) for v in eps.values()]):
                rows += res
        conf = {json.loads(l)["frame_id"]: json.loads(l) for l in open(a.run / "rounds/R11/conf.jsonl")}
        for r in rows:
            q = conf.get(r["frame_id"], {}).get("q_mem_first:prefix_L8")
            r["q_mem"] = 1.0 if q is None else q; r["has_mem"] = 0.0 if q is None else 1.0
        (out / "features.jsonl").write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
        print(len(rows)); return
    rows = [json.loads(l) for l in open(out / "features.jsonl")]
    F = {k: np.array([r[k] for r in rows], float) for k in ("hand_d", "height", "lognodes", "grip", "q_mem", "has_mem")}
    H = np.log(np.clip(np.array([r["hsr"] for r in rows]), 1e-9, 1))
    for i, c in enumerate(PHASES):
        F[f"hsr_{c}"] = H[:, i]
    eps = np.array([r["episode_id"] for r in rows])
    y3 = np.array([PHASES.index(PHASE_OF[r["grasp_state"]]) for r in rows])
    yh = np.array([int(r["grasp_state"] == "held") for r in rows])
    hsr = [f"hsr_{c}" for c in PHASES]; geo = ["hand_d", "height", "lognodes"]; feat = ["q_mem", "has_mem"]
    res = {"phase": {}, "held": {}}
    res["phase"]["P1 HSR argmax"] = metrics(y3, H.argmax(1), 3, eps)
    res["phase"]["P0 rule grip<0.9 -> place"] = metrics(y3, np.where(F["grip"] < 0.9, 1, 0), 3, eps)
    for name, cols in (("LR[hsr]", hsr), ("LR[grip+geo]", ["grip"] + geo), ("LR[hsr+grip]", hsr + ["grip"]),
                       ("LR[hsr+grip+geo]", hsr + ["grip"] + geo), ("LR[hsr+grip+geo+feat]", hsr + ["grip"] + geo + feat)):
        res["phase"][name] = metrics(y3, loeo(F, cols, y3, eps, 3), 3, eps)
    res["held"]["rule grip<0.9"] = metrics(yh, (F["grip"] < 0.9).astype(int), 2, eps)
    res["held"]["HSR place>0.5"] = metrics(yh, (H[:, 1] > np.log(0.5)).astype(int), 2, eps)
    for name, cols in (("LR[grip+geo]", ["grip"] + geo), ("LR[hsr+grip]", hsr + ["grip"]), ("LR[hsr+grip+geo]", hsr + ["grip"] + geo),
                       ("LR[hsr+grip+geo+feat]", hsr + ["grip"] + geo + feat)):
        res["held"][name] = metrics(yh, loeo(F, cols, yh, eps, 2), 2, eps)
    comps = {}
    for task, ref in (("phase", "P1 HSR argmax"), ("held", "rule grip<0.9")):
        for k, v in res[task].items():
            if k != ref:
                comps[f"{task}: {k} - {ref}"] = paired(v["episode_acc"], res[task][ref]["episode_acc"])
    comps["phase: LR[hsr+grip+geo+feat] - LR[hsr+grip+geo]"] = paired(res["phase"]["LR[hsr+grip+geo+feat]"]["episode_acc"], res["phase"]["LR[hsr+grip+geo]"]["episode_acc"])
    comps["phase: LR[hsr+grip+geo] - LR[hsr+grip]"] = paired(res["phase"]["LR[hsr+grip+geo]"]["episode_acc"], res["phase"]["LR[hsr+grip]"]["episode_acc"])
    comps["held: LR[hsr+grip+geo+feat] - LR[hsr+grip+geo]"] = paired(res["held"]["LR[hsr+grip+geo+feat]"]["episode_acc"], res["held"]["LR[hsr+grip+geo]"]["episode_acc"])
    (out / "summary.json").write_text(json.dumps(dict(n_frames=len(rows), class_counts={"phase": np.bincount(y3, minlength=3).tolist(), "held": np.bincount(yh, minlength=2).tolist()},
                                                      results={t: {k: {kk: vv for kk, vv in v.items() if kk != "episode_acc"} for k, v in d.items()} for t, d in res.items()},
                                                      paired=comps), indent=1))
    for t, d in res.items():
        print(t)
        for k, v in d.items():
            print(f"  {k:28s} acc={v['accuracy']:.3f} macroF1={v['macro_f1']:.3f} recall={ {kk: round(vv, 3) for kk, vv in v['recall'].items()} }")
    for k, v in comps.items():
        print(f"  {k}: {v['diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] {v['better']}/{v['worse']}")


if __name__ == "__main__":
    main()
