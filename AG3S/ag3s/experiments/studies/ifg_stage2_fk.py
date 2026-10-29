"""IFG-S2FK — stage-2 components of the adopted method B, re-measured on held-out WITHOUT the rendered robot mask (FK spheres).

Task: AG3S/ag3s/docs/handoff/IFG-S2FK.task.md (fixed before results). CPU only. No locked file is modified: the locked
per-frame / per-episode functions are called through the held-out adapter (`ifg_heldout_eval.install`).

Stages (all write under --out):
  prep      robot_mask_fk/<frame>.npz  = the FK-sphere mask stored in the held-out replay npz (key robot_mask, unchanged)
            run/  pseudo run: s0/replay, s0/labels -> held-out prep; rounds/R6/robot_mask -> robot_mask_fk;
                  rounds/R6/rows, rounds/R5/rows -> the locked held-out rows (A-fk / G:N references for the summary)
  memory    `ifg_memory.episode(center=True)` on the pseudo run -> run/rounds/R10b/rows (A-fk = arm "R6", B-fk = Mem-first...)
            and run/rounds/R11b/conf.jsonl (gate score); then the locked summary rule (`ifg_heldout_summary`)
  features  per frame and per robot-mask condition (render | fk): the `ifg_phase.episode_features` quantities
            (hand_d, height, lognodes, grip, hsr) -> features_<cond>.jsonl
  phase     rules fixed in the task (held_rule / phase_rule, D_hold 0.131 m) vs HSR argmax / grip rule / locked LR / LOEO LR
"""
import argparse, collections, json, os, pathlib, multiprocessing as mp

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_heldout_eval as HE
from AG3S.ag3s.experiments.studies import ifg_reader as R

ST = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development")
PREP = ST / "20261016T030741+0900_f15e587290f6_19ea09c7a8b8/heldout_eval"
LOCKED = ST / "20261016T030741+0900_f15e587290f6_19ea09c7a8b8/heldout_locked"
DUMP = ST / "20261016T000420+0900_30955b6ca8ef_2498252a873f/heldout/dump"
SNAP_XML = ST / "20261016T000420+0900_30955b6ca8ef_2498252a873f/snapshot/robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml"
LOCK = ST / "20261015T181706+0900_30955b6ca8ef_d4b3bafb79ba/LOCK_heldout.json"
#: fixed in the task before results: hand proxy -> finger grasp point 0.0813 m (model, teleop) + fruit radius bound 0.05 m
D_HOLD = 0.0813 + 0.05
GRIP_CLOSED = 0.9
PHASES = ("pick", "place", "home")
PHASE_OF = {"pre": "pick", "post": "pick", "held": "place", "placed": "home"}


def link(p: pathlib.Path, target: pathlib.Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() and not p.is_symlink():
        os.symlink(os.path.relpath(target, p.parent), p)


def stage_prep(out: pathlib.Path):
    md = out / "robot_mask_fk"; md.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((PREP / "replay").glob("*.npz")):
        g = md / f.name
        if g.exists():
            continue
        with np.load(f) as z:
            np.savez_compressed(g, robot_mask=z["robot_mask"], source=str(f), note="serving FK-sphere self-filter (held-out replay)")
        n += 1
    run = out / "run"
    link(run / "s0/replay", PREP / "replay"); link(run / "s0/labels", PREP / "labels")
    link(run / "rounds/R6/robot_mask", md)
    link(run / "rounds/R6/rows", LOCKED / "run/rounds/R6/rows"); link(run / "rounds/R5/rows", LOCKED / "run/rounds/R5/rows")
    print("prep: fk masks written", n, "total", len(list(md.glob("*.npz"))), flush=True)
    return run


def stage_memory(out: pathlib.Path, workers: int):
    HE.install()
    run = out / "run"
    refs, skipped = HE.heldout_refs(DUMP, PREP)
    (out / "frames.json").write_text(json.dumps(dict(n=len(refs), skipped=skipped, frames=[dict(
        frame_id=r.frame_id, episode_id=r.episode_id, kind=r.kind, grasp_state=r.moment, stratum=r.done["stratum"]) for r in refs]), indent=0))
    (run / "rounds/R10b/rows").mkdir(parents=True, exist_ok=True); (run / "rounds/R11b").mkdir(parents=True, exist_ok=True)
    eps = collections.defaultdict(list)
    for r in refs:
        eps[r.episode_id].append(r)
    allconf = []
    with mp.get_context("spawn").Pool(min(workers, 4)) as p:
        for rows, conf in p.imap_unordered(HE._mem, [(v, run, True) for v in eps.values()]):
            for fid, rs in rows.items():
                (run / "rounds/R10b/rows" / f"{fid}.jsonl").write_text("".join(json.dumps(x, default=float) + "\n" for x in rs))
            allconf += conf
            print("memory episode done", len(allconf), flush=True)
    with open(run / "rounds/R11b/conf.jsonl", "w") as f:
        for c in allconf:
            f.write(json.dumps(c, default=float) + "\n")
    print("memory done", len(allconf), flush=True)


def _episode_features(args):
    refs, mask_dir = args
    HE.install()
    import mujoco
    from AG3S.ag3s.experiments.studies import ifg_graph as G
    from AG3S.ag3s.experiments.studies import ifg_memory as M
    from AG3S.ag3s.experiments.studies.t47_replay import GRIPPER_ROOT_LINKS
    m = mujoco.MjModel.from_xml_path(str(SNAP_XML)); d = mujoco.MjData(m)
    roots = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in GRIPPER_ROOT_LINKS}
    below = {n: [] for n in roots}
    for b in range(m.nbody):
        j = b
        while j > 0:
            j = int(m.body_parentid[j])
            for n, r in roots.items():
                if j == r:
                    below[n].append(b)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    cfg = G.GraphConfig()
    out = []
    for ref in sorted(refs, key=lambda r: r.frame):
        with np.load(PREP / "replay" / f"{ref.frame_id}.npz") as z:
            depth, K, T, qpos = z["depth"], z["K"], z["T_base_cam"], z["qpos"]
        with np.load(mask_dir / f"{ref.frame_id}.npz") as z:
            rmask = z["robot_mask"]
        with np.load(PREP / "labels" / f"{ref.frame_id}.npz") as z:
            class_map, classes = z["class_map"], [str(c) for c in z["classes"]]
        d.qpos[:] = qpos; mujoco.mj_kinematics(m, d)
        Rb, tb = d.xmat[bid].reshape(3, 3), d.xpos[bid]
        tcp = [Rb.T @ (d.xpos[below[n]].mean(0) - tb) for n in GRIPPER_ROOT_LINKS]
        fin = R.load_frame(ref, features=())
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
        out.append(dict(frame_id=ref.frame_id, episode_id=ref.episode_id, frame=ref.frame, kind=ref.kind, stratum=ref.done["stratum"],
                        grasp_state=ref.done["grasp_state"], hand_d=hand_d, height=height, lognodes=float(np.log1p(nn)),
                        grip=float(min(fin.raw_state[7], fin.raw_state[15])), hsr=hsr.tolist()))
    return out


def stage_features(out: pathlib.Path, workers: int, cond: str):
    HE.install()
    refs, _ = HE.heldout_refs(DUMP, PREP)
    eps = collections.defaultdict(list)
    for r in refs:
        eps[r.episode_id].append(r)
    mask_dir = PREP / "robot_mask_render" if cond == "render" else out / "robot_mask_fk"
    rows = []
    with mp.get_context("spawn").Pool(min(workers, 4)) as p:
        for res in p.imap_unordered(_episode_features, [(v, mask_dir) for v in eps.values()]):
            rows += res
            print("features", cond, len(rows), flush=True)
    (out / f"features_{cond}.jsonl").write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))


# ------------------------------------------------------------------------------------------------------------- phase / held
def rules(rows):
    """held_rule / phase_rule (task-fixed). rows of one episode in time order -> lists of predictions."""
    held, phase, seen = [], [], False
    for r in rows:
        h = r["grip"] < GRIP_CLOSED and (not np.isfinite(r["hand_d"]) or r["hand_d"] <= D_HOLD)
        held.append(int(h))
        if h:
            phase.append(1); seen = True
        elif seen and r["grip"] >= GRIP_CLOSED:
            phase.append(2)
        else:
            phase.append(0)
    return held, phase


def stage_phase(out: pathlib.Path):
    from AG3S.ag3s.experiments.studies import ifg_phase as P
    lock = json.loads(LOCK.read_text())["arms"]["phase_held"]
    res = {"D_hold_m": D_HOLD, "grip_closed": GRIP_CLOSED, "locked_lr": {k: lock[k] for k in ("model", "columns")}}
    for cond in ("render", "fk"):
        f = out / f"features_{cond}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(l) for l in open(f)]
        rows.sort(key=lambda r: (r["episode_id"], r["frame"]))
        for subset in ("success", "all"):
            rs = [r for r in rows if subset == "all" or r["kind"] == "success"]
            eps = np.array([r["episode_id"] for r in rs])
            y3 = np.array([PHASES.index(PHASE_OF[r["grasp_state"]]) for r in rs]); yh = np.array([int(r["grasp_state"] == "held") for r in rs])
            H = np.log(np.clip(np.array([r["hsr"] for r in rs]), 1e-9, 1))
            F = {k: np.array([r[k] for r in rs], float) for k in ("hand_d", "height", "lognodes", "grip")}
            for i, c in enumerate(PHASES):
                F[f"hsr_{c}"] = H[:, i]
            hr, pr = np.zeros(len(rs), int), np.zeros(len(rs), int)
            for e in np.unique(eps):
                ix = np.flatnonzero(eps == e)
                a, b = rules([rs[i] for i in ix]); hr[ix] = a; pr[ix] = b
            X = np.column_stack([F[c] for c in lock["columns"]])
            Xs = np.nan_to_num((X - np.array(lock["standardise_mean"])) / np.array(lock["standardise_sd"]))
            def lk(name):
                W = np.array(lock[name]); return (np.hstack([Xs, np.ones((len(Xs), 1))]) @ W).argmax(1)
            hsr_cols = [f"hsr_{c}" for c in PHASES]; geo = ["hand_d", "height", "lognodes"]
            ph = {"HSR argmax (serving)": H.argmax(1), "rule phase_rule (no learning)": pr, "LR locked (dev render weights)": lk("W_phase"),
                  "LR LOEO on this set (reference only)": P.loeo(F, hsr_cols + ["grip"] + geo, y3, eps, 3)}
            hd = {"grip<0.9 (serving)": (F["grip"] < GRIP_CLOSED).astype(int), "rule held_rule (no learning)": hr,
                  "LR LOEO on this set (reference only)": P.loeo(F, hsr_cols + ["grip"] + geo, yh, eps, 2)}
            if "W_held" in lock:
                hd["LR locked (dev render weights)"] = lk("W_held")
            blk = {"n_frames": len(rs), "n_episodes": int(len(np.unique(eps))), "region_missing_frames": int(np.sum(~np.isfinite(F["hand_d"]))),
                   "class_counts": {"phase": np.bincount(y3, minlength=3).tolist(), "held": np.bincount(yh, minlength=2).tolist()},
                   "phase": {k: P.metrics(y3, v, 3, eps) for k, v in ph.items()}, "held": {k: P.metrics(yh, v, 2, eps) for k, v in hd.items()}}
            pairs = {}
            for task, d_, refk in (("phase", blk["phase"], "HSR argmax (serving)"), ("held", blk["held"], "grip<0.9 (serving)")):
                for k in d_:
                    if k != refk:
                        pairs[f"{task}: {k} - {refk}"] = P.paired(d_[k]["episode_acc"], d_[refk]["episode_acc"])
                lkk = "LR locked (dev render weights)"
                rk = "rule phase_rule (no learning)" if task == "phase" else "rule held_rule (no learning)"
                if lkk in d_:
                    pairs[f"{task}: {rk} - {lkk}"] = P.paired(d_[rk]["episode_acc"], d_[lkk]["episode_acc"])
            blk["paired"] = pairs
            for t in ("phase", "held"):
                for k in blk[t]:
                    blk[t][k] = {kk: vv for kk, vv in blk[t][k].items() if kk != "episode_acc"}
            res[f"{cond}|{subset}"] = blk
    (out / "phase_summary.json").write_text(json.dumps(res, indent=1, default=float))
    for k, v in res.items():
        if "|" in k:
            print(k, "n", v["n_frames"], "missing region", v["region_missing_frames"])
            for t in ("phase", "held"):
                print("  ", t, {kk: round(vv["accuracy"], 3) for kk, vv in v[t].items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--stage", choices=["prep", "memory", "features", "phase"], required=True)
    ap.add_argument("--cond", choices=["render", "fk"], default="fk")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == "prep":
        stage_prep(a.out)
    elif a.stage == "memory":
        stage_memory(a.out, a.workers)
    elif a.stage == "features":
        stage_features(a.out, a.workers, a.cond)
    else:
        stage_phase(a.out)


if __name__ == "__main__":
    main()
