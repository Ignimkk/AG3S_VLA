"""(.venv-ag3s, CPU) SL-PE offline replay — identity from where the hand closes + late latch + split from fixtures.
(2026-11-08 19:xx KST, dev-30)

Same records / scenes / GT roles as dev-e5's quantify.py (GT only scores). Per pre-held call (searching / latched /
closing) of the gt24att records, rules are replayed on the *recorded* cloud; the arm motion is the recorded one, so this
is a counterfactual of perception only (the TO's interventions that followed the wrong identity stay in the trajectory).

Rules
  R0   recorded grounding (`target_points`).
  SEED seed point = plan_close tip (finger-tip midpoint at the first row of the incoming chunk whose gripper command is
       below the latch threshold) when the chunk has one, else none (R0 is kept).
  SPLIT(seed): non-support points within `CROP` m of the seed; local horizontal support = the highest 5 mm z-bin at or below
       seed z + `ABOVE` whose points span >= `PLANE_SPAN` m in x or y (cabinet top, drawer floor, stove plate); drop points
       at or below that bin top + `PLANE_TOL`; 10 mm-grid 26-connected components; keep the component nearest to the seed
       (<= `REACH` m). If it is larger than `MAX_EXT` m, keep only its points within `BALL` m of the seed.
  PE   identity per call = SPLIT(plan_close) when available, else R0.
  PE+L late latch: identity is free until the first call with plan_close whose SPLIT exists for 2 consecutive calls with
       centroid shift <= `LOCK_SHIFT`; then locked: each later call re-runs SPLIT seeded at the locked centroid (tracking).
       Before the lock the identity is the R0 one (status quo while searching).
Scores per call: majority role of the identity points (<= 10 mm to a true surface), purity (target share), target recall
  (share of cloud points labelled target that are in the identity set), and M4 = target voxels in the scene obstacle set
  (dev-e5 `scene_obstacles` with the identity points as target) plus M4x = same after also removing the identity points
  (+ 15 mm) before the component step (= "scene obstacles minus the target candidate").
Run: SCENE_DIR=... [SAFELIBERO_PATCH=1] python pe_replay.py '<run glob>' <tag>
"""
import collections
import json
import pathlib
import sys

import mujoco
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pe_signal as PS  # noqa: E402
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.stages import close_seed as CS  # noqa: E402
Q = PS.Q

OUT = pathlib.Path(__file__).resolve().parent
CROP, ABOVE, PLANE_SPAN, PLANE_TOL, REACH, MAX_EXT, BALL, LOCK_SHIFT = 0.15, 0.02, 0.10, 0.004, 0.06, 0.20, 0.08, 0.03
LOCK_TRACK = 0.06
GRASP_EXT = 0.15  # longest side above which a component is not one graspable object  # after the lock, a plan_close component within 60 mm of the locked centroid is the same object
GRID = float(__import__("os").environ.get("PE_GRID", "0.010"))
PLANE_CELLS = 40   # >= 40 occupied 1 cm x-y cells (40 cm^2) of upward-facing points = a support, not a bowl bottom


def upward(P, k=12):
    """points whose local PCA normal is within ~25 deg of vertical."""
    if len(P) < k:
        return np.zeros(len(P), bool)
    _, nb = cKDTree(P).query(P, k=k)
    X = P[nb] - P[nb].mean(axis=1, keepdims=True)
    C = np.einsum("nki,nkj->nij", X, X)
    w, v = np.linalg.eigh(C)
    return np.abs(v[:, 2, 0]) >= 0.9


def xy_cells(q):
    return len(np.unique(np.floor(q[:, :2] / 0.01).astype(int), axis=0))


def vertical_walls(P, rng, n_iter=200, tol=0.004, min_in=40, min_span=0.10, max_planes=3):
    """mask of points on large flat vertical planes (drawer / cabinet walls): RANSAC on 2 points + vertical axis."""
    wall = np.zeros(len(P), bool)
    for _ in range(max_planes):
        cand = np.where(~wall)[0]
        if len(cand) < min_in:
            break
        best = None
        for _ in range(n_iter):
            a, b = P[rng.choice(cand, 2, replace=False)]
            t = (b - a)[:2]
            if np.linalg.norm(t) < 0.02:
                continue
            nrm = np.array([-t[1], t[0], 0.0]) / np.linalg.norm(t)
            inl = cand[np.abs((P[cand] - a) @ nrm) <= tol]
            if best is None or len(inl) > len(best[0]):
                best = (inl, t / np.linalg.norm(t))
        if best is None or len(best[0]) < min_in:
            break
        inl, t = best
        if np.ptp(P[inl][:, :2] @ t) < min_span:
            break
        wall[inl] = True
    return wall


def split(pts, seed):
    """the server module (single source): AG3S/ag3s/stages/close_seed.split_at_seed."""
    idx, _ = CS.split_at_seed(pts, seed)
    return idx


def obstacles_minus(pts, support, tgt, dest_aabb):
    """M4x: remove the identity points (+15 mm) first, then dev-e5's scene_obstacles copy."""
    if tgt is None or not len(tgt):
        return Q.scene_obstacles(pts, support, tgt, dest_aabb)
    far = cKDTree(tgt).query(pts)[0] > 0.015
    return Q.scene_obstacles(pts[far], support[far], None, dest_aabb)


def score(P_world, tree, R, cloud_lab):
    rc = Q.role_counts(P_world, tree, R, 0.010) if P_world is not None and len(P_world) else None
    return rc


def main():
    run = next(Q.E.RUNS.glob(sys.argv[1]))
    tag = sys.argv[2]
    eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
    order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
    success = {o: bool(e["official_success"]) for o, e in zip(order, eps)}
    ch = PS.episode_chunks(run, order)
    res = {}
    for stem in order:
        if not (Q.SCENE / f"{stem}_scene.xml").exists():
            continue
        info, m, role_geom, cache, st_cache = PS.scene_for(stem)
        d = mujoco.MjData(m)
        geom = np.load(run / "episodes" / f"{stem}_geom.npz")
        grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
        calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
        st_tree = st_roles = None
        lock_c, cand_prev, rows = None, None, []
        for ts, cp in ch.get(stem, []):
            call = calls[ts]
            g = call.get("grasp") or {}
            latch = g.get("state")
            if latch not in PS.PRE:
                continue
            i = grow[int(call["obs_index"])]
            d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]
            mujoco.mj_forward(m, d)
            if st_tree is None:
                SS, SSG = st_cache.world(d)
                st_tree, st_roles = cKDTree(SS), role_geom[SSG]
            S, SG = cache.world(d)
            tree = Q.SplitTree(st_tree, st_roles, S, role_geom[SG])
            R = tree.roles
            Tw = geom["T_world_base"][i]
            W = lambda X: np.asarray(X, float).reshape(-1, 3) @ Tw[:3, :3].T + Tw[:3, 3]
            z = np.load(cp)
            cloud = z["filtered_cloud"].astype(float)
            sup = z["support_mask"].astype(bool)
            ns = np.where(~sup)[0]
            dist, j = tree.query(W(cloud))
            cl_lab = np.where(dist <= 0.010, R[j], "none")
            n_tgt_cloud = int((cl_lab == "target").sum())
            summ = json.loads(str(z["summary_json"]))
            dd = summ.get("destination")
            aabb = (np.array(dd["aabb_min"]), np.array(dd["aabb_max"])) if dd and "aabb_min" in dd else None
            P0 = z["target_points"].astype(float) if "target_points" in z.files else np.zeros((0, 3))
            pc = (g.get("tip_alignment") or {}).get("plan_close")
            # identity of each rule as base-frame points (R0 = the recorded target points: while the identity is occluded
            # they are the kept last geometry and have no index in this cloud, so they are scored as points, like M1)
            idp = {"R0": P0 if len(P0) else None}
            seed_ids = seed_ids_ns = None
            if pc:
                seed_ids_ns = split(cloud[ns], np.asarray(pc["tip_m"], float))
                seed_ids = ns[seed_ids_ns] if seed_ids_ns is not None else None
            idp["PE"] = cloud[seed_ids] if seed_ids is not None else idp["R0"]
            # late latch
            if lock_c is None and seed_ids is not None:
                c = cloud[seed_ids].mean(axis=0)
                if cand_prev is not None and np.linalg.norm(c - cand_prev) <= LOCK_SHIFT:
                    lock_c = c
                cand_prev = c
            elif lock_c is None:
                cand_prev = None
            if lock_c is not None:
                s_ = seed_ids_ns if (seed_ids is not None
                                     and np.linalg.norm(cloud[seed_ids].mean(axis=0) - lock_c) <= LOCK_TRACK) \
                    else split(cloud[ns], lock_c)
                if s_ is not None:
                    lock_c = cloud[ns[s_]].mean(axis=0)
                idp["PEL"] = cloud[ns[s_]] if s_ is not None else None
            else:
                idp["PEL"] = idp["R0"]
            row = {"seq": ts, "latch": latch, "plan_close": bool(pc), "locked": lock_c is not None,
                   "gstat": summ.get("grounding_status"), "R0_recorded": bool(len(P0)),
                   "seed_cut": seed_ids is not None, "n_target_cloud": n_tgt_cloud}
            tcloud = cloud[cl_lab == "target"]
            for k, tg in idp.items():
                if tg is None or not len(tg):
                    row[k] = {"majority": "no target"}
                    continue
                dd_, jj = tree.query(W(tg))
                lab = np.where(dd_ <= 0.010, R[jj], "none")
                c = collections.Counter(lab.tolist())
                rc = (float(np.mean(cKDTree(tg).query(tcloud)[0] <= 0.010)) if len(tcloud) else None)
                r_ = {"n": int(len(tg)), "majority": c.most_common(1)[0][0],
                      "purity": round(c.get("target", 0) / len(tg), 3),
                      "recall": None if rc is None else round(rc, 3)}
                ob = Q.scene_obstacles(cloud, sup, tg, aabb)
                if len(ob):
                    oc = Q.role_counts(W(ob), tree, R, 0.015)
                    r_["M4"] = {"target": oc.get("target", 0), "robot": oc.get("robot", 0), "n": oc["n"],
                                "fixture": oc.get("fixture", 0)}
                else:
                    r_["M4"] = {"target": 0, "robot": 0, "n": 0, "fixture": 0}
                row[k] = r_
            rows.append(row)
        res[stem] = {"success": success[stem], "target": info["target"], "rows": rows}
        print(stem, "ok" if success[stem] else "FAIL", len(rows), flush=True)
    (OUT / f"replay_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
