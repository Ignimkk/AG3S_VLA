"""(.venv-ag3s, CPU) SL-PE step 0 — how good is the VLA's own "where the hand closes" signal as a target identity?
(2026-11-08 19:xx KST, dev-30)

Input: gt24att closed-loop records (same runs / scenes / GT roles as dev-e5's quantify.py; GT only scores).
Per pre-held call (latch searching / latched / closing):
  - plan_close tip (base frame) = `grasp.tip_alignment.plan_close.tip_m`: finger-tip midpoint at the first row of the
    incoming policy chunk whose gripper command is below the latch threshold (recorded, measurement only).
  - end of the predicted EE path = `predicted_ee_base[-1]` (response model, 5 steps).
  For each probe point: role of the nearest true *object* surface (target / distractor / destination / other object),
  its distance, and the distance to the true target surface.
Also: calls between the first plan_close and the first `closing` (how early the signal exists), first-latch call.
Run: SCENE_DIR=<scene dir> [SAFELIBERO_PATCH=1] python pe_signal.py '<run glob>' <tag>
"""
import json
import pathlib
import sys

import mujoco
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-08_GT없는_인식_정량")
import quantify as Q  # noqa: E402  (reads SCENE_DIR / SAFELIBERO_PATCH)

OUT = pathlib.Path(__file__).resolve().parent
OBJ = ("target", "distractor", "destination", "other object")
PRE = ("searching", "latched", "closing")


def episode_chunks(run, order):
    ch, ep_i, last = {}, -1, None
    for c in sorted((run / "server_constraints" / "run_0000").glob("chunk_*.npz")):
        ts = int(np.load(c)["t_step"])
        if last is None or ts <= last:
            ep_i += 1
        last = ts
        ch.setdefault(order[ep_i], []).append((ts, c))
    return ch


def scene_for(stem):
    """MuJoCo model + GT role per geom for one episode (same role rules as quantify.py)."""
    suite, t = stem.rsplit("_t", 1)[0], int(stem.split("_t")[-1].split("_")[0])
    info = Q.tt.task_target(suite, t)
    target, dest = info["target"], info.get("destination")
    tcat = Q.category(target)
    m = mujoco.MjModel.from_xml_path(str(Q.SCENE / f"{stem}_scene.xml"))
    arr = np.load(Q.SCENE / f"{stem}_model_arrays.npz")
    for k in ("body_pos", "body_quat"):
        getattr(m, k)[:] = arr[k]
    bn = [m.body(i).name for i in range(m.nbody)]
    gbody = np.array([bn[b] for b in m.geom_bodyid])
    objs = list(info["objects"])

    def role(b):
        if b.startswith(Q.E.ROBOT):
            return "robot"
        if b.startswith(target + "_") or b == target:
            return "target"
        if dest and (b.startswith(dest + "_") or b == dest):
            return "destination"
        if b.startswith(Q.FIX):
            return "fixture"
        for o in objs:
            if b.startswith(o + "_"):
                return "distractor" if Q.category(o) == tcat else "other object"
        return "fixture"

    role_geom = np.array([role(b) for b in gbody])
    vis = np.isin(m.geom_group, [1, 2])
    static = (m.body_weldid[m.geom_bodyid] == 0)
    return info, m, role_geom, Q.SurfaceCache(m, vis & ~static), Q.SurfaceCache(m, vis & static)


def probe(p_world, S, R):
    """nearest object role, its distance (mm), distance to the target surface (mm)."""
    obj = np.isin(R, OBJ)
    if not obj.any():
        return None
    d_all, j = cKDTree(S[obj]).query(p_world)
    tg = R == "target"
    d_t = float(cKDTree(S[tg]).query(p_world)[0]) if tg.any() else None
    return {"role": str(R[obj][j]), "d_mm": round(1e3 * float(d_all), 1),
            "d_target_mm": None if d_t is None else round(1e3 * d_t, 1)}


def main():
    run = next(Q.E.RUNS.glob(sys.argv[1]))
    tag = sys.argv[2]
    eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
    order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
    success = {o: bool(e["official_success"]) for o, e in zip(order, eps)}
    ch = episode_chunks(run, order)
    res = {}
    for stem in order:
        if not (Q.SCENE / f"{stem}_scene.xml").exists():
            continue
        info, m, role_geom, cache, _ = scene_for(stem)
        d = mujoco.MjData(m)
        geom = np.load(run / "episodes" / f"{stem}_geom.npz")
        grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
        calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
        cnpz = np.load(run / "episodes" / f"{stem}_calls.npz")
        cidx = {int(o): k for k, o in enumerate(cnpz["call_obs_index"])}
        rows = []
        for ts, _ in ch.get(stem, []):
            call = calls[ts]
            g = call.get("grasp") or {}
            latch = g.get("state")
            if latch not in PRE:
                continue
            i = grow[int(call["obs_index"])]
            d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]
            mujoco.mj_forward(m, d)
            S, SG = cache.world(d)
            R = role_geom[SG]
            Tw = geom["T_world_base"][i]
            W = lambda X: np.asarray(X, float).reshape(-1, 3) @ Tw[:3, :3].T + Tw[:3, 3]
            row = {"seq": ts, "obs": int(call["obs_index"]), "latch": latch}
            pc = (g.get("tip_alignment") or {}).get("plan_close")
            if pc:
                row["plan_close"] = {"row": int(pc["row"]), **probe(W(pc["tip_m"])[0], S, R)}
            k = cidx.get(int(call["obs_index"]))
            if k is not None:
                row["path_end"] = probe(W(cnpz["predicted_ee_base"][k][-1])[0], S, R)
            rows.append(row)
        first_pc = next((r["seq"] for r in rows if "plan_close" in r), None)
        first_close = next((r["seq"] for r in rows if r["latch"] == "closing"), None)
        first_latch = next((r["seq"] for r in rows if r["latch"] == "latched"), None)
        res[stem] = {"success": success[stem], "target": info["target"], "rows": rows,
                     "first_plan_close_seq": first_pc, "first_closing_seq": first_close, "first_latched_seq": first_latch}
        print(stem, "ok" if success[stem] else "FAIL", "latched@", first_latch, "plan_close@", first_pc,
              "closing@", first_close, flush=True)
    (OUT / f"signal_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
