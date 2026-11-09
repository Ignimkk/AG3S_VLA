"""(.venv-ag3s, CPU) User question 2026-11-09 02:1x (dev-4a): "the hand fails to grasp / fails to place -- does AG3S treat
the target (grasp) or the plate (place) as an obstacle?"

Per call of a closed-loop run (records `--record-constraints full`): latch phase, whether the executed action changed,
HOLD, pass-through reason, and the GT owner of the row that refused the pass-through:
  worst obstacle-object row (`to.passthrough_clear.obstacle_rows.worst`, robot sphere centre `point_m`, base frame)
  -> world (`T_world_base` of that obs) -> nearest non-robot true visual surface (dev-38 quantify roles: target /
  destination / distractor / other object / fixture) + its distance.
Phase split: approach (searching / latched), closing, held-far (hand > 12 cm from destination in x-y), held-near (place).
GT is used only for scoring. Run: SCENE_DIR=<scene dir> [SAFELIBERO_PATCH=1] python pe_rowowner.py '<run glob>' <tag>
"""
import collections
import json
import pathlib
import sys

import mujoco
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-08_PE_경로끝점_identity")
import pe_signal as PS  # noqa: E402

Q = PS.Q
OUT = pathlib.Path(__file__).resolve().parent


def phase(state, hand_w, dest_w):
    if state in ("searching", "latched"):
        return "approach"
    if state == "closing":
        return "closing"
    if state == "held":
        if dest_w is not None and np.linalg.norm(hand_w[:2] - dest_w[:2]) <= 0.12:
            return "held_place"
        return "held_carry"
    return "other"


def main():
    runs = [p for p in sorted(Q.E.RUNS.glob(sys.argv[1])) if not p.name.startswith("INVALID")
            and (p / "episodes.jsonl").is_file()]
    tag = sys.argv[2]
    res = {}
    for run in runs:
        eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
        order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
        succ = {o: bool(e["official_success"]) for o, e in zip(order, eps)}
        for stem in order:
            if not (Q.SCENE / f"{stem}_scene.xml").exists():
                print(stem, "no scene", flush=True)
                continue
            info, m, rg, cache, stc = PS.scene_for(stem)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            S1, G1 = stc.world(d)                       # welded geoms: built once per episode
            R1 = rg[G1]
            k1 = R1 != "robot"
            st_tree, R1k = cKDTree(S1[k1]), R1[k1]
            hand_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
            geom = np.load(run / "episodes" / f"{stem}_geom.npz")
            grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
            calls = [json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")]
            rows = []
            for c in calls:
                g = c.get("grasp") or {}
                st = g.get("state")
                pc = (c.get("to") or {}).get("passthrough_clear") or {}
                changed = float(c.get("max_abs_action_change_executed") or 0) > 1e-6
                i = grow.get(int(c["obs_index"]))
                if i is None:
                    continue
                d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]
                mujoco.mj_forward(m, d)
                S2, G2 = cache.world(d)
                R2 = rg[G2]
                k2 = R2 != "robot"
                tree = Q.SplitTree(st_tree, R1k, S2[k2], R2[k2])
                Rk = tree.roles
                S = np.concatenate([S1, S2])
                R = np.concatenate([R1, R2])
                dest_pts = S[R == "destination"]
                dest_w = dest_pts.mean(0) if len(dest_pts) else None
                hand_w = d.xpos[hand_bid].copy() if hand_bid >= 0 else np.zeros(3)
                tgt = S[R == "target"]
                row = {"seq": c["seq"], "state": st, "phase": phase(st, hand_w, dest_w), "changed": changed,
                       "hold": bool(c.get("hold")), "reason": pc.get("reason"),
                       "hand_target_mm": float(np.linalg.norm(tgt - hand_w, axis=1).min() * 1e3) if len(tgt) else None}
                w = ((pc.get("obstacle_rows") or {}).get("worst") or {})
                if w.get("point_m") is not None:
                    Tw = geom["T_world_base"][i]
                    p = Tw[:3, :3] @ np.asarray(w["point_m"], float) + Tw[:3, 3]
                    dd, j = tree.query(p[None])
                    dd, j = float(dd[0]), int(j[0])
                    row.update(worst_link=w.get("link"), worst_clear_mm=round(float(w["clearance_m"]) * 1e3, 1),
                               owner=str(Rk[j]), owner_mm=round(float(dd) * 1e3, 1))
                rows.append(row)
            res[f"{run.name[:60]}/{stem}"] = {"success": succ[stem], "rows": rows}
            # per-episode summary: refused calls (changed or HOLD) by phase x owner
            cnt = collections.Counter()
            for r in rows:
                if r["changed"] or r["hold"]:
                    who = r.get("owner") if r["reason"] == "obstacle" else f"[{r['reason']}]"
                    cnt[(r["phase"], who)] += 1
            print(f"{'ok ' if succ[stem] else 'FAIL'} {run.name[16:60]} {stem.split('spatial_')[-1]}: "
                  + ", ".join(f"{k[0]}:{k[1]}={v}" for k, v in sorted(cnt.items())), flush=True)
    (OUT / f"rowowner_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
