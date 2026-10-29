"""Which server decision changed a VLA-success episode? Per-request trace of one AG3S run (CPU only, read only).

  PYTHONPATH=.:integration/rby1/manipulation/src MUJOCO_GL=osmesa .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_regression_trace --replay <replay json> [...] --bodies <scene_bodies.json> \
      [--obstacles <obstacle config json>] --out <json>

The VLA-alone run and the AG3S run share the episode seed, so their first chunk is the same (smoke gate); they part at the first request where
the server executes something other than the VLA's reference chunk: a TrajOpt edit (refined rows), a HOLD, or a truncated execution.
Per server request (aligned replay, `ifg_10ep_handreach` alignment check) this records:
  to_push_mm      max TCP displacement refined vs reference over the executed rows (FK, teleop keyframe)
  hold            exec_feedback n_hold > 0, hold_kind / hold_reason
  verdict         verdict.reasons kinds; to.status
  worst           to.metrics.max_violation_pair: link, tier, server obstacle label, clearance (mm), query point
  worst_body      the GT body nearest to the occupied voxel closest to the worst query point (ESDF < 0 within 6 cm, in the layer of the
                  worst row's tier: target_free -> the target-free layer, fine, coarse):
                  target / <other fruit> / crate / table / obstacle:<slot> / unknown, and the distance (mm) to it
  grasp           latch state, attached
`first_change` = the first request with to_push_mm >= PUSH_MM or a HOLD — the earliest place the AG3S run left the VLA's plan.
"""
import argparse, glob, json, pathlib, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

PUSH_MM = 5.0
FRUIT_R = 0.045          # m: fruit radius allowance for body attribution
TABLE_Z = 0.82


LAYER = {"target_free": "esdf_free_", "fine": "esdf_", "coarse": "esdf_coarse_"}


def nearest_occupied(c, p, tier, r=0.06):
    """Nearest voxel < 0 to `p` in the layer the worst row was answered by (`tier`: the target-free layer for target_free rows —
    the target removed — else fine / coarse); an unknown tier searches fine and coarse."""
    best = None
    for pre in ([LAYER[tier]] if tier in LAYER else ["esdf_", "esdf_coarse_"]):
        if pre + "distance" not in c:
            continue
        o = c[pre + "origin"]; v = float(c[pre + "voxel_size"]); d = c[pre + "distance"]
        lo = np.maximum(np.floor((p - r - o) / v).astype(int), 0); hi = np.minimum(np.ceil((p + r - o) / v).astype(int) + 1, d.shape)
        ijk = np.argwhere(d[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] < 0) + lo
        if not len(ijk):
            continue
        ctr = o + (ijk + 0.5) * v; dist = np.linalg.norm(ctr - p, axis=1); i = int(np.argmin(dist))
        if best is None or dist[i] < best[1]:
            best = (ctr[i], float(dist[i]), pre.rstrip("_"))
    return best


def attribute(q, gt_target, bodies, target_name, crate, obstacles):
    cands = [("target", float(np.linalg.norm(q - gt_target)) - FRUIT_R)]
    for name, pos in bodies.items():
        if name in ("crate", target_name):
            continue
        cands.append((name, float(np.linalg.norm(q - np.asarray(pos))) - FRUIT_R))
    if crate is not None:
        dxy = np.abs(q[:2] - crate[:2]) - np.array([0.13, 0.17])   # crate half extents (approx.)
        cands.append(("crate", float(np.linalg.norm(np.maximum(dxy, 0.0)))))
    for ob in obstacles:
        p0 = np.asarray(ob["position"]); dxy = float(np.linalg.norm(q[:2] - p0[:2])) - float(ob["radius"])
        if TABLE_Z <= q[2] <= p0[2] + float(ob["height"]) + 0.01:
            cands.append((f"obstacle:{ob['slot']}", max(dxy, 0.0)))
    cands.append(("table", abs(float(q[2]) - TABLE_Z)))
    name, d = min(cands, key=lambda x: x[1])
    return (name if d <= 0.03 else "unknown"), round(d * 1000, 1)


def trace(jf, bodies_all, obst_cfg, kin, base_q, arm_targets):
    j = json.load(open(jf)); z = np.load(jf[:-5] + ".npz")
    al = j["server_chunk_alignment"]["block"]; chk = [c.get("dist_m") for c in j["server_chunk_alignment"].get("checks", []) if c.get("plan_index") == 0]
    out = {"replay": jf, "episode": j["episode"], "seed": j["seed"], "side": j["side"], "target": j["target_fruit"], "profile": j.get("profile")}
    if not al.get("found") or al.get("offset") is None or not chk or chk[0] is None or chk[0] > 0.05:
        out["aligned"] = False; return out
    out["aligned"] = True
    bodies = bodies_all.get(str(j["episode"]), {})
    obstacles = []
    if obst_cfg is not None and j.get("profile") in (obst_cfg.get("profiles") or {}):
        obstacles = obst_cfg["profiles"][j["profile"]]["obstacles"]
    side = j["side"]; pi = z["plan_index"]; rows = []
    for k in range(int(pi.max()) + 1):
        idx = np.flatnonzero(pi == k)
        if not len(idx):
            continue
        c = np.load(f"{al['chunk_dir']}/chunk_{k + al['offset']:05d}.npz", allow_pickle=True); s = json.loads(str(c["summary_json"]))
        ef = s.get("exec_feedback") or {}; g = s.get("grasp") or {}; to = s.get("to") or {}
        n_exec = int((s.get("execution_path") or {}).get("execution_length") or 8)
        push = max(float(np.linalg.norm(kin.tcp(base_q, side, arm_targets("rby1_16d", y_)[side]) - kin.tcp(base_q, side, arm_targets("rby1_16d", x_)[side])))
                   for x_, y_ in zip(c["reference_chunk"][:n_exec], c["refined_chunk"][:n_exec])) * 1000
        mv = (to.get("metrics") or {}).get("max_violation_pair")
        mv = eval(mv) if isinstance(mv, str) else (mv or {})   # recorded as a Python repr
        gt = z["target_pos"][idx[0]]
        row = {"k": k, "t": int(idx[0]), "to_push_mm": round(push, 1), "hold": int((ef.get("n_hold") or 0) > 0),
               "hold_kind": ef.get("hold_kind"), "hold_reason": ef.get("hold_reason"), "to_status": to.get("status"),
               "verdict": [r.get("kind") for r in ((s.get("verdict") or {}).get("reasons") or [])],
               "worst_link": mv.get("link"), "worst_tier": mv.get("tier"), "worst_label": mv.get("obstacle"),
               "worst_clearance_mm": None if mv.get("clearance_m") is None else round(float(mv["clearance_m"]) * 1000, 1),
               "reference_clearance_min_mm": None if s.get("clearance_min_mm") is None else round(float(s["clearance_min_mm"]), 1),
               "grasp": g.get("state"), "attached": g.get("attached") is not None,
               "lift_mm": round(float(gt[2] - z["target_pos"][0, 2]) * 1000, 1)}
        if mv.get("point_m") is not None:
            p = np.asarray(mv["point_m"], np.float64); occ = nearest_occupied(c, p, mv.get("tier"))
            if occ is not None:
                crate = z["crate_pos"][idx[0]] if "crate_pos" in z.files else None
                row["worst_body"], row["worst_body_mm"] = attribute(occ[0], gt, bodies, j["target_fruit"], crate, obstacles)
                row["worst_voxel_mm"] = round(occ[1] * 1000, 1); row["worst_voxel_tier"] = occ[2]
        rows.append(row)
    out["rows"] = rows
    out["first_change"] = next((r for r in rows if r["to_push_mm"] >= PUSH_MM or r["hold"]), None)
    out["changes"] = [r for r in rows if r["to_push_mm"] >= PUSH_MM or r["hold"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", nargs="+", required=True); ap.add_argument("--bodies", nargs="+", required=True)
    ap.add_argument("--obstacles", nargs="*", default=[]); ap.add_argument("--model-xml", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    import mujoco
    from integration.rby1.bringup.avoidance_arrow import TcpKinematics, arm_targets
    m = mujoco.MjModel.from_xml_path(a.model_xml)
    base_q = np.array(m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")]); kin = TcpKinematics(m)
    bodies = {}
    for b in a.bodies:
        bodies.update(json.load(open(b)))
    out = []
    for jf in sorted(x for pat in a.replay for x in glob.glob(pat)):
        ep = json.load(open(jf))["episode"]
        oc = None
        for o in a.obstacles:
            cand = json.load(open(o))
            if any(str(ep) in name for name in cand.get("profiles", {})):
                oc = cand
        out.append(trace(jf, bodies, oc, kin, base_q, arm_targets))
    a.out.write_text(json.dumps({"definitions": __doc__, "push_mm": PUSH_MM, "runs": out}, indent=1, default=float))
    for r in out:
        if not r.get("aligned"):
            print(r["episode"], "not aligned"); continue
        fc = r["first_change"]
        print(f"ep{r['episode']} {r['side']} {r['target']}  changes {len(r['changes'])}  first: "
              + ("none" if fc is None else
                 f"k{fc['k']} push {fc['to_push_mm']} mm hold {fc['hold']} {fc['hold_reason'] or ''} | {fc['worst_link']} {fc['worst_tier']} "
                 f"label={fc['worst_label']} clr {fc['worst_clearance_mm']} | body {fc.get('worst_body')} ({fc.get('worst_body_mm')} mm) grasp {fc['grasp']}"))


if __name__ == "__main__":
    main()
