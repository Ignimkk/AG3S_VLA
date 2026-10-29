"""T33 closed-loop records: GT apple points left in the recorded raw_cloud (fused cloud after the server's self-filter incl. H1 held
spheres) for every held chunk (grasp.attach chunk .. chunk before detach). GT apple point rule as offline_a_held_points.py (<= 5 mm,
closer to apple than to any other body / left finger; candidates within 90 mm of the GT apple centre). in_hand = GT apple offset
in the palm... not recomputed here: 'apple_to_left_finger_mid_mm' (planning-row MuJoCo ee_finger_l1/l2 midpoint) is reported instead."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T33'))
m, d, probe, obj, fing = P.build()
B = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
F1, F2 = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("ee_finger_l1", "ee_finger_l2"))
out = {"definitions": __doc__, "runs": {}}
for cond in ("E3a", "E3b"):
    A = json.loads((V / "p" / f"{cond}_analysis.json").read_text())["runs"]
    for run, r in A.items():
        ch = [c for c in r["chunks"] if c.get("server_chunk")]
        ia = next((i for i, c in enumerate(ch) if (c.get("grasp_latch") or {}).get("attach")), None)
        if ia is None:
            continue
        idet = next((i for i, c in enumerate(ch) if i > ia and (c.get("grasp_latch") or {}).get("detach")), len(ch))
        rows = {int(x["seq"]): x for x in (json.loads(l) for l in open(V / f"{cond}_rec/{run}/frames.jsonl") if l.strip()) if x.get("kind") == "planning"}
        res = []
        for c in ch[ia:idet]:
            z = np.load(V / f"{cond}_rec" / c["server_chunk"], allow_pickle=False)
            cloud = np.asarray(z["raw_cloud"], np.float64); row = rows[c["seq"]]
            P.set_q(m, d, row["qpos"]); Rb = d.xmat[B].reshape(3, 3).copy(); pb = d.xpos[B].copy()
            fm = 0.5 * (d.xpos[F1] + d.xpos[F2]).copy()
            cw = cloud @ Rb.T + pb; ap = np.asarray(row["object_poses"]["apple"]["pos"])
            n = 0
            for j in np.flatnonzero(np.linalg.norm(cw - ap, axis=1) <= 0.09):
                body, cat, per = P.nearest(m, d, probe, obj, cw[j]); fd = min(P.gdist(m, d, probe, g) for g in fing)
                if body == "apple" and per["apple"] <= 0.005 and per["apple"] < fd:
                    n += 1
            P.set_q(m, d, row["qpos"])
            res.append({"t": c["t"], "phase": c["phase"], "n_gt_apple_pts_5mm": n, "apple_to_left_finger_mid_mm": float(1000 * np.linalg.norm(ap - fm)), "apple_dz_mm": c["apple_dz_mm"]})
        out["runs"][run] = res
        print(run, [(x["t"], x["n_gt_apple_pts_5mm"], round(x["apple_to_left_finger_mid_mm"])) for x in res], flush=True)
(V / "off/a/held_points_t33.json").write_text(json.dumps(out, indent=1))
