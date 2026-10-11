"""T44dA verifier (CPU only): per-chunk record for every CL run (72) -> CLA/chunks/<NAME>.jsonl
Inputs (read-only): CL/analysis/<NAME>.json (server_chunks[].server_file = paired server chunk npz), CL/runs/<ARM>/R/<NAME>/frames.jsonl.
Per chunk: server summary (verdict reasons, worst violating row tier/link/point, TO status, authorized-row tiers, ball, exclusion/coarse counts),
GT fruit poses from the planning row (object_poses), and FK (constraint sphere model, links gripper, same options as the CL server) of
reference_chunk vs refined_chunk rows 0..15 for the ACTIVE hand (palm + 2 fingers, URDF names):
  hand_mid = mean of the two active fingers' sphere centres ; d_mid_target = |hand_mid - GT target body pos|
  push_mm(row) = d_mid_target(refined) - d_mid_target(reference) ; disp_mm = max over active-hand spheres of |refined - reference|
  fruit gaps: signed distance (convex hull of the fruit's MuJoCo collision meshes, facet max = exact inside / lower bound outside) minus sphere radius,
  min over the active fingers' spheres, for target and for the nearest other fruit, reference vs refined, rows 0..7 (executed window).
usage: extract_chunks.py NAME [NAME ...]   (run from ${AG3S_PROJECT_ROOT}-t44cl, nice 19)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import os, sys, json, pathlib
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/CLA/p'))
import numpy as np
import fkmodel as F
from scipy.spatial import ConvexHull
import mujoco

CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL')); OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CLA/chunks'))
FRUITS = ("apple", "banana", "orange", "pear")
_m = mujoco.MjModel.from_xml_path(F.XML)

def _quat_R(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, np.float64)); return R.reshape(3, 3)

def fruit_local_verts():
    out = {}
    for b in FRUITS:
        bid = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, b); V = []
        for g in range(_m.ngeom):
            if _m.geom_bodyid[g] != bid or not (_m.geom_contype[g] or _m.geom_conaffinity[g]):
                continue
            mid = _m.geom_dataid[g]; a = _m.mesh_vertadr[mid]; n = _m.mesh_vertnum[mid]
            v = _m.mesh_vert[a:a + n].astype(np.float64)
            V.append(v @ _quat_R(_m.geom_quat[g]).T + _m.geom_pos[g])
        out[b] = np.concatenate(V)
    return out
FV = fruit_local_verts()

def hull_sd(pose, b, P):
    """signed distance of points P (N,3) to the fruit's collision-vertex convex hull at pose (pos, quat)."""
    V = FV[b] @ _quat_R(pose["quat"]).T + np.asarray(pose["pos"])
    h = ConvexHull(V); E = h.equations  # n.x + d <= 0 inside
    return (P @ E[:, :3].T + E[:, 3]).max(1)

def run(name):
    arm = name.split("_ep")[0]
    a = json.load(open(CL / "analysis" / f"{name}.json"))
    rows = [json.loads(l) for l in open((pathlib.Path(_ag3s_workspace_path('')) / a["run_dir"]) / "frames.jsonl")]
    P = {r["seq"]: r for r in rows if r["kind"] == "planning"}
    side = a["used_arm"]; s1 = "l" if side == "left" else "r"
    sc, qm = F.model("gripper")
    links = list(qm.sphere_link_names); rad = np.asarray(qm.radii)
    fing = np.array([l in (f"ee_finger_{s1}1", f"ee_finger_{s1}2") for l in links])
    hand = fing | np.array([l == f"ee_{side}" for l in links])
    tgt = a["target_fruit"]; others = [f for f in FRUITS if f != tgt]
    OUT.mkdir(parents=True, exist_ok=True)
    fo = open(OUT / f"{name}.jsonl", "w")
    for c in a.get("server_chunks") or []:
        z = np.load(c["server_file"], allow_pickle=True); s = json.loads(str(z["summary_json"]))
        p = P.get(c["seq"]) or {}
        v = s.get("verdict") or {}; cl = v.get("classification") or {}
        to = s.get("to") or {}; tm = to.get("metrics") or {}
        mvp = tm.get("max_violation_pair"); mvp = eval(mvp) if isinstance(mvp, str) else mvp
        win = s.get("esdf_stats", {}).get("window"); win = eval(win) if isinstance(win, str) else (win or {})
        tf = s.get("esdf_stats", {}).get("target_free"); tf = eval(tf) if isinstance(tf, str) else (tf or {})
        ball = (tf or {}).get("ball") or {}
        te = s.get("target_exclusion") or {}; co = te.get("coarse") or {}
        g = s.get("grasp") or {}; ta = g.get("tip_alignment") or {}
        rec = {"name": name, "arm": arm, "seq": c["seq"], "t": c.get("t"), "hold": bool(c.get("hold")),
               "gate_kinds": (p.get("gate") or {}).get("hold_kinds"), "hold_reason": p.get("hold_reason"),
               "reasons": [{"kind": r.get("kind"), "action": r.get("action"), "detail": (r.get("detail") or "")[:300]} for r in (v.get("reasons") or [])],
               "cls": {k: cl.get(k) for k in ("n_violating", "n_allowed", "n_unresolved", "n_collision", "collision_by_why", "residual_violation_m")},
               "collision_worst": cl.get("collision_worst"), "unresolved_worst": cl.get("unresolved_worst"), "allowed_worst": cl.get("allowed_worst"),
               "follow_to": v.get("follow_to"),
               "to_status": to.get("status"), "to_returned": tm.get("returned"), "to_clearance_m": tm.get("clearance_m"),
               "mvp": {k: (mvp or {}).get(k) for k in ("block", "clearance_m", "step", "query", "link", "tier", "point_m", "authorized_rows")} if mvp else None,
               "win_by_tier": (win.get("authorized_query") or {}).get("by_tier"), "win_by_tier_now": (win.get("authorized_query") or {}).get("by_tier_now"),
               "grasp_state": g.get("state"), "attached": bool(g.get("attached")), "tip_now_d3_mm": (ta.get("now") or {}).get("d3_mm"),
               "ball_centre": ball.get("centre_m"), "ball_r_mm": ball.get("radius_mm"), "ball_fit_r_mm": (ball.get("fit") or {}).get("radius_mm"),
               "support_z": ball.get("support_plane_z_m"),
               "n_kept_nontarget": te.get("n_kept_nontarget"), "n_ambiguous": te.get("n_ambiguous"), "n_unowned_removed": te.get("n_unowned_removed"),
               "n_removed": te.get("n_removed"), "n_seeds_in_ball": te.get("n_seeds_in_ball"), "excl_mode": te.get("mode"),
               "coarse": {k: co.get(k) for k in ("mode", "n_seeds_in_ball", "n_removed", "n_shared_kept", "n_sign_candidates")} if co else None,
               "coarse_plane_band": co.get("plane_band"), "coarse_sign": co.get("sign"),
               "split_n_split": ((s.get("manipulated") or {}).get("split") or {}).get("n_split"),
               "perm": {k: (s.get("target_permission") or {}).get(k) for k in ("policy", "authorized_links", "n_rows", "n_rows_excluded", "excluded_links", "held")},
               "lookahead": tm.get("lookahead"), "clearance_min_mm": s.get("clearance_min_mm"),
               "object_poses": p.get("object_poses"), "subtask": (s.get("subtask") or {}).get("label")}
        ref = np.asarray(z["reference_chunk"], np.float64); fin = np.asarray(z["refined_chunk"], np.float64)
        qpos = p.get("qpos")
        if qpos is not None:
            pr = F.params_from_qpos(sc, qm, qpos)
            QR = F.chunk_q(sc, qm, qpos, ref[:16]); QF = F.chunk_q(sc, qm, qpos, fin[:16])
            SR = np.stack([F.spheres(qm, q, pr) for q in QR]); SF = np.stack([F.spheres(qm, q, pr) for q in QF])
            op = p["object_poses"]; tp = np.asarray(op[tgt]["pos"])
            mR = SR[:, fing].mean(1); mF = SF[:, fing].mean(1)
            dR = np.linalg.norm(mR - tp, axis=1); dF = np.linalg.norm(mF - tp, axis=1)
            disp = np.linalg.norm(SF[:, hand] - SR[:, hand], axis=2).max(1)
            def gap(S, b):
                X = S[:8][:, fing].reshape(-1, 3); r = np.tile(rad[fing], 8)
                return float((hull_sd(op[b], b, X) - r).min())
            rec["fk"] = {"d_mid_target_ref_mm": (dR * 1e3).round(2).tolist(), "d_mid_target_fin_mm": (dF * 1e3).round(2).tolist(),
                         "push_mm": ((dF - dR) * 1e3).round(2).tolist(), "disp_hand_mm": (disp * 1e3).round(2).tolist(),
                         "dq_max_rad_0_7": float(np.abs(fin[:8, list(range(7)) + list(range(8, 15))] - ref[:8, list(range(7)) + list(range(8, 15))]).max()),
                         "gap_target_ref_mm": gap(SR, tgt) * 1e3, "gap_target_fin_mm": gap(SF, tgt) * 1e3,
                         "gap_other_ref_mm": min(gap(SR, b) for b in others) * 1e3, "gap_other_fin_mm": min(gap(SF, b) for b in others) * 1e3,
                         "other_nearest": min(others, key=lambda b: gap(SF, b)),
                         "mid_ref_row7": mR[7].round(4).tolist(), "mid_fin_row7": mF[7].round(4).tolist(), "target_pos": tp.round(4).tolist(),
                         "grip_ref_rows0_7": ref[:8, 7 if side == "left" else 15].round(3).tolist()}
        fo.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    fo.close()

if __name__ == "__main__":
    for n in sys.argv[1:]:
        if (OUT / f"{n}.jsonl").exists():
            continue
        run(n); print("done", n, flush=True)
