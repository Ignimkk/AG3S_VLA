"""T43 Z6 (2) offline — which recorded HOLD chunks the `follow_to` rule would have executed, and the
TRUE crate distance of each such chunk's plan (implementer, CPU only, no new runs).

usage: follow_offline.py OUT_JSON KEY [KEY ...]      (KEY as in outputs/verify/T43/Y/raw/server_chunks.json)

Per HOLD chunk (server_chunks `hold`), from the server record only (no MuJoCo in the rule):
  hold kinds     summary.verdict.reasons with action hold. Rule step 1: must be exactly {collision}.
  robot rows     summary.verdict.classification.collision_by_why. Keys other than 'attached_point' = robot
                 rows violated; their margins (T41a obstacle margin 10 mm) are not rebuilt here, so such a
                 chunk is reported as 'robot_rows' (not counted as an execution).
  held rows      rebuilt: field = recorded fine/coarse grids (`fixed_input_ablation.build_field`, the server's
                 `CuroboEsdfField`), held query spheres = summary grasp.attached.held.spheres_parent_frame (the
                 TO's spheres this frame) on the palm pose of the constraint model (`link_pose`) at each refined
                 plan step (refined_chunk rows 0-7 = the TO window) and at q_now (planning-row qpos).
                 gap = d - r. Margin = 20 mm on destination-labelled held rows, else esdf_margin 0 (V5).
                 The label layer is NOT recorded, so the rule is evaluated without it where it can be:
                 (i) exact and label-free: a held row with gap < 0 is violated whatever its margin -> min_gap >= 0.
                 plan_min = the recorded worst row's clearance (server value, exact; it is a held destination row
                 in every chunk here). fidelity_mm = rebuilt gap - 20 mm - recorded clearance at that row.
                 Violated rows, label-bounded: a row whose clearance as a destination row (gap - 20 mm) would be
                 below the recorded worst cannot carry the label (it would have been the worst row) -> margin 0,
                 violated only if gap < 0. Certain rows = gap < 0 plus the recorded worst row; possible rows =
                 those plus every row that may carry the label and would then be violated.
                 (ii) bracketed over the current state (q_now) of those spheres: now_low = min over the possible
                 spheres of gap_now - 20 mm (all labelled), now_high = min over the certain spheres of gap_now
                 (none labelled). plan_min < now_low -> 'worsens' (certain); plan_min >= now_high -> follow
                 (certain); between -> 'ambiguous_label'.
                 (iii) max |refined - reference| over rows 0-7, arm columns > 1e-6 rad.
                 The streak limit is not applied: after the first execution the closed loop diverges, so only
                 the first eligible chunk of a HOLD stretch is a faithful counterfactual.
TRUE crate distance (offline only, the Y mesh-vertex method, y_metrics.Geo): apple collision-mesh vertices at the
chunk's pre-state (Y replay npz) moved rigidly with the palm along the refined plan (palm motion in the base frame
from the constraint model, base pose from MuJoCo FK of the planning-row qpos), point-to-OBB signed distance to the
13 crate boxes, min over vertices; per step and min. 'apple_in_hand' = chunk t < Y release_t."""
from common.workspace import workspace_path as _ag3s_workspace_path
import ast, json, sys, pathlib, functools, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
from AG3S.trajopt.types import ChunkLayout
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/Y/p'))
import y_metrics as YM  # noqa: E402

SC = _ag3s_workspace_path('outputs/verify/T43/Y/raw/server_chunks.json')
AN = _ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')
YV = _ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43Y.verify.json')
REPLAY = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Y/replay'))
H, TOL, DEST_M, VOX = 8, 1e-4, 0.020, 0.005
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x  # noqa: E731


class _Args:
    links = None; sphere_spacing = None; max_spheres_per_capsule = None; capsule_radius_scale = None
    max_sphere_radius = None; esdf_margin = None; target_field_policy = None; phase = None
    manipulators = None; model_xml = None; hand = None; allow_uncertified = False; rows_per_step = None


@functools.lru_cache(maxsize=8)
def setup(server_log, manifest_path):
    man = json.loads(pathlib.Path(manifest_path).read_text())
    man["scene"]["model_xml"] = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
    log = FIA.parse_server_log(server_log)
    st = FIA.resolve_settings(_Args, log, man)
    robot, qadr, _ = FIA.build_robot(st)
    _, _, _, maxd, unk = FIA.contact_policy(st, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
    return dict(st=st, robot=robot, qadr=qadr, layout=ChunkLayout.rby1(robot.joint_names), outside=maxd if unk == "free" else -maxd)


def aabb_dist(p, lo, hi):
    return np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=-1)


def held_rows(G, field, held, parent, Q, dest, plane=None):
    """(gap[S, n], clearance[S, n], label[S, n]) at the full-q poses Q (S, nq). A row whose nearest
    surface is the support plane (|d - height over the plane| <= 2 fine voxels, the T41a support rule)
    is not a destination row."""
    loc = np.array([c["centre_m"] for c in held], float); rad = np.array([c["radius_mm"] / 1e3 for c in held])
    pts = []
    for q in Q:
        T = np.asarray(G["robot"].link_pose(q, parent), float)
        pts.append(loc @ T[:3, :3].T + T[:3, 3])
    pts = np.asarray(pts)
    d = np.asarray(field.distance(pts.reshape(-1, 3)), float).reshape(pts.shape[:2])
    gap = d - rad[None]
    lab = np.zeros_like(gap, bool)
    if dest is not None:
        lab = aabb_dist(pts, dest[0], dest[1]) <= d + VOX
        if plane is not None:
            lab &= ~(np.abs(d - (pts @ plane[0] - plane[1])) <= 2 * VOX)
    return gap, gap - np.where(lab, DEST_M, 0.0), lab, pts


def main():
    out_path, keys = sys.argv[1], sys.argv[2:]
    S = json.load(open(SC))["runs"]; A = json.load(open(AN))["runs"]
    Yt = json.load(open(YV))["numbers"]["Y1"]["timeline"]
    geo = YM.Geo(); res = {"definitions": __doc__, "runs": {}}
    for key in keys:
        r = S[key]; a = A[key]; rd = pathlib.Path(a["run_dir"])
        first = pathlib.Path(r["server_segment"]["first"]); i0 = int(first.stem.split("_")[1])
        G = setup(str(first.parent.parent.parent / "serve_safe.log"), str(rd / "manifest.json"))
        layout = G["layout"]
        plan = {p["seq"]: p for p in (json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()) if p.get("kind") == "planning"}
        rz = np.load(REPLAY / f"{key.replace('/', '_')}.npz")
        release_t = (Yt.get(key) or {}).get("release_t")
        rows = []
        z_radii = {}
        for n, c in enumerate(r["chunks"]):
            if not c["hold"]:
                continue
            with np.load(first.parent / f"chunk_{i0 + n:05d}.npz", allow_pickle=True) as z:
                assert int(z["t_step"]) == c["seq"]
                s = json.loads(str(z["summary_json"]))
                field = FIA.build_field(z, G["outside"])
                refc = np.asarray(z["refined_chunk"], float); refr = np.asarray(z["reference_chunk"], float)
                z_radii[key] = np.asarray(z["sphere_radii"])
            rec = {"t": c["t"], "seq": c["seq"]}
            reasons = (s.get("verdict") or {}).get("reasons") or []
            hold_kinds = sorted({x["kind"] for x in reasons if x.get("action") == "hold"})
            cls = (s.get("verdict") or {}).get("classification") or {}
            rec["hold_kinds"] = hold_kinds; rec["collision_by_why"] = cls.get("collision_by_why")
            mv = lit((s.get("to") or {}).get("metrics", {}).get("max_violation_pair")) or {}
            rec["recorded_worst"] = {k: mv.get(k) for k in ("link", "step", "clearance_m", "obstacle", "tier")}
            arm = [k for k in range(refc.shape[1]) if k not in (7, 15)]
            rec["refined_minus_reference_rad"] = float(np.abs(refc[:H, arm] - refr[:H, arm]).max())
            att = (s.get("grasp") or {}).get("attached") or {}
            held = ((att.get("held") or {}).get("spheres_parent_frame")) or []
            qn = FIA.q_now_from_qpos(np.asarray(plan[c["seq"]]["qpos"], float), G["qadr"])
            traj = layout.chunk_to_trajectory(refc)[:, :H]
            Q = layout.full_q(traj, qn).T                                  # (H, nq)
            dest = s.get("destination") or {}
            box = (np.asarray(dest["aabb_min"], float), np.asarray(dest["aabb_max"], float)) if dest.get("aabb_min") else None
            pl = ((att.get("held") or {}).get("support") or {}).get("plane")
            plane = None if not pl else (np.asarray(pl["normal"], float), float(pl["offset_m"]))
            why = None
            if hold_kinds != ["collision"]:
                why = "hold_kinds"
            elif set((cls.get("collision_by_why") or {})) - {"attached_point"}:
                why = "robot_rows"
            elif not held:
                why = "no_held"
            if held:
                gap, _, _, pts = held_rows(G, field, held, att["parent_link"], Q, None)
                gap0, _, _, _ = held_rows(G, field, held, att["parent_link"], qn[None], None)
                nr = int(np.asarray(z_radii[key]).shape[0]) if key in z_radii else 122
                qw = int(mv["query"]) - nr if mv.get("query") is not None else None
                rec["min_gap_m"] = float(gap.min())                       # label-free: gap < 0 is violated anyway
                rec["plan_min_m"] = None if mv.get("clearance_m") is None else float(mv["clearance_m"])
                if qw is not None and 0 <= qw < gap.shape[1] and rec["plan_min_m"] is not None:
                    worst = rec["plan_min_m"]
                    rec["fidelity_mm"] = round(1e3 * (float(gap[int(mv["step"]), qw]) - DEST_M - worst), 3)
                    # a row whose clearance as a destination row would be below the recorded worst cannot be
                    # destination-labelled (it would have been the worst row): margin 0, violated only if gap < 0
                    can_dest = gap - DEST_M >= worst - TOL
                    viol_certain = (gap < -TOL)
                    viol_certain[int(mv["step"]), qw] = True
                    viol_possible = viol_certain | (can_dest & (gap - DEST_M < -TOL))
                    qc = sorted({int(j) for j in np.argwhere(viol_certain)[:, 1]})
                    qp = sorted({int(j) for j in np.argwhere(viol_possible)[:, 1]})
                    rec["violated_spheres_certain"], rec["violated_spheres_possible"] = qc, qp
                    rec["n_rows_possible"] = int(viol_possible.sum())
                    rec["recorded_n_collision"] = cls.get("n_collision")
                    rec["gap_now_mm"] = [round(1e3 * float(v), 2) for v in gap0[0]]
                    # current state of those spheres: destination margin (low) or none (high) at q_now
                    rec["now_low_m"] = float(min(gap0[0, j] for j in qp)) - DEST_M
                    rec["now_high_m"] = float(min(gap0[0, j] for j in qc))
                if why is None:
                    if rec["min_gap_m"] < 0:
                        why = "penetration"
                    elif rec.get("now_low_m") is None:
                        why = "no_worst_row"
                    elif rec["refined_minus_reference_rad"] <= 1e-6:
                        why = "refined_is_reference"
                    elif rec["plan_min_m"] < rec["now_low_m"]:
                        why = "worsens"
                    elif rec["plan_min_m"] < rec["now_high_m"]:
                        why = "ambiguous_label"
            rec["follow_to"] = why is None; rec["refused"] = why
            # TRUE crate distance along the plan (offline only)
            i = int(np.flatnonzero(rz["t"] == c["t"])[0])
            m, d = geo_model()
            d.qpos[:] = plan[c["seq"]]["qpos"]; mujoco.mj_kinematics(m, d)
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
            Bw = np.eye(4); Bw[:3, :3] = d.xmat[bid].reshape(3, 3); Bw[:3, 3] = d.xpos[bid]
            P0 = np.asarray(G["robot"].link_pose(qn, att.get("parent_link") or "ee_left"), float)
            Ra, Rc = YM.qmat(rz["apple_quat"][i]), YM.qmat(rz["crate_quat"][i])
            V = rz["apple_pos"][i] + Ra @ geo.a_pos + (Ra @ geo.a_R @ geo.av.T).T
            per = []
            for k in range(H):
                Pk = np.asarray(G["robot"].link_pose(Q[k], att.get("parent_link") or "ee_left"), float)
                M = Bw @ Pk @ np.linalg.inv(P0) @ np.linalg.inv(Bw)
                Vk = V @ M[:3, :3].T + M[:3, 3]
                per.append(1e3 * crate_sd(geo, Vk, rz["crate_pos"][i], Rc))
            rec["true_crate_mm_now"] = round(1e3 * crate_sd(geo, V, rz["crate_pos"][i], Rc), 2)
            rec["true_crate_mm_plan"] = [round(v, 2) for v in per]; rec["true_crate_mm_plan_min"] = round(min(per), 2)
            rec["apple_in_hand"] = release_t is None or c["t"] < release_t
            rows.append(rec)
            mm = lambda k: None if rec.get(k) is None else round(1e3 * rec[k], 1)  # noqa: E731
            print(key, c["t"], "follow" if rec["follow_to"] else rec["refused"], "fid", rec.get("fidelity_mm"),
                  "gap", mm("min_gap_m"), "plan", mm("plan_min_m"), "now", mm("now_low_m"), mm("now_high_m"), "Qv", rec.get("violated_spheres_certain"), rec.get("violated_spheres_possible"),
                  "true", rec["true_crate_mm_now"], rec["true_crate_mm_plan_min"], "inhand", rec["apple_in_hand"], flush=True)
        res["runs"][key] = {"release_t": release_t, "chunks": rows}
    pathlib.Path(out_path).write_text(json.dumps(res, indent=1, default=float))


_GM = {}


def geo_model():
    if "m" not in _GM:
        _GM["m"] = mujoco.MjModel.from_xml_path(YM.XML); _GM["d"] = mujoco.MjData(_GM["m"])
    return _GM["m"], _GM["d"]


def crate_sd(geo, V, cp, Rc):
    Vc = (Rc.T @ (V - cp).T).T
    best = np.inf
    for _, gp, gR, h in geo.boxes:
        q = (gR.T @ (Vc - gp).T).T; dv = np.abs(q) - h
        sd = np.linalg.norm(np.maximum(dv, 0), axis=1) + np.minimum(dv.max(axis=1), 0)
        best = min(best, float(sd.min()))
    return best


if __name__ == "__main__":
    main()
