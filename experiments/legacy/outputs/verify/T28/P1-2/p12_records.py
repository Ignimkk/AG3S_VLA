"""T28 P1-2 — per-chunk diagnosis of existing closed-loop records T16 / T14 (ep 1807).

Pairing: server `chunk_k.npz` <-> local planning row seq k+1, t_step = 8k (T14.audit §1).
Frames: server points (max_violation_pair.point_m, target_centroid) are in the robot `base` body frame;
converted to MuJoCo world with the base body pose from FK of that planning row's qpos.
Nearest real object: MuJoCo `mj_geomDistance` from a 0.1 mm probe sphere at the point to every
collision geom of non-robot bodies (apple, crate, table, banana, orange, pear, shelf, office, floor);
mesh geoms are evaluated as their convex hull (MuJoCo collision convention). Robot geoms excluded.
Finger->apple: min mj_geomDistance between ee_finger_l1/l2 collision geoms and apple_geom at that row's qpos.
Grasp attempt: first planning row whose policy chunk (actions_reference) has left-gripper cmd (col 7,
1=open) < 0.5 anywhere in rows 0..7 (executed window).
HOLD kind (legacy gate, these runs predate T23): from planning row ipc/safe/verdict:
  ipc in {timeout, stale, error,...} -> that string; safe False & geometry_certified False -> 'uncertified';
  safe False & trajopt_status violated -> 'violated'.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import mujoco

XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
RUNS = {"t16": pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260928_t16')),
        "t14": pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260928_t14'))}
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/P1-2'))
CATEGORY = {"apple": "apple", "crate": "crate", "table": "table", "banana": "other_fruit",
            "orange": "other_fruit", "pear": "other_fruit", "shelf": "other", "office": "other",
            "world": "other"}


def build():
    spec = mujoco.MjSpec.from_file(XML)
    b = spec.worldbody.add_body(name="t28_probe", mocap=True)
    b.add_geom(name="t28_probe_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1e-4, 0, 0],
               contype=0, conaffinity=0)
    m = spec.compile(); d = mujoco.MjData(m)
    probe = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "t28_probe_geom")
    global PROBE_MOCAP
    PROBE_MOCAP = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "t28_probe")])
    obj_geoms = {}
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        root = m.body(m.body_rootid[m.geom_bodyid[g]]).name
        body = m.body(m.geom_bodyid[g]).name
        if body in CATEGORY:
            obj_geoms.setdefault(body, []).append(g)
    fing = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ("ee_finger_l1", "ee_finger_l2")
            and (m.geom_contype[g] or m.geom_conaffinity[g])]
    return m, d, probe, obj_geoms, fing


N_INVALID = [0]


def gdist(m, d, g1, g2, distmax=10.0):
    """mj_geomDistance with a witness check: |dist| must equal |from-to| within 1 mm.
    (Measured: MuJoCo 3.11 mesh-mesh GJK occasionally returns 0.0 with witness points 186 mm apart.)
    Invalid pairs return +inf and are counted in N_INVALID."""
    ft = np.zeros(6)
    dist = float(mujoco.mj_geomDistance(m, d, g1, g2, distmax, ft))
    if dist >= distmax:
        return dist
    if abs(abs(dist) - float(np.linalg.norm(ft[:3] - ft[3:]))) > 1e-3:
        N_INVALID[0] += 1
        return float("inf")
    return dist


def set_q(m, d, qpos):
    d.qpos[:len(qpos)] = qpos
    mujoco.mj_kinematics(m, d)


def base_to_world(m, d, p):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    R = d.xmat[bid].reshape(3, 3)
    return d.xpos[bid] + R @ np.asarray(p, float)


def nearest(m, d, probe, obj_geoms, pw):
    d.mocap_pos[PROBE_MOCAP] = pw
    mujoco.mj_kinematics(m, d)
    ft = np.zeros(6)
    per = {}
    for body, gs in obj_geoms.items():
        per[body] = min(gdist(m, d, probe, g) for g in gs)
    body = min(per, key=per.get)
    return body, CATEGORY[body], per


def main():
    m, d, probe, obj_geoms, fing = build()
    apple_geom = obj_geoms["apple"]
    res = {"definitions": __doc__, "runs": {}}
    for name, run in RUNS.items():
        ld = next(p for p in run.iterdir() if p.is_dir() and p.name != "server_constraints")
        rows = [json.loads(l) for l in (ld / "frames.jsonl").read_text().splitlines() if l.strip()]
        man = json.loads((ld / "manifest.json").read_text())
        plan = {int(r["seq"]): r for r in rows if r["kind"] == "planning"}
        ctrl = [r for r in rows if r["kind"] == "control"]
        files = sorted((run / "server_constraints/run_0000").glob("chunk_*.npz"))
        chunks = []
        for f in files:
            k = int(f.stem.split("_")[1])
            z = np.load(f, allow_pickle=True)
            s = json.loads(str(z["summary_json"]))
            pr = plan.get(k + 1)
            e = {"chunk": k, "t_step": 8 * k, "local_seq": k + 1, "has_local_row": pr is not None}
            if pr is None:
                chunks.append(e); continue
            assert pr["t_step"] == 8 * k, (name, k, pr["t_step"])
            refined = np.asarray(pr["actions"]); rec_ref = z["refined_chunk"]
            e["pair_check_max_abs_refined_vs_local_actions"] = float(np.abs(refined - rec_ref).max())
            v = pr.get("verdict") or {}
            e["executed"] = bool(pr.get("safe"))
            e["ipc"] = pr.get("ipc")
            kind = None
            if not e["executed"]:
                if pr.get("ipc") not in ("ok", "unsafe"):
                    kind = pr.get("ipc")
                elif v.get("geometry_certified") is False:
                    kind = "uncertified"
                elif v.get("trajopt_status") == "violated":
                    kind = "violated"
                else:
                    kind = "other"
            e["hold_kind"] = kind
            e["hold_reason"] = pr.get("hold_reason")
            e["trajopt_status"] = v.get("trajopt_status")
            e["geometry_certified"] = v.get("geometry_certified")
            e["ag3s_status"] = v.get("ag3s_status")
            e["server_timing_ms"] = v.get("timing_ms")
            e["client_policy_infer_ms"] = (pr.get("timing_ms") or {}).get("policy_infer")
            e["field_state"] = (pr.get("field") or {}).get("state")
            e["server_validity"] = s.get("validity"); e["server_status"] = s.get("status")
            e["server_notes"] = s.get("notes")
            # worst row
            mv = pr.get("max_violation_pair") or {}
            e["max_violation_m"] = v.get("max_violation_m")
            e["worst_row"] = {kk: mv.get(kk) for kk in ("block", "clearance_m", "step", "query", "link", "obstacle")}
            set_q(m, d, pr["qpos"])
            op = pr["object_poses"]
            apple_w = np.asarray(op["apple"]["pos"]); crate_w = np.asarray(op["crate"]["pos"])
            if mv.get("point_m") is not None:
                pw = base_to_world(m, d, mv["point_m"])
                body, cat, per = nearest(m, d, probe, obj_geoms, pw)
                set_q(m, d, pr["qpos"])
                e["worst_point_world"] = pw.tolist()
                e["worst_nearest_body"] = body; e["worst_nearest_category"] = cat
                e["worst_nearest_surface_mm"] = 1000 * per[body]
                e["worst_surface_mm_by_body"] = {b: 1000 * x for b, x in per.items()}
                e["worst_point_to_apple_centre_mm"] = float(1000 * np.linalg.norm(pw - apple_w))
                # sphere radius of that query (if a robot sphere row)
                q = mv.get("query")
                if q is not None and q < len(z["sphere_radii"]):
                    e["worst_sphere_radius_mm"] = float(1000 * z["sphere_radii"][q])
                    e["worst_sphere_link_npz"] = s["sphere_link_names"][q] if q < len(s.get("sphere_link_names", [])) else None
                else:
                    e["worst_query_is_attached_point"] = q is not None
            # grounding target
            e["has_target"] = s.get("has_target"); e["grounding_status"] = s.get("grounding_status")
            e["grounding_best_score"] = s.get("grounding_best_score")
            e["target_n_points"] = s.get("target_n_points")
            tc = z["target_centroid"] if "target_centroid" in z.files else None
            if tc is not None and np.all(np.isfinite(tc)) and s.get("has_target"):
                tw = base_to_world(m, d, tc)
                body, cat, per = nearest(m, d, probe, obj_geoms, tw)
                set_q(m, d, pr["qpos"])
                e["target_centroid_world"] = tw.tolist()
                e["target_to_apple_centre_mm"] = float(1000 * np.linalg.norm(tw - apple_w))
                e["target_to_crate_centre_mm"] = float(1000 * np.linalg.norm(tw - crate_w))
                centres = {b: np.asarray(op[b]["pos"]) for b in ("apple", "banana", "orange", "pear", "crate")}
                cd = {b: float(1000 * np.linalg.norm(tw - c)) for b, c in centres.items()}
                e["target_nearest_body_centre"] = min(cd, key=cd.get)
                e["target_centre_dist_mm"] = cd
                e["target_nearest_surface_body"] = body
                e["target_is_apple"] = e["target_nearest_body_centre"] == "apple"
            else:
                e["target_is_apple"] = None
            # finger -> apple (GT geoms)
            ft = np.zeros(6)
            e["finger_to_apple_surface_mm"] = 1000 * min(gdist(m, d, gf, ga, 1.0)
                                                         for gf in fing for ga in apple_geom)
            e["apple_dz_mm"] = None  # filled below
            # gripper
            ar = np.asarray(pr["actions_reference"]); aa = np.asarray(pr["actions"])
            e["policy_grip_l_cmd_rows0_7"] = ar[:8, 7].tolist()
            e["policy_grip_l_cmd_min_rows0_7"] = float(ar[:8, 7].min())
            e["sent_grip_l_cmd_min_rows0_7"] = float(aa[:8, 7].min())
            e["refined_minus_reference_exec_max_deg"] = float(np.degrees(np.abs(aa[:8, :] - ar[:8, :])[:, [0,1,2,3,4,5,6,8,9,10,11,12,13,14]].max()))
            cs = [c for c in ctrl if c["chunk_seq"] == k + 1]
            e["n_control_rows"] = len(cs)
            e["n_control_executed"] = sum(1 for c in cs if c.get("executed"))
            e["grip_l_measured_open_frac"] = [abs(c["qpos"][27]) / 0.045 for c in cs]
            chunks.append(e)
        z0 = plan[1]["object_poses"]["apple"]["pos"][2]
        for e in chunks:
            if e.get("has_local_row"):
                e["apple_dz_mm"] = 1000 * (plan[e["local_seq"]]["object_poses"]["apple"]["pos"][2] - z0)
        L = [e for e in chunks if e.get("has_local_row")]
        grasp_attempt = next((e["t_step"] for e in L if e["policy_grip_l_cmd_min_rows0_7"] < 0.5), None)
        switch = next((e["t_step"] for e in L if e.get("target_is_apple") is False
                       and e.get("target_nearest_body_centre") == "crate"), None)
        first_not_apple = next((e["t_step"] for e in L if e.get("target_is_apple") is False), None)
        swallowed = [{"t_step": e["t_step"], "chunk": e["chunk"], "policy_min_cmd": e["policy_grip_l_cmd_min_rows0_7"],
                      "hold_kind": e["hold_kind"]}
                     for e in L if not e["executed"] and e["policy_grip_l_cmd_min_rows0_7"] < 0.9]
        ipcs = {}
        for e in L:
            ipcs[e["ipc"]] = ipcs.get(e["ipc"], 0) + 1
        inf_srv = [e["server_timing_ms"]["infer"] for e in L if e.get("server_timing_ms")]
        tot_srv = [e["server_timing_ms"]["total"] for e in L if e.get("server_timing_ms")]
        cli = [e["client_policy_infer_ms"] for e in L if e.get("client_policy_infer_ms") is not None]
        summ = {
            "local_dir": str(ld), "argv": man["extra"]["argv"], "safe_timeout_s": man["policy"].get("safe_timeout_s"),
            "n_server_chunks": len(chunks), "n_paired": len(L),
            "pair_check_max_abs": max(e["pair_check_max_abs_refined_vs_local_actions"] for e in L),
            "n_executed": sum(e["executed"] for e in L), "n_hold": sum(not e["executed"] for e in L),
            "hold_kinds": {k: sum(1 for e in L if e["hold_kind"] == k) for k in {e["hold_kind"] for e in L if e["hold_kind"]}},
            "ipc_counts": ipcs, "n_timeout": ipcs.get("timeout", 0), "n_stale": ipcs.get("stale", 0),
            "field_state_counts_planning": {s: sum(1 for e in L if e["field_state"] == s) for s in {e["field_state"] for e in L}},
            "server_infer_ms": {"first": inf_srv[0], "median_excl_first": float(np.median(inf_srv[1:])), "max_excl_first": float(max(inf_srv[1:]))},
            "server_total_ms": {"first": tot_srv[0], "median_excl_first": float(np.median(tot_srv[1:])), "max_excl_first": float(max(tot_srv[1:]))},
            "client_policy_infer_ms": {"first": cli[0], "median_excl_first": float(np.median(cli[1:])), "max_excl_first": float(max(cli[1:]))},
            "grasp_attempt_t_first_policy_close_cmd_lt_0p5": grasp_attempt,
            "target_first_crate_t": switch, "target_first_not_apple_t": first_not_apple,
            "target_switch_minus_grasp_attempt_steps": (switch - grasp_attempt) if (switch is not None and grasp_attempt is not None) else None,
            "hold_chunks_with_policy_grip_cmd_lt_0p9": swallowed,
            "worst_nearest_category_counts": {c: sum(1 for e in L if e.get("worst_nearest_category") == c) for c in {e.get("worst_nearest_category") for e in L}},
            "worst_nearest_category_counts_hold_only": {c: sum(1 for e in L if not e["executed"] and e.get("worst_nearest_category") == c) for c in {e.get("worst_nearest_category") for e in L if not e["executed"]}},
            "worst_link_counts_hold_only": {c: sum(1 for e in L if not e["executed"] and e["worst_row"].get("link") == c) for c in {e["worst_row"].get("link") for e in L if not e["executed"]}},
            "apple_dz_max_mm": max(e["apple_dz_mm"] for e in L),
            "min_finger_to_apple_surface_mm": min(e["finger_to_apple_surface_mm"] for e in L),
            "t_first_finger_apple_contact_le_0mm": next((e["t_step"] for e in L if e["finger_to_apple_surface_mm"] <= 0), None),
        }
        res["runs"][name] = {"summary": summ, "chunks": chunks}
        print(name, json.dumps(summ, indent=1, default=str)[:4000])
    res["n_invalid_geomdistance_pairs_skipped"] = N_INVALID[0]
    print("invalid geomDistance pairs skipped:", N_INVALID[0])
    (OUT / "p12_records.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
