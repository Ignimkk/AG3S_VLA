"""T43 V10 — cause of every V10 HOLD chunk: recorded binding row (link, class, clearance) + nearest field surface identity (verifier, CPU only, recorded data).
Method = outputs/impl/T43HM/p/hm_survey.py (HM survey): field = FIA.build_field(server chunk record); query sphere centre P = to.metrics.max_violation_pair.point_m (base frame);
d = field.distance(P), g = field.gradient(P), S = P - d*g/|g| = nearest field surface point; GT identity of S = nearest MuJoCo collision-geom group at the observation state
(planning-row qpos; hurdle | table | crate | apple | fruit | shelf | room | robot) by geom_point_sd; 'unidentified' = no geom within IDENT_MM of S.
r = sphere_radii[query] (robot rows, query < n_robot) | held query set radius [query - n_robot] (held rows). d_minus_r_mm vs recorded clearance gives the margin the TO applied.
held rows additionally: self_cap_mm = signed distance of S to the attach-time fit spheres (record held.fit_spheres, base frame; capture pose, offline: fit spheres only, runtime volume also has the pre-grasp place).
usage: hold_cause_v10.py OUT_JSON [RUN_KEY ...]   (RUN_KEY like V10/H/V10_ep1834_s18341; default = every V10 run with a HOLD chunk)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys, time
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import numpy as np, mujoco  # noqa: E402
import hold_w as HW  # noqa: E402
import dx_common as C  # noqa: E402
from AG3S.trajopt.experiments import fixed_input_ablation as FIA  # noqa: E402

T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
GROUPS = ("hurdle", "table", "crate", "apple", "fruit", "shelf", "room", "robot")
IDENT_MM = 20.0


def main():
    out_path = pathlib.Path(sys.argv[1]); runs = json.load(open(T43 / "V10/analysis/analysis.json"))["runs"]
    keys = sys.argv[2:] or [k for k, r in sorted(runs.items()) if r["n_hold_chunks"]]
    cl = HW.load_client(); res = {"definitions": __doc__, "client_snapshot": str(HW.HY), "runs": {}}
    for key in keys:
        t0w = time.time(); a = runs[key]; rd = pathlib.Path(a["run_dir"])
        man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = int(man["scene"]["episode_index"])
        z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"])
        stt = rd.parent / f"{rd.name}.start"
        cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None) if stt.exists() else None
        L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; mg = L["obstacle_manager"]
        hurdle = set(int(g) for g in mg.collision_geom_ids) if mg is not None else set()
        side = "right" if ep == 1834 else "left"; SD = side[0].upper()
        apple_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple"); apple_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom")
        ggroup = {}
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            root = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.body_rootid[int(m.geom_bodyid[g])])) or ""
            if g in hurdle: ggroup[g] = "hurdle"
            elif root == "table": ggroup[g] = "table"
            elif root == "crate": ggroup[g] = "crate"
            elif g == apple_g: ggroup[g] = "apple"
            elif root in ("banana", "orange", "pear"): ggroup[g] = "fruit"
            elif root == "shelf": ggroup[g] = "shelf"
            elif root in ("world", "office"): ggroup[g] = "room"
            elif not root.startswith("pick_obstacle"): ggroup[g] = "robot"
        crate_geoms = [g for g, v in ggroup.items() if v == "crate"]

        def setq(q):
            d.qpos[:] = q; mujoco.mj_kinematics(m, d)

        def mesh_verts(g):
            mid = int(m.geom_dataid[g]); a0 = int(m.mesh_vertadr[mid]); n = int(m.mesh_vertnum[mid])
            V = m.mesh_vert[a0:a0 + n].astype(float)
            return V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]

        def group_dist(Pw):
            out = np.full((len(Pw), len(GROUPS)), np.inf)
            for g, gr in ggroup.items():
                t = int(m.geom_type[g])
                if t == 0: dd = Pw[:, 2] - d.geom_xpos[g][2]
                elif t == 7:
                    V = mesh_verts(g); dd = np.linalg.norm(Pw[:, None, :] - V[None], axis=2).min(1)
                else:
                    try: dd = C.geom_point_sd(m, d, g, Pw)
                    except ValueError: continue
                j = GROUPS.index(gr); out[:, j] = np.minimum(out[:, j], dd)
            return out

        rows_f = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
        plan = {r["seq"]: r for r in rows_f if r["kind"] == "planning"}
        seg = C.lit(a.get("server_segment")); chunks = C.segment_chunks(seg)
        st, outside, om, logp = C.server_settings(seg, rd / "manifest.json")
        sc = {c["seq"]: c for c in a["server_chunks"]}; att_t = next((c["t"] for c in a["server_chunks"] if c["attached"]), None)
        n_att_seen = {}
        cnt = 0
        for c in a["server_chunks"]:
            if c["attached"]:
                cnt += 1; n_att_seen[c["seq"]] = cnt
        recs = []
        for ch in a["chunks"]:
            if not ch["hold"]:
                continue
            seq = ch["seq"]; zz = np.load(chunks[seq], allow_pickle=True); s = json.loads(str(zz["summary_json"]))
            mv = C.lit((s.get("to") or {}).get("metrics", {}).get("max_violation_pair")) or {}
            n_robot = int(zz["sphere_radii"].shape[0]); q = mv.get("query")
            qcap = np.asarray(plan[seq]["qpos"], float); setq(qcap); Tb = C.T_world_base(m, d)
            apple_w = d.xpos[apple_b].copy()
            crate_d = min(float(C.geom_point_sd(m, d, g, apple_w[None])[0]) for g in crate_geoms) if crate_geoms else np.inf
            rec = {"seq": seq, "t": int(ch["t"]), "hold_kinds": ch.get("hold_kinds"), "attached": bool(sc[seq]["attached"]), "n_att": n_att_seen.get(seq),
                   "phase": ("pre_attach" if not sc[seq]["attached"] else ("lift" if n_att_seen[seq] <= 3 else ("crate" if crate_d <= 0.080 else "carry"))),
                   "apple_crate_mm": 1e3 * crate_d, "to_status": sc[seq]["to_status"],
                   "binding_recorded": {"rows": sc[seq]["binding"]["rows"], "link": sc[seq]["binding"]["link"], "category": sc[seq]["binding"]["category"], "clearance_mm": sc[seq]["binding"]["clearance_mm"]},
                   "worst_row": {"query": q, "step": mv.get("step"), "link": mv.get("link"), "class": mv.get("obstacle_margin_class"), "obstacle_label": mv.get("obstacle"),
                                 "tier": mv.get("tier"), "point_m": mv.get("point_m"), "clearance_mm": None if mv.get("clearance_m") is None else 1e3 * float(mv["clearance_m"]), "held_obstacle_margin": mv.get("held_obstacle_margin")},
                   "held_rows_metrics": sc[seq].get("held_rows"), "is_held_row": bool(q is not None and q >= n_robot), "n_robot_spheres": n_robot}
            if mv.get("point_m") is not None and q is not None:
                field = FIA.build_field(zz, outside); vox = float(field.grid.voxel_size)
                P = np.asarray(mv["point_m"], float)[None]
                dd = float(np.asarray(field.distance(P), float)[0]); gg = np.asarray(field.gradient(P), float)[0]; nr = float(np.linalg.norm(gg))
                ghat = gg / nr if nr > 1e-12 else np.zeros(3); Sb = P[0] - dd * ghat; Sw = Tb[:3, :3] @ Sb + Tb[:3, 3]
                if q < n_robot:
                    r_ = float(zz["sphere_radii"][q])
                else:
                    used, sets, flk, bc = C.held_sets(s); H = sets.get(used) or []
                    r_ = float(H[q - n_robot][1]) if 0 <= q - n_robot < len(H) else None
                rec["surface"] = {"d_mm": 1e3 * dd, "r_mm": None if r_ is None else 1e3 * r_, "d_minus_r_mm": None if r_ is None else 1e3 * (dd - r_), "grad_norm": nr, "voxel_mm": 1e3 * vox,
                                  "S_base_m": Sb.tolist(), "S_world_m": Sw.tolist()}
                gd = group_dist(Sw[None])[0]; j = int(np.argmin(gd))
                rec["surface"]["gt_dist_by_group_mm"] = {GROUPS[i]: (None if not np.isfinite(gd[i]) else round(1e3 * float(gd[i]), 2)) for i in range(len(GROUPS))}
                rec["surface"]["gt_group_nearest"] = GROUPS[j]; rec["surface"]["gt_nearest_mm"] = 1e3 * float(gd[j])
                rec["surface"]["identity"] = GROUPS[j] if 1e3 * float(gd[j]) <= IDENT_MM else "unidentified"
                if rec["is_held_row"]:
                    g_ = s.get("grasp") or {}; att = C.lit(g_.get("attached")) or {}; hrec = C.lit(att.get("held")) or {}
                    fit = [(np.asarray(x["centre_m"], float), 1e-3 * float(x["radius_mm"])) for x in (hrec.get("fit_spheres") or [])]
                    if fit:
                        Fc = np.array([c for c, _ in fit]); Fr = np.array([r for _, r in fit]); rec["surface"]["self_cap_mm"] = float(1e3 * C.union_sd(Sb, Fc, Fr)[0])
                    rec["surface"]["apple_dist_mm_gt"] = rec["surface"]["gt_dist_by_group_mm"]["apple"]
            recs.append(rec)
        res["runs"][key] = {"key": key, "n_hold": len(recs), "side": side, "chunks": recs, "sec": round(time.time() - t0w, 1)}
        print(key, "hold", len(recs), "sec", round(time.time() - t0w, 1), flush=True)
        out_path.write_text(json.dumps(res, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
