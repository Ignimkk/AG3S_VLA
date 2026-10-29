"""T45 Audit 3: hand-fruit_other contact events of T44d (q4_contacts.json: C0 / C1a / C1b) -> which MJCF hand geom touched, where, and whether
that point lies inside the server sphere union (own link / active hand) and inside candidate models.
Contact location: recorded control-row qpos (replay qpos_mismatch_rows = 0, so recorded == replay pre-step state) for rows t_first..t_last+1;
per row and per (hand geom, fruit geom) pair of the event, mj_geomDistance -> nearest point pair. Rows with distance <= 1 mm are 'contact rows';
the hand-side point is classified. MJCF body -> URDF link by position (align.json)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, glob, pathlib, numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T45/p'))
import common as C
from coverage import capsules_world, cap_gap
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T45'))
CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL'))
align = json.load(open(OUT / "align.json")); B2L = {v["nearest_body_by_sphere_position"]: L for L, v in align.items()}
sc = C.scene()
MODELS = {"server": C.model("gripper")}
if len(sys.argv) > 1 and sys.argv[1] == "cands":
    import candidates as K
    MODELS.update(K.models())
QNF = C.model("gripper", finger_cover=None)
ev_all = json.load(open(_ag3s_workspace_path('outputs/verify/T44/CLA/q4_contacts.json')))
out = []
for e in ev_all:
    name = e["name"]; arm = e["arm"]
    rj = json.load(open(CL / "replay" / f"{arm}_R_{name}.json"))
    evs = [x for x in rj["contact_events"]["hand-fruit_other"] if x["t_first"] == e["t_first"]]
    pairs = evs[0]["pairs"]
    a = json.load(open(CL / "analysis" / f"{name}.json")); side = a["used_arm"]; s1 = "l" if side == "left" else "r"
    rows = {x["t_step"]: x for x in (json.loads(l) for l in open(pathlib.Path(a["run_dir"]) / "frames.jsonl")) if x["kind"] == "control"}
    hand_links = [f"ee_{side}", f"ee_finger_{s1}1", f"ee_finger_{s1}2"]
    pts = []
    for t in range(e["t_first"], e["t_last"] + 2):
        if t not in rows: continue
        qpos = np.asarray(rows[t]["qpos"]); C.forward(qpos)
        sph = {k: C.spheres_world(qm, qpos) for k, qm in MODELS.items()}
        for p in pairs:
            gh = [x for x in p if x.startswith("geom#")][0]; gf = [x for x in p if not x.startswith("geom#")][0]
            g1 = int(gh[5:]); g2 = mujoco.mj_name2id(C.M, mujoco.mjtObj.mjOBJ_GEOM, gf)
            ft = np.zeros(6); d = mujoco.mj_geomDistance(C.M, C.D, g1, g2, 0.05, ft)
            if d > 1e-3: continue
            src = "geomDistance_fromto"
            if d <= 0.0:   # penetrating: mj_geomDistance gives 0 and no usable fromto -> use the solver contacts of this pair (contact.pos = mid point)
                cs = [C.D.contact[i] for i in range(C.D.ncon) if {int(C.D.contact[i].geom1), int(C.D.contact[i].geom2)} == {g1, g2}]
                if not cs: continue
                ci = min(cs, key=lambda c: c.dist); d = float(ci.dist); src = "contact_pos"
                ft[:3] = np.asarray(ci.pos)
            ph = ft[:3]; body = C.BNAME(C.M.geom_bodyid[g1]); L = B2L[body]
            mesh = mujoco.mj_id2name(C.M, mujoco.mjtObj.mjOBJ_MESH, int(C.M.geom_dataid[g1]))
            bid = C.M.geom_bodyid[g1]; Rb = C.D.xmat[bid].reshape(3, 3); pb = Rb.T @ (ph - C.D.xpos[bid])
            rec = {"t": t, "geom": g1, "mesh": mesh, "mjcf_body": body, "urdf_link": L, "fruit_geom": gf, "dist_mm": d * 1e3, "point_source": src,
                   "point_world": ph.tolist(), "point_body_mm": (pb * 1e3).tolist()}
            for k, qm in MODELS.items():
                cen, rad = sph[k]; links = np.asarray(qm.sphere_link_names)
                own = links == L; hand = np.isin(links, hand_links)
                rec[f"{k}_gap_own_mm"] = float(C.gap(ph[None], cen[own], rad[own])[0] * 1e3)
                rec[f"{k}_gap_hand_mm"] = float(C.gap(ph[None], cen[hand], rad[hand])[0] * 1e3)
            rec["source_capsule_r1_gap_mm"] = float(cap_gap(ph[None], capsules_world(QNF, L, qpos))[0] * 1e3)
            pts.append(rec)
    rec = {"arm": arm, "name": name, "t_first": e["t_first"], "t_last": e["t_last"], "max_force_N": e["max_force_N"], "fruit": e["fruit"],
           "grasp_state": e["grasp_state"], "pairs": pairs, "n_contact_points": len(pts)}
    if pts:
        for k in list(MODELS) :
            g = np.array([p[f"{k}_gap_own_mm"] for p in pts]); gh = np.array([p[f"{k}_gap_hand_mm"] for p in pts])
            rec[f"{k}_gap_own_mm_min_med_max"] = [float(g.min()), float(np.median(g)), float(g.max())]
            rec[f"{k}_gap_hand_mm_min_med_max"] = [float(gh.min()), float(np.median(gh)), float(gh.max())]
            rec[f"{k}_frac_points_outside_hand_union"] = float(np.mean(gh > 0))
        gc = np.array([p["source_capsule_r1_gap_mm"] for p in pts]); rec["source_capsule_r1_gap_mm_min_med_max"] = [float(gc.min()), float(np.median(gc)), float(gc.max())]
        rec["bodies"] = sorted({p["mjcf_body"] for p in pts}); rec["meshes"] = sorted({p["mesh"] for p in pts})
    rec["points"] = pts
    out.append(rec)
    print(arm, name, e["t_first"], rec.get("bodies"), rec.get("meshes"), "n", len(pts), "server own", rec.get("server_gap_own_mm_min_med_max"), "hand", rec.get("server_gap_hand_mm_min_med_max"),
          {k: rec.get(f"{k}_gap_hand_mm_min_med_max") for k in MODELS if k != "server"}, flush=True)
fn = "events_cands.json" if len(sys.argv) > 1 else "events.json"
json.dump(out, open(OUT / fn, "w"), indent=1)
