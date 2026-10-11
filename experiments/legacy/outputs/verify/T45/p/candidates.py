"""T45 Audit 5 (offline only, nothing implemented in benchmark/): candidate hand sphere models built with the SAME project builders
(read-only calls), differing from the server model only on the palm links ee_left / ee_right.
  palm_scale1   : capsule_radius_scale_by_link {ee_left: 1.0, ee_right: 1.0} (fingers: finger cover as server)
  palm_body_oX  : palm gap-filling capsules replaced by 'body spheres' from gripper_cover.finger_body_spheres applied to the palm
                  collision surface (same algorithm as --finger-cover, overflow X mm; surface spacing 1 mm, candidate grid 2 mm)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, time, hashlib, numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T45/p'))
import common as C
_built = {}

def _palm_body_model(overflow_mm):
    from AG3S.ag3s.experiments.sources.mujoco_source import HEAD_JOINTS, gap_filling_capsules
    from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf
    from AG3S.ag3s.robot_models.urdf_sphere_chain import UrdfCapsule, FINGER_BODY_ROLE
    from AG3S.ag3s.experiments.sources.gripper_cover import finger_body_capsules, finger_body_spheres, link_surface
    from AG3S.trajopt.serve_safe import constraint_link_filter
    sc = C.scene(); base = C.model("gripper")
    urdf = parse_urdf(RBY1_URDF)
    head = {n: float(sc.data.qpos[sc._qadr[n]]) for n in HEAD_JOINTS if n in sc._qadr}
    gap = gap_filling_capsules(sc.model)
    fl = ["ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2"]
    fcaps, _ = finger_body_capsules(sc.model, base, fl, overflow=0.0025)
    rep = {}; pcaps = []; cache = {}
    for palm in ("ee_right", "ee_left"):
        surf = link_surface(sc.model, base, palm, spacing=0.001)
        h = hashlib.sha1(np.ascontiguousarray(surf.points).tobytes()).hexdigest()
        t0 = time.time()
        fn = f"${AG3S_PROJECT_ROOT}/outputs/verify/T45/palm_body_o{overflow_mm:g}_{h[:12]}.npz"
        if h not in cache:
            import os, pickle
            if os.path.exists(fn):
                z = np.load(fn, allow_pickle=True); cache[h] = (z["c"], z["r"], json.loads(str(z["rep"])))
            else:
                cache[h] = finger_body_spheres(surf, overflow=overflow_mm * 1e-3, grid=0.002, union_eps=0.001)
                rp = {k: v for k, v in cache[h][2].items() if isinstance(v, (int, float, str))}
                np.savez(fn, c=cache[h][0], r=cache[h][1], rep=json.dumps(rp))
        c, r, rp = cache[h]
        rep[palm] = {"n": int(len(r)), "r_mm": [float(r.min() * 1e3), float(r.max() * 1e3)], "build_s": time.time() - t0,
                     **{k: v for k, v in rp.items() if isinstance(v, (int, float, str))}}
        for cc, rr in zip(c, r):
            o = np.eye(4); o[:3, 3] = cc
            pcaps.append(UrdfCapsule(link=palm, origin=o, radius=float(rr), length=0.0, role=FINGER_BODY_ROLE))
    by = {}
    for cap in list(fcaps) + pcaps: by.setdefault(cap.link, []).append(cap)
    new, placed = [], set()
    for cap in gap:
        if cap.link in by:
            if cap.link not in placed: new.extend(by[cap.link]); placed.add(cap.link)
            continue
        new.append(cap)
    m = UrdfSphereChain(urdf, extra_capsules=new, fixed_joint_values=head, link_filter=constraint_link_filter("gripper"), **C.SERVER_SPHERES)
    m.palm_body_report = rep
    return m

def models():
    if _built: return _built
    _built["palm_scale1"] = C.model("gripper", sphere_options={**C.SERVER_SPHERES, "capsule_radius_scale_by_link": {"ee_left": 1.0, "ee_right": 1.0}}, key="palm_scale1")
    for o in (2.5, 5.0):
        _built[f"palm_body_o{o:g}"] = _palm_body_model(o)
    return _built

if __name__ == "__main__":
    from coverage import link_report, qpos_at, OPEN
    out = {}
    qm_nofc = C.model("gripper", finger_cover=None)
    allm = {"server": C.model("gripper"), **models()}
    # sanity: rebuilding server palm path with no palm replacement must equal the server model -> compare finger + palm spheres of palm_body to server for fingers
    for k, qm in allm.items():
        rec = {"n_spheres_total": int(qm.n_spheres), "per_link_n": {L: int(sum(1 for n in qm.sphere_link_names if n == L)) for L in sorted(set(qm.sphere_link_names))},
               "palm_body_report": getattr(qm, "palm_body_report", None)}
        for nm in ("open", "half"):
            q = qpos_at(OPEN[nm])
            rep, _ = link_report(qm, qm_nofc, "ee_right", q, ["ee_right", "ee_finger_r1", "ee_finger_r2"])
            rec[f"ee_right_{nm}"] = {k2: rep[k2] for k2 in ("spheres_own", "own_spheres_vs_mjcf", "hand_union_vs_mjcf", "excess")}
            # grasp channel: palm spheres vs real palm, along the tip direction, inside the finger channel (world frame at this qpos)
            C.forward(q)
            pb = C.D.xpos[C.BODY("EE_BODY_R")]; f1 = C.D.xpos[C.BODY("ee_finger_r1")]; f2 = C.D.xpos[C.BODY("ee_finger_r2")]
            u = (f1 - f2) / np.linalg.norm(f1 - f2); mid = 0.5 * (f1 + f2); tip = mid - pb; tip -= u * (u @ tip); tip /= np.linalg.norm(tip)
            cen, rad = C.spheres_world(qm, q); links = np.asarray(qm.sphere_link_names); own = links == "ee_right"
            P = np.vstack([C.hull_sample(h[1])[0] for h in C.body_hulls(C.BODY("EE_BODY_R"))])
            F = np.vstack([C.hull_sample(h[1])[0] for b in ("ee_finger_r1", "ee_finger_r2") for h in C.body_hulls(C.BODY(b))])
            half = float(np.min(np.abs((F - mid) @ u)))  # inner face half-gap of the fingers at this opening
            ch_s = np.abs((cen[own] - mid) @ u) < half + rad[own]
            ch_p = np.abs((P - mid) @ u) < half
            rec[f"ee_right_{nm}"]["channel"] = {"finger_inner_half_gap_mm": half * 1e3,
                "palm_mesh_tipmost_in_channel_mm": float(((P[ch_p] - pb) @ tip).max() * 1e3) if ch_p.any() else None,
                "palm_spheres_tipmost_in_channel_mm": float((((cen[own] - pb) @ tip) + rad[own])[ch_s].max() * 1e3) if ch_s.any() else None,
                "fingertip_tipmost_mm": float(((F - pb) @ tip).max() * 1e3),
                "palm_spheres_lateral_reach_mm": float((np.abs((cen[own] - mid) @ u) + rad[own]).max() * 1e3),
                "palm_mesh_lateral_reach_mm": float(np.abs((P - mid) @ u).max() * 1e3)}
        out[k] = rec
        print(k, rec["n_spheres_total"], rec["per_link_n"], rec["palm_body_report"], json.dumps({n: (rec[f"ee_right_{n}"]["own_spheres_vs_mjcf"]["frac_out"], rec[f"ee_right_{n}"]["own_spheres_vs_mjcf"]["out_max_mm"], rec[f"ee_right_{n}"]["excess"]["protrusion_max_mm"], rec[f"ee_right_{n}"]["channel"]) for n in ("open", "half")}), flush=True)
    json.dump(out, open(_ag3s_workspace_path('outputs/verify/T45/candidates.json'), "w"), indent=1)
