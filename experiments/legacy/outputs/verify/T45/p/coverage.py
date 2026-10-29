"""T45 Audit 1+2: per hand link spheres of the server constraint model vs MuJoCo collision geometry (convex hulls) and the
capsules the spheres were made from, all in the MuJoCo WORLD frame at one recorded qpos, finger opening open / half / closed.
Pairing URDF link <-> MJCF body by sphere position (align.json), never by name."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T45/p'))
import common as C
from scipy.spatial import cKDTree
OUT = _ag3s_workspace_path('outputs/verify/T45')
align = json.load(open(f"{OUT}/align.json"))
L2B = {L: v["nearest_body_by_sphere_position"] for L, v in align.items()}
r = [json.loads(l) for l in open(_ag3s_workspace_path('outputs/verify/T44/CL/runs/C1a/R/C1a_ep3_s31/frames.jsonl'))]
QPOS0 = np.asarray([x for x in r if x["kind"] == "control" and x["t_step"] == 224][0]["qpos"])
sc = C.scene()
FJ = [n for n in ("gripper_finger_r1", "gripper_finger_r2", "gripper_finger_l1", "gripper_finger_l2")]
def open_val(n):
    j = C.M.joint(n).id; lo, hi = C.M.jnt_range[j]; return lo if abs(lo) > abs(hi) else hi
OPEN = {"open": 1.0, "half": 0.5, "closed": 0.0}
def qpos_at(frac):
    q = QPOS0.copy()
    for n in FJ: q[sc._qadr[n]] = frac * open_val(n)
    return q

def rigid_neighbour_bodies(bid):
    root = bid
    while C.M.body_jntnum[root] == 0 and C.M.body_parentid[root] != 0: root = C.M.body_parentid[root]
    grp, st = [], [root]
    while st:
        b = st.pop(); grp.append(b)
        st += [k for k in range(C.M.nbody) if C.M.body_parentid[k] == b and k != b and C.M.body_jntnum[k] == 0]
    return [b for b in grp if b != bid]

def cap_surface(p0, p1, rad, n=4000, rng=np.random.default_rng(1)):
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    t = rng.random(n)[:, None]
    ax = p1 - p0; L = np.linalg.norm(ax)
    if L < 1e-9: return p0 + rad * d
    u = ax / L; d = d - (d @ u)[:, None] * u; d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    side = p0 + t * ax + rad * d
    e = rng.normal(size=(n // 2, 3)); e /= np.linalg.norm(e, axis=1, keepdims=True)
    caps = np.vstack([p0 + rad * e[(e @ u) < 0], p1 + rad * e[(e @ u) > 0]])
    return np.vstack([side, caps])

def cap_gap(P, caps):
    g = np.full(len(P), np.inf)
    for p0, p1, rad in caps:
        ax = p1 - p0; L2 = max(ax @ ax, 1e-18)
        t = np.clip((P - p0) @ ax / L2, 0, 1)
        g = np.minimum(g, np.linalg.norm(P - (p0 + t[:, None] * ax), axis=1) - rad)
    return g

def stats(g):
    out = g[g > 0]
    return {"n": int(g.size), "frac_out": float(np.mean(g > 0)), "frac_out_gt1mm": float(np.mean(g > 1e-3)),
            "frac_out_gt2p5mm": float(np.mean(g > 2.5e-3)), "frac_out_gt5mm": float(np.mean(g > 5e-3)),
            "out_median_mm": float(np.median(out) * 1e3) if out.size else 0.0,
            "out_p95_mm": float(np.percentile(out, 95) * 1e3) if out.size else 0.0,
            "out_max_mm": float(out.max() * 1e3) if out.size else 0.0,
            "all_p95_mm": float(np.percentile(g, 95) * 1e3)}

def capsules_world(qm, L, qpos, role=None):
    """capsules of link L (link frame) -> world via URDF FK."""
    T = qm.link_pose(C.q_from_qpos(sc, qm, qpos), L, C.params_from_qpos(sc, qm, qpos))
    out = []
    for c in qm.capsules:
        if c.link != L or (role and getattr(c, "role", "collision") != role): continue
        Tc = T @ c.origin
        p0 = Tc[:3, 3] - Tc[:3, 2] * c.length / 2; p1 = Tc[:3, 3] + Tc[:3, 2] * c.length / 2
        out.append((p0, p1, float(c.radius)))
    return out

def link_report(qm, qm_nofc, L, qpos, hand_links, rng_seed=0):
    C.forward(qpos)
    bid = C.BODY(L2B[L])
    hulls = C.body_hulls(bid)
    eqs = [h[2] for h in hulls]
    P_all, owner = [], []
    for i, (g, V, eq) in enumerate(hulls):
        P, _, _ = C.hull_sample(V, rng=np.random.default_rng(rng_seed + i)); P_all.append(P); owner.append(np.full(len(P), i))
    P = np.vstack(P_all); owner = np.concatenate(owner)
    # drop samples buried > 0.2 mm in another piece of the same body (not surface)
    buried = np.zeros(len(P), bool)
    for i, eq in enumerate(eqs):
        m = owner != i
        buried[m] |= (-np.max(P[m] @ eq[:, :3].T + eq[:, 3], axis=1)) > 2e-4
    P = P[~buried]
    # samples inside a rigidly attached neighbour body's hull (palm top under FT sensor / wrist) = unreachable
    nb = []
    for b in rigid_neighbour_bodies(bid):
        nb += [h[2] for h in C.body_hulls(b)]
    hidden = (C.inside_depth(P, nb) > -5e-4) if nb else np.zeros(len(P), bool)
    Pe = P[~hidden]
    cen, rad = C.spheres_world(qm, qpos); links = np.asarray(qm.sphere_link_names)
    own = links == L; hand = np.isin(links, hand_links)
    caps_model = capsules_world(qm_nofc, L, qpos)        # capsules the server spheres came from (gap-filling; fingers: before finger cover)
    rep = {"mjcf_body": L2B[L], "n_geoms": len(hulls), "n_surface_samples_exposed": int(len(Pe)), "n_hidden_by_rigid_neighbour": int(hidden.sum()),
           "spheres_own": {"n": int(own.sum()), "r_mm_min": float(rad[own].min() * 1e3), "r_mm_max": float(rad[own].max() * 1e3)},
           "own_spheres_vs_mjcf": stats(C.gap(Pe, cen[own], rad[own])),
           "hand_union_vs_mjcf": stats(C.gap(Pe, cen[hand], rad[hand])),
           "source_capsules": [{"r_mm": c[2] * 1e3, "L_mm": float(np.linalg.norm(c[1] - c[0]) * 1e3)} for c in caps_model],
           "source_capsules_vs_mjcf": stats(cap_gap(Pe, caps_model)) if caps_model else None}
    if caps_model:
        S = np.vstack([cap_surface(*c) for c in caps_model])
        S = S[cap_gap(S, caps_model) > -1e-6]       # outer surface of the capsule union
        rep["own_spheres_vs_source_capsule_surface"] = stats(C.gap(S, cen[own], rad[own]))
    # excess volume: 1 mm voxels inside own sphere union but outside every collision hull of this body
    lo = (cen[own] - rad[own, None]).min(0); hi = (cen[own] + rad[own, None]).max(0)
    ax = [np.arange(lo[k], hi[k] + 1e-3, 1e-3) for k in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    inU = C.gap(G, cen[own], rad[own]) <= 0
    U = G[inU]; outside = C.inside_depth(U, eqs) < 0
    tree = cKDTree(P)
    dist_out = tree.query(U[outside])[0] if outside.any() else np.zeros(0)
    rep["excess"] = {"union_volume_cm3": float(inU.sum() * 1e-3), "union_outside_geom_cm3": float(outside.sum() * 1e-3),
                     "frac_union_outside_geom": float(outside.mean()) if inU.any() else None,
                     "protrusion_max_mm": float(dist_out.max() * 1e3) if dist_out.size else 0.0,
                     "geom_hull_volume_cm3_upper": float(sum(__import__('scipy.spatial', fromlist=['ConvexHull']).ConvexHull(h[1]).volume for h in hulls) * 1e6)}
    return rep, Pe

if __name__ == "__main__":
    res = {"qpos_source": "outputs/verify/T44/CL/runs/C1a/R/C1a_ep3_s31/frames.jsonl control t_step 224", "finger_open_value": {n: float(open_val(n)) for n in FJ},
           "models": {}}
    for links in ("gripper", "arms"):
        qm = C.model(links); qm_nofc = C.model(links, finger_cover=None)
        mres = {"n_spheres_total": int(qm.n_spheres), "per_link": {}}
        hand_links = {"right": ["ee_right", "ee_finger_r1", "ee_finger_r2"], "left": ["ee_left", "ee_finger_l1", "ee_finger_l2"]}
        for L in ["ee_right", "ee_finger_r1", "ee_finger_r2", "ee_left", "ee_finger_l1", "ee_finger_l2"]:
            c, rr = qm.local_spheres(L)
            roles = [ro for n, ro in zip(qm.sphere_link_names, qm.sphere_roles) if n == L]
            mres["per_link"][L] = {"spheres_link_frame": {"centre_m": np.round(c, 6).tolist(), "r_m": np.round(rr, 6).tolist(), "role": roles}}
        if links == "gripper" or True:
            for side, HL in hand_links.items():
                for L in HL:
                    for nm, frac in OPEN.items():
                        rep, _ = link_report(qm, qm_nofc, L, qpos_at(frac), HL)
                        mres["per_link"][L][nm] = rep
                        o = rep["own_spheres_vs_mjcf"]; h = rep["hand_union_vs_mjcf"]
                        print(links, L, nm, "own out %.3f p95 %.1f max %.1f | hand out %.3f max %.1f | caps out %s | excess %.1f cm3" % (
                            o["frac_out"], o["out_p95_mm"], o["out_max_mm"], h["frac_out"], h["out_max_mm"],
                            None if rep["source_capsules_vs_mjcf"] is None else round(rep["source_capsules_vs_mjcf"]["frac_out"], 3),
                            rep["excess"]["union_outside_geom_cm3"]), flush=True)
        res["models"][links] = mres
    # same-link sphere identity gripper vs arms
    a, b = C.model("gripper"), C.model("arms")
    res["gripper_vs_arms_hand_spheres_identical"] = {L: bool(np.allclose(a.local_spheres(L)[0], b.local_spheres(L)[0]) and np.allclose(a.local_spheres(L)[1], b.local_spheres(L)[1]))
                                                     for L in ["ee_right", "ee_finger_r1", "ee_finger_r2", "ee_left", "ee_finger_l1", "ee_finger_l2"]}
    from AG3S.ag3s.robot_models import RBY1_URDF, parse_urdf
    u = parse_urdf(RBY1_URDF)
    res["urdf_file_capsules_on_hand_links"] = [{"link": c.link, "r_mm": c.radius * 1e3, "L_mm": c.length * 1e3} for c in u.capsules
                                               if c.link in ("ee_right", "ee_left", "ee_finger_r1", "ee_finger_r2", "ee_finger_l1", "ee_finger_l2")]
    res["urdf_file"] = str(RBY1_URDF)
    print("identical", res["gripper_vs_arms_hand_spheres_identical"]); print("urdf caps", res["urdf_file_capsules_on_hand_links"])
    json.dump(res, open(f"{OUT}/coverage.json", "w"), indent=1)
