"""T43 FR prototype — finger body sphere reduction options on one finger mesh (link frame), CPU only.

For each option: n spheres, uncovered targets (count, max depth = gap to the union surface), sampled overflow, inner-face x
(-> max opening), radius range.  Options: overflow sweep, min radius (candidate filter / radius floor), containment removal,
max count K (greedy truncated), cover tolerance (gap), exact ILP minimum (scipy milp).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, time
import numpy as np, mujoco
from scipy.spatial import cKDTree
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, build_constraint_robot_model
from AG3S.ag3s.experiments.sources import gripper_cover as GC

XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
E3B = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
m = mujoco.MjModel.from_xml_path(XML); scene = TransportScene.attach(m, mujoco.MjData(m))
bare = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=E3B)
surf = GC.link_surface(m, bare, "ee_finger_l1", spacing=0.0005)
outer = GC.finger_body_targets(surf)
T = surf.points[outer]
lo, hi = surf.vertices.min(0), surf.vertices.max(0)
axes = [np.arange(a, b + 1e-9, 0.0005) for a, b in zip(lo, hi)]
G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
piece = GC.hull_depth(G, surf.hulls); ins = piece > 0
G, piece = G[ins], piece[ins]
depth = np.maximum(piece, cKDTree(T).query(G)[0] - 0.0005)
print("targets", T.shape, "candidates", G.shape, flush=True)
tree = cKDTree(T)


def greedy(C, Rq, kmax=None):
    covers = [np.asarray(tree.query_ball_point(c, float(r)), np.int64) for c, r in zip(C, Rq)]
    gain = np.array([cv.size for cv in covers]); unc = np.ones(len(T), bool); chosen = []
    while unc.any() and (kmax is None or len(chosen) < kmax):
        order = np.lexsort((Rq, -gain)); best, bg = -1, 0
        for j in order:
            if gain[j] <= bg:
                break
            g = int(unc[covers[j]].sum()); gain[j] = g
            if g > bg:
                best, bg = j, g
        if best < 0:
            break
        chosen.append(best); unc[covers[best]] = False
    cnt = np.zeros(len(T), np.int64)
    for j in chosen:
        cnt[covers[j]] += 1
    kept = list(chosen)
    for j in reversed(chosen):
        if covers[j].size and np.all(cnt[covers[j]] >= 2):
            cnt[covers[j]] -= 1; kept.remove(j)
    return np.array(kept), covers


def metrics(C, R, name, extra=None):
    gap = GC._gap(T, C, R)
    allpts = GC._gap(surf.points, C, R)
    ov = GC._sampled_overflow(C, R, surf.hulls, surf.points) * 1e3
    inner_x = float((C[:, 0] - R).min() * 1e3)
    rec = {"name": name, "n": int(len(R)), "r_mm": [round(R.min() * 1e3, 2), round(R.max() * 1e3, 2)],
           "unc_targets": int((gap > 1e-9).sum()), "max_unc_mm": round(max(gap.max(), 0) * 1e3, 3),
           "unc_gt1mm": int((gap > 1e-3).sum()), "p99_gap_mm": round(max(np.percentile(gap, 99), 0) * 1e3, 3),
           "overflow_mm": round(ov, 3), "inner_x_mm": round(inner_x, 2), **(extra or {})}
    print(json.dumps(rec), flush=True)
    return rec


out = []
t0 = time.time()
for ov in (2.5, 3, 4, 5, 6):
    R = depth + ov * 1e-3
    idx, _ = greedy(G, R)
    out.append(metrics(G[idx], R[idx], f"overflow {ov}", {"sec": round(time.time() - t0, 1)}))
    if ov == 2.5:
        C25, R25 = G[idx], R[idx]
        # containment removal (sphere i inside sphere j)
        D = np.linalg.norm(C25[:, None] - C25[None], axis=2)
        cont = (D + R25[:, None] <= R25[None, :] + 1e-12) & ~np.eye(len(R25), dtype=bool)
        print("contained spheres at 2.5:", int(cont.any(1).sum()))
# min radius as candidate filter and as floor
for rmin in (3.0, 3.5, 4.0, 5.0):
    R = depth + 2.5e-3
    keep = R >= rmin * 1e-3
    idx, _ = greedy(G[keep], R[keep])
    out.append(metrics(G[keep][idx], R[keep][idx], f"min_r filter {rmin}"))
    Rf = np.maximum(R, rmin * 1e-3)
    idx, _ = greedy(G, Rf)
    out.append(metrics(G[idx], Rf[idx], f"min_r floor {rmin}"))
# max count
for ov in (2.5, 4.0):
    R = depth + ov * 1e-3
    for K in (8, 12, 16, 24, 32):
        idx, _ = greedy(G, R, K)
        out.append(metrics(G[idx], R[idx], f"overflow {ov} K {K}"))
# cover tolerance gap
for g in (0.5, 1.0, 1.5, 2.0):
    R = depth + 2.5e-3
    idx, _ = greedy(G, R + g * 1e-3)
    out.append(metrics(G[idx], R[idx], f"overflow 2.5 tol {g}"))
json.dump(out, open(_ag3s_workspace_path('outputs/impl/T43FR/proto_options.json'), "w"), indent=1)
