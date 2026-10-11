"""T43 FR prototype — exact minimum set cover (scipy milp / HiGHS) vs greedy, same candidates and guarantees.

Candidates = mesh-interior grid points at `cand_grid` (coarser than 0.5 mm to keep the ILP small) plus the greedy's chosen
centres (so the ILP can never be worse than greedy). Cover = target inside the sphere (R = depth + overflow).
usage: proto_ilp.py OVERFLOW_MM [CAND_GRID_MM] [TIME_LIMIT_S] [GAP_MM]  (gap = cover tolerance: target within gap of a sphere)
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, time
import numpy as np, mujoco
from scipy.spatial import cKDTree
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, build_constraint_robot_model
from AG3S.ag3s.experiments.sources import gripper_cover as GC
import logging; logging.disable(logging.WARNING)

ov = float(sys.argv[1]) * 1e-3
cg = float(sys.argv[2]) * 1e-3 if len(sys.argv) > 2 else 0.001
tl = float(sys.argv[3]) if len(sys.argv) > 3 else 300
gp = float(sys.argv[4]) * 1e-3 if len(sys.argv) > 4 else 0.0
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
E3B = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
m = mujoco.MjModel.from_xml_path(XML); scene = TransportScene.attach(m, mujoco.MjData(m))
bare = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=E3B)
surf = GC.link_surface(m, bare, "ee_finger_l1", spacing=0.0005)
C0, R0, rep = GC.finger_body_spheres(surf, overflow=ov)  # greedy, gap 0 (a feasible start for any gap)
T = surf.points[GC.finger_body_targets(surf)]
tree = cKDTree(T)
lo, hi = surf.vertices.min(0), surf.vertices.max(0)
axes = [np.arange(a, b + 1e-9, cg) for a, b in zip(lo, hi)]
G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
piece = GC.hull_depth(G, surf.hulls); ins = piece > 0
G, piece = G[ins], piece[ins]
depth = np.maximum(piece, tree.query(G)[0] - 0.0005)
C = np.vstack([G, C0]); R = np.concatenate([depth + ov, R0])
cols = [np.asarray(tree.query_ball_point(c, float(r + gp)), np.int64) for c, r in zip(C, R)]
# dominance-free not attempted; build sparse A (targets x candidates)
indptr = np.cumsum([0] + [len(c) for c in cols]); indices = np.concatenate(cols)
A = csc_matrix((np.ones(len(indices)), indices, indptr), shape=(len(T), len(C)))
print("greedy", len(R0), "candidates", len(C), "nnz", A.nnz, flush=True)
t = time.time()
res = milp(c=np.ones(len(C)), constraints=LinearConstraint(A, lb=1, ub=np.inf), integrality=np.ones(len(C)),
           bounds=Bounds(0, 1), options={"time_limit": tl, "disp": False, "mip_rel_gap": 0.0})
x = res.x > 0.5 if res.x is not None else None
print("ilp status", res.status, res.message, "n", None if x is None else int(x.sum()),
      "bound", getattr(res, "mip_dual_bound", None), "sec %.1f" % (time.time() - t), flush=True)
if x is not None:
    Cs, Rs = C[x], R[x]
    gap = GC._gap(T, Cs, Rs)
    out = {"overflow_mm": ov * 1e3, "gap_mm": gp * 1e3, "centres_mm": np.round(Cs * 1e3, 4).tolist(), "radii_mm": np.round(Rs * 1e3, 4).tolist(), "n_uncovered": int((gap > 1e-9).sum()), "cand_grid_mm": cg * 1e3, "greedy_n": int(len(R0)), "ilp_n": int(x.sum()),
           "ilp_bound": getattr(res, "mip_dual_bound", None), "status": int(res.status), "sec": time.time() - t,
           "max_gap_mm": float(max(gap.max(), 0) * 1e3),
           "overflow_sampled_mm": GC._sampled_overflow(Cs, Rs, surf.hulls, surf.points) * 1e3,
           "inner_x_mm": float((Cs[:, 0] - Rs).min() * 1e3)}
    print(json.dumps(out), flush=True)
    json.dump(out, open(f"${AG3S_PROJECT_ROOT}/outputs/impl/T43FR/proto_ilp_{sys.argv[1]}_g{gp*1e3:g}_tl{int(tl)}.json", "w"), indent=1)
