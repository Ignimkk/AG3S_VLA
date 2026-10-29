from common.workspace import workspace_path as _ag3s_workspace_path
import time, json, numpy as np, mujoco
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
from AG3S.trajopt.serve_safe import constraint_link_filter
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m = mujoco.MjModel.from_xml_path(XML); sc = TransportScene.attach(m, mujoco.MjData(m))
S = dict(capsule_radius_scale=0.05, sphere_spacing=0.4, max_spheres_per_capsule=32)
for links in ("gripper", "arms"):
    t = time.time()
    base = build_constraint_robot_model(sc, link_filter=constraint_link_filter(links), sphere_options=S, finger_cover={"hands": "both"})
    pc = build_constraint_robot_model(sc, link_filter=constraint_link_filter(links), sphere_options=S, finger_cover={"hands": "both"}, palm_cover=True)
    print(links, base.n_spheres, "->", pc.n_spheres, "added", pc.n_spheres - base.n_spheres, f"{time.time()-t:.1f}s")
    from collections import Counter
    print(" per link", dict(Counter(pc.sphere_link_names)))
    rep = pc.palm_cover_report; print(" report", {k: v for k, v in rep.items() if k != "links"}, {l: {k: r[k] for k in ("n_spheres","radius_mm","n_targets","n_uncovered","max_gap_mm","overflow_sampled_mm")} for l, r in rep["links"].items()})
z = np.load(_ag3s_workspace_path('outputs/verify/T45/palm_body_o5_0b966459a6ff.npz'), allow_pickle=True)
for L in ("ee_right", "ee_left"):
    c, r = pc.local_spheres(L)
    print(L, "audit npz equal: centres", np.array_equal(c, z["c"]), "radii", np.array_equal(r, z["r"]), np.abs(c - z["c"]).max() if c.shape == z["c"].shape else c.shape)
# rest of model identical to base (non-palm)
nb = np.asarray(base.sphere_link_names); npc = np.asarray(pc.sphere_link_names)
q = np.random.default_rng(1).uniform(-1, 1, base.nq)
a, ra = base.sphere_centers_numeric(q); b, rb = pc.sphere_centers_numeric(q)
kb = ~np.isin(nb, ["ee_left", "ee_right"]); kp = ~np.isin(npc, ["ee_left", "ee_right"])
print("non-palm identical", np.array_equal(nb[kb], npc[kp]), np.array_equal(a[kb], b[kp]), np.array_equal(ra[kb], rb[kp]))
print("first palm idx", np.flatnonzero(np.isin(npc, ["ee_left","ee_right"]))[[0,-1]], "roles", set(r for r, n in zip(pc.sphere_roles, npc) if n in ("ee_left","ee_right")))
c, r = pc.local_spheres("ee_right")
print(r.dtype, z["r"].dtype, np.abs(r - z["r"]).max(), r[:3], z["r"][:3])
