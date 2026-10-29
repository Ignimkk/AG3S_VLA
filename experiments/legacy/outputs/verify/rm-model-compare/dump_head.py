"""Cross-check: BEFORE model reconstructed from current code vs build_robot_model of the HEAD tree
(git cee8f84 snapshot at outputs/verify/T28/p3/0f/tree_HEAD, the T14-era self-filter model).
Run with PYTHONPATH=<tree_HEAD>:${AG3S_PROJECT_ROOT}. Compares sphere centres/radii at the same q."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, numpy as np, mujoco
import benchmark
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
D = _ag3s_workspace_path('outputs/verify/rm-model-compare')
m = mujoco.MjModel.from_xml_path(XML)
rm = build_robot_model(TransportScene.attach(m, mujoco.MjData(m)))
raw = json.load(open(f"{D}/raw.json")); z = np.load(f"{D}/raw.npz")
q = np.asarray(raw["robot_state_q"])
c, r = rm.sphere_centers_numeric(q)[:2] if isinstance(rm.sphere_centers_numeric(q), tuple) else (rm.sphere_centers_numeric(q), rm.radii)
res = dict(benchmark_from=benchmark.__file__, n_spheres_head=int(len(r)), n_spheres_before=int(len(z["before_radii"])),
           same_count=bool(len(r) == len(z["before_radii"])))
if res["same_count"]:
    res["max_abs_centre_diff_m"] = float(np.abs(c - z["before_centers"]).max())
    res["max_abs_radius_diff_m"] = float(np.abs(r - z["before_radii"]).max())
    res["links_equal"] = bool(list(rm.sphere_link_names) == list(z["before_links"]))
print(json.dumps(res, indent=1)); json.dump(res, open(f"{D}/crosscheck_head.json", "w"), indent=1)
