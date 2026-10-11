"""T43 FC — existing mechanisms vs the finger collision mesh (URDF ee_finger_l1, link frame; all four fingers are equal).
rows: E3b (scale 0.05), --capsule-radius-scale-link ee_finger_*=S (S 0.5, 1.0), T43R --gripper-cover (finger part), FC (2.0, 2.5 mm).
coverage = all collision-surface samples (0.5 mm, internal faces removed); overflow = sphere-surface samples outside the mesh,
distance to the nearest mesh sample (max)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np, mujoco
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, FINGER_LINKS, build_constraint_robot_model
from AG3S.ag3s.experiments.sources.gripper_cover import link_surface, finger_body_targets, _gap, _sampled_overflow
XML=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m=mujoco.MjModel.from_xml_path(XML); scene=TransportScene.attach(m, mujoco.MjData(m))
E3B=dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
def build(**kw):
    o=dict(E3B); o.update(kw.pop("so",{})); return build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=o, **kw)
variants={"E3b (now)":build(),
 "scale-link 0.5":build(so={"capsule_radius_scale_by_link":{l:0.5 for l in FINGER_LINKS}}),
 "scale-link 1.0 (URDF-equivalent)":build(so={"capsule_radius_scale_by_link":{l:1.0 for l in FINGER_LINKS}}),
 "T43R gripper-cover (finger)":build(gripper_cover=True),
 "FC overflow 2.0":build(finger_cover={"overflow":0.002}),
 "FC overflow 2.5 (default)":build(finger_cover={"overflow":0.0025})}
link="ee_finger_l1"
surf=link_surface(m, variants["E3b (now)"], link, spacing=0.0005)
tgt=surf.points[finger_body_targets(surf)]
lo,hi=surf.vertices.min(0),surf.vertices.max(0)
out={"mesh_mm":{"x":[lo[0]*1e3,hi[0]*1e3],"y":[lo[1]*1e3,hi[1]*1e3],"z":[lo[2]*1e3,hi[2]*1e3]},"rows":{}}
for name,rob in variants.items():
    c,r=rob.local_spheres(link); g=_gap(tgt,c,r)
    op={o["parent_link"]:o["inner_gap_m"] for o in rob.gripper_openings()}
    row={"n_finger":int(r.size),"n_total":rob.n_spheres,"r_mm":[float(r.min()*1e3),float(r.max()*1e3)],
         "frac_inside":float(np.mean(g<=0)),"frac_within_2.5mm":float(np.mean(g<=0.0025)),"max_gap_mm":float(max(g.max(),0)*1e3),
         "overflow_mm":_sampled_overflow(c,r,surf.hulls,surf.points)*1e3,
         "inner_face_x_mm":float((c[:,0]-r).min()*1e3),"tip_z_mm":float((c[:,2]-r).min()*1e3),"y_mm":float((np.abs(c[:,1])+r).max()*1e3),
         "max_opening_mm":op["ee_left"]*1e3}
    out["rows"][name]=row
    print(f"{name:34s} n {row['n_finger']:4d}/{row['n_total']:5d} r {row['r_mm'][0]:.1f}-{row['r_mm'][1]:.1f} inside {row['frac_inside']:.3f} <=2.5 {row['frac_within_2.5mm']:.3f} gap {row['max_gap_mm']:.2f} overflow {row['overflow_mm']:.2f} inner_x {row['inner_face_x_mm']:.1f} tip_z {row['tip_z_mm']:.1f} |y| {row['y_mm']:.1f} open {row['max_opening_mm']:.1f}", flush=True)
json.dump(out,open(_ag3s_workspace_path('outputs/impl/T43FC/alt_mechanisms.json'),"w"),indent=1)
