from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, types, numpy as np
from AG3S.trajopt.experiments.ta_offline import Run, variant_config, nearest_surface
from AG3S.trajopt.refiner import TrajOptChunkRefiner
idx = json.load(open(_ag3s_workspace_path('outputs/verify/T43/V8/analysis/analysis.json')))["runs"]
key = sys.argv[1]; seqs = [int(x) for x in sys.argv[2:]]
run = Run(key, idx[key], types.SimpleNamespace(hm=False, no_plane_rows=False))
cfg = variant_config(run.base_cfg, "A16", False)
holder = {}
ref = TrajOptChunkRefiner(run.robot, run.layout, lambda c: holder["v"], cfg)
opt = ref.optimizer; lin = opt.linearizer
for seq in seqs:
    inp, _ = run.inputs(seq); s = inp["summary"]; sc = inp["scene"]; q = inp["q_now"]
    R = run.layout.chunk_to_trajectory(inp["ref_chunk"])[:, :16]
    lin.set_joint_parameter_path(run.finger_path(s, inp["ref_chunk"], 16))
    lin.set_attached(sc.attached_points, sc.attached_parent_link)
    cen = lin.sphere_states(R, q)[0]
    clear = lin._esdf_clearance(cen, sc)[..., 0]
    names = list(run.robot.sphere_link_names) + ["held"] * (cen.shape[1] - lin.n_spheres)
    pose = (inp["row"].get("object_poses") or {})
    pose = json.loads(pose.replace("'", '"')) if isinstance(pose, str) else pose
    apple = np.asarray((pose.get("apple") or {}).get("pos", [np.nan] * 3))
    print(f"== {key} seq {seq} phase {s.get('grasp',{}).get('state')} attached {inp['held'] is not None} apple_gt {apple.round(3)} target_c {None if inp['target_centroid'] is None else inp['target_centroid'].round(3)}")
    k, j = np.unravel_index(np.argsort(clear[8:], axis=None)[:6], clear[8:].shape)
    for kk, jj in zip(k, j):
        p = cen[8 + kk, jj]; sp = nearest_surface(sc.esdf, p[None])[0]
        dobj = lin._manipulated_distance(p[None], sc)
        print(f"  step {8+kk} q{jj} {names[jj]} clear {1e3*clear[8+kk,jj]:.1f}mm  p {p.round(3)} surf {sp.round(3)} |surf-apple| {1e3*np.linalg.norm(sp-apple):.0f}mm  surf_z-plane {1e3*(sp@inp['plane'][:3]-inp['plane'][3]) if inp['plane'] is not None else None:.0f}mm  d_obj {None if dobj is None else round(1e3*float(dobj[0]),1)}")
