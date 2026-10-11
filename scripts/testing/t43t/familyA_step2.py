"""T43 T step 2 on the family-A deadlocks: the attach snapshot from the **pre-closing** geometry
(`closing_keep_geometry`: the last latched chunk's target points, captured at that chunk's pose) instead of
the recorded 33/34-point closing cap — AG3S.attach rebuilt on the recorded inputs (T43 R2 harness). Held
chunks: held / manipulated spheres = the rebuilt ones; the field = the recorded field with the rebuilt held
spheres' volume freed at q_now (the per-frame `held_free`, **emulated**, EDT on the recorded ESDF; the
recorded field already had the old, smaller spheres freed). Fidelity: the rebuild on the recorded inputs must
give the recorded spheres."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, re, dataclasses, importlib.util
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import recov_replay as RR
import body_emul as BE
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
from AG3S.ag3s.robot_models.held_object import held_spheres_of

spec = importlib.util.spec_from_file_location("r2", _ag3s_workspace_path('scripts/testing/t43t/r2_replay.py'))
R2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R2)


def rebuild(r2run, a, points, geometry_seq, body_cover=False):
    from AG3S.ag3s.stages.geometry import fit_sphere
    from AG3S.ag3s.types import TargetGeometry
    g = r2run.summ[a]["grasp"]; held = g["attached"]["held"]
    with np.load(r2run.npz[a - 1], allow_pickle=True) as zp:
        field_prev = FIA.build_field(zp, r2run.G["outside"])
    ag = R2.make_ag3s(r2run.G["robot"], body_cover)
    pl = (held.get("support") or {}).get("plane")
    ag._last_support_surfaces = [] if pl is None else [R2.plane_surface(pl)]
    ag._last_esdf_field = field_prev
    pts = np.asarray(points, float)
    tg = TargetGeometry(id=0, points=pts, point_indices=np.arange(len(pts)), centroid=pts.mean(0),
                        bounding_geometry=fit_sphere(pts), attention_score=0.9, confidence=0.9)
    att = ag.attach(tg, robot_state=r2run.q_now(a), parent_link=g["parent_link"],
                    geometry_state=r2run.q_now(geometry_seq))
    sp = held_spheres_of(att)
    if body_cover:                      # T43 R2: the J1-lifted body cover is the query set near the table
        sp = ag._held_query["lifted"]
    return att, np.array([c for c, _ in sp]), np.array([r for _, r in sp]), ag.held_record


def main(key, seqs, out, body_cover=False):
    run = RR.Run(key)
    rd = RR.Q.RUNS[key]
    r2 = R2.Run(rd, RR.Q.IX[rd])
    a = r2.attaches()[0]
    held_rec = r2.summ[a]["grasp"]["attached"]["held"]
    f = int(re.match(r"frame (\d+)", held_rec["snapshot_state"]).group(1))
    pre = max(k for k in sorted(r2.summ) if k < a and r2.summ[k]["grasp"]["state"] == "latched")
    with np.load(r2.npz[a], allow_pickle=True) as z:
        pts_rec = np.asarray(z["target_points"], float)
    with np.load(r2.npz[pre], allow_pickle=True) as z:
        pts_pre = np.asarray(z["target_points"], float)
    att0, loc0, rad0, _ = rebuild(r2, a, pts_rec, f + 1)
    rec_loc = np.array([c["centre_m"] for c in held_rec["spheres_parent_frame"]])
    fid = float(np.abs(loc0 - rec_loc).max()) if loc0.shape == rec_loc.shape else None
    att1, loc1, rad1, hrec1 = rebuild(r2, a, pts_pre, pre, body_cover)
    res = {"attach": a, "pre": pre, "n_points": [len(pts_rec), len(pts_pre)], "fidelity_max_m": fid,
           "spheres_rec_mm": [round(1e3 * r, 1) for r in rad0], "spheres_step2_mm": [round(1e3 * r, 1) for r in rad1],
           "parent": att1.parent_link, "chunks": {}}
    print(key, "attach", a, "pre", pre, "pts", res["n_points"], "fid", fid, "spheres", res["spheres_rec_mm"], "->", res["spheres_step2_mm"], flush=True)
    G = run.G
    plane = held_rec["support"]["plane"]; z_min = float(plane["offset_m"]) + 0.005

    def scene_fn(seq, s, qn):
        with np.load(run.ix["chunks"][str(seq)]["npz"], allow_pickle=False) as zz:
            z = BE.Z({k: zz[k] for k in zz.files})
        T = np.asarray(G["robot"].link_pose(qn, att1.parent_link), float)
        world = loc1 @ T[:3, :3].T + T[:3, 3]
        fz, n = BE.emulated_npz(z, list(zip(world, rad1)), z_min)
        scene_fn.freed = n
        sc, _ = BE.scene_with(G, z, s, qn, loc1, rad1, att1.parent_link, fz)
        return sc

    for seq in seqs:
        r = run.solve(seq, scene_fn=scene_fn)
        lift = RR.lift_mm(run, r)
        res["chunks"][seq] = {k: v for k, v in r.items() if k not in ("chunk", "trajectory", "q_now", "reference_traj")}
        res["chunks"][seq].update(lift_mm=lift, freed=scene_fn.freed)
        print(key, seq, "step2", "hold" if r["hold"] else "EXEC", r["reasons"], r["status"], "lift", lift,
              "raw robot/held", r["raw_robot_min_mm"], r["raw_held_min_mm"], "freed", scene_fn.freed, flush=True)
    pathlib.Path(out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3],
         body_cover=len(sys.argv) > 4 and sys.argv[4] == "body")
