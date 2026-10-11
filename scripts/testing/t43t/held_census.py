"""T43 T step 2 side effect — the attach snapshot from the pre-closing geometry makes other held spheres.
For every recorded attach (T42-1, T41 c): rebuild AG3S.attach on the recorded inputs (fidelity) and with the
last latched chunk's target points (step 2); on each held chunk (attach .. detach) the held query rows along the
**recorded refined plan** on the recorded field — step 2 with its own spheres' volume freed (emulated, as
held_free does every held frame). Counts held chunks whose worst held row is below 0 (recorded spheres vs step 2)
and where (nearest surface height over the table plane)."""
import json, sys, pathlib, re, dataclasses
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import familyA_step2 as F
import body_emul as BE
import qnow as Q
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
from AG3S.trajopt.linearize import CollisionLinearizer


def runs():
    return [rd for rd, v in Q.IX.items() if "chunks" in v and v.get("complete")
            and ("/T42/1/" in rd or "/T41/c/" in rd)]


def held_min(G, lin, z, s, qn, loc, rad, parent, field_npz, plane_z):
    refc = np.asarray(z["refined_chunk"], float)
    traj = G["layout"].chunk_to_trajectory(refc)[:, :8]
    sc, _ = BE.scene_with(G, z, s, qn, loc, rad, parent, field_npz)
    lin.set_attached(sc.attached_points, sc.attached_parent_link)
    cl = lin.esdf_clearance(traj, qn, sc)[..., 0][:, lin.n_spheres:]
    k = np.unravel_index(np.argmin(cl), cl.shape)
    c = lin.sphere_states(traj, qn)[0][k[0], lin.n_spheres + k[1]]
    d = float(sc.esdf.distance(c[None])[0]); g = np.asarray(sc.esdf.gradient(c[None]), float)[0]
    g /= max(np.linalg.norm(g), 1e-12)
    return float(cl.min()), float((c - d * g)[2] - plane_z)


def one(rd):
    try:
        r2 = F.R2.Run(rd, Q.IX[rd])
        atts = r2.attaches()
        if not atts:
            return {"run": rd, "attaches": []}
        G = Q.setup(str(pathlib.Path(Q.IX[rd]["server"]) / "serve_safe.log"), str(pathlib.Path(rd) / "manifest.json"))
        lin = CollisionLinearizer(G["robot"], G["layout"], 8)
        out = {"run": rd.split("/runs/")[-1], "attaches": []}
        for a in atts:
            held = r2.summ[a]["grasp"]["attached"]["held"]
            f = int(re.match(r"frame (\d+)", held["snapshot_state"]).group(1))
            pres = [k for k in sorted(r2.summ) if k < a and r2.summ[k]["grasp"]["state"] == "latched"]
            if not pres:
                continue
            pre = pres[-1]
            with np.load(r2.npz[a]) as z:
                pts_a = np.asarray(z["target_points"], float)
            with np.load(r2.npz[pre]) as z:
                pts_p = np.asarray(z["target_points"], float)
            _, loc0, rad0, _ = F.rebuild(r2, a, pts_a, f + 1)
            att1, loc1, rad1, rec1 = F.rebuild(r2, a, pts_p, pre)
            rec_loc = np.array([c["centre_m"] for c in held["spheres_parent_frame"]])
            fid = float(np.abs(loc0 - rec_loc).max()) if loc0.shape == rec_loc.shape else None
            plane_z = float(held["support"]["plane"]["offset_m"]) if held.get("support", {}).get("plane") else 0.82
            ent = {"attach": a, "pre": pre, "fid_m": fid, "n_points": [len(pts_a), len(pts_p)],
                   "r_rec_mm": [round(1e3 * x, 1) for x in rad0], "r_step2_mm": [round(1e3 * x, 1) for x in rad1],
                   "mode_step2": rec1.get("mode"), "chunks": []}
            for s_ in r2.held_seqs(a):
                with np.load(r2.npz[s_]) as zz:
                    z = BE.Z({k: zz[k] for k in zz.files})
                s = json.loads(str(z["summary_json"]))
                qn = r2.q_now(s_)
                parent = s["grasp"]["attached"]["parent_link"]
                m0, h0 = held_min(G, lin, z, s, qn, loc0, rad0, parent, None, plane_z)
                T = np.asarray(G["robot"].link_pose(qn, att1.parent_link), float)
                world = loc1 @ T[:3, :3].T + T[:3, 3]
                fz, _ = BE.emulated_npz(z, list(zip(world, rad1)), plane_z + 0.005)
                m1, h1 = held_min(G, lin, z, s, qn, loc1, rad1, att1.parent_link, fz, plane_z)
                ent["chunks"].append({"seq": s_, "rec_mm": round(1e3 * m0, 2), "rec_surface_above_plane_mm": round(1e3 * h0, 1),
                                      "step2_mm": round(1e3 * m1, 2), "step2_surface_above_plane_mm": round(1e3 * h1, 1),
                                      "executed": int((s.get("exec_feedback") or {}).get("n_exec") or 0)})
            lin.set_attached(None, None)
            out["attaches"].append(ent)
        print("done", rd, flush=True)
        return out
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"run": rd, "error": traceback.format_exc()[-800:]}


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=int(sys.argv[2])) as ex:
        res = list(ex.map(one, runs()))
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, default=str))
