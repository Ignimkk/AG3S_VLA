"""T43 T fix 2 evidence — does the held deadlock clear when the held object's whole body leaves the field?

Per held chunk: held geometry = recorded spheres (as run) or the R2 body cover (rebuilt by AG3S.attach on
the recorded attach inputs, `outputs/impl/T43R2/replay.py`); field = recorded, or the recorded field with
that geometry's volume (above plane + 1 fine voxel) set free — **emulated** on the recorded ESDF: occupied =
d <= 0, the freed occupied voxels removed, the change in Euclidean distance to the remaining occupancy
(scipy EDT, per layer) added to the recorded distance. Then the TO rows along stay -> policy chunk."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, dataclasses, collections, importlib.util
import numpy as np
from scipy.ndimage import distance_transform_edt
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import recov_replay as RR
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
from AG3S.trajopt.safe_policy import classify_violations
from AG3S.ag3s.robot_models.held_object import sphere_volume_points

spec = importlib.util.spec_from_file_location("r2", _ag3s_workspace_path('outputs/impl/T43R2/replay.py'))
R2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R2)


def free_layer(dist, origin, voxel, spheres_world, z_min):
    d = np.asarray(dist, np.float32).astype(np.float64)
    shape = np.array(d.shape)
    pts = sphere_volume_points(spheres_world, 0.5 * voxel, z_min=z_min)
    freed = np.zeros(d.shape, bool)
    if len(pts):
        idx = np.round((pts - np.asarray(origin)) / voxel).astype(int)
        ok = np.all((idx >= 0) & (idx < shape), axis=1)
        idx = idx[ok]
        freed[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    occ = d <= 0.0
    occ2 = occ & ~freed
    if occ2.sum() == occ.sum():
        return dist, 0
    D_all = distance_transform_edt(~occ) * voxel
    D_rem = distance_transform_edt(~occ2) * voxel
    new = np.where(occ2, d, d + (D_rem - D_all))
    new = np.where(occ & ~occ2, D_rem, new)
    return new.astype(np.float16), int((occ & ~occ2).sum())


class Z(dict):
    files = property(lambda self: list(self.keys()))


def emulated_npz(z, spheres_world, z_min):
    out = Z({k: z[k] for k in z.files})
    n = {}
    for pre in ("esdf", "esdf_coarse"):
        if f"{pre}_distance" in z.files:
            out[f"{pre}_distance"], n[pre] = free_layer(z[f"{pre}_distance"], z[f"{pre}_origin"],
                                                        float(z[f"{pre}_voxel_size"]), spheres_world, z_min)
    return out, n


def body_spheres(run_key):
    rd = RR.Q.RUNS[run_key]
    ix = RR.Q.IX[rd]
    run = R2.Run(rd, ix)
    a = run.attaches()[0]
    ag, att, held = run.rebuild(a, body_cover=True)
    q = ag._held_query
    return a, {"body": q["body"], "lifted": q["lifted"], "parent": q["parent_link"]}


def scene_with(G, z, s, qn, local, radii, parent, field_npz=None):
    field = FIA.build_field(field_npz if field_npz is not None else z, G["outside"])
    scene = FIA.make_scene(field, G["robot"].radii, G["st"].esdf_margin)
    T = np.asarray(G["robot"].link_pose(qn, parent), float)
    world = local @ T[:3, :3].T + T[:3, 3]
    return dataclasses.replace(scene, manipulated_spheres=world, manipulated_sphere_radii=radii,
                               manipulated_link_margin=G["margin"], attached_points=local,
                               attached_parent_link=parent, attached_radii=radii), world


PADS = (0.0075, 0.010, 0.015)


def main(key, seqs, out):
    from AG3S.trajopt.safe_policy import recovery_floor_rows  # T43 T fix 1 (since dropped; runs on the fix-1 tree only)
    run = RR.Run(key); G = run.G
    a, bc = body_spheres(key)
    res = {"attach_seq": a, "body_r_mm": [round(1e3 * r, 1) for _, r in bc["lifted"]]}
    print(key, "attach", a, "body spheres", len(bc["lifted"]), res["body_r_mm"], flush=True)
    from AG3S.trajopt.linearize import CollisionLinearizer
    for seq in seqs:
        with np.load(run.ix["chunks"][str(seq)]["npz"], allow_pickle=False) as zz:
            z = Z({k: zz[k] for k in zz.files})
        s = json.loads(str(z["summary_json"]))
        refc = np.asarray(z["reference_chunk"], float)
        qn = FIA.q_now_from_qpos(np.asarray(run.plan[seq]["qpos"], float), G["qadr"])
        held = s["grasp"]["attached"]["held"]
        rec_local = np.array([c["centre_m"] for c in held["spheres_parent_frame"]])
        rec_r = np.array([c["radius_mm"] / 1e3 for c in held["spheres_parent_frame"]])
        b_local = np.array([c for c, _ in bc["lifted"]]); b_r = np.array([r for _, r in bc["lifted"]])
        plane = held["support"]["plane"]; z_min = float(plane["offset_m"]) + 0.005
        lin = CollisionLinearizer(G["robot"], G["layout"], 8)
        meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
        path, _ = G["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, 8, default=G["robot"].param_vector())
        lin.set_joint_parameter_path(path)
        auth = s["verdict"]["classification"]["authorized_links"]; man = s.get("manipulated") or {}
        stay = np.tile(qn[G["layout"].q_indices][:, None], (1, 8))
        ref = G["layout"].chunk_to_trajectory(refc)[:, :8]
        out_seq = {}
        # the manipulated object's observed points on the last pre-grasp (latched) chunk, at their place
        pre = max(k for k in range(1, a) if json.loads(str(np.load(run.ix["chunks"][str(k)]["npz"])["summary_json"]))["grasp"]["state"] == "latched")
        with np.load(run.ix["chunks"][str(pre)]["npz"]) as zp:
            obs_pts = np.asarray(zp["target_points"], float)
        variants = {"record": (rec_local, rec_r, False), "body": (b_local, b_r, False),
                    "body+free": (b_local, b_r, True)}
        for pad in PADS:
            variants[f"record+free_obs{int(pad*1e3)}"] = (rec_local, rec_r, ("obs", pad))
        for name, (loc, rad, free) in variants.items():
            T = np.asarray(G["robot"].link_pose(qn, bc["parent"] if name != "record" else s["grasp"]["attached"]["parent_link"]), float)
            parent = bc["parent"] if name != "record" else s["grasp"]["attached"]["parent_link"]
            fz, nfreed = (None, {})
            if free is True:
                world = loc @ T[:3, :3].T + T[:3, 3]
                fz, nfreed = emulated_npz(z, list(zip(world, rad)), z_min)
            elif free:
                fz, nfreed = emulated_npz(z, [(p_, free[1]) for p_ in obs_pts], z_min)
            scene, _ = scene_with(G, z, s, qn, loc, rad, parent, fz)
            lin.set_attached(scene.attached_points, scene.attached_parent_link)
            floor, rec = recovery_floor_rows(lin, qn, scene, authorized_links=auth, manipulated=man, tolerance=1e-4)
            fs = dataclasses.replace(scene, esdf_floor=floor) if floor is not None else scene
            rows = {}
            for al in (0.0, 0.2, 0.5, 1.0):
                tr = stay + al * (ref - stay)
                cls = classify_violations(lin, tr, qn, fs, authorized_links=auth, tolerance=1e-4, manipulated=man, keep_rows=True)
                agg = collections.defaultdict(lambda: [0, 0.0])
                for r in cls["rows"]:
                    k = f'{r["class"]}|{r["why"]}|{r["link"]}'
                    agg[k][0] += 1; agg[k][1] = min(agg[k][1], round(1e3 * r["clearance_m"], 2))
                rows[al] = dict(agg)
            out_seq[name] = {"freed_voxels": nfreed, "recovery_rows": None if rec is None else len(rec.get("queries", [])), "rows": rows}
            print(key, seq, name, "pre", pre, "n_obs", len(obs_pts), "freed", nfreed, "recov rows", out_seq[name]["recovery_rows"], flush=True)
            for al, agg in rows.items():
                print("     a", al, {k: v for k, v in sorted(agg.items(), key=lambda kv: kv[1][1]) if not k.startswith("allowed")} or "no non-allowed rows",
                      "| allowed worst", min([v[1] for k, v in agg.items() if k.startswith("allowed")], default=None), flush=True)
        res[seq] = out_seq
    pathlib.Path(out).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3])
