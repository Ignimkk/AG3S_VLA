"""T32 H4 (b) — rebuild the recorded field of one E3b chunk with the server's builder and sweep one parameter at a time.

usage (cwd = outputs/verify/T32/h4/runroot):
  MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}/outputs/verify/T31diag/snap ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python \
      ../real_rebuild.py E3b_ep1807_r1 30 [sweep|server]
Input: re-rendered depth for seq 1..k+1 (render_real.py; the server integrates one frame per chunk since reset,
esdf_stats.frames = k+1), attached points of chunk k (export_attached.py), fine window centre from the recorded
esdf_origin. Output: real/rebuild_<run>_chunk<k>.json (+ .npz of line profiles).
Wall line = T31 D4 wall_profile: base-x through the GT crate centre, y = crate centre y, z = crate centre z + 4 mm,
+-150 mm, 0.5 mm samples, EsdfField.distance (trilinear). Split at the crate centre x: lo = crate_wall_nx, hi = crate_wall_px.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, time, copy
import numpy as np
import mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T32/h4'))
import h4_common as H

T30 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T32/h4/real'))


def load_frames(run, seqs, cams=None):
    frames = []
    for seq in seqs:
        z = np.load(OUT / f"frames_{run}" / f"seq{seq:03d}.npz")
        names = sorted({k.split("/", 1)[1] for k in z.files})
        order = [c for c in ("zed_left", "wrist_cam_l", "wrist_cam_r") if c in names]
        use = order if cams is None else [c for c in order if c in cams]
        fr = []
        for c in use:
            fr.append(H.CameraDepth(name={"zed_left": "head", "wrist_cam_l": "left_wrist", "wrist_cam_r": "right_wrist"}[c],
                                    depth=z[f"depth_mm/{c}"].astype(np.float64) / 1000.0,
                                    camera_intrinsics=z[f"K/{c}"], T_base_cam=z[f"T/{c}"], robot_mask=z[f"mask/{c}"]))
        frames.append(fr)
    return frames


def gt_crate(run, seq):
    rd = T30 / f"{run.split('_')[0]}_rec" / run
    man = json.loads((rd / "manifest.json").read_text())
    rows = {int(x["seq"]): x for x in (json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()) if x.get("kind") == "planning"}
    m = mujoco.MjModel.from_xml_path(man["scene"]["model_xml"]); d = mujoco.MjData(m)
    d.qpos[:] = rows[seq]["qpos"]; mujoco.mj_forward(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base"); c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")
    Rb = d.xmat[b].reshape(3, 3); tb = d.xpos[b]
    cc = (d.xpos[c] - tb) @ Rb
    Rc = Rb.T @ d.xmat[c].reshape(3, 3)
    walls = {}
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if n in ("crate_wall_px", "crate_wall_nx"):
            walls[n] = {"centre_base_m": ((d.geom_xpos[g] - tb) @ Rb).tolist(), "half_size_m": m.geom_size[g].tolist()}
    return cc, Rc, walls


def recorded(run, k):
    A = json.loads((T30 / "p" / f"{run.split('_')[0]}_analysis.json").read_text())["runs"][run]
    c = [c for c in A["chunks"] if c.get("t") == 8 * k][0]
    z = np.load(T30 / f"{run.split('_')[0]}_rec" / c["server_chunk"], allow_pickle=False)
    fine = H.layer_from_arrays(z["esdf_distance"].astype(np.float32), z["esdf_origin"], float(z["esdf_voxel_size"]))
    coarse = H.layer_from_arrays(z["esdf_coarse_distance"].astype(np.float32), z["esdf_coarse_origin"], float(z["esdf_coarse_voxel_size"]))
    s = json.loads(str(z["summary_json"]))
    return c, z, fine, coarse, s


def compare(rec_layer, new_layer, box=None):
    a = np.asarray(rec_layer.distance_grid, np.float64); b = np.asarray(new_layer.distance_grid, np.float64)
    if a.shape != b.shape or not np.allclose(rec_layer.grid.origin, new_layer.grid.origin, atol=1e-5):
        return {"comparable": False}
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(a) < 0.39) & (np.abs(b) < 0.39)
    if box is not None:
        g = rec_layer.grid
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in g.shape], indexing="ij"), -1)
        p = g.origin + idx * g.voxel_size
        m &= np.all((p >= box[0]) & (p <= box[1]), axis=-1)
    d = np.abs(a - b)[m]
    return {"comparable": True, "n": int(m.sum()), "frac_abs_diff_le_1mm": float((d <= 0.001).mean()),
            "frac_abs_diff_le_5mm": float((d <= 0.005).mean()), "median_abs_diff_mm": float(1000 * np.median(d)),
            "p99_abs_diff_mm": float(1000 * np.percentile(d, 99)),
            "sign_agree_frac": float((np.sign(a[m]) == np.sign(b[m])).mean()),
            "n_neg_recorded": int((a[m] < 0).sum()), "n_neg_rebuilt": int((b[m] < 0).sum())}


def measure(built, p0, u, split, extra_tsdf=True):
    res = {}
    prof = {}
    for name in ("coarse", "fine"):
        if name not in built["tiers"]:
            continue
        L = built["tiers"][name]["layer"]
        out = H.band(L, p0, u, split=split)
        if isinstance(out, dict):
            res[name] = out; continue
        r, s, v = out
        sv, vv = H.voxel_samples(L, p0, u)
        r["voxel_centres_lt0"] = int(np.sum(vv[np.isfinite(vv)] < 0))
        r["voxel_size_m"] = float(built["tiers"][name]["vs"])
        r["sign_stats"] = {k: v2 for k, v2 in built["tiers"][name]["extra"].items() if k != "ball"}
        res[name] = r
        prof[name] = np.stack([s, v])
        prof[name + "_vox"] = np.stack([sv, vv])
    if extra_tsdf:
        s, pts = H.line_points(p0, u)
        sdf, w = H.tsdf_at(built["itg"], pts)
        res["tsdf"] = {"width_lt0_mm": float(0.5 * np.sum(sdf < 0)), "width_observed_mm": float(0.5 * np.sum(np.isfinite(sdf))),
                       "segments_lt0_m": H.segments(s, np.nan_to_num(sdf, nan=1.0) < 0),
                       "width_lt0_mm_lo": float(0.5 * np.sum(sdf[s < split] < 0)), "width_lt0_mm_hi": float(0.5 * np.sum(sdf[s >= split] < 0)),
                       "weight_max_on_line": float(np.nanmax(w)) if len(w) else None}
        prof["tsdf"] = np.stack([s, sdf, w])
    res["truncation_m"] = built["truncation_m"]; res["mapper_cfg"] = built["mapper_cfg"]
    return res, prof


def configs():
    S = dict(H.SERVER)
    out = [("server", S, {})]
    for tv in (0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
        out.append((f"trunc_{tv * 20:.0f}mm", {**S, "trunc_vox": tv}, {"axis": "truncation", "value_mm": tv * 20}))
    for tvx in (0.0025, 0.004, 0.0075, 0.010, 0.020):
        out.append((f"tsdf_{tvx * 1000:g}mm", {**S, "tsdf": tvx}, {"axis": "tsdf_voxel", "value_mm": tvx * 1000}))
    for fv in (0.0025, 0.010):
        out.append((f"fine_{fv * 1000:g}mm", {**S, "fine": fv}, {"axis": "fine_esdf_voxel", "value_mm": fv * 1000}))
    for cv in (0.010, 0.040):
        out.append((f"coarse_{cv * 1000:g}mm_trunc60", {**S, "voxel": cv, "trunc_vox": 0.060 / cv}, {"axis": "coarse_esdf_voxel (trunc fixed 60 mm)", "value_mm": cv * 1000}))
        out.append((f"coarse_{cv * 1000:g}mm_coupled", {**S, "voxel": cv}, {"axis": "coarse_esdf_voxel (trunc = 3 x voxel, server coupling)", "value_mm": cv * 1000}))
    for th in (0.0, 1.0, 3.0, 1e6):
        out.append((f"thr_{th:g}", {**S, "thr": th}, {"axis": "attached_sign_threshold_voxels", "value": th}))
    out.append(("no_attached", S, {"axis": "attached points", "value": "none", "attached": False}))
    out.append(("seeding_scatter", {**S, "seeding": "scatter"}, {"axis": "seeding_method", "value": "scatter"}))
    for mw in (1.0, 10.0):
        out.append((f"minw_{mw:g}", {**S, "min_weight": mw}, {"axis": "minimum_tsdf_weight", "value": mw}))
    for nf in (1, 5):
        out.append((f"frames_last{nf}", S, {"axis": "frames integrated", "value": nf, "frames": nf}))
    for cams in (("zed_left",), ("wrist_cam_l", "wrist_cam_r"), ("zed_left", "wrist_cam_l"), ("zed_left", "wrist_cam_r")):
        out.append(("cams_" + "+".join(cams), S, {"axis": "cameras", "value": list(cams), "cams": cams}))
    return out


def main():
    run = sys.argv[1]; k = int(sys.argv[2]); mode = sys.argv[3] if len(sys.argv) > 3 else "sweep"
    c, z, rec_fine, rec_coarse, s = recorded(run, k)
    seq = int(c["seq"])
    cc, Rc, walls = gt_crate(run, seq)
    p0 = np.array([cc[0], cc[1], cc[2] + 0.004]); u = np.array([1.0, 0.0, 0.0])
    split = 0.0
    fine_centre = H.fine_centre_from_origin(z["esdf_origin"], z["esdf_shape"], float(z["esdf_voxel_size"]))
    att_path = OUT / f"attached_{run}_chunk{k:05d}.npy"
    attached = np.load(att_path) if att_path.exists() else None
    n_frames_rec = (s.get("esdf_stats") or {}).get("frames")
    res = {"run": run, "chunk": k, "seq": seq, "t": c["t"], "server_chunk": c["server_chunk"], "recorded_frames": n_frames_rec,
           "gt_crate_centre_base_m": cc.tolist(), "gt_walls": walls, "line": {"p0_base_m": p0.tolist(), "u": u.tolist(), "half_m": 0.15, "step_m": 0.0005,
                                                                                     "split_s_m": split, "lo": "crate_wall_nx", "hi": "crate_wall_px"},
           "fine_centre_m": fine_centre.tolist(), "n_attached": None if attached is None else int(len(attached)), "configs": {}}
    # recorded bands on the same line
    rr = {}
    for name, L in (("coarse", rec_coarse), ("fine", rec_fine)):
        o = H.band(L, p0, u, split=split)
        rr[name] = o if isinstance(o, dict) else o[0]
    res["recorded"] = rr
    outp = OUT / f"rebuild_{run}_chunk{k:05d}{'' if mode == 'sweep' else '_' + mode}.json"
    profs = {}
    allc = configs()
    if mode == "sweep":
        todo = allc
    elif mode == "short":
        keep = {"server", "trunc_20mm", "trunc_30mm", "trunc_40mm", "trunc_80mm", "cams_zed_left", "frames_last1", "no_attached"}
        todo = [x for x in allc if x[0] in keep]
    else:
        todo = allc[:1]
    for name, cfg, meta in todo:
        t0 = time.monotonic()
        seqs = list(range(1, seq + 1))
        if "frames" in meta:
            seqs = seqs[-meta["frames"]:]
        frames = load_frames(run, seqs, meta.get("cams"))
        b = H.make_builder(**cfg)
        att = None if meta.get("attached") is False else attached
        try:
            built = H.build(b, frames, fine_centre if cfg["fine"] == H.SERVER["fine"] else fine_centre, att)
        except Exception as e:  # record, do not hide
            res["configs"][name] = {"cfg": cfg, "meta": {k2: v for k2, v in meta.items() if k2 != "cams"}, "error": repr(e)}
            print(name, "ERROR", repr(e), flush=True)
            continue
        r, prof = measure(built, p0, u, split)
        r["cfg"] = cfg; r["meta"] = {k2: (list(v) if isinstance(v, tuple) else v) for k2, v in meta.items()}
        r["n_frames"] = len(frames); r["n_cameras"] = len(frames[0])
        if name == "server" or name == "no_attached":
            r["vs_recorded"] = {"fine": compare(rec_fine, built["tiers"]["fine"]["layer"]),
                                "coarse": compare(rec_coarse, built["tiers"]["coarse"]["layer"]),
                                "coarse_crate_box": compare(rec_coarse, built["tiers"]["coarse"]["layer"],
                                                            box=(cc - [0.15, 0.2, 0.1], cc + [0.15, 0.2, 0.1])),
                                "fine_crate_box": compare(rec_fine, built["tiers"]["fine"]["layer"],
                                                          box=(cc - [0.15, 0.2, 0.1], cc + [0.15, 0.2, 0.1]))}
            for tn in ("fine", "coarse"):
                r[tn]["eikonal"] = H.eikonal(built["tiers"][tn]["values"], built["tiers"][tn]["vs"])
            np.savez_compressed(OUT / f"field_{run}_chunk{k:05d}_{name}.npz",
                                fine=built["tiers"]["fine"]["values"].astype(np.float16), fine_origin=built["tiers"]["fine"]["origin"],
                                coarse=built["tiers"]["coarse"]["values"].astype(np.float16), coarse_origin=built["tiers"]["coarse"]["origin"],
                                fine_vs=built["tiers"]["fine"]["vs"], coarse_vs=built["tiers"]["coarse"]["vs"])
        r["build_s"] = time.monotonic() - t0
        res["configs"][name] = r
        for kk, vv in prof.items():
            profs[f"{name}/{kk}"] = vv
        f = r.get("fine", {}); co = r.get("coarse", {})
        print(f"{name:32s} fine {f.get('width_lt0_mm')} ({f.get('width_lt0_mm_lo')}/{f.get('width_lt0_mm_hi')})  coarse {co.get('width_lt0_mm')} "
              f"({co.get('width_lt0_mm_lo')}/{co.get('width_lt0_mm_hi')})  tsdf<0 {r['tsdf']['width_lt0_mm']} ({r['tsdf']['width_lt0_mm_lo']}/{r['tsdf']['width_lt0_mm_hi']})  {r['build_s']:.1f}s", flush=True)
        del built, b
        import torch; torch.cuda.empty_cache()
        outp.write_text(json.dumps(res, indent=1, default=float))
        np.savez_compressed(outp.with_suffix(".npz"), **profs)
    print("recorded", json.dumps({k2: {kk: v for kk, v in v2.items() if kk.startswith("width")} for k2, v2 in rr.items()}))


if __name__ == "__main__":
    main()
