"""T32 H4 (a) — synthetic wall: analytic depth from the same camera model, same builder, same line measurement.

usage (cwd = outputs/verify/T32/h4/runroot):
  MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}/outputs/verify/T31diag/snap ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python ../synth.py [matrix|sweep|all]
Scene (base frame, metres): wall = axis-aligned box, centre x0 = 0.6029 (GT crate_wall_px x at chunk 30), y0 = -0.0155,
z from 0.828 to 0.940 (crate wall height 112 mm) or 0.828 to 1.400 ('tall'), y half-extent 0.150, thickness T (x).
Backdrop = table slab top z 0.820, 40 mm thick, x in [0.30, 1.00], y in [-0.50, 0.50] (table top of the transport scene).
Depth: per-pixel ray cast (pixel centre u+0.5, v+0.5), z-depth, wire-quantised to mm; no hit -> 0 (not integrated).
Cameras: K = zed_left intrinsics (640x480, TransportScene.intrinsics). Views:
  head_1side  : zed_left T_base_cam re-rendered at E3b_ep1807_r1 seq 31 (sees the wall from -x, from above)
  head_2side  : head_1side + a camera mirrored through the wall mid-plane x = x0 (look-at the mirrored target)
  normal_1side: look-at camera at (x0 - D, y0, z_mid), D = head camera distance to wall centre, optical axis +x
  normal_2side: normal_1side + its mirror (optical axis -x)
Frames: the same depth integrated N times (default N = 31, the chunk-30 frame count).
Line: base-x through (x0, y0, 0.884), +-150 mm, 0.5 mm samples (T31 D4 wall_profile definition).
Fine window centre = recorded chunk-30 fine window centre.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, time
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T32/h4'))
import h4_common as H

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T32/h4/synth')); OUT.mkdir(parents=True, exist_ok=True)
REAL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T32/h4/real'))
X0, Y0, ZL = 0.6028990339048239, -0.015542460486829916, 0.884
FINE_CENTRE = None


def boxes(T, tall=False):
    wall = (np.array([X0 - T / 2, Y0 - 0.150, 0.828]), np.array([X0 + T / 2, Y0 + 0.150, 1.400 if tall else 0.940]))
    table = (np.array([0.30, -0.50, 0.780]), np.array([1.00, 0.50, 0.820]))
    return [wall, table]


def raycast(K, Tbc, bxs, h=480, w=640):
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    R, o = Tbc[:3, :3], Tbc[:3, 3]
    dw = dc @ R.T
    best = np.full(len(dw), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for lo, hi in bxs:
            t1 = (lo - o) / dw; t2 = (hi - o) / dw
            tmin = np.nanmax(np.minimum(t1, t2), axis=1); tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= tmin) & (tmax > 0)
            t = np.where(tmin > 0, tmin, tmax)
            best = np.where(hit & (t < best), t, best)
    depth = np.where(np.isfinite(best), best, 0.0).reshape(h, w)
    return np.clip(np.rint(depth * 1000.0), 0, 65535) / 1000.0


def look_at(pos, target, up=(0, 0, 1)):
    z = np.asarray(target, float) - pos; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4); T[:3, :3] = np.stack([x, y, z], 1); T[:3, 3] = pos
    return T


def cameras():
    z = np.load(REAL / "frames_E3b_ep1807_r1" / "seq031.npz")
    K = z["K/zed_left"]; Th = z["T/zed_left"]
    wc = np.array([X0, Y0, ZL])
    Dist = float(np.linalg.norm(Th[:3, 3] - wc))
    # mirror of the head camera through x = X0: position and its optical-axis target
    tgt = Th[:3, 3] + Th[:3, 2] * Dist
    mpos = Th[:3, 3].copy(); mpos[0] = 2 * X0 - mpos[0]
    mtgt = tgt.copy(); mtgt[0] = 2 * X0 - mtgt[0]
    Tm = look_at(mpos, mtgt)
    Tn = look_at(np.array([X0 - Dist, Y0, ZL]), wc)
    Tn2 = look_at(np.array([X0 + Dist, Y0, ZL]), wc)
    views = {"head_1side": [Th], "head_2side": [Th, Tm], "normal_1side": [Tn], "normal_2side": [Tn, Tn2]}
    geo = {"K": K.tolist(), "head_pos_base_m": Th[:3, 3].tolist(), "head_axis": Th[:3, 2].tolist(), "distance_to_wall_centre_m": Dist,
           "head_axis_angle_to_wall_normal_deg": float(np.degrees(np.arccos(abs(Th[:3, 2][0])))),
           "head_depression_deg": float(np.degrees(np.arcsin(-Th[:3, 2][2])))}
    return K, views, geo


def frames_for(K, Ts, bxs, n):
    cams = [H.CameraDepth(name=f"c{i}", depth=raycast(K, T, bxs), camera_intrinsics=K, T_base_cam=T, robot_mask=None) for i, T in enumerate(Ts)]
    return [cams] * n


def run_one(K, Ts, bxs, cfg, n, T):
    b = H.make_builder(**cfg)
    built = H.build(b, frames_for(K, Ts, bxs, n), FINE_CENTRE)
    p0 = np.array([X0, Y0, ZL]); u = np.array([1.0, 0, 0])
    r = {"cfg": cfg, "n_frames": n, "truncation_m": built["truncation_m"], "mapper_cfg": built["mapper_cfg"]}
    prof = {}
    for name in ("coarse", "fine"):
        o = H.band(built["tiers"][name]["layer"], p0, u)
        if isinstance(o, dict):
            r[name] = o; continue
        rr, s, v = o
        rr["excess_over_wall_mm"] = rr["width_lt0_mm"] - 1000 * T
        segs = rr["segments_lt0_m"]
        if segs:
            rr["extent_front_mm"] = 1000 * (-T / 2 - min(a for a, _ in segs))   # >0: band starts before the -x face
            rr["extent_back_mm"] = 1000 * (max(b for _, b in segs) - T / 2)     # >0: band ends beyond the +x face
        rr["eikonal"] = H.eikonal(built["tiers"][name]["values"], built["tiers"][name]["vs"])
        r[name] = rr; prof[name] = np.stack([s, v])
        sv, vv = H.voxel_samples(built["tiers"][name]["layer"], p0, u); prof[name + "_vox"] = np.stack([sv, vv])
    s, pts = H.line_points(p0, u)
    sdf, w = H.tsdf_at(built["itg"], pts)
    segs = H.segments(s, np.nan_to_num(sdf, nan=1.0) < 0)
    r["tsdf"] = {"width_lt0_mm": float(0.5 * np.sum(sdf < 0)), "segments_lt0_m": segs,
                 "extent_back_mm": (1000 * (max(b for _, b in segs) - T / 2)) if segs else None,
                 "extent_front_mm": (1000 * (-T / 2 - min(a for a, _ in segs))) if segs else None}
    prof["tsdf"] = np.stack([s, sdf, w])
    import torch; del built, b; torch.cuda.empty_cache()
    return r, prof


def main():
    global FINE_CENTRE
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    zr = np.load(_ag3s_workspace_path('outputs/verify/T30/E3b_rec/server_8206/server_constraints/run_0000/chunk_00030.npz'))
    FINE_CENTRE = H.fine_centre_from_origin(zr["esdf_origin"], zr["esdf_shape"], float(zr["esdf_voxel_size"]))
    K, views, geo = cameras()
    S = dict(H.SERVER)
    outp = OUT / f"synth_{mode}.json"
    res = {"camera_geometry": geo, "wall_x0_m": X0, "line_z_m": ZL, "fine_centre_m": FINE_CENTRE.tolist(), "runs": {}}
    profs = {}
    todo = []
    if mode in ("matrix", "all"):
        for T in (0.016, 0.008):
            for tall in (False, True):
                for vn in views:
                    todo.append((f"matrix/T{T * 1000:.0f}/{'tall' if tall else 'crate_h'}/{vn}", T, tall, vn, S, 31))
    if mode in ("sweep", "all"):
        for vn in ("head_1side", "normal_1side", "head_2side"):
            T = 0.016; tall = False
            base = f"sweep/{vn}"
            for tv in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
                todo.append((f"{base}/trunc_{tv * 20:.0f}mm", T, tall, vn, {**S, "trunc_vox": tv}, 31))
            for tvx in (0.0025, 0.004, 0.0075, 0.010, 0.020):
                todo.append((f"{base}/tsdf_{tvx * 1000:g}mm", T, tall, vn, {**S, "tsdf": tvx}, 31))
            for fv in (0.0025, 0.010):
                todo.append((f"{base}/fine_{fv * 1000:g}mm", T, tall, vn, {**S, "fine": fv}, 31))
            for cv in (0.010, 0.040):
                todo.append((f"{base}/coarse_{cv * 1000:g}mm_trunc60", T, tall, vn, {**S, "voxel": cv, "trunc_vox": 0.060 / cv}, 31))
                todo.append((f"{base}/coarse_{cv * 1000:g}mm_coupled", T, tall, vn, {**S, "voxel": cv}, 31))
            todo.append((f"{base}/frames_1", T, tall, vn, S, 1))
            todo.append((f"{base}/seeding_scatter", T, tall, vn, {**S, "seeding": "scatter"}, 31))
            for mw in (1.0, 10.0):
                todo.append((f"{base}/minw_{mw:g}", T, tall, vn, {**S, "min_weight": mw}, 31))
            for th in (0.0, 3.0):
                todo.append((f"{base}/thr_{th:g}", T, tall, vn, {**S, "thr": th}, 31))
    if outp.exists() and "--resume" in sys.argv:
        old = json.loads(outp.read_text()); res["runs"].update(old.get("runs", {}))
        if outp.with_suffix(".npz").exists():
            zz = np.load(outp.with_suffix(".npz")); profs.update({k: zz[k] for k in zz.files})
        res["resumed_after_session_cut"] = sorted(old.get("runs", {}))[:0] or len(old.get("runs", {}))
    for name, T, tall, vn, cfg, n in todo:
        if name in res["runs"] and "error" not in res["runs"][name]:
            continue
        t0 = time.monotonic()
        try:
            r, prof = run_one(K, views[vn], boxes(T, tall), cfg, n, T)
        except Exception as e:
            res["runs"][name] = {"error": repr(e), "cfg": cfg}; print(name, "ERROR", repr(e), flush=True); continue
        r.update({"thickness_mm": T * 1000, "tall": tall, "view": vn, "build_s": time.monotonic() - t0})
        res["runs"][name] = r
        for k, v in prof.items():
            profs[f"{name}/{k}"] = v
        f, c = r.get("fine", {}), r.get("coarse", {})
        print(f"{name:45s} fine {f.get('width_lt0_mm')} [front {f.get('extent_front_mm')}, back {f.get('extent_back_mm')}]  coarse {c.get('width_lt0_mm')}  tsdf<0 {r['tsdf']['width_lt0_mm']} back {r['tsdf']['extent_back_mm']}", flush=True)
        outp.write_text(json.dumps(res, indent=1, default=float))
        np.savez_compressed(outp.with_suffix(".npz"), **profs)


if __name__ == "__main__":
    main()
