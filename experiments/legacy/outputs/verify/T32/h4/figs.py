"""T32 H4 figures -> AG3S/ag3s/docs/figures/t32/t32-h4-*.png (+ .json sidecars).
usage: MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}/outputs/verify/T31diag/snap ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python figs.py
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T31diag/snap'))
from AG3S.ag3s.experiments.common import figstyle as fs

H4 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T32/h4'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t32')); FIG.mkdir(parents=True, exist_ok=True)
V = json.loads(pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T32.h4.verify.json')).read_text())
N = V["numbers"]; R30 = N["real_chunk30"]
C = fs.CATEGORICAL
fs.use_korean()
plt.rcParams["axes.unicode_minus"] = False
cc = np.array(R30["gt_walls"]["crate_wall_px"]["centre_base_m"]); cc[0] = (R30["gt_walls"]["crate_wall_px"]["centre_base_m"][0] + R30["gt_walls"]["crate_wall_nx"]["centre_base_m"][0]) / 2
CRATE = np.array(json.loads((H4 / "real" / "rebuild_E3b_ep1807_r1_chunk00030.json").read_text())["gt_crate_centre_base_m"])
LINE_Z = R30["line"]["p0_base_m"][2]


def save(fig, name, side):
    p = FIG / f"t32-h4-{name}.png"
    fig.savefig(p, dpi=120); plt.close(fig)
    side = {"figure": str(p.relative_to(_ag3s_workspace_path(''))), "verify_json": "AG3S/ag3s/docs/handoff/T32.h4.verify.json",
            "script": "outputs/verify/T32/h4/figs.py", **side}
    p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float))
    print(p)


def crate_boxes_xz():
    """GT crate boxes intersected with y = crate centre y (base frame; crate unrotated to 1e-3)."""
    c = CRATE
    return [("wall_nx", c[0] - 0.086, c[2] + 0.004, 0.004, 0.056), ("wall_px", c[0] + 0.086, c[2] + 0.004, 0.004, 0.056),
            ("floor", c[0], c[2] - 0.056, 0.090, 0.004)]


def draw_crate(ax, s_axis=False, color=None, lw=1.2):
    for n, x, z, hx, hz in crate_boxes_xz():
        x0 = x - (CRATE[0] if s_axis else 0)
        ax.add_patch(Rectangle((x0 - hx, z - hz), 2 * hx, 2 * hz, fill=False, ec=color or C[1], lw=lw))


# ------------------------------------------------------------------ 1. layout
def fig_layout():
    import mujoco
    rows = {int(x["seq"]): x for x in (json.loads(l) for l in open(_ag3s_workspace_path('outputs/verify/T30/E3b_rec/E3b_ep1807_r1/frames.jsonl')) if l.strip()) if x.get("kind") == "planning"}
    m = mujoco.MjModel.from_xml_path(_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')); d = mujoco.MjData(m)
    d.qpos[:] = rows[31]["qpos"]; mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 480, 640); r.update_scene(d, camera="zed_left"); rgb = r.render().copy()
    rays = json.loads((H4 / "real" / "camera_rays_seq031.json").read_text())
    z = np.load(H4 / "real" / "frames_E3b_ep1807_r1" / "seq031.npz")
    K, T = z["K/zed_left"], z["T/zed_left"]
    s = np.linspace(-0.15, 0.15, 61); P = CRATE[None] + np.stack([s, 0 * s, 0 * s + 0.004], 1)
    pc = (P - T[:3, 3]) @ T[:3, :3]; uv = pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.2), gridspec_kw={"width_ratios": [1.25, 1.2, 1.0]})
    ax[0].imshow(rgb); ax[0].plot(uv[:, 0], uv[:, 1], "-", color=C[3], lw=2.5)
    ax[0].plot(uv[[0, -1], 0], uv[[0, -1], 1], "o", color=C[3], ms=5)
    ax[0].set_title("zed_left view, E3b_ep1807_r1 seq 31 (chunk 30, t=240)\nyellow: measurement line (base-x, +-150 mm, z = crate centre + 4 mm)", fontsize=9)
    ax[0].axis("off")
    a = ax[1]
    draw_crate(a)
    a.plot([CRATE[0] - 0.15, CRATE[0] + 0.15], [LINE_Z] * 2, "-", color=C[3], lw=2, label="measurement line")
    a.add_patch(Rectangle((0.40, 0.78), 0.6, 0.04, fill=True, fc=fs.GRID_INK, ec="none", alpha=0.6))
    for i, cam in enumerate(("zed_left", "wrist_cam_l", "wrist_cam_r")):
        o = np.array(rays[cam]["pos_base_m"]); col = C[[0, 2, 4][i]]
        a.plot(o[0], o[2], "^", color=col, ms=9, label=cam)
        for face in ("nx_outer_face", "px_inner_face"):
            du = np.array(rays[cam]["rays"][face]["ray_unit"]); L = rays[cam]["rays"][face]["range_m"]
            e = o + du * L
            a.plot([o[0], e[0]], [o[2], e[2]], "--", color=col, lw=0.8)
    a.set_xlim(0.05, 0.75); a.set_ylim(0.75, 1.5); a.set_aspect("equal")
    a.set_xlabel("x (m, base)"); a.set_ylabel("z (m, base)")
    a.set_title("side view (xz at crate centre y): GT crate walls 8 mm each (orange),\ncamera centres at seq 31 and rays to the nx outer / px inner face on the line", fontsize=9)
    a.legend(fontsize=8, loc="upper right", frameon=False)
    # synthetic layout
    b = ax[2]; g = N["synthetic"]["camera_geometry"]; x0 = N["synthetic"]["wall_x0_m"]
    b.add_patch(Rectangle((x0 - 0.008, 0.828), 0.016, 0.112, fill=True, fc=C[1], alpha=0.5, ec=C[1]))
    b.add_patch(Rectangle((0.30, 0.78), 0.7, 0.04, fill=True, fc=fs.GRID_INK, ec="none", alpha=0.6))
    hp = np.array(g["head_pos_base_m"]); D = g["distance_to_wall_centre_m"]
    mp = hp.copy(); mp[0] = 2 * x0 - hp[0]
    for p, lab, col in ((hp, "head_1side (zed_left pose)", C[0]), (mp, "mirror (head_2side)", C[6]),
                        (np.array([x0 - D, 0, LINE_Z]), "normal_1side", C[2]), (np.array([x0 + D, 0, LINE_Z]), "mirror (normal_2side)", C[5])):
        b.plot(p[0], p[2], "^", color=col, ms=9, label=lab); b.plot([p[0], x0], [p[2], LINE_Z], "--", color=col, lw=0.8)
    b.plot([x0 - 0.15, x0 + 0.15], [LINE_Z] * 2, "-", color=C[3], lw=2)
    b.set_xlim(-0.3, 1.5); b.set_ylim(0.6, 1.6); b.set_aspect("equal"); b.set_xlabel("x (m, base)"); b.set_ylabel("z (m, base)")
    b.set_title(f"synthetic wall (16 or 8 mm) at GT crate_wall_px x = {x0:.4f} m\nfour camera views, same K (fx = fy = 240, 640x480)", fontsize=9)
    b.legend(fontsize=7.5, loc="upper left", frameon=False)
    fs.style_axes(fig, list(ax[1:]))
    fig.tight_layout()
    save(fig, "layout", {"line_uv_endpoints_px": uv[[0, -1]].tolist(), "camera_rays": rays, "synthetic_camera_geometry": g,
                         "crate_centre_base_m": CRATE.tolist(), "line_z_m": LINE_Z})


# ------------------------------------------------------------------ 2. profiles
def fig_profiles():
    zr = np.load(H4 / "real" / "rebuild_E3b_ep1807_r1_chunk00030.npz")
    zs = np.load(H4 / "synth" / "synth_all.npz")
    rec = {}
    import sys as _s; _s.path.insert(0, str(H4))
    from h4_common import layer_from_arrays
    z = np.load(_ag3s_workspace_path('outputs/verify/T30/E3b_rec/server_8206/server_constraints/run_0000/chunk_00030.npz'))
    s = zr["server/fine"][0]; P = np.asarray(R30["line"]["p0_base_m"])[None] + s[:, None] * np.array([1, 0, 0])[None]
    for n, key, o, vs in (("fine", "esdf_distance", "esdf_origin", "esdf_voxel_size"), ("coarse", "esdf_coarse_distance", "esdf_coarse_origin", "esdf_coarse_voxel_size")):
        L = layer_from_arrays(z[key].astype(np.float32), z[o], float(z[vs])); rec[n] = np.asarray(L.distance(P))
    fig, ax = plt.subplots(3, 1, figsize=(13, 13), sharex=False)

    def walls(a, offs=(-0.086, 0.086), t=0.008):
        for c0 in offs:
            a.axvspan(1000 * (c0 - t / 2), 1000 * (c0 + t / 2), color=C[1], alpha=0.25, lw=0)
        a.axhline(0, color=fs.INK, lw=0.8)
    a = ax[0]; walls(a)
    a.plot(1000 * s, 1000 * rec["fine"], color=C[0], lw=2.2, alpha=0.5, label="recorded fine 5 mm")
    a.plot(1000 * s, 1000 * zr["server/fine"][1], color=C[0], lw=1, ls="--", label="rebuilt fine (server settings)")
    a.plot(1000 * s, 1000 * rec["coarse"], color=C[2], lw=2.2, alpha=0.5, label="recorded coarse 20 mm")
    a.plot(1000 * s, 1000 * zr["server/coarse"][1], color=C[2], lw=1, ls="--", label="rebuilt coarse")
    ts = zr["server/tsdf"]; a.plot(1000 * ts[0], 1000 * ts[1], color=C[6], lw=1.2, label="TSDF sdf (5 mm voxel, trunc 60 mm)")
    a.set_ylim(-70, 90); a.set_xlim(-150, 150); a.set_ylabel("value (mm)")
    a.set_title("E3b_ep1807_r1 chunk 30: values along the line (s = x - crate centre x); orange = GT walls nx / px (8 mm each)", fontsize=10)
    a.legend(fontsize=8, ncol=3, frameon=False, loc="upper center")
    a = ax[1]; walls(a)
    for i, tn in enumerate(("trunc_20mm", "trunc_30mm", "trunc_40mm", "server", "trunc_80mm")):
        v = zr[f"{tn}/fine"]; lab = {"server": "trunc 60 mm (server)"}.get(tn, tn.replace("_", " "))
        a.plot(1000 * v[0], 1000 * v[1], color=fs.SEQ_BLUE[min(len(fs.SEQ_BLUE) - 1, 2 + 2 * i)], lw=1.4, label=f"fine, {lab}")
    a.set_ylim(-60, 60); a.set_xlim(-150, 150); a.set_ylabel("fine d (mm)")
    a.set_title("chunk 30 rebuilt fine layer, TSDF truncation varied (all else server)", fontsize=10); a.legend(fontsize=8, ncol=3, frameon=False)
    a = ax[2]; walls(a, offs=(0.0,), t=0.016)
    for i, vn in enumerate(("head_1side", "head_2side", "normal_1side", "normal_2side")):
        v = zs[f"matrix/T16/crate_h/{vn}/fine"]; a.plot(1000 * v[0], 1000 * v[1], color=C[[0, 6, 2, 5][i]], lw=1.4, label=f"fine, {vn}")
        t = zs[f"matrix/T16/crate_h/{vn}/tsdf"]; a.plot(1000 * t[0], 1000 * t[1], color=C[[0, 6, 2, 5][i]], lw=0.8, ls=":", label=f"TSDF, {vn}")
    a.set_ylim(-70, 70); a.set_xlim(-150, 150); a.set_xlabel("s along base-x (mm)"); a.set_ylabel("value (mm)")
    a.set_title("synthetic 16 mm wall (orange), server settings, 31 identical frames; camera(s) on -x side for *_1side", fontsize=10)
    a.legend(fontsize=8, ncol=4, frameon=False)
    fs.style_axes(fig, list(ax)); fig.tight_layout()
    srv = [r for r in R30["settings"] if r["setting"] == "server"][0]
    save(fig, "profiles", {"recorded_band_mm": R30["recorded"], "rebuilt_server": {"fine": srv["fine"]["width_lt0_mm"], "coarse": srv["coarse"]["width_lt0_mm"], "tsdf_lt0": srv["tsdf_lt0"]["width_mm"]},
                           "synthetic_T16": {r["setting"]: {"fine": r["fine"].get("width_lt0_mm"), "tsdf_lt0": r["tsdf_lt0"]["width_mm"]} for r in N["synthetic"]["settings"] if r["setting"].startswith("matrix/T16/crate_h")}})


# ------------------------------------------------------------------ 3. slices
def fig_slices():
    z = np.load(H4 / "real" / "slices_chunk00030.npz"); xs, zs = z["xs"], z["zs"]
    panels = [("recorded_fine", "recorded fine 5 mm (server)"), ("server_fine", "rebuilt fine, server (trunc 60 mm)"),
              ("trunc_40mm_fine", "rebuilt fine, trunc 40 mm"), ("trunc_20mm_fine", "rebuilt fine, trunc 20 mm"),
              ("recorded_coarse", "recorded coarse 20 mm"), ("server_tsdf", "TSDF sdf, server (trunc 60 mm)"),
              ("trunc_20mm_tsdf", "TSDF sdf, trunc 20 mm"), ("cams_zed_left_fine", "rebuilt fine, zed_left only")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
    side = {"y_m": float(CRATE[1]), "panels": {}}
    ext = [xs[0], xs[-1], zs[0], zs[-1]]
    for a, (k, t) in zip(axes.ravel(), panels):
        V_ = z[k]
        a.imshow(np.clip(V_, 0, 0.1), origin="lower", extent=ext, cmap=fs.sequential_cmap(), vmin=0, vmax=0.1, interpolation="nearest")
        a.imshow(np.where(V_ < 0, 1.0, np.nan), origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([C[7]]), alpha=0.6, interpolation="nearest")
        draw_crate(a, color=C[1], lw=1.4)
        a.axhline(LINE_Z, color=C[3], lw=1.0)
        a.set_title(t, fontsize=9); a.set_xlabel("x (m, base)"); a.set_ylabel("z (m, base)"); a.set_aspect("equal")
        row = np.argmin(np.abs(zs - LINE_Z)); line = V_[row]
        side["panels"][k] = {"title": t, "n_lt0": int(np.sum(V_ < 0)), "n_finite": int(np.isfinite(V_).sum()),
                             "line_row_width_lt0_mm": float(np.sum(line < 0) * 1.0)}
    fig.suptitle("xz slices at crate centre y, E3b_ep1807_r1 chunk 30 (red: value < 0; blue: 0-100 mm; orange: GT crate boxes; yellow: measurement line)", fontsize=11)
    fs.style_axes(fig, list(axes.ravel())); fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "slices", side)


# ------------------------------------------------------------------ 4. band width vs parameter
def fig_graph():
    def get(rows, name, tier):
        r = [x for x in rows if x["setting"] == name]
        if not r or "error" in r[0]:
            return np.nan
        if tier == "tsdf":
            return r[0]["tsdf_lt0"]["width_mm"]
        return r[0][tier].get("width_lt0_mm", np.nan)
    side = {}
    fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
    # (a) truncation, real chunks
    a = ax[0, 0]; tr = [10, 15, 20, 30, 40, 60, 80]
    names = {10: "trunc_10mm", 15: "trunc_15mm", 20: "trunc_20mm", 30: "trunc_30mm", 40: "trunc_40mm", 60: "server", 80: "trunc_80mm"}
    rows = R30["settings"]
    for tier, col, mk in (("fine", C[0], "o"), ("coarse", C[2], "s"), ("tsdf", C[6], "^")):
        y = [get(rows, names[t], tier) for t in tr]; a.plot(tr, y, mk + "-", color=col, label=f"chunk 30 {tier if tier != 'tsdf' else 'TSDF<0'}")
        side[f"trunc_chunk30_{tier}"] = dict(zip(tr, y))
    for k, ls in (("16", ":"), ("20", "--"), ("36", "-.")):
        rr = N.get(f"real_chunk{k}", {}).get("settings", [])
        t2 = [20, 30, 40, 60, 80]
        y = [get(rr, names[t], "fine") for t in t2]; a.plot(t2, y, ls, color=C[0], lw=1, label=f"chunk {k} fine")
        side[f"trunc_chunk{k}_fine"] = dict(zip(t2, y))
    a.axhline(16, color=C[1], lw=1.2, ls="--", label="GT solid on line (2 x 8 mm)")
    a.axvline(60, color=fs.INK_2, lw=0.8, ls=":")
    a.set_xlabel("TSDF truncation distance (mm)  [server: 3 x voxel_size 20 mm = 60]"); a.set_ylabel("d<0 width on line (mm)")
    a.set_title("(a) real chunk 30 (and 16/20/36): truncation", fontsize=10); a.legend(fontsize=7.5, frameon=False)
    # (b) truncation, synthetic
    a = ax[0, 1]
    for vn, col in (("head_1side", C[0]), ("normal_1side", C[2]), ("head_2side", C[6])):
        srows = N["synthetic"]["settings"]
        def sget(n, tier):
            r = [x for x in srows if x["setting"] == n]
            if not r or "error" in r[0]: return np.nan
            return r[0]["tsdf_lt0"]["width_mm"] if tier == "tsdf" else r[0][tier].get("width_lt0_mm", np.nan)
        y = [sget(f"sweep/{vn}/trunc_{t}mm", "fine") for t in tr]; a.plot(tr, y, "o-", color=col, label=f"{vn} fine")
        y2 = [sget(f"sweep/{vn}/trunc_{t}mm", "tsdf") for t in tr]; a.plot(tr, y2, "^:", color=col, lw=1, label=f"{vn} TSDF<0")
        side[f"synth_trunc_{vn}"] = {"fine": dict(zip(tr, y)), "tsdf": dict(zip(tr, y2))}
    a.axhline(16, color=C[1], lw=1.2, ls="--", label="wall 16 mm")
    a.set_xlabel("TSDF truncation distance (mm)"); a.set_ylabel("d<0 width on line (mm)")
    a.set_title("(b) synthetic 16 mm wall: truncation", fontsize=10); a.legend(fontsize=7.5, frameon=False)
    # (c) voxel sizes (chunk 30)
    a = ax[0, 2]
    tv = [2.5, 4, 5, 7.5, 10, 20]; tn = {2.5: "tsdf_2.5mm", 4: "tsdf_4mm", 5: "server", 7.5: "tsdf_7.5mm", 10: "tsdf_10mm", 20: "tsdf_20mm"}
    for tier, col, mk in (("fine", C[0], "o"), ("coarse", C[2], "s"), ("tsdf", C[6], "^")):
        y = [get(rows, tn[t], tier) for t in tv]; a.plot(tv, y, mk + "-", color=col, label=f"TSDF voxel sweep: {tier if tier != 'tsdf' else 'TSDF<0'}")
        side[f"tsdf_voxel_{tier}"] = dict(zip(tv, y))
    fv = {5: "server", 10: "fine_10mm"}; y = [get(rows, fv[t], "fine") for t in (5, 10)]
    a.plot([5, 10], y, "D--", color=C[4], label="fine ESDF voxel 5 / 10 mm: fine"); side["fine_esdf_voxel_fine"] = dict(zip((5, 10), y))
    cv = {10: "coarse_10mm_trunc60", 20: "server", 40: "coarse_40mm_trunc60"}; y = [get(rows, cv[t], "coarse") for t in (10, 20, 40)]
    a.plot([10, 20, 40], y, "P--", color=C[5], label="coarse ESDF voxel 10/20/40 mm (trunc 60): coarse"); side["coarse_esdf_voxel_trunc60_coarse"] = dict(zip((10, 20, 40), y))
    cvc = {10: "coarse_10mm_coupled", 20: "server", 40: "coarse_40mm_coupled"}; y = [get(rows, cvc[t], "coarse") for t in (10, 20, 40)]
    a.plot([10, 20, 40], y, "X:", color=C[7], label="coarse ESDF voxel, trunc = 3 x voxel: coarse"); side["coarse_esdf_voxel_coupled_coarse"] = dict(zip((10, 20, 40), y))
    a.axhline(16, color=C[1], lw=1.2, ls="--")
    a.set_xlabel("voxel size (mm)"); a.set_ylabel("d<0 width on line (mm)"); a.set_title("(c) chunk 30: TSDF voxel / ESDF voxel", fontsize=10)
    a.legend(fontsize=7.5, frameon=False)
    # (d) categorical, chunk 30
    a = ax[1, 0]
    cats = ["server", "thr_0", "thr_1", "thr_3", "thr_1e+06", "no_attached", "seeding_scatter", "minw_1", "minw_10", "frames_last1", "frames_last5",
            "cams_zed_left", "cams_wrist_cam_l+wrist_cam_r", "cams_zed_left+wrist_cam_l", "cams_zed_left+wrist_cam_r"]
    labs = ["server", "sign thr 0", "sign thr 1.0", "sign thr 3.0", "sign thr off", "no attached pts", "seeding scatter", "min weight 1", "min weight 10",
            "last 1 frame", "last 5 frames", "zed_left only", "wrists only", "zed_left+wrist_l", "zed_left+wrist_r"]
    xx = np.arange(len(cats)); w = 0.27
    for j, (tier, col) in enumerate((("fine", C[0]), ("coarse", C[2]), ("tsdf", C[6]))):
        y = [get(rows, c, tier) for c in cats]; a.bar(xx + (j - 1) * w, y, w, color=col, label=tier if tier != "tsdf" else "TSDF<0")
        side[f"categorical_{tier}"] = dict(zip(cats, y))
    a.axhline(16, color=C[1], lw=1.2, ls="--"); a.set_xticks(xx); a.set_xticklabels(labs, rotation=60, ha="right", fontsize=8)
    a.set_ylabel("d<0 width on line (mm)"); a.set_title("(d) chunk 30: sign rule / seeding / weight / frames / cameras", fontsize=10); a.legend(fontsize=8, frameon=False)
    # (e) synthetic views and thickness
    a = ax[1, 1]
    vns = ["head_1side", "head_2side", "normal_1side", "normal_2side"]
    for j, (T, col) in enumerate(((16, C[0]), (8, C[4]))):
        srows = {x["setting"]: x for x in N["synthetic"]["settings"]}
        y = [srows[f"matrix/T{T}/crate_h/{v}"]["fine"]["width_lt0_mm"] for v in vns]
        y2 = [srows[f"matrix/T{T}/crate_h/{v}"]["tsdf_lt0"]["width_mm"] for v in vns]
        a.bar(np.arange(4) + (j - 0.5) * 0.38, y, 0.36, color=col, label=f"{T} mm wall: fine d<0")
        a.plot(np.arange(4) + (j - 0.5) * 0.38, y2, "^", color=fs.INK, ms=6, label="TSDF<0" if j == 0 else None)
        side[f"synth_views_T{T}"] = {"fine": dict(zip(vns, y)), "tsdf": dict(zip(vns, y2))}
    a.axhline(16, color=C[0], lw=1, ls="--"); a.axhline(8, color=C[4], lw=1, ls="--")
    a.set_xticks(range(4)); a.set_xticklabels(vns); a.set_ylabel("width on line (mm)")
    a.set_title("(e) synthetic: views x thickness (server settings)", fontsize=10); a.legend(fontsize=8, frameon=False)
    # (f) chunks: recorded vs rebuilt
    a = ax[1, 2]
    ks = [16, 20, 30, 36]
    for j, (tier, col) in enumerate((("fine", C[0]), ("coarse", C[2]))):
        rec = [N[f"real_chunk{k}"]["recorded"][tier]["width_lt0_mm"] for k in ks]
        reb = [get(N[f"real_chunk{k}"]["settings"], "server", tier) for k in ks]
        a.plot(ks, rec, "o-", color=col, lw=2.5, alpha=0.45, label=f"recorded {tier}"); a.plot(ks, reb, "x--", color=col, label=f"rebuilt {tier}")
        side[f"chunks_{tier}"] = {"recorded": dict(zip(ks, rec)), "rebuilt": dict(zip(ks, reb))}
    a.axhline(16, color=C[1], lw=1.2, ls="--"); a.set_xlabel("chunk (t = 8 x chunk)"); a.set_ylabel("d<0 width on line (mm)")
    a.set_title("(f) recorded vs rebuilt (server settings), 4 chunks", fontsize=10); a.legend(fontsize=8, frameon=False)
    fs.style_axes(fig, list(ax.ravel())); fig.tight_layout()
    save(fig, "bandwidth-vs-param", side)


# ------------------------------------------------------------------ 5. table
def fig_table():
    rows = R30["settings"]
    order = ["server", "trunc_10mm", "trunc_15mm", "trunc_20mm", "trunc_30mm", "trunc_40mm", "trunc_80mm", "tsdf_2.5mm", "tsdf_4mm", "tsdf_7.5mm", "tsdf_10mm", "tsdf_20mm",
             "fine_2.5mm", "fine_10mm", "coarse_10mm_trunc60", "coarse_40mm_trunc60", "coarse_10mm_coupled", "coarse_40mm_coupled",
             "thr_0", "thr_1", "thr_3", "thr_1e+06", "no_attached", "seeding_scatter", "minw_1", "minw_10", "frames_last1", "frames_last5",
             "cams_zed_left", "cams_wrist_cam_l+wrist_cam_r", "cams_zed_left+wrist_cam_l", "cams_zed_left+wrist_cam_r"]
    by = {r["setting"]: r for r in rows}
    cells = [["recorded (server)", "60", "5", "5 / 20", "31 x 3", *[f"{R30['recorded']['fine']['width_lt0_mm']:.1f}", f"{R30['recorded']['fine']['width_lt0_mm_lo']:.1f} / {R30['recorded']['fine']['width_lt0_mm_hi']:.1f}",
              f"{R30['recorded']['coarse']['width_lt0_mm']:.1f}", f"{R30['recorded']['coarse']['width_lt0_mm_lo']:.1f} / {R30['recorded']['coarse']['width_lt0_mm_hi']:.1f}", "-"]]]
    for n in order:
        r = by.get(n)
        if r is None:
            continue
        f, c = r["fine"], r["coarse"]
        fw = f"{f['width_lt0_mm']:.1f}" if "width_lt0_mm" in f else "not covered"
        fs_ = f"{f['nx_mm']:.1f} / {f['px_mm']:.1f}" if "nx_mm" in f else "-"
        cw = f"{c['width_lt0_mm']:.1f}" if "width_lt0_mm" in c else "-"
        cs = f"{c['nx_mm']:.1f} / {c['px_mm']:.1f}" if "nx_mm" in c else "-"
        cells.append([n, f"{r['truncation_mm']:g}", f"{r['tsdf_voxel_mm']:g}", f"{r['fine_voxel_mm'] or 0:g} / {r['coarse_voxel_mm']:g}",
                      f"{r['n_frames']} x {r['n_cameras']}", fw, fs_, cw, cs, f"{r['tsdf_lt0']['width_mm']:.1f}"])
    dec = R30["sum_decomposition_server"]
    fig, ax = plt.subplots(figsize=(16, 11.5)); ax.axis("off")
    cols = ["setting (chunk 30)", "trunc mm", "TSDF vox mm", "ESDF vox fine / coarse", "frames x cams", "fine d<0 mm", "fine nx / px", "coarse d<0 mm", "coarse nx / px", "TSDF<0 mm"]
    tb = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center",
                  colWidths=[0.2, 0.06, 0.07, 0.11, 0.08, 0.08, 0.1, 0.08, 0.1, 0.08])
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.32)
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor(fs.GRID_INK)
        if i == 0: cell.set_facecolor(fs.SEQ_BLUE[0])
        if i in (1, 2): cell.set_facecolor("#fff3d6")
    txt = (f"GT on the line: crate_wall_nx 8 mm + crate_wall_px 8 mm = 16 mm.   Server-setting sum along the line (chunk 30):\n"
           f"TSDF<0 width {dec['tsdf_lt0_width_mm']:.1f} mm = 16 + {dec['step1_tsdf_lt0_minus_gt_mm']:.1f} (integration, trunc 60 mm);  "
           f"fine d<0 = TSDF<0 {dec['step2_fine_esdf_lt0_minus_tsdf_lt0_mm']:+.1f} = {dec['fine_total_mm']:.1f} mm;  coarse d<0 = TSDF<0 {dec['step2_coarse_esdf_lt0_minus_tsdf_lt0_mm']:+.1f} = {dec['coarse_total_mm']:.1f} mm;  attached sign rule: 0.0\n"
           "truncation = esdf.truncation_voxels (3.0) x esdf.voxel_size (0.020 coarse ESDF voxel) -> MapperCfg.truncation_distance 0.060 m (ag3s/config.py:867, curobo_builder.py:173)\n"
           "sign: d < 0 iff TSDF(sdf_sum/weight) at the ESDF voxel centre < 0 (builder_esdf.py compute_esdf_from_min_tsdf_kernel); TSDF: sdf = depth - z_cam kept for sdf >= -truncation")
    ax.text(0.0, -0.02, txt, transform=ax.transAxes, fontsize=9, va="bottom", family="monospace")
    ax.set_title("T32 H4: d<0 band width on the crate wall line, one parameter varied at a time (E3b_ep1807_r1 chunk 30, t = 240)", fontsize=11)
    save(fig, "table", {"columns": cols, "rows": cells, "sum_decomposition": dec})


if __name__ == "__main__":
    fig_layout(); fig_profiles(); fig_slices(); fig_graph(); fig_table()
