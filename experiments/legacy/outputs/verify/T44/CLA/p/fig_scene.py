"""T44dA figure (a): real scene of one C1b first-divergence / pre-close chunk (C1b_ep123_s1231 seq 17 by default).
Recorded grids (float16) re-queried on CPU (field.py), FK of reference vs refined (fkmodel.py), GT fruit hulls from the planning-row object_poses.
-> AG3S/ag3s/docs/figures/t44/t44da-scene.png + .json   (run from ${AG3S_PROJECT_ROOT}-t44cl)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import os, sys, json, pathlib
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/CLA/p')); sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import fkmodel as F, field as FD
from extract_chunks import hull_sd, FRUITS, FV, _quat_R
from AG3S.ag3s.experiments.common import figstyle
NAME = sys.argv[1] if len(sys.argv) > 1 else "C1b_ep123_s1231"; SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 17
CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL')); FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t44'))
a = json.load(open(CL / "analysis" / f"{NAME}.json")); side = a["used_arm"]; s1 = side[0]; tgt = a["target_fruit"]
rows = [json.loads(l) for l in open(pathlib.Path(_ag3s_workspace_path('')) / a["run_dir"] / "frames.jsonl")]
p = [r for r in rows if r["kind"] == "planning" and r["seq"] == SEQ][0]
c = [x for x in a["server_chunks"] if x["seq"] == SEQ][0]; z = np.load(c["server_file"], allow_pickle=True); s = json.loads(str(z["summary_json"]))
sc, qm = F.model("gripper"); links = np.asarray(qm.sphere_link_names); rad = np.asarray(qm.radii)
mask = np.asarray(s["target_permission"]["target_field_exclude"], bool)
fing = np.array([l in (f"ee_finger_{s1}1", f"ee_finger_{s1}2") for l in links]); act = (fing | (links == f"ee_{side}")) & mask
pr = F.params_from_qpos(sc, qm, p["qpos"]); op = p["object_poses"]
SR = np.stack([F.spheres(qm, q, pr) for q in F.chunk_q(sc, qm, p["qpos"], np.asarray(z["reference_chunk"], np.float64)[:16])])
SF = np.stack([F.spheres(qm, q, pr) for q in F.chunk_q(sc, qm, p["qpos"], np.asarray(z["refined_chunk"], np.float64)[:16])])
tf = s["esdf_stats"]["target_free"]; tf = eval(tf) if isinstance(tf, str) else tf; ball = tf["ball"]; sz = ball["support_plane_z_m"]
bc = np.asarray(ball["centre_m"]); br = ball["radius_mm"] / 1e3
# worst authorized active row along the reference rows 0..7 (coarse ON)
X = SR[:8][:, act].reshape(-1, 3); r = np.tile(rad[act], 8); q = FD.query(z, X, True)
con = q["d_on"] - r; coff = q["d_off"] - r; i = int(np.argmin(con)); W = X[i]
row_w, sph_w = divmod(i, int(act.sum()))
y0 = W[1]
xs = np.arange(W[0] - 0.09, W[0] + 0.09, 0.0015); zs = np.arange(sz - 0.03, sz + 0.13, 0.0015)
XX, ZZ = np.meshgrid(xs, zs); P = np.stack([XX.ravel(), np.full(XX.size, y0), ZZ.ravel()], 1)
G = FD.query(z, P, True)
on = G["d_on"].reshape(XX.shape) * 1e3; off = G["d_off"].reshape(XX.shape) * 1e3; tier = G["tier_on"].reshape(XX.shape)
hT = hull_sd(op[tgt], tgt, P).reshape(XX.shape)
others = [b for b in FRUITS if b != tgt]
figstyle.use_korean()
fig, ax = plt.subplots(1, 4, figsize=(19, 5.2), gridspec_kw={"width_ratios": [1, 1.15, 1.15, 1.15]})
figstyle.style_axes(fig, ax)
C = figstyle.CATEGORICAL
# panel 0: top view layout
a0 = ax[0]
for k, b in enumerate([tgt] + others):
    V = FV[b] @ _quat_R(op[b]["quat"]).T + np.asarray(op[b]["pos"])
    from scipy.spatial import ConvexHull
    h = ConvexHull(V[:, :2]); poly = V[h.vertices, :2]
    a0.fill(poly[:, 0], poly[:, 1], color=C[1] if b == tgt else C[2 + k % 4], alpha=0.35, lw=1, ec=figstyle.INK_2)
    a0.text(op[b]["pos"][0], op[b]["pos"][1], b + (" (target)" if b == tgt else ""), fontsize=7, ha="center", color=figstyle.INK, clip_on=True)
a0.add_patch(plt.Circle(bc[:2], br, fill=False, ls="--", color=figstyle.INK_2, lw=1))
fr = SR[:8][:, fing].reshape(-1, 3); ff = SF[:8][:, fing].reshape(-1, 3)
a0.scatter(fr[:, 0], fr[:, 1], s=4, color=C[0], label="finger spheres, reference rows 0-7")
a0.scatter(ff[:, 0], ff[:, 1], s=4, color=C[3], label="finger spheres, refined rows 0-7")
a0.axhline(y0, color=C[7], lw=1.2); a0.text(W[0] - 0.14, y0 + 0.004, f"slice y = {y0:.3f} m", color=C[7], fontsize=7, clip_on=True)
a0.set_aspect("equal"); a0.set_xlim(W[0] - 0.15, W[0] + 0.15); a0.set_ylim(y0 - 0.15, y0 + 0.15)
a0.set_xlabel("x (m)"); a0.set_ylabel("y (m)"); a0.set_title("top view (x-y): GT fruit hulls, target ball (dashed), slice line", fontsize=9)
a0.legend(fontsize=6, loc="lower left")
ext = [xs[0], xs[-1], zs[0], zs[-1]]
lim = 30
def deco(axx, title):
    axx.contour(XX, ZZ, hT, levels=[0], colors=[C[1]], linewidths=1.2)
    axx.axhline(sz, color=figstyle.INK, lw=1); axx.text(xs[0] + 0.002, sz - 0.008, "support plane", fontsize=7)
    th = np.linspace(0, 2 * np.pi, 100); rr = np.sqrt(max(br ** 2 - (y0 - bc[1]) ** 2, 0))
    axx.plot(bc[0] + rr * np.cos(th), bc[2] + rr * np.sin(th), ls="--", color=figstyle.INK_2, lw=0.8)
    near = np.abs(SR[:8][:, act][..., 1] - y0) < 0.012
    for S_, col in ((SR, C[0]), (SF, C[3])):
        P8 = S_[:8][:, act]; R8 = np.tile(rad[act], (8, 1))
        for (x_, _, z_), r_ in zip(P8[near], R8[near]):
            axx.add_patch(plt.Circle((x_, z_), r_, fill=False, color=col, lw=0.6))
    axx.plot(W[0], W[2], marker="x", color=C[7], ms=8)
    axx.set_xlim(ext[0], ext[1]); axx.set_ylim(ext[2], ext[3]); axx.set_aspect("equal")
    axx.set_xlabel("x (m)"); axx.set_ylabel("z (m)"); axx.set_title(title, fontsize=9)
norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
im1 = ax[1].imshow(np.clip(on, -lim, lim), origin="lower", extent=ext, cmap="RdBu", norm=norm); ax[1].contour(XX, ZZ, on, levels=[0], colors="k", linewidths=0.6)
deco(ax[1], "target-free distance, coarse ON (C1b as served), mm")
im2 = ax[2].imshow(np.clip(off, -lim, lim), origin="lower", extent=ext, cmap="RdBu", norm=norm); ax[2].contour(XX, ZZ, off, levels=[0], colors="k", linewidths=0.6)
deco(ax[2], "same grids, coarse OFF (C1a formula), mm")
diff = off - on
im3 = ax[3].imshow(np.clip(diff, 0, 15), origin="lower", extent=ext, cmap=figstyle.sequential_cmap(), vmin=0, vmax=15)
ax[3].contour(XX, ZZ, (tier == "target_free_coarse").astype(float), levels=[0.5], colors=[C[4]], linewidths=1.0)
deco(ax[3], "OFF - ON (mm); pink line = rows answered by target_free_coarse", )
for im, axx in ((im1, ax[1]), (im2, ax[2]), (im3, ax[3])):
    fig.colorbar(im, ax=axx, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7)
lr = row_w
fig.suptitle(f"{NAME} seq {SEQ} (t {p['t_step']}) - {tgt}, {side} arm.  worst authorized row (x): ON {1e3*con[i]:.1f} mm vs OFF {1e3*coff[i]:.1f} mm,\n"
             f"tier {q['tier_on'][i]}, sphere bottom {1e3*(W[2]-r[i]-sz):.1f} mm above support, GT gap target {1e3*(hull_sd(op[tgt], tgt, W[None])[0]-r[i]):.1f} mm.  "
             "\n"
             f"Blue circles = reference, orange = refined (TO).  Orange line = GT target hull section.", fontsize=9, color=figstyle.INK)
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.8, wspace=0.28)
out = FIG / "t44da-scene.png"; fig.savefig(out, dpi=150)
mid = lambda S_: S_[:, fing].mean(1)
side_json = {"source": {"run": NAME, "seq": SEQ, "t": p["t_step"], "server_chunk": c["server_file"], "frames": str(pathlib.Path(a["run_dir"]) / "frames.jsonl")},
             "slice": {"plane": "x-z", "y_m": float(y0), "x_range_m": [float(xs[0]), float(xs[-1])], "z_range_m": [float(zs[0]), float(zs[-1])], "step_m": 0.0015},
             "support_plane_z_m": sz, "ball_centre_m": bc.tolist(), "ball_radius_mm": ball["radius_mm"],
             "worst_row": {"row": int(row_w), "point_m": W.tolist(), "radius_mm": float(1e3 * r[i]), "clearance_on_mm": float(1e3 * con[i]), "clearance_off_mm": float(1e3 * coff[i]),
                           "tier_on": str(q["tier_on"][i]), "tier_off": str(q["tier_off"][i]), "bottom_above_support_mm": float(1e3 * (W[2] - r[i] - sz)),
                           "gt_gap_target_mm": float(1e3 * (hull_sd(op[tgt], tgt, W[None])[0] - r[i]))},
             "hand_mid_row7_ref_m": mid(SR)[7].tolist(), "hand_mid_row7_refined_m": mid(SF)[7].tolist(),
             "hand_mid_row7_dz_mm": float(1e3 * (mid(SF)[7][2] - mid(SR)[7][2])),
             "authorized_active_rows_0_7": {"min_on_mm": float(1e3 * con.min()), "min_off_mm": float(1e3 * coff.min()),
                                            "n_cross10": int(((coff >= 0.010) & (con < 0.010)).sum()), "n": int(len(con))},
             "slice_cells_tier_target_free_coarse": int((tier == "target_free_coarse").sum()), "slice_cells": int(tier.size),
             "slice_off_minus_on_mm_max": float(diff.max()), "verdict_reasons": [(q_["kind"], q_["action"]) for q_ in s["verdict"]["reasons"]],
             "hold": bool(c.get("hold")), "to_status": s["to"]["status"], "script": "outputs/verify/T44/CLA/p/fig_scene.py"}
def zero_lvl(Vv):
    out = []
    for j in range(Vv.shape[1]):
        col = Vv[:, j]; k = np.where((zs >= sz) & (col > 0))[0]
        out.append(zs[k[0]] - sz if len(k) else np.nan)
    return np.asarray(out) * 1e3
zon, zoff = zero_lvl(on), zero_lvl(off)
side_json["support_zero_level_above_plane_mm"] = {"on_median": float(np.nanmedian(zon)), "off_median": float(np.nanmedian(zoff)),
    "definition": "per slice column x: lowest z >= support plane where the target-free distance > 0, minus support z; median over columns"}
side_json["note"] = "the slice passes through the worst finger sphere (y0), beside the target; the GT target hull is not cut by this plane unless an orange contour shows"
json.dump(side_json, open(FIG / "t44da-scene.json", "w"), indent=1)
print(json.dumps(side_json["worst_row"]), side_json["hand_mid_row7_dz_mm"], side_json["authorized_active_rows_0_7"])
