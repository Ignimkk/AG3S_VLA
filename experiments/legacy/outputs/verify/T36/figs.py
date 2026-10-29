"""T36-baseline figures (rule A): scene / trend / table, each with a .json sidecar of the numbers it draws.
usage: <py with matplotlib> outputs/verify/T36/figs.py"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs

T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T36'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t36'))
FIG.mkdir(parents=True, exist_ok=True)
fs.use_korean()
C = fs.CATEGORICAL
MARGIN = 0.05

leg = json.loads((T / "legacy/legacy_shim.json").read_text())["frames"]
leg2 = json.loads((T / "legacy/legacy_openpilive.json").read_text())["frames"]
runs = {k: json.loads((T / f"curobo/curobo_{k}.json").read_text())["frames"] for k in ("r1", "r2", "r3", "r4")}
ovl = {k: json.loads((T / f"curobo/curobo_{k}.json").read_text())["frames"] for k in ("r0_overlap", "r0b_overlap")}
n = len(leg)
fr = np.arange(n)
L = np.array([r["clearance_before_mm"] for r in leg])
CU = {k: np.array([r["clearance_before_mm"] for r in v]) for k, v in runs.items()}
phase = [r["phase"] for r in leg]
sl = np.load(T / "scene/scene_legacy.npz")
sc = np.load(T / "scene/scene_curobo.npz")
tier = [str(sc[f"tier_{i}"][0]) for i in range(n)]


def sidecar(name, obj):
    (FIG / f"{name}.json").write_text(json.dumps(obj, indent=1))


# ---------------------------------------------------------------- 2. trend
fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
fs.style_axes(fig, axes)
ax = axes[0]
ax.axhline(0, color=fs.INK_2, lw=0.8)
ax.plot(fr, L, "-o", color=C[0], lw=1.6, ms=5, label="legacy (numpy) - 1 run (+1 control, identical)")
marks = ["s", "D", "^", "v"]
for j, (k, v) in enumerate(CU.items()):
    ax.plot(fr, v, "-" if j == 0 else "none", marker=marks[j], color=C[1], lw=1.6, ms=9 - 1.6 * j,
            mfc="none" if j else C[1], label=f"cuRobo {k}" + (" (r2-r4 bitwise equal)" if j == 0 else ""))
ax.set_ylabel("clearance_before (mm)")
ax.set_title("T36 - clearance_before per frame, run_16d_ep1800, legacy vs cuRobo x 4 runs", fontsize=10, color=fs.INK)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")
for i in fr:
    ax.text(i, 30, phase[i][:3], ha="center", fontsize=6.5, color=fs.INK_2)
ax.set_ylim(-140, 36)
ax2 = axes[1]
d = CU["r1"] - L
ax2.bar(fr, d, color=[C[1] if x > 0 else C[0] for x in d], width=0.6)
ax2.axhline(0, color=fs.INK_2, lw=0.8)
for i in fr:
    ax2.text(i, d[i] + (0.8 if d[i] >= 0 else -2.2), f"{d[i]:+.1f}", ha="center", fontsize=6.5, color=fs.INK)
flip = [i for i in fr if L[i] < 0 <= CU["r1"][i]]
for i in flip:
    ax2.text(i, -6.5, "flip", ha="center", fontsize=6.5, color=C[7])
ax2.set_ylabel("cuRobo - legacy (mm)")
ax2.set_xlabel("frame i")
ax2.set_xticks(fr)
ax2.set_ylim(-8, 24)
maxdev = max(float(np.max(np.abs(v - CU["r1"]))) for v in CU.values())
fig.text(0.99, 0.005, f"max |cuRobo rk - r1| over 15 frames x 4 runs = {maxdev:.1f} mm (bitwise)", ha="right",
         fontsize=7, color=fs.INK_2)
fig.tight_layout()
fig.savefig(FIG / "t36-trend.png", dpi=160)
plt.close(fig)
sidecar("t36-trend", {
    "source": ["outputs/verify/T36/legacy/legacy_shim.json", "outputs/verify/T36/legacy/legacy_openpilive.json"]
              + [f"outputs/verify/T36/curobo/curobo_{k}.json" for k in CU],
    "frames": fr.tolist(), "phase": phase,
    "legacy_clearance_before_mm": L.tolist(),
    "legacy_control_openpi_live_clearance_before_mm": [r["clearance_before_mm"] for r in leg2],
    **{f"curobo_{k}_clearance_before_mm": v.tolist() for k, v in CU.items()},
    "curobo_minus_legacy_mm": d.tolist(),
    "frames_legacy_negative_curobo_nonnegative": [int(i) for i in flip],
    "max_abs_curobo_rk_minus_r1_mm": maxdev})

# ---------------------------------------------------------------- 3. table
cols = ["i", "phase", "legacy", "cuRobo r1", "r2", "r3", "r4", "cuRobo-legacy", "leg<0", "cu<0", "cu tier"]
rows = []
for i in fr:
    rows.append([str(i), phase[i], f"{L[i]:+.4f}"] + [f"{CU[k][i]:+.4f}" for k in CU]
                + [f"{d[i]:+.4f}", "Y" if L[i] < 0 else "", "Y" if CU["r1"][i] < 0 else "", tier[i]])
summ = [["start violated", "", f"{int((L<0).sum())}/15"] + [f"{int((CU[k]<0).sum())}/15" for k in CU] + ["", "", "", ""],
        ["has_target", "", f"{sum(bool(r['target']) for r in leg)}/15"]
        + [f"{sum(bool(r['target']) for r in runs[k])}/15" for k in CU] + ["", "", "", ""]]
fig, ax = plt.subplots(figsize=(11.5, 5.6))
fs.style_axes(fig, ax)
ax.axis("off")
tb = ax.table(cellText=rows + summ, colLabels=cols, loc="upper center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(7.6)
tb.scale(1, 1.28)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor(fs.GRID_INK)
    cell.set_facecolor(fs.SURFACE)
    if r == 0:
        cell.set_text_props(color=fs.INK, weight="bold")
    elif r <= n and c == 7 and i in fr:
        pass
    if 1 <= r <= n and c in (9,) and rows[r - 1][9] == "" and rows[r - 1][8] == "Y":
        cell.set_facecolor("#fbe3d8")
    if r > n:
        cell.set_facecolor("#eef3fb")
ax.set_title("T36 - per-frame clearance_before (mm), legacy vs cuRobo x 4 runs  (command: skill baseline"
             " + --esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005)", fontsize=8.5, color=fs.INK)
foot = (f"frame0 clearance_before: legacy {float(L[0])!r} mm | cuRobo r1..r4 {float(CU['r1'][0])!r} mm (all 4 equal)\n"
        f"cuRobo r1..r4: every per-frame key except ag3s_ms/to_ms bitwise equal; field-level sha1 of all tiers equal in 2 extra AG3S-only processes\n"
        f"'cu tier' = which cuRobo tier answers at the worst sphere centre (answer_tier). Shaded = legacy<0 but cuRobo>=0 (frames {', '.join(map(str, flip))}).\n"
        f"cuRobo-path features exercised in this rollout: separate fine TSDF (fine 30 mm / coarse 60 mm truncation) = yes; "
        f"held-trace freeing = 0 points; target-free layer = 0 layers (policy relax); attached = 0")
fig.text(0.02, -0.06, foot, fontsize=7, color=fs.INK_2, va="bottom")
fig.savefig(FIG / "t36-table.png", dpi=160, bbox_inches="tight")
plt.close(fig)
sidecar("t36-table", {"columns": cols, "rows": rows, "summary_rows": summ,
                      "frame0_clearance_before_mm": {"legacy": float(L[0]), **{f"curobo_{k}": float(CU[k][0]) for k in CU}},
                      "shaded_frames": [int(i) for i in flip]})

# ---------------------------------------------------------------- 1. scene
g = sl["grid_offsets"]
SHOW = (0, 4)
fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
fs.style_axes(fig, axes)
cmap = fs.sequential_cmap()
side = {}
for row, i in enumerate(SHOW):
    depth = sc[f"depth_{i}"]; K = sc[f"K_{i}"]; Tbc = sc[f"T_{i}"]
    ax = axes[row, 0]
    ax.imshow(np.where(depth > 0, depth, np.nan), cmap=cmap, vmin=0.3, vmax=2.0)
    for nm, z, col in (("legacy", sl, C[0]), ("cuRobo", sc, C[1])):
        w = z[f"worst_{i}"]; p = w[2:5]
        pc = np.linalg.inv(Tbc) @ np.r_[p, 1.0]
        u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
        rpx = K[0, 0] * w[5] / pc[2]
        ax.add_patch(Circle((u, v), rpx, fill=False, ec=col, lw=1.6, ls="-" if nm == "legacy" else "--"))
        ax.plot(u, v, "+", color=col, ms=8)
    ax.set_title(f"frame {i} ({phase[i]}) - zed_left depth, worst sphere\n(solid legacy, dashed cuRobo)",
                 fontsize=8.5, color=fs.INK)
    ax.set_xticks([]); ax.set_yticks([])
    for col_i, (nm, z) in enumerate((("legacy", sl), ("cuRobo", sc)), start=1):
        ax = axes[row, col_i]
        w = z[f"worst_{i}"]; c = w[2:5]; r = w[5]
        S = z[f"slicexz_{i}"]
        ext = [c[0] + g[0], c[0] + g[-1], c[2] + g[0], c[2] + g[-1]]
        im = ax.imshow(np.clip(S, -0.05, 0.25), origin="lower", extent=ext, cmap=cmap, vmin=-0.05, vmax=0.25)
        X, Y = np.meshgrid(c[0] + g, c[2] + g, indexing="xy")
        ax.contour(X, Y, S, levels=[0.0], colors=[fs.INK], linewidths=1.2)
        ax.contour(X, Y, S, levels=[r + MARGIN], colors=[C[7]], linewidths=0.9, linestyles="dotted")
        ax.add_patch(Circle((c[0], c[2]), r, fill=False, ec=C[1] if nm == "cuRobo" else C[0], lw=1.6))
        ax.add_patch(Circle((c[0], c[2]), r + MARGIN, fill=False, ec=fs.INK_2, lw=0.8, ls="--"))
        ax.plot(c[0], c[2], "+", color=fs.INK)
        t = f" tier={tier[i]}" if nm == "cuRobo" else ""
        ax.set_title(f"{nm}: x-z slice at y={c[1]:+.3f} m\n{str(z[f'link_{i}'])} step {int(w[0])}, "
                     f"clearance {w[6]*1000:+.3f} mm{t}", fontsize=8, color=fs.INK)
        ax.set_xlabel("x base (m)"); ax.set_ylabel("z base (m)")
        side[f"frame{i}_{nm}"] = {"link": str(z[f"link_{i}"]), "step": int(w[0]), "sphere": int(w[1]),
                                  "centre_m": c.tolist(), "radius_m": float(r),
                                  "clearance_mm": float(w[6] * 1000), "full_violation_mm": float(w[7] * 1000),
                                  "d_at_centre_m": float(S[len(g) // 2, len(g) // 2]), "slice_plane": "x-z at y = centre y",
                                  **({"answer_tier": tier[i]} if nm == "cuRobo" else {})}
cb = fig.colorbar(im, ax=axes[:, 1:], shrink=0.6, label="ESDF distance (m), clipped [-0.05, 0.25]")
fig.suptitle("T36 - where the frame-0 / frame-4 clearance_before comes from (black: d=0, red dotted: d=r+50 mm; "
             "circles: sphere r and r+margin)", fontsize=9.5, color=fs.INK)
fig.savefig(FIG / "t36-scene.png", dpi=150, bbox_inches="tight")
plt.close(fig)
sidecar("t36-scene", {"source": ["outputs/verify/T36/scene/scene_legacy.npz", "outputs/verify/T36/scene/scene_curobo.npz"],
                      "slice_half_width_m": float(-g[0]), "slice_res_m": float(g[1] - g[0]), "margin_m": MARGIN,
                      "frames": list(SHOW), "worst": side,
                      "curobo_answer_tier_per_frame": tier})
print("ok", list(FIG.iterdir()))
