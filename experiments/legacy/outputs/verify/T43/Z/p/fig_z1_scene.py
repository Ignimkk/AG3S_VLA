"""T43 Z1 figure t43z-held-sphere-scene — MuJoCo re-render (recorded qpos of the control row at the chunk start) of ep1925 s19253,
V4 at t 216 (V4 policy open chunk, executed) and V5 at t 304 (HOLD chunk, apple centre 80 mm above the crate origin), oblique front (azimuth 180, elevation -32) + top close-ups,
with the server's held query spheres (raw/server_chunks.json of T43 Y: held_query.centre_world_now, radii) drawn solid and their
r + 20 mm destination-margin shells drawn faint. True crate signed distance per sphere (Y server_y.py, MuJoCo boxes) in the titles."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/Y/p'))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
import render_y as RY  # noqa: E402

FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
S = json.load(open(_ag3s_workspace_path('outputs/verify/T43/Y/raw/server_chunks.json')))["runs"]
A = json.load(open(_ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')))["runs"]
MG = 20.0
CASES = [("V4/H/V4_ep1925_s19253", 216), ("V5/H/V5_ep1925_s19253", 304)]
COL = [tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) for h in figstyle.CATEGORICAL[:4]]
side = {"definitions": __doc__, "margin_mm": MG, "panels": {}}
figstyle.use_korean()
fig, axes = plt.subplots(2, 2, figsize=(14, 14.6))
figstyle.style_axes(fig, axes)
for r, (key, t) in enumerate(CASES):
    rd = A[key]["run_dir"]; M, ctrl = RY.scene_for(rd, "left")
    c = next(x for x in S[key]["chunks"] if x["t"] == t); hq = c["held_query"]
    cen = np.array(hq["centre_world_now"]); rad = 1e-3 * np.array(hq["radii_mm"])
    sph = [(p, rr, (*COL[i % 4], 0.45)) for i, (p, rr) in enumerate(zip(cen, rad))] + \
          [(p, rr + 1e-3 * MG, (*COL[i % 4], 0.13)) for i, (p, rr) in enumerate(zip(cen, rad))]
    look = tuple(0.5 * (cen.mean(axis=0) + np.array([0.52, 0.0, 0.88])))
    cams = {"oblique": dict(lookat=look, distance=0.55, azimuth=180.0, elevation=-32.0), "top": dict(lookat=look, distance=0.62, azimuth=90.0, elevation=-89.0)}
    for j, (vn, cam) in enumerate(cams.items()):
        img = RY.render(M, np.array(ctrl[t]["qpos"], float), cam, sph, size=(760, 760))
        ax = axes[r, j]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{key.split('/')[0]} ep1925 s19253 · t {t} · {'HOLD' if c['hold'] else 'executed'} · {vn} view", fontsize=11, color=figstyle.INK)
        side["panels"][f"{key}@{t}_{vn}"] = {"camera": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cam.items()}}
    txt = (f"held query set '{hq['set']}': r = {', '.join('%.1f' % x for x in hq['radii_mm'])} mm (solid), r + {MG:.0f} mm shells (faint)\n"
           f"true crate signed distance of each sphere (r subtracted): {', '.join('%.1f' % x for x in hq['crate_sd_mm_now'])} mm · "
           f"apple centre in crate frame [{', '.join('%.0f' % v for v in c['apple_in_crate_mm'])}] mm · TO worst row {c['worst']['link']} vs "
           f"{c['worst']['obstacle']} {c['worst']['clearance_mm']:.1f} mm")
    axes[r, 0].text(0.0, -0.025, txt, transform=axes[r, 0].transAxes, fontsize=9, color=figstyle.INK_2, va="top")
    side["panels"][f"{key}@{t}"] = {"hold": c["hold"], "held_query": hq, "apple_in_crate_mm": c["apple_in_crate_mm"], "worst": c["worst"]}
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.04, hspace=0.12, wspace=0.02)
fig.suptitle("T43 Z1 — held query spheres in the MuJoCo scene, ep1925 s19253 (colour = sphere index 0 / 1 / 2, same as t43z-held-sphere-closeup)",
             fontsize=11.5, color=figstyle.INK)
out = FIG / "t43z-held-sphere-scene.png"
assert not out.exists() or "--overwrite" in sys.argv, out
fig.savefig(out, dpi=100, facecolor=figstyle.SURFACE)
side.update(figure=str(out), script=__file__, sources=["outputs/verify/T43/Y/raw/server_chunks.json", "outputs/verify/T43/W/analysis/analysis.json"])
json.dump(side, open(FIG / "t43z-held-sphere-scene.json", "w"), indent=1, default=float)
print(out)
