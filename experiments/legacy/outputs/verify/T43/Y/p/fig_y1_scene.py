"""T43 Y figure: t43y-y1-scene — V5 ep1925 s19253 HOLD start and deepest held-row violation, with the held query spheres
(body cover, server record) drawn in the MuJoCo scene; V4 same pair at its open command for comparison. Numbers from raw/*.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/Y/p'))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
import render_y as RY  # noqa: E402

Y = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Y')); FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
S = json.load(open(Y / "raw/server_chunks.json"))["runs"]; E = json.load(open(Y / "raw/esdf_held.json"))["runs"]
A = json.load(open(_ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')))["runs"]
K5, K4 = "V5/H/V5_ep1925_s19253", "V4/H/V4_ep1925_s19253"
C = figstyle.CATEGORICAL


def chunk(key, t):
    return next(c for c in S[key]["chunks"] if c["t"] == t)


def esdf(key, t):
    return next((x for x in E[key] if x["t"] == t), None)


hold = [x for x in E[K5] if x["hold"]]
t_hold = min(x["t"] for x in hold); deep = min(hold, key=lambda x: x["to_clearance_mm"]); t_deep = deep["t"]
t_v4 = 216   # V4 chunk where its policy open command first appears (y_metrics policy_open_t)
CLOSE = dict(lookat=(0.50, -0.02, 0.93), distance=0.75, azimuth=90.0, elevation=-6.0)
SPH = [(0.92, 0.41, 0.20, 0.35), (0.93, 0.63, 0.0, 0.35), (0.11, 0.69, 0.48, 0.35), (0.91, 0.48, 0.64, 0.35)]


def panel_imgs(key, t, mode):
    a = A[key]; rd = a["run_dir"]; M, ctrl = RY.scene_for(rd, "left")
    c = chunk(key, t); hq = c.get("held_query") or {}
    plan = {p["t_step"]: p for p in (json.loads(l) for l in open(pathlib.Path(rd) / "frames.jsonl")) if p["kind"] == "planning"}
    q = np.array(ctrl[t]["qpos"], float)
    if mode == "worst":
        k = int(c["worst"]["step"]); Aa = np.asarray(plan[t]["actions"], float); q = np.array(plan[t]["qpos"], float); q[20:27] = Aa[k, 0:7]
        cen = hq.get("centre_world_worst_step") or []
    else:
        cen = hq.get("centre_world_now") or []
    sph = [(p, 1e-3 * r, SPH[i % 4]) for i, (p, r) in enumerate(zip(cen, hq.get("radii_mm") or []))]
    vid = RY.video_frame(rd, t)
    full = RY.render(M, np.array(ctrl[t]["qpos"], float), RY.REC, sph if mode == "now" else ())
    close = RY.render(M, q, CLOSE, sph)
    return vid, full, close, c, sph


figstyle.use_korean()
fig, axes = plt.subplots(3, 3, figsize=(15, 10.8))
figstyle.style_axes(fig, axes)
rows = [(K5, t_hold, "now", f"V5 ep1925 s19253 · t {t_hold} (first HOLD chunk) · measured pose"),
        (K5, t_deep, "worst", f"V5 ep1925 s19253 · t {t_deep} (most negative TO clearance) · planned row {chunk(K5, t_deep)['worst']['step']} (not executed)"),
        (K4, t_v4, "now", f"V4 ep1925 s19253 · t {t_v4} (V4 policy open command, executed) · measured pose")]
side = {}
crop = (slice(60, 330), slice(150, 520))
for r, (key, t, mode, title) in enumerate(rows):
    vid, full, close, c, sph = panel_imgs(key, t, mode)
    axes[r, 0].imshow(vid[crop]); axes[r, 0].set_title("third_person_front.mp4 frame %d (crop)" % t, fontsize=9, color=figstyle.INK)
    axes[r, 1].imshow(full[crop]); axes[r, 1].set_title("MuJoCo re-render, same camera + held query spheres", fontsize=9, color=figstyle.INK)
    axes[r, 2].imshow(close); axes[r, 2].set_title("close-up side view (azimuth 90, elev -6)", fontsize=9, color=figstyle.INK)
    for ax in axes[r]:
        ax.set_xticks([]); ax.set_yticks([])
    e = esdf(key, t); hq = c.get("held_query") or {}
    sd = hq.get("crate_sd_mm_worst_step") if mode == "worst" else hq.get("crate_sd_mm_now")
    txt = [title, f"held query spheres (body cover, set '{hq.get('set')}'): r = {', '.join('%.1f' % x for x in hq.get('radii_mm') or [])} mm",
           f"worst TO row: {c['worst']['link']} vs {c['worst']['obstacle']}, TO clearance {c['worst']['clearance_mm']:.1f} mm, gate {'HOLD' if c['hold'] else 'execute'}"]
    if e:
        txt.append(f"recorded ESDF at sphere centre {e['d_fine_mm']:.1f} mm - r {e['radius_mm']:.1f} = gap {e['esdf_gap_mm']:.1f} mm; implied margin {e['implied_margin_mm']:.1f} mm")
    txt.append("true crate signed distance per sphere (MuJoCo boxes): " + ", ".join("%.1f" % x for x in sd or []) + " mm")
    txt.append(f"apple centre in crate frame [{', '.join('%.0f' % v for v in c['apple_in_crate_mm'])}] mm")
    axes[r, 0].text(0.0, 1.13, "\n".join(txt[:1]), transform=axes[r, 0].transAxes, fontsize=10, color=figstyle.INK, va="bottom")
    axes[r, 2].text(1.02, 0.98, "\n".join(txt[1:]), transform=axes[r, 2].transAxes, fontsize=8, color=figstyle.INK_2, va="top", wrap=True)
    side[f"{key}@{t}"] = {"mode": mode, "worst": c["worst"], "hold": c["hold"], "held_query": {k: hq.get(k) for k in ("set", "radii_mm")}, "crate_sd_mm": sd,
                         "esdf": e, "apple_in_crate_mm": c["apple_in_crate_mm"], "spheres_world": [list(map(float, p)) for p, _, _ in sph]}
fig.subplots_adjust(left=0.01, right=0.74, top=0.92, bottom=0.01, hspace=0.22, wspace=0.03)
fig.suptitle("T43 Y1 — V5 ep1925 s19253: HOLD binding row = held query sphere vs destination (crate); sphere colours = held sphere index 0 / 1 / 2", fontsize=11, color=figstyle.INK, y=0.995)
out = FIG / "t43y-y1-scene.png"; fig.savefig(out, dpi=110, facecolor=figstyle.SURFACE)
json.dump({"figure": str(out), "script": __file__, "panels": side, "close_camera": CLOSE, "video_camera": RY.REC, "crop_rows_cols": [[60, 330], [150, 520]],
           "sources": ["outputs/verify/T43/Y/raw/server_chunks.json", "outputs/verify/T43/Y/raw/esdf_held.json"]}, open(FIG / "t43y-y1-scene.json", "w"), indent=1, default=float)
print(out)
