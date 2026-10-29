"""T43 V11A scene figure (verifier). usage: PYTHONPATH=${AG3S_PROJECT_ROOT} fig_scene_v11.py OUT_PNG [EP:SEED]
Adapted from T43/XM/p/fig_scene_xm.py (same geometry, same rows). Pair V8 vs V11A of the same episode:seed (default 1807:18073).
Top: third_person_front.mp4 frame (frame t = post state of control row t) at the row of the largest executed palm deflection of the carry-window chunks.
Bottom: side view across the bar (x = s horizontal distance across the bar, mm, + toward crate; y = z - bar top, mm):
  grey thick = measured palm path (rows), black thin = commanded palm path (executed_actions on the measured state), red dots = measured apple centre in hand;
  per window chunk: dashed = reference palm rows k = 0..7, solid = refined (reference + D); arrow at k = 7 = D; bar = circle r 12 mm."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, subprocess, tempfile
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out = pathlib.Path(sys.argv[1]); assert not out.exists() and not out.with_suffix(".json").exists(), f"{out} exists"
EPS = sys.argv[2] if len(sys.argv) > 2 else "1807:18073"; ep, seed = EPS.split(":")
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
RAWF = {"V8": T / "XM/raw/cross_V8.json", "V11A": T / "V11/raw/cross_V11A.json"}
figstyle.use_korean(); C = figstyle.CATEGORICAL
side = {"script": __file__, "pair": EPS, "panels": {}}
fig = plt.figure(figsize=(16, 11)); gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
for j, arm in enumerate(("V8", "V11A")):
    key = f"{arm}/H/{arm}_ep{ep}_s{seed}"
    r = json.load(open(RAWF[arm]))["runs"][key]; rows = r["rows"]
    tx = r["crossing_rows"]["palm_M"]["carry"]
    win = [c for c in r["chunks"] if c["in_window"] and c["grasp_state"] != "placed" and c["t0"] <= tx + 8 and c["t0"] >= tx - 48]
    best = None
    for c in win:
        for k, d in enumerate(c["deflection"]["palm"]):
            t = c["t0"] + k
            if t < len(rows) and rows[t]["executed"] and (best is None or d["mag"] > best[1]):
                best = (t, d["mag"])
    tf = best[0]
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg", "-v", "error", "-i", r["mp4"], "-vf", f"select='eq(n\\,{tf})'", "-vsync", "0", f"{td}/f.png"], check=True)
        img = mpimg.imread(f"{td}/f.png")
    ax = fig.add_subplot(gs[0, j]); ax.imshow(img); ax.axis("off")
    ax.set_title(f"{arm} ep{ep} s{seed} video frame t={tf} ({tf / 15:.1f} s), executed palm |D| = {best[1]:.0f} mm", fontsize=10)
    ax2 = fig.add_subplot(gs[1, j])
    lo, hi = min(c["t0"] for c in win) - 4, tx + 16; rg = range(lo, min(hi, len(rows)))
    ax2.plot([rows[t]["s_M"]["palm"] for t in rg], [rows[t]["z_M"]["palm"] for t in rg], color="#888888", lw=2.4, label="measured palm (rows)", zorder=2)
    ax2.plot([rows[t]["s_C"]["palm"] for t in rg], [rows[t]["z_C"]["palm"] for t in rg], color="black", lw=1.0, label="commanded palm (executed_actions)", zorder=2)
    aT = [t for t in rg if rows[t]["inhand"]]
    ax2.scatter([rows[t]["s_M"]["apple"] for t in aT], [rows[t]["z_M"]["apple"] for t in aT], s=10, color=C[7], label="measured apple centre (in hand)", zorder=3)
    pan = []
    for i, c in enumerate(win):
        dp = c["deflection"]["palm"]
        sr = [d["s_rfc_mm"] for d in dp]; zr = [d["z_rfc_minus_bar_top_mm"] for d in dp]
        sf = [d["s_rfc_mm"] + d["dn"] for d in dp]; zf = [d["z_rfc_minus_bar_top_mm"] + d["dz"] for d in dp]; col = C[i % 6]
        ax2.plot(sr, zr, ls="--", color=col, lw=1.2, zorder=3)
        ax2.plot(sf, zf, ls="-", color=col, lw=1.8, zorder=4, label=f"seq {c['seq']} t0={c['t0']} |D|max={max(d['mag'] for d in dp):.0f}")
        ax2.annotate("", xy=(sf[-1], zf[-1]), xytext=(sr[-1], zr[-1]), arrowprops=dict(arrowstyle="->", color=col, lw=1.6), zorder=6)
        pan.append({"seq": c["seq"], "t0": c["t0"], "ref_s": sr, "ref_z": zr, "rfd_s": sf, "rfd_z": zf, "D_k": [[round(d["mag"], 2), round(d["dz"], 2), round(d["away"], 2), round(d["du"], 2)] for d in dp]})
    ax2.add_patch(plt.Circle((0, -r["bar"]["radius_mm"]), r["bar"]["radius_mm"], color=C[3], zorder=7)); ax2.axvline(0, color=figstyle.GRID_INK, lw=0.8, zorder=1)
    ax2.set_aspect("equal"); ax2.set_xlabel("s = horizontal distance across the bar (mm, + toward crate)"); ax2.set_ylabel("z - bar top (mm)")
    ax2.set_title(f"{arm}: palm reference (dashed) vs refined (solid), arrow = D at k=7; bar = circle", fontsize=10); ax2.legend(fontsize=7, loc="upper left")
    cx = r["crossing"]
    side["panels"][arm] = {"key": key, "video_frame_t": tf, "executed_palm_D_mm": best[1], "palm_cross_t": tx, "success": r["success"],
                           "apple_bottom_minus_bar_top_mm_at_apple_crossing": {s_: (cx.get(f"carry_apple_{s_}") or {}).get("apple_bottom_minus_bar_top_mm") for s_ in ("C", "M")},
                           "measured_palm": {"t": list(rg), "s": [rows[t]["s_M"]["palm"] for t in rg], "z": [rows[t]["z_M"]["palm"] for t in rg]},
                           "commanded_palm": {"s": [rows[t]["s_C"]["palm"] for t in rg], "z": [rows[t]["z_C"]["palm"] for t in rg]}, "chunks": pan}
figstyle.style_axes(fig, [a for a in fig.axes if a.get_xlabel()])
fig.suptitle(f"T43 V11A: hurdle crossing, TO deflection D = refined - reference (palm), V8 vs V11A, ep{ep} s{seed}", fontsize=12)
fig.tight_layout(); fig.savefig(out, dpi=110, facecolor=figstyle.SURFACE)
out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(out)
