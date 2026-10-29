"""T43 XM scene figure (verifier). usage: PYTHONPATH=${AG3S_PROJECT_ROOT} fig_scene_xm.py OUT_PNG
Pair ep1807 s18073 (same episode:seed): V8 (no servo model, no stay start) vs V9 (+servo model path rows, stay start).
Top: third_person_front.mp4 frames (frame t = post state of control row t; green arrow = the client's refined - reference TCP overlay).
Bottom: side view across the bar in the crossing frame of xm_cross.py: x = s (mm, horizontal distance across the bar, + toward the crate),
  y = z - bar top (mm).  grey = measured palm path (EE_BODY origin, rows), red dots = measured apple centre while in hand;
  per window chunk (carry, t0 in the shown range): dashed = reference palm rows k = 0..7, solid = refined palm rows (reference + D),
  arrow at k = 7 = D projected on (n, z).  Bar = circle r 12 mm.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, subprocess, tempfile
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out = pathlib.Path(sys.argv[1]); assert not out.exists(), f"{out} exists"
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/XM/raw'))
PAIR = {"V8": ("V8/H/V8_ep1807_s18073", None), "V9": ("V9/H/V9_ep1807_s18073", None)}
figstyle.use_korean()
C = figstyle.CATEGORICAL
side = {"script": __file__, "panels": {}}
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
for j, arm in enumerate(("V8", "V9")):
    key = PAIR[arm][0]
    r = json.load(open(RAW / f"cross_{arm}.json"))["runs"][key]
    rows = r["rows"]; ch = {c["seq"]: c for c in r["chunks"]}
    tx = r["crossing_rows"]["palm_M"]["carry"]
    win = [c for c in r["chunks"] if c["in_window"] and c["grasp_state"] != "placed" and c["t0"] <= tx + 8 and c["t0"] >= tx - 48]
    # frame: the row of the largest executed palm deflection in those chunks
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
    ax.set_title(f"{arm} ep1807 s18073 video frame t={tf} ({tf / 15:.1f} s), executed palm |D| = {best[1]:.0f} mm", fontsize=10)
    ax2 = fig.add_subplot(gs[1, j])
    lo, hi = min(c["t0"] for c in win) - 4, tx + 16
    sM = [rows[t]["s_M"]["palm"] for t in range(lo, min(hi, len(rows)))]; zM = [rows[t]["z_M"]["palm"] for t in range(lo, min(hi, len(rows)))]
    ax2.plot(sM, zM, color="#888888", lw=2.2, label="measured palm (rows)", zorder=2)
    aT = [t for t in range(lo, min(hi, len(rows))) if rows[t]["inhand"]]
    ax2.scatter([rows[t]["s_M"]["apple"] for t in aT], [rows[t]["z_M"]["apple"] for t in aT], s=10, color=C[7], label="measured apple centre (in hand)", zorder=3)
    if r["drop_t"] is not None and lo <= r["drop_t"] < hi:
        t = r["drop_t"]; ax2.scatter([rows[t]["s_M"]["palm"]], [rows[t]["z_M"]["palm"]], marker="x", s=90, color=C[7], zorder=5, label=f"apple leaves hand t={t}")
    pan = []
    for i, c in enumerate(win):
        dp = c["deflection"]["palm"]
        sr = [d["s_rfc_mm"] for d in dp]; zr = [d["z_rfc_minus_bar_top_mm"] for d in dp]
        sf = [d["s_rfc_mm"] + d["dn"] for d in dp]; zf = [d["z_rfc_minus_bar_top_mm"] + d["dz"] for d in dp]
        col = C[i % 6]
        ax2.plot(sr, zr, ls="--", color=col, lw=1.2, zorder=3)
        ax2.plot(sf, zf, ls="-", color=col, lw=1.8, zorder=4, label=f"seq {c['seq']} t0={c['t0']} start={str(c['initial_source']).split(' ')[0]} |D|max={max(d['mag'] for d in dp):.0f}")
        ax2.annotate("", xy=(sf[-1], zf[-1]), xytext=(sr[-1], zr[-1]), arrowprops=dict(arrowstyle="->", color=col, lw=1.6), zorder=6)
        pan.append({"seq": c["seq"], "t0": c["t0"], "initial_source": c["initial_source"], "ref_s": sr, "ref_z": zr, "rfd_s": sf, "rfd_z": zf,
                    "D_k": [[round(d["mag"], 2), round(d["dz"], 2), round(d["away"], 2), round(d["du"], 2)] for d in dp],
                    "binding": c["binding"], "executed": c["executed"]})
    circ = plt.Circle((0, -r["bar"]["radius_mm"]), r["bar"]["radius_mm"], color=C[3], zorder=7)
    ax2.add_patch(circ); ax2.axvline(0, color=figstyle.GRID_INK, lw=0.8, zorder=1)
    ax2.set_aspect("equal"); ax2.set_xlabel("s = horizontal distance across the bar (mm, + toward crate)"); ax2.set_ylabel("z - bar top (mm)")
    ax2.set_title(f"{arm}: palm reference (dashed) vs refined (solid), arrow = D at k=7; bar = circle", fontsize=10)
    ax2.legend(fontsize=7, loc="upper left", ncol=1)
    side["panels"][arm] = {"key": key, "video_frame_t": tf, "executed_palm_D_mm": best[1], "palm_cross_t": tx, "drop_t": r["drop_t"],
                           "measured_palm": {"t": list(range(lo, min(hi, len(rows)))), "s": sM, "z": zM}, "chunks": pan}
figstyle.style_axes(fig, [a for a in fig.axes if a.get_xlabel()])
fig.suptitle("T43 XM: hurdle crossing, TO deflection D = refined - reference (palm), V8 vs V9, ep1807 s18073", fontsize=12)
fig.tight_layout()
fig.savefig(out, dpi=110)
out.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float))
print(out)
