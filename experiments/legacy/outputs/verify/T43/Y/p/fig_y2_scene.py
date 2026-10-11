"""T43 Y figure: t43y-y2-scene — place moment of one pair, V4 vs V5 (third_person_front.mp4 frames, crop).
Pair rule: among the 14 both-success pairs, the largest |V5 - V4| of release_apple_floor_gap_mm (raw/y_metrics.json)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/Y/p'))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
import render_y as RY  # noqa: E402

Y = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Y')); FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
R = json.load(open(Y / "raw/y_metrics.json"))["runs"]; A = json.load(open(_ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')))["runs"]
pairs = [(k, k.replace("V4", "V5")) for k in sorted(R) if k.startswith("V4/") and k.replace("V4", "V5") in R and R[k]["success"] and R[k.replace("V4", "V5")]["success"]]
k4, k5 = max(pairs, key=lambda p: abs(R[p[1]]["release_apple_floor_gap_mm"] - R[p[0]]["release_apple_floor_gap_mm"]))
figstyle.use_korean()
fig, ax = plt.subplots(2, 3, figsize=(15, 7.8)); figstyle.style_axes(fig, ax)
crop = (slice(80, 330), slice(170, 500)); side = {"pair": [k4, k5], "rule": __doc__.splitlines()[1], "panels": {}}
for r, k in enumerate((k4, k5)):
    o = R[k]; s = o["series"]; t = np.array(s["t"]); rel = o["release_t"]
    for c, (tt, lab) in enumerate(((rel - 1, "last row with apple-hand contact"), (rel + 3, "release + 3 rows"), (rel + 20, "release + 20 rows"))):
        tt = int(min(tt, t[-1])); i = int(np.searchsorted(t, tt))
        ax[r, c].imshow(RY.video_frame(A[k]["run_dir"], tt)[crop]); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        info = {"t": tt, "apple_floor_gap_mm": s["apple_floor_gap_mm"][i], "apple_z_crate_mm": s["apple_in_crate_z_mm"][i], "grip_cmd": s["grip_cmd"][i],
                "finger_open": s["finger_open"][i], "apple_speed_mm_s": s["apple_speed_mm_s"][i]}
        side["panels"][f"{k}@{tt}"] = info
        ax[r, c].set_title(f"{k.split('/')[0]} t {tt} ({lab})\napple-floor gap {info['apple_floor_gap_mm']:.0f} mm · cmd {info['grip_cmd']:.2f} · opening {info['finger_open']:.2f} · |v| {info['apple_speed_mm_s']:.0f} mm/s",
                           fontsize=8.5, color=figstyle.INK)
    ax[r, 0].text(-0.02, 0.5, f"{k.split('/')[0]}\nrelease mode {o['release_mode']}\nrelease_t {rel}, applied open {o['applied_open_t']}\nxy travel 2 s {o['post_release_horiz_travel_mm']:.0f} mm",
                  transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=8.5, color=figstyle.INK_2)
fig.suptitle(f"T43 Y2 — place moment, pair {k4.split('/', 1)[1].replace('V4_', '')} (pair with the largest |V5 - V4| apple-floor gap at release), third_person_front.mp4 crops", fontsize=10.5, color=figstyle.INK)
fig.subplots_adjust(left=0.14, right=0.99, top=0.86, bottom=0.02, wspace=0.04, hspace=0.3)
out = FIG / "t43y-y2-scene.png"; fig.savefig(out, dpi=115, facecolor=figstyle.SURFACE)
json.dump({"figure": str(out), "script": __file__, **side, "crop_rows_cols": [[80, 330], [170, 500]]}, open(FIG / "t43y-y2-scene.json", "w"), indent=1, default=float)
print(out, k4, k5)
