"""(2026-11-11 dev-57) Figure: true robot-obstacle distance before each E1a collision (pe123458), with the server path of
every call (pass-through / action QP / HOLD). Reads first_contact_<tag>.json and exec_clearance_scan_<tag>.json.
usage: python fig_first_contact.py <tag>"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

O = "/mnt/dev/AG3S_VLA/outputs/2026-11-11_SOTA_개선/offline/"
tag = sys.argv[1]
fc = [r for r in json.load(open(O + f"first_contact_{tag}.json")) if r["collide_t"] is not None]
scan = [x for x in json.load(open(O + f"exec_clearance_scan_{tag}.json")) if x["bench"] == "E1a40"]
col = {"pass": "#4477AA", "qp": "#EE6677", "hold": "#BBBBBB", "other": "#BBBBBB"}
n = len(fc)
ncol = 6
nrow = (n + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.3 * nrow), sharey=True)
axes = axes.reshape(-1)
for ax, r in zip(axes, fc):
    td = sorted((int(k), v[0]) for k, v in r["true_robot_dist_mm"].items())
    f = [a for a, _ in td]
    d = [max(-25.0, b) for _, b in td]
    ct = r["collide_t"]
    ax.axhspan(-25, 0, color="#EE6677", alpha=0.08)
    ax.axhline(10, color="k", lw=0.6, ls=":")
    ax.plot([x - ct for x in f], d, "-o", color="k", ms=2.5, lw=1.0)
    calls = [x for x in scan if x["task"] == r["task"] and x["init"] == r["init"]]
    for x in calls:
        f0 = 20 + x["seq"] * 5   # obs_index of the call: 20 wait steps, 5 executed steps per call
        if f0 - ct < -13 or f0 - ct > 2:
            continue
        ax.axvline(f0 - ct, color=col.get(x["path"], "#BBBBBB"), lw=2.2, alpha=0.75)
    by = r["touch_log"][0]["touch"][0].split("/")[1].replace("_collision", "").replace("gripper0_", "").replace("robot0_", "") if r["touch_log"] else "indirect"
    ax.set_title(f"t{r['task']} e{r['init']}  {r['obstacle'].replace('_obstacle_1', '')}\nfirst: {by}, {r['max_displacement_mm']:.0f} mm", fontsize=8)
    ax.set_xlim(-13, 2.5)
    ax.set_ylim(-25, 90)
    ax.tick_params(labelsize=7)
for ax in axes[n:]:
    ax.axis("off")
for ax in axes[::ncol]:
    ax.set_ylabel("true robot-obstacle distance [mm]", fontsize=7)
for ax in axes[max(0, n - ncol):n]:
    ax.set_xlabel("control step relative to the collision step", fontsize=7)
handles = [plt.Line2D([], [], color=col["pass"], lw=3, label="call: pass-through"),
           plt.Line2D([], [], color=col["qp"], lw=3, label="call: action QP"),
           plt.Line2D([], [], color="k", lw=0.6, ls=":", label="10 mm (obstacle floor)")]
fig.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
fig.suptitle(f"E1a 40 ({tag}): true robot-obstacle distance before each collision (MuJoCo replay, scoring only)", fontsize=10)
fig.tight_layout(rect=(0, 0.02, 1, 0.96))
out = O + f"fig_first_contact_{tag}.png"
fig.savefig(out, dpi=140)
print(out)
