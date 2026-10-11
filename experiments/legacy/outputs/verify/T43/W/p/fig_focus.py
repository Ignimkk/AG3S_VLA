"""T43W focus figure (verifier): per-run paired values of two arms (A vs B) from numbers.json files.
usage: fig_focus.py OUT_NAME ARM_A:NUMBERS_JSON ARM_B:NUMBERS_JSON [--overwrite]
   e.g. fig_focus.py t43w-v5-focus V4:analysis/numbers.json V5:analysis/numbers.json
-> AG3S/ag3s/docs/figures/t43/OUT_NAME.png + .json

panels (hurdle runs, stage H, paired by (episode, seed); one x position per pair, A and B markers side by side):
  1 outcome            : 2 = success, 1 = grasp only, 0 = neither (analyze_e0 rules)
  2 closing deflection : max over the run's closing chunks of outward_mm = |tip(refined) - apple| - |tip(reference) - apple| at control row 7
                         (> 0 = pushed away from the apple; analyze_t43w 'focus'); number of closing chunks with outward > 2 mm in the label
  3 carve-out radius   : median over the run's closing chunks of esdf_stats.target_free.ball.radius_mm
  4 held sphere        : largest held query sphere radius at attach (grasp.attached.held.spheres), mm
  5 held-row violation : number of attached chunks whose waypoint worst row is a held sphere with clearance < 0 (min clearance in the label)
  6 TO ms              : median of server to.solve_ms over the run's chunks (first excluded)
colours: A = CATEGORICAL slot 2, B = slot 4 (figstyle)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle

OUT = pathlib.Path(os.environ.get("T43W_FIGOUT", _ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))); OUT.mkdir(parents=True, exist_ok=True)
name = sys.argv[1]
(armA, fA), (armB, fB) = [a.split(":", 1) for a in sys.argv[2:4]]
NA = json.load(open(fA))["per_run"]; NB = json.load(open(fB))["per_run"]
figstyle.use_korean()
CA, CB = figstyle.CATEGORICAL[2], figstyle.CATEGORICAL[4]


def runs(N, arm):
    return {(v["episode"], v["seed"]): v for v in N.values() if v["arm"] == arm and v["stage"] == "H"}


RA, RB = runs(NA, armA), runs(NB, armB)
keys = sorted(set(RA) & set(RB), key=lambda k: (k[0], k[1]))
lab = [f"{e}\ns{s}" for e, s in keys]


def val(v, what):
    if what == "outcome":
        return 2 if v["success"] else (1 if v["grasp"] else 0)
    if what == "outward":
        return (v.get("closing") or {}).get("outward_mm_max")
    if what == "carve":
        return ((v.get("closing") or {}).get("carve_ball_radius_mm") or {}).get("median")
    if what == "held":
        h = [x["largest_radius_mm"] for x in v.get("attach_held") or [] if x.get("largest_radius_mm") is not None]
        return max(h) if h else None
    if what == "viol":
        return len(v.get("held_row_violation_chunks") or [])
    if what == "to":
        return (v.get("to_solve_ms") or {}).get("median")


PAN = [("outcome", "outcome (2 S / 1 G / 0 -)"), ("outward", "closing: max outward tip shift (mm)"), ("carve", "closing: carve-out ball radius, median (mm)"),
       ("held", "attach: largest held sphere radius (mm)"), ("viol", "held-row violation chunks"), ("to", "TO solve ms, median per run")]
fig, axes = plt.subplots(len(PAN), 1, figsize=(max(10, 0.75 * len(keys) + 3), 2.3 * len(PAN)), sharex=True)
figstyle.style_axes(fig, axes)
x = np.arange(len(keys)); side = {"definitions": __doc__, "arm_a": [armA, fA], "arm_b": [armB, fB], "pairs": [f"{e}_s{s}" for e, s in keys], "panels": {}}
for ax, (w, t) in zip(axes, PAN):
    a = [val(RA[k], w) for k in keys]; b = [val(RB[k], w) for k in keys]
    side["panels"][w] = {armA: a, armB: b}
    for xs, ys, c, l in ((x - 0.15, a, CA, armA), (x + 0.15, b, CB, armB)):
        xx = [xi for xi, yi in zip(xs, ys) if yi is not None]; yy = [yi for yi in ys if yi is not None]
        ax.scatter(xx, yy, color=c, s=28, label=l, zorder=3)
    if w == "outward":
        ax.axhline(2.0, color=figstyle.INK_2, lw=0.6, ls="--"); ax.axhline(0.0, color=figstyle.INK, lw=0.6)
        for xi, k in zip(x, keys):
            ax.text(xi, ax.get_ylim()[1] if False else 0, "", fontsize=6)
    if w == "to":
        ax.axhline(1000.0 * 8 / 15, color=figstyle.INK, lw=0.8, ls="--")
    ax.set_title(t, fontsize=9, color=figstyle.INK)
axes[0].set_yticks([0, 1, 2]); axes[0].legend(fontsize=8, frameon=False, loc="upper right")
axes[-1].set_xticks(x); axes[-1].set_xticklabels(lab, fontsize=7)
side["n_outward_gt_2mm"] = {armA: [(RA[k].get("closing") or {}).get("n_outward_gt_2mm") for k in keys], armB: [(RB[k].get("closing") or {}).get("n_outward_gt_2mm") for k in keys]}
side["n_closing_chunks"] = {armA: [(RA[k].get("closing") or {}).get("n_chunks") for k in keys], armB: [(RB[k].get("closing") or {}).get("n_chunks") for k in keys]}
side["held_viol_min_mm"] = {arm: [min([c["clearance_mm"] for c in (R[k].get("held_row_violation_chunks") or [])], default=None) for k in keys] for arm, R in ((armA, RA), (armB, RB))}
fig.suptitle(f"T43 W per-run pairs: {armA} vs {armB} (overbar P2 hurdle runs)", color=figstyle.INK, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
p = OUT / f"{name}.png"
assert not p.exists() or "--overwrite" in sys.argv, f"{p} exists"
fig.savefig(p, dpi=150, facecolor=figstyle.SURFACE); plt.close(fig)
(OUT / f"{name}.json").write_text(json.dumps(side, indent=1, default=float)); print(p)
