"""T21 verifier figures (rule A): scene / graph / table. Numbers come from replay_*.json and
summary_*.json in this directory; each figure gets a .json sidecar with the numbers it drew."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle

OUT = Path(_ag3s_workspace_path('outputs/verify/t21'))
FIG = Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t21'))
FIG.mkdir(parents=True, exist_ok=True)
KO = figstyle.use_korean()
C = figstyle.CATEGORICAL

RULES = [  # key, label, colour slot
    ("live", "old: live recorded (target_centroid, crate)", 0),
    ("a_t20", "old rule, manipulated=T20 apple", 1),
    ("b1_rec", "b1 hand_swept q_now, manip=recorded", 2),
    ("b1_t20", "b1 hand_swept q_now, manip=T20 apple", 3),
    ("b2_t20", "b2 hand_swept q_now+path8, manip=T20 apple", 4),
]


def load(run):
    return (json.loads((OUT / f"replay_{run}.json").read_text()),
            json.loads((OUT / f"summary_{run}.json").read_text()))


# ------------------------------------------------------------------------------------------ scene
def scene():
    d, _ = load("t14")
    c = [x for x in d["chunks"] if x["chunk"] == 12][0]
    import AG3S.trajopt.experiments.fixed_input_ablation as T  # noqa: F401 (same env)
    rec = np.load(f"${AG3S_PROJECT_ROOT}/outputs/live_test/20260928_t14/server_constraints/run_0000/chunk_00012.npz")
    cloud = rec["filtered_cloud"].astype(float)
    hand = np.asarray(c["hand_centres_m"])
    radii = np.asarray(c["hand_radii_m"])
    apple = np.asarray(c["object_poses"]["apple"]["pos"])
    crate = np.asarray(c["object_poses"]["crate"]["pos"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    figstyle.style_axes(fig, axes)
    views = [(0, 1, "top view (x-y)", "x [m]", "y [m]"), (0, 2, "side view (x-z)", "x [m]", "z [m]")]
    sidecar = {"chunk": 12, "t_step": c["t_step"], "windows": {}, "apple_m": apple.tolist(),
               "crate_m": crate.tolist(), "hand_q_now_m": hand[0].tolist(), "hand_radii_m": radii.tolist()}
    sel = np.all((cloud > [0.15, -0.55, 0.55]) & (cloud < [1.0, 0.65, 1.3]), axis=1)
    sub = cloud[sel][::3]
    for ax, (i, j, title, xl, yl) in zip(axes, views):
        ax.scatter(sub[:, i], sub[:, j], s=1.2, c="#9c9a94", alpha=0.6, lw=0, zorder=1,
                   label="recorded cloud t=96 (filtered)")
        for key, lab, slot in RULES:
            w = c["windows"][key]
            lo = np.asarray(w["lower_m"]); hi = np.asarray(w["upper_m"])
            ax.add_patch(Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j], fill=False,
                                   ec=C[slot], lw=1.8, ls="-" if key != "live" else "--", zorder=3,
                                   label=f"{lab} [{w.get('basis')}{'/' + w['overflow'] if w.get('overflow') else ''}]"))
            sidecar["windows"][key] = {"lower_m": lo.tolist(), "upper_m": hi.tolist(),
                                       "basis": w.get("basis"), "overflow": w.get("overflow")}
        for s in range(hand.shape[1]):
            ax.add_patch(Circle((hand[0, s, i], hand[0, s, j]), radii[s], color=C[6], alpha=0.8, lw=0, zorder=4))
        pth = hand[1:].reshape(-1, 3)
        ax.scatter(pth[:, i], pth[:, j], s=3, color=C[6], alpha=0.35, zorder=4,
                   label="authorized spheres: q_now (filled) / ref steps 1-8 (dots)")
        ax.add_patch(Circle((apple[i], apple[j]), 0.033, fill=False, ec=C[7], lw=2, zorder=5))
        ax.plot([], [], color=C[7], lw=2, label="apple (MuJoCo truth, r~33 mm)")
        ax.plot(crate[i], crate[j], marker="s", color=figstyle.INK, ms=6, zorder=5, ls="")
        ax.set_title(title, fontsize=10, color=figstyle.INK)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_aspect("equal")
        ax.set_xlim(0.15, 1.0)
        ax.set_ylim((-0.55, 0.65) if j == 1 else (0.5, 1.3))
    axes[0].plot([], [], marker="s", color=figstyle.INK, ls="", label="crate body (truth)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, frameon=False)
    fig.suptitle("T21 - fine / target-free window placement, T14 chunk 12 (t=96). Window = 128^3 x 5 mm (0.635 m interp. range)",
                 fontsize=11, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0.17, 1, 0.95))
    fig.savefig(FIG / "t21-scene.png", dpi=150, facecolor=figstyle.SURFACE)
    (FIG / "t21-scene.json").write_text(json.dumps(sidecar, indent=1))
    plt.close(fig)


# ------------------------------------------------------------------------------------------ graph
GRAPH_RULES = [("live_exact", "old live, exact recorded field", 0),
               ("a_t20", "old rule, manip=T20 apple (proxy)", 1),
               ("b1_rec", "b1 q_now, manip=recorded (proxy)", 2),
               ("b1_t20", "b1 q_now, manip=T20 apple (proxy)", 3),
               ("b2_t20", "b2 q_now+path8, manip=T20 apple (proxy)", 4)]
COL_RULES = [("live_exact", "exact recorded field (= T24)", 0), ("live_proxy", "proxy at live window (control)", 5),
             ("b1_rec", "b1, manip=recorded", 2), ("b2_rec", "b2, manip=recorded", 6),
             ("b1_t20", "b1, manip=T20 apple", 3), ("b2_t20", "b2, manip=T20 apple", 4)]


def graph():
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex="col")
    figstyle.style_axes(fig, axes)
    side = {}
    for col, run in enumerate(("t14", "t17")):
        _, S = load(run)
        t = S["t_steps"]
        side[run] = {"t_step": t, "chunks": S["chunks"], "frac_all": {}, "unresolved_rows": {}, "collision_median_mm": {}, "collision_max_mm": {}}
        ax = axes[0, col]
        for key, lab, slot in GRAPH_RULES:
            rows = S["per_rule"][key]["per_chunk"]
            y = [r["frac_all"] for r in rows]
            ax.plot(t, y, marker="o", color=C[slot], label=lab, lw=1.6, ms=4)
            side[run]["frac_all"][key] = y
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel("authorized query points in window\n(q_now + ref steps 1-8)")
        ax.set_title(f"{run.upper()} - fraction inside the fine window (geometric)", fontsize=10, color=figstyle.INK)
        ax = axes[1, col]
        for key, lab, slot in GRAPH_RULES:
            rows = S["per_rule"][key]["per_chunk"]
            y = [r["by_tier_all"]["coarse"] + r["by_tier_all"]["outside"] for r in rows]
            ax.plot(t, y, marker="o", color=C[slot], label=lab, lw=1.6, ms=4)
            side[run]["unresolved_rows"][key] = y
        n = S["per_rule"]["live_exact"]["per_chunk"][0]["n_all"]
        ax.set_ylabel(f"rows answered by coarse/outside\n(of {n} per chunk)")
        ax.set_title(f"{run.upper()} - authorized rows answered by coarse tier (answer_tier)", fontsize=10, color=figstyle.INK)
        ax = axes[2, col]
        for key, lab, slot in COL_RULES:
            v = S["collision"][key]
            ax.plot(t, v["per_chunk_median_mm"], marker="o", color=C[slot], label=lab, lw=1.6, ms=4,
                    ls="--" if key == "live_proxy" else "-")
            side[run]["collision_median_mm"][key] = v["per_chunk_median_mm"]
            side[run]["collision_max_mm"][key] = v["per_chunk_max_mm"]
        ax.set_ylabel("fingertip displacement vs reference\nper-chunk median of 8 exec steps [mm]")
        ax.set_xlabel("t_step (chunk k: t = 8k)")
        ax.set_title(f"{run.upper()} - variant 'collision' (T24 tool, clock off)", fontsize=10, color=figstyle.INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.965))
    h, l = axes[2, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("T21 - window coverage, answering tier and fingertip displacement per chunk", fontsize=11, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(FIG / "t21-graph.png", dpi=150, facecolor=figstyle.SURFACE)
    (FIG / "t21-graph.json").write_text(json.dumps(side, indent=1))
    plt.close(fig)


# ------------------------------------------------------------------------------------------ table
TABLE_RULES = ["live_exact", "live_proxy", "a_rec", "b1_rec", "b2_rec", "a_t20", "b1_t20", "b2_t20",
               "a_default", "b1_default", "b2_default"]


def table():
    header = ["run", "rule / field", "field", "chunks", "basis (count)", "overflow (count)", "in-window frac (mean)",
              "tier totals (all authorized rows)", "finger chunks resolved", "collision med / max [mm]"]
    rows = []
    side = {}
    for run in ("t14", "t17"):
        _, S = load(run)
        side[run] = {}
        for key in TABLE_RULES:
            v = S["per_rule"][key]
            tiers = {t: n for t, n in v["by_tier_all_total"].items() if n}
            col = S["collision"].get(key)
            colstr = "-" if col is None else f"{col['pooled_median_mm']:.1f} / {col['pooled_max_mm']:.1f}"
            fieldtype = "exact" if key == "live_exact" else "proxy"
            rows.append([run.upper(), key, fieldtype, str(v["n_chunks"]),
                         ", ".join(f"{b}:{n}" for b, n in v["basis_counts"].items()),
                         ", ".join(f"{o}:{n}" for o, n in v["overflow_counts"].items()),
                         f"{v['frac_all_mean']:.3f}",
                         ", ".join(f"{t}:{n}" for t, n in tiers.items()) + f"  (/{v['n_rows_total']})",
                         f"{v['chunks_finger_all_resolved']}/{v['n_chunks']}", colstr])
            side[run][key] = {"basis_counts": v["basis_counts"], "overflow_counts": v["overflow_counts"],
                              "frac_all_mean": v["frac_all_mean"], "by_tier_all_total": v["by_tier_all_total"],
                              "n_rows_total": v["n_rows_total"],
                              "chunks_finger_all_resolved": v["chunks_finger_all_resolved"],
                              "collision_pooled_median_mm": None if col is None else col["pooled_median_mm"],
                              "collision_pooled_max_mm": None if col is None else col["pooled_max_mm"]}
    fig, ax = plt.subplots(figsize=(17, 0.3 * (len(rows) + 5)))
    fig.patch.set_facecolor(figstyle.SURFACE)
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=header, bbox=[0, 0.1, 1, 0.86], cellLoc="left", colLoc="left")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.5)
    tb.auto_set_column_width(list(range(len(header))))
    for (r, cc), cell in tb.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        cell.set_facecolor(figstyle.SURFACE if r else "#eeede9")
        if r and rows[r - 1][1] == "live_exact":
            cell.set_facecolor("#e6effb")
    note = ("rows = authorized-link spheres (ee_left, ee_finger_l1, ee_finger_l2) at q_now + reference steps 1-8.  "
            "exact = recorded cuRobo grids; proxy = recorded coarse grid + fine/target-free layers rebuilt from the "
            "accumulated recorded cloud (unsigned nearest-point distance, no depth).  "
            "a = target_centroid, b1 = hand_swept with q_now only, b2 = hand_swept with q_now + reference steps 1-8; "
            "_rec = recorded target points, _t20 = T20 TargetConfirm (min_score 0.1, lost 20), _default = current defaults.")
    fig.text(0.01, 0.01, note, fontsize=7, color=figstyle.INK_2, wrap=True)
    fig.suptitle("T21 - overflow, basis and answering-tier distribution per rule (T14 chunks 11-20, T17 chunks 9-15)",
                 fontsize=10.5, color=figstyle.INK)
    fig.savefig(FIG / "t21-table.png", dpi=150, facecolor=figstyle.SURFACE)
    (FIG / "t21-table.json").write_text(json.dumps(side, indent=1))
    plt.close(fig)


if __name__ == "__main__":
    scene(); graph(); table()
    print("ok", sorted(p.name for p in FIG.iterdir()))
