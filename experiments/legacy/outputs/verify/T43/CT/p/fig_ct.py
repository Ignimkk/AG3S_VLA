"""T43 CT figures — t43ct-table (per-arm table, contact list, McNemar pairs), t43ct-graph (S old / release / contact bars + contact timeline per run).
usage: fig_ct.py CT_JSON OUT_DIR   (refuses to overwrite an existing figure name)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from AG3S.ag3s.experiments.common import figstyle as FS

c = json.load(open(sys.argv[1])); out = pathlib.Path(sys.argv[2]); ko = FS.use_korean()
ARMS = [a for a in ["E0", "V1", "V3", "V4", "V5", "V5D", "V7g", "V5S", "V5B"] if a in c["arms"]]
for n in ("t43ct-table", "t43ct-graph"):
    for ext in (".png", ".json"):
        assert not (out / (n + ext)).exists(), f"exists: {n}{ext}"
A = c["arms"]; P = c["pairs"]

# ---------------- table
fig = plt.figure(figsize=(15, 13)); FS.style_axes(fig, [])
gs = fig.add_gridspec(3, 1, height_ratios=[len(ARMS) + 1.6, len(c["contact_table"]) + 1.6, 5.2], hspace=0.25)
def tbl(ax, head, rows, title, widths=None):
    ax.axis("off"); ax.set_title(title, loc="left", fontsize=11, color=FS.INK)
    t = ax.table(cellText=rows, colLabels=head, loc="upper left", cellLoc="center", colLoc="center", colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.75)
    for (r, k), cell in t.get_celld().items():
        cell.set_edgecolor(FS.GRID_INK); cell.set_facecolor(FS.SURFACE if r else "#e9e8e3")
        cell.get_text().set_color(FS.INK)
        if r == 0: cell.get_text().set_fontweight("bold")
    return t
rows1 = [[a, A[a]["n"], A[a]["S_old"], A[a]["S_release"], A[a]["S_contact"], A[a]["pre_release_contact_runs"], A[a]["post_release_contact_runs"], A[a]["obstacle_stop_runs"],
          ", ".join(f"{k}:{v}" for k, v in sorted(A[a]["pre_release_contact_by_kind"].items())) or "-"] for a in ARMS]
tbl(fig.add_subplot(gs[0]), ["arm\n(H stage)", "n", "S old", "S release\n(Z4)", "S contact-\ninclusive", "runs with\npre-release contact", "runs with\npost-release contact", "runs ended by\nobstacle stop", "pre-release contact\nkind (runs)"],
    rows1, "t43ct-table (1/3)  per-arm: new_success = old S and Z4 release and zero robot/apple hurdle contact up to and including release", [.09, .04, .07, .1, .1, .15, .15, .14, .16])
rows2 = [[e["phase"], e["body"], e["kind"], e["n_runs"], " ".join(f"{k}:{v}" for k, v in e["runs_by_arm"].items()), e["first_t_min"], f'{min(e["first_t_values"])}-{max(e["first_t_values"])}', f'{e["max_pen_mm"]:.2f}', f'{e["max_force_N"]:.1f}'] for e in c["contact_table"]]
tbl(fig.add_subplot(gs[1]), ["phase", "body", "kind", "run-records", "by arm", "first t\nmin", "first t\nrange", "max penetration\n(mm)", "max force\n(N)"], rows2,
    "t43ct-table (2/3)  contacts by body and phase (run-record = one body in one phase of one run; t = control row)", [.11, .13, .09, .07, .30, .06, .08, .08, .08])
rows3 = []
for pn, v in P.items():
    for f in ("success", "success_rel", "success_ct"):
        x = v[f]; a, b = pn.split("_vs_")
        rows3.append([pn, f, x["n_pairs"], f'{a} {x["A_total"]}/{x["n_pairs"]}  {b} {x["B_total"]}/{x["n_pairs"]}', x["both"], x["A_only"], x["B_only"], x["neither"], f'{x["mcnemar_exact_p"]:.4f}'])
tbl(fig.add_subplot(gs[2]), ["pair (A vs B)", "metric", "pairs", "S totals", "both", "A only", "B only", "neither", "McNemar exact p"], rows3,
    "t43ct-table (3/3)  paired 2x2 by (episode, seed); success = old, success_rel = Z4, success_ct = contact-inclusive", [.14, .1, .05, .25, .05, .06, .06, .06, .11])
fig.suptitle(f"T43 CT contact-inclusive re-scoring  (H stage, {c['n_runs']} runs; V7g H {c['v7g_H_done']}/14)", x=0.01, ha="left", fontsize=13, color=FS.INK)
fig.savefig(out / "t43ct-table.png", dpi=130, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
side = {"figure": "t43ct-table.png", "source": str(pathlib.Path(sys.argv[1])), "per_arm": {a: {k: v for k, v in A[a].items() if not k.endswith("keys") and k != "success_ct_runs" and k != "flip_rel_to_ct"} for a in ARMS},
        "contact_table": c["contact_table"], "pairs": {pn: {f: {k: x[k] for k in x} for f, x in v.items()} for pn, v in P.items()}}
(out / "t43ct-table.json").write_text(json.dumps(side, indent=1, default=float))

# ---------------- graph
fig = plt.figure(figsize=(15, 11)); gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.45], wspace=0.22)
ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1]); FS.style_axes(fig, [ax, ax2])
x = np.arange(len(ARMS)); w = 0.27
for i, (k, lab) in enumerate((("S_old", "S old"), ("S_release", "S release (Z4)"), ("S_contact", "S contact-inclusive"))):
    v = [A[a][k] for a in ARMS]; b = ax.bar(x + (i - 1) * w, v, w * 0.92, color=FS.CATEGORICAL[[0, 3, 1][i]], label=lab)
    for xx, vv in zip(x + (i - 1) * w, v): ax.text(xx, vv + 0.2, str(vv), ha="center", fontsize=7, color=FS.INK)
ax.set_xticks(x); ax.set_xticklabels(ARMS); ax.set_ylim(0, 15.5); ax.set_ylabel("successful runs out of 14 (H stage)", color=FS.INK_2)
ax.legend(frameon=False, fontsize=9, loc="upper left"); ax.set_title("S under the three rules", loc="left", fontsize=11, color=FS.INK)
ax.grid(axis="y", color=FS.GRID_INK, lw=0.6); ax.set_axisbelow(True)
# timeline
PH = {"before_grasp": FS.CATEGORICAL[3], "carry": FS.CATEGORICAL[1], "after_release": FS.CATEGORICAL[6]}
y = 0; yt, yl = [], []; seps = []
for a in ARMS:
    L = [o for o in c["runs"].values() if o["arm"] == a]; L.sort(key=lambda o: (o["episode"], o["seed"]))
    y0 = y
    for o in L:
        ax2.plot([0, o["n_control_rows"]], [y, y], color=FS.GRID_INK, lw=1.2, zorder=1)
        if o["grasp_t"] is not None: ax2.plot(o["grasp_t"], y, marker="|", color=FS.CATEGORICAL[2], ms=8, mew=1.6, zorder=2)
        if o["release_t"] is not None: ax2.plot(o["release_t"], y, marker="|", color=FS.INK, ms=8, mew=1.6, zorder=2)
        for e in o["contacts"]:
            ax2.plot([e["first_t"] - 3, max(e["last_t"] + 1, e["first_t"] + 5)], [y, y], color=PH[e["phase"]], lw=6, solid_capstyle="butt", zorder=5)
        if o["obstacle_stop"]: ax2.plot(o["n_control_rows"], y, marker="x", color=FS.CATEGORICAL[7], ms=5, zorder=4, alpha=0.8)
        if o["success_ct"]: ax2.plot(618, y, marker="o", color=FS.CATEGORICAL[2], ms=3.5, zorder=4)
        y += 1
    yt.append((y0 + y - 1) / 2); yl.append(a); seps.append(y - 0.5); y += 0.7
ax2.set_yticks(yt); ax2.set_yticklabels(yl); ax2.invert_yaxis(); ax2.set_xlim(0, 630); ax2.set_xlabel("control row t (15 Hz)", color=FS.INK_2)
ax2.set_title("Hurdle contacts per run (robot or apple; bars drawn at least 8 rows wide), runs ordered by episode/seed", loc="left", fontsize=11, color=FS.INK)
from matplotlib.lines import Line2D
h = [Line2D([0], [0], color=PH[k], lw=4.5, label=f"contact, {k}") for k in PH] + [Line2D([0], [0], marker="|", color=FS.CATEGORICAL[2], lw=0, mew=1.6, ms=8, label="grasp_t"),
     Line2D([0], [0], marker="|", color=FS.INK, lw=0, mew=1.6, ms=8, label="release_t (Z4)"), Line2D([0], [0], marker="x", color=FS.CATEGORICAL[7], lw=0, ms=5, label="run ended by obstacle stop"),
     Line2D([0], [0], marker="o", color=FS.CATEGORICAL[2], lw=0, ms=4, label="contact-inclusive success (right edge)")]
fig.legend(handles=h, frameon=False, fontsize=8, loc="lower right", ncol=4, bbox_to_anchor=(0.98, 0.0))
fig.suptitle("T43 CT  (H stage; each row = one run; before_grasp = t < grasp_t or no grasp, carry = grasp_t..release_t, after_release = t > release_t)", x=0.01, ha="left", fontsize=11, color=FS.INK)
fig.savefig(out / "t43ct-graph.png", dpi=130, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
rows = {a: [{"key": o["key"], "n_control_rows": o["n_control_rows"], "grasp_t": o["grasp_t"], "release_t": o["release_t"], "obstacle_stop": o["obstacle_stop"], "success": o["success"],
             "success_rel": o["success_rel"], "success_ct": o["success_ct"], "contacts": [{k: e[k] for k in ("body", "phase", "first_t", "last_t", "max_pen_mm")} for e in o["contacts"]]}
            for o in sorted((o for o in c["runs"].values() if o["arm"] == a), key=lambda o: (o["episode"], o["seed"]))] for a in ARMS}
(out / "t43ct-graph.json").write_text(json.dumps({"figure": "t43ct-graph.png", "source": str(pathlib.Path(sys.argv[1])), "bars": {a: {k: A[a][k] for k in ("S_old", "S_release", "S_contact")} for a in ARMS}, "timeline_runs": rows}, indent=1, default=float))
print("korean font:", ko)
