"""T43 S2 figures.
stills-<arm>: per episode (rows) the settled grasp pose (first close command + 16 control rows) of the DEMO replay, the first E0 and the
  first E3b policy run (columns; render_s2.py, close camera from +x).  Panel text: run outcome (S success, g grasp only, - no grasp; T42-1
  rule), tip dz / tip dxy (fingertip midpoint vs fruit centre, mm), tilt (approach vs -z, deg).
graph: tip dxy vs tip dz at the settled row; marker = source (dataset demo state, DEMO replay, E0, E3b), colour = arm; one point per run.
table: per episode: dataset events, DEMO replay outcome (success / robust grasp), replay tracking (used-arm joints vs dataset state, max deg;
  final target xy vs dataset final, mm), and settled geometry medians (tip dz, tip dxy, tilt, approach yaw (mirrored), closing yaw
  (mirrored, mod 180)) for dataset / DEMO / E0 / E3b with policy outcomes.
-> AG3S/ag3s/docs/figures/t43/t43s2-stills-right.png, t43s2-stills-left.png, t43s2-graph.png, t43s2-table.png (+ .json)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle
figstyle.use_korean()
S2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S2')); OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
A = json.loads((S2 / "raw/s2_analysis.json").read_text())["episodes"]; IDX = json.loads((S2 / "render/index.json").read_text())
C = figstyle.CATEGORICAL
RIGHT = ["1834", "1977", "1623", "1215", "1959", "1984"]; LEFT = ["1925", "1828", "1830", "983", "1957", "1992"]
oc = lambda r: "S" if r["success"] else ("g" if r["grasp"] else "-")
G = lambda r: (r["geom"] or {}).get("settled")

# stills
for arm, eps in (("right", RIGHT), ("left", LEFT)):
    fig, axes = plt.subplots(len(eps), 3, figsize=(10.5, 2.9 * len(eps))); fig.patch.set_facecolor(figstyle.SURFACE)
    data = {"definitions": __doc__, "panels": {}}
    for i, ep in enumerate(eps):
        e = A[ep]; cols = [("DEMO", e["demo_replay"])]
        for cond in ("E0", "E3b"):
            rr = [r for r in e["policy_runs"] if r["cond"] == cond and G(r)]
            cols.append((cond, rr[0] if rr else None))
        for j, (cond, r) in enumerate(cols):
            ax = axes[i, j]; ax.set_axis_off()
            if r is None:
                ax.text(0.5, 0.5, f"no {cond} run", ha="center", va="center", color=figstyle.INK_2, transform=ax.transAxes); continue
            key = "DEMO" if cond == "DEMO" else cond + "_" + pathlib.Path(r["run_dir"]).name.split("_")[-1]
            p = IDX[f"{ep}/{key}"]["image"]; ax.imshow(Image.open(p)); g = G(r)
            lab = "demo replay" if cond == "DEMO" else f"{cond} {key.split('_')[1]}"
            ax.set_title(f"ep{ep} {e['fruit']}  {lab}  [{oc(r)}]\ntip dz {g['tip_dz_mm']:+.1f}  dxy {g['tip_dxy_mm']:.1f} mm  tilt {g['tilt_deg']:.1f} deg",
                         fontsize=8.5, color=figstyle.INK)
            data["panels"][f"{ep}/{key}"] = {"image": p, "outcome": oc(r), "t": g["t"], **{k: g[k] for k in ("tip_dz_mm", "tip_dxy_mm", "tilt_deg")}}
    fig.suptitle(f"T43 S2 {arm} arm: grasp pose at close + 16 control rows; dataset actions replayed open loop (left) vs policy rollouts", color=figstyle.INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(OUT / f"t43s2-stills-{arm}.png", dpi=105, facecolor=figstyle.SURFACE)
    (OUT / f"t43s2-stills-{arm}.json").write_text(json.dumps(data, indent=1)); print(OUT / f"t43s2-stills-{arm}.png")

# graph
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True); figstyle.style_axes(fig, axes)
gd = {"definitions": __doc__, "points": []}
MK = {"dataset": "s", "DEMO": "o", "E0": "^", "E3b": "v"}
for ax, (arm, eps) in zip(axes, (("right", RIGHT), ("left", LEFT))):
    for src, mk in MK.items():
        X, Y = [], []
        for ep in eps:
            e = A[ep]
            if src == "dataset":
                g = (e["dataset_geom"] or {}).get("settled"); runs = [(None, g)] if g else []
            elif src == "DEMO":
                runs = [(e["demo_replay"], G(e["demo_replay"]))]
            else:
                runs = [(r, G(r)) for r in e["policy_runs"] if r["cond"] == src and G(r)]
            for r, g in runs:
                if g is None:
                    continue
                X.append(g["tip_dxy_mm"]); Y.append(g["tip_dz_mm"])
                gd["points"].append({"arm": arm, "episode": ep, "source": src, "outcome": None if r is None else oc(r), "tip_dxy_mm": g["tip_dxy_mm"], "tip_dz_mm": g["tip_dz_mm"]})
        col = {"dataset": figstyle.INK_2, "DEMO": C[0], "E0": C[1], "E3b": C[2]}[src]
        ax.scatter(X, Y, marker=mk, s=36, facecolors="none" if src == "dataset" else col, edgecolors=col, label=f"{src} (n={len(X)})", lw=1.2)
    ax.set_title(f"{arm} arm ({len(eps)} episodes)", color=figstyle.INK, fontsize=10)
    ax.set_xlabel("tip dxy from fruit centre (mm)"); ax.set_ylabel("tip dz from fruit centre (mm)")
    ax.axhline(0, color=figstyle.GRID_INK, lw=0.8); ax.legend(fontsize=8, frameon=False)
ax.set_ylim(-35, 100)
fig.suptitle("T43 S2: fingertip midpoint vs fruit centre at close + 16 rows (dataset state, demo replay, policy E0 / E3b; one point per run)", color=figstyle.INK, fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "t43s2-graph.png", dpi=120, facecolor=figstyle.SURFACE)
(OUT / "t43s2-graph.json").write_text(json.dumps(gd, indent=1)); print(OUT / "t43s2-graph.png")

# table
HEAD = ["ep", "arm", "fruit", "split", "DEMO replay", "robust grasp", "track max deg", "final xy vs ds (mm)",
        "tip dz ds / DEMO / E0 / E3b", "tip dxy ds / DEMO / E0 / E3b", "tilt ds / DEMO / E0 / E3b", "appr yaw DEMO / E0 / E3b", "E0 runs", "E3b runs"]
med = lambda v: "-" if not v else f"{float(np.median(v)):.1f}"
rows, td = [], {"definitions": __doc__, "rows": []}
for ep in RIGHT + LEFT:
    e = A[ep]; d = e["demo_replay"]; ds = (e["dataset_geom"] or {}).get("settled") or {}; gD = G(d) or {}
    P = {c: [G(r) for r in e["policy_runs"] if r["cond"] == c and G(r)] for c in ("E0", "E3b")}
    O = {c: "".join(oc(r) for r in e["policy_runs"] if r["cond"] == c) or "none" for c in ("E0", "E3b")}
    f = lambda k: " / ".join([f"{ds[k]:.1f}" if k in ds else "-", f"{gD[k]:.1f}" if k in gD else "-", med([g[k] for g in P["E0"]]), med([g[k] for g in P["E3b"]])])
    tr = e["demo_tracking"]
    r = [ep, e["arm"], e["fruit"], e["split"], oc(d), "yes" if d["robust_grasp"] else "no", f"{tr['arm_joint_maxabs_deg_all']:.2f}",
         f"{tr['final_target_xy_vs_dataset_final_mm']:.1f}", f("tip_dz_mm"), f("tip_dxy_mm"), f("tilt_deg"),
         " / ".join([f"{gD['approach_yaw_deg_mirrored']:.1f}" if gD else "-", med([g["approach_yaw_deg_mirrored"] for g in P["E0"]]), med([g["approach_yaw_deg_mirrored"] for g in P["E3b"]])]),
         O["E0"], O["E3b"]]
    rows.append(r); td["rows"].append(dict(zip(HEAD, r)))
fig, ax = plt.subplots(figsize=(22, 0.42 * len(rows) + 1.6)); fig.patch.set_facecolor(figstyle.SURFACE); ax.set_axis_off()
t = ax.table(cellText=rows, colLabels=HEAD, loc="center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.35)
for (i, j), cell in t.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if i == 0:
        cell.set_facecolor(figstyle.SEQ_BLUE[0]); cell.set_text_props(color=figstyle.INK)
    elif rows[i - 1][1] == "right":
        cell.set_facecolor("#f3f2ee")
ax.set_title("T43 S2: dataset actions replayed open loop in the client scene (same reset + 1000-step warmup) vs policy rollouts (T42-1 / T39). "
             "S = success, g = grasp only, - = no grasp (T42-1 rule); geometry at close + 16 rows, medians over runs; ds = dataset state", color=figstyle.INK, fontsize=9.5)
fig.tight_layout(); fig.savefig(OUT / "t43s2-table.png", dpi=130, facecolor=figstyle.SURFACE)
(OUT / "t43s2-table.json").write_text(json.dumps(td, indent=1)); print(OUT / "t43s2-table.png")
