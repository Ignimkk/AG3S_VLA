"""T28 Phase 2R figures: success grid table, apple z timelines, refined-reference per chunk, latency.
Reads p2r/{E0,E1}_old_summary.json and p2r/{E0,E1,E2}_rec_summary.json (analyze_p2r.py). Sidecar .json per png."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as ml
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
fs.use_korean()
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/p2r'))
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t28'))
C = fs.CATEGORICAL
EPS = (1807, 1800, 1808)
SRC = {}
for cond in ("E0", "E1", "E2"):
    for kind in ("rec", "old"):
        p = V / f"{cond}_{kind}_summary.json"
        if p.exists():
            SRC[(cond, kind)] = json.loads(p.read_text())


def new(name):
    p = F / name
    assert not p.exists(), f"{p} exists"
    return p


def status(r):
    return "success" if r["success"] else ("grasp_only" if r["grasp"] else "no_grasp")


COL = {"success": C[2], "grasp_only": C[3], "no_grasp": C[7]}

# ---------------- 1. success grid table
rows_src = [(c, k) for c in ("E0", "E1", "E2") for k in ("rec", "old") if (c, k) in SRC]
cell, colors, side = [], [], {"cells": {}, "summary": {}}
for c, k in rows_src:
    runs = SRC[(c, k)]["runs"]; line = []; lc = []
    for ep in EPS:
        for rk in (1, 2):
            r = runs.get(f"{c}_ep{ep}_r{rk}")
            if r is None:
                line.append("-"); lc.append(fs.SURFACE); continue
            s = status(r)
            txt = {"success": f"S  g{r['grasp_t']} p{r['place_t']}", "grasp_only": f"G  g{r['grasp_t']}  no place",
                   "no_grasp": f"F  dz {r['apple_dz_max_mm']:.0f}mm"}[s]
            line.append(txt); lc.append(COL[s] + "55")
            side["cells"][f"{c}_{k}_ep{ep}_r{rk}"] = {"status": s, "grasp_t": r["grasp_t"], "place_t": r["place_t"],
                                                      "apple_dz_max_mm": r["apple_dz_max_mm"], "run_dir": r["run_dir"]}
    ns = sum(1 for r in runs.values() if r["success"]); ng = sum(1 for r in runs.values() if r["grasp"])
    line.append(f"{ns}/{len(runs)}  (grasp {ng})"); lc.append(fs.SURFACE)
    cell.append(line); colors.append(lc)
    side["summary"][f"{c}_{k}"] = {"n_success": ns, "n_grasp": ng, "n_runs": len(runs)}
for c in ("E0", "E1", "E2"):
    runs = [r for k in ("rec", "old") if (c, k) in SRC for r in SRC[(c, k)]["runs"].values()]
    side["summary"][f"{c}_pooled"] = {"n_success": sum(r["success"] for r in runs), "n_grasp": sum(r["grasp"] for r in runs), "n_runs": len(runs),
                                      "per_episode_success": {str(ep): f"{sum(r['success'] for r in runs if r['episode_index']==ep)}/{sum(1 for r in runs if r['episode_index']==ep)}" for ep in EPS}}
fig, axes = plt.subplots(2, 1, figsize=(15, 5.2), gridspec_kw={"height_ratios": [len(rows_src) + 1, 4]})
for ax in axes:
    ax.axis("off")
lab = {"rec": "recorded 2026-09-29", "old": "unrecorded 2026-09-28"}
tb = axes[0].table(cellText=cell, cellColours=colors, rowLabels=[f"{c}  {lab[k]}" for c, k in rows_src],
                   colLabels=[f"ep{ep} r{rk}" for ep in EPS for rk in (1, 2)] + ["success / runs"], loc="upper center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.5)
axes[0].set_title("T28 P2R success grid (S success: g grasp t, p place t | G grasp, no place | F no grasp: max apple dz)", fontsize=10, color=fs.INK)
srows = []
for c in ("E0", "E1", "E2"):
    rr = side["summary"].get(f"{c}_rec", {}); pp = side["summary"][f"{c}_pooled"]
    srows.append([c, f"{rr.get('n_success','-')}/{rr.get('n_runs','-')}", f"{pp['n_success']}/{pp['n_runs']}", f"{pp['n_grasp']}/{pp['n_runs']}"]
                 + [pp["per_episode_success"][str(ep)] for ep in EPS])
tb2 = axes[1].table(cellText=srows, colLabels=["condition", "success recorded", "success pooled", "grasp pooled", "ep1807 pooled", "ep1800 pooled", "ep1808 pooled"],
                    loc="upper center", cellLoc="center")
tb2.auto_set_font_size(False); tb2.set_fontsize(9); tb2.scale(1, 1.4)
fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
p = new("t28-p2-success-grid.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
p.with_suffix(".json").write_text(json.dumps(side, indent=1))

# ---------------- 2. apple z timelines per condition
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
side = {}
for i, c in enumerate(("E0", "E1", "E2")):
    ax = axes[i]
    for k in ("old", "rec"):
        if (c, k) not in SRC:
            continue
        for name, r in SRC[(c, k)]["runs"].items():
            tl = r["timeline_control"]; col = C[EPS.index(r["episode_index"])]
            ls = "-" if name.endswith("r1") else "--"
            ax.plot(tl["t"], tl["apple_dz_mm"], ls, color=col, lw=1.7 if k == "rec" else 0.9, alpha=1.0 if k == "rec" else 0.45)
            side[f"{c}_{k}_{name}"] = {"status": status(r), "grasp_t": r["grasp_t"], "place_t": r["place_t"], "apple_dz_max_mm": r["apple_dz_max_mm"],
                                       "t_apple_dz_max": r["t_apple_dz_max"], "t_first_close_cmd": r["t_first_close_cmd"]}
    ax.axhline(50, color=fs.INK_2, lw=0.8, ls=":")
    lows = {f"{k}:{n}": (min(r["timeline_control"]["apple_dz_mm"]), r["timeline_control"]["t"][int(np.argmin(r["timeline_control"]["apple_dz_mm"]))])
            for k in ("old", "rec") if (c, k) in SRC for n, r in SRC[(c, k)]["runs"].items() if min(r["timeline_control"]["apple_dz_mm"]) < -60}
    ax.set_ylim(-60, 275)
    for j, (n, (v, tv)) in enumerate(lows.items()):
        ax.annotate(f"{n}: apple dz min {v:.0f} mm at t={tv} (below axis)", xy=(tv, -60), xytext=(tv + 20, -45 + 18 * j), fontsize=8, color=fs.INK_2,
                    arrowprops={"arrowstyle": "->", "color": fs.INK_2, "lw": 0.8})
        side[f"{c}_below_axis_{n}"] = {"apple_dz_min_mm": v, "t": tv}
    ns = {k: (sum(r["success"] for r in SRC[(c, k)]["runs"].values()), len(SRC[(c, k)]["runs"])) for k in ("rec", "old") if (c, k) in SRC}
    ax.set_title(f"{c}: success recorded {ns['rec'][0]}/{ns['rec'][1]}" + (f", unrecorded {ns['old'][0]}/{ns['old'][1]}" if "old" in ns else ""), fontsize=10, color=fs.INK)
    ax.set_ylabel("apple z - z0 (mm)")
axes[-1].set_xlabel("control t_step (15 Hz)")
h = [ml.Line2D([], [], color=C[j], lw=1.7, label=f"ep{ep}") for j, ep in enumerate(EPS)]
h += [ml.Line2D([], [], color=fs.INK_2, lw=1.7, ls="-", label="r1"), ml.Line2D([], [], color=fs.INK_2, lw=1.7, ls="--", label="r2"),
      ml.Line2D([], [], color=fs.INK_2, lw=1.7, label="recorded (thick)"), ml.Line2D([], [], color=fs.INK_2, lw=0.9, alpha=0.45, label="unrecorded (thin)")]
axes[0].legend(handles=h, ncol=7, fontsize=8, frameon=False, loc="upper right")
fig.suptitle("T28 P2R apple z timeline (control-row qpos, 50 mm grasp line dotted)", fontsize=11, color=fs.INK)
fs.style_axes(fig, axes); fig.tight_layout()
p = new("t28-p2-apple-z.png"); fig.savefig(p, dpi=130); plt.close(fig)
p.with_suffix(".json").write_text(json.dumps(side, indent=1))

# ---------------- 3. refined - reference per chunk (+ E2 limit overshoot, SQP iterations)
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
side = {}
for c, ax in (("E1", axes[0]), ("E2", axes[1])):
    for k in ("old", "rec"):
        if (c, k) not in SRC or "server_records" not in SRC[(c, k)]:
            continue
        for name, sr in SRC[(c, k)]["server_records"].items():
            ep = int(name.split("_ep")[1][:4]) if "_ep" in name else None
            t = [8 * (x["seq"] - 1) for x in sr["chunks"]]; dv = [x["max_abs_deviation_deg"] for x in sr["chunks"]]
            ax.plot(t, dv, "-" if name.endswith("r1") else "--", color=C[EPS.index(ep)] if ep in EPS else fs.INK_2,
                    lw=1.4 if k == "rec" else 0.8, alpha=1 if k == "rec" else 0.5, marker="o", ms=2)
            side[f"{c}_{k}_{name}"] = {"deviation_deg": sr["deviation_deg"], "per_chunk_max_abs_deviation_deg": dv, "t": t,
                                      "sqp_returned_counts": sr["sqp_returned_counts"], "sqp_iterations_counts": sr["sqp_iterations_counts"],
                                      "limit_overshoot_reference_max": sr["limit_overshoot_reference_max"], "limit_overshoot_refined_max": sr["limit_overshoot_refined_max"]}
            if c == "E2" and k == "rec":
                axes[2].plot(t, [x["limit_overshoot_reference"]["position"] for x in sr["chunks"]], ":", color=C[EPS.index(ep)], lw=1)
                axes[2].plot(t, [x["limit_overshoot_refined"]["position"] for x in sr["chunks"]], "-" if name.endswith("r1") else "--", color=C[EPS.index(ep)], lw=1.4)
                axes[3].plot(t, [x["sqp_iterations"] for x in sr["chunks"]], "-" if name.endswith("r1") else "--", color=C[EPS.index(ep)], lw=1, marker="o", ms=2, drawstyle="steps-post")
    ax.set_ylabel(f"{c}: max |refined-ref| (deg)")
axes[0].set_title("server record max_abs_deviation_deg per chunk (8 planned steps, 14 arm joints); thick recorded, thin unrecorded", fontsize=9.5, color=fs.INK)
axes[1].set_title("E2 (--no-perception, default objective + limits), recorded runs", fontsize=9.5, color=fs.INK)
axes[2].set_ylabel("E2 position overshoot (rad)"); axes[2].set_title("E2 limit_overshoot position: reference dotted, refined solid/dashed", fontsize=9.5, color=fs.INK)
axes[3].set_ylabel("E2 SQP iterations"); axes[3].set_xlabel("chunk start t_step (8 x (seq-1))")
h = [ml.Line2D([], [], color=C[j], lw=1.5, label=f"ep{ep}") for j, ep in enumerate(EPS)]
axes[0].legend(handles=h, ncol=3, fontsize=8, frameon=False)
fs.style_axes(fig, axes); fig.tight_layout()
p = new("t28-p2-refined-minus-reference.png"); fig.savefig(p, dpi=130); plt.close(fig)
p.with_suffix(".json").write_text(json.dumps(side, indent=1))

# ---------------- 4. latency: client round trip + server split
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
side = {}
data, labels = [], []
for c in ("E0", "E1", "E2"):
    for k in ("rec", "old"):
        if (c, k) not in SRC:
            continue
        v = []
        for name, r in SRC[(c, k)]["runs"].items():
            rows = [json.loads(l) for l in (pathlib.Path(r["run_dir"]) / "frames.jsonl").read_text().splitlines() if l.strip()]
            inf = [x["timing_ms"]["policy_infer"] for x in rows if x["kind"] == "planning" and x.get("timing_ms")]
            v += inf[1:]
        data.append(v); labels.append(f"{c} {k}")
        side[f"{c}_{k}_client_round_trip_ms_excl_first"] = {"n": len(v), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}
bp = axes[0].boxplot(data, labels=labels, showfliers=True, patch_artist=True, flierprops={"ms": 2})
for j, b in enumerate(bp["boxes"]):
    b.set_facecolor(C[["E0", "E1", "E2"].index(labels[j][:2])] + "66")
axes[0].set_ylabel("client round trip policy_infer (ms), first chunk of each run excluded")
axes[0].set_title("client round trip per chunk", fontsize=10, color=fs.INK)
sd, sl = [], []
for c in ("E1", "E2"):
    for k in ("rec", "old"):
        if (c, k) not in SRC:
            continue
        for part in ("infer", "trajopt", "total"):
            v = []
            for name, r in SRC[(c, k)]["runs"].items():
                rows = [json.loads(l) for l in (pathlib.Path(r["run_dir"]) / "frames.jsonl").read_text().splitlines() if l.strip()]
                vt = [(x.get("verdict") or {}).get("timing_ms") for x in rows if x["kind"] == "planning"]
                v += [x[part] for x in vt[1:] if x]
            sd.append(v); sl.append(f"{c} {k}\n{part}")
            side[f"{c}_{k}_server_{part}_ms_excl_first"] = {"n": len(v), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}
bp = axes[1].boxplot(sd, labels=sl, showfliers=True, patch_artist=True, flierprops={"ms": 2})
for j, b in enumerate(bp["boxes"]):
    b.set_facecolor(C[["E0", "E1", "E2"].index(sl[j][:2])] + "66")
axes[1].set_ylabel("server verdict.timing_ms (ms)"); axes[1].set_title("server split (E1/E2): infer / trajopt / total", fontsize=10, color=fs.INK)
axes[1].tick_params(axis="x", labelsize=7)
side["note"] = "recorded runs (rec) ran E0/E1/E2 concurrently on one GPU (3 servers + 3 clients); unrecorded runs (old) ran one condition at a time"
fs.style_axes(fig, axes); fig.tight_layout()
p = new("t28-p2-latency.png"); fig.savefig(p, dpi=130); plt.close(fig)
p.with_suffix(".json").write_text(json.dumps(side, indent=1))
print("done")
