"""T30 closed-loop figures (copied from T28/p3/fig_e3.py: paths -> T30, grid adds E3a/E3b before T30 from T28/p3, bands add
association / exclusion-ball rows from p/<COND>_analysis_extra.json). Original docstring: Reads p2r/{E0,E1}_{rec,old}_summary.json, p2r/E2_rec_summary.json (Phase 2R) and
p3/{E3a,E3b}_analysis.json + _summary.json (this phase).
  t30-success-grid.png      : condition x episode grid E0..E3b
  t30-apple-z.png           : apple z timelines E3a / E3b (control-row qpos)
  t30-bands-<COND>.png      : per chunk bands per run — manipulated label (nearest object centre of the manipulated centroid),
                                 worst-row nearest GT object (x = inside apple), gate (E3a would_hold kind / E3b HOLD kind),
                                 latch state, refined-reference deg, GT apple dz
  t30-keyframes-<COND>.png  : mp4 keyframe contact sheet (same columns as t28-p2-keyframes-*)
  t30-latency.png           : server time split infer / AG3S / ESDF / TO per condition
Every png has a .json sidecar with the numbers drawn."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import imageio.v2 as iio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as ml
from matplotlib.patches import Patch
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
fs.use_korean()
V2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/p2r'))
V28 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/p3'))
V3 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30/p'))
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t30'))
C = fs.CATEGORICAL
EPS = (1807, 1800, 1808)
WHAT = sys.argv[1:] or ["grid", "applez", "bands", "keyframes", "latency"]


def new(name):
    p = F / name
    assert not p.exists(), f"{p} exists"
    return p


def status(r):
    return "success" if r["success"] else ("grasp_only" if r["grasp"] else "no_grasp")


COL = {"success": C[2], "grasp_only": C[3], "no_grasp": C[7]}
SRC = {}
for cond in ("E0", "E1", "E2"):
    for kind in ("rec", "old"):
        p = V2 / f"{cond}_{kind}_summary.json"
        if p.exists():
            SRC[(cond, kind)] = json.loads(p.read_text())["runs"]
AN, SU, EX = {}, {}, {}
for cond in ("E3a", "E3b"):
    SRC[(cond, "pre")] = {k: v["summary"] for k, v in json.loads((V28 / f"{cond}_analysis.json").read_text())["runs"].items()}
    p = V3 / f"{cond}_analysis.json"
    if p.exists():
        AN[cond] = json.loads(p.read_text())
        SRC[(cond, "rec")] = {k: v["summary"] for k, v in AN[cond]["runs"].items()}
        SU[cond] = json.loads((V3 / f"{cond}_summary.json").read_text())
        px = V3 / f"{cond}_analysis_extra.json"
        EX[cond] = json.loads(px.read_text())["runs"] if px.exists() else {}

# duplicates (dup_check.json, dup_check_cross.json): same policy RNG sequence; 'rN' = same condition's run N, other text = note only
DUPS = {"E3a_ep1800_r2": "r1", "E3a_ep1808_r2": "r1", "E3a_ep1807_r2": "r1 RNG*",
        "E3a_ep1800_r3": "E3b ep1800 r1"}
DUPS_PRE = {"E3a_ep1807_r2": "r1"}  # T28 P3 E3a (dup_check.json)
if "grid" in WHAT:
    order = [(c, k) for c in ("E0", "E1", "E2") for k in ("rec", "old") if (c, k) in SRC] + \
            [(c, k) for k in ("pre", "rec") for c in ("E3a", "E3b") if (c, k) in SRC]
    cell, colors, side = [], [], {"cells": {}, "summary": {}}
    for c, k in order:
        runs = SRC[(c, k)]; line = []; lc = []
        for ep in EPS:
            for rk in (1, 2, 3):
                r = runs.get(f"{c}_ep{ep}_r{rk}")
                if r is None:
                    line.append("-"); lc.append(fs.SURFACE); continue
                s = status(r)
                txt = {"success": f"S  g{r['grasp_t']} p{r['place_t']}", "grasp_only": f"G  g{r['grasp_t']}  no place",
                       "no_grasp": f"F  dz {r['apple_dz_max_mm']:.0f}mm"}[s]
                dk = f"{c}_ep{ep}_r{rk}"
                dd = DUPS if k == "rec" else (DUPS_PRE if k == "pre" else {})
                if dk in dd:
                    txt += f"  (={dd[dk]})"
                    side["cells"].setdefault("_duplicates", {})[f"{k}:{dk}"] = dd[dk]
                line.append(txt); lc.append(COL[s] + "55")
                side["cells"][f"{c}_{k}_ep{ep}_r{rk}"] = {"status": s, "grasp_t": r["grasp_t"], "place_t": r["place_t"],
                                                          "apple_dz_max_mm": r["apple_dz_max_mm"], "run_dir": r["run_dir"]}
        ns = sum(1 for r in runs.values() if r["success"]); ng = sum(1 for r in runs.values() if r["grasp"])
        dd = DUPS if k == "rec" else (DUPS_PRE if k == "pre" else {})
        dist = {n: r for n, r in runs.items() if not (n in dd and dd[n].startswith("r"))}
        nsd = sum(1 for r in dist.values() if r["success"])
        line.append(f"{ns}/{len(runs)}  (grasp {ng})" + (f"\ndistinct {nsd}/{len(dist)}" if len(dist) != len(runs) else "")); lc.append(fs.SURFACE)
        side["summary"][f"{c}_{k}_distinct"] = {"n_success": nsd, "n_runs": len(dist), "excluded": sorted(set(runs) - set(dist))}
        cell.append(line); colors.append(lc)
        side["summary"][f"{c}_{k}"] = {"n_success": ns, "n_grasp": ng, "n_runs": len(runs)}
    lab = {"rec": "recorded", "old": "unrecorded 09-28", "pre": "before T30 (T28 P3)"}
    note = {"E0": "VLA only", "E1": "TO tracking only", "E2": "+objective/limits", "E3a": "+distance field, gate off", "E3b": "+distance field, gate reasons (alone)"}
    fig, ax = plt.subplots(figsize=(22, 0.62 * len(order) + 1.6)); ax.axis("off")
    tb = ax.table(cellText=cell, cellColours=colors, rowLabels=[f"{c} {'after T30' if (k == 'rec' and c.startswith('E3')) else lab[k]} — {note[c]}" for c, k in order],
                  colLabels=[f"ep{ep} r{rk}" for ep in EPS for rk in (1, 2, 3)] + ["success / runs"], loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.9)
    ax.set_title("E0-E2 (T28 P2R), E3a/E3b before T30 (T28 P3) and after T30 success grid (S success: g grasp t, p place t | G grasp, no place | F no grasp: max apple dz; (=x) duplicate sample, * diverges from r1 at seq 33)", fontsize=10, color=fs.INK)
    fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
    p = new("t30-success-grid.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)

if "applez" in WHAT:
    fig, axes = plt.subplots(len(AN), 1, figsize=(12, 3.6 * len(AN)), sharex=True, squeeze=False)
    side = {}
    for i, c in enumerate(AN):
        ax = axes[i, 0]
        for name, r in AN[c]["runs"].items():
            s = r["summary"]; tl = s["timeline_control"]; col = C[EPS.index(s["episode_index"])]
            ax.plot(tl["t"], tl["apple_dz_mm"], {"r1": "-", "r2": "--", "r3": ":"}[name[-2:]], color=col, lw=1.6)
            if s["t_first_close_cmd"] is not None:
                ax.axvline(s["t_first_close_cmd"], color=col, lw=0.6, ls=":")
            side[f"{c}_{name}"] = {"status": status(s), "grasp_t": s["grasp_t"], "place_t": s["place_t"], "apple_dz_max_mm": s["apple_dz_max_mm"],
                                   "t_apple_dz_max": s["t_apple_dz_max"], "t_first_close_cmd": s["t_first_close_cmd"],
                                   "apple_dz_min_mm": float(min(tl["apple_dz_mm"]))}
        ax.axhline(50, color=fs.INK_2, lw=0.8, ls=":")
        lo = min(min(r["summary"]["timeline_control"]["apple_dz_mm"]) for r in AN[c]["runs"].values())
        ax.set_ylim(max(lo - 10, -80), 275)
        ns = sum(r["summary"]["success"] for r in AN[c]["runs"].values())
        ax.set_title(f"{c}: success {ns}/{len(AN[c]['runs'])} (dotted verticals: first close command per run)", fontsize=10, color=fs.INK)
        ax.set_ylabel("apple z - z0 (mm)")
    axes[-1, 0].set_xlabel("control t_step (15 Hz)")
    h = [ml.Line2D([], [], color=C[j], lw=1.6, label=f"ep{ep}") for j, ep in enumerate(EPS)]
    h += [ml.Line2D([], [], color=fs.INK_2, ls="-", label="r1"), ml.Line2D([], [], color=fs.INK_2, ls="--", label="r2")]
    axes[0, 0].legend(handles=h, ncol=5, fontsize=8, frameon=False, loc="upper right")
    fig.suptitle("T30 apple z timeline E3a / E3b (control-row qpos, 50 mm grasp line dotted)", fontsize=11, color=fs.INK)
    fs.style_axes(fig, axes.ravel()); fig.tight_layout()
    p = new("t30-apple-z.png"); fig.savefig(p, dpi=130); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)

OBJ_COL = {"apple": C[1], "crate": C[0], "table": fs.GRID_INK, "orange": C[3], "pear": C[5], "banana": C[4], "none": fs.INK_2,
           "shelf": C[6], "office": C[6], "world": C[6]}
KIND_COL = {"collision": C[7], "uncertified": C[3], "unverified": C[6], "comms": C[4], "budget_only": C[2], "none": fs.SURFACE}
LATCH_COL = {"searching": fs.GRID_INK, "latched": C[0], "closing": C[3], "held": C[2], "placed": C[5], "released": C[6], "None": fs.SURFACE}

if "bands" in WHAT:
    for c in AN:
        runs = AN[c]["runs"]
        names = [f"{c}_ep{ep}_r{k}" for ep in EPS for k in (1, 2, 3) if f"{c}_ep{ep}_r{k}" in runs]
        fig, axes = plt.subplots(len(names), 1, figsize=(15, 2.8 * len(names)), sharex=True, squeeze=False)
        side = {"condition": c, "rows": ["manipulated (nearest centre)", "worst row nearest object (x inside apple)",
                                         "E3a would_hold kind / E3b HOLD kind", "latch state"], "runs": {}}
        for i, name in enumerate(names):
            ax = axes[i, 0]; ch = runs[name]["chunks"]; s = runs[name]["summary"]
            rec = {"t": [], "manipulated": [], "worst": [], "worst_inside_apple": [], "gate_kind": [], "latch": [], "refined_minus_reference_deg": [], "apple_dz_mm": []}
            for e in ch:
                t = e["t"]
                man = (e.get("manipulated") or {}).get("nearest_centre", "none") if "server_chunk" in e else "none"
                wo = e["worst"].get("nearest_body") or "none"
                g = e.get("gate") or {}
                if c == "E3a":
                    ks = g.get("would_hold_kinds") or []
                else:
                    ks = (g.get("hold_kinds") or g.get("kinds") or []) if e["n_executed"] < e["n_control"] else []
                if e.get("ipc") in ("timeout", "stale", "error"):
                    ks = ["comms"]
                k0 = ks[0] if ks else "none"
                la = str((e.get("grasp_latch") or {}).get("state"))
                xe = next((x for x in EX.get(c, {}).get(name, []) if x["seq"] == e["seq"]), {})
                bv = xe.get("ball_vs_gt_apple")
                ballc = fs.INK_2 if bv is None else (C[2] if bv["whole_inside"] else (C[3] if bv["centre_inside"] else C[7]))
                ao = (xe.get("association") or {}).get("outcome")
                rs = (xe.get("decision") or {}).get("reason")
                assc = C[7] if rs == "association_rejected" else (C[4] if rs == "subset_kept_anchor" else (fs.GRID_INK if ao else fs.SURFACE))
                for y, colr in ((5, assc), (4, ballc), (3, OBJ_COL.get(man, C[6])), (2, OBJ_COL.get(wo, C[6])), (1, KIND_COL.get(k0.split(":")[0], C[6])), (0, LATCH_COL.get(la, C[6]))):
                    ax.add_patch(plt.Rectangle((t, y), 8, 0.85, color=colr, lw=0))
                rec.setdefault("ball", []).append(None if bv is None else ("whole" if bv["whole_inside"] else ("centre" if bv["centre_inside"] else "outside")))
                rec.setdefault("association_reason", []).append(rs)
                if e["worst"].get("inside_apple"):
                    ax.plot(t + 4, 2.42, "x", color=fs.INK, ms=5, mew=1.2)
                for kk, vv in (("t", t), ("manipulated", man), ("worst", wo), ("worst_inside_apple", e["worst"].get("inside_apple")), ("gate_kind", ks),
                               ("latch", la), ("refined_minus_reference_deg", e.get("refined_minus_reference_deg")), ("apple_dz_mm", e["apple_dz_mm"])):
                    rec[kk].append(vv)
            ax2 = ax.twinx()
            ax2.plot([t + 4 for t in rec["t"]], rec["refined_minus_reference_deg"], color=fs.INK, lw=0.9)
            ax2.plot([t + 4 for t in rec["t"]], [x / 25 for x in rec["apple_dz_mm"]], color=C[2], lw=1.2, ls="--")
            ax2.set_ylim(-1, 12); ax2.set_ylabel("|ref-refined| deg (black)\napple dz/25 mm (green)", fontsize=7)
            if s["t_first_close_cmd"] is not None:
                ax.axvline(s["t_first_close_cmd"], color=fs.INK, lw=0.8, ls=":")
            ax.set_ylim(-0.1, 6); ax.set_yticks([0.42, 1.42, 2.42, 3.42, 4.42, 5.42]); ax.set_yticklabels(["latch", "HOLD/would", "worst obj", "manipulated", "ball vs apple", "association"], fontsize=7)
            ax.set_xlim(0, 600)
            res = status(s)
            ax.set_title(f"{name}: {res}  grasp t {s['grasp_t']}  place t {s['place_t']}  first close cmd t {s['t_first_close_cmd']}", fontsize=9, color=fs.INK)
            side["runs"][name] = rec
        axes[-1, 0].set_xlabel("control t_step (chunk start; 8 steps per chunk)")
        hs = [Patch(color=v, label=f"obj {k}") for k, v in OBJ_COL.items() if k in ("apple", "crate", "table", "orange", "pear", "banana", "none")]
        hs += [Patch(color=v, label=f"kind {k}") for k, v in KIND_COL.items() if k not in ("none", "budget_only")]
        hs += [Patch(color=v, label=f"latch {k}") for k, v in LATCH_COL.items() if k != "None"]
        hs += [Patch(color=C[2], label="ball: whole apple in"), Patch(color=C[3], label="ball: centre only"), Patch(color=C[7], label="ball: apple out"),
               Patch(color=fs.INK_2, label="no ball"), Patch(color=C[7], label="assoc rejected"), Patch(color=C[4], label="subset_kept_anchor"),
               Patch(color=fs.GRID_INK, label="assoc accepted")]
        fig.legend(handles=hs, ncol=9, fontsize=7, frameon=False, loc="lower center")
        fig.suptitle(f"T30 {c} per-chunk bands (" + ("would_hold kinds, gate off" if c == "E3a" else "HOLD kinds, gate reasons + fixed hold") + ")", fontsize=11)
        fs.style_axes(fig, axes.ravel()); fig.tight_layout(rect=(0, 0.05, 1, 1))
        p = new(f"t30-bands-{c}.png"); fig.savefig(p, dpi=115); plt.close(fig)
        p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=str)); print(p)

if "keyframes" in WHAT:
    for cond in AN:
        S = {k: v["summary"] for k, v in AN[cond]["runs"].items()}
        order = [f"{cond}_ep{ep}_r{k}" for ep in EPS for k in (1, 2, 3) if f"{cond}_ep{ep}_r{k}" in S]
        fig, axes = plt.subplots(len(order), 5, figsize=(16, 2.55 * len(order)))
        axes = np.atleast_2d(axes)
        side = {"condition": cond, "frame_mapping": "mp4 frame k = state after control step k = control row t=k+1; keyframe for row t uses frame max(t-1,0) (p2r/fig_keyframes.py)", "rows": {}}
        for i, name in enumerate(order):
            r = S[name]
            rd = iio.get_reader(r["mp4"]); n = rd.count_frames()
            gt = r.get("grasp_t_control_rows")
            ts = [0, r["t_first_close_cmd"], gt if gt is not None else r["t_min_finger_to_apple"], r["t_apple_dz_max"], n]
            labs = ["start", "close cmd", "grasp" if gt is not None else "closest (no grasp)", "max lift", f"last frame {n-1}"]
            idx = [None if t is None else min(max(t - 1, 0), n - 1) for t in ts]
            tl = r["timeline_control"]; row = []
            for j, (t, k, lab) in enumerate(zip(ts, idx, labs)):
                ax = axes[i, j]; ax.set_xticks([]); ax.set_yticks([])
                if k is None:
                    ax.text(0.5, 0.5, "none", ha="center", va="center", transform=ax.transAxes, color=fs.INK_2)
                    row.append({"col": lab, "t": None, "frame": None}); continue
                img = rd.get_data(k); ax.imshow(img[40:420, 60:600])
                tt = min(t, len(tl["t"]) - 1)
                dz = tl["apple_dz_mm"][tt]; gm = tl["grip_l_measured"][tt]; fa = tl["finger_to_apple_surface_mm"][tt]
                ax.set_title(f"{lab}  " + (f"t={t}  " if j < 4 else "") + f"dz={dz:.0f}mm  grip={gm:.2f}  f-a={'>=1000' if fa >= 999.9 else format(fa, '.0f')}mm", fontsize=7.5, color=fs.INK)
                row.append({"col": lab, "t": t, "frame": k, "apple_dz_mm": dz, "grip_l_measured": gm, "finger_to_apple_mm": fa})
                for sp in ax.spines.values():
                    sp.set_color(COL[status(r)]); sp.set_linewidth(2)
            axes[i, 0].set_ylabel(f"{name.replace(cond + '_', '')}\n{status(r)}", fontsize=8.5, color=fs.INK)
            side["rows"][name] = {"success": r["success"], "grasp": r["grasp"], "place": r["place"], "n_frames": n, "mp4": r["mp4"], "keyframes": row}
            rd.close()
        fig.patch.set_facecolor(fs.SURFACE)
        fig.suptitle(f"T30 {cond} — third-person front view keyframes (border: green success / amber grasp only / red no grasp)", fontsize=11, color=fs.INK)
        fig.tight_layout()
        p = new(f"t30-keyframes-{cond}.png"); fig.savefig(p, dpi=110); plt.close(fig)
        p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)

if "latency" in WHAT:
    fig, ax = plt.subplots(figsize=(12, 4.8)); side = {}
    data, labels = [], []
    for c in AN:
        for part in ("infer", "ag3s", "esdf", "trajopt", "total", "client"):
            v = []
            for name, r in AN[c]["runs"].items():
                for e in r["chunks"][1:]:
                    if part == "esdf":
                        x = (e.get("server_profile_ms") or {}).get("esdf")
                    elif part == "client":
                        x = e.get("client_policy_infer_ms")
                    else:
                        x = (e.get("server_timing_ms") or {}).get(part)
                    if x is not None:
                        v.append(x)
            data.append(v); labels.append(f"{c}\n{part}")
            side[f"{c}_{part}_ms_excl_first"] = {"n": len(v), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(max(v))} if v else None
    bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True, flierprops={"ms": 2})
    for j, b in enumerate(bp["boxes"]):
        b.set_facecolor(C[0 if labels[j].startswith("E3a") else 1] + "66")
    ax.set_ylabel("ms (first chunk of each run excluded)"); ax.tick_params(axis="x", labelsize=7)
    ax.set_title("T30 server time split: infer / AG3S (incl. ESDF) / ESDF / TO / total, and client round trip. E3a: 2 servers concurrent; E3b: alone (T30)", fontsize=9.5)
    fs.style_axes(fig, [ax]); fig.tight_layout()
    p = new("t30-latency.png"); fig.savefig(p, dpi=130); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)
