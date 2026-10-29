"""T37 figures (derived from T34/p/fig_e3.py). Output AG3S/ag3s/docs/figures/t37/t37-*. Every png has a .json sidecar.
  t37-success-grid.png        : condition x episode grid — E0-E2 (T28 P2R), E3a/E3b at T30, T33, T34, T37
  t37-contact-height.png      : contact midpoint z - apple centre z at settle (T35-diag settle.py metric) and min measured opening in the
                                24 control rows after settle; all runs so far (T35-diag a/settle.json 42 runs + T37 geom/a/settle.json 12 runs).
                                marker: o = first grasp held (no contact loss in the hold window), ^ = contact lost in the window; colour success/failure
  t37-tipdev.png              : fingertip deviation refined - reference (Meter.compare, mm, planned horizon max) per chunk in latch state
                                closing / held vs other states; T34 records vs T37 records (p/<COND>_tipdev[_T34].json)
  t37-bands-<COND>.png        : per chunk bands: TO returned, continuity gate (off_by set), manipulated, worst-row object, HOLD/would_hold kind,
                                latch state (A attach / D detach / R attach_revoked); twin axis fingertip deviation (mm) and apple dz
  t37-keyframes-<COND>.png    : mp4 keyframes (start, close cmd, grasp / closest, max lift, last frame)
  t37-run-table.png           : per-run numbers table T37 E3a / E3b
usage: .venv-openpi-live/bin/python fig_t37.py [grid contact tipdev bands keyframes table]"""
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
V30 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30/p'))
V33 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T33/p'))
V34 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T34/p'))
V3 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T37/p'))
G35 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T35diag/a/settle.json'))
G37 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T37/geom/a/settle.json'))
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t37')); F.mkdir(parents=True, exist_ok=True)
C = fs.CATEGORICAL
EPS = (1807, 1800, 1808)
WHAT = sys.argv[1:] or ["grid", "contact", "tipdev", "bands", "keyframes", "table"]


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
AN, SU, TD, TD34, X = {}, {}, {}, {}, {}
for cond in ("E3a", "E3b"):
    SRC[(cond, "t30")] = {k: v["summary"] for k, v in json.loads((V30 / f"{cond}_analysis.json").read_text())["runs"].items()}
    SRC[(cond, "t33")] = {k: v["summary"] for k, v in json.loads((V33 / f"{cond}_analysis.json").read_text())["runs"].items()}
    SRC[(cond, "t34")] = {k: v["summary"] for k, v in json.loads((V34 / f"{cond}_analysis.json").read_text())["runs"].items()}
    p = V3 / f"{cond}_analysis.json"
    if p.exists():
        AN[cond] = json.loads(p.read_text())
        SRC[(cond, "t37")] = {k: v["summary"] for k, v in AN[cond]["runs"].items()}
        SU[cond] = json.loads((V3 / f"{cond}_summary.json").read_text())
        if (V3 / f"{cond}_tipdev.json").exists():
            TD[cond] = json.loads((V3 / f"{cond}_tipdev.json").read_text())["runs"]
        if (V3 / f"{cond}_t37_extra.json").exists():
            X[cond] = json.loads((V3 / f"{cond}_t37_extra.json").read_text())["runs"]
    if (V3 / f"{cond}_tipdev_T34.json").exists():
        TD34[cond] = json.loads((V3 / f"{cond}_tipdev_T34.json").read_text())["runs"]
DUPS_T30 = {"E3a_ep1800_r2": "r1", "E3a_ep1808_r2": "r1", "E3a_ep1807_r2": "r1 RNG*", "E3a_ep1800_r3": "E3b ep1800 r1"}
DUP37 = json.loads((V3 / "dup_check.json").read_text()) if (V3 / "dup_check.json").exists() else {}
DUPS_T37 = {}
for k, v in DUP37.items():
    if v.get("identical"):
        tag, cond, ep, a, _, b = k.split("_")
        DUPS_T37[f"{cond}_{ep}_{b}"] = a

if "grid" in WHAT:
    order = [(c, k) for c in ("E0", "E1", "E2") for k in ("rec", "old") if (c, k) in SRC] + \
            [(c, k) for k in ("t30", "t33", "t34", "t37") for c in ("E3a", "E3b") if (c, k) in SRC]
    cell, colors, side = [], [], {"cells": {}, "summary": {}, "sources": {"E0-E2": str(V2), "t30": str(V30), "t33": str(V33), "t34": str(V34), "t37": str(V3)}}
    for c, k in order:
        runs = SRC[(c, k)]; line = []; lc = []
        dd = DUPS_T37 if k == "t37" else (DUPS_T30 if k == "t30" else {})
        for ep in EPS:
            for rk in (1, 2, 3):
                r = runs.get(f"{c}_ep{ep}_r{rk}")
                if r is None:
                    line.append("-"); lc.append(fs.SURFACE); continue
                s = status(r)
                txt = {"success": f"S  g{r['grasp_t']} p{r['place_t']}", "grasp_only": f"G  g{r['grasp_t']}  no place",
                       "no_grasp": f"F  dz {r['apple_dz_max_mm']:.0f}mm"}[s]
                dk = f"{c}_ep{ep}_r{rk}"
                if dk in dd:
                    txt += f"  (={dd[dk]})"
                line.append(txt); lc.append(COL[s] + "55")
                side["cells"][f"{c}_{k}_ep{ep}_r{rk}"] = {"status": s, "grasp_t": r["grasp_t"], "place_t": r["place_t"],
                                                          "apple_dz_max_mm": r["apple_dz_max_mm"], "run_dir": r["run_dir"]}
        ns = sum(1 for r in runs.values() if r["success"]); ng = sum(1 for r in runs.values() if r["grasp"])
        line.append(f"{ns}/{len(runs)}  (grasp {ng})"); lc.append(fs.SURFACE)
        cell.append(line); colors.append(lc)
        side["summary"][f"{c}_{k}"] = {"n_success": ns, "n_grasp": ng, "n_runs": len(runs)}
    lab = {"rec": "recorded", "old": "unrecorded 09-28", "t30": "at T30", "t33": "at T33", "t34": "at T34", "t37": "at T37"}
    note = {"E0": "VLA only", "E1": "TO tracking only", "E2": "+objective/limits", "E3a": "+distance field, gate off", "E3b": "+distance field, gate reasons (alone)"}
    fig, ax = plt.subplots(figsize=(22, 0.62 * len(order) + 1.6)); ax.axis("off")
    tb = ax.table(cellText=cell, cellColours=colors, rowLabels=[f"{c} {lab[k]} — {note[c]}" for c, k in order],
                  colLabels=[f"ep{ep} r{rk}" for ep in EPS for rk in (1, 2, 3)] + ["success / runs"], loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.9)
    ax.set_title("Success grid: E0-E2 (T28 P2R), E3a/E3b at T30, T33, T34 (J1+J2), T37 (K1 continuity off in closing/held)   "
                 "S success: g grasp t, p place t | G grasp, no place | F no grasp: max apple dz | (=x) identical sample", fontsize=10, color=fs.INK)
    fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
    p = new("t37-success-grid.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)

if "contact" in WHAT:
    S35 = json.loads(G35.read_text())["runs"]; S37 = json.loads(G37.read_text())["runs"]
    pts = []
    for k, v in S35.items():
        g = k.split("/")[0]
        g = {"E3a_rec": "T34 E3a", "E3b_rec": "T34 E3b"}.get(g, g)
        pts.append((g, k, v))
    for k, v in S37.items():
        g = {"E3a_rec": "T37 E3a", "E3b_rec": "T37 E3b"}[k.split("/")[0]]
        pts.append((g, "T37/" + k, v))
    GROUPS = ["E0", "E0_rec", "E1", "E1_rec", "E2_rec", "T34 E3a", "T34 E3b", "T37 E3a", "T37 E3b"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8)); fs.style_axes(fig, ax)
    side = {"sources": [str(G35), str(G37)], "metric": "contact_mid_rel_apple_mm[2] at t_settle (T35-diag settle.py); hold_min_open = min measured opening in 24 rows after settle; first_grasp_held = contact_lost_t is None",
            "points": [], "group_stats": {}}
    rng = np.random.default_rng(1)
    for g, k, v in pts:
        gi = GROUPS.index(g); x = gi + rng.uniform(-0.18, 0.18)
        ok = bool(v["success"]); held = v["contact_lost_t"] is None
        kw = dict(color=C[0] if ok else C[1], marker="o" if held else "^", s=46, edgecolor=fs.SURFACE, linewidth=1, zorder=3)
        z = v["contact_mid_rel_apple_mm"][2]
        ax[0].scatter(x, z, **kw); ax[1].scatter(x, v["hold_min_open"], **kw)
        if g.startswith("T37"):
            ax[0].annotate(k.split("/")[-1].split("_", 1)[1], (x, z), fontsize=6.3, color=fs.INK_2, xytext=(4, 2), textcoords="offset points")
        side["points"].append({"group": g, "run": k, "success": ok, "first_grasp_held": held, "contact_mid_z_mm": z, "hold_min_open": v["hold_min_open"],
                               "t_settle": v["t_settle"], "contact_lost_t": v["contact_lost_t"]})
    for g in GROUPS:
        zs = [p["contact_mid_z_mm"] for p in side["points"] if p["group"] == g]
        if zs:
            side["group_stats"][g] = {"n": len(zs), "min": min(zs), "median": float(np.median(zs)), "max": max(zs),
                                      "n_success": sum(p["success"] for p in side["points"] if p["group"] == g),
                                      "n_first_grasp_held": sum(p["first_grasp_held"] for p in side["points"] if p["group"] == g)}
    for a in ax:
        a.set_xticks(range(len(GROUPS))); a.set_xticklabels(GROUPS, fontsize=8, rotation=20)
        a.grid(axis="y", color=fs.GRID_INK, lw=0.6)
        a.axvspan(6.5, 8.5, color=fs.GRID_INK, alpha=0.25, lw=0)
    ax[0].axhline(0, color=fs.INK_2, lw=1)
    ax[0].set_ylabel("contact midpoint z - apple centre z\nat settle (mm)")
    ax[0].set_title("contact height at the end of the first close cycle", fontsize=9, color=fs.INK)
    ax[1].axhline(0.70, color=fs.INK_2, lw=1, ls=":"); ax[1].text(-0.4, 0.705, "apple width ~0.70", fontsize=7, color=fs.INK_2)
    ax[1].set_ylabel("min measured opening,\n24 rows after settle (1 = open)")
    ax[1].set_title("finger opening after settle", fontsize=9, color=fs.INK)
    hs = [ml.Line2D([], [], color=C[0], marker="o", ls="", label="success, first grasp held"), ml.Line2D([], [], color=C[0], marker="^", ls="", label="success, contact lost in window"),
          ml.Line2D([], [], color=C[1], marker="o", ls="", label="failure, first grasp held"), ml.Line2D([], [], color=C[1], marker="^", ls="", label="failure, contact lost in window")]
    ax[0].legend(handles=hs, fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(f"Contact height at settle, all {len(pts)} closed-loop runs on episodes 1800/1807/1808 (T35-diag metric; shaded = T37)", fontsize=10, color=fs.INK)
    fig.tight_layout()
    p = new("t37-contact-height.png"); fig.savefig(p, dpi=140); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(p)

if "tipdev" in WHAT:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True); fs.style_axes(fig, axes)
    side = {"sources": {"T34": [str(V3 / f"{c}_tipdev_T34.json") for c in ("E3a", "E3b")], "T37": [str(V3 / f"{c}_tipdev.json") for c in ("E3a", "E3b")]},
            "metric": "Meter.compare(refined, reference, q_now).fingertip_disp_mm_max over the planned horizon (8 steps)", "groups": {}}
    cols = [("T34", "E3a"), ("T37", "E3a"), ("T34", "E3b"), ("T37", "E3b")]
    rng = np.random.default_rng(2)
    for ai, stset in enumerate((("closing", "held"), None)):
        ax = axes[ai]
        for gi, (tag, c) in enumerate(cols):
            src = (TD34 if tag == "T34" else TD).get(c) or {}
            vals = []
            for run, r in src.items():
                for ch in r["chunks"]:
                    ins = ch["state"] in ("closing", "held")
                    if (stset is not None and ins) or (stset is None and not ins):
                        vals.append(ch["fingertip_disp_mm_max"])
                        x = gi + rng.uniform(-0.2, 0.2)
                        ax.plot(x, ch["fingertip_disp_mm_max"], "o", ms=3, alpha=0.6,
                                color={"closing": C[3], "held": C[2]}.get(ch["state"], C[0]))
            if vals:
                ax.hlines(np.median(vals), gi - 0.3, gi + 0.3, color=fs.INK, lw=1.6)
                side["groups"][f"{tag}_{c}_{'closing_held' if stset else 'other'}"] = {"n": len(vals), "median": float(np.median(vals)), "p90": float(np.percentile(vals, 90)),
                                                                                         "max": float(max(vals)), "n_gt_1mm": int(sum(v > 1 for v in vals))}
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{t} {c}" for t, c in cols], fontsize=8)
        ax.grid(axis="y", color=fs.GRID_INK, lw=0.6)
        ax.set_title("chunks with latch closing / held" if stset else "all other chunks (searching / latched / placed / ...)", fontsize=9, color=fs.INK)
    axes[0].set_ylabel("fingertip |refined - reference| max (mm)")
    hs = [ml.Line2D([], [], color=C[3], marker="o", ls="", label="closing"), ml.Line2D([], [], color=C[2], marker="o", ls="", label="held"),
          ml.Line2D([], [], color=C[0], marker="o", ls="", label="other state"), ml.Line2D([], [], color=fs.INK, lw=1.6, label="median")]
    axes[1].legend(handles=hs, fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Fingertip deviation refined - reference per chunk (recorded chunks, Meter.compare), T34 records vs T37 records", fontsize=10, color=fs.INK)
    fig.tight_layout()
    p = new("t37-tipdev.png"); fig.savefig(p, dpi=140); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(p)

OBJ_COL = {"apple": C[1], "crate": C[0], "table": fs.GRID_INK, "orange": C[3], "pear": C[5], "banana": C[4], "none": fs.INK_2,
           "shelf": C[6], "office": C[6], "world": C[6]}
KIND_COL = {"collision": C[7], "uncertified": C[3], "unverified": C[6], "comms": C[4], "budget_only": C[2], "none": fs.SURFACE}
LATCH_COL = {"searching": fs.GRID_INK, "latched": C[0], "closing": C[3], "held": C[2], "placed": C[5], "released": C[6], "None": fs.SURFACE}

if "bands" in WHAT:
    for c in AN:
        runs = AN[c]["runs"]
        names = [f"{c}_ep{ep}_r{k}" for ep in EPS for k in (1, 2, 3) if f"{c}_ep{ep}_r{k}" in runs]
        fig, axes = plt.subplots(len(names), 1, figsize=(15, 2.8 * len(names)), sharex=True, squeeze=False)
        side = {"condition": c, "rows": ["latch state", "E3a would_hold kind / E3b HOLD kind", "worst row nearest object (x inside apple)",
                                         "manipulated (nearest centre)", "continuity gate (off_by set; colour = latch state)", "TO returned"], "runs": {}}
        for i, name in enumerate(names):
            ax = axes[i, 0]; ch = runs[name]["chunks"]; s = runs[name]["summary"]
            tdm = {x["seq"]: x for x in (TD.get(c, {}).get(name, {}).get("chunks") or [])}
            rec = {"t": [], "latch": [], "gate_kind": [], "worst": [], "manipulated": [], "continuity_off_by": [], "to_returned": [],
                   "fingertip_dev_mm": [], "apple_dz_mm": [], "latch_marks": []}
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
                tm = e.get("to_metrics") or {}
                rs = tm.get("returned"); off = tm.get("continuity_off_by")
                assc = {"initial": fs.GRID_INK, "accepted": C[0], "projection": C[3], "best_unaccepted": C[7]}.get(rs, fs.SURFACE)
                gatec = LATCH_COL.get(la, C[6]) if off else fs.SURFACE
                for y, colr in ((5, assc), (4, gatec), (3, OBJ_COL.get(man, C[6])), (2, OBJ_COL.get(wo, C[6])), (1, KIND_COL.get(k0.split(":")[0], C[6])), (0, LATCH_COL.get(la, C[6]))):
                    ax.add_patch(plt.Rectangle((t, y), 8, 0.85, color=colr, lw=0))
                gl = e.get("grasp_latch") or {}
                mk = "R" if (gl.get("detach") and "attach_revoked" in str(gl.get("note"))) else ("A" if gl.get("attach") else ("D" if gl.get("detach") else None))
                if mk:
                    ax.text(t + 4, 0.42, mk, ha="center", va="center", fontsize=7, color=fs.INK, fontweight="bold")
                    rec["latch_marks"].append([t, mk])
                if e["worst"].get("inside_apple"):
                    ax.plot(t + 4, 2.42, "x", color=fs.INK, ms=5, mew=1.2)
                td = tdm.get(e["seq"])
                for kk, vv in (("t", t), ("latch", la), ("gate_kind", ks), ("worst", wo), ("manipulated", man), ("continuity_off_by", off),
                               ("to_returned", rs), ("fingertip_dev_mm", None if td is None else td["fingertip_disp_mm_max"]), ("apple_dz_mm", e["apple_dz_mm"])):
                    rec[kk].append(vv)
            ax2 = ax.twinx()
            xs = [t + 4 for t, v in zip(rec["t"], rec["fingertip_dev_mm"]) if v is not None]
            ax2.plot(xs, [v for v in rec["fingertip_dev_mm"] if v is not None], color=fs.INK, lw=0.9)
            ax2.plot([t + 4 for t in rec["t"]], [x / 4 for x in rec["apple_dz_mm"]], color=C[2], lw=1.2, ls="--")
            ax2.set_ylim(-5, 75); ax2.set_ylabel("fingertip dev mm (black)\napple dz/4 mm (green)", fontsize=7)
            if s["t_first_close_cmd"] is not None:
                ax.axvline(s["t_first_close_cmd"], color=fs.INK, lw=0.8, ls=":")
            ax.set_ylim(-0.1, 6); ax.set_yticks([0.42, 1.42, 2.42, 3.42, 4.42, 5.42])
            ax.set_yticklabels(["latch", "HOLD/would", "worst obj", "manipulated", "continuity OFF", "TO returned"], fontsize=7)
            ax.set_xlim(0, 600)
            ngate = sum(1 for v in rec["continuity_off_by"] if v)
            ax.set_title(f"{name}: {status(s)}  grasp t {s['grasp_t']}  place t {s['place_t']}  first close cmd t {s['t_first_close_cmd']}  gate-off chunks {ngate}", fontsize=9, color=fs.INK)
            side["runs"][name] = rec
        axes[-1, 0].set_xlabel("control t_step (chunk start; 8 steps per chunk)")
        hs = [Patch(color=v, label=f"obj {k}") for k, v in OBJ_COL.items() if k in ("apple", "crate", "table", "orange", "pear", "banana", "none")]
        hs += [Patch(color=v, label=f"kind {k}") for k, v in KIND_COL.items() if k not in ("none", "budget_only")]
        hs += [Patch(color=v, label=f"latch {k}") for k, v in LATCH_COL.items() if k != "None"]
        hs += [Patch(color=fs.GRID_INK, label="TO initial"), Patch(color=C[0], label="TO accepted"), Patch(color=C[3], label="TO projection"), Patch(color=C[7], label="TO best_unaccepted")]
        fig.legend(handles=hs, ncol=9, fontsize=7, frameon=False, loc="lower center")
        fig.suptitle(f"T37 {c} per-chunk bands (" + ("would_hold kinds, gate off" if c == "E3a" else "HOLD kinds, gate reasons + fixed hold") +
                     "); continuity OFF row filled where to.metrics.continuity_off_by is set", fontsize=11)
        fs.style_axes(fig, axes.ravel()); fig.tight_layout(rect=(0, 0.05, 1, 1))
        p = new(f"t37-bands-{c}.png"); fig.savefig(p, dpi=115); plt.close(fig)
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
        fig.suptitle(f"T37 {cond} — third-person front view keyframes (border: green success / amber grasp only / red no grasp)", fontsize=11, color=fs.INK)
        fig.tight_layout()
        p = new(f"t37-keyframes-{cond}.png"); fig.savefig(p, dpi=110); plt.close(fig)
        p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)

if "table" in WHAT:
    S37 = json.loads(G37.read_text())["runs"]
    hdr = ["run", "result", "grasp t", "place t", "dz max\nmm", "gate-off\nchunks", "closing/held\nchunks", "tip dev c/h\nmax mm", "tip dev other\nmax mm",
           "contact z\nsettle mm", "first grasp\nheld", "close\ncycles", "HOLD / would_hold\nkinds", "attach t", "detach t", "revoke t", "comms HOLD\n(timeout+stale+error)"]
    rows, colors, side = [], [], {"columns": hdr, "rows": {}}
    for c in ("E3a", "E3b"):
        if c not in AN:
            continue
        for ep in EPS:
            for k in (1, 2):
                name = f"{c}_ep{ep}_r{k}"
                if name not in AN[c]["runs"]:
                    continue
                s = AN[c]["runs"][name]["summary"]; su = SU[c]["runs"][name]; td = TD.get(c, {}).get(name, {}); x = X.get(c, {}).get(name, {})
                st = S37.get(f"{c}_rec/{name}", {})
                kinds = su["hold_kind_counts"] if c == "E3b" else su["would_hold_kind_counts"]
                f1 = lambda v: "-" if v is None else f"{v:.1f}"
                row = [name, status(s), s["grasp_t"] or "-", s["place_t"] or "-", f"{s['apple_dz_max_mm']:.1f}", td.get("n_gate_on", "-"),
                       f"{td.get('n_closing', 0)}/{td.get('n_held', 0)}", f1(td.get("closing_held_tip_mm_max")), f1(td.get("other_tip_mm_max")),
                       f1((st.get("contact_mid_rel_apple_mm") or [None, None, None])[2]), "-" if not st else ("yes" if st["contact_lost_t"] is None else f"lost t{st['contact_lost_t']}"),
                       s.get("n_close_cycles_applied_lt_0p5"), ", ".join(f"{a}:{b}" for a, b in kinds.items()) or "0",
                       ",".join(map(str, su["attach_t"])) or "-", ",".join(map(str, su["detach_t"])) or "-", ",".join(map(str, x.get("revoke_t") or [])) or "-",
                       json.loads((V3 / f"frag_{c}.json").read_text())["runs"][name]["client_log"]["comms_timeout_stale_error"]]
                rows.append([str(v) for v in row]); colors.append([COL[status(s)] + "40"] + [fs.SURFACE] * (len(hdr) - 1))
                side["rows"][name] = dict(zip(hdr, row))
    fig, ax = plt.subplots(figsize=(24, 0.5 * len(rows) + 1.6)); ax.axis("off")
    tb = ax.table(cellText=rows, cellColours=colors, colLabels=hdr, loc="upper center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 2.1)
    ax.set_title("T37 per-run numbers (E3a gate off; E3b gate reasons, alone). gate-off chunks = to.metrics.continuity_off_by set; tip dev = Meter.compare fingertip |refined - reference| max; "
                 "contact z = T35-diag settle metric", fontsize=9.5, color=fs.INK)
    fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
    p = new("t37-run-table.png"); fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=str)); print(p)
