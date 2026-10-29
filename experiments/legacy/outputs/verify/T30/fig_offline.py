"""T30 offline figures (#1 F1 counterfactual, #2 F3 replay, #3 F4 replay). usage: fig_offline.py [f1] [f3] [f4]
  t30-off1-f1-counterfactual.png : violated chunks per condition x trajectory (reference / refined) x rule (old / new), stacked by the
                                   worst row's GT nearest object; hatched part = worst row is target_free + manipulated-object margin
  t30-off2-f3-coverage.png       : per group (pre-grasp non-frozen / frozen / after switch) x rule: GT apple whole in ball / centre only /
                                   outside / no ball; + occupancy (voxels near apple kept outside the ball) for uncovered chunks
  t30-off3-f4-latch.png          : T17 measured opening per request with old (T22) / new attach times; empty-hand sweep table old vs new
Each png has a .json sidecar with the plotted numbers."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
fs.use_korean(); C = fs.CATEGORICAL
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30'))
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t30'))
WHAT = sys.argv[1:] or ["f1", "f3", "f4"]
OBJ = ["apple", "banana", "orange", "pear", "crate", "table"]
OBJ_COL = {"apple": C[1], "crate": C[0], "table": fs.GRID_INK, "orange": C[3], "pear": C[5], "banana": C[4]}


def new(name):
    p = F / name
    assert not p.exists(), f"{p} exists"
    return p


if "f1" in WHAT:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    side = {}
    for ax, cond in zip(axes, ("E3a", "E3b")):
        R = json.load(open(V / f"off1/f1_{cond}.json"))
        ch = [c for c in R["chunks"] if "skipped" not in c]
        xl, x = [], 0
        for tn in ("reference", "refined"):
            for rn in ("old", "new"):
                k = f"{tn}_{rn}"
                Vv = [c[k] for c in ch if c[k]["violated"]]
                cnt = collections.Counter(v.get("gt_nearest") for v in Vv)
                tft = collections.Counter(v.get("gt_nearest") for v in Vv if v.get("tier") == "target_free" and v.get("target_margin_row"))
                b = 0
                for o in OBJ:
                    n = cnt.get(o, 0)
                    if n:
                        ax.bar(x, n, bottom=b, color=OBJ_COL[o], width=0.7)
                        if tft.get(o):
                            ax.bar(x, tft[o], bottom=b, color="none", edgecolor=fs.INK, hatch="///", width=0.7, lw=0)
                        b += n
                ax.text(x, b + 1, f"{len(Vv)}\n({sum(tft.values())} tf+target)", ha="center", va="bottom", fontsize=8, color=fs.INK)
                side[f"{cond}_{k}"] = {"n_violated": len(Vv), "by_gt_nearest": dict(cnt), "target_free_target_by_gt": dict(tft)}
                xl.append(f"{tn}\n{'pre-T30 rule' if rn == 'old' else 'T30 rule'}"); x += 1
        ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, fontsize=8)
        ax.set_ylabel("chunks with worst clearance < 0")
        ax.set_title(f"{cond}: {len(ch)} chunks evaluated ({len(R['chunks']) - len(ch)} attached skipped)", fontsize=10, color=fs.INK)
        ax.set_ylim(0, max(10, ax.get_ylim()[1] * 1.15))
    hs = [Patch(color=OBJ_COL[o], label=f"worst row nearest GT: {o}") for o in OBJ] + [Patch(facecolor="none", edgecolor=fs.INK, hatch="///", label="worst row: target_free tier + object margin")]
    fig.legend(handles=hs, ncol=4, fontsize=7.5, frameon=False, loc="lower center")
    fig.suptitle("T30 #1 F1 counterfactual on pre-T30 E3 records: same inputs, pre-T30 rule vs T30 rule (T24 fixed-input tool)", fontsize=11, color=fs.INK)
    fs.style_axes(fig, list(axes)); fig.tight_layout(rect=(0, 0.1, 1, 1))
    p = new("t30-off1-f1-counterfactual.png"); fig.savefig(p, dpi=130); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)

if "f3" in WHAT:
    S = json.load(open(V / "off2/f3_summary.json"))["pooled"]
    groups = [("E3", "pre_grasp_nonfrozen"), ("E3", "frozen"), ("E3", "after_switch"), ("t14", "pre_grasp_nonfrozen"), ("t17", "pre_grasp_nonfrozen")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]; x = 0; xl = []; side = {}
    for who, gn in groups:
        for rule in ("old", "new"):
            d = (S.get(who, {}).get(gn) or {}).get(rule)
            if d is None:
                continue
            whole = d["n_whole_apple_inside_ball"]; centre = d["n_apple_centre_inside_ball"] - whole
            out = d["n_ball"] - d["n_apple_centre_inside_ball"]; nob = d["n_no_ball"]
            b = 0
            for n, colr in ((whole, C[2]), (centre, C[3]), (out, C[7]), (nob, fs.INK_2)):
                ax.bar(x, n, bottom=b, color=colr, width=0.7); b += n
            ax.text(x, b + 3, f"{d['n']}\nrej {d['n_association_rejected']}\nsub {d['n_subset_kept_anchor']}", ha="center", fontsize=7, color=fs.INK)
            xl.append(f"{who}\n{gn.replace('_nonfrozen', '').replace('_', ' ')}\n{'T26 rule' if rule == 'old' else 'T30'}"); x += 1
            side[f"{who}_{gn}_{rule}"] = {k: d[k] for k in ("n", "n_whole_apple_inside_ball", "n_apple_centre_inside_ball", "n_ball", "n_no_ball",
                                                           "n_association_rejected", "n_subset_kept_anchor", "n_manip_not_apple")}
    ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, fontsize=7); ax.set_ylabel("chunks")
    ax.legend(handles=[Patch(color=C[2], label="whole GT apple (+34 mm) in ball"), Patch(color=C[3], label="apple centre only"),
                       Patch(color=C[7], label="apple centre outside ball"), Patch(color=fs.INK_2, label="no ball (gate)")], fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("GT apple vs replayed exclusion ball (rej = association_rejected, sub = subset_kept_anchor)", fontsize=9.5, color=fs.INK)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    R = json.load(open(V / "off2/f3_replay.json"))
    OC = json.load(open(V / "off2/f3_occupancy.json"))
    G = [r for L in OC["runs"].values() for r in L if r["phase"] == "pre_grasp" and not r["frozen"]]
    vals = [r["fine_out_new_ball"] for r in G if not r["new_whole_inside"]]
    vals_old = [r["fine_out_old_ball"] for r in G if not r["old_whole_inside"]]
    vals_tf = [r["tf_recorded"] for r in G if not r["old_whole_inside"] and "tf_recorded" in r]
    ax = axes[1]
    bins = np.linspace(0, 2200, 23)
    ax.hist(vals_tf, bins=bins, histtype="step", color=C[1], lw=1.6, label=f"recorded target-free layer, T26 ball uncovered (n={len(vals_tf)}, >0: {sum(1 for v in vals_tf if v > 0)})")
    ax.hist(vals_old, bins=bins, histtype="step", color=C[0], lw=1.2, ls="--", label=f"fine minus T26 ball (n={len(vals_old)}, >0: {sum(1 for v in vals_old if v > 0)})")
    ax.hist(vals, bins=bins, histtype="step", color=C[2], lw=1.6, label=f"fine minus T30 ball (n={len(vals)}, >0: {sum(1 for v in vals if v > 0)})")
    ax.set_xlabel("occupied 5 mm voxels: within GT apple + 15 mm, above table, outside ball"); ax.set_ylabel("chunks")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("E3 pre-grasp non-frozen chunks whose ball does not hold the whole apple", fontsize=8.5, color=fs.INK)
    side["occupancy_uncovered"] = {"fine_minus_T30_ball": vals, "fine_minus_T26_ball": vals_old, "recorded_target_free_T26_ball": vals_tf}
    fig.suptitle("T30 #2 F3 replay on recorded grounding inputs (E3a+E3b 12 runs, T14, T17): exclusion ball coverage of the GT apple", fontsize=11, color=fs.INK)
    fs.style_axes(fig, list(axes)); fig.tight_layout()
    p = new("t30-off2-f3-coverage.png"); fig.savefig(p, dpi=130); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)

if "f4" in WHAT:
    R = json.load(open(V / "off3/f4_replay.json"))
    old17 = json.load(open(_ag3s_workspace_path('outputs/verify/T22/replay_t17_apple.json')))
    oldsw = json.load(open(_ag3s_workspace_path('outputs/verify/T22/empty_sweep_t22.json')))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]; side = {}
    for i, key in enumerate(("t17_apple", "t16_apple", "t14_apple")):
        rows = R["live"][key]["rows"]
        t = [r["t_step"] for r in rows]; op = [r["measured"] for r in rows]
        ax.plot(t, op, color=C[i], lw=1.5, label=f"{key.split('_')[0]} measured opening")
        for a in R["live"][key]["summary"]["attach_t"]:
            ax.axvline(a, color=C[i], lw=2, ls="-")
        for a in R["live"][key]["summary"]["legacy_attach_t"]:
            ax.axvline(a, color=C[i], lw=1, ls=":")
        side[key] = {"t": t, "measured": op, "attach_t_T30": R["live"][key]["summary"]["attach_t"], "detach_t_T30": R["live"][key]["summary"]["detach_t"]}
    side["t17_apple"]["attach_t_T22_evidence_rule"] = old17["summary"]["attach_t"]
    ax.set_xlim(60, 300); ax.set_xlabel("request t_step"); ax.set_ylabel("measured left opening (1 = open)")
    ax.set_title(f"T14/T16/T17 replay: solid = T30 attach ({', '.join(k.split('_')[0] + ' ' + str(R['live'][k]['summary']['attach_t']) for k in ('t17_apple', 't16_apple', 't14_apple'))}), dotted = legacy gripper attach; T22 rule T17 attach {old17['summary']['attach_t']}", fontsize=8, color=fs.INK)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]; ax.axis("off")
    cell, colr = [], []
    for tgt in (0.0, 0.3, 0.57):
        line, lc = [], []
        for s in range(8):
            o = next(r for r in oldsw if r["close_target"] == tgt and r["start_step_in_chunk"] == s)
            n = next(r for r in R["empty_sweep"] if r["close_target"] == tgt and r["start_step_in_chunk"] == s)
            line.append(f"{'A' if o['attach_req'] else '-'} / {'A' if n['attach_req'] else '-'}")
            lc.append((C[7] + "55") if n["attach_req"] else ((C[3] + "55") if o["attach_req"] else fs.SURFACE))
        cell.append(line); colr.append(lc)
    tb = ax.table(cellText=cell, cellColours=colr, rowLabels=[f"close to {t}" for t in (0.0, 0.3, 0.57)], colLabels=[f"start {s}" for s in range(8)], loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.8)
    ax.set_title(f"empty-hand MuJoCo sweep: attach T22 rule / T30 rule (A = attach)\nT22 {sum(1 for r in oldsw if r['attach_req'])}/24, T30 {sum(1 for r in R['empty_sweep'] if r['attach_req'])}/24", fontsize=9, color=fs.INK)
    side["empty_sweep"] = {"T22": {f"{r['close_target']}_{r['start_step_in_chunk']}": r["attach_req"] for r in oldsw},
                           "T30": {f"{r['close_target']}_{r['start_step_in_chunk']}": r["attach_req"] for r in R["empty_sweep"]}}
    side["e3"] = {k: {"attach_t_T30": v["attach_t"], "recorded_attach_t": v["recorded_attach_t"]} for k, v in R["e3"].items()}
    fig.suptitle(f"T30 #3 F4 replay: settled = measured opening change < settle_epsilon {R['latch_config']['settle_epsilon']}", fontsize=11, color=fs.INK)
    fs.style_axes(fig, [axes[0]]); fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
    p = new("t30-off3-f4-latch.png"); fig.savefig(p, dpi=130); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)
