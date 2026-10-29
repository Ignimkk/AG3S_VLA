"""T44dA figure (b): cause aggregates. Inputs CLA/{approach_stats,close_geom,table_lift,q3_hold,q4_contacts,c2_armrows}.json + chunks/C2_*.jsonl.
-> AG3S/ag3s/docs/figures/t44/t44da-graph.png + .json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, glob, pathlib, collections
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle
C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CLA')); FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t44'))
ap = json.load(open(C / "approach_stats.json")); cg = json.load(open(C / "close_geom.json")); tl = json.load(open(C / "table_lift.json"))
q3 = json.load(open(C / "q3_hold.json")); q4 = json.load(open(C / "q4_contacts.json")); c2 = json.load(open(C / "c2_armrows.json"))
pr = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T44d.verify.json')))["numbers"]["per_run"]
ARMS = ("C0", "C1a", "C1b", "C2"); K = figstyle.CATEGORICAL
figstyle.use_korean()
fig, ax = plt.subplots(2, 3, figsize=(18, 10)); figstyle.style_axes(fig, ax); ax = ax.ravel(); side = {}
# (1) TO lift of the hand in the last 4 pre-close chunks
data = [[c["dz7_mm"] for n, v in ap["per_run"].items() if n.startswith(a + "_") for c in v["chunks"]] for a in ARMS]
bp = ax[0].boxplot(data, labels=ARMS, showfliers=False, patch_artist=True)
for p, col in zip(bp["boxes"], K): p.set_facecolor(col); p.set_alpha(0.45)
for i, d in enumerate(data): ax[0].scatter(np.full(len(d), i + 1) + np.random.default_rng(0).uniform(-0.15, 0.15, len(d)), d, s=6, color=K[i])
ax[0].axhline(0, color=figstyle.GRID_INK); ax[0].set_ylabel("dz of hand midpoint, refined - reference, row 7 (mm)")
ax[0].set_title("(1) TO lift of the active hand, last 4 chunks before the first close (n=72/arm)", fontsize=9)
side["p1_dz7_mm"] = {a: {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)), "frac_gt_3mm": float(np.mean(np.asarray(d) > 3)), "n": len(d)} for a, d in zip(ARMS, data)}
# (2) support zero level ON vs OFF
labs = ["under ball centre", "ring r = fit radius", "ring r = ball radius + 10 mm"]; keys = ["centre", "ringfit", "ringout"]
x = np.arange(3); w = 0.35
off = [tl["C1b"][k + "_off_median_mm"] for k in keys]; on = [tl["C1b"][k + "_on_median_mm"] for k in keys]
ax[1].bar(x - w / 2, off, w, color=K[0], label="coarse OFF (C1a formula)"); ax[1].bar(x + w / 2, on, w, color=K[1], label="coarse ON (C1b served)")
for xi, (a_, b_) in enumerate(zip(off, on)): ax[1].text(xi - w / 2, a_ + 0.2, f"{a_:.1f}", ha="center", fontsize=8); ax[1].text(xi + w / 2, b_ + 0.2, f"{b_:.1f}", ha="center", fontsize=8)
ax[1].set_xticks(x); ax[1].set_xticklabels(labs, fontsize=8); ax[1].set_ylabel("target-free zero level above support plane (mm), median")
ax[1].set_title("(2) where the target-free field says 'surface' near the target, C1b approach chunks (n=72)", fontsize=9); ax[1].legend(fontsize=8)
side["p2_zero_level_mm"] = {k: {"off": o, "on": n_} for k, o, n_ in zip(keys, off, on)}
# (3) grasp height at close + 8 rows vs first grasp ok
for i, a in enumerate(ARMS[:3]):
    pts = [(v["at_c8"]["dz"], bool(v["first_grasp_ok"])) for n, v in cg.items() if n.startswith(a + "_") and v.get("t_c") is not None]
    jit = np.random.default_rng(1).uniform(-0.12, 0.12, len(pts))
    for (dz, ok), j in zip(pts, jit):
        ax[2].scatter(i + j, dz, s=28, color=K[2] if ok else K[7], marker="o" if ok else "x")
ax[2].set_xticks(range(3)); ax[2].set_xticklabels(ARMS[:3]); ax[2].set_ylabel("finger midpoint - target origin, z, 8 rows after first close (mm)")
ax[2].scatter([], [], color=K[2], marker="o", label="first grasp ok"); ax[2].scatter([], [], color=K[7], marker="x", label="not ok"); ax[2].legend(fontsize=8)
ax[2].set_title("(3) grasp height at close (MuJoCo replay); C2 omitted (median 149 mm, never near)", fontsize=9)
side["p3_dz_c8_mm"] = {a: {"ok": [v["at_c8"]["dz"] for n, v in cg.items() if n.startswith(a + "_") and v.get("t_c") is not None and v["first_grasp_ok"]],
                           "not_ok": [v["at_c8"]["dz"] for n, v in cg.items() if n.startswith(a + "_") and v.get("t_c") is not None and not v["first_grasp_ok"]]} for a in ARMS}
# (4) HOLD chunks by cause
cause_map = {"uncertified_after_place": "uncertified after place", "authorized_target_free_close": "authorized finger, target-free, at close",
             "authorized_target_free_pre_close": "authorized finger, target-free, other", "authorized_target_free_post_close": "authorized finger, target-free, other",
             "held_object_rows": "held object rows", "strict_unauthorized_link": "unauthorized link (strict / arm link)"}
def c2_causes():
    cnt = collections.Counter(); runs = collections.defaultdict(set)
    for f in glob.glob(str(C / "chunks" / "C2_*.jsonl")):
        n = pathlib.Path(f).stem; sc = (cg.get(n) or {}).get("seq_c")
        for r in map(json.loads, open(f)):
            if not r["hold"]: continue
            cw = r.get("collision_worst") or {}
            k = ("authorized_target_free_close" if sc is not None and sc <= r["seq"] <= sc + 2 else "authorized_target_free_pre_close") if cw.get("why") == "answer_excludes_target" \
                else ("strict_unauthorized_link" if cw.get("why") == "unauthorized_link" else "other")
            cnt[k] += 1; runs[k].add(n)
    return cnt, runs
cats = ["uncertified after place", "authorized finger, target-free, at close", "authorized finger, target-free, other", "held object rows", "unauthorized link (strict / arm link)", "other"]
tab = {}
for a in ARMS:
    if a == "C2":
        cnt, runs = c2_causes(); src = {k: (cnt[k], len(runs[k])) for k in cnt}
    else:
        src = {k: (v, len(q3[a]["hold_runs_by_cause"][k])) for k, v in q3[a]["hold_chunks_by_cause"].items()}
    t = collections.Counter(); tr = collections.Counter()
    for k, (v, nr) in src.items():
        cc = cause_map.get(k, "other"); t[cc] += v; tr[cc] += nr
    tab[a] = {"chunks": dict(t), "runs_by_cause": dict(tr)}
bottom = np.zeros(4)
for j, cc in enumerate(cats):
    vals = np.array([tab[a]["chunks"].get(cc, 0) for a in ARMS])
    ax[3].bar(ARMS, vals, bottom=bottom, color=K[j], label=cc); bottom += vals
for i, a in enumerate(ARMS): ax[3].text(i, bottom[i] + 2, f"{int(bottom[i])}", ha="center", fontsize=8)
ax[3].set_ylabel("HOLD chunks"); ax[3].legend(fontsize=7); ax[3].set_title("(4) HOLD chunks by cause (worst collision row / verdict)", fontsize=9)
side["p4_hold_by_cause"] = tab
# (5) C2 arm-link rows
W = c2["worst_arm_rows"]
ax[4].hist([w["d_main_minus_r_mm"] for w in W], bins=20, color=K[0], alpha=0.6, label="field distance - r (arm_5 sphere)")
ax[4].hist([w["gt_gap_target_mm"] for w in W], bins=20, color=K[1], alpha=0.6, label="GT hull gap to target")
ax[4].hist([w["clearance_mm"] for w in W], bins=20, color=K[7], alpha=0.6, label="TO clearance (after 50 mm target margin)")
ax[4].axvline(50, color=figstyle.INK_2, ls="--", lw=1); ax[4].text(51, ax[4].get_ylim()[1] * 0.9, "50 mm", fontsize=8)
ax[4].set_xlabel("mm"); ax[4].set_ylabel("C2 chunks whose worst TO row is an arm link")
ax[4].set_title(f"(5) C2: worst row = link_*_arm_5 in {len(W)} chunks / {len({w['name'] for w in W})} runs", fontsize=9); ax[4].legend(fontsize=7)
side["p5_c2_arm_rows"] = {"n_chunks": len(W), "n_runs": len({w["name"] for w in W}), "links": dict(collections.Counter(w["link"] for w in W)),
                          "median_clearance_mm": float(np.median([w["clearance_mm"] for w in W])), "median_d_main_minus_r_mm": float(np.median([w["d_main_minus_r_mm"] for w in W])),
                          "median_gt_gap_target_mm": float(np.median([w["gt_gap_target_mm"] for w in W])),
                          "implied_margin_mm_median": float(np.median([w["d_main_minus_r_mm"] - w["clearance_mm"] for w in W]))}
# (6) hand-fruit_other contact events
ev = [e for e in q4 if pr[e["name"]]["release_t"] is None or e["t_first"] <= pr[e["name"]]["release_t"]]
for i, a in enumerate(("C0", "C1a", "C1b")):
    E = [e for e in ev if e["arm"] == a]
    for e in E:
        palm = any(b.startswith("EE_BODY") for b in e["hand_bodies_mjcf"])
        ax[5].scatter(e["frac_distractor_verts_f_tf_lt_7p5mm"], e["to_pred_min_tf_clearance_mm"], s=20 + 2 * e["max_force_N"] ** 0.5 * 5,
                      color=K[i], marker="s" if palm else "o", alpha=0.75)
    ax[5].scatter([], [], color=K[i], label=a)
ax[5].scatter([], [], color=figstyle.INK_2, marker="s", label="palm (EE_BODY)"); ax[5].scatter([], [], color=figstyle.INK_2, marker="o", label="finger")
ax[5].set_xlabel("fraction of distractor mesh vertices with served target-free distance < 7.5 mm"); ax[5].set_ylabel("TO-predicted min target-free clearance, contact rows (mm)")
ax[5].set_title("(6) hand-fruit_other contact events before release", fontsize=9); ax[5].legend(fontsize=7)
side["p6_events"] = [{k: e[k] for k in ("arm", "name", "t_first", "t_last", "max_force_N", "hand_bodies_mjcf", "fruit", "grasp_state", "frac_distractor_verts_f_tf_lt_7p5mm",
                                        "frac_distractor_verts_f_main_lt_7p5mm", "to_pred_min_tf_clearance_mm", "plan_gt_gap_to_distractor_mm", "actual_gt_gap_to_distractor_mm", "servo_err_hand_mm", "n_ambiguous", "n_kept_nontarget")} for e in ev]
fig.suptitle("T44dA - causes behind the CL-run differences (C0 / C1a / C1b / C2, 18 runs per arm)", fontsize=11, color=figstyle.INK)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(FIG / "t44da-graph.png", dpi=140)
side["sources"] = ["outputs/verify/T44/CLA/" + s for s in ("approach_stats.json", "close_geom.json", "table_lift.json", "q3_hold.json", "q4_contacts.json", "c2_armrows.json", "chunks/C2_*.jsonl")]
side["script"] = "outputs/verify/T44/CLA/p/fig_graph.py"
json.dump(side, open(FIG / "t44da-graph.json", "w"), indent=1)
print(json.dumps(side["p4_hold_by_cause"]), json.dumps(side["p1_dz7_mm"]))
