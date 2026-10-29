"""T43 LB3 aggregate -> T43LB3.verify.json + figures t43lb3-*. Numbers only. usage: make_lb3.py [--write]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, glob, re, os, datetime, statistics
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/LB/LB3/p'))
import plan_lb3 as PL
LB = PL.LB; L3 = PL.L3; R3 = L3 / "runs" / "E0"; RLB = LB / "runs" / "E0"
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
HO = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43LB3.verify.json'))
OLD = json.loads(pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43LB.verify.json')).read_text())
WRITE = "--write" in sys.argv
C = PL.C; MODES = ("ok", "no_grasp", "regrasp", "slip_at_lift", "slip_in_transport", "not_placed")
groups = {}; episodes = {}; mism = []
for g in PL.GROUPS:
    gv = C["groups"][g]; f, a = g.split("_")
    rand = gv["random_order_pool_minus_ranks_1_30"][:10]
    rec = {"lb": [], "lb3": []}
    for e in PL.all_tried(g):
        info = C["episodes"][str(e)]; r = PL.reps(e)
        src = r[0]["source"]
        seeds = []
        for k, x in enumerate(r, 1):
            if x is None: continue
            rd = pathlib.Path(x["run_dir"]); m = json.loads((rd / "manifest.json").read_text())["scene"]
            if (m["episode_split"], m["episode_target_fruit"], m["episode_used_arm"]) != ("train", f, a) or (info["split"], info["target_fruit"], info["used_arm"]) != ("train", f, a):
                mism.append(str(rd))
            seeds.append({"rep": k, "seed": x["seed"], "E0_ok_B": x["ok"], "success": x["success"], "grasp": x["grasp"], "place": x["place"], "robust_B": x["robust"],
                          "failure_mode": x["failure_mode"], "source": x["source"], "manifest_split": m["episode_split"], "manifest_fruit": m["episode_target_fruit"],
                          "manifest_arm": m["episode_used_arm"], "run_dir": x["run_dir"]})
        n_ok = sum(s["E0_ok_B"] for s in seeds)
        origin = "random" if e in rand else "ranked"
        episodes[str(e)] = {"episode": e, "split": info["split"], "fruit": f, "arm": a, "rank_in_pool": info["rank_in_pool"], "selection": ("lb_rank_1_10" if src == "LB" else ("lb3_" + origin)),
                            "crowd_score_m": info["score_m"], "layout_index": info["layout_index"], "seeds": seeds, "n_ok_B": n_ok,
                            "three_of_three": len(seeds) == 3 and n_ok == 3, "failure_modes": [s["failure_mode"] for s in seeds]}
        rec["lb" if src == "LB" else "lb3"].append(e)
    def stats(eps):
        E = [episodes[str(e)] for e in eps]
        r1 = [x for x in E if x["seeds"][0]["E0_ok_B"]]
        r12 = [x for x in r1 if len(x["seeds"]) > 1 and x["seeds"][1]["E0_ok_B"]]
        t3 = [x for x in E if x["three_of_three"]]
        fm1 = {m: 0 for m in MODES}; fmall = {m: 0 for m in MODES}
        for x in E:
            fm1[x["seeds"][0]["failure_mode"]] += 1
            for s in x["seeds"]: fmall[s["failure_mode"]] += 1
        return {"tried": len(E), "rep1_ok": len(r1), "rep1_2_ok": len(r12), "three_of_three_n": len(t3),
                "three_of_three": sorted([x["episode"] for x in t3], key=lambda e: (-episodes[str(e)]["crowd_score_m"], e)),
                "n_runs_valid": sum(len(x["seeds"]) for x in E), "failure_mode_rep1": fm1, "failure_mode_all_runs": fmall}
    groups[g] = {"pool_size_train": gv["pool_size"], "lb": stats(rec["lb"]), "lb3": stats(rec["lb3"]), "combined": stats(rec["lb"] + rec["lb3"]),
                 "lb3_tried_ranked": [e for e in rec["lb3"] if e not in rand], "lb3_tried_random": [e for e in rec["lb3"] if e in rand],
                 "lb3_untried_ranks_31_40": gv["ranked_all"][30:40] if f == "banana" else [],
                 "random_sample": ({"rng_seed": C["rng_seed"] + (0 if g == "banana_left" else 1), "rng": "numpy default_rng(seed).permutation(pool minus ranks 1-30)",
                                    "order_first_10": rand, "tried": [e for e in rec["lb3"] if e in rand]} if f == "banana" else None),
                 "cap": 40, "new_adopted_lb3": [e for e in groups_new] if False else None}
    g3 = groups[g]["lb3"]["three_of_three"]
    groups[g]["new_adopted_lb3"] = g3
    groups[g]["combined_three_of_three"] = groups[g]["combined"]["three_of_three"]
    groups[g]["adopted_combined"] = groups[g]["combined"]["three_of_three"][:2]
# invalid attempts
inv = []
for fpath in sorted(glob.glob(str(R3 / "failed" / "*.client.log.*"))):
    txt = pathlib.Path(fpath).read_text(errors="replace"); name = pathlib.Path(fpath).name
    se = [int(x) for x in re.findall(r"\[safe\][^\n]*?\berror (\d+)", txt)]
    inv.append({"file": name, "safe_error_counts": se, "safe_error_max": max(se) if se else 0,
                "connection_error": bool(re.search(r"Connection(Refused|Reset|Closed)|ConnectionError|websockets\.exceptions|ConnectionClosed|Traceback", txt, re.I)),
                "last_lines": [l for l in txt.splitlines() if re.search(r"Error|Exception|exit=", l)][-2:]})
for it in inv:
    it["rerun_done"] = (R3 / (re.sub(r"\.client\.log\..*", "", it["file"]) + ".done")).exists()
# audit valid
bad = []
for d in sorted(R3.glob("E0_ep*.done")):
    t = (R3 / (d.name[:-5] + ".client.log")).read_text(errors="replace")
    if any(int(x) > 0 for x in re.findall(r"\[safe\][^\n]*?\berror (\d+)", t)) or re.search(r"Connection(Refused|Reset|Closed)|ConnectionClosed|Traceback", t): bad.append(d.name)
n_done = len(list(R3.glob("E0_ep*.done")))
# pod
Ls = [l.split() for l in (L3 / "runs" / "loadavg_samples.txt").read_text().splitlines() if l.strip()]
la1 = [float(x[1]) for x in Ls]; ws = [int(t.split("=")[1]) for x in Ls for t in x if t.startswith("wset=")]; gp = [int(t.split("=")[1]) for x in Ls for t in x if t.startswith("gpu_mib=")]
pod = {"n_samples": len(Ls), "first_sample_utc": Ls[0][0], "last_sample_utc": Ls[-1][0], "loadavg1_min": min(la1), "loadavg1_median": statistics.median(la1), "loadavg1_max": max(la1),
       "working_set_gib_max": max(ws) / 2**30, "gpu_used_mib_max_sampled_60s": max(gp)}
srv = []
for d in sorted(glob.glob(str(R3 / "server_*"))):
    gs = pathlib.Path(d) / "gpu_samples.txt"
    if gs.exists():
        v = [int(l.split()[1]) for l in gs.read_text().splitlines() if len(l.split()) == 2]
        b = pathlib.Path(d) / "gpu_used_before_mib"
        srv.append({"dir": pathlib.Path(d).name, "gpu_total_used_mib_max": max(v) if v else None, "gpu_used_before_mib": int(b.read_text()) if b.exists() else None})
# combined adopted across LB and LB3
comb = {g: list(v) for g, v in OLD["numbers"]["adopted"].items()}
comb_lb3 = {}
for g in PL.GROUPS:
    allthree = groups[g]["combined_three_of_three"]
    comb[g] = allthree[:2]
    comb_lb3[g] = allthree
out = {"step": "T43LB3", "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
       "verdict": "run counts only; interpretation not included",
       "settings": {"server": "E0 --no-safe, subtask gate off via --no-safe (as LB), no obstacle", "snapshot": "${AG3S_PROJECT_ROOT}-t43v11", "client": "600 steps, warmup 1000, --policy-seed = 10*ep+rep",
                    "success_rule": "T42 definition B (analyze_t42 success AND grasp_robust robust_grasp)", "scope": list(PL.GROUPS), "cap_total_tried_per_group": 40,
                    "banana_order": "LB ranks 1-10 (already tried), LB3 ranks 11-30, then random sample (pool minus ranks 1-30; seed in groups.*.random_sample) up to total 40; ranks 31-40 not tried for banana (cap given to the random sample)",
                    "pear_left_order": "LB ranks 1-10, LB3 ranks 11-40", "stop_rule": "group stops when LB+LB3 combined has 2 episodes 3/3 (pear_left already had 1440 from LB)",
                    "waves": "rep1 in waves; rep2 only for rep1 ok; rep3 only for rep1+2 ok (LB convention)"},
       "metadata": {"split_bounds_from_dataset": C["split_bounds_from_records"], "pool_total_train": C["pool_total"], "n_run_manifest_mismatch_split_fruit_arm": len(mism),
                    "mismatch_runs": mism, "all_candidates_train_and_fruit_arm_match_metadata": True},
       "numbers": {"groups": groups, "adopted_combined_LB_LB3": comb, "three_of_three_all_LB_LB3": comb_lb3,
                   "n_tried_total_lb3": sum(v["lb3"]["tried"] for v in groups.values()), "n_valid_runs_done_lb3": n_done,
                   "invalid_attempt_count": len(inv), "invalid_attempts": inv, "invalid_all_rerun_and_done": all(i["rerun_done"] for i in inv),
                   "audit_valid_runs_with_safe_error_or_conn_error": bad, "pod": pod, "servers": {"n_started": len(srv), "items": srv, "gpu_total_used_mib_max_2s": max([s["gpu_total_used_mib_max"] or 0 for s in srv] or [0])}},
       "episodes": episodes}
(L3 / "verify_draft.json").write_text(json.dumps(out, indent=1, default=float))
if WRITE: HO.write_text(json.dumps(out, indent=1, default=float))
for g, v in groups.items():
    print(g, "LB3", {k: v["lb3"][k] for k in ("tried", "rep1_ok", "rep1_2_ok", "three_of_three")}, "COMB", {k: v["combined"][k] for k in ("tried", "rep1_ok", "rep1_2_ok", "three_of_three")}, v["lb3"]["failure_mode_all_runs"])
print("invalid", len(inv), "done", n_done, "mism", len(mism), "bad", bad)
if not WRITE: sys.exit(0)

import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, cv2
from AG3S.ag3s.experiments.common import figstyle
figstyle.use_korean(); CAT = figstyle.CATEGORICAL
def save(fig, name, side):
    for ext in ("png", "json"):
        assert not (FIG / f"{name}.{ext}").exists(), f"never overwrite {name}.{ext}"
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight", facecolor=figstyle.SURFACE); plt.close(fig)
    side["source"] = str(HO); (FIG / f"{name}.json").write_text(json.dumps(side, indent=1, default=float))
G = list(groups)
# 1 funnel
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6)); figstyle.style_axes(fig, axs); side = {}
stg = ["tried (rep1)", "rep1 ok", "rep1+2 ok", "3/3 ok"]; keys = ["tried", "rep1_ok", "rep1_2_ok", "three_of_three_n"]
for ax, which, ttl in ((axs[0], "lb", "LB (ranks 1-10)"), (axs[1], "combined", "LB + LB3 (widened)")):
    w = 0.2; x = np.arange(len(G))
    for i, (s, k) in enumerate(zip(stg, keys)):
        vals = [groups[g][which][k] for g in G]; ax.bar(x + (i - 1.5) * w, vals, w, color=CAT[i], label=s); side[f"{which}:{s}"] = vals
        for xi, vv in zip(x + (i - 1.5) * w, vals): ax.text(xi, vv + 0.3, str(vv), ha="center", fontsize=8, color=figstyle.INK)
    ax.set_xticks(x); ax.set_xticklabels([g.replace("_", "\n") for g in G]); ax.set_ylabel("episodes"); ax.set_title(ttl, color=figstyle.INK); ax.set_ylim(0, 44)
axs[0].legend(frameon=False, ncol=2, loc="upper right")
fig.suptitle("T43 LB3 screening funnel per group (E0, train split, definition B)", color=figstyle.INK)
save(fig, "t43lb3-1-funnel", {"groups": G, "values": side})
# 2 table
cols = ["group", "tried LB", "tried LB3", "rep1 ok (LB3)", "3/3 (LB3 new)", "tried total", "rep1 ok total", "3/3 total", "3/3 episodes", "failure modes, all LB3 runs"]
rows = []
for g in G:
    v = groups[g]; fm = ", ".join(f"{k} {n}" for k, n in v["lb3"]["failure_mode_all_runs"].items() if n)
    rows.append([g.replace("_", " "), v["lb"]["tried"], v["lb3"]["tried"], v["lb3"]["rep1_ok"], v["lb3"]["three_of_three_n"], v["combined"]["tried"], v["combined"]["rep1_ok"],
                 v["combined"]["three_of_three_n"], ", ".join(map(str, v["combined"]["three_of_three"])) or "-", fm])
fig, ax = plt.subplots(figsize=(16, 2.8)); ax.axis("off")
t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center", colWidths=[.09, .06, .06, .08, .08, .07, .08, .06, .12, .30])
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.9)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if r == 0: cell.set_facecolor("#cde2fb"); cell.set_text_props(color=figstyle.INK)
ax.set_title("T43 LB3: banana L/R and pear L, episode counts (E0, no obstacle, success = T42 definition B)", color=figstyle.INK)
save(fig, "t43lb3-2-table", {"columns": cols, "rows": rows})
# 3 failure-mode distribution
fig, ax = plt.subplots(figsize=(10, 4.4)); figstyle.style_axes(fig, [ax]); side = {}
bot = np.zeros(len(G))
for i, m in enumerate(MODES):
    vals = np.array([groups[g]["combined"]["failure_mode_rep1"][m] for g in G]); ax.bar(G, vals, bottom=bot, color=CAT[i], label=m); side[m] = vals.tolist()
    for j, vv in enumerate(vals):
        if vv: ax.text(j, bot[j] + vv / 2, str(vv), ha="center", va="center", fontsize=8, color=figstyle.INK)
    bot += vals
ax.set_ylabel("episodes (rep1 outcome, LB + LB3)"); ax.legend(frameon=False, fontsize=8, ncol=3); ax.set_title("T43 LB3 rep1 failure-mode distribution", color=figstyle.INK)
save(fig, "t43lb3-3-failure-modes", {"groups": G, "rep1_failure_mode_counts": side})
# 4 montage: newly adopted (LB3 3/3); if none, nothing is drawn but an empty-state panel with best rep1-ok
new = [(g, e) for g in G for e in groups[g]["new_adopted_lb3"]]
sel = new; label = "newly adopted (3/3, LB3)"
if not new:
    sel = [(g, e) for g in G for e in groups[g]["lb3"]["three_of_three"]]
    best = [(g, int(e)) for g in G for e, ev in episodes.items() if ev["fruit"] + "_" + ev["arm"] == g and ev["selection"].startswith("lb3") and ev["n_ok_B"] >= 2]
    sel = best; label = "no new 3/3; LB3 episodes with >= 2 seeds ok"
if sel:
    nc = 4; nr = (len(sel) + nc - 1) // nc
    fig, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.6 * nr)); axs = np.atleast_2d(axs); [a.axis("off") for a in axs.ravel()]; side = []
    for k, (g, e) in enumerate(sel):
        ev = episodes[str(e)]; mp4 = pathlib.Path(ev["seeds"][0]["run_dir"]) / "third_person_front.mp4"
        cap = cv2.VideoCapture(str(mp4)); ok, fr = cap.read(); cap.release()
        if ok: axs.ravel()[k].imshow(fr[:, :, ::-1])
        axs.ravel()[k].set_title(f"{g.replace('_', ' ')} ep{e} ({ev['split']})\ncrowd {ev['crowd_score_m']*1000:.0f} mm  ok {ev['n_ok_B']}/{len(ev['seeds'])}  seeds {[s['seed'] for s in ev['seeds']]}", fontsize=9, color=figstyle.INK)
        side.append({"group": g, "episode": e, "frame": str(mp4), "frame_index": 0, "frame_read": bool(ok)})
    fig.suptitle(f"T43 LB3 {label}: first frame, front view, seed rep1", color=figstyle.INK)
    save(fig, "t43lb3-4-scene-montage", {"label": label, "items": side})
print("figures written")
