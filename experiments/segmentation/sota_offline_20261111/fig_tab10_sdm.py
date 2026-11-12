#!/usr/bin/env python3
"""(2026-11-12 sdm-f6) Rule-A figure for plan §9.31: one tenth of the paper-table episodes with the frozen version (SafeLIBERO
part, 32 episodes, init 19) and the cause analysis of the 20 episodes that are not safe successes.
(a) real scene: Long L-II task 3 (the mug is in the hand, never latched as held) and Long L-I task 2 (hovering over the second mug);
(b) every episode: share of pre-grasp calls whose recognised target is a task object, by suite, coloured by outcome;
(c) rejected calls of the 20 episodes by the real body that blocked them; (d) safe successes per suite, AG3S vs pi0.5 alone.
Inputs: tab10_cause.py JSON (target_truth.py + surface_owner_task.py results) and table10.py JSON.
usage: OMP_NUM_THREADS=1 .venv-ag3s/bin/python experiments/segmentation/sota_offline_20261111/fig_tab10_sdm.py CAUSE.json TABLE.json OUT.png"""
import glob, json, sys
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.experiments.common import figstyle

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
C = json.load(open(sys.argv[1])); T = json.load(open(sys.argv[2])); out = sys.argv[3]
E = C["episodes"]
ko = figstyle.use_korean()
fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.34, wspace=0.28)
axes, meta = [], []
shots = [("long", "LII", 3, 0.30, "mug in the hand; the server never latches 'held'"), ("long", "LII", 3, 0.60, "the arm has left the scene with the mug"),
         ("long", "LI", 2, 0.75, "hovering over the second mug"), ("goal", "LI", 1, 0.75, "bowl put back on the table, not on the cabinet")]
for k, (s, l, t, fr, note) in enumerate(shots):
    ax = fig.add_subplot(gs[0, k]); axes.append(ax)
    f = glob.glob(B + f"2*_ms-tab10sdm-{s}-{l}-i19-*/videos/*t{t}_e19_*.mp4")[0]
    rd = imageio.get_reader(f); n = rd.count_frames()
    ax.imshow(rd.get_data(min(int(fr * n), n - 1))); ax.set_xticks([]); ax.set_yticks([])
    e = [x for x in E if x["cell"] == f"{s} L-{l[1:]}" and x["task"] == t][0]
    ax.set_title(f"(a{k + 1}) {s.capitalize()} L-{l[1:]} task {t}: {e['label']}\n{note}\ntarget on a task object in {e['target_on_task_object']} of {e['target_calls']} calls",
                 fontsize=9)
    meta.append({"cell": e["cell"], "task": t, "frame_fraction": fr, "frames": n, "label": e["label"]})
axs = fig.add_subplot(gs[1, 0:2]); axes.append(axs)
suites = ["spatial", "object", "goal", "long"]
rng = np.random.default_rng(0)
for r in E:
    y = suites.index(r["cell"].split()[0]) + (0.16 if r["cell"].endswith("L-II") else -0.16) + rng.uniform(-0.05, 0.05)
    ss = r["label"] == "SS"
    axs.scatter(100 * r["share"], y, s=70, color=figstyle.CATEGORICAL[0] if ss else figstyle.CATEGORICAL[1], marker="o" if ss else "X",
                edgecolor="white", linewidth=0.6, zorder=3)
axs.axvline(50, color=figstyle.INK, lw=1.0, ls="--")
axs.scatter([], [], s=70, color=figstyle.CATEGORICAL[0], marker="o", label="safe success")
axs.scatter([], [], s=70, color=figstyle.CATEGORICAL[1], marker="X", label="not a safe success")
axs.set_yticks(range(4)); axs.set_yticklabels([s.capitalize() for s in suites]); axs.set_ylim(3.6, -0.6)
axs.set_xlim(-4, 104); axs.set_xlabel("pre-grasp calls whose recognised target is a task object (%)  -  lower row of each suite = L-II")
lo, hi = C["summary"]["low_share"], C["summary"]["high_share"]
axs.set_title(f"(b) target identity vs outcome, 32 episodes: below 50 % -> {lo[1]} safe success of {lo[0]}; at or above -> {hi[1]} of {hi[0]}", fontsize=10)
axs.legend(fontsize=8.5, loc="lower right")
axb = fig.add_subplot(gs[1, 2]); axes.append(axb)
blk = C["summary"]["blocked_calls"]
names = ["task object", "SafeLIBERO obstacle", "other scene object", "robot itself"]
vals = [blk.get(n, 0) for n in names]
axb.barh(range(4), vals, color=[figstyle.CATEGORICAL[1], figstyle.CATEGORICAL[0], figstyle.CATEGORICAL[2], figstyle.CATEGORICAL[3]])
for i, v in enumerate(vals):
    axb.text(v + 4, i, f"{v} ({100 * v / sum(vals):.0f} %)", va="center", fontsize=9)
axb.set_yticks(range(4)); axb.set_yticklabels(["task\nobject", "SafeLIBERO\nobstacle", "other scene\nobject", "robot\nitself"], fontsize=8.5)
axb.invert_yaxis(); axb.set_xlim(0, max(vals) * 1.35); axb.set_xlabel("rejected calls (20 episodes)")
axb.set_title("(c) what blocked the rejected calls", fontsize=10)
axc = fig.add_subplot(gs[1, 3]); axes.append(axc)
S = T["safelibero"]; w = 0.38
def cnt(arm, s):
    v = [x for l in ("I", "II") for x in S[arm].get(f"{s} L-{l}", {}).values()]
    return sum(x["label"] == "SS" for x in v), len(v)
a = [cnt("AG3S-VLA (pkgFrs)", s) for s in suites]; p = [cnt("pi0.5 alone", s) for s in suites]
x = np.arange(4)
axc.bar(x - w / 2, [v[0] for v in p], w, color=figstyle.CATEGORICAL[2], label="pi0.5 alone")
axc.bar(x + w / 2, [v[0] for v in a], w, color=figstyle.CATEGORICAL[0], label="AG3S (frozen)")
for i in range(4):
    axc.text(i - w / 2, p[i][0] + 0.12, f"{p[i][0]}/{p[i][1]}", ha="center", fontsize=8.5)
    axc.text(i + w / 2, a[i][0] + 0.12, f"{a[i][0]}/{a[i][1]}", ha="center", fontsize=8.5)
axc.set_xticks(x); axc.set_xticklabels([s.capitalize() for s in suites]); axc.set_ylim(0, 8.8); axc.set_ylabel("safe successes (L-I + L-II)")
axc.legend(fontsize=8.5, loc="upper right"); axc.set_title("(d) safe successes per suite", fontsize=10)
figstyle.style_axes(fig, axes[4:])
t1 = T["tab1_counts"]["AG3S-VLA (pkgFrs)"]
fig.suptitle(f"1/10 of the paper-table episodes (SafeLIBERO init 19, 32 ep), frozen AG3S: L-I TSR/SSR/CAR {t1['L-I'][0]}/{t1['L-I'][1]}/{t1['L-I'][2]} of 16, "
             f"L-II {t1['L-II'][0]}/{t1['L-II'][1]}/{t1['L-II'][2]} of 16 - half of the rejected calls in the failed episodes are blocked by the task object itself",
             fontsize=11.5, y=0.985)
fig.savefig(out, dpi=140, bbox_inches="tight")
json.dump({"figure": out, "sources": sys.argv[1:3], "set": "SafeLIBERO 4 suites x L-I/L-II x task 0-3 x init 19, seed 7, main_aegis (1/10 of the planned 320)",
           "tab1_counts_tsr_ssr_car_n": T["tab1_counts"], "safe_success_per_suite": {s: {"ag3s": a[i], "pi05_alone": p[i]} for i, s in enumerate(suites)},
           "target_identity": C["summary"], "episodes": E, "scene_frames": meta,
           "note": "pi0.5 alone: Long L-II not measured yet, Long L-I from an interrupted run; target share uses the top-3 owners of target_truth.py (lower bound)",
           "korean_font": bool(ko)}, open(out[:-4] + ".json", "w"), indent=1, ensure_ascii=False)
print("ok", vals, a, p)
