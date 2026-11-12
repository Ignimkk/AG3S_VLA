#!/usr/bin/env python3
"""(2026-11-13 sdm-f6) Rule-A figure for plan §9.34: SL-SOTA18 -gh156 (hand-sized clusters are graspable) on the 1/10 table set.
(a) real scene, last video frame, frozen vs -gh156: SafeLIBERO Long L-II task 3 (mug + pudding) and LIBERO-10 task 4 (two mugs);
(b) pre-grasp calls in which a mug-sized cluster (all principal extents 80-156 mm) was not_graspable / admissible, six mug episodes;
(c) the 1/10 set, frozen vs -gh156: LIBERO successes, SafeLIBERO L-I / L-II TSR, SSR, CAR (episode counts).
usage: OMP_NUM_THREADS=1 .venv-ag3s/bin/python experiments/segmentation/sota_offline_20261111/fig_gh156_sdm.py STAGE2.json OUT.png"""
import glob, json, sys
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.experiments.common import figstyle

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
R = json.load(open(sys.argv[1])); out = sys.argv[2]
ko = figstyle.use_korean()


def eps_of(D):
    ch = sorted(glob.glob(D + "/server_constraints/run_0000/chunk_*.npz")); eps = []; cur = []; last = -1
    for f in ch:
        ts = int(np.load(f, allow_pickle=True)["t_step"])
        if ts <= last and cur:
            eps.append(cur); cur = []
        cur.append(f); last = ts
    eps.append(cur)
    return eps


def mug_calls(pat, idx):
    files = eps_of(sorted(glob.glob(B + pat))[-1])[idx]
    n = rej = adm = 0
    for f in files:
        sm = json.loads(str(np.load(f, allow_pickle=True)["summary_json"]))
        if ((sm.get("grasp") or {}).get("state")) in ("held", "closing"):
            continue
        n += 1
        big = [c for c in ((sm.get("admissibility") or {}).get("clusters") or []) if c.get("extents_mm") and min(c["extents_mm"]) >= 80 and max(c["extents_mm"]) <= 156]
        rej += any(c["reason"] == "not_graspable" for c in big); adm += any(c["reason"] == "admissible" for c in big)
    return n, rej, adm


def frame(pat, t):
    f = glob.glob(B + pat + f"/videos/*t{t}_e19_*.mp4")[-1]
    rd = imageio.get_reader(f); n = rd.count_frames()
    return rd.get_data(n - 1)


fig = plt.figure(figsize=(17, 9.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.36, wspace=0.26)
axes = []
shots = [("2*_ms-tab10sdm-long-LII-i19-*", 3, "Long L-II task 3, frozen", R["safelibero"]["long L-II"]["frozen"]["3"][0]),
         ("2*_tab10sdm-gh156-long-LII-*", 3, "Long L-II task 3, -gh156", R["safelibero"]["long L-II"]["gh156"]["3"][0]),
         ("2*_tab10sdm-lb-10-vla_v11ab-*", 4, "LIBERO-10 task 4, frozen", "success" if R["libero"]["10"]["frozen"]["4"][2] else "failure"),
         ("2*_tab10sdm-gh156-lb-10-*", 4, "LIBERO-10 task 4, -gh156", "success" if R["libero"]["10"]["gh156"]["4"][2] else "failure")]
for k, (pat, t, name, res) in enumerate(shots):
    ax = fig.add_subplot(gs[0, k]); ax.imshow(frame(pat, t)); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"(a{k + 1}) {name}: {res} (last frame)", fontsize=9.5)
cases = [("Long L-I t2", "2*_ms-tab10sdm-long-LI-i19-*", "2*_tab10sdm-gh156-long-LI-*", 2, 0), ("Long L-I t3", "2*_ms-tab10sdm-long-LI-i19-*", "2*_tab10sdm-gh156-long-LI-*", 3, 1),
         ("Long L-II t2", "2*_ms-tab10sdm-long-LII-i19-*", "2*_tab10sdm-gh156-long-LII-*", 2, 0), ("Long L-II t3", "2*_ms-tab10sdm-long-LII-i19-*", "2*_tab10sdm-gh156-long-LII-*", 3, 1),
         ("LIBERO-10 t4", "2*_tab10sdm-lb-10-vla_v11ab-*", "2*_tab10sdm-gh156-lb-10-*", 2, 0), ("LIBERO-10 t6", "2*_tab10sdm-lb-10-vla_v11ab-*", "2*_tab10sdm-gh156-lb-10-*", 3, 1)]
axm = fig.add_subplot(gs[1, 0:2]); axes.append(axm)
rec = []
for j, (name, pf, pg, i0, i1) in enumerate(cases):
    n0, r0, a0 = mug_calls(pf, i0); n1, r1, a1 = mug_calls(pg, i1)
    axm.bar(j - 0.2, 100 * r0 / n0, 0.38, color=figstyle.CATEGORICAL[1], label="frozen: mug-sized cluster not graspable" if j == 0 else None)
    axm.bar(j + 0.2, 100 * a1 / n1, 0.38, color=figstyle.CATEGORICAL[0], label="-gh156: mug-sized cluster admissible" if j == 0 else None)
    axm.text(j - 0.2, 100 * r0 / n0 + 1.5, f"{r0}/{n0}", ha="center", fontsize=8); axm.text(j + 0.2, 100 * a1 / n1 + 1.5, f"{a1}/{n1}", ha="center", fontsize=8)
    rec.append({"episode": name, "frozen": {"pre_grasp_calls": n0, "not_graspable": r0, "admissible": a0}, "gh156": {"pre_grasp_calls": n1, "not_graspable": r1, "admissible": a1}})
axm.set_xticks(range(len(cases))); axm.set_xticklabels([c[0] for c in cases], fontsize=8.5); axm.set_ylim(0, 135); axm.set_ylabel("% of pre-grasp calls")
axm.legend(fontsize=8.5, loc="upper right"); axm.set_title("(b) the mug as a target candidate (frozen: admissible in 0 calls; -gh156: not graspable in 0 calls)", fontsize=10)
axc = fig.add_subplot(gs[1, 2:4]); axes.append(axc)
def cnt(v):
    tsr = ssr = car = 0
    for cell, d in R["safelibero"].items():
        for lab, _, ok in d[v].values():
            tsr += ok; ssr += lab == "SS"; car += lab in ("SS", "F")
    return tsr, ssr, car
lib = {v: sum(x[2] for s in R["libero"].values() for x in s[v].values()) for v in ("frozen", "gh156", "pi05")}
f, g = cnt("frozen"), cnt("gh156")
labs = ["LIBERO SR\n(of 20)", "SafeLIBERO TSR\n(of 32)", "SafeLIBERO SSR\n(of 32)", "SafeLIBERO CAR\n(of 32)"]
vf, vg = [lib["frozen"], f[0], f[1], f[2]], [lib["gh156"], g[0], g[1], g[2]]
x = np.arange(4)
axc.bar(x - 0.2, vf, 0.38, color=figstyle.CATEGORICAL[1], label="frozen (pkgFrs)"); axc.bar(x + 0.2, vg, 0.38, color=figstyle.CATEGORICAL[0], label="-gh156")
for i in range(4):
    axc.text(i - 0.2, vf[i] + 0.4, str(vf[i]), ha="center", fontsize=9); axc.text(i + 0.2, vg[i] + 0.4, str(vg[i]), ha="center", fontsize=9)
axc.plot([-0.45, 0.45], [lib["pi05"]] * 2, color=figstyle.INK, lw=1.3, ls="--", label=f"pi0.5 alone on LIBERO ({lib['pi05']})")
axc.set_xticks(x); axc.set_xticklabels(labs, fontsize=8.5); axc.set_ylim(0, 30); axc.set_ylabel("episodes"); axc.legend(fontsize=8.5, loc="upper left")
axc.set_title("(c) the 1/10 table set (52 episodes), same episodes: no episode is lost", fontsize=10)
figstyle.style_axes(fig, axes)
fig.suptitle("SL-SOTA18 -gh156: the mug was rejected as 'not graspable' (narrowest extent > 78 mm gripper opening); admitting hand-sized clusters turns 2 of 6 mug episodes into successes",
             fontsize=11.5, y=0.985)
fig.savefig(out, dpi=140, bbox_inches="tight")
json.dump({"figure": out, "source": sys.argv[1], "set": "1/10 of the paper-table episodes: SafeLIBERO 32 ep + LIBERO 20 ep, init 19, seed 7",
           "mug_candidate_calls": rec, "libero_success_of_20": lib, "safelibero_tsr_ssr_car_of_32": {"frozen": f, "gh156": g},
           "korean_font": bool(ko)}, open(out[:-4] + ".json", "w"), indent=1, ensure_ascii=False)
print(lib, f, g)
