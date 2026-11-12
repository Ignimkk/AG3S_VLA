#!/usr/bin/env python3
"""(2026-11-12 sdm-f6) Rule-A figure for plan §9.25 (SDM): A-1 pkgFrs E1a 40 and A-3 SL-SOTA14 -zp5 screen (rejected).
(a) real scene: E1a task 0 init 14 and task 3 init 8 (both lost by -zp5), pkgFrs vs -zp5, last video frame;
(b) per-episode outcome: pkgFrs E1a 40 (SDM) and -zp5 task 0 · 3 (SDM; tasks 1 · 2 not measured);
(c) safe successes on tasks 0 and 3. AMRC values of -zp are the numbers written in plan §9.22 (no run on this machine).
usage: OMP_NUM_THREADS=1 .venv-ag3s/bin/python experiments/segmentation/sota_offline_20261111/fig_zp5_sdm.py OUT.png"""
import glob, json, sys
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.experiments.common import figstyle

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
out = sys.argv[1]
TAGS = [("pkgFrs (SDM)", "sub-pkgFrssdm"), ("-rs -zp5 (SDM)", "sub-pkgSzp5sdm")]
AMRC_ZP = {0: 10, 3: 2}          # plan §9.22, -hb-qs-rs-zp-tro on AMRC (written values)


def run(tag, t):
    return sorted(glob.glob(B + f"2*_{tag}-e1a-spatial-LI-t{t}-*"))[-1]


def e1a(tag):
    r = {}
    for t in range(4):
        ds = sorted(glob.glob(B + f"2*_{tag}-e1a-spatial-LI-t{t}-*"))
        if not ds:
            continue
        for l in open(ds[-1] + "/episodes.jsonl"):
            e = json.loads(l)
            sl = e.get("safelibero") or {}
            s, c = bool(e.get("official_success")), bool(sl.get("collided"))
            r[(t, e["init_state_index"])] = ("SS" if s and not c else "S+C" if s else "F+C" if c else "F",
                                             1000 * float(sl.get("max_displacement_m") or 0), int(e.get("policy_calls") or 0))
    return r


def frame(tag, t, init, frac):
    f = glob.glob(run(tag, t) + f"/videos/*t{t}_e{init}_*.mp4")[0]
    rd = imageio.get_reader(f)
    n = rd.count_frames()
    return np.asarray(rd.get_data(min(int(frac * n), n - 1))), n


ko = figstyle.use_korean()
R = {name: e1a(tag) for name, tag in TAGS}
assert len(R["pkgFrs (SDM)"]) == 40 and len(R["-rs -zp5 (SDM)"]) == 20, {k: len(v) for k, v in R.items()}
fig = plt.figure(figsize=(17, 9.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.9], hspace=0.3, wspace=0.12)
axes = []
shots = [("pkgFrs (SDM)", 0, 14, 0.99), ("-rs -zp5 (SDM)", 0, 14, 0.99), ("pkgFrs (SDM)", 3, 8, 0.99), ("-rs -zp5 (SDM)", 3, 8, 0.99)]
meta = []
for k, (name, t, i, fr) in enumerate(shots):
    ax = fig.add_subplot(gs[0, k]); axes.append(ax)
    tag = dict(TAGS)[name]
    try:
        img, n = frame(tag, t, i, fr)
        ax.imshow(img)
    except Exception as exc:  # the video is not there: say so rather than draw nothing
        n = 0
        ax.text(0.5, 0.5, f"no video\n{type(exc).__name__}", ha="center", va="center", transform=ax.transAxes)
    res, disp, calls = R[name][(t, i)]
    ax.set_title(f"(a{k + 1}) task {t} init {i}, {name}: {res}, {calls} calls", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    meta.append({"tag": tag, "task": t, "init": i, "fraction": fr, "frames": n, "result": res, "policy_calls": calls})
axg = fig.add_subplot(gs[1, 0:3]); axes.append(axg)
order = ["SS", "S+C", "F+C", "F", "-"]
cols = [figstyle.CATEGORICAL[0], figstyle.CATEGORICAL[1], figstyle.CATEGORICAL[3], figstyle.GRID_INK, "#ffffff"]
keys = [(t, i) for t in range(4) for i in range(5, 15)]
M = np.array([[order.index(R[name].get(k, ("-",))[0]) for k in keys] for name, _ in TAGS])
axg.imshow(M, cmap=ListedColormap(cols), vmin=-0.5, vmax=4.5, aspect="auto")
for r, (name, _) in enumerate(TAGS):
    for c, k in enumerate(keys):
        if k not in R[name]:
            continue
        cls, disp, _ = R[name][k]
        txt = cls if disp < 1.0 else f"{cls}\n{disp:.0f}"
        axg.text(c, r, txt, ha="center", va="center", fontsize=6.5, color="white" if cls != "F" else figstyle.INK)
for t in (1, 2):
    axg.text(10 * t + 4.5, 1, "not measured\n(screen stopped after\ntasks 0 and 3)", ha="center", va="center", fontsize=8, color=figstyle.INK)
axg.set_yticks(range(len(TAGS))); axg.set_yticklabels([n for n, _ in TAGS])
axg.set_xticks(range(40)); axg.set_xticklabels([str(i) for _, i in keys], fontsize=7)
for t in range(1, 4):
    axg.axvline(10 * t - 0.5, color=figstyle.INK, lw=1.5)
for t in range(4):
    axg.text(10 * t + 4.5, -0.56, f"task {t}", ha="center", va="bottom", fontsize=10)
axg.grid(False)
axg.set_xlabel("init (E1a = SafeLIBERO Spatial L-I task 0-3 x init 5-14, seed 7); number = obstacle displacement in mm")
axg.set_title("(b) episode outcome: SS safe success, S+C success with collision, F+C failure with collision, F failure", fontsize=10, pad=30)
axb = fig.add_subplot(gs[1, 3]); axes.append(axb)
w = 0.26
per = {name: {t: sum(R[name][(t, i)][0] == "SS" for i in range(5, 15)) for t in (0, 3)} for name, _ in TAGS}
series = [("pkgFrs (SDM)", per["pkgFrs (SDM)"]), ("-rs -zp5 (SDM)", per["-rs -zp5 (SDM)"]), ("-rs -zp (AMRC, plan §9.22)", AMRC_ZP)]
for j, (name, v) in enumerate(series):
    xs = np.arange(2) + (j - 1) * w
    axb.bar(xs, [v[0], v[3]], w, color=figstyle.CATEGORICAL[j], label=name)
    for x, y in zip(xs, [v[0], v[3]]):
        axb.text(x, y + 0.15, str(y), ha="center", fontsize=8)
axb.plot([-0.45, 0.45], [9, 9], color=figstyle.INK, lw=1.3, ls="--", label="pass line: task 0 >= 9; task 3 >= pkgFrs")
axb.plot([0.55, 1.45], [per["pkgFrs (SDM)"][3]] * 2, color=figstyle.INK, lw=1.3, ls="--")
axb.set_xticks(range(2)); axb.set_xticklabels(["task 0", "task 3"])
axb.set_ylim(0, 17); axb.set_ylabel("safe successes / 10")
axb.legend(fontsize=7.5, loc="upper right")
axb.set_title("(c) safe successes on the screened tasks", fontsize=10)
figstyle.style_axes(fig, axes)
v = list(R["pkgFrs (SDM)"].values())
tot = {"tsr": sum(x[0] in ("SS", "S+C") for x in v), "car": sum(x[0] in ("SS", "F") for x in v), "ssr": sum(x[0] == "SS" for x in v),
       "over_50mm": sum(x[1] > 50 for x in v)}
fig.suptitle(f"SDM: pkgFrs E1a 40 = TSR {tot['tsr']} / CAR {tot['car']} / SSR {tot['ssr']};  SL-SOTA14 -zp5 (palm rows kept on executed steps 0-4 only) "
             "fails the screen on tasks 0 and 3 (rejected)", fontsize=12, y=0.97)
fig.savefig(out, dpi=140, bbox_inches="tight")
a, b = R["pkgFrs (SDM)"], R["-rs -zp5 (SDM)"]
json.dump({"figure": out, "set": "E1a = SafeLIBERO Spatial L-I task 0-3 x init 5-14, seed 7 (development split); -zp5 measured on tasks 0 and 3 only",
           "pkgFrs_sdm_e1a40": tot, "per_task_ssr_pkgFrs_sdm": [sum(a[(t, i)][0] == "SS" for i in range(5, 15)) for t in range(4)],
           "screen_ssr": {n: {"task0": v[0], "task3": v[3]} for n, v in series},
           "screen_tsr_car_ssr": {name: {f"task{t}": [sum(R[name][(t, i)][0] in ("SS", "S+C") for i in range(5, 15)),
                                                       sum(R[name][(t, i)][0] in ("SS", "F") for i in range(5, 15)),
                                                       sum(R[name][(t, i)][0] == "SS" for i in range(5, 15))] for t in (0, 3)} for name, _ in TAGS},
           "episodes": {name: {f"t{t}e{i}": {"class": x[0], "displacement_mm": round(x[1], 1), "policy_calls": x[2]} for (t, i), x in sorted(R[name].items())} for name in R},
           "zp5_vs_pkgFrs": {"gain": [f"t{t}e{i}" for (t, i) in sorted(b) if b[(t, i)][0] == "SS" and a[(t, i)][0] != "SS"],
                             "loss": [f"t{t}e{i}" for (t, i) in sorted(b) if b[(t, i)][0] != "SS" and a[(t, i)][0] == "SS"]},
           "amrc_zp_written_values": {"source": "plan §9.22", "task0_ssr": 10, "task3_ssr": 2},
           "scene_frames": meta, "korean_font": bool(ko)}, open(out[:-4] + ".json", "w"), indent=1, ensure_ascii=False)
print(tot, per)
