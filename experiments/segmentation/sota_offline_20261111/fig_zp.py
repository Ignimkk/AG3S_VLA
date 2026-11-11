#!/usr/bin/env python3
"""(2026-11-12 dev-57) Rule-A figure for plan §9.22 (SL-SOTA13 -zp, not adopted) and §9.23 A-0.
(a) real scene: E1a task 0 init 7 (gained) and task 3 init 10 (lost), pkgFin vs -zp, same video fraction;
(b) per-episode outcome of E1a 40 for pkgFin / -rs -zr60 / -rs -zp; (c) safe successes per task; totals table.
usage: OMP_NUM_THREADS=1 .venv-ag3s/bin/python outputs/2026-11-11_SOTA_개선/fig_zp.py OUT.png"""
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
TAGS = [("pkgFin", ["sub-pkgFin"]), ("-rs -zr60", ["sub-pkgFin2", "sub-pkgSrz"]), ("-rs -zp", ["sub-pkgSzp"])]


def run(tag, t):
    return sorted(glob.glob(B + f"2*_{tag}-e1a-spatial-LI-t{t}-*"))[-1]


def e1a(tags):
    r = {}
    for t in range(4):
        for tag in tags:
            ds = sorted(glob.glob(B + f"2*_{tag}-e1a-spatial-LI-t{t}-*"))
            if not ds:
                continue
            for l in open(ds[-1] + "/episodes.jsonl"):
                e = json.loads(l)
                sl = e.get("safelibero") or {}
                s, c = bool(e.get("official_success")), bool(sl.get("collided"))
                r[(t, e["init_state_index"])] = "SS" if s and not c else "S+C" if s else "F+C" if c else "F"
            break
    return r


def frame(tag, t, init, frac):
    f = glob.glob(run(tag, t) + f"/videos/*t{t}_e{init}_*.mp4")[0]
    rd = imageio.get_reader(f)
    n = rd.count_frames()
    return np.asarray(rd.get_data(min(int(frac * n), n - 1))), n


ko = figstyle.use_korean()
R = {name: e1a(tags) for name, tags in TAGS}
for name in R:
    assert len(R[name]) == 40, (name, len(R[name]))
fig = plt.figure(figsize=(17, 10.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15], hspace=0.28, wspace=0.12)
axes = []
shots = [("sub-pkgFin", 0, 7, 0.45, "task 0 init 7, pkgFin"), ("sub-pkgSzp", 0, 7, 0.45, "task 0 init 7, -rs -zp"),
         ("sub-pkgFin", 3, 10, 0.98, "task 3 init 10, pkgFin"), ("sub-pkgSzp", 3, 10, 0.98, "task 3 init 10, -rs -zp")]
meta = []
for k, (tag, t, i, fr, title) in enumerate(shots):
    ax = fig.add_subplot(gs[0, k]); axes.append(ax)
    try:
        img, n = frame(tag, t, i, fr)
        ax.imshow(img)
    except Exception as exc:  # the video is not there: say so rather than draw nothing
        n = 0
        ax.text(0.5, 0.5, f"no video\n{type(exc).__name__}", ha="center", va="center", transform=ax.transAxes)
    res = R["pkgFin" if tag == "sub-pkgFin" else "-rs -zp"][(t, i)]
    ax.set_title(f"(a{k + 1}) {title}: {res}  (frame {int(fr * n)}/{n})", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    meta.append({"tag": tag, "task": t, "init": i, "fraction": fr, "frames": n, "result": res})
# (b) outcome grid
axg = fig.add_subplot(gs[1, 0:3]); axes.append(axg)
order = ["SS", "S+C", "F+C", "F"]
cols = [figstyle.CATEGORICAL[0], figstyle.CATEGORICAL[1], figstyle.CATEGORICAL[3], figstyle.GRID_INK]
keys = [(t, i) for t in range(4) for i in range(5, 15)]
M = np.array([[order.index(R[name][k]) for k in keys] for name, _ in TAGS])
axg.imshow(M, cmap=ListedColormap(cols), vmin=-0.5, vmax=3.5, aspect="auto")
for r in range(M.shape[0]):
    for c in range(M.shape[1]):
        axg.text(c, r, order[M[r, c]], ha="center", va="center", fontsize=6.5, color="white" if M[r, c] != 3 else figstyle.INK)
axg.set_yticks(range(len(TAGS))); axg.set_yticklabels([n for n, _ in TAGS])
axg.set_xticks(range(40)); axg.set_xticklabels([str(i) for _, i in keys], fontsize=7)
for t in range(1, 4):
    axg.axvline(10 * t - 0.5, color=figstyle.INK, lw=1.5)
for t in range(4):
    axg.text(10 * t + 4.5, -0.56, f"task {t}", ha="center", va="bottom", fontsize=10)
axg.set_xlabel("init (E1a 40 = SafeLIBERO Spatial L-I task 0-3 x init 5-14, seed 7)")
axg.set_title("(b) episode outcome: SS safe success, S+C success with collision, F+C failure with collision, F failure", fontsize=10, pad=30)
# (c) per-task SSR + totals
axb = fig.add_subplot(gs[1, 3]); axes.append(axb)
w = 0.26
tot = {}
for j, (name, _) in enumerate(TAGS):
    per = [sum(R[name][(t, i)] == "SS" for i in range(5, 15)) for t in range(4)]
    ts = sum(v in ("SS", "S+C") for v in R[name].values()); ca = sum(v in ("SS", "F") for v in R[name].values())
    tot[name] = {"per_task_ssr": per, "tsr": ts, "car": ca, "ssr": sum(per)}
    axb.bar(np.arange(4) + (j - 1) * w, per, w, color=figstyle.CATEGORICAL[j], label=f"{name}: {ts} / {ca} / {sum(per)}")
    for t, v in enumerate(per):
        axb.text(t + (j - 1) * w, v + 0.15, str(v), ha="center", fontsize=8)
axb.set_xticks(range(4)); axb.set_xticklabels([f"task {t}" for t in range(4)])
axb.set_ylim(0, 12.5); axb.set_ylabel("safe successes / 10")
axb.legend(title="TSR / CAR / SSR of 40", fontsize=8, title_fontsize=8, loc="upper right")
axb.set_title("(c) safe successes per task", fontsize=10)
figstyle.style_axes(fig, axes)
fig.suptitle("SL-SOTA13 -zp: keeping the palm's obstacle rows inside the grasp zone - task 0 gains, tasks 2 and 3 lose (not adopted)", fontsize=12, y=0.97)
fig.savefig(out, dpi=140, bbox_inches="tight")
gain = sorted(k for k in keys if R["-rs -zp"][k] == "SS" and R["pkgFin"][k] != "SS"); loss = sorted(k for k in keys if R["-rs -zp"][k] != "SS" and R["pkgFin"][k] == "SS")
json.dump({"figure": out, "set": "E1a 40 = SafeLIBERO Spatial L-I task 0-3 x init 5-14, seed 7 (development split)", "totals": tot,
           "episodes": {name: {f"t{t}e{i}": v for (t, i), v in sorted(R[name].items())} for name in R},
           "zp_vs_pkgFin": {"gain": [f"t{t}e{i}" for t, i in gain], "loss": [f"t{t}e{i}" for t, i in loss]},
           "scene_frames": meta, "korean_font": bool(ko)}, open(out[:-4] + ".json", "w"), indent=1, ensure_ascii=False)
print(tot)
