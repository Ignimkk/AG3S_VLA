#!/usr/bin/env python3
"""(2026-11-11 dev-57) Rule-A figure of the E1a 40 comparison: per-episode outcome grid, TSR / CAR / SSR bars, failure forms.
usage: OMP_NUM_THREADS=1 .venv-ag3s/bin/python outputs/2026-11-11_SOTA_개선/fig_e1a40.py OUT.png LABEL=TAG [LABEL=TAG ...]"""
import glob, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.experiments.common import figstyle

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
out = sys.argv[1]
arms = [a.split("=", 1) for a in sys.argv[2:]]
CLS = ["SS", "S+C", "F", "F+C"]


def load(tag):
    r = {}
    for t in range(4):
        ds = sorted(glob.glob(B + f"2*_{tag}-e1a-spatial-LI-t{t}-*"))
        for l in open(ds[-1] + "/episodes.jsonl"):
            e = json.loads(l)
            sl = e.get("safelibero") or {}
            s, c = bool(e.get("official_success")), bool(sl.get("collided"))
            r[(t, e["init_state_index"])] = ("SS" if s and not c else "S+C" if s else "F+C" if c else "F",
                                             1000 * float(sl.get("max_displacement_m") or 0))
    return r


ko = figstyle.use_korean()
R = [load(tag) for _, tag in arms]
keys = [(t, i) for t in range(4) for i in range(5, 15)]
C = figstyle.CATEGORICAL
colors = [C[2], C[3], "#bdbcb6", C[7]]            # SS, S+C, F, F+C
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2.6, 1]})
figstyle.style_axes(fig, axes)
ax = axes[0]
grid = np.array([[CLS.index(r[k][0]) for k in keys] for r in R])
ax.imshow(grid, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto")
for a in range(len(R)):
    for j, k in enumerate(keys):
        cls, disp = R[a][k]
        if disp >= 1.0:
            ax.text(j, a, f"{disp:.0f}", ha="center", va="center", fontsize=6.5, color=figstyle.INK)
ax.set_yticks(range(len(R)))
ax.set_yticklabels([n for n, _ in arms], fontsize=9)
ax.set_xticks(range(40))
ax.set_xticklabels([str(i) for _, i in keys], fontsize=6.5)
for t in range(1, 4):
    ax.axvline(10 * t - 0.5, color=figstyle.INK, lw=1.2)
for t in range(4):
    ax.text(10 * t + 4.5, -0.75, f"task {t}", ha="center", fontsize=9, color=figstyle.INK)
ax.set_xlabel("init index (E1a: SafeLIBERO Spatial L-I, init 5-14); number = obstacle displacement in mm (>= 1 mm = collision)" if not ko
              else "init index (E1a: SafeLIBERO Spatial L-I, init 5–14); 숫자 = 장애물 변위 mm (1 mm 이상 = 충돌)", fontsize=8.5)
ax.grid(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
lab = ["safe success", "success + collision", "failure (no collision)", "failure + collision"] if not ko else \
      ["안전 성공", "성공 + 충돌", "실패 (충돌 없음)", "실패 + 충돌"]
ax.legend(handles, lab, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5, frameon=False)
ax = axes[1]
stats = []
for r in R:
    v = list(r.values())
    stats.append((sum(x[0] in ("SS", "S+C") for x in v), sum(x[0] in ("SS", "F") for x in v), sum(x[0] == "SS" for x in v)))
w = 0.8 / len(R)
for a, (n, _) in enumerate(arms):
    xs = np.arange(3) + (a - (len(R) - 1) / 2) * w
    bars = ax.bar(xs, [100 * s / 40 for s in stats[a]], w * 0.92, color=C[a], label=n)
    for x, s in zip(xs, stats[a]):
        ax.text(x, 100 * s / 40 + 1.2, str(s), ha="center", fontsize=8, color=figstyle.INK)
for x, ref in zip(range(3), (63.0, 67.5, 53.0)):
    ax.plot([x - 0.45, x + 0.45], [ref, ref], color=figstyle.INK, lw=1.3, ls="--")
ax.plot([], [], color=figstyle.INK, lw=1.3, ls="--", label="KNOWS Spatial L-I (reported)" if not ko else "KNOWS Spatial L-I (원문 보고값)")
ax.set_xticks(range(3))
ax.set_xticklabels(["TSR", "CAR", "SSR"])
ax.set_ylabel("% of 40 episodes (count above the bar)" if not ko else "40 ep 중 % (막대 위 = ep 수)", fontsize=9)
ax.set_ylim(0, 100)
ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
fig.suptitle("E1a 40 (development split): per-episode outcome and totals" if not ko else "E1a 40 (개발 분할): episode 별 결과와 합계",
             fontsize=11, color=figstyle.INK)
fig.tight_layout()
fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
json.dump({"arms": [{"label": n, "tag": tag, "TSR": s[0], "CAR": s[1], "SSR": s[2], "n": 40,
                     "episodes": {f"t{k[0]}_e{k[1]}": {"class": r[k][0], "displacement_mm": round(r[k][1], 1)} for k in keys}}
                    for (n, tag), s, r in zip(arms, stats, R)],
           "reference": {"KNOWS Spatial L-I reported (TSR, CAR, SSR %)": [63.0, 67.5, 53.0]},
           "source": "results/benchmark/*_<tag>-e1a-spatial-LI-t<k>-*/episodes.jsonl"},
          open(out.rsplit(".", 1)[0] + ".json", "w"), ensure_ascii=False, indent=1)
print(out, stats, "korean", ko)
