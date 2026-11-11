#!/usr/bin/env python3
"""(2026-11-11 dev-57) Rule-A figure for plan §9.12-§9.13: E1a 40 outcome grid + totals and the LIBERO 27 grid of
the current config, the §9.9 candidate and the -qs candidate, with the action-QP solver statistics.
usage: OMP_NUM_THREADS=1 .venv-ag3s/bin/python outputs/2026-11-11_SOTA_개선/fig_qs.py OUT.png"""
import ast, glob, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.experiments.common import figstyle

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
out = sys.argv[1]
ARMS = [("current config (pe123458)", "qe-pe123458", "gtto-pe123458"), ("§9.9 candidate (-pt6-pg4-hr150)", "sub-pkgEhr150", "gtto-pkgEhr150"),
        ("+ scaled QP (-qs)", "sub-pkgFqs", "gtto-pkgFqs")]
CLS = ["SS", "S+C", "F", "F+C"]
LT = [0, 1, 2, 3, 4, 6, 7, 8, 9]


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    return x or {}


def qp_stats(dirs):
    n = cap = 0
    for d in dirs:
        for f in glob.glob(d + "/episodes/*_calls.jsonl"):
            for l in open(f):
                qp = lit((json.loads(l).get("to") or {}).get("ee_action_qp"))
                if not qp.get("status"):
                    continue
                n += 1
                it = qp.get("osqp_iter")
                it = max(it) if isinstance(it, (list, tuple)) and it else it
                last = it
                cap += bool(last is not None and last >= 20000)
    return n, cap


def e1a(tag):
    r, dirs = {}, []
    for t in range(4):
        d = sorted(glob.glob(B + f"2*_{tag}-e1a-spatial-LI-t{t}-*"))[-1]
        dirs.append(d)
        for l in open(d + "/episodes.jsonl"):
            e = json.loads(l)
            sl = e.get("safelibero") or {}
            s, c = bool(e.get("official_success")), bool(sl.get("collided"))
            r[(t, e["init_state_index"])] = ("SS" if s and not c else "S+C" if s else "F+C" if c else "F",
                                             1000 * float(sl.get("max_displacement_m") or 0))
    return r, dirs


def libero(tag):
    r, dirs = {}, []
    for s in (7, 8, 9):
        d = sorted(glob.glob(B + f"2*_{tag}-libero9s{s}-*"))[-1]
        dirs.append(d)
        for l in open(d + "/episodes.jsonl"):
            e = json.loads(l)
            r[(s, int(e["task_id"]))] = bool(e.get("official_success"))
    return r, dirs


ko = figstyle.use_korean()
E, L, QE, QL = [], [], [], []
for _, te, tl in ARMS:
    r, d = e1a(te); E.append(r); QE.append(qp_stats(d))
    r, d = libero(tl); L.append(r); QL.append(qp_stats(d))
keys = [(t, i) for t in range(4) for i in range(5, 15)]
lkeys = [(s, t) for s in (7, 8, 9) for t in LT]
C = figstyle.CATEGORICAL
colors = [C[2], C[3], "#bdbcb6", C[7]]
fig = plt.figure(figsize=(15, 8.2))
gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1], height_ratios=[1, 1])
a0, a1, a2, a3 = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2])
figstyle.style_axes(fig, [a0, a1, a2, a3])
names = [n for n, _, _ in ARMS]
# (a) E1a grid
grid = np.array([[CLS.index(r[k][0]) for k in keys] for r in E])
a0.imshow(grid, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto")
for a in range(3):
    for j, k in enumerate(keys):
        if E[a][k][1] >= 1.0:
            a0.text(j, a, f"{E[a][k][1]:.0f}", ha="center", va="center", fontsize=6.5, color=figstyle.INK)
a0.set_yticks(range(3)); a0.set_yticklabels(names, fontsize=8.5)
a0.set_xticks(range(40)); a0.set_xticklabels([str(i) for _, i in keys], fontsize=6.5)
for t in range(1, 4):
    a0.axvline(10 * t - 0.5, color=figstyle.INK, lw=1.2)
for t in range(4):
    a0.text(10 * t + 4.5, -0.72, f"task {t}", ha="center", fontsize=9, color=figstyle.INK)
a0.set_title("(a) E1a 40 — SafeLIBERO Spatial L-I, init 5–14 (" + ("숫자 = 장애물 변위 mm" if ko else "number = obstacle displacement mm") + ")", fontsize=10, loc="left", pad=24)
a0.grid(False)
lab = ["안전 성공", "성공 + 충돌", "실패 (충돌 없음)", "실패 + 충돌"] if ko else ["safe success", "success + collision", "failure (no collision)", "failure + collision"]
a0.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], lab, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5, frameon=False)
# (b) totals
stats = []
for r in E:
    v = list(r.values())
    stats.append((sum(x[0] in ("SS", "S+C") for x in v), sum(x[0] in ("SS", "F") for x in v), sum(x[0] == "SS" for x in v)))
w = 0.8 / 3
for a in range(3):
    xs = np.arange(3) + (a - 1) * w
    a1.bar(xs, [100 * s / 40 for s in stats[a]], w * 0.92, color=C[a], label=names[a])
    for x, s in zip(xs, stats[a]):
        a1.text(x, 100 * s / 40 + 1.2, str(s), ha="center", fontsize=8, color=figstyle.INK)
for x, ref in zip(range(3), (63.0, 67.5, 53.0)):
    a1.plot([x - 0.45, x + 0.45], [ref, ref], color=figstyle.INK, lw=1.3, ls="--")
a1.plot([], [], color=figstyle.INK, lw=1.3, ls="--", label="KNOWS Spatial L-I (" + ("원문 보고값" if ko else "reported") + ")")
a1.set_xticks(range(3)); a1.set_xticklabels(["TSR", "CAR", "SSR"]); a1.set_ylim(0, 100)
a1.set_title("(b) E1a 40: " + ("40 ep 중 % (막대 위 = ep 수)" if ko else "% of 40 ep (count above)"), fontsize=10, loc="left")
a1.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1)
# (c) LIBERO grid
g2 = np.array([[0 if r[k] else 2 for k in lkeys] for r in L])
a2.imshow(g2, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto")
a2.set_yticks(range(3)); a2.set_yticklabels([f"{n}  {sum(r.values())}/27" for n, r in zip(names, L)], fontsize=8.5)
a2.set_xticks(range(27)); a2.set_xticklabels([f"t{t}" for _, t in lkeys], fontsize=7.5)
for s in range(1, 3):
    a2.axvline(9 * s - 0.5, color=figstyle.INK, lw=1.2)
for i, s in enumerate((7, 8, 9)):
    a2.text(9 * i + 4, -0.72, f"seed {s}", ha="center", fontsize=9, color=figstyle.INK)
a2.set_title("(c) LIBERO 27 guard — libero_spatial 9 task × 3 seed (" + ("초록 = 성공" if ko else "green = success") + ")", fontsize=10, loc="left", pad=24)
a2.grid(False)
# (d) QP solver
xs = np.arange(2)
for a in range(3):
    n = [QE[a][0], QL[a][0]]; cap = [QE[a][1], QL[a][1]]
    x = xs + (a - 1) * w
    a3.bar(x, n, w * 0.92, color=C[a], alpha=0.35)
    a3.bar(x, cap, w * 0.92, color=C[a])
    for xx, nn, cc in zip(x, n, cap):
        a3.text(xx, nn + 12 + 38 * (a == 2), f"{cc}/{nn}", ha="center", fontsize=7, color=figstyle.INK)
a3.set_xticks(xs); a3.set_xticklabels(["E1a 40", "LIBERO 27"])
a3.set_title("(d) action-QP requests, dark = at OSQP cap", fontsize=9, loc="left")
fig.suptitle(("scaled QP (-qs): E1a 는 유지하고 LIBERO 를 되돌림 — E1a 40 · LIBERO 27 · QP 수렴" if ko else
              "scaled QP (-qs): keeps E1a, restores LIBERO — E1a 40, LIBERO 27, QP convergence"), fontsize=12, color=figstyle.INK)
fig.tight_layout()
fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
json.dump({"arms": [{"label": n, "e1a_tag": te, "libero_tag": tl, "E1a": {"TSR": s[0], "CAR": s[1], "SSR": s[2], "n": 40,
                     "episodes": {f"t{k[0]}_e{k[1]}": {"class": r[k][0], "displacement_mm": round(r[k][1], 1)} for k in keys}},
                     "LIBERO27": {"success": sum(l.values()), "episodes": {f"s{k[0]}_t{k[1]}": l[k] for k in lkeys}},
                     "action_qp": {"E1a": {"corrected_requests": qe[0], "at_osqp_cap": qe[1]}, "LIBERO27": {"corrected_requests": ql[0], "at_osqp_cap": ql[1]}}}
                    for (n, te, tl), s, r, l, qe, ql in zip(ARMS, stats, E, L, QE, QL)],
           "reference": {"KNOWS Spatial L-I reported (TSR, CAR, SSR %)": [63.0, 67.5, 53.0]},
           "source": "results/benchmark/*_<tag>-e1a-spatial-LI-t<k>-*/ and *_<tag>-libero9s<seed>-*/ (episodes.jsonl, episodes/*_calls.jsonl)"},
          open(out.rsplit(".", 1)[0] + ".json", "w"), ensure_ascii=False, indent=1)
print(out, stats, [sum(l.values()) for l in L], QE, QL, "korean", ko)
