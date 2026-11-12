#!/usr/bin/env python3
"""(2026-11-12 sdm-f6) Rule-A figure for plan §9.25 A-5 / A-6 (SDM): freeze candidate pkgFrs on the other-suite set,
init 15-18 (7 cells x 16 ep = 112 ep) + Spatial L-I init 15-18 (16 ep), next to the tranche-1 baseline ports on the
same episodes (outputs/2026-11-09_baseline_tranche/t1/blocks.csv; AMRC runs, seed 7, main_aegis).
(a) real scene: last video frame of four pkgFrs episodes (one safe success, three failure forms);
(b) pooled TSR / CAR / SSR on the 112 ep; (c) safe successes per cell (16 ep each); (d) form of the pkgFrs episodes
that are not a safe success (shape.py output, the 112 ep measured on SDM).
pkgFrs Goal L-I / L-II init 15,16 were measured on AMRC only and their episode records are not on this storage:
their counts are the numbers written in plan §9.22 (AMRC_GOAL below); everything else is read from the runs.
usage: OMP_NUM_THREADS=1 .venv-ag3s/bin/python experiments/segmentation/sota_offline_20261111/fig_ms_sdm.py SHAPE.json OUT.png"""
import csv, glob, json, os, sys
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.experiments.common import figstyle

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
ROWS = list(csv.DictReader(open("/mnt/dev/AG3S_VLA/outputs/2026-11-09_baseline_tranche/t1/blocks.csv")))
shape_json, out = sys.argv[1], sys.argv[2]
CELLS = [("object", "I"), ("object", "II"), ("goal", "I"), ("goal", "II"), ("long", "I"), ("long", "II"), ("spatial", "II"), ("spatial", "I")]
AMRC_GOAL = {("goal", "I"): (8, 3, 3, 1), ("goal", "II"): (8, 3, 4, 2)}   # plan §9.22: n, TSR, CAR, SSR of pkgFrs, init 15,16
ARMS = [("pkgFrs (AG3S freeze candidate)", "pkgFrs"), ("pi0.5 only", "none"), ("AEGIS port (detector)", "aegis"),
        ("multi-link CBF port (detector)", "multilink_cbf"), ("multi-link CBF port (GT)", "multilink_cbf-gt")]
KEYS = [(t, i) for t in range(4) for i in (15, 16, 17, 18)]


def load_dir(d):
    r = {}
    if not os.path.isfile(d + "/episodes.jsonl"):
        return r
    for l in open(d + "/episodes.jsonl"):
        if l.strip():
            e = json.loads(l)
            s = e.get("safelibero") or {}
            r[(int(e["task_id"]), int(e["init_state_index"]))] = (bool(e["official_success"]), bool(s.get("collided")),
                                                                  1000 * float(s.get("max_displacement_m") or 0), int(e.get("policy_calls") or 0))
    return r


def frs(cell):
    r, dirs = {}, {}
    for d in sorted(glob.glob(B + f"2*_ms-pkgFrssdm-{cell[0]}-L{cell[1]}-i*")):
        ep = load_dir(d)
        r.update(ep)
        dirs.update({k: d for k in ep})
    return r, dirs


def arm(a, cell):
    rr = [r for r in ROWS if r["arm"] == a and r["suite"] == cell[0] and r["level"] == cell[1] and r["status"] == "current"]
    return load_dir(B + rr[-1]["run"])


def cnt(ep, keys):
    return np.array([len(keys), sum(ep[k][0] for k in keys), sum(not ep[k][1] for k in keys), sum(ep[k][0] and not ep[k][1] for k in keys)])


def cls(v):
    return "safe success" if v[0] and not v[1] else "success + collision" if v[0] else "failure + collision" if v[1] else "failure"


FRS, DIRS, per = {}, {}, {}
for cell in CELLS:
    FRS[cell], DIRS[cell] = frs(cell)
    row = {}
    have = [k for k in KEYS if k in FRS[cell]]
    c = cnt(FRS[cell], have)
    if cell in AMRC_GOAL:
        assert len(have) == 8, (cell, len(have))
        c = c + np.array(AMRC_GOAL[cell])
    assert c[0] == 16, (cell, c)
    row["pkgFrs"] = c
    for _, a in ARMS[1:]:
        ep = arm(a, cell)
        assert all(k in ep for k in KEYS), (a, cell)
        row[a] = cnt(ep, KEYS)
    per[cell] = row
MS = CELLS[:7]
pooled = {a: sum(per[c][a] for c in MS) for _, a in ARMS}
ko = figstyle.use_korean()
C = figstyle.CATEGORICAL
fig = plt.figure(figsize=(18, 10.4))
gs = fig.add_gridspec(2, 24, height_ratios=[1.0, 1.05], hspace=0.3, wspace=1.2)
axes, meta = [], []
base_o2 = arm("none", ("object", "II"))
t0, i0 = next(k for k in KEYS if k[1] in (17, 18) and FRS[("object", "II")][k][0] and not FRS[("object", "II")][k][1] and base_o2[k][1])
SHOTS = [(("object", "II"), t0, i0, "pi0.5 alone moves an obstacle on this episode"), (("long", "I"), 3, 18, "form: grasped, released, not completed"),
         (("goal", "I"), 2, 17, "form: HOLD to the end"), (("spatial", "I"), 3, 18, "form: no grasp, arm-link rows block the reach")]
for k, (cell, t, i, note) in enumerate(SHOTS):
    ax = fig.add_subplot(gs[0, 6 * k:6 * k + 6]); axes.append(ax)
    v = FRS[cell][(t, i)]
    n = 0
    try:
        f = glob.glob(DIRS[cell][(t, i)] + f"/videos/*_t{t}_e{i}_*.mp4")[0]
        rd = imageio.get_reader(f)
        n = rd.count_frames()
        ax.imshow(np.asarray(rd.get_data(n - 2)))
    except Exception as exc:  # the video is not there: say so rather than draw nothing
        ax.text(0.5, 0.5, f"no video\n{type(exc).__name__}", ha="center", va="center", transform=ax.transAxes)
    base = arm("none", cell)[(t, i)]
    disp = f" ({v[2]:.0f} mm)" if v[1] else ""
    ax.set_title(f"(a{k + 1}) {cell[0].capitalize()} L-{cell[1]} task {t} init {i}\npkgFrs: {cls(v)}{disp}; pi0.5 only: {cls(base)}\n{note}", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    meta.append({"cell": f"{cell[0]} L-{cell[1]}", "task": t, "init": i, "pkgFrs": cls(v), "displacement_mm": round(v[2], 1),
                 "policy_calls": v[3], "pi05_only": cls(base), "frames": n, "run": os.path.basename(DIRS[cell][(t, i)])})
ax = fig.add_subplot(gs[1, 0:7]); axes.append(ax)
w = 0.16
for a, (name, key) in enumerate(ARMS):
    for m in range(3):
        x = m + (a - 2) * w
        v = pooled[key][m + 1]
        ax.bar(x, 100 * v / 112, w * 0.94, color=C[a], label=name if m == 0 else None)
        ax.text(x, 100 * v / 112 + 1.0, str(int(v)), ha="center", fontsize=7.5, color=figstyle.INK)
ax.set_xticks(range(3)); ax.set_xticklabels(["TSR (success)", "CAR (no collision)", "SSR (safe success)"])
ax.set_ylim(0, 108); ax.set_ylabel("% of 112 episodes (number = episodes)")
ax.legend(fontsize=7.5, loc="upper left", ncol=2)
ax.set_title("(b) other suites, init 15-18 (7 cells, 112 ep)", fontsize=9.5)
ax = fig.add_subplot(gs[1, 8:16]); axes.append(ax)
w = 0.26
for a, (name, key) in enumerate(ARMS[:3]):
    xs = np.arange(len(CELLS)) + (a - 1) * w
    ys = [per[c][key][3] for c in CELLS]
    ax.bar(xs, ys, w * 0.94, color=C[a], label=name)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.15, str(int(y)), ha="center", fontsize=7.5, color=figstyle.INK)
ax.axvline(6.5, color=figstyle.INK_2, lw=1.0, ls=":")
ax.set_xticks(range(len(CELLS))); ax.set_xticklabels([f"{s.capitalize()}\nL-{l}" for s, l in CELLS], fontsize=8)
ax.set_ylim(0, 17); ax.set_ylabel("safe successes / 16")
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("(c) safe successes per cell; Spatial L-I is outside the 112 ep\n(pkgFrs Goal init 15,16 = AMRC written values)", fontsize=9.5)
ax = fig.add_subplot(gs[1, 20:24]); axes.append(ax)
sh = json.load(open(shape_json))
GROUPS = [("grasped, released, not completed", lambda k: k.startswith("쥐었다")),
          ("no grasp: QP persists, hand / finger rows", lambda k: k.startswith("못 쥠: 쥐기 전 QP") and "손" in k),
          ("no grasp: QP persists, arm-link rows", lambda k: k.startswith("못 쥠: 쥐기 전 QP") and "팔" in k),
          ("no grasp: few QP calls", lambda k: k.startswith("못 쥠: QP 적음")),
          ("still holding at the end", lambda k: k.startswith("쥔 채")),
          ("HOLD to the end", lambda k: k.startswith("끝까지")),
          ("success + collision", lambda k: k.startswith("성공"))]
vals = [sum(v for k, v in sh["classes"].items() if g(k)) for _, g in GROUPS]
assert sum(vals) == sh["n"] - sh["safe_success"], (vals, sh["n"], sh["safe_success"])
ys = np.arange(len(GROUPS))[::-1]
ax.barh(ys, vals, 0.62, color=C[0])
for y, v in zip(ys, vals):
    ax.text(v + 0.4, y, str(v), va="center", fontsize=8, color=figstyle.INK)
ax.set_yticks(ys); ax.set_yticklabels([g for g, _ in GROUPS], fontsize=8)
ax.set_xlim(0, max(vals) + 5); ax.set_xlabel("episodes")
ax.set_title(f"(d) form of the {sh['n'] - sh['safe_success']} pkgFrs episodes that are not\na safe success ({sh['n']} ep measured on SDM)", fontsize=9.5)
figstyle.style_axes(fig, axes)
p = pooled["pkgFrs"]
fig.suptitle(f"SDM, pkgFrs on init 15-18 (not used for tuning; not the report split): other suites TSR {p[1]} / CAR {p[2]} / SSR {p[3]} of 112;  "
             f"Spatial L-I {' / '.join(str(int(x)) for x in per[('spatial', 'I')]['pkgFrs'][1:])} of 16", fontsize=12, y=0.965)
fig.savefig(out, dpi=140, bbox_inches="tight")
json.dump({"figure": out, "set": "SafeLIBERO Object/Goal/Long x L-I/L-II + Spatial L-II (7 cells) and Spatial L-I, task 0-3 x init 15-18, seed 7, main_aegis",
           "counts_are": "n, TSR, CAR, SSR", "pooled_7_cells_112ep": {a: [int(x) for x in v] for a, v in pooled.items()},
           "per_cell_16ep": {f"{s} L-{l}": {a: [int(x) for x in v] for a, v in per[(s, l)].items()} for s, l in CELLS},
           "pkgFrs_goal_init15_16_written_values": {"source": "plan §9.22 (AMRC; episode records not on SDM storage)",
                                                    **{f"{s} L-{l}": list(v) for (s, l), v in AMRC_GOAL.items()}},
           "pkgFrs_init17_18_56ep": [int(x) for x in sum(cnt(FRS[c], [k for k in KEYS if k[1] in (17, 18)]) for c in MS)],
           "pkgFrs_over_50mm": {f"{s} L-{l}": sorted(f"t{t}e{i}:{v[2]:.0f}mm" for (t, i), v in FRS[(s, l)].items() if v[2] > 50) for s, l in CELLS},
           "shape_not_safe_success": dict(zip([g for g, _ in GROUPS], vals)), "shape_source": shape_json,
           "scene_frames": meta, "korean_font": bool(ko)}, open(out[:-4] + ".json", "w"), indent=1, ensure_ascii=False)
print({a: [int(x) for x in v] for a, v in pooled.items()})
print({f"{s} L-{l}": [int(x) for x in per[(s, l)]["pkgFrs"]] for s, l in CELLS})
