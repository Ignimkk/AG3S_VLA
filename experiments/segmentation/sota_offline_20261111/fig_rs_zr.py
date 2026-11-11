#!/usr/bin/env python3
"""(2026-11-12 dev-57) Rule-A figure for plan §9.20-§9.22: the two rule-level faults found in pkgFin's E1a failures and the
candidates SL-SOTA11 (-rs) / SL-SOTA12 (-zr60).  (a) real scene: E1a task 0 init 13 at the hand-box contact, pkgFin vs the
60 mm zone; (b) task 0 init 11 grasp-latch timeline, pkgFin vs -rs; (c) E1a safe successes per task; (d) totals.
usage: OMP_NUM_THREADS=1 .venv-ag3s/bin/python outputs/2026-11-11_SOTA_개선/fig_rs_zr.py OUT.png"""
import ast, glob, json, sys
import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.experiments.common import figstyle

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
out = sys.argv[1]


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    return x or {}


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


def frame(tag, init, k):
    d = run(tag, 0)
    f = glob.glob(d + f"/videos/*t0_e{init}_*.mp4")[0]
    rd = imageio.get_reader(f)
    n = rd.count_frames()
    return np.asarray(rd.get_data(min(k, n - 1))), n


def latch(tag, init):
    d = run(tag, 0)
    e = [x for x in map(json.loads, open(d + "/episodes.jsonl")) if x["init_state_index"] == init][0]
    cs = [json.loads(x) for x in open(f"{d}/episodes/safelibero_spatial_t0_e{e['episode_idx']}_calls.jsonl")]
    rows = []
    for c in cs:
        g = c.get("grasp") or {}
        ev = lit(g.get("evidence"))
        rows.append((c["seq"], ev.get("opening_after"), ev.get("reach_mm"), g.get("state")))
    return rows


ko = figstyle.use_korean()
C = figstyle.CATEGORICAL
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
ax_a1, ax_a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_b = fig.add_subplot(gs[0, 2:])
ax_c = fig.add_subplot(gs[1, :3])
ax_d = fig.add_subplot(gs[1, 3])
figstyle.style_axes(fig, [ax_b, ax_c])
# (a) the contact frame: obs 60-62 = video frame ~41 (20 wait steps)
K = 42
for ax, tag, title in ((ax_a1, "sub-pkgFin", "pkgFin (zone 100 mm): hand touches the box, 12.1 mm"),
                       (ax_a2, "sub-pkgSrz", "zone 60 mm (-rs-zr60): no contact")):
    img, n = frame(tag, 13, K)
    ax.imshow(img)
    ax.set_title(title, fontsize=9, color=figstyle.INK)
    ax.set_xticks([]); ax.set_yticks([])
ax_a1.set_ylabel("(a) E1a task 0 init 13, env step %d" % K, fontsize=9, color=figstyle.INK)
# (b) latch timeline
side = {}
for j, (tag, name) in enumerate((("sub-pkgFin", "pkgFin"), ("sub-pkgSrs", "-rs"))):
    rows = latch(tag, 11)
    seq = [r[0] for r in rows]
    reach = [np.nan if r[2] is None else r[2] for r in rows]
    ax_b.plot(seq, reach, color=C[j], lw=1.8, label=f"reach mm ({name})")
    held = [r[0] for r in rows if r[3] == "held"]
    if held:
        ax_b.axvline(held[0], color=C[j], ls=":", lw=1.5)
        ax_b.text(held[0] + 0.3, 205, f"{name}: HELD at call {held[0]}", fontsize=8, color=C[j])
    side[name] = {"calls": len(rows), "first_held_call": held[0] if held else None,
                  "reach_mm": [None if np.isnan(x) else x for x in reach],
                  "opening": [r[1] for r in rows], "state": [r[3] for r in rows]}
ax_b.axhline(120, color=figstyle.INK, ls="--", lw=1.2)
ax_b.text(0.3, 123, "reach limit 120 mm", fontsize=8, color=figstyle.INK)
ax2 = ax_b.twinx()
rows = latch("sub-pkgFin", 11)
ax2.plot([r[0] for r in rows], [np.nan if r[1] is None else r[1] for r in rows], color=figstyle.INK_2, lw=1.0, alpha=0.8)
ax2.set_ylabel("gripper opening (pkgFin, grey)", fontsize=8, color=figstyle.INK_2)
ax2.set_ylim(0, 1.05)
ax_b.set_ylim(0, 220); ax_b.set_xlim(0, 60)
ax_b.set_xlabel("policy call"); ax_b.set_ylabel("reach to the manipulated geometry (mm)")
ax_b.set_title("(b) task 0 init 11: reach passes while closing (calls 15-18), fails once the\nopening settles (19) -> pkgFin stays CLOSING to the end; -rs attaches at 19", fontsize=9, loc="left")
ax_b.legend(fontsize=8, frameon=False, loc="center right")
# (c) per-task SSR
arms = [("current config", ["qe-pe123458"]), ("-qs", ["sub-pkgFqs"]), ("pkgFin (-hb-qs)", ["sub-pkgFin"]),
        ("pkgFin2 (-rs-zr60)", ["sub-pkgFin2", "sub-pkgSrz"]), ("pkgFrs (-rs): t0 measured, t1-3 = pkgFin", ["sub-pkgSrs", "sub-pkgFin"])]
R = [e1a(t) for _, t in arms]
w = 0.8 / len(arms)
tot = []
for a, ((name, _), r) in enumerate(zip(arms, R)):
    per = [sum(1 for k, v in r.items() if k[0] == t and v == "SS") for t in range(4)]
    tsr = sum(v in ("SS", "S+C") for v in r.values()); car = sum(v in ("SS", "F") for v in r.values()); ssr = sum(v == "SS" for v in r.values())
    tot.append((name, len(r), tsr, car, ssr, per))
    xs = np.arange(5) + (a - (len(arms) - 1) / 2) * w
    vals = per + [ssr / 4.0]
    ax_c.bar(xs, vals, w * 0.92, color=C[a], label=name, hatch="//" if a == 4 else None, edgecolor=figstyle.SURFACE)
    for x, v, lab in zip(xs, vals, per + [ssr]):
        ax_c.text(x, v + 0.12, str(lab), ha="center", fontsize=8, color=figstyle.INK)
ax_c.set_xticks(range(5)); ax_c.set_xticklabels(["task 0", "task 1", "task 2", "task 3", "all 40 (bar = count / 4)"])
ax_c.set_ylim(0, 11.5); ax_c.set_ylabel("safe successes / 10")
ax_c.axhline(5.3, color=figstyle.INK, ls="--", lw=1.0); ax_c.text(4.42, 5.42, "KNOWS 53.0 %", fontsize=7.5, color=figstyle.INK)
ax_c.set_title("(c) E1a 40 safe successes per task (SafeLIBERO Spatial L-I, init 5-14, seed 7)", fontsize=9, loc="left")
ax_c.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
# (d) totals table
ax_d.axis("off")
cell = [[n.split(" (")[0].split(":")[0], str(t), str(c), str(s)] for n, _, t, c, s, _ in tot]
tb = ax_d.table(cellText=cell, colLabels=["E1a 40", "TSR", "CAR", "SSR"], loc="center", cellLoc="center", colWidths=[0.46, 0.18, 0.18, 0.18])
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1.0, 1.6)
ax_d.set_title("(d) totals (episodes of 40)", fontsize=9, loc="left", color=figstyle.INK)
fig.suptitle("Two rule-level faults in the remaining E1a failures: grasp latch stuck on reach (-rs) and the 100 mm grasp-zone release (-zr60)",
             fontsize=11.5, color=figstyle.INK)
fig.tight_layout()
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
json.dump({"scene": {"episode": "E1a task 0 init 13", "video_frame": K, "runs": ["sub-pkgFin", "sub-pkgSrz"]},
           "latch_t0_e11": side, "reach_limit_mm": 120.0,
           "e1a": [{"label": n, "n": k, "TSR": t, "CAR": c, "SSR": s, "SSR_per_task": per} for n, k, t, c, s, per in tot],
           "note": "pkgFrs: task 0 measured (sub-pkgSrs), tasks 1-3 are pkgFin's runs (the -rs rule does not fire there in the offline latch replay)",
           "reference": {"KNOWS Spatial L-I reported SSR %": 53.0}}, open(out.rsplit(".", 1)[0] + ".json", "w"), ensure_ascii=False, indent=1)
print(out, tot, "korean", ko)
