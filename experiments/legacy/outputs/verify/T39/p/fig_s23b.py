"""T39 S2-3b figure: t39-s23b-divergence.png (+ .json). left: per pair, chunk 1-5, a_exec / b (log y), c at chunk 2-3;
right: a_exec over all E3b chunks by latch state and by gate action. usage: fig_s23b.py S23B_JSON OUT_DIR"""
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-o1")
from AG3S.ag3s.experiments.common import figstyle as fs  # noqa: E402
fs.use_korean()
R = json.loads(pathlib.Path(sys.argv[1]).read_text()); F = pathlib.Path(sys.argv[2])
p = F / "t39-s23b-divergence.png"; assert not p.exists()
C = fs.CATEGORICAL; FLOOR = 1e-9
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
ax = axes[0]; side = {"floor_for_log_plot": FLOOR, "series": {}}
for i, (key, lab) in enumerate((("a_exec", "a: E3b executed - E3b policy reference"), ("b", "b: E3b policy reference - E0 executed"),
                                ("c_state16", "c: state diff at chunk start (16D)"))):
    for k in range(1, 6):
        v = [e["chunks"][str(k)][key] for e in R["per_pair"] if key in e["chunks"][str(k)]]
        if not v:
            continue
        x = k + (i - 1) * 0.25 + np.random.default_rng(k + 10 * i).uniform(-0.07, 0.07, len(v))
        ax.scatter(x, np.maximum(v, FLOOR), s=8, color=C[i], alpha=0.7, label=lab if k == 2 else None, edgecolor="none")
        side["series"][f"{key}_chunk{k}"] = {"median": float(np.median(v)), "max": float(max(v)), "n": len(v)}
ax.set_yscale("log"); ax.axhline(1e-3, color=fs.INK, ls="--", lw=0.8); ax.text(5.45, 1e-3, "1e-3", fontsize=7, va="center")
ax.axhline(1e-4, color=fs.INK_2, ls=":", lw=0.8); ax.text(5.45, 1e-4, "1e-4", fontsize=7, va="center")
ax.set_xticks(range(1, 6)); ax.set_xlabel("chunk seq (8 control steps each)"); ax.set_ylabel("max abs (rad / gripper cmd), 48 pairs")
ax.legend(fontsize=7.5, frameon=False, loc="lower right"); ax.set_title(f"per pair, chunks 1-5 (values < {FLOOR:g} drawn at {FLOOR:g})", fontsize=9, color=fs.INK)
ax = axes[1]
groups = [("latch " + k, v) for k, v in R["a_exec_all_E3b_chunks"]["by_latch_state"].items()] + \
         [("gate " + k, v) for k, v in R["a_exec_all_E3b_chunks"]["by_gate_action"].items()]
ys = np.arange(len(groups))
for j, (lab, st) in enumerate(groups):
    ax.plot([st["min"], st["max"]], [j, j], color=fs.GRID_INK, lw=1)
    ax.scatter([max(st["median"], FLOOR)], [j], color=C[0] if lab.startswith("latch") else C[1], s=30, zorder=3)
    ax.scatter([st["p95"]], [j], color=C[0] if lab.startswith("latch") else C[1], marker="|", s=80, zorder=3)
    ax.text(st["max"] * 1.1, j, f"n={st['n']}", fontsize=7, va="center", color=fs.INK_2)
ax.set_xscale("log"); ax.set_yticks(ys); ax.set_yticklabels([g[0] for g in groups], fontsize=8)
ax.set_xlabel("a: E3b executed - policy reference, per chunk (dot median, | p95, line min-max)")
ax.set_title("all E3b chunks by latch state and gate action", fontsize=9, color=fs.INK)
side["groups"] = {g[0]: g[1] for g in groups}
fs.style_axes(fig, axes); fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
p.with_suffix(".json").write_text(json.dumps(side, indent=1)); print(p)
