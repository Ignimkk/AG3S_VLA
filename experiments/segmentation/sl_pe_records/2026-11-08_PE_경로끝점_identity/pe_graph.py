"""Rule A graph (2026-11-08, dev-30): per group of gt24att episodes, on the pre-held calls whose chunk has a close command,
(left) share of calls whose identity majority is the target, (right) share whose scene obstacle set holds >= 3 target voxels.
R0 = recorded attention grounding (calls where it was recorded), PE = SL-PE1 cut under the grasp point (same calls).
Writes pe_graph.png + pe_graph.json."""
import glob, json, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.experiments.common import figstyle
HERE = pathlib.Path(__file__).resolve().parent
rows = []
for f in sorted(HERE.glob("replay_*.json")):
    if "summary" in f.name: continue
    for st, e in json.loads(f.read_text()).items():
        for r in e["rows"]:
            r["_g"] = ("SafeLIBERO" if "safelibero" in f.name else "LIBERO") + (" success" if e["success"] else " failed")
            rows.append(r)
groups = ["LIBERO success", "LIBERO failed", "SafeLIBERO success", "SafeLIBERO failed"]
out = {}
for g in groups:
    rr = [r for r in rows if r["_g"] == g and r["plan_close"] and r.get("R0_recorded")]
    out[g] = {"n_calls": len(rr),
              "acc": {k: float(np.mean([r[k].get("majority") == "target" for r in rr])) for k in ("R0", "PE")},
              "m4": {k: float(np.mean([r[k].get("M4", {}).get("target", 0) >= 3 for r in rr])) for k in ("R0", "PE")}}
figstyle.use_korean()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
figstyle.style_axes(fig, axes)
C = figstyle.CATEGORICAL
x = np.arange(len(groups)); w = 0.36
for ax, key, title in ((axes[0], "acc", "identity = target (share of calls)"),
                       (axes[1], "m4", "target left in the scene obstacle set (share of calls)")):
    for k, (rule, col, lab) in enumerate((("R0", C[1], "R0 recorded attention grounding"), ("PE", C[0], "SL-PE1 grasp-point cut"))):
        v = [100 * out[g][key][rule] for g in groups]
        bars = ax.bar(x + (k - 0.5) * w, v, w - 0.02, color=col, label=lab)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.0f}", ha="center", fontsize=8, color=figstyle.INK)
    ax.set_xticks(x, [f"{g}\n(n={out[g]['n_calls']})" for g in groups], fontsize=8)
    ax.set_ylim(0, 110); ax.set_ylabel("%"); ax.set_title(title, fontsize=9)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False)
fig.suptitle("gt24att records, pre-held calls with a close command in the chunk (LIBERO 27 + SafeLIBERO 9 ep)", fontsize=10)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(HERE / "pe_graph.png", dpi=150)
(HERE / "pe_graph.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
