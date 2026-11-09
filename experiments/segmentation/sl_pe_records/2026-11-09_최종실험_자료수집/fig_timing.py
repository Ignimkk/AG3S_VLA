"""(2026-11-09 12:4x dev-4a) Figure for the Tab. S11 draft: client-boundary latency p50 (bar) / p95 (tick) per E1a arm, and the
AG3S noGT server split. Source timing_summary.json (DERIVED from existing runs)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open("timing_summary.json"))
arms = ["pi0.5 only", "AEGIS port (detector)", "multi-link CBF (detector)", "KNOWS port (detector)", "AG3S B5 (GT)",
        "AG3S B5-tro (GT)", "AG3S noGT pe123458"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
p50 = [r[a]["t_client"]["p50"] for a in arms]
p95 = [r[a]["t_client"]["p95"] for a in arms]
cols = ["#8d99ae"] * 4 + ["#e9c46a", "#e9c46a", "#2a9d8f"]
ax[0].barh(range(len(arms)), p50, color=cols, edgecolor="white")
ax[0].scatter(p95, range(len(arms)), marker="|", s=300, color="#333")
for i, (a, b) in enumerate(zip(p50, p95)):
    ax[0].text(b + 60, i, f"{a:.0f} / {b:.0f} ms", va="center", fontsize=8)
ax[0].set_yticks(range(len(arms)), arms, fontsize=8)
ax[0].invert_yaxis()
ax[0].set_xlabel("request -> returned chunk (ms), bar p50, tick p95")
ax[0].set_xlim(0, 4600)
ax[0].set_title("E1a Spatial LI, client boundary (incl. policy)", fontsize=10)
s = r["AG3S noGT pe123458"]
st = r["AG3S noGT pe123458 stages (profile_ms)"]
parts = [("policy + attention (infer)", s["infer"]), ("scene reconstruction", st["scene_reconstruction"]),
         ("target grounding", st["target_grounding"]), ("ESDF", st["esdf"]), ("constraint generation", st["constraint_generation"]),
         ("trajectory refinement", s["trajopt"])]
ax[1].barh(range(len(parts)), [p[1]["p50"] for p in parts], color="#2a9d8f", edgecolor="white")
ax[1].scatter([p[1]["p95"] for p in parts], range(len(parts)), marker="|", s=300, color="#333")
for i, (_, v) in enumerate(parts):
    ax[1].text(v["p95"] + 25, i, f"{v['p50']:.0f} / {v['p95']:.0f}", va="center", fontsize=8)
ax[1].set_yticks(range(len(parts)), [p[0] for p in parts], fontsize=8)
ax[1].invert_yaxis()
ax[1].set_xlim(0, 1700)
ax[1].set_xlabel("ms (p50 bar / p95 tick)")
ax[1].set_title("AG3S noGT (pe123458) server stages", fontsize=10)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("timing_e1a.png", dpi=150)
