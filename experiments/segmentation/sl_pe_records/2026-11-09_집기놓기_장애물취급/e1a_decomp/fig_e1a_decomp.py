"""(2026-11-09 12:1x dev-4a) Figure: E1a 40 outcome classes per arm + pe123458 collision mechanisms (from e1a_decomp.json,
passthrough_window / contact_geom / pe11_offline analyses)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("e1a_decomp.json"))
arms = [("gt24att", "noGT before\n(gt24att)"), ("pe123458", "noGT final\n(pe123458)"), ("gt23t100", "B5 GT"), ("gt23tro", "B5-tro GT")]
cls = [("SS", "safe success", "#2a9d8f"), ("S+C", "success + collision", "#e9c46a"), ("F+C", "fail + collision", "#e76f51"),
       ("F", "fail, no collision", "#8d99ae")]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
for i, (tg, lab) in enumerate(arms):
    b = 0
    for k, name, col in cls:
        n = sum(r["cls"] == k for r in d[tg].values())
        ax[0].bar(i, n, bottom=b, color=col, edgecolor="white", linewidth=2, label=name if i == 0 else None)
        if n:
            ax[0].text(i, b + n / 2, str(n), ha="center", va="center", fontsize=9, color="black")
        b += n
ax[0].axhline(23, color="#444", ls="--", lw=1)
ax[0].text(-0.45, 23.6, "CBF GT SSR 23", ha="left", fontsize=8, color="#444")
ax[0].set_xticks(range(len(arms)), [a[1] for a in arms], fontsize=9)
ax[0].set_ylabel("episodes (n = 40)")
ax[0].set_title("E1a SafeLIBERO Spatial LI: outcome per episode", fontsize=10)
ax[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=4, frameon=False)
mech = [("C: obstacle dropped from obstacle set\n(same component as target) -> PE11 keeps it", 6, "#e76f51"),
        ("A: passed through, obstacle-row\npenetration <= tau_obs 10 mm", 5, "#f4a261"),
        ("B: action QP said clear,\nreal motion still hit", 3, "#e9c46a"),
        ("D: obstacle not perceived\n(occluded while held / closing)", 2, "#8d99ae"),
        ("E: obstacle in set, arm link\nrows not obstacle rows (held)", 1, "#bcc3cf")]
y = list(range(len(mech)))[::-1]
ax[1].barh(y, [m[1] for m in mech], color=[m[2] for m in mech], edgecolor="white", linewidth=2)
for yy, m in zip(y, mech):
    ax[1].text(m[1] + 0.1, yy, str(m[1]), va="center", fontsize=9)
ax[1].set_yticks(y, [m[0] for m in mech], fontsize=8)
ax[1].set_xlabel("collided episodes of pe123458 (17)")
ax[1].set_title("pe123458 E1a: why the 17 collisions happened", fontsize=10)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("e1a_decomp.png", dpi=150)
