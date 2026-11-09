"""Figure: pe123458-tro vs pe123458 (dev-4a, 2026-11-09 15:3x KST). (a) protocol counts, (b) E1a collision size per episode."""
import glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); B = "/mnt/dev/AG3S_VLA-storage/results/benchmark"
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
r = json.load(open(f"{HERE}/tro_compare.json"))
TRO, BASE = "#2a78d6", "#9a9a9a"
items = [("Cal LIBERO\n(18)", 13, 14), ("LIBERO 27", 23, 22),
         ("SafeLIBERO 11\nsafe succ", r["SafeLIBERO 11"]["tro"]["SSR"], r["SafeLIBERO 11"]["base"]["SSR"]),
         ("LIBERO-Safety\n10 succ", r["LIBERO-Safety 10"]["tro"]["SSR"], r["LIBERO-Safety 10"]["base"]["SSR"]),
         ("E1a 40 SSR\n(dev split)", r["E1a 40 (init 5-14)"]["tro"]["SSR"], r["E1a 40 (init 5-14)"]["base"]["SSR"]),
         ("E1a 40 CAR", r["E1a 40 (init 5-14)"]["tro"]["CAR"], r["E1a 40 (init 5-14)"]["base"]["CAR"])]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
x = range(len(items)); w = 0.38
for i, (lab, t, b) in enumerate(items):
    ax[0].bar(i - w / 2 - 0.01, t, w, color=TRO); ax[0].bar(i + w / 2 + 0.01, b, w, color=BASE)
    ax[0].text(i - w / 2, t + 0.3, str(t), ha="center", fontsize=9); ax[0].text(i + w / 2, b + 0.3, str(b), ha="center", fontsize=9)
ax[0].set_xticks(list(x)); ax[0].set_xticklabels([i[0] for i in items], fontsize=8.5); ax[0].set_ylabel("episodes")
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(color=TRO, label="pe123458-tro"), Patch(color=BASE, label="pe123458 (current)")], frameon=False, fontsize=9); ax[0].set_title("(a) same protocol, HEAD 827bc5e3 / a811331c", fontsize=10)
for s in ("top", "right"): ax[0].spines[s].set_visible(False)
for k, (pat, col, lab) in enumerate([("qe-pe123458tro-e1a", TRO, "tro"), ("qe-pe123458-e1a", BASE, "current")]):
    ds = {}
    for d in sorted(glob.glob(f"{B}/2026*_{pat}-*")): ds[d.split("_", 2)[2].split("-vla")[0]] = d
    disp = sorted(1000 * json.loads(l)["safelibero"]["max_displacement_m"] for d in ds.values() for l in open(d + "/episodes.jsonl")
                  if json.loads(l)["safelibero"].get("collided"))
    ax[1].plot(range(1, len(disp) + 1), disp, "o-", color=col, lw=2, ms=8, label=f"{lab}: {len(disp)} collided ep")
ax[1].axhline(10, color="#555", lw=1, ls="--"); ax[1].text(0.6, 11.5, "10 mm (tau_obs)", fontsize=8, color="#555")
ax[1].set_yscale("log"); ax[1].set_xlabel("collided episodes, sorted"); ax[1].set_ylabel("obstacle max displacement (mm)")
ax[1].legend(frameon=False, fontsize=9); ax[1].set_title("(b) E1a 40 collision size", fontsize=10)
for s in ("top", "right"): ax[1].spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{HERE}/tro_compare.png", dpi=130); print("saved")
