"""(2026-11-09, dev-4a) figure: per version, the calls whose executed action AG3S changed or HOLD-ed, by latch phase and by the
GT owner of the refusing obstacle-object row (scoring only). Small multiples, one panel per version, LIBERO t3·4·7·9 x seeds.
usage: python fig_rowowner.py <out.png> <label>=<tag1,tag2,...> ..."""
import collections
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
PHASES = [("approach", "approach (pre-grasp)"), ("closing", "closing"), ("held_carry", "held: carry"), ("held_place", "held: place (< 12 cm of plate)")]
OWNERS = [("target", "target object (incl. the held bowl)", "#2a78d6"), ("destination", "destination (plate)", "#eb6834"),
          ("fixture", "fixture (cabinet / drawer / table)", "#1baf7a"), ("other", "other object", "#eda100"),
          ("uncert", "HOLD: geometry not certified", "#9a9994")]


def owner_key(r):
    if r["reason"] != "obstacle":
        return "uncert" if r["reason"] == "geometry not certified" else "other"
    o = r.get("owner")
    return o if o in ("target", "destination", "fixture") else "other"


def main():
    out = sys.argv[1]
    panels = []
    for arg in sys.argv[2:]:
        label, tags = arg.split("=", 1)
        cnt = collections.Counter()
        n_ep = n_ok = 0
        for t in tags.split(","):
            d = json.loads((HERE / f"rowowner_{t}.json").read_text())
            for v in d.values():
                n_ep += 1
                n_ok += v["success"]
                for r in v["rows"]:
                    if r["changed"] or r["hold"]:
                        cnt[(r["phase"], owner_key(r))] += 1
        panels.append((label, cnt, n_ep, n_ok))
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 10})
    fig, axes = plt.subplots(1, len(panels), figsize=(max(4.6 * len(panels), 9.5), 4.2), sharex=True, sharey=True)
    axes = [axes] if len(panels) == 1 else axes
    vmax = max(sum(c[(p, o)] for o, _, _ in OWNERS) for _, c, _, _ in panels for p, _ in PHASES)
    for ax, (label, cnt, n_ep, n_ok) in zip(axes, panels):
        ys = list(range(len(PHASES)))[::-1]
        for y, (p, _) in zip(ys, PHASES):
            left = 0
            for o, _, col in OWNERS:
                w = cnt[(p, o)]
                if w:
                    ax.barh(y, w - 0.15, left=left, height=0.55, color=col, edgecolor="#fcfcfb", linewidth=1)
                    left += w
            ax.text(left + 0.4, y, str(left), va="center", fontsize=9, color="#52514e")
        ax.set_yticks(ys, [n for _, n in PHASES])
        ax.set_title(f"{label}\nsuccess {n_ok} / {n_ep} ep", fontsize=10)
        ax.set_xlim(0, vmax * 1.15 + 1)
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("calls changed or HOLD by AG3S")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in OWNERS]
    fig.legend(handles, [n for _, n, _ in OWNERS], loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("What really is next to the refusing obstacle-object row (GT for scoring only) - LIBERO t3, t4, t7, t9", fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    print(out)


if __name__ == "__main__":
    main()
