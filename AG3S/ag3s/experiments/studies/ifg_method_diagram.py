"""Pipeline diagram: AG3S target grounding before this study, the original plan, and the current dev candidate.
  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_method_diagram --out AG3S/ag3s/docs/figures/ifg
"""
import argparse, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

ROWS = [
    ("Before (serving AG3S, B0)", [
        ("VLA attention map\n(16x16 per camera)", 7), ("lift to 3D points\n(1 cm voxels)", 7), ("robot removal:\nFK sphere volumes", 7),
        ("seeds: top 5% points\ngrow 3 cm + DBSCAN", 7), ("pick cluster by\nattention score", 7), ("target cluster + ball\n-> contact exemption", 7)]),
    ("Original plan (A+F+3D)", [
        ("noun attention\ntop-5 patch seeds", 1), ("VLA internal feature\nprototype (cosine)", 1), ("depth surface graph\n(5 mm nodes)", 1),
        ("feature in edges\n+ feature evidence", 1), ("energy minimisation\n(continuous)", 1), ("target surface", 1)]),
    ("Now (dev candidate)", [
        ("noun attention seeds\n(table pixels excluded)", 0), ("depth surface graph\n(edges only along\ncontinuous depth)", 0),
        ("robot removal: rendered\nrobot depth match", 0), ("energy: seed=1, table=0\n-> keep top seed region", 0),
        ("safety gate: feature\nmemory agreement", 2), ("phase / held from\nHSR + gripper + 3D", 2)]),
    ("Next (planned)", [
        ("held-out episodes\n(locked settings)", 6), ("region -> exclusion\ngeometry / permission", 6), ("gate -> keep uncertain\nregion as obstacle", 6),
        ("phase-dependent link\ncontact constraints", 6), ("closed-loop + occlusion\ntest (separate)", 6), ("strong baselines:\nGrounded SAM / GT mask", 6)]),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(17, 8.5)); figstyle.style_axes(fig, ax); ax.axis("off")
    ax.set_xlim(0, 6.6); ax.set_ylim(-0.2, len(ROWS))
    for r, (name, boxes) in enumerate(ROWS):
        y = len(ROWS) - 1 - r
        ax.text(0.02, y + 0.42, name, fontsize=10, fontweight="bold", va="center")
        for i, (txt, col) in enumerate(boxes):
            x = 0.05 + i * 1.08
            ax.add_patch(FancyBboxPatch((x, y), 0.98, 0.32, boxstyle="round,pad=0.02", fc=figstyle.CATEGORICAL[col], alpha=0.18,
                                        ec=figstyle.CATEGORICAL[col]))
            ax.text(x + 0.49, y + 0.16, txt, ha="center", va="center", fontsize=8)
            if i < len(boxes) - 1:
                ax.annotate("", xy=(x + 1.06, y + 0.16), xytext=(x + 0.99, y + 0.16), arrowprops=dict(arrowstyle="->", color=figstyle.INK_2))
    ax.text(0.05, -0.15, "red = existing serving; orange = original feature-graph plan (rejected on dev); blue = current candidate; "
            "teal = feature / 3D reuse kept (safety gate, phase); purple = planned. All numbers are dev-only (no held-out yet).", fontsize=8)
    fig.tight_layout(); fig.savefig(a.out / "ifg-method-evolution.png", dpi=150); plt.close(fig)
    (a.out / "ifg-method-evolution.json").write_text(json.dumps({n: [b[0] for b in bx] for n, bx in ROWS}, indent=1))
    print("ok")


if __name__ == "__main__":
    main()
