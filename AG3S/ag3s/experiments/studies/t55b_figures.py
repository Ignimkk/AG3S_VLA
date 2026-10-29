"""Figures for T55b (#48) and the T54c P · H add-on — read the verify.json numbers and the T55 re-sim images only.

usage: t55b_figures.py FIG_ROOT      (.venv-ag3s; writes FIG_ROOT/t55b/* and FIG_ROOT/t54c/t54c-windows-*)
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

HANDOFF = ROOT / "AG3S/ag3s/docs/handoff"
SCENE = ("ep1804", "instr_banana", "instr_orange")  # same episode: HELD lost (banana) vs HELD kept (orange)


def _save(fig, path: pathlib.Path, numbers: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SystemExit(f"{path} exists — pick a new name (figures rule)")
    fig.savefig(path, dpi=150, facecolor=figstyle.SURFACE)
    path.with_suffix(".json").write_text(json.dumps(numbers, indent=1, default=float))
    plt.close(fig)
    print("->", path)


def t55b_graph(root: pathlib.Path) -> None:
    v = json.loads((HANDOFF / "T55b.verify.json").read_text())["numbers"]
    ins = v["by_condition"]["instruction"]
    att = v["attach_manipulated_is"]["instruction"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    rows = [("reach_mm (latch, HELD missed)", ins["held_false_reach_mm"]),
            ("hand → held fruit (MuJoCo, HELD missed)", ins["held_false_held_fruit_mm"]),
            ("reach_mm (latch, HELD kept)", ins["held_true_reach_mm"])]
    for i, (lab, q) in enumerate(rows):
        ax.plot([q["min"], q["max"]], [i, i], color=figstyle.SEQ_BLUE[4], lw=3)
        ax.plot([q["median"]], [i], "o", color=figstyle.CATEGORICAL[i], ms=8)
        ax.text(q["max"] + 10, i, f"median {q['median']:.0f} (n {q['n']})", va="center", fontsize=8)
    lim = ins["reach_limit_mm"][0]
    ax.axvline(lim, color=figstyle.CATEGORICAL[7], ls="--")
    ax.text(lim + 5, 2.4, f"reach limit {lim:.0f} mm", color=figstyle.CATEGORICAL[7], fontsize=8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.set_xlabel("distance from the latch parent link (mm), min–max and median")
    ax.set_ylim(-0.6, 2.8)
    ax.set_title("Instruction changed: HELD missed frames vs reach limit")
    ax = axes[1]
    order = ["held", "noun", "other", "never_held", "no_p_grasp"]
    labels = ["attach: held fruit", "attach: instructed fruit", "attach: other fruit",
              "grasped, never HELD", "no grasp"]
    vals = [att.get(k, 0) for k in order]
    ax.barh(range(len(order)), vals, color=[figstyle.CATEGORICAL[i] for i in range(len(order))])
    for i, n in enumerate(vals):
        ax.text(n + 0.3, i, str(n), va="center", fontsize=8)
    ax.set_yticks(range(len(order)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("instruction traces (test E0 × 3 prompts)")
    ax.set_title("Manipulated object at the attach request")
    figstyle.style_axes(fig, axes)
    fig.tight_layout()
    _save(fig, root / "t55b/t55b-held-fn-reach.png",
          {"source": "T55b.verify.json", "instruction": {k: ins[k] for k in (
              "held_false_reach_mm", "held_false_held_fruit_mm", "held_true_reach_mm", "reach_limit_mm",
              "held_false_by_cause", "n_p_grasp", "n_held_false")}, "attach_manipulated_is": att})


def t55b_scene(root: pathlib.Path) -> None:
    v = json.loads((HANDOFF / "T55b.verify.json").read_text())
    t55 = pathlib.Path(v["inputs"]["t55_dir"] if "inputs" in v else v["numbers"]["inputs"]["t55_dir"])
    key, bad, good = SCENE
    z = np.load(t55 / f"resim/test/{key}__E0.npz", allow_pickle=False)
    fruits = [str(f) for f in z["fruits"]]
    qb = np.load(t55 / f"perception/test/t55-legacy/{key}__E0__{bad}.npz", allow_pickle=False)
    qg = np.load(t55 / f"perception/test/t55-legacy/{key}__E0__{good}.npz", allow_pickle=False)
    held_any = np.array([any(str(z["phys_state"][i, j]) == "held" for j in range(len(fruits)))
                         for i in range(len(z["phys_state"]))])
    f = int(np.argmax(held_any & (qb["latch_reground__phase"] != "held")))
    nums = {"key": key, "frame": f, "t_step": int(z["t_step"][f]), "fruits": fruits, "panels": {}}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, cam in zip(axes, ("image_cam_high", "image_cam_left_wrist", "image_cam_right_wrist")):
        ax.imshow(z[cam][f])
        ax.set_title(cam.replace("image_", ""), fontsize=9)
        ax.axis("off")
    rows = []
    for tag, q in ((bad, qb), (good, qg)):
        g = json.loads(str(q["grasp_json"][f])) if str(q["grasp_json"][f]) else {}
        ev = g.get("evidence") or {}
        c = np.asarray(q["target_centroid"][f], float)
        near = fruits[int(np.argmin(np.linalg.norm(z["fruit_pos"][f] - c, axis=1)))] if np.all(np.isfinite(c)) else None
        rows.append(f"prompt noun={tag.split('_')[1]}: latch={q['latch_reground__phase'][f]}, manipulated≈{near}, "
                    f"reach={ev.get('reach_mm')} mm, failed={[k for k, x in (ev.get('checks') or {}).items() if x is False]}")
        nums["panels"][tag] = {"latch": str(q["latch_reground__phase"][f]), "manipulated_nearest": near,
                               "reach_mm": ev.get("reach_mm"), "checks": ev.get("checks")}
    held = [fr for j, fr in enumerate(fruits) if str(z["phys_state"][f, j]) == "held"]
    nums["held_fruit_truth"] = held
    fig.suptitle(f"{key} frame {f} (same images, two instructions) — fruit in hand (MuJoCo): {held}", fontsize=10)
    fig.text(0.01, 0.02, "\n".join(rows), fontsize=8, family="monospace")
    fig.patch.set_facecolor(figstyle.SURFACE)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    _save(fig, root / "t55b/t55b-scene-instruction.png", nums)


def t54c_windows(root: pathlib.Path) -> None:
    v = json.loads((HANDOFF / "T54c_windows.verify.json").read_text())["numbers"]
    g = v["grounding"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ws = ("full", "P", "H")
    nums = {"identity_diff_vs_L8H2_pp": {}, "closed_loop": {}}
    ax = axes[0]
    for ci, (cond, arm) in enumerate((("current", "N"), ("current", "A"), ("hold_attach", "N"), ("hold_attach", "A"))):
        xs, ys, lo, hi = [], [], [], []
        for i, w in enumerate(ws):
            r = g[w]["summary"][cond]["metrics"]["identity"][arm]
            xs.append(i + (ci - 1.5) * 0.12)
            ys.append(100 * r["diff_vs_L8H2"])
            lo.append(100 * r["diff_ci95_episode_bootstrap"][0])
            hi.append(100 * r["diff_ci95_episode_bootstrap"][1])
        ys_, lo_, hi_ = np.array(ys), np.array(lo), np.array(hi)
        ax.errorbar(xs, ys_, yerr=[ys_ - lo_, hi_ - ys_], fmt="o", capsize=3, color=figstyle.CATEGORICAL[ci],
                    label=f"{arm} − L8H2, {cond}")
        nums["identity_diff_vs_L8H2_pp"][f"{arm}|{cond}"] = {w: [y, l, h] for w, y, l, h in zip(ws, ys, lo, hi)}
    ax.axhline(0, color=figstyle.GRID_INK)
    ax.set_xticks(range(3), ws)
    ax.set_ylabel("identity difference (%p), episode bootstrap 95% CI")
    ax.set_title("T54c offline grounding replay by window")
    ax.legend(fontsize=7)
    ax = axes[1]
    cl = v["closed_loop_C1a"]
    keys = ("success", "success_ct", "hold_chunks", "hold_runs", "completion_s_median")
    cell = [[("-" if cl[w].get(k) is None else (f"{cl[w][k]:.1f}" if isinstance(cl[w][k], float) else str(cl[w][k])))
             for k in keys] for w in ws]
    ax.axis("off")
    t = ax.table(cellText=cell, rowLabels=[f"{w} window" for w in ws], colLabels=list(keys), loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1, 1.6)
    ax.set_title(f"T54c serving record (C1a, 16 runs; P reached {v['endpoints']['P_reached']}, "
                 f"H reached {v['endpoints']['H_reached']})", fontsize=9)
    nums["closed_loop"] = {w: {k: cl[w].get(k) for k in keys} for w in ws}
    nums["endpoints"] = {k: v["endpoints"][k] for k in ("n_runs", "P_reached", "H_reached",
                                                          "frames_removed_P", "frames_removed_H")}
    figstyle.style_axes(fig, [axes[0]])
    fig.tight_layout()
    _save(fig, root / "t54c/t54c-windows-ph.png", nums)


if __name__ == "__main__":
    r = pathlib.Path(sys.argv[1])
    t55b_graph(r)
    t55b_scene(r)
    t54c_windows(r)
