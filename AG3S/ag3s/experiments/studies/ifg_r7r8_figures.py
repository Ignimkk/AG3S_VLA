"""IFG R7 / R8 figures (+ JSON sidecars):
  ifg-r7-feature-revalidation.png  paired IoU / recall difference vs G-Q+top1 (R6 candidate) for every +top1 arm
  ifg-r8-gradient.png              gradient vs attention target-localisation metrics + one real frame (right wrist)
  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_r7r8_figures --run $RUN --out AG3S/ag3s/docs/figures/ifg
"""
import argparse, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_reader as R  # noqa: E402
from AG3S.ag3s.experiments.studies.ifg_grad_diag import gt_224, patch_up  # noqa: E402


def r7(run, out):
    comps = json.loads((run / "rounds/R7/paired_summary.json").read_text())["comparisons"]
    C = {(c["a"][0], c["metric"]): c for c in comps if c["b"][0] == "G-Q+top1"}
    order = []
    for m in ("FGe-Q", "FGu-Q", "FG-Q", "FG-R"):
        for f in R.FEATURES:
            order.append(f"{m}:{f}+top1")
    order += ["G-R+top1", "FG-Q-shuffle:prefix_L8+top1", "G-Q:O+top1", "FG-Q:prefix_L8:O+top1"]
    order = [a for a in order if (a, "iou") in C]
    col = lambda a: figstyle.CATEGORICAL[4 if "shuffle" in a else 2 if ":O" in a else {"FGe-Q": 6, "FGu-Q": 3, "FG-Q": 1, "FG-R": 5, "G-R+top1": 0}.get(a.split(":")[0], 0)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True); figstyle.style_axes(fig, axes)
    for ax, met, line in ((axes[0], "iou", 0.02), (axes[1], "recall", -0.02)):
        for i, a in enumerate(order):
            c = C[(a, met)]
            ax.errorbar(c["diff"], i, xerr=[[c["diff"] - c["ci95"][0]], [c["ci95"][1] - c["diff"]]], fmt="o", color=col(a), capsize=2)
        ax.axvline(0, color=figstyle.INK, lw=0.8); ax.axvline(line, color=figstyle.INK_2, lw=0.8, ls="--")
        ax.set_xlabel(f"paired episode {met} diff vs G-Q+top1 (R6 candidate), 95% CI")
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order, fontsize=8); axes[0].invert_yaxis()
    fig.suptitle("R7: attention + feature + 3D on the improved base (render mask, non-support seeds). Dashed: +0.02 effect / -0.02 recall constraint", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "ifg-r7-feature-revalidation.png", dpi=150); plt.close(fig)
    (out / "ifg-r7-feature-revalidation.json").write_text(json.dumps({a: {m: C[(a, m)] for m in ("iou", "recall")} for a in order}, indent=1))


def r8(run, out, fid="T53_ep1839_f023"):
    s = json.loads((run / "rounds/R8/diag/grad_summary.json").read_text())
    maps = ["N", "A", "grip_gx", "grip_g", "arm_gx", "arm_g"]
    ref = {r.frame_id: r for r in R.list_frames()}[fid]
    with np.load(ref.path / "grad.npz") as z:
        gg, ag = z["gripper_pixel_saliency"], z["arm_motion_pixel_saliency"]
    with np.load(ref.path / "derived.npz") as z:
        nup = patch_up(z["noun_target_L10H7"])
    with np.load(ref.path / "inputs.npz") as z:
        img = np.clip((z["image_right_wrist_0_rgb"] + 1) / 2, 0, 1)
    with np.load(run / "s0/labels" / f"{fid}.npz") as z:
        cls = gt_224(z["class_map"])
    fig = plt.figure(figsize=(18, 7.5)); gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15])
    ax0 = fig.add_subplot(gs[0, :2]); ax1 = fig.add_subplot(gs[0, 2:])
    figstyle.style_axes(fig, [ax0, ax1])
    x = np.arange(len(maps))
    ax0.bar(x - 0.2, [s[m]["r_precision"] for m in maps], 0.4, color=figstyle.CATEGORICAL[0], label="area-matched precision")
    ax0.bar(x + 0.2, [s[m]["hit1"] for m in maps], 0.4, color=figstyle.CATEGORICAL[1], label="argmax pixel on target")
    ax0.set_xticks(x); ax0.set_xticklabels(maps); ax0.set_ylim(0, 1); ax0.legend(frameon=False, fontsize=8); ax0.set_title("180 frames, 3 cameras, 224x224 pixels", fontsize=9)
    ax1.bar(x - 0.2, [s[m]["target_mass"] for m in maps], 0.4, color=figstyle.CATEGORICAL[2], label="map mass on target")
    ax1.bar(x + 0.2, [s[m]["robot_in_top"] for m in maps], 0.4, color=figstyle.CATEGORICAL[3], label="robot share of top pixels")
    ax1.set_xticks(x); ax1.set_xticklabels(maps); ax1.set_ylim(0, 1); ax1.legend(frameon=False, fontsize=8)
    panels = [("policy image (right wrist)", img, None), ("GT target (evaluation only)", (cls[2] == 0).astype(float), "gray"),
              ("noun N L10H7 (16x16)", nup[2], "Blues"), ("gripper grad x input (224x224)", gg[2], "Blues"), ("arm grad x input (224x224)", ag[2], "Blues")]
    for i, (t, im, cm) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i]); ax.imshow(im, cmap=cm) if cm else ax.imshow(im)
        ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"R8: pixel gradients vs attention as target locator. Example {fid} ({ref.moment})", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "ifg-r8-gradient.png", dpi=150); plt.close(fig)
    (out / "ifg-r8-gradient.json").write_text(json.dumps(dict(summary=s, example=fid), indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args(); r7(a.run, a.out); r8(a.run, a.out); print("ok")


if __name__ == "__main__":
    main()
