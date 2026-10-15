"""IFG figures (rule A): real-scene overlays, episode-level graph, number table. Each PNG has a JSON sidecar.

  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_figures \
      --run $RUN --out AG3S/ag3s/docs/figures/ifg
Reads `$RUN/s3` (rows, masks, paired_summary.json), `$RUN/s0/replay` (K, T) and dump RGB.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_reader as R  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_summary as S  # noqa: E402

CAMS = ("head (zed_left)", "left wrist", "right wrist")
SCENE_ARMS = ("G:N", "FG:N:projector", "F:N:projector")


def _project(pts, K, T, hw=(480, 640), out=224):
    pc = (pts - T[:3, 3]) @ T[:3, :3]
    ok = pc[:, 2] > 0.05
    u = K[0, 0] * pc[:, 0] / np.maximum(pc[:, 2], 1e-6) + K[0, 2]
    v = K[1, 1] * pc[:, 1] / np.maximum(pc[:, 2], 1e-6) + K[1, 2]
    ok &= (u >= 0) & (u < hw[1]) & (v >= 0) & (v < hw[0])
    return u * out / hw[1], v * out / hw[0], ok


def scene_figure(run: pathlib.Path, fid: str, out: pathlib.Path, rows_by_frame: dict) -> dict:
    ref = {r.frame_id: r for r in R.list_frames()}[fid]
    with np.load(ref.path / "inputs.npz") as z:
        imgs = [np.clip((z[f"image_{c}"] + 1) / 2, 0, 1) for c in R.DUMP_CAMERAS]
    with np.load(run / "s0" / "replay" / f"{fid}.npz") as z:
        K, T = z["K"], z["T_base_cam"]
    mk = np.load(run / "s3" / "masks" / f"{fid}.npz")
    pos, ne = mk["node_pos"].astype(float), mk["node_eval"]
    gt = mk["eval_target"].astype(float)[ne] >= 0.5
    # camera that sees most GT-target nodes
    cnt = [int(_project(pos[gt], K[c], T[c])[2].sum()) for c in range(3)]
    c = int(np.argmax(cnt))
    ko = figstyle.use_korean()
    fig, axes = plt.subplots(1, 1 + len(SCENE_ARMS), figsize=(4 * (1 + len(SCENE_ARMS)), 4.4))
    figstyle.style_axes(fig, axes)
    u, v, ok = _project(pos, K[c], T[c])
    col_tp, col_fp, col_fn = figstyle.CATEGORICAL[2], figstyle.CATEGORICAL[1], figstyle.CATEGORICAL[0]
    side = dict(frame_id=fid, camera=CAMS[c], panels={})
    axes[0].imshow(imgs[c]); axes[0].scatter(u[ok & gt], v[ok & gt], s=1.5, c=col_fn, lw=0)
    axes[0].set_title(("GT target (평가 전용)" if ko else "GT target (evaluation only)"), fontsize=10)
    for ax, arm in zip(axes[1:], SCENE_ARMS):
        m = mk["m:" + arm].astype(float)
        pred = np.isfinite(m) & (m >= 0.5)
        ax.imshow(imgs[c])
        for sel, col in ((ok & pred & gt, col_tp), (ok & pred & ~gt, col_fp), (ok & ~pred & gt, col_fn)):
            ax.scatter(u[sel], v[sel], s=1.5, c=col, lw=0)
        r = rows_by_frame.get((fid, arm))
        iou = None if r is None else r.get("iou")
        ax.set_title(f"{arm}  IoU={iou:.2f}" if iou is not None else arm, fontsize=10)
        side["panels"][arm] = dict(iou=iou, precision=None if r is None else r.get("precision"),
                                   recall=None if r is None else r.get("recall"),
                                   neighbor_fp=None if r is None else r.get("neighbor_fp"))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], ls="", marker="o", color=col_tp, label="TP"),
                        Line2D([], [], ls="", marker="o", color=col_fp, label="FP"),
                        Line2D([], [], ls="", marker="o", color=col_fn, label="FN / GT")],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{fid} ({ref.moment}) - 5 mm voxel nodes projected on the policy image, tau=0.5", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    p = out / f"ifg-scene-{fid}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    (out / f"ifg-scene-{fid}.json").write_text(json.dumps(side, indent=1, default=float))
    return side


def episode_graph(summary: dict, epm: dict, out: pathlib.Path) -> dict:
    arms = []
    for sd in ("N", "A", "P"):
        arms += [f"G:{sd}", f"FGe:{sd}:projector", f"FGu:{sd}:projector", f"FG:{sd}:projector", f"F:{sd}:projector"]
    base = [k[0] for k in epm if k[0].startswith("T54")]
    arms += sorted(set(base))
    rng = np.random.default_rng(S.SEED)
    vals = {}
    for a in arms:
        eps = epm.get((a, 0.5, "voxel"), {})
        v = np.array([d["iou"] for d in eps.values() if "iou" in d])
        if v.size == 0:
            continue
        boot = v[rng.integers(0, v.size, (S.B, v.size))].mean(1)
        vals[a] = dict(mean=float(v.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                       n_episodes=int(v.size))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw=dict(width_ratios=[1.1, 1]))
    figstyle.style_axes(fig, [ax, ax2])
    names = list(vals)
    method_col = {"G": 0, "FGe": 6, "FGu": 3, "FG": 1, "F": 4, "T54region": 7, "T54ball": 5}
    for i, a in enumerate(names):
        col = figstyle.CATEGORICAL[method_col.get(a.split(":")[0], 0)]
        ax.errorbar(vals[a]["mean"], i, xerr=[[vals[a]["mean"] - vals[a]["ci95"][0]], [vals[a]["ci95"][1] - vals[a]["mean"]]],
                    fmt="o", color=col, capsize=2)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("episode-mean visible 3D IoU (5 mm voxels, tau 0.5), 95% bootstrap CI"); ax.set_xlim(0, 1)
    # paired FG - G per feature x seed
    comps = [c for c in summary["comparisons"] if c["metric"] == "iou" and c["a"][0].startswith("FG:")
             and c["b"][0].startswith("G:") and c["a"][1] == 0.5]
    labels = [c["a"][0] for c in comps]
    for i, c in enumerate(comps):
        col = figstyle.CATEGORICAL[1] if c["ci95"][1] < 0 else (figstyle.CATEGORICAL[2] if c["ci95"][0] > 0 else figstyle.INK_2)
        ax2.errorbar(c["diff"], i, xerr=[[c["diff"] - c["ci95"][0]], [c["ci95"][1] - c["diff"]]], fmt="o", color=col, capsize=2)
    ax2.axvline(0, color=figstyle.INK, lw=0.8)
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels([f"{l} - G:{l.split(':')[1]}" for l in labels], fontsize=8)
    ax2.invert_yaxis(); ax2.set_xlabel("paired episode IoU difference, 95% CI")
    fig.suptitle(f"IFG S3 dev: {summary['n_episodes']} episodes / {summary['n_frames']} frames (T53+T54c, all dev)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "ifg-episode-iou.png", dpi=150); plt.close(fig)
    side = dict(arms=vals, paired_fg_minus_g=comps, bootstrap=summary["bootstrap"])
    (out / "ifg-episode-iou.json").write_text(json.dumps(side, indent=1))
    return side


def table_figure(epm: dict, out: pathlib.Path) -> dict:
    arms = ["G:N", "FGe:N:projector", "FGu:N:projector", "FG:N:projector", "F:N:projector", "FG:N:prefix_L8",
            "G:A", "FG:A:projector", "G:P", "FG:P:projector"]
    arms += sorted({k[0] for k in epm if k[0].startswith("T54")})
    cols = ["iou", "precision", "recall", "neighbor_fp", "support_fp", "robot_fp", "unknown_weight"]
    data = []
    for a in arms:
        eps = epm.get((a, 0.5, "voxel"))
        if not eps:
            continue
        data.append([a] + [float(np.nanmean([d.get(c, np.nan) for d in eps.values()])) for c in cols] + [len(eps)])
    fig, ax = plt.subplots(figsize=(13, 0.45 * len(data) + 1.2))
    figstyle.style_axes(fig, ax); ax.axis("off")
    cell = [[r[0]] + [f"{x:.3f}" if i < 3 else f"{x:.0f}" for i, x in enumerate(r[1:-1])] + [str(r[-1])] for r in data]
    t = ax.table(cellText=cell, colLabels=["arm", "IoU", "precision", "recall", "neighbor FP", "support FP",
                                           "robot FP", "unknown (target)", "episodes"], loc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.3)
    ax.set_title("Episode means, tau 0.5, 5 mm voxel occupancy (FP columns = voxel counts)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "ifg-table.png", dpi=150); plt.close(fig)
    side = dict(columns=["arm"] + cols + ["episodes"], rows=data)
    (out / "ifg-table.json").write_text(json.dumps(side, indent=1))
    return side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, _ = S.load_rows(a.run / "s3", [a.run / "baseline"] if a.baseline else [], S.s0_valid(a.run / "s0"))
    summary = json.loads((a.run / "s3" / "paired_summary.json").read_text())
    excl = set(summary.get("excluded_incomplete", []))
    rows = [r for r in rows if r["frame_id"] not in excl]
    epm = S.episode_means(rows)
    rbf = {(r["frame_id"], r["arm_id"]): r for r in rows if r["level"] == "voxel" and r.get("threshold") == 0.5}
    episode_graph(summary, epm, a.out)
    table_figure(epm, a.out)
    # scene overlays: the debug frame family, then the largest FG<G and FG>G frames (failure overlays)
    d = defaultdict(dict)
    for (fid, arm), r in rbf.items():
        if arm in ("G:N", "FG:N:projector") and r.get("iou") is not None:
            d[fid][arm] = r["iou"]
    diffs = sorted((v["FG:N:projector"] - v["G:N"], f) for f, v in d.items() if len(v) == 2)
    picks = ["T53_ep1839_f023", diffs[0][1], diffs[-1][1]]
    sides = [scene_figure(a.run, f, a.out, rbf) for f in dict.fromkeys(picks)]
    print(json.dumps(dict(scene=[s["frame_id"] for s in sides], fg_minus_g_extremes=[diffs[0], diffs[-1]]), default=float))


if __name__ == "__main__":
    main()
