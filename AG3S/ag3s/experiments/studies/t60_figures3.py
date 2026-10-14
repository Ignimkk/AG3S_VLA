"""T60 step 3 figures — calibration records at their endpoints (real scene), T47 held-out on_target and T48a
class composition per window (graph), selections per window (table, from T60.verify.json).

usage: t60_figures3.py RUN_DIR FIG_DIR      (.venv-openpi-live)
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402

STORE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage"))
SCENE_KEYS = ("ep1807", "ep0")


def scene(run: pathlib.Path, fig_dir: pathlib.Path, ko: bool) -> dict:
    ends = json.loads((run / "verify/T60/step3/records/record_endpoints.json").read_text())["records"]
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2 * len(SCENE_KEYS), 4, height_ratios=[1.2, 0.8] * len(SCENE_KEYS))
    side = {}
    for i, key in enumerate(SCENE_KEYS):
        e = ends[key]
        rec = STORE / e["record"]
        with np.load(run / f"verify/T60/step3/records/{key}.npz") as z:
            s = {k: z[k] for k in z.files}
        ts = s["t_step"]
        f_p = int(np.searchsorted(ts, e["t_P"]))
        f_h = int(np.searchsorted(ts, e["t_H"])) if e["t_H"] is not None else len(ts) - 1
        picks = [("first in-crate frame" if not ko else "처음 in-crate", int(np.flatnonzero(s["in_crate"])[0])),
                 ("last kept (P)" if not ko else "P 마지막 유지 frame", int(np.flatnonzero(s["keep_frames_P"])[-1])),
                 ("last kept (H)" if not ko else "H 마지막 유지 frame", int(np.flatnonzero(s["keep_frames_H"])[-1])),
                 ("last recorded" if not ko else "기록 마지막 frame", len(ts) - 1)]
        for j, (lab, f) in enumerate(picks):
            ax = fig.add_subplot(gs[2 * i, j])
            with np.load(rec / f"step_{f:05d}.npz") as z:
                ax.imshow(z["image_cam_high"])
            ax.set_title(f"{lab}: frame {f} (t={int(ts[f])})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        ax = fig.add_subplot(gs[2 * i + 1, :])
        ax.plot(ts, s["q_err"], color=figstyle.CATEGORICAL[0], lw=1.4, label="max|q−q0| (rad)")
        ax.plot(ts, s["tcp_err"] * 5, color=figstyle.CATEGORICAL[1], lw=1.4, label="TCP error ×5 (m)")
        ax.plot(ts, np.minimum(s["speed"], 0.5) * 2, color=figstyle.CATEGORICAL[2], lw=1, label="target speed ×2 (m/s)")
        ax.fill_between(ts, 0, 1.6, where=s["in_crate"], color=figstyle.SEQ_BLUE[0], alpha=0.6, lw=0, label="in crate")
        ax.fill_between(ts, 1.6, 1.75, where=s["hand_target"], color=figstyle.CATEGORICAL[3], lw=0, label="hand–target contact")
        for lab, t, c in (("t_P", e["t_P"], figstyle.CATEGORICAL[4]), ("t_H", e["t_H"], figstyle.CATEGORICAL[5])):
            if t is not None:
                ax.axvline(t, color=c, lw=1.6); ax.text(t + 4, 1.45, lab, color=c, fontsize=9)
                ax.axvspan(t, ts[-1], color=c, alpha=0.06)
        ax.set_ylim(0, 1.8); ax.set_xlim(0, ts[-1]); ax.set_xlabel("t_step (control row)")
        ax.set_title(f"{key} ({e['role']}, {e['used_arm']} arm, {e['target']}) — frames {e['n_frames']}, "
                     f"removed P {e['frames_removed_P']} / H {e['frames_removed_H']}", loc="left", fontsize=10)
        if i == 0:
            ax.legend(fontsize=7.5, ncol=5, loc="upper left", bbox_to_anchor=(0, -0.3), frameon=False)
        side[key] = {"endpoints": e, "picked_frames": dict(picks)}
    figstyle.style_axes(fig, fig.axes)
    fig.suptitle("T60 — head 선택 calibration record 의 종료 시점 (cam_high, inference frame)" if ko else
                 "T60 — calibration records at their endpoints (cam_high)", fontsize=13)
    fig.tight_layout(); fig.savefig(fig_dir / "t60-records-scene.png", dpi=120); plt.close(fig)
    T.write_json(fig_dir / "t60-records-scene.json", side)
    return side


def graphs(v: dict, fig_dir: pathlib.Path, ko: bool) -> dict:
    n3 = v["numbers"]["step3"]
    e2e = n3["t47_e2e_heldout"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    ax = axes[0]
    srcs = ["l8h2", "topk_common", "topk_camera", "head_avg", "oracle"]
    runs = [("published_s0", "full σ=0"), ("P_s0", "P σ=0"), ("H_s0", "H σ=0"), ("P_s1", "P σ=1")]
    side = {"t47": {}}
    for i, (rk, lab) in enumerate(runs):
        vals = [e2e[rk]["pooled"].get(s, {}).get("on_target", np.nan) for s in srcs]
        xs = np.arange(len(srcs)) + (i - 1.5) * 0.2
        ax.bar(xs, vals, 0.2, color=figstyle.SEQ_BLUE[[1, 4, 6, -1][i]], label=lab)
        for x, y in zip(xs, vals):
            if np.isfinite(y):
                ax.text(x, y, f"{int(y)}", ha="center", va="bottom", fontsize=7)
        side["t47"][rk] = dict(zip(srcs, vals))
    ax.set_xticks(np.arange(len(srcs)))
    ax.set_xticklabels([f"{s}\n{e2e['published_s0']['source_cells'].get(s) if s=='topk_common' else ''}→\n{e2e['P_s0']['source_cells'].get(s) if s=='topk_common' else ''}"
                        if s == "topk_common" else s for s in srcs], fontsize=7.5)
    ax.set_ylabel("held-out on_target (of 60 frames)"); ax.set_ylim(0, 66)
    ax.set_title("T47 held-out 3D — 선택이 바뀐 공통 top-3" if ko else "T47 held-out 3D", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    ax = axes[1]
    t48 = n3["t48a_ep1807"]["windows"]
    cls = ("pick", "place", "home")
    bottom = np.zeros(3)
    for ci, c in enumerate(cls):
        vals = np.array([t48[w]["gt_class_frames"][c] for w in ("full", "P", "H")])
        ax.bar(np.arange(3), vals, bottom=bottom, color=figstyle.CATEGORICAL[ci], label=f"GT {c}")
        bottom += vals
    for xi, w in enumerate(("full", "P", "H")):
        a = t48[w]
        ax.text(xi, bottom[xi] + 1, f"C0 {a['accuracy']['C0']['k']}/{a['accuracy']['C0']['n']}\n"
                                    f"C1→pick {a['C1_to_pick_all']['k']}/{a['C1_to_pick_all']['n']}\n"
                                    f"place→pick {a['C1_to_pick']['place']['k']}/{a['C1_to_pick']['place']['n']}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(["full", "P", "H"]); ax.set_ylim(0, 105)
    ax.set_ylabel("ep1807 host frames"); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("T48a ep1807 — 창별 frame 구성과 요약" if ko else "T48a ep1807 per window", loc="left")
    side["t48a"] = t48
    figstyle.style_axes(fig, axes)
    fig.tight_layout(); fig.savefig(fig_dir / "t60-selection-graphs.png", dpi=130); plt.close(fig)
    T.write_json(fig_dir / "t60-selection-graphs.json", side)
    return side


def table(v: dict, fig_dir: pathlib.Path, ko: bool) -> dict:
    n3 = v["numbers"]["step3"]
    fmt = lambda cells: " · ".join(f"L{a}H{b}" for a, b in cells) if cells else "—"
    rows, side = [], {}
    t47 = n3["t47_select"]; e2e = n3["t47_e2e_heldout"]
    rows.append(["T47 label-free common top-3 (d0 last, cam τ)"] + [fmt(t47[w]["d0_last_cam_common_top3"]) for w in ("full", "P", "H")])
    rows.append(["T47 GT-oracle top-3 (selection records)"] + [fmt(e2e[k]["oracle_d0_last_top3"]) for k in ("published_s0", "P_s0", "H_s0")])
    rows.append(["T47 common top-3 held-out on_target σ0 (L8H2)"] +
                [f"{e2e[k]['pooled']['topk_common']['on_target']}/60 ({e2e[k]['pooled']['l8h2']['on_target']}/60)" for k in ("published_s0", "P_s0", "H_s0")])
    t49 = n3["t49_select"]
    for q, lab in (("target__vs_target", "T49 noun target"), ("basket__vs_destination", "T49 basket")):
        rows.append([f"{lab} GT top-3 (primary)"] + [fmt(t49[w][q]["gt_top3"]) for w in ("full", "P", "H")])
        rows.append([f"{lab} label-free top-3 (secondary)"] + [fmt(t49[w][q]["label_free_cam_top3"]) for w in ("full", "P", "H")])
    t51 = n3["t51"]
    for lab in ("gt", "self"):
        rows.append([f"T51 P4 selected ({lab} label)"] +
                    [(" + ".join(t51[w][lab]["ksel_selected"]["cells"]) if t51[w].get(lab) else "pending") for w in ("full", "P", "H")])
        rows.append([f"T51 P2 3D top-1 ({lab}, scene mean)"] +
                    [(f"{t51[w][lab]['score3d_ranking_top5'][0]['cell']} {t51[w][lab]['score3d_ranking_top5'][0][('gt' if lab=='gt' else 'self') + '_identity']:.3f}"
                      if t51[w].get(lab) and t51[w][lab].get("score3d_ranking_top5") else "pending") for w in ("full", "P", "H")])
    fig, ax = plt.subplots(figsize=(17, 0.5 + 0.42 * len(rows)))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=["", "full (published)", "P", "H"], loc="center", cellLoc="center",
                  colWidths=[0.34, 0.22, 0.22, 0.22])
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.45)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(figstyle.GRID_INK)
        if i == 0:
            c.set_facecolor(figstyle.SEQ_BLUE[0])
        elif j > 1 and rows[i - 1][j] != rows[i - 1][1]:
            c.set_facecolor("#fde9dd")
    ax.set_title("T60 — 선택 결과의 창별 비교 (주황 = full 과 다름)" if ko else "T60 — selections per window (orange = differs from full)",
                 fontsize=12)
    figstyle.style_axes(fig, [ax])
    fig.savefig(fig_dir / "t60-selection-table.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    side = {"rows": rows}
    T.write_json(fig_dir / "t60-selection-table.json", side)
    return side


def main(run: pathlib.Path, fig_dir: pathlib.Path) -> None:
    ko = figstyle.use_korean()
    v = json.loads((ROOT / "AG3S/ag3s/docs/handoff/T60.verify.json").read_text())
    scene(run, fig_dir, ko)
    graphs(v, fig_dir, ko)
    table(v, fig_dir, ko)
    print("ok")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
