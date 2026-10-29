"""T1-a figure 3 종. 저장 위치는 AG3S/ag3s/docs/figures/t1-a/ 만 쓴다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1-a/make_figures.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from AG3S.ag3s.experiments.common import figstyle

KO = figstyle.use_korean()
C = figstyle.CATEGORICAL
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t1-a'))
OUT.mkdir(parents=True, exist_ok=True)
VER = pathlib.Path(_ag3s_workspace_path('outputs/verify/T1-a'))
RECORD = "run_16d_ep1800"
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def T(ko: str, en: str) -> str:
    return ko if KO else en


# ------------------------------------------------------------------ 1. 실제 씬
def scene_figure() -> dict:
    from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
    import mujoco

    run = load_run(RECORD, limit=15)
    scene = replay_scene(run)
    name = lambda i: mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
    apple = [i for i in range(scene.model.nbody) if "apple" in name(i)]

    i = 10
    pose_scene(scene, run.steps[i])
    panels = []
    for c in CAMS:
        fr = scene.capture(c)
        d = np.asarray(fr.depth, float)
        b = np.asarray(fr.body_ids)
        m = np.isin(b, apple)
        panels.append({"cam": c, "depth": d, "mask": m, "px": int(m.sum()),
                       "pos": np.asarray(scene.T_base_cam(c), float)[:3, 3]})
    apple_xyz = np.asarray(scene.body_position_in_base("apple"), float)

    blob = np.load(_ag3s_workspace_path('AG3S/ag3s/asset/data/attention_16d_ep1800.npz'))
    att = np.asarray(blob["attention"], np.float32)
    aggs = [str(a) for a in blob["aggregations"]]
    grid14 = att[i, 0, aggs.index("last"), 8, 2, 0]   # 14D 셀 L8H2, denoise 0, agg last, cam_high

    fig = plt.figure(figsize=(15.0, 8.6))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.05, 1.0], hspace=0.28, wspace=0.18)

    # --- 배치도 ---
    ax = fig.add_subplot(gs[0, :2])
    for k, p in enumerate(panels):
        ax.scatter([p["pos"][0]], [p["pos"][1]], s=110, marker="^", color=C[k],
                   zorder=5, label=f"{p['cam']}  ({p['px']} px)")
        ax.annotate("", xy=(apple_xyz[0], apple_xyz[1]), xytext=(p["pos"][0], p["pos"][1]),
                    arrowprops=dict(arrowstyle="->", color=C[k], lw=1.1,
                                    alpha=0.9 if p["px"] else 0.25,
                                    ls="-" if p["px"] else ":"))
    ax.scatter([apple_xyz[0]], [apple_xyz[1]], s=150, marker="*", color=C[7], zorder=6,
               label=T("사과 (참 위치)", "apple (true position)"))
    ax.set_xlabel("base x [m]"); ax.set_ylabel("base y [m]")
    ax.set_title(T("배치도 — 세 카메라와 사과. 실선은 사과가 보이는 카메라",
                   "Layout - three cameras and the apple. Solid = camera that sees it"),
                 fontsize=10, color=figstyle.INK)
    ax.legend(fontsize=7, frameon=False, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    figstyle.style_axes(fig, np.array([ax]))

    # --- 14D 셀의 attention 격자 ---
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(grid14, cmap=figstyle.sequential_cmap(), interpolation="nearest")
    pk = np.unravel_index(int(np.argmax(grid14)), grid14.shape)
    ax.scatter([pk[1]], [pk[0]], s=90, marker="x", color=C[1], linewidths=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(T("14D 셀 L8H2 의 16x16 attention\n(x = 최대 칸)",
                   "14D cell L8H2, 16x16 attention\n(x = peak)"), fontsize=9, color=figstyle.INK)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
    txt = [
        T("기록", "record") + f": {RECORD}   " + T("프레임", "frame") + f" {i}",
        "",
        T("grounding 이 읽는 카메라: cam_high = zed_left", "grounding reads: cam_high = zed_left"),
        T("  (esdf_rollout.py:137 ci = cameras.index('cam_high'))",
          "  (esdf_rollout.py:137 ci = cameras.index('cam_high'))"),
        "",
        T("이 프레임의 사과 픽셀 수", "apple pixels this frame") + ":",
    ] + [f"    {p['cam']:14} {p['px']:6d} px" for p in panels]
    ax.text(0.0, 0.98, "\n".join(txt), va="top", ha="left", fontsize=9, color=figstyle.INK,
            family=matplotlib.rcParams["font.family"][0])

    # --- 세 카메라의 depth + 사과 픽셀 ---
    for k, p in enumerate(panels):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(p["depth"], cmap=figstyle.sequential_cmap(), vmin=0.2, vmax=2.0)
        if p["px"]:
            ys, xs = np.nonzero(p["mask"])
            ax.scatter(xs, ys, s=0.6, color=C[7], alpha=0.85)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{p['cam']}  " + T("사과 ", "apple ") + f"{p['px']} px",
                     fontsize=9, color=C[7] if p["px"] else figstyle.INK_2)

    ax = fig.add_subplot(gs[1, 3]); ax.axis("off")
    vis = json.loads((VER / "visibility_ep1800.json").read_text())
    vl = json.loads((VER / "visibility_long.json").read_text())
    lines = [T("사과가 보이는 프레임 수", "frames where the apple is visible"), ""]
    for tag, v in (("ep1800 (15)", vis), ("long16d (50)", vl)):
        lines.append(f"  {tag}")
        for c in CAMS:
            s = v["summary"][c]
            lines.append(f"    {c:14} {s['frames_with_any_px']:3d}  "
                         + T("최대", "max") + f" {s['max_px']}")
    ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, color=figstyle.INK,
            family=matplotlib.rcParams["font.family"][0])

    fig.suptitle(T("T1-a — attention 은 cam_high 에서 읽는데 사과는 그 영상에 없다",
                   "T1-a - attention is read from cam_high, but the apple is not in that image"),
                 fontsize=13, color=figstyle.INK)
    fig.savefig(OUT / "t1a-scene.png", dpi=130, facecolor=figstyle.SURFACE, bbox_inches="tight")
    plt.close(fig)
    return {"frame": i, "apple_pos_base": apple_xyz.tolist(),
            "apple_px": {p["cam"]: p["px"] for p in panels}}


# ------------------------------------------------------------------ 2. 그래프
def graph_figure() -> dict:
    vis = json.loads((VER / "visibility_ep1800.json").read_text())
    vl = json.loads((VER / "visibility_long.json").read_text())
    rep = json.loads((VER / "repro.json").read_text())
    sw = json.loads((VER / "sweep_ep1800.json").read_text()) if (VER / "sweep_ep1800.json").exists() else None

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    figstyle.style_axes(fig, axes)

    # (a) 프레임별 사과 픽셀
    ax = axes[0]
    for k, c in enumerate(CAMS):
        ax.plot([r["frame"] for r in vl["rows"]], [r[c] for r in vl["rows"]],
                "-o", ms=3, color=C[k], label=c)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel(T("프레임 (긴 기록 50)", "frame (long record, 50)"))
    ax.set_ylabel(T("사과 픽셀 수", "apple pixels"))
    ax.set_title(T("사과를 보는 카메라는 왼 손목뿐", "only the left wrist camera sees the apple"),
                 fontsize=11, color=figstyle.INK)
    ax.legend(fontsize=8, frameon=False)

    # (b) 1296 조합 분포
    ax = axes[1]
    if sw:
        pk = np.array([r["peak_on_target_frames"] for r in sw["stageA"]["rows"]])
        b = sw["stageB"]["rows"]
        lbl = [f"L{r['layer']}H{r['head']}\nd{r['denoise']} {r['agg'][:4]}" for r in b]
        xi = np.arange(len(b))
        w = 0.4
        ax.bar(xi - w / 2, [r["n_has_target"] for r in b], w, color=C[0],
               label=T("has_target (무언가를 지었다)", "has_target (grounded something)"))
        ax.bar(xi + w / 2, [r["n_on_apple"] for r in b], w, color=C[7],
               label=T("on_apple (그것이 사과였다)", "on_apple (it was the apple)"))
        for j, r in enumerate(b):
            if r["is_14d_cell"]:
                ax.annotate("14D", (xi[j], -0.9), ha="center", fontsize=8, color=C[1])
        ax.annotate(T("on_apple 은 13 조합 전부 0 — 빨간 막대가 하나도 없다",
                      "on_apple is 0 for all 13 combos - no red bar exists"),
                    (len(b) / 2 - 0.5, 12.6), ha="center", fontsize=8, color=C[7])
        ax.set_xticks(xi); ax.set_xticklabels(lbl, fontsize=6)
        ax.set_ylabel(T("프레임 수 (15 중)", "frames (of 15)"))
        ax.set_ylim(-1.5, 15.5)
        ax.set_title(T(f"B 단계 — 1296 조합 전부 peak_on_target {int(pk.max())}/15",
                       f"stage B - all 1296 combos peak_on_target {int(pk.max())}/15"),
                     fontsize=11, color=figstyle.INK)
        ax.legend(fontsize=7, frameon=False)
    else:
        ax.text(0.5, 0.5, "sweep pending", ha="center", va="center")

    # (c) 재현성 — 프레임별 clearance 폭
    ax = axes[2]
    fr = rep["frames"]
    x = [f["frame"] for f in fr]
    runs = rep["groups"]["identical"]
    # legacy_attn 과 repeat_2 는 소수 셋째 자리까지 같다 — 그냥 그리면 하나가 숨는다.
    style = {"R2_legacy_attn": dict(lw=4.0, alpha=0.35, ms=0),
             "R2_repeat_2": dict(lw=1.4, ms=3)}
    for k, t in enumerate(runs):
        lab = t.replace("R2_", "")
        if t == "R2_legacy_attn":
            lab += T("  (repeat_2 와 완전히 같음)", "  (identical to repeat_2)")
        ax.plot(x, [f["arm_after_mm"][t] for f in fr], "-o", color=C[k], label=lab,
                **style.get(t, dict(lw=1.4, ms=3)))
    ax.axhline(0, color=figstyle.INK_2, lw=1, ls="--")
    for f in fr:
        if f["flipped"]:
            ax.axvspan(f["frame"] - 0.4, f["frame"] + 0.4, color=C[7], alpha=0.10, lw=0)
    ax.set_xlabel(T("프레임", "frame")); ax.set_ylabel(T("팔 여유거리 (해 뒤) [mm]", "arm clearance after [mm]"))
    ax.set_title(T(f"같은 명령 4 회 — 뒤집힌 프레임 {rep['n_flipped_frames']} 개 (음영)",
                   f"4 identical runs - {rep['n_flipped_frames']} frames flip (shaded)"),
                 fontsize=11, color=figstyle.INK)
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle(T("T1-a — 셀 선택 앞의 조건, 그리고 재현성",
                   "T1-a - the condition before cell choice, and reproducibility"),
                 fontsize=13, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "t1a-trend.png", dpi=130, facecolor=figstyle.SURFACE, bbox_inches="tight")
    plt.close(fig)
    return {"visibility_summary_ep1800": vis["summary"],
            "visibility_summary_long": vl["summary"],
            "flipped_frames": rep["flipped_frames"],
            "identical_feasible": rep["identical_feasible"]}


# ------------------------------------------------------------------ 3. 표
def table_figure() -> dict:
    rep = json.loads((VER / "repro.json").read_text())
    vis = json.loads((VER / "visibility_ep1800.json").read_text())
    vl = json.loads((VER / "visibility_long.json").read_text())
    sw = json.loads((VER / "sweep_ep1800.json").read_text()) if (VER / "sweep_ep1800.json").exists() else None
    c14 = best = None
    if sw:
        c14 = [r for r in sw["stageB"]["rows"] if r["is_14d_cell"]][0]
        best = sw["stageB"]["rows"][0]

    rows = [
        [T("14D 셀 L8H2 가 16D 에서 사과를 짚나", "does 14D cell L8H2 ground the apple on 16D"),
         "L8 H2 denoise0 last",
         (f"on_apple {c14['n_on_apple']}/15" if c14 else "—"),
         (T(f"has_target 는 {c14['n_has_target']}/15 — 사과가 아닌 것을 지었다",
            f"has_target {c14['n_has_target']}/15 - grounded something else") if c14 else "—")],
        [T("1296 조합 중 최선 (사과 기준)", "best of 1296 combos (by on_apple)"),
         (f"L{best['layer']}H{best['head']} d={best['denoise']} {best['agg']}" if best else "—"),
         (f"on_apple {best['n_on_apple']}/15" if best else "—"),
         T("A 단계 peak_on_target 최대 0/15", "stage A peak_on_target max 0/15")],
        [T("사과가 cam_high 에 보이나 (ep1800)", "apple visible in cam_high (ep1800)"),
         "zed_left", f"{vis['summary']['zed_left']['frames_with_any_px']}/15 프레임",
         T("최대 0 픽셀", "max 0 px")],
        [T("사과가 cam_high 에 보이나 (long16d)", "apple visible in cam_high (long16d)"),
         "zed_left", f"{vl['summary']['zed_left']['frames_with_any_px']}/50 프레임",
         T("최대 0 픽셀", "max 0 px")],
        [T("사과를 보는 카메라", "camera that does see the apple"),
         "wrist_cam_l",
         f"{vl['summary']['wrist_cam_l']['frames_with_any_px']}/50 프레임",
         T(f"중앙 {vl['summary']['wrist_cam_l']['median_px']} 픽셀",
           f"median {vl['summary']['wrist_cam_l']['median_px']} px")],
        [T("같은 명령 4 회 feasible", "feasible over 4 identical runs"),
         "legacy", str(rep["identical_feasible"]),
         T(f"반복 중앙값 {rep['identical_iter_median']}", f"iter median {rep['identical_iter_median']}")],
        [T("뒤집힌 프레임", "frames that flip"), "—",
         f"{rep['n_flipped_frames']}/15 {rep['flipped_frames']}",
         T("전부 한 실행(repeat_1)에서만", "all in one run (repeat_1) only")],
    ]
    head = [T("무엇", "what"), T("어디", "where"), T("수치", "number"), T("덧붙임", "note")]

    fig, ax = plt.subplots(figsize=(15.5, 4.6))
    fig.patch.set_facecolor(figstyle.SURFACE); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=head, loc="center",
                  colWidths=[0.34, 0.17, 0.22, 0.27])
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 2.0)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK); cell.set_linewidth(0.6)
        cell.get_text().set_color(figstyle.INK)
        cell.set_facecolor("#efeeea" if r == 0 else figstyle.SURFACE)
        cell.get_text().set_ha("left")
    ax.set_title(T("T1-a — 측정값만. 판정은 lead 와 사용자가 한다",
                   "T1-a - measurements only; the call belongs to lead and user"),
                 fontsize=12, color=figstyle.INK, pad=16)
    fig.savefig(OUT / "t1a-table.png", dpi=130, facecolor=figstyle.SURFACE, bbox_inches="tight")
    plt.close(fig)
    return {"rows": rows}


if __name__ == "__main__":
    print("korean:", KO)
    data = {"scene": scene_figure(), "graph": graph_figure(), "table": table_figure()}
    for nm, key in (("t1a-scene", "scene"), ("t1a-trend", "graph"), ("t1a-table", "table")):
        (OUT / f"{nm}.json").write_text(json.dumps(data[key], indent=2, ensure_ascii=False))
        print("wrote", OUT / f"{nm}.png", "+ .json")
