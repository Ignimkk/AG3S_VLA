"""X3 figure 3종 — `status` 는 clearance 가 아니라 기하 인증을 보고 뒤집힌다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/X3/make_x3_figures.py

숫자 출처는 `outputs/verify/X3/probe_headfix_v2.json` 하나다.
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
import mujoco

from AG3S.ag3s.experiments.common import figstyle
from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

KO = figstyle.use_korean()
C = figstyle.CATEGORICAL
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/x3'))
PROBE = pathlib.Path(_ag3s_workspace_path('outputs/verify/X3/probe_headfix_v2.json'))
RECORD = _ag3s_workspace_path('outputs/live_test/20260925_headfix16d/records/run_0000/run_0000')

OK, BAD = C[2], C[7]   # feasible = 초록, violated = 빨강


def T(ko: str, en: str) -> str:
    return ko if KO else en


def scene_figure(d, pair) -> dict:
    """인증된 마지막 프레임과 인증이 꺼진 첫 프레임을 나란히 놓는다."""
    a, b = pair
    run = load_run(RECORD, limit=15)
    scene = replay_scene(run)
    name = lambda i: mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
    apple_ids = [i for i in range(scene.model.nbody) if "apple" in name(i)]

    caps = {}
    for i in (a, b):
        pose_scene(scene, run.steps[i])
        cf = scene.capture("zed_left")
        caps[i] = (np.asarray(cf.depth), np.isin(np.asarray(cf.body_ids), apple_ids))

    fig = plt.figure(figsize=(13.2, 7.6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.16)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    figstyle.style_axes(fig, np.array(axes))

    rows = {m["i"]: m for m in d["frames"]}
    for col, i in enumerate((a, b)):
        m = rows[i]
        col_ink = OK if m["status"] == "feasible" else BAD
        ax = axes[col]
        ax.imshow(run.steps[i].images["cam_high"])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(T(f"프레임 {i} — cam_high", f"frame {i} — cam_high"), fontsize=10)

        ax = axes[col + 3]
        ax.imshow(np.clip(caps[i][0], 0, 3.0), cmap="Greys", vmin=0, vmax=4.2)
        ov = np.zeros(caps[i][1].shape + (4,))
        ov[caps[i][1]] = matplotlib.colors.to_rgba(C[1])
        ax.imshow(ov)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(T(f"zed_left depth · 사과 {int(caps[i][1].sum())} px",
                       f"zed_left depth · apple {int(caps[i][1].sum())} px"), fontsize=10)
        ax.set_xlabel(T(f"{m['status']}   clearance {m['clearance_after_mm_rollout']:+.2f} mm",
                        f"{m['status']}   clearance {m['clearance_after_mm_rollout']:+.2f} mm"),
                      color=col_ink, fontsize=10)

    # 오른쪽 세로 한 칸 — 판정이 어디서 갈리나
    ax = axes[2]
    ax.axis("off")
    ma, mb = rows[a], rows[b]
    lines = [
        T("두 프레임이 갈리는 자리", "where the two frames part"), "",
        T(f"clearance      {ma['clearance_after_mm_rollout']:+.2f} mm  →  "
          f"{mb['clearance_after_mm_rollout']:+.2f} mm   (둘 다 양수)",
          f"clearance      {ma['clearance_after_mm_rollout']:+.2f} mm  ->  "
          f"{mb['clearance_after_mm_rollout']:+.2f} mm   (both positive)"),
        T(f"validity       {ma['validity']}  →  {mb['validity']}",
          f"validity       {ma['validity']}  ->  {mb['validity']}"),
        T(f"certified      {ma['geometry_certified']}  →  {mb['geometry_certified']}",
          f"certified      {ma['geometry_certified']}  ->  {mb['geometry_certified']}"),
        T(f"status         {ma['status']}  →  {mb['status']}",
          f"status         {ma['status']}  ->  {mb['status']}"), "",
        T("clearance 는 나빠지지 않았다.", "clearance did not get worse."),
        T("뒤집은 것은 기하 인증이다.", "what flipped it is the geometry certificate."),
    ]
    ax.text(0.0, 0.95, "\n".join(lines), va="top", ha="left", fontsize=10,
            family="monospace" if not KO else None, color=figstyle.INK,
            transform=ax.transAxes, linespacing=1.7)

    ax = axes[5]
    ax.axis("off")
    note = next((n for n in rows[b]["notes"] if "certify" in n), "")
    ax.text(0.0, 0.95, T("SQP 가 프레임 " + str(b) + " 에 붙인 note",
                         f"the note SQP attached to frame {b}"),
            va="top", ha="left", fontsize=10, color=figstyle.INK, transform=ax.transAxes)
    ax.text(0.0, 0.78, "\n".join(_wrap(note, 46)), va="top", ha="left", fontsize=8.5,
            color=BAD, transform=ax.transAxes, linespacing=1.6)

    fig.suptitle(T("X3 — clearance 가 양수인데 violated 인 이유",
                   "X3 — why a positive clearance still reads violated"),
                 fontsize=12, color=figstyle.INK)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "x3-scene.png", dpi=150, bbox_inches="tight", facecolor=figstyle.SURFACE)
    plt.close(fig)
    return {"pair": [a, b],
            "apple_px": {str(i): int(caps[i][1].sum()) for i in (a, b)},
            "clearance_mm": {str(i): rows[i]["clearance_after_mm_rollout"] for i in (a, b)},
            "validity": {str(i): rows[i]["validity"] for i in (a, b)},
            "status": {str(i): rows[i]["status"] for i in (a, b)}}


def _wrap(s: str, n: int) -> list[str]:
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > n:
            out.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def trend_figure(d) -> dict:
    fr = d["frames"]
    x = [m["i"] for m in fr]
    clr = [m["clearance_after_mm_rollout"] for m in fr]
    cert = [bool(m["geometry_certified"]) for m in fr]
    delta = [m["clearance_delta_mm"] for m in fr]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    figstyle.style_axes(fig, axes)

    ax = axes[0]
    ax.axhline(0, color=figstyle.INK, lw=1.2)
    for xi, c, m in zip(x, clr, fr):
        col = OK if m["status"] == "feasible" else BAD
        ax.plot([xi], [c], "o", color=col, ms=8)
    ax.plot(x, clr, "-", color=figstyle.GRID_INK, lw=1, zorder=0)
    ax.plot([], [], "o", color=OK, label="feasible")
    ax.plot([], [], "o", color=BAD, label="violated")
    ax.set_xlabel(T("정책 호출", "policy call"))
    ax.set_ylabel(T("clearance_after (mm)", "clearance_after (mm)"))
    ax.set_title(T("15 프레임 전부 0 위에 있다", "all 15 frames sit above zero"), fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.bar(x, [1 if c else 0 for c in cert],
           color=[OK if c else BAD for c in cert], width=0.7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([T("False", "False"), T("True", "True")])
    ax.set_xlabel(T("정책 호출", "policy call"))
    ax.set_title(T("geometry_certified — status 와 같은 모양이다",
                   "geometry_certified — same shape as status"), fontsize=10)

    ax = axes[2]
    ax.axhline(0, color=figstyle.INK, lw=1.2)
    ax.plot(x, delta, "-o", color=C[0], ms=5)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(T("정책 호출", "policy call"))
    ax.set_ylabel(T("차 (mm)", "delta (mm)"))
    ax.set_title(T("full_violation: states 준 것 − 안 준 것",
                   "full_violation: with states - without"), fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "x3-trend.png", dpi=150, bbox_inches="tight", facecolor=figstyle.SURFACE)
    plt.close(fig)
    return {"clearance_after_mm": clr, "geometry_certified": cert,
            "clearance_delta_mm": delta,
            "status": [m["status"] for m in fr]}


def table_figure(d) -> dict:
    fr = d["frames"]
    head = [T("프레임", "frame"), T("단계", "phase"), "status",
            T("clearance_after", "clearance_after"), "validity", "certified",
            T("인증 note", "cert note"), "iter"]
    body = [[str(m["i"]), m["phase"], m["status"],
             f"{m['clearance_after_mm_rollout']:+.2f} mm", str(m["validity"]),
             str(m["geometry_certified"]), "O" if m["has_certification_note"] else "-",
             str(m["iterations"])] for m in fr]

    fig, ax = plt.subplots(figsize=(11.4, 6.4))
    figstyle.style_axes(fig, np.array([ax]))
    ax.axis("off")
    tab = ax.table(cellText=body, colLabels=head, cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.45)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        if r == 0:
            cell.set_facecolor("#eef2f7")
        else:
            cell.set_facecolor("#eef7f0" if fr[r - 1]["status"] == "feasible" else "#fdeeee")
    s = d["summary"]
    ax.set_title(T(f"X3 — violated {s['n_violated']}/{s['n_frames']} · "
                   f"certified False {s['n_geometry_certified_false']} · "
                   f"clearance 양수인데 violated {s['n_violated_with_positive_clearance']} · "
                   f"full_violation 두 벌의 최대 차 {s['clearance_delta_mm_max_abs']:.1f} mm",
                   f"X3 — violated {s['n_violated']}/{s['n_frames']} · "
                   f"certified False {s['n_geometry_certified_false']} · "
                   f"violated with positive clearance {s['n_violated_with_positive_clearance']} · "
                   f"max |delta| {s['clearance_delta_mm_max_abs']:.1f} mm"),
                 fontsize=10.5, color=figstyle.INK, pad=14)
    fig.savefig(OUT / "x3-table.png", dpi=150, bbox_inches="tight", facecolor=figstyle.SURFACE)
    plt.close(fig)
    return {"summary": d["summary"], "rows": body, "header": head}


def main() -> None:
    d = json.loads(PROBE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)

    cert = [m["i"] for m in d["frames"] if m["geometry_certified"]]
    unc = [m["i"] for m in d["frames"] if not m["geometry_certified"]]
    pair = (cert[-1] if cert else d["frames"][0]["i"], unc[0] if unc else d["frames"][-1]["i"])

    s = scene_figure(d, pair)
    (OUT / "x3-scene.json").write_text(json.dumps(
        {"figure": "x3-scene.png", "source": str(PROBE), "numbers": s}, indent=1,
        ensure_ascii=False))
    print("wrote", OUT / "x3-scene.png")

    t = trend_figure(d)
    (OUT / "x3-trend.json").write_text(json.dumps(
        {"figure": "x3-trend.png", "source": str(PROBE), "numbers": t}, indent=1,
        ensure_ascii=False))
    print("wrote", OUT / "x3-trend.png")

    tb = table_figure(d)
    (OUT / "x3-table.json").write_text(json.dumps(
        {"figure": "x3-table.png", "source": str(PROBE), "numbers": tb}, indent=1,
        ensure_ascii=False))
    print("wrote", OUT / "x3-table.png")


if __name__ == "__main__":
    main()
