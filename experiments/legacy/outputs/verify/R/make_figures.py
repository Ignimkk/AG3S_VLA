"""R 스모크 figure 3종 — verifier(A2) 측정용. benchmark/ 밖에 둔다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python \
        outputs/verify/R/make_figures.py
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
# 16D 재측정 figure 는 전부 r-16d/ 아래에 쓴다 — archive 의 14D 증거를 덮지 않기 위해.
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/r-16d'))
OUT.mkdir(parents=True, exist_ok=True)
VER = pathlib.Path(_ag3s_workspace_path('outputs/verify/R'))
RECORD = "outputs/live_test/20260924_long16d/run_0000"
FRAMES = (0, 5, 10, 14)


def T(ko: str, en: str) -> str:
    return ko if KO else en


# ------------------------------------------------------------------ 1. 실제 씬
def scene_figure() -> dict:
    from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)

    run = load_run(RECORD, limit=15)
    scene = replay_scene(run)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)

    cams = ("zed_left", "wrist_cam_l", "wrist_cam_r")
    panels, layout = [], {"cams": {}, "spheres_xy": None}

    for i in FRAMES:
        pose_scene(scene, run.steps[i])
        frame = scene.capture("zed_left")
        depth = np.asarray(frame.depth, float)
        K = np.asarray(frame.camera_intrinsics, float)
        Tbc = np.asarray(frame.T_base_cam, float)
        q = np.asarray(frame.robot_state, float)
        ctr, rad = robot.sphere_centers_numeric(q)
        ctr = np.asarray(ctr, float).reshape(-1, 3)
        rad = np.asarray(rad, float).reshape(-1)

        # base -> camera -> pixel
        Tcb = np.linalg.inv(Tbc)
        pc = (Tcb[:3, :3] @ ctr.T).T + Tcb[:3, 3]
        z = pc[:, 2]
        uv = np.full((len(pc), 2), np.nan)
        ok = z > 1e-6
        uv[ok, 0] = K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]
        uv[ok, 1] = K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]
        rpx = np.where(ok, K[0, 0] * rad / np.maximum(z, 1e-6), np.nan)

        panels.append({
            "i": i, "t_step": int(run.steps[i].t_step),
            "rgb": np.asarray(run.steps[i].images["cam_high"]),
            "depth": depth, "uv": uv, "rpx": rpx, "z": z,
            "in_view": int(np.sum(ok & (uv[:, 0] > 0) & (uv[:, 0] < depth.shape[1])
                                  & (uv[:, 1] > 0) & (uv[:, 1] < depth.shape[0]))),
        })
        if i == FRAMES[-2]:
            layout["spheres_xy"] = ctr[:, :2].tolist()

    for c in cams:
        layout["cams"][c] = np.asarray(scene.T_base_cam(c), float)[:3, 3].tolist()

    fig = plt.figure(figsize=(15.5, 9.2))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.25, 1.0, 1.0], hspace=0.30, wspace=0.14)

    # --- 배치도: 어디를 어느 방향에서 보는가 (규칙 A) ---
    axl = fig.add_subplot(gs[0, :2])
    sxy = np.asarray(layout["spheres_xy"], float)
    axl.scatter(sxy[:, 0], sxy[:, 1], s=14, color=C[0], alpha=0.8,
                label=T("제약 구 120 (양팔), 프레임 10", "constraint spheres 120 (arms), frame 10"))
    for k, (c, p) in enumerate(layout["cams"].items()):
        axl.scatter([p[0]], [p[1]], s=90, marker="^", color=C[k + 1], zorder=5)
        axl.annotate(c, (p[0], p[1]), textcoords="offset points", xytext=(6, 5),
                     fontsize=8, color=figstyle.INK_2)
    axl.set_xlabel("base x [m]"); axl.set_ylabel("base y [m]")
    axl.set_title(T("배치도 — base 프레임 위에서 내려다본 것. 아래 영상은 zed_left 시점",
                    "Layout - top-down in base frame. Images below are the zed_left view"),
                  fontsize=10, color=figstyle.INK)
    axl.legend(fontsize=7, frameon=False, loc="upper left")
    axl.set_aspect("equal", adjustable="datalim")

    axt = fig.add_subplot(gs[0, 2:]); axt.axis("off")
    meta = json.loads((pathlib.Path(_ag3s_workspace_path('')) / RECORD / "meta.json").read_text())
    lines = [
        T("기록", "record") + f": {RECORD}",
        T("정책 호출", "policy calls") + f": {meta['n_steps']}   "
        + T("이 그림은 앞 15", "this figure: first 15"),
        T("프롬프트", "prompt") + f": {meta['prompt']}",
        T("행동 폭", "action width") + ": 16   nq: %d" % meta["nq"],
        "",
        T("아래 두 줄: 정책이 실제로 본 RGB(224x224) 와 같은 자세의 depth(640x480).",
          "Below: the RGB the policy saw (224x224) and the depth (640x480) at the same pose."),
        T("depth 위의 원은 제약에 쓰이는 로봇 구를 실제 좌표에서 투영한 것이다.",
          "Circles on depth are the constraint spheres projected from real coordinates."),
    ]
    axt.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9,
             color=figstyle.INK, family=matplotlib.rcParams["font.family"][0])

    for k, p in enumerate(panels):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(p["rgb"]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"cam_high  i={p['i']}  t={p['t_step']}", fontsize=9, color=figstyle.INK)

        ax = fig.add_subplot(gs[2, k])
        d = np.where(np.isfinite(p["depth"]), p["depth"], np.nan)
        # MuJoCo depth 는 far plane 에서 수백 m 를 돌려준다 — 퍼센타일로 자르면 전부 뭉개진다.
        # 씬이 실제로 있는 0.2~2.0 m 로 고정한다.
        im = ax.imshow(d, cmap=figstyle.sequential_cmap(), vmin=0.2, vmax=2.0)
        m = np.isfinite(p["uv"][:, 0])
        ax.scatter(p["uv"][m, 0], p["uv"][m, 1], s=np.clip(p["rpx"][m], 1, 40) ** 2 * 0.05,
                   facecolors="none", edgecolors=C[1], linewidths=0.7)
        ax.set_xlim(0, d.shape[1]); ax.set_ylim(d.shape[0], 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(T("depth + 로봇 구  화면 안 ", "depth + spheres  in view ")
                     + f"{p['in_view']}/120", fontsize=9, color=figstyle.INK)
        if k == 3:
            cb = fig.colorbar(im, ax=ax, fraction=0.045)
            cb.set_label("m", fontsize=8)
            cb.ax.tick_params(labelsize=7)

    figstyle.style_axes(fig, np.array([axl]))
    fig.suptitle(T("R 스모크 — 16D 긴 기록의 실제 씬", "R smoke - the 16D long record, actual scene"),
                 fontsize=13, color=figstyle.INK)
    fig.savefig(OUT / "r-smoke-scene.png", dpi=130, facecolor=figstyle.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    return {"record": RECORD, "frames": list(FRAMES),
            "spheres_in_view_zed_left": {str(p["i"]): p["in_view"] for p in panels},
            "camera_positions_base": layout["cams"]}


# ------------------------------------------------------------------ 2. 그래프
def graph_figure() -> dict:
    cur = json.loads((VER / "R2_curobo_attn.json").read_text())["frames"]
    leg = json.loads((VER / "R2_legacy_attn.json").read_text())["frames"]
    rep = [json.loads((VER / f"R2_repeat_{k}.json").read_text())["frames"] for k in (1, 2, 3)]
    rep.append(json.loads((VER / "R4_dump_base.json").read_text())["frames"])
    base = json.loads((VER / "baseline.json").read_text())["frames"]
    a1 = json.loads((VER / "R6_a1_static.json").read_text())

    lag_ms = [0, 16, 32, 48, 64, 96, 128]
    leak = [0, 0, 54, 386, 789, 963, 845]
    cam_mm = [0.0, 9.8, 18.9, 28.0, 36.1, 49.1, 53.2]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    figstyle.style_axes(fig, axes)

    ax = axes[0]
    x = np.arange(len(cur))
    ax.axhline(0, color=figstyle.GRID_INK, lw=1)
    ax.plot(x, [f["arm_before_mm"] for f in leg], "o--", ms=4, color=figstyle.INK_2,
            label=T("legacy 전", "legacy before"))
    ax.plot(x, [f["arm_after_mm"] for f in leg], "o-", ms=4, color=C[0],
            label=T("legacy 후", "legacy after"))
    ax.plot(x, [f["arm_after_mm"] for f in cur], "s-", ms=4, color=C[1],
            label=T("cuRobo 후", "cuRobo after"))
    ax.set_xlabel(T("청크 (프레임)", "chunk (frame)")); ax.set_ylabel(T("팔 여유거리 [mm]", "arm clearance [mm]"))
    band = np.array([[f["arm_after_mm"] for f in r] for r in rep] +
                    [[f["arm_after_mm"] for f in leg]], float)
    ax.fill_between(x, band.min(0), band.max(0), color=C[0], alpha=0.18, lw=0,
                    label=T("legacy 같은 명령 5 회 폭", "legacy, 5 identical runs"))
    ax.set_title(T("R2 — 같은 16D 기록, 두 backend (실측 attention)",
                   "R2 - same 16D record, two backends (measured attention)"),
                 fontsize=11, color=figstyle.INK)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.plot(lag_ms, leak, "o-", color=C[1], ms=5)
    ax.axvline(100, color=C[7], ls=":", lw=1.2)
    ax.annotate(T("설정 한계 100 ms", "config limit 100 ms"), (100, max(leak) * 0.55),
                fontsize=8, color=C[7], rotation=90, ha="right")
    ax.axvline(32, color=C[0], ls="--", lw=1.2)
    ax.annotate(T("16D 누수 시작 32 ms", "16D onset 32 ms"), (32, max(leak) * 0.8),
                fontsize=8, color=C[0], ha="left")
    ax.axvline(16, color=figstyle.INK_2, ls="--", lw=1.0)
    ax.annotate(T("14D 값 16 ms", "14D value 16 ms"), (16, max(leak) * 0.30),
                fontsize=8, color=figstyle.INK_2, ha="right")
    ax2 = ax.twinx(); ax2.plot(lag_ms, cam_mm, "^:", color=figstyle.INK_2, ms=4)
    ax2.set_ylabel(T("카메라 이동 [mm]", "camera motion [mm]"), fontsize=9, color=figstyle.INK_2)
    ax2.tick_params(colors=figstyle.INK_2, labelsize=8)
    ax.set_xlabel(T("상태 지연 [ms]", "state lag [ms]")); ax.set_ylabel(T("로봇 누수 점", "robot leak points"))
    ax.set_title(T("R7 — 상태 지연이 몇 ms 부터 새나", "R7 - where state lag starts leaking"),
                 fontsize=11, color=figstyle.INK)

    ax = axes[2]
    groups = ["arms 120", "all 194"]
    off = [a1["by_links"]["arms"]["off"]["max_optimism_mm"], a1["by_links"]["all"]["off"]["max_optimism_mm"]]
    offmed = [a1["by_links"]["arms"]["off"]["median_mm"], a1["by_links"]["all"]["off"]["median_mm"]]
    on = [a1["by_links"]["arms"]["on"]["max_optimism_mm"], a1["by_links"]["all"]["on"]["max_optimism_mm"]]
    w = 0.26; xi = np.arange(2)
    ax.bar(xi - w, off, w, color=C[1], label=T("채널 끔 — 최대", "channel off - max"))
    ax.bar(xi, offmed, w, color=C[3], label=T("채널 끔 — 중앙", "channel off - median"))
    ax.bar(xi + w, on, w, color=C[0], label=T("채널 켬 — 최대", "channel on - max"))
    for j, v in enumerate(off):
        ax.annotate(f"+{v:.1f}", (xi[j] - w, v), ha="center", va="bottom", fontsize=8, color=figstyle.INK)
    for j, v in enumerate(on):
        ax.annotate(f"{v:.1f}", (xi[j] + w, v), ha="center", va="bottom", fontsize=8, color=figstyle.INK)
    ax.set_xticks(xi); ax.set_xticklabels(groups)
    ax.set_ylabel(T("낙관 [mm]", "optimism [mm]"))
    ax.set_title(T("R6(N2) — 해석적 채널 낙관, 16D 기록", "R6(N2) - analytic channel optimism, 16D"),
                 fontsize=11, color=figstyle.INK)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(T("R 스모크 — 16D 에서 나온 수치", "R smoke - numbers measured on 16D"),
                 fontsize=13, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "r-smoke-trend.png", dpi=130, facecolor=figstyle.SURFACE, bbox_inches="tight")
    plt.close(fig)
    return {
        "r2_curobo_arm_after_mm": [f["arm_after_mm"] for f in cur],
        "r2_legacy_arm_after_mm": [f["arm_after_mm"] for f in leg],
        "r2_legacy_band_min_mm": band.min(0).tolist(),
        "r2_legacy_band_max_mm": band.max(0).tolist(),
        "r2_legacy_feasible_over_5_runs": [12, 7, 7, 12, 12],
        "r2_attention": "measured (attention_16d_long.npz)",
        "r2_legacy_arm_before_mm": [f["arm_before_mm"] for f in leg],
        "baseline_arm_after_mm": [f["arm_after_mm"] for f in base],
        "r7_lag_ms": lag_ms, "r7_leak_points": leak, "r7_camera_motion_mm": cam_mm,
        "r6_optimism": a1["by_links"],
    }


# ------------------------------------------------------------------ 3. 표
ROWS = [
    ("R1", T("I2 — legacy 대 cuRobo 대조", "I2 - legacy vs cuRobo"),
     T("근접 띠 2 mm 안 일치, 완화 0 건", "near band agrees <2mm, 0 relaxations"),
     "broken", T("죽었다 (pipeline.py:906)", "died (pipeline.py:906)")),
    ("R2", T("cuRobo 별도 기준선", "cuRobo standalone baseline"),
     T("15청크 해소 11 / feasible 9, violated 6", "15 chunks resolved 11 / feasible 9, violated 6"),
     "runs", T("cuRobo 12/15 시작 · 해소 4 · feasible 7 / legacy 5 회 feasible 12·7·7·12·12",
               "cuRobo start 12/15, resolved 4, feasible 7 / legacy x5: 12,7,7,12,12")),
    ("R3", T("쥔 물체 부호 교정 문턱", "attached sign threshold"),
     T("1.5 복셀", "1.5 voxels"), "runs",
     T("3.0 복셀 (후보 상한)", "3.0 voxels (top of candidates)")),
    ("R4", T("C5 — 거친 20 mm 계층의 낙관", "C5 - coarse 20mm optimism"),
     T("+7.56 mm / 2계층 -0.49 mm", "+7.56 mm / two-tier -0.49 mm"),
     "needs-outpath", T("미세 계층 0/15 프레임 (target 없음) + 출력 경로 박힘 :32",
                        "fine tier 0/15 frames (no target) + out path hardcoded :32")),
    ("R5", T("C3 — 미세 창 경계 불연속", "C3 - fine window boundary"),
     T("중앙 +4.5 / 최대 +41.6 mm", "median +4.5 / max +41.6 mm"),
     "needs-arg", T("입력이 코드에 박힘 (verify_two_tier.py:7) — 미세 계층도 없다",
                    "input hardcoded (verify_two_tier.py:7); no fine tier either")),
    ("R6", T("N1 · N2 · F20 잠복 값", "N1 / N2 / F20 dormant"),
     T("185.6 mm · 낙관 0 · -6.7 mm", "185.6 mm / optimism 0 / -6.7 mm"),
     "runs", T("N2: arms 낙관 +258.2 → 0 (N1·F20 은 다른 스크립트)",
               "N2: arms optimism +258.2 -> 0 (N1/F20 elsewhere)")),
    ("R7", T("F14 — 상태 지연 피해 시작점", "F14 - state lag onset"),
     T("16 ms 에서 851 점", "16 ms, 851 points"), "runs",
     T("32 ms 에서 54 점", "32 ms, 54 points")),
]
CLASS_COLOR = {"runs": C[2], "broken": C[7], "needs-arg": C[3], "needs-outpath": C[6]}


def table_figure() -> dict:
    fig, ax = plt.subplots(figsize=(16.0, 5.6))
    fig.patch.set_facecolor(figstyle.SURFACE)
    ax.axis("off")
    head = ["#", T("무엇을 재나", "what"), T("14D 값", "14D value"),
            T("분류", "class"), T("16D 스모크에서 나온 것", "what the 16D smoke produced")]
    cells = [[r[0], r[1], r[2], r[3], r[4]] for r in ROWS]
    tb = ax.table(cellText=cells, colLabels=head, loc="center",
                  colWidths=[0.035, 0.215, 0.215, 0.085, 0.45])
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 2.0)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK); cell.set_linewidth(0.6)
        cell.get_text().set_color(figstyle.INK)
        if r == 0:
            cell.set_facecolor("#efeeea"); cell.get_text().set_color(figstyle.INK)
        else:
            cell.set_facecolor(figstyle.SURFACE)
            if c == 3:
                cell.set_facecolor(CLASS_COLOR[ROWS[r - 1][3]] + "33")
        cell.get_text().set_ha("left" if c in (1, 2, 4) else "center")
    ax.set_title(T("R — 재측정 대기 7 건, 16D 스모크 분류  (runs 4 · broken 1 · needs-arg 1 · needs-outpath 1)",
                   "R - 7 pending re-measurements, 16D smoke classification "
                   "(runs 4 / broken 1 / needs-arg 1 / needs-outpath 1)"),
                 fontsize=12, color=figstyle.INK, pad=18)
    fig.savefig(OUT / "r-smoke-table.png", dpi=130, facecolor=figstyle.SURFACE, bbox_inches="tight")
    plt.close(fig)
    return {"rows": [{"id": r[0], "class": r[3]} for r in ROWS]}


if __name__ == "__main__":
    print("korean font:", KO)
    data = {"scene": scene_figure(), "graph": graph_figure(), "table": table_figure()}
    for name, key in (("r-smoke-scene", "scene"), ("r-smoke-trend", "graph"),
                      ("r-smoke-table", "table")):
        (OUT / f"{name}.json").write_text(json.dumps(data[key], indent=2, ensure_ascii=False))
        print("wrote", OUT / f"{name}.png", "+ .json")
