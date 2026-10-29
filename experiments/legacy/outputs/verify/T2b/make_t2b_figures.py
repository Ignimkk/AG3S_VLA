"""T2-b figure 3종 — 실제 씬(ESDF 단면 + 렌더) · 그래프(프레임 추이) · 표.

`outputs/verify/T2b/t2b_attached_residual.json` + `t2b_summary.json` + `slices.npz` 만 읽는다.
`figures/t2/` 의 것을 덮지 않는다 — 전부 `figures/t2b/` 아래 새 이름.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T2b/make_t2b_figures.py
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
from matplotlib.patches import Circle, Rectangle

from AG3S.ag3s.experiments.common import figstyle

OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t2b'))
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T2b/t2b_attached_residual.json'))
SUM = pathlib.Path(_ag3s_workspace_path('outputs/verify/T2b/t2b_summary.json'))
NPZ = pathlib.Path(_ag3s_workspace_path('outputs/verify/T2b/slices.npz'))

ATTACH, CARVE_ON, TRANSPORT, DETACH = 19, 20, 21, 32
SCENE_FRAMES = (0, 25, 47)
PROBE_R = 0.050


def save(fig, name: str, sidecar: dict) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    if png.exists():
        raise SystemExit(f"{png} 이 이미 있다 — 덮지 않는다. 새 이름을 정한다.")
    fig.savefig(png, dpi=150, bbox_inches="tight", facecolor=figstyle.SURFACE)
    (OUT / f"{name}.json").write_text(json.dumps(sidecar, indent=1, ensure_ascii=False))
    plt.close(fig)
    print(f"wrote {png}")
    return png


def fig_scene(z, S, kor: bool) -> None:
    """실제 씬 — 배치도(어디를 어떻게 잘랐나) 먼저, 그다음 렌더, 그다음 같은 프레임의 ESDF 단면."""
    apple0 = z["probe_apple_frame0"]
    pear = z["probe_pear_static"]
    zcut = float(z[f"slice_z_{SCENE_FRAMES[0]}"][0])

    fig = plt.figure(figsize=(15.5, 13.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.78, 1.0, 1.05], hspace=0.30, wspace=0.16)

    # --- 배치도: 옆에서 본 단면 위치 -----------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.axhline(zcut, color=figstyle.CATEGORICAL[1], lw=2.0, zorder=3)
    ax.text(0.86, zcut + 0.012,
            (f"수평 절단면  z = {zcut:.3f} m (base)" if kor else
             f"horizontal cut  z = {zcut:.3f} m (base)"),
            color=figstyle.CATEGORICAL[1], fontsize=11, ha="right")
    # table_top geom: base 프레임 중심 z = 0.80, 반높이 0.02 -> 윗면 0.82. y 는 -0.5~0.5
    ax.add_patch(Rectangle((-0.5, 0.78), 1.0, 0.04, facecolor=figstyle.GRID_INK,
                           edgecolor="none", alpha=0.55, zorder=1))
    ax.text(-0.49, 0.766, ("테이블 윗면 z = 0.820 m" if kor else "table top z = 0.820 m"),
            color=figstyle.INK_2, fontsize=10)
    for (name, p, c) in (("apple @ frame 0", apple0, figstyle.CATEGORICAL[0]),
                         ("pear (static)", pear, figstyle.CATEGORICAL[2])):
        ax.add_patch(Circle((p[1], p[2]), PROBE_R, facecolor=c, alpha=0.18,
                            edgecolor=c, lw=1.6, zorder=2))
        ax.plot([p[1]], [p[2]], "o", color=c, ms=6, zorder=4)
        ax.text(p[1], p[2] + PROBE_R + 0.010, name, color=c, fontsize=10, ha="center")
    ax.set_xlabel("y (m, base)")
    ax.set_ylabel("z (m, base)")
    ax.set_xlim(-0.52, 0.52)
    ax.set_ylim(0.74, 0.99)
    ax.set_aspect("equal")
    ax.set_title(("배치도 — 아래 단면은 이 평면을 위에서 내려다본 것이다. 탐침 구 반지름 50 mm"
                  if kor else
                  "layout — the slices below look down on this plane. probe sphere radius 50 mm"),
                 fontsize=12, loc="left")

    # --- 렌더 ----------------------------------------------------------------------------
    for col, i in enumerate(SCENE_FRAMES):
        axi = fig.add_subplot(gs[1, col])
        axi.imshow(z[f"rgb_{i}"])
        axi.set_xticks([]); axi.set_yticks([])
        dist = S["apple_probe_series"]["apple_body_dist_from_probe_mm"][i]
        axi.set_title((f"zed_left, frame {i} — 사과가 탐침에서 {dist:.0f} mm" if kor else
                       f"zed_left, frame {i} — apple {dist:.0f} mm from probe"), fontsize=11)

    # --- ESDF 단면 -----------------------------------------------------------------------
    vmax = 0.12
    for col, i in enumerate(SCENE_FRAMES):
        axi = fig.add_subplot(gs[2, col])
        sl = z[f"slice_{i}"]
        ext = z[f"extent_{i}"]
        # imshow 는 (row=y, col=x) 를 원하므로 전치. x 는 위로, y 는 왼쪽으로 가는 base 프레임을
        # 그대로 두고 x 를 세로축에 둔다.
        im = axi.imshow(sl.T, origin="lower", cmap=figstyle.sequential_cmap(),
                        vmin=-0.05, vmax=vmax,
                        extent=(ext[0], ext[1], ext[2], ext[3]), aspect="equal")
        # 점유(거리 <= 0) 복셀 테두리
        axi.contour(np.linspace(ext[0], ext[1], sl.shape[0]),
                    np.linspace(ext[2], ext[3], sl.shape[1]),
                    sl.T, levels=[0.0], colors=[figstyle.INK], linewidths=1.0)
        for (p, c) in ((apple0, figstyle.CATEGORICAL[0]), (pear, figstyle.CATEGORICAL[2])):
            axi.add_patch(Circle((p[0], p[1]), PROBE_R, facecolor="none", edgecolor=c,
                                 lw=2.0, zorder=5))
        ap = z[f"apple_{i}"]
        axi.plot([ap[0]], [ap[1]], "*", color=figstyle.CATEGORICAL[3], ms=14, zorder=6)
        key = f"attached_pts_{i}"
        if key in z.files:
            pts = z[key]
            axi.plot(pts[:, 0], pts[:, 1], ".", color=figstyle.CATEGORICAL[7], ms=5, zorder=7)
        up = S["apple_probe_series"]["n_occupied_upper_hemisphere"][i]
        cen = S["apple_probe_series"]["centre_distance_mm"][i]
        axi.set_xlabel("x (m, base)")
        if col == 0:
            axi.set_ylabel("y (m, base)")
        axi.set_title((f"frame {i} — 사과 탐침 위반구 {up}, 중심 {cen:+.1f} mm" if kor else
                       f"frame {i} — apple probe upper {up}, centre {cen:+.1f} mm"), fontsize=11)
        if col == 2:
            cb = fig.colorbar(im, ax=axi, fraction=0.046, pad=0.03)
            cb.set_label("ESDF (m)" if not kor else "ESDF 거리 (m)", fontsize=10)

    fig.suptitle(("T2-b — 사과가 떠난 자리(파랑 원)와 원래 거기 있는 배(초록 원). "
                  "검은 선 = 점유 경계, 별 = 사과 참값, 보라 점 = attached point"
                  if kor else
                  "T2-b — the site the apple left (blue) vs the pear that never moved (green). "
                  "black = occupancy boundary, star = apple ground truth, purple = attached points"),
                 fontsize=13, y=0.935)
    save(fig, "t2b-scene-residual-slices", {
        "slice_frames": list(SCENE_FRAMES),
        "slice_z_base_m": zcut,
        "probe_radius_m": PROBE_R,
        "probe_apple_frame0_base": [float(v) for v in apple0],
        "probe_pear_static_base": [float(v) for v in pear],
        "apple_body_dist_from_probe_mm": {str(i): S["apple_probe_series"]["apple_body_dist_from_probe_mm"][i]
                                          for i in SCENE_FRAMES},
        "apple_probe_n_occupied_upper": {str(i): S["apple_probe_series"]["n_occupied_upper_hemisphere"][i]
                                         for i in SCENE_FRAMES},
        "apple_probe_centre_distance_mm": {str(i): S["apple_probe_series"]["centre_distance_mm"][i]
                                           for i in SCENE_FRAMES},
        "source": "outputs/verify/T2b/slices.npz, outputs/verify/T2b/t2b_summary.json",
    })


def fig_graph(raw, S, kor: bool) -> None:
    fr = raw["frames"]
    idx = np.array([f["i"] for f in fr])
    rows = {r["i"]: r for r in S["held_object_in_obstacle_field_per_frame"]}
    ai = S["apple_probe_series"]
    pi = S["pear_probe_series"]

    fig, axes = plt.subplots(3, 1, figsize=(13.0, 12.0), sharex=True)
    figstyle.style_axes(fig, axes)

    def marks(ax):
        ax.axvspan(ATTACH, DETACH, color=figstyle.GRID_INK, alpha=0.30, zorder=0)
        for f, lab, c in ((ATTACH, "attach", figstyle.CATEGORICAL[1]),
                          (TRANSPORT, "transport", figstyle.CATEGORICAL[3]),
                          (DETACH, "detach", figstyle.CATEGORICAL[4])):
            ax.axvline(f, color=c, lw=1.2, ls="--", zorder=1)

    # (a) 쥔 물체 점이 obstacle field 에서 무슨 값을 받나
    ax = axes[0]
    hi = sorted(rows)
    main_min = [rows[i]["main_min_mm"] for i in hi]
    ctrl_min = [rows[i]["ctrl_min_mm"] for i in hi]
    ax.axhline(0.0, color=figstyle.INK, lw=1.0, zorder=2)
    ax.plot(hi, main_min, "-o", color=figstyle.CATEGORICAL[0], ms=5,
            label=("production (attach 됨) 최소" if kor else "production (attached) min"))
    ax.plot(hi, ctrl_min, "-s", color=figstyle.CATEGORICAL[1], ms=5,
            label=("대조: attach 없음 최소" if kor else "control: no attach, min"))
    for i in hi:
        n = rows[i]["main_n_nonpos"]
        if n:
            ax.annotate(f"{n}/{rows[i]['n']}", (i, rows[i]["main_min_mm"]),
                        textcoords="offset points", xytext=(4, -14),
                        color=figstyle.CATEGORICAL[0], fontsize=10)
    ax2 = ax.twinx()
    ax2.bar(hi, [rows[i]["carved"] for i in hi], width=0.55,
            color=figstyle.CATEGORICAL[2], alpha=0.45, zorder=0,
            label=("carve 된 복셀 수" if kor else "voxels carved"))
    ax2.set_ylabel(("carve 복셀 수" if kor else "voxels carved"),
                   color=figstyle.CATEGORICAL[2], fontsize=10)
    ax2.set_ylim(0, 30)
    marks(ax)
    ax.set_ylabel(("쥔 물체 점의 ESDF 거리 (mm)" if kor else "ESDF at held-object points (mm)"))
    ax.annotate(("회색 띠 = HELD 구간 (frame 19~31)" if kor else "grey band = HELD (frames 19-31)"),
                (33, 132), color=figstyle.INK_2, fontsize=10)
    ax.set_title(("(a) 쥔 물체 점 18 개가 obstacle field 에서 받는 최소 거리. "
                  "음수 = 아직 장애물. 숫자는 거리 <= 0 인 점 수"
                  if kor else
                  "(a) min ESDF over the 18 held-object points. negative = still an obstacle. "
                  "annotation = how many points are <= 0"), fontsize=12, loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=10)

    # (b) 사과 탐침
    ax = axes[1]
    ax.bar(idx, ai["n_occupied_upper_hemisphere"], width=0.7,
           color=figstyle.CATEGORICAL[0], alpha=0.55,
           label=("사과 자리: 위쪽 반구 점유 복셀" if kor else "apple site: occupied voxels, upper half"))
    ax.plot(idx, pi["n_occupied_upper_hemisphere"], "-", color=figstyle.CATEGORICAL[2], lw=2.0,
            label=("배 자리(움직이지 않음): 같은 수" if kor else "pear site (never moved): same count"))
    axb = ax.twinx()
    axb.plot(idx, ai["centre_distance_mm"], "-", color=figstyle.CATEGORICAL[1], lw=1.6,
             label=("사과 자리 중심 ESDF (mm)" if kor else "apple site centre ESDF (mm)"))
    axb.axhline(0.0, color=figstyle.CATEGORICAL[1], lw=0.8, ls=":")
    axb.set_ylabel(("중심 ESDF (mm)" if kor else "centre ESDF (mm)"), color=figstyle.CATEGORICAL[1])
    first_zero = S["residual_after_transport_start"]["upper_hemisphere_first_permanently_zero_frame"]
    first_pos = S["residual_after_transport_start"]["centre_first_permanently_positive_frame"]
    ax.axvline(first_zero, color=figstyle.CATEGORICAL[0], lw=1.4)
    ax.annotate((f"위쪽 반구 0 (frame {first_zero})" if kor else f"upper half = 0 (frame {first_zero})"),
                (first_zero, max(ai["n_occupied_upper_hemisphere"]) * 0.36),
                textcoords="offset points", xytext=(6, 0),
                color=figstyle.CATEGORICAL[0], fontsize=10)
    axb.axvline(first_pos, color=figstyle.CATEGORICAL[1], lw=1.4)
    axb.annotate((f"중심 양수 (frame {first_pos})" if kor else f"centre positive (frame {first_pos})"),
                 (first_pos, 0.0), textcoords="offset points", xytext=(-168, -26),
                 color=figstyle.CATEGORICAL[1], fontsize=10)
    marks(ax)
    ax.set_ylabel(("점유 복셀 수 (반지름 50 mm 구, 위쪽 반구)" if kor else
                   "occupied voxels (r = 50 mm, upper half)"))
    ax.set_title(("(b) 누적 TSDF 잔상 — 사과가 떠난 자리 대 배가 있는 자리"
                  if kor else
                  "(b) accumulated TSDF residual — the site the apple left vs where the pear sits"),
                 fontsize=12, loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=10)

    # (c) 사과 참값 거리와 전구 점유
    ax = axes[2]
    ax.plot(idx, ai["apple_body_dist_from_probe_mm"], "-", color=figstyle.CATEGORICAL[3], lw=2.0,
            label=("사과 참값의 탐침 중심 거리 (mm)" if kor else "apple ground truth distance to probe (mm)"))
    ax.axhline(PROBE_R * 1000.0, color=figstyle.CATEGORICAL[3], lw=0.9, ls=":")
    ax.annotate(("탐침 반지름 50 mm" if kor else "probe radius 50 mm"),
                (2, PROBE_R * 1000.0), textcoords="offset points", xytext=(0, 6),
                color=figstyle.CATEGORICAL[3], fontsize=10)
    axc = ax.twinx()
    axc.plot(idx, ai["n_occupied_sphere"], "-", color=figstyle.CATEGORICAL[0], lw=1.6,
             label=("사과 자리: 구 전체 점유 복셀" if kor else "apple site: occupied voxels, full sphere"))
    axc.plot(idx, pi["n_occupied_sphere"], "-", color=figstyle.CATEGORICAL[2], lw=1.6,
             label=("배 자리: 구 전체 점유 복셀" if kor else "pear site: occupied voxels, full sphere"))
    axc.set_ylabel(("구 전체 점유 복셀" if kor else "occupied voxels, full sphere"))
    axc.set_ylim(0, 55)
    marks(ax)
    ax.set_xlabel("observation frame (0..74)")
    ax.set_ylabel(("사과 참값 거리 (mm)" if kor else "apple ground truth distance (mm)"))
    ax.set_title(("(c) 구 전체로 세면 테이블 면이 늘 들어온다 — 그래서 (b) 는 위쪽 반구만 센다"
                  if kor else
                  "(c) the full sphere always contains the table top — hence the upper half in (b)"),
                 fontsize=12, loc="left")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axc.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=10)
    ax.set_xlim(-1, 75)

    fig.suptitle(("T2-b ep1807 — attach 뒤 쥔 물체가 obstacle field 에서 빠지는 시점, "
                  "그리고 떠난 자리 잔상의 지속"
                  if kor else
                  "T2-b ep1807 — when the held object leaves the obstacle field, "
                  "and how long the vacated site stays occupied"), fontsize=13, y=0.945)
    save(fig, "t2b-timeline-attached-residual", {
        "held_object_in_obstacle_field_per_frame": S["held_object_in_obstacle_field_per_frame"],
        "apple_probe_series": ai, "pear_probe_series": pi,
        "event_frames": {"attach": ATTACH, "carve_first_effective": CARVE_ON,
                         "transport_start": TRANSPORT, "detach": DETACH},
        "residual_after_transport_start": S["residual_after_transport_start"],
        "source": "outputs/verify/T2b/t2b_attached_residual.json",
    })


def fig_table(S, kor: bool) -> None:
    r = S["residual_after_transport_start"]
    rows = [
        ("측정 1 — attach 뒤 쥔 물체" if kor else "measurement 1 - held object after attach", "", ""),
        ("attached 가 처음 not None" if kor else "attached first not None",
         f"frame {S['attached_first_frame']}", "T2 attached = frame 19"),
        ("attached 구 개수 / 반지름" if kor else "attached sphere count / radius",
         f"{S['n_spheres']} / {S['sphere_radii_mm_frame19'][0]:.2f} mm", "attached.py:212"),
        ("attached 스냅샷 점 수" if kor else "attached snapshot points",
         f"{S['n_snapshot_points']}", "point_voxel = 10 mm"),
        ("attached 점의 base 프레임 bbox" if kor else "attached points bbox in base",
         " x ".join(f"{v:.1f}" for v in S["points_in_base_extent_mm_frame19"]) + " mm",
         "attached.py:53"),
        ("쥔 점이 필드에서 거리 <= 0 인 프레임 (production)" if kor else
         "frames with held points at ESDF <= 0 (production)",
         str(S["frames_with_held_points_nonpositive_main"]), "19 에서 14/18 점, 최소 -26.23 mm"),
        ("같은 것, attach 없는 대조" if kor else "same, control without attach",
         str(S["frames_with_held_points_nonpositive_control"]), "20 에서도 14/18 점"),
        ("attached carve 가 0 이 아닌 프레임" if kor else "frames where attached carve > 0",
         json.dumps(S["carve_nonzero_frames"]), "frame 19 는 0"),
        ("cluster 구성이 대조와 다른 프레임 수" if kor else "frames where cluster mix differs from control",
         "0 / 75", "n_object/unknown/overflow 전부 동일"),
        ("held 구간 쥔 물체 참값 80 mm 안의 cluster" if kor else
         "clusters within 80 mm of held ground truth",
         "0 (frame 19~31 전부)", "최근접 비-target cluster 101.6~308.3 mm"),
        ("attached 가 None 으로 돌아온 프레임" if kor else "frame attached returns to None",
         f"frame {S['attached_none_again_frame']}", "T2 detached = frame 32"),
        ("", "", ""),
        ("측정 2 — 누적 TSDF 잔상" if kor else "measurement 2 - accumulated TSDF residual", "", ""),
        ("탐침" if kor else "probe",
         f"r = {S['probe_radius_m']*1000:.0f} mm", "apple frame0 자리 / pear 자리"),
        ("사과 참값이 탐침 밖으로" if kor else "apple ground truth leaves the probe",
         f"frame {r['first_frame_apple_body_farther_than_50mm']}", "transport_start = frame 21"),
        ("위쪽 반구 점유가 영구 0 이 되는 프레임" if kor else
         "upper half occupancy first permanently zero",
         f"frame {r['upper_hemisphere_first_permanently_zero_frame']}",
         f"frame 0 에는 12 복셀"),
        ("residual_frames_after_transport_start (위쪽 반구)" if kor else
         "residual_frames_after_transport_start (upper half)",
         f"{r['residual_frames_after_transport_start_upper_hemisphere']} / {r['frames_21_to_74_total']}",
         "frame 21~46"),
        ("탐침 중심 ESDF 가 영구 양수가 되는 프레임" if kor else
         "probe centre ESDF first permanently positive",
         f"frame {r['centre_first_permanently_positive_frame']}", "그 전까지 -19.99 ~ -28.28 mm"),
        ("residual_frames_after_transport_start (중심 음수)" if kor else
         "residual_frames_after_transport_start (centre negative)",
         f"{r['residual_frames_after_transport_start_centre_negative']} / {r['frames_21_to_74_total']}",
         "frame 21~66"),
        ("구 전체 점유가 0 이 되는 프레임" if kor else "full sphere occupancy first zero",
         "없음 (22~46 복셀)" if kor else "never (22-46 voxels)", "테이블 면이 늘 들어온다"),
        ("대조 — 배 자리 (drift 0.6 mm)" if kor else "control - pear site (drift 0.6 mm)",
         f"위쪽 반구 {S['pear_control_stable']['n_occupied_upper_range'][0]}~"
         f"{S['pear_control_stable']['n_occupied_upper_range'][1]} 복셀",
         f"중심 {S['pear_control_stable']['centre_distance_mm_range'][0]:.1f} ~ "
         f"{S['pear_control_stable']['centre_distance_mm_range'][1]:.1f} mm, 0 인 프레임 없음"),
        ("decay 설정" if kor else "decay settings",
         "time_decay 1.0 / frustum_decay 1.0", "config.py:480,484 — 감쇠 없음"),
    ]

    fig, ax = plt.subplots(figsize=(15.0, 11.0))
    figstyle.style_axes(fig, ax)
    ax.axis("off")
    head = (["항목", "값", "근거 · 비고"] if kor else ["item", "value", "basis / note"])
    tbl = ax.table(cellText=[list(r) for r in rows], colLabels=head,
                   cellLoc="left", colLoc="left", loc="upper center",
                   colWidths=[0.42, 0.24, 0.34])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1, 1.62)
    for (rr, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        cell.get_text().set_color(figstyle.INK)
        if rr == 0:
            cell.set_facecolor(figstyle.GRID_INK)
            cell.get_text().set_color(figstyle.INK)
        elif rows[rr - 1][1] == "" and rows[rr - 1][0]:
            cell.set_facecolor(figstyle.CATEGORICAL[0])
            cell.get_text().set_color("#ffffff")
    ax.set_title(("T2-b ep1807 — T2 의 미측정 두 항목. 전부 observation frame 0..74 기준"
                  if kor else
                  "T2-b ep1807 - the two unmeasured T2 items. observation frames 0..74"),
                 fontsize=13, loc="left", pad=16)
    save(fig, "t2b-numbers-table", {"rows": rows, "source": "outputs/verify/T2b/t2b_summary.json"})


def main() -> None:
    kor = figstyle.use_korean()
    raw = json.loads(RAW.read_text())
    S = json.loads(SUM.read_text())
    z = np.load(NPZ)
    fig_scene(z, S, kor)
    fig_graph(raw, S, kor)
    fig_table(S, kor)


if __name__ == "__main__":
    main()
