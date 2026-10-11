"""0930 세미나 — 모듈 설명용 16D 그림 (구버전 14D 그림 교체).

실행:  /mnt/dev/work/.venv-ag3s/bin/python benchmark/ag3s/docs/figures/ppt-0930/make_ppt_figures_16d.py
출력:  같은 폴더의 *-16d.png · 크롭 그림 · ppt-0930-figures-16d.json · 가림 처리한 mp4

새로 그리는 셋(Attention lifting · Target grounding · Distance adapter)은 전부 **한 장면**이다 —
held-out ep1807 의 첫 정책 호출 (아키텍처 그림과 같은 장면).
  RGB   : outputs/live_test/20260925_ep1807/run_0000/step_00000.npz
  나머지: outputs/live_test/20260928_t17/server_constraints/run_0000/chunk_00000.npz
          (serve_safe 가 그 호출에서 실제로 만든 attention · 점군 · seed · Target · ESDF 3 층 · 구 여유거리)
첫 호출이라 두 기록의 씬이 같다(같은 에피소드 초기 상태).
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from PIL import Image  # noqa: E402

from benchmark.ag3s.experiments.common import figstyle  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
F = ROOT / "benchmark/ag3s/docs/figures"
REC = ROOT / "outputs/live_test/20260925_ep1807/run_0000/step_00000.npz"
CHUNK = ROOT / "outputs/live_test/20260928_t17/server_constraints/run_0000/chunk_00000.npz"
C = figstyle.CATEGORICAL
INK, NOTE, SURFACE = figstyle.INK, "#8f8e89", figstyle.SURFACE
TARGET = C[7]
CMAP = figstyle.sequential_cmap()
LIGHT_D = matplotlib.colors.LinearSegmentedColormap.from_list(
    "d_light", [figstyle.SEQ_BLUE[5], figstyle.SEQ_BLUE[1], "#ffffff"])
CAMS = [("attention_head", "image_cam_high", "head (cam_high)"),
        ("attention_left_wrist", "image_cam_left_wrist", "left wrist"),
        ("attention_right_wrist", "image_cam_right_wrist", "right wrist")]
META: dict = {}

#: name → (source, crop fraction (x0, y0, x1, y1)) · 여러 조각을 세로로 붙이려면 list
#: mask — 원본 비율 좌표 상자를 배경색으로 덮는다 (영어 진단 범례 · 발견 문구)
MASKS = {
    "tg-pregrasp": [(0.03, 0.68, 0.15, 0.93)],
    "esdf-voxels3d": [(0.01, 0.13, 0.15, 0.30), (0.545, 0.13, 0.685, 0.30)],
}
CROPS = {
    "rm-scene-16d": (F / "live-test/i1-venv-field.png", [(0, 0.055, 1, 0.505)]),
    "tg-pregrasp": (F / "t30/t30-scene-pregrasp-E3b-E3b_ep1807_r1.png",
                    [(0.02, 0.125, 0.335, 0.99), (0.70, 0.235, 0.995, 0.815)]),   # 가로로 붙인다
    "esdf-voxels3d": (F / "t31/t31-d4-voxels3d-E3b_ep1807_r1-t240.png", [(0, 0.08, 1, 1)]),
    "trajopt-solve": (F / "to-design-solve.png", [(0, 0.065, 1, 1)]),
    "future-oscillation": (F / "t31/t31-d1-oscillation-timeline.png",
                           [(0, 0.050, 1, 0.200), (0, 0.365, 1, 0.523)]),  # 세로로 붙인다
}
HORIZONTAL = {"tg-pregrasp"}


def crops():
    for name, (src, boxes) in CROPS.items():
        im = Image.open(src).convert("RGB")
        W, H = im.size
        from PIL import ImageDraw
        for (a, b, c, d) in MASKS.get(name, []):
            ImageDraw.Draw(im).rectangle([a * W, b * H, c * W, d * H], fill=im.getpixel((3, 3)))  # 그림 배경색
        parts = [im.crop((int(a * W), int(b * H), int(c * W), int(d * H))) for a, b, c, d in boxes]
        if name in HORIZONTAL:
            h = max(p.height for p in parts)
            parts = [p.resize((int(p.width * h / p.height), h)) for p in parts]
            out = Image.new("RGB", (sum(p.width for p in parts) + 20 * (len(parts) - 1), h), SURFACE)
            x = 0
            for p in parts:
                out.paste(p, (x, 0))
                x += p.width + 20
        else:
            w = max(p.width for p in parts)
            out = Image.new("RGB", (w, sum(p.height for p in parts) + 16 * (len(parts) - 1)), SURFACE)
            y = 0
            for p in parts:
                out.paste(p, (0, y))
                y += p.height + 16
        out.save(OUT / f"{name}.png")
        META[name] = {"src": str(src.relative_to(ROOT)), "crop": boxes, "size": out.size}


def load():
    return np.load(REC, allow_pickle=True), np.load(CHUNK, allow_pickle=True)


def top_axes(ax, title=None):
    """로봇 뒤에서 내려다본 top view — 세로 = x(앞), 가로 = y(+y 가 왼쪽)."""
    ax.set_xlim(0.45, -0.40)
    ax.set_ylim(0.35, 0.85)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c8c7c2")
    if title:
        ax.set_title(title, fontsize=11, color=INK, pad=5)


def crop_scene(P):
    return (P[:, 0] > 0.35) & (P[:, 0] < 0.85) & (P[:, 1] > -0.40) & (P[:, 1] < 0.45) & \
           (P[:, 2] > 0.78) & (P[:, 2] < 1.16)


def fig_attention(rec, ch):
    """16×16 attention (3 camera) → RGB overlay → 3D 점마다 attention."""
    fig = plt.figure(figsize=(12.0, 4.9), facecolor=SURFACE)
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1.75], wspace=0.08, hspace=0.30,
                  left=0.01, right=0.99, top=0.86, bottom=0.02)
    gamma = matplotlib.colors.PowerNorm(0.45, 0, 1)
    for i, (akey, rkey, name) in enumerate(CAMS):
        a = np.asarray(ch[akey], np.float64)
        an = (a - a.min()) / max(a.max() - a.min(), 1e-12)
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(an, cmap=CMAP, norm=gamma, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name, fontsize=11, color=INK, pad=4)
        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(rec[rkey], extent=(0, 224, 224, 0))
        ax2.imshow(np.ma.masked_less(an ** 0.45, 0.25), extent=(0, 224, 224, 0), cmap=CMAP, alpha=0.6,
                   vmin=0, vmax=1, interpolation="bilinear")
        ax2.set_xticks([])
        ax2.set_yticks([])
    fig.text(0.005, 0.93, "① π0.5 attention — camera 마다 16 × 16", fontsize=13, color=INK, weight="bold")
    fig.text(0.005, 0.455, "② 정책이 본 RGB 에 겹침", fontsize=13, color=INK, weight="bold")
    P = np.asarray(ch["attention_cloud"], np.float64)
    av = np.asarray(ch["attention_values"], np.float64)
    m = np.flatnonzero(crop_scene(P))
    rng = np.random.default_rng(0)
    m = m[rng.permutation(len(m))[:14000]]
    m = m[np.argsort(av[m])]
    ax3 = fig.add_subplot(gs[:, 3], projection="3d")
    lo = av[m] < 0.35
    ax3.scatter(P[m[lo], 0], P[m[lo], 1], P[m[lo], 2], c="#c9c8c3", s=1.2, lw=0, depthshade=False)
    hi = m[~lo]
    ax3.scatter(P[hi, 0], P[hi, 1], P[hi, 2], c=av[hi], cmap=CMAP, s=4, lw=0, vmin=0.2, vmax=1,
                depthshade=False)
    tc = np.asarray(ch["target_centroid"], np.float64)
    ax3.view_init(elev=30, azim=-155)
    ax3.set_xlim(0.35, 0.85)
    ax3.set_ylim(-0.40, 0.45)
    ax3.set_zlim(0.78, 1.10)
    ax3.set_box_aspect((0.5, 0.85, 0.32), zoom=1.25)
    ax3.set_xlabel("x [m] 앞", fontsize=9, labelpad=-4)
    ax3.set_ylabel("y [m] 왼", fontsize=9, labelpad=-4)
    ax3.set_zlabel("z [m]", fontsize=9, labelpad=-6)
    ax3.tick_params(labelsize=7, pad=-2)
    for a in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        a.pane.set_facecolor((1, 1, 1, 0))
        a._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
    ax3.text(tc[0], tc[1], tc[2] + 0.07, "apple", fontsize=11, color=TARGET, ha="center", weight="bold")
    fig.text(0.60, 0.93, "③ 3D lifting — 점마다 자기 pixel 의 attention", fontsize=13, color=INK,
             weight="bold")
    fig.text(0.60, 0.885, "회색 = attention 낮음 · 파랑이 진할수록 높음", fontsize=10, color=NOTE)
    fig.savefig(OUT / "ap-attention-16d.png", dpi=220, facecolor=SURFACE)
    plt.close(fig)
    META["ap-attention-16d"] = {"n_points_drawn": int(len(m)), "n_points_total": int(len(P)),
                                "attention_raw_max": {k: float(np.asarray(ch[k]).max()) for k, _, _ in CAMS}}


def fig_grounding(ch):
    """① attention 상위 점(seed) → ② Target(빨강) vs Obstacle(회색) · 지지면(옅은)."""
    P = np.asarray(ch["attention_cloud"], np.float64)
    av = np.asarray(ch["attention_values"], np.float64)
    seeds = np.asarray(ch["seed_indices"], int)
    tp = np.asarray(ch["target_points"], np.float64)
    sup = np.asarray(ch["support_mask"], bool)
    m = crop_scene(P)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), facecolor=SURFACE,
                             gridspec_kw=dict(wspace=0.06, left=0.01, right=0.99, top=0.84, bottom=0.03))
    a0, a1, a2 = axes
    top_axes(a0, "① attention 이 높은 점")
    idx = np.flatnonzero(m)
    idx = idx[np.argsort(av[idx])]
    a0.scatter(P[idx, 1], P[idx, 0], c="#c9c8c3", s=1.6, lw=0)
    hi = idx[av[idx] >= 0.35]
    a0.scatter(P[hi, 1], P[hi, 0], c=av[hi], cmap=CMAP, s=4, lw=0, vmin=0.2, vmax=1)
    top_axes(a1, f"② seed — attention 상위 점 ({len(seeds)} 개)")
    a1.scatter(P[idx, 1], P[idx, 0], c="#c9c8c3", s=1.6, lw=0)
    sm = seeds[m[seeds]]
    a1.scatter(P[sm, 1], P[sm, 0], c=C[3], s=9, lw=0)
    top_axes(a2, "③ 3D region growing → Target vs Obstacle")
    obst = idx[~sup[idx]]
    supi = idx[sup[idx]]
    a2.scatter(P[supi, 1], P[supi, 0], c="#e4e3de", s=1.4, lw=0)
    a2.scatter(P[obst, 1], P[obst, 0], c="#6f6e69", s=1.8, lw=0)
    a2.scatter(tp[:, 1], tp[:, 0], c=TARGET, s=9, lw=0)
    from matplotlib.lines import Line2D
    a2.legend(handles=[Line2D([], [], marker="o", ls="", color=TARGET, label=f"Target ({len(tp)} 점)"),
                       Line2D([], [], marker="o", ls="", color="#8f8e89", label="Obstacle"),
                       Line2D([], [], marker="o", ls="", color="#d9d8d3", label="지지면(테이블)")],
              loc="upper right", fontsize=9, frameon=True, facecolor="white", edgecolor="none",
              handletextpad=0.2, borderpad=0.3)
    fig.text(0.005, 0.93, "top view (로봇 뒤에서 내려다봄)", fontsize=10, color=NOTE)
    fig.savefig(OUT / "tg-grounding-16d.png", dpi=220, facecolor=SURFACE)
    plt.close(fig)
    META["tg-grounding-16d"] = {"n_seeds": int(len(seeds)), "n_target_points": int(len(tp)),
                                "n_support_points": int(sup.sum()),
                                "target_centroid_m": np.asarray(ch["target_centroid"]).round(4).tolist()}


def _trilinear(grid, origin, vs, p):
    """grid 값 · 원점(칸 중심) · voxel → 점 p 에서 삼선형 보간."""
    g = (p - origin) / vs
    i0 = np.floor(g).astype(int)
    f = g - i0
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1]) * (f[2] if dz else 1 - f[2])
                ii = np.clip(i0 + (dx, dy, dz), 0, np.array(grid.shape) - 1)
                out += w * grid[tuple(ii)]
    return out


def fig_adapter(ch):
    """① 구 하나의 d · r · margin → 여유거리  ② ESDF 단면 위 구와 ∇d  ③ 구 전체 여유거리."""
    D = np.asarray(ch["esdf_distance"], np.float64)
    o = np.asarray(ch["esdf_origin"], np.float64)
    vs = float(ch["esdf_voxel_size"])
    sc = np.asarray(ch["sphere_centres"], np.float64)
    sr = np.asarray(ch["sphere_radii"], np.float64)
    clr = np.asarray(ch["clearance"], np.float64)[0]            # 계획 첫 스텝, [m]
    summ = json.loads(str(ch["summary_json"]))
    names = summ["sphere_link_names"]
    gx, gy, gz = np.gradient(D, vs)
    # 왼손 구가 모인 평면
    tc = np.asarray(ch["target_centroid"], np.float64)
    y0 = float(tc[1])                                          # Target(사과) 중심을 지나는 평면
    j = int(round((y0 - o[1]) / vs))
    sl = D[:, j, :]                                            # (x, z)
    nx, nz = sl.shape
    ext = (o[0] - vs / 2, o[0] + (nx - 0.5) * vs, o[2] - vs / 2, o[2] + (nz - 0.5) * vs)
    margin = float(sr.mean() * 0 + 0.010)                       # t17 서버: esdf_margin 10 mm (serve_safe.log)
    fig = plt.figure(figsize=(12.0, 4.4), facecolor=SURFACE)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.9, 1.35, 1.1], wspace=0.25, left=0.02, right=0.98,
                  top=0.84, bottom=0.14)
    # ① schematic
    a0 = fig.add_subplot(gs[0, 0])
    a0.set_xlim(-0.2, 2.6)
    a0.set_ylim(-0.35, 2.2)
    a0.set_aspect("equal")
    a0.axis("off")
    a0.fill_between([-0.2, 2.6], -0.35, 0.0, color="#d9d8d3")
    a0.text(2.5, -0.2, "물체 표면", ha="right", fontsize=10, color=INK)
    cx, cy, r, mg = 1.0, 1.35, 0.42, 0.30
    a0.add_patch(plt.Circle((cx, cy), r, fc=matplotlib.colors.to_rgba(C[0], 0.25), ec=C[0], lw=1.8))
    a0.add_patch(plt.Circle((cx, cy), r + mg, fill=False, ec=C[0], lw=1.2, ls=(0, (4, 3))))
    a0.plot(cx, cy, "o", color=INK, ms=4)
    a0.annotate("", xy=(cx, 0.0), xytext=(cx, cy), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    a0.text(cx + 0.06, 0.62, "d", fontsize=13, color=INK, style="italic")
    a0.annotate("", xy=(cx + r * 0.72, cy + r * 0.69), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-", color=C[0], lw=1.4))
    a0.text(cx + 0.36, cy + 0.14, "r", fontsize=13, color=C[0], style="italic")
    a0.text(cx + r + mg + 0.05, cy + 0.2, "margin", fontsize=10, color=C[0])
    a0.text(1.2, -0.62, "여유거리 = d − r − margin ≥ 0", ha="center", fontsize=12, color=INK, weight="bold")
    a0.set_ylim(-0.8, 2.2)
    a0.set_title("① 구 하나에 대한 제약", fontsize=12, color=INK, pad=6)
    # ② slice + spheres + gradient
    a1 = fig.add_subplot(gs[0, 1])
    a1.imshow(np.ma.masked_where(sl.T <= 0, np.clip(sl.T, 0, 0.08)), origin="lower", extent=ext,
              cmap=LIGHT_D, vmin=0, vmax=0.08, interpolation="nearest")
    a1.imshow(np.ma.masked_where(sl.T > 0, np.ones_like(sl.T)), origin="lower", extent=ext,
              cmap=matplotlib.colors.ListedColormap([INK]), interpolation="nearest")
    near = np.abs(sc[:, 1] - y0) < 0.05
    norm = matplotlib.colors.Normalize(0, 0.12)
    sphere_c = CMAP.reversed()
    for k in np.flatnonzero(near):
        a1.add_patch(plt.Circle((sc[k, 0], sc[k, 2]), sr[k], fc=sphere_c(norm(clr[k])), ec=INK, lw=0.4))
    ks = np.flatnonzero(near)
    ks = ks[np.argsort(clr[ks])][:8]
    for k in ks:
        g = np.array([_trilinear(gx, o, vs, sc[k]), _trilinear(gz, o, vs, sc[k])])
        g = g / max(np.linalg.norm(g), 1e-9) * 0.035
        a1.annotate("", xy=(sc[k, 0] + g[0], sc[k, 2] + g[1]), xytext=(sc[k, 0], sc[k, 2]),
                    arrowprops=dict(arrowstyle="-|>", color=C[2], lw=1.3, mutation_scale=9))
    a1.set_xlim(tc[0] - 0.17, tc[0] + 0.17)
    a1.set_ylim(tc[2] - 0.08, tc[2] + 0.30)
    a1.text(tc[0], tc[2] + 0.05, "Target (apple)", ha="center", fontsize=9.5, color=TARGET, weight="bold")
    a1.set_aspect("equal")
    a1.tick_params(labelsize=8)
    a1.set_xlabel("x [m] (앞)", fontsize=9)
    a1.set_ylabel("z [m] (위)", fontsize=9)
    a1.set_title(f"② 사과를 지나는 수직 단면 (fine ESDF)\n검정 = 물체 안 · 원 = gripper 구 · 초록 = ∇d",
                 fontsize=11, color=INK, pad=6)
    # ③ all spheres
    a2 = fig.add_subplot(gs[0, 2])
    order = np.argsort(clr)
    a2.bar(np.arange(len(order)), clr[order] * 1000, color=[sphere_c(norm(v)) for v in clr[order]], width=1.0)
    a2.axhline(0, color=TARGET, lw=1, ls=(0, (4, 2)))
    a2.text(len(order) * 0.02, clr.max() * 1000 * 0.9, "0 아래로 내려가면 violated", fontsize=9, color=TARGET)
    a2.set_xlabel(f"gripper 제약 구 {len(order)} 개 (여유거리 오름차순)", fontsize=9)
    a2.set_ylabel("여유거리 [mm]", fontsize=9)
    a2.tick_params(labelsize=8)
    a2.set_title(f"③ 구마다 한 행 — 최소 {clr.min() * 1000:.1f} mm", fontsize=11, color=INK, pad=6)
    figstyle.style_axes(fig, [a2])
    fig.savefig(OUT / "adapter-16d.png", dpi=220, facecolor=SURFACE)
    plt.close(fig)
    META["adapter-16d"] = {"slice_y_m": y0, "n_spheres": int(len(sc)), "n_spheres_in_slice": int(near.sum()),
                           "esdf_margin_m": margin, "margin_source": "outputs/live_test/20260928_t17/serve_safe.log",
                           "clearance_min_mm": float(clr.min() * 1000),
                           "sphere_radius_mm": [float(sr.min() * 1000), float(sr.max() * 1000)]}


def fig_esdf_tiers(ch):
    """Target 높이의 수평 단면 — coarse 20 mm · fine 5 mm · target-free 5 mm (같은 창, 같은 색)."""
    tc = np.asarray(ch["target_centroid"], np.float64)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.3), facecolor=SURFACE,
                             gridspec_kw=dict(wspace=0.05, left=0.01, right=0.99, top=0.83, bottom=0.08))
    stats = {}
    for ax, (key, okey, vkey, title) in zip(axes, [
            ("esdf_coarse_distance", "esdf_coarse_origin", "esdf_coarse_voxel_size", "coarse 20 mm — 작업공간 전체"),
            ("esdf_distance", "esdf_origin", "esdf_voxel_size", "fine 5 mm — 손 주변 창"),
            ("esdf_free_distance", "esdf_free_origin", "esdf_free_voxel_size", "target-free 5 mm — Target 제외")]):
        D = np.asarray(ch[key], np.float64)
        o = np.asarray(ch[okey], np.float64)
        vs = float(ch[vkey])
        k = int(round((tc[2] - o[2]) / vs))
        img = D[:, :, k]
        nx, ny = img.shape
        ext = (o[1] - vs / 2, o[1] + (ny - 0.5) * vs, o[0] - vs / 2, o[0] + (nx - 0.5) * vs)
        ax.imshow(np.ma.masked_where(img <= 0, np.clip(img, 0, 0.06)), origin="lower", extent=ext,
                  cmap=LIGHT_D, vmin=0, vmax=0.06, interpolation="nearest")
        ax.imshow(np.ma.masked_where(img > 0, np.ones_like(img)), origin="lower", extent=ext,
                  cmap=matplotlib.colors.ListedColormap([INK]), interpolation="nearest")
        ax.add_patch(plt.Circle((tc[1], tc[0]), 0.055, fill=False, ec=TARGET, lw=2.0, ls=(0, (4, 2))))
        ax.set_xlim(tc[1] + 0.24, tc[1] - 0.24)
        ax.set_ylim(tc[0] - 0.20, tc[0] + 0.20)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c8c7c2")
        ax.set_title(title, fontsize=12.5, color=INK, pad=6)
        stats[key] = {"voxel_m": vs, "k_index": k, "occupied_voxels_in_slice": int((img <= 0).sum())}
    axes[0].text(tc[1] + 0.20, tc[0] + 0.12, "Target", color=TARGET, fontsize=12, weight="bold")
    axes[2].text(tc[1] + 0.20, tc[0] + 0.12, "Target 만 사라짐", color=TARGET, fontsize=12, weight="bold")
    fig.text(0.5, 0.015, f"top view · Target 높이 z = {tc[2]:.3f} m 수평 단면 · 같은 40 cm 창 · 검정 = 물체 안쪽(d ≤ 0) · "
             "파랑 = 표면에 가까움", ha="center", fontsize=10, color=NOTE)
    fig.savefig(OUT / "esdf-tiers-16d.png", dpi=220, facecolor=SURFACE)
    plt.close(fig)
    META["esdf-tiers-16d"] = {"target_centroid_m": tc.round(4).tolist(), **stats}


def mp4():
    """t31 애니메이션 — 진단 문구가 든 제목 줄 두 곳을 가리고 포스터 프레임을 뽑는다."""
    src = F / "t31/t31-d4-anim-E3b_ep1807_r1.mp4"
    dst = OUT / "esdf-anim-ep1807.mp4"
    boxes = ["drawbox=x=0:y=0:w=iw:h=32:color=0xFCFCFB:t=fill",
             "drawbox=x=950:y=80:w=490:h=22:color=0xFCFCFB:t=fill"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", ",".join(boxes),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "8", "-i", str(dst), "-frames:v", "1",
                    str(OUT / "esdf-anim-ep1807-poster.png")], check=True)
    META["esdf-anim-ep1807"] = {"src": str(src.relative_to(ROOT)), "masked": boxes}


def main():
    if not figstyle.use_korean():
        raise SystemExit("CJK font 없음")
    rec, ch = load()
    crops()
    fig_attention(rec, ch)
    fig_grounding(ch)
    fig_adapter(ch)
    fig_esdf_tiers(ch)
    mp4()
    META["_scene"] = {"rgb": str(REC.relative_to(ROOT)), "chunk": str(CHUNK.relative_to(ROOT))}
    (OUT / "ppt-0930-figures-16d.json").write_text(json.dumps(META, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in META.items() if k.endswith("16d")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
