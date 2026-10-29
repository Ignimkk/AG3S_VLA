"""슬라이드 12 (Distance Field Adapter) 용 그림 두 장.

  adapter-two-maps.png   distance(p) · gradient(p) — 같은 단면을 '일반 지도' 와 'target-free 지도' 로
  margin-rules.png       물체 종류 × link 그룹 × 단계 별 요구 여유(margin) 표

실행 (.venv-ag3s 는 NFS 에서 멈춰 있어 openpi-live 로 돌린다):
  cd ${AG3S_PROJECT_ROOT} && .venv-openpi-live/bin/python AG3S/ag3s/docs/figures/ppt-0930/make_adapter_margin_figs.py

장면: held-out ep1807, t17 shadow 기록의 chunk 8 (t_step 9) — 사과가 Target 인 마지막 접근 frame,
      왼손 손가락이 사과 중심 위 약 84 mm. 거리장은 serve_safe 가 그 호출에서 실제로 만든 fine 5 mm 층과
      target-free 5 mm 층이다.
margin 값의 출처는 sidecar json 의 `sources`.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
CHUNK = ROOT / "outputs/live_test/20260928_t17/server_constraints/run_0000/chunk_00008.npz"
C = figstyle.CATEGORICAL
INK, NOTE, SURFACE = figstyle.INK, "#8f8e89", figstyle.SURFACE
TARGET = C[7]
AUTH = C[1]      # 잡는 손 (권한 link)
OTHER = C[0]     # 그 밖의 link
ARROW = C[2]
LIGHT_D = matplotlib.colors.LinearSegmentedColormap.from_list(
    "d_light", [figstyle.SEQ_BLUE[6], figstyle.SEQ_BLUE[1], "#ffffff"])


def trilinear(grid, origin, vs, p):
    g = (np.asarray(p, float) - origin) / vs
    i0 = np.floor(g).astype(int)
    f = g - i0
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1]) * (f[2] if dz else 1 - f[2])
                ii = np.clip(i0 + (dx, dy, dz), 0, np.array(grid.shape) - 1)
                out += w * grid[tuple(ii)]
    return float(out)


def fig_two_maps(meta):
    ch = np.load(CHUNK, allow_pickle=True)
    summ = json.loads(str(ch["summary_json"]))
    names = np.array(summ["sphere_link_names"])
    sc = np.asarray(ch["sphere_centres"], float)
    sr = np.asarray(ch["sphere_radii"], float)
    tc = np.asarray(ch["target_centroid"], float)
    tp = np.asarray(ch["target_points"], float)
    layers = {
        "main": (np.asarray(ch["esdf_distance"], float), np.asarray(ch["esdf_origin"], float),
                 float(ch["esdf_voxel_size"])),
        "free": (np.asarray(ch["esdf_free_distance"], float), np.asarray(ch["esdf_free_origin"], float),
                 float(ch["esdf_free_voxel_size"])),
    }
    left = np.isin(names, ["ee_left", "ee_finger_l1", "ee_finger_l2"])
    y0 = float(np.median(sc[left & (np.isin(names, ["ee_finger_l1", "ee_finger_l2"])), 1]))
    near = left & (np.abs(sc[:, 1] - y0) < 0.02)
    # 대표 구: 사과에 가장 가까운 손가락 구
    fing = np.flatnonzero(near & np.isin(names, ["ee_finger_l1", "ee_finger_l2"]))
    rep = fing[np.argmin(np.linalg.norm(sc[fing] - tc, axis=1))]

    fig = plt.figure(figsize=(12.0, 5.0), facecolor=SURFACE)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.78], wspace=0.10,
                  left=0.035, right=0.99, top=0.83, bottom=0.10)
    xlim = (tc[0] - 0.13, tc[0] + 0.13)
    zlim = (tc[2] - 0.075, tc[2] + 0.165)
    reps = {}
    for col, (key, title, sub) in enumerate([
            ("main", "① 일반 지도 (사과 포함)", "팔 · 손목 · 반대손 이 묻는 지도"),
            ("free", "② target-free 지도 (사과만 지움)", "잡는 손 (손바닥 · 손가락) 이 묻는 지도")]):
        D, o, vs = layers[key]
        gx, gy, gz = np.gradient(D, vs)
        j = int(round((y0 - o[1]) / vs))
        sl = D[:, j, :]
        nx, nz = sl.shape
        ext = (o[0] - vs / 2, o[0] + (nx - 0.5) * vs, o[2] - vs / 2, o[2] + (nz - 0.5) * vs)
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(np.ma.masked_where(sl.T <= 0, np.clip(sl.T, 0, 0.08)), origin="lower", extent=ext,
                  cmap=LIGHT_D, vmin=0, vmax=0.08, interpolation="nearest")
        ax.imshow(np.ma.masked_where(sl.T > 0, np.ones_like(sl.T)), origin="lower", extent=ext,
                  cmap=matplotlib.colors.ListedColormap([INK]), interpolation="nearest")
        # 사과 참 위치(관측 점의 x–z 투영 윤곽)
        ring = plt.Circle((tc[0], tc[2]), 0.040, fill=False, ec=TARGET, lw=1.8, ls=(0, (4, 2)))
        ax.add_patch(ring)
        for k in np.flatnonzero(near):
            ax.add_patch(plt.Circle((sc[k, 0], sc[k, 2]), sr[k], fc=matplotlib.colors.to_rgba(AUTH, 0.35),
                                    ec=AUTH, lw=0.7))
        for k in np.flatnonzero(near)[::2]:
            g = np.array([trilinear(gx, o, vs, sc[k]), trilinear(gz, o, vs, sc[k])])
            n = np.linalg.norm(g)
            if n < 1e-6:
                continue
            g = g / n * 0.028
            ax.annotate("", xy=(sc[k, 0] + g[0], sc[k, 2] + g[1]), xytext=(sc[k, 0], sc[k, 2]),
                        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.4, mutation_scale=10))
        d_rep = trilinear(D, o, vs, sc[rep])
        g_rep = np.array([trilinear(gx, o, vs, sc[rep]), trilinear(gy, o, vs, sc[rep]),
                          trilinear(gz, o, vs, sc[rep])])
        reps[key] = {"distance_m": d_rep, "gradient": (g_rep / max(np.linalg.norm(g_rep), 1e-9)).round(3).tolist(),
                     "gradient_norm": float(np.linalg.norm(g_rep))}
        ax.plot(sc[rep, 0], sc[rep, 2], "o", ms=4, color=INK, zorder=5)
        ax.text(sc[rep, 0] + 0.012, sc[rep, 2] - 0.004,
                f"distance = {d_rep * 1000:.0f} mm", fontsize=10.5, color=INK, weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.text(tc[0], tc[2] - 0.058, "사과 (Target)", ha="center", fontsize=10, color=TARGET, weight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c8c7c2")
        ax.set_title(title, fontsize=13, color=INK, weight="bold", pad=18)
        ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5,
                color=NOTE)
    # 오른쪽 설명
    axt = fig.add_subplot(gs[0, 2])
    axt.axis("off")
    lines = [
        ("distance(p)", "구 중심 p 에서 가장 가까운 표면까지 거리.\n음수 = 물체 안. 격자에서 trilinear 보간."),
        ("gradient(p)", "거리가 가장 빨리 커지는 방향 (단위 벡터)\n= 가장 가까운 표면에서 멀어지는 방향."),
        ("제약 한 줄", "distance(p) − r − margin ≥ 0\ngradient 로 '어느 관절을 움직이면\n여유가 커지는지' 를 계산한다."),
        ("두 지도", "같은 위치라도 ① 에서는 사과가 가장 가까워\n위로 밀어 올리고, ② 에서는 사과가 없어\n손가락이 사과로 내려갈 수 있다."),
    ]
    y = 0.98
    for head, body in lines:
        axt.text(0.0, y, head, fontsize=12, weight="bold", color=INK, va="top", transform=axt.transAxes)
        axt.text(0.0, y - 0.065, body, fontsize=10.2, color=NOTE, va="top", transform=axt.transAxes,
                 linespacing=1.35)
        y -= 0.07 + 0.058 * (body.count("\n") + 1) + 0.045
    fig.text(0.035, 0.035,
             f"ep1807 · 사과에 접근하는 frame (t = 9) · 왼손을 지나는 수직 x–z 단면 (y = {y0:.3f} m) · fine 5 mm 층 · "
             "주황 원 = 왼손 gripper 구 · 초록 화살표 = gradient · 검정 = 물체 안 · 점선 = 사과 위치 · "
             "target-free 정책 실행 (exclude-authorized)",
             fontsize=9, color=NOTE)
    fig.savefig(OUT / "adapter-two-maps.png", dpi=230, facecolor=SURFACE)
    plt.close(fig)
    meta["adapter-two-maps"] = {
        "src": str(CHUNK.relative_to(ROOT)), "t_step": int(ch["t_step"]), "slice_y_m": y0,
        "target_centroid_m": tc.round(4).tolist(), "n_spheres_drawn": int(near.sum()),
        "representative_sphere": {"link": str(names[rep]), "centre_m": sc[rep].round(4).tolist(),
                                  "radius_mm": float(sr[rep] * 1000), **{k: v for k, v in reps.items()}},
    }


def fig_margin_rules(meta):
    fig = plt.figure(figsize=(12.0, 5.6), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 56)
    ax.axis("off")
    cols = [("일반 장애물", "테이블 · 다른 과일 · 미확인"), ("Target (잡을 사과)", "grounding 1위 물체"),
            ("목적지 (바구니)", "label layer 로 판별")]
    rows = [("잡는 손", "손바닥 · 손가락 (권한 link 3)", AUTH),
            ("그 밖의 로봇", "팔 · 몸통 · 반대손", OTHER),
            ("쥔 사과", "로봇에 붙은 점", C[3])]
    X0, CW, Y0, RH, HW = 22, 32, 44.5, 11.2, 20
    for j, (h, s) in enumerate(cols):
        x = X0 + j * CW
        ax.text(x + CW / 2, Y0 + 6.4, h, ha="center", fontsize=13, weight="bold", color=INK)
        ax.text(x + CW / 2, Y0 + 3.4, s, ha="center", fontsize=9.5, color=NOTE)
    for i, (h, s, c) in enumerate(rows):
        y = Y0 - (i + 1) * RH
        ax.add_patch(FancyBboxPatch((1, y + 0.6), HW - 1.5, RH - 1.2, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=matplotlib.colors.to_rgba(c, 0.12), ec=c, lw=1.2))
        ax.text(2.5, y + RH / 2 + 1.4, h, fontsize=13, weight="bold", color=INK, va="center")
        ax.text(2.5, y + RH / 2 - 1.9, s, fontsize=9.5, color=NOTE, va="center")

    def cell(i, j, big, small=None, color=INK, fc="#ffffff"):
        x = X0 + j * CW
        y = Y0 - (i + 1) * RH
        ax.add_patch(FancyBboxPatch((x + 0.6, y + 0.6), CW - 1.2, RH - 1.2,
                                    boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec="#d9d8d3", lw=1.0))
        ax.text(x + CW / 2, y + RH / 2 + (1.5 if small else 0), big, ha="center", va="center",
                fontsize=17 if len(big) < 8 else 12.5, weight="bold", color=color)
        if small:
            ax.text(x + CW / 2, y + RH / 2 - 2.6, small, ha="center", va="center", fontsize=9.3,
                    color=NOTE, linespacing=1.25)

    cell(0, 0, "50 mm")
    # 잡는 손 × Target: 단계 사다리
    x = X0 + CW
    y = Y0 - RH
    ax.add_patch(FancyBboxPatch((x + 0.6, y + 0.6), CW - 1.2, RH - 1.2, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=matplotlib.colors.to_rgba(AUTH, 0.07), ec=AUTH, lw=1.4))
    steps = [("TRANSIT", 50), ("APPROACH", 20), ("PRE_GRASP", 5), ("GRASP", 0)]
    bw = (CW - 5) / 4
    for k, (p, v) in enumerate(steps):
        bx = x + 2.5 + k * bw
        h = 3.2 * v / 50
        ax.add_patch(plt.Rectangle((bx + 0.6, y + 2.7), bw - 1.2, max(h, 0.15), fc=AUTH, ec="none", alpha=0.85))
        ax.text(bx + bw / 2, y + 2.7 + max(h, 0.15) + 0.3, f"{v}", ha="center", va="bottom", fontsize=10,
                weight="bold", color=INK)
        ax.text(bx + bw / 2, y + 1.7, p, ha="center", fontsize=7.6, color=NOTE)
    ax.text(x + CW / 2, y + RH - 2.0, "코드 기본(relax): 단계별로 줄어든다 [mm]", ha="center", va="center",
            fontsize=9.3, color=INK)
    cell(0, 2, "50 mm")
    cell(1, 0, "50 mm")
    cell(1, 1, "50 mm", "단계와 무관 — 팔꿈치가 사과를\n누르는 것은 파지가 아니라 충돌")
    cell(1, 2, "50 mm")
    cell(2, 0, "50 mm")
    cell(2, 1, "—", "자기 자신")
    cell(2, 2, "20 mm", "사과를 바구니 안으로 넣어야 한다\n(실제 틈 약 33 mm)", color=C[3],
         fc=matplotlib.colors.to_rgba(C[3], 0.08))
    notes = [
        "• 0 mm 는 '제약을 끈다' 가 아니다 — 닿는 것만 허용하고 파고드는 것은 여전히 막는다.",
        "• target-free 정책 (E3 실험 설정, --target-field-policy exclude-authorized): 잡는 손은 사과를 지운 지도를 묻는다 —",
        "   그 지도에는 사과가 없으므로 위 단계 여유는 붙지 않고, 나머지 물체에 대한 50 mm 만 남는다.",
        "• 모르는 link · 모르는 단계 · 모르는 물체는 완화하지 않는다 (50 mm). 잘못 빡빡하면 돌아갈 뿐이지만, 잘못 느슨하면 부딪힌다.",
    ]
    for k, t in enumerate(notes):
        ax.text(1.5, 8.3 - k * 2.0, t, fontsize=10, color=INK if k == 0 else NOTE, va="center")
    fig.savefig(OUT / "margin-rules.png", dpi=230, facecolor=SURFACE)
    plt.close(fig)
    meta["margin-rules"] = {
        "values_mm": {"obstacle": 50, "target_nonauthorized": 50,
                      "target_authorized_by_phase": dict(steps), "held_vs_destination": 20,
                      "robot_vs_destination": 50},
        "sources": {
            "esdf_margin_default": "AG3S/trajopt/serve_safe.py:903 (--esdf-margin 0.05)",
            "phase_rules": "AG3S/ag3s/config.py:616-626 (margin_scale 1.0/0.4/0.1/0; ×0.05)",
            "contact_links": "AG3S/ag3s/config.py:631-634",
            "destination_margin": "AG3S/ag3s/config.py:997 (0.020), applied only to held rows "
                                  "AG3S/trajopt/linearize.py:717-725",
            "target_free_exempt": "AG3S/trajopt/linearize.py:698-705 (T30 F1)",
            "contact_margin_zero_not_off": "AG3S/ag3s/constraints/clearance.py:1-26",
            "note": "t17 기록 실행은 --esdf-margin 10 mm 로 돌았다 (outputs/live_test/20260928_t17/serve_safe.log)",
        },
    }


def main():
    if not figstyle.use_korean():
        raise SystemExit("CJK font 없음")
    meta = {}
    fig_two_maps(meta)
    fig_margin_rules(meta)
    (OUT / "adapter-margin-figs.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta["adapter-two-maps"]["representative_sphere"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
