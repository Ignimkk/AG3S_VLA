"""0930 세미나 슬라이드용 그림 — 기존 figure 의 crop 과 새 그림 하나.

실행:  ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python AG3S/ag3s/docs/figures/ppt-0930/make_ppt_figures.py
출력:  같은 폴더의 *.png 와 ppt-0930-figures.json (원본 경로 · crop 영역 · 새 그림의 숫자)

crop 은 원본 크기에 대한 비율(x0, y0, x1, y1)로 적는다. 원본 그림은 건드리지 않는다.
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
from PIL import Image, ImageDraw  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
D = ROOT / "AG3S/ag3s/docs"
A = D / "archive/14d-era-20260923/figures"
F = D / "figures"

#: name → (source, crop fraction (x0, y0, x1, y1), mask boxes [(x0, y0, x1, y1) fraction, fill sample (x, y)])
CROPS = {
    "rm-scene": (A / "ppt/ppt-rm-01-scene.png", (0, 0.05, 1, 0.86), []),
    "rm-onoff": (A / "ppt/ppt-rm-04-beforeafter.png", (0, 0, 1, 1),
                 [((0.60, 0.872, 0.905, 0.900), (0.58, 0.70))]),   # "발견 C1 …" 한 줄을 상자 색으로 덮는다
    "ap-attention": (A / "ppt/ppt-ap-01-attention.png", (0, 0.17, 1, 0.94), []),
    "tg-method": (A / "ppt/ppt-tg-01-method.png", (0, 0.085, 1, 1), []),
    "esdf-role": (A / "doc-curobo-role.png", (0, 0.035, 1, 0.47), []),
    "adapter-top": (A / "doc-stage-field-adapter.png", (0, 0.04, 1, 0.47), []),
    # Experiment
    "x-selffilter": (F / "t14/t14-selffilter-erases-target.png", (0, 0.20, 0.745, 0.87), []),
    "x-target-drift": (F / "t5d/t5d-topcluster-band-and-score.png", (0.08, 0.086, 1, 0.527), []),
    "x-eikonal": (A / "doc-curobo-role.png", (0.337, 0.52, 0.63, 0.995), []),
    "x-two-tier": (A / "curobo-ground-truth.png", (0.37, 0.06, 1, 0.52), []),
    "x-dest-margin": (A / "destination-margin.png", (0, 0, 0.335, 1), []),
    "x-hold-vs-shadow": (F / "t14/t16-t17-hold-vs-shadow.png", (0, 0.065, 1, 0.62), []),
}


def crop(name, src, box, masks):
    im = Image.open(src).convert("RGB")
    W, H = im.size
    for (mx0, my0, mx1, my1), (sx, sy) in masks:
        fill = im.getpixel((int(sx * W), int(sy * H)))
        ImageDraw.Draw(im).rectangle([mx0 * W, my0 * H, mx1 * W, my1 * H], fill=fill)
    x0, y0, x1, y1 = box
    out = im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
    out.save(OUT / f"{name}.png")
    return {"src": str(src.relative_to(ROOT)), "crop": box, "masked": [m[0] for m in masks],
            "size": out.size}


def target_free_figure():
    """fine ESDF 와 target-free layer 를 Target 높이에서 수평으로 자른 두 장 (16D, ep1807 첫 호출)."""
    ch = np.load(ROOT / "outputs/live_test/20260928_t17/server_constraints/run_0000/chunk_00000.npz",
                 allow_pickle=True)
    tc = np.asarray(ch["target_centroid"], np.float64)
    vs = float(ch["esdf_voxel_size"])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "esdf_light", [figstyle.SEQ_BLUE[5], figstyle.SEQ_BLUE[1], "#ffffff"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), facecolor=figstyle.SURFACE)
    stats = {}
    for ax, key, okey, title in [
            (axes[0], "esdf_distance", "esdf_origin", "fine ESDF — Target 포함"),
            (axes[1], "esdf_free_distance", "esdf_free_origin", "target-free layer — Target 제외")]:
        o = np.asarray(ch[okey], np.float64)
        k = int(round((tc[2] - o[2]) / vs))
        img = np.asarray(ch[key][:, :, k], np.float64)
        nx, ny = img.shape
        ext = (o[1] - vs / 2, o[1] + (ny - 0.5) * vs, o[0] - vs / 2, o[0] + (nx - 0.5) * vs)
        ax.imshow(np.ma.masked_where(img <= 0, np.clip(img, 0, 0.06)), origin="lower", extent=ext,
                  cmap=cmap, vmin=0, vmax=0.06, interpolation="nearest")
        ax.imshow(np.ma.masked_where(img > 0, np.ones_like(img)), origin="lower", extent=ext,
                  cmap=matplotlib.colors.ListedColormap([figstyle.INK]), interpolation="nearest")
        ax.add_patch(plt.Circle((tc[1], tc[0]), 0.055, fill=False, ec=figstyle.CATEGORICAL[7],
                                lw=2.0, ls=(0, (4, 2))))
        ax.set_xlim(tc[1] + 0.22, tc[1] - 0.22)
        ax.set_ylim(tc[0] - 0.20, tc[0] + 0.20)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c8c7c2")
        ax.set_title(title, fontsize=12, color=figstyle.INK, pad=6)
        stats[key] = {"k_index": k, "occupied_voxels_in_slice": int((img <= 0).sum())}
    axes[0].annotate("Target\n(apple)", xy=(tc[1] - 0.055, tc[0]), xytext=(tc[1] - 0.15, tc[0] + 0.12),
                     fontsize=10, color=figstyle.CATEGORICAL[7],
                     arrowprops=dict(arrowstyle="->", color=figstyle.CATEGORICAL[7]))
    axes[1].annotate("Target 만 사라지고\n다른 물체는 남는다", xy=(tc[1] - 0.055, tc[0]),
                     xytext=(tc[1] - 0.08, tc[0] + 0.15), fontsize=10, color=figstyle.CATEGORICAL[7],
                     arrowprops=dict(arrowstyle="->", color=figstyle.CATEGORICAL[7]))
    fig.text(0.5, 0.02, f"top view · Target 높이 z = {tc[2]:.3f} m 수평 단면 · 검정 = 물체 안쪽(d ≤ 0) · "
             "파랑 = 표면에 가까움", ha="center", fontsize=9, color="#8f8e89")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "esdf-target-free.png", dpi=250, facecolor=figstyle.SURFACE)
    plt.close(fig)
    return {"src": "outputs/live_test/20260928_t17/server_constraints/run_0000/chunk_00000.npz",
            "target_centroid_m": tc.round(4).tolist(), "voxel_m": vs, **stats}


def main():
    if not figstyle.use_korean():
        raise SystemExit("CJK font 없음")
    meta = {name: crop(name, *spec) for name, spec in CROPS.items()}
    meta["esdf-target-free"] = target_free_figure()
    (OUT / "ppt-0930-figures.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    for k, v in meta.items():
        print(k, v.get("size", ""))


if __name__ == "__main__":
    main()
