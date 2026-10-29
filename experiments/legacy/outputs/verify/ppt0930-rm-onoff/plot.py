"""rm-onoff-16d.png — compute.py 의 raw npz / json 만 읽어 그린다.

    PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python outputs/verify/ppt0930-rm-onoff/plot.py [out.png]
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle as fs  # noqa: E402

HERE = pathlib.Path(_ag3s_workspace_path('outputs/verify/ppt0930-rm-onoff'))
OUTPNG = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "rm-onoff-16d.png"
raw = np.load(HERE / "rm_onoff_raw.npz")
num = json.loads((HERE / "rm_onoff_numbers.json").read_text())

import matplotlib.font_manager as fm  # noqa: E402
p = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if pathlib.Path(p).exists():
    fm.fontManager.addfont(p)
KO = fs.use_korean()
assert KO, "no CJK font"

HI = fs.CATEGORICAL[1]      # robot-mask highlight (orange)
OFFC = fs.CATEGORICAL[7]    # OFF (red)
ONC = fs.CATEGORICAL[2]     # ON (green)
DARK = "#07182f"            # d <= 0

CAMS = (("zed_left", "머리 카메라"), ("wrist_cam_l", "왼손목 카메라"), ("wrist_cam_r", "오른손목 카메라"))

fig = plt.figure(figsize=(12.6, 6.0))
fig.patch.set_facecolor(fs.SURFACE)
fig.text(0.012, 0.985, "Robot mask (self-filter) OFF vs ON  —  같은 depth 3장, 같은 cuRobo 설정, 마스크만 다르다",
         fontsize=14.5, weight="bold", color=fs.INK, va="top")
fig.text(0.012, 0.935, "episode 1807 · 첫 정책 호출 (t = 0) · 16D · coarse 20 mm + fine 5 mm ESDF · "
         f"self-filter {num['self_filter']['n_spheres']} 구, inflation "
         f"{num['ag3s_config']['pointcloud']['self_filter_inflation']*1000:.0f} mm",
         fontsize=10.5, color=fs.INK_2, va="top")

# ---------------------------------------------------------------- (a) cameras
L0, W0 = 0.012, 0.615
cw = (W0 - 2 * 0.008) / 3
ch = cw * 12.6 / 6.0 * 3 / 4
top = 0.885
for k, (c, lab) in enumerate(CAMS):
    ax = fig.add_axes([L0 + k * (cw + 0.008), top - ch, cw, ch])
    rgb = raw[f"rgb_{c}"].astype(float) / 255.0
    m = raw[f"mask_{c}"]
    img = rgb * 0.55 + 0.45 * 0.92       # 배경을 흐리게 해서 마스크가 보이게
    col = np.array(matplotlib.colors.to_rgb(HI))
    img[m] = 0.25 * rgb[m] + 0.75 * col
    ax.imshow(img)
    # 마스크 윤곽
    ax.contour(m.astype(float), levels=[0.5], colors=[HI], linewidths=1.0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(fs.GRID_INK)
    pc = num["robot_mask_per_camera"][c]["erased_pct_of_image"]
    ax.text(0.02, 0.97, lab, transform=ax.transAxes, fontsize=10.5, weight="bold", color=fs.INK,
            va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.text(0.98, 0.03, f"지움 {pc:.1f} %", transform=ax.transAxes, fontsize=13, weight="bold",
            color="white", ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc=HI, ec="none"))
fig.text(L0, top - ch - 0.012, "주황 = robot mask (현재 관절각의 로봇 구 안에 떨어진 depth 픽셀) — "
         "ON 에서는 이 광선을 적분하지 않는다", fontsize=9.5, color=fs.INK_2, va="top")

# ---------------------------------------------------------------- (b) slices
xs, zs = raw["slice_x"], raw["slice_z"]
y0 = float(raw["slice_y"])
XL, ZL = (-0.10, 0.86), (0.74, 1.40)
ix = (xs >= XL[0] - 1e-9) & (xs <= XL[1] + 1e-9)
iz = (zs >= ZL[0] - 1e-9) & (zs <= ZL[1] + 1e-9)
VMAX = 0.30
ramp = LinearSegmentedColormap.from_list("ramp", fs.SEQ_BLUE[::-1][1:] + ["#f4f8fd"])
cen, rad = raw["centres_arms"], raw["radii_arms"]
dy = np.abs(cen[:, 1] - y0)
cut = dy < rad
sec = np.sqrt(np.clip(rad ** 2 - dy ** 2, 0, None))

sw = 0.26
sh = sw * 12.6 / 6.0 * (ZL[1] - ZL[0]) / (XL[1] - XL[0])
sb = 0.075
for k, (mode, title, tc) in enumerate((("off", "마스크 OFF", OFFC), ("on", "마스크 ON", ONC))):
    ax = fig.add_axes([0.052 + k * (sw + 0.015), sb, sw, sh])
    S = raw[f"slice_{mode}"][np.ix_(iz, ix)]
    img = np.ma.masked_less_equal(S, 0.0)
    cm = ramp.copy(); cm.set_bad(DARK)
    im = ax.imshow(np.clip(img, 0, VMAX), origin="lower", cmap=cm, vmin=0, vmax=VMAX,
                   extent=(xs[ix][0] - 0.0025, xs[ix][-1] + 0.0025, zs[iz][0] - 0.0025, zs[iz][-1] + 0.0025),
                   interpolation="nearest")
    ax.contour(xs[ix], zs[iz], S, levels=[0.0], colors=["white"], linewidths=0.6)
    clr = raw[f"clr_arms_{mode}"]
    for i in np.where(cut)[0]:
        bad = clr[i] < 0
        ax.add_patch(plt.Circle((cen[i, 0], cen[i, 2]), sec[i], fill=False,
                                lw=1.3 if bad else 0.9, color=OFFC if bad else fs.INK_2,
                                zorder=5))
    ax.set_xlim(*XL); ax.set_ylim(*ZL); ax.set_aspect("equal")
    ax.tick_params(labelsize=8.5, colors=fs.INK_2, length=2)
    ax.set_xlabel("x [m] (로봇 앞)", fontsize=9.5, color=fs.INK_2, labelpad=1)
    if k == 0:
        ax.set_ylabel("z [m]", fontsize=9.5, color=fs.INK_2, labelpad=1)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=13, weight="bold", color=tc, pad=3, loc="left")
    for s in ax.spines.values():
        s.set_color(fs.GRID_INK)
cax = fig.add_axes([0.052 + 2 * sw + 0.015 + 0.006, sb, 0.009, sh])
cb = fig.colorbar(im, cax=cax, extend="max")
cb.set_label("거리 d [m]  (검정 = d ≤ 0)", fontsize=9, color=fs.INK_2)
cb.ax.tick_params(labelsize=8, colors=fs.INK_2)
fig.text(0.012, sb + sh + 0.045, f"ESDF 단면 — 왼팔을 지나는 수직 x–z 평면 (y = {y0:+.3f} m), "
         "원 = 그 평면에 걸린 팔 구의 단면, 빨강 = d − r − 50 mm < 0",
         fontsize=9.5, color=fs.INK_2, va="bottom")

# ---------------------------------------------------------------- inset: 배치도 (위에서 본 x–y)
ax = fig.add_axes([0.705, 0.555, 0.28, 0.33])
fc, fr_ = raw["filter_centres"], raw["filter_radii"]
for c_, r_ in zip(fc, fr_):
    ax.add_patch(plt.Circle((c_[1], c_[0]), r_, color="#bdbcb6", alpha=0.35, lw=0))
objs = num["objects_in_base_m"]
# 장애물 발자국 = ON 필드(coarse)에서 d <= 0 인 복셀 중 테이블 면 위 (z > 0.845 m) 를 위에서 투영 — 그린 모양이 아니라 측정값
V = raw["on_coarse_values"]; O = raw["on_coarse_origin"]; vs = float(raw["on_coarse_voxel"])
gx = O[0] + vs * np.arange(V.shape[0]); gy = O[1] + vs * np.arange(V.shape[1]); gz = O[2] + vs * np.arange(V.shape[2])
occ = ((V <= 0) & (gz[None, None, :] > 0.845) & (gz[None, None, :] < 1.3)).any(axis=2)
ii, jj = np.nonzero(occ)
ax.scatter(gy[jj], gx[ii], s=5, marker="s", color=fs.CATEGORICAL[4], lw=0)
for name, lab in (("crate", "crate"), ("apple", "사과")):
    p = objs[name]
    ax.text(p[1], p[0] + 0.17 if name == "crate" else p[0] + 0.06, lab, ha="center", fontsize=8.5,
            color=fs.INK)
ax.plot([y0, y0], [XL[0], XL[1]], color=fs.INK, lw=2.0)
ax.text(y0 + 0.02, XL[1] - 0.02, "단면", fontsize=9.5, weight="bold", color=fs.INK, va="top")
ax.set_xlim(0.55, -0.55)            # 로봇 왼쪽(+y)을 화면 왼쪽에
ax.set_ylim(-0.35, 0.95)
ax.set_aspect("equal")
ax.tick_params(labelsize=7.5, colors=fs.INK_2, length=2)
ax.set_xlabel("y [m]  (+ = 로봇 왼쪽)", fontsize=8.5, color=fs.INK_2, labelpad=1)
ax.set_ylabel("x [m]", fontsize=8.5, color=fs.INK_2, labelpad=1)
ax.set_title("배치도 — 위에서 본 장면 (회색 = self-filter 구, 분홍 = 테이블 위 장애물)", fontsize=8.8, color=fs.INK, pad=3)
for s in ax.spines.values():
    s.set_color(fs.GRID_INK)
ax.set_facecolor(fs.SURFACE)

# ---------------------------------------------------------------- (c) before / after
ax = fig.add_axes([0.695, 0.035, 0.295, 0.45]); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
from matplotlib.patches import FancyBboxPatch  # noqa: E402
ax.add_patch(FancyBboxPatch((0.1, 0.1), 9.8, 9.8, boxstyle="round,pad=0.05,rounding_size=0.3",
                            fc="#f1f0ec", ec=fs.GRID_INK, lw=1.0))
ax.text(5.0, 9.35, "제약 구 여유거리  d − r − 50 mm", ha="center", va="top", fontsize=11,
        weight="bold", color=fs.INK)
ax.text(3.9, 8.1, "OFF", ha="center", fontsize=11.5, weight="bold", color=OFFC)
ax.text(7.7, 8.1, "ON", ha="center", fontsize=11.5, weight="bold", color=ONC)
rows = (("gripper", "손 구 (서빙 기본)"), ("arms", "양팔 구"))
y = 6.9
for key, lab in rows:
    e = num["constraint_spheres"][key]
    ax.text(0.45, y, f"{lab}\n{e['n_spheres']} 개", fontsize=9.5, color=fs.INK_2, va="center",
            linespacing=1.3)
    ax.text(3.9, y + 0.25, f"{e['off']['n_violated']}", ha="center", va="center", fontsize=22,
            weight="bold", color=OFFC)
    ax.text(7.7, y + 0.25, f"{e['on']['n_violated']}", ha="center", va="center", fontsize=22,
            weight="bold", color=ONC)
    ax.annotate("", xy=(6.6, y + 0.25), xytext=(5.0, y + 0.25),
                arrowprops=dict(arrowstyle="-|>", color=fs.INK_2, lw=1.4))
    ax.text(3.9, y - 0.95, f"최악 {e['off']['worst_clearance_mm']:+.0f} mm", ha="center",
            fontsize=9.5, color=OFFC)
    ax.text(7.7, y - 0.95, f"최악 {e['on']['worst_clearance_mm']:+.0f} mm", ha="center",
            fontsize=9.5, color=ONC)
    y -= 2.75
ax.text(5.0, 0.55, "큰 숫자 = 위반 구 (여유 < 0) 개수", ha="center", fontsize=8.8,
        color=fs.INK_2)

fig.savefig(OUTPNG, dpi=200, facecolor=fs.SURFACE)
print("wrote", OUTPNG)
