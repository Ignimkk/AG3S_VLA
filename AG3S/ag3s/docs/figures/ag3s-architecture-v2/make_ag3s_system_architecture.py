"""AG3S system architecture — π0.5 attention → Target/Obstacle 분리 → cuRobo ESDF → trajopt 거리 제약.

실행:  ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python AG3S/ag3s/docs/figures/ag3s-architecture-v2/make_ag3s_system_architecture.py
출력:  같은 폴더의 ag3s-system-architecture.{png,pdf,svg,json}

도식 위의 숫자는 코드/설정에서 읽은 값이고(`NUMBERS`), 썸네일은 **한 장면**에서 뽑았다:
held-out ep1807 의 첫 정책 호출(target = apple, 파지 전).
  * RGB            — outputs/live_test/20260925_ep1807/run_0000/step_00000.npz (t = 0)
  * 나머지 전부     — outputs/live_test/20260928_t17/server_constraints/run_0000/chunk_00000.npz
                     (serve_safe 가 그 호출에서 실제로 쓴 attention · point cloud · target ·
                      ESDF 3 층 · sphere clearance · reference/refined chunk; shadow run)
첫 호출이라 두 rollout 의 씬이 같다(같은 에피소드 초기 상태). 그 뒤 프레임은 rollout 마다 갈라지므로
섞지 않는다.
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent
STEM = "ag3s-system-architecture"
REC = ROOT / "outputs/live_test/20260925_ep1807/run_0000/step_00000.npz"
CHUNK = ROOT / "outputs/live_test/20260928_t17/server_constraints/run_0000/chunk_00000.npz"

C = figstyle.CATEGORICAL
INK, INK_2, SURFACE = figstyle.INK, figstyle.INK_2, figstyle.SURFACE
VLA_C = C[0]       # blue   — VLA, action-chunk flow
AG3S_C = C[1]      # orange — AG3S
CUROBO_C = C[2]    # green  — cuRobo (external)
TO_C = C[3]        # amber  — trajopt (external)
ATT_C = C[6]       # purple — attention → field → constraint flow
TARGET_C = C[7]    # red    — Target
NOTE = "#8f8e89"   # 부연설명 글자 — 주석처럼 연한 회색
LINE = "#c8c7c2"   # 보조 테두리 · 썸네일 틀
WHITE = "#ffffff"
GEO_C = "#a09f9a"  # gray   — geometry flow (주 흐름보다 한 단계 물러나게)
LOOP_C = INK       # black  — closed loop
CMAP = figstyle.sequential_cmap()
LEFT_ARM = list(range(7))  # 16D = [L arm 0-6, L grip 7, R arm 8-14, R grip 15] (wire.gripper_columns)

#: 도식에 적힌 숫자와 그 출처. sidecar 로 그대로 나간다.
NUMBERS = {
    "euler_steps": {"value": 10, "src": "AG3S/trajopt/attention_policy.py:55 (num_denoise_steps=10)"},
    "attention_cell": {"value": {"denoise_step": 0, "layer": 8, "head": 2, "agg": "last"},
                       "src": "AG3S/trajopt/attention_policy.py:30-33"},
    "attention_grid": {"value": [16, 16], "src": "AG3S/trajopt/attention_policy.py:51"},
    "action_horizon": {"value": 50, "src": "AG3S/trajopt/config.py:40 / openpi pi0_config.py:26"},
    "action_dim_robot": {"value": 16, "src": "16D policy pi05_rby1_randomized_pick_place_16d_lora"},
    "gripper_columns": {"value": [7, 15], "src": "AG3S/trajopt/safe_policy.py:666"},
    "execution_length_K": {"value": 8, "src": "AG3S/trajopt/config.py:41"},
    "control_hz": {"value": 15.0, "src": "AG3S/trajopt/config.py:42"},
    "chunk_period_ms": {"value": 533.3, "src": "outputs/live_test/20260928_t16/execute_ep1807/manifest.json timing"},
    "sqp_max_iterations": {"value": 3, "src": "AG3S/trajopt/config.py:315"},
    "cost_weights": {"value": {"w_track": 1.0, "w_smooth": 0.05, "w_continuity": 0.5, "w_slack": 1e3},
                     "src": "AG3S/trajopt/config.py:146-157"},
    "seed_percentile": {"value": 95.0, "src": "AG3S/ag3s/config.py:45"},
    "w_attention/w_geometry": {"value": [0.7, 0.3], "src": "AG3S/ag3s/config.py:169-170"},
    "target_confirm_frames": {"value": 3, "src": "AG3S/ag3s/config.py:113"},
    "esdf_coarse_voxel_m": {"value": 0.020, "src": "AG3S/trajopt/serve_safe.py:889 (--voxel)"},
    "esdf_fine_voxel_m": {"value": 0.005, "src": "AG3S/trajopt/serve_safe.py:1040 (--fine-voxel)"},
    "tsdf_voxel_m": {"value": 0.005, "src": "AG3S/trajopt/serve_safe.py:1042 (--tsdf-voxel)"},
    "esdf_grid": {"value": 128, "src": "chunk summary_json esdf_layers"},
    "esdf_margin_m": {"value": 0.05, "src": "AG3S/trajopt/serve_safe.py:903 (--esdf-margin)"},
    "destination_margin_m": {"value": 0.020, "src": "AG3S/ag3s/config.py:897"},
    "depth_hw": {"value": [480, 640], "src": "run meta.json depth_hw"},
    "cameras_policy": {"value": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                       "src": "AG3S/trajopt/serve_safe.py:838-840"},
    "cameras_mujoco": {"value": ["zed_left", "wrist_cam_l", "wrist_cam_r"],
                       "src": "AG3S/trajopt/serve_safe.py:838-840"},
    "prompt_ep1807": {"value": "Pick up the apple and place it in the basket.",
                      "src": "outputs/live_test/20260925_ep1807/run_0000/meta.json"},
}




# ── slide geometry ─────────────────────────────────────────────────────────
#: 16:9 슬라이드 한 장(13.333 × 7.5 in)을 그대로 쓴다. 1 unit = 3 pt 라서 글자 크기(pt)와
#: 상자 크기(unit)를 같은 자로 잰다: 10 pt 한 줄 ≈ 4.3 unit (행간 1.3).
W, H = 320, 180
FIG_IN = (13.333, 7.5)
PT = 1 / 3.0          # 1 pt 를 unit 으로
F_SEC = 14            # VLA · AG3S 같은 구역 제목
F_TITLE = 11.5        # 상자 제목
F_BODY = 10           # 상자 본문
F_LABEL = 9.5         # 화살표 라벨
F_NOTE = 7.5          # 연한 회색 부연설명 (본문·부제·라벨·범례)
F_CAP = 6.5           # 썸네일 캡션
F_TICK = 5.5


def line_h(pt, spacing=1.3):
    return pt * spacing * PT


def tint(hex_color: str, a: float) -> tuple:
    r, g, b = matplotlib.colors.to_rgb(hex_color)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def box(ax, x0, y0, x1, y1, color, *, fill=0.10, lw=1.4, ls="-", r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=(WHITE if fill == "white" else tint(color, fill) if fill
                                    else "none"), ec=color,
                                lw=lw, ls=ls, zorder=z))


def module(ax, x0, y0, x1, y1, color, title, body="", *, fill="white", lw=1.2, ls="-",
           tsize=F_TITLE, bsize=F_NOTE, z=3):
    """제목 + 본문. 본문이 끝나는 y 를 돌려준다 — 썸네일은 그 아래에 놓는다."""
    box(ax, x0, y0, x1, y1, color, fill=fill, lw=lw, ls=ls, z=z)
    ty = y1 - 1.3
    ax.text(x0 + 1.5, ty, title, ha="left", va="top", fontsize=tsize, fontweight="bold",
            color=INK, zorder=z + 1)
    by = ty - line_h(tsize, 1.35)
    if body:
        ax.text(x0 + 1.5, by, body, ha="left", va="top", fontsize=bsize, color=NOTE,
                linespacing=1.3, zorder=z + 1)
        by -= line_h(bsize) * (body.count("\n") + 1)
    return by


def arrow(ax, pts, color, *, lw=1.6, ls="-", label=None, lpos=None, lsize=None,
          lcolor=None, head=11, z=6):
    for (xa, ya), (xb, yb) in zip(pts[:-2], pts[1:-1]):
        ax.plot([xa, xb], [ya, yb], color=color, lw=lw, ls=ls, zorder=z,
                solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=head,
                                 color=color, lw=lw, ls=ls, zorder=z, shrinkA=0, shrinkB=0))
    if label:
        # 주 흐름 라벨(색 지정)은 F_LABEL, 회색 부연 라벨은 F_NOTE
        ax.text(*lpos, label, ha="center", va="center",
                fontsize=lsize or (F_LABEL if lcolor else F_NOTE),
                color=lcolor or NOTE, zorder=z + 2, linespacing=1.2,
                bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"))


class Canvas:
    """데이터 좌표(W×H unit)로 inset axes 를 놓는다."""

    def __init__(self, fig):
        self.fig = fig

    def inset(self, x0, y0, x1, y1, **kw):
        return self.fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H], **kw)


# ── thumbnails ─────────────────────────────────────────────────────────────
def crop_scene(P):
    return (P[:, 0] > 0.35) & (P[:, 0] < 0.85) & (P[:, 1] > -0.40) & (P[:, 1] < 0.45) & \
           (P[:, 2] > 0.78) & (P[:, 2] < 1.16)


def frame(ax, edge, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LINE)
        sp.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=F_CAP, pad=1.5, color=NOTE)


def scene_top(ax, title, edge):
    """로봇 뒤에서 내려다본 top view — 세로 = x(앞), 가로 = y(+y 가 왼쪽). ESDF 단면과 같은 방향."""
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0.45, -0.40)
    ax.set_ylim(0.35, 0.85)
    ax.set_aspect("equal")
    frame(ax, edge, title)


def thumb_attention_overlay(ax, rgb, att, name):
    ax.imshow(rgb, extent=(0, 224, 224, 0))
    a = att.astype(np.float64)
    a = (a - a.min()) / max(a.max() - a.min(), 1e-12)
    ax.imshow(a, extent=(0, 224, 224, 0), cmap=CMAP, alpha=0.55, interpolation="bilinear")
    frame(ax, ATT_C, name)


def esdf_slice(ax, dist, origin, vs, z, title, *, target_xy, vmax=0.06):
    k = int(round((z - origin[2]) / vs))
    img = np.asarray(dist[:, :, k], np.float64)  # rows = x, cols = y
    nx, ny = img.shape
    ext = (origin[1] - vs / 2, origin[1] + (ny - 0.5) * vs,
           origin[0] - vs / 2, origin[0] + (nx - 0.5) * vs)
    free = np.ma.masked_where(img <= 0, np.clip(img, 0, vmax))
    # 빈 공간은 옅게 — 표면 가까이만 파랗게, 멀어질수록 흰색. 눈이 검은 물체와 빨간 고리로 간다.
    light = matplotlib.colors.LinearSegmentedColormap.from_list(
        "esdf_light", [figstyle.SEQ_BLUE[5], figstyle.SEQ_BLUE[1], WHITE])
    ax.imshow(free, origin="lower", extent=ext, cmap=light, vmin=0, vmax=vmax,
              interpolation="nearest")
    occ = np.ma.masked_where(img > 0, np.ones_like(img))
    ax.imshow(occ, origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([INK]),
              interpolation="nearest")
    ax.add_patch(plt.Circle((target_xy[1], target_xy[0]), 0.055, fill=False, ec=TARGET_C,
                            lw=1.3, ls=(0, (3, 1.5))))
    # +y 를 왼쪽에 — 로봇 뒤에서 앞을 보는 방향. Target 중심 40 cm 만 자른다.
    ax.set_xlim(target_xy[1] + 0.20, target_xy[1] - 0.20)
    ax.set_ylim(target_xy[0] - 0.20, target_xy[0] + 0.20)
    frame(ax, CUROBO_C, title)
    return int((img <= 0).sum()), k


def small_plot(fig, ax, title):
    figstyle.style_axes(fig, ax)
    ax.set_facecolor(WHITE)
    ax.tick_params(labelsize=F_TICK, length=1.5, pad=1, colors=NOTE)
    ax.grid(False)
    for name, sp in ax.spines.items():
        sp.set_visible(name in ("left", "bottom"))
        sp.set_color(LINE)
        sp.set_linewidth(0.6)
    ax.set_title(title, fontsize=F_CAP, pad=2, color=NOTE)


def main() -> None:
    if not figstyle.use_korean():
        raise SystemExit("CJK font 없음")
    plt.rcParams["mathtext.fontset"] = "dejavusans"

    rec = np.load(REC, allow_pickle=True)
    ch = np.load(CHUNK, allow_pickle=True)
    summ = json.loads(str(ch["summary_json"]))

    fig = plt.figure(figsize=FIG_IN, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    cv = Canvas(fig)

    # ── legend (한 줄) ─────────────────────────────────────────────────────
    lx = 3
    for c, lw, ls, txt in [(ATT_C, 3.0, "-", "attention → distance constraint"),
                           (VLA_C, 3.0, "-", "action chunk → initial trajectory"),
                           (LOOP_C, 2.2, "-", "closed loop"),
                           (GEO_C, 1.4, "-", "geometry"),
                           (NOTE, 1.2, (0, (4, 2)), "external module")]:
        ax.plot([lx, lx + 8], [177.8, 177.8], color=c, lw=lw, ls=ls)
        ax.text(lx + 9.5, 177.8, txt, fontsize=F_NOTE, va="center", color=NOTE)
        lx += 13 + len(txt) * F_NOTE * 0.52 * PT + 6

    # ── machines ───────────────────────────────────────────────────────────
    box(ax, 1, 1, 44, 174.5, NOTE, fill=0, lw=1.0, ls=(0, (6, 3)), r=1.5, z=0)
    ax.text(2.5, 173.6, "LOCAL PC · MuJoCo", fontsize=F_TITLE, fontweight="bold", va="top",
            color=INK)
    box(ax, 47, 5, 301, 174.5, NOTE, fill=0, lw=1.0, ls=(0, (6, 3)), r=1.5, z=0)
    ax.text(48.5, 173.6, "GPU SERVER · serve_safe.py", fontsize=F_TITLE, fontweight="bold",
            va="top", color=INK)
    ax.text(110, 173.3, "π0.5 + AG3S + trajopt · one process", fontsize=F_NOTE, va="top",
            color=NOTE)
    ax.text(45.5, 55, "websocket · SSH tunnel", rotation=90, ha="center", va="center",
            fontsize=F_CAP, color=NOTE, bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))

    # ── INPUT (local PC) ───────────────────────────────────────────────────
    ax.text(2.5, 167.4, "INPUT", fontsize=F_NOTE, fontweight="bold", va="top", color=NOTE)
    ib = dict(lw=1.0, tsize=10.5)
    module(ax, 3, 142, 42, 160, LINE, "Instruction",
           "\"Pick up the apple and\n place it in the basket.\"", **ib)
    module(ax, 3, 120, 42, 139, LINE, "RGB × 3", **ib)
    for i, key in enumerate(["image_cam_high", "image_cam_left_wrist", "image_cam_right_wrist"]):
        x0 = 4.2 + i * 12.6
        a_ = cv.inset(x0, 121.3, x0 + 11.8, 133.1)
        a_.imshow(rec[key])
        frame(a_, INK_2)
    module(ax, 3, 105, 42, 117, LINE, r"Robot state $q \in \mathbb{R}^{16}$",
           "(arm 7 + gripper) × 2", **ib)
    module(ax, 3, 80, 42, 102, LINE, "RGB-D × 3",
           "depth 480×640\n" r"$K$,  $T_{base\leftarrow cam}$" "\n" "capture-time pose", **ib)
    module(ax, 3, 6, 42, 24, LOOP_C, "MuJoCo sim",
           "execute 8 / 50 steps\n15 Hz", lw=1.2, tsize=10.5)
    arrow(ax, [(22.5, 24), (22.5, 80)], LOOP_C, lw=2.2, head=12,
          label="next\nobservation", lpos=(22.5, 52))

    # ── VLA ────────────────────────────────────────────────────────────────
    box(ax, 50, 110, 299, 167, VLA_C, fill=0.04, lw=1.6, r=1.8)
    ax.text(52, 165.8, "VLA · π0.5", fontsize=F_SEC, fontweight="bold", va="top", color=VLA_C)
    ax.text(78, 165.4, "16D LoRA fine-tune", fontsize=F_NOTE, va="top", color=NOTE)

    module(ax, 53, 122, 96, 154, VLA_C, "VLM prefix",
           "PaliGemma 2B\n• image tokens 3 × 256\n• instruction\n• state")

    ex0, ex1, ey0, ey1 = 100, 233, 113, 159
    box(ax, ex0, ey0, ex1, ey1, VLA_C, fill="white", lw=1.2, z=3)
    ax.text(ex0 + 1.5, ey1 - 1.3, "Action expert · flow matching · 10 Euler steps",
            fontsize=F_TITLE, fontweight="bold", va="top", color=INK, zorder=4)
    ax.text(ex0 + 1.5, ey1 - 6.8,
            r"$A_{t-\delta} = A_t - \delta\, v_\theta(A_t, t),\quad \delta = 0.1$",
            fontsize=F_NOTE + 0.5, va="top", color=NOTE, zorder=4)
    cy0, chh = 136, 10
    ax.add_patch(Rectangle((ex0 + 1.5, cy0), 10, chh, fc=WHITE, ec=LINE, lw=0.9,
                           zorder=4))
    ax.text(ex0 + 6.5, cy0 + chh / 2, "noise", ha="center", va="center", fontsize=F_CAP,
            color=NOTE, zorder=5)
    sw, sg, sx0 = 10.4, 1.3, ex0 + 13.5
    for k in range(10):
        x = sx0 + k * (sw + sg)
        is0 = k == 0
        ax.add_patch(Rectangle((x, cy0), sw, chh,
                               fc=tint(ATT_C, 0.22) if is0 else tint(VLA_C, 0.08),
                               ec=ATT_C if is0 else tint(VLA_C, 0.55), lw=1.8 if is0 else 0.8, zorder=4))
        ax.text(x + sw / 2, cy0 + 6.6, f"k={k}", ha="center", va="center", fontsize=F_BODY - 0.5,
                fontweight="bold" if is0 else "normal", zorder=5)
        ax.text(x + sw / 2, cy0 + 2.6, f"t={1 - k / 10:.1f}", ha="center", va="center",
                fontsize=F_TICK, color=NOTE, zorder=5)
        xp = x - sg if k else ex0 + 11.5
        ax.add_patch(FancyArrowPatch((xp, cy0 + chh / 2), (x, cy0 + chh / 2), arrowstyle="-|>",
                                     mutation_scale=6, color=LINE, lw=0.7, zorder=5,
                                     shrinkA=0, shrinkB=0))
    chain_end = sx0 + 10 * sw + 9 * sg

    # attention readout + real maps
    rx0, rx1, ry0, ry1 = ex0 + 1.5, ex1 - 1.5, 114.5, 133.5
    box(ax, rx0, ry0, rx1, ry1, ATT_C, fill=0.08, lw=1.4, z=5)
    ax.text(rx0 + 1.5, ry1 - 1.2, "Attention readout · k=0", fontsize=F_TITLE,
            fontweight="bold", va="top", color=INK, zorder=6)
    ax.text(rx0 + 1.5, ry1 - 6.4, "layer 8 · head 2\nquery: action token\nkey: image tokens",
            fontsize=F_NOTE, va="top", color=NOTE, zorder=6, linespacing=1.3)
    for i, (akey, rkey, name) in enumerate([
            ("attention_head", "image_cam_high", "cam_high"),
            ("attention_left_wrist", "image_cam_left_wrist", "left_wrist"),
            ("attention_right_wrist", "image_cam_right_wrist", "right_wrist")]):
        x0 = 173.5 + i * 19.5
        thumb_attention_overlay(cv.inset(x0, 115.5, x0 + 14.5, 130.0), rec[rkey], ch[akey], name)
    ax.add_patch(FancyArrowPatch((sx0 + sw / 2, cy0), (sx0 + sw / 2, ry1), arrowstyle="-|>",
                                 mutation_scale=9, color=ATT_C, lw=1.6, zorder=6,
                                 shrinkA=0, shrinkB=0))
    arrow(ax, [(96, cy0 + chh / 2), (ex0, cy0 + chh / 2)], VLA_C, lw=1.5, head=10)

    # action chunk tile + real reference chunk
    module(ax, 237, 113, 296, 159, VLA_C, "Action chunk", r"$A \in \mathbb{R}^{50 \times 16}$",
           lw=1.2)
    arrow(ax, [(chain_end, cy0 + chh / 2), (237, cy0 + chh / 2)], VLA_C, lw=2.0, head=11)
    ref = np.asarray(ch["reference_chunk"], np.float64)
    pa = cv.inset(245, 120.5, 294, 145)
    for j in LEFT_ARM:
        pa.plot(np.arange(50), ref[:, j], color=C[j], lw=1.0)
    pa.axvspan(-0.5, 7.5, color=tint(LOOP_C, 0.07), lw=0)
    pa.set_xlim(-0.5, 49.5)
    pa.set_xticks([0, 8, 49])
    small_plot(fig, pa, "left arm joints · K = 8 shaded")

    # ── inputs → server ────────────────────────────────────────────────────
    bus = 48.5  # 세 입력을 한 줄로 모아 prefix 에 넣는다
    for y in (151, 129.5, 111):
        ax.plot([42, bus], [y, y], color=GEO_C, lw=1.3, zorder=6, solid_capstyle="round")
    ax.plot([bus, bus], [111, 151], color=GEO_C, lw=1.3, zorder=6, solid_capstyle="round")
    arrow(ax, [(bus, 138), (53, 138)], GEO_C, lw=1.3, head=10)
    arrow(ax, [(42, 107), (52, 107), (52, 94), (56, 94)], GEO_C, lw=1.3, head=10)
    arrow(ax, [(42, 86), (56, 86)], GEO_C, lw=1.3, head=10)

    # ── AG3S ───────────────────────────────────────────────────────────────
    box(ax, 50, 7, 299, 106, AG3S_C, fill=0.04, lw=1.6, r=1.8)
    ax.text(52, 104.8, "AG3S", fontsize=F_SEC, fontweight="bold", va="top", color=AG3S_C)
    ax.text(67, 104.4, "Attention-Guided 3D Safety Scene", fontsize=F_NOTE, va="top",
            color=NOTE)

    P = np.asarray(ch["attention_cloud"], np.float64)
    av = np.asarray(ch["attention_values"], np.float64)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(crop_scene(P))
    idx = idx[rng.permutation(len(idx))[:9000]]
    idx = idx[np.argsort(av[idx])]  # 뜨거운 점을 위에 그린다
    tp = np.asarray(ch["target_points"], np.float64)
    tc = np.asarray(ch["target_centroid"], np.float64)

    # top row: ① ② ③  (① 는 썸네일이 없으므로 글에 맞춘 높이)
    top1 = 99
    module(ax, 56, 80, 103, top1, AG3S_C, "① Robot mask",
           "self-filter\nFK → URDF spheres\nremove robot pixels")
    by = module(ax, 118, 63, 178, top1, AG3S_C, "② Attention lifting",
                "2D attention → 3D points")
    a2 = cv.inset(125, 64.5, 171, by - 3.2)
    scene_top(a2, "attention (top view)", AG3S_C)
    a2.scatter(P[idx, 1], P[idx, 0], c=av[idx], cmap=CMAP, s=0.5, lw=0, vmin=0, vmax=1)
    by = module(ax, 193, 63, 253, top1, AG3S_C, "③ Target grounding",
                "3D clusters → Target vs Obstacle")
    a3 = cv.inset(200, 64.5, 246, by - 3.2)
    scene_top(a3, "Target (red) · Obstacle (gray)", AG3S_C)
    a3.scatter(P[idx, 1], P[idx, 0], c=GEO_C, s=0.4, lw=0, alpha=0.45)
    a3.scatter(tp[:, 1], tp[:, 0], c=TARGET_C, s=2.5, lw=0)

    # bottom row
    bo0, bo1 = 10, 54           # external 점선 상자
    bi0, bi1 = 12.5, 48         # 안쪽 모듈
    box(ax, 53, bo0, 143, bo1, CUROBO_C, fill=0, lw=1.2, ls=(0, (5, 3)), r=1.4, z=2)
    ax.text(54.5, bo1 - 1.0, "external · cuRobo", fontsize=F_LABEL, fontweight="bold",
            va="top", color=CUROBO_C, zorder=3)
    by = module(ax, 56, bi0, 140, bi1, CUROBO_C, "④ TSDF → ESDF",
                "coarse 20 mm · fine 5 mm")
    fine_o = np.asarray(ch["esdf_origin"], np.float64)
    free_o = np.asarray(ch["esdf_free_origin"], np.float64)
    vs = float(ch["esdf_voxel_size"])
    sh = by - 3.2 - (bi0 + 1.2)
    s1 = cv.inset(62, bi0 + 1.2, 62 + sh, by - 3.2)
    n_occ_fine, kz = esdf_slice(s1, ch["esdf_distance"], fine_o, vs, tc[2], "fine ESDF",
                                target_xy=tc)
    s2 = cv.inset(98, bi0 + 1.2, 98 + sh, by - 3.2)
    n_occ_free, _ = esdf_slice(s2, ch["esdf_free_distance"], free_o,
                               float(ch["esdf_free_voxel_size"]), tc[2], "target-free layer",
                               target_xy=tc)
    ax.text((62 + sh + 98) / 2, bi0 + 1.2 + sh / 2, "→", fontsize=16, ha="center", va="center",
            color=TARGET_C, zorder=6)
    ax.text(98 + sh + 1.5, bi0 + 1.2 + sh / 2, "black\n= inside\n\nring\n= Target",
            fontsize=F_TICK, ha="left", va="center", color=NOTE, zorder=6, linespacing=1.2)

    by = module(ax, 155, bi0, 207, bi1, AG3S_C, "⑤ Distance adapter",
                "distance · gradient\nmargin by label")
    a5 = cv.inset(158, bi0 + 1.2, 204, by - 3.2)
    scene_top(a5, "sphere clearance", AG3S_C)
    a5.scatter(P[idx, 1], P[idx, 0], c=GEO_C, s=0.35, lw=0, alpha=0.3)
    sc = np.asarray(ch["sphere_centres"], np.float64)
    clr = np.asarray(ch["clearance"], np.float64).min(axis=0)
    a5.scatter(sc[:, 1], sc[:, 0], c=clr, cmap=CMAP.reversed(), s=7, lw=0.25,
               edgecolors=NOTE, vmin=0, vmax=0.2, zorder=3)

    box(ax, 217, bo0, 296, bo1, TO_C, fill=0, lw=1.2, ls=(0, (5, 3)), r=1.4, z=2)
    ax.text(218.5, bo1 - 1.0, "external · trajopt", fontsize=F_LABEL, fontweight="bold",
            va="top", color="#a26d00", zorder=3)
    by = module(ax, 220, bi0, 293, bi1, TO_C, "⑥ Trajectory optimization",
                r"SQP + OSQP ·  init $Q^0 = A$" "\n" r"s.t. $d - r - \mathrm{margin} \geq 0$",
                )
    d6 = cv.inset(231, bi0 + 5.0, 291, by - 3.4)
    delta = (np.asarray(ch["refined_chunk"], np.float64) - ref)[:8] * 1000.0
    for j in LEFT_ARM:
        d6.plot(np.arange(8), delta[:, j], color=C[j], lw=1.0, marker="o", ms=1.6)
    d6.axhline(0, color=LINE, lw=0.5)
    d6.set_xticks(range(8))
    small_plot(fig, d6, "refined − reference (mrad)")

    # ── flows ──────────────────────────────────────────────────────────────
    arrow(ax, [(103, 92), (118, 92)], GEO_C, lw=1.6, label="points", lpos=(110.5, 96))
    arrow(ax, [(92, 80), (92, bi1)], GEO_C, lw=1.6, label="masked\ndepth",
          lpos=(92, 66))
    arrow(ax, [(148, ry0), (148, top1)], ATT_C, lw=2.8, head=13,
          label="attention 3 × 16×16", lpos=(165, 108.2), lcolor=ATT_C)
    arrow(ax, [(178, 92), (193, 92)], ATT_C, lw=2.8, head=13)
    arrow(ax, [(223, 63), (223, 58.5), (122, 58.5), (122, bi1)], ATT_C, lw=2.8, head=13,
          label="Target exclusion", lpos=(170, 58.5), lcolor=ATT_C)
    arrow(ax, [(140, 38), (155, 38)], ATT_C, lw=2.8, head=13, label="ESDF", lpos=(147.5, 42.5), lcolor=ATT_C)
    arrow(ax, [(207, 38), (220, 38)], ATT_C, lw=2.8, head=13,
          label=r"$d,\ \nabla d$", lpos=(213.5, 42.8), lcolor=ATT_C)
    arrow(ax, [(220, 22), (207, 22)], GEO_C, lw=1.3, ls=(0, (4, 2)), head=10,
          label="query", lpos=(213.5, 26.3))
    arrow(ax, [(275, 113), (275, bi1)], VLA_C, lw=2.8, head=13,
          label=r"action chunk" "\n" r"$\rightarrow Q^0$", lpos=(275, 80), lcolor=VLA_C)

    # ── OUTPUT & closed loop ───────────────────────────────────────────────
    ax.text(303, 45, "OUTPUT", fontsize=F_BODY, fontweight="bold", va="top", color=INK)
    ax.text(303, 40.5, r"$Q^\ast$" "\n+ verdict", fontsize=F_NOTE, va="top", color=NOTE,
            linespacing=1.25)
    arrow(ax, [(293, 30), (312, 30), (312, 2.8), (34, 2.8), (34, 6)], LOOP_C, lw=2.2, head=11,
          label="closed loop · execute K = 8 steps @ 15 Hz → next observation",
          lpos=(172, 2.8))

    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{STEM}.{ext}", dpi=300 if ext == "png" else None,
                    facecolor=SURFACE)

    thumbs = {
        "scene": {"episode": 1807, "policy_call": 1, "rgb_record": str(REC.relative_to(ROOT)),
                  "server_chunk": str(CHUNK.relative_to(ROOT))},
        "target_centroid_m": tc.round(4).tolist(),
        "target_n_points": int(len(tp)),
        "esdf_slice": {"z_m": float(tc[2]), "k_index": kz, "voxel_m": vs,
                       "fine_occupied_voxels_in_slice": n_occ_fine,
                       "target_free_occupied_voxels_in_slice": n_occ_free},
        "sphere_clearance_min_mm": float(clr.min() * 1000),
        "n_spheres": int(len(sc)),
        "to": {k: summ["to"][k] for k in ("status", "sqp_iterations", "solve_ms",
                                         "time_budget_hit")},
        "refined_minus_reference_left_arm_max_abs_mrad": float(np.abs(delta[:, LEFT_ARM]).max()),
    }
    (OUT / f"{STEM}.json").write_text(json.dumps(
        {"figure": f"{STEM}.png", "generator": pathlib.Path(__file__).name,
         "slide": {"size_in": list(FIG_IN), "aspect": "16:9",
                   "font_pt": {"section": F_SEC, "title": F_TITLE, "body": F_BODY,
                               "label": F_LABEL, "note": F_NOTE, "caption": F_CAP,
                               "tick": F_TICK}},
         "numbers": NUMBERS, "thumbnails": thumbs}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(thumbs, indent=1))


if __name__ == "__main__":
    main()
