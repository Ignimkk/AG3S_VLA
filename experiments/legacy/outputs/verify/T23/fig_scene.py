"""T23 figure (1) 실제 씬 — T16 t=72–96 손가락 구 중심 궤적, legacy vs fixed, 사과 위에."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
import mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from AG3S.ag3s.experiments.common import figstyle

FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t23'))
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T23/m3'))
raw = json.loads((V / "hold_drift.json").read_text())
ana = json.loads((V / "hold_drift_analysis.json").read_text())
ko = figstyle.use_korean()
T_END = 96
N = T_END - 72 + 1
m = mujoco.MjModel.from_xml_path(raw["model_xml"])
d = mujoco.MjData(m)
W, H = 800, 600
m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
ren = mujoco.Renderer(m, height=H, width=W)
apple_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE

runs = {"legacy": raw["runs"]["replay|legacy|hold"], "fixed": raw["runs"]["replay|fixed|hold"]}
anas = {"legacy": ana["runs"]["replay|legacy|hold"], "fixed": ana["runs"]["replay|fixed|hold"]}


def world_path(run):
    out = []
    for e in run["log"][:N]:
        R = np.asarray(e["base_R"]); p = np.asarray(e["base_world"])
        out.append(R @ np.asarray(e["finger_centre_base"]) + p)
    return np.asarray(out)


paths = {k: world_path(r) for k, r in runs.items()}
t16 = np.asarray(anas["legacy"]["t16_recorded_finger_centre_base"][:N])  # base ≈ world (offset < 0.1 mm)
apple0 = np.asarray(runs["legacy"]["log"][0]["apple_world"])


def project(pts):
    gl = mujoco.mjv_averageCamera(ren.scene.camera[0], ren.scene.camera[1])  # mono 렌더는 두 눈의 평균
    pos, fwd, up = np.asarray(gl.pos), np.asarray(gl.forward), np.asarray(gl.up)
    right = np.cross(fwd, up)
    hw = gl.frustum_width if gl.frustum_width > 0 else (gl.frustum_top - gl.frustum_bottom) / 2 * W / H
    left_ = gl.frustum_center - hw; right_ = gl.frustum_center + hw
    uv = []
    for p in np.atleast_2d(pts):
        v = p - pos; zc = v @ fwd
        x = (v @ right) * gl.frustum_near / zc; y = (v @ up) * gl.frustum_near / zc
        uv.append(((x - left_) / (right_ - left_) * W, (gl.frustum_top - y) / (gl.frustum_top - gl.frustum_bottom) * H))
    return np.asarray(uv)


def render(qpos):
    d.qpos[:] = qpos; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    cam.lookat[:] = apple0 + np.array([0.0, 0.0, 0.05])
    cam.distance, cam.azimuth, cam.elevation = 0.55, 255.0, -15.0
    ren.update_scene(d, camera=cam)
    return ren.render()


col = {"legacy": figstyle.CATEGORICAL[1], "fixed": figstyle.CATEGORICAL[0], "t16": figstyle.INK}
lab = ({"legacy": "legacy (매 step rby1_state 재명령)", "fixed": "fixed (HoldController q_hold)", "t16": "T16 기록 qpos (실측)"} if ko else
       {"legacy": "legacy (re-command rby1_state each step)", "fixed": "fixed (HoldController q_hold)", "t16": "T16 recorded qpos"})
fig = plt.figure(figsize=(16, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9])
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
figstyle.style_axes(fig, axes)
sidecar = {"t_range": [72, T_END], "camera": {"lookat": (apple0 + [0, 0, 0.05]).tolist(), "distance": 0.55, "azimuth": 255.0, "elevation": -15.0},
           "source": "outputs/verify/T23/m3/hold_drift.json runs replay|legacy|hold, replay|fixed|hold; T16 recorded = frames.jsonl control qpos",
           "finger_centre": "mean of ee_finger_l1/l2 sphere centres of the T16 constraint model (122 spheres)", "panels": {}}
for ax, k in zip(axes[:2], ("legacy", "fixed")):
    img = render(np.asarray(runs[k]["log"][N - 1]["qpos"]))
    ax.imshow(img)
    for kk in ("t16", "legacy", "fixed"):
        P = t16 if kk == "t16" else paths[kk]
        uv = project(P)
        ax.plot(uv[:, 0], uv[:, 1], color=col[kk], lw=2.2 if kk != "t16" else 1.2,
                ls="--" if kk == "t16" else "-", label=lab[kk], zorder=3)
        ax.plot(uv[0, 0], uv[0, 1], "o", color=col[kk], ms=6, mec="white", zorder=4)
        ax.plot(uv[-1, 0], uv[-1, 1], "s", color=col[kk], ms=6, mec="white", zorder=4)
    ua = project(apple0)[0]
    ax.plot(ua[0], ua[1], "+", color=figstyle.CATEGORICAL[2], ms=14, mew=2, zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    dr = anas[k]["finger_drift_from_entry_mm"][N - 1]
    ax.set_title((f"{k}: t={T_END} 의 MuJoCo 장면 · 손가락 이동 {dr:.1f} mm (t=72 대비)" if ko else
                  f"{k}: MuJoCo scene at t={T_END} · finger moved {dr:.1f} mm since t=72"), fontsize=10)
    sidecar["panels"][k] = {"finger_drift_from_t72_mm_at_t96": dr}
axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
axes[1].text(10, 20, "o: t=72   ■: t=96   +: 사과 중심" if ko else "o: t=72  ■: t=96  +: apple centre", fontsize=9, color="white", va="top")
# 옆에서 본 좌표 (base x–z)
ax = axes[2]
for kk in ("t16", "legacy", "fixed"):
    P = t16 if kk == "t16" else paths[kk]
    ax.plot(P[:, 0] * 1000, P[:, 2] * 1000, color=col[kk], lw=2.2 if kk != "t16" else 1.2, ls="--" if kk == "t16" else "-", label=lab[kk])
    ax.plot(P[0, 0] * 1000, P[0, 2] * 1000, "o", color=col[kk], mec="white")
    ax.plot(P[-1, 0] * 1000, P[-1, 2] * 1000, "s", color=col[kk], mec="white")
ax.add_patch(Circle((apple0[0] * 1000, apple0[2] * 1000), 45, fill=False, ec=figstyle.CATEGORICAL[2], lw=1.5, ls=":"))
ax.add_patch(Circle((apple0[0] * 1000, apple0[2] * 1000), 33, fill=True, fc=figstyle.CATEGORICAL[2], alpha=0.25, ec="none"))
ax.plot(apple0[0] * 1000, apple0[2] * 1000, "+", color=figstyle.CATEGORICAL[2], ms=12, mew=2)
ax.set_aspect("equal")
ax.set_xlabel("x (mm, world)"); ax.set_ylabel("z (mm, world)")
ax.set_title("옆에서 본 손가락 구 중심 궤적 t=72–96 (점선 원: 가정 반지름 45 mm, 채움: mesh ≈33 mm)" if ko else
             "side view, finger centre t=72–96 (dotted: assumed r=45 mm, filled: mesh ~33 mm)", fontsize=9)
ax.legend(fontsize=7, loc="upper right")
ax.set_xlim(490, 700); ax.set_ylim(790, 960)
ax.text(0.02, 0.02, "o: t=72   ■: t=96", transform=ax.transAxes, fontsize=8)
sidecar["paths_world_m"] = {k: paths[k].tolist() for k in paths}
sidecar["t16_recorded_path_base_m"] = t16.tolist()
sidecar["apple_world_t72"] = apple0.tolist()
fig.suptitle(("T23 #3 — HOLD 중 손가락 궤적: legacy vs fixed (MuJoCo 단위 재생, T16 t=0–71 실행 청크 재적용 후 t=72 부터 HOLD)" if ko else
              "T23 #3 — finger path during HOLD: legacy vs fixed (MuJoCo unit replay)"), fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "t23-hold-scene.png", dpi=130)
(FIG / "t23-hold-scene.json").write_text(json.dumps(sidecar, indent=1))
print("saved", FIG / "t23-hold-scene.png")
