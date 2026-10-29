"""IFG audit gallery — the observation modules (ATS · HSR · LTE) on the real scene, 3rd-person view. CPU only, read only.

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_audit_gallery --run <IFG-APPLE run> --fig AG3S/ag3s/docs/figures/ifg-audit --eps 250 954 56

Records: IFG-APPLE-SPLIT arm SPL (target split, current serving observation path). Per request k: the client planning row (full
MuJoCo qpos) and the server chunk record (`chunk_dir/chunk_{k+offset}.npz`, `server_chunk_alignment`).

Camera: the recording camera of `pi05_infer.py --view front` (free camera, lookat (0.45, 0, 0.85), distance 1.7 m, azimuth 180 deg,
elevation -18 deg, fovy from the model), rendered from the planning-row qpos of request k (the state the server saw).

1. `gallery-esdf-ep<E>.png` — per instant: LEFT the ESDF on the horizontal plane at the target centre height (carry: crate-wall
   height, 4 cm above the crate origin), sampled at the pixels of the same camera (each pixel ray meets the plane; grey = no hit
   within 0.6 m); RIGHT the real scene from the same camera. While held the target is excluded from GT (it is part of the robot).
   Layer: target-free (what the grasping fingers query) while the latch is searching / latched / closing, else main.
   Black line = field 0, grey dashed = 10 mm (obstacle margin), white dashed = GT 0 (probe-sphere `mj_geomDistance` to the collision
   geoms of fruits, crate, table, shelf; the target fruit excluded for the target-free layer; robot excluded as the field's
   self-filter does). Runtime-added obstacles (e.g. the 'place' profile crate post) are not in the base model, so not in GT.
2. `gallery-sequence-ep<E>.png` — real-scene frames over the episode with: VLA attention (points with attention >= 0.5, colour =
   value), the selected target cluster (green), the tracked target centre and exclusion radius (`manipulated`), the hand collision
   spheres (cyan = links allowed to touch the target, grey = other hand); text = GT stage / HSR latch / probe label / tracked state.
   Bottom: per request, attention share on the target and target-cluster precision (both vs the GT apple as a sphere of its
   collision radius, 10 mm tolerance), with the GT stage, latch and probe as bands.
"""
import argparse, ast, glob, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)
W, H = 640, 480
P = (lambda x: ast.literal_eval(x) if isinstance(x, str) and x[:1] in "{[(" else x)   # noqa: E731
STAGES = ["approach", "grasp", "transport", "placed"]
STAGE_KO = {"approach": "접근", "grasp": "잡기", "transport": "운반", "placed": "놓음 후", "searching": "찾기", "latched": "접근(고정)",
            "closing": "닫는 중", "held": "쥠", "placed_l": "놓음", "pick": "pick", "place": "place", "home": "home", "None": "-"}
SCENE_ROOTS = ("apple", "banana", "orange", "pear", "crate", "table", "shelf")


class Scene:
    def __init__(self, xml):
        import mujoco
        spec = mujoco.MjSpec.from_file(str(xml))
        b = spec.worldbody.add_body(name="audit_probe", mocap=True)
        b.add_geom(name="audit_probe_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.0005, 0, 0], contype=0, conaffinity=0,
                   rgba=[0, 0, 0, 0])
        self.mj = mujoco
        self.m = spec.compile(); self.d = mujoco.MjData(self.m)
        self.probe = self.m.geom("audit_probe_geom").id
        self.r = mujoco.Renderer(self.m, H, W)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.45, 0.0, 0.85]; cam.distance = 1.7; cam.azimuth = 180.0; cam.elevation = -18.0
        self.cam = cam
        m = self.m
        self.geoms = {}
        for i in range(m.ngeom):
            root = m.body(m.body_rootid[m.geom_bodyid[i]]).name
            if root in SCENE_ROOTS and m.geom_contype[i]:
                self.geoms.setdefault(root, []).append(i)

    def set(self, qpos):
        self.d.qpos[:len(qpos)] = qpos
        self.mj.mj_forward(self.m, self.d)

    def render(self):
        self.r.update_scene(self.d, camera=self.cam)
        img = self.r.render()
        gc = self.r.scene.camera[0]
        self.pos = np.array(gc.pos); self.fwd = np.array(gc.forward); self.up = np.array(gc.up)
        self.right = np.cross(self.fwd, self.up); self.near = gc.frustum_near; self.ht = gc.frustum_top; self.hw = self.ht * W / H
        return img

    def project(self, p):
        v = np.atleast_2d(p) - self.pos
        zc = v @ self.fwd
        u = ((v @ self.right) * self.near / zc + self.hw) / (2 * self.hw) * W
        vv = (self.ht - (v @ self.up) * self.near / zc) / (2 * self.ht) * H
        return u, vv, zc

    def plane_points(self, c, box, step=2, normal=None):
        """World points where the camera rays of the pixels of `box` (u0, u1, v0, v1; every `step` px) meet the camera-facing
        plane through c. Returns (points[nv, nu, 3], pixel centres u, v)."""
        u0, u1, v0, v1 = box
        us = np.arange(u0, u1, step) + 0.5 * step; vs = np.arange(v0, v1, step) + 0.5 * step
        U, V = np.meshgrid(us, vs)
        xn = U / W * 2 * self.hw - self.hw; yn = self.ht - V / H * 2 * self.ht
        dirs = self.fwd[None, None] * self.near + self.right[None, None] * xn[..., None] + self.up[None, None] * yn[..., None]
        n = self.fwd if normal is None else np.asarray(normal, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((c - self.pos) @ n) / (dirs @ n)
        t = np.where(t > 0, t, np.nan)
        return self.pos + dirs * t[..., None], us, vs

    def gt_distance(self, pts, exclude=(), dmax=0.12):
        """Distance (m) from each point to the nearest collision geom of the scene objects (not the robot), capped at dmax."""
        m, d, mj = self.m, self.d, self.mj
        ids = [g for root, gs in self.geoms.items() if root not in exclude for g in gs]
        centres = d.geom_xpos[ids]; rb = m.geom_rbound[ids]
        out = np.full(len(pts), dmax)
        ft = np.zeros(6)
        for i, p in enumerate(pts):
            lb = np.linalg.norm(centres - p, axis=1) - rb
            cand = [ids[j] for j in np.flatnonzero(lb < dmax)]
            if not cand:
                continue
            d.geom_xpos[self.probe] = p
            out[i] = min(dmax, min(mj.mj_geomDistance(m, d, self.probe, g, dmax, ft) for g in cand) + 0.0005)
        return out


def load_episode(K, ep):
    jf = glob.glob(str(K / "replay" / f"runs_SPL_SPL_ep{ep}_*.json"))[0]
    al = json.load(open(jf))["server_chunk_alignment"]["block"]
    rd = next(p for p in glob.glob(str(K / "runs" / "SPL" / f"SPL_ep{ep}_s*")) if pathlib.Path(p).is_dir())
    rows = [json.loads(l) for l in open(rd + "/frames.jsonl") if '"kind": "planning"' in l or '"kind": "control"' in l]
    plan = [r for r in rows if r["kind"] == "planning"]
    ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}
    man = json.load(open(rd + "/manifest.json"))["scene"]
    return {"al": al, "plan": plan, "ctrl": ctrl, "fruit": man["episode_target_fruit"], "arm": man["episode_used_arm"]}


def in_crate(op, fruit):
    w, x, y, z = op["crate"]["quat"]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    loc = R.T @ (np.asarray(op[fruit]["pos"]) - np.asarray(op["crate"]["pos"]))
    return bool(np.all(np.abs(loc[:2]) < np.array([0.077, 0.137])) and -0.052 < loc[2] < 0.075)


def gt_stages(E):
    """GT stage per request: approach (before the used arm's first close command), grasp (closing, fruit not lifted 1 cm),
    transport (lifted, not in the crate), placed (in the crate)."""
    gi = 0 if E["arm"] == "left" else 1
    z0 = E["plan"][0]["object_poses"][E["fruit"]]["pos"][2]
    tclose = next((t for t, c in sorted(E["ctrl"].items()) if ((c.get("applied_ctrl") or {}).get("gripper") or [1, 1])[gi] < 0.5), None)
    out = []
    for r in E["plan"]:
        op = r["object_poses"]; t = r["t_step"]
        if in_crate(op, E["fruit"]):
            out.append("placed")
        elif op[E["fruit"]]["pos"][2] - z0 >= 0.01:
            out.append("transport")
        elif tclose is not None and t >= tclose:
            out.append("grasp")
        else:
            out.append("approach")
    return out


def arr(z, key, shape=(0, 3)):
    return z[key] if key in z.files else np.zeros(shape)


def chunk(E, k):
    z = np.load(f"{E['al']['chunk_dir']}/chunk_{k + E['al']['offset']:05d}.npz", allow_pickle=True)
    return z, json.loads(str(z["summary_json"]))


def per_request(S, E, k, gt_r):
    """attention share on the target, target-cluster precision (GT target fruit as a sphere, 10 mm tolerance)."""
    z, s = chunk(E, k)
    op = E["plan"][k]["object_poses"]; c = np.asarray(op[E["fruit"]]["pos"])
    pts = arr(z, "filtered_cloud"); att = arr(z, "attention_values", (0,)).astype(float)
    near = np.linalg.norm(pts - c, axis=1) <= gt_r + 0.010
    top = att >= 0.5
    share = float(near[top].mean()) if top.any() else float("nan")
    tp = arr(z, "target_points")
    g = P(s.get("grasp")) or {}; sub = P(s.get("subtask")) or {}; man = P(s.get("manipulated")) or {}
    pre = str(g.get("state")) in ("searching", "latched", "closing")
    prec = float((np.linalg.norm(tp - c, axis=1) <= gt_r + 0.010).mean()) if len(tp) and pre else float("nan")
    return {"k": k, "t": int(E["plan"][k]["t_step"]), "att_share": share, "precision": prec, "n_target": int(len(tp)),
            "latch": str(g.get("state")), "probe": str(sub.get("label")), "tracked": str(man.get("state"))}


def esdf_figure(S, E, ep, ks, gt, out):
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field
    HALF_U, HALF_V = 130, 98
    fig, axes = plt.subplots(len(ks), 2, figsize=(14, 5.2 * len(ks)))
    axes = np.atleast_2d(axes); FS.style_axes(fig, list(axes.ravel()))
    cmap = FS.sequential_cmap().copy(); cmap.set_bad(FS.GRID_INK)
    sidecar = []
    for row, k in zip(axes, ks):
        z, s = chunk(E, k)
        latch = str((P(s.get("grasp")) or {}).get("state"))
        S.set(E["plan"][k]["qpos"]); img = S.render()
        op = E["plan"][k]["object_poses"]
        c = np.asarray(op[E["fruit"]]["pos"])
        held = latch == "held"
        cut = c if not held else np.asarray(op["crate"]["pos"]) + np.array([0.0, 0.0, 0.04])   # carry: crate-wall height
        uc, vc, _ = S.project(c if not held else (c + cut) / 2)
        box = (int(uc[0]) - HALF_U, int(uc[0]) + HALF_U, int(vc[0]) - HALF_V, int(vc[0]) + HALF_V)
        grid, xs, ys = S.plane_points(cut, box, step=2, normal=(0.0, 0.0, 1.0)); nv, nu = grid.shape[:2]
        pts = grid.reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts - cut, axis=1) < 0.6)
        pts = np.where(valid[:, None], pts, cut)
        field = build_field(z, outside_distance=1.0)
        free = latch in ("searching", "latched", "closing") and field.has_target_free
        dist = np.asarray(field.target_free_distance(pts) if free else field.distance(pts), float)
        g = S.gt_distance(pts, exclude=(E["fruit"],) if (free or held) else ())   # held: the apple is part of the robot
        dist[~valid] = np.nan; g[~valid] = np.nan
        dist = dist.reshape(nv, nu); g = g.reshape(nv, nu)
        ax, bx = row
        ext = (box[0], box[1], box[3], box[2])
        shown = np.clip(dist, 0, 0.08)
        ax.imshow(shown, cmap=cmap, vmin=0, vmax=0.08, extent=ext)
        inside = np.ma.masked_where(~(dist <= 0), np.ones_like(dist))
        ax.imshow(inside, cmap=matplotlib.colors.ListedColormap([FS.CATEGORICAL[7]]), alpha=0.55, extent=ext)
        ax.contour(xs, ys, dist, levels=[0.0], colors=[FS.INK], linewidths=1.2)
        ax.contour(xs, ys, dist, levels=[0.010], colors=[FS.INK_2], linewidths=0.8, linestyles="--")
        ax.contour(xs, ys, g, levels=[0.0015], colors=["white"], linewidths=1.3, linestyles="--")
        u, v, _ = S.project(c); ax.plot(u, v, "+", color="white", ms=10)
        bx.imshow(img); bx.plot(u, v, "+", color=FS.CATEGORICAL[2], ms=12, mew=2)
        for a in (ax, bx):
            a.set_xlim(box[0], box[1]); a.set_ylim(box[3], box[2]); a.set_xticks([]); a.set_yticks([])
        ok = np.isfinite(g) & np.isfinite(dist)
        err = (dist - g)[ok & (np.abs(np.nan_to_num(g, nan=1.0)) < 0.03)]
        stage = gt[k]
        ax.set_title(T(f"ep{ep} 요청 {k} (t={E['plan'][k]['t_step'] / 15:.1f} s) · {'바구니 벽 높이' if held else '사과 중심 높이'} 수평 단면 · {'target 제외 층 (손가락이 보는 거리장)' if free else 'main 층'}\n"
                       f"GT 단계 {STAGE_KO[stage]} · HSR {STAGE_KO.get(latch, latch)} · GT 3 cm 이내 오차 중앙값 {np.median(err) * 1000:+.1f} mm",
                       f"ep{ep} request {k} · {'target-free' if free else 'main'} layer\nGT {stage} · latch {latch}"), fontsize=9, loc="left")
        bx.set_title(T("같은 시점의 실제 씬 (3rd person, + = target 사과 중심)", "real scene, same camera"), fontsize=9, loc="left")
        sidecar.append({"k": k, "t_step": int(E["plan"][k]["t_step"]), "latch": latch, "gt_stage": stage, "layer": "target_free" if free else "main",
                        "slice_centre_m": cut.round(4).tolist(), "slice": "horizontal", "gt_excludes_target": bool(free or held), "err_within_30mm_median_mm": float(np.median(err) * 1000) if err.size else None,
                        "err_within_30mm_p95_abs_mm": float(np.percentile(np.abs(err), 95) * 1000) if err.size else None,
                        "n_err_points": int(err.size)})
    fig.suptitle(T(f"거리장 vs 실제 씬 — ep{ep}. 왼쪽: target 사과 중심 높이의 수평면 ESDF를 같은 카메라로 본 것 (색 = 거리 0–8 cm, 빨강 = 거리장 안쪽, 검정선 = 거리장 0, "
                   "회색 점선 = 10 mm, 흰 점선 = GT 물체 경계). 오른쪽: 같은 카메라의 실제 씬",
                   f"ESDF vs real scene, ep{ep}"), fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    p = out / f"gallery-esdf-ep{ep}.png"
    fig.savefig(p, dpi=130, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps({"episode": ep, "instants": sidecar, "definitions": __doc__,
                                                  "script": "AG3S/ag3s/experiments/studies/ifg_audit_gallery.py"}, indent=1))
    return p


def sequence_figure(S, E, ep, ks, gt, series, out):
    n = len(ks); ncol = 3; nrow = -(-n // ncol)
    fig = plt.figure(figsize=(18, 5.0 * nrow + 5.2))
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[1.0] * nrow + [0.95])
    axes = [fig.add_subplot(gs[i // ncol, i % ncol]) for i in range(n)]; tx = fig.add_subplot(gs[nrow, :])
    FS.style_axes(fig, axes + [tx])
    side = []
    for ax, k in zip(axes, ks):
        z, s = chunk(E, k)
        S.set(E["plan"][k]["qpos"]); img = S.render()
        ax.imshow(img)
        pts = arr(z, "filtered_cloud"); att = arr(z, "attention_values", (0,)).astype(float)
        sel = att >= 0.5
        if sel.any():
            u, v, _ = S.project(pts[sel]); ax.scatter(u, v, c=att[sel], cmap="magma", vmin=0.5, vmax=1, s=4, alpha=0.6, lw=0)
        tp = arr(z, "target_points")
        if len(tp):
            u, v, _ = S.project(tp); ax.scatter(u, v, s=5, color=FS.CATEGORICAL[2], lw=0)
        man = P(s.get("manipulated")) or {}
        if man.get("centroid") is not None:
            cc = np.asarray(man["centroid"]); u, v, zc = S.project(cc)
            rad = float(man.get("radius") or 0) * (S.near / zc[0]) / (2 * S.hw) * W
            ax.add_patch(plt.Circle((u[0], v[0]), rad, fill=False, color=FS.CATEGORICAL[2], lw=1.2, ls="--"))
            ax.plot(u, v, "x", color=FS.CATEGORICAL[2], ms=8, mew=2)
        names = [str(x) for x in (s.get("sphere_link_names") or [])]
        auth = set((P(s.get("target_permission")) or {}).get("authorized_links") or [])
        sc = arr(z, "sphere_centres")[:len(names)]
        if len(sc):
            u, v, _ = S.project(sc)
            col = [FS.CATEGORICAL[0] if nm in auth else FS.GRID_INK for nm in names]
            ax.scatter(u, v, s=1.5, c=col, lw=0, alpha=0.7)
        g = P(s.get("grasp")) or {}; sub = P(s.get("subtask")) or {}
        ax.set_xlim(130, 530); ax.set_ylim(335, 95); ax.set_xticks([]); ax.set_yticks([])
        latch = str(g.get("state"))
        ax.set_title(T(f"요청 {k} · t={E['plan'][k]['t_step'] / 15:.1f} s\nGT {STAGE_KO[gt[k]]} | HSR {STAGE_KO.get(latch, latch)} | probe {sub.get('label')}\n"
                       f"추적 {man.get('state')} · target 점 {len(tp)}",
                       f"k {k} · GT {gt[k]} | latch {latch} | probe {sub.get('label')}\ntracked {man.get('state')} · {len(tp)} pts"),
                     fontsize=8.5, loc="left")
        side.append({"k": k, "gt_stage": gt[k], "latch": latch, "probe": sub.get("label"), "tracked": man.get("state"),
                     "n_target_points": int(len(tp)), "authorized_links": sorted(auth)})
    ks_all = [r["k"] for r in series]
    tx.plot(ks_all, [r["att_share"] for r in series], color=FS.CATEGORICAL[3], lw=1.6, label=T("상위 attention 점(≥ 0.5) 중 사과 위 비율", "top-attention share on target"))
    tx.plot(ks_all, [r["precision"] for r in series], color=FS.CATEGORICAL[2], lw=1.6, label=T("target 덩어리 precision (쥐기 전만)", "target-cluster precision (pre-grasp)"))
    for k in ks:
        tx.axvline(k, color=FS.GRID_INK, lw=0.8)
    bands = [("GT", [gt[r["k"]] for r in series], {"approach": 0, "grasp": 1, "transport": 2, "placed": 3}),
             ("HSR", [r["latch"] for r in series], {"searching": 0, "latched": 0, "closing": 1, "held": 2, "placed": 3}),
             ("probe", [r["probe"] for r in series], {"pick": 0, "place": 2, "home": 3})]
    for j, (lab, vals, code) in enumerate(bands):
        y0 = -0.12 - 0.11 * j
        for k, vv in zip(ks_all, vals):
            ci = code.get(vv)
            if ci is not None:
                tx.add_patch(plt.Rectangle((k - 0.5, y0), 1, 0.09, color=FS.CATEGORICAL[[0, 1, 5, 6][ci]], lw=0))
        tx.text(-1.5, y0 + 0.045, lab, ha="right", va="center", fontsize=8)
    kpl = next((r["k"] for r in series if r["gt_stage"] == "placed"), max(ks_all))
    tx.set_ylim(-0.48, 1.05); tx.set_xlim(-0.5, min(max(ks_all), kpl + 3) + 0.5)
    tx.set_xlabel(T("요청 (chunk)", "request"), fontsize=8.5)
    from matplotlib.patches import Patch
    handles, labels = tx.get_legend_handles_labels()
    handles += [Patch(color=FS.CATEGORICAL[i], label=l) for i, l in zip([0, 1, 5, 6], [T("pick: 접근", "approach"), T("pick: 잡기", "grasp"),
                                                                                      T("place: 운반", "transport"), T("놓음 후", "placed")])]
    tx.legend(handles=handles, fontsize=7.5, frameon=False, ncol=3, loc="upper right")
    tx.set_title(T("요청별 attention · 분할 정확도와 단계 판정 (아래 띠: GT / HSR latch / VLA probe)", "per request"), fontsize=9, loc="left")
    fig.suptitle(T(f"ep{ep} ({E['arm']} 손) — 실제 씬에서 본 ATS · HSR: 보라 = VLA attention(≥ 0.5), 초록 = 선택한 target 덩어리 · 추적 중심(x) · 예외 반경(점선), "
                   "파랑 = 사과에 닿아도 되는 손 구, 회색 = 다른 손 구 (팔 링크는 충돌 모델에 없음)", f"ep{ep} ATS / HSR on the real scene"),
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = out / f"gallery-sequence-ep{ep}.png"
    fig.savefig(p, dpi=130, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps({"episode": ep, "frames": side, "series": series, "definitions": __doc__,
                                                  "script": "AG3S/ag3s/experiments/studies/ifg_audit_gallery.py"}, indent=1))
    return p


def pick_instants(E, gt, n_req):
    latches = []
    for k in range(n_req):
        _, s = chunk(E, k)
        latches.append(str((P(s.get("grasp")) or {}).get("state")))
    first = lambda st: next((k for k, l in enumerate(latches) if l == st), None)   # noqa: E731
    last = lambda st: max((k for k, l in enumerate(latches) if l == st), default=None)   # noqa: E731
    lat0, latN, cl0, he0, heN, pl0 = first("latched"), last("latched"), first("closing"), first("held"), last("held"), first("placed")
    mid_held = None if he0 is None else (he0 + heN) // 2
    esdf = [k for k in (lat0 if lat0 is None else max(lat0, (lat0 + (latN or lat0)) // 2), latN, cl0, mid_held) if k is not None]
    seq = sorted(dict.fromkeys(k for k in (lat0, latN, cl0, he0, mid_held, pl0) if k is not None))
    if len(seq) < 6:   # a failed grasp has no held / placed: fill evenly between the first and the last request
        for k in np.linspace(seq[0] if seq else 0, n_req - 1, 8).astype(int):
            if len(seq) >= 6:
                break
            if all(abs(int(k) - x) > 3 for x in seq):
                seq.append(int(k))
    return sorted(dict.fromkeys(esdf)), sorted(seq), latches


def links_table(K, out):
    """Per HSR latch state over all runs of the kit: which links may touch the target and how the target is kept out of their way."""
    import zipfile
    from collections import Counter, defaultdict
    rows = defaultdict(Counter); n = Counter()
    for jf in sorted(glob.glob(str(K / "replay" / "runs_SPL_SPL_ep*.json"))):
        al = json.load(open(jf))["server_chunk_alignment"]["block"]
        for k in range(int(al["block_len"])):
            with zipfile.ZipFile(f"{al['chunk_dir']}/chunk_{k + al['offset']:05d}.npz") as zf:
                s = json.loads(str(np.load(zf.open("summary_json.npy"), allow_pickle=True)))
            g = P(s.get("grasp")) or {}; st = str(g.get("state")); n[st] += 1
            ex = P(s.get("exclusion")) or {}
            rows[st]["+".join(ex.get("mechanism") or []) or "none"] += 1
            names = [str(x) for x in s.get("sphere_link_names") or []]
    order = ["searching", "latched", "closing", "held", "placed"]
    task = {"searching": "pick · 찾기", "latched": "pick · 접근", "closing": "pick · 잡기", "held": "place · 운반", "placed": "place · 놓은 뒤"}
    mech_ko = {"target_free_layer+contact_margin": "사과를 뺀 거리장을 손가락 · 손바닥만 봄",
               "attached_query+held_free+self_filter": "쥔 사과를 손에 붙은 구로 검사, 그 자리는 거리장에서 비움",
               "none": "예외 없음 (사과도 일반 물체)"}
    cell = [[task[st], str(n[st]), "\n".join(f"{mech_ko.get(m, m)}  ({c})" for m, c in rows[st].most_common())] for st in order if n[st]]
    fig, ax = plt.subplots(figsize=(13, 3.6)); ax.axis("off")
    tb = ax.table(cellText=cell, colLabels=[T("작업 단계 (HSR latch)", "stage"), T("요청 수", "requests"), T("target 사과 처리 (요청 수)", "target handling")],
                  loc="center", cellLoc="left", colWidths=[0.18, 0.08, 0.74])
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 2.6)
    link_counts = Counter(names)
    ax.set_title(T(f"단계별 링크 처리 — 사과 24 장면 1,800 요청. 충돌 모델 = 손 링크 6개의 구 {sum(link_counts.values())}개 "
                   f"({', '.join(f'{k} {v}' for k, v in sorted(link_counts.items()))}); 팔 · 손목 · 팔꿈치는 검사하지 않음. "
                   "사과에 닿아도 되는 링크 = 쥐는 손의 손바닥 + 손가락 2개", "link handling by stage"), fontsize=9.5, loc="left")
    p = out / "gallery-links-table.png"
    fig.savefig(p, dpi=150, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps({"requests": dict(n), "mechanism_by_state": {k: dict(v) for k, v in rows.items()},
                                                  "sphere_links": dict(link_counts), "script": "AG3S/ag3s/experiments/studies/ifg_audit_gallery.py"},
                                                 indent=1, ensure_ascii=False))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--fig", type=pathlib.Path, required=True)
    ap.add_argument("--eps", type=int, nargs="+", default=[250, 954, 211, 56])
    ap.add_argument("--links-table", action="store_true")
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    K = a.run / "verify/IFG-APPLE-SPLIT"
    if a.links_table:
        print(links_table(K, a.fig))
    S = Scene(a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
    for ep in a.eps:
        E = load_episode(K, ep)
        gt = gt_stages(E)
        n_req = int(E["al"]["block_len"])
        gt_r = 0.033   # apple collision mesh half-extents 31 / 33 / 33 mm (model geom_size)
        ks_esdf, ks_seq, _ = pick_instants(E, gt, n_req)
        series = [per_request(S, E, k, gt_r) for k in range(n_req)]
        for r, st in zip(series, gt):
            r["gt_stage"] = st
        print(esdf_figure(S, E, ep, ks_esdf, gt, a.fig))
        print(sequence_figure(S, E, ep, ks_seq, gt, series, a.fig))


if __name__ == "__main__":
    main()
