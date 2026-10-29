"""T46f A14 — 손목 보정 형상 (`--links gripper_wrist`, `wrist_cover`) 의 coverage audit.

보정한 link (`link_*_arm_5` · `link_*_arm_6` · `FT_sensor_*`) 마다:

1. **형상 누락** — MJCF collision 겉면 표본 (구를 고를 때 쓴 1 mm 보다 2 배 촘촘한 0.5 mm) 중 구 합집합 밖 비율 · 밖 거리
   (median / p95 / max). 대상 = 강체 이웃 (손바닥 · arm_6 · FT) 에 묻히지 않은 겉면 (`reachable`) 과 겉면 전부 (`all`).
   collision mesh 정점도 같이. 보정 전 모델 (`--links arms`: URDF arm_5 capsule · arm_6 gap-filling 사슬 · FT 구 없음) 과
   나란히.
2. **과도한 팽창** — 구 표면 표본 중 자기 mesh 껍질 밖 점의 자기 겉면까지 거리 max / p95 (실측 넘침). 그 중 강체 이웃
   (로봇 자신) 안에 든 점을 뺀 것 (`into_free_space`) 도.
3. **손 영역 침범** — 비권한 구 (보정 link) 와 같은 쪽 권한 손의
   (a) MJCF collision geom (손바닥 `EE_BODY_*` + 손가락 둘): **world frame**, MuJoCo FK 로 놓은 geom 과 URDF FK 로 놓은 구.
       팔 관절 · 손가락을 무작위로 (손가락은 MJCF 범위 안). 침범 = `r − dist(c, hull)` > 0. dist 는 지지 평면 하한
       (`lb`, 보수적 — 침범을 과대평가할 수 있다) 과 0.5 mm 표본 거리 (`sampled`) 둘 다.
   (b) 권한 구 (같은 제약 모델의 `ee_<S>` · `ee_finger_<s>*` 구): 구–구 겹침 깊이 `r1 + r2 − |c1 − c2|`.
   같은 자세에서 **frame 검사**: 보정 구 중심의 MJCF 자기 geom 깊이 (world) 와 link frame 깊이의 차.

MuJoCo 와 URDF 의 손가락 이름이 왼손에서 뒤바뀌어 있다 (T43 OC) — 이 audit 은 MJCF 손 geom 을 **MJCF body 이름**
(`EE_BODY_L`, `ee_finger_l1`, `ee_finger_l2` — 두 손가락 모두) 으로 고르므로 짝짓기가 없다.

    MUJOCO_GL=osmesa PYTHONPATH=<checkout> python -m AG3S.ag3s.experiments.reports.wrist_cover_audit \
        --out $AG3S_RUN_DIR/impl/T46f-A14/audit.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

#: CL3 / T44d 서버 flag (`--capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32`).
SERVER = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)

#: 판정 기준 — **측정 전에 고정** (T46f §8 "단계 1 audit 에서 FT sensor 형상 누락이나 손 영역 침범이 남으면 단계 2 로
#: 넘어가지 않는다"). 표본 (0.5 mm) 은 구를 고른 표본 (1 mm) 보다 촘촘하므로 그 사이 조각이 밖에 남는 것은 표본 간격
#: 크기 (≤ 0.5 mm) 까지 허용한다. FT 는 손 keep-out 이 맞닿은 띠 (손바닥 홈) 를 다 담지 못할 수 있어 2.5 mm
#: (`gripper_cover.COVER_MAX_GAP`) 까지.
THRESHOLDS = {
    "reachable_max_outside_mm": {"arm_5": 0.5, "arm_6": 0.5, "FT": 2.5},
    "reachable_frac_outside_gt_0p5mm": {"arm_5": 0.001, "arm_6": 0.001, "FT": 0.01},
    "overflow_max_excess_mm": 0.5,        # 넘침 실측 ≤ overflow + 0.5 mm
    # 침범 = 0 이 목표다. 구는 keep-out 상한으로 손 껍질에 **접할 수 있고** (r = 지지 평면 거리), world frame 에서 URDF FK
    # 와 MuJoCo FK 의 부동소수 차 (frame 검사 ~0.5 µm) 만큼 수치 침범이 보인다 → 1 µm 까지는 수치 오차로 본다.
    # (smoke 실행 4 자세에서 0.36 µm 를 보고 0.0 → 0.001 mm 로 정했다 — 본 측정 전, impl.md 에 기록.)
    "mjcf_hand_penetration_sampled_mm": 0.001,
    "mjcf_hand_penetration_lb_mm": 0.001,  # 지지 평면 하한 (보수적; 부동소수 여유 1 µm)
    "frame_depth_diff_mm": 0.01,
    # 권한 구와의 겹침은 권한 구가 MJCF 손 밖으로 나온 만큼 (palm 사슬 · palm cover overflow) 까지만 허용
    "auth_overlap_excess_over_auth_protrusion_mm": 0.01,
}

SIDES = ("left", "right")


def _kind(link: str) -> str:
    return "FT" if link.startswith("FT_sensor") else ("arm_5" if link.endswith("arm_5") else "arm_6")


def _gap_stats(gap: np.ndarray) -> dict:
    out = gap[gap > 0.0]
    return {
        "n": int(gap.size),
        "frac_outside": float(np.mean(gap > 0.0)) if gap.size else None,
        "frac_outside_gt_0p5mm": float(np.mean(gap > 0.0005)) if gap.size else None,
        "outside_median_mm": float(np.median(out) * 1e3) if out.size else 0.0,
        "outside_p95_mm": float(np.percentile(out, 95) * 1e3) if out.size else 0.0,
        "outside_max_mm": float(out.max() * 1e3) if out.size else 0.0,
    }


def _overflow(C, R, surface, neighbour_hulls, n_dir: int = 2000) -> dict:
    """구 표면 표본 중 자기 껍질 밖 점 → 자기 겉면 표본까지 거리 (실측 넘침)."""
    from scipy.spatial import cKDTree

    from AG3S.ag3s.experiments.sources.gripper_cover import hull_depth

    if R.size == 0:
        return {"max_mm": None, "p95_mm": None, "into_free_space_max_mm": None}
    tree = cKDTree(surface.points)
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n_dir, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    d_all, d_free = [], []
    for c, r in zip(C, R):
        P = c + r * v
        out = P[hull_depth(P, surface.hulls) < 0.0]
        if not out.size:
            continue
        d = tree.query(out)[0]
        d_all.append(d)
        if neighbour_hulls:
            d_free.append(d[hull_depth(out, neighbour_hulls) < 0.0])
        else:
            d_free.append(d)
    d_all = np.concatenate(d_all) if d_all else np.zeros(0)
    d_free = np.concatenate(d_free) if d_free else np.zeros(0)
    return {"max_mm": float(d_all.max() * 1e3) if d_all.size else 0.0,
            "p95_mm": float(np.percentile(d_all, 95) * 1e3) if d_all.size else 0.0,
            "into_free_space_max_mm": float(d_free.max() * 1e3) if d_free.size else 0.0,
            "n_dir_per_sphere": int(n_dir)}


def _neighbour_hulls(model, surface_body: str, T_link_body: np.ndarray):
    """강체 이웃 (같은 강체 묶음의 다른 body) collision 껍질 — link frame."""
    from AG3S.ag3s.experiments.sources.gripper_cover import _geom_pieces, _hull_eq, _rigid_neighbours
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, surface_body)
    hulls = []
    for other, T_bo in _rigid_neighbours(model, bid):
        T = T_link_body @ T_bo
        for verts in _geom_pieces(model, other):
            if verts.shape[0] >= 4:
                hulls.append(_hull_eq(verts @ T[:3, :3].T + T[:3, 3]))
    return tuple(hulls)


def coverage(model_mj, models: dict, links, spacing: float) -> dict:
    from AG3S.ag3s.experiments.sources import gripper_cover as gc

    out = {}
    ref = models["after"]
    for link in links:
        surf = gc.wrist_surface(model_mj, ref, link, spacing=spacing)
        outer = gc.finger_body_targets(surf)
        reach = outer & ~surf.hidden
        T = gc._mjcf_body_in_link(model_mj, ref, link, surf.body)
        neigh = _neighbour_hulls(model_mj, surf.body, T)
        rec = {"mujoco_body": surf.body, "n_samples_outer": int(outer.sum()),
               "n_samples_reachable": int(reach.sum()), "n_hidden": int(surf.hidden.sum()),
               "sample_spacing_mm": spacing * 1e3}
        # mesh 정점 중 강체 이웃 껍질 안 (0.5 mm 안) 이 아닌 것 — 닿을 수 있는 정점
        v_reach = (gc.hull_depth(surf.vertices, neigh) < -gc._NEIGHBOUR_TOL) if neigh else \
            np.ones(surf.vertices.shape[0], bool)
        keep = gc.hand_keep_out(model_mj, ref, link)
        rec["n_vertices"], rec["n_vertices_reachable"] = int(surf.vertices.shape[0]), int(v_reach.sum())
        for tag, m in models.items():
            c, r = m.local_spheres(link)
            roles = sorted({ro for n, ro in zip(m.sphere_link_names, m.sphere_roles) if n == link})
            g_reach = gc._gap(surf.points[reach], c, r)
            rec[tag] = {
                "n_spheres": int(r.size), "roles": roles,
                "radius_mm": [float(r.min() * 1e3), float(r.max() * 1e3)] if r.size else None,
                "reachable": _gap_stats(g_reach),
                "all_outer": _gap_stats(gc._gap(surf.points[outer], c, r)),
                "vertices": _gap_stats(gc._gap(surf.vertices, c, r)),
                "vertices_reachable": _gap_stats(gc._gap(surf.vertices[v_reach], c, r)),
                "overflow": _overflow(c, r, surf, neigh),
            }
            # 밖에 남은 (> 0.5 mm) 닿을 수 있는 표본이 어디 있나: 강체 이웃 · 권한 손 keep-out 까지 거리
            far = g_reach > 0.0005
            # T46f A14b: 0 < 밖 ≤ 0.5 mm 표본도 같은 거리 (강체 이웃 · 손 keep-out · z) 를 싣는다 — 분류 기준은 없다
            # (보고용 거리만; 위치 분류는 > 0.5 mm 표본에만 근거가 있다, 사용자 판정 A1)
            near_out = (g_reach > 0.0) & ~far
            for key, sel in (("outside_gt_0p5mm_where", far), ("outside_le_0p5mm_where", near_out)):
                if not (r.size and sel.any()):
                    continue
                P = surf.points[reach][sel]
                rec[tag][key] = {
                    "n": int(sel.sum()),
                    "dist_to_rigid_neighbour_mm": (
                        [float(x) for x in np.percentile(-gc.hull_depth(P, neigh) * 1e3, [0, 50, 100])]
                        if neigh else None),
                    "dist_to_hand_keep_out_mm": [float(x) for x in np.percentile(
                        gc.keep_out_clearance(P, keep) * 1e3, [0, 50, 100])],
                    "z_link_mm": [float(P[:, 2].min() * 1e3), float(P[:, 2].max() * 1e3)],
                }
        out[link] = rec
    return out


def _hand_geoms(model_mj, side: str):
    """같은 쪽 손의 MJCF collision 조각: `[(body_id, verts_body, A_body, b_body, samples_body)]` — MJCF body 이름."""
    from AG3S.ag3s.experiments.sources.gripper_cover import _geom_pieces, _hull_eq, _hull_samples
    import mujoco

    s = "L" if side == "left" else "R"
    f = "l" if side == "left" else "r"
    out = []
    for body in (f"EE_BODY_{s}", f"ee_finger_{f}1", f"ee_finger_{f}2"):
        bid = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, body)
        for verts in _geom_pieces(model_mj, bid):
            from scipy.spatial import cKDTree

            A, b = _hull_eq(verts)
            pts, _, _ = _hull_samples(verts, 0.0005)
            out.append((bid, body, A, b, cKDTree(pts)))
    return out


def _own_geoms(model_mj, body: str):
    from AG3S.ag3s.experiments.sources.gripper_cover import _geom_pieces, _hull_eq
    import mujoco

    bid = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, body)
    return bid, [_hull_eq(v) for v in _geom_pieces(model_mj, bid)]


def _world(data, bid):
    T = np.eye(4)
    T[:3, :3] = data.xmat[bid].reshape(3, 3)
    T[:3, 3] = data.xpos[bid]
    return T


def hand_overlap(scene, models: dict, links, n_poses: int, seed: int) -> dict:
    """world frame: 비권한 (보정) 구 대 같은 쪽 MJCF 손 geom · 권한 구. frame 검사도."""
    import mujoco

    from AG3S.ag3s.experiments.sources import gripper_cover as gc
    from AG3S.ag3s.experiments.reports.grounding_report import WRIST_LINKS_BY_SIDE

    mj, data = scene.model, mujoco.MjData(scene.model)
    qadr = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i): int(mj.jnt_qposadr[i])
            for i in range(mj.njnt)}
    jrange = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i): mj.jnt_range[i].copy()
              for i in range(mj.njnt)}
    base_id = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "base")
    hands = {side: _hand_geoms(mj, side) for side in SIDES}
    rng = np.random.default_rng(seed)
    ref = models["after"]
    lo, hi = ref.joint_limits()
    own = {link: _own_geoms(mj, gc.WRIST_MJCF_BODY.get(link, link)) for link in links}  # (body id, hulls)
    # link frame 깊이 (구 중심) — frame 검사의 기준
    local_depth = {}
    for link in links:
        surf = gc.wrist_surface(mj, ref, link)
        c, _ = ref.local_spheres(link)
        local_depth[link] = gc.hull_depth(c, surf.hulls)
    fingers_mj = [n for n in qadr if n and n.startswith("gripper_finger_")]
    res = {tag: {} for tag in models}
    frame_diff = 0.0
    for tag, m in models.items():
        names = np.asarray(m.sphere_link_names)
        for side in SIDES:
            for link in WRIST_LINKS_BY_SIDE[side]:
                res[tag][link] = {"n_spheres": int((names == link).sum()),
                                  "mjcf_pen_lb_max_mm": -np.inf, "mjcf_pen_sampled_max_mm": -np.inf,
                                  "n_sphere_pose_pen_lb": 0, "n_sphere_pose_pen_sampled": 0,
                                  "auth_overlap_max_mm": -np.inf, "n_pair_pose_overlap": 0,
                                  "pen_by_mjcf_body": {}}
    urdf_fingers = [n for n in ref.param_joint_names]
    t0 = time.time()
    for k in range(n_poses):
        q = rng.uniform(lo, hi)
        mujoco.mj_resetData(mj, data)
        for i, name in enumerate(ref.joint_names):
            if name in qadr:
                data.qpos[qadr[name]] = q[i]
        # MJCF 손가락 — 범위 안 무작위 (열림 · 닫힘 끝 포함)
        fvals = {}
        for name in fingers_mj:
            a, b = jrange[name]
            u = rng.choice([0.0, 1.0, rng.uniform()]) if k % 3 else (0.0 if k % 2 else 1.0)
            fvals[name] = float(a + u * (b - a))
            data.qpos[qadr[name]] = fvals[name]
        mujoco.mj_kinematics(mj, data)
        T_wb = _world(data, base_id)
        # 손 geom (world)
        # (body frame 의 껍질 · 표본 트리, 그 body 의 world pose) — 구 중심을 body frame 으로 옮겨 잰다
        hw = {side: [(body, A, b, tree, _world(data, bid)) for bid, body, A, b, tree in hands[side]]
              for side in SIDES}
        for tag, m in models.items():
            # 권한 구 위치: URDF 손가락 parameter 를 MJCF 와 무관하게 범위 안 무작위 (FK 만)
            params = {}
            for name in urdf_fingers:
                j = m.model.joint_by_name(name)
                params[name] = float(j.lower + rng.uniform() * (j.upper - j.lower))
            m.set_joint_parameters(params)
            C, R = m.sphere_centers_numeric(q)
            C = np.asarray(C) @ T_wb[:3, :3].T + T_wb[:3, 3]
            R = np.asarray(R)
            names = np.asarray(m.sphere_link_names)
            for side in SIDES:
                auth_links = ("ee_" + side, f"ee_finger_{side[0]}1", f"ee_finger_{side[0]}2")
                amask = np.isin(names, auth_links)
                for link in WRIST_LINKS_BY_SIDE[side]:
                    idx = np.flatnonzero(names == link)
                    if not idx.size:
                        continue
                    rec = res[tag][link]
                    c, r = C[idx], R[idx]
                    lb = np.full(idx.size, np.inf)
                    samp = np.full(idx.size, np.inf)
                    for body, A, b, tree, T in hw[side]:
                        cb = (c - T[:3, 3]) @ T[:3, :3]
                        lb_p = np.max(cb @ A.T + b, axis=1)
                        d_s = tree.query(cb)[0]
                        inside = lb_p < 0.0
                        d_s = np.where(inside, -d_s, d_s)
                        pen = r - d_s
                        if np.any(pen > 0):
                            rec["pen_by_mjcf_body"][body] = max(
                                rec["pen_by_mjcf_body"].get(body, 0.0), float(pen.max() * 1e3))
                        lb = np.minimum(lb, lb_p)
                        samp = np.minimum(samp, d_s)
                    pen_lb = r - lb
                    pen_s = r - samp
                    rec["mjcf_pen_lb_max_mm"] = max(rec["mjcf_pen_lb_max_mm"], float(pen_lb.max() * 1e3))
                    rec["mjcf_pen_sampled_max_mm"] = max(rec["mjcf_pen_sampled_max_mm"],
                                                         float(pen_s.max() * 1e3))
                    rec["n_sphere_pose_pen_lb"] += int((pen_lb > 1e-6).sum())
                    rec["n_sphere_pose_pen_sampled"] += int((pen_s > 0).sum())
                    if amask.any():
                        ca, ra = C[amask], R[amask]
                        ov = (r[:, None] + ra[None]) - np.linalg.norm(c[:, None] - ca[None], axis=2)
                        rec["auth_overlap_max_mm"] = max(rec["auth_overlap_max_mm"], float(ov.max() * 1e3))
                        rec["n_pair_pose_overlap"] += int((ov > 0).sum())
                    if tag == "after" and link in own:
                        # frame 검사: URDF FK 로 놓은 구 중심을 MJCF body frame 으로 → 그 body 의 껍질 깊이 =
                        # link frame 깊이 (`T_link_body` 는 손목 셋 다 단위 행렬 — `_mjcf_body_in_link`)
                        bid, hulls = own[link]
                        T = _world(data, bid)
                        cb = (c - T[:3, 3]) @ T[:3, :3]
                        T_lb = gc._mjcf_body_in_link(mj, m, link, gc.WRIST_MJCF_BODY.get(link, link))
                        d_world = gc.hull_depth(cb @ T_lb[:3, :3].T + T_lb[:3, 3], _link_hulls(mj, m, link))
                        frame_diff = max(frame_diff,
                                         float(np.abs(d_world - local_depth[link]).max() * 1e3))
    for tag in models:
        models[tag].set_joint_parameters(None)
        for link, rec in res[tag].items():
            for key in ("mjcf_pen_lb_max_mm", "mjcf_pen_sampled_max_mm", "auth_overlap_max_mm"):
                if rec[key] == -np.inf:
                    rec[key] = None
    protrusion = {tag: {side: auth_protrusion(mj, m, side) for side in SIDES}
                  for tag, m in models.items()}
    return {"n_poses": int(n_poses), "seed": int(seed), "sec": round(time.time() - t0, 1),
            "frame_depth_diff_max_mm": frame_diff, "by_model": res,
            "auth_protrusion_mm": protrusion}


def auth_protrusion(mj, m, side: str) -> dict:
    """권한 구 (손바닥 · 손가락) 가 자기 MJCF collision 껍질 밖으로 나온 최대 거리 (mm), link 마다 — 구 표면 표본 중
    껍질 밖 점의 겉면까지 거리 (`gripper_cover._sampled_overflow`, 실측 넘침과 같은 정의). 비권한 구와의 겹침이 이
    값보다 크지 않으면 그 겹침은 **권한 구가 손 밖으로 나온 몫**이지 비권한 구가 손 안에 들어간 것이 아니다."""
    from AG3S.ag3s.experiments.sources import gripper_cover as gc
    from AG3S.ag3s.experiments.sources.mujoco_source import MJCF_BODY_ALIASES

    out = {}
    for link in gc.WRIST_HAND_LINKS[side]:
        body = MJCF_BODY_ALIASES.get(link, link)
        surf = gc.body_collision_surface(mj, body, gc._mjcf_body_in_link(mj, m, link, body), 0.001)
        c, r = m.local_spheres(link)
        if r.size:
            out[link] = float(gc._sampled_overflow(c, r, surf[3], surf[0], n=2000) * 1e3)
    return out


_LINK_HULLS: dict = {}


def _link_hulls(mj, m, link):
    from AG3S.ag3s.experiments.sources import gripper_cover as gc

    if link not in _LINK_HULLS:
        _LINK_HULLS[link] = gc.wrist_surface(mj, m, link).hulls
    return _LINK_HULLS[link]


def main(argv=None) -> dict:
    import mujoco

    from AG3S.ag3s.experiments.reports.grounding_report import (
        WRIST_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    from AG3S.trajopt.serve_safe import constraint_link_filter
    from common.workspace import workspace_path

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-xml", default=str(workspace_path(
        "robots/rby1/description/models/rby1a/mujoco/model_transport.xml")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--overflow-mm", type=float, default=None,
                    help="wrist cover overflow (mm). 기본 = gripper_cover.WRIST_COVER_OVERFLOW")
    ap.add_argument("--spacing-mm", type=float, default=0.5)
    ap.add_argument("--n-poses", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--palm-cover", action="store_true",
                    help="권한 손바닥 구를 palm cover (T45 A10) 로 — 겹침 (b) 용 추가 모델")
    args = ap.parse_args(argv)

    mj = mujoco.MjModel.from_xml_path(str(pathlib.Path(args.model_xml).resolve()))
    scene = TransportScene.attach(mj, mujoco.MjData(mj))
    wc = {"hands": "both"}
    if args.overflow_mm is not None:
        wc["overflow"] = float(args.overflow_mm) * 1e-3
    t0 = time.time()
    models = {
        "before_arms": build_constraint_robot_model(
            scene, link_filter=constraint_link_filter("arms"), sphere_options=SERVER,
            finger_cover={"hands": "both"}),
        "after": build_constraint_robot_model(
            scene, link_filter=constraint_link_filter("gripper_wrist"), sphere_options=SERVER,
            finger_cover={"hands": "both"}, wrist_cover=wc),
    }
    if args.palm_cover:
        models["after_palm_cover"] = build_constraint_robot_model(
            scene, link_filter=constraint_link_filter("gripper_wrist"), sphere_options=SERVER,
            finger_cover={"hands": "both"}, palm_cover=True, wrist_cover=wc)
    build_sec = time.time() - t0
    out = {
        "step": "T46f-A14", "thresholds_fixed_before_measuring": THRESHOLDS,
        "server_sphere_options": SERVER, "wrist_cover": wc,
        "n_spheres": {k: int(m.n_spheres) for k, m in models.items()},
        "per_link_n_spheres": {k: {n: int(sum(1 for x in m.sphere_link_names if x == n))
                                   for n in dict.fromkeys(m.sphere_link_names)}
                               for k, m in models.items()},
        "wrist_cover_report": models["after"].wrist_cover_report,
        "build_sec": round(build_sec, 1),
    }
    out["coverage"] = coverage(mj, {k: models[k] for k in ("before_arms", "after")},
                               WRIST_LINKS, args.spacing_mm * 1e-3)
    out["hand_overlap"] = hand_overlap(scene, models, WRIST_LINKS, args.n_poses, args.seed)
    out["verdict"] = verdict(out)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(out["verdict"], indent=1, ensure_ascii=False))
    return out


def verdict(out: dict) -> dict:
    """사전 고정 기준 (`THRESHOLDS`) 으로 link 마다 통과 여부."""
    th = THRESHOLDS
    ov_mm = 1e3 * float(out["wrist_cover_report"]["options"]["overflow"])
    v = {}
    for link, rec in out["coverage"].items():
        kind = _kind(link)
        a = rec["after"]
        ho = out["hand_overlap"]["by_model"]["after"][link]
        checks = {
            "reachable_max_outside": a["reachable"]["outside_max_mm"] <= th["reachable_max_outside_mm"][kind],
            "reachable_frac_outside_gt_0p5mm":
                a["reachable"]["frac_outside_gt_0p5mm"] <= th["reachable_frac_outside_gt_0p5mm"][kind],
            "overflow": a["overflow"]["max_mm"] <= ov_mm + th["overflow_max_excess_mm"],
            "mjcf_hand_penetration_sampled": (ho["mjcf_pen_sampled_max_mm"] or -1.0)
                <= th["mjcf_hand_penetration_sampled_mm"],
            "mjcf_hand_penetration_lb": (ho["mjcf_pen_lb_max_mm"] or -1.0)
                <= th["mjcf_hand_penetration_lb_mm"],
        }
        side = "left" if (link.endswith("_L") or "_left_" in link) else "right"
        for tag in [t for t in out["hand_overlap"]["by_model"] if t.startswith("after")]:
            ov = out["hand_overlap"]["by_model"][tag][link]["auth_overlap_max_mm"]
            prot = max(out["hand_overlap"]["auth_protrusion_mm"][tag][side].values())
            checks[f"auth_overlap_within_auth_protrusion[{tag}]"] = (
                ov is None or ov <= max(prot, 0.0) + th["auth_overlap_excess_over_auth_protrusion_mm"])
        v[link] = {"pass": bool(all(checks.values())), **checks}
    v["frame_depth_diff"] = out["hand_overlap"]["frame_depth_diff_max_mm"] <= th["frame_depth_diff_mm"]
    v["all_pass"] = bool(all(x["pass"] for k, x in v.items() if isinstance(x, dict))
                         and v["frame_depth_diff"])
    return v


if __name__ == "__main__":
    main()
