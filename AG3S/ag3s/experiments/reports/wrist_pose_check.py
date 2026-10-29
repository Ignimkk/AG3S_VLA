"""T46f A14b — 실험 자세 점검 (사용자 판정 A7): 비허용 손목 구가 **실제 실험 자세에서** 활성 손의 접촉 영역을
침범해 가짜 제약을 만드는가.

입력: T46 CL3 closed-loop 기록 (C1a · C1aF1 · C1aF1F2 × 18 run) 의 `frames.jsonl` **planning 행** (`kind == "planning"`)
— 서버가 그 chunk 를 계획할 때의 MuJoCo 전체 `qpos` (팔 · 손가락 · 과일 free joint). 활성 손 = CL3 analysis 의
`used_arm`, target = `target_fruit`. 입력은 읽기만 한다.

자세마다 world frame (MuJoCo FK 로 MJCF geom, URDF FK 로 구 — `wrist_cover_audit.hand_overlap` 과 같은 배치):

* (a) 활성 쪽 비허용 구 (`link_<s>_arm_5` · `link_<s>_arm_6` · `FT_sensor_<S>`) 의 활성 손 MJCF collision geom 침범
  `r − dist(c, piece)` max. 손 = 손바닥 `EE_BODY_<S>` + 손가락 둘 (MJCF body 이름 — 두 손가락을 다 쓰므로 왼손 이름
  뒤바뀜과 무관). 두 가지: **sweep** (손가락 prismatic 이동 전 구간의 볼록 껍질, `gripper_cover.hand_pieces`) 과
  **current** (그 자세의 손가락 값). dist 는 볼록 조각까지 **정확한** Euclid 거리 (밖: 껍질 삼각형까지 최소 거리,
  안: −가장 가까운 면 평면 거리). sweep ⊇ current 이므로 sweep 값 ≥ current 값.
* (b) 같은 비허용 구와 활성 손 권한 구 (`ee_<side>` · `ee_finger_<s>1/2`, 그 자세의 손가락 parameter) 의 겹침
  `r1 + r2 − |c1 − c2|` max. 비교값 = 그 모델의 권한 구 돌출 (`wrist_cover_audit.auth_protrusion`, 같은 쪽 권한
  link 의 max).
* (c) target 표면 (MJCF collision mesh 겉면, 표본 1 mm) 중 활성 손 MJCF geom (current) 에서 5 mm 안인 점 (= 손 접촉
  영역 안 target) 이 비허용 구 안 (`|p − c| < r`) 에 드는 점 수 = 가짜 제약 후보. 주 집합 = 활성 쪽 손목 구,
  보조 = 양쪽 손목 구 전부.
* (d) 참고 (판정 없음): 활성 쪽 손목 구 ↔ 반대 손 MJCF geom (current), 반대쪽 손목 구 ↔ 활성 손 MJCF geom 의 최소
  표면 거리 `dist − r`.
* frame 검사: URDF FK 손바닥 · 손가락 link 원점 대 MuJoCo body 원점 (`T_link_body` 반영, 손가락은 위치로 짝짓기).

URDF 손가락 parameter ← MuJoCo qpos: **위치로 맞춘 대응** (`finger_param_map`). 왼손 손가락 이름이 MJCF 와 URDF 에서
서로 바뀌어 있다 (T43 OC: MJCF `ee_finger_l2` = URDF `ee_finger_l1`). 같은 이름의 MuJoCo joint 값을 쓰면 URDF 손가락이
반대편 MJCF 손가락 joint 값으로 놓이고, 두 손가락 값이 다를 때 그 차만큼 (CL3 에서 최대 0.209 mm) frame 검사가
어긋난다 (1 회차 `pose_check_run1.json`, `frame_diff` fail — 이 대응을 고친 것이 2 회차).

판정 기준 (`POSE_THRESHOLDS`) 은 **측정 전에 코드에 고정했다** (T46f A14b 3).

    MUJOCO_GL=osmesa PYTHONPATH=<checkout> python -m AG3S.ag3s.experiments.reports.wrist_pose_check \
        --out $AG3S_RUN_DIR/impl/T46f-A14b/pose_check.json --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

from AG3S.ag3s.experiments.reports.wrist_cover_audit import SERVER, THRESHOLDS

#: T46 CL3 (snapshot 218229ac) — `verify/T46/CL3` 아래 `analysis/<NAME>.json` · `runs/<ARM>/R/<NAME>/frames.jsonl`.
CL3_DIR = ("results/development/20261012T155130+0900_218229ac6464_5fb5ff542c33/verify/T46/CL3")
CL3_ARMS = ("C1a", "C1aF1", "C1aF1F2")

#: 판정 기준 — **측정 전에 고정** (T46f 사용자 판정 A14b 3 · A8). 모델 `after` (= 기본 `gripper_wrist`) 에만 건다.
POSE_THRESHOLDS = {
    # (a) 비허용 구의 활성 손 MJCF geom 침범 — 수치 허용오차 (A8: URDF/MuJoCo FK 부동소수). sweep 에 건다.
    "a_mjcf_hand_penetration_mm": THRESHOLDS["mjcf_hand_penetration_sampled_mm"],     # 0.001
    # (b) 권한 구 겹침 ≤ 그 쪽 권한 구 돌출 (audit 와 같은 여유 0.01 mm)
    "b_auth_overlap_excess_over_auth_protrusion_mm":
        THRESHOLDS["auth_overlap_excess_over_auth_protrusion_mm"],                       # 0.01
    # (c) 손 접촉 영역 안 target 표면 점이 활성 쪽 비허용 구 안에 드는 점 수
    "c_false_constraint_points": 0,
    # frame 검사 (audit 와 같은 값)
    "frame_diff_mm": THRESHOLDS["frame_depth_diff_mm"],                                  # 0.01
}
#: (c) 손 접촉 영역 = 활성 손 MJCF geom (current) 까지 이 거리 (m) 안.
HAND_REGION = 0.005
#: target 겉면 표본 간격 (m).
TARGET_SPACING = 0.001
#: 정확한 거리를 계산하는 범위 (m) — 지지 평면 하한 `lb` 가 이보다 멀면 `lb` 만 쓴다 (`lb ≤ 정확한 거리`).
EXACT_WITHIN = 0.02
FRUITS = ("apple", "banana", "orange", "pear")
SIDES = ("left", "right")


# =============================================================================================
# 볼록 조각까지 정확한 거리
# =============================================================================================


def point_triangle_distance(P: np.ndarray, tri: np.ndarray, block: int = 512) -> np.ndarray:
    """`min_t |p − closest(p, tri_t)|` (m) — 점 `P[n,3]` 에서 삼각형 `tri[T,3,3]` 까지 정확한 최소 거리 (Ericson,
    Real-Time Collision Detection 5.1.5 의 영역 판정)."""
    P = np.asarray(P, np.float64).reshape(-1, 3)
    out = np.empty(P.shape[0])
    a, b, c = tri[None, :, 0], tri[None, :, 1], tri[None, :, 2]
    ab, ac, bc = b - a, c - a, c - b
    dot = lambda x, y: np.einsum("ntk,ntk->nt", x, y)
    for s in range(0, P.shape[0], block):
        p = P[s:s + block][:, None, :]
        ap, bp, cp = p - a, p - b, p - c
        n_ = p.shape[0]
        AB, AC = np.broadcast_to(ab, (n_,) + ab.shape[1:]), np.broadcast_to(ac, (n_,) + ac.shape[1:])
        d1, d2 = dot(AB, ap), dot(AC, ap)
        d3, d4 = dot(AB, bp), dot(AC, bp)
        d5, d6 = dot(AB, cp), dot(AC, cp)
        vc = d1 * d4 - d3 * d2
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4
        with np.errstate(divide="ignore", invalid="ignore"):
            v_ab = np.clip(d1 / (d1 - d3), 0.0, 1.0)
            w_ac = np.clip(d2 / (d2 - d6), 0.0, 1.0)
            w_bc = np.clip((d4 - d3) / ((d4 - d3) + (d5 - d6)), 0.0, 1.0)
            den = va + vb + vc
            v_in = np.where(den != 0, vb / den, 0.0)
            w_in = np.where(den != 0, vc / den, 0.0)
        A_, B_, C_ = (np.broadcast_to(x, (n_,) + x.shape[1:]) for x in (a, b, c))
        BC = np.broadcast_to(bc, (n_,) + bc.shape[1:])
        conds = [(d1 <= 0) & (d2 <= 0),
                 (d3 >= 0) & (d4 <= d3),
                 (vc <= 0) & (d1 >= 0) & (d3 <= 0),
                 (d6 >= 0) & (d5 <= d6),
                 (vb <= 0) & (d2 >= 0) & (d6 <= 0),
                 (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)]
        cands = [A_, B_, A_ + v_ab[..., None] * AB, C_, A_ + w_ac[..., None] * AC,
                 B_ + w_bc[..., None] * BC]
        q = A_ + v_in[..., None] * AB + w_in[..., None] * AC
        done = np.zeros(d1.shape, bool)
        for cond, cand in zip(conds, cands):
            m = cond & ~done
            q = np.where(m[..., None], cand, q)
            done |= m
        out[s:s + block] = np.sqrt(((p - q) ** 2).sum(-1)).min(1)
    return out


class ConvexPiece:
    """볼록 조각 하나 (자기 frame): 면 평면 `A p + b ≤ 0` (단위 법선) · 껍질 삼각형 · 경계 구."""

    def __init__(self, verts: np.ndarray):
        from scipy.spatial import ConvexHull

        verts = np.asarray(verts, np.float64)
        h = ConvexHull(verts)
        self.A, self.b = h.equations[:, :3].copy(), h.equations[:, 3].copy()
        self.tri = verts[h.simplices]
        self.centre = verts[h.vertices].mean(0)
        self.radius = float(np.linalg.norm(verts[h.vertices] - self.centre, axis=1).max())

    def signed_distance(self, P: np.ndarray, exact_within: float = EXACT_WITHIN
                        ) -> tuple[np.ndarray, np.ndarray]:
        """`(d, exact)` — 안: `max(A p + b)` (≤ 0, 정확), 밖: 그 값이 `exact_within` 이하면 삼각형까지 정확한 거리,
        더 멀면 `max(A p + b)` (하한) 그대로 (`exact = False`)."""
        P = np.asarray(P, np.float64).reshape(-1, 3)
        lb = np.max(P @ self.A.T + self.b, axis=1)
        d = lb.copy()
        exact = lb <= 0.0
        m = (lb > 0.0) & (lb <= exact_within)
        if m.any():
            d[m] = point_triangle_distance(P[m], self.tri)
            exact |= m
        return d, exact


def pieces_signed_distance(P: np.ndarray, placed, exact_within: float = EXACT_WITHIN):
    """`placed = [(ConvexPiece, T_world_piece)]` → 점마다 조각 합집합까지 `(d, exact)` (min over pieces)."""
    P = np.asarray(P, np.float64).reshape(-1, 3)
    d = np.full(P.shape[0], np.inf)
    ex = np.zeros(P.shape[0], bool)
    for piece, T in placed:
        Pl = (P - T[:3, 3]) @ T[:3, :3]
        far = np.linalg.norm(Pl - piece.centre, axis=1) - piece.radius
        sel = far <= max(exact_within, 0.0) + 1e-9
        dd = far.copy()          # 경계 구 밖 → 경계 구까지 거리 (≤ 정확한 거리, 하한)
        e = np.zeros(P.shape[0], bool)
        if sel.any():
            dd[sel], e[sel] = piece.signed_distance(Pl[sel], exact_within)
        better = dd < d
        d = np.where(better, dd, d)
        ex = np.where(better, e, ex)
    return d, ex


# =============================================================================================
# 모델 · 장면
# =============================================================================================


def _world(data, bid):
    T = np.eye(4)
    T[:3, :3] = data.xmat[bid].reshape(3, 3)
    T[:3, 3] = data.xpos[bid]
    return T


def hand_bodies(side: str) -> tuple[str, str, str]:
    """활성 손의 MJCF body 이름 (손바닥 · 손가락 둘)."""
    s, f = ("L", "l") if side == "left" else ("R", "r")
    return (f"EE_BODY_{s}", f"ee_finger_{f}1", f"ee_finger_{f}2")


def wrist_links(side: str) -> tuple[str, ...]:
    from AG3S.ag3s.experiments.reports.grounding_report import WRIST_LINKS_BY_SIDE

    return WRIST_LINKS_BY_SIDE[side]


def auth_links(side: str) -> tuple[str, ...]:
    from AG3S.ag3s.experiments.sources.gripper_cover import WRIST_HAND_LINKS

    return WRIST_HAND_LINKS[side]


class Context:
    """프로세스 공유 (fork 전에 만든다): MuJoCo 모델 · 손 조각 · target 겉면 · 구 모델 · 권한 구 돌출."""

    def __init__(self, scene, models: dict):
        import mujoco

        from AG3S.ag3s.experiments.reports.wrist_cover_audit import auth_protrusion
        from AG3S.ag3s.experiments.sources import gripper_cover as gc
        from AG3S.ag3s.experiments.sources.mujoco_source import MJCF_BODY_ALIASES

        self.mj = scene.model
        self.scene = scene
        self.models = models
        mj = self.mj
        bid = lambda n: mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, n)
        self.bid = {n: bid(n) for side in SIDES for n in hand_bodies(side)}
        self.bid["base"] = bid("base")
        # current: MJCF body frame 의 조각
        self.hand_current = {side: [(n, ConvexPiece(v)) for n in hand_bodies(side)
                                    for v in gc._geom_pieces(mj, self.bid[n])] for side in SIDES}
        # sweep: URDF 손바닥 link frame 의 조각 (손가락은 prismatic 두 한계 정점의 볼록 껍질)
        ref = models["after"]
        self.hand_sweep = {side: [ConvexPiece(v) for v in gc.hand_pieces(mj, ref, side)] for side in SIDES}
        palm_body = {side: hand_bodies(side)[0] for side in SIDES}
        self.T_palmlink_body = {side: gc._mjcf_body_in_link(mj, ref, auth_links(side)[0], palm_body[side])
                                for side in SIDES}
        # 손가락 frame 검사용: URDF 손가락 link → MJCF body 변환 (위치로 짝짓기 — 이름에 기대지 않는다)
        self.T_link_body = {}
        for side in SIDES:
            for link in auth_links(side)[1:]:
                body = MJCF_BODY_ALIASES.get(link, link)
                self.T_link_body[link] = gc._mjcf_body_in_link(mj, ref, link, body)
        # target 겉면 (body frame, 겉면만)
        self.fruit_bid = {f: bid(f) for f in FRUITS}
        self.fruit_surface = {}
        for f in FRUITS:
            pts, _, _, _, _ = gc.body_collision_surface(mj, f, np.eye(4), TARGET_SPACING)
            self.fruit_surface[f] = pts
        self.protrusion = {tag: {side: auth_protrusion(mj, m, side) for side in SIDES}
                           for tag, m in models.items()}
        self.qadr = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i): int(mj.jnt_qposadr[i])
                     for i in range(mj.njnt)}
        self.finger_map = {id(m): finger_param_map(mj, m, self.qadr) for m in models.values()}
        self.finger_map_record = {tag: {k: [v[0], v[1]] for k, v in self.finger_map[id(m)].items()}
                                  for tag, m in models.items()}


def finger_param_map(mj, m, qadr: dict) -> dict:
    """`{URDF 손가락 parameter: (MuJoCo joint, 부호)}` — **위치로** 맞춘다 (이름에 기대지 않는다, T43 OC).

    손가락마다 서로 다른 시험 값을 MuJoCo 에 넣고, URDF 손가락 link (`T_link_body` 반영) 가 그 손 MJCF 손가락 body 중
    하나에 1 µm 안으로 놓이는 (joint, 부호) 를 고른다. 시험 값 두 벌 모두에서 맞아야 한다."""
    import mujoco

    from AG3S.ag3s.experiments.sources import gripper_cover as gc
    from AG3S.ag3s.experiments.sources.mujoco_source import MJCF_BODY_ALIASES

    names = tuple(getattr(m, "param_joint_names", ()) or ())
    if not names:
        return {}
    data = mujoco.MjData(mj)
    mj_f = [n for n in qadr if n and n.startswith("gripper_finger_")]
    jbody = {n: int(mj.jnt_bodyid[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in mj_f}
    rng = np.random.default_rng(7)
    trials = []
    for _ in range(2):
        vals = {}
        for n in mj_f:
            lo_, hi_ = mj.jnt_range[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, n)]
            vals[n] = float(lo_ + rng.uniform(0.2, 0.8) * (hi_ - lo_))
        trials.append(vals)
    lo, hi = m.joint_limits()
    q0 = 0.5 * (np.asarray(lo) + np.asarray(hi))
    out = {}
    for i, pname in enumerate(names):
        link = m.model.joint_by_name(pname).child
        body = MJCF_BODY_ALIASES.get(link, link)
        T_lb = gc._mjcf_body_in_link(mj, m, link, body)
        found = []
        for jn in mj_f:
            for sign in (1.0, -1.0):
                ok = True
                for vals in trials:
                    mujoco.mj_resetData(mj, data)
                    for k, n in enumerate(m.joint_names):
                        if n in qadr:
                            data.qpos[qadr[n]] = q0[k]
                    for n, v in vals.items():
                        data.qpos[qadr[n]] = v
                    mujoco.mj_kinematics(mj, data)
                    params = m.param_vector().copy()
                    params[i] = sign * vals[jn]
                    T_wb = _world(data, mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, "base"))
                    Tl = T_wb @ m.link_pose(q0, link, params) @ T_lb
                    if np.linalg.norm(Tl[:3, 3] - data.xpos[jbody[jn]]) > 1e-6:
                        ok = False
                        break
                if ok:
                    found.append((jn, sign))
        if len(found) != 1:
            raise ValueError(f"URDF 손가락 parameter {pname!r} ↔ MuJoCo joint 를 위치로 정할 수 없다: {found}")
        out[pname] = found[0]
    return out


def _q_params(ctx: Context, m, qpos: np.ndarray):
    q = np.asarray([qpos[ctx.qadr[n]] for n in m.joint_names], np.float64)
    names = tuple(getattr(m, "param_joint_names", ()) or ())
    if not names:
        return q, None
    fmap = ctx.finger_map[id(m)]
    params = np.asarray([fmap[n][1] * qpos[ctx.qadr[fmap[n][0]]] for n in names], np.float64)
    return q, params


def pose_metrics(ctx: Context, data, qpos: np.ndarray, side: str, target: str) -> dict:
    """한 planning 행 (MuJoCo 전체 qpos) 의 (a)–(d) · frame 검사, 모델마다."""
    import mujoco

    mj = ctx.mj
    data.qpos[:] = qpos
    mujoco.mj_kinematics(mj, data)
    other = "right" if side == "left" else "left"
    T_wb = _world(data, ctx.bid["base"])
    placed_cur = {s: [(p, _world(data, ctx.bid[n])) for n, p in ctx.hand_current[s]] for s in SIDES}
    T_palm_w = {s: _world(data, ctx.bid[hand_bodies(s)[0]]) @ np.linalg.inv(ctx.T_palmlink_body[s])
                for s in SIDES}
    placed_sweep = [(p, T_palm_w[side]) for p in ctx.hand_sweep[side]]
    # target 겉면 (world) 중 활성 손 (current) 5 mm 안
    fb = ctx.fruit_bid[target]
    Tf = _world(data, fb)
    P = ctx.fruit_surface[target] @ Tf[:3, :3].T + Tf[:3, 3]
    hand_c = np.mean([T[:3, 3] for _, T in placed_cur[side]], axis=0)
    hand_r = max(np.linalg.norm(T[:3, :3] @ p.centre + T[:3, 3] - hand_c) + p.radius
                 for p, T in placed_cur[side])
    near = np.linalg.norm(P - hand_c, axis=1) <= hand_r + HAND_REGION
    region = np.zeros(P.shape[0], bool)
    d_t_min = None
    if near.any():
        dP, _ = pieces_signed_distance(P[near], placed_cur[side], exact_within=HAND_REGION + 0.001)
        region[np.flatnonzero(near)] = dP <= HAND_REGION
        d_t_min = float(dP.min())
    R_region = P[region]
    out = {"n_target_points": int(P.shape[0]), "n_target_in_hand_region": int(region.sum()),
           "target_hand_min_mm": None if d_t_min is None else d_t_min * 1e3, "models": {}}
    for tag, m in ctx.models.items():
        q, params = _q_params(ctx, m, qpos)
        C, R = m.sphere_centers_numeric(q, params) if params is not None else m.sphere_centers_numeric(q)
        C = np.asarray(C, np.float64) @ T_wb[:3, :3].T + T_wb[:3, 3]
        R = np.asarray(R, np.float64)
        names = np.asarray(m.sphere_link_names)
        rec: dict = {}
        na = np.isin(names, wrist_links(side))
        na_o = np.isin(names, wrist_links(other))
        au = np.isin(names, auth_links(side))
        # frame 검사 (모델마다 같은 URDF FK): 손바닥 link 원점 · 손가락 link 원점
        fd = 0.0
        Tp = T_wb @ m.link_pose(q, auth_links(side)[0], params)
        fd = max(fd, float(np.linalg.norm(Tp[:3, 3] - T_palm_w[side][:3, 3])))
        mj_f = [_world(data, ctx.bid[n])[:3, 3] for n in hand_bodies(side)[1:]]
        for link in auth_links(side)[1:]:
            Tl = T_wb @ m.link_pose(q, link, params) @ ctx.T_link_body[link]
            fd = max(fd, float(min(np.linalg.norm(Tl[:3, 3] - x) for x in mj_f)))
        rec["frame_diff_mm"] = fd * 1e3
        if na.any():
            c, r = C[na], R[na]
            # 정확한 거리 범위 = 가장 큰 구 반지름 + 5 mm (그보다 먼 하한으로는 침범이 생기지 않는다)
            reach = float(r.max()) + 0.005
            for key, placed in (("a_sweep", placed_sweep), ("a_current", placed_cur[side])):
                d, ex = pieces_signed_distance(c, placed, exact_within=reach)
                pen = r - d
                k = int(np.argmax(pen))
                rec[f"{key}_pen_max_mm"] = float(pen[k] * 1e3)
                rec[f"{key}_exact"] = bool(ex[k])
                rec[f"{key}_argmax_link"] = str(names[na][k])
                rec[f"{key}_n_pen_gt_tol"] = int((pen > POSE_THRESHOLDS["a_mjcf_hand_penetration_mm"] * 1e-3)
                                                  .sum())
            if au.any():
                ov = (r[:, None] + R[au][None]) - np.linalg.norm(c[:, None] - C[au][None], axis=2)
                i, j = np.unravel_index(int(np.argmax(ov)), ov.shape)
                rec["b_auth_overlap_max_mm"] = float(ov[i, j] * 1e3)
                rec["b_pair"] = [str(names[na][i]), str(names[au][j])]
            # (c) 손 접촉 영역 안 target 점이 비허용 구 안에
            if R_region.shape[0]:
                g = np.linalg.norm(R_region[:, None] - c[None], axis=2) - r[None]
                gmin = g.min(1)
                rec["c_n_points_inside_active_wrist"] = int((gmin < 0.0).sum())
                rec["c_min_gap_active_wrist_mm"] = float(gmin.min() * 1e3)
                allw = na | na_o
                g2 = (np.linalg.norm(R_region[:, None] - C[allw][None], axis=2) - R[allw][None]).min(1)
                rec["c_n_points_inside_any_wrist"] = int((g2 < 0.0).sum())
            else:
                rec["c_n_points_inside_active_wrist"] = 0
                rec["c_n_points_inside_any_wrist"] = 0
                rec["c_min_gap_active_wrist_mm"] = None
            # (d) 참고: 활성 손목 구 ↔ 반대 손 MJCF geom
            d, ex = pieces_signed_distance(c, placed_cur[other], exact_within=0.05)
            k = int(np.argmin(d - r))
            rec["d_active_wrist_to_other_hand_mm"] = float((d - r)[k] * 1e3)
            rec["d_active_wrist_to_other_hand_exact"] = bool(ex[k])
        if na_o.any():
            d, ex = pieces_signed_distance(C[na_o], placed_cur[side], exact_within=0.05)
            k = int(np.argmin(d - R[na_o]))
            rec["d_other_wrist_to_active_hand_mm"] = float((d - R[na_o])[k] * 1e3)
            rec["d_other_wrist_to_active_hand_exact"] = bool(ex[k])
        out["models"][tag] = rec
    return out


# =============================================================================================
# 실행
# =============================================================================================

_CTX: Context | None = None


def run_one(job: tuple) -> dict:
    """`(name, run_dir, side, target, close_seq)` → run 기록 (행마다 · 요약)."""
    import mujoco

    name, run_dir, side, target, close_seq, max_rows = job
    ctx = _CTX
    data = mujoco.MjData(ctx.mj)
    rows = [json.loads(l) for l in open(pathlib.Path(run_dir) / "frames.jsonl")]
    plan = [r for r in rows if r.get("kind") == "planning" and r.get("qpos") is not None]
    if max_rows is not None:
        plan = plan[:int(max_rows)]
    per_row = []
    pose_check = 0.0
    for r in plan:
        qpos = np.asarray(r["qpos"], np.float64)
        if qpos.shape[0] != ctx.mj.nq:
            raise ValueError(f"{name} seq {r['seq']}: qpos {qpos.shape[0]} != nq {ctx.mj.nq}")
        res = pose_metrics(ctx, data, qpos, side, target)
        op = (r.get("object_poses") or {}).get(target)
        if op is not None:
            pose_check = max(pose_check, float(np.linalg.norm(
                np.asarray(op["pos"]) - data.xpos[ctx.fruit_bid[target]]) * 1e3))
        res["seq"] = int(r["seq"])
        res["phase"] = ("unknown" if close_seq is None else
                        ("pre_close" if int(r["seq"]) <= int(close_seq) else "post_close"))
        per_row.append(res)
    return {"name": name, "side": side, "target": target, "close_seq": close_seq,
            "n_planning_rows": len(per_row), "target_pose_check_mm": pose_check, "rows": per_row}


def _at_max(rows: list, tag: str, key: str, flag: str):
    best = None
    for r in rows:
        v = r["models"][tag].get(key)
        if v is not None and (best is None or v > best[0]):
            best = (v, r["models"][tag].get(flag))
    return None if best is None else bool(best[1])


def _agg(rows: list, tag: str) -> dict:
    """행 기록 → 모델 `tag` 의 요약 (max / min / 합)."""
    def vals(k):
        return [r["models"][tag][k] for r in rows if r["models"][tag].get(k) is not None]

    def mx(k):
        v = vals(k)
        return max(v) if v else None

    def mn(k):
        v = vals(k)
        return min(v) if v else None

    return {"n_rows": len(rows),
            "a_sweep_pen_max_mm": mx("a_sweep_pen_max_mm"),
            "a_current_pen_max_mm": mx("a_current_pen_max_mm"),
            "a_sweep_n_sphere_rows_pen_gt_tol": int(sum(vals("a_sweep_n_pen_gt_tol"))),
            "a_sweep_max_is_exact": _at_max(rows, tag, "a_sweep_pen_max_mm", "a_sweep_exact"),
            "b_auth_overlap_max_mm": mx("b_auth_overlap_max_mm"),
            "c_n_rows_target_in_hand_region": int(sum(r["n_target_in_hand_region"] > 0 for r in rows)),
            "c_n_target_points_in_hand_region": int(sum(r["n_target_in_hand_region"] for r in rows)),
            "c_n_points_inside_active_wrist": int(sum(vals("c_n_points_inside_active_wrist"))),
            "c_n_points_inside_any_wrist": int(sum(vals("c_n_points_inside_any_wrist"))),
            "c_min_gap_active_wrist_mm": mn("c_min_gap_active_wrist_mm"),
            "d_active_wrist_to_other_hand_min_mm": mn("d_active_wrist_to_other_hand_mm"),
            "d_other_wrist_to_active_hand_min_mm": mn("d_other_wrist_to_active_hand_mm"),
            "frame_diff_max_mm": mx("frame_diff_mm")}


def verdict(out: dict) -> dict:
    th = POSE_THRESHOLDS
    a = out["summary"]["all"]["after"]
    prot = max(max(v.values()) for v in out["auth_protrusion_mm"]["after"].values())
    checks = {
        "a_mjcf_hand_penetration_sweep": a["a_sweep_pen_max_mm"] <= th["a_mjcf_hand_penetration_mm"],
        "b_auth_overlap_within_auth_protrusion": (
            a["b_auth_overlap_max_mm"] is None
            or a["b_auth_overlap_max_mm"] <= max(prot, 0.0) + th["b_auth_overlap_excess_over_auth_protrusion_mm"]),
        "c_false_constraint_points": a["c_n_points_inside_active_wrist"] <= th["c_false_constraint_points"],
        "frame_diff": a["frame_diff_max_mm"] <= th["frame_diff_mm"],
    }
    # 쪽마다 권한 구 돌출로도 (b) — 활성 손이 왼손인 run 은 왼손 돌출과 비교
    for side in SIDES:
        s = out["summary"]["by_side"].get(side, {}).get("after")
        if s and s["b_auth_overlap_max_mm"] is not None:
            p = max(out["auth_protrusion_mm"]["after"][side].values())
            checks[f"b_by_side[{side}]"] = (
                s["b_auth_overlap_max_mm"] <= max(p, 0.0) + th["b_auth_overlap_excess_over_auth_protrusion_mm"])
    return {**checks, "all_pass": bool(all(checks.values()))}


def main(argv=None) -> dict:
    import multiprocessing as mp

    import mujoco

    from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    from AG3S.trajopt.serve_safe import constraint_link_filter
    from common.workspace import shared_storage_root, workspace_path

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-xml", default=str(workspace_path(
        "robots/rby1/description/models/rby1a/mujoco/model_transport.xml")))
    ap.add_argument("--cl3", default=str(shared_storage_root() / CL3_DIR))
    ap.add_argument("--arms", nargs="+", default=list(CL3_ARMS))
    ap.add_argument("--names", nargs="*", default=None, help="run 이름만 (smoke)")
    ap.add_argument("--max-rows", type=int, default=None, help="run 마다 planning 행 앞 N 개 (smoke)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-compare", action="store_true",
                    help="비교 모델 (A14 inset 없음 · --links arms) 을 짓지 않는다")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    global _CTX
    mj = mujoco.MjModel.from_xml_path(str(pathlib.Path(args.model_xml).resolve()))
    scene = TransportScene.attach(mj, mujoco.MjData(mj))
    t0 = time.time()
    fc = {"hands": "both"}
    models = {"after": build_constraint_robot_model(
        scene, link_filter=constraint_link_filter("gripper_wrist"), sphere_options=SERVER,
        finger_cover=fc, wrist_cover={"hands": "both"})}
    if not args.no_compare:
        models["a14_no_inset"] = build_constraint_robot_model(
            scene, link_filter=constraint_link_filter("gripper_wrist"), sphere_options=SERVER,
            finger_cover=fc, wrist_cover={"hands": "both", "inset": {}})
        models["before_arms"] = build_constraint_robot_model(
            scene, link_filter=constraint_link_filter("arms"), sphere_options=SERVER, finger_cover=fc)
    build_sec = time.time() - t0
    _CTX = Context(scene, models)
    cl3 = pathlib.Path(args.cl3)
    jobs = []
    for arm in args.arms:
        for f in sorted((cl3 / "analysis").glob(f"{arm}_ep*.json")):
            if args.names and f.stem not in args.names:
                continue
            a = json.load(open(f))
            if a.get("arm") != arm:
                continue
            close = (a.get("pre_grasp") or {}).get("close_seq")
            jobs.append((a["name"], a["run_dir"], a["used_arm"], a["target_fruit"], close, args.max_rows))
    t1 = time.time()
    ctx_mp = mp.get_context("fork")
    with ctx_mp.Pool(max(1, min(int(args.workers), 4))) as pool:
        runs = pool.map(run_one, jobs, chunksize=1)
    tags = list(models)
    summary = {"all": {}, "by_arm": {}, "by_side": {}, "by_phase": {}}
    allrows = [x for r in runs for x in r["rows"]]
    for tag in tags:
        summary["all"][tag] = _agg(allrows, tag)
    for arm in args.arms:
        rr = [x for r in runs if r["name"].split("_ep")[0] == arm for x in r["rows"]]
        if rr:
            summary["by_arm"][arm] = {tag: _agg(rr, tag) for tag in tags}
    for side in SIDES:
        rr = [x for r in runs if r["side"] == side for x in r["rows"]]
        if rr:
            summary["by_side"][side] = {tag: _agg(rr, tag) for tag in tags}
    for ph in ("pre_close", "post_close", "unknown"):
        rr = [x for x in allrows if x["phase"] == ph]
        if rr:
            summary["by_phase"][ph] = {tag: _agg(rr, tag) for tag in tags}
    out = {
        "step": "T46f-A14b pose check", "thresholds_fixed_before_measuring": POSE_THRESHOLDS,
        "hand_region_mm": HAND_REGION * 1e3, "target_spacing_mm": TARGET_SPACING * 1e3,
        "exact_within_mm": EXACT_WITHIN * 1e3,
        "inputs": {"cl3": str(cl3), "arms": list(args.arms), "n_runs": len(runs),
                   "n_planning_rows": len(allrows)},
        "server_sphere_options": SERVER,
        "n_spheres": {k: int(m.n_spheres) for k, m in models.items()},
        "wrist_cover_options": {k: getattr(m, "wrist_cover_report", {}).get("options")
                                for k, m in models.items()},
        "auth_protrusion_mm": _CTX.protrusion,
        "finger_param_map": _CTX.finger_map_record,
        "build_sec": round(build_sec, 1), "measure_sec": round(time.time() - t1, 1),
        "target_pose_check_max_mm": max((r["target_pose_check_mm"] for r in runs), default=None),
        "summary": summary,
        "per_run": {r["name"]: {"side": r["side"], "target": r["target"], "close_seq": r["close_seq"],
                                "n_planning_rows": r["n_planning_rows"],
                                **{tag: _agg(r["rows"], tag) for tag in tags}} for r in runs},
    }
    out["verdict"] = verdict(out)
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    rows_path = path.with_name(path.stem + "_rows.jsonl")
    with open(rows_path, "w") as fh:
        for r in runs:
            for x in r["rows"]:
                fh.write(json.dumps({"name": r["name"], **x}, default=float) + "\n")
    print(json.dumps({"verdict": out["verdict"], "all_after": summary["all"]["after"]}, indent=1,
                     ensure_ascii=False))
    return out


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    main()
