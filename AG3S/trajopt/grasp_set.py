"""T72 — collision-aware geometric grasp set as a reach constraint in TrajOpt (`cost.grasp_set`, default off).

`serve_safe --grasp-set-to` (task `AG3S/ag3s/docs/handoff/T72.task.md`, fixed before results). With the flag off the
controller is never built: no candidate, no reach box, no record key, the request is solved as before.

Per request (`SafePolicy._grasp_set_request`, at the end of `_scene_fn` — after the snapshot the TrajOpt solves on is
final, so the field check below reads exactly that scene):

1. **Target memory** — the ATS target points of every pre-grasp request are merged (`TargetPointMemory`, as G1).
2. **Window** (task §2c "작동 창", independent of HSR6 ④): latch `latched` with the policy chunk closing at row
   0 < c < H, or latch `closing`; and the A6 "grasp moment" window (palm site within `grasp_set_gate` of the target,
   `SafePolicy._grasp_moment_window`). Otherwise the request is untouched (`reason`).
3. **VLA intent** T_v = (p_v, R_v): the hold link at the policy's close row c (FK of the chunk's row c). Frozen while
   closing (the last latched intent of this grasp attempt; none -> this closing request's chunk, frozen from then on).
4. **Candidate set G** (task §2a) — `generate_candidates` (mu `grasp_set_mu`, observed contacts only, budget
   `grasp_set_time_budget_ms`, at most `grasp_set_max_candidates`, IK required). Two collision checks: the generator's
   own point checks (palm / finger boxes, completed target, approach segment) and **the TrajOpt's own rows**
   (`field_check`): at the grasp and pre-grasp IK the grasping hand's spheres (palm body, palm shell, fingers) are
   queried with the same linearizer functions the solver uses (`_esdf_clearance`, `_clearances_from`) on the same
   `SceneSnapshot` (field, margins, target-free rows, A6 / NBC relaxations): ``d - r - margin >= 0`` on every row.
5. **Quality** (task §2b) ``Q = (q_fric + q_centre + q_support) / 3`` (`quality`).
6. **Objective** (task §2c) ``J = dq' W dq / sigma_q^2 + lambda_I [(dp / sigma_p)^2 + (dtheta / sigma_theta)^2] +
   lambda_Q (1 - Q)`` (`objective`); eligibility dp <= `grasp_set_max_pos`, dtheta <= `grasp_set_max_rot_deg`.
   Selection = argmin J with hysteresis (`choose`).
7. **Reach constraint** — a `GraspAnchor` with a per-axis `grasp_set_box` box on `grasp_set_rows` rows from c (closing:
   rows 1.., only while the palm is within `grasp_set_closing_radius` of the grasp), slack at `w_slack`
   (`goal_box_slack_scale = 1`), rotation ``grasp_set_w_rot ||Log(R_g^T R)||^2`` (finger-swap twin nearest the hand
   now), no position cost, no tracking change (`reach_constraint = True`).
8. After the solve (`reach_record`): `reach_err_mm` = |hold link at the returned chunk's first reach row - g|, the same
   definition whichever path (standard, safety-first re-solve, GPU / CPU seed) gave the trajectory.

Simulator ground truth is never read here.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Optional

import numpy as np

ACTIVE = ("latched", "closing")


# ------------------------------------------------------------------------------------------------ task §2b-c formulas
def quality(alignment: float, rho: Optional[float], support, *, mu: float, min_support: int) -> dict[str, Any]:
    """Task §2b: `Q = (q_fric + q_centre + q_support) / 3`, each in [0, 1].

    q_fric = clip((alignment - cos atan(mu)) / (1 - cos atan(mu)), 0, 1); q_centre = 1 - min(rho, 1) (rho = None: no
    sphere fit -> 0.5); q_support = min(min(support) / (2 min_support), 1)."""
    c = math.cos(math.atan(float(mu)))
    q_fric = min(max((float(alignment) - c) / (1.0 - c), 0.0), 1.0)
    r = 0.5 if rho is None else float(rho)
    q_centre = 1.0 - min(r, 1.0)
    q_support = min(float(min(int(s) for s in support)) / (2.0 * float(min_support)), 1.0)
    return {"Q": (q_fric + q_centre + q_support) / 3.0, "q_fric": q_fric, "q_centre": q_centre,
            "q_support": q_support, "rho": None if rho is None else float(rho)}


def objective(dq: np.ndarray, W: np.ndarray, dp: float, dtheta: float, Q: float, cost) -> dict[str, float]:
    """Task §2c: the three parts of `J_i` (`J_track` = dq' W dq / sigma_q^2, `J_intent`, `J_quality`) and their sum."""
    dq = np.asarray(dq, np.float64).reshape(-1)
    W = np.asarray(W, np.float64).reshape(-1)
    j_track = float(dq @ (W * dq)) / float(cost.grasp_set_sigma_q) ** 2
    st = math.radians(float(cost.grasp_set_sigma_theta_deg))
    j_intent = float(cost.grasp_set_lambda_intent) * ((float(dp) / float(cost.grasp_set_sigma_p)) ** 2
                                                       + (float(dtheta) / st) ** 2)
    j_quality = float(cost.grasp_set_lambda_quality) * (1.0 - float(Q))
    return {"J": j_track + j_intent + j_quality, "J_track": j_track, "J_intent": j_intent, "J_quality": j_quality}


def eligible(dp: float, dtheta: float, cost) -> bool:
    """Task §2c: within `grasp_set_max_pos` (m) and `grasp_set_max_rot_deg` of the VLA intent."""
    return float(dp) <= float(cost.grasp_set_max_pos) and float(dtheta) <= math.radians(float(cost.grasp_set_max_rot_deg))


def same_grasp(T_a: np.ndarray, T_b: np.ndarray, cost) -> bool:
    """Task §2c "같은 grasp": hold-link positions within `grasp_set_same_pos` and closing axes (hold-link x, sign-agnostic
    — the finger swap is the same grasp) within `grasp_set_same_axis_deg`."""
    T_a, T_b = np.asarray(T_a, np.float64), np.asarray(T_b, np.float64)
    if float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])) > float(cost.grasp_set_same_pos):
        return False
    return abs(float(T_a[:3, 0] @ T_b[:3, 0])) >= math.cos(math.radians(float(cost.grasp_set_same_axis_deg)))


def choose(J: list[float], T: list[np.ndarray], ok: list[bool], prev_T: Optional[np.ndarray], cost
           ) -> tuple[Optional[int], dict[str, Any]]:
    """Task §2c selection: argmin J over the eligible candidates (ties: the earlier index). Hysteresis: when the previous
    request's grasp is still valid (an eligible candidate is the same grasp), switch to the argmin only when its J is
    below `grasp_set_switch_ratio` x the J of that (best-matching) candidate."""
    idx = [i for i, o in enumerate(ok) if o]
    if not idx:
        return None, {"previous": "none" if prev_T is None else "no eligible candidate"}
    best = min(idx, key=lambda i: (float(J[i]), i))
    if prev_T is None:
        return best, {"previous": "none", "changed": False}
    match = [i for i in idx if same_grasp(T[i], prev_T, cost)]
    if not match:
        return best, {"previous": "lost", "changed": True}
    m = min(match, key=lambda i: (float(J[i]), i))
    if best != m and float(J[best]) < float(cost.grasp_set_switch_ratio) * float(J[m]):
        return best, {"previous": "switched", "changed": True, "kept_id": int(m), "kept_J": float(J[m])}
    return m, {"previous": "kept", "changed": False, **({"argmin_id": int(best)} if best != m else {})}


# ------------------------------------------------------------------------------------------------ TrajOpt row check
def hand_rows(robot_model, hold_link: str, grasp_link: str) -> np.ndarray:
    """`(S,)` bool: the grasping hand's sphere rows — every sphere on the hold link (`ee_<side>`: palm body + palm shell,
    `safe_policy.palm_contact_rows`) and on its finger links (the grasp link without trailing digits, as IFG-NBC)."""
    names = [str(n) for n in getattr(robot_model, "sphere_link_names", ())]
    prefix = str(grasp_link).rstrip("0123456789")
    return np.asarray([n == str(hold_link) or n.startswith(prefix) for n in names], bool)


def field_check(linearizer, scene, q_full: list[np.ndarray], params: Optional[list[np.ndarray]], rows: np.ndarray
                ) -> tuple[bool, dict[str, Any]]:
    """Task §2a-2 — the TrajOpt's own clearance of the masked sphere rows at configurations `q_full` (full model q each;
    `params`: the finger joints per configuration, None = the model's). The solver's functions on the solver's scene:
    `linearizer.config_states` (sphere centres), `_clearances_from` (support-plane rows) and `_esdf_clearance` (field
    rows: `d - r - margin` with every per-row rule of that `SceneSnapshot`). OK when every masked row is >= 0. The
    linearizer's per-call record attributes are restored (the solve that follows writes its own)."""
    from AG3S.trajopt.linearize import _CLEARANCE_SIDE_ATTRS, _MISSING

    full = np.stack([np.asarray(q, np.float64) for q in q_full], axis=1)
    P = None if params is None else np.stack([np.asarray(p, np.float64) for p in params], axis=1)
    saved = {a: linearizer.__dict__.get(a, _MISSING) for a in _CLEARANCE_SIDE_ATTRS}
    try:
        centres, _ = linearizer.config_states(full, P, with_jac=False)
        cand, plane, _ = linearizer._clearances_from(centres, scene)
        esdf = linearizer._esdf_clearance(centres, scene)
    finally:
        for a, v in saved.items():
            if v is _MISSING:
                linearizer.__dict__.pop(a, None)
            else:
                linearizer.__dict__[a] = v
    S = int(linearizer.n_spheres)
    rows = np.asarray(rows, bool).reshape(-1)[:S]
    n = full.shape[1]
    blocks = {"field": esdf[:, :S, 0] if esdf.size else np.full((n, S), np.inf),
              "plane": plane.min(axis=2) if plane.size else np.full((n, S), np.inf),
              "candidate": cand.min(axis=2) if cand.size else np.full((n, S), np.inf)}
    worst = {"clearance_mm": math.inf}
    per_pose = []
    for k in range(n):
        m_k = math.inf
        for name, b in blocks.items():
            v = b[k][rows]
            if v.size and float(v.min()) < m_k:
                m_k = float(v.min())
                if m_k < worst["clearance_mm"] / 1000.0:
                    j = int(np.flatnonzero(rows)[int(np.argmin(v))])
                    worst = {"pose": k, "block": name, "row": j, "clearance_mm": round(m_k * 1000.0, 3),
                             "link": str(linearizer.robot_model.sphere_link_names[j])}
        per_pose.append(None if not math.isfinite(m_k) else round(m_k * 1000.0, 3))
    ok = all(v is None or v >= 0.0 for v in per_pose)
    return ok, {"ok": bool(ok), "min_clearance_mm": per_pose, "n_rows": int(rows.sum()),
                **({"worst": worst} if math.isfinite(worst["clearance_mm"]) else {})}


# ------------------------------------------------------------------------------------------------ controller
class GraspSetController:
    """The per-episode state of T72 (target memory, frozen intent, previous selection) and its per-request steps."""

    def __init__(self, cost, constraint_robot_model, layout, hold_links: dict, gripper_columns, closed_below: float,
                 limits_cfg, horizon: int, w_track: float = 1.0):
        from AG3S.ag3s.grasp.candidates import GeoGraspConfig
        from AG3S.ag3s.grasp.memory import TargetPointMemory

        self.cost = cost
        self.model = constraint_robot_model
        self.layout = layout
        self.hold_links = dict(hold_links)
        self.gripper_columns = tuple(gripper_columns)
        self.closed_below = float(closed_below)
        self.limits_cfg = limits_cfg
        self.horizon = int(horizon)
        self.w_track = float(w_track)
        n = int(cost.grasp_set_max_candidates)
        # task §2a: the existing generator, mu 0.8, observed contacts only, 150 ms, at most 8 candidates, IK required.
        # `ik_accept` = `max_candidates` so up to 8 pass IK (the generator's default 3 stops after three).
        self.gcfg = GeoGraspConfig(friction_mu=float(cost.grasp_set_mu), max_candidates=n, ik_accept=n,
                                   max_ik=max(n, GeoGraspConfig().max_ik),
                                   time_budget_ms=float(cost.grasp_set_time_budget_ms))
        self.memory = TargetPointMemory()
        self._iks: dict[str, Any] = {}
        self.reset()
        for hand in self.hold_links:      # CasADi IK built now, not on the first request of an episode
            try:
                self._ik(hand)
            except Exception:   # noqa: BLE001 — a model without that hand: built (and failing) on use
                pass

    def reset(self) -> None:
        self.memory.reset()
        self._phase = "searching"
        self._intent: Optional[dict[str, Any]] = None
        self._prev_T: Optional[np.ndarray] = None
        self._anchor = None
        self._reach: Optional[dict[str, Any]] = None
        self.last: dict[str, Any] = {}

    # --- helpers --------------------------------------------------------------------------------------------------
    def _ik(self, hand: str):
        if hand not in self._iks:
            from AG3S.ag3s.grasp.ik import LinkPoseIK, mjcf_limits
            from AG3S.trajopt.limits import limits_model_xml

            names = list(self.model.joint_names)
            idx = [i for i, n in enumerate(names) if n.startswith(f"{hand}_arm_")]
            try:
                lim = mjcf_limits(limits_model_xml(self.limits_cfg), [names[i] for i in idx])
            except Exception:   # noqa: BLE001 — no MJCF: the URDF limits (narrower wrist; said in the record)
                lim = None
            self._iks[hand] = (LinkPoseIK(self.model, self.hold_links.get(hand, f"ee_{hand}"), idx, limits=lim), idx,
                               "mjcf" if lim is not None else "urdf")
        return self._iks[hand]

    def _pose(self, q_full: np.ndarray, link: str) -> np.ndarray:
        from AG3S.ag3s.constraints.attached import _link_pose_numeric

        return np.asarray(_link_pose_numeric(self.model, q_full, link), np.float64)

    def _chunk_q(self, chunk: np.ndarray, row: int, q_now: np.ndarray) -> np.ndarray:
        traj = self.layout.chunk_to_trajectory(np.asarray(chunk, np.float64))
        return self.layout.full_q(traj[:, row:row + 1], q_now)[:, 0]

    def close_row(self, chunk: np.ndarray, hand: str) -> Optional[int]:
        col = self.gripper_columns[0 if hand == "left" else 1]
        closes = np.flatnonzero(np.asarray(chunk)[:, col] < self.closed_below)
        return int(closes[0]) if closes.size else None

    def _attempt(self, phase: str) -> None:
        """A new grasp attempt (episode start, after a place, a re-grasp after a lost / failed close): the frozen
        intent and the previous selection are dropped (the memory follows the target as G1's)."""
        prev, self._phase = self._phase, str(phase)
        if (self._phase in ("searching", "placed") and prev not in ("searching", "placed")) or \
                (self._phase == "latched" and prev in ("held", "closing")):
            self._intent, self._prev_T = None, None

    # --- per request ----------------------------------------------------------------------------------------------
    def request(self, *, phase: str, hand: str, grasp_link: str, manip, cloud_points: Optional[np.ndarray],
                q_now: np.ndarray, chunk: np.ndarray, window: Optional[tuple], window_rec: dict, scene, linearizer,
                track_scale: Optional[np.ndarray] = None):
        """This request's reach `GraspAnchor` or None; the record is `self.last` (`summary_json.grasp.grasp_set`)."""
        from scipy.spatial import cKDTree

        t0 = time.perf_counter()
        cost = self.cost
        self._anchor, self._reach = None, None
        rec: dict[str, Any] = {"active": False, "phase": str(phase), "hand": hand}
        self.last = rec
        self._attempt(phase)
        q_now = np.asarray(q_now, np.float64)
        chunk = np.asarray(chunk, np.float64)
        target_id = None if manip is None else int(manip.id)
        state = None if manip is None else str(getattr(manip, "state", None))
        geom = None if manip is None else getattr(manip, "geometry", None)
        tp = None if (geom is None or state != "visible") else np.asarray(geom.points, np.float64)
        cen = None if manip is None else np.asarray(manip.centroid, np.float64).reshape(3)
        if self._phase in ("searching",) + ACTIVE and target_id is not None:
            rec["memory"] = self.memory.update(target_id, tp, cen, observed=state == "visible")
        else:
            rec["memory"] = self.memory.update(None, None, None)

        def done(reason: str):
            rec["reason"] = reason
            rec["gen_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            return None

        if self._phase not in ACTIVE:
            return done(f"phase {self._phase}")
        H = self.horizon
        link = self.hold_links.get(hand, f"ee_{hand}")
        c = self.close_row(chunk, hand)
        rec["vla_close_row"] = c
        if self._phase == "latched" and (c is None or not 0 < c < H):
            return done("no close planned in this horizon" if c is None or c >= H else "chunk starts closed")
        rec["window"] = {k: window_rec.get(k) for k in ("site_to_target_mm", "reason") if k in window_rec}
        if window is None:
            return done(f"grasp-moment window closed: {window_rec.get('reason')}")
        # VLA intent T_v (task §2c): the hold link at the policy's close row; frozen while closing
        if self._phase == "latched":
            q_v = self._chunk_q(chunk, c, q_now)
            self._intent = {"q": q_v, "T": self._pose(q_v, link), "close_row": int(c), "source": "latched close row"}
        elif self._intent is None:
            row = c if (c is not None and c < H) else 0
            q_v = self._chunk_q(chunk, row, q_now)
            self._intent = {"q": q_v, "T": self._pose(q_v, link), "close_row": int(row),
                            "source": "closing start (no latched intent)"}
        intent = self._intent
        T_v, q_v = intent["T"], intent["q"]
        rec["intent"] = {"source": intent["source"], "close_row": intent["close_row"],
                         "p_m": [round(float(v), 5) for v in T_v[:3, 3]], "frozen": self._phase == "closing"}
        if scene is None:
            return done("no TrajOpt scene this request")
        pts = self.memory.points
        if not len(pts):
            return done("no target points")
        sc = None
        if cloud_points is not None and len(cloud_points):
            sc = np.asarray(cloud_points, np.float64).reshape(-1, 3)
            d, _ = cKDTree(pts).query(sc, distance_upper_bound=0.005)
            sc = sc[~np.isfinite(d)]
        # the TrajOpt row check of each candidate at its grasp / pre-grasp IK (task §2a-2)
        from AG3S.ag3s.grasp.candidates import generate_candidates, grasp_rot_angle, nearest_symmetric
        from AG3S.ag3s.grasp.gripper import spec_for
        from AG3S.trajopt.geo_grasp import chord_offset

        ik, idx, lim_src = self._ik(hand)
        rows = hand_rows(linearizer.robot_model, link, grasp_link)
        goal_row = 1 if self._phase == "closing" else int(c)
        params = None
        if getattr(linearizer, "param_names", ()):
            path = np.asarray(linearizer.param_path(), np.float64)
            params = [path[:, min(goal_row, path.shape[1] - 1)], path[:, 0]]
        field_rejected: list[dict[str, Any]] = []

        def config_check(g, q_sub, qp_sub):
            qg, qp = q_v.copy(), q_v.copy()
            qg[idx], qp[idx] = q_sub, qp_sub
            ok, info = field_check(linearizer, scene, [qg, qp], params, rows)
            if not ok:
                field_rejected.append({"pair": int(g.pair), "roll_deg": round(math.degrees(g.roll), 1),
                                       "min_clearance_mm": info["min_clearance_mm"], "worst": info.get("worst")})
            return ok, info

        ik_fn = lambda T, seed: ik.solve_seeds(T, q_now, ik.wrist_seeds(seed, extra=[q_now[idx]]))  # noqa: E731
        t_gen = time.perf_counter()
        if cost.grasp_set_field_only:   # T72b ②: the TrajOpt field check (config_check) is the only scene check
            sc = None
        cs = generate_candidates(pts, sc, spec_for(hand), self.gcfg, vla_pose=T_v, ik_fn=ik_fn, ik_seed=q_v[idx],
                                 config_check_fn=config_check)
        st = cs.stats
        rec["generator_ms"] = round((time.perf_counter() - t_gen) * 1000.0, 2)
        rej = dict(st.get("rejected") or {})
        rec["n_rejected"] = {"point_check": int(sum(rej.values())), "point_check_detail": rej,
                             "field_check": int(st.get("n_config_rejected", 0)), "ik": int(st.get("n_ik_failed", 0)),
                             "eligibility": 0}
        rec.update(n_candidates=len(cs.candidates), n_pairs_kept=st.get("n_pairs_kept"),
                   n_collision_free=st.get("n_collision_free"), n_ik_tried=st.get("n_ik_tried"),
                   budget_hit=st.get("budget_hit"), ik_limits=lim_src,
                   **({"generator_reason": st["reason"]} if st.get("reason") else {}))
        if field_rejected:
            rec["field_rejected"] = field_rejected
        # quality, objective, eligibility (task §2b-c)
        q_ix = np.asarray(self.layout.q_indices, int)
        W = self.w_track * (np.ones(q_ix.size) if track_scale is None
                            else np.asarray(track_scale, np.float64).reshape(-1))
        surf = cs.surface
        J, Ts, ok, cands = [], [], [], []
        for g in cs.candidates:
            qg = q_v.copy()
            qg[idx] = g.q_grasp
            dq = (qg - q_v)[q_ix]
            dp = float(np.linalg.norm(g.T[:3, 3] - T_v[:3, 3]))
            dth = grasp_rot_angle(T_v[:3, :3], g.T[:3, :3])
            qq = quality(g.alignment, chord_offset(g.contacts, surf), g.support, mu=float(cost.grasp_set_mu),
                         min_support=int(self.gcfg.min_support))
            jj = objective(dq, W, dp, dth, qq["Q"], cost)
            e = eligible(dp, dth, cost)
            J.append(jj["J"])
            Ts.append(g.T)
            ok.append(e)
            cands.append({"id": int(g.index), "eligible": bool(e), "Q": round(qq["Q"], 4),
                          "q_fric": round(qq["q_fric"], 4), "q_centre": round(qq["q_centre"], 4),
                          "q_support": round(qq["q_support"], 4), "rho": None if qq["rho"] is None else round(qq["rho"], 4),
                          "J": round(jj["J"], 4), "J_track": round(jj["J_track"], 4),
                          "J_intent": round(jj["J_intent"], 4), "J_quality": round(jj["J_quality"], 4),
                          "dp_mm": round(dp * 1000.0, 2), "dtheta_deg": round(math.degrees(dth), 2),
                          "dq_norm": round(float(np.linalg.norm(dq)), 4),
                          "position": [round(float(v), 5) for v in g.T[:3, 3]],
                          "field_min_mm": (g.checks.get("config") or {}).get("min_clearance_mm")})
        rec["n_rejected"]["eligibility"] = int(sum(1 for e in ok if not e))
        rec["n_eligible"] = int(sum(ok))
        rec["candidates"] = cands
        i, sel = choose(J, Ts, ok, self._prev_T, cost)
        if i is None:
            return done("no candidate" if not cs.candidates else "no eligible candidate")
        g = cs.candidates[i]
        self._prev_T = np.asarray(g.T, np.float64).copy()
        T_hand = self._pose(q_now, link)
        R_goal = nearest_symmetric(g.T[:3, :3], T_hand[:3, :3])
        sel_rec = {"id": int(g.index), "J": cands[i]["J"], "Q": cands[i]["Q"], "dp_mm": cands[i]["dp_mm"],
                   "dtheta_deg": cands[i]["dtheta_deg"], "dq_norm": cands[i]["dq_norm"], **sel}
        rec["selected"] = sel_rec
        rows_n = int(cost.grasp_set_rows)
        explicit = None
        if self._phase == "closing":
            dpalm = float(np.linalg.norm(T_hand[:3, 3] - g.T[:3, 3]))
            rec["palm_to_grasp_mm"] = round(dpalm * 1000.0, 1)
            if dpalm > float(cost.grasp_set_closing_radius):
                return done("closing away from the grasp")
            explicit = tuple(range(1, min(rows_n, H - 1) + 1))
        from AG3S.trajopt.grasp_anchor import GraspAnchor

        anchor = GraspAnchor(link=link, goal=np.asarray(g.T[:3, 3], np.float64).copy(), close_row=c, w_tcp=0.0,
                             w_goal=0.0, goal_rows=rows_n, goal_box=float(cost.grasp_set_box), goal_R=R_goal,
                             w_rot=float(cost.grasp_set_w_rot), explicit_rows=explicit,
                             goal_kind="grasp-set hold" if explicit else "grasp-set",
                             tol_pos=float(cost.grasp_set_box), tol_rot=math.radians(float(cost.grasp_set_max_rot_deg)),
                             goal_box_slack_scale=1.0, reach_constraint=True)
        mask = anchor.goal_mask(H)
        self._anchor = anchor
        self._reach = {"link": link, "goal": anchor.goal, "R": R_goal, "rows": [int(k) for k in np.flatnonzero(mask)],
                       "q_now": q_now}
        rec.update(active=True, kind=anchor.goal_kind, goal_m=[round(float(v), 5) for v in anchor.goal],
                   rows=self._reach["rows"], box_mm=round(float(cost.grasp_set_box) * 1000.0, 2))
        rec["gen_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        return anchor

    def reach_record(self, refined: np.ndarray, solve_ms: Optional[float]) -> None:
        """After the solve: the returned chunk's hold link on the reach rows vs the selected grasp (`reach_err_mm` = the
        first reach row), the per-axis box excess and the rotation error; `solve_ms` = the TrajOpt's solve time."""
        rec = self.last
        if not rec.get("active") or self._reach is None:
            return
        from AG3S.ag3s.grasp.candidates import rot_angle

        r = self._reach
        rows = r["rows"]
        rec["solve_ms"] = None if solve_ms is None else round(float(solve_ms), 2)
        if not rows:
            rec["reach_err_mm"] = None
            return
        g = np.asarray(r["goal"], np.float64)
        err, exc, rot = [], [], []
        for k in rows:
            T = self._pose(self._chunk_q(refined, k, r["q_now"]), r["link"])
            err.append(float(np.linalg.norm(T[:3, 3] - g)))
            exc.append(float(np.max(np.maximum(np.abs(T[:3, 3] - g) - float(self.cost.grasp_set_box), 0.0))))
            rot.append(rot_angle(T[:3, :3], r["R"]))
        rec.update(reach_err_mm=round(err[0] * 1000.0, 2), reach_err_rows_mm=[round(e * 1000.0, 2) for e in err],
                   reach_box_excess_max_mm=round(max(exc) * 1000.0, 3),
                   reach_rot_err_deg=round(math.degrees(rot[0]), 2))


__all__ = ["ACTIVE", "GraspSetController", "choose", "eligible", "field_check", "hand_rows", "objective", "quality",
           "same_grasp"]
