"""AG3S-GEO-GRASP — the geometric 6-D grasp goal inside `SafePolicy` (`cost.geo_grasp`, default off).

`GeoGraspController` is what `SafePolicy` calls; with `cost.geo_grasp = False` it is never built and no request
differs. Per request (after the grasp latch ran, so the phase is this chunk's final one):

1. `step` — the task state (`AG3S.ag3s.grasp.planner`): ATS target points (`manipulated.geometry.points`, merged
   before the grasp), scene points (the self-filtered cloud without the target), the policy's planned close pose
   (preference / IK seed). One grasp is kept; it is never chosen inside the optimizer.
2. `anchor` — the `GraspAnchor` (variant "GEO") of this chunk, by phase:

   | phase | goal | rows |
   |---|---|---|
   | `latched`, the policy chunk closes at row c > 0 | grasp pose | `grasp_goal_rows` rows from c |
   | `latched`, no close in the horizon, palm within `geo_approach_radius` of the pre-grasp | toward the pre-grasp, at most `geo_approach_step` from the palm (rotation by the same fraction) | last planned row |
   | `closing`, palm within `geo_close_hold_radius` of the grasp | grasp pose (hold the hand while the fingers close; no lift before `held`) | rows 1..`grasp_goal_rows` |
   | otherwise | none — the chunk is solved as without the feature | — |

   The goal rotation is the grasp's or its finger-swapped twin, whichever is closer to the hand now.
3. `track_scale` / `continuity_off` (B2, only with `geo_track_scale < 1` / `geo_continuity_off`): while a goal is
   active the grasping arm's tracking weight is scaled and the continuity term dropped; from `held` on the scale
   returns to 1 over `geo_carry_requests` requests (continuity as without the feature).
4. `close_gate` — in `latched`, when the refined chunk starts closing inside the execution window, the close is kept
   only if the planned hand at that row is within `geo_close_tol_*` of the grasp; otherwise the grip column stays
   open on those rows (the next request plans again). After `geo_close_timeout` deferred requests, or when the goal
   is lost after control started and its one re-selection, the controller stops the robot: the chunk is held and the
   run is marked a geometric-control failure (`failed`). An unbounded HOLD is never a success.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

ACTIVE = ("latched", "closing")


def _so3_exp(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + math.sin(th) / th * K + (1.0 - math.cos(th)) / th ** 2 * (K @ K)


def chord_offset(contacts, surface) -> Optional[float]:
    """rho = distance from the observed sphere centre to the line through the two contacts, over the sphere radius
    (0 = through the centre); None without two contacts or a sphere fit (task §3e)."""
    if contacts is None or surface is None or getattr(surface, "centre_method", None) != "sphere" \
            or not getattr(surface, "radius", None):
        return None
    p = np.asarray(contacts, np.float64).reshape(2, 3)
    u = p[1] - p[0]
    n = float(np.linalg.norm(u))
    if n < 1e-9:
        return None
    u /= n
    v = np.asarray(surface.centre, np.float64).reshape(3) - p[0]
    return float(np.linalg.norm(v - (v @ u) * u) / float(surface.radius))


class GeoGraspController:
    def __init__(self, cost, constraint_robot_model, layout, hold_links: dict, gripper_columns, closed_below: float,
                 limits_cfg, horizon: int, execution_length: int):
        from AG3S.ag3s.grasp.candidates import GeoGraspConfig

        self.cost = cost
        self.model = constraint_robot_model
        self.layout = layout
        self.hold_links = dict(hold_links)
        self.gripper_columns = tuple(gripper_columns)
        self.closed_below = float(closed_below)
        self.limits_cfg = limits_cfg
        self.horizon = int(horizon)
        self.execution_length = int(execution_length)
        self.gcfg = GeoGraspConfig(friction_mu=float(cost.geo_grasp_mu), selection=str(cost.geo_grasp_selection),
                                   wrist_check=bool(getattr(cost, "geo_wrist_check", False)),
                                   arm_check=bool(getattr(cost, "geo_arm_check", False)),
                                   obstacle_memory=bool(getattr(cost, "geo_obstacle_memory", False)),
                                   obstacle_source=str(getattr(cost, "geo_obstacle_source", "points")),
                                   **({} if getattr(cost, "geo_time_budget_ms", None) is None
                                      else {"time_budget_ms": float(cost.geo_time_budget_ms)}))
        self._iks: dict[str, Any] = {}
        self._planners: dict[str, Any] = {}
        self._shapes: dict[str, Any] = {}
        self._q_now: Optional[np.ndarray] = None
        self._field_fn = None
        self.reset()
        # build the IK functions now (CasADi, ~0.6 s each) so the first request of an episode does not pay for it
        for hand in self.hold_links:
            try:
                self._ik(hand)
            except Exception:   # noqa: BLE001 — a model without that hand: built (and failing) on use instead
                pass

    # ------------------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        for p in self._planners.values():
            p.reset()
        self.started = False          # a goal has been active in this grasp attempt
        self.failed: Optional[str] = None
        self.deferred = 0
        self.held_requests = 0
        self.n_goal_chunks = 0
        self.last: dict[str, Any] = {}
        self._anchor_active = False
        self._correct_goal = None
        self._phase = "searching"
        self._hand = "left"

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

    def _arm_shapes(self, hand: str) -> dict:
        """The wrist / arm occupancy shapes of `hand` from the MJCF (task §3g-1), built once (~7 s)."""
        if hand not in self._shapes:
            from AG3S.ag3s.grasp.arm_shape import measure_arm_from_mjcf
            from AG3S.trajopt.limits import limits_model_xml

            from AG3S.ag3s.runtime.asset_path import resolve_asset

            xml = resolve_asset(limits_model_xml(self.limits_cfg), what="MJCF for the grasp wrist / arm shapes")
            self._shapes[hand] = measure_arm_from_mjcf(str(xml), hand,
                                                       hold_link=self.hold_links.get(hand, f"ee_{hand}"))
        return self._shapes[hand]

    def _planner(self, hand: str):
        if hand not in self._planners:
            from AG3S.ag3s.grasp.gripper import spec_for, with_wrist
            from AG3S.ag3s.grasp.planner import GeoGraspPlanner

            spec = spec_for(hand)
            arm_fn = None
            if self.gcfg.wrist_check:
                spec = with_wrist(spec, self._arm_shapes(hand)["wrist"], self.gcfg.scene_margin)
            if self.gcfg.arm_check:
                from AG3S.ag3s.grasp.arm_shape import ArmChecker

                _, idx, _ = self._ik(hand)

                def full_q(q_sub, idx=idx):
                    q = np.asarray(self._q_now, np.float64).copy()
                    q[idx] = q_sub
                    return q

                arm_fn = ArmChecker(self._arm_shapes(hand)["arm"], self._pose, full_q, self.gcfg.scene_margin)
            self._planners[hand] = GeoGraspPlanner(spec, self.gcfg, arm_fn=arm_fn)
        return self._planners[hand]

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

    # ------------------------------------------------------------------------------------------------------------
    def step(self, *, phase: str, hand: str, manip, cloud_points: Optional[np.ndarray], q_now: np.ndarray,
             chunk: np.ndarray, allowed_links: tuple = (), field=None) -> dict[str, Any]:
        """Run the task state for this request. Returns the record (also `self.last`). `field`: this request's
        trajectory-optimizer distance field (`constraint_set.esdf`); with `geo_obstacle_source = "field"` the grasp
        checks read its target-free distance (task §3j)."""
        from scipy.spatial import cKDTree

        self._hand = hand
        prev, self._phase = self._phase, str(phase)
        if (self._phase in ("searching", "placed") and prev not in ("searching", "placed")) or \
                (self._phase == "latched" and prev in ("held", "closing")):
            # a new grasp attempt (episode start, after a place, or a re-grasp after a lost / failed close)
            self.started, self.deferred = False, 0
            if self._phase != "latched":
                self.held_requests = 0
        if self._phase == "held":
            self.held_requests = self.held_requests + 1 if prev == "held" else 1
        link = self.hold_links.get(hand, f"ee_{hand}")
        ik, idx, lim_src = self._ik(hand)
        pl = self._planner(hand)
        # the policy's planned hand at its close row (else the 16th row): preference and IK seed
        c = self.close_row(chunk, hand)
        row = c if c is not None else min(15, len(chunk) - 1)
        q_vla = self._chunk_q(chunk, row, q_now)
        T_vla = self._pose(q_vla, link)
        target_id = None if manip is None else int(manip.id)
        state = None if manip is None else str(getattr(manip, "state", None))
        geom = None if manip is None else getattr(manip, "geometry", None)
        tp = None if (geom is None or state != "visible") else np.asarray(geom.points, np.float64)
        cen = None if manip is None else np.asarray(manip.centroid, np.float64).reshape(3)
        scene = None
        if cloud_points is not None and len(cloud_points):
            scene = np.asarray(cloud_points, np.float64).reshape(-1, 3)
            ref = pl.memory.points if len(pl.memory.points) else tp
            if ref is not None and len(ref):
                d, _ = cKDTree(ref).query(scene, distance_upper_bound=0.005)
                scene = scene[~np.isfinite(d)]
        ik_fn = lambda T, seed: ik.solve_seeds(T, q_now, ik.wrist_seeds(seed, extra=[q_now[idx]]))  # noqa: E731
        self._q_now = np.asarray(q_now, np.float64)
        self._field_fn, field_note = None, None
        if self.gcfg.obstacle_source == "field":
            if field is not None and getattr(field, "has_target_free", False):
                self._field_fn = field.target_free_distance
            else:
                # without a target-free layer the field answers the apple itself as an obstacle: no field this request
                field_note = "no target-free field layer: scene points used" if field is not None else "no field"
        pl.ik_fn = ik_fn
        r = pl.step(phase=self._phase, target_id=target_id, target_state=state, target_points=tp, target_centroid=cen,
                    scene_points=scene, vla_pose=T_vla, ik_seed=q_vla[idx], allowed_links=tuple(allowed_links),
                    field_fn=self._field_fn)
        st = r["state"]
        rec = {"phase": self._phase, "hand": hand, "generation": r["generation"], "n_candidates": r["n_candidates"],
               "valid": bool(st.get("valid")), "reason": st.get("reason"), "event": st.get("event"),
               "grasp": st.get("grasp"), "compute_ms": r["compute_ms"], "memory": r.get("memory"),
               "ik_limits": lim_src, "vla_close_row": c, "started": self.started, "failed": self.failed,
               "deferred": self.deferred}
        cs = r.get("candidate_stats") or {}
        if cs:
            rec["candidate_stats"] = {k: cs[k] for k in ("n_target_points", "n_pairs_kept", "n_collision_free",
                                                         "n_ik_tried", "n_ik_failed", "reason", "compute_ms",
                                                         # task §3g-1 (present only with those checks on)
                                                         "rejected", "n_arm_rejected", "budget_hit",
                                                         "n_scene_points_near", "n_scene_points_arm", "n_poses")
                                      if k in cs}
        if r.get("obstacles"):
            rec["obstacles"] = r["obstacles"]
        if self.gcfg.obstacle_source == "field":
            rec["obstacle_source"] = "field" if self._field_fn is not None else field_note
        self.last = rec
        return rec

    @property
    def selected(self):
        pl = self._planners.get(self._hand)
        return None if pl is None else pl.selected

    # ------------------------------------------------------------------------------------------------------------
    def anchor(self, q_now: np.ndarray, chunk: np.ndarray):
        """This chunk's `GraspAnchor` (variant "GEO") or None; fills `self.last["anchor"]`."""
        from AG3S.ag3s.grasp.candidates import nearest_symmetric
        from AG3S.trajopt.grasp_anchor import GraspAnchor

        cost = self.cost
        rec: dict[str, Any] = {"active": False}
        self.last["anchor"] = rec
        self._anchor_active = False
        self._correct_goal = None
        if getattr(cost, "geo_goal_mode", "redirect") == "correct":
            if self._phase == "latched" and not self.failed:
                return self._correct_anchor(q_now, chunk, rec)
            rec["reason"] = self.failed or f"correct mode: no goal in phase {self._phase}"
            return None
        g = self.selected
        if self._phase not in ACTIVE or self.failed:
            rec["reason"] = self.failed or f"phase {self._phase}"
            return None
        if g is None:
            rec["reason"] = "no valid grasp"
            return None
        link = self.hold_links.get(self._hand, f"ee_{self._hand}")
        T_hand = self._pose(q_now, link)
        R_goal = nearest_symmetric(g.T[:3, :3], T_hand[:3, :3])
        c = self.close_row(chunk, self._hand)
        H = self.horizon
        rows_n = int(getattr(cost, "grasp_goal_rows", 3))
        kind, goal, rows, close_row = None, None, None, None
        if self._phase == "closing":
            d = float(np.linalg.norm(T_hand[:3, 3] - g.T[:3, 3]))
            rec["palm_to_grasp_mm"] = round(d * 1000, 1)
            if d <= float(cost.geo_close_hold_radius):
                kind, goal, rows = "hold-at-grasp", g.T[:3, 3], tuple(range(1, min(rows_n, H - 1) + 1))
            else:
                # the fingers are closing somewhere else (not on this grasp): never drag the hand there mid-close
                rec["reason"] = "closing away from the grasp"
                return None
        elif c is not None and 0 < c < H:
            kind, goal, close_row = "grasp", g.T[:3, 3], c
        elif c is None or c >= H:
            # approach (no close in this horizon): the pre-grasp pose on the last planned row, only when near
            d = float(np.linalg.norm(T_hand[:3, 3] - g.T_pre[:3, 3]))
            rec["palm_to_pregrasp_mm"] = round(d * 1000, 1)
            if d <= float(cost.geo_approach_radius):
                # progress toward the pre-grasp, never more than `geo_approach_step` per horizon (no far goal forced
                # onto the end of a short horizon): position along the line, rotation by the same fraction
                from AG3S.ag3s.grasp.ik import so3_log

                f = min(1.0, float(cost.geo_approach_step) / max(d, 1e-9))
                goal = T_hand[:3, 3] + f * (g.T_pre[:3, 3] - T_hand[:3, 3])
                w = so3_log(T_hand[:3, :3].T @ R_goal) * f
                R_goal = T_hand[:3, :3] @ _so3_exp(w)
                kind, rows = ("pre-grasp" if f >= 1.0 else "toward pre-grasp"), (H - 1,)
                rec["progress_fraction"] = round(f, 3)
        if kind is None:
            rec["reason"] = ("chunk starts closed" if c == 0 else "palm outside the approach radius")
            return None
        self.started = True
        self._anchor_active = True
        self.n_goal_chunks += 1
        rec.update(active=True, kind=kind, goal_m=[round(float(v), 5) for v in goal],
                   rows=list(rows) if rows is not None else f"{rows_n} from close row {close_row}",
                   grasp_id=int(g.grasp_id),
                   hand_to_goal_mm=round(float(np.linalg.norm(T_hand[:3, 3] - goal)) * 1000, 1))
        return GraspAnchor(link=link, goal=np.asarray(goal, np.float64), close_row=close_row, w_tcp=0.0,
                           w_goal=float(cost.geo_w_pos), goal_rows=rows_n, goal_R=R_goal, w_rot=float(cost.geo_w_rot),
                           explicit_rows=rows, goal_kind=kind, tol_pos=float(cost.geo_close_tol_pos),
                           tol_rot=math.radians(float(cost.geo_close_tol_rot_deg)))

    def _correct_anchor(self, q_now: np.ndarray, chunk: np.ndarray, rec: dict[str, Any]):
        """`geo_goal_mode = "correct"` (user ruling 2026-10-27 after the pilot): no approach goal; on a chunk whose
        policy plans a close at row c, the valid geometric grasp nearest to the policy's own close pose — the tracked one
        or one of this request's candidates — becomes the goal from row c, only when it lies within `geo_correct_pos` /
        `geo_correct_rot_deg` of that pose (a small correction of where the policy closes, never a different grasp).
        The pilot showed the redirecting goal moved the hand 19 -> 53 mm off the apple at the policy's close (run 121)."""
        from AG3S.ag3s.grasp.candidates import grasp_rot_angle, nearest_symmetric
        from AG3S.trajopt.grasp_anchor import GraspAnchor

        cost, H = self.cost, self.horizon
        rows_n = int(getattr(cost, "grasp_goal_rows", 3))
        c = self.close_row(chunk, self._hand)
        if c is None or not (0 < c < H):
            rec["reason"] = "no close planned in this horizon" if c is None or c >= H else "chunk starts closed"
            return None
        link = self.hold_links.get(self._hand, f"ee_{self._hand}")
        q_now = np.asarray(q_now, np.float64)
        T_vla = self._pose(self._chunk_q(chunk, c, q_now), link)
        pool = []
        if self.selected is not None:
            pool.append(("tracked", self.selected.T, self.selected.grasp_id, getattr(self.selected, "contacts", None)))
        cs = getattr(self._planners.get(self._hand), "last_set", None)
        for gc in ([] if cs is None else cs.candidates):
            pool.append(("candidate", gc.T, gc.index, gc.contacts))
        tau = float(getattr(cost, "geo_correct_max_offset", 0.0))
        surf = None if cs is None else getattr(cs, "surface", None)
        best, n_off = None, 0
        for src, T, gid, contacts in pool:
            rho = chord_offset(contacts, surf)
            if tau > 0.0 and (rho is None or rho > tau):
                n_off += 1                        # §3e: an off-centre chord (or no sphere to tell) never corrects
                continue
            dp = float(np.linalg.norm(T[:3, 3] - T_vla[:3, 3]))
            dr = grasp_rot_angle(T_vla[:3, :3], T[:3, :3])
            key = dp + 0.05 * dr
            if best is None or key < best[0]:
                best = (key, src, T, gid, dp, dr, rho)
        if tau > 0.0:
            rec["centre_check"] = {"max_offset": tau, "excluded": n_off}
        if best is None:
            rec["reason"] = "no centred valid grasp" if n_off else "no valid grasp"
            return None
        _, src, T, gid, dp, dr, rho = best
        rec.update(nearest_source=src, nearest_pos_mm=round(dp * 1000, 1), nearest_rot_deg=round(math.degrees(dr), 1),
                   chord_offset=None if rho is None else round(rho, 3))
        if dp > float(cost.geo_correct_pos) or dr > math.radians(float(cost.geo_correct_rot_deg)):
            rec["reason"] = "no valid grasp near the policy's close"
            return None
        R_goal = nearest_symmetric(T[:3, :3], T_vla[:3, :3])
        self.started = True
        self._anchor_active = True
        self.n_goal_chunks += 1
        self._correct_goal = (np.asarray(T[:3, 3], np.float64).copy(), R_goal.copy())
        rec.update(active=True, kind="correct", goal_m=[round(float(v), 5) for v in T[:3, 3]],
                   rows=f"{rows_n} from close row {c}", grasp_id=int(gid))
        return GraspAnchor(link=link, goal=np.asarray(T[:3, 3], np.float64), close_row=c, w_tcp=0.0,
                           w_goal=float(cost.geo_w_pos), goal_rows=rows_n, goal_R=R_goal, w_rot=float(cost.geo_w_rot),
                           goal_kind="correct", tol_pos=float(cost.geo_close_tol_pos),
                           tol_rot=math.radians(float(cost.geo_close_tol_rot_deg)))

    # ------------------------------------------------------------------------------------------------------------
    def anchor_state(self) -> tuple:
        """AG3S-CORE-EVAL-HSR6 ④ — what `anchor` may change in this controller (`started`, `n_goal_chunks`), taken
        before it so `withdraw_anchor` can put it back."""
        return (bool(self.started), int(self.n_goal_chunks))

    def withdraw_anchor(self, state: tuple, reason: str) -> None:
        """AG3S-CORE-EVAL-HSR6 ④ — the goal `anchor` just produced is not given to the TO this chunk: no active goal
        (so no B2 tracking scale, no continuity-off, the close gate sees no correction = the policy's close), the
        controller's `started` / `n_goal_chunks` as before the call. The record keeps the withdrawn goal."""
        self.started, self.n_goal_chunks = bool(state[0]), int(state[1])
        self._anchor_active = False
        self._correct_goal = None
        rec = self.last.setdefault("anchor", {})
        rec.update(active=False, hsr6_withdrawn=True, reason=str(reason))

    # ------------------------------------------------------------------------------------------------------------
    def track_scale(self, nq_opt_names: list) -> Optional[np.ndarray]:
        """B2 — the grasping arm's tracking scale this chunk, or None (unchanged)."""
        s = float(self.cost.geo_track_scale)
        rec: dict[str, Any] = {"scale": None}
        self.last["track"] = rec
        if s >= 1.0:
            return None
        if self._anchor_active:
            k = s
        elif self._phase == "held" and self.started and 0 < self.held_requests <= int(self.cost.geo_carry_requests):
            n = max(int(self.cost.geo_carry_requests), 1)
            k = s + (1.0 - s) * self.held_requests / n
            if k >= 1.0:
                return None
        else:
            return None
        rec["scale"] = round(k, 4)
        return np.array([k if str(nm).startswith(f"{self._hand}_arm_") else 1.0 for nm in nq_opt_names], np.float64)

    def continuity_off(self) -> Optional[str]:
        """B2 — `"geo-grasp goal active"` while a goal is active (with `geo_continuity_off`), else None."""
        off = "geo-grasp goal active" if (self.cost.geo_continuity_off and self._anchor_active) else None
        self.last["continuity_off"] = off
        return off

    # ------------------------------------------------------------------------------------------------------------
    def close_gate(self, refined: np.ndarray, q_now: np.ndarray) -> tuple[np.ndarray, Optional[str]]:
        """`(chunk, hold_reason)`: the refined chunk with a deferred close (grip column kept open), and a reason to
        hold the chunk when the geometric control failed. Fills `self.last["close_gate"]`."""
        from AG3S.ag3s.grasp.candidates import grasp_rot_angle

        rec: dict[str, Any] = {"applied": False}
        self.last["close_gate"] = rec
        if self.failed:
            rec["failed"] = self.failed
            return refined, f"geometric grasp control failed: {self.failed}"
        if not self.started or self._phase not in ACTIVE:
            return refined, None
        g = self.selected
        if g is None and getattr(self.cost, "geo_goal_mode", "redirect") != "correct":
            # lost after control started and its one re-selection was used too (task_state) -> stop
            if "re-selection used" in str(self.last.get("reason")):
                self.failed = "grasp goal lost after its re-selection"
                rec["failed"] = self.failed
                return refined, f"geometric grasp control failed: {self.failed}"
        if self._phase != "latched":
            return refined, None          # the fingers are already closing: never re-open them
        hand = self._hand
        col = self.gripper_columns[0 if hand == "left" else 1]
        c = self.close_row(refined, hand)
        if c is None or c >= self.execution_length:
            return refined, None
        if getattr(self.cost, "geo_goal_mode", "redirect") == "correct":
            # correction mode: the policy's own close is never vetoed; only a close this chunk corrected is held back
            # when the optimizer did not bring the hand onto the corrected grasp
            if getattr(self, "_correct_goal", None) is None:
                rec["decision"] = "close"
                rec["by"] = "no correction this chunk: the policy's close"
                self.deferred = 0
                return refined, None
            link = self.hold_links.get(hand, f"ee_{hand}")
            T = self._pose(self._chunk_q(refined, c, q_now), link)
            gp, gR = self._correct_goal
            dp = float(np.linalg.norm(T[:3, 3] - gp))
            dr = grasp_rot_angle(T[:3, :3], gR)
            rec.update(close_row=c, pos_err_mm=round(dp * 1000, 1), rot_err_deg=round(math.degrees(dr), 1))
            if dp <= float(self.cost.geo_close_tol_pos) and dr <= math.radians(float(self.cost.geo_close_tol_rot_deg)):
                rec["decision"], rec["by"] = "close", "on the corrected grasp"
                self.deferred = 0
                return refined, None
            ok, why = False, "the corrected close missed its grasp"
            self.deferred += 1
            rec.update(applied=True, decision="defer", reason=why, deferred=self.deferred)
            if self.deferred > int(self.cost.geo_close_timeout):
                self.failed = f"close deferred {self.deferred} requests (> {int(self.cost.geo_close_timeout)})"
                rec["failed"] = self.failed
                return refined, f"geometric grasp control failed: {self.failed}"
            out = np.array(refined, np.float64, copy=True)
            open_value = float(np.max(out[:c, col])) if c > 0 else 1.0
            out[c:, col] = max(open_value, self.closed_below + 1e-3)
            return out, None
        from AG3S.ag3s.grasp.candidates import close_pose_check

        link = self.hold_links.get(hand, f"ee_{hand}")
        T = self._pose(self._chunk_q(refined, c, q_now), link)
        rec["close_row"] = c
        ok, why = False, "no grasp goal"
        if g is not None:
            dp = float(np.linalg.norm(T[:3, 3] - g.T[:3, 3]))
            dr = grasp_rot_angle(T[:3, :3], g.T[:3, :3])
            rec.update(pos_err_mm=round(dp * 1000, 1), rot_err_deg=round(math.degrees(dr), 1))
            ok = dp <= float(self.cost.geo_close_tol_pos) and dr <= math.radians(float(self.cost.geo_close_tol_rot_deg))
            why = None if ok else "planned hand outside the grasp tolerance"
            rec["by"] = "near the selected grasp" if ok else None
        if not ok and getattr(self.cost, "geo_close_gate", "selected") == "valid-pose":
            # the planned close pose judged on its own: an antipodal pair on the observed surface under both pads and
            # a collision-free open hand (AG3S-GEO-GRASP pilot 2026-10-26: the policy's own close was deferred 9 times
            # in 5 of 12 runs because it was not near *our* grasp)
            cs = getattr(self._planners.get(hand), "last_set", None)
            valid, info = close_pose_check(T, self._planner(hand).gripper, self.gcfg,
                                           None if cs is None else cs.surface, None if cs is None else cs.scene,
                                           None if cs is None else cs.shell,
                                           field_fn=getattr(self, "_field_fn", None))
            rec["pose_check"] = info
            if valid is None:
                ok, why, rec["by"] = True, None, "unverifiable (no observed surface): not blocked"
            elif valid:
                ok, why, rec["by"] = True, None, "the planned close pose is a valid grasp"
            else:
                why = f"planned close pose invalid: {info.get('reason')}"
        if ok:
            self.deferred = 0
            rec["decision"] = "close"
            return refined, None
        self.deferred += 1
        rec.update(applied=True, decision="defer", reason=why, deferred=self.deferred)
        if self.deferred > int(self.cost.geo_close_timeout):
            self.failed = f"close deferred {self.deferred} requests (> {int(self.cost.geo_close_timeout)})"
            rec["failed"] = self.failed
            return refined, f"geometric grasp control failed: {self.failed}"
        out = np.array(refined, np.float64, copy=True)
        # keep the grip where it was before the close row (fully open, 1.0, when the chunk closes from row 0)
        open_value = float(np.max(out[:c, col])) if c > 0 else 1.0
        out[c:, col] = max(open_value, self.closed_below + 1e-3)
        return out, None


__all__ = ["GeoGraspController"]
