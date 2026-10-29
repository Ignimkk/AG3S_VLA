"""AG3S-GEO-GRASP — one shared task state: which object, which phase, which grasp, and why it is (in)valid.

ATS gives the target identity and its observed points (`ManipulatedIdentity`), HSR the phase (the grasp latch:
``searching / latched / closing / held / placed``), LTE the contact permission (authorized links). This object only
**remembers the choice** between requests; it does not duplicate any of those state machines.

Rules (each request, `update`):

* Active only before the grasp is confirmed (``latched`` / ``closing``). ``held`` keeps the last grasp as the record of
  what was grasped (`carry`), ``searching`` / ``placed`` clear it.
* The target identity and the selected contact relation persist across chunks; the **pose follows the observation**:
  the tracked grasp is moved with the target's centre (sphere-fit centre, not an orientation estimate — the apple's
  full rotation is not trusted), then re-matched to a fresh candidate within `match_pos` / `match_rot_deg`.
* Invalidated (reason recorded) on: target id change, target lost, centre jump > `max_centre_jump`, the tracked pose
  failing the collision checks against the newest observation, or (when IK is given) no IK for it.
* A different valid candidate replaces the tracked one only when the tracked one is invalid, or when its key is better
  by `switch_margin` — small score changes never switch the grasp. While ``closing`` nothing switches.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

import numpy as np

from AG3S.ag3s.grasp.candidates import (CandidateSet, GeoGraspConfig, GraspCandidate, grasp_rot_angle, pose_checks,
                                        preference_key)
from AG3S.ag3s.grasp.gripper import ParallelGripperSpec

ACTIVE_PHASES = ("latched", "closing")


@dataclasses.dataclass
class SelectedGrasp:
    grasp_id: int                     # stable id of this choice within the episode
    T: np.ndarray                     # hold-link pose (base frame)
    T_pre: np.ndarray
    open_width: float
    width: float
    contacts: np.ndarray              # (2, 3) contact points (base frame), moved with the target centre
    centre: np.ndarray                # target centre when this pose was last set
    key: float
    since_request: int
    last_matched_request: int
    q_grasp: Optional[np.ndarray] = None
    q_pre: Optional[np.ndarray] = None
    alignment: float = 0.0

    def shifted(self, delta: np.ndarray) -> "SelectedGrasp":
        g = dataclasses.replace(self, T=self.T.copy(), T_pre=self.T_pre.copy(), contacts=self.contacts + delta,
                                centre=self.centre + delta)
        g.T[:3, 3] += delta
        g.T_pre[:3, 3] += delta
        return g


class GraspTaskState:
    """See the module docstring. One per hand and episode (`reset` between episodes)."""

    def __init__(self, gripper: ParallelGripperSpec, cfg: GeoGraspConfig = GeoGraspConfig()):
        self.gripper = gripper
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.target_id: Optional[int] = None
        self.phase: str = "searching"
        self.observed_time: Optional[float] = None
        self.confidence: Optional[float] = None
        self.allowed_links: tuple[str, ...] = ()
        self.selected: Optional[SelectedGrasp] = None
        self.carry: Optional[SelectedGrasp] = None
        self.valid: bool = False
        self.reason: str = "no target"
        self._request = -1
        self._next_id = 0
        self.n_switches = 0
        self.n_invalidations = 0
        self.closing_reselects = 0
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------------------------------------------------
    def update(self, *, phase: str, target_id: Optional[int], target_state: Optional[str],
               candidates: Optional[CandidateSet], observed_time: Optional[float] = None,
               confidence: Optional[float] = None, allowed_links: tuple[str, ...] = (),
               ik_fn=None, ik_seed: Optional[np.ndarray] = None, same_request: bool = False,
               vla_pose: Optional[np.ndarray] = None, arm_fn=None) -> dict[str, Any]:
        """One request. `candidates`: this request's `generate_candidates` result (None = not run). Returns the record.
        `same_request`: a second update within the same request (after a full generation the first one asked for)."""
        if not same_request:
            self._request += 1
        cfg = self.cfg
        rec: dict[str, Any] = {"request": self._request, "phase": str(phase), "target_id": target_id,
                               "target_state": target_state, "event": None}
        self.phase = str(phase)
        self.allowed_links = tuple(allowed_links)
        if self.phase == "held":
            if self.selected is not None:
                self.carry = self.selected
            self.selected = None
            return self._finish(rec, False, "grasp confirmed (held): carry", "held")
        if self.phase not in ACTIVE_PHASES:
            ev = "cleared" if self.selected is not None else None
            self.selected = None
            self.carry = None if self.phase in ("searching", "placed") else self.carry
            self.target_id = target_id
            self.closing_reselects = 0
            return self._finish(rec, False, f"phase {self.phase}", ev)
        # target identity
        if target_id is None:
            self._drop(rec, "no target")
            return self._finish(rec, False, "no target", "invalidated")
        if self.target_id is not None and int(target_id) != int(self.target_id):
            self._drop(rec, "target changed")
        self.target_id = int(target_id)
        if target_state == "lost":
            self._drop(rec, "target lost")
            return self._finish(rec, False, "target lost", "invalidated")
        self.observed_time = observed_time if observed_time is not None else self.observed_time
        self.confidence = confidence if confidence is not None else self.confidence
        cands = [] if candidates is None else list(candidates.candidates)
        surf = None if candidates is None else candidates.surface
        centre = None if surf is None else np.asarray(surf.centre, np.float64)
        closing = self.phase == "closing"
        g = self.selected
        if g is not None and closing:
            # while the fingers close the grasp is frozen: the hand occludes and pushes the target, so the newest
            # observation is not evidence against it (only an identity change, above, drops it)
            self.selected = g
            return self._finish(rec, True, "ok", "frozen while closing")
        if g is not None and centre is not None:
            delta = centre - g.centre
            if float(np.linalg.norm(delta)) > cfg.max_centre_jump:
                rec["centre_jump_mm"] = round(float(np.linalg.norm(delta)) * 1000, 1)
                self._drop(rec, "target centre jumped")
                g = None
            else:
                g = g.shifted(delta)
                rec["centre_shift_mm"] = round(float(np.linalg.norm(delta)) * 1000, 2)
        if g is not None:
            match = self._match(g, cands)
            if match is not None:
                g = self._adopt(match, keep_id=g.grasp_id, since=g.since_request, centre=g.centre)
                rec["event"] = "matched"
                rec["matched_index"] = match.index
            elif candidates is not None and candidates.scene is not None:
                ok, checks, why = pose_checks(g.T, g.T_pre, self.gripper.body_boxes(g.open_width), candidates.scene,
                                              candidates.shell, cfg,
                                              wrist=tuple(self.gripper.wrist) if cfg.wrist_check else (),
                                              field_fn=candidates.field_fn, gripper=self.gripper,
                                              open_width=g.open_width)
                rec["tracked_checks"] = checks
                if ok and ik_fn is not None:
                    q, info = ik_fn(g.T, g.q_grasp if g.q_grasp is not None else ik_seed)
                    rec["tracked_ik"] = info
                    if q is None:
                        ok, why = False, "no IK"
                    else:
                        g.q_grasp = q
                        obs = candidates.field_fn if candidates.field_fn is not None else candidates.scene_arm
                        if arm_fn is not None and cfg.arm_check and obs is not None:
                            a = arm_fn(q, obs)
                            rec["tracked_arm"] = a
                            if a["hits"]:
                                ok, why = False, "arm_collision"
                if not ok:
                    self._drop(rec, f"tracked grasp failed: {why}")
                    g = None
                else:
                    rec["event"] = "kept (unmatched)"
            else:
                rec["event"] = "kept (no new observation)"
        if g is not None and not closing and cands:
            # the tracked grasp's key against this request's policy pose, so it compares with this request's best
            g.key = preference_key(g.T, g.alignment, vla_pose, cfg, cfg.selection)
            best = cands[0]
            if g.key - best.key > cfg.switch_margin and self._match(g, [best]) is None:
                rec["switch_gain"] = round(float(g.key - best.key), 4)
                g = self._new(best, centre)
                self.n_switches += 1
                rec["event"] = "switched (better by margin)"
        if g is None:
            if closing and (self.closing_reselects >= cfg.closing_reselect_budget or not cands):
                # plan §3: a grasp lost after the geometric control started is re-selected once; then it stays lost
                self.selected = None
                return self._finish(rec, False, "grasp lost while closing" + (
                    " (re-selection used)" if self.closing_reselects else " (no candidate)"), rec["event"] or "lost while closing")
            if closing:
                self.closing_reselects += 1
                rec["closing_reselect"] = self.closing_reselects
            if not cands:
                self.selected = None
                why = (candidates.stats.get("reason") if candidates is not None else "candidates not computed")
                return self._finish(rec, False, f"no candidate: {why}", rec["event"] or "none")
            g = self._new(cands[0], centre)
            rec["event"] = "selected" if rec["event"] is None else rec["event"] + " -> selected"
        self.selected = g
        return self._finish(rec, True, "ok", rec["event"])

    # ------------------------------------------------------------------------------------------------------------
    def _match(self, g: SelectedGrasp, cands: list[GraspCandidate]) -> Optional[GraspCandidate]:
        best, best_d = None, float("inf")
        for c in cands:
            dp = float(np.linalg.norm(c.T[:3, 3] - g.T[:3, 3]))
            dr = grasp_rot_angle(g.T[:3, :3], c.T[:3, :3])
            if dp <= self.cfg.match_pos and dr <= math.radians(self.cfg.match_rot_deg):
                d = dp + self.cfg.vla_rot_weight * dr
                if d < best_d:
                    best, best_d = c, d
        return best

    def _adopt(self, c: GraspCandidate, *, keep_id: int, since: int, centre: Optional[np.ndarray]) -> SelectedGrasp:
        centre = np.asarray(c.contacts.mean(axis=0) if centre is None else centre, np.float64).copy()
        return SelectedGrasp(grasp_id=keep_id, T=c.T.copy(), T_pre=c.T_pre.copy(), open_width=c.open_width, width=c.width,
                          contacts=c.contacts.copy(), centre=centre, key=c.key, since_request=since,
                             last_matched_request=self._request, q_grasp=c.q_grasp, q_pre=c.q_pre,
                             alignment=c.alignment)

    def _new(self, c: GraspCandidate, centre: Optional[np.ndarray]) -> SelectedGrasp:
        gid = self._next_id
        self._next_id += 1
        return self._adopt(c, keep_id=gid, since=self._request, centre=centre)

    def _drop(self, rec: dict[str, Any], why: str) -> None:
        """Forget the tracked grasp (counted when there was one) and note why; the caller finishes the record."""
        if self.selected is not None:
            self.n_invalidations += 1
            rec.setdefault("invalidated", []).append(why)
            rec["event"] = "invalidated"
        self.selected = None
        self.reason = why

    def _finish(self, rec: dict[str, Any], valid: bool, reason: str, event: Optional[str]) -> dict[str, Any]:
        self.valid = bool(valid)
        self.reason = reason
        rec.update(valid=self.valid, reason=reason, event=event if event is not None else rec.get("event"))
        g = self.selected
        if g is not None:
            rec["grasp"] = {"grasp_id": g.grasp_id, "position": [round(float(v), 5) for v in g.T[:3, 3]],
                            "open_width_mm": round(g.open_width * 1000, 2), "width_mm": round(g.width * 1000, 2),
                            "since_request": g.since_request, "key": round(float(g.key), 5)}
        rec["n_switches"] = self.n_switches
        rec["n_invalidations"] = self.n_invalidations
        if self.history and self.history[-1].get("request") == rec["request"]:
            self.history[-1] = dict(rec, first_pass=self.history[-1])
        else:
            self.history.append(rec)
        return rec


__all__ = ["GraspTaskState", "SelectedGrasp", "ACTIVE_PHASES"]
