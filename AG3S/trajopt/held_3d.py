"""IFG-S2SERVE — the **training-free 3D held rule** in the grasp latch (`serve_safe --held-3d-rule on`, default off).

The rule was fixed in `docs/handoff/IFG-S2FK.task.md` before results (`experiments/studies/ifg_stage2_fk.py`,
`rules` / `_episode_features`, constants imported from there):

    held_3d = grip < GRIP_CLOSED (0.9)  and  (no target  or  hand_d ≤ D_HOLD (0.0813 + 0.05 = 0.1313 m))

| quantity | offline (`ifg_stage2_fk._episode_features`) | here |
|---|---|---|
| grip | ``min(raw_state[7], raw_state[15])`` (the 16D state's two gripper columns) | the same columns of this request's policy state (`SafePolicy.gripper_columns`) |
| hand proxy | per hand, the mean world position of every MJCF body **below** ``link_{left,right}_arm_6`` (`t47_replay.GRIPPER_ROOT_LINKS`), expressed in the ``base`` body frame, at the recorded ``qpos`` | the same bodies of the server's `--model-xml`, posed from the request's 20-joint `robot_state` + the measured finger openings (the pose of `runtime.robot_render_mask.RobotRenderMask.pose` — used for FK only, nothing is rendered and no mask is made) |
| target centre | this frame's attention-surface region centroid (`r6_choice`) | the server's grounded target centroid of the **previous** request (`SafePolicy._held_3d_target_centre`: the attach decision runs before this request's grounding; lead decision 2026-10-19, one request ≈ 8 control steps late). No target → gripper condition only (offline: no region → ``hand_d = nan`` → gripper only) |

`SafePolicy._advance_grasp` ANDs ``held_3d`` into the CLOSING → HELD checks (`GraspLatch.observe_grasp(extra_checks=…)`);
revoke / release / slip are unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

import numpy as np


def _constants():
    from AG3S.ag3s.experiments.studies.ifg_stage2_fk import D_HOLD, GRIP_CLOSED

    return float(D_HOLD), float(GRIP_CLOSED)


#: `ifg_stage2_fk.D_HOLD` / `GRIP_CLOSED` (fixed in IFG-S2FK.task.md before results).
HELD_3D_MAX_M, HELD_3D_GRIP_CLOSED = _constants()
HELD_3D_MODES = ("off", "on")


def held_3d_rule(grip: Optional[float], hand_d: Optional[float], *, max_m: float = HELD_3D_MAX_M,
                 grip_closed: float = HELD_3D_GRIP_CLOSED) -> tuple[bool, bool]:
    """`(held_gripper, held_3d)` — `ifg_stage2_fk.rules`' per-frame line. `hand_d` None / nan = no target (gripper
    condition only). An unknown `grip` is not closed (fail-closed: no attach)."""
    held_gripper = grip is not None and np.isfinite(float(grip)) and float(grip) < float(grip_closed)
    no_target = hand_d is None or not np.isfinite(float(hand_d))
    return bool(held_gripper), bool(held_gripper and (no_target or float(hand_d) <= float(max_m)))


class HandProxyFK:
    """Hand proxies (base frame) from the server's MJCF, posed like `RobotRenderMask.pose` (FK only)."""

    def __init__(self, model_xml: str):
        from AG3S.ag3s.experiments.studies.t47_replay import GRIPPER_ROOT_LINKS
        from AG3S.ag3s.runtime.robot_render_mask import RobotRenderMask

        # The posing code (request joints → qpos, client finger convention, joint equalities) is the IFG-CL1 one;
        # the object's render / mask methods are never called here.
        self._poser = RobotRenderMask(model_xml)
        m = self._poser.model
        mj = self._poser.mujoco
        self.model_xml = self._poser.model_xml
        self.roots = tuple(GRIPPER_ROOT_LINKS)
        roots = {n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in self.roots}
        missing = [n for n, r in roots.items() if r < 0]
        if missing:
            raise ValueError(f"{self.model_xml}: no body {missing} (hand proxy roots)")
        # `ifg_stage2_fk._episode_features`: every body whose ancestor chain contains the root (the root excluded)
        below = {n: [] for n in roots}
        for b in range(m.nbody):
            j = b
            while j > 0:
                j = int(m.body_parentid[j])
                for n, r in roots.items():
                    if j == r:
                        below[n].append(b)
        empty = [n for n, v in below.items() if not v]
        if empty:
            raise ValueError(f"{self.model_xml}: no body below {empty}")
        self._below = {n: np.asarray(v, np.int64) for n, v in below.items()}
        self._base = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "base")

    def record(self) -> dict[str, Any]:
        return {"model_xml": self.model_xml, "roots": list(self.roots),
                "n_bodies": {n: int(len(v)) for n, v in self._below.items()},
                "finger_convention": self._poser.record().get("finger_convention")}

    def proxies(self, robot_state, openings: Optional[Mapping[str, Optional[float]]] = None) -> dict[str, np.ndarray]:
        """`{root link: (3,) base-frame point}` at `robot_state` (20,) with `{hand: normalised opening}`."""
        self._poser.pose(robot_state, openings)
        d = self._poser.data
        Rb, tb = d.xmat[self._base].reshape(3, 3), d.xpos[self._base]
        return {n: Rb.T @ (d.xpos[v].mean(0) - tb) for n, v in self._below.items()}


@dataclasses.dataclass
class Held3DRule:
    fk: HandProxyFK
    max_m: float = HELD_3D_MAX_M
    grip_closed: float = HELD_3D_GRIP_CLOSED

    def record_meta(self) -> dict[str, Any]:
        return {"max_m": float(self.max_m), "grip_closed": float(self.grip_closed),
                "target_centre": "previous request's grounded target centroid (no target -> gripper only)",
                **self.fk.record()}

    def evaluate(self, *, state16, gripper_columns, robot_state, openings, target_centre) -> dict[str, Any]:
        """The per-request record: `grip`, `hand_d` (m, None = no target), `held_gripper`, `held_3d`."""
        rec: dict[str, Any] = {"grip": None, "hand_d": None, "held_gripper": False, "held_3d": False,
                               "target_centre": None, "proxies": None, "why": None}
        try:
            s = np.asarray(state16, np.float64).reshape(-1)
            rec["grip"] = float(min(s[int(gripper_columns[0])], s[int(gripper_columns[1])]))
        except Exception as exc:  # noqa: BLE001 — no grip → not closed (fail-closed)
            rec["why"] = f"no_grip ({type(exc).__name__}: {exc})"
        hand_d = None
        if target_centre is not None:
            c = np.asarray(target_centre, np.float64).reshape(3)
            rec["target_centre"] = [round(float(v), 5) for v in c]
            try:
                px = self.fk.proxies(robot_state, openings)
                rec["proxies"] = {n: [round(float(v), 5) for v in p] for n, p in px.items()}
                hand_d = float(min(np.linalg.norm(c - p) for p in px.values()))
            except Exception as exc:  # noqa: BLE001 — FK failed: the distance is unknown → not held (fail-closed)
                rec["why"] = f"fk_failed ({type(exc).__name__}: {exc})"
                rec["hand_d"] = None
                rec["held_gripper"] = held_3d_rule(rec["grip"], None, max_m=self.max_m,
                                                   grip_closed=self.grip_closed)[0]
                return rec
        rec["hand_d"] = hand_d
        if target_centre is None:
            rec["why"] = rec["why"] or "no_target (gripper only)"
        rec["held_gripper"], rec["held_3d"] = held_3d_rule(rec["grip"], hand_d, max_m=self.max_m,
                                                          grip_closed=self.grip_closed)
        return rec


__all__ = ["HELD_3D_GRIP_CLOSED", "HELD_3D_MAX_M", "HELD_3D_MODES", "Held3DRule", "HandProxyFK", "held_3d_rule"]
