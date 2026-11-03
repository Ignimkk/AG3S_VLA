"""T91c — the T91b task-phase recogniser as the source of the 3-stage gate label (`serve_safe --subtask-from-phase-machine`).

`PhaseMachine` (T91 / T91b, `phase_machine.py`) reads physical evidence only: the grasping hand's measured opening, the hand
site, the ATS target / destination centroids, the VLA chunk's gripper column and — T91b — how much of the region between the
fingertips of the wrist camera shows a surface within finger reach (`finger_gap.surface_fraction` of the wrist **depth**).
This module is the thin adapter between one serving request and that machine, shared by the server (`SafePolicy`) and the
offline replay (`experiments/studies/t91c_gate_replay.py`) so that both build the observation the same way:

  request input (server `ag3s/...` scene keys)            -> PhaseObservation field
  state[7 | 15] of the grasping hand (`_measured_openings`)  gripper_raw
  hold link site `<side>_ee` at the newest `robot_state`     ee_pos            (`safe_policy.hold_site_position`)
  `ag3s.manipulated` (centroid, state == "visible")          target_pos / target_fresh   -- the **previous request's** value
  `ag3s.destination.centroid`                                destination_pos             -- the previous request's value
  VLA chunk gripper column (`result["actions"]`)             chunk_gripper
  `ag3s/depth/wrist_cam_<l|r>` (uint16 mm, stride 4)         finger_gap = surface fraction in the RBY1 wrist ROI

The target / destination are the previous request's because the gate label is handed to AG3S **before** this request's
grounding (`SafePolicy._scene_fn`); a request without them (first request, after reset) gives `None`, which the machine's
missing-input rules handle. A missing wrist depth gives `finger_gap = None` -> the T91 rules (co-motion / occlusion) decide.

The 5 stages become the 3-stage one-hot the gate has always been given (`hsr5.subtask3_probabilities`, the T82 table):
approach · grasp -> pick, carry · place -> place, retreat -> home. AG3S' `SubtaskDebounce` is unchanged.

Parameters: T91 thresholds as chosen on train scenes (`PhaseConfig` defaults == `verify/T91/thresholds.json`, tested) and the
T91b depth arm (`finger_gap_source="depth"`, `gap_tau=0.3`, `gap_m=1`, ROI / depth window `finger_gap.rby1_wrist()`, tested
against `verify/T91b/t91b_eval.json` `gap_arms.depth`). Nothing is fitted here.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from AG3S.trajopt.finger_gap import rby1_wrist, surface_fraction
from AG3S.trajopt.hsr5 import subtask3_probabilities
from AG3S.trajopt.phase_machine import PhaseConfig, PhaseMachine, PhaseObservation

#: `ag3s/depth/<camera>` name of each hand's wrist camera (`wire.DEFAULT_CAMERAS`).
WRIST_CAMERA = {"left": "wrist_cam_l", "right": "wrist_cam_r"}
#: the wrist depth is read at stride 4 (480 x 640 -> 120 x 160): the resolution T91b chose its ROI / thresholds on.
WRIST_STRIDE = 4
#: T91b depth arm (`t91b_eval.json` `gap_arms.depth`): the finger-gap settings on top of the T91 defaults.
GAP_SETTINGS = {"finger_gap_mode": "on", "finger_gap_source": "depth", "gap_tau": 0.3, "gap_m": 1}
#: the evidence keys of one request kept in the record (`summary_json.phase_machine.evidence`).
EVIDENCE_KEYS = ("open_norm", "rate", "closed", "close_intent", "open_intent", "d_target_mm", "d_dest_mm", "near", "fresh",
                 "comotion", "blocked", "moved_since_close_mm", "gap_depth", "gap_hit", "gap_ok", "gap_used")


def phase_config(side: str, gripper_column: int) -> PhaseConfig:
    """T91 + T91b depth-arm settings for the grasping hand `side`; `gripper_column` = its column in the 16-D chunk."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return PhaseConfig(chunk_gripper_column=int(gripper_column), **GAP_SETTINGS)


def finger_gap_from_depth_mm(depth_mm: Any) -> Optional[float]:
    """Surface fraction between the fingertips from a wrist depth image in uint16 millimetres (None = no usable image)."""
    if depth_mm is None:
        return None
    d = np.asarray(depth_mm)
    if d.ndim != 2 or d.size == 0:
        return None
    return surface_fraction(d[::WRIST_STRIDE, ::WRIST_STRIDE].astype(np.float64) / 1000.0, rby1_wrist())


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


class PhaseGate:
    """One run's gate-label source: a `PhaseMachine` stepped once per request. `reset()` between runs."""

    def __init__(self, gripper_columns: Sequence[int] = (7, 15)) -> None:
        self.columns = {"left": int(gripper_columns[0]), "right": int(gripper_columns[1])}
        self.reset()

    def reset(self) -> None:
        self.machine: Optional[PhaseMachine] = None
        self.hand: Optional[str] = None
        self.n = 0

    def step(self, *, hand: str, gripper_raw: Any, ee_pos: Any = None, target_pos: Any = None, target_fresh: bool = False,
             destination_pos: Any = None, chunk_gripper: Any = None, wrist_depth_mm: Any = None
             ) -> tuple[Optional[dict[str, float]], dict[str, Any]]:
        """`(3-stage one-hot or None, record)` for this request. No usable gripper opening = no stage this request
        (`None` label, the gate then reads "no label" exactly as for a probe that did not answer); the machine is not stepped."""
        hand = str(hand)
        if hand not in self.columns:
            raise ValueError(f"unknown hand {hand!r}")
        if self.machine is None or hand != self.hand:
            self.machine, self.hand = PhaseMachine(phase_config(hand, self.columns[hand])), hand
        raw = _finite(gripper_raw)
        gap = finger_gap_from_depth_mm(wrist_depth_mm)
        record: dict[str, Any] = {"source": "phase_machine", "hand": hand, "n": self.n,
                                  "inputs": {"ee": ee_pos is not None, "target": target_pos is not None,
                                             "destination": destination_pos is not None,
                                             "chunk": chunk_gripper is not None, "wrist_depth": gap is not None}}
        self.n += 1
        if raw is None:
            record.update(stage=None, label=None, reason="no gripper opening in the request")
            return None, record
        cg = None if chunk_gripper is None else np.asarray(chunk_gripper, np.float64).reshape(-1)
        stage, ev = self.machine.step(PhaseObservation(
            gripper_raw=raw, ee_pos=ee_pos, target_pos=target_pos, target_fresh=bool(target_fresh),
            destination_pos=destination_pos, chunk_gripper=cg, finger_gap=gap))
        p = subtask3_probabilities(stage)
        record.update(stage=stage, label=None if p is None else max(p, key=p.get),
                      evidence={k: ev[k] for k in EVIDENCE_KEYS if k in ev})
        return p, record
