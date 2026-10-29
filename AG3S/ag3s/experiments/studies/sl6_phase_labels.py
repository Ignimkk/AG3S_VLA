"""SL6 B3 — Panda physical phase labels of LIBERO policy requests (the RB-Y1 T64 Part B truth, ported).

Rule (``AG3S/ag3s/docs/handoff/SL6.task.md`` §3 B3, fixed by the lead before results; RB-Y1 source
`t64_partB.truth_phases`, thresholds 2 cm / 5 cm unchanged)::

    grasp      from the start of a gripper close command (LIBERO action[-1] > 0) whose close run reaches two-finger
               contact with the target, around that contact, until lift
    lift       both fingers touch the target ∧ (target rose ≥ 2 cm above z0 ∨ the target lost every contact with a
               non-robot geom — libero_spatial t3 / t5 start the bowl on the cookie box / ramekin)
    transport  lifted ∧ target moved ≥ 5 cm horizontally from the lift point
    placement  target xy inside the destination's xy box (world AABB of its collision geoms), once lifted
    approach   everything else (before the grasp, failed closes, drops, retries)
    retreat    none (the episode ends at the success instant)

Operational reading (this module; the table above is the rule):

* frames are the recorded observations ``0 … N-1`` (``phase_export --finger-contacts`` track, image-instant states);
  the label of a request is the label of its frame ``obs_index = chunk_t[k]``;
* ``close(f)`` = the action that produced observation ``f`` closes: ``action[f - 1, -1] > 0`` (``f ≥ 1``); a *close
  run* is a maximal run of such frames (wait steps send −1 = open);
* ``two(f)`` = left ∧ right finger–target contact (`phase_export.FINGER_BODIES`);
  ``z0`` = target z at the first request frame (the bowl settles during the wait steps, `libero_phases`);
* ``lifted(f)`` = two(f) ∧ (z(f) − z0 ≥ 2 cm ∨ n_contact_target_env(f) == 0);
* the lift point = target xy at the first lifted frame since the start of the latest close run;
* grasp window of a close run that has a two-finger frame: from its first frame up to its first lifted frame, plus
  any later two-finger, not-lifted frame of the same run (held on the support again — T64: held ∧ rise < 2 cm);
  a close run without a two-finger frame is approach (a failed close);
* placement wins over lift / transport (T64: over the crate, held or released), but only after the first lift;
* after a lift, a frame that is neither two-finger nor in the destination box is approach (drop / failure, T64
  ``failure`` folded into approach as the task says) — counted as ``lost_hold_frames``.

Pure numpy; no MuJoCo (the contacts come from the export).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

#: Panda class order = `AG3S.trajopt.hsr6.PHASES6` without retreat (an ordered subset, SL6 B2).
PHASES5 = ("approach", "grasp", "lift", "transport", "placement")
RISE_M = 0.02
TRAVEL_M = 0.05
RULE_VERSION = "sl6-b3-1"
DEFINITION = __doc__


def close_frames(actions: np.ndarray, n_frames: int) -> np.ndarray:
    """``close[f]`` = action ``f - 1`` closes the gripper (``action[:, -1] > 0``); frame 0 has no action."""
    a = np.asarray(actions, np.float64)
    out = np.zeros(int(n_frames), bool)
    k = min(len(a), int(n_frames) - 1)
    out[1:k + 1] = a[:k, -1] > 0
    return out


def runs_of(mask: np.ndarray) -> np.ndarray:
    """Run id per frame (−1 outside a run, 0, 1, … in order)."""
    m = np.asarray(mask, bool)
    rid = np.full(len(m), -1, np.int64)
    r = -1
    for i in range(len(m)):
        if m[i]:
            if i == 0 or not m[i - 1]:
                r += 1
            rid[i] = r
    return rid


def in_box(xy: np.ndarray, box: np.ndarray) -> np.ndarray:
    """``xy (N, 2)`` strictly inside ``box (N, 4)`` ``[xmin, ymin, xmax, ymax]``."""
    xy = np.asarray(xy, np.float64)
    b = np.asarray(box, np.float64)
    return (xy[:, 0] > b[:, 0]) & (xy[:, 0] < b[:, 2]) & (xy[:, 1] > b[:, 1]) & (xy[:, 1] < b[:, 3])


def frame_phases(track: Mapping[str, np.ndarray], actions: np.ndarray, call_frames: Sequence[int]
                 ) -> tuple[np.ndarray, dict[str, Any]]:
    """Per recorded frame: class index into `PHASES5`, and the episode's event info.

    ``track``: ``phase_export --finger-contacts`` arrays (``frames``, ``target_xpos``, ``contact_left_finger_target``,
    ``contact_right_finger_target``, ``n_contact_target_env``, ``destination_xy_box``); ``actions`` the episode's
    executed ``action (T, 7)``; ``call_frames`` its request frames (``chunk_t``)."""
    frames = np.asarray(track["frames"], np.int64).reshape(-1)
    n = len(frames)
    if not np.array_equal(frames, np.arange(n)):
        raise ValueError("track frames must be 0 … N-1 (every recorded observation)")
    pos = np.asarray(track["target_xpos"], np.float64)
    two = (np.asarray(track["contact_left_finger_target"], bool)
           & np.asarray(track["contact_right_finger_target"], bool))
    n_env = np.asarray(track["n_contact_target_env"], np.int64)
    dest = in_box(pos[:, :2], track["destination_xy_box"])
    calls = sorted(int(c) for c in call_frames)
    z0 = float(pos[calls[0], 2])
    lifted = two & ((pos[:, 2] - z0 >= RISE_M) | (n_env == 0))
    close = close_frames(actions, n)
    rid = runs_of(close)
    n_runs = int(rid.max()) + 1
    run_two = np.array([bool(two[rid == r].any()) for r in range(n_runs)], bool)

    lab = np.zeros(n, np.int64)
    ever_lift = False
    lift_xy: Optional[np.ndarray] = None
    run_lifted = np.zeros(max(n_runs, 1), bool)
    cur_run = -1
    lost = 0
    events = {"first_close": None, "first_two": None, "first_lift": None, "first_transport": None,
              "first_placement": None}
    for f in range(n):
        if rid[f] >= 0 and rid[f] != cur_run:          # a new close run starts: a new lift point
            cur_run = int(rid[f])
            lift_xy = None
        if lifted[f]:
            if lift_xy is None:
                lift_xy = pos[f, :2].copy()
            ever_lift = True
            if cur_run >= 0:
                run_lifted[cur_run] = True
        in_run = rid[f] >= 0 and run_two[rid[f]]
        if ever_lift and dest[f]:
            lab[f] = 4
        elif lifted[f]:
            lab[f] = 2 if float(np.linalg.norm(pos[f, :2] - lift_xy)) < TRAVEL_M else 3
        elif in_run and (not run_lifted[rid[f]] or two[f]):
            lab[f] = 1
        else:
            lab[f] = 0
            if ever_lift:
                lost += 1
        if close[f] and events["first_close"] is None:
            events["first_close"] = f
        if two[f] and events["first_two"] is None:
            events["first_two"] = f
        for c, key in ((2, "first_lift"), (3, "first_transport"), (4, "first_placement")):
            if lab[f] == c and events[key] is None:
                events[key] = f
    info = {"z0": z0, "close_runs": n_runs, "close_runs_with_two_finger": int(run_two.sum()),
            "lost_hold_frames": int(lost), "frames": n, **events,
            "rise_max_m": float(pos[:, 2].max() - z0)}
    return lab, info


def request_phases(track: Mapping[str, np.ndarray], actions: np.ndarray, chunk_t: Sequence[int]
                   ) -> tuple[np.ndarray, dict[str, Any]]:
    """Class index per request (frame ``chunk_t[k]``) and the episode info (+ ``frame_labels``)."""
    lab, info = frame_phases(track, actions, chunk_t)
    ct = np.asarray(chunk_t, np.int64).reshape(-1)
    info["frame_labels"] = lab
    return lab[ct], info


__all__ = ["DEFINITION", "PHASES5", "RISE_M", "RULE_VERSION", "TRAVEL_M", "close_frames", "frame_phases", "in_box",
           "request_phases", "runs_of"]
