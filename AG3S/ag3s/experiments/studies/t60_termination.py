"""T60 — task termination predicates P (placement complete) and H (ready-pose return complete).

`doc/reviews/TERMINATION_SCOPE_AUDIT_20261013_KO.md` 의 "두 종료 규칙", as code. Pure numpy; the loader
reads the existing closed-loop records (CL2 / CL3 / T53) without changing them.

Row convention (one control row = one 15 Hz policy step, the replay of `replay_contacts.py`)
    ``state[k]``      replay state at the START of control row k (k = 0 … N; ``state[N]`` = final state).
    ``contact[k]``    contact sub-steps DURING row k.
    ``speed[k]``      ``|target_pos[k+1] − target_pos[k]| · control_hz`` (mean target speed over row k).
    A predicate value ``ok[k]`` uses ``state[k]``, ``contact[k]`` and ``speed[k]``; it is known at the END of
    row k. A completion time ``t_end`` is therefore the last control row whose action and contacts belong to
    the task; an inference / observation with ``t_step > t_end`` is post-completion.

Rule P — placement complete (stable)
    ``ok_P[k] = in_crate(state[k]) ∧ no hand–target contact in row k ∧ speed[k] ≤ speed_tol``.
    ``t_P`` = the row at which ``ok_P`` has held for ``dwell_p`` consecutive rows for the first time
    (causal: the dwell is part of the task). ``in_crate`` = the `analyze_t42` / `transport.py:object_in_crate`
    centre test (crate-local |x| < 0.077, |y| < 0.137, −0.052 < z < 0.075). ``speed_tol`` default = the
    expert collector's ``TARGET_LINEAR_SPEED_TOLERANCE_MPS`` (0.030 m/s).

Rule H — ready-pose return complete (after P)
    ``ok_H[k] = q_err[k] ≤ q_tol ∧ tcp_err[k] ≤ tcp_tol`` for ``k > t_P``, where ``q_err`` = max |q − q_ready|
    over the used arm's 7 joints and ``tcp_err`` = |TCP − TCP_ready|. ``q_ready`` / ``TCP_ready`` = the
    episode's own initial pose (row 0 after warm-up; the expert's ``rest_ctrl`` is the same episode-noised
    ready pose). ``t_H`` = the row at which ``ok_H`` has held for ``dwell_h`` rows for the first time.
    H never ends before P, so a run whose start pose equals the ready pose does not end at t = 0.

A run without P (or without H) keeps its full horizon. The same predicate is used for every arm.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

PROTOCOL_VERSION = "T60-v1"
CRATE_HALF_XY = np.array([0.082, 0.142])
CRATE_MARGIN = 0.005
CRATE_Z = (-0.052, 0.075)
EXPERT_SPEED_TOL = 0.030  # transport_atomic.TARGET_LINEAR_SPEED_TOLERANCE_MPS
CONTROL_HZ = 15
CHUNK = 8
ARM_SLICE = {"left": slice(0, 7), "right": slice(8, 15)}
DISALLOWED = ("hand-fruit_other", "robot_other-target", "robot_other-fruit_other")
FRUIT_DISALLOWED = ("hand-fruit_other", "robot_other-fruit_other")


@dataclass(frozen=True)
class Params:
    """Predicate parameters. Defaults = the proposal shown to the user before any arm-level result."""

    dwell_p: int = 8          # rows (= one 8-row chunk, 0.53 s)
    speed_tol: float = EXPERT_SPEED_TOL
    q_tol: float = 0.35       # rad, max over the used arm's 7 joints
    tcp_tol: float = 0.12     # m
    dwell_h: int = 8          # rows


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def in_crate(target_pos: np.ndarray, crate_pos: np.ndarray, crate_quat: np.ndarray) -> np.ndarray:
    """Per-state centre test, same rule as `analyze_t42.in_crate` (vectorised)."""
    target_pos = np.atleast_2d(target_pos); crate_pos = np.atleast_2d(crate_pos)
    crate_quat = np.atleast_2d(crate_quat)
    local = np.stack([quat_to_R(q).T @ (p - c) for p, c, q in zip(target_pos, crate_pos, crate_quat)])
    lim = CRATE_HALF_XY - CRATE_MARGIN
    return ((np.abs(local[:, 0]) < lim[0]) & (np.abs(local[:, 1]) < lim[1])
            & (local[:, 2] > CRATE_Z[0]) & (local[:, 2] < CRATE_Z[1]))


def first_dwell(ok: np.ndarray, dwell: int, start: int = 0) -> Optional[int]:
    """Row at which ``ok`` has been true for ``dwell`` consecutive rows (rows ≥ start) for the first time."""
    if dwell < 1:
        raise ValueError("dwell must be >= 1")
    run = 0
    for k in range(start, len(ok)):
        run = run + 1 if ok[k] else 0
        if run >= dwell:
            return k
    return None


def rule_p_ok(inc_state: np.ndarray, hand_target_contact: np.ndarray, speed: np.ndarray,
              speed_tol: float) -> np.ndarray:
    n = len(hand_target_contact)
    return np.asarray(inc_state[:n], bool) & ~np.asarray(hand_target_contact, bool) & (np.asarray(speed[:n]) <= speed_tol)


def rule_h_ok(q_err: np.ndarray, tcp_err: np.ndarray, q_tol: float, tcp_tol: float) -> np.ndarray:
    n = min(len(q_err), len(tcp_err))
    return (np.asarray(q_err[:n]) <= q_tol) & (np.asarray(tcp_err[:n]) <= tcp_tol)


def endpoints(series: dict, p: Params) -> dict:
    """``t_P`` / ``t_H`` (control rows, or None) from per-row series (see `series_from_record`)."""
    ok_p = rule_p_ok(series["in_crate"], series["hand_target"], series["speed"], p.speed_tol)
    t_p = first_dwell(ok_p, p.dwell_p)
    t_h = None
    if t_p is not None:
        ok_h = rule_h_ok(series["q_err"], series["tcp_err"], p.q_tol, p.tcp_tol)
        t_h = first_dwell(ok_h, p.dwell_h, start=t_p + 1)
    return {"t_P": t_p, "t_H": t_h}


def endpoints_sampled(t_step: np.ndarray, in_crate_s: np.ndarray, hand_target_s: np.ndarray,
                      speed_s: np.ndarray, q_err_s: np.ndarray, tcp_err_s: np.ndarray, p: Params,
                      chunk: int = CHUNK) -> dict:
    """Frame-resolution P / H for step records that hold only the inference-time state (every `chunk`
    rows). Two consecutive samples f, f+1 that both satisfy the predicate stand for the `chunk` rows
    between them, so ``t̂ = t_step[f] + chunk − 1`` (the row whose end state is the sample f+1).
    Only for ``dwell_p == dwell_h == chunk``; validated against the row-level `endpoints` on the
    closed-loop records (`t60_sampled_check`)."""
    if p.dwell_p != chunk or p.dwell_h != chunk:
        raise ValueError("sampled endpoints assume dwell == one chunk")
    t_step = np.asarray(t_step)
    okp = (np.asarray(in_crate_s, bool) & ~np.asarray(hand_target_s, bool)
           & (np.asarray(speed_s) <= p.speed_tol))
    okh = (np.asarray(q_err_s) <= p.q_tol) & (np.asarray(tcp_err_s) <= p.tcp_tol)
    t_p = t_h = None
    for f in range(len(t_step) - 1):
        if okp[f] and okp[f + 1] and t_step[f + 1] - t_step[f] == chunk:
            t_p = int(t_step[f]) + chunk - 1
            break
    if t_p is not None:
        for f in range(len(t_step) - 1):
            if t_step[f] > t_p and okh[f] and okh[f + 1] and t_step[f + 1] - t_step[f] == chunk:
                t_h = int(t_step[f]) + chunk - 1
                break
    return {"t_P": t_p, "t_H": t_h}


def keep_rows(n_rows: int, t_end: Optional[int]) -> np.ndarray:
    """Control rows that belong to the task (rows ≤ t_end; all rows when not completed)."""
    keep = np.ones(n_rows, bool)
    if t_end is not None:
        keep[t_end + 1:] = False
    return keep


def keep_frames(chunk_start_steps: np.ndarray, t_end: Optional[int]) -> np.ndarray:
    """Inference frames whose observation is not after completion (t_step ≤ t_end)."""
    s = np.asarray(chunk_start_steps)
    return np.ones(len(s), bool) if t_end is None else s <= t_end


# ---------------------------------------------------------------- record loader (read-only)

def series_from_record(replay_npz: pathlib.Path, traj_npz: pathlib.Path, used_arm: str) -> dict:
    rp = np.load(replay_npz, allow_pickle=True)
    tj = np.load(traj_npz, allow_pickle=True)
    hz = int(tj["control_hz"]) if "control_hz" in tj.files else CONTROL_HZ
    tp = rp["target_pos"]
    q = tj["measured_qpos"]
    sl = ARM_SLICE[used_arm]
    tcp = rp["tcp"]
    n = len(rp["t"])
    out = {
        "t": rp["t"], "seq": rp["seq"], "executed": rp["executed"], "hold": rp["hold"],
        "in_crate": in_crate(tp, rp["crate_pos"], rp["crate_quat"]),
        "hand_target": rp["cnt_hand-target"] > 0,
        "speed": np.linalg.norm(np.diff(tp, axis=0), axis=1)[:n] * hz,
        "q_err": np.abs(q[:, sl] - q[0, sl]).max(axis=1)[:n],
        "tcp_err": np.linalg.norm(tcp - tcp[0], axis=1)[:n],
        "chunk_start_steps": tj["chunk_start_steps"],
        "control_hz": hz,
    }
    for cls in DISALLOWED + ("hand-target",):
        out["cnt_" + cls] = rp["cnt_" + cls]
    return out


def write_json(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_default) + "\n")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Params):
        return asdict(o)
    raise TypeError(type(o))
