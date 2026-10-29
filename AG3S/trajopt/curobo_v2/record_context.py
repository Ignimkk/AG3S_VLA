"""T63 step 2 — the server context of one recorded chunk, as **our TO received it** (pure numpy).

B2 / B3 must see the same history our TO saw — not cuRobo's own previous output chained offline.
The record does not store the context arrays themselves, but it stores what they were built from:

| context key (`SafePolicy.infer`) | how the server built it | rebuilt here from |
|---|---|---|
| `previous_physical_chunk` + `previous_executed_steps` -> continuity tail (`refiner._continuity_reference`) | the cached plan (`_previous_plan[executed_chunk]`) of the feedback's seq, cut at `n_exec` | `summary.exec_continuity` (`used`, `previous_chunk`, `aligned_from_step`, `plan_seq`) + the **previous record** `chunk_{k-1}` (`refined_chunk` / `reference_chunk`, whose `t_step` must equal `plan_seq`) |
| `deflection_history` (`SafePolicy._ta_history`) | `applied_arm[-1] − previous reference[:, row]`, `row = len(executed) − 1` | `summary.exec_feedback.applied_arm` + the previous record's `reference_chunk`; checked against `summary.ta_history.deflection.max_abs_rad` |
| `command_history` (servo accel only) | `applied_arm[-2:].T` | `summary.exec_feedback.applied_arm` |

The previous plan in the server was float64; the record keeps float32 (difference ~1e-7 rad, the
deflection check above measures it). `applied_arm` columns are the TO's optimized-joint order,
which is the c-space order here (checked: the executed rows equal the previous refined chunk's).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Optional

import numpy as np

ARM_COLUMNS = [*range(0, 7), *range(8, 15)]


def _arm(chunk: np.ndarray) -> np.ndarray:
    return np.asarray(chunk, np.float64)[:, ARM_COLUMNS]


def previous_record_path(chunk_file: str | pathlib.Path, chunk_index: int) -> pathlib.Path:
    return pathlib.Path(chunk_file).with_name(f"chunk_{int(chunk_index) - 1:05d}.npz")


def chunk_record_context(chunk_file: str | pathlib.Path) -> dict[str, Any]:
    """`{continuity_tail (L,14) | None, deflection_history (14,) | None, command_history (14,2) |
    None, notes}` for one record. Every `None` carries its reason in `notes`."""
    path = pathlib.Path(chunk_file)
    with np.load(path, allow_pickle=False) as z:
        summary = json.loads(str(z["summary_json"]))
        k = int(z["chunk_index"])
    ec = summary.get("exec_continuity") or {}
    fb = summary.get("exec_feedback") or {}
    ta = summary.get("ta_history") or {}
    out: dict[str, Any] = {"chunk_index": k, "continuity_tail": None, "deflection_history": None,
                           "command_history": None, "notes": {}}
    prev_path = previous_record_path(path, k)
    prev = None
    if prev_path.exists():
        with np.load(prev_path, allow_pickle=False) as pz:
            prev = {"t_step": int(pz["t_step"]),
                    "refined_chunk": np.asarray(pz["refined_chunk"], np.float64),
                    "reference_chunk": np.asarray(pz["reference_chunk"], np.float64)}

    def prev_ok(seq) -> bool:
        return prev is not None and seq is not None and prev["t_step"] == int(seq)

    # ---- continuity (refiner._continuity_reference, with the SafePolicy feedback alignment) ----
    if not ec.get("used"):
        out["notes"]["continuity"] = (f"not used by the TO: mode={ec.get('mode')!r}, "
                                      f"source={ec.get('source')!r}, off_by={ec.get('off_by')!r}")
    elif not prev_ok(ec.get("plan_seq")):
        out["notes"]["continuity"] = (f"previous record {prev_path.name} missing or its t_step != "
                                      f"plan_seq {ec.get('plan_seq')}")
    else:
        which = str(ec.get("previous_chunk"))
        start = int(ec.get("aligned_from_step"))
        tail = _arm(prev[f"{which}_chunk"])[start:]
        out["continuity_tail"] = tail if len(tail) else None
        out["notes"]["continuity"] = f"{which} chunk of t_step {prev['t_step']} from step {start}"

    arm = None
    if fb.get("available") and fb.get("applied_arm") is not None:
        arm = np.asarray(fb["applied_arm"], np.float64)

    # ---- deflection history (SafePolicy._ta_history) ----
    defl = ta.get("deflection")
    if defl is None:
        out["notes"]["deflection"] = f"no history recorded: {ta.get('deflection_note')!r}"
    elif arm is None or not prev_ok(fb.get("seq")):
        out["notes"]["deflection"] = "recorded, but applied_arm / previous record unavailable"
    else:
        row = int(defl["row"])
        executed = list(fb.get("executed") or [])
        if row != len(executed) - 1:
            raise ValueError(f"{path.name}: ta_history row {row} != len(executed)-1")
        d = arm[-1] - _arm(prev["reference_chunk"])[row]
        err = abs(float(np.abs(d).max()) - float(defl["max_abs_rad"]))
        out["deflection_history"] = d
        out["deflection_check_abs_err_rad"] = err
        out["notes"]["deflection"] = f"applied_arm[-1] - reference(t_step {prev['t_step']})[{row}]"

    # ---- command history (servo accel; not active in the baseline) ----
    if arm is not None and arm.shape[0] >= 2:
        out["command_history"] = np.ascontiguousarray(arm[-2:].T)
    return out


__all__ = ["ARM_COLUMNS", "chunk_record_context", "previous_record_path"]
