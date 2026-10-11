"""T43 Z3 — replay a recorded T43 W server run through `GraspLatch(evidence=True)` (helper, not a test).

The latch is fed what the live latch was fed, read from the run's server chunk records in request
order (the analyze_t43w pairing rule — `analysis.json` `server_segment`):

* execution feedback → `grasp_signal_from_feedback(summary.exec_feedback, hand, 0.85)`
* reach · manipulated id/state · placed → that request's `summary.grasp.evidence` (what the live latch saw)
* `held_observation` → None. In these runs the manipulated object is never observed on a request that
  ends held (`summary.manipulated.state` is `occluded` there — `observed_while_held` counts the
  exceptions), so the live `AG3S.held_observation()` returned None as well.
* an attach the live pipeline refused (`summary.grasp.rejected`) → `revert_attach`.

The target is locked before the first request with the same name 3 times (as the T34 replay): attach
decisions read only closing-time evidence, so the replayed states must equal the recorded ones —
`replay(..., slip_detach=False)["mismatch"]` checks that.

Planning-row time `t` of a request = `frames.jsonl` planning row `t_step` with `seq == summary.t_step`.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
from typing import Optional

import numpy as np

W = pathlib.Path(_ag3s_workspace_path('experiments/legacy/outputs/verify/T43/W'))
ANALYSIS = W / "analysis/analysis.json"
Y_METRICS = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify/T43/Y/raw/y_metrics.json'))

_CACHE: dict = {}


def available() -> bool:
    return ANALYSIS.exists() and Y_METRICS.exists()


def analysis() -> dict:
    if "a" not in _CACHE:
        _CACHE["a"] = json.loads(ANALYSIS.read_text())
    return _CACHE["a"]


def truth() -> dict:
    """MuJoCo release facts per run (T43 Y `y_metrics.json`) — for scoring only, never fed to the latch."""
    if "y" not in _CACHE:
        _CACHE["y"] = json.loads(Y_METRICS.read_text())["runs"]
    return _CACHE["y"]


def requests(key: str) -> list:
    """`[(t, summary)]` of the run's server segment, request order."""
    if key in _CACHE:
        return _CACHE[key]
    r = analysis()["runs"][key]
    seg = r["server_segment"]
    first = pathlib.Path(seg["first"])
    i0 = int(first.stem.split("_")[1])
    rows = [json.loads(line) for line in open(pathlib.Path(r["run_dir"]) / "frames.jsonl")
            if line.strip()]
    plan_t = {int(p["seq"]): int(p["t_step"]) for p in rows if p["kind"] == "planning"}
    out = []
    for n in range(int(seg["n"])):
        with np.load(first.parent / f"chunk_{i0 + n:05d}.npz", allow_pickle=True) as z:
            s = json.loads(str(z["summary_json"]))
            seq = int(z["t_step"])
        out.append((plan_t[seq], s))
    _CACHE[key] = out
    return out


def replay(key: str, *, slip_detach: bool, **cfg) -> dict:
    from AG3S.trajopt.grasp_latch import (
        GraspLatch,
        LatchConfig,
        grasp_signal_from_feedback,
    )

    side = analysis()["runs"][key]["side"]
    hand = 0 if side == "left" else 1
    latch = GraspLatch(LatchConfig(slip_detach=slip_detach, **cfg), evidence=True,
                       external_destination=True)
    for _ in range(3):
        latch.update(label="obj0", score=0.9, runner_up=0.1, gripper=1.0)
    rows, mismatch = [], []
    first_slip: Optional[int] = None
    observed_while_held = 0
    for t, s in requests(key):
        g = s.get("grasp") or {}
        ev = g.get("evidence") or {}
        reach = ev.get("reach_mm")
        signal = grasp_signal_from_feedback(s.get("exec_feedback"), hand, 0.85)
        e = latch.observe_grasp(
            signal, reach_m=None if reach is None else float(reach) / 1000.0,
            manipulated_id=ev.get("manipulated_id"), manipulated_state=ev.get("manipulated_state"),
            placed=bool(ev.get("placed", False)), held_observation=None)
        if e.attach and g.get("rejected"):
            e = latch.revert_attach(str((g.get("rejected") or {}).get("reason", "rejected")))
        if g.get("state") == "held" and (s.get("manipulated") or {}).get("state") == "visible":
            observed_while_held += 1
        if e.slipped and first_slip is None:
            first_slip = t
        busy = {"closing", "held", "placed"}
        if (g.get("state") in busy or latch.phase.value in busy) and g.get("state") != latch.phase.value:
            mismatch.append((t, g["state"], latch.phase.value))
        rows.append({"t": t, "recorded": g.get("state"), "replayed": latch.phase.value,
                     "detach": bool(e.detach), "slipped": bool(e.slipped),
                     "slip": (e.evidence or {}).get("slip")})
    return {"key": key, "rows": rows, "mismatch": mismatch, "first_slip_t": first_slip,
            "observed_while_held": observed_while_held}


def scored_keys() -> list:
    """Runs with a server segment and a MuJoCo release record (T43 Y: V4 · V5, H · B)."""
    a, y = analysis()["runs"], truth()
    return sorted(k for k in a if a[k].get("server_segment") and k in y)
