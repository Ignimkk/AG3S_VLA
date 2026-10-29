"""AG3S-PRE-CORE-TO E1 — offline settings and the recorded-run grasp-goal gate for `tp_timing`.

The recorded closed-loop runs did not serve IFG-TOA, so their records carry no frozen grasp goal. This module rebuilds
it from the record with **the server's own gate** (`SafePolicy._grasp_anchor_gate`, called on a stub that holds only
what the gate reads), request by request in `t_step` order:

| gate input | from the record (`chunk_*.npz`) |
|---|---|
| latch state (final, after `_run_latch`) | `summary_json.grasp.state` |
| revoked attach | `summary_json.grasp.attach_revoked` |
| hand · hold link · parent link | `summary_json.grasp.hand` · `.hold_link` · `.parent_link` (else the run's manifest hand, `ee_<hand>`) |
| manipulated object (id, centroid) | `summary_json.manipulated.id` · `.centroid` |
| the policy chunk | `reference_chunk` (the chunk handed to the refiner) |
| `q_now` | the client planning row `qpos` (`fixed_input_ablation.q_now_from_qpos`, as `ta_offline.Run.inputs`) |
| close threshold | `LatchConfig().gripper_closed_below` (the server sets no flag for it) |
| goal / anchor radii | the variant's `cost.grasp_goal_max_from_target` (120 mm) · `cost.grasp_anchor_radius` (150 mm) |

So the goal is frozen at the first pre-grasp (latched / closing) request whose policy chunk plans a close for the hand
(the palm FK at the close row, other joints at `q_now`), if it lies within 120 mm of the manipulated centroid; it is
cleared when the latch leaves latched / closing, an attach is revoked or the target id changes; the anchor is handed to
the optimizer when the palm at `q_now` is within 150 mm of the goal. A request without a planning row is not seen by
the gate (`skipped`), as `tp_timing` does not solve it either.

Settings (`tp_timing --settings-module AG3S.trajopt.experiments.e1_goal`, each on top of the run's server config):

| name | what |
|---|---|
| `E0` | the recorded server config (no anchor) |
| `E1C` | `--grasp-anchor 2000 --grasp-anchor-path-weight 0`: goal term only (`w_grasp_goal` 2000, `w_grasp_tcp` 0) |
| `E1B10` · `E1B20` | `--grasp-goal-box 10` / `20`: the per-axis goal box, no palm cost |
| `TOA` | `--grasp-anchor 2000` (IFG-TOA as served in IFG-APPLE-TOA3), for reference |

`goal_eval` (every setting, also E0): the palm at the goal rows of the returned trajectory and of the policy chunk vs
the rebuilt goal, on the chunks where the gate hands out an anchor (`probe` gate, E1-B-shaped, default radii).

    python -m AG3S.trajopt.experiments.e1_goal summarize OUT/*.json   # per-setting table of a tp_timing pass
"""
from __future__ import annotations

import dataclasses
import json
import sys
import types
from typing import Any, Optional

import numpy as np

W_E1C = 2000.0

SETTINGS: dict[str, dict] = {
    "E0": {},
    "E1C": {"cost": {"w_grasp_tcp": 0.0, "w_grasp_goal": W_E1C}},
    "E1B10": {"cost": {"grasp_goal_box": 0.010}},
    "E1B20": {"cost": {"grasp_goal_box": 0.020}},
    "TOA": {"cost": {"w_grasp_tcp": W_E1C, "w_grasp_goal": W_E1C}},
}


def grasp_on(cost) -> bool:
    """Does this cost config switch the grasp-goal gate on (IFG-TOA, E1-C or E1-B)?"""
    return (float(getattr(cost, "w_grasp_tcp", 0.0) or 0.0) > 0.0
            or float(getattr(cost, "w_grasp_goal", 0.0) or 0.0) > 0.0
            or float(getattr(cost, "grasp_goal_box", 0.0) or 0.0) > 0.0)


class RecordedGraspGate:
    """`SafePolicy._grasp_anchor_gate` on a recorded run (one stub per cost config, state carried request to request)."""

    def __init__(self, run, cost):
        from AG3S.trajopt import wire
        from AG3S.trajopt.grasp_latch import GraspPhase, LatchConfig
        from AG3S.trajopt.safe_policy import SafePolicy

        p = object.__new__(SafePolicy)
        p.to_config = types.SimpleNamespace(cost=cost)
        p._latch = types.SimpleNamespace(phase=GraspPhase.SEARCHING,
                                         config=types.SimpleNamespace(
                                             gripper_closed_below=float(LatchConfig().gripper_closed_below)))
        p._grasp_goal = None
        p._last_grasp_anchor = {}
        p._last_grasp_record = {}
        p.constraint_robot_model = run.robot
        p.layout = run.layout
        p.gripper_columns = wire.gripper_columns(run.layout.nq_opt // 2)
        p.ag3s = types.SimpleNamespace(manipulated=None, frame_index=-1)
        p.grasp_links = {}
        p.hold_links = {}
        self.policy = p
        self.run = run

    def step(self, seq: int, summary: dict, reference_chunk: np.ndarray, q_now: np.ndarray):
        """`(GraspAnchor | None, gate record)` of request `seq`."""
        from AG3S.trajopt.grasp_latch import GraspPhase

        p = self.policy
        g = summary.get("grasp") or {}
        hand = str(g.get("hand") or self.run.hand)
        parent = str(g.get("parent_link") or ("ee_finger_l1" if hand == "left" else "ee_finger_r1"))
        p.grasp_links = {hand: (parent, ())}
        p.hold_links = {hand: str(g.get("hold_link") or f"ee_{hand}")}
        p._latch.phase = GraspPhase(str(g.get("state") or "searching"))
        m = summary.get("manipulated")
        p.ag3s.manipulated = (types.SimpleNamespace(id=int(m["id"]), centroid=np.asarray(m["centroid"], np.float64))
                              if isinstance(m, dict) and m.get("centroid") is not None and m.get("id") is not None
                              else None)
        p.ag3s.frame_index = int(seq)
        p._last_grasp_record = {"attach_revoked": g.get("attach_revoked")} if g.get("attach_revoked") else {}
        p._pending = {"chunk": np.asarray(reference_chunk, np.float64)}
        anchor = p._grasp_anchor_gate(np.asarray(q_now, np.float64), [hand])
        return anchor, dict(p._last_grasp_anchor)


def probe_cost(base_cost):
    """The gate's cost for `goal_eval`: E1-B-shaped (the gate only needs it on), the base radii."""
    return dataclasses.replace(base_cost, w_grasp_tcp=0.0, w_grasp_goal=0.0, grasp_goal_box=0.01)


def rebuild_gates(run, costs: dict[str, Any]) -> dict[str, dict[int, tuple]]:
    """`{name: {seq: (anchor | None, record)}}` for every cost in `costs` (+ `"_probe"`), over all of `run`'s requests
    in `t_step` order (independent of `--first-chunk` / `--max-chunks`)."""
    from AG3S.trajopt.experiments import fixed_input_ablation as FIA

    gates = {name: RecordedGraspGate(run, c) for name, c in costs.items()}
    out: dict[str, dict[int, tuple]] = {name: {} for name in gates}
    for seq in sorted(run.chunks):
        row = run.plan.get(seq)
        if row is None:
            continue
        with np.load(run.chunks[seq], allow_pickle=True) as z:
            s = json.loads(str(z["summary_json"]))
            ref = np.asarray(z["reference_chunk"], np.float64)
        q_now = FIA.q_now_from_qpos(row["qpos"], run.qadr)
        for name, gate in gates.items():
            out[name][seq] = gate.step(seq, s, ref, q_now)
    return out


def goal_eval(run, anchor, trajectory: np.ndarray, reference: np.ndarray, q_now: np.ndarray) -> Optional[dict]:
    """Palm vs the frozen goal on the goal rows, returned trajectory and policy chunk (mm). None without goal rows."""
    from AG3S.ag3s.constraints.attached import _link_pose_numeric

    if anchor is None:
        return None
    traj = np.asarray(trajectory, np.float64)
    rows = np.flatnonzero(anchor.goal_mask(traj.shape[1]))
    if not rows.size:
        return None
    g = np.asarray(anchor.goal, np.float64).reshape(1, 3)

    def palm(Q):
        full = run.layout.full_q(np.asarray(Q, np.float64)[:, rows], q_now)
        return np.stack([np.asarray(_link_pose_numeric(run.robot, full[:, i], anchor.link)[:3, 3], np.float64)
                         for i in range(rows.size)])

    out: dict[str, Any] = {"goal_rows": [int(k) for k in rows]}
    for tag, Q in (("result", traj), ("reference", np.asarray(reference, np.float64)[:, :traj.shape[1]])):
        d = palm(Q) - g
        out[f"{tag}_err_mm"] = round(float(np.max(np.linalg.norm(d, axis=1))) * 1e3, 3)
        out[f"{tag}_err_axis_max_mm"] = round(float(np.max(np.abs(d))) * 1e3, 3)
    return out


def anchor_summary(anchor) -> Optional[dict]:
    if anchor is None:
        return None
    return {"variant": anchor.variant, "close_row": anchor.close_row, "goal_m": [round(float(v), 5) for v in anchor.goal],
            "w_tcp": anchor.w_tcp, "w_goal": anchor.w_goal, "goal_box_mm": round(float(anchor.goal_box) * 1e3, 3)}


# ======================================================================================
# summary of a tp_timing pass
# ======================================================================================

def summarize(paths: list[str]) -> dict[str, Any]:
    """Per setting: `all_chunks` (every solved chunk: n, violation > 0.1 mm, max violation, solve ms median / p95) and,
    over the chunks where the probe gate hands out an anchor with goal rows (`goal_eval` present): n, violation
    > 0.1 mm, max violation, goal error (axis max) median / p90 vs the policy's, fallback used, solve ms, status."""
    acc: dict[str, dict[str, list]] = {}
    every: dict[str, dict[str, list]] = {}
    for path in paths:
        d = json.load(open(path))
        for ch in d.get("chunks", []):
            for v, e in (ch.get("variants") or {}).items():
                al = every.setdefault(v, {"viol": [], "ms": []})
                al["viol"].append(float(e.get("max_violation_mm") or 0.0))
                al["ms"].append(float(e.get("solve_ms") or 0.0))
                ge = e.get("goal_eval")
                if ge is None:
                    continue
                a = acc.setdefault(v, {"viol": [], "err": [], "err_ref": [], "fb": [], "ms": [], "status": []})
                a["viol"].append(float(e.get("max_violation_mm") or 0.0))
                a["err"].append(float(ge["result_err_axis_max_mm"]))
                a["err_ref"].append(float(ge["reference_err_axis_max_mm"]))
                a["fb"].append(bool((e.get("grasp_anchor") or {}).get("fallback_used")))
                a["ms"].append(float(e.get("solve_ms") or 0.0))
                a["status"].append(e.get("status"))
    table = {}
    for v, al in every.items():
        viol = np.asarray(al["viol"])
        table[v] = {"all_chunks": {"n": int(viol.size), "n_violation_gt_0p1mm": int((viol > 0.1).sum()),
                                   "violation_mm_max": round(float(viol.max(initial=0.0)), 2),
                                   "solve_ms_median": round(float(np.median(al["ms"])), 1),
                                   "solve_ms_p95": round(float(np.percentile(al["ms"], 95)), 1)}}
    for v, a in acc.items():
        viol, err = np.asarray(a["viol"]), np.asarray(a["err"])
        table[v] |= {"n_goal_chunks": int(viol.size), "n_violation_gt_0p1mm": int((viol > 0.1).sum()),
                    "violation_mm_max": round(float(viol.max(initial=0.0)), 2),
                    "goal_err_axis_mm_median": round(float(np.median(err)), 2) if err.size else None,
                    "goal_err_axis_mm_p90": round(float(np.percentile(err, 90)), 2) if err.size else None,
                    "policy_goal_err_axis_mm_median": round(float(np.median(a["err_ref"])), 2) if err.size else None,
                    "n_fallback_used": int(np.sum(a["fb"])),
                    "solve_ms_median": round(float(np.median(a["ms"])), 1) if err.size else None,
                    "solve_ms_p95": round(float(np.percentile(a["ms"], 95)), 1) if err.size else None,
                    "status": {s: a["status"].count(s) for s in sorted(set(map(str, a["status"])))}}
    return table


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "summarize":
        raise SystemExit("usage: python -m AG3S.trajopt.experiments.e1_goal summarize OUT.json [OUT.json ...]")
    print(json.dumps(summarize(argv[1:]), indent=1))


if __name__ == "__main__":
    sys.exit(main())
