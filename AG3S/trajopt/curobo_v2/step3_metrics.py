"""T63 step 3 — the metric definitions of the common evaluator (pure numpy, no AG3S import).

`step3_eval.py` (run on the baseline snapshot) computes, for **every** candidate trajectory of a
chunk, the same three per-step arrays with the same code — the TO-view clearance `(50, S)` (our TO's
`CollisionLinearizer.esdf_clearance` with the record's contact policy, per-step served finger
opening), the TCP path `(50, 3)` and the arm joints `(50, 14)` — and hands them to
`candidate_metrics` here. Nothing in this module knows which method produced a trajectory.

Metric table (task T63 step 3, fixed before measuring):

| key | definition | unit |
|---|---|---|
| `min_clearance_mm_<w>` | min over steps `< W` and all spheres of the TO-view clearance — **certified clearance**: includes the server's `obstacle_margin` (10 mm on non-target obstacle rows) and the contact margins of target rows | mm |
| `penetration_<w>` | `min_clearance_mm_<w> < 0` — a **certified-clearance violation (10 mm margin included)**, not a physical penetration (T63b user decision 2 wording) | bool (aggregated: chunk count) |
| `phys_min_clearance_mm_<w>` | the same evaluator with every margin 0 (`obstacle_margin`, contact `manipulated_link_margin`, `esdf_margin`, destination margin): `d − r` per row, where `d` is the row's field answer under the **unchanged** contact policy (rows allowed to touch the target ask the target-free layer, T30 F1) — **physical clearance** (T63b user decision 2) | mm |
| `phys_penetration_<w>` | `phys_min_clearance_mm_<w> < 0` — a sphere inside a surface (physical penetration) | bool (aggregated: chunk count) |
| `tcp_disp_mm_mean_<w>` / `_max_<w>` | `‖tcp_k − tcp_ref_k‖` over steps `< W`; TCP = mean of the manipulating hand's finger spheres (`fixed_input_ablation.FINGER_LINKS`) | mm |
| `goal_error_joint_rad` | `max_j |q_49 − ref_49|` (chunk end) | rad |
| `goal_error_tcp_mm` | `‖tcp_49 − tcp_ref_49‖` | mm |
| `n_vel_violation_steps_<w>` | steps `k < W` with some joint `|q_k − q_{k−1}| > max_step + LIMIT_TOL_RAD`, `q_{−1} = q_now` (the TO's hard anchor) | count |
| `n_acc_violation_steps_<w>` | steps `1 ≤ k < W` with some joint `|q_k − 2q_{k−1} + q_{k−2}| > max_step_change + LIMIT_TOL_RAD` (step 0 needs the command before `q_now`, not recorded) | count |
| `n_pos_violation_steps_<w>` | steps `k < W` with some joint outside `[lower, upper]` by more than `LIMIT_TOL_RAD` | count |
| `max_vel_excess_rad_s_<w>` / `max_acc_excess_rad_s2_<w>` | the largest excess over the limit (0 = compliant) | rad/s, rad/s² |
| `jerk_rms_rad_s3_<w>` | RMS over steps `2 ≤ k < W` and the 14 joints of the 3rd difference of `[q_now, q_0, …]`, divided by `dt³` | rad/s³ |

Windows `W`: `exec8` (the executed steps), `plan16` (the TO's plan window) — the task's two; `all50`
(the whole chunk) is supplementary. Limits are the TO's effective limits (`robot_spec.json`
`limits`: model_xml position ± margin, URDF velocity / acceleration × 0.9), per 15 Hz step.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

FRAME_DT = 1.0 / 15.0
CONTROL_PERIOD_MS = 1000.0 / 15.0
WINDOWS: dict[str, int] = {"exec8": 8, "plan16": 16, "all50": 50}
#: TO-active chunk (user decision 2): recorded refined vs reference TCP > 1 mm anywhere in the plan
#: window (`plan16`)
TO_ACTIVE_TCP_MM = 1.0
#: a joint-limit excess below this (rad per step, rad per step²) is not counted — the TO's own
#: records carry ~5e-6 rad/step² QP-tolerance overshoots. The excess itself is reported unthresholded.
LIMIT_TOL_RAD = 1e-6


# =================================================================================================
# limits
# =================================================================================================

def step_limits(spec_limits: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """`robot_spec.json` `limits` (rad, rad/s, rad/s²) -> per-step bounds of the 15 Hz grid."""
    dt = float(spec_limits.get("dt", FRAME_DT))
    vel = np.asarray(spec_limits["velocity"], np.float64)
    acc = np.asarray(spec_limits["acceleration"], np.float64)
    return {"lower": np.asarray(spec_limits["lower"], np.float64),
            "upper": np.asarray(spec_limits["upper"], np.float64),
            "max_step": vel * dt, "max_step_change": acc * dt * dt, "dt": dt}


def kinematics(q: np.ndarray, q_now: np.ndarray, lim: Mapping[str, Any],
               windows: Mapping[str, int] = WINDOWS, tol: float = LIMIT_TOL_RAD) -> dict[str, Any]:
    """Limit-violation step counts, largest excesses and jerk RMS of `q (T, nq)` after `q_now`."""
    q = np.asarray(q, np.float64)
    X = np.vstack([np.asarray(q_now, np.float64).reshape(1, -1), q])
    dt = float(lim["dt"])
    T = q.shape[0]
    v_ex = (np.abs(np.diff(X, axis=0)) - lim["max_step"]).max(axis=1)                # step k
    a_ex = np.full(T, -np.inf)
    a_ex[1:] = (np.abs(np.diff(X, n=2, axis=0)) - lim["max_step_change"]).max(axis=1)  # k >= 1
    p_ex = np.maximum(lim["lower"] - q, q - lim["upper"]).max(axis=1)
    jerk = np.diff(X, n=3, axis=0)                                                     # k >= 2
    out: dict[str, Any] = {}
    for name, W in windows.items():
        W = min(int(W), T)
        out[f"n_vel_violation_steps_{name}"] = int((v_ex[:W] > tol).sum())
        out[f"n_acc_violation_steps_{name}"] = int((a_ex[:W] > tol).sum())
        out[f"n_pos_violation_steps_{name}"] = int((p_ex[:W] > tol).sum())
        out[f"max_vel_excess_rad_s_{name}"] = float(max(v_ex[:W].max(), 0.0) / dt)
        out[f"max_acc_excess_rad_s2_{name}"] = (float(max(a_ex[1:W].max(), 0.0) / dt ** 2)
                                                 if W > 1 else 0.0)
        j = jerk[:max(W - 2, 0)]
        out[f"jerk_rms_rad_s3_{name}"] = (float(np.sqrt(np.mean(j ** 2)) / dt ** 3)
                                          if j.size else None)
    return out


# =================================================================================================
# one candidate
# =================================================================================================

def physical_metrics(clearance_phys_m: np.ndarray, windows: Mapping[str, int] = WINDOWS
                     ) -> dict[str, Any]:
    """`phys_min_clearance_mm_<w>` / `phys_penetration_<w>` of a `(T, S)` physical clearance (m)."""
    cl = np.asarray(clearance_phys_m, np.float64) * 1e3
    out: dict[str, Any] = {}
    for name, W in windows.items():
        W = min(int(W), cl.shape[0])
        m = float(cl[:W].min())
        out[f"phys_min_clearance_mm_{name}"] = m
        out[f"phys_penetration_{name}"] = bool(m < 0.0)
    return out


def candidate_metrics(q: Optional[np.ndarray], q_now: np.ndarray, clearance_m: Optional[np.ndarray],
                      tcp_m: Optional[np.ndarray], ref_q: np.ndarray, ref_tcp_m: np.ndarray,
                      lim: Mapping[str, Any], windows: Mapping[str, int] = WINDOWS,
                      clearance_phys_m: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Every metric of the table above for one trajectory. `q None` (no trajectory) or a non-finite
    one gives only `{"has_trajectory", "finite"}` — counted, never averaged. The physical keys are
    present only when `clearance_phys_m` is given."""
    if q is None:
        return {"has_trajectory": False, "finite": False}
    q = np.asarray(q, np.float64)
    if not np.all(np.isfinite(q)):
        return {"has_trajectory": True, "finite": False}
    cl = np.asarray(clearance_m, np.float64) * 1e3
    tcp = np.asarray(tcp_m, np.float64)
    rtcp = np.asarray(ref_tcp_m, np.float64)
    disp = np.linalg.norm(tcp - rtcp, axis=1) * 1e3
    out: dict[str, Any] = {"has_trajectory": True, "finite": True}
    for name, W in windows.items():
        W = min(int(W), q.shape[0])
        m = float(cl[:W].min())
        out[f"min_clearance_mm_{name}"] = m
        out[f"penetration_{name}"] = bool(m < 0.0)
        out[f"tcp_disp_mm_mean_{name}"] = float(disp[:W].mean())
        out[f"tcp_disp_mm_max_{name}"] = float(disp[:W].max())
    Wp = min(windows.get("plan16", 16), q.shape[0])
    k, s = np.unravel_index(int(np.argmin(cl[:Wp])), cl[:Wp].shape)
    out["worst_plan16"] = {"step": int(k), "sphere": int(s)}
    if clearance_phys_m is not None:
        out.update(physical_metrics(clearance_phys_m, windows))
    out["goal_error_joint_rad"] = float(np.abs(q[-1] - np.asarray(ref_q, np.float64)[-1]).max())
    out["goal_error_tcp_mm"] = float(disp[-1])
    out.update(kinematics(q, q_now, lim, windows))
    return out


# =================================================================================================
# aggregation
# =================================================================================================

def summarize(values: Iterable[Any]) -> dict[str, Any]:
    x = np.asarray([v for v in values if v is not None], np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "median": float(np.median(x)), "mean": float(x.mean()),
            "min": float(x.min()), "p5": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95)), "max": float(x.max())}


#: chunk splits (task T63 step 3 / user decision 2)
SPLITS: dict[str, Callable[[Mapping[str, Any]], bool]] = {
    "all": lambda r: True,
    "to_active": lambda r: bool(r["to_active"]),
    "to_inactive": lambda r: not r["to_active"],
    "box_free": lambda r: r["box_layer"] == "free",
    "box_main_fine": lambda r: r["box_layer"] == "main_fine",
    "pre_grasp": lambda r: not r["grasp_active"],
    "post_grasp": lambda r: bool(r["grasp_active"]),
}

METRIC_KEYS: tuple[str, ...] = tuple(
    [f"min_clearance_mm_{w}" for w in WINDOWS]
    + [f"phys_min_clearance_mm_{w}" for w in WINDOWS]
    + [f"tcp_disp_mm_{s}_{w}" for w in WINDOWS for s in ("mean", "max")]
    + ["goal_error_joint_rad", "goal_error_tcp_mm"]
    + [f"jerk_rms_rad_s3_{w}" for w in WINDOWS]
    + [f"max_vel_excess_rad_s_{w}" for w in WINDOWS]
    + [f"max_acc_excess_rad_s2_{w}" for w in WINDOWS])
COUNT_KEYS: tuple[str, ...] = tuple(f"n_{k}_violation_steps_{w}" for k in ("vel", "acc", "pos")
                                    for w in WINDOWS)


def timing_summary(solves: Sequence[Mapping[str, Any]], key: str = "wall_ms") -> dict[str, Any]:
    """Cold and warm solve times kept apart; the fraction of warm solves over one control period."""
    cold = [float(s[key]) for s in solves if s.get(key) is not None and s.get("cold")]
    warm = [float(s[key]) for s in solves if s.get(key) is not None and not s.get("cold")]
    out: dict[str, Any] = {"key": key, "cold_ms": cold, "warm": summarize(warm)}
    out["warm_n_over_66p7ms"] = int(sum(w > CONTROL_PERIOD_MS for w in warm))
    out["warm_fraction_over_66p7ms"] = (float(out["warm_n_over_66p7ms"] / len(warm))
                                        if warm else None)
    return out


def aggregate_candidate(rows: Sequence[Mapping[str, Any]], cand: str) -> dict[str, Any]:
    """One candidate over a set of chunk rows (`row["candidates"][cand]`)."""
    entries = [r["candidates"].get(cand) for r in rows]
    present = [e for e in entries if e is not None]
    ok = [e["metrics"] for e in present if e["metrics"].get("finite")]
    out: dict[str, Any] = {
        "n_chunks": len(rows),
        "n_not_run": int(sum(e is None or not e["metrics"].get("has_trajectory") for e in entries)),
        "n_nonfinite": int(sum(e["metrics"].get("has_trajectory") and not e["metrics"].get("finite")
                               for e in present)),
        "n_evaluated": len(ok),
        "n_solver_failure": int(sum(not (e.get("solve") or {}).get("success", True)
                                    for e in present)),
        "status_counts": {},
    }
    for e in present:
        st = str((e.get("solve") or {}).get("status"))
        out["status_counts"][st] = out["status_counts"].get(st, 0) + 1
    for w in WINDOWS:
        out[f"n_penetration_{w}"] = int(sum(m[f"penetration_{w}"] for m in ok))
    # physical clearance (T63b): counted over the chunks that carry it (rows written before it
    # existed have no key -> `n_phys_evaluated` says how many did)
    phys = [m for m in ok if "phys_min_clearance_mm_exec8" in m]
    out["n_phys_evaluated"] = len(phys)
    for w in WINDOWS:
        out[f"n_phys_penetration_{w}"] = int(sum(m[f"phys_penetration_{w}"] for m in phys))
    out["metrics"] = {k: summarize(m.get(k) for m in ok) for k in METRIC_KEYS}
    out["kinematic_counts"] = {
        k: {"total_steps": int(sum(m[k] for m in ok)),
            "n_chunks_with_any": int(sum(m[k] > 0 for m in ok))} for k in COUNT_KEYS}
    solves = [e["solve"] for e in present if e.get("solve")]
    devices = sorted({str(s.get("device")) for s in solves})
    out["timing"] = {"device": devices, "wall": timing_summary(solves, "wall_ms")}
    if any(s.get("gpu_solve_ms") is not None for s in solves):
        out["timing"]["gpu_solve"] = timing_summary(solves, "gpu_solve_ms")
    return out


def aggregate(rows: Sequence[Mapping[str, Any]], candidates: Sequence[str],
              splits: Mapping[str, Callable[[Mapping[str, Any]], bool]] = SPLITS
              ) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, pred in splits.items():
        sel = [r for r in rows if pred(r)]
        out[name] = {"n_chunks": len(sel),
                     "candidates": {c: aggregate_candidate(sel, c) for c in candidates}}
    return out


__all__ = ["CONTROL_PERIOD_MS", "COUNT_KEYS", "FRAME_DT", "LIMIT_TOL_RAD", "METRIC_KEYS", "SPLITS",
           "TO_ACTIVE_TCP_MM", "WINDOWS", "aggregate", "aggregate_candidate", "candidate_metrics",
           "kinematics", "physical_metrics", "step_limits", "summarize", "timing_summary"]
