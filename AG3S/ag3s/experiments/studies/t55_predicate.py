"""T55 — physical grasp predicates (the answer), persistence, the 4 states and the error metrics (pure numpy).

`AG3S/ag3s/docs/handoff/T55.task.md` 사전 고정 1 / 4 / 5. Nothing here touches MuJoCo, jax or the latch:
the inputs are per-policy-frame arrays that `t55_resim` (physics) and `t55_study` (sources) produce.

Predicates (사전 고정 1)
-----------------------
Per policy frame, from the physical state of **every** fruit (`t48b_scene.classify` on the pose the camera
saw — in the re-simulation that is the kinematics / contacts of the last `mj_step`, exactly what the
recorded camera saw, see `t55_resim`):

* ``held_by[fruit]`` = the hand that holds it (``classify`` state ``held``: both finger bodies of that hand
  touch it and it touches neither table nor crate), else ``None``.
* **raw P_grasp** = the episode's *used* arm holds **some** fruit (`held_by[f] == used_arm` for any f).
* **raw P_task** = the used arm holds the **instructed** target (`held_by[target] == used_arm`). Changing the
  instruction changes only `target`, so P_task changes and P_grasp does not.
* ``placed`` = the target's centre is inside the crate's inner volume and no hand holds it (T53 ``placed``).

Persistence ``K`` (same K for onset and release, every source and condition):

* ``causal`` (default; the task's wording "연속 K policy frame 동안 성립할 때 참이 된다"): a hysteresis that
  starts False and flips its state after ``K`` consecutive raw frames that disagree with it. Onset is
  therefore reported at the K-th frame of a true run; a true run shorter than K never counts; a false gap
  shorter than K inside a hold does not release.
* ``runs`` (alternative, non-causal): true runs shorter than K are dropped, then false gaps shorter than K
  between two true runs are filled; onsets stay at the first raw frame.

``K = 1`` is the raw predicate in both modes.

Four states (P_task view, after persistence; precedence top-down)::

    target_held    P_task
    wrong_object   P_grasp and not P_task        (a different object in the active hand)
    placed         not P_grasp and target placed
    empty          otherwise                     (empty hand, closing on nothing, slip, ...)

Combination rule (사전 고정 4)
-----------------------------
``task_ready = HELD and feature_target_held`` where HELD is the strong rule's phase (`grasp_latch` phase
``held``) and ``feature_target_held`` is the feature's per-frame flag after its own causal persistence
(`feature_persist`, p consecutive frames). `combine` returns ``task_ready`` and passes the latch trace
(phase / attach / detach) through **untouched** — the feature never alters attach or detach.

Errors (P_task basis unless stated; ``window`` selects the frames of a condition)
-------------------------------------------------------------------------------
* false pass   ``ready and not truth``   rate = count / #(window and not truth)
* false reject ``not ready and truth``   rate = count / #(window and truth)
* latency      per onset event of ``truth`` inside the window (a rising edge of the persistent predicate;
  the first window frame counts as an onset when truth is already true there): frames from the onset to
  the first frame with ``ready`` inside the same true run; no such frame = undetected (censored, reported
  as a count, never imputed).
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional, Sequence

import numpy as np

STATES4 = ("empty", "wrong_object", "target_held", "placed")
PERSIST_MODES = ("causal", "runs")
HANDS = ("left", "right")


# ----------------------------------------------------------------------------- raw predicates


def raw_predicates(held_by: Mapping[str, Sequence[Optional[str]]], target: str, used_arm: str,
                   in_crate: Optional[Sequence[bool]] = None) -> dict[str, np.ndarray]:
    """Per-frame raw P_grasp / P_task / placed / held object from ``held_by[fruit][frame]`` (hand or None).

    `in_crate` = the target centre inside the crate volume (per frame); ``placed`` = in_crate and no hand
    holds the target. ``held_object`` = the fruit the used arm holds ("" when none; the first in sorted
    order if the hand touched two at once — reported via ``n_held``)."""
    if used_arm not in HANDS:
        raise ValueError(f"used_arm must be one of {HANDS}, got {used_arm!r}")
    if target not in held_by:
        raise ValueError(f"target {target!r} not among {sorted(held_by)}")
    fruits = sorted(held_by)
    n = len(held_by[target])
    if any(len(held_by[f]) != n for f in fruits):
        raise ValueError("held_by sequences differ in length")
    mine = np.array([[held_by[f][i] == used_arm for f in fruits] for i in range(n)], bool).reshape(n, len(fruits))
    p_grasp = mine.any(1)
    p_task = np.array([held_by[target][i] == used_arm for i in range(n)], bool)
    target_any = np.array([held_by[target][i] is not None for i in range(n)], bool)
    if in_crate is None:
        placed = np.zeros(n, bool)
    else:
        ic = np.asarray(in_crate, bool).reshape(-1)
        if ic.shape != (n,):
            raise ValueError(f"in_crate has {ic.shape}, expected ({n},)")
        placed = ic & ~target_any
    held_object = np.array([fruits[int(np.flatnonzero(r)[0])] if r.any() else "" for r in mine], dtype="U16")
    return {"p_grasp": p_grasp, "p_task": p_task, "placed": placed, "held_object": held_object,
            "n_held": mine.sum(1).astype(np.int64)}


# ----------------------------------------------------------------------------- persistence


def persist(raw: Sequence[bool], K: int, mode: str = "causal") -> np.ndarray:
    """Persistence filter of a boolean frame sequence (module docstring)."""
    x = np.asarray(raw, bool).reshape(-1)
    K = int(K)
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if mode not in PERSIST_MODES:
        raise ValueError(f"mode must be one of {PERSIST_MODES}, got {mode!r}")
    if K == 1 or len(x) == 0:
        return x.copy()
    if mode == "causal":
        out = np.zeros(len(x), bool)
        state, run = False, 0
        for i, v in enumerate(x):
            run = run + 1 if bool(v) != state else 0
            if run >= K:
                state, run = bool(v), 0
            out[i] = state
        return out
    # runs: drop short true runs, then fill short false gaps bounded by true on both sides
    out = x.copy()
    for s, e in runs(out, True):
        if e - s < K:
            out[s:e] = False
    for s, e in runs(out, False):
        if e - s < K and s > 0 and e < len(out):
            out[s:e] = True
    return out


def runs(x: Sequence[bool], value: bool = True) -> list[tuple[int, int]]:
    """Maximal runs ``[start, end)`` of `value` in `x`."""
    a = np.asarray(x, bool).reshape(-1)
    out, start = [], None
    for i, v in enumerate(a):
        if bool(v) == bool(value):
            if start is None:
                start = i
        elif start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(a)))
    return out


def onsets(x: Sequence[bool], window: Optional[Sequence[bool]] = None) -> np.ndarray:
    """Onset frames of `x` inside `window`: rising edges, plus the first window frame of each window run
    when `x` is already true there."""
    a = np.asarray(x, bool).reshape(-1)
    w = np.ones(len(a), bool) if window is None else np.asarray(window, bool).reshape(-1)
    if w.shape != a.shape:
        raise ValueError("window shape differs")
    out = []
    for i in range(len(a)):
        if not (a[i] and w[i]):
            continue
        if i == 0 or not a[i - 1] or not w[i - 1]:
            out.append(i)
    return np.asarray(out, np.int64)


def four_state(p_grasp: Sequence[bool], p_task: Sequence[bool], placed: Sequence[bool]) -> np.ndarray:
    pg, pt, pl = (np.asarray(v, bool).reshape(-1) for v in (p_grasp, p_task, placed))
    if not (pg.shape == pt.shape == pl.shape):
        raise ValueError("shapes differ")
    out = np.full(len(pg), "empty", dtype="U12")
    out[pl & ~pg] = "placed"
    out[pg & ~pt] = "wrong_object"
    out[pt] = "target_held"
    return out


def state_counts(states: Sequence[str]) -> dict[str, int]:
    s = np.asarray(states)
    return {k: int((s == k).sum()) for k in STATES4}


def predicates(held_by: Mapping[str, Sequence[Optional[str]]], target: str, used_arm: str, *,
               in_crate: Optional[Sequence[bool]] = None, K: int = 1, mode: str = "causal") -> dict:
    """Raw + persistent predicates and the 4 states for one (frames, target) pair."""
    r = raw_predicates(held_by, target, used_arm, in_crate)
    pg = persist(r["p_grasp"], K, mode)
    pt = persist(r["p_task"], K, mode)
    pt &= pg  # P_task implies P_grasp (both use the same K; guard against mode edge cases)
    return {"p_grasp_raw": r["p_grasp"], "p_task_raw": r["p_task"], "p_grasp": pg, "p_task": pt,
            "placed": r["placed"], "held_object": r["held_object"], "n_held": r["n_held"],
            "state4": four_state(pg, pt, r["placed"]), "K": int(K), "mode": mode}


# ----------------------------------------------------------------------------- sources / combination


def feature_persist(flag: Sequence[bool], p: int) -> np.ndarray:
    """Causal p-frame AND: true at frame i when frames i-p+1..i are all true (p = 1: the flag itself)."""
    f = np.asarray(flag, bool).reshape(-1)
    p = int(p)
    if p < 1:
        raise ValueError("p must be >= 1")
    if p == 1:
        return f.copy()
    out = np.zeros(len(f), bool)
    run = 0
    for i, v in enumerate(f):
        run = run + 1 if v else 0
        out[i] = run >= p
    return out


def held_from_phase(phase: Sequence[str]) -> np.ndarray:
    return np.asarray([str(p) == "held" for p in phase], bool)


@dataclasses.dataclass(frozen=True)
class LatchTrace:
    """Per-request strong-rule output (`t55_resim.offline_latch`)."""

    phase: tuple
    attach: tuple
    detach: tuple

    @classmethod
    def from_arrays(cls, phase, attach, detach) -> "LatchTrace":
        return cls(tuple(str(x) for x in phase), tuple(bool(x) for x in attach),
                   tuple(bool(x) for x in detach))


def combine(trace: LatchTrace, feature_flag: Optional[Sequence[bool]], p: int = 1) -> dict:
    """``task_ready`` of the strong rule alone (`feature_flag is None`) or of ``HELD and feature``.

    The latch trace is returned as-is (same tuples): the feature never alters attach / detach / phase."""
    held = held_from_phase(trace.phase)
    if feature_flag is None:
        ready = held.copy()
        feat = None
    else:
        feat = feature_persist(feature_flag, p)
        if feat.shape != held.shape:
            raise ValueError(f"feature has {feat.shape}, latch {held.shape}")
        ready = held & feat
    return {"task_ready": ready, "held": held, "feature": feat, "phase": trace.phase,
            "attach": trace.attach, "detach": trace.detach}


# ----------------------------------------------------------------------------- errors


def errors(ready: Sequence[bool], truth: Sequence[bool], window: Optional[Sequence[bool]] = None) -> dict:
    r = np.asarray(ready, bool).reshape(-1)
    t = np.asarray(truth, bool).reshape(-1)
    w = np.ones(len(r), bool) if window is None else np.asarray(window, bool).reshape(-1)
    if not (r.shape == t.shape == w.shape):
        raise ValueError(f"shapes differ: ready {r.shape} truth {t.shape} window {w.shape}")
    neg, pos = w & ~t, w & t
    fp, fr = int((r & neg).sum()), int((~r & pos).sum())
    return {"n_frames": int(w.sum()), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
            "false_pass": fp, "false_reject": fr,
            "false_pass_rate": (fp / int(neg.sum())) if neg.any() else None,
            "false_reject_rate": (fr / int(pos.sum())) if pos.any() else None}


def latencies(ready: Sequence[bool], truth: Sequence[bool],
              window: Optional[Sequence[bool]] = None) -> list[dict]:
    """One entry per onset event of `truth` in `window` (module docstring)."""
    r = np.asarray(ready, bool).reshape(-1)
    t = np.asarray(truth, bool).reshape(-1)
    w = np.ones(len(r), bool) if window is None else np.asarray(window, bool).reshape(-1)
    out = []
    for s in onsets(t, w):
        e = s
        while e < len(t) and t[e] and w[e]:
            e += 1
        hit = np.flatnonzero(r[s:e])
        out.append({"onset": int(s), "run_end": int(e), "run_len": int(e - s),
                    "latency": (int(hit[0]) if hit.size else None), "detected": bool(hit.size)})
    return out


def latency_summary(events: Sequence[dict]) -> dict:
    lat = [e["latency"] for e in events if e["latency"] is not None]
    return {"n_events": len(events), "n_detected": len(lat), "n_undetected": len(events) - len(lat),
            "latency_mean": (float(np.mean(lat)) if lat else None),
            "latency_median": (float(np.median(lat)) if lat else None)}


def paired_latency(events_a: Sequence[dict], events_b: Sequence[dict]) -> list[Optional[int]]:
    """Per event (same onsets — same truth and window): latency(b) - latency(a); None when either is
    undetected."""
    if [e["onset"] for e in events_a] != [e["onset"] for e in events_b]:
        raise ValueError("events differ (different truth / window)")
    return [None if (a["latency"] is None or b["latency"] is None) else int(b["latency"] - a["latency"])
            for a, b in zip(events_a, events_b)]


# ----------------------------------------------------------------------------- natural events


def natural_events(pred: Mapping[str, np.ndarray], gripper_closed: Sequence[bool],
                   phase: Optional[Sequence[str]] = None) -> dict:
    """What happened in an unmodified rollout (사전 고정 3 "자연 발생"), from persistent predicates.

    * ``closed_empty_frames``  used-hand gripper closed (state < closed threshold, given) and not P_grasp
    * ``wrong_object_frames``  P_grasp and not P_task
    * ``slips``                falling edges of P_task not followed by ``placed`` within the same false run
    * ``grasp_onsets``         rising edges of P_grasp (> 1 = re-grasp / retry)
    * ``closing_attempts``     strong-rule entries into ``closing`` (when `phase` is given)
    """
    pg, pt, pl = (np.asarray(pred[k], bool) for k in ("p_grasp", "p_task", "placed"))
    gc = np.asarray(gripper_closed, bool).reshape(-1)
    slips = 0
    for s, e in runs(pt, False):
        if s == 0:
            continue
        if not pl[s:e].any():
            slips += 1
    out = {"closed_empty_frames": int((gc & ~pg).sum()), "wrong_object_frames": int((pg & ~pt).sum()),
           "slips": int(slips), "grasp_onsets": int(len(onsets(pg))),
           "target_onsets": int(len(onsets(pt)))}
    if phase is not None:
        ph = [str(x) for x in phase]
        out["closing_attempts"] = int(sum(1 for i in range(len(ph))
                                          if ph[i] == "closing" and (i == 0 or ph[i - 1] != "closing")))
        out["held_frames"] = int(sum(1 for x in ph if x == "held"))
    return out


__all__ = ["LatchTrace", "PERSIST_MODES", "STATES4", "combine", "errors", "feature_persist", "four_state",
           "held_from_phase", "latencies", "latency_summary", "natural_events", "onsets", "paired_latency",
           "persist", "predicates", "raw_predicates", "runs", "state_counts"]
