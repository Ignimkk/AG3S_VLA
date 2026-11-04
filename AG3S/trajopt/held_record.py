"""SL10c — what the server knew about the held object and the support planes, for the constraint record.

`SafePolicy._record` hands this module's output to `ConstraintRecordWriter.record(held_record=)` when the writer is in
`full` mode. **Record only**: every function here reads; none writes into the solver, the linearizer, the pipeline or
the scene. The one thing that is computed (`attached_states` of the solver's own linearizer) is a pure forward
kinematics call on copies.

Why these keys (SL10 J1, `outputs/2026-11-05_SL10_제약기록_J0J1_계기근거/README.md`): while the hand holds the bowl the server
HOLDs mainly because the held spheres (`attached:panda_hand[i]`, `why = attached_point`) overlap the bowl's own occupied
voxels, although `held_free` frees the held volume. To tell *which side is wrong* one needs, per call: the spheres that
were freed, the spheres that were queried, the definition of the held spheres in the parent frame, and the support
planes. Frames and units are written next to each key in `constraint_record.HELD_RECORD_KEYS` and in
`AG3S/ag3s/docs/handoff/SL10c.rec.impl.md`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

__all__ = ["build_held_record"]


def _plain(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}


def build_held_record(*, constraint_set, snapshot, linearizer, trajectory, q_now, held_capture=None,
                      held_free: Optional[dict[str, Any]] = None,
                      finger_plan: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """`{"arrays": {key: ndarray}, "summary": dict}` for `ConstraintRecordWriter.record(held_record=)`.

    Args:
        constraint_set: the frame's `CollisionConstraintSet` (or None) — `support_surfaces` → `support_planes`.
        snapshot: the `SceneSnapshot` the solver saw (`attached_points` / `attached_radii` / `attached_parent_link`).
        linearizer: the **solver's** `CollisionLinearizer` (the one that has `set_attached`), for the query spheres.
        trajectory: `(nq_opt, H)` the refined plan (`TrajOptResult.trajectory`) or None.
        q_now: `(nq_model,)` the state the plan starts from or None.
        held_capture: `AG3S.held_capture_spheres` (`(C, 4)` base) or None.
        held_free: `AG3S.held_free_record` (the stash of `_held_free_points`) or None.
        finger_plan: `SafePolicy._last_finger["plan"]` (`{"opening_norm": {hand: [H values]}}`) or None.

    Keys are present only when the thing exists (no held object → no `held_*` / `attached_*` keys), except
    `support_planes`, which is always written (an empty `(0, 4)` says "no plane", absence would say "not recorded").
    """
    from AG3S.trajopt.linearize import support_planes_of

    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}

    arrays["support_planes"] = np.asarray(support_planes_of(constraint_set), np.float64).reshape(-1, 4)
    summary["n_support_planes"] = int(arrays["support_planes"].shape[0])

    if held_free is not None:
        arrays["held_free_spheres"] = np.asarray(held_free["spheres"], np.float64).reshape(-1, 4)
        arrays["held_free_first_frame"] = np.asarray(held_free["first"], bool).reshape(-1)
        summary["held_free"] = _plain(held_free)
    else:
        summary["held_free"] = None

    cap = None if held_capture is None else np.asarray(held_capture, np.float64).reshape(-1, 4)
    if cap is not None and cap.shape[0]:
        arrays["held_capture_spheres"] = cap

    points = getattr(snapshot, "attached_points", None)
    if points is not None and len(points):
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        radii = getattr(snapshot, "attached_radii", None)
        radii = np.zeros(len(pts)) if radii is None else np.asarray(radii, np.float64).reshape(-1)
        parent = str(getattr(snapshot, "attached_parent_link", ""))
        arrays["held_local_spheres"] = np.column_stack([pts, radii])
        arrays["attached_parent_link"] = np.asarray(parent)
        summary["attached"] = {"parent_link": parent, "n": int(len(pts)),
                               "radii_given": getattr(snapshot, "attached_radii", None) is not None}
        _query_spheres(arrays, summary, linearizer, trajectory, q_now, radii, len(pts), parent)
    else:
        summary["attached"] = None

    opening = (finger_plan or {}).get("opening_norm") if isinstance(finger_plan, dict) else None
    if opening:
        hands = sorted(str(h) for h in opening)
        arrays["finger_plan_hands"] = np.asarray(hands)
        arrays["finger_plan_opening"] = np.asarray([np.asarray(opening[h], np.float64) for h in hands]).reshape(len(hands), -1)
    return {"arrays": arrays, "summary": summary}


def _query_spheres(arrays, summary, linearizer, trajectory, q_now, radii, n, parent) -> None:
    """`attached_query_*`: where the solver's linearizer put the held spheres (base frame).

    `attached_query_spheres` is at `q_now` — the state the freed volume (`held_free_spheres`) was computed at — so the two
    can be laid on each other. `attached_query_centres` is along the refined plan, i.e. what the TO queried.
    Both from `linearizer.attached_states`, which is a forward-kinematics call (no state of the solver is written, apart from
    the lazily built probe cache the solver itself builds on its first solve).
    """
    note = None
    if linearizer is None or trajectory is None or q_now is None:
        note = "no solver linearizer / trajectory / q_now"
    elif int(getattr(linearizer, "n_attached", 0)) != n:
        note = (f"the solver's linearizer holds {int(getattr(linearizer, 'n_attached', 0))} query points, "
                f"the scene {n}")
    elif str(getattr(linearizer, "_attached_link", parent)) != parent:
        note = f"the solver's linearizer holds link {linearizer._attached_link!r}, the scene {parent!r}"  # noqa: SLF001
    if note is None:
        try:
            layout = linearizer.layout
            q = np.asarray(q_now, np.float64).reshape(-1)
            traj = np.asarray(trajectory, np.float64)
            plan_pos = np.asarray(linearizer.attached_states(layout.full_q(traj, q))[0], np.float64)
            stay = np.tile(q[layout.q_indices].reshape(-1, 1), (1, traj.shape[1]))
            now_pos = np.asarray(linearizer.attached_states(layout.full_q(stay, q))[0], np.float64)
            arrays["attached_query_centres"] = plan_pos
            arrays["attached_query_spheres"] = np.column_stack([now_pos[0], radii])
        except Exception as exc:  # noqa: BLE001 — a record must not kill the policy; say why instead
            note = f"{type(exc).__name__}: {exc}"
    summary["attached_query"] = {"available": note is None, **({} if note is None else {"why_not": note})}
