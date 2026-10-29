"""T54b — the ``hold_attach`` replay condition: T54's hold, plus the recorded grasp latch · attach · detach.

`T54b.task.md`. T54's ``hold`` (`t54_replay.run_frames`) injected only the latch *flags* and never replayed
`attach` / `detach` (`T54.verify.json` deviation 5). This module adds a third condition and leaves the
other two untouched (`t54_replay` is not modified; ``hold`` stays available for T54 reproducibility):

``hold_attach``  the episode in time order, `TargetConfirm` on, **one AG3S per arm** (each arm accumulates
                 its own grounding history — nothing of one arm reaches another). Per frame, every arm gets
                 the **same recorded** serving inputs, in `SafePolicy._scene_fn` order:

                 1. finger joints from the replayed scene (as T54, `t47b_replay._ground`);
                 2. the grasp event of that chunk's record (`grasp.attach` / `grasp.detach`, evidence mode):
                    attach = `AG3S.attach(<this arm's own manipulated geometry>, robot_state=<newest
                    observation q>, parent_link, allowed_contact_links, label="manipulated", reach,
                    hold_link)` — the call `SafePolicy._advance_grasp` makes, with the recorded hand's
                    links and the recorded reach limit; `AttachRejected` (no usable manipulated object,
                    reach, admissibility) is caught and recorded per arm, as serving reverts the latch.
                    detach = `AG3S.detach()`;
                 3. the latch flags of T54 (`t54_replay.apply_freeze`: `set_grasp_active`,
                    `set_grasp_latched`, `set_subtask`, `set_placed`) — the freeze state;
                 4. `process_multi_debug`.

                 The **timing and parameters** of every attach / detach are the recorded ones and identical
                 for all arms; the **geometry** attached is each arm's own `AG3S.manipulated` (no arm receives
                 the serving L8H2 geometry). An arm whose own manipulated object is unusable / out of reach
                 is therefore not attached at that frame (recorded as ``attach_result = rejected``);
                 ``attach_reach="off"`` (study option) skips the reach gate (still the admissibility gate).

Fidelity (`fidelity_fields`, per arm and frame; the verifier reports the L8H2 arm): the replayed grounding
against the serving chunk record of the same frame — target present, the chosen target (identity by GT
labels of the serving target points vs the replayed target points, both labelled by the same frame's GT
labeller; centroid distance ≤ `SAME_OBJECT_M` = same object), the manipulated object (centroid distance
≤ `SAME_OBJECT_M`), attached, frozen. `fidelity_summary` aggregates (frames, episode means, by phase).

Limitations (also in ``T54b.impl.md``): the serving call's ``execution_path`` / ``active_manipulators`` and
the cuRobo field are not replayed (T54 profile c1a, legacy ESDF); the replayed trajectory is the serving
L8H2 one for every arm.

CPU only. Run by the verifier.
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t54_arms as arms_mod
from AG3S.ag3s.experiments.studies import t54_replay as rp

CONDITION = "hold_attach"
#: Default reach (m) when the record has no ``reach_limit_mm`` (`trajopt.grasp_latch.LatchConfig.reach`).
DEFAULT_REACH_M = 0.12
ATTACH_REACH = ("serving", "off")
#: Two centroids within this distance are the same object (fidelity only; fruit ≈ 50–90 mm).
SAME_OBJECT_M = 0.03
LATCH_STATES = ("searching", "latched", "closing", "held", "placed")


# ------------------------------------------------------------------------- recorded events


def grasp_links_of(hand: Optional[str]) -> tuple[Optional[str], tuple, Optional[str]]:
    """``(parent_link, allowed_contact_links, hold_link)`` of `SafePolicy`'s defaults for a hand."""
    from AG3S.trajopt.safe_policy import DEFAULT_GRASP_LINKS, DEFAULT_HOLD_LINKS

    if hand not in DEFAULT_GRASP_LINKS:
        return None, (), None
    parent, allowed = DEFAULT_GRASP_LINKS[hand]
    return str(parent), tuple(str(x) for x in allowed), str(DEFAULT_HOLD_LINKS[hand])


def serving_ref(summary: Mapping[str, Any]) -> dict:
    """What the serving AG3S ended this chunk with (the fidelity reference; no arrays)."""
    m = summary.get("manipulated") or {}
    ex = summary.get("exclusion") or {}
    dec = (summary.get("admissibility") or {}).get("decision") or {}
    c = m.get("centroid")
    return {"has_target": bool(summary.get("has_target")),
            "target_n_points": summary.get("target_n_points"),
            "manip_id": m.get("id"), "manip_state": m.get("state"),
            "manip_centroid": None if c is None else [float(x) for x in c],
            "attached": ex.get("attached_label") is not None,
            "frozen": dec.get("frozen"), "decision_mode": dec.get("mode")}


def grasp_event(summary: Mapping[str, Any], chunk: Optional[pathlib.Path] = None) -> dict:
    """One chunk's recorded serving inputs: T54's freeze (`t54_replay.freeze_inputs`) + latch state,
    attach / detach and their parameters + the serving reference (`serving_ref`)."""
    f = rp.freeze_inputs(summary)
    g = summary.get("grasp") or {}
    ex = summary.get("exclusion") or {}
    hand = g.get("hand")
    parent_d, allowed, hold_d = grasp_links_of(hand)
    parent = g.get("parent_link") or parent_d
    hold = g.get("hold_link") or hold_d
    reach_mm = (g.get("evidence") or {}).get("reach_limit_mm")
    auth = ex.get("authorized_links")
    links_consistent = (None if not auth or hand is None
                        else set(allowed) | ({hold} if hold else set()) == set(auth))
    return {**f,
            "latch_state": g.get("state"), "latch_state_before": g.get("state_before"),
            "attach": bool(g.get("attach")) and g.get("rejected") is None,
            "detach": bool(g.get("detach")),
            "serving_attach_rejected": g.get("rejected") is not None,
            "attach_revoked": bool(g.get("attach_revoked")), "held_slip": bool(g.get("held_slip")),
            "grasp_mode": g.get("mode"), "hand": hand, "parent_link": parent, "hold_link": hold,
            "allowed_contact_links": list(allowed),
            "reach_m": DEFAULT_REACH_M if reach_mm is None else float(reach_mm) / 1000.0,
            "links_consistent": links_consistent, "serving": serving_ref(summary),
            "chunk": None if chunk is None else str(chunk)}


def events_from_chunks(chunks: Sequence[pathlib.Path]) -> list[dict]:
    return [grasp_event(rp.read_summary(pathlib.Path(c)), pathlib.Path(c)) for c in chunks]


def event_problems(events: Sequence[Mapping[str, Any]]) -> list[str]:
    """Records the replay cannot reproduce faithfully (refused by `run_frames_attach`)."""
    out = []
    for i, e in enumerate(events):
        if (e["attach"] or e["detach"]) and e.get("grasp_mode") != "evidence":
            out.append(f"frame {i}: grasp mode {e.get('grasp_mode')!r} (only 'evidence' attaches before "
                       "the call; a legacy attach happens after it)")
        if e["attach"] and (e.get("parent_link") is None or e.get("hand") is None):
            out.append(f"frame {i}: attach without a hand / parent link")
        if e["attach"] and e["detach"]:
            out.append(f"frame {i}: attach and detach in one chunk")
    return out


def events_summary(events: Sequence[Mapping[str, Any]]) -> dict:
    states: dict[str, int] = {}
    for e in events:
        states[str(e.get("latch_state"))] = states.get(str(e.get("latch_state")), 0) + 1
    return {**rp.freeze_summary(events),
            "latch_state_frames": dict(sorted(states.items())),
            "attach_frames": [i for i, e in enumerate(events) if e["attach"]],
            "detach_frames": [i for i, e in enumerate(events) if e["detach"]],
            "serving_attach_rejected": int(sum(e["serving_attach_rejected"] for e in events)),
            "attach_revoked": int(sum(e["attach_revoked"] for e in events)),
            "held_slip": int(sum(e["held_slip"] for e in events)),
            "links_inconsistent": int(sum(e["links_consistent"] is False for e in events)),
            "serving_attached_frames": int(sum(bool(e["serving"]["attached"]) for e in events)),
            "problems": event_problems(events)}


# ------------------------------------------------------------------------------ injection


def newest_q(observations: Sequence[Any]) -> np.ndarray:
    return np.asarray(max(observations, key=lambda o: o.timestamp).robot_state, np.float64)


def inject_grasp(ag, ev: Optional[Mapping[str, Any]], observations: Sequence[Any], *,
                 attach_reach: str = "serving") -> dict:
    """Replay one recorded attach / detach on one arm's AG3S (`SafePolicy._advance_grasp`). Returns
    what happened on this arm; `AttachRejected` is recorded, any other error propagates."""
    from AG3S.ag3s.constraints.attached import AttachRejected

    if attach_reach not in ATTACH_REACH:
        raise ValueError(f"attach_reach {attach_reach!r} not in {ATTACH_REACH}")
    out: dict[str, Any] = {"attach_called": False, "detach_called": False, "attach_result": None}
    if ev is None:
        return out
    if ev["attach"]:
        manip = getattr(ag, "manipulated", None)
        usable = bool(manip is not None and getattr(manip, "usable", False))
        out.update({"attach_called": True,
                    "manip_id_at_attach": None if manip is None else int(manip.id),
                    "manip_state_at_attach": None if manip is None else str(manip.state)})
        try:
            if not usable:
                raise AttachRejected("manipulated object has no usable geometry",
                                     parent_link=ev["parent_link"])
            ag.attach(manip.geometry, robot_state=newest_q(observations),
                      parent_link=ev["parent_link"],
                      allowed_contact_links=tuple(ev["allowed_contact_links"]), label="manipulated",
                      reach=float(ev["reach_m"]) if attach_reach == "serving" else None,
                      hold_link=ev["hold_link"])
            out["attach_result"] = "attached"
        except AttachRejected as exc:
            out["attach_result"] = "rejected"
            out["attach_rejected"] = exc.record()
    elif ev["detach"]:
        released = ag.detach()
        out.update({"detach_called": True, "released": released is not None})
    return out


def mujoco_prepare(ag, fr: dict, att_maps: dict) -> tuple[list, str]:
    """`t47b_replay._ground` up to (not including) the call: fingers from the scene, then unpack."""
    from AG3S.ag3s.experiments.sources.mujoco_source import set_finger_joints_from_scene
    from AG3S.trajopt import wire

    set_finger_joints_from_scene(fr["scene"], ag.robot_model)
    if ag.constraint_robot_model is not None:
        set_finger_joints_from_scene(fr["scene"], ag.constraint_robot_model)
    obs = wire.unpack_camera_observations(fr["sc"], robot_model=ag.robot_model, attention=att_maps)
    return obs, fr["phase"]


def serving_order_ground(ev: Optional[Mapping[str, Any]], *, prepare: Callable = mujoco_prepare,
                         attach_reach: str = "serving", box: Optional[dict] = None) -> Callable:
    """A `ground_fn` for `t54_replay.ground_arm`: prepare → grasp event → latch flags → call."""
    box = {} if box is None else box

    def ground(ag, fr, att):
        obs, phase = prepare(ag, fr, att)
        box["grasp"] = inject_grasp(ag, ev, obs, attach_reach=attach_reach)
        rp.apply_freeze(ag, ev)
        t0 = time.perf_counter()
        cs, debug = ag.process_multi_debug(obs, phase=phase)
        ms = (time.perf_counter() - t0) * 1000.0
        box["cs"] = cs
        return cs, debug, ms
    return ground


# ------------------------------------------------------------------------------- fidelity


def load_serving_arrays(ev: Optional[Mapping[str, Any]]) -> dict:
    """The serving chunk's chosen target points / centroid (empty when it had none)."""
    out = {"target_points": np.zeros((0, 3)), "target_centroid": None}
    if ev is None or not ev.get("chunk"):
        return out
    with np.load(ev["chunk"], allow_pickle=False) as z:
        if "target_points" in z.files:
            out["target_points"] = np.asarray(z["target_points"], np.float64).reshape(-1, 3)
        if "target_centroid" in z.files and np.asarray(z["target_centroid"]).size == 3:
            out["target_centroid"] = np.asarray(z["target_centroid"], np.float64).reshape(3)
    return out


def _label(points, label_of) -> Optional[str]:
    from AG3S.ag3s.experiments.studies.t47b_metrics import class_counts, plurality

    pts = np.asarray(points, np.float64).reshape(-1, 3)
    return plurality(class_counts(label_of(pts))) if len(pts) else None


def _dist(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(np.linalg.norm(np.asarray(a, np.float64).reshape(3) - np.asarray(b, np.float64).reshape(3)))


def _same(has_a: bool, has_b: bool, d: Optional[float], tol: float) -> bool:
    if not has_a and not has_b:
        return True
    if has_a != has_b:
        return False
    return d is not None and d <= tol


def fidelity_fields(cs, ag, ev: Mapping[str, Any], srv_arrays: Mapping[str, Any], *, label_of,
                    gt_centre, decision_frozen: Optional[bool],
                    same_object_m: float = SAME_OBJECT_M) -> dict:
    """One arm's replayed frame against the serving record of that frame (module doc)."""
    srv = ev["serving"]
    rep_has = bool(cs.has_target)
    srv_has = bool(srv["has_target"])
    srv_pts = srv_arrays["target_points"]
    srv_lab = _label(srv_pts, label_of) if srv_has else None
    rep_lab = _label(cs.target.points, label_of) if rep_has else None
    srv_c = srv_arrays.get("target_centroid")
    if srv_c is None and srv_has and len(srv_pts):
        srv_c = srv_pts.mean(axis=0)
    rep_c = np.asarray(cs.target.centroid, np.float64) if rep_has else None
    d_chosen = _dist(srv_c, rep_c)
    m = getattr(ag, "manipulated", None)
    rep_mc = None if m is None else np.asarray(m.centroid, np.float64)
    srv_mc = srv["manip_centroid"]
    d_manip = _dist(srv_mc, rep_mc)
    rep_att = getattr(ag, "attached", None) is not None
    srv_frozen = srv["frozen"]
    return {
        "srv_has_target": srv_has, "rep_has_target": rep_has, "has_target_agree": srv_has == rep_has,
        "srv_identity_label": srv_lab, "rep_identity_label": rep_lab,
        "srv_identity_correct": srv_lab == "target", "rep_identity_correct": rep_lab == "target",
        "identity_agree": (srv_lab == "target") == (rep_lab == "target"),
        "identity_label_agree": srv_lab == rep_lab,
        "chosen_centroid_dist_m": d_chosen,
        "chosen_same_object": _same(srv_has, rep_has, d_chosen, same_object_m),
        "srv_manip_id": srv["manip_id"], "rep_manip_id": None if m is None else int(m.id),
        "srv_manip_state": srv["manip_state"], "rep_manip_state": None if m is None else str(m.state),
        "manip_centroid_dist_m": d_manip,
        "manip_same_object": _same(srv_mc is not None, rep_mc is not None, d_manip, same_object_m),
        "srv_manip_to_gt_target_m": _dist(srv_mc, gt_centre),
        "rep_manip_to_gt_target_m": _dist(rep_mc, gt_centre),
        "srv_attached": bool(srv["attached"]), "rep_attached": rep_att,
        "attached_agree": bool(srv["attached"]) == rep_att,
        "srv_frozen": srv_frozen, "rep_frozen": decision_frozen,
        "frozen_agree": (None if srv_frozen is None or decision_frozen is None
                         else bool(srv_frozen) == bool(decision_frozen)),
    }


AGREE_FIELDS = ("has_target_agree", "identity_agree", "identity_label_agree", "chosen_same_object",
                "manip_same_object", "attached_agree", "frozen_agree")


def fidelity_summary(rows: Mapping[str, Sequence[Mapping[str, Any]]], *, phase_key: str = "phase") -> dict:
    """``rows`` = ``{key: [hold_attach rows of one arm]}`` → agreement rates (frame-pooled and episode
    mean), per episode and per row phase; serving vs replay identity counts; attach outcomes."""
    def rates(rs):
        out = {}
        for f in AGREE_FIELDS:
            v = [bool(r["fidelity"][f]) for r in rs if r["fidelity"].get(f) is not None]
            out[f] = {"rate": (sum(v) / len(v)) if v else float("nan"), "n": len(v)}
        d = [r["fidelity"]["chosen_centroid_dist_m"] for r in rs
             if r["fidelity"]["chosen_centroid_dist_m"] is not None]
        out["chosen_centroid_dist_m_median"] = float(np.median(d)) if d else float("nan")
        out["srv_identity_correct"] = int(sum(bool(r["fidelity"]["srv_identity_correct"]) for r in rs))
        out["rep_identity_correct"] = int(sum(bool(r["fidelity"]["rep_identity_correct"]) for r in rs))
        out["n_frames"] = len(rs)
        return out

    per_ep = {k: rates(rs) for k, rs in rows.items()}
    pooled = rates([r for rs in rows.values() for r in rs])
    ep_mean = {f: float(np.nanmean([per_ep[k][f]["rate"] for k in per_ep])) if per_ep else float("nan")
               for f in AGREE_FIELDS}
    phases: dict[str, list] = {}
    for rs in rows.values():
        for r in rs:
            phases.setdefault(str(r.get(phase_key)), []).append(r)
    attach = {}
    for rs in rows.values():
        for r in rs:
            res = (r.get("attach_outcome") or {}).get("attach_result")
            if res is not None:
                attach[res] = attach.get(res, 0) + 1
    return {"pooled": pooled, "episode_mean": ep_mean, "per_episode": per_ep,
            "by_phase": {p: rates(rs) for p, rs in sorted(phases.items())},
            "attach_results": attach, "same_object_m": SAME_OBJECT_M,
            "definitions": (
                "agree = replay == serving per frame; identity via GT labels of the chosen points "
                "(strict plurality, unlabelled ignored); same object = both absent, or both present "
                f"and centroids within {SAME_OBJECT_M * 1000:.0f} mm")}


# ----------------------------------------------------------------------------- frame loop


def run_frames_attach(frames: Iterable[dict], *, n_f: int, maps: Mapping[str, np.ndarray],
                      arms: Sequence[arms_mod.ArmSpec], build: Callable[[], Any],
                      events: Sequence[Mapping[str, Any]], prepare: Callable = mujoco_prepare,
                      attach_reach: str = "serving",
                      serving_arrays: Callable[[Mapping[str, Any]], dict] = load_serving_arrays,
                      same_object_m: float = SAME_OBJECT_M,
                      log: Callable[[str], None] = print) -> dict[str, list[dict]]:
    """``{arm: [row per frame]}`` of the ``hold_attach`` condition (module doc)."""
    from AG3S.ag3s.experiments.studies import t47b_metrics as bm

    if len(events) != n_f:
        raise ValueError(f"{len(events)} recorded events for {n_f} frames")
    probs = event_problems(events)
    if probs:
        raise ValueError("records the hold_attach replay cannot reproduce:\n  " + "\n  ".join(probs))
    ags = {a.name: build() for a in arms}
    if len({id(x) for x in ags.values()}) != len(ags):
        raise ValueError("build() must return a new AG3S per arm")
    rows: dict[str, list[dict]] = {a.name: [] for a in arms}
    t0 = time.time()
    for fr in frames:
        i, step = fr["i"], fr["step"]
        ev = events[i]
        srv_arr = serving_arrays(ev)
        marks = []
        for a in arms:
            ag = ags[a.name]
            box: dict = {}
            r = rp.ground_arm(ag, fr, a, maps, label_of=fr["label_of"], gt_centre=fr["gt_centre"],
                              ground_fn=serving_order_ground(ev, prepare=prepare,
                                                             attach_reach=attach_reach, box=box))
            fid = fidelity_fields(box["cs"], ag, ev, srv_arr, label_of=fr["label_of"],
                                  gt_centre=fr["gt_centre"], decision_frozen=r.get("decision_frozen"),
                                  same_object_m=same_object_m)
            rows[a.name].append({
                "frame": i, "t_step": int(step.t_step), "phase": fr["phase"],
                "phase_group": bm.phase_group(fr["phase"]), "arm": a.name, "condition": CONDITION,
                "freeze_injected": True, "freeze": {k: ev[k] for k in rp.FREEZE_KEYS},
                "grasp_injected": {"latch_state": ev.get("latch_state"),
                                   "latch_state_before": ev.get("latch_state_before"),
                                   "attach": bool(ev["attach"]), "detach": bool(ev["detach"]),
                                   "attach_reach": attach_reach},
                "attach_outcome": box["grasp"], "fidelity": fid, **r})
            marks.append("I" if r["identity_correct"] else ("s" if r["has_target"] else "-"))
        log(f"  t54b {CONDITION} frame {i + 1}/{fr['n_f']} t={step.t_step} "
            f"{'A' if ev['attach'] else ('D' if ev['detach'] else ' ')} {''.join(marks)}  "
            f"({time.time() - t0:.0f}s)")
    return rows


def run_record_attach(record_path: pathlib.Path, *, target: str, maps: Mapping[str, np.ndarray],
                      arms: Sequence[arms_mod.ArmSpec], build: Callable[[], Any],
                      events: Sequence[Mapping[str, Any]], frames: Optional[Sequence[int]] = None,
                      attach_reach: str = "serving", label_radius: float = 0.003,
                      log: Callable[[str], None] = print) -> dict[str, list[dict]]:
    """`run_frames_attach` over the MuJoCo replay of one step record (`t47b_replay._frames`, osmesa)."""
    from AG3S.ag3s.experiments.sources.policy_record import POLICY_CAMERA_NAMES, load_run
    from AG3S.ag3s.experiments.studies import t47b_replay as br

    n_f = len(load_run(record_path))
    need = arms_mod.sources_needed(arms)
    br._check_sources({s: maps[s] for s in need}, n_f, len(POLICY_CAMERA_NAMES))
    return run_frames_attach(br._frames(record_path, target=target, frames=frames,
                                        label_radius=label_radius, log=log),
                             n_f=n_f, maps=maps, arms=arms, build=build, events=events,
                             attach_reach=attach_reach, log=log)


__all__ = ["AGREE_FIELDS", "ATTACH_REACH", "CONDITION", "DEFAULT_REACH_M", "SAME_OBJECT_M",
           "event_problems", "events_from_chunks", "events_summary", "fidelity_fields",
           "fidelity_summary", "grasp_event", "grasp_links_of", "inject_grasp", "load_serving_arrays",
           "mujoco_prepare", "run_frames_attach", "run_record_attach", "serving_order_ground",
           "serving_ref"]
