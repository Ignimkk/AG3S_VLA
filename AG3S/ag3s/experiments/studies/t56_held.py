"""T56 — the HELD event: from the serving chunk records (T53) or an offline `grasp_latch` run (dev).

`T56.task.md` 사전 고정 2: S1 / S2 are triggered by the `grasp_latch` **HELD** state — what a real robot
has (execution feedback), not the simulator's contact predicate (S0's ``both_contact``). HELD is a
trigger, never a certification: an empty grasp or a wrong object can be HELD too
(`t56_selfscore.label_check` counts how often the label it produces is not the GT target).

Two sources, recorded per record (``source``):

``server_chunk``   the T53 rollout server's own latch, per chunk ``summary_json.grasp`` (`safe_policy`
                   ``_last_grasp_record``: ``state`` after the call, ``attach`` / ``detach``,
                   ``attach_revoked``, ``held_slip``, ``hand``). Chunk i ↔ frame i (checked by the
                   caller via the T54 pairing). Event = the first frame whose call attached and left the
                   latch ``held``.
``offline_latch``  dev records carry no latch record (recorded without the safe server). The latch is
                   run offline on the **recorded execution facts** of each step record, per hand,
                   with `grasp_latch.GraspLatch(evidence=True)` and `grasp_signal_from_feedback` unchanged:

                   * executed feedback of chunk i−1 = the first ``n`` rows of step i−1's ``actions``
                     gripper columns (`wire.gripper_columns`: 7, 15), every step executed (no HOLD in
                     these records), ``n`` = t_step(i) − t_step(i−1) capped at the chunk length;
                   * measured opening at request i = step i's ``state`` gripper columns (the client's
                     `rby1_state()` convention, ``|q| / |RBY1_GRIPPER_OPEN|``);
                   * ``reach`` = distance from the hand's grasp parent-link origin (`t56_replay`
                     ``parent_world``) to the **manipulated centroid** = the chosen cluster centroid of a
                     reference candidate (default ``last:L8H2``, the serving cell) at the last frame
                     before i where it named a target (``visible`` at i−1, ``occluded`` before, none →
                     the check fails, as in serving). ``reach_source`` is recorded; ``not_checked``
                     (reach / manipulated treated as passed) is available as an explicit option;
                   * the latch is put in LATCHED at frame 0 (the target is assumed confirmed; recorded);
                     ``placed`` = False (no success oracle); ``held_observation`` = None (the T34
                     observation checks are not run; the opening checks are).
                   The hand whose HELD comes first is the event hand (both hands are recorded).

T55 re-simulations (optional, `resim_checks`): the HELD label error on the natural (``E0``) and the
intervened (``empty`` / ``slip``) re-simulations, from the physical state at the first HELD request
(the re-simulation keeps no cloud) — reported next to the T53 natural cases, never used to choose.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Any, Mapping, Optional, Sequence

import numpy as np

HANDS = ("left", "right")
HELD = "held"
REACH_SOURCES = ("reference_candidate", "not_checked")


@dataclasses.dataclass
class HeldEvent:
    source: str
    hand: Optional[str]
    pos: Optional[int]  # record frame index of the event (None = no event)
    states: list  # latch state per entry of `frames` (after that frame's call)
    exit_pos: Optional[int] = None  # first record frame after `pos` that is not HELD
    exit_reason: Optional[str] = None  # release / slip / revoked / other:<state> / None (held to the end)
    details: dict = dataclasses.field(default_factory=dict)
    frames: list = dataclasses.field(default_factory=list)  # record frame index of each state

    def state_at(self, frame: int) -> Optional[str]:
        fr = self.frames or list(range(len(self.states)))
        try:
            return self.states[fr.index(int(frame))]
        except ValueError:
            return None

    @property
    def has_event(self) -> bool:
        return self.pos is not None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def _exit_after(states: Sequence[str], pos: int) -> tuple[Optional[int], Optional[str]]:
    """First position after `pos` (positions into `states`) whose state is not HELD."""
    for j in range(pos + 1, len(states)):
        if states[j] != HELD:
            return j, None
    return None, None


# ------------------------------------------------------------------------------ server


def from_summaries(summaries: Sequence[Mapping[str, Any]]) -> HeldEvent:
    """HELD event of the serving latch from the per-chunk ``summary_json`` (chunk i = position i)."""
    states, hand, pos, info = [], None, None, []
    for i, s in enumerate(summaries):
        g = (s or {}).get("grasp") or {}
        st = str(g.get("state")) if g.get("state") is not None else "none"
        states.append(st)
        info.append({"attach": bool(g.get("attach", False)), "detach": bool(g.get("detach", False)),
                     "revoked": g.get("attach_revoked") is not None,
                     "slip": bool(g.get("held_slip", False)), "hand": g.get("hand"),
                     "rejected": g.get("rejected") is not None})
        if pos is None and info[-1]["attach"] and st == HELD:
            pos, hand = i, (str(g["hand"]) if g.get("hand") is not None else None)
    ev = HeldEvent("server_chunk", hand, pos, states, frames=list(range(len(states))))
    if pos is not None:
        ex, _ = _exit_after(states, pos)
        ev.exit_pos = ex
        if ex is not None:
            it = info[ex]
            ev.exit_reason = ("revoked" if it["revoked"] else "slip" if it["slip"]
                              else "release" if it["detach"] else f"other:{states[ex]}")
    ev.details = {"n_attach": int(sum(x["attach"] for x in info)),
                  "n_held_frames": int(sum(s == HELD for s in states)),
                  "n_rejected": int(sum(x["rejected"] for x in info)),
                  "hands_seen": sorted({str(x["hand"]) for x in info if x["hand"] is not None})}
    return ev


def from_chunks(chunks: Sequence[pathlib.Path]) -> HeldEvent:
    from AG3S.ag3s.experiments.studies.t54_replay import read_summary

    return from_summaries([read_summary(pathlib.Path(c)) for c in chunks])


# ------------------------------------------------------------------------------ offline


@dataclasses.dataclass(frozen=True)
class OfflineLatchConfig:
    """Offline latch knobs (recorded in the prereg). `slip_detach` follows the T53 server
    (``--held-slip-detach`` on, `T53` grasp records carry ``held_slip``)."""

    reach_source: str = "reference_candidate"
    reference_candidate: str = "last:L8H2"
    reference_condition: str = "hold"
    slip_detach: bool = True

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def manipulated_track(centroids: Sequence[Optional[np.ndarray]]) -> list[tuple[Optional[np.ndarray], Optional[str]]]:
    """Per position i: the manipulated geometry the serving latch would see **before** frame i's
    grounding: (last centroid named at a position < i, ``visible`` if named at i−1 else ``occluded``),
    or (None, None) when nothing was named yet."""
    out, last, last_pos = [], None, None
    for i in range(len(centroids)):
        if last is None:
            out.append((None, None))
        else:
            out.append((last, "visible" if last_pos == i - 1 else "occluded"))
        c = centroids[i]
        if c is not None and np.all(np.isfinite(c)):
            last, last_pos = np.asarray(c, np.float64), i
    return out


def offline_latch(*, t_step: Sequence[int], state: np.ndarray, actions: np.ndarray,
                  parent_world: np.ndarray, manip_world: Sequence[Optional[np.ndarray]],
                  cfg: OfflineLatchConfig = OfflineLatchConfig(),
                  frames: Optional[Sequence[int]] = None) -> HeldEvent:
    """Run `GraspLatch(evidence=True)` per hand on recorded execution facts (module doc).

    `state` (F, 16), `actions` (F, H, 16), `parent_world` (F, 2, 3) per hand, `manip_world` per position:
    the reference candidate's chosen centroid in world (None when it named nothing). `frames` = the
    record frame index of each row (default 0..F−1); the event / exit are returned as frame indices."""
    from AG3S.trajopt import wire
    from AG3S.trajopt.grasp_latch import GraspLatch, LatchConfig, grasp_signal_from_feedback

    if cfg.reach_source not in REACH_SOURCES:
        raise ValueError(f"reach_source {cfg.reach_source!r} not in {REACH_SOURCES}")
    ts = [int(t) for t in t_step]
    st = np.asarray(state, np.float64)
    act = np.asarray(actions, np.float64)
    n_f = len(ts)
    fr = list(range(n_f)) if frames is None else [int(i) for i in frames]
    if len(fr) != n_f:
        raise ValueError("frames / t_step lengths differ")
    if st.shape[0] != n_f or act.shape[0] != n_f:
        raise ValueError("state / actions frame counts differ")
    cols = wire.gripper_columns(int(st.shape[1]) // 2 - 1)
    track = manipulated_track(list(manip_world))
    per_hand: dict[str, dict] = {}
    for hi, hand in enumerate(HANDS):
        lcfg = LatchConfig(slip_detach=bool(cfg.slip_detach))
        latch = GraspLatch(lcfg, evidence=True)
        for _ in range(lcfg.confirm_frames):
            latch.update(label="target", score=1.0, runner_up=0.0, gripper=1.0)
        states = [latch.phase.value]
        notes: list = [None]
        attach_pos, flags = None, [{}]
        for i in range(1, n_f):
            n = max(0, min(ts[i] - ts[i - 1], act.shape[1]))
            applied = act[i - 1, :n][:, list(cols)]
            fb = {"available": True, "seq": i - 1, "executed": [True] * n,
                  "applied_gripper": applied, "measured_gripper": st[i, list(cols)]}
            sig = grasp_signal_from_feedback(fb, hi, lcfg.gripper_closed_below)
            centroid, mstate = track[i]
            if cfg.reach_source == "not_checked":
                reach, mid, mst = 0.0, 1, "visible"
            elif centroid is None:
                reach, mid, mst = None, None, None
            else:
                reach = float(np.linalg.norm(np.asarray(parent_world[i, hi], np.float64) - centroid))
                mid, mst = 1, mstate
            ev = latch.observe_grasp(sig, reach_m=reach, manipulated_id=mid, manipulated_state=mst,
                                     placed=False, held_observation=None)
            states.append(latch.phase.value)
            notes.append(ev.note)
            flags.append({"attach": bool(ev.attach), "detach": bool(ev.detach),
                          "revoked": bool(ev.revoked), "slipped": bool(ev.slipped)})
            if attach_pos is None and ev.attach and latch.phase.value == HELD:
                attach_pos = i
        per_hand[hand] = {"states": states, "pos": attach_pos, "flags": flags,
                          "notes": [n for n in notes if n]}
    cand = [(per_hand[h]["pos"], k, h) for k, h in enumerate(HANDS) if per_hand[h]["pos"] is not None]
    if not cand:
        return HeldEvent("offline_latch", None, None, per_hand["left"]["states"], frames=fr,
                         details={"config": cfg.as_dict(), "per_hand": {
                             h: {"pos": None, "n_held": int(sum(s == HELD for s in per_hand[h]["states"]))}
                             for h in HANDS}})
    pos, _, hand = min(cand)
    states = per_hand[hand]["states"]
    ev = HeldEvent("offline_latch", hand, fr[pos], states, frames=fr)
    ex, _ = _exit_after(states, pos)
    ev.exit_pos = None if ex is None else fr[ex]
    if ex is not None:
        f = per_hand[hand]["flags"][ex]
        ev.exit_reason = ("revoked" if f.get("revoked") else "slip" if f.get("slipped")
                          else "release" if f.get("detach") else f"other:{states[ex]}")
    ev.details = {"config": cfg.as_dict(),
                  "assumptions": ["latch LATCHED at frame 0 (target confirmed)", "placed=False",
                                  "held_observation=None (T34 observation checks not run)",
                                  "all recorded steps executed (no HOLD)"],
                  "per_hand": {h: {"pos": None if per_hand[h]["pos"] is None else fr[per_hand[h]["pos"]],
                                   "n_held": int(sum(s == HELD for s in per_hand[h]["states"]))}
                               for h in HANDS}}
    return ev


def reference_centroids(geom, cand: str) -> list[Optional[np.ndarray]]:
    """World centroid of `cand`'s chosen cluster per position (None when it named nothing)."""
    out = []
    Twb = np.asarray(geom.z["T_world_base"], np.float64)
    for j, i in enumerate(geom.frames):
        p = geom.chosen_points(cand, i)
        if not len(p):
            out.append(None)
            continue
        c = p.mean(axis=0)
        out.append(Twb[j, :3, :3] @ c + Twb[j, :3, 3])
    return out


def write_event(path: pathlib.Path, ev: HeldEvent) -> None:
    from AG3S.ag3s.experiments.studies.t47b_metrics import jsonable

    pathlib.Path(path).write_text(json.dumps(jsonable(ev.as_dict()), indent=1))


def read_event(path: pathlib.Path) -> HeldEvent:
    d = json.loads(pathlib.Path(path).read_text())
    return HeldEvent(**d)


# ------------------------------------------------------------------------------ T55 re-simulation


RESIM_MODES = ("record", "oracle")
RESIM_VERDICTS = ("correct", "wrong_object", "empty")


def first_held(phase: Sequence[str], attach: Optional[Sequence[bool]] = None) -> Optional[int]:
    """First request whose latch state is HELD after an attach (``attach`` given) or after a non-HELD
    state — the trigger S1 would use."""
    ph = [str(x) for x in phase]
    for i, st in enumerate(ph):
        if st != HELD:
            continue
        if attach is not None:
            if bool(attach[i]):
                return i
        elif i == 0 or ph[i - 1] != HELD:
            return i
    return None


def resim_held_check(z: Mapping[str, Any], modes: Sequence[str] = RESIM_MODES) -> dict:
    """HELD-event label check on one T55 re-simulation npz (`t55_resim.variant_payload`).

    The re-simulation stores no point cloud, so the check is the **physical** state at the HELD request
    (`t48b_scene.classify` per fruit, the camera-seen pose): which fruit the used hand holds (both
    fingers on it, off the table / crate). verdict ``correct`` = the target, ``wrong_object`` = another
    fruit, ``empty`` = none. A HELD on an empty / wrong grasp would put the S1 ball on something that is
    not the target — counted as a label error (HELD is a trigger, not a certification). The episode's
    **first** HELD is used (what S1 uses), with ``after_intervention`` = at or after the variant's
    ``window_start``."""
    fruits = [str(f) for f in np.asarray(z["fruits"]).reshape(-1)]
    target = str(np.asarray(z["target"]))
    arm = str(np.asarray(z["used_arm"]))
    st = np.asarray(z["phys_state"])
    hd = np.asarray(z["phys_hand"])
    ws = int(np.asarray(z.get("window_start", -1)))
    out: dict[str, Any] = {"key": str(np.asarray(z.get("key", ""))), "tag": str(np.asarray(z.get("tag", ""))),
                           "target": target, "used_arm": arm, "window_start": None if ws < 0 else ws,
                           "check": "physical state at the HELD request (no cloud in the re-simulation)",
                           "modes": {}}
    for mode in modes:
        kp = f"latch_{mode}__phase"
        if kp not in z:
            continue
        att = z.get(f"latch_{mode}__attach")
        pos = first_held(list(np.asarray(z[kp])), None if att is None else list(np.asarray(att)))
        if pos is None:
            out["modes"][mode] = {"has_event": False}
            continue
        held = [f for j, f in enumerate(fruits) if str(st[pos, j]) == "held" and str(hd[pos, j]) == arm]
        verdict = "correct" if target in held else ("wrong_object" if held else "empty")
        out["modes"][mode] = {"has_event": True, "pos": int(pos), "held_fruits": held, "verdict": verdict,
                              "label_correct": verdict == "correct",
                              "after_intervention": (None if ws < 0 else bool(pos >= ws))}
    return out


def resim_variant_group(tag: str) -> str:
    """``E0`` / ``empty`` / ``slip`` / other prefix of a T55 variant tag."""
    t = str(tag)
    return "E0" if t == "E0" else t.split("_", 1)[0]


def resim_summary(checks: Sequence[Mapping[str, Any]], modes: Sequence[str] = RESIM_MODES) -> dict:
    """Label-error rate of the HELD events per latch mode and variant group (and pooled)."""
    out: dict[str, Any] = {}
    for mode in modes:
        groups: dict[str, dict] = {}
        for c in checks:
            m = c.get("modes", {}).get(mode)
            if m is None:
                continue
            for g in (resim_variant_group(c.get("tag", "")), "all"):
                r = groups.setdefault(g, {"n_runs": 0, "n_held_events": 0, "n_label_wrong": 0,
                                          "verdicts": {}, "wrong": []})
                r["n_runs"] += 1
                if not m.get("has_event"):
                    continue
                r["n_held_events"] += 1
                r["verdicts"][m["verdict"]] = r["verdicts"].get(m["verdict"], 0) + 1
                if not m["label_correct"]:
                    r["n_label_wrong"] += 1
                    r["wrong"].append(f"{c.get('key')}__{c.get('tag')}")
        for r in groups.values():
            r["held_label_error_rate"] = (r["n_label_wrong"] / r["n_held_events"]) if r["n_held_events"] else None
        if groups:
            out[mode] = groups
    return out


def resim_checks(resim_dir: pathlib.Path, keys: Optional[Sequence[str]] = None) -> dict:
    """`resim_held_check` for every ``<key>__<tag>.npz`` of a T55 resim set directory (read only)."""
    rd = pathlib.Path(resim_dir)
    if not rd.is_dir():
        raise SystemExit(f"no T55 resim directory {rd}")
    checks = []
    for p in sorted(rd.glob("*__*.npz")):
        if p.name.endswith(".partial.npz"):
            continue
        key = p.name.split("__", 1)[0]
        if keys is not None and key not in keys:
            continue
        with np.load(p, allow_pickle=False) as z:
            need = [k for k in ("fruits", "target", "used_arm", "phys_state", "phys_hand") if k not in z.files]
            if need:
                raise SystemExit(f"{p}: not a T55 resim npz (missing {need})")
            zz = {k: z[k] for k in z.files if k in ("fruits", "target", "used_arm", "phys_state", "phys_hand",
                                                     "window_start", "key", "tag")
                  or k.startswith("latch_")}
        c = resim_held_check(zz)
        c["path"] = str(p)
        checks.append(c)
    return {"dir": str(rd), "n_runs": len(checks), "summary": resim_summary(checks), "runs": checks}


__all__ = ["RESIM_MODES", "RESIM_VERDICTS", "first_held", "resim_checks", "resim_held_check",
           "resim_summary", "resim_variant_group", "HELD", "HANDS", "HeldEvent", "OfflineLatchConfig", "REACH_SOURCES", "from_chunks",
           "from_summaries", "manipulated_track", "offline_latch", "read_event", "reference_centroids",
           "write_event"]
