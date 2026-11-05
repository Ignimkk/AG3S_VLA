"""SL11 §5 -- offline replay of the Panda grasp latch on recorded calls, with the old (single finger) or the new (two-finger gap)
opening. An ESTIMATE: the recorded calls were produced by the old latch, so the arm/hand trajectory, the object and every later
input are the old closed loop's; a real run with the new latch decides. Read-only: nothing here changes the server or a record.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-ag3s/bin/python \\
        -m AG3S.trajopt.experiments.panda_latch_replay --run-dir <run> [--run-dir <run2> ...] --out ${AG3S_RUN_DIR}/replay.json

``<run>`` = a ``--record-constraints full`` vla_v11ab run (``episodes/<stem>_{calls.jsonl,calls.npz,geom.npz}``, ``<stem>.npz``,
``server_constraints/run_*/chunk_*.npz``). Per call the latch inputs are rebuilt from the record:

| latch input | where |
|---|---|
| ``GraspSignal`` (closing steps, last command, measured opening) | chunk ``summary_json.exec_feedback`` -> ``grasp_signal_from_feedback`` |
| new opening | ``(q1 - q2) / 0.08`` of ``<stem>_geom.npz`` ``robot_state/head[obs_index]`` (the request's ``panda/robot_state``) |
| reach, manipulated id / state, sticky-reach context | ``grasp.evidence.reach_mm`` / ``manipulated_*``, ``grasp.alignment`` |
| held observation (revoke window) | ``grasp.evidence.hold_checks`` (``observed_*``, ``hand_rise_mm``, ``object_rise_mm``) |
| HSR6 phase for attach-assist | the call row's ``hsr6.phase`` |
| ``placed`` | ``grasp.evidence.placed`` |

Not replayed: ``AG3S.attach`` rejection (recorded ``grasp.rejected``, counted), the destination latch. The SEARCHING -> LATCHED lock is
taken from the record (the replay forces LATCHED when the recorded latch is no longer searching).

Ground truth is the one of ``outputs/2026-11-05_쥔물체_attach_거리장제거_점검/scripts/held_check.py``: the target is the object
lifted highest during the episode; a call is "in hand" when it is more than 3 cm above its first height and within 10 cm of the EE.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import pathlib
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.trajopt.grasp_latch import GraspLatch, GraspPhase, LatchConfig, grasp_signal_from_feedback

FINGER_OPEN_M = 0.04          # one finger's travel (the old opening is |q1| / FINGER_OPEN_M)
GAP_OPEN_M = 0.08             # two fingers (the new opening is (q1 - q2) / GAP_OPEN_M)
FAR_M = 0.12                  # target this far from the EE and not lifted = the hand certainly holds nothing of it


def gap_opening(q1: float, q2: float) -> float:
    """``(q1 - q2) / 0.08`` clipped to [0, 1] -- the same formula as ``server.gripper_gap_norm`` (kept free of the server import)."""
    return float(min(1.0, max(0.0, (float(q1) - float(q2)) / GAP_OPEN_M)))


def single_opening(q1: float) -> float:
    return abs(float(q1)) / FINGER_OPEN_M


@dataclasses.dataclass
class ReplayCall:
    seq: int
    obs_index: int
    q1: float
    q2: float
    feedback: Optional[dict]            # chunk summary exec_feedback (None: no chunk)
    grasp: dict                          # the recorded ``grasp`` block of the call row
    hsr6_phase: Optional[str]
    lifted: bool                        # ground truth: in hand
    dee: float
    dz: float


def _qpos_layout(names) -> dict:
    adr, i = {}, 0
    for n in names:
        n = str(n)
        k = 7 if n.endswith("_joint0") else 1
        adr[n] = (i, k)
        i += k
    return adr


def load_episode(run_dir: pathlib.Path, stem: str, chunk_by_seq: dict) -> list[ReplayCall]:
    d = pathlib.Path(run_dir) / "episodes"
    rows = [json.loads(line) for line in (d / f"{stem}_calls.jsonl").read_text().splitlines() if line.strip()]
    geom = np.load(d / f"{stem}_geom.npz")
    ep = np.load(d / f"{stem}.npz")
    names = list(geom["model_joint_names"])
    adr = _qpos_layout(names)
    obs_idx = list(geom["obs_index"])
    sim = geom["sim_state"]
    rs = geom["robot_state/head"]
    ee = ep["obs_robot0_eef_pos"]
    objs = [n for n in names if str(n).endswith("_joint0") and "hand" not in str(n)]
    zpos = {n: sim[:, 1 + adr[str(n)][0] + 2] for n in objs}
    tgt = max(objs, key=lambda n: zpos[n].max() - zpos[n][0])
    z0 = zpos[tgt][0]
    out: list[ReplayCall] = []
    for r in rows:
        oi = r.get("obs_index")
        if oi not in obs_idx:
            continue
        f = obs_idx.index(oi)
        a = adr[str(tgt)][0]
        p = sim[f, 1 + a:1 + a + 3]
        dee = float(np.linalg.norm(p - ee[oi]))
        fb = None
        chunk = chunk_by_seq.get(int(r["seq"]))
        if chunk is not None:
            with np.load(chunk) as z:
                fb = json.loads(str(z["summary_json"])).get("exec_feedback")
        out.append(ReplayCall(seq=int(r["seq"]), obs_index=int(oi), q1=float(rs[f, 7]), q2=float(rs[f, 8]), feedback=fb,
                              grasp=r.get("grasp") or {}, hsr6_phase=(r.get("hsr6") or {}).get("phase"),
                              lifted=bool(p[2] > z0 + 0.03 and dee < 0.10), dee=dee, dz=float(p[2] - z0)))
    return out


def _held_observation(hold_checks: Optional[dict]) -> Optional[dict]:
    """Inverse of ``_hold_checks``' record: the dict ``AG3S.held_observation()`` gave (``revoke_centre_slack`` 0.040)."""
    hc = hold_checks or {}
    if "observed_distance_mm" not in hc:
        return None
    return {"centre_distance_m": hc["observed_distance_mm"] / 1000.0,
            "held_radius_m": hc["observed_limit_mm"] / 1000.0 - 0.040,
            "hand_rise_m": hc["hand_rise_mm"] / 1000.0, "object_rise_m": hc["object_rise_mm"] / 1000.0}


def feedback_with_opening(feedback: dict, opening: float) -> dict:
    fb = dict(feedback)
    mg = np.asarray(fb.get("measured_gripper", (1.0, 1.0)), np.float64).reshape(-1).copy()
    mg[0] = opening
    fb["measured_gripper"] = mg
    return fb


def replay_episode(calls: Sequence[ReplayCall], config: LatchConfig, *, gap: bool,
                   attach_assist: bool = True) -> list[dict]:
    """Run the latch over ``calls``. ``gap`` False = the recorded single-finger opening, True = the two-finger gap opening."""
    latch = GraspLatch(config, evidence=True, external_destination=True)
    out: list[dict] = []
    for c in calls:
        g = c.grasp
        ev = g.get("evidence")
        if not ev or c.feedback is None:
            out.append({"seq": c.seq, "state": g.get("state"), "recorded_state": g.get("state"), "skipped": True})
            continue
        if latch.phase is GraspPhase.SEARCHING and g.get("state_before") not in (None, "searching"):
            latch.phase = GraspPhase.LATCHED
        opening = gap_opening(c.q1, c.q2) if gap else single_opening(c.q1)
        fb = feedback_with_opening(c.feedback, opening) if c.feedback.get("available") else c.feedback
        signal = grasp_signal_from_feedback(fb, 0, config.gripper_closed_below)
        align = g.get("alignment") or {}
        reach = ev.get("reach_mm")
        event = latch.observe_grasp(
            signal, reach_m=None if reach is None else float(reach) / 1000.0,
            manipulated_id=ev.get("manipulated_id"), manipulated_state=ev.get("manipulated_state"),
            placed=bool(ev.get("placed", False)), held_observation=_held_observation(ev.get("hold_checks")),
            **({"reach_context": {"geometry_frame": align.get("geometry_frame"),
                                  "reach_state": align.get("reach_state", "now")}} if config.reach_sticky else {}),
            **({"attach_assist": {"phase": c.hsr6_phase}} if attach_assist else {}))
        if event.attach and g.get("rejected"):
            event = latch.revert_attach(str((g.get("rejected") or {}).get("reason", "recorded AG3S.attach rejection")))
        out.append({"seq": c.seq, "state": latch.phase.value, "recorded_state": g.get("state"),
                    "opening": signal.opening, "attach": bool(event.attach), "detach": bool(event.detach),
                    "revoked": bool(event.revoked), "slipped": bool(event.slipped), "note": event.note})
    return out


def summarise(per_episode: dict[str, tuple[Sequence[ReplayCall], Sequence[dict]]]) -> dict:
    """In-hand HELD fraction, HELD calls that are not in hand, agreement with the recorded states."""
    n_in = n_in_held = held_not_in = held_far = n_held = n_skipped = n_agree = n_calls = 0
    not_held = collections.Counter()
    per = {}
    for stem, (calls, res) in per_episode.items():
        e = dict(in_hand=0, in_hand_held=0, held=0, held_not_in_hand=0, held_target_far=0, agree=0, calls=len(calls))
        for c, r in zip(calls, res):
            n_calls += 1
            if r.get("skipped"):
                n_skipped += 1
                continue
            n_agree += int(r["state"] == r["recorded_state"])
            e["agree"] += int(r["state"] == r["recorded_state"])
            held = r["state"] == "held"
            e["held"] += int(held)
            if c.lifted:
                e["in_hand"] += 1
                e["in_hand_held"] += int(held)
                if not held:
                    not_held[r["state"]] += 1
            elif held:
                e["held_not_in_hand"] += 1
                e["held_target_far"] += int(c.dee >= FAR_M)     # HELD with the target 12 cm or more from the EE: certainly empty
        n_in += e["in_hand"]
        n_in_held += e["in_hand_held"]
        n_held += e["held"]
        held_not_in += e["held_not_in_hand"]
        held_far += e["held_target_far"]
        per[stem] = e
    return {"in_hand_calls": n_in, "in_hand_held": n_in_held,
            "in_hand_held_fraction": None if not n_in else round(n_in_held / n_in, 4),
            "held_calls": n_held, "held_not_in_hand": held_not_in, "held_target_far": held_far, "in_hand_not_held_by_state": dict(not_held),
            "agree_with_recorded": n_agree, "calls": n_calls, "skipped": n_skipped, "per_episode": per}


def load_run(run_dir: pathlib.Path) -> dict[str, list[ReplayCall]]:
    from AG3S.trajopt.experiments import panda_fixed_input_ablation as T

    run_dir = pathlib.Path(run_dir)
    cdir = sorted((run_dir / "server_constraints").glob("run_*"))[0]
    eps: dict[str, list[ReplayCall]] = {}
    for stem in sorted(p.name[:-len("_calls.jsonl")] for p in (run_dir / "episodes").glob("*_calls.jsonl")):
        calls = T.load_episode_calls(run_dir / "episodes", stem)
        T.attach_chunks(calls, cdir)
        eps[stem] = load_episode(run_dir, stem, {c.seq: c.chunk_path for c in calls if c.chunk_path is not None})
    return eps


def recorded_config(**changes: Any) -> LatchConfig:
    """The pc version's latch: the Panda calibration + the options ``pc`` turns on (slip detach, sticky reach, drop-from-open)."""
    from AG3S.adapters.panda.server import panda_latch_config

    return dataclasses.replace(panda_latch_config(), slip_detach=True, reach_sticky=True, drop_from_open=True, **changes)


def gap_config(**changes: Any) -> LatchConfig:
    """The ``-pc-gap`` version's latch: :func:`recorded_config` with the gap-opening thresholds."""
    from AG3S.adapters.panda.server import panda_latch_config

    return dataclasses.replace(panda_latch_config(gap_opening=True), slip_detach=True, reach_sticky=True, drop_from_open=True,
                               **changes)


def pre_release_fraction(per_episode_calls: dict[str, Sequence[ReplayCall]], results: dict[str, Sequence[dict]],
                         reference: dict[str, Sequence[dict]]) -> tuple[int, int]:
    """(HELD, in-hand) over the in-hand calls before the first PLACED of ``reference`` in each episode. The calls after a release
    cannot be HELD again in the replay (PLACED is terminal; a real run re-grasps on its own trajectory), so they say nothing about
    the opening -- this keeps the denominator the same for both variants."""
    held = n = 0
    for stem, calls in per_episode_calls.items():
        ref = reference[stem]
        cut = next((i for i, x in enumerate(ref) if x["state"] == "placed"), len(calls))
        for i, (c, r) in enumerate(zip(calls, results[stem])):
            if i < cut and c.lifted:
                n += 1
                held += int(r["state"] == "held")
    return held, n


def separation_table(eps: dict[str, Sequence[ReplayCall]]) -> dict:
    """Settled closed calls (last executed command < 0.1, gap change < 1.6 mm) binned by the finger gap, per ground-truth class:
    ``in_hand`` (held_check definition), ``target_far`` (not in hand, target >= 12 cm from the EE: nothing of it between the pads),
    ``target_near`` (not in hand, target < 12 cm). Also the pass rate of the empty-hand criteria on each class."""
    bins = [(0.0, 1.35), (1.35, 3.2), (3.2, 6.0), (6.0, 12.0)]
    table = {f"{lo}-{hi} mm": collections.Counter() for lo, hi in bins}
    passes = collections.defaultdict(lambda: collections.Counter())
    for calls in eps.values():
        prev = None
        for c in calls:
            gap = (c.q1 - c.q2) * 1000.0
            dgap = None if prev is None else abs(gap - prev)
            prev = gap
            fb = c.feedback
            if fb is None or not fb.get("available") or dgap is None or dgap >= 1.6 or gap >= 12.0:
                continue
            ex = [bool(v) for v in fb.get("executed", [])]
            ap = np.asarray(fb.get("applied_gripper", [[1.0, 1.0]]), np.float64).reshape(-1, 2)
            cmd = [ap[i, 0] for i, f in enumerate(ex) if f and i < len(ap)]
            if not cmd or cmd[-1] >= 0.1:
                continue
            cls = "in_hand" if c.lifted else ("target_far" if c.dee >= FAR_M else "target_near")
            for lo, hi in bins:
                if lo <= gap < hi:
                    table[f"{lo}-{hi} mm"][cls] += 1
            passes["single_finger |q1|/0.04 >= 0.0253 (old not_empty)"][cls + ("_pass" if abs(c.q1) / FINGER_OPEN_M >= 0.025325 else "_fail")] += 1
            passes["gap/0.08 >= 0.04 (new not_empty)"][cls + ("_pass" if gap / 80.0 >= 0.04 else "_fail")] += 1
            passes["gap/0.08 >= 0.01678"][cls + ("_pass" if gap / 80.0 >= 0.01678 else "_fail")] += 1
    return {"by_gap": {k: dict(v) for k, v in table.items()}, "criteria": {k: dict(v) for k, v in passes.items()}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", action="append", required=True, type=pathlib.Path,
                    help="one benchmark = one or more runs; give --bench NAME=RUN_DIR[,RUN_DIR...] instead for several benches")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args(argv)
    eps: dict[str, list[ReplayCall]] = {}
    for run in args.run_dir:
        for stem, calls in load_run(run).items():
            eps[f"{run.name[:15]}:{stem}"] = calls
    old = {s: replay_episode(c, recorded_config(), gap=False) for s, c in eps.items()}
    new = {s: replay_episode(c, gap_config(), gap=True) for s, c in eps.items()}
    out = {"single_finger_recorded_config": summarise({s: (eps[s], old[s]) for s in eps}),
           "gap_opening": summarise({s: (eps[s], new[s]) for s in eps}),
           "pre_release_in_hand_held": {"single_finger": pre_release_fraction(eps, old, new),
                                        "gap_opening": pre_release_fraction(eps, new, new)},
           "separation": separation_table(eps)}
    args.out.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, {a: b for a, b in v.items() if a != "per_episode"} if isinstance(v, dict) else v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
