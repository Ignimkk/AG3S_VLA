"""T46c A12 — offline replay of the geometry-certification predicate on recorded server chunks.

The certification question (`safe_policy.geometry_certified` → `ag3s.types.subtask_no_target`) is
asked again on a record rebuilt from each chunk's `summary_json`, under the recorded rule (`W0`) or
with `clustering.placed_no_seed_certify` on (`W1`). Only the certification changes; the field, the
trajectory and every other verdict reason are the recorded ones. A chunk whose certification
flips loses its `uncertified` reason (and gains `subtask_no_target`, as the server writes it); the
client gate (`wire.gate_decision`) then decides on the rest.

    MUJOCO_GL=osmesa JAX_PLATFORMS=cpu PYTHONPATH=${AG3S_PROJECT_ROOT} \\
        .venv-openpi-live/bin/python -m AG3S.trajopt.experiments.certification_replay \\
        --cl <CL3 dir with analysis/*.json> --out <json>

Variant × recorded arm: `W1` on the C1a records, `F1W1` = `W1` on the C1a+F1 records,
`F1F2W1` = `W1` on the C1a+F1+F2 records (the arm is the recording; this tool does not replay
the field — `ta_offline` variants `F1W1` do that, per chunk with a TO re-solve).

What the rebuilt record carries (T46b `certification_replay`): status · grounding_status ·
validity (as the enums) · has_target → target · `manipulated` (the identity record) ·
`exclusion.attached_label` → attached · `target_permission.target_field_exclude` ·
metrics {subtask, manipulated, exclusion}. **Not recorded:** `manipulated_link_margin` (None here;
T46b's replay agreed with the recorded reason on 1350/1350 chunks per arm with that assumption) —
`fidelity` re-checks the agreement on every chunk this tool reads.

Per chunk what-if, open loop: a chunk that would have executed changes the robot state and every
later chunk; that closed-loop effect is not computed here.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import types
from typing import Any, Iterable, Optional

import numpy as np

TOOL = "AG3S.trajopt.experiments.certification_replay"

#: variant → keyword arguments of `record_from_summary` (None = as recorded).
VARIANTS: dict[str, dict[str, Any]] = {
    "W0": {"placed_no_seed_certify": None},
    "W1": {"placed_no_seed_certify": True},
}

#: Report names for `W1` per recorded arm (the task's variant names).
W1_NAMES = {"C1a": "W1", "C1aF1": "F1W1", "C1aF1F2": "F1F2W1"}


def _enum(cls, value):
    try:
        return cls(value)
    except ValueError:
        return value


def record_from_summary(summary: dict[str, Any], *,
                        placed_no_seed_certify: Optional[bool] = None) -> Any:
    """A `CollisionConstraintSet`-shaped record from one chunk's `summary_json`.

    `placed_no_seed_certify`: None = the recorded `subtask` record as is; True = the key the pipeline
    writes with `clustering.placed_no_seed_certify` on; False = that key removed.
    """
    from AG3S.ag3s.types import ConstraintValidity, GroundingStatus, PipelineStatus

    tp = summary.get("target_permission") or {}
    ex = summary.get("exclusion")
    sub = summary.get("subtask")
    if isinstance(sub, dict) and placed_no_seed_certify is not None:
        sub = {k: v for k, v in sub.items() if k != "placed_no_seed_certify"}
        if placed_no_seed_certify:
            sub["placed_no_seed_certify"] = True
    return types.SimpleNamespace(
        status=_enum(PipelineStatus, summary.get("status")),
        grounding_status=_enum(GroundingStatus, summary.get("grounding_status")),
        validity=_enum(ConstraintValidity, summary.get("validity")),
        target=(object() if summary.get("has_target") else None),
        manipulated=summary.get("manipulated"),
        attached=(object() if (ex or {}).get("attached_label") else None),
        target_field_exclude=tp.get("target_field_exclude"),
        manipulated_link_margin=None,
        metrics={"subtask": sub, "manipulated": summary.get("manipulated"), "exclusion": ex})


def certified(summary: dict[str, Any], **spec: Any) -> bool:
    """`safe_policy.geometry_certified` on the rebuilt record (the predicate the server asks)."""
    from AG3S.trajopt.safe_policy import geometry_certified

    return bool(geometry_certified(record_from_summary(summary, **spec)))


def subtask_certified(summary: dict[str, Any], **spec: Any) -> bool:
    """`ag3s.types.subtask_no_target` on the rebuilt record."""
    from AG3S.ag3s.types import subtask_no_target

    return bool(subtask_no_target(record_from_summary(summary, **spec)))


def recorded_reasons(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((summary.get("verdict") or {}).get("reasons")) or [])


def replay_reasons(summary: dict[str, Any], **spec: Any) -> list[dict[str, Any]]:
    """The recorded verdict reasons with the certification re-asked.

    W1 only widens the predicate, so the one change is: certified under `spec` but not as recorded
    → the recorded `uncertified` reason is replaced by `subtask_no_target` (what the server writes
    when the subtask predicate certified the frame). Every other reason — a `collision` row above
    all — is the recorded one. Otherwise the recorded reasons are returned unchanged.
    """
    from AG3S.trajopt import wire

    reasons = recorded_reasons(summary)
    if not (certified(summary, **spec) and not certified(summary)):
        return reasons
    out = [r for r in reasons if r.get("kind") != "uncertified"]
    if not any(r.get("kind") == "subtask_no_target" for r in out):
        out.insert(0, wire.make_reason("subtask_no_target", "replayed (T46c W1)",
                                       grounding_status=summary.get("grounding_status")))
    return out


def decision(reasons: Iterable[dict[str, Any]]) -> str:
    from AG3S.trajopt import wire

    return wire.gate_decision(list(reasons))[0]


def phase_of(summary: dict[str, Any]) -> Optional[str]:
    return ((summary.get("grasp") or {}).get("state"))


def _summary(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return json.loads(str(z["summary_json"]))


def replay_runs(cl_dir: pathlib.Path, variants: Iterable[str] = ("W0", "W1"),
                arms: Optional[Iterable[str]] = None) -> dict[str, Any]:
    """Per arm and variant: predicate fidelity (W0 vs the recorded reasons), the chunks whose
    certification flips, the execute conversions among the recorded HOLD chunks, and the
    after-close HOLD counts (T46b definition: client HOLD rows on server chunks with `seq >
    close_seq`; uncertified = the HOLD kinds contain `uncertified`)."""
    variants = list(variants)
    arms = None if arms is None else set(arms)
    out: dict[str, Any] = {}
    for af in sorted((cl_dir / "analysis").glob("*.json")):
        a = json.loads(af.read_text())
        if arms is not None and a["arm"] not in arms:
            continue
        hold = {h["seq"]: h for h in a.get("hold") or []}
        close = (a.get("pre_grasp") or {}).get("close_seq")
        A = out.setdefault(a["arm"], {"runs": 0, "n_chunks": 0,
                                      "fidelity": collections.Counter(),
                                      "fidelity_to_certified": collections.Counter(),
                                      "variants": {}})
        A["runs"] += 1
        for c in a.get("server_chunks") or []:
            s = _summary(c["server_file"])
            seq = c["seq"]
            h = hold.get(seq)
            rec_hold = h is not None
            after = close is not None and seq > close
            rec_unc = rec_hold and "uncertified" in (h.get("kinds") or [])
            A["n_chunks"] += 1
            kinds = [r.get("kind") for r in recorded_reasons(s)]
            A["fidelity"][f"pred={subtask_certified(s)}|rec={'subtask_no_target' in kinds}"] += 1
            to_cert = ((s.get("to") or {}).get("metrics") or {}).get("geometry_certified")
            if to_cert is not None:
                A["fidelity_to_certified"][f"pred={certified(s)}|rec={bool(to_cert)}"] += 1
            for v in variants:
                spec = VARIANTS[v]
                V = A["variants"].setdefault(v, {
                    "flipped": collections.Counter(), "execute_conversions": 0,
                    "conversions_by": collections.Counter(),
                    "conversion_runs": collections.Counter(),
                    "certified_but_still_hold": 0, "certified_recorded_execute": 0,
                    "after_close_hold": {"total": 0, "uncertified": 0},
                    "after_close_hold_recorded": {"total": 0, "uncertified": 0}})
                c0, c1 = certified(s), certified(s, **spec)
                if c0 != c1:
                    V["flipped"][f"{s.get('grounding_status')}|{phase_of(s)}|"
                                 f"{c0}->{c1}"] += 1
                new_dec = decision(replay_reasons(s, **spec))
                newly = c1 and not c0
                conv = rec_hold and new_dec == "execute" and newly
                if conv:
                    V["execute_conversions"] += 1
                    sub = s.get("subtask") or {}
                    V["conversions_by"][f"grounding={s.get('grounding_status')}|"
                                        f"phase={phase_of(s)}|label={sub.get('label')}|"
                                        f"placed_seen={sub.get('placed_seen')}"] += 1
                    V["conversion_runs"][a["name"]] += 1
                if newly and new_dec == "hold":
                    # certified now, but another reason (a violating row) still holds it
                    V["certified_but_still_hold"] += 1
                if newly and not rec_hold:
                    V["certified_recorded_execute"] += 1
                if after:
                    V["after_close_hold_recorded"]["total"] += int(rec_hold)
                    V["after_close_hold_recorded"]["uncertified"] += int(rec_unc)
                    still = rec_hold and not conv
                    V["after_close_hold"]["total"] += int(still)
                    V["after_close_hold"]["uncertified"] += int(still and rec_unc and not newly)
    for A in out.values():
        A["fidelity"] = dict(A["fidelity"])
        A["fidelity_to_certified"] = dict(A["fidelity_to_certified"])
        for V in A["variants"].values():
            for k in ("flipped", "conversions_by", "conversion_runs"):
                V[k] = dict(V[k])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cl", required=True, help="closed-loop dir with analysis/<NAME>.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args(argv)
    res = replay_runs(pathlib.Path(args.cl), args.variants, args.arms)
    summary = {arm: {"W1_name": W1_NAMES.get(arm),
                     **{v: {"execute_conversions": R["variants"][v]["execute_conversions"],
                            "after_close_hold": R["variants"][v]["after_close_hold"]}
                        for v in R["variants"]}}
               for arm, R in res.items()}
    out = {"tool": TOOL, "cl": str(args.cl), "variants": {v: VARIANTS[v] for v in args.variants},
           "summary": summary, "arms": res}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
