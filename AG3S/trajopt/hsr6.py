"""AG3S-CORE-EVAL-HSR6 — the 6-phase task-phase probe (HSR6) in serving, behind four default-off flags.

**What HSR6 is.** T64 Part A's probe (`AG3S/ag3s/asset/subtask_probe/hsr6_t64_kv_L4.npz`, a copy of
`verify/T64/partA/probe_6.npz`, md5 f0bb92aa…): StandardScaler + multinomial logistic regression (C 0.03) on the same
`kv_L4` feature the serving 3-class HSR (`SubtaskProbe`, `kv_L4_v1`) already reads — the prefix KV pass of the
attention copy, no extra forward. `AttentionPolicy(hsr6_probe=...)` puts ``result["hsr6"] = SubtaskProbe.predict``
(``{"p": {phase: prob}, "argmax", "probe"}``) next to ``result["subtask"]``. The decision is the **raw argmax** of
that request (no debounce, `handoff/AG3S-CORE-EVAL-HSR6.task.md` §1).

**The four uses** (task §2, rules fixed before results; every flag off = the code path before this module):

| # | flag | rule (this module) | where it acts |
|---|---|---|---|
| ① | `attach_assist` | attach = strong HELD ∨ (phase ∈ {lift, transport, placement} ∧ `closed` ∧ `blocked`) — `attach_assist_ok` | `GraspLatch.observe_grasp(attach_assist=...)` (CLOSING) via `SafePolicy._advance_grasp` |
| ② | `held_gate` | the held-object allowance is withdrawn only when phase ∈ {approach, retreat} (or no label); otherwise (grasp, lift, transport, placement) it holds while `blocked` ∨ latch HELD — `held_allowance` (AG3S-CORE-EVAL-A6 §3, session B T66) | `AG3S.set_held_allowance` → `_frame_held_attached` (held-body cover lift off the table / crate floor) |
| ③ | `target_exempt` | target exemption (target-free field view, per-link target margin, target tests) only outside retreat: in retreat the not-held target is an obstacle — `withdraw_target_exemption` | `SafePolicy._scene_fn` → `strip_target_permission` on the `SceneSnapshot` |
| ④ | `objective` | IFG-NBC neighbour allowance, the A4 close-row goal and the G1 slot only when the existing rule ∧ phase = grasp — `objective_allowed` | `_neighbour_contact_gate`, `_scene_fn` (A4), `_geo_request` (G1) |
| ⑤ | `placement_contact` | (SL6-H, Panda) phase = placement ∧ (latch HELD ∨ (latch CLOSING ∧ `closed` ∧ `blocked`)) — `placement_contact_ok`: the held object's own query rows are not blocked by a surface **beneath** them (`placement_contact_spec`) | `SafePolicy._hsr6_placement_contact` → `SceneSnapshot.placement_contact` → `CollisionLinearizer._placement_contact_margin` |

**T67 ② — the ④ window (`objective_window`, `--hsr6-objective-window`, default off).** With it ④ also lets the three
objective terms stay when HSR6 = approach **and** the A6 "grasp moment" window is open (latch latched / closing, no
revoked attach, the grasping hand's palm site within `collision.neighbour_palm_contact_gate` = 60 mm of the target —
`SafePolicy._grasp_moment_window`, the same computation as A6). The last approach requests before the close are
labelled approach (T66b): ④ withdrew the finger allowance, A4 and G1 exactly there. Off: ④ as above.

A request without an HSR6 label (probe failed / not loaded) is ``phase None``: ① no assist, ② no allowance (the
stricter side), ③ the current rule, ④ no objective term, ⑤ no placement contact.

**② fix (AG3S-CORE-EVAL-A6 §3, user ruling 2026-10-29 on session B T66).** The first rule (allowance only in lift ·
transport · placement) withdrew the lift on the 1–3 requests right after the grasp, while HSR6 still read grasp (the
apple still on the table, T64 defines grasp until a 2 cm rise): the held spheres met the table, the standard solve
failed and the GPU multi-seed answer turned the arm 6–10° (`outputs/2026-10-29_HSR6_쥔직후_궤적급변_원인/`). Now only
approach · retreat (a hand that is not holding by the probe's reading) withdraw it.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Mapping, Optional

import numpy as np

from AG3S.ag3s.stages.subtask_probe import SubtaskProbe

__all__ = [
    "EXEMPT_PHASES",
    "HELD_WITHDRAW_PHASES",
    "HOLD_PHASES",
    "Hsr6Flags",
    "Hsr6SubsetProbe",
    "OBJECTIVE_PHASE",
    "OBJECTIVE_WINDOW_PHASE",
    "PHASES6",
    "PROBE_NAME",
    "RETREAT",
    "attach_assist_ok",
    "default_probe_path",
    "held_allowance",
    "is_ordered_subset",
    "load_hsr6_probe",
    "objective_allowed",
    "PLACEMENT",
    "PLACEMENT_BENEATH_COS",
    "PLACEMENT_LIFT_PHASES",
    "PLACEMENT_CONTACT_CHECKS",
    "phase_of",
    "placement_contact_ok",
    "placement_contact_spec",
    "strip_target_permission",
    "withdraw_target_exemption",
]

#: Class order of the T64 probe (`classes` in the npz; checked on load).
PHASES6 = ("approach", "grasp", "lift", "transport", "placement", "retreat")
#: ① — the held phases (T64 Part C `HOLD_PHASES = (2, 3, 4)`).
HOLD_PHASES = frozenset({"lift", "transport", "placement"})
#: ② (AG3S-CORE-EVAL-A6 §3) — the phases that withdraw the held-object allowance: the hand is not holding.
HELD_WITHDRAW_PHASES = frozenset({"approach", "retreat"})
#: ③ — the phases where the target-free view is the rule (task §2 ③); retreat withdraws it, the rest keep today's rule.
EXEMPT_PHASES = frozenset({"approach", "grasp"})
RETREAT = "retreat"
#: ④ — the only phase in which the close-row goal (A4 / G1) and the neighbour allowance stay on.
OBJECTIVE_PHASE = "grasp"
#: ⑤ (SL6-H) — the only phase in which the held object may touch the surfaces beneath it.
PLACEMENT = "placement"
#: ⑤ SL10e (`--hsr6-placement-contact-lift`) -- the phases when that flag is on: placement or lift (not transport: a carried
#: object must not be allowed to touch what it passes over). Recorded as ``hsr6.detail.placement_contact.phase_rule``.
PLACEMENT_LIFT_PHASES = frozenset({PLACEMENT, "lift"})
PLACEMENT_LIFT_RULE = "placement|lift"
#: ⑤ — the CLOSING evidence that counts as "held" for ⑤ (task SL6 §11 H1: `closed` ∧ `blocked`, as ① uses them).
PLACEMENT_CONTACT_CHECKS = ("closed", "blocked")
#: ⑤ — "beneath": the field's unit gradient at the held sphere's centre points up within 45° (``ĝ_z ≥ cos 45°``),
#: i.e. the nearest surface lies in the downward 45° cone (or, centre inside a body, the way out is up).
PLACEMENT_BENEATH_COS = float(np.cos(np.deg2rad(45.0)))
#: T67 ② — the phase in which the ④ window (`Hsr6Flags.objective_window`) may keep the objective terms.
OBJECTIVE_WINDOW_PHASE = "approach"
PROBE_NAME = "hsr6_t64_kv_L4"


def default_probe_path() -> pathlib.Path:
    """`AG3S/ag3s/asset/subtask_probe/hsr6_t64_kv_L4.npz`, located from this file (no symlinks)."""
    return pathlib.Path(__file__).resolve().parents[1] / "ag3s" / "asset" / "subtask_probe" / f"{PROBE_NAME}.npz"


def is_ordered_subset(classes, full=PHASES6) -> bool:
    """SL6 B2 — `classes` are ≥ 2 distinct phases of `full`, in `full`'s order."""
    classes = tuple(str(c) for c in classes)
    if len(classes) < 2 or len(set(classes)) != len(classes) or not set(classes) <= set(full):
        return False
    idx = [full.index(c) for c in classes]
    return idx == sorted(idx)


def load_hsr6_probe(path: Optional[str | pathlib.Path] = None):
    """The 6-phase probe as a `SubtaskProbe` (same `proba` as the serving HSR). Refuses a wrong class order / dim.

    SL6 B2: a probe whose ``classes`` are an ordered subset of `PHASES6` (Panda: no retreat) is accepted and
    wrapped in `Hsr6SubsetProbe` — its `predict` reports every phase of `PHASES6`, the absent ones with
    probability 0, so `phase_of` never picks one and a rule that needs it (③ retreat) never fires. A probe
    with exactly `PHASES6` is returned as the plain `SubtaskProbe` (RB-Y1 unchanged)."""
    probe = SubtaskProbe.load(default_probe_path() if path is None else path)
    if tuple(probe.classes) != PHASES6 and not is_ordered_subset(probe.classes):
        raise ValueError(f"HSR6 probe {path or default_probe_path()}: classes {probe.classes} are neither "
                         f"{PHASES6} nor an ordered subset of it")
    if probe.dim != 512:
        raise ValueError(f"HSR6 probe {path or default_probe_path()}: dim {probe.dim} != 512 (kv_L4)")
    if tuple(probe.classes) != PHASES6:
        return Hsr6SubsetProbe(probe)
    return probe


class Hsr6SubsetProbe(SubtaskProbe):
    """SL6 B2 — a `SubtaskProbe` over an ordered subset of `PHASES6`; `predict` pads the absent phases with
    probability 0.0 (`absent`), the argmax is over the probe's own classes (= `SubtaskProbe.predict`)."""

    def __init__(self, probe: SubtaskProbe):
        super().__init__(probe.mean, probe.scale, probe.coef, probe.intercept, classes=probe.classes,
                         name=probe.name)
        self.absent = tuple(c for c in PHASES6 if c not in self.classes)

    def predict(self, x: np.ndarray) -> dict[str, Any]:
        out = super().predict(x)
        own = out["p"]
        out["p"] = {c: (float(own[c]) if c in own else 0.0) for c in PHASES6}
        return out

    def record(self) -> dict[str, Any]:
        return {**super().record(), "phases_absent": list(self.absent)}


@dataclasses.dataclass(frozen=True)
class Hsr6Flags:
    """The four uses (task §2). All False = off (`SafePolicy(hsr6=None)` is the same).

    `objective_window` (T67 ②) is not a use of its own: it widens ④ (`objective`) and is refused without it
    (`serve_safe.reject_hsr6_flags`). It is listed by `names()` only when on (the record of a run without it is as
    before)."""

    attach_assist: bool = False
    held_gate: bool = False
    target_exempt: bool = False
    objective: bool = False
    #: ⑤ SL6-H (`--hsr6-placement-contact`) — after the four flags above, so they keep their order in `names()`.
    placement_contact: bool = False
    #: T67 ② (`--hsr6-objective-window`, merged 2026-11-03 after `placement_contact`).
    objective_window: bool = False
    #: SL10e (`--hsr6-placement-contact-lift`): ⑤'s phase condition becomes placement ∨ lift. Not a use of its own: refused
    #: without `placement_contact` (`serve_safe.reject_hsr6_flags`); after `objective_window` so `names()` keeps its order.
    placement_contact_lift: bool = False

    @property
    def on(self) -> bool:
        # `objective_window` alone is refused at start (`reject_hsr6_flags`); it counts here so that check sees it
        return bool(self.attach_assist or self.held_gate or self.target_exempt or self.objective
                    or self.placement_contact or self.objective_window or self.placement_contact_lift)

    def names(self) -> list[str]:
        return [f.name for f in dataclasses.fields(self) if getattr(self, f.name)]


def phase_of(record: Any) -> Optional[str]:
    """`result["hsr6"]` → its argmax phase, or None (no record / unknown class / non-finite probabilities)."""
    if not isinstance(record, Mapping):
        return None
    p = record.get("p")
    if isinstance(p, Mapping) and p:
        vals = [float(p.get(c, np.nan)) for c in PHASES6]
        if not all(np.isfinite(vals)):
            return None
        return PHASES6[int(np.argmax(vals))]       # ties -> the earlier phase (numpy argmax), as the offline code
    arg = record.get("argmax")
    return str(arg) if arg in PHASES6 else None


def attach_assist_ok(phase: Optional[str], checks: Mapping[str, Any]) -> bool:
    """① (T64 Part C): phase ∈ {lift, transport, placement} ∧ the CLOSING checks `closed` ∧ `blocked`."""
    return bool(phase in HOLD_PHASES and checks.get("closed") is True and checks.get("blocked") is True)


def held_allowance(phase: Optional[str], *, latch_held: bool, blocked: bool) -> bool:
    """② (fixed, AG3S-CORE-EVAL-A6 §3): phase ∉ {approach, retreat} ∧ phase labelled ∧ (finger `blocked` ∨ latch HELD).

    | phase | allowance |
    |---|---|
    | approach · retreat | withdrawn (the fix: only these) |
    | grasp | `blocked` ∨ HELD — kept right after the grasp while HSR6 still reads grasp (was: withdrawn) |
    | lift · transport · placement | `blocked` ∨ HELD (unchanged) |
    | None (no label) | withdrawn (unchanged: the stricter side) |
    """
    if phase is None or phase in HELD_WITHDRAW_PHASES or phase not in PHASES6:
        return False
    return bool(bool(blocked) or bool(latch_held))


def withdraw_target_exemption(phase: Optional[str]) -> bool:
    """③ : True only in retreat (approach / grasp keep the target-free view; lift · transport · placement keep
    today's held-object rule; no label keeps today's rule)."""
    return phase == RETREAT


def objective_allowed(phase: Optional[str], *, window: Optional[bool] = None) -> bool:
    """④ : the close-row goal and the neighbour allowance only in grasp (no label: off).

    T67 ② — `window` is the A6 "grasp moment" window of this request when `--hsr6-objective-window` is on (`None` =
    the flag is off): then approach ∧ window open also keeps them. Any other phase is unaffected by the window."""
    if phase == OBJECTIVE_PHASE:
        return True
    return bool(window) if (window is not None and phase == OBJECTIVE_WINDOW_PHASE) else False


def placement_contact_ok(phase: Optional[str], *, latch: Optional[str], checks: Mapping[str, Any],
                         lift: bool = False) -> bool:
    """⑤ (SL6-H, task SL6 §11 H1, fixed before results): phase = placement ∧ (latch HELD ∨ (latch CLOSING ∧ the
    CLOSING evidence `closed` ∧ `blocked`)). `latch` is the `GraspPhase` value (``"held"`` · ``"closing"`` · …).

    | phase | latch | evidence | ⑤ |
    |---|---|---|---|
    | placement | held | any | on |
    | placement | closing | closed ∧ blocked | on |
    | placement | closing | otherwise | off |
    | placement | searching · latched · placed | any | off |
    | anything else / None | any | any | off |

    SL10e: with ``lift=True`` (`--hsr6-placement-contact-lift`) the phase column is placement **or lift** -- every row of
    the table above holds for lift as for placement (the latch condition is unchanged), transport · approach · grasp ·
    retreat stay off. ``lift=False`` (default) = the table as written.
    """
    if phase != PLACEMENT and not (lift and phase in PLACEMENT_LIFT_PHASES):
        return False
    if latch == "held":
        return True
    return bool(latch == "closing" and all(checks.get(k) is True for k in PLACEMENT_CONTACT_CHECKS))


def placement_contact_spec(voxel_m: Optional[float] = None) -> dict:
    """⑤ — the `SceneSnapshot.placement_contact` the optimizer reads (`CollisionLinearizer._placement_contact_margin`).

    ``cos_min``: the "beneath" cone (`PLACEMENT_BENEATH_COS`). ``footprint_tol_m``: how far outside the held object's xy
    footprint (the union of its spheres' xy discs at the same step) the nearest surface point may lie — ``None`` = one
    voxel of the field the row reads."""
    return {"cos_min": PLACEMENT_BENEATH_COS, "footprint_tol_m": None if voxel_m is None else float(voxel_m)}


def strip_target_permission(snapshot) -> tuple[Any, dict]:
    """③ — the `SceneSnapshot` with no permission on the (not held) target: every robot row reads the full field
    (the target is in it), no per-link target margin, no target class (it is an obstacle row, `obstacle_margin`
    applies), no target volume. Returns `(snapshot, record)`; `record["changed"]` is False when there was no
    permission to strip (then the snapshot is returned as is)."""
    fields = {
        "target_free_mask": getattr(snapshot, "target_free_mask", None),
        "target_free_exempt": getattr(snapshot, "target_free_exempt", None),
        "manipulated_link_margin": getattr(snapshot, "manipulated_link_margin", None),
        "manipulated_points": getattr(snapshot, "manipulated_points", None),
        "manipulated_spheres": getattr(snapshot, "manipulated_spheres", None),
        "target_contact_rows": getattr(snapshot, "target_contact_rows", None),
        "target_volume": getattr(snapshot, "target_volume", None),
    }
    present = sorted(k for k, v in fields.items() if v is not None
                     and not (k == "target_contact_rows" and not np.asarray(v, bool).any()))
    rec = {"changed": bool(present), "stripped": present}
    if not present:
        return snapshot, rec
    snapshot = dataclasses.replace(
        snapshot, target_free_mask=None, target_free_exempt=None, manipulated_link_margin=None,
        manipulated_points=None, manipulated_spheres=None, manipulated_sphere_radii=None,
        target_contact_rows=None, target_volume=None)
    return snapshot, rec
