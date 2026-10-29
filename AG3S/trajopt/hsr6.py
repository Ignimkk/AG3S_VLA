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
| ② | `held_gate` | the held-object allowance only when phase ∈ {lift, transport, placement} ∧ (`blocked` ∨ latch HELD) — `held_allowance` | `AG3S.set_held_allowance` → `_frame_held_attached` (held-body cover lift off the table / crate floor) |
| ③ | `target_exempt` | target exemption (target-free field view, per-link target margin, target tests) only outside retreat: in retreat the not-held target is an obstacle — `withdraw_target_exemption` | `SafePolicy._scene_fn` → `strip_target_permission` on the `SceneSnapshot` |
| ④ | `objective` | IFG-NBC neighbour allowance, the A4 close-row goal and the G1 slot only when the existing rule ∧ phase = grasp — `objective_allowed` | `_neighbour_contact_gate`, `_scene_fn` (A4), `_geo_request` (G1) |

A request without an HSR6 label (probe failed / not loaded) is ``phase None``: ① no assist, ② no allowance (the
stricter side), ③ the current rule, ④ no objective term.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "EXEMPT_PHASES",
    "HOLD_PHASES",
    "Hsr6Flags",
    "OBJECTIVE_PHASE",
    "PHASES6",
    "PROBE_NAME",
    "RETREAT",
    "attach_assist_ok",
    "default_probe_path",
    "held_allowance",
    "load_hsr6_probe",
    "objective_allowed",
    "phase_of",
    "strip_target_permission",
    "withdraw_target_exemption",
]

#: Class order of the T64 probe (`classes` in the npz; checked on load).
PHASES6 = ("approach", "grasp", "lift", "transport", "placement", "retreat")
#: ① / ② — the held phases (T64 Part C `HOLD_PHASES = (2, 3, 4)`).
HOLD_PHASES = frozenset({"lift", "transport", "placement"})
#: ③ — the phases where the target-free view is the rule (task §2 ③); retreat withdraws it, the rest keep today's rule.
EXEMPT_PHASES = frozenset({"approach", "grasp"})
RETREAT = "retreat"
#: ④ — the only phase in which the close-row goal (A4 / G1) and the neighbour allowance stay on.
OBJECTIVE_PHASE = "grasp"
PROBE_NAME = "hsr6_t64_kv_L4"


def default_probe_path() -> pathlib.Path:
    """`AG3S/ag3s/asset/subtask_probe/hsr6_t64_kv_L4.npz`, located from this file (no symlinks)."""
    return pathlib.Path(__file__).resolve().parents[1] / "ag3s" / "asset" / "subtask_probe" / f"{PROBE_NAME}.npz"


def load_hsr6_probe(path: Optional[str | pathlib.Path] = None):
    """The 6-phase probe as a `SubtaskProbe` (same `proba` as the serving HSR). Refuses a wrong class order / dim."""
    from AG3S.ag3s.stages.subtask_probe import SubtaskProbe

    probe = SubtaskProbe.load(default_probe_path() if path is None else path)
    if tuple(probe.classes) != PHASES6:
        raise ValueError(f"HSR6 probe {path or default_probe_path()}: classes {probe.classes} != {PHASES6}")
    if probe.dim != 512:
        raise ValueError(f"HSR6 probe {path or default_probe_path()}: dim {probe.dim} != 512 (kv_L4)")
    return probe


@dataclasses.dataclass(frozen=True)
class Hsr6Flags:
    """The four uses (task §2). All False = off (`SafePolicy(hsr6=None)` is the same)."""

    attach_assist: bool = False
    held_gate: bool = False
    target_exempt: bool = False
    objective: bool = False

    @property
    def on(self) -> bool:
        return bool(self.attach_assist or self.held_gate or self.target_exempt or self.objective)

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
    """② : phase ∈ {lift, transport, placement} ∧ (finger `blocked` ∨ latch HELD)."""
    return bool(phase in HOLD_PHASES and (bool(blocked) or bool(latch_held)))


def withdraw_target_exemption(phase: Optional[str]) -> bool:
    """③ : True only in retreat (approach / grasp keep the target-free view; lift · transport · placement keep
    today's held-object rule; no label keeps today's rule)."""
    return phase == RETREAT


def objective_allowed(phase: Optional[str]) -> bool:
    """④ : the close-row goal and the neighbour allowance only in grasp (no label: off)."""
    return phase == OBJECTIVE_PHASE


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
