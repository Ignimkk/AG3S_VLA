"""T82 — the 5-stage task-phase classifier (HSR5, VLM hidden layer 4) in serving, behind one default-off flag.

**What HSR5 is.** T77's selected classifier (`AG3S/ag3s/asset/subtask_probe/hsr5_t77_h4_img3_last.npz`, a copy of
`verify/T77/probe/models/h4_img3_last.npz`, md5 985be918…): StandardScaler + multinomial logistic regression (C 0.03)
over the PaliGemma **hidden state h_4** of the prefix pass — [cam_high mean | left_wrist mean | right_wrist mean |
last valid prefix token] = 4 × 2048 = 8192 (`hsr5_feature`, the rows (0, 1, 2, 4) of T77's
`pi05_hidden.summary_rows`). The feature comes from the attention copy's noun prefix pass with openpi's
`return_hidden` turned on for that pass only (`attention_source.JaxPrefixBackend`, no extra forward).
`AttentionPolicy(hsr5_probe=...)` puts ``result["hsr5"] = {"p": {stage: prob}, "argmax", "probe"}``. The decision is
the **raw argmax** of the request (no debounce), as HSR6.

**What `--hsr5` replaces** (`handoff/T82.task.md` §2, fixed before results; the rules themselves are the same code):

| rule | today's input | with `--hsr5` (this module) |
|---|---|---|
| new manipulated-object gate (`clustering.subtask_gate`) | 3-stage label (pick allows) | `subtask3_of`: approach · grasp → pick (allow), carry · place → place, retreat → home (block; the PLACED ∧ home release reads retreat) — fed to `AG3S.set_subtask` as a one-hot, debounced there as before |
| ① attach assist · ② held gate · ③ target exemption · ④ objective (`--hsr6-*`) | HSR6 phase | `rule_phase6`: approach → approach, grasp → grasp, carry → transport, place → placement, retreat → retreat — so ① = {carry, place}, ② withdraws in {approach, retreat}, ③ = retreat, ④ = grasp (T67 window: approach) |
| deflection-rate · continuity after grasp (`--deflection-rate-after-grasp`, `--continuity-after-grasp`) | latch HELD onwards | `after_grasp_on`: stage ∈ {carry, place, retreat} |
| **new:** closing release | none (an empty close keeps CLOSING) | latch CLOSING ∧ stage ∉ {carry, place} ∧ the last executed gripper command ≥ `gripper_closed_below` (the value a command must drop under to count as a close), 2 requests in a row → LATCHED (`GraspLatch.observe_grasp(closing_release=...)`; the ATS freeze `set_grasp_active` follows the latch) |

**No prediction on a request** (probe failed / not loaded) = today's "no label" handling: the subtask gate gets None
(as a request without a 3-stage label), the HSR6 rules get phase None (their stricter side), the after-grasp terms
fall back to the latch rule, and the closing release does not count (the streak restarts).

carry → *transport* (not *lift*): the two are treated alike by every HSR6 rule (`hsr6.HOLD_PHASES`, ② and ③), T77
merged them because their constraint sets are the same.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "AFTER_GRASP_STAGES",
    "CLOSING_KEEP_STAGES",
    "FEATURE_DIM",
    "FEATURE_ROWS",
    "HIDDEN_LAYER",
    "Hsr5Config",
    "PROBE_MD5",
    "PROBE_NAME",
    "RULE_PHASE6",
    "STAGES5",
    "SUBTASK3",
    "after_grasp_on",
    "default_probe_path",
    "hsr5_feature",
    "load_hsr5_probe",
    "rule_phase6",
    "stage_of",
    "subtask3_of",
    "subtask3_probabilities",
]

#: Class order of the T77 probe (`classes` in the npz; checked on load) — `t77_labels.PHASES5`.
STAGES5 = ("approach", "grasp", "carry", "place", "retreat")
PROBE_NAME = "hsr5_t77_h4_img3_last"
PROBE_MD5 = "985be91854b46ee354449025b2ae1885"
#: The hidden layer (HF numbering: h_0 = the input embedding, h_l = output of block l, before the final norm).
HIDDEN_LAYER = 4
#: `pi05_hidden.SUMMARY_ROWS` rows the feature keeps: the three camera means and the last valid prefix token
#: (`t77_probe.POOLS["img3+last"]`), concatenated in this order.
FEATURE_ROWS = (0, 1, 2, 4)
FEATURE_DIM = 4 * 2048

#: The 3-stage subtask label each stage stands for at the new-object gate (T82 §2 row 1).
SUBTASK3 = {"approach": "pick", "grasp": "pick", "carry": "place", "place": "place", "retreat": "home"}
#: The HSR6 phase each stage stands for in the four `--hsr6-*` rules (T82 §2 rows ①–④).
RULE_PHASE6 = {"approach": "approach", "grasp": "grasp", "carry": "transport", "place": "placement",
               "retreat": "retreat"}
#: Stages in which the deflection-rate and continuity terms are on (T82 §2 row 6).
AFTER_GRASP_STAGES = frozenset({"carry", "place", "retreat"})
#: Stages in which a CLOSING latch is never released by the reopen rule (T82 §2 row 7).
CLOSING_KEEP_STAGES = frozenset({"carry", "place"})


def default_probe_path() -> pathlib.Path:
    """`AG3S/ag3s/asset/subtask_probe/hsr5_t77_h4_img3_last.npz`, located from this file (no symlinks)."""
    return pathlib.Path(__file__).resolve().parents[1] / "ag3s" / "asset" / "subtask_probe" / f"{PROBE_NAME}.npz"


def load_hsr5_probe(path: Optional[str | pathlib.Path] = None):
    """The 5-stage probe as a `SubtaskProbe` (same `proba` as the serving HSR / HSR6). Refuses a wrong class order /
    dim. The name is the file stem (the T77 npz has no `name` key)."""
    from AG3S.ag3s.stages.subtask_probe import SubtaskProbe

    p = default_probe_path() if path in (None, "default") else pathlib.Path(path)
    probe = SubtaskProbe.load(p)
    if tuple(probe.classes) != STAGES5:
        raise ValueError(f"HSR5 probe {p}: classes {probe.classes} != {STAGES5}")
    if probe.dim != FEATURE_DIM:
        raise ValueError(f"HSR5 probe {p}: dim {probe.dim} != {FEATURE_DIM} (h4 img3+last)")
    return probe


def hsr5_feature(xp, hidden_layer, mask):
    """The 8192-D feature of one sample: ``hidden_layer`` = h_4 ``[T, D]`` (float32), ``mask`` = the prefix mask
    ``[T]``. Exactly T77's offline definition: `pi05_hidden.summary_rows` (the same function, numpy or jax) on that
    layer, rows (0, 1, 2, 4) flattened in order — `t77_probe.load_X` (``A[:, (0, 1, 2, 4)].reshape(n, -1)``)."""
    from AG3S.ag3s.experiments.sources.pi05_hidden import summary_rows

    rows, _ = summary_rows(xp, hidden_layer[None], mask)        # [1, 5, D]
    return rows[0, np.asarray(FEATURE_ROWS)].reshape(-1)


def stage_of(record: Any) -> Optional[str]:
    """`result["hsr5"]` → its argmax stage, or None (no record / unknown class / non-finite probabilities)."""
    if not isinstance(record, Mapping):
        return None
    p = record.get("p")
    if isinstance(p, Mapping) and p:
        vals = [float(p.get(c, np.nan)) for c in STAGES5]
        if not all(np.isfinite(vals)):
            return None
        return STAGES5[int(np.argmax(vals))]       # ties -> the earlier stage (numpy argmax), as the offline code
    arg = record.get("argmax")
    return str(arg) if arg in STAGES5 else None


def subtask3_of(stage: Optional[str]) -> Optional[str]:
    return SUBTASK3.get(stage) if stage is not None else None


def subtask3_probabilities(stage: Optional[str]) -> Optional[dict[str, float]]:
    """The `AG3S.set_subtask` input for a stage: the one-hot of `subtask3_of` (the raw argmax decides; the existing
    `SubtaskDebounce` then confirms it as before), None without a stage (= a request without a 3-stage label)."""
    label = subtask3_of(stage)
    if label is None:
        return None
    return {c: (1.0 if c == label else 0.0) for c in ("pick", "place", "home")}


def rule_phase6(stage: Optional[str]) -> Optional[str]:
    """The HSR6 phase the `--hsr6-*` rules read for a stage (None stays None: the rules' no-label side)."""
    return RULE_PHASE6.get(stage) if stage is not None else None


def after_grasp_on(stage: Optional[str]) -> Optional[bool]:
    """Deflection-rate / continuity terms on (True) or off (False) for a stage; None without a stage (the caller then
    uses the latch rule, today's behaviour)."""
    if stage is None:
        return None
    return stage in AFTER_GRASP_STAGES


@dataclasses.dataclass(frozen=True)
class Hsr5Config:
    """`serve_safe --hsr5 [NPZ]`. `SafePolicy(hsr5=None)` (default) = off: nothing here is read.

    `release_frames`: consecutive requests the closing release needs (task §2: 2)."""

    probe: str = "shipped:" + PROBE_NAME
    release_frames: int = 2
    # --- T85 — begin
    #: `serve_safe --hsr-gt-stage` (analysis only): the stage is the MuJoCo ground-truth stage the client sends, not a
    #: classifier's argmax. False (default) = the record below is the one from before.
    gt_stage: bool = False
    # --- T85 — end

    def record(self) -> dict[str, Any]:
        # --- T85 — begin
        if self.gt_stage:
            return {"probe": self.probe, "release_frames": int(self.release_frames),
                    "decision": "MuJoCo ground-truth stage per request (analysis only, client --safe-gt-stage)",
                    "gt_stage": True}
        # --- T85 — end
        return {"probe": self.probe, "release_frames": int(self.release_frames),
                "decision": "raw argmax per request"}


# --- T85 — begin
GT_PROBE_NAME = "gt"


def gt_stage_record(stage: Optional[str]) -> Optional[dict[str, Any]]:
    """T85 — the client's ground-truth stage as a `result["hsr5"]`-shaped record (`{"p": one-hot, "argmax", "probe":
    "gt"}`), so `stage_of` and every rule after it read it exactly as a classifier's. None (the request carried no
    stage, or an unknown name) = no record = today's "no label" handling."""
    if stage not in STAGES5:
        return None
    return {"p": {c: (1.0 if c == stage else 0.0) for c in STAGES5}, "argmax": stage, "probe": GT_PROBE_NAME}
# --- T85 — end
