"""IFG-S2SERVE — the **feature-memory confidence alert** (IFG R11) on the server (`--confidence-gate on`, default off).

The offline definition is the locked study code: `experiments/studies/ifg_memory.py` (`episode(center=True)`, the
R10b / R11 path) scored by `ifg_confidence.py`, with the LOCK arm `LOCK_heldout.json → arms.confidence_gate`
(`score = q_mem_chain:prefix_L8 (centered)`, `abstain_if_below = 0.5904207590672415`). Held-out evidence with the
FK-sphere robot mask: `docs/handoff/IFG-S2FK.verify.json` (`memory_gate.fk.gate_*`). This module computes the same
score per request, in time order, from what the server has:

| input | offline (`ifg_memory.episode`) | here |
|---|---|---|
| region | `r6_choice(geo, cfg, N)` → `chosen` (top1-only-ns:N, robot mask of the run) | `attention_surface.select_target_surface(arm=ARM_SPHERES)` on the request's depth + the serving FK-sphere mask (never the rendered mask), `selected_nodes` → the same boolean node mask (the selection is the same: `region_gate`'s `px` argument changes only the prototype score `q`, not the top-1 region; `tests/ag3s/test_ifg_s2serve_gate.py` checks it on held-out frames) |
| feature | `prefix_L8.npz resid_out_image` (768 × 2048, stored float16) | `AttentionPolicy` result `IFG_PREFIX_L8_KEY` — the same tensor (`attention_source.prefix_resid_impl`, the collector's layer loop) with the same float16 rounding (`dump_rounding`) |
| feature space | `U = normalize(normalize(f) − mean_rows)` (`center=True`) | `unit_patch_features` — the same two lines |
| candidates | connected regions of the non-support graph, ≥ `MIN_NODES` nodes, bbox max side ≤ `MAX_SIDE` | `candidate_regions` — the same lines (constants imported) |
| memory | `chain`: previous frame's final choice (switch at δ 0.1, else the R6 choice) | `ConfidenceGate.memory`, emptied by `reset()` (the server's episode boundary) |

**Score** = `f_ch · M` (unit region feature of this frame's choice · chain memory), defined only when a memory, a
chosen region and at least one candidate exist. **Alert** = score < threshold. No score → no alert; the reason is
`why` (`no_feature` · `no_region` · `no_memory` · `no_candidates`). What the alert does (withhold the target exemption)
is the pipeline's business (`AG3S._confidence_gate_step`); this module only scores.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

import numpy as np

#: `LOCK_heldout.json → arms.confidence_gate.abstain_if_below` (10th percentile of the dev score, 147 frames).
GATE_THRESHOLD = 0.5904207590672415
#: The chain memory follows the δ = 0.1 arm (`ifg_memory`: "chain = this frame's final Mem-chain choice at delta 0.1").
MEMORY_SWITCH_DELTA = 0.1
#: The score's feature (`q_mem_chain:prefix_L8`).
FEATURE = "prefix_L8"
#: Image rows of the prefix (3 cameras × 16 × 16, `CAMERA_ORDER`).
N_IMAGE_TOKENS = 768
GATE_MODES = ("off", "on")


def dump_rounding(resid: np.ndarray) -> np.ndarray:
    """The held-out collector's storage step (`hc_dump.py`: float16 when every value is finite in float16, else float32),
    returned as float32 — what `ifg_heldout_eval.load_heldout_frame` reads back."""
    r = np.asarray(resid, np.float32)
    if not np.isfinite(r).all():
        return r
    with np.errstate(over="ignore"):   # an overflow is the collector's float32 fallback, not an error
        h = r.astype(np.float16)
    return h.astype(np.float32) if np.isfinite(h).all() else r


def unit_patch_features(resid: np.ndarray) -> np.ndarray:
    """`ifg_memory.episode(center=True)`: ``U = normalize(f)``, then ``normalize(U − U.mean(0))`` (float64)."""
    from AG3S.ag3s.experiments.studies import ifg_reader as R

    u = R.normalize(np.asarray(resid))
    return R.normalize(u - u.mean(0, keepdims=True))


def candidate_regions(geo):
    """`(comp, {region id: patch-pixel row sum (768,)})` — `ifg_memory.episode`'s candidate block, verbatim."""
    from AG3S.ag3s.experiments.studies import ifg_memory as M
    from AG3S.ag3s.experiments.studies import ifg_round_mass as RM

    comp = RM.regions(geo, ~geo.support)
    ok = comp >= 0
    nreg = comp.max() + 1
    size = np.bincount(comp[ok], minlength=nreg)
    lo = np.full((nreg, 3), np.inf)
    hi = np.full((nreg, 3), -np.inf)
    np.minimum.at(lo, comp[ok], geo.pos[ok])
    np.maximum.at(hi, comp[ok], geo.pos[ok])
    cand = np.flatnonzero((size >= M.MIN_NODES) & ((hi - lo).max(1) <= M.MAX_SIDE))
    px = geo.patch_px.tocsr()
    reg_w = {int(r): np.asarray(px[comp == r].sum(0)).ravel() for r in cand}
    return comp, reg_w


class ConfidenceGate:
    """Per-episode chain memory + the R11 score. One per AG3S instance; `reset()` at the episode boundary."""

    def __init__(self, *, threshold: float = GATE_THRESHOLD):
        self.threshold = float(threshold)
        if not np.isfinite(self.threshold):
            raise ValueError(f"confidence gate threshold must be finite, got {threshold!r}")
        self.memory: Optional[np.ndarray] = None
        self.n_frames = 0

    def reset(self) -> None:
        self.memory = None
        self.n_frames = 0

    def record_meta(self) -> dict[str, Any]:
        return {"score": "q_mem_chain:prefix_L8 (centered)", "threshold": self.threshold,
                "memory": "chain", "switch_delta": MEMORY_SWITCH_DELTA}

    def step(self, geo, chosen: Optional[np.ndarray], resid: Optional[np.ndarray]) -> dict[str, Any]:
        """One frame, in time order. `geo` = this frame's `ifg_graph` geometry (None when there is none), `chosen` =
        boolean mask over its nodes (the selected region; None / empty = no region), `resid` = (768, D) prefix-L8 image
        rows (None = no feature). Returns the record; the memory is updated as `ifg_memory` does."""
        from AG3S.ag3s.experiments.studies import ifg_memory as M

        t0 = time.perf_counter()
        self.n_frames += 1
        rec: dict[str, Any] = {"gate_score": None, "gate_alert": False, "gate_why": None,
                               "threshold": self.threshold, "had_memory": self.memory is not None,
                               "switched": False, "memory_updated": False, "ms": {}}
        if resid is None:
            rec["gate_why"] = "no_feature"
            rec["ms"]["total"] = round((time.perf_counter() - t0) * 1e3, 3)
            return rec
        resid = np.asarray(resid)
        if resid.ndim != 2 or resid.shape[0] != N_IMAGE_TOKENS:
            rec["gate_why"] = f"no_feature (shape {tuple(resid.shape)})"
            rec["ms"]["total"] = round((time.perf_counter() - t0) * 1e3, 3)
            return rec
        chosen = None if chosen is None or geo is None else np.asarray(chosen, bool)
        if chosen is None or not chosen.any():
            # offline: f_ch is None → no score, the chain memory keeps its value
            rec["gate_why"] = "no_region"
            rec["ms"]["total"] = round((time.perf_counter() - t0) * 1e3, 3)
            return rec
        t = time.perf_counter()
        U = unit_patch_features(resid)
        rec["ms"]["features"] = round((time.perf_counter() - t) * 1e3, 3)
        t = time.perf_counter()
        comp, reg_w = candidate_regions(geo)
        ok = comp >= 0
        px = geo.patch_px.tocsr()
        chosen_w = np.asarray(px[chosen].sum(0)).ravel()
        r6_reg = int(np.bincount(comp[chosen & ok]).argmax()) if (chosen & ok).any() else -1
        f_ch = M.unit_rows(chosen_w, U)
        rec["ms"]["regions"] = round((time.perf_counter() - t) * 1e3, 3)
        rec["n_candidates"] = int(len(reg_w))
        t = time.perf_counter()
        Mem = self.memory
        cos = {} if Mem is None else {r: float(M.unit_rows(w, U) @ Mem) for r, w in reg_w.items()}
        switched = False
        best = None
        if Mem is None:
            rec["gate_why"] = "no_memory"
        elif not cos:
            rec["gate_why"] = "no_candidates"
        else:
            cur = float(f_ch @ Mem)
            best = max(cos, key=cos.get)
            switched = bool(best != r6_reg and cos[best] - cur >= MEMORY_SWITCH_DELTA)
            rec["gate_score"] = cur
            rec["gate_alert"] = bool(cur < self.threshold)
            rec["best_candidate_cos"] = float(cos[best])
        # memory update after this frame's decision (offline order): switched → the best region, else this choice
        if switched:
            self.memory = M.unit_rows(reg_w[best], U)
        else:
            self.memory = f_ch
        rec["switched"] = switched
        rec["memory_updated"] = True
        rec["ms"]["score"] = round((time.perf_counter() - t) * 1e3, 3)
        rec["ms"]["total"] = round((time.perf_counter() - t0) * 1e3, 3)
        return rec


def surface_selection(views: Sequence[Any]):
    """The gate's region: `select_target_surface` (top1-only-ns:N, the views' robot mask), code images skipped.
    Returns `(geometry | None, chosen node mask | None, surface record)`."""
    from AG3S.ag3s.runtime.attention_surface import ARM_SPHERES, select_target_surface

    res = select_target_surface(views, arm=ARM_SPHERES, keep_debug=True, codes=False)
    geo = res.geometry
    chosen = None
    if geo is not None and len(res.selected_nodes):
        chosen = np.zeros(len(geo.pos), bool)
        chosen[res.selected_nodes] = True
    return geo, chosen, dict(res.record)


__all__ = ["ConfidenceGate", "FEATURE", "GATE_MODES", "GATE_THRESHOLD", "MEMORY_SWITCH_DELTA", "N_IMAGE_TOKENS",
           "candidate_regions", "dump_rounding", "surface_selection", "unit_patch_features"]
