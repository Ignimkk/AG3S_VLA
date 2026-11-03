"""T91 — benchmark-agnostic task-phase recogniser: a physical state machine plus the VLA's own gripper intent.

The 5-stage classifier of T77/T82 reads this VLA checkpoint's hidden space and was fitted on this simulator's labels,
so it must be refitted for every robot / benchmark. This module needs **no learning**. Its inputs exist on any
manipulation benchmark that has a gripper and a segmentable target:

  gripper opening   measured, one number per request, normalised per robot (`GripperScale`): the largest opening seen
                    in the first ``open_ref_requests`` requests = 1 (open), ``closed_limit`` = 0
  hand position     the grasping hand's end-effector site (``ee_pos``), same frame as the target / destination
  target            AG3S ATS target centroid (``manipulated.centroid``) and whether it was freshly observed this
                    request (``target_fresh``; an occluded target keeps its last centroid, which is not evidence)
  destination       AG3S destination centroid (``destination.centroid``), or ``None`` while not registered
  VLA intent        the gripper column of the VLA chunk (``chunk_gripper``, first ``intent_rows`` rows) in the same
                    0 (closed) .. 1 (open) scale — "about to close / about to open"

The stages are the ones whose **collision constraint changes** for AG3S (same names as T77): approach · grasp · carry
· place · retreat. ``step(obs) -> (stage, evidence)`` is causal, one call per request.

Rules (all thresholds in `PhaseConfig`; fixed on train scenes only, see `handoff/T91.impl.md`):

  approach  anything not below
  grasp     (target within ``d_near`` of the hand, or the hand closed where the target was when it closed while the
            target is now hidden) ∧ (close intent in the chunk ∨ the opening is shrinking ∨ closed) ∧ not carrying
  carry     closed ∧ not contradicted by the target, and one of
              (a) **co-motion**: over the last ``comotion_window`` requests the freshly observed target centroid moved
                  with the hand (displacement ratio >= ``comotion_ratio``, direction cos >= ``comotion_cos``; judged
                  only when the hand moved >= ``comotion_min_hand_mm``);
              (b) the previous stage was carry/place and the hand is still closed on something (opening >=
                  ``blocked_min_open``, stalled) while the evidence is undecidable (target hidden / hand still);
              (c) ``occluded_bootstrap``: the target was within ``d_near`` when the hand closed, is hidden now (the
                  hand covers it), the closure is stalled on something, and the hand moved >= ``bootstrap_hand_mm``
                  from where it closed. (On RBY1 the apple is hidden in ~98 % of carry requests, so (c) starts nearly
                  every carry; (a) mainly acts as a veto.)
            **Closed but the freshly observed target does not follow (co-motion False) is NOT carry** — closing on
            nothing is never read as holding.
  place     carry ∧ (hand within ``d_dest`` of the destination (xy) ∨ the chunk commands opening); sticky until retreat
  retreat   after any carry: the closure no longer holds the target (opened, slipped, target left behind)

The module is pure Python / numpy: no MuJoCo, no AG3S pipeline, no robot name. Robot-specific things are settings:
`PhaseConfig.chunk_gripper_column`, the gripper scale, and the frame the caller puts positions in.

T91b (``finger_gap_mode="on"``, default **off** = every output above unchanged): carry is judged from the wrist camera.
The caller supplies per request how much of the region between the fingertips shows a surface within finger reach
(``PhaseObservation.finger_gap``, `finger_gap.surface_fraction`; main) and/or how much of it the ATS target points cover
when projected into the wrist camera (``finger_gap_proj``; reference). A request is a *hit* when the evidence of
``finger_gap_source`` is >= its threshold. Then

  carry = closed ∧ opening >= ``blocked_min_open`` (a fully closed hand shows its own fingers there) ∧ gap evidence ∧
          (already carrying ∨ the hand moved >= ``bootstrap_hand_mm`` since the closure began)

  gap evidence: to enter, ``gap_m`` consecutive hits; to stay, at least one hit in the last ``gap_m`` requests.
  Closed with the gap empty is NOT carry (empty hand). Co-motion and the occlusion rule (a)-(c) are not used while the
  evidence is available; when a request has no wrist evidence (``None``) the T91 rules decide it. The other stages
  (grasp, place, retreat) follow the rules above unchanged.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Optional, Sequence

import numpy as np

STAGES = ("approach", "grasp", "carry", "place", "retreat")


class PhaseConfigError(ValueError):
    """Raised for an invalid `PhaseConfig`."""


@dataclasses.dataclass(frozen=True)
class PhaseConfig:
    """Thresholds and robot settings of `PhaseMachine`. The defaults are the T91 values chosen on the 210 train-scene
    runs (`t91_phase_eval select`, `handoff/T91.impl.md`, frozen before any test number was computed); the ones not
    listed there (`open_thr`, `closed_thr`, `shrink_rate`, `intent_*_thr`, `open_ref_requests`, ...) are the
    a-priori values the selection grid kept (`closed_thr`/`open_thr` were flat within 0.3 pp on train). Distances in metres."""

    # --- robot / benchmark settings ---
    chunk_gripper_column: int = 7        # RBY1 left hand; right = 15 (`ChunkLayout.rby1` passthrough columns)
    closed_limit: float = 0.0            # raw opening that counts as fully closed (normalisation zero)
    open_ref_requests: int = 3           # N: the opening reference = max over the first N requests
    chunk_open_value: float = 1.0        # chunk gripper command meaning open
    chunk_closed_value: float = 0.0      # chunk gripper command meaning closed
    # --- gripper state (normalised 0..1) ---
    open_thr: float = 0.90               # >= this = open
    closed_thr: float = 0.85             # <= this = closed (between = moving)
    shrink_rate: float = 0.02            # opening drop per request that counts as "shrinking"
    # --- VLA intent (normalised chunk command) ---
    intent_rows: int = 16                # K: leading chunk rows read
    intent_close_thr: float = 0.85       # a command below this inside K rows = "close soon"
    intent_open_thr: float = 0.90        # a command above this inside K rows while closed = "open soon"
    # --- geometry ---
    d_near: float = 0.06                 # hand – target distance for grasp
    d_dest: float = 0.12                 # hand – destination distance for place
    dest_xy_only: bool = True            # compare in the horizontal plane (destination is a container region)
    # --- co-motion ---
    comotion_window: int = 4             # W requests
    comotion_ratio: float = 0.4          # rho: |target displacement| / |hand displacement|
    comotion_cos: float = 0.8            # c: direction agreement
    comotion_min_hand_mm: float = 15.0   # hand must have moved this much for the test to be decisive
    blocked_min_open: float = 0.10       # a closed hand with opening below this holds nothing
    blocked_max_change: float = 0.03     # opening change per request that still counts as "stalled"
    occluded_bootstrap: bool = True      # let carry start from occlusion + blocked closure alone (no seen co-motion)
    bootstrap_hand_mm: float = 20.0      # ... once the hand carried the closure this far from where it closed
    # --- T91b: wrist-camera "target between the fingers" evidence (default off: T91 behaviour, bit for bit) ---
    open_ref_value: Optional[float] = None   # fixed raw opening that counts as 1 (None: max of the first N requests)
    finger_gap_mode: str = "off"         # "off" | "on"
    finger_gap_source: str = "depth"     # "depth" (surface fraction, main) | "projection" | "both" (and) | "either" (or)
    gap_tau: float = 0.30                # depth surface fraction >= this = hit        (RBY1 train: depth arm)
    gap_tau_proj: float = 0.10           # target projection fraction >= this = hit    (RBY1 train: projection arm)
    gap_m: int = 1                       # consecutive hits to enter carry (and misses to leave it)  (depth arm; projection arm: 3)

    def __post_init__(self) -> None:
        for name in ("open_thr", "closed_thr", "intent_close_thr", "intent_open_thr", "comotion_cos"):
            if not isinstance(getattr(self, name), (int, float)):
                raise PhaseConfigError(f"{name} must be a number")
        if not self.closed_thr <= self.open_thr:
            raise PhaseConfigError("closed_thr must be <= open_thr")
        if self.open_ref_requests < 1 or self.intent_rows < 1 or self.comotion_window < 1:
            raise PhaseConfigError("open_ref_requests, intent_rows, comotion_window must be >= 1")
        if self.chunk_open_value == self.chunk_closed_value:
            raise PhaseConfigError("chunk_open_value must differ from chunk_closed_value")
        if self.d_near <= 0 or self.d_dest <= 0:
            raise PhaseConfigError("d_near and d_dest must be positive")
        if self.finger_gap_mode not in ("off", "on"):
            raise PhaseConfigError("finger_gap_mode must be 'off' or 'on'")
        if self.finger_gap_source not in ("depth", "projection", "both", "either"):
            raise PhaseConfigError("finger_gap_source must be depth / projection / both / either")
        if self.gap_m < 1:
            raise PhaseConfigError("gap_m must be >= 1")
        if self.open_ref_value is not None and self.open_ref_value <= self.closed_limit:
            raise PhaseConfigError("open_ref_value must exceed closed_limit")

    @staticmethod
    def rby1(side: str, **kw: Any) -> "PhaseConfig":
        """RB-Y1 16-D chunk: left gripper column 7, right 15 (`ChunkLayout.rby1`, arm_joint_dim 7)."""
        if side not in ("left", "right"):
            raise PhaseConfigError(f"side must be 'left' or 'right', got {side!r}")
        return PhaseConfig(chunk_gripper_column=7 if side == "left" else 15, **kw)

    @staticmethod
    def panda(**kw: Any) -> "PhaseConfig":
        """Panda / LIBERO 7-D action chunk (6 arm EE-delta + gripper): gripper column 6, command +1 = close, -1 = open;
        opening = finger qpos / 0.04 m (1 = open, `gripper_calibration.json` ``normalisation``); a hand closed on nothing
        settles at 0.0125, on a thin rim at >= 0.038 -> ``blocked_min_open`` = the benchmark's calibrated ``empty_below``
        0.025325. These robot settings (`PANDA_ROBOT`) cannot be overridden through ``kw``; every other threshold is
        taken from ``kw`` (the RBY1-train values) and the defaults."""
        return PhaseConfig(**{**kw, **PANDA_ROBOT})


#: robot-specific settings of the Panda adapter (T91b §3.3: region between the fingers is in `finger_gap`, the rest here)
PANDA_ROBOT = dict(chunk_gripper_column=6, chunk_open_value=-1.0, chunk_closed_value=1.0, open_ref_value=0.04,
                   blocked_min_open=0.025325)


@dataclasses.dataclass
class GripperScale:
    """Per-robot opening normalisation: ``(raw - closed_limit) / (open_ref - closed_limit)`` where ``open_ref`` is the
    largest raw opening in the first N requests (the gripper starts open on every benchmark we target). Until N requests
    are seen the reference is the running maximum, so the value can only be too large early, never negative."""

    closed_limit: float = 0.0
    n_ref: int = 3
    fixed_open_ref: Optional[float] = None   # T91b: robot constant (Panda 0.04 m) instead of the first-N maximum
    _seen: list = dataclasses.field(default_factory=list)

    @property
    def open_ref(self) -> Optional[float]:
        if self.fixed_open_ref is not None:
            return self.fixed_open_ref
        return max(self._seen) if self._seen else None

    def update(self, raw: float) -> float:
        if len(self._seen) < self.n_ref:
            self._seen.append(float(raw))
        span = (self.open_ref or 0.0) - self.closed_limit
        if span <= 1e-9:
            return 1.0
        return float(np.clip((float(raw) - self.closed_limit) / span, 0.0, 1.5))

    def normalise_chunk(self, cmd: Sequence[float], open_value: float, closed_value: float) -> np.ndarray:
        """Chunk gripper command -> 0 closed .. 1 open (a fixed command scale of the policy, not measured)."""
        a = np.asarray(cmd, float)
        return (a - closed_value) / (open_value - closed_value)


@dataclasses.dataclass
class PhaseObservation:
    """What one request provides. Missing things are ``None``. No ground truth, no robot name."""

    gripper_raw: float                            # measured opening of the grasping hand (raw units)
    ee_pos: Optional[Sequence[float]] = None      # grasping-hand end-effector site, shared frame
    target_pos: Optional[Sequence[float]] = None  # ATS target centroid (last known when not fresh)
    target_fresh: bool = False                    # centroid observed this request (ATS state "visible")
    destination_pos: Optional[Sequence[float]] = None
    chunk_gripper: Optional[Sequence[float]] = None   # VLA chunk gripper column, leading rows, raw command scale
    finger_gap: Optional[float] = None            # T91b: wrist-depth surface fraction between the fingertips (0..1)
    finger_gap_proj: Optional[float] = None       # T91b: ATS target projection fraction between the fingertips (0..1)


def _vec(x: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if x is None:
        return None
    a = np.asarray(x, float).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        return None
    return a


class PhaseMachine:
    """Causal 5-stage recogniser. One instance per run (``reset()`` between runs)."""

    def __init__(self, config: Optional[PhaseConfig] = None) -> None:
        self.cfg = config or PhaseConfig()
        self.reset()

    def reset(self) -> None:
        self.scale = GripperScale(self.cfg.closed_limit, self.cfg.open_ref_requests, self.cfg.open_ref_value)
        self.gap_hist: collections.deque = collections.deque(maxlen=self.cfg.gap_m)
        self.hist: collections.deque = collections.deque(maxlen=self.cfg.comotion_window + 1)
        self.prev_open: Optional[float] = None
        self.stage: str = "approach"
        self.ever_carry = False
        self.n = 0
        self._was_closed = False
        self.close_ee: Optional[np.ndarray] = None
        self.close_near = False

    # ------------------------------------------------------------------ evidence
    def comotion(self) -> tuple[Optional[bool], dict]:
        """Does the freshly observed target follow the hand over the history window? None = not decidable."""
        c = self.cfg
        fresh = [(e, t) for e, t, f in self.hist if f and e is not None and t is not None]
        info: dict = {"n_fresh": len(fresh)}
        if len(fresh) < 2:
            return None, info
        (e0, t0), (e1, t1) = fresh[0], fresh[-1]
        dh, dt = e1 - e0, t1 - t0
        nh = float(np.linalg.norm(dh))
        info["hand_mm"] = round(nh * 1e3, 1)
        info["target_mm"] = round(float(np.linalg.norm(dt)) * 1e3, 1)
        if nh * 1e3 < c.comotion_min_hand_mm:
            return None, info
        ratio = float(np.linalg.norm(dt)) / nh
        cos = float(dt @ dh / (np.linalg.norm(dt) * nh + 1e-12))
        info.update(ratio=round(ratio, 3), cos=round(cos, 3))
        return bool(ratio >= c.comotion_ratio and cos >= c.comotion_cos), info

    def _gap_hit(self, obs: PhaseObservation) -> Optional[bool]:
        """T91b: is the target between the fingers in this request? None = no wrist evidence (or mode off)."""
        c = self.cfg
        if c.finger_gap_mode != "on":
            return None
        d = None if obs.finger_gap is None or not np.isfinite(obs.finger_gap) else bool(obs.finger_gap >= c.gap_tau)
        p = (None if obs.finger_gap_proj is None or not np.isfinite(obs.finger_gap_proj)
             else bool(obs.finger_gap_proj >= c.gap_tau_proj))
        src = c.finger_gap_source
        if src == "depth":
            return d
        if src == "projection":
            return p
        if d is None or p is None:                         # both / either need both numbers
            return d if p is None else p
        return (d and p) if src == "both" else (d or p)

    # ------------------------------------------------------------------ one request
    def step(self, obs: PhaseObservation) -> tuple[str, dict]:
        c = self.cfg
        g = self.scale.update(obs.gripper_raw)
        ee, tgt, dst = _vec(obs.ee_pos), _vec(obs.target_pos), _vec(obs.destination_pos)
        fresh = bool(obs.target_fresh) and tgt is not None
        self.hist.append((ee, tgt, fresh))
        rate = 0.0 if self.prev_open is None else g - self.prev_open       # per request
        self.prev_open = g
        is_open, is_closed = g >= c.open_thr, g <= c.closed_thr
        shrinking = rate <= -c.shrink_rate

        close_intent = open_intent = False
        if obs.chunk_gripper is not None and len(obs.chunk_gripper) > 0:
            cmd = self.scale.normalise_chunk(np.asarray(obs.chunk_gripper, float)[: c.intent_rows],
                                             c.chunk_open_value, c.chunk_closed_value)
            close_intent = bool(np.any(cmd < c.intent_close_thr))
            open_intent = bool(np.any(cmd > c.intent_open_thr))

        d_t = None if (ee is None or tgt is None) else float(np.linalg.norm(ee - tgt))
        near = d_t is not None and d_t <= c.d_near
        d_d = None
        if ee is not None and dst is not None:
            d_d = float(np.linalg.norm((ee - dst)[:2] if c.dest_xy_only else ee - dst))
        at_dest = d_d is not None and d_d <= c.d_dest

        # where the hand was, and whether the target was within reach, when this closure began
        if is_closed and not self._was_closed:
            self.close_ee, self.close_near = ee, near
        elif not is_closed:
            self.close_ee, self.close_near = None, False
        self._was_closed = is_closed
        moved = 0.0 if (ee is None or self.close_ee is None) else float(np.linalg.norm(ee - self.close_ee))

        co, co_info = self.comotion()
        prev_carry = self.stage in ("carry", "place")
        blocked = is_closed and g >= c.blocked_min_open and abs(rate) <= c.blocked_max_change

        gap_hit = self._gap_hit(obs)
        if gap_hit is not None:
            self.gap_hist.append(gap_hit)
        gap_on = gap_hit is not None
        gap_ok = None
        if gap_on:
            hist = list(self.gap_hist)
            gap_ok = any(hist) if prev_carry else (len(hist) >= c.gap_m and all(hist))

        carry = False
        if gap_on:
            # T91b: closed ∧ not an empty fully-closed hand ∧ target between the fingers ∧ (carrying already ∨ moved away)
            carry = bool(is_closed and g >= c.blocked_min_open and gap_ok
                         and (prev_carry or moved * 1e3 >= c.bootstrap_hand_mm))
        elif is_closed and co is not False:
            if co is True:
                carry = True
            elif blocked and prev_carry:
                carry = True                       # hidden by the hand / hand still: the closure on something stays
            elif (blocked and c.occluded_bootstrap and self.close_near and not fresh
                  and moved * 1e3 >= c.bootstrap_hand_mm):
                carry = True                       # target gone behind the hand, hand carried it away from the grasp
        if carry:
            sticky = self.stage == "place"
            stage = "place" if (sticky or at_dest or (prev_carry and open_intent)) else "carry"
            self.ever_carry = True
        elif self.ever_carry:
            stage = "retreat"
        elif (near or (is_closed and self.close_near and not fresh)) and (close_intent or shrinking or is_closed):
            stage = "grasp"
        else:
            stage = "approach"
        self.stage = stage
        self.n += 1
        r = lambda v: None if v is None else round(v * 1e3, 1)  # noqa: E731
        ev = {"open_norm": round(g, 3), "rate": round(rate, 3), "closed": is_closed, "open": is_open,
              "close_intent": close_intent, "open_intent": open_intent, "d_target_mm": r(d_t), "d_dest_mm": r(d_d),
              "near": near, "fresh": fresh, "comotion": co, "blocked": blocked, "moved_since_close_mm": r(moved),
              **{f"co_{k}": v for k, v in co_info.items()}}
        if c.finger_gap_mode == "on":
            ev.update(gap_depth=None if obs.finger_gap is None else round(float(obs.finger_gap), 3),
                      gap_proj=None if obs.finger_gap_proj is None else round(float(obs.finger_gap_proj), 3),
                      gap_hit=gap_hit, gap_ok=gap_ok, gap_used=gap_on)
        return stage, ev
