"""T48a — pure numpy pieces of the HSR counterfactual (no openpi, no video, no storage reads).

HSR (Hidden-state Sub-task Recognition) = the `kv_L4_v1` linear probe on the frozen 16D VLA prefix
feature (`AG3S/ag3s/stages/subtask_probe.py`). T48a changes one input at a time and asks whether
the probe (and three baselines) follow the change:

    C0   nothing changed                         gt = dataset phase class of the frame
    C1   prompt target noun -> each distractor   gt = pick (the distractor is never touched)
    C1p  same target, the other two templates    gt = original
    C2g  used-arm gripper value flipped          gt = original (the image shows the real state)
    C2s  whole 16D state from frame B            gt = original (image side)
    C2i  three images from frame B               gt = class of B (image side)

C2s / C2i come from the same (A, B) pair: C2s = image A + state B, C2i = image B + state A.

Everything here is deterministic numpy so `tests/ag3s/test_t48_hsr_counterfactual.py` can pin it on
synthetic data. Sources of each definition are cited inline and collected in `T48.impl.md`.

Conventions that the task leaves open (fixed here, listed in `T48.impl.md`):

* **A counterfactual replaces the current observation only.** History-based baselines (simple rule
  ever-closed, strong-rule proxy) see the *host* frame's original history up to t-1 and the
  variant's state at t. The frame-index baseline uses the host frame's clock.
* The host of every variant is frame A (the frame whose row the variant is attached to).
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

# ----------------------------------------------------------------------------- constants

#: Probe / label class order (`subtask_probe.py:48` CLASSES; SUBTASK-b `probe.py:28`).
CLASSES = ("pick", "place", "home")
PICK, PLACE, HOME = 0, 1, 2
#: Variant families, in the order they are generated for a frame.
VARIANTS = ("C0", "C1", "C1p", "C2g", "C2s", "C2i")
#: Gripper dims of the 16D state: [left arm 0-6, left gripper 7, right arm 8-14, right gripper 15]
#: (`aloha_policy.py:160` layout comment; checked on the data: `used_arm` left -> state[7] closes,
#: right -> state[15], 50/50 val episodes).
GRIPPER_INDEX = {"left": 7, "right": 15}
#: SUBTASK-b simple-rule closed threshold: `min(s7, s15) < 0.9` (`probe.py:104`, `extract.py:206`).
RULE_CLOSED_BELOW = 0.9
#: Fruits of the dataset and their free-joint qpos start (SUBTASK-b `extract.py:60`).
FRUITS = ("apple", "banana", "orange", "pear")
FRUIT_QPOS = {"apple": 38, "banana": 45, "orange": 52, "pear": 59}
CRATE_QPOS = 31  # SUBTASK-b `extract.py:67` q[31:34]
#: SUBTASK-b closed-loop GT -> class (`probe.py:29` GT_MAP).
GT_MAP = {"reach": PICK, "held": PLACE, "released": HOME}


def phase_to_class(phase) -> np.ndarray:
    """Dataset `phase_index` -> class; -1 for phases outside 0-13 (SUBTASK-b `probe.py:33-39`;
    `kv_L4_v1.json` `class_from_phase_index`: pick 0-3, place 4-8, home 9-13)."""
    p = np.asarray(phase)
    c = np.full(p.shape, -1, dtype=np.int64)
    c[(p >= 0) & (p <= 3)] = PICK
    c[(p >= 4) & (p <= 8)] = PLACE
    c[(p >= 9) & (p <= 13)] = HOME
    return c


def closed_loop_gt(qpos: np.ndarray, state: np.ndarray, fruit: str) -> str:
    """SUBTASK-b `extract.py:63-70` (`closed_loop_gt`, itself `subtask_probe/subtask_episode.py:
    ground_truth()`): gripper closed (min(s7, s15) < 0.9) -> held; else the fruit within 0.15 m (xy)
    of the crate and below z 1.0 -> released; else reach."""
    q, s = np.asarray(qpos, np.float64), np.asarray(state, np.float64)
    a = FRUIT_QPOS[fruit]
    obj, crate = q[a:a + 3], q[CRATE_QPOS:CRATE_QPOS + 3]
    closed = bool(min(s[7], s[15]) < RULE_CLOSED_BELOW)
    in_crate = bool(np.linalg.norm(obj[:2] - crate[:2]) < 0.15 and obj[2] < 1.0)
    return "held" if closed else ("released" if in_crate else "reach")


# ----------------------------------------------------------------------------- prompts

_FRUIT_RE = re.compile(r"\b(" + "|".join(FRUITS) + r")\b")


def prompt_template(prompt: str) -> tuple[str, str]:
    """`(template, fruit)` of a dataset prompt: the single fruit noun replaced by ``{fruit}``.

    Raises when the prompt names zero or several fruits (then a noun swap is ambiguous).
    """
    found = _FRUIT_RE.findall(prompt)
    if len(found) != 1:
        raise ValueError(f"prompt must name exactly one fruit, got {found!r}: {prompt!r}")
    fruit = found[0]
    return _FRUIT_RE.sub("{fruit}", prompt), fruit


def templates_from_tasks(task_prompts: Iterable[str]) -> list[str]:
    """The distinct templates of `meta/tasks.jsonl`, sorted. Each must occur with every fruit."""
    by_t: dict[str, set] = {}
    for p in task_prompts:
        t, f = prompt_template(p)
        by_t.setdefault(t, set()).add(f)
    for t, fs in by_t.items():
        if set(fs) != set(FRUITS):
            raise ValueError(f"template {t!r} occurs with {sorted(fs)}, not all of {FRUITS}")
    return sorted(by_t)


def fill(template: str, fruit: str) -> str:
    if template.count("{fruit}") != 1:
        raise ValueError(f"template must hold exactly one {{fruit}}: {template!r}")
    return template.replace("{fruit}", fruit)


# ----------------------------------------------------------------------------- gripper flip


@dataclasses.dataclass(frozen=True)
class GripperReps:
    """Representative open / closed gripper values (dataset distribution, see `train_gripper_reps`)."""

    open: float
    closed: float
    threshold: float = RULE_CLOSED_BELOW


def flip_gripper(state: np.ndarray, gripper_index: int, reps: GripperReps) -> np.ndarray:
    """C2g: a copy of the 16D state with only the used-arm gripper flipped open <-> closed.

    Closed = value below `reps.threshold` (the SUBTASK-b rule threshold, 0.9); closed -> `reps.open`,
    open -> `reps.closed`. The other 15 dims (incl. the idle arm's gripper) are untouched.
    """
    s = np.array(state, dtype=np.float64, copy=True)
    if s.shape != (16,):
        raise ValueError(f"expected a 16D state, got {s.shape}")
    if gripper_index not in (7, 15):
        raise ValueError(f"gripper index must be 7 or 15, got {gripper_index}")
    s[gripper_index] = reps.open if s[gripper_index] < reps.threshold else reps.closed
    return s


# ----------------------------------------------------------------------------- units and variants


@dataclasses.dataclass
class Unit:
    """One episode (dataset) or one recorded run, with its sampled frames.

    `seq_states` is the full gripper/state history the rules read (dataset: every parquet frame;
    recorded run: every record). `rows` index the sampled frames into it.
    """

    unit_id: str
    set_name: str                      # "val" | "ep1807"
    prompt: str
    target: str
    distractors: tuple[str, ...]
    gripper_index: int
    seq_states: np.ndarray             # (T, 16) float64
    rows: np.ndarray                   # (N,) int, index into seq_states
    clock: np.ndarray                  # (N,) int, frame_index (dataset) / t_step (run)
    y: np.ndarray                      # (N,) int class (gt of C0)
    proxy_cadence: int                 # history stride of the strong-rule proxy (requests)
    seed_key: int                      # stable integer for the pair RNG
    extra: dict = dataclasses.field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.rows))

    def state(self, i: int) -> np.ndarray:
        return np.asarray(self.seq_states[int(self.rows[i])], np.float64)


@dataclasses.dataclass
class VariantTable:
    """Flat per-unit table of variants (one row per model input)."""

    variant: np.ndarray     # (V,) str  family
    sub: np.ndarray         # (V,) str  distractor fruit / template index / ""
    host: np.ndarray        # (V,) int  sampled row of the host frame A
    image_row: np.ndarray   # (V,) int  row whose three images are fed
    state_row: np.ndarray   # (V,) int  row whose state the variant starts from
    prompt: np.ndarray      # (V,) str
    state: np.ndarray       # (V, 16) float64 — the state actually fed
    gt: np.ndarray          # (V,) int
    y_image: np.ndarray     # (V,) int class of the image source
    y_state: np.ndarray     # (V,) int class of the state source

    def __len__(self) -> int:
        return int(len(self.variant))

    def as_arrays(self, prefix: str = "") -> dict[str, np.ndarray]:
        return {prefix + f.name: np.asarray(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def from_arrays(cls, d: Mapping[str, np.ndarray], prefix: str = "") -> "VariantTable":
        return cls(**{f.name: np.asarray(d[prefix + f.name]) for f in dataclasses.fields(cls)})

    def digest(self) -> str:
        """sha256 over everything that defines the model inputs and labels (plan identity)."""
        h = hashlib.sha256()
        for f in dataclasses.fields(self):
            a = np.asarray(getattr(self, f.name))
            if a.dtype.kind in "US":
                h.update("\x1f".join(str(x) for x in a.reshape(-1)).encode())
            else:
                h.update(np.ascontiguousarray(a.astype(np.float64 if a.dtype.kind == "f" else np.int64)).tobytes())
            h.update(b"|")
        return h.hexdigest()


def sample_pairs(y: np.ndarray, seed: int, seed_key: int) -> np.ndarray:
    """C2 partner B for every row A: same unit, different class, fixed seed.

    For row A (class c): a class c' != c is drawn uniformly among the classes present in the unit,
    then B uniformly among the rows of class c'. RNG = `default_rng([seed, seed_key])`, consumed in
    row order, so the result depends only on (seed, unit, labels). Rows with y < 0 and rows whose
    unit has no other class get -1.
    """
    y = np.asarray(y, np.int64)
    rng = np.random.default_rng([int(seed), int(seed_key)])
    present = sorted(int(c) for c in np.unique(y[y >= 0]))
    rows_of = {c: np.flatnonzero(y == c) for c in present}
    out = np.full(len(y), -1, np.int64)
    for a in range(len(y)):
        if y[a] < 0:
            continue
        others = [c for c in present if c != y[a]]
        if not others:
            continue
        c2 = others[int(rng.integers(len(others)))]
        cand = rows_of[c2]
        out[a] = int(cand[int(rng.integers(len(cand)))])
    return out


def build_variants(unit: Unit, templates: Sequence[str], reps: GripperReps, *, seed: int = 0,
                   families: Sequence[str] = VARIANTS) -> VariantTable:
    """All variants of one unit, frame by frame in `VARIANTS` order (C1 in distractor order, C1p in
    template order). Rows with y < 0 are skipped (SUBTASK-b `probe.py:122` drops them)."""
    template, fruit = prompt_template(unit.prompt)
    if fruit != unit.target:
        raise ValueError(f"{unit.unit_id}: prompt fruit {fruit!r} != metadata target {unit.target!r}")
    if template not in templates:
        raise ValueError(f"{unit.unit_id}: template {template!r} not among {list(templates)}")
    dis = tuple(unit.distractors)
    if len(dis) != 3 or unit.target in dis or len(set(dis)) != 3:
        raise ValueError(f"{unit.unit_id}: distractors {dis} (target {unit.target})")
    other_t = [(i, t) for i, t in enumerate(templates) if t != template]
    pairs = sample_pairs(unit.y, seed, unit.seed_key)
    rec: dict[str, list] = {f.name: [] for f in dataclasses.fields(VariantTable)}

    def add(variant, sub, host, image_row, state_row, prompt, state, gt):
        rec["variant"].append(variant)
        rec["sub"].append(sub)
        rec["host"].append(host)
        rec["image_row"].append(image_row)
        rec["state_row"].append(state_row)
        rec["prompt"].append(prompt)
        rec["state"].append(np.asarray(state, np.float64))
        rec["gt"].append(int(gt))
        rec["y_image"].append(int(unit.y[image_row]))
        rec["y_state"].append(int(unit.y[state_row]))

    for a in range(unit.n):
        ya = int(unit.y[a])
        if ya < 0:
            continue
        sa = unit.state(a)
        if "C0" in families:
            add("C0", "", a, a, a, unit.prompt, sa, ya)
        if "C1" in families:
            for d in dis:
                add("C1", d, a, a, a, fill(template, d), sa, PICK)
        if "C1p" in families:
            for ti, t in other_t:
                add("C1p", f"T{ti}", a, a, a, fill(t, unit.target), sa, ya)
        if "C2g" in families:
            add("C2g", "", a, a, a, unit.prompt, flip_gripper(sa, unit.gripper_index, reps), ya)
        b = int(pairs[a])
        if b >= 0:
            if "C2s" in families:   # image A + state B; answer = image side (A)
                add("C2s", "", a, a, b, unit.prompt, unit.state(b), ya)
            if "C2i" in families:   # image B + state A; answer = image side (B)
                add("C2i", "", a, b, a, unit.prompt, sa, int(unit.y[b]))
    n = len(rec["variant"])
    return VariantTable(
        variant=np.asarray(rec["variant"], dtype="U4"),
        sub=np.asarray(rec["sub"], dtype="U16"),
        host=np.asarray(rec["host"], np.int64),
        image_row=np.asarray(rec["image_row"], np.int64),
        state_row=np.asarray(rec["state_row"], np.int64),
        prompt=np.asarray(rec["prompt"], dtype="U128"),
        state=np.asarray(rec["state"], np.float64).reshape(n, 16),
        gt=np.asarray(rec["gt"], np.int64),
        y_image=np.asarray(rec["y_image"], np.int64),
        y_state=np.asarray(rec["y_state"], np.int64),
    )


# ----------------------------------------------------------------------------- state string


def discretize_state(state: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """numpy replica of the serving state string (for reporting only; extraction uses the policy's
    own `_input_transform`): quantile Normalize (`transforms.py:141-145`, float32 like
    `extract.py:134`) then `np.digitize(state, linspace(-1, 1, 257)[:-1]) - 1`
    (`tokenizer.py:26`). Returns (..., 16) ints in [-1, 255]."""
    s = np.asarray(state, np.float32)
    q01 = np.asarray(q01, np.float32)[..., : s.shape[-1]]
    q99 = np.asarray(q99, np.float32)[..., : s.shape[-1]]
    norm = (s - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    return np.digitize(norm, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1


# ----------------------------------------------------------------------------- baselines


def gripper_min(states: np.ndarray) -> np.ndarray:
    s = np.asarray(states, np.float64)
    return np.minimum(s[..., 7], s[..., 15])


def simple_rule(state_now: np.ndarray, closed_before: bool) -> int:
    """SUBTASK-b rule (`probe.py:101-105`, history `extract.py:206`): closed now -> place; else
    closed at any earlier frame of the episode -> home; else pick. `closed_before` covers frames
    < t of the host's original history; the frame t itself is the (variant) `state_now`."""
    closed = bool(gripper_min(state_now) < RULE_CLOSED_BELOW)
    if closed:
        return PLACE
    return HOME if closed_before else PICK


@dataclasses.dataclass(frozen=True)
class ProxyConfig:
    """Offline strong-rule **proxy** thresholds, copied from `AG3S/trajopt/grasp_latch.py`
    `LatchConfig` (gripper_closed_below 0.85, release_frames 2, settle_epsilon 0.02, min_drop 0.05,
    empty_below 0.10). The real latch also needs executed-command feedback (`blocked_gap`,
    `applied_gripper`) and the manipulated-object reach — not available offline, so this is a
    proxy and not the strong rule."""

    closed_below: float = 0.85
    release_frames: int = 2
    settle_epsilon: float = 0.02
    min_drop: float = 0.05
    empty_below: float = 0.10


_PRE, _CLOSING, _HELD, _PLACED = 0, 1, 2, 3
_PROXY_CLASS = {_PRE: PICK, _CLOSING: PICK, _HELD: PLACE, _PLACED: HOME}


def proxy_run(g_seq: Sequence[float], cfg: ProxyConfig = ProxyConfig()) -> int:
    """Run the proxy latch over a sequence of gripper openings (one value per request) and return
    the class of the final state.

        PRE     -(g < closed_below)-------------------------------> CLOSING   (pick)
        CLOSING -(g >= closed_below)------------------------------> PRE       (attempt abandoned)
        CLOSING -(|g - g_prev| < settle, g_pre - g >= min_drop,
                  g > empty_below)--------------------------------> HELD      (place)
        HELD    -(g >= closed_below for release_frames requests)--> PLACED    (home, sticky)

    `g_pre` is the opening of the request before the close (the latch's `opening_before`).
    """
    st, prev, g_pre, open_streak = _PRE, None, 1.0, 0
    for g in g_seq:
        g = float(g)
        if st == _PRE:
            if g < cfg.closed_below:
                st, g_pre = _CLOSING, (prev if prev is not None else 1.0)
        elif st == _CLOSING:
            if g >= cfg.closed_below:
                st = _PRE
            elif (prev is not None and abs(g - prev) < cfg.settle_epsilon
                  and (g_pre - g) >= cfg.min_drop and g > cfg.empty_below):
                st = _HELD
        elif st == _HELD:
            if g >= cfg.closed_below:
                open_streak += 1
                if open_streak >= cfg.release_frames:
                    st = _PLACED
            else:
                open_streak = 0
        prev = g
    return _PROXY_CLASS[st]


def proxy_history(seq_g: np.ndarray, seq_idx: int, cadence: int) -> np.ndarray:
    """The proxy's request history before `seq_idx`: values at seq_idx - cadence, - 2 cadence, ...
    (>= 0), oldest first. Dataset: cadence 8 = one request per open-loop chunk (`meta.json`
    `open_loop_horizon` 8 of the recorded runs); recorded run: cadence 1 (each record is a request)."""
    if cadence < 1:
        raise ValueError("cadence must be >= 1")
    idx = np.arange(seq_idx - cadence, -1, -cadence)[::-1]
    return np.asarray(seq_g, np.float64)[idx]


def frame_index_majority(clock: np.ndarray, y: np.ndarray) -> dict[int, int]:
    """SUBTASK-b controls C1 (`controls.py:35-37`): majority class per frame_index over the train
    frames (ties -> lowest class, `np.bincount(...).argmax()`)."""
    clock, y = np.asarray(clock, np.int64), np.asarray(y, np.int64)
    keep = y >= 0
    clock, y = clock[keep], y[keep]
    return {int(t): int(np.bincount(y[clock == t], minlength=3).argmax()) for t in np.unique(clock)}


def frame_index_predict(t: int, majority: Mapping[int, int], *, fallback: str = "zero") -> tuple[int, bool]:
    """`(class, used_fallback)`. `fallback="zero"` is SUBTASK-b's `maj.get(t, 0)` (`controls.py:38`);
    `"nearest"` takes the nearest train frame_index (ties -> the smaller), used for the recorded
    run whose 8-step clock is not on the train stride-6 grid."""
    t = int(t)
    if t in majority:
        return int(majority[t]), False
    if fallback == "zero":
        return PICK, True
    if fallback == "nearest":
        keys = np.asarray(sorted(majority), np.int64)
        k = keys[np.argmin(np.abs(keys - t) * 2 + (keys > t))]
        return int(majority[int(k)]), True
    raise ValueError(f"unknown fallback {fallback!r}")


def baseline_predictions(unit: Unit, table: VariantTable, majority: Mapping[int, int], *,
                         fallback: str, proxy_cfg: ProxyConfig = ProxyConfig()) -> dict[str, np.ndarray]:
    """`{"rule", "proxy", "frame_index"}` -> (V,) class per variant row. Host = frame A: history is
    A's original history before t; the variant's state is the frame-t input; the clock is A's."""
    g_seq = gripper_min(unit.seq_states)
    closed_seq = g_seq < RULE_CLOSED_BELOW
    closed_before = np.concatenate([[False], np.cumsum(closed_seq) > 0])  # any closed in [0, i)
    out = {k: np.empty(len(table), np.int64) for k in ("rule", "proxy", "frame_index")}
    fi_used_fallback = np.zeros(len(table), bool)
    for v in range(len(table)):
        a = int(table.host[v])
        i = int(unit.rows[a])
        s = table.state[v]
        out["rule"][v] = simple_rule(s, bool(closed_before[i]))
        hist = proxy_history(g_seq, i, unit.proxy_cadence)
        out["proxy"][v] = proxy_run(np.concatenate([hist, [gripper_min(s)]]), proxy_cfg)
        out["frame_index"][v], fi_used_fallback[v] = frame_index_predict(unit.clock[a], majority,
                                                                         fallback=fallback)
    out["_frame_index_fallback"] = fi_used_fallback
    return out


# ----------------------------------------------------------------------------- metrics


def _ratio(k: int, n: int) -> Optional[float]:
    return float(k) / float(n) if n else None


def episode_summary(per_unit: Mapping[str, tuple[int, int]]) -> dict:
    """Per-episode ratios k/n (units with n = 0 dropped) -> median, IQR, plus the frame-pooled
    ratio. Frames are not independent trials; both views are reported side by side."""
    ks = [k for k, n in per_unit.values() if n]
    ns = [n for k, n in per_unit.values() if n]
    r = np.asarray([k / n for k, n in zip(ks, ns)], np.float64)
    pooled_k, pooled_n = int(sum(ks)), int(sum(ns))
    return {
        "pooled": _ratio(pooled_k, pooled_n), "k": pooled_k, "n": pooled_n,
        "n_episodes": int(len(r)),
        "episode_median": float(np.median(r)) if len(r) else None,
        "episode_q25": float(np.percentile(r, 25)) if len(r) else None,
        "episode_q75": float(np.percentile(r, 75)) if len(r) else None,
    }


def _group(units: np.ndarray, hit: np.ndarray, mask: np.ndarray) -> dict:
    per = {}
    for u in np.unique(units):
        m = mask & (units == u)
        per[str(u)] = (int(hit[m].sum()), int(m.sum()))
    return episode_summary(per)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> list[list[int]]:
    cm = np.zeros((k, k), np.int64)
    for t, p in zip(np.asarray(y_true, np.int64), np.asarray(y_pred, np.int64)):
        if 0 <= t < k and 0 <= p < k:
            cm[t, p] += 1
    return cm.tolist()


def score_method(units: np.ndarray, variant: np.ndarray, sub: np.ndarray, host_key: np.ndarray,
                 gt: np.ndarray, y_host: np.ndarray, y_image: np.ndarray, y_state: np.ndarray,
                 pred: np.ndarray, pair_key: Optional[np.ndarray] = None) -> dict:
    """All T48 numbers for one method on one set. Every input is a flat (V,) array over all units.

    `host_key` identifies the host frame (unit + row) so C1p / C2g can be compared with the C0
    prediction of the same frame. `y_host` = C0 gt of the host frame. Rows with `pred < 0` (missing
    features) are excluded everywhere and counted in `n_missing`.

    Returns:
      accuracy[family]               pred == gt                        (pooled + episode median/IQR)
      C0.confusion                   gt x pred
      C1.to_pick[orig class]         pred == pick, by the host's original class
      C1.matrix                      orig class x pred (counts; the stacked bar)
      C1.to_pick_by_distractor       pooled, per distractor fruit
      C1p.retain[orig class|all]     pred == C0 pred of the same host
      C2g.changed[orig class|all]    pred != C0 pred of the same host
      C2g.matrix                     orig class x pred
      C2s / C2i.follow               state side / image side / other (y_state != y_image always)
      C2_pairs.joint                 (C2s outcome, C2i outcome) 3x3 counts over pairs
    """
    units, variant = np.asarray(units).astype(str), np.asarray(variant).astype(str)
    pred = np.asarray(pred, np.int64)
    valid = pred >= 0
    res: dict = {"n_rows": int(len(pred)), "n_missing": int((~valid).sum())}
    res["accuracy"] = {}
    for fam in VARIANTS:
        m = valid & (variant == fam)
        if m.any():
            res["accuracy"][fam] = _group(units, pred == gt, m)
    m0 = valid & (variant == "C0")
    res["C0"] = {"confusion": confusion(gt[m0], pred[m0])}
    c0_pred = {str(k): int(p) for k, p in zip(np.asarray(host_key)[m0], pred[m0])}
    hk = np.asarray(host_key).astype(str)
    base = np.asarray([c0_pred.get(k, -1) for k in hk], np.int64)

    m1 = valid & (variant == "C1")
    if m1.any():
        res["C1"] = {
            "to_pick": {CLASSES[c]: _group(units, pred == PICK, m1 & (y_host == c)) for c in range(3)},
            "to_pick_all": _group(units, pred == PICK, m1),
            "matrix": confusion(y_host[m1], pred[m1]),
            "to_pick_by_distractor": {
                str(d): _ratio(int(((pred == PICK) & m1 & (sub == d)).sum()), int((m1 & (sub == d)).sum()))
                for d in np.unique(np.asarray(sub)[m1])},
        }
    for fam, key, same in (("C1p", "retain", True), ("C2g", "changed", False)):
        m = valid & (variant == fam) & (base >= 0)
        if not m.any():
            continue
        hit = (pred == base) if same else (pred != base)
        res[fam] = {key: {CLASSES[c]: _group(units, hit, m & (y_host == c)) for c in range(3)},
                    key + "_all": _group(units, hit, m),
                    "matrix": confusion(y_host[m], pred[m]),
                    "n_without_c0": int((valid & (variant == fam) & (base < 0)).sum())}
    for fam in ("C2s", "C2i"):
        m = valid & (variant == fam)
        if not m.any():
            continue
        fs, fi = pred == y_state, pred == y_image
        res[fam] = {"follow": {"state": _group(units, fs, m), "image": _group(units, fi, m),
                               "other": _group(units, ~fs & ~fi, m)},
                    "by_image_state_class": {
                        f"img_{CLASSES[ci]}__state_{CLASSES[cs]}": {
                            "n": int((mm := m & (y_image == ci) & (y_state == cs)).sum()),
                            "state": _ratio(int((fs & mm).sum()), int(mm.sum())),
                            "image": _ratio(int((fi & mm).sum()), int(mm.sum())),
                            "other": _ratio(int((~fs & ~fi & mm).sum()), int(mm.sum()))}
                        for ci in range(3) for cs in range(3) if ci != cs}}
    if pair_key is not None:
        res["C2_pairs"] = _pair_joint(np.asarray(pair_key).astype(str), variant, pred, y_state, y_image, valid)
    return res


def _outcome(p: int, ys: int, yi: int) -> str:
    return "state" if p == ys else ("image" if p == yi else "other")


def _pair_joint(pair_key, variant, pred, y_state, y_image, valid) -> dict:
    """Pairs whose C2s and C2i rows are both valid: 3x3 counts of (C2s outcome, C2i outcome)."""
    s = {k: i for i, k in enumerate(pair_key) if variant[i] == "C2s" and valid[i]}
    i_ = {k: i for i, k in enumerate(pair_key) if variant[i] == "C2i" and valid[i]}
    names = ("state", "image", "other")
    joint = {f"C2s_{a}__C2i_{b}": 0 for a in names for b in names}
    for k in sorted(set(s) & set(i_)):
        a = _outcome(int(pred[s[k]]), int(y_state[s[k]]), int(y_image[s[k]]))
        b = _outcome(int(pred[i_[k]]), int(y_state[i_[k]]), int(y_image[i_[k]]))
        joint[f"C2s_{a}__C2i_{b}"] += 1
    return {"n_pairs": int(len(set(s) & set(i_))), "joint": joint}
