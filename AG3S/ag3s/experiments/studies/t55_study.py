"""T55 — strong rule vs ``HELD ∧ feature`` against the physical predicates: dev CV -> prereg -> test (CPU).

`AG3S/ag3s/docs/handoff/T55.task.md` 사전 고정 1–5, 구현 4, 추가 사전 고정 1–3 (user decision 2026-10-13). Inputs
are this run's re-simulations (`t55_resim`, ``verify/T55/resim/<set>/``), features (`t55_features`,
``verify/T55/feat/<set>/``) and the re-grounded strong rule (`t55_perception`,
``verify/T55/perception/<set>/<perception>/``).

Strong-rule latch modes (``LATCH_MODES``)
-----------------------------------------
``reground`` (primary, dev and test)  the serving AG3S L8H2 grounding re-run on each re-sim's own image and depth
                                      (`t55_perception`), one trace per (re-sim, prompt): the instruction-changed
                                      prompts have their own latch (their own attention).
``record`` (test, sensitivity)        the server's recorded perception (stale after an intervention; the
                                      instruction traces reuse E0's).
``oracle`` (auxiliary diagnostic)     MuJoCo reach (separates perception errors; instruction reuses E0's).

Baselines (`t55_resim.BASELINE_OF_SET`): dev = the unintervened re-sim (its difference to the recording is
reported, not required to be 0); test = the record (interventions need an exact E0). Slip: records whose
unintervened re-sim never reaches a hold have no slip trace; they are listed (``slip_excluded``) with the reason.

Traces (one per re-sim run and prompt)
--------------------------------------
=============  =====================================  ==================================================
condition      frames (``window``)                    answer
=============  =====================================  ==================================================
natural        ``E0`` with its own prompt, all frames  physics of the re-sim (= the record when E0 exact)
instruction    ``E0`` images/state, prompt noun        physics with ``target`` = the new noun: P_task changes,
               swapped (``instr_<fruit>``), all        P_grasp does not. ``reground``: the strong rule re-run
               frames                                  with the changed prompt's attention (own trace);
                                                       ``record`` / ``oracle``: E0's trace
empty / slip   ``empty_*`` / ``slip_*`` re-sims,       physics of the intervened re-sim
               frames >= ``window_start``
val_c0/val_c1  T48a val frames (dev only, no physics,  phase label: place = P_task (orig) / never (instr),
               no strong rule)                         place = P_grasp; ``truth_kind = phase_label``
=============  =====================================  ==================================================

``group`` = ``<set>:<key>`` — the episode block: the original and every variant (instruction, empty, slip)
of an episode share it, so they are in the same CV fold and the same bootstrap block.

Truth (`t55_predicate.predicates`): raw P_grasp / P_task from `t48b_scene.classify` of every fruit on the pose
the camera saw, then the causal persistence ``K`` (same K for onset and release, every source / condition).
Val traces are not persisted (stride-6 dataset frames; phase labels are already segments).

Arms (`t55_predicate.combine`; the latch trace is never altered)::

    strong   task_ready = HELD                       (grasp_latch phase ``held``)
    hsr      task_ready = HELD ∧ persist_p(f_hsr)    f_hsr  = kv_L4_v1 place   (argmax, or p(place) >= tau)
    noun     task_ready = HELD ∧ persist_p(f_noun)   f_noun = noun-token probe p >= tau (frames with noun_ok)

Dev stage (fixes everything, writes ``T55.prereg.json``; nothing from the test set is read)
------------------------------------------------------------------------------------------
1. ``K`` — candidates ``--k-candidates`` (1 2 3). Rule (momentary contacts must not count as holds): per dev
   E0 trace, excess = max(0, #onsets of persistent P_grasp − #strong-rule closing attempts); K = the
   smallest candidate with the least total excess. ``--K`` overrides (recorded as an override).
2. HSR — grid tau x p (``--hsr-tau``, ``--feature-persist``) on the dev re-sim traces.
3. noun probe — feature (``noun_v`` layer l of the target token, or ``noun_hidden_final``) x C, binary L2
   logistic regression after standardisation (`t50_core.fit_logreg`, K = 2) on persistent P_task, episode
   GroupKFold (`t50_core.group_kfold`, groups = episode blocks); the pair with the least out-of-fold
   log-loss wins; then tau x p on the out-of-fold probabilities; the final probe is refit on all dev rows
   (``noun_probe.npz``, md5 in the prereg). Training rows: natural / instruction / val frames, and the window
   frames of empty / slip (their earlier frames duplicate E0).
4. Combination selection (both arms): among the grid points within the tolerance on dev (pooled window
   frames of every dev condition, P_task): Δ false-reject rate <= ``--tol-false-reject``, Δ mean paired
   latency <= ``--tol-latency``, Δ undetected onsets <= ``--tol-undetected`` (**user-fixed** 2026-10-13:
   +5 %p, +1 frame, 0 events — `USER_TOL`; dev spread is not used as a tolerance, the dev episode-block
   bootstrap of the chosen point is reported as achievability / uncertainty only) — the one with the fewest false
   passes (ties: smaller Δ false reject, smaller p, then the grid order). No grid point within tolerance ->
   ``feasible: false`` and the point with the smallest total violation, then fewest false passes.
   Intervention tags: ``--interventions`` (the dev choice of timing / method, default every tag in resim/dev);
   the test stage scores only these tags (others are listed as ignored; same tag with other parameters is
   refused).
5. ``T55.prereg.json`` (refuses to overwrite): K, persistence mode, latch modes, the two arms, tolerances,
   intervention tags, bootstrap spec, dev tables, input md5s, git HEAD. STOP — the lead asks the user.

Test stage (reads the prereg only; its md5 is checked before and after)
-----------------------------------------------------------------------
Per condition x arm x predicate (P_task, P_grasp): false pass / false reject (counts, rates), onset latency
(paired with the strong rule per onset event; undetected = censored count), 4-state breakdown of false passes;
paired differences (arm − strong) with CI: episode block bootstrap (B, seed, percentile 95 %, groups
resampled with replacement), moving-block bootstrap (5 frames within each trace) as sensitivity, exact
McNemar on discordant frames as a frame-level reference only. The strong rule against P_grasp is its own
table. Natural events of each T53 episode (`t55_predicate.natural_events`) are reported apart from the
artificial conditions.

Usage::

    source scripts/setup/env.sh; export AG3S_RUN_DIR=<the T55 run>; M=AG3S.ag3s.experiments.studies.t55_study
    OMP_NUM_THREADS=4 .venv-ag3s/bin/python -m $M --stage dev [--perception t55-legacy]   # -> T55.prereg.json, STOP
    OMP_NUM_THREADS=4 .venv-ag3s/bin/python -m $M --stage test --prereg $AG3S_RUN_DIR/verify/T55/T55.prereg.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t55_predicate as pr

STEP = "T55"
STAGES = ("dev", "test")
PREREG_NAME = "T55.prereg.json"
PREREG_VERSION = "t55-prereg-v2"
CONDITIONS = ("natural", "instruction", "empty", "slip")
VAL_CONDITIONS = ("val_c0", "val_c1")
ARMS = ("strong", "hsr", "noun")
PREDICATES = ("p_task", "p_grasp")
BOOT_B = 10000
BOOT_SEED = 5507
BOOT_LEVEL = 0.95
MOVING_BLOCK = 5
CLOSED_BELOW = 0.85          # LatchConfig.gripper_closed_below
#: Tolerated worsening vs the strong rule alone — **user-fixed** before measurement (T55.task.md 추가 사전 고정 3,
#: 2026-10-13): false reject +5 %p, mean detection latency +1 frame, additional undetected hold onsets 0.
USER_TOL = {"false_reject_rate": 0.05, "latency_frames": 1.0, "undetected_events": 0}
USER_TOL_SOURCE = ("user decision 2026-10-13 (T55.task.md 추가 사전 고정 3): fixed before measurement, "
                   "not derived from dev")
DEFAULT_TOL = dict(USER_TOL)
LATCH_MODES = ("reground", "record", "oracle")
PRIMARY_MODE = "reground"
DEFAULT_PERCEPTION = "t55-legacy"


# ------------------------------------------------------------------------------ traces


@dataclasses.dataclass
class Trace:
    set_name: str
    key: str
    tag: str
    prompt_tag: str
    condition: str
    group: str
    truth_kind: str                      # "physics" | "phase_label"
    used_arm: str
    noun: str
    window: np.ndarray                   # (F,) bool
    held_by: Optional[dict] = None       # fruit -> [hand | None] (physics)
    in_crate: Optional[np.ndarray] = None  # (F,) bool, the noun's fruit
    phase_truth: Optional[dict] = None   # val: {"p_grasp", "p_task", "placed"}
    latch: dict = dataclasses.field(default_factory=dict)   # mode -> {"phase","attach","detach"}
    gripper_used: Optional[np.ndarray] = None
    feat_path: Optional[str] = None
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.window))

    @property
    def tid(self) -> str:
        return f"{self.set_name}:{self.key}__{self.tag}__{self.prompt_tag}"


def condition_of(tag: str, prompt_tag: str) -> str:
    if tag == "val":
        return "val_c0" if prompt_tag == "orig" else "val_c1"
    if tag == "E0":
        return "natural" if prompt_tag == "orig" else "instruction"
    for c in ("empty", "slip"):
        if tag.startswith(c):
            return c
    raise ValueError(f"unknown tag {tag!r}")


def truth(trace: Trace, K: int, mode: str = "causal") -> dict:
    """Persistent P_grasp / P_task / placed / state4 of a trace (val: phase labels, no persistence)."""
    if trace.truth_kind == "phase_label":
        t = trace.phase_truth
        pg, pt, pl = (np.asarray(t[k], bool) for k in ("p_grasp", "p_task", "placed"))
        return {"p_grasp": pg, "p_task": pt & pg, "placed": pl, "p_grasp_raw": pg, "p_task_raw": pt,
                "state4": pr.four_state(pg, pt & pg, pl), "K": 1}
    return pr.predicates(trace.held_by, trace.noun, trace.used_arm, in_crate=trace.in_crate, K=K, mode=mode)


def _held_by(z) -> dict:
    fruits = [str(f) for f in z["fruits"]]
    st, hd = z["phys_state"], z["phys_hand"]
    return {f: [(str(hd[i, j]) or None) if str(st[i, j]) == "held" else None for i in range(st.shape[0])]
            for j, f in enumerate(fruits)}


def load_perception_latch(path: pathlib.Path) -> Optional[dict]:
    """``{"phase","attach","detach"}`` of a `t55_perception` output (the ``reground`` latch), or None."""
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[f"latch_reground__{k}"]) for k in ("phase", "attach", "detach")}


def load_resim_traces(resim_dir: pathlib.Path, feat_dir: Optional[pathlib.Path], set_name: str, *,
                      instruction: bool = True, meta_ds: Optional[dict] = None,
                      perception_dir: Optional[pathlib.Path] = None) -> list[Trace]:
    """Every re-sim npz of a set; for E0 also its instruction prompts (`t55_features.instruction_prompts`).
    ``reground`` latch per (re-sim, prompt) from `perception_dir` (absent file -> no ``reground`` mode)."""
    from AG3S.ag3s.experiments.studies import t55_features as tf

    out = []
    for p in sorted(pathlib.Path(resim_dir).glob("*__*.npz")):
        if p.name.endswith(".partial.npz"):
            continue
        key, tag = p.stem.split("__", 1)
        with np.load(p, allow_pickle=False) as z:
            fruits = [str(f) for f in z["fruits"]]
            hb = _held_by(z)
            ic = np.asarray(z["in_crate"], bool)
            used = str(z["used_arm"])
            target = str(z["target"])
            prompt = str(z["prompt"])
            ep = int(z["episode"])
            ws = int(z["window_start"])
            st = np.asarray(z["state"], np.float64)
            latch = {}
            for mode in ("record", "oracle"):
                if f"latch_{mode}__phase" in z.files:
                    latch[mode] = {k: np.asarray(z[f"latch_{mode}__{k}"]) for k in ("phase", "attach", "detach")}
            meta = json.loads(str(z["meta"]))
        F = len(st)
        gu = st[:, 7 if used == "left" else 15]
        window = np.ones(F, bool) if ws < 0 else (np.arange(F) >= ws)
        prompts = [("orig", prompt, target)]
        if instruction and tag == "E0":
            if meta_ds is None:
                from AG3S.ag3s.experiments.studies import t53_records as rc

                meta_ds = rc.load_dataset_meta()
            prompts = tf.instruction_prompts(prompt, target, meta_ds[ep]["non_target_fruits"])
        for ptag, _prompt, noun in prompts:
            fp = None if feat_dir is None else pathlib.Path(feat_dir) / f"{key}__{tag}__{ptag}.npz"
            lt = dict(latch)
            pp = None
            if perception_dir is not None:
                pp = pathlib.Path(perception_dir) / f"{key}__{tag}__{ptag}.npz"
                rg = load_perception_latch(pp)
                if rg is not None:
                    if len(rg["phase"]) != F:
                        raise SystemExit(f"{pp}: {len(rg['phase'])} requests, re-sim {F} frames")
                    lt["reground"] = rg
            out.append(Trace(set_name=set_name, key=key, tag=tag, prompt_tag=ptag, condition=condition_of(tag, ptag),
                             group=f"{set_name}:{key}", truth_kind="physics", used_arm=used, noun=noun,
                             window=window.copy(), held_by=hb, in_crate=ic[:, fruits.index(noun)], latch=lt,
                             gripper_used=gu, feat_path=(str(fp) if fp is not None and fp.is_file() else None),
                             meta={"intervention": meta.get("intervention"), "info": meta.get("info"),
                                   "e0_exact": (meta.get("parity") or {}).get("exact", meta.get("e0_exact")),
                                   "baseline": meta.get("baseline"), "resim": str(p),
                                   "parity": {k: v for k, v in (meta.get("parity") or {}).items()
                                              if k not in ("per_frame", "latch_record_states")} or None,
                                   "perception": (str(pp) if pp is not None and pp.is_file() else None),
                                   "episode": ep, "window_start": ws}))
    return out


def load_val_traces(val_dir: pathlib.Path) -> list[Trace]:
    from AG3S.ag3s.experiments.studies.t48_hsr_core import PLACE

    out = []
    for p in sorted(pathlib.Path(val_dir).glob("ep_*__*.npz")):
        key, ptag = p.stem.split("__", 1)
        with np.load(p, allow_pickle=False) as z:
            y = np.asarray(z["phase_class"], np.int64)
            noun = str(z["noun"])
            used = str(z["used_arm"])
        place = y == PLACE
        out.append(Trace(set_name="dev", key=key, tag="val", prompt_tag=ptag, condition=condition_of("val", ptag),
                         group=f"val:{key}", truth_kind="phase_label", used_arm=used, noun=noun,
                         window=np.ones(len(y), bool),
                         phase_truth={"p_grasp": place, "p_task": place & (ptag == "orig"),
                                      "placed": np.zeros(len(y), bool)},
                         feat_path=str(p)))
    return out


# ------------------------------------------------------------------------------ features


class FeatureCache:
    """Per trace: kv_l4 (F, 512), the target-token noun feature by name, noun_ok."""

    def __init__(self):
        self._d: dict[str, dict] = {}

    def get(self, tr: Trace) -> dict:
        if tr.tid not in self._d:
            if tr.feat_path is None:
                raise FileNotFoundError(f"no feature file for {tr.tid}")
            with np.load(tr.feat_path, allow_pickle=False) as z:
                self._d[tr.tid] = {"kv_l4": np.asarray(z["kv_l4"], np.float32),
                                   "noun_v": np.asarray(z["noun_v"][:, 0], np.float32),        # (F, nL, 256)
                                   "noun_hidden_final": np.asarray(z["noun_hidden_final"][:, 0], np.float32),
                                   "noun_layers": [int(x) for x in z["noun_layers"]],
                                   "noun_ok": np.asarray(z["noun_ok"], bool)}
            if len(self._d[tr.tid]["kv_l4"]) != tr.n:
                raise ValueError(f"{tr.tid}: {len(self._d[tr.tid]['kv_l4'])} feature frames, trace {tr.n}")
        return self._d[tr.tid]

    def put(self, tr: Trace, feats: dict) -> None:
        self._d[tr.tid] = feats


def noun_matrix(feats: dict, name: str) -> np.ndarray:
    """``hidden`` -> noun_hidden_final; ``v<l>`` -> noun_v at layer l."""
    if name == "hidden":
        return feats["noun_hidden_final"]
    layer = int(name[1:])
    return feats["noun_v"][:, feats["noun_layers"].index(layer)]


def hsr_place_proba(kv: np.ndarray, probe=None) -> tuple[np.ndarray, np.ndarray]:
    """(p(place), argmax == place) of kv_L4_v1."""
    if probe is None:
        from AG3S.ag3s.stages.subtask_probe import SubtaskProbe

        probe = SubtaskProbe.load()
    p = probe.proba(np.asarray(kv, np.float64))
    j = list(probe.classes).index("place")
    return p[:, j], p.argmax(1) == j


def flag_from(p: np.ndarray, tau, argmax: Optional[np.ndarray] = None, ok: Optional[np.ndarray] = None) -> np.ndarray:
    f = (np.asarray(argmax, bool) if tau == "argmax" else np.asarray(p, np.float64) >= float(tau))
    return f & (np.ones(len(f), bool) if ok is None else np.asarray(ok, bool))


# ------------------------------------------------------------------------------ metrics


def held_of(tr: Trace, mode: str) -> Optional[np.ndarray]:
    lt = tr.latch.get(mode)
    return None if lt is None else pr.held_from_phase(lt["phase"])


def ready_of(tr: Trace, mode: str, flag: Optional[np.ndarray], p: int) -> np.ndarray:
    lt = tr.latch[mode]
    trace = pr.LatchTrace.from_arrays(lt["phase"], lt["attach"], lt["detach"])
    return pr.combine(trace, flag, p)["task_ready"]


def trace_metrics(ready: np.ndarray, truth_: np.ndarray, window: np.ndarray, state4: Optional[np.ndarray] = None) -> dict:
    e = pr.errors(ready, truth_, window)
    ev = pr.latencies(ready, truth_, window)
    e["events"] = ev
    if state4 is not None:
        fp = np.asarray(ready, bool) & ~np.asarray(truth_, bool) & np.asarray(window, bool)
        e["false_pass_by_state"] = {s: int((fp & (np.asarray(state4) == s)).sum()) for s in pr.STATES4}
    return e


def paired_counts(m_s: dict, m_a: dict) -> dict:
    """Additive per-trace counts for an (arm, strong) pair."""
    d = pr.paired_latency(m_s["events"], m_a["events"])
    lat = [x for x in d if x is not None]
    und_s = sum(1 for e in m_s["events"] if e["latency"] is None)
    und_a = sum(1 for e in m_a["events"] if e["latency"] is None)
    return {"neg": m_s["n_neg"], "pos": m_s["n_pos"], "fp_s": m_s["false_pass"], "fp_a": m_a["false_pass"],
            "fr_s": m_s["false_reject"], "fr_a": m_a["false_reject"], "lat_sum": float(sum(lat)),
            "lat_n": len(lat), "und_s": und_s, "und_a": und_a, "events": len(m_s["events"])}


COUNT_KEYS = ("neg", "pos", "fp_s", "fp_a", "fr_s", "fr_a", "lat_sum", "lat_n", "und_s", "und_a", "events")


def _div(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def pair_stats(c: Mapping[str, Any]) -> dict:
    """Rates and differences from summed counts (works on arrays: one value per bootstrap draw)."""
    return {"fp_rate_strong": _div(c["fp_s"], c["neg"]), "fp_rate_arm": _div(c["fp_a"], c["neg"]),
            "fr_rate_strong": _div(c["fr_s"], c["pos"]), "fr_rate_arm": _div(c["fr_a"], c["pos"]),
            "d_fp_rate": _div(np.asarray(c["fp_a"], float) - c["fp_s"], c["neg"]),
            "d_fr_rate": _div(np.asarray(c["fr_a"], float) - c["fr_s"], c["pos"]),
            "d_latency_mean": _div(c["lat_sum"], c["lat_n"]),
            "d_fp_count": np.asarray(c["fp_a"], float) - np.asarray(c["fp_s"], float),
            "d_fr_count": np.asarray(c["fr_a"], float) - np.asarray(c["fr_s"], float),
            "d_undetected": np.asarray(c["und_a"], float) - np.asarray(c["und_s"], float)}


def sum_counts(rows: Sequence[Mapping]) -> dict:
    return {k: float(sum(r[k] for r in rows)) for k in COUNT_KEYS}


def _finite(x) -> Optional[float]:
    x = float(x)
    return x if np.isfinite(x) else None


def episode_bootstrap(per_group: Mapping[str, Mapping], *, B: int = BOOT_B, seed: int = BOOT_SEED,
                      level: float = BOOT_LEVEL) -> dict:
    """Episode block bootstrap of `pair_stats`: groups resampled with replacement, counts summed."""
    groups = sorted(per_group)
    if not groups:
        return {}
    M = np.asarray([[per_group[g][k] for k in COUNT_KEYS] for g in groups], np.float64)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(groups), size=(int(B), len(groups)))
    S = M[idx].sum(1)
    draws = pair_stats({k: S[:, i] for i, k in enumerate(COUNT_KEYS)})
    point = pair_stats({k: M[:, i].sum() for i, k in enumerate(COUNT_KEYS)})
    a = (1.0 - level) / 2.0
    out = {}
    for k, v in draws.items():
        v = np.asarray(v, np.float64)
        fin = v[np.isfinite(v)]
        out[k] = {"point": _finite(point[k]),
                  "lo": (float(np.quantile(fin, a)) if fin.size else None),
                  "hi": (float(np.quantile(fin, 1 - a)) if fin.size else None),
                  "n_finite": int(fin.size)}
    out["_spec"] = {"B": int(B), "seed": int(seed), "level": level, "n_groups": len(groups), "unit": "episode block"}
    return out


def moving_block_bootstrap(frame_rows: Sequence[np.ndarray], *, block: int = MOVING_BLOCK, B: int = BOOT_B,
                           seed: int = BOOT_SEED, level: float = BOOT_LEVEL) -> dict:
    """Frame-level sensitivity: each trace's (n, 6) indicator matrix [neg, pos, fp_s, fp_a, fr_s, fr_a] is
    rebuilt from moving blocks of `block` frames (starts uniform on [0, n − block]), truncated to n."""
    rng = np.random.default_rng(int(seed))
    total = np.zeros((int(B), 6))
    for X in frame_rows:
        X = np.asarray(X, np.float64)
        n = len(X)
        if n == 0:
            continue
        b = min(int(block), n)
        nb = -(-n // b)
        starts = rng.integers(0, n - b + 1, size=(int(B), nb))
        ix = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(int(B), -1)[:, :n]
        total += X[ix].sum(1)
    c = dict(zip(("neg", "pos", "fp_s", "fp_a", "fr_s", "fr_a"), total.T))
    a = (1.0 - level) / 2.0
    out = {}
    for k, v in (("d_fp_rate", _div(c["fp_a"] - c["fp_s"], c["neg"])), ("d_fr_rate", _div(c["fr_a"] - c["fr_s"], c["pos"]))):
        fin = v[np.isfinite(v)]
        out[k] = {"lo": float(np.quantile(fin, a)) if fin.size else None,
                  "hi": float(np.quantile(fin, 1 - a)) if fin.size else None}
    out["_spec"] = {"B": int(B), "seed": int(seed), "block": int(block), "unit": "moving block of frames within a trace"}
    return out


def mcnemar(b: int, c: int) -> dict:
    """Exact two-sided McNemar on discordant counts (frame level, reference only)."""
    from scipy.stats import binomtest

    n = int(b) + int(c)
    return {"b_arm_only": int(b), "c_strong_only": int(c),
            "p_exact": (float(binomtest(int(b), n, 0.5).pvalue) if n else None)}


# ------------------------------------------------------------------------------ evaluation of one config


def evaluate(traces: Sequence[Trace], truths: Mapping[str, dict], *, mode: str,
             flags: Optional[Mapping[str, np.ndarray]], p: int, predicate: str = "p_task") -> dict:
    """Pooled and per-condition paired counts of an arm (flags None = strong alone vs itself)."""
    rows = {}
    for tr in traces:
        if tr.latch.get(mode) is None:
            continue
        t = truths[tr.tid]
        r_s = ready_of(tr, mode, None, 1)
        r_a = r_s if flags is None else ready_of(tr, mode, flags[tr.tid], p)
        m_s = trace_metrics(r_s, t[predicate], tr.window)
        m_a = trace_metrics(r_a, t[predicate], tr.window)
        rows[tr.tid] = (tr, paired_counts(m_s, m_a))
    out = {"pooled": pair_stats(sum_counts([c for _, c in rows.values()])),
           "counts": sum_counts([c for _, c in rows.values()]), "by_condition": {}}
    for cond in CONDITIONS:
        sel = [c for tr, c in rows.values() if tr.condition == cond]
        if sel:
            out["by_condition"][cond] = {"counts": sum_counts(sel), "stats": pair_stats(sum_counts(sel))}
    return out


def _scalar_stats(st: Mapping) -> dict:
    return {k: _finite(v) for k, v in st.items()}


def select_config(cands: Sequence[dict], tol: Mapping[str, float]) -> dict:
    """`cands`: ``{"config": {...}, "counts": summed counts}`` in grid order. Module docstring step 4."""
    scored = []
    for i, c in enumerate(cands):
        s = _scalar_stats(pair_stats(c["counts"]))
        dfr = s["d_fr_rate"] or 0.0
        dlat = s["d_latency_mean"] or 0.0
        dund = s["d_undetected"] or 0.0
        viol = (max(0.0, dfr - tol["false_reject_rate"]) + max(0.0, dlat - tol["latency_frames"]) / 10.0
                + max(0.0, dund - tol["undetected_events"]) / 10.0)
        scored.append({"i": i, "config": c["config"], "stats": s, "violation": viol,
                       "feasible": viol == 0.0, "fp": c["counts"]["fp_a"]})
    feas = [s for s in scored if s["feasible"]]
    pool = feas if feas else scored
    key = ((lambda s: (s["fp"], s["stats"]["d_fr_rate"] or 0.0, s["config"].get("p", 1), s["i"])) if feas else
           (lambda s: (s["violation"], s["fp"], s["config"].get("p", 1), s["i"])))
    best = min(pool, key=key)
    return {"chosen": best["config"], "feasible": bool(feas), "chosen_stats": best["stats"],
            "grid": [{"config": s["config"], "stats": s["stats"], "violation": s["violation"]} for s in scored]}


# ------------------------------------------------------------------------------ K


def select_K(traces: Sequence[Trace], candidates: Sequence[int], *, mode: str, persist_mode: str = "causal") -> dict:
    rows = []
    for K in candidates:
        excess, per = 0, {}
        for tr in traces:
            if tr.condition != "natural" or tr.truth_kind != "physics" or tr.latch.get(mode) is None:
                continue
            t = truth(tr, K, persist_mode)
            ne = pr.natural_events(t, np.asarray(tr.gripper_used) < CLOSED_BELOW, tr.latch[mode]["phase"])
            e = max(0, ne["grasp_onsets"] - ne["closing_attempts"])
            excess += e
            per[tr.key] = {"grasp_onsets": ne["grasp_onsets"], "closing_attempts": ne["closing_attempts"],
                           "excess": e, "p_grasp_frames": int(t["p_grasp"].sum()), "p_task_frames": int(t["p_task"].sum())}
        rows.append({"K": int(K), "excess": int(excess), "per_trace": per})
    best = min(rows, key=lambda r: (r["excess"], r["K"]))
    return {"chosen": best["K"], "rule": "smallest K with the least total excess = max(0, #P_grasp onsets − "
                                          "#strong-rule closing attempts) over dev E0 traces", "table": rows}


# ------------------------------------------------------------------------------ noun probe


def fit_binary(X: np.ndarray, y: np.ndarray, C: float):
    from AG3S.ag3s.experiments.studies.t50_core import fit_logreg

    return fit_logreg(X, np.asarray(y, np.int64), C, n_classes=2)


def oof_proba(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float, folds) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for tr, te in folds:
        m = fit_binary(X[tr], y[tr], C)
        out[te] = m.proba(X[te])[:, 1]
    return out


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, np.float64), 1e-12, 1 - 1e-12)
    y = np.asarray(y, bool)
    return float(-np.mean(np.where(y, np.log(p), np.log(1 - p))))


def training_mask(tr: Trace) -> np.ndarray:
    return tr.window.copy() if tr.condition in ("empty", "slip") else np.ones(tr.n, bool)


def noun_cv(traces: Sequence[Trace], truths: Mapping[str, dict], cache: FeatureCache, *, feature_names: Sequence[str],
            C_grid: Sequence[float], n_splits: int = 5) -> dict:
    """Feature x C by out-of-fold log-loss; returns the per-trace OOF probabilities of the winner (all frames)."""
    from AG3S.ag3s.experiments.studies.t50_core import group_kfold

    groups_all = sorted({tr.group for tr in traces})
    k = min(int(n_splits), len(groups_all))
    if k < 2:
        raise SystemExit(f"noun probe CV needs >= 2 episode groups, have {groups_all}")
    # group -> fold (group_kfold on one row per frame of the training rows)
    tr_rows = [(tr, np.flatnonzero(training_mask(tr) & cache.get(tr)["noun_ok"])) for tr in traces]
    g_rows = np.concatenate([np.full(len(ix), tr.group, dtype=object) for tr, ix in tr_rows])
    y = np.concatenate([truths[tr.tid]["p_task"][ix] for tr, ix in tr_rows]).astype(np.int64)
    folds = group_kfold(g_rows.astype(str), k)
    fold_of_group = {}
    for f, (_, te) in enumerate(folds):
        for g in set(g_rows[te]):
            if str(g) in fold_of_group:
                raise RuntimeError(f"episode block {g} split across folds")
            fold_of_group[str(g)] = f
    table, best = [], None
    for name in feature_names:
        X = np.concatenate([noun_matrix(cache.get(tr), name)[ix] for tr, ix in tr_rows]).astype(np.float64)
        for C in C_grid:
            p = oof_proba(X, y, g_rows, float(C), folds)
            ll = log_loss(p, y)
            table.append({"feature": name, "C": float(C), "oof_log_loss": ll,
                          "oof_acc_at_0.5": float(((p >= 0.5) == y.astype(bool)).mean())})
            if best is None or ll < best["oof_log_loss"] - 1e-12:
                best = table[-1]
    # winner: fold models -> probabilities on every frame of the held-out groups' traces
    name, C = best["feature"], best["C"]
    X = np.concatenate([noun_matrix(cache.get(tr), name)[ix] for tr, ix in tr_rows]).astype(np.float64)
    models = [fit_binary(X[tr_i], y[tr_i], C) for tr_i, _ in folds]
    oof = {}
    for tr in traces:
        # a group without training rows (every frame noun_ok False) was never trained on: any fold model is
        # out-of-sample for it — fold 0
        f = fold_of_group.setdefault(tr.group, 0)
        Xa = noun_matrix(cache.get(tr), name).astype(np.float64)
        oof[tr.tid] = models[f].proba(Xa)[:, 1]
    return {"chosen": {"feature": name, "C": C}, "table": table, "n_rows": int(len(y)), "n_pos": int(y.sum()),
            "n_groups": len(groups_all), "n_splits": k, "oof": oof,
            "fold_of_group": fold_of_group, "X_y": (X, y)}


def save_probe(path: pathlib.Path, model, *, feature: str, C: float) -> str:
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    rs.save_npz_atomic(pathlib.Path(path), mean=model.mean, scale=model.scale, coef=model.coef,
                       intercept=model.intercept, C=np.asarray(C), feature=np.asarray(feature),
                       n_iter=np.asarray(model.n_iter), converged=np.asarray(model.converged))
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def load_probe(path: pathlib.Path):
    from AG3S.ag3s.experiments.studies.t50_core import LogReg

    with np.load(path, allow_pickle=False) as z:
        return LogReg(mean=z["mean"], scale=z["scale"], coef=z["coef"], intercept=z["intercept"], C=float(z["C"]),
                      n_iter=int(z["n_iter"]), converged=bool(z["converged"])), str(z["feature"])


# ------------------------------------------------------------------------------ stages


def git_head() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=pathlib.Path(__file__).resolve().parents[4]).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def grid_flags(traces, values: Mapping[str, np.ndarray], tau, argmax: Optional[Mapping] = None,
               ok: Optional[Mapping] = None) -> dict:
    return {tr.tid: flag_from(values[tr.tid], tau, None if argmax is None else argmax[tr.tid],
                              None if ok is None else ok[tr.tid]) for tr in traces}


def combination_grid(traces, truths, *, mode: str, values, taus, persists, argmax=None, ok=None) -> list[dict]:
    cands = []
    for tau in taus:
        fl = grid_flags(traces, values, tau, argmax, ok)
        for p in persists:
            ev = evaluate(traces, truths, mode=mode, flags=fl, p=int(p))
            cands.append({"config": {"tau": tau, "p": int(p)}, "counts": ev["counts"],
                          "by_condition": {c: _scalar_stats(v["stats"]) for c, v in ev["by_condition"].items()}})
    return cands


def _canon(v) -> str:
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    return json.dumps(rs.jsonable(v), sort_keys=True)


def intervention_tags(traces: Sequence[Trace]) -> dict:
    """tag -> intervention parameters; refuses one tag with two parameter sets."""
    out = {}
    for tr in traces:
        if tr.condition in ("empty", "slip") and tr.meta.get("intervention"):
            v = tr.meta["intervention"]
            if tr.tag in out and _canon(out[tr.tag]) != _canon(v):
                raise SystemExit(f"intervention tag {tr.tag} has two parameter sets ({tr.tid})")
            out[tr.tag] = v
    return out


def parse_tau(xs: Sequence[str]) -> list:
    return [x if x == "argmax" else float(x) for x in xs]


def slip_exclusions(resim_dir: pathlib.Path) -> dict:
    """``{key: {slip tag: reason}}`` — records whose unintervened re-sim never reaches a hold (`t55_resim`)."""
    out: dict = {}
    for p in sorted(pathlib.Path(resim_dir).glob("*__slip_*.json")):
        m = json.loads(p.read_text())
        if m.get("excluded_from_slip_evaluation"):
            key, tag = p.stem.split("__", 1)
            out.setdefault(key, {})[tag] = m.get("not_applicable")
    return out


def check_baselines(traces: Sequence[Trace], set_name: str) -> None:
    """dev: every trace is the unintervened-re-sim baseline; test: the record, interventions need an exact E0."""
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    want = rs.BASELINE_OF_SET[set_name]
    bad = sorted({tr.tid for tr in traces if tr.meta.get("baseline") != want})
    if bad:
        raise SystemExit(f"{set_name} re-sims whose baseline is not {want!r}: {bad[:5]}")
    if set_name == "test":
        bad = sorted({tr.tid for tr in traces if tr.condition in ("empty", "slip") and not tr.meta.get("e0_exact")})
        if bad:
            raise SystemExit(f"test interventions without an exact E0 (STOP and report): {bad[:5]}")


def require_mode(traces: Sequence[Trace], mode: str) -> None:
    missing = [tr.tid for tr in traces if tr.latch.get(mode) is None]
    if missing:
        raise SystemExit(f"{len(missing)} traces without the {mode!r} latch (t55_perception for reground): {missing[:5]}")


def tolerance_record(args) -> dict:
    tol = {"false_reject_rate": float(args.tol_false_reject), "latency_frames": float(args.tol_latency),
           "undetected_events": float(args.tol_undetected)}
    over = sorted(k for k in tol if tol[k] != float(USER_TOL[k]))
    return {**tol, "user_fixed": not over, "source": USER_TOL_SOURCE if not over else
            f"OVERRIDDEN on the command line ({over}) — not the user-fixed values {USER_TOL}",
            "user_values": dict(USER_TOL), "overridden": over,
            "units": {"false_reject_rate": "rate difference (arm − strong), 0.05 = +5 %p",
                      "latency_frames": "mean paired onset latency difference, policy frames",
                      "undetected_events": "additional undetected hold-onset events"}}


def group_counts(traces: Sequence[Trace], truths: Mapping[str, dict], *, mode: str,
                 flags: Optional[Mapping[str, np.ndarray]], p: int, predicate: str = "p_task") -> dict:
    """Per episode block: the summed paired counts of an arm (for the dev bootstrap)."""
    out: dict[str, dict] = {}
    for tr in traces:
        if tr.latch.get(mode) is None:
            continue
        t = truths[tr.tid][predicate]
        m_s = trace_metrics(ready_of(tr, mode, None, 1), t, tr.window)
        m_a = trace_metrics(ready_of(tr, mode, None if flags is None else flags[tr.tid], p if flags is not None else 1),
                            t, tr.window)
        g = out.setdefault(tr.group, {k: 0.0 for k in COUNT_KEYS})
        for k, v in paired_counts(m_s, m_a).items():
            g[k] += v
    return out


def stage_dev(args) -> dict:
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    root = rs.out_root("")
    prereg_path = root / PREREG_NAME
    if prereg_path.exists() and not args.dry_run:
        raise SystemExit(f"{prereg_path} exists — the prereg is written once (move it away on the lead's word)")
    resim_dir, feat_dir = root / "resim" / "dev", root / "feat" / "dev"
    pdir = root / "perception" / "dev" / args.perception
    traces = load_resim_traces(resim_dir, feat_dir, "dev", perception_dir=pdir)
    if not traces:
        raise SystemExit(f"no dev re-sims under {resim_dir}")
    check_baselines(traces, "dev")
    if args.interventions is not None:
        have = {t.tag for t in traces if t.condition in ("empty", "slip")}
        unknown = sorted(set(args.interventions) - have)
        if unknown:
            raise SystemExit(f"--interventions {unknown} not among the dev re-sims {sorted(have)}")
        traces = [t for t in traces if t.condition not in ("empty", "slip") or t.tag in args.interventions]
    val = load_val_traces(feat_dir / "val") if (feat_dir / "val").is_dir() and not args.no_val else []
    mode = args.latch_mode_dev
    require_mode(traces, mode)
    kres = select_K(traces, args.k_candidates, mode=mode, persist_mode=args.persist_mode)
    K = int(args.K) if args.K is not None else kres["chosen"]
    all_tr = traces + val
    truths = {tr.tid: truth(tr, K, args.persist_mode) for tr in all_tr}
    strong = evaluate(traces, truths, mode=mode, flags=None, p=1)
    strong_pg = evaluate(traces, truths, mode=mode, flags=None, p=1, predicate="p_grasp")
    payload: dict = {"version": PREREG_VERSION, "step": STEP, "created": rs.now_kst(), "git_head": git_head(),
                     "argv": sys.argv, "inputs": {"resim_dir": str(resim_dir), "feat_dir": str(feat_dir),
                                                  "n_traces": len(traces), "n_val_traces": len(val),
                                                  "groups": sorted({t.group for t in all_tr}),
                                                  "resim_md5": {pathlib.Path(t.meta["resim"]).name: rs.md5_of(t.meta["resim"])
                                                                for t in traces if t.prompt_tag == "orig"}},
                     "truth": {"K": K, "K_override": args.K is not None, "K_selection": kres,
                               "persistence": args.persist_mode, "pose": "render pose (the pose the camera saw)"},
                     "latch_mode_dev": mode, "latch_mode_test": args.latch_mode_test,
                     "latch_mode_test_secondary": [m for m in LATCH_MODES if m != args.latch_mode_test],
                     "perception": {"name": args.perception, "dir": str(pdir),
                                    "md5": {pathlib.Path(t.meta["perception"]).name: rs.md5_of(t.meta["perception"])
                                            for t in traces if t.meta.get("perception")}},
                     "baseline": dict(rs.BASELINE_OF_SET),
                     "slip_excluded_dev": slip_exclusions(resim_dir),
                     "dev_parity_to_record": {t.key: t.meta.get("parity") for t in traces
                                              if t.tag == "E0" and t.prompt_tag == "orig"},
                     "interventions": intervention_tags(traces),
                     "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "level": BOOT_LEVEL, "unit": "episode block",
                                   "moving_block_frames": MOVING_BLOCK},
                     "tolerance": tolerance_record(args),
                     "strong_dev": {"p_task": {c: _scalar_stats(v["stats"]) for c, v in strong["by_condition"].items()},
                                    "p_grasp_counts": {c: v["counts"] for c, v in strong_pg["by_condition"].items()}},
                     "conditions_present": sorted({t.condition for t in all_tr}),
                     "e0_exact": {t.key: t.meta.get("e0_exact") for t in traces if t.tag == "E0" and t.prompt_tag == "orig"}}
    if args.no_features:
        payload["note"] = "--no-features: K and the strong rule only; no prereg written"
        rs.write_json(root / "dev_no_features.json", payload)
        print(f"[dev] K={K} (excess table {[(r['K'], r['excess']) for r in kres['table']]}); strong rule only")
        return payload
    missing = [t.tid for t in all_tr if t.feat_path is None]
    if missing:
        raise SystemExit(f"{len(missing)} traces without features (t55_features extract): {missing[:5]}")
    cache = FeatureCache()
    tol = {"false_reject_rate": args.tol_false_reject, "latency_frames": args.tol_latency,
           "undetected_events": args.tol_undetected}
    # HSR
    from AG3S.ag3s.stages.subtask_probe import SubtaskProbe, default_probe_path

    probe = SubtaskProbe.load()
    hp, ha = {}, {}
    for tr in traces:
        hp[tr.tid], ha[tr.tid] = hsr_place_proba(cache.get(tr)["kv_l4"], probe)
    hsr_c = combination_grid(traces, truths, mode=mode, values=hp, argmax=ha, taus=parse_tau(args.hsr_tau),
                             persists=args.feature_persist)
    hsr_sel = select_config(hsr_c, tol)
    # noun
    names = [f"v{l}" for l in args.noun_cv_layers] + (["hidden"] if not args.no_hidden else [])
    cv = noun_cv(all_tr, truths, cache, feature_names=names, C_grid=args.C_grid, n_splits=args.n_splits)
    ok = {tr.tid: cache.get(tr)["noun_ok"] for tr in traces}
    noun_c = combination_grid(traces, truths, mode=mode, values=cv["oof"], ok=ok, taus=parse_tau(args.noun_tau),
                              persists=args.feature_persist)
    noun_sel = select_config(noun_c, tol)
    X, y = cv["X_y"]
    final = fit_binary(X, y, cv["chosen"]["C"])
    probe_path = root / "noun_probe.npz"
    probe_md5 = save_probe(probe_path, final, feature=cv["chosen"]["feature"], C=cv["chosen"]["C"])
    payload["arms"] = {
        "strong": {"task_ready": "HELD"},
        "hsr": {"task_ready": "HELD and persist_p(kv_L4_v1 place)", "probe": str(default_probe_path()),
                "probe_md5": rs.md5_of(default_probe_path()), **hsr_sel["chosen"], "feasible": hsr_sel["feasible"],
                "dev_stats": hsr_sel["chosen_stats"]},
        "noun": {"task_ready": "HELD and persist_p(noun probe p >= tau), noun_ok required",
                 "feature": cv["chosen"]["feature"], "C": cv["chosen"]["C"], "probe": str(probe_path),
                 "probe_md5": probe_md5, "probe_converged": bool(final.converged), **noun_sel["chosen"],
                 "feasible": noun_sel["feasible"], "dev_stats": noun_sel["chosen_stats"],
                 "train_rows": cv["n_rows"], "train_pos": cv["n_pos"]}}
    payload["dev_tables"] = {"hsr_grid": hsr_sel["grid"], "noun_grid": noun_sel["grid"], "noun_cv": cv["table"],
                             "noun_cv_spec": {"n_groups": cv["n_groups"], "n_splits": cv["n_splits"],
                                              "fold_of_group": cv["fold_of_group"]},
                             "hsr_by_condition": next(c["by_condition"] for c in hsr_c if c["config"] == hsr_sel["chosen"]),
                             "noun_by_condition": next(c["by_condition"] for c in noun_c if c["config"] == noun_sel["chosen"])}
    # achievability / measurement uncertainty only (user decision: dev spread is not a tolerance)
    hfl = grid_flags(traces, hp, hsr_sel["chosen"]["tau"], ha)
    nfl = grid_flags(traces, cv["oof"], noun_sel["chosen"]["tau"], None, ok)
    payload["dev_uncertainty"] = {
        "note": "episode-block bootstrap of the chosen dev points (P_task, pooled conditions); reported for "
                "achievability and uncertainty only — never used as a tolerance",
        "hsr": episode_bootstrap(group_counts(traces, truths, mode=mode, flags=hfl, p=int(hsr_sel["chosen"]["p"])),
                                 B=args.dev_boot_b, seed=BOOT_SEED),
        "noun": episode_bootstrap(group_counts(traces, truths, mode=mode, flags=nfl, p=int(noun_sel["chosen"]["p"])),
                                  B=args.dev_boot_b, seed=BOOT_SEED)}
    if args.dry_run:
        rs.write_json(root / "dev_dry_run.json", payload)
        print(f"[dev] dry run -> {root / 'dev_dry_run.json'}")
        return payload
    rs.write_json(prereg_path, payload)
    print(f"[dev] K={K}; hsr {hsr_sel['chosen']} (feasible {hsr_sel['feasible']}); noun {cv['chosen']} "
          f"{noun_sel['chosen']} (feasible {noun_sel['feasible']}) -> {prereg_path}. STOP: lead / user confirm.")
    return payload


def _check_prereg(prereg: Mapping) -> None:
    need = ("version", "truth", "arms", "latch_mode_test", "bootstrap", "interventions", "perception", "tolerance")
    miss = [k for k in need if k not in prereg]
    if miss or prereg.get("version") != PREREG_VERSION:
        raise SystemExit(f"not a T55 prereg (missing {miss}, version {prereg.get('version')})")


def score_test(traces: Sequence[Trace], prereg: Mapping, cache: FeatureCache, *, mode: str,
               B: int, seed: int, block: int) -> dict:
    """Every condition x arm x predicate (module docstring). `traces` must all be test traces."""
    K = int(prereg["truth"]["K"])
    pm = prereg["truth"].get("persistence", "causal")
    truths = {tr.tid: truth(tr, K, pm) for tr in traces}
    arms = prereg["arms"]
    hp, ha = {}, {}
    for tr in traces:
        hp[tr.tid], ha[tr.tid] = hsr_place_proba(cache.get(tr)["kv_l4"])
    model, feat = load_probe(pathlib.Path(arms["noun"]["probe"]))
    if feat != arms["noun"]["feature"]:
        raise SystemExit(f"noun probe feature {feat} != prereg {arms['noun']['feature']}")
    npb = {tr.tid: model.proba(noun_matrix(cache.get(tr), feat).astype(np.float64))[:, 1] for tr in traces}
    flags = {"hsr": {tr.tid: flag_from(hp[tr.tid], arms["hsr"]["tau"], ha[tr.tid]) for tr in traces},
             "noun": {tr.tid: flag_from(npb[tr.tid], arms["noun"]["tau"], ok=cache.get(tr)["noun_ok"]) for tr in traces}}
    persist = {"hsr": int(arms["hsr"]["p"]), "noun": int(arms["noun"]["p"])}
    res: dict = {"K": K, "latch_mode": mode, "by": {}, "per_episode": [], "state4": {}}
    for cond in CONDITIONS:
        ct = [tr for tr in traces if tr.condition == cond and tr.latch.get(mode) is not None]
        if not ct:
            continue
        res["state4"][cond] = pr.state_counts(np.concatenate([truths[tr.tid]["state4"][tr.window] for tr in ct]))
        for pred in PREDICATES:
            m_strong = {tr.tid: trace_metrics(ready_of(tr, mode, None, 1), truths[tr.tid][pred], tr.window,
                                              truths[tr.tid]["state4"]) for tr in ct}
            # the strong rule on its own
            strong_tab = {"false_pass": sum(m["false_pass"] for m in m_strong.values()),
                          "false_reject": sum(m["false_reject"] for m in m_strong.values()),
                          "n_neg": sum(m["n_neg"] for m in m_strong.values()),
                          "n_pos": sum(m["n_pos"] for m in m_strong.values()),
                          "false_pass_by_state": {s: sum(m["false_pass_by_state"][s] for m in m_strong.values())
                                                  for s in pr.STATES4},
                          "latency": pr.latency_summary([e for m in m_strong.values() for e in m["events"]])}
            res["by"].setdefault(cond, {}).setdefault(pred, {})["strong"] = strong_tab
            for arm in ("hsr", "noun"):
                per_group: dict[str, dict] = {}
                mb_rows, b_disc, c_disc = [], 0, 0
                fp_state = {s: 0 for s in pr.STATES4}
                ev_all = []
                for tr in ct:
                    t = truths[tr.tid][pred]
                    r_s = ready_of(tr, mode, None, 1)
                    r_a = ready_of(tr, mode, flags[arm][tr.tid], persist[arm])
                    m_a = trace_metrics(r_a, t, tr.window, truths[tr.tid]["state4"])
                    pc = paired_counts(m_strong[tr.tid], m_a)
                    g = per_group.setdefault(tr.group, {k: 0.0 for k in COUNT_KEYS})
                    for k in COUNT_KEYS:
                        g[k] += pc[k]
                    w = tr.window
                    neg, pos = w & ~t, w & t
                    mb_rows.append(np.stack([neg, pos, r_s & neg, r_a & neg, ~r_s & pos, ~r_a & pos], 1)[w])
                    b_disc += int((r_a & ~r_s & w).sum())
                    c_disc += int((~r_a & r_s & w).sum())
                    for s in pr.STATES4:
                        fp_state[s] += m_a["false_pass_by_state"][s]
                    ev_all += m_a["events"]
                    res["per_episode"].append({"condition": cond, "predicate": pred, "arm": arm, "trace": tr.tid,
                                               "group": tr.group, **pc})
                pooled = pair_stats(sum_counts(list(per_group.values())))
                res["by"][cond][pred][arm] = {
                    "pooled": _scalar_stats(pooled), "counts": sum_counts(list(per_group.values())),
                    "false_pass_by_state": fp_state, "latency": pr.latency_summary(ev_all),
                    "episode_bootstrap": episode_bootstrap(per_group, B=B, seed=seed),
                    "moving_block": moving_block_bootstrap(mb_rows, block=block, B=B, seed=seed),
                    "frame_mcnemar_reference": mcnemar(b_disc, c_disc)}
    return res


def stage_test(args) -> dict:
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    pp = pathlib.Path(args.prereg)
    if not pp.is_file():
        raise SystemExit(f"--prereg {pp} not found (dev stage first, then the lead's confirmation)")
    md5_before = rs.md5_of(pp)
    prereg = json.loads(pp.read_text())
    _check_prereg(prereg)
    root = rs.out_root("")
    resim_dir, feat_dir = root / "resim" / "test", root / "feat" / "test"
    pdir = root / "perception" / "test" / prereg["perception"]["name"]
    traces = load_resim_traces(resim_dir, feat_dir, "test", perception_dir=pdir)
    if not traces:
        raise SystemExit(f"no test re-sims under {resim_dir}")
    check_baselines(traces, "test")
    want = prereg["interventions"]
    ivt = [t for t in traces if t.condition in ("empty", "slip")]
    ignored = sorted({t.tag for t in ivt if t.tag not in want})
    bad = sorted({t.tid for t in ivt if t.tag in want and _canon(t.meta.get("intervention")) != _canon(want[t.tag])})
    if bad:
        raise SystemExit(f"test interventions {bad[:5]} are not the prereg's (same tag, other parameters)")
    traces = [t for t in traces if t.condition not in ("empty", "slip") or t.tag in want]
    missing_tags = sorted(set(want) - {t.tag for t in ivt})
    missing = [t.tid for t in traces if t.feat_path is None]
    if missing:
        raise SystemExit(f"{len(missing)} test traces without features: {missing[:5]}")
    require_mode(traces, prereg["latch_mode_test"])
    cache = FeatureCache()
    out = {"step": STEP, "stage": "test", "created": rs.now_kst(), "git_head": git_head(), "argv": sys.argv,
           "prereg": str(pp), "prereg_md5": md5_before, "missing_intervention_tags": missing_tags,
           "ignored_intervention_tags": ignored, "slip_excluded": slip_exclusions(resim_dir),
           "perception_dir": str(pdir), "tolerance": prereg["tolerance"],
           "e0_exact": {t.key: t.meta.get("e0_exact") for t in traces if t.tag == "E0" and t.prompt_tag == "orig"},
           "results": {}}
    modes = [prereg["latch_mode_test"]] + list(prereg.get("latch_mode_test_secondary") or [])
    for mode in modes:
        if not any(t.latch.get(mode) is not None for t in traces):
            out["results"][mode] = {"unavailable": True}
            continue
        out["results"][mode] = score_test(traces, prereg, cache, mode=mode, B=int(prereg["bootstrap"]["B"]),
                                          seed=int(prereg["bootstrap"]["seed"]),
                                          block=int(prereg["bootstrap"]["moving_block_frames"]))
    K = int(prereg["truth"]["K"])
    nat = {}
    pm = prereg["truth"].get("persistence", "causal")
    m0 = prereg["latch_mode_test"]
    for tr in traces:
        if tr.condition == "natural":
            t = truth(tr, K, pm)
            nat[tr.key] = pr.natural_events(t, np.asarray(tr.gripper_used) < CLOSED_BELOW,
                                            None if tr.latch.get(m0) is None else tr.latch[m0]["phase"])
    out["natural_events"] = nat
    out["primary"] = prereg["latch_mode_test"]
    path = root / "test_results.json"
    rs.write_json(path, out)
    if rs.md5_of(pp) != md5_before:
        raise SystemExit("the prereg changed during the test stage")
    print(f"[test] -> {path} (prereg md5 {md5_before} unchanged)")
    return out


# ------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--prereg", default=None, help="test: the prereg written by the dev stage")
    ap.add_argument("--k-candidates", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--K", type=int, default=None, help="dev: override the K rule (recorded)")
    ap.add_argument("--persist-mode", default="causal", choices=pr.PERSIST_MODES)
    ap.add_argument("--latch-mode-dev", default=PRIMARY_MODE, choices=("reground", "oracle"))
    ap.add_argument("--latch-mode-test", default=PRIMARY_MODE, choices=LATCH_MODES)
    ap.add_argument("--perception", default=DEFAULT_PERCEPTION,
                    help="dev: perception/<set>/<this>/ of t55_perception (<attention source>-<backend>)")
    ap.add_argument("--dev-boot-b", type=int, default=BOOT_B, help="dev: bootstrap draws of the uncertainty report")
    ap.add_argument("--hsr-tau", nargs="+", default=["argmax", "0.5", "0.7", "0.9"])
    ap.add_argument("--noun-tau", nargs="+", default=["0.3", "0.5", "0.7", "0.9"])
    ap.add_argument("--feature-persist", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--noun-cv-layers", type=int, nargs="+", default=list(range(18)))
    ap.add_argument("--no-hidden", action="store_true", help="dev: leave noun_hidden_final out of the CV")
    ap.add_argument("--C-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--tol-false-reject", type=float, default=DEFAULT_TOL["false_reject_rate"])
    ap.add_argument("--tol-latency", type=float, default=DEFAULT_TOL["latency_frames"])
    ap.add_argument("--tol-undetected", type=float, default=DEFAULT_TOL["undetected_events"])
    ap.add_argument("--no-val", action="store_true", help="dev: leave the T48a val frames out")
    ap.add_argument("--interventions", nargs="+", default=None,
                    help="dev: the intervention tags fixed for T55 (default every tag present in resim/dev)")
    ap.add_argument("--no-features", action="store_true", help="dev: K and the strong rule only (no GPU output yet)")
    ap.add_argument("--dry-run", action="store_true", help="dev: write dev_dry_run.json instead of the prereg")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> dict:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.stage == "test" and not args.prereg:
        ap.error("--stage test needs --prereg")
    if any(k < 1 for k in args.k_candidates) or (args.K is not None and args.K < 1):
        ap.error("K >= 1")
    t0 = time.time()
    out = stage_dev(args) if args.stage == "dev" else stage_test(args)
    print(f"[{STEP}] {args.stage} done in {time.time() - t0:.1f}s")
    return out


__all__ = ["ARMS", "CONDITIONS", "LATCH_MODES", "USER_TOL", "FeatureCache", "Trace", "check_baselines",
           "combination_grid", "condition_of", "episode_bootstrap", "group_counts", "slip_exclusions",
           "tolerance_record",
           "evaluate", "flag_from", "load_resim_traces", "load_val_traces", "main", "mcnemar", "moving_block_bootstrap",
           "noun_cv", "pair_stats", "paired_counts", "score_test", "select_K", "select_config", "stage_dev",
           "stage_test", "trace_metrics", "truth"]


if __name__ == "__main__":
    main()
