"""T55 auxiliary — T60 termination windows (P / H) next to the full test window (offline, CPU, read-only inputs).

User decision (2026-10-14, `T55.task.md` "사용자 판정 (2026-10-14 …)" and the e4 relay): the **main T55 verdict stays
on the full window**; the T60 windows P (placement complete) and H (ready-pose return complete) are an
**auxiliary** analysis reported **side by side** with it. T55 also scores post-release / return behaviour, so the
P-cut never replaces the full evaluation. The mask definition is fixed before this analysis (T60-v1,
`t60_termination`) and applied identically to every arm; the primary window is never swapped.

Nothing here changes `t55_study` / `t55_predicate` (imported only; their md5s are recorded in the output).

Mask rule (fixed before any windowed number was computed)
---------------------------------------------------------
* Source: the T60 step-1 masks of the 16 T53 episodes, ``masks/T53__C1a_<key>_s<seed>.npz`` (`t60_termination`
  ``keep_frames_P`` / ``keep_frames_H``: inference frame f kept iff ``t_step[f] <= t_end``; a run that never reaches
  P (or H) keeps all 75 frames). The file's keep arrays are re-derived from its ``t_P`` / ``t_H`` and
  ``chunk_start_steps`` with `t60_termination.keep_frames` and must agree.
* The masks are computed on the **recorded** T53 rollout. T55's E0 re-sim is bit-exact to that record for all 16
  test episodes (``test_results.json`` ``e0_exact``), and its ``t_step`` must equal the mask's
  ``chunk_start_steps`` (checked per trace).
* **One mask per episode, applied to every trace of that episode**: ``natural`` (E0, own prompt), ``instruction``
  (E0 images with the swapped nouns), ``empty`` and ``slip`` (ghost / park) re-sims. The interventions replay the
  same recorded executed actions open loop, so the robot's task timeline (place, return) is the recorded one;
  the intervened physics is **not** used to re-derive P / H (a parked target is never placed, which would keep
  the full horizon for exactly the conditions whose behaviour after the intervention is in question and make the
  window differ between conditions of one episode).
* Scored frames of a trace in window W = the trace's own condition window (all frames, or frames >=
  ``window_start`` for empty / slip) AND the episode's W mask. Everything else (truth persistence K, feature
  persistence p, latch trace, prereg arms) is computed on the full trace exactly as in `t55_study.score_test`;
  only the set of scored frames changes. The windows are prefixes (t_step <= t_end), and every filter is causal,
  so a windowed score equals the score of the trace truncated at t_end.
* The same mask goes to the strong rule and both combined arms (one ``Trace.window`` per trace, shared by all
  arms), to both predicates and to all latch modes.

Scoring = `t55_study.score_test` unchanged (per condition x predicate x arm: counts, pooled rates, paired
latency, episode-block bootstrap B / seed from the prereg (10000 / 5507), moving block, McNemar reference).
**Pooled decision rows** (P_task, every condition): per-trace paired counts summed per episode block over all
conditions, then divided (`t55_study.pair_stats`, the lead's 합산 definition), with the same episode-block
bootstrap; the prereg tolerances (+5 %p false reject, +1 frame mean paired latency, 0 extra undetected onsets) are
evaluated mechanically for every window (auxiliary — the verdict is the full window's).

Parity: before any P / H number is written, the full window must reproduce ``test_results.json`` exactly
(``results[mode]`` for every latch mode — counts, rates, bootstrap CIs, per-trace rows — and ``natural_events``);
otherwise the run stops (exit 2) without P / H output.

Usage::

    source scripts/setup/env.sh
    export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 PYTHONPATH=$PWD .venv-ag3s/bin/python -m \\
        AG3S.ag3s.experiments.studies.t55_windows [--t55-run <T55 run>] [--t60-step1 <T60 step1 dir>]
    # -> $AG3S_RUN_DIR/verify/T55_windows/{t55_windows.json, side_by_side.csv, decision.csv, scores_<W>.json}
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t55_predicate as pr
from AG3S.ag3s.experiments.studies import t55_study as st
from AG3S.ag3s.experiments.studies import t60_termination as T

STEP = "T55_windows"
VERSION = "t55-windows-v1"
WINDOWS = ("full", "P", "H")
MASK_WINDOWS = ("P", "H")
_DEV = "/mnt/dev/AG3S_VLA-storage/results/development"
DEFAULT_T55_RUN = f"{_DEV}/20261013T171325+0900_cf38c9dbca7b_2ebfd8a5a43c"
DEFAULT_T60_STEP1 = f"{_DEV}/20261013T223348+0900_cf38c9dbca7b_8aef81831126/verify/T60/step1"
OUT_SUB = "verify/T55_windows"
PINNED = ("t55_study.py", "t55_predicate.py", "t55_resim.py", "t55_features.py", "t55_perception.py",
          "t60_termination.py")
DECISION_ARMS = ("hsr", "noun")
CI_KEYS = ("d_fp_rate", "d_fr_rate", "d_latency_mean", "d_undetected", "d_fp_count", "d_fr_count")
MASK_RULE = ("one T60-v1 mask per T53 episode (recorded rollout = bit-exact T55 E0), applied to every trace of the "
             "episode (natural, instruction x3, empty, slip ghost/park) and to every arm / predicate / latch mode; "
             "scored frames = condition window AND (t_step <= t_end); no P (or H) -> all frames kept")


# ------------------------------------------------------------------------------ masks


def mask_path(step1: pathlib.Path, key: str) -> pathlib.Path:
    m = sorted((pathlib.Path(step1) / "masks").glob(f"T53__C1a_{key}_s*.npz"))
    if len(m) != 1:
        raise SystemExit(f"[{key}] expected one T60 mask T53__C1a_{key}_s*.npz under {step1}/masks, found {m}")
    return m[0]


def _t_end(v) -> Optional[int]:
    v = int(v)
    return None if v < 0 else v


def load_mask(path: pathlib.Path) -> dict:
    """One episode's T60 endpoints and keep arrays; the keep arrays are re-derived and must agree."""
    with np.load(path, allow_pickle=False) as z:
        proto = str(z["protocol"])
        css = np.asarray(z["chunk_start_steps"], np.int64)
        t_p, t_h = _t_end(z["t_P"]), _t_end(z["t_H"])
        keep = {w: np.asarray(z[f"keep_frames_{w}"], bool) for w in MASK_WINDOWS}
    if proto != T.PROTOCOL_VERSION:
        raise SystemExit(f"{path}: protocol {proto!r}, expected {T.PROTOCOL_VERSION!r}")
    for w, t in (("P", t_p), ("H", t_h)):
        if not np.array_equal(keep[w], T.keep_frames(css, t)):
            raise SystemExit(f"{path}: keep_frames_{w} differs from t_step <= t_{w} ({t})")
    if t_h is not None and (t_p is None or t_h <= t_p):
        raise SystemExit(f"{path}: t_H {t_h} without an earlier t_P {t_p}")
    return {"path": str(path), "md5": hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest(), "t_P": t_p,
            "t_H": t_h, "chunk_start_steps": css, "P": keep["P"], "H": keep["H"]}


def load_masks(step1: pathlib.Path, keys: Sequence[str]) -> dict:
    return {k: load_mask(mask_path(step1, k)) for k in sorted(set(keys))}


def episode_keep(mask: Mapping, window: str, n: int) -> np.ndarray:
    if window == "full":
        return np.ones(n, bool)
    if window not in MASK_WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {window!r}")
    k = np.asarray(mask[window], bool)
    if len(k) != n:
        raise ValueError(f"mask has {len(k)} frames, trace {n}")
    return k.copy()


# ------------------------------------------------------------------------------ traces


def trace_t_step(tr: st.Trace) -> np.ndarray:
    with np.load(tr.meta["resim"], allow_pickle=False) as z:
        return np.asarray(z["t_step"], np.int64)


def check_alignment(traces: Sequence[st.Trace], masks: Mapping[str, Mapping],
                    t_steps: Optional[Mapping[str, np.ndarray]] = None) -> dict:
    """Every trace's re-sim t_step must equal its episode mask's chunk_start_steps."""
    n = 0
    for tr in traces:
        ts = t_steps[tr.tid] if t_steps is not None else trace_t_step(tr)
        css = masks[tr.key]["chunk_start_steps"]
        if len(ts) != tr.n or not np.array_equal(np.asarray(ts), np.asarray(css)):
            raise SystemExit(f"{tr.tid}: re-sim t_step differs from the T60 mask chunk_start_steps")
        n += 1
    return {"traces_checked": n, "all_equal": True}


def apply_window(traces: Sequence[st.Trace], masks: Mapping[str, Mapping], window: str) -> list[st.Trace]:
    """Copies of `traces` whose ``window`` = own condition window AND the episode mask of `window`.
    Every other field (latch, truth inputs, features) is shared, so all arms see the same frames."""
    out = []
    for tr in traces:
        keep = episode_keep(masks.get(tr.key, {}), window, tr.n) if window != "full" else np.ones(tr.n, bool)
        out.append(dataclasses.replace(tr, window=np.asarray(tr.window, bool) & keep))
    return out


def window_frames(traces: Sequence[st.Trace]) -> dict:
    """Scored frames per condition, and traces left without a scored frame."""
    out: dict = {}
    for tr in traces:
        c = out.setdefault(tr.condition, {"traces": 0, "frames": 0, "empty_traces": []})
        c["traces"] += 1
        c["frames"] += int(np.asarray(tr.window, bool).sum())
        if not np.asarray(tr.window, bool).any():
            c["empty_traces"].append(tr.tid)
    return out


# ------------------------------------------------------------------------------ scoring


def score_window(traces: Sequence[st.Trace], prereg: Mapping, cache: st.FeatureCache,
                 modes: Sequence[str]) -> dict:
    """`t55_study.score_test` for every latch mode (unavailable modes reported as such)."""
    out = {}
    bs = prereg["bootstrap"]
    for mode in modes:
        if not any(t.latch.get(mode) is not None for t in traces):
            out[mode] = {"unavailable": True}
            continue
        out[mode] = st.score_test(traces, prereg, cache, mode=mode, B=int(bs["B"]), seed=int(bs["seed"]),
                                  block=int(bs["moving_block_frames"]))
    return out


def natural_events(traces: Sequence[st.Trace], prereg: Mapping, mode: str) -> dict:
    """`t55_predicate.natural_events` of the natural traces on their scored frames (full window = all frames,
    the same call as `t55_study.stage_test`)."""
    K = int(prereg["truth"]["K"])
    pm = prereg["truth"].get("persistence", "causal")
    out = {}
    for tr in traces:
        if tr.condition != "natural":
            continue
        t = st.truth(tr, K, pm)
        w = np.asarray(tr.window, bool)
        sub = {k: np.asarray(t[k])[w] for k in ("p_grasp", "p_task", "placed")}
        ph = None if tr.latch.get(mode) is None else np.asarray(tr.latch[mode]["phase"])[w]
        out[tr.key] = pr.natural_events(sub, (np.asarray(tr.gripper_used) < st.CLOSED_BELOW)[w], ph)
    return out


def pooled_decision(res_mode: Mapping, prereg: Mapping, *, predicate: str = "p_task",
                    arms: Sequence[str] = DECISION_ARMS) -> dict:
    """Pooled (every condition) paired counts per arm from the per-trace rows, the episode-block bootstrap and
    the prereg tolerances evaluated mechanically."""
    tol = prereg["tolerance"]
    bs = prereg["bootstrap"]
    out = {}
    for arm in arms:
        rows = [r for r in res_mode["per_episode"] if r["predicate"] == predicate and r["arm"] == arm]
        per_group: dict[str, dict] = {}
        for r in rows:
            g = per_group.setdefault(r["group"], {k: 0.0 for k in st.COUNT_KEYS})
            for k in st.COUNT_KEYS:
                g[k] += r[k]
        counts = st.sum_counts(list(per_group.values()))
        stats = st._scalar_stats(st.pair_stats(counts))
        boot = st.episode_bootstrap(per_group, B=int(bs["B"]), seed=int(bs["seed"]))
        dfr, dlat, dund = stats["d_fr_rate"], stats["d_latency_mean"], stats["d_undetected"]
        checks = {"false_reject": None if dfr is None else bool(dfr <= float(tol["false_reject_rate"])),
                  "latency": None if dlat is None else bool(dlat <= float(tol["latency_frames"])),
                  "undetected": None if dund is None else bool(dund <= float(tol["undetected_events"]))}
        out[arm] = {"counts": counts, "pooled": stats, "episode_bootstrap": boot,
                    "conditions": sorted({r["condition"] for r in rows}), "n_traces": len(rows),
                    "tolerance_checks": checks,
                    "within_tolerance": (None if any(v is None for v in checks.values())
                                         else all(checks.values())),
                    "fewer_false_passes": None if stats["d_fp_count"] is None else bool(stats["d_fp_count"] < 0)}
    out["_tolerance"] = {k: tol[k] for k in ("false_reject_rate", "latency_frames", "undetected_events")}
    return out


# ------------------------------------------------------------------------------ parity


def canonical(x: Any) -> Any:
    """What `t55_resim.write_json` stores (NaN / inf -> None), read back."""
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    return json.loads(json.dumps(rs.jsonable(x)))


def first_difference(a: Any, b: Any, path: str = "") -> Optional[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{path}: keys {sorted(set(a) ^ set(b))[:5]}"
        for k in a:
            d = first_difference(a[k], b[k], f"{path}/{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if a != b or type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return f"{path}: {a!r} != {b!r}"
    return None


def parity(mine_full: Mapping, mine_nat: Mapping, stored: Mapping) -> dict:
    out = {"modes": {}, "natural_events": None, "exact": True}
    for mode, res in mine_full.items():
        ref = stored["results"].get(mode)
        d = "missing in test_results.json" if ref is None else first_difference(canonical(res), ref)
        out["modes"][mode] = {"exact": d is None, "first_difference": d}
        out["exact"] &= d is None
    d = first_difference(canonical(mine_nat), stored.get("natural_events"))
    out["natural_events"] = {"exact": d is None, "first_difference": d}
    out["exact"] &= d is None
    return out


# ------------------------------------------------------------------------------ tables


def _ci(boot: Mapping, k: str) -> list:
    b = boot.get(k) or {}
    return [b.get("lo"), b.get("hi")]


def side_by_side(scores: Mapping[str, Mapping], mode: str) -> list[dict]:
    """One row per condition x predicate x arm x window (window last so full / P / H sit together)."""
    rows = []
    full = scores["full"][mode]
    for cond in st.CONDITIONS:
        if cond not in full["by"]:
            continue
        for pred in st.PREDICATES:
            for arm in st.ARMS:
                for w in WINDOWS:
                    r = scores[w][mode]["by"][cond][pred][arm]
                    row = {"mode": mode, "condition": cond, "predicate": pred, "arm": arm, "window": w}
                    if arm == "strong":
                        lat = r["latency"]
                        row.update({"n_neg": r["n_neg"], "n_pos": r["n_pos"], "fp": r["false_pass"],
                                    "fr": r["false_reject"],
                                    "fp_rate": (r["false_pass"] / r["n_neg"]) if r["n_neg"] else None,
                                    "fr_rate": (r["false_reject"] / r["n_pos"]) if r["n_pos"] else None,
                                    "fp_wrong_object": r["false_pass_by_state"]["wrong_object"],
                                    "fp_empty": r["false_pass_by_state"]["empty"],
                                    "fp_placed": r["false_pass_by_state"]["placed"],
                                    "events": lat["n_events"], "undetected": lat["n_undetected"],
                                    "latency_mean": lat["latency_mean"]})
                    else:
                        c, p, b = r["counts"], r["pooled"], r["episode_bootstrap"]
                        lat = r["latency"]
                        row.update({"n_neg": c["neg"], "n_pos": c["pos"], "fp": c["fp_a"], "fr": c["fr_a"],
                                    "fp_rate": p["fp_rate_arm"], "fr_rate": p["fr_rate_arm"],
                                    "fp_wrong_object": r["false_pass_by_state"]["wrong_object"],
                                    "fp_empty": r["false_pass_by_state"]["empty"],
                                    "fp_placed": r["false_pass_by_state"]["placed"],
                                    "events": lat["n_events"], "undetected": lat["n_undetected"],
                                    "latency_mean": lat["latency_mean"],
                                    "fp_strong": c["fp_s"], "fr_strong": c["fr_s"]})
                        for k in CI_KEYS:
                            row[k] = p[k]
                            row[f"{k}_lo"], row[f"{k}_hi"] = _ci(b, k)
                    rows.append(row)
    return rows


def decision_rows(decisions: Mapping[str, Mapping], mode: str) -> list[dict]:
    rows = []
    for arm in DECISION_ARMS:
        for w in WINDOWS:
            d = decisions[w][mode][arm]
            c, p, b = d["counts"], d["pooled"], d["episode_bootstrap"]
            row = {"mode": mode, "arm": arm, "window": w, "neg": c["neg"], "pos": c["pos"], "fp_strong": c["fp_s"],
                   "fp_arm": c["fp_a"], "fr_strong": c["fr_s"], "fr_arm": c["fr_a"], "events": c["events"],
                   "lat_n": c["lat_n"], "und_strong": c["und_s"], "und_arm": c["und_a"]}
            for k in CI_KEYS:
                row[k] = p[k]
                row[f"{k}_lo"], row[f"{k}_hi"] = _ci(b, k)
            row.update({f"pass_{k}": v for k, v in d["tolerance_checks"].items()})
            row["within_tolerance"] = d["within_tolerance"]
            rows.append(row)
    return rows


def write_csv(path: pathlib.Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


def headline(scores: Mapping, decisions: Mapping, mode: str) -> dict:
    out: dict = {"pooled_p_task": {}, "instruction_wrong_object_false_pass": {}}
    for arm in DECISION_ARMS:
        out["pooled_p_task"][arm] = {}
        for w in WINDOWS:
            d = decisions[w][mode][arm]
            p, b = d["pooled"], d["episode_bootstrap"]
            out["pooled_p_task"][arm][w] = {
                **{k: {"point": p[k], "ci95": _ci(b, k)} for k in ("d_fp_rate", "d_fr_rate", "d_latency_mean",
                                                                     "d_undetected")},
                "fp_strong": d["counts"]["fp_s"], "fp_arm": d["counts"]["fp_a"], "neg": d["counts"]["neg"],
                "fr_strong": d["counts"]["fr_s"], "fr_arm": d["counts"]["fr_a"], "pos": d["counts"]["pos"],
                "within_tolerance": d["within_tolerance"]}
    for w in WINDOWS:
        by = scores[w][mode]["by"].get("instruction", {}).get("p_task")
        if by is None:
            continue
        out["instruction_wrong_object_false_pass"][w] = {
            "strong": by["strong"]["false_pass_by_state"]["wrong_object"],
            **{a: by[a]["false_pass_by_state"]["wrong_object"] for a in DECISION_ARMS},
            "n_neg": by["strong"]["n_neg"], "n_wrong_object_frames": scores[w][mode]["state4"]["instruction"]["wrong_object"]}
    return out


# ------------------------------------------------------------------------------ main


def md5_of(path) -> str:
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def pinned_md5() -> dict:
    here = pathlib.Path(__file__).resolve().parent
    return {n: md5_of(here / n) for n in PINNED}


def out_dir(run_dir: Optional[str], forbid: Sequence[str]) -> pathlib.Path:
    from common.run_output import resolve_output_path

    run = run_dir or os.environ.get("AG3S_RUN_DIR")
    if not run:
        raise SystemExit('AG3S_RUN_DIR is not set — export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python '
                         '-m common.run_output)" (a new run)')
    for f in forbid:
        if pathlib.Path(run).resolve() == pathlib.Path(f).resolve():
            raise SystemExit(f"output run {run} is an input run (read-only)")
    return resolve_output_path(OUT_SUB, run_dir=run, default_name=OUT_SUB)


def load_test_traces(t55_root: pathlib.Path, prereg: Mapping) -> list[st.Trace]:
    """The traces `t55_study.stage_test` scores (same loader, same intervention filter)."""
    resim_dir, feat_dir = t55_root / "resim" / "test", t55_root / "feat" / "test"
    pdir = t55_root / "perception" / "test" / prereg["perception"]["name"]
    traces = st.load_resim_traces(resim_dir, feat_dir, "test", perception_dir=pdir)
    if not traces:
        raise SystemExit(f"no test re-sims under {resim_dir}")
    st.check_baselines(traces, "test")
    want = prereg["interventions"]
    traces = [t for t in traces if t.condition not in ("empty", "slip") or t.tag in want]
    bad = [t.tid for t in traces if t.condition in ("empty", "slip")
           and st._canon(t.meta.get("intervention")) != st._canon(want[t.tag])]
    if bad:
        raise SystemExit(f"interventions {bad[:5]} differ from the prereg's")
    if any(t.feat_path is None for t in traces):
        raise SystemExit("test traces without features")
    return traces


def run(args) -> dict:
    t0 = time.time()
    t55_root = pathlib.Path(args.t55_run) / "verify" / "T55"
    prereg_p, results_p = t55_root / st.PREREG_NAME, t55_root / "test_results.json"
    md5_in = {"prereg": md5_of(prereg_p), "test_results": md5_of(results_p)}
    code_md5 = pinned_md5()
    prereg = json.loads(prereg_p.read_text())
    st._check_prereg(prereg)
    stored = json.loads(results_p.read_text())
    if stored.get("prereg_md5") != md5_in["prereg"]:
        raise SystemExit("test_results.json was scored with another prereg")
    od = out_dir(args.out_run, [args.t55_run, str(pathlib.Path(args.t60_step1).parents[2])])
    traces = load_test_traces(t55_root, prereg)
    keys = sorted({t.key for t in traces})
    masks = load_masks(pathlib.Path(args.t60_step1), keys)
    align = check_alignment(traces, masks)
    modes = [prereg["latch_mode_test"]] + list(prereg.get("latch_mode_test_secondary") or [])
    cache = st.FeatureCache()
    m0 = prereg["latch_mode_test"]
    # 1) full window and parity — nothing else is written unless it holds
    full = score_window(traces, prereg, cache, modes)
    nat_full = natural_events(traces, prereg, m0)
    par = parity(full, nat_full, stored)
    print(f"[{STEP}] parity full vs test_results.json: {par['exact']} "
          f"{ {m: v['exact'] for m, v in par['modes'].items()} } natural_events {par['natural_events']['exact']}")
    meta = {"step": STEP, "version": VERSION, "created": _now(), "git_head": st.git_head(), "argv": sys.argv,
            "inputs": {"t55_root": str(t55_root), "prereg": str(prereg_p), "test_results": str(results_p),
                       "md5": md5_in, "t60_step1": str(args.t60_step1), "t60_protocol": T.PROTOCOL_VERSION,
                       "t60_params": dataclasses.asdict(T.Params())},
            "code_md5_start": code_md5, "mask_rule": MASK_RULE, "definitions": __doc__,
            "user_decision": ("main verdict = full window; P / H auxiliary, side by side; mask fixed before "
                              "analysis, identical for every arm; the primary window is never swapped; T55 "
                              "also evaluates post-release / return behaviour, so P never replaces full"),
            "primary_mode": m0, "modes": modes, "alignment": align, "parity": par}
    if not par["exact"]:
        rs_write(od / "t55_windows.json", {**meta, "status": "STOP: full-window parity failed — no P / H output"})
        raise SystemExit(2)
    scores = {"full": full}
    frames = {"full": window_frames(traces)}
    nat = {"full": nat_full}
    for w in MASK_WINDOWS:
        tw = apply_window(traces, masks, w)
        frames[w] = window_frames(tw)
        scores[w] = score_window(tw, prereg, cache, modes)
        nat[w] = natural_events(tw, prereg, m0)
    decisions = {w: {m: pooled_decision(scores[w][m], prereg) for m in modes
                     if not scores[w][m].get("unavailable")} for w in WINDOWS}
    endpoints = {k: {"t_P": m["t_P"], "t_H": m["t_H"], "frames_kept_P": int(m["P"].sum()),
                     "frames_kept_H": int(m["H"].sum()), "n_frames": int(len(m["P"])), "mask": m["path"],
                     "mask_md5": m["md5"]} for k, m in masks.items()}
    out = {**meta, "status": "ok", "endpoints": endpoints,
           "frames_removed": {w: int(sum(e["n_frames"] - e[f"frames_kept_{w}"] for e in endpoints.values()))
                              for w in MASK_WINDOWS},
           "scored_frames": frames, "natural_events": nat, "pooled_decision": decisions,
           "headline": {m: headline(scores, decisions, m) for m in modes if not full[m].get("unavailable")},
           "outputs": {w: str(od / f"scores_{w}.json") for w in WINDOWS}}
    for w in WINDOWS:
        rs_write(od / f"scores_{w}.json", {"window": w, "results": scores[w], "natural_events": nat[w]})
    sbs = [r for m in modes if not full[m].get("unavailable") for r in side_by_side(scores, m)]
    dec = [r for m in modes if not full[m].get("unavailable") for r in decision_rows(decisions, m)]
    write_csv(od / "side_by_side.csv", sbs)
    write_csv(od / "decision.csv", dec)
    out["outputs"].update({"side_by_side_csv": str(od / "side_by_side.csv"), "decision_csv": str(od / "decision.csv")})
    end_md5 = {"prereg": md5_of(prereg_p), "test_results": md5_of(results_p)}
    out["code_md5_end"] = pinned_md5()
    out["inputs_unchanged"] = end_md5 == md5_in and out["code_md5_end"] == code_md5
    out["seconds"] = time.time() - t0
    rs_write(od / "t55_windows.json", out)
    print(f"[{STEP}] -> {od} ({out['seconds']:.1f}s; inputs unchanged {out['inputs_unchanged']})")
    return out


def rs_write(path: pathlib.Path, payload) -> None:
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    rs.write_json(path, payload)


def _now() -> str:
    from AG3S.ag3s.experiments.studies import t55_resim as rs

    return rs.now_kst()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--t55-run", default=DEFAULT_T55_RUN, help="the T55 run (read only)")
    ap.add_argument("--t60-step1", default=DEFAULT_T60_STEP1, help="T60 verify/T60/step1 (masks/, read only)")
    ap.add_argument("--out-run", default=None, help="new run dir (default $AG3S_RUN_DIR)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> dict:
    return run(build_parser().parse_args(argv))


__all__ = ["WINDOWS", "apply_window", "canonical", "check_alignment", "decision_rows", "episode_keep",
           "first_difference", "load_mask", "load_masks", "natural_events", "parity", "pooled_decision",
           "score_window", "side_by_side", "window_frames"]


if __name__ == "__main__":
    main()
