"""T56 auxiliary — T60 termination windows (P / H) on the T56 test selection metrics (prepared; run after T56 test).

User decision (2026-10-14, `T56.task.md` "사용자 판정 (2026-10-14 …)"): the **main T56 verdict stays on the full
window**; T60's P (placement complete) / H (ready-pose return complete) windows are an **auxiliary** analysis,
reported side by side; the mask is fixed before analysis and applied identically to every candidate (cell / arm)
and every self-score variant (S0 / S1 / S2); the primary window is never swapped.

Nothing here changes a `t56_*` / `t54_*` file: `t56_study` / `t56_select` / `t56_selfscore` are imported and called
unchanged (`load_record`, `score_record`, `episode_tables`, `evaluate_group`, `public`). Inputs are read only.

Mask rule (fixed before any windowed number is computed)
--------------------------------------------------------
* T60 step-1 masks of the 16 T53 episodes (``masks/T53__C1a_<key>_s<seed>.npz``, protocol T60-v1;
  ``keep_frames_P`` / ``keep_frames_H``: inference frame f kept iff ``t_step[f] <= t_end``; no P (or H) -> all
  frames). The keep arrays are re-derived from ``t_P`` / ``t_H`` (`t55_windows.load_mask`) and the T53 GT
  cache ``t_step`` must equal the mask's ``chunk_start_steps``.
* One mask per episode, applied to every condition (current / hold), kind (cell / arm), candidate and variant.
  The replay geometry positions kept are ``{j : keep[geom.frames[j]]}`` (frames are T53 frame indices); all
  geometries of a record must hold the same frames.
* Two scopes per window, mirroring the existing phase split of `t56_study.analyze`:
  ``both`` (primary auxiliary) — self episode scores **and** GT episode means use only the window's frames
  (``episode_tables(..., phase_self=True)``): post-completion frames neither select nor are graded.
  ``gt`` — self score from every frame (the test-stage selection), GT graded on the window only
  (``phase_self=False``): how the full-window choice fares on task frames.
  The HELD event that defines the S1 / S2 label comes before placement, so it is inside both windows.
* Selection, tie-break (resolution, S1 coverage, name), regret (two kinds), τ-b (undefined counted, never 0),
  auxiliary losses and the paired episode bootstrap (B / seed from the prereg, the same draw matrix for every
  window) are `t56_select` unchanged. The prereg's mechanical criteria (`t56_study.stage_test`) are evaluated per
  window and scope (auxiliary).

Parity: the full window must reproduce ``t56_summary.json`` ``results[cond][group]["all"]`` and
``criteria_mechanical`` exactly (NaN-aware) before P / H is written; otherwise the run stops (exit 2).

Usage (after the T56 test stage wrote ``verify/T56/test/analysis/t56_summary.json``)::

    source scripts/setup/env.sh
    export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$PWD \\
        .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.t56_windows [--t56-run <T56 run>]
    # -> $AG3S_RUN_DIR/verify/T56_windows/{t56_windows.json, windows.csv}
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
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

STEP = "T56_windows"
VERSION = "t56-windows-v1"
WINDOWS = ("full", "P", "H")
MASK_WINDOWS = ("P", "H")
SCOPES = ("both", "gt")
_DEV = "/mnt/dev/AG3S_VLA-storage/results/development"
DEFAULT_T56_RUN = f"{_DEV}/20261013T223613+0900_cf38c9dbca7b_60a2394b8ccc"
DEFAULT_T60_STEP1 = f"{_DEV}/20261013T223348+0900_cf38c9dbca7b_8aef81831126/verify/T60/step1"
SUMMARY_REL = "verify/T56/test/analysis/t56_summary.json"
OUT_SUB = "verify/T56_windows"
PINNED = ("t56_study.py", "t56_select.py", "t56_selfscore.py", "t56_replay.py", "t56_held.py", "t56_cells.py",
          "t54_stats.py", "t60_termination.py")
TABLE_KEYS = ("all:identity:regret", "all:identity:tie_regret", "all:purity:regret", "all:coverage:regret",
              "all:no_target:selected", "all:identity:selected", "all:identity:best")
MASK_RULE = ("one T60-v1 mask per T53 episode (recorded rollout), applied to every condition / kind / candidate / "
             "variant; geometry positions kept = keep[geom.frames[j]]; scope 'both' restricts self scores and GT, "
             "scope 'gt' restricts GT only; no P (or H) -> all frames kept")


# ------------------------------------------------------------------------------ positions


def window_positions(frames: Sequence[int], t_step: Sequence[int], mask: Mapping, window: str) -> list[int]:
    """Geometry positions j whose T53 frame ``frames[j]`` is kept by the episode's `window` mask."""
    css = np.asarray(mask["chunk_start_steps"], np.int64)
    if not np.array_equal(np.asarray(t_step, np.int64), css):
        raise SystemExit("GT t_step differs from the T60 mask chunk_start_steps")
    if window == "full":
        return list(range(len(frames)))
    keep = np.asarray(mask[window], bool)
    return [j for j, i in enumerate(frames) if bool(keep[int(i)])]


def record_positions(rd, t_step: Sequence[int], mask: Mapping) -> dict:
    """``{P: positions, H: positions}`` of one `t56_study.RecordData`; every geometry must hold the same frames."""
    frames = None
    for ck, g in rd.geom.items():
        f = [int(i) for i in g.frames]
        if frames is None:
            frames = f
        elif f != frames:
            raise SystemExit(f"[{rd.key}] geometry {ck} holds other frames than the first one")
    return {w: window_positions(frames or [], t_step, mask, w) for w in MASK_WINDOWS}


def with_positions(rds: Sequence, positions: Mapping[str, Mapping[str, list]]) -> list:
    """Copies of the records whose ``phase_pos`` holds the window positions (the hook `episode_tables` reads)."""
    return [dataclasses.replace(rd, phase_pos=dict(positions[rd.key])) for rd in rds]


# ------------------------------------------------------------------------------ analysis


def analyze_windows(rds: Sequence, positions: Mapping[str, Mapping[str, list]], conds: Sequence[str],
                    kinds: Sequence[str], resolution: float, W: Optional[np.ndarray],
                    fs_fn: Callable[[Any, str, str], Mapping]) -> dict:
    """``res[cond][group] = {"full": evaluate_group, "P": {"both", "gt"}, "H": {"both", "gt"}}``.

    `fs_fn(rd, cond, kind)` = the record's frame scores (`t56_study.score_record` with the prereg's S1 / S2)."""
    from AG3S.ag3s.experiments.studies import t56_study as st

    keys = [rd.key for rd in rds]
    rdw = with_positions(rds, positions)
    out: dict[str, Any] = {}
    for c in conds:
        out[c] = {}
        for k in kinds:
            fs = {rd.key: fs_fn(rd, c, k) for rd in rds}
            for gname, group in st.group_names(k, rds[0].names[k]).items():
                if not group:
                    continue
                rec = {"full": st.evaluate_group(st.episode_tables(rds, c, k, group, fs), group, keys, resolution, W)}
                for w in MASK_WINDOWS:
                    rec[w] = {}
                    for scope in SCOPES:
                        tab = st.episode_tables(rdw, c, k, group, fs, phase=w, phase_self=(scope == "both"))
                        rec[w][scope] = st.evaluate_group(tab, group, keys, resolution, W)
                out[c][gname] = rec
    return out


def criteria(rec: Mapping, pre: Mapping) -> dict:
    """`t56_study.stage_test`'s mechanical criteria of one evaluate_group result (same expressions)."""
    s1, s0 = rec["S1"]["summary"], rec["S0"]["summary"]

    def num(x):
        return None if x is None else float(x)
    r1, r0 = num(s1.get("all:identity:tie_regret")), num(s0.get("all:identity:tie_regret"))
    rt1, rt0 = num(s1.get("all:identity:regret")), num(s0.get("all:identity:regret"))
    t1, t0_ = s1["tau_b:identity"]["mean_defined"], s0["tau_b:identity"]["mean_defined"]
    return {
        "regret_within_tol": {"tie_regret": r1, "regret": rt1, "tol": pre["regret_tol"],
                              "pass_tie_regret": None if r1 is None else r1 <= pre["regret_tol"],
                              "pass_regret": None if rt1 is None else rt1 <= pre["regret_tol"]},
        "vs_S0": {"tie_regret_S1": r1, "tie_regret_S0": r0, "regret_S1": rt1, "regret_S0": rt0,
                  "tie_regret_lower": None if r1 is None or r0 is None else r1 < r0,
                  "regret_lower": None if rt1 is None or rt0 is None else rt1 < rt0,
                  "tau_b_S1": t1, "tau_b_S0": t0_,
                  "tau_b_higher": (None if t1 is None or t0_ is None else t1 > t0_),
                  "tau_b_note": ("undefined on one side — judged by regret" if t1 is None or t0_ is None
                                 else "")},
        "aux_losses": {"purity_loss": s1.get("all:purity:regret"), "coverage_loss": s1.get("all:coverage:regret"),
                       "no_target_selected": s1.get("all:no_target:selected"),
                       "purity_tol": pre["purity_tol"], "coverage_tol": pre["coverage_tol"],
                       "pass_purity": (None if s1.get("all:purity:regret") is None
                                       else s1["all:purity:regret"] <= pre["purity_tol"]),
                       "pass_coverage": (None if s1.get("all:coverage:regret") is None
                                         else s1["all:coverage:regret"] <= pre["coverage_tol"])},
    }


def public_windows(res: Mapping) -> dict:
    from AG3S.ag3s.experiments.studies import t56_study as st

    return {c: {g: {"full": st.public(r["full"]),
                    **{w: {sc: st.public(r[w][sc]) for sc in SCOPES} for w in MASK_WINDOWS}}
                for g, r in grp.items()} for c, grp in res.items()}


def criteria_windows(res: Mapping, pre: Mapping, primary: str) -> dict:
    pc = primary if primary in res else next(iter(res))
    out = {}
    for g, r in res[pc].items():
        out[g] = {"full": criteria(r["full"], pre),
                  **{w: {sc: criteria(r[w][sc], pre) for sc in SCOPES} for w in MASK_WINDOWS}}
    return out


# ------------------------------------------------------------------------------ parity / tables


def canonical(x: Any) -> Any:
    """What `t56_study.write_json` stores (NaN kept, inf -> strings), read back."""
    from AG3S.ag3s.experiments.studies import t56_study as st

    return json.loads(json.dumps(st.jsonable(x), allow_nan=True))


def first_difference(a: Any, b: Any, path: str = "") -> Optional[str]:
    """NaN-aware structural comparison; None when equal."""
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
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return None
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def parity(pub: Mapping, crit: Mapping, summary: Mapping, primary: str) -> dict:
    out = {"groups": {}, "criteria": None, "exact": True}
    for c, grp in pub.items():
        for g, r in grp.items():
            ref = summary.get("results", {}).get(c, {}).get(g, {}).get("all")
            d = "missing in t56_summary.json" if ref is None else first_difference(canonical(r["full"]), ref)
            out["groups"][f"{c}:{g}"] = {"exact": d is None, "first_difference": d}
            out["exact"] &= d is None
    mine = {g: v["full"] for g, v in crit.items()}
    d = first_difference(canonical(mine), summary.get("criteria_mechanical"))
    out["criteria"] = {"exact": d is None, "first_difference": d, "condition": primary}
    out["exact"] &= d is None
    return out


def table_rows(pub: Mapping) -> list[dict]:
    rows = []
    for c, grp in pub.items():
        for g, r in grp.items():
            cells = [("full", "-", r["full"])] + [(w, sc, r[w][sc]) for w in MASK_WINDOWS for sc in SCOPES]
            for w, sc, res in cells:
                for v in ("S0", "S1", "S2"):
                    s = res.get(v)
                    if s is None:
                        continue
                    row = {"condition": c, "group": g, "variant": v, "window": w, "scope": sc,
                           "selected": s.get("selected"), "tie_size": s.get("tie_size"),
                           "n_event_episodes": s.get("n_event_episodes"),
                           "tau_b_identity": s.get("tau_b:identity", {}).get("mean_defined"),
                           "tau_b_identity_n_undefined": s.get("tau_b:identity", {}).get("n_undefined")}
                    for k in TABLE_KEYS:
                        row[k] = s.get(k)
                        ci = s.get(k + ":ci95") or [None, None]
                        row[k + ":lo"], row[k + ":hi"] = ci[0], ci[1]
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


# ------------------------------------------------------------------------------ main


def md5_of(path) -> str:
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def pinned_md5() -> dict:
    here = pathlib.Path(__file__).resolve().parent
    return {n: md5_of(here / n) for n in PINNED}


def stage_args(summary: Mapping, t56_run: str):
    """The T56 test stage's own argument namespace (from the summary's ``args``), re-parsed and configured."""
    from AG3S.ag3s.experiments.studies import t56_study as st

    a = summary["args"]
    argv = ["--run-dir", t56_run, "--set", "test", "--stage", "test", "--t53-run", a["t53_run"],
            "--t51-run", a["t51_run"], "--t49-root", a["t49_root"], "--t47-verify", a["t47_verify"],
            "--cell-set", a["cell_set"], "--kinds", *a["kinds"], "--conditions", *a["conditions"]]
    if a.get("t54_run"):
        argv += ["--t54-run", a["t54_run"]]
    if a.get("records"):
        argv += ["--records", *a["records"]]
    if a.get("cells"):
        argv += ["--cells", *a["cells"]]
    args = st.build_parser().parse_args(argv)
    st.configure(args)
    return args


def out_dir(run_dir: Optional[str], forbid: Sequence[str]) -> pathlib.Path:
    from common.run_output import resolve_output_path

    run = run_dir or os.environ.get("AG3S_RUN_DIR")
    if not run:
        raise SystemExit('AG3S_RUN_DIR is not set — export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python '
                         '-m common.run_output)" (a new run)')
    for f in forbid:
        if f and pathlib.Path(run).resolve() == pathlib.Path(f).resolve():
            raise SystemExit(f"output run {run} is an input run (read-only)")
    return resolve_output_path(OUT_SUB, run_dir=run, default_name=OUT_SUB)


def run(a) -> dict:
    from AG3S.ag3s.experiments.studies import t55_windows as w55
    from AG3S.ag3s.experiments.studies import t56_select as sel_mod
    from AG3S.ag3s.experiments.studies import t56_selfscore as ss
    from AG3S.ag3s.experiments.studies import t56_study as st

    t0 = time.time()
    sp = pathlib.Path(a.t56_run) / SUMMARY_REL
    if not sp.is_file():
        raise SystemExit(f"no T56 test summary {sp} — run the T56 test stage first")
    summary = json.loads(sp.read_text())
    md5_in = {"t56_summary": md5_of(sp)}
    code_md5 = pinned_md5()
    args = stage_args(summary, a.t56_run)
    root = pathlib.Path(a.t56_run) / "verify" / "T56"
    pre_p = pathlib.Path(summary["prereg_path"])
    pre = st.read_prereg(pre_p)
    if pre["sha256"] != summary["prereg_sha256"]:
        raise SystemExit("the prereg differs from the one the T56 test stage read")
    md5_in["prereg"] = md5_of(pre_p)
    od = out_dir(a.out_run, [a.t56_run, args.t53_run, args.t54_run, args.t51_run,
                             str(pathlib.Path(a.t60_step1).parents[2])])
    p1, p2 = ss.S1Params(**pre["s1"]), ss.S2Params(**pre["s2"])
    names = {"cell": list(pre["candidates"]["cell"]), "arm": list(pre["candidates"]["arm"])}
    keys = list(summary["keys"])
    kinds, conds = list(args.kinds), list(args.conditions)
    rds = [st.load_record(args, root, k, names, conds, kinds) for k in keys]
    masks = w55.load_masks(pathlib.Path(a.t60_step1), keys)
    positions = {rd.key: record_positions(rd, st.gt_frames(args, rd.key)["t_step"], masks[rd.key]) for rd in rds}
    W = sel_mod.episode_draws(len(rds), int(pre["bootstrap"]["B"]), int(pre["bootstrap"]["seed"]))
    res = analyze_windows(rds, positions, conds, kinds, float(pre["resolution"]), W,
                          fs_fn=lambda rd, c, k: st.score_record(rd, c, k, p1, p2))
    pub = public_windows(res)
    crit = criteria_windows(res, pre, st.PRIMARY_CONDITION)
    par = parity(pub, crit, summary, st.PRIMARY_CONDITION)
    print(f"[{STEP}] parity full vs t56_summary.json: {par['exact']}")
    meta = {"step": STEP, "version": VERSION, "created": st.kst_now(), "git": st.git_state(), "argv": sys.argv,
            "inputs": {"t56_summary": str(sp), "prereg": str(pre_p), "prereg_sha256": pre["sha256"], "md5": md5_in,
                       "t60_step1": str(a.t60_step1), "test_args": summary["args"]},
            "code_md5_start": code_md5, "mask_rule": MASK_RULE, "definitions": __doc__,
            "user_decision": ("main verdict = full window; P / H auxiliary, side by side; mask fixed before "
                              "analysis, identical for every candidate / variant; the primary window is never "
                              "swapped"),
            "parity": par}
    if not par["exact"]:
        st.write_json(od / "t56_windows.json", {**meta, "status": "STOP: full-window parity failed — no P / H output"})
        raise SystemExit(2)
    endpoints = {k: {"t_P": m["t_P"], "t_H": m["t_H"], "mask": m["path"], "mask_md5": m["md5"],
                     **{f"positions_kept_{w}": len(positions[k][w]) for w in MASK_WINDOWS}}
                 for k, m in masks.items()}
    out = {**meta, "status": "ok", "keys": keys, "endpoints": endpoints, "results": pub,
           "criteria_mechanical": crit, "primary_condition": st.PRIMARY_CONDITION}
    write_csv(od / "windows.csv", table_rows(pub))
    out["code_md5_end"] = pinned_md5()
    out["inputs_unchanged"] = (md5_of(sp) == md5_in["t56_summary"] and md5_of(pre_p) == md5_in["prereg"]
                               and out["code_md5_end"] == code_md5)
    out["seconds"] = time.time() - t0
    st.write_json(od / "t56_windows.json", out)
    print(f"[{STEP}] -> {od} ({out['seconds']:.1f}s; inputs unchanged {out['inputs_unchanged']})")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--t56-run", default=DEFAULT_T56_RUN, help="the T56 run whose test stage finished (read only)")
    ap.add_argument("--t60-step1", default=DEFAULT_T60_STEP1, help="T60 verify/T60/step1 (masks/, read only)")
    ap.add_argument("--out-run", default=None, help="new run dir (default $AG3S_RUN_DIR)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> dict:
    return run(build_parser().parse_args(argv))


__all__ = ["WINDOWS", "SCOPES", "analyze_windows", "criteria", "criteria_windows", "first_difference", "parity",
           "public_windows", "record_positions", "table_rows", "window_positions", "with_positions"]


if __name__ == "__main__":
    main()
