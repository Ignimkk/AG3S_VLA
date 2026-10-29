"""T56 CLI — does a self score predict GT quality well enough to pick a cell / a T54 arm?

`T56.task.md` (사전 고정 1–5 are the user's; nothing here changes them). Modules: candidates
`t56_cells`, replay `t56_replay`, HELD event `t56_held`, self scores `t56_selfscore`, selection / regret /
τ-b / bootstrap `t56_select`. Every output goes to ``<--run-dir>/verify/T56/`` (a new development run;
never the T53 / T54 run, whose outputs are read only).

Stages::

    inputs   CPU, seconds. Locate + pair the inputs of the set; the cell candidate list.
             → <out>/<set>/inputs.json (exit != 0 on a problem)
    replay   CPU, MuJoCo osmesa, <= 4 workers; one job per (record, condition, kind, batch).
             → <out>/<set>/replay/<key>__<cond>__<kind>_bNN.{jsonl,npz,meta.json} (meta = done)
    held     CPU, seconds. HELD event per record: test = T53 server chunks; dev = offline grasp_latch on
             the recorded execution facts (`t56_held`). → <out>/<set>/held/<key>.json
    dev      dev set only. S1 / S2 parameter grid on dev, the pre-declared choice rule, and the dev
             numbers → <out>/T56.prereg.json (+ <out>/dev/dev_summary.json). **Stop here**: the lead
             gets the user's confirmation before the test stage.
    test     test set only. Reads T56.prereg.json **only** (no parameter flag is accepted) → regret
             (two kinds), tie rate / tie-set size, τ-b (undefined counted), auxiliary losses, phase split,
             paired bootstrap, HELD-event label error → <out>/test/analysis/t56_summary.json

Sets: ``dev`` = t2 · headfix · ep1807 · ep1850 · ep1900 (T47) + ep0 · ep500 · ep1200 (T51 calibration),
profile t52 (`long` and `ep1800` are excluded — recorded before the X2 fix with head_1 = 0.0, cam_high
faces the wall; lead update 2026-10-13); ``test`` = the 16 T53 episodes, profile c1a.

Example (verifier)::

    R=<new run dir>; M=AG3S.ag3s.experiments.studies.t56_study
    export MUJOCO_GL=osmesa OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$PWD
    .venv-ag3s/bin/python -m $M --run-dir $R --set dev  --stage inputs
    .venv-ag3s/bin/python -m $M --run-dir $R --set dev  --stage replay --workers 4
    .venv-ag3s/bin/python -m $M --run-dir $R --set dev  --stage held
    .venv-ag3s/bin/python -m $M --run-dir $R --set dev  --stage dev       # → T56.prereg.json, STOP
    .venv-ag3s/bin/python -m $M --run-dir $R --set test --stage inputs
    .venv-ag3s/bin/python -m $M --run-dir $R --set test --stage replay --workers 4
    .venv-ag3s/bin/python -m $M --run-dir $R --set test --stage held
    .venv-ag3s/bin/python -m $M --run-dir $R --set test --stage test [--t54-run <T54 run>]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t56_cells as cells_mod
from AG3S.ag3s.experiments.studies import t56_select as sel_mod
from AG3S.ag3s.experiments.studies import t56_selfscore as ss

STEP = "T56"
MAX_WORKERS = 4
SETS = ("dev", "test")
STAGES = ("inputs", "replay", "held", "dev", "test")
KINDS = ("cell", "arm")
CONDITIONS = ("current", "hold")
PRIMARY_CONDITION = "current"
#: Dev records (`T56.task.md` 사전 고정 5, lead update 2026-10-13: `long`, `ep1800` excluded).
DEV_KEYS = ("t2", "headfix", "ep1807", "ep1850", "ep1900", "ep0", "ep500", "ep1200")
DEV_EXCLUDED = {"long": "recorded before the X2 fix (head_1 = 0.0, cam_high faces the wall)",
                "ep1800": "recorded before the X2 fix (head_1 = 0.0, cam_high faces the wall)"}
PHASES = ("pre", "held", "post", "placed")
_DEV = "/mnt/dev/AG3S_VLA-storage/results/development"
DEFAULTS = {
    "t53_run": f"{_DEV}/20261013T100210+0900_4a540f1a18d7_cb037c425d2f",
    "t51_run": cells_mod.T51_RUN,
    "t49_root": f"{_DEV}/20261012T190547+0900_28b981600bf7_c39a577332fe/verify/T49",
    "t47_verify": f"{_DEV}/20261012T141412+0900_bd778e5e3453_d7a51b456f36/verify/T47",
}
#: Dev-stage defaults (practical starting values; the dev stage writes what it used into the prereg).
DEV_DEFAULTS = {"resolution": 0.01, "regret_tol": 0.03, "purity_tol": 0.05, "coverage_tol": 0.05}
TIE_RESOLUTIONS_REPORTED = (0.0, 0.005, 0.01, 0.02)
#: The S1 grid, in tie-break order (first = simplest / T51-like).
S1_GRID = [ss.S1Params(margin=m, combine=c, normalize=n)
           for m in (0.02, 0.01, 0.03) for c in ("harmonic", "geometric", "min", "mean", "product")
           for n in ("none", "frame_max")]
S2_GRID = [ss.S2Params(comove_tol=t, min_hand_motion=mm, reference=r)
           for t in (0.01, 0.005, 0.015) for mm in (0.03, 0.05) for r in ("held", "previous")]
PREREG_NAME = "T56.prereg.json"
#: Parameter flags only the dev stage may take (the test stage reads the prereg only).
DEV_ONLY_FLAGS = ("resolution", "regret_tol", "purity_tol", "coverage_tol", "s1", "s2")
T54_PARITY_FIELDS = ("identity_correct", "has_target", "chosen_cluster_purity", "coverage", "n_cloud",
                     "chosen_n", "post_split_digest", "chosen_rank")


# ------------------------------------------------------------------------------ plumbing


def kst_now() -> str:
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat()


def jsonable(x):
    from AG3S.ag3s.experiments.studies.t47b_metrics import jsonable as jb

    return jb(x)


def write_json(path: pathlib.Path, payload: Any) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(jsonable(payload), indent=1, allow_nan=True))
    tmp.replace(path)


def out_root(run_dir: str, protected: Sequence[Optional[str]] = ()) -> pathlib.Path:
    from common.run_output import resolve_output_path

    try:
        root = resolve_output_path("verify/T56", run_dir=run_dir, default_name="verify/T56")
    except ValueError as exc:
        raise SystemExit(f"--run-dir {run_dir}: {exc}") from exc
    run = pathlib.Path(run_dir).resolve()
    for p in protected:
        if p and run == pathlib.Path(p).resolve():
            raise SystemExit(f"--run-dir must be a new run, not an input run ({p})")
    return root


def git_state() -> dict:
    from AG3S.ag3s.experiments.studies.t47_head_selection import git_state as gs

    return gs()


def run_info(stage: str, args) -> dict:
    return {"step": STEP, "stage": stage, "set": args.set, "argv": sys.argv,
            "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
            "started_kst": kst_now(), "git": git_state(), "python": sys.version.split()[0]}


# --------------------------------------------------------------------------------- inputs


def configure(args) -> None:
    from AG3S.ag3s.experiments.studies import t51_records as rc

    rc.configure(pathlib.Path(args.t51_run) / "verify" / "T51")
    if args.set == "test":
        from AG3S.ag3s.experiments.studies import t54_study as st54

        args._t53_cfg = st54.configure_t53(args.t53_run)
    else:
        args._t53_cfg = None


def resolve_keys(args) -> list[str]:
    if args.set == "dev":
        keys = list(args.records or DEV_KEYS)
        bad = [k for k in keys if k in DEV_EXCLUDED]
        if bad:
            raise SystemExit(f"dev records {bad} are excluded from T56: {DEV_EXCLUDED}")
        from AG3S.ag3s.experiments.studies import t51_records as rc

        unknown = [k for k in keys if k not in rc.ALL_KEYS]
        if unknown:
            raise SystemExit(f"unknown dev record(s) {unknown}")
        return keys
    from AG3S.ag3s.experiments.studies import t53_records as rec

    return rec.resolve_keys(args.records)


def record_inputs(set_: str, key: str, args) -> dict:
    """Paths + pairing of one record (no render)."""
    from AG3S.ag3s.experiments.sources.policy_record import load_run
    from AG3S.ag3s.experiments.studies import t54_study as st54
    from AG3S.ag3s.experiments.studies.t49_noun_query_extract import load_extract

    if set_ == "test":
        ns = argparse.Namespace(arms=None, server_roots=None, _t53_cfg=args._t53_cfg)
        info = st54.record_inputs("test", key, ns)
        from AG3S.ag3s.experiments.studies import t53_records as rec

        info["gt"] = str(rec.gt_cache_path(key))
        return info
    from AG3S.ag3s.experiments.studies import t51_records as rc
    from AG3S.ag3s.experiments.studies import t51b_noun_extract as ne

    info: dict[str, Any] = {"key": key, "set": set_, "problems": [], "profile": "t52",
                            "server_proc": None, "chunks": None}
    record, npz = rc.record_path(key), rc.npz_path(key)
    noun = (ne.extract_path(pathlib.Path(args.t51_run) / "verify" / "T51b" / "noun_extract", key)
            if rc.is_extra(key) else ne.t49_extract_path(pathlib.Path(args.t49_root), key))
    gt = rc.gt_cache_path(key, pathlib.Path(args.t47_verify))
    run = load_run(record)
    t_rec = [int(s.t_step) for s in run.steps]
    for p, what in ((npz, "action npz"), (noun, "noun extract"), (gt, "GT cache")):
        if not pathlib.Path(p).is_file():
            info["problems"].append(f"no {what} {p}")
    if pathlib.Path(npz).is_file():
        _, _, _, att_t = st54.load_action(pathlib.Path(npz))
        if att_t is not None and att_t != t_rec:
            info["problems"].append("action npz t_step != record t_step")
    if pathlib.Path(noun).is_file():
        ex = load_extract(pathlib.Path(noun))
        if [int(t) for t in ex["t_step"]] != t_rec:
            info["problems"].append("noun extract t_step != record t_step")
        if not bool(np.asarray(ex["ok"], bool).all()):
            info["problems"].append("noun extract has failed frames")
    if pathlib.Path(gt).is_file():
        with np.load(gt, allow_pickle=False) as z:
            if [int(t) for t in z["t_step"]] != t_rec:
                info["problems"].append("GT cache t_step != record t_step")
    info.update({"record": str(record), "npz": str(npz), "noun": str(noun), "gt": str(gt),
                 "target": rc.target_of(key), "n_frames": len(t_rec), "prompt": run.prompt})
    return info


def candidates(args) -> dict:
    cc = cells_mod.cell_candidates(args.t51_run, args.cell_set)
    cells = list(args.cells) if args.cells else cc["cells"]
    unknown = [c for c in cells if c not in cc["cells"]] if args.cells else []
    return {"cell": sorted(cells), "arm": sorted(cells_mod.arm_names()), "cell_source": cc,
            "cells_restricted": bool(args.cells), "cells_not_in_t51_sets": unknown}


def stage_inputs(args, root: pathlib.Path) -> dict:
    keys = resolve_keys(args)
    recs, probs = {}, []
    for k in keys:
        info = record_inputs(args.set, k, args)
        info.pop("chunks", None)
        recs[k] = info
        probs += [f"[{k}] {p}" for p in info["problems"]]
    payload = {**run_info("inputs", args), "records": recs, "candidates": candidates(args),
               "dev_excluded": DEV_EXCLUDED, "problems": probs, "all_ok": not probs}
    write_json(root / args.set / "inputs.json", payload)
    print(json.dumps({"records": keys, "n_cells": len(payload["candidates"]["cell"]),
                      "n_arms": len(payload["candidates"]["arm"]), "problems": probs}, indent=1))
    if probs:
        raise SystemExit("inputs: problems:\n  " + "\n  ".join(probs))
    return payload


# --------------------------------------------------------------------------------- replay


def job_stem(key: str, cond: str, kind: str, b: int) -> str:
    return f"{key}__{cond}__{kind}_b{b:02d}"


def replay_dir(root: pathlib.Path, set_: str) -> pathlib.Path:
    return root / set_ / "replay"


def _job(spec: dict) -> dict:
    """One (record, condition, kind, batch) replay in a spawned worker."""
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import resource

    from AG3S.ag3s.experiments.studies import t51_records as rc
    from AG3S.ag3s.experiments.studies import t54_replay as rp
    from AG3S.ag3s.experiments.studies import t54_study as st54
    from AG3S.ag3s.experiments.studies import t56_replay as r56
    from AG3S.ag3s.experiments.studies.t49_noun_query_extract import load_extract

    rc.configure(spec["t51_extra"])
    if spec["set"] == "test":
        from AG3S.ag3s.experiments.studies import t53_records as rec

        rec.configure(run_dir=spec["t53_run"], episodes_root=spec["episodes_root"])
    stem = pathlib.Path(spec["stem"])
    log_p = stem.with_suffix(".log")

    def log(msg: str) -> None:
        with open(log_p, "a") as fh:
            fh.write(msg + "\n")

    t0 = time.time()
    names = spec["names"]
    action = st54.load_action(pathlib.Path(spec["npz"]))
    noun_ex = (load_extract(pathlib.Path(spec["noun"])) if cells_mod.needs_noun(spec["kind"], names)
               else None)
    specs, maps = cells_mod.candidate_maps(spec["kind"], names, action=action, noun_ex=noun_ex)
    del action
    build, prov = rp.make_builder(spec["profile"], pathlib.Path(spec["record"]),
                                  server_proc=(None if spec["server_proc"] is None
                                               else pathlib.Path(spec["server_proc"])))
    freeze = None
    if spec["condition"] == "hold" and spec.get("chunks"):
        freeze = rp.freeze_from_chunks([pathlib.Path(c) for c in spec["chunks"]])
    rows, arrays = r56.run_job(pathlib.Path(spec["record"]), target=spec["target"], kind=spec["kind"],
                               names=names, maps=maps, condition=spec["condition"], build=build,
                               freeze=freeze, frames=spec.get("frames"), log=log, specs=specs)
    r56.save_npz(stem.with_suffix(".npz"), arrays)
    tmp = stem.with_suffix(".partial")
    with open(tmp, "w") as fh:
        for s in specs:
            for r in rows[s.name]:
                fh.write(json.dumps(jsonable({"record": spec["key"], **r}), allow_nan=True) + "\n")
    tmp.replace(stem.with_suffix(".jsonl"))
    meta = {"key": spec["key"], "condition": spec["condition"], "kind": spec["kind"], "names": names,
            "profile": prov, "freeze_injected": freeze is not None,
            "freeze_summary": None if freeze is None else rp.freeze_summary(freeze),
            "frames": spec.get("frames"), "n_rows": int(sum(len(v) for v in rows.values())),
            "seconds": time.time() - t0,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "finished_kst": kst_now()}
    write_json(stem.with_suffix(".meta.json"), meta)
    return {"stem": stem.name, "seconds": meta["seconds"], "peak_rss_mb": meta["peak_rss_mb"]}


def plan_jobs(args, root: pathlib.Path) -> tuple[list[dict], list[str]]:
    keys = resolve_keys(args)
    cand = candidates(args)
    rdir = replay_dir(root, args.set)
    rdir.mkdir(parents=True, exist_ok=True)
    specs, skipped = [], []
    for k in keys:
        info = record_inputs(args.set, k, args)
        if info["problems"]:
            raise SystemExit(f"[{k}] input problems (run --stage inputs): {info['problems']}")
        for cond in args.conditions:
            for kind in args.kinds:
                size = args.batch if kind == "cell" else len(cand["arm"])
                for b, names in enumerate(cells_mod.batches(cand[kind], size)):
                    stem = rdir / job_stem(k, cond, kind, b)
                    if stem.with_suffix(".meta.json").is_file() and not args.force:
                        meta = json.loads(stem.with_suffix(".meta.json").read_text())
                        if list(meta.get("names") or []) != list(names) or meta.get("frames") != args.frames:
                            raise SystemExit(f"{stem.name}: done with other candidates / frames "
                                             f"({meta.get('names')}, {meta.get('frames')}) — use a new run "
                                             "dir or --force")
                        skipped.append(stem.name)
                        continue
                    for suf in (".jsonl", ".npz", ".log", ".partial", ".meta.json"):
                        if stem.with_suffix(suf).exists():
                            stem.with_suffix(suf).unlink()
                    specs.append({"set": args.set, "key": k, "condition": cond, "kind": kind,
                                  "names": names, "stem": str(stem), "npz": info["npz"],
                                  "noun": info["noun"], "record": info["record"],
                                  "target": info["target"], "profile": info["profile"],
                                  "server_proc": info.get("server_proc"),
                                  "chunks": info.get("chunks"), "frames": args.frames,
                                  "t51_extra": str(pathlib.Path(args.t51_run) / "verify" / "T51"),
                                  "t53_run": args.t53_run,
                                  "episodes_root": (args._t53_cfg or {}).get("episodes_root")})
    return specs, skipped


def stage_replay(args, root: pathlib.Path) -> dict:
    import concurrent.futures as cf
    import multiprocessing as mp

    if not 1 <= args.workers <= MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS}")
    specs, skipped = plan_jobs(args, root)
    print(f"replay: {len(specs)} job(s), {len(skipped)} done already, {args.workers} worker(s)",
          flush=True)
    results, failures = [], []
    t0 = time.time()
    if specs:
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_job, s): s for s in specs}
            for fut in cf.as_completed(futs):
                s = futs[fut]
                try:
                    r = fut.result()
                    results.append(r)
                    print(f"  done {r['stem']} {r['seconds']:.0f}s peak {r['peak_rss_mb']:.0f} MB "
                          f"({time.time() - t0:.0f}s)", flush=True)
                except Exception as exc:  # noqa: BLE001 — recorded; other jobs continue
                    failures.append({"stem": pathlib.Path(s["stem"]).name,
                                     "error": f"{type(exc).__name__}: {exc}"})
                    print(f"  FAILED {pathlib.Path(s['stem']).name}: {exc}", flush=True)
    payload = {**run_info("replay", args), "jobs": results, "failures": failures,
               "skipped_done": skipped, "seconds": time.time() - t0, "finished_kst": kst_now()}
    write_json(root / args.set / f"replay_{int(time.time())}.json", payload)
    if failures:
        raise SystemExit(f"replay: {len(failures)} job(s) failed")
    return payload


def load_kind(root: pathlib.Path, set_: str, key: str, cond: str, kind: str, names: Sequence[str]):
    """``(Geometry merged over batches, rows by candidate)``; refuses a missing / partial job."""
    from AG3S.ag3s.experiments.studies import t56_replay as r56

    stems = sorted(replay_dir(root, set_).glob(f"{key}__{cond}__{kind}_b*.meta.json"))
    if not stems:
        raise SystemExit(f"missing replay {key} {cond} {kind} (run --stage replay)")
    parts, rows = [], {}
    for m in stems:
        base = m.with_name(m.name[:-len(".meta.json")])
        parts.append(r56.load_npz(base.with_suffix(".npz")))
        for line in open(base.with_suffix(".jsonl")):
            r = json.loads(line)
            rows.setdefault(r["arm"], []).append(r)
    geom = r56.merge(parts)
    missing = [n for n in names if n not in geom.cpos]
    if missing:
        raise SystemExit(f"{key} {cond} {kind}: no replay for {missing}")
    return geom, {k: sorted(v, key=lambda r: r["frame"]) for k, v in rows.items()}


# ----------------------------------------------------------------------------------- held


def held_path(root: pathlib.Path, set_: str, key: str) -> pathlib.Path:
    return root / set_ / "held" / f"{key}.json"


def offline_config(args) -> "Any":
    from AG3S.ag3s.experiments.studies.t56_held import OfflineLatchConfig

    return OfflineLatchConfig(reach_source=args.reach_source, reference_candidate=args.reach_candidate,
                              reference_condition=args.reach_condition,
                              slip_detach=not args.no_slip_detach)


def held_for(args, root: pathlib.Path, key: str, cfg=None):
    from AG3S.ag3s.experiments.studies import t56_held as hd

    if args.set == "test":
        info = record_inputs("test", key, args)
        chunks = info.get("chunks") or []
        if not chunks:
            return hd.HeldEvent("server_chunk", None, None, [], details={"note": "no server chunks"})
        if len(chunks) != info["n_frames"]:
            raise SystemExit(f"[{key}] {len(chunks)} chunks for {info['n_frames']} frames")
        return hd.from_chunks([pathlib.Path(c) for c in chunks])
    cfg = cfg or offline_config(args)
    ref = cfg.reference_candidate
    kind = "cell" if ref in candidates(args)["cell"] else "arm"
    conds = [cfg.reference_condition] + [c for c in CONDITIONS if c != cfg.reference_condition]
    geom, used = None, None
    for c in conds:
        try:
            geom, _ = load_kind(root, args.set, key, c, kind, [ref])
            used = c
            break
        except SystemExit:
            continue
    if geom is None:
        raise SystemExit(f"[{key}] offline latch: no replay of the reference candidate {ref}")
    manip = hd.reference_centroids(geom, ref) if cfg.reach_source == "reference_candidate" else \
        [None] * len(geom.frames)
    ev = hd.offline_latch(t_step=geom.z["t_step"], state=geom.z["state"], actions=geom.z["actions"],
                          parent_world=geom.z["parent_world"], manip_world=manip, cfg=cfg,
                          frames=geom.frames)
    ev.details["reference_condition_used"] = used
    return ev


def stage_held(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t56_held as hd

    out = {}
    for k in resolve_keys(args):
        ev = held_for(args, root, k)
        p = held_path(root, args.set, k)
        p.parent.mkdir(parents=True, exist_ok=True)
        hd.write_event(p, ev)
        out[k] = {"source": ev.source, "hand": ev.hand, "frame": ev.pos, "exit": ev.exit_pos,
                  "exit_reason": ev.exit_reason}
    write_json(root / args.set / "held" / "held_summary.json",
               {**run_info("held", args), "records": out,
                "n_with_event": int(sum(v["frame"] is not None for v in out.values())),
                "n_without_event": int(sum(v["frame"] is None for v in out.values())),
                "records_without_event": sorted(k for k, v in out.items() if v["frame"] is None),
                "source_counts": {src: int(sum(v["source"] == src for v in out.values()))
                                  for src in sorted({v["source"] for v in out.values()})}})
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------- GT values


def gt_frames(args, key: str) -> dict:
    """``{t_step, has_target (F,), phase (F,) or None}`` from the GT cache (+ T53 grasp npz)."""
    from AG3S.ag3s.experiments.studies import t54_stats as st

    if args.set == "test":
        from AG3S.ag3s.experiments.studies import t53_records as rec

        g = rec.load_gt(key)
        with np.load(rec.grasp_path(key), allow_pickle=False) as z:
            phase = [str(s) for s in z["grasp_state"]]
            if [int(t) for t in z["t_step"]] != [int(t) for t in g["t_step"]]:
                raise SystemExit(f"[{key}] grasp t_step differs from the GT cache")
    else:
        from AG3S.ag3s.experiments.studies import t51_records as rc

        g = rc.load_gt(pathlib.Path(args.t47_verify), key)
        phase = None
    return {"t_step": [int(t) for t in g["t_step"]],
            "has_target": st.has_target_frames(g["class_px"], st.MIN_TARGET_PX), "phase": phase}


def gt_matrix(rows: dict, names: Sequence[str], geom, gtf: dict) -> dict:
    """Per metric ``(n_cand, n_pos)`` frame values on target frames (NaN elsewhere)."""
    from AG3S.ag3s.experiments.studies import t54_stats as st

    n_c, n_f = len(names), len(geom.frames)
    out = {m: np.full((n_c, n_f), np.nan) for m in sel_mod.GT_METRICS}
    for a, nm in enumerate(names):
        for r in rows[nm]:
            j = geom.fpos[int(r["frame"])]
            i = int(r["frame"])
            if int(r["t_step"]) != gtf["t_step"][i]:
                raise SystemExit(f"row frame {i} t_step {r['t_step']} != GT {gtf['t_step'][i]}")
            if not gtf["has_target"][i]:
                continue
            v = st.frame_values(r)
            for m in sel_mod.GT_METRICS:
                out[m][a, j] = v[m]
    return out


def t54_rows(t54_run: str, key: str, cond: str) -> dict:
    p = pathlib.Path(t54_run) / "verify" / "T54" / "test" / "rows" / f"{key}__{cond}.jsonl"
    if not p.with_suffix(".meta.json").is_file():
        raise SystemExit(f"T54 rows {p} missing or incomplete")
    by: dict[str, list] = {}
    for line in open(p):
        r = json.loads(line)
        by.setdefault(r["arm"], []).append(r)
    return {k: sorted(v, key=lambda r: r["frame"]) for k, v in by.items()}


def _same(a, b) -> bool:
    if a == b:
        return True
    try:
        return math.isnan(float(a)) and math.isnan(float(b))
    except (TypeError, ValueError):
        return False


def t54_parity(mine: dict, ref: dict, fields: Sequence[str] = T54_PARITY_FIELDS) -> dict:
    n, mism = 0, []
    for arm, rs in mine.items():
        rb = {int(r["frame"]): r for r in ref.get(arm, [])}
        for r in rs:
            q = rb.get(int(r["frame"]))
            if q is None:
                mism.append({"arm": arm, "frame": int(r["frame"]), "field": "<missing in T54>"})
                continue
            for f in fields:
                if f in q:
                    n += 1
                    if not _same(q[f], r.get(f)):
                        mism.append({"arm": arm, "frame": int(r["frame"]), "field": f,
                                     "t54": q[f], "t56": r.get(f)})
    return {"fields_compared": n, "n_mismatch": len(mism), "mismatch": mism[:20]}


# ------------------------------------------------------------------------------ record data


@dataclasses.dataclass
class RecordData:
    key: str
    held: Any
    geom: dict  # (cond, kind) -> Geometry
    gt: dict  # (cond, kind) -> {metric: (n_c, n_f)}
    phase_pos: Optional[dict]  # phase -> positions (per (cond, kind) geometry frames are the same)
    names: dict  # kind -> names
    gt_source: dict  # (cond, kind) -> "t56_replay" | "t54_rows"
    parity: dict


def load_record(args, root: pathlib.Path, key: str, names: dict, conds: Sequence[str],
                kinds: Sequence[str]) -> RecordData:
    from AG3S.ag3s.experiments.studies import t56_held as hd

    hp = held_path(root, args.set, key)
    if not hp.is_file():
        raise SystemExit(f"[{key}] no HELD event file {hp} (run --stage held)")
    held = hd.read_event(hp)
    gtf = gt_frames(args, key)
    geom, gt, src, par = {}, {}, {}, {}
    phase_pos = None
    for c in conds:
        for k in kinds:
            g, rows = load_kind(root, args.set, key, c, k, names[k])
            geom[(c, k)] = g
            src[(c, k)] = "t56_replay"
            if k == "arm" and getattr(args, "t54_run", None) and args.set == "test":
                ref = t54_rows(args.t54_run, key, c)
                par[(c, k)] = t54_parity({n: rows[n] for n in names[k]}, ref)
                if par[(c, k)]["n_mismatch"]:
                    raise SystemExit(f"[{key} {c}] T56 arm replay differs from the T54 rows: "
                                     f"{par[(c, k)]['mismatch'][:5]}")
                rows = {n: ref[n] for n in names[k]}
                src[(c, k)] = "t54_rows"
            gt[(c, k)] = gt_matrix(rows, names[k], g, gtf)
            if gtf["phase"] is not None and phase_pos is None:
                phase_pos = {ph: [j for j, i in enumerate(g.frames) if gtf["phase"][i] == ph]
                             for ph in PHASES}
    return RecordData(key, held, geom, gt, phase_pos, names, src, par)


# ------------------------------------------------------------------------------ scoring


_CACHE: dict = {}


def score_record(rd: RecordData, cond: str, kind: str, p1: ss.S1Params, p2: ss.S2Params) -> dict:
    """`t56_selfscore.frame_scores` with the parts memoised per (record, condition, kind) and the
    parameters each part depends on (the dev grid re-combines them)."""
    g = rd.geom[(cond, kind)]
    cands = rd.names[kind]
    base = (rd.key, cond, kind, id(g))
    k0 = base + ("s0",)
    if k0 not in _CACHE:
        l0 = ss.s0_label(g)
        _CACHE[k0] = (l0, ss.s0_frame_scores(g, cands, l0))
    l0, s0 = _CACHE[k0]
    k1 = base + ("s1", p1.margin, p1.cover_tol)
    if k1 not in _CACHE:
        l1 = ss.s1_label(g, rd.held, p1)
        _CACHE[k1] = (l1,) + ss.s1_components(g, cands, l1, p1)
    l1, i1, c1 = _CACHE[k1]
    k2 = k1 + ("s2", tuple(sorted(p2.as_dict().items())))
    if k2 not in _CACHE:
        ext = ss.s2_extension(g, rd.held, l1, p2) if l1 is not None else dict(ss.EMPTY_EXT)
        _CACHE[k2] = (ext,) + ss.s2_components(g, cands, ext, p1, p2)
    ext, i2, c2 = _CACHE[k2]
    return ss.compose(l0=l0, s0=s0, l1=l1, i1=i1, c1=c1, ext=ext, i2=i2, c2=c2, p1=p1)


def episode_tables(rds: Sequence[RecordData], cond: str, kind: str, group: Sequence[str],
                   fs_by_key: dict, phase: Optional[str] = None, phase_self: bool = False) -> dict:
    """``self[v] (n_ep, n_c)``, ``tb``, ``gt[m]`` for one group (candidate subset of a kind)."""
    names_all = rds[0].names[kind]
    cols = [names_all.index(n) for n in group]
    n_ep, n_c = len(rds), len(group)
    selfm = {v: np.full((n_ep, n_c), np.nan) for v in ss.VARIANTS}
    tb = np.full((n_ep, n_c), np.nan)
    gt = {m: np.full((n_ep, n_c), np.nan) for m in sel_mod.GT_METRICS}
    for e, rd in enumerate(rds):
        pos = None
        if phase is not None:
            if rd.phase_pos is None:
                continue
            pos = rd.phase_pos[phase]
        ep = ss.episode_scores(fs_by_key[rd.key], positions=pos if phase_self else None)
        for v in ss.VARIANTS:
            selfm[v][e] = ep[v][cols]
        tb[e] = ep["S1_coverage"][cols]
        for m in sel_mod.GT_METRICS:
            mat = rd.gt[(cond, kind)][m][cols]
            if pos is not None:
                mask = np.zeros(mat.shape[1], bool)
                mask[pos] = True
                mat = np.where(mask[None, :], mat, np.nan)
            n = np.isfinite(mat).sum(axis=1)
            with np.errstate(invalid="ignore"):
                gt[m][e] = np.where(n > 0, np.nansum(np.where(np.isfinite(mat), mat, 0), axis=1)
                                    / np.maximum(n, 1), np.nan)
    return {"self": selfm, "tb": tb, "gt": gt}


def evaluate_group(tab: dict, group: Sequence[str], keys: Sequence[str], resolution: float,
                   W: Optional[np.ndarray]) -> dict:
    out = {}
    for v in ss.VARIANTS:
        pt = sel_mod.evaluate(tab["self"][v], tab["tb"], tab["gt"], group, resolution=resolution)
        rec: dict[str, Any] = {"_point": pt}
        if W is not None:
            bt = sel_mod.evaluate(tab["self"][v], tab["tb"], tab["gt"], group, resolution=resolution, W=W)
            rec["_boot"] = bt
            rec["summary"] = sel_mod.summarize(pt, bt, group)
        else:
            s = int(pt["selected"][0])
            rec["summary"] = {"selected": None if s < 0 else group[s],
                              "tie_size": int(pt["tie_size"][0]),
                              **{k: (None if not math.isfinite(float(x[0])) else float(x[0]))
                                 for k, x in pt.items() if ":" in k}}
        rec["summary"]["tie_set"] = sel_mod.tie_set(pt, group, resolution)
        rec["summary"]["n_event_episodes"] = int(np.isfinite(tab["self"][v]).any(axis=1).sum())
        rec["summary"]["episodes_without_event"] = [k for k, row in zip(keys, tab["self"][v])
                                                    if not np.isfinite(row).any()]
        rec["summary"]["episode_ties"] = sel_mod.episode_ties(tab["self"][v], resolution, keys)
        for m in ("identity", "purity", "coverage"):
            taus = sel_mod.tau_per_episode_status(tab["self"][v], tab["gt"][m])
            ts = sel_mod.tau_summary(taus, keys, W)
            if "_boot" in ts:
                rec.setdefault("_tau_boot", {})[m] = ts.pop("_boot")
                ts["mean_defined_ci95"] = sel_mod.ci(rec["_tau_boot"][m])
            rec["summary"][f"tau_b:{m}"] = ts
        out[v] = rec
    if W is not None:
        keys_p = [k for k in out["S1"]["_point"] if ":" in k]
        out["_paired"] = {}
        for a, b in (("S1", "S0"), ("S2", "S0"), ("S2", "S1")):
            d = sel_mod.paired(out[a]["_boot"], out[b]["_boot"], out[a]["_point"], out[b]["_point"], keys_p)
            ta, tb_ = out[a].get("_tau_boot", {}), out[b].get("_tau_boot", {})
            for m in ta:
                pa = out[a]["summary"][f"tau_b:{m}"]["mean_defined"]
                pb = out[b]["summary"][f"tau_b:{m}"]["mean_defined"]
                d[f"tau_b:{m}:mean_defined"] = {
                    "diff": None if pa is None or pb is None else pa - pb,
                    "ci95": sel_mod.ci(np.asarray(ta[m]) - np.asarray(tb_[m])),
                    "note": ("undefined on one side — compare by regret" if pa is None or pb is None
                             else "")}
            out["_paired"][f"{a}-{b}"] = d
    return out


def public(res: dict) -> dict:
    """Drop the arrays (keys starting with '_') for JSON."""
    out = {}
    for k, v in res.items():
        if k == "_paired":
            out["paired"] = v
        elif isinstance(v, dict):
            out[k] = v["summary"] if "summary" in v else public(v)
    return out


def group_names(kind: str, names: Sequence[str]) -> dict:
    if kind == "cell":
        return cells_mod.groups_of(names)
    return {cells_mod.ARM_GROUP: sorted(names)}


def analyze(rds: Sequence[RecordData], conds: Sequence[str], kinds: Sequence[str], p1: ss.S1Params,
            p2: ss.S2Params, resolution: float, W: Optional[np.ndarray], phases: bool) -> dict:
    keys = [rd.key for rd in rds]
    out: dict[str, Any] = {}
    for c in conds:
        out[c] = {}
        for k in kinds:
            fs = {rd.key: score_record(rd, c, k, p1, p2) for rd in rds}
            for gname, group in group_names(k, rds[0].names[k]).items():
                if not group:
                    continue
                tab = episode_tables(rds, c, k, group, fs)
                rec = {"all": evaluate_group(tab, group, keys, resolution, W)}
                if phases:
                    rec["phase_gt"] = {}
                    rec["phase_both"] = {}
                    for ph in PHASES:
                        t1 = episode_tables(rds, c, k, group, fs, phase=ph, phase_self=False)
                        rec["phase_gt"][ph] = evaluate_group(t1, group, keys, resolution, W)
                        t2 = episode_tables(rds, c, k, group, fs, phase=ph, phase_self=True)
                        rec["phase_both"][ph] = evaluate_group(t2, group, keys, resolution, W)
                out[c][gname] = rec
            out[c][f"_fs_{k}"] = fs
    return out


def label_checks(rds: Sequence[RecordData], cond: str, kind: str, p1: ss.S1Params, p2: ss.S2Params,
                 fs: dict, args) -> dict:
    """HELD-event (S1) and both_contact (S0) label GT checks per record, and the rates."""
    per, phys = {}, {}
    for rd in rds:
        g = rd.geom[(cond, kind)]
        f = fs[rd.key]
        l1 = f["labels"]["S1"]
        ext = ss.s2_extension(g, rd.held, l1, p2) if l1 is not None else None
        per[rd.key] = {"S1_held": ss.label_check(g, l1, ext), "S0_both_contact": ss.label_check(g, f["labels"]["S0"]),
                       "held_source": rd.held.source, "held_hand": rd.held.hand,
                       "held_frame": rd.held.pos, "held_exit": rd.held.exit_pos,
                       "held_exit_reason": rd.held.exit_reason,
                       "s2_extension": f["s2_extension"], "s2_frames": f["s2_frames"]}
        if args.set == "test" and rd.held.pos is not None:
            gtf = gt_frames(args, rd.key)
            per[rd.key]["phys_state_at_held"] = gtf["phase"][int(rd.held.pos)]
    with_ev = [k for k, v in per.items() if v["S1_held"].get("has_label")]
    wrong = [k for k in with_ev if not per[k]["S1_held"]["event_label_correct"]]
    kinds_ = {}
    for k in with_ev:
        vv = per[k]["S1_held"]["event_verdict"]
        kinds_[vv] = kinds_.get(vv, 0) + 1
    return {"per_record": per, "n_held_events": len(with_ev), "n_label_wrong": len(wrong),
            "held_label_error_rate": (len(wrong) / len(with_ev)) if with_ev else None,
            "wrong_records": wrong, "verdict_counts": kinds_,
            "records_without_held": [k for k in per if not per[k]["S1_held"].get("has_label")],
            "note": "HELD is a trigger, not a certification: these are label errors of the HELD label"}


def t55_label_checks(args, keys: Sequence[str]) -> Optional[dict]:
    """HELD label error on the T55 re-simulations of this set (`t56_held.resim_checks`), when
    ``--t55-run`` is given. Reported only; nothing is chosen with it."""
    if not getattr(args, "t55_run", None):
        return None
    from AG3S.ag3s.experiments.studies import t56_held as hd

    d = pathlib.Path(args.t55_run) / "verify" / "T55" / "resim" / args.set
    return hd.resim_checks(d, keys)


# --------------------------------------------------------------------------------- dev stage


def _objective(res: dict, cond: str) -> tuple:
    """Dev choice rule: smaller mean (over the 'cells' and 'arms' groups) of the all-episode identity
    tie-set-average regret; then larger mean τ-b (defined; None = −inf); grid order breaks the rest."""
    regs, taus = [], []
    for g in ("cells", cells_mod.ARM_GROUP):
        if g not in res[cond]:
            continue
        s = res[cond][g]["all"]
        regs.append(s["S1"]["summary"].get("all:identity:tie_regret"))
        t = s["S1"]["summary"]["tau_b:identity"]["mean_defined"]
        taus.append(-math.inf if t is None else t)
    regs = [math.inf if r is None else r for r in regs]
    return (float(np.mean(regs)) if regs else math.inf, -(float(np.mean(taus)) if taus else -math.inf))


def _objective_s2(res: dict, cond: str) -> tuple:
    regs, taus = [], []
    for g in ("cells", cells_mod.ARM_GROUP):
        if g not in res[cond]:
            continue
        s = res[cond][g]["all"]
        regs.append(s["S2"]["summary"].get("all:identity:tie_regret"))
        t = s["S2"]["summary"]["tau_b:identity"]["mean_defined"]
        taus.append(-math.inf if t is None else t)
    regs = [math.inf if r is None else r for r in regs]
    return (float(np.mean(regs)) if regs else math.inf, -(float(np.mean(taus)) if taus else -math.inf))


def _grid_row(res: dict, cond: str, variant: str) -> dict:
    row = {}
    for g in ("cells", "cells_action", "cells_noun", cells_mod.ARM_GROUP):
        if g in res[cond]:
            s = res[cond][g]["all"][variant]["summary"]
            row[g] = {"selected": s["selected"], "tie_size": s["tie_size"],
                      "regret": s.get("all:identity:regret"),
                      "tie_regret": s.get("all:identity:tie_regret"),
                      "tau_b_identity": s["tau_b:identity"]["mean_defined"],
                      "tau_b_undefined": s["tau_b:identity"]["n_undefined"]}
    return row


def search_space(fixed1, fixed2, args) -> dict:
    """What the dev stage searched (recorded in the prereg) and what it took as given."""
    def axes(grid, fields):
        return {f: sorted({getattr(q, f) for q in grid}, key=str) for f in fields}

    return {
        "s1": ("cli" if fixed1 else {"n": len(S1_GRID), **axes(S1_GRID, ("margin", "combine", "normalize",
                                                                         "cover_tol"))}),
        "s2": ("cli" if fixed2 else {"n": len(S2_GRID), **axes(S2_GRID, ("comove_tol", "min_hand_motion",
                                                                         "reference", "min_points",
                                                                         "stop_on_lost"))}),
        "given_not_searched": {f: getattr(args, f) for f in ("resolution", "regret_tol", "purity_tol",
                                                             "coverage_tol")},
        "given_source": {f: ("cli" if getattr(args, f"_{f}_cli", False) else "DEV_DEFAULTS")
                         for f in ("resolution", "regret_tol", "purity_tol", "coverage_tol")},
        "resolution_sensitivity_reported": list(TIE_RESOLUTIONS_REPORTED),
    }


def parse_params(text: Optional[str], cls):
    if not text:
        return None
    return cls(**json.loads(text))


def stage_dev(args, root: pathlib.Path) -> dict:
    if args.set != "dev":
        raise SystemExit("--stage dev is for --set dev")
    for f in ("resolution", "regret_tol", "purity_tol", "coverage_tol"):
        setattr(args, f"_{f}_cli", getattr(args, f) is not None)
        if getattr(args, f) is None:
            setattr(args, f, DEV_DEFAULTS[f])
    keys = resolve_keys(args)
    cand = candidates(args)
    names = {"cell": cand["cell"], "arm": cand["arm"]}
    kinds = list(args.kinds)
    rds = [load_record(args, root, k, names, [PRIMARY_CONDITION], kinds) for k in keys]
    cond = PRIMARY_CONDITION
    t0 = time.time()
    fixed1 = parse_params(args.s1, ss.S1Params)
    fixed2 = parse_params(args.s2, ss.S2Params)
    grid1 = [fixed1] if fixed1 else S1_GRID
    rows1 = []
    best1, bobj = None, None
    for i, p1 in enumerate(grid1):
        res = analyze(rds, [cond], kinds, p1, ss.S2Params(), args.resolution, None, False)
        obj = _objective(res, cond)
        rows1.append({"order": i, "s1": p1.as_dict(), "objective": list(obj), **_grid_row(res, cond, "S1")})
        if bobj is None or obj < bobj:
            best1, bobj = p1, obj
    grid2 = [fixed2] if fixed2 else S2_GRID
    rows2 = []
    best2, bobj2 = None, None
    for i, p2 in enumerate(grid2):
        res = analyze(rds, [cond], kinds, best1, p2, args.resolution, None, False)
        obj = _objective_s2(res, cond)
        rows2.append({"order": i, "s2": p2.as_dict(), "objective": list(obj), **_grid_row(res, cond, "S2")})
        if bobj2 is None or obj < bobj2:
            best2, bobj2 = p2, obj
    W = sel_mod.episode_draws(len(rds), args.n_boot, args.seed)
    conds_all = [c for c in args.conditions]
    rds_all = [load_record(args, root, k, names, conds_all, kinds) for k in keys] \
        if conds_all != [cond] else rds
    final = analyze(rds_all, conds_all, kinds, best1, best2, args.resolution, W, False)
    ties = {}
    for r in TIE_RESOLUTIONS_REPORTED:
        rr = analyze(rds, [cond], kinds, best1, best2, r, None, False)
        ties[str(r)] = {g: {v: {"tie_size": rr[cond][g]["all"][v]["summary"]["tie_size"],
                                "selected": rr[cond][g]["all"][v]["summary"]["selected"]}
                            for v in ss.VARIANTS}
                        for g in rr[cond] if not g.startswith("_")}
    checks = label_checks(rds, cond, kinds[0], best1, best2, final[cond][f"_fs_{kinds[0]}"], args)
    t55 = t55_label_checks(args, keys)
    prereg = {
        "step": STEP, "kind": "preregistration", "written_kst": kst_now(), "git": git_state(),
        "set_used": "dev", "dev_records": keys, "dev_excluded": DEV_EXCLUDED,
        "resolution": args.resolution, "s1": best1.as_dict(), "s2": best2.as_dict(),
        "regret_tol": args.regret_tol, "purity_tol": args.purity_tol, "coverage_tol": args.coverage_tol,
        "tolerance_note": ("practical values given to the dev stage (defaults aligned with T54's −3 %p "
                           "identity / −5 %p coverage); not derived statistically — confirm with the user"),
        "selected_by": {"s1": "cli" if fixed1 else "dev rule (min mean tie-set regret over cells+arms, "
                                                     "then max mean τ-b, then grid order)",
                        "s2": "cli" if fixed2 else "same rule on S2 with the chosen S1"},
        "search_space": search_space(fixed1, fixed2, args),
        "tie_break": ["score >= max - resolution → tie set",
                      "larger S1 coverage component (episode mean over the variant's event episodes)",
                      "lexicographically first candidate name", "a 2D lift never enters"],
        "candidates": {"cell": cand["cell"], "arm": cand["arm"], "cell_set": args.cell_set,
                       "cell_sources": cand["cell_source"]["sources"]},
        "offline_latch": offline_config(args).as_dict(),
        "bootstrap": {"B": args.n_boot, "seed": args.seed, "ci": list(sel_mod.CI), "unit": "episode"},
        "primary_condition": PRIMARY_CONDITION, "phases": list(PHASES),
        "label_note": ("S1 / S2 were developed with GT (dev regret / τ-b); at inference they use no GT. "
                       "Not fully label-free."),
    }
    prereg["sha256"] = hashlib.sha256(json.dumps(prereg, sort_keys=True, default=str).encode()).hexdigest()
    summary = {**run_info("dev", args), "prereg": prereg, "grid_s1": rows1, "grid_s2": rows2,
               "final": {c: {g: public(v) for g, v in final[c].items() if not g.startswith("_")}
                         for c in conds_all},
               "tie_by_resolution": ties, "label_checks": checks, "t55_resim_label_checks": t55,
               "seconds": time.time() - t0}
    write_json(root / "dev" / "dev_summary.json", summary)
    write_json(root / PREREG_NAME, prereg)
    print(json.dumps({"s1": prereg["s1"], "s2": prereg["s2"], "resolution": prereg["resolution"],
                      "prereg": str(root / PREREG_NAME)}, indent=1))
    print("STOP: the lead confirms T56.prereg.json with the user before --stage test.")
    return prereg


# -------------------------------------------------------------------------------- test stage


def read_prereg(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"no prereg {path} — run --set dev --stage dev first (and get it confirmed)")
    d = json.loads(path.read_text())
    want = d.get("sha256")
    body = {k: v for k, v in d.items() if k != "sha256"}
    got = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    if want != got:
        raise SystemExit(f"{path}: sha256 mismatch (edited after the dev stage?)")
    return d


def stage_test(args, root: pathlib.Path) -> dict:
    if args.set != "test":
        raise SystemExit("--stage test is for --set test")
    bad = [f for f in DEV_ONLY_FLAGS if getattr(args, f) is not None]
    if bad:
        raise SystemExit(f"--stage test reads the prereg only; remove {['--' + b.replace('_', '-') for b in bad]}")
    if "arm" in args.kinds and not args.t54_run:
        raise SystemExit("--stage test with arms needs --t54-run (T54 test rows are the arms' GT, "
                         "T56.task.md 사전 고정 1)")
    pre_p = pathlib.Path(args.prereg) if args.prereg else root / PREREG_NAME
    pre = read_prereg(pre_p)
    p1, p2 = ss.S1Params(**pre["s1"]), ss.S2Params(**pre["s2"])
    names = {"cell": list(pre["candidates"]["cell"]), "arm": list(pre["candidates"]["arm"])}
    keys = resolve_keys(args)
    kinds = list(args.kinds)
    conds = list(args.conditions)
    rds = [load_record(args, root, k, names, conds, kinds) for k in keys]
    W = sel_mod.episode_draws(len(rds), int(pre["bootstrap"]["B"]), int(pre["bootstrap"]["seed"]))
    t0 = time.time()
    res = analyze(rds, conds, kinds, p1, p2, float(pre["resolution"]), W, True)
    checks = {c: label_checks(rds, c, kinds[0], p1, p2, res[c][f"_fs_{kinds[0]}"], args) for c in conds}
    crit = {}
    pc = PRIMARY_CONDITION if PRIMARY_CONDITION in res else conds[0]
    for g in [g for g in res[pc] if not g.startswith("_")]:
        s = res[pc][g]["all"]
        s1, s0 = s["S1"]["summary"], s["S0"]["summary"]

        def num(x):
            return None if x is None else float(x)
        r1, r0 = num(s1.get("all:identity:tie_regret")), num(s0.get("all:identity:tie_regret"))
        rt1, rt0 = num(s1.get("all:identity:regret")), num(s0.get("all:identity:regret"))
        t1, t0_ = s1["tau_b:identity"]["mean_defined"], s0["tau_b:identity"]["mean_defined"]
        crit[g] = {
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
    out = {**run_info("test", args), "prereg_path": str(pre_p), "prereg_sha256": pre["sha256"],
           "prereg": pre, "keys": keys,
           "gt_source": {f"{rd.key}:{c}:{k}": v for rd in rds for (c, k), v in rd.gt_source.items()},
           "t54_parity": {f"{rd.key}:{c}:{k}": v for rd in rds for (c, k), v in rd.parity.items()},
           "results": {c: {g: {kk: (public(vv) if kk == "all" else {ph: public(x) for ph, x in vv.items()})
                               for kk, vv in rec.items()}
                           for g, rec in res[c].items() if not g.startswith("_")} for c in conds},
           "label_checks": checks, "criteria_mechanical": crit,
           "definitions": {"selfscore": ss.__doc__, "select": sel_mod.__doc__},
           "t55_resim_label_checks": (t55_label_checks(args, keys) if args.t55_run else
                                      "not given (--t55-run)"),
           "seconds": time.time() - t0}
    write_json(root / "test" / "analysis" / "t56_summary.json", out)
    print(json.dumps(crit, indent=1, default=str)[:4000])
    return out


# ------------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="new development run (outputs: verify/T56/)")
    ap.add_argument("--set", choices=SETS, required=True)
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--records", nargs="*", default=None)
    ap.add_argument("--kinds", nargs="*", default=list(KINDS), choices=list(KINDS))
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--cells", nargs="*", default=None, help="restrict the cell candidates (smoke)")
    ap.add_argument("--cell-set", default=cells_mod.DEFAULT_CELL_SET, choices=list(cells_mod.CELL_SETS))
    ap.add_argument("--batch", type=int, default=10, help="cells per replay job (memory in hold)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="smoke: frame indices only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--t53-run", default=DEFAULTS["t53_run"])
    ap.add_argument("--t51-run", default=DEFAULTS["t51_run"])
    ap.add_argument("--t49-root", default=DEFAULTS["t49_root"])
    ap.add_argument("--t47-verify", default=DEFAULTS["t47_verify"])
    ap.add_argument("--t54-run", default=None, help="test: read T54 test rows for the arms (GT) + parity")
    ap.add_argument("--t55-run", default=None,
                    help="dev / test: T55 run whose verify/T55/resim/<set>/ re-simulations get the HELD "
                         "label check (reported only)")
    # offline latch (dev)
    ap.add_argument("--reach-source", default="reference_candidate",
                    choices=["reference_candidate", "not_checked"])
    ap.add_argument("--reach-candidate", default="last:L8H2")
    ap.add_argument("--reach-condition", default="hold", choices=list(CONDITIONS))
    ap.add_argument("--no-slip-detach", action="store_true")
    # dev-only parameters (the test stage refuses them)
    ap.add_argument("--resolution", type=float, default=None)
    ap.add_argument("--regret-tol", type=float, default=None)
    ap.add_argument("--purity-tol", type=float, default=None)
    ap.add_argument("--coverage-tol", type=float, default=None)
    ap.add_argument("--s1", default=None, help='dev: fix S1 instead of the grid, JSON of S1Params')
    ap.add_argument("--s2", default=None, help='dev: fix S2 instead of the grid, JSON of S2Params')
    ap.add_argument("--n-boot", type=int, default=sel_mod.N_BOOT)
    ap.add_argument("--seed", type=int, default=sel_mod.SEED)
    ap.add_argument("--prereg", default=None, help="test: prereg path (default <out>/T56.prereg.json)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.frames is not None and args.stage in ("dev", "test"):
        ap.error("--frames is a replay smoke option")
    if args.stage == "test" and (args.n_boot, args.seed) != (sel_mod.N_BOOT, sel_mod.SEED):
        ap.error("--stage test takes B / seed from the prereg")
    root = out_root(args.run_dir, [args.t53_run, args.t54_run, args.t51_run, args.t55_run])
    configure(args)
    {"inputs": stage_inputs, "replay": stage_replay, "held": stage_held, "dev": stage_dev,
     "test": stage_test}[args.stage](args, root)


if __name__ == "__main__":
    main()
