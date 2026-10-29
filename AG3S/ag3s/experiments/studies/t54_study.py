"""T54 CLI — grounding comparison of the fixed arms on dev (parity) and T53 (final test).

`T54.task.md`. Arms: `t54_arms.ARMS`; conditions / profiles: `t54_replay`; metrics / bootstrap:
`t54_stats`. Every output goes to ``<--run-dir>/verify/T54/`` (a development run of the shared storage,
never the T53 run). Inputs are read only.

Stages::

    inputs    CPU, seconds. Locate and pair every input of the set (records, action npz, noun extract,
              T53 chunk records → recorded freeze inputs, server argv → profile); no render.
              → <out>/<set>/inputs.json (exit != 0 on a problem)
    budget    test only, CPU, seconds. Reads the T53 **serving** records only (never a T54 row): the
              server's per-chunk AG3S `profile_ms` (total and per stage), the client's
              `policy_infer` ms, the wall-clock interval between chunk observations and the nominal
              chunk period (planning t_step stride × control step). → <out>/test/budget.json
    replay    CPU, MuJoCo osmesa, <= 4 workers, one job per (record, condition).
              → <out>/<set>/rows/<key>__<condition>.jsonl (+ .meta.json, written last = done)
    parity    dev only: the ``hold`` rows of L8H2 and N (= the T47b/T51/T52 loop: sequential,
              TargetConfirm on, nothing injected, t52 profile) vs T52 / T51 / T51b rows on the T52
              parity fields. → <out>/dev/parity.json (exit != 0 on a mismatch). The ``current``
              rows have no earlier reference; their differences from T52 are counted for information.
    analyze   frames, episode means, paired differences, episode / moving-block bootstraps, phase
              split, T57 entry rows. → <out>/<set>/analysis/t54_summary.json, t54_frames.jsonl

Sets: ``dev`` = T47 records (held-out 4 by default: headfix · ep1800 · ep1850 · ep1900; ``--records``
may add long · t2 · ep1807), profile t52; ``test`` = the 16 T53 episodes, profile c1a.

Example (verifier)::

    R=<new run dir>   T53=<T53 run dir>
    M=AG3S.ag3s.experiments.studies.t54_study
    export MUJOCO_GL=osmesa OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$PWD
    .venv-ag3s/bin/python -m $M --run-dir $R --set dev  --stage inputs
    .venv-ag3s/bin/python -m $M --run-dir $R --set dev  --stage replay --workers 4
    .venv-ag3s/bin/python -m $M --run-dir $R --set dev  --stage parity
    .venv-ag3s/bin/python -m $M --run-dir $R --set test --t53-run $T53 --stage inputs
    .venv-ag3s/bin/python -m $M --run-dir $R --set test --t53-run $T53 --stage budget
    .venv-ag3s/bin/python -m $M --run-dir $R --set test --t53-run $T53 --stage replay --workers 4
    .venv-ag3s/bin/python -m $M --run-dir $R --set test --t53-run $T53 --stage analyze \\
        [--extract-ms extract_ms.json] [--budget-ms <ms>]

T54b / T54c extensions (defaults reproduce T54 exactly):

* ``--conditions ... hold_attach`` (T54b): `t54b_hold` — the hold replay with the recorded grasp latch,
  attach / detach and freeze injected identically into every arm. ``--attach-reach serving|off``.
  ``--stage fidelity``: the ``hold_attach`` rows of one arm (default L8H2) against the serving chunk
  records → ``<out>/test/analysis/t54b_fidelity.json``.
* ``--registry t54c --records-run <T54c run>`` (T54c): the test set is `t54c_records` (16 new episodes);
  the records' directory is ``<run>/verify/T54c``; outputs default to ``<run-dir>/verify/T54c/replay``.
  Bootstrap seed default 5417 (T54c 사전 고정), 5407 for t53.
* ``--out-subdir`` (e.g. ``verify/T54b``), ``--extra-rows-dir`` (read rows of a condition another run
  made, read only), ``--entry-arms`` (default A) and ``--hold-condition`` (default ``hold``) for the T57
  entry table.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t54_arms as arms_mod

STEP = "T54"
MAX_WORKERS = 4
SETS = ("dev", "test")
STAGES = ("inputs", "budget", "replay", "parity", "analyze", "fidelity")
CONDITIONS = ("current", "hold", "hold_attach")
REGISTRIES = ("t53", "t54c")
#: Pre-fixed bootstrap seed per registry (T54 사전 고정 4: 5407; T54c 사전 고정: 5417).
SEED_BY_REGISTRY = {"t53": 5407, "t54c": 5417}
OUT_SUBDIR_BY_REGISTRY = {"t53": "verify/T54", "t54c": "verify/T54c/replay"}
DEV_PARITY_KEYS = ("headfix", "ep1800", "ep1850", "ep1900")
#: Read-only reference runs (the ones `T52.verify.json` lists as inputs).
_DEV = "/mnt/dev/AG3S_VLA-storage/results/development"
DEFAULTS = {
    "t49_root": f"{_DEV}/20261012T190547+0900_28b981600bf7_c39a577332fe/verify/T49",
    "t52_rows": f"{_DEV}/20261013T094624+0900_693dcedeaadc_a2c23cf2a275/verify/T52/raw/replay",
    "t51_rows": f"{_DEV}/20261012T180318+0900_28b981600bf7_852011fdcae0/verify/T51/rows",
    "t51b_rows": f"{_DEV}/20261012T180318+0900_28b981600bf7_852011fdcae0/verify/T51b/rows",
    "t47_gt": f"{_DEV}/20261012T141412+0900_bd778e5e3453_d7a51b456f36/verify/T47/gt_cache",
    "t53_run": f"{_DEV}/20261013T100210+0900_4a540f1a18d7_cb037c425d2f",
}
#: T52 `parity` fields (`t52_analyze.py` `parity`).
PARITY_FIELDS = ("identity_correct", "on_target", "has_target", "chosen_cluster_purity",
                 "seed_precision", "seed_recall", "n_seeds", "n_seeds_target", "chosen_n",
                 "chosen__other_object", "seed__other_object", "target_residual_rate", "n_cloud",
                 "centroid_error_m")
#: arm -> (T52 source, T51/T51b dir key, T51/T51b source file stem).
PARITY_REFS = {"L8H2": ("l8h2", "t51_rows", "last_L8H2"),
               "N": ("noun_L10H7", "t51b_rows", "noun_target_L10H7")}


# ------------------------------------------------------------------------------ plumbing


def kst_now() -> str:
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat()


def write_json(path: pathlib.Path, payload: Any) -> None:
    from AG3S.ag3s.experiments.studies.t47_head_selection import write_json as wj

    wj(path, payload)


def jsonable(x):
    from AG3S.ag3s.experiments.studies.t47b_metrics import jsonable as jb

    return jb(x)


def out_root(run_dir: str, t53_run: Optional[str] = None, *, subdir: str = "verify/T54",
             refuse_records_run: bool = True) -> pathlib.Path:
    """``<run_dir>/<subdir>``. With `refuse_records_run` (the T53 registry) the run must not be the
    records run. A T54c run may hold its records (``verify/T54c/<episode dirs>``) and this study's
    outputs (``verify/T54c/replay``) side by side; the subdir must then not be the records directory."""
    from common.run_output import resolve_output_path

    try:
        root = resolve_output_path(subdir, run_dir=run_dir, default_name=subdir)
    except ValueError as exc:
        raise SystemExit(f"--run-dir {run_dir}: {exc}") from exc
    run = pathlib.Path(run_dir).resolve()
    if t53_run is not None and run == pathlib.Path(t53_run).resolve():
        if refuse_records_run:
            raise SystemExit("--run-dir must be a new run, not the T53 run (inputs are read only)")
        if pathlib.PurePosixPath(subdir).parts in (("verify", "T54c"), ("verify", "T53")):
            raise SystemExit(f"--out-subdir {subdir} is the records directory")
    return root


def records_module(registry: str):
    """The registry module (`t53_records` or the T54c clone, `t54c_records.records`)."""
    from AG3S.ag3s.experiments.studies.t54c_records import registry_module

    return registry_module(registry)


def _rec(args):
    return records_module(getattr(args, "registry", "t53"))


def run_info(stage: str, args) -> dict:
    from AG3S.ag3s.experiments.studies.t47_head_selection import git_state

    return {"step": STEP, "stage": stage, "set": args.set, "argv": sys.argv,
            "args": {k: v for k, v in vars(args).items()}, "started_kst": kst_now(),
            "git": git_state(), "python": sys.version.split()[0],
            "arms": {a.name: a.as_dict() for a in arms_mod.resolve_arms(args.arms)},
            "sources": arms_mod.SOURCES}


# --------------------------------------------------------------------------------- inputs


def load_action(npz: pathlib.Path):
    """T47-format action attention (`t47_head_selection.load_attention` checks)."""
    from AG3S.ag3s.experiments.studies.t47_head_selection import POLICY_CAMERAS

    with np.load(npz, allow_pickle=False) as z:
        att = np.asarray(z["attention"])
        dn = [int(d) for d in z["denoise_steps"]]
        agl = [str(a) for a in z["aggregations"]]
        cams = [str(c) for c in z["cameras"]]
        t_step = [int(t) for t in z["t_step"]] if "t_step" in z.files else None
    if att.ndim != 8 or att.shape[-2:] != (16, 16) or tuple(cams) != POLICY_CAMERAS:
        raise SystemExit(f"{npz}: unexpected attention {att.shape} / cameras {cams}")
    return att, dn, agl, t_step


def build_maps(att, dn, agl, noun_ex: Optional[dict], sources: Sequence[str]) -> dict:
    out = {}
    block = None
    for s in sources:
        spec = arms_mod.SOURCES[s]
        if spec["kind"] == "action":
            out[s] = arms_mod.action_map(att, dn, agl, spec)
        else:
            if noun_ex is None:
                raise SystemExit(f"source {s} needs the noun extract")
            if block is None:
                from AG3S.ag3s.experiments.studies.t49_noun_query import block_fc

                block = block_fc(noun_ex, "target")
            out[s] = arms_mod.noun_map(block, spec)
    return out


def configure_t53(t53_run: str, registry: str = "t53") -> dict:
    """Configure the registry on its records run; the episode / server roots come from that run's
    ``layout.json`` (`t53_extract --stage layout`, or `t54c_extract` for T54c)."""
    rec = records_module(registry)
    lay_p = rec.t53_dir_of_run(t53_run, check_storage=False) / "layout.json"
    lay = json.loads(lay_p.read_text()) if lay_p.is_file() else {"args": {}}
    eroot = lay["args"].get("episodes_root")
    rec.configure(run_dir=t53_run, episodes_root=eroot)
    return {"layout": str(lay_p) if lay_p.is_file() else None, "episodes_root": eroot,
            "server_roots": list(lay["args"].get("server_roots") or [])}


def record_inputs(set_: str, key: str, args) -> dict:
    """Paths + pairing of one record (no render). Raises SystemExit on an ambiguity / mismatch."""
    from AG3S.ag3s.experiments.studies.t49_noun_query_extract import load_extract

    need = arms_mod.sources_needed(arms_mod.resolve_arms(args.arms))
    noun_needed = any(arms_mod.SOURCES[s]["kind"] == "noun" for s in need)
    info: dict[str, Any] = {"key": key, "set": set_, "problems": []}
    if set_ == "dev":
        from AG3S.ag3s.experiments.studies import t47_head_selection as hs
        from AG3S.ag3s.experiments.studies import t49_noun_query as t49

        if key not in hs.RECORDS:
            raise SystemExit(f"dev key {key} not a T47 record ({list(hs.RECORDS)})")
        record, npz = hs.record_path(key), hs.npz_path(key)
        noun = t49.extract_path(pathlib.Path(args.t49_root), "orig", key)
        target = hs.target_of(key)
        info.update({"profile": "t52", "server_proc": None, "chunks": None})
    else:
        rec = _rec(args)

        if rec.episode_of(key).episode == 1800:
            raise SystemExit("scene 1800 is never part of the T54 final test")
        files = rec.locate(key)
        record, npz, noun = files.record, rec.npz_path(key), rec.noun_path(key)
        target = rec.target_of(key)
        plan = [r for r in rec.read_frames(files.frames) if r.get("kind") == "planning"] \
            if files.frames is not None else []
        roots = [pathlib.Path(p) for p in (args.server_roots or args._t53_cfg["server_roots"])]
        ch = rec.locate_chunks(key, ep_dir=files.episode_dir, server_roots=roots, planning=plan)
        chunks = list(ch["chunks"])
        info["chunks_source"] = ch["source"]
        if chunks:
            _, cp = rec.chunk_problems(chunks, plan)
            info["problems"] += cp
            server_dir = pathlib.Path(chunks[0]).parents[2]
            proc = server_dir / "server_proc.txt"
            if not proc.is_file():
                info["problems"].append(f"no server_proc.txt in {server_dir}")
        else:
            proc = None
            info["problems"].append(f"no server chunks ({ch['source']}): hold cannot inject freeze")
        info.update({"profile": "c1a", "server_proc": None if proc is None else str(proc),
                     "chunks": [str(c) for c in chunks]})
    from AG3S.ag3s.experiments.sources.policy_record import load_run

    run = load_run(record)
    t_rec = [int(s.t_step) for s in run.steps]
    att_t = None
    if not pathlib.Path(npz).is_file():
        info["problems"].append(f"no action npz {npz}")
    else:
        _, _, _, att_t = load_action(pathlib.Path(npz))
        if att_t is not None and att_t != t_rec:
            info["problems"].append(f"action npz t_step != record t_step ({npz})")
    if noun_needed:
        if not pathlib.Path(noun).is_file():
            info["problems"].append(f"no noun extract {noun}")
        else:
            ex = load_extract(pathlib.Path(noun))
            if [int(t) for t in ex["t_step"]] != t_rec:
                info["problems"].append(f"noun extract t_step != record t_step ({noun})")
            if not bool(np.asarray(ex["ok"], bool).all()):
                info["problems"].append(f"noun extract has failed frames {np.flatnonzero(~np.asarray(ex['ok'], bool)).tolist()}")
            info["noun_prefix_ms_median"] = float(np.median(np.asarray(ex["prefix_ms"], np.float64))) \
                if "prefix_ms" in ex else None
    info.update({"record": str(record), "npz": str(npz), "noun": str(noun), "target": target,
                 "n_frames": len(t_rec), "prompt": run.prompt})
    return info


def resolve_set_keys(args) -> list[str]:
    if args.set == "dev":
        from AG3S.ag3s.experiments.studies import t47_head_selection as hs

        keys = list(args.records or DEV_PARITY_KEYS)
        bad = [k for k in keys if k not in hs.RECORDS]
    else:
        keys = _rec(args).resolve_keys(args.records)
        bad = []
    if bad:
        raise SystemExit(f"unknown {args.set} record(s) {bad}")
    return keys


def stage_inputs(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t54_replay as rp

    keys = resolve_set_keys(args)
    recs, probs = {}, []
    argv_by_proc: dict[str, list] = {}
    for k in keys:
        info = record_inputs(args.set, k, args)
        if info.get("chunks"):
            fz = rp.freeze_from_chunks([pathlib.Path(c) for c in info["chunks"]])
            info["freeze_summary"] = rp.freeze_summary(fz)
            if len(fz) != info["n_frames"]:
                info["problems"].append(f"{len(fz)} chunks for {info['n_frames']} frames")
            if "hold_attach" in args.conditions:
                from AG3S.ag3s.experiments.studies import t54b_hold as hb

                evs = hb.events_summary(hb.events_from_chunks([pathlib.Path(c) for c in info["chunks"]]))
                info["hold_attach_events"] = evs
                info["problems"] += [f"hold_attach: {p}" for p in evs["problems"]]
        if info.get("server_proc"):
            argv_by_proc.setdefault(info["server_proc"], rp.server_argv(pathlib.Path(info["server_proc"])))
        info.pop("chunks", None)
        recs[k] = info
        probs += [f"[{k}] {p}" for p in info["problems"]]
    profiles = {}
    for proc, argv in argv_by_proc.items():
        _, kw, rec = rp.c1a_build_kwargs(argv)
        profiles[proc] = {"build_kwargs": {k: (v if isinstance(v, (int, float, str, bool, type(None), dict, list)) else repr(v)) for k, v in kw.items()}, **rec}
    if len({json.dumps(p["build_kwargs"], sort_keys=True, default=str) for p in profiles.values()}) > 1:
        probs.append("the T53 servers' AG3S build arguments differ between server processes")
    payload = {**run_info("inputs", args), "records": recs, "c1a_profiles": profiles,
               "problems": probs, "all_ok": not probs}
    if args.set == "test":
        rec = _rec(args)
        payload["registry"] = args.registry
        payload["t53"] = {"run": args.t53_run, **args._t53_cfg, "provenance": rec.provenance(keys)}
    write_json(root / args.set / "inputs.json", payload)
    print(json.dumps({k: (v.get("freeze_summary"), v["problems"]) for k, v in recs.items()},
                     indent=1, default=str)[:4000])
    if probs:
        raise SystemExit("inputs: problems:\n  " + "\n  ".join(probs))
    return payload


# --------------------------------------------------------------------------------- budget


def _dist(x) -> dict:
    v = np.asarray([float(a) for a in x if a is not None and math.isfinite(float(a))], np.float64)
    if not v.size:
        return {"median": float("nan"), "p95": float("nan"), "mean": float("nan"), "n": 0}
    return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "mean": float(v.mean()), "n": int(v.size)}


def stage_budget(args, root: pathlib.Path) -> dict:
    """T53 serving timing (module doc). Facts only; which number is "the budget" is the verifier's."""
    if args.set != "test":
        raise SystemExit("--stage budget is for --set test (the T53 serving records)")
    from AG3S.ag3s.experiments.sources.policy_record import control_step_ms, load_run
    from AG3S.ag3s.experiments.studies import t54_replay as rp

    rec = _rec(args)

    keys = resolve_set_keys(args)
    per_stage: dict[str, list] = {}
    totals, lift_ground, infer, interval, period = [], [], [], [], []
    per_ep = {}
    for k in keys:
        info = record_inputs("test", k, args)
        chunks = [pathlib.Path(c) for c in (info.get("chunks") or [])]
        ep_tot = []
        for c in chunks:
            prof = rp.read_summary(c).get("profile_ms") or {}
            for st_, v in prof.items():
                per_stage.setdefault(st_, []).append(v)
            t = float(sum(float(v) for v in prof.values()))
            totals.append(t)
            ep_tot.append(t)
            lift_ground.append(float(prof.get("attention_lifting", 0.0)) + float(prof.get("target_grounding", 0.0)))
        files = rec.locate(k)
        rows = rec.read_frames(files.frames) if files.frames is not None else []
        plan = [r for r in rows if r.get("kind") == "planning"]
        obs = [r for r in rows if r.get("kind") == "observation"]
        infer += [((r.get("timing_ms") or {}).get("policy_infer")) for r in plan]
        st_obs = [r.get("newest_stamp") for r in obs if r.get("newest_stamp") is not None]
        interval += [1000.0 * (b - a) for a, b in zip(st_obs[:-1], st_obs[1:])]
        dt = control_step_ms(load_run(files.record))
        ts = [int(r["t_step"]) for r in plan]
        period += [(b - a) * dt for a, b in zip(ts[:-1], ts[1:])]
        per_ep[k] = {"n_chunks": len(chunks), "ag3s_profile_total_ms": _dist(ep_tot),
                     "control_step_ms": dt}
    payload = {**run_info("budget", args),
               "source": "T53 serving records only: server chunk summary_json.profile_ms, client "
                         "frames.jsonl (planning timing_ms.policy_infer, observation newest_stamp, "
                         "planning t_step), step record meta ctrl_hz",
               "ag3s_profile_total_ms_per_chunk": _dist(totals),
               "ag3s_profile_stage_ms_per_chunk": {k: _dist(v) for k, v in sorted(per_stage.items())},
               "attention_lifting_plus_target_grounding_ms_per_chunk": _dist(lift_ground),
               "policy_infer_ms_per_chunk": _dist(infer),
               "observation_interval_ms_wall": _dist(interval),
               "chunk_period_ms_nominal": _dist(period),
               "per_episode": per_ep,
               "note": ("profile_ms covers AG3S stages inside the server (scene_reconstruction includes the "
                        "per-camera attention lift); attention extraction itself runs in the policy "
                        "wrapper and is not in profile_ms")}
    write_json(root / "test" / "budget.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k.endswith("_chunk") or k.endswith("_wall")
                      or k.endswith("_nominal")}, indent=1, default=str)[:3000])
    return payload


# --------------------------------------------------------------------------------- replay


def rows_dir(root: pathlib.Path, set_: str) -> pathlib.Path:
    return root / set_ / "rows"


def rows_path(root: pathlib.Path, set_: str, key: str, cond: str) -> pathlib.Path:
    return rows_dir(root, set_) / f"{key}__{cond}.jsonl"


def _job(spec: dict) -> dict:
    """One (record, condition) replay in a worker process (spawned: env first)."""
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import resource

    from AG3S.ag3s.experiments.studies import t54_replay as rp
    from AG3S.ag3s.experiments.studies.t49_noun_query_extract import load_extract

    if spec["set"] == "test":
        rec = records_module(spec.get("registry", "t53"))
        rec.configure(run_dir=spec["t53_run"], episodes_root=spec["episodes_root"])
    key, cond = spec["key"], spec["condition"]
    out = pathlib.Path(spec["rows"])
    log_p = out.with_suffix(".log")
    lines: list[str] = []

    def log(msg: str) -> None:
        lines.append(msg)
        with open(log_p, "a") as fh:
            fh.write(msg + "\n")

    t0 = time.time()
    arms = arms_mod.resolve_arms(spec["arms"])
    need = arms_mod.sources_needed(arms)
    att, dn, agl, _ = load_action(pathlib.Path(spec["npz"]))
    noun_ex = (load_extract(pathlib.Path(spec["noun"]))
               if any(arms_mod.SOURCES[s]["kind"] == "noun" for s in need) else None)
    maps = build_maps(att, dn, agl, noun_ex, need)
    del att
    build, prov = rp.make_builder(spec["profile"], pathlib.Path(spec["record"]),
                                  server_proc=(None if spec["server_proc"] is None
                                               else pathlib.Path(spec["server_proc"])))
    freeze = None
    events_meta: dict = {}
    if cond == "hold_attach":
        # T54b: recorded latch + attach / detach + freeze, identical for every arm (t54b_hold)
        from AG3S.ag3s.experiments.studies import t54b_hold as hb

        if not spec.get("chunks"):
            raise SystemExit(f"[{key}] hold_attach needs the serving chunk records")
        events = hb.events_from_chunks([pathlib.Path(c) for c in spec["chunks"]])
        freeze = events
        events_meta = {"hold_attach_events": hb.events_summary(events),
                       "attach_reach": spec.get("attach_reach", "serving")}
        rows = hb.run_record_attach(pathlib.Path(spec["record"]), target=spec["target"], maps=maps,
                                    arms=arms, build=build, events=events, frames=spec.get("frames"),
                                    attach_reach=spec.get("attach_reach", "serving"), log=log)
    else:
        if cond == "hold" and spec.get("chunks"):
            freeze = rp.freeze_from_chunks([pathlib.Path(c) for c in spec["chunks"]])
        rows = rp.run_record(pathlib.Path(spec["record"]), target=spec["target"], maps=maps,
                             arms=arms, condition=cond, build=build, freeze=freeze,
                             frames=spec.get("frames"), log=log)
    tmp = out.with_suffix(".partial")
    with open(tmp, "w") as fh:
        for a in arms:
            for r in rows[a.name]:
                fh.write(json.dumps(jsonable({"record": key, **r}), allow_nan=True) + "\n")
    tmp.replace(out)
    meta = {"key": key, "condition": cond, "set": spec["set"], "arms": [a.name for a in arms],
            "profile": prov, "freeze_injected": freeze is not None,
            "freeze_summary": None if freeze is None else rp.freeze_summary(freeze),
            "freeze": freeze, "frames": spec.get("frames"),
            **events_meta,
            "registry": spec.get("registry", "t53"),
            "n_rows": int(sum(len(v) for v in rows.values())),
            "seconds": time.time() - t0,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "finished_kst": kst_now()}
    write_json(out.with_suffix(".meta.json"), meta)
    return {"key": key, "condition": cond, "seconds": meta["seconds"],
            "peak_rss_mb": meta["peak_rss_mb"], "n_rows": meta["n_rows"]}


def stage_replay(args, root: pathlib.Path) -> dict:
    import concurrent.futures as cf
    import multiprocessing as mp

    if args.workers < 1 or args.workers > MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS}")
    keys = resolve_set_keys(args)
    if "hold_attach" in args.conditions and args.set != "test":
        raise SystemExit("hold_attach needs serving chunk records (--set test)")
    rows_dir(root, args.set).mkdir(parents=True, exist_ok=True)
    specs, skipped = [], []
    for k in keys:
        info = record_inputs(args.set, k, args)
        if info["problems"] and not (args.set == "test" and args.conditions == ["current"]):
            raise SystemExit(f"[{k}] input problems (run --stage inputs): {info['problems']}")
        for cond in args.conditions:
            p = rows_path(root, args.set, k, cond)
            if p.with_suffix(".meta.json").is_file() and not args.force:
                skipped.append(p.name)
                continue
            for stale in (p, p.with_suffix(".log"), p.with_suffix(".partial")):
                if stale.exists():
                    stale.unlink()
            specs.append({"set": args.set, "key": k, "condition": cond, "rows": str(p),
                          "arms": args.arms, "npz": info["npz"], "noun": info["noun"],
                          "record": info["record"], "target": info["target"],
                          "profile": info["profile"], "server_proc": info["server_proc"],
                          "chunks": info.get("chunks"), "frames": args.frames,
                          "t53_run": getattr(args, "t53_run", None),
                          "episodes_root": (args._t53_cfg or {}).get("episodes_root")
                          if args.set == "test" else None,
                          "registry": getattr(args, "registry", "t53"),
                          "attach_reach": getattr(args, "attach_reach", "serving")})
    print(f"replay: {len(specs)} job(s), {len(skipped)} done already, {args.workers} worker(s)",
          flush=True)
    results, failures = [], []
    t0 = time.time()
    if specs:
        with cf.ProcessPoolExecutor(max_workers=args.workers,
                                    mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(_job, s): s for s in specs}
            for fut in cf.as_completed(futs):
                s = futs[fut]
                try:
                    r = fut.result()
                    results.append(r)
                    print(f"  done {s['key']} {s['condition']} {r['seconds']:.0f}s "
                          f"peak {r['peak_rss_mb']:.0f} MB ({time.time() - t0:.0f}s)", flush=True)
                except Exception as exc:  # noqa: BLE001 — recorded, other jobs continue
                    failures.append({"key": s["key"], "condition": s["condition"],
                                     "error": f"{type(exc).__name__}: {exc}"})
                    print(f"  FAILED {s['key']} {s['condition']}: {exc}", flush=True)
    payload = {**run_info("replay", args), "jobs": results, "failures": failures,
               "skipped_done": skipped, "seconds": time.time() - t0, "finished_kst": kst_now()}
    write_json(root / args.set / f"replay_{int(time.time())}.json", payload)
    if failures:
        raise SystemExit(f"replay: {len(failures)} job(s) failed")
    return payload


def find_rows(root: pathlib.Path, set_: str, key: str, cond: str,
              extra_dirs: Sequence[str | os.PathLike] = ()) -> pathlib.Path:
    """This run's rows file, else the first ``--extra-rows-dir`` (a ``rows/`` directory of another
    run, read only) that has a finished one."""
    p = rows_path(root, set_, key, cond)
    if p.with_suffix(".meta.json").is_file():
        return p
    for d in extra_dirs or ():
        q = pathlib.Path(d) / p.name
        if q.with_suffix(".meta.json").is_file():
            return q
    return p


def load_rows(root: pathlib.Path, set_: str, keys: Sequence[str], conds: Sequence[str],
              arms: Sequence[str], extra_dirs: Sequence[str | os.PathLike] = (),
              sources: Optional[dict] = None) -> dict:
    out: dict[tuple, list] = {}
    for k in keys:
        for c in conds:
            p = find_rows(root, set_, k, c, extra_dirs)
            if not p.with_suffix(".meta.json").is_file():
                raise SystemExit(f"missing rows {p} (run --stage replay)")
            if sources is not None:
                sources[f"{k}__{c}"] = str(p)
            by: dict[str, list] = {}
            for line in open(p):
                r = json.loads(line)
                by.setdefault(r["arm"], []).append(r)
            for a in arms:
                if a not in by:
                    raise SystemExit(f"{p}: no rows for arm {a}")
                out[(k, c, a)] = sorted(by[a], key=lambda r: r["frame"])
    return out


# --------------------------------------------------------------------------------- parity


def _same(a, b) -> bool:
    if a == b:
        return True
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def compare_rows(mine: Sequence[dict], ref: Sequence[dict], fields: Sequence[str] = PARITY_FIELDS) -> dict:
    ref_by = {int(r["frame"]): r for r in ref}
    n, mism, missing = 0, [], []
    for r in mine:
        q = ref_by.get(int(r["frame"]))
        if q is None:
            missing.append(int(r["frame"]))
            continue
        for f in fields:
            if f not in q:
                continue
            n += 1
            if not _same(q[f], r.get(f)):
                mism.append({"frame": int(r["frame"]), "field": f, "ref": q[f], "mine": r.get(f)})
    return {"fields_compared": n, "n_mismatch": len(mism), "mismatch": mism[:20],
            "frames_mismatch": sorted({m["frame"] for m in mism}),
            "frames_missing_in_ref": missing}


def stage_parity(args, root: pathlib.Path) -> dict:
    if args.set != "dev":
        raise SystemExit("--stage parity is for --set dev")
    keys = [k for k in resolve_set_keys(args) if k in DEV_PARITY_KEYS]
    res: dict[str, Any] = {"hold_vs_reference": {}, "current_vs_t52_information": {}}
    total = 0
    for arm, (t52_src, dir_key, stem) in PARITY_REFS.items():
        for k in keys:
            rows = load_rows(root, "dev", [k], ["hold"], [arm])[(k, "hold", arm)]
            t52 = [json.loads(ln) for ln in open(pathlib.Path(args.t52_rows) / f"rows__{k}.jsonl")]
            t52 = [r for r in t52 if r["source"] == t52_src]
            t51 = [json.loads(ln) for ln in open(pathlib.Path(getattr(args, dir_key)) / k / f"{stem}.jsonl")]
            c1, c2 = compare_rows(rows, t52), compare_rows(rows, t51)
            res["hold_vs_reference"][f"{arm}:{k}:T52:{t52_src}"] = c1
            res["hold_vs_reference"][f"{arm}:{k}:{dir_key}:{stem}"] = c2
            total += c1["n_mismatch"] + c2["n_mismatch"] + len(c1["frames_missing_in_ref"])
            cur_p = rows_path(root, "dev", k, "current")
            if cur_p.with_suffix(".meta.json").is_file():
                cur = load_rows(root, "dev", [k], ["current"], [arm])[(k, "current", arm)]
                c3 = compare_rows(cur, t52)
                res["current_vs_t52_information"][f"{arm}:{k}"] = {
                    "fields_compared": c3["fields_compared"], "n_differences": c3["n_mismatch"],
                    "frames_differing": c3["frames_mismatch"],
                    "identity_differs_frames": compare_rows(cur, t52, ("identity_correct",))["frames_mismatch"]}
    payload = {**run_info("parity", args), "fields": list(PARITY_FIELDS), **res,
               "n_mismatch_total": total, "parity_ok": total == 0,
               "note": ("parity = the hold condition (sequential, TargetConfirm on, nothing injected, "
                        "t52 profile) — the loop of the T51/T51b/T52 rows. The current-frame rows "
                        "have no earlier reference; their differences are information only.")}
    write_json(root / "dev" / "parity.json", payload)
    print(json.dumps({k: v["n_mismatch"] for k, v in res["hold_vs_reference"].items()}, indent=1))
    if total:
        raise SystemExit(f"parity: {total} mismatch(es)")
    return payload


# -------------------------------------------------------------------------------- analyze


def stage_analyze(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t54_stats as st

    keys = resolve_set_keys(args)
    arms = [a.name for a in arms_mod.resolve_arms(args.arms)]
    if arms_mod.BASELINE not in arms:
        raise SystemExit(f"--arms must include {arms_mod.BASELINE}")
    conds = list(args.conditions)
    row_sources: dict = {}
    rows = load_rows(root, args.set, keys, conds, arms, extra_dirs=args.extra_rows_dir,
                     sources=row_sources)
    gt, grasp = {}, None
    if args.set == "test":
        rec = _rec(args)
        gt = {k: rec.load_gt(k) for k in keys}
        grasp = {}
        for k in keys:
            p = rec.grasp_path(k)
            if not p.is_file():
                raise SystemExit(f"[{k}] no grasp npz {p} (t53_extract --stage grasp)")
            with np.load(p, allow_pickle=False) as z:
                grasp[k] = {"t_step": np.asarray(z["t_step"]), "grasp_state": np.asarray(z["grasp_state"])}
    else:
        for k in keys:
            with np.load(pathlib.Path(args.t47_gt) / f"{k}.npz", allow_pickle=False) as z:
                gt[k] = {"t_step": np.asarray(z["t_step"]), "class_px": np.asarray(z["class_px"])}
    frames = st.build_frames(rows, gt, grasp, min_px=args.min_target_px)
    extract_ms = None
    if args.extract_ms:
        extract_ms = json.loads(pathlib.Path(args.extract_ms).read_text())
    kinds = {a: list(arms_mod.ARMS[a].extract_kinds()) for a in arms}
    summary = st.analyze(frames, keys=keys, arms=arms, conditions=conds, base=arms_mod.BASELINE,
                         n_boot=args.n_boot, seed=args.seed, block=args.block,
                         phases=st.GRASP_STATES if grasp is not None else (),
                         extract_ms=extract_ms, extract_kinds=kinds)
    hold_c = args.hold_condition
    by_arm = {}
    if args.set == "test" and set(conds) >= {"current", hold_c}:
        for a in (args.entry_arms or [arms_mod.MAIN_ARM]):
            if a in arms and a != arms_mod.BASELINE:
                # T54c 사전 고정: on the new data the identity difference must also keep T53's sign
                by_arm[a] = st.entry_criteria(summary, arm=a, base=arms_mod.BASELINE,
                                              budget_ms=args.budget_ms, hold_condition=hold_c,
                                              direction_positive=(args.registry == "t54c"))
    crit = by_arm.get(arms_mod.MAIN_ARM)
    payload = {**run_info("analyze", args), "definitions": st.__doc__,
               "min_target_px": args.min_target_px, "extract_ms": extract_ms,
               "extract_kinds": kinds, "summary": summary, "t57_entry": crit,
               "t57_entry_by_arm": by_arm, "hold_condition": hold_c, "row_sources": row_sources,
               "scene_1800_in_set": any(k in ("headfix", "ep1800", "long", "t2") for k in keys)}
    adir = root / args.set / "analysis"
    write_json(adir / "t54_summary.json", payload)
    with open(adir / "t54_frames.jsonl", "w") as fh:
        for f in frames:
            fh.write(json.dumps(f, allow_nan=True) + "\n")
    print(f"analyze: {len(frames)} frame rows -> {adir}")
    return payload


# ------------------------------------------------------------------------------- fidelity


def stage_fidelity(args, root: pathlib.Path) -> dict:
    """T54b 점검: the ``hold_attach`` rows of one arm (default L8H2) against the serving chunk records
    of the same frames (`t54b_hold.fidelity_summary`). Numbers only."""
    from AG3S.ag3s.experiments.studies import t54b_hold as hb

    if args.set != "test":
        raise SystemExit("--stage fidelity is for --set test (serving chunk records)")
    keys = resolve_set_keys(args)
    arm = args.fidelity_arm
    row_sources: dict = {}
    rows = load_rows(root, "test", keys, [hb.CONDITION], [arm], extra_dirs=args.extra_rows_dir,
                     sources=row_sources)
    by_key = {k: rows[(k, hb.CONDITION, arm)] for k in keys}
    summ = hb.fidelity_summary(by_key)
    payload = {**run_info("fidelity", args), "arm": arm, "condition": hb.CONDITION,
               "row_sources": row_sources, "fidelity": summ}
    write_json(root / "test" / "analysis" / "t54b_fidelity.json", payload)
    print(json.dumps({"arm": arm, "pooled": {f: summ["pooled"][f] for f in hb.AGREE_FIELDS},
                      "episode_mean": summ["episode_mean"],
                      "attach_results": summ["attach_results"]}, indent=1, default=str))
    return payload


# ------------------------------------------------------------------------------------ CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="new development run (outputs: <run>/<--out-subdir>, default verify/T54)")
    ap.add_argument("--registry", choices=REGISTRIES, default="t53",
                    help="test set: t53 (T54 / T54b) or t54c (16 new episodes, t54c_records)")
    ap.add_argument("--out-subdir", default=None,
                    help=f"default {OUT_SUBDIR_BY_REGISTRY} by registry (e.g. verify/T54b for T54b)")
    ap.add_argument("--set", choices=SETS, required=True)
    ap.add_argument("--stage", choices=STAGES, required=True)
    ap.add_argument("--records", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None, help=f"default all: {list(arms_mod.ARMS)}")
    ap.add_argument("--conditions", nargs="*", default=["current", "hold"], choices=list(CONDITIONS),
                    help="hold_attach = T54b (recorded latch + attach / detach + freeze)")
    ap.add_argument("--attach-reach", choices=["serving", "off"], default="serving",
                    help="hold_attach: the reach gate of the injected attach (serving = recorded limit)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="smoke: frame indices only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--t53-run", "--records-run", dest="t53_run", default=None,
                    help=f"records run of the test registry (t53 default {DEFAULTS['t53_run']}; "
                         "t54c: required)")
    ap.add_argument("--server-roots", nargs="*", default=None,
                    help="server record roots (default: from the records run's layout.json)")
    ap.add_argument("--extra-rows-dir", nargs="*", default=[],
                    help="read-only rows/ directories of other runs (used when a condition's rows "
                         "are not in this run, e.g. T54's current rows for T54b)")
    ap.add_argument("--t49-root", default=DEFAULTS["t49_root"])
    ap.add_argument("--t52-rows", default=DEFAULTS["t52_rows"])
    ap.add_argument("--t51-rows", default=DEFAULTS["t51_rows"])
    ap.add_argument("--t51b-rows", default=DEFAULTS["t51b_rows"])
    ap.add_argument("--t47-gt", default=DEFAULTS["t47_gt"])
    ap.add_argument("--min-target-px", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=SEED_BY_REGISTRY["t53"],
                    help=f"bootstrap seed (pre-fixed by registry: {SEED_BY_REGISTRY}; without --seed the "
                         "t54c registry uses 5417)")
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--extract-ms", default=None,
                    help="JSON {action: ms, noun: ms, noun_k3: ms} per-frame extraction cost (same device)")
    ap.add_argument("--budget-ms", type=float, default=None)
    ap.add_argument("--entry-arms", nargs="*", default=None,
                    help=f"arms the T57 entry table is computed for (default [{arms_mod.MAIN_ARM}]; "
                         "T54c: N B)")
    ap.add_argument("--hold-condition", default="hold", choices=["hold", "hold_attach"],
                    help="condition of the entry table's 시간 유지 row (T54b / T54c: hold_attach)")
    ap.add_argument("--fidelity-arm", default=arms_mod.BASELINE)
    return ap


def resolve_seed(args, argv: Sequence[str]) -> int:
    """Set ``args.seed`` to the registry's pre-fixed seed unless ``--seed`` was given; return the
    pre-fixed one (T54 5407, T54c 5417)."""
    fixed = SEED_BY_REGISTRY[args.registry]
    if not any(a == "--seed" or str(a).startswith("--seed=") for a in argv):
        args.seed = fixed
    return fixed


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.frames is not None and args.stage == "analyze":
        ap.error("--frames is a replay smoke option")
    if args.t53_run is None:
        if args.registry == "t54c" and args.set == "test":
            ap.error("--registry t54c needs --records-run <the T54c run>")
        args.t53_run = DEFAULTS["t53_run"]
    fixed_seed = resolve_seed(args, sys.argv[1:] if argv is None else argv)
    if (args.n_boot, args.seed, args.block, args.min_target_px) != (10000, fixed_seed, 5, 200):
        print(f"NOTE: non-default bootstrap / visibility settings (사전 고정: B=10000, seed {fixed_seed}, "
              "block 5, 200 px) — exploratory only", flush=True)
    subdir = args.out_subdir or OUT_SUBDIR_BY_REGISTRY[args.registry]
    root = out_root(args.run_dir, args.t53_run, subdir=subdir,
                    refuse_records_run=(args.registry == "t53"))
    args._t53_cfg = configure_t53(args.t53_run, args.registry) if args.set == "test" else None
    {"inputs": stage_inputs, "budget": stage_budget, "replay": stage_replay,
     "parity": stage_parity, "analyze": stage_analyze,
     "fidelity": stage_fidelity}[args.stage](args, root)


if __name__ == "__main__":
    main()
