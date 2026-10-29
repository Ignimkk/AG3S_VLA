"""T50 — is the HSR phase probe's insensitivity to the instruction target caused by probe training,
by the representation, or by our 16D LoRA fine-tuning?

    source scripts/setup/env.sh
    export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"   # once
    M=AG3S.ag3s.experiments.studies.t50_hsr_rep
    # T50a (CPU, LoRA features of the T48 run; read-only input)
    .venv-ag3s/bin/python -m $M --model lora --stage sensitivity
    .venv-ag3s/bin/python -m $M --model lora --stage probe --mode valcv
    # T50b (GPU, after CL3): base val + train, LoRA train (C0, C1)
    JAX_PLATFORMS=cpu .venv-openpi-live/bin/python -m $M --stage check-params          # CPU, metadata only
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m $M --model base --stage extract --sets val train
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m $M --model lora --stage extract --sets train
    .venv-ag3s/bin/python -m $M --model base --stage sensitivity
    .venv-ag3s/bin/python -m $M --model base --stage probe --mode valcv
    .venv-ag3s/bin/python -m $M --model base --stage probe --mode trainval
    .venv-ag3s/bin/python -m $M --model lora --stage probe --mode trainval
    .venv-ag3s/bin/python -m $M --stage compare

Output: ``$AG3S_RUN_DIR/verify/T50/<model>/`` (refuses without ``AG3S_RUN_DIR``):

``<tag>/sensitivity/``   sensitivity.json (①), frames.csv (per host frame d_c1, d_c1p, r, d_adj per part)
``<tag>/probe/``         probe_<mode>.json (②, ②′: per part x mix metrics, folds, rule_check),
                         predictions_<mode>.csv
``feat/<ftag>/<set>/``   T50b feature cache (T48 format); ``extract/`` timing.json, extract_meta.json
``../compare/``          compare.json / compare.csv (base vs LoRA, same table)
``../check_params.json`` parameter-tree / transform check (CPU, metadata only)

`<tag>` = ``full`` or ``smoke_first<N>`` (``--max-units N``). Definitions: `t50_core` docstring.
Task: ``AG3S/ag3s/docs/handoff/T50.task.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time
from typing import Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t50_core as core
from AG3S.ag3s.experiments.studies.t48_hsr_counterfactual import git_state, now_kst, write_json

STAGES = ("sensitivity", "probe", "extract", "compare", "check-params")
#: LoRA val features of T48a (verifier run 2026-10-12, `T48.verify.json` run_dir) — read only.
T48_LORA_VAL_FEAT_REL = ("results/development/20261012T155229+0900_218229ac6464_2e66675de9ee/"
                         "verify/T48/feat/full/val")


# ----------------------------------------------------------------------------- plumbing


def out_root() -> pathlib.Path:
    """``$AG3S_RUN_DIR/verify/T50`` — refuses when the run dir is unset (no default location)."""
    from common.run_output import resolve_output_path

    run_dir = os.environ.get("AG3S_RUN_DIR")
    if not run_dir:
        raise SystemExit(
            "AG3S_RUN_DIR is not set. Allocate a run first:\n"
            '  export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"')
    root = resolve_output_path("verify/T50", run_dir=run_dir, default_name="verify/T50")
    root.mkdir(parents=True, exist_ok=True)
    return root


def storage_root(args) -> pathlib.Path:
    if args.storage:
        return pathlib.Path(args.storage)
    from common.workspace import shared_storage_root

    return pathlib.Path(shared_storage_root())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--model", choices=("lora", "base"), default="lora",
                    help="whose features (selects the output dir and the default feature dirs)")
    ap.add_argument("--mode", choices=core.MODES, default="valcv",
                    help="probe: valcv = ② (i) episode CV on val; trainval = ② (ii) train split -> val")
    ap.add_argument("--val-feat", default=None,
                    help="val feature dir (T48 format). Default: lora = the T48 run cache, "
                         "base = <out>/base/feat/full/val")
    ap.add_argument("--train-feat", default=None, help="train feature dir. Default <out>/<model>/feat/full/train")
    ap.add_argument("--parts", nargs="+", default=list(core.PARTS), choices=list(core.PARTS))
    ap.add_argument("--mixes", nargs="+", default=list(core.MIXES), choices=list(core.MIXES))
    ap.add_argument("--seed", type=int, default=0, help="C1 row choice / label shuffle (probe); C2 pairs (extract)")
    ap.add_argument("--max-units", type=int, default=None,
                    help="first N units only (smoke; CPU outputs go to smoke_first<N>/)")
    # extract (GPU)
    ap.add_argument("--sets", nargs="+", default=["val", "train"], choices=["val", "train"])
    ap.add_argument("--max-frames", type=int, default=None, help="extract smoke: first N frames per unit")
    ap.add_argument("--batch", type=int, default=4, help="extract: inputs per prefix call (T48a used 4)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--config", default=None, help="train config (default the 16D LoRA config)")
    ap.add_argument("--checkpoint", default=None, help="LoRA checkpoint (params + norm-stat assets)")
    ap.add_argument("--base-checkpoint", default=None, help="pi05_base dir (default storage cache)")
    ap.add_argument("--storage", default=None, help="default $AG3S_STORAGE_ROOT")
    ap.add_argument("--dataset", default=None)
    args = ap.parse_args(argv)
    for k in ("max_units", "max_frames", "batch"):
        v = getattr(args, k)
        if v is not None and v < 1:
            ap.error(f"--{k.replace('_', '-')} must be >= 1")
    if args.warmup < 0:
        ap.error("--warmup must be >= 0")
    if args.stage == "probe" and args.mode == "valcv" and args.max_units is not None and args.max_units < 2 * core.N_SPLITS:
        ap.error(f"--mode valcv needs >= {2 * core.N_SPLITS} units (nested GroupKFold({core.N_SPLITS}))")
    return args


def cpu_tag(args) -> str:
    return "full" if args.max_units is None else f"smoke_first{args.max_units}"


def feat_tag(args) -> str:
    return "full" if args.max_frames is None else f"max{args.max_frames}"


def val_feat_dir(args, root: pathlib.Path) -> pathlib.Path:
    if args.val_feat:
        return pathlib.Path(args.val_feat)
    if args.model == "lora":
        return storage_root(args) / T48_LORA_VAL_FEAT_REL
    return root / "base" / "feat" / "full" / "val"


def train_feat_dir(args, root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(args.train_feat) if args.train_feat else root / args.model / "feat" / "full" / "train"


def _source(path: pathlib.Path, fs: core.FeatureSet) -> dict:
    return {"dir": str(path), "n_units": len(fs.units), "n_rows": len(fs),
            "rows_per_family": {f: int((fs.variant == f).sum()) for f in sorted(set(fs.variant.tolist()))},
            "plan_digests": fs.digests}


# ----------------------------------------------------------------------------- ① sensitivity


def stage_sensitivity(args, root: pathlib.Path) -> dict:
    vdir = val_feat_dir(args, root)
    fs = core.load_feature_dir(vdir, max_units=args.max_units, families=("C0", "C1", "C1p"))
    t0 = time.time()
    frames = core.sensitivity_frames(fs)
    summary = core.sensitivity_summary(frames)
    out_dir = root / args.model / cpu_tag(args) / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_c0 = int((fs.variant == "C0").sum())
    res = {"created": now_kst(), "git": git_state(), "args": vars(args), "model": args.model,
           "source": _source(vdir, fs), "n_frames": int(len(frames["unit"])), "n_c0_rows": n_c0,
           "n_frames_skipped": n_c0 - int(len(frames["unit"])),
           "definition": {
               "distance": "relative L2 inside the part: ||f_X - f_C0|| / ||f_C0||",
               "d_c1": "mean over the 3 distractor prompts (C1)", "d_c1p": "mean over the 2 other templates (C1p)",
               "r": "d_c1 / d_c1p per frame", "d_adj": "C0 of the next sampled frame (stride 6) vs C0",
               "parts": {k: [v.start, v.stop] for k, v in core.PARTS.items()},
               "aggregation": "per-episode median -> median/IQR over episodes; frame-pooled quartiles"},
           "summary": summary, "wall_s": time.time() - t0}
    write_json(out_dir / "sensitivity.json", res)
    keys = [k for k in frames if k not in ("unit",)]
    with open(out_dir / "frames.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", *keys])
        for i in range(len(frames["unit"])):
            w.writerow([frames["unit"][i], *[_fmt(frames[k][i]) for k in keys]])
    for p in core.PARTS:
        e = summary[p]["r"]["all"]["episode"]
        print(f"[sensitivity {args.model}] {p:5s} r episode median={_fmt(e['median'])} "
              f"IQR=[{_fmt(e['q25'])}, {_fmt(e['q75'])}] (n_ep={e['n_episodes']})")
    return res


def _fmt(x) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (float, np.floating)):
        return "nan" if not np.isfinite(x) else f"{float(x):.6g}"
    return str(x)


# ----------------------------------------------------------------------------- ② probe


def stage_probe(args, root: pathlib.Path) -> dict:
    vdir = val_feat_dir(args, root)
    val = core.load_feature_dir(vdir, max_units=args.max_units)
    train = None
    if args.mode == "trainval":
        tdir = train_feat_dir(args, root)
        train = core.load_feature_dir(tdir, max_units=args.max_units, families=("C0", "C1"))
    out_dir = root / args.model / cpu_tag(args) / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = core.outer_folds(val) if args.mode == "valcv" else None
    res: dict = {"created": now_kst(), "git": git_state(), "args": vars(args), "model": args.model,
                 "mode": args.mode, "source_val": _source(vdir, val),
                 "source_train": None if train is None else _source(train_feat_dir(args, root), train),
                 "outer_folds": folds, "C_grid": list(core.C_GRID), "n_splits": core.N_SPLITS,
                 "solver": {"max_iter": core.MAX_ITER, "tol": core.TOL,
                            "objective": "mean CE + 1/(2 C n) ||coef||^2, StandardScaler, L-BFGS-B (sklearn 1.5)"},
                 "definitions": DEFINITIONS, "cells": {}}
    preds: dict[str, np.ndarray] = {}
    for part in args.parts:
        for mix in args.mixes:
            key = f"{part}__{mix}"
            t0 = time.time()
            if args.mode == "valcv":
                r = core.run_valcv(val, part, mix, seed=args.seed, folds=folds)
            else:
                r = core.run_trainval(train, val, part, mix, seed=args.seed)
            ev = core.evaluate(val, r["pred"])
            res["cells"][key] = {"part": part, "mix": mix, "role": ROLE.get(key, "extra"),
                                 "metrics": ev, "rule_check": core.rule_check(ev), "folds": r["folds"],
                                 "wall_s": time.time() - t0}
            preds[key] = r["prob"]
            c0, c1 = ev["accuracy"].get("C0", {}), ev.get("C1_instr", {})
            print(f"[probe {args.model} {args.mode}] {key:28s} C0={_fmt(c0.get('pooled'))} "
                  f"C1_instr={_fmt(c1.get('pooled'))} ({time.time() - t0:.0f}s)", flush=True)
            write_json(out_dir / f"probe_{args.mode}.json", res)
    with open(out_dir / f"predictions_{args.mode}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "variant", "sub", "host", "clock", "y_host", "gt",
                    *[f"{k}__p_{c}" for k in preds for c in core.CLASSES]])
        for i in range(len(val)):
            w.writerow([val.unit[i], val.variant[i], val.sub[i], int(val.host[i]), int(val.clock[i]),
                        int(val.y_host[i]), int(val.gt[i]),
                        *[f"{x:.6g}" for k in preds for x in preds[k][i]]])
    return res


ROLE = {"full__counterfactual": "primary (②)", "text__counterfactual": "control ②′ text half",
        "image__counterfactual": "control ②′ image half", "full__shuffled": "control ②′ label shuffle",
        "full__standard": "reference: standard probe (C0 rows only)"}

DEFINITIONS = {
    "training_rows": {"counterfactual": "C0 rows (phase class) + 1 C1 row per C0 frame (seeded distractor) labelled pick (1:1)",
                      "standard": "C0 rows only", "shuffled": "as counterfactual, C1 labels uniform random (seeded)"},
    "valcv": "outer GroupKFold(5) over val episodes; inner GroupKFold(5) over the training episodes picks C "
             "(mean held-out accuracy on the training mix); refit; predict all rows of the held-out episodes",
    "trainval": "train split (C0 + C1) -> C by GroupKFold(5) over train episodes -> val episodes",
    "C1_instr": "C1 rows whose host frame is originally place/home: pred == pick",
    "accuracy": "pred == gt per family (C1 gt = pick, C2s/C2i gt = image side)",
    "aggregation": "frame-pooled k/n and per-episode ratio median/IQR",
}


# ----------------------------------------------------------------------------- compare


def stage_compare(args, root: pathlib.Path) -> dict:
    tag = cpu_tag(args)
    out: dict = {"created": now_kst(), "tag": tag, "rows": [], "sensitivity": {}, "val_plan_digests_equal": None}
    digests = {}
    for model in ("lora", "base"):
        sp = root / model / tag / "sensitivity" / "sensitivity.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            out["sensitivity"][model] = {p: s["summary"][p]["r"]["all"] for p in s["summary"]}
            digests[model] = s["source"]["plan_digests"]
        for mode in core.MODES:
            pp = root / model / tag / "probe" / f"probe_{mode}.json"
            if not pp.exists():
                continue
            pr = json.loads(pp.read_text())
            digests.setdefault(model, pr["source_val"]["plan_digests"])
            for key, cell in pr["cells"].items():
                m, rc = cell["metrics"], cell["rule_check"]
                out["rows"].append({
                    "model": model, "mode": mode, "part": cell["part"], "mix": cell["mix"], "role": cell["role"],
                    "C0_pooled": rc["pooled"]["C0"], "C0_episode_median": rc["episode_median"]["C0"],
                    "C1_instr_pooled": rc["pooled"]["C1_instr"],
                    "C1_instr_episode_median": rc["episode_median"]["C1_instr"],
                    "C1_instr_episode_q25": m.get("C1_instr", {}).get("episode_q25"),
                    "C1_instr_episode_q75": m.get("C1_instr", {}).get("episode_q75"),
                    "pass_pooled": rc["pooled"]["pass"], "pass_episode_median": rc["episode_median"]["pass"],
                    "C": [f["C"] for f in cell["folds"]]})
    if "lora" in digests and "base" in digests:
        common = sorted(set(digests["lora"]) & set(digests["base"]))
        out["val_plan_digests_equal"] = {"n_common_units": len(common),
                                         "n_equal": sum(digests["lora"][u] == digests["base"][u] for u in common),
                                         "only_lora": sorted(set(digests["lora"]) - set(digests["base"])),
                                         "only_base": sorted(set(digests["base"]) - set(digests["lora"]))}
    cdir = root / "compare"
    write_json(cdir / f"compare_{tag}.json", out)
    if out["rows"]:
        with open(cdir / f"compare_{tag}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out["rows"][0]))
            w.writeheader()
            w.writerows(out["rows"])
    print(f"[compare] {len(out['rows'])} rows; val digests: {json.dumps(out['val_plan_digests_equal'])}")
    return out


# ----------------------------------------------------------------------------- GPU extract / CPU check


def _paths(args):
    from AG3S.ag3s.experiments.studies import t48_hsr_io as io

    return io.Paths.from_storage(args.storage, dataset=args.dataset, checkpoint=args.checkpoint)


def _config(args) -> str:
    from AG3S.ag3s.experiments.studies import t48_hsr_io as io

    return args.config or io.CONFIG_NAME


def stage_check_params(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t50_extract as ex

    paths = _paths(args)
    base = pathlib.Path(args.base_checkpoint) if args.base_checkpoint else ex.base_checkpoint(args.storage)
    res = ex.check_params(_config(args), paths.checkpoint, base)
    res["created"] = now_kst()
    write_json(root / "check_params.json", res)
    print(f"[check-params] lora loadable={res['lora']['loadable']} base loadable={res['base']['loadable']} "
          f"transforms_equal={res['transforms_equal']} base_has_assets={res['base_has_assets']}")
    return res


def stage_extract(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t48_hsr_io as io
    from AG3S.ag3s.experiments.studies import t50_extract as ex
    from AG3S.ag3s.experiments.studies.t48_hsr_counterfactual import Plan

    procs = ex.gpu_guard(args.allow_busy_gpu)
    if args.model == "lora" and "val" in args.sets:
        print("[extract] note: LoRA val features already exist in the T48 run cache "
              f"({T48_LORA_VAL_FEAT_REL}); extracting them again here", flush=True)
    paths = _paths(args)
    base = pathlib.Path(args.base_checkpoint) if args.base_checkpoint else ex.base_checkpoint(args.storage)
    for k, p in (("dataset", paths.dataset), ("checkpoint", paths.checkpoint)) + \
            ((("base_checkpoint", base),) if args.model == "base" else ()):
        if not p.exists():
            raise SystemExit(f"{k} not found: {p}")
    mdir = root / args.model
    plan_args = argparse.Namespace(max_units=args.max_units, max_frames=args.max_frames, seed=args.seed)
    plan = Plan(plan_args, paths, mdir)
    t_wall = time.time()
    extractor = ex.make_extractor(args.model, config_name=_config(args), lora_checkpoint=str(paths.checkpoint),
                                  base_dir=str(base), batch=args.batch)
    ns = io.norm_stats_state(paths.checkpoint)
    meta: dict = {"created": now_kst(), "git": git_state(), "args": vars(args), "model": args.model,
                  "base_dir": str(base), "lora_checkpoint": str(paths.checkpoint),
                  "gpu_processes_before": procs, "load_s": extractor.load_s, "batch_parity": None}
    meta_path = mdir / "extract" / "extract_meta.json"
    for s in args.sets:
        units = ex.iter_units(plan, s, args.max_units, args.max_frames)
        ex.run_extraction(extractor, units, mdir / "feat" / feat_tag(args) / s,
                          images_fn=lambda u, rows: ex.dataset_images(u, paths.dataset, rows),
                          batch=args.batch, warmup=args.warmup, q01=ns["q01"], q99=ns["q99"], set_name=s,
                          meta=meta, on_unit=lambda m: write_json(meta_path, m))
    timing = {"wall_s": time.time() - t_wall, **extractor.timing(), "memory": extractor.memory(),
              "gpu_processes_before": procs, "model": args.model, "sets": args.sets, "created": now_kst()}
    write_json(mdir / "extract" / "timing.json", timing)
    write_json(meta_path, meta)
    print("[extract] timing", json.dumps(timing), flush=True)
    return timing


# ----------------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = out_root()
    if args.stage == "sensitivity":
        stage_sensitivity(args, root)
    elif args.stage == "probe":
        stage_probe(args, root)
    elif args.stage == "compare":
        stage_compare(args, root)
    elif args.stage == "check-params":
        stage_check_params(args, root)
    else:
        stage_extract(args, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
