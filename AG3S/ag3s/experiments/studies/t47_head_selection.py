"""T47 — re-select the attention head(s) with the localization-head criteria, on 16D and three cameras.

    source scripts/setup/env.sh
    export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"   # once
    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-ag3s/bin/python -m \\
        AG3S.ag3s.experiments.studies.t47_head_selection --stage {gt,select,validate,e2e} [...]

Stages (all raw output under ``$AG3S_RUN_DIR/verify/T47/``; the CLI refuses to run without
``AG3S_RUN_DIR``):

``gt``        MuJoCo replay (osmesa, CPU) of each record -> ``gt_cache/<key>.npz``: per (frame,
              camera) 16x16 class coverage + pixel counts, target position, RGB self-check against
              the recorded policy images. **Rendered once**; later stages read only the cache.
              Needs user approval (T47.task.md "실행 승인").
``select``    Label-free (no GT): S_cam / S_img, spatial entropy, knee tau, selection frequency,
              common and per-camera top-k (k = 1..5), leave-one-record-out. -> ``select/``.
``validate``  Held-out 2D scoring with the GT cache: target mass, lift, peak_on_target, IoU,
              class-mass decomposition, Spearman (selection rank vs held-out IoU / lift), the
              baselines and the GT-oracle (chosen on the *selection* records' GT). -> ``validate/``.
``e2e``       3D grounding on the live serving path (`t47_replay.run_e2e_record`), once per
              attention source. Needs ``select`` (and ``validate`` for the oracle source) and user
              approval (MuJoCo replay). -> ``e2e/``.

Paper: Kang et al., arXiv 2503.06287 (`papers/`). The pure functions live in `t47_head_metrics`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies.t47_head_metrics import (
    CLASS_NAMES, EXCLUDED_LAYERS, cell_list, combine_maps, eligible_cells, head_average,
    image_attention_sum, map_scores, rank_cells, selection_frequency,
    spatial_entropy_batch, spearman)

STEP = "T47"
#: Camera axis of every T47 npz (`policy_record.POLICY_CAMERA_NAMES` order).
POLICY_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
#: The serving cell (`AG3S/trajopt/attention_policy.py:37-41`).
SERVING_CELL = {"denoise": 0, "agg": "last", "layer": 8, "head": 2}
MAX_WORKERS = 4


@dataclasses.dataclass(frozen=True)
class RecordSpec:
    key: str
    npz: str  # under $AG3S_STORAGE_ROOT/assets/attention/
    record: str  # step record, storage-relative
    role: str  # "selection" | "heldout"


#: T47.task.md "입력". ep1800: the npz's `records` is `.../20260924_baseline16d/records/run_0000/
#: run_0000`; storage has no such directory, but `recordings/run_16d_ep1800/meta.json` carries
#: `--record-ag3s outputs/live_test/20260924_baseline16d/records/run_0000` (15 steps) — the same
#: rollout, verified at load time by `check_pairing` (argv path, prompt, frame count, t_step).
RECORDS: dict[str, RecordSpec] = {r.key: r for r in (
    RecordSpec("long", "attention_16d_long.npz",
               "results/workspace/live_test/20260924_long16d/run_0000", "selection"),
    RecordSpec("t2", "attention_16d_t2.npz",
               "results/workspace/live_test/20260925_t2/run_0000/run_0000", "selection"),
    RecordSpec("ep1807", "attention_16d_ep1807.npz",
               "results/workspace/live_test/20260925_ep1807/run_0000", "selection"),
    RecordSpec("headfix", "attention_16d_headfix.npz",
               "results/workspace/live_test/20260925_headfix16d/records/run_0000/run_0000",
               "heldout"),
    RecordSpec("ep1850", "attention_16d_ep1850.npz",
               "results/workspace/live_test/20260925_t1/ep1850/run_0000", "heldout"),
    RecordSpec("ep1900", "attention_16d_ep1900.npz",
               "results/workspace/live_test/20260925_t1/ep1900/run_0000", "heldout"),
    RecordSpec("ep1800", "attention_16d_ep1800.npz", "recordings/run_16d_ep1800", "heldout"),
)}
SELECTION_KEYS = tuple(k for k, r in RECORDS.items() if r.role == "selection")
HELDOUT_KEYS = tuple(k for k, r in RECORDS.items() if r.role == "heldout")


# ------------------------------------------------------------------------------ plumbing


def storage_root() -> pathlib.Path:
    from common.workspace import shared_storage_root

    return shared_storage_root()


def npz_path(key: str) -> pathlib.Path:
    return storage_root() / "assets" / "attention" / RECORDS[key].npz


def record_path(key: str) -> pathlib.Path:
    return storage_root() / RECORDS[key].record


def record_available(key: str) -> bool:
    return (record_path(key) / "meta.json").is_file() and npz_path(key).is_file()


def git_state() -> dict:
    from common.workspace import PROJECT_ROOT

    def run(*a):
        try:
            return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *a], text=True,
                                           stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"head": run("rev-parse", "HEAD").strip(),
            "branch": run("branch", "--show-current").strip(),
            "status_porcelain": [ln for ln in run("status", "--porcelain").splitlines() if ln]}


def out_root() -> pathlib.Path:
    """``$AG3S_RUN_DIR/verify/T47`` — refuses when the run dir is unset (no default location)."""
    from common.run_output import resolve_output_path

    run_dir = os.environ.get("AG3S_RUN_DIR")
    if not run_dir:
        raise SystemExit(
            "AG3S_RUN_DIR is not set. Allocate a run first:\n"
            '  export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"')
    root = resolve_output_path("verify/T47", run_dir=run_dir, default_name="verify/T47")
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pathlib.Path):
            return str(o)
        raise TypeError(type(o))

    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=default,
                               allow_nan=True))


def run_info(stage: str, args: argparse.Namespace) -> dict:
    return {"step": STEP, "stage": stage, "argv": sys.argv,
            "args": {k: v for k, v in vars(args).items()},
            "started_kst": _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(),
            "git": git_state(), "storage_root": str(storage_root()),
            "python": sys.version.split()[0]}


def _episode_index(meta: dict) -> Optional[int]:
    argv = list(meta.get("argv") or [])
    if "--episode-index" in argv:
        try:
            return int(argv[argv.index("--episode-index") + 1])
        except (ValueError, IndexError):
            return None
    return None


def check_pairing(key: str) -> dict:
    """The npz and the step record are the same rollout: argv path, prompt, frame count, t_step.

    Raises on any mismatch — a silently mixed pair is the failure `policy_record.
    require_same_record` exists for. Reads only `t_step` / meta (no images).
    """
    rp = record_path(key)
    meta = json.loads((rp / "meta.json").read_text())
    files = sorted(rp.glob("step_*.npz"))
    t_rec = []
    for f in files:
        with np.load(f) as d:
            t_rec.append(int(d["t_step"]))
    with np.load(npz_path(key), allow_pickle=False) as blob:
        npz_records = str(blob["records"])
        npz_prompt = str(blob["prompt"])
        t_npz = [int(t) for t in blob["t_step"]]
        n_npz = int(blob["attention"].shape[0])
    argv = list(meta.get("argv") or [])
    rec_arg = argv[argv.index("--record-ag3s") + 1] if "--record-ag3s" in argv else None
    # The writer puts each rollout in `<--record-ag3s>/run_XXXX`; the npz names that directory.
    # Storage may have renamed the leaf (`run_0000` -> `run_16d_ep1800`), so the *parent* must
    # match exactly and a different leaf is reported, not refused (prompt / count / t_step still
    # have to match).
    npz_parent, npz_leaf = npz_records.rstrip("/").rsplit("/", 1)
    want_parent = None if rec_arg is None else rec_arg.rstrip("/").removeprefix("./")
    path_ok = want_parent is not None and npz_parent.endswith("/" + want_parent)
    renamed = path_ok and npz_leaf != rp.name
    problems = []
    if not path_ok:
        problems.append(f"npz records={npz_records!r} vs meta --record-ag3s={rec_arg!r}/{rp.name}")
    if npz_prompt != meta.get("prompt"):
        problems.append(f"prompt {npz_prompt!r} vs {meta.get('prompt')!r}")
    if n_npz != len(files) or t_npz != t_rec:
        problems.append(f"frames npz {n_npz} t_step {t_npz[:4]}... vs record {len(files)} "
                        f"t_step {t_rec[:4]}...")
    if problems:
        raise SystemExit(f"[{key}] attention npz and step record are not the same rollout:\n  "
                         + "\n  ".join(problems))
    return {"key": key, "role": RECORDS[key].role, "npz": str(npz_path(key)),
            "record": str(rp), "npz_records_field": npz_records, "prompt": npz_prompt,
            "n_frames": n_npz, "episode_index": _episode_index(meta),
            "pairing": "argv path + prompt + frame count + t_step all match",
            "record_dir_renamed": (f"{npz_leaf} -> {rp.name}" if renamed else None)}


def target_of(key: str) -> str:
    from AG3S.ag3s.experiments.reports.attention_report import target_from_prompt

    with np.load(npz_path(key), allow_pickle=False) as blob:
        prompt = str(blob["prompt"])
    target = target_from_prompt(prompt)
    if target is None:
        raise SystemExit(f"[{key}] cannot parse the target from prompt {prompt!r}")
    return target


def load_attention(key: str):
    """``(attention float16 (F, D, A, L, H, C, 16, 16), denoise list, agg list, cameras list)``."""
    with np.load(npz_path(key), allow_pickle=False) as blob:
        att = np.asarray(blob["attention"])
        denoise = [int(d) for d in blob["denoise_steps"]]
        aggs = [str(a) for a in blob["aggregations"]]
        cams = [str(c) for c in blob["cameras"]]
    if att.ndim != 8 or att.shape[-2:] != (16, 16):
        raise SystemExit(f"[{key}] unexpected attention shape {att.shape}")
    if tuple(cams) != POLICY_CAMERAS:
        raise SystemExit(f"[{key}] camera axis {cams} != {POLICY_CAMERAS}")
    return att, denoise, aggs, cams


def resolve_keys(requested: Optional[Sequence[str]], default: Sequence[str]) -> list[str]:
    keys = list(requested or default)
    unknown = [k for k in keys if k not in RECORDS]
    if unknown:
        raise SystemExit(f"unknown record key(s) {unknown}; have {sorted(RECORDS)}")
    return keys


def available_keys(keys: Sequence[str], *, log=print) -> tuple[list[str], list[dict]]:
    ok, missing = [], []
    for k in keys:
        if record_available(k):
            ok.append(k)
        else:
            missing.append({"key": k, "npz": str(npz_path(k)), "record": str(record_path(k)),
                            "why": "npz or step record (meta.json) missing in storage"})
            log(f"[{k}] excluded: {missing[-1]['why']} ({record_path(k)})")
    return ok, missing


def setting_name(d: int, agg: str) -> str:
    return f"d{int(d)}_{agg}"


# ------------------------------------------------------------------------------ stage gt


def gt_cache_path(root: pathlib.Path, key: str) -> pathlib.Path:
    return root / "gt_cache" / f"{key}.npz"


def _gt_one(key: str, cache: str, rgb_check_frames: int) -> dict:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.studies.t47_replay import extract_gt

    info = check_pairing(key)
    t0 = time.time()
    arrays = extract_gt(record_path(key), target=target_of(key),
                        rgb_check_frames=rgb_check_frames,
                        log=lambda s: print(f"[{key}]{s}", flush=True))
    arrays["record_key"] = np.asarray(key)
    arrays["pairing"] = np.asarray(json.dumps(info))
    pathlib.Path(cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **arrays)
    tpx = arrays["class_px"][:, :, 0]
    return {**info, "cache": cache, "seconds": time.time() - t0,
            "target": str(arrays["target"]),
            "target_visible_ge200px_per_camera": {c: int((tpx[:, i] >= 200).sum())
                                                  for i, c in enumerate(POLICY_CAMERAS)},
            "rgb_check_index": arrays["rgb_check_index"].tolist(),
            "rgb_mad": arrays["rgb_mad"].tolist(),
            "rgb_frac_gt16": arrays["rgb_frac_gt16"].tolist()}


def stage_gt(args, root: pathlib.Path) -> dict:
    keys, missing = available_keys(resolve_keys(args.records, list(RECORDS)))
    todo, skipped = [], []
    for k in keys:
        cache = gt_cache_path(root, k)
        if cache.exists() and not args.force:
            skipped.append(k)
            print(f"[{k}] cache exists, skipping ({cache}) — --force to re-render")
        else:
            todo.append(k)
    for k in keys:  # pairing is cheap; fail before rendering anything
        check_pairing(k)
    results = []
    if args.workers > 1 and len(todo) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(todo)),
                                    mp_context=mp.get_context("spawn")) as ex:
            futs = [ex.submit(_gt_one, k, str(gt_cache_path(root, k)), args.rgb_check_frames)
                    for k in todo]
            results = [f.result() for f in futs]
    else:
        results = [_gt_one(k, str(gt_cache_path(root, k)), args.rgb_check_frames) for k in todo]
    summary = {"rendered": results, "skipped_cached": skipped, "excluded": missing}
    write_json(root / "gt_cache" / "gt_summary.json", {**run_info("gt", args), **summary})
    return summary


def load_gt(root: pathlib.Path, key: str) -> dict:
    path = gt_cache_path(root, key)
    if not path.exists():
        raise SystemExit(f"[{key}] no GT cache at {path} — run --stage gt first")
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


# -------------------------------------------------------------------------- stage select


def record_statistics(att: np.ndarray, *, chunk_frames: int = 8) -> tuple[np.ndarray, np.ndarray,
                                                                         np.ndarray]:
    """``S_cam``, spatial entropy and component count, each ``(F, D, A, L, H, C)``."""
    s_cam = np.zeros(att.shape[:6], np.float64)
    ent = np.zeros(att.shape[:6], np.float64)
    ncomp = np.zeros(att.shape[:6], np.int16)
    for f0 in range(0, att.shape[0], chunk_frames):
        block = np.asarray(att[f0:f0 + chunk_frames], np.float64)
        s, _ = image_attention_sum(block)
        s_cam[f0:f0 + chunk_frames] = s
        e, c = spatial_entropy_batch(block)
        ent[f0:f0 + chunk_frames] = e
        ncomp[f0:f0 + chunk_frames] = c
    return s_cam, ent, ncomp


def _cells_view(x: np.ndarray, cells: list[tuple[int, int]]) -> np.ndarray:
    """``(F, L, H, C)`` -> ``(F, C, n_cells)`` in `cells` order."""
    ls = np.asarray([c[0] for c in cells])
    hs = np.asarray([c[1] for c in cells])
    return np.moveaxis(x[:, ls, hs, :], 1, 2)


def _select_block(ent: np.ndarray, s: np.ndarray, elig: np.ndarray, cells, *, n_lowest: int,
                  kmax: int) -> dict:
    """One sample pool -> frequency, ranking, top-k. `ent`/`s`/`elig` are ``(S, n_cells)``."""
    sf = selection_frequency(ent, elig, n_lowest=n_lowest)
    import warnings

    with warnings.catch_warnings():  # a never-eligible cell has no entropy mean -> NaN, on purpose
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_e = np.nanmean(np.where(elig, ent, np.nan), axis=0)
    mean_s = s.mean(axis=0)
    order = rank_cells(sf.frequency, mean_entropy=mean_e, mean_sum=mean_s)
    rank_pos = np.empty(len(cells), np.int64)
    rank_pos[order] = np.arange(1, len(cells) + 1)
    return {
        "n_samples": sf.n_samples,
        "n_samples_without_candidate": int((sf.n_selected == 0).sum()),
        "frequency": sf.frequency, "counts": sf.counts, "rank": rank_pos,
        "mean_entropy_eligible": mean_e, "mean_S": mean_s,
        "eligible_rate": elig.mean(axis=0),
        "topk": {str(k): [list(cells[i]) for i in order[:k]] for k in range(1, kmax + 1)},
        "ranking": [{"layer": cells[i][0], "head": cells[i][1],
                     "frequency": float(sf.frequency[i]), "count": int(sf.counts[i]),
                     "mean_entropy_eligible": float(mean_e[i]), "mean_S": float(mean_s[i]),
                     "eligible_rate": float(elig[:, i].mean())} for i in order[:20]],
    }


def select_from_stats(stats: dict[str, dict], keys: Sequence[str], d: int, a: int, cells,
                      *, tau_source: str, tau_method: str, n_lowest: int, kmax: int) -> dict:
    """Common + per-camera selection over the records in `keys` for denoise index `d`, agg `a`."""
    E, S, EL = [], [], []
    taus, found = [], []
    for k in keys:
        ent = _cells_view(stats[k]["entropy"][:, d, a], cells)  # (F, C, n)
        s = _cells_view(stats[k]["S_cam"][:, d, a], cells)
        if tau_source == "cam":
            el, tau, fd = eligible_cells(s, method=tau_method, axis=-1)  # tau (F, C)
        elif tau_source == "img":
            s_img = s.sum(axis=1)  # (F, n)
            el1, tau1, fd1 = eligible_cells(s_img, method=tau_method, axis=-1)
            el = np.repeat(el1[:, None, :], s.shape[1], axis=1)
            tau = np.repeat(tau1[:, None], s.shape[1], axis=1)
            fd = np.repeat(fd1[:, None], s.shape[1], axis=1)
        else:
            raise ValueError(tau_source)
        E.append(ent), S.append(s), EL.append(el), taus.append(tau), found.append(fd)
    E, S, EL = np.concatenate(E), np.concatenate(S), np.concatenate(EL)
    tau_all, found_all = np.concatenate(taus), np.concatenate(found)
    n_c = E.shape[1]
    common = _select_block(E.reshape(-1, E.shape[2]), S.reshape(-1, S.shape[2]),
                           EL.reshape(-1, EL.shape[2]), cells, n_lowest=n_lowest, kmax=kmax)
    per_cam = {POLICY_CAMERAS[c]: _select_block(E[:, c], S[:, c], EL[:, c], cells,
                                                n_lowest=n_lowest, kmax=kmax)
               for c in range(n_c)}
    tau_summary = {POLICY_CAMERAS[c]: {
        "tau_mean": float(tau_all[:, c].mean()), "tau_median": float(np.median(tau_all[:, c])),
        "n_eligible_mean": float(EL[:, c].sum(axis=1).mean()),
        "knee_found_rate": float(found_all[:, c].mean())} for c in range(n_c)}
    return {"common": common, "per_camera": per_cam, "tau_summary": tau_summary,
            "tau": tau_all, "eligible": EL}


def _jsonable_block(block: dict) -> dict:
    drop = {"frequency", "counts", "rank", "mean_entropy_eligible", "mean_S", "eligible_rate"}
    return {k: v for k, v in block.items() if k not in drop}


def stage_select(args, root: pathlib.Path) -> dict:
    keys, missing = available_keys(resolve_keys(args.records, SELECTION_KEYS))
    if not keys:
        raise SystemExit("no selection record available")
    pairing = [check_pairing(k) for k in keys]
    stats: dict[str, dict] = {}
    denoise = aggs = None
    for k in keys:
        att, dn, ag, _ = load_attention(k)
        if args.max_frames:
            att = att[: args.max_frames]
        denoise, aggs = dn, ag
        t0 = time.time()
        s_cam, ent, ncomp = record_statistics(att)
        stats[k] = {"S_cam": s_cam, "entropy": ent, "ncomp": ncomp}
        print(f"[{k}] S / entropy over {att.shape[0]} frames x {np.prod(att.shape[1:6])} maps "
              f"({time.time() - t0:.1f}s)")
    n_layers, n_heads = stats[keys[0]]["S_cam"].shape[3:5]
    cells = cell_list(n_layers, n_heads)
    arrays: dict[str, np.ndarray] = {"cells": np.asarray(cells, np.int64),
                                     "cameras": np.asarray(POLICY_CAMERAS),
                                     "denoise_steps": np.asarray(denoise),
                                     "aggregations": np.asarray(aggs)}
    for k in keys:
        arrays[f"S_cam__{k}"] = stats[k]["S_cam"].astype(np.float32)
        arrays[f"entropy__{k}"] = stats[k]["entropy"].astype(np.float32)
        arrays[f"ncomp__{k}"] = stats[k]["ncomp"]
    settings: dict[str, dict] = {}
    for di, d in enumerate(denoise):
        if d not in args.denoise:
            continue
        for ai, agg in enumerate(aggs):
            name = setting_name(d, agg)
            settings[name] = {"denoise": d, "agg": agg, "tau_source": {}}
            for src in ("cam", "img"):
                res = select_from_stats(stats, keys, di, ai, cells, tau_source=src,
                                        tau_method=args.tau_method, n_lowest=args.n_lowest,
                                        kmax=args.kmax)
                # the other knee definition, reported next to the one used (not used to select)
                other = "curvature" if args.tau_method == "kneedle" else "kneedle"
                alt = select_from_stats(stats, keys, di, ai, cells, tau_source=src,
                                        tau_method=other, n_lowest=args.n_lowest, kmax=args.kmax)
                loro = []
                if len(keys) > 1:
                    full_c = set(map(tuple, res["common"]["topk"]["3"]))
                    for out in keys:
                        rest = [k for k in keys if k != out]
                        r2 = select_from_stats(stats, rest, di, ai, cells, tau_source=src,
                                               tau_method=args.tau_method,
                                               n_lowest=args.n_lowest, kmax=args.kmax)
                        top_c = set(map(tuple, r2["common"]["topk"]["3"]))
                        cams = {}
                        for cam in POLICY_CAMERAS:
                            full = set(map(tuple, res["per_camera"][cam]["topk"]["3"]))
                            got = set(map(tuple, r2["per_camera"][cam]["topk"]["3"]))
                            cams[cam] = {"top3": sorted(got), "overlap_with_full": len(full & got),
                                         "same_set": got == full}
                        loro.append({"left_out": out,
                                     "common": {"top3": sorted(top_c),
                                                "overlap_with_full": len(full_c & top_c),
                                                "same_set": top_c == full_c},
                                     "per_camera": cams})
                per_record = {}
                for k in keys:
                    r1 = select_from_stats(stats, [k], di, ai, cells, tau_source=src,
                                           tau_method=args.tau_method, n_lowest=args.n_lowest,
                                           kmax=args.kmax)
                    per_record[k] = {"common_top10": r1["common"]["ranking"][:10],
                                     "per_camera_top3": {c: r1["per_camera"][c]["topk"]["3"]
                                                         for c in POLICY_CAMERAS}}
                settings[name]["tau_source"][src] = {
                    "common": _jsonable_block(res["common"]),
                    "per_camera": {c: _jsonable_block(b) for c, b in res["per_camera"].items()},
                    "tau_summary": res["tau_summary"],
                    f"tau_summary_{other}": alt["tau_summary"],
                    f"common_top3_with_{other}": alt["common"]["topk"]["3"],
                    f"per_camera_top3_with_{other}": {c: alt["per_camera"][c]["topk"]["3"]
                                                      for c in POLICY_CAMERAS},
                    "leave_one_record_out": loro,
                    "per_record": per_record,
                }
                pre = f"{name}__{src}"
                arrays[f"{pre}__tau"] = res["tau"]
                arrays[f"{pre}__eligible"] = res["eligible"]
                arrays[f"{pre}__tau_{other}"] = alt["tau"]
                for field in ("frequency", "counts", "rank", "mean_entropy_eligible", "mean_S",
                              "eligible_rate"):
                    arrays[f"{pre}__common__{field}"] = res["common"][field]
                    arrays[f"{pre}__per_camera__{field}"] = np.stack(
                        [res["per_camera"][c][field] for c in POLICY_CAMERAS])
            print(f"[select] {name}: common top3 (cam tau) = "
                  f"{settings[name]['tau_source']['cam']['common']['topk']['3']}")
    payload = {
        **run_info("select", args),
        "records": pairing, "excluded": missing,
        "policy": {"excluded_layers": list(EXCLUDED_LAYERS), "n_cells": len(cells),
                   "n_lowest": args.n_lowest, "tau_method": args.tau_method,
                   "tau_source_primary": "cam", "tau_source_variant": "img", "kmax": args.kmax,
                   "sample_definition": {
                       "common": "every (frame, camera) of every selection record",
                       "per_camera": "every frame of every selection record, that camera only"},
                   "max_frames": args.max_frames, "uses_gt": False},
        "settings": settings,
    }
    out = root / "select"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "selection.json", payload)
    np.savez_compressed(out / "selection_arrays.npz", **arrays)
    print(f"[select] wrote {out / 'selection.json'} and selection_arrays.npz")
    return payload


def load_selection(root: pathlib.Path) -> dict:
    path = root / "select" / "selection.json"
    if not path.exists():
        raise SystemExit(f"no {path} — run --stage select first")
    return json.loads(path.read_text())


# ------------------------------------------------------------------------ stage validate


def source_maps(att: np.ndarray, denoise: list[int], aggs: list[str], selection: dict, *,
                setting: str, sources: Sequence[str], sigma_by_source: Optional[dict] = None,
                oracle: Optional[dict] = None) -> dict[str, np.ndarray]:
    """Named ``(F, C, 16, 16)`` maps for one record.

    Source names: ``l8h2`` (serving cell, always d0/last) · ``head_avg`` · ``uniform`` ·
    ``topk_common_k{k}_s{sigma}_{cam|img}`` · ``topk_camera_k{k}_s{sigma}_{cam|img}`` ·
    ``oracle_top{1|3}_s{sigma}`` (needs `oracle`) · ``cell_L{l}H{h}`` (single cell, sigma 0).
    """
    st = selection["settings"][setting]
    di, ai = denoise.index(int(st["denoise"])), aggs.index(st["agg"])
    block = att[:, di, ai]  # (F, L, H, C, 16, 16)
    block_c = np.moveaxis(block, 3, 1)  # (F, C, L, H, 16, 16)
    out: dict[str, np.ndarray] = {}
    for name in sources:
        if name == "l8h2":
            sd, sa = denoise.index(SERVING_CELL["denoise"]), aggs.index(SERVING_CELL["agg"])
            out[name] = np.asarray(att[:, sd, sa, SERVING_CELL["layer"], SERVING_CELL["head"]],
                                   np.float64)
        elif name == "head_avg":
            out[name] = head_average(np.asarray(block_c, np.float64))
        elif name == "uniform":
            out[name] = np.ones(block_c.shape[:2] + block_c.shape[-2:], np.float64)
        elif (m := re.fullmatch(r"topk_(common|camera)_k(\d+)_s([\d.]+)_(cam|img)", name)):
            kind, k, sigma, src = m.group(1), m.group(2), float(m.group(3)), m.group(4)
            sel = st["tau_source"][src]
            if kind == "common":
                out[name] = combine_maps(block_c, [tuple(c) for c in sel["common"]["topk"][k]],
                                         sigma)
            else:
                out[name] = np.stack([
                    combine_maps(block_c[:, ci],
                                 [tuple(c) for c in sel["per_camera"][cam]["topk"][k]], sigma)
                    for ci, cam in enumerate(POLICY_CAMERAS)], axis=1)
        elif (m := re.fullmatch(r"oracle_top(\d+)_s([\d.]+)", name)):
            if oracle is None:
                raise SystemExit(f"source {name} needs the oracle (run --stage validate first)")
            cells = [tuple(c) for c in oracle["settings"][setting][f"top{m.group(1)}"]]
            out[name] = combine_maps(block_c, cells, float(m.group(2)))
        elif (m := re.fullmatch(r"cell_L(\d+)H(\d+)", name)):
            out[name] = np.asarray(block_c[:, :, int(m.group(1)), int(m.group(2))], np.float64)
        else:
            raise SystemExit(f"unknown source {name!r}")
    return out


def _flat_scores(maps: np.ndarray, gt: dict) -> dict:
    f, c = maps.shape[:2]
    return map_scores(maps.reshape((f * c,) + maps.shape[2:]),
                      np.asarray(gt["class_cov"], np.float64).reshape((f * c,) +
                                                                       gt["class_cov"].shape[2:]))


def _aggregate(scores: dict, keep: np.ndarray, cam_of: np.ndarray, rec_of: np.ndarray,
               rec_names: Sequence[str]) -> dict:
    def summ(mask):
        n = int(mask.sum())
        res = {"n": n}
        for f in ("target_mass", "lift", "peak_on_target", "iou@0.5", "iou@0.25"):
            v = scores[f][mask]
            fin = np.isfinite(v)
            res[f] = float(v[fin].mean()) if fin.any() else float("nan")
            if f.startswith("iou"):
                res[f"{f}_n_gt_empty"] = int((~fin).sum())
        cm = scores["class_mass"][mask]
        res["class_mass"] = {c: (float(np.nanmean(cm[:, i])) if n else float("nan"))
                             for i, c in enumerate(CLASS_NAMES)}
        return res

    out = {"pooled": summ(keep)}
    out["per_camera"] = {cam: summ(keep & (cam_of == i)) for i, cam in enumerate(POLICY_CAMERAS)}
    out["per_record"] = {r: summ(keep & (rec_of == i)) for i, r in enumerate(rec_names)}
    return out


def compute_oracle(root: pathlib.Path, selection: dict, keys: Sequence[str], settings, *,
                   min_target_px: int) -> dict:
    """GT-oracle cells on the **selection** records: rank by pooled peak_on_target (3 cameras),
    tie-break by mean lift. Upper reference only — never a candidate."""
    res = {"settings": {}, "records": list(keys), "min_target_px": min_target_px,
           "criterion": "peak_on_target over kept (frame, camera) of the selection records; "
                        "tie-break mean lift"}
    cache = {k: (load_attention(k), load_gt(root, k)) for k in keys}
    for setting in settings:
        st = selection["settings"][setting]
        n_l, n_h = None, None
        per_cell_peak, per_cell_lift, cells = None, None, None
        for k, ((att, dn, ag, _), gt) in cache.items():
            di, ai = dn.index(int(st["denoise"])), ag.index(st["agg"])
            block_c = np.moveaxis(np.asarray(att[:, di, ai], np.float64), 3, 1)
            n_l, n_h = block_c.shape[2:4]
            cells = cell_list(n_l, n_h)
            keep = (np.asarray(gt["class_px"])[:, :, 0] >= min_target_px).reshape(-1)
            if per_cell_peak is None:
                per_cell_peak = [[] for _ in cells]
                per_cell_lift = [[] for _ in cells]
            for ci, (l, h) in enumerate(cells):
                sc = _flat_scores(block_c[:, :, l, h], gt)
                per_cell_peak[ci].append(sc["peak_on_target"][keep])
                per_cell_lift[ci].append(sc["lift"][keep])
        peak = np.asarray([np.concatenate(p).mean() if sum(map(len, p)) else np.nan
                           for p in per_cell_peak])
        lift = np.asarray([np.nanmean(np.concatenate(p)) if sum(map(len, p)) else np.nan
                           for p in per_cell_lift])
        order = np.lexsort((np.arange(len(cells)), -np.nan_to_num(lift, nan=-np.inf),
                            -np.nan_to_num(peak, nan=-np.inf)))
        res["settings"][setting] = {
            "top1": [list(cells[order[0]])], "top3": [list(cells[i]) for i in order[:3]],
            "ranking": [{"layer": cells[i][0], "head": cells[i][1],
                         "peak_on_target": float(peak[i]), "lift": float(lift[i])}
                        for i in order[:10]],
        }
    return res


def stage_validate(args, root: pathlib.Path) -> dict:
    selection = load_selection(root)
    held, missing = available_keys(resolve_keys(args.records, HELDOUT_KEYS))
    sel_keys = [r["key"] for r in selection["records"]]
    settings = [s for s in selection["settings"]
                if selection["settings"][s]["denoise"] in args.denoise
                and selection["settings"][s]["agg"] in args.aggs]
    if not settings:
        raise SystemExit(f"no selection setting matches --denoise {args.denoise} --aggs {args.aggs}")
    oracle = compute_oracle(root, selection, sel_keys, settings, min_target_px=args.min_target_px)
    write_json(root / "validate" / "oracle.json", {**run_info("validate-oracle", args), **oracle})
    held_data = {k: (load_attention(k), load_gt(root, k), check_pairing(k)) for k in held}
    payload: dict[str, Any] = {**run_info("validate", args), "heldout": [v[2] for v in held_data.values()],
                               "excluded": missing, "min_target_px": args.min_target_px,
                               "sigmas": args.sigmas, "kmax": args.kmax, "settings": {}}
    for setting in settings:
        st = selection["settings"][setting]
        names = ["l8h2", "head_avg", "uniform"]
        for src in ("cam", "img"):
            for k in range(1, args.kmax + 1):
                for s in args.sigmas:
                    names += [f"topk_common_k{k}_s{s:g}_{src}", f"topk_camera_k{k}_s{s:g}_{src}"]
        for s in args.sigmas:
            names += [f"oracle_top1_s{s:g}", f"oracle_top3_s{s:g}"]
        first_att = next(iter(held_data.values()))[0][0]
        all_cells = cell_list(first_att.shape[3], first_att.shape[4])
        cell_names = [f"cell_L{l}H{h}" for l, h in all_cells]
        scores: dict[str, list] = {n: [] for n in names + cell_names}
        keep_l, cam_l, rec_l, frame_l = [], [], [], []
        excluded_counts = {}
        for ri, (k, ((att, dn, ag, _), gt, _info)) in enumerate(held_data.items()):
            maps = source_maps(att, dn, ag, selection, setting=setting, sources=names + cell_names,
                               oracle=oracle)
            f = att.shape[0]
            tpx = np.asarray(gt["class_px"])[:, :, 0]
            keep_l.append((tpx >= args.min_target_px).reshape(-1))
            cam_l.append(np.tile(np.arange(len(POLICY_CAMERAS)), f))
            rec_l.append(np.full(f * len(POLICY_CAMERAS), ri))
            frame_l.append(np.repeat(np.arange(f), len(POLICY_CAMERAS)))
            excluded_counts[k] = {c: int((tpx[:, i] < args.min_target_px).sum())
                                  for i, c in enumerate(POLICY_CAMERAS)}
            for n in names + cell_names:
                scores[n].append(_flat_scores(maps[n], gt))
            # combined maps of every named source (not the 128 single cells) for the figures
            (root / "validate").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(root / "validate" / f"maps__{setting}__{k}.npz",
                                **{n: maps[n].astype(np.float32) for n in names})
        keep, cam_of = np.concatenate(keep_l), np.concatenate(cam_l)
        rec_of, frame_of = np.concatenate(rec_l), np.concatenate(frame_l)
        merged = {n: {f: np.concatenate([s[f] for s in scores[n]]) for f in scores[n][0]}
                  for n in scores}
        rec_names = list(held_data)
        agg_by_source = {n: _aggregate(merged[n], keep, cam_of, rec_of, rec_names) for n in names}
        # Spearman: selection frequency / rank (common, per camera) vs held-out per-cell metrics.
        cell_metric = {n: _aggregate(merged[n], keep, cam_of, rec_of, rec_names) for n in cell_names}
        sp = {}
        arr_path = root / "select" / "selection_arrays.npz"
        with np.load(arr_path, allow_pickle=False) as sa:
            for src in ("cam", "img"):
                pre = f"{setting}__{src}"
                freq_c, rank_c = sa[f"{pre}__common__frequency"], sa[f"{pre}__common__rank"]
                freq_p, rank_p = sa[f"{pre}__per_camera__frequency"], sa[f"{pre}__per_camera__rank"]
                entry = {"common": {}, "per_camera": {}}
                for metric in ("iou@0.5", "iou@0.25", "lift", "peak_on_target", "target_mass"):
                    y = np.asarray([cell_metric[c]["pooled"][metric] for c in cell_names])
                    entry["common"][metric] = {"freq": spearman(freq_c, y),
                                               "rank": spearman(rank_c, y)}
                    for ci, cam in enumerate(POLICY_CAMERAS):
                        yc = np.asarray([cell_metric[c]["per_camera"][cam][metric]
                                         for c in cell_names])
                        entry["per_camera"].setdefault(cam, {})[metric] = {
                            "freq": spearman(freq_p[ci], yc), "rank": spearman(rank_p[ci], yc)}
                sp[src] = entry
        payload["settings"][setting] = {
            "denoise": st["denoise"], "agg": st["agg"],
            "n_samples_total": int(keep.size), "n_samples_kept": int(keep.sum()),
            "excluded_lt_min_target_px": excluded_counts,
            "sources": agg_by_source,
            "source_cells": {
                "l8h2": [[SERVING_CELL["layer"], SERVING_CELL["head"]]],
                **{f"topk_common_k{k}_{src}": st["tau_source"][src]["common"]["topk"][str(k)]
                   for src in ("cam", "img") for k in range(1, args.kmax + 1)},
                **{f"topk_camera_k{k}_{src}": {c: st["tau_source"][src]["per_camera"][c]["topk"][str(k)]
                                               for c in POLICY_CAMERAS}
                   for src in ("cam", "img") for k in range(1, args.kmax + 1)},
                "oracle_top1": oracle["settings"][setting]["top1"],
                "oracle_top3": oracle["settings"][setting]["top3"]},
            "single_cell": {n: {"pooled": cell_metric[n]["pooled"]} for n in cell_names},
            "spearman": sp,
        }
        per_sample = {"keep": keep, "camera": cam_of, "record": rec_of, "frame": frame_of,
                      "record_names": np.asarray(rec_names), "class_names": np.asarray(CLASS_NAMES)}
        for n in names:
            for f, v in merged[n].items():
                per_sample[f"{n}__{f}"] = v
        np.savez_compressed(root / "validate" / f"per_sample__{setting}.npz", **per_sample)
        print(f"[validate] {setting}: kept {int(keep.sum())}/{keep.size} (frame, camera) samples")
    write_json(root / "validate" / "validate.json", payload)
    print(f"[validate] wrote {root / 'validate' / 'validate.json'}")
    return payload


# ----------------------------------------------------------------------------- stage e2e


E2E_DEFAULT_SOURCES = ("l8h2", "topk_common", "topk_camera", "head_avg", "oracle")


def e2e_source_names(requested: Sequence[str], *, k: int, sigma: float, tau_source: str
                     ) -> dict[str, str]:
    """Short e2e name -> `source_maps` name."""
    table = {"l8h2": "l8h2", "head_avg": "head_avg", "uniform": "uniform",
             "topk_common": f"topk_common_k{k}_s{sigma:g}_{tau_source}",
             "topk_camera": f"topk_camera_k{k}_s{sigma:g}_{tau_source}",
             "oracle": f"oracle_top{k}_s{sigma:g}"}
    bad = [s for s in requested if s not in table]
    if bad:
        raise SystemExit(f"unknown e2e source(s) {bad}; have {sorted(table)}")
    return {s: table[s] for s in requested}


def _e2e_one(key: str, out_dir: str, sources: dict, frames, opts: dict) -> dict:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.studies.t47_replay import run_e2e_record, summarize_e2e

    rows = run_e2e_record(record_path(key), target=target_of(key), sources=sources,
                          frames=frames, log=lambda s: print(f"[{key}]{s}", flush=True), **opts)
    for r in rows:
        r["record"] = key
    path = pathlib.Path(out_dir) / f"rows__{key}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return {"key": key, "rows": str(path), "summary": summarize_e2e(rows), "n_rows": len(rows)}


def stage_e2e(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies.t47_replay import summarize_e2e

    selection = load_selection(root)
    setting = setting_name(args.e2e_denoise, args.e2e_agg)
    if setting not in selection["settings"]:
        raise SystemExit(f"selection has no setting {setting}")
    names = e2e_source_names(args.sources, k=args.k, sigma=args.sigma, tau_source="cam")
    oracle = None
    if "oracle" in names:
        opath = root / "validate" / "oracle.json"
        if not opath.exists():
            raise SystemExit(f"source 'oracle' needs {opath} — run --stage validate first "
                             "(or drop it with --sources)")
        oracle = json.loads(opath.read_text())
        if setting not in oracle["settings"]:
            raise SystemExit(f"oracle.json has no setting {setting}")
    keys, missing = available_keys(resolve_keys(args.records, HELDOUT_KEYS))
    if "oracle" in names and any(RECORDS[k].role == "selection" for k in keys):
        raise SystemExit("the GT-oracle is chosen on the selection records — it is not scored "
                         "there (T47.task.md #7); drop 'oracle' from --sources")
    pairing = [check_pairing(k) for k in keys]
    jobs = []
    for k in keys:
        att, dn, ag, _ = load_attention(k)
        maps = source_maps(att, dn, ag, selection, setting=setting,
                           sources=list(names.values()), oracle=oracle)
        frames = None if not args.max_frames else list(range(min(args.max_frames, att.shape[0])))
        jobs.append((k, {short: maps[full].astype(np.float32) for short, full in names.items()},
                     frames))
    opts = {"voxel": args.voxel, "range_max": args.range_max, "links": args.links,
            "esdf_backend": args.esdf_backend, "on_target_radius": args.on_target_radius,
            "label_radius": args.label_radius}
    out_dir = root / "e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.workers > 1 and len(jobs) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                    mp_context=mp.get_context("spawn")) as ex:
            results = [f.result() for f in [ex.submit(_e2e_one, k, str(out_dir), s, fr, opts)
                                            for k, s, fr in jobs]]
    else:
        results = [_e2e_one(k, str(out_dir), s, fr, opts) for k, s, fr in jobs]
    all_rows = []
    for r in results:
        with open(r["rows"]) as fh:
            all_rows += [json.loads(line) for line in fh]
    payload = {**run_info("e2e", args), "records": pairing, "excluded": missing,
               "setting": setting, "sources": names,
               "source_cells": {
                   "l8h2": [[SERVING_CELL["layer"], SERVING_CELL["head"]]],
                   "topk_common": selection["settings"][setting]["tau_source"]["cam"]["common"]["topk"].get(str(args.k)),
                   "topk_camera": {c: selection["settings"][setting]["tau_source"]["cam"]["per_camera"][c]["topk"].get(str(args.k))
                                   for c in POLICY_CAMERAS},
                   "oracle": None if oracle is None else oracle["settings"][setting].get(f"top{args.k}")},
               "options": opts, "workers": args.workers,
               "per_record": {r["key"]: r["summary"] for r in results},
               "pooled": summarize_e2e(all_rows)}
    tag = "heldout" if all(RECORDS[k].role == "heldout" for k in keys) else "mixed"
    if all(RECORDS[k].role == "selection" for k in keys):
        tag = "selection"
    write_json(out_dir / f"summary__{tag}.json", payload)
    print(f"[e2e] wrote {out_dir / f'summary__{tag}.json'}")
    return payload


# ---------------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("gt", "select", "validate", "e2e"))
    ap.add_argument("--records", nargs="+", default=None, metavar="KEY",
                    help=f"record keys {sorted(RECORDS)}. Defaults: gt = all; select = "
                         f"{list(SELECTION_KEYS)}; validate / e2e = {list(HELDOUT_KEYS)}")
    ap.add_argument("--workers", type=int, default=1,
                    help=f"gt / e2e: records in parallel (spawned processes, <= {MAX_WORKERS}). "
                         "Use 1 when the e2e ms column matters")
    # gt
    ap.add_argument("--force", action="store_true", help="gt: re-render even if the cache exists")
    ap.add_argument("--rgb-check-frames", type=int, default=3,
                    help="gt: frames per record whose policy RGB is re-rendered and diffed")
    # select
    ap.add_argument("--tau-method", choices=("kneedle", "curvature"), default="kneedle")
    ap.add_argument("--n-lowest", type=int, default=10)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--denoise", type=int, nargs="+", default=[0, 4, 9],
                    help="select / validate: denoise steps to process (0 = primary)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="select / e2e: first N frames per record only (smoke)")
    # validate
    ap.add_argument("--aggs", nargs="+", default=["last", "mean", "first"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--min-target-px", type=int, default=200)
    # e2e
    ap.add_argument("--sources", nargs="+", default=list(E2E_DEFAULT_SOURCES))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="e2e: smoothing of the combined map (0 = raw sum, T47.task I4)")
    ap.add_argument("--e2e-denoise", type=int, default=0)
    ap.add_argument("--e2e-agg", default="last")
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--links", default="gripper")
    ap.add_argument("--esdf-backend", choices=("legacy", "curobo"), default="legacy")
    ap.add_argument("--on-target-radius", type=float, default=0.10)
    ap.add_argument("--label-radius", type=float, default=0.003)
    return ap


def validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 1 <= args.workers <= MAX_WORKERS:
        ap.error(f"--workers must be 1..{MAX_WORKERS} (pod limit)")
    if args.n_lowest < 1 or args.kmax < 1:
        ap.error("--n-lowest and --kmax must be >= 1")
    if args.k < 1 or args.k > args.kmax:
        ap.error("--k must be in 1..--kmax")
    if args.sigma < 0 or any(s < 0 for s in args.sigmas):
        ap.error("sigma must be >= 0")
    if args.max_frames < 0:
        ap.error("--max-frames must be >= 0")
    if args.min_target_px < 0:
        ap.error("--min-target-px must be >= 0")
    if args.stage == "e2e" and args.e2e_agg not in ("last", "mean", "first"):
        ap.error("--e2e-agg must be last / mean / first")
    if args.stage == "e2e" and args.esdf_backend == "curobo":
        ap.error("--esdf-backend curobo needs the GPU lock and .venv-openpi-live; grounding does "
                 "not depend on the field backend — keep legacy (CPU) for T47")
    if args.records:
        resolve_keys(args.records, ())


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    validate_args(ap, args)
    root = out_root()
    t0 = time.time()
    {"gt": stage_gt, "select": stage_select, "validate": stage_validate,
     "e2e": stage_e2e}[args.stage](args, root)
    print(f"[{STEP}] stage {args.stage} done in {time.time() - t0:.1f}s -> {root}")


if __name__ == "__main__":
    main()
