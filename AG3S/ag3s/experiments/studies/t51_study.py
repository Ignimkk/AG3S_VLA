"""T51 — attention head selection as a pre-registered protocol: filter -> calibration -> bootstrap ->
nested CV -> held-out -> ablation, with GT labels and with the grasp-event self label (RB-Y1 sim).

    source scripts/setup/env.sh
    export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"      # once
    T51="env MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python \\
         -m AG3S.ag3s.experiments.studies.t51_study"
    $T51 --stage gt_extra                     # GT cache of the extra calibration runs (MuJoCo, CPU)
    $T51 --stage filter                       # P1   label-free, no MuJoCo
    $T51 --stage selflabel                    # self label (mj_forward only, no render)
    $T51 --stage score2d --label gt           # P2a  2D + 2D->3D preselection (also --label self)
    $T51 --stage score3d                      # P2b  3D rows of every preselected cell (MuJoCo replay)
    $T51 --stage bootstrap --label gt         # P3   (also --label self)
    $T51 --stage ksel --label gt              # P4   nested CV over k (runs the k>1 sums it needs)
    $T51 --stage heldout                      # P5   selected vs L8H2 on the held-out records
    $T51 --stage ablation                     # P6   paper top-3 · head average · 20 random · oracle
    $T51 --stage compare                      # self label vs GT: agreement, held-out difference

Output: ``$AG3S_RUN_DIR/verify/T51/`` (the CLI refuses without ``AG3S_RUN_DIR``). Read only: T47's run
(``--t47-dir``, default the 2026-10-12 run: GT cache, ``select/selection.json``, ``validate/oracle.json``)
and T47b's (``--t47b-dir``: the sigma-0 e2e rows, parity of the L8H2 rows). The 3D rows of every
(record, source) are cached under ``rows/<record>/<source>.jsonl`` and shared by every stage and both
labels; a stage only replays what is missing (resumable). Protocol rules: `t51_protocol`.

Records (`t51_records`): calibration = T47 selection (``long`` · ``t2`` · ``ep1807``) + the recorded
VLA-training runs ``ep0`` · ``ep500`` · ``ep1200`` (`T51.task.md` 판정 보완 2026-10-12; their attention
npz come from `t51_extract` (GPU) and their GT cache from ``--stage gt_extra``, both under
``--extra-dir``, default this run's ``verify/T51``). Held-out = T47 held-out, unchanged (same-scene
``headfix`` · ``ep1800``, new-scene ``ep1850`` · ``ep1900``). The calibration has five scenes —
episode 1800 (``long`` · ``t2``), 1807, 0, 500, 1200 — and P3 (``--unit scene``) / P4 resample and
cross-validate **scenes**: the runs of one scene are drawn together and never split between train and
test. Every payload names the training episodes (``calibration.vla_training_episodes``).

Aggregation unit (``--aggregate``, 2026-10-12 집계 단위 수정): ``record`` (default, the original
behaviour) ranks on the unweighted mean over records; ``scene`` (`T51.task.md` 판정 보완 → 집계 단위,
pre-registered) averages the runs of a scene first, then the scenes — in the 2D preselection (score2d,
global and every CV fold), the P2 3D ranking (score3d), the P3 bootstrap (each resample: mean of the
drawn scenes' means; needs ``--unit scene``), the P4 fold rankers (the CV score of a test scene was
already the mean of its runs) and the held-out / ablation numbers that consume the selection. Every
``scene`` output carries the suffix ``__scene`` (``score2d__scene.json``, ``ksel__scene.json``, ...) and
reads only ``__scene`` inputs, so the record-mean outputs are never overwritten. The 3D rows are shared.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t47_head_metrics as hm
from AG3S.ag3s.experiments.studies import t47_head_selection as hs
from AG3S.ag3s.experiments.studies import t47b_metrics as bm
from AG3S.ag3s.experiments.studies import t51_protocol as pr
from AG3S.ag3s.experiments.studies import t51_records as rc

STEP = "T51"
T47_RUN = "results/development/20261012T141412+0900_bd778e5e3453_d7a51b456f36"
CALIB_KEYS = tuple(rc.CALIB_KEYS)
HELDOUT_KEYS = tuple(rc.HELDOUT_KEYS)
LABELS = ("gt", "self")
SERVING = pr.Cell("last", 8, 2)
MAX_WORKERS = 2
NOUN_QUERY = "noun_target"
#: Special (non-combo) sources of P6.
SPECIAL_SOURCES = ("paper3_s1", "headavg_last")
#: ``--aggregate``: ``record`` = unweighted mean over records (default, original); ``scene`` = mean of
#: scene means (`T51.task.md` 판정 보완 → 집계 단위).
AGGREGATES = ("record", "scene")


# ------------------------------------------------------------------------------ plumbing


def out_root() -> pathlib.Path:
    """``$AG3S_RUN_DIR/verify/T51`` (created); refuses when the run dir is unset."""
    from common.run_output import resolve_output_path

    run_dir = os.environ.get("AG3S_RUN_DIR")
    if not run_dir:
        raise SystemExit(
            "AG3S_RUN_DIR is not set. Allocate a run first:\n"
            '  export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"')
    root = resolve_output_path("verify/T51", run_dir=run_dir, default_name="verify/T51")
    root.mkdir(parents=True, exist_ok=True)
    return root


def t47_dir(args) -> pathlib.Path:
    p = (pathlib.Path(args.t47_dir) if getattr(args, "t47_dir", None)
         else hs.storage_root() / T47_RUN / "verify" / "T47")
    for need in ("gt_cache", "select/selection.json", "validate/oracle.json"):
        if not (p / need).exists():
            raise SystemExit(f"T47 run incomplete: {p / need} missing (pass --t47-dir)")
    return p


def t47b_dir(args) -> pathlib.Path:
    return (pathlib.Path(args.t47b_dir) if getattr(args, "t47b_dir", None)
            else hs.storage_root() / T47_RUN / "verify" / "T47b")


def write_json(path: pathlib.Path, payload: Any) -> None:
    hs.write_json(path, bm.jsonable(payload))


def read_json(path: pathlib.Path, what: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing {path} — run {what} first")
    return json.loads(path.read_text())


def run_info(stage: str, args: argparse.Namespace) -> dict:
    return {"step": STEP, "stage": stage, "argv": sys.argv,
            "args": {k: v for k, v in vars(args).items()},
            "started_kst": _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(),
            "git": hs.git_state(), "storage_root": str(hs.storage_root()),
            "run_dir": os.environ.get("AG3S_RUN_DIR"), "python": sys.version.split()[0],
            "extra_dir": rc.extra_dir_or_none(),
            "calibration": rc.provenance(CALIB_KEYS),
            "seeds": {"bootstrap": pr.SEED_BOOTSTRAP, "ksel": pr.SEED_KSEL,
                      "random_cells": pr.SEED_RANDOM_CELLS}}


def queries(args) -> list[str]:
    q = list(pr.ACTION_QUERIES)
    if getattr(args, "t49_dir", None):
        q.append(NOUN_QUERY)
    return q


# ---------------------------------------------------------------------------- aggregation


def agg_mode(args) -> str:
    """``record`` (default) or ``scene`` from ``args.aggregate`` (absent -> ``record``)."""
    mode = getattr(args, "aggregate", None) or "record"
    if mode not in AGGREGATES:
        raise SystemExit(f"--aggregate must be one of {AGGREGATES}, got {mode!r}")
    return mode


def agg_tag(mode: str) -> str:
    """File-name suffix of an aggregation: ``""`` for ``record`` (original names), ``__scene``."""
    return "" if mode == "record" else f"__{mode}"


def scene_groups(keys: Sequence[str], mode: str) -> Optional[list[str]]:
    """Scene id of each record under ``scene`` aggregation; ``None`` (= record mean) otherwise."""
    return [rc.scene_of(k) for k in keys] if mode == "scene" else None


def calib_mean(per_record: np.ndarray, keys: Sequence[str], mode: str) -> np.ndarray:
    """Point estimate over the rows (records `keys`) of ``per_record``: record mean or scene mean."""
    return pr.aggregate_mean(per_record, scene_groups(keys, mode))


# ------------------------------------------------------------------------------ attention


def query_blocks(key: str, qs: Sequence[str], t49: Optional[str] = None) -> dict[str, np.ndarray]:
    """``{query: (F, 18, 8, 3, 16, 16)}`` raw maps of one record at denoise 0."""
    att, dn, ag, _ = rc.load_attention(key)
    di = dn.index(0)
    out = {}
    for q in qs:
        if q in pr.ACTION_QUERIES:
            out[q] = att[:, di, ag.index(q)]
        elif q == NOUN_QUERY:
            if not t49:
                raise SystemExit(f"query {q} needs --t49-dir")
            if rc.is_extra(key):
                raise SystemExit(f"[{key}] T49 noun rows exist only for T47's records; drop "
                                 "--t49-dir or the extra calibration runs")
            from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx

            ex = nx.load_extract(pathlib.Path(t49) / "extract" / "orig" / f"{key}.npz")
            blk = nx.query_block(ex, "target")
            if blk.shape[0] != att.shape[0]:
                raise SystemExit(f"[{key}] T49 noun maps have {blk.shape[0]} frames, npz {att.shape[0]}")
            out[q] = blk
        else:
            raise SystemExit(f"unknown query {q!r}")
    return out


def cell_map(blocks: dict, cell: pr.Cell) -> np.ndarray:
    """``(F, 3, 16, 16)`` float64 of one cell (sigma 0)."""
    return np.asarray(blocks[cell.query][:, cell.layer, cell.head], np.float64)


def t47_cells(t47: pathlib.Path) -> dict[str, list[pr.Cell]]:
    """T47's head sets at d0 · last (read only): paper label-free common top-3, GT-oracle top-3."""
    sel = json.loads((t47 / "select" / "selection.json").read_text())
    ora = json.loads((t47 / "validate" / "oracle.json").read_text())
    common = sel["settings"]["d0_last"]["tau_source"]["cam"]["common"]["topk"]["3"]
    oracle = ora["settings"]["d0_last"]["top3"]
    return {"paper3": [pr.Cell("last", int(l), int(h)) for l, h in common],
            "oracle3": [pr.Cell("last", int(l), int(h)) for l, h in oracle]}


def source_maps(blocks: dict, sid: str, t47: pathlib.Path) -> np.ndarray:
    """``(F, 3, 16, 16)`` raw maps of a source id: a cell, a ``sum[...]`` of cells, or a special."""
    if sid == "paper3_s1":
        cells = t47_cells(t47)["paper3"]
        block_c = np.moveaxis(np.asarray(blocks["last"], np.float64), 3, 1)  # (F, C, L, H, 16, 16)
        return hm.combine_maps(block_c, [(c.layer, c.head) for c in cells], 1.0)
    if sid == "headavg_last":
        block_c = np.moveaxis(np.asarray(blocks["last"], np.float64), 3, 1)
        return hm.head_average(block_c)
    cells = pr.combo_cells(sid)
    return np.sum([cell_map(blocks, c) for c in cells], axis=0)


# ------------------------------------------------------------------------ GT (extra runs)


def _gt_extra_one(key: str, cache: str, rgb_check_frames: int, extra: str) -> dict:
    """T47's `_gt_one` (`t47_head_selection.py:274-294`) for an extra run: the same
    `t47_replay.extract_gt` call and the same added keys, so the cache has T47's format."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.studies.t47_replay import extract_gt

    rc.configure(extra)
    if rc.npz_path(key).is_file():
        info = rc.check_pairing(key)
    else:  # the GT does not need the attention; pairing is re-checked by every stage that loads it
        meta = rc.meta_of(key)
        probs = rc.episode_problems(key, meta)
        if probs:
            raise SystemExit(f"[{key}] " + "; ".join(probs))
        info = {"key": key, "role": rc.EXTRA_ROLE, "npz": None, "record": str(rc.record_path(key)),
                "prompt": meta["prompt"], "episode_index": rc.EXTRA_RECORDS[key].episode,
                "scene_id": rc.scene_of(key), "vla_training_episode": True,
                "pairing": "attention npz not extracted yet: record + episode only"}
    t0 = time.time()
    arrays = extract_gt(rc.record_path(key), target=rc.target_of(key),
                        rgb_check_frames=rgb_check_frames,
                        log=lambda s: print(f"[{key}]{s}", flush=True))
    arrays["record_key"] = np.asarray(key)
    arrays["pairing"] = np.asarray(json.dumps(info))
    path = pathlib.Path(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)
    tpx = arrays["class_px"][:, :, 0]
    return {**info, "cache": cache, "seconds": time.time() - t0, "target": str(arrays["target"]),
            "n_frames": int(tpx.shape[0]),
            "target_visible_ge200px_per_camera": {c: int((tpx[:, i] >= 200).sum())
                                                  for i, c in enumerate(hs.POLICY_CAMERAS)},
            "rgb_check_index": arrays["rgb_check_index"].tolist(),
            "rgb_mad": arrays["rgb_mad"].tolist(),
            "rgb_frac_gt16": arrays["rgb_frac_gt16"].tolist()}


def stage_gt_extra(args, root: pathlib.Path) -> dict:
    """GT cache (T47 format) of the extra calibration runs -> ``<extra dir>/gt_cache_extra/``.
    T47's cache is never touched. Cached files are skipped unless ``--force``."""
    keys = rc.resolve_keys(args.records, rc.EXTRA_KEYS)
    bad = [k for k in keys if not rc.is_extra(k)]
    if bad:
        raise SystemExit(f"gt_extra is for the extra runs {rc.EXTRA_KEYS} only (T47's cache is "
                         f"read-only), got {bad}")
    extra = str(rc.extra_dir())
    todo = [k for k in keys if args.force or not rc.gt_cache_path(k).is_file()]
    for k in keys:
        print(f"[gt_extra] {k}: {'render' if k in todo else 'cached'} -> {rc.gt_cache_path(k)}")
    if args.workers > 1 and len(todo) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(todo)),
                                    mp_context=mp.get_context("spawn")) as ex:
            res = [f.result() for f in [ex.submit(_gt_extra_one, k, str(rc.gt_cache_path(k)),
                                                  args.rgb_check_frames, extra) for k in todo]]
    else:
        res = [_gt_extra_one(k, str(rc.gt_cache_path(k)), args.rgb_check_frames, extra)
               for k in todo]
    payload = {**run_info("gt_extra", args), "provenance": rc.provenance(keys),
               "rendered": res, "skipped_cached": [k for k in keys if k not in todo],
               "format": "T47 gt_cache (t47_replay.extract_gt + record_key + pairing)"}
    write_json(rc.extra_dir() / "gt_cache_extra" / "gt_summary.json", payload)
    print(f"[gt_extra] {len(res)} rendered, {len(keys) - len(res)} cached")
    return payload


# ------------------------------------------------------------------------------ P1 filter


def stage_filter(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t47b_sink as sk

    keys = rc.resolve_keys(args.records, CALIB_KEYS)
    pairing = [rc.check_pairing(k) for k in keys]
    qs = queries(args)
    t47 = t47_dir(args)
    per_q: dict[str, dict] = {q: {} for q in qs}
    n_frames = {}
    for k in keys:
        blocks = query_blocks(k, qs, args.t49_dir)
        for q in qs:
            b = np.asarray(blocks[q][:, pr.LAYERS[0]:pr.LAYERS[-1] + 1], np.float64)
            st = sk.map_stats(b)  # each (F, 16, 8, 3)
            for f, v in st.items():
                per_q[q].setdefault(f, []).append(v)
        n_frames[k] = int(blocks[qs[0]].shape[0])
        print(f"[filter] {k}: {n_frames[k]} frames x {len(qs)} queries")
    arrays: dict[str, np.ndarray] = {"queries": np.asarray(qs), "layers": np.asarray(pr.LAYERS),
                                     "heads": np.asarray(pr.HEADS),
                                     "cameras": np.asarray(hs.POLICY_CAMERAS)}
    result = {"thresholds": pr.FILTER_THRESHOLDS, "queries": {}, "eligible": [], "excluded": {}}
    for q in qs:
        pf = {f: np.concatenate(v) for f, v in per_q[q].items()}
        summ = pr.filter_summary(pf)
        reasons = pr.filter_reasons(summ)
        elig = pr.filter_eligible(reasons)  # (16, 8)
        for f in ("mass_median", "modal_fraction", "top1_share_mean", "largest_share_of_grid_mean",
                  "n_components_mean", "entropy_mean", "S_cam_mean"):
            arrays[f"{q}__{f}"] = summ[f]
        for r in (*pr.FILTER_REASONS, "any"):
            arrays[f"{q}__reason__{r}"] = reasons[r]
        arrays[f"{q}__eligible"] = elig
        counts = {r: {"cells_any_camera": int(reasons[r].any(axis=-1).sum()),
                      "per_camera": {c: int(reasons[r][..., i].sum())
                                     for i, c in enumerate(hs.POLICY_CAMERAS)}}
                  for r in pr.FILTER_REASONS}
        result["queries"][q] = {"n_cells": int(elig.size), "n_eligible": int(elig.sum()),
                                "reason_counts": counts}
        for li, l in enumerate(pr.LAYERS):
            for h in pr.HEADS:
                cid = pr.Cell(q, l, h).id
                if elig[li, h]:
                    result["eligible"].append(cid)
                else:
                    result["excluded"][cid] = {
                        c: [r for r in pr.FILTER_REASONS if reasons[r][li, h, ci]]
                        for ci, c in enumerate(hs.POLICY_CAMERAS)
                        if reasons["any"][li, h, ci]}
    marks = {"l8h2": [SERVING], **t47_cells(t47)}
    marked = {}
    for name, cells in marks.items():
        rows = []
        for c in cells:
            if c.query not in qs:
                continue
            li = pr.LAYERS.index(c.layer)
            rows.append({"cell": c.id, "eligible": c.id in result["eligible"],
                         "reasons": result["excluded"].get(c.id, {}),
                         "per_camera": {cam: {f: float(arrays[f"{c.query}__{f}"][li, c.head, ci])
                                              for f in ("mass_median", "modal_fraction",
                                                        "top1_share_mean",
                                                        "largest_share_of_grid_mean")}
                                        for ci, cam in enumerate(hs.POLICY_CAMERAS)}})
        marked[name] = rows
    payload = {**run_info("filter", args), "records": pairing, "n_frames": n_frames,
               "provenance": rc.provenance(keys),
               "uses_gt": False, **result, "marked": marked,
               "definitions": {
                   "mass_median": "median over every calibration frame of S_cam (16x16 sum)",
                   "modal_fraction": "share of frames whose argmax patch is the most frequent one "
                                     "(T47b #5 (a), frames pooled over the calibration records)",
                   "top1_share_mean": "frame mean of max patch / S_cam (T47b #5 (b))",
                   "largest_share_of_grid_mean": "frame mean of largest 8-connected component of the "
                                                 "mean-binarised map / 256 (T47b #5 (c))",
                   "eligible": "no reason on any camera"}}
    out = root / "filter"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "filter_arrays.npz", **arrays)
    write_json(out / "filter.json", payload)
    print(f"[filter] eligible {len(result['eligible'])} / "
          f"{sum(v['n_cells'] for v in result['queries'].values())} -> {out}")
    return payload


def load_filter(root: pathlib.Path) -> dict:
    return read_json(root / "filter" / "filter.json", "--stage filter")


# ------------------------------------------------------------------------------ self label


def selflabel_path(root: pathlib.Path, key: str) -> pathlib.Path:
    return root / "selflabel" / f"{key}.npz"


def load_selflabel(root: pathlib.Path, key: str, *, required: bool = True) -> Optional[dict]:
    p = selflabel_path(root, key)
    if not p.is_file():
        if required:
            raise SystemExit(f"[{key}] no self label at {p} — run --stage selflabel first")
        return None
    with np.load(p, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _selflabel_one(key: str, path: str, t47: str, event: str, extra: Optional[str] = None) -> dict:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.studies import t51_selflabel as sl

    if extra is not None:
        rc.configure(extra)
    gt = rc.load_gt(pathlib.Path(t47), key)
    t0 = time.time()
    arr = sl.build_selflabel(rc.record_path(key), target=rc.target_of(key), event=event, gt=gt,
                             log=lambda s: print(f"[{key}]{s}", flush=True))
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arr)
    return {"key": key, "seconds": time.time() - t0, **sl.summarize_selflabel(arr)}


def stage_selflabel(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t51_selflabel as sl

    t47 = t47_dir(args)
    keys = rc.resolve_keys(args.records, list(rc.ALL_KEYS))
    pairing = [rc.check_pairing(k) for k in keys]
    todo = [k for k in keys if args.force or not selflabel_path(root, k).is_file()]
    extra = str(rc.extra_dir())
    res = []
    if args.workers > 1 and len(todo) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(todo)),
                                    mp_context=mp.get_context("spawn")) as ex:
            res = [f.result() for f in [ex.submit(_selflabel_one, k, str(selflabel_path(root, k)),
                                                  str(t47), args.event, extra) for k in todo]]
    else:
        res = [_selflabel_one(k, str(selflabel_path(root, k)), str(t47), args.event, extra)
               for k in todo]
    summaries = {}
    for k in keys:
        arr = load_selflabel(root, k)
        summaries[k] = {"role": rc.role(k), "scene_group": rc.scene_group(k),
                        "scene_id": rc.scene_of(k),
                        "vla_training_episode": k in rc.VLA_TRAINING_KEYS,
                        **sl.summarize_selflabel(arr)}
    payload = {**run_info("selflabel", args), "records": pairing, "computed": [r["key"] for r in res],
               "seconds": {r["key"]: r["seconds"] for r in res},
               "constants": {"event": args.event, "r_obj_clip_m": [sl.R_OBJ_MIN, sl.R_OBJ_MAX],
                             "self_ball_margin_m": sl.SELF_BALL_MARGIN,
                             "render_hw": list(sl.RENDER_HW), "label_stride": sl.LABEL_STRIDE,
                             "min_self_px": sl.MIN_SELF_PX, "motion_tol_m": sl.MOTION_TOL_M,
                             "patch_hit_cov": sl.PATCH_HIT_COV},
               "per_record": summaries}
    write_json(root / "selflabel" / "selflabel.json", payload)
    print(f"[selflabel] {len(res)} computed, {len(keys) - len(res)} cached -> {root / 'selflabel'}")
    return payload


def label_data(label: str, key: str, root: pathlib.Path, t47: pathlib.Path, min_px: int
               ) -> tuple[np.ndarray, np.ndarray]:
    """``(class_cov (F, C, K, 16, 16), keep (F, C))`` of one record for the 2D score."""
    if label == "gt":
        gt = rc.load_gt(t47, key)
        cov = np.asarray(gt["class_cov"], np.float64)
        keep = np.asarray(gt["class_px"])[:, :, 0] >= min_px
        return cov, keep
    if label == "self":
        s = load_selflabel(root, key)
        c = np.asarray(s["self_cov"], np.float64)
        cov = np.stack([c, 1.0 - c], axis=2)
        keep = np.asarray(s["labeled"], bool)[:, None] & (np.asarray(s["self_px"]) >= min_px)
        return cov, keep
    raise ValueError(label)


# --------------------------------------------------------------------------- P2a score2d


def train_subsets(units: Sequence[str], group_of: Optional[dict] = None) -> list[tuple[str, ...]]:
    """Every training set the nested CV ranks on (outer LOEO trains and their inner trains).

    With `group_of` (record -> scene) the folds leave one **scene** out (`pr.group_train_sets`): the
    record tuples are exactly those `pr.grouped_cv`'s ranker passes to the per-record ranking."""
    if group_of is not None:
        return pr.group_train_sets(units, group_of)
    out: list[tuple[str, ...]] = []
    for train, _ in pr.loeo_folds(units):
        out.append(tuple(train))
        if len(train) >= 2:
            out += [tuple(t2) for t2, _ in pr.loeo_folds(train)]
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def subset_key(units: Sequence[str]) -> str:
    return "|".join(units)


def stage_score2d(args, root: pathlib.Path) -> dict:
    label = args.label
    mode = agg_mode(args)
    tag = agg_tag(mode)
    filt = load_filter(root)
    t47 = t47_dir(args)
    keys = rc.resolve_keys(args.records, CALIB_KEYS)
    qs = list(filt["queries"])
    if args.t49_dir and NOUN_QUERY not in qs:
        raise SystemExit("filter was run without --t49-dir; rerun filter with the noun query")
    cells = pr.omega(qs)
    ids = [c.id for c in cells]
    eligible = np.asarray([i in set(filt["eligible"]) for i in ids])
    n_q, n_l, n_h = len(qs), len(pr.LAYERS), len(pr.HEADS)
    metrics = ("lift", "peak_on_target", "target_mass")
    arr = {m: np.full((len(keys), n_q, n_l, n_h), np.nan) for m in metrics}
    n_kept = {}
    for ei, k in enumerate(keys):
        cov, keep = label_data(label, k, root, t47, args.min_target_px)
        blocks = query_blocks(k, qs, args.t49_dir)
        n_kept[k] = int(keep.sum())
        for qi, q in enumerate(qs):
            b = blocks[q][:, pr.LAYERS[0]:pr.LAYERS[-1] + 1]
            sc = pr.record_2d_scores(b, cov, keep)
            for m in metrics:
                arr[m][ei, qi] = sc[m]
        print(f"[score2d/{label}] {k}: kept {n_kept[k]} (frame, camera) samples")
    flat = {m: arr[m].reshape(len(keys), -1) for m in metrics}  # (E, N) in omega order
    tc = t47_cells(t47)
    forced_cells = [SERVING, *tc["oracle3"]]
    forced = [ids.index(c.id) for c in forced_cells if c.id in ids]

    def pick(units):
        sel = [keys.index(u) for u in units]
        return pr.preselect(calib_mean(flat["lift"][sel], units, mode),
                            calib_mean(flat["peak_on_target"][sel], units, mode),
                            eligible, n_top=args.preselect_top, forced=forced)

    glob = pick(keys)
    scenes = rc.scene_map(keys)
    folds = {subset_key(t): pick(t) for t in train_subsets(keys, scenes)}
    cand = sorted(set(glob["scored_set"]).union(*[set(f["scored_set"]) for f in folds.values()]))
    s2 = calib_mean(flat["lift"], keys, mode)
    peak2 = calib_mean(flat["peak_on_target"], keys, mode)
    mass2 = calib_mean(flat["target_mass"], keys, mode)
    order = pr.rank_order(s2, peak2, eligible)

    def as_ids(d):
        return {k: ([ids[i] for i in v] if isinstance(v, list) else v) for k, v in d.items()}

    payload = {**run_info("score2d", args), "label": label, "records": keys, "queries": qs,
               "provenance": rc.provenance(keys), "cv_unit": "scene", "scene_of": scenes,
               "n_kept_samples": n_kept, "min_target_px": args.min_target_px,
               "preselect_top": args.preselect_top,
               "forced": [c.id for c in forced_cells],
               "rule": f"top-N eligible cells by calibration 2D lift ({mode} mean; tie 2D peak, "
                       "then Ω index) + forced (L8H2, T47 oracle top-3); forced cells failing P1 are "
                       "3D-scored but not ranked",
               **({"aggregate": mode} if mode != "record" else {}),
               "global": as_ids(glob),
               "folds": {k: as_ids(v) for k, v in folds.items()},
               "candidates_3d": [ids[i] for i in cand],
               "ranking_2d_top30": [{"cell": ids[i], "lift": float(s2[i]),
                                     "peak_on_target": float(peak2[i]),
                                     "target_mass": float(mass2[i]),
                                     "per_record_lift": {k: float(flat["lift"][e, i])
                                                         for e, k in enumerate(keys)}}
                                    for i in order[:30]],
               "marked": {c.id: {"lift": float(s2[ids.index(c.id)]),
                                 "rank_2d": (int(np.flatnonzero(order == ids.index(c.id))[0]) + 1
                                             if ids.index(c.id) in set(order.tolist()) else None),
                                 "eligible": bool(eligible[ids.index(c.id)])}
                          for c in [SERVING, *tc["oracle3"], *tc["paper3"]] if c.id in ids}}
    out = root / label / "score2d"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"score2d_arrays{tag}.npz", records=np.asarray(keys), queries=np.asarray(qs),
                        layers=np.asarray(pr.LAYERS), heads=np.asarray(pr.HEADS),
                        eligible=eligible.reshape(n_q, n_l, n_h),
                        **{m: arr[m] for m in metrics})
    write_json(out / f"score2d{tag}.json", payload)
    print(f"[score2d/{label}{tag}] global top {len(glob['top'])} + forced -> {len(cand)} 3D candidates "
          f"(incl. fold preselections) -> {out}")
    return payload


def load_score2d(root: pathlib.Path, label: str, tag: str = "") -> tuple[dict, dict]:
    """score2d payload and arrays of `label`; ``tag`` = `agg_tag` (``__scene`` reads the scene run)."""
    js = read_json(root / label / "score2d" / f"score2d{tag}.json",
                   f"--stage score2d --label {label}" + (f" --aggregate {tag[2:]}" if tag else ""))
    with np.load(root / label / "score2d" / f"score2d_arrays{tag}.npz", allow_pickle=False) as d:
        arrs = {k: d[k] for k in d.files}
    return js, arrs


def lift_2d(arrs: dict, cell_id: str, metric: str = "lift") -> np.ndarray:
    """Per-record 2D `metric` of a cell from the score2d arrays (NaN for a non-cell source)."""
    try:
        c = pr.parse_cell(cell_id)
    except ValueError:
        return np.full(len(arrs["records"]), np.nan)
    qs = [str(q) for q in arrs["queries"]]
    if c.query not in qs:
        return np.full(len(arrs["records"]), np.nan)
    return arrs[metric][:, qs.index(c.query), pr.LAYERS.index(c.layer), c.head]


# ------------------------------------------------------------------------------ 3D rows


def rows_path(root: pathlib.Path, key: str, sid: str, *, smoke: bool = False) -> pathlib.Path:
    return root / ("rows_smoke" if smoke else "rows") / key / f"{pr.safe_name(sid)}.jsonl"


def load_rows(root: pathlib.Path, key: str, sid: str) -> Optional[list[dict]]:
    p = rows_path(root, key, sid)
    if not p.is_file():
        return None
    with p.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _complete(root: pathlib.Path, key: str, sid: str, n_frames: int) -> bool:
    rows = load_rows(root, key, sid)
    return rows is not None and len(rows) == n_frames


def _write_rows(path: pathlib.Path, rows: list[dict], key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as fh:
        for r in rows:
            r["record"] = key
            r["scene_group"] = rc.scene_group(key)
            r["scene_id"] = rc.scene_of(key)
            fh.write(json.dumps(bm.jsonable(r)) + "\n")
    tmp.replace(path)


def _replay_job(key: str, root: str, sids: list[str], opts: dict, t47: str, t49: Optional[str],
                batch: int, max_frames: int, need_self: bool, extra: Optional[str] = None) -> dict:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.studies import t51_replay as rp

    if extra is not None:
        rc.configure(extra)
    rootp = pathlib.Path(root)
    qs = sorted({c.query for s in sids if s not in SPECIAL_SOURCES for c in pr.combo_cells(s)}
                | ({"last"} if any(s in SPECIAL_SOURCES for s in sids) else set()))
    blocks = query_blocks(key, qs, t49)
    n_f = int(next(iter(blocks.values())).shape[0])
    selflabel = load_selflabel(rootp, key, required=need_self)
    frames = None if not max_frames else list(range(min(max_frames, n_f)))
    done = []
    for b0 in range(0, len(sids), batch):
        part = sids[b0:b0 + batch]
        maps = {s: source_maps(blocks, s, pathlib.Path(t47)).astype(np.float32) for s in part}
        t0 = time.time()
        rows = rp.run_sources(rc.record_path(key), target=rc.target_of(key), sources=maps,
                              selflabel=selflabel, frames=frames,
                              log=lambda s: print(f"[{key}]{s}", flush=True), **opts)
        for s in part:
            _write_rows(rows_path(rootp, key, s, smoke=bool(max_frames)), rows[s], key)
        done += part
        print(f"[{key}] wrote {len(part)} sources ({time.time() - t0:.0f}s)", flush=True)
    return {"key": key, "sources": done, "n_frames": n_f}


def ensure_rows(root: pathlib.Path, args, plan: dict[str, list[str]], *, need_self: bool) -> dict:
    """Replay what is missing of ``{record: [source ids]}`` (resumable; smoke runs go to
    ``rows_smoke/`` and never count as complete)."""
    opts = {"voxel": args.voxel, "range_max": args.range_max, "links": args.links,
            "esdf_backend": "legacy", "on_target_radius": args.on_target_radius,
            "label_radius": args.label_radius, "membership_radius": args.membership_radius}
    jobs = []
    for key, sids in plan.items():
        n_f = rc.n_frames(key)
        todo = sorted(set(sids)) if args.max_frames else sorted(
            s for s in set(sids) if not _complete(root, key, s, n_f))
        if todo:
            jobs.append((key, todo))
        print(f"[rows] {key}: {len(set(sids)) - len(todo)} cached, {len(todo)} to replay")
    t47 = str(t47_dir(args))
    extra = str(rc.extra_dir())
    res = []
    if args.workers > 1 and len(jobs) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                    mp_context=mp.get_context("spawn")) as ex:
            res = [f.result() for f in [ex.submit(_replay_job, k, str(root), s, opts, t47,
                                                  args.t49_dir, args.source_batch, args.max_frames,
                                                  need_self, extra) for k, s in jobs]]
    else:
        res = [_replay_job(k, str(root), s, opts, t47, args.t49_dir, args.source_batch,
                           args.max_frames, need_self, extra) for k, s in jobs]
    return {"replayed": {r["key"]: r["sources"] for r in res}}


# ----------------------------------------------------------------------------- 3D metrics


def record_metrics(rows: Sequence[dict]) -> dict[str, float]:
    """Record-level numbers of one source: GT on every frame, GT on the self-labeled frames only,
    and the self label (self-labeled frames only). NaN when no frame qualifies."""
    def rate(rs, f):
        v = [bool(r.get(f)) for r in rs]
        return float(np.mean(v)) if v else float("nan")

    def mean(rs, f):
        v = np.asarray([float(r[f]) for r in rs if r.get(f) is not None], np.float64)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    lab = [r for r in rows if r.get("self_labeled")]
    has = [r for r in rows if r.get("has_target")]
    return {
        "n_frames": len(rows), "n_self_labeled": len(lab),
        "gt_identity": rate(rows, "identity_correct"),
        "gt_identity_on_self_frames": rate(lab, "identity_correct"),
        "self_identity": rate(lab, "self_identity_correct"),
        "on_target": rate(rows, "on_target"),
        "has_target": rate(rows, "has_target"),
        "purity_mean": mean(has, "chosen_cluster_purity"),
        "nontarget_seed_loss_per_frame": mean(rows, "nontarget_seed_loss"),
        "other_object_loss_per_frame": mean(rows, "nontarget_seed_loss__other_object"),
        "self_out_n_per_labeled_frame": mean(lab, "self_out_n"),
    }


PRIMARY = {"gt": "gt_identity", "self": "self_identity"}


def metric_matrix(root: pathlib.Path, keys: Sequence[str], sids: Sequence[str], field: str
                  ) -> np.ndarray:
    """``(E, N)`` record-level `field` of each source (raises when a row file is missing)."""
    out = np.full((len(keys), len(sids)), np.nan)
    for e, k in enumerate(keys):
        for j, s in enumerate(sids):
            rows = load_rows(root, k, s)
            if rows is None:
                raise SystemExit(f"[{k}] no 3D rows for {s} — run the stage that replays it")
            out[e, j] = record_metrics(rows)[field]
    return out


def stage_score3d(args, root: pathlib.Path) -> dict:
    keys = rc.resolve_keys(args.records, CALIB_KEYS)
    tag = agg_tag(agg_mode(args))
    labels = [l for l in args.labels if (root / l / "score2d" / f"score2d{tag}.json").is_file()]
    if not labels and not args.cells:
        raise SystemExit(f"no score2d{tag} output for any label — run --stage score2d first")
    sids: set[str] = set(args.cells or [])
    for l in labels:
        sids |= set(load_score2d(root, l, tag)[0]["candidates_3d"])
    plan = {k: sorted(sids) for k in keys}
    rep = ensure_rows(root, args, plan, need_self=not args.no_selflabel)
    if args.max_frames:
        print("[score3d] smoke run (--max-frames): rows in rows_smoke/, no aggregation")
        return rep
    out = {}
    for l in labels:
        out[l] = aggregate3d(root, l, args)
    return {**rep, "aggregated": list(out)}


def ranking_3d(root: pathlib.Path, label: str, units: Sequence[str], ranked_ids: Sequence[str],
               arrs: dict, keys_all: Sequence[str], mode: str = "record") -> list[str]:
    """Ranked cell ids on `units` (P2 rule: primary 3D rate, tie 2D lift, then Ω order), both
    aggregated over `units` by `mode` (``record`` mean or ``scene`` mean, `calib_mean`)."""
    ids = list(ranked_ids)
    sel = [list(keys_all).index(u) for u in units]
    s = calib_mean(metric_matrix(root, units, ids, PRIMARY[label]), units, mode)
    t = calib_mean(np.stack([lift_2d(arrs, i) for i in ids], axis=1)[sel], units, mode)
    # Ω order = the order of `ranked_ids` (score2d writes them in Ω index order)
    return [ids[i] for i in pr.rank_order(s, t)]


def aggregate3d(root: pathlib.Path, label: str, args) -> dict:
    mode = agg_mode(args)
    tag = agg_tag(mode)
    js, arrs = load_score2d(root, label, tag)
    keys = list(js["records"])
    cand = list(js["candidates_3d"])
    ranked = list(js["global"]["ranked_set"])
    fields = ("gt_identity", "gt_identity_on_self_frames", "self_identity", "on_target",
              "purity_mean", "nontarget_seed_loss_per_frame", "other_object_loss_per_frame")
    per = {f: metric_matrix(root, keys, cand, f) for f in fields}
    order = ranking_3d(root, label, keys, ranked, arrs, keys, mode)
    lift = np.stack([lift_2d(arrs, c) for c in cand], axis=1)
    _, members = pr.group_members([rc.scene_of(k) for k in keys])
    prim = per[PRIMARY[label]]
    scene_vals = np.stack([pr.unit_mean(prim[m]) for m in members])  # (G, N)
    p = PRIMARY[label]
    cells = {}
    for j, c in enumerate(cand):
        cells[c] = {"in_ranked_set": c in ranked, "lift_2d_mean": float(pr.unit_mean(lift[:, j])),
                    f"{p}_scene_mean": float(pr.unit_mean(scene_vals[:, j])),
                    "peak_on_target_2d_mean": float(pr.unit_mean(lift_2d(arrs, c, "peak_on_target"))),
                    **{f"{f}_mean": float(pr.unit_mean(per[f][:, j])) for f in fields},
                    "per_record": {k: {f: float(per[f][e, j]) for f in fields}
                                   for e, k in enumerate(keys)}}
        if mode == "scene":
            cells[c]["lift_2d_scene_mean"] = float(calib_mean(lift[:, j], keys, mode))
    if mode == "record":
        ranking = [{"rank": i + 1, "cell": c, p: cells[c][f"{p}_mean"],
                    "lift_2d": cells[c]["lift_2d_mean"]} for i, c in enumerate(order)]
        aggregation = ("record = frames pooled; calibration = unweighted mean over records "
                       "(ranking); <primary>_scene_mean = mean of scene means (reported, "
                       "not ranked); gt_identity over all frames, self_identity over "
                       "self-labeled frames")
    else:
        ranking = [{"rank": i + 1, "cell": c, p: cells[c][f"{p}_scene_mean"],
                    "lift_2d": cells[c]["lift_2d_scene_mean"],
                    f"{p}_record_mean": cells[c][f"{p}_mean"]} for i, c in enumerate(order)]
        aggregation = ("record = frames pooled; calibration = mean of scene means (runs of a scene "
                       "averaged first; ranking, T51.task.md 판정 보완 → 집계 단위; tie = scene-mean "
                       "2D lift); <field>_mean = unweighted mean over records (reported, not "
                       "ranked); gt_identity over all frames, self_identity over self-labeled frames")
    payload = {**run_info("score3d", args), "label": label, "records": keys,
               "provenance": rc.provenance(keys),
               "primary": p, "candidates_3d": cand, "ranked_set": ranked,
               "ranking": ranking,
               "top1": order[0] if order else None, "cells": cells,
               "aggregation": aggregation,
               **({"aggregate": mode, "scene_of": rc.scene_map(keys)} if mode != "record" else {})}
    out = root / label / "score3d"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"score3d_arrays{tag}.npz", records=np.asarray(keys),
                        cells=np.asarray(cand), **{f: per[f] for f in fields})
    write_json(out / f"score3d{tag}.json", payload)
    print(f"[score3d/{label}{tag}] top1 {payload['top1']} -> {out}")
    return payload


# ------------------------------------------------------------------------------- P3


def unit_matrix(root: pathlib.Path, keys: Sequence[str], sids: Sequence[str], field: str,
                unit: str) -> tuple[list[str], np.ndarray, list[str]]:
    """``(unit ids, (U, N) matrix, record of each unit)`` for ``unit`` = ``scene`` / ``record`` (one row
    per record; ``scene`` groups the rows in `stage_bootstrap`) or ``block:N`` (exploratory: N
    consecutive recorded frames of one record form a unit)."""
    if unit in ("record", "scene"):
        return list(keys), metric_matrix(root, keys, sids, field), list(keys)
    if not unit.startswith("block:"):
        raise SystemExit(f"--unit must be scene, record or block:N, got {unit!r}")
    size = int(unit.split(":", 1)[1])
    if size < 1:
        raise SystemExit("block size must be >= 1")
    units, rec_of, cols = [], [], []
    for k in keys:
        rows_by = {s: load_rows(root, k, s) for s in sids}
        n = len(next(iter(rows_by.values())))
        for b0 in range(0, n, size):
            units.append(f"{k}#{b0 // size}")
            rec_of.append(k)
            cols.append([record_metrics([r for r in rows_by[s] if b0 <= r["frame"] < b0 + size])[field]
                         for s in sids])
    return units, np.asarray(cols, np.float64), rec_of


def stage_bootstrap(args, root: pathlib.Path) -> dict:
    label = args.label
    mode = agg_mode(args)
    if mode == "scene" and args.unit != "scene":
        raise SystemExit("--aggregate scene resamples scenes: use --unit scene")
    js, arrs = load_score2d(root, label, agg_tag(mode))
    keys = list(js["records"])
    ranked = list(js["global"]["ranked_set"])
    units, S, rec_of = unit_matrix(root, keys, ranked, PRIMARY[label], args.unit)
    lift = np.stack([lift_2d(arrs, c) for c in ranked], axis=1)  # (E, N)
    T = lift[[keys.index(r) for r in rec_of]]
    groups = [rc.scene_of(r) for r in rec_of] if args.unit == "scene" else None
    bs = pr.bootstrap_ranking(S, T, np.ones(len(ranked), bool), n_boot=args.n_boot,
                              seed=pr.SEED_BOOTSTRAP, groups=groups, group_mean=mode == "scene")
    if groups is not None:
        units = list(bs["groups"])
    order = np.argsort(-bs["p_rank1"], kind="stable")
    full = pr.rank_order(calib_mean(S, rec_of, mode), calib_mean(T, rec_of, mode))
    top1 = ranked[full[0]]

    def row(i):
        r = {"cell": ranked[i], "p_rank1": float(bs["p_rank1"][i]), "p_top3": float(bs["p_top3"][i]),
             "mean": float(bs["mean"][i]), "ci95": [float(bs["ci_lo"][i]), float(bs["ci_hi"][i])],
             "full_sample_rank": int(np.flatnonzero(full == i)[0]) + 1}
        if bs["exact"]:
            r["exact_p_rank1"] = float(bs["exact"]["p_rank1"][i])
            r["exact_p_top3"] = float(bs["exact"]["p_top3"][i])
        return r

    payload = {**run_info("bootstrap", args), "label": label, "primary": PRIMARY[label],
               "provenance": rc.provenance(keys), "records": keys,
               "unit": args.unit, "units": units, "n_units": len(units), "n_boot": bs["n_boot"],
               "unit_definition": {
                   "scene": "pre-registered (판정 보완): draw scenes with replacement, every run of a "
                            "drawn scene enters; " + ("record-level mean" if mode == "record" else
                                                      "mean of the drawn scenes' means (runs of a "
                                                      "scene averaged first; 집계 단위)"),
                   "record": "exploratory: draw records (long and t2 drawn independently)",
                   "block": "exploratory: N consecutive frames of one record"}[
                       args.unit.split(":")[0]],
               "units_records": ({g: [keys[i] for i in m] for g, m in bs["group_members"].items()}
                                 if groups is not None else None),
               "seed": bs["seed"], "n_distinct_resamples": bs["n_distinct_resamples"],
               "rank1_tied_rate": bs["rank1_tied_rate"],
               "exact_rank1_tied_rate": (bs["exact"] or {}).get("rank1_tied_rate"),
               "full_sample_top1": top1,
               "full_sample_top1_p_rank1": float(bs["p_rank1"][full[0]]),
               "stable_single_cell": bool(bs["p_rank1"][full[0]] >= 0.5),
               "stability_rule": "single cell claimed only when P(rank 1) of the full-sample winner "
                                 ">= 0.5 (T51.task.md 되돌아올 지점)",
               "top10_by_p_rank1": [row(i) for i in order[:10]],
               "all": [row(i) for i in range(len(ranked))],
               **({"aggregate": mode} if mode != "record" else {})}
    tag = ("" if args.unit == "scene" else "__" + args.unit.replace(":", "")) + agg_tag(mode)
    write_json(root / label / "bootstrap" / f"bootstrap{tag}.json", payload)
    print(f"[bootstrap/{label}{tag}] top1 {top1} P(rank1)={payload['full_sample_top1_p_rank1']:.3f} "
          f"({len(units)} units, {bs['n_distinct_resamples']} distinct resamples)")
    return payload


# ------------------------------------------------------------------------------- P4


def make_rank_fn(root: pathlib.Path, label: str, js: dict, arrs: dict, mode: str = "record"):
    """Ranker of a training tuple of records: the score2d preselection of that tuple (``js`` must be the
    score2d payload of the same `mode`), ranked by `ranking_3d` with `mode` aggregation."""
    keys = list(js["records"])
    cache: dict = {}

    def rank_fn(train):
        train = tuple(train)
        if train not in cache:
            k = subset_key(train)
            fold = js["global"] if list(train) == keys else js["folds"].get(k)
            if fold is None:
                raise SystemExit(f"score2d has no preselection for training set {train}")
            cache[train] = ranking_3d(root, label, train, fold["ranked_set"], arrs, keys, mode)
        return cache[train]

    return rank_fn


def stage_ksel(args, root: pathlib.Path) -> dict:
    label = args.label
    mode = agg_mode(args)
    tag = agg_tag(mode)
    js, arrs = load_score2d(root, label, tag)
    keys = list(js["records"])
    rank_fn = make_rank_fn(root, label, js, arrs, mode)
    group_of = rc.scene_map(keys)

    def eval_fn(cid, unit):
        rows = load_rows(root, unit, cid)
        if rows is None:
            raise SystemExit(f"[{unit}] no rows for {cid}")
        return record_metrics(rows)[PRIMARY[label]]

    # P4 leaves one *scene* out (판정 보완): units = scenes; a scene's score = mean of its runs; the
    # fold rankers aggregate their training records by `mode` (record mean or scene mean)
    scenes, rank_s, eval_s, records_of = pr.grouped_cv(keys, group_of, rank_fn, eval_fn)
    need = pr.expand_pairs(pr.required_pairs(scenes, pr.K_CHOICES, rank_s), records_of)
    plan: dict[str, list[str]] = {}
    for cid, k in sorted(need):
        if load_rows(root, k, cid) is None:
            plan.setdefault(k, []).append(cid)
    if plan:
        if args.no_run:
            raise SystemExit(f"missing 3D rows for {sum(map(len, plan.values()))} (combo, record) "
                             f"pairs: {plan}")
        ensure_rows(root, args, plan, need_self=True)

    res = pr.nested_cv(scenes, pr.K_CHOICES, rank_s, eval_s, n_boot=args.n_boot, seed=pr.SEED_KSEL)
    res["cv_mean_ci95"] = {k: pr.bootstrap_mean_ci(v, n_boot=args.n_boot, seed=pr.SEED_KSEL)
                           for k, v in res["cv"]["scores"].items()}
    k_star = int(res["choice"]["chosen_k"])
    ranked_all = list(rank_fn(tuple(keys)))
    combo = pr.topk(rank_fn, keys, k_star)
    payload = {**run_info("ksel", args), "label": label, "primary": PRIMARY[label],
               "provenance": rc.provenance(keys), "records": keys,
               "cv_unit": "scene", "units": scenes, "units_records": records_of,
               "k_choices": list(pr.K_CHOICES), "required_pairs": sorted(need),
               "replayed_for_ksel": plan, **res,
               "selected": {"k": k_star, "combo": combo, "cells": [c.id for c in pr.combo_cells(combo)]},
               "ranking_all_calibration_top10": ranked_all[:10],
               "single_head_fixed": k_star == 1,
               **({"aggregate": mode,
                   "aggregation": "fold rankers: mean of scene means of the training records; CV "
                                  "score of a test scene: mean of its runs; CV mean: over scenes"}
                  if mode != "record" else {})}
    write_json(root / label / "ksel" / f"ksel{tag}.json", payload)
    print(f"[ksel/{label}{tag}] k*={k_star} -> {combo}; nested mean {res['nested_mean']:.3f}")
    return payload


def load_selected(root: pathlib.Path, label: str, tag: str = "") -> Optional[dict]:
    p = root / label / "ksel" / f"ksel{tag}.json"
    return json.loads(p.read_text())["selected"] if p.is_file() else None


# ------------------------------------------------------------------------------- P5 / P6


def heldout_summary(root: pathlib.Path, keys: Sequence[str], sids: Sequence[str],
                    mode: str = "record") -> dict:
    """Per source: record-level GT numbers, means by scene group, and `t47b_metrics.summarize`
    (scene x phase strata, per record). ``mode="scene"`` adds ``scene_mean_gt_identity`` (mean of the
    held-out scenes' means: headfix and ep1800 share scene1800) and ``by_scene_id``."""
    out = {}
    for s in sids:
        rows_all, per_rec = [], {}
        for k in keys:
            rows = load_rows(root, k, s) or []
            rows_all += rows
            per_rec[k] = record_metrics(rows)
        by_scene = {}
        for grp in ("same_scene", "new_scene"):
            ks = [k for k in keys if rc.scene_group(k) == grp]
            if ks:
                by_scene[grp] = {f: float(pr.unit_mean(np.asarray([per_rec[k][f] for k in ks])))
                                 for f in ("gt_identity", "on_target", "purity_mean",
                                           "nontarget_seed_loss_per_frame",
                                           "other_object_loss_per_frame")}
                by_scene[grp]["records"] = ks
        gt_id = np.asarray([per_rec[k]["gt_identity"] for k in keys], np.float64)
        out[s] = {"per_record": per_rec,
                  "record_mean_gt_identity": float(pr.unit_mean(gt_id)),
                  "by_scene": by_scene,
                  "strata": bm.summarize(rows_all).get(s, {}) if rows_all else {}}
        if mode == "scene" and keys:
            names, gm = pr.group_means(gt_id, scene_groups(keys, mode))
            out[s]["scene_mean_gt_identity"] = float(pr.unit_mean(gm))
            out[s]["by_scene_id"] = {nm: {"gt_identity": float(v),
                                          "records": [k for k in keys if rc.scene_of(k) == nm]}
                                     for nm, v in zip(names, gm)}
    return out


def point_gt_identity(summary_entry: dict, mode: str) -> float:
    """The held-out point estimate of one source under `mode` (record mean / scene mean)."""
    return summary_entry["scene_mean_gt_identity" if mode == "scene" else "record_mean_gt_identity"]


def parity_vs_t47b(root: pathlib.Path, t47b: pathlib.Path, keys: Sequence[str]) -> dict:
    """T51 ``last:L8H2`` rows vs T47b ``l8h2`` sigma-0 rows (same frames): fields that must agree."""
    fields = ("has_target", "on_target", "identity_correct", "grounding_status", "n_cloud",
              "n_cloud_target", "n_seeds", "n_seeds_target", "chosen_n", "nontarget_seed_loss")
    out = {}
    for k in keys:
        p = t47b / "e2e" / f"rows__{k}__s0.jsonl"
        mine = load_rows(root, k, SERVING.id)
        if not p.is_file() or mine is None:
            out[k] = {"available": False}
            continue
        with p.open() as fh:
            theirs = {r["frame"]: r for r in (json.loads(x) for x in fh if x.strip())
                      if r["source"] == "l8h2"}
        same, diffs = 0, []
        for r in mine:
            t = theirs.get(r["frame"])
            if t is None:
                continue
            bad = [f for f in fields if r.get(f) != t.get(f)]
            x, y = r.get("centroid_error_m"), t.get("centroid_error_m")
            if (x is None) != (y is None) or (isinstance(x, float) and isinstance(y, float)
                                               and np.isfinite(x) != np.isfinite(y)) or \
                    (isinstance(x, float) and isinstance(y, float) and np.isfinite(x)
                     and abs(x - y) > 1e-9):
                bad.append("centroid_error_m")
            if bad:
                diffs.append({"frame": r["frame"], "fields": bad})
            else:
                same += 1
        out[k] = {"available": True, "n_frames": len(mine), "identical": same, "differing": diffs}
    return out


def stage_heldout(args, root: pathlib.Path) -> dict:
    mode = agg_mode(args)
    tag = agg_tag(mode)
    keys = rc.resolve_keys(args.records, HELDOUT_KEYS)
    pairing = [rc.check_pairing(k) for k in keys]
    selected = {l: load_selected(root, l, tag) for l in args.labels}
    selected = {l: v for l, v in selected.items() if v}
    if not selected:
        raise SystemExit(f"no ksel{tag} output — run --stage ksel first")
    sids = sorted({SERVING.id, *[v["combo"] for v in selected.values()]})
    rep = ensure_rows(root, args, {k: sids for k in keys}, need_self=False)
    if args.max_frames:
        return rep
    payload = {**run_info("heldout", args), "records": pairing, "selected": selected,
               "serving": SERVING.id, "sources": sids,
               "summary": heldout_summary(root, keys, sids, mode),
               "parity_l8h2_vs_t47b_s0": parity_vs_t47b(root, t47b_dir(args), keys),
               **({"aggregate": mode} if mode != "record" else {})}
    write_json(root / "heldout" / f"heldout{tag}.json", payload)
    print(f"[heldout{tag}] {len(sids)} sources x {len(keys)} records -> {root / 'heldout'}")
    return payload


def stage_ablation(args, root: pathlib.Path) -> dict:
    mode = agg_mode(args)
    tag = agg_tag(mode)
    keys = rc.resolve_keys(args.records, HELDOUT_KEYS)
    t47 = t47_dir(args)
    tc = t47_cells(t47)
    sel = load_selected(root, "gt", tag)
    if sel is None:
        raise SystemExit(f"ablation needs the GT selection — run --stage ksel --label gt first "
                         f"(ksel{tag}.json)")
    k_star = int(sel["k"])
    filt = load_filter(root)
    randoms = pr.random_cell_sets(filt["eligible"], n=args.n_random, k=k_star,
                                  seed=pr.SEED_RANDOM_CELLS)
    js_gt, _ = load_score2d(root, "gt", tag)
    candidates = list(js_gt["global"]["scored_set"]) if not args.no_candidates else []
    named = {"selected_gt": sel["combo"], "l8h2": SERVING.id,
             "paper3_s0": pr.combo_id(tc["paper3"]), "paper3_s1": "paper3_s1",
             "headavg": "headavg_last", "oracle3_t47": pr.combo_id(tc["oracle3"])}
    self_sel = load_selected(root, "self", tag)
    if self_sel:
        named["selected_self"] = self_sel["combo"]
    sids = sorted(set(named.values()) | set(randoms) | set(candidates))
    rep = ensure_rows(root, args, {k: sids for k in keys}, need_self=False)
    if args.max_frames:
        return rep
    summ = heldout_summary(root, keys, sids, mode)

    def rm(s):  # the point estimate of `mode` (record mean by default, scene mean with --aggregate scene)
        return point_gt_identity(summ[s], mode)

    rand_vals = np.asarray([rm(s) for s in randoms])
    cand_vals = {c: rm(c) for c in candidates}
    best_c = max(cand_vals, key=lambda c: (np.nan_to_num(cand_vals[c], nan=-1), c)) if cand_vals else None
    sel_v = rm(sel["combo"])
    payload = {**run_info("ablation", args), "records": keys, "k_star_gt": k_star,
               "named": named, "random_sets": randoms, "random_seed": pr.SEED_RANDOM_CELLS,
               "random_pool": "P1-eligible cells (filter.json eligible), k* cells per draw",
               "random_distribution": {
                   "values": rand_vals.tolist(),
                   "median": float(np.nanmedian(rand_vals)) if rand_vals.size else float("nan"),
                   "p05": float(np.nanpercentile(rand_vals, 5)) if rand_vals.size else float("nan"),
                   "p95": float(np.nanpercentile(rand_vals, 95)) if rand_vals.size else float("nan"),
                   "share_below_selected_gt": float(np.mean(rand_vals < sel_v)) if rand_vals.size else float("nan"),
                   "share_at_or_above_selected_gt": float(np.mean(rand_vals >= sel_v)) if rand_vals.size else float("nan")},
               "oracle_restricted": {
                   "definition": "best single cell on the held-out GT among the GT 3D-scored set "
                                 "(global preselection + forced) — chosen on held-out, an upper "
                                 "reference only",
                   "cell": best_c, "value": cand_vals.get(best_c) if best_c else None,
                   "all": cand_vals},
               "record_mean_gt_identity": {n: summ[s]["record_mean_gt_identity"]
                                           for n, s in named.items()},
               "summary": summ,
               **({"aggregate": mode,
                   "scene_mean_gt_identity": {n: rm(s) for n, s in named.items()},
                   "aggregation": "random distribution and restricted oracle use the held-out "
                                  "scene mean (headfix + ep1800 = scene1800 averaged first)"}
                  if mode != "record" else {})}
    write_json(root / "ablation" / f"ablation{tag}.json", payload)
    print(f"[ablation{tag}] {len(sids)} sources x {len(keys)} records -> {root / 'ablation'}")
    return payload


# ------------------------------------------------------------------------------ annotate


def stage_annotate(args, root: pathlib.Path) -> dict:
    """Fallback (T51.task.md 되돌아올 지점, exploratory): how many human-annotated calibration frames
    would recover the GT selection? Frame-level draws from the GT 3D rows of the ranked set."""
    mode = agg_mode(args)
    tag = agg_tag(mode)
    js, arrs = load_score2d(root, "gt", tag)
    keys = list(js["records"])
    ranked = list(js["global"]["ranked_set"])
    cols = []
    for c in ranked:
        v = []
        for k in keys:
            rows = load_rows(root, k, c)
            if rows is None:
                raise SystemExit(f"[{k}] no rows for {c} — run --stage score3d")
            v += [float(bool(r.get("identity_correct"))) for r in rows]
        cols.append(v)
    X = np.asarray(cols, np.float64).T  # (frames, N)
    tie = calib_mean(np.stack([lift_2d(arrs, c) for c in ranked], axis=1), keys, mode)
    full = pr.rank_order(calib_mean(metric_matrix(root, keys, ranked, "gt_identity"), keys, mode), tie)
    ns = [n for n in args.annotate_frames if n <= X.shape[0]] + [X.shape[0]]
    curve = pr.frames_needed_curve(X, np.ones(len(ranked), bool), list(full[:3]), sorted(set(ns)),
                                   n_boot=args.n_boot, seed=pr.SEED_BOOTSTRAP, tiebreak=tie)
    payload = {**run_info("annotate", args), "records": keys, "n_frames_pool": int(X.shape[0]),
               "reference_top3": [ranked[i] for i in full[:3]],
               "note": "frames pooled over the calibration records, drawn without replacement; the "
                       f"reference is the {mode}-mean GT ranking (P2)", "curve": curve,
               **({"aggregate": mode} if mode != "record" else {})}
    write_json(root / "gt" / "annotate" / f"annotate{tag}.json", payload)
    print(f"[annotate{tag}] {len(curve)} budgets -> {root / 'gt' / 'annotate'}")
    return payload


# ------------------------------------------------------------------------------- compare


def stage_compare(args, root: pathlib.Path) -> dict:
    mode = agg_mode(args)
    tag = agg_tag(mode)
    out: dict[str, Any] = {**run_info("compare", args)}
    if mode != "record":
        out["aggregate"] = mode
    got = {}
    for l in LABELS:
        d = {}
        for name, path in (("score3d", root / l / "score3d" / f"score3d{tag}.json"),
                           ("bootstrap", root / l / "bootstrap" / f"bootstrap{tag}.json"),
                           ("ksel", root / l / "ksel" / f"ksel{tag}.json")):
            if path.is_file():
                d[name] = json.loads(path.read_text())
        got[l] = d
    if not all("score3d" in got[l] for l in LABELS):
        raise SystemExit(f"compare needs score3d{tag} for both labels")
    top = {l: [r["cell"] for r in got[l]["score3d"]["ranking"][:3]] for l in LABELS}
    out["p2_top1"] = {l: top[l][0] if top[l] else None for l in LABELS}
    out["p2_top1_same"] = out["p2_top1"]["gt"] == out["p2_top1"]["self"]
    out["p2_top3"] = top
    out["p2_top3_overlap"] = len(set(top["gt"]) & set(top["self"]))
    if all("bootstrap" in got[l] for l in LABELS):
        def p1(l, cell):
            return next((r["p_rank1"] for r in got[l]["bootstrap"]["all"] if r["cell"] == cell), None)
        out["p3_cross"] = {"p_rank1_self_of_gt_top1": p1("self", out["p2_top1"]["gt"]),
                           "p_rank1_gt_of_self_top1": p1("gt", out["p2_top1"]["self"])}
    if all("ksel" in got[l] for l in LABELS):
        sel = {l: got[l]["ksel"]["selected"] for l in LABELS}
        out["p4_selected"] = sel
        out["p4_same_k"] = sel["gt"]["k"] == sel["self"]["k"]
        out["p4_same_combo"] = sel["gt"]["combo"] == sel["self"]["combo"]
    ho = root / "heldout" / f"heldout{tag}.json"
    if ho.is_file() and "p4_selected" in out:
        h = json.loads(ho.read_text())["summary"]
        g, s = out["p4_selected"]["gt"]["combo"], out["p4_selected"]["self"]["combo"]
        if g in h and s in h:
            recs = list(h[g]["per_record"])
            out["p5_heldout_gt_identity"] = {
                "per_record": {k: {"gt_selected": h[g]["per_record"][k]["gt_identity"],
                                   "self_selected": h[s]["per_record"][k]["gt_identity"],
                                   "self_minus_gt": h[s]["per_record"][k]["gt_identity"]
                                   - h[g]["per_record"][k]["gt_identity"]} for k in recs},
                "record_mean": {"gt_selected": h[g]["record_mean_gt_identity"],
                                "self_selected": h[s]["record_mean_gt_identity"]}}
            if mode == "scene":
                out["p5_heldout_gt_identity"]["scene_mean"] = {
                    "gt_selected": h[g].get("scene_mean_gt_identity"),
                    "self_selected": h[s].get("scene_mean_gt_identity")}
    sl = root / "selflabel" / "selflabel.json"
    if sl.is_file():
        out["selflabel"] = json.loads(sl.read_text())["per_record"]
    # self-label vs GT on the same frames for the GT-selected cell (label noise vs frame subset)
    sc = got["gt"]["score3d"]["cells"]
    out["gt_top1_rates_all_vs_self_frames"] = {
        c: {"gt_identity_mean": sc[c]["gt_identity_mean"],
            "gt_identity_on_self_frames_mean": sc[c]["gt_identity_on_self_frames_mean"],
            "self_identity_mean": sc[c]["self_identity_mean"]}
        for c in top["gt"] if c in sc}
    if mode == "scene":
        out["gt_top1_rates_scene_mean"] = {c: sc[c].get("gt_identity_scene_mean") for c in top["gt"]
                                           if c in sc}
    write_json(root / "compare" / f"compare{tag}.json", out)
    print(f"[compare{tag}] -> {root / 'compare'}")
    return out


# ---------------------------------------------------------------------------------- main


STAGES = ("gt_extra", "filter", "selflabel", "score2d", "score3d", "bootstrap", "ksel", "heldout",
          "ablation", "compare", "annotate")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--label", choices=LABELS, default="gt",
                    help="score2d / bootstrap / ksel: which label calibrates")
    ap.add_argument("--labels", nargs="+", choices=LABELS, default=list(LABELS),
                    help="score3d / heldout: labels whose preselection / selection to use")
    ap.add_argument("--records", nargs="+", default=None, metavar="KEY")
    ap.add_argument("--workers", type=int, default=1, help=f"records in parallel (<= {MAX_WORKERS})")
    ap.add_argument("--source-batch", type=int, default=12,
                    help="sources replayed together per pass over a record (one AG3S each)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="smoke: first N frames only; rows go to rows_smoke/ and nothing aggregates")
    ap.add_argument("--t47-dir", default=None, help=f"T47 verify dir (default $AG3S_STORAGE_ROOT/{T47_RUN}/verify/T47)")
    ap.add_argument("--t47b-dir", default=None, help="T47b verify dir (parity of the L8H2 rows)")
    ap.add_argument("--t49-dir", default=None,
                    help="T49 verify dir: adds the noun-token query (extract/orig/<record>.npz) to Ω")
    ap.add_argument("--event", choices=("both_contact", "held"), default="both_contact",
                    help="selflabel: grasp event definition")
    ap.add_argument("--force", action="store_true", help="selflabel: recompute cached labels")
    ap.add_argument("--min-target-px", type=int, default=200)
    ap.add_argument("--preselect-top", type=int, default=pr.PRESELECT_TOP)
    ap.add_argument("--n-boot", type=int, default=pr.N_BOOT)
    ap.add_argument("--unit", default="scene",
                    help="bootstrap resampling unit: scene (pre-registered, 판정 보완), record or "
                         "block:N (exploratory)")
    ap.add_argument("--aggregate", choices=AGGREGATES, default="record",
                    help="calibration point estimate: record = unweighted mean over records "
                         "(default, original outputs); scene = mean of scene means (T51.task.md "
                         "판정 보완 → 집계 단위) in score2d · score3d · bootstrap · ksel · heldout · "
                         "ablation · annotate · compare; outputs and inputs carry the suffix __scene")
    ap.add_argument("--extra-dir", default=None,
                    help="T51 dir holding attention/attention_16d_<ep0|ep500|ep1200>.npz and "
                         "gt_cache_extra/ (default: this run's $AG3S_RUN_DIR/verify/T51)")
    ap.add_argument("--rgb-check-frames", type=int, default=3,
                    help="gt_extra: frames whose replayed RGB is compared with the recorded image")
    ap.add_argument("--n-random", type=int, default=pr.N_RANDOM)
    ap.add_argument("--no-run", action="store_true", help="ksel: refuse instead of replaying")
    ap.add_argument("--no-candidates", action="store_true",
                    help="ablation: skip the single-cell candidates (no restricted oracle)")
    ap.add_argument("--no-selflabel", action="store_true",
                    help="score3d: allow rows without self-label fields")
    ap.add_argument("--cells", nargs="+", default=None, help="score3d: extra source ids")
    ap.add_argument("--annotate-frames", type=int, nargs="+", default=[5, 10, 20, 40, 80, 120],
                    help="annotate: frame budgets (the full pool is always added)")
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--links", default="gripper")
    ap.add_argument("--on-target-radius", type=float, default=0.10)
    ap.add_argument("--label-radius", type=float, default=0.003)
    ap.add_argument("--membership-radius", type=float, default=0.005)
    return ap


def validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 1 <= args.workers <= MAX_WORKERS:
        ap.error(f"--workers must be 1..{MAX_WORKERS} (T51 CPU budget)")
    if args.source_batch < 1 or args.max_frames < 0 or args.min_target_px < 0:
        ap.error("--source-batch >= 1, --max-frames / --min-target-px >= 0")
    if args.preselect_top != pr.PRESELECT_TOP:
        print(f"[warn] --preselect-top {args.preselect_top} differs from the pre-registered "
              f"{pr.PRESELECT_TOP} (exploratory)")
    if args.n_boot < 100:
        ap.error("--n-boot must be >= 100")
    if args.records:
        rc.resolve_keys(args.records, ())
    if args.cells:
        for c in args.cells:
            if c not in SPECIAL_SOURCES:
                try:
                    pr.combo_cells(c)
                except ValueError as exc:
                    ap.error(str(exc))
    if args.unit not in ("scene", "record") and not args.unit.startswith("block:"):
        ap.error("--unit must be scene, record or block:N")
    if getattr(args, "aggregate", "record") == "scene" and args.stage == "bootstrap" \
            and args.unit != "scene":
        ap.error("--aggregate scene resamples scenes: use --unit scene")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    validate_args(ap, args)
    root = out_root()
    rc.configure(args.extra_dir or root)
    t0 = time.time()
    globals()[f"stage_{args.stage}"](args, root)
    print(f"[{STEP}] stage {args.stage} done in {time.time() - t0:.1f}s -> {root}")


if __name__ == "__main__":
    main()
