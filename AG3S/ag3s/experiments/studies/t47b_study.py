"""T47b — head aggregation variants (B1), label-based 3D metrics (B2), raw-amplitude audit (B3) and
the sink diagnostic (#5), on top of T47's run.

    source scripts/setup/env.sh
    export AG3S_RUN_DIR=<T47's run dir>          # reuses verify/T47/{gt_cache,select,validate}
    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-ag3s/bin/python -m \\
        AG3S.ag3s.experiments.studies.t47b_study --stage {check,audit,sink,b2d,e2e,b3,parity,summarize}

All output goes to ``$AG3S_RUN_DIR/verify/T47b/``; the CLI refuses to run without ``AG3S_RUN_DIR``
and (except ``audit``) without ``$AG3S_RUN_DIR/verify/T47``. Head sets are **read** from T47's
``select/selection.json`` (common and per-camera top-3, ``d0_last``, tau source ``cam``) and
``validate/oracle.json`` (GT-oracle top-3, chosen on the selection records) — never re-selected.

Stages (MuJoCo = osmesa replay, CPU; none uses the GPU):

``check``     #0  T47 run dir, GT cache file list + sha256, select / oracle file hashes, the head sets,
              npz <-> record pairing and GT-cache t_step. No MuJoCo.
``audit``     B3  the file:line table (`t47b_b3.AUDIT`) with current line numbers. No MuJoCo.
``sink``      #5  per cell (L2-17 x 8) x camera over every record / frame at d0 · last: modal-argmax
              fraction, top-1 share, components, largest component, entropy, S_cam. No MuJoCo.
``b2d``       #2  2D held-out scores (lift, IoU, class mass, ...) per source x sigma, from the GT cache,
              plus head-mass / camera diagnostics of the aggregation. No MuJoCo.
``e2e``       #3 #4  live-path 3D grounding per source with T47's row fields + B2. MuJoCo.
``b3``        #1  grounding on map vs factor x map (default x3 and the exact control x4). MuJoCo.
``parity``    T47b e2e rows at sigma 0 vs T47's e2e rows (same source, same frame). No MuJoCo.
``summarize`` re-aggregate the e2e rows of one sigma (records run separately). No MuJoCo.

Sources (`source_names`): ``l8h2`` (serving cell) · ``common3_raw_sum`` (T47's failed common top-3,
reference row) · ``{oracle3,camera3,headavg}_{raw_sum,mass_norm_joint,mass_norm_per_camera}``
(T47b.task.md "범위 변경" 1). Every name except ``l8h2`` carries the smoothing as ``_s<sigma>``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t47_head_metrics as hm
from AG3S.ag3s.experiments.studies import t47_head_selection as hs
from AG3S.ag3s.experiments.studies import t47b_aggregation as ag
from AG3S.ag3s.experiments.studies import t47b_metrics as bm

STEP = "T47b"
SETTING = "d0_last"  # T47 pre-registration: denoise 0 · agg last (the serving query)
DENOISE, AGG = 0, "last"
HEAD_SETS = ("oracle3", "camera3", "headavg")
SOURCE_BASES = (("l8h2", "common3_raw_sum")
                + tuple(f"{s}_{m}" for s in HEAD_SETS for m in ag.AGGREGATIONS))
DEFAULT_B3_SOURCES = ("l8h2", "oracle3_raw_sum")
MAX_WORKERS = hs.MAX_WORKERS
MAX_SINK_WORKERS = 2
#: T47 e2e source -> T47b base name (for `parity`, sigma 0).
PARITY_MAP = {"l8h2": "l8h2", "topk_common": "common3_raw_sum", "topk_camera": "camera3_raw_sum",
              "head_avg": "headavg_raw_sum", "oracle": "oracle3_raw_sum"}


# ------------------------------------------------------------------------------ plumbing


def _run_dir() -> pathlib.Path:
    run_dir = os.environ.get("AG3S_RUN_DIR")
    if not run_dir:
        raise SystemExit(
            "AG3S_RUN_DIR is not set. T47b reuses T47's run (GT cache, select, oracle):\n"
            "  export AG3S_RUN_DIR=<the run dir that holds verify/T47>")
    return pathlib.Path(run_dir)


def out_root() -> pathlib.Path:
    """``$AG3S_RUN_DIR/verify/T47b`` (created)."""
    from common.run_output import resolve_output_path

    run_dir = _run_dir()
    root = resolve_output_path("verify/T47b", run_dir=run_dir, default_name="verify/T47b")
    root.mkdir(parents=True, exist_ok=True)
    return root


def t47_root(*, required: bool = True) -> pathlib.Path:
    """``$AG3S_RUN_DIR/verify/T47`` — read only; refuses when it does not exist."""
    from common.run_output import resolve_output_path

    run_dir = _run_dir()
    root = resolve_output_path("verify/T47", run_dir=run_dir, default_name="verify/T47")
    if required and not root.is_dir():
        raise SystemExit(f"no T47 output at {root} — T47b reuses T47's run; point AG3S_RUN_DIR "
                         "at the run that holds verify/T47")
    return root


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: pathlib.Path, payload: Any) -> None:
    hs.write_json(path, bm.jsonable(payload))


def run_info(stage: str, args: argparse.Namespace) -> dict:
    return {"step": STEP, "stage": stage, "argv": sys.argv,
            "args": {k: v for k, v in vars(args).items()},
            "started_kst": _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(),
            "git": hs.git_state(), "storage_root": str(hs.storage_root()),
            "run_dir": str(_run_dir()), "python": sys.version.split()[0]}


def sigma_tag(sigma: float) -> str:
    return f"s{float(sigma):g}"


# ----------------------------------------------------------------------------- head sets


def load_head_sets(t47: pathlib.Path) -> dict[str, Any]:
    """The head sets T47 wrote (never re-selected) + the hashes of the files they came from."""
    sel_p, ora_p = t47 / "select" / "selection.json", t47 / "validate" / "oracle.json"
    for p in (sel_p, ora_p):
        if not p.is_file():
            raise SystemExit(f"missing {p} — T47 select / validate must have run")
    sel = json.loads(sel_p.read_text())
    ora = json.loads(ora_p.read_text())
    if SETTING not in sel["settings"] or SETTING not in ora["settings"]:
        raise SystemExit(f"T47 outputs have no setting {SETTING}")
    st = sel["settings"][SETTING]["tau_source"]["cam"]
    per_cam = st["per_camera"]
    return {
        "common3": [tuple(c) for c in st["common"]["topk"]["3"]],
        "camera3": [[tuple(c) for c in per_cam[cam]["topk"]["3"]] for cam in hs.POLICY_CAMERAS],
        "oracle3": [tuple(c) for c in ora["settings"][SETTING]["top3"]],
        "l8h2": [(hs.SERVING_CELL["layer"], hs.SERVING_CELL["head"])],
        "oracle_records": ora.get("records"),
        "files": {"selection.json": {"path": str(sel_p), "sha256": sha256(sel_p)},
                  "oracle.json": {"path": str(ora_p), "sha256": sha256(ora_p)}},
        "selection_policy": sel.get("policy"),
    }


def source_names(bases: Sequence[str], sigma: float) -> dict[str, str]:
    """Base name -> row name (``l8h2`` unchanged, the rest ``_s<sigma>``)."""
    bad = [b for b in bases if b not in SOURCE_BASES]
    if bad:
        raise SystemExit(f"unknown source(s) {bad}; have {list(SOURCE_BASES)}")
    return {b: (b if b == "l8h2" else f"{b}_{sigma_tag(sigma)}") for b in bases}


def setting_block(att: np.ndarray, denoise: list[int], aggs: list[str]) -> np.ndarray:
    """``(F, C, L, H, 16, 16)`` (float16 as stored) of the d0 · last setting."""
    di, ai = denoise.index(DENOISE), aggs.index(AGG)
    return np.moveaxis(np.asarray(att[:, di, ai]), 3, 1)


def build_sources(att: np.ndarray, denoise: list[int], aggs: list[str], heads: dict,
                  bases: Sequence[str], sigma: float) -> dict[str, np.ndarray]:
    """Row name -> ``(F, 3, 16, 16)`` float64 combined maps of one record."""
    block = setting_block(att, denoise, aggs)
    names = source_names(bases, sigma)
    out: dict[str, np.ndarray] = {}
    for base, name in names.items():
        if base == "l8h2":
            sd, sa = denoise.index(hs.SERVING_CELL["denoise"]), aggs.index(hs.SERVING_CELL["agg"])
            out[name] = np.asarray(att[:, sd, sa, hs.SERVING_CELL["layer"],
                                       hs.SERVING_CELL["head"]], np.float64)
            continue
        hset, mode = base.split("_", 1)
        if hset == "common3":
            out[name] = ag.combine_heads(block, heads["common3"], sigma, mode=mode)
        elif hset == "oracle3":
            out[name] = ag.combine_heads(block, heads["oracle3"], sigma, mode=mode)
        elif hset == "camera3":
            out[name] = ag.combine_heads_per_camera(block, heads["camera3"], sigma, mode=mode)
        elif hset == "headavg":
            out[name] = ag.head_average_variant(block, sigma, mode=mode)
        else:  # pragma: no cover — SOURCE_BASES guards this
            raise SystemExit(f"unknown head set {hset}")
    return out


def _refuse_oracle_on_selection(bases: Sequence[str], keys: Sequence[str]) -> None:
    if any(b.startswith("oracle3") for b in bases) and any(
            hs.RECORDS[k].role == "selection" for k in keys):
        raise SystemExit("the GT-oracle top-3 is chosen on the selection records' GT — it is not "
                         "scored there (T47.task.md #7); drop the oracle3_* sources")


# ------------------------------------------------------------------------------ check / audit


def stage_check(args, root: pathlib.Path) -> dict:
    t47 = t47_root()
    gt_dir = t47 / "gt_cache"
    files = sorted(p for p in gt_dir.glob("*") if p.is_file()) if gt_dir.is_dir() else []
    cache = [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
    heads = load_head_sets(t47)
    records = []
    for k in hs.RECORDS:
        rec: dict[str, Any] = {"key": k, "role": hs.RECORDS[k].role,
                               "scene_group": bm.scene_group(k)}
        if not hs.record_available(k):
            rec["available"] = False
            records.append(rec)
            continue
        rec["available"] = True
        rec["pairing"] = hs.check_pairing(k)
        cp = hs.gt_cache_path(t47, k)
        if cp.exists():
            with np.load(cp, allow_pickle=False) as d:
                t_gt = [int(t) for t in d["t_step"]]
            with np.load(hs.npz_path(k), allow_pickle=False) as d:
                t_npz = [int(t) for t in d["t_step"]]
            rec["gt_cache_t_step_matches_npz"] = t_gt == t_npz
            rec["gt_cache_frames"] = len(t_gt)
        else:
            rec["gt_cache_t_step_matches_npz"] = None
        records.append(rec)
    payload = {**run_info("check", args), "t47_root": str(t47),
               "gt_cache": {"n_files": len(cache), "files": cache},
               "head_sets": {k: v for k, v in heads.items()},
               "records": records,
               "t47_e2e_args": _t47_e2e_args(t47)}
    write_json(root / "check" / "check.json", payload)
    print(f"[check] {len(cache)} GT-cache files; head sets: common3={heads['common3']} "
          f"oracle3={heads['oracle3']} camera3={heads['camera3']}")
    return payload


def _t47_e2e_args(t47: pathlib.Path) -> dict:
    """The smoothing / k T47's e2e ran with (its summaries' ``args``) — T47 e2e used sigma 0."""
    out = {}
    for p in sorted((t47 / "e2e").glob("summary__*.json")):
        a = json.loads(p.read_text()).get("args", {})
        out[p.name] = {k: a.get(k) for k in ("sigma", "k", "e2e_denoise", "e2e_agg", "sources")}
    return out


def stage_audit(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t47b_b3 as b3

    table = b3.audit_table()
    payload = {**run_info("audit", args), "audit": table,
               "moved_or_missing": [r["id"] for r in table if not r["found"] or r.get("moved")]}
    write_json(root / "b3" / "audit.json", payload)
    print(f"[audit] {len(table)} points; moved/missing: {payload['moved_or_missing']}")
    return payload


# ------------------------------------------------------------------------------- sink (#5)


def _sink_one(key: str) -> dict:
    from AG3S.ag3s.experiments.studies import t47b_sink as sk

    att, dn, agg_list, _ = hs.load_attention(key)
    block = np.asarray(setting_block(att, dn, agg_list), np.float64)  # (F, C, L, H, 16, 16)
    block = np.moveaxis(block[:, :, 2:], 1, 3)  # (F, L-2, H, C, 16, 16)
    st = sk.map_stats(block)
    return {"key": key, "stats": st, "mean_map": block.mean(axis=0), "n_frames": block.shape[0]}


def stage_sink(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t47b_sink as sk

    t47 = t47_root()
    heads = load_head_sets(t47)
    keys, missing = hs.available_keys(hs.resolve_keys(args.records, list(hs.RECORDS)))
    if args.workers > 1 and len(keys) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=min(args.workers, MAX_SINK_WORKERS, len(keys)),
                                    mp_context=mp.get_context("spawn")) as ex:
            res = [f.result() for f in [ex.submit(_sink_one, k) for k in keys]]
    else:
        res = [_sink_one(k) for k in keys]
    fields = list(res[0]["stats"])
    per_frame = {f: np.concatenate([r["stats"][f] for r in res]) for f in fields}
    rec_idx = np.concatenate([np.full(r["n_frames"], i) for i, r in enumerate(res)])
    frame_idx = np.concatenate([np.arange(r["n_frames"]) for r in res])
    pooled = sk.summarize_cells(per_frame)
    per_record = {r["key"]: sk.summarize_cells(r["stats"]) for r in res}
    weights = np.asarray([r["n_frames"] for r in res], np.float64)
    mean_map = np.tensordot(weights / weights.sum(), np.stack([r["mean_map"] for r in res]), 1)
    n_layers = per_frame["S_cam"].shape[1] + 2
    layers = list(range(2, n_layers))
    marks = {"l8h2": heads["l8h2"], "common3": heads["common3"], "oracle3": heads["oracle3"],
             **{f"camera3_{c}": heads["camera3"][i] for i, c in enumerate(hs.POLICY_CAMERAS)}}
    marked = {}
    for name, cells in marks.items():
        cams = ([hs.POLICY_CAMERAS.index(name.split("camera3_", 1)[1])]
                if name.startswith("camera3_") else range(len(hs.POLICY_CAMERAS)))
        rows = []
        for l, h in cells:
            for ci in cams:
                li = int(l) - 2
                v = {k: pooled[k][li, int(h), ci] for k in pooled if k != "modal_patch_rc"}
                rc = pooled["modal_patch_rc"][li, int(h), ci]
                rows.append({"layer": int(l), "head": int(h), "camera": hs.POLICY_CAMERAS[ci],
                             **{k: (float(x) if np.ndim(x) == 0 else x) for k, x in v.items()},
                             "modal_patch_rc": [int(rc[0]), int(rc[1])],
                             "descriptive_label": sk.classify_cell(
                                 float(v["modal_fraction"]), float(v["top1_share_mean"]),
                                 float(v["largest_share_of_grid_mean"]))})
        marked[name] = rows
    arrays = {"layers": np.asarray(layers), "heads": np.arange(per_frame["S_cam"].shape[2]),
              "cameras": np.asarray(hs.POLICY_CAMERAS), "records": np.asarray([r["key"] for r in res]),
              "frame_record": rec_idx, "frame_index": frame_idx, "mean_map": mean_map.astype(np.float32),
              **{f"frame__{k}": v for k, v in per_frame.items()},
              **{f"pooled__{k}": v for k, v in pooled.items()},
              **{f"record__{key}__{k}": v for key, s in per_record.items() for k, v in s.items()}}
    out = root / "sink"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "sink_arrays.npz", **arrays)
    payload = {**run_info("sink", args), "records": [r["key"] for r in res], "excluded": missing,
               "n_frames": {r["key"]: r["n_frames"] for r in res}, "setting": SETTING,
               "axes": "pooled__* are (layer 2..17, head 0..7, camera); frame__* lead with frame",
               "definitions": sk.__doc__, "descriptive_label_thresholds":
                   {"sink_modal": 0.8, "sink_top1": 0.3, "blob_grid": 0.15,
                    "note": "a label for reading only; not used to select anything"},
               "head_sets": {k: v for k, v in heads.items() if k != "files"},
               "head_set_files": heads["files"], "marked": marked}
    write_json(out / "sink.json", payload)
    print(f"[sink] {sum(r['n_frames'] for r in res)} frames x {len(layers)}x8 cells x 3 cameras "
          f"-> {out}")
    return payload


# ------------------------------------------------------------------------------- b2d (#2)


def _diagnostics(block: np.ndarray, heads: dict, maps: dict[str, np.ndarray], sigma: float
                 ) -> dict:
    """Head-mass dominance of each head set and camera effects of the aggregation (one record)."""
    out: dict[str, Any] = {}
    for hset in ("common3", "oracle3"):
        m = ag.head_masses(block, heads[hset])
        out[hset] = {"dominance_mean": float(np.nanmean(m["dominance"])),
                     "dominance_median": float(np.nanmedian(m["dominance"])),
                     "share_of_sum_mean": np.nanmean(m["share_of_sum"], axis=0).tolist(),
                     "joint_mass_mean": m["joint"].mean(axis=0).tolist(),
                     "camera_share_mean": np.nanmean(m["camera_share"], axis=0).tolist(),
                     "zero_mass": ag.zero_mass_counts(block, heads[hset])}
    out["camera3"] = {cam: {"dominance_mean": float(np.nanmean(
        ag.head_masses(block, heads["camera3"][ci])["dominance"])),
        "zero_mass": ag.zero_mass_counts(block, heads["camera3"][ci])}
        for ci, cam in enumerate(hs.POLICY_CAMERAS)}
    tag = sigma_tag(sigma)
    for hset in HEAD_SETS:
        ref = maps.get(f"{hset}_raw_sum_{tag}")
        if ref is None:
            continue
        top_ref = ag.top_camera(ref)
        entry = {"raw_sum_camera_mass_share_mean": np.nanmean(ag.camera_mass_share(ref),
                                                               axis=0).tolist()}
        for mode in ag.AGGREGATIONS[1:]:
            mm = maps.get(f"{hset}_{mode}_{tag}")
            if mm is None:
                continue
            entry[mode] = {"top_camera_agrees_with_raw_sum": float((ag.top_camera(mm)
                                                                    == top_ref).mean()),
                           "camera_mass_share_mean": np.nanmean(ag.camera_mass_share(mm),
                                                                axis=0).tolist()}
        out[f"aggregation_effect__{hset}"] = entry
    return out


def stage_b2d(args, root: pathlib.Path) -> dict:
    t47 = t47_root()
    heads = load_head_sets(t47)
    keys, missing = hs.available_keys(hs.resolve_keys(args.records, hs.HELDOUT_KEYS))
    bases = list(args.sources or SOURCE_BASES)
    _refuse_oracle_on_selection(bases, keys)
    data = {k: (hs.load_attention(k), hs.load_gt(t47, k)) for k in keys}
    payload: dict[str, Any] = {**run_info("b2d", args), "records": keys, "excluded": missing,
                               "setting": SETTING, "min_target_px": args.min_target_px,
                               "head_sets": {k: v for k, v in heads.items() if k != "files"},
                               "head_set_files": heads["files"],
                               "aggregation_role": ag.AGGREGATION_ROLE, "sigmas": {}}
    out = root / "b2d"
    out.mkdir(parents=True, exist_ok=True)
    for sigma in args.sigmas:
        tag = sigma_tag(sigma)
        scores: dict[str, list] = {}
        keep_l, cam_l, rec_l, frame_l, diags = [], [], [], [], {}
        for ri, (k, ((att, dn, agl, _), gt)) in enumerate(data.items()):
            maps = build_sources(att, dn, agl, heads, bases, sigma)
            block = setting_block(att, dn, agl)
            diags[k] = _diagnostics(block, heads, maps, sigma)
            f = att.shape[0]
            tpx = np.asarray(gt["class_px"])[:, :, 0]
            keep_l.append((tpx >= args.min_target_px).reshape(-1))
            cam_l.append(np.tile(np.arange(len(hs.POLICY_CAMERAS)), f))
            rec_l.append(np.full(f * len(hs.POLICY_CAMERAS), ri))
            frame_l.append(np.repeat(np.arange(f), len(hs.POLICY_CAMERAS)))
            for n, m in maps.items():
                scores.setdefault(n, []).append(hs._flat_scores(m, gt))
            np.savez_compressed(out / f"maps__{tag}__{k}.npz",
                                **{n: m.astype(np.float32) for n, m in maps.items()})
        keep, cam_of = np.concatenate(keep_l), np.concatenate(cam_l)
        rec_of, frame_of = np.concatenate(rec_l), np.concatenate(frame_l)
        rec_names = list(data)
        merged = {n: {fld: np.concatenate([s[fld] for s in v]) for fld in v[0]}
                  for n, v in scores.items()}
        sources = {}
        for n, sc in merged.items():
            agg = hs._aggregate(sc, keep, cam_of, rec_of, rec_names)
            groups = {}
            for grp in ("same_scene", "new_scene"):
                recs = [r for r in rec_names if bm.scene_group(r) == grp]
                if recs:
                    groups[grp] = {
                        m: float(np.nanmean([agg["per_record"][r][m] for r in recs]))
                        for m in ("target_mass", "lift", "peak_on_target", "iou@0.5", "iou@0.25")}
                    groups[grp]["records"] = recs
                    groups[grp]["class_mass"] = {c: float(np.nanmean(
                        [agg["per_record"][r]["class_mass"][c] for r in recs]))
                        for c in hm.CLASS_NAMES}
            agg["record_mean_by_scene"] = groups
            sources[n] = agg
        per_sample = {"keep": keep, "camera": cam_of, "record": rec_of, "frame": frame_of,
                      "record_names": np.asarray(rec_names), "class_names": np.asarray(hm.CLASS_NAMES)}
        for n, sc in merged.items():
            for fld, v in sc.items():
                per_sample[f"{n}__{fld}"] = v
        np.savez_compressed(out / f"per_sample__{tag}.npz", **per_sample)
        payload["sigmas"][tag] = {"sigma": sigma, "n_samples_total": int(keep.size),
                                  "n_samples_kept": int(keep.sum()), "sources": sources,
                                  "diagnostics": diags}
        print(f"[b2d] sigma {sigma:g}: {len(merged)} sources, kept {int(keep.sum())}/{keep.size}")
    write_json(out / "b2d.json", payload)
    print(f"[b2d] wrote {out / 'b2d.json'}")
    return payload


# ------------------------------------------------------------------------------- e2e (#3 #4)


def _replay_opts(args) -> dict:
    return {"voxel": args.voxel, "range_max": args.range_max, "links": args.links,
            "esdf_backend": args.esdf_backend}


def _e2e_one(key: str, out_dir: str, sources: dict, frames, opts: dict, tag: str) -> dict:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.studies.t47b_replay import run_b2_record

    rows = run_b2_record(hs.record_path(key), target=hs.target_of(key), sources=sources,
                         frames=frames, log=lambda s: print(f"[{key}]{s}", flush=True), **opts)
    for r in rows:
        r["record"] = key
        r["scene_group"] = bm.scene_group(key)
    path = pathlib.Path(out_dir) / f"rows__{key}__{tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(bm.jsonable(r)) + "\n")
    return {"key": key, "rows": str(path), "n_rows": len(rows)}


def _summarize_rows(root: pathlib.Path, tag: str, args, extra: dict) -> dict:
    rows = []
    files = sorted((root / "e2e").glob(f"rows__*__{tag}.jsonl"))
    for p in files:
        with p.open() as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise SystemExit(f"no e2e rows for {tag} under {root / 'e2e'}")
    payload = {**run_info("e2e-summary", args), **extra, "row_files": [str(p) for p in files],
               "n_rows": len(rows),
               "aggregation": "per record (frames pooled within a record), then the unweighted "
                              "mean over records; strata scene x phase (t47b_metrics.summarize)",
               "summary": bm.summarize(rows)}
    write_json(root / "e2e" / f"summary__{tag}.json", payload)
    print(f"[e2e] wrote {root / 'e2e' / f'summary__{tag}.json'} ({len(rows)} rows)")
    return payload


def stage_e2e(args, root: pathlib.Path) -> dict:
    t47 = t47_root()
    heads = load_head_sets(t47)
    keys, missing = hs.available_keys(hs.resolve_keys(args.records, hs.HELDOUT_KEYS))
    bases = list(args.sources or SOURCE_BASES)
    _refuse_oracle_on_selection(bases, keys)
    pairing = [hs.check_pairing(k) for k in keys]
    tag = sigma_tag(args.sigma)
    jobs = []
    for k in keys:
        att, dn, agl, _ = hs.load_attention(k)
        maps = build_sources(att, dn, agl, heads, bases, args.sigma)
        frames = None if not args.max_frames else list(range(min(args.max_frames, att.shape[0])))
        jobs.append((k, {n: m.astype(np.float32) for n, m in maps.items()}, frames))
        del att
    opts = {**_replay_opts(args), "on_target_radius": args.on_target_radius,
            "label_radius": args.label_radius, "ball_voxel": args.ball_voxel,
            "membership_radius": args.membership_radius}
    out_dir = root / "e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.workers > 1 and len(jobs) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                    mp_context=mp.get_context("spawn")) as ex:
            res = [f.result() for f in [ex.submit(_e2e_one, k, str(out_dir), s, fr, opts, tag)
                                        for k, s, fr in jobs]]
    else:
        res = [_e2e_one(k, str(out_dir), s, fr, opts, tag) for k, s, fr in jobs]
    return _summarize_rows(root, tag, args, {
        "records_run": [r["key"] for r in res], "records": pairing, "excluded": missing,
        "sigma": args.sigma, "setting": SETTING, "sources": source_names(bases, args.sigma),
        "head_sets": {k: v for k, v in heads.items() if k != "files"},
        "head_set_files": heads["files"], "aggregation_role": ag.AGGREGATION_ROLE,
        "options": opts, "workers": args.workers})


def stage_summarize(args, root: pathlib.Path) -> dict:
    return _summarize_rows(root, sigma_tag(args.sigma), args, {"sigma": args.sigma,
                                                               "resummarized": True})


# --------------------------------------------------------------------------------- b3 (#1)


def _b3_one(key: str, out_dir: str, sources: dict, factors, frames, opts: dict) -> dict:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.studies.t47b_replay import run_b3_record

    rows = run_b3_record(hs.record_path(key), target=hs.target_of(key), sources=sources,
                         factors=factors, frames=frames,
                         log=lambda s: print(f"[{key}]{s}", flush=True), **opts)
    for r in rows:
        r["record"] = key
    path = pathlib.Path(out_dir) / f"rows__{key}.jsonl"
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(bm.jsonable(r)) + "\n")
    return {"key": key, "rows": str(path), "n_rows": len(rows)}


def stage_b3(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t47b_b3 as b3

    t47 = t47_root()
    heads = load_head_sets(t47)
    keys, missing = hs.available_keys(hs.resolve_keys(args.records, hs.HELDOUT_KEYS))
    bases = list(args.sources or DEFAULT_B3_SOURCES)
    _refuse_oracle_on_selection(bases, keys)
    jobs = []
    for k in keys:
        att, dn, agl, _ = hs.load_attention(k)
        maps = build_sources(att, dn, agl, heads, bases, args.sigma)  # float64; scaled in replay
        frames = None if not args.max_frames else list(range(min(args.max_frames, att.shape[0])))
        jobs.append((k, maps, frames))
        del att
    out_dir = root / "b3"
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = _replay_opts(args)
    if args.workers > 1 and len(jobs) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                    mp_context=mp.get_context("spawn")) as ex:
            res = [f.result() for f in [ex.submit(_b3_one, k, str(out_dir), s, args.factors, fr,
                                                  opts) for k, s, fr in jobs]]
    else:
        res = [_b3_one(k, str(out_dir), s, args.factors, fr, opts) for k, s, fr in jobs]
    rows = []
    for r in res:
        with open(r["rows"]) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    payload = {**run_info("b3", args), "records": [r["key"] for r in res], "excluded": missing,
               "sources": source_names(bases, args.sigma), "factors": args.factors,
               "head_sets": {k: v for k, v in heads.items() if k != "files"},
               "summary": b3.summarize_b3(rows), "audit": b3.audit_table(),
               "definitions": {"decision_identical": "same cloud, seeds, peak, cluster partition, "
                                                     "split reasons, status, has_target, chosen",
                               "bit_identical": "decision_identical and raw == float32(k*raw), "
                                                "normalised values, scores, centroid bit-equal"}}
    write_json(out_dir / "summary.json", payload)
    print(f"[b3] wrote {out_dir / 'summary.json'}")
    return payload


# ------------------------------------------------------------------------------- parity


def compare_rows(t47_rows: Sequence[dict], t47b_rows: Sequence[dict], *,
                 centroid_tol: float = 1e-9) -> dict:
    """Per T47 source: frames where every T47 field agrees with the T47b row of the mapped source."""
    fields = ("has_target", "on_target", "silent_failure", "grounding_status", "n_clusters",
              "n_cloud", "n_cloud_target", "n_seeds", "n_seeds_target", "seed_camera")
    floats = ("centroid_error_m", "seed_precision", "seed_recall", "chosen_target_fraction")
    b_index = {(r["record"], r["frame"], r["source"]): r for r in t47b_rows}
    out: dict[str, Any] = {}
    for src47, base in PARITY_MAP.items():
        name = base if base == "l8h2" else f"{base}_s0"
        rs = [r for r in t47_rows if r["source"] == src47]
        matched, same, diffs = 0, 0, []
        for r in rs:
            b = b_index.get((r["record"], r["frame"], name))
            if b is None:
                continue
            matched += 1
            bad = [f for f in fields if r.get(f) != b.get(f)]
            for f in floats:
                x, y = r.get(f), b.get(f)
                xn = x is None or (isinstance(x, float) and not np.isfinite(x))
                yn = y is None or (isinstance(y, float) and not np.isfinite(y))
                if xn != yn or (not xn and abs(float(x) - float(y)) > centroid_tol):
                    bad.append(f)
            if bad:
                diffs.append({"record": r["record"], "frame": r["frame"], "fields": bad})
            else:
                same += 1
        out[src47] = {"t47b_source": name, "n_t47_rows": len(rs), "matched": matched,
                      "identical": same, "differing": diffs}
    return out


def stage_parity(args, root: pathlib.Path) -> dict:
    t47 = t47_root()
    t47_rows, b_rows = [], []
    for p in sorted((t47 / "e2e").glob("rows__*.jsonl")):
        with p.open() as fh:
            t47_rows += [json.loads(line) for line in fh if line.strip()]
    for p in sorted((root / "e2e").glob("rows__*__s0.jsonl")):
        with p.open() as fh:
            b_rows += [json.loads(line) for line in fh if line.strip()]
    if not b_rows:
        raise SystemExit("no T47b sigma-0 e2e rows — run --stage e2e --sigma 0 first")
    payload = {**run_info("parity", args), "comparison": compare_rows(t47_rows, b_rows)}
    write_json(root / "e2e" / "parity_vs_t47.json", payload)
    print(f"[parity] wrote {root / 'e2e' / 'parity_vs_t47.json'}")
    return payload


# ---------------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("check", "audit", "sink", "b2d", "e2e", "b3",
                                                       "parity", "summarize"))
    ap.add_argument("--records", nargs="+", default=None, metavar="KEY",
                    help=f"record keys {sorted(hs.RECORDS)}. Defaults: sink = all; b2d / e2e / b3 "
                         f"= held-out {list(hs.HELDOUT_KEYS)}")
    ap.add_argument("--workers", type=int, default=1,
                    help=f"records in parallel (spawn; <= {MAX_WORKERS}, sink <= {MAX_SINK_WORKERS})")
    ap.add_argument("--sources", nargs="+", default=None,
                    help=f"base names {list(SOURCE_BASES)}; default all (b3: "
                         f"{list(DEFAULT_B3_SOURCES)})")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="e2e / b3 / summarize: smoothing (grid cells). 1 = pre-registered; "
                         "0 reproduces T47's e2e maps (parity)")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 0.0], help="b2d")
    ap.add_argument("--factors", type=float, nargs="+", default=[3.0, 4.0],
                    help="b3: attention multipliers (3 = task; 4 = exact float control)")
    ap.add_argument("--max-frames", type=int, default=0, help="e2e / b3: first N frames (smoke)")
    ap.add_argument("--min-target-px", type=int, default=200)
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--links", default="gripper")
    ap.add_argument("--esdf-backend", choices=("legacy", "curobo"), default="legacy")
    ap.add_argument("--on-target-radius", type=float, default=0.10)
    ap.add_argument("--label-radius", type=float, default=0.003)
    ap.add_argument("--ball-voxel", type=float, default=0.005,
                    help="e2e: pad voxel of the post-hoc exclusion ball (live fine voxel 5 mm)")
    ap.add_argument("--membership-radius", type=float, default=0.005,
                    help="e2e: proximity radius when the chosen geometry has no valid indices")
    return ap


def validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    limit = MAX_SINK_WORKERS if args.stage == "sink" else MAX_WORKERS
    if not 1 <= args.workers <= limit:
        ap.error(f"--workers must be 1..{limit} for --stage {args.stage}")
    if args.sigma < 0 or any(s < 0 for s in args.sigmas):
        ap.error("sigma must be >= 0")
    if any(f <= 0 for f in args.factors):
        ap.error("--factors must be > 0")
    if args.max_frames < 0 or args.min_target_px < 0:
        ap.error("--max-frames / --min-target-px must be >= 0")
    if args.ball_voxel <= 0 or args.membership_radius <= 0:
        ap.error("--ball-voxel / --membership-radius must be > 0")
    if args.esdf_backend == "curobo":
        ap.error("--esdf-backend curobo needs the GPU; grounding does not depend on the field "
                 "backend — keep legacy (CPU), as T47")
    if args.sources:
        bad = [s for s in args.sources if s not in SOURCE_BASES]
        if bad:
            ap.error(f"unknown source(s) {bad}; have {list(SOURCE_BASES)}")
    if args.records:
        hs.resolve_keys(args.records, ())


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    validate_args(ap, args)
    root = out_root()
    t0 = time.time()
    {"check": stage_check, "audit": stage_audit, "sink": stage_sink, "b2d": stage_b2d,
     "e2e": stage_e2e, "b3": stage_b3, "parity": stage_parity,
     "summarize": stage_summarize}[args.stage](args, root)
    print(f"[{STEP}] stage {args.stage} done in {time.time() - t0:.1f}s -> {root}")


if __name__ == "__main__":
    main()
