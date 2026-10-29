"""T49 — noun-token query: target / destination maps from the prompt's noun rows of the VLA prefix.

    source scripts/setup/env.sh
    export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"   # once
    M=AG3S.ag3s.experiments.studies.t49_noun_query
    # CPU (openpi venv, no GPU, no params): noun positions of every frame x prompt variant
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= .venv-openpi-live/bin/python -m $M --stage tokens
    # GPU (verifier, under the lock): prefix-only noun rows, cached per frame, resumable
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m $M --stage extract
    # CPU (.venv-ag3s)
    MUJOCO_GL=osmesa .venv-ag3s/bin/python -m $M --stage select
    MUJOCO_GL=osmesa .venv-ag3s/bin/python -m $M --stage validate
    MUJOCO_GL=osmesa .venv-ag3s/bin/python -m $M --stage swap
    MUJOCO_GL=osmesa .venv-ag3s/bin/python -m $M --stage e2e        # MuJoCo replay (osmesa)

All output goes to ``$AG3S_RUN_DIR/verify/T49/`` (refused without ``AG3S_RUN_DIR``). T47's run is
read only (``--t47-dir``, default the 2026-10-12 T47 run): ``gt_cache/`` (class / body coverage)
and ``validate/oracle.json`` (T47 GT-oracle top-3, action query).

Stages:

``tokens``    CPU. The policy input transform built without the model (`t48_hsr_features.
              build_cpu_input_transform`) on every frame x prompt variant -> noun index, prefix row,
              pieces around it, errors. -> ``tokens/tokens.json``.
``extract``   GPU. `t49_noun_query_extract` -> ``extract/<variant>/<record>.npz``: maps
              (F, query, 18, 8, 3, 16, 16) float16 for queries (target, basket, last_text) + per-row
              image / text / BOS / self mass, token indices, ok / error per frame, prefix ms.
              Variants: ``orig`` (recorded prompt) and ``swap_<fruit>`` (target noun replaced by each
              other fruit). Resumable per frame. -> also ``extract/extract_meta.json`` (timing,
              memory, L8H2 prefix+suffix cost on the same frames).
``select``    CPU. Selection records (long, t2, ep1807): primary = GT lift ranking of all 18 x 8 cells
              (target map vs target coverage, basket map vs crate coverage), top-k; secondary = T47's
              label-free procedure on the same noun rows. -> ``select/``.
``validate``  CPU. Held-out 2D: class mass, lift, IoU, peak (vs target and vs crate) of noun primary /
              secondary / head-average, basket maps, L8H2 and T47 oracle top-3 (action); per-cell
              held-out matrices; #1 mass split; #5 separation per frame, record-level by phase.
``e2e``       CPU + MuJoCo. 3D grounding on the live serving path with T47's row fields + T47b B2
              (`t47b_replay.run_b2_record`), record-level summary (`t47b_metrics.summarize`).
``swap``      CPU. Prompt swap (exploratory): does the target map move to the newly named fruit?

Task: `AG3S/ag3s/docs/handoff/T49.task.md`. Pure pieces: `t49_noun_query_core`.
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
from AG3S.ag3s.experiments.studies import t49_noun_query_core as core

STEP = "T49"
CONFIG = "pi05_rby1_randomized_pick_place_16d_lora"
#: Storage-relative checkpoint (the attention npz were made from the same step, 29999).
CHECKPOINT = ("checkpoints/rby1/pi05_rby1_randomized_pick_place_16d_lora/"
              "rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999")
#: T47's run (2026-10-12), read only.
T47_RUN = "results/development/20261012T141412+0900_bd778e5e3453_d7a51b456f36"
STAGES = ("tokens", "extract", "select", "validate", "e2e", "swap")
MAX_WORKERS = 2
#: e2e source names -> what they are.
E2E_SOURCES = ("noun_gt", "noun_lf", "noun_headavg", "l8h2", "t47_oracle3")
E2E_DEFAULT = ("noun_gt", "l8h2")
#: Sources chosen on the selection records' GT — never scored there.
GT_CHOSEN = ("noun_gt", "t47_oracle3")


# ------------------------------------------------------------------------------ plumbing


def _run_dir() -> pathlib.Path:
    run_dir = os.environ.get("AG3S_RUN_DIR")
    if not run_dir:
        raise SystemExit(
            "AG3S_RUN_DIR is not set. Allocate a run first:\n"
            '  export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"')
    return pathlib.Path(run_dir)


def out_root() -> pathlib.Path:
    """``$AG3S_RUN_DIR/verify/T49`` (created)."""
    from common.run_output import resolve_output_path

    root = resolve_output_path("verify/T49", run_dir=_run_dir(), default_name="verify/T49")
    root.mkdir(parents=True, exist_ok=True)
    return root


def t47_dir(args) -> pathlib.Path:
    """T47's ``verify/T47`` (read only); refuses when its GT cache is missing."""
    p = pathlib.Path(args.t47_dir) if args.t47_dir else hs.storage_root() / T47_RUN / "verify" / "T47"
    if not (p / "gt_cache").is_dir():
        raise SystemExit(f"no T47 GT cache under {p} — pass --t47-dir <run>/verify/T47")
    return p


def checkpoint_path(args) -> pathlib.Path:
    p = pathlib.Path(args.checkpoint) if args.checkpoint else hs.storage_root() / CHECKPOINT
    if not p.is_dir():
        raise SystemExit(f"checkpoint {p} not found")
    return p


def write_json(path: pathlib.Path, payload: Any) -> None:
    hs.write_json(path, bm.jsonable(payload))


def run_info(stage: str, args: argparse.Namespace) -> dict:
    return {"step": STEP, "stage": stage, "argv": sys.argv,
            "args": {k: v for k, v in vars(args).items()},
            "started_kst": _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(),
            "git": hs.git_state(), "storage_root": str(hs.storage_root()),
            "run_dir": os.environ.get("AG3S_RUN_DIR"), "python": sys.version.split()[0]}


def extract_path(root: pathlib.Path, variant: str, key: str) -> pathlib.Path:
    return root / "extract" / variant / f"{key}.npz"


def frames_dir(root: pathlib.Path, variant: str, key: str) -> pathlib.Path:
    return root / "extract" / variant / f"{key}.frames"


def record_prompt(key: str) -> str:
    return str(json.loads((hs.record_path(key) / "meta.json").read_text())["prompt"])


def plan_for(keys: Sequence[str], swap_keys: Sequence[str], *, swaps: bool) -> list[dict]:
    """``[{key, variant, prompt, noun, swapped_from, target}]`` for every record x prompt variant."""
    out = []
    for k in keys:
        target = hs.target_of(k)
        prompt = record_prompt(k)
        for item in core.variant_plan(prompt, target, swaps=swaps and k in swap_keys):
            out.append({"key": k, "target": target, **item})
    return out


def phase_groups(key: str, t_steps: Sequence[int]) -> tuple[list[str], list[str], dict]:
    """``(phase, phase_group)`` per frame — T47b's definition (`t47b_metrics.phase_group`:
    post_grasp = `policy_record.phase_for == "grasp"`), boundaries from the whole record."""
    from AG3S.ag3s.experiments.sources.policy_record import phase_boundaries_for_path, phase_for

    bounds, evidence = phase_boundaries_for_path(hs.record_path(key))
    ph = [phase_for(int(t), bounds) for t in t_steps]
    return ph, [bm.phase_group(p) for p in ph], {"boundaries": list(bounds), "evidence": evidence}


def _keys(args, default: Sequence[str]) -> list[str]:
    keys, missing = hs.available_keys(hs.resolve_keys(args.records, default))
    if missing:
        print(f"[{STEP}] excluded (missing in storage): {[m['key'] for m in missing]}")
    if not keys:
        raise SystemExit("no record available")
    return keys


def load_orig(root: pathlib.Path, key: str, variant: str = "orig") -> dict:
    from AG3S.ag3s.experiments.studies.t49_noun_query_extract import load_extract

    p = extract_path(root, variant, key)
    if not p.exists():
        raise SystemExit(f"[{key}] no extract at {p} — run --stage extract first")
    return load_extract(p)


def check_alignment(key: str, ex: dict, gt: dict) -> None:
    if [int(t) for t in ex["t_step"]] != [int(t) for t in gt["t_step"]]:
        raise SystemExit(f"[{key}] extract t_step {list(ex['t_step'])[:4]}... != GT cache "
                         f"{list(gt['t_step'])[:4]}...")


def block_fc(ex: dict, query: str) -> np.ndarray:
    """``(F, C, L, H, 16, 16)`` float64 of one query (T47 `source_maps` axis order)."""
    from AG3S.ag3s.experiments.studies.t49_noun_query_extract import query_block

    return np.moveaxis(query_block(ex, query), 3, 1)


def keep_mask(gt: dict, ok: np.ndarray, cls: int, min_px: int) -> np.ndarray:
    """``(F, C)``: the class has >= `min_px` pixels in that camera and the frame extracted."""
    return (np.asarray(gt["class_px"])[:, :, cls] >= min_px) & np.asarray(ok, bool)[:, None]


# ------------------------------------------------------------------------------ tokens


def stage_tokens(args, root: pathlib.Path) -> dict:
    """CPU noun-position check of every frame x variant through the real input transform."""
    import types

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from AG3S.ag3s.experiments.sources.pi05_attention import observation_from_step
    from AG3S.ag3s.experiments.sources.policy_record import load_run
    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx
    from AG3S.ag3s.experiments.studies.t48_hsr_features import build_cpu_input_transform

    ckpt = checkpoint_path(args)
    fn, decode, info = build_cpu_input_transform(args.config, str(ckpt))
    shim = types.SimpleNamespace(_input_transform=fn)
    sp = nx.sentencepiece_of(args.config)
    ids = nx.noun_ids_for(sp, core.FRUITS)
    keys = _keys(args, list(hs.RECORDS))
    plan = plan_for(keys, args.swap_records or keys, swaps=not args.no_swaps)
    results = []
    for item in plan:
        record = load_run(hs.record_path(item["key"]))
        pos_seen: dict[str, set] = {q: set() for q in core.QUERY_NAMES}
        errors, n_valid, around, decoded = [], [], None, None
        for i, step in enumerate(record):
            obs = observation_from_step(shim, step, item["prompt"])
            toks = np.asarray(obs.tokenized_prompt)[0]
            mask = np.asarray(obs.tokenized_prompt_mask)[0].astype(bool)
            pos, err = nx.lookup(toks, mask, item["noun"], ids, int(sp.bos_id()))
            n_valid.append(int(mask.sum()))
            if err:
                errors.append({"frame": i, "t_step": int(step.t_step), "errors": err})
            for q, p in pos.items():
                pos_seen[q].add(int(p))
            if i == 0:
                decoded = str(sp.decode([int(t) for t in toks[mask]]))
                around = {q: nx.pieces_around(sp, toks, p) for q, p in pos.items()}
        results.append({
            "key": item["key"], "variant": item["variant"], "prompt": item["prompt"],
            "noun": item["noun"], "n_frames": len(record), "n_error_frames": len(errors),
            "errors": errors[:20],
            "text_index": {q: sorted(v) for q, v in pos_seen.items()},
            "prefix_row": {q: sorted(core.N_IMAGE_TOKENS + x for x in v)
                           for q, v in pos_seen.items()},
            "n_text_valid_min": min(n_valid), "n_text_valid_max": max(n_valid),
            "pieces_around_frame0": around, "decoded_frame0": decoded})
        print(f"[tokens] {item['variant']:>12s} {item['key']:>8s}: target idx "
              f"{sorted(pos_seen['target'])} basket {sorted(pos_seen['basket'])} "
              f"last {sorted(pos_seen['last_text'])} errors {len(errors)}/{len(record)}")
    payload = {**run_info("tokens", args), "transform": info, "piece_ids": ids,
               "bos_id": int(sp.bos_id()), "max_token_len": info.get("max_token_len"),
               "n_items": len(results), "n_error_frames": sum(r["n_error_frames"] for r in results),
               "items": results}
    write_json(root / "tokens" / "tokens.json", payload)
    print(f"[tokens] wrote {root / 'tokens' / 'tokens.json'}")
    return payload


# ------------------------------------------------------------------------------ extract


def stage_extract(args, root: pathlib.Path) -> dict:
    for k, v in (("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
                 ("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")):
        os.environ.setdefault(k, v)
    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx

    pids = nx.gpu_compute_pids()
    if pids is None:
        raise SystemExit("nvidia-smi is unavailable — the extract stage needs the GPU")
    if pids and not args.allow_busy_gpu:
        raise SystemExit(f"GPU has compute processes {pids} — wait for them (never kill another "
                         "session's process); run under flock /tmp/ag3s-vla-gpu-validation.lock")
    import jax

    if jax.default_backend() != "gpu":
        raise SystemExit(f"jax backend is {jax.default_backend()!r}, not gpu")

    from AG3S.ag3s.experiments.sources.pi05_attention import load_policy, observation_from_step
    from AG3S.ag3s.experiments.sources.policy_record import load_run, record_fingerprint

    ckpt = checkpoint_path(args)
    keys = _keys(args, list(hs.RECORDS))
    plan = plan_for(keys, args.swap_records or keys, swaps=not args.no_swaps)
    for k in keys:
        hs.check_pairing(k)
    t_load = time.perf_counter()
    policy = load_policy(args.config, str(ckpt))
    load_s = time.perf_counter() - t_load
    sp = nx.sentencepiece_of(args.config)
    ids = nx.noun_ids_for(sp, core.FRUITS)
    bos = int(sp.bos_id())
    ex = nx.NounQueryExtractor(policy._model)  # noqa: SLF001
    n_l = n_h = None
    compile_s = None
    results = []
    records: dict[str, Any] = {}
    for item in plan:
        key, variant = item["key"], item["variant"]
        if key not in records:
            records[key] = load_run(hs.record_path(key))
        record = records[key]
        stamp = nx.stamp_of(variant=variant, key=key, prompt=item["prompt"], noun=item["noun"],
                            checkpoint=str(ckpt), config=args.config,
                            fingerprint=record_fingerprint(record), n_frames=len(record))
        final = extract_path(root, variant, key)
        fdir = frames_dir(root, variant, key)
        if args.force:
            final.unlink(missing_ok=True)
            import shutil

            shutil.rmtree(fdir, ignore_errors=True)
        if final.exists():
            got = nx.load_extract(final)["stamp_dict"]
            if got != json.loads(json.dumps(stamp, sort_keys=True)):
                raise SystemExit(f"{final} was made for {got} — pass --force to redo")
            print(f"[extract] {variant}/{key} cached ({final})")
            results.append({"variant": variant, "record": key, "cached": True, "path": str(final)})
            continue

        def run_frame(step, item=item):
            nonlocal n_l, n_h, compile_s
            obs = observation_from_step(policy, step, item["prompt"])
            toks = np.asarray(obs.tokenized_prompt)[0]
            mask = np.asarray(obs.tokenized_prompt_mask)[0].astype(bool)
            pos, err = nx.lookup(toks, mask, item["noun"], ids, bos)
            if err:
                if n_l is None:  # learn (L, H) (and compile) from a dummy pass on BOS rows
                    t0 = time.perf_counter()
                    dummy = ex.rows(obs, np.full(len(core.QUERY_NAMES), core.N_IMAGE_TOKENS,
                                                 np.int32), timed=False)
                    compile_s = time.perf_counter() - t0
                    n_l, n_h = dummy["image"].shape[:2]
                return {"ok": False, "error": "; ".join(err), "fields": nx.empty_fields(n_l, n_h),
                        "prefix_ms": float("nan"), "n_text_valid": int(mask.sum())}
            rows = core.prefix_rows(pos)
            if compile_s is None:  # first call compiles; time it apart
                t0 = time.perf_counter()
                out = ex.rows(obs, rows, timed=False)
                compile_s = time.perf_counter() - t0
            out = ex.rows(obs, rows)
            n_l, n_h = out["image"].shape[:2]
            return {"ok": True, "error": "", "fields": nx.fields_from_rows(out, pos, toks),
                    "prefix_ms": ex.t_call[-1] * 1e3, "n_text_valid": int(mask.sum())}

        res = nx.extract_record(frames_dir=fdir, final_path=final, stamp=stamp,
                                steps=record.steps, run_frame=run_frame,
                                keep_frames=args.keep_frames)
        print(f"[extract] {variant}/{key}: {res['n_frames']} frames ({res['computed']} computed, "
              f"{res['resumed']} resumed), errors {res['n_errors']}")
        results.append(res)
    cost = _cost(args, ex, policy, records, plan, observation_from_step, ids, bos, nx)
    payload = {**run_info("extract", args), "checkpoint": str(ckpt), "config": args.config,
               "gpu_compute_pids_before": pids, "jax_backend": jax.default_backend(),
               "devices": [str(d) for d in jax.devices()],
               "piece_ids": ids, "bos_id": bos, "queries": list(core.QUERY_NAMES),
               "load_s": load_s, "compile_s": compile_s,
               "prefix_rows_timing": nx.timing_summary(
                   ex.t_call, "noun prefix pass (embed_prefix + 18 layers + in-jit slice) per frame, "
                              "block_until_ready; excludes input transform, compile, npz write"),
               "memory": ex.memory(), "cost": cost, "results": results}
    write_json(root / "extract" / "extract_meta.json", payload)
    print(f"[extract] wrote {root / 'extract' / 'extract_meta.json'}")
    return payload


def _cost(args, ex, policy, records, plan, observation_from_step, ids, bos, nx) -> dict:
    """#8 — the same frames through the noun prefix pass and through L8H2's serving sampler
    (`AttentionSampler.attention(obs, [0])` = prefix + `num_steps` suffix passes, as
    `attention_policy` calls it)."""
    if not args.cost_frames:
        return {"skipped": True}
    from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler

    item = next(p for p in plan if p["variant"] == "orig")
    steps = records[item["key"]].steps[: args.cost_frames]
    sampler = AttentionSampler(policy._model)  # noqa: SLF001 — attention_policy default num_steps 10
    noun_t, l8_t = [], []
    for j, step in enumerate(steps):
        obs = observation_from_step(policy, step, item["prompt"])
        toks = np.asarray(obs.tokenized_prompt)[0]
        mask = np.asarray(obs.tokenized_prompt_mask)[0].astype(bool)
        pos, err = nx.lookup(toks, mask, item["noun"], ids, bos)
        if err:
            continue
        rows = core.prefix_rows(pos)
        t0 = time.perf_counter()
        ex.rows(obs, rows, timed=False)
        t1 = time.perf_counter()
        sampler.attention(obs, [0], noise_seed=0)  # returns host numpy (blocks)
        t2 = time.perf_counter()
        if j >= 2:  # the first two calls compile / warm the sampler's executables
            noun_t.append(t1 - t0)
            l8_t.append(t2 - t1)
    return {"record": item["key"], "frames": len(steps), "warmup_dropped": 2,
            "noun_prefix": nx.timing_summary(noun_t, "noun prefix pass incl. preprocess"),
            "l8h2_prefix_plus_suffix": nx.timing_summary(
                l8_t, "AttentionSampler.attention(obs, [0]) incl. preprocess: prefix + 10 suffix "
                      "passes (the serving attention path)"),
            "memory_after": ex.memory()}


# ------------------------------------------------------------------------------ select


def _lf_select(blocks_by_key: dict[str, np.ndarray], keys: Sequence[str], args) -> dict:
    """T47's label-free procedure on noun rows (`t47_head_selection.select_from_stats`)."""
    stats = {}
    for k in keys:
        b = blocks_by_key[k]  # (F_ok, L, H, C, 16, 16)
        s_cam, ent, _ = hs.record_statistics(b[:, None, None])
        stats[k] = {"S_cam": s_cam, "entropy": ent}
    n_l, n_h = next(iter(blocks_by_key.values())).shape[1:3]
    cells = hm.cell_list(n_l, n_h)
    out = {}
    for src in ("cam", "img"):
        res = hs.select_from_stats(stats, keys, 0, 0, cells, tau_source=src,
                                   tau_method=args.tau_method, n_lowest=args.n_lowest,
                                   kmax=args.kmax)
        loro = []
        if len(keys) > 1:
            full = set(map(tuple, res["common"]["topk"]["3"]))
            for left in keys:
                r2 = hs.select_from_stats(stats, [k for k in keys if k != left], 0, 0, cells,
                                          tau_source=src, tau_method=args.tau_method,
                                          n_lowest=args.n_lowest, kmax=args.kmax)
                got = set(map(tuple, r2["common"]["topk"]["3"]))
                loro.append({"left_out": left, "top3": sorted(got),
                             "overlap_with_full": len(full & got)})
        out[src] = {"common": hs._jsonable_block(res["common"]),  # noqa: SLF001
                    "per_camera_top3": {c: res["per_camera"][c]["topk"]["3"]
                                        for c in hs.POLICY_CAMERAS},
                    "tau_summary": res["tau_summary"], "leave_one_record_out": loro,
                    "_frequency": res["common"]["frequency"], "_cells": cells}
    return out


def stage_select(args, root: pathlib.Path) -> dict:
    t47 = t47_dir(args)
    keys = _keys(args, hs.SELECTION_KEYS)
    data = {}
    for k in keys:
        ex, gt = load_orig(root, k), hs.load_gt(t47, k)
        check_alignment(k, ex, gt)
        data[k] = (ex, gt)
    payload: dict[str, Any] = {**run_info("select", args), "records": keys, "t47_dir": str(t47),
                               "min_target_px": args.min_target_px, "queries": {},
                               "policy": {"primary": "GT: mean lift over kept (frame, camera) of the "
                                                     "selection records, all 18x8 cells; tie-break "
                                                     "peak_on_target, index",
                                          "secondary": "T47 label-free (S_cam kneedle tau, spatial "
                                                       "entropy, frequency, L0/L1 excluded)",
                                          "uses_gt_primary": True, "uses_gt_secondary": False}}
    arrays: dict[str, np.ndarray] = {}
    # query -> GT class it is ranked against
    targets = (("target", core.TARGET_CLASS), ("basket", core.DESTINATION_CLASS),
               ("last_text", core.TARGET_CLASS), ("last_text", core.DESTINATION_CLASS))
    for query, cls in targets:
        name = f"{query}__vs_{hm.CLASS_NAMES[cls]}"
        blocks, covs, keeps, per_rec = [], [], [], {}
        for k, (ex, gt) in data.items():
            b = block_fc(ex, query)
            kp = keep_mask(gt, ex["ok"], cls, args.min_target_px)
            blocks.append(b), covs.append(np.asarray(gt["class_cov"])), keeps.append(kp)
            m1 = core.cell_metrics([b], [gt["class_cov"]], [kp], target_index=cls)
            per_rec[k] = [list(c) for c in core.rank_by_lift(m1["lift"], m1["peak_on_target"])[:3]]
        met = core.cell_metrics(blocks, covs, keeps, target_index=cls)
        order = core.rank_by_lift(met["lift"], met["peak_on_target"])
        loro = []
        for i, left in enumerate(keys):
            if len(keys) < 2:
                break
            idx = [j for j in range(len(keys)) if j != i]
            m2 = core.cell_metrics([blocks[j] for j in idx], [covs[j] for j in idx],
                                   [keeps[j] for j in idx], target_index=cls)
            top = core.rank_by_lift(m2["lift"], m2["peak_on_target"])[:3]
            loro.append({"left_out": left, "top3": [list(c) for c in top],
                         "overlap_with_full": len(set(top) & set(order[:3]))})
        entry = {"gt_class": hm.CLASS_NAMES[cls], "n_samples": met["n"],
                 "gt_topk": {str(k): [list(c) for c in order[:k]] for k in range(1, args.kmax + 1)},
                 "gt_ranking": core.ranking_table(order, met),
                 "gt_per_record_top3": per_rec, "gt_leave_one_record_out": loro}
        for f in ("lift", "peak_on_target", "target_mass"):
            arrays[f"{name}__{f}"] = met[f]
        payload["queries"][name] = entry
    for query in ("target", "basket"):
        ok_blocks = {k: np.asarray(ex["maps"][np.asarray(ex["ok"], bool),
                                              [str(q) for q in ex["query_names"]].index(query)],
                                   np.float64)
                     for k, (ex, _) in data.items()}
        lf = _lf_select(ok_blocks, keys, args)
        for src, res in lf.items():
            arrays[f"lf__{query}__{src}__frequency"] = res.pop("_frequency")
            arrays[f"lf__cells"] = np.asarray(res.pop("_cells"), np.int64)
        cls = core.TARGET_CLASS if query == "target" else core.DESTINATION_CLASS
        payload["queries"][f"{query}__vs_{hm.CLASS_NAMES[cls]}"]["label_free"] = lf
    payload["extract_errors"] = {k: int((~np.asarray(ex["ok"], bool)).sum())
                                 for k, (ex, _) in data.items()}
    out = root / "select"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "select.json", payload)
    np.savez_compressed(out / "select_arrays.npz", **arrays)
    for q in ("target__vs_target", "basket__vs_destination"):
        print(f"[select] {q}: GT top3 {payload['queries'][q]['gt_topk']['3']} | label-free top3 "
              f"{payload['queries'][q]['label_free']['cam']['common']['topk']['3']}")
    return payload


def load_select(root: pathlib.Path) -> dict:
    p = root / "select" / "select.json"
    if not p.exists():
        raise SystemExit(f"no {p} — run --stage select first")
    return json.loads(p.read_text())


def cellsets(sel: dict, k: int) -> dict[str, list]:
    """The head sets the CLI uses, from `select.json`."""
    t = sel["queries"]["target__vs_target"]
    b = sel["queries"]["basket__vs_destination"]
    return {"noun_gt": t["gt_topk"][str(k)], "noun_lf": t["label_free"]["cam"]["common"]["topk"][str(k)],
            "basket_gt": b["gt_topk"][str(k)],
            "basket_lf": b["label_free"]["cam"]["common"]["topk"][str(k)]}


def t47_oracle_cells(t47: pathlib.Path) -> list:
    p = t47 / "validate" / "oracle.json"
    if not p.exists():
        raise SystemExit(f"no {p} (T47 GT-oracle)")
    return json.loads(p.read_text())["settings"]["d0_last"]["top3"]


def action_maps(key: str, oracle_cells, sigmas: Sequence[float]) -> dict[str, np.ndarray]:
    """``l8h2`` (serving cell, d0 · last) and ``t47_oracle3_s<σ>`` from T47's action npz."""
    from AG3S.ag3s.experiments.studies.t47b_study import setting_block

    att, dn, agl, _ = hs.load_attention(key)
    sd, sa = dn.index(hs.SERVING_CELL["denoise"]), agl.index(hs.SERVING_CELL["agg"])
    out = {"l8h2": np.asarray(att[:, sd, sa, hs.SERVING_CELL["layer"], hs.SERVING_CELL["head"]],
                              np.float64)}
    block = np.asarray(setting_block(att, dn, agl), np.float64)
    for s in sigmas:
        out[f"t47_oracle3_s{s:g}"] = core.combined(block, oracle_cells, s)
    return out


def noun_sources(ex: dict, sel: dict, *, kmax: int, sigmas: Sequence[float]) -> dict[str, np.ndarray]:
    """Named ``(F, C, 16, 16)`` maps from the noun rows of one record."""
    t, b, last = block_fc(ex, "target"), block_fc(ex, "basket"), block_fc(ex, "last_text")
    out: dict[str, np.ndarray] = {"noun_headavg": hm.head_average(t),
                                  "basket_headavg": hm.head_average(b),
                                  "lasttext_headavg": hm.head_average(last)}
    kmax = min(int(kmax), len(sel["queries"]["target__vs_target"]["gt_topk"]))
    for k in range(1, kmax + 1):
        cs = cellsets(sel, k)
        for s in sigmas:
            tag = f"k{k}_s{s:g}"
            out[f"noun_gt_{tag}"] = core.combined(t, cs["noun_gt"], s)
            out[f"noun_lf_{tag}"] = core.combined(t, cs["noun_lf"], s)
            out[f"basket_gt_{tag}"] = core.combined(b, cs["basket_gt"], s)
            out[f"basket_lf_{tag}"] = core.combined(b, cs["basket_lf"], s)
            out[f"basket_samecells_{tag}"] = core.combined(b, cs["noun_gt"], s)
    return out


# ----------------------------------------------------------------------------- validate


def _scene_means(agg: dict, rec_names: Sequence[str]) -> dict:
    groups = {}
    for grp in ("same_scene", "new_scene", "selection"):
        recs = [r for r in rec_names if bm.scene_group(r) == grp]
        if not recs:
            continue
        groups[grp] = {m: float(np.nanmean([agg["per_record"][r][m] for r in recs]))
                       for m in ("target_mass", "lift", "peak_on_target", "iou@0.5", "iou@0.25")}
        groups[grp]["records"] = recs
        groups[grp]["class_mass"] = {c: float(np.nanmean([agg["per_record"][r]["class_mass"][c]
                                                          for r in recs]))
                                     for c in hm.CLASS_NAMES}
    return groups


def _mass(root: pathlib.Path, keys: Sequence[str], args) -> dict:
    out = {}
    exs = {k: load_orig(root, k) for k in keys if extract_path(root, "orig", k).exists()}
    for qi, q in enumerate(core.QUERY_NAMES):
        cols = {f: [] for f in ("image_mass", "text_mass", "bos", "self", "cam")}
        for ex in exs.values():
            ok = np.asarray(ex["ok"], bool)
            for f in ("image_mass", "text_mass", "bos", "self"):
                cols[f].append(np.asarray(ex[f][ok, qi], np.float64))
            cols["cam"].append(np.asarray(ex["maps"][ok, qi], np.float64).sum(axis=(-1, -2)))
        cat = {f: np.concatenate(v) for f, v in cols.items()}
        out[q] = core.mass_summary(cat["image_mass"], cat["text_mass"], cat["bos"], cat["self"],
                                   cat["cam"], min_image_mass=args.min_image_mass)
    return {"records": list(exs), "per_query": out}


SEPARATION_PAIRS = {
    "primary": ("noun_gt_k3_s1", "basket_gt_k3_s1"),
    "secondary": ("noun_lf_k3_s1", "basket_lf_k3_s1"),
    "samecells": ("noun_gt_k3_s1", "basket_samecells_k3_s1"),
    "headavg": ("noun_headavg", "basket_headavg"),
}


def stage_validate(args, root: pathlib.Path) -> dict:
    t47 = t47_dir(args)
    sel = load_select(root)
    ocells = t47_oracle_cells(t47)
    keys = _keys(args, hs.HELDOUT_KEYS)
    sigmas = sorted(set([1.0, 0.0] + list(args.sigmas)))
    out = root / "validate"
    out.mkdir(parents=True, exist_ok=True)
    scores: dict[str, dict[str, list]] = {}
    keeps: dict[str, list] = {"target": [], "destination": []}
    cam_l, rec_l, frame_l = [], [], []
    cell_inputs: dict[str, tuple[list, list, list]] = {}
    sep_rows: list[dict] = []
    phases = {}
    for ri, k in enumerate(keys):
        ex, gt = load_orig(root, k), hs.load_gt(t47, k)
        check_alignment(k, ex, gt)
        ok = np.asarray(ex["ok"], bool)
        maps = {**noun_sources(ex, sel, kmax=args.kmax, sigmas=sigmas),
                **action_maps(k, ocells, sigmas)}
        f = ok.size
        kt = keep_mask(gt, ok, core.TARGET_CLASS, args.min_target_px)
        kd = keep_mask(gt, ok, core.DESTINATION_CLASS, args.min_target_px)
        keeps["target"].append(kt.reshape(-1)), keeps["destination"].append(kd.reshape(-1))
        cam_l.append(np.tile(np.arange(3), f)), rec_l.append(np.full(f * 3, ri))
        frame_l.append(np.repeat(np.arange(f), 3))
        cov = np.asarray(gt["class_cov"], np.float64)
        for n, m in maps.items():
            for vs, cls in (("target", core.TARGET_CLASS), ("destination", core.DESTINATION_CLASS)):
                scores.setdefault(n, {}).setdefault(vs, []).append(
                    core.flat_scores(m, cov, target_index=cls))
        np.savez_compressed(out / f"maps__{k}.npz", **{n: m.astype(np.float32)
                                                      for n, m in maps.items()})
        for q, cls in (("target", core.TARGET_CLASS), ("basket", core.DESTINATION_CLASS),
                       ("last_text", core.TARGET_CLASS), ("last_text", core.DESTINATION_CLASS)):
            name = f"{q}__vs_{hm.CLASS_NAMES[cls]}"
            b, c, kk = cell_inputs.setdefault(name, ([], [], []))
            b.append(block_fc(ex, q)), c.append(cov)
            kk.append(kt if cls == core.TARGET_CLASS else kd)
        ph, pg, ev = phase_groups(k, ex["t_step"])
        phases[k] = ev
        for pair, (tn, bn) in SEPARATION_PAIRS.items():
            for i in np.flatnonzero(ok):
                row = core.separation_frame(maps[tn][i], maps[bn][i], cov[i],
                                            np.asarray(gt["class_px"])[i],
                                            min_px=args.min_target_px)
                sep_rows.append({"source": pair, "target_map": tn, "basket_map": bn,
                                 "record": k, "scene_group": bm.scene_group(k), "frame": int(i),
                                 "t_step": int(ex["t_step"][i]), "phase": ph[i],
                                 "phase_group": pg[i], **row})
    cam_of, rec_of = np.concatenate(cam_l), np.concatenate(rec_l)
    frame_of = np.concatenate(frame_l)
    kt_all, kd_all = np.concatenate(keeps["target"]), np.concatenate(keeps["destination"])
    sources: dict[str, Any] = {}
    per_sample = {"camera": cam_of, "record": rec_of, "frame": frame_of, "keep_target": kt_all,
                  "keep_destination": kd_all, "record_names": np.asarray(keys),
                  "class_names": np.asarray(hm.CLASS_NAMES)}
    for n, by in scores.items():
        sources[n] = {}
        for vs, lst in by.items():
            merged = {fld: np.concatenate([s[fld] for s in lst]) for fld in lst[0]}
            keep = kt_all if vs == "target" else kd_all
            agg = hs._aggregate(merged, keep, cam_of, rec_of, keys)  # noqa: SLF001
            agg["record_mean_by_scene"] = _scene_means(agg, keys)
            sources[n][f"vs_{vs}"] = agg
            for fld in ("lift", "target_mass", "peak_on_target", "iou@0.25", "iou@0.5"):
                per_sample[f"{n}__vs_{vs}__{fld}"] = merged[fld]
            per_sample[f"{n}__vs_{vs}__class_mass"] = merged["class_mass"]
    cell_arrays = {}
    for name, (b, c, kk) in cell_inputs.items():
        met = core.cell_metrics(b, c, kk, target_index=(core.TARGET_CLASS if name.endswith("target")
                                                         else core.DESTINATION_CLASS))
        for fld in ("lift", "peak_on_target", "target_mass"):
            cell_arrays[f"{name}__{fld}"] = met[fld]
    np.savez_compressed(out / "per_sample.npz", **per_sample)
    np.savez_compressed(out / "per_cell_heldout.npz", **cell_arrays)
    with (out / "separation_rows.jsonl").open("w") as fh:
        for r in sep_rows:
            fh.write(json.dumps(bm.jsonable(r)) + "\n")
    payload = {
        **run_info("validate", args), "records": keys, "t47_dir": str(t47),
        "min_target_px": args.min_target_px, "sigmas": sigmas, "kmax": args.kmax,
        "cells": {**{n: c for n, c in cellsets(sel, 3).items()}, "t47_oracle3_action": ocells,
                  "l8h2_action": [[hs.SERVING_CELL["layer"], hs.SERVING_CELL["head"]]]},
        "roles": {"primary_target": "noun_gt_k3_s1", "secondary_target": "noun_lf_k3_s1",
                  "primary_destination": "basket_gt_k3_s1",
                  "secondary_destination": "basket_lf_k3_s1",
                  "variant_destination_same_cells": "basket_samecells_k3_s1",
                  "baselines": ["l8h2", "t47_oracle3_s1", "noun_headavg"]},
        "n_samples": {"total": int(kt_all.size), "kept_target": int(kt_all.sum()),
                      "kept_destination": int(kd_all.sum())},
        "sources": sources,
        "mass": _mass(root, args.mass_records or list(hs.RECORDS), args),
        "separation": {"pairs": SEPARATION_PAIRS,
                       "definition": core.separation_frame.__doc__,
                       "summary": core.summarize_records(sep_rows, core.SEPARATION_RATES,
                                                         core.SEPARATION_MEANS)},
        "phase_boundaries": phases,
    }
    write_json(out / "validate.json", payload)
    print(f"[validate] {len(sources)} sources x 2 GT classes over {keys}; wrote {out}")
    return payload


# ---------------------------------------------------------------------------------- e2e


def e2e_maps(root: pathlib.Path, key: str, sel: dict, ocells, names: Sequence[str], *, k: int,
             sigma: float) -> tuple[dict[str, np.ndarray], np.ndarray]:
    ex = load_orig(root, key)
    t = block_fc(ex, "target")
    cs = cellsets(sel, k)
    out: dict[str, np.ndarray] = {}
    for n in names:
        if n == "noun_gt":
            out[n] = core.combined(t, cs["noun_gt"], sigma)
        elif n == "noun_lf":
            out[n] = core.combined(t, cs["noun_lf"], sigma)
        elif n == "noun_headavg":
            out[n] = hm.head_average(t)
        elif n == "l8h2":
            out[n] = action_maps(key, ocells, [])["l8h2"]
        elif n == "t47_oracle3":
            out[n] = action_maps(key, ocells, [sigma])[f"t47_oracle3_s{sigma:g}"]
        else:  # pragma: no cover — validate_args guards this
            raise SystemExit(f"unknown e2e source {n}")
    return out, np.asarray(ex["ok"], bool)


def _e2e_one(key: str, out_dir: str, sources: dict, frames, opts: dict) -> dict:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.studies.t47b_replay import run_b2_record

    rows = run_b2_record(hs.record_path(key), target=hs.target_of(key), sources=sources,
                         frames=frames, log=lambda s: print(f"[{key}]{s}", flush=True), **opts)
    for r in rows:
        r["record"] = key
        r["scene_group"] = bm.scene_group(key)
    path = pathlib.Path(out_dir) / f"rows__{key}.jsonl"
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(bm.jsonable(r)) + "\n")
    return {"key": key, "rows": str(path), "n_rows": len(rows)}


def stage_e2e(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies.t47_replay import summarize_e2e

    t47 = t47_dir(args)
    sel = load_select(root)
    ocells = t47_oracle_cells(t47)
    keys = _keys(args, hs.HELDOUT_KEYS)
    names = list(args.sources or E2E_DEFAULT)
    if any(n in GT_CHOSEN for n in names) and any(hs.RECORDS[k].role == "selection" for k in keys):
        raise SystemExit(f"{[n for n in names if n in GT_CHOSEN]} are chosen on the selection "
                         "records' GT — not scored there; drop them or pick held-out records")
    pairing = [hs.check_pairing(k) for k in keys]
    jobs, skipped = [], {}
    for k in keys:
        maps, ok = e2e_maps(root, k, sel, ocells, names, k=args.k, sigma=args.sigma)
        frames = [int(i) for i in np.flatnonzero(ok)]
        if args.max_frames:
            frames = frames[: args.max_frames]
        skipped[k] = [int(i) for i in np.flatnonzero(~ok)]
        jobs.append((k, {n: m.astype(np.float32) for n, m in maps.items()},
                     None if (len(frames) == ok.size and not args.max_frames) else frames))
    opts = {"voxel": args.voxel, "range_max": args.range_max, "links": args.links,
            "esdf_backend": "legacy", "on_target_radius": args.on_target_radius,
            "label_radius": args.label_radius, "ball_voxel": args.ball_voxel,
            "membership_radius": args.membership_radius}
    out_dir = root / "e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.workers > 1 and len(jobs) > 1:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)),
                                    mp_context=mp.get_context("spawn")) as ex:
            res = [f.result() for f in [ex.submit(_e2e_one, k, str(out_dir), s, fr, opts)
                                        for k, s, fr in jobs]]
    else:
        res = [_e2e_one(k, str(out_dir), s, fr, opts) for k, s, fr in jobs]
    rows = []
    for r in res:
        with open(r["rows"]) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    payload = {**run_info("e2e", args), "records": pairing, "sources": names,
               "cells": {"noun_gt": cellsets(sel, args.k)["noun_gt"],
                         "noun_lf": cellsets(sel, args.k)["noun_lf"],
                         "t47_oracle3": ocells,
                         "l8h2": [[hs.SERVING_CELL["layer"], hs.SERVING_CELL["head"]]]},
               "k": args.k, "sigma": args.sigma, "options": opts,
               "frames_skipped_extract_error": skipped, "n_rows": len(rows),
               "aggregation": "per record (frames pooled), then unweighted mean over records; "
                              "strata scene x phase (t47b_metrics.summarize)",
               "summary": bm.summarize(rows),
               "t47_style_per_record": {r["key"]: summarize_e2e(
                   [x for x in rows if x["record"] == r["key"]]) for r in res},
               "t47_style_pooled": summarize_e2e(rows)}
    write_json(out_dir / "summary.json", payload)
    print(f"[e2e] wrote {out_dir / 'summary.json'} ({len(rows)} rows)")
    return payload


# --------------------------------------------------------------------------------- swap


def basket_swap_rows(orig_b, swap_b, gt, i, crate_idx, min_px) -> dict:
    cov = np.asarray(gt["body_cov"][i], np.float64)
    px = np.asarray(gt["body_px"][i])
    lo = core._camera_lifts(orig_b, cov[:, crate_idx], px[:, crate_idx], min_px)  # noqa: SLF001
    ls = core._camera_lifts(swap_b, cov[:, crate_idx], px[:, crate_idx], min_px)  # noqa: SLF001
    return {"map_corr": core.map_corr(orig_b, swap_b),
            "orig_crate_lift": core._nanmean(lo), "swap_crate_lift": core._nanmean(ls)}  # noqa: SLF001


def stage_swap(args, root: pathlib.Path) -> dict:
    t47 = t47_dir(args)
    sel = load_select(root)
    keys = _keys(args, list(hs.RECORDS))
    cs = cellsets(sel, args.k)
    rows, brows, done = [], [], []
    for k in keys:
        if not extract_path(root, "orig", k).exists():
            print(f"[swap] {k}: no orig extract, skipped")
            continue
        ex0, gt = load_orig(root, k), hs.load_gt(t47, k)
        check_alignment(k, ex0, gt)
        target = hs.target_of(k)
        named = [str(n) for n in gt["named_bodies"]]
        ph, pg, _ = phase_groups(k, ex0["t_step"])
        t0, b0 = block_fc(ex0, "target"), block_fc(ex0, "basket")
        for fruit in core.swap_fruits(target):
            p = extract_path(root, core.variant_name(fruit), k)
            if not p.exists():
                print(f"[swap] {k}: no {p.parent.name} extract, skipped")
                continue
            ex1 = load_orig(root, k, core.variant_name(fruit))
            if [int(t) for t in ex1["t_step"]] != [int(t) for t in ex0["t_step"]]:
                raise SystemExit(f"[{k}] swap extract frames differ from orig")
            done.append({"record": k, "fruit": fruit, "prompt": str(ex1["prompt"])})
            t1, b1 = block_fc(ex1, "target"), block_fc(ex1, "basket")
            ok = np.asarray(ex0["ok"], bool) & np.asarray(ex1["ok"], bool)
            sets = {"noun_gt": (core.combined(t0, cs["noun_gt"], args.sigma),
                                core.combined(t1, cs["noun_gt"], args.sigma)),
                    "noun_lf": (core.combined(t0, cs["noun_lf"], args.sigma),
                                core.combined(t1, cs["noun_lf"], args.sigma)),
                    "noun_headavg": (hm.head_average(t0), hm.head_average(t1))}
            bmaps = (core.combined(b0, cs["basket_gt"], args.sigma),
                     core.combined(b1, cs["basket_gt"], args.sigma))
            base = {"record": k, "role": hs.RECORDS[k].role, "scene_group": bm.scene_group(k),
                    "swap_fruit": fruit, "orig_target": target}
            for i in np.flatnonzero(ok):
                meta = {**base, "frame": int(i), "t_step": int(ex0["t_step"][i]), "phase": ph[i],
                        "phase_group": pg[i]}
                for src, (mo, ms) in sets.items():
                    r = core.swap_frame(mo[i], ms[i], gt["body_cov"][i], gt["body_px"][i],
                                        new_index=named.index(fruit),
                                        orig_index=named.index(target),
                                        min_px=args.min_target_px)
                    rows.append({**meta, "source": src, **r})
                brows.append({**meta, "source": "basket_gt",
                              **basket_swap_rows(bmaps[0][i], bmaps[1][i], gt, i,
                                                 named.index(core.DESTINATION_BODY),
                                                 args.min_target_px)})
    if not rows:
        raise SystemExit("no swap extract found — run --stage extract (without --no-swaps)")
    out = root / "swap"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "swap_rows.jsonl").open("w") as fh:
        for r in rows + brows:
            fh.write(json.dumps(bm.jsonable(r)) + "\n")
    payload = {**run_info("swap", args), "t47_dir": str(t47), "variants": done,
               "cells": {"noun_gt": cs["noun_gt"], "noun_lf": cs["noun_lf"],
                         "basket_gt": cs["basket_gt"]},
               "k": args.k, "sigma": args.sigma, "min_target_px": args.min_target_px,
               "definition": core.swap_frame.__doc__,
               "summary": core.summarize_records(rows, core.SWAP_RATES, core.SWAP_MEANS,
                                                 extra_strata=("role", "swap_fruit")),
               "basket_stability": core.summarize_records(
                   brows, {}, {"map_corr": None, "orig_crate_lift": None, "swap_crate_lift": None},
                   extra_strata=("role", "swap_fruit"))}
    write_json(out / "swap.json", payload)
    print(f"[swap] {len(done)} (record, fruit) variants, {len(rows)} rows -> {out}")
    return payload


# ---------------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--records", nargs="+", default=None, metavar="KEY",
                    help=f"record keys {sorted(hs.RECORDS)}. Defaults: tokens / extract / swap = all; "
                         f"select = {list(hs.SELECTION_KEYS)}; validate / e2e = "
                         f"{list(hs.HELDOUT_KEYS)}")
    ap.add_argument("--t47-dir", default=None,
                    help=f"T47's verify/T47 (read only). Default $AG3S_STORAGE_ROOT/{T47_RUN}/verify/T47")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--checkpoint", default=None,
                    help=f"default $AG3S_STORAGE_ROOT/{CHECKPOINT}")
    # tokens / extract
    ap.add_argument("--no-swaps", action="store_true", help="tokens / extract: recorded prompt only")
    ap.add_argument("--swap-records", nargs="+", default=None, metavar="KEY",
                    help="tokens / extract: records that also get the prompt-swap variants "
                         "(default: every record processed)")
    ap.add_argument("--force", action="store_true", help="extract: redo cached records")
    ap.add_argument("--keep-frames", action="store_true",
                    help="extract: keep the per-frame cache after the record npz is written")
    ap.add_argument("--cost-frames", type=int, default=10,
                    help="extract: frames timed through both the noun pass and L8H2's sampler (#8); "
                         "0 = skip")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="extract: run even if nvidia-smi lists compute processes (never kills)")
    # select / validate
    ap.add_argument("--min-target-px", type=int, default=200)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--tau-method", choices=("kneedle", "curvature"), default="kneedle")
    ap.add_argument("--n-lowest", type=int, default=10)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 0.0])
    ap.add_argument("--min-image-mass", type=float, default=0.01,
                    help="validate #1: layers whose median image mass is below this are listed "
                         "(reading aid)")
    ap.add_argument("--mass-records", nargs="+", default=None, metavar="KEY",
                    help="validate #1: records pooled for the mass split (default all extracted)")
    # e2e / swap
    ap.add_argument("--sources", nargs="+", default=None, help=f"e2e: {list(E2E_SOURCES)}")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=1.0, help="e2e / swap: smoothing (pre-registered 1)")
    ap.add_argument("--workers", type=int, default=1, help=f"e2e: records in parallel (<= {MAX_WORKERS})")
    ap.add_argument("--max-frames", type=int, default=0, help="e2e: first N frames (smoke)")
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--links", default="gripper")
    ap.add_argument("--on-target-radius", type=float, default=0.10)
    ap.add_argument("--label-radius", type=float, default=0.003)
    ap.add_argument("--ball-voxel", type=float, default=0.005)
    ap.add_argument("--membership-radius", type=float, default=0.005)
    return ap


def validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 1 <= args.workers <= MAX_WORKERS:
        ap.error(f"--workers must be 1..{MAX_WORKERS}")
    if args.kmax < 3 or not 1 <= args.k <= args.kmax:
        ap.error("--kmax must be >= 3 (the pre-registered k) and --k in 1..--kmax")
    if args.n_lowest < 1:
        ap.error("--n-lowest must be >= 1")
    if args.sigma < 0 or any(s < 0 for s in args.sigmas):
        ap.error("sigma must be >= 0")
    if args.min_target_px < 0 or args.max_frames < 0 or args.cost_frames < 0:
        ap.error("--min-target-px / --max-frames / --cost-frames must be >= 0")
    if args.sources:
        bad = [s for s in args.sources if s not in E2E_SOURCES]
        if bad:
            ap.error(f"unknown e2e source(s) {bad}; have {list(E2E_SOURCES)}")
    for opt in ("records", "swap_records", "mass_records"):
        if getattr(args, opt):
            hs.resolve_keys(getattr(args, opt), ())


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    validate_args(ap, args)
    root = out_root()
    t0 = time.time()
    {"tokens": stage_tokens, "extract": stage_extract, "select": stage_select,
     "validate": stage_validate, "e2e": stage_e2e, "swap": stage_swap}[args.stage](args, root)
    print(f"[{STEP}] stage {args.stage} done in {time.time() - t0:.1f}s -> {root}")


if __name__ == "__main__":
    main()
