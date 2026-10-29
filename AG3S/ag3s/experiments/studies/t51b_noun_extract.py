"""T51b — T49's noun-row attention for the extra calibration runs ep0 · ep500 · ep1200 (GPU; A1 does not run it).

`T51b.task.md` (2026-10-12): the T51 protocol (P1–P6) is re-run with the **noun-token query rows** in Ω.
T49 extracted those rows only for T47's seven records (``<T49 run>/verify/T49/extract/orig/<key>.npz``);
the three recorded VLA-training runs that T51's 판정 보완 added to the calibration have none. This
module produces them with **T49's extractor, unchanged**: `t49_noun_query_extract.NounQueryExtractor`
(one jitted prefix pass per frame, rows sliced inside the jit), `lookup` (the noun piece must occur
exactly once among the valid text tokens, else the frame is ``ok = False`` with zero maps),
`stamp_of` / `extract_record` (resumable per-frame cache, atomic record npz) and `fields_from_rows`.
The per-frame body (`make_run_frame`) is the body of T49's ``stage_extract.run_frame``
(`t49_noun_query.py:317-340`) with the policy / extractor injected. The only differences from T49's
``--stage extract --no-swaps`` are the record registry (`t51_records`: the extra keys resolve to
``results/workspace/live_test/20260925_train/<ep>/run_0000``) and the output directory. Prompt = the
step record's ``meta.json`` prompt (variant ``orig``), noun = the target parsed from it
(`t51_records.target_of`), queries = (``target``, ``basket``, ``last_text``) — the same rows and the same
npz format as T49's ``orig`` extracts.

    source scripts/setup/env.sh; export AG3S_RUN_DIR=<the T51 run>
    M=AG3S.ag3s.experiments.studies.t51b_noun_extract
    # CPU, openpi venv, no params (reads only the checkpoint's norm stats): noun positions of every frame
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= .venv-openpi-live/bin/python -m $M --stage tokens
    # GPU, under the shared lock, nothing else on the GPU (the stage refuses otherwise)
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m $M --stage parity
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m $M --stage extract
    # CPU, any venv: stamp / format vs T49's orig extracts, t_step vs the T51 attention npz
    .venv-ag3s/bin/python -m $M --stage check

Stages:

``tokens``   CPU. `t48_hsr_features.build_cpu_input_transform` (no model) on every frame of the extra
             runs -> noun index, prefix row, errors -> ``tokens.json`` (T49 ``--stage tokens``'s check).
``parity``   GPU. Re-extract the first ``--parity-frames`` (4) frames of ``ep1807`` (a T47 record T49
             extracted) into ``parity/ep1807.npz`` and compare with T49's ``extract/orig/ep1807.npz``
             (`compare_noun`) -> ``parity_ep1807.json``. Answers "does this environment reproduce T49".
``extract``  GPU. ``orig/<key>.npz`` for ``ep0`` · ``ep500`` · ``ep1200`` (+ ``extract_meta.json``).
             Resumable per frame (``orig/<key>.frames/``); a finished file with the same stamp is skipped.
``check``    CPU. Every extract against T49's ``orig/ep1807.npz`` (same keys, dtypes, trailing shapes,
             checkpoint, config, queries), the record (prompt, noun, t_step) and the T51 attention npz
             (t_step) -> ``check.json``; exits non-zero on any problem.

Output: ``$AG3S_RUN_DIR/verify/T51b/noun_extract/`` (refused without ``AG3S_RUN_DIR``). Read only: T49's
run (``--t49-dir``), the T51 extra runs' attention (``--extra-dir``, default ``$AG3S_RUN_DIR/verify/T51``),
the checkpoint. Nothing touches jax / openpi at import time.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t47_head_selection as hs
from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx
from AG3S.ag3s.experiments.studies import t51_records as rc

STEP = "T51b"
STAGES = ("tokens", "parity", "extract", "check")
#: T49's run (2026-10-12): ``verify/T49/extract/orig/<T47 key>.npz`` — read only.
T49_RUN = "results/development/20261012T190547+0900_28b981600bf7_c39a577332fe"
VARIANT = "orig"
PARITY_KEY = "ep1807"
PARITY_FRAMES = 4
#: Per-frame arrays (frame axis first) — compared on trailing shape and dtype in `check`.
PER_FRAME = (*nx.FRAME_FIELDS, "t_step", "ok", "error", "prefix_ms", "n_text_valid")
#: Scalar / fixed arrays of the record npz (compared on dtype kind and value where meaningful).
FIXED = ("query_names", "cameras", "stamp", "variant", "record", "prompt", "noun", "checkpoint",
         "config", "record_fingerprint")


# ------------------------------------------------------------------------------ plumbing


def out_root() -> pathlib.Path:
    """``$AG3S_RUN_DIR/verify/T51b/noun_extract`` (created); refuses when the run dir is unset."""
    from common.run_output import resolve_output_path

    run_dir = os.environ.get("AG3S_RUN_DIR")
    if not run_dir:
        raise SystemExit(
            "AG3S_RUN_DIR is not set. Use the T51 run:\n"
            "  export AG3S_RUN_DIR=<the run whose verify/T51 holds the extra runs' attention>")
    root = resolve_output_path("verify/T51b/noun_extract", run_dir=run_dir,
                               default_name="verify/T51b/noun_extract")
    root.mkdir(parents=True, exist_ok=True)
    return root


def t49_dir(args) -> pathlib.Path:
    """T49's ``verify/T49`` (read only); refuses when its ``extract/orig`` is missing."""
    p = (pathlib.Path(args.t49_dir) if getattr(args, "t49_dir", None)
         else hs.storage_root() / T49_RUN / "verify" / "T49")
    if not (p / "extract" / VARIANT).is_dir():
        raise SystemExit(f"no T49 extract/{VARIANT} under {p} — pass --t49-dir <run>/verify/T49")
    return p


def t49_extract_path(t49: pathlib.Path, key: str) -> pathlib.Path:
    return pathlib.Path(t49) / "extract" / VARIANT / f"{key}.npz"


def checkpoint_path(args) -> pathlib.Path:
    p = pathlib.Path(args.checkpoint) if args.checkpoint else hs.storage_root() / rc.CHECKPOINT_REL
    if not str(p).rstrip("/").endswith(rc.CHECKPOINT_TAIL):
        raise SystemExit(f"checkpoint {p} is not step {rc.CHECKPOINT_TAIL} (T49 / T47 used that step)")
    if not p.is_dir():
        raise SystemExit(f"checkpoint {p} not found")
    return p


def extract_path(root: pathlib.Path, key: str, variant: str = VARIANT) -> pathlib.Path:
    return pathlib.Path(root) / variant / f"{key}.npz"


def frames_dir(root: pathlib.Path, key: str, variant: str = VARIANT) -> pathlib.Path:
    return pathlib.Path(root) / variant / f"{key}.frames"


def write_json(path: pathlib.Path, payload: Any) -> None:
    from AG3S.ag3s.experiments.studies import t47b_metrics as bm

    hs.write_json(pathlib.Path(path), bm.jsonable(payload))


def run_info(stage: str, args: argparse.Namespace) -> dict:
    return {"step": STEP, "stage": stage, "argv": sys.argv,
            "args": {k: v for k, v in vars(args).items()},
            "started_kst": _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(),
            "git": hs.git_state(), "storage_root": str(hs.storage_root()),
            "run_dir": os.environ.get("AG3S_RUN_DIR"), "python": sys.version.split()[0],
            "extra_dir": rc.extra_dir_or_none(),
            "provenance": rc.provenance(rc.EXTRA_KEYS),
            "extractor": "t49_noun_query_extract.NounQueryExtractor (unchanged), variant orig"}


def extra_keys(requested: Optional[Sequence[str]]) -> list[str]:
    keys = rc.resolve_keys(requested, rc.EXTRA_KEYS)
    bad = [k for k in keys if not rc.is_extra(k)]
    if bad:
        raise SystemExit(f"T49 already extracted the T47 records {bad} (read them from --t49-dir); "
                         f"this stage is for {rc.EXTRA_KEYS} only")
    return keys


def plan_item(key: str) -> dict:
    """``{key, target, variant, prompt, noun, swapped_from}`` — T49's ``orig`` plan item for `key`
    (`t49_noun_query.plan_for` with ``swaps=False``), from the `t51_records` registry."""
    prompt = str(rc.meta_of(key)["prompt"])
    target = rc.target_of(key)
    if target not in core.FRUITS:
        raise SystemExit(f"[{key}] target {target!r} is not one of T49's nouns {core.FRUITS}")
    item = core.variant_plan(prompt, target, swaps=False)[0]
    return {"key": key, "target": target, **item}


def pairing(key: str) -> dict:
    """The step record exists and is the registered scene; when the T51 attention npz exists, the full
    `t51_records.check_pairing` too (the T51b study aligns the noun rows to that npz's t_step)."""
    rp = rc.record_path(key)
    if not (rp / "meta.json").is_file():
        raise SystemExit(f"[{key}] step record missing: {rp}")
    if rc.is_extra(key) and not rc.npz_path(key).is_file():
        probs = rc.episode_problems(key, rc.meta_of(key))
        if probs:
            raise SystemExit(f"[{key}] " + "; ".join(probs))
        return {"key": key, "record": str(rp), "attention_npz": None,
                "pairing": "record + episode only (T51 attention npz not found)"}
    info = rc.check_pairing(key)
    return {**info, "pairing_t51": "t51_records.check_pairing"}


# ------------------------------------------------------------------------------ frame body


def make_run_frame(*, observe: Callable[[Any, str], Any], extractor, ids: dict[str, int], bos: int,
                   item: dict, state: dict) -> Callable[[Any], dict]:
    """T49's ``run_frame`` (`t49_noun_query.py:317-340`) with the policy (`observe(step, prompt)` =
    `observation_from_step(policy, step, prompt)`) and the extractor injected.

    `state` carries ``n_l`` / ``n_h`` (learned from the first pass, or a dummy pass on BOS rows when
    the first frame fails) and ``compile_s`` (the first call, timed apart) across frames and records.
    """

    def run_frame(step) -> dict:
        obs = observe(step, item["prompt"])
        toks = np.asarray(obs.tokenized_prompt)[0]
        mask = np.asarray(obs.tokenized_prompt_mask)[0].astype(bool)
        pos, err = nx.lookup(toks, mask, item["noun"], ids, bos)
        if err:
            if state.get("n_l") is None:  # learn (L, H) (and compile) from a dummy pass on BOS rows
                t0 = time.perf_counter()
                dummy = extractor.rows(obs, np.full(len(core.QUERY_NAMES), core.N_IMAGE_TOKENS,
                                                    np.int32), timed=False)
                state["compile_s"] = time.perf_counter() - t0
                state["n_l"], state["n_h"] = dummy["image"].shape[:2]
            return {"ok": False, "error": "; ".join(err),
                    "fields": nx.empty_fields(state["n_l"], state["n_h"]),
                    "prefix_ms": float("nan"), "n_text_valid": int(mask.sum())}
        rows = core.prefix_rows(pos)
        if state.get("compile_s") is None:  # first call compiles; time it apart
            t0 = time.perf_counter()
            extractor.rows(obs, rows, timed=False)
            state["compile_s"] = time.perf_counter() - t0
        out = extractor.rows(obs, rows)
        state["n_l"], state["n_h"] = out["image"].shape[:2]
        return {"ok": True, "error": "", "fields": nx.fields_from_rows(out, pos, toks),
                "prefix_ms": extractor.t_call[-1] * 1e3, "n_text_valid": int(mask.sum())}

    return run_frame


def extract_one(*, item: dict, record, root: pathlib.Path, variant: str, checkpoint: str, config: str,
                fingerprint: str, run_frame: Callable[[Any], dict], force: bool = False,
                keep_frames: bool = False, limit: int = 0, log: Callable[[str], None] = print) -> dict:
    """One (variant, record) through `t49_noun_query_extract.extract_record` with T49's stamp.

    `limit` > 0 keeps the first `limit` frames (parity); the stamp's ``n_frames`` says so. A finished
    file with the same stamp is skipped; one with another stamp is refused unless `force`.
    """
    steps = list(record.steps)[: limit or None]
    stamp = nx.stamp_of(variant=variant, key=item["key"], prompt=item["prompt"], noun=item["noun"],
                        checkpoint=str(checkpoint), config=config, fingerprint=fingerprint,
                        n_frames=len(steps))
    final = extract_path(root, item["key"], variant)
    fdir = frames_dir(root, item["key"], variant)
    if force:
        final.unlink(missing_ok=True)
        shutil.rmtree(fdir, ignore_errors=True)
    if final.exists():
        got = nx.load_extract(final)["stamp_dict"]
        if got != json.loads(json.dumps(stamp, sort_keys=True)):
            raise SystemExit(f"{final} was made for {got} — pass --force to redo")
        log(f"[{STEP}] {variant}/{item['key']} cached ({final})")
        return {"variant": variant, "record": item["key"], "cached": True, "path": str(final)}
    res = nx.extract_record(frames_dir=fdir, final_path=final, stamp=stamp, steps=steps,
                            run_frame=run_frame, keep_frames=keep_frames, log=log)
    log(f"[{STEP}] {variant}/{item['key']}: {res['n_frames']} frames ({res['computed']} computed, "
        f"{res['resumed']} resumed), errors {res['n_errors']}")
    return res


# ------------------------------------------------------------------------------ comparison


def compare_noun(mine: dict, ref: dict, n: Optional[int] = None) -> dict:
    """Numbers (no verdict) between two noun extracts on their first `n` frames: per query, max / mean
    |Δ| of the maps (float16 on disk), argmax-patch agreement over (frame, layer, head, camera), the
    Pearson r of the flattened maps; mass-summary max |Δ|; token indices and t_step equal or not."""
    n = int(n or min(len(mine["t_step"]), len(ref["t_step"])))
    out: dict[str, Any] = {"n_frames": n,
                           "t_step_equal": bool(np.array_equal(mine["t_step"][:n], ref["t_step"][:n])),
                           "token_index_equal": bool(np.array_equal(mine["token_index"][:n],
                                                                    ref["token_index"][:n])),
                           "token_id_equal": bool(np.array_equal(mine["token_id"][:n],
                                                                 ref["token_id"][:n])),
                           "ok_mine": int(np.asarray(mine["ok"][:n]).sum()),
                           "ok_ref": int(np.asarray(ref["ok"][:n]).sum()), "queries": {}}
    qm = [str(q) for q in mine["query_names"]]
    qr = [str(q) for q in ref["query_names"]]
    for q in qm:
        if q not in qr:
            continue
        a = np.asarray(mine["maps"][:n, qm.index(q)], np.float64)
        b = np.asarray(ref["maps"][:n, qr.index(q)], np.float64)
        d = np.abs(a - b)
        fa = a.reshape(a.shape[:4] + (-1,))
        fb = b.reshape(b.shape[:4] + (-1,))
        am = np.argmax(fa, axis=-1) == np.argmax(fb, axis=-1)
        x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
        den = float(np.sqrt((x * x).sum() * (y * y).sum()))
        out["queries"][q] = {"bitwise_equal": bool(np.array_equal(a, b)),
                             "max_abs": float(d.max()) if d.size else 0.0,
                             "mean_abs": float(d.mean()) if d.size else 0.0,
                             "argmax_agree": float(am.mean()) if am.size else float("nan"),
                             "pearson_r": float((x * y).sum() / den) if den > 0 else float("nan")}
    for f in ("image_mass", "text_mass", "bos", "self"):
        a = np.asarray(mine[f][:n], np.float64)
        b = np.asarray(ref[f][:n], np.float64)
        out[f"{f}_max_abs"] = float(np.abs(a - b).max()) if a.size else 0.0
    return out


def format_problems(ex: dict, ref: dict) -> list[str]:
    """Differences of one extract from T49's reference extract in keys, dtypes and trailing shapes
    (frame counts may differ) and in the stamp's checkpoint / config / queries."""
    probs = []
    for k in (*PER_FRAME, *FIXED):
        if k not in ref:
            continue
        if k not in ex:
            probs.append(f"missing {k}")
            continue
        a, b = np.asarray(ex[k]), np.asarray(ref[k])
        if a.dtype.kind != b.dtype.kind or (a.dtype.kind not in "US" and a.dtype != b.dtype):
            probs.append(f"{k} dtype {a.dtype} != T49 {b.dtype}")
        if k in PER_FRAME and a.shape[1:] != b.shape[1:]:
            probs.append(f"{k} shape {a.shape} vs T49 {b.shape} (trailing must match)")
        if k in ("query_names", "cameras") and [str(x) for x in a] != [str(x) for x in b]:
            probs.append(f"{k} {[str(x) for x in a]} != T49 {[str(x) for x in b]}")
    sm, sr = ex.get("stamp_dict", {}), ref.get("stamp_dict", {})
    for k in ("checkpoint", "config", "queries", "variant"):
        if sm.get(k) != sr.get(k):
            probs.append(f"stamp {k} {sm.get(k)!r} != T49 {sr.get(k)!r}")
    if sorted(sm) != sorted(sr):
        probs.append(f"stamp keys {sorted(sm)} != T49 {sorted(sr)}")
    return probs


def record_problems(key: str, ex: dict, t_record: Sequence[int],
                    t_attention: Optional[Sequence[int]]) -> list[str]:
    """The extract belongs to `key`: stamp record / prompt / noun, frame count and t_step against the
    step record and (when given) the T51 attention npz; every frame extracted (``ok``)."""
    probs = []
    st = ex.get("stamp_dict", {})
    item = plan_item(key)
    for k, want in (("record", key), ("prompt", item["prompt"]), ("noun", item["noun"]),
                    ("variant", VARIANT)):
        if st.get(k) != want:
            probs.append(f"stamp {k} {st.get(k)!r} != {want!r}")
    t = [int(x) for x in np.asarray(ex["t_step"])]
    if t != [int(x) for x in t_record]:
        probs.append(f"t_step {t[:4]}... ({len(t)}) != step record {list(t_record)[:4]}... "
                     f"({len(t_record)})")
    if t_attention is not None and t != [int(x) for x in t_attention]:
        probs.append(f"t_step != T51 attention npz t_step ({len(t_attention)} frames)")
    n_bad = int((~np.asarray(ex["ok"], bool)).sum())
    if n_bad:
        errs = sorted({str(e) for e in ex["error"] if str(e)})
        probs.append(f"{n_bad} frame(s) not extracted (ok=False): {errs[:3]}")
    return probs


def attention_t_steps(key: str) -> Optional[list[int]]:
    """t_step of the T51 attention npz of `key` (only that member is read), None when absent."""
    p = rc.npz_path(key)
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as d:
        return [int(x) for x in d["t_step"]]


def record_t_steps(key: str) -> list[int]:
    return rc._record_t_steps(rc.record_path(key))  # noqa: SLF001 — the registry's own reader


# ------------------------------------------------------------------------------ stages


def stage_tokens(args, root: pathlib.Path) -> dict:
    """CPU noun-position check of every frame of the extra runs through the real input transform."""
    import types

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from AG3S.ag3s.experiments.sources.pi05_attention import observation_from_step
    from AG3S.ag3s.experiments.sources.policy_record import load_run
    from AG3S.ag3s.experiments.studies.t48_hsr_features import build_cpu_input_transform

    ckpt = checkpoint_path(args)
    fn, _decode, info = build_cpu_input_transform(args.config, str(ckpt))
    shim = types.SimpleNamespace(_input_transform=fn)
    sp = nx.sentencepiece_of(args.config)
    ids = nx.noun_ids_for(sp, core.FRUITS)
    keys = extra_keys(args.records)
    results = []
    for k in keys:
        item = plan_item(k)
        record = load_run(rc.record_path(k))
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
        results.append({"key": k, "variant": VARIANT, "prompt": item["prompt"], "noun": item["noun"],
                        "n_frames": len(record), "n_error_frames": len(errors), "errors": errors[:20],
                        "text_index": {q: sorted(v) for q, v in pos_seen.items()},
                        "prefix_row": {q: sorted(core.N_IMAGE_TOKENS + x for x in v)
                                       for q, v in pos_seen.items()},
                        "n_text_valid_min": min(n_valid), "n_text_valid_max": max(n_valid),
                        "pieces_around_frame0": around, "decoded_frame0": decoded})
        print(f"[tokens] {k:>7s} {item['noun']:>7s}: target idx {sorted(pos_seen['target'])} basket "
              f"{sorted(pos_seen['basket'])} last {sorted(pos_seen['last_text'])} errors "
              f"{len(errors)}/{len(record)}")
    payload = {**run_info("tokens", args), "transform": info, "piece_ids": ids,
               "bos_id": int(sp.bos_id()), "n_error_frames": sum(r["n_error_frames"] for r in results),
               "items": results}
    write_json(root / "tokens.json", payload)
    print(f"[tokens] wrote {root / 'tokens.json'}")
    return payload


def gpu_guard(args) -> list[int]:
    """T49's guard (`t49_noun_query.stage_extract`): nvidia-smi present, no compute process (unless
    ``--allow-busy-gpu``), jax on the GPU. Sets the XLA env of `doc/active/SHARED_STORAGE_AND_PYTHON.md`."""
    for k, v in (("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
                 ("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")):
        os.environ.setdefault(k, v)
    pids = nx.gpu_compute_pids()
    if pids is None:
        raise SystemExit("nvidia-smi is unavailable — this stage needs the GPU")
    if pids and not args.allow_busy_gpu:
        raise SystemExit(f"GPU has compute processes {pids} — wait for them (never kill another "
                         "session's process); run under flock /tmp/ag3s-vla-gpu-validation.lock")
    import jax

    if jax.default_backend() != "gpu":
        raise SystemExit(f"jax backend is {jax.default_backend()!r}, not gpu")
    return pids


def _gpu_session(args) -> dict:
    """Load the policy (T49's `load_policy` path) and build the extractor and tokenizer ids once."""
    pids = gpu_guard(args)
    import jax

    from AG3S.ag3s.experiments.sources.pi05_attention import load_policy, observation_from_step

    ckpt = checkpoint_path(args)
    t0 = time.perf_counter()
    policy = load_policy(args.config, str(ckpt))
    load_s = time.perf_counter() - t0
    sp = nx.sentencepiece_of(args.config)
    ids = nx.noun_ids_for(sp, core.FRUITS)
    ex = nx.NounQueryExtractor(policy._model)  # noqa: SLF001
    return {"pids": pids, "jax": jax, "policy": policy, "ckpt": ckpt, "load_s": load_s, "ids": ids,
            "bos": int(sp.bos_id()), "ex": ex, "state": {},
            "observe": lambda step, prompt: observation_from_step(policy, step, prompt)}


def _session_meta(s: dict) -> dict:
    return {"checkpoint": str(s["ckpt"]), "gpu_compute_pids_before": s["pids"],
            "jax_backend": s["jax"].default_backend(), "devices": [str(d) for d in s["jax"].devices()],
            "piece_ids": s["ids"], "bos_id": s["bos"], "queries": list(core.QUERY_NAMES),
            "load_s": s["load_s"], "compile_s": s["state"].get("compile_s"),
            "prefix_rows_timing": nx.timing_summary(
                s["ex"].t_call, "noun prefix pass (embed_prefix + 18 layers + in-jit slice) per frame, "
                                "block_until_ready; excludes input transform, compile, npz write"),
            "memory": s["ex"].memory()}


def _run(s: dict, key: str, root: pathlib.Path, variant: str, args, limit: int = 0) -> dict:
    from AG3S.ag3s.experiments.sources.policy_record import load_run, record_fingerprint

    item = plan_item(key)
    record = load_run(rc.record_path(key))
    run_frame = make_run_frame(observe=s["observe"], extractor=s["ex"], ids=s["ids"], bos=s["bos"],
                               item=item, state=s["state"])
    return extract_one(item=item, record=record, root=root, variant=variant,
                       checkpoint=str(s["ckpt"]), config=args.config,
                       fingerprint=record_fingerprint(record), run_frame=run_frame, force=args.force,
                       keep_frames=args.keep_frames, limit=limit)


def stage_parity(args, root: pathlib.Path) -> dict:
    key = args.parity_record
    if rc.is_extra(key):
        raise SystemExit("parity needs a record T49 extracted (a T47 key)")
    t49 = t49_dir(args)
    ref_path = t49_extract_path(t49, key)
    if not ref_path.is_file():
        raise SystemExit(f"no T49 extract {ref_path}")
    info = pairing(key)
    s = _gpu_session(args)
    res = _run(s, key, root, "parity", args, limit=args.parity_frames)
    cmp = compare_noun(nx.load_extract(extract_path(root, key, "parity")),
                       nx.load_extract(ref_path), args.parity_frames)
    payload = {**run_info("parity", args), **_session_meta(s), "record": info, "result": res,
               "reference": str(ref_path), "comparison": cmp,
               "note": "numbers only; float16 storage and GPU reduction order — bitwise equality is "
                       "not expected (the verdict is the verifier's)"}
    write_json(root / f"parity_{key}.json", payload)
    print(f"[parity] {key}: " + ", ".join(f"{q} max|Δ| {v['max_abs']:.3g} argmax {v['argmax_agree']:.3f}"
                                         for q, v in cmp["queries"].items()))
    return payload


def stage_extract(args, root: pathlib.Path) -> dict:
    keys = extra_keys(args.records)
    infos = [pairing(k) for k in keys]
    s = _gpu_session(args)
    results = [_run(s, k, root, VARIANT, args) for k in keys]
    payload = {**run_info("extract", args), **_session_meta(s), "config": args.config,
               "records": infos, "results": results}
    write_json(root / "extract_meta.json", payload)
    print(f"[extract] wrote {root / 'extract_meta.json'}")
    return payload


def stage_check(args, root: pathlib.Path) -> dict:
    t49 = t49_dir(args)
    ref_path = t49_extract_path(t49, args.reference_record)
    ref = nx.load_extract(ref_path)
    keys = extra_keys(args.records)
    per, all_ok = {}, True
    for k in keys:
        p = extract_path(root, k)
        if not p.is_file():
            per[k] = {"path": str(p), "problems": ["missing — run --stage extract (GPU) first"]}
            all_ok = False
            continue
        ex = nx.load_extract(p)
        t_att = attention_t_steps(k)
        probs = format_problems(ex, ref) + record_problems(k, ex, record_t_steps(k), t_att)
        per[k] = {"path": str(p), "n_frames": int(len(ex["t_step"])),
                  "n_ok": int(np.asarray(ex["ok"]).sum()), "stamp": ex["stamp_dict"],
                  "t51_attention_npz": str(rc.npz_path(k)) if t_att is not None else None,
                  "token_index_unique": {q: sorted({int(x) for x in ex["token_index"][:, i]})
                                         for i, q in enumerate(str(x) for x in ex["query_names"])},
                  "problems": probs}
        all_ok &= not probs
        print(f"[check] {k}: {per[k]['n_frames']} frames, problems {len(probs)}")
    payload = {**run_info("check", args), "reference": str(ref_path), "records": per, "ok": all_ok}
    write_json(root / "check.json", payload)
    if not all_ok:
        raise SystemExit(f"[check] problems: {json.dumps({k: v['problems'] for k, v in per.items() if v['problems']})}")
    print(f"[check] all {len(keys)} extracts match T49's format -> {root / 'check.json'}")
    return payload


# ---------------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--records", nargs="+", default=None, metavar="KEY",
                    help=f"extra runs (default {list(rc.EXTRA_KEYS)})")
    ap.add_argument("--t49-dir", default=None,
                    help=f"T49 verify dir (default $AG3S_STORAGE_ROOT/{T49_RUN}/verify/T49)")
    ap.add_argument("--extra-dir", default=None,
                    help="T51 dir with attention/attention_16d_<ep>.npz (default $AG3S_RUN_DIR/verify/T51)")
    ap.add_argument("--config", default=rc.CONFIG)
    ap.add_argument("--checkpoint", default=None,
                    help=f"default $AG3S_STORAGE_ROOT/{rc.CHECKPOINT_REL}")
    ap.add_argument("--parity-record", default=PARITY_KEY)
    ap.add_argument("--parity-frames", type=int, default=PARITY_FRAMES)
    ap.add_argument("--reference-record", default=PARITY_KEY,
                    help="check: T49 orig extract whose format the new extracts must have")
    ap.add_argument("--force", action="store_true", help="extract / parity: redo cached records")
    ap.add_argument("--keep-frames", action="store_true", help="keep the per-frame cache")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="run even when nvidia-smi lists compute processes (not for shared use)")
    return ap


def validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.parity_frames < 1:
        ap.error("--parity-frames >= 1")
    if args.records:
        rc.resolve_keys(args.records, ())
    if args.config != rc.CONFIG:
        ap.error(f"--config must be {rc.CONFIG} (the config of T49's extracts)")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    validate_args(ap, args)
    root = out_root()
    rc.configure(args.extra_dir)
    t0 = time.time()
    globals()[f"stage_{args.stage}"](args, root)
    print(f"[{STEP}] noun_extract stage {args.stage} done in {time.time() - t0:.1f}s -> {root}")


if __name__ == "__main__":
    main()
