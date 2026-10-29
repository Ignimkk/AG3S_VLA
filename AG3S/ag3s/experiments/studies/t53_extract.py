"""T53 — attention · noun rows · kv_L4 / noun-token features · GT for the 16 final-test rollouts (A1 does not run it).

`T53.task.md` "생성할 것" 1–4: everything is made by the **existing producers, unchanged**; this module only
points them at the T53 records (`t53_records`) and writes under ``<run>/verify/T53/``:

1. ``attention/attention_16d_<key>.npz`` — T47 format. The producer CLI
   `AG3S.ag3s.experiments.sources.pi05_attention` as a subprocess with exactly `t51_extract`'s command
   line (`t51_extract.producer_argv`: ``--config pi05_rby1_randomized_pick_place_16d_lora --num-denoise
   10 --denoise-steps 0 4 9 --agg mean first last --store-dtype float16 --noise-seeds 0``), same
   checkpoint step (`t51_records.CHECKPOINT_REL`), format checked by `t51_records.format_problems`.
2. ``noun/orig/<key>.npz`` — T49 format. `t49_noun_query_extract.NounQueryExtractor` through
   `t51b_noun_extract.make_run_frame` / `extract_one` (T49 stamp, per-frame resume), variant ``orig``,
   queries (target, basket, last_text), prompt = the step record's prompt, noun = the registered target.
3. ``feat/<key>.npz`` — (a) ``kv_l4`` (F, 512): `t48_hsr_features.FeatureExtractor` at batch 1 (the
   serving reduction `AttentionSampler._jit_kv_feature`), inputs `FeatureExtractor.inputs` (state cast
   to float32, the policy's own input transform), i.e. the T48 / SUBTASK-b feature; (b) the **noun-token
   feature** (T55 candidate; the definition is fixed here and T55 chooses among its parts):

   * ``noun_v`` (F, Q, nL, 256) float32 — the **V cache** of the prefix pass at prefix row
     ``768 + text index`` of each query token (Q = target, basket, last_text, `t49_noun_query_core.
     QUERY_NAMES`), kv head 0 (gemma_2b is multi-query: one kv head), for each layer of
     ``noun_layers`` (``--noun-layers``, default all 18). V_l = W_V^l · RMSNorm_l(x_l) is the
     layer-l residual stream at that token projected by the layer's value matrix — the same cache
     tensor `kv_L4` averages (`AttentionSampler._kv_feature`: layer 4, head 0, mean over text / image).
   * ``noun_hidden_final`` (F, Q, 2048) float32 — the prefix output of `PaliGemma.llm` after the final
     RMSNorm (``normed[0]``) at the same rows.

   Same model load (`pi05_attention.load_policy`, ``return_attn_probs=True``), same preprocessed
   observation as (a) (`FeatureExtractor._obs`: one `preprocess_observation`), one extra jitted prefix
   pass (`NounTokenExtractor`) that gathers the rows inside the jit. Token positions:
   `t49_noun_query_extract.lookup` (noun piece exactly once among the valid text tokens, else the frame
   is ``noun_ok = False`` with zeros). ``feat_parity`` checks the gather against the sampler's own KV
   cache (`AttentionSampler._jit_prefix`) and (a) against SUBTASK-b's stored ``normal__kv_L4``.
4. ``gt/<key>.npz`` — T47 GT cache: `t47_replay.extract_gt` (CPU, osmesa) + ``record_key`` + ``pairing``
   (as `t47_head_selection._gt_one`). ``gt/<key>.grasp.npz`` — per-frame physical grasp state
   (`grasp_predicates`): T48b's contact predicate `t48b_scene.classify` on the pose the camera saw
   (`t48b_scene.render_qpos`, one physics step back except a no-warm-up first record) and on the recorded
   qpos: ``held`` = the target touches **both finger bodies of one hand** and neither table nor crate
   (MJCF body names on both sides — the T43 left-finger MJCF/URDF swap cannot arise: no URDF name is
   used and both fingers of a hand are required). ``grasp_state`` per frame:

       held             held by the episode's used arm
       placed           target centre inside the crate's inner volume and not held (``in_crate``)
       pre              before the first ``held`` frame, not placed (on table / closing / pushed)
       post             after the first ``held`` frame, neither held nor placed (set-down, drop, ...)
       held_other_hand  held by the other arm

   plus SUBTASK-b's ``closed_loop_gt`` (T48 label: gripper closed -> held, else near crate -> released,
   else reach) for the phase comparison.

Stages (every command takes ``--run-dir <the T53 run>``; outputs go to ``<run>/verify/T53/``)::

    source scripts/setup/env.sh; R=<the T53 run dir>; M=AG3S.ag3s.experiments.studies.t53_extract
    L=/tmp/ag3s-vla-gpu-validation.lock
    .venv-ag3s/bin/python -m $M --run-dir $R --stage layout                    # CPU: files + pairing
    flock $L .venv-openpi-live/bin/python -m $M --run-dir $R --stage attention_parity
    flock $L .venv-openpi-live/bin/python -m $M --run-dir $R --stage attention
    flock $L .venv-openpi-live/bin/python -m $M --run-dir $R --stage noun_parity
    flock $L .venv-openpi-live/bin/python -m $M --run-dir $R --stage noun
    flock $L .venv-openpi-live/bin/python -m $M --run-dir $R --stage feat_parity
    flock $L .venv-openpi-live/bin/python -m $M --run-dir $R --stage feat
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 .venv-ag3s/bin/python -m $M --run-dir $R --stage gt
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 .venv-ag3s/bin/python -m $M --run-dir $R --stage grasp
    .venv-ag3s/bin/python -m $M --run-dir $R --stage check

GPU stages refuse unless the shared lock is held by someone else (the ``flock`` wrapper; checked with a
non-blocking ``flock`` probe, ``--no-lock-check`` to skip) and nvidia-smi lists no compute process
(``--allow-busy-gpu`` only on the lead's word). The parent of ``attention`` never imports jax. CPU
stages run one worker. Finished files are skipped (``--force`` redoes them); a file that exists but does
not pair is refused, never overwritten silently.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import os
import pathlib
import shutil
import sys
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t53_records as rc

STEP = "T53"
GPU_LOCK = "/tmp/ag3s-vla-gpu-validation.lock"
GPU_STAGES = ("attention_parity", "attention", "noun_parity", "noun", "feat_parity", "feat")
CPU_STAGES = ("layout", "gt", "grasp", "check")
STAGES = ("layout", *GPU_STAGES, "gt", "grasp", "check")
#: T47 record used for every parity stage (T47 asset, T49 extract and SUBTASK-b features exist for it).
PARITY_KEY = "ep1807"
PARITY_FRAMES = 4
N_LAYERS = 18
#: SUBTASK-b's stored features of ep1807 (`t48_hsr_io.SUBTASK_B_FEAT_REL` / closed run file).
SUBTASK_B_EP1807 = "results/workspace/verify/SUBTASK-b/feat/closed/ep1807_apple.npz"
#: A T47-format GT cache whose keys / dtypes / trailing shapes every T53 cache must have (T51's
#: gt_extra of ep0, made by the same `extract_gt` + record_key + pairing).
GT_REFERENCE_REL = ("results/development/20261012T180318+0900_28b981600bf7_852011fdcae0/verify/T51/"
                    "gt_cache_extra/ep0.npz")
GT_PER_FRAME = ("t_step", "class_cov", "class_px", "body_cov", "body_px", "marker_px", "label_small",
                "target_pos_base", "fruit_pos_base")
GRASP_STATES = ("pre", "held", "placed", "post", "held_other_hand")
FEAT_VERSION = "t53-feat-v1"
GRASP_VERSION = "t53-grasp-v1"


# ------------------------------------------------------------------------------ plumbing


def now_kst() -> str:
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat()


def write_json(path: pathlib.Path, payload: Any) -> None:
    from AG3S.ag3s.experiments.studies import t47_head_selection as hs
    from AG3S.ag3s.experiments.studies import t47b_metrics as bm

    hs.write_json(pathlib.Path(path), bm.jsonable(payload))


def save_npz_atomic(path: pathlib.Path, **arrays) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name[:-4] + ".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def run_info(stage: str, args: argparse.Namespace) -> dict:
    from AG3S.ag3s.experiments.studies import t47_head_selection as hs

    return {"step": STEP, "stage": stage, "argv": sys.argv,
            "args": {k: v for k, v in vars(args).items()}, "started_kst": now_kst(),
            "git": hs.git_state(), "storage_root": str(rc.storage_root()),
            "t53_dir": str(rc.t53_dir()), "episodes_root": str(rc.episodes_root()),
            "python": sys.executable, "python_version": sys.version.split()[0]}


def lock_is_held(path: str = GPU_LOCK) -> Optional[bool]:
    """True when another open file description holds an exclusive ``flock`` on `path` (the caller's
    ``flock`` wrapper), False when it could be taken (released at once), None when `path` is absent."""
    if not os.path.exists(path):
        return None
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def require_gpu_lock(args) -> None:
    if args.no_lock_check:
        return
    held = lock_is_held(args.gpu_lock)
    if not held:
        raise SystemExit(f"GPU stage {args.stage!r} must run under the shared lock: "
                         f"flock {args.gpu_lock} <this command> (lock held: {held}); "
                         "--no-lock-check only when the lead says so")


def checkpoint(args) -> pathlib.Path:
    """The T47 / T49 / T51 checkpoint step (`t51_extract.checkpoint_path`: params/ + step tail)."""
    from AG3S.ag3s.experiments.studies import t51_extract as ex51

    return ex51.checkpoint_path(args)


def preflight(keys: Sequence[str]) -> dict[str, rc.EpisodeFiles]:
    """Before any model load: every record exists, is the registered episode and prompt."""
    out, bad = {}, {}
    for k in keys:
        files = rc.locate(k)
        probs = rc.record_problems(k, rc.read_meta(files.record))
        if probs:
            bad[k] = probs
        out[k] = files
    if bad:
        raise SystemExit("records do not match the T53 registry:\n  "
                         + "\n  ".join(f"{k}: {'; '.join(p)}" for k, p in bad.items()))
    return out


# ------------------------------------------------------------------------------ layout (CPU)


def stage_layout(args) -> dict:
    keys = rc.resolve_keys(args.records)
    try:
        ds_probs = rc.dataset_problems(rc.load_dataset_meta(args.dataset))
    except OSError as exc:
        ds_probs = [f"dataset metadata unreadable: {exc}"]
    per, ok = {}, not ds_probs
    for k in keys:
        try:
            info, probs = rc.pairing_report(k, require_chunks=args.require_chunks,
                                            server_roots=[pathlib.Path(p) for p in args.server_roots])
        except SystemExit as exc:
            info, probs = {"key": k}, [str(exc)]
        per[k] = {**info, "problems": probs}
        ok &= not probs
        print(f"[layout] {k}: {info.get('n_frames')} frames, seed {info.get('policy_seed')}, "
              f"chunks {info.get('chunks', {}).get('n')}, problems {len(probs)}"
              + ("" if not probs else f" -> {probs[0]}"))
    payload = {**run_info("layout", args), "registry": rc.registry_table(),
               "dataset_problems": ds_probs, "records": per, "all_ok": ok,
               "provenance": rc.provenance(keys)}
    write_json(rc.t53_dir() / "layout.json", payload)
    if not ok:
        raise SystemExit(f"[layout] problems — see {rc.t53_dir() / 'layout.json'}")
    return payload


# ------------------------------------------------------------------------------ 1. attention


def stage_attention_parity(args) -> dict:
    """`t51_extract.stage_parity` with the output under ``T53/attention``."""
    from AG3S.ag3s.experiments.studies import t47_head_selection as hs
    from AG3S.ag3s.experiments.studies import t51_extract as ex51
    from AG3S.ag3s.experiments.studies import t51_records as rc51

    require_gpu_lock(args)
    ex51.gpu_guard(args)
    key = args.parity_record
    info = rc51.check_pairing(key)
    ckpt = checkpoint(args)
    n = int(args.parity_frames)
    d = rc.out_dir("attention")
    dest = d / f"parity_{key}_first{n}.npz"
    argv = ex51.producer_argv(hs.record_path(key), ckpt, dest, limit=n)
    res = ex51.run_producer(argv, d / f"parity_{key}.log", ex51.producer_env(args.jax_platforms))
    if res["returncode"] != 0 or not dest.is_file():
        raise SystemExit(f"producer failed ({res}); see {res['log']}")
    with np.load(dest, allow_pickle=False) as new, np.load(hs.npz_path(key), allow_pickle=False) as ref:
        cmp = ex51.compare_attention(new["attention"], ref["attention"][:n])
        t_same = bool(np.array_equal(new["t_step"], ref["t_step"][:n]))
        fields = {k: new[k] for k in new.files if k != "attention"}
    shape, dtype = rc51.npz_array_header(dest, "attention")
    payload = {**run_info("attention_parity", args), "record": info, "checkpoint": str(ckpt),
               "argv_producer": argv, "producer": res, "t_step_same": t_same,
               "format_problems": rc51.format_problems(fields, shape, dtype), "comparison": cmp,
               "output": str(dest),
               "note": "numbers only; float16 storage and GPU reduction order — the verdict is the verifier's"}
    write_json(d / f"parity_{key}.json", payload)
    print(f"[attention_parity] {key} first {n}: argmax same {cmp['argmax_patch_same_share']:.4f}, "
          f"max |d| {cmp['max_abs']:.2e}, L8H2 r {cmp['serving_cell_pearson_r']:.5f}")
    return payload


def _attention_ok(key: str, record: pathlib.Path) -> tuple[dict, list[str]]:
    meta = rc.read_meta(record)
    return rc.attention_problems(rc.npz_path(key), record, str(meta["prompt"]),
                                 rc.record_t_steps(record))


def stage_attention(args) -> dict:
    from AG3S.ag3s.experiments.studies import t51_extract as ex51

    require_gpu_lock(args)
    keys = rc.resolve_keys(args.records)
    files = preflight(keys)
    pids = ex51.gpu_guard(args)
    ckpt = checkpoint(args)
    env = ex51.producer_env(args.jax_platforms)
    d = rc.out_dir("attention")
    done, skipped = [], []
    for k in keys:
        rec = files[k].record
        final = rc.npz_path(k)
        if final.is_file() and not args.force:
            info, probs = _attention_ok(k, rec)
            if not probs:
                skipped.append(info)
                print(f"[attention] {k}: exists and pairs, skipping ({final})")
                continue
            raise SystemExit(f"[{k}] {final} exists but does not pair ({probs}) — inspect it; "
                             "--force to re-extract")
        part = final.with_name(final.stem + ".partial.npz")
        argv = ex51.producer_argv(rec, ckpt, part)
        print(f"[attention] {k}: {rec} -> {part}", flush=True)
        res = ex51.run_producer(argv, d / f"extract_{k}.log", env)
        if res["returncode"] != 0 or not part.is_file():
            raise SystemExit(f"[{k}] producer failed ({res}); see {res['log']}")
        part.replace(final)
        info, probs = _attention_ok(k, rec)
        if probs:
            raise SystemExit(f"[{k}] extracted npz does not pair: {probs} (left at {final})")
        done.append({**info, **res, "bytes": final.stat().st_size, "argv": argv})
        print(f"[attention] {k}: {info['n_frames']} frames in {res['seconds']:.0f}s", flush=True)
    payload = {**run_info("attention", args), "checkpoint": str(ckpt),
               "producer": ex51.PRODUCER, "producer_args": list(ex51.PRODUCER_ARGS),
               "gpu_compute_pids_before": pids, "extracted": done, "skipped": skipped,
               "provenance": rc.provenance(keys)}
    write_json(d / "extract.json", payload)
    return payload


# ------------------------------------------------------------------------------ 2. noun rows


def noun_item(key: str, prompt: str) -> dict:
    """T49's ``orig`` plan item (`t49_noun_query_core.variant_plan`, no swaps) of a T53 key."""
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core

    target = rc.target_of(key)
    if target not in core.FRUITS:
        raise SystemExit(f"[{key}] target {target!r} not in {core.FRUITS}")
    return {"key": key, "target": target, **core.variant_plan(prompt, target, swaps=False)[0]}


def stage_noun_parity(args) -> dict:
    """`t51b_noun_extract.stage_parity`: the first frames of ep1807 vs T49's ``orig`` extract."""
    from AG3S.ag3s.experiments.sources.policy_record import load_run, record_fingerprint
    from AG3S.ag3s.experiments.studies import t47_head_selection as hs
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx
    from AG3S.ag3s.experiments.studies import t51b_noun_extract as ne

    require_gpu_lock(args)
    key = args.parity_record
    ref_path = ne.t49_extract_path(ne.t49_dir(args), key)
    if not ref_path.is_file():
        raise SystemExit(f"no T49 extract {ref_path}")
    record = load_run(hs.record_path(key))
    item = {"key": key, "target": hs.target_of(key),
            **core.variant_plan(record.prompt, hs.target_of(key), swaps=False)[0]}
    s = ne._gpu_session(args)  # noqa: SLF001 — T51b's session (guard, load_policy, extractor, ids)
    run_frame = ne.make_run_frame(observe=s["observe"], extractor=s["ex"], ids=s["ids"], bos=s["bos"],
                                  item=item, state=s["state"])
    root = rc.out_dir("noun")
    res = ne.extract_one(item=item, record=record, root=root, variant="parity", checkpoint=str(s["ckpt"]),
                         config=args.config, fingerprint=record_fingerprint(record), run_frame=run_frame,
                         force=args.force, keep_frames=args.keep_frames, limit=args.parity_frames)
    cmp = ne.compare_noun(nx.load_extract(ne.extract_path(root, key, "parity")),
                          nx.load_extract(ref_path), args.parity_frames)
    payload = {**run_info("noun_parity", args), **ne._session_meta(s), "result": res,  # noqa: SLF001
               "reference": str(ref_path), "comparison": cmp,
               "note": "numbers only; bitwise equality is not expected (the verdict is the verifier's)"}
    write_json(root / f"parity_{key}.json", payload)
    print(f"[noun_parity] {key}: " + ", ".join(
        f"{q} max|Δ| {v['max_abs']:.3g} argmax {v['argmax_agree']:.3f}" for q, v in cmp["queries"].items()))
    return payload


def stage_noun(args) -> dict:
    from AG3S.ag3s.experiments.sources.policy_record import load_run, record_fingerprint
    from AG3S.ag3s.experiments.studies import t51b_noun_extract as ne

    require_gpu_lock(args)
    keys = rc.resolve_keys(args.records)
    files = preflight(keys)
    s = ne._gpu_session(args)  # noqa: SLF001
    root = rc.out_dir("noun")
    results, items = [], {}
    for k in keys:
        record = load_run(files[k].record)
        item = noun_item(k, record.prompt)
        items[k] = item
        run_frame = ne.make_run_frame(observe=s["observe"], extractor=s["ex"], ids=s["ids"],
                                      bos=s["bos"], item=item, state=s["state"])
        results.append(ne.extract_one(item=item, record=record, root=root, variant=rc.NOUN_VARIANT,
                                      checkpoint=str(s["ckpt"]), config=args.config,
                                      fingerprint=record_fingerprint(record), run_frame=run_frame,
                                      force=args.force, keep_frames=args.keep_frames))
    payload = {**run_info("noun", args), **ne._session_meta(s), "config": args.config,  # noqa: SLF001
               "items": items, "results": results, "provenance": rc.provenance(keys),
               "extractor": "t49_noun_query_extract.NounQueryExtractor via t51b_noun_extract (unchanged)"}
    write_json(root / "extract_meta.json", payload)
    return payload


def noun_record_problems(key: str, ex: dict, t_record: Sequence[int],
                         t_attention: Optional[Sequence[int]], prompt: str) -> list[str]:
    """`t51b_noun_extract.record_problems` for a T53 key (registry instead of `t51_records`)."""
    probs = []
    st = ex.get("stamp_dict", {})
    item = noun_item(key, prompt)
    for k, want in (("record", key), ("prompt", item["prompt"]), ("noun", item["noun"]),
                    ("variant", rc.NOUN_VARIANT)):
        if st.get(k) != want:
            probs.append(f"stamp {k} {st.get(k)!r} != {want!r}")
    t = [int(x) for x in np.asarray(ex["t_step"])]
    if t != [int(x) for x in t_record]:
        probs.append(f"t_step ({len(t)}) != step record ({len(t_record)})")
    if t_attention is not None and t != [int(x) for x in t_attention]:
        probs.append(f"t_step != attention npz t_step ({len(t_attention)} frames)")
    n_bad = int((~np.asarray(ex["ok"], bool)).sum())
    if n_bad:
        errs = sorted({str(e) for e in ex["error"] if str(e)})
        probs.append(f"{n_bad} frame(s) not extracted (ok=False): {errs[:3]}")
    return probs


# ------------------------------------------------------------------------------ 3. features


def noun_token_impl(model, obs, rows, layers, *, make_attn_mask, xp):
    """Traced body of `NounTokenExtractor`: one prefix pass (as `AttentionSampler._prefix`), then
    ``v`` = V cache ``[layers, batch 0, rows, kv head 0, :]`` -> (nL, R, H) and ``hidden_final`` =
    ``normed[0][batch 0, rows]`` -> (R, D), both float32. `obs` is already preprocessed."""
    tokens, mask, ar_mask = model.embed_prefix(obs)
    n_image = tokens.shape[1] - obs.tokenized_prompt.shape[1]
    from AG3S.ag3s.experiments.studies.t49_noun_query_core import N_IMAGE_TOKENS

    if n_image != N_IMAGE_TOKENS:  # static shapes: checked at trace time
        raise ValueError(f"prefix has {n_image} image tokens, expected {N_IMAGE_TOKENS}")
    attn = make_attn_mask(mask, ar_mask)
    positions = xp.cumsum(mask, axis=1) - 1
    out = model.PaliGemma.llm([tokens, None], mask=attn, positions=positions)
    normed, kv = out[0], out[1]
    v = kv[1]  # (L, B, S, K, H)
    v_sel = v[layers][:, 0][:, rows][:, :, 0, :]
    h_sel = normed[0][0][rows]
    return {"v": v_sel.astype(xp.float32), "hidden_final": h_sel.astype(xp.float32)}


class NounTokenExtractor:
    """One jitted prefix executable returning `noun_token_impl` for given rows / layers."""

    def __init__(self, model):
        import flax.nnx as nnx
        import jax

        self._jax = jax
        self._nnx = nnx
        self._graphdef, self._state = nnx.split(model)
        self._jit = jax.jit(self._impl)
        self.t_call: list[float] = []

    def _impl(self, state, obs, rows, layers):
        import jax.numpy as jnp

        from openpi.models.pi0 import make_attn_mask

        model = self._nnx.merge(self._graphdef, state)
        return noun_token_impl(model, obs, rows, layers, make_attn_mask=make_attn_mask, xp=jnp)

    def rows(self, obs, rows: np.ndarray, layers: np.ndarray, *, timed: bool = True) -> dict:
        import jax.numpy as jnp

        r = jnp.asarray(np.asarray(rows, np.int32))
        ly = jnp.asarray(np.asarray(layers, np.int32))
        t0 = time.perf_counter()
        out = self._jax.block_until_ready(self._jit(self._state, obs, r, ly))
        if timed:
            self.t_call.append(time.perf_counter() - t0)
        return {k: np.asarray(v, np.float32) for k, v in out.items()}


def feature_frame(inputs: dict, *, kv_fn: Callable[[dict], np.ndarray], obs_fn: Callable[[dict], Any],
                  noun_fn: Callable[[Any, np.ndarray], dict], noun: str, ids: dict, bos: int,
                  n_layers: int, hidden_dim: Optional[int] = None, head_dim: Optional[int] = None,
                  state: Optional[dict] = None) -> dict:
    """One frame of the feat stage. `inputs` = `FeatureExtractor.inputs(...)` (no batch axis);
    `kv_fn` -> (512,) kv_L4; `obs_fn` -> preprocessed observation; `noun_fn(obs, rows)` -> ``{"v":
    (nL, R, H), "hidden_final": (R, D)}``. A frame whose noun is not found exactly once gets zeros
    (shapes from `state`, learned from the first successful frame or given)."""
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx

    state = {} if state is None else state
    toks = np.asarray(inputs["tokenized_prompt"]).reshape(-1)
    mask = np.asarray(inputs["tokenized_prompt_mask"]).reshape(-1).astype(bool)
    kv = np.asarray(kv_fn(inputs), np.float32).reshape(-1)
    pos, err = nx.lookup(toks, mask, noun, ids, bos)
    q = len(core.QUERY_NAMES)
    if err:
        obs = obs_fn(inputs)
        if state.get("H") is None:  # learn the shapes from a pass on BOS rows
            dummy = noun_fn(obs, np.full(q, core.N_IMAGE_TOKENS, np.int32))
            state["H"], state["D"] = dummy["v"].shape[-1], dummy["hidden_final"].shape[-1]
        return {"kv_l4": kv, "noun_ok": False, "noun_error": "; ".join(err),
                "noun_v": np.zeros((q, n_layers, state["H"]), np.float32),
                "noun_hidden_final": np.zeros((q, state["D"]), np.float32),
                "noun_token_index": np.full(q, -1, np.int64), "noun_prefix_row": np.full(q, -1, np.int64),
                "noun_token_id": np.full(q, -1, np.int64), "n_text_valid": int(mask.sum())}
    rows = core.prefix_rows(pos)
    out = noun_fn(obs_fn(inputs), rows)
    v = np.asarray(out["v"], np.float32)  # (nL, R, H)
    if v.shape[:2] != (n_layers, q):
        raise ValueError(f"noun V shape {v.shape} != ({n_layers}, {q}, H)")
    h = np.asarray(out["hidden_final"], np.float32)
    state["H"], state["D"] = v.shape[-1], h.shape[-1]
    tidx = np.asarray([pos[name] for name in core.QUERY_NAMES], np.int64)
    return {"kv_l4": kv, "noun_ok": True, "noun_error": "",
            "noun_v": np.moveaxis(v, 1, 0),  # (Q, nL, H)
            "noun_hidden_final": h, "noun_token_index": tidx, "noun_prefix_row": tidx + core.N_IMAGE_TOKENS,
            "noun_token_id": toks[tidx].astype(np.int64), "n_text_valid": int(mask.sum())}


FEAT_FRAME_FIELDS = ("kv_l4", "noun_v", "noun_hidden_final", "noun_token_index", "noun_prefix_row",
                     "noun_token_id")


def assemble_feat(frames: Sequence[dict], t_step: Sequence[int], stamp: dict,
                  extra: Optional[dict] = None) -> dict[str, np.ndarray]:
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core

    if len(frames) != len(t_step) or len(frames) != int(stamp["n_frames"]):
        raise ValueError(f"{len(frames)} frames, {len(t_step)} t_step, stamp {stamp['n_frames']}")
    out = {k: np.stack([np.asarray(f[k]) for f in frames]) for k in FEAT_FRAME_FIELDS}
    out.update({
        "t_step": np.asarray(t_step, np.int64),
        "noun_ok": np.asarray([bool(f["noun_ok"]) for f in frames]),
        "noun_error": np.asarray([str(f["noun_error"]) for f in frames]),
        "n_text_valid": np.asarray([int(f["n_text_valid"]) for f in frames], np.int64),
        "query_names": np.asarray(core.QUERY_NAMES),
        "noun_layers": np.asarray(stamp["noun_layers"], np.int64),
        "stamp": np.asarray(json.dumps(stamp, sort_keys=True)),
        **{k: np.asarray(stamp[k]) for k in ("record", "prompt", "noun", "checkpoint", "config",
                                              "record_fingerprint", "version")},
    })
    for k, v in (extra or {}).items():
        out[k] = np.asarray(v)
    return out


def feat_stamp(*, key: str, prompt: str, noun: str, checkpoint: str, config: str, fingerprint: str,
               n_frames: int, layers: Sequence[int]) -> dict:
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core

    return {"version": FEAT_VERSION, "record": key, "prompt": prompt, "noun": noun,
            "checkpoint": str(checkpoint), "config": config, "record_fingerprint": fingerprint,
            "n_frames": int(n_frames), "noun_layers": [int(x) for x in layers],
            "queries": list(core.QUERY_NAMES),
            "kv_l4": "t48_hsr_features.FeatureExtractor batch 1 (AttentionSampler._jit_kv_feature)",
            "noun_v": "V cache [layer, batch 0, 768 + text index, kv head 0, :]",
            "noun_hidden_final": "PaliGemma.llm prefix output after final RMSNorm at 768 + text index"}


def load_stamp(path: pathlib.Path) -> Optional[dict]:
    try:
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z["stamp"]))
    except Exception:  # noqa: BLE001
        return None


def _feat_session(args) -> dict:
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx
    from AG3S.ag3s.experiments.studies import t51b_noun_extract as ne
    from AG3S.ag3s.experiments.studies.t48_hsr_features import FeatureExtractor

    pids = ne.gpu_guard(args)  # nvidia-smi empty, XLA env, jax backend gpu
    ckpt = checkpoint(args)
    fx = FeatureExtractor(args.config, str(ckpt), batch=1)
    nt = NounTokenExtractor(fx.policy._model)  # noqa: SLF001
    sp = nx.sentencepiece_of(args.config)
    return {"pids": pids, "ckpt": ckpt, "fx": fx, "nt": nt, "ids": nx.noun_ids_for(sp, core.FRUITS),
            "bos": int(sp.bos_id()), "state": {}}


def _feat_fns(s: dict, layers: np.ndarray):
    fx, nt = s["fx"], s["nt"]

    def kv_fn(inputs):
        return fx.features([inputs])[0]

    def obs_fn(inputs):
        return fx._obs([inputs])  # noqa: SLF001 — the extractor's own (stack + preprocess once)

    def noun_fn(obs, rows):
        if s["state"].get("compiled") is None:
            t0 = time.perf_counter()
            nt.rows(obs, rows, layers, timed=False)
            s["state"]["compiled"] = time.perf_counter() - t0
        return nt.rows(obs, rows, layers)

    return kv_fn, obs_fn, noun_fn


def stage_feat_parity(args) -> dict:
    """kv_L4 of the first frames of ep1807 vs SUBTASK-b's ``normal__kv_L4`` (by ``file_index``), and the
    noun-token V gather vs the sampler's own cache (`AttentionSampler._jit_prefix`) on the same obs."""
    from AG3S.ag3s.experiments.sources.policy_record import load_run
    from AG3S.ag3s.experiments.studies import t47_head_selection as hs
    from AG3S.ag3s.experiments.studies import t49_noun_query_core as core
    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx

    require_gpu_lock(args)
    key = args.parity_record
    ref_path = pathlib.Path(args.subtask_b_feat or rc.storage_root() / SUBTASK_B_EP1807)
    if not ref_path.is_file():
        raise SystemExit(f"no SUBTASK-b feature file {ref_path}")
    record = load_run(hs.record_path(key), limit=args.parity_frames)
    target = hs.target_of(key)
    s = _feat_session(args)
    fx, layers = s["fx"], np.asarray(args.noun_layers, np.int32)
    kv_fn, obs_fn, noun_fn = _feat_fns(s, layers)
    with np.load(ref_path, allow_pickle=False) as z:
        ref_kv = np.asarray(z["normal__kv_L4"], np.float64)
        ref_idx = [int(x) for x in z["file_index"]]
    rows_out, v_diff, det = [], [], []
    for i, step in enumerate(record):
        inputs = fx.inputs(step.state, step.images, record.prompt)
        got = np.asarray(kv_fn(inputs), np.float64)
        r = {"frame": i, "t_step": int(step.t_step)}
        if i in ref_idx:
            want = ref_kv[ref_idx.index(i)]
            r["kv_l4_rel_l2"] = float(np.linalg.norm(got - want) / np.linalg.norm(want))
            r["kv_l4_max_abs"] = float(np.abs(got - want).max())
        toks = np.asarray(inputs["tokenized_prompt"])
        mask = np.asarray(inputs["tokenized_prompt_mask"]).astype(bool)
        pos, err = nx.lookup(toks, mask, target, s["ids"], s["bos"])
        if not err:
            obs = obs_fn(inputs)
            rows = core.prefix_rows(pos)
            a = noun_fn(obs, rows)
            b = noun_fn(obs, rows)
            det.append(bool(np.array_equal(a["v"], b["v"]) and np.array_equal(a["hidden_final"],
                                                                               b["hidden_final"])))
            kv, _mask = fx.sampler._jit_prefix(fx.sampler._state, obs)  # noqa: SLF001
            ref_v = np.asarray(kv[1], np.float32)[layers][:, 0][:, rows][:, :, 0, :]
            v_diff.append(float(np.abs(ref_v - a["v"]).max()))
            r["noun_v_vs_sampler_cache_max_abs"] = v_diff[-1]
        else:
            r["noun_error"] = err
        rows_out.append(r)
    rel = [r["kv_l4_rel_l2"] for r in rows_out if "kv_l4_rel_l2" in r]
    payload = {**run_info("feat_parity", args), "record": key, "reference": str(ref_path),
               "checkpoint": str(s["ckpt"]), "noun_layers": layers.tolist(), "frames": rows_out,
               "kv_l4_rel_l2_max": max(rel) if rel else None,
               "noun_v_vs_sampler_cache_max_abs": max(v_diff) if v_diff else None,
               "noun_token_deterministic": all(det) if det else None,
               "timing": fx.timing(), "memory": fx.memory(),
               "note": "numbers only; the verdict is the verifier's"}
    write_json(rc.out_dir("feat") / f"parity_{key}.json", payload)
    print(f"[feat_parity] {key}: kv_L4 rel L2 max {payload['kv_l4_rel_l2_max']}, noun V vs cache max "
          f"|Δ| {payload['noun_v_vs_sampler_cache_max_abs']}, deterministic "
          f"{payload['noun_token_deterministic']}")
    return payload


def stage_feat(args) -> dict:
    from AG3S.ag3s.experiments.sources.policy_record import load_run, record_fingerprint

    require_gpu_lock(args)
    keys = rc.resolve_keys(args.records)
    files = preflight(keys)
    layers = np.asarray(args.noun_layers, np.int32)
    s = None
    out_d = rc.out_dir("feat")
    results = []
    for k in keys:
        record = load_run(files[k].record)
        noun = rc.target_of(k)
        stamp = feat_stamp(key=k, prompt=record.prompt, noun=noun,
                           checkpoint=str(checkpoint(args)), config=args.config,
                           fingerprint=record_fingerprint(record), n_frames=len(record),
                           layers=layers.tolist())
        final = rc.feat_path(k)
        if final.is_file() and not args.force:
            got = load_stamp(final)
            if got == json.loads(json.dumps(stamp, sort_keys=True)):
                print(f"[feat] {k}: cached ({final})")
                results.append({"key": k, "cached": True, "path": str(final)})
                continue
            raise SystemExit(f"{final} was made for {got} — --force to redo")
        if s is None:
            s = _feat_session(args)
            kv_fn, obs_fn, noun_fn = _feat_fns(s, layers)
        fx = s["fx"]
        t0 = time.time()
        frames, decoded = [], []
        for step in record:
            inputs = fx.inputs(step.state, step.images, record.prompt)
            if not decoded:
                decoded.append(fx.decoded_prompt(inputs))
            frames.append(feature_frame(inputs, kv_fn=kv_fn, obs_fn=obs_fn, noun_fn=noun_fn, noun=noun,
                                        ids=s["ids"], bos=s["bos"], n_layers=len(layers),
                                        state=s["state"]))
        payload = assemble_feat(frames, [st.t_step for st in record], stamp,
                                extra={"decoded_prompt_frame0": decoded[0] if decoded else ""})
        save_npz_atomic(final, **payload)
        n_bad = int((~payload["noun_ok"]).sum())
        results.append({"key": k, "n_frames": len(frames), "noun_errors": n_bad,
                        "seconds": time.time() - t0, "path": str(final)})
        print(f"[feat] {k}: {len(frames)} frames, noun errors {n_bad}, {time.time() - t0:.1f}s", flush=True)
    meta = {**run_info("feat", args), "results": results, "noun_layers": layers.tolist(),
            "provenance": rc.provenance(keys)}
    if s is not None:
        meta.update({"checkpoint": str(s["ckpt"]), "gpu_compute_pids_before": s["pids"],
                     "kv_timing": s["fx"].timing(), "memory": s["fx"].memory(),
                     "noun_compile_s": s["state"].get("compiled"),
                     "noun_ms_median": (float(np.median(s["nt"].t_call) * 1e3) if s["nt"].t_call
                                        else None)})
    write_json(out_d / "extract_meta.json", meta)
    return meta


# ------------------------------------------------------------------------------ 4. GT (CPU)


def gt_one(key: str, record: pathlib.Path, *, rgb_check_frames: int) -> dict:
    """`t47_head_selection._gt_one` for a T53 key: `t47_replay.extract_gt` + record_key + pairing."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.studies.t47_head_selection import POLICY_CAMERAS
    from AG3S.ag3s.experiments.studies.t47_replay import extract_gt

    meta = rc.read_meta(record)
    pairing = {"key": key, "record": str(record), "prompt": meta.get("prompt"),
               "episode_index": rc.episode_of(key).episode, "scene_id": rc.scene_of(key),
               "role": rc.ROLE, "pairing": "t53_records.record_problems (episode, prompt, target)"}
    t0 = time.time()
    arrays = extract_gt(record, target=rc.target_of(key), rgb_check_frames=rgb_check_frames,
                        log=lambda m: print(f"[{key}]{m}", flush=True))
    arrays["record_key"] = np.asarray(key)
    arrays["pairing"] = np.asarray(json.dumps(pairing))
    save_npz_atomic(rc.gt_cache_path(key), **arrays)
    tpx = arrays["class_px"][:, :, 0]
    return {**pairing, "cache": str(rc.gt_cache_path(key)), "seconds": time.time() - t0,
            "n_frames": int(tpx.shape[0]),
            "target_visible_ge200px_per_camera": {c: int((tpx[:, i] >= 200).sum())
                                                  for i, c in enumerate(POLICY_CAMERAS)},
            "rgb_mad": arrays["rgb_mad"].tolist(), "rgb_frac_gt16": arrays["rgb_frac_gt16"].tolist()}


def stage_gt(args) -> dict:
    keys = rc.resolve_keys(args.records)
    files = preflight(keys)
    res, skipped = [], []
    for k in keys:
        if rc.gt_cache_path(k).is_file() and not args.force:
            skipped.append(k)
            print(f"[gt] {k}: cached ({rc.gt_cache_path(k)})")
            continue
        res.append(gt_one(k, files[k].record, rgb_check_frames=args.rgb_check_frames))
        print(f"[gt] {k}: {res[-1]['n_frames']} frames in {res[-1]['seconds']:.0f}s, target >=200 px "
              f"{res[-1]['target_visible_ge200px_per_camera']}", flush=True)
    payload = {**run_info("gt", args), "rendered": res, "skipped_cached": skipped, "workers": 1,
               "format": "T47 gt_cache (t47_replay.extract_gt + record_key + pairing)",
               "provenance": rc.provenance(keys)}
    write_json(rc.out_dir("gt") / "gt_summary.json", payload)
    return payload


def grasp_state_sequence(states: Sequence[str], hands: Sequence[Optional[str]], used_arm: str) -> list[str]:
    """Per-frame grasp state from T48b physical states (module docstring, item 4)."""
    out, seen = [], False
    for s, h in zip(states, hands):
        if s == "held" and h == used_arm:
            seen = True
            out.append("held")
        elif s == "held":
            out.append("held_other_hand")
        elif s == "in_crate":
            out.append("placed")
        else:
            out.append("post" if seen else "pre")
    return out


def state_counts(labels: Sequence[str], names: Sequence[str] = GRASP_STATES) -> dict[str, int]:
    return {n: int(sum(x == n for x in labels)) for n in names}


def grasp_predicates(record: pathlib.Path, *, target: str, used_arm: str) -> dict[str, np.ndarray]:
    """Per frame of the step record: T48b `classify` on the render pose and on the recorded qpos,
    the grasp-state sequences, SUBTASK-b `closed_loop_gt`, target / crate world positions."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco

    from AG3S.ag3s.experiments.sources.policy_record import load_run, provenance_arrays
    from AG3S.ag3s.experiments.studies import t48_hsr_core as hc
    from AG3S.ag3s.experiments.studies import t48b_scene as sc

    run = load_run(record)
    meta = run.meta
    model, xml = sc.load_record_model(meta)
    idx = sc.SceneIndex.from_model(model)
    data = mujoco.MjData(model)
    n = len(run)
    cols: dict[str, list] = {k: [] for k in ("state", "hand", "touching", "penetration", "fingers",
                                             "state_q", "hand_q", "target_pos", "crate_pos")}
    follows = []
    for i, step in enumerate(run):
        after = sc.observation_follows_step(meta, i)
        follows.append(after)
        q_ren = sc.render_qpos(model, step.qpos, step.qvel, after_step=after)
        for suffix, q in (("", q_ren), ("_q", step.qpos)):
            data.qpos[:] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            p = sc.classify(model, data, idx, target)
            cols["state" + suffix].append(p.state)
            cols["hand" + suffix].append(p.hand or "")
            if suffix == "":
                cols["touching"].append("|".join(p.touching))
                cols["penetration"].append(p.penetration)
                cols["fingers"].append([p.finger_contact["left"], p.finger_contact["right"]])
                cols["target_pos"].append(np.asarray(data.xpos[idx.fruit_body[target]], np.float64))
                cols["crate_pos"].append(np.asarray(data.xpos[idx.crate_body], np.float64))
    gs = grasp_state_sequence(cols["state"], [h or None for h in cols["hand"]], used_arm)
    gs_q = grasp_state_sequence(cols["state_q"], [h or None for h in cols["hand_q"]], used_arm)
    cl = [hc.closed_loop_gt(st.qpos, st.state, target) for st in run]
    return {
        "version": np.asarray(GRASP_VERSION),
        "t_step": np.asarray([st.t_step for st in run], np.int64),
        "target": np.asarray(target), "used_arm": np.asarray(used_arm),
        "prompt": np.asarray(run.prompt), "model_xml": np.asarray(str(xml)),
        "grasp_states": np.asarray(GRASP_STATES),
        "grasp_state": np.asarray(gs), "grasp_state_qpos": np.asarray(gs_q),
        "phys_state": np.asarray(cols["state"]), "phys_hand": np.asarray(cols["hand"]),
        "phys_state_qpos": np.asarray(cols["state_q"]), "phys_hand_qpos": np.asarray(cols["hand_q"]),
        "touching": np.asarray(cols["touching"]),
        "penetration_m": np.asarray(cols["penetration"], np.float64),
        "finger_contact_lr": np.asarray(cols["fingers"], np.int64).reshape(n, 2),
        "observation_follows_step": np.asarray(follows, bool),
        "target_pos_world": np.stack(cols["target_pos"]), "crate_pos_world": np.stack(cols["crate_pos"]),
        "used_gripper": np.asarray([st.state[hc.GRIPPER_INDEX[used_arm]] for st in run], np.float64),
        "closed_loop_gt": np.asarray(cl), "closed_loop_class": np.asarray([hc.GT_MAP[g] for g in cl],
                                                                          np.int64),
        **provenance_arrays(run, n),
    }


def grasp_summary(arr: dict) -> dict:
    gs = [str(x) for x in arr["grasp_state"]]
    first_held = next((i for i, x in enumerate(gs) if x == "held"), None)
    return {"n_frames": len(gs), "grasp_state": state_counts(gs),
            "grasp_state_qpos": state_counts([str(x) for x in arr["grasp_state_qpos"]]),
            "phys_state": state_counts([str(x) for x in arr["phys_state"]],
                                       ("on_table", "held", "grasping", "in_crate", "other")),
            "closed_loop_gt": state_counts([str(x) for x in arr["closed_loop_gt"]],
                                           ("reach", "held", "released")),
            "first_held_frame": first_held,
            "render_vs_qpos_disagree": int(sum(a != b for a, b in zip(gs, arr["grasp_state_qpos"]))),
            "penetration_max_m": float(np.max(arr["penetration_m"])) if len(gs) else None}


def stage_grasp(args) -> dict:
    keys = rc.resolve_keys(args.records)
    files = preflight(keys)
    per, skipped = {}, []
    for k in keys:
        p = rc.grasp_path(k)
        if p.is_file() and not args.force:
            skipped.append(k)
            with np.load(p, allow_pickle=False) as z:
                per[k] = {"cached": True, **grasp_summary({f: z[f] for f in z.files})}
            continue
        t0 = time.time()
        e = rc.episode_of(k)
        arr = grasp_predicates(files[k].record, target=e.target, used_arm=e.used_arm)
        save_npz_atomic(p, **arr)
        per[k] = {"seconds": time.time() - t0, **grasp_summary(arr)}
        print(f"[grasp] {k}: {per[k]['grasp_state']} ({per[k]['seconds']:.1f}s)", flush=True)
    payload = {**run_info("grasp", args), "records": per, "skipped_cached": skipped,
               "definition": {"held": "t48b_scene.classify: target touches both finger bodies of one hand "
                                      "(MJCF names) and neither table nor crate; render pose",
                              "grasp_state": "pre | held (used arm) | placed (in_crate) | post | "
                                             "held_other_hand (t53_extract.grasp_state_sequence)",
                              "closed_loop_gt": "t48_hsr_core.closed_loop_gt (SUBTASK-b)"},
               "provenance": rc.provenance(keys)}
    write_json(rc.out_dir("gt") / "grasp_summary.json", payload)
    return payload


# ------------------------------------------------------------------------------ check (CPU)


def array_format_problems(ex: dict, ref: dict, per_frame: Sequence[str]) -> list[str]:
    """Keys present in `ref` missing from `ex`, dtype kind / dtype, trailing shape of per-frame keys."""
    probs = []
    for k in ref:
        if k not in ex:
            probs.append(f"missing {k}")
            continue
        a, b = np.asarray(ex[k]), np.asarray(ref[k])
        if a.dtype.kind != b.dtype.kind or (a.dtype.kind not in "US" and a.dtype != b.dtype):
            probs.append(f"{k} dtype {a.dtype} != reference {b.dtype}")
        if k in per_frame and a.shape[1:] != b.shape[1:]:
            probs.append(f"{k} shape {a.shape} vs reference {b.shape}")
    return probs


def feat_problems(key: str, z: dict, t_rec: Sequence[int], layers: Sequence[int]) -> list[str]:
    from AG3S.ag3s.experiments.studies import t51_records as rc51

    probs = []
    n = len(t_rec)
    if [int(x) for x in z.get("t_step", [])] != list(t_rec):
        probs.append("t_step != step record")
    want = {"kv_l4": (n, 512), "noun_hidden_final": (n, 3, None), "noun_v": (n, 3, len(layers), None)}
    for k, shp in want.items():
        if k not in z:
            probs.append(f"missing {k}")
            continue
        got = np.asarray(z[k]).shape
        if len(got) != len(shp) or any(w is not None and g != w for g, w in zip(got, shp)):
            probs.append(f"{k} shape {got} != {shp}")
    st = json.loads(str(z["stamp"])) if "stamp" in z else {}
    if st.get("record") != key or st.get("noun") != rc.target_of(key):
        probs.append(f"stamp record/noun {st.get('record')!r}/{st.get('noun')!r}")
    if not str(st.get("checkpoint", "")).rstrip("/").endswith(rc51.CHECKPOINT_TAIL):
        probs.append(f"checkpoint {st.get('checkpoint')!r} is not step {rc51.CHECKPOINT_TAIL}")
    if "noun_ok" in z and not bool(np.all(z["noun_ok"])):
        probs.append(f"{int((~np.asarray(z['noun_ok'], bool)).sum())} frame(s) without noun feature")
    return probs


def _load(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def stage_check(args) -> dict:
    from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx
    from AG3S.ag3s.experiments.studies import t51b_noun_extract as ne

    keys = rc.resolve_keys(args.records)
    parts = list(args.parts)
    noun_ref = gt_ref = None
    notes = []
    if "noun" in parts:
        try:
            noun_ref = nx.load_extract(ne.t49_extract_path(ne.t49_dir(args), PARITY_KEY))
        except (SystemExit, OSError) as exc:
            notes.append(f"noun reference unavailable: {exc}")
    if "gt" in parts:
        gp = pathlib.Path(args.gt_reference or rc.storage_root() / GT_REFERENCE_REL)
        if gp.is_file():
            gt_ref = _load(gp)
        else:
            notes.append(f"GT reference {gp} missing")
    per, ok = {}, not notes
    for k in keys:
        r: dict[str, Any] = {}
        probs: list[str] = []
        try:
            info, pp = rc.pairing_report(k, require_chunks=args.require_chunks,
                                         server_roots=[pathlib.Path(p) for p in args.server_roots])
            r["pairing"] = info
            probs += [f"pairing: {p}" for p in pp]
            rec = rc.record_dir(k)
            t_rec = rc.record_t_steps(rec)
            prompt = str(rc.read_meta(rec)["prompt"])
        except SystemExit as exc:
            per[k] = {"problems": [str(exc)]}
            ok = False
            continue
        t_att = None
        if "attention" in parts:
            if not rc.npz_path(k).is_file():
                probs.append("attention: missing")
            else:
                with np.load(rc.npz_path(k), allow_pickle=False) as z:
                    t_att = [int(x) for x in z["t_step"]]
        if "noun" in parts:
            p = rc.noun_path(k)
            if not p.is_file():
                probs.append("noun: missing")
            else:
                ex = nx.load_extract(p)
                pr = noun_record_problems(k, ex, t_rec, t_att, prompt)
                if noun_ref is not None:
                    pr = ne.format_problems(ex, noun_ref) + pr
                probs += [f"noun: {x}" for x in pr]
                r["noun"] = {"n_frames": int(len(ex["t_step"])), "n_ok": int(np.asarray(ex["ok"]).sum())}
        if "feat" in parts:
            p = rc.feat_path(k)
            if not p.is_file():
                probs.append("feat: missing")
            else:
                z = _load(p)
                layers = [int(x) for x in z.get("noun_layers", [])]
                probs += [f"feat: {x}" for x in feat_problems(k, z, t_rec, layers)]
                r["feat"] = {"n_frames": int(len(z["t_step"])), "noun_layers": layers}
        if "gt" in parts:
            p = rc.gt_cache_path(k)
            if not p.is_file():
                probs.append("gt: missing")
            else:
                z = _load(p)
                pr = [] if gt_ref is None else array_format_problems(z, gt_ref, GT_PER_FRAME)
                if [int(x) for x in z["t_step"]] != list(t_rec):
                    pr.append("t_step != step record")
                if str(z["target"]) != rc.target_of(k) or str(z["record_key"]) != k:
                    pr.append(f"target/record_key {str(z['target'])}/{str(z['record_key'])}")
                probs += [f"gt: {x}" for x in pr]
                tpx = z["class_px"][:, :, 0]
                r["gt"] = {"n_frames": int(tpx.shape[0]),
                           "target_visible_ge200px_per_camera": [int((tpx[:, i] >= 200).sum())
                                                                 for i in range(tpx.shape[1])]}
        if "grasp" in parts:
            p = rc.grasp_path(k)
            if not p.is_file():
                probs.append("grasp: missing")
            else:
                z = _load(p)
                if [int(x) for x in z["t_step"]] != list(t_rec):
                    probs.append("grasp: t_step != step record")
                if str(z["target"]) != rc.target_of(k):
                    probs.append(f"grasp: target {str(z['target'])}")
                r["grasp"] = grasp_summary(z)
        r["problems"] = probs
        per[k] = r
        ok &= not probs
        print(f"[check] {k}: {len(probs)} problem(s)" + ("" if not probs else f" -> {probs[:2]}"))
    payload = {**run_info("check", args), "parts": parts, "notes": notes, "records": per, "all_ok": ok,
               "provenance": rc.provenance(keys)}
    write_json(rc.t53_dir() / "check.json", payload)
    if not ok:
        raise SystemExit(f"[check] problems — see {rc.t53_dir() / 'check.json'}")
    return payload


# ---------------------------------------------------------------------------------- main

CHECK_PARTS = ("attention", "noun", "feat", "gt", "grasp")


def build_parser() -> argparse.ArgumentParser:
    from AG3S.ag3s.experiments.studies import t51_records as rc51

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--run-dir", required=True,
                    help="the T53 run (results/development/<run>); outputs go to <run>/verify/T53/")
    ap.add_argument("--episodes-root", default=None,
                    help="where the episode directories are (default <run>/verify/T53)")
    ap.add_argument("--records", nargs="+", default=None, metavar="KEY",
                    help=f"subset of {list(rc.KEYS)} (default all 16)")
    ap.add_argument("--config", default=rc51.CONFIG)
    ap.add_argument("--checkpoint", default=None, help=f"default $AG3S_STORAGE_ROOT/{rc51.CHECKPOINT_REL}")
    ap.add_argument("--noun-layers", type=int, nargs="+", default=list(range(N_LAYERS)),
                    help="feat: layers of the noun-token V feature (default all 18)")
    ap.add_argument("--parity-record", default=PARITY_KEY)
    ap.add_argument("--parity-frames", type=int, default=PARITY_FRAMES)
    ap.add_argument("--t49-dir", default=None, help="T49 verify dir (noun_parity / check reference)")
    ap.add_argument("--subtask-b-feat", default=None,
                    help=f"feat_parity reference (default $AG3S_STORAGE_ROOT/{SUBTASK_B_EP1807})")
    ap.add_argument("--gt-reference", default=None,
                    help=f"check: T47-format GT cache to compare keys with (default {GT_REFERENCE_REL})")
    ap.add_argument("--dataset", default=None, help="layout: dataset root (default storage)")
    ap.add_argument("--server-roots", nargs="*", default=[],
                    help="layout / check: server record roots for chunk pairing when the chunks are "
                         "not beneath the episode dirs")
    ap.add_argument("--require-chunks", action="store_true",
                    help="layout / check: a missing chunk set is a problem")
    ap.add_argument("--parts", nargs="+", default=list(CHECK_PARTS), choices=CHECK_PARTS,
                    help="check: which outputs to check")
    ap.add_argument("--rgb-check-frames", type=int, default=3)
    ap.add_argument("--jax-platforms", default="cuda",
                    help="attention: JAX_PLATFORMS of the producer subprocess")
    ap.add_argument("--gpu-lock", default=GPU_LOCK)
    ap.add_argument("--no-lock-check", action="store_true")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="only when the lead confirmed the listed GPU processes are harmless")
    ap.add_argument("--force", action="store_true", help="redo finished files")
    ap.add_argument("--keep-frames", action="store_true", help="noun: keep the per-frame cache")
    return ap


def validate_args(ap: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    from AG3S.ag3s.experiments.studies import t51_records as rc51

    if args.parity_frames < 1:
        ap.error("--parity-frames >= 1")
    if args.rgb_check_frames < 0:
        ap.error("--rgb-check-frames >= 0")
    if args.config != rc51.CONFIG:
        ap.error(f"--config must be {rc51.CONFIG} (the config of the T47 / T49 / T48 assets)")
    layers = list(args.noun_layers)
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= x < N_LAYERS for x in layers):
        ap.error(f"--noun-layers: unique values in 0..{N_LAYERS - 1}")
    args.noun_layers = sorted(layers)
    if args.parity_record not in rc51.ALL_KEYS or rc51.is_extra(args.parity_record):
        ap.error("--parity-record must be a T47 record (stored references exist only for those)")
    for p in args.server_roots:
        if not pathlib.Path(p).is_dir():
            ap.error(f"--server-roots {p} is not a directory")
    if args.records:
        args.records = rc.resolve_keys(args.records)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    validate_args(ap, args)
    rc.configure(run_dir=args.run_dir, episodes_root=args.episodes_root)
    rc.t53_dir().mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    globals()[f"stage_{args.stage}"](args)
    print(f"[{STEP}] {args.stage} done in {time.time() - t0:.1f}s -> {rc.t53_dir()}")


__all__ = ["CHECK_PARTS", "GPU_STAGES", "GRASP_STATES", "NounTokenExtractor", "STAGES",
           "array_format_problems", "assemble_feat", "feat_problems", "feat_stamp", "feature_frame",
           "grasp_predicates", "grasp_state_sequence", "grasp_summary", "lock_is_held", "main",
           "noun_item", "noun_record_problems", "noun_token_impl", "state_counts"]


if __name__ == "__main__":
    main()
