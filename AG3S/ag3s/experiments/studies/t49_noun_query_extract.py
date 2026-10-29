"""T49 — prefix-only noun-row attention extraction (GPU) and its resumable per-frame cache.

**What is copied from where.** `NounQueryExtractor.__init__` / `_prefix_rows` are
`pi05_attention.AttentionSampler.__init__` + `_prefix` (`AG3S/ag3s/experiments/sources/
pi05_attention.py:79-105`): ``nnx.split`` once, ``jax.jit`` once, ``nnx.merge`` inside the trace,
``embed_prefix -> make_attn_mask -> positions = cumsum(mask) - 1 -> PaliGemma.llm([tokens, None])``.
The one change: ``_prefix`` keeps element ``[1]`` (the KV cache) and drops the attention
probabilities; here element ``[2]`` (``probs [L, B, K, G, T, S]``, `gemma.py:254,317,342,431-432`)
is kept and sliced **inside the jit** by `t49_noun_query_core.slice_probs` to ``[L, G, R, 768]`` plus
three ``[L, G, R]`` mass summaries — the ``[18, 1, 1, 8, 968, 968]`` tensor never reaches the host.

The observation is `pi05_attention.observation_from_step` (`:209-224`, the policy's own
``_input_transform``) followed by **one** `preprocess_observation` (as `AttentionSampler._sample`
`:184-185`), the model is `pi05_attention.load_policy` (`:59-68`, ``return_attn_probs=True``) —
what `attention_policy.load_attention_model` serves. No denoising, no noise seed: the prefix is
deterministic.

Noun positions are looked up in ``obs.tokenized_prompt`` by piece id
(`t49_noun_query_core.query_positions`); a frame whose noun is found 0 or >= 2 times is written
with ``ok = False`` and zero maps, never guessed.

Nothing touches jax / openpi at import time.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t49_noun_query_core as core

#: Per-frame arrays written to the cache and stacked by `assemble_frames` (name -> per-frame shape
#: builder from (Q, L, H)). ``maps`` is float16 on disk, like the action-query npz.
FRAME_FIELDS = ("maps", "image_mass", "text_mass", "bos", "self", "token_index", "prefix_row",
                "token_id")


# ------------------------------------------------------------------------- jitted pass


def prefix_rows_impl(model, obs, rows, *, make_attn_mask, xp):
    """The traced body: prefix pass of `model` on `obs`, query `rows` sliced by `slice_probs`.

    Split out of the class so it can be exercised with a stand-in model (unit test, jax CPU).
    """
    tokens, mask, ar_mask = model.embed_prefix(obs)
    n_image = tokens.shape[1] - obs.tokenized_prompt.shape[1]
    if n_image != core.N_IMAGE_TOKENS:  # static shapes: checked at trace time
        raise ValueError(f"prefix has {n_image} image tokens, expected {core.N_IMAGE_TOKENS}")
    attn = make_attn_mask(mask, ar_mask)
    positions = xp.cumsum(mask, axis=1) - 1
    out = model.PaliGemma.llm([tokens, None], mask=attn, positions=positions)
    if len(out) != 3:
        raise ValueError("PaliGemma.llm returned no attention probabilities — load the model with "
                         "return_attn_probs=True (pi05_attention.load_policy)")
    return core.slice_probs(out[2], rows, n_image=core.N_IMAGE_TOKENS, xp=xp)


class NounQueryExtractor:
    """One jitted prefix executable that returns the requested query rows' image attention."""

    def __init__(self, model):
        import flax.nnx as nnx
        import jax

        self._jax = jax
        self._nnx = nnx
        self._graphdef, self._state = nnx.split(model)
        self._jit = jax.jit(self._prefix_rows)
        self.t_call: list[float] = []

    def _prefix_rows(self, state, obs, rows):
        import jax.numpy as jnp

        from openpi.models.pi0 import make_attn_mask

        model = self._nnx.merge(self._graphdef, state)
        return prefix_rows_impl(model, obs, rows, make_attn_mask=make_attn_mask, xp=jnp)

    def rows(self, obs, rows: np.ndarray, *, timed: bool = True) -> dict[str, np.ndarray]:
        """`obs` straight from `observation_from_step` (preprocessed here, exactly once)."""
        import jax.numpy as jnp

        from openpi.models import model as _model

        obs = _model.preprocess_observation(None, obs, train=False)
        r = jnp.asarray(np.asarray(rows, np.int32))
        t0 = time.perf_counter()
        out = self._jax.block_until_ready(self._jit(self._state, obs, r))
        dt = time.perf_counter() - t0
        if timed:
            self.t_call.append(dt)
        return {k: np.asarray(v, np.float32) for k, v in out.items()}

    def memory(self) -> dict:
        try:
            stats = self._jax.devices()[0].memory_stats() or {}
        except Exception:  # noqa: BLE001 — not every backend reports
            stats = {}
        keep = ("bytes_in_use", "peak_bytes_in_use", "bytes_limit")
        return {k: int(stats[k]) for k in keep if k in stats}


def timing_summary(seconds: Sequence[float], note: str) -> dict:
    t = np.asarray(list(seconds), np.float64)
    if not t.size:
        return {"n": 0, "note": note}
    return {"n": int(t.size), "ms_median": float(np.median(t) * 1e3),
            "ms_p90": float(np.percentile(t, 90) * 1e3), "ms_mean": float(t.mean() * 1e3),
            "total_s": float(t.sum()), "note": note}


# --------------------------------------------------------------------------- tokenizer


def sentencepiece_of(config_name: str):
    """The SentencePiece processor the policy's `TokenizePrompt` uses (`tokenizer.py:14-20`)."""
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training import config as _config

    cfg = _config.get_config(config_name)
    return PaligemmaTokenizer(cfg.model.max_token_len)._tokenizer  # noqa: SLF001


def noun_ids_for(sp, nouns: Sequence[str]) -> dict[str, int]:
    return core.piece_ids(sp, sorted(set(nouns) | {core.DESTINATION_NOUN}))


def lookup(token_ids, mask, noun: str, ids: dict[str, int], bos_id: int):
    """`query_positions` for one variant: ``target`` = its noun, ``basket`` = the destination."""
    return core.query_positions(token_ids, mask,
                                {"target": ids[noun], "basket": ids[core.DESTINATION_NOUN]},
                                bos_id=bos_id)


def pieces_around(sp, token_ids, index: int, width: int = 2) -> list[str]:
    lo = max(0, int(index) - width)
    return [str(sp.id_to_piece(int(t))) for t in np.asarray(token_ids)[lo:int(index) + width + 1]]


# --------------------------------------------------------------------------- GPU guard


def gpu_compute_pids() -> Optional[list[int]]:
    """PIDs of compute processes on the GPUs (`nvidia-smi`), None when nvidia-smi is unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True,
            stderr=subprocess.DEVNULL, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return [int(x) for x in out.split() if x.strip().isdigit()]


# ------------------------------------------------------------------------ frame cache


def stamp_of(*, variant: str, key: str, prompt: str, noun: str, checkpoint: str, config: str,
             fingerprint: str, n_frames: int) -> dict:
    return {"variant": variant, "record": key, "prompt": prompt, "noun": noun,
            "checkpoint": str(checkpoint), "config": config, "record_fingerprint": fingerprint,
            "n_frames": int(n_frames), "queries": list(core.QUERY_NAMES)}


def frame_path(frames_dir: pathlib.Path, i: int) -> pathlib.Path:
    return pathlib.Path(frames_dir) / f"f{int(i):04d}.npz"


def write_frame(path: pathlib.Path, *, stamp: dict, t_step: int, ok: bool, error: str,
                fields: dict[str, np.ndarray], prefix_ms: float, n_text_valid: int) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, stamp=np.asarray(json.dumps(stamp, sort_keys=True)),
                        t_step=np.asarray(int(t_step)), ok=np.asarray(bool(ok)),
                        error=np.asarray(str(error)), prefix_ms=np.asarray(float(prefix_ms)),
                        n_text_valid=np.asarray(int(n_text_valid)),
                        **{k: fields[k] for k in FRAME_FIELDS})
    tmp.replace(path)  # atomic: a crash never leaves a half-written frame that looks complete


def read_frame(path: pathlib.Path, stamp: dict) -> dict:
    with np.load(path, allow_pickle=False) as d:
        got = json.loads(str(d["stamp"]))
        if got != json.loads(json.dumps(stamp, sort_keys=True)):
            raise SystemExit(f"{path}: cached frame was made for {got}, not {stamp} — "
                             "delete the frame directory or pass --force")
        return {k: d[k] for k in d.files}


def empty_fields(n_layers: int, n_heads: int) -> dict[str, np.ndarray]:
    q, g = len(core.QUERY_NAMES), 16
    z = np.zeros((q, n_layers, n_heads), np.float32)
    return {"maps": np.zeros((q, n_layers, n_heads, 3, g, g), np.float16),
            "image_mass": z.copy(), "text_mass": z.copy(), "bos": z.copy(), "self": z.copy(),
            "token_index": np.full(q, -1, np.int64), "prefix_row": np.full(q, -1, np.int64),
            "token_id": np.full(q, -1, np.int64)}


def fields_from_rows(out: dict[str, np.ndarray], positions: dict[str, int],
                     token_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Device output of one pass -> the per-frame cache fields (queries first)."""
    maps = core.rows_to_maps(out["image"])  # (Q, L, H, C, 16, 16)
    tidx = np.asarray([positions[q] for q in core.QUERY_NAMES], np.int64)
    return {"maps": maps.astype(np.float16),
            "image_mass": np.moveaxis(out["image_mass"], 2, 0).astype(np.float32),
            "text_mass": np.moveaxis(out["text_mass"], 2, 0).astype(np.float32),
            "bos": np.moveaxis(out["bos"], 2, 0).astype(np.float32),
            "self": np.moveaxis(out["self"], 2, 0).astype(np.float32),
            "token_index": tidx, "prefix_row": tidx + core.N_IMAGE_TOKENS,
            "token_id": np.asarray(token_ids)[tidx].astype(np.int64)}


def assemble_frames(frames: Sequence[dict], stamp: dict) -> dict[str, np.ndarray]:
    """Per-frame dicts (`read_frame`) -> one record npz payload (frame axis first)."""
    if len(frames) != int(stamp["n_frames"]):
        raise ValueError(f"{len(frames)} frames, stamp says {stamp['n_frames']}")
    out = {k: np.stack([np.asarray(f[k]) for f in frames]) for k in FRAME_FIELDS}
    out.update({
        "t_step": np.asarray([int(f["t_step"]) for f in frames], np.int64),
        "ok": np.asarray([bool(f["ok"]) for f in frames]),
        "error": np.asarray([str(f["error"]) for f in frames]),
        "prefix_ms": np.asarray([float(f["prefix_ms"]) for f in frames], np.float64),
        "n_text_valid": np.asarray([int(f["n_text_valid"]) for f in frames], np.int64),
        "query_names": np.asarray(core.QUERY_NAMES),
        "cameras": np.asarray(("cam_high", "cam_left_wrist", "cam_right_wrist")),
        "stamp": np.asarray(json.dumps(stamp, sort_keys=True)),
        **{k: np.asarray(v if v is not None else "") for k, v in stamp.items()
           if k in ("variant", "record", "prompt", "noun", "checkpoint", "config",
                    "record_fingerprint")},
    })
    return out


def extract_record(*, frames_dir: pathlib.Path, final_path: pathlib.Path, stamp: dict,
                   steps: Sequence[Any], run_frame: Callable[[Any], dict],
                   log: Callable[[str], None] = print, keep_frames: bool = False) -> dict:
    """Run (or resume) one (variant, record): cached frames are read back, missing ones computed.

    `run_frame(step)` returns ``{"ok", "error", "fields", "prefix_ms", "n_text_valid"}``. When every
    frame is present the record npz is written (atomically) and the frame directory removed
    (unless `keep_frames`). Returns counts.
    """
    frames_dir = pathlib.Path(frames_dir)
    done, computed = [], 0
    t0 = time.time()
    for i, step in enumerate(steps):
        fp = frame_path(frames_dir, i)
        if fp.exists():
            done.append(read_frame(fp, stamp))
            continue
        res = run_frame(step)
        write_frame(fp, stamp=stamp, t_step=int(step.t_step), ok=res["ok"], error=res["error"],
                    fields=res["fields"], prefix_ms=res["prefix_ms"],
                    n_text_valid=res["n_text_valid"])
        done.append(read_frame(fp, stamp))
        computed += 1
        if i % 10 == 0 or i == len(steps) - 1:
            log(f"  [{stamp['variant']}/{stamp['record']}] frame {i + 1}/{len(steps)} "
                f"({time.time() - t0:.1f}s)")
    payload = assemble_frames(done, stamp)
    final_path = pathlib.Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_name(final_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(final_path)
    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return {"variant": stamp["variant"], "record": stamp["record"], "n_frames": len(done),
            "computed": computed, "resumed": len(done) - computed,
            "n_errors": int((~payload["ok"]).sum()),
            "errors": sorted(set(str(e) for e in payload["error"] if str(e))),
            "path": str(final_path)}


def load_extract(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        out = {k: d[k] for k in d.files}
    out["stamp_dict"] = json.loads(str(out["stamp"]))
    return out


def query_block(ex: dict, query: str) -> np.ndarray:
    """``(F, L, H, C, 16, 16)`` float64 of one query (failed frames are zeros; mask with ``ok``)."""
    qi = [str(q) for q in ex["query_names"]].index(query)
    return np.asarray(ex["maps"][:, qi], np.float64)


__all__ = ["FRAME_FIELDS", "NounQueryExtractor", "assemble_frames", "empty_fields",
           "extract_record", "fields_from_rows", "frame_path", "gpu_compute_pids",
           "load_extract", "lookup", "noun_ids_for", "pieces_around", "prefix_rows_impl",
           "query_block", "read_frame", "sentencepiece_of", "stamp_of", "timing_summary",
           "write_frame"]
