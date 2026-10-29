"""T51 extension — attention npz of the recorded VLA-training runs ep0 / ep500 / ep1200 (GPU; A1 does not run it).

`T51.task.md` 판정 보완 (2026-10-12): the calibration gains three recorded rollouts of training episodes;
their attention must be **exactly** the T47 ``attention_16d_*.npz`` format (same axes, denoise 0/4/9,
agg mean/first/last, noise seed 0, float16, same checkpoint step). So this does not re-implement the
sampler: it runs the unchanged producer `AG3S.ag3s.experiments.sources.pi05_attention` (``main``,
`pi05_attention.py:227-301`: `load_run` -> `load_policy` (`:59-68`, ``return_attn_probs=True``) ->
`AttentionSampler.attention` (`:154-167` -> `_sample` `:178-206`, Gaussian start from
``jax.random.key(seed)``) -> `observation_from_step` (`:209-224`) -> camera slices (`:274-282`) ->
``savez_compressed`` (`:286-299`)) once per record, as a subprocess, with the arguments of the original
extraction (`experiments/legacy/outputs/verify/attn16d/extract.sh`: ``--config
pi05_rby1_randomized_pick_place_16d_lora --noise-seeds 0``, everything else at the producer defaults —
pinned explicitly here so a default change upstream cannot drift the format).

    source scripts/setup/env.sh; export AG3S_RUN_DIR=<the T51 run>
    M=AG3S.ag3s.experiments.studies.t51_extract
    # GPU, under the shared lock, nothing else on the GPU (the stage refuses otherwise):
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m $M --stage parity
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m $M --stage extract
    # CPU, any venv: format + pairing of what was written
    .venv-ag3s/bin/python -m $M --stage check

Stages:

``parity``   re-extract the first ``--parity-frames`` (4) frames of a T47 record (``--parity-record``,
             default ``ep1807``) with the storage checkpoint into ``attention/parity_<key>_first<N>.npz``
             and compare with the stored asset (`compare_attention`). Answers "does this environment +
             the storage checkpoint reproduce the T47 assets" before the new runs are trusted.
``extract``  ``attention/attention_16d_<key>.npz`` for ``ep0`` · ``ep500`` · ``ep1200`` (resumable: a
             file that passes `t51_records.check_pairing` is skipped). Written to ``*.partial.npz`` and
             renamed only after the check. Logs per record in ``attention/extract_<key>.log``.
``check``    CPU: `t51_records.check_pairing` (format = `ATTENTION_FORMAT`, prompt, frames, t_step,
             resolved record path, episode) of every extracted file -> ``attention/check.json``.

Output: ``$AG3S_RUN_DIR/verify/T51/attention/`` (``--extra-dir`` overrides the T51 dir). Never written
into storage assets. The parent process imports neither jax nor openpi; the producer subprocess runs
with ``JAX_PLATFORMS=cuda`` so a missing GPU fails instead of silently running on CPU.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t47_head_selection as hs
from AG3S.ag3s.experiments.studies import t51_records as rc

STEP = "T51"
PRODUCER = "AG3S.ag3s.experiments.sources.pi05_attention"
#: The producer's defaults (`pi05_attention.py:233-245`) as used for every T47 asset, pinned.
PRODUCER_ARGS = ("--num-denoise", "10", "--denoise-steps", "0", "4", "9",
                 "--agg", "mean", "first", "last", "--store-dtype", "float16", "--noise-seeds", "0")
GPU_LOCK = "/tmp/ag3s-vla-gpu-validation.lock"


def out_dir() -> pathlib.Path:
    d = rc.extra_dir() / "attention"
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_path(args) -> pathlib.Path:
    p = pathlib.Path(args.checkpoint) if args.checkpoint else hs.storage_root() / rc.CHECKPOINT_REL
    if not (p / "params").is_dir():
        raise SystemExit(f"checkpoint {p} has no params/")
    if not str(p).rstrip("/").endswith(rc.CHECKPOINT_TAIL):
        raise SystemExit(f"checkpoint {p} is not step {rc.CHECKPOINT_TAIL} (the T47 assets' step)")
    return p


def producer_argv(record: pathlib.Path, checkpoint: pathlib.Path, out: pathlib.Path, *,
                  python: str = sys.executable, limit: Optional[int] = None) -> list[str]:
    """The producer command line: `extract.sh`'s arguments + the pinned defaults."""
    argv = [python, "-m", PRODUCER, "--records", str(record), "--checkpoint", str(checkpoint),
            "--config", rc.CONFIG, "--out", str(out), *PRODUCER_ARGS]
    if limit is not None:
        argv += ["--limit", str(int(limit))]
    return argv


def producer_env(platforms: str = "cuda") -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    env["PYTHONUNBUFFERED"] = "1"
    if platforms:
        env["JAX_PLATFORMS"] = platforms
    return env


def gpu_guard(args) -> list[int]:
    from AG3S.ag3s.experiments.studies.t49_noun_query_extract import gpu_compute_pids

    pids = gpu_compute_pids()
    if pids is None:
        raise SystemExit("nvidia-smi is unavailable — the GPU stages need the GPU")
    if pids and not args.allow_busy_gpu:
        raise SystemExit(f"GPU has compute processes {pids} — wait for them (never kill another "
                         f"session's process); run under flock {GPU_LOCK}")
    return pids


def run_producer(argv: Sequence[str], log_path: pathlib.Path, env: dict) -> dict:
    t0 = time.time()
    with log_path.open("w") as log:
        log.write("$ " + " ".join(argv) + "\n")
        log.flush()
        rc_ = subprocess.call(list(argv), stdout=log, stderr=subprocess.STDOUT, env=env)
    return {"returncode": int(rc_), "seconds": time.time() - t0, "log": str(log_path)}


def run_info(stage: str, args) -> dict:
    return {"step": STEP, "sub": "extract", "stage": stage, "argv": sys.argv,
            "args": {k: v for k, v in vars(args).items()},
            "started_kst": _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(),
            "git": hs.git_state(), "storage_root": str(hs.storage_root()),
            "extra_dir": str(rc.extra_dir()), "python": sys.executable,
            "python_version": sys.version.split()[0]}


def write_json(path: pathlib.Path, payload: Any) -> None:
    from AG3S.ag3s.experiments.studies import t47b_metrics as bm

    hs.write_json(path, bm.jsonable(payload))


# --------------------------------------------------------------------------- comparison


def compare_attention(new: np.ndarray, ref: np.ndarray) -> dict:
    """Two ``(F, D, A, L, H, C, 16, 16)`` attention arrays of the same frames.

    Reports the absolute difference (max / mean, and relative to the mean map mass), whether every
    16x16 map has the same argmax patch (share over all (frame, d, a, layer, head, camera) maps), and
    the Pearson r of the serving cell's maps (d0 · last · L8H2, all frames and cameras). float16
    storage and GPU reduction order make bit equality not expected; the numbers are reported, the
    verdict is the verifier's.
    """
    a = np.asarray(new, np.float32)
    b = np.asarray(ref, np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape {a.shape} vs {b.shape}")
    d = np.abs(a - b)
    fa = a.reshape(*a.shape[:-2], -1)
    fb = b.reshape(*b.shape[:-2], -1)
    same_arg = fa.argmax(axis=-1) == fb.argmax(axis=-1)
    mass = float(fb.sum(axis=-1).mean())
    sa = a[:, 0, 2, 8, 2].reshape(-1)  # d0 (index 0), agg last (index 2), L8, H2
    sb = b[:, 0, 2, 8, 2].reshape(-1)
    r = float(np.corrcoef(sa, sb)[0, 1]) if sa.std() > 0 and sb.std() > 0 else float("nan")
    return {"shape": list(a.shape), "bitwise_equal": bool(np.array_equal(a, b)),
            "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "mean_map_mass_ref": mass,
            "max_abs_over_mean_mass": float(d.max() / mass) if mass > 0 else float("nan"),
            "argmax_patch_same_share": float(same_arg.mean()),
            "argmax_patch_same_share_d0": float(same_arg[:, 0].mean()),
            "serving_cell_pearson_r": r}


# ------------------------------------------------------------------------------ stages


def stage_parity(args) -> dict:
    gpu_guard(args)
    key = args.parity_record
    if rc.is_extra(key):
        raise SystemExit("--parity-record must be a T47 record (a stored asset to compare with)")
    info = rc.check_pairing(key)
    ckpt = checkpoint_path(args)
    n = int(args.parity_frames)
    dest = out_dir() / f"parity_{key}_first{n}.npz"
    argv = producer_argv(rc.record_path(key), ckpt, dest, limit=n)
    res = run_producer(argv, out_dir() / f"parity_{key}.log", producer_env(args.jax_platforms))
    if res["returncode"] != 0 or not dest.is_file():
        raise SystemExit(f"producer failed ({res}); see {res['log']}")
    with np.load(dest, allow_pickle=False) as new, np.load(rc.npz_path(key), allow_pickle=False) as ref:
        cmp = compare_attention(new["attention"], ref["attention"][:n])
        t_same = bool(np.array_equal(new["t_step"], ref["t_step"][:n]))
        fields = {k: new[k] for k in new.files if k != "attention"}
        shape, dtype = rc.npz_array_header(dest, "attention")
    payload = {**run_info("parity", args), "record": info, "checkpoint": str(ckpt),
               "argv_producer": argv, "producer": res, "t_step_same": t_same,
               "format_problems": rc.format_problems(fields, shape, dtype),
               "comparison": cmp, "output": str(dest)}
    write_json(out_dir() / f"parity_{key}.json", payload)
    print(f"[parity] {key} first {n}: argmax same {cmp['argmax_patch_same_share']:.4f}, max |d| "
          f"{cmp['max_abs']:.2e}, L8H2 r {cmp['serving_cell_pearson_r']:.5f}")
    return payload


def _passes(key: str) -> Optional[dict]:
    try:
        return rc.check_pairing(key)
    except SystemExit:
        return None


def stage_extract(args) -> dict:
    gpu_guard(args)
    ckpt = checkpoint_path(args)
    keys = rc.resolve_keys(args.records, rc.EXTRA_KEYS)
    if [k for k in keys if not rc.is_extra(k)]:
        raise SystemExit(f"extract writes only the extra runs {rc.EXTRA_KEYS}; T47 assets are read-only")
    env = producer_env(args.jax_platforms)
    done, skipped = [], []
    for k in keys:
        final = rc.npz_path(k)
        if final.is_file() and not args.force:
            info = _passes(k)
            if info is not None:
                skipped.append(info)
                print(f"[extract] {k}: exists and pairs, skipping ({final})")
                continue
            raise SystemExit(f"[{k}] {final} exists but does not pair with its record — inspect it "
                             "(never overwritten silently); --force to re-extract")
        part = final.with_name(final.stem + ".partial.npz")
        argv = producer_argv(rc.record_path(k), ckpt, part)
        print(f"[extract] {k}: {' '.join(argv[1:4])} ... -> {part}", flush=True)
        res = run_producer(argv, out_dir() / f"extract_{k}.log", env)
        if res["returncode"] != 0 or not part.is_file():
            raise SystemExit(f"[{k}] producer failed ({res}); see {res['log']}")
        part.replace(final)
        info = rc.check_pairing(k)  # raises (and leaves the file for inspection) on any mismatch
        done.append({**info, **res, "bytes": final.stat().st_size})
        print(f"[extract] {k}: {info['n_frames']} frames in {res['seconds']:.0f}s -> {final}", flush=True)
    payload = {**run_info("extract", args), "checkpoint": str(ckpt), "producer": PRODUCER,
               "producer_args": list(PRODUCER_ARGS), "config": rc.CONFIG,
               "extracted": done, "skipped": skipped, "provenance": rc.provenance(keys)}
    write_json(out_dir() / "extract.json", payload)
    return payload


def stage_check(args) -> dict:
    keys = rc.resolve_keys(args.records, rc.EXTRA_KEYS)
    out, ok = {}, True
    for k in keys:
        try:
            out[k] = {"ok": True, **rc.check_pairing(k)}
        except SystemExit as exc:
            ok = False
            out[k] = {"ok": False, "error": str(exc), "npz": str(rc.npz_path(k))}
        print(f"[check] {k}: {'ok' if out[k]['ok'] else out[k]['error']}")
    payload = {**run_info("check", args), "all_ok": ok, "records": out,
               "format": {k: list(v) if isinstance(v, tuple) else v
                          for k, v in rc.ATTENTION_FORMAT.items()},
               "provenance": rc.provenance(keys)}
    write_json(out_dir() / "check.json", payload)
    if not ok:
        raise SystemExit("some extra-run npz do not pair — see check.json")
    return payload


STAGES = ("parity", "extract", "check")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--records", nargs="+", default=None, metavar="KEY",
                    help=f"extract / check: subset of {list(rc.EXTRA_KEYS)}")
    ap.add_argument("--extra-dir", default=None,
                    help="T51 dir to write attention/ under (default $AG3S_RUN_DIR/verify/T51)")
    ap.add_argument("--checkpoint", default=None,
                    help=f"default $AG3S_STORAGE_ROOT/{rc.CHECKPOINT_REL}")
    ap.add_argument("--parity-record", default="ep1807")
    ap.add_argument("--parity-frames", type=int, default=4)
    ap.add_argument("--jax-platforms", default="cuda",
                    help="JAX_PLATFORMS of the producer subprocess ('' = jax default)")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="only when the lead confirmed the listed GPU processes are harmless")
    ap.add_argument("--force", action="store_true", help="extract: re-extract existing files")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.parity_frames < 1:
        ap.error("--parity-frames >= 1")
    if not args.extra_dir and not os.environ.get("AG3S_RUN_DIR"):
        raise SystemExit("AG3S_RUN_DIR is not set (export the T51 run) and no --extra-dir")
    rc.configure(args.extra_dir)
    t0 = time.time()
    globals()[f"stage_{args.stage}"](args)
    print(f"[{STEP} extract] {args.stage} done in {time.time() - t0:.1f}s -> {rc.extra_dir()}")


if __name__ == "__main__":
    main()
