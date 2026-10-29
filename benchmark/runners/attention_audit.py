"""SL3 offline attention audit: V11AB's attention read-out on recorded pi05_libero requests (.venv-openpi-live, GPU).

For every policy call of one or more **source runs** (``launch_libero_pi05_only.py`` runs with
``policy_records``; geometry / segmentation optional) this runs V11AB's attention path -- a second
``return_attn_probs=True`` copy of the checkpoint driven by `AttentionSampler`, denoise step 0,
``AGGREGATIONS["last"]``, cell ``[layer 8, head 2]`` (`AG3S/trajopt/attention_policy.py`) -- through
:class:`AG3S.adapters.panda.attention.PandaAttentionReader`, and writes the result to a **new run**
(``common.run_artifacts.create_run``). Source runs are only read; their run id, path, manifest and
``episodes.jsonl`` SHA-256 and each policy record's SHA-256 are recorded.

Per call it stores the V11AB cell over the 768 image tokens (+ 16x16 grids for ``head`` /
``left_wrist`` in policy and OpenCV orientation), the image-block mass of the cell and of every
``[layer, head]``, the whole pooled ``[L, H, 768]`` array (float16, diagnosis only), the ``kv_L4``
subtask feature of the same prefix pass (what the V11AB server computes with the probe on; SL-6 input),
and the sampler wall time. Each call is joined to its episode: ``policy_records/step_<n>.npy`` is the
n-th request the server received; the episodes of ``episodes.jsonl`` (in order) own consecutive blocks
of ``policy_calls`` requests, and the join is **checked**: recorded state == episode ``policy_state[k]``,
prompt == ``task_description``, and both images == the client preprocessing of the episode's raw
observation at ``obs_index = chunk_t[k]`` (bit-exact). A mismatch stops the run.

Run layout::

    <run>/attention/<suite>_t<task>_e<episode>_attn.npz   per-episode arrays (keys: see ``EPISODE_KEYS``)
    <run>/attention_index.jsonl                           one row per episode: source paths, joins, target
    <run>/token_layout.json                               read from the model input + check result
    <run>/summary.json                                    counts, latency, GPU memory, V11AB equivalence

SL6 B5 -- ``--noun-rows`` (default off; without it the run is the SL3 / SL4b audit, unchanged): instead of the V11AB
action cell, every request's **target-noun query row** (`AG3S.trajopt.attention_source.noun_lookup` with
``NounTokens.from_config(config, rule="libero")``, SL6 B1: last piece of the head word of the target phrase) is read
from one prefix pass (`t49_noun_query_extract.NounQueryExtractor` = `prefix_rows_impl` + `slice_probs`, the slice the
served ``noun_target:L<l>H<h>`` source takes, `attention_source.prefix_noun_impl`): image attention of layers
``--noun-layers`` (default 2-17) x 8 heads x 768 image tokens (all three blocks; the right-wrist block is padding).
Writes ``<run>/noun_rows/<stem>_noun.npz`` (keys ``NOUN_EPISODE_KEYS``) and ``noun_index.jsonl`` (same join checks,
BDDL target + segmentation instances); `AG3S/ag3s/experiments/studies/sl6_head_select.py` (B6) reads them::

    .venv-openpi-live/bin/python benchmark/runners/attention_audit.py --noun-rows \
        --source-run 20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4 --run-id sl6b-noun-rows-dev-e1-4

Example (verifier; GPU)::

    source scripts/setup/env.sh
    .venv-openpi-live/bin/python benchmark/runners/attention_audit.py \\
        --source-run <dev rollout run> --run-id sl3-attention-audit-dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from common.workspace import PROJECT_ROOT, shared_storage_root

STEP = "SL3"
CONDITION = "attention audit (offline): V11AB attention read-out on recorded pi05_libero policy requests"
POLICY_CONFIG = "pi05_libero"
REQUEST_KEYS = ("observation/image", "observation/wrist_image", "observation/state", "prompt")
CAMERAS = ("head", "left_wrist")
EPISODE_KEYS = ("cell_tokens", "cell_grid_policy", "cell_grid_opencv", "cameras", "block_mass_cell", "block_mass_all",
                "pooled_f16", "kv_feature", "call_k", "global_step", "chunk_t", "obs_index", "record_sha256", "seconds")
# --- SL6 B5: --noun-rows ---------------------------------------------------------------------------------------
NOUN_STEP = "SL6-B5"
NOUN_CONDITION = ("noun-row attention audit (offline): target-noun prefix row -> image tokens, layers x heads, on "
                  "recorded pi05_libero policy requests (SL6 B5, head re-selection input)")
#: Layers of the noun-query candidate set (RB-Y1 T51b: layers 2-17 x heads 0-7 = 128 cells; SL6 task §4).
NOUN_LAYERS = tuple(range(2, 18))
NOUN_EPISODE_KEYS = ("noun_image", "noun_mass", "layers", "rows", "token_index", "noun_ok", "call_k", "global_step",
                     "chunk_t", "obs_index", "record_sha256", "seconds")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- source runs ---------------------------------------------------------------------------------------------
def resolve_source_run(value: str) -> Path:
    """A run directory below results/benchmark (absolute path or bare run name); read only."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and len(candidate.parts) == 1:
        candidate = shared_storage_root() / "results/benchmark" / candidate
    candidate = candidate.resolve()
    base = (shared_storage_root() / "results/benchmark").resolve()
    if not candidate.is_relative_to(base) or candidate == base or not candidate.is_dir():
        raise ValueError(f"source run must be a directory below {base}: {candidate}")
    return candidate


def load_policy_record(path) -> dict:
    """``PolicyRecorder`` file: flat ``{"inputs/observation/image": ..., "outputs/actions": ...}``."""
    return np.load(str(path), allow_pickle=True).item()


def request_of(record: dict) -> dict:
    """The websocket request the server received (``LiberoInputs`` keys)."""
    return {key: record["inputs/" + key] for key in REQUEST_KEYS}


def episode_stem(suite: str, task_id: int, episode_idx: int) -> str:
    return f"{suite}_t{int(task_id)}_e{int(episode_idx)}"


def index_source_run(run_dir: Path) -> dict:
    """Episodes of ``run_dir`` with their policy records (no record is opened here)."""
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines() if line.strip()]
    records = sorted((run_dir / "policy_records").glob("step_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    if [int(p.stem.split("_")[1]) for p in records] != list(range(len(records))):
        raise ValueError(f"{run_dir}/policy_records is not step_0..step_{len(records) - 1}")
    episodes, offset = [], 0
    for row in rows:
        n = int(row["policy_calls"])
        stem = episode_stem(row["suite"], row["task_id"], row["episode_idx"])
        geom = run_dir / "episodes" / f"{stem}_geom.npz"
        episodes.append({"suite": row["suite"], "task_id": int(row["task_id"]), "episode_idx": int(row["episode_idx"]),
                         "init_state_index": row.get("init_state_index"), "task_description": row["task_description"],
                         "status": row.get("status"), "official_success": row.get("official_success"),
                         "policy_calls": n, "npz": row["npz"], "geom_npz": str(geom.relative_to(run_dir)) if geom.exists() else None,
                         "global_steps": list(range(offset, offset + n))})
        offset += n
    if offset != len(records):
        raise ValueError(f"{run_dir}: episodes.jsonl has {offset} policy calls, policy_records has {len(records)}")
    settings = manifest.get("settings", {})
    return {"run_dir": str(run_dir), "run_id": run_dir.name, "commit": manifest.get("commit"),
            "branch": manifest.get("branch"), "dirty": manifest.get("dirty"), "condition": settings.get("condition"),
            "step": settings.get("step"), "policy_config": settings.get("policy_config"),
            "checkpoint": (settings.get("checkpoint") or {}).get("path"),
            "checkpoint_aggregate_sha256": (settings.get("checkpoint") or {}).get("aggregate_sha256"),
            "geometry_capture": {k: (settings.get("geometry_capture") or {}).get(k) for k in
                                 ("capture_geometry", "capture_segmentation", "capture_float_depth")},
            "manifest_sha256": sha256_file(manifest_path),
            "episodes_jsonl_sha256": sha256_file(run_dir / "episodes.jsonl"),
            "policy_records": len(records), "episodes": episodes}


def client_images(raw_agentview: np.ndarray, raw_wrist: np.ndarray, resize: int):
    """The client's policy images from raw robosuite observations (libero_pi05_client.run_episode)."""
    from openpi_client import image_tools

    out = []
    for raw in (raw_agentview, raw_wrist):
        img = np.ascontiguousarray(raw[::-1, ::-1])
        out.append(image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize)))
    return out


def check_join(record: dict, episode_npz, k: int, task_description: str, resize: int) -> dict:
    """Bit-exact join of policy record ``k`` of an episode to the episode npz (raises on mismatch)."""
    t = int(episode_npz["chunk_t"][k])
    state_ok = bool(np.array_equal(np.asarray(record["inputs/observation/state"], np.float64),
                                   np.asarray(episode_npz["policy_state"][k], np.float64)))
    prompt_ok = str(record["inputs/prompt"]) == str(task_description)
    img, wrist = client_images(episode_npz["obs_agentview_image"][t], episode_npz["obs_robot0_eye_in_hand_image"][t],
                               resize)
    image_ok = bool(np.array_equal(img, record["inputs/observation/image"]))
    wrist_ok = bool(np.array_equal(wrist, record["inputs/observation/wrist_image"]))
    out = {"chunk_t": t, "obs_index": t, "state": state_ok, "prompt": prompt_ok, "image": image_ok, "wrist_image": wrist_ok}
    if not (state_ok and prompt_ok and image_ok and wrist_ok):
        raise ValueError(f"policy record does not join episode call {k}: {out}")
    return out


def episode_target(episode: dict, run_dir: Path) -> dict:
    """BDDL target + segmentation instances (geom ids) when the source run has a geometry npz."""
    from benchmark.adapters.libero_panda import task_target as tt

    target = tt.task_target(episode["suite"], episode["task_id"])
    if episode.get("geom_npz"):
        with np.load(str(run_dir / episode["geom_npz"])) as geom:
            target["instances_geom_ids"] = tt.scene_instances(geom, target)
            target["segmentation_cameras"] = [c for c in CAMERAS if f"segmentation_geom/{c}" in geom.files]
    return target


# --- run settings --------------------------------------------------------------------------------------------
def build_settings(args, sources: list, checkpoint: dict, *, probe: bool = True) -> dict:
    from AG3S.adapters.panda import attention as pa
    from benchmark.runners import launch_libero_pi05_only as launcher

    return {
        "step": STEP, "condition": CONDITION, "policy_config": args.config,
        "ag3s": {"enabled": False, "path": "offline: AG3S.adapters.panda.attention.PandaAttentionReader -> V11AB "
                                           "AttentionSampler (no serving, no safety layer)"},
        "attention_sources": [{k: v for k, v in s.items() if k != "episodes"}
                              | {"episodes": [{k: e[k] for k in ("suite", "task_id", "episode_idx", "policy_calls",
                                                                    "status", "npz", "geom_npz")} for e in s["episodes"]]}
                              for s in sources],
        "checkpoint": checkpoint,
        "v11ab_cell": pa.v11ab_cell(),
        "token_blocks": [{"model_key": b.model_key, "tokens": list(b.tokens), "wire_camera": b.wire_camera,
                          "robosuite_camera": b.robosuite_camera, "image_mask": b.image_mask,
                          "v11ab_policy_camera": b.v11ab_policy_camera} for b in pa.PI05_LIBERO_TOKEN_BLOCKS],
        "mapping": pa.mapping_conventions(),
        "attention": {"num_denoise_steps": args.num_denoise, "noise_seed": args.noise_seed,
                      "subtask_feature": not args.no_feature, "full_grid": not args.no_full_grid,
                      "full_grid_dtype": "float16", "check_v11ab_calls": args.check_v11ab,
                      "limit_calls": args.limit_calls, "task_ids": args.task_ids},
        "dev_split": "LIBERO init state index 0-4 reserved for dev (lead proposal, SL3 task); this run reads the "
                     "source runs' episodes as they are",
        "privileged_inputs": "none for attention (recorded policy requests only); segmentation / BDDL target only in "
                             "attention_index.jsonl for scoring",
        "interpreters": {"audit": sys.executable,
                         "openpi_live": (launcher.interpreter_info(Path(sys.executable), launcher.OPENPI_PACKAGES)
                                         if probe else sys.executable)},
        "environment": {key: os.environ.get(key) for key in
                        ("XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "OPENPI_DATA_HOME", "AG3S_STORAGE_ROOT",
                         "CUDA_VISIBLE_DEVICES", "JAX_COMPILATION_CACHE_DIR")},
        "commands": {"audit": [sys.executable, *sys.argv]},
    }


# --- GPU part ------------------------------------------------------------------------------------------------
def device_memory() -> dict:
    import jax

    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 -- not every backend reports
        stats = {}
    return {k: int(stats[k]) for k in ("bytes_in_use", "peak_bytes_in_use", "bytes_limit") if k in stats}


def percentile_summary(values) -> dict:
    v = np.asarray(values, np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "p50_ms": float(np.percentile(v, 50) * 1e3), "p95_ms": float(np.percentile(v, 95) * 1e3),
            "mean_ms": float(v.mean() * 1e3), "max_ms": float(v.max() * 1e3)}


def audit_episode(reader, source: dict, episode: dict, out_dir: Path, args, *, check_left: list) -> dict:
    from AG3S.adapters.panda import attention as pa

    run_dir = Path(source["run_dir"])
    resize = 224
    calls = episode["global_steps"][: args.limit_calls] if args.limit_calls else episode["global_steps"]
    rows = {key: [] for key in EPISODE_KEYS if key != "cameras"}
    joins, v11ab = [], []
    with np.load(str(run_dir / episode["npz"])) as ep:
        ep = {k: ep[k] for k in ("chunk_t", "policy_state", "obs_agentview_image", "obs_robot0_eye_in_hand_image")}
        for k, step in enumerate(calls):
            path = run_dir / "policy_records" / f"step_{step}.npy"
            record = load_policy_record(path)
            join = check_join(record, ep, k, episode["task_description"], resize)
            request = request_of(record)
            res = reader.read(request, full=not args.no_full_grid, with_feature=not args.no_feature)
            pooled = res["pooled"]
            cell = res["cell_tokens"]
            grids = np.stack([res["grids"][c] for c in CAMERAS])
            rows["cell_tokens"].append(cell)
            rows["cell_grid_policy"].append(grids)
            rows["cell_grid_opencv"].append(pa.unmirror_grid(grids))
            rows["block_mass_cell"].append(np.array([cell[b.tokens[0]:b.tokens[1]].sum()
                                                     for b in pa.PI05_LIBERO_TOKEN_BLOCKS], np.float32))
            if pooled is not None:
                rows["block_mass_all"].append(np.stack([pooled[..., b.tokens[0]:b.tokens[1]].sum(-1)
                                                        for b in pa.PI05_LIBERO_TOKEN_BLOCKS], -1).astype(np.float32))
                rows["pooled_f16"].append(pooled.astype(np.float16))
            if res["feature"] is not None:
                rows["kv_feature"].append(np.asarray(res["feature"], np.float32))
            rows["call_k"].append(k)
            rows["global_step"].append(step)
            rows["chunk_t"].append(join["chunk_t"])
            rows["obs_index"].append(join["obs_index"])
            rows["record_sha256"].append(sha256_file(path))
            rows["seconds"].append(res["seconds"])
            joins.append(join)
            if check_left[0] > 0:
                ref = pa.v11ab_attention_policy_grids(reader.policy, request, noise_seed=reader.noise_seed,
                                                      num_denoise_steps=reader.num_denoise_steps)
                diffs = {}
                for b in pa.PI05_LIBERO_TOKEN_BLOCKS:
                    mine = cell[b.tokens[0]:b.tokens[1]].reshape(pa.GRID, pa.GRID)
                    diffs[b.model_key] = float(np.max(np.abs(mine - ref[b.v11ab_policy_camera])))
                v11ab.append({"global_step": step, "max_abs_diff": diffs, "bit_equal": all(v == 0.0 for v in diffs.values())})
                check_left[0] -= 1
            print(f"  t{episode['task_id']} e{episode['episode_idx']} call {k + 1}/{len(calls)} "
                  f"{res['seconds'] * 1e3:.1f} ms", flush=True)
    arrays = {key: np.asarray(value) for key, value in rows.items() if value}
    arrays["cameras"] = np.array(CAMERAS)
    arrays["record_sha256"] = np.array(rows["record_sha256"])
    name = episode_stem(episode["suite"], episode["task_id"], episode["episode_idx"]) + "_attn.npz"
    target = out_dir / "attention" / name
    with open(target, "xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"attn_npz": str(target.relative_to(out_dir)), "calls": len(calls), "joins": joins, "v11ab_check": v11ab}


def run(args) -> int:
    from AG3S.adapters.panda import attention as pa
    from benchmark.runners import launch_libero_pi05_only as launcher
    from common.run_artifacts import create_run, record_execution

    if os.environ.get("AG3S_PROJECT_ROOT") != str(PROJECT_ROOT):
        raise RuntimeError("source scripts/setup/env.sh of this checkout first")
    sources = [index_source_run(resolve_source_run(value)) for value in args.source_run]
    for s in sources:
        if s["policy_config"] not in (None, args.config):
            raise ValueError(f"source run {s['run_id']} used policy config {s['policy_config']}, not {args.config}")
        if args.task_ids is not None:
            s["episodes"] = [e for e in s["episodes"] if e["task_id"] in set(args.task_ids)]
    checkpoint_dir = Path(args.checkpoint_dir or sources[0]["checkpoint"] or launcher.default_checkpoint())
    checkpoint = launcher.checkpoint_identity(checkpoint_dir, require=not args.dry_run)
    for s in sources:
        if s["checkpoint_aggregate_sha256"] and checkpoint.get("aggregate_sha256") not in (None, s["checkpoint_aggregate_sha256"]):
            raise ValueError(f"checkpoint {checkpoint_dir} is not the one source run {s['run_id']} was served with")
    # Targets / segmentation instances first: a task the BDDL link cannot resolve fails before the GPU work.
    targets = {(s["run_id"], e["task_id"], e["episode_idx"]): episode_target(e, Path(s["run_dir"]))
               for s in sources for e in s["episodes"]}
    settings = build_settings(args, sources, checkpoint, probe=not args.dry_run)
    if args.dry_run:
        calls = sum(min(len(e["global_steps"]), args.limit_calls or len(e["global_steps"]))
                    for s in sources for e in s["episodes"])
        print(json.dumps({"settings": settings, "calls": calls}, indent=1, default=str))
        return 0

    run_dir = create_run(run_id=args.run_id, settings=settings, checkpoint=str(checkpoint_dir))
    (run_dir / "attention").mkdir()
    print(run_dir, flush=True)
    record_execution(run_dir, checkpoint=str(checkpoint_dir), settings={"role": "attention_audit", "argv": sys.argv,
                                                                         "pid": os.getpid()})
    result = {"role": "attention_audit_result"}
    started = time.monotonic()
    try:
        from AG3S.ag3s.experiments.sources.pi05_attention import load_policy

        t0 = time.monotonic()
        policy = load_policy(args.config, str(checkpoint_dir))
        result["load_seconds"] = time.monotonic() - t0
        result["memory_after_load"] = device_memory()
        reader = pa.PandaAttentionReader(policy, num_denoise_steps=args.num_denoise, noise_seed=args.noise_seed)

        first = next(e for s in sources for e in s["episodes"])
        first_src = next(s for s in sources if first in s["episodes"])
        req = request_of(load_policy_record(Path(first_src["run_dir"]) / "policy_records" / f"step_{first['global_steps'][0]}.npy"))
        layout = reader.token_layout(req)
        problems = pa.check_token_layout(layout)
        (run_dir / "token_layout.json").write_text(json.dumps({"layout": layout, "problems": problems,
                                                              "expected": settings["token_blocks"]}, indent=1) + "\n")
        result["token_layout_problems"] = problems
        if problems and not args.allow_layout_mismatch:
            raise ValueError(f"pi05_libero token layout differs from PI05_LIBERO_TOKEN_BLOCKS: {problems}")

        check_left = [int(args.check_v11ab)]
        seconds, v11ab, n_calls = [], [], 0
        with open(run_dir / "attention_index.jsonl", "x") as index:
            for s in sources:
                for e in s["episodes"]:
                    out = audit_episode(reader, s, e, run_dir, args, check_left=check_left)
                    with np.load(str(run_dir / out["attn_npz"])) as arr:
                        seconds.extend(arr["seconds"].tolist())
                    v11ab.extend(out["v11ab_check"])
                    n_calls += out["calls"]
                    row = {"source_run": s["run_id"], "source_run_dir": s["run_dir"],
                           **{k: e[k] for k in ("suite", "task_id", "episode_idx", "init_state_index", "task_description",
                                                "status", "official_success", "policy_calls", "npz", "geom_npz")},
                           "npz_sha256": sha256_file(Path(s["run_dir"]) / e["npz"]),
                           "geom_npz_sha256": sha256_file(Path(s["run_dir"]) / e["geom_npz"]) if e["geom_npz"] else None,
                           "target": targets[(s["run_id"], e["task_id"], e["episode_idx"])],
                           **out}
                    index.write(json.dumps(row) + "\n")
                    index.flush()
        result.update({"calls": n_calls, "episodes": sum(len(s["episodes"]) for s in sources),
                       "first_call_seconds": seconds[0] if seconds else None,
                       "latency_after_first": percentile_summary(seconds[1:]),
                       "memory_end": device_memory(),
                       "v11ab_equivalence": {"checked_calls": len(v11ab), "all_bit_equal": all(c["bit_equal"] for c in v11ab),
                                             "max_abs_diff": max((max(c["max_abs_diff"].values()) for c in v11ab), default=None)}})
        (run_dir / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        record_execution(run_dir, checkpoint=str(checkpoint_dir), settings=result)
        print(json.dumps({k: v for k, v in result.items() if k != "role"}, default=str), flush=True)
    return 0


# --- SL6 B5: noun rows ---------------------------------------------------------------------------------------
class NounRowReader:
    """Target-noun row attention of one recorded request (SL6 B5).

    ``policy``: the ``return_attn_probs=True`` policy (`pi05_attention.load_policy`). ``tokens``: `NounTokens` with
    ``rule="libero"`` (default: built from ``config``). ``extractor`` / ``observation_fn`` are injectable for CPU
    tests (default: `NounQueryExtractor(policy._model)` and the batched `Observation` of the input transform)."""

    def __init__(self, policy, *, config: str = POLICY_CONFIG, layers=NOUN_LAYERS, tokens=None, extractor=None,
                 observation_fn=None):
        from AG3S.trajopt.attention_source import NounTokens

        self.policy = policy
        self.layers = tuple(int(l) for l in layers)
        self.tokens = tokens if tokens is not None else NounTokens.from_config(config, rule="libero")
        if self.tokens.rule != "libero":
            raise ValueError("the noun-row audit reads the LIBERO target noun: NounTokens(rule='libero')")
        if extractor is None:
            from AG3S.ag3s.experiments.studies.t49_noun_query_extract import NounQueryExtractor

            extractor = NounQueryExtractor(policy._model)  # noqa: SLF001
        self.extractor = extractor
        self.observation_fn = observation_fn or self._observation

    @staticmethod
    def _observation(inputs):
        import jax
        import jax.numpy as jnp

        from openpi.models import model as _model

        return _model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None, ...], dict(inputs)))

    def read(self, request: dict) -> dict:
        """``{"ok", "lookup" (record), "rows" (3,), "image" (len(layers), H, 768) f32 of the target row,
        "mass" {image_mass, text_mass, bos, self: (len(layers), H, 3)}, "seconds"}``; not ok -> zeros, never guessed."""
        from AG3S.trajopt.attention_source import noun_lookup

        inputs = self.policy._input_transform(dict(request))  # noqa: SLF001
        look = noun_lookup(inputs, request.get("prompt"), self.tokens)
        if not look.ok:
            return {"ok": False, "lookup": look.record(), "rows": None, "image": None, "mass": None, "seconds": 0.0}
        t0 = time.perf_counter()
        out = self.extractor.rows(self.observation_fn(inputs), np.asarray(look.rows, np.int32), timed=False)
        seconds = time.perf_counter() - t0
        sel = list(self.layers)
        image = np.asarray(out["image"], np.float32)[sel][:, :, 0, :]
        mass = {k: np.asarray(out[k], np.float32)[sel] for k in ("image_mass", "text_mass", "bos", "self")}
        return {"ok": True, "lookup": look.record(), "rows": np.asarray(look.rows, np.int32), "image": image,
                "mass": mass, "seconds": seconds}


def audit_noun_episode(reader: NounRowReader, source: dict, episode: dict, out_dir: Path, args) -> dict:
    run_dir = Path(source["run_dir"])
    calls = episode["global_steps"][: args.limit_calls] if args.limit_calls else episode["global_steps"]
    n_l, n_h = len(reader.layers), None
    rows = {k: [] for k in NOUN_EPISODE_KEYS if k != "layers"}
    joins, lookups = [], []
    with np.load(str(run_dir / episode["npz"])) as ep:
        ep = {k: ep[k] for k in ("chunk_t", "policy_state", "obs_agentview_image", "obs_robot0_eye_in_hand_image")}
        for k, step in enumerate(calls):
            path = run_dir / "policy_records" / f"step_{step}.npy"
            record = load_policy_record(path)
            join = check_join(record, ep, k, episode["task_description"], 224)
            res = reader.read(request_of(record))
            if res["ok"]:
                n_h = res["image"].shape[1]
                rows["noun_image"].append(res["image"])
                rows["noun_mass"].append(np.stack([res["mass"][m] for m in ("image_mass", "text_mass", "bos", "self")]))
                rows["rows"].append(res["rows"])
            else:
                rows["noun_image"].append(None)
                rows["noun_mass"].append(None)
                rows["rows"].append(np.full(3, -1, np.int32))
            rows["token_index"].append(-1 if res["lookup"]["token_index"] is None else int(res["lookup"]["token_index"]))
            rows["noun_ok"].append(bool(res["ok"]))
            rows["call_k"].append(k)
            rows["global_step"].append(step)
            rows["chunk_t"].append(join["chunk_t"])
            rows["obs_index"].append(join["obs_index"])
            rows["record_sha256"].append(sha256_file(path))
            rows["seconds"].append(res["seconds"])
            joins.append(join)
            lookups.append(res["lookup"])
            print(f"  noun t{episode['task_id']} e{episode['episode_idx']} call {k + 1}/{len(calls)} "
                  f"{res['lookup']['noun']} ok={res['ok']} {res['seconds'] * 1e3:.1f} ms", flush=True)
    if n_h is None:
        raise ValueError(f"t{episode['task_id']} e{episode['episode_idx']}: no request had a target noun "
                         f"({lookups[0]['noun_error'] if lookups else 'no calls'})")
    for key in ("noun_image", "noun_mass"):
        shape = next(v.shape for v in rows[key] if v is not None)
        rows[key] = [np.zeros(shape, np.float32) if v is None else v for v in rows[key]]
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    arrays["noun_image"] = arrays["noun_image"].astype(np.float32)
    arrays["layers"] = np.asarray(reader.layers, np.int64)
    arrays["noun_mass_names"] = np.array(["image_mass", "text_mass", "bos", "self"])
    arrays["record_sha256"] = np.array(rows["record_sha256"])
    name = episode_stem(episode["suite"], episode["task_id"], episode["episode_idx"]) + "_noun.npz"
    target = out_dir / "noun_rows" / name
    with open(target, "xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"noun_npz": str(target.relative_to(out_dir)), "calls": len(calls), "joins": joins,
            "noun_ok": int(sum(rows["noun_ok"])), "lookup_first": lookups[0] if lookups else None,
            "layers": list(reader.layers), "heads": n_h, "shape_noun_image": list(arrays["noun_image"].shape)}


def run_noun_rows(args) -> int:
    """SL6 B5 (``--noun-rows``): the target-noun row audit run (GPU unless ``JAX_PLATFORMS=cpu``)."""
    from benchmark.runners import launch_libero_pi05_only as launcher
    from common.run_artifacts import create_run, record_execution

    if os.environ.get("AG3S_PROJECT_ROOT") != str(PROJECT_ROOT):
        raise RuntimeError("source scripts/setup/env.sh of this checkout first")
    layers = _layer_range(args.noun_layers)
    sources = [index_source_run(resolve_source_run(value)) for value in args.source_run]
    for s in sources:
        if s["policy_config"] not in (None, args.config):
            raise ValueError(f"source run {s['run_id']} used policy config {s['policy_config']}, not {args.config}")
        if args.task_ids is not None:
            s["episodes"] = [e for e in s["episodes"] if e["task_id"] in set(args.task_ids)]
    checkpoint_dir = Path(args.checkpoint_dir or sources[0]["checkpoint"] or launcher.default_checkpoint())
    checkpoint = launcher.checkpoint_identity(checkpoint_dir, require=not args.dry_run)
    for s in sources:
        if s["checkpoint_aggregate_sha256"] and checkpoint.get("aggregate_sha256") not in (None, s["checkpoint_aggregate_sha256"]):
            raise ValueError(f"checkpoint {checkpoint_dir} is not the one source run {s['run_id']} was served with")
    targets = {(s["run_id"], e["task_id"], e["episode_idx"]): episode_target(e, Path(s["run_dir"]))
               for s in sources for e in s["episodes"]}
    settings = build_settings(args, sources, checkpoint, probe=not args.dry_run)
    settings.update({"step": NOUN_STEP, "condition": NOUN_CONDITION,
                     "noun_rows": {"layers": list(layers), "heads": "all (8)", "row": "target (B1 libero rule, last piece)",
                                   "tokens": "NounTokens.from_config(config, rule='libero')",
                                   "extraction": "t49_noun_query_extract.NounQueryExtractor (prefix pass, slice_probs)",
                                   "dtype": "float32"}})
    settings["attention"] = {"noun_rows": True, "limit_calls": args.limit_calls, "task_ids": args.task_ids}
    if args.dry_run:
        calls = sum(min(len(e["global_steps"]), args.limit_calls or len(e["global_steps"]))
                    for s in sources for e in s["episodes"])
        print(json.dumps({"settings": settings, "calls": calls}, indent=1, default=str))
        return 0

    run_dir = create_run(run_id=args.run_id, settings=settings, checkpoint=str(checkpoint_dir))
    (run_dir / "noun_rows").mkdir()
    print(run_dir, flush=True)
    record_execution(run_dir, checkpoint=str(checkpoint_dir), settings={"role": "noun_row_audit", "argv": sys.argv,
                                                                         "pid": os.getpid()})
    result = {"role": "noun_row_audit_result"}
    started = time.monotonic()
    try:
        from AG3S.ag3s.experiments.sources.pi05_attention import load_policy

        t0 = time.monotonic()
        policy = load_policy(args.config, str(checkpoint_dir))
        result["load_seconds"] = time.monotonic() - t0
        result["memory_after_load"] = device_memory()
        reader = NounRowReader(policy, config=args.config, layers=layers)
        seconds, n_calls, n_ok = [], 0, 0
        with open(run_dir / "noun_index.jsonl", "x") as index:
            for s in sources:
                for e in s["episodes"]:
                    out = audit_noun_episode(reader, s, e, run_dir, args)
                    with np.load(str(run_dir / out["noun_npz"])) as arr:
                        seconds.extend(arr["seconds"][arr["noun_ok"]].tolist())
                    n_calls += out["calls"]
                    n_ok += out["noun_ok"]
                    row = {"source_run": s["run_id"], "source_run_dir": s["run_dir"],
                           **{k: e[k] for k in ("suite", "task_id", "episode_idx", "init_state_index", "task_description",
                                                "status", "official_success", "policy_calls", "npz", "geom_npz")},
                           "npz_sha256": sha256_file(Path(s["run_dir"]) / e["npz"]),
                           "geom_npz_sha256": sha256_file(Path(s["run_dir"]) / e["geom_npz"]) if e["geom_npz"] else None,
                           "target": targets[(s["run_id"], e["task_id"], e["episode_idx"])], **out}
                    index.write(json.dumps(row) + "\n")
                    index.flush()
        result.update({"calls": n_calls, "noun_ok": n_ok, "episodes": sum(len(s["episodes"]) for s in sources),
                       "first_call_seconds": seconds[0] if seconds else None,
                       "latency_after_first": percentile_summary(seconds[1:]), "memory_end": device_memory()})
        (run_dir / "summary.json").write_text(json.dumps(result, indent=1) + "\n")
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        record_execution(run_dir, checkpoint=str(checkpoint_dir), settings=result)
        print(json.dumps({k: v for k, v in result.items() if k != "role"}, default=str), flush=True)
    return 0


def _layer_range(text: str) -> tuple[int, ...]:
    """``"2-17"`` or ``"2,3,9"`` -> layer indices."""
    text = str(text).strip()
    if "-" in text and "," not in text:
        lo, hi = (int(v) for v in text.split("-"))
        return tuple(range(lo, hi + 1))
    return tuple(int(v) for v in text.split(",") if v.strip())


def _int_list(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-run", action="append", required=True,
                        help="source run dir name or path below results/benchmark (repeatable)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--config", default=POLICY_CONFIG)
    parser.add_argument("--checkpoint-dir", default=None, help="default: the first source run's checkpoint")
    parser.add_argument("--task-ids", type=_int_list, default=None, help="only these task ids (default: all)")
    parser.add_argument("--limit-calls", type=int, default=None, help="only the first N calls of each episode")
    parser.add_argument("--num-denoise", type=int, default=10, help="AttentionSampler Euler steps (V11AB: 10)")
    parser.add_argument("--noise-seed", type=int, default=0, help="AttentionSampler noise seed (V11AB: 0)")
    parser.add_argument("--no-full-grid", action="store_true", help="do not store the pooled [L, H, 768] array")
    parser.add_argument("--no-feature", action="store_true",
                        help="attention() instead of attention_and_subtask_feature() (V11AB serves with the feature)")
    parser.add_argument("--check-v11ab", type=int, default=2,
                        help="compare this many calls with V11AB's own AttentionPolicy read-out (extra pass each)")
    parser.add_argument("--allow-layout-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="index + settings only; no run, no GPU")
    parser.add_argument("--noun-rows", action="store_true",
                        help="SL6 B5: target-noun prefix row -> image attention (layers x heads x 768) instead of the "
                             "V11AB action cell (default off = the SL3 audit)")
    parser.add_argument("--noun-layers", default="2-17", help="with --noun-rows: layers to keep (default 2-17)")
    return parser


def main(argv=None) -> int:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    args = build_parser().parse_args(argv)
    return run_noun_rows(args) if args.noun_rows else run(args)


if __name__ == "__main__":
    sys.exit(main())
