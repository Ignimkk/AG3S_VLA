"""T50b — `kv_L4` feature extraction for `pi05_base` and the 16D LoRA checkpoint (GPU stage, not run by A1).

Same inputs, same transforms, only the parameters differ:

* **LoRA** (`--model lora`): T48's `FeatureExtractor` unchanged (`t48_hsr_features.py:34-65`) —
  `pi05_attention.load_policy(config, checkpoint)` -> `create_trained_policy`.
* **base** (`--model base`): the same train config `pi05_rby1_randomized_pick_place_16d_lora` with
  only the two Gemma variants switched to their non-LoRA names (`gemma_2b_lora` -> `gemma_2b`,
  `gemma_300m_lora` -> `gemma_300m`; `gemma.py:79-107` — same width / depth / heads, the LoRA
  variants only add `lora_configs`). `pi05_base/params` is the initial weight of our LoRA run
  (`training/config.py:1139-1141` `CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`).
  The LoRA adapters are *not* zero-initialised (`lora.py:20`, normal(0.01) for `lora_a` and
  `lora_b`), so loading base params into the LoRA model would add random adapters — hence the
  non-LoRA variant. `pi05_base` has no `assets/`, so the norm stats are passed explicitly from the
  LoRA checkpoint (`create_trained_policy(norm_stats=...)`, `policy_config.py:23, 64-70`). The data
  config (`AlohaInputs(adapt_to_pi=False)`, quantile `Normalize`, `ResizeImages`,
  `TokenizePrompt(discrete_state_input=True)`, `PadStatesAndActions(32)`) is built from the same
  train config; `--stage check-params` verifies the transform chain is identical.

Units and variants are T48's (`t48_hsr_counterfactual.Plan`, same seed -> same plan digests):

* `val`: all six families (C0, C1, C1p, C2g, C2s, C2i) of the 50 val episodes;
* `train`: C0 and C1 (3 distractors each) of the 200 train episodes `range(0, 1600, 8)`
  (SUBTASK-b train split, `t48_hsr_io.TRAIN_EPISODES`).

Cache format = T48's (`feat/<tag>/<set>/<unit>.npz`, `t48_hsr_counterfactual.py:459-472`), so
`t50_core.load_feature_dir` reads both. Resumable: a file with the same plan digest is skipped; a
different digest stops the run (never overwritten).

Nothing heavy runs at import time; jax / openpi are imported inside functions.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import time
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from AG3S.ag3s.experiments.studies import t48_hsr_core as t48core
from AG3S.ag3s.experiments.studies import t48_hsr_io as io
from AG3S.ag3s.experiments.studies.t48_hsr_features import FeatureExtractor

MODELS = ("lora", "base")
EXTRACT_SETS = ("val", "train")
#: `pi05_base` under the shared storage (T50.task.md).
BASE_REL = "cache/rby1/openpi/openpi-assets/checkpoints/pi05_base"
TRAIN_FAMILIES = ("C0", "C1")
#: Variant names: LoRA -> plain (`gemma.py:79-107`).
NON_LORA = {"gemma_2b_lora": "gemma_2b", "gemma_300m_lora": "gemma_300m"}


def base_checkpoint(storage: Optional[str | pathlib.Path] = None) -> pathlib.Path:
    if storage is None:
        from common.workspace import shared_storage_root

        storage = shared_storage_root()
    return pathlib.Path(storage) / BASE_REL


# ----------------------------------------------------------------------------- configs


def train_config_for(config_name: str, model: str, *, return_attn_probs: bool = True):
    """The 16D train config; for `base` with the non-LoRA Gemma variants. Data config untouched."""
    from openpi.training import config as _config

    if model not in MODELS:
        raise ValueError(f"model {model!r} not in {MODELS}")
    cfg = _config.get_config(config_name)
    m = dataclasses.replace(cfg.model, return_attn_probs=return_attn_probs)
    if model == "base":
        m = dataclasses.replace(m, paligemma_variant=NON_LORA.get(m.paligemma_variant, m.paligemma_variant),
                                action_expert_variant=NON_LORA.get(m.action_expert_variant,
                                                                   m.action_expert_variant))
    return dataclasses.replace(cfg, model=m)


def load_base_policy(config_name: str, base_dir: str | pathlib.Path, norm_checkpoint: str | pathlib.Path):
    """`pi05_base` params in the 16D config (non-LoRA variants) with the LoRA checkpoint's norm stats."""
    from openpi.policies import policy_config as _pc
    from openpi.training import checkpoints as _checkpoints

    cfg = train_config_for(config_name, "base")
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = _checkpoints.load_norm_stats(pathlib.Path(norm_checkpoint) / "assets", data_config.asset_id)
    return _pc.create_trained_policy(cfg, pathlib.Path(base_dir), norm_stats=norm_stats)


class BaseFeatureExtractor(FeatureExtractor):
    """T48 `FeatureExtractor` with the policy from `load_base_policy` (`t48_hsr_features.py:35-65`,
    same sampler / tokenizer / batched reduction; only the policy loader differs)."""

    def __init__(self, config_name: str, base_dir: str, norm_checkpoint: str, *, batch: int = 1):  # noqa: D107
        import jax
        import jax.numpy as jnp

        from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler
        from AG3S.ag3s.stages.subtask_probe import KV_HEAD, KV_LAYER, N_IMAGE_TOKENS
        from openpi.models.tokenizer import PaligemmaTokenizer

        if batch < 1:
            raise ValueError("batch must be >= 1")
        self._jax = jax
        self.batch = int(batch)
        t0 = time.perf_counter()
        self.policy = load_base_policy(config_name, base_dir, norm_checkpoint)
        self.load_s = time.perf_counter() - t0
        self.sampler = AttentionSampler(self.policy._model)  # noqa: SLF001
        cfg = train_config_for(config_name, "base")
        self._tok = PaligemmaTokenizer(cfg.model.max_token_len)

        def kv_batched(kv, prefix_mask):
            v = kv[1][KV_LAYER, :, :, KV_HEAD, :].astype(jnp.float32)
            tm = prefix_mask[:, N_IMAGE_TOKENS:].astype(jnp.float32)
            txt = (v[:, N_IMAGE_TOKENS:] * tm[..., None]).sum(1) / tm.sum(1, keepdims=True)
            img = v[:, :N_IMAGE_TOKENS].mean(1)
            return jnp.concatenate([txt, img], axis=-1)

        self._kv_batched = jax.jit(kv_batched)
        self.t_prefix: list[float] = []
        self.n_per_call: list[int] = []


def make_extractor(model: str, *, config_name: str, lora_checkpoint: str, base_dir: str, batch: int):
    if model == "lora":
        return FeatureExtractor(config_name, lora_checkpoint, batch=batch)
    if model == "base":
        return BaseFeatureExtractor(config_name, base_dir, lora_checkpoint, batch=batch)
    raise ValueError(model)


# ----------------------------------------------------------------------------- CPU checks (no weights)


def checkpoint_param_shapes(params_dir: str | pathlib.Path) -> dict[tuple, tuple]:
    """`{key path: shape}` from an orbax checkpoint's `_METADATA` (metadata JSON only, no array data);
    a trailing nnx `"value"` is dropped like `model.restore_params` (`model.py:327-331`)."""
    import ast

    meta = json.loads((pathlib.Path(params_dir) / "_METADATA").read_text())["tree_metadata"]
    out = {}
    for k, v in meta.items():
        path = tuple(ast.literal_eval(k))
        out[path] = tuple(v["value_metadata"].get("write_shape") or ())
    if out and all(p[-1] == "value" for p in out):
        out = {p[:-1]: s for p, s in out.items()}
    return {p[1:] if p and p[0] == "params" else p: s for p, s in out.items()}


def model_param_shapes(config_name: str, model: str) -> dict[tuple, tuple]:
    """Abstract (`nnx.eval_shape`, no allocation) parameter shapes of the model `model.load` builds
    (`model.py:233-241`)."""
    import flax.nnx as nnx
    import flax.traverse_util as tu
    import jax

    cfg = train_config_for(config_name, model)
    m = nnx.eval_shape(cfg.model.create, jax.random.key(0))
    _, state = nnx.split(m)
    flat = tu.flatten_dict(state.to_pure_dict())
    return {tuple(str(x) for x in k): tuple(v.shape) for k, v in flat.items()}


def compare_param_trees(expected: dict, got: dict) -> dict:
    """What `model.load(..., remove_extra_params=True)` would do: extra checkpoint keys are dropped;
    missing keys or shape mismatches would fail `check_pytree_equality`."""
    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    bad = sorted(k for k in set(expected) & set(got) if tuple(expected[k]) != tuple(got[k]))
    return {"n_model": len(expected), "n_checkpoint": len(got), "missing_in_checkpoint": [list(k) for k in missing],
            "extra_in_checkpoint": [list(k) for k in extra][:50], "n_extra": len(extra),
            "shape_mismatch": [{"key": list(k), "model": list(expected[k]), "checkpoint": list(got[k])} for k in bad],
            "loadable": not missing and not bad}


def transform_chain(config_name: str, model: str) -> list[str]:
    """reprs of the data + model input transforms the policy would apply (no norm stats)."""
    import re

    cfg = train_config_for(config_name, model)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    out = []
    for t in (*dc.data_transforms.inputs, *dc.model_transforms.inputs):
        r = re.sub(r" at 0x[0-9a-fA-F]+", "", repr(t))   # object addresses differ between instances
        tok = getattr(t, "tokenizer", None)
        if tok is not None:
            r += f" [tokenizer max_len={getattr(tok, '_max_len', None)}]"
        out.append(r)
    return out + [f"use_quantile_norm={dc.use_quantile_norm}", f"asset_id={dc.asset_id}"]


def check_params(config_name: str, lora_checkpoint: pathlib.Path, base_dir: pathlib.Path) -> dict:
    """CPU-only, metadata-only: (1) both checkpoints' parameter trees fit the model each is loaded
    into, (2) the transform chain is identical for both, (3) the prefix-relevant config is equal."""
    out: dict = {"config": config_name, "lora_checkpoint": str(lora_checkpoint), "base_dir": str(base_dir)}
    for model, params in (("lora", lora_checkpoint / "params"), ("base", base_dir / "params")):
        out[model] = compare_param_trees(model_param_shapes(config_name, model), checkpoint_param_shapes(params))
        cfg = train_config_for(config_name, model)
        out[model]["model_config"] = {k: str(getattr(cfg.model, k)) for k in (
            "pi05", "paligemma_variant", "action_expert_variant", "action_dim", "action_horizon",
            "max_token_len", "discrete_state_input")}
    ta, tb = transform_chain(config_name, "lora"), transform_chain(config_name, "base")
    out["transforms_equal"] = ta == tb
    out["transforms"] = ta
    out["base_has_assets"] = (base_dir / "assets").exists()
    out["lora_norm_stats"] = str(lora_checkpoint / "assets")
    return out


# ----------------------------------------------------------------------------- units


def dataset_images(unit: t48core.Unit, dataset: pathlib.Path, rows: Iterable[int]) -> dict[int, dict]:
    """`{sampled row: {cam: HWC uint8}}` — the dataset branch of `t48_hsr_counterfactual.unit_images`
    (`:392-403`), usable for train units too. Stops on frames beyond the video length (SUBTASK-b
    `extract.py:200` would have skipped them, so the plan would not be SUBTASK-b's)."""
    rows = sorted(set(int(r) for r in rows))
    ep = int(unit.extra["episode"])
    vids = {c: io.decode_video(io.video_path(dataset, ep, c)) for c in io.CAMS}
    nvid = min(len(v) for v in vids.values())
    too_far = [int(unit.clock[r]) for r in rows if int(unit.clock[r]) >= nvid]
    if too_far:
        raise RuntimeError(f"{unit.unit_id}: frames {too_far[:5]} beyond video length {nvid}")
    unit.extra["n_video"] = nvid
    return {r: {c: vids[c][int(unit.clock[r])] for c in io.CAMS} for r in rows}


def iter_units(plan, set_name: str, max_units: Optional[int], max_frames: Optional[int]
               ) -> Iterator[tuple[t48core.Unit, t48core.VariantTable]]:
    """(unit, table) of a set. `plan` = `t48_hsr_counterfactual.Plan` (templates, gripper reps, seed)."""
    if set_name == "val":
        for unit in plan.units("val"):
            yield unit, plan.table(unit)
    elif set_name == "train":
        eps = io.TRAIN_EPISODES[:max_units] if max_units else io.TRAIN_EPISODES
        for ep in eps:
            unit = io.dataset_unit(plan.paths.dataset, ep, plan.tasks, plan.meta, max_frames=max_frames)
            unit = dataclasses.replace(unit, set_name="train")
            yield unit, t48core.build_variants(unit, plan.templates, plan.reps, seed=plan.args.seed,
                                               families=TRAIN_FAMILIES)
    else:
        raise ValueError(set_name)


# ----------------------------------------------------------------------------- extraction loop


def extract_unit(ex, unit: t48core.Unit, table: t48core.VariantTable, images: dict[int, dict], *,
                 batch: int, q01: np.ndarray, q99: np.ndarray, set_name: str) -> dict[str, np.ndarray]:
    """Features + T48 cache arrays of one unit (`t48_hsr_counterfactual.py:438-469`)."""
    from AG3S.ag3s.experiments.studies.t48_hsr_counterfactual import parse_state_string

    def make(v):
        return ex.inputs(table.state[v], images[int(table.image_row[v])], str(table.prompt[v]))

    n_before = len(ex.t_prefix)
    feats, decoded, n_txt = [], [], []
    for i in range(0, len(table), batch):
        xs = [make(v) for v in range(i, min(i + batch, len(table)))]
        feats.append(ex.features(xs))
        decoded += [ex.decoded_prompt(x) for x in xs]
        n_txt += [int(np.asarray(x["tokenized_prompt_mask"]).sum()) for x in xs]
    feats = np.concatenate(feats)
    decoded = np.asarray(decoded)
    bins = np.stack([b if (b := parse_state_string(d)) is not None else np.full(16, -9, np.int64)
                     for d in decoded])
    rep = t48core.discretize_state(table.state, q01, q99)
    arrs = table.as_arrays("plan__")
    arrs.update(feature=feats.astype(np.float32), plan_digest=np.asarray(table.digest()),
                decoded_prompt=decoded, n_text_tokens=np.asarray(n_txt, np.int64), state_bins_serving=bins,
                state_bins_replica_mismatch=np.asarray(int((bins != rep).any(1).sum())),
                unit_id=np.asarray(unit.unit_id), set_name=np.asarray(set_name),
                clock=unit.clock, y=unit.y, rows=unit.rows,
                prefix_s=np.asarray(ex.t_prefix[n_before:], np.float64),
                n_video=np.asarray(int(unit.extra.get("n_video", -1))))
    return arrs


def run_extraction(ex, units: Iterable[tuple[t48core.Unit, t48core.VariantTable]], fdir: pathlib.Path, *,
                   images_fn: Callable[[t48core.Unit, np.ndarray], dict], batch: int, warmup: int,
                   q01: np.ndarray, q99: np.ndarray, set_name: str, meta: dict,
                   on_unit: Optional[Callable[[dict], None]] = None) -> dict:
    """Cache every unit's features under `fdir` (resumable, digest-checked). `meta` collects
    per-unit timing and the batch parity; returned."""
    fdir.mkdir(parents=True, exist_ok=True)
    meta.setdefault("units", {})
    for unit, table in units:
        dig = table.digest()
        path = fdir / f"{unit.unit_id}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z["plan_digest"]) == dig:
                    print(f"[extract {set_name}] {unit.unit_id} cached", flush=True)
                    continue
            raise SystemExit(f"{path} exists with a different plan digest — the plan changed "
                             f"(seed / data / code). Use a new AG3S_RUN_DIR; not overwriting.")
        t0 = time.time()
        imgs = images_fn(unit, table.image_row)
        if not meta.get("warmed"):
            first = [ex.inputs(table.state[v], imgs[int(table.image_row[v])], str(table.prompt[v]))
                     for v in range(min(batch, len(table)))]
            ex.warmup(first[0], warmup)
            meta["batch_parity"] = ex.batch_parity(first)
            meta["warmed"] = True
        n_before = len(ex.t_prefix)
        arrs = extract_unit(ex, unit, table, imgs, batch=batch, q01=q01, q99=q99, set_name=set_name)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, **arrs)
        tmp.replace(path)
        per = np.asarray(ex.t_prefix[n_before:]) / np.maximum(np.asarray(ex.n_per_call[n_before:]), 1)
        meta["units"][f"{set_name}/{unit.unit_id}"] = {
            "variants": len(table), "wall_s": time.time() - t0,
            "per_input_ms_median": float(np.median(per) * 1e3) if len(per) else None,
            "replica_mismatch": int(arrs["state_bins_replica_mismatch"])}
        print(f"[extract {set_name}] {unit.unit_id} V={len(table)} {time.time() - t0:.1f}s "
              f"mem={ex.memory()}", flush=True)
        if on_unit is not None:
            on_unit(meta)
    return meta


def gpu_guard(allow_busy: bool) -> list[str]:
    from AG3S.ag3s.experiments.studies.t48_hsr_counterfactual import gpu_compute_processes

    procs = gpu_compute_processes()
    if procs and not allow_busy:
        raise SystemExit("other GPU compute processes are running (nvidia-smi):\n  " + "\n  ".join(procs)
                         + "\nWait for them (do not stop them) or pass --allow-busy-gpu if the lead approved.")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    return procs
