"""Extract pi0.5 PaliGemma intermediate-layer hidden states on the CPU, from recorded observations (T73).

What it gives you: for every requested layer l, the **residual stream** of the PaliGemma (VLM) expert
over the prefix tokens -- three 16x16 SigLIP camera grids followed by the 200 text tokens -- as
``[n_layer_sel, T, 2048]`` per frame. Layer numbering follows HF ``output_hidden_states``:
``h_0`` = the input embedding, ``h_l`` = the output of transformer block l (l = 1 ... 18), before the
final norm. This is not the KV cache ``kv_L4`` that `pi05_attention.AttentionSampler` reduces (that is
RMSNorm(h) projected by W_v to 256 dims); it is the 2048-dim stream itself.

How: `Pi0Config.return_hidden_states=True` (inference-only, adds no parameters, loads the same
checkpoint) and **the prefix pass only** -- `embed_prefix` + the PaliGemma LLM over the prefix. No
denoising and no action expert run. The observation goes through the policy's own input transform
(`pi05_attention.observation_from_step`), so the tokens are the ones the served policy saw.

**CPU only, by construction.** `main()` sets ``JAX_PLATFORMS=cpu`` and ``CUDA_VISIBLE_DEVICES=""``
before JAX is imported, pins the process to ``--threads`` CPUs (XLA's CPU thread pool follows the
affinity mask), and refuses to run if JAX still reports a non-CPU backend. It never takes the GPU
lock. Importing this module does not import JAX.

Inputs (``--records``, auto-detected):

* an AG3S record ``run_XXXX/`` directory (``meta.json`` + ``step_*.npz``, the format
  `policy_record.load_run` reads; only ``state``, ``t_step`` and the three ``image_*`` arrays are read);
* an OpenPI ``policy_records/session_*/`` directory (``step_N.npy`` written by ``--record``).

Usage (from the checkout root)::

    source scripts/setup/env.sh
    export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"
    MUJOCO_GL=osmesa PYTHONPATH=$PWD .venv-openpi-live/bin/python -m \\
        AG3S.ag3s.experiments.sources.pi05_hidden \\
        --records <.../ag3s/run_0000 or .../policy_records/session_xxx> \\
        --frames 0 10 20 \\
        --layers all --tokens all --pool none --dtype float16 \\
        --out "$AG3S_RUN_DIR/hidden/my_frames.npz"

``--checkpoint`` defaults to the 16D 30k step under ``$AG3S_STORAGE_ROOT`` (the one serve_safe uses),
``--config`` to ``pi05_rby1_randomized_pick_place_16d_lora``. ``--out`` defaults to
``$AG3S_RUN_DIR/hidden/<record name>.npz``.

Output ``.npz`` keys (F = frames, Ls = selected layers, Tsel = selected tokens, D = 2048):

* ``hidden`` -- ``--pool none``: ``[F, Ls, Tsel, D]``; ``--pool mean``: ``[F, Ls, G, D]`` with one row per
  group in ``pool_groups`` (each camera's 256-token mean, and the mean over *valid* text tokens).
  float16 by default (``--dtype float32`` to keep full precision; the model itself runs in bfloat16).
* ``layers`` [Ls] -- layer numbers (0 = embedding, 18 = last block output, pre final norm).
* ``token_index`` [Tsel] -- prefix positions of the stored tokens; ``token_group`` [Tsel] -- 0/1/2 =
  camera (``camera_names`` order), 3 = text; ``token_valid`` [F, Tsel] -- False on text padding.
* ``camera_names`` [3], ``camera_token_ranges`` [3, 2] (= `policy_record.CAMERA_BINDINGS`),
  ``text_token_range`` [2], ``prefix_mask`` [F, T] -- the full prefix validity mask.
* ``text_token_ids`` [F, 200], ``text_token_pieces`` [F, 200] (SentencePiece pieces; pi0.5 also
  writes the discretized state into the text, so these differ per frame).
* ``frame_id`` [F] (``<file stem>@t<t_step>``), ``t_step`` [F], ``source_files`` [F], ``prompt``.
* ``checkpoint``, ``config``, ``code_commit``, ``code_dirty`` (openpi gemma / this tool uncommitted).
* ``absmax`` [F, 19] -- per-layer max |h| in the model dtype (float16 overflows above 65504; the tool
  aborts rather than write inf -- rerun with ``--dtype float32`` if it ever trips).
* ``check_v_relerr`` [F, 18], ``check_final_norm_relerr`` [F] -- the self-checks below (``--no-check``
  skips them and stores NaN).
* ``compile_s``, ``frame_s`` [F] -- AOT compile time and per-frame prefix time (compile excluded).

A ``.json`` sidecar with the same metadata and timings is written next to the ``.npz``.

Self-checks, per frame (relative Frobenius error, float32 arithmetic on bfloat16 values):

1. V from ``h_{l}``: ``kv_einsum(pre_attention_norm_l(h_l))[1]`` against the prefix KV cache's V of
   block l+1 (l = 0 ... 17) -- the stored stream is the one that block actually read.
2. ``final_norm(h_18)`` against the prefix output of the same pass (what the flag-off path returns).

Cost (16 threads, measured in T73): see ``AG3S/ag3s/docs/handoff/T73.impl.md``.

T77 additions (all opt-in; with none of them given the tool behaves exactly as above):

* ``--device {cpu,gpu}`` (default ``cpu`` = the CPU-only path above). ``gpu`` does **not** force
  ``JAX_PLATFORMS`` and does not pin CPUs -- run it inside ``scripts/setup/gpu_lock.py shared``; it refuses to
  run if JAX comes up without a GPU.
* ``--pool summary`` -- per frame and selected layer five rows ``SUMMARY_ROWS`` (each camera's 256-token mean,
  the mean over valid text tokens, the **last valid text token**), float32, plus the same frame's ``kv_L4``
  (SUBTASK-b definition: layer-4 prefix V cache, ``[mean over valid text | mean over the 768 image tokens]``,
  512-D, float32). Pooled inside the jitted pass (`summary_rows`, shared numpy / jax code), so only
  ``[Ls, 5, 2048]`` leaves the device. One ``.npz`` per episode / run under ``--out-dir`` (written atomically;
  an existing file is skipped, so a killed job resumes). The V-cache / final-norm self-check runs on the first
  frame of every file (``--no-check`` skips it).
* inputs for ``--pool summary``: ``--dataset <LeRobot root> --episodes 0:1600:8 [--frame-stride 6]`` (expert
  parquet + mp4, the SUBTASK-b ``extract.py`` reading: frames with ``frame_index % stride == 0`` and inside the
  shortest video, state as float32, prompt = ``tasks.jsonl[task_index]``; stores ``phase_index``,
  ``frame_index``, ``episode``), or ``--records RUN [RUN ...]`` / ``--spec t77_spec.json`` (AG3S client records,
  every request step; the output name is the spec label).

T79 addition (opt-in, closed-loop records only):

* ``--pool patches`` -- per selected request the hidden of the **head camera's 256 image tokens** (``cam_high``,
  prefix positions 0..255 = the 16 x 16 SigLIP grid of the 224 x 224 policy image, row-major) at the selected layers,
  float32 ``[Ls, 256, 2048]`` (``--patch-dtype float16`` halves it; aborts on |h| > 65504); sliced inside the jitted
  pass (`PatchExtractor`). Inputs ``--spec`` / ``--records`` as
  for summary; ``--requests t79_events.json`` (its ``patch_requests`` = {label: [request, ...]}, or a plain
  {label: [...]} json) keeps only those requests -- request ``j`` = the ``j``-th client step sorted by ``t_step``
  (the T77 pairing with the server chunks). Without ``--requests`` every request is stored. One ``.npz`` per run
  under ``--out-dir`` (atomic, resumable, V-cache / final-norm self-check on the first stored request).

T86 additions (opt-in, only with ``--pool patches``; with none of the three flags the tool is the T79 tool above,
bit for bit):

* ``--cameras cam_high cam_left_wrist cam_right_wrist`` -- store those cameras' 256 image tokens (default: the head
  camera only, the T79 file layout). Any of the three T86 flags switches the output to the **T86 layout** (below).
* ``--prompt-override FILE`` -- json ``{label: {request: [prompt, ...]}}`` (or a ``t86_samples.json`` with that under
  ``prompt_override``): every listed request is forwarded once more per extra prompt, **same image**, only the
  prompt tokens differ (counterfactual prompts, `t86_labels`).
* ``--noun-attention [L:H]`` (default cell 10:7 = serving ``noun_target:L10H7``) -- the noun attention in the same
  forward: the prefix attention **row of the prompt's target-noun token** (`attention_source.noun_lookup` on the
  tokens of *that* prompt, so a changed prompt moves the row), layer L head H, the 768 image columns, raw softmax
  mass. Needs the model loaded with ``return_attn_probs`` as well as ``return_hidden_states`` (this flag does it).

T86 layout, one ``.npz`` per run (R requests, Ls layers, C cameras, Q = counterfactual forwards): ``patches``
``[R, C, Ls, 256, D]`` (original prompt), ``cf_patches`` ``[Q, C, Ls, 256, D]`` with ``cf_request`` ``[Q]`` (position
in ``request``) and ``cf_prompt`` ``[Q]``; ``noun_att`` ``[R, 768]`` / ``cf_noun_att`` ``[Q, 768]`` (NaN where the noun
was not found; ``noun_ok`` / ``cf_noun_ok``, ``noun_token_index``, ``noun_mass`` ``[., 2]`` = image / text mass of the
row); ``camera_names``, ``layers``, ``request``, ``t_step``, ``prompt`` ``[R]``, self-check errors, timings.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.sources.policy_record import (
    CAMERA_BINDINGS,
    LANGUAGE_TOKEN_START,
    POLICY_CAMERA_NAMES,
)

DEFAULT_CONFIG = "pi05_rby1_randomized_pick_place_16d_lora"
DEFAULT_CHECKPOINT_REL = (
    "checkpoints/rby1/pi05_rby1_randomized_pick_place_16d_lora/"
    "rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999"
)
TEXT_GROUP = len(POLICY_CAMERA_NAMES)  # token_group id of text tokens
FLOAT16_MAX = 65504.0
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
#: Files whose uncommitted state changes what this tool computes.
CODE_FILES = (
    "third_party/openpi/src/openpi/models/gemma.py",
    "third_party/openpi/src/openpi/models/pi0.py",
    "third_party/openpi/src/openpi/models/pi0_config.py",
    "AG3S/ag3s/experiments/sources/pi05_hidden.py",
)


# ------------------------------------------------------------------------------------ environment


def force_cpu(threads: int) -> list[int]:
    """CPU-only JAX with at most `threads` worker threads. Must run before JAX is imported."""
    if "jax" in sys.modules:
        import jax

        if jax.default_backend() != "cpu":
            raise SystemExit("JAX is already initialised on a non-CPU backend; run this tool in its own process")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTEROP_THREADS",
                "TF_NUM_INTRAOP_THREADS"):
        os.environ[var] = str(int(threads))
    cpus = sorted(os.sched_getaffinity(0))
    chosen = cpus[: max(1, int(threads))]
    os.sched_setaffinity(0, chosen)
    return chosen


def setup_gpu() -> list[int]:
    """T77 ``--device gpu``: no JAX_PLATFORMS / CUDA_VISIBLE_DEVICES forcing, no CPU pinning (the caller holds the
    shared GPU lock). Only keeps the project's no-preallocation default."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return sorted(os.sched_getaffinity(0))


def check_backend(device: str) -> None:
    """After `import jax`: CPU-only for ``cpu`` (T73 check, unchanged), a GPU backend for ``gpu``."""
    import jax

    if device == "cpu":
        if jax.default_backend() != "cpu" or any(d.platform != "cpu" for d in jax.devices()):
            raise SystemExit(f"expected CPU-only JAX, got {jax.devices()}")
    elif jax.default_backend() != "gpu":
        raise SystemExit(f"--device gpu but JAX backend is {jax.default_backend()} ({jax.devices()})")


def code_provenance() -> dict[str, Any]:
    def _git(*a):
        return subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True, check=False).stdout

    commit = _git("rev-parse", "HEAD").strip()
    dirty = _git("status", "--porcelain", "--", *CODE_FILES).strip()
    return {"code_commit": commit, "code_dirty": bool(dirty), "code_dirty_files": dirty.splitlines()}


# ------------------------------------------------------------------------------------ inputs


@dataclasses.dataclass(frozen=True)
class Frame:
    frame_id: str
    t_step: int
    source: str
    state: np.ndarray
    images: dict[str, np.ndarray]  # policy key -> (224, 224, 3) uint8 HWC
    prompt: Optional[str]


def _openpi_step_files(path: pathlib.Path) -> list[pathlib.Path]:
    files = list(path.glob("step_*.npy"))
    return sorted(files, key=lambda f: int(f.stem.split("_")[1]))


def list_frames(records: str | pathlib.Path) -> tuple[str, list[pathlib.Path], Optional[str]]:
    """``(kind, step files in order, prompt from meta or None)`` for an AG3S run or an OpenPI session."""
    path = pathlib.Path(records)
    if (path / "meta.json").exists():
        files = sorted(path.glob("step_*.npz"))
        prompt = json.loads((path / "meta.json").read_text()).get("prompt")
        kind = "ag3s"
    else:
        files = _openpi_step_files(path)
        prompt = None
        kind = "openpi"
    if not files:
        raise SystemExit(f"{path}: neither an AG3S run_XXXX (meta.json + step_*.npz) nor an OpenPI "
                         f"policy_records session (step_N.npy)")
    return kind, files, prompt


def read_frame(kind: str, file: pathlib.Path, meta_prompt: Optional[str]) -> Frame:
    """One recorded observation; reads only what the policy input needs (same keys as `load_run`)."""
    if kind == "ag3s":
        with np.load(file) as d:
            t_step = int(d["t_step"])
            return Frame(
                frame_id=f"{file.stem}@t{t_step}", t_step=t_step, source=str(file),
                state=np.asarray(d["state"], np.float64),
                images={k: np.asarray(d[f"image_{k}"]) for k in POLICY_CAMERA_NAMES},
                prompt=meta_prompt,
            )
    d = np.load(file, allow_pickle=True).item()  # written by openpi PolicyRecorder (trusted: our own runs)
    step = int(file.stem.split("_")[1])
    images = {}
    for k in POLICY_CAMERA_NAMES:
        im = np.asarray(d[f"inputs/images/{k}"])
        if im.ndim == 3 and im.shape[0] in (1, 3) and im.shape[-1] not in (1, 3):
            im = np.transpose(im, (1, 2, 0))  # recorded CHW -> HWC (observation_from_step goes back)
        images[k] = im
    prompt = d.get("inputs/prompt")
    return Frame(frame_id=f"{file.stem}@t{step}", t_step=step, source=str(file),
                 state=np.asarray(d["inputs/state"], np.float64), images=images,
                 prompt=None if prompt is None else str(prompt))


# ------------------------------------------------------------------------------------ token map


def token_selection(tokens: str, total: int) -> tuple[np.ndarray, np.ndarray]:
    """``(token_index, token_group)`` for ``all | image | text`` over a prefix of length `total`."""
    idx, grp = [], []
    if tokens in ("all", "image"):
        for g, cam in enumerate(POLICY_CAMERA_NAMES):
            lo, hi = CAMERA_BINDINGS[cam][2]
            idx.append(np.arange(lo, hi))
            grp.append(np.full(hi - lo, g))
    if tokens in ("all", "text"):
        idx.append(np.arange(LANGUAGE_TOKEN_START, total))
        grp.append(np.full(total - LANGUAGE_TOKEN_START, TEXT_GROUP))
    if not idx:
        raise ValueError(f"unknown --tokens {tokens!r}")
    return np.concatenate(idx).astype(np.int64), np.concatenate(grp).astype(np.int64)


def pool_groups(tokens: str) -> list[str]:
    out = []
    if tokens in ("all", "image"):
        out += list(POLICY_CAMERA_NAMES)
    if tokens in ("all", "text"):
        out.append("text")
    return out


def select_and_pool(hidden: np.ndarray, prefix_mask: np.ndarray, layers: Sequence[int], tokens: str,
                    pool: str) -> tuple[np.ndarray, np.ndarray]:
    """`hidden` [L+1, T, D] (float32) -> ``(stored [Ls, Tsel|G, D], valid [Tsel])``.

    `mean` pools each camera over its 256 tokens and the text over valid (unpadded) tokens only.
    """
    total = hidden.shape[1]
    idx, grp = token_selection(tokens, total)
    valid = np.asarray(prefix_mask, bool)[idx]
    h = hidden[np.asarray(layers)]  # [Ls, T, D]
    if pool == "none":
        return h[:, idx], valid
    if pool != "mean":
        raise ValueError(f"unknown --pool {pool!r}")
    rows = []
    for name in pool_groups(tokens):
        g = TEXT_GROUP if name == "text" else POLICY_CAMERA_NAMES.index(name)
        sel = idx[(grp == g) & valid]
        rows.append(h[:, sel].mean(axis=1) if sel.size else np.full(h[:, 0].shape, np.nan, h.dtype))
    return np.stack(rows, axis=1), valid


# ------------------------------------------------------------------------------------ self-checks


def relerr(a, b) -> float:
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _layer(tree, i: int):
    import jax

    return jax.tree.map(lambda a: a[i], tree)


def recompute_v(llm_params: dict, hidden, config) -> list:
    """V of block l+1 from the stored h_l (l = 0 ... depth-1), with the model's own modules.

    `llm_params` is the linen-style parameter dict of `gemma.Module` (``layers``, ``final_norm`` ...,
    leading layer axis on ``layers``); `hidden` is ``[depth + 1, B, T, D]``; `config` the expert-0
    `gemma.Config`. Returns a list of ``[B, T, K, H]`` arrays.
    """
    import openpi.models.gemma as _gemma
    import openpi.models.lora as _lora

    layers = llm_params["layers"]
    kv_key = "kv_einsum" if config.num_kv_heads != config.num_heads else "qkv_einsum"
    if kv_key != "kv_einsum":
        raise NotImplementedError("only the multi-query (kv_einsum) layout of gemma_2b is supported")
    norm = _gemma.RMSNorm()
    kv_einsum = _lora.Einsum(shape=(2, config.num_kv_heads, config.width, config.head_dim),
                             lora_config=config.lora_configs.get("attn"))
    out = []
    for i in range(config.depth):
        x, _ = norm.apply({"params": _layer(layers["pre_attention_norm"], i)}, hidden[i], None)
        _, v = kv_einsum.apply({"params": _layer(layers["attn"][kv_key], i)}, "BSD,2KDH->2BSKH", x)
        out.append(v)
    return out


def final_norm(llm_params: dict, h_last):
    import openpi.models.gemma as _gemma

    return _gemma.RMSNorm().apply({"params": llm_params["final_norm"]}, h_last, None)[0]


def self_check(llm_params: dict, hidden, kv_v, prefix_out, config) -> tuple[np.ndarray, float]:
    """``(v_relerr [depth], final_norm_relerr)`` -- see the module docstring, checks 1 and 2."""
    v_err = np.asarray([relerr(v, kv_v[i]) for i, v in enumerate(recompute_v(llm_params, hidden, config))])
    return v_err, relerr(final_norm(llm_params, hidden[-1]), prefix_out)


# ------------------------------------------------------------------------------------ model


def load_hidden_policy(config_name: str, checkpoint: str, attn_probs: bool = False):
    """The trained policy with `return_hidden_states=True` (attention probabilities stay off unless `attn_probs`,
    T86 ``--noun-attention``)."""
    from openpi.policies import policy_config as _pc
    from openpi.shared import download
    from openpi.training import config as _config

    cfg = _config.get_config(config_name)
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, return_hidden_states=True))
    if attn_probs:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, return_attn_probs=True))
    return _pc.create_trained_policy(cfg, download.maybe_download(checkpoint)), cfg


class HiddenExtractor:
    """One jitted prefix pass -> ``(prefix_out [B,T,D], kv V [L,B,T,K,H], hidden [L+1,B,T,D], mask [B,T])``."""

    def __init__(self, model):
        import flax.nnx as nnx
        import jax

        self._nnx = nnx
        self._graphdef, self._state = nnx.split(model)
        self._jit = jax.jit(self._prefix)
        self._compiled = None
        self.compile_s = None

    def _prefix(self, state, obs):
        import jax.numpy as jnp

        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask

        model = self._nnx.merge(self._graphdef, state)
        obs = _model.preprocess_observation(None, obs, train=False)  # exactly once, as sample_actions
        tokens, mask, ar_mask = model.embed_prefix(obs)
        attn = make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1
        (prefix_out, _), kv, hidden = model.PaliGemma.llm([tokens, None], mask=attn, positions=positions)
        return prefix_out, kv[1], hidden, mask

    def compile(self, obs) -> float:
        """AOT compile (timed separately from the per-frame pass). `lower` rebuilds `Observation` with
        `ArgInfo` leaves, which its jaxtyping check rejects -- hence `disable_typechecking`, as
        `Pi0Config.inputs_spec` does for its abstract observation."""
        from openpi.shared import array_typing as at

        t0 = time.perf_counter()
        with at.disable_typechecking():
            self._compiled = self._jit.lower(self._state, obs).compile()
        self.compile_s = time.perf_counter() - t0
        return self.compile_s

    def __call__(self, obs):
        import jax

        from openpi.shared import array_typing as at

        if self._compiled is None:
            self.compile(obs)
        with at.disable_typechecking():
            out = self._compiled(self._state, obs)
        return jax.block_until_ready(out)

    def llm_params(self) -> dict:
        state = self._nnx.state(self._nnx.merge(self._graphdef, self._state), self._nnx.Param)
        return state.to_pure_dict()["PaliGemma"]["llm"]


def parse_layers(spec: Sequence[str], depth: int) -> list[int]:
    if len(spec) == 1 and spec[0] == "all":
        return list(range(depth + 1))
    layers = sorted(set(int(s) for s in spec))
    if not layers or layers[0] < 0 or layers[-1] > depth:
        raise SystemExit(f"--layers must be 'all' or integers in 0..{depth}")
    return layers


# ------------------------------------------------------------------------------------ T77 summary pooling

#: Row order of ``--pool summary`` (one row per layer each).
SUMMARY_ROWS = (*POLICY_CAMERA_NAMES, "text_mean", "text_last")
N_IMAGE_TOKENS = LANGUAGE_TOKEN_START  # 3 x 256 camera tokens precede the text
KV_LAYER = 4  # kv_L4 (SUBTASK-b / serving HSR probe feature)


def summary_rows(xp, hidden, mask):
    """``(rows [L, 5, D], last_text_index)`` from ``hidden [L, T, D]`` (float32) and the prefix mask ``[T]``.

    Rows: per camera the mean over its valid tokens (``CAMERA_BINDINGS`` ranges), the mean over valid text
    tokens, and the hidden at the **last valid prefix position** (images are always valid and the text mask is
    contiguous from position 768, so this is the last valid text token). `xp` = numpy or jax.numpy -- the same
    code runs on the host (tests, reference) and inside the jitted pass.
    """
    mask = xp.asarray(mask)
    mf = mask.astype(hidden.dtype)
    rows = []
    for cam in POLICY_CAMERA_NAMES:
        lo, hi = CAMERA_BINDINGS[cam][2]
        m = mf[lo:hi]
        rows.append((hidden[:, lo:hi] * m[None, :, None]).sum(1) / m.sum())
    tm = mf[N_IMAGE_TOKENS:]
    rows.append((hidden[:, N_IMAGE_TOKENS:] * tm[None, :, None]).sum(1) / tm.sum())
    total = hidden.shape[1]
    last = total - 1 - xp.argmax(mask[::-1].astype(xp.int32))  # last True position
    rows.append(hidden[:, last])
    return xp.stack(rows, axis=1), last


def kv_l4_feature(xp, v_layer, mask):
    """SUBTASK-b ``kv_L{l}``: ``v_layer [T, 256]`` (one KV head) -> ``[mean over valid text | mean over the 768
    image tokens]`` (512,), float32 -- `extract.py:_prefix` verbatim (image mean unmasked, text mean masked)."""
    v = v_layer.astype(xp.float32)
    tm = xp.asarray(mask)[N_IMAGE_TOKENS:].astype(xp.float32)
    txt = (v[N_IMAGE_TOKENS:] * tm[:, None]).sum(0) / tm.sum()
    img = v[:N_IMAGE_TOKENS].mean(0)
    return xp.concatenate([txt, img])


class SummaryExtractor(HiddenExtractor):
    """`HiddenExtractor` whose jitted pass also pools on the device:
    ``(rows [L+1, 5, D] f32, kv_L4 [512] f32, absmax [L+1] f32, last_text_index, mask [T])`` for batch row 0.
    The full outputs (for the self-check) stay available through ``__call__`` (compiled separately)."""

    def __init__(self, model, kv_layer: int = KV_LAYER):
        import jax

        super().__init__(model)
        self.kv_layer = int(kv_layer)
        self._sjit = jax.jit(self._summary)
        self._scompiled = None
        self.summary_compile_s = None

    def _summary(self, state, obs):
        import jax.numpy as jnp

        _, kv_v, hidden, mask = self._prefix(state, obs)
        h = hidden[:, 0].astype(jnp.float32)  # [L+1, T, D]
        m = mask[0]
        rows, last = summary_rows(jnp, h, m)
        kv4 = kv_l4_feature(jnp, kv_v[self.kv_layer, 0, :, 0, :], m)
        absmax = jnp.abs(h).max(axis=(1, 2))
        return rows, kv4, absmax, last, m

    def summary(self, obs):
        import jax

        from openpi.shared import array_typing as at

        with at.disable_typechecking():
            if self._scompiled is None:
                t0 = time.perf_counter()
                self._scompiled = self._sjit.lower(self._state, obs).compile()
                self.summary_compile_s = time.perf_counter() - t0
            out = self._scompiled(self._state, obs)
        return jax.block_until_ready(out)


def parse_episodes(spec: Sequence[str]) -> list[int]:
    """``--episodes``: integers and/or ``lo:hi[:step]`` ranges (Python slice semantics), deduplicated, in order."""
    out: list[int] = []
    for s in spec:
        if ":" in s:
            parts = [int(p) for p in s.split(":")]
            out.extend(range(*parts))
        else:
            out.append(int(s))
    seen, uniq = set(), []
    for e in out:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq


def dataset_episode_frames(root: str | pathlib.Path, episode: int, frame_stride: int) -> tuple[list[Frame], dict]:
    """Expert episode -> frames, SUBTASK-b `extract.py:run_dataset` reading (parquet + three mp4, frames with
    ``frame_index % frame_stride == 0`` inside the shortest video, state as float32, prompt from tasks.jsonl).
    Returns ``(frames, info)`` with ``info`` = ``phase_index`` / ``frame_index`` arrays and episode metadata."""
    import av
    import pandas as pd

    root = pathlib.Path(root)
    tasks = {json.loads(line)["task_index"]: json.loads(line)["task"]
             for line in (root / "meta" / "tasks.jsonl").read_text().splitlines() if line.strip()}
    ch = episode // 1000
    df = pd.read_parquet(root / f"data/chunk-{ch:03d}/episode_{episode:06d}.parquet")

    def decode(path):
        with av.open(str(path)) as c:
            return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]

    vids = {c: decode(root / f"videos/chunk-{ch:03d}/observation.images.{c}/episode_{episode:06d}.mp4")
            for c in POLICY_CAMERA_NAMES}
    nvid = min(len(v) for v in vids.values())
    states = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    fidx = df["frame_index"].to_numpy()
    phase = df["phase_index"].to_numpy()
    prompt = tasks[int(df["task_index"].iloc[0])]
    frames, ph, fi_out, skipped = [], [], [], []
    for i in range(len(df)):
        fi = int(fidx[i])
        if fi % frame_stride:
            continue
        if fi >= nvid:
            skipped.append(fi)
            continue
        frames.append(Frame(frame_id=f"ep{episode}@f{fi}", t_step=fi, source=f"episode_{episode:06d}",
                            state=np.asarray(states[i], np.float32), images={c: vids[c][fi] for c in POLICY_CAMERA_NAMES},
                            prompt=prompt))
        ph.append(int(phase[i]))
        fi_out.append(fi)
    info = {"phase_index": np.asarray(ph, np.int64), "frame_index": np.asarray(fi_out, np.int64),
            "episode": int(episode), "prompt": prompt, "n_parquet": int(len(df)), "n_video": int(nvid),
            "frames_beyond_video": skipped}
    return frames, info


def safe_name(label: str) -> str:
    """Spec label -> file stem (keeps letters incl. Hangul, digits, ``-_.``; everything else -> ``_``)."""
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(label))


def peak_rss_mib() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ------------------------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", nargs="+", default=None,
                    help="AG3S run_XXXX/ (meta.json + step_*.npz) or OpenPI policy_records/session_*/ "
                         "(several only with --pool summary)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "gpu"),
                    help="cpu (default): JAX_PLATFORMS=cpu + CPU pinning, as T73. gpu: no forcing -- run inside "
                         "scripts/setup/gpu_lock.py shared (T77)")
    ap.add_argument("--dataset", default=None,
                    help="T77: LeRobot dataset root (expert parquet + mp4); needs --pool summary and --episodes")
    ap.add_argument("--episodes", nargs="+", default=None,
                    help="T77: episode numbers and/or lo:hi[:step] ranges (with --dataset)")
    ap.add_argument("--frame-stride", type=int, default=6, help="T77: every N-th frame_index (with --dataset)")
    ap.add_argument("--spec", default=None,
                    help="T77: t77_labels spec.json; extracts every entry's client_dir/ag3s/run_0000 "
                         "(with --pool summary)")
    ap.add_argument("--only", nargs="+", default=None, help="T77: with --spec, only these labels")
    ap.add_argument("--out-dir", default=None,
                    help="T77 --pool summary: one npz per episode / run here (default $AG3S_RUN_DIR/hidden_summary)")
    ap.add_argument("--frames", type=int, nargs="+", default=None,
                    help="positions in the sorted step files (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N frames (after --frames)")
    ap.add_argument("--checkpoint", default=None,
                    help=f"default: $AG3S_STORAGE_ROOT/{DEFAULT_CHECKPOINT_REL}")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--layers", nargs="+", default=["all"], help="'all' (0..18) or layer numbers")
    ap.add_argument("--tokens", default="all", choices=("all", "image", "text"))
    ap.add_argument("--pool", default="none", choices=("none", "mean", "summary", "patches"))
    ap.add_argument("--patch-dtype", default="float32", choices=("float32", "float16"),
                    help="T79 --pool patches storage dtype (float16 halves the size; aborts if |h| > 65504)")
    ap.add_argument("--requests", default=None,
                    help="T79 --pool patches: json with {label: [request, ...]} (or t79_events.json, key "
                         "'patch_requests'); only these requests are extracted")
    ap.add_argument("--cameras", nargs="+", default=None, choices=POLICY_CAMERA_NAMES,
                    help="T86 --pool patches: cameras to store (default: head only = the T79 layout)")
    ap.add_argument("--prompt-override", default=None,
                    help="T86 --pool patches: json {label: {request: [prompt, ...]}} -- extra forward(s) of the same "
                         "image with another prompt")
    ap.add_argument("--noun-attention", nargs="?", const="10:7", default=None, metavar="L:H",
                    help="T86 --pool patches: also store the noun-token attention row (layer:head, default 10:7)")
    ap.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    ap.add_argument("--prompt", default=None, help="override the recorded prompt")
    ap.add_argument("--threads", type=int, default=16, help="CPU threads (pins the process to this many CPUs)")
    ap.add_argument("--no-check", action="store_true", help="skip the V-cache / final-norm self-checks")
    ap.add_argument("--out", default=None, help="output .npz (default $AG3S_RUN_DIR/hidden/<record>.npz)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.pool != "patches" and (args.cameras or args.prompt_override or args.noun_attention):
        raise SystemExit("--cameras / --prompt-override / --noun-attention need --pool patches")
    if args.pool == "summary":
        return main_summary(args)
    if args.pool == "patches":
        return main_patches(args)
    if args.dataset or args.spec or args.episodes or args.requests:
        raise SystemExit("--dataset / --spec / --episodes need --pool summary (--requests: --pool patches)")
    if not args.records or len(args.records) != 1:
        raise SystemExit("--records takes exactly one record directory (several only with --pool summary)")
    args.records = args.records[0]
    if args.device == "gpu":
        cpus = setup_gpu()
    else:
        if args.threads > 16:
            print(f"[pi05_hidden] note: --threads {args.threads} > 16 (project limit is 16)")
        cpus = force_cpu(args.threads)

    kind, files, meta_prompt = list_frames(args.records)
    picks = list(range(len(files))) if args.frames is None else list(args.frames)
    if args.limit is not None:
        picks = picks[: int(args.limit)]
    bad = [p for p in picks if not 0 <= p < len(files)]
    if bad:
        raise SystemExit(f"--frames out of range 0..{len(files) - 1}: {bad}")
    frames = [read_frame(kind, files[p], meta_prompt) for p in picks]

    storage = os.environ.get("AG3S_STORAGE_ROOT")
    ckpt = args.checkpoint or (f"{storage}/{DEFAULT_CHECKPOINT_REL}" if storage else None)
    if ckpt is None:
        raise SystemExit("--checkpoint not given and AG3S_STORAGE_ROOT unset (source scripts/setup/env.sh)")
    if args.out:
        out = pathlib.Path(args.out)
    else:
        run_dir = os.environ.get("AG3S_RUN_DIR")
        if not run_dir:
            raise SystemExit("--out not given and AG3S_RUN_DIR unset "
                             "(export AG3S_RUN_DIR=\"$(.venv-openpi-live/bin/python -m common.run_output)\")")
        name = pathlib.Path(args.records).resolve()
        out = pathlib.Path(run_dir) / "hidden" / f"{name.parent.name}_{name.name}.npz"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")

    import jax  # noqa: E402 -- after force_cpu

    check_backend(args.device)
    print(f"[pi05_hidden] {kind} record, {len(frames)} frames, backend={jax.default_backend()}, "
          f"cpus={len(cpus)}, checkpoint={ckpt}")

    from openpi.models import gemma as _gemma
    from openpi.models.tokenizer import PaligemmaTokenizer

    from AG3S.ag3s.experiments.sources.pi05_attention import observation_from_step

    t0 = time.perf_counter()
    policy, train_cfg = load_hidden_policy(args.config, ckpt)
    load_s = time.perf_counter() - t0
    rss_after_load = peak_rss_mib()
    vlm_cfg = _gemma.get_config(train_cfg.model.paligemma_variant)
    depth = vlm_cfg.depth
    layers = parse_layers(args.layers, depth)
    sp = PaligemmaTokenizer(train_cfg.model.max_token_len)._tokenizer  # noqa: SLF001
    extractor = HiddenExtractor(policy._model)  # noqa: SLF001
    llm_params = None if args.no_check else extractor.llm_params()
    print(f"[pi05_hidden] model loaded in {load_s:.1f}s (peak RSS {rss_after_load / 1024:.2f} GiB), "
          f"layers={layers}")

    n_f = len(frames)
    store = None
    rec: dict[str, list] = {k: [] for k in ("prefix_mask", "text_token_ids", "text_token_pieces", "absmax",
                                            "check_v_relerr", "check_final_norm_relerr", "frame_s",
                                            "token_valid")}
    for fi, fr in enumerate(frames):
        prompt = args.prompt or fr.prompt
        if prompt is None:
            raise SystemExit(f"{fr.source}: no prompt recorded; pass --prompt")
        obs = observation_from_step(policy, fr, prompt)
        if extractor.compile_s is None:
            print(f"[pi05_hidden] compiling prefix pass ...")
            extractor.compile(obs)
            print(f"[pi05_hidden] compiled in {extractor.compile_s:.1f}s")
        t1 = time.perf_counter()
        prefix_out, kv_v, hidden, mask = extractor(obs)
        frame_s = time.perf_counter() - t1
        h32 = np.asarray(hidden[:, 0], np.float32)  # [L+1, T, D]
        pmask = np.asarray(mask[0], bool)
        absmax = np.abs(h32).max(axis=(1, 2))
        if args.dtype == "float16" and float(absmax[layers].max()) > FLOAT16_MAX:
            raise SystemExit(f"{fr.frame_id}: |h| up to {absmax.max():.0f} > float16 max; use --dtype float32")
        if args.no_check:
            v_err, fn_err = np.full(depth, np.nan), float("nan")
        else:
            v_err, fn_err = self_check(llm_params, hidden, kv_v, prefix_out, vlm_cfg)
        stored, valid = select_and_pool(h32, pmask, layers, args.tokens, args.pool)
        if store is None:
            store = np.empty((n_f, *stored.shape), args.dtype)
            print(f"[pi05_hidden] hidden per frame {stored.shape} -> {store.nbytes / 2**20:.1f} MiB total")
        store[fi] = stored.astype(args.dtype)
        tok = np.asarray(obs.tokenized_prompt[0], np.int64)
        rec["prefix_mask"].append(pmask)
        rec["text_token_ids"].append(tok)
        rec["text_token_pieces"].append(np.asarray([sp.id_to_piece(int(t)) for t in tok]))
        rec["absmax"].append(absmax.astype(np.float32))
        rec["check_v_relerr"].append(np.asarray(v_err, np.float64))
        rec["check_final_norm_relerr"].append(fn_err)
        rec["frame_s"].append(frame_s)
        rec["token_valid"].append(valid)
        print(f"  frame {fi + 1}/{n_f} {fr.frame_id}: {frame_s:.1f}s, V relerr max "
              f"{np.nanmax(v_err) if not args.no_check else float('nan'):.2e}, final-norm relerr {fn_err:.2e}, "
              f"peak RSS {peak_rss_mib() / 1024:.2f} GiB")

    total = len(rec["prefix_mask"][0])
    idx, grp = token_selection(args.tokens, total)
    cam_ranges = np.asarray([CAMERA_BINDINGS[c][2] for c in POLICY_CAMERA_NAMES], np.int64)
    prov = code_provenance()
    meta = {
        "step": "T73", "records": str(pathlib.Path(args.records).resolve()), "record_kind": kind,
        "checkpoint": str(ckpt), "config": args.config, "prompt": args.prompt or frames[0].prompt,
        "layers": layers, "tokens": args.tokens, "pool": args.pool, "dtype": args.dtype,
        "pool_groups": pool_groups(args.tokens) if args.pool == "mean" else None,
        "frame_id": [f.frame_id for f in frames], "frames": picks, "threads": len(cpus),
        "load_s": load_s, "compile_s": extractor.compile_s, "frame_s": rec["frame_s"],
        "peak_rss_mib": peak_rss_mib(), "rss_after_load_mib": rss_after_load,
        "check_v_relerr_max": None if args.no_check else float(np.max(rec["check_v_relerr"])),
        "check_final_norm_relerr_max": None if args.no_check else float(np.max(rec["check_final_norm_relerr"])),
        "argv": sys.argv, **prov,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        hidden=store,
        layers=np.asarray(layers, np.int64),
        token_index=idx if args.pool == "none" else np.zeros(0, np.int64),
        token_group=grp if args.pool == "none" else np.zeros(0, np.int64),
        token_valid=np.stack(rec["token_valid"]),
        pool_groups=np.asarray(pool_groups(args.tokens) if args.pool == "mean" else [], dtype=str),
        camera_names=np.asarray(POLICY_CAMERA_NAMES),
        camera_token_ranges=cam_ranges,
        text_token_range=np.asarray([LANGUAGE_TOKEN_START, total], np.int64),
        prefix_mask=np.stack(rec["prefix_mask"]),
        text_token_ids=np.stack(rec["text_token_ids"]),
        text_token_pieces=np.stack(rec["text_token_pieces"]),
        frame_id=np.asarray([f.frame_id for f in frames]),
        t_step=np.asarray([f.t_step for f in frames], np.int64),
        source_files=np.asarray([f.source for f in frames]),
        prompt=np.asarray(meta["prompt"]),
        checkpoint=np.asarray(str(ckpt)),
        config=np.asarray(args.config),
        code_commit=np.asarray(prov["code_commit"]),
        code_dirty=np.asarray(prov["code_dirty"]),
        absmax=np.stack(rec["absmax"]),
        check_v_relerr=np.stack(rec["check_v_relerr"]),
        check_final_norm_relerr=np.asarray(rec["check_final_norm_relerr"], np.float64),
        compile_s=np.asarray(extractor.compile_s, np.float64),
        frame_s=np.asarray(rec["frame_s"], np.float64),
    )
    meta["out"] = str(out)
    meta["out_bytes"] = out.stat().st_size
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"[pi05_hidden] wrote {out} hidden={store.shape} {args.dtype} ({meta['out_bytes'] / 2**20:.1f} MiB); "
          f"compile {extractor.compile_s:.1f}s, frame {np.mean(rec['frame_s']):.1f}s, "
          f"peak RSS {meta['peak_rss_mib'] / 1024:.2f} GiB")


# ------------------------------------------------------------------------------------ T77 --pool summary


def summary_jobs(args) -> list[dict]:
    """Work items ``{name, kind, ...}`` of a ``--pool summary`` run (one output file each)."""
    jobs = []
    if args.dataset:
        if not args.episodes:
            raise SystemExit("--dataset needs --episodes")
        for ep in parse_episodes(args.episodes):
            jobs.append({"name": f"ep_{ep:06d}", "kind": "dataset", "episode": ep})
    if args.spec:
        spec = json.loads(pathlib.Path(args.spec).read_text())
        entries = spec["runs"] if isinstance(spec, dict) else spec
        for e in entries:
            if args.only and e["label"] not in args.only:
                continue
            jobs.append({"name": safe_name(e["label"]), "kind": "ag3s", "label": e["label"],
                         "records": str(pathlib.Path(e["client_dir"]) / "ag3s" / "run_0000")})
    for r in args.records or []:
        rp = pathlib.Path(r).resolve()
        jobs.append({"name": safe_name(f"{rp.parent.name}_{rp.name}"), "kind": "records", "label": None,
                     "records": str(rp)})
    names = [j["name"] for j in jobs]
    if len(set(names)) != len(names):
        raise SystemExit("output names collide; check --records / --spec labels")
    if not jobs:
        raise SystemExit("--pool summary needs --dataset/--episodes, --spec or --records")
    return jobs


def _job_frames(job: dict, args) -> tuple[list[Frame], dict]:
    if job["kind"] == "dataset":
        return dataset_episode_frames(args.dataset, job["episode"], args.frame_stride)
    kind, files, meta_prompt = list_frames(job["records"])
    frames = [read_frame(kind, f, meta_prompt) for f in files]
    if args.limit is not None:
        frames = frames[: int(args.limit)]
    return frames, {"records": job["records"], "label": job.get("label"), "record_kind": kind}


def main_summary(args) -> None:
    """T77: summary rows + kv_L4 per frame, one npz per episode / run (resumable)."""
    if args.out:
        raise SystemExit("--pool summary writes one file per episode / run: use --out-dir, not --out")
    if args.device == "gpu":
        cpus = setup_gpu()
    else:
        if args.threads > 16:
            print(f"[pi05_hidden] note: --threads {args.threads} > 16 (project limit is 16)")
        cpus = force_cpu(args.threads)
    jobs = summary_jobs(args)
    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
    else:
        run_dir = os.environ.get("AG3S_RUN_DIR")
        if not run_dir:
            raise SystemExit("--out-dir not given and AG3S_RUN_DIR unset")
        out_dir = pathlib.Path(run_dir) / "hidden_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out_dir / f"{j['name']}.npz").exists()]
    print(f"[pi05_hidden] summary: {len(jobs)} files, {len(jobs) - len(todo)} exist (skipped), {len(todo)} to do; "
          f"out {out_dir}")
    if not todo:
        return
    storage = os.environ.get("AG3S_STORAGE_ROOT")
    ckpt = args.checkpoint or (f"{storage}/{DEFAULT_CHECKPOINT_REL}" if storage else None)
    if ckpt is None:
        raise SystemExit("--checkpoint not given and AG3S_STORAGE_ROOT unset (source scripts/setup/env.sh)")

    import jax  # noqa: E402,F401 -- after force_cpu / setup_gpu

    check_backend(args.device)
    from openpi.models import gemma as _gemma

    from AG3S.ag3s.experiments.sources.pi05_attention import observation_from_step

    t0 = time.perf_counter()
    policy, train_cfg = load_hidden_policy(args.config, ckpt)
    load_s = time.perf_counter() - t0
    vlm_cfg = _gemma.get_config(train_cfg.model.paligemma_variant)
    layers = parse_layers(args.layers, vlm_cfg.depth)
    extractor = SummaryExtractor(policy._model)  # noqa: SLF001
    llm_params = None if args.no_check else extractor.llm_params()
    prov = code_provenance()
    print(f"[pi05_hidden] backend={jax.default_backend()} model loaded in {load_s:.1f}s, layers={layers}")

    for ji, job in enumerate(todo):
        out = out_dir / f"{job['name']}.npz"
        tj = time.perf_counter()
        frames, info = _job_frames(job, args)
        read_s = time.perf_counter() - tj
        if not frames:
            print(f"  [{ji + 1}/{len(todo)}] {job['name']}: no frames, skipped")
            continue
        rows, kv4, absmax, last_idx, n_text, frame_s, prompts = [], [], [], [], [], [], []
        v_err, fn_err = np.full(vlm_cfg.depth, np.nan), float("nan")
        for fi, fr in enumerate(frames):
            prompt = args.prompt or fr.prompt
            if prompt is None:
                raise SystemExit(f"{fr.source}: no prompt recorded; pass --prompt")
            obs = observation_from_step(policy, fr, prompt)
            if fi == 0 and not args.no_check:
                prefix_out, kv_v, hidden, _ = extractor(obs)
                v_err, fn_err = self_check(llm_params, hidden, kv_v, prefix_out, vlm_cfg)
                del prefix_out, kv_v, hidden
            if extractor.summary_compile_s is None:
                extractor.summary(obs)  # compile + warm-up, excluded from frame_s
                print(f"[pi05_hidden] summary pass compiled in {extractor.summary_compile_s:.1f}s")
            t1 = time.perf_counter()
            r, k, a, last, m = extractor.summary(obs)
            frame_s.append(time.perf_counter() - t1)
            r = np.asarray(r, np.float32)
            rows.append(r[np.asarray(layers)])
            kv4.append(np.asarray(k, np.float32))
            absmax.append(np.asarray(a, np.float32))
            last_idx.append(int(last))
            n_text.append(int(np.asarray(m)[N_IMAGE_TOKENS:].sum()))
            prompts.append(prompt)
        if not np.all(np.isfinite(np.stack(rows))):
            raise SystemExit(f"{job['name']}: non-finite pooled hidden")
        arrs = dict(
            hidden_summary=np.stack(rows).astype(np.float32),
            kv_L4=np.stack(kv4).astype(np.float32),
            layers=np.asarray(layers, np.int64),
            summary_rows=np.asarray(SUMMARY_ROWS),
            last_text_index=np.asarray(last_idx, np.int64),
            n_valid_text=np.asarray(n_text, np.int64),
            absmax=np.stack(absmax),
            t_step=np.asarray([f.t_step for f in frames], np.int64),
            frame_id=np.asarray([f.frame_id for f in frames]),
            state=np.stack([np.asarray(f.state, np.float64) for f in frames]),
            prompt=np.asarray(prompts[0]),
            prompts=np.asarray(prompts),
            check_v_relerr=np.asarray(v_err, np.float64),
            check_final_norm_relerr=np.asarray(fn_err, np.float64),
            frame_s=np.asarray(frame_s, np.float64),
            device=np.asarray(args.device),
            checkpoint=np.asarray(str(ckpt)),
            config=np.asarray(args.config),
            code_commit=np.asarray(prov["code_commit"]),
            code_dirty=np.asarray(prov["code_dirty"]),
        )
        if job["kind"] == "dataset":
            arrs.update(phase_index=info["phase_index"], frame_index=info["frame_index"],
                        episode=np.asarray(info["episode"], np.int64), n_parquet=np.asarray(info["n_parquet"]),
                        n_video=np.asarray(info["n_video"]))
        else:
            arrs.update(source_files=np.asarray([f.source for f in frames]), records=np.asarray(info["records"]),
                        label=np.asarray(info["label"] or job["name"]))
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, **arrs)
        tmp.rename(out)
        print(f"  [{ji + 1}/{len(todo)}] {job['name']}: {len(frames)} frames, read {read_s:.1f}s, "
              f"frame {np.median(frame_s):.3f}s (median), {out.stat().st_size / 2**20:.1f} MiB, "
              f"V relerr max {np.nanmax(v_err) if not args.no_check else float('nan'):.2e}, "
              f"|h| max {np.max(np.stack(absmax)):.3g}", flush=True)
    print(f"[pi05_hidden] summary done ({len(todo)} files), peak RSS {peak_rss_mib() / 1024:.2f} GiB, cpus {len(cpus)}")


# ------------------------------------------------------------------------------------ T79 --pool patches

HEAD_CAMERA = POLICY_CAMERA_NAMES[0]  # cam_high = zed_left
GRID = 16
#: the sliced pass and the full pass are compiled separately (bfloat16 model): allow fusion-order rounding only
SLICE_RELERR_MAX = 1e-2


class PatchExtractor(HiddenExtractor):
    """`HiddenExtractor` whose jitted pass returns only the head camera's image tokens at the selected layers:
    ``(patches [Ls, 256, D] f32, absmax [Ls] f32)`` for batch row 0. ``layers`` are Python ints (static)."""

    def __init__(self, model, layers: Sequence[int], camera: str = HEAD_CAMERA):
        import jax

        super().__init__(model)
        self.layers = tuple(int(x) for x in layers)
        self.token_range = tuple(int(x) for x in CAMERA_BINDINGS[camera][2])
        self._pjit = jax.jit(self._patches)
        self._pcompiled = None
        self.patch_compile_s = None

    def _patches(self, state, obs):
        import jax.numpy as jnp

        _, _, hidden, _ = self._prefix(state, obs)
        lo, hi = self.token_range
        h = hidden[jnp.asarray(self.layers), 0, lo:hi].astype(jnp.float32)  # [Ls, 256, D]
        return h, jnp.abs(h).max(axis=(1, 2))

    def patches(self, obs):
        import jax

        from openpi.shared import array_typing as at

        with at.disable_typechecking():
            if self._pcompiled is None:
                t0 = time.perf_counter()
                self._pcompiled = self._pjit.lower(self._state, obs).compile()
                self.patch_compile_s = time.perf_counter() - t0
            out = self._pcompiled(self._state, obs)
        return jax.block_until_ready(out)


def load_request_lists(path: Optional[str]) -> Optional[dict[str, list[int]]]:
    """``--requests`` json -> {label: sorted unique request indices} (accepts a t79_events.json)."""
    if not path:
        return None
    d = json.loads(pathlib.Path(path).read_text())
    if isinstance(d, dict) and "patch_requests" in d:
        d = d["patch_requests"]
    return {str(k): sorted({int(x) for x in v}) for k, v in d.items()}


def request_files(records: str) -> tuple[str, list[pathlib.Path], Optional[str], list[int]]:
    """``(kind, step files sorted by t_step, prompt, t_steps)`` -- request j = the j-th entry (T77 pairing)."""
    kind, files, prompt = list_frames(records)
    if kind != "ag3s":
        raise SystemExit(f"{records}: --pool patches reads AG3S client records (run_XXXX) only")
    ts = []
    for f in files:
        with np.load(f) as z:
            ts.append(int(z["t_step"]))
    order = np.argsort(np.asarray(ts), kind="stable")
    return kind, [files[i] for i in order], prompt, [ts[i] for i in order]


def main_patches(args) -> None:
    """T79: head-camera image-token hidden of the selected requests, one npz per run (resumable)."""
    if args.out:
        raise SystemExit("--pool patches writes one file per run: use --out-dir, not --out")
    if args.dataset or args.episodes:
        raise SystemExit("--pool patches reads closed-loop records (--spec / --records), not --dataset")
    if t86_mode(args):
        return main_patches_t86(args)
    if args.device == "gpu":
        cpus = setup_gpu()
    else:
        if args.threads > 16:
            print(f"[pi05_hidden] note: --threads {args.threads} > 16 (project limit is 16)")
        cpus = force_cpu(args.threads)
    jobs = summary_jobs(args)
    wanted = load_request_lists(args.requests)
    if wanted is not None:
        jobs = [j for j in jobs if (j.get("label") or j["name"]) in wanted and wanted[j.get("label") or j["name"]]]
    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
    else:
        run_dir = os.environ.get("AG3S_RUN_DIR")
        if not run_dir:
            raise SystemExit("--out-dir not given and AG3S_RUN_DIR unset")
        out_dir = pathlib.Path(run_dir) / "hidden_patches"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out_dir / f"{j['name']}.npz").exists()]
    print(f"[pi05_hidden] patches: {len(jobs)} files, {len(jobs) - len(todo)} exist (skipped), {len(todo)} to do; "
          f"out {out_dir}")
    if not todo:
        return
    storage = os.environ.get("AG3S_STORAGE_ROOT")
    ckpt = args.checkpoint or (f"{storage}/{DEFAULT_CHECKPOINT_REL}" if storage else None)
    if ckpt is None:
        raise SystemExit("--checkpoint not given and AG3S_STORAGE_ROOT unset (source scripts/setup/env.sh)")

    import jax  # noqa: E402,F401 -- after force_cpu / setup_gpu

    check_backend(args.device)
    from openpi.models import gemma as _gemma

    from AG3S.ag3s.experiments.sources.pi05_attention import observation_from_step

    t0 = time.perf_counter()
    policy, train_cfg = load_hidden_policy(args.config, ckpt)
    load_s = time.perf_counter() - t0
    vlm_cfg = _gemma.get_config(train_cfg.model.paligemma_variant)
    layers = parse_layers(args.layers, vlm_cfg.depth)
    extractor = PatchExtractor(policy._model, layers)  # noqa: SLF001
    llm_params = None if args.no_check else extractor.llm_params()
    prov = code_provenance()
    print(f"[pi05_hidden] backend={jax.default_backend()} model loaded in {load_s:.1f}s, layers={layers}, "
          f"head tokens {extractor.token_range}")

    for ji, job in enumerate(todo):
        out = out_dir / f"{job['name']}.npz"
        tj = time.perf_counter()
        kind, files, meta_prompt, t_steps = request_files(job["records"])
        reqs = list(range(len(files))) if wanted is None else wanted[job.get("label") or job["name"]]
        bad = [r for r in reqs if not 0 <= r < len(files)]
        if bad:
            raise SystemExit(f"{job['name']}: requests out of range 0..{len(files) - 1}: {bad[:5]}")
        frames = [read_frame(kind, files[r], meta_prompt) for r in reqs]
        read_s = time.perf_counter() - tj
        store = np.empty((len(frames), len(layers), extractor.token_range[1] - extractor.token_range[0],
                          vlm_cfg.width), args.patch_dtype)
        absmax, frame_s, prompts = [], [], []
        v_err, fn_err, slice_err = np.full(vlm_cfg.depth, np.nan), float("nan"), float("nan")
        for fi, fr in enumerate(frames):
            prompt = args.prompt or fr.prompt
            if prompt is None:
                raise SystemExit(f"{fr.source}: no prompt recorded; pass --prompt")
            obs = observation_from_step(policy, fr, prompt)
            if fi == 0 and not args.no_check:
                prefix_out, kv_v, hidden, _ = extractor(obs)
                v_err, fn_err = self_check(llm_params, hidden, kv_v, prefix_out, vlm_cfg)
                full = np.asarray(hidden[np.asarray(layers), 0, extractor.token_range[0]:extractor.token_range[1]],
                                  np.float32)
                del prefix_out, kv_v, hidden
            if extractor.patch_compile_s is None:
                extractor.patches(obs)  # compile + warm-up, excluded from frame_s
                print(f"[pi05_hidden] patch pass compiled in {extractor.patch_compile_s:.1f}s")
            t1 = time.perf_counter()
            h, a = extractor.patches(obs)
            frame_s.append(time.perf_counter() - t1)
            if args.patch_dtype == "float16" and float(np.max(a)) > FLOAT16_MAX:
                raise SystemExit(f"{fr.frame_id}: |h| up to {float(np.max(a)):.0f} > float16 max; use float32")
            store[fi] = np.asarray(h, np.float32).astype(args.patch_dtype)
            absmax.append(np.asarray(a, np.float32))
            prompts.append(prompt)
            if fi == 0 and not args.no_check:
                slice_err = relerr(np.asarray(h, np.float32), full)  # sliced pass vs the full pass' slice
                if slice_err > SLICE_RELERR_MAX:
                    raise SystemExit(f"{job['name']}: sliced patch pass differs from the full pass ({slice_err:.2e})")
                del full
        if not np.all(np.isfinite(store)):
            raise SystemExit(f"{job['name']}: non-finite hidden")
        arrs = dict(
            patches=store, layers=np.asarray(layers, np.int64), request=np.asarray(reqs, np.int64),
            t_step=np.asarray([t_steps[r] for r in reqs], np.int64), n_requests=np.asarray(len(files), np.int64),
            frame_id=np.asarray([f.frame_id for f in frames]), source_files=np.asarray([f.source for f in frames]),
            state=np.stack([np.asarray(f.state, np.float64) for f in frames]),
            camera=np.asarray(HEAD_CAMERA), token_range=np.asarray(extractor.token_range, np.int64),
            grid=np.asarray(GRID, np.int64), absmax=np.stack(absmax), prompt=np.asarray(prompts[0]),
            check_v_relerr=np.asarray(v_err, np.float64), check_final_norm_relerr=np.asarray(fn_err, np.float64),
            check_slice_relerr=np.asarray(slice_err, np.float64),
            frame_s=np.asarray(frame_s, np.float64), device=np.asarray(args.device), checkpoint=np.asarray(str(ckpt)),
            config=np.asarray(args.config), code_commit=np.asarray(prov["code_commit"]),
            code_dirty=np.asarray(prov["code_dirty"]), records=np.asarray(job["records"]),
            label=np.asarray(job.get("label") or job["name"]),
        )
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, **arrs)
        tmp.rename(out)
        print(f"  [{ji + 1}/{len(todo)}] {job['name']}: {len(frames)}/{len(files)} requests, read {read_s:.1f}s, "
              f"frame {np.median(frame_s):.3f}s (median), {out.stat().st_size / 2**20:.1f} MiB "
              f"({out.stat().st_size / 2**20 / len(frames):.2f} MiB/request), "
              f"V relerr max {np.nanmax(v_err) if not args.no_check else float('nan'):.2e}, "
              f"slice relerr {slice_err:.2e}, |h| max {np.max(np.stack(absmax)):.3g}", flush=True)
    print(f"[pi05_hidden] patches done ({len(todo)} files), peak RSS {peak_rss_mib() / 1024:.2f} GiB, cpus {len(cpus)}")


# ------------------------------------------------------------------------------------ T86 --pool patches (3 cameras, prompts, noun row)

#: noun-attention cell default: serving ``noun_target:L10H7`` (`attention_source`).
NOUN_CELL = (10, 7)
#: image + text mass of one attention row must sum to ~1 (softmax over all columns, padding masked to ~0).
NOUN_MASS_TOL = 2e-2


def t86_mode(args) -> bool:
    """True when any T86 flag of ``--pool patches`` is given (else the T79 tool runs unchanged)."""
    return bool(args.cameras or args.prompt_override or args.noun_attention)


def parse_noun_cell(spec: Optional[str]) -> Optional[tuple[int, int]]:
    """``--noun-attention`` value ``"L:H"`` -> ``(layer, head)``; None -> None."""
    if spec is None:
        return None
    try:
        layer, head = (int(x) for x in str(spec).split(":"))
    except ValueError as e:
        raise SystemExit(f"--noun-attention takes LAYER:HEAD (e.g. 10:7), got {spec!r}") from e
    if not (0 <= layer < 18 and 0 <= head < 8):
        raise SystemExit(f"--noun-attention {spec!r} is outside 18 layers x 8 heads")
    return layer, head


def load_prompt_overrides(path: Optional[str]) -> dict[str, dict[int, list[str]]]:
    """``--prompt-override`` json -> ``{label: {request: [prompt, ...]}}``; accepts a ``t86_samples.json``
    (key ``prompt_override``). Requests are ints, prompts kept in the listed order."""
    if not path:
        return {}
    d = json.loads(pathlib.Path(path).read_text())
    if isinstance(d, dict) and "prompt_override" in d:
        d = d["prompt_override"]
    out: dict[str, dict[int, list[str]]] = {}
    for label, per in d.items():
        out[str(label)] = {int(r): [str(x) for x in ([ps] if isinstance(ps, str) else ps)] for r, ps in per.items()}
    return out


def probe_outputs(xp, hidden, probs, rows, layers, ranges, cell):
    """The traced body of `ProbeExtractor` (numpy / jax.numpy shared, tested without a model).

    `hidden` ``[L+1, B, T, D]``; `probs` ``[L, B, K, G, T, S]`` or None; `rows` ``[R]`` int prefix rows (the noun
    token rows); `layers` ints; `ranges` per camera ``(lo, hi)`` token ranges; `cell` ``(layer, head)`` or None.
    Returns ``{"h": [C, Ls, hi - lo, D] f32 (batch 0), "absmax": [Ls] f32, and with probs: "att": [R, 768],
    "image_mass": [R], "text_mass": [R]}`` (the row slices are `t49_noun_query_core.slice_probs`, as serving)."""
    idx = xp.asarray(tuple(int(x) for x in layers))
    h = xp.stack([hidden[idx, 0, lo:hi] for lo, hi in ranges]).astype(xp.float32)
    out: dict[str, Any] = {"h": h, "absmax": xp.abs(h).max(axis=(0, 2, 3))}
    if probs is not None and cell is not None:
        from AG3S.ag3s.experiments.studies.t49_noun_query_core import N_IMAGE_TOKENS, slice_probs

        sp = slice_probs(probs, rows, n_image=N_IMAGE_TOKENS, xp=xp)
        layer, head = cell
        out["att"] = sp["image"][int(layer), int(head)]
        out["image_mass"] = sp["image_mass"][int(layer), int(head)]
        out["text_mass"] = sp["text_mass"][int(layer), int(head)]
    return out


class ProbeExtractor(HiddenExtractor):
    """T86: `HiddenExtractor` whose jitted pass returns the selected cameras' image tokens at the selected layers and,
    with a noun cell, the noun-token attention row of the **same** forward (`probe_outputs`). The model must carry
    ``return_hidden_states`` (and ``return_attn_probs`` for a cell: `load_hidden_policy(..., attn_probs=True)`).
    ``__call__`` / ``_prefix`` keep the `HiddenExtractor` contract (the self-checks run through them)."""

    def __init__(self, model, layers: Sequence[int], cameras: Sequence[str], cell: Optional[tuple[int, int]] = None):
        import jax

        super().__init__(model)
        self.layers = tuple(int(x) for x in layers)
        self.cameras = tuple(cameras)
        self.ranges = tuple(tuple(int(x) for x in CAMERA_BINDINGS[c][2]) for c in self.cameras)
        self.cell = None if cell is None else (int(cell[0]), int(cell[1]))
        self._pjit = jax.jit(self._probe)
        self._pcompiled = None
        self.probe_compile_s = None

    def _forward(self, state, obs):
        import jax.numpy as jnp

        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask

        model = self._nnx.merge(self._graphdef, state)
        obs = _model.preprocess_observation(None, obs, train=False)  # exactly once, as sample_actions
        tokens, mask, ar_mask = model.embed_prefix(obs)
        attn = make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1
        out = model.PaliGemma.llm([tokens, None], mask=attn, positions=positions)
        want = 4 if self.cell is not None else 3
        if len(out) != want:
            raise ValueError(f"PaliGemma.llm returned {len(out)} outputs, expected {want} "
                             f"(hidden states{' + attention probabilities' if self.cell is not None else ''})")
        (prefix_out, _), kv = out[0], out[1]
        probs = out[2] if self.cell is not None else None
        return prefix_out, kv[1], out[-1], mask, probs

    def _prefix(self, state, obs):
        prefix_out, kv_v, hidden, mask, _ = self._forward(state, obs)
        return prefix_out, kv_v, hidden, mask

    def _probe(self, state, obs, rows):
        _, _, hidden, _, probs = self._forward(state, obs)
        import jax.numpy as jnp

        return probe_outputs(jnp, hidden, probs, rows, self.layers, self.ranges, self.cell)

    def probe(self, obs, rows):
        import jax
        import jax.numpy as jnp

        from openpi.shared import array_typing as at

        rows = jnp.asarray(np.asarray(rows, np.int32))
        with at.disable_typechecking():
            if self._pcompiled is None:
                t0 = time.perf_counter()
                self._pcompiled = self._pjit.lower(self._state, obs, rows).compile()
                self.probe_compile_s = time.perf_counter() - t0
            out = self._pcompiled(self._state, obs, rows)
        return jax.block_until_ready(out)


def main_patches_t86(args) -> None:
    """T86: selected cameras' image-token hidden (+ noun row) of the selected requests, with extra prompts, one npz per
    run (resumable). Reached from `main_patches` when a T86 flag is given."""
    cameras = tuple(args.cameras or (HEAD_CAMERA,))
    cell = parse_noun_cell(args.noun_attention)
    if args.device == "gpu":
        cpus = setup_gpu()
    else:
        if args.threads > 16:
            print(f"[pi05_hidden] note: --threads {args.threads} > 16 (project limit is 16)")
        cpus = force_cpu(args.threads)
    jobs = summary_jobs(args)
    wanted = load_request_lists(args.requests)
    overrides = load_prompt_overrides(args.prompt_override)
    if wanted is not None:
        jobs = [j for j in jobs if (j.get("label") or j["name"]) in wanted and wanted[j.get("label") or j["name"]]]
    if args.out_dir:
        out_dir = pathlib.Path(args.out_dir)
    else:
        run_dir = os.environ.get("AG3S_RUN_DIR")
        if not run_dir:
            raise SystemExit("--out-dir not given and AG3S_RUN_DIR unset")
        out_dir = pathlib.Path(run_dir) / "hidden_patches_t86"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out_dir / f"{j['name']}.npz").exists()]
    print(f"[pi05_hidden] T86 patches: {len(jobs)} files, {len(jobs) - len(todo)} exist (skipped), {len(todo)} to do; "
          f"cameras {list(cameras)}, noun cell {cell}, out {out_dir}")
    if not todo:
        return
    storage = os.environ.get("AG3S_STORAGE_ROOT")
    ckpt = args.checkpoint or (f"{storage}/{DEFAULT_CHECKPOINT_REL}" if storage else None)
    if ckpt is None:
        raise SystemExit("--checkpoint not given and AG3S_STORAGE_ROOT unset (source scripts/setup/env.sh)")

    import jax  # noqa: E402,F401 -- after force_cpu / setup_gpu

    check_backend(args.device)
    from openpi.models import gemma as _gemma

    from AG3S.ag3s.experiments.sources.pi05_attention import observation_from_step

    if cell is not None:
        from AG3S.trajopt.attention_source import NounTokens, noun_lookup

        tokens = NounTokens.from_config(args.config)
    t0 = time.perf_counter()
    policy, train_cfg = load_hidden_policy(args.config, ckpt, attn_probs=cell is not None)
    load_s = time.perf_counter() - t0
    vlm_cfg = _gemma.get_config(train_cfg.model.paligemma_variant)
    layers = parse_layers(args.layers, vlm_cfg.depth)
    extractor = ProbeExtractor(policy._model, layers, cameras, cell)  # noqa: SLF001
    llm_params = None if args.no_check else extractor.llm_params()
    prov = code_provenance()
    n_tok = extractor.ranges[0][1] - extractor.ranges[0][0]
    print(f"[pi05_hidden] backend={jax.default_backend()} model loaded in {load_s:.1f}s, layers={layers}, "
          f"camera tokens {extractor.ranges}")

    def forward(fr, prompt):
        """One forward of frame `fr` with `prompt` -> (h [C, Ls, 256, D] f32, absmax [Ls], noun dict, seconds)."""
        obs = observation_from_step(policy, fr, prompt)
        noun = {"ok": False, "token": -1, "att": np.full(768, np.nan, np.float32), "mass": np.full(2, np.nan)}
        rows = np.zeros(3, np.int32)
        if cell is not None:
            lk = noun_lookup({"tokenized_prompt": np.asarray(obs.tokenized_prompt)[0],
                              "tokenized_prompt_mask": np.asarray(obs.tokenized_prompt_mask)[0]}, prompt, tokens)
            noun["lookup"] = lk
            if lk.ok:
                rows = np.asarray(lk.rows, np.int32)
                noun["ok"], noun["token"] = True, int(lk.token_index)
        if extractor.probe_compile_s is None:
            extractor.probe(obs, rows)  # compile + warm-up, excluded from the timing
            print(f"[pi05_hidden] T86 probe pass compiled in {extractor.probe_compile_s:.1f}s")
        t1 = time.perf_counter()
        res = extractor.probe(obs, rows)
        dt = time.perf_counter() - t1
        if cell is not None and noun["ok"]:
            noun["att"] = np.asarray(res["att"], np.float32)[0]  # row 0 = the target noun
            noun["mass"] = np.asarray([res["image_mass"][0], res["text_mass"][0]], np.float64)
        return obs, np.asarray(res["h"], np.float32), np.asarray(res["absmax"], np.float32), noun, dt

    for ji, job in enumerate(todo):
        out = out_dir / f"{job['name']}.npz"
        tj = time.perf_counter()
        kind, files, meta_prompt, t_steps = request_files(job["records"])
        label = job.get("label") or job["name"]
        reqs = list(range(len(files))) if wanted is None else wanted[label]
        bad = [r for r in reqs if not 0 <= r < len(files)]
        if bad:
            raise SystemExit(f"{job['name']}: requests out of range 0..{len(files) - 1}: {bad[:5]}")
        extra = overrides.get(label, {})
        unknown = sorted(set(extra) - set(reqs))
        if unknown:
            raise SystemExit(f"{job['name']}: --prompt-override requests not in the extracted set: {unknown[:5]}")
        frames = [read_frame(kind, files[r], meta_prompt) for r in reqs]
        read_s = time.perf_counter() - tj
        n_cf = sum(len(extra.get(r, [])) for r in reqs)
        shape = (len(frames), len(cameras), len(layers), n_tok, vlm_cfg.width)
        store = np.empty(shape, args.patch_dtype)
        cf_store = np.empty((n_cf, *shape[1:]), args.patch_dtype)
        att = np.full((len(frames), 768), np.nan, np.float32)
        cf_att = np.full((n_cf, 768), np.nan, np.float32)
        ok, cf_ok = np.zeros(len(frames), bool), np.zeros(n_cf, bool)
        tok_i, cf_tok_i = np.full(len(frames), -1, np.int64), np.full(n_cf, -1, np.int64)
        mass, cf_mass = np.full((len(frames), 2), np.nan), np.full((n_cf, 2), np.nan)
        cf_request, cf_prompt = np.zeros(n_cf, np.int64), np.empty(n_cf, object)
        prompts, absmax, frame_s = [], [], []
        v_err, fn_err, slice_err, mass_err = np.full(vlm_cfg.depth, np.nan), float("nan"), float("nan"), float("nan")
        q = 0
        for fi, fr in enumerate(frames):
            prompt = args.prompt or fr.prompt
            if prompt is None:
                raise SystemExit(f"{fr.source}: no prompt recorded; pass --prompt")
            prompts.append(prompt)
            todo_prompts = [(None, prompt)] + [(k, p) for k, p in enumerate(extra.get(reqs[fi], []))]
            for k, pr in todo_prompts:
                obs, h, a, noun, dt = forward(fr, pr)
                if args.patch_dtype == "float16" and float(np.max(a)) > FLOAT16_MAX:
                    raise SystemExit(f"{fr.frame_id}: |h| up to {float(np.max(a)):.0f} > float16 max; use float32")
                if fi == 0 and k is None and not args.no_check:
                    prefix_out, kv_v, hidden, _ = extractor(obs)
                    v_err, fn_err = self_check(llm_params, hidden, kv_v, prefix_out, vlm_cfg)
                    full = np.stack([np.asarray(hidden[np.asarray(layers), 0, lo:hi], np.float32)
                                     for lo, hi in extractor.ranges])
                    del prefix_out, kv_v, hidden
                    slice_err = relerr(h, full)
                    if slice_err > SLICE_RELERR_MAX:
                        raise SystemExit(f"{job['name']}: probe pass differs from the full pass ({slice_err:.2e})")
                    if cell is not None and noun["ok"]:
                        mass_err = float(abs(noun["mass"].sum() - 1.0))
                        if mass_err > NOUN_MASS_TOL:
                            raise SystemExit(f"{job['name']}: noun row mass {noun['mass']} does not sum to 1")
                if k is None:
                    store[fi] = h.astype(args.patch_dtype)
                    att[fi], ok[fi], tok_i[fi], mass[fi] = noun["att"], noun["ok"], noun["token"], noun["mass"]
                    absmax.append(a)
                    frame_s.append(dt)
                else:
                    cf_store[q] = h.astype(args.patch_dtype)
                    cf_att[q], cf_ok[q], cf_tok_i[q], cf_mass[q] = noun["att"], noun["ok"], noun["token"], noun["mass"]
                    cf_request[q], cf_prompt[q] = fi, pr
                    q += 1
        if not (np.all(np.isfinite(store)) and np.all(np.isfinite(cf_store))):
            raise SystemExit(f"{job['name']}: non-finite hidden")
        arrs = dict(
            patches=store, cf_patches=cf_store, cf_request=cf_request, cf_prompt=np.asarray(cf_prompt.tolist(), dtype=str),
            noun_att=att, cf_noun_att=cf_att, noun_ok=ok, cf_noun_ok=cf_ok, noun_token_index=tok_i,
            cf_noun_token_index=cf_tok_i, noun_mass=mass, cf_noun_mass=cf_mass,
            noun_cell=np.asarray(cell if cell is not None else (-1, -1), np.int64),
            camera_names=np.asarray(cameras), layers=np.asarray(layers, np.int64), request=np.asarray(reqs, np.int64),
            t_step=np.asarray([t_steps[r] for r in reqs], np.int64), n_requests=np.asarray(len(files), np.int64),
            frame_id=np.asarray([f.frame_id for f in frames]), source_files=np.asarray([f.source for f in frames]),
            state=np.stack([np.asarray(f.state, np.float64) for f in frames]), grid=np.asarray(GRID, np.int64),
            absmax=np.stack(absmax), prompt=np.asarray(prompts), check_v_relerr=np.asarray(v_err, np.float64),
            check_final_norm_relerr=np.asarray(fn_err, np.float64), check_slice_relerr=np.asarray(slice_err, np.float64),
            check_noun_mass_err=np.asarray(mass_err, np.float64), frame_s=np.asarray(frame_s, np.float64),
            device=np.asarray(args.device), checkpoint=np.asarray(str(ckpt)), config=np.asarray(args.config),
            code_commit=np.asarray(prov["code_commit"]), code_dirty=np.asarray(prov["code_dirty"]),
            records=np.asarray(job["records"]), label=np.asarray(label),
        )
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez(tmp, **arrs)
        tmp.rename(out)
        nbytes = out.stat().st_size
        print(f"  [{ji + 1}/{len(todo)}] {job['name']}: {len(frames)}/{len(files)} requests + {n_cf} counterfactual, "
              f"read {read_s:.1f}s, frame {np.median(frame_s):.3f}s (median), {nbytes / 2**20:.1f} MiB "
              f"({nbytes / 2**20 / max(1, len(frames) + n_cf):.2f} MiB/forward), "
              f"V relerr max {np.nanmax(v_err) if not args.no_check else float('nan'):.2e}, "
              f"slice relerr {slice_err:.2e}, noun mass err {mass_err:.2e}, |h| max {np.max(np.stack(absmax)):.3g}",
              flush=True)
    print(f"[pi05_hidden] T86 patches done ({len(todo)} files), peak RSS {peak_rss_mib() / 1024:.2f} GiB, cpus {len(cpus)}")


if __name__ == "__main__":
    main()
