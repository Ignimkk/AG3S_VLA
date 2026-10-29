"""T51 LIBERO — the same protocol (P1–P3) on `pi05_libero`, from the KNOWS P0b collection.

    source scripts/setup/env.sh
    export AG3S_RUN_DIR=...                                   # the T51 run
    M=AG3S.ag3s.experiments.studies.t51_libero
    .venv-ag3s/bin/python -m $M --stage survey                # L0: what storage holds (CPU, seconds)
    .venv-ag3s/bin/python -m $M --stage gt                    # GT target coverage per scored step (CPU)
    # GPU (L1 approved 2026-10-12; NOT run by the implementer — the verifier, under the lock):
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m $M --stage extract
    .venv-ag3s/bin/python -m $M --stage filter                # P1 (label-free, both cameras)
    .venv-ag3s/bin/python -m $M --stage score2d               # P2 2D (agentview GT) = primary
    .venv-ag3s/bin/python -m $M --stage bootstrap             # P3 (task bootstrap; --unit episode)

L1 decision (`T51.task.md` 판정 보완, 2026-10-12): one attention pass of `pi05_libero` over the stored
P0b frames with this checkout's ``.venv-openpi-live`` and ``checkpoints/panda/pi05_libero``, **without**
the KNOWS reproduction check (``--init zeros`` stays available but is not part of the protocol and
writes to its own ``attention_zeros/``). Primary = 2D lift. P3 resamples **tasks** (the two episodes of
one LIBERO task share the scene layout), the LIBERO analogue of the RB-Y1 scene bootstrap.

Aggregation (``--aggregate``, 2026-10-12 집계 단위 수정): ``episode`` (default, the original outputs) ranks
on the unweighted mean over episodes and the task bootstrap averages the drawn tasks' episodes;
``task`` (`T51.task.md` 판정 보완 → 집계 단위: LIBERO point estimate and bootstrap by **task**) averages
the episodes of a task first, then the tasks (score2d ranking; bootstrap: mean of the drawn tasks'
means, needs ``--unit task``). Outputs carry ``__task`` (``score2d__task.json``,
``score2d_arrays__task.npz``, ``bootstrap__task.json``). With two finite episodes in every task both
estimators coincide; they differ when a task has a missing / all-NaN episode.

L0 finding (`--stage survey` re-derives it): ``$AG3S_STORAGE_ROOT/datasets/workspace/knows_p0b/`` holds the
40 P0b episodes of `Baseline/knows_vla/collect_p0b.py` (4 suites x 5 tasks x 2 episodes): per control step
the 224x224 agent-view and wrist images the policy saw, state, action, the 256x256 agent-view GT instance
segmentation and depth, and the BDDL goal objects. **No attention is stored** — `probe_p0b.py` computed it
on the fly (JAX forward of `pi05_libero`) and only printed hit rates (``datasets/workspace/p0b_sweep.log``,
denoise 0 · mean only). So P1–P3 cannot run on CPU from storage. What is missing is one attention forward
pass over the stored frames (`--stage extract`, GPU, `.venv-openpi-live`, checkpoint
``checkpoints/panda/pi05_libero``) — **not** a new LIBERO rollout and not the LIBERO venv.

Label and metric (LIBERO): GT from the stored segmentation, agent-view only (the wrist camera has no
segmentation in P0b). Target = the BDDL manipulated object(s) before the first gripper close, the
destination(s) after (the `probe_p0b` rule, `probe_p0b.py:207-239`). **No 3D metric**: there is no
AG3S grounding for the LIBERO scene, so the primary is the **2D lift** (the RB-Y1 2D preselection metric);
`peak_on_target` and the KNOWS hit (Eq. 3, K = 5, beta = -1; `probe_p0b._score`) are secondary — the KNOWS
hit at denoise 0 · mean · zero-initialised noise reproduces `p0b_sweep.log` and is the extraction's parity
check.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import re
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t47_head_selection as hs
from AG3S.ag3s.experiments.studies import t47b_metrics as bm
from AG3S.ag3s.experiments.studies import t51_protocol as pr

STEP = "T51"
P0B_DIR = "datasets/workspace/knows_p0b"
SWEEP_LOG = "datasets/workspace/p0b_sweep.log"
CHECKPOINT = "checkpoints/panda/pi05_libero"
CONFIG = "pi05_libero"
#: Prefix token slices: `LiberoInputs` puts the agent view in base_0_rgb and the wrist in
#: left_wrist_0_rgb (`third_party/openpi/src/openpi/policies/libero_policy.py:59-62`); right wrist is a
#: masked zero image.
CAMERAS = ("agentview", "wrist")
CAMERA_SLICES = {"agentview": (0, 256), "wrist": (256, 512)}
STRIDE = 4  # P0b `--stride 4` (1,479 scored steps)
RESIZE = 224
GRID = 16
PATCH = RESIZE // GRID  # 14
#: T47's 200 px at 640x480 as an image fraction, at 224x224: 200 / 307200 * 50176 = 32.7 -> 33 px.
MIN_TARGET_PX = 33
KNOWS_WINDOW = 5
KNOWS_BETA = -1.0
#: `probe_p0b.ROBOT_TOKENS` (`:31`): robot instances are never candidates.
ROBOT_TOKENS = ("Panda", "Mount", "Gripper", "Robot")
DENOISE_STEPS = (0,)
AGGS = ("last", "mean", "first")


def out_root() -> pathlib.Path:
    from common.run_output import resolve_output_path

    run_dir = os.environ.get("AG3S_RUN_DIR")
    if not run_dir:
        raise SystemExit("AG3S_RUN_DIR is not set (export it to the T51 run dir)")
    root = resolve_output_path("verify/T51/libero", run_dir=run_dir, default_name="verify/T51/libero")
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json(path: pathlib.Path, payload: Any) -> None:
    hs.write_json(path, bm.jsonable(payload))


def run_info(stage: str, args) -> dict:
    return {"step": STEP, "sub": "libero", "stage": stage, "argv": sys.argv,
            "args": {k: v for k, v in vars(args).items()},
            "started_kst": _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(),
            "git": hs.git_state(), "storage_root": str(hs.storage_root()),
            "python": sys.version.split()[0]}


def episodes(storage: Optional[pathlib.Path] = None) -> list[pathlib.Path]:
    root = (storage or hs.storage_root()) / P0B_DIR
    return sorted(root.glob("*.npz"))


# ------------------------------------------------------------------------------ GT (pure)


def is_robot(name: str) -> bool:
    return any(tok in name for tok in ROBOT_TOKENS)


def to_instance(name: str, names: Sequence[str]) -> Optional[str]:
    """A BDDL goal argument (often a *region*) -> the segmentation instance it belongs to: the
    longest instance name that prefixes it (`probe_p0b._to_instance`, `:37-46`)."""
    if name in names:
        return name
    cands = [n for n in names if name.startswith(n)]
    return max(cands, key=len) if cands else None


def coverage_and_area(seg_full: np.ndarray, obj_ids: Sequence[int], resize: int = RESIZE
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Per-object 16x16 patch coverage and area at `resize` px (`probe_p0b.coverage_and_area`,
    `:49-66`: nearest-neighbour 256 -> 224, exact for a square resize-with-pad)."""
    h = seg_full.shape[0]
    idx = (np.arange(resize) * h / resize).astype(np.int64)
    seg = np.asarray(seg_full)[np.ix_(idx, idx)]
    patch = resize // GRID
    cov = np.zeros((len(obj_ids), GRID, GRID), np.float32)
    area = np.zeros(len(obj_ids), np.float32)
    for k, oid in enumerate(obj_ids):
        m = (seg == oid).astype(np.float32)
        area[k] = m.sum()
        cov[k] = m.reshape(GRID, patch, GRID, patch).mean(axis=(1, 3))
    return cov, area


def episode_gt(d: dict, *, stride: int = STRIDE) -> dict[str, Any]:
    """Scored steps, per-object coverage / area and the phase target set of one P0b episode
    (`probe_p0b._one_episode`, `:196-239`, without the candidate shuffle — order is irrelevant to
    the coverage; the KNOWS hit breaks ties the `probe_p0b` way via `knows_hits(perm=...)`)."""
    id2name = {int(s.split(":", 1)[0]): s.split(":", 1)[1] for s in np.asarray(d["id2name"]).tolist()}
    obj_ids = [i for i in sorted(id2name) if i != 0 and not is_robot(id2name[i])]
    names = [id2name[i] for i in obj_ids]
    reach = sorted({j for n in np.asarray(d["manipulated"]).tolist() if (j := to_instance(str(n), names))})
    place = sorted({j for n in np.asarray(d["destinations"]).tolist() if (j := to_instance(str(n), names))})
    if not reach:
        raise ValueError(f"no goal object resolved from {list(d['manipulated'])}")
    if not place:
        place = reach
    keep = np.arange(0, len(np.asarray(d["t"])), int(stride))
    grip = np.asarray(d["action"])[keep, 6]
    closed = np.flatnonzero(grip > 0)
    grasp_t = int(closed[0]) if closed.size else len(keep)
    seg = np.asarray(d["seg_full"])
    cov = np.zeros((len(keep), len(obj_ids), GRID, GRID), np.float32)
    area = np.zeros((len(keep), len(obj_ids)), np.float32)
    for t, src in enumerate(keep):
        cov[t], area[t] = coverage_and_area(seg[src], obj_ids)
    tmask = np.zeros((len(keep), len(obj_ids)), bool)
    for t in range(len(keep)):
        for n in (reach if t < grasp_t else place):
            tmask[t, names.index(n)] = True
    target_cov = np.clip((cov * tmask[:, :, None, None]).sum(axis=1), 0.0, 1.0)
    target_area = (area * tmask).sum(axis=1)
    return {"keep": keep, "grasp_t": grasp_t, "names": names, "obj_ids": np.asarray(obj_ids),
            "reach": reach, "place": place, "cov": cov, "area": area, "target_mask": tmask,
            "target_cov": target_cov, "target_area": target_area,
            "chance": float(tmask.sum(axis=1).mean() / len(names))}


def knows_hits(mass: np.ndarray, area: np.ndarray, target_mask: np.ndarray, *,
               window: int = KNOWS_WINDOW, perm: Optional[np.ndarray] = None) -> np.ndarray:
    """KNOWS Eq. (3) hit per step (`probe_p0b._score`, `:242-273`, beta = -1): density =
    windowed sum of object mass / windowed sum of object area; hit = argmax object in the target set.
    `mass` ``(T, ..., O)``, `area` / `target_mask` ``(T, O)``. `perm` reorders the candidates before the
    argmax (P0b shuffled them with ``default_rng(0)`` so ties do not favour id 1)."""
    m = np.asarray(mass, np.float64)
    a = np.asarray(area, np.float64)
    tm = np.asarray(target_mask, bool)
    if perm is not None:
        m, a, tm = m[..., perm], a[:, perm], tm[:, perm]
    hits = np.zeros(m.shape[:-1], bool)
    for t in range(m.shape[0]):
        s0 = max(0, t - int(window) + 1)
        num = m[s0:t + 1].sum(axis=0)
        den = np.maximum(a[s0:t + 1].sum(axis=0), 1e-6)
        top = np.argmax(num / den, axis=-1)
        hits[t] = tm[t][top]
    return hits


# ------------------------------------------------------------------------------ stages


def stage_survey(args, root: pathlib.Path) -> dict:
    storage = hs.storage_root()
    eps = []
    total, scored = 0, 0
    for p in episodes(storage):
        import zipfile

        with zipfile.ZipFile(p) as z:
            members = [i.filename[:-4] for i in z.infolist()]
        with np.load(p, allow_pickle=False) as d:
            n = int(d["t"].shape[0])
            eps.append({"file": p.name, "suite": str(d["suite"]), "task_id": int(d["task_id"]),
                        "episode": int(d["episode"]), "success": bool(d["success"]),
                        "n_steps": n, "n_scored_stride4": len(range(0, n, STRIDE)),
                        "prompt": str(d["prompt"]), "keys": members,
                        "has_attention": any(re.search(r"att", k) for k in members),
                        "seg_full_shape": list(d["seg_full"].shape)})
        total += n
        scored += len(range(0, n, STRIDE))
    att_assets = sorted(str(p.relative_to(storage)) for p in (storage / "assets" / "attention").glob("*")
                        if "libero" in p.name.lower())
    ckpt = storage / CHECKPOINT
    sweep = storage / SWEEP_LOG
    sweep_info = None
    if sweep.is_file():
        txt = sweep.read_text()
        sweep_info = {"path": str(sweep),
                      "scored_steps": [int(x) for x in re.findall(r"scored steps: (\d+)", txt)],
                      "settings": re.findall(r"denoise=(\d+)\s+agg=(\w+)", txt),
                      "best_cell": re.findall(r"best cell overall: .*", txt)}
    payload = {**run_info("survey", args), "p0b_dir": str(storage / P0B_DIR),
               "n_episodes": len(eps), "n_steps_total": total, "n_scored_stride4": scored,
               "suites": sorted({e["suite"] for e in eps}),
               "any_attention_in_p0b": any(e["has_attention"] for e in eps),
               "libero_attention_assets": att_assets,
               "checkpoint": {"path": str(ckpt), "exists": ckpt.is_dir(),
                              "entries": sorted(p.name for p in ckpt.iterdir()) if ckpt.is_dir() else []},
               "sweep_log": sweep_info, "episodes": eps,
               "conclusion": ("P0b stores frames + agent-view GT segmentation + depth but no per-cell "
                              "attention; P1-P3 need one attention forward pass of pi05_libero over the "
                              "stored frames (GPU, .venv-openpi-live). No new LIBERO rollout is needed.")}
    write_json(root / "survey.json", payload)
    print(f"[survey] {len(eps)} episodes, {scored} scored steps (stride {STRIDE}); attention stored: "
          f"{payload['any_attention_in_p0b']}; checkpoint: {ckpt.is_dir()}")
    return payload


def stage_gt(args, root: pathlib.Path) -> dict:
    out = root / "gt"
    out.mkdir(parents=True, exist_ok=True)
    summ = {}
    for p in episodes():
        with np.load(p, allow_pickle=False) as d:
            g = episode_gt({k: d[k] for k in ("id2name", "manipulated", "destinations", "t",
                                              "action", "seg_full")}, stride=args.stride)
        np.savez_compressed(out / p.name, keep=g["keep"], grasp_t=g["grasp_t"],
                            names=np.asarray(g["names"]), obj_ids=g["obj_ids"],
                            reach=np.asarray(g["reach"]), place=np.asarray(g["place"]),
                            cov=g["cov"], area=g["area"], target_mask=g["target_mask"],
                            target_cov=g["target_cov"], target_area=g["target_area"])
        summ[p.stem] = {"n_scored": int(len(g["keep"])), "grasp_t": g["grasp_t"],
                        "reach": g["reach"], "place": g["place"], "chance": g["chance"],
                        "n_target_ge_min_px": int((g["target_area"] >= args.min_target_px).sum())}
    write_json(out / "gt.json", {**run_info("gt", args), "min_target_px": args.min_target_px,
                                 "episodes": summ})
    print(f"[gt] {len(summ)} episodes -> {out}")
    return summ


# ------------------------------------------------------------------------- GPU extraction


def libero_observation(policy, image, wrist, state, prompt):
    """`probe_p0b._observation` (`:89-103`): the LIBERO policy input transform + batch axis."""
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model

    inputs = policy._input_transform({  # noqa: SLF001
        "observation/image": image, "observation/wrist_image": wrist,
        "observation/state": state, "prompt": prompt})
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
    return _model.Observation.from_dict(inputs)


def zero_init_attention(sampler, obs, denoise_steps) -> dict:
    """`AttentionSampler._sample` with the Euler loop started from **zeros** (what `probe_p0b`
    did, `:160-171`) instead of Gaussian noise — only for the P0b parity check."""
    import jax.numpy as jnp

    from openpi.models import model as _model

    obs = _model.preprocess_observation(None, obs, train=False)
    kv, prefix_mask = sampler._jit_prefix(sampler._state, obs)  # noqa: SLF001
    wanted = set(int(s) for s in denoise_steps)
    dt = -1.0 / sampler._num_steps  # noqa: SLF001
    x_t = jnp.zeros((obs.state.shape[0], sampler._action_horizon, sampler._action_dim), jnp.float32)  # noqa: SLF001
    tm = jnp.asarray(1.0, jnp.float32)
    out = {}
    for step in range(sampler._num_steps):  # noqa: SLF001
        v_t, probs = sampler._jit_suffix(sampler._state, obs, x_t, tm, kv, prefix_mask)  # noqa: SLF001
        if step in wanted:
            out[step] = np.asarray(probs)
        x_t = x_t + dt * v_t
        tm = tm + dt
    return out


def blocks_to_grids(blocks: dict, denoise: Sequence[int], aggs: Sequence[str]) -> np.ndarray:
    """``{step: [L, heads, action_tokens, >=512]}`` -> ``(D, A, L, H, 2, 16, 16)`` (agent view, wrist)."""
    from AG3S.ag3s.experiments.sources.pi05_attention import AGGREGATIONS

    first = blocks[denoise[0]]
    n_l, n_h = first.shape[:2]
    g = np.zeros((len(denoise), len(aggs), n_l, n_h, len(CAMERAS), GRID, GRID), np.float32)
    for di, d in enumerate(denoise):
        for ai, a in enumerate(aggs):
            pooled = AGGREGATIONS[a](np.asarray(blocks[d]))
            for ci, c in enumerate(CAMERAS):
                lo, hi = CAMERA_SLICES[c]
                g[di, ai, :, :, ci] = pooled[..., lo:hi].reshape(n_l, n_h, GRID, GRID)
    return g


def attention_dir(root: pathlib.Path, init: str = "gaussian") -> pathlib.Path:
    """``attention/`` for the protocol (Gaussian start, seed 0 = the RB-Y1 extraction); the optional
    zero-start pass (the P0b probe's) never mixes with it."""
    return root / ("attention" if init == "gaussian" else f"attention_{init}")


def stage_extract(args, root: pathlib.Path) -> dict:
    """GPU. One npz per P0b episode under ``attention/``; resumable per episode.

    The sampler path is RB-Y1's: `pi05_attention.load_policy` (`pi05_attention.py:59-68`,
    ``return_attn_probs=True``) and `AttentionSampler.attention(obs, (0,), noise_seed=0)`
    (`:154-167` -> `_sample` `:178-206`); only the observation differs (`libero_observation`, the
    LIBERO input transform, `probe_p0b.py:89-103`) and the token slices (`CAMERA_SLICES`)."""
    for k, v in (("XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
                 ("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")):
        os.environ.setdefault(k, v)
    from AG3S.ag3s.experiments.studies.t49_noun_query_extract import gpu_compute_pids

    busy = gpu_compute_pids()
    if busy is None:
        raise SystemExit("nvidia-smi is unavailable — the extract stage needs the GPU")
    if busy and not args.allow_busy_gpu:
        raise SystemExit(f"GPU has compute processes {busy} — wait (never kill another session); "
                         "run under flock /tmp/ag3s-vla-gpu-validation.lock")
    import jax

    if jax.default_backend() != "gpu":
        raise SystemExit(f"jax backend is {jax.default_backend()!r}, not gpu")
    from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler, load_policy

    ckpt = hs.storage_root() / CHECKPOINT
    t_load = time.time()
    policy = load_policy(CONFIG, str(ckpt))
    load_s = time.time() - t_load
    sampler = AttentionSampler(policy._model, num_steps=10)  # noqa: SLF001
    out = attention_dir(root, args.init)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    eps = episodes()
    if args.episodes:
        want = set(args.episodes)
        eps = [p for p in eps if p.stem in want or p.name in want]
        if not eps:
            raise SystemExit(f"no P0b episode matches {sorted(want)}")
    for p in eps:
        dest = out / p.name
        if dest.is_file() and not args.force:
            continue
        with np.load(p, allow_pickle=False) as d:
            # read each array once: NpzFile decompresses the whole member on every d[key] access
            images, wrists, states = d["image"], d["wrist_image"], d["state"]
            keep = np.arange(0, int(d["t"].shape[0]), args.stride)
            prompt = str(d["prompt"])
            grids = []
            t0 = time.time()
            for src in keep:
                obs = libero_observation(policy, images[src], wrists[src], states[src], prompt)
                blocks = (zero_init_attention(sampler, obs, DENOISE_STEPS) if args.init == "zeros"
                          else sampler.attention(obs, DENOISE_STEPS, noise_seed=0))
                grids.append(blocks_to_grids(blocks, DENOISE_STEPS, AGGS))
        part = dest.with_name(dest.stem + ".partial.npz")
        np.savez_compressed(part, attention=np.stack(grids).astype(np.float16), src_index=keep,
                            denoise_steps=np.asarray(DENOISE_STEPS), aggregations=np.asarray(AGGS),
                            cameras=np.asarray(CAMERAS), prompt=np.asarray(prompt),
                            init=np.asarray(args.init), checkpoint=np.asarray(str(ckpt)),
                            episode=np.asarray(p.name), noise_seed=np.asarray(0))
        part.replace(dest)
        done.append({"episode": p.name, "steps": int(len(keep)), "seconds": time.time() - t0})
        print(f"[extract] {p.name}: {len(keep)} steps ({time.time() - t0:.0f}s)", flush=True)
    mem = None
    try:
        mem = {k: int(v) for k, v in (jax.devices()[0].memory_stats() or {}).items()
               if k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit")}
    except Exception:  # noqa: BLE001 - memory stats are optional
        pass
    write_json(out / f"extract_{args.init}.json", {**run_info("extract", args), "done": done,
                                                    "checkpoint": str(ckpt), "config": CONFIG,
                                                    "load_s": load_s, "memory": mem,
                                                    "jax_backend": jax.default_backend()})
    return {"done": done}


# ------------------------------------------------------------------------- CPU P1–P3


def load_episode(root: pathlib.Path, name: str) -> tuple[dict, dict]:
    att_p, gt_p = attention_dir(root) / name, root / "gt" / name
    if not att_p.is_file():
        raise SystemExit(f"no attention for {name} at {att_p} — --stage extract (GPU) first")
    if not gt_p.is_file():
        raise SystemExit(f"no GT for {name} — --stage gt first")
    with np.load(att_p, allow_pickle=False) as a, np.load(gt_p, allow_pickle=False) as g:
        att = {k: a[k] for k in a.files}
        gt = {k: g[k] for k in g.files}
    if not np.array_equal(att["src_index"], gt["keep"]):
        raise SystemExit(f"{name}: attention steps != GT steps (stride mismatch)")
    return att, gt


def _names(root: pathlib.Path) -> list[str]:
    names = sorted(p.name for p in attention_dir(root).glob("*.npz")
                   if not p.name.endswith(".partial.npz"))
    if not names:
        raise SystemExit(f"no attention under {root / 'attention'} — run --stage extract (GPU)")
    return names


def stage_filter(args, root: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t47b_sink as sk

    names = _names(root)
    per: dict[str, dict] = {}
    aggs = None
    for n in names:
        att, _ = load_episode(root, n)
        aggs = [str(a) for a in att["aggregations"]]
        for ai, a in enumerate(aggs):
            b = np.asarray(att["attention"][:, 0, ai, pr.LAYERS[0]:pr.LAYERS[-1] + 1], np.float64)
            for f, v in sk.map_stats(b).items():
                per.setdefault(a, {}).setdefault(f, []).append(v)
    eligible, excluded, arrays = [], {}, {}
    for a in aggs:
        summ = pr.filter_summary({f: np.concatenate(v) for f, v in per[a].items()})
        reasons = pr.filter_reasons(summ)
        el = pr.filter_eligible(reasons)
        arrays[f"{a}__eligible"] = el
        for f in ("mass_median", "modal_fraction", "top1_share_mean", "largest_share_of_grid_mean"):
            arrays[f"{a}__{f}"] = summ[f]
        for li, l in enumerate(pr.LAYERS):
            for h in pr.HEADS:
                cid = pr.Cell(a, l, h).id
                if el[li, h]:
                    eligible.append(cid)
                else:
                    excluded[cid] = {c: [r for r in pr.FILTER_REASONS if reasons[r][li, h, ci]]
                                     for ci, c in enumerate(CAMERAS) if reasons["any"][li, h, ci]}
    np.savez_compressed(root / "filter_arrays.npz", **arrays)
    payload = {**run_info("filter", args), "episodes": names, "queries": aggs,
               "thresholds": pr.FILTER_THRESHOLDS, "eligible": eligible, "excluded": excluded}
    write_json(root / "filter.json", payload)
    print(f"[libero filter] eligible {len(eligible)}")
    return payload


AGGREGATES = ("episode", "task")


def agg_mode(args) -> str:
    mode = getattr(args, "aggregate", None) or "episode"
    if mode not in AGGREGATES:
        raise SystemExit(f"--aggregate must be one of {AGGREGATES}, got {mode!r}")
    return mode


def agg_tag(mode: str) -> str:
    """``""`` for ``episode`` (original names), ``__task`` for task aggregation."""
    return "" if mode == "episode" else f"__{mode}"


def stage_score2d(args, root: pathlib.Path) -> dict:
    mode = agg_mode(args)
    tag = agg_tag(mode)
    names = _names(root)
    filt = json.loads((root / "filter.json").read_text())
    qs = list(filt["queries"])
    cells = pr.omega(qs)
    ids = [c.id for c in cells]
    eligible = np.asarray([i in set(filt["eligible"]) for i in ids])
    E, n_q = len(names), len(qs)
    lift = np.full((E, n_q, len(pr.LAYERS), 8), np.nan)
    peak = np.full_like(lift, np.nan)
    hit = np.full_like(lift, np.nan)
    for e, n in enumerate(names):
        att, gt = load_episode(root, n)
        tc = np.asarray(gt["target_cov"], np.float64)
        cov = np.stack([tc, 1.0 - tc], axis=1)[:, None]  # (T, C=1, K=2, 16, 16)
        keep = (np.asarray(gt["target_area"]) >= args.min_target_px)[:, None]
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(gt["names"]))
        flat_cov = np.asarray(gt["cov"], np.float64).reshape(len(tc), len(gt["names"]), -1)
        for qi, q in enumerate(qs):
            ai = [str(a) for a in att["aggregations"]].index(q)
            b = np.asarray(att["attention"][:, 0, ai, pr.LAYERS[0]:pr.LAYERS[-1] + 1, :, :1],
                           np.float64)  # agent view only: (T, L, H, 1, 16, 16)
            sc = pr.record_2d_scores(b, cov, keep)
            lift[e, qi], peak[e, qi] = sc["lift"], sc["peak_on_target"]
            mass = np.einsum("tlhp,top->tlho", b[:, :, :, 0].reshape(b.shape[0], b.shape[1],
                                                                    b.shape[2], -1), flat_cov)
            hit[e, qi] = knows_hits(mass, gt["area"], gt["target_mask"], perm=perm).mean(axis=0)
        print(f"[libero score2d] {n}")
    S = lift.reshape(E, -1)
    T = peak.reshape(E, -1)
    groups = [task_of(n) for n in names] if mode == "task" else None
    s_m, t_m = pr.aggregate_mean(S, groups), pr.aggregate_mean(T, groups)
    h_m = pr.aggregate_mean(hit.reshape(E, -1), groups)
    order = pr.rank_order(s_m, t_m, eligible)
    np.savez_compressed(root / f"score2d_arrays{tag}.npz", episodes=np.asarray(names), queries=np.asarray(qs),
                        lift=lift, peak_on_target=peak, knows_hit=hit,
                        eligible=eligible.reshape(n_q, len(pr.LAYERS), 8))
    payload = {**run_info("score2d", args), "episodes": names, "primary": "2D lift (agent view)",
               "no_3d": "no AG3S grounding for the LIBERO scene — 2D primary",
               "min_target_px": args.min_target_px,
               "ranking_top20": [{"cell": ids[i], "lift": float(s_m[i]),
                                  "peak_on_target": float(t_m[i]),
                                  "knows_hit": float(h_m[i])}
                                 for i in order[:20]],
               "top1": ids[order[0]] if order.size else None,
               "paper_cell_knows_hit_mean_mean": (float(pr.aggregate_mean(hit[:, qs.index("mean"), 10, 3],
                                                                          groups))
                                                  if "mean" in qs else None),
               **({"aggregate": mode, "tasks": sorted(set(groups))} if mode != "episode" else {})}
    write_json(root / f"score2d{tag}.json", payload)
    return payload


_TASK_RE = re.compile(r"^(?P<task>.+_task\d+)_ep\d+(\.npz)?$")


def task_of(name: str) -> str:
    """``libero_10_task0_ep1.npz`` -> ``libero_10_task0`` (the scene-layout group of an episode)."""
    m = _TASK_RE.match(str(name))
    if not m:
        raise ValueError(f"cannot read the task of P0b episode {name!r}")
    return m.group("task")


def stage_bootstrap(args, root: pathlib.Path) -> dict:
    mode = agg_mode(args)
    tag = agg_tag(mode)
    if mode == "task" and args.unit != "task":
        raise SystemExit("--aggregate task resamples tasks: use --unit task")
    with np.load(root / f"score2d_arrays{tag}.npz", allow_pickle=False) as d:
        lift, peak, el = d["lift"], d["peak_on_target"], d["eligible"].reshape(-1)
        qs = [str(q) for q in d["queries"]]
        names = [str(n) for n in d["episodes"]]
    ids = [c.id for c in pr.omega(qs)]
    E = lift.shape[0]
    groups = [task_of(n) for n in names] if args.unit == "task" else None
    bs = pr.bootstrap_ranking(lift.reshape(E, -1), peak.reshape(E, -1), el, n_boot=args.n_boot,
                              seed=pr.SEED_BOOTSTRAP, groups=groups, group_mean=mode == "task")
    full = pr.rank_order(bs["mean"], pr.aggregate_mean(peak.reshape(E, -1),
                                                       groups if mode == "task" else None), el)
    order = np.argsort(-bs["p_rank1"], kind="stable")
    payload = {**run_info("bootstrap", args), "episodes": names, "n_boot": bs["n_boot"],
               "unit": args.unit, "n_units": bs.get("n_groups", E),
               "seed": bs["seed"], "primary": "2D lift (agent view)",
               "full_sample_top1": ids[full[0]],
               "full_sample_top1_p_rank1": float(bs["p_rank1"][full[0]]),
               "stable_single_cell": bool(bs["p_rank1"][full[0]] >= 0.5),
               "top10": [{"cell": ids[i], "p_rank1": float(bs["p_rank1"][i]),
                          "p_top3": float(bs["p_top3"][i]), "mean": float(bs["mean"][i]),
                          "ci95": [float(bs["ci_lo"][i]), float(bs["ci_hi"][i])]}
                         for i in order[:10]],
               **({"aggregate": mode} if mode != "episode" else {})}
    name = ("bootstrap" if args.unit == "task" else f"bootstrap__{args.unit}") + tag
    write_json(root / f"{name}.json", payload)
    return payload


STAGES = ("survey", "gt", "extract", "filter", "score2d", "bootstrap")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--min-target-px", type=int, default=MIN_TARGET_PX)
    ap.add_argument("--n-boot", type=int, default=pr.N_BOOT)
    ap.add_argument("--init", choices=("gaussian", "zeros"), default="gaussian",
                    help="extract: Euler start (zeros = the P0b probe, for parity with p0b_sweep.log)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--episodes", nargs="+", default=None,
                    help="extract: only these P0b episodes (stem or file name), e.g. a 1-episode smoke")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="extract: only when the lead confirmed the listed GPU processes are harmless")
    ap.add_argument("--unit", choices=("task", "episode"), default="task",
                    help="bootstrap: resample tasks (default; both episodes of a task together) or "
                         "episodes (exploratory)")
    ap.add_argument("--aggregate", choices=AGGREGATES, default="episode",
                    help="score2d / bootstrap point estimate: episode mean (default, original "
                         "outputs) or task mean (T51.task.md 집계 단위; suffix __task)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    root = out_root()
    t0 = time.time()
    globals()[f"stage_{args.stage}"](args, root)
    print(f"[{STEP} libero] {args.stage} done in {time.time() - t0:.1f}s -> {root}")


if __name__ == "__main__":
    main()
