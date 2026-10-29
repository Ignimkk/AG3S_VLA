"""T51 record registry: T47's seven records plus the three recorded VLA-training runs, with scene ids.

`T51.task.md` "판정 보완" (2026-10-12): the calibration set gains the recorded rollouts of training
episodes ``ep0`` · ``ep500`` · ``ep1200`` (``results/workspace/live_test/20260925_train/``), so the
calibration has five distinct scenes — episode 1800 (``long`` · ``t2``), 1807, 0, 500, 1200 — and the
P3 bootstrap / P4 CV resample **scenes** (runs of one scene travel together). These three episodes were
in the VLA's training data; every T51 payload says so (`provenance`). Held-out is unchanged.

T47's registry (`t47_head_selection.RECORDS`) is read-only here: the seven T47 keys dispatch to its
functions unchanged; the three extra keys read their attention npz and GT cache from the T51 run
(``<extra dir>/attention/attention_16d_<key>.npz`` from `t51_extract` — GPU — and
``<extra dir>/gt_cache_extra/<key>.npz`` from ``t51_study --stage gt_extra`` — CPU). ``<extra dir>``
defaults to ``$AG3S_RUN_DIR/verify/T51``; `configure` overrides it (CLI ``--extra-dir``), and every
spawned worker calls `configure` with the parent's value. Nothing is ever written into storage assets.

Scene id = the recorded ``--episode-index`` (the randomized-scene seed of `pi05_infer.py`), checked
against the step record's ``meta.json`` by `check_pairing`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import zipfile
from typing import Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t47_head_selection as hs

#: Storage-relative checkpoint every ``attention_16d_*.npz`` came from (step 29999; the npz
#: ``checkpoint`` field of the T47 assets ends in the same three components).
CHECKPOINT_REL = ("checkpoints/rby1/pi05_rby1_randomized_pick_place_16d_lora/"
                  "rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999")
CHECKPOINT_TAIL = ("pi05_rby1_randomized_pick_place_16d_lora/"
                   "rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999")
CONFIG = "pi05_rby1_randomized_pick_place_16d_lora"

#: The on-disk format of the T47 ``attention_16d_*.npz`` (`pi05_attention.py:286-299` with the
#: defaults of `:233-245` and ``--noise-seeds 0``; the original `extract.sh` in
#: `experiments/legacy/outputs/verify/attn16d/`). Frame axis first, then exactly this.
ATTENTION_FORMAT = {
    "keys": ("attention", "t_step", "denoise_steps", "aggregations", "cameras", "prompt",
             "noise_seeds", "records", "checkpoint"),
    "trailing_shape": (3, 3, 18, 8, 3, 16, 16),  # (denoise, agg, layer, head, camera, 16, 16)
    "dtype": "float16",
    "denoise_steps": (0, 4, 9),
    "aggregations": ("mean", "first", "last"),
    "cameras": tuple(hs.POLICY_CAMERAS),
    "noise_seeds": (0,),
}


@dataclasses.dataclass(frozen=True)
class ExtraSpec:
    key: str
    record: str  # storage-relative step record (run_XXXX)
    episode: int  # recorded --episode-index


TRAIN_RUNS = "results/workspace/live_test/20260925_train"
EXTRA_RECORDS: dict[str, ExtraSpec] = {s.key: s for s in (
    ExtraSpec("ep0", f"{TRAIN_RUNS}/ep0/run_0000", 0),
    ExtraSpec("ep500", f"{TRAIN_RUNS}/ep500/run_0000", 500),
    ExtraSpec("ep1200", f"{TRAIN_RUNS}/ep1200/run_0000", 1200),
)}
EXTRA_KEYS = tuple(EXTRA_RECORDS)
#: Role of the extra runs (T47's are "selection" / "heldout").
EXTRA_ROLE = "calibration_train"
#: Calibration = T47 selection + the extra runs (판정 보완); held-out = T47 held-out, unchanged.
CALIB_KEYS = tuple(hs.SELECTION_KEYS) + EXTRA_KEYS
HELDOUT_KEYS = tuple(hs.HELDOUT_KEYS)
ALL_KEYS = tuple(hs.RECORDS) + EXTRA_KEYS
#: Recorded --episode-index of every record (T47's from their meta.json, 2026-10-12).
EPISODE_OF = {"long": 1800, "t2": 1800, "ep1807": 1807, "headfix": 1800, "ep1800": 1800,
              "ep1850": 1850, "ep1900": 1900,
              **{k: s.episode for k, s in EXTRA_RECORDS.items()}}
#: Episodes of the VLA training split among these records (판정 보완: stated in every result).
VLA_TRAINING_KEYS = EXTRA_KEYS


def scene_of(key: str) -> str:
    """``scene<episode index>`` — runs that start from the same randomized scene share it."""
    if key not in EPISODE_OF:
        raise KeyError(f"no scene id for record {key!r}")
    return f"scene{EPISODE_OF[key]}"


def scene_map(keys: Sequence[str]) -> dict[str, str]:
    return {k: scene_of(k) for k in keys}


# ------------------------------------------------------------------------------ extra dir

_EXTRA_DIR: Optional[pathlib.Path] = None


def configure(extra_dir: Optional[str | os.PathLike]) -> None:
    """Where the extra runs' attention / GT cache live (``None`` -> ``$AG3S_RUN_DIR/verify/T51``)."""
    global _EXTRA_DIR
    _EXTRA_DIR = None if extra_dir is None else pathlib.Path(extra_dir)


def extra_dir() -> pathlib.Path:
    if _EXTRA_DIR is not None:
        return _EXTRA_DIR
    run_dir = os.environ.get("AG3S_RUN_DIR")
    if not run_dir:
        raise SystemExit("AG3S_RUN_DIR is not set and no --extra-dir given (extra runs' npz / GT)")
    return pathlib.Path(run_dir) / "verify" / "T51"


def extra_dir_or_none() -> Optional[str]:
    """`extra_dir` as a string for payloads, None when neither is configured."""
    if _EXTRA_DIR is None and not os.environ.get("AG3S_RUN_DIR"):
        return None
    return str(extra_dir())


# ------------------------------------------------------------------------------ dispatch


def is_extra(key: str) -> bool:
    return key in EXTRA_RECORDS


def resolve_keys(requested: Optional[Sequence[str]], default: Sequence[str]) -> list[str]:
    keys = list(requested or default)
    unknown = [k for k in keys if k not in ALL_KEYS]
    if unknown:
        raise SystemExit(f"unknown record key(s) {unknown}; have {sorted(ALL_KEYS)}")
    if len(set(keys)) != len(keys):
        raise SystemExit(f"duplicate record keys {keys}")
    return keys


def role(key: str) -> str:
    return EXTRA_ROLE if is_extra(key) else hs.RECORDS[key].role


def scene_group(key: str) -> str:
    """T47b's stratum names (`t47b_metrics.SCENE_GROUP`); the extra runs are calibration
    (``selection``) like T47's selection records."""
    from AG3S.ag3s.experiments.studies import t47b_metrics as bm

    return "selection" if is_extra(key) else bm.scene_group(key)


def npz_path(key: str) -> pathlib.Path:
    if is_extra(key):
        return extra_dir() / "attention" / f"attention_16d_{key}.npz"
    return hs.npz_path(key)


def record_path(key: str) -> pathlib.Path:
    if is_extra(key):
        return hs.storage_root() / EXTRA_RECORDS[key].record
    return hs.record_path(key)


def gt_cache_path(key: str, t47: Optional[pathlib.Path] = None) -> pathlib.Path:
    if is_extra(key):
        return extra_dir() / "gt_cache_extra" / f"{key}.npz"
    if t47 is None:
        raise ValueError("T47 keys need the T47 verify dir")
    return hs.gt_cache_path(pathlib.Path(t47), key)


def load_gt(t47: pathlib.Path, key: str) -> dict:
    if not is_extra(key):
        return hs.load_gt(t47, key)
    p = gt_cache_path(key)
    if not p.is_file():
        raise SystemExit(f"[{key}] no GT cache at {p} — run t51_study --stage gt_extra first")
    with np.load(p, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def meta_of(key: str) -> dict:
    return json.loads((record_path(key) / "meta.json").read_text())


def target_of(key: str) -> str:
    """Extra runs: from the step record's prompt (the npz prompt must equal it — `check_pairing`)."""
    if not is_extra(key):
        return hs.target_of(key)
    from AG3S.ag3s.experiments.reports.attention_report import target_from_prompt

    prompt = str(meta_of(key)["prompt"])
    t = target_from_prompt(prompt)
    if t is None:
        raise SystemExit(f"[{key}] cannot parse the target from prompt {prompt!r}")
    return t


# ------------------------------------------------------------------------ npz format


def npz_array_header(path: pathlib.Path, name: str) -> tuple[tuple[int, ...], str]:
    """``(shape, dtype str)`` of one array of an npz from its header only (no decompression of the
    data — the attention array is ~0.2 GB per record)."""
    with zipfile.ZipFile(path) as z, z.open(f"{name}.npy") as fh:
        version = np.lib.format.read_magic(fh)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, _fortran, dtype = reader(fh)
    return tuple(int(s) for s in shape), np.dtype(dtype).name


def format_problems(fields: dict, shape: Sequence[int], dtype: str) -> list[str]:
    """Differences of one npz from `ATTENTION_FORMAT` (empty = same format as the T47 assets).

    `fields` = the small arrays (everything but ``attention``); `shape` / `dtype` = the attention
    array's header.
    """
    f = ATTENTION_FORMAT
    out = []
    missing = [k for k in f["keys"] if k != "attention" and k not in fields]
    if missing:
        out.append(f"missing keys {missing}")
    if len(shape) != 1 + len(f["trailing_shape"]) or tuple(shape[1:]) != f["trailing_shape"]:
        out.append(f"attention shape {tuple(shape)} != (F, *{f['trailing_shape']})")
    if dtype != f["dtype"]:
        out.append(f"attention dtype {dtype} != {f['dtype']}")
    for k, conv in (("denoise_steps", int), ("aggregations", str), ("cameras", str),
                    ("noise_seeds", int)):
        if k in fields:
            got = tuple(conv(x) for x in np.asarray(fields[k]).reshape(-1))
            if got != tuple(f[k]):
                out.append(f"{k} {got} != {tuple(f[k])}")
    if "checkpoint" in fields and not str(fields["checkpoint"]).rstrip("/").endswith(CHECKPOINT_TAIL):
        out.append(f"checkpoint {str(fields['checkpoint'])!r} is not step {CHECKPOINT_TAIL}")
    if "t_step" in fields and shape and np.asarray(fields["t_step"]).shape != (shape[0],):
        out.append(f"t_step {np.asarray(fields['t_step']).shape} vs {shape[0]} frames")
    return out


def read_npz_format(path: pathlib.Path) -> tuple[dict, tuple[int, ...], str]:
    shape, dtype = npz_array_header(path, "attention")
    with np.load(path, allow_pickle=False) as blob:
        fields = {k: blob[k] for k in blob.files if k != "attention"}
    return fields, shape, dtype


# ------------------------------------------------------------------------------ pairing


def _record_t_steps(rp: pathlib.Path) -> list[int]:
    out = []
    for f in sorted(rp.glob("step_*.npz")):
        with np.load(f) as d:
            out.append(int(d["t_step"]))
    return out


def episode_problems(key: str, meta: dict) -> list[str]:
    ep = hs._episode_index(meta)  # noqa: SLF001 — T47's reader of the recorded --episode-index
    want = EPISODE_OF[key]
    return [] if ep == want else [f"meta --episode-index {ep} != registered scene episode {want}"]


def check_pairing(key: str) -> dict:
    """T47 keys: `t47_head_selection.check_pairing` (unchanged) + the scene id check. Extra keys:
    the npz was made from this record (its ``records`` field is the resolved record directory, the
    producer writes ``Path(--records).resolve()``, `pi05_attention.py:297`), prompt / frame count /
    t_step agree, the format is `ATTENTION_FORMAT`, and the recorded episode is the registered scene.
    Raises on any mismatch."""
    if not is_extra(key):
        info = hs.check_pairing(key)
        probs = episode_problems(key, meta_of(key))
        if probs:
            raise SystemExit(f"[{key}] " + "; ".join(probs))
        return {**info, "scene_id": scene_of(key), "vla_training_episode": False}
    rp = record_path(key)
    if not (rp / "meta.json").is_file():
        raise SystemExit(f"[{key}] step record missing: {rp}")
    npz = npz_path(key)
    if not npz.is_file():
        raise SystemExit(f"[{key}] no attention npz at {npz} — run "
                         "`t51_extract --stage extract` (GPU) first, or pass --extra-dir")
    meta = json.loads((rp / "meta.json").read_text())
    t_rec = _record_t_steps(rp)
    fields, shape, dtype = read_npz_format(npz)
    problems = format_problems(fields, shape, dtype)
    rec_field = str(fields.get("records", ""))
    if pathlib.Path(rec_field).resolve() != rp.resolve():
        problems.append(f"npz records={rec_field!r} vs {rp.resolve()}")
    if str(fields.get("prompt")) != meta.get("prompt"):
        problems.append(f"prompt {str(fields.get('prompt'))!r} vs {meta.get('prompt')!r}")
    t_npz = [int(t) for t in np.asarray(fields.get("t_step", []))]
    if shape[0] != len(t_rec) or t_npz != t_rec:
        problems.append(f"frames npz {shape[0]} t_step {t_npz[:4]}... vs record {len(t_rec)} "
                        f"t_step {t_rec[:4]}...")
    problems += episode_problems(key, meta)
    if problems:
        raise SystemExit(f"[{key}] attention npz and step record do not pair:\n  "
                         + "\n  ".join(problems))
    return {"key": key, "role": EXTRA_ROLE, "npz": str(npz), "record": str(rp),
            "npz_records_field": rec_field, "prompt": str(fields["prompt"]), "n_frames": shape[0],
            "episode_index": EXTRA_RECORDS[key].episode, "scene_id": scene_of(key),
            "checkpoint": str(fields["checkpoint"]), "vla_training_episode": True,
            "pairing": "resolved records path + prompt + frame count + t_step + format + episode"}


def load_attention(key: str):
    """``(attention float16 (F, D, A, L, H, C, 16, 16), denoise list, agg list, cameras list)`` —
    `t47_head_selection.load_attention` for T47 keys, the same checks for the extra runs."""
    if not is_extra(key):
        return hs.load_attention(key)
    p = npz_path(key)
    if not p.is_file():
        raise SystemExit(f"[{key}] no attention npz at {p} — run t51_extract --stage extract (GPU)")
    with np.load(p, allow_pickle=False) as blob:
        att = np.asarray(blob["attention"])
        denoise = [int(d) for d in blob["denoise_steps"]]
        aggs = [str(a) for a in blob["aggregations"]]
        cams = [str(c) for c in blob["cameras"]]
    if att.ndim != 8 or att.shape[-2:] != (16, 16):
        raise SystemExit(f"[{key}] unexpected attention shape {att.shape}")
    if tuple(cams) != tuple(hs.POLICY_CAMERAS):
        raise SystemExit(f"[{key}] camera axis {cams} != {hs.POLICY_CAMERAS}")
    return att, denoise, aggs, cams


def n_frames(key: str) -> int:
    with np.load(npz_path(key), allow_pickle=False) as blob:
        return int(blob["t_step"].shape[0])


def provenance(keys: Sequence[str]) -> dict:
    """What every T51 payload states about its records (판정 보완: training episodes are named)."""
    return {"records": list(keys), "scene_of": scene_map(keys),
            "n_scenes": len({scene_of(k) for k in keys}),
            "vla_training_episodes": [k for k in keys if k in VLA_TRAINING_KEYS],
            "note": ("ep0 / ep500 / ep1200 are recorded rollouts of episodes that were in the VLA "
                     "training data (20260925_train); held-out records are not")}


__all__ = [
    "ALL_KEYS", "ATTENTION_FORMAT", "CALIB_KEYS", "CHECKPOINT_REL", "CHECKPOINT_TAIL", "CONFIG",
    "EPISODE_OF", "EXTRA_KEYS", "EXTRA_RECORDS", "EXTRA_ROLE", "HELDOUT_KEYS", "VLA_TRAINING_KEYS",
    "check_pairing", "configure", "episode_problems", "extra_dir", "extra_dir_or_none",
    "format_problems", "gt_cache_path",
    "is_extra", "load_attention", "load_gt", "n_frames", "npz_array_header", "npz_path", "provenance",
    "read_npz_format", "record_path", "resolve_keys", "role", "scene_group", "scene_map", "scene_of",
    "target_of",
]
