"""T53 record registry: the 16 never-used test-split episodes recorded AG3S-ON, and where their files are.

`T53.task.md` (2026-10-13): episodes 1840 · 1809 · 1842 · 1843 · 1860 · 1845 · 1862 · 1815 · 1848 · 1849 ·
1866 · 1851 · 1804 · 1853 · 1822 · 1839 (layout slot 0–15, one each; four fruits, both arms) are the
**final test** of T54–T56. None of them was used for head selection, diagnosis or a setting decision,
and none is in the VLA fine-tune range (episodes 0–1599). Every key here has role ``final_test``.

Identity per episode (`T53Episode`): key ``ep<N>``, episode index N (= the scene: `pi05_infer.py`
resets the randomized scene of ``--episode-index N``), scene id ``scene<N>`` (the `t51_records`
convention), target fruit / used arm / layout slot / prompt from the dataset's
``meta/randomized_episodes.jsonl``. `EXPECTED` is that table as read on 2026-10-13; `dataset_problems`
re-reads the file and refuses a mismatch, so a different dataset cannot silently relabel the targets.

Layout of the rollouts (written by the verifier's client / server; the root is the **argument** of
every CLI, never a constant)::

    <run>/verify/T53/                      (`configure(run_dir=...)` or `configure(t53_dir=...)`)
      <episode dir>/                       ep1840 | 1840 | <ARM>_ep1840_s<seed> (one, see `episode_dir`)
        ag3s/run_XXXX/meta.json, step_*.npz      --record-ag3s   (exactly one run_XXXX)
        frames.jsonl, manifest.json              --record-frames (or frames/ beneath)
        <any>.npz with executed_actions          --trajectory-out (traj.npz by the CL3 template)
        **/chunk_*.npz                           server --record-constraints (when recorded per episode)
      attention/ noun/ feat/ gt/           outputs of `t53_extract`

When the server's constraint chunks are not beneath the episode directory, `locate_chunks` takes the
server record roots and pairs a segment (chunks between two ``t_step == 1`` restarts, the
`analyze_cl.segments` rule of the CL3 template) to the episode by ``refined_chunk`` == planning-row
``actions`` (|Δ| <= 1e-6, `analyze_cl.pair_seg`).

Pairing (`pairing_report`): step record ↔ registry (``--episode-index``, prompt == dataset prompt, the
prompt's fruit == registered target), step record ↔ frames.jsonl (planning rows: count, t_step; manifest
episode / target / arm / prompt), ↔ trajectory (``chunk_start_steps`` == t_step, prompt, seed), ↔ the
attention npz when present (`t51_records.format_problems` + resolved ``records`` path + prompt + t_step).
Problems are returned (``check`` stages) or raised (`check_pairing`).

CPU only, no MuJoCo, no jax.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import zipfile
from typing import Any, Iterable, Optional, Sequence

import numpy as np

STEP = "T53"
ROLE = "final_test"
#: The dataset whose ``meta/randomized_episodes.jsonl`` defines the episodes (`t48_hsr_io.DATASET_REL`).
DATASET_REL = "datasets/rby1/rby1_randomized_pick_place_16d_v1"
#: VLA fine-tune episodes (`third_party/openpi/.../training/config.py` ``episodes=tuple(range(1600))``,
#: `t48b_hsr.VLA_TRAIN_EPISODES`).
VLA_TRAIN_EPISODES = range(0, 1600)

#: episode -> (target fruit, used arm, layout slot, dataset prompt), read from
#: ``randomized_episodes.jsonl`` on 2026-10-13 (split ``test`` for all 16). Order = T53.task.md order.
EXPECTED: dict[int, tuple[str, str, int, str]] = {
    1840: ("banana", "right", 0, "Pick up the banana and place it in the basket."),
    1809: ("banana", "left", 1, "Pick up the banana and place it in the basket."),
    1842: ("pear", "right", 2, "Put the pear in the basket."),
    1843: ("apple", "right", 3, "Pick up the apple and place it in the basket."),
    1860: ("orange", "left", 4, "Put the orange in the basket."),
    1845: ("pear", "left", 5, "Pick up the pear and place it in the basket."),
    1862: ("apple", "right", 6, "Place the apple into the basket."),
    1815: ("apple", "right", 7, "Put the apple in the basket."),
    1848: ("orange", "left", 8, "Place the orange into the basket."),
    1849: ("banana", "right", 9, "Put the banana in the basket."),
    1866: ("orange", "right", 10, "Put the orange in the basket."),
    1851: ("orange", "left", 11, "Place the orange into the basket."),
    1804: ("apple", "right", 12, "Put the apple in the basket."),
    1853: ("pear", "left", 13, "Pick up the pear and place it in the basket."),
    1822: ("banana", "right", 14, "Place the banana into the basket."),
    1839: ("orange", "right", 15, "Place the orange into the basket."),
}
EPISODES = tuple(EXPECTED)
DATASET_SPLIT = "test"


@dataclasses.dataclass(frozen=True)
class T53Episode:
    key: str
    episode: int
    target: str
    used_arm: str
    layout_slot: int
    prompt: str
    role: str = ROLE

    @property
    def scene_id(self) -> str:
        return f"scene{self.episode}"


REGISTRY: dict[str, T53Episode] = {
    f"ep{e}": T53Episode(f"ep{e}", e, t, a, s, p) for e, (t, a, s, p) in EXPECTED.items()}
KEYS = tuple(REGISTRY)

# ------------------------------------------------------------------------------ registry


def resolve_keys(requested: Optional[Sequence[str]]) -> list[str]:
    """Keys in registry order when `requested` is empty; otherwise `requested` (unknown / duplicate
    keys refused). ``1840`` is accepted for ``ep1840``."""
    if not requested:
        return list(KEYS)
    keys = [k if str(k).startswith("ep") else f"ep{k}" for k in requested]
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown T53 key(s) {unknown}; have {list(KEYS)}")
    if len(set(keys)) != len(keys):
        raise SystemExit(f"duplicate T53 keys {keys}")
    return keys


def episode_of(key: str) -> T53Episode:
    if key not in REGISTRY:
        raise KeyError(f"no T53 episode {key!r}")
    return REGISTRY[key]


def scene_of(key: str) -> str:
    return episode_of(key).scene_id


def target_of(key: str) -> str:
    return episode_of(key).target


def provenance(keys: Sequence[str]) -> dict:
    """What every T53 payload states about its records."""
    return {"records": list(keys), "role": ROLE, "scene_of": {k: scene_of(k) for k in keys},
            "n_scenes": len({scene_of(k) for k in keys}),
            "vla_training_episodes": [k for k in keys if episode_of(k).episode in VLA_TRAIN_EPISODES],
            "note": ("T53 final test: test-split episodes never used for head selection, diagnosis "
                     "or a setting decision (T53.task.md); results here must not change settings")}


def registry_table() -> list[dict]:
    return [{"key": e.key, "episode": e.episode, "scene_id": e.scene_id, "target": e.target,
             "used_arm": e.used_arm, "layout_slot": e.layout_slot, "prompt": e.prompt, "role": e.role}
            for e in REGISTRY.values()]


def storage_root() -> pathlib.Path:
    from common.workspace import shared_storage_root

    return pathlib.Path(shared_storage_root())


def dataset_problems(episodes_meta: dict[int, dict]) -> list[str]:
    """`EXPECTED` against ``randomized_episodes.jsonl`` rows (``{episode_index: row}``)."""
    out = []
    for ep, (target, arm, slot, prompt) in EXPECTED.items():
        r = episodes_meta.get(ep)
        if r is None:
            out.append(f"episode {ep} missing from the dataset metadata")
            continue
        got = (r.get("target_fruit"), r.get("used_arm"), r.get("layout_index"), r.get("prompt"),
               r.get("split"))
        want = (target, arm, slot, prompt, DATASET_SPLIT)
        if got != want:
            out.append(f"episode {ep}: dataset (target, arm, slot, prompt, split) {got} != {want}")
    slots = sorted(v[2] for v in EXPECTED.values())
    if slots != list(range(16)):
        out.append(f"layout slots {slots} are not 0..15 once each")
    return out


def load_dataset_meta(dataset: Optional[pathlib.Path] = None) -> dict[int, dict]:
    path = pathlib.Path(dataset or storage_root() / DATASET_REL) / "meta" / "randomized_episodes.jsonl"
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[int(r["episode_index"])] = r
    return out


# ------------------------------------------------------------------------------ roots

_T53_DIR: Optional[pathlib.Path] = None
_EPISODES_ROOT: Optional[pathlib.Path] = None
#: Sub-directory of an episode directory that ``--record-ag3s`` pointed at.
RECORD_SUBDIR = "ag3s"


def t53_dir_of_run(run_dir: str | os.PathLike, *, check_storage: bool = True) -> pathlib.Path:
    """``<run>/verify/T53``. With `check_storage` the run must be a development run of the shared
    storage (`common.run_output.resolve_output_path`: under ``results/development``, no escaping link)."""
    if check_storage:
        from common.run_output import resolve_output_path

        try:
            return resolve_output_path("verify/T53", run_dir=run_dir, default_name="verify/T53")
        except ValueError as exc:
            raise SystemExit(f"--run-dir {run_dir}: {exc}") from exc
    return pathlib.Path(run_dir) / "verify" / "T53"


def configure(*, run_dir: Optional[str | os.PathLike] = None,
              t53_dir: Optional[str | os.PathLike] = None,
              episodes_root: Optional[str | os.PathLike] = None, check_storage: bool = True) -> None:
    """Set the T53 directory (outputs + default episode root). Exactly one of `run_dir` / `t53_dir`
    (or neither, to reset). `episodes_root` overrides where the episode directories are searched."""
    global _T53_DIR, _EPISODES_ROOT
    if run_dir is not None and t53_dir is not None:
        raise ValueError("give run_dir or t53_dir, not both")
    if run_dir is not None:
        _T53_DIR = t53_dir_of_run(run_dir, check_storage=check_storage)
    elif t53_dir is not None:
        _T53_DIR = pathlib.Path(t53_dir)
    else:
        _T53_DIR = None
    _EPISODES_ROOT = None if episodes_root is None else pathlib.Path(episodes_root)


def t53_dir() -> pathlib.Path:
    if _T53_DIR is None:
        raise SystemExit("T53 directory not configured — pass --run-dir <the T53 run>")
    return _T53_DIR


def episodes_root() -> pathlib.Path:
    return _EPISODES_ROOT if _EPISODES_ROOT is not None else t53_dir()


OUTPUT_KINDS = ("attention", "noun", "feat", "gt")


def out_dir(kind: str, *, create: bool = True) -> pathlib.Path:
    if kind not in OUTPUT_KINDS:
        raise ValueError(f"output kind {kind!r} not in {OUTPUT_KINDS}")
    d = t53_dir() / kind
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def npz_path(key: str) -> pathlib.Path:
    """Action attention, T47 file name (``attention_16d_<key>.npz``)."""
    episode_of(key)
    return t53_dir() / "attention" / f"attention_16d_{key}.npz"


NOUN_VARIANT = "orig"


def noun_path(key: str) -> pathlib.Path:
    """T49 ``orig`` noun extract (`t51b_noun_extract.extract_path` layout under ``noun/``)."""
    episode_of(key)
    return t53_dir() / "noun" / NOUN_VARIANT / f"{key}.npz"


def feat_path(key: str) -> pathlib.Path:
    episode_of(key)
    return t53_dir() / "feat" / f"{key}.npz"


def gt_cache_path(key: str) -> pathlib.Path:
    """T47-format GT cache (+ ``record_key``, ``pairing``, as `t47_head_selection._gt_one`)."""
    episode_of(key)
    return t53_dir() / "gt" / f"{key}.npz"


def grasp_path(key: str) -> pathlib.Path:
    """Per-frame physical grasp state (`t53_extract.grasp_predicates`)."""
    episode_of(key)
    return t53_dir() / "gt" / f"{key}.grasp.npz"


def load_gt(key: str) -> dict:
    p = gt_cache_path(key)
    if not p.is_file():
        raise SystemExit(f"[{key}] no GT cache at {p} — run t53_extract --stage gt first")
    with np.load(p, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


# ------------------------------------------------------------------------------ locating


def _unique(paths: Iterable[pathlib.Path], what: str, key: str) -> Optional[pathlib.Path]:
    found = sorted({p.resolve(): p for p in paths}.values())
    if len(found) > 1:
        raise SystemExit(f"[{key}] {len(found)} candidate {what}: {[str(p) for p in found]} — "
                         "move the stale one aside (never guessed)")
    return found[0] if found else None


def episode_dir(key: str) -> pathlib.Path:
    """The episode's directory under `episodes_root`: ``ep<N>``, ``<N>``, ``*_ep<N>_s*`` or
    ``*_ep<N>`` (exactly one must exist; ``failed/`` is never searched)."""
    ep = episode_of(key).episode
    root = episodes_root()
    if not root.is_dir():
        raise SystemExit(f"episodes root {root} does not exist")
    cands = [root / f"ep{ep}", root / str(ep), *root.glob(f"*_ep{ep}_s*"), *root.glob(f"*_ep{ep}")]
    d = _unique((p for p in cands if p.is_dir()), "episode directories", key)
    if d is None:
        raise SystemExit(f"[{key}] no episode directory under {root} (ep{ep} | {ep} | *_ep{ep}_s*)")
    return d


def record_dir(key: str, ep_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """The ``--record-ag3s`` run directory (``ag3s/run_XXXX`` with meta.json; or ``ag3s/`` itself)."""
    ep_dir = ep_dir or episode_dir(key)
    base = ep_dir / RECORD_SUBDIR
    if (base / "meta.json").is_file():
        return base
    runs = [p for p in base.glob("run_*") if (p / "meta.json").is_file()] if base.is_dir() else []
    d = _unique(runs, f"step records under {base}", key)
    if d is None:
        raise SystemExit(f"[{key}] no step record (meta.json) under {base}")
    return d


def frames_path(key: str, ep_dir: Optional[pathlib.Path] = None) -> Optional[pathlib.Path]:
    ep_dir = ep_dir or episode_dir(key)
    return _unique((p for p in (ep_dir / "frames.jsonl", ep_dir / "frames" / "frames.jsonl")
                    if p.is_file()), "frames.jsonl", key)


def npz_members(path: pathlib.Path) -> set[str]:
    try:
        with zipfile.ZipFile(path) as z:
            return {n[:-4] if n.endswith(".npy") else n for n in z.namelist()}
    except (zipfile.BadZipFile, OSError):
        return set()


def trajectory_path(key: str, ep_dir: Optional[pathlib.Path] = None) -> Optional[pathlib.Path]:
    """The ``--trajectory-out`` npz: the one npz directly in the episode dir (or ``trajectories/``)
    that has ``executed_actions`` and ``chunk_start_steps``."""
    ep_dir = ep_dir or episode_dir(key)
    cands = [*ep_dir.glob("*.npz"), *(ep_dir / "trajectories").glob("*.npz")]
    return _unique((p for p in cands
                    if {"executed_actions", "chunk_start_steps"} <= npz_members(p)),
                   "trajectory npz", key)


def chunk_t_step(path: pathlib.Path) -> Optional[int]:
    try:
        with np.load(path, allow_pickle=False) as z:
            return int(z["t_step"])
    except Exception:  # noqa: BLE001 — a damaged chunk is reported, not fatal
        return None


def segments(record_dirs: Iterable[pathlib.Path]) -> list[list[pathlib.Path]]:
    """Chunk files of server record dirs split into rollouts: a new segment starts at ``t_step == 1``
    (the server's per-request sequence restarts with a client) — CL3 ``analyze_cl.segments``."""
    segs: list[list[pathlib.Path]] = []
    for rec in record_dirs:
        cur: list[pathlib.Path] = []
        for f in sorted(pathlib.Path(rec).glob("chunk_*.npz")):
            ts = chunk_t_step(f)
            if ts is None:
                continue
            if ts == 1 and cur:
                segs.append(cur)
                cur = []
            cur.append(f)
        if cur:
            segs.append(cur)
    return segs


def pair_segment(segs: Sequence[Sequence[pathlib.Path]], planning_actions: Sequence[np.ndarray],
                 used: Iterable[int] = (), tol: float = 1e-6) -> Optional[int]:
    """Index of the segment whose ``refined_chunk`` equals the planning-row ``actions`` chunk for
    chunk (same count, |Δ| <= `tol`) — CL3 ``analyze_cl.pair_seg``; None when no segment matches."""
    used = set(used)
    acts = [np.asarray(a, np.float64) for a in planning_actions]
    for i, sg in enumerate(segs):
        if i in used or len(sg) != len(acts):
            continue
        ok = True
        for f, a in zip(sg, acts):
            with np.load(f, allow_pickle=False) as z:
                if "refined_chunk" not in z.files:
                    ok = False
                    break
                r = np.asarray(z["refined_chunk"], np.float64)
            if r.shape != a.shape or float(np.abs(r - a).max()) > tol:
                ok = False
                break
        if ok:
            return i
    return None


def locate_chunks(key: str, *, ep_dir: Optional[pathlib.Path] = None,
                  server_roots: Sequence[pathlib.Path] = (),
                  planning: Optional[Sequence[dict]] = None) -> dict:
    """``{"chunks": [paths], "source": ...}`` — chunk files beneath the episode dir (one record dir);
    else the segment of `server_roots` (dirs holding ``chunk_*.npz``, or parents searched for
    ``server_constraints/run_*``) paired by planning actions; else no chunks."""
    ep_dir = ep_dir or episode_dir(key)
    local = sorted(ep_dir.rglob("chunk_*.npz"))
    if local:
        parents = sorted({p.parent for p in local})
        if len(parents) > 1:
            raise SystemExit(f"[{key}] chunk files in {len(parents)} directories under {ep_dir}: "
                             f"{[str(p) for p in parents]}")
        return {"chunks": local, "source": f"episode dir {parents[0]}"}
    if server_roots and planning:
        recs: list[pathlib.Path] = []
        for r in server_roots:
            r = pathlib.Path(r)
            recs += [r] if any(r.glob("chunk_*.npz")) else sorted(r.glob("**/server_constraints/run_*"))
        segs = segments(recs)
        acts = [np.asarray(p["actions"], np.float64) for p in planning if p.get("actions") is not None]
        i = pair_segment(segs, acts) if len(acts) == len(planning) else None
        if i is not None:
            return {"chunks": list(segs[i]), "source": f"server segment {i} of {len(segs)} "
                                                       "(refined_chunk == planning actions)"}
        return {"chunks": [], "source": f"no server segment of {len(segs)} pairs with the planning rows"}
    return {"chunks": [], "source": "none found"}


@dataclasses.dataclass
class EpisodeFiles:
    key: str
    episode_dir: pathlib.Path
    record: pathlib.Path
    frames: Optional[pathlib.Path]
    manifest: Optional[pathlib.Path]
    trajectory: Optional[pathlib.Path]

    def as_dict(self) -> dict:
        return {k: (None if v is None else str(v)) for k, v in dataclasses.asdict(self).items()}


def locate(key: str) -> EpisodeFiles:
    d = episode_dir(key)
    rec = record_dir(key, d)
    fr = frames_path(key, d)
    man = None
    if fr is not None and (fr.parent / "manifest.json").is_file():
        man = fr.parent / "manifest.json"
    return EpisodeFiles(key=key, episode_dir=d, record=rec, frames=fr, manifest=man,
                        trajectory=trajectory_path(key, d))


def record_path(key: str) -> pathlib.Path:
    return record_dir(key)


# ------------------------------------------------------------------------------ readers


def read_meta(record: pathlib.Path) -> dict:
    return json.loads((pathlib.Path(record) / "meta.json").read_text())


def record_t_steps(record: pathlib.Path) -> list[int]:
    out = []
    for f in sorted(pathlib.Path(record).glob("step_*.npz")):
        with np.load(f) as d:
            out.append(int(d["t_step"]))
    return out


def argv_value(argv: Sequence[str], flag: str) -> Optional[str]:
    argv = list(argv or [])
    if flag in argv and argv.index(flag) + 1 < len(argv):
        return argv[argv.index(flag) + 1]
    return None


def meta_episode(meta: dict) -> Optional[int]:
    v = argv_value(meta.get("argv") or [], "--episode-index")
    try:
        return None if v is None else int(v)
    except ValueError:
        return None


def read_frames(path: pathlib.Path) -> list[dict]:
    return [json.loads(ln) for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]


def prompt_fruit(prompt: str) -> Optional[str]:
    """The single fruit a prompt names (`t48_hsr_core.prompt_template`), None if 0 or >= 2."""
    from AG3S.ag3s.experiments.studies.t48_hsr_core import prompt_template

    try:
        return prompt_template(prompt)[1]
    except ValueError:
        return None


def seed_of(files: EpisodeFiles) -> Optional[int]:
    """The rollout's ``--policy-seed`` (manifest, else trajectory, else the step meta argv)."""
    if files.manifest is not None:
        s = json.loads(files.manifest.read_text()).get("policy_seed")
        if s is not None:
            return int(s)
    if files.trajectory is not None:
        with np.load(files.trajectory, allow_pickle=False) as z:
            if "policy_seed" in z.files:
                return int(z["policy_seed"])
    v = argv_value(read_meta(files.record).get("argv") or [], "--policy-seed")
    return None if v is None else int(v)


# ------------------------------------------------------------------------------ pairing


def record_problems(key: str, meta: dict) -> list[str]:
    e = episode_of(key)
    out = []
    ep = meta_episode(meta)
    if ep != e.episode:
        out.append(f"meta --episode-index {ep} != registered {e.episode}")
    if meta.get("prompt") != e.prompt:
        out.append(f"meta prompt {meta.get('prompt')!r} != dataset prompt {e.prompt!r}")
    fruit = prompt_fruit(str(meta.get("prompt", "")))
    if fruit != e.target:
        out.append(f"prompt fruit {fruit!r} != registered target {e.target!r}")
    return out


def frames_problems(key: str, rows: Sequence[dict], manifest: Optional[dict], t_rec: Sequence[int],
                    prompt: str) -> tuple[dict, list[str]]:
    """frames.jsonl planning rows and the manifest against the step record."""
    e = episode_of(key)
    plan = [r for r in rows if r.get("kind") == "planning"]
    t_plan = [int(r["t_step"]) for r in plan]
    info: dict[str, Any] = {"n_planning": len(plan),
                            "n_control": sum(r.get("kind") == "control" for r in rows),
                            "n_observation": sum(r.get("kind") == "observation" for r in rows)}
    probs = []
    if t_plan != list(t_rec):
        probs.append(f"frames.jsonl planning t_step ({len(t_plan)}) {t_plan[:4]}... != step record "
                     f"({len(t_rec)}) {list(t_rec)[:4]}...")
    if manifest is not None:
        sc = manifest.get("scene") or {}
        pol = manifest.get("policy") or {}
        for name, got, want in (("scene.episode_index", sc.get("episode_index"), e.episode),
                                ("scene.episode_target_fruit", sc.get("episode_target_fruit"), e.target),
                                ("scene.episode_used_arm", sc.get("episode_used_arm"), e.used_arm),
                                ("policy.prompt", pol.get("prompt"), prompt)):
            if got != want:
                probs.append(f"manifest {name} {got!r} != {want!r}")
        info["policy_seed"] = manifest.get("policy_seed")
    else:
        probs.append("manifest.json missing next to frames.jsonl")
    return info, probs


def actions_vs_record(plan_rows: Sequence[dict], record: pathlib.Path) -> dict:
    """Informational: max |planning ``actions`` (server refined) - step npz ``actions``| (the client's
    chunk). Not a pairing criterion (a HOLD may replace the executed chunk)."""
    files = sorted(pathlib.Path(record).glob("step_*.npz"))
    if len(files) != len(plan_rows):
        return {"compared": 0}
    diffs = []
    for f, r in zip(files, plan_rows):
        if r.get("actions") is None:
            continue
        with np.load(f) as d:
            a = np.asarray(d["actions"], np.float64)
        b = np.asarray(r["actions"], np.float64)
        diffs.append(float(np.abs(a - b).max()) if a.shape == b.shape else float("inf"))
    return {"compared": len(diffs), "max_abs": max(diffs) if diffs else None,
            "n_over_1e-4": int(sum(d > 1e-4 for d in diffs))}


def trajectory_problems(path: pathlib.Path, t_rec: Sequence[int], prompt: str,
                        seed: Optional[int]) -> tuple[dict, list[str]]:
    with np.load(path, allow_pickle=False) as z:
        starts = [int(x) for x in z["chunk_start_steps"]]
        info = {"n_chunks": len(starts), "executed_shape": list(z["executed_actions"].shape),
                "policy_seed": int(z["policy_seed"]) if "policy_seed" in z.files else None}
        tp = str(z["prompt"]) if "prompt" in z.files else None
    probs = []
    if starts != list(t_rec):
        probs.append(f"trajectory chunk_start_steps ({len(starts)}) != step record t_step ({len(t_rec)})")
    if tp is not None and tp != prompt:
        probs.append(f"trajectory prompt {tp!r} != {prompt!r}")
    if seed is not None and info["policy_seed"] is not None and info["policy_seed"] != seed:
        probs.append(f"trajectory policy_seed {info['policy_seed']} != manifest {seed}")
    return info, probs


def attention_problems(npz: pathlib.Path, record: pathlib.Path, prompt: str,
                       t_rec: Sequence[int]) -> tuple[dict, list[str]]:
    """The T51 extra-run check (`t51_records.check_pairing` body) for a T53 record."""
    from AG3S.ag3s.experiments.studies import t51_records as rc51

    fields, shape, dtype = rc51.read_npz_format(npz)
    probs = rc51.format_problems(fields, shape, dtype)
    rec_field = str(fields.get("records", ""))
    if pathlib.Path(rec_field).resolve() != pathlib.Path(record).resolve():
        probs.append(f"npz records={rec_field!r} vs {pathlib.Path(record).resolve()}")
    if str(fields.get("prompt")) != prompt:
        probs.append(f"npz prompt {str(fields.get('prompt'))!r} vs {prompt!r}")
    t_npz = [int(t) for t in np.asarray(fields.get("t_step", []))]
    if (shape[0] if shape else -1) != len(t_rec) or t_npz != list(t_rec):
        probs.append(f"npz frames {shape[0] if shape else None} t_step {t_npz[:4]}... vs record "
                     f"{len(t_rec)} {list(t_rec)[:4]}...")
    return {"npz": str(npz), "n_frames": shape[0] if shape else None, "dtype": dtype,
            "checkpoint": str(fields.get("checkpoint"))}, probs


def chunk_problems(chunks: Sequence[pathlib.Path], plan_rows: Sequence[dict]) -> tuple[dict, list[str]]:
    """Chunk count and server sequence (chunk ``t_step``) against the planning rows' ``seq``."""
    seq_c = [chunk_t_step(p) for p in chunks]
    seq_p = [r.get("seq") for r in plan_rows]
    info = {"n_chunks": len(chunks), "first": str(chunks[0]) if chunks else None}
    probs = []
    if len(chunks) != len(plan_rows):
        probs.append(f"{len(chunks)} server chunks != {len(plan_rows)} planning rows")
    elif seq_c != seq_p:
        probs.append(f"chunk t_step {seq_c[:4]}... != planning seq {seq_p[:4]}...")
    return info, probs


def pairing_report(key: str, *, require_frames: bool = True, require_trajectory: bool = True,
                   require_chunks: bool = False, attention: bool = True,
                   server_roots: Sequence[pathlib.Path] = ()) -> tuple[dict, list[str]]:
    """``(info, problems)`` of one episode (see the module docstring). Never raises on a mismatch;
    raises SystemExit only when the files cannot be located unambiguously."""
    e = episode_of(key)
    files = locate(key)
    meta = read_meta(files.record)
    prompt = str(meta.get("prompt"))
    t_rec = record_t_steps(files.record)
    probs = record_problems(key, meta)
    if not t_rec:
        probs.append(f"no step_*.npz under {files.record}")
    info: dict[str, Any] = {"key": key, "episode": e.episode, "scene_id": e.scene_id, "role": e.role,
                            "target": e.target, "used_arm": e.used_arm, "layout_slot": e.layout_slot,
                            "prompt": prompt, "files": files.as_dict(), "n_frames": len(t_rec),
                            "t_step_first_last": [t_rec[0], t_rec[-1]] if t_rec else None,
                            "meta_n_steps": meta.get("n_steps")}
    if meta.get("n_steps") is not None and int(meta["n_steps"]) != len(t_rec):
        probs.append(f"meta n_steps {meta['n_steps']} != {len(t_rec)} step files (writer not closed?)")
    plan: list[dict] = []
    if files.frames is not None:
        rows = read_frames(files.frames)
        plan = [r for r in rows if r.get("kind") == "planning"]
        man = json.loads(files.manifest.read_text()) if files.manifest is not None else None
        fi, fp = frames_problems(key, rows, man, t_rec, prompt)
        info["frames"] = {**fi, "actions_vs_record": actions_vs_record(plan, files.record)}
        probs += fp
    elif require_frames:
        probs.append(f"no frames.jsonl in {files.episode_dir}")
    seed = seed_of(files)
    info["policy_seed"] = seed
    if files.trajectory is not None:
        ti, tp = trajectory_problems(files.trajectory, t_rec, prompt, seed)
        info["trajectory"] = ti
        probs += tp
    elif require_trajectory:
        probs.append(f"no trajectory npz (executed_actions + chunk_start_steps) in {files.episode_dir}")
    ch = locate_chunks(key, ep_dir=files.episode_dir, server_roots=server_roots, planning=plan)
    info["chunks"] = {"source": ch["source"], "n": len(ch["chunks"])}
    if ch["chunks"] and plan:
        ci, cp = chunk_problems(ch["chunks"], plan)
        info["chunks"].update(ci)
        probs += cp
    elif require_chunks:
        probs.append(f"no server constraint chunks ({ch['source']})")
    if attention and npz_path(key).is_file():
        ai, ap = attention_problems(npz_path(key), files.record, prompt, t_rec)
        info["attention"] = ai
        probs += ap
    return info, probs


def check_pairing(key: str, **kw) -> dict:
    info, probs = pairing_report(key, **kw)
    if probs:
        raise SystemExit(f"[{key}] T53 record does not pair:\n  " + "\n  ".join(probs))
    return info


__all__ = [
    "DATASET_REL", "EPISODES", "EXPECTED", "EpisodeFiles", "KEYS", "NOUN_VARIANT", "OUTPUT_KINDS",
    "RECORD_SUBDIR", "REGISTRY", "ROLE", "STEP", "T53Episode", "actions_vs_record",
    "attention_problems", "check_pairing", "chunk_problems", "chunk_t_step", "configure",
    "dataset_problems", "episode_dir", "episode_of", "episodes_root", "feat_path",
    "frames_path", "frames_problems", "grasp_path", "gt_cache_path", "load_dataset_meta", "load_gt",
    "locate", "locate_chunks", "meta_episode", "noun_path", "npz_members", "npz_path", "out_dir",
    "pair_segment", "pairing_report", "prompt_fruit", "provenance", "read_frames", "read_meta",
    "record_dir", "record_path", "record_problems", "record_t_steps", "registry_table",
    "resolve_keys", "scene_of", "seed_of", "segments", "t53_dir", "t53_dir_of_run", "target_of",
    "trajectory_path", "trajectory_problems",
]
