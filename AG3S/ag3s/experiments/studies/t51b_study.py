"""T51b — the T51 head-selection protocol (P1–P6) with the noun-token query rows in Ω.

`T51b.task.md` (2026-10-12): the same pre-registered protocol as T51 (`T51.task.md` P1–P6, thresholds,
scene-level aggregation, seeds) is run on

  (a) ``--omega noun``      Ω = the noun query only (``noun_target`` × L2–17 × 8 heads = 128 cells);
  (b) ``--omega combined``  Ω = action (d0 × last / mean / first) + noun (512 cells);

and (c) ``--stage heldout_compare`` puts the action best (T51's ``gt/ksel`` selection), the noun best
((a)'s) and the combined best ((b)'s) side by side on the held-out records, with the same GT 3D
``identity_correct`` and a paired per-frame comparison. ``--stage summary`` gathers (a)–(c) with the
query family of every cell (figure material for the verifier).

**Nothing in `t51_study` is edited** (a T51 verifier runs it). Every stage is `t51_study`'s own
``stage_*`` function, run with these module attributes rebound for the T51b process (`install`) —
and for every spawned replay worker (the worker entry is ``functools.partial(_replay_entry, cfg)``,
which rebinds them in the child before calling the original ``_replay_job``):

  ``query_blocks``    noun rows for **every** calibration / held-out record: T47 keys from T49's
                      ``extract/orig/<key>.npz``, the extra runs (ep0 · ep500 · ep1200) from
                      `t51b_noun_extract` (``verify/T51b/noun_extract/orig/<key>.npz``). Each file is
                      checked (stamp record / prompt / noun / config / checkpoint step, t_step equal to
                      the step record and to the attention npz, no failed frame). Action rows: the
                      original function. A noun-only Ω never loads the action attention.
  ``queries``         Ω's queries of ``--omega``.
  ``rows_path``       the 3D row cache is ``verify/T51b/rows/`` (shared by (a), (b), (c)).
  ``load_rows``       T51b's cache first, then T51's ``verify/T51/rows/`` (**read only**): action rows
                      are the same (record, source, replay options) as T51's, so they are not replayed.
                      The replay options are checked against T51's payloads first (`check_replay_opts`).
  ``selflabel_path``  T51's ``selflabel/`` (read only; the self label does not depend on Ω).
  ``run_info``        payloads say ``step: T51b`` and carry the T51b configuration.

Pre-registered rules are T51's, unchanged. The only Ω-dependent consequence: the forced 3D cells of the
2D -> 3D preselection (L8H2 + T47 oracle top-3) are action cells, so a noun-only Ω has none.

``--aggregate scene`` (t51_study's option, 2026-10-12 집계 단위 수정) runs every protocol stage with the
scene-mean estimator and ``__scene`` file names; ``heldout_compare`` / ``summary`` then read T51's and
(a) / (b)'s ``__scene`` outputs and write ``heldout_compare__scene.json`` / ``summary__scene.json``.
The default (``record``) reads and writes the original names.

    source scripts/setup/env.sh; export AG3S_RUN_DIR=<the T51 run>
    B="env MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python \\
       -m AG3S.ag3s.experiments.studies.t51b_study"
    $B --omega noun --stage filter            # then score2d (gt, self), score3d, bootstrap, ksel,
    $B --omega combined --stage filter        #   heldout, ablation, compare — as t51_study
    $B --stage heldout_compare                # (c)
    $B --stage summary                        # (a)–(c) with query families

Output: ``$AG3S_RUN_DIR/verify/T51b/{noun,combined,heldout_compare,summary,rows}/``. Read only: T51
(``--t51-dir``, default ``$AG3S_RUN_DIR/verify/T51``; also the default ``--extra-dir``), T49
(``--t49-dir``), T47 / T47b (as t51_study).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import math
import os
import pathlib
import time
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t49_noun_query_extract as nx
from AG3S.ag3s.experiments.studies import t51_protocol as pr
from AG3S.ag3s.experiments.studies import t51_records as rc
from AG3S.ag3s.experiments.studies import t51_study as st
from AG3S.ag3s.experiments.studies import t51b_noun_extract as ne

STEP = "T51b"
NOUN = st.NOUN_QUERY
OMEGAS = {"noun": (NOUN,), "combined": (*pr.ACTION_QUERIES, NOUN)}
PROTOCOL_STAGES = ("filter", "score2d", "score3d", "bootstrap", "ksel", "heldout", "ablation",
                   "compare", "annotate")
OWN_STAGES = ("heldout_compare", "summary")
STAGES = PROTOCOL_STAGES + OWN_STAGES
#: Replay options that change a 3D row (`t51_study.ensure_rows` ``opts``) — must equal T51's for reuse.
REPLAY_OPTS = ("voxel", "range_max", "links", "on_target_radius", "label_radius", "membership_radius")
#: T51 payloads whose ``args`` record the replay options of T51's rows.
T51_OPT_PAYLOADS = ("gt/score3d/score3d.json", "self/score3d/score3d.json", "gt/ksel/ksel.json",
                    "self/ksel/ksel.json", "heldout/heldout.json", "ablation/ablation.json")
#: `t51_study` attributes rebound by `install` (all must exist — a refactor of t51_study fails loudly).
PATCHED = ("query_blocks", "queries", "rows_path", "load_rows", "selflabel_path", "run_info",
           "_replay_job")
_ORIG = {name: getattr(st, name) for name in PATCHED}


@dataclasses.dataclass(frozen=True)
class Config:
    """Everything a T51b process (and each spawned replay worker) needs; picklable."""

    omega: Optional[str]  # "noun" | "combined" | None (cross-Ω stages)
    t49_dir: str  # T49 verify dir (noun rows of T47's records)
    noun_extra_dir: str  # t51b_noun_extract output (noun rows of the extra runs)
    t51_dir: str  # T51 verify dir (selflabel/, rows/ — read only)
    rows_dir: str  # T51b row cache (verify/T51b/rows)
    t51_rows: bool = True  # read T51's rows when T51b has none
    t51_rows_checked: tuple = ()  # T51 payloads whose replay options matched

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(self).items()}


_CFG: Optional[Config] = None


def _need() -> Config:
    if _CFG is None:
        raise RuntimeError("t51b_study.install(cfg) was not called")
    return _CFG


# ------------------------------------------------------------------------------ rebinding


def install(cfg: Config) -> None:
    """Rebind `PATCHED` in `t51_study` for this process (idempotent)."""
    global _CFG
    _CFG = cfg
    st.query_blocks = query_blocks
    st.queries = queries
    st.rows_path = rows_path
    st.load_rows = load_rows
    st.selflabel_path = selflabel_path
    st.run_info = run_info
    st._replay_job = functools.partial(_replay_entry, cfg)  # noqa: SLF001


def uninstall() -> None:
    global _CFG
    _CFG = None
    for name, fn in _ORIG.items():
        setattr(st, name, fn)


@contextlib.contextmanager
def installed(cfg: Config) -> Iterator[Config]:
    install(cfg)
    try:
        yield cfg
    finally:
        uninstall()


def _replay_entry(cfg: Config, *args, **kwargs):
    """Replay-worker entry: in a spawned child `t51_study` is freshly imported (nothing rebound), so
    rebind first, then run the original ``_replay_job`` (its global lookups see the rebound names)."""
    if _CFG != cfg:
        install(cfg)
    return _ORIG["_replay_job"](*args, **kwargs)


# ------------------------------------------------------------------------------ noun rows


def noun_path(key: str, cfg: Optional[Config] = None) -> pathlib.Path:
    cfg = cfg or _need()
    if rc.is_extra(key):
        return ne.extract_path(pathlib.Path(cfg.noun_extra_dir), key)
    return ne.t49_extract_path(pathlib.Path(cfg.t49_dir), key)


def noun_problems(key: str, ex: dict) -> list[str]:
    """`t51b_noun_extract.record_problems` (stamp record / prompt / noun, t_step vs the step record and
    the attention npz, no failed frame) + the stamp's config and checkpoint step."""
    stamp = ex.get("stamp_dict", {})
    probs = []
    if stamp.get("config") != rc.CONFIG:
        probs.append(f"stamp config {stamp.get('config')!r} != {rc.CONFIG!r}")
    if not str(stamp.get("checkpoint", "")).rstrip("/").endswith(rc.CHECKPOINT_TAIL):
        probs.append(f"stamp checkpoint {stamp.get('checkpoint')!r} is not step {rc.CHECKPOINT_TAIL}")
    if [str(q) for q in ex.get("query_names", [])][:1] != ["target"]:
        probs.append(f"query_names {list(ex.get('query_names', []))} (target first expected)")
    return probs + ne.record_problems(key, ex, ne.record_t_steps(key), ne.attention_t_steps(key))


def noun_block(key: str, cfg: Optional[Config] = None) -> np.ndarray:
    """``(F, 18, 8, 3, 16, 16)`` float64 noun (``target``) rows of one record, checked."""
    p = noun_path(key, cfg)
    if not p.is_file():
        hint = ("run t51b_noun_extract --stage extract (GPU) first, or pass --noun-extra-dir"
                if rc.is_extra(key) else "pass --t49-dir <T49 run>/verify/T49")
        raise SystemExit(f"[{key}] no noun extract at {p} — {hint}")
    ex = nx.load_extract(p)
    probs = noun_problems(key, ex)
    if probs:
        raise SystemExit(f"[{key}] noun extract {p} does not pair:\n  " + "\n  ".join(probs))
    return nx.query_block(ex, "target")


def query_blocks(key: str, qs: Sequence[str], t49: Optional[str] = None) -> dict[str, np.ndarray]:
    """`t51_study.query_blocks` with the noun query available for every record (see module doc)."""
    unknown = [q for q in qs if q not in pr.ACTION_QUERIES and q != NOUN]
    if unknown:
        raise SystemExit(f"unknown query {unknown}")
    action = [q for q in qs if q in pr.ACTION_QUERIES]
    out = dict(_ORIG["query_blocks"](key, action, None)) if action else {}
    if NOUN in qs:
        blk = noun_block(key)
        if action and blk.shape[0] != out[action[0]].shape[0]:
            raise SystemExit(f"[{key}] noun rows have {blk.shape[0]} frames, attention npz "
                             f"{out[action[0]].shape[0]}")
        out[NOUN] = blk
    return {q: out[q] for q in qs}


def queries(args) -> list[str]:
    omega = getattr(args, "omega", None) or _need().omega
    if omega not in OMEGAS:
        raise SystemExit(f"--omega must be one of {list(OMEGAS)} for this stage")
    return list(OMEGAS[omega])


# ------------------------------------------------------------------------------ rows / labels


def rows_path(root: pathlib.Path, key: str, sid: str, *, smoke: bool = False) -> pathlib.Path:
    """T51b's shared row cache (``root`` is ignored: (a), (b) and (c) share one cache)."""
    base = pathlib.Path(_need().rows_dir)
    return (base.with_name(base.name + "_smoke") if smoke else base) / key / f"{pr.safe_name(sid)}.jsonl"


def t51_rows_path(key: str, sid: str, cfg: Optional[Config] = None) -> pathlib.Path:
    cfg = cfg or _need()
    return pathlib.Path(cfg.t51_dir) / "rows" / key / f"{pr.safe_name(sid)}.jsonl"


def row_origin(key: str, sid: str) -> str:
    """``t51b`` / ``t51`` (read-through) / ``missing`` — where `load_rows` finds a (record, source)."""
    cfg = _need()
    if rows_path(pathlib.Path("."), key, sid).is_file():
        return "t51b"
    if cfg.t51_rows and t51_rows_path(key, sid, cfg).is_file():
        return "t51"
    return "missing"


def load_rows(root: pathlib.Path, key: str, sid: str) -> Optional[list[dict]]:
    origin = row_origin(key, sid)
    if origin == "missing":
        return None
    p = rows_path(root, key, sid) if origin == "t51b" else t51_rows_path(key, sid)
    with p.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def selflabel_path(root: pathlib.Path, key: str) -> pathlib.Path:
    return pathlib.Path(_need().t51_dir) / "selflabel" / f"{key}.npz"


def run_info(stage: str, args) -> dict:
    cfg = _need()
    d = _ORIG["run_info"](stage, args)
    d["step"] = STEP
    d["t51b"] = {**cfg.as_dict(),
                 "omega_queries": list(OMEGAS[cfg.omega]) if cfg.omega in OMEGAS else None,
                 "rebound_t51_study": list(PATCHED),
                 "noun_sources": {"t47_records": "T49 extract/orig (t49_dir)",
                                  "extra_runs": "t51b_noun_extract orig (noun_extra_dir)"}}
    return d


def check_replay_opts(t51: pathlib.Path, args) -> tuple:
    """T51's rows are reused only when T51 replayed with the same options: every T51 payload found
    (`T51_OPT_PAYLOADS`) must have the same `REPLAY_OPTS`. Refuses when T51 has rows but no payload."""
    checked, bad = [], []
    for rel in T51_OPT_PAYLOADS:
        p = pathlib.Path(t51) / rel
        if not p.is_file():
            continue
        got = json.loads(p.read_text()).get("args", {})
        diff = {k: (got.get(k), getattr(args, k)) for k in REPLAY_OPTS if got.get(k) != getattr(args, k)}
        if diff:
            bad.append(f"{rel}: {diff}")
        checked.append(rel)
    if bad:
        raise SystemExit("T51's rows were replayed with other options (T51, T51b):\n  " + "\n  ".join(bad)
                         + "\npass the same options or --no-t51-rows")
    if not checked and (pathlib.Path(t51) / "rows").is_dir():
        raise SystemExit(f"{t51}/rows exists but no T51 payload records its replay options — "
                         "pass --no-t51-rows")
    return tuple(checked)


# ------------------------------------------------------------------------------ (c) held-out


def family(source_id: str) -> str:
    """``noun`` / ``action`` / ``noun+action`` of a cell or ``sum[...]``; ``special`` otherwise."""
    try:
        cells = pr.combo_cells(source_id)
    except ValueError:
        return "special"
    fams = sorted({"noun" if c.query == NOUN else "action" for c in cells})
    return "+".join(fams)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p (binomial, p = 0.5) on the discordant counts."""
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return float(min(1.0, 2.0 * tail))


def paired_identity(rows_a: dict[str, list[dict]], rows_b: dict[str, list[dict]],
                    keys: Sequence[str]) -> dict:
    """Per record and pooled (by scene group and all): frames where A / B have ``identity_correct``
    (``both`` · ``only_a`` · ``only_b`` · ``neither``), the rate difference A − B and the exact McNemar p
    (exploratory: frames of one record are not independent). Frames are matched on ``frame``."""
    def counts(pairs):
        n11 = sum(1 for a, b in pairs if a and b)
        n10 = sum(1 for a, b in pairs if a and not b)
        n01 = sum(1 for a, b in pairs if b and not a)
        n = len(pairs)
        return {"n_frames": n, "both": n11, "only_a": n10, "only_b": n01, "neither": n - n11 - n10 - n01,
                "rate_a": (n11 + n10) / n if n else float("nan"),
                "rate_b": (n11 + n01) / n if n else float("nan"),
                "a_minus_b": (n10 - n01) / n if n else float("nan"),
                "mcnemar_exact_p": mcnemar_exact(n10, n01)}

    per, pooled = {}, {"same_scene": [], "new_scene": [], "all": []}
    for k in keys:
        a = {r["frame"]: bool(r.get("identity_correct")) for r in rows_a.get(k) or []}
        b = {r["frame"]: bool(r.get("identity_correct")) for r in rows_b.get(k) or []}
        pairs = [(a[f], b[f]) for f in sorted(set(a) & set(b))]
        per[k] = {**counts(pairs), "unmatched_frames": len(set(a) ^ set(b))}
        pooled["all"] += pairs
        g = rc.scene_group(k)
        if g in pooled:
            pooled[g] += pairs
    diffs = np.asarray([per[k]["a_minus_b"] for k in keys], np.float64)
    return {"per_record": per, "pooled": {g: counts(p) for g, p in pooled.items() if p},
            "record_mean_a_minus_b": float(pr.unit_mean(diffs)) if diffs.size else float("nan"),
            "note": "exploratory frame-level counts; the record-level numbers are the protocol's"}


SELECTION_SOURCES = (("action", "t51"), ("noun", "noun"), ("combined", "combined"))
PAIRS = (("noun_best_gt", "action_best_gt"), ("combined_best_gt", "action_best_gt"),
         ("noun_best_gt", "l8h2"), ("action_best_gt", "l8h2"), ("combined_best_gt", "noun_best_gt"),
         ("noun_best_self", "action_best_self"))


def selections(t51: pathlib.Path, base: pathlib.Path, labels: Sequence[str], tag: str = ""
               ) -> dict[str, dict]:
    """``{<action|noun|combined>_best_<label>: ksel selected}`` from T51 (action) and T51b (a) / (b);
    ``tag`` = `t51_study.agg_tag` (``__scene`` reads the scene-mean selections)."""
    out = {}
    for label in labels:
        for name, where in SELECTION_SOURCES:
            d = pathlib.Path(t51) if where == "t51" else pathlib.Path(base) / where
            p = d / label / "ksel" / f"ksel{tag}.json"
            if p.is_file():
                js = json.loads(p.read_text())
                out[f"{name}_best_{label}"] = {**js["selected"], "path": str(p),
                                               "nested_mean": js.get("nested_mean")}
    missing = [n for n in ("action_best_gt", "noun_best_gt") if n not in out]
    if missing:
        raise SystemExit(f"heldout_compare needs {missing}: T51 gt/ksel{tag} (action) and "
                         "--omega noun --stage ksel --label gt (noun)"
                         + (f" --aggregate {tag[2:]}" if tag else ""))
    return out


def stage_heldout_compare(args, root: pathlib.Path) -> dict:
    cfg = _need()
    mode = st.agg_mode(args)
    tag = st.agg_tag(mode)
    base = root.parent
    keys = rc.resolve_keys(args.records, st.HELDOUT_KEYS)
    pairing = [rc.check_pairing(k) for k in keys]
    sel = selections(pathlib.Path(cfg.t51_dir), base, args.labels, tag)
    named = {n: v["combo"] for n, v in sel.items()}
    named["l8h2"] = st.SERVING.id
    sids = sorted(set(named.values()))
    rep = st.ensure_rows(root, args, {k: sids for k in keys}, need_self=False)
    if args.max_frames:
        return rep
    summ = st.heldout_summary(root, keys, sids, mode)
    rows = {s: {k: load_rows(root, k, s) or [] for k in keys} for s in sids}
    paired = {f"{a}__vs__{b}": {"a": named[a], "b": named[b],
                                **paired_identity(rows[named[a]], rows[named[b]], keys)}
              for a, b in PAIRS if a in named and b in named}
    payload = {**st.run_info("heldout_compare", args), "records": pairing, "selections": sel,
               "named": named, "family": {n: family(s) for n, s in named.items()},
               "scoring": "GT 3D identity_correct on every held-out frame (T51 P5), every source",
               "record_mean_gt_identity": {n: summ[s]["record_mean_gt_identity"] for n, s in named.items()},
               "per_record_gt_identity": {n: {k: summ[s]["per_record"][k]["gt_identity"] for k in keys}
                                          for n, s in named.items()},
               "by_scene": {n: summ[s]["by_scene"] for n, s in named.items()},
               "paired": paired, "row_origin": {s: {k: row_origin(k, s) for k in keys} for s in sids},
               "summary": summ,
               **({"aggregate": mode,
                   "scene_mean_gt_identity": {n: summ[s]["scene_mean_gt_identity"]
                                              for n, s in named.items()}}
                  if mode != "record" else {})}
    st.write_json(root / f"heldout_compare{tag}.json", payload)
    print(f"[heldout_compare{tag}] {len(sids)} sources x {len(keys)} records -> {root}")
    return payload


# ------------------------------------------------------------------------------ summary


def _bootstrap_view(p: pathlib.Path) -> Optional[dict]:
    if not p.is_file():
        return None
    js = json.loads(p.read_text())
    rows = [{**r, "family": family(r["cell"])} for r in js["all"]]
    by_fam: dict[str, float] = {}
    for r in rows:
        by_fam[r["family"]] = by_fam.get(r["family"], 0.0) + float(r["p_rank1"])
    return {"unit": js.get("unit"), "n_units": js.get("n_units"),
            "full_sample_top1": js.get("full_sample_top1"),
            "full_sample_top1_family": family(js["full_sample_top1"]) if js.get("full_sample_top1") else None,
            "full_sample_top1_p_rank1": js.get("full_sample_top1_p_rank1"),
            "stable_single_cell": js.get("stable_single_cell"),
            "p_rank1_by_family": by_fam, "cells": rows}


def _ksel_view(p: pathlib.Path) -> Optional[dict]:
    if not p.is_file():
        return None
    js = json.loads(p.read_text())
    sel = js["selected"]
    return {"selected": sel, "family": family(sel["combo"]), "nested_mean": js.get("nested_mean"),
            "cv_mean_ci95": js.get("cv_mean_ci95"), "single_head_fixed": js.get("single_head_fixed"),
            "ranking_all_calibration_top10": [{"cell": c, "family": family(c)}
                                              for c in js.get("ranking_all_calibration_top10", [])]}


def _score3d_view(p: pathlib.Path) -> Optional[dict]:
    if not p.is_file():
        return None
    js = json.loads(p.read_text())
    return {"primary": js.get("primary"), "top1": js.get("top1"),
            "n_candidates": len(js.get("candidates_3d", [])),
            "candidates_by_family": {f: sum(1 for c in js.get("candidates_3d", []) if family(c) == f)
                                     for f in ("action", "noun")},
            "ranking_top10": [{**r, "family": family(r["cell"])} for r in js.get("ranking", [])[:10]]}


def stage_summary(args, root: pathlib.Path) -> dict:
    cfg = _need()
    mode = st.agg_mode(args)
    tag = st.agg_tag(mode)
    base = root.parent
    dirs = {"action_t51": pathlib.Path(cfg.t51_dir), "noun": base / "noun", "combined": base / "combined"}
    out: dict[str, Any] = {**st.run_info("summary", args), "omegas": {}}
    if mode != "record":
        out["aggregate"] = mode
    for name, d in dirs.items():
        out["omegas"][name] = {
            label: {"score3d": _score3d_view(d / label / "score3d" / f"score3d{tag}.json"),
                    "bootstrap": _bootstrap_view(d / label / "bootstrap" / f"bootstrap{tag}.json"),
                    "ksel": _ksel_view(d / label / "ksel" / f"ksel{tag}.json")}
            for label in args.labels}
    hc = base / "heldout_compare" / f"heldout_compare{tag}.json"
    if hc.is_file():
        js = json.loads(hc.read_text())
        fields = ("named", "family", "record_mean_gt_identity", "per_record_gt_identity", "by_scene",
                  "paired") + (("scene_mean_gt_identity",) if mode != "record" else ())
        out["heldout_compare"] = {k: js[k] for k in fields}
    origin: dict[str, dict] = {}
    for name in ("noun", "combined"):
        for label in args.labels:
            p = base / name / label / "score3d" / f"score3d{tag}.json"
            if p.is_file():
                js = json.loads(p.read_text())
                tally: dict[str, int] = {}
                for k in js["records"]:
                    for c in js["candidates_3d"]:
                        o = row_origin(k, c)
                        tally[o] = tally.get(o, 0) + 1
                origin[f"{name}/{label}"] = tally
    out["score3d_row_origin"] = origin
    st.write_json(root / f"summary{tag}.json", out)
    print(f"[summary] -> {root / f'summary{tag}.json'}")
    return out


# ---------------------------------------------------------------------------------- main


def build_parser():
    ap = st.build_parser()
    ap.description = __doc__.split("\n")[0]
    for a in ap._actions:  # noqa: SLF001 — widen t51_study's --stage choices to T51b's stages
        if a.dest == "stage":
            a.choices = STAGES
            a.help = f"{list(PROTOCOL_STAGES)} (t51_study's, need --omega) or {list(OWN_STAGES)}"
    ap.add_argument("--omega", choices=tuple(OMEGAS), default=None,
                    help="protocol stages: noun (a) or combined (b) Ω")
    ap.add_argument("--t51-dir", default=None,
                    help="T51 verify dir (read only: selflabel/, rows/, ksel; default "
                         "$AG3S_RUN_DIR/verify/T51). Also the default --extra-dir")
    ap.add_argument("--noun-extra-dir", default=None,
                    help="t51b_noun_extract output (default $AG3S_RUN_DIR/verify/T51b/noun_extract)")
    ap.add_argument("--no-t51-rows", action="store_true",
                    help="do not read T51's 3D rows (replay every source into verify/T51b/rows)")
    return ap


def validate_args(ap, args) -> None:
    st.validate_args(ap, args)
    if args.stage in PROTOCOL_STAGES and args.omega is None:
        ap.error(f"--stage {args.stage} needs --omega {{noun,combined}}")
    if args.stage in OWN_STAGES and args.omega is not None:
        ap.error(f"--stage {args.stage} compares the Ωs; drop --omega")


def configure(args) -> tuple[Config, pathlib.Path]:
    """Resolve the directories, check reuse of T51's rows, install. Returns ``(cfg, verify/T51b)``."""
    from common.run_output import resolve_output_path

    run_dir = os.environ.get("AG3S_RUN_DIR")
    if not run_dir:
        raise SystemExit("AG3S_RUN_DIR is not set. Use the T51 run:\n"
                         "  export AG3S_RUN_DIR=<the run whose verify/T51 T51b builds on>")
    base = resolve_output_path("verify/T51b", run_dir=run_dir, default_name="verify/T51b")
    t51 = pathlib.Path(args.t51_dir) if args.t51_dir else pathlib.Path(run_dir) / "verify" / "T51"
    if not (t51 / "selflabel").is_dir():
        raise SystemExit(f"no T51 selflabel/ under {t51} — pass --t51-dir <T51 run>/verify/T51")
    rc.configure(args.extra_dir or t51)
    args.t49_dir = str(ne.t49_dir(args))  # t51_study's stages read args.t49_dir
    checked = () if args.no_t51_rows else check_replay_opts(t51, args)
    cfg = Config(omega=args.omega, t49_dir=args.t49_dir,
                 noun_extra_dir=str(args.noun_extra_dir or base / "noun_extract"), t51_dir=str(t51),
                 rows_dir=str(base / "rows"), t51_rows=not args.no_t51_rows, t51_rows_checked=checked)
    install(cfg)
    return cfg, base


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    validate_args(ap, args)
    cfg, base = configure(args)
    root = base / (args.omega if args.stage in PROTOCOL_STAGES else args.stage)
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fn = getattr(st, f"stage_{args.stage}") if args.stage in PROTOCOL_STAGES else globals()[f"stage_{args.stage}"]
    fn(args, root)
    print(f"[{STEP}] stage {args.stage} (omega {cfg.omega}) done in {time.time() - t0:.1f}s -> {root}")


if __name__ == "__main__":
    main()
