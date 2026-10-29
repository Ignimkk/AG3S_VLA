"""T54c record registry: 16 new, never-used test-split episodes (one per layout slot) and the selection rule.

`T54c.task.md` "새 data": for every layout slot 0–15 of ``meta/randomized_episodes.jsonl`` (split ``test``),
the **smallest episode index** that no record · log · other-session reservation has used. Metadata only —
no rollout result is read. This module holds

* the rule (`select_episodes`) and the evidence scan behind "used" (`scan_usage`): episode mentions in
  path names and text files of the shared storage (``results/``, ``recordings/``, ``assets/``), and in
  ``AG3S/ag3s/docs/**/*.md`` + every text file of ``AG3S/ag3s/docs/handoff/`` (+ ``doc/**/*.md``), plus
  `EXCLUDE_RESERVED` (episodes other sessions reserved, e.g. T58);
* the frozen result `EXPECTED` (the table the scan produced on 2026-10-13, before any T54c rollout) —
  `selection_problems` re-reads the dataset and refuses a mismatch;
* `records()`: a **separate module object** with the `t53_records` API (same code, executed again) whose
  registry is the T54c one and whose output directory is ``<run>/verify/T54c``. `t53_records` itself is
  never modified, so T53 code paths keep their behaviour; `t54c_extract` / `t54_study --registry t54c`
  read through this object.

What counts as a use (`PATTERNS`, all restricted to indices 1800–1999 = the test split):

=========  =====================================================  ==========================
name       form                                                   where
=========  =====================================================  ==========================
``ep``     ``ep1840`` · ``ep_1840`` (not preceded by a letter)    paths, storage text, docs
``scene``  ``scene1840`` · ``scene_1840``                         paths, storage text, docs
``epi``    ``episode 1840`` · ``"episode_index": 1840`` ·         paths, storage text, docs
           ``--episode-index 1840`` · ``episode=1840``
``seed``   ``s18401`` (the ``<N>1`` policy-seed convention)       paths, storage text, docs
``pair``   ``1840:18401`` (CL launcher ``EP:SEED`` lists)         storage text, docs
``list``   ``"episodes": [..., 1840, ...]``                       storage text, docs
``bare``   any standalone 1800–1999 number                        docs / handoff only
=========  =====================================================  ==========================

``bare`` is deliberately conservative (a doc that lists ``1840 · 1809 · …`` — the T53 table — has no
context word); a false positive only moves a slot to its next index. Paths whose name contains ``t54c``
(this STEP's own run outputs and documents) are skipped so that re-running the scan after the rollout does
not count T54c itself; the frozen `EXPECTED` remains the registry either way.

CPU only, no MuJoCo, no jax. Memory: one file at a time.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import importlib.util
import json
import os
import pathlib
import re
import sys
import time
import types
from typing import Any, Iterable, Mapping, Optional, Sequence

STEP = "T54c"
ROLE = "t54c_followup_test"
VERIFY_SUBDIR = "verify/T54c"
DATASET_SPLIT = "test"
TEST_RANGE = range(1800, 2000)
N_SLOTS = 16
#: policy seed = episode * 10 + 1 (`T54c.task.md` "policy seed = <N>1", the T53 convention).
SEED_SUFFIX = 1
#: T54c rollout ports (`T54c.task.md`: 8292 · 8293, the T47-line share). Even slot -> first port.
PORTS = (8292, 8293)

#: Episodes other sessions reserved — never selected, listed in the evidence table with the reason.
#: The lead passes further reservations here (or on the CLI, ``--exclude EP:REASON``).
EXCLUDE_RESERVED: dict[int, str] = {
    1925: "reserved by T58 (dev-70) / T42-1 library (seed 19251)",
    1830: "reserved by T58 (dev-70) / T42-1 library (seed 18301)",
    1623: "reserved by T58 (dev-70) / T42-1 library (seed 16231)",
}

#: The 16 selected episodes: episode -> (target fruit, used arm, layout slot, dataset prompt), as read from
#: ``randomized_episodes.jsonl`` (split ``test``) on 2026-10-13 by `select_episodes` over the
#: `scan_usage` evidence (``--stage select``; table in ``T54c.impl.md``). Order = layout slot.
EXPECTED: dict[int, tuple[str, str, int, str]] = {}  # filled below (see `_EXPECTED_ROWS`)

# ------------------------------------------------------------------------------ usage scan

_N = rb"(1[89]\d\d)"
PATTERNS: dict[str, re.Pattern] = {
    "ep": re.compile(rb"(?<![A-Za-z])ep[_-]?" + _N + rb"(?!\d)"),
    "scene": re.compile(rb"(?<![A-Za-z])scene[_-]?" + _N + rb"(?!\d)"),
    "epi": re.compile(rb"(?i)episode[_ -]?(?:index|idx|id)?[\"']?\s*[:=]?\s*" + _N + rb"(?!\d)"),
    "seed": re.compile(rb"(?<![A-Za-z0-9])s" + _N + rb"1(?!\d)"),
    "pair": re.compile(rb"(?<![\d.])" + _N + rb":\1\d(?!\d)"),
    "list": re.compile(rb"(?i)episodes[\"']?\s*[:=]\s*[\[(]([^\])]{0,4000})[\])]"),
    # standalone number: not inside a word / hex / decimal / path, not a ``py:1916`` or ``1916-1920`` line ref
    "bare": re.compile(rb"(?<![\w.:/\-])" + _N + rb"(?![\w.]|-\d)"),
}
PATH_PATTERNS = ("ep", "scene", "epi", "seed")
STORAGE_TEXT_PATTERNS = ("ep", "scene", "epi", "seed", "pair", "list")
DOC_PATTERNS = ("ep", "scene", "epi", "seed", "pair", "list", "bare")
TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".log", ".md", ".sh", ".py", ".csv", ".out", ".yaml",
                 ".yml", ".cmd", ".diff", ".err", ".tsv", ".toml", ".cfg", ".ini"}
_LIST_NUM = re.compile(rb"(?<![\d.])" + _N + rb"(?![\d.])")
#: Skip this STEP's own outputs / documents (see module doc).
SELF_SKIP = re.compile(r"(?i)t54c")
#: A file naming at least this many distinct test episodes (of 200) is a dataset-wide catalog (`UsageIndex`).
CATALOG_MIN_EPISODES = 100
STORAGE_PATH_ROOTS = ("results", "recordings", "assets")
STORAGE_TEXT_ROOTS = ("results", "recordings")


def _episodes_in(name: str, data: bytes) -> Iterable[tuple[int, int]]:
    """``(episode, offset)`` of every match of pattern `name` in `data` (test range only)."""
    pat = PATTERNS[name]
    for m in pat.finditer(data):
        if name == "list":
            for n in _LIST_NUM.finditer(m.group(1)):
                yield int(n.group(1)), m.start(1) + n.start(1)
        else:
            yield int(m.group(1)), m.start(1)


@dataclasses.dataclass
class UsageIndex:
    """Per episode: hit count, files, the source kinds and up to `max_examples` examples."""

    max_examples: int = 4
    hits: dict[int, dict] = dataclasses.field(default_factory=dict)
    scanned: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: path -> distinct test-range episodes it mentions (evidence; `catalog_files`)
    file_eps: dict[str, set] = dataclasses.field(default_factory=dict)

    def add(self, ep: int, *, kind: str, pattern: str, path: str, snippet: str, n: int = 1) -> None:
        if ep not in TEST_RANGE:
            return
        self.file_eps.setdefault(path, set()).add(ep)
        h = self.hits.setdefault(ep, {"n_hits": 0, "files": set(), "kinds": set(), "patterns": set(),
                                      "examples": []})
        h["n_hits"] += n
        h["kinds"].add(kind)
        h["patterns"].add(pattern)
        if path not in h["files"]:
            h["files"].add(path)
            if len(h["examples"]) < self.max_examples:
                h["examples"].append({"kind": kind, "pattern": pattern, "path": path,
                                      "snippet": snippet})

    def catalog_files(self, min_episodes: Optional[int] = None) -> dict[str, int]:
        """Files that mention at least `min_episodes` distinct test episodes (default
        `CATALOG_MIN_EPISODES` = half the split): dataset-wide catalogs (e.g. T43 S ``demo_grasp.json``,
        the training-demonstration geometry of every dataset episode), not a use of any one episode."""
        k = CATALOG_MIN_EPISODES if min_episodes is None else int(min_episodes)
        return {p: len(v) for p, v in self.file_eps.items() if len(v) >= k}

    def used(self) -> set[int]:
        """Episodes with at least one mention outside the catalog files."""
        cat = self.catalog_files()
        out: set[int] = set()
        for p, eps in self.file_eps.items():
            if p not in cat:
                out |= eps
        return out

    def as_json(self) -> dict:
        used = self.used()
        return {"scanned": self.scanned, "catalog_files": self.catalog_files(),
                "catalog_rule": f"a file mentioning >= {CATALOG_MIN_EPISODES} distinct test episodes is a "
                                "catalog; its mentions alone do not make an episode used",
                "catalog_only_episodes": sorted(set(self.hits) - used),
                "hits": {str(e): {"n_hits": h["n_hits"], "n_files": len(h["files"]),
                                  "kinds": sorted(h["kinds"]), "patterns": sorted(h["patterns"]),
                                  "examples": h["examples"]}
                         for e, h in sorted(self.hits.items())}}


def text_hits(data: bytes, patterns: Sequence[str]) -> list[tuple[int, str, int, str]]:
    """``[(episode, pattern, count, first snippet)]`` of one text (test range only)."""
    agg: dict[tuple[int, str], list] = {}
    for name in patterns:
        for ep, off in _episodes_in(name, data):
            if ep not in TEST_RANGE:
                continue
            k = (ep, name)
            if k in agg:
                agg[k][0] += 1
                continue
            lo, hi = max(0, off - 40), min(len(data), off + 44)
            agg[k] = [1, data[lo:hi].decode("utf-8", "replace").replace("\n", " ")]
    return [(ep, name, c, snip) for (ep, name), (c, snip) in agg.items()]


def scan_text(index: UsageIndex, data: bytes, *, path: str, kind: str, patterns: Sequence[str]) -> None:
    for ep, name, c, snip in text_hits(data, patterns):
        index.add(ep, kind=kind, pattern=name, path=path, snippet=snip, n=c)


def _scan_files_worker(args) -> tuple[int, int, list]:
    paths, patterns = args
    n, nbytes, out = 0, 0, []
    for p in paths:
        try:
            data = pathlib.Path(p).read_bytes()
        except OSError:
            continue
        n += 1
        nbytes += len(data)
        h = text_hits(data, patterns)
        if h:
            out.append((p, h))
    return n, nbytes, out


def scan_path_names(index: UsageIndex, root: pathlib.Path, *, kind: str) -> int:
    """Every directory and file name under `root` (no symlink follow). Returns the entry count."""
    n = 0
    for d, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [x for x in dirs if not SELF_SKIP.search(x)]
        for name in dirs + files:
            if SELF_SKIP.search(name):
                continue
            n += 1
            b = name.encode("utf-8", "replace")
            for pat in PATH_PATTERNS:
                for ep, _ in _episodes_in(pat, b):
                    index.add(ep, kind=kind, pattern=f"path:{pat}", path=os.path.join(d, name),
                              snippet=name)
    return n


def iter_text_files(root: pathlib.Path, *, suffixes: Optional[set] = None,
                    max_bytes: Optional[int] = None) -> Iterable[pathlib.Path]:
    suffixes = TEXT_SUFFIXES if suffixes is None else suffixes
    for d, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [x for x in dirs if not SELF_SKIP.search(x)]
        for name in files:
            if SELF_SKIP.search(name):
                continue
            p = pathlib.Path(d) / name
            if p.suffix.lower() not in suffixes or p.is_symlink():
                continue
            if max_bytes is not None and p.stat().st_size > max_bytes:
                continue
            yield p


def scan_text_files(index: UsageIndex, root: pathlib.Path, *, kind: str, patterns: Sequence[str],
                    suffixes: Optional[set] = None, max_bytes: Optional[int] = None,
                    log=None, workers: int = 1) -> dict:
    """Text contents under `root`; `workers` > 1 reads in a process pool (spawn; batches of 500 files,
    each worker holds one file at a time)."""
    t0 = time.time()
    paths = [str(p) for p in iter_text_files(root, suffixes=suffixes, max_bytes=max_bytes)]
    batches = [(paths[i:i + 500], tuple(patterns)) for i in range(0, len(paths), 500)]
    n, nbytes = 0, 0

    def merge(res):
        nonlocal n, nbytes
        n += res[0]
        nbytes += res[1]
        for p, hits in res[2]:
            for ep, name, c, snip in hits:
                index.add(ep, kind=kind, pattern=name, path=p, snippet=snip, n=c)

    if workers <= 1 or len(batches) <= 1:
        for b in batches:
            merge(_scan_files_worker(b))
    else:
        import concurrent.futures as cf
        import multiprocessing as mp

        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            for i, res in enumerate(ex.map(_scan_files_worker, batches)):
                merge(res)
                if log is not None and (i + 1) % 20 == 0:
                    log(f"  [{kind}] {n} files, {nbytes / 1e9:.2f} GB, {time.time() - t0:.0f}s")
    return {"files": n, "bytes": nbytes, "seconds": round(time.time() - t0, 1), "workers": workers}


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def storage_root() -> pathlib.Path:
    from common.workspace import shared_storage_root

    return pathlib.Path(shared_storage_root())


def scan_usage(*, storage: Optional[pathlib.Path] = None, repo: Optional[pathlib.Path] = None,
               storage_text: bool = True, extra_doc_roots: Sequence[pathlib.Path] = (),
               workers: int = 1, log=print) -> UsageIndex:
    """The evidence behind "used" (module doc). `storage_text=False` skips the storage text-content
    pass (path names only) — for smoke runs; the registry was made with it on."""
    storage = pathlib.Path(storage or storage_root())
    repo = pathlib.Path(repo or repo_root())
    idx = UsageIndex()
    for r in STORAGE_PATH_ROOTS:
        root = storage / r
        if root.is_dir():
            t0 = time.time()
            n = scan_path_names(idx, root, kind=f"storage_path:{r}")
            idx.scanned[f"storage_path:{r}"] = {"root": str(root), "entries": n,
                                                "seconds": round(time.time() - t0, 1)}
            log(f"  path names {root}: {n} entries")
    if storage_text:
        for r in STORAGE_TEXT_ROOTS:
            root = storage / r
            if root.is_dir():
                idx.scanned[f"storage_text:{r}"] = {"root": str(root), **scan_text_files(
                    idx, root, kind=f"storage_text:{r}", patterns=STORAGE_TEXT_PATTERNS, log=log,
                    workers=workers)}
                log(f"  text {root}: {idx.scanned[f'storage_text:{r}']}")
    docs = repo / "AG3S" / "ag3s" / "docs"
    if docs.is_dir():
        idx.scanned["docs_md"] = {"root": str(docs), **scan_text_files(
            idx, docs, kind="docs_md", patterns=DOC_PATTERNS, suffixes={".md"})}
        hand = docs / "handoff"
        if hand.is_dir():
            # handoff .md is in docs_md (with ``bare``); the json / txt hand-offs carry numbers of every
            # kind (counts, md5 hex, line refs), so only the context patterns apply to them
            idx.scanned["handoff"] = {"root": str(hand), **scan_text_files(
                idx, hand, kind="handoff", patterns=STORAGE_TEXT_PATTERNS,
                suffixes=TEXT_SUFFIXES - {".md"})}
    for extra in [repo / "doc", *map(pathlib.Path, extra_doc_roots)]:
        if extra.is_dir():
            idx.scanned[f"doc:{extra}"] = {"root": str(extra), **scan_text_files(
                idx, extra, kind=f"doc:{extra.name}", patterns=STORAGE_TEXT_PATTERNS,
                suffixes={".md"})}
    return idx


# ---------------------------------------------------------------------------- selection


DATASET_REL = "datasets/rby1/rby1_randomized_pick_place_16d_v1"


def load_dataset_meta(dataset: Optional[pathlib.Path] = None) -> dict[int, dict]:
    path = pathlib.Path(dataset or storage_root() / DATASET_REL) / "meta" / "randomized_episodes.jsonl"
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[int(r["episode_index"])] = r
    return out


def select_episodes(meta: Mapping[int, Mapping[str, Any]], used: Iterable[int],
                    reserved: Mapping[int, str]) -> tuple[dict[int, int], list[dict]]:
    """``({slot: episode}, table)``: per layout slot the smallest test-split index neither used nor
    reserved. `table` = one row per test episode examined in index order up to the chosen one (the
    exclusion evidence rows), plus the chosen row."""
    used, chosen, table = set(used), {}, []
    by_slot: dict[int, list[int]] = {}
    for ep, r in meta.items():
        if r.get("split") == DATASET_SPLIT:
            by_slot.setdefault(int(r["layout_index"]), []).append(int(ep))
    for slot in range(N_SLOTS):
        for ep in sorted(by_slot.get(slot, [])):
            reasons = []
            if ep in reserved:
                reasons.append(f"reserved: {reserved[ep]}")
            if ep in used:
                reasons.append("used (scan evidence)")
            table.append({"slot": slot, "episode": ep, "excluded": bool(reasons), "reasons": reasons})
            if not reasons:
                chosen[slot] = ep
                break
        else:
            raise SystemExit(f"layout slot {slot}: every test episode is used or reserved")
    return chosen, table


def registry_rows(meta: Mapping[int, Mapping[str, Any]], chosen: Mapping[int, int]) -> dict[int, tuple]:
    out = {}
    for slot in sorted(chosen):
        r = meta[chosen[slot]]
        out[chosen[slot]] = (r["target_fruit"], r["used_arm"], int(r["layout_index"]), r["prompt"])
    return out


def seed_of(episode: int) -> int:
    return int(episode) * 10 + SEED_SUFFIX


def port_of(slot: int) -> int:
    return PORTS[int(slot) % 2]


def seeds_table(expected: Optional[Mapping[int, tuple]] = None) -> list[dict]:
    """``[{slot, episode, seed, port}]`` in slot order (the T53 ``seeds.json`` layout)."""
    expected = EXPECTED if expected is None else expected
    rows = sorted(((v[2], e) for e, v in expected.items()))
    return [{"slot": s, "episode": e, "seed": seed_of(e), "port": port_of(s)} for s, e in rows]


def selection_problems(meta: Mapping[int, Mapping[str, Any]],
                       expected: Optional[Mapping[int, tuple]] = None,
                       reserved: Optional[Mapping[int, str]] = None) -> list[str]:
    """`EXPECTED` against the dataset (target, arm, slot, prompt, split), one per slot 0–15, none
    reserved, none in T53."""
    from AG3S.ag3s.experiments.studies import t53_records as rc53

    expected = EXPECTED if expected is None else expected
    reserved = EXCLUDE_RESERVED if reserved is None else reserved
    out = []
    for ep, (target, arm, slot, prompt) in expected.items():
        r = meta.get(ep)
        if r is None:
            out.append(f"episode {ep} missing from the dataset metadata")
            continue
        got = (r.get("target_fruit"), r.get("used_arm"), r.get("layout_index"), r.get("prompt"),
               r.get("split"))
        want = (target, arm, slot, prompt, DATASET_SPLIT)
        if got != want:
            out.append(f"episode {ep}: dataset (target, arm, slot, prompt, split) {got} != {want}")
        if ep in reserved:
            out.append(f"episode {ep} is reserved ({reserved[ep]})")
        if ep in rc53.EXPECTED:
            out.append(f"episode {ep} is a T53 episode")
    slots = sorted(v[2] for v in expected.values())
    if slots != list(range(N_SLOTS)):
        out.append(f"layout slots {slots} are not 0..15 once each")
    return out


# -------------------------------------------------------------------- frozen registry

#: (slot, episode, target, used arm, prompt) — written by ``--stage select`` on 2026-10-13 (see
#: ``T54c.impl.md`` for the evidence table). `selection_problems` re-checks it against the dataset.
_EXPECTED_ROWS: tuple[tuple[int, int, str, str, str], ...] = (
    (0, 1856, 'orange', 'left', 'Place the orange into the basket.'),
    (1, 1825, 'apple', 'right', 'Put the apple in the basket.'),
    (2, 1826, 'banana', 'right', 'Pick up the banana and place it in the basket.'),
    (3, 1811, 'pear', 'right', 'Place the pear into the basket.'),
    (4, 1844, 'apple', 'right', 'Place the apple into the basket.'),
    (5, 1813, 'pear', 'right', 'Pick up the pear and place it in the basket.'),
    (6, 1814, 'pear', 'right', 'Place the pear into the basket.'),
    (7, 1831, 'pear', 'left', 'Put the pear in the basket.'),
    (8, 1816, 'orange', 'right', 'Place the orange into the basket.'),
    (9, 1817, 'apple', 'right', 'Pick up the apple and place it in the basket.'),
    (10, 1882, 'banana', 'left', 'Put the banana in the basket.'),
    (11, 1835, 'orange', 'left', 'Put the orange in the basket.'),
    (12, 1820, 'banana', 'right', 'Put the banana in the basket.'),
    (13, 1821, 'orange', 'left', 'Pick up the orange and place it in the basket.'),
    (14, 1806, 'orange', 'right', 'Pick up the orange and place it in the basket.'),
    (15, 1903, 'apple', 'right', 'Put the apple in the basket.'),
)
#: The test episodes the scan excluded below a chosen one (slot order), with the first evidence — the
#: exclusion table of 2026-10-13 (full table: t54c_selection.json evidence_table).
KNOWN_USED: dict[int, str] = {
    1808: "1453 files [bare,ep,epi,list,pair,path,seed] e.g. figures/t30/t30-scene-pregrasp-E3a-E3a_ep1808_r1.json",
    1824: "266 files [bare,ep,epi,pair,path,scene,seed] e.g. figures/t39/t39-scene-1824_18241.png",
    1840: "103 files [bare,ep,epi,list,pair,path,scene,seed] e.g. test/rows/ep1840__hold.jsonl",
    1809: "94 files [bare,ep,epi,list,pair,path,scene,seed] e.g. test/rows/ep1809__hold_attach.log",
    1810: "21 files [bare,ep,epi,path] e.g. 2/smoke/t42_2_smoke_ep1810_crate_post.png",
    1812: "docs bare number only: AG3S_T47_LOG.md table value '| 1803 | 1812 |' (+ the demo_grasp catalog)",
    1828: "733 files [bare,ep,epi,list,pair,path,scene,seed] e.g. figures/t40/t40a-scene-1828_18281.png",
    1815: "75 files [bare,ep,epi,list,pair,path,scene,seed] e.g. test/rows/ep1815__current.meta.json",
    1800: "3962 files [bare,ep,epi,list,pair,path,scene,seed] e.g. figures/subtask-d/subtask-d-scene-1800_s18001.json",
    1801: "5 files [bare,ep,epi,path] e.g. live_test/20260924_t0/t0_ep1801",
    1802: "21 files [bare,ep,epi,path] e.g. 2/smoke/t42_2_smoke_ep1802_overbar.png",
    1818: "209 files [bare,ep,epi,pair,path,seed] e.g. Z5/replay_t39/E0_B_E0_ep1818_s18181.npz",
    1834: "1869 files [bare,ep,epi,list,pair,path,scene,seed] e.g. figures/t42/t42-2-scene-1834-place.png",
    1850: "712 files [bare,ep,epi,list,path,scene] e.g. t1/cards/ep1850",
    1866: "81 files [bare,ep,epi,list,pair,path,scene,seed] e.g. T54/scene/ep1866_f60.npz",
    1803: "31 files [bare,ep,epi,path] e.g. live_test/20260924_t0/t0_ep1803",
    1819: "258 files [bare,ep,epi,pair,path,scene,seed] e.g. figures/t39/t39-scene-1819_18192.json",
    1804: "77 files [bare,ep,epi,list,pair,path,scene,seed] e.g. test/rows/ep1804__hold.meta.json",
    1805: "5 files [bare,ep,epi,path] e.g. live_test/20260924_t0/t0_ep1805",
    1807: "7119 files [bare,ep,epi,list,pair,path,scene,seed] e.g. T48b/render/ep1807__E0.npz",
    1823: "59 files [ep,epi,pair,path,seed] e.g. T43R2/scan/T42_1_runs_E3b_noobs_E3b_ep1823_s18231.json",
    1839: "74 files [bare,ep,epi,list,pair,path,scene,seed] e.g. test/rows/ep1839__current.jsonl",
    1855: "docs bare number only: handoff/T43Q.impl.md line ref '1617-1621, 1855' (+ the demo_grasp catalog)",
    1871: "docs bare number only: handoff/T23.impl.md diff hunk '+1871,8' (+ the demo_grasp catalog)",
    1887: "252 files [bare,ep,epi,pair,path,scene,seed] e.g. figures/t39/t39-scene-1887_18871.png",
}
EXPECTED.update({e: (t, a, s, p) for s, e, t, a, p in _EXPECTED_ROWS})
EPISODES = tuple(EXPECTED)


def _clone_t53_records() -> types.ModuleType:
    """`t53_records` executed again as an independent module, then re-pointed at T54c."""
    # The clone executes under the original's module name; `dataclasses` resolves string annotations
    # through sys.modules[that name], so the original must be imported first (t54_study's path did not).
    from AG3S.ag3s.experiments.studies import t53_records as _original  # noqa: F401

    spec = importlib.util.find_spec("AG3S.ag3s.experiments.studies.t53_records")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # never put in sys.modules: `t53_records` stays the original
    mod.__name__ = "AG3S.ag3s.experiments.studies.t54c_records.records"
    configure_registry(mod, EXPECTED)
    return mod


def configure_registry(mod: types.ModuleType, expected: Mapping[int, tuple], *,
                       step: str = STEP, role: str = ROLE, subdir: str = VERIFY_SUBDIR) -> None:
    """Point a `t53_records`-API module object at another registry and output sub-directory."""
    mod.STEP = step
    mod.ROLE = role
    mod.EXPECTED = dict(expected)
    mod.EPISODES = tuple(expected)
    Ep = mod.T53Episode
    mod.REGISTRY = {f"ep{e}": Ep(f"ep{e}", e, t, a, s, p, role) for e, (t, a, s, p) in expected.items()}
    mod.KEYS = tuple(mod.REGISTRY)

    def t53_dir_of_run(run_dir, *, check_storage: bool = True) -> pathlib.Path:
        if check_storage:
            from common.run_output import resolve_output_path

            try:
                return resolve_output_path(subdir, run_dir=run_dir, default_name=subdir)
            except ValueError as exc:
                raise SystemExit(f"--run-dir {run_dir}: {exc}") from exc
        return pathlib.Path(run_dir) / subdir

    def provenance(keys):
        return {"records": list(keys), "role": role, "scene_of": {k: mod.scene_of(k) for k in keys},
                "n_scenes": len({mod.scene_of(k) for k in keys}),
                "vla_training_episodes": [k for k in keys
                                          if mod.episode_of(k).episode in mod.VLA_TRAIN_EPISODES],
                "note": (f"{step}: test-split episodes chosen by the pre-fixed rule (t54c_records) "
                         "and never used before; results here must not change settings")}

    mod.t53_dir_of_run = t53_dir_of_run
    mod.provenance = provenance
    mod.configure()          # reset any directory


_RECORDS: Optional[types.ModuleType] = None


def records() -> types.ModuleType:
    """The T54c registry with the `t53_records` API (cached per process)."""
    global _RECORDS
    if _RECORDS is None:
        if not EXPECTED:
            raise SystemExit("t54c_records.EXPECTED is empty — run --stage select first")
        _RECORDS = _clone_t53_records()
    return _RECORDS


def registry_module(name: str) -> types.ModuleType:
    """``t53`` -> `t53_records` itself, ``t54c`` -> `records()`."""
    if name == "t53":
        from AG3S.ag3s.experiments.studies import t53_records as rc53

        return rc53
    if name == "t54c":
        return records()
    raise SystemExit(f"unknown registry {name!r} (t53 | t54c)")


# ------------------------------------------------------------------------------------ CLI


def kst_now() -> str:
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat()


def parse_exclude(items: Sequence[str]) -> dict[int, str]:
    out = {}
    for it in items or ():
        ep, _, reason = str(it).partition(":")
        out[int(ep)] = reason or "reserved (CLI)"
    return out


def evidence_table(table: Sequence[dict], index: UsageIndex, reserved: Mapping[int, str]) -> list[dict]:
    """The selection rows with the evidence of every exclusion (first examples, counts)."""
    hits = index.as_json()["hits"]
    out = []
    for r in table:
        h = hits.get(str(r["episode"]))
        out.append({**r, "reserved_reason": reserved.get(r["episode"]),
                    "n_hits": 0 if h is None else h["n_hits"],
                    "n_files": 0 if h is None else h["n_files"],
                    "kinds": [] if h is None else h["kinds"],
                    "patterns": [] if h is None else h["patterns"],
                    "examples": [] if h is None else h["examples"][:2]})
    return out


def stage_select(args) -> dict:
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reserved = {**EXCLUDE_RESERVED, **parse_exclude(args.exclude)}
    t0 = time.time()
    idx = scan_usage(storage_text=not args.paths_only, extra_doc_roots=args.extra_doc_roots,
                     workers=args.workers)
    (out / "t54c_usage_index.json").write_text(json.dumps(idx.as_json(), indent=1, ensure_ascii=False))
    meta = load_dataset_meta(args.dataset)
    chosen, table = select_episodes(meta, idx.used(), reserved)
    rows = registry_rows(meta, chosen)
    ev = evidence_table(table, idx, reserved)
    reserved_rows = [{"episode": e, "reason": r, "split": (meta.get(e) or {}).get("split"),
                      "layout_slot": (meta.get(e) or {}).get("layout_index"),
                      "scan_hits": idx.as_json()["hits"].get(str(e), {}).get("n_hits", 0)}
                     for e, r in sorted(reserved.items())]
    payload = {"step": STEP, "stage": "select", "started_kst": kst_now(), "argv": sys.argv,
               "rule": ("per layout slot 0-15 of the test split: the smallest episode index with no "
                        "usage evidence and not reserved"),
               "seconds": round(time.time() - t0, 1), "scanned": idx.scanned,
               "catalog_files": idx.catalog_files(),
               "catalog_only_episodes": sorted(set(idx.hits) - idx.used()),
               "reserved": reserved_rows,
               "chosen": [{"slot": s, "episode": e, "target": rows[e][0], "used_arm": rows[e][1],
                           "prompt": rows[e][3], "seed": seed_of(e), "port": port_of(s)}
                          for s, e in sorted(chosen.items())],
               "evidence_table": ev,
               "frozen_registry_matches": (None if not EXPECTED else
                                           {e: rows.get(e) == v for e, v in EXPECTED.items()}),
               "frozen_registry_problems": (selection_problems(meta) if EXPECTED else None)}
    (out / "t54c_selection.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(json.dumps(payload["chosen"], indent=1, ensure_ascii=False))
    return payload


def stage_check(args) -> dict:
    meta = load_dataset_meta(args.dataset)
    probs = selection_problems(meta, reserved={**EXCLUDE_RESERVED, **parse_exclude(args.exclude)})
    print(json.dumps({"expected": seeds_table(), "problems": probs}, indent=1))
    if probs:
        raise SystemExit("selection problems:\n  " + "\n  ".join(probs))
    return {"problems": probs}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("select", "check"), required=True)
    ap.add_argument("--out", default=None, help="select: output dir (e.g. <run>/verify/T54c/selection)")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--exclude", nargs="*", default=[], metavar="EP:REASON",
                    help="extra reserved episodes (added to EXCLUDE_RESERVED)")
    ap.add_argument("--extra-doc-roots", nargs="*", default=[])
    ap.add_argument("--paths-only", action="store_true", help="smoke: skip the storage text pass")
    ap.add_argument("--workers", type=int, default=8,
                    help="processes for the storage text pass (each holds one file; ~8.8 GB of text)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.stage == "select" and not args.out:
        ap.error("--stage select needs --out")
    {"select": stage_select, "check": stage_check}[args.stage](args)


__all__ = ["CATALOG_MIN_EPISODES", "EPISODES", "EXCLUDE_RESERVED", "EXPECTED", "KNOWN_USED", "PATTERNS", "PORTS", "ROLE", "STEP",
           "UsageIndex", "configure_registry", "evidence_table", "load_dataset_meta", "main",
           "port_of", "records", "registry_module", "registry_rows", "scan_text", "scan_usage",
           "seed_of", "seeds_table", "select_episodes", "selection_problems"]


if __name__ == "__main__":
    main()
