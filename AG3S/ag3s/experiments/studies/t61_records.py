"""T61 held-out registry (16 new test-split episodes), its extraction wrapper and its rollout launcher.

`handoff/T61.task.md` "사전 고정 — held-out": no existing record was unused by selection / development, so 16 new
episodes are rolled out exactly as T53 / T54c (AG3S ON current serving, C1a flags, policy seed = episode·10+1,
600 step, ``--record-ag3s``). The selection rule (metadata only, fixed before any rollout):

* test split; the 88 episodes that appear in any repo document or storage ``results`` directory are "used";
* one episode per layout slot 0–15; exactly 4 per target fruit; exactly 8 per used arm; at most 3 per
  (fruit, arm) — (apple, right) at most 4 because no unused (apple, left) test episode exists;
* depth-first over slots 0 → 15, episode index ascending; the first feasible assignment.

`EXPECTED` is that result (frozen 2026-10-14); `selection_problems` re-checks it against the dataset.
Everything else reuses the T54c machinery unchanged:

* `records()` — `t53_records` executed again and re-pointed at this registry / ``verify/T61`` (as T54c);
* `extract_main` — `t53_extract` with this registry (as `t54c_extract`);
* `write_rollout_p` — `t54c_rollout.write_rollout_p` with this registry, then the T54c names in the generated
  ``p/`` renamed to T61, and the snapshot rule widened to allow offline-analysis code changes since
  ``cf38c9db`` (``AG3S/ag3s/experiments/studies/`` and ``tests/``; the six serving files are md5-checked).

usage::

    .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.t61_records --stage check
    .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.t61_records --stage rollout-p --run-dir $R --src-p <T53 p>
    .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.t61_records --stage extract -- <t53_extract args>
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import pathlib
import sys
import types
from typing import Iterator, Mapping, Optional, Sequence

STEP = "T61"
ROLE = "t61_heldout"
VERIFY_SUBDIR = "verify/T61"
N_SLOTS = 16
PORTS = (8292, 8293)
DATASET = pathlib.Path("/mnt/dev/AG3S_VLA-storage/datasets/rby1/rby1_randomized_pick_place_16d_v1/meta/randomized_episodes.jsonl")

#: (slot, episode, target, used arm, prompt) — frozen 2026-10-14 before any rollout.
_EXPECTED_ROWS: tuple[tuple[int, int, str, str, str], ...] = (
    (0, 1872, 'pear', 'left', 'Put the pear in the basket.'),
    (1, 1841, 'banana', 'left', 'Pick up the banana and place it in the basket.'),
    (2, 1858, 'pear', 'right', 'Pick up the pear and place it in the basket.'),
    (3, 1827, 'banana', 'left', 'Pick up the banana and place it in the basket.'),
    (4, 1876, 'banana', 'right', 'Pick up the banana and place it in the basket.'),
    (5, 1829, 'pear', 'right', 'Pick up the pear and place it in the basket.'),
    (6, 1846, 'orange', 'left', 'Pick up the orange and place it in the basket.'),
    (7, 1879, 'apple', 'right', 'Pick up the apple and place it in the basket.'),
    (8, 1880, 'banana', 'left', 'Pick up the banana and place it in the basket.'),
    (9, 1833, 'apple', 'right', 'Pick up the apple and place it in the basket.'),
    (10, 1898, 'apple', 'right', 'Put the apple in the basket.'),
    (11, 1867, 'apple', 'right', 'Place the apple into the basket.'),
    (12, 1916, 'pear', 'left', 'Place the pear into the basket.'),
    (13, 1869, 'orange', 'right', 'Place the orange into the basket.'),
    (14, 1838, 'orange', 'left', 'Place the orange into the basket.'),
    (15, 1919, 'orange', 'left', 'Place the orange into the basket.'),
)
EXPECTED: dict[int, tuple[str, str, int, str]] = {e: (t, a, s, p) for s, e, t, a, p in _EXPECTED_ROWS}
EPISODES = tuple(EXPECTED)
#: The conservative "used" set of the scan (2026-10-14): test episodes named in repo docs or storage results.
USED: tuple[int, ...] = (
    1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816,
    1817, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1828, 1830, 1831, 1834, 1835, 1836, 1839,
    1840, 1842, 1843, 1844, 1845, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1860, 1861,
    1862, 1863, 1866, 1871, 1878, 1881, 1882, 1883, 1887, 1888, 1896, 1900, 1903, 1907, 1908, 1914, 1921,
    1925, 1927, 1939, 1940, 1952, 1957, 1958, 1959, 1965, 1967, 1968, 1972, 1976, 1977, 1979, 1982, 1983,
    1984, 1992, 1995)


def seed_of(ep: int) -> int:
    return int(ep) * 10 + 1


def port_of(slot: int) -> int:
    return PORTS[int(slot) % 2]


def seeds_table(expected: Optional[Mapping[int, tuple]] = None) -> list[dict]:
    expected = EXPECTED if expected is None else expected
    rows = sorted(((v[2], e) for e, v in expected.items()))
    return [{"slot": s, "episode": e, "seed": seed_of(e), "port": port_of(s)} for s, e in rows]


def load_meta(path: pathlib.Path = DATASET) -> dict[int, dict]:
    return {json.loads(l)["episode_index"]: json.loads(l) for l in path.read_text().splitlines() if l.strip()}


def selection_problems(meta: Mapping[int, Mapping], expected: Optional[Mapping[int, tuple]] = None) -> list[str]:
    from collections import Counter

    expected = EXPECTED if expected is None else expected
    out = []
    for ep, (target, arm, slot, prompt) in expected.items():
        r = meta.get(ep)
        if r is None:
            out.append(f"episode {ep} missing")
            continue
        got = (r.get("target_fruit"), r.get("used_arm"), r.get("layout_index"), r.get("prompt"), r.get("split"))
        if got != (target, arm, slot, prompt, "test"):
            out.append(f"episode {ep}: dataset {got} != {(target, arm, slot, prompt, 'test')}")
        if ep in USED:
            out.append(f"episode {ep} is in the used set")
    if sorted(v[2] for v in expected.values()) != list(range(N_SLOTS)):
        out.append("layout slots are not 0..15 once each")
    fr = Counter(v[0] for v in expected.values()); ar = Counter(v[1] for v in expected.values())
    if set(fr.values()) != {4} or set(ar.values()) != {8}:
        out.append(f"fruit / arm counts {dict(fr)} {dict(ar)} are not 4 each / 8 each")
    return out


# ------------------------------------------------------------------------------ registry (as T54c)

_RECORDS: Optional[types.ModuleType] = None


def records() -> types.ModuleType:
    """`t53_records` executed again, pointed at the T61 registry and ``verify/T61``."""
    global _RECORDS
    if _RECORDS is None:
        from AG3S.ag3s.experiments.studies import t53_records as _original  # noqa: F401
        from AG3S.ag3s.experiments.studies import t54c_records as t54c

        spec = importlib.util.find_spec("AG3S.ag3s.experiments.studies.t53_records")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.__name__ = "AG3S.ag3s.experiments.studies.t61_records.records"
        t54c.configure_registry(mod, EXPECTED, step=STEP, role=ROLE, subdir=VERIFY_SUBDIR)
        _RECORDS = mod
    return _RECORDS


@contextlib.contextmanager
def t61_registry() -> Iterator:
    from AG3S.ag3s.experiments.studies import t53_extract as ex

    saved = (ex.rc, ex.STEP)
    ex.rc, ex.STEP = records(), STEP
    try:
        yield ex
    finally:
        ex.rc, ex.STEP = saved


def extract_main(argv: Optional[Sequence[str]] = None) -> None:
    with t61_registry() as ex:
        ex.main(argv)


# ------------------------------------------------------------------------------ rollout p/ (as T54c)

RENAMES = (("launch_t54c", "launch_t61"), ("code_md5_t54c", "code_md5_t61"), ("T54C_GPU_LOCK_HELD", "T61_GPU_LOCK_HELD"),
           ("verify/T54c", VERIFY_SUBDIR), ("T54c", "T61"), ("t54c", "t61"))
OLD_NONDOC = r"grep -vE '\.md$|^AG3S/ag3s/docs/|^doc/|^paper/|^papers/'"
NEW_NONDOC = r"grep -vE '\.md$|^AG3S/ag3s/docs/|^doc/|^paper/|^papers/|^AG3S/ag3s/experiments/studies/|^tests/'"


def write_rollout_p(src_p: pathlib.Path, run_dir: pathlib.Path, *, force: bool = False) -> dict:
    from AG3S.ag3s.experiments.studies import t54c_rollout as tr

    shim = types.SimpleNamespace(EXPECTED=EXPECTED, N_SLOTS=N_SLOTS, PORTS=PORTS, VERIFY_SUBDIR=VERIFY_SUBDIR,
                                 seeds_table=seeds_table)
    saved = (tr.reg, tr.STEP)
    tr.reg, tr.STEP = shim, STEP
    try:
        rec = tr.write_rollout_p(src_p, run_dir, force=force)
    finally:
        tr.reg, tr.STEP = saved
    p = pathlib.Path(rec["run"]) / VERIFY_SUBDIR / "p"
    renamed = {}
    for f in sorted(p.iterdir()):
        if f.is_dir():
            continue
        text = f.read_text()
        n_total = 0
        for old, new in RENAMES:
            n_total += text.count(old)
            text = text.replace(old, new)
        if f.name == "make_snapshot.sh":
            if text.count(OLD_NONDOC) != 1:
                raise SystemExit("make_snapshot.sh: non-doc filter not found exactly once")
            text = text.replace(OLD_NONDOC, NEW_NONDOC)
        f.write_text(text)
        renamed[f.name] = n_total
        if "t54c" in f.name:
            f.rename(f.with_name(f.name.replace("t54c", "t61")))
    rec["t61_renames"] = renamed
    rec["t61_snapshot_rule"] = ("allowed non-doc changes since cf38c9db: AG3S/ag3s/experiments/studies/ and tests/ "
                                "(offline analysis only); serving files are compared by md5_files.sh")
    (p / "GENERATED.json").write_text(json.dumps(rec, indent=1))
    return rec


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    sep = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("check", "rollout-p", "extract"), required=True)
    ap.add_argument("--run-dir"); ap.add_argument("--src-p"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv[:sep])
    if a.stage == "check":
        probs = selection_problems(load_meta())
        print(json.dumps({"problems": probs, "seeds": seeds_table()}, indent=1))
        raise SystemExit(1 if probs else 0)
    if a.stage == "rollout-p":
        rec = write_rollout_p(pathlib.Path(a.src_p), pathlib.Path(a.run_dir), force=a.force)
        print(json.dumps({k: rec[k] for k in ("run", "ports", "t61_renames")}, indent=1))
        return
    extract_main(argv[sep + 1:])


if __name__ == "__main__":
    main()
