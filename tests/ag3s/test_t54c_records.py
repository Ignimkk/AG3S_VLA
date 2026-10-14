"""T54c registry, selection rule and usage scan (`t54c_records`), the extract wrapper (`t54c_extract`) and the
`t54_study` registry option. CPU only; the dataset check runs only when the shared storage is present."""

from __future__ import annotations

import pathlib

import pytest

from AG3S.ag3s.experiments.studies import t53_records as rc53
from AG3S.ag3s.experiments.studies import t54c_records as T


def _meta(slots: dict[int, list[int]], split="test"):
    out = {}
    for s, eps in slots.items():
        for e in eps:
            out[e] = {"episode_index": e, "split": split, "layout_index": s, "target_fruit": "apple",
                      "used_arm": "left" if e % 2 else "right", "prompt": f"p{e}"}
    return out


# ------------------------------------------------------------------------------ the rule


def test_select_takes_the_smallest_unused_unreserved_index_per_slot():
    slots = {s: [1800 + s, 1816 + s, 1832 + s] for s in range(16)}
    meta = _meta(slots)
    meta[1700] = {"episode_index": 1700, "split": "validation", "layout_index": 0}
    used = {1800, 1816, 1805}
    chosen, table = T.select_episodes(meta, used, {1801: "reserved by X"})
    assert chosen[0] == 1832 and chosen[1] == 1817 and chosen[5] == 1821 and chosen[2] == 1802
    rows0 = [r for r in table if r["slot"] == 0]
    assert [r["episode"] for r in rows0] == [1800, 1816, 1832]
    assert rows0[0]["excluded"] and not rows0[-1]["excluded"]
    r1 = [r for r in table if r["slot"] == 1][0]
    assert r1["reasons"] == ["reserved: reserved by X"]
    with pytest.raises(SystemExit):
        T.select_episodes(meta, set(range(1800, 1900)), {})


def test_reserved_set_has_the_t58_episodes():
    assert set(T.EXCLUDE_RESERVED) >= {1925, 1830, 1623}
    assert all("T58" in r for r in T.EXCLUDE_RESERVED.values())
    assert T.parse_exclude(["1999:held by T60", "1998"]) == {1999: "held by T60", 1998: "reserved (CLI)"}


# ------------------------------------------------------------------------------ the scan


@pytest.mark.parametrize("text,eps", [
    (b"C1a_ep1840_s18401", {1840}),
    (b'"episode_index": 1842,', {1842}),
    (b"--episode-index 1843 --max-steps", {1843}),
    (b"scene1809 and scene_1810", {1809, 1810}),
    (b"LIST 1925:19251 1830:18301", {1925, 1830}),
    (b'"episodes": [1801, 1802, 300]', {1801, 1802}),
    (b"step1850 rep1851 1.1855 18555 T18:45 2026-10-13", set()),
    (b"ep0 ep500 ep1200 ep2001", set()),
])
def test_context_patterns(text, eps):
    idx = T.UsageIndex()
    T.scan_text(idx, text, path="f", kind="k", patterns=T.STORAGE_TEXT_PATTERNS)
    assert idx.used() == eps


def test_bare_numbers_count_only_in_docs():
    idx = T.UsageIndex()
    T.scan_text(idx, "1840 · 1809 · 1842".encode(), path="d.md", kind="docs_md", patterns=T.DOC_PATTERNS)
    assert idx.used() == {1840, 1809, 1842}
    idx2 = T.UsageIndex()
    T.scan_text(idx2, b"1840 1809", path="x.json", kind="storage", patterns=T.STORAGE_TEXT_PATTERNS)
    assert idx2.used() == set()


def test_path_scan_skips_t54c_outputs(tmp_path):
    (tmp_path / "runs" / "C1a_ep1811_s18111").mkdir(parents=True)
    (tmp_path / "verify" / "T54c" / "runs" / "C1a_ep1812_s18121").mkdir(parents=True)
    (tmp_path / "x_T54c_ep1813.json").write_text("{}")
    idx = T.UsageIndex()
    n = T.scan_path_names(idx, tmp_path, kind="p")
    assert idx.used() == {1811} and n >= 3
    h = idx.as_json()["hits"]["1811"]
    assert h["n_files"] == 1 and h["examples"][0]["pattern"] == "path:ep"


def test_text_scan_reads_only_text_files(tmp_path):
    (tmp_path / "a.json").write_text('{"episode_index": 1814}')
    (tmp_path / "b.npz").write_bytes(b"ep1815")
    (tmp_path / "T54c.md").write_text("ep1816")
    idx = T.UsageIndex()
    info = T.scan_text_files(idx, tmp_path, kind="t", patterns=T.STORAGE_TEXT_PATTERNS)
    assert idx.used() == {1814} and info["files"] == 1


# ------------------------------------------------------------------------- frozen registry


def test_frozen_registry_shape():
    assert len(T.EXPECTED) == 16
    assert sorted(v[2] for v in T.EXPECTED.values()) == list(range(16))
    assert not set(T.EXPECTED) & set(rc53.EXPECTED)
    assert not set(T.EXPECTED) & set(T.EXCLUDE_RESERVED)
    assert all(e in T.TEST_RANGE for e in T.EXPECTED)
    seeds = T.seeds_table()
    assert [x["slot"] for x in seeds] == list(range(16))
    assert all(x["seed"] == x["episode"] * 10 + 1 for x in seeds)
    assert [x["port"] for x in seeds] == [8292, 8293] * 8


def test_frozen_registry_matches_the_dataset():
    try:
        meta = T.load_dataset_meta()
    except (OSError, SystemExit):
        pytest.skip("shared storage dataset not available")
    assert T.selection_problems(meta) == []
    # every chosen episode is the smallest test index of its slot outside the documented exclusions
    for e, (_, _, slot, _) in T.EXPECTED.items():
        smaller = [x for x, r in meta.items() if r["split"] == "test" and r["layout_index"] == slot and x < e]
        assert all(x in rc53.EXPECTED or x in T.EXCLUDE_RESERVED or x in T.KNOWN_USED for x in smaller), (e, smaller)


def test_records_clone_is_independent_of_t53_records(tmp_path):
    rc = T.records()
    assert rc is not rc53 and rc.STEP == "T54c"
    assert set(rc.KEYS) == {f"ep{e}" for e in T.EXPECTED}
    assert set(rc53.KEYS) == {f"ep{e}" for e in rc53.EXPECTED}           # T53 untouched
    assert rc.t53_dir_of_run(tmp_path, check_storage=False) == tmp_path / "verify" / "T54c"
    assert rc53.t53_dir_of_run(tmp_path, check_storage=False) == tmp_path / "verify" / "T53"
    k = rc.KEYS[0]
    assert rc.episode_of(k).role == T.ROLE and rc.provenance([k])["role"] == T.ROLE
    rc.configure(t53_dir=tmp_path / "x")
    assert rc.npz_path(k) == tmp_path / "x" / "attention" / f"attention_16d_{k}.npz"
    with pytest.raises(SystemExit):
        rc53.resolve_keys([k])
    assert T.registry_module("t53") is rc53 and T.registry_module("t54c") is rc
    with pytest.raises(SystemExit):
        T.registry_module("t99")
    rc.configure()


def test_extract_wrapper_swaps_and_restores_the_registry():
    from AG3S.ag3s.experiments.studies import t53_extract as ex
    from AG3S.ag3s.experiments.studies import t54c_extract as w

    with w.t54c_registry() as m:
        assert m.rc is T.records() and m.STEP == "T54c"
        ap = m.build_parser()
        args = ap.parse_args(["--stage", "layout", "--run-dir", "/x", "--records", T.records().KEYS[0]])
        m.validate_args(ap, args)
    assert ex.rc is rc53 and ex.STEP == "T53"


# --------------------------------------------------------------------- t54_study options


def test_t54_study_registry_options(tmp_path):
    from AG3S.ag3s.experiments.studies import t54_study as S

    ap = S.build_parser()
    a = ap.parse_args(["--run-dir", "r", "--set", "test", "--stage", "inputs"])
    assert a.registry == "t53" and a.seed == 5407 and a.conditions == ["current", "hold"]
    assert a.hold_condition == "hold" and a.entry_arms is None and a.attach_reach == "serving"
    b = ap.parse_args(["--run-dir", "r", "--set", "test", "--stage", "replay", "--registry", "t54c",
                       "--records-run", "/rr", "--conditions", "hold_attach"])
    assert b.t53_run == "/rr" and b.conditions == ["hold_attach"]
    with pytest.raises(SystemExit):
        S.main(["--run-dir", str(tmp_path), "--set", "test", "--stage", "inputs", "--registry", "t54c"])
    assert S.SEED_BY_REGISTRY == {"t53": 5407, "t54c": 5417}
    assert S.records_module("t54c") is T.records()
    argv = ["--run-dir", "r", "--set", "test", "--stage", "analyze", "--registry", "t54c", "--records-run", "/rr"]
    c = ap.parse_args(argv)
    assert S.resolve_seed(c, argv) == 5417 and c.seed == 5417
    d = ap.parse_args(argv + ["--seed", "1"])
    assert S.resolve_seed(d, argv + ["--seed", "1"]) == 5417 and d.seed == 1     # explicit = exploratory
    e = ap.parse_args(argv[:6])
    assert S.resolve_seed(e, argv[:6]) == 5407 and e.seed == 5407
