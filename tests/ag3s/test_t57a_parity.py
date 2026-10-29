"""T57-A parity / cost script (`t57a_parity`) — the CPU pieces only (the GPU stages are run by the verifier).

Pinned: the lean stored-map read equals `t54_arms.noun_map` (σ0, k1) on a synthetic T49-format extract; the row
comparison ignores timing keys and NaN == NaN; the CPU timing of `SafePolicy._task_gate` runs; the CLI refuses a
missing T54c run / bad frames.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t57a_parity as m


def _extract(F=4, seed=0):
    rng = np.random.default_rng(seed)
    maps = (rng.random((F, 3, 18, 8, 3, 16, 16)) / 300).astype(np.float16)
    return {"maps": maps, "query_names": np.asarray(("target", "basket", "last_text"))}


def test_stored_noun_maps_is_t54_noun_map():
    from AG3S.ag3s.experiments.studies import t54_arms as arms
    from AG3S.ag3s.experiments.studies.t49_noun_query import block_fc

    ex = _extract()
    got = m.stored_noun_maps(ex)
    want = arms.noun_map(block_fc(ex, "target"), arms.SOURCES["noun"])
    assert got.dtype == np.float32 and got.shape == (4, 3, 16, 16)
    assert np.array_equal(got, want)
    assert np.array_equal(m.stored_noun_map(ex, 2).astype(np.float32), got[2])


def test_row_diff_ignores_timing_and_nan():
    a = {"frame": 1, "ag3s_ms": 4.0, "ms_profile": {"x": 1}, "centroid_error_m": float("nan"), "clusters": [{"n": 3}]}
    b = {"frame": 1, "ag3s_ms": 9.0, "ms_profile": {"x": 2}, "centroid_error_m": float("nan"), "clusters": [{"n": 3}]}
    assert m._row_diff(a, b) == []                                                   # noqa: SLF001
    b["clusters"] = [{"n": 4}]
    assert m._row_diff(a, b) == ["clusters"]                                         # noqa: SLF001


def test_gate_cpu_cost_runs():
    out = m.gate_cpu_cost(n=50)
    assert out["n"] == 50 and out["median_ms"] >= 0.0


def test_cli_refusals(tmp_path):
    ap = m.build_parser()
    for argv in (["--stage", "noun"], ["--stage", "replay", "--t54c-run", str(tmp_path)],
                 ["--stage", "cost", "--t54c-run", str(tmp_path)]):
        with pytest.raises(SystemExit):
            m.validate_args(ap, ap.parse_args(argv))
    (tmp_path / "verify" / "T54c").mkdir(parents=True)
    with pytest.raises(SystemExit):
        m.validate_args(ap, ap.parse_args(["--stage", "noun", "--t54c-run", str(tmp_path), "--frames", "-1"]))
    m.validate_args(ap, ap.parse_args(["--stage", "noun", "--t54c-run", str(tmp_path)]))
    with pytest.raises(SystemExit):
        m.validate_args(ap, ap.parse_args(["--stage", "probe", "--t55-run-dir", str(tmp_path)]))
