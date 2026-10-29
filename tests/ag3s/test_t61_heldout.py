"""T61 held-out — pure helpers (no storage, no MuJoCo)."""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t61_heldout as H


def test_frozen_selection_sha_and_sources():
    sel = H.load_selection()  # refuses when the committed file was edited
    srcs = H.source_cells(sel)
    ids = list(srcs)
    assert ids[0] == H.selected_id(sel)
    assert H.BASELINE in srcs and all(r in srcs for r in H.REFERENCES)
    assert len(ids) == len(set(ids))


def test_selected_id_sums_cells():
    assert H.selected_id({"selected": {"cells": ["mean:L10H7"]}}) == "mean:L10H7"
    assert H.selected_id({"selected": {"cells": ["a:L1H1", "b:L2H2"]}}) == "sum[a:L1H1+b:L2H2]"


def test_source_cells_drops_duplicate_baseline():
    srcs = H.source_cells({"selected": {"cells": [H.BASELINE]}})
    assert list(srcs)[0] == H.BASELINE and list(srcs).count(H.BASELINE) == 1


def test_window_rows_keeps_masked_frames():
    rows = [{"frame": i} for i in range(4)]
    assert [r["frame"] for r in H.window_rows(rows, np.array([1, 0, 1, 0], bool))] == [0, 2]
    assert len(H.window_rows(rows, None)) == 4


def test_bootstrap_intervals_cover_the_mean():
    v = np.array([0.1, -0.05, 0.2, 0.0, 0.05])
    lo, hi = H.boot_ci(v, n_boot=2000, seed=1)
    assert lo <= v.mean() <= hi
    lo, hi = H.moving_block_ci(np.r_[np.ones(20), -np.ones(20)], block=5, n_boot=2000, seed=1)
    assert lo <= 0.0 <= hi
    assert H.boot_ci(np.zeros(5)) == pytest.approx([0.0, 0.0])
