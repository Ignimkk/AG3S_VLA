"""T60 step 3b — the T51 / T51b head-selection protocol re-run on the P / H frame windows.

`t51_study` / `t51b_study` run unchanged except for four helpers, patched in the parent process only:
    out_root     -> ``--root`` (a fresh T60 directory; the T51 run is read only via ``--extra-dir``)
    stage_filter -> the label-free P1 statistics pool only the kept frames (attention sliced)
    label_data   -> the 2D (frame, camera) ``keep`` is ANDed with the kept frames (GT and self label)
    load_rows    -> the 3D rows are filtered to the kept frames (``row["frame"]``); `_complete` keeps
                    counting the full record so a cached full-record row file is never replayed again
Spawned replay workers are not patched: a missing (record, source) is replayed on every frame of the
record exactly like T51, and the window is applied when the rows are read. ``--window full`` uses an
all-true mask and must reproduce the published T51 outputs.

Row cache caution: rows are keyed by source id. ``sum[...]`` / cell ids name their content, but the special
source ``paper3_s1`` (T47 paper top-3, sigma 1) is a *name* whose cells come from ``--t47-dir``. When the
root's rows were copied from a run with another T47 directory, delete ``rows/*/paper3_s1.jsonl`` first
(T60 did this for P / H on 2026-10-14 00:12 KST after an ablation read the stale full-window rows).

usage: t60_t51.py --window {full,P,H} --mask-dir DIR --root DIR [--module t51_study|t51b_study] -- <t51 args>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from functools import lru_cache

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from AG3S.ag3s.experiments.studies import t51_study as st  # noqa: E402

_STATE = {"window": "full", "mask_dir": None, "slice": False}


@lru_cache(maxsize=None)
def frame_mask(key: str) -> np.ndarray | None:
    if _STATE["window"] == "full":
        return None
    with np.load(pathlib.Path(_STATE["mask_dir"]) / f"{key}.npz") as z:
        return z[f"keep_frames_{_STATE['window']}"].astype(bool)


def install(window: str, mask_dir: str, root: pathlib.Path) -> None:
    _STATE.update(window=window, mask_dir=mask_dir)
    orig_qb, orig_ld, orig_lr, orig_filter = st.query_blocks, st.label_data, st.load_rows, st.stage_filter

    def query_blocks(key, qs, t49=None):
        out = orig_qb(key, qs, t49)
        m = frame_mask(key)
        if _STATE["slice"] and m is not None:
            n = next(iter(out.values())).shape[0]
            assert len(m) == n, (key, len(m), n)
            out = {q: v[m] for q, v in out.items()}
        return out

    def label_data(label, key, root_, t47, min_px):
        cov, keep = orig_ld(label, key, root_, t47, min_px)
        m = frame_mask(key)
        if m is not None:
            assert len(m) == keep.shape[0], (key, len(m), keep.shape)
            keep = keep & m[:, None]
        return cov, keep

    def load_rows(root_, key, sid):
        rows = orig_lr(root_, key, sid)
        m = frame_mask(key)
        if rows is None or m is None:
            return rows
        return [r for r in rows if m[int(r["frame"])]]

    def _complete(root_, key, sid, n_frames):
        rows = orig_lr(root_, key, sid)
        return rows is not None and len(rows) == n_frames

    def stage_filter(args, root_):
        _STATE["slice"] = True
        try:
            return orig_filter(args, root_)
        finally:
            _STATE["slice"] = False

    st.query_blocks, st.label_data, st.load_rows, st._complete = query_blocks, label_data, load_rows, _complete
    st.stage_filter = stage_filter
    st.out_root = lambda: (root.mkdir(parents=True, exist_ok=True), root)[1]


def main() -> None:
    argv = sys.argv[1:]
    sep = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=("full", "P", "H"), required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--module", default="t51_study", choices=("t51_study", "t51b_study"))
    a = ap.parse_args(argv[:sep])
    root = pathlib.Path(a.root)
    install(a.window, a.mask_dir, root)
    (root / "t60_window.json").write_text(json.dumps({"window": a.window, "mask_dir": a.mask_dir,
                                                      "module": a.module, "argv": argv}) + "\n")
    if a.module == "t51b_study":
        from AG3S.ag3s.experiments.studies import t51b_study as mod
        if hasattr(mod, "out_root"):
            mod.out_root = st.out_root
        for name in ("load_rows", "label_data", "query_blocks"):
            if hasattr(mod, name):
                setattr(mod, name, getattr(st, name))
        mod.main(argv[sep + 1:])
    else:
        st.main(argv[sep + 1:])


if __name__ == "__main__":
    main()
