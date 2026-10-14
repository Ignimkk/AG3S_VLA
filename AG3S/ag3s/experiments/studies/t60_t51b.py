"""T60c — the T51b protocol (noun / combined Ω) re-run on the P / H frame windows.

`t51b_study` runs unchanged; after its own `install` rebinds `t51_study`, this wrapper adds the T60 window
on top (parent process only, exactly as `t60_t51`):
    stage_filter  -> attention sliced to the kept frames (label-free P1 statistics)
    label_data    -> the 2D ``keep`` ANDed with the kept frames
    load_rows     -> T51b's reader (its own cache, then the T51 rows of ``--t51-dir``) filtered to kept frames
    _complete     -> counts the unfiltered rows (a cached full-record file is never replayed again)
The T51b output base (``verify/T51b``) is redirected to ``--base``. ``--t51-dir`` must be the T60 T51 root of
the same window (its selection, rows and self label) and ``--t47-dir`` the T60 T47 root of the window.
`check_replay_opts` also reads the ``__scene`` payloads of ``--t51-dir`` (the T60 roots have only those).
Before the run the copied T51b row cache must hold only cell-named sources (no ``paper3_s1``; checked).

usage: t60_t51b.py --window {full,P,H} --mask-dir DIR --base DIR -- <t51b_study args>
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

import common.run_output as ro  # noqa: E402
from AG3S.ag3s.experiments.studies import t51_study as st  # noqa: E402
from AG3S.ag3s.experiments.studies import t51b_study as mb  # noqa: E402

_STATE = {"window": "full", "mask_dir": None, "slice": False, "done": False}


@lru_cache(maxsize=None)
def frame_mask(key: str):
    if _STATE["window"] == "full":
        return None
    with np.load(pathlib.Path(_STATE["mask_dir"]) / f"{key}.npz") as z:
        return z[f"keep_frames_{_STATE['window']}"].astype(bool)


def _apply_window() -> None:
    qb, ld, lr, flt = st.query_blocks, st.label_data, st.load_rows, st.stage_filter

    def query_blocks(key, qs, t49=None):
        out = qb(key, qs, t49)
        m = frame_mask(key)
        if _STATE["slice"] and m is not None:
            n = next(iter(out.values())).shape[0]
            assert len(m) == n, (key, len(m), n)
            out = {q: v[m] for q, v in out.items()}
        return out

    def label_data(label, key, root_, t47, min_px):
        cov, keep = ld(label, key, root_, t47, min_px)
        m = frame_mask(key)
        if m is not None:
            assert len(m) == keep.shape[0], (key, len(m), keep.shape)
            keep = keep & m[:, None]
        return cov, keep

    def load_rows(root_, key, sid):
        rows = lr(root_, key, sid)
        m = frame_mask(key)
        if rows is None or m is None:
            return rows
        return [r for r in rows if m[int(r["frame"])]]

    def _complete(root_, key, sid, n_frames):
        rows = lr(root_, key, sid)
        return rows is not None and len(rows) == n_frames

    def stage_filter(args, root_):
        _STATE["slice"] = True
        try:
            return flt(args, root_)
        finally:
            _STATE["slice"] = False

    st.query_blocks, st.label_data, st.load_rows, st._complete = query_blocks, label_data, load_rows, _complete
    st.stage_filter = stage_filter


def main() -> None:
    argv = sys.argv[1:]
    sep = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=("full", "P", "H"), required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--base", required=True)
    a = ap.parse_args(argv[:sep])
    base = pathlib.Path(a.base)
    base.mkdir(parents=True, exist_ok=True)
    stale = sorted(str(p) for p in (base / "rows").rglob("paper3_s1*.jsonl")) if (base / "rows").is_dir() else []
    if stale:
        raise SystemExit(f"name-keyed special-source rows in the T51b cache (delete first): {stale[:3]}")
    _STATE.update(window=a.window, mask_dir=a.mask_dir)
    orig_res = ro.resolve_output_path
    ro.resolve_output_path = (lambda path, *, run_dir, default_name:
                              base if default_name == "verify/T51b" else orig_res(path, run_dir=run_dir,
                                                                                    default_name=default_name))
    orig_install = mb.install

    def install(cfg):
        orig_install(cfg)
        if not _STATE["done"]:
            _apply_window()
            _STATE["done"] = True

    mb.install = install
    # the T60 window T51 roots hold the scene-mean payloads; they record the same replay options
    mb.T51_OPT_PAYLOADS = tuple(mb.T51_OPT_PAYLOADS) + tuple(r.replace(".json", "__scene.json")
                                                             for r in mb.T51_OPT_PAYLOADS)
    (base / "t60_window.json").write_text(json.dumps({"window": a.window, "mask_dir": a.mask_dir,
                                                      "argv": argv}) + "\n")
    mb.main(argv[sep + 1:])


if __name__ == "__main__":
    main()
