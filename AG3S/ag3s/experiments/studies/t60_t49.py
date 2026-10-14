"""T60 step 3e — T49 noun-query cell selection re-run on the P / H frame windows.

`t49_noun_query --stage select` unchanged except (parent process):
    out_root        -> ``--root`` (its ``extract/orig/<key>.npz`` are symlinks to the T49 run, read only)
    load_orig       -> frames after completion dropped from every per-frame array of the extract
    hs.load_gt      -> the same frames dropped (alignment is still checked by `check_alignment`)
``--window full`` keeps every frame and must reproduce the published ``select/select.json``.

usage: t60_t49.py --window {full,P,H} --mask-dir DIR --root DIR -- <t49_noun_query args>
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
from AG3S.ag3s.experiments.studies import t47_head_selection as hs  # noqa: E402
from AG3S.ag3s.experiments.studies import t49_noun_query as nq  # noqa: E402

_STATE = {"window": "full", "mask_dir": None}


@lru_cache(maxsize=None)
def frame_mask(key: str):
    if _STATE["window"] == "full":
        return None
    with np.load(pathlib.Path(_STATE["mask_dir"]) / f"{key}.npz") as z:
        return z[f"keep_frames_{_STATE['window']}"].astype(bool)


def _slice(d: dict, m, n: int) -> dict:
    return {k: (np.asarray(v)[m] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v)
            for k, v in d.items()}


def install(window: str, mask_dir: str, root: pathlib.Path) -> None:
    _STATE.update(window=window, mask_dir=mask_dir)
    orig_lo, orig_gt = nq.load_orig, hs.load_gt

    def load_orig(root_, key, variant="orig"):
        ex = orig_lo(root_, key, variant)
        m = frame_mask(key)
        if m is None:
            return ex
        n = len(np.asarray(ex["t_step"]))
        assert len(m) == n, (key, len(m), n)
        return _slice(ex, m, n)

    def load_gt(root_, key):
        gt = orig_gt(root_, key)
        m = frame_mask(key)
        if m is None:
            return gt
        n = len(np.asarray(gt["t_step"]))
        assert len(m) == n, (key, len(m), n)
        return _slice(gt, m, n)

    nq.load_orig, hs.load_gt = load_orig, load_gt
    nq.out_root = lambda: (root.mkdir(parents=True, exist_ok=True), root)[1]


def main() -> None:
    argv = sys.argv[1:]
    sep = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=("full", "P", "H"), required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args(argv[:sep])
    root = pathlib.Path(a.root)
    install(a.window, a.mask_dir, root)
    (root / "t60_window.json").write_text(json.dumps({"window": a.window, "mask_dir": a.mask_dir,
                                                      "argv": argv}) + "\n")
    nq.main(argv[sep + 1:])


if __name__ == "__main__":
    main()
