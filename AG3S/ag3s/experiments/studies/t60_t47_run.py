"""T60 step 3d — T47 stages (select · validate (+ GT-oracle) · e2e) re-run on the P / H frame windows.

`t47_head_selection` runs unchanged except three helpers (parent process):
    out_root        -> ``--root`` (fresh T60 directory holding a copy of T47's ``gt_cache``)
    load_attention  -> frames after completion dropped (T60 record mask of that key)
    load_gt         -> the same frames dropped, so attention and GT stay aligned
Held-out records never reach P (15 frames), so validate / e2e on them read every frame; only the selection
(label-free on ``long`` · ``t2`` · ``ep1807``) and the GT-oracle (selection records) see the window.
The resulting ``select/selection.json`` / ``validate/oracle.json`` are what a windowed T51 run reads
(``--t47-dir``).

usage: t60_t47_run.py --window {P,H} --mask-dir DIR --root DIR -- <t47_head_selection args>
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

_STATE = {"window": None, "mask_dir": None}


@lru_cache(maxsize=None)
def frame_mask(key: str) -> np.ndarray:
    with np.load(pathlib.Path(_STATE["mask_dir"]) / f"{key}.npz") as z:
        return z[f"keep_frames_{_STATE['window']}"].astype(bool)


def install(window: str, mask_dir: str, root: pathlib.Path) -> None:
    _STATE.update(window=window, mask_dir=mask_dir)
    orig_att, orig_gt = hs.load_attention, hs.load_gt

    def load_attention(key):
        att, dn, ag, extra = orig_att(key)
        m = frame_mask(key)
        assert len(m) == att.shape[0], (key, len(m), att.shape)
        return att[m], dn, ag, extra

    def load_gt(root_, key):
        gt = orig_gt(root_, key)
        m = frame_mask(key)
        n = len(np.asarray(gt["t_step"]))
        assert len(m) == n, (key, len(m), n)
        return {k: (np.asarray(v)[m] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v)
                for k, v in gt.items()}

    hs.load_attention, hs.load_gt = load_attention, load_gt
    hs.out_root = lambda: (root.mkdir(parents=True, exist_ok=True), root)[1]


def main() -> None:
    argv = sys.argv[1:]
    sep = argv.index("--") if "--" in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", choices=("P", "H"), required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args(argv[:sep])
    root = pathlib.Path(a.root)
    install(a.window, a.mask_dir, root)
    (root / "t60_window.json").write_text(json.dumps({"window": a.window, "mask_dir": a.mask_dir,
                                                      "argv": argv}) + "\n")
    hs.main(argv[sep + 1:])


if __name__ == "__main__":
    main()
