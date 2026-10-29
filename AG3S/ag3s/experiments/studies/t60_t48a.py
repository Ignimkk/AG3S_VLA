"""T60 step 3c — T48a closed-loop ep1807 auxiliary set re-summarised on the P / H frame windows.

Reads `verify/T48/score/predictions_ep1807.csv` (one row per variant × host frame; `host_row` = frame
index of the ep1807 record) and keeps the host frames the T60 record mask keeps. Same summaries as T48:
accuracy (probe == gt) per variant, C1 → pick share by the host frame's original class (`y_host`),
C0 confusion (gt × probe). Classes 0/1/2 =
pick / place / home.

usage: t60_t48a.py T48_RUN_DIR MASK_NPZ OUT_JSON
"""

from __future__ import annotations

import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402

CLASSES = ("pick", "place", "home")


def summarize(rows: list[dict]) -> dict:
    out = {"accuracy": {}, "n_host_frames": len({int(r["host_row"]) for r in rows})}
    for v in sorted({r["variant"] for r in rows}):
        rs = [r for r in rows if r["variant"] == v]
        k = sum(int(r["probe"]) == int(r["gt"]) for r in rs)
        out["accuracy"][v] = {"k": k, "n": len(rs), "rate": k / len(rs) if rs else None}
    c1 = [r for r in rows if r["variant"] == "C1"]
    out["C1_to_pick_all"] = {"k": sum(int(r["probe"]) == 0 for r in c1), "n": len(c1)}
    out["C1_to_pick"] = {c: {"k": sum(int(r["probe"]) == 0 for r in c1 if int(r["y_host"]) == i),
                             "n": sum(int(r["y_host"]) == i for r in c1)} for i, c in enumerate(CLASSES)}
    c0 = [r for r in rows if r["variant"] == "C0"]
    conf = np.zeros((3, 3), int)
    for r in c0:
        conf[int(r["gt"]), int(r["probe"])] += 1
    out["C0_confusion"] = conf.tolist()
    out["gt_class_frames"] = {c: int((conf.sum(axis=1))[i]) for i, c in enumerate(CLASSES)}
    return out


def main(t48: pathlib.Path, mask_npz: pathlib.Path, out_json: pathlib.Path) -> None:
    with (t48 / "verify/T48/score/predictions_ep1807.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    with np.load(mask_npz) as z:
        masks = {"full": np.ones(len(z["keep_frames_P"]), bool), "P": z["keep_frames_P"].astype(bool),
                 "H": z["keep_frames_H"].astype(bool)}
        ends = {"t_P": int(z["t_P"]), "t_H": int(z["t_H"])}
    res = {"protocol": T.PROTOCOL_VERSION, "source": str(t48), "endpoints": ends, "windows": {}}
    for w, m in masks.items():
        res["windows"][w] = summarize([r for r in rows if m[int(r["host_row"])]])
    T.write_json(out_json, res)
    for w, s in res["windows"].items():
        print(w, s["n_host_frames"], s["gt_class_frames"], {v: round(a["rate"], 3) for v, a in s["accuracy"].items()},
              "C1->pick", s["C1_to_pick_all"], s["C1_to_pick"])


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
