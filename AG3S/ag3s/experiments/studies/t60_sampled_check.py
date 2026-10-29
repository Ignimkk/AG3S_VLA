"""T60 step 3 prerequisite — is the frame-resolution (inference-time sample) P / H a faithful stand-in?

On the 142 closed-loop runs (row-level truth from `t60_timestamps`) the series are subsampled at the
chunk starts (contact = contact sub-step in that row, speed = that row's mean speed) and fed to
`endpoints_sampled`. Reported: per run the kept-frame mask agreement with the row-level mask.

usage: t60_sampled_check.py STEP1_DIR OUT_JSON
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402
from AG3S.ag3s.experiments.studies.t60_timestamps import SETS  # noqa: E402


def main(step1: pathlib.Path, out_json: pathlib.Path) -> None:
    ts = json.loads((step1 / "timestamps.json").read_text())
    p = T.Params()
    per, tot = {}, {"P": [0, 0, 0], "H": [0, 0, 0]}  # frames agree, frames total, runs exact
    cls = {"P": 0, "H": 0}
    for key, r in ts["runs"].items():
        base = SETS[r["set"]]
        s = T.series_from_record(base / "replay" / f"{r['arm']}_R_{r['name']}.npz",
                                 base / "runs" / r["arm"] / "R" / r["name"] / "traj.npz", r["used_arm"])
        cs = np.asarray(s["chunk_start_steps"])
        e = T.endpoints_sampled(cs, s["in_crate"][cs], s["hand_target"][cs], s["speed"][cs],
                                s["q_err"][cs], s["tcp_err"][cs], p)
        rec = {"row": {"t_P": r["t_P"], "t_H": r["t_H"]}, "sampled": e}
        for w in ("P", "H"):
            a = T.keep_frames(cs, r[f"t_{w}"]); b = T.keep_frames(cs, e[f"t_{w}"])
            tot[w][0] += int((a == b).sum()); tot[w][1] += len(a); tot[w][2] += int((a == b).all())
            cls[w] += int((r[f"t_{w}"] is None) != (e[f"t_{w}"] is None))
            rec[f"frames_diff_{w}"] = int(b.sum() - a.sum())
        per[key] = rec
    diffs = {w: np.array([v[f"frames_diff_{w}"] for v in per.values()]) for w in ("P", "H")}
    out = {"n_runs": len(per),
           "frame_agreement": {w: tot[w][0] / tot[w][1] for w in ("P", "H")},
           "runs_exact": {w: tot[w][2] for w in ("P", "H")},
           "completion_class_disagreements": cls,
           "frames_kept_diff_sampled_minus_row": {w: {"min": int(d.min()), "max": int(d.max()),
                                                      "counts": {str(k): int((d == k).sum()) for k in np.unique(d)}}
                                                  for w, d in diffs.items()},
           "per_run": per}
    T.write_json(out_json, out)
    print(json.dumps({k: v for k, v in out.items() if k != "per_run"}, indent=1))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
