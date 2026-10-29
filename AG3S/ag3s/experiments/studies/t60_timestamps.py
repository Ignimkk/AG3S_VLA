"""T60 step 1 — P / H completion timestamps and endpoint masks for the existing closed-loop records.

Read-only over CL2 (T44e, 72 runs), CL3 (T46, 54 runs) and T53 (16 runs). Writes, under OUT_DIR:
    timestamps.json   per run: t_P / t_H (primary params), recorded place_t / release_t, the sensitivity grid,
                      and alignment checks against the recorded analysis
    masks/<SET>__<NAME>.npz   keep_rows_P / keep_rows_H (600 control rows), keep_frames_P / keep_frames_H
                      (75 inference frames), protocol version
No arm-level aggregate is computed here (the parameters are confirmed by the user first).

usage: t60_timestamps.py OUT_DIR
"""

from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402

STORE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development")
SETS = {
    "CL2": STORE / "20261012T040647+0900_f14ae3e6b62b_d600316277ef/verify/T44/CL2",
    "CL3": STORE / "20261012T155130+0900_218229ac6464_5fb5ff542c33/verify/T46/CL3",
    "T53": STORE / "20261013T100210+0900_4a540f1a18d7_cb037c425d2f/verify/T53",
}
GRID = {
    "dwell_p": (1, 8, 23),            # 1 row · one chunk · 1.5 s (expert TARGET_SETTLE_SECS)
    "q_tol": (0.25, 0.35, 0.50),
    "tcp_tol": (0.08, 0.12, 0.16),
    "dwell_h": (1, 8, 15),
}


def runs():
    for set_name, base in SETS.items():
        for npz in sorted((base / "replay").glob("*.npz")):
            arm, _, name = npz.stem.split("_", 2)
            yield set_name, base, arm, name, npz


def main(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)
    primary = T.Params()
    res = {"protocol": T.PROTOCOL_VERSION, "primary": primary, "grid": GRID,
           "sources": {k: str(v) for k, v in SETS.items()}, "runs": {}}
    for set_name, base, arm, name, npz in runs():
        an = json.loads((base / "analysis" / f"{name}.json").read_text())
        rj = json.loads(npz.with_suffix(".json").read_text())
        traj = base / "runs" / arm / "R" / name / "traj.npz"
        s = T.series_from_record(npz, traj, an["used_arm"])
        e = T.endpoints(s, primary)
        n = len(s["t"])
        grid = {}
        for dp, qt, tt, dh in itertools.product(*GRID.values()):
            g = T.endpoints(s, T.Params(dwell_p=dp, q_tol=qt, tcp_tol=tt, dwell_h=dh))
            grid[f"{dp}|{qt}|{tt}|{dh}"] = [g["t_P"], g["t_H"]]
        masks = {
            "keep_rows_P": T.keep_rows(n, e["t_P"]), "keep_rows_H": T.keep_rows(n, e["t_H"]),
            "keep_frames_P": T.keep_frames(s["chunk_start_steps"], e["t_P"]),
            "keep_frames_H": T.keep_frames(s["chunk_start_steps"], e["t_H"]),
        }
        np.savez_compressed(out_dir / "masks" / f"{set_name}__{name}.npz", protocol=T.PROTOCOL_VERSION,
                            t_P=-1 if e["t_P"] is None else e["t_P"], t_H=-1 if e["t_H"] is None else e["t_H"],
                            chunk_start_steps=s["chunk_start_steps"], **masks)
        t_p = e["t_P"]
        res["runs"][f"{set_name}/{name}"] = {
            "set": set_name, "arm": arm, "name": name, "episode": an["episode"], "seed": an["seed"],
            "used_arm": an["used_arm"], "n_rows": n,
            "recorded": {"success": an["success"], "grasp": an["grasp"], "place": an["place"],
                         "grasp_t": an["grasp_t"], "place_t": an["place_t"],
                         "release_t_robust": an.get("release_t_robust"),
                         "qpos_mismatch_rows": rj.get("qpos_mismatch_rows")},
            "t_P": t_p, "t_H": e["t_H"],
            "rows_removed_P": int((~masks["keep_rows_P"]).sum()), "rows_removed_H": int((~masks["keep_rows_H"]).sum()),
            "frames_removed_P": int((~masks["keep_frames_P"]).sum()),
            "frames_removed_H": int((~masks["keep_frames_H"]).sum()),
            "hand_target_after_P": None if t_p is None else int(s["hand_target"][t_p + 1:].sum()),
            "out_of_crate_after_P": None if t_p is None else int((~s["in_crate"][t_p + 1:n]).sum()),
            "q_err_min_after_P": None if t_p is None else float(s["q_err"][t_p + 1:].min()) if t_p + 1 < n else None,
            "tcp_err_min_after_P": None if t_p is None else float(s["tcp_err"][t_p + 1:].min()) if t_p + 1 < n else None,
            "q_err_final": float(s["q_err"][-1]), "tcp_err_final": float(s["tcp_err"][-1]),
            "grid": grid,
        }
    T.write_json(out_dir / "timestamps.json", res)
    print(len(res["runs"]), "runs ->", out_dir)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
