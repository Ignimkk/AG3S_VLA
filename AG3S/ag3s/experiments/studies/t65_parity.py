"""T65 parity gate — the noun-source replay must reproduce the recorded core-evaluation server chunks.

`handoff/T65.task.md` "parity gate": per matched request (same episode, same t_step)
    selection   grounding_status and manipulated id / state equal  -> agreement rate (gate >= 0.99)
    target      Chamfer (median of nearest-neighbour distances, both ways) between target point sets
                (gate: median over requests <= 5 mm)
    attention   head-camera attention map: max |diff| and argmax-equal (reported)

usage: python t65_parity.py REPLAY_DIR GROUP OUT.json   (REPLAY_DIR = verify/T65/<source>/<group>)
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

CORE_V = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261028T224856+0900_b3b69311f948_8784a1265958"
                      "/verify/AG3S-CORE-EVAL")


def split_by_restart(files):
    files = sorted(files)
    if not files:
        return []
    ts = [int(np.load(f)["t_step"]) for f in files]
    segs, cur = [], [files[0]]
    for f, a, b in zip(files[1:], ts, ts[1:]):
        if b <= a:
            segs.append(cur)
            cur = []
        cur.append(f)
    segs.append(cur)
    return segs


def replay_segments(d: pathlib.Path) -> dict:
    order = (d / "order.txt").read_text().split()
    segs = split_by_restart(list((d / "server_constraints" / "run_0000").glob("chunk_*.npz")))
    if len(segs) != len(order):
        raise SystemExit(f"{d}: {len(segs)} segments for {len(order)} episodes")
    return dict(zip(order, segs))


def recorded_segments(group: str) -> dict:
    """{client name: [chunk files]} of the recorded servers of `group` (all servers, split per episode)."""
    out = {}
    if group.startswith("OCC"):
        for sd in sorted((CORE_V / group).glob("server_*")):
            order = (sd / "order.txt").read_text().split()
            segs = split_by_restart(list((sd / "server_constraints" / "run_0000").glob("chunk_*.npz")))
            out.update(dict(zip(order, segs)))
        return out
    import os

    sys.path.insert(0, str(CORE_V.parent.parent / "snapshot_core"))
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde

    for ep, (pdir, ch) in tde.episodes_of(CORE_V / group, "B_ROI_FPR").items():
        out[pathlib.Path(pdir).parent.parent.name] = [pathlib.Path(f) for _, f, _ in ch]
    return out


def summary(f):
    z = np.load(f, allow_pickle=True)
    s = json.loads(str(z["summary_json"]))
    m = s.get("manipulated") or {}
    tp = np.asarray(z["target_points"], float) if "target_points" in z.files else np.zeros((0, 3))
    att = np.asarray(z["attention_head"], float) if "attention_head" in z.files else None
    return int(z["t_step"]), {"gs": s.get("grounding_status"), "id": m.get("id"), "state": m.get("state"),
                              "tp": tp, "att": att}


def chamfer(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    return float(np.median(np.r_[cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]]))


def main(rep_dir: pathlib.Path, group: str, out: pathlib.Path) -> None:
    rep, rec = replay_segments(rep_dir), recorded_segments(group)
    rows = []
    for name, rfiles in rep.items():
        recd = {t: v for t, v in (summary(f) for f in rec[name])}
        for f in rfiles:
            t, a = summary(f)
            b = recd.get(t)
            if b is None:
                continue
            row = {"client": name, "t_step": t, "sel_equal": (a["gs"], a["id"], a["state"]) == (b["gs"], b["id"], b["state"]),
                   "gs": [a["gs"], b["gs"]], "id": [a["id"], b["id"]], "chamfer_m": chamfer(a["tp"], b["tp"])}
            if a["att"] is not None and b["att"] is not None and a["att"].shape == b["att"].shape:
                row["att_maxabs"] = float(np.max(np.abs(a["att"] - b["att"])))
                row["att_argmax_equal"] = bool(np.argmax(a["att"]) == np.argmax(b["att"]))
            rows.append(row)
    ch = np.array([r["chamfer_m"] for r in rows], float)
    res = {"n": len(rows), "selection_agreement": float(np.mean([r["sel_equal"] for r in rows])) if rows else None,
           "chamfer_median_m": float(np.median(ch[np.isfinite(ch)])) if np.isfinite(ch).any() else None,
           "chamfer_inf": int((~np.isfinite(ch)).sum()),
           "att_argmax_equal": float(np.mean([r["att_argmax_equal"] for r in rows if "att_argmax_equal" in r]))
           if any("att_argmax_equal" in r for r in rows) else None,
           "att_maxabs_max": max((r["att_maxabs"] for r in rows if "att_maxabs" in r), default=None),
           "gate": {"selection_agreement_min": 0.99, "chamfer_median_max_m": 0.005}, "rows": rows}
    res["gate_pass"] = bool(res["selection_agreement"] is not None and res["selection_agreement"] >= 0.99
                            and res["chamfer_median_m"] is not None and res["chamfer_median_m"] <= 0.005)
    out.write_text(json.dumps(res, indent=1, default=float))
    print({k: v for k, v in res.items() if k != "rows"})


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]))
