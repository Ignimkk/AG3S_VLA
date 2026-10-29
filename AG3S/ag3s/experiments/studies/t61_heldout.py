"""T61 held-out — the frozen selection vs L8H2 on the 16 new episodes (`handoff/T61.task.md` "사전 고정 — held-out").

Run only after `T61.selection.json` is committed (it is: `b49f5f6c`); this script refuses a selection whose sha256
differs from the frozen one. Nothing about the selection is recomputed here.

Stages (outputs under ``--out``, a session-B run dir; the T61 run dir is only read, except that ``p/`` scripts of
that run wrote its own ``replay/`` · ``analysis/`` · ``agg.json`` before this script runs):

``endpoints``  T60 P · H endpoints of every held-out run from the T61 replay npz (`replay_contacts.py`, run by
               ``p/run_replays_cl.sh``) — `t60_timestamps.main` with its set table pointed at T61 — and the
               closed-loop window table of the **serving record** (C1a = L8H2 serving) via `t60_closed_loop.main`
               (needs ``agg.json`` · ``analysis/`` from ``p/analyze_cl.py`` · ``p/agg_cl.py``). These are the results
               of the configuration that collected the record; they are not per-source closed-loop results.
``replay``     3D grounding rows per (episode, source) with the **T51 replay options** used for the calibration
               (`t51_replay.run_sources` defaults: voxel 0.020, legacy ESDF, gripper links, on_target 0.10 m, label
               0.003 m, membership 0.005 m; AG3S at the server defaults, subtask gate off, one AG3S per source).
               Sources = the frozen selection, L8H2 (`last:L8H2`, baseline — forced only here), and two references
               reported only: T51 full selection `last:L12H4`, T54 N `noun_target:L10H7`. Maps come from the T61
               extraction (`t56_cells.cell_maps`: T47-format action npz, T49-format noun extract; raw σ0). A k > 1
               selection is the sum of its cell maps (T51 ``sum[...]``).
``evaluate``   per episode and window (full · P · H): `t51_study.record_metrics` (GT ``identity_correct`` rate =
               the calibration primary, purity on ``has_target`` frames) and coverage (`t61_select._coverage`).
               Primary = **P window, episode-mean GT identity**, paired difference selection − L8H2 with an episode
               bootstrap (B 10000, seed 6102, percentile 95 %) and a moving-block (5 frames) sensitivity on the
               per-frame paired differences. Reported next to it: full · H, purity, coverage, phase breakdown (T61
               ``grasp_state``), grounding-status counts, the references. No source is re-chosen from these numbers.

usage::

    $PY -m AG3S.ag3s.experiments.studies.t61_heldout --stage endpoints --out $RUN
    $PY -m AG3S.ag3s.experiments.studies.t61_heldout --stage replay --out $RUN --workers 4
    $PY -m AG3S.ag3s.experiments.studies.t61_heldout --stage evaluate --out $RUN
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Mapping, Optional, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

RUN61 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261014T183529+0900_53f3911df197_dd2fad8756a6")
SELECTION = ROOT / "AG3S/ag3s/docs/handoff/T61.selection.json"
SELECTION_SHA256 = "c3f3b60f11f08aa21dcc525c6ba079b3d0fa4699b5017451607ba12d8cda9a14"
BASELINE = "last:L8H2"
REFERENCES = ("last:L12H4", "noun_target:L10H7")
WINDOWS = ("full", "P", "H")
N_BOOT, SEED, BLOCK = 10000, 6102, 5
SUB = "verify/T61_heldout"


def load_selection() -> dict:
    raw = SELECTION.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SELECTION_SHA256:
        raise SystemExit("T61.selection.json differs from the frozen sha256 — refusing")
    return json.loads(raw)


def selected_id(sel: Mapping) -> str:
    cells = list(sel["selected"]["cells"])
    return cells[0] if len(cells) == 1 else "sum[" + "+".join(cells) + "]"


def source_cells(sel: Mapping) -> dict[str, list[str]]:
    """``{source id: [cells]}`` in report order (selection, baseline, references), duplicates dropped."""
    out = {selected_id(sel): list(sel["selected"]["cells"])}
    for c in (BASELINE, *REFERENCES):
        out.setdefault(c, [c])
    return out


def records():
    from AG3S.ag3s.experiments.studies import t61_records as t61

    rc = t61.records()
    lay = json.loads((RUN61 / "verify/T61/layout.json").read_text())
    rc.configure(run_dir=RUN61, episodes_root=lay["args"].get("episodes_root"))
    return rc


# ------------------------------------------------------------------------------ endpoints


def stage_endpoints(out: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t60_closed_loop as CL
    from AG3S.ag3s.experiments.studies import t60_timestamps as TS

    base = RUN61 / "verify/T61"
    TS.SETS.clear()
    TS.SETS["T61"] = base
    CL.SETS, CL.PAIRS = TS.SETS, {"T61": []}
    step1 = out / SUB / "step1"
    TS.main(step1)
    if (base / "agg.json").is_file():
        CL.main(step1, out / SUB / "closed_loop_serving_record.json")
    else:
        print("no agg.json in the T61 run — run p/analyze_cl.py · p/agg_cl.py first (closed-loop table skipped)")
    return {"step1": str(step1)}


def load_masks(out: pathlib.Path) -> dict:
    masks = {}
    for p in sorted((out / SUB / "step1/masks").glob("T61__C1a_*.npz")):
        key = p.stem.split("__", 1)[1].split("_")[1]
        with np.load(p) as z:
            masks[key] = {"P": z["keep_frames_P"].astype(bool), "H": z["keep_frames_H"].astype(bool),
                          "t_P": int(z["t_P"]), "t_H": int(z["t_H"]),
                          "chunk_start_steps": [int(t) for t in z["chunk_start_steps"]]}
    return masks


# ------------------------------------------------------------------------------ replay


def rows_path(out: pathlib.Path, key: str, sid: str) -> pathlib.Path:
    return out / SUB / "rows" / f"{key}__{sid.replace(':', '_').replace('+', '-')}.jsonl"


def maps_for(rc, key: str, srcs: Mapping[str, Sequence[str]]) -> dict[str, np.ndarray]:
    from AG3S.ag3s.experiments.studies import t54_study as st54
    from AG3S.ag3s.experiments.studies import t56_cells as cells_mod
    from AG3S.ag3s.experiments.studies.t49_noun_query_extract import load_extract

    all_cells = sorted({c for cs in srcs.values() for c in cs})
    action = st54.load_action(rc.npz_path(key))
    noun_ex = load_extract(rc.noun_path(key)) if any(c.startswith("noun_target:") for c in all_cells) else None
    cm = cells_mod.cell_maps(all_cells, action=action, noun_ex=noun_ex)
    return {sid: np.sum([cm[c] for c in cs], axis=0).astype(np.float32) for sid, cs in srcs.items()}


def _replay_one(job: tuple) -> dict:
    out, key, srcs = job
    out = pathlib.Path(out)
    from AG3S.ag3s.experiments.studies import t51_replay as rp

    rc = records()
    todo = {s: c for s, c in srcs.items() if not rows_path(out, key, s).is_file()}
    if not todo:
        return {"key": key, "skipped": True}
    t0 = time.time()
    rows = rp.run_sources(rc.record_path(key), target=rc.target_of(key), sources=maps_for(rc, key, todo),
                          log=lambda s: None)
    for sid, rs in rows.items():
        p = rows_path(out, key, sid)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".partial")
        tmp.write_text("".join(json.dumps(r, default=float) + "\n" for r in rs))
        tmp.rename(p)
    return {"key": key, "sources": list(todo), "seconds": time.time() - t0}


def stage_replay(out: pathlib.Path, workers: int, keys: Optional[Sequence[str]] = None) -> dict:
    import multiprocessing as mp

    sel = load_selection()
    srcs = source_cells(sel)
    rc = records()
    keys = list(keys or rc.resolve_keys(None))
    jobs = [(str(out), k, srcs) for k in keys]
    if workers <= 1:
        res = [_replay_one(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(min(workers, 4)) as pool:
            res = list(pool.imap_unordered(_replay_one, jobs))
    (out / SUB).mkdir(parents=True, exist_ok=True)
    (out / SUB / "replay_jobs.json").write_text(json.dumps(res, indent=1))
    return {"jobs": res}


# ------------------------------------------------------------------------------ evaluate


def load_rows(out: pathlib.Path, key: str, sid: str) -> list[dict]:
    return [json.loads(x) for x in rows_path(out, key, sid).read_text().splitlines() if x.strip()]


def window_rows(rows: Sequence[dict], keep: Optional[np.ndarray]) -> list[dict]:
    return list(rows) if keep is None else [r for r in rows if keep[int(r["frame"])]]


def boot_ci(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> list[float]:
    v = np.asarray(values, np.float64)
    draws = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    b = v[draws].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def moving_block_ci(diffs: np.ndarray, block: int = BLOCK, n_boot: int = N_BOOT, seed: int = SEED) -> list[float]:
    d = np.asarray(diffs, np.float64)
    n = len(d)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    b = d[idx].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def stage_evaluate(out: pathlib.Path) -> dict:
    from AG3S.ag3s.experiments.studies import t51_study as st
    from AG3S.ag3s.experiments.studies import t61_select as ts

    sel = load_selection()
    srcs = list(source_cells(sel))
    sid_sel = srcs[0]
    rc = records()
    keys = list(rc.resolve_keys(None))
    masks = load_masks(out)
    assert sorted(masks) == sorted(keys), (sorted(masks), keys)
    rows = {(k, s): load_rows(out, k, s) for k in keys for s in srcs}
    for k in keys:  # frame index ↔ inference step must agree with the replay mask
        steps = [int(r["t_step"]) for r in rows[(k, BASELINE)]]
        assert steps == masks[k]["chunk_start_steps"], (k, len(steps), len(masks[k]["chunk_start_steps"]))
    grasp = {}
    for k in keys:
        with np.load(rc.grasp_path(k), allow_pickle=True) as z:
            grasp[k] = dict(zip([int(t) for t in z["t_step"]], [str(g) for g in z["grasp_state"]]))
    res = {"selection": sel["selected"], "selection_sha256": SELECTION_SHA256, "sources": srcs,
           "baseline": BASELINE, "references": list(REFERENCES), "keys": keys,
           "bootstrap": {"unit": "episode", "B": N_BOOT, "seed": SEED, "moving_block": BLOCK},
           "endpoints": {k: {"t_P": None if m["t_P"] < 0 else m["t_P"], "t_H": None if m["t_H"] < 0 else m["t_H"],
                             "frames_kept_P": int(m["P"].sum()), "frames_kept_H": int(m["H"].sum())}
                         for k, m in masks.items()},
           "windows": {}}
    for w in WINDOWS:
        per_ep = {s: {} for s in srcs}
        frame_diff = {s: [] for s in srcs if s != BASELINE}
        status = {s: {} for s in srcs}
        phase = {s: {} for s in srcs}
        for k in keys:
            keep = None if w == "full" else masks[k][w]
            base_by_frame = {int(r["frame"]): bool(r.get("identity_correct"))
                             for r in window_rows(rows[(k, BASELINE)], keep)}
            for s in srcs:
                rs = window_rows(rows[(k, s)], keep)
                m = st.record_metrics(rs)
                m["coverage"] = ts._coverage(rs)
                per_ep[s][k] = m
                for r in rs:
                    status[s][str(r.get("grounding_status"))] = status[s].get(str(r.get("grounding_status")), 0) + 1
                    ph = grasp[k].get(int(r["t_step"]), "unknown")
                    acc = phase[s].setdefault(ph, [0, 0])
                    acc[0] += int(bool(r.get("identity_correct")))
                    acc[1] += 1
                if s != BASELINE:
                    frame_diff[s] += [int(bool(r.get("identity_correct"))) - int(base_by_frame[int(r["frame"])])
                                      for r in rs if int(r["frame"]) in base_by_frame]
        summ = {}
        for s in srcs:
            gid = np.array([per_ep[s][k]["gt_identity"] for k in keys], np.float64)
            pur = np.array([per_ep[s][k]["purity_mean"] for k in keys], np.float64)
            cov = np.array([per_ep[s][k]["coverage"] for k in keys], np.float64)
            row = {"episode_mean_gt_identity": float(np.nanmean(gid)), "episode_mean_purity": float(np.nanmean(pur)),
                   "episode_mean_coverage": float(np.nanmean(cov)),
                   "n_frames": int(sum(per_ep[s][k]["n_frames"] for k in keys)),
                   "per_episode": per_ep[s], "grounding_status": status[s],
                   "phase_identity": {p: {"rate": a[0] / a[1] if a[1] else None, "n": a[1]} for p, a in phase[s].items()}}
            if s != BASELINE:
                b = np.array([per_ep[BASELINE][k]["gt_identity"] for k in keys], np.float64)
                d = gid - b
                ok = np.isfinite(d)
                row["paired_vs_L8H2"] = {
                    "mean_diff": float(np.mean(d[ok])), "ci95_episode_bootstrap": boot_ci(d[ok]),
                    "ci95_moving_block_frames": moving_block_ci(np.asarray(frame_diff[s])) if frame_diff[s] else None,
                    "n_episodes": int(ok.sum()), "n_up": int((d[ok] > 0).sum()), "n_down": int((d[ok] < 0).sum()),
                    "purity_diff": float(np.nanmean(pur - np.array([per_ep[BASELINE][k]["purity_mean"] for k in keys]))),
                    "coverage_diff": float(np.nanmean(cov - np.array([per_ep[BASELINE][k]["coverage"] for k in keys])))}
            summ[s] = row
        res["windows"][w] = summ
        print(w, {s: round(summ[s]["episode_mean_gt_identity"], 4) for s in srcs},
              {s: (round(summ[s]["paired_vs_L8H2"]["mean_diff"], 4), [round(x, 4) for x in summ[s]["paired_vs_L8H2"]["ci95_episode_bootstrap"]])
               for s in srcs if s != BASELINE})
    p = out / SUB / "heldout.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    return {"out": str(p), "primary": res["windows"]["P"][sid_sel].get("paired_vs_L8H2")}


def main(argv: Optional[Sequence[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=("endpoints", "replay", "evaluate"))
    ap.add_argument("--out", required=True, type=pathlib.Path, help="session-B run dir")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keys", nargs="*")
    a = ap.parse_args(argv)
    if a.stage == "endpoints":
        return stage_endpoints(a.out)
    if a.stage == "replay":
        return stage_replay(a.out, a.workers, a.keys)
    return stage_evaluate(a.out)


if __name__ == "__main__":
    main()
