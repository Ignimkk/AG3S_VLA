"""SL6 B6 — noun-query attention head re-selection for the Panda VLA (``SL6.task.md`` §4, rule fixed before results).

* candidates: noun-query cells layer 2–17 × head 0–7 (128, RB-Y1 T51b set) from the B5 run
  (``attention_audit.py --noun-rows``: ``noun_rows/<stem>_noun.npz`` ``noun_image (calls, layers, 8, 768)``);
  references (not candidates): the carried-over ``noun_target:L10H7`` (it is one of the 128) and Panda V11AB
  ``action:last:L8H2`` (``pooled_f16[:, 8, 2]`` of the SL4b audit run, when given);
* data: libero_spatial task 0–9 × init 1–4 (dev), requests whose B3 label is **approach**, **head** camera;
* truth: image-instant head segmentation of the request (``segP/head`` of the B3 ``phase_export`` track — the state
  the request's image shows; the rollout's own ``segmentation_geom/head`` is rendered at return time, SL3:
  100–300 px differ; ``--seg-source rollout`` uses it instead) → the BDDL target's geoms;
* per request and cell: the 16×16 head grid is un-mirrored and resampled to 256×256 by V11AB's own resampling
  (`AG3S.adapters.panda.attention.attention_pixel_map` = ``GridAttentionAdapter("bilinear").to_pixel_map``), then
    identity  = among the top 1 % pixels by attention (656 of 65 536; ties at the threshold -> lower flat index) the
                instance with the most pixels is the target (strictly more than every other label; labels = every
                BDDL object / fixture, ``robot`` (robot0_ / gripper0_ / mount0_ bodies), ``other`` geoms, ``none``
                (−1));
    mass      = target-pixel attention / all-pixel attention;
    coverage  = share of target pixels inside the top 10 % pixels (undefined when the target is not visible);
* score of a cell = mean over the 10 tasks of the per-task mean over requests (coverage: of the requests where the
  target is visible); rank by (identity, mass, coverage), then lower layer, lower head;
* decision: the top cell, unless its identity is less than 2 %p above L10H7's -> keep L10H7.

usage (``.venv-ag3s``, CPU)::

    python -m AG3S.ag3s.experiments.studies.sl6_head_select --noun-run <B5 run> --labels-run <B3/B4 run> \\
        [--attn-run <SL4b audit run>] --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np

N_PIX = 256 * 256
TOP_IDENTITY = 0.01
TOP_COVERAGE = 0.10
KEEP_CELL = (10, 7)
KEEP_MARGIN = 0.02
REFERENCE_ACTION_CELL = (8, 2)
HEAD_TOKENS = (0, 256)
ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")
EVAL_LABEL = "approach"


# ------------------------------------------------------------------------------------------------ pure parts


def resample_batch(grids: np.ndarray, out: int = 256) -> np.ndarray:
    """``(C, 16, 16)`` -> ``(C, out, out)``: `attention_lifting._resample(…, "bilinear")` per grid, same float32 ops."""
    src = np.asarray(grids, np.float32)
    _, h, w = src.shape
    H = W = int(out)
    fy = np.clip((np.arange(H) + 0.5) * h / H - 0.5, 0.0, h - 1.0)
    fx = np.clip((np.arange(W) + 0.5) * w / W - 0.5, 0.0, w - 1.0)
    y0 = np.floor(fy).astype(np.int64)
    x0 = np.floor(fx).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (fy - y0).astype(np.float32)[:, None]
    wx = (fx - x0).astype(np.float32)[None, :]
    a = src[:, y0][:, :, x0]
    b = src[:, y0][:, :, x1]
    c = src[:, y1][:, :, x0]
    d = src[:, y1][:, :, x1]
    top = a * (1.0 - wx) + b * wx
    bot = c * (1.0 - wx) + d * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


def pixel_maps(head_grids_policy: np.ndarray) -> np.ndarray:
    """Policy-orientation ``(C, 16, 16)`` head grids -> OpenCV-orientation ``(C, 256, 256)`` pixel maps
    (`attention.attention_pixel_map`: un-mirror = column reversal, then the bilinear resampling)."""
    g = np.ascontiguousarray(np.asarray(head_grids_policy, np.float32)[..., ::-1])
    return resample_batch(g)


def top_mask(values: np.ndarray, frac: float) -> np.ndarray:
    """Boolean mask of the ``ceil(frac · N)`` largest entries of a flat array; ties at the threshold -> lower index
    (= ``np.argsort(-v, kind="stable")[:k]``)."""
    v = np.asarray(values).reshape(-1)
    n = v.size
    k = int(math.ceil(frac * n))
    thr = np.partition(v, n - k)[n - k]
    m = v > thr
    need = k - int(m.sum())
    if need > 0:
        eq = np.flatnonzero(v == thr)[:need]
        m[eq] = True
    return m


def label_image(seg: np.ndarray, instances: Mapping[str, Sequence[int]], target: str, robot_geoms: Sequence[int]
                ) -> tuple[np.ndarray, list[str]]:
    """Geom-id segmentation -> label index per pixel; names[0] = target, then the other instances, ``robot``,
    ``other``, ``none``."""
    names = [target] + sorted(n for n in instances if n != target) + ["robot", "other", "none"]
    idx = {n: i for i, n in enumerate(names)}
    seg = np.asarray(seg, np.int64)
    n_geom = int(max(seg.max(), max((max(v) for v in instances.values() if len(v)), default=0),
                     max(robot_geoms, default=0))) + 1
    lut = np.full(n_geom + 1, idx["other"], np.int64)        # index n_geom+... unused; -1 handled below
    for g in robot_geoms:
        lut[int(g)] = idx["robot"]
    for n, gs in instances.items():
        for g in gs:
            lut[int(g)] = idx[n]
    out = np.where(seg < 0, idx["none"], lut[np.clip(seg, 0, n_geom)])
    return out, names


def request_metrics(maps: np.ndarray, labels: np.ndarray, n_labels: int) -> dict[str, np.ndarray]:
    """Per cell (row of ``maps (C, 256, 256)``): identity (bool), mass, coverage (nan when no target pixel)."""
    lab = np.asarray(labels).reshape(-1)
    tgt = lab == 0
    n_t = int(tgt.sum())
    C = maps.shape[0]
    ident = np.zeros(C, bool)
    mass = np.zeros(C)
    cov = np.full(C, np.nan)
    for c in range(C):
        v = maps[c].reshape(-1)
        tot = float(v.sum(dtype=np.float64))
        mass[c] = float(v[tgt].sum(dtype=np.float64)) / tot if tot > 0 else 0.0
        m1 = top_mask(v, TOP_IDENTITY)
        counts = np.bincount(lab[m1], minlength=n_labels)
        ident[c] = bool(counts[0] > np.max(counts[1:])) if n_t else False
        if n_t:
            cov[c] = float(top_mask(v, TOP_COVERAGE)[tgt].mean())
    return {"identity": ident, "mass": mass, "coverage": cov}


def task_average(per_request: Mapping[int, list], nan_skip: bool = False) -> tuple[np.ndarray, dict]:
    """``{task: [(C,) arrays]}`` -> mean over tasks of the per-task mean (``nan_skip``: nanmean)."""
    agg = np.nanmean if nan_skip else np.mean
    per_task = {t: agg(np.stack(v), axis=0) for t, v in sorted(per_request.items()) if v}
    return agg(np.stack(list(per_task.values())), axis=0), per_task


def rank(identity: np.ndarray, mass: np.ndarray, coverage: np.ndarray, cells: Sequence[tuple[int, int]]) -> list[int]:
    """Cell indices ordered by (identity, mass, coverage) descending, then lower layer, lower head."""
    cov = np.where(np.isnan(coverage), -np.inf, coverage)
    return sorted(range(len(cells)), key=lambda i: (-identity[i], -mass[i], -cov[i], cells[i][0], cells[i][1]))


def decide(identity: np.ndarray, order: Sequence[int], cells: Sequence[tuple[int, int]],
           keep=KEEP_CELL, margin=KEEP_MARGIN) -> dict:
    best = order[0]
    k = cells.index(tuple(keep))
    diff = float(identity[best] - identity[k])
    chosen = cells[best] if (best != k and diff >= margin) else tuple(keep)
    return {"top_cell": list(cells[best]), "top_identity": float(identity[best]),
            "keep_cell": list(keep), "keep_identity": float(identity[k]), "diff_pp": 100.0 * diff,
            "rule": f"top cell unless its identity is < {100 * margin:.0f} %p above L{keep[0]}H{keep[1]} -> keep",
            "chosen_cell": list(chosen), "spec": f"noun_target:L{chosen[0]}H{chosen[1]}",
            "kept_L10H7": bool(tuple(chosen) == tuple(keep))}


# ------------------------------------------------------------------------------------------------ I/O


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_labels(labels_run: pathlib.Path) -> dict[str, dict]:
    rows = [json.loads(l) for l in (labels_run / "labels" / "request_labels.jsonl").read_text().splitlines() if l.strip()]
    return {r["stem"]: r for r in rows}


def select(noun_run: pathlib.Path, labels_run: pathlib.Path, attn_run: Optional[pathlib.Path] = None, *,
           seg_source: str = "image_instant", phase_dir: Optional[pathlib.Path] = None) -> dict:
    labels = load_labels(labels_run)
    phase_dir = phase_dir or labels_run / "phase"
    index = [json.loads(l) for l in (noun_run / "noun_index.jsonl").read_text().splitlines() if l.strip()]
    attn_index = None
    if attn_run is not None:
        attn_index = {(r["task_id"], r["episode_idx"]): r for r in
                      (json.loads(l) for l in (attn_run / "attention_index.jsonl").read_text().splitlines() if l.strip())}
    per_req: dict[str, dict[int, list]] = {m: {} for m in ("identity", "mass", "coverage")}
    ref_req: dict[str, dict[int, list]] = {m: {} for m in ("identity", "mass", "coverage")}
    cells: Optional[list[tuple[int, int]]] = None
    counts = {"requests_used": 0, "requests_target_invisible": 0, "requests_noun_not_ok": 0, "episodes": 0}
    episodes = []
    for row in index:
        stem = f"{row['suite']}_t{int(row['task_id'])}_e{int(row['episode_idx'])}"
        init = int(row.get("init_state_index") if row.get("init_state_index") is not None else -1)
        if row["suite"] != "libero_spatial" or init not in (1, 2, 3, 4):
            raise ValueError(f"{stem}: not a SL6 dev episode (libero_spatial init 1-4)")
        lab_row = labels[stem]
        tgt = row["target"]["target"]
        instances = {k: list(v) for k, v in row["target"]["instances_geom_ids"].items()}
        geom_path = pathlib.Path(row["source_run_dir"]) / row["geom_npz"]
        with np.load(geom_path) as g:
            gb, bn = np.asarray(g["model_geom_bodyid"]), [str(b) for b in g["model_body_names"]]
            rec_seg = g["segmentation_geom/head"] if seg_source == "rollout" else None
            obs_all = np.asarray(g["obs_index"], np.int64)
        robot_geoms = [i for i, b in enumerate(gb.tolist()) if bn[int(b)].startswith(ROBOT_PREFIXES)]
        with np.load(noun_run / row["noun_npz"]) as z:
            img = np.asarray(z["noun_image"], np.float32)
            layers = [int(x) for x in z["layers"]]
            ok = np.asarray(z["noun_ok"], bool)
            obs = np.asarray(z["obs_index"], np.int64)
        if list(obs) != list(lab_row["frames"]):
            raise ValueError(f"{stem}: noun-row requests != B3 label frames")
        H = img.shape[2]
        this_cells = [(l, h) for l in layers for h in range(H)]
        if cells is None:
            cells = this_cells
        elif cells != this_cells:
            raise ValueError(f"{stem}: another cell set")
        with np.load(phase_dir / f"phase_{stem}.npz") as t:
            call_frames = [int(x) for x in t["call_frames"]]
            seg_p = t["segP/head"] if seg_source == "image_instant" else None
        ref_pooled = None
        if attn_index is not None:
            ai = attn_index[(int(row["task_id"]), int(row["episode_idx"]))]
            with np.load(attn_run / ai["attn_npz"]) as a:
                if not np.array_equal(np.asarray(a["obs_index"], np.int64), obs):
                    raise ValueError(f"{stem}: SL4b audit requests differ")
                ref_pooled = np.asarray(a["pooled_f16"][:, REFERENCE_ACTION_CELL[0], REFERENCE_ACTION_CELL[1],
                                        HEAD_TOKENS[0]:HEAD_TOKENS[1]], np.float32)
        used = 0
        for k, (f, lab) in enumerate(zip(obs, lab_row["labels"])):
            if lab != EVAL_LABEL:
                continue
            if not ok[k]:
                counts["requests_noun_not_ok"] += 1
                continue
            seg = seg_p[call_frames.index(int(f))] if seg_p is not None else rec_seg[int(np.flatnonzero(obs_all == f)[0])]
            lab_img, names = label_image(seg, instances, tgt, robot_geoms)
            grids = img[k, :, :, HEAD_TOKENS[0]:HEAD_TOKENS[1]].reshape(-1, 16, 16)
            met = request_metrics(pixel_maps(grids), lab_img, len(names))
            t = int(row["task_id"])
            for m in per_req:
                per_req[m].setdefault(t, []).append(met[m].astype(np.float64))
            if ref_pooled is not None:
                rm = request_metrics(pixel_maps(ref_pooled[k][None].reshape(1, 16, 16)), lab_img, len(names))
                for m in ref_req:
                    ref_req[m].setdefault(t, []).append(rm[m].astype(np.float64))
            counts["requests_target_invisible"] += int(not (lab_img == 0).any())
            used += 1
        counts["requests_used"] += used
        counts["episodes"] += 1
        episodes.append({"stem": stem, "task_id": int(row["task_id"]), "init": init, "approach_requests": used,
                         "target": tgt, "noun": (row.get("lookup_first") or {}).get("noun")})
    if cells is None or not counts["requests_used"]:
        raise ValueError("no approach request to score")
    ident, ident_t = task_average(per_req["identity"])
    mass, mass_t = task_average(per_req["mass"])
    cov, cov_t = task_average(per_req["coverage"], nan_skip=True)
    order = rank(ident, mass, cov, cells)
    dec = decide(ident, order, cells)
    L = sorted({c[0] for c in cells})
    Hn = max(c[1] for c in cells) + 1

    def table(v):
        return np.asarray(v).reshape(len(L), Hn).tolist()

    def cell_row(i):
        c = cells[i]
        return {"cell": list(c), "identity": float(ident[i]), "mass": float(mass[i]), "coverage": float(cov[i]),
                "identity_per_task": {str(t): float(v[i]) for t, v in ident_t.items()}}

    out = {"step": "SL6-B6", "definition": __doc__, "seg_source": seg_source,
           "inputs": {"noun_run": str(noun_run), "labels_run": str(labels_run), "attn_run": None if attn_run is None
                      else str(attn_run), "phase_dir": str(phase_dir)},
           "counts": counts, "layers": L, "heads": Hn,
           "identity": table(ident), "mass": table(mass), "coverage": table(cov),
           "ranking_top10": [cell_row(i) for i in order[:10]],
           "L10H7": cell_row(cells.index(KEEP_CELL)),
           "rank_of_L10H7": int(order.index(cells.index(KEEP_CELL))) + 1,
           "decision": dec, "episodes": episodes}
    if ref_req["identity"]:
        ri, ri_t = task_average(ref_req["identity"])
        rmass, _ = task_average(ref_req["mass"])
        rcov, _ = task_average(ref_req["coverage"], nan_skip=True)
        out["reference_action_last_L8H2"] = {
            "note": "Panda V11AB action:last:L8H2 (SL4b audit pooled_f16, float16) — reference only, not a candidate",
            "identity": float(ri[0]), "mass": float(rmass[0]), "coverage": float(rcov[0]),
            "identity_per_task": {str(t): float(v[0]) for t, v in ri_t.items()}}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--noun-run", required=True, type=pathlib.Path)
    ap.add_argument("--labels-run", required=True, type=pathlib.Path)
    ap.add_argument("--attn-run", default=None, type=pathlib.Path, help="SL4b audit run: action:last:L8H2 reference")
    ap.add_argument("--phase-dir", default=None, type=pathlib.Path)
    ap.add_argument("--seg-source", choices=("image_instant", "rollout"), default="image_instant")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    res = select(a.noun_run, a.labels_run, a.attn_run, seg_source=a.seg_source, phase_dir=a.phase_dir)
    with open(a.out, "x") as f:
        f.write(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"decision": res["decision"], "counts": res["counts"],
                      "top3": res["ranking_top10"][:3], "L10H7": res["L10H7"]}, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
