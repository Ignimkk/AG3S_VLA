"""Score all 1296 attention cells on the 16D held-out records, phase-aware, per camera.

Scoring = `attention_report.score_maps` (unchanged, imported), i.e. the 14D metrics:
peak_on_target (argmax patch target coverage > 0.05), hit_beta for beta in {0, 0.5, 1}
(object-level KNOWS mass/area^beta argmax over COMPETITOR_BODIES with >= 100 px), target lift.
Ranking key = 14D `attention_report.py:443`: sort(-peak, -hit_b0.5, -lift).

Coverage: policy camera geometry (299x224 -> 224x224, gt.py `cov_pol`).
Visibility rules in 640x480 px as in 14D (`area_640`): target >= 200 px, competitor >= 100 px.

Phases (left/right gripper column of the 16D state, `policy_record.grasp_onset`):
  reach  : frames before the gripper first leaves its open value     -> target = prompt fruit
  carry  : from that frame until the gripper is back >= 0.9 (release) -> target = crate
  after_release : not scored (counted).

    PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-openpi-live/bin/python score.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib

import numpy as np

from AG3S.ag3s.experiments.reports.attention_report import (
    COMPETITOR_BODIES, score_maps, target_from_prompt)
from AG3S.ag3s.experiments.sources.policy_record import gripper_columns_for_state

D = pathlib.Path(_ag3s_workspace_path('outputs/verify/attn16d'))
EPISODES = ["ep1800_t2", "ep1807", "ep1850", "ep1900"]
SUPPLEMENT = ["ep1800_headfix"]
BETAS = (0.0, 0.5, 1.0)
MIN_TARGET_PX = 200.0
MIN_BODY_PX = 100.0
PATCH_HIT = 0.05
OPEN_TOL = 0.02
RELEASE_LEVEL = 0.9
CAMS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
VIEWS = CAMS + ["any"]
KEY_CELLS = {"L8h2_d0_last": (0, "last", 8, 2), "L7h0_d0_last": (0, "last", 7, 0)}


def phases(state):
    cols = gripper_columns_for_state(state.shape[1])
    onset, col = None, None
    for c in cols:
        tr = state[:, c]
        hit = np.flatnonzero(tr < tr[0] - OPEN_TOL)
        if hit.size:
            onset, col = int(hit[0]), c
            break
    n = len(state)
    lab = np.array(["reach"] * n, dtype=object)
    info = {"onset_index": onset, "gripper_col": col, "release_index": None}
    if onset is not None:
        tr = state[:, col]
        rel = [i for i in range(onset + 1, n) if tr[i] >= RELEASE_LEVEL]
        release = rel[0] if rel else n
        info["release_index"] = rel[0] if rel else None
        lab[onset:release] = "carry"
        lab[release:] = "after_release"
    return lab, info


def per_frame(att, cov, area, keep_ti, ti, competitor_mask):
    """att: (F,16,16) maps; returns per-frame arrays (peak_hit, hit_b*, lift, mass)."""
    s = score_maps(att, cov, area, keep_ti, ti, competitor_mask=competitor_mask, betas=BETAS,
                   min_body_px=MIN_BODY_PX, patch_hit_cov=PATCH_HIT)
    out = {"peak": s["peak_target_cov"] > PATCH_HIT, "lift": s["lift"][:, ti], "mass": s["mass"][:, ti]}
    for b in BETAS:
        out[f"hit_b{b:g}"] = s[f"winner_b{b:g}"] == ti
    return out


METRICS = ["peak", "hit_b0", "hit_b0.5", "hit_b1", "lift", "mass"]


def load_episode(name):
    att = np.load(D / f"att_{name}.npz")
    gt = np.load(D / f"gt_{name}.npz")
    A = np.asarray(att["attention"], np.float32)  # (F, d, a, L, H, cam, 16, 16)
    assert [str(c) for c in att["cameras"]] == CAMS
    assert A.shape[0] == gt["cov_pol"].shape[0]
    prompt = str(gt["prompt"])
    return dict(A=A, gt=gt, prompt=prompt, fruit=target_from_prompt(prompt),
                denoise=[int(x) for x in att["denoise_steps"]], aggs=[str(x) for x in att["aggregations"]],
                bodies=[str(b) for b in gt["bodies"]], state=gt["state"], t_step=gt["t_step"])


def frame_table(ep, phase_name, cov_key="cov_pol"):
    """Per-frame indicators for every cell, per camera. Returns dict view -> metric -> (F_kept, d,a,L,H)."""
    A, gt, bodies = ep["A"], ep["gt"], ep["bodies"]
    lab, _ = phases(ep["state"])
    target = ep["fruit"] if phase_name == "reach" else "crate"
    ti = bodies.index(target)
    competitor = np.asarray([b in COMPETITOR_BODIES for b in bodies], bool)
    frames = np.flatnonzero(lab == phase_name)
    nd, na, nl, nh = A.shape[1:5]
    res = {}
    kept = {}
    for ci, cam in enumerate(CAMS):
        cov = gt[cov_key][frames, ci]
        area = gt["area_640"][frames, ci]
        keep = area[:, ti] >= MIN_TARGET_PX
        kept[cam] = keep
        arr = {m: np.full((len(frames), nd, na, nl, nh), np.nan) for m in METRICS}
        if keep.any():
            for d in range(nd):
                for a in range(na):
                    for l in range(nl):
                        for h in range(nh):
                            pf = per_frame(A[frames, d, a, l, h, ci], cov, area, keep, ti, competitor)
                            for m in METRICS:
                                arr[m][:, d, a, l, h] = pf[m]
        res[cam] = arr
    # baselines per camera: head-average at every (d, a), uniform
    base = {}
    for ci, cam in enumerate(CAMS):
        cov = gt[cov_key][frames, ci]
        area = gt["area_640"][frames, ci]
        keep = kept[cam]
        b = {}
        for d in range(nd):
            for a in range(na):
                ha = A[frames, d, a, :, :, ci].mean(axis=(1, 2))
                b[(d, a)] = per_frame(ha, cov, area, keep, ti, competitor) if len(frames) else None
        b["uniform"] = per_frame(np.ones((len(frames), 16, 16), np.float32), cov, area, keep, ti,
                                 competitor) if len(frames) else None
        base[cam] = b
    return dict(frames=frames, target=target, kept=kept, cells=res, base=base)


def agg_view(tables, view):
    """Pool frames across episode tables for one view -> dict metric -> (d,a,L,H) plus n."""
    parts = {m: [] for m in METRICS}
    n = 0
    for t in tables:
        if view == "any":
            anykeep = np.zeros(len(t["frames"]), bool)
            for cam in CAMS:
                anykeep |= t["kept"][cam]
            if not anykeep.any():
                continue
            stack = {m: np.stack([np.where(t["kept"][cam][:, None, None, None, None],
                                           t["cells"][cam][m], np.nan) for cam in CAMS])
                     for m in METRICS}
            for m in METRICS:
                v = stack[m][:, anykeep]
                if m in ("lift", "mass"):
                    parts[m].append(np.nanmax(v, axis=0))
                else:
                    parts[m].append(np.nanmax(np.nan_to_num(v, nan=0.0), axis=0))
            n += int(anykeep.sum())
        else:
            k = t["kept"][view]
            if not k.any():
                continue
            for m in METRICS:
                parts[m].append(t["cells"][view][m][k])
            n += int(k.sum())
    if n == 0:
        return None
    return {"n": n, **{m: np.concatenate(parts[m]).astype(np.float64).mean(axis=0) for m in METRICS}}


def agg_base(tables, view, which):
    """Same pooling for a baseline map (which = (d,a) or 'uniform')."""
    parts = {m: [] for m in METRICS}
    n = 0
    for t in tables:
        if view == "any":
            anykeep = np.zeros(len(t["frames"]), bool)
            for cam in CAMS:
                anykeep |= t["kept"][cam]
            if not anykeep.any():
                continue
            for m in METRICS:
                v = np.stack([np.where(t["kept"][cam], t["base"][cam][which][m].astype(np.float64), np.nan)
                              for cam in CAMS])[:, anykeep]
                parts[m].append(np.nanmax(v, axis=0) if m in ("lift", "mass")
                                else np.nanmax(np.nan_to_num(v, nan=0.0), axis=0))
            n += int(anykeep.sum())
        else:
            k = t["kept"][view]
            if not k.any():
                continue
            for m in METRICS:
                parts[m].append(t["base"][view][which][m][k].astype(np.float64))
            n += int(k.sum())
    if n == 0:
        return None
    return {"n": n, **{m: float(np.concatenate(parts[m]).mean()) for m in METRICS}}


def ranking(cube, denoise, aggs):
    rows = []
    nd, na, nl, nh = cube["peak"].shape
    for d in range(nd):
        for a in range(na):
            for l in range(nl):
                for h in range(nh):
                    rows.append({"denoise": denoise[d], "agg": aggs[a], "layer": l, "head": h,
                                 **{m: float(cube[m][d, a, l, h]) for m in METRICS}})
    rows.sort(key=lambda r: (-r["peak"], -r["hit_b0.5"], -r["lift"]))
    return rows


def cell_rank(rows, cell):
    d, a, l, h = cell
    key = lambda r: (-r["peak"], -r["hit_b0.5"], -r["lift"])  # noqa: E731
    me = next(r for r in rows if (r["denoise"], r["agg"], r["layer"], r["head"]) == (d, a, l, h))
    better = sum(1 for r in rows if key(r) < key(me))
    peak_better = sum(1 for r in rows if r["peak"] > me["peak"] + 1e-12)
    peak_ties = sum(1 for r in rows if abs(r["peak"] - me["peak"]) <= 1e-12)
    return {"rank": better + 1, "rank_peak_only_best": peak_better + 1,
            "peak_ties": peak_ties, **{m: me[m] for m in METRICS}}


def main():
    eps = {n: load_episode(n) for n in EPISODES + SUPPLEMENT}
    denoise, aggs = eps[EPISODES[0]]["denoise"], eps[EPISODES[0]]["aggs"]
    d0, alast = denoise.index(0), aggs.index("last")
    out = {"episodes": {}, "phases": {}}
    cubes = {}
    for name, ep in eps.items():
        lab, info = phases(ep["state"])
        out["episodes"][name] = {"prompt": ep["prompt"], "fruit": ep["fruit"], "n_frames": int(len(lab)),
                                 "phase_counts": {p: int((lab == p).sum()) for p in ("reach", "carry", "after_release")},
                                 **info}
    for phase in ("reach", "carry"):
        tabs = {n: frame_table(eps[n], phase) for n in EPISODES + SUPPLEMENT}
        ph = {"kept": {n: {c: [int(tabs[n]["kept"][c].sum()), int(len(tabs[n]["frames"]))] for c in CAMS}
                       for n in tabs},
              "target": {n: tabs[n]["target"] for n in tabs}}
        scopes = {"pooled": [tabs[n] for n in EPISODES], **{n: [tabs[n]] for n in EPISODES + SUPPLEMENT}}
        ph["scopes"] = {}
        for sname, tl in scopes.items():
            ph["scopes"][sname] = {}
            for view in VIEWS:
                cube = agg_view(tl, view)
                if cube is None:
                    ph["scopes"][sname][view] = None
                    continue
                cubes[(phase, sname, view)] = cube
                rows = ranking(cube, denoise, aggs)
                ent = {"n_frames": cube["n"], "top15": rows[:15],
                       "key_cells": {k: cell_rank(rows, (v[0], v[1], v[2], v[3])) for k, v in KEY_CELLS.items()},
                       "baseline_head_average_d0_last": agg_base(tl, view, (d0, alast)),
                       "baseline_head_average_top_cell_setting": agg_base(
                           tl, view, (denoise.index(rows[0]["denoise"]), aggs.index(rows[0]["agg"]))),
                       "baseline_uniform": agg_base(tl, view, "uniform"),
                       "n_cells_peak_ge_L8h2": int(sum(1 for r in rows if r["peak"] >= cube["peak"][d0, alast, 8, 2] - 1e-12)),
                       "max_peak": rows[0]["peak"],
                       "n_cells_at_max_peak": int(sum(1 for r in rows if r["peak"] >= rows[0]["peak"] - 1e-12))}
                ph["scopes"][sname][view] = ent
        out["phases"][phase] = ph
        # cross-check with 640x480 coverage for the two key cells (pooled)
        tabs640 = [frame_table(eps[n], phase, cov_key="cov_640") for n in EPISODES]
        xc = {}
        for view in VIEWS:
            cube = agg_view(tabs640, view)
            if cube is None:
                continue
            rows = ranking(cube, denoise, aggs)
            xc[view] = {"top1": rows[0], **{k: cell_rank(rows, v) for k, v in KEY_CELLS.items()}}
        out["phases"][phase]["crosscheck_cov640_pooled"] = xc
    # save cubes
    np.savez_compressed(D / "cubes.npz", **{f"{p}|{s}|{v}|{m}": c[m] for (p, s, v), c in cubes.items()
                                             for m in METRICS},
                        **{f"{p}|{s}|{v}|n": np.asarray(c["n"]) for (p, s, v), c in cubes.items()})
    out["config"] = {"episodes": EPISODES, "supplement": SUPPLEMENT, "betas": BETAS,
                     "min_target_px_640x480": MIN_TARGET_PX, "min_body_px_640x480": MIN_BODY_PX,
                     "patch_hit_cov": PATCH_HIT, "open_tol": OPEN_TOL, "release_level": RELEASE_LEVEL,
                     "denoise": denoise, "aggs": aggs, "competitors": list(COMPETITOR_BODIES),
                     "rank_key": "sort(-peak, -hit_b0.5, -lift)  (attention_report.py:443)"}
    (D / "score.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("wrote", D / "score.json")


if __name__ == "__main__":
    main()
