"""T79 — ATS time tracking by hidden-state re-search: method, metrics, train-only selection, verdict
(`handoff/T79.task.md` §1, §3; fixed before results).

Inputs (produced elsewhere, read only):
  EVENTS   `t79_events` json (+ its ``.geom.npz``): pre-grasp rows with GT visible-surface centre ``vis_c``, recorded
           ATS (``centroid``, target-point classes), reference request ``ref``, ``appearance_patches`` of ATS-correct
           requests, ``event`` ∈ {a, b, ab, normal, no_ref}, and per-patch depth tables (N, S, C) + noun attention.
  PATCHES  `pi05_hidden --pool patches` npz per run (``<safe label>.npz``): head image-token hidden ``[R, Ls, 256,
           2048]`` of the event / sampled-normal requests and their references (``request`` = order index).

Hidden re-search at request j with reference r = ``ref(j)``, layer l:
  a      = mean over r's appearance patches of h_l(r, patch)                      (apple appearance vector)
  s_k    = cos(h_l(j, k), a), k = 0..255                                          (similarity map)
  region = the largest 4-connected component of {k : s_k >= τ} (ties: larger Σ s)  -> abstain if empty
  centre = Σ_{k ∈ region} S_k / Σ N_k  (head depth points of those patches, robot spheres and support removed)
           -> abstain if the region has no depth point.
Reference methods (no verdict): the same with the recorded noun attention (L10H7, ``attention / max``) as s_k, and
with "centred" hidden (each frame's 256-patch mean subtracted from its patches before the cosine).
Baseline (the current ATS): the mean of the recorded target points (abstain if none) — the T75 ATS centre definition;
lead 2026-10-31, before results: the ball centre ``manipulated.centroid`` sits 20–28 mm from the visible-surface centre
even when ATS is right (smoke), so it is only reported (``baseline_ball_centre``), not used for the verdict.

Metrics per scored request (GT = the head-visible apple surface centre ``vis_c``):
  found   centre within 20 mm of ``vis_c``
  wrong   centre > 50 mm from ``vis_c``, or the majority GT class (8 mm; none excluded unless all none) of the points
          that made the centre is another object (fruit / crate / table / obstacle / robot)
  abstain no centre (counts as neither found nor wrong)
Scored set: pre-grasp event / sampled-normal requests with a reference, a GT visible surface (``vis_n`` > 0) and the
hidden of both j and r stored. Baseline and re-search are scored on the same set. Rates are over that set.

Selection (train scenes only; test never consulted): over layers {4, 8, 12, 16} × τ grid, keep the settings with
train event wrong <= 10 % and train normal found >= baseline normal found − 5 %p; among them the highest train event
found (ties -> higher τ, then lower layer). If none is admissible: highest (event found − event wrong), flagged.
Verdict (test scenes, all three): event found(hidden) − event found(baseline) >= 30 %p ∧ event wrong(hidden) <= 10 %
∧ normal found(hidden) >= normal found(baseline) − 5 %p.

usage (.venv-ag3s, CPU):
  PYTHONPATH=$PWD .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.t79_track OUT_DIR \\
      --events t79_events.json --patches-dir hidden/patches [--focus LABEL --focus-requests 15:75]
Writes ``OUT_DIR/t79_track.json`` (selection table, chosen setting, test metrics, verdict, layer curve, per-request
results, focus-run table with similarity maps for figures).
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies.t79_events import CLASSES, EVENT_TYPES, GRID

FOUND_MM = 20.0
WRONG_MM = 50.0
EVENT_GAIN = 0.30
WRONG_MAX = 0.10
NORMAL_DROP = 0.05
LAYERS = (4, 8, 12, 16)
#: uniform 0.001 steps over [0, 1): the raw cosines of this residual stream sit high (a large component shared by
#: every patch), so a 0.01 grid would not resolve the useful range.
TAU_GRID = tuple(round(x, 3) for x in np.arange(0.0, 1.0, 0.001))
APPLE, NONE = CLASSES.index("apple"), CLASSES.index("none")
FOCUS_LABEL = "T71/REG_R0_k0/runs/B_ROI_FPR/B_ROI_FPR_ep211_s2113"


# ================================================================================ pure parts (tested)


def safe_name(label: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(label))


def largest_component(score: np.ndarray, tau: float) -> np.ndarray:
    """Patch ids (row-major on the 16 × 16 grid) of the largest 4-connected component of ``score >= tau``
    (ties: larger summed score, then the lower first id). Empty if nothing passes."""
    from scipy import ndimage

    s = np.asarray(score, float).reshape(GRID, GRID)
    lab, n = ndimage.label(s >= tau)  # default structure = 4-connectivity
    if n == 0:
        return np.zeros(0, np.int64)
    flat, sf = lab.ravel(), s.ravel()
    size = np.bincount(flat, minlength=n + 1)[1:]
    mass = np.bincount(flat, weights=np.where(flat > 0, sf, 0.0), minlength=n + 1)[1:]
    best = sorted(range(n), key=lambda i: (-size[i], -mass[i], i))[0] + 1
    return np.flatnonzero(flat == best)


def centre_of(patches: np.ndarray, N: np.ndarray, S: np.ndarray, C: np.ndarray) -> tuple[Optional[np.ndarray], int, int]:
    """``(centre or None, majority class id, n points)`` of the depth points of a patch set."""
    patches = np.asarray(patches, np.int64)
    n = int(N[patches].sum()) if len(patches) else 0
    if n == 0:
        return None, NONE, 0
    cnt = C[patches].sum(0)
    return S[patches].sum(0) / n, majority_class(cnt), n


def majority_class(counts: Sequence[int]) -> int:
    """Most common class excluding ``none`` (``none`` only if every point is none)."""
    c = np.asarray(counts, np.int64).copy()
    if c.sum() == 0:
        return NONE
    c[NONE] = -1
    return int(np.argmax(c)) if c.max() > 0 else NONE


def score(centre, majority: int, vis_c) -> dict:
    """found / wrong / abstain of one centre against the GT visible-surface centre."""
    if centre is None:
        return {"abstain": True, "found": False, "wrong": False, "err_mm": None}
    err = float(np.linalg.norm(np.asarray(centre, float) - np.asarray(vis_c, float)) * 1e3)
    return {"abstain": False, "found": err <= FOUND_MM,
            "wrong": bool(err > WRONG_MM or majority not in (APPLE, NONE)), "err_mm": round(err, 2)}


def cosine_map(H: np.ndarray, a: np.ndarray) -> np.ndarray:
    """cos(H[k], a) for H [256, D] and a [D] (float64 arithmetic)."""
    H = np.asarray(H, np.float64)
    a = np.asarray(a, np.float64)
    den = np.linalg.norm(H, axis=1) * np.linalg.norm(a)
    return np.where(den > 0, H @ a / np.maximum(den, 1e-300), 0.0)


def rates(res: Sequence[dict]) -> dict:
    n = len(res)
    f = sum(r["found"] for r in res)
    w = sum(r["wrong"] for r in res)
    ab = sum(r["abstain"] for r in res)
    return {"n": n, "found": f / n if n else None, "wrong": w / n if n else None, "abstain": ab / n if n else None,
            "n_found": f, "n_wrong": w, "n_abstain": ab}


def admissible(ev: dict, nm: dict, base_nm: dict) -> bool:
    return bool(ev["wrong"] is not None and ev["wrong"] <= WRONG_MAX and nm["found"] is not None
                and base_nm["found"] is not None and nm["found"] >= base_nm["found"] - NORMAL_DROP - 1e-12)


def select(table: Sequence[dict], base_nm_found: float) -> dict:
    """Train-only choice over ``table`` rows ``{layer, tau, event: rates, normal: rates}`` (module docstring rule)."""
    ok = [t for t in table if admissible(t["event"], t["normal"], {"found": base_nm_found})]
    if ok:
        best = sorted(ok, key=lambda t: (-t["event"]["found"], -t["tau"], t["layer"]))[0]
        return {**best, "admissible": True, "n_admissible": len(ok)}
    best = sorted(table, key=lambda t: (-(t["event"]["found"] - t["event"]["wrong"]), -t["tau"], t["layer"]))[0]
    return {**best, "admissible": False, "n_admissible": 0}


def verdict(ev_h: dict, ev_b: dict, nm_h: dict, nm_b: dict) -> dict:
    c1 = ev_h["found"] - ev_b["found"] >= EVENT_GAIN - 1e-12
    c2 = ev_h["wrong"] <= WRONG_MAX + 1e-12
    c3 = nm_h["found"] >= nm_b["found"] - NORMAL_DROP - 1e-12
    return {"event_found_gain": ev_h["found"] - ev_b["found"], "event_found_gain_ok": bool(c1),
            "event_wrong": ev_h["wrong"], "event_wrong_ok": bool(c2),
            "normal_found_drop": nm_b["found"] - nm_h["found"], "normal_found_ok": bool(c3),
            "hidden_better": bool(c1 and c2 and c3)}


# ================================================================================ data


def scored_rows(events: dict, patches_dir: pathlib.Path, layers: Sequence[int]):
    """Per scored request: metadata + similarity maps ``sims[method] [L or 1, 256]``. Yields one run at a time."""
    for ri, run in enumerate(events["runs"]):
        rows = {x["request"]: x for x in run["rows"]}
        cand = [x for x in run["rows"] if x.get("sampled") and x["ref"] >= 0 and x.get("geom_index") is not None]
        f = patches_dir / f"{safe_name(run['label'])}.npz"
        H, pos, lay = None, {}, None
        if f.exists():
            with np.load(f) as z:
                lay = [int(v) for v in z["layers"]]
                li = [lay.index(l) for l in layers if l in lay]
                H = np.asarray(z["patches"][:, li], np.float32)  # [R, L, 256, D]
                pos = {int(r): i for i, r in enumerate(z["request"])}
        out = []
        for x in cand:
            ref = rows[x["ref"]]
            item = {"run": ri, "label": run["label"], "split": run["split"], "scene": run["scene"], "row": x,
                    "has_hidden": H is not None and x["request"] in pos and ref["request"] in pos}
            sims = {}
            if item["has_hidden"]:
                app = np.asarray(ref.get("appearance_patches") or [], np.int64)
                Hj, Hr = H[pos[x["request"]]], H[pos[ref["request"]]]
                if len(app):
                    sims["hidden"] = np.stack([cosine_map(Hj[k], Hr[k, app].mean(0)) for k in range(Hj.shape[0])])
                    Hjc = Hj - Hj.mean(axis=1, keepdims=True)
                    Hrc = Hr - Hr.mean(axis=1, keepdims=True)
                    sims["centred"] = np.stack([cosine_map(Hjc[k], Hrc[k, app].mean(0)) for k in range(Hj.shape[0])])
            item["sims"] = sims
            out.append(item)
        yield out, (lay if lay is not None else None)


def evaluate_setting(items, geom, method: str, li: int, tau: float) -> list[dict]:
    return evaluate_layer(items, geom, method, li, [tau])[tau]


def evaluate_layer(items, geom, method: str, li: int, taus: Sequence[float]) -> dict[float, list[dict]]:
    """``{tau: [score per item]}``. The region depends on τ only through the set {s >= τ}: results are cached by the
    number of patches at or above τ (<= 257 distinct sets per map)."""
    out = {t: [] for t in taus}
    for it in items:
        x = it["row"]
        g = x["geom_index"]
        s = it["sims"].get(method) if method != "attention" else it.get("att")
        if s is None:
            r0 = score(None, NONE, x["vis_c"])
            for t in taus:
                out[t].append(r0)
            continue
        sm = np.asarray(s[li] if np.ndim(s) == 2 else s, float)
        srt = np.sort(sm)
        memo = {}
        for t in taus:
            k = len(srt) - int(np.searchsorted(srt, t, side="left"))  # patches with s >= t
            if k not in memo:
                reg = largest_component(sm, t)
                c, maj, _ = centre_of(reg, geom["N"][g], geom["S"][g], geom["C"][g])
                memo[k] = score(c, maj, x["vis_c"])
            out[t].append(memo[k])
    return out


def baseline(items, which: str = "centroid") -> list[dict]:
    out = []
    for it in items:
        x = it["row"]
        c = x.get("centroid") if which == "centroid" else x.get("tp_centroid")
        maj = majority_class(x["class_counts_8mm"]) if x.get("class_counts_8mm") else NONE
        out.append(score(c, maj, x["vis_c"]))
    return out


def split_rates(items, res, split: str) -> tuple[dict, dict]:
    ev = [r for it, r in zip(items, res) if it["split"] == split and it["row"]["event"] in EVENT_TYPES]
    nm = [r for it, r in zip(items, res) if it["split"] == split and it["row"]["event"] == "normal"]
    return rates(ev), rates(nm)


def run_track(out_dir: pathlib.Path, events_path: pathlib.Path, patches_dir: pathlib.Path,
              layers: Sequence[int] = LAYERS, taus: Sequence[float] = TAU_GRID, focus: Optional[str] = FOCUS_LABEL,
              focus_requests: Optional[Sequence[int]] = None, fixed: Optional[tuple] = None) -> dict:
    events = json.loads(events_path.read_text())
    gpath = pathlib.Path(events.get("geom") or events_path.with_suffix(".geom.npz"))
    if not gpath.exists():
        gpath = events_path.with_suffix(".geom.npz")
    with np.load(gpath) as z:
        geom = {k: np.asarray(z[k]) for k in ("N", "S", "C", "attention")}
    items, focus_items, n_cand, n_no_hidden, n_no_vis = [], [], 0, 0, 0
    want = set(focus_requests) if focus_requests is not None else None
    stored_layers = None
    for run_items, lay in scored_rows(events, patches_dir, layers):
        stored_layers = stored_layers or lay
        for it in run_items:
            n_cand += 1
            if it["has_hidden"]:
                a = geom["attention"][it["row"]["geom_index"]].astype(np.float64)
                it["att"] = a / a.max() if np.isfinite(a).all() and a.max() > 0 else None
            if focus and it["label"] == focus and (want is None or it["row"]["request"] in want):
                focus_items.append(it)  # the focus table keeps requests without a visible GT surface too
            if not it["has_hidden"]:
                n_no_hidden += 1
                continue
            if not it["row"]["vis_n"]:
                n_no_vis += 1
                continue
            items.append(it)
    if not items:
        raise SystemExit("no scored request (patches missing?)")
    layers = [l for l in layers if stored_layers is None or l in stored_layers]
    base = baseline(items, "tp")              # verdict baseline = target-point mean (T75 definition)
    base_tp = baseline(items)                 # reported only: the recorded ball centre
    out: dict[str, Any] = {"step": "T79", "events": str(events_path), "patches_dir": str(patches_dir),
                           "layers": list(layers), "tau_grid": list(taus),
                           "criteria": {"found_mm": FOUND_MM, "wrong_mm": WRONG_MM, "event_gain": EVENT_GAIN,
                                        "wrong_max": WRONG_MAX, "normal_drop": NORMAL_DROP},
                           "counts": {"candidates": n_cand, "no_hidden": n_no_hidden, "no_visible_gt": n_no_vis,
                                      "scored": len(items),
                                      "scored_by_split_event": {
                                          s: {e: sum(1 for it in items if it["split"] == s and it["row"]["event"] == e)
                                              for e in (*EVENT_TYPES, "normal")} for s in ("train", "test")}}}
    out["baseline"] = {s: dict(zip(("event", "normal"), split_rates(items, base, s))) for s in ("train", "test")}
    out["baseline_ball_centre"] = {s: dict(zip(("event", "normal"), split_rates(items, base_tp, s))) for s in ("train", "test")}
    results = {}
    train_has_rows = any(it["split"] == "train" and it["row"]["event"] in EVENT_TYPES for it in items) and any(
        it["split"] == "train" and it["row"]["event"] == "normal" for it in items)
    out["fixed_setting"] = None if fixed is None else {"layer": fixed[0], "tau": fixed[1],
                                                       "note": "smoke only, not the train selection"}
    for method in ("hidden", "centred", "attention"):
        table = []
        lis = list(enumerate(layers)) if method != "attention" else [(0, 10)]
        cache = {}
        for li, l in lis:
            by_tau = evaluate_layer(items, geom, method, li, taus)
            for tau in taus:
                res = by_tau[tau]
                cache[(l, tau)] = res
                ev, nm = split_rates(items, res, "train")
                tev, tnm = split_rates(items, res, "test")
                table.append({"layer": l, "tau": tau, "event": ev, "normal": nm, "test_event": tev, "test_normal": tnm})
        train_table = [{k: t[k] for k in ("layer", "tau", "event", "normal")} for t in table]
        if fixed is not None:  # smoke / diagnostics only: not the pre-registered train selection
            fl, ftau = (fixed[0] if method != "attention" else 10), float(fixed[1])
            sel = {**[t for t in train_table if t["layer"] == fl and abs(t["tau"] - ftau) < 1e-9][0],
                   "admissible": None, "n_admissible": None}
        elif not train_has_rows:
            raise SystemExit("no scored train-scene request: the selection needs train scenes "
                             "(use --fixed-layer/--fixed-tau for a smoke run only)")
        else:
            sel = select(train_table, out["baseline"]["train"]["normal"]["found"])
        res = cache[(sel["layer"], sel["tau"])]
        tev, tnm = split_rates(items, res, "test")
        m = {"selected": {k: sel[k] for k in ("layer", "tau", "admissible", "n_admissible")},
             "train": {"event": sel["event"], "normal": sel["normal"]}, "test": {"event": tev, "normal": tnm},
             "verdict": verdict(tev, out["baseline"]["test"]["event"], tnm, out["baseline"]["test"]["normal"])}
        if method == "hidden" and train_has_rows:
            curve = []
            for l in layers:
                sl = select([t for t in train_table if t["layer"] == l], out["baseline"]["train"]["normal"]["found"])
                e2, n2 = split_rates(items, cache[(l, sl["tau"])], "test")
                curve.append({"layer": l, "tau": sl["tau"], "admissible": sl["admissible"], "train_event": sl["event"],
                              "train_normal": sl["normal"], "test_event": e2, "test_normal": n2})
            m["layer_curve"] = curve
        m["table"] = table
        out[method] = m
        results[method] = res
    out["verdict"] = out["hidden"]["verdict"]
    out["note"] = ("verdict = hidden (raw cosine) at the train-selected (layer, tau); centred / attention are "
                   "reference rows (no verdict)")
    per = []
    li_sel = layers.index(out["hidden"]["selected"]["layer"])
    for k, it in enumerate(items):
        x = it["row"]
        per.append({"label": it["label"], "split": it["split"], "scene": it["scene"], "request": x["request"],
                    "ref": x["ref"], "event": x["event"], "state": x["state"], "moved_mm": x["moved_mm"],
                    "vis_n": x["vis_n"], "baseline": base[k], "baseline_ball_centre": base_tp[k],
                    "hidden": results["hidden"][k], "centred": results["centred"][k], "attention": results["attention"][k]})
    out["per_request"] = per
    # ---- focus run (T71 R0 regression 211, requests 15–74): table + maps for the real-scene figure. Rows without a
    # head-visible GT surface are kept (found / wrong undefined -> None); the distance to the GT apple centre is
    # given for every row.
    if focus:
        fr = []
        tau = out["hidden"]["selected"]["tau"]
        for it in focus_items:
            x = it["row"]
            g = x["geom_index"]
            row = {"request": x["request"], "ref": x["ref"], "event": x["event"], "state": x["state"],
                   "moved_mm": x["moved_mm"], "vis_n": x["vis_n"], "apple_c": x["apple_c"], "vis_c": x["vis_c"],
                   "baseline_centroid": x.get("tp_centroid"), "baseline_ball_centre": x.get("centroid"),
                   "has_hidden": it["has_hidden"]}
            centres = {"baseline": (x.get("tp_centroid"),   # target-point mean, as the verdict baseline
                                    majority_class(x["class_counts_8mm"]) if x.get("class_counts_8mm") else NONE)}
            if it["has_hidden"] and "hidden" in it["sims"]:
                sm = it["sims"]["hidden"][li_sel]
                reg = largest_component(sm, tau)
                c, maj, n = centre_of(reg, geom["N"][g], geom["S"][g], geom["C"][g])
                row.update({"hidden_centre": None if c is None else c.tolist(), "hidden_region": reg.tolist(),
                            "hidden_majority": CLASSES[maj], "hidden_n_points": n, "sim_map": np.round(sm, 4).tolist()})
                centres["hidden"] = (c, maj)
            if it.get("att") is not None:
                at_sel = out["attention"]["selected"]["tau"]
                c, maj, _ = centre_of(largest_component(it["att"], at_sel), geom["N"][g], geom["S"][g], geom["C"][g])
                centres["attention"] = (c, maj)
            for name, (c, maj) in centres.items():
                row[f"{name}_to_apple_centre_mm"] = (None if c is None else round(float(
                    np.linalg.norm(np.asarray(c, float) - np.asarray(x["apple_c"], float)) * 1e3), 2))
                row[name] = score(c, maj, x["vis_c"]) if x["vis_n"] else None
            fr.append(row)
        fr.sort(key=lambda r: r["request"])
        out["focus"] = {"label": focus, "layer": out["hidden"]["selected"]["layer"], "tau": tau,
                        "requests": sorted(want) if want is not None else None, "rows": fr,
                        "rates_scored": {m: rates([r[m] for r in fr if r.get(m) is not None])
                                         for m in ("baseline", "hidden", "attention")},
                        "n_rows": len(fr), "n_without_visible_gt": sum(1 for r in fr if not r["vis_n"])}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "t79_track.json").write_text(json.dumps(out, ensure_ascii=False, indent=0, default=float))
    return out


def parse_range(spec: Optional[str]) -> Optional[list[int]]:
    if not spec:
        return None
    if ":" in spec:
        return list(range(*[int(p) for p in spec.split(":")]))
    return [int(p) for p in spec.split(",")]


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--events", required=True)
    ap.add_argument("--patches-dir", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    ap.add_argument("--focus", default=FOCUS_LABEL)
    ap.add_argument("--focus-requests", default="15:75", help="lo:hi or comma list (default 15:75 = requests 15–74)")
    ap.add_argument("--fixed-layer", type=int, default=None, help="smoke only: skip the train selection")
    ap.add_argument("--fixed-tau", type=float, default=None, help="smoke only (with --fixed-layer)")
    a = ap.parse_args(argv)
    fixed = None if a.fixed_layer is None else (a.fixed_layer, a.fixed_tau)
    out = run_track(pathlib.Path(a.out_dir), pathlib.Path(a.events), pathlib.Path(a.patches_dir), a.layers,
                    TAU_GRID, a.focus, parse_range(a.focus_requests), fixed)
    brief = {k: out[k] for k in ("counts", "baseline", "verdict")}
    brief["hidden"] = {k: out["hidden"][k] for k in ("selected", "train", "test")}
    brief["attention"] = {k: out["attention"][k] for k in ("selected", "test")}
    print(json.dumps(brief, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
