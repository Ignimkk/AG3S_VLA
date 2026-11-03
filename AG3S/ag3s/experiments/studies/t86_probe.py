"""T86 — ATS target finding: hidden image-patch probe versus noun attention (`handoff/T86.task.md` §1, §3; fixed before
results).

Inputs (produced elsewhere, read only):
  SAMPLES  `t86_labels sample` json (+ its ``.labels.npz``): kept requests per run with split / scene, class shares of
           every camera token (``frac``), counterfactual fruit / prompt.
  PATCHES  `pi05_hidden --pool patches --cameras ... --noun-attention --prompt-override` npz per run (T86 layout):
           hidden of the 3 cameras' 256 tokens at layers {4, 8, 12, 16} for the original prompt (``patches``) and the
           counterfactual prompt (``cf_patches``), and the noun-attention row (L10H7) of each forward.

Maps compared (one 16 x 16 map per camera and forward):
  A  noun attention  the forward's own ``noun_target:L10H7`` row (the ATS signal today), raw softmax mass.
  B  hidden probe    per camera, layer l: standardise + logistic regression on the layer-l hidden of a token ->
                     p(target); trained on **train scenes only**: every forward (original prompt = target fruit of the
                     recorded prompt, counterfactual prompt = its fruit) whose camera shows >= 1 target token; positives =
                     the target tokens, negatives = a seeded sample of ``--neg-per-image`` other tokens.
  B' centred         the same with each forward's 256-token mean subtracted from the hidden first (the common component
                     of T79).
  Layer l and the regularisation C (``C_GRID`` = T77's) are chosen by **GroupKFold over the train scenes** only:
  held-out object-selection accuracy over original + counterfactual forwards, ties -> larger mean AUROC, then smaller C
  (and lower layer). Test scenes are read only after those choices are fixed.

Metrics per forward and camera, over forwards whose camera has >= 1 target token for the prompt's fruit (A and B on the
same set, i.e. forwards whose noun was found):
  top1        the highest token is a target token                                                   (task §3.1)
  selected    the largest 4-connected component of the top-k tokens (k = number of target tokens; ties: larger summed
              score) has the target as its strictly most frequent label (target / other / background)    (§3.2)
  auroc       target tokens against the rest                                                              (§3.3)
  follow      `selected` on the **counterfactual-prompt** forwards (target = the changed fruit): the component moved to
              the changed fruit                                                                           (§3.4)
Labels per token and fruit: `t86_labels.patch_labels` (>= t target / >= t other objects / background; t = 0.20, the same
value for target and other and for all three cameras, T86.task.md §2.1; read from the samples json).

Judged numbers are **scene means** (T86.task.md §2.1): every test scene (T77 ``scene`` name, e.g. 회귀211, 허들211) gets
the same weight; a scene with fewer than 20 requests that have a target token (per kind: original / counterfactual) is
left out of that mean and marked in the tables. The pooled (request-weighted) rates stay in the json as reference.

Verdict (test scenes, head camera main; wrist cameras reported, same threshold): B (or B') is better iff
  selected(B) >= selected(A) + 0.10  and  follow(B) >= 0.80  and  follow(B) >= follow(A),
selected = on the original-prompt forwards. Below that: numbers are reported and nothing else follows (serving
connection is the user's decision).

usage (.venv-ag3s, CPU; one process per camera, <= 3 workers):
  PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=4 .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.t86_probe OUT_DIR \\
      --samples t86_samples.json --patches-dir hidden/patches [--cameras cam_high ...] [--jobs 3]
Writes ``OUT_DIR/t86_probe.json`` (counts, cv tables, selection, test metrics of A / B / B' per camera, layer curve,
verdict) and ``OUT_DIR/test_maps.npz`` (A / B / B' maps and labels of every test forward, for figures).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.sources.policy_record import POLICY_CAMERA_NAMES
from AG3S.ag3s.experiments.studies.t86_labels import BG, OTHER, TARGET, fruit_of_prompt, patch_labels

GRID = 16
N_TOK = GRID * GRID
LAYERS = (4, 8, 12, 16)
VARIANTS = ("raw", "centred")
#: the T77 / T64 grid (`t64_phase6.C_GRID`) -- not imported to keep this tool free of the T64 module's data paths
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
PRIMARY_CAMERA = "cam_high"
NOUN_CELL_LABEL = "noun_target:L10H7"
SEL_GAIN = 0.10
FOLLOW_MIN = 0.80
#: a scene enters the scene mean only with >= this many requests that have a target token (T86.task.md §2.1)
MIN_SCENE_REQUESTS = 20
NEG_PER_IMAGE = 64
N_FOLDS = 4
MAX_ITER = 300


# ================================================================================ pure parts (tested)


def auroc(score: np.ndarray, positive: np.ndarray) -> Optional[float]:
    """Rank AUROC (ties averaged) of `score` for the boolean `positive`; None if either class is empty."""
    from scipy.stats import rankdata

    s = np.asarray(score, np.float64)
    pos = np.asarray(positive, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def top_component(score: np.ndarray, k: int) -> np.ndarray:
    """Token ids of the largest 4-connected component of the top-`k` tokens by score (stable order: higher score, then
    lower id); ties between components: larger summed score, then the lower first id."""
    from AG3S.ag3s.experiments.studies.t79_track import largest_component

    s = np.asarray(score, np.float64).reshape(-1)
    order = np.argsort(-s, kind="stable")[: max(1, int(k))]
    masked = np.full_like(s, -1e30)
    masked[order] = s[order]
    return largest_component(masked, -1e29)


def majority_label(labels: np.ndarray, tokens: np.ndarray) -> int:
    """Strictly most frequent label among `tokens` (BG / OTHER / TARGET); a tie -> -1 (no majority)."""
    c = np.bincount(np.asarray(labels, np.int64)[np.asarray(tokens, np.int64)], minlength=3)
    top = int(np.argmax(c))
    return top if int((c == c[top]).sum()) == 1 else -1


def map_metrics(score: np.ndarray, labels: np.ndarray) -> Optional[dict]:
    """top1 / selected / auroc of one map against one label map; None if there is no target token."""
    labels = np.asarray(labels)
    tgt = labels == TARGET
    k = int(tgt.sum())
    if k == 0:
        return None
    s = np.asarray(score, np.float64).reshape(-1)
    comp = top_component(s, k)
    return {"k": k, "top1": bool(tgt[int(np.argsort(-s, kind="stable")[0])]),
            "selected": bool(majority_label(labels, comp) == TARGET), "auroc": auroc(s, tgt),
            "component": int(len(comp)), "comp_target_share": float(tgt[comp].mean()) if len(comp) else 0.0}


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(max(0.0, c - h)), 4), round(float(min(1.0, c + h)), 4)]


def rates(rows: Sequence[Optional[dict]]) -> dict:
    """Mean top1 / selected / AUROC of metric rows (None rows are dropped, AUROC over rows that have one)."""
    rows = [r for r in rows if r is not None]
    n = len(rows)
    au = [r["auroc"] for r in rows if r["auroc"] is not None]
    sel = sum(r["selected"] for r in rows)
    return {"n": n, "top1": (sum(r["top1"] for r in rows) / n if n else None),
            "selected": (sel / n if n else None), "selected_ci95": wilson(sel, n),
            "auroc": (float(np.mean(au)) if au else None)}


def centred(h: np.ndarray) -> np.ndarray:
    """Subtract the mean over the token axis (-2) -- B' (each forward's common component)."""
    h = np.asarray(h, np.float32)
    return h - h.mean(axis=-2, keepdims=True)


def cv_pick(table: dict) -> tuple[float, dict]:
    """``{C: (selected, auroc)}`` -> ``(best C, its record)``: highest selected, then AUROC, then smaller C."""
    best = max(table, key=lambda c: (round(table[c][0], 12), round(table[c][1], 12), -c))
    return float(best), {"selected": table[best][0], "auroc": table[best][1]}


def pick_layer(per_layer: dict) -> int:
    """``{layer: (selected, auroc)}`` -> best layer: highest selected, then AUROC, then lower layer."""
    return int(max(per_layer, key=lambda l: (round(per_layer[l][0], 12), round(per_layer[l][1], 12), -l)))


def verdict(a: dict, b: dict) -> dict:
    """Task §3: B better than A. `a`, `b` = {"selected": orig-prompt selection rate, "follow": cf rate}."""
    sel = b["selected"] is not None and a["selected"] is not None and b["selected"] >= a["selected"] + SEL_GAIN - 1e-12
    fol_min = b["follow"] is not None and b["follow"] >= FOLLOW_MIN - 1e-12
    fol_ge = b["follow"] is not None and a["follow"] is not None and b["follow"] >= a["follow"] - 1e-12
    gain = None if b["selected"] is None or a["selected"] is None else b["selected"] - a["selected"]
    return {"selected_gain": gain, "cond_selected_ge_A_plus_10pp": bool(sel),
            "cond_follow_ge_0p8": bool(fol_min), "cond_follow_ge_A": bool(fol_ge),
            "better": bool(sel and fol_min and fol_ge)}


def group_folds(groups: Sequence, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """`t50_core.group_kfold` over scene groups (deterministic, lightest-fold assignment); folds = min(n, #groups)."""
    from AG3S.ag3s.experiments.studies.t50_core import group_kfold

    g = np.asarray(groups)
    return group_kfold(g, min(int(n_folds), len(np.unique(g))))


def training_rows(h: np.ndarray, labels: np.ndarray, idx: np.ndarray, neg: int, rng: np.random.Generator):
    """Positives (all target tokens) and ``neg`` sampled non-target tokens of each forward in `idx`.
    `h` [n, 256, D], `labels` [n, 256] -> ``(X [m, D] float32, y [m])``."""
    X, y = [], []
    for i in idx:
        t = np.flatnonzero(labels[i] == TARGET)
        o = np.flatnonzero(labels[i] != TARGET)
        o = rng.choice(o, min(int(neg), len(o)), replace=False) if len(o) else o
        sel = np.concatenate([t, o])
        X.append(np.asarray(h[i][sel], np.float32))
        y.append(np.concatenate([np.ones(len(t), np.int64), np.zeros(len(o), np.int64)]))
    return np.concatenate(X), np.concatenate(y)


def score_tokens(model, h: np.ndarray) -> np.ndarray:
    """p(target) of every token of one forward ``h [256, D]``."""
    return model.proba(np.asarray(h, np.float64))[:, 1]


# ================================================================================ data


def safe_name(label: str) -> str:
    from AG3S.ag3s.experiments.sources.pi05_hidden import safe_name as _sn

    return _sn(label)


class Forward:
    """One forward (original or counterfactual prompt) of one kept request."""

    __slots__ = ("run", "label", "split", "scene", "scene_name", "request", "kind", "fruit", "prompt", "noun_ok", "att", "labels",
                 "file", "pos")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def load_samples(path: str) -> tuple[dict, np.lib.npyio.NpzFile]:
    d = json.loads(pathlib.Path(path).read_text())
    z = np.load(d["labels_npz"], allow_pickle=False)
    return d, z


def forwards_of_file(d: dict, z, run_index: int, pz) -> list[Forward]:
    """Forward records of one patches npz (originals first, then counterfactuals), labels per camera."""
    run = d["runs"][run_index]
    rows = {int(r): i for i, r in enumerate(np.asarray(z["request"])[np.asarray(z["run"]) == run_index])}
    first = int(np.flatnonzero(np.asarray(z["run"]) == run_index)[0]) if rows else 0
    cams = [str(c) for c in pz["camera_names"]]
    cam_idx = [POLICY_CAMERA_NAMES.index(c) for c in cams]
    out = []
    reqs = [int(r) for r in pz["request"]]
    cf_req = [int(x) for x in pz["cf_request"]]
    cf_prompt = [str(x) for x in pz["cf_prompt"]]
    cf_fruit = {s["request"]: s["cf_fruit"] for s in run["samples"]}
    for kind, n in (("orig", len(reqs)), ("cf", len(cf_req))):
        for p in range(n):
            rpos = p if kind == "orig" else cf_req[p]
            req = reqs[rpos]
            fi = first + rows[req]
            frac = np.asarray(z["frac"][fi], np.float32)  # [3, 256, 8]
            if kind == "orig":
                prompt, fruit = str(pz["prompt"][rpos]), None
                fruit = fruit_of_prompt(prompt)
                att, ok = np.asarray(pz["noun_att"][p], np.float32), bool(pz["noun_ok"][p])
            else:
                prompt, fruit = cf_prompt[p], cf_fruit[req]
                att, ok = np.asarray(pz["cf_noun_att"][p], np.float32), bool(pz["cf_noun_ok"][p])
            if fruit is None:
                raise ValueError(f"{run['label']} request {req}: counterfactual forward without a recorded fruit")
            labels = np.stack([patch_labels(frac[c], fruit, d["params"]["target_frac"]) for c in cam_idx])
            out.append(Forward(run=run_index, label=run["label"], split=run["split"], scene=run["scene_number"],
                               scene_name=run.get("scene") or run["scene_number"], request=req, kind=kind, fruit=fruit, prompt=prompt, noun_ok=ok, att=att,
                               labels=labels, file=None, pos=(kind, p)))
    return out


def camera_slice(att: np.ndarray, cam: str) -> np.ndarray:
    from AG3S.ag3s.experiments.sources.policy_record import CAMERA_BINDINGS

    lo, hi = CAMERA_BINDINGS[cam][2]
    return np.asarray(att, np.float32)[lo:hi]


class TrainSet:
    """All train forwards of one camera: hidden per layer (float16 [n, 256, D]), labels, scenes."""

    def __init__(self):
        self.fw: list[Forward] = []
        self.h: dict[int, list] = collections.defaultdict(list)

    def add_file(self, fws: Sequence[Forward], pz, cam: str, layers: Sequence[int]) -> None:
        cams = [str(c) for c in pz["camera_names"]]
        ci = cams.index(cam)
        lays = [int(x) for x in pz["layers"]]
        for f in fws:
            f.labels = f.labels[ci]
        P, CF = pz["patches"], pz["cf_patches"]  # one read of each (an NpzFile re-reads a member on every access)
        for l in layers:
            li = lays.index(l)
            blocks = [P[:, ci, li]] + ([CF[:, ci, li]] if len(CF) else [])
            self.h[l].append(np.concatenate(blocks))
        self.fw.extend(fws)

    def arrays(self, layer: int):
        return np.concatenate(self.h[layer]) if self.h[layer] else np.zeros((0, N_TOK, 0), np.float16)


def eligible(fw: Sequence[Forward]) -> np.ndarray:
    """Forwards with >= 1 target token in this camera and a found noun (the common evaluation set)."""
    return np.asarray([bool((f.labels == TARGET).any()) and f.noun_ok for f in fw], bool)


# ================================================================================ selection on train, test on test


def prepare(h: np.ndarray, variant: str) -> np.ndarray:
    return h if variant == "raw" else centred(h)


def cv_scores(ts: TrainSet, layer: int, variant: str, c_grid: Sequence[float], neg: int, n_folds: int, seed: int,
              max_iter: int, log=print) -> dict:
    """Held-out (selected, auroc) per C, GroupKFold over train scenes; ``{C: (selected, auroc)}`` + fit info."""
    from AG3S.ag3s.experiments.studies.t50_core import fit_logreg

    el = np.flatnonzero(eligible(ts.fw))
    H = ts.arrays(layer)
    labels = np.stack([ts.fw[i].labels for i in el])
    scenes = np.asarray([ts.fw[i].scene for i in el])
    folds = group_folds(scenes, n_folds)
    table = {float(c): [[], []] for c in c_grid}
    rng = np.random.default_rng([seed, layer, 0 if variant == "raw" else 1])
    for fi, (tr, te) in enumerate(folds):
        Htr = prepare(H[el[tr]], variant)
        X, y = training_rows(Htr, labels[tr], np.arange(len(tr)), neg, rng)
        Hte = prepare(H[el[te]], variant)
        for c in c_grid:
            m = fit_logreg(X, y, float(c), n_classes=2, max_iter=max_iter)
            rows = [map_metrics(score_tokens(m, Hte[j]), labels[te[j]]) for j in range(len(te))]
            r = rates(rows)
            if r["n"]:
                table[float(c)][0].append((r["selected"], r["n"]))
                table[float(c)][1].append(r["auroc"])
        log(f"    cv {variant} L{layer} fold {fi + 1}/{len(folds)} done", flush=True)
    out = {}
    for c, (sel, au) in table.items():
        n = sum(k for _, k in sel)
        out[c] = (float(sum(s * k for s, k in sel) / max(n, 1)), float(np.mean([a for a in au if a is not None] or [0.0])))
    return out


def fit_final(ts: TrainSet, layer: int, variant: str, c: float, neg: int, seed: int, max_iter: int):
    from AG3S.ag3s.experiments.studies.t50_core import fit_logreg

    el = np.flatnonzero(eligible(ts.fw))
    H = prepare(ts.arrays(layer)[el], variant)
    labels = np.stack([ts.fw[i].labels for i in el])
    rng = np.random.default_rng([seed, layer, 2, 0 if variant == "raw" else 1])
    X, y = training_rows(H, labels, np.arange(len(el)), neg, rng)
    return fit_logreg(X, y, float(c), n_classes=2, max_iter=max_iter), int(len(y))


def scene_rates(rows: Sequence[dict], min_n: Optional[int] = None) -> dict:
    """Per-scene rates and the **scene mean** (every scene the same weight) of rows ``{kind, scene, m}``.

    Per kind (``orig`` = original prompt, ``cf`` = counterfactual prompt) and scene: `rates` + ``included`` (>= `min_n`
    forwards that have a target token; the rows only hold such forwards). The mean is over included scenes only; the
    scenes left out (they have rows but fewer than `min_n`) are listed under ``excluded`` and stay in ``per_scene``."""
    min_n = MIN_SCENE_REQUESTS if min_n is None else min_n
    scenes = sorted({r["scene"] for r in rows})
    out: dict[str, Any] = {"min_requests_per_scene": int(min_n), "per_scene": {}, "mean": {}}
    for kind in ("orig", "cf"):
        per = {}
        for sc in scenes:
            r = rates([x["m"] for x in rows if x["kind"] == kind and x["scene"] == sc])
            r["included"] = bool(r["n"] >= min_n)
            per[sc] = r
        inc = [v for v in per.values() if v["included"]]
        mean: dict[str, Any] = {"n_scenes": len(inc), "scenes": [k for k, v in per.items() if v["included"]],
                                "excluded": [k for k, v in per.items() if v["n"] and not v["included"]]}
        for key in ("top1", "selected", "auroc"):
            vals = [v[key] for v in inc if v[key] is not None]
            mean[key] = float(np.mean(vals)) if vals else None
        out["per_scene"][kind], out["mean"][kind] = per, mean
    return out


def split_rates(rows: Sequence[dict]) -> dict:
    """Aggregate per-forward results ``{kind, scene, m}`` of one method: pooled (request-weighted) orig / cf / both
    rates, and `scene_rates` (the judged numbers)."""
    return {"orig": rates([r["m"] for r in rows if r["kind"] == "orig"]),
            "cf": rates([r["m"] for r in rows if r["kind"] == "cf"]),
            "all": rates([r["m"] for r in rows]), "scene": scene_rates(rows)}


def run_camera(cam: str, samples_json: str, patches_dir: str, layers: Sequence[int], c_grid: Sequence[float],
               neg: int, n_folds: int, seed: int, max_iter: int, out_dir: pathlib.Path, log=print) -> dict:
    d, z = load_samples(samples_json)
    pdir = pathlib.Path(patches_dir)
    ts, test_files, missing = TrainSet(), [], collections.Counter()
    for ri, run in enumerate(d["runs"]):
        if not run["samples"]:
            continue
        f = pdir / f"{safe_name(run['label'])}.npz"
        if not f.exists():
            missing[run["split"]] += 1
            continue
        if run["split"] == "train":
            with np.load(f) as pz:
                ts.add_file(forwards_of_file(d, z, ri, pz), pz, cam, layers)
        else:
            test_files.append((ri, f))
    train_scenes = {f.scene for f in ts.fw}
    test_scenes = {d["runs"][ri]["scene_number"] for ri, _ in test_files}
    if train_scenes & test_scenes:
        raise SystemExit(f"scene numbers on both sides: {sorted(train_scenes & test_scenes)}")
    el = eligible(ts.fw)
    log(f"[{cam}] train forwards {len(ts.fw)} ({int(el.sum())} eligible), train scenes {sorted(train_scenes)}, "
        f"test files {len(test_files)}, missing {dict(missing)}", flush=True)

    # ---- selection on the train scenes only
    cv, chosen = {}, {}
    for variant in VARIANTS:
        per_layer, per_c = {}, {}
        for l in layers:
            tab = cv_scores(ts, l, variant, c_grid, neg, n_folds, seed, max_iter, log)
            c, rec = cv_pick(tab)
            per_layer[l], per_c[l] = (rec["selected"], rec["auroc"]), {"C": c, "table": {str(k): v for k, v in tab.items()}}
        best = pick_layer(per_layer)
        cv[variant] = per_c
        chosen[variant] = {"layer": best, "C": per_c[best]["C"], "cv_selected": per_layer[best][0],
                           "cv_auroc": per_layer[best][1]}
        log(f"[{cam}] {variant}: chosen layer {best}, C {chosen[variant]['C']}, cv selected "
            f"{chosen[variant]['cv_selected']:.3f}", flush=True)
    models = {}
    for variant in VARIANTS:
        for l in layers:
            models[(variant, l)], n_rows = fit_final(ts, l, variant, cv[variant][l]["C"], neg, seed, max_iter)
    train_counts = {"forwards": len(ts.fw), "eligible": int(el.sum()),
                    "orig": int(sum(1 for f, e in zip(ts.fw, el) if e and f.kind == "orig")),
                    "cf": int(sum(1 for f, e in zip(ts.fw, el) if e and f.kind == "cf"))}
    del ts

    # ---- test (fixed choices; scenes never seen above)
    res_a: list = []
    res_m: dict = {k: [] for k in models}
    maps: dict[str, list] = {"A": [], "B": [], "Bc": [], "labels": [], "meta": []}
    for ri, f in test_files:
        with np.load(f) as pz:
            fws = forwards_of_file(d, z, ri, pz)
            cams = [str(c) for c in pz["camera_names"]]
            ci = cams.index(cam)
            lays = [int(x) for x in pz["layers"]]
            P, CF = pz["patches"], pz["cf_patches"]
            for fw in fws:
                kind, p = fw.pos
                lab = fw.labels[ci]
                if not (lab == TARGET).any() or not fw.noun_ok:
                    continue
                a_map = camera_slice(fw.att, cam)
                res_a.append({"kind": kind, "scene": fw.scene_name, "m": map_metrics(a_map, lab)})
                maps_here = {}
                for variant in VARIANTS:
                    for l in layers:
                        h = np.asarray((P if kind == "orig" else CF)[p, ci, lays.index(l)], np.float32)
                        h = h if variant == "raw" else centred(h)
                        s = score_tokens(models[(variant, l)], h)
                        res_m[(variant, l)].append({"kind": kind, "scene": fw.scene_name, "m": map_metrics(s, lab)})
                        if l == chosen[variant]["layer"]:
                            maps_here[variant] = s
                maps["A"].append(a_map)
                maps["B"].append(maps_here["raw"])
                maps["Bc"].append(maps_here["centred"])
                maps["labels"].append(lab)
                maps["meta"].append([ri, fw.request, 0 if kind == "orig" else 1])
    base = split_rates(res_a)
    test = {"A": base}
    for variant in VARIANTS:
        for l in layers:
            test[f"{variant}_L{l}"] = split_rates(res_m[(variant, l)])
    sel_key = {v: f"{v}_L{chosen[v]['layer']}" for v in VARIANTS}
    verdicts = {}
    for v in VARIANTS:
        bs = test[sel_key[v]]
        a = {"selected": base["scene"]["mean"]["orig"]["selected"], "follow": base["scene"]["mean"]["cf"]["selected"]}
        b = {"selected": bs["scene"]["mean"]["orig"]["selected"], "follow": bs["scene"]["mean"]["cf"]["selected"]}
        pooled = {"A": {"selected": base["orig"]["selected"], "follow": base["cf"]["selected"]},
                  "B": {"selected": bs["orig"]["selected"], "follow": bs["cf"]["selected"]}}
        verdicts[v] = {"basis": "scene mean (every scene the same weight; scenes with < "
                                f"{MIN_SCENE_REQUESTS} target-token requests left out)", "A": a, "B": b,
                       **verdict(a, b), "selected_setting": chosen[v], "pooled_reference": pooled,
                       "scenes_orig": bs["scene"]["mean"]["orig"]["scenes"],
                       "scenes_cf": bs["scene"]["mean"]["cf"]["scenes"],
                       "scenes_excluded_orig": bs["scene"]["mean"]["orig"]["excluded"],
                       "scenes_excluded_cf": bs["scene"]["mean"]["cf"]["excluded"]}
    np.savez_compressed(out_dir / f"test_maps_{cam}.npz", A=np.asarray(maps["A"], np.float32),
                        B=np.asarray(maps["B"], np.float32), Bc=np.asarray(maps["Bc"], np.float32),
                        labels=np.asarray(maps["labels"], np.int8), meta=np.asarray(maps["meta"], np.int64),
                        meta_columns=np.asarray(["run_index", "request", "kind_is_cf"]))
    return {"camera": cam, "train": train_counts, "train_scenes": sorted(train_scenes), "test_scenes": sorted(test_scenes),
            "missing_files": dict(missing), "cv": cv, "selected": chosen, "test": test,
            "chance": {"top1": float(np.mean([r["m"]["k"] for r in res_a if r["m"]]) / N_TOK) if res_a else None},
            "verdict": verdicts, "n_test_forwards": len(res_a)}


# --- T90 — begin: probe asset export (`--export NPZ`)
def selection_of(doc: dict, variant: str = "raw") -> dict[str, dict]:
    """``t86_probe.json`` → ``{camera: {"layer", "C"}}`` of the settings T86's train cross-validation chose (``cameras.*.
    selected.<variant>``)."""
    out = {}
    for cam, rec in doc["cameras"].items():
        sel = rec["selected"][variant]
        out[cam] = {"layer": int(sel["layer"]), "C": float(sel["C"]), "cv_selected": float(sel["cv_selected"]),
                    "cv_auroc": float(sel["cv_auroc"])}
    return out


def fit_camera_for_export(cam: str, samples_json: str, patches_dir: str, layer: int, c: float, neg: int, seed: int,
                          max_iter: int, variant: str = "raw", log=print):
    """Refit the chosen setting on **all train forwards** of one camera (the same rows, rng and solver as T86's
    ``fit_final`` — so the model is the one whose test numbers T86 reported) and return ``(LogReg, counts)``. Nothing of
    the test scenes is read."""
    d, z = load_samples(samples_json)
    pdir = pathlib.Path(patches_dir)
    ts, missing = TrainSet(), 0
    for ri, run in enumerate(d["runs"]):
        if not run["samples"] or run["split"] != "train":
            continue
        f = pdir / f"{safe_name(run['label'])}.npz"
        if not f.exists():
            missing += 1
            continue
        with np.load(f) as pz:
            ts.add_file(forwards_of_file(d, z, ri, pz), pz, cam, [layer])
    el = eligible(ts.fw)
    log(f"[{cam}] export: train forwards {len(ts.fw)} ({int(el.sum())} eligible), missing files {missing}; "
        f"{variant} L{layer} C {c}", flush=True)
    model, n_rows = fit_final(ts, layer, variant, c, neg, seed, max_iter)
    return model, {"forwards": len(ts.fw), "eligible": int(el.sum()), "rows": int(n_rows), "missing_files": missing,
                   "converged": bool(model.converged), "n_iter": int(model.n_iter)}


def export_asset(npz_path: str, samples_json: str, patches_dir: str, selection_json: str, *, neg: int = NEG_PER_IMAGE,
                 seed: int = 0, max_iter: int = MAX_ITER, variant: str = "raw", log=print) -> dict:
    """T90 §1.1 — write ``npz_path`` (+ ``.json``): per camera the standardiser (mean · scale), the 2-class coefficients and
    intercept of the setting T86's train cross-validation selected (`selection_of`), refit on the whole train set.
    The refit uses T86's seed / neg / max_iter (read them from ``t86_probe.json`` ``params``; the defaults are T86's)."""
    import hashlib
    import subprocess

    from AG3S.trajopt.hidden_probe import CAMERAS, save_asset

    doc = json.loads(pathlib.Path(selection_json).read_text())
    params = doc.get("params", {})
    neg, seed, max_iter = int(params.get("neg_per_image", neg)), int(params.get("seed", seed)), int(params.get("max_iter", max_iter))
    sel = selection_of(doc, variant)
    missing = [c for c in CAMERAS if c not in sel]
    if missing:
        raise SystemExit(f"{selection_json}: no selected setting for {missing}")
    models, counts = {}, {}
    for cam in CAMERAS:
        models[cam], counts[cam] = fit_camera_for_export(cam, samples_json, patches_dir, sel[cam]["layer"], sel[cam]["C"],
                                                         neg, seed, max_iter, variant, log)
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=str(pathlib.Path(__file__).parent),
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        head = ""
    layers = {c: sel[c]["layer"] for c in CAMERAS}
    out = save_asset(npz_path, models, layers, variant=variant,
                     meta={"neg_per_image": neg, "seed": seed, "max_iter": max_iter, "selection_json": str(selection_json),
                           "samples_json": str(samples_json), "patches_dir": str(patches_dir)})
    md5 = hashlib.md5(out.read_bytes()).hexdigest()
    side = {"step": "T90", "asset": str(out), "md5": md5, "variant": variant, "classes": ["other", "target"],
            "score": "p(target) = sigmoid(logit); logit = standardised hidden of one image token @ (coef[1]-coef[0]) + (b1-b0)",
            "source": {"samples": str(samples_json), "patches_dir": str(patches_dir), "selection": str(selection_json),
                       "selection_md5": hashlib.md5(pathlib.Path(selection_json).read_bytes()).hexdigest(), "git": head},
            "protocol": {"fit": "all train forwards of the camera (>=1 target token, noun found); positives = target tokens, "
                                "negatives = seeded sample of neg_per_image other tokens per forward (t86_probe.fit_final)",
                         "neg_per_image": neg, "seed": seed, "max_iter": max_iter},
            "cameras": {c: {"layer": sel[c]["layer"], "C": sel[c]["C"], "cv_selected": sel[c]["cv_selected"],
                            "cv_auroc": sel[c]["cv_auroc"], **counts[c]} for c in CAMERAS}}
    out.with_suffix(".json").write_text(json.dumps(side, ensure_ascii=False, indent=1))
    log(f"exported {out} md5 {md5}", flush=True)
    return side
# --- T90 — end


def _camera_job(a):
    cam, kw = a
    return run_camera(cam, **kw)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--patches-dir", required=True)
    ap.add_argument("--cameras", nargs="+", default=list(POLICY_CAMERA_NAMES), choices=POLICY_CAMERA_NAMES)
    ap.add_argument("--layers", nargs="+", type=int, default=list(LAYERS))
    ap.add_argument("--c-grid", nargs="+", type=float, default=list(C_GRID))
    ap.add_argument("--neg-per-image", type=int, default=NEG_PER_IMAGE)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1, help="one process per camera (<= 3)")
    # --- T90 — begin
    ap.add_argument("--export", default=None, metavar="NPZ",
                    help="T90 — instead of the cross-validation / test: refit the settings T86 selected (--selection, a "
                         "t86_probe.json) on all train forwards and write the serving asset NPZ (+ .json). Default off: "
                         "this tool runs as before")
    ap.add_argument("--selection", default=None, metavar="JSON", help="t86_probe.json whose cameras.*.selected.raw is exported")
    # --- T90 — end
    a = ap.parse_args(argv)
    # --- T90 — begin
    if a.export is not None:
        if not a.selection:
            raise SystemExit("--export needs --selection t86_probe.json")
        export_asset(a.export, a.samples, a.patches_dir, a.selection)
        return
    if a.selection:
        raise SystemExit("--selection is read by --export only")
    if a.out is None:
        raise SystemExit("OUT_DIR is required (except with --export)")
    # --- T90 — end
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(samples_json=a.samples, patches_dir=a.patches_dir, layers=a.layers, c_grid=a.c_grid,
              neg=a.neg_per_image, n_folds=a.folds, seed=a.seed, max_iter=a.max_iter, out_dir=out)
    jobs = [(c, kw) for c in a.cameras]
    n = max(1, min(int(a.jobs), 3, len(jobs)))
    if n == 1:
        res = [_camera_job(j) for j in jobs]
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(n) as pool:
            res = pool.map(_camera_job, jobs)
    summary = {r["camera"]: r for r in res}
    primary = summary.get(PRIMARY_CAMERA)
    doc = {"step": "T86", "samples": a.samples, "patches_dir": a.patches_dir, "definitions": __doc__,
           "params": {"layers": a.layers, "c_grid": a.c_grid, "neg_per_image": a.neg_per_image, "folds": a.folds,
                      "max_iter": a.max_iter, "seed": a.seed, "sel_gain": SEL_GAIN, "follow_min": FOLLOW_MIN,
                      "noun_cell": NOUN_CELL_LABEL},
           "cameras": summary,
           "verdict_head": None if primary is None else primary["verdict"],
           "better_than_noun_attention_head": None if primary is None
           else bool(any(v["better"] for v in primary["verdict"].values()))}
    (out / "t86_probe.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({"better_head": doc["better_than_noun_attention_head"],
                      "verdict_head": doc["verdict_head"]}, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
