"""T86 — ATS target finding by a hidden-patch probe: request sample, ground-truth patch labels, counterfactual fruit
(`handoff/T86.task.md` §2; fixed before results).

Built on the T77 spec (`t77_labels spec`: runs, request pairing **by order**, scene split test = {211, 954, 122, 119})
and the T77 labels (`gt.phase5`). For every AG3S closed-loop run it draws a seeded uniform sample of **pre-grasp**
requests (`gt.phase5` approach or grasp — held has not happened yet), renders the MuJoCo **segmentation** of the three
policy cameras at the VLA input geometry and keeps a request only if the head camera shows >= 1 target patch:

  render    the scene of that request (client qpos in the scene XML with the obstacle slots activated as the client did,
            `task_dynamic_map_eval.scene_truth`), pose = the kinematics the camera saw (one physics step back,
            `t48b_scene.render_qpos`; request 0 falls back to qpos itself when that matches the recorded RGB better),
            through `t48b_scene.PolicyRenderer` (299 x 224 render, the policy source size), segmentation mode: geom ids,
            visual geoms included (the same ``geomgroup`` the RGB uses).
  224 grid  299 columns -> 224 by nearest source column ``floor((u + 0.5) * 299 / 224)`` (a bilinear resize cannot blend
            ids; rows are 1:1). A policy pixel (u, v) belongs to head / wrist token ``patch_of_policy_pixel(u, v)``
            (`t79_events`: SigLIP 14 x 14 px patches, row-major, 16 x 16). The 224 image is the 4:3 render stretched, so
            the patch grid is exactly the ``40 x 30`` px grid of the 640 x 480 depth (`t79_events` docstring).
  classes   root body of the geom -> ``apple banana orange pear crate robot obstacle scene`` (``scene`` = table, office,
            shelf, world, nothing/sky). ``frac[camera, token, class]`` = share of the token's 196 pixels.
  label     for a prompt naming fruit f: **target** if ``frac[f] >= t``, else **other** if the shares of all other
            *objects* (every class except ``scene`` and f) sum to ``>= t``, else **background** (`patch_labels`);
            ``t = --target-frac`` = 0.20 (T86.task.md §2.1; the first draft's 0.30 left the head camera 8 % of the
            pre-grasp requests, see ``survey``), the same value for target and other, all three cameras.

Counterfactual: for a kept request, a fruit among banana / orange / pear that has >= 1 target patch in the head camera
is drawn (seeded); its prompt ``CF_TEMPLATE`` (``Pick up the <fruit> and place it in the basket.``, task §2) is what
`pi05_hidden --pool patches --prompt-override` forwards again with the same image. ``--cf-template same`` instead
fills the run's recorded prompt (``Place the <fruit> into the basket.``) — same template as the original forward.
No such fruit -> no counterfactual for that request (the request is kept).

``survey`` (informational, no sampling) lists for **every** pre-grasp request how many head tokens the original fruit
covers at share thresholds 0.30 / 0.20 / 0.10 / 0.05 — how many requests a given ``--target-frac`` keeps.

``sample`` writes ``OUT.json`` (rows with request, t_step, patch counts per camera, counterfactual fruit and prompt,
RGB parity; ``patch_requests`` = {label: requests} for ``--requests``; ``prompt_override`` for ``--prompt-override``;
counts) and ``OUT.labels.npz`` (``frac`` float16 ``[N, 3, 256, 8]``, ``run`` / ``request`` / ``cf_fruit`` per row).

usage (.venv-ag3s or .venv-openpi-live, CPU, MuJoCo osmesa):
  SNAP=<snapshot_t67> PYTHONPATH=$PWD:$SNAP/integration/rby1/manipulation/src MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=2 \\
    python -m AG3S.ag3s.experiments.studies.t86_labels sample OUT.json --spec t77_spec.json --labels t77_labels.json \\
        [--per-run-train 2] [--per-run-test 3] [--only LABEL ...] [--jobs 2]
  ... t86_labels survey OUT.json --spec t77_spec.json --labels t77_labels.json [--jobs 2]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import zlib
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.sources.policy_record import POLICY_CAMERA_NAMES
from AG3S.ag3s.experiments.studies.t79_events import GRID, PATCH_PX, POLICY_PX, patch_of_policy_pixel

SEG_CLASSES = ("apple", "banana", "orange", "pear", "crate", "robot", "obstacle", "scene")
FRUIT_CLASSES = SEG_CLASSES[:4]
SCENE = SEG_CLASSES.index("scene")
#: patch share threshold of a target / other token (T86.task.md §2.1, decided before results: 0.20 instead of the first
#: 0.30 -- the apple is ~8 px across in the head view, 0.30 of a 14 x 14 token is almost never reached).
TARGET_FRAC = 0.20
BG, OTHER, TARGET = 0, 1, 2
LABEL_NAMES = ("background", "other", "target")
PRE_GRASP_PHASES = ("approach", "grasp")
SOURCE_W, SOURCE_H = 299, 224          # `recording.POLICY_SOURCE_WIDTH / HEIGHT`
CF_TEMPLATE = "Pick up the {fruit} and place it in the basket."
CF_FRUITS = ("banana", "orange", "pear")
#: request 0 may show `qpos` itself (no physics step before it) — the other pose is tried when the RGB parity is worse
#: than this mean absolute gray-level difference.
PARITY_OK_MAD = 1.0
MAX_TRIES = 60
_FRUIT_RE = re.compile(r"\b(" + "|".join(FRUIT_CLASSES) + r")\b")


# ================================================================================ pure rules (tested)


def geom_classes(root_body_names: Sequence[str]) -> np.ndarray:
    """Class id per geom from the **root body name** of the geom (`SEG_CLASSES`)."""
    out = []
    for n in root_body_names:
        if n in FRUIT_CLASSES:
            out.append(SEG_CLASSES.index(n))
        elif n == "crate":
            out.append(SEG_CLASSES.index("crate"))
        elif n == "base":
            out.append(SEG_CLASSES.index("robot"))
        elif n.startswith("pick_obstacle"):
            out.append(SEG_CLASSES.index("obstacle"))
        else:
            out.append(SCENE)
    return np.asarray(out, np.int64)


def policy_grid_ids(src: np.ndarray) -> np.ndarray:
    """Segmentation ``[H, W]`` (299 x 224 render) -> the 224 x 224 policy grid by nearest source column / row."""
    src = np.asarray(src)
    h, w = src.shape
    cols = np.minimum(((np.arange(POLICY_PX) + 0.5) * w / POLICY_PX).astype(np.int64), w - 1)
    rows = np.minimum(((np.arange(POLICY_PX) + 0.5) * h / POLICY_PX).astype(np.int64), h - 1)
    return src[np.ix_(rows, cols)]


def patch_fractions(geom_ids: np.ndarray, geom_cls: np.ndarray) -> np.ndarray:
    """``[256, n_class]`` share of each head / wrist token's pixels per class, from a segmentation ``[H, W]`` of geom
    ids (-1 = no geom -> ``scene``), via the 224-px policy grid and `patch_of_policy_pixel`."""
    g = policy_grid_ids(geom_ids)
    cls = np.where(g >= 0, np.asarray(geom_cls)[np.clip(g, 0, len(geom_cls) - 1)], SCENE)
    v, u = np.mgrid[0:POLICY_PX, 0:POLICY_PX]
    patch = patch_of_policy_pixel(u.ravel(), v.ravel())
    n = len(SEG_CLASSES)
    counts = np.bincount(patch * n + cls.ravel(), minlength=GRID * GRID * n).reshape(GRID * GRID, n)
    return counts / float(PATCH_PX * PATCH_PX)


def patch_labels(frac: np.ndarray, fruit: str, target_frac: float = TARGET_FRAC,
                 other_frac: Optional[float] = None) -> np.ndarray:
    """``[256]`` int8: `TARGET` if the fruit covers >= `target_frac` of the token, else `OTHER` if the other objects
    (all classes except ``scene`` and the fruit) together cover >= `other_frac` (default: the same value as
    `target_frac`, T86.task.md §2.1), else `BG`."""
    other_frac = target_frac if other_frac is None else other_frac
    frac = np.asarray(frac, np.float64)
    f = SEG_CLASSES.index(fruit)
    tgt = frac[:, f] >= target_frac - 1e-9
    objects = frac[:, [i for i in range(len(SEG_CLASSES)) if i not in (SCENE, f)]].sum(1)
    oth = (~tgt) & (objects >= other_frac - 1e-9)
    return np.where(tgt, TARGET, np.where(oth, OTHER, BG)).astype(np.int8)


def fruit_of_prompt(prompt: str) -> str:
    """The single fruit noun of a prompt (raises if zero or several)."""
    found = _FRUIT_RE.findall(prompt)
    if len(found) != 1:
        raise ValueError(f"prompt must name exactly one fruit, got {found!r}: {prompt!r}")
    return found[0]


def cf_prompt_for(fruit: str, recorded: Optional[str], template: str = "task") -> str:
    """Counterfactual prompt: ``template="task"`` -> `CF_TEMPLATE`; ``"same"`` -> the recorded prompt with its fruit
    replaced; anything else is used as a ``{fruit}`` template."""
    if template == "task":
        return CF_TEMPLATE.format(fruit=fruit)
    if template == "same":
        if not recorded:
            raise ValueError("--cf-template same needs the recorded prompt")
        return _FRUIT_RE.sub(fruit, recorded)
    if template.count("{fruit}") != 1:
        raise ValueError(f"template must contain one {{fruit}}: {template!r}")
    return template.format(fruit=fruit)


def pick_counterfactual(frac_head: np.ndarray, rng: np.random.Generator, orig_fruit: str = "apple",
                        target_frac: float = TARGET_FRAC) -> Optional[str]:
    """A fruit other than `orig_fruit` among `CF_FRUITS` with >= 1 target patch in the head camera (seeded draw);
    None if there is none."""
    ok = [f for f in CF_FRUITS if f != orig_fruit
          and int((patch_labels(frac_head, f, target_frac) == TARGET).sum()) >= 1]
    return str(ok[int(rng.integers(len(ok)))]) if ok else None


def request_order(candidates: Sequence[int], label: str, seed: int = 0) -> list[int]:
    """Seeded uniform order of the candidate requests of one run (the sample is its prefix)."""
    rng = np.random.default_rng([seed, zlib.crc32(("t86:" + label).encode())])
    return [int(x) for x in rng.permutation(np.asarray(sorted(candidates), np.int64))] if len(candidates) else []


# ================================================================================ MuJoCo / record side


class SegScene:
    """One scene (obstacle configuration): the policy renderer on the obstacle-activated model, geom classes."""

    def __init__(self, model_xml: str, obstacle: Sequence):
        import mujoco

        from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde
        from AG3S.ag3s.experiments.studies.t48b_scene import PolicyRenderer

        self.mj = mujoco
        self.gt = tde.scene_truth(model_xml, (obstacle[0], obstacle[1]))  # activates the obstacle slots
        m = self.m = self.gt.m
        self.pr = PolicyRenderer(m)
        self.pr.data = self.gt.d  # the data the activation touched (mocap / slots), not a fresh one
        roots = [m.body(int(m.body_rootid[int(m.geom_bodyid[g])])).name for g in range(m.ngeom)]
        self.geom_cls = geom_classes(roots)

    def _pose(self, qpos, qvel, after_step: bool) -> None:
        from AG3S.ag3s.experiments.studies.t48b_scene import render_qpos

        self.pr.pose(render_qpos(self.m, qpos, qvel, after_step=after_step))

    def _seg(self, camera: str) -> np.ndarray:
        r = self.pr.renderer
        r.enable_segmentation_rendering()
        try:
            r.update_scene(self.pr.data, camera=camera)
            seg = np.asarray(r.render())
        finally:
            r.disable_segmentation_rendering()
        ids = seg[..., 0].astype(np.int64)
        return np.where(seg[..., 1] == int(self.mj.mjtObj.mjOBJ_GEOM), ids, -1)

    def head_frac(self, qpos, qvel) -> np.ndarray:
        """``[256, 8]`` class shares of the head camera only (the screening render: pose one step back, no RGB)."""
        from AG3S.ag3s.experiments.studies.t48b_scene import POLICY_CAMERAS

        self._pose(qpos, qvel, True)
        return patch_fractions(self._seg(POLICY_CAMERAS[POLICY_CAMERA_NAMES[0]]), self.geom_cls)

    def render(self, qpos, qvel, recorded: dict, first_request: bool) -> dict:
        """``{"frac": [3, 256, 8], "mad": {camera: gray-level MAD vs the recorded image}, "after_step": bool}``."""
        from AG3S.ag3s.experiments.studies.t48b_scene import POLICY_CAMERAS, image_parity

        tried = []
        for after in (True, False) if first_request else (True,):
            self._pose(qpos, qvel, after)
            rgb = self.pr.render()
            mad = {c: v["mad"] for c, v in image_parity(recorded, rgb).items()}
            tried.append((float(np.mean(list(mad.values()))), after, mad))
            if tried[-1][0] <= PARITY_OK_MAD:
                break
        _, after, mad = min(tried, key=lambda t: t[0])
        self._pose(qpos, qvel, after)
        frac = np.stack([patch_fractions(self._seg(POLICY_CAMERAS[c]), self.geom_cls) for c in POLICY_CAMERA_NAMES])
        return {"frac": frac.astype(np.float32), "mad": mad, "after_step": bool(after)}


def _meta_prompt(client_dir: str) -> Optional[str]:
    f = pathlib.Path(client_dir, "ag3s", "run_0000", "meta.json")
    return json.loads(f.read_text()).get("prompt") if f.exists() else None


def sample_run(scene: SegScene, entry: dict, lab: dict, k: int, seed: int = 0, cf_template: str = "task",
               max_tries: int = MAX_TRIES, target_frac: float = TARGET_FRAC) -> tuple[dict, list]:
    """Rows of one run (up to `k` kept requests) + their ``frac`` tables ``[(request, frac [3, 256, 8])]``."""
    from AG3S.ag3s.experiments.studies.t79_events import _sorted_steps

    steps = _sorted_steps(entry["client_dir"])
    recorded_prompt = _meta_prompt(entry["client_dir"])
    orig_fruit = fruit_of_prompt(recorded_prompt) if recorded_prompt else "apple"
    cand = [int(r["request"]) for r in lab["rows"]
            if "gt" in r and r["gt"].get("phase5") in PRE_GRASP_PHASES and int(r["request"]) < len(steps)]
    order = request_order(cand, entry["label"], seed)
    rng = np.random.default_rng([seed, zlib.crc32(("t86cf:" + entry["label"]).encode())])
    rows, tables, tries, rejected = [], [], 0, 0
    for j in order:
        if len(rows) >= k or tries >= max_tries:
            break
        tries += 1
        with np.load(steps[j]) as st:
            qpos, qvel = np.asarray(st["qpos"], float), np.asarray(st["qvel"], float)
            rec = {c: np.asarray(st[f"image_{c}"]) for c in POLICY_CAMERA_NAMES}
            t_step = int(st["t_step"])
        if j != 0:  # screening: the head segmentation alone (request 0 needs the pose choice of the full render)
            quick = scene.head_frac(qpos, qvel)
            if int((patch_labels(quick, orig_fruit, target_frac) == TARGET).sum()) < 1:
                rejected += 1
                continue
        res = scene.render(qpos, qvel, rec, first_request=(j == 0))
        frac = res["frac"]
        n_t = [int((patch_labels(frac[c], orig_fruit, target_frac) == TARGET).sum()) for c in range(len(frac))]
        if n_t[0] < 1:
            rejected += 1
            continue
        cf = pick_counterfactual(frac[0], rng, orig_fruit, target_frac)
        rows.append({"request": j, "t_step": t_step, "phase5": lab["rows"][j]["gt"]["phase5"],
                     "n_target_patches": dict(zip(POLICY_CAMERA_NAMES, n_t)),
                     "n_cf_target_patches_head": (int((patch_labels(frac[0], cf, target_frac) == TARGET).sum())
                                                  if cf else 0),
                     "cf_fruit": cf, "cf_prompt": cf_prompt_for(cf, recorded_prompt, cf_template) if cf else None,
                     "rgb_mad": res["mad"], "pose_after_step": res["after_step"]})
        tables.append((j, frac))
    out = {kk: entry.get(kk) for kk in ("label", "client_dir", "kit", "config", "scene", "scene_number", "split",
                                        "stratum", "episode", "seed", "side", "obstacle")}
    out.update({"recorded_prompt": recorded_prompt, "orig_fruit": orig_fruit, "n_candidates": len(cand),
                "tries": tries, "rejected_no_head_target": rejected, "samples": rows})
    return out, tables


# ================================================================================ survey (yield of the 30 % rule)

SURVEY_THRESHOLDS = (0.30, 0.20, 0.10, 0.05)


def survey_run(scene: SegScene, entry: dict, lab: dict) -> dict:
    """Head-camera apple tokens of **every** pre-grasp request of one run, at several share thresholds (informational:
    how many requests a given ``--target-frac`` keeps). No sampling, no RGB, no wrist cameras."""
    from AG3S.ag3s.experiments.studies.t79_events import _sorted_steps

    steps = _sorted_steps(entry["client_dir"])
    recorded_prompt = _meta_prompt(entry["client_dir"])
    orig = fruit_of_prompt(recorded_prompt) if recorded_prompt else "apple"
    f_orig = SEG_CLASSES.index(orig)
    rows = []
    for r in lab["rows"]:
        j = int(r["request"])
        if "gt" not in r or r["gt"].get("phase5") not in PRE_GRASP_PHASES or j >= len(steps):
            continue
        with np.load(steps[j]) as st:
            frac = scene.head_frac(np.asarray(st["qpos"], float), np.asarray(st["qvel"], float))
        share = frac[:, f_orig]
        rows.append({"request": j, "phase5": r["gt"]["phase5"], "apple_share_max": round(float(share.max()), 3),
                     "n_tokens": {str(t): int((share >= t - 1e-9).sum()) for t in SURVEY_THRESHOLDS},
                     "cf_fruits_at_target_frac": [f for f in CF_FRUITS
                                        if int((frac[:, SEG_CLASSES.index(f)] >= TARGET_FRAC - 1e-9).sum()) >= 1]})
    out = {kk: entry.get(kk) for kk in ("label", "kit", "scene", "scene_number", "split", "obstacle")}
    out.update({"n_candidates": len(rows), "rows": rows})
    return out


def survey_summary(runs: Sequence[dict]) -> dict:
    out: dict[str, Any] = {}
    for split in ("train", "test"):
        rs = [r for r in runs if r["split"] == split]
        d: dict[str, Any] = {"runs": len(rs), "candidate_requests": sum(r["n_candidates"] for r in rs)}
        for t in SURVEY_THRESHOLDS:
            key = str(t)
            req = sum(1 for r in rs for x in r["rows"] if x["n_tokens"][key] >= 1)
            runs_ok = sum(1 for r in rs if any(x["n_tokens"][key] >= 1 for x in r["rows"]))
            d[f"share>={t}"] = {"requests_with_head_target": req, "runs_with_>=1": runs_ok,
                                "request_fraction": req / max(1, d["candidate_requests"])}
        d[f"share>={TARGET_FRAC}_with_counterfactual_fruit"] = sum(1 for r in rs for x in r["rows"]
                                                       if x["n_tokens"][str(TARGET_FRAC)] >= 1 and x["cf_fruits_at_target_frac"])
        by_scene: dict[str, list] = collections.defaultdict(lambda: [0, 0, 0])
        for r in rs:
            c = by_scene[r["scene"]]
            c[0] += 1
            c[1] += any(x["n_tokens"][str(TARGET_FRAC)] >= 1 for x in r["rows"])
            c[2] += sum(x["n_tokens"][str(TARGET_FRAC)] >= 1 for x in r["rows"])
        d[f"by_scene_[runs, runs_with_target_requests, target_requests]_at_{TARGET_FRAC}"] = {k: v for k, v in sorted(by_scene.items())}
        out[split] = d
    return out


# ================================================================================ driver


def _survey_worker(args) -> list:
    entries, labels, model_xml = args
    scene, key, res = None, None, []
    for e in entries:
        kk = tuple(e["obstacle"])
        if kk != key:
            scene, key = None, None
            scene, key = SegScene(model_xml, kk), kk
        run = survey_run(scene, e, labels[e["label"]])
        print(f"{e['label']} [{e['split']}]: {run['n_candidates']} candidates, "
              f"{sum(1 for x in run['rows'] if x['n_tokens']['0.3'] >= 1)} with a head target token", flush=True)
        res.append(run)
    return res


def build_survey(out: pathlib.Path, spec_path: pathlib.Path, labels_path: pathlib.Path, model_xml: str,
                 only: Optional[list] = None, jobs: int = 1) -> dict:
    spec = json.loads(spec_path.read_text())
    entries = spec["runs"] if isinstance(spec, dict) else spec
    if only:
        entries = [e for e in entries if e["label"] in only]
    labels = {r["label"]: r for r in json.loads(labels_path.read_text())["runs"]}
    by_scene = collections.defaultdict(list)
    for e in entries:
        by_scene[tuple(e["obstacle"])].append(e)
    jobs = max(1, min(int(jobs), 2))
    buckets = [[] for _ in range(jobs)]
    for _, es in sorted(by_scene.items(), key=lambda kv: -len(kv[1])):
        min(buckets, key=len).extend(es)
    work = [(b, {e["label"]: labels[e["label"]] for e in b}, model_xml) for b in buckets if b]
    if len(work) == 1:
        parts = _survey_worker(work[0])
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(len(work)) as pool:
            parts = [x for p in pool.map(_survey_worker, work) for x in p]
    order = {e["label"]: i for i, e in enumerate(entries)}
    runs = sorted(parts, key=lambda r: order[r["label"]])
    res = {"step": "T86-survey", "spec": str(spec_path), "labels": str(labels_path), "thresholds": SURVEY_THRESHOLDS,
           "summary": survey_summary(runs), "runs": runs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=0, default=float))
    return res


def _worker(args) -> list:
    entries, labels, model_xml, kw = args
    k_by_split = {"train": kw["per_run_train"], "test": kw["per_run_test"]}
    scene, key, res = None, None, []
    for e in entries:  # grouped by obstacle configuration: one scene (renderer + GT) at a time
        kk = tuple(e["obstacle"])
        if kk != key:
            scene, key = None, None
            scene, key = SegScene(model_xml, kk), kk
        k = k_by_split[e["split"]]
        run, tables = sample_run(scene, e, labels[e["label"]], k, kw["seed"], kw["cf_template"], kw["max_tries"],
                                 kw["target_frac"])
        print(f"{e['label']} [{e['split']}]: {len(run['samples'])}/{k} kept in {run['tries']} tries "
              f"(candidates {run['n_candidates']}, no head target {run['rejected_no_head_target']})", flush=True)
        res.append((run, tables))
    return res


def build_samples(out: pathlib.Path, spec_path: pathlib.Path, labels_path: pathlib.Path, model_xml: str,
                  per_run_train: int = 2, per_run_test: int = 3, only: Optional[list] = None, seed: int = 0,
                  jobs: int = 1, cf_template: str = "task", max_tries: int = MAX_TRIES,
                  target_frac: float = TARGET_FRAC) -> dict:
    spec = json.loads(spec_path.read_text())
    entries = spec["runs"] if isinstance(spec, dict) else spec
    if only:
        entries = [e for e in entries if e["label"] in only]
    labels = {r["label"]: r for r in json.loads(labels_path.read_text())["runs"]}
    missing = [e["label"] for e in entries if e["label"] not in labels]
    if missing:
        raise SystemExit(f"{len(missing)} spec runs have no T77 labels, e.g. {missing[:3]}")
    by_scene = collections.defaultdict(list)
    for e in entries:
        by_scene[tuple(e["obstacle"])].append(e)
    jobs = max(1, min(int(jobs), 2))  # pod rule: at most 2 workers for this STEP
    buckets = [[] for _ in range(jobs)]
    for _, es in sorted(by_scene.items(), key=lambda kv: -len(kv[1])):
        min(buckets, key=len).extend(es)
    kw = {"per_run_train": per_run_train, "per_run_test": per_run_test, "seed": seed, "cf_template": cf_template,
          "max_tries": max_tries, "target_frac": target_frac}
    work = [(b, {e["label"]: labels[e["label"]] for e in b}, model_xml, kw) for b in buckets if b]
    if len(work) == 1:
        parts = _worker(work[0])
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(len(work)) as pool:
            parts = [x for p in pool.map(_worker, work) for x in p]
    order = {e["label"]: i for i, e in enumerate(entries)}
    results = sorted(parts, key=lambda x: order[x[0]["label"]])
    runs = [r for r, _ in results]
    G_run, G_req, G_frac, G_cf = [], [], [], []
    for ri, (run, tables) in enumerate(results):
        by_req = {s["request"]: s for s in run["samples"]}
        for req, frac in tables:
            G_run.append(ri)
            G_req.append(req)
            G_frac.append(frac)
            G_cf.append(by_req[req]["cf_fruit"] or "")
    npz = out.with_suffix(".labels.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz, run=np.asarray(G_run, np.int32), request=np.asarray(G_req, np.int32),
                        frac=np.asarray(G_frac, np.float16).reshape(-1, len(POLICY_CAMERA_NAMES), GRID * GRID,
                                                                    len(SEG_CLASSES)),
                        cf_fruit=np.asarray(G_cf, dtype=str), labels=np.asarray([r["label"] for r in runs]),
                        classes=np.asarray(SEG_CLASSES), cameras=np.asarray(POLICY_CAMERA_NAMES))
    res = {"step": "T86", "spec": str(spec_path), "labels": str(labels_path), "model_xml": model_xml,
           "labels_npz": str(npz), "definitions": __doc__,
           "params": {"per_run_train": per_run_train, "per_run_test": per_run_test, "seed": seed,
                      "cf_template": cf_template, "cf_prompt_template": CF_TEMPLATE if cf_template == "task" else cf_template,
                      "target_frac": target_frac, "other_frac": target_frac, "max_tries": max_tries,
                      "classes": SEG_CLASSES, "pre_grasp_phases": PRE_GRASP_PHASES},
           "summary": summarise(runs),
           "patch_requests": {r["label"]: [s["request"] for s in r["samples"]] for r in runs},
           "prompt_override": {r["label"]: {str(s["request"]): [s["cf_prompt"]] for s in r["samples"] if s["cf_prompt"]}
                               for r in runs},
           "runs": runs}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=0, default=float))
    return res


def summarise(runs: Sequence[dict]) -> dict:
    out: dict[str, Any] = {}
    for split in ("train", "test"):
        rs = [r for r in runs if r["split"] == split]
        sm = [s for r in rs for s in r["samples"]]
        out[split] = {
            "runs": len(rs), "runs_with_sample": sum(1 for r in rs if r["samples"]), "requests": len(sm),
            "with_counterfactual": sum(1 for s in sm if s["cf_prompt"]),
            "cf_fruits": dict(collections.Counter(s["cf_fruit"] for s in sm if s["cf_fruit"])),
            "candidates": sum(r["n_candidates"] for r in rs), "tries": sum(r["tries"] for r in rs),
            "rejected_no_head_target": sum(r["rejected_no_head_target"] for r in rs),
            "target_patch_requests_by_camera": {c: sum(1 for s in sm if s["n_target_patches"][c] >= 1)
                                                for c in POLICY_CAMERA_NAMES},
            "rgb_mad_mean": {c: (float(np.mean([s["rgb_mad"][c] for s in sm])) if sm else None)
                             for c in POLICY_CAMERA_NAMES},
            "rgb_mad_max": {c: (float(np.max([s["rgb_mad"][c] for s in sm])) if sm else None)
                            for c in POLICY_CAMERA_NAMES},
            "scenes": dict(collections.Counter(r["scene"] for r in rs for _ in r["samples"])),
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sample")
    a.add_argument("out")
    a.add_argument("--spec", required=True)
    a.add_argument("--labels", required=True)
    a.add_argument("--model-xml", default=None)
    a.add_argument("--only", nargs="+", default=None)
    a.add_argument("--per-run-train", type=int, default=2, help="kept requests per train run (default 2)")
    a.add_argument("--per-run-test", type=int, default=3, help="kept requests per test run (default 3)")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--cf-template", default="task",
                   help="'task' (default: Pick up the <fruit> and place it in the basket.), 'same' (the recorded "
                        "prompt's template) or a '{fruit}' template")
    a.add_argument("--max-tries", type=int, default=MAX_TRIES, help="rendered candidates per run before giving up")
    a.add_argument("--target-frac", type=float, default=TARGET_FRAC,
                   help="pixel share that makes a token a target (and an other) token (T86.task.md §2.1: 0.20)")
    a.add_argument("--jobs", type=int, default=1, help="worker processes (<= 2)")
    b = sub.add_parser("survey", help="head-camera apple tokens of every pre-grasp request at several share thresholds")
    b.add_argument("out")
    b.add_argument("--spec", required=True)
    b.add_argument("--labels", required=True)
    b.add_argument("--model-xml", default=None)
    b.add_argument("--only", nargs="+", default=None)
    b.add_argument("--jobs", type=int, default=1, help="worker processes (<= 2)")
    args = ap.parse_args(argv)
    from AG3S.ag3s.experiments.studies.t77_labels import default_xml

    if args.cmd == "survey":
        res = build_survey(pathlib.Path(args.out), pathlib.Path(args.spec), pathlib.Path(args.labels),
                           args.model_xml or default_xml(), args.only, args.jobs)
        print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
        return

    res = build_samples(pathlib.Path(args.out), pathlib.Path(args.spec), pathlib.Path(args.labels),
                        args.model_xml or default_xml(), args.per_run_train, args.per_run_test, args.only, args.seed,
                        args.jobs, args.cf_template, args.max_tries, args.target_frac)
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    main()
