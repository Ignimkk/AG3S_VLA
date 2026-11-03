"""T91 — offline evaluation of the benchmark-agnostic phase machine (`AG3S/trajopt/phase_machine.py`) on the T77 records.

Data (read only, produced by T77): ``t77_spec.json`` (427 AG3S closed-loop runs, scene split fixed in T77 §2) and
``t77_labels.json`` (per request MuJoCo ground truth ``gt.phase5`` / ``gt.held`` / ``gt.palm_dist_mm`` and the recorded
serving values). Scene 허들211 (item HURDLE_ep211_s2113) is excluded everywhere (user 2026-11-03).

Sub-commands
  extract   per run: the phase machine's inputs from the records — measured gripper (client ``state`` column of the
            grasping hand), hand EE site (MuJoCo forward kinematics of the client ``qpos``, base frame; the site is
            ``<side>_ee``), ATS ``manipulated`` (centroid · state) and ``destination`` centroid and the VLA chunk
            ``reference_chunk`` gripper column (server chunk summary / arrays, i-th chunk <-> i-th client step
            sorted by t_step, `t77_labels.label_run`). The simulator's apple position and ground-truth phase are
            stored beside them for **scoring and diagnostics only** (never fed to the machine).
  select    thresholds chosen on TRAIN scenes only (grid, rule below); writes ``thresholds.json``.
  evaluate  test metrics + verdict from the frozen ``thresholds.json``.

T91b adds (E12, `handoff/T91b.task.md`; the T91 sub-commands above are unchanged):
  extract-wrist  per run: the grasping hand's wrist depth (stride 4, uint16 mm) and the ATS target points projected into that camera,
                 rows aligned with the T91 feature file (separate directory)
  gap-select     ROI x depth window x tau x m on TRAIN scenes only (exhaustive grid, same rule as `select`: danger <= 1 % budget, then
                 accuracy, ties keep the earlier candidate), depth arm (main) and projection arm (reference); writes gap_thresholds.json
  gap-evaluate   test metrics + verdicts: A (T77 classifier), T91 machine, T91b depth, T91b projection (+ both / either reported only)
The SafeLIBERO side is `t91b_safelibero.py` (adapter + report) and `t91b_safelibero_export.py` (py3.8 physics re-render).

Comparison (test scenes 211 · 954 · 122 · 119, scene-mean unless stated):
  A  the 5-stage hidden-state classifier of T77 (``kv_L4`` recorded predictions, `probe/t77_probe.json`)
  B  this state machine
  C  the recorded grasp latch state only (summary ``grasp.state``: latched -> carry, else approach)  [reference]
Metrics: accuracy (request · scene mean), **dangerous mislabel rate** = GT approach/grasp predicted carry/place
(the task's definition; the T77 definition that also counts retreat is reported as ``false_carry_t77``), stage
transition delay (requests between the GT transition and the first prediction of the new stage), success and failed
runs separately (success = GT reaches ``place``).

Threshold rule (train scenes only; fixed before any test number): maximise train scene-mean accuracy subject to train
scene-mean dangerous rate <= ``DANGER_BUDGET`` (a stated safety budget, not tuned); if none satisfies it, minimise the
rate. Accuracy differences <= ``ACC_TIE`` (0.2 pp) are ties and the earlier candidate in grid order stays; the grid lists the
values that generalise to other objects first (e.g. the loosest ``closed_thr``).
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import itertools
import json
import os
import pathlib
import sys
import time
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from AG3S.trajopt.phase_machine import STAGES, PhaseConfig, PhaseMachine, PhaseObservation

EXCLUDED_SCENES = ("허들211",)           # 2026-11-03 user instruction (item HURDLE_ep211_s2113)
GRIP_STATE_COL = {"left": 7, "right": 15}  # RBY1 16-D: client `state` and chunk share the layout
CHUNK_ROWS = 16                           # leading chunk rows stored (>= every K in the grid)
DANGER_BUDGET = 0.01
#: the thresholds `select` chose on the 210 train runs (grid v3, frozen before any test number); `PhaseConfig` defaults
#: equal these (tested). Parameters not listed keep their a-priori default.
TRAIN_SELECTED = {"occluded_bootstrap": True, "bootstrap_hand_mm": 20.0, "blocked_min_open": 0.10, "d_near": 0.06,
                  "comotion_window": 4, "comotion_ratio": 0.4, "comotion_cos": 0.8, "intent_rows": 16, "d_dest": 0.12,
                  "closed_thr": 0.85, "open_thr": 0.90}   # last two: the a-priori values kept (no grid candidate beat them by > 0.2 pp)
ACC_TIE = 0.002                           # accuracy differences <= this are ties (earlier candidate kept)
T77 = ("/mnt/dev/AG3S_VLA-storage/results/development/20261030T004052+0900_8ab3c7ef7757_1c61eaabd79c/verify/T77")
XML_REL = "robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml"
SNAP_DEFAULT = ("/mnt/dev/AG3S_VLA-storage/results/development/20261029T150945+0900_5f7a99f26dbf_a49d7cbd070b/"
                "snapshot_t67")


def safe_name(label: str) -> str:
    return label.replace("/", "__")


def keep_run(entry: dict) -> bool:
    return entry.get("scene") not in EXCLUDED_SCENES


# ================================================================================ extract


class FK:
    """Hand site and apple (diagnostic) in the robot base frame from a client qpos. Obstacles do not matter."""

    def __init__(self, xml: str) -> None:
        import mujoco

        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        bid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.base, self.apple = bid("base"), bid("apple")
        self.site = {s: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_ee") for s in ("left", "right")}
        if min(self.base, self.apple, *self.site.values()) < 0:
            raise RuntimeError("model without base / apple / <side>_ee")

    def at(self, qpos: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[:] = qpos
        self.mj.mj_kinematics(self.m, self.d)
        R, o = self.d.xmat[self.base].reshape(3, 3), self.d.xpos[self.base]
        return (self.d.site_xpos[self.site[side]] - o) @ R, (self.d.xpos[self.apple] - o) @ R


def extract_run(fk: FK, entry: dict, lab: dict) -> dict:
    """Arrays of one run, one row per labelled request (those with ``gt``)."""
    from AG3S.ag3s.experiments.studies.t79_events import _sorted_steps

    side = entry["side"]
    gcol = GRIP_STATE_COL[side]
    steps = _sorted_steps(entry["client_dir"])
    rows = {k: [] for k in ("request", "t_step", "grip_raw", "ee", "apple_gt", "tgt", "tgt_fresh", "tgt_state", "dest",
                            "chunk_grip", "latch", "gt", "held", "palm_mm", "hsr6_to5", "has_target")}
    for r in lab["rows"]:
        j = r["request"]
        if "gt" not in r or j >= len(steps):
            continue
        with np.load(steps[j]) as st:
            qpos = np.asarray(st["qpos"], float)
            state = np.asarray(st["state"], float)
            t_step = int(st["t_step"])
        ee, apple = fk.at(qpos, side)
        with np.load(entry["chunks"][j], allow_pickle=True) as z:
            summ = json.loads(str(z["summary_json"]))
            ref = np.asarray(z["reference_chunk"], float)[:CHUNK_ROWS, GRIP_STATE_COL[side]]
        man = summ.get("manipulated") or {}
        dst = summ.get("destination") or {}
        cen = man.get("centroid")
        rows["request"].append(j)
        rows["t_step"].append(t_step)
        rows["grip_raw"].append(float(state[gcol]))
        rows["ee"].append(ee)
        rows["apple_gt"].append(apple)
        rows["tgt"].append(np.asarray(cen, float) if cen is not None else np.full(3, np.nan))
        rows["tgt_state"].append(str(man.get("state")))
        rows["tgt_fresh"].append(man.get("state") == "visible")
        rows["dest"].append(np.asarray(dst["centroid"], float) if dst.get("centroid") is not None else np.full(3, np.nan))
        rows["chunk_grip"].append(np.pad(ref, (0, CHUNK_ROWS - len(ref)), constant_values=np.nan))
        rows["latch"].append(str((summ.get("grasp") or {}).get("state")))
        rows["gt"].append(r["gt"]["phase5"])
        rows["held"].append(bool(r["gt"]["held"]))
        rows["palm_mm"].append(float(r["gt"]["palm_dist_mm"]))
        rows["hsr6_to5"].append(str(r["serving"].get("hsr6_to5")))
        rows["has_target"].append(bool(summ.get("has_target")))
    out = {k: np.asarray(v) for k, v in rows.items()}
    return out


# ---- T91b: wrist-camera inputs (separate feature dir, same request rows as `extract_run`)
WRIST_STRIDE = 4          # stored wrist depth = depth[::4, ::4] (480 x 640 -> 120 x 160, uint16 mm)
WRIST_MAX_POINTS = 128    # ATS target points kept per request (evenly subsampled), projected into the wrist camera


def extract_wrist_run(entry: dict, feat: dict) -> dict:
    """Per labelled request of the T91 feature file `feat`: the grasping hand's wrist depth (strided) and the ATS target points
    projected into that camera (``u, v, z`` at 640 x 480, NaN-padded). Rows are aligned with ``feat['request']``."""
    from AG3S.ag3s.experiments.studies.t79_events import _sorted_steps
    from AG3S.trajopt.finger_gap import project_points

    side = entry["side"]
    sfx = "l" if side == "left" else "r"
    steps = _sorted_steps(entry["client_dir"])
    wd, tp, hw = [], [], None
    K0 = None
    for j, t_step in zip(feat["request"], feat["t_step"]):
        with np.load(steps[int(j)]) as st:
            if int(st["t_step"]) != int(t_step):
                raise ValueError("t_step mismatch with the T91 feature file")
            d = np.asarray(st[f"depth_wrist_cam_{sfx}"])
            K = np.asarray(st[f"K_wrist_cam_{sfx}"], float)
            T = np.asarray(st[f"T_base_cam_wrist_cam_{sfx}"], float)
        hw, K0 = d.shape, K
        wd.append(d[::WRIST_STRIDE, ::WRIST_STRIDE])
        with np.load(entry["chunks"][int(j)], allow_pickle=False) as z:
            pts = np.asarray(z["target_points"], float).reshape(-1, 3) if "target_points" in z.files else np.zeros((0, 3))
        if len(pts) > WRIST_MAX_POINTS:
            pts = pts[np.linspace(0, len(pts) - 1, WRIST_MAX_POINTS).astype(int)]
        uvz = np.full((WRIST_MAX_POINTS, 3), np.nan, np.float32)
        if len(pts):
            uvz[: len(pts)] = project_points(pts, K, T)
        tp.append(uvz)
    return {"wd": np.asarray(wd, np.uint16), "tp_uvz": np.asarray(tp, np.float32), "K": K0,
            "image_hw": np.asarray(hw), "request": np.asarray(feat["request"]), "t_step": np.asarray(feat["t_step"])}


def cmd_extract_wrist(args: argparse.Namespace) -> None:
    spec = json.loads(pathlib.Path(args.spec).read_text())["runs"]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [e for e in spec if keep_run(e) and (args.split in (None, e["split"]))
            and (pathlib.Path(args.feat) / f"{safe_name(e['label'])}.npz").exists()]
    if args.only:
        todo = [e for e in todo if e["label"] in args.only]
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        todo = todo[i::n]
    if args.limit:
        todo = todo[: args.limit]
    t0 = time.time()
    for i, e in enumerate(todo):
        f = out / f"{safe_name(e['label'])}.npz"
        if f.exists():
            continue
        with np.load(pathlib.Path(args.feat) / f"{safe_name(e['label'])}.npz", allow_pickle=False) as z:
            feat = {"request": z["request"], "t_step": z["t_step"]}
        w = extract_wrist_run(e, feat)
        tmp = f.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **w)
        os.replace(tmp, f)
        print(f"[{i + 1}/{len(todo)}] {e['label']} {len(w['request'])} req ({time.time() - t0:.0f}s)", flush=True)


def cmd_extract(args: argparse.Namespace) -> None:
    spec = json.loads(pathlib.Path(args.spec).read_text())["runs"]
    labs = {r["label"]: r for r in json.loads(pathlib.Path(args.labels).read_text())["runs"]}
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fk = FK(args.xml)
    todo = [e for e in spec if keep_run(e) and (args.split in (None, e["split"]))]
    if args.only:
        todo = [e for e in todo if e["label"] in args.only]
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        todo = todo[i::n]
    if args.limit:
        todo = todo[: args.limit]
    t0 = time.time()
    for i, e in enumerate(todo):
        f = out / f"{safe_name(e['label'])}.npz"
        if f.exists():
            continue
        feat = extract_run(fk, e, labs[e["label"]])
        tmp = f.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **feat, side=e["side"])
        os.replace(tmp, f)
        print(f"[{i + 1}/{len(todo)}] {e['label']} {len(feat['request'])} req ({time.time() - t0:.0f}s)", flush=True)


# ================================================================================ machine over records

IDX = {n: i for i, n in enumerate(STAGES)}
APPROACH, GRASP, CARRY, PLACE, RETREAT = range(5)
LATCH_TO5 = {"held": CARRY, "closing": GRASP, "placed": RETREAT}   # C: the recorded latch state alone (reference)


def load_runs(feat_dir: str, spec: Sequence[dict], split: Optional[str] = None) -> list[dict]:
    """Cached features joined with the spec's scene / split. Only runs that were extracted are returned."""
    out = []
    for e in spec:
        if not keep_run(e) or (split and e["split"] != split):
            continue
        f = pathlib.Path(feat_dir) / f"{safe_name(e['label'])}.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d.update(label=e["label"], scene=e["scene"], split=e["split"], side=str(e["side"]), kit=e["kit"])
        d["y"] = np.array([IDX[g] for g in d["gt"]], int)
        out.append(d)
    return out


def observations(run: dict, gap: Optional[dict] = None) -> Iterable[PhaseObservation]:
    """The machine's inputs only: no simulator apple position, no ground-truth phase. ``gap`` (T91b) = per-request arrays
    ``{"depth": (n,), "proj": (n,)}`` of the wrist-camera finger-gap evidence (NaN = none)."""
    for i in range(len(run["request"])):
        fg = fp = None
        if gap is not None:
            fg = None if np.isnan(gap["depth"][i]) else float(gap["depth"][i])
            fp = None if np.isnan(gap["proj"][i]) else float(gap["proj"][i])
        yield PhaseObservation(
            gripper_raw=float(run["grip_raw"][i]), ee_pos=run["ee"][i],
            target_pos=run["tgt"][i], target_fresh=bool(run["tgt_fresh"][i]),
            destination_pos=run["dest"][i], chunk_gripper=run["chunk_grip"][i], finger_gap=fg, finger_gap_proj=fp)


def predict_run(run: dict, params: dict, trace: bool = False, gap: Optional[dict] = None):
    m = PhaseMachine(PhaseConfig.rby1(run["side"], **params))
    pred, ev = [], []
    for obs in observations(run, gap):
        st, e = m.step(obs)
        pred.append(IDX[st])
        if trace:
            ev.append(e)
    return (np.array(pred, int), ev) if trace else np.array(pred, int)


def latch_run(run: dict) -> np.ndarray:
    return np.array([LATCH_TO5.get(str(s), APPROACH) for s in run["latch"]], int)


def is_success(run: dict) -> bool:
    """Proxy: the simulator ground truth reached ``place`` (apple held inside the basket footprint)."""
    return bool(np.any(run["y"] == PLACE))


# ================================================================================ metrics (tested)


def danger_counts(y: np.ndarray, p: np.ndarray, with_retreat: bool = False) -> tuple[int, int]:
    """(# GT approach/grasp, # of those predicted carry/place [and retreat for the T77 definition])."""
    y, p = np.asarray(y), np.asarray(p)
    m = np.isin(y, (APPROACH, GRASP))
    bad = (CARRY, PLACE, RETREAT) if with_retreat else (CARRY, PLACE)
    return int(m.sum()), int(np.isin(p[m], bad).sum())


def scene_mean(runs: Sequence[dict], preds: Sequence[np.ndarray], fn) -> Optional[float]:
    """Mean over scenes of ``fn(y_all, p_all)`` (scene = runs pooled, requests weighted); None entries are skipped."""
    by = collections.defaultdict(lambda: ([], []))
    for r, p in zip(runs, preds, strict=True):
        by[r["scene"]][0].append(r["y"])
        by[r["scene"]][1].append(p)
    vals = []
    for ys, ps in by.values():
        v = fn(np.concatenate(ys), np.concatenate(ps))
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def _acc(y, p):
    return float(np.mean(y == p)) if len(y) else None


def _danger(y, p, with_retreat=False):
    n, k = danger_counts(y, p, with_retreat)
    return (k / n) if n else None


def summarize(runs: Sequence[dict], preds: Sequence[np.ndarray]) -> dict:
    from AG3S.ag3s.experiments.studies.t64_phase6 import boundary_delays, prf

    y = np.concatenate([r["y"] for r in runs]) if runs else np.zeros(0, int)
    p = np.concatenate(list(preds)) if runs else np.zeros(0, int)
    n, k = danger_counts(y, p)
    _, k77 = danger_counts(y, p, True)
    pr, rc, f1, sup, cm = prf(y, p, len(STAGES)) if len(y) else (np.zeros(5),) * 4 + (np.zeros((5, 5), int),)
    bd = boundary_delays(list(preds), [r["y"] for r in runs], [np.asarray(r["t_step"]) for r in runs], len(STAGES))
    return {
        "n_runs": len(runs), "n_requests": int(len(y)),
        "accuracy_request": _acc(y, p), "accuracy_scene_mean": scene_mean(runs, preds, _acc),
        "danger_request": {"rate": (k / n) if n else None, "n_not_held": n, "n_dangerous": k},
        "danger_scene_mean": scene_mean(runs, preds, _danger),
        "false_carry_t77_request": (k77 / n) if n else None,
        "false_carry_t77_scene_mean": scene_mean(runs, preds, lambda a, b: _danger(a, b, True)),
        "macro_f1": float(np.mean(f1)) if len(y) else None,
        "per_class_recall": {STAGES[i]: float(rc[i]) for i in range(len(STAGES))},
        "confusion": cm.tolist(), "boundary_delay": bd,
    }


def summarize_groups(runs: Sequence[dict], preds: Sequence[np.ndarray]) -> dict:
    ok = [i for i, r in enumerate(runs) if is_success(r)]
    bad = [i for i in range(len(runs)) if i not in set(ok)]
    pick = lambda idx: summarize([runs[i] for i in idx], [preds[i] for i in idx])  # noqa: E731
    return {"all": summarize(runs, preds), "success_runs": pick(ok), "failed_runs": pick(bad)}


# ================================================================================ threshold selection (train only)

#: coordinate-descent groups: each is a dict of parameter -> candidate values, searched jointly (grid inside the group).
SEARCH_GROUPS = (
    # the structural switch first: judged on its own, not trapped by earlier distance thresholds (v1 order did that)
    {"occluded_bootstrap": (False, True), "bootstrap_hand_mm": (20.0, 40.0, 60.0), "blocked_min_open": (0.10, 0.20, 0.30)},
    {"d_near": (0.12, 0.10, 0.08, 0.06, 0.05, 0.04)},   # most tolerant of ATS centroid offset first
    {"closed_thr": (0.90, 0.85, 0.80), "open_thr": (0.92, 0.96)},   # loosest-for-other-objects first
    {"comotion_window": (2, 3, 4, 6), "comotion_ratio": (0.4, 0.6, 0.8), "comotion_cos": (0.6, 0.8)},
    {"intent_rows": (4, 8, 16), "d_dest": (0.08, 0.12, 0.16, 0.20)},
)
SEARCH_PASSES = 2


def score_params(runs: Sequence[dict], params: dict) -> dict:
    preds = [predict_run(r, params) for r in runs]
    return {"accuracy_scene_mean": scene_mean(runs, preds, _acc), "danger_scene_mean": scene_mean(runs, preds, _danger)}


def better(a: dict, b: Optional[dict], budget: float = DANGER_BUDGET) -> bool:
    """Is score ``a`` better than ``b`` under the rule: feasible (danger <= budget) beats infeasible; among feasible
    higher accuracy (by more than `ACC_TIE`); among infeasible lower danger. Strict, so ties keep the earlier candidate."""
    if b is None:
        return True
    fa, fb = a["danger_scene_mean"] <= budget, b["danger_scene_mean"] <= budget
    if fa != fb:
        return fa
    if fa:
        return a["accuracy_scene_mean"] > b["accuracy_scene_mean"] + ACC_TIE
    return a["danger_scene_mean"] < b["danger_scene_mean"] - 1e-12


def select_thresholds(train_runs: Sequence[dict], start: Optional[dict] = None, log=print) -> dict:
    assert train_runs and all(r["split"] == "train" for r in train_runs), "threshold selection sees train scenes only"
    cur = dict(start or {})
    best, trail = None, []
    for pas in range(SEARCH_PASSES):
        changed = False
        for grp in SEARCH_GROUPS:
            names = list(grp)
            for vals in itertools.product(*(grp[n] for n in names)):
                cand = {**cur, **dict(zip(names, vals))}
                sc = score_params(train_runs, cand)
                trail.append({"pass": pas, "params": {n: v for n, v in zip(names, vals)}, **sc})
                if better(sc, best):
                    best, cur, changed = sc, cand, True
            log(f"[select] pass {pas} {names}: {({n: cur.get(n) for n in names})} -> {best}")
        if not changed:
            break
    return {"params": cur, "train_score": best, "budget": DANGER_BUDGET, "n_configs": len(trail), "trail": trail}


def cmd_select(args: argparse.Namespace) -> None:
    spec = json.loads(pathlib.Path(args.spec).read_text())["runs"]
    tr = load_runs(args.feat, spec, "train")
    res = select_thresholds(tr)
    import dataclasses

    res.update(train_runs=len(tr), train_scenes=sorted({r["scene"] for r in tr}),
               full_config=dataclasses.asdict(PhaseConfig(**res["params"])))
    pathlib.Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("params", "train_score", "n_configs")}, ensure_ascii=False, indent=1))



# ================================================================================ T91b: wrist-camera finger-gap evidence

#: ROI / depth-window grid for the RBY1 wrist camera (train only). ROI = column band centred on the hand-site column
#: (u = 0.5, the `<side>_ee` site projects to pixel (320, 472) at 0.10 m on every RBY1 wrist image), rows from ``v_top`` to the
#: site row 0.98. Listed in the order ties keep (earlier wins): moderate first.
GAP_ROIS = tuple((0.5 - h, 0.5 + h, vt, 0.98) for vt in (0.72, 0.62) for h in (0.15, 0.10, 0.20))
GAP_DMAX = (0.11, 0.09, 0.13)
GAP_DMIN = 0.03
GAP_TAU = (0.2, 0.1, 0.3, 0.4, 0.5)
GAP_M = (2, 1, 3)
SPLAT_FRAC = 0.012


#: what `gap-select` chose on the 210 RBY1 train runs (frozen before any test number; `PhaseConfig` / `finger_gap.rby1_wrist` defaults
#: equal these, tested). ``depth`` is the main arm; the projection arm is the reference.
GAP_SELECTED = {"depth": {"gap_tau": 0.3, "gap_m": 1, "roi": (0.35, 0.65, 0.72, 0.98), "d_min": 0.03, "d_max": 0.09},
                "projection": {"gap_tau_proj": 0.1, "gap_m": 3, "roi": (0.35, 0.65, 0.62, 0.98)}}


def load_wrist(runs: Sequence[dict], wrist_dir: str) -> list[dict]:
    """Attach the strided wrist depth / projected target points to the T91 feature dicts (rows must align)."""
    out = []
    for r in runs:
        f = pathlib.Path(wrist_dir) / f"{safe_name(r['label'])}.npz"
        if not f.exists():
            continue
        with np.load(f, allow_pickle=False) as z:
            w = {k: z[k] for k in z.files}
        if not (np.array_equal(w["request"], r["request"]) and np.array_equal(w["t_step"], r["t_step"])):
            raise ValueError(f"{r['label']}: wrist rows differ from the T91 feature rows")
        out.append({**r, **{k: w[k] for k in ("wd", "tp_uvz", "K", "image_hw")}})
    return out


def fg_config(roi: Sequence[float], d_min: float, d_max: float):
    from AG3S.trajopt.finger_gap import FingerGapConfig

    return FingerGapConfig(roi_u=(roi[0], roi[1]), roi_v=(roi[2], roi[3]), d_min=d_min, d_max=d_max, splat_frac=SPLAT_FRAC)


def depth_fraction(run: dict, cfg) -> np.ndarray:
    """Per request share of ROI pixels in the depth window (on the stored stride-4 depth), vectorised over requests."""
    wd = run["wd"]
    r0, r1, c0, c1 = cfg.box(wd.shape[1], wd.shape[2])
    roi = wd[:, r0:r1, c0:c1].astype(np.float32) / 1000.0
    ok = (roi >= cfg.d_min) & (roi <= cfg.d_max)
    return ok.reshape(len(wd), -1).mean(axis=1)


def projection_fractions(run: dict, cfg) -> np.ndarray:
    """Per request share of ROI pixels covered by the projected ATS target points; NaN where the request has no points."""
    from AG3S.trajopt.finger_gap import projection_fraction

    hw = tuple(int(x) for x in run["image_hw"])
    out = np.full(len(run["tp_uvz"]), np.nan)
    for i, a in enumerate(run["tp_uvz"]):
        v = projection_fraction(a, hw, cfg)
        if v is not None:
            out[i] = v
    return out


def gap_arrays(run: dict, roi: Sequence[float], d_min: float, d_max: float, cache: Optional[dict] = None) -> dict:
    """{"depth", "proj"} evidence arrays of one run for one ROI / window (``proj`` ignores the depth window)."""
    cfg = fg_config(roi, d_min, d_max)
    key_d, key_p = (run["label"], tuple(roi), d_min, d_max), (run["label"], tuple(roi), "proj")
    if cache is None:
        cache = {}
    if key_d not in cache:
        cache[key_d] = depth_fraction(run, cfg)
    if key_p not in cache:
        cache[key_p] = projection_fractions(run, cfg)
    return {"depth": cache[key_d], "proj": cache[key_p]}


def score_gap(runs: Sequence[dict], params: dict, roi, d_min, d_max, cache: dict) -> dict:
    preds = [predict_run(r, params, gap=gap_arrays(r, roi, d_min, d_max, cache)) for r in runs]
    return {"accuracy_scene_mean": scene_mean(runs, preds, _acc), "danger_scene_mean": scene_mean(runs, preds, _danger)}


def select_gap(train_runs: Sequence[dict], base: dict, source: str, log=print) -> dict:
    """Exhaustive grid over (ROI, d_max[, tau], m) on TRAIN runs with the T91 rule (`better`: feasible danger <= budget, then
    accuracy, ties keep the earlier candidate). ``source`` "depth" searches ROI x d_max x tau x m; "projection" ROI x tau x m
    (the depth window is irrelevant to it). All T91 thresholds in ``base`` stay fixed."""
    assert train_runs and all(r["split"] == "train" for r in train_runs), "threshold selection sees train scenes only"
    cache: dict = {}
    best, best_cfg, trail = None, None, []
    dmaxs = GAP_DMAX if source == "depth" else (GAP_DMAX[0],)
    for roi, dmax, tau, m in itertools.product(GAP_ROIS, dmaxs, GAP_TAU, GAP_M):
        prm = {**base, "finger_gap_mode": "on", "finger_gap_source": source, "gap_m": m,
               **({"gap_tau": tau} if source == "depth" else {"gap_tau_proj": tau})}
        sc = score_gap(train_runs, prm, roi, GAP_DMIN, dmax, cache)
        trail.append({"roi": list(roi), "d_max": dmax, "tau": tau, "m": m, **sc})
        if better(sc, best):
            best, best_cfg = sc, {"params": prm, "roi": list(roi), "d_min": GAP_DMIN, "d_max": dmax}
    log(f"[gap-select {source}] {best_cfg} -> {best}")
    return {"source": source, "selected": best_cfg, "train_score": best, "budget": DANGER_BUDGET, "n_configs": len(trail),
            "trail": trail}


def cmd_gap_select(args: argparse.Namespace) -> None:
    spec = json.loads(pathlib.Path(args.spec).read_text())["runs"]
    tr = load_wrist(load_runs(args.feat, spec, "train"), args.wrist)
    base = json.loads(pathlib.Path(args.t91_thresholds).read_text())["params"]
    res = {"step": "T91b", "t91_params": base, "train_runs": len(tr), "train_scenes": sorted({r["scene"] for r in tr}),
           "arms": {src: select_gap(tr, base, src) for src in ("depth", "projection")}}
    pathlib.Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({s: {k: a[k] for k in ("selected", "train_score", "n_configs")} for s, a in res["arms"].items()},
                     ensure_ascii=False, indent=1, default=float))


def gap_arm_predictions(runs: Sequence[dict], sel: dict, cache: Optional[dict] = None, **override) -> list[np.ndarray]:
    prm = {**sel["params"], **override}
    cache = {} if cache is None else cache
    return [predict_run(r, prm, gap=gap_arrays(r, sel["roi"], sel["d_min"], sel["d_max"], cache)) for r in runs]


def evidence_separation(runs: Sequence[dict], sel: dict) -> dict:
    """Diagnostic: the depth-evidence distribution by ground truth (never an input): closed hand holding (GT carry/place) vs
    closed hand not holding (GT approach/grasp/retreat) vs open hand."""
    cache: dict = {}
    g, held, closed = [], [], []
    for r in runs:
        gp = gap_arrays(r, sel["roi"], sel["d_min"], sel["d_max"], cache)["depth"]
        m = PhaseMachine(PhaseConfig.rby1(r["side"], **{**sel["params"], "finger_gap_mode": "off"}))
        for i, obs in enumerate(observations(r)):
            _, ev = m.step(obs)
            g.append(gp[i])
            held.append(r["y"][i] in (CARRY, PLACE))
            closed.append(ev["closed"])
    g, held, closed = np.array(g), np.array(held), np.array(closed)
    q = lambda a: None if len(a) == 0 else [float(np.percentile(a, p)) for p in (5, 25, 50, 75, 95)]  # noqa: E731
    return {"percentiles_5_25_50_75_95": {"closed_holding": q(g[closed & held]), "closed_not_holding": q(g[closed & ~held]),
                                           "open": q(g[~closed])},
            "n": {"closed_holding": int((closed & held).sum()), "closed_not_holding": int((closed & ~held).sum()),
                  "open": int((~closed).sum())}}


def cmd_gap_evaluate(args: argparse.Namespace) -> None:
    spec = json.loads(pathlib.Path(args.spec).read_text())["runs"]
    base = json.loads(pathlib.Path(args.thresholds).read_text())["params"]
    gthr = json.loads(pathlib.Path(args.gap_thresholds).read_text())
    arms = {k: a["selected"] for k, a in gthr["arms"].items()}
    te = load_wrist(load_runs(args.feat, spec, "test"), args.wrist)
    assert te and all(r["split"] == "test" and r["scene"] not in EXCLUDED_SCENES for r in te)
    pa = a_predictions(args.probe, te, getattr(args, "a_key", "h4_img3+last"))
    keep = [i for i, v in enumerate(pa) if v is not None]
    runs = [te[i] for i in keep]
    pa = [pa[i] for i in keep]
    cache: dict = {}
    pred = {"A": pa, "B_T91": [predict_run(r, base) for r in runs],
            "T91b_depth": gap_arm_predictions(runs, arms["depth"], cache),
            "T91b_projection": gap_arm_predictions(runs, arms["projection"], cache)}
    # reported-only combinations: depth selection's ROI / window, both thresholds as selected
    both = {**arms["depth"], "params": {**arms["depth"]["params"], "gap_tau_proj": arms["projection"]["params"]["gap_tau_proj"],
                                        "gap_m": arms["depth"]["params"]["gap_m"]}}
    # projection uses its own ROI: evaluate the combination with the depth ROI for both numbers (no re-selection)
    pred["T91b_both_reported_only"] = gap_arm_predictions(runs, both, cache, finger_gap_source="both")
    pred["T91b_either_reported_only"] = gap_arm_predictions(runs, both, cache, finger_gap_source="either")
    res = {"step": "T91b", "t91_params": base, "gap_arms": arms, "excluded_scenes": list(EXCLUDED_SCENES),
           "test_scenes": sorted({r["scene"] for r in runs}), "n_runs": len(runs), "n_runs_without_A": len(te) - len(runs),
           **{k: summarize_groups(runs, v) for k, v in pred.items()}}
    res["verdict_T91b_depth_vs_A"] = verdict(res["A"]["all"], res["T91b_depth"]["all"])
    res["verdict_T91b_projection_vs_A"] = verdict(res["A"]["all"], res["T91b_projection"]["all"])
    res["verdict_B_T91_vs_A"] = verdict(res["A"]["all"], res["B_T91"]["all"])
    res["evidence_separation_depth"] = evidence_separation(runs, arms["depth"])
    scenes = sorted({r["scene"] for r in runs})
    res["per_scene"] = {sc: {n: {"accuracy": _acc(np.concatenate([r["y"] for r in runs if r["scene"] == sc]),
                                                  np.concatenate([p[i] for i, r in enumerate(runs) if r["scene"] == sc])),
                                 "danger": _danger(np.concatenate([r["y"] for r in runs if r["scene"] == sc]),
                                                   np.concatenate([p[i] for i, r in enumerate(runs) if r["scene"] == sc]))}
                            for n, p in pred.items()} for sc in scenes}
    res["predictions"] = {r["label"]: {"t_step": r["t_step"].tolist(), "gt": r["y"].tolist(),
                                       **{n: p[i].tolist() for n, p in pred.items()}} for i, r in enumerate(runs)}
    pathlib.Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("verdict_T91b_depth_vs_A", "verdict_T91b_projection_vs_A", "verdict_B_T91_vs_A")},
                     ensure_ascii=False, indent=1))


# ================================================================================ test evaluation


def a_predictions(probe_json: str, runs: Sequence[dict], key: str = "h4_img3+last") -> list[Optional[np.ndarray]]:
    """T77 test predictions of classifier `key` joined on (run label, client t_step); None where the run is not in the probe.
    Default = the ADOPTED 5-stage classifier (hidden layer 4, ADOPTED A9.2; lead 2026-11-03, before the test evaluation).
    `kv_L4` (layer-4 KV retrained on physics labels) is reported as a second reference."""
    tp = json.loads(pathlib.Path(probe_json).read_text())["test_predictions"]
    table = {}
    for g, t, v in zip(tp["group"], tp["t_step"], tp[key], strict=True):
        table[(g[3:] if g.startswith("cl:") else g, int(t))] = int(v)
    out = []
    for r in runs:
        v = [table.get((r["label"], int(t))) for t in r["t_step"]]
        out.append(None if any(x is None for x in v) else np.array(v, int))
    return out


def verdict(a: dict, b: dict, margin: float = 0.05) -> dict:
    """§2: B is usable = scene-mean danger(B) <= danger(A) and scene-mean accuracy(B) >= accuracy(A) - 5 pp."""
    da, db = a["danger_scene_mean"], b["danger_scene_mean"]
    aa, ab = a["accuracy_scene_mean"], b["accuracy_scene_mean"]
    c1, c2 = db <= da, ab >= aa - margin
    return {"danger_not_above_A": bool(c1), "accuracy_within_5pp_of_A": bool(c2), "B_usable": bool(c1 and c2),
            "values": {"danger_scene_mean": {"A": da, "B": db}, "accuracy_scene_mean": {"A": aa, "B": ab}}}


def cmd_evaluate(args: argparse.Namespace) -> None:
    spec = json.loads(pathlib.Path(args.spec).read_text())["runs"]
    thr = json.loads(pathlib.Path(args.thresholds).read_text())
    params = thr["params"]
    te = load_runs(args.feat, spec, "test")
    assert te and all(r["split"] == "test" and r["scene"] not in EXCLUDED_SCENES for r in te)
    pa = a_predictions(args.probe, te, getattr(args, "a_key", "h4_img3+last"))
    keep = [i for i, v in enumerate(pa) if v is not None]
    runs = [te[i] for i in keep]
    pb = [predict_run(r, params) for r in runs]
    pc = [latch_run(r) for r in runs]
    pa = [pa[i] for i in keep]
    res = {"step": "T91", "params": params, "excluded_scenes": list(EXCLUDED_SCENES),
           "test_scenes": sorted({r["scene"] for r in runs}), "n_runs": len(runs),
           "n_runs_without_A": len(te) - len(runs),
           "A_T77_kvL4": summarize_groups(runs, pa), "B_phase_machine": summarize_groups(runs, pb),
           "C_latch_only": summarize_groups(runs, pc),
           "per_scene": {sc: {n: {"accuracy": _acc(np.concatenate([r["y"] for r in runs if r["scene"] == sc]),
                                                   np.concatenate([p[i] for i, r in enumerate(runs) if r["scene"] == sc])),
                                  "danger": _danger(np.concatenate([r["y"] for r in runs if r["scene"] == sc]),
                                                    np.concatenate([p[i] for i, r in enumerate(runs) if r["scene"] == sc]))}
                             for n, p in (("A", pa), ("B", pb), ("C", pc))}
                         for sc in sorted({r["scene"] for r in runs})}}
    res["verdict"] = verdict(res["A_T77_kvL4"]["all"], res["B_phase_machine"]["all"])
    res["predictions"] = {r["label"]: {"t_step": r["t_step"].tolist(), "gt": r["y"].tolist(), "A": pa[i].tolist(),
                                       "B": pb[i].tolist(), "C": pc[i].tolist()} for i, r in enumerate(runs)}
    pathlib.Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({"verdict": res["verdict"]}, ensure_ascii=False, indent=1))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("extract")
    a.add_argument("out")
    a.add_argument("--spec", default=f"{T77}/smoke/t77_spec.json")
    a.add_argument("--labels", default=f"{T77}/t77_labels.json")
    a.add_argument("--xml", default=os.environ.get("SNAP", SNAP_DEFAULT) + "/" + XML_REL)
    a.add_argument("--split", choices=("train", "test"), default=None)
    a.add_argument("--only", nargs="+", default=None)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--shard", default=None, help="i/n: every n-th run starting at i (one process per shard)")
    w = sub.add_parser("extract-wrist")
    w.add_argument("out")
    w.add_argument("--feat", required=True, help="T91 feature dir (request rows to align with)")
    w.add_argument("--spec", default=f"{T77}/smoke/t77_spec.json")
    w.add_argument("--split", choices=("train", "test"), default=None)
    w.add_argument("--only", nargs="+", default=None)
    w.add_argument("--limit", type=int, default=0)
    w.add_argument("--shard", default=None)
    b = sub.add_parser("select")
    b.add_argument("out")
    b.add_argument("--feat", required=True)
    b.add_argument("--spec", default=f"{T77}/smoke/t77_spec.json")
    c = sub.add_parser("evaluate")
    c.add_argument("out")
    c.add_argument("--feat", required=True)
    c.add_argument("--thresholds", required=True)
    c.add_argument("--spec", default=f"{T77}/smoke/t77_spec.json")
    c.add_argument("--probe", default=f"{T77}/probe/t77_probe.json")
    c.add_argument("--a-key", default="h4_img3+last", help="T77 test_predictions key used as comparison A (adopted: h4_img3+last; reference: kv_L4)")
    g = sub.add_parser("gap-select", help="T91b: ROI / depth window / tau / m on TRAIN scenes (depth and projection arms)")
    g.add_argument("out")
    g.add_argument("--feat", required=True)
    g.add_argument("--wrist", required=True)
    g.add_argument("--t91-thresholds", required=True, help="the frozen T91 thresholds.json (kept fixed)")
    g.add_argument("--spec", default=f"{T77}/smoke/t77_spec.json")
    h = sub.add_parser("gap-evaluate", help="T91b: test metrics + verdicts from frozen thresholds (lead runs this)")
    h.add_argument("out")
    h.add_argument("--feat", required=True)
    h.add_argument("--wrist", required=True)
    h.add_argument("--thresholds", required=True, help="the frozen T91 thresholds.json")
    h.add_argument("--gap-thresholds", required=True, help="gap_thresholds.json written by gap-select")
    h.add_argument("--spec", default=f"{T77}/smoke/t77_spec.json")
    h.add_argument("--probe", default=f"{T77}/probe/t77_probe.json")
    h.add_argument("--a-key", default="h4_img3+last")
    args = ap.parse_args(argv)
    {"extract": cmd_extract, "extract-wrist": cmd_extract_wrist, "select": cmd_select,
     "evaluate": cmd_evaluate, "gap-select": cmd_gap_select, "gap-evaluate": cmd_gap_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
