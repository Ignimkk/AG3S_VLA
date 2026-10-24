"""Fair ATS comparison on one point set: point-level target labels on the V11AB fused / filtered cloud
(BASELINE-EVAL, Implementer F).

Why: the harness scores ATS with its *representative target points* (``target_points``, ~150 per call) against the
5 mm soft-voxel grid of all depth pixels, while a segmenter is scored with every lifted mask pixel -- two different
representations (PAPER-BASELINES-2: ATS recall 0.068 is a representation effect, not a comparison). Here every arm
labels **the same points**: the cloud V11AB itself built and filtered at that policy call (``perception_replay``
``cloud_points``; recorded per call unless the run used ``--no-clouds``), so ATS and the segmenters share the
representation and the resolution.

Per policy call ``k`` (obs_index ``t``) and cloud point ``x``:

* **GT label** -- the scene-GT instance whose visual surface is nearest to ``x`` if <= ``near`` (10 mm, SL4 / SL4b
  ``ats_calibration.label_points``: exact point-triangle distance), else ``robot`` if within ``near`` of a franka
  collision-sphere surface (the perception run's ``sph_all_*``), else ``none``. Instance -> class: BDDL target ->
  ``target``, destination -> ``destination``, other object -> ``other_object``, support fixture (table) ->
  ``support``, other fixture -> ``fixture``, robot pedestal -> ``mount``;
* **ATS** -- ``--ats-repr`` (ATS-CLOUD-DIAG; the chosen one is the ``ats_v11ab`` row = the paired reference):

  - ``cloud_target`` (default, BASELINE-EVAL F): ``cloud_target[x]`` -- the points of a target cluster found in
    *this* cloud ("fresh-only"; empty for stand-in targets and when ``has_target`` is False);
  - ``target_points``: ``x`` within ``--ats-tol`` (5 mm) of a recorded ``target_points`` point (has_target calls) --
    the V11AB target geometry, including the stand-ins (earlier-frame geometry, empty ``point_indices``);
  - ``exclusion``: as ``target_points``, plus the has_target=False calls whose manipulated geometry the ESDF still
    excludes (``manipulated_state`` set): the last has_target call's ``target_points`` of the episode carried
    forward (checked per call: its centroid == ``manipulated_centroid`` within 1e-5 m) -- the target region cuRobo
    actually received. Non-default runs add the ``cloud_target`` flags as the ``ats_v11ab_cloud_target`` row.

  Every call gets a category (``ats_calls.jsonl``; non-default runs also add ``ats_call_cat`` to every row):
  ``fresh`` (has_target, non-empty ``target_indices``), ``anchor`` (stand-in, ``manipulated_state`` visible),
  ``occluded`` (stand-in, occluded), ``excluded_no_target`` (has_target False, geometry carried forward),
  ``no_admissible`` (has_target False, nothing excluded; SL4: grounding_status ``no_admissible``), ``other``.
  ``ats_v11ab_seeds`` = ``cloud_seed`` (attention seeds, reference row);
* **segmenter arm** -- ``x`` is positive if it projects into the arm's saved mask in **any** camera that observed it
  (``visibility="observed"``: the fused point's camera bitmask ``cloud_camera_mask``, head = 1, left_wrist = 2;
  ``"depth"``: |recorded depth - camera z| <= ``depth_tol`` at the pixel; ``"any"``: no visibility test). Pixel =
  ``round(u), round(v)`` of the pinhole projection with the source run's K / T_base_cam of the same geom row
  (``lift.project_points``, OpenCV frame = the mask frame).

Point metrics per call (``G`` = GT target points, ``P`` = positive points): ``iou = |P&G| / |P|G|``,
``precision = |P&G| / |P|`` (None if P empty), ``recall = |P&G| / |G|`` (None if G empty; then the call is
``target_not_visible`` and iou is None), ``neighbour_frac = |P & (other_object | destination)| / |P|`` (+ support /
robot / none fractions). Aggregation as the harness: call -> episode mean -> mean over episodes, plus pooled sums;
paired bootstrap of every arm against ATS over episodes (2,000 resamples, seed 20261015, ``eval/metrics.py``).

Run (``.venv-baselines`` or ``.venv-ag3s``, CPU)::

    PYTHONPATH=$PWD .venv-baselines/bin/python -m Baseline.paper_baselines.eval.cloud_labeling \\
        --ats-run 20261012T225820_46a08d6ff644_sl4-perception-dev \\
        --grounding-run 20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack --run-id baseline-eval-cloud-sl3dev

    # ATS-CLOUD-DIAG fair-comparison row: the ESDF-excluded geometry matched to the cloud at 5 mm
    ... --ats-repr exclusion --ats-tol 0.005 --run-id <id>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from Baseline.paper_baselines.eval.native_metrics import (PROJECT_ROOT, arm_dir_name, load_mask_npz, make_output,
                                                         resolve_run, run_settings)

CAMERAS = ("head", "left_wrist")
CAMERA_BIT = {"head": 1, "left_wrist": 2}  # perception_replay.CAMERA_BIT
NEAR_M = 0.010  # benchmark.evaluators.ats_calibration.NEAR_M (SL4 #2 labels)
DEPTH_TOL_M = 0.010  # ats_calibration.VIS_TOL_M
VISIBILITY = ("observed", "depth", "any")
SUPPORT_FIXTURES = ("main_table", "kitchen_table", "study_table", "living_room_table", "coffee_table", "floor", "table")
POINT_CLASSES = ("target", "destination", "other_object", "fixture", "support", "mount", "robot", "none")
NEIGHBOUR = ("other_object", "destination")
ATS_ARM = "ats_v11ab|noun_attention"
ATS_SEED_ARM = "ats_v11ab_seeds|noun_attention"
ATS_CT_ARM = "ats_v11ab_cloud_target|noun_attention"  # cloud_target flags when --ats-repr is not cloud_target
PAIRED_METRICS = ("iou", "precision", "recall", "neighbour_frac")
_CLOUD_KEYS = ("cloud_points", "cloud_camera_mask", "cloud_target", "cloud_seed")
ATS_REPRS = ("cloud_target", "target_points", "exclusion")
ATS_TOL_M = 0.005  # half the V11AB fusion voxel (fusion.fusion_voxel_size_m = 0.01)
ATS_CATEGORIES = ("fresh", "anchor", "occluded", "excluded_no_target", "no_admissible", "other")
CARRY_CENTROID_TOL_M = 1e-5
_ATS_GEOM_KEYS = ("target_points", "target_points_offsets", "target_indices_offsets", "manipulated_state",
                  "manipulated_centroid", "grounding_status")


# --- pure functions ------------------------------------------------------------------------------------------------
def role_class(name: str, role: str) -> str:
    """Scene-GT instance (name, role) -> point class."""
    if role == "target":
        return "target"
    if role == "destination":
        return "destination"
    if role == "object":
        return "other_object"
    if role == "fixture":
        return "support" if name in SUPPORT_FIXTURES else "fixture"
    if role == "mount":
        return "mount"
    return "none"


def classes_from_labels(labels: Sequence[str], name_to_class: Mapping[str, str]) -> np.ndarray:
    """``label_points`` names (instance / robot / none) -> class codes (index into :data:`POINT_CLASSES`)."""
    lut = {**{n: POINT_CLASSES.index(c) for n, c in name_to_class.items()},
           "robot": POINT_CLASSES.index("robot"), "none": POINT_CLASSES.index("none")}
    return np.array([lut[str(x)] for x in labels], np.int8)


def mask_positive(points: np.ndarray, cam_bits: np.ndarray | None, masks: Mapping[str, np.ndarray],
                  K: Mapping[str, np.ndarray], T: Mapping[str, np.ndarray], *, visibility: str = "observed",
                  depth_m: Mapping[str, np.ndarray] | None = None, depth_tol: float = DEPTH_TOL_M) -> np.ndarray:
    """``(N,)`` bool: the point projects into ``masks[cam]`` for at least one camera that sees it."""
    from Baseline.paper_baselines import lift as L

    if visibility not in VISIBILITY:
        raise ValueError(f"visibility must be one of {VISIBILITY}")
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    out = np.zeros(len(pts), bool)
    if not len(pts):
        return out
    for cam, m in masks.items():
        m = np.asarray(m, bool)
        H, W = m.shape
        u, v, z = L.project_points(pts, K[cam], T[cam])
        ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
        ok = (z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if visibility == "observed":
            if cam_bits is None:
                raise ValueError("visibility='observed' needs the cloud camera bitmask")
            ok &= (np.asarray(cam_bits, np.uint8) & CAMERA_BIT[cam]) != 0
        elif visibility == "depth":
            d = np.asarray(depth_m[cam], np.float64)
            dd = np.full(len(pts), np.inf)
            dd[ok] = np.abs(d[vi[ok], ui[ok]] - z[ok])
            ok &= dd <= depth_tol
        hit = np.zeros(len(pts), bool)
        hit[ok] = m[vi[ok], ui[ok]]
        out |= hit
    return out


def point_metrics(pred: np.ndarray, cls: np.ndarray) -> dict:
    """Point-level metrics of a positive set ``pred (N,) bool`` against class codes ``cls (N,)``."""
    p = np.asarray(pred, bool)
    c = np.asarray(cls, np.int64)
    g = c == POINT_CLASSES.index("target")
    tp, n_p, n_g = int(np.count_nonzero(p & g)), int(p.sum()), int(g.sum())
    union = int(np.count_nonzero(p | g))
    out: dict[str, Any] = {"n_points": int(len(p)), "n_pred": n_p, "n_gt": n_g, "tp": tp, "fp": n_p - tp, "fn": n_g - tp,
                           "iou": tp / union if n_g > 0 else None, "precision": tp / n_p if n_p > 0 else None,
                           "recall": tp / n_g if n_g > 0 else None, "found": n_p > 0}
    for k in POINT_CLASSES:
        out[f"pred_{k}"] = int(np.count_nonzero(p & (c == POINT_CLASSES.index(k))))
    n_nb = sum(out[f"pred_{k}"] for k in NEIGHBOUR)
    out["pred_neighbour"] = n_nb
    out["neighbour_frac"] = n_nb / n_p if n_p > 0 else None
    for k in ("support", "robot", "none", "fixture"):
        out[f"{k}_frac"] = out[f"pred_{k}"] / n_p if n_p > 0 else None
    if n_g == 0:
        out["missing_reason"] = "target_not_visible"
    return out


def ragged(P: Mapping[str, np.ndarray], key: str, k: int) -> np.ndarray:
    o = P[key + "_offsets"]
    return P[key][o[k]:o[k + 1]]


def flags_near(points: np.ndarray, geom: np.ndarray | None, tol: float) -> np.ndarray:
    """``(N,)`` bool: the cloud point lies within ``tol`` (+1e-9) of a point of ``geom`` (None / empty -> none)."""
    from scipy.spatial import cKDTree

    pts = np.asarray(points, np.float64).reshape(-1, 3)
    f = np.zeros(len(pts), bool)
    if geom is not None and len(geom) and len(pts):
        d, _ = cKDTree(np.asarray(geom, np.float64).reshape(-1, 3)).query(pts, distance_upper_bound=tol + 1e-9)
        f = np.isfinite(d) & (d <= tol + 1e-9)
    return f


def ats_episode_geometry(has_target: Sequence[bool], n_target_indices: Sequence[int], manip_state: Sequence[str],
                         target_points: Sequence[np.ndarray], manip_centroid: Sequence[np.ndarray] | None = None) -> list:
    """Per call of one episode (in order): ATS call category and target geometries.

    ``tp`` = the recorded ``target_points`` (has_target calls, else None); ``excl`` = the geometry the ESDF excludes:
    ``tp`` on has_target calls, on has_target=False calls with ``manipulated_state`` set the last has_target call's
    ``target_points`` (``carried_from`` = that call; ``carry_centroid_err_m`` = |its mean - manipulated_centroid|).
    """
    out, last, last_k = [], None, None
    for k, has in enumerate(has_target):
        state = str(manip_state[k])
        tp = np.asarray(target_points[k], np.float64).reshape(-1, 3)
        rec: dict[str, Any] = {"tp": None, "excl": None, "carried_from": None, "carry_centroid_err_m": None}
        if has:
            rec["tp"] = rec["excl"] = tp
            last, last_k = tp, k
            if int(n_target_indices[k]) > 0:
                rec["cat"] = "fresh"
            else:
                rec["cat"] = {"visible": "anchor", "occluded": "occluded"}.get(state, "other")
        elif state and last is not None:
            rec.update(cat="excluded_no_target", excl=last, carried_from=last_k)
            if manip_centroid is not None:
                c = np.asarray(manip_centroid[k], np.float64)
                rec["carry_centroid_err_m"] = float(np.linalg.norm(last.mean(0) - c)) if len(last) else None
        else:
            rec["cat"] = "no_admissible"
        out.append(rec)
    return out


def aggregate_arm(rows: Sequence[Mapping]) -> dict:
    from Baseline.paper_baselines.eval import metrics as M

    out: dict[str, Any] = {"n_calls": len(rows), "n_episodes": len({r["episode_id"] for r in rows})}
    for m in PAIRED_METRICS + ("support_frac", "robot_frac", "none_frac"):
        out[m] = M.aggregate(rows, m)
    valid = [r for r in rows if r.get("iou") is not None]
    sums = {k: int(sum(r[k] for r in valid)) for k in ("tp", "fp", "fn", "n_pred", "n_gt", "pred_neighbour")}
    out["pooled"] = {**sums, "iou": sums["tp"] / max(1, sums["tp"] + sums["fp"] + sums["fn"]),
                     "precision": sums["tp"] / sums["n_pred"] if sums["n_pred"] else None,
                     "recall": sums["tp"] / sums["n_gt"] if sums["n_gt"] else None,
                     "neighbour_frac": sums["pred_neighbour"] / sums["n_pred"] if sums["n_pred"] else None}
    out["found_rate"] = M.found_rate(bool(r["found"]) for r in rows)
    return out


def summarize(rows: Sequence[Mapping], ref_arm: str = ATS_ARM) -> dict:
    from Baseline.paper_baselines.eval import metrics as M

    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    out: dict[str, Any] = {"arms": {a: aggregate_arm(rs) for a, rs in sorted(by.items())}, "paired_vs_ats": {}}
    if ref_arm in by:
        for a, rs in sorted(by.items()):
            if a == ref_arm:
                continue
            out["paired_vs_ats"][a] = {m: M.paired_bootstrap(M.episode_means(rs, m), M.episode_means(by[ref_arm], m))
                                       for m in PAIRED_METRICS}
    return out


def summarize_by_category(rows: Sequence[Mapping], arms: Sequence[str] = (ATS_ARM, ATS_CT_ARM)) -> dict:
    """``{category: {"n_calls", arm: aggregate_arm}}`` over rows carrying ``ats_call_cat`` (non-default runs)."""
    out: dict[str, Any] = {}
    for cat in ATS_CATEGORIES:
        sel = [r for r in rows if r.get("ats_call_cat") == cat]
        if not sel:
            continue
        out[cat] = {"n_calls": len({(r["episode_id"], r["call_k"]) for r in sel})}
        for a in arms:
            rs = [r for r in sel if r["arm"] == a]
            if rs:
                out[cat][a] = aggregate_arm(rs)
    return out


# --- per episode -------------------------------------------------------------------------------------------------------
class PerceptionEpisode:
    """The arrays of one ``perception/<stem>_perception.npz`` this evaluation needs (clouds, flags, robot spheres)."""

    def __init__(self, path: pathlib.Path):
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            if "cloud_points" not in files:
                raise LookupError(f"{path.name}: no cloud_points (perception run recorded with --no-clouds)")
            keep = ["obs_index", "geom_row", "has_target", "sph_all_center", "sph_all_radius"]
            keep += [k for c in _CLOUD_KEYS for k in (c, c + "_offsets")]
            keep += list(_ATS_GEOM_KEYS)
            self.P = {k: z[k] for k in keep if k in files}
        self.path = path
        if not np.array_equal(self.P["cloud_points_offsets"], self.P["cloud_camera_mask_offsets"]):
            raise ValueError(f"{path.name}: cloud_camera_mask is not aligned with cloud_points")

    @property
    def obs_index(self) -> np.ndarray:
        return np.asarray(self.P["obs_index"]).astype(np.int64)

    def call(self, k: int) -> dict:
        P = self.P
        has = bool(P["has_target"][k])
        return {"points": ragged(P, "cloud_points", k).astype(np.float64), "cam_bits": ragged(P, "cloud_camera_mask", k),
                "ats": ragged(P, "cloud_target", k).astype(bool) & has, "seed": ragged(P, "cloud_seed", k).astype(bool),
                "has_target": has, "geom_row": int(P["geom_row"][k]),
                "robot_c": np.asarray(P["sph_all_center"][k], np.float64) if "sph_all_center" in P else None,
                "robot_r": np.asarray(P["sph_all_radius"], np.float64) if "sph_all_radius" in P else None}

    def ats_geometry(self) -> list:
        """:func:`ats_episode_geometry` of this episode (+ ``grounding_status`` / ``manipulated_state`` / sizes)."""
        P = self.P
        missing = [k for k in _ATS_GEOM_KEYS[:5] if k not in P]
        if missing:
            raise LookupError(f"{self.path.name}: no {missing} (needed for --ats-repr target_points / exclusion)")
        n = len(P["obs_index"])
        ti = np.diff(P["target_indices_offsets"])
        geo = ats_episode_geometry(P["has_target"], ti, P["manipulated_state"],
                                   [ragged(P, "target_points", k) for k in range(n)], P["manipulated_centroid"])
        for k, g in enumerate(geo):
            g.update(grounding_status=str(P["grounding_status"][k]) if "grounding_status" in P else None,
                     manipulated_state=str(P["manipulated_state"][k]), n_target_indices=int(ti[k]),
                     n_target_points=int(np.diff(P["target_points_offsets"])[k]))
        return geo


def robot_surface_distance(points: np.ndarray, centres: np.ndarray, radii: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """``(N,)`` min over robot spheres of ``|x - c| - r`` (chunked; 61 spheres)."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    c = np.asarray(centres, np.float64).reshape(-1, 3)
    r = np.asarray(radii, np.float64).reshape(-1)
    out = np.empty(len(p))
    for s in range(0, len(p), chunk):
        d = np.linalg.norm(p[s:s + chunk, None, :] - c[None], axis=2) - r[None]
        out[s:s + chunk] = d.min(1)
    return out


def arm_from_dir(adir: str) -> str:
    """Mask dir name -> arm label (inverse of ``native_metrics.arm_dir_name``; providers use ``_``, modes ``-``)."""
    spec, _, pm = adir.partition("__")
    name, dash, mode = spec.partition("-")
    return f"{name}:{mode}|{pm}" if dash else f"{name}|{pm}"


def gt_classes(points: np.ndarray, scene, t: int, robot_c, robot_r, near: float = NEAR_M) -> np.ndarray:
    """Class code of every point (nearest scene instance within ``near``, else robot, else none)."""
    from benchmark.evaluators.ats_calibration import label_points

    row = scene.frame_row(int(t))
    names = [x.name for x in scene.instances()]
    D = scene.exact_distance_near(points, row, max_distance=near) if len(points) else np.zeros((0, len(names)))
    rob = None
    if robot_c is not None and robot_r is not None and len(points):
        rob = robot_surface_distance(points, robot_c, robot_r)
    labels = label_points(D, names, rob, near=near)
    n2c = {x.name: role_class(x.name, x.role) for x in scene.instances()}
    return classes_from_labels(labels, n2c)


def run(args) -> pathlib.Path:
    ats_run = resolve_run(args.ats_run)
    grun = resolve_run(args.grounding_run) if args.grounding_run else None
    st = run_settings(grun) if grun else {}
    source = resolve_run(args.source_run or st.get("source_run") or _perception_source(ats_run))
    scene_arg = args.scene_gt or st.get("_scene_gt_abs") or st.get("scene_gt")
    if not scene_arg:
        raise SystemExit("--scene-gt is required (GT labels)")
    scene_dir = pathlib.Path(scene_arg)
    scene_dir = scene_dir if scene_dir.is_absolute() else (PROJECT_ROOT / scene_dir).resolve()
    mask_arms = []
    if grun is not None and (grun / "masks").is_dir():
        mask_arms = sorted(d.name for d in (grun / "masks").iterdir() if d.is_dir())
        if args.arms:
            want = {arm_dir_name(a) if "|" in a else a for a in args.arms}
            mask_arms = [a for a in mask_arms if a in want]
    stems = sorted(p.name[: -len("_perception.npz")] for p in (ats_run / "perception").glob("*_perception.npz"))
    if args.episodes:
        stems = [s for s in stems if s in set(args.episodes)]
    if args.limit_episodes:
        stems = stems[: args.limit_episodes]
    settings = {"ats_run": str(ats_run), "grounding_run": str(grun) if grun else None, "source_run": str(source),
                "scene_gt": str(scene_dir), "mask_arms": mask_arms, "episodes": stems, "visibility": args.visibility,
                "near_m": args.near, "depth_tol_m": args.depth_tol, "ats_repr": args.ats_repr,
                "ats_tol_m": args.ats_tol, "argv": sys.argv}
    if args.ats_repr not in ATS_REPRS:
        raise SystemExit(f"--ats-repr must be one of {ATS_REPRS}")
    if not args.ats_tol >= 0:
        raise SystemExit("--ats-tol must be >= 0")
    default_repr = args.ats_repr == "cloud_target"
    # a perception run without clouds: report and stop before creating an output directory
    first = ats_run / "perception" / f"{stems[0]}_perception.npz"
    try:
        pe0 = PerceptionEpisode(first)
    except LookupError as e:
        print(json.dumps({"no_clouds": str(e), "proposal": NO_CLOUD_PROPOSAL}, indent=1))
        raise SystemExit(2)
    if not default_repr:
        pe0.ats_geometry()  # LookupError before an output directory exists
    out = make_output(args.run_id, args.out_dir, settings, "Baseline/paper_baselines/eval/cloud_labeling.py")
    from benchmark.evaluators.libero_scene_gt import SceneGT

    rows_f = (out / "calls.jsonl").open("x")
    cats_f = (out / "ats_calls.jsonl").open("x")
    rows: list[dict] = []
    t0 = time.time()
    for stem in stems:
        pe = PerceptionEpisode(ats_run / "perception" / f"{stem}_perception.npz")
        try:
            geo = pe.ats_geometry()
        except LookupError:
            if not default_repr:
                raise
            geo = None  # an older perception run: cloud_target only, no categories
        scene = SceneGT(scene_dir / f"scene_{stem}.npz")
        geom_path = source / "episodes" / f"{stem}_geom.npz"
        with np.load(geom_path, allow_pickle=False) as g:
            Kall = {c: g[f"K/{c}"] for c in args.cameras}
            Tall = {c: g[f"T_base_cam/{c}"] for c in args.cameras}
            Dall = {c: g[f"depth_mm/{c}"] for c in args.cameras} if args.visibility == "depth" else None
        obs = pe.obs_index
        calls = range(len(obs)) if not args.limit_calls else range(min(args.limit_calls, len(obs)))
        dump_calls = set(_pick_dump(len(obs), args.dump_per_episode)) if args.dump_figure_data else set()
        for k in calls:
            t = int(obs[k])
            c = pe.call(k)
            cls = gt_classes(c["points"], scene, t, c["robot_c"], c["robot_r"], args.near)
            base = {"episode_id": stem, "call_k": int(k), "obs_index": t, "has_target": c["has_target"]}
            g = geo[k] if geo is not None else None
            if default_repr:  # BASELINE-EVAL F rows, byte for byte
                arm_flags = {ATS_ARM: c["ats"], ATS_SEED_ARM: c["seed"]}
            else:
                err = g["carry_centroid_err_m"]
                if args.ats_repr == "exclusion" and g["carried_from"] is not None \
                        and (err is None or err >= CARRY_CENTROID_TOL_M):
                    raise ValueError(f"{stem} call {k}: carried-forward geometry (call {g['carried_from']}) centroid is "
                                     f"{err} m from manipulated_centroid -- not what the ESDF got")
                geom = g["tp"] if args.ats_repr == "target_points" else g["excl"]
                base["ats_call_cat"] = g["cat"]
                arm_flags = {ATS_ARM: flags_near(c["points"], geom, args.ats_tol), ATS_CT_ARM: c["ats"],
                             ATS_SEED_ARM: c["seed"]}
            r_ = c["geom_row"]
            K = {cam: Kall[cam][r_] for cam in args.cameras}
            T = {cam: Tall[cam][r_] for cam in args.cameras}
            D = {cam: Dall[cam][r_].astype(np.float64) / 1000.0 for cam in args.cameras} if Dall is not None else None
            for adir in mask_arms:
                p = grun / "masks" / adir / f"{stem}_{t:04d}.npz"
                if not p.exists():
                    continue
                masks = {cam: m for cam, m in load_mask_npz(p).items() if cam in args.cameras}
                arm = arm_from_dir(adir)
                arm_flags[arm] = mask_positive(c["points"], c["cam_bits"], masks, K, T, visibility=args.visibility,
                                               depth_m=D, depth_tol=args.depth_tol)
            for arm, flags in arm_flags.items():
                row = {**base, "arm": arm, **point_metrics(flags, cls)}
                rows_f.write(json.dumps(row) + "\n")
                rows.append(row)
            crow = {"episode_id": stem, "call_k": int(k), "obs_index": t, "has_target": c["has_target"],
                    "ats_repr": args.ats_repr, "n_cloud": int(len(c["points"])), "n_cloud_target": int(c["ats"].sum()),
                    "n_ats_pred": int(np.count_nonzero(arm_flags[ATS_ARM]))}
            if g is not None:
                crow.update(ats_call_cat=g["cat"], **{q: g[q] for q in (
                    "grounding_status", "manipulated_state", "n_target_indices", "n_target_points", "carried_from",
                    "carry_centroid_err_m")})
            cats_f.write(json.dumps(crow) + "\n")
            if k in dump_calls:
                d = out / "figure_dump"
                d.mkdir(exist_ok=True)
                np.savez_compressed(d / f"cloud_{stem}_{t:04d}.npz", points=c["points"].astype(np.float32),
                                    gt_class=cls, cam_bits=c["cam_bits"], point_classes=np.array(POINT_CLASSES),
                                    arms=np.array(list(arm_flags)), flags=np.stack([arm_flags[a] for a in arm_flags]),
                                    geom_row=np.int64(r_), meta_json=np.array(json.dumps({**base, "source_run": str(source)})))
        rows_f.flush()
        cats_f.flush()
        print(f"[cloud_labeling] {stem}: {len(calls)} calls, {time.time() - t0:.1f}s", flush=True)
    rows_f.close()
    cats_f.close()
    summary = summarize(rows)
    if not default_repr:
        summary["ats_by_category"] = summarize_by_category(rows)
    summary.update(settings=settings, definitions=__doc__)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"[cloud_labeling] wrote {out}")
    return out


NO_CLOUD_PROPOSAL = ("the perception run did not record clouds (--no-clouds). Alternatives (not implemented): "
                     "(a) re-run perception_replay on the same frames with clouds (CPU/GPU, same commit); "
                     "(b) dilate the ATS target points to the 5 mm eval voxels they touch (same voxel set as the "
                     "segmenters) and score both on the harness grid; (c) label every eval voxel instead of cloud "
                     "points, giving ATS the voxels within one V11AB fusion voxel (1 cm) of a target point.")


def _perception_source(ats_run: pathlib.Path) -> str:
    man = json.loads((ats_run / "run_manifest.json").read_text())
    return man["settings"]["sources"]["dev"]["run_dir"]


def _pick_dump(n: int, per_episode: int) -> list[int]:
    if per_episode <= 0 or n == 0:
        return []
    return sorted(set(np.round(np.linspace(0, n - 1, per_episode + 2)[1:-1]).astype(int).tolist())) if per_episode > 1 \
        else [n // 2]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ats-run", required=True, help="perception_replay run (cloud_points recorded)")
    ap.add_argument("--grounding-run", help="offline_grounding run with masks/ (--save-masks)")
    ap.add_argument("--source-run", help="default: grounding run settings, else the perception run's dev source")
    ap.add_argument("--scene-gt", help="default: grounding run provenance")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    ap.add_argument("--visibility", default="observed", choices=VISIBILITY)
    ap.add_argument("--near", type=float, default=NEAR_M)
    ap.add_argument("--depth-tol", type=float, default=DEPTH_TOL_M)
    ap.add_argument("--ats-repr", default="cloud_target", choices=ATS_REPRS,
                    help="ATS positive set (ATS-CLOUD-DIAG): cloud_target = fresh cluster only (BASELINE-EVAL F, "
                         "default); target_points / exclusion = cloud points within --ats-tol of the target geometry")
    ap.add_argument("--ats-tol", type=float, default=ATS_TOL_M,
                    help="m, for --ats-repr target_points / exclusion (exact: 1e-6)")
    ap.add_argument("--episodes", nargs="*")
    ap.add_argument("--limit-episodes", type=int)
    ap.add_argument("--limit-calls", type=int)
    ap.add_argument("--dump-figure-data", action="store_true", help="figure_dump/cloud_<stem>_<t>.npz (points + labels)")
    ap.add_argument("--dump-per-episode", type=int, default=1)
    ap.add_argument("--run-id")
    ap.add_argument("--out-dir")
    return ap


def main(argv=None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
