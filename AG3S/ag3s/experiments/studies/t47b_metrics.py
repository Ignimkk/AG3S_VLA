"""T47b B2 — 3D mis-separation metrics from GT point labels (pure numpy, no MuJoCo).

T47's `on_target` (grounding centroid within 10 cm of the GT target body origin) cannot tell a
cluster that is the apple from one that is the apple **plus the neighbouring pear**, or from the pear
alone when the pear sits 8 cm away. These metrics read the GT label of every point instead.

Point labels are `exclusion_e2.LABELS` (`AG3S/ag3s/experiments/live/exclusion_e2.py:83`), produced by
`exclusion_e2.label_classes` + `gt_points` + `point_labeler` (nearest GT depth pixel of any camera
within 3 mm) — the same labeller T47's e2e uses. They are renamed to the T47b vocabulary:

=================  =========================  ==============================================
T47b class         `exclusion_e2.LABELS`      bodies (`exclusion_e2.label_classes`, :221-237)
=================  =========================  ==============================================
``target``         ``target``                 the prompt's fruit
``other_object``   ``distractor``             the other fruits
``destination``    ``destination``            crate
``robot``          ``robot``                  `mujoco_source.is_robot_body` (gripper included)
``table``          ``support``                table
``other``          ``other``                  anything else (shelf, floor, walls)
``shared``         ``shared``                 (not produced by `point_labeler`; kept for safety)
``unlabelled``     ``unlabelled``             no GT depth point within the radius
=================  =========================  ==============================================

Three levels of "what leaves the obstacle set" (frame metrics, `frame_b2`):

``cluster`` (primary)  the chosen cluster — `ConstraintSet.target.point_indices` (= the grounding
                       target), i.e. the geometry the exclusion ball / target-free layer is built
                       from (`pipeline.py:2598` `_target_ball(target_free_points, ...)`).
``seed``               the attention seeds `GroundingResult.seed_indices` (`target_grounding.py:1679-
                       1682`: percentile cut over the normalised fused attention, support excluded).
                       What T47's seed precision / recall read.
``ball``               cloud points inside the exclusion ball built from the chosen cluster
                       (``|p - centre| <= radius and p_z >= z_min``, `AG3S._target_ball`) — the
                       point-level image of the T13 ball carve. Optional (needs the ball).

Definitions (per frame; `T_obs` = cloud points labelled ``target`` and not in the support mask —
the same denominator as T47's seed recall, `t47_replay.py` ``n_cloud_t``):

* ``identity_correct`` — the plurality class of the chosen cluster's **labelled** points (``unlabelled``
  excluded) is ``target``, strictly (a tie is not correct). No target named -> False.
* ``chosen_cluster_purity`` — target points / all chosen points (``unlabelled`` in the denominator).
  NaN when nothing was named.
* ``target_residual`` — ``|T_obs \\ X|`` and ``|T_obs \\ X| / |T_obs|`` (NaN rate when ``T_obs`` is
  empty), X = the level's point set. No target named -> every observed target point stays.
* ``nontarget_seed_loss`` — labelled non-target points (``other_object``, ``destination``, ``robot``,
  ``table``, ``other``, ``shared``) in X, total and per class. The headline class is
  ``other_object`` (a neighbouring fruit that would stop being an obstacle).

Aggregation (`summarize`) is **per record (episode)**: frames are pooled inside a record, then every
record contributes one value; the cross-record summary is the unweighted mean over records plus the
per-record values. Frames are never counted as independent trials.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

import numpy as np

#: `exclusion_e2.LABELS`, in its order (a unit test pins the equality).
POINT_LABELS = ("target", "distractor", "support", "destination", "robot", "other", "shared",
                "unlabelled")
#: `exclusion_e2.LABELS` name -> T47b class name.
CLASS_ALIAS = {"target": "target", "distractor": "other_object", "support": "table",
               "destination": "destination", "robot": "robot", "other": "other",
               "shared": "shared", "unlabelled": "unlabelled"}
B2_CLASSES = tuple(CLASS_ALIAS[n] for n in POINT_LABELS)
NONTARGET_CLASSES = ("other_object", "destination", "robot", "table", "other", "shared")
LEVELS = ("cluster", "seed", "ball")
#: Which records start from which scene (T47.task.md, lead update 2026-10-12; `meta.json`
#: ``--episode-index``): headfix / ep1800 share episode 1800 with the selection records long / t2.
SCENE_GROUP = {"headfix": "same_scene", "ep1800": "same_scene", "ep1850": "new_scene",
               "ep1900": "new_scene", "long": "selection", "t2": "selection",
               "ep1807": "selection"}
#: `policy_record.phase_for` names; ``grasp`` (t_step >= the grasp boundary) is post-grasp.
POST_GRASP_PHASES = ("grasp",)


def phase_group(phase: Optional[str]) -> str:
    """``post_grasp`` for `POST_GRASP_PHASES`, ``pre_grasp`` for transit / approach / pre_grasp."""
    return "post_grasp" if phase in POST_GRASP_PHASES else "pre_grasp"


def scene_group(record: str) -> str:
    return SCENE_GROUP.get(record, "unknown")


# ------------------------------------------------------------------------------ per frame


def class_counts(labels: np.ndarray) -> dict[str, int]:
    """`POINT_LABELS` indices -> ``{T47b class: count}`` over every class (zeros included)."""
    lab = np.asarray(labels, np.int64).reshape(-1)
    if lab.size and (lab.min() < 0 or lab.max() >= len(POINT_LABELS)):
        raise ValueError(f"label index out of range 0..{len(POINT_LABELS) - 1}")
    c = np.bincount(lab, minlength=len(POINT_LABELS))
    return {CLASS_ALIAS[n]: int(c[i]) for i, n in enumerate(POINT_LABELS)}


def plurality(counts: dict[str, int]) -> Optional[str]:
    """The strictly largest class among the labelled ones (``unlabelled`` ignored); tie / empty -> None."""
    items = [(k, v) for k, v in counts.items() if k != "unlabelled" and v > 0]
    if not items:
        return None
    best = max(v for _, v in items)
    top = [k for k, v in items if v == best]
    return top[0] if len(top) == 1 else None


def index_mask(n: int, indices: Optional[Iterable[int]]) -> np.ndarray:
    m = np.zeros(int(n), bool)
    if indices is None:
        return m
    idx = np.asarray(list(indices) if not isinstance(indices, np.ndarray) else indices,
                     np.int64).reshape(-1)
    if idx.size and (idx.min() < 0 or idx.max() >= n):
        raise ValueError(f"indices out of range for {n} points")
    m[idx] = True
    return m


def membership_mask(cloud_points: np.ndarray, member_points: np.ndarray,
                    member_indices: Optional[np.ndarray] = None, *, radius: float = 0.005,
                    exact_tol: float = 1e-9) -> tuple[np.ndarray, str]:
    """Which cloud points belong to a geometry: ``(mask, source)``.

    ``indices`` when `member_indices` index this cloud (``cloud[idx] == points`` within `exact_tol`,
    the check `pipeline._nontarget_points` makes); otherwise ``proximity`` — cloud points within
    `radius` of a member point (an ``occluded`` / anchor geometry from an older frame carries no
    valid indices). ``empty`` when there are no member points.
    """
    cloud = np.asarray(cloud_points, np.float64).reshape(-1, 3)
    pts = np.asarray(member_points, np.float64).reshape(-1, 3)
    n = len(cloud)
    if member_indices is not None:
        idx = np.asarray(member_indices, np.int64).reshape(-1)
        if (idx.size and idx.size == len(pts) and idx.min() >= 0 and idx.max() < n
                and np.allclose(cloud[idx], pts, atol=exact_tol, rtol=0.0)):
            return index_mask(n, idx), "indices"
    if not len(pts) or not n:
        return np.zeros(n, bool), "empty"
    from scipy.spatial import cKDTree

    d, _ = cKDTree(pts).query(cloud, k=1)
    return d <= float(radius), "proximity"


def ball_mask(cloud_points: np.ndarray, centre, radius: float, z_min: float) -> np.ndarray:
    """Cloud points inside the T13 ball: ``|p - centre| <= radius`` and ``p_z >= z_min``."""
    p = np.asarray(cloud_points, np.float64).reshape(-1, 3)
    c = np.asarray(centre, np.float64).reshape(3)
    return (np.linalg.norm(p - c, axis=1) <= float(radius)) & (p[:, 2] >= float(z_min))


def _level(labels: np.ndarray, member: np.ndarray, t_obs: np.ndarray, prefix: str) -> dict:
    counts = class_counts(labels[member])
    n_t = int(t_obs.sum())
    resid = int((t_obs & ~member).sum())
    out = {f"{prefix}n": int(member.sum()),
           **{f"{prefix}_{c}": counts[c] for c in B2_CLASSES},
           f"{prefix}nontarget": int(sum(counts[c] for c in NONTARGET_CLASSES)),
           f"{prefix}target_residual_n": resid,
           f"{prefix}target_residual_rate": (resid / n_t) if n_t else float("nan")}
    return out


def frame_b2(labels: np.ndarray, *, support_mask: Optional[np.ndarray],
             seed_indices: Optional[np.ndarray], chosen_mask: Optional[np.ndarray],
             has_target: bool, ball: Optional[np.ndarray] = None) -> dict[str, Any]:
    """B2 metrics of one frame (module docstring). Every mask / index refers to the same cloud.

    `labels` ``(N,)`` `POINT_LABELS` indices; `support_mask` ``(N,)`` (None = no support); `chosen_mask`
    ``(N,)`` the chosen cluster (ignored when `has_target` is False); `ball` ``(N,)`` or None (no
    ball built -> the ball fields are NaN / None and ``ball_built`` is False).
    """
    lab = np.asarray(labels, np.int64).reshape(-1)
    n = lab.size
    sup = np.zeros(n, bool) if support_mask is None else np.asarray(support_mask, bool).reshape(-1)
    if sup.size != n:
        raise ValueError(f"support_mask has {sup.size} entries for {n} labels")
    t_obs = (lab == POINT_LABELS.index("target")) & ~sup
    chosen = (np.zeros(n, bool) if not has_target or chosen_mask is None
              else np.asarray(chosen_mask, bool).reshape(-1))
    if chosen.size != n:
        raise ValueError(f"chosen_mask has {chosen.size} entries for {n} labels")
    seeds = index_mask(n, seed_indices)

    out: dict[str, Any] = {"n_cloud": int(n), "n_target_obs": int(t_obs.sum()),
                           "has_target_b2": bool(has_target)}
    cl = _level(lab, chosen, t_obs, "chosen_")
    counts = {c: cl[f"chosen__{c}"] for c in B2_CLASSES}
    ident = plurality(counts) if has_target else None
    n_ch = cl["chosen_n"]
    out.update({
        "identity_label": ident,
        "identity_correct": bool(has_target and ident == "target"),
        "chosen_cluster_purity": (counts["target"] / n_ch) if (has_target and n_ch) else float("nan"),
        "chosen_n": n_ch,
        **{f"chosen__{c}": counts[c] for c in B2_CLASSES},
        "target_residual_n": cl["chosen_target_residual_n"],
        "target_residual_rate": cl["chosen_target_residual_rate"],
        "nontarget_seed_loss": cl["chosen_nontarget"],
        **{f"nontarget_seed_loss__{c}": counts[c] for c in NONTARGET_CLASSES},
    })
    sd = _level(lab, seeds, t_obs, "seed_")
    out.update({"seed_n": sd["seed_n"], **{f"seed__{c}": sd[f"seed__{c}"] for c in B2_CLASSES},
                "seed_nontarget": sd["seed_nontarget"],
                "seed_target_residual_n": sd["seed_target_residual_n"],
                "seed_target_residual_rate": sd["seed_target_residual_rate"]})
    if ball is None:
        out.update({"ball_built": False, "ball_n": None, "ball_nontarget": None,
                    "ball_target_residual_n": None, "ball_target_residual_rate": float("nan"),
                    **{f"ball__{c}": None for c in B2_CLASSES}})
    else:
        b = np.asarray(ball, bool).reshape(-1)
        if b.size != n:
            raise ValueError(f"ball mask has {b.size} entries for {n} labels")
        bl = _level(lab, b, t_obs, "ball_")
        out.update({"ball_built": True, "ball_n": bl["ball_n"],
                    **{f"ball__{c}": bl[f"ball__{c}"] for c in B2_CLASSES},
                    "ball_nontarget": bl["ball_nontarget"],
                    "ball_target_residual_n": bl["ball_target_residual_n"],
                    "ball_target_residual_rate": bl["ball_target_residual_rate"]})
    return out


# ------------------------------------------------------------------------- per record


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _mean(values) -> float:
    v = np.asarray([_f(x) for x in values], np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _median(values) -> float:
    v = np.asarray([_f(x) for x in values], np.float64)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else float("nan")


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def record_level(rows: Sequence[dict]) -> dict[str, Any]:
    """One record's frames (one source) -> record-level numbers (frames pooled inside the record)."""
    n = len(rows)
    has = [r for r in rows if r.get("has_target")]
    on = [bool(r.get("on_target")) for r in rows]
    ident = [bool(r.get("identity_correct")) for r in rows]
    out: dict[str, Any] = {
        "n_frames": n, "n_has_target": len(has),
        "on_target": sum(on), "identity_correct": sum(ident),
        "on_target_rate": _ratio(sum(on), n),
        "identity_correct_rate": _ratio(sum(ident), n),
        "identity_correct_given_has_target": _ratio(sum(bool(r.get("identity_correct"))
                                                        for r in has), len(has)),
        "silent_failure_rate": _ratio(sum(bool(r.get("has_target")) and not bool(r.get("on_target"))
                                          for r in rows), n),
        # the label-based counterpart: a target was named and it is not (mostly) the target
        "identity_failure_rate": _ratio(sum(bool(r.get("has_target"))
                                            and not bool(r.get("identity_correct")) for r in rows), n),
        "on_target_not_identity": sum(o and not i for o, i in zip(on, ident)),
        "identity_not_on_target": sum(i and not o for o, i in zip(on, ident)),
        "chosen_cluster_purity_mean": _mean(r.get("chosen_cluster_purity") for r in has),
        "chosen_cluster_purity_median": _median(r.get("chosen_cluster_purity") for r in has),
    }
    out["on_target_identity_disagree_rate"] = _ratio(
        out["on_target_not_identity"] + out["identity_not_on_target"], n)
    t_obs = sum(int(r.get("n_target_obs", 0)) for r in rows)
    out["n_target_obs_sum"] = t_obs
    for lvl, pre in (("cluster", ""), ("seed", "seed_"), ("ball", "ball_")):
        rows_l = rows if lvl != "ball" else [r for r in rows if r.get("ball_built")]
        t_l = sum(int(r.get("n_target_obs", 0)) for r in rows_l)
        res = sum(int(r.get(f"{pre}target_residual_n") or 0) for r in rows_l)
        out[f"{lvl}_target_residual_n_sum"] = res
        out[f"{lvl}_target_residual_ratio"] = _ratio(res, t_l)
        out[f"{lvl}_target_residual_rate_frame_mean"] = _mean(
            r.get(f"{pre}target_residual_rate") for r in rows_l)
        loss_key = "nontarget_seed_loss" if lvl == "cluster" else f"{pre}nontarget"
        cls_pre = "nontarget_seed_loss__" if lvl == "cluster" else f"{pre}_"
        out[f"{lvl}_nontarget_loss_sum"] = sum(int(r.get(loss_key) or 0) for r in rows_l)
        out[f"{lvl}_nontarget_loss_per_frame"] = _ratio(out[f"{lvl}_nontarget_loss_sum"], len(rows_l))
        for c in NONTARGET_CLASSES:
            out[f"{lvl}_nontarget_loss__{c}_sum"] = sum(int(r.get(f"{cls_pre}{c}") or 0)
                                                        for r in rows_l)
        out[f"{lvl}_frames_with_other_object_loss"] = sum(
            int(r.get(f"{cls_pre}other_object") or 0) > 0 for r in rows_l)
    out["ball_built_frames"] = sum(bool(r.get("ball_built")) for r in rows)
    return out


def _strata(row: dict) -> list[str]:
    s, p = scene_group(row["record"]), phase_group(row.get("phase"))
    return ["all", f"scene={s}", f"phase={p}", f"scene={s}|phase={p}"]


def summarize(rows: Sequence[dict]) -> dict[str, Any]:
    """``{source: {stratum: {"per_record": {record: record_level}, "record_mean": {...},
    "record_min": {...}, "record_max": {...}, "n_records": n}}}``.

    Strata: ``all``, ``scene=<same_scene|new_scene|selection>``, ``phase=<pre_grasp|post_grasp>`` and
    their crossing. ``record_mean`` is the unweighted mean of the per-record numbers (NaN records
    skipped, their count in ``n_records_finite``).
    """
    out: dict[str, Any] = {}
    for src in sorted({r["source"] for r in rows}):
        groups: dict[str, dict[str, list]] = {}
        for r in rows:
            if r["source"] != src:
                continue
            for st in _strata(r):
                groups.setdefault(st, {}).setdefault(r["record"], []).append(r)
        per_src = {}
        for st in sorted(groups):
            per_rec = {rec: record_level(rs) for rec, rs in sorted(groups[st].items())}
            keys = [k for k, v in next(iter(per_rec.values())).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
            mean, lo, hi, nfin = {}, {}, {}, {}
            for k in keys:
                v = np.asarray([_f(per_rec[r][k]) for r in per_rec], np.float64)
                fin = v[np.isfinite(v)]
                mean[k] = float(fin.mean()) if fin.size else float("nan")
                lo[k] = float(fin.min()) if fin.size else float("nan")
                hi[k] = float(fin.max()) if fin.size else float("nan")
                nfin[k] = int(fin.size)
            per_src[st] = {"n_records": len(per_rec), "per_record": per_rec, "record_mean": mean,
                           "record_min": lo, "record_max": hi, "n_records_finite": nfin}
        out[src] = per_src
    return out


def jsonable(x):
    """NaN-safe plain types for JSON (NaN stays NaN; numpy scalars -> Python)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


__all__ = ["B2_CLASSES", "CLASS_ALIAS", "LEVELS", "NONTARGET_CLASSES", "POINT_LABELS",
           "POST_GRASP_PHASES", "SCENE_GROUP", "ball_mask", "class_counts", "frame_b2",
           "index_mask", "jsonable", "membership_mask", "phase_group", "plurality",
           "record_level", "scene_group", "summarize"]
