"""T56 — self scores S0 / S1 / S2 per candidate and frame, the labels they come from, and GT checks.

`T56.task.md` 사전 고정 2. All scores read only the stored replay geometry (`t56_replay.Geometry`):
the chosen cluster, the frame's filtered cloud and the pipeline's support mask, and the robot pads.
GT point labels are read only by `label_check` (HELD-event label error), never by a score.

Labels (one per record; frames are positions of the record's replayed frames, in time order)

``S0`` (T51 `self_identity`, the baseline)
    Event = sim ``both_contact`` (`t51_selflabel.detect_grasp`, the first frame both fingers of a hand
    touch the target and that hand later holds it). Ball centre = midpoint of that hand's two pads at
    the event (world, fixed), radius = `t51_selflabel.object_radius` + `SELF_BALL_MARGIN` (0.02 m).
    Labeled frames = frames **before** the event. Frame score = ``self_identity_correct`` (strictly more
    chosen points inside the ball than outside; no output → 0). The T51 definition, unchanged.
``S1`` (main)
    Event = `grasp_latch` **HELD** (`t56_held`). Ball centre = midpoint of the HELD hand's pads at the
    HELD frame (world, fixed — the "object did not move before the grasp" assumption, as S0), radius =
    object radius from the pad separation + ``margin`` (dev-fixed). Labeled frames = frames before the
    HELD frame. Two continuous components per (candidate, frame):

    * ``inclusion`` = share of the chosen cluster's points inside the ball (no output → 0);
    * ``coverage``  = share of the label cloud's **non-support** points inside the ball that have a
      chosen point within ``cover_tol`` (no output → 0). A frame whose ball holds no non-support point
      is not scorable (counted, left out).

    Frame score = ``combine(inclusion, coverage)`` (dev-fixed: ``harmonic`` / ``geometric`` / ``min`` /
    ``product`` / ``mean``) — reported together so neither a large ball nor a large cluster wins by
    itself — then optionally normalised per frame by the best candidate of the group (``frame_max``).
``S2`` (auxiliary) = S1 + a post-HELD extension
    After the HELD frame and **while the latch stays HELD** (the first non-HELD frame — release, slip,
    revoke — stops the extension for the rest of the episode; it is not restarted by a later HELD),
    a frame is labeled only when there is evidence that points **move with the hand**:

    * the hand (pad midpoint, world) moved at least ``min_hand_motion`` from where it was at the HELD
      frame (without motion, a static neighbour also keeps its hand coordinates);
    * the label set = non-support points within the hand ball (same radius as S1, centred on the
      current pad midpoint) whose hand-frame coordinates (finger-body rotation, pad-midpoint origin)
      are within ``comove_tol`` of the reference set — the hand-frame points inside the hand ball at
      the HELD frame (``reference = held``) or at the previous labeled frame (``previous``);
    * fewer than ``min_points`` co-moving points → the object is not following: stop the extension
      (``stop_on_lost``) or only skip the frame.

    Being near the hand alone never makes a point positive. Components on an extension frame:
    ``inclusion`` = share of chosen points within ``comove_tol`` of a label point, ``coverage`` = share of
    label points with a chosen point within ``cover_tol``.

Label cloud of a frame = the stored cloud with the smallest content digest among that frame's calls
(in the ``current`` condition every call of a frame sees the same cloud).
Episode score of a candidate = mean of its frame scores over the labeled, scorable frames; the episode
"coverage component" (S1's) is the tie-break input of `t56_select`.

No neighbour-negative label is used anywhere (`T56.task.md`).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t51_selflabel as sl

COMBINES = ("harmonic", "geometric", "min", "product", "mean")
NORMALIZES = ("none", "frame_max")
REFERENCES = ("held", "previous")
VARIANTS = ("S0", "S1", "S2")
HAND_INDEX = {"left": 0, "right": 1}
TARGET_LABEL = 0  # `t47b_metrics.POINT_LABELS.index("target")`


@dataclasses.dataclass(frozen=True)
class S1Params:
    margin: float = 0.02
    combine: str = "harmonic"
    normalize: str = "none"
    cover_tol: float = 0.005

    def __post_init__(self):
        if self.combine not in COMBINES:
            raise ValueError(f"combine {self.combine!r} not in {COMBINES}")
        if self.normalize not in NORMALIZES:
            raise ValueError(f"normalize {self.normalize!r} not in {NORMALIZES}")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class S2Params:
    comove_tol: float = 0.01
    min_hand_motion: float = 0.03
    reference: str = "held"
    min_points: int = 5
    stop_on_lost: bool = True

    def __post_init__(self):
        if self.reference not in REFERENCES:
            raise ValueError(f"reference {self.reference!r} not in {REFERENCES}")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def combine(inc: np.ndarray, cov: np.ndarray, how: str) -> np.ndarray:
    i, c = np.asarray(inc, np.float64), np.asarray(cov, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if how == "harmonic":
            s = np.where(i + c > 0, 2 * i * c / np.where(i + c > 0, i + c, 1.0), 0.0)
        elif how == "geometric":
            s = np.sqrt(np.clip(i, 0, None) * np.clip(c, 0, None))
        elif how == "min":
            s = np.minimum(i, c)
        elif how == "product":
            s = i * c
        elif how == "mean":
            s = 0.5 * (i + c)
        else:
            raise ValueError(f"combine {how!r}")
    return np.where(np.isfinite(i) & np.isfinite(c), s, np.nan)


def normalize(scores: np.ndarray, how: str) -> np.ndarray:
    """`scores` (n_cand, n_frames); ``frame_max`` divides each frame by its best candidate (> 0)."""
    s = np.asarray(scores, np.float64)
    if how == "none":
        return s
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.nanmax(np.where(np.isfinite(s), s, -np.inf), axis=0)
        return np.where(np.isfinite(s), np.where(m > 0, s / np.where(m > 0, m, 1.0), 0.0), np.nan)


def to_base(T_world_base: np.ndarray, p_world: np.ndarray) -> np.ndarray:
    return sl.to_frame(T_world_base, p_world)


def to_world(T_world_base: np.ndarray, p_base: np.ndarray) -> np.ndarray:
    T = np.asarray(T_world_base, np.float64)
    p = np.asarray(p_base, np.float64).reshape(-1, 3)
    return p @ T[:3, :3].T + T[:3, 3]


def _nn_within(query: np.ndarray, ref: np.ndarray, tol: float) -> np.ndarray:
    """``(len(query),)`` bool: a `ref` point within `tol` of each query point."""
    q = np.asarray(query, np.float64).reshape(-1, 3)
    r = np.asarray(ref, np.float64).reshape(-1, 3)
    if not len(q):
        return np.zeros(0, bool)
    if not len(r):
        return np.zeros(len(q), bool)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(r).query(q, k=1)
    return d <= float(tol)


# --------------------------------------------------------------------------------- labels


def label_cloud(geom, frame: int) -> Optional[dict]:
    """The frame's label cloud (module doc): the stored cloud with the smallest digest."""
    cache = geom.__dict__.setdefault("_label_cloud_id", {})
    if int(frame) not in cache:
        cf = np.flatnonzero(np.asarray(geom.z["cloud_frame"]) == int(frame))
        if not cf.size:
            cache[int(frame)] = None
        else:
            dig = [str(geom.z["cloud_digest"][k]) for k in cf]
            cache[int(frame)] = int(cf[int(np.argmin(dig))])
    cid = cache[int(frame)]
    return None if cid is None else geom.cloud(cid)


def _pads(geom, pos: int, hand: str) -> np.ndarray:
    return np.asarray(geom.z["pads_world"][pos, HAND_INDEX[hand]], np.float64)


def ball_from_pads(geom, pos: int, hand: str, margin: float) -> dict:
    p1, p2 = _pads(geom, pos, hand)
    r_obj = sl.object_radius(p1, p2, float(geom.z["pad_half"][HAND_INDEX[hand]]))
    return {"centre_world": 0.5 * (p1 + p2), "r_obj": r_obj, "radius": r_obj + float(margin)}


def s0_label(geom) -> Optional[dict]:
    """T51's both_contact label on the replayed frames (None when no event)."""
    code = {-1: None, 0: "left", 1: "right"}
    both = [code[int(c)] for c in geom.z["both_hand"]]
    held = [code[int(c)] for c in geom.z["held_hand"]]
    ev = sl.detect_grasp(both, held, event="both_contact")
    if ev is None:
        return None
    pos, hand = ev
    b = ball_from_pads(geom, pos, hand, sl.SELF_BALL_MARGIN)
    return {"variant": "S0", "event_pos": int(pos), "event_frame": geom.frames[pos], "hand": hand,
            **b, "labeled": list(range(pos))}


def s1_label(geom, held_event, p: S1Params) -> Optional[dict]:
    """S1 ball label from a `t56_held.HeldEvent` (None when the record has no HELD event)."""
    if held_event is None or not held_event.has_event or held_event.hand is None:
        return None
    if int(held_event.pos) not in geom.fpos:
        raise ValueError(f"HELD frame {held_event.pos} was not replayed")
    pos = geom.fpos[int(held_event.pos)]
    b = ball_from_pads(geom, pos, held_event.hand, p.margin)
    return {"variant": "S1", "event_pos": int(pos), "event_frame": int(held_event.pos),
            "hand": held_event.hand, **b, "labeled": list(range(pos))}


def hand_frame(geom, pos: int, hand: str) -> tuple[np.ndarray, np.ndarray]:
    """``(R_world_hand, origin_world)``: finger-body rotation and pad midpoint."""
    R = np.asarray(geom.z["finger_R_world"][pos, HAND_INDEX[hand]], np.float64)
    return R, _pads(geom, pos, hand).mean(axis=0)


def _hand_ball_points(geom, pos: int, hand: str, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """``(base points, hand-frame points)`` of the label cloud's non-support points within `radius`
    of the pad midpoint at `pos`."""
    cl = label_cloud(geom, geom.frames[pos])
    if cl is None or not len(cl["pts"]):
        return np.zeros((0, 3)), np.zeros((0, 3))
    Twb = np.asarray(geom.z["T_world_base"][pos], np.float64)
    R, o = hand_frame(geom, pos, hand)
    o_base = to_base(Twb, o)
    keep = (~np.asarray(cl["support"], bool)) & (np.linalg.norm(cl["pts"] - o_base, axis=1) <= radius)
    pb = cl["pts"][keep]
    pw = to_world(Twb, pb)
    return pb, (pw - o) @ R


def s2_extension(geom, held_event, s1: Optional[dict], p: S2Params) -> dict:
    """S2's post-HELD label frames: ``{pos: base points of the co-moving label set}`` + a trace."""
    out: dict[str, Any] = {"frames": {}, "trace": [], "stopped": None}
    if s1 is None:
        return out
    hand, h, radius = s1["hand"], s1["event_pos"], s1["radius"]
    _, ref = _hand_ball_points(geom, h, hand, radius)
    out["n_reference"] = int(len(ref))
    if len(ref) < p.min_points:
        out["stopped"] = {"pos": h, "reason": "reference_too_small"}
        return out
    o_h = hand_frame(geom, h, hand)[1]
    for pos in range(h + 1, len(geom.frames)):
        fr = geom.frames[pos]
        st = held_event.state_at(fr)
        if st != "held":
            out["stopped"] = {"pos": pos, "reason": f"latch_{st}"}
            break
        o = hand_frame(geom, pos, hand)[1]
        motion = float(np.linalg.norm(o - o_h))
        if motion < p.min_hand_motion:
            out["trace"].append({"pos": pos, "skip": "no_hand_motion", "motion_m": motion})
            continue
        pb, ph = _hand_ball_points(geom, pos, hand, radius)
        co = _nn_within(ph, ref, p.comove_tol)
        n_co = int(co.sum())
        if n_co < p.min_points:
            out["trace"].append({"pos": pos, "skip": "not_following", "n_comove": n_co,
                                 "n_ball": int(len(pb)), "motion_m": motion})
            if p.stop_on_lost:
                out["stopped"] = {"pos": pos, "reason": "not_following"}
                break
            continue
        out["frames"][pos] = pb[co]
        out["trace"].append({"pos": pos, "n_comove": n_co, "n_ball": int(len(pb)), "motion_m": motion})
        if p.reference == "previous":
            ref = ph[co]
    return out


# --------------------------------------------------------------------------------- scores


def ball_components(geom, cand: str, pos: int, centre_base: np.ndarray, radius: float,
                    cover_tol: float) -> tuple[float, float]:
    """``(inclusion, coverage)`` of one candidate on one ball-labeled frame (NaN coverage when the
    ball holds no non-support point of the label cloud)."""
    fr = geom.frames[pos]
    cl = label_cloud(geom, fr)
    if cl is None:
        return float("nan"), float("nan")
    inball = (~np.asarray(cl["support"], bool)) & (np.linalg.norm(cl["pts"] - centre_base, axis=1) <= radius)
    if not inball.any():
        return float("nan"), float("nan")
    P = geom.chosen_points(cand, fr)
    if not len(P):
        return 0.0, 0.0
    inc = float((np.linalg.norm(P - centre_base, axis=1) <= radius).mean())
    cov = float(_nn_within(cl["pts"][inball], P, cover_tol).mean())
    return inc, cov


def set_components(geom, cand: str, pos: int, label_pts: np.ndarray, p1: S1Params,
                   p2: S2Params) -> tuple[float, float]:
    """``(inclusion, coverage)`` of one candidate on an S2 extension frame."""
    P = geom.chosen_points(cand, geom.frames[pos])
    if not len(label_pts):
        return float("nan"), float("nan")
    if not len(P):
        return 0.0, 0.0
    inc = float(_nn_within(P, label_pts, p2.comove_tol).mean())
    cov = float(_nn_within(label_pts, P, p1.cover_tol).mean())
    return inc, cov


def s0_frame_scores(geom, cands: Sequence[str], lab: Optional[dict]) -> np.ndarray:
    n_f = len(geom.frames)
    out = np.full((len(cands), n_f), np.nan)
    if lab is None:
        return out
    Twb = np.asarray(geom.z["T_world_base"], np.float64)
    for pos in lab["labeled"]:
        c = to_base(Twb[pos], lab["centre_world"])
        for a, cd in enumerate(cands):
            P = geom.chosen_points(cd, geom.frames[pos])
            out[a, pos] = float(sl.self_identity(P, c, lab["radius"])["self_identity_correct"])
    return out


def s1_components(geom, cands: Sequence[str], lab: Optional[dict], p: S1Params) -> tuple[np.ndarray, np.ndarray]:
    n_f = len(geom.frames)
    inc = np.full((len(cands), n_f), np.nan)
    cov = np.full((len(cands), n_f), np.nan)
    if lab is None:
        return inc, cov
    Twb = np.asarray(geom.z["T_world_base"], np.float64)
    for pos in lab["labeled"]:
        c = to_base(Twb[pos], lab["centre_world"])
        for a, cd in enumerate(cands):
            inc[a, pos], cov[a, pos] = ball_components(geom, cd, pos, c, lab["radius"], p.cover_tol)
    return inc, cov


def s2_components(geom, cands: Sequence[str], ext: Mapping, p1: S1Params, p2: S2Params
                  ) -> tuple[np.ndarray, np.ndarray]:
    n_f = len(geom.frames)
    inc = np.full((len(cands), n_f), np.nan)
    cov = np.full((len(cands), n_f), np.nan)
    for pos, pts in ext.get("frames", {}).items():
        for a, cd in enumerate(cands):
            inc[a, pos], cov[a, pos] = set_components(geom, cd, pos, pts, p1, p2)
    return inc, cov


def compose(*, l0, s0, l1, i1, c1, ext, i2, c2, p1: S1Params) -> dict:
    """Assemble the variants' frame matrices from precomputed parts (`frame_scores` / caching callers)."""
    f1 = normalize(combine(i1, c1, p1.combine), p1.normalize)
    f2x = normalize(combine(i2, c2, p1.combine), p1.normalize)
    f2 = np.where(np.isfinite(f1), f1, f2x)
    return {"labels": {"S0": l0, "S1": l1},
            "s2_extension": {k: v for k, v in ext.items() if k != "frames"},
            "s2_frames": sorted(int(k) for k in ext.get("frames", {})),
            "frame": {"S0": s0, "S1": f1, "S2": f2},
            "inclusion": {"S1": i1, "S2": np.where(np.isfinite(i1), i1, i2)},
            "coverage": {"S1": c1, "S2": np.where(np.isfinite(c1), c1, c2)}}


EMPTY_EXT = {"frames": {}, "trace": [], "stopped": None}


def frame_scores(geom, cands: Sequence[str], held_event, p1: S1Params, p2: S2Params) -> dict:
    """All variants' frame matrices (n_cand, n_frames) for one record, plus the labels and traces."""
    l0 = s0_label(geom)
    l1 = s1_label(geom, held_event, p1)
    ext = s2_extension(geom, held_event, l1, p2) if l1 is not None else dict(EMPTY_EXT)
    s0 = s0_frame_scores(geom, cands, l0)
    i1, c1 = s1_components(geom, cands, l1, p1)
    i2, c2 = s2_components(geom, cands, ext, p1, p2)
    return compose(l0=l0, s0=s0, l1=l1, i1=i1, c1=c1, ext=ext, i2=i2, c2=c2, p1=p1)


def episode_scores(fs: Mapping, *, positions: Optional[Sequence[int]] = None) -> dict:
    """Per variant: ``(n_cand,)`` episode score (mean over labeled scorable frames, restricted to
    `positions` when given), frame counts; plus S1's coverage component (the tie-break input)."""
    def mean(m):
        m = np.asarray(m, np.float64)
        if positions is not None:
            mask = np.zeros(m.shape[1], bool)
            mask[[p for p in positions if 0 <= p < m.shape[1]]] = True
            m = np.where(mask[None, :], m, np.nan)
        n = np.isfinite(m).sum(axis=1)
        with np.errstate(invalid="ignore"):
            v = np.where(n > 0, np.nansum(np.where(np.isfinite(m), m, 0.0), axis=1) / np.maximum(n, 1),
                         np.nan)
        return v, n

    out: dict[str, Any] = {}
    for v in VARIANTS:
        out[v], out[f"{v}_n"] = mean(fs["frame"][v])
    out["S1_coverage"], _ = mean(fs["coverage"]["S1"])
    out["S1_inclusion"], _ = mean(fs["inclusion"]["S1"])
    return out


# ------------------------------------------------------------------------------- GT checks


def label_class(labels: np.ndarray) -> Optional[str]:
    """Strict plurality class (T47b names) of labelled points; None when empty / tied."""
    from AG3S.ag3s.experiments.studies import t47b_metrics as bm

    return bm.plurality(bm.class_counts(np.asarray(labels, np.int64)))


def _verdict(cls: Optional[str]) -> str:
    if cls == "target":
        return "correct"
    if cls is None:
        return "empty_or_tied"
    if cls == "other_object":
        return "wrong_object"
    return f"other:{cls}"


def label_check(geom, lab: Optional[dict], ext: Optional[Mapping] = None) -> dict:
    """GT check of a ball label (sim only): the strict-plurality GT class of the label cloud's
    non-support points inside the ball at the event frame and on every labeled frame; and of the S2
    co-moving sets. HELD is a trigger, so a wrong class here is a label error, counted not hidden."""
    if lab is None:
        return {"has_label": False}
    Twb = np.asarray(geom.z["T_world_base"], np.float64)

    def at(pos):
        cl = label_cloud(geom, geom.frames[pos])
        if cl is None:
            return None, 0
        c = to_base(Twb[pos], lab["centre_world"])
        m = (~np.asarray(cl["support"], bool)) & (np.linalg.norm(cl["pts"] - c, axis=1) <= lab["radius"])
        return label_class(cl["labels"][m]), int(m.sum())

    ev_cls, ev_n = at(lab["event_pos"])
    per = [at(p) for p in lab["labeled"]]
    out = {"has_label": True, "event_frame": lab["event_frame"], "event_class": ev_cls,
           "event_n_points": ev_n, "event_verdict": _verdict(ev_cls),
           "event_label_correct": ev_cls == "target",
           "labeled_frames": len(per),
           "labeled_correct": int(sum(c == "target" for c, _ in per)),
           "labeled_empty": int(sum(n == 0 for _, n in per)),
           "labeled_verdicts": {}}
    for c, _ in per:
        k = _verdict(c)
        out["labeled_verdicts"][k] = out["labeled_verdicts"].get(k, 0) + 1
    if ext is not None and ext.get("frames"):
        vs = {}
        for pos, pts in ext["frames"].items():
            cl = label_cloud(geom, geom.frames[pos])
            m = _nn_within(cl["pts"], pts, 1e-6)
            k = _verdict(label_class(cl["labels"][m]))
            vs[k] = vs.get(k, 0) + 1
        out["s2_extension_verdicts"] = vs
    return out


__all__ = ["COMBINES", "NORMALIZES", "REFERENCES", "S1Params", "S2Params", "VARIANTS",
           "EMPTY_EXT", "ball_components", "ball_from_pads", "combine", "compose", "episode_scores", "frame_scores",
           "hand_frame", "label_check", "label_class", "label_cloud", "normalize", "s0_label",
           "s1_label", "s2_extension", "set_components", "to_base", "to_world"]
