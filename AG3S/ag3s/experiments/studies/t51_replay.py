"""T51 — the MuJoCo replay half: 3D grounding rows for many attention sources of one record.

The frame loop is T47b's (`t47b_replay._frames` · `_build` · `_ground` · `t47_row`, the live serving
path: `wire.pack_request -> strip_request -> unpack_camera_observations(attention=...) ->
AG3S.process_multi_debug`, AG3S from `serve_safe.build_ag3s` at the server defaults, subtask gate
off, one AG3S per source so TargetConfirm state never crosses sources). It is imported, not copied,
so a T51 row of ``last:L8H2`` is the T47b row of ``l8h2`` at sigma 0 (`t51_study --stage heldout`
checks that parity on the held-out records).

Per (frame, source) row: T47's fields, T47b's B2 fields without the post-hoc ball
(`t47b_metrics.frame_b2(ball=None)` — the ball is not part of the T51 score and costs a fit per row),
and the self-label fields when a self label is given (`t51_selflabel.self_identity` on the chosen
cluster's points against that frame's grasp point in the base frame; only frames before the grasp
event are labeled).

Renders (osmesa, CPU): run by the verifier. The implementer only smoke-runs a few frames.
"""

from __future__ import annotations

import time
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t47b_metrics as bm
from AG3S.ag3s.experiments.studies import t47b_replay as br
from AG3S.ag3s.experiments.studies import t51_selflabel as sl

#: Row fields the T51 summaries read (the rest of `frame_b2` is kept too).
SELF_FIELDS = ("self_labeled", "self_identity_correct", "self_in_n", "self_out_n",
               "self_centroid_dist_m")


def self_fields(cs, *, labeled: bool, centre: Optional[np.ndarray], radius: float) -> dict:
    """Self-label row fields of one grounding result (NaN / None when the frame is unlabeled)."""
    if not labeled or centre is None or not np.all(np.isfinite(centre)):
        return {"self_labeled": False, "self_identity_correct": None, "self_in_n": None,
                "self_out_n": None, "self_centroid_dist_m": float("nan")}
    pts = (np.asarray(cs.target.points, np.float64) if bool(cs.has_target)
           else np.zeros((0, 3)))
    return {"self_labeled": True, **sl.self_identity(pts, centre, radius)}


def run_sources(record_path, *, target: str, sources: Mapping[str, np.ndarray],
                selflabel: Optional[Mapping[str, np.ndarray]] = None,
                frames: Optional[Sequence[int]] = None, voxel: float = 0.020,
                range_max: float = 2.0, links: str = "gripper", esdf_backend: str = "legacy",
                on_target_radius: float = 0.10, label_radius: float = 0.003,
                membership_radius: float = 0.005,
                log: Callable[[str], None] = print) -> dict[str, list[dict]]:
    """``{source id: [row per frame]}`` for one record.

    `sources` maps a source id to raw ``(F, 3, 16, 16)`` maps in `POLICY_CAMERA_NAMES` order (sigma 0;
    the caller sums cells). `selflabel` is the record's `t51_selflabel.build_selflabel` output.
    """
    from AG3S.ag3s.experiments.sources.policy_record import POLICY_CAMERA_NAMES, load_run

    n_f = len(load_run(record_path))
    br._check_sources(dict(sources), n_f, len(POLICY_CAMERA_NAMES))
    if selflabel is not None and len(np.asarray(selflabel["labeled"])) != n_f:
        raise ValueError("self label frame count does not match the record")
    ags = br._build(record_path, list(sources), voxel=voxel, range_max=range_max, links=links,
                    esdf_backend=esdf_backend)
    rows: dict[str, list[dict]] = {s: [] for s in sources}
    t_start = time.time()
    for fr in br._frames(record_path, target=target, frames=frames, label_radius=label_radius,
                         log=log):
        i, step = fr["i"], fr["step"]
        lab = bool(selflabel is not None and np.asarray(selflabel["labeled"])[i])
        centre = None if selflabel is None else np.asarray(selflabel["grasp_point_base"])[i]
        radius = float("nan") if selflabel is None else float(np.asarray(selflabel["r_ball"]))
        marks = []
        for name, ag in ags.items():
            att = {mj: np.asarray(sources[name][i, ci], np.float32)
                   for ci, mj in enumerate(fr["mj_cams"])}
            cs, debug, ms = br._ground(ag, fr, att)
            row, arrays = br.t47_row(cs, debug, label_of=fr["label_of"],
                                     gt_centre=fr["gt_centre"], on_target_radius=on_target_radius)
            has = bool(cs.has_target)
            chosen = np.zeros(len(arrays["pts"]), bool)
            membership = "none"
            if has:
                chosen, membership = bm.membership_mask(
                    arrays["pts"], np.asarray(cs.target.points),
                    np.asarray(cs.target.point_indices), radius=membership_radius)
            b2 = bm.frame_b2(arrays["labels"], support_mask=arrays["support"],
                             seed_indices=arrays["seeds"], chosen_mask=chosen, has_target=has,
                             ball=None)
            rows[name].append({"frame": i, "t_step": int(step.t_step), "phase": fr["phase"],
                               "phase_group": bm.phase_group(fr["phase"]), "source": name,
                               **row, "ag3s_ms": ms, **b2, "chosen_membership": membership,
                               **self_fields(cs, labeled=lab, centre=centre, radius=radius)})
            r = rows[name][-1]
            marks.append("I" if r["identity_correct"] else ("T" if r["on_target"]
                                                             else ("s" if has else "-")))
        log(f"  t51 frame {i + 1}/{fr['n_f']} t={step.t_step} {fr['phase']} "
            f"{'L' if lab else ' '} {''.join(marks)}  ({time.time() - t_start:.0f}s)")
    return rows


__all__ = ["SELF_FIELDS", "run_sources", "self_fields"]
