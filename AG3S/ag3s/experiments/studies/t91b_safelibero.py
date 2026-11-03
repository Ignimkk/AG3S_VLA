"""T91b — SafeLIBERO adapter and report for the phase machine (`AG3S/trajopt/phase_machine.py`). Python 3.11, CPU, read only.

Inputs (all read only):
  RUN      the VLA-only SafeLIBERO rollouts `.../results/benchmark/20261017T151126_..._sl5f-safelibero-LI-dev-LI-vla_only/episodes/`
           ``<stem>.npz`` (``obs_robot0_eef_pos``, ``obs_robot0_gripper_qpos`` (T,2), ``action_chunks`` (calls,10,7), ``chunk_t``),
           ``<stem>_geom.npz`` (``depth_mm/left_wrist`` 256x256, ``K/left_wrist``, ``T_base_cam/left_wrist``, ``T_world_base`` per frame)
  REPLAY   the offline ATS replay of the Panda final configuration
           `.../results/benchmark/20261102T1430_..._safelibero-seg-dev-ats-physical-20261102/` ``frames.jsonl`` (per policy call:
           target centroid, ATS state) and ``clouds/<stem>.npz`` (target points per call; base frame)
  PHASE    ``phase_<stem>.npz`` written by `t91b_safelibero_export.py` (physics re-render of ``sim_state_image``)

Per policy call ``c`` at frame ``f = chunk_t[c]`` the machine's inputs are

  gripper_raw      ``abs(gripper_qpos[f, 0])`` (finger joint 1, metres); `PhaseConfig.panda` fixes 1 = 0.04 m
  ee_pos           ``obs_robot0_eef_pos[f]`` (world; the grip site) moved into the robot base frame with ``T_world_base[f]``
                   (the ATS centroid / target points are base-frame; the physics GT below stays in the world frame)
  target_pos       ATS target centroid of that call (base frame), ``target_fresh`` = ATS state "visible"
  destination_pos  ``None`` (the replay records no destination); variant ``destination oracle`` = the BDDL plate body
                   position at the first call (a GT input, reported separately, never the headline)
  chunk_gripper    ``action_chunks[c, :, 6]`` (+1 close, -1 open)
  finger_gap       surface fraction of the wrist-depth ROI (`finger_gap.surface_fraction`); finger_gap_proj = the ATS target points
                   (base frame) projected through ``K`` / ``T_base_cam`` of that frame

Ground-truth 5 stages (T77 definition on the re-rendered physics, per call frame):
  held     = target touched by BOTH finger bodies ∧ target body z >= z(first call) + 20 mm
  place    = held ∧ target xy inside the destination (plate) collision-geom AABB xy
  carry    = held ∧ not place;  retreat = not held after having been held
  grasp    = never held ∧ |eef - target body| <= 60 mm (spec, headline);  approach = otherwise
  (sensitivities, reported, not the headline: ``grasp_or_touch`` = any finger touch also counts as grasp; ``robot_contact`` = "held"
  by any robot-body contact with the target instead of both finger bodies)
The classifier of T77 was fitted on RB-Y1 and does not exist for this checkpoint: only the T91 machine (gap evidence off) and the
T91b machine (gap evidence on) are compared.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t91_phase_eval as E
from AG3S.trajopt.finger_gap import FingerGapConfig, project_points, projection_fraction, surface_fraction
from AG3S.trajopt.phase_machine import STAGES, PhaseConfig, PhaseMachine, PhaseObservation

BENCH_RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
RUN = BENCH_RESULTS / "20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only"
REPLAY = BENCH_RESULTS / "20261102T1430_96fda38abd1b_safelibero-seg-dev-ats-physical-20261102"
LIFT_M = 0.02
GRASP_WINDOW_M = 0.06
GRIPPER_OPEN_M = 0.04
#: Panda wrist ROI (robot setting, fixed before any phase number): column band centred on the grip-site column (the site projects to
#: pixel (127.5, 213.5) at 0.097 m on every frame) and the rows directly above the finger-pad rows 195..212 (hardware geometry from
#: the wrist segmentation), so that the pads themselves are not "a surface between the fingers".
PANDA_ROI = ((0.38, 0.62), (0.62, 0.76))
DANGER_OK = 0.02            # lead: SafeLIBERO "worth moving to the benchmark" if the dangerous mislabel rate <= 2 %


def panda_gap_config(d_min: float, d_max: float, splat_frac: float = E.SPLAT_FRAC,
                     roi: Sequence = PANDA_ROI) -> FingerGapConfig:
    return FingerGapConfig(roi_u=tuple(roi[0]), roi_v=tuple(roi[1]), d_min=d_min, d_max=d_max, splat_frac=splat_frac)


# ================================================================================ ground truth (pure, tested)


def gt_phases(target_xpos: np.ndarray, left: np.ndarray, right: np.ndarray, dest_box: np.ndarray, eef: np.ndarray,
              call_frames: Sequence[int], *, grasp_or_touch: bool = False) -> np.ndarray:
    """Integer 5-stage ground truth (indices of `STAGES`) of the policy-call frames. All arrays are per recorded frame."""
    frames = [int(f) for f in call_frames]
    z0 = float(target_xpos[frames[0], 2])
    out, ever = [], False
    for f in frames:
        rise = float(target_xpos[f, 2]) - z0
        held = bool(left[f] and right[f] and rise >= LIFT_M)
        box = dest_box[f]
        in_dest = bool(box[0] <= target_xpos[f, 0] <= box[2] and box[1] <= target_xpos[f, 1] <= box[3])
        d = float(np.linalg.norm(np.asarray(eef[f]) - np.asarray(target_xpos[f])))
        if held:
            ever = True
            out.append(E.PLACE if in_dest else E.CARRY)
        elif ever:
            out.append(E.RETREAT)
        elif d <= GRASP_WINDOW_M or (grasp_or_touch and bool(left[f] or right[f])):
            out.append(E.GRASP)
        else:
            out.append(E.APPROACH)
    return np.array(out, int)


# ================================================================================ episode inputs


def load_replay(replay: pathlib.Path = REPLAY) -> dict:
    out = collections.defaultdict(dict)
    for line in (replay / "frames.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["stem"]][int(r["t"])] = r
    return out


def episode_inputs(stem: str, phase_dir: pathlib.Path, replay: dict, run: pathlib.Path = RUN,
                   replay_dir: pathlib.Path = REPLAY) -> Optional[dict]:
    """Per-call arrays of one episode, only calls the ATS replay covers; None if the replay has none."""
    with np.load(run / "episodes" / f"{stem}.npz") as e:
        chunk_t = e["chunk_t"].astype(int)
        qpos = e["obs_robot0_gripper_qpos"]
        eef = e["obs_robot0_eef_pos"]
        chunks = e["action_chunks"]
    with np.load(phase_dir / f"phase_{stem}.npz", allow_pickle=False) as p:
        ph = {k: p[k] for k in ("target_xpos", "contact_left_finger_target", "contact_right_finger_target", "destination_xy_box",
                                "destination_xpos", "contact_robot_target")}
    with np.load(run / "episodes" / f"{stem}_geom.npz", allow_pickle=False) as g:
        depth = g["depth_mm/left_wrist"][chunk_t]
        K = g["K/left_wrist"][chunk_t]
        T = g["T_base_cam/left_wrist"][chunk_t]
        T_wb = g["T_world_base"][chunk_t]
    cl = np.load(replay_dir / "clouds" / f"{stem}.npz", allow_pickle=False)
    if not np.array_equal(cl["call_frames"].astype(int), chunk_t):
        raise ValueError(f"{stem}: replay call frames differ from the rollout's")
    off, pts = cl["offsets"], cl["target_points"]
    rows = replay.get(stem, {})
    keep = [c for c, t in enumerate(chunk_t) if int(t) in rows]
    if not keep:
        return None
    gt_main = gt_phases(ph["target_xpos"], ph["contact_left_finger_target"], ph["contact_right_finger_target"],
                        ph["destination_xy_box"], eef, chunk_t)
    gt_touch = gt_phases(ph["target_xpos"], ph["contact_left_finger_target"], ph["contact_right_finger_target"],
                         ph["destination_xy_box"], eef, chunk_t, grasp_or_touch=True)
    # the ATS (centroid, target points) works in the robot base frame; the record's eef / the physics are world frame
    ee_base = np.einsum("nij,nj->ni", np.transpose(T_wb[:, :3, :3], (0, 2, 1)), eef[chunk_t] - T_wb[:, :3, 3])
    dest_base = (ph["destination_xpos"][chunk_t[0]] - T_wb[0, :3, 3]) @ T_wb[0, :3, :3]
    # sensitivity: "held" by any robot-body contact instead of both finger bodies (rim / pad-tip holds the finger rule misses)
    gt_robot = gt_phases(ph["target_xpos"], ph["contact_robot_target"], ph["contact_robot_target"], ph["destination_xy_box"],
                         eef, chunk_t)
    cen = np.full((len(chunk_t), 3), np.nan)
    state = np.array(["none"] * len(chunk_t), dtype=object)
    for c in keep:
        r = rows[int(chunk_t[c])]
        if r.get("centroid") is not None:
            cen[c] = r["centroid"]
        state[c] = str(r.get("manipulated_state"))
    return {"stem": stem, "task": int(stem.split("_t")[1].split("_")[0]), "chunk_t": chunk_t, "keep": np.array(keep),
            "grip_raw": np.abs(qpos[chunk_t, 0]), "ee": ee_base, "centroid": cen, "state": state,
            "chunk_grip": chunks[:, :, 6], "depth_m": depth.astype(np.float32) / 1000.0, "K": K, "T": T,
            "target_points": [np.asarray(pts[off[c]:off[c + 1]], float) for c in range(len(chunk_t))],
            "dest_oracle": dest_base, "gt": gt_main, "gt_touch": gt_touch, "gt_robot": gt_robot,
            "n_calls": len(chunk_t)}


def gap_evidence(ep: dict, cfg: FingerGapConfig) -> dict:
    """{"depth", "proj"} (n_calls,) arrays (NaN = none)."""
    n = ep["n_calls"]
    dep, prj = np.full(n, np.nan), np.full(n, np.nan)
    for c in range(n):
        v = surface_fraction(ep["depth_m"][c], cfg)
        dep[c] = np.nan if v is None else v
        uvz = project_points(ep["target_points"][c], ep["K"][c], ep["T"][c]) if len(ep["target_points"][c]) else None
        if uvz is not None:
            w = projection_fraction(uvz, ep["depth_m"][c].shape, cfg)
            prj[c] = np.nan if w is None else w
    return {"depth": dep, "proj": prj}


def observations(ep: dict, gap: Optional[dict], dest: Optional[np.ndarray] = None):
    """Machine inputs of the calls the replay covers (no GT except the optional destination oracle ``dest``)."""
    for c in ep["keep"]:
        fg = fp = None
        if gap is not None:
            fg = None if np.isnan(gap["depth"][c]) else float(gap["depth"][c])
            fp = None if np.isnan(gap["proj"][c]) else float(gap["proj"][c])
        cen = ep["centroid"][c]
        yield PhaseObservation(
            gripper_raw=float(ep["grip_raw"][c]), ee_pos=ep["ee"][c], target_pos=None if np.isnan(cen).any() else cen,
            target_fresh=ep["state"][c] == "visible", destination_pos=dest, chunk_gripper=ep["chunk_grip"][c],
            finger_gap=fg, finger_gap_proj=fp)


def predict(ep: dict, params: dict, gap: Optional[dict] = None, dest: Optional[np.ndarray] = None) -> np.ndarray:
    m = PhaseMachine(PhaseConfig.panda(**params))
    return np.array([E.IDX[m.step(o)[0]] for o in observations(ep, gap, dest)], int)


# ================================================================================ report


def as_run(ep: dict, gt_key: str = "gt") -> dict:
    keep = ep["keep"]
    return {"label": ep["stem"], "scene": f"task{ep['task']}", "split": "test", "y": ep[gt_key][keep],
            "t_step": ep["chunk_t"][keep], "request": keep}


def merged_accuracy(runs: Sequence[dict], preds: Sequence[np.ndarray]) -> Optional[float]:
    """Scene-mean accuracy with carry and place merged (the replay has no destination, so place is only reachable by the
    VLA's open command): 4 stages approach / grasp / carry+place / retreat."""
    f = lambda a: np.where(a == E.PLACE, E.CARRY, a)  # noqa: E731
    return E.scene_mean(runs, [f(p) for p in preds], lambda y, p: float(np.mean(f(y) == p)) if len(y) else None)


def report(eps: Sequence[dict], arms: dict, gt_key: str = "gt") -> dict:
    """``arms``: name -> list of predictions aligned with ``eps`` (kept calls)."""
    runs = [as_run(ep, gt_key) for ep in eps]
    out = {}
    for name, preds in arms.items():
        s = E.summarize(runs, preds)
        s["accuracy_scene_mean_carry_place_merged"] = merged_accuracy(runs, preds)
        d = s["danger_scene_mean"]
        s["danger_le_2pct_scene_mean"] = None if d is None else bool(d <= DANGER_OK)
        s["danger_le_2pct_request"] = None if s["danger_request"]["rate"] is None else bool(s["danger_request"]["rate"] <= DANGER_OK)
        out[name] = s
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--phase-dirs", nargs="+", required=True, help="dirs with phase_<stem>.npz from t91b_safelibero_export.py")
    ap.add_argument("--t91-thresholds", required=True, help="frozen T91 thresholds.json (RBY1 train)")
    ap.add_argument("--gap-thresholds", required=True, help="gap_thresholds.json (RBY1 train) written by t91_phase_eval gap-select")
    ap.add_argument("--limit", type=int, default=0, help="first N episodes only (smoke)")
    args = ap.parse_args(argv)
    base = json.loads(pathlib.Path(args.t91_thresholds).read_text())["params"]
    arms_sel = {k: v["selected"] for k, v in json.loads(pathlib.Path(args.gap_thresholds).read_text())["arms"].items()}
    replay = load_replay()
    phase_dirs = [pathlib.Path(d) for d in args.phase_dirs]
    stems = sorted(p.stem[len("phase_"):] for d in phase_dirs for p in d.glob("phase_*.npz"))
    if args.limit:
        stems = stems[: args.limit]
    eps = []
    for stem in stems:
        d = next(d for d in phase_dirs if (d / f"phase_{stem}.npz").exists())
        ep = episode_inputs(stem, d, replay)
        if ep is not None:
            eps.append(ep)
    dsel, psel = arms_sel["depth"], arms_sel["projection"]
    gcfg_d = panda_gap_config(dsel["d_min"], dsel["d_max"])
    gcfg_p = panda_gap_config(psel["d_min"], psel["d_max"])
    ev_d = [gap_evidence(ep, gcfg_d) for ep in eps]
    ev_p = [gap_evidence(ep, gcfg_p) for ep in eps]
    prm_d, prm_p = dsel["params"], psel["params"]
    both = {**prm_d, "gap_tau_proj": prm_p["gap_tau_proj"]}
    out = {"step": "T91b-safelibero", "n_episodes": len(eps), "n_calls_scored": int(sum(len(e["keep"]) for e in eps)),
           "n_calls_total": int(sum(e["n_calls"] for e in eps)), "panda_roi": PANDA_ROI, "gap_arms_selected_on_rby1_train": arms_sel,
           "gt_phase_counts": {STAGES[i]: int(sum(int((e["gt"][e["keep"]] == i).sum()) for e in eps)) for i in range(5)}}
    for tag, dest_oracle in (("destination_none", False), ("destination_oracle_GT_plate", True)):
        dest = lambda ep: ep["dest_oracle"] if dest_oracle else None  # noqa: E731
        preds = {
            "T91_machine_gap_off": [predict(ep, base, None, dest(ep)) for ep in eps],
            "T91b_depth": [predict(ep, prm_d, g, dest(ep)) for ep, g in zip(eps, ev_d)],
            "T91b_projection": [predict(ep, prm_p, g, dest(ep)) for ep, g in zip(eps, ev_p)],
            "T91b_both_reported_only": [predict(ep, {**both, "finger_gap_source": "both"}, g, dest(ep))
                                        for ep, g in zip(eps, ev_d)],
        }
        out[tag] = report(eps, preds)
        out[tag + "_predictions"] = {ep["stem"]: {"call_t": ep["chunk_t"][ep["keep"]].tolist(), "gt": ep["gt"][ep["keep"]].tolist(),
                                                   **{n: p[i].tolist() for n, p in preds.items()}} for i, ep in enumerate(eps)}
    out["destination_none_gt_grasp_or_touch_sensitivity"] = report(
        eps, {"T91b_depth": [predict(ep, prm_d, g) for ep, g in zip(eps, ev_d)]}, gt_key="gt_touch")
    out["destination_none_gt_robot_contact_sensitivity"] = report(
        eps, {"T91_machine_gap_off": [predict(ep, base) for ep in eps], "T91b_depth": [predict(ep, prm_d, g) for ep, g in zip(eps, ev_d)]},
        gt_key="gt_robot")
    # sensitivity (reported only, nothing is selected from it): the depth arm with other Panda ROI / window settings
    sens = {}
    for roi_name, roi in (("roi_main", PANDA_ROI), ("roi_low_incl_pads", ((0.35, 0.65), (0.72, 0.98))),
                          ("roi_high", ((0.38, 0.62), (0.50, 0.76)))):
        for dmax in (dsel["d_max"], 0.11, 0.13):
            cfg = panda_gap_config(dsel["d_min"], dmax, roi=roi)
            ev = [gap_evidence(ep, cfg) for ep in eps]
            sens[f"{roi_name}_dmax{dmax}"] = report(eps, {"T91b_depth": [predict(ep, prm_d, g) for ep, g in zip(eps, ev)]})["T91b_depth"]
    out["sensitivity_depth_arm_REPORTED_ONLY"] = {k: {"accuracy_scene_mean": v["accuracy_scene_mean"],
                                                     "danger_scene_mean": v["danger_scene_mean"],
                                                     "danger_request": v["danger_request"]["rate"],
                                                     "per_class_recall": v["per_class_recall"]} for k, v in sens.items()}
    out["evidence_by_gt"] = {}
    gd = np.concatenate([g["depth"][e["keep"]] for e, g in zip(eps, ev_d)])
    gy = np.concatenate([e["gt"][e["keep"]] for e in eps])
    for i, n in enumerate(STAGES):
        v = gd[gy == i]
        out["evidence_by_gt"][n] = {"n": int(len(v)), "depth_fraction_percentiles_5_25_50_75_95":
                                    None if len(v) == 0 else [float(np.percentile(v, p)) for p in (5, 25, 50, 75, 95)]}
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    h = out["destination_none"]
    print(json.dumps({k: {"acc": v["accuracy_scene_mean"], "danger_scene": v["danger_scene_mean"],
                          "danger_req": v["danger_request"]["rate"]} for k, v in h.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
