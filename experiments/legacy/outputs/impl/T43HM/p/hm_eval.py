"""T43 HM — offline evaluation of the held-row rule with the production code (implementer, CPU only, recorded data only).

usage: PYTHONPATH=${AG3S_PROJECT_ROOT} hm_eval.py OUT_JSON

Input: outputs/impl/T43HM/raw/survey_{V7g,V8}.json (hm_survey.py: per attached chunk, per refined row k = 0..7 the held query
centres P (base frame) and radii, the record's table plane and attach-time fit spheres; for executed rows also the
servo-predicted pose, T43/HX/raw pred_q14) + the chunk's recorded fields (FIA.build_field).
Per (chunk, row, pose) the production `CollisionLinearizer._esdf_clearance` (AG3S/trajopt/linearize.py) is run on the held
rows (one far dummy robot sphere first) with
  off : obstacle_margin 10 mm, held_obstacle_margin False                 (= V8: held rows margin 0)
  on  : obstacle_margin 10 mm, held_obstacle_margin True, support_planes = [record plane],
        held_capture_spheres = record `held.fit_spheres` (base, capture pose)  [runtime: + the pre-grasp place sphere, not recorded]
esdf_margin 0 (V7g / V8 log).  The label layer is not recorded -> destination bracket:
  B0 "no label"     : no row carries the destination label (off margin 0, on: obstacle rows 10 mm)          -> upper bound
  B1 "crate=dest"   : every held row whose nearest surface is GT crate carries it (off = on = 20 mm there)   -> lower bound
GT group of the nearest surface point: from the survey (MuJoCo geoms).  A chunk "newly violates" when min over rows 0..7 and held
queries of the on-clearance < 0 while the off-clearance >= 0 (B0 / B1 alike); the TO then has to change the plan (a HOLD if it
cannot -- the re-solve is not reproducible offline: held queries are not in FIA).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import numpy as np  # noqa: E402
import dx_common as C  # noqa: E402
from AG3S.trajopt.experiments import fixed_input_ablation as FIA  # noqa: E402
from AG3S.trajopt.linearize import OBSTACLE_CLASSES, CollisionLinearizer, SceneSnapshot  # noqa: E402

RAW = pathlib.Path(_ag3s_workspace_path('outputs/impl/T43HM/raw'))
ANALYSIS = {"V7g": _ag3s_workspace_path('outputs/verify/T43/CT/raw/analysis_v7g.json'), "V8": _ag3s_workspace_path('outputs/verify/T43/V8/analysis/analysis.json')}
M = 0.010
DEST_M = 0.020


def scene(field, held, plane, fit):
    extra = {}
    if held:
        extra = {"held_obstacle_margin": True,
                 "held_capture_spheres": None if fit is None else np.asarray(fit, float)}
    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0), candidate_active=np.zeros(0, bool),
        d_safe=np.zeros((1, 0)), plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=np.array([0.005]), candidate_ids=np.zeros(0, np.int64), esdf=field, esdf_margin=0.0,
        obstacle_margin=M, support_planes=None if plane is None else np.asarray([plane], float), **extra)


def main():
    out_path = pathlib.Path(sys.argv[1])
    res = {"definitions": __doc__, "rows": []}
    for arm in ("V7g", "V8"):
        sv = json.load(open(RAW / f"survey_{arm}.json"))["runs"]
        an = json.load(open(ANALYSIS[arm]))["runs"]
        for key, run in sorted(sv.items()):
            if not run.get("chunks"):
                continue
            seg = C.lit(an[key]["server_segment"]); chunks = C.segment_chunks(seg)
            st, outside, om, logp = C.server_settings(seg, pathlib.Path(an[key]["run_dir"]) / "manifest.json")
            for ch in run["chunks"]:
                z = np.load(chunks[ch["seq"]], allow_pickle=True); field = FIA.build_field(z, outside)
                for rr in ch["rows"]:
                    for src in ("plan", "servo"):
                        p = rr.get(src)
                        if not p:
                            continue
                        P = np.asarray(p["P_base"], float); r = np.asarray(p["r_mm"], float) / 1e3
                        lin = object.__new__(CollisionLinearizer)
                        lin.robot_radii = np.array([0.005]); lin.query_radii = np.r_[0.005, r]
                        lin.n_spheres = 1; lin._attached_local = None
                        cen = np.vstack([[P[0] + np.array([0.0, 0.0, 0.6])], P])[None]
                        off = lin._esdf_clearance(cen, scene(field, False, ch["plane"], ch["fit_spheres"]))[0, 1:, 0]
                        on = lin._esdf_clearance(cen, scene(field, True, ch["plane"], ch["fit_spheres"]))[0, 1:, 0]
                        cls = [OBSTACLE_CLASSES[int(c)] for c in lin._last_obstacle_class[0, 1:]]
                        gt = p["gt_group"]
                        crate = np.array([g == "crate" for g in gt])
                        # B1: crate-nearest rows carry the destination label -> margin 20 off, max(20, 10) = 20 on
                        off_b1 = np.where(crate, off - DEST_M, off); on_b1 = np.where(crate, off - DEST_M, on)
                        res["rows"].append({
                            "arm": arm, "key": key, "seq": ch["seq"], "t0": ch["t0"], "phase": ch["phase"], "n_att": ch["n_att"],
                            "k": rr["k"], "t": rr["t"], "executed": rr["executed"], "src": src, "gate": ch["gate"],
                            "to_status": ch["to_status"], "worst_class": ch["worst"]["class"], "worst_obstacle": ch["worst"]["obstacle"],
                            "off_mm": (1e3 * off).round(3).tolist(), "on_mm": (1e3 * on).round(3).tolist(),
                            "off_b1_mm": (1e3 * off_b1).round(3).tolist(), "on_b1_mm": (1e3 * on_b1).round(3).tolist(),
                            "cls": cls, "gt": gt, "gt_dist_mm": p["gt_dist_mm"], "true_hurdle_mm": p.get("true_hurdle_mm"),
                            "self_cap_mm": p.get("self_cap_mm"), "d_minus_r_mm": p["d_minus_r_mm"]})
            print(arm, key, flush=True)
    out_path.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
