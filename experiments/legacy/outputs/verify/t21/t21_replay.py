"""T21 verifier — fine / target-free window placement replayed on T14 / T17 server records.

What is reproduced (no depth in the records, so cuRobo fields cannot be rebuilt):

* window placement: `curobo_builder.place_fine_window` (working tree) with the pipeline's pads
  (`geometry.safety_margin` + `esdf.fine_window_pad_voxels` x fine voxel), hand = contact-authorized
  link spheres from `robot.sphere_centers_numeric` (the pipeline's FK) at q_now (b1) and at q_now +
  reference chunk steps 0..7 via `layout.full_q` (b2). Old rule (a) = `placement="target_centroid"`.
* tiers, EXACT: recorded field (`fixed_input_ablation.build_field`) -> `CuroboEsdfField.answer_tier`
  (only for the recorded live window a_rec).
* tiers, PROXY: coarse layer = recorded coarse grid (exact); fine and target-free layers at the
  replayed window = unsigned nearest-cloud-point distance (cKDTree on the recorded filtered_cloud,
  5 mm, 128^3, clipped at esdf.max_distance) ; target-free = same minus cloud points inside the
  TargetBall (centre, radius, z_min) of the manipulated geometry. Proxy fidelity is measured by
  building the proxy at the recorded window with the recorded ball and comparing with the exact
  recorded field.
* #2: `fixed_input_ablation` variant `collision` (budget clock off) on the exact field and on the
  proxy fields (per-chunk fresh solver, same as T24 m3).

Run: MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-ag3s/bin/python outputs/verify/t21/t21_replay.py <t14|t17>
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import ast
import copy
import json
import pathlib
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

import AG3S.trajopt.experiments.fixed_input_ablation as T
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.fields.curobo_builder import FineWindowRequest, place_fine_window
from AG3S.ag3s.fields.curobo_field import (CuroboEsdfField, count_tiers, layer_from_arrays,
                                                unresolved_rows)

ROOT = pathlib.Path(_ag3s_workspace_path(''))
OUT = ROOT / "outputs/verify/t21"
PARTIAL = ROOT / "AG3S/ag3s/docs/handoff/T21.verify.partial.json"
RUNS = {
    "t14": dict(d="20260928_t14", fr="execute_ep1807", chunks=list(range(11, 21))),
    "t17": dict(d="20260928_t17", fr="shadow_ep1807", chunks=list(range(9, 16))),
}
SHAPE = (128, 128, 128)
FINE = 0.005
FINGERS = ("ee_finger_l1", "ee_finger_l2")


def partial_append(entry):
    data = json.loads(PARTIAL.read_text())
    data["entries"].append(entry)
    tmp = PARTIAL.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(PARTIAL)


# ---------------------------------------------------------------------------------------------
# setup (same as fixed_input_ablation.main, variants = collision, clock off)
# ---------------------------------------------------------------------------------------------
def setup(run):
    from AG3S.trajopt.config import CostConfig, SqpConfig
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.types import ChunkLayout

    r = RUNS[run]
    base = ROOT / f"outputs/live_test/{r['d']}"
    argv = ["--constraints", str(base / "server_constraints/run_0000"),
            "--frames", str(base / r["fr"] / "frames.jsonl"),
            "--server-log", str(base / "serve_safe.log"),
            "--out", str(OUT / f"unused_{run}.json"),
            "--variants", "collision", "--budget-clock", "off", "--no-chain"]
    args = T.build_parser().parse_args(argv)
    frames = pathlib.Path(args.frames)
    manifest = json.loads(frames.with_name("manifest.json").read_text())
    log = T.parse_server_log(args.server_log)
    settings = T.resolve_settings(args, log, manifest)
    robot, qadr, xml_path = T.build_robot(settings)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, max_distance, unknown_policy = T.contact_policy(
        settings, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
    outside = max_distance if unknown_policy == "free" else -max_distance
    rows = [json.loads(l) for l in frames.read_text().splitlines() if l.strip()]
    plan_rows = {int(x["seq"]): x for x in rows if x.get("kind") == "planning"}
    files = sorted(pathlib.Path(args.constraints).glob("chunk_*.npz"), key=T._chunk_index)
    recorded_iters = {}
    for f in files:
        with np.load(f) as z:
            to = json.loads(str(z["summary_json"])).get("to") or {}
            recorded_iters[int(z["chunk_index"])] = int(to.get("sqp_iterations") or to.get("iterations") or 1)
    sqp_default = SqpConfig()
    rc = dict(settings.cost)
    track = {"w_track": rc["w_track"], "w_slack": rc["w_slack"]}
    off = {"w_smooth": 0.0, "w_continuity": 0.0}
    budget = {"time_budget_ms": T.NO_CLOCK_MS, "max_iterations": int(sqp_default.max_iterations)}
    cfgf = lambda k: T.to_config(settings, collision=True, cost={**track, **off}, sqp=budget)
    variant_configs = {"collision": cfgf}
    planning = cfgf(0)
    limits = build_limits(robot, layout, dt=planning.horizon.dt, config=planning.limits)
    meter = T.Meter(robot, layout, planning.horizon.planned, settings.hand, limits)
    active_cfg = T.Instrumented(robot, layout, planning, {"value": None}).refiner.optimizer.config
    sh = T.Shared(args=args, settings=settings, robot=robot, qadr=qadr, layout=layout, meter=meter,
                  plan_rows=plan_rows, variant_configs=variant_configs,
                  active_rows_config=active_cfg, outside_distance=outside,
                  recorded_iters=recorded_iters)
    return sh, {T._chunk_index(f): f for f in files}, xml_path, max_distance


# ---------------------------------------------------------------------------------------------
# truth geometry (MuJoCo, qpos of the planning row)
# ---------------------------------------------------------------------------------------------
class Truth:
    BODIES = ("table", "crate", "apple", "banana", "orange", "pear")

    def __init__(self, xml_path):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.geoms = {}
        for b in self.BODIES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
            self.geoms[b] = [g for g in range(self.model.ngeom)
                             if self.model.geom_bodyid[g] == bid and self.model.geom_contype[g]]

    def set(self, qpos):
        self.data.qpos[:] = np.asarray(qpos, np.float64)
        self.mj.mj_forward(self.model, self.data)

    def body_distance(self, pts, body):
        """(N,) signed distance to the body's collision geoms: box exact, mesh = ellipsoid on its
        geom_size (approximate, used for fruit only)."""
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        best = np.full(len(pts), np.inf)
        for g in self.geoms[body]:
            c = self.data.geom_xpos[g]
            R = self.data.geom_xmat[g].reshape(3, 3)
            s = self.model.geom_size[g]
            local = (pts - c) @ R
            if self.model.geom_type[g] == self.mj.mjtGeom.mjGEOM_BOX:
                q = np.abs(local) - s
                out = np.linalg.norm(np.maximum(q, 0.0), axis=1)
                ins = np.minimum(np.max(q, axis=1), 0.0)
                d = out + ins
            else:  # ellipsoid approximation (mesh fruit)
                k = np.linalg.norm(local / s, axis=1)
                d = (k - 1.0) * np.min(s)
            best = np.minimum(best, d)
        return best

    def label_points(self, pts, tol=0.015):
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        D = np.stack([self.body_distance(pts, b) for b in self.BODIES], axis=1)
        lab = np.full(len(pts), "other", dtype="<U8")
        j = np.argmin(D, axis=1)
        ok = D[np.arange(len(pts)), j] <= tol
        lab[ok] = np.asarray(self.BODIES)[j[ok]]
        return lab


# ---------------------------------------------------------------------------------------------
# proxy layers
# ---------------------------------------------------------------------------------------------
_axes_cache = {}


def lattice(lower):
    key = tuple(np.round(lower, 9))
    if key not in _axes_cache:
        ax = [lower[i] + FINE * np.arange(SHAPE[i]) for i in range(3)]
        _axes_cache.clear()
        _axes_cache[key] = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    return _axes_cache[key]


def proxy_layer(points, lower, maxd):
    lower = np.asarray(lower, np.float64)
    upper = lower + (np.asarray(SHAPE) - 1) * FINE
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    near = np.all((pts >= lower - maxd) & (pts <= upper + maxd), axis=1)
    pts = pts[near]
    if not len(pts):
        vals = np.full(SHAPE, maxd, np.float32)
    else:
        d, _ = cKDTree(pts).query(lattice(lower), distance_upper_bound=maxd, workers=-1)
        vals = np.minimum(d, maxd).reshape(SHAPE).astype(np.float32)
    return layer_from_arrays(vals, lower, FINE)


def ball_mask(points, ball):
    p = np.asarray(points, np.float64)
    c = np.asarray(ball["centre_m"], np.float64)
    r = float(ball["radius_mm"]) / 1000.0
    return (np.linalg.norm(p - c, axis=1) <= r) & (p[:, 2] >= float(ball["z_min_m"]))


def make_ball(points, plane_z, pad=FINE):
    """`AG3S._target_ball` rule (quantile 1.0 + 1 fine voxel, z_min = plane + 1 voxel)."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    c = p.mean(axis=0)
    rq = float(np.quantile(np.linalg.norm(p - c, axis=1), 1.0))
    return {"centre_m": c.tolist(), "radius_mm": (rq + pad) * 1000.0,
            "z_min_m": float(plane_z) + pad, "n_points": int(len(p)), "support_plane_z_m": float(plane_z)}


def composite(coarse_layer, fine_layer, free_layer, outside):
    cl = copy.copy(coarse_layer)
    return CuroboEsdfField((cl, fine_layer), outside_distance=outside,
                           target_free_layers=(free_layer,) if free_layer is not None else ())


# ---------------------------------------------------------------------------------------------
def tier_report(field, pts, radii, margin, target_free=True):
    tiers = np.asarray(field.answer_tier(pts, target_free=target_free))
    val = field.target_free_distance(pts) if target_free else field.distance(pts)
    unres, bounded = unresolved_rows(tiers, np.asarray(val) - radii - margin)
    return tiers, np.asarray(val), {"by_tier": count_tiers(tiers), "n_unresolved": int(unres.sum()),
                                   "n_bounded": int(bounded.sum()), "n": int(len(pts))}


def in_window(lower, upper, pts):
    return np.all((pts >= lower) & (pts <= upper), axis=1)


def window_dict(w):
    if w is None:
        return None
    s = w.summary()
    return s


FIELD_KEYS_M = ("rec", "t20", "default")
COLLISION_FIELDS = ("live_exact", "live_proxy", "b1_rec", "b2_rec", "a_t20", "b1_t20", "b2_t20")


def process_fields(c, sh, rec, inp, hand, radii, aidx, fidx, wins, M, balls, cloud, labels, truth,
                   margin, maxd):
    import dataclasses
    field_exact = inp.field
    coarse = field_exact.layers[0]
    outside = sh.outside_distance
    n_steps, S = hand.shape[:2]
    pts = hand.reshape(-1, 3)
    r_all = np.tile(radii, n_steps)
    step_of = np.repeat(np.arange(n_steps), S)
    finger = np.tile(np.isin(np.arange(S), fidx), n_steps)
    d_table = truth.body_distance(pts, "table")
    d_crate = truth.body_distance(pts, "crate")
    d_apple = truth.body_distance(pts, "apple")
    c["truth_at_query"] = {"definition": "signed distance from authorized sphere CENTRES to MuJoCo "
                           "collision geoms (box exact; apple = ellipsoid approx on geom_size)",
                           "finger_min_mm": {"table": float(d_table[finger].min() * 1000),
                                             "crate": float(d_crate[finger].min() * 1000),
                                             "apple": float(d_apple[finger].min() * 1000)}}
    # ball removals on the cloud (window independent)
    c["ball_removed_by_label"] = {}
    for m, ball in balls.items():
        if ball is None:
            c["ball_removed_by_label"][m] = None
            continue
        rm = ball_mask(cloud, ball)
        c["ball_removed_by_label"][m] = {b: [int((rm & (labels == b)).sum()), int((labels == b).sum())]
                                         for b in list(Truth.BODIES) + ["other"]}

    fine_cache, free_cache = {}, {}

    def fine_for(lower):
        key = tuple(np.round(lower, 7))
        if key not in fine_cache:
            fine_cache[key] = proxy_layer(cloud, lower, maxd)
        return fine_cache[key]

    def free_for(lower, m):
        key = (tuple(np.round(lower, 7)), m)
        if key not in free_cache:
            ball = balls[m]
            keep = ~ball_mask(cloud, ball) if ball else np.ones(len(cloud), bool)
            free_cache[key] = (proxy_layer(cloud[keep], lower, maxd), keep)
        return free_cache[key]

    def evaluate(field, lower, upper, keep=None):
        lower = np.asarray(lower); upper = np.asarray(upper)
        tiers, val, rep = tier_report(field, pts, r_all, margin)
        inside = in_window(lower, upper, pts)
        e = {"lower_m": lower.tolist(), "upper_m": upper.tolist(),
             "fraction_in_window_all": float(inside.mean()),
             "fraction_in_window_now": float(inside[step_of == 0].mean()),
             "fraction_in_window_path": float(inside[step_of > 0].mean()),
             "finger_fraction_in_window_all": float(inside[finger].mean()),
             "by_tier_all": rep["by_tier"], "n_unresolved_all": rep["n_unresolved"],
             "n_bounded_all": rep["n_bounded"], "n_all": rep["n"],
             "by_tier_now": count_tiers(tiers[step_of == 0]),
             "by_tier_path": count_tiers(tiers[step_of > 0]),
             "finger_by_tier_all": count_tiers(tiers[finger]),
             "finger_by_tier_now": count_tiers(tiers[finger & (step_of == 0)]),
             "tiers": tiers.tolist()}
        # #3 at finger query points: target-free answer vs truth
        tf = val[finger]
        nt = np.minimum(d_table[finger], d_crate[finger])
        e["m3_finger"] = {
            "target_free_distance_min_mm": float(tf.min() * 1000),
            "truth_table_min_mm": float(d_table[finger].min() * 1000),
            "truth_crate_min_mm": float(d_crate[finger].min() * 1000),
            "truth_apple_min_mm": float(d_apple[finger].min() * 1000),
            "tf_minus_min_table_crate_mm": {"median": float(np.median(tf - nt) * 1000),
                                            "max": float(np.max(tf - nt) * 1000),
                                            "min": float(np.min(tf - nt) * 1000)},
            "n_points": int(finger.sum()),
            "n_tf_exceeds_truth_table_crate_by_20mm": int(((tf - nt) > 0.020).sum()),
        }
        if keep is not None:
            kept = np.flatnonzero(keep)
            _, nn = cKDTree(cloud[kept]).query(pts[finger])
            nl = labels[kept][nn]
            e["m3_finger_nearest_kept_label"] = {b: int((nl == b).sum()) for b in list(Truth.BODIES) + ["other"]}
            win_pts = in_window(lower, upper, cloud)
            e["m3_cloud_in_window"] = {b: {"in_window": int((win_pts & (labels == b)).sum()),
                                           "kept_in_target_free": int((win_pts & keep & (labels == b)).sum())}
                                       for b in list(Truth.BODIES) + ["other"]}
        return e, tiers

    res = {}
    live = wins["live"]
    e, tiers_exact = evaluate(field_exact, live["lower_m"], live["upper_m"])
    e["field"] = "exact (recorded grids)"
    res["live_exact"] = e
    fields = {"live_exact": field_exact}
    # proxy at live window with the recorded ball -> fidelity control
    fl = fine_for(np.asarray(live["lower_m"]))
    fr, keep = free_for(np.asarray(live["lower_m"]), "rec")
    fp = composite(coarse, fl, fr, outside)
    e, tiers_proxy = evaluate(fp, live["lower_m"], live["upper_m"], keep)
    e["field"] = "proxy at live window, recorded ball"
    res["live_proxy"] = e
    fields["live_proxy"] = fp
    # proxy fidelity
    ex_fine = field_exact.layers[1]
    ex_free = field_exact.target_free_layers[0] if field_exact.target_free_layers else None
    inside = in_window(np.asarray(live["lower_m"]), np.asarray(live["upper_m"]), pts)
    fid = {"tier_agreement_all": float((tiers_exact == tiers_proxy).mean()),
           "n_query": int(len(pts)), "n_query_in_window": int(inside.sum())}
    if inside.any():
        dif = fl.distance(pts[inside]) - ex_fine.distance(pts[inside])
        fid["fine_query_diff_mm"] = {"median": float(np.median(dif) * 1000), "max_abs": float(np.abs(dif).max() * 1000)}
        if ex_free is not None:
            dif2 = fr.distance(pts[inside]) - ex_free.distance(pts[inside])
            fid["free_query_diff_mm"] = {"median": float(np.median(dif2) * 1000), "max_abs": float(np.abs(dif2).max() * 1000)}
    rng = np.random.default_rng(0)
    g = np.asarray(ex_fine.distance_grid, np.float32)
    sel = np.flatnonzero((g.reshape(-1) > 0) & (g.reshape(-1) < 0.10))
    if sel.size:
        sel = rng.choice(sel, size=min(20000, sel.size), replace=False)
        ijk = np.stack(np.unravel_index(sel, g.shape), -1)
        xyz = np.asarray(live["lower_m"]) + ijk * FINE
        dd = (np.asarray(fl.distance_grid).reshape(-1)[sel] - g.reshape(-1)[sel])
        fid["fine_voxel_diff_mm_exact_0_100mm"] = {"n": int(sel.size),
            "p10": float(np.percentile(dd, 10) * 1000), "median": float(np.median(dd) * 1000),
            "p90": float(np.percentile(dd, 90) * 1000), "mean_abs": float(np.mean(np.abs(dd)) * 1000)}
    zmin = float((balls.get("rec") or {}).get("support_plane_z_m", 0.82)) + 0.01
    sel2 = np.flatnonzero((g.reshape(-1) > 0) & (g.reshape(-1) < 0.10))
    if sel2.size:
        ijk = np.stack(np.unravel_index(sel2, g.shape), -1)
        xyz = np.asarray(live["lower_m"]) + ijk * FINE
        sel2 = sel2[xyz[:, 2] > zmin]
    if sel2.size:
        sel2 = rng.choice(sel2, size=min(20000, sel2.size), replace=False)
        dd = (np.asarray(fl.distance_grid).reshape(-1)[sel2] - g.reshape(-1)[sel2])
        fid["fine_voxel_diff_mm_exact_0_100mm_above_support"] = {"n": int(sel2.size), "z_min_m": zmin,
            "p10": float(np.percentile(dd, 10) * 1000), "median": float(np.median(dd) * 1000),
            "p90": float(np.percentile(dd, 90) * 1000), "mean_abs": float(np.mean(np.abs(dd)) * 1000)}
    c["proxy_fidelity"] = fid
    # replayed windows
    for m in FIELD_KEYS_M:
        for rn in ("a", "b1", "b2"):
            key = f"{rn}_{m}"
            w = wins.get(key)
            if w is None:
                res[key] = None
                continue
            lower = np.asarray(w["lower_m"]); upper = np.asarray(w["upper_m"])
            fl = fine_for(lower)
            fr, keep = free_for(lower, m)
            fp = composite(coarse, fl, fr, outside)
            e, _ = evaluate(fp, lower, upper, keep)
            e["field"] = f"proxy, ball of manipulated={m}"
            e["basis"] = w.get("basis"); e["overflow"] = w.get("overflow")
            res[key] = e
            fields[key] = fp
    c["fields"] = res
    # #2 collision variant
    margin_vec, mask, authorized, _, _ = T.contact_policy(
        sh.settings, sh.robot.sphere_link_names, has_target=True,
        has_free=True, attached=False)
    col = {}
    for key in COLLISION_FIELDS:
        if key not in fields:
            continue
        mname = "t20" if key.endswith("t20") else "rec"
        if key == "live_exact":
            scene = inp.scene
        else:
            scene = T.make_scene(fields[key], sh.meter.lin.robot_radii, sh.settings.esdf_margin,
                                 target_points=M[mname], margin=margin_vec, mask=mask)
        inp2 = dataclasses.replace(inp, field=fields[key], scene=scene)
        r2 = {k2: v2 for k2, v2 in rec.items()}
        T.run_variants(sh, r2, inp2, None, {})
        v = r2["variants"]["collision"]
        wr = v.get("worst_row")
        if isinstance(wr, str):
            wr = ast.literal_eval(wr)
        ar = r2["active_rows_at_reference"]
        col[key] = {"fingertip_disp_mm_per_step": v["exec_window"]["fingertip_disp_mm_per_step"],
                    "fingertip_disp_mm_max": v["exec_window"]["fingertip_disp_mm_max"],
                    "joint_dev_max_deg": v["exec_window"]["joint_dev_max_deg"],
                    "returned": v.get("returned"), "status": v.get("status"),
                    "sqp_iterations": v.get("sqp_iterations"),
                    "worst_row": wr,
                    "clearance_at_returned_min_mm": v["clearance_at_returned"]["min_mm"],
                    "active_rows_selected_by_tier": ar["selected_by_tier"],
                    "active_rows_selected_by_link": ar["selected_by_link"],
                    "reference_min_clearance_per_step_mm": ar["min_clearance_per_step_mm"]}
    c["collision_variant"] = col
    return c


def summarize_chunk(c):
    out = {"t_step": c.get("t_step"), "proxy_fidelity": c.get("proxy_fidelity")}
    for k, e in (c.get("fields") or {}).items():
        if e is None:
            out[k] = None
            continue
        out[k] = {"basis": e.get("basis"), "overflow": e.get("overflow"),
                  "frac_all": e["fraction_in_window_all"], "frac_now": e["fraction_in_window_now"],
                  "by_tier_all": e["by_tier_all"], "finger_by_tier_all": e["finger_by_tier_all"],
                  "n_unresolved_all": e["n_unresolved_all"], "m3": e["m3_finger"]}
    for k, v in (c.get("collision_variant") or {}).items():
        d = np.asarray(v["fingertip_disp_mm_per_step"])
        out.setdefault("collision", {})[k] = {"median_mm": float(np.median(d)), "max_mm": float(d.max()),
                                              "returned": v["returned"],
                                              "worst_tier": (v["worst_row"] or {}).get("tier"),
                                              "authorized_by_tier": ((v["worst_row"] or {}).get("authorized_rows") or {}).get("by_tier")}
    return out


CLOUD_MODE = "accumulated"
_acc = {}


def accumulated_cloud(by_index, k, dedup=0.0025):
    """Union of the recorded filtered clouds of chunks 0..k (the TSDF integrates every chunk frame
    and keeps surfaces it has seen; decay is off in these records). Deduplicated on a 2.5 mm lattice."""
    if k in _acc:
        return _acc[k]
    pts = np.concatenate([np.load(by_index[j])["filtered_cloud"].astype(np.float64)
                          for j in sorted(by_index) if j <= k])
    key = np.round(pts / dedup).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    _acc.clear()
    _acc[k] = pts[np.sort(first)]
    return _acc[k]


def main(run):
    sh, by_index, xml_path, maxd = setup(run)
    ag = AG3SConfig()
    margin = float(ag.geometry.safety_margin if ag.esdf.fine_window_margin is None else ag.esdf.fine_window_margin)
    interp = float(ag.esdf.fine_window_pad_voxels) * FINE
    truth = Truth(xml_path)
    manip = np.load(OUT / f"manip_{run}.npz")
    names = [str(n) for n in sh.robot.sphere_link_names]
    out = {"run": run, "settings": sh.settings.to_json(), "pads": {"margin_m": margin, "interp_m": interp,
           "hand_pad_m": margin + interp, "manipulated_pad_m": interp},
           "max_distance_m": maxd, "chunks": []}
    t0 = time.time()
    for k in RUNS[run]["chunks"]:
        rec, inp = T.prepare_chunk(sh, by_index[k])
        c = {"chunk": k, "seq": rec.get("seq"), "t_step": rec.get("t_step"),
             "fidelity_within_tol": (rec.get("fidelity") or {}).get("within_tolerance"),
             "skipped_reason": rec.get("skipped_reason")}
        if inp is None:
            out["chunks"].append(c); print(run, k, "skipped", rec.get("skipped_reason"), flush=True)
            continue
        z = inp.z
        row = sh.plan_rows[inp.seq]
        truth.set(row["qpos"])
        q_now = inp.q_now
        authorized = rec["contact_policy"]["authorized_links"]
        aidx = np.asarray([i for i, n in enumerate(names) if n in authorized], np.int64)
        fidx_in_a = np.asarray([j for j, i in enumerate(aidx) if names[i] in FINGERS], np.int64)
        full_path = sh.layout.full_q(inp.reference, q_now).T  # (8, nq)
        cen0, rad = sh.robot.sphere_centers_numeric(q_now)
        hand = [np.asarray(cen0)[aidx]]
        for q in full_path:
            hand.append(np.asarray(sh.robot.sphere_centers_numeric(q)[0])[aidx])
        hand = np.stack(hand)  # (9, S, 3)
        radii = np.asarray(rad, np.float64)[aidx]
        lin_c = sh.meter.lin.sphere_states(inp.reference, q_now)[0][:, aidx]
        c["fk_vs_linearizer_max_mm"] = float(np.abs(lin_c - hand[1:]).max() * 1000)
        c["authorized_links"] = authorized
        c["n_authorized_spheres"] = int(len(aidx))
        c["hand_centres_m"] = np.round(hand, 5).tolist()
        c["hand_radii_m"] = radii.tolist()
        stats = json.loads(str(z["summary_json"])).get("esdf_stats") or {}
        rec_tf = ast.literal_eval(stats["target_free"]) if isinstance(stats.get("target_free"), str) else stats.get("target_free")
        rec_ball = (rec_tf or {}).get("ball")
        plane_z = rec_ball["support_plane_z_m"] if rec_ball else 0.82
        cloud = accumulated_cloud(by_index, k) if CLOUD_MODE == "accumulated" else z["filtered_cloud"].astype(np.float64)
        c["cloud_mode"] = CLOUD_MODE
        labels = truth.label_points(cloud)
        c["cloud_n"] = int(len(cloud))
        c["cloud_labels"] = {b: int((labels == b).sum()) for b in list(Truth.BODIES) + ["other"]}
        c["object_poses"] = row.get("object_poses")
        # manipulated sources
        M = {"rec": np.asarray(z["target_points"], np.float64),
             "t20": manip[f"rec_{k}"], "default": manip[f"default_{k}"]}
        c["manipulated"] = {m: {"n": int(len(p)), "centroid_m": (p.mean(0).tolist() if len(p) else None),
                                "to_apple_mm": (float(np.linalg.norm(p.mean(0) - np.asarray(row["object_poses"]["apple"]["pos"])) * 1000) if len(p) else None)}
                            for m, p in M.items()}
        balls = {"rec": rec_ball, "t20": make_ball(M["t20"], plane_z) if len(M["t20"]) else None,
                 "default": make_ball(M["default"], plane_z) if len(M["default"]) else None}
        c["balls"] = balls
        # --- windows
        rec_lower = np.asarray(z["esdf_origin"], np.float64)
        wins = {"live": {"lower_m": rec_lower.tolist(), "upper_m": (rec_lower + 127 * FINE).tolist(),
                          "basis": "recorded", "overflow": None}}
        for m, pts in M.items():
            mp = pts if len(pts) else None
            reqs = {
                "a": FineWindowRequest(manipulated_points=mp, manipulated_pad=interp, placement="target_centroid",
                                       hand_centres=hand[:1], hand_radii=radii, hand_pad=margin + interp, hand_source="q_now"),
                "b1": FineWindowRequest(manipulated_points=mp, manipulated_pad=interp, placement="hand_swept",
                                        hand_centres=hand[:1], hand_radii=radii, hand_pad=margin + interp,
                                        hand_source="q_now", links=tuple(names[i] for i in aidx)),
                "b2": FineWindowRequest(manipulated_points=mp, manipulated_pad=interp, placement="hand_swept",
                                        hand_centres=hand, hand_radii=radii, hand_pad=margin + interp,
                                        hand_source="q_now+path", links=tuple(names[i] for i in aidx)),
            }
            for rn, req in reqs.items():
                w = place_fine_window(req, shape=SHAPE, voxel_size=FINE)
                wins[f"{rn}_{m}"] = window_dict(w)
        c["windows"] = wins
        # sanity: recorded live window vs place_fine_window(target_centroid) on the recorded target points
        c["live_vs_a_rec_lower_mm"] = float(np.abs(rec_lower - np.asarray(wins["a_rec"]["lower_m"])).max() * 1000)
        c = process_fields(c, sh, rec, inp, hand, radii, aidx, fidx_in_a, wins, M, balls, cloud, labels,
                           truth, margin, maxd)
        out["chunks"].append(c)
        partial_append({"kind": "t21_replay_chunk", "run": run, "chunk": k,
                        "summary": summarize_chunk(c)})
        print(run, k, "done", round(time.time() - t0, 1), "s", flush=True)
    path = OUT / f"replay_{run}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=T._jsonable))
    print("wrote", path)
    return out


if __name__ == "__main__":
    main(sys.argv[1])
