"""IFG-CL2 offline replay — the hybrid target extent (`attention_surface_hybrid`) on recorded IFG-CL1 closed-loop chunks.

For every server chunk of a run (its `analysis/<run>.json` `server_segment`, chunk k ↔ client step k) with a fine
target-free ball, rebuild the fine target-free split **from the record alone** and compare, over the same seed set, with
the GT of the replayed scene (target / neighbour fruit / crate convex hulls ± `TOL`, the `ifg_cl1_keptseeds` rule):

| variant | removed seeds |
|---|---|
| `recorded` | the run's own trace (`target_free_removed_ijk`) — SURF: the oracle-path split, C1aN: `ball_owned` |
| `estimated` | `ball_owned` (`_ball_owned_split`) re-run on this chunk's TSDF — the C1aN rule (`replay_ok` = it equals the recorded trace for C1aN chunks) |
| `hybrid:<rule>` | `surface_protect_split` with the server function `attention_surface_hybrid.hybrid_evidence` (identity rule `both` · `instance` · `centre`) |
| `hybrid:noguard` | the protection applied whatever the identity guard says (what the guard prevents) |
| `hybrid:both+x1.0` | IFG-CL2b — `hybrid:both` with the other nodes within 1.0 fusion voxel (10 mm) of the manipulated geometry points dropped from the other evidence (`other_exclude_voxels=1.0`, the fixed server default). The other rows are the IFG-CL2 rule (`0`) |

Inputs, all recorded: seeds = voxels of the recorded fine main tier with distance exactly 0 (the cuRobo seed sites; checked
against `n_seeds_in_ball` / the C1aN trace) inside the recorded ball; the `ball_owned` evidence = recorded target points and
`filtered_cloud − target_point_indices − support_mask` (or the 5 mm proximity drop, as `AG3S._nontarget_points`); the
attention surface = `select_target_surface` on the **client record's** depth / K / T (uint16 mm, what the server received),
the chunk's float16 attention maps and robot mask = the recorded code image's 0 pixels (codes are recorded only on
selected frames; `codes_parity_px` = pixels where the replayed code image differs from the recorded one). Chunks whose
frame abstained (no recorded codes) are `attention_surface_abstained` for the hybrid, as on the server. C1aN chunks have
no code images: only `recorded` / `estimated` are evaluated there.

Seed universe per chunk = **every seed of the fine tier** (all distance-0 voxels; every variant removes a subset of
them), so all variants and both arms are counted on the same set: `kept_target` = GT-target seeds not removed (a seed the
fine target-free tier keeps is an obstacle to the authorised fingers), split into inside / outside the ball
(`kept_target_in_ball`), and above the ball's support guard `z_min` (`kept_target_above_floor` — below it are the table
seeds under the fruit that every extent keeps on purpose, yet they lie within `TOL` of the target hull). This differs from `ifg_cl1_keptseeds` (kept-as-non-target trace seeds only — the ball for
`ball_owned`, the oracle candidates for SURF); that count is `recorded.keptseeds_kept_target`. Classes are exclusive
in the order target → neighbour fruit → crate → other.

  MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \\
      AG3S.ag3s.experiments.studies.cl2_hybrid_replay --cl1 <run>/verify/IFG-CL1 --out <dir> [--runs NAME ...] [--workers 2]
  ... --cl1 <...> --out <dir> --summary        # <dir>/summary.json (per arm × outcome)
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import pathlib
import shlex
import time

import numpy as np

TOL = 0.005
CAM = {"head": "zed_left", "left_wrist": "wrist_cam_l", "right_wrist": "wrist_cam_r"}
RULES = ("both", "instance", "centre")
#: IFG-CL2b — the fixed exclusion (fusion voxels) replayed next to the IFG-CL2 rule (0). Only this value; not a search.
CL2B_EXCLUDE_VOXELS = 1.0
CL2B_KEY = f"both+x{CL2B_EXCLUDE_VOXELS:.1f}"


def _server_owned_args(server_dir: pathlib.Path) -> dict:
    """`ball_owned` parameters of the server that wrote the chunk (its `server_proc.txt` command line; config defaults)."""
    from AG3S.ag3s.config import EsdfConfig

    d = EsdfConfig()
    out = {"tie_voxels": float(d.owned_tie_voxels), "max_dist_voxels": float(d.owned_max_dist_voxels),
           "unowned": str(d.owned_unowned), "ambiguous": str(d.owned_ambiguous),
           "nontarget_evidence": str(d.nontarget_evidence), "argv": None}
    p = server_dir / "server_proc.txt"
    if p.exists():
        line = next((ln for ln in p.read_text().splitlines() if "serve_safe" in ln), "")
        argv = shlex.split(line)
        out["argv"] = line
        for flag, key, typ in (("--owned-tie-voxels", "tie_voxels", float),
                               ("--owned-max-dist-voxels", "max_dist_voxels", float),
                               ("--owned-unowned", "unowned", str), ("--owned-ambiguous", "ambiguous", str),
                               ("--nontarget-evidence", "nontarget_evidence", str)):
            if flag in argv:
                out[key] = typ(argv[argv.index(flag) + 1])
    return out


def _ball_cand(ball, origin, vs, shape):
    r = float(ball.radius)
    lo = np.maximum(np.floor((ball.centre - r - origin) / vs + 0.5).astype(np.int64), 0)
    hi = np.minimum(np.ceil((ball.centre + r - origin) / vs + 0.5).astype(np.int64) + 1, np.array(shape))
    if np.any(hi <= lo):
        return np.zeros((0, 3), np.int64)
    ax = [np.arange(int(lo[i]), int(hi[i])) for i in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    c = origin + g * vs
    ins = (np.linalg.norm(c - ball.centre, axis=1) <= r) & (c[:, 2] >= float(ball.z_min))
    return g[ins]


def _nontarget(z, tp):
    """`AG3S._nontarget_points` on the recorded cloud: `(points, source)`."""
    from scipy.spatial import cKDTree

    cl = z["filtered_cloud"].astype(np.float64)
    keep = ~np.asarray(z["support_mask"], bool)
    idx = np.asarray(z["target_point_indices"], np.int64) if "target_point_indices" in z.files else np.zeros(0, int)
    if len(idx) == len(tp) and len(idx) and idx.min() >= 0 and idx.max() < len(cl) and np.allclose(cl[idx], tp,
                                                                                                    atol=1e-6):
        keep[idx] = False
        return cl[keep], "indices"
    if len(tp):
        d, _ = cKDTree(tp).query(cl)
        keep &= ~(d <= 0.005)
    return cl[keep], "proximity"


def _gt_hulls(scene, target, Tb):
    """`{"target": hull, "neighbour": [hulls], "crate": [hulls]}` in the base frame."""
    import mujoco
    from scipy.spatial import ConvexHull

    from AG3S.ag3s.experiments.studies.ifg_cl1_keptseeds import target_points
    from AG3S.ag3s.runtime.oracle_segmentation import GRASPABLE_OBJECTS

    m, d = scene.model, scene.data
    to_base = lambda p: p @ Tb[:3, :3].T + Tb[:3, 3]
    out = {"target": ConvexHull(to_base(target_points(m, d, target))), "neighbour": [], "crate": []}
    for o in GRASPABLE_OBJECTS:
        if o == target or mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o) < 0:
            continue
        p = target_points(m, d, o)
        if len(p) >= 4:
            out["neighbour"].append(ConvexHull(to_base(p)))
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")
    if cid >= 0:
        for g in range(m.ngeom):
            if m.body_rootid[m.geom_bodyid[g]] != cid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            s = m.geom_size[g]
            v = np.array([[x, y, w] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for w in (-s[2], s[2])])
            p = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            out["crate"].append(ConvexHull(to_base(p), qhull_options="QJ"))
    return out


def _classify(hulls, x):
    from AG3S.ag3s.experiments.studies.ifg_cl1_keptseeds import hull_dist

    n = len(x)
    if not n:
        return np.zeros(0, np.int8)
    tgt = hull_dist(hulls["target"], x) <= TOL
    nb = np.zeros(n, bool)
    for h in hulls["neighbour"]:
        nb |= hull_dist(h, x) <= TOL
    cr = np.zeros(n, bool)
    for h in hulls["crate"]:
        cr |= hull_dist(h, x) <= TOL
    cls = np.full(n, 3, np.int8)            # 0 target · 1 neighbour fruit · 2 crate · 3 other
    cls[cr] = 2
    cls[nb] = 1
    cls[tgt] = 0
    return cls


def _metrics(cls_u, removed_u, in_ball, above):
    rm = np.asarray(removed_u, bool)
    kt = (cls_u == 0) & ~rm
    return {"n_removed": int(rm.sum()),
            "kept_target": int(kt.sum()), "kept_target_above_floor": int((kt & above).sum()),
            "kept_target_in_ball": int((kt & in_ball).sum()),
            "removed_target": int(((cls_u == 0) & rm).sum()),
            "removed_neighbour": int(((cls_u == 1) & rm).sum()), "removed_crate": int(((cls_u == 2) & rm).sum()),
            "removed_other": int(((cls_u == 3) & rm).sum())}


def run_one(cl1: pathlib.Path, name: str, all_chunks: bool = False, only_k=None) -> dict:
    """One run's chunk rows (see the module docstring). `only_k` = these chunk indices only (tests)."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene
    from AG3S.ag3s.fields.curobo_builder import OwnedEvidence, TargetBall, _ball_owned_split, surface_protect_split
    from AG3S.ag3s.runtime.attention_surface import CAMERA_ORDER, SurfaceView, select_target_surface
    from AG3S.ag3s.runtime.attention_surface_hybrid import hybrid_evidence, hybrid_samples

    a = json.loads((cl1 / "analysis" / f"{name}.json").read_text())
    rd = pathlib.Path(a["run_dir"])
    rec = load_run(rd / "ag3s/run_0000")
    target = json.loads((rd / "manifest.json").read_text())["scene"]["episode_target_fruit"]
    seg = a["server_segment"]
    d0 = pathlib.Path(seg["first"]).parent
    i0 = int(pathlib.Path(seg["first"]).stem.split("_")[1])
    owned_args = _server_owned_args(d0.parent.parent)
    if owned_args["nontarget_evidence"] != "proximity":
        raise NotImplementedError("T46 F1 surface evidence is not replayed here")
    grasp_t = a.get("grasp_t")
    scene = TransportScene(rec.meta["model_xml"], height=64, width=64, settle_steps=0)
    rows = []
    try:
        for k in range(min(int(seg["n"]), len(rec.steps))):
            step = rec.steps[k]
            if only_k is not None and k not in only_k:
                continue
            if not all_chunks and grasp_t is not None and int(step.t_step) >= int(grasp_t):
                continue
            z = np.load(d0 / f"chunk_{i0 + k:05d}.npz", allow_pickle=True)
            if "target_free_trace_origin" not in z.files or "esdf_distance" not in z.files:
                continue
            tf = ast.literal_eval(json.loads(str(z["summary_json"]))["esdf_stats"]["target_free"])
            b = tf.get("ball")
            if not b:
                continue
            row = {"k": k, "t_step": int(step.t_step), "mode": tf.get("mode"), "method": tf.get("exclusion_method")}
            ball = TargetBall(centre=np.asarray(b["centre_m"], np.float64), radius=float(b["radius_mm"]) / 1e3,
                              z_min=float(b["z_min_m"]), provenance={})
            org = np.asarray(z["esdf_origin"], np.float64)
            vs = float(z["esdf_voxel_size"])
            if not np.allclose(org, z["target_free_trace_origin"]):
                raise RuntimeError(f"{name} k={k}: fine tier and trace origins differ")
            D = z["esdf_distance"]
            cand = _ball_cand(ball, org, vs, D.shape)
            seed = D[tuple(cand.T)] == 0
            centres = org + cand * vs
            tp = z["target_points"].astype(np.float64)
            nt, src = _nontarget(z, tp)
            owned = OwnedEvidence(target_points=tp, nontarget_points=nt, tie_voxels=owned_args["tie_voxels"],
                                  max_dist_voxels=owned_args["max_dist_voxels"], unowned=owned_args["unowned"],
                                  ambiguous=owned_args["ambiguous"])
            t = time.perf_counter()
            est = _ball_owned_split(centres, seed, ball, owned, vs)
            row["est_split_ms"] = round((time.perf_counter() - t) * 1e3, 3)
            row["n_seeds_in_ball"] = int(seed.sum())
            row["nontarget_source"] = src
            rec_rem = z["target_free_removed_ijk"].astype(np.int64)
            rec_seeds = np.vstack([z[f"target_free_{n}"].astype(np.int64).reshape(-1, 3)
                                   for n in ("removed_ijk", "nontarget_kept_ijk", "ambiguous_ijk", "unowned_kept_ijk")
                                   if f"target_free_{n}" in z.files])
            if row["method"] == "ball_owned":
                row["replay_ok"] = bool(
                    set(map(tuple, cand[est["removed"]])) == set(map(tuple, rec_rem))
                    and int(seed.sum()) == int(tf.get("n_seeds_in_ball", -1)))
            # the seed universe: every seed of the fine tier
            uni = np.argwhere(D == 0).astype(np.int64)
            key = lambda g: (g[:, 0] * 4096 + g[:, 1]) * 4096 + g[:, 2]
            ku = key(uni)
            in_set = lambda g: np.isin(ku, key(g)) if len(g) else np.zeros(len(ku), bool)
            row["n_trace_seeds_not_in_tier"] = int((~np.isin(key(rec_seeds), ku)).sum()) if len(rec_seeds) else 0
            in_ball = in_set(cand[seed])
            # GT
            pose_scene(scene, step)
            Tb = np.linalg.inv(scene.body_pose("base"))
            hulls = _gt_hulls(scene, target, Tb)
            cls = _classify(hulls, org + uni * vs)
            above = (org[2] + uni[:, 2] * vs) >= float(ball.z_min)
            met = lambda rm: _metrics(cls, rm, in_ball, above)
            row["n_universe"] = int(len(uni))
            row["gt_universe"] = {c: int((cls == i).sum()) for i, c in enumerate(("target", "neighbour", "crate",
                                                                                  "other"))}
            row["recorded"] = met(in_set(rec_rem))
            if "target_free_nontarget_kept_ijk" in z.files:   # the `ifg_cl1_keptseeds` count (kept-as-non-target only)
                kk = z["target_free_nontarget_kept_ijk"].astype(np.int64)
                row["recorded"]["keptseeds_kept_target"] = int(((cls == 0) & in_set(kk)).sum())
            row["estimated"] = met(in_set(cand[est["removed"]]))
            # attention surface (codes recorded only on selected frames)
            surf_rec = (tf.get("attention_surface") or {})
            has_codes = all(f"attention_surface_codes_{n}" in z.files for n in CAMERA_ORDER)
            row["surface_recorded"] = {"present": bool(surf_rec), "applied": surf_rec.get("applied"),
                                       "why": surf_rec.get("why"), "has_codes": has_codes,
                                       "oracle_why": (tf.get("oracle") or {}).get("why"),
                                       "oracle_identity_code": (tf.get("oracle") or {}).get("pipeline_code")}
            if not has_codes:
                if surf_rec:
                    row["hybrid_why"] = f"attention_surface_abstained ({surf_rec.get('why')})"
                rows.append(row)
                continue
            raw = np.load(rd / "ag3s/run_0000" / f"step_{k:05d}.npz")
            views = []
            for n in CAMERA_ORDER:
                c = CAM[n]
                views.append(SurfaceView(name=n, depth_m=raw[f"depth_{c}"].astype(np.float64) * 0.001,
                                         K=raw[f"K_{c}"], T_base_cam=raw[f"T_base_cam_{c}"],
                                         robot_mask=z[f"attention_surface_codes_{n}"] == 0,
                                         attention=z[f"attention_{n}"].astype(np.float32)))
            t = time.perf_counter()
            res = select_target_surface(views, keep_debug=True)
            row["select_ms"] = round((time.perf_counter() - t) * 1e3, 3)
            row["select_record_ms"] = res.record.get("ms")
            if res.codes is None:
                row["codes_parity_px"] = None
                row["hybrid_why"] = f"replay_abstained ({res.record.get('why')})"
                rows.append(row)
                continue
            row["codes_parity_px"] = int(sum((res.codes[n] != z[f"attention_surface_codes_{n}"]).sum()
                                             for n in CAMERA_ORDER))
            t = time.perf_counter()
            samples = hybrid_samples(res)
            row["samples_ms"] = round((time.perf_counter() - t) * 1e3, 3)
            hyb = {}
            for rule in RULES + ("noguard", CL2B_KEY):
                t = time.perf_counter()
                ev, hrec = hybrid_evidence(samples, res.record, tp, ball, radius=0.010,
                                           rule="both" if rule in ("noguard", CL2B_KEY) else rule,
                                           **({"other_exclude_voxels": CL2B_EXCLUDE_VOXELS} if rule == CL2B_KEY
                                              else {}))
                guard_ms = (time.perf_counter() - t) * 1e3
                if rule == "noguard" and ev is None:
                    from AG3S.ag3s.fields.curobo_builder import SurfaceProtectEvidence

                    ev = SurfaceProtectEvidence(target_points=samples.target, other_points=samples.other)
                if ev is None:
                    rm = est["removed"]
                    split_ms = 0.0
                    n_prot = 0
                else:
                    t = time.perf_counter()
                    sp = surface_protect_split(centres, seed, ball, owned, ev, vs)
                    split_ms = (time.perf_counter() - t) * 1e3
                    rm = sp["removed"]
                    n_prot = int(sp["protected"].sum())
                hyb[rule] = {**met(in_set(cand[rm])), "applied": ev is not None, "n_protected": n_prot,
                             "why": hrec.get("why"), "guard_ms": round(guard_ms, 3), "split_ms": round(split_ms, 3)}
                if "other_exclude" in hrec:   # IFG-CL2b
                    hyb[rule]["other_exclude"] = hrec["other_exclude"]
                if rule == "both":
                    row["identity"] = hrec.get("identity")
            row["hybrid"] = hyb
            row["n_target_nodes"] = int(len(samples.target))
            row["n_other_nodes"] = int(len(samples.other))
            rows.append(row)
    finally:
        scene.close()
    return {"name": name, "arm": a["arm"], "episode": a["episode"], "seed": a.get("seed"), "success": a["success"],
            "failure_mode": a["failure_mode"], "grasp_t": grasp_t, "target": target, "owned": owned_args,
            "pre_grasp_only": not all_chunks, "n_chunks": len(rows), "chunks": rows}


VARIANTS = ("recorded", "estimated", "hybrid:both", "hybrid:instance", "hybrid:centre", "hybrid:noguard",
            f"hybrid:{CL2B_KEY}")
METRICS = ("kept_target", "kept_target_above_floor", "kept_target_in_ball", "removed_neighbour", "removed_crate",
           "removed_target", "n_removed")


def _variant(row, v):
    if v in ("recorded", "estimated"):
        return row[v]
    rule = v.split(":", 1)[1]
    h = row.get("hybrid")
    return row["estimated"] if h is None else h[rule]   # no codes → the hybrid is `ball_owned` that frame


def summarise(out_dir: pathlib.Path) -> dict:
    """Per group (arm × success) per-chunk medians / means of the replay rows, protection-off reasons, ms, parity."""
    from collections import Counter

    runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json")) if p.name != "summary.json"]
    groups: dict = {}
    for r in runs:
        g = f"{r['arm']}_{'success' if r['success'] else 'fail'}"
        groups.setdefault(g, []).append(r)
    out: dict = {"groups": {}}
    for g, rs in sorted(groups.items()):
        rows = [c for r in rs for c in r["chunks"]]
        q: dict = {"n_runs": len(rs), "n_chunks": len(rows), "runs": [r["name"] for r in rs]}
        for v in VARIANTS:
            if v.startswith("hybrid") and not any("hybrid" in c or "hybrid_why" in c for c in rows):
                continue
            if v.startswith("hybrid") and any(v.split(":", 1)[1] not in c["hybrid"] for c in rows if c.get("hybrid")):
                continue   # run files written before IFG-CL2b
            vals = {m: np.array([_variant(c, v)[m] for c in rows], float) for m in METRICS}
            q[v] = {m: {"median": float(np.median(x)) if len(x) else None, "mean": float(x.mean()) if len(x) else None,
                        "sum": float(x.sum())} for m, x in vals.items()}
        kk = [c["recorded"].get("keptseeds_kept_target") for c in rows if "keptseeds_kept_target" in c["recorded"]]
        q["recorded_keptseeds_kept_target_median"] = float(np.median(kk)) if kk else None
        why = Counter()
        n_on = 0
        for c in rows:
            h = c.get("hybrid")
            if h is not None and h["both"]["applied"]:
                n_on += 1
            else:
                why[(h["both"]["why"] if h is not None else c.get("hybrid_why")) or "no_attention_surface_record"] += 1
        q["protection"] = {"n_on": n_on, "frac_on": n_on / max(len(rows), 1), "off_why": dict(why.most_common())}
        prot = [c["hybrid"]["both"]["n_protected"] for c in rows if c.get("hybrid") and c["hybrid"]["both"]["applied"]]
        q["n_protected_when_on"] = {"median": float(np.median(prot)) if prot else None,
                                    "sum": int(np.sum(prot)) if prot else 0}
        hx = [c["hybrid"][CL2B_KEY] for c in rows if c.get("hybrid") and CL2B_KEY in c["hybrid"]
              and c["hybrid"][CL2B_KEY]["applied"]]
        if hx:   # IFG-CL2b
            q[f"n_protected_when_on_{CL2B_KEY}"] = {"median": float(np.median([h["n_protected"] for h in hx])),
                                                     "sum": int(np.sum([h["n_protected"] for h in hx]))}
            ox = [h["other_exclude"] for h in hx if "other_exclude" in h]
            q[f"other_exclude_{CL2B_KEY}"] = {
                "n": len(ox), "n_no_geometry": int(sum("why" in o for o in ox)),
                "n_excluded_median": float(np.median([o["n_excluded"] for o in ox])) if ox else None,
                "frac_excluded_median": float(np.median([o["n_excluded"] / max(o["n_before"], 1) for o in ox]))
                if ox else None}
        for k in ("select_ms", "samples_ms", "est_split_ms"):
            x = [c[k] for c in rows if c.get(k) is not None]
            q[k] = {"median": float(np.median(x)), "p90": float(np.percentile(x, 90))} if x else None
        for k in ("guard_ms", "split_ms"):
            x = [c["hybrid"]["both"][k] for c in rows if c.get("hybrid") and c["hybrid"]["both"]["applied"]]
            q[k] = {"median": float(np.median(x)), "p90": float(np.percentile(x, 90))} if x else None
        par = [c["codes_parity_px"] for c in rows if c.get("codes_parity_px") is not None]
        q["codes_parity"] = {"n": len(par), "n_exact": int(sum(p == 0 for p in par)), "max_px": max(par) if par else None}
        ok = [c["replay_ok"] for c in rows if "replay_ok" in c]
        q["ball_owned_replay_ok"] = {"n": len(ok), "n_ok": int(sum(ok))}
        out["groups"][g] = q
    (out_dir / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cl1", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--all-chunks", action="store_true", help="also the chunks at / after grasp_t")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--summary", action="store_true", help="only (re)write <out>/summary.json from the run files")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.summary:
        print(json.dumps(summarise(a.out), indent=1, default=float)[:20000])
        return
    names = a.runs or sorted(pathlib.Path(p).stem for p in glob.glob(str(a.cl1 / "analysis" / "*.json")))
    todo = [n for n in names if not (a.out / f"{n}.json").exists()]
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(max(1, min(a.workers, 4))) as p:
        for r in p.starmap(run_one, [(a.cl1, n, a.all_chunks) for n in todo]):
            (a.out / f"{r['name']}.json").write_text(json.dumps(r, indent=1, default=float))
            print(r["name"], r["failure_mode"], "chunks", r["n_chunks"], flush=True)


if __name__ == "__main__":
    main()
