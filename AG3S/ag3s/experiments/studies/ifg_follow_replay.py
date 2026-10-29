"""IFG-FOLLOW offline — re-ground the new comparison configuration's server chunks (IFG-APPLE-NBC-TARGETSURFACE, 10 apple
scenes) with its clustering keys, then with `kept_geometry_follow` on top (`ifg_split_replay` method: fresh TargetConfirm,
the recorded attention cloud / support mask / hand spheres / latch phase). CPU only, read only.

  MUJOCO_GL=osmesa OPENBLAS_NUM_THREADS=1 .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.studies.ifg_follow_replay \
      --verify <run>/verify/IFG-APPLE-NBC-TARGETSURFACE --model-xml <snapshot>/.../model_transport.xml --out <out.json>

Per latched / closing request with nothing attached: the surface-fit centre of the manipulated geometry vs the GT apple centre
(MuJoCo state of the policy record of the same request, base frame), off and on; the `follow_kept` record. Parity: the off
replay's centroid vs the recorded `manipulated.centroid`.
"""
import argparse, glob, json, pathlib, re, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
HAND = ("ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")
#: The new comparison configuration's clustering keys (start_server.sh T2 + T44A + target split + --hold-follow-to 3
#: + --held-slip-detach).
SERVING = {"clustering": {"attention_split_radius": 0.012, "attention_split_ratio_source": "raw",
                          "closing_keep_geometry": True, "latched_identity_hold": True, "supported_max_bottom": 0.06,
                          "attention_split_partition": True, "split_supersedes_anchor": True,
                          "hold_follow_to": 3, "held_slip_detach": True}}
FOLLOW = {"clustering": {**SERVING["clustering"], "kept_geometry_follow": True}}


def episodes(verify):
    runs = {}
    for r in sorted(glob.glob(f"{verify}/runs/*/*_ep*_s*")):
        mo = re.search(r"_ep(\d+)_s(\d+)$", r)
        if mo and pathlib.Path(r).is_dir():
            runs[int(mo.group(2))] = (int(mo.group(1)), pathlib.Path(r))
    out = {}
    for sdir in sorted(glob.glob(f"{verify}/runs/server_*/server_constraints/run_0000")):
        seg, j = None, -1
        for f in sorted(glob.glob(sdir + "/chunk_*.npz")):
            with np.load(f) as z:
                seed = (json.loads(str(z["summary_json"])).get("policy_seed") or {}).get("episode")
            j = j + 1 if seed == seg else 0
            seg = seed
            if seed in runs:
                ep, rdir = runs[seed]
                out.setdefault(ep, {"run": rdir, "chunks": []})["chunks"].append((j, f))
    return out


def gt_apple(m, d, mujoco, step_npz):
    d.qpos[:] = np.load(step_npz)["qpos"]
    mujoco.mj_forward(m, d)
    b = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)   # noqa: E731
    R, o = d.xmat[b("base")].reshape(3, 3), d.xpos[b("base")]
    return (d.xpos[b("apple")] - o) @ R


def replay(ep_rec, overrides, gt_of):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.robot_models.held_object import fit_surface_sphere
    from AG3S.ag3s.stages.admissibility import HandSpheres
    from AG3S.ag3s.stages.support_surface import fit_support_surfaces
    from AG3S.ag3s.stages.target_grounding import TargetConfirm, ground_target
    from AG3S.ag3s.types import AttentionPointCloud, PointCloud
    cfg = AG3SConfig().with_overrides(overrides)
    conf = TargetConfirm(cfg.clustering, max_opening=0.07146,
                         overlap_distance=float(cfg.pointcloud.voxel_size) or cfg.clustering.eps)
    rows = []
    for j, f in ep_rec["chunks"]:
        with np.load(f, allow_pickle=False) as c:
            s = json.loads(str(c["summary_json"]))
            excl, g = s.get("exclusion") or {}, s.get("grasp") or {}
            conf.freeze(bool(excl.get("grasp_active")) or bool(g.get("attached")))
            conf.note_latched(g.get("state") in ("latched", "closing"))
            conf.note_pre_attach_grasp(bool(excl.get("grasp_active")) and not bool(g.get("attached")))
            conf.note_subtask((s.get("subtask") or {}).get("p"))
            conf.note_placed(g.get("state") == "placed")
            names = s.get("sphere_link_names") or []
            keep = np.array([n in HAND for n in names], bool)
            if keep.any():
                conf.note_hand(HandSpheres(np.asarray(c["sphere_centres"], float)[keep],
                                           np.asarray(c["sphere_radii"], float)[keep],
                                           tuple(n for n, kk in zip(names, keep) if kk)))
            if getattr(conf, "supported_max_bottom", None) is not None:
                surf, _ = fit_support_surfaces(PointCloud(points=np.asarray(c["filtered_cloud"], np.float64)),
                                               cfg.support_surface, timestamp=0.0)
                conf.note_support(np.asarray([[*np.asarray(sf.normal, float), float(sf.offset)] for sf in surf],
                                             float).reshape(-1, 4) if surf else None)
            apc = AttentionPointCloud(PointCloud(c["attention_cloud"].astype(np.float64)), c["attention_values"],
                                      raw_attention=c["attention_raw"])
            res = ground_target(apc, cfg.clustering, seed_percentile=cfg.attention.seed_percentile,
                                seed_threshold=cfg.attention.seed_threshold, exclude_mask=c["support_mask"],
                                min_radius=cfg.geometry.min_radius, timestamp=float(c["t_step"]), confirm=conf)
        if g.get("state") not in ("latched", "closing") or g.get("attach"):
            continue
        step = ep_rec["run"] / "ag3s" / "run_0000" / f"step_{j:05d}.npz"
        tgt = res.target if res is not None else None
        if tgt is None or not step.exists():
            continue
        fit = fit_surface_sphere(np.asarray(tgt.points, np.float64).reshape(-1, 3))
        apple = gt_of(step)
        rec_c = (s.get("manipulated") or {}).get("centroid")
        mt = tgt.metrics or {}
        rows.append({"request": j, "state": g.get("state"),
                     "fit_err_mm": round(1000 * float(np.linalg.norm(np.asarray(fit.centre) - apple)), 2),
                     "centroid_vs_record_mm": None if rec_c is None else round(
                         1000 * float(np.linalg.norm(np.asarray(tgt.centroid).reshape(3) - np.asarray(rec_c))), 3),
                     "follow_applied": mt.get("kept_follow_applied"), "follow_shift_mm": mt.get("kept_follow_shift_mm")})
    return rows


def stats(v):
    v = np.asarray([x for x in v if x is not None], float)
    return {"n": int(len(v)), "median": round(float(np.median(v)), 2), "p90": round(float(np.percentile(v, 90)), 2),
            "max": round(float(v.max()), 2)} if len(v) else {"n": 0}


def main():
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True)
    ap.add_argument("--model-xml", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(a.model_xml)
    d = mujoco.MjData(m)
    gt_of = lambda p: gt_apple(m, d, mujoco, p)   # noqa: E731
    eps = episodes(a.verify)
    out = {"serving_overrides": SERVING, "follow_overrides": FOLLOW, "episodes": {}}
    for ep, rec in sorted(eps.items()):
        off, on = replay(rec, SERVING, gt_of), replay(rec, FOLLOW, gt_of)
        out["episodes"][str(ep)] = {"off": off, "on": on,
                                    "off_fit_err": stats([r["fit_err_mm"] for r in off]),
                                    "on_fit_err": stats([r["fit_err_mm"] for r in on]),
                                    "parity_off_centroid_mm": stats([r["centroid_vs_record_mm"] for r in off]),
                                    "on_applied": int(sum(1 for r in on if r["follow_applied"] == 1.0)),
                                    "on_ran": int(sum(1 for r in on if r["follow_applied"] is not None))}
        print(ep, out["episodes"][str(ep)]["off_fit_err"], out["episodes"][str(ep)]["on_fit_err"],
              "parity", out["episodes"][str(ep)]["parity_off_centroid_mm"], "applied",
              out["episodes"][str(ep)]["on_applied"], "/", out["episodes"][str(ep)]["on_ran"], flush=True)
    allr = lambda key: [r for e in out["episodes"].values() for r in e[key]]   # noqa: E731
    out["summary"] = {"off": stats([r["fit_err_mm"] for r in allr("off")]),
                      "on": stats([r["fit_err_mm"] for r in allr("on")]),
                      "off_closing": stats([r["fit_err_mm"] for r in allr("off") if r["state"] == "closing"]),
                      "on_closing": stats([r["fit_err_mm"] for r in allr("on") if r["state"] == "closing"]),
                      "parity_off_centroid_mm": stats([r["centroid_vs_record_mm"] for r in allr("off")])}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
