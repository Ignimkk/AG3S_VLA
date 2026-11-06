"""Where are the voxels removed from the grasping-hand (target-free) layer? Per-voxel diagnosis of the core evaluation.

Same input and the same class rule as `AG3S-CORE-EVAL/module_eval.py` (2026-10-28): every pre-grasp request of the 18
core-evaluation runs (pick phase, before the first two-finger contact), removed voxel centres from the server chunk
(`target_free_removed_ijk`) against the MuJoCo state of the same request (client step qpos). Class = nearest ground-truth
surface sample within 8 mm (objects, robot, apple stem / leaf); otherwise `apple_inside` if within the apple's collision
radius of its centre; otherwise `none`.

Added per voxel (read only, no rule changed): distance to the nearest apple surface sample, distance to the apple centre,
inside / outside the served target ball, nearest robot geom and its body, distance to the nearest sample of any class.

usage: MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/analysis/lte_removed_voxels.py OUT_DIR
"""
from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

CORE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261028T224856+0900_b3b69311f948_8784a1265958")
CE = CORE / "verify/AG3S-CORE-EVAL"
SNAP = CORE / "snapshot_core"
sys.path.insert(0, str(SNAP))
sys.path.insert(0, str(SNAP / "integration/rby1/manipulation/src"))
from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde  # noqa: E402

XML = str(SNAP / "robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml")
CLASS_MM = 8.0


def main(out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((CE / "module/spec_core.json").read_text())
    core = {r["label"]: r for r in json.loads((CE / "module/core.json").read_text())}
    cache, vox, per_req = {}, [], []
    for r in spec:
        lab = r["label"]
        pre = []
        for x in core[lab]["rows"]:
            if "gt" not in x:
                continue
            if x["gt"]["phase"] != "pick" or x["gt"]["fingers_touch"] >= 2:
                break
            pre.append(x["request"])
        key = tuple(r["obstacle"])
        if key not in cache:
            gt = tde.scene_truth(XML, (r["obstacle"][0], r["obstacle"][1]))
            cache[key] = (gt, tde.RobotSurface(gt))
        gt, rob = cache[key]
        m = gt.m
        apple_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
        apple_i = [i for i, (k, n, b) in enumerate(gt.kind) if n == "apple"]
        apple_r = float(max(m.geom_size[g].max() for g in range(m.ngeom)
                            if int(m.geom_bodyid[g]) == apple_b and m.geom_contype[g] != 0))
        apple_vis = [(g, gt._sample_local(g)) for g in range(m.ngeom) if int(m.geom_bodyid[g]) == apple_b
                     and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and float(m.geom_size[g].max()) < 0.02]
        rob_body = np.concatenate([[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))] * len(p)
                                   for g, p in zip(rob.geoms, rob.local)])
        steps = sorted(pathlib.Path(r["client_dir"]).glob("ag3s/run_0000/step_*.npz"), key=lambda p: int(np.load(p)["t_step"]))
        chunks = r["chunks"] if isinstance(r["chunks"], list) else eval(r["chunks"])  # noqa: S307 (spec written by module_eval)
        for j in pre:
            z = np.load(chunks[j], allow_pickle=True)
            if "target_free_removed_ijk" not in z.files or not len(z["target_free_removed_ijk"]):
                continue
            s = json.loads(str(z["summary_json"]))
            qpos = np.asarray(np.load(steps[j])["qpos"], float)
            samples, _ = gt.at(qpos)
            R, o = gt.base_pose
            apple_c = (gt.d.xpos[apple_b] - o) @ R
            P = np.asarray(z["target_free_trace_origin"], float) + (np.asarray(z["target_free_removed_ijk"], float) + .5) \
                * float(z["target_free_trace_voxel_size"])
            vis = [((loc @ gt.d.geom_xmat[g].reshape(3, 3).T + gt.d.geom_xpos[g]) - o) @ R for g, loc in apple_vis]
            rp = rob.at()
            allp = np.vstack([p for p in samples if len(p)] + [rp] + vis)
            cls = np.concatenate([[("apple" if gt.kind[i][1] == "apple" else gt.kind[i][0])] * len(p)
                                  for i, p in enumerate(samples) if len(p)] + [["robot"] * len(rp)]
                                 + [["apple_stem_leaf"] * len(v) for v in vis])
            dd, ii = cKDTree(allp).query(P)
            c = np.where(dd * 1e3 <= CLASS_MM, cls[ii], "none")
            c = np.where((c == "none") & (np.linalg.norm(P - apple_c, axis=1) <= apple_r), "apple_inside", c)
            ap = np.vstack([samples[i] for i in apple_i])
            d_apple = cKDTree(ap).query(P)[0] * 1e3
            r_centre = np.linalg.norm(P - apple_c, axis=1) * 1e3
            drob, irob = cKDTree(rp).query(P)
            man = s.get("manipulated") or {}
            if man.get("centroid") is not None and man.get("radius"):
                in_ball = np.linalg.norm(P - np.asarray(man["centroid"], float), axis=1) <= float(man["radius"])
                ball_r = float(man["radius"]) * 1e3
            else:
                in_ball, ball_r = np.zeros(len(P), bool), None
            te = s.get("target_exclusion") or {}
            per_req.append({"run": lab, "request": j, "n": int(len(P)), "ball_radius_mm": ball_r, "apple_r_mm": apple_r * 1e3,
                            "n_unowned_removed": te.get("n_unowned_removed"),
                            "classes": dict(collections.Counter(c.tolist()))})
            for k in range(len(P)):
                vox.append({"run": lab, "request": j, "cls": str(c[k]), "d_any_mm": round(float(dd[k] * 1e3), 2),
                            "nearest_any": str(cls[ii[k]]), "d_apple_surf_mm": round(float(d_apple[k]), 2),
                            "r_centre_mm": round(float(r_centre[k]), 2), "apple_r_mm": round(apple_r * 1e3, 1),
                            "in_ball": bool(in_ball[k]), "d_robot_mm": round(float(drob[k] * 1e3), 2),
                            "robot_body": str(rob_body[irob[k]]),
                            "z_rel_apple_bottom_mm": round(float((P[k, 2] - (apple_c[2] - apple_r)) * 1e3), 2)})
        print(lab, len(pre), flush=True)
    (out / "lte_removed_voxels.json").write_text(json.dumps(vox))
    (out / "lte_removed_requests.json").write_text(json.dumps(per_req, ensure_ascii=False, indent=0))
    tot = collections.Counter(v["cls"] for v in vox)
    print(dict(tot), sum(tot.values()))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
