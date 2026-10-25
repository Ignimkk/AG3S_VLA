"""AG3S-NBC-1345 design check — would a thicker gripper sphere model see the hand touching the neighbour? (no rollout)

For the requests of one recorded episode, the AG3S constraint sphere model is built from the server argv plus each variant's
flags (`serve_safe.ag3s_from_args`), put at the request's measured joints and finger opening (policy record `qpos`, 16D state
gripper columns), and compared with the MuJoCo scene of the same request (base frame):
  geom_to_neighbour_mm       true distance between the robot body's collision geoms and the neighbour (GT surfaces)
  spheres_to_neighbour_mm    what the sphere model believes: min over the gripper links' spheres of |p − c| − r, p on the
                             neighbour's GT surface (the clearance the TrajOpt rows would see from a perfect field)
  contact_region_outside_mm  the body's samples within 2 mm of its closest approach: min over all spheres of |p − c| − r
                             (> 0 = the sphere model does not cover the touching part)

  MUJOCO_GL=osmesa .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.studies.sphere_coverage_check \
      --kit <V> --configuration B_ROI_FPR --argv-file <argv.txt> --episode 1345 --requests 11 12 13 14 15 16 17 \
      --body EE_BODY_L --neighbour orange --variant base= --variant "gripper1=--capsule-radius-scale-link gripper=1.0" --out <json>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex

import numpy as np

from AG3S.ag3s.experiments.studies import dynamic_field_replay as dfr
from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kit", required=True)
    ap.add_argument("--configuration", required=True)
    ap.add_argument("--argv-file", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--requests", nargs="+", type=int, required=True)
    ap.add_argument("--body", required=True, help="MuJoCo body whose collision geoms are the touching part")
    ap.add_argument("--neighbour", required=True)
    ap.add_argument("--variant", action="append", required=True, help="NAME=extra server flags")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from scipy.spatial import cKDTree
    from AG3S.ag3s.robot_models.gripper_state import FingerJointMap
    from AG3S.trajopt import serve_safe

    base_argv = shlex.split(pathlib.Path(a.argv_file).read_text())
    base_argv = [t for i, t in enumerate(base_argv) if t not in ("--run-dir", "--record-constraints")
                 and (i == 0 or base_argv[i - 1] not in ("--run-dir", "--record-constraints"))]
    model_xml = base_argv[base_argv.index("--model-xml") + 1]
    pdir, chunks = tde.episodes_of(pathlib.Path(a.kit), a.configuration)[a.episode]
    gt = tde.scene_truth(model_xml, tde.obstacle_profile(pdir.parent.parent))
    m, d = gt.m, gt.d
    body = m.body(a.body).id
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and (m.geom_contype[g] or m.geom_conaffinity[g])]
    local = {g: gt._sample_local(g) for g in geoms}
    nb_idx = [i for i, (k, n, b) in enumerate(gt.kind) if n == a.neighbour]
    out = {"episode": a.episode, "body": a.body, "n_geoms": len(geoms), "neighbour": a.neighbour, "variants": {}}
    for spec in a.variant:
        name, _, flags = spec.partition("=")
        args = serve_safe.build_parser().parse_args(base_argv + shlex.split(flags))
        ag3s = serve_safe.ag3s_from_args(args, plan_horizon_steps=None, rows_per_step=None)
        model = ag3s.constraint_robot_model
        fmap = FingerJointMap(model)
        names = list(model.sphere_link_names)
        rows = []
        for j, chunk_file, _ in chunks:
            if j not in a.requests:
                continue
            step = np.load(pdir / f"step_{j:05d}.npz")
            qpos = np.asarray(step["qpos"], np.float64)
            samples, _ = gt.at(qpos)
            R, o = gt.base_pose
            nb = np.vstack([samples[i] for i in nb_idx])
            gp = np.vstack([((local[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) - o) @ R for g in geoms])
            st = np.asarray(step["state"], np.float64).reshape(-1)
            model.set_joint_parameters(fmap.joint_values({"left": float(st[7]), "right": float(st[15])}))
            c, r = model.sphere_centers_numeric(dfr.robot_state_from_qpos(m, qpos))
            grip = [k for k, n in enumerate(names) if n.startswith("ee_")]
            dn = cKDTree(nb).query(gp)[0]
            close = dn <= dn.min() + 0.002
            outside = (np.linalg.norm(gp[close][:, None] - c[None], axis=2) - r[None]).min(axis=1)
            s2n = np.linalg.norm(nb[:, None] - c[grip][None], axis=2) - r[grip][None]
            k_near = grip[int(np.unravel_index(np.argmin(s2n), s2n.shape)[1])]
            rows.append({"request": j, "geom_to_neighbour_mm": round(float(dn.min()) * 1e3, 3),
                         "spheres_to_neighbour_mm": round(float(s2n.min()) * 1e3, 3), "nearest_sphere_link": names[k_near],
                         "contact_region_outside_mm": [round(float(outside.min()) * 1e3, 3), round(float(outside.max()) * 1e3, 3)],
                         "gripper_radius_mm": [round(float(r[grip].min()) * 1e3, 2), round(float(r[grip].max()) * 1e3, 2)]})
            print(name, json.dumps(rows[-1]), flush=True)
        out["variants"][name] = {"flags": flags, "rows": rows}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
