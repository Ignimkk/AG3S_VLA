"""AG3S-ROI-CARVE-GPU R4 — what did the server see where a robot geom touched a neighbour fruit? (recorded fields, no re-run)

For the requests of one episode, from the closed-loop kit's recorded chunks (`--record-constraints`) and the policy record
(request-time qpos), against the MuJoCo scene of the same request:
  neighbour   the touched body's GT surface samples: the recorded field value there (main fine `esdf`, coarse
              `esdf_coarse`, hand `esdf_free` when recorded) — min, and the fraction farther than `MISSED_MM`
              (forgotten), and how many are visible now (depth within `VISIBLE_MM`)
  geom        the touching robot geom's surface samples: the GT distance to the neighbour (min), and how far they lie outside
              the recorded AG3S spheres of the link the server checks for it (`sphere_link_names`): min over spheres of
              |p − c| − r — > 0 means the sphere model does not cover that point
  plan        the recorded clearance (min over the chunk's planned steps) of that link's spheres, and the TrajOpt status

  MUJOCO_GL=osmesa .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.studies.neighbour_contact_diag \
      --kit <V> --configuration B_ROI_FPR --model-xml <xml> --episode 1345 --requests 10 11 12 13 14 15 16 \
      --geom 1052 --neighbour orange --link ee_left --out <file.json>
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde

MISSED_MM = 10.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kit", required=True)
    ap.add_argument("--configuration", required=True)
    ap.add_argument("--model-xml", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--requests", nargs="+", type=int, required=True)
    ap.add_argument("--geom", type=int, required=True, help="MuJoCo geom id of the touching robot geom")
    ap.add_argument("--neighbour", required=True, help="body name of the touched neighbour")
    ap.add_argument("--link", nargs="+", required=True, help="AG3S sphere link name(s) the server checks for that geom")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    kit = pathlib.Path(a.kit)
    eps = tde.episodes_of(kit, a.configuration)
    pdir, chunks = eps[a.episode]
    gt = tde.scene_truth(a.model_xml, tde.obstacle_profile(pdir.parent.parent))
    nb_idx = [i for i, (k, n, b) in enumerate(gt.kind) if n == a.neighbour]
    if not nb_idx:
        raise SystemExit(f"no GT geom of body {a.neighbour!r}: {sorted({n for _, n, _ in gt.kind})}")
    geom_local = gt._sample_local(a.geom)
    m, d = gt.m, gt.d
    rows = []
    for j, chunk_file, state in chunks:
        if j not in a.requests:
            continue
        step = np.load(pdir / f"step_{j:05d}.npz")
        samples, _ = gt.at(np.asarray(step["qpos"], np.float64))
        nb = np.vstack([samples[i] for i in nb_idx])
        R, o = gt.base_pose
        gp = ((geom_local @ d.geom_xmat[a.geom].reshape(3, 3).T + d.geom_xpos[a.geom]) - o) @ R
        z = np.load(chunk_file, allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        row = {"request": j, "chunk": pathlib.Path(chunk_file).name, "t_step": int(z["t_step"]), "grasp_state": state,
               "subtask": s.get("subtask"), "to_status": (s.get("to") or {}).get("status")}
        from scipy.spatial import cKDTree
        row["geom_to_neighbour_gt_min_mm"] = round(float(cKDTree(nb).query(gp)[0].min()) * 1e3, 3)
        vis = tde._visible(nb, step)
        row["neighbour_n_samples"] = int(len(nb))
        row["neighbour_visible_frac"] = round(float(vis.mean()), 4)
        for prefix in ("esdf", "esdf_coarse", "esdf_free"):
            g = tde._grid(z, prefix)
            if g is None:
                continue
            v = tde._lookup(g, nb)
            ok = np.isfinite(v)
            row[f"{prefix}_at_neighbour"] = {
                "n_in_grid": int(ok.sum()),
                "min_mm": None if not ok.any() else round(float(v[ok].min()) * 1e3, 3),
                "forgotten_frac": None if not ok.any() else round(float((v[ok] > MISSED_MM / 1e3).mean()), 4),
                "forgotten_frac_visible": None if not (ok & vis).any() else
                round(float((v[ok & vis] > MISSED_MM / 1e3).mean()), 4)}
        names = list(s.get("sphere_link_names") or [])
        sel = [k for k, n in enumerate(names) if n in a.link]
        if sel:
            c = np.asarray(z["sphere_centres"], np.float64)[sel]
            r = np.asarray(z["sphere_radii"], np.float64)[sel]
            outside = (np.linalg.norm(gp[:, None, :] - c[None], axis=2) - r[None]).min(axis=1)
            row["geom_outside_link_spheres_mm"] = {"max": round(float(outside.max()) * 1e3, 3),
                                                   "frac_uncovered": round(float((outside > 0).mean()), 4),
                                                   "n_spheres": len(sel)}
            near = cKDTree(nb).query(gp)[0] <= 0.01   # the geom's samples within 10 mm of the neighbour
            row["geom_near_neighbour"] = {"n": int(near.sum()),
                                          "outside_spheres_mm_max": None if not near.any() else
                                          round(float(outside[near].max()) * 1e3, 3)}
            cl = np.asarray(z["clearance"], np.float64)[:, sel]
            row["link_planned_clearance_min_mm"] = round(float(cl.min()) * 1e3, 3)
        # the contact region (the geom's samples within 2 mm of its closest approach) against every recorded sphere
        dn = cKDTree(nb).query(gp)[0]
        close = dn <= dn.min() + 0.002
        ca = np.asarray(z["sphere_centres"], np.float64)
        ra = np.asarray(z["sphere_radii"], np.float64)
        out_all = np.linalg.norm(gp[close][:, None, :] - ca[None], axis=2) - ra[None]
        near_link = [names[k] for k in out_all.argmin(axis=1)]
        row["contact_region"] = {"n": int(close.sum()),
                                 "outside_nearest_sphere_mm": [round(float(out_all.min(axis=1).min()) * 1e3, 3),
                                                               round(float(out_all.min(axis=1).max()) * 1e3, 3)],
                                 "nearest_sphere_links": {n: near_link.count(n) for n in sorted(set(near_link))}}
        for n in sorted(set(near_link)):
            k = [i for i, x in enumerate(names) if x == n]
            row["contact_region"].setdefault("link_planned_clearance_min_mm", {})[n] = \
                round(float(np.asarray(z["clearance"], np.float64)[:, k].min()) * 1e3, 3)
        row["neighbour_contact"] = (s.get("grasp") or {}).get("neighbour_contact")
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = {"episode": a.episode, "configuration": a.configuration, "geom": a.geom,
           "geom_name": m.geom(a.geom).name, "geom_body": m.body(int(m.geom_bodyid[a.geom])).name,
           "neighbour": a.neighbour, "link": a.link, "missed_mm": MISSED_MM, "rows": rows}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
