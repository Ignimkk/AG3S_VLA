"""T43 Z6 (1) offline — recompute the held query spheres of recorded T43 W attaches with the new cap
(`held_object.cap_held_spheres`, the code under test) and measure them against the apple mesh and
the crate inner box (implementer, CPU only, no new runs).

usage: cap_offline.py OUT_JSON

Inputs: Z1's per-run attach record (outputs/verify/T43/Z/raw/z1_held.json: apple mesh vertices
relative to the apple centre, EE pose at the attach row, apple centre) and the server's attach chunk
(grasp.attached.held: body_cover.body_parent_frame / lifted_parent_frame, body_radius_mm,
filter_spheres_parent_frame[0] = the fit sphere whose centre is the fitted centre).
Cap reference = (fit centre, body radius) — `held_cap_reference` with fit accepted (every run here).
The cap is applied in the parent (palm) frame; distances are rigid-invariant. Apple-relative world =
ee_R · c + ee_pos − apple_centre (Z1's frame; checked against Z1's uncapped sets).
Per set: n, max radius, max |c − apple centre|, half_extent_x/y = (max(c+r) − min(c−r))/2 along world
x / y, need = half_extent + 20 mm, fits = need ≤ 82 (x) / 142 (y); mesh_outside_max_mm = max over apple
mesh vertices of min over spheres (|v − c| − r) (≤ 0: every vertex covered), covered_fraction."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
from AG3S.ag3s.robot_models.held_object import cap_held_spheres

Z1 = _ag3s_workspace_path('outputs/verify/T43/Z/raw/z1_held.json')
SC = _ag3s_workspace_path('outputs/verify/T43/Y/raw/server_chunks.json')
SEGMENTS = ["V5/H/V5_ep1925_s19253", "V5/H/V5_ep1807_s18071", "V5/H/V5_ep1807_s18073", "V5/B/V5_ep1807_s18071"]
SPHERE = ["V5/H/V5_ep1925_s19251", "V5/H/V5_ep1834_s18343", "V4/H/V4_ep1925_s19253", "V5/H/V5_ep1982_s19822",
          "V5/B/V5_ep1995_s19952"]
ALLOW = [0.005, 0.010, 0.015]
MARGIN, IN_X, IN_Y = 20.0, 82.0, 142.0


def attach_held(sc, key, attach_seq):
    r = sc[key]; first = pathlib.Path(r["server_segment"]["first"]); i0 = int(first.stem.split("_")[1])
    for n in range(r["server_segment"]["n"]):
        z = np.load(first.parent / f"chunk_{i0 + n:05d}.npz", allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        if s.get("grasp", {}).get("attach"):
            assert int(z["t_step"]) == attach_seq, (key, int(z["t_step"]), attach_seq)
            return s["grasp"]["attached"]["held"]
    raise RuntimeError(f"{key}: no attach chunk")


def to_rel(sph, R, p, a):
    return [((R @ np.asarray(c) * 1000.0 + p - a), r * 1000.0) for c, r in sph]


def metrics(rel, mesh):
    c = np.array([x[0] for x in rel]); r = np.array([x[1] for x in rel])
    hx = 0.5 * (np.max(c[:, 0] + r) - np.min(c[:, 0] - r)); hy = 0.5 * (np.max(c[:, 1] + r) - np.min(c[:, 1] - r))
    gap = np.min(np.linalg.norm(mesh[:, None, :] - c[None], axis=2) - r[None], axis=1)
    return {"n": int(len(r)), "radii_mm": [round(float(v), 2) for v in r], "max_radius_mm": round(float(r.max()), 2),
            "max_centre_offset_mm": round(float(np.linalg.norm(c, axis=1).max()), 2),
            "cover_radius_about_apple_mm": round(float(np.max(np.linalg.norm(c, axis=1) + r)), 2),
            "half_extent_x_mm": round(float(hx), 2), "need_x_mm": round(float(hx + MARGIN), 2), "fits_x": bool(hx + MARGIN <= IN_X),
            "half_extent_y_mm": round(float(hy), 2), "need_y_mm": round(float(hy + MARGIN), 2), "fits_y": bool(hy + MARGIN <= IN_Y),
            "mesh_outside_max_mm": round(float(gap.max()), 2), "covered_fraction": round(float((gap <= 0.0).mean()), 4)}


def main():
    Z = json.load(open(Z1))["runs"]; S = json.load(open(SC))["runs"]; out = {"definitions": __doc__, "allowances_mm": [a * 1000 for a in ALLOW], "runs": {}}
    for key in SEGMENTS + SPHERE:
        z = Z[key]; held = attach_held(S, key, z["attach_seq"]); bc = held["body_cover"]
        R = np.asarray(z["ee_R"], float); p = np.asarray(z["ee_pos_mm"], float); a = np.asarray(z["apple_centre_mm"], float)
        mesh = np.asarray(z["apple_mesh_rel_mm"], float)
        body = [(np.asarray(s["centre_m"], float), s["radius_mm"] / 1000.0) for s in bc["body_parent_frame"]]
        lifted = [(np.asarray(s["centre_m"], float), s["radius_mm"] / 1000.0) for s in bc["lifted_parent_frame"]]
        fit_c = np.asarray(held["filter_spheres_parent_frame"][0]["centre_m"], float)
        w = float(bc["body_radius_mm"]) / 1000.0
        # sanity: our transform reproduces Z1's uncapped body set
        chk = max(float(np.linalg.norm(rc - np.asarray(zs["centre_rel_apple_mm"]))) for (rc, _), zs in
                  zip(to_rel(body, R, p, a), z["sets"]["body_parent_frame"]))
        rec = {"mode": bc.get("mode"), "segments_reason": bc.get("segments_reason"), "fit_method": (bc.get("fit") or {}).get("method"),
               "half_width_mm": round(w * 1000.0, 3), "query_set_at_attach": z["query_set"], "z1_transform_check_mm": round(chk, 4),
               "fit_centre_offset_from_apple_mm": round(float(np.linalg.norm(R @ fit_c * 1000.0 + p - a)), 2),
               "body": metrics(to_rel(body, R, p, a), mesh), "lifted": metrics(to_rel(lifted, R, p, a), mesh), "capped": {}}
        for al in ALLOW:
            capped, crec = cap_held_spheres(body, fit_c, w, al)
            rec["capped"][f"{al * 1000:.0f}"] = {**metrics(to_rel(capped, R, p, a), mesh), "n_removed": crec["n_removed"],
                                                "n_radius_capped": crec["n_radius_capped"], "n_centre_moved": crec["n_centre_moved"]}
        out["runs"][key] = rec
        c10 = rec["capped"]["10"]
        print(f"{key:28s} {rec['mode']:8s} w {rec['half_width_mm']:.1f} | body n{rec['body']['n']} r{rec['body']['max_radius_mm']} "
              f"needx {rec['body']['need_x_mm']} out {rec['body']['mesh_outside_max_mm']} | cap10 n{c10['n']} r{c10['max_radius_mm']} "
              f"off{c10['max_centre_offset_mm']} needx {c10['need_x_mm']} fits {c10['fits_x']} out {c10['mesh_outside_max_mm']} "
              f"cov {c10['covered_fraction']} | chk {chk:.3f}", flush=True)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
