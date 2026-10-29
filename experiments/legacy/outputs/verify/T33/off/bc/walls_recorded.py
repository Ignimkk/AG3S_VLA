"""T33 (c) on the closed-loop records: crate wall d<0 widths in the RECORDED fields of every T33 E3a/E3b chunk whose fine window
covers the crate lines. Lines as bc_rebuild.py / T32 H4: through the GT crate centre (planning-row object pose -> base) at z = centre +
4 mm, along base x (lo = crate_wall_nx, hi = crate_wall_px) and base y (lo = ny, hi = py), +-150 mm, 0.5 mm, trilinear distance
of the recorded layer (T24 tool layer_from_arrays via fixed_input_ablation.build_field layers). Tiers: fine (esdf_distance), coarse,
free (target-free layer when recorded). Per half-line: [d<0 width lo, d<0 width hi, min distance lo (mm), min distance hi (mm)]; a half-line with 0 mm of d<0 is
counted as zero-width (its min distance says how close to 0 the field gets there).
usage: walls_recorded.py -> walls_recorded.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np, mujoco
from AG3S.ag3s.fields.curobo_field import layer_from_arrays
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T33'))
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
B = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base"); CR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")


def band(layer, p0, u, half=0.15, step=0.0005):
    s = np.arange(-half, half, step); pts = p0[None] + s[:, None] * u[None]
    g = layer.grid; last = np.asarray(g.origin) + (np.asarray(g.shape) - 1) * g.voxel_size
    ins = np.all((pts >= g.origin) & (pts <= last), axis=1)
    if not ins.all():
        return None
    v = np.asarray(layer.distance(pts), np.float64); a = s < 0
    return [float(1000 * step * (v[a] < 0).sum()), float(1000 * step * (v[~a] < 0).sum()), float(1000 * v[a].min()), float(1000 * v[~a].min())]


out = {"definitions": __doc__, "runs": {}}
for cond in ("E3a", "E3b"):
    A = json.loads((V / "p" / f"{cond}_analysis.json").read_text())["runs"]
    for run, r in A.items():
        rows = {int(x["seq"]): x for x in (json.loads(l) for l in (V / f"{cond}_rec/{run}/frames.jsonl").read_text().splitlines() if l.strip()) if x.get("kind") == "planning"}
        res = []
        for c in r["chunks"]:
            if not c.get("server_chunk"):
                continue
            z = np.load(V / f"{cond}_rec" / c["server_chunk"], allow_pickle=False)
            row = rows[c["seq"]]; d.qpos[:] = row["qpos"]; mujoco.mj_forward(m, d)
            Rb = d.xmat[B].reshape(3, 3); cc = (d.xpos[CR] - d.xpos[B]) @ Rb
            p0 = cc + np.array([0, 0, 0.004])
            e = {"t": c["t"], "phase": c["phase"], "latch": (c.get("grasp_latch") or {}).get("state")}
            for tier, key in (("fine", "esdf"), ("coarse", "esdf_coarse"), ("free", "esdf_free")):
                if f"{key}_distance" not in z.files:
                    continue
                L = layer_from_arrays(z[f"{key}_distance"].astype(np.float32), z[f"{key}_origin"], float(z[f"{key}_voxel_size"]))
                e[tier] = {"x_nx_px": band(L, p0, np.array([1.0, 0, 0])), "y_ny_py": band(L, p0, np.array([0, 1.0, 0]))}
            res.append(e)
        out["runs"][run] = res
        fx = [e["fine"]["x_nx_px"] for e in res if e.get("fine") and e["fine"]["x_nx_px"]]
        print(run, "fine x covered", len(fx), "/", len(res), "zero-lo", sum(1 for a in fx if a[0] == 0), "zero-hi", sum(1 for a in fx if a[1] == 0), flush=True)
(V / "off/bc/walls_recorded.json").write_text(json.dumps(out, indent=1))
