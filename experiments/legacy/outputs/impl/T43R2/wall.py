"""For flagged chunks (new<0<=old, or release with new<0): which scene geom is nearest the worst held row's
sphere (on), and the analytic clearance |c - geom| - r (boxes exact, meshes by vertices)."""
import sys, json, glob, pathlib, numpy as np, mujoco
sys.path.insert(0, "/tmp/claude-0/-mnt-dev-work/769cea2f-2a69-4fbe-beed-8be40017af85/scratchpad/r2")
from multiprocessing import Pool
OUT = pathlib.Path("/tmp/claude-0/-mnt-dev-work/769cea2f-2a69-4fbe-beed-8be40017af85/scratchpad/r2")

def one(path):
    from replay import Run, plane_surface, held_spheres_of
    d = json.load(open(path))
    flagged = [(A, c) for A in d["attaches"] if "error" not in A for c in A["chunks"]
               if (min(c["new_mm"]) < 0 <= min(c["old_mm"])) or (c["last_held"] and min(c["new_mm"]) < 0)]
    if not flagged: return []
    idx = json.load(open(sys.argv[1])); run = d["run"]
    R = Run(run, idx[run])
    man = json.loads((pathlib.Path(run) / "manifest.json").read_text())
    xml = man["scene"]["model_xml"]
    argv = (man.get("extra") or {}).get("argv") or []
    if "--obstacle-config" in argv: xml = xml.replace("model_transport.xml", "model_transport_pick_place_obstacles.xml")
    m = mujoco.MjModel.from_xml_path(xml); dd = mujoco.MjData(m)
    fruit = man["scene"].get("episode_target_fruit") or "apple"
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    cand = []
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]; bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not (m.geom_contype[g] or m.geom_conaffinity[g]): continue
        if bn in ("crate", "table") or bn in ("apple", "orange", "pear", "banana") and bn != fruit or "obstacle" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""):
            cand.append(g)
    out = []
    cache = {}
    for A, c in flagged:
        a = A["attach"]
        if a not in cache:
            _, _, rec = R.rebuild(a, False); ag1, _, _ = R.rebuild(a, True); cache[a] = (ag1, rec)
        ag1, rec = cache[a]; s = c["seq"]
        field, chunk = R.field(s)
        with np.load(R.npz[s], allow_pickle=True) as zz: ref = np.asarray(zz["reference_chunk"], float)
        lay = R.G["layout"]; q = R.q_now(s)
        fa = ag1._frame_held_attached(q, [plane_surface(rec["support"]["plane"])], lay.full_q(lay.chunk_to_trajectory(ref)[:, :8], q).T)
        _, rows = R.held_rows(s, fa, field, chunk)
        k, i = np.unravel_index(np.argmin(rows), rows.shape)
        full = lay.full_q(lay.chunk_to_trajectory(chunk)[:, :8], q)
        T = R.G["robot"].link_pose(full[:, k], fa.parent_link)
        cc, r = held_spheres_of(fa)[i]; cb = T[:3, :3] @ cc + T[:3, 3]
        dd.qpos[:] = np.asarray(R.plan[s]["qpos"], float); mujoco.mj_kinematics(m, dd)
        cw = dd.xpos[base] + dd.xmat[base].reshape(3, 3) @ cb
        best, who = 9.0, None
        for g in cand:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                Rg = dd.geom_xmat[g].reshape(3, 3); p = Rg.T @ (cw - dd.geom_xpos[g]); h = m.geom_size[g]
                dist = float(np.linalg.norm(np.maximum(np.abs(p) - h, 0.0)))
                if np.all(np.abs(p) <= h): dist = -float(np.min(h - np.abs(p)))
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                did = m.geom_dataid[g]; va, nv = m.mesh_vertadr[did], m.mesh_vertnum[did]
                V = dd.geom_xpos[g] + m.mesh_vert[va:va + nv].astype(float) @ dd.geom_xmat[g].reshape(3, 3).T
                dist = float(np.linalg.norm(V - cw, axis=1).min())
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                Rg = dd.geom_xmat[g].reshape(3, 3); a0 = dd.geom_xpos[g] - m.geom_size[g][1] * Rg[:, 2]; b0 = dd.geom_xpos[g] + m.geom_size[g][1] * Rg[:, 2]
                t = np.clip((cw - a0) @ (b0 - a0) / ((b0 - a0) @ (b0 - a0)), 0, 1); dist = float(np.linalg.norm(cw - (a0 + t * (b0 - a0)))) - m.geom_size[g][0]
            else:
                continue
            if dist < best: best, who = dist, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        out.append(dict(run=run.split("verify/")[1], fruit=fruit, mode=A["body"].get("mode"), seq=s, last=c["last_held"], near=c["near"],
                        old=min(c["old_mm"]), new=float(1e3 * rows.min()), step=int(k), sphere=int(i), r_mm=1e3 * r,
                        nearest=who, analytic_mm=1e3 * (best - r), z_above_table_mm=1e3 * (cw[2] - 0.82) ))
    return out

if __name__ == "__main__":
    files = sorted(glob.glob(str(OUT / "scan4" / (sys.argv[2] + "*.json"))))
    res = []
    with Pool(4) as p:
        for r in p.imap_unordered(one, files): res.extend(r)
    (OUT / f"wall_{sys.argv[2].replace('/','_')}.json").write_text(json.dumps(res))
    print(len(res))
