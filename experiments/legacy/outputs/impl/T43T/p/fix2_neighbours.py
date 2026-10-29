"""T43 T fix 2 risk — at every recorded attach: how close does any other object (ground truth: fruit and
crate bodies, point-to-geom distance from each capture point: boxes exact, meshes = nearest mesh vertex (an upper
bound of the surface distance, dense meshes)) come to the manipulated object's last pre-grasp observed points?
A neighbour inside the pad would be partly freed at the pick place while the fruit is held."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import mujoco
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import ground_replay as G

XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
OBJ = ("apple", "banana", "orange", "pear", "crate")

def main(out):
    m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
    geoms = {b: [g for g in range(m.ngeom) if m.geom_bodyid[g] == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                 and (m.geom_contype[g] or m.geom_conaffinity[g])] for b in OBJ}
    res = []
    for rd, ix in G.IX.items():
        if "chunks" not in ix or not ix.get("complete"):
            continue
        summ = {}
        for k, v in ix["chunks"].items():
            with np.load(v["npz"]) as z:
                summ[int(k)] = json.loads(str(z["summary_json"]))
        attaches = [k for k in sorted(summ) if (summ[k].get("grasp") or {}).get("attach")]
        if not attaches:
            continue
        man = json.loads((pathlib.Path(rd) / "manifest.json").read_text())
        fruit = man["scene"]["episode_target_fruit"]
        plan = {}
        for l in (pathlib.Path(rd) / "frames.jsonl").read_text().splitlines():
            if '"planning"' in l[:40]:
                r = json.loads(l); plan[r["seq"]] = r
        for a in attaches:
            pre = [k for k in sorted(summ) if k < a and summ[k]["grasp"]["state"] == "latched"]
            if not pre:
                continue
            pre = pre[-1]
            with np.load(ix["chunks"][str(pre)]["npz"]) as z:
                pts = np.asarray(z["target_points"], float)
            poses = plan[pre]["object_poses"]
            # bodies are free joints: set qpos of each object from the recorded pose
            for b in OBJ:
                jid = m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)]
                adr = m.jnt_qposadr[jid]
                d.qpos[adr:adr + 3] = poses[b]["pos"]; d.qpos[adr + 3:adr + 7] = poses[b]["quat"]
            mujoco.mj_kinematics(m, d)
            best = {}
            for b in OBJ:
                if b == fruit:
                    continue
                dmin = np.inf
                for g in geoms[b]:
                    for p in pts:
                        # point-to-geom distance via a tiny sphere: use mj_geomDistance with a temp geom is not
                        # available; approximate with the geom's bounding sphere and its box for crate walls
                        c = d.geom_xpos[g]; R = d.geom_xmat[g].reshape(3, 3)
                        loc = R.T @ (p - c)
                        t = m.geom_type[g]
                        if t == mujoco.mjtGeom.mjGEOM_BOX:
                            q = np.abs(loc) - m.geom_size[g]
                            dist = np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0)
                        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                            dist = np.linalg.norm(loc) - m.geom_size[g][0]
                        elif t == mujoco.mjtGeom.mjGEOM_MESH:
                            mid = m.geom_dataid[g]
                            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                            dist = float(np.min(np.linalg.norm(v - loc, axis=1)))   # nearest vertex
                        else:
                            dist = np.linalg.norm(loc) - m.geom_rbound[g]
                        dmin = min(dmin, dist)
                best[b] = round(1e3 * float(dmin), 1)
            res.append({"run": rd.split("/runs/")[-1], "attach": a, "pre": pre, "fruit": fruit,
                        "n_points": int(len(pts)), "neighbour_mm": best,
                        "min_mm": min(best.values())})
    pathlib.Path(out).write_text(json.dumps(res, indent=0))
    mins = np.array([r["min_mm"] for r in res])
    print(len(res), "attaches; min distance to any other object (mm): q0/q5/q50", np.percentile(mins, [0, 5, 50]).round(1),
          "; within 15 mm:", int((mins <= 15).sum()), "within 10 mm:", int((mins <= 10).sum()))

if __name__ == "__main__":
    main(sys.argv[1])
