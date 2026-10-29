"""Shared loaders for T31-diag D4 figures (base frame of the robot, metres)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, types
import numpy as np
import mujoco

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-3'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T31diag/d2'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T31diag/d3'))
import p12_records as P  # noqa
from p13a_spheres import body_mesh_world  # noqa
import d2_rows_ablation as D2  # noqa
import d3_records as D3  # noqa
import AG3S.trajopt.experiments.fixed_input_ablation as T  # noqa

T30 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t31'))
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
BOX = 6


class Run:
    def __init__(self, run):
        self.run = run
        self.cond = run.split("_")[0]
        A = json.loads((T30 / "p" / f"{self.cond}_analysis.json").read_text())["runs"][run]
        self.A = A
        self.run_dir = T30 / f"{self.cond}_rec" / run
        self.rows = {int(x["seq"]): x for x in (json.loads(l) for l in (self.run_dir / "frames.jsonl").read_text().splitlines() if l.strip()) if x.get("kind") == "planning"}
        self.chunks = [c for c in A["chunks"] if c.get("server_chunk")]
        self.by_t = {c["t"]: c for c in self.chunks}
        port = self.chunks[0]["server_chunk"].split("/")[0].split("_")[1]
        self.sh = D2.setup(self.cond, port, self.run_dir)
        self.att = D3.reconstruct_attached(self.cond, self.chunks, self.rows, self.sh)
        self.m = mujoco.MjModel.from_xml_path(XML); self.d = mujoco.MjData(self.m)
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.crate_geoms = [g for g in range(self.m.ngeom) if self.m.body(self.m.geom_bodyid[g]).name == "crate"
                            and (self.m.geom_contype[g] or self.m.geom_conaffinity[g]) and self.m.geom_type[g] == BOX]
        names = [str(n) for n in self.sh["robot"].sphere_link_names]
        self.S = len(names)
        self.finger_idx = [i for i, n in enumerate(names) if n in ("ee_finger_l1", "ee_finger_l2")]
        self.palm_idx = [i for i, n in enumerate(names) if n == "ee_left"]

    def load(self, t):
        c = self.by_t[t]
        z = np.load(T30 / f"{self.cond}_rec" / c["server_chunk"], allow_pickle=False)
        s = json.loads(str(z["summary_json"]))
        row = self.rows[c["seq"]]
        self.d.qpos[:] = row["qpos"]; mujoco.mj_forward(self.m, self.d)
        Rb = self.d.xmat[self.base].reshape(3, 3).copy(); tb = self.d.xpos[self.base].copy()
        ch = types.SimpleNamespace(t=t, c=c, z=z, s=s, row=row, Rb=Rb, tb=tb)
        ch.to_base = lambda pw: (np.asarray(pw) - tb) @ Rb
        ch.field = T.build_field(z, outside_distance=self.sh["outside"])
        ch.boxes = []
        for g in self.crate_geoms:
            Rg = self.d.geom_xmat[g].reshape(3, 3); pg = self.d.geom_xpos[g]; h = self.m.geom_size[g]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
            ch.boxes.append(ch.to_base(corners @ Rg.T + pg))
        ch.apple_mesh = ch.to_base(body_mesh_world(self.m, self.d, "apple", True))
        ch.apple_centre = ch.to_base(self.d.xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "apple")])
        ch.crate_centre = ch.to_base(self.d.xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "crate")])
        ch.crate_rim_z = max(b[:, 2].max() for b in ch.boxes[:5])  # floor + 4 walls
        ch.table_top_z = ch.to_base(np.array([0, 0, 0.82]))[2]
        # sphere trajectories (reference / refined), 8 steps
        lin, layout, robot = self.sh["meter"].lin, self.sh["layout"], self.sh["robot"]
        q_now = T.q_now_from_qpos(row["qpos"], self.sh["qadr"])
        refc = np.asarray(z["reference_chunk"], np.float64)
        fj = s.get("finger_joints") or {}
        Pp, _ = self.sh["fmap"].path(fj.get("opening_norm") or {}, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon, default=robot.param_vector())
        lin.set_joint_parameter_path(Pp)
        attached = self.att is not None and int((s.get("esdf_stats") or {}).get("n_attached_points", 0) or 0) > 0
        lin.set_attached(self.att.points if attached else None, "ee_finger_l1" if attached else None)
        ch.attached = attached
        ch.traj = {}
        for k, key in (("reference", "reference_chunk"), ("refined", "refined_chunk")):
            tr = layout.chunk_to_trajectory(np.asarray(z[key], np.float64))[:, :lin.horizon]
            cen = lin.sphere_states(tr, q_now)[0]
            ch.traj[k] = {"centres": cen, "radii": np.concatenate([lin.robot_radii, np.zeros(cen.shape[1] - self.S)])}
        lin.set_joint_parameter_path(None); lin.set_attached(None, None)
        return ch


def plane_box_polygon(corners, axis, value):
    """Intersection polygon (2D, the two other axes) of a convex box (8 corners) with plane x[axis] == value."""
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
    pts = []
    for i, j in edges:
        a, b = corners[i], corners[j]
        da, db = a[axis] - value, b[axis] - value
        if da == db:
            continue
        if da * db <= 0:
            u = da / (da - db); pts.append(a + u * (b - a))
    if len(pts) < 3:
        return None
    other = [k for k in range(3) if k != axis]
    q = np.array(pts)[:, other]
    cen = q.mean(axis=0); ang = np.arctan2(q[:, 1] - cen[1], q[:, 0] - cen[0])
    return q[np.argsort(ang)]


def slice_values(layer, axis, value, lim_a, lim_b, step):
    other = [k for k in range(3) if k != axis]
    ua = np.arange(lim_a[0], lim_a[1] + 1e-9, step); ub = np.arange(lim_b[0], lim_b[1] + 1e-9, step)
    A, B = np.meshgrid(ua, ub, indexing="xy")
    pts = np.zeros((A.size, 3)); pts[:, other[0]] = A.ravel(); pts[:, other[1]] = B.ravel(); pts[:, axis] = value
    g = layer.grid
    last = g.origin + (np.asarray(g.shape) - 1) * g.voxel_size
    inside = np.all((pts >= g.origin) & (pts <= last), axis=1)
    v = np.full(len(pts), np.nan)
    if inside.any():
        v[inside] = np.asarray(layer.distance(pts[inside]), np.float64)
    return ua, ub, v.reshape(A.shape)
