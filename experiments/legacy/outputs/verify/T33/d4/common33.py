"""T33 loaders for the place-phase ESDF animation (copy of T31diag/d4/fig_common.py with T33 paths; held object = the recorded
T32 held spheres (grasp.attached.held.spheres_parent_frame on the palm ee_left) instead of the T30 attached points)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, types
import numpy as np
import mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-3'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T31diag/d4'))
import p12_records as P  # noqa
from p13a_spheres import body_mesh_world  # noqa
from fig_common import plane_box_polygon, slice_values  # noqa
import AG3S.trajopt.experiments.fixed_input_ablation as T  # noqa
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap  # noqa

T33 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T33'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t33'))
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
BOX = 6


def setup(cond, port, run_dir):
    from AG3S.trajopt.limits import build_limits
    from AG3S.trajopt.types import ChunkLayout
    rec = T33 / f"{cond}_rec"
    argv = ["--constraints", str(rec / f"server_{port}/server_constraints/run_0000"), "--frames", str(run_dir / "frames.jsonl"),
            "--server-log", str(rec / f"serve_safe_{port}.log"), "--out", "/dev/null"]
    args = T.build_parser().parse_args(argv)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    settings = T.resolve_settings(args, T.parse_server_log(args.server_log), manifest)
    robot, qadr, xml = T.build_robot(settings)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, maxd, unk = T.contact_policy(settings, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
    planning = T.to_config(settings, collision=True, cost=dict(settings.cost), sqp={})
    limits = build_limits(robot, layout, dt=planning.horizon.dt, config=planning.limits)
    meter = T.Meter(robot, layout, planning.horizon.planned, settings.hand, limits)
    return dict(settings=settings, robot=robot, qadr=qadr, layout=layout, meter=meter, outside=maxd if unk == "free" else -maxd,
                fmap=FingerJointMap(robot))


class Run:
    def __init__(self, run):
        self.run = run; self.cond = run.split("_")[0]
        A = json.loads((T33 / "p" / f"{self.cond}_analysis.json").read_text())["runs"][run]
        self.A = A; self.run_dir = T33 / f"{self.cond}_rec" / run
        self.rows = {int(x["seq"]): x for x in (json.loads(l) for l in (self.run_dir / "frames.jsonl").read_text().splitlines() if l.strip()) if x.get("kind") == "planning"}
        self.chunks = [c for c in A["chunks"] if c.get("server_chunk")]
        self.by_t = {c["t"]: c for c in self.chunks}
        port = self.chunks[0]["server_chunk"].split("/")[0].split("_")[1]
        self.sh = setup(self.cond, port, self.run_dir)
        self.m = mujoco.MjModel.from_xml_path(XML); self.d = mujoco.MjData(self.m)
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.crate_geoms = [g for g in range(self.m.ngeom) if self.m.body(self.m.geom_bodyid[g]).name == "crate"
                            and (self.m.geom_contype[g] or self.m.geom_conaffinity[g]) and self.m.geom_type[g] == BOX]
        names = [str(n) for n in self.sh["robot"].sphere_link_names]
        self.S = len(names)
        self.finger_idx = [i for i, n in enumerate(names) if n in ("ee_finger_l1", "ee_finger_l2")]

    def load(self, t):
        c = self.by_t[t]
        z = np.load(T33 / f"{self.cond}_rec" / c["server_chunk"], allow_pickle=False)
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
        ch.apple_centre = ch.to_base(self.d.xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "apple")])
        ch.table_top_z = ch.to_base(np.array([0, 0, 0.82]))[2]
        lin, layout, robot = self.sh["meter"].lin, self.sh["layout"], self.sh["robot"]
        q_now = T.q_now_from_qpos(row["qpos"], self.sh["qadr"])
        refc = np.asarray(z["reference_chunk"], np.float64)
        fj = s.get("finger_joints") or {}
        Pp, _ = self.sh["fmap"].path(fj.get("opening_norm") or {}, {"left": refc[:, 7], "right": refc[:, 15]}, lin.horizon, default=robot.param_vector())
        lin.set_joint_parameter_path(Pp)
        held = ((s.get("grasp") or {}).get("attached") or {})
        held = held.get("held") if isinstance(held, dict) else None
        if isinstance(held, str):
            import ast; held = ast.literal_eval(held)
        loc = [(np.asarray(x["centre_m"], float), x["radius_mm"] / 1000.0) for x in (held or {}).get("spheres_parent_frame", [])]
        ch.held_radii = np.array([r for _, r in loc])
        ch.traj = {}
        for k, key in (("reference", "reference_chunk"), ("refined", "refined_chunk")):
            tr = layout.chunk_to_trajectory(np.asarray(z[key], np.float64))[:, :lin.horizon]
            cen = lin.sphere_states(tr, q_now)[0]
            hc = []
            for j in range(tr.shape[1]):
                q = np.asarray(q_now, float).copy(); q[layout.q_indices] = tr[:, j]
                Tp = robot.link_pose(q, "ee_left")
                hc.append(np.array([Tp[:3, :3] @ c_ + Tp[:3, 3] for c_, _ in loc]).reshape(-1, 3))
            ch.traj[k] = {"centres": cen, "radii": lin.robot_radii, "held": np.array(hc)}
        lin.set_joint_parameter_path(None)
        return ch
