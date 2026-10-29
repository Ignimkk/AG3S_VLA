"""Scratch harness (main code): rebuild the attach with AG3S (held_body_cover on/off) from recorded inputs,
then the TO's held rows (CollisionLinearizer.esdf_clearance) per held chunk on the recorded field + refined plan."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, re, dataclasses, functools
import numpy as np
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
from AG3S.trajopt.types import ChunkLayout
from AG3S.trajopt.linearize import CollisionLinearizer
from AG3S.ag3s.robot_models.held_object import held_spheres_of
from AG3S.ag3s.types import SupportSurface

H = 8

class _Args:
    links = None; sphere_spacing = None; max_spheres_per_capsule = None; capsule_radius_scale = None
    max_sphere_radius = None; esdf_margin = None; target_field_policy = None; phase = None
    manipulators = None; model_xml = None; hand = None; allow_uncertified = False; rows_per_step = None

@functools.lru_cache(maxsize=8)
def setup(server_log, manifest_path):
    man = json.loads(pathlib.Path(manifest_path).read_text())
    log = FIA.parse_server_log(server_log)
    man["scene"]["model_xml"] = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
    settings = FIA.resolve_settings(_Args, log, man)
    robot, qadr, _ = FIA.build_robot(settings)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, max_distance, unknown_policy = FIA.contact_policy(settings, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
    outside = max_distance if unknown_policy == "free" else -max_distance
    lin = CollisionLinearizer(robot, layout, H)
    return dict(settings=settings, robot=robot, qadr=qadr, layout=layout, outside=outside, lin=lin)

def plane_surface(pl, sid=1):
    return SupportSurface(id=sid, normal=np.asarray(pl["normal"], float), offset=float(pl["offset_m"]),
                          point_indices=np.zeros(0, np.int64), point_count=1000, rms_error=0.001)

def make_ag3s(robot, body_cover):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    cfg = AG3SConfig().with_overrides({"constraint": {"horizon": 2, "max_attached_primitives": 4},
                                       "clustering": {"gripper_max_opening": 0.07146},
                                       "esdf": {"backend": "curobo", "voxel_size": 0.02, "fine_voxel_size": 0.005,
                                                "tsdf_voxel_size": 0.005}})
    a = AG3S(cfg, robot_model=robot, constraint_robot_model=robot, attached_parent_links=("ee_left", "ee_right"))
    a.held_body_cover = bool(body_cover)
    return a

class Run:
    def __init__(self, run_dir, chunk_index):
        self.rd = pathlib.Path(run_dir); ix = chunk_index
        self.G = setup(str(pathlib.Path(ix["server"]) / "serve_safe.log"), str(self.rd / "manifest.json"))
        rows = [json.loads(l) for l in (self.rd / "frames.jsonl").read_text().splitlines() if l.strip()]
        self.plan = {r["seq"]: r for r in rows if r["kind"] == "planning"}
        self.ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}
        self.npz = {int(s): v["npz"] for s, v in ix["chunks"].items()}
        self.summ = {}
        for s, f in self.npz.items():
            with np.load(f, allow_pickle=True) as z:
                self.summ[s] = json.loads(str(z["summary_json"]))

    def q_now(self, seq):
        return FIA.q_now_from_qpos(np.asarray(self.plan[seq]["qpos"], float), self.G["qadr"])

    def attaches(self):
        return [s for s in sorted(self.summ) if (self.summ[s].get("grasp") or {}).get("attach")
                and ((self.summ[s]["grasp"].get("attached") or {}).get("held"))]

    def held_seqs(self, a):
        out = []
        for s in sorted(self.summ):
            if s < a: continue
            g = self.summ[s].get("grasp") or {}
            if g.get("state") != "held" or not g.get("attached"): break
            out.append(s)
        return out

    def rebuild(self, a, body_cover):
        """AG3S.attach on the recorded inputs of the attach chunk `a`."""
        from AG3S.ag3s.stages.geometry import fit_sphere
        from AG3S.ag3s.types import TargetGeometry
        g = self.summ[a]["grasp"]; held = g["attached"]["held"]
        f = int(re.match(r"frame (\d+)", held["snapshot_state"]).group(1))
        with np.load(self.npz[a], allow_pickle=True) as z:
            pts = np.asarray(z["target_points"], float).reshape(-1, 3)
        with np.load(self.npz[a - 1], allow_pickle=True) as zp:
            field_prev = FIA.build_field(zp, self.G["outside"])
        ag = make_ag3s(self.G["robot"], body_cover)
        pl = (held.get("support") or {}).get("plane")
        ag._last_support_surfaces = [] if pl is None else [plane_surface(pl)]
        ag._last_esdf_field = field_prev
        tg = TargetGeometry(id=0, points=pts, point_indices=np.arange(len(pts)), centroid=pts.mean(0),
                            bounding_geometry=fit_sphere(pts), attention_score=0.9, confidence=0.9)
        hand = g.get("hand", "left")
        att = ag.attach(tg, robot_state=self.q_now(a), parent_link=g["parent_link"],
                        geometry_state=self.q_now(f + 1))
        return ag, att, held

    def field(self, s):
        with np.load(self.npz[s], allow_pickle=True) as z:
            return FIA.build_field(z, self.G["outside"]), np.asarray(z["refined_chunk"], float)

    def held_rows(self, s, att, field=None, chunk=None):
        """(H,) min held-row clearance along the recorded refined plan, via the TO's linearizer."""
        G = self.G; lin = G["lin"]
        if field is None:
            field, chunk = self.field(s)
        q = self.q_now(s)
        traj = G["layout"].chunk_to_trajectory(chunk)[:, :H]
        sp = held_spheres_of(att)
        local = np.array([c for c, _ in sp]); radii = np.array([r for _, r in sp])
        lin.set_attached(local, att.parent_link, radii)
        scene = FIA.make_scene(field, G["robot"].radii, G["settings"].esdf_margin)
        scene = dataclasses.replace(scene, attached_points=local, attached_parent_link=att.parent_link, attached_radii=radii)
        cl = lin.esdf_clearance(traj, q, scene)[..., 0]
        lin.set_attached(None, None)
        return cl[:, lin.n_spheres:].min(axis=1), cl[:, lin.n_spheres:]

    def held_at_q(self, q_full, att, field):
        T = self.G["robot"].link_pose(q_full, att.parent_link)
        sp = held_spheres_of(att)
        c = np.array([T[:3, :3] @ cc + T[:3, 3] for cc, _ in sp]); r = np.array([rr for _, rr in sp])
        return float((field.distance(c) - r - self.G["settings"].esdf_margin).min())
