"""AG3S-GEO-GRASP S1 — geometric grasp candidates on recorded closed-loop requests, scored against ground truth.

For each run (server chunk records + client step records of one closed-loop episode) and each pre-grasp request (HSR
latch ``latched`` / ``closing``): the target points (ATS manipulated object, approximated from the recorded cloud — see
the task doc §3), candidates under both selection rules, the task state per rule, and the evaluation-only ground-truth
check of every candidate (apple mesh at the recorded qpos, obstacle collision geoms). Writes one JSON per run plus the
inputs of the figures (`npz`).

usage: python -m AG3S.ag3s.experiments.studies.geo_grasp_offline --out DIR --gt-dir DIR RUN [RUN ...]
       RUN = NAME:SERVER_DIR:CLIENT_DIR:GT_NAME
       task §3g-1 options (default off): --wrist-check --arm-check --obstacle-memory; --time-budget-ms MS (generation
       budget; omitted = the config default); task §3j: --obstacle-source field (the recorded optimizer field: the
       target-free fine layer inside its window, the coarse layer outside, trilinear). Candidate records then carry their IK joints (`q_grasp` / `q_pre`) and
       each request its client step file, for the exact MuJoCo check.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import time
from typing import Any, Optional

import numpy as np

from AG3S.ag3s.grasp.candidates import GeoGraspConfig, generate_candidates, grasp_rot_angle
from AG3S.ag3s.grasp.gripper import RBY1_LEFT
from AG3S.ag3s.grasp.ik import LinkPoseIK
from AG3S.ag3s.grasp.surface import estimate_surface

GRIP_COL = 7
LEFT_ARM_COLS = list(range(7))
CLOSE_BELOW = 0.95
GT_CONTACT_TOL = 0.005
SELECTIONS = ("vla", "geometric")


def recorded_field(z):
    """The trajectory optimizer's target-free distance from one server chunk record (task §3j reproduction): the fine
    target-free layer inside its window, the coarse layer outside, trilinear. None without a target-free layer."""
    if "esdf_free_distance" not in z.files or "esdf_coarse_distance" not in z.files:
        return None
    from scipy.ndimage import map_coordinates

    F = np.asarray(z["esdf_free_distance"], np.float32); o = np.asarray(z["esdf_free_origin"], float)
    h = float(z["esdf_free_voxel_size"])
    Cg = np.asarray(z["esdf_coarse_distance"], np.float32); oc = np.asarray(z["esdf_coarse_origin"], float)
    hc = float(z["esdf_coarse_voxel_size"])
    hi = np.array(F.shape) - 1

    def fn(P):
        P = np.asarray(P, float).reshape(-1, 3)
        idx = (P - o) / h
        inside = np.all((idx >= 0) & (idx <= hi), axis=1)
        d = np.empty(len(P))
        if inside.any():
            d[inside] = map_coordinates(F, idx[inside].T, order=1)
        if (~inside).any():
            d[~inside] = map_coordinates(Cg, ((P[~inside] - oc) / hc).T, order=1, mode="nearest")
        return d
    return fn


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sample_geom(g: dict, step: float = 0.004) -> np.ndarray:
    """Surface samples (world) of a MuJoCo primitive collision geom (box 6, cylinder 5, capsule 3, sphere 2)."""
    t, s = int(g["type"]), np.asarray(g["size"], float)
    R, p = np.asarray(g["mat"], float).reshape(3, 3), np.asarray(g["pos"], float)
    if t == 6:
        ax = [np.arange(-s[i], s[i] + 1e-9, step) if s[i] > 0 else np.zeros(1) for i in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        P = P[np.any(np.isclose(np.abs(P), s, atol=step * 0.51), axis=1)]
    elif t in (5, 3):
        r, h = s[0], s[1]
        th = np.arange(0, 2 * math.pi, step / max(r, 1e-3))
        zz = np.arange(-h, h + 1e-9, step)
        T, Z = np.meshgrid(th, zz)
        P = np.column_stack([r * np.cos(T.ravel()), r * np.sin(T.ravel()), Z.ravel()])
    elif t == 2:
        n = max(int(4 * math.pi * s[0] ** 2 / step ** 2), 20)
        d = np.random.default_rng(0).normal(size=(n, 3))
        P = s[0] * d / np.linalg.norm(d, axis=1)[:, None]
    else:
        return np.zeros((0, 3))
    return P @ R.T + p


class GroundTruth:
    """Evaluation only: the apple's **collision** geom (the convex mesh MuJoCo contacts, not the visual mesh / stem /
    leaf) at a qpos — its surface sampled every ~2 mm with face normals — and the active obstacle surfaces near it."""

    def __init__(self, path: pathlib.Path, collision_vertices: Optional[np.ndarray] = None):
        z = np.load(path, allow_pickle=True)
        self.V = (np.asarray(collision_vertices, float) if collision_vertices is not None
                  else np.asarray(z["apple_vertices_body"], float))
        self._surface_body()
        self.qadr = int(z["apple_qadr"])
        self.jq = json.loads(str(z["joint_qadr"]))
        self.obstacles = json.loads(str(z["obstacles"]))

    def _surface_body(self, step: float = 0.002):
        from scipy.spatial import ConvexHull

        h = ConvexHull(self.V)
        pts, nrm = [], []
        c = self.V.mean(axis=0)
        for tri, eq in zip(h.simplices, h.equations):
            a, b, cc = self.V[tri]
            n = eq[:3] / np.linalg.norm(eq[:3])
            if n @ (a - c) < 0:
                n = -n
            k = max(int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(cc - a)) / step)), 1)
            for i in range(k + 1):
                for j in range(k + 1 - i):
                    pts.append(a + (b - a) * i / k + (cc - a) * j / k)
                    nrm.append(n)
        self.S_body, self.N_body = np.array(pts), np.array(nrm)

    def apple(self, qpos):
        """Surface samples (world) of the collision geom at `qpos` and the body position."""
        pos, quat = qpos[self.qadr:self.qadr + 3], qpos[self.qadr + 3:self.qadr + 7]
        return self.S_body @ quat_mat(quat).T + pos, np.asarray(pos, float)

    def apple_normals(self, qpos):
        quat = qpos[self.qadr + 3:self.qadr + 7]
        return self.N_body @ quat_mat(quat).T

    def obstacle_points(self, centre, radius=0.3):
        pts = [sample_geom(g) for g in self.obstacles
               if "collision" in g["name"] and np.linalg.norm(np.asarray(g["pos"]) - centre) < radius + max(g["size"])]
        pts = [p for p in pts if len(p)]
        P = np.vstack(pts) if pts else np.zeros((0, 3))
        return P[np.linalg.norm(P - centre, axis=1) < radius] if len(P) else P


def gt_check(g, apple_v, apple_n, obst, cfg: GeoGraspConfig) -> dict[str, Any]:
    """The task doc §3 ground-truth test of one candidate."""
    from scipy.spatial import cKDTree

    from AG3S.ag3s.grasp.candidates import _hits

    tree = cKDTree(apple_v)
    d, idx = tree.query(g.contacts)
    u = (g.contacts[1] - g.contacts[0]) / np.linalg.norm(g.contacts[1] - g.contacts[0])
    c1 = float(-apple_n[idx[0]] @ u)
    c2 = float(apple_n[idx[1]] @ u)
    boxes = RBY1_LEFT.body_boxes(g.open_width)
    hit_apple = _hits(apple_v, g.T, boxes, -0.001)
    hit_obst = _hits(obst, g.T, boxes, 0.0)
    ok_contact = bool(np.all(d <= GT_CONTACT_TOL))
    ok_anti = bool(min(c1, c2) >= cfg.cone_cos)
    ok_free = hit_apple == 0 and hit_obst == 0
    return {"contact_err_mm": [round(float(v) * 1000, 2) for v in d],
            "gt_normal_angle_deg": [round(math.degrees(math.acos(max(-1, min(1, c)))), 1) for c in (c1, c2)],
            "gt_apple_hits": int(hit_apple), "gt_obstacle_hits": int(hit_obst),
            "ok_contact": ok_contact, "ok_antipodal": ok_anti, "ok_free": bool(ok_free),
            "gt_valid": bool(ok_contact and ok_anti and ok_free)}


def _selected_gt(sel_g, T_vla, apple_v, n_gt, obst, cfg):
    from AG3S.ag3s.grasp.candidates import GraspCandidate

    pseudo = GraspCandidate(index=-1, contacts=sel_g.contacts, normals=np.zeros((2, 3)), width=sel_g.width,
                            open_width=sel_g.open_width, depth=0.0, roll=0.0, T=sel_g.T, T_pre=sel_g.T_pre,
                            alignment=0.0, support=(0, 0))
    out = gt_check(pseudo, apple_v, n_gt, obst, cfg)
    out["grasp_id"] = sel_g.grasp_id
    out["vla_pos_err_mm"] = round(float(np.linalg.norm(sel_g.T[:3, 3] - T_vla[:3, 3])) * 1000, 1)
    out["vla_rot_err_deg"] = round(math.degrees(grasp_rot_angle(T_vla[:3, :3], sel_g.T[:3, :3])), 1)
    out["position"] = [round(float(v), 5) for v in sel_g.T[:3, 3]]
    return out


def run_one(name, server_dir, client_dir, gt: GroundTruth, cfg: GeoGraspConfig, out_dir: pathlib.Path, ik: LinkPoseIK,
            robot, left_idx, spec=RBY1_LEFT, arm_shapes=None) -> dict[str, Any]:
    from scipy.spatial import cKDTree

    from AG3S.ag3s.grasp.planner import GeoGraspPlanner

    chunks = sorted((pathlib.Path(server_dir) / "run_0000").glob("chunk_*.npz"))
    steps = sorted((pathlib.Path(client_dir) / "ag3s" / "run_0000").glob("step_*.npz"))
    n = min(len(chunks), len(steps))
    names = robot.joint_names
    holder: dict[str, Any] = {}
    ik_fn = lambda T, seed: ik.solve_seeds(T, holder["q"], ik.wrist_seeds(seed, extra=[holder["q"][left_idx]]))  # noqa: E731
    arm_fn = None
    if cfg.arm_check:
        from AG3S.ag3s.grasp.arm_shape import ArmChecker

        def full_q(q_sub):
            q = holder["q"].copy()
            q[left_idx] = q_sub
            return q

        arm_fn = ArmChecker(arm_shapes, robot.link_pose, full_q, cfg.scene_margin)
    planners = {s: GeoGraspPlanner(spec, dataclasses.replace(cfg, selection=s), ik_fn=ik_fn, arm_fn=arm_fn)
                for s in SELECTIONS}
    qrec = lambda q: None if q is None else [round(float(v), 6) for v in q]  # noqa: E731
    reqs, fig, decision = [], {}, None
    for k in range(n):
        z = np.load(chunks[k], allow_pickle=True)
        summ = json.loads(str(z["summary_json"]))
        step = np.load(steps[k])
        qpos = np.asarray(step["qpos"], float)
        phase = (summ.get("grasp") or {}).get("state") or "searching"
        man = summ.get("manipulated") or None
        q_full = np.array([qpos[gt.jq[nm]] if nm in gt.jq else 0.0 for nm in names])
        holder["q"] = q_full
        rec: dict[str, Any] = {"request": k, "t_step": int(step["t_step"]), "phase": phase,
                               "target_id": None if man is None else man.get("id"),
                               "target_state": None if man is None else man.get("state"), "step_file": str(steps[k])}
        cloud = np.asarray(z["filtered_cloud"], float)
        fin = np.all(np.isfinite(cloud), axis=1)
        sup = np.asarray(z["support_mask"], bool)
        if man is not None and man.get("centroid") is not None:
            cen = np.asarray(man["centroid"], float)
            near = fin & ~sup & (np.linalg.norm(cloud - cen, axis=1) <= float(man.get("radius") or 0.04) + 0.005)
        else:
            cen, near = None, np.zeros(len(cloud), bool)
        ref = np.asarray(z["reference_chunk"], float)
        closes = np.flatnonzero(ref[:, GRIP_COL] < CLOSE_BELOW)
        row = int(closes[0]) if closes.size else 15
        q_vla = q_full.copy()
        q_vla[left_idx] = ref[row, LEFT_ARM_COLS]
        T_vla = robot.link_pose(q_vla, "ee_left")
        rec.update(vla_row=row, vla_close_planned=bool(closes.size), n_frame_target_points=int(near.sum()))
        apple_v, apple_c = gt.apple(qpos)
        active = phase in ("latched", "closing")
        if active:
            n_gt = gt.apple_normals(qpos)
            obst = gt.obstacle_points(apple_c)
            if near.any():
                rec["frame_target_purity"] = round(float(np.mean(cKDTree(apple_v).query(cloud[near])[0] <= 0.005)), 4)
        field_fn = recorded_field(z) if cfg.obstacle_source == "field" else None
        if cfg.obstacle_source == "field":
            rec["field_available"] = field_fn is not None
        for sel, pl in planners.items():
            r = pl.step(field_fn=field_fn, phase=phase, target_id=None if man is None else man.get("id"),
                        target_state=None if man is None else man.get("state"), target_points=cloud[near],
                        target_centroid=cen, scene_points=cloud[fin & ~near], vla_pose=T_vla, ik_seed=q_vla[left_idx])
            out = {"generation": r["generation"], "compute_ms": r["compute_ms"], "n_candidates": r["n_candidates"],
                   "memory": r.get("memory"), "state": r["state"],
                   "candidate_stats": r["candidate_stats"]}
            if active:
                mem = pl.memory.points
                out["memory_purity"] = (round(float(np.mean(cKDTree(apple_v).query(mem)[0] <= 0.005)), 4)
                                        if len(mem) else None)
                cs = pl.last_set
                if cs is not None and cs.candidates:
                    out["candidates"] = [dict(g.record(), gt=gt_check(g, apple_v, n_gt, obst, cfg),
                                              q_grasp=qrec(g.q_grasp), q_pre=qrec(g.q_pre)) for g in cs.candidates]
                if pl.selected is not None:
                    out["selected_gt"] = _selected_gt(pl.selected, T_vla, apple_v, n_gt, obst, cfg)
                    out["selected_gt"].update(q_grasp=qrec(pl.selected.q_grasp), q_pre=qrec(pl.selected.q_pre))
            rec[sel] = out
        if decision is None and phase == "latched" and closes.size:
            decision = k
            fig = {"target_frame": cloud[near], "scene": cloud[fin & ~near & (np.linalg.norm(cloud - apple_c, axis=1) < 0.25)],
                   "apple_v": apple_v, "obst": obst, "T_vla": T_vla}
            for sel, pl in planners.items():
                fig[f"{sel}_memory"] = pl.memory.points
                cs = pl.last_set
                cl = [] if cs is None else cs.candidates
                fig[f"{sel}_T"] = np.array([g.T for g in cl]).reshape(-1, 4, 4)
                fig[f"{sel}_open"] = np.array([g.open_width for g in cl])
                fig[f"{sel}_contacts"] = np.array([g.contacts for g in cl]).reshape(-1, 2, 3)
                fig[f"{sel}_sel_T"] = pl.selected.T if pl.selected is not None else np.full((4, 4), np.nan)
                fig[f"{sel}_sel_open"] = np.array(pl.selected.open_width if pl.selected is not None else np.nan)
                fig[f"{sel}_sel_contacts"] = pl.selected.contacts if pl.selected is not None else np.full((2, 3), np.nan)
                if cs is not None and cs.surface is not None:
                    fig[f"{sel}_surf_pts"], fig[f"{sel}_surf_n"] = cs.surface.points, cs.surface.normals
        reqs.append(rec)
    out = {"name": name, "server_dir": str(server_dir), "client_dir": str(client_dir), "n_requests": n,
           "decision_request": decision, "config": cfg.record(), "requests": reqs}
    (out_dir / f"{name}.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    if fig:
        np.savez_compressed(out_dir / f"{name}_decision.npz", **fig)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--apple-geoms", default=None, help="npz from gt_geoms.py: the apple collision mesh (body frame)")
    ap.add_argument("--model-xml", default="robots/rby1/description/models/rby1a/mujoco/model_transport.xml")
    ap.add_argument("--wrist-check", action="store_true")
    ap.add_argument("--arm-check", action="store_true")
    ap.add_argument("--obstacle-memory", action="store_true")
    ap.add_argument("--time-budget-ms", type=float, default=None)
    ap.add_argument("--obstacle-source", choices=("points", "field"), default="points")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

    robot = UrdfSphereChain(parse_urdf(RBY1_URDF))
    left_idx = [i for i, nm in enumerate(robot.joint_names) if nm.startswith("left_arm_")]
    from AG3S.ag3s.grasp.ik import mjcf_limits

    xml = pathlib.Path(a.model_xml).resolve()
    ik = LinkPoseIK(robot, "ee_left", left_idx, limits=mjcf_limits(str(xml), [robot.joint_names[i] for i in left_idx]))
    over: dict[str, Any] = {"wrist_check": a.wrist_check, "arm_check": a.arm_check, "obstacle_memory": a.obstacle_memory,
                            "obstacle_source": a.obstacle_source}
    if a.time_budget_ms is not None:
        over["time_budget_ms"] = float(a.time_budget_ms)
    cfg = GeoGraspConfig(**over)
    gspec, arm_shapes = RBY1_LEFT, None
    if cfg.wrist_check or cfg.arm_check:
        from AG3S.ag3s.grasp.arm_shape import measure_arm_from_mjcf
        from AG3S.ag3s.grasp.gripper import with_wrist

        shapes = measure_arm_from_mjcf(str(xml), "left")
        arm_shapes = shapes["arm"]
        if cfg.wrist_check:
            gspec = with_wrist(RBY1_LEFT, shapes["wrist"], cfg.scene_margin)
    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.record(), indent=1))
    for spec in a.runs:
        name, sdir, cdir, gtname = spec.split(":")
        cv = None
        if a.apple_geoms:
            zz = np.load(a.apple_geoms, allow_pickle=True)
            key = [k for k in zz.files if k.startswith("i") and json.loads(str(zz[k]))["contype"] == 1]
            cv = zz["v" + key[0][1:]]
        gt = GroundTruth(pathlib.Path(a.gt_dir) / f"{gtname}.npz", collision_vertices=cv)
        t = time.time()
        r = run_one(name, sdir, cdir, gt, cfg, out_dir, ik, robot, left_idx, spec=gspec, arm_shapes=arm_shapes)
        print(name, "requests", r["n_requests"], "decision", r["decision_request"], f"{time.time() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
