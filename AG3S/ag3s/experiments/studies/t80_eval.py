"""T80 evaluation — which apple is the AG3S target on, and does it follow a moved apple (task `T80.task.md` §3, §6).

Inputs per run: the client directory (`ag3s/run_0000/step_*.npz`: per request qpos / depth / K / T; `traj.npz`
with the T80 keys; `t80.json`) and the server's chunk records (`server_constraints/run_0000/chunk_*.npz`), split per
client by `order.txt` and the first request of each episode (`summary.policy_seed.applied` is set only there).
The i-th client request pairs with the i-th chunk of that run (`module_eval` convention).

Ground truth = the client's own model (`t80_scene_ops.build_model_with_extra_apple`), surface samples of the
non-robot collision geoms (`dynamic_field_replay.GroundTruth` sampler) with `apple_2` as its own class.

Definitions (fixed before results, §3):
  target class      AG3S target points within CLASS_MM of each apple's GT surface: n1 (apple), n2 (apple_2).
                    none if n1 + n2 = 0; both if min(n1, n2) >= BOTH_SHARE x (n1 + n2); else the larger one.
  grasping hand     the hand whose palm (`<arm>_ee` site) comes closest to either apple during the run.
  hand apple        per request j: the apple whose grasping-palm distance shrinks most from request j to j + 1,
                    if it shrinks by more than APPROACH_MM; else "none" (the hand is not approaching an apple).
  agreement         target class == hand apple, over the approach requests (before the first GT-held request; all
                    requests when nothing is held) where the hand apple is not "none" (A, B).
  GT held           both fingers of the grasping hand touch that apple and it rose >= LIFT_MM (module_eval).
  removed classes   the hand layer's removed voxels (`target_free_removed_ijk`) classed by the nearest GT sample
                    within CLASS_MM (apple / apple_2 / fruit / crate / table / robot / none), at the last request
                    that has removed voxels at or before the request before the first GT-held one (else before the
                    closest approach).
  C, after the move (the first request whose qpos has the moved apple):
    new centre      centroid of the apple's GT surface samples visible (depth within 10 mm) in any camera
                    (all samples when none is visible, flagged);
    follow          requests until |target centroid - new centre| <= FOLLOW_MM ("not followed" otherwise);
    old region      within OLD_RADIUS_MM of the apple centre before the move and farther than CLASS_MM from the
                    apple's current GT surface (so the near side of the moved apple is not counted);
    old target pts / old removed voxels per request; requests until both are 0 (and stay 0 for the run).
Success / contacts are reference only (§6): requested apple in the crate at the end; MuJoCo contact rows of the
robot with the other apple, the table, the crate at the recorded control steps (frames.jsonl qpos).

  MUJOCO_GL=osmesa PYTHONPATH=<snapshot> python -m AG3S.ag3s.experiments.studies.t80_eval \\
      --out-dir <V>/ag3s/<SUB> --items <kit>/items.json --out <json>
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys

import numpy as np

from AG3S.ag3s.experiments.studies.dynamic_field_replay import FRUITS, GroundTruth
from AG3S.ag3s.experiments.studies.task_dynamic_map_eval import RobotSurface, _visible

ROOT = pathlib.Path(__file__).resolve().parents[4]
CLASS_MM = 8.0
BOTH_SHARE = 0.25
APPROACH_MM = 5.0
LIFT_MM = 20.0
FOLLOW_MM = 20.0
OLD_RADIUS_MM = 40.0
APPLES = ("apple", "apple_2")


def ops_module():
    d = str(ROOT / "integration" / "rby1" / "bringup")
    if d not in sys.path:
        sys.path.insert(0, d)
    import t80_scene_ops
    return t80_scene_ops


class T80Truth(GroundTruth):
    """`GroundTruth` of the T80 client model: `apple_2` is a fruit of its own class (`kind[i][1] == "apple_2"`)."""

    def __init__(self, model):
        import mujoco
        self.mj = mujoco
        self.m = model
        self.d = mujoco.MjData(model)
        m = self.m
        self.base = base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")

        def in_robot(b):
            while b > 0:
                if b == base:
                    return True
                b = int(m.body_parentid[b])
            return False

        self.geoms, self.local, self.kind, self.planes = [], [], [], []
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            b = int(m.geom_bodyid[g])
            if in_robot(b):
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                self.planes.append(g)
                continue
            name = m.body(b).name
            kind = ("fruit" if (name in FRUITS or name in APPLES) else "crate" if name == "crate" else
                    "table" if "table" in name or b == 0 else "obstacle")
            pts = self._sample_local(g)
            if len(pts):
                self.geoms.append(g)
                self.local.append(pts)
                self.kind.append((kind, name, b))
        self.movable = [i for i, (k, n, b) in enumerate(self.kind) if self.m.body_jntnum[b] > 0]
        self.local_full = [p.copy() for p in self.local]


def build_truth(model_xml: str, *, extra: bool = True) -> T80Truth:
    """The client's model: with `apple_2` (A, B: `--extra-apple`) or the shipped one (C)."""
    if extra:
        return T80Truth(ops_module().build_model_with_extra_apple(model_xml))
    import mujoco
    return T80Truth(mujoco.MjModel.from_xml_path(model_xml))


def class_labels(points: np.ndarray, samples, kind, robot_pts: np.ndarray, tol_mm: float = CLASS_MM) -> np.ndarray:
    """Nearest-GT-sample class of every point within `tol_mm`: apple / apple_2 / fruit / crate / table / robot / none."""
    from scipy.spatial import cKDTree
    if not len(points):
        return np.zeros(0, dtype=object)
    allp, lab = [], []
    for i, p in enumerate(samples):
        if len(p):
            k, n, _ = kind[i]
            allp.append(p)
            lab += [n if n in APPLES else k] * len(p)
    if len(robot_pts):
        allp.append(robot_pts)
        lab += ["robot"] * len(robot_pts)
    tree = cKDTree(np.vstack(allp))
    dd, ii = tree.query(points)
    lab = np.asarray(lab, dtype=object)
    return np.where(dd * 1e3 <= tol_mm, lab[ii], "none")


def target_class(n1: int, n2: int, both_share: float = BOTH_SHARE) -> str:
    tot = n1 + n2
    if tot == 0:
        return "none"
    if min(n1, n2) >= both_share * tot:
        return "both"
    return "apple" if n1 > n2 else "apple_2"


def apple_counts(tp: np.ndarray, samples, kind, tol_mm: float = CLASS_MM) -> tuple[int, int]:
    """Target points within `tol_mm` of each apple's GT surface (a point near both counts for both)."""
    from scipy.spatial import cKDTree
    out = []
    for name in APPLES:
        pts = [samples[i] for i, (k, n, b) in enumerate(kind) if n == name and len(samples[i])]
        if not len(tp) or not pts:
            out.append(0)
            continue
        out.append(int((cKDTree(np.vstack(pts)).query(tp)[0] * 1e3 <= tol_mm).sum()))
    return out[0], out[1]


def hand_apple(dist_now: dict, dist_next: dict, approach_mm: float = APPROACH_MM) -> str:
    """The apple whose palm distance shrinks most (by more than `approach_mm`), else none."""
    best, best_d = "none", -approach_mm / 1e3
    for a in dist_now:
        if a in dist_next:
            dd = dist_next[a] - dist_now[a]
            if dd < best_d:
                best, best_d = a, dd
    return best


def split_server_chunks(srv_dir: pathlib.Path) -> dict:
    """{client name: [chunk files]} of one server: episodes start where `policy_seed.applied` is set, in order.txt order."""
    order = [x.strip() for x in (srv_dir / "order.txt").read_text().splitlines() if x.strip()]
    files = sorted(glob.glob(str(srv_dir / "server_constraints" / "run_*" / "chunk_*.npz")))
    segs = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            s = json.loads(str(z["summary_json"]))
        if (s.get("policy_seed") or {}).get("applied") is not None or not segs:
            segs.append([])
        segs[-1].append(f)
    if len(segs) != len(order):
        raise ValueError(f"{srv_dir}: {len(segs)} episodes in the chunk records but {len(order)} names in order.txt")
    return dict(zip(order, segs))


class _Probe:
    """Palm sites, apple bodies, fingers of the T80 model."""

    def __init__(self, m):
        import mujoco
        self.mj, self.m = mujoco, m
        self.site = {a: m.site(f"{a}_ee").id for a in ("left", "right")}
        self.body = {a: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, a) for a in APPLES}
        self.body = {a: b for a, b in self.body.items() if b >= 0}   # C runs on the shipped model: no apple_2
        self.apples = tuple(self.body)
        self.fingers = {"left": {m.body("ee_finger_l1").id, m.body("ee_finger_l2").id},
                        "right": {m.body("ee_finger_r1").id, m.body("ee_finger_r2").id}}
        self.robot_bodies = set()
        base = m.body("base").id
        for b in range(m.nbody):
            c = b
            while c > 0 and c != base:
                c = int(m.body_parentid[c])
            if c == base:
                self.robot_bodies.add(b)

    def palm_dist(self, d) -> dict:
        return {arm: {a: float(np.linalg.norm(d.site_xpos[s] - d.xpos[self.body[a]])) for a in self.apples}
                for arm, s in self.site.items()}

    def touching(self, d, body: int) -> set:
        out = set()
        for c in d.contact[:d.ncon]:
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            if body in (b1, b2):
                out.add(b2 if b1 == body else b1)
        return out


def evaluate_run(item: dict, run_dir: pathlib.Path, chunks: list, gt: T80Truth) -> dict:
    """Per-request rows and a summary of one T80 run (see the module doc)."""
    rob = RobotSurface(gt)
    m, d = gt.m, gt.d
    pr = _Probe(m)
    steps = sorted(glob.glob(str(run_dir / "ag3s" / "run_0000" / "step_*.npz")),
                   key=lambda p: int(np.load(p)["t_step"]))
    n = min(len(steps), len(chunks))
    t80 = json.loads((run_dir / "t80.json").read_text()) if (run_dir / "t80.json").exists() else {}
    move = ((t80.get("target_move") or {}).get("record")) if t80 else None
    # pass 1: GT state per request
    rows, palm, z0 = [], [], {}
    for j in range(n):
        st = np.load(steps[j])
        qpos = np.asarray(st["qpos"], float)
        samples, _ = gt.at(qpos)
        R, o = gt.base_pose
        pd = pr.palm_dist(d)
        palm.append(pd)
        apple_c = {a: (d.xpos[pr.body[a]] - o) @ R for a in pr.apples}
        for a in pr.apples:
            z0.setdefault(a, float(apple_c[a][2]))
        held = {}
        for arm in ("left", "right"):
            for a in pr.apples:
                touch = pr.touching(d, pr.body[a])
                if pr.fingers[arm] <= touch and (apple_c[a][2] - z0[a]) * 1e3 >= LIFT_MM:
                    held[a] = arm
        rows.append({"request": j, "t_step": int(st["t_step"]), "apple_centre_base": {a: apple_c[a].tolist() for a in pr.apples},
                     "palm_mm": {arm: {a: round(v * 1e3, 1) for a, v in x.items()} for arm, x in pd.items()},
                     "gt_held": held})
    hand = min(("left", "right"), key=lambda arm: min(min(p[arm].values()) for p in palm)) if palm else None
    first_held = next((r["request"] for r in rows if r["gt_held"]), None)
    closest = (int(np.argmin([min(p[hand].values()) for p in palm])) if palm else None)
    pre_grasp = (first_held - 1 if first_held else closest)
    move_req = None
    if move is not None:
        qa = int(m.jnt_qposadr[m.joint("apple_free").id])
        for j in range(n):
            q = np.asarray(np.load(steps[j])["qpos"], float)
            if abs(q[qa] - move["qpos_after"][0]) < 0.03 and abs(q[qa + 1] - move["qpos_after"][1]) < 0.03 \
                    and np.hypot(q[qa] - move["qpos_before"][0], q[qa + 1] - move["qpos_before"][1]) > 0.03:
                move_req = j
                break
    old_c = None
    if move_req is not None:
        # the apple centre before the move (world, from the move record) in the base frame of the move request
        gt.at(np.asarray(np.load(steps[move_req])["qpos"], float))
        Rp, op = gt.base_pose
        old_c = (np.asarray(move["qpos_before"][:3], float) - op) @ Rp
    # pass 2: AG3S records
    for j in range(n):
        st = np.load(steps[j])
        qpos = np.asarray(st["qpos"], float)
        samples, _ = gt.at(qpos)
        R, o = gt.base_pose
        robot_pts = rob.at()
        row = rows[j]
        with np.load(chunks[j], allow_pickle=True) as z:
            s = json.loads(str(z["summary_json"]))
            tp = np.asarray(z["target_points"], float).reshape(-1, 3) if "target_points" in z.files else np.zeros((0, 3))
            removed = None
            if "target_free_removed_ijk" in z.files and len(z["target_free_removed_ijk"]):
                removed = (np.asarray(z["target_free_trace_origin"], float)
                           + (np.asarray(z["target_free_removed_ijk"], float) + 0.5) * float(z["target_free_trace_voxel_size"]))
        n1, n2 = apple_counts(tp, samples, gt.kind)
        row["target"] = {"n_points": int(len(tp)), "n_apple": n1, "n_apple_2": n2, "class": target_class(n1, n2),
                         "has_target": bool(s.get("has_target")), "grounding": s.get("grounding_status"),
                         "centroid_base": (tp.mean(0).tolist() if len(tp) else None)}
        row["hand_apple"] = (hand_apple(palm[j][hand], palm[j + 1][hand]) if hand and j + 1 < n else "none")
        row["grasp_state"] = (s.get("grasp") or {}).get("state")
        row["subtask"] = (s.get("subtask") or {}).get("label")
        if removed is not None:
            row["removed_by_class"] = dict(collections.Counter(class_labels(removed, samples, gt.kind, robot_pts).tolist()))
        if move_req is not None:
            if j >= move_req:
                ai = [i for i, (k, nm, b) in enumerate(gt.kind) if nm == "apple"]
                ap = np.vstack([samples[i] for i in ai])
                vis = _visible(ap, st)
                centre = (ap[vis] if vis.any() else ap).mean(0)
                from scipy.spatial import cKDTree
                tree = cKDTree(ap)
                c = {"new_centre_base": centre.tolist(), "new_centre_visible": bool(vis.any()),
                     "target_to_new_mm": (round(float(np.linalg.norm(tp.mean(0) - centre) * 1e3), 1) if len(tp) else None)}

                def old_region(P):
                    if P is None or not len(P):
                        return 0
                    near_old = np.linalg.norm(P - old_c, axis=1) * 1e3 <= OLD_RADIUS_MM
                    off_new = tree.query(P)[0] * 1e3 > CLASS_MM
                    return int((near_old & off_new).sum())
                c["old_target_points"] = old_region(tp)
                c["old_removed_voxels"] = old_region(removed)
                row["c"] = c
    # summary
    summ = {"name": item["name"], "experiment": item["experiment"], "n_requests": n, "n_client_steps": len(steps),
            "n_chunks": len(chunks), "grasping_hand": hand, "first_gt_held_request": first_held,
            "held_apple": (rows[first_held]["gt_held"] if first_held is not None else None),
            "requested_body": item.get("requested_body"), "pre_grasp_request": pre_grasp}
    if item["experiment"] in ("A", "B"):
        cls = [r["target"]["class"] for r in rows if "target" in r]
        summ["target_class_counts"] = dict(collections.Counter(cls))
        approach = [r for r in rows if first_held is None or r["request"] < first_held]
        judged = [r for r in approach if r.get("hand_apple", "none") != "none"]
        agree = [r for r in judged if r["target"]["class"] == r["hand_apple"]]
        summ["agreement"] = {"n_judged": len(judged), "n_agree": len(agree),
                             "rate": (round(len(agree) / len(judged), 3) if judged else None),
                             "first_disagreement_request": next((r["request"] for r in judged
                                                                 if r["target"]["class"] != r["hand_apple"]), None)}
        summ["vla_picked_requested"] = (None if first_held is None else
                                        item.get("requested_body") in rows[first_held]["gt_held"])
        rb = [r for r in rows[:(pre_grasp + 1 if pre_grasp is not None else 0)] if r.get("removed_by_class")]
        summ["removed_before_grasp"] = ({"request": rb[-1]["request"], "by_class": rb[-1]["removed_by_class"]}
                                        if rb else None)
    if item["experiment"] == "C":
        summ["move_request"] = move_req
        summ["move_record"] = move
        after = [r for r in rows if "c" in r]
        fol = next((k for k, r in enumerate(after)
                    if r["c"]["target_to_new_mm"] is not None and r["c"]["target_to_new_mm"] <= FOLLOW_MM), None)
        summ["requests_to_follow"] = fol if fol is not None else "not followed"
        cleared = None
        for k in range(len(after)):
            if all(r["c"]["old_target_points"] == 0 and r["c"]["old_removed_voxels"] == 0 for r in after[k:]):
                cleared = k
                break
        summ["requests_to_clear_old"] = cleared if cleared is not None else "not cleared"
        summ["grasped_new_position"] = bool(first_held is not None and move_req is not None and first_held >= move_req
                                            and "apple" in rows[first_held]["gt_held"])
    return {"summary": summ, "rows": rows}


def reference_outcome(item: dict, run_dir: pathlib.Path, gt: T80Truth) -> dict:
    """Reference only (§6): requested apple in the crate at the end; robot contact rows with the other apple, the
    table and the crate over the recorded control steps."""
    m, d = gt.m, gt.d
    pr = _Probe(m)
    sys.path.insert(0, str(ROOT / "integration" / "rby1" / "manipulation" / "src"))
    from rby1_manipulation.evaluation.transport import object_in_crate
    rows = [json.loads(x) for x in (run_dir / "frames.jsonl").read_text().splitlines() if x.strip()]
    ctrl = [r for r in rows if r.get("kind") == "control" and r.get("qpos")]
    other = [a for a in pr.apples if a != item.get("requested_body")]
    names = {"table": {m.body("table").id}, "crate": {m.body("crate").id},
             "other_apple": {pr.body[a] for a in other}}
    cnt = {k: 0 for k in names}
    for r in ctrl:
        d.qpos[:] = r["qpos"]
        gt.mj.mj_forward(m, d)
        for c in d.contact[:d.ncon]:
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            for k, bs in names.items():
                if (b1 in pr.robot_bodies and b2 in bs) or (b2 in pr.robot_bodies and b1 in bs):
                    cnt[k] += 1
    in_crate = {}
    if ctrl:
        d.qpos[:] = ctrl[-1]["qpos"]
        gt.mj.mj_forward(m, d)
        in_crate = {a: bool(object_in_crate(m, d, a)) for a in pr.apples}
    return {"robot_contact_rows": cnt, "n_control_rows": len(ctrl), "in_crate_last": in_crate,
            "requested_in_crate": in_crate.get(item.get("requested_body"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, type=pathlib.Path, help="launcher output <V>/ag3s/<SUB> (runs/, server_*)")
    ap.add_argument("--items", required=True, type=pathlib.Path)
    ap.add_argument("--model-xml", default=str(ROOT / "robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
    ap.add_argument("--out", required=True, type=pathlib.Path)
    args = ap.parse_args(argv)
    items = {it["name"]: it for it in json.loads(args.items.read_text())}
    chunks = {}
    for srv in sorted(args.out_dir.glob("server_*")):
        if (srv / "order.txt").exists():
            chunks.update(split_server_chunks(srv))
    truths = {}
    res = []
    for name, item in items.items():
        rd = args.out_dir / "runs" / name
        if name not in chunks or not rd.exists():
            res.append({"summary": {"name": name, "missing": True}})
            continue
        first = sorted(glob.glob(str(rd / "ag3s" / "run_0000" / "step_*.npz")))
        extra = bool(first) and len(np.load(first[0])["qpos"]) > 66
        if extra not in truths:
            truths[extra] = build_truth(args.model_xml, extra=extra)
        gt = truths[extra]
        r = evaluate_run(item, rd, chunks[name], gt)
        r["summary"]["reference"] = reference_outcome(item, rd, gt)
        res.append(r)
        print(json.dumps(r["summary"], default=str), flush=True)
    args.out.write_text(json.dumps(res, indent=0, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
