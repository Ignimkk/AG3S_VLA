"""T80 scene generator — two apples / a moved apple (task `AG3S/ag3s/docs/handoff/T80.task.md` §2, §4.2).

Fixed before any result:

* **Training range** (`training_range`): the 16D dataset metadata (`meta/randomized_episodes.jsonl`), split `train`,
  target = apple, per used arm: the apple's initial xy relative to the basket centre (basket yaw is 0 in every
  episode, so basket axes = world axes), percentile `LO_PCT`–`HI_PCT` per axis (a box).
* **Screen left / right** = the head camera `zed_left` image: its image-right axis is world −y (checked on the
  settled scene, `screen_axis`), so screen-left = world +y = robot left = the left hand's side of the basket.
* **Episodes** (`candidate_episodes`, task §6): the pool = VLA-alone (E0) runs of IFG-APPLE (policy seed =
  10 x episode + 3) in the clear stratum with `primary` success, `failure_mode == "ok"`, no regrasp and no
  neighbour-fruit contact. Per experiment one episode: among the pool episodes whose layout passes every check
  (`feasible`), the one with the most head-camera pixels of the least visible apple that matters
  (`visibility_score`; 640 x 480 `zed_left` segmentation of the settled scene), then nearest the range centre.
* **Surface gap** (`surface_gap`): `mj_geomDistance` between the collision geoms of two bodies (meshes as convex
  hulls) in the settled scene; ≥ `MIN_GAP_M` to every other fruit and the crate. On the table: the apple's xy at least
  its radius + `TABLE_MARGIN_M` inside the table top.
* **Settled positions**: the scene is reset exactly as `pi05_infer.py` does it (`reset_randomized_scene` on the
  `apple_2` model, `--remove-objects` parking, `apple_2` placed with the settled apple's z / orientation), then
  `WARMUP_STEPS` steps (the kit's `--warmup-steps`) with the arms held: every apple must move < `SETTLE_TOL_M`.

Layouts (task §6, lead decision after the implementer's geometry measurement — §2 cannot be met as written):
  A  the other side's two non-target fruits are removed; `apple_2` at the mirror point (dx, -dy) of the apple, moved
     to the nearest point of the other hand's range box with the gap condition (1 mm grid search). Same-side
     neighbour fruit stays.
  B  the same side's non-target fruit is removed; the apple and `apple_2` side by side in y (screen left / right)
     with a 31 mm surface gap (1 mm above the 30 mm floor), centred on the range centre in y (the apple moved there
     with `--target-offset-xy`), both at the apple's dx. How far each is outside the 5-95 % box is reported.
  C  every non-apple fruit is removed; the apple starts at its hand's range-box centre (`--target-offset-xy`) and is
     moved +-60 mm along world x (screen up / down: forward / back) when the palm first comes within 150 mm.

  MUJOCO_GL=osmesa PYTHONPATH=. .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.studies.t80_scenes --out DIR
writes DIR/scenes.json (geometry + checks), DIR/items.json (the 6 runs) and DIR/args/<NAME>.args (one client argument
per line, the launcher reads them with `mapfile`).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
INFER = ROOT / "integration" / "rby1" / "bringup" / "pi05_infer.py"
#: IFG-APPLE aggregate (VLA alone = arm E0, seeds = 10 × episode + 3), the pool of VLA-alone successes.
IFG_APPLE_AGG = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/"
                             "20261019T132719+0900_f15e587290f6_11408fd6d27f/verify/IFG-APPLE/agg.json")
LO_PCT, HI_PCT = 5.0, 95.0
MIN_GAP_M = 0.030
B_GAP_M = (0.030, 0.050)
C_MOVE_M = 0.060
C_PALM_MM = 150.0
TABLE_MARGIN_M = 0.005
WARMUP_STEPS = 1000
SETTLE_TOL_M = 0.010   # an apple may roll a few mm in the warm-up (measured up to 8 mm); 10 mm = "not rolling away"
FRUITS = ("apple", "banana", "orange", "pear")
PROMPTS = {
    "A": {"left": "Pick up the apple on the left side of the basket and place it in the basket.",
          "right": "Pick up the apple on the right side of the basket and place it in the basket."},
    "B": {"left": "Pick up the left apple and place it in the basket.",
          "right": "Pick up the right apple and place it in the basket."},
}
#: screen side -> sign of world y (zed_left: image-right = world −y).
SCREEN_Y = {"left": +1.0, "right": -1.0}


def load_infer():
    spec = importlib.util.spec_from_file_location("_t80_pi05_infer", INFER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ops_module():
    d = str(INFER.parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    import t80_scene_ops
    return t80_scene_ops


# ---------------------------------------------------------------------------------------------------- range
def training_range(records, lo: float = LO_PCT, hi: float = HI_PCT) -> dict:
    """Per used arm: percentile box of the apple's initial xy relative to the basket (train split, target apple)."""
    out = {}
    for arm in ("left", "right"):
        rel = np.array([np.asarray(r["initial_fruit_poses"]["apple"][:2], float) - np.asarray(r["basket_pose"][:2], float)
                        for r in records if r.get("split") == "train" and r["target_fruit"] == "apple"
                        and r["used_arm"] == arm])
        if not len(rel):
            raise ValueError(f"no train episodes with target apple and arm {arm}")
        out[arm] = {"n": int(len(rel)), "pct": [lo, hi],
                    "dx": [float(np.percentile(rel[:, 0], lo)), float(np.percentile(rel[:, 0], hi))],
                    "dy": [float(np.percentile(rel[:, 1], lo)), float(np.percentile(rel[:, 1], hi))],
                    "dx_minmax": [float(rel[:, 0].min()), float(rel[:, 0].max())],
                    "dy_minmax": [float(rel[:, 1].min()), float(rel[:, 1].max())]}
    return out


def box_centre(rng_arm: dict) -> np.ndarray:
    return np.array([np.mean(rng_arm["dx"]), np.mean(rng_arm["dy"])])


def outside_mm(rel_xy, rng_arm: dict, key: str = "") -> float:
    """How far (mm, Euclidean) `rel_xy` is outside the box (`dx`/`dy`, or `dx_minmax`/`dy_minmax` with key '_minmax')."""
    bx, by = rng_arm["dx" + key], rng_arm["dy" + key]
    ex = max(bx[0] - rel_xy[0], 0.0, rel_xy[0] - bx[1])
    ey = max(by[0] - rel_xy[1], 0.0, rel_xy[1] - by[1])
    return float(np.hypot(ex, ey) * 1e3)


def in_range(rel_xy, rng_arm: dict) -> bool:
    return outside_mm(rel_xy, rng_arm) == 0.0


# ---------------------------------------------------------------------------------------------------- episodes
def candidate_episodes(agg_path: pathlib.Path, records_by_ep: dict, rng: dict) -> list[dict]:
    """The VLA-alone pool (module doc): qualifying E0 episodes of both hands, nearest the range centre first."""
    agg = json.loads(pathlib.Path(agg_path).read_text())
    out = []
    for r in agg["rows"]:
        e = r["arms"]["E0"]
        rec = records_by_ep.get(int(r["episode"]))
        if (r["stratum"] != "clear" or not e.get("primary") or e.get("failure_mode") != "ok" or e.get("regrasp")
                or (e.get("contacts") or {}).get("neighbour") or rec is None or rec["target_fruit"] != "apple"):
            continue
        arm = rec["used_arm"]
        rel = np.asarray(rec["initial_fruit_poses"]["apple"][:2], float) - np.asarray(rec["basket_pose"][:2], float)
        out.append({"episode": int(r["episode"]), "seed": int(r["seed"]), "arm": arm,
                    "apple_rel_xy": [float(v) for v in rel],
                    "dist_to_box_centre_mm": float(np.linalg.norm(rel - box_centre(rng[arm])) * 1e3)})
    out.sort(key=lambda x: (x["dist_to_box_centre_mm"], x["episode"]))
    return out


def visibility_score(kind: str, rep: dict) -> int:
    """Head-camera pixels of the least visible apple that matters: A / B both apples, C the apple at the start and at
    both destinations."""
    if kind == "C":
        return int(min(min(r["head_visible_px"]["apple"], r["destination"]["head_visible_px"]) for r in rep.values()))
    return int(min(rep["head_visible_px"].values()))


def feasible(kind: str, rep: dict) -> list[str]:
    """Reasons a layout is not usable (empty = usable)."""
    bad = []
    reps = list(rep.values()) if kind == "C" else [rep]
    for r in reps:
        for b, x in r["settle_displacement_mm"].items():
            if x > SETTLE_TOL_M * 1e3:
                bad.append(f"{b} settles {x:.2f} mm")
        for b, o in r["objects"].items():
            if not o["on_table"]:
                bad.append(f"{b} off the table")
            if b == "apple_2" or kind == "B":
                for other, g in o["gaps_mm"].items():
                    if other not in ("apple", "apple_2") and g < MIN_GAP_M * 1e3 - 0.05:
                        bad.append(f"{b}-{other} gap {g} mm")
        if kind == "C" and not r["destination"]["on_table"]:
            bad.append("destination off the table")
        if kind == "C" and r["destination"]["gap_to_crate_mm"] < MIN_GAP_M * 1e3:
            bad.append("destination near the crate")
    return sorted(set(bad))


# ---------------------------------------------------------------------------------------------------- geometry
def collision_geoms(m, body: str) -> list[int]:
    import mujoco
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b and (m.geom_contype[g] or m.geom_conaffinity[g])]


def _vertex_distance(m, d, ga: int, gb: int) -> float:
    """Fallback distance from the mesh vertices of `ga` (world) to geom `gb` (box: exact point-to-box; mesh:
    nearest vertex — an over-estimate of at most the vertex spacing)."""
    import mujoco
    from scipy.spatial import cKDTree

    def verts(g):
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]

    if m.geom_type[ga] != mujoco.mjtGeom.mjGEOM_MESH:
        ga, gb = gb, ga
    pa = verts(ga)
    if m.geom_type[gb] == mujoco.mjtGeom.mjGEOM_BOX:
        loc = (pa - d.geom_xpos[gb]) @ d.geom_xmat[gb].reshape(3, 3)
        q = np.abs(loc) - m.geom_size[gb]
        out = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)
        return float(out.min())
    return float(cKDTree(verts(gb)).query(pa)[0].min())


def surface_gap(m, d, body_a: str, body_b: str, distmax: float = 0.5) -> float:
    """Smallest `mj_geomDistance` between the collision geoms of two bodies (metres; negative = penetration).

    `mj_geomDistance` (MuJoCo 3.11, mesh vs box) returned 0.0 for a pair 0.42 m apart (apple vs the far crate wall,
    measured 2026-10-31). A result ≤ 0 is therefore accepted only when `mj_collision` also reports a contact between
    the two geoms; otherwise the vertex fallback `_vertex_distance` is used."""
    import mujoco
    fromto = np.zeros(6)
    best = distmax
    pairs = None
    for ga in collision_geoms(m, body_a):
        for gb in collision_geoms(m, body_b):
            v = float(mujoco.mj_geomDistance(m, d, ga, gb, distmax, fromto))
            if v <= 0.0:
                if pairs is None:
                    mujoco.mj_collision(m, d)
                    pairs = {tuple(sorted((int(c.geom1), int(c.geom2)))) for c in d.contact[:d.ncon]}
                if tuple(sorted((ga, gb))) not in pairs:
                    v = _vertex_distance(m, d, ga, gb)
            best = min(best, v)
    return best


def on_table(m, d, xy, radius: float) -> bool:
    import mujoco
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    c, s = d.geom_xpos[g], m.geom_size[g]
    return bool(abs(xy[0] - c[0]) <= s[0] - radius - TABLE_MARGIN_M and abs(xy[1] - c[1]) <= s[1] - radius - TABLE_MARGIN_M)


def apple_radius(m) -> float:
    """Horizontal half-extent of the apple collision mesh (largest |x|, |y| vertex)."""
    import mujoco
    out = 0.0
    for g in collision_geoms(m, "apple"):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            out = max(out, float(np.abs(v[:, :2]).max()))
    return out


class Scene:
    """One settled scene of the client (apple_2 model), built exactly like `pi05_infer.main` builds it."""

    def __init__(self, infer, ops, episode: int, *, target_offset=None, remove=(), extra_xy=None, extra_model=True):
        import mujoco
        self.mj, self.infer, self.ops = mujoco, infer, ops
        # the client loads the apple_2 model only with --extra-apple (C runs on the shipped model)
        self.extra_model = bool(extra_model or extra_xy is not None)
        self.m = (ops.build_model_with_extra_apple(infer.MODEL_XML_TRANSPORT) if self.extra_model
                  else mujoco.MjModel.from_xml_path(infer.MODEL_XML_TRANSPORT))
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop"))
        infer.hold_keyframe_pose(m, d)
        for i in range(m.nu):   # as main(): every actuator at the reset pose, arm_6 at the teleop key
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
        rec = infer.load_randomized_episode(episode)
        if target_offset is not None:
            rec = infer.apply_target_offset_xy(rec, target_offset)
        self.record = rec
        infer.reset_randomized_scene(m, d, rec)
        for k, obj in enumerate(remove):
            ops.place_free_body(m, d, f"{obj}_free", [-1.5, -2.5 - 0.3 * k, 0.1, 1.0, 0.0, 0.0, 0.0])
        mujoco.mj_forward(m, d)
        self.removed = tuple(remove)
        if extra_xy is not None:
            qa, _ = ops.free_qpos_adr(m, "apple_free")
            ops.place_free_body(m, d, ops.EXTRA_JOINT, [extra_xy[0], extra_xy[1], *d.qpos[qa + 2:qa + 7]])
            mujoco.mj_forward(m, d)
        self.has_extra = extra_xy is not None

    def pos(self, body: str) -> np.ndarray:
        return self.d.xpos[self.m.body(body).id].copy()

    def basket(self) -> np.ndarray:
        return self.pos("crate")

    def rel(self, body: str) -> np.ndarray:
        return (self.pos(body) - self.basket())[:2]

    def present_fruits(self) -> list[str]:
        out = [f for f in FRUITS if f not in self.removed]
        return out + (["apple_2"] if self.has_extra else [])   # (a parked apple_2 is never a neighbour)

    def gaps(self, body: str) -> dict:
        return {o: surface_gap(self.m, self.d, body, o) for o in self.present_fruits() + ["crate"] if o != body}

    def move_extra(self, xy) -> None:
        qa, _ = self.ops.free_qpos_adr(self.m, self.ops.EXTRA_JOINT)
        self.d.qpos[qa:qa + 2] = xy
        self.mj.mj_forward(self.m, self.d)

    def settle_displacement(self, steps: int = WARMUP_STEPS) -> dict:
        """The warm-up on a full copy of the state (`mj_copyData`: solver warm start included, so the result is the
        client's); per apple the displacement in metres."""
        mj = self.mj
        d2 = mj.MjData(self.m)
        mj.mj_copyData(d2, self.m, self.d)
        bodies = ["apple"] + (["apple_2"] if self.has_extra else [])
        p0 = {b: d2.xpos[self.m.body(b).id].copy() for b in bodies}
        for _ in range(steps):
            mj.mj_step(self.m, d2)
        return {b: float(np.linalg.norm(d2.xpos[self.m.body(b).id] - p0[b])) for b in bodies}

    def warm_up(self, steps: int = WARMUP_STEPS) -> dict:
        """The client's warm-up (`--warmup-steps`) in place: `mj_forward` (as `main` does before it) then `steps`
        `mj_step`. Returns the placed apple positions (world) and every apple's displacement (m)."""
        mj = self.mj
        mj.mj_forward(self.m, self.d)
        bodies = ["apple"] + (["apple_2"] if self.has_extra else [])
        p0 = {b: self.d.xpos[self.m.body(b).id].copy() for b in bodies}
        for _ in range(steps):
            mj.mj_step(self.m, self.d)
        self.placed = {b: p0[b].tolist() for b in bodies}
        self.displacement = {b: float(np.linalg.norm(self.d.xpos[self.m.body(b).id] - p0[b])) for b in bodies}
        return self.displacement

    def screen_axis(self) -> np.ndarray:
        """World direction of the head camera's image-right axis."""
        cid = self.m.camera("zed_left").id
        return self.d.cam_xmat[cid].reshape(3, 3)[:, 0].copy()


def nearest_feasible(scene: Scene, start_rel, rng_arm: dict, *, step: float = 0.001, max_r: float = 0.12):
    """Nearest point (basket frame, 1 mm grid, growing square rings) to `start_rel` inside the range box where apple_2
    has a surface gap ≥ MIN_GAP_M to every other object and is on the table. None when there is none within `max_r`."""
    b = scene.basket()[:2]
    r_apple = apple_radius(scene.m)
    start = np.asarray(start_rel, float)
    best = None
    n = int(round(max_r / step))
    for ring in range(n + 1):
        pts = []
        for i in range(-ring, ring + 1):
            for j in range(-ring, ring + 1):
                if max(abs(i), abs(j)) == ring:
                    pts.append(start + step * np.array([i, j]))
        pts.sort(key=lambda p: (float(np.linalg.norm(p - start)), p[0], p[1]))
        for p in pts:
            if best is not None and np.linalg.norm(p - start) >= best[0]:
                break
            if not in_range(p, rng_arm) or not on_table(scene.m, scene.d, b + p, r_apple):
                continue
            scene.move_extra(b + p)
            if min(scene.gaps("apple_2").values()) >= MIN_GAP_M:
                best = (float(np.linalg.norm(p - start)), p.copy())
                break
        if best is not None and ring * step > best[0]:
            break
    return None if best is None else best[1]


def pair_offset_for_gap(scene: Scene, gap: float, axis=np.array([0.0, 1.0])) -> float:
    """Centre distance along `axis` at which apple and apple_2 (both at the apple's settled orientation) have surface
    gap `gap` (bisection on mj_geomDistance)."""
    a = scene.pos("apple")[:2]
    lo, hi = 0.0, 0.3
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        scene.move_extra(a + axis * mid)
        if surface_gap(scene.m, scene.d, "apple", "apple_2") < gap:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------------------------------------- layouts
def _other(arm: str) -> str:
    return "right" if arm == "left" else "left"


def fruits_on_side(rec: dict, side_sign: float) -> list[str]:
    b = np.asarray(rec["basket_pose"][:2], float)
    return [f for f in FRUITS if f != rec["target_fruit"]
            and np.sign(rec["initial_fruit_poses"][f][1] - b[1]) == side_sign]


def _report(scene: Scene, rng: dict, bodies) -> dict:
    out = {"basket_world_xy": [float(v) for v in scene.basket()[:2]], "removed": list(scene.removed), "objects": {}}
    r_apple = apple_radius(scene.m)
    for body in bodies:
        rel = scene.rel(body)
        side = "left" if rel[1] > 0 else "right"
        out["objects"][body] = {
            "world_xyz": [float(v) for v in scene.pos(body)], "basket_rel_xy": [float(v) for v in rel],
            "range_hand": side, "outside_5_95_mm": round(outside_mm(rel, rng[side]), 1),
            "outside_minmax_mm": round(outside_mm(rel, rng[side], "_minmax"), 1),
            "on_table": on_table(scene.m, scene.d, scene.pos(body)[:2], r_apple),
            "gaps_mm": {k: round(v * 1e3, 1) for k, v in scene.gaps(body).items()}}
    return out


def r4(v) -> float:
    """The client gets every metre value with 4 decimals (`client_args`); the scenes are built from those."""
    return float(f"{float(v):.4f}")


def _warm_report(scene: Scene, rng: dict, bodies) -> dict:
    """Warm up (the client's 1000 steps), then the report on the state the first observation sees."""
    disp = scene.warm_up()
    rep = _report(scene, rng, bodies)
    rep["placed_world_xyz"] = scene.placed
    rep["settle_displacement_mm"] = {k: round(v * 1e3, 3) for k, v in disp.items()}
    rep["head_visible_px"] = head_visible_pixels(scene, list(bodies))
    return rep


def layout_A(infer, ops, ep: dict, rng: dict) -> dict:
    rec = infer.load_randomized_episode(ep["episode"])
    other = _other(ep["arm"])
    remove = fruits_on_side(rec, SCREEN_Y[other])
    scene = Scene(infer, ops, ep["episode"], remove=remove, extra_xy=[-5.0, -5.0])
    a_rel = scene.rel("apple")
    mirror = np.array([a_rel[0], -a_rel[1]])
    p = nearest_feasible(scene, mirror, rng[other])
    if p is None:
        raise ValueError(f"A ep{ep['episode']}: no feasible apple_2 point near the mirror {mirror}")
    xy = [r4(v) for v in scene.basket()[:2] + p]
    scene = Scene(infer, ops, ep["episode"], remove=remove, extra_xy=xy)
    rep = _warm_report(scene, rng, ["apple", "apple_2"])
    rep.update(mirror_rel_xy=[float(v) for v in mirror], shift_from_mirror_mm=round(float(np.linalg.norm(p - mirror) * 1e3), 1),
               client={"--extra-apple": xy, "--remove-objects": remove})
    # which apple sits on which screen side (screen left = world +y)
    rep["screen"] = {("left" if scene.rel(b)[1] > 0 else "right"): b for b in ("apple", "apple_2")}
    return rep


def layout_B(infer, ops, ep: dict, rng: dict, *, target_gap: float = MIN_GAP_M + 0.001, tries: int = 6) -> dict:
    """B, with the gap judged after the warm-up: apple_2's placement is corrected along y until the post-warm-up
    surface gap is within [30, 50] mm (aiming at `target_gap`), at most `tries` client-equivalent warm-ups."""
    rec = infer.load_randomized_episode(ep["episode"])
    arm = ep["arm"]
    remove = fruits_on_side(rec, SCREEN_Y[arm])
    probe = Scene(infer, ops, ep["episode"], remove=remove, extra_xy=[-5.0, -5.0])
    sep = pair_offset_for_gap(probe, MIN_GAP_M)
    a_rel = probe.rel("apple")
    yc = float(np.mean(rng[arm]["dy"]))
    # the pair centred on the range centre in y: apple = inner (closer to the basket), apple_2 = outer
    s = SCREEN_Y[arm]
    dy_apple = r4(yc - s * sep / 2.0 - a_rel[1])
    scene = Scene(infer, ops, ep["episode"], target_offset=(0.0, dy_apple), remove=remove, extra_xy=[-5.0, -5.0])
    sep = pair_offset_for_gap(scene, target_gap, axis=np.array([0.0, s]))
    a2 = scene.pos("apple")[:2] + np.array([0.0, s * sep])
    history = []
    for _ in range(tries):
        a2 = np.array([r4(a2[0]), r4(a2[1])])
        scene = Scene(infer, ops, ep["episode"], target_offset=(0.0, dy_apple), remove=remove, extra_xy=a2)
        scene.warm_up()
        gap = surface_gap(scene.m, scene.d, "apple", "apple_2")
        history.append({"extra_apple": a2.tolist(), "gap_after_warmup_mm": round(gap * 1e3, 2)})
        if B_GAP_M[0] <= gap <= B_GAP_M[1]:
            break
        a2 = a2 + np.array([0.0, s * (target_gap - gap)])
    else:
        raise ValueError(f"B ep{ep['episode']}: post-warm-up gap not within {B_GAP_M} after {tries} tries: {history}")
    scene = Scene(infer, ops, ep["episode"], target_offset=(0.0, dy_apple), remove=remove, extra_xy=a2)
    rep = _warm_report(scene, rng, ["apple", "apple_2"])
    gap = surface_gap(scene.m, scene.d, "apple", "apple_2")
    rep.update(apple_apple_gap_mm=round(gap * 1e3, 1), placement_tries=history,
               pair_centre_distance_mm=round(float(np.linalg.norm(scene.pos("apple")[:2] - scene.pos("apple_2")[:2])) * 1e3, 1),
               pair_centre_rel_y=float(np.mean([scene.rel("apple")[1], scene.rel("apple_2")[1]])),
               range_centre_rel_y=yc,
               client={"--target-offset-xy": [0.0, dy_apple], "--extra-apple": a2.tolist(), "--remove-objects": remove})
    rep["screen"] = {("left" if scene.rel(b)[1] > scene.rel("apple_2" if b == "apple" else "apple")[1] else "right"): b
                     for b in ("apple", "apple_2")}
    return rep


def layout_C(infer, ops, ep: dict, rng: dict) -> dict:
    """C: every non-apple fruit removed, the apple starts at its hand's range-box centre (`--target-offset-xy`) and is
    moved ±`C_MOVE_M` along world x (screen up / down: + x = forward = away from the robot = image up). The client
    runs C on the shipped model (no --extra-apple), so does this."""
    arm = ep["arm"]
    remove = [f for f in FRUITS if f != "apple"]
    probe = Scene(infer, ops, ep["episode"], remove=remove, extra_model=False)
    off = [r4(v) for v in box_centre(rng[arm]) - probe.rel("apple")]
    out = {}
    for direction, sign in (("forward", +1.0), ("back", -1.0)):
        dxy = [r4(sign * C_MOVE_M), 0.0]
        scene = Scene(infer, ops, ep["episode"], target_offset=tuple(off), remove=remove, extra_model=False)
        rep = _warm_report(scene, rng, ["apple"])
        rep["initial_palm_distance_mm"] = round(ops.palm_target_distance(scene.m, scene.d, arm, "apple") * 1e3, 1)
        dest = scene.pos("apple")[:2] + np.asarray(dxy)
        dest_rel = dest - scene.basket()[:2]
        qa, _ = ops.free_qpos_adr(scene.m, "apple_free")
        start_q = scene.d.qpos[qa:qa + 7].copy()
        scene.d.qpos[qa:qa + 2] = dest
        scene.mj.mj_forward(scene.m, scene.d)
        rep["destination"] = {"world_xy": [float(v) for v in dest], "basket_rel_xy": [float(v) for v in dest_rel],
                              "outside_5_95_mm": round(outside_mm(dest_rel, rng[arm]), 1),
                              "outside_minmax_mm": round(outside_mm(dest_rel, rng[arm], "_minmax"), 1),
                              "on_table": on_table(scene.m, scene.d, dest, apple_radius(scene.m)),
                              "gap_to_crate_mm": round(surface_gap(scene.m, scene.d, "apple", "crate") * 1e3, 1),
                              "head_visible_px": head_visible_pixels(scene, ["apple"])["apple"],
                              "note": "the move happens when the palm first comes within 150 mm; dxy is relative to "
                                      "the apple's position then — this is the destination from the first-frame pose"}
        scene.d.qpos[qa:qa + 7] = start_q
        scene.mj.mj_forward(scene.m, scene.d)
        rep["client"] = {"--target-offset-xy": off, "--remove-objects": remove,
                         "--move-target-when-palm-mm": C_PALM_MM, "--move-target-dxy": dxy}
        out[direction] = rep
    return out


def head_visible_pixels(scene: "Scene", bodies, *, width: int = 640, height: int = 480) -> dict:
    """Pixels of each body (any of its geoms) in the head camera `zed_left` segmentation of the current state."""
    import mujoco
    r = mujoco.Renderer(scene.m, height, width)
    try:
        r.enable_segmentation_rendering()
        r.update_scene(scene.d, camera="zed_left")
        seg = r.render()
    finally:
        r.close()
    oid, otype = seg[..., 0], seg[..., 1]
    is_geom = otype == int(mujoco.mjtObj.mjOBJ_GEOM)
    body = np.full(oid.shape, -1)
    body[is_geom] = scene.m.geom_bodyid[oid[is_geom]]
    return {b: int((body == scene.m.body(b).id).sum()) for b in bodies}


# ---------------------------------------------------------------------------------------------------- items
def _fmt(v: float) -> str:
    return f"{v:.4f}"


def client_args(prompt, client: dict) -> list[str]:
    out = ["--prompt", prompt] if prompt else []
    if "--target-offset-xy" in client:
        out += ["--target-offset-xy", *map(_fmt, client["--target-offset-xy"])]
    if client.get("--remove-objects"):
        out += ["--remove-objects", *client["--remove-objects"]]
    if "--extra-apple" in client:
        out += ["--extra-apple", *map(_fmt, client["--extra-apple"])]
    if "--move-target-when-palm-mm" in client:
        out += ["--move-target-when-palm-mm", f"{client['--move-target-when-palm-mm']:g}",
                "--move-target-dxy", *map(_fmt, client["--move-target-dxy"])]
    return out


def build_items(chosen: dict) -> list[dict]:
    items = []
    A = chosen["A"]
    for screen in ("left", "right"):
        items.append({"name": f"A_ep{A['episode']}_{screen}", "experiment": "A", "episode": A["episode"],
                      "seed": A["seed"], "episode_arm": A["arm"], "prompt_side": screen,
                      "requested_body": A["layout"]["screen"][screen],
                      "safe_arm": screen,   # the hand on the requested apple's side (screen left = robot left)
                      "prompt": PROMPTS["A"][screen], "args": client_args(PROMPTS["A"][screen], A["layout"]["client"])})
    B = chosen["B"]
    for screen in ("left", "right"):
        items.append({"name": f"B_ep{B['episode']}_{screen}", "experiment": "B", "episode": B["episode"],
                      "seed": B["seed"], "episode_arm": B["arm"], "prompt_side": screen,
                      "requested_body": B["layout"]["screen"][screen], "safe_arm": B["arm"],
                      "prompt": PROMPTS["B"][screen], "args": client_args(PROMPTS["B"][screen], B["layout"]["client"])})
    C = chosen["C"]
    for direction in ("forward", "back"):
        items.append({"name": f"C_ep{C['episode']}_{direction}", "experiment": "C", "episode": C["episode"],
                      "seed": C["seed"], "episode_arm": C["arm"], "prompt_side": None, "requested_body": "apple",
                      "move_direction": direction, "safe_arm": C["arm"], "prompt": None,
                      "args": client_args(None, C["layout"][direction]["client"])})
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--agg", type=pathlib.Path, default=IFG_APPLE_AGG)
    args = ap.parse_args(argv)
    infer, ops = load_infer(), ops_module()
    path = infer.RANDOMIZED_16D_DATASET / "meta" / "randomized_episodes.jsonl"
    records = [json.loads(line) for line in path.open()]
    rng = training_range(records)
    by_ep = {r["episode_index"]: r for r in records}
    pool = candidate_episodes(args.agg, by_ep, rng)
    layouts = {"A": layout_A, "B": layout_B, "C": layout_C}
    table, chosen = {k: [] for k in layouts}, {}
    for ep in pool:
        probe = Scene(infer, ops, ep["episode"])
        ep["screen_right_world_axis"] = [round(float(v), 4) for v in probe.screen_axis()]
        for kind, fn in layouts.items():
            try:
                rep = fn(infer, ops, ep, rng)
                why = feasible(kind, rep)
                score = visibility_score(kind, rep)
            except ValueError as exc:
                rep, why, score = None, [str(exc)], -1
            table[kind].append({**ep, "feasible": not why, "why_not": why, "visible_px": score, "layout": rep})
        print(f"ep{ep['episode']} ({ep['arm']}): " + ", ".join(
            f"{k} {'ok' if table[k][-1]['feasible'] else 'NO'} {table[k][-1]['visible_px']} px" for k in layouts), flush=True)
    for kind in layouts:
        ok = [x for x in table[kind] if x["feasible"]]
        if not ok:
            raise SystemExit(f"no feasible episode for experiment {kind}")
        # most visible (least visible apple that matters), then nearest the range centre, then the episode index
        ok.sort(key=lambda x: (-x["visible_px"], x["dist_to_box_centre_mm"], x["episode"]))
        chosen[kind] = ok[0]
    items = build_items(chosen)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args").mkdir(exist_ok=True)
    meta = {"dataset_meta": str(path), "training_range": rng, "min_gap_m": MIN_GAP_M, "b_gap_m": list(B_GAP_M),
            "c_move_m": C_MOVE_M, "c_palm_mm": C_PALM_MM, "warmup_steps": WARMUP_STEPS, "settle_tol_m": SETTLE_TOL_M,
            "screen": "zed_left image-right = world -y; screen-left = world +y = robot left; image-up = world +x",
            "selection": "per experiment: feasible layouts, max head-camera pixels of the least visible apple, then "
                         "nearest range centre, then episode index",
            "pool": pool, "chosen": {k: {x: v[x] for x in ("episode", "seed", "arm", "visible_px")} for k, v in chosen.items()},
            "candidates": {k: [{x: c[x] for x in ("episode", "arm", "feasible", "why_not", "visible_px")} for c in v]
                           for k, v in table.items()},
            "scenes": {k: v for k, v in chosen.items()}}
    (args.out / "scenes.json").write_text(json.dumps(meta, indent=1))
    (args.out / "items.json").write_text(json.dumps(items, indent=1))
    for it in items:
        (args.out / "args" / f"{it['name']}.args").write_text("".join(a + "\n" for a in it["args"]))
    print(json.dumps({"chosen": meta["chosen"], "n_items": len(items)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
