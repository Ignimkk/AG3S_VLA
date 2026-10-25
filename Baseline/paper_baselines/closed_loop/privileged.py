"""Privileged (simulator ground-truth) obstacle inputs of the closed loop (BASELINE-BENCH-READY).

The client sends them only with ``libero_safe_client.py --emit-privileged-obstacles`` (default off). Wire keys (world
frame, read from ``env.sim`` at the request):

| wire key | content |
|---|---|
| ``ag3s/privileged/frame`` | ``"world"`` |
| ``ag3s/privileged/obstacles`` | list per SafeLIBERO obstacle object (joint name contains ``obstacle``): ``name``, ``active`` (``CollisionMonitor`` rule), ``pos``, ``quat_wxyz`` (root body), ``geoms`` = ``[{id, name, type, size[3], pos[3], rot[9] row-major, collision, group}]`` (MuJoCo ``geom_*``) |
| ``ag3s/privileged/monitor_obstacle`` | the obstacle the client's ``CollisionMonitor`` scores (``None`` before the wait ends) |
| ``ag3s/privileged/segmentation_geom/<cam>`` | int16 (H, W) geom id per pixel, OpenCV rows (same frame as ``ag3s/depth/<cam>``); only with ``--capture-segmentation`` |

``extras`` handed to the methods (``SceneObservation.extras``, key names of ``methods/interface.PRIVILEGED_PREFIX``,
the same keys the offline loaders already use -- ``safety_common.load_recorded_call``, ``guidance_common.scene``):

| extras key | from | read by |
|---|---|---|
| ``privileged/obstacles`` | the wire list, poses turned into the **base frame** (``panda/T_world_base``) | records |
| ``privileged/obstacle_names`` | display names of the active obstacles (``moka_pot_obstacle_1`` -> ``moka_pot_1``, so ``safety_common.phrase_from_object`` gives the prompt ``"moka pot"``) | ``ObstacleCloudSource(detector, detector_prompt="obstacle_names")`` |
| ``privileged/obstacle_geoms`` | active obstacles as primitives (``GEOM_SHAPES``), base frame | ``guidance_common.scene.build_scene`` (priority 1) |
| ``privileged/segmentation_geom`` ``{cam: (H, W)}``, ``privileged/instances_geom_ids`` ``{display name: [geom ids]}``, ``privileged/target`` / ``destination`` = ``None``, ``privileged/fixture_names`` = ``[]`` | wire segmentation + every geom of each active obstacle | ``ObstacleCloudSource(gt_seg)`` (only the SafeLIBERO obstacles are instances, so nothing else becomes an obstacle) |
"""
from __future__ import annotations

import numpy as np

WIRE_PREFIX = "ag3s/privileged/"
EXTRAS_PREFIX = "privileged/"  # = methods.interface.PRIVILEGED_PREFIX (imported lazily: this module is numpy-only)
#: MuJoCo ``mjtGeom``
MJ_SPHERE, MJ_CAPSULE, MJ_ELLIPSOID, MJ_CYLINDER, MJ_BOX, MJ_MESH = 2, 3, 4, 5, 6, 7
#: how ``privileged/obstacle_geoms`` is built from the active obstacle geoms
#:   ``collision`` -- every collision geom (contype / conaffinity != 0) as its primitive (SafeLIBERO objects: boxes)
#:   ``repo``      -- the CFM repo rule (``live_viewer.GroundTruthObstacleTracker`` / ``proprioception._initialize_
#:                    obstacle_geoms``): every geom whose name contains "obstacle", a mesh as a box of ``geom_size``
#:   ``sphere``    -- one sphere per obstacle object enclosing its collision geoms (the CFM paper's "GT sphere")
GEOM_SHAPES = ("collision", "repo", "sphere")


def present(request: dict) -> bool:
    return (WIRE_PREFIX + "obstacles") in request


def display_name(name: str) -> str:
    """``moka_pot_obstacle_1`` -> ``moka_pot_1`` (the ``obstacle`` token is SafeLIBERO bookkeeping, not the object)."""
    parts = [p for p in str(name).split("_") if p != "obstacle"]
    return "_".join(parts) or str(name)


def _to_base(T_bw: np.ndarray, pos, rot9) -> tuple[np.ndarray, np.ndarray]:
    p = T_bw[:3, :3] @ np.asarray(pos, np.float64).reshape(3) + T_bw[:3, 3]
    R = T_bw[:3, :3] @ np.asarray(rot9, np.float64).reshape(3, 3)
    return p, R


def obstacles_base(request: dict, T_world_base: np.ndarray) -> list[dict]:
    """The wire obstacle list with every pose in the robot base frame (numpy arrays)."""
    if request.get(WIRE_PREFIX + "frame", "world") != "world":
        raise ValueError(f"unexpected {WIRE_PREFIX}frame {request.get(WIRE_PREFIX + 'frame')!r}")
    T_bw = np.linalg.inv(np.asarray(T_world_base, np.float64).reshape(4, 4))
    out = []
    for o in request[WIRE_PREFIX + "obstacles"]:
        p, _ = _to_base(T_bw, o["pos"], np.eye(3).reshape(9))
        geoms = []
        for g in o["geoms"]:
            gp, gR = _to_base(T_bw, g["pos"], g["rot"])
            geoms.append({"id": int(g["id"]), "name": str(g["name"]), "type": int(g["type"]),
                          "size": np.asarray(g["size"], np.float64).reshape(3), "pos": gp, "rot": gR,
                          "collision": bool(g["collision"]), "group": int(g.get("group", 0))})
        out.append({"name": str(o["name"]), "display_name": display_name(o["name"]), "active": bool(o["active"]),
                    "pos": p, "geoms": geoms})
    return out


def _primitive(g: dict, *, mesh_as_box: bool) -> dict | None:
    t, s = g["type"], g["size"]
    base = {"pos": g["pos"], "rot": g["rot"], "name": g["name"]}
    if t == MJ_SPHERE:
        return {**base, "type": "sphere", "size": [float(s[0])]}
    if t == MJ_BOX or (t == MJ_MESH and mesh_as_box) or t == MJ_ELLIPSOID:  # ellipsoid: its bounding box
        return {**base, "type": "box", "size": [float(v) for v in s[:3]]}
    if t == MJ_CYLINDER:
        return {**base, "type": "cylinder", "size": [float(s[0]), float(s[1])]}
    if t == MJ_CAPSULE:  # enclosing cylinder (radius r, half length + r)
        return {**base, "type": "cylinder", "size": [float(s[0]), float(s[1] + s[0])]}
    return None


def _box_corners(g: dict) -> np.ndarray:
    h = np.asarray(g["size"][:3], np.float64)
    if g["type"] == MJ_SPHERE:
        h = np.full(3, float(g["size"][0]))
    elif g["type"] in (MJ_CYLINDER, MJ_CAPSULE):
        h = np.array([g["size"][0], g["size"][0], g["size"][1] + (g["size"][0] if g["type"] == MJ_CAPSULE else 0.0)])
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64) * h
    return c @ np.asarray(g["rot"]).T + np.asarray(g["pos"])


def obstacle_geoms(obstacles: list[dict], shape: str = "collision") -> list[dict]:
    """``privileged/obstacle_geoms`` (``guidance_common.geometry.Obstacles.from_primitives`` format, base frame) of
    the **active** obstacles."""
    if shape not in GEOM_SHAPES:
        raise ValueError(f"shape must be one of {GEOM_SHAPES}, got {shape!r}")
    prims = []
    for o in obstacles:
        if not o["active"]:
            continue
        if shape == "sphere":
            pts = [_box_corners(g) for g in o["geoms"] if g["collision"]]
            if not pts:
                continue
            P = np.concatenate(pts)
            c = 0.5 * (P.min(0) + P.max(0))
            prims.append({"type": "sphere", "pos": c, "size": [float(np.linalg.norm(P - c, axis=1).max())],
                          "name": o["display_name"]})
            continue
        for g in o["geoms"]:
            keep = g["collision"] if shape == "collision" else ("obstacle" in g["name"].lower())
            if keep:
                p = _primitive(g, mesh_as_box=(shape == "repo"))
                if p is not None:
                    prims.append(p)
    return prims


def extras(request: dict, T_world_base: np.ndarray, *, shape: str = "collision") -> dict:
    """All privileged ``SceneObservation.extras`` of one request (see the module table). Raises ``KeyError`` when
    the client did not send them (``--emit-privileged-obstacles`` missing)."""
    if not present(request):
        raise KeyError(f"{WIRE_PREFIX}obstacles missing: run the client with --emit-privileged-obstacles")
    obst = obstacles_base(request, T_world_base)
    active = [o for o in obst if o["active"]]
    out = {EXTRAS_PREFIX + "obstacles": obst,
           EXTRAS_PREFIX + "obstacle_names": [o["display_name"] for o in active],
           EXTRAS_PREFIX + "obstacle_geoms": obstacle_geoms(obst, shape),
           EXTRAS_PREFIX + "obstacle_geoms_shape": shape,
           EXTRAS_PREFIX + "monitor_obstacle": request.get(WIRE_PREFIX + "monitor_obstacle"),
           EXTRAS_PREFIX + "instances_geom_ids": {o["display_name"]: [g["id"] for g in o["geoms"]] for o in active},
           EXTRAS_PREFIX + "target": None, EXTRAS_PREFIX + "destination": None, EXTRAS_PREFIX + "fixture_names": []}
    seg = {k[len(WIRE_PREFIX + "segmentation_geom/"):]: np.array(v) for k, v in request.items()
           if k.startswith(WIRE_PREFIX + "segmentation_geom/")}
    if seg:
        out[EXTRAS_PREFIX + "segmentation_geom"] = seg
    return out


def summary(ex: dict) -> dict:
    """Short per-call record (``calls.jsonl``)."""
    obst = ex.get(EXTRAS_PREFIX + "obstacles") or []
    return {"active": [o["name"] for o in obst if o["active"]], "n_objects": len(obst),
            "monitor_obstacle": ex.get(EXTRAS_PREFIX + "monitor_obstacle"),
            "n_geoms": len(ex.get(EXTRAS_PREFIX + "obstacle_geoms") or []),
            "geoms_shape": ex.get(EXTRAS_PREFIX + "obstacle_geoms_shape"),
            "segmentation": sorted(ex.get(EXTRAS_PREFIX + "segmentation_geom") or {}),
            "centers": {o["display_name"]: np.round(o["pos"], 4).tolist() for o in obst if o["active"]}}


__all__ = ["GEOM_SHAPES", "WIRE_PREFIX", "display_name", "extras", "obstacle_geoms", "obstacles_base", "present",
           "summary"]
