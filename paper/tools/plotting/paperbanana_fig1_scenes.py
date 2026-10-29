"""Hurdle scenes for the two Figure 1 variants (VLA-only failures vs AG3S successes).

Source: T43 paired closed-loop runs, episode 1982, one red overbar across the crate (profile t43p2_overbar);
E0 = VLA only (--no-safe), V11AB = AG3S main configuration; same seed within each pair. ``frames.jsonl`` holds the full 66-D qpos of every control step; the hurdle is placed with the
client's own PickPlaceObstacleManager and the run's profile config. Frames are MuJoCo replays of the recorded
states (no re-simulation). Overlays: the recorded left tool-centre path (FK of the recorded qpos) and, for a
failure, the closest gripper/hurdle point (mj_geomDistance).

    MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/paperbanana_fig1_scenes.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "integration/rby1/manipulation/src"))
from rby1_manipulation.simulation.pick_place_obstacles import (  # noqa: E402
    PickPlaceObstacleManager, load_pick_place_obstacle_config)

T40 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/workspace/verify/T40")
XML = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261013T100210+0900_4a540f1a18d7_cb037c425d2f/"
                   "snapshot/robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml")
OUT = REPO / "paper/figures/paperbanana/figure1/variants/assets"

RED = (0.85, 0.12, 0.10, 1.0)
GOLD = (0.95, 0.70, 0.02, 1.0)
GRIPPER = ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2")

T43 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/workspace/verify/T43")
CFG = T43 / "P2/configs/t43p2_ep1982_s19822.json"      # one overbar for all four scenes (ep1982)
PROFILE = "t43p2_overbar"
CAM_PICK = {"lookat": [0.56, 0.14, 0.97], "dist": 0.88, "az": 150, "el": -20}
CAM_PLACE = {"lookat": [0.55, 0.10, 1.00], "dist": 0.98, "az": 150, "el": -18}

# name: (run dir, frame t, path from t0, colour, camera). Same episode, same overbar; pick pair seed 19822,
# place pair seed 19823. E0 = VLA only (--no-safe), V11AB = AG3S (main configuration).
SCENES = {
    "vla_pick_fail": (T43 / "W/runs/E0/H/E0_ep1982_s19822", 197, 90, RED, CAM_PICK),
    "ag3s_pick_ok": (T43 / "V11/runs/V11AB/H/V11AB_ep1982_s19822", 140, 60, GOLD, CAM_PICK),
    "vla_place_fail": (T43 / "W/runs/E0/H/E0_ep1982_s19823", 245, 150, RED, CAM_PLACE),
    "ag3s_place_ok": (T43 / "V11/runs/V11AB/H/V11AB_ep1982_s19823", 212, 120, GOLD, CAM_PLACE),
}


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Replay:
    def __init__(self, W=1600, H=1200):
        self.m = mujoco.MjModel.from_xml_path(str(XML))
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = W, H
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, H, W, max_geom=20000)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
        bodies = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in GRIPPER]
        self.grip = [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in bodies]

    def hurdle_geoms(self):
        out = []
        for g in range(self.m.ngeom):
            n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "obstacle" in n and "collision" in n:
                out.append(g)
        return out

    def load(self, run_dir, cfg_path=CFG, prof=PROFILE):
        rows = [json.loads(l)["qpos"] for l in open(run_dir / "frames.jsonl") if '"kind": "control"' in l]
        if getattr(self, "mgr", None) is not None:
            self.mgr.park_all()
        self.mgr = PickPlaceObstacleManager(self.m, self.d, load_pick_place_obstacle_config(str(cfg_path)))
        self.prof = prof
        return rows

    def set(self, q):
        self.d.qpos[:] = q
        self.mgr.activate(self.prof)
        mujoco.mj_forward(self.m, self.d)

    def closest(self):
        best = (np.inf, None)
        ft = np.zeros(6)
        for a in self.grip:
            for b in self.hurdle_geoms():
                dist = mujoco.mj_geomDistance(self.m, self.d, a, b, 0.3, ft)
                if dist < best[0]:
                    best = (dist, (ft[:3] + ft[3:]) / 2)
        return best

    def render(self, cam, extras):
        c = mujoco.MjvCamera()
        c.lookat[:], c.distance, c.azimuth, c.elevation = cam["lookat"], cam["dist"], cam["az"], cam["el"]
        self.r.update_scene(self.d, c)
        sc = self.r.scene
        for kind, a, b, rgba in extras:
            g = sc.geoms[sc.ngeom]
            if kind == "sphere":
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([b, 0, 0]), np.asarray(a, float),
                                    np.eye(3).ravel(), np.asarray(rgba, np.float32))
            else:
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                    np.asarray(rgba, np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, b, np.asarray(a[0], float), np.asarray(a[1], float))
            sc.ngeom += 1
        return self.r.render().copy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rp = Replay()
    index = {}
    for old in OUT.glob("*.png"):
        old.unlink(); old.with_suffix(".json").unlink(missing_ok=True)
    for name, (run, t, t0, col, cam) in SCENES.items():
        rows = rp.load(run)
        t = min(t, len(rows) - 1)
        path = []
        for k in range(t0, t + 1, 2):
            rp.set(rows[k])
            path.append(rp.d.site_xpos[rp.site].copy())
        rp.set(rows[t])
        ex = [("line", (path[i], path[i + 1]), 0.0045, col) for i in range(len(path) - 1)]
        ex.append(("sphere", path[-1], 0.009, col))
        dist, pt = rp.closest()
        if col == RED and pt is not None:
            ex += [("sphere", pt, 0.030, (1.0, 0.1, 0.05, 0.28)), ("sphere", pt, 0.016, (1.0, 0.1, 0.05, 0.55))]
        img = rp.render(cam, ex)
        p = OUT / f"{name}.png"
        Image.fromarray(img).save(p)
        src = run / "frames.jsonl"
        meta = {"asset": str(p.relative_to(REPO)), "run": str(run), "profile": PROFILE, "config": str(CFG),
                "control_step": t,
                "path_from_step": t0, "gripper_hurdle_distance_m": float(dist), "sources": {str(src): sha(src)},
                "note": "MuJoCo replay of the recorded qpos with the run's hurdle; left tool-centre path by FK.",
                "script": "paper/tools/plotting/paperbanana_fig1_scenes.py"}
        p.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
        index[name] = meta
        print("wrote", p.relative_to(REPO), f"d={dist * 1000:.1f} mm")
    (OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
