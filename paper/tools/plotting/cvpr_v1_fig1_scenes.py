"""Hurdle scenes for CVPR v1 Figure 1: VLA only (E0, T43 W) vs VLA + AG3S (SURF arm, IFG-HURDLE), same episode and seed.

Episode 1968 ("Put the apple in the basket."), seed 19681, obstacle profile t43p2_overbar (one red bar across the
crate). Both runs are MuJoCo replays of the recorded 66-D qpos in ``frames.jsonl`` (``"kind": "control"`` rows);
nothing is re-simulated. Overlays: the recorded left tool-centre path (FK of the recorded qpos; red = VLA only,
gold = VLA + AG3S) and, for the VLA-only failure frame, the closest gripper/bar point (mj_geomDistance).

Outputs (new files only): paper/figures/cvpr_v1/assets/<name>.png + .json

    MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_fig1_scenes.py            # final 4 frames
    MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/cvpr_v1_fig1_scenes.py explore OUT  # contact sheets
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

STORAGE = pathlib.Path("/mnt/dev/AG3S_VLA-storage")
T43 = STORAGE / "results/workspace/verify/T43"
IFG = STORAGE / "results/development/20261017T171201+0900_f15e587290f6_137b5f75d9e3"
XML = IFG / "snapshot/robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml"
OUT = REPO / "paper/figures/cvpr_v1/assets"

EP, SEED = 1968, 19681
CFG = T43 / f"P2/configs/t43p2_ep{EP}_s{SEED}.json"
PROFILE = "t43p2_overbar"
RUNS = {"E0": T43 / f"W/runs/E0/H/E0_ep{EP}_s{SEED}", "SURF": IFG / f"verify/IFG-HURDLE/runs/SURF/SURF_ep{EP}_s{SEED}"}
RED = (0.85, 0.12, 0.10, 1.0)
GOLD = (0.95, 0.70, 0.02, 1.0)
GRIPPER = ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2")

# frame choices (control-step index) and cameras; see NOTES.md for how they were chosen
SCENES = {
    # name: (arm, step t, path start t0, colour, camera, note)
    "vla_pick": ("E0", 200, 120, RED, "pick",
                 "VLA only, grasp moment (grasp_t = 200 in T43W.verify.json numbers.per_run); the apple is lifted beside the bar"),
    "vla_carry_fail": ("E0", 234, 154, RED, "carry",
                       "VLA only, last recorded control step (n_control = 235): finger/bar contact -> client obstacle stop; "
                       "red marker = closest gripper/bar point (mj_geomDistance)"),
    "ag3s_pick": ("SURF", 156, 76, GOLD, "pick",
                  "VLA + AG3S (SURF), apple just lifted (attach_t = 144, grasp_t = 160 in IFG-HURDLE agg.json)"),
    "ag3s_carry": ("SURF", 196, 116, GOLD, "carry",
                   "VLA + AG3S (SURF), tool centre directly above the bar line (y = 0.104 m vs bar y = 0.1058 m); "
                   "between grasp_t 160 and place_t 256"),
}
CAMS = {   # same camera for both rows of a column (paired view)
    "pick": {"lookat": [0.56, 0.24, 0.94], "dist": 0.72, "az": 150, "el": -18},
    "carry": {"lookat": [0.55, 0.15, 1.02], "dist": 0.85, "az": 150, "el": -15},
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
        self.apple = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "apple")
        self.mgr = None

    def hurdle_geoms(self):
        out = []
        for g in range(self.m.ngeom):
            n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "obstacle" in n and "collision" in n:
                out.append(g)
        return out

    def load(self, run_dir):
        rows = [json.loads(l)["qpos"] for l in open(run_dir / "frames.jsonl") if '"kind": "control"' in l]
        if self.mgr is not None:
            self.mgr.park_all()
        self.mgr = PickPlaceObstacleManager(self.m, self.d, load_pick_place_obstacle_config(str(CFG)))
        return rows

    def set(self, q):
        self.d.qpos[:] = q
        self.mgr.activate(PROFILE)
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


def frame(rp, rows, t, t0, col, cam, mark_closest):
    t = min(t, len(rows) - 1)
    path = []
    for k in range(t0, t + 1, 2):
        rp.set(rows[k])
        path.append(rp.d.site_xpos[rp.site].copy())
    rp.set(rows[t])
    ex = [("line", (path[i], path[i + 1]), 0.0045, col) for i in range(len(path) - 1)]
    ex.append(("sphere", path[-1], 0.009, col))
    dist, pt = rp.closest()
    if mark_closest and pt is not None:
        ex += [("sphere", pt, 0.030, (1.0, 0.1, 0.05, 0.28)), ("sphere", pt, 0.016, (1.0, 0.1, 0.05, 0.55))]
    img = rp.render(cam, ex)
    return img, float(dist), rp.d.xpos[rp.apple].copy(), rp.d.site_xpos[rp.site].copy()


def explore(outdir):
    """Contact sheets: candidate steps for each run and camera, to choose SCENES by eye."""
    outdir = pathlib.Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rp = Replay(800, 600)
    cfg = json.load(open(CFG))["profiles"][PROFILE]["obstacles"][0]
    print("bar", cfg["position"], "height", cfg["bar_height"], "len", cfg["length"], "yaw", cfg["yaw"])
    for arm, run in RUNS.items():
        rows = rp.load(run)
        print(arm, "n_control", len(rows))
        # per-step geometry table
        tab = []
        for t in range(0, len(rows), 4):
            rp.set(rows[t])
            d, _ = rp.closest()
            tab.append((t, rp.d.site_xpos[rp.site].copy(), rp.d.xpos[rp.apple].copy(), d))
        for t, ee, ap, d in tab:
            print(f"  t={t:3d} ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}) apple=({ap[0]:.3f},{ap[1]:.3f},{ap[2]:.3f}) grip-bar={d*1000:6.1f} mm")
        steps = {"E0": [150, 180, 196, 200, 210, 220, 228, 232, 234], "SURF": [120, 140, 150, 156, 160, 180, 195, 205, 215, 230, 250]}[arm]
        for cname, cam in CAMS.items():
            tiles = []
            for t in steps:
                img, d, _, _ = frame(rp, rows, t, max(t - 80, 0), RED if arm == "E0" else GOLD, cam, arm == "E0")
                tiles.append(np.asarray(Image.fromarray(img).resize((400, 300))))
            n = len(tiles); cols = 4; rws = (n + cols - 1) // cols
            sheet = np.full((rws * 300, cols * 400, 3), 255, np.uint8)
            for i, tl in enumerate(tiles):
                r_, c_ = divmod(i, cols); sheet[r_ * 300:(r_ + 1) * 300, c_ * 400:(c_ + 1) * 400] = tl
            Image.fromarray(sheet).save(outdir / f"sheet_{arm}_{cname}.png")
            print("wrote", outdir / f"sheet_{arm}_{cname}.png", steps)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rp = Replay()
    index = {}
    cache = {}
    for name, (arm, t, t0, col, camk, note) in SCENES.items():
        if arm not in cache:
            cache[arm] = rp.load(RUNS[arm])
        else:
            rp.load(RUNS[arm])  # re-activate the manager for this run (same config)
        rows = cache[arm]
        img, dist, apple, ee = frame(rp, rows, t, t0, col, CAMS[camk], name == "vla_carry_fail")
        p = OUT / f"{name}.png"
        Image.fromarray(img).save(p)
        src = RUNS[arm] / "frames.jsonl"
        meta = {"asset": str(p.relative_to(REPO)), "arm": arm, "run": str(RUNS[arm]), "episode": EP, "seed": SEED,
                "profile": PROFILE, "config": str(CFG), "model_xml": str(XML), "control_step": t, "n_control_steps": len(rows),
                "path_from_step": t0, "camera": CAMS[camk], "gripper_bar_distance_m": dist,
                "apple_xyz_m": apple.tolist(), "left_tool_centre_xyz_m": ee.tolist(), "note": note,
                "sources": {str(src): sha(src), str(CFG): sha(CFG), str(XML): sha(XML)},
                "renderer": f"MuJoCo {mujoco.__version__} replay of the recorded qpos (no re-simulation); OSMesa, CPU",
                "script": "paper/tools/plotting/cvpr_v1_fig1_scenes.py"}
        p.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
        index[name] = meta
        print("wrote", p.relative_to(REPO), f"t={t} grip-bar={dist * 1000:.1f} mm")
    (OUT / "fig1_scenes_index.json").write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "explore":
        explore(sys.argv[2])
    else:
        main()
