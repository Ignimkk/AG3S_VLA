"""T43 Y — MuJoCo re-render of a recorded control row with overlay spheres (verifier, CPU, MUJOCO_GL=osmesa).
Scene = snapshot model_transport_pick_place_obstacles.xml + the run's obstacle profile placed by the client's PickPlaceObstacleManager
(hurdle_metrics.manager), qpos = the control row's recorded qpos (pose before that row's physics). Video camera = pi05_infer
configure_view_camera('front') (lookat (0.45, 0, 0.85), distance 1.7, azimuth 180, elevation -18, 640 x 480), the view of third_person_front.mp4."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import hurdle_metrics as HM  # noqa: E402

REC = dict(lookat=(0.45, 0.0, 0.85), distance=1.7, azimuth=180.0, elevation=-18.0)   # pi05_infer.configure_view_camera('front') = the mp4 view


def scene_for(run_dir, side):
    rd = pathlib.Path(run_dir)
    st = (rd.parent / f"{rd.name}.start").read_text().splitlines()
    ob = next((l.split("=", 1)[1] for l in st if l.startswith("obstacle=")), ":")
    HM.set_arm(side); HM._M.clear(); M = HM.model()
    if ob and ob != ":":
        c, p = ob.split(":", 1); HM.manager(c, p)
    rows = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
    ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}
    return M, ctrl


def render(M, qpos, cam, spheres=(), size=(480, 640)):
    m, d = M["m"], M["d"]
    d.qpos[:] = qpos; mujoco.mj_forward(m, d)
    r = render.cache.get(size)
    if r is None:
        r = mujoco.Renderer(m, size[0], size[1]); render.cache[size] = r
    c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = cam["lookat"]; c.distance = cam["distance"]; c.azimuth = cam["azimuth"]; c.elevation = cam["elevation"]
    r.update_scene(d, camera=c)
    scn = r.scene
    for pos, rad, rgba in spheres:
        if scn.ngeom >= scn.maxgeom:
            break
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0], float), np.asarray(pos, float),
                            np.eye(3).reshape(-1), np.asarray(rgba, np.float32))
        scn.ngeom += 1
    return r.render().copy()


render.cache = {}


def project(M, cam, pts, size=(480, 640)):
    """pixel (u, v) of world points for a free camera (MuJoCo convention: fovy = model vis.global.fovy)."""
    m = M["m"]; H, W = size
    az, el = np.radians(cam["azimuth"]), np.radians(cam["elevation"])
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = np.asarray(cam["lookat"]) - cam["distance"] * fwd
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    f = 0.5 * H / np.tan(np.radians(m.vis.global_.fovy) / 2)
    out = []
    for p in pts:
        v = np.asarray(p) - eye; z = v @ fwd
        out.append((W / 2 + f * (v @ right) / z, H / 2 - f * (v @ up) / z))
    return np.array(out)


def video_frame(run_dir, i):
    import cv2
    c = cv2.VideoCapture(str(pathlib.Path(run_dir) / "third_person_front.mp4")); c.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = c.read(); c.release()
    return None if not ok else f[:, :, ::-1].copy()
