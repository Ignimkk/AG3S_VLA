"""ON 에서 남은 위반 구의 최근접 표면이 무엇인가 — 마스크 안 된 depth 점을 MuJoCo segmentation 으로 이름 붙인다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-openpi-live/bin/python outputs/verify/ppt0930-rm-onoff/diag_residual.py
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np  # noqa: E402

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/ppt0930-rm-onoff'))
REC = _ag3s_workspace_path('outputs/live_test/20260925_ep1807/run_0000')


def main():
    import mujoco
    from scipy.spatial import cKDTree

    from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
    from AG3S.ag3s.experiments.sources.mujoco_source import is_robot_body

    raw = np.load(OUT / "rm_onoff_raw.npz")
    run = load_run(REC, limit=1)
    scene = replay_scene(run)
    pose_scene(scene, run.steps[0])
    names = {i: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
             for i in range(scene.model.nbody)}
    pts_all, lab_all, masked_all, cam_all = [], [], [], []
    leak = {}
    for c in ("zed_left", "wrist_cam_l", "wrist_cam_r"):
        fr = scene.capture(c)
        assert np.allclose(fr.depth, raw[f"depth_{c}"], atol=1e-5)
        K, T = raw[f"K_{c}"], raw[f"T_{c}"]
        z = raw[f"depth_{c}"].astype(np.float64)
        h, w = z.shape
        vv, uu = np.mgrid[0:h, 0:w]
        ok = (z > 0.05) & (z < 3.0)
        x = (uu - K[0, 2]) * z / K[0, 0]
        y = (vv - K[1, 2]) * z / K[1, 1]
        P = np.stack([x[ok], y[ok], z[ok]], 1) @ T[:3, :3].T + T[:3, 3]
        ids = np.asarray(fr.body_ids)[ok]
        m = raw[f"mask_{c}"][ok]
        pts_all.append(P); lab_all.append(ids); masked_all.append(m); cam_all.append(np.full(len(P), c))
        robot = np.array([is_robot_body(names.get(int(i), "")) for i in ids])
        leak[c] = {"robot_px": int(robot.sum()), "robot_px_unmasked": int((robot & ~m).sum()),
                   "nonrobot_px_masked": int((~robot & m).sum()),
                   "unmasked_robot_bodies": {names[int(b)]: int(n) for b, n in zip(*np.unique(ids[robot & ~m], return_counts=True))},
                   "masked_nonrobot_bodies": {names[int(b)]: int(n) for b, n in zip(*np.unique(ids[~robot & m], return_counts=True))}}
    P = np.concatenate(pts_all); L = np.concatenate(lab_all); M = np.concatenate(masked_all); C = np.concatenate(cam_all)
    keep = ~M
    tree = cKDTree(P[keep])
    Lk, Ck = L[keep], C[keep]
    out = {"leak_by_camera": leak, "violated_on_nearest": []}
    for model in ("arms",):
        c, r, lk = raw[f"centres_{model}"], raw[f"radii_{model}"], raw[f"links_{model}"]
        clr = raw[f"clr_{model}_on"]
        for i in np.where(clr < 0)[0]:
            dist, j = tree.query(c[i], k=20)
            out["violated_on_nearest"].append({
                "link": str(lk[i]), "centre": np.round(c[i], 4).tolist(), "r_mm": 1000 * float(r[i]),
                "clearance_on_mm": 1000 * float(clr[i]),
                "nearest_unmasked_point_mm": 1000 * float(dist[0]),
                "nearest_bodies": sorted({names[int(Lk[k])] for k in j}),
                "nearest_cameras": sorted({str(Ck[k]) for k in j})})
    (OUT / "diag_residual.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
