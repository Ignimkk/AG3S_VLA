"""robot mask (self-filter) 계산 시간 — BEFORE(218 구, 여유 50 mm) vs NOW(587 구, 여유 0).

compute.py 와 같은 frame(T14 closed-loop ep1807, seq 13, t=96)·같은 두 모델·같은 점군(세 camera 의
유효 depth pixel 전부, 간격 없음)에서 `filter_robot_points` 만 잰다. FK(구 중심) 포함.
각 조건 warm-up 2 회 뒤 N 회 반복, 중앙값. 결과: timing.json
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, time, pathlib, statistics
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.types import PointCloud
from AG3S.ag3s.stages.robot_filter import filter_robot_points
from AG3S.ag3s.experiments.sources.mujoco_source import (
    TransportScene, set_finger_joints_from_scene, gap_filling_capsules, HEAD_JOINTS)
from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model
from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

OUTD = pathlib.Path(_ag3s_workspace_path('outputs/verify/rm-model-compare'))
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
REC = _ag3s_workspace_path('outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl')
SEQ, N = 13, 15
CAMS = ("wrist_cam_l", "zed_left", "wrist_cam_r")

m = mujoco.MjModel.from_xml_path(XML)
scene0 = TransportScene.attach(m, mujoco.MjData(m))
head = {n: float(scene0.data.qpos[scene0._qadr[n]]) for n in HEAD_JOINTS if n in scene0._qadr}
before = UrdfSphereChain(parse_urdf(RBY1_URDF), extra_capsules=list(gap_filling_capsules(m)),
                         fixed_joint_values=head)
now = build_robot_model(scene0)

d = mujoco.MjData(m); scene = TransportScene.attach(m, d)
fr = [json.loads(l) for l in open(REC) if '"planning"' in l]
fr = [r for r in fr if r["seq"] == SEQ][0]
d.qpos[:] = np.asarray(fr["qpos"], float); d.qvel[:] = 0; mujoco.mj_forward(m, d)
q = scene.robot_state()
set_finger_joints_from_scene(scene, now)

clouds = {}
for cam in CAMS:
    f = scene.capture(cam)
    valid = np.isfinite(f.depth) & (f.depth > 0) & (f.body_ids >= 0)
    vs, us = np.nonzero(valid); K = f.camera_intrinsics; zz = f.depth[vs, us]
    pc = np.stack([(us - K[0, 2]) * zz / K[0, 0], (vs - K[1, 2]) * zz / K[1, 1], zz], 1)
    pb = (f.T_base_cam[:3, :3] @ pc.T).T + f.T_base_cam[:3, 3]
    clouds[cam] = PointCloud(points=pb, uv=np.stack([us, vs], 1).astype(np.int32))

MODELS = {"before": (before, 0.05), "now": (now, 0.0)}
out = {"record": REC, "seq": SEQ, "t_step": fr["t_step"], "repeats": N, "cameras": {}, "total_ms": {}}
for cam, cloud in clouds.items():
    out["cameras"][cam] = {"n_points": int(len(cloud.points))}
for key, (rm, infl) in MODELS.items():
    cfg = PointCloudConfig(self_filter_inflation=infl)
    per_cam = {}
    for cam, cloud in clouds.items():
        for _ in range(2):
            filter_robot_points(cloud, rm, q, cfg)
        ts = []
        for _ in range(N):
            t0 = time.perf_counter(); filter_robot_points(cloud, rm, q, cfg); ts.append((time.perf_counter() - t0) * 1e3)
        per_cam[cam] = {"median_ms": statistics.median(ts), "min_ms": min(ts), "max_ms": max(ts)}
        out["cameras"][cam][key] = per_cam[cam]
    out["total_ms"][key] = sum(v["median_ms"] for v in per_cam.values())
    out[f"{key}_model"] = {"n_spheres": int(rm.n_spheres), "inflation_m": infl}
out["delta_total_ms"] = out["total_ms"]["now"] - out["total_ms"]["before"]
(OUTD / "timing.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=1))
