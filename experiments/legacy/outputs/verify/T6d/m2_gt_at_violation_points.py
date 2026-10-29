"""T6d 재는 것 2 — cuRobo 단계. 팔뚝이 피하는 자리에 참값으로 무엇이 있는가.

각 planning frame 의 `max_violation_pair.point_m` 에서 **참값 최근접 표면까지의 거리**를 잰다.
방법: 기록의 `qpos`(66, 물체 free joint 포함) 로 MuJoCo 씬을 놓고, `MjSpec` 으로 넣은
**mocap probe sphere**(r = 0.5 mm, contype/conaffinity = 0) 를 그 좌표에 옮긴 뒤
`mj_geomDistance(probe, geom, distmax)` 로 표면거리를 잰다. probe 반지름은 되돌려 더한다.

`T5f` 함정: `mj_geomDistance` 는 mesh 쌍에서 질의가 실패하면 **정확히 0.0** 을 돌려준다.
그래서 `distmax` 를 0.02 · 0.05 · 0.10 셋으로 재고, 0.0 을 포함/제외 둘 다 낸다 (0.1 m 초과는 쓰지 않는다).

frame 확인도 함께: `point_m` 은 base frame, `object_poses` 는 MuJoCo world frame 이라
`body "base"` 의 world pose 를 찍어 둘이 같은 frame 인지 기록한다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))

MODEL_XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
RUNS = {
    "closed_loop_approach": _ag3s_workspace_path('outputs/live_test/local_t6_from_pc/frames.jsonl'),
    "shadow_approach": _ag3s_workspace_path('outputs/live_test/local_shadow_from_pc/frames.jsonl'),
    "closed_loop_grasp": _ag3s_workspace_path('outputs/live_test/local_grasp_from_pc/frames.jsonl'),
}
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d/m2_gt_at_violation_points.json'))
SCENE_BODIES = ("apple", "banana", "orange", "pear", "crate", "table")
DISTMAXES = (0.02, 0.05, 0.10)
PROBE_R = 0.0005


def build_probe_model():
    import mujoco

    spec = mujoco.MjSpec.from_file(MODEL_XML)
    body = spec.worldbody.add_body(name="t6d_probe", mocap=True, pos=[0.0, 0.0, 5.0])
    g = body.add_geom(name="t6d_probe_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[PROBE_R, 0.0, 0.0], contype=0, conaffinity=0, mass=0.0)
    g.group = 4
    m = spec.compile()
    return m, mujoco.MjData(m)


def main() -> None:
    import mujoco

    m, d = build_probe_model()
    probe_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "t6d_probe_geom")
    probe_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "t6d_probe")
    mocap_id = int(m.body_mocapid[probe_b])
    base_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")

    def collidable(g: int) -> bool:
        return int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0

    # 씬 geom: 물체·상자·책상. 로봇 geom: 그 밖의 충돌 geom 전부 (probe 제외).
    scene_geoms: list[tuple[int, str]] = []
    body_of: dict[int, str] = {}
    for name in SCENE_BODIES:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0:
            raise SystemExit(f"body not found: {name}")
        body_of[b] = name
    scene_bodies = set(body_of)
    robot_geoms: list[tuple[int, str]] = []
    for g in range(m.ngeom):
        if g == probe_g or not collidable(g):
            continue
        b = int(m.geom_bodyid[g])
        if b in scene_bodies:
            scene_geoms.append((g, body_of[b]))
        elif b != probe_b:
            robot_geoms.append((g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"))

    print(f"scene geoms {len(scene_geoms)} · robot geoms {len(robot_geoms)} · nq {m.nq}")

    out: dict = {
        "model_xml": MODEL_XML,
        "probe_radius_m": PROBE_R,
        "distmaxes_m": list(DISTMAXES),
        "n_scene_geoms": len(scene_geoms),
        "n_robot_geoms": len(robot_geoms),
        "runs": {},
    }

    for run, path in RUNS.items():
        rows = []
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("kind") != "planning":
                    continue
                pair = rec.get("max_violation_pair") or {}
                pt = pair.get("point_m")
                if pt is None:
                    continue
                qpos = np.asarray(rec["qpos"], np.float64)
                d.qpos[: qpos.size] = qpos
                d.qvel[:] = 0.0
                p = np.asarray(pt, np.float64)
                d.mocap_pos[mocap_id] = p
                d.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
                mujoco.mj_forward(m, d)

                base_pos = np.array(d.xpos[base_b], np.float64)
                base_quat = np.array(d.xquat[base_b], np.float64)

                row = {
                    "seq": int(rec["seq"]),
                    "t_step": int(rec["t_step"]),
                    "link": pair.get("link"),
                    "query": pair.get("query"),
                    "clearance_mm": float(pair["clearance_m"]) * 1000.0,
                    "point_m": [float(v) for v in p],
                    "hold_reason": rec.get("hold_reason", ""),
                    "trajopt_status": rec["verdict"]["trajopt_status"],
                    "base_world_pos": [float(v) for v in base_pos],
                    "base_world_quat": [float(v) for v in base_quat],
                    "by_distmax": {},
                }
                # 사과 중심까지의 직선거리 (참값 pose 에서, 표면 아님)
                ap = rec["object_poses"]["apple"]["pos"]
                row["dist_to_apple_centre_mm"] = float(
                    np.linalg.norm(p - np.asarray(ap, np.float64))) * 1000.0

                for dm in DISTMAXES:
                    def scan(pairs):
                        inc = (float("inf"), None)
                        exc = (float("inf"), None)
                        zeros = 0
                        per: dict[str, float] = {}
                        for g, label in pairs:
                            dd = mujoco.mj_geomDistance(m, d, probe_g, g, dm, None) + PROBE_R
                            raw = dd - PROBE_R
                            if raw == 0.0:
                                zeros += 1
                            if dd < inc[0]:
                                inc = (dd, label)
                            if raw != 0.0:
                                if dd < exc[0]:
                                    exc = (dd, label)
                                if label not in per or dd < per[label]:
                                    per[label] = dd
                        return {
                            "min_mm_including_zeros": inc[0] * 1000.0,
                            "min_label_including_zeros": inc[1],
                            "min_mm_dropping_zeros": exc[0] * 1000.0,
                            "min_label_dropping_zeros": exc[1],
                            "n_pairs_exactly_zero": zeros,
                            "saturated": bool(exc[0] >= dm - 1e-12),
                            "per_label_mm": {k: round(v * 1000.0, 4) for k, v in sorted(per.items())},
                        }

                    row["by_distmax"][f"{dm:.2f}"] = {
                        "scene": scan(scene_geoms),
                        "robot": scan(robot_geoms),
                    }
                rows.append(row)
        out["runs"][run] = rows
        holds = [r for r in rows if r["hold_reason"]]
        print(f"#### {run}: {len(rows)} frames, {len(holds)} hold")
        for r in rows:
            s = r["by_distmax"]["0.10"]["scene"]
            rb = r["by_distmax"]["0.10"]["robot"]
            if r["hold_reason"] or r["seq"] in (1, 75):
                print(f"  seq {r['seq']:3d} t={r['t_step']:4d} {r['link']:20s} "
                      f"clear={r['clearance_mm']:8.3f} mm  gt_scene={s['min_mm_dropping_zeros']:8.3f} "
                      f"({s['min_label_dropping_zeros']})  gt_robot={rb['min_mm_dropping_zeros']:8.3f} "
                      f"({rb['min_label_dropping_zeros']})  apple_centre={r['dist_to_apple_centre_mm']:7.2f}")

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
