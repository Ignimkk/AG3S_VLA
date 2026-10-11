"""T8c s4 — 길이 축. `link_*_arm_5` capsule 을 **짧게 잘랐을 때** 남는 위반과 잃는 보호.

**근사임을 밝힌다.** capsule 길이를 줄이는 flag 가 없으므로, T8a tensor 에서
`link_*_arm_5` 의 구 중 **중심이 z_cut 보다 원위(link frame 에서 더 작은 z)** 인 것을
행에서 빼는 것으로 근사한다. 남는 구의 반지름은 그대로 81.25 mm 다 — 실제로 capsule 을
짧게 만들면 구 개수가 줄어 `_effective_radius` 의 spacing 항이 커지므로 반지름이 달라진다.
그 차이는 여기서 재지 않는다 (`not_measured`).

같이 내는 것:
  * z_cut 별로 **축 위 어느 구간이 어떤 구에도 안 덮이는지** (link frame z, mm)
  * 해부학적 landmark (손목 원점 · FT sensor · 손바닥 `ee_left` · 손가락 뿌리) 가
    **제약 모델의 어떤 구 안에 있는지** — 실제 기록 자세에서 3D 로.
    `ARM_LINKS` 에 `ee_left` 가 없으므로 손바닥을 덮는 것은 지금 `arm_5` 덩어리뿐이다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T6d'))

from m3_to_refined_vs_reference import (  # noqa: E402
    ACT_COL, JOINT_ORDER, MODEL_XML, QADR, SHADOW, load_planning)

T8A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8a'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8c/s4_length_axis.json'))
MARGIN = 10.0
EXEC_H = 8
AUTH = ("ee_finger_l1", "ee_finger_l2", "link_left_arm_5")
ARM5 = ("link_left_arm_5", "link_right_arm_5")
Z_CUTS_M = (None, -0.05, -0.10, -0.1087, -0.15)
LANDMARKS = ("link_left_arm_6", "FT_sensor_L", "ee_left", "ee_finger_l1", "ee_finger_l2")
# 3D coverage 를 재는 기록 자세 (T8a/T8c 가 쓴 임계 행들)
PROBE_ROWS = ((24, 17), (24, 4), (12, 5), (5, 48), (21, 7))


def main() -> None:
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    from AG3S.ag3s.robot_models import RBY1_URDF, parse_urdf

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    scene = TransportScene.attach(m, d)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    names = list(resolve_link_names(robot, robot.n_spheres))
    _, radii = robot.sphere_centers_numeric(np.zeros(robot.nq))
    radii = np.asarray(radii, np.float64).reshape(-1) * 1000.0    # mm

    # --- link_left_arm_5 frame 에서의 축 좌표 ---------------------------------------------
    urdf = parse_urdf(RBY1_URDF)
    cap5 = next(c for c in urdf.capsules if c.link == "link_left_arm_5")
    R5 = np.asarray(cap5.origin[:3, :3], np.float64)
    p5 = np.asarray(cap5.origin[:3, 3], np.float64)
    axis_aligned = bool(np.allclose(R5, np.eye(3), atol=1e-4))
    local5 = [(i, np.asarray(c, np.float64))
              for i, (ln, c, _) in enumerate(robot._local_spheres) if ln == "link_left_arm_5"]
    z_local = {i: float(c[2]) for i, c in local5}
    r5_mm = float(radii[local5[0][0]])
    cap_axis = (float(p5[2] - cap5.length / 2.0), float(p5[2] + cap5.length / 2.0))
    cap_surface = (cap_axis[0] - cap5.radius, cap_axis[1] + cap5.radius)

    # landmark 를 link_left_arm_5 frame 으로 (고정 joint 는 origin 만, arm_6 은 값 0)
    parent_joint = {j.child: j for j in urdf.joints}
    def frame_in_arm5(link: str) -> np.ndarray:
        T = np.eye(4)
        chain = []
        cur = link
        while cur != "link_left_arm_5":
            j = parent_joint[cur]
            chain.append(j)
            cur = j.parent
        for j in reversed(chain):
            T = T @ np.asarray(j.origin, np.float64)
        return T
    # NOTE: link_left_arm_6 아래는 회전 joint 하나(left_arm_6)가 있다. frame_in_arm5 는 joint
    # 값 0 으로 계산하므로 z 축 좌표는 정확하지만(회전축이 z), x/y 는 joint 값에 따라 돈다.
    landmark_z = {}
    for ln in LANDMARKS:
        T = frame_in_arm5(ln)
        landmark_z[ln] = {"pos_in_arm5_frame_m": [round(float(v), 6) for v in T[:3, 3]],
                          "z_mm": round(float(T[2, 3]) * 1000.0, 3)}

    # --- tensor ---------------------------------------------------------------------------
    t = np.load(T8A / "m0_clearance_tensor.npz", allow_pickle=True)
    clear = t["clearance_m"].astype(np.float64) * 1000.0
    bodies = [str(b) for b in t["bodies"]]
    tnames = [str(n) for n in t["link_names"]]
    assert tnames == names
    F, H, S, B = clear.shape
    ai = bodies.index("apple")
    auth = np.asarray([n in AUTH for n in names], bool)
    keep_row = np.ones((S, B), bool)
    keep_row[auth, ai] = False

    recs = load_planning(SHADOW)
    windows = {"full50": slice(0, H), "exec8": slice(0, EXEC_H)}

    results = []
    for zc in Z_CUTS_M:
        if zc is None:
            drop = np.zeros(S, bool)
            kept_idx = [i for i, _ in local5]
        else:
            drop = np.asarray([names[i] in ARM5 and z_local.get(i, 1e9) < zc
                               if names[i] in ARM5 else False for i in range(S)], bool)
            # 오른팔도 같은 local z 를 쓴다 (같은 capsule)
            z_local_all = {}
            for i, (ln, c, _) in enumerate(robot._local_spheres):
                if ln in ARM5:
                    z_local_all[i] = float(c[2])
            drop = np.asarray([(names[i] in ARM5) and (z_local_all[i] < zc)
                               for i in range(S)], bool)
            kept_idx = [i for i, _ in local5 if z_local[i] >= zc]
        sphere_keep = ~drop
        entry = {
            "z_cut_m": zc,
            "z_cut_mm": None if zc is None else round(zc * 1000.0, 3),
            "n_arm5_spheres_kept_per_arm": len(kept_idx),
            "n_arm5_spheres_dropped_per_arm": len(local5) - len(kept_idx),
            "n_spheres_dropped_total": int(drop.sum()),
            "kept_sphere_axis_z_mm": sorted(round(z_local[i] * 1000.0, 3) for i in kept_idx),
            "dropped_sphere_axis_z_mm": sorted(round(z_local[i] * 1000.0, 3)
                                               for i, _ in local5 if i not in kept_idx),
        }
        # 축 위 덮이는 구간 (link frame z, mm) — 남은 arm5 구만
        if kept_idx:
            zmin = min(z_local[i] for i in kept_idx) * 1000.0 - r5_mm
            zmax = max(z_local[i] for i in kept_idx) * 1000.0 + r5_mm
            entry["arm5_covered_axis_z_mm"] = [round(zmin, 3), round(zmax, 3)]
            entry["uncovered_distal_interval_z_mm"] = [
                round(cap_surface[0] * 1000.0, 3), round(zmin, 3)]
            entry["uncovered_distal_length_mm"] = round(zmin - cap_surface[0] * 1000.0, 3)
        else:
            entry["arm5_covered_axis_z_mm"] = None
            entry["uncovered_distal_interval_z_mm"] = [round(cap_surface[0] * 1000.0, 3),
                                                       round(cap_surface[1] * 1000.0, 3)]
            entry["uncovered_distal_length_mm"] = round(cap5.length * 1000.0 + 2 * cap5.radius
                                                        * 1000.0, 3)
        entry["landmarks_inside_uncovered_interval_by_axis_z"] = [
            ln for ln, v in landmark_z.items()
            if entry["uncovered_distal_interval_z_mm"][0] <= v["z_mm"]
            <= entry["uncovered_distal_interval_z_mm"][1]]

        # 3D: landmark 가 제약 모델의 어떤 구 안에 있는가 (기록 자세에서)
        cov = {ln: [] for ln in LANDMARKS}
        for ch, step in PROBE_ROWS:
            rec = recs[ch]
            qpos = np.asarray(rec["qpos"], np.float64)
            chunk = np.asarray(rec["actions_reference"], np.float64)
            d.qpos[: qpos.size] = qpos
            for n_, ci in ACT_COL.items():
                d.qpos[QADR[n_]] = float(chunk[step, ci])
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            q = np.asarray([qpos[QADR[n_]] if n_.startswith("torso_")
                            else float(chunk[step, ACT_COL[n_]]) for n_ in JOINT_ORDER],
                           np.float64)
            cen = np.asarray(robot.sphere_centers_numeric(q)[0], np.float64).reshape(-1, 3) * 1000.0
            a5 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link_left_arm_5")
            P5 = np.asarray(d.xpos[a5], np.float64)
            M5 = np.asarray(d.xmat[a5], np.float64).reshape(3, 3)
            for ln in LANDMARKS:
                # landmark 의 world 위치 = link_left_arm_5 frame + URDF 고정 offset.
                # MJCF 에 같은 body 가 있으면 그것과 대조한다 (ee_left 는 MJCF body 가 없다).
                off = np.asarray(frame_in_arm5(ln)[:3, 3], np.float64)
                p = (P5 + M5 @ off) * 1000.0
                bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                        "FT_SENSOR_L" if ln == "FT_sensor_L" else ln)
                cross_mm = (None if bid < 0 else
                            round(float(np.linalg.norm(
                                np.asarray(d.xpos[bid], np.float64) * 1000.0 - p)), 4))
                dist = np.linalg.norm(cen - p[None, :], axis=1) - radii
                dd = np.where(sphere_keep, dist, np.inf)
                j = int(np.argmin(dd))
                cov[ln].append({
                    "chunk": ch, "step": step,
                    "depth_inside_nearest_sphere_mm": round(float(-dd[j]), 3),
                    "covered": bool(dd[j] <= 0.0),
                    "nearest_sphere_link": names[j], "nearest_sphere_index": j,
                    "covered_by_arm5_only": bool(dd[j] <= 0.0 and names[j] in ARM5),
                    "mjcf_body_cross_check_mm": cross_mm,
                })
        entry["landmark_coverage_3d"] = {
            ln: {"origin_in_arm5_frame_z_mm": landmark_z[ln]["z_mm"],
                 "covered_in_all_probe_rows": all(v and v["covered"] for v in cov[ln]),
                 "covered_in_any_probe_row": any(v and v["covered"] for v in cov[ln]),
                 "n_probe_rows_covered": sum(1 for v in cov[ln] if v and v["covered"]),
                 "n_probe_rows": len(cov[ln]),
                 "rows": cov[ln]} for ln in LANDMARKS}

        # 남는 위반
        entry["windows"] = {}
        for wname, w in windows.items():
            cw = np.where((keep_row & sphere_keep[:, None])[None, None, :, :],
                          clear[:, w], np.inf)
            le0 = (cw - MARGIN) <= 0.0
            pairs = []
            for ln in sorted(set(names)):
                sel = np.asarray([n == ln for n in names], bool) & sphere_keep
                if not sel.any():
                    continue
                for bi, bn in enumerate(bodies):
                    if not keep_row[sel, bi].any():
                        continue
                    v = cw[:, :, sel, bi]
                    n = int(((v - MARGIN) <= 0.0).sum())
                    if n:
                        cm = v.min(axis=(1, 2))
                        pairs.append({"link": ln, "obstacle_body": bn, "n_rows_slack_le0": n,
                                      "n_chunks_slack_le0": int((cm - MARGIN <= 0.0).sum()),
                                      "min_clearance_mm": round(float(v.min()), 4),
                                      "min_slack_mm": round(float(v.min()) - MARGIN, 4)})
            entry["windows"][wname] = {
                "n_rows_slack_le0": int(le0.sum()),
                "n_chunks_slack_le0": int(le0.any(axis=(1, 2, 3)).sum()),
                "n_chunks": F,
                "min_slack_mm": round(float(cw.min()) - MARGIN, 4),
                "feasible_at_10mm": bool(le0.sum() == 0),
                "violating_pairs": sorted(pairs, key=lambda p: p["min_slack_mm"]),
            }
        results.append(entry)
        f50, e8 = entry["windows"]["full50"], entry["windows"]["exec8"]
        print(f"z_cut {str(entry['z_cut_mm']):>8s} mm  kept {entry['n_arm5_spheres_kept_per_arm']}/5 "
              f"| full50 rows {f50['n_rows_slack_le0']:5d} chunks {f50['n_chunks_slack_le0']:3d} "
              f"min {f50['min_slack_mm']:8.2f} | exec8 rows {e8['n_rows_slack_le0']:4d} "
              f"chunks {e8['n_chunks_slack_le0']:3d} min {e8['min_slack_mm']:8.2f} | "
              f"uncovered distal {entry['uncovered_distal_length_mm']:7.1f} mm  "
              f"palm covered {entry['landmark_coverage_3d']['ee_left']['n_probe_rows_covered']}"
              f"/{entry['landmark_coverage_3d']['ee_left']['n_probe_rows']}")

    out = {
        "approximation": "capsule 길이를 줄이는 flag 가 없어, tensor 에서 link_*_arm_5 의 구 중 "
                         "중심 z < z_cut 인 것을 빼는 것으로 근사했다. 남는 구의 반지름은 "
                         "81.25 mm 그대로다 — 실제로 capsule 을 짧게 만들면 구 개수가 줄어 "
                         "_effective_radius 의 spacing 항이 커져 반지름이 달라진다. "
                         "그 차이는 여기서 재지 않았다.",
        "condition": {"apple_rows_removed_for_links": list(AUTH), "margin_mm": MARGIN,
                      "capsule_radius_scale": 1.0, "n_chunks": F, "horizon": H},
        "capsule_link_left_arm_5": {
            "radius_mm": round(float(cap5.radius) * 1000.0, 3),
            "length_mm": round(float(cap5.length) * 1000.0, 3),
            "origin_pos_m": [round(float(v), 6) for v in p5],
            "origin_rotation_is_identity": axis_aligned,
            "axis_z_range_mm": [round(cap_axis[0] * 1000.0, 3), round(cap_axis[1] * 1000.0, 3)],
            "surface_z_range_mm": [round(cap_surface[0] * 1000.0, 3),
                                   round(cap_surface[1] * 1000.0, 3)],
            "n_spheres": len(local5),
            "sphere_radius_mm": round(r5_mm, 4),
            "sphere_axis_z_mm": sorted(round(v * 1000.0, 3) for v in z_local.values()),
            "sphere_spacing_mm": round((max(z_local.values()) - min(z_local.values()))
                                       * 1000.0 / (len(local5) - 1), 3),
        },
        "other_arm_capsules_mm": {
            c.link: {"radius_mm": round(float(c.radius) * 1000.0, 3),
                     "length_mm": round(float(c.length) * 1000.0, 3)}
            for c in urdf.capsules if "arm" in c.link},
        "landmark_z_in_arm5_frame": landmark_z,
        "constraint_model_link_set": {
            "ARM_LINKS": sorted(set(names)),
            "ee_left_in_constraint_model": "ee_left" in set(names),
            "ee_right_in_constraint_model": "ee_right" in set(names),
            "note": "ARM_LINKS = link_{side}_arm_{0..6} + ee_finger_{l,r}{1,2}. "
                    "ee_left / ee_right (손바닥) 는 제약 모델에 없다.",
        },
        "probe_rows": [{"chunk": c, "step": s} for c, s in PROBE_ROWS],
        "z_cut_results": results,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("\nlandmark z in link_left_arm_5 frame [mm]:",
          json.dumps({k: v["z_mm"] for k, v in landmark_z.items()}))
    print("capsule surface z range [mm]:", out["capsule_link_left_arm_5"]["surface_z_range_mm"])
    print("sphere axis z [mm]:", out["capsule_link_left_arm_5"]["sphere_axis_z_mm"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
