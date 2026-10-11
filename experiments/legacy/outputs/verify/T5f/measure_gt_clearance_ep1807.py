"""T5f — MuJoCo 참값 clearance 만 따로, distmax 셋으로 교차 확인한다.

`measure_target_identity_ep1807.py` 안의 같은 계산이 3 개 frame(seq 2·18·75)에서
`distmax 0.10` 과 `0.05` 가 어긋났다. 어긋난 값은 셋 다 **정확히 0.0** 이었다 — mesh geom
쌍에서 거리 질의가 실패하면 `mj_geomDistance` 가 0.0 을 돌려준다 (실제 거리가 0.37~0.56 m 인
쌍에서도 그랬다. distmax 0.3 m 부터 나타나고 0.1 m 이하에서 거의 사라진다).

그래서 여기서는

  · distmax **0.02 · 0.05 · 0.10** 세 값으로 각각 재고,
  · 각 값에서 **정확히 0.0 을 포함한 최소값** 과 **0.0 을 실패로 보고 버린 최소값** 을 둘 다 내고,
  · 실패 개수를 프레임마다 센다.

카메라 캡처·AG3S 를 돌리지 않으므로 빠르다. `benchmark/**` 를 고치지 않는다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5f/measure_gt_clearance_ep1807.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys
import time

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np  # noqa: E402

RECORDS = "outputs/live_test/20260925_ep1807/run_0000"
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5f/gt_clearance_ep1807.json'))
OBJECT_BODIES = ("apple", "banana", "orange", "pear", "crate", "table")
DISTMAXES = (0.02, 0.05, 0.10)


def main() -> None:
    import mujoco

    from AG3S.ag3s.experiments.reports.grounding_report import ARM_LINKS
    from AG3S.ag3s.experiments.sources.policy_record import (
        load_run, pose_scene, replay_scene)

    t0 = time.time()
    run = load_run(RECORDS)
    scene = replay_scene(run)
    m, d = scene.model, scene.data

    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in OBJECT_BODIES}
    arm_body = {}
    for ln in ARM_LINKS:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ln)
        if b < 0:
            raise SystemExit(f"arm link body not found: {ln}")
        arm_body[b] = ln

    def collidable(g: int) -> bool:
        return int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0

    arm_geoms = [(g, arm_body[int(m.geom_bodyid[g])]) for g in range(m.ngeom)
                 if int(m.geom_bodyid[g]) in arm_body and collidable(g)]
    obj_geoms = [(g, n) for n in OBJECT_BODIES for g in range(m.ngeom)
                 if int(m.geom_bodyid[g]) == bid[n] and collidable(g)]

    frames = []
    for i, step in enumerate(run):
        pose_scene(scene, step)
        row = {"seq": i + 1, "t_step": int(step.t_step), "by_distmax": {}}
        for dm in DISTMAXES:
            inc = (float("inf"), None)
            exc = (float("inf"), None)
            zeros = 0
            per_obj_exc = {}
            for og, on in obj_geoms:
                for ag, ln in arm_geoms:
                    dd = mujoco.mj_geomDistance(m, d, ag, og, dm, None)
                    if dd == 0.0:
                        zeros += 1
                    if dd < inc[0]:
                        inc = (dd, (ln, on))
                    if dd != 0.0:
                        if dd < exc[0]:
                            exc = (dd, (ln, on))
                        cur = per_obj_exc.get(on)
                        if cur is None or dd < cur[0]:
                            per_obj_exc[on] = (dd, ln)
            row["by_distmax"][f"{dm:.2f}"] = {
                "min_mm_including_zeros": inc[0] * 1000.0,
                "min_pair_including_zeros": list(inc[1]) if inc[1] else None,
                "min_mm_dropping_failed_zeros": exc[0] * 1000.0,
                "min_pair_dropping_failed_zeros": list(exc[1]) if exc[1] else None,
                "n_pairs_exactly_zero": zeros,
                "saturated": bool(exc[0] >= dm - 1e-12),
                "per_object_mm_dropping_zeros": {k: round(v[0] * 1000.0, 4)
                                                 for k, v in per_obj_exc.items()},
                "per_object_link": {k: v[1] for k, v in per_obj_exc.items()},
            }
        a = row["by_distmax"]["0.02"]["min_mm_dropping_failed_zeros"]
        b = row["by_distmax"]["0.05"]["min_mm_dropping_failed_zeros"]
        c = row["by_distmax"]["0.10"]["min_mm_dropping_failed_zeros"]
        # 0.02 는 20 mm 에서 saturate 되므로, 셋이 "같다" 는 뜻은 saturate 를 감안한 일치다.
        row["agree_0p05_vs_0p10"] = bool(abs(b - c) < 1e-6
                                         or (b >= 50.0 - 1e-6 and c >= 100.0 - 1e-6))
        row["agree_0p02_vs_0p05"] = bool(abs(a - b) < 1e-6 or a >= 20.0 - 1e-6)
        frames.append(row)
        if i % 10 == 0 or i == len(run) - 1:
            print(f"  [seq {i+1:3d}] t={row['t_step']:4d} "
                  f"0.02={a:8.4f} 0.05={b:8.4f} 0.10={c:8.4f} mm  "
                  f"pair={row['by_distmax']['0.10']['min_pair_dropping_failed_zeros']} "
                  f"zeros(0.10)={row['by_distmax']['0.10']['n_pairs_exactly_zero']}", flush=True)
    scene.close()

    out = {
        "records": RECORDS,
        "n_frames": len(frames),
        "n_arm_geoms": len(arm_geoms),
        "n_object_geoms": len(obj_geoms),
        "arm_links": list(ARM_LINKS),
        "object_bodies": list(OBJECT_BODIES),
        "distmaxes_m": list(DISTMAXES),
        "rule": ("mj_geomDistance(arm link 충돌 geom, 물체 충돌 geom, distmax) 최소 부호거리. "
                 "그 프레임의 관측 qpos 한 자세에서다 — 궤적 전체가 아니다. 정확히 0.0 은 mesh "
                 "거리 질의 실패로 보고 따로 센다"),
        "n_frames_disagree_0p05_vs_0p10": sum(1 for f in frames if not f["agree_0p05_vs_0p10"]),
        "seqs_disagree_0p05_vs_0p10": [f["seq"] for f in frames if not f["agree_0p05_vs_0p10"]],
        "n_frames_disagree_0p02_vs_0p05": sum(1 for f in frames if not f["agree_0p02_vs_0p05"]),
        "seqs_disagree_0p02_vs_0p05": [f["seq"] for f in frames if not f["agree_0p02_vs_0p05"]],
        "wall_s": time.time() - t0,
        "frames": frames,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\ndisagree 0.05 vs 0.10: {out['seqs_disagree_0p05_vs_0p10']}")
    print(f"disagree 0.02 vs 0.05: {out['seqs_disagree_0p02_vs_0p05']}")
    print(f"wrote {OUT}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
