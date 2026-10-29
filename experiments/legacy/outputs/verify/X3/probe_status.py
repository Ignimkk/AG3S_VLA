"""X3 측정 2·3·4 — `status` 가 무엇을 보고 내려지나.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/X3/probe_status.py --records <run dir> --attention <npz> \
        --out outputs/verify/X3/probe_headfix.json

**esdf_rollout 을 베끼지 않는다.** 같은 진입점을 그대로 부르고 `TrajectoryOptimizer._finish`
만 감싼다. 베끼면 한쪽만 고쳤을 때 두 수치가 다른 코드 위에서 나온다.

감싼 자리에서 프레임마다 꺼내는 것:
  - `geometry_certified`   — AG3S 가 그 프레임 씬을 빠짐없이 설명했다고 보증하는가
  - `status` 와 `notes`    — sqp.py:287-293 이 붙이는 인증 실패 문장이 있는가
  - `full_violation` 두 벌 — `states` 를 주고 잰 것과 안 주고 잰 것 (측정 4)

benchmark/** 는 건드리지 않는다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse
import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np

CALLS: list[dict] = []
SCENES: list[dict] = []
TRAJ: dict[str, np.ndarray] = {}
CERT_PHRASE = "could not certify the geometry"


def install_max_points(value: int) -> None:
    """`pointcloud.max_points` 만 갈아 끼운다. esdf_rollout 에는 이 인자가 없다.

    `AG3SConfig.from_dict` 를 감싸서 rollout 이 만든 dict 에 한 칸을 더 넣는다.
    다른 설정은 손대지 않는다 — 바뀌는 것이 상한 하나여야 차이를 그것에 돌릴 수 있다.
    """
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.trajopt.experiments import esdf_rollout as _er

    original_from_dict = AG3SConfig.from_dict

    def patched_from_dict(d):
        d = dict(d)
        pc = dict(d.get("pointcloud") or {})
        pc["max_points"] = int(value)
        d["pointcloud"] = pc
        return original_from_dict(d)

    AG3SConfig.from_dict = staticmethod(patched_from_dict)
    _er.AG3SConfig = AG3SConfig


def install(out_path: pathlib.Path):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    original = TrajectoryOptimizer._finish

    def patched(self, trajectory, reference, q_now, scene, template, iterations, slack_norm,
                reference_violation, previous_chunk, notes, started, geometry_certified,
                states=None, timing=None, qp_iterations=0):
        frame = int(self.frame_index)
        notes_in = list(notes)
        with_states = float(self.linearizer.full_violation(trajectory, q_now, scene, states))
        without_states = float(self.linearizer.full_violation(trajectory, q_now, scene))

        result = original(self, trajectory, reference, q_now, scene, template, iterations,
                          slack_norm, reference_violation, previous_chunk, notes, started,
                          geometry_certified, states=states, timing=timing,
                          qp_iterations=qp_iterations)

        # **layout.full_q 로 펼쳐서 저장한다.** 궤적은 자유 관절 14 개뿐이고 (양팔 7+7),
        # 토르소 6 개는 들어 있지 않다. 14 행을 20 개 관절 이름에 그대로 갖다 붙이면
        # 팔 각도가 토르소로 들어가 로봇이 뒤틀린다 — 실제로 한 번 그렇게 영상을 냈다.
        TRAJ[f"traj_{frame:02d}"] = np.asarray(
            self.layout.full_q(np.asarray(result.trajectory, float), q_now), float)
        TRAJ[f"ref_{frame:02d}"] = np.asarray(
            self.layout.full_q(np.asarray(reference, float)[:, :np.asarray(result.trajectory).shape[1]],
                               q_now), float)
        TRAJ[f"qnow_{frame:02d}"] = np.asarray(q_now, float)

        notes_out = [str(n) for n in result.notes]
        CALLS.append({
            "frame_index": frame,
            "status": str(result.status.value),
            "safe": bool(result.safe),
            "iterations": int(result.iterations),
            "geometry_certified": bool(geometry_certified),
            "states_passed": states is not None,
            # 측정 4 — 같은 trajectory 를 두 인자로 재면 다른 값이 나오나.
            "clearance_with_states_mm": with_states * 1000.0,
            "clearance_without_states_mm": without_states * 1000.0,
            "clearance_delta_mm": (with_states - without_states) * 1000.0,
            "max_violation_mm": float(result.max_violation) * 1000.0,
            "violation_tolerance_mm": float(self.config.safety.violation_tolerance) * 1000.0,
            "require_certified_geometry": bool(self.config.safety.require_certified_geometry),
            "notes_in": [str(n) for n in notes_in],
            "notes_out": notes_out,
            "has_certification_note": any(CERT_PHRASE in n for n in notes_out),
        })
        return result

    TrajectoryOptimizer._finish = patched

    # `validity` 는 `_finish` 까지 bool 하나로만 내려간다 (`geometry_certified`).
    # 어느 값이라서 False 인지 — valid / degraded / incomplete — 는 `cs` 에만 있다.
    from AG3S.trajopt.experiments import esdf_rollout as _er

    original_scene = _er.scene_from_constraint_set

    def patched_scene(cs, radii, cfg):
        SCENES.append({
            "validity": str(getattr(cs.validity, "value", cs.validity)),
            "geometry_certified": bool(cs.geometry_certified),
            "has_target": bool(cs.has_target),
            "n_candidates": len(cs.candidates),
            "esdf_backend": str(cs.esdf.stats.get("backend", "legacy")),
            "esdf_unknown_fraction": (None if cs.esdf.unknown_fraction is None
                                      else float(cs.esdf.unknown_fraction)),
            "notes": [str(n) for n in (getattr(cs, "notes", None) or [])],
        })
        return original_scene(cs, radii, cfg)

    _er.scene_from_constraint_set = patched_scene
    return original


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--attention", required=True)
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-traj", default=None,
                    help="SQP 가 낸 궤적과 정책 원본 청크를 이 npz 로 저장한다 (영상용)")
    ap.add_argument("--max-points", type=int, default=None,
                    help="pointcloud.max_points 를 이 값으로. 기본은 config 기본값 60000")
    ap.add_argument("--rollout-json", default=None,
                    help="esdf_rollout 이 쓸 out-json. 기본은 --out 옆의 _rollout.json")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rollout_json = pathlib.Path(args.rollout_json or str(out).replace(".json", "_rollout.json"))

    if args.max_points is not None:
        install_max_points(args.max_points)
    install(out)

    from AG3S.trajopt.experiments import esdf_rollout

    sys.argv = [
        "esdf_rollout",
        "--records", args.records,
        "--attention", args.attention,
        "--frames", str(args.frames),
        "--voxel", "0.020",
        "--esdf-margin", "0.05",
        "--cameras", "all",
        "--support-surfaces", "field",
        "--constraint-links", "arms",
        "--out-json", str(rollout_json),
        "--out-doc", str(out.parent / "probe_scratch.md"),
    ]
    esdf_rollout.main()

    rows = json.loads(rollout_json.read_text())["frames"]
    # `_finish` 는 한 solve 안에서 두 번 불릴 수 있다 (조기 종료 경로). 마지막 것이 반환된다.
    last: dict[int, dict] = {}
    for c in CALLS:
        last[c["frame_index"]] = c
    merged = []
    for i, row in enumerate(rows):
        c = last.get(i, {})
        sc = SCENES[i] if i < len(SCENES) else {}
        merged.append({
            "validity": sc.get("validity"),
            "ag3s_notes": sc.get("notes", []),
            "i": row["i"], "t_step": row["t_step"], "phase": row["phase"],
            "status": row["status"],
            "target": row["target"],
            "clearance_after_mm_rollout": row["clearance_after_mm"],
            "clearance_with_states_mm": c.get("clearance_with_states_mm"),
            "clearance_without_states_mm": c.get("clearance_without_states_mm"),
            "clearance_delta_mm": c.get("clearance_delta_mm"),
            "geometry_certified": c.get("geometry_certified"),
            "has_certification_note": c.get("has_certification_note"),
            "iterations": row["iterations"],
            "notes": c.get("notes_out", []),
        })

    viol = [m for m in merged if m["status"] == "violated"]
    uncert = [m for m in merged if m["geometry_certified"] is False]
    cert_note = [m for m in merged if m["has_certification_note"]]
    deltas = [abs(m["clearance_delta_mm"]) for m in merged if m["clearance_delta_mm"] is not None]
    pos_clear_violated = [m for m in viol if m["clearance_after_mm_rollout"] > 0]

    summary = {
        "n_frames": len(merged),
        "n_violated": len(viol),
        "n_geometry_certified_false": len(uncert),
        "n_with_certification_note": len(cert_note),
        "n_violated_with_positive_clearance": len(pos_clear_violated),
        "violated_frames": [m["i"] for m in viol],
        "uncertified_frames": [m["i"] for m in uncert],
        "certification_note_frames": [m["i"] for m in cert_note],
        "clearance_delta_mm_max_abs": max(deltas) if deltas else None,
        "max_points": args.max_points,
        "n_finish_calls": len(CALLS),
        "validity_counts": {v: sum(1 for m in merged if m["validity"] == v)
                            for v in sorted({m["validity"] for m in merged if m["validity"]})},
        "require_certified_geometry": CALLS[0]["require_certified_geometry"] if CALLS else None,
        "violation_tolerance_mm": CALLS[0]["violation_tolerance_mm"] if CALLS else None,
    }

    out.write_text(json.dumps({
        "records": args.records, "attention": args.attention,
        "rollout_json": str(rollout_json),
        "method": "esdf_rollout.main() 을 그대로 부르고 TrajectoryOptimizer._finish 만 감쌌다",
        "summary": summary, "frames": merged,
        "all_finish_calls": CALLS, "all_constraint_sets": SCENES,
    }, indent=1, ensure_ascii=False))

    if args.dump_traj:
        from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS
        np.savez_compressed(args.dump_traj, joint_names=np.array(DEFAULT_RBY1_JOINTS),
                            **TRAJ)
        print(f"wrote {args.dump_traj}  ({len(TRAJ)} arrays)")

    print(f"\nwrote {out}")
    print(f"  violated {summary['n_violated']} / {summary['n_frames']}")
    print(f"  geometry_certified == False  {summary['n_geometry_certified_false']}")
    print(f"  인증 실패 note 를 단 프레임   {summary['n_with_certification_note']}")
    print(f"  clearance 양수인데 violated   {summary['n_violated_with_positive_clearance']}")
    print(f"  full_violation 두 벌의 최대 차 {summary['clearance_delta_mm_max_abs']} mm")
    print(f"  validity                      {summary['validity_counts']}")


if __name__ == "__main__":
    main()
