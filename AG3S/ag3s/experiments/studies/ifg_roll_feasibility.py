"""IFG option B feasibility — can a wrist roll alone keep the VLA-alone grasp point and clear the neighbour? CPU only, read only.

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_roll_feasibility --run <IFG-APPLE run> --ep 954 --fig AG3S/ag3s/docs/figures/ifg-split

Question (user ruling 2026-10-21: try B -- keep the neighbour-object rule, let TrajOpt take another approach pose -- and go to A if B fails):
at the pose where VLA alone (IFG-APPLE E0) started to close, does rotating the hand about its last joint (`<side>_arm_6`, 0 mm TCP
translation per degree in the IFG ep954 per-joint analysis) give a finger pose that the AG3S rule accepts?

Inputs
  pose      : E0 control-row qpos (frames.jsonl) at the first step whose gripper command < 0.5 (`replay grip_cmd`), and the steps after
              it until the fingers stop (`--after` steps), mapped to the server's robot model (`fixed_input_ablation.q_now_from_qpos`);
              finger joints = the qpos finger joints (the model's parameter joints)
  field     : the server record of the AG3S run (`--kit:arm`, default IFG-APPLE-TOW:SPLW) at the last request before its first close,
              rebuilt with `fixed_input_ablation.build_field`; the scene is static before the grasp (same episode / seed)
  robot     : the run's own server model (`ta_offline.server_args` + `build_robot`)
Per roll offset (deg) on `<side>_arm_6`:
  clearance = min over the hand's finger / gripper spheres of (target-free layer distance - radius), i.e. the target exempt, every
              other object kept (the rule the run used); obstacle margin of the run (`--obstacle-margin`) drawn as the requirement
  tcp shift = hold-link (palm) origin displacement vs roll 0 (mm)
"""
import argparse, glob, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--ep", type=int, default=954)
    ap.add_argument("--fig", type=pathlib.Path, required=True)
    ap.add_argument("--cmp", default="IFG-APPLE-TOW:SPLW", metavar="KIT:ARM")
    ap.add_argument("--after", type=int, default=30, help="E0 steps after the first close command to evaluate (fingers closing)")
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field, q_now_from_qpos
    from AG3S.trajopt.experiments.ta_offline import build_robot, server_args
    from AG3S.ag3s.constraints.attached import _link_pose_numeric

    v = a.run / "verify"
    kit, arm = a.cmp.split(":")
    jf = glob.glob(str(v / kit / "replay" / f"runs_{arm}_{arm}_ep{a.ep}_*.json"))[0]
    j = json.load(open(jf)); side = j["side"]
    al = j["server_chunk_alignment"]["block"]
    # the AG3S run's last request before its first close (latch state 'closing')
    k_field = None
    for k in range(80):
        p = pathlib.Path(f"{al['chunk_dir']}/chunk_{k + al['offset']:05d}.npz")
        if not p.exists():
            break
        s = json.loads(str(np.load(p, allow_pickle=True)["summary_json"]))
        if str((s.get("grasp") or {}).get("state")).startswith("clos"):
            break
        k_field = k
    zf = np.load(f"{al['chunk_dir']}/chunk_{k_field + al['offset']:05d}.npz", allow_pickle=True)
    sargs, _ = server_args(pathlib.Path(al["chunk_dir"]))
    robot, qadr = build_robot(sargs)
    field = build_field(zf, outside_distance=1.0)
    margin = float(sargs.obstacle_margin)
    names = [str(n) for n in robot.sphere_link_names]
    hand_tag = "_r" if side == "right" else "_l"
    hand = np.array([("finger" in n or "gripper" in n or n.startswith("ee")) and (hand_tag in n or side in n) for n in names])
    hold_link = "ee_right" if side == "right" else "ee_left"

    # E0 poses
    rd = next(p for p in glob.glob(str(v / "IFG-APPLE/runs/E0" / f"E0_ep{a.ep}_s*")) if pathlib.Path(p).is_dir())
    je = glob.glob(str(v / "IFG-APPLE/replay" / f"runs_E0_E0_ep{a.ep}_*.json"))[0]
    ze = np.load(je[:-5] + ".npz")
    t_close = int(np.flatnonzero(ze["grip_cmd"] < 0.5)[0])
    ctrl = {}
    for line in open(rd + "/frames.jsonl"):
        r = json.loads(line)
        if r.get("kind") == "control" and r.get("qpos") is not None:
            ctrl[int(r["t_step"])] = r["qpos"]
    params = list(getattr(robot, "param_joint_names", ()) or ())
    roll_j = f"{side}_arm_6"
    jn = list(robot.joint_names); ji = jn.index(roll_j)
    lo, hi = (float(x[ji]) for x in robot.joint_limits())
    rolls = np.arange(-90, 91, 5)
    steps = [t for t in (t_close, t_close + a.after // 2, t_close + a.after) if t in ctrl]
    res = {}
    for t in steps:
        qpos = np.asarray(ctrl[t], np.float64)
        q = q_now_from_qpos(qpos, qadr)
        if params:
            robot.set_joint_parameters({p: float(qpos[qadr[p]]) for p in params if p in qadr})
        p0 = np.asarray(_link_pose_numeric(robot, q, hold_link)[:3, 3])
        rows = []
        for r in rolls:
            qq = q.copy(); qq[ji] += np.radians(r)
            c, rad = robot.sphere_centers_numeric(qq)
            c, rad = np.asarray(c)[hand], np.asarray(rad)[hand]
            d = np.asarray(field.target_free_distance(c)) - rad
            rows.append({"roll_deg": int(r), "clearance_mm": round(float(d.min()) * 1000, 1),
                         "worst_sphere": names[int(np.flatnonzero(hand)[int(np.argmin(d))])],
                         "palm_shift_mm": round(float(np.linalg.norm(np.asarray(_link_pose_numeric(robot, qq, hold_link)[:3, 3]) - p0)) * 1000, 2),
                         "within_limits": bool(lo <= qq[ji] <= hi)})
        res[t] = {"q_roll_joint_rad": float(q[ji]), "rows": rows,
                  "finger_qpos": {p: float(qpos[qadr[p]]) for p in params if p in qadr}}
    if params:
        robot.set_joint_parameters(None)

    fig, ax = plt.subplots(figsize=(11, 5.4)); FS.style_axes(fig, [ax])
    for i, t in enumerate(steps):
        rr = res[t]["rows"]
        ax.plot([x["roll_deg"] for x in rr], [x["clearance_mm"] for x in rr], "-o", ms=3, color=FS.CATEGORICAL[i],
                label=T(f"VLA 단독 t={t} (닫기 명령 +{t - t_close} step)", f"VLA alone t={t} (close +{t - t_close})"))
    ax.axhline(margin * 1000, color=FS.CATEGORICAL[7], ls="--", lw=1, label=T(f"AG3S 요구 여유 {margin * 1000:.0f} mm", f"AG3S margin {margin * 1000:.0f} mm"))
    ax.axhline(0, color=FS.INK_2, lw=0.6); ax.axvline(0, color=FS.INK_2, lw=0.6, ls=":")
    ax.set_xlabel(T(f"{roll_j} 회전 추가 (deg) — 손끝 위치 그대로, 손 방향만 바뀜", f"{roll_j} roll offset (deg)"), fontsize=8.5)
    ax.set_ylabel(T("손 · 손가락 구 최소 거리 (target 제외, mm)", "hand sphere clearance, target exempt (mm)"), fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False)
    ok0 = [x["roll_deg"] for x in res[steps[0]]["rows"] if x["clearance_mm"] >= margin * 1000 and x["within_limits"]]
    ax.set_title(T(f"ep{a.ep} ({side}): VLA 단독이 닫던 자세에서 손목만 돌리면 이웃 물체 여유를 만족하는가 — 닫기 시작 시점 만족 각도: "
                   f"{(str(min(ok0)) + '…' + str(max(ok0)) + '°') if ok0 else '없음'}",
                   f"ep{a.ep} ({side}): wrist roll only at the VLA-alone close pose -- roll range meeting the margin at the close: "
                   f"{(str(min(ok0)) + '..' + str(max(ok0)) + ' deg') if ok0 else 'none'}"), fontsize=9.5, loc="left")
    fig.tight_layout()
    p = a.fig / f"ifg-roll-feasibility-ep{a.ep}.png"
    fig.savefig(p, dpi=175, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps({"episode": a.ep, "side": side, "field_from": f"{kit}:{arm} request {k_field}",
                                                  "e0_first_close_t": t_close, "margin_mm": margin * 1000, "roll_joint": roll_j,
                                                  "results": {str(t): r for t, r in res.items()}, "definitions": __doc__,
                                                  "script": "AG3S/ag3s/experiments/studies/ifg_roll_feasibility.py"}, indent=1))
    print(p, "ok rolls at close:", ok0)
    for t in steps:
        print(t, [(x["roll_deg"], x["clearance_mm"], x["worst_sphere"]) for x in res[t]["rows"] if x["roll_deg"] % 15 == 0])


if __name__ == "__main__":
    main()
