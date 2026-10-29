"""IFG goal drift — does the VLA's intended grasp point move after TrajOpt pushes the hand? CPU only, read only.

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_goal_drift --run <IFG-APPLE run> --ep 954 --fig AG3S/ag3s/docs/figures/ifg-split

Per request k of one scene, VLA alone (IFG-APPLE E0) and target split (IFG-APPLE-SPLIT SPL), same episode / seed:
  intended grasp point = TCP (FK at the teleop keyframe) of the first row of the VLA's predicted chunk (traj.npz predicted_chunks, the
                         policy output before TrajOpt) whose gripper command for the used hand is < 0.5, minus the GT target position at t=0
                         (the target does not move before the grasp); requests whose chunk has no close row are skipped
  actual hand          = replay TCP at the request's first row minus the same target position
  TrajOpt deflection   = commanded TCP of the refined chunk minus that of the reference chunk, rows 0 and 7 (server record, SPL only)
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
GRIP_COL = {"left": 7, "right": 15}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--ep", type=int, default=954)
    ap.add_argument("--fig", type=pathlib.Path, required=True)
    ap.add_argument("--cmp", default="IFG-APPLE-SPLIT:SPL", metavar="KIT:ARM",
                    help="the AG3S run to compare with VLA alone (default the target-split records)")
    ap.add_argument("--tag", default="", help="suffix of the figure name")
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    import mujoco
    from integration.rby1.bringup.avoidance_arrow import TcpKinematics, arm_targets
    v = a.run / "verify"
    m = mujoco.MjModel.from_xml_path(str(a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
    bq = np.array(m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")]); kin = TcpKinematics(m)
    kit_c, arm_c = a.cmp.split(":")
    src = {"E0": ("IFG-APPLE", "E0"), "SPL": (kit_c, arm_c)}   # key "SPL" = the compared AG3S run
    lab = {"E0": T("VLA 단독", "VLA alone"), "SPL": T("AG3S (target 분리)", "AG3S (target split)")}
    out = {}
    for arm, (kit, name) in src.items():
        rd = next(p for p in glob.glob(str(v / kit / "runs" / name / f"{name}_ep{a.ep}_s*")) if pathlib.Path(p).is_dir())
        jf = glob.glob(str(v / kit / "replay" / f"runs_{name}_{name}_ep{a.ep}_*.json"))[0]
        j = json.load(open(jf)); z = np.load(jf[:-5] + ".npz"); t = np.load(rd + "/traj.npz", allow_pickle=True)
        side = j["side"]; gc = GRIP_COL[side]; ap0 = z["target_pos"][0]
        tcp = lambda q: kin.tcp(bq, side, arm_targets("rby1_16d", q)[side])
        P, st = t["predicted_chunks"], t["chunk_start_steps"]; rows = []
        al = j["server_chunk_alignment"]["block"] if arm == "SPL" else None
        for k in range(len(P)):
            ic = np.flatnonzero(P[k][:, gc] < 0.5)
            s0 = int(st[k])
            if s0 >= len(z["tcp"]):
                break
            r = {"k": k, "t": s0, "actual_mm": ((z["tcp"][s0] - ap0) * 1000).tolist(),
                 "intended_mm": None if not len(ic) else ((tcp(P[k][ic[0]]) - ap0) * 1000).tolist(),
                 "intended_row": None if not len(ic) else int(ic[0])}
            if al is not None:
                c = np.load(f"{al['chunk_dir']}/chunk_{k + al['offset']:05d}.npz", allow_pickle=True)
                R, F = c["reference_chunk"], c["refined_chunk"]
                r["deflection_row0_mm"] = ((tcp(F[0]) - tcp(R[0])) * 1000).tolist()
                r["deflection_row7_mm"] = ((tcp(F[7]) - tcp(R[7])) * 1000).tolist()
            rows.append(r)
        g = z["grip_cmd"]; i0 = int(np.flatnonzero(g < 0.5)[0])
        out[arm] = {"side": side, "target_t0_m": ap0.tolist(), "first_close_t": i0,
                    "tcp_at_first_close_mm": ((z["tcp"][i0] - ap0) * 1000).tolist(), "rows": rows}
    # requests where both chunks carry a close row, from the first one on
    ks = [r["k"] for r in out["SPL"]["rows"] if r["intended_mm"] is not None]
    k0, k1 = min(ks) - 4, out["SPL"]["rows"][-1]["k"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2)); FS.style_axes(fig, list(axes))
    names = (T("앞 (x, + = 로봇에서 멀어짐)", "x forward"), T("옆 (y)", "y lateral"), T("위 (z)", "z up"))
    for d, ax in enumerate(axes):
        for arm, col in (("E0", FS.CATEGORICAL[2]), ("SPL", FS.CATEGORICAL[1])):
            R = [r for r in out[arm]["rows"] if k0 <= r["k"] <= min(k1, 19)]
            ax.plot([r["k"] for r in R if r["intended_mm"]], [r["intended_mm"][d] for r in R if r["intended_mm"]], "-o", ms=3.5, color=col,
                    label=T(f"{lab[arm]}: 잡으려는 지점 (VLA 예측 chunk 의 닫는 행)", f"{lab[arm]}: intended grasp point"))
            ax.plot([r["k"] for r in R], [r["actual_mm"][d] for r in R], ":", lw=1.2, color=col,
                    label=T(f"{lab[arm]}: 실제 손끝", f"{lab[arm]}: actual hand"))
        R = [r for r in out["SPL"]["rows"] if k0 <= r["k"] <= min(k1, 19)]
        ax.bar([r["k"] for r in R], [r["deflection_row0_mm"][d] for r in R], width=0.5, color=FS.GRID_INK,
               label=T("AG3S: TrajOpt 꺾임 (chunk 첫 행, 명령 기준)", "AG3S: TrajOpt deflection (row 0, commanded)"))
        ax.axhline(0, color=FS.INK_2, lw=0.5)
        ax.set_title(names[d], fontsize=9, loc="left"); ax.set_xlabel(T("서버 요청 k", "request k"), fontsize=8)
        ax.set_ylim(-40, 80 if d < 2 else 120)
        if d == 0:
            ax.set_ylabel(T("사과 위치 기준 (mm)", "relative to the apple (mm)"), fontsize=8); ax.legend(fontsize=6.6, frameon=False, loc="upper left")
    fig.suptitle(T(f"ep{a.ep}: VLA 단독은 잡으려는 지점이 사과 기준으로 고정되지만, AG3S 에서는 TrajOpt 가 바나나를 피해 손을 민 뒤(요청 12–16) "
                   "VLA 가 그 손을 따라 잡을 지점을 앞 · 옆으로 옮긴다",
                   f"ep{a.ep}: the VLA's intended grasp point is fixed relative to the apple alone; with AG3S it follows the hand TrajOpt pushed"),
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p = a.fig / f"ifg-goal-drift-ep{a.ep}{a.tag}.png"
    fig.savefig(p, dpi=175, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps({"episode": a.ep, "compared": a.cmp, "arms": out, "definitions": __doc__,
                                                  "script": "AG3S/ag3s/experiments/studies/ifg_goal_drift.py"}, indent=1, default=float))
    print(p)


if __name__ == "__main__":
    main()
