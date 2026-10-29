"""IFG ep954 — (a) does TrajOpt change the grasp pose or only translate the hand? (b) why does the arm hesitate after a failed grasp?
CPU only, read only.

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_954_pose_hesitation --run <IFG-APPLE run> --fig AG3S/ag3s/docs/figures/ifg-split

(a) target split (IFG-APPLE-SPLIT SPL) ep954, requests 10-19: max over the executed rows of the TCP translation and rotation between
    the refined and the reference chunk (FK at the teleop keyframe, site frame).
(b) after the first close, requests 19-30: HSR subtask probability p(pick) recorded by the server (SPL); the policy's plan change between
    consecutive requests = max over 8 rows |TCP(chunk_k row i) - TCP(chunk_{k-1} row 8+i)| (traj.npz predicted_chunks), for SPL (failed
    grasp) and VLA alone (IFG-APPLE E0, successful grasp); TrajOpt push (SPL) and the reference chunk's minimum clearance (SPL).
"""
import argparse, glob, json, os, pathlib, sys

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
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--fig", type=pathlib.Path, required=True)
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    import mujoco
    from integration.rby1.bringup.avoidance_arrow import TcpKinematics, arm_targets
    v = a.run / "verify"; ep = 954; side = "right"
    m = mujoco.MjModel.from_xml_path(str(a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
    bq = np.array(m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")]); kin = TcpKinematics(m)

    def pose(q):
        p = kin.tcp(bq, side, arm_targets("rby1_16d", q)[side])
        return p, kin._scratch.site_xmat[kin.site_id[side]].reshape(3, 3).copy()

    ang = lambda A, B: float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))   # noqa: E731

    def load(kit, arm):
        rd = next(p for p in glob.glob(str(v / kit / "runs" / arm / f"{arm}_ep{ep}_s*")) if os.path.isdir(p))
        jf = glob.glob(str(v / kit / "replay" / f"runs_{arm}_{arm}_ep{ep}_*.json"))[0]
        return np.load(rd + "/traj.npz", allow_pickle=True), json.load(open(jf)), np.load(jf[:-5] + ".npz")

    tS, jS, zS = load("IFG-APPLE-SPLIT", "SPL"); tE, _, zE = load("IFG-APPLE", "E0")
    al = jS["server_chunk_alignment"]["block"]
    chunk = lambda k: np.load(f"{al['chunk_dir']}/chunk_{k + al['offset']:05d}.npz", allow_pickle=True)   # noqa: E731
    A = []
    for k in range(10, 20):
        c = chunk(k); tr, ro = [], []
        for i in range(8):
            p0, R0 = pose(c["reference_chunk"][i]); p1, R1 = pose(c["refined_chunk"][i])
            tr.append(np.linalg.norm(p1 - p0) * 1000); ro.append(ang(R0, R1))
        A.append({"k": k, "translation_mm": max(tr), "rotation_deg": max(ro)})
    iE = int(np.flatnonzero(zE["grip_cmd"] < 0.5)[0]); iS = int(np.flatnonzero(zS["grip_cmd"] < 0.5)[0])
    close_rot = ang(zE["tcp_R"][iE], zS["tcp_R"][iS])
    B = []
    for k in range(19, 31):
        c = chunk(k); s = json.loads(str(c["summary_json"]))
        chg = lambda P: max(np.linalg.norm(pose(P[k][i])[0] - pose(P[k - 1][8 + i])[0]) for i in range(8)) * 1000   # noqa: E731
        B.append({"k": k, "p_pick": float(((s.get("subtask") or {}).get("p") or {}).get("pick", np.nan)),
                  "plan_change_mm_ag3s": chg(tS["predicted_chunks"]), "plan_change_mm_vla_alone": chg(tE["predicted_chunks"]),
                  "to_push_mm": max(np.linalg.norm(pose(c["refined_chunk"][i])[0] - pose(c["reference_chunk"][i])[0]) for i in range(8)) * 1000,
                  "reference_clearance_min_mm": s.get("clearance_min_mm"), "latch": (s.get("grasp") or {}).get("state")})

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(16, 5.4), gridspec_kw={"width_ratios": [1, 1.4]}); FS.style_axes(fig, [ax, bx])
    ks = [r["k"] for r in A]
    ax.bar(np.array(ks) - 0.2, [r["translation_mm"] for r in A], width=0.4, color=FS.CATEGORICAL[0], label=T("이동 (mm)", "translation (mm)"))
    ax.bar(np.array(ks) + 0.2, [r["rotation_deg"] for r in A], width=0.4, color=FS.CATEGORICAL[3], label=T("회전 (deg)", "rotation (deg)"))
    ax.set_xlabel(T("서버 요청 k", "request k"), fontsize=8); ax.legend(fontsize=7.5, frameon=False)
    ax.set_title(T(f"a  TrajOpt 수정 (명령 기준, 실행 8 행 최대): 이동 6–19 mm 에 회전은 2° 이하\n   닫는 순간 그리퍼 방향 차이 (VLA 단독 대 AG3S) {close_rot:.1f}°",
                   f"a  TrajOpt edit: translation 6-19 mm, rotation <= 2 deg; gripper orientation at close vs VLA alone {close_rot:.1f} deg"),
                 fontsize=9, loc="left")
    kb = [r["k"] for r in B]
    bx.plot(kb, [r["plan_change_mm_ag3s"] for r in B], "-o", ms=3.5, color=FS.CATEGORICAL[1], label=T("AG3S (쥐지 못함): 요청 사이 VLA 계획 변화", "AG3S (failed grasp): policy plan change"))
    bx.plot(kb, [r["plan_change_mm_vla_alone"] for r in B], "-o", ms=3.5, color=FS.CATEGORICAL[2], label=T("VLA 단독 (쥠): 요청 사이 VLA 계획 변화", "VLA alone (grasped): policy plan change"))
    bx.bar(kb, [r["to_push_mm"] for r in B], width=0.5, color=FS.GRID_INK, label=T("AG3S: TrajOpt 밀기 (명령 기준)", "AG3S: TrajOpt push"))
    bx.set_ylabel("mm", fontsize=8); bx.set_xlabel(T("서버 요청 k (19 = 처음 닫음)", "request k (19 = first close)"), fontsize=8)
    b2 = bx.twinx(); b2.plot(kb, [r["p_pick"] for r in B], "--", color=FS.CATEGORICAL[6], lw=1.4, label=T("AG3S: HSR p(pick)", "AG3S: HSR p(pick)"))
    b2.set_ylim(0, 1.05); b2.set_ylabel("p(pick)", fontsize=8)
    h1, l1 = bx.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels(); bx.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper right")
    bx.set_title(T("b  쥐지 못한 뒤 바구니로 가는 동안: VLA 계획이 요청마다 크게 바뀌고 (단계 판단도 흔들림), TrajOpt 가 직전 계획 쪽으로 당김\n"
                   "   (VLA 원래 계획의 장애물 여유 ≥ 42 mm — 충돌 회피가 아님)",
                   "b  after the failed grasp: the policy's plan changes a lot per request (phase wavers); TrajOpt pulls toward the previous plan (no collision)"),
                 fontsize=9, loc="left")
    fig.suptitle(T("ep954: (a) TrajOpt 는 손을 평행 이동만 시켜 바나나를 피한다 — 자세를 바꿔 잡는 선택지가 없다. (b) 쥐지 못한 뒤의 머뭇거림은 "
                   "VLA 계획 · 단계 판단(p(pick))의 흔들림과, 그 변화를 직전 계획 쪽으로 당기는 TrajOpt 수정이 함께 나타난다",
                   "ep954: (a) TrajOpt translates the hand only; (b) the hesitation = policy conflict + TrajOpt continuity"), fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p = a.fig / "ifg-954-pose-hesitation.png"
    fig.savefig(p, dpi=175, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps({"episode": ep, "a_to_edit": A, "close_orientation_diff_deg": close_rot, "b_after_close": B,
                                                  "definitions": __doc__, "script": "AG3S/ag3s/experiments/studies/ifg_954_pose_hesitation.py"},
                                                 indent=1, default=float))
    print(p)


if __name__ == "__main__":
    main()
