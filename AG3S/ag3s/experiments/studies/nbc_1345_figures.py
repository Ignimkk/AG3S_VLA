"""AG3S-NBC-1345 figures (rule A): the palm sphere model against the hand mesh (scene), the hand–orange contact over the
episode per condition (graph), and the four conditions side by side (table).

  MUJOCO_GL=osmesa .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.studies.nbc_1345_figures \
      --verify <run>/verify/AG3S-NBC-1345 --reference <run>/verify/AG3S-ROI-CARVE-GPU-SMOKE --tdm-run <E4 run> \
      --argv-file <argv.txt> --out AG3S/ag3s/docs/figures/nbc-1345
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shlex

import numpy as np

CONDS = (("B_ROI_FPR", "기준 (허용 20 mm · 지금 구)", "reference"), ("NBC0", "허용 끔", "allowance off"),
         ("PALM", "손바닥 구 URDF 굵기", "palm at URDF radius"), ("NBC0_PALM", "둘 다", "both"))


def _replay(kit: pathlib.Path, c: str):
    j = json.loads((kit / "replay" / f"runs_{c}_{c}_ep1345_s13453.json").read_text())
    z = np.load(kit / "replay" / f"runs_{c}_{c}_ep1345_s13453.npz")
    a = json.loads((kit / "analysis" / f"{c}_ep1345_s13453.json").read_text())
    return j, z, a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--tdm-run", required=True)
    ap.add_argument("--argv-file", required=True)
    ap.add_argument("--request", type=int, default=13)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from scipy.spatial import cKDTree
    from AG3S.ag3s.experiments.common import figstyle
    from AG3S.ag3s.experiments.studies import dynamic_field_replay as dfr
    from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde
    from AG3S.ag3s.robot_models.gripper_state import FingerJointMap
    from AG3S.trajopt import serve_safe

    ko = figstyle.use_korean()
    C = figstyle.CATEGORICAL
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    V, REF = pathlib.Path(a.verify), pathlib.Path(a.reference)
    kits = {"B_ROI_FPR": REF, "NBC0": V, "PALM": V, "NBC0_PALM": V}
    data = {c: _replay(kits[c], c) for c, _, _ in CONDS}

    # ---- table
    rows, side_rows = [], {}
    for c, k_ko, k_en in CONDS:
        j, z, an = data[c]
        cr = j["contact_rows"]
        g = j["geom_detail"].get("hand-fruit_other", [])
        other = {k: v for k, v in cr.items() if v and k not in ("apple-crate", "apple-hand", "apple-table", "hand-target", "hand-fruit_other")}
        r = {"hand_orange_rows": int(cr.get("hand-fruit_other", 0)),
             "hand_orange_min_mm": None if not g else round(min(x["min_dist_m"] for x in g) * 1e3, 2),
             "hand_orange_t": None if not g else [min(x["t_first"] for x in g), max(x["t_last"] for x in g)],
             "success": bool(an["success"]), "grasp": bool(an["grasp"]), "failure_mode": an.get("failure_mode"),
             "n_close_cycles": an.get("n_close_cycles"), "close_seq": (an.get("pre_grasp") or {}).get("close_seq"),
             "n_planning": an.get("n_planning"), "other_forbidden_rows": other}
        side_rows[c] = r
        rows.append([k_ko if ko else k_en, str(r["hand_orange_rows"]), "—" if r["hand_orange_min_mm"] is None else f"{r['hand_orange_min_mm']}",
                     ("성공" if r["success"] else "실패") if ko else ("success" if r["success"] else "fail"),
                     f"{r['failure_mode']} · 닫기 {r['n_close_cycles']} 회" if ko else f"{r['failure_mode']} · {r['n_close_cycles']} closes",
                     "없음" if not other else json.dumps(other)])
    fig, ax = plt.subplots(figsize=(12, 2.6))
    figstyle.style_axes(fig, ax)
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=(["조건", "손–오렌지 접촉 row", "최대 겹침 (mm, 음수)", "성공", "쥐기", "다른 금지 접촉"] if ko else
                                           ["condition", "hand–orange rows", "min distance (mm)", "success", "grasp", "other forbidden"]),
                 loc="center", cellLoc="center", colWidths=[0.26, 0.14, 0.15, 0.08, 0.22, 0.15])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.7)
    ax.set_title("1345 오렌지 접촉 원인 분리 (seed 13453, 조건마다 1 episode)" if ko else "1345 orange contact: factor separation", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "nbc-1345-table.png", dpi=150)
    plt.close(fig)
    (out / "nbc-1345-table.json").write_text(json.dumps({"rows": side_rows, "source": {c: str(kits[c]) for c in kits}}, indent=1, ensure_ascii=False))

    # ---- graph: per control step, hand–orange and hand–apple contact; the close command
    fig, axes = plt.subplots(len(CONDS), 1, figsize=(12, 7), sharex=True)
    figstyle.style_axes(fig, axes)
    side_g = {}
    for ax, (c, k_ko, k_en) in zip(axes, CONDS):
        j, z, an = data[c]
        tt = np.asarray(z["t"])
        o = np.asarray(z["cnt_hand-fruit_other"]) > 0
        h = np.asarray(z["cnt_hand-target"]) > 0
        grip = np.asarray(z["grip_cmd"])
        ax.fill_between(tt, 0, o * 1.0, step="mid", color=C[1], alpha=0.8, label="손–오렌지 접촉" if ko else "hand–orange contact")
        ax.fill_between(tt, 0, h * 0.5, step="mid", color=C[0], alpha=0.6, label="손–사과 접촉" if ko else "hand–apple contact")
        ax.plot(tt, grip, color=figstyle.INK, lw=0.8, label="gripper 명령 (1 = 열림)" if ko else "gripper command (1 = open)")
        ax.set_ylim(-0.05, 1.1)
        ax.set_yticks([])
        ax.set_ylabel(k_ko if ko else k_en, rotation=0, ha="right", va="center", fontsize=8)
        side_g[c] = {"t_orange": tt[o].tolist(), "t_apple": tt[h].tolist(), "n_steps": int(len(tt))}
    axes[0].legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    axes[-1].set_xlabel("control step")
    fig.suptitle("손–오렌지 접촉과 쥐기: 손바닥을 굵게 하면 오렌지에 닿지 않지만 사과를 쥐지 못한다" if ko else
                 "hand–orange contact and grasp per condition", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "nbc-1345-timeline.png", dpi=150)
    plt.close(fig)
    (out / "nbc-1345-timeline.json").write_text(json.dumps(side_g))

    # ---- scene: request-time hand mesh, orange, apple and the palm / finger spheres (current vs URDF radius)
    tdm = pathlib.Path(a.tdm_run)
    kit = tdm / "verify" / "AG3S-TASK-DYNAMIC-MAP-E4"
    pdir, chunks = tde.episodes_of(kit, "B_ROI_FPR")[1345]
    base_argv = shlex.split(pathlib.Path(a.argv_file).read_text())
    base_argv = [x for i, x in enumerate(base_argv) if x not in ("--run-dir", "--record-constraints")
                 and (i == 0 or base_argv[i - 1] not in ("--run-dir", "--record-constraints"))]
    gt = tde.scene_truth(base_argv[base_argv.index("--model-xml") + 1], tde.obstacle_profile(pdir.parent.parent))
    m, d = gt.m, gt.d
    step = np.load(pdir / f"step_{a.request:05d}.npz")
    qpos = np.asarray(step["qpos"], np.float64)
    samples, _ = gt.at(qpos)
    R, o = gt.base_pose
    nb = np.vstack([samples[i] for i, (k, n, b) in enumerate(gt.kind) if n == "orange"])
    ap_ = np.vstack([samples[i] for i, (k, n, b) in enumerate(gt.kind) if n == "apple"])
    body = m.body("EE_BODY_L").id
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and (m.geom_contype[g] or m.geom_conaffinity[g])]
    gp = np.vstack([((gt._sample_local(g) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) - o) @ R for g in geoms])
    st = np.asarray(step["state"], np.float64).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    figstyle.style_axes(fig, axes)
    side_s = {"request": a.request, "variants": {}}
    dn = cKDTree(nb).query(gp)[0]
    cp = gp[dn <= dn.min() + 0.002].mean(axis=0)
    win = 0.09
    for ax, (name, flags, title) in zip(axes, (("current", "", ("지금 구 모델 (capsule 배율 0.05)", "current (scale 0.05)")),
                                               ("palm1", "--capsule-radius-scale-link ee_left=1.0 ee_right=1.0",
                                                ("손바닥 구 URDF 굵기 (ee_left 1.0)", "palm at URDF radius")))):
        args = serve_safe.build_parser().parse_args(base_argv + shlex.split(flags))
        model = serve_safe.ag3s_from_args(args, plan_horizon_steps=None, rows_per_step=None).constraint_robot_model
        model.set_joint_parameters(FingerJointMap(model).joint_values({"left": float(st[7]), "right": float(st[15])}))
        cen, rad = model.sphere_centers_numeric(dfr.robot_state_from_qpos(m, qpos))
        names = list(model.sphere_link_names)
        sel = lambda p: np.all(np.abs(p[:, :2] - cp[:2]) < win, axis=1)   # noqa: E731
        ax.scatter(*nb[sel(nb)][:, :2].T, s=3, color=C[1], label="오렌지" if ko else "orange")
        ax.scatter(*ap_[sel(ap_)][:, :2].T, s=3, color=C[2], label="사과 (target)" if ko else "apple (target)")
        ax.scatter(*gp[sel(gp)][:, :2].T, s=2, color=C[6], alpha=0.4, label="왼손 몸체 mesh" if ko else "left hand body mesh")
        ax.scatter(*gp[dn <= dn.min() + 0.002][:, :2].T, s=8, color=C[7], label="접촉 부위" if ko else "contact region")
        for k, (cc, rr) in enumerate(zip(cen, rad)):
            if names[k] in ("ee_left", "ee_finger_l1", "ee_finger_l2") and np.all(np.abs(cc[:2] - cp[:2]) < win):
                ax.add_patch(Circle((cc[0], cc[1]), rr, fill=False, lw=0.8, color={"ee_left": C[3], "ee_finger_l1": C[0], "ee_finger_l2": C[4]}[names[k]]))
        for L, col in (("ee_left", C[3]), ("ee_finger_l1", C[0]), ("ee_finger_l2", C[4])):
            ax.plot([], [], color=col, lw=1, label=f"AG3S 구 {L}" if ko else f"AG3S spheres {L}")
        grip = [k for k, n in enumerate(names) if n.startswith("ee_")]
        s2n = float((np.linalg.norm(nb[:, None] - cen[grip][None], axis=2) - rad[grip][None]).min())
        s2a = float((np.linalg.norm(ap_[:, None] - cen[[k for k in grip if names[k] == "ee_left"]][None], axis=2)
                     - rad[[k for k in grip if names[k] == "ee_left"]][None]).min())
        side_s["variants"][name] = {"flags": flags, "spheres_to_orange_mm": round(s2n * 1e3, 2), "palm_spheres_to_apple_mm": round(s2a * 1e3, 2),
                                    "palm_radius_mm": round(float(rad[[k for k in grip if names[k] == "ee_left"]].max()) * 1e3, 2)}
        ax.set_xlim(cp[0] - win, cp[0] + win)
        ax.set_ylim(cp[1] - win, cp[1] + win)
        ax.set_aspect("equal")
        ax.set_xlabel("base x (m)")
        ax.set_ylabel("base y (m)")
        ax.set_title((f"{title[0 if ko else 1]} — 구→오렌지 {s2n * 1e3:.1f} mm, 손바닥 구→사과 {s2a * 1e3:.1f} mm" if ko else
                      f"{title[1]} — spheres→orange {s2n * 1e3:.1f} mm, palm→apple {s2a * 1e3:.1f} mm"), fontsize=9)
    side_s["geom_to_orange_mm"] = round(float(dn.min()) * 1e3, 2)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle((f"episode 1345 요청 {a.request}, 위에서 본 손 (실제 거리 손–오렌지 {dn.min() * 1e3:.1f} mm)" if ko else
                  f"episode 1345 request {a.request}, top view"), fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "nbc-1345-scene.png", dpi=150)
    plt.close(fig)
    (out / "nbc-1345-scene.json").write_text(json.dumps(side_s, indent=1))
    print(json.dumps({"table": side_rows, "scene": side_s}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
