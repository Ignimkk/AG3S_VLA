"""AG3S-PALM-COVER-1345 figures (rule A): the palm-cover spheres on the hand mesh (scene), TrajOpt's deflection and worst row per
chunk (graph), and the 1345 conditions side by side (table).

  MUJOCO_GL=osmesa .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.studies.palm_cover_1345_figures \
      --verify <run>/verify/AG3S-PALM-COVER-1345 --nbc <run>/verify/AG3S-NBC-1345 --reference <run>/verify/AG3S-ROI-CARVE-GPU-SMOKE \
      --tdm-run <E4 run> --argv-file <argv.txt> --out AG3S/ag3s/docs/figures/palm-cover
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import pathlib
import shlex

import numpy as np


def _chunks(kit: pathlib.Path, arm: str) -> list[dict]:
    """Per server chunk of the 1345 run of `arm`: t, grasp state, TrajOpt's largest deflection and its worst row."""
    rows = []
    for sdir in sorted(glob.glob(str(kit / "runs" / "server_*"))):
        if (pathlib.Path(sdir) / "arm").read_text().strip() != arm:
            continue
        for f in sorted(glob.glob(sdir + "/server_constraints/run_0000/chunk_*.npz")):
            z = np.load(f, allow_pickle=True)
            s = json.loads(str(z["summary_json"]))
            if (s.get("policy_seed") or {}).get("episode") != 13453:
                continue
            mv = ((s.get("to") or {}).get("metrics") or {}).get("max_violation_pair")
            mv = ast.literal_eval(mv) if isinstance(mv, str) else (mv or {})
            rows.append({"t": int(z["t_step"]), "state": (s.get("grasp") or {}).get("state"),
                         "deflection_rad": round(float(np.abs(z["refined_chunk"][:16, :14] - z["reference_chunk"][:16, :14]).max()), 4),
                         "worst_link": mv.get("link"), "worst_clearance_mm": None if mv.get("clearance_m") is None else round(mv["clearance_m"] * 1e3, 2)})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", required=True)
    ap.add_argument("--nbc", required=True)
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
    V, NBC, REF = pathlib.Path(a.verify), pathlib.Path(a.nbc), pathlib.Path(a.reference)
    conds = (("B_ROI_FPR", REF, "지금 구 모델 (기준)", "current spheres (reference)"),
             ("PALM", NBC, "손바닥 URDF 굵기", "palm at URDF radius"),
             ("PC", V, "palm cover (mesh 에 맞춘 구)", "palm cover (mesh-fitted)"))

    # ---- table
    rows, side = [], {}
    for arm, kit, k_ko, k_en in conds:
        j = json.loads((kit / "replay" / f"runs_{arm}_{arm}_ep1345_s13453.json").read_text())
        an = json.loads((kit / "analysis" / f"{arm}_ep1345_s13453.json").read_text())
        ch = _chunks(kit, arm)
        pre = [c for c in ch if c["state"] == "latched" and c["t"] >= 12]
        r = {"hand_orange_rows": int(j["contact_rows"].get("hand-fruit_other", 0)), "success": bool(an["success"]),
             "failure_mode": an.get("failure_mode"), "n_close_cycles": an.get("n_close_cycles"),
             "max_deflection_before_close_rad": max((c["deflection_rad"] for c in pre), default=None),
             "palm_worst_row_chunks_before_close": sum(1 for c in pre if c["worst_link"] == "ee_left"), "n_latched_chunks_from_t12": len(pre)}
        side[arm] = r
        rows.append([k_ko if ko else k_en, str(r["hand_orange_rows"]), ("성공" if r["success"] else "실패") if ko else str(r["success"]),
                     f"{r['failure_mode']} · 닫기 {r['n_close_cycles']}" if ko else f"{r['failure_mode']} · {r['n_close_cycles']}",
                     f"{r['max_deflection_before_close_rad']:.3f}", f"{r['palm_worst_row_chunks_before_close']} / {r['n_latched_chunks_from_t12']}"])
    fig, ax = plt.subplots(figsize=(12, 2.4))
    figstyle.style_axes(fig, ax)
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=(["손바닥 구", "손–오렌지 접촉 row", "성공", "쥐기", "쥐기 전 상태 chunk 의 TrajOpt 최대 수정 (rad)",
                                            "그중 손바닥이 최악 행인 chunk"] if ko else
                                           ["palm spheres", "hand–orange rows", "success", "grasp", "max deflection, latched chunks (rad)",
                                            "palm worst-row chunks (latched)"]),
                 loc="center", cellLoc="center", colWidths=[0.24, 0.13, 0.07, 0.17, 0.2, 0.19])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.7)
    ax.set_title("1345 (seed 13453): 손바닥을 정확히 덮으면 오렌지 접촉은 없어지지만 쥐기를 잃는다" if ko else
                 "1345: covering the palm removes the orange contact but loses the grasp", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "palm-cover-1345-table.png", dpi=150)
    plt.close(fig)
    (out / "palm-cover-1345-table.json").write_text(json.dumps({"rows": side, "definitions": {
        "window": "every chunk whose grasp state is latched with t_step >= 12, over the whole episode — includes re-approaches after failed closes",
        "max_deflection_before_close_rad": "max over that window of max |refined - reference| over the first 16 rows and 14 arm columns",
        "palm_worst_row_chunks_before_close": "chunks of that window whose TrajOpt max_violation_pair link is ee_left (out of n_latched_chunks_from_t12)"}},
        indent=1, ensure_ascii=False))

    # ---- graph: deflection per chunk, palm-worst chunks marked, close / hold
    fig, ax = plt.subplots(figsize=(12, 4))
    figstyle.style_axes(fig, ax)
    side_g = {}
    for (arm, kit, k_ko, k_en), col in zip(conds, (C[0], C[3], C[2])):
        ch = [c for c in _chunks(kit, arm) if c["t"] <= 30]
        tt = [c["t"] for c in ch]
        ax.plot(tt, [c["deflection_rad"] for c in ch], color=col, lw=1.4, marker="o", ms=3, label=k_ko if ko else k_en)
        pw = [c for c in ch if c["worst_link"] == "ee_left"]
        ax.scatter([c["t"] for c in pw], [c["deflection_rad"] for c in pw], s=60, facecolors="none", edgecolors=col, lw=1.2)
        first_close = next((c["t"] for c in ch if c["state"] in ("closing", "held")), None)
        if first_close is not None:
            ax.axvline(first_close, color=col, ls=":", lw=1)
        side_g[arm] = ch
    ax.scatter([], [], s=60, facecolors="none", edgecolors=figstyle.INK, label="손바닥이 TrajOpt 최악 행" if ko else "palm is the worst row")
    ax.plot([], [], color=figstyle.INK, ls=":", label="닫기 시작" if ko else "closing starts")
    ax.set_xlabel("server chunk (t_step)")
    ax.set_ylabel("TrajOpt 수정량 max |refined − reference| (rad)" if ko else "deflection (rad)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("손바닥 구가 오렌지를 피하려고 팔을 틀고, 닫기가 늦어진다" if ko else "the palm spheres steer the arm off the orange", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "palm-cover-1345-deflection.png", dpi=150)
    plt.close(fig)
    (out / "palm-cover-1345-deflection.json").write_text(json.dumps(side_g))

    # ---- scene: palm-cover spheres vs the current ones on the hand mesh (E4 record, request-time joints)
    tdm = pathlib.Path(a.tdm_run)
    pdir, chunks = tde.episodes_of(tdm / "verify" / "AG3S-TASK-DYNAMIC-MAP-E4", "B_ROI_FPR")[1345]
    base = shlex.split(pathlib.Path(a.argv_file).read_text())
    base = [x for i, x in enumerate(base) if x not in ("--run-dir", "--record-constraints")
            and (i == 0 or base[i - 1] not in ("--run-dir", "--record-constraints"))]
    gt = tde.scene_truth(base[base.index("--model-xml") + 1], tde.obstacle_profile(pdir.parent.parent))
    m, d = gt.m, gt.d
    step = np.load(pdir / f"step_{a.request:05d}.npz")
    qpos = np.asarray(step["qpos"], np.float64)
    samples, _ = gt.at(qpos)
    R, o = gt.base_pose
    nb = np.vstack([samples[i] for i, (k, n, b) in enumerate(gt.kind) if n == "orange"])
    apl = np.vstack([samples[i] for i, (k, n, b) in enumerate(gt.kind) if n == "apple"])
    body = m.body("EE_BODY_L").id
    gp = np.vstack([((gt._sample_local(g) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) - o) @ R
                    for g in range(m.ngeom) if m.geom_bodyid[g] == body and (m.geom_contype[g] or m.geom_conaffinity[g])])
    st = np.asarray(step["state"], np.float64).reshape(-1)
    dn = cKDTree(nb).query(gp)[0]
    cp = gp[dn <= dn.min() + 0.002].mean(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    figstyle.style_axes(fig, axes)
    win = 0.08
    side_s = {"request": a.request, "variants": {}}
    for ax, (name, flags, title) in zip(axes, (("current", "", ("지금 구 모델", "current spheres")),
                                               ("palm_cover", "--palm-cover on", ("palm cover (mesh 에 맞춘 구)", "palm cover")))):
        args = serve_safe.build_parser().parse_args(base + shlex.split(flags))
        model = serve_safe.ag3s_from_args(args, plan_horizon_steps=None, rows_per_step=None).constraint_robot_model
        model.set_joint_parameters(FingerJointMap(model).joint_values({"left": float(st[7]), "right": float(st[15])}))
        cen, rad = model.sphere_centers_numeric(dfr.robot_state_from_qpos(m, qpos))
        names = list(model.sphere_link_names)
        palm = [k for k, n in enumerate(names) if n == "ee_left"]
        sel = lambda p: np.all(np.abs(p[:, :2] - cp[:2]) < win, axis=1)   # noqa: E731
        ax.scatter(*nb[sel(nb)][:, :2].T, s=3, color=C[1], label="오렌지" if ko else "orange")
        ax.scatter(*apl[sel(apl)][:, :2].T, s=3, color=C[2], label="사과" if ko else "apple")
        ax.scatter(*gp[sel(gp)][:, :2].T, s=2, color=C[6], alpha=0.35, label="왼손 몸체 mesh" if ko else "left hand body mesh")
        ax.scatter(*gp[dn <= dn.min() + 0.002][:, :2].T, s=8, color=C[7], label="접촉 부위" if ko else "contact region")
        for k in palm:
            if np.all(np.abs(cen[k, :2] - cp[:2]) < win):
                ax.add_patch(Circle((cen[k, 0], cen[k, 1]), rad[k], fill=False, lw=0.6, color=C[3]))
        ax.plot([], [], color=C[3], lw=1, label="AG3S 손바닥 구" if ko else "AG3S palm spheres")
        out_c = (np.linalg.norm(gp[dn <= dn.min() + 0.002][:, None] - cen[None], axis=2) - rad[None]).min(axis=1)
        p2o = float((np.linalg.norm(nb[:, None] - cen[palm][None], axis=2) - rad[palm][None]).min())
        p2a = float((np.linalg.norm(apl[:, None] - cen[palm][None], axis=2) - rad[palm][None]).min())
        side_s["variants"][name] = {"flags": flags, "n_palm_spheres": len(palm), "contact_region_outside_mm": [round(float(out_c.min()) * 1e3, 2), round(float(out_c.max()) * 1e3, 2)],
                                    "palm_to_orange_mm": round(p2o * 1e3, 2), "palm_to_apple_mm": round(p2a * 1e3, 2)}
        ax.set_xlim(cp[0] - win, cp[0] + win)
        ax.set_ylim(cp[1] - win, cp[1] + win)
        ax.set_aspect("equal")
        ax.set_xlabel("base x (m)")
        ax.set_ylabel("base y (m)")
        ax.set_title((f"{title[0]}: 손바닥 구 {len(palm)} 개, 손바닥→오렌지 {p2o * 1e3:.1f} mm, →사과 {p2a * 1e3:.1f} mm" if ko else
                      f"{title[1]}: {len(palm)} palm spheres, palm→orange {p2o * 1e3:.1f} mm, →apple {p2a * 1e3:.1f} mm"), fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle((f"episode 1345 요청 {a.request}, 위에서 본 왼손 (실제 손–오렌지 {dn.min() * 1e3:.1f} mm)" if ko else
                  f"episode 1345 request {a.request}, top view"), fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "palm-cover-1345-scene.png", dpi=150)
    plt.close(fig)
    side_s["geom_to_orange_mm"] = round(float(dn.min()) * 1e3, 2)
    (out / "palm-cover-1345-scene.json").write_text(json.dumps(side_s, indent=1))
    print(json.dumps({"table": side, "scene": side_s}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
