"""IFG-APPLE-LOOKAHEADOFF — does the T43 TA (A) hurdle look-ahead break VLA-alone-success grasps? CPU only, read only.

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_lookahead_off --run <IFG-APPLE run> --fig AG3S/ag3s/docs/figures/ifg-split

10 VLA-alone-success apple scenes (seeds.json of the kit). Three configurations on the same episode / seed: VLA alone (IFG-APPLE E0),
target split (IFG-APPLE-SPLIT SPL, look-ahead on), target split without the look-ahead (SPLN). Primary = locked contact-free success (agg3.py
over the copies in <kit>/agg3arm). Per run, around the first close command (requests kc-4 .. kc+2, kc = the request whose rows contain the
first gripper command < 0.5): the largest commanded TCP push of TrajOpt (refined vs reference, executed rows, FK at the teleop keyframe),
HOLD requests, look-ahead gate open requests, and the GT body nearest to the occupied voxel behind the worst row (`ifg_regression_trace`).
"""
import argparse, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402
from AG3S.ag3s.experiments.studies.ifg_regression_trace import trace  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)
CFG = ("E0", "SPL", "SPLN")
LAB = {"E0": T("VLA 단독", "VLA alone"), "SPL": T("target 분리 (look-ahead 켬)", "target split (look-ahead on)"),
       "SPLN": T("target 분리, look-ahead 끔", "target split, look-ahead off")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--fig", type=pathlib.Path, required=True)
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    import mujoco
    from integration.rby1.bringup.avoidance_arrow import TcpKinematics, arm_targets
    v = a.run / "verify"; K = v / "IFG-APPLE-LOOKAHEADOFF"; M = K / "agg3arm"
    m = mujoco.MjModel.from_xml_path(str(a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
    base_q = np.array(m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")]); kin = TcpKinematics(m)
    S = json.load(open(M / "seeds.json"))["items"]
    bodies = json.load(open(v / "IFG-APPLE-FAIL/scene_bodies.json"))
    arms = {}
    for f in ("agg_E0_vs_SPLN.json", "agg_SPL_vs_SPLN.json"):
        for r in json.load(open(M / f))["rows"]:
            arms.setdefault(r["episode"], {}).update(r["arms"])
    out = []
    for it in S:
        ep = it["episode"]; row = {"episode": ep, "stratum": it["stratum"], "side": it["side"], "cfg": {}}
        oc = json.load(open(it["obstacle_config"])) if it["obstacle_config"] else None
        for cfg in CFG:
            x = arms[ep][cfg]
            rec = {"primary": bool(x["primary"]), "failure_mode": x["failure_mode"], "hold_chunks": x["hold_chunks"],
                   "neighbour_rows": sum(x["contacts"]["neighbour_rows"].values())}
            if cfg != "E0":
                jf = next(M.glob(f"replay/runs_{cfg}_{cfg}_ep{ep}_*.json"))
                tr = trace(str(jf), bodies, oc, kin, base_q, arm_targets)
                z = np.load(str(jf)[:-5] + ".npz"); ic = np.flatnonzero(z["grip_cmd"] < 0.5)
                kc = int(z["plan_index"][ic[0]]) if len(ic) else None
                win = [r for r in tr.get("rows", []) if kc is not None and kc - 4 <= r["k"] <= kc + 2]
                gates = []
                al = json.load(open(jf))["server_chunk_alignment"]["block"]
                for r in win:
                    c = np.load(f"{al['chunk_dir']}/chunk_{r['k'] + al['offset']:05d}.npz", allow_pickle=True)
                    la = (json.loads(str(c["summary_json"])).get("to") or {}).get("metrics", {}).get("lookahead")
                    la = eval(la) if isinstance(la, str) else (la or {})   # recorded as a Python repr
                    gates.append(la.get("gate") == "open")
                rec.update(close_k=kc, max_push_mm_near_close=max([r["to_push_mm"] for r in win] or [0.0]),
                           hold_requests_near_close=sum(r["hold"] for r in win), gate_open_near_close=int(sum(gates)),
                           worst_bodies_near_close=[r.get("worst_body") for r in win],
                           target_residual_violations_near_close=sum(1 for r in win if r.get("worst_body") == "target"
                                                                     and (r["reference_clearance_min_mm"] or 99) < 5.0),
                           window=[{k_: r[k_] for k_ in ("k", "to_push_mm", "hold", "worst_body", "reference_clearance_min_mm", "grasp")} for r in win])
            row["cfg"][cfg] = rec
        out.append(row)
    N = {cfg: int(sum(r["cfg"][cfg]["primary"] for r in out)) for cfg in CFG}
    for f in ("agg_SPL_vs_SPLN.json", "agg_E0_vs_SPLN.json", "agg_E0_vs_SPL.json"):
        N[f[4:-5]] = json.load(open(M / f))["paired_primary"]
    (M / "ifg_lookahead_off_numbers.json").write_text(json.dumps({"numbers": N, "rows": out}, indent=1, default=float))

    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.25)
    ax = fig.add_subplot(gs[0]); FS.style_axes(fig, [ax])
    for i, r in enumerate(out):
        for jx, cfg in enumerate(CFG):
            x = r["cfg"][cfg]
            ax.scatter(jx, i, s=260, color=FS.CATEGORICAL[2] if x["primary"] else FS.CATEGORICAL[7], edgecolor=FS.INK_2, lw=0.5, zorder=3)
            ax.text(jx, i, "✓" if x["primary"] else "×", ha="center", va="center", fontsize=9, color=FS.SURFACE, zorder=4)
            if not x["primary"]:
                ax.text(jx + 0.2, i + 0.28, x["failure_mode"][:12], fontsize=6.3, color=FS.INK_2)
    ax.set_xticks(range(3)); ax.set_xticklabels([LAB[c].replace(" (", "\n(").replace(", ", ",\n") for c in CFG], fontsize=7.8)
    ax.set_yticks(range(len(out)))
    ax.set_yticklabels([f"ep{r['episode']} {r['side']} [{T('바구니 옆 장애물', 'place') if r['stratum'] == 'place' else T('장애물 없음', 'clear')}]" for r in out], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(-0.5, 2.8)
    ax.set_title(T(f"a  VLA 단독 성공 사과 10 장면: 성공 {N['E0']} / {N['SPL']} / {N['SPLN']}\n   look-ahead 를 꺼도 954 · 211 은 그대로 실패, 320 은 회복, 1837 은 새로 실패",
                   f"a  10 VLA-success apple scenes: {N['E0']} / {N['SPL']} / {N['SPLN']}\n   look-ahead off: 954 · 211 still fail, 320 recovers, 1837 newly fails"),
                 fontsize=9, loc="left", color=FS.INK)
    bx = fig.add_subplot(gs[1]); FS.style_axes(fig, [bx])
    xs = np.arange(len(out)); w = 0.38
    for off, cfg, col in ((-w / 2, "SPL", FS.CATEGORICAL[0]), (w / 2, "SPLN", FS.CATEGORICAL[1])):
        vals = [r["cfg"][cfg]["max_push_mm_near_close"] for r in out]
        bx.bar(xs + off, vals, width=w, color=col, label=T(f"{LAB[cfg]}: 닫기 전후 최대 손끝 밀기 (명령 기준)", f"{LAB[cfg]}: max TCP push near close (commanded)"))
        for xi, r in zip(xs, out):
            c = r["cfg"][cfg]
            tag = ("H" * min(c["hold_requests_near_close"], 3)) + (f" T{c['target_residual_violations_near_close']}" if c["target_residual_violations_near_close"] else "")
            if tag:
                bx.text(xi + off, c["max_push_mm_near_close"] + 3, tag, ha="center", fontsize=6.5, color=FS.INK_2, rotation=90)
            if not c["primary"]:
                bx.scatter(xi + off, -6, marker="x", color=FS.CATEGORICAL[7], s=22)
    bx.set_xticks(xs); bx.set_xticklabels([f"ep{r['episode']}" for r in out], fontsize=8)
    bx.set_ylabel(T("손끝 밀기 (mm, 닫기 4 요청 전 ~ 2 요청 뒤)", "TCP push (mm, close-4 .. close+2)"), fontsize=8)
    bx.set_ylim(-12, max(160, max(r["cfg"][c]["max_push_mm_near_close"] for r in out for c in ("SPL", "SPLN")) + 25))
    bx.legend(fontsize=7, frameon=False, loc="upper right")
    bx.set_title(T("b  닫는 순간의 TrajOpt 수정 (H = HOLD 요청, T n = 목표를 지운 거리장에 남은 사과 voxel 에 걸린 위반 요청 수, × = 실패)",
                   "b  TrajOpt edits at closing (H = HOLD, T n = violations against residual target voxels, × = failure)"),
                 fontsize=8.6, loc="left", color=FS.INK)
    fig.suptitle(T("look-ahead (허들 대비) 를 꺼도 VLA 단독 성공 → 실패는 줄지 않는다 (7 → 7 / 10). 두 구성 모두 닫는 순간 손가락이 남은 사과 voxel · 이웃 과일에 걸려 TrajOpt 가 밀거나 멈춘다",
                   "turning the hurdle look-ahead off does not remove the regressions (7 -> 7 / 10); in both, fingers hit residual target voxels or a neighbour at closing"),
                 fontsize=10, x=0.01, ha="left", y=1.0)
    p1 = a.fig / "ifg-lookahead-off.png"
    fig.savefig(p1, dpi=180, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    side = {"numbers": N, "rows": [{k: v_ for k, v_ in r.items() if k != "cfg"} | {"cfg": {c: {kk: vv for kk, vv in d.items() if kk != "window"} for c, d in r["cfg"].items()}} for r in out],
            "windows": {str(r["episode"]): {c: r["cfg"][c].get("window") for c in ("SPL", "SPLN")} for r in out},
            "definitions": __doc__, "sources": {"kit": str(K), "agg": str(M), "script": "AG3S/ag3s/experiments/studies/ifg_lookahead_off.py"}}
    p1.with_suffix(".json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=float))
    print(p1)


if __name__ == "__main__":
    main()
