"""IFG-APPLE-FAIL — why the apple-scene runs failed (24 scenes x VLA alone E0 / current serving C1aN / new config S2).
CPU only, read only: aggregates, MuJoCo replays of the client records, server chunk records, the run snapshot model (FK), and the
initial fruit positions rebuilt by ifg_apple_scene_bodies.py (scene_bodies.json in --out).

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_apple_fail --run <IFG-APPLE run> --out <run>/verify/IFG-APPLE-FAIL --fig AG3S/ag3s/docs/figures/ifg-apple

Per run (C1aN, S2; E0 has no server records):
  close      first control row with grip command < 0.5; geometry at that row: tip = |finger midpoint - GT apple|, dz = its z part
  extent     max over the chunks before close of the server target cluster's largest extent (manipulated.admissibility.extents_mm)
  push       chunks before close with tip <= 0.12 m: mean z of TCP(refined) - TCP(reference) over the 8 executed rows (FK, teleop keyframe),
             max |TCP(refined) - TCP(reference)|, HOLD chunks
  revoke     grasp.attach_revoked while the GT apple is >= 0.05 m above its start height
Scene: nearest other fruit (banana / orange / pear) to the apple, xy centre distance at reset.
Failure groups (post hoc, written after reading the records; evaluator only):
  vla_also   E0 also failed (primary)
  carry      a revoke while lifted
  approach   no grasp and (near HOLD >= 3 or near push dz >= 4 mm)
  other      the rest (trajectory divergence without a single server decision near the apple)
Strict metric (reported, not the locked primary): primary AND no hand / robot_other - neighbour-fruit contact rows.
"""
import argparse, collections, glob, json, pathlib, shutil, subprocess, sys, tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)
ARMS = ("E0", "C1aN", "S2")
SIDE_IDX = {"left": list(range(0, 7)), "right": list(range(8, 15))}
NEAR_TIP, NEAR_FRUIT = 0.12, 0.082
STEPS_PER_S = 15.0


def replay_of(v, arm, ep):
    sub = "IFG-APPLE-S2" if arm == "S2" else "IFG-APPLE"
    jf = glob.glob(str(v / sub / "replay" / f"runs_{arm}_{arm}_ep{ep}_*.json"))[0]
    return json.load(open(jf)), np.load(jf[:-5] + ".npz")


def close_geom(z):
    c = np.flatnonzero(z["grip_cmd"] < 0.5)
    if not len(c):
        return None
    t = int(c[0]); d = z["finger_mid"][t] - z["target_pos"][t]
    return {"t": t, "tip_m": float(np.linalg.norm(d)), "dz_m": float(d[2])}


def server_stats(j, z, kin, base):
    al = j["server_chunk_alignment"]["block"]; cd, off = pathlib.Path(al["chunk_dir"]), int(al["offset"])
    side = j["side"]; pi = z["plan_index"]; tgt = z["target_pos"]; z0 = tgt[0, 2]
    cg = close_geom(z); tclose = cg["t"] if cg else 600
    from integration.rby1.bringup.avoidance_arrow import arm_targets
    ext, near, revokes, holds_lifted = [], [], [], 0
    for k in range(int(pi.max()) + 1):
        idx = np.flatnonzero(pi == k)
        if not len(idx):
            continue
        t0 = int(idx[0]); c = np.load(cd / f"chunk_{k + off:05d}.npz", allow_pickle=True)
        s = json.loads(str(c["summary_json"])); g = s.get("grasp") or {}; ef = s.get("exec_feedback") or {}
        lift = float(tgt[t0, 2] - z0); hold = bool((ef.get("n_hold") or 0) > 0)
        rv = g.get("attach_revoked")
        if rv and lift >= 0.05:
            revokes.append({"k": k, "lift_m": lift, "reasons": rv.get("reasons"),
                            "observed_centroid": (rv.get("anchor") or {}).get("centroid")})
        if lift >= 0.05 and hold:
            holds_lifted += 1
        if t0 >= tclose:
            continue
        e = ((s.get("manipulated") or {}).get("admissibility") or {}).get("extents_mm")
        if e:
            ext.append(max(e))
        tip = float(np.linalg.norm(z["finger_mid"][t0] - tgt[t0]))
        if tip <= NEAR_TIP:
            ref, rf = c["reference_chunk"][:8], c["refined_chunk"][:8]
            dd = np.array([kin.tcp(base, side, arm_targets("rby1_16d", r1)[side]) - kin.tcp(base, side, arm_targets("rby1_16d", r0)[side])
                           for r0, r1 in zip(ref, rf)])
            near.append({"k": k, "dz": float(dd[:, 2].mean()), "d": float(np.linalg.norm(dd, axis=1).max()), "hold": hold})
    return {"close": cg, "extent_mm_max": max(ext) if ext else None,
            "near_n": len(near), "near_push_dz_mm": float(np.mean([n["dz"] for n in near]) * 1e3) if near else None,
            "near_push_dmax_mm": float(max(n["d"] for n in near) * 1e3) if near else None,
            "near_hold": int(sum(n["hold"] for n in near)), "revokes_lifted": revokes, "hold_chunks_lifted": holds_lifted}


def video_frame(mp4, t_s, tmp, box):
    if shutil.which("ffmpeg") is None or not mp4.exists():
        return None
    png = tmp / f"{mp4.parent.name}_{t_s:.3f}.png"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t_s:.3f}", "-i", str(mp4), "-frames:v", "1", str(png)], capture_output=True)
    if r.returncode != 0 or not png.exists():
        return None
    im = plt.imread(png); x0, y0, x1, y1 = box
    return im[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--fig", type=pathlib.Path, required=True)
    a = ap.parse_args()
    import mujoco
    from integration.rby1.bringup.avoidance_arrow import TcpKinematics
    v = a.run / "verify"
    m = mujoco.MjModel.from_xml_path(str(a.run / "snapshot/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
    base = np.array(m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")], np.float64); kin = TcpKinematics(m)
    A = {r["episode"]: r for r in json.load(open(v / "IFG-APPLE-3ARM/agg_E0_vs_C1aN.json"))["rows"]}
    B = {r["episode"]: r for r in json.load(open(v / "IFG-APPLE-3ARM/agg_C1aN_vs_S2.json"))["rows"]}
    bodies = json.load(open(a.out / "scene_bodies.json"))
    rows = []
    for ep in sorted(A):
        arm_rec = {"E0": A[ep]["arms"]["E0"], "C1aN": A[ep]["arms"]["C1aN"], "S2": B[ep]["arms"]["S2"]}
        b = bodies[str(ep)]; ap_xy = np.array(b["apple"][:2])
        nb = min((float(np.linalg.norm(np.array(b[f][:2]) - ap_xy)), f) for f in ("banana", "orange", "pear"))
        row = {"episode": ep, "stratum": A[ep]["stratum"], "nearest_fruit": nb[1], "nearest_fruit_m": nb[0], "arms": {}}
        for arm in ARMS:
            r = arm_rec[arm]; c = r["contacts"]; nbr = sum(c["neighbour_rows"].values())
            j, z = replay_of(v, arm, ep)
            st = server_stats(j, z, kin, base) if arm != "E0" else {"close": close_geom(z)}
            row["arms"][arm] = {"primary": bool(r["primary"]), "failure_mode": r["failure_mode"], "hold_chunks": r["hold_chunks"],
                                "neighbour_rows": nbr, "strict": bool(r["primary"]) and nbr == 0, "run": j["run_dir"], "side": j["side"], **st}
        rows.append(row)

    def group(row, arm):
        x = row["arms"][arm]
        if x["primary"]:
            return None
        if not row["arms"]["E0"]["primary"]:
            return "vla_also"
        if x.get("revokes_lifted"):
            return "carry"
        if x["failure_mode"] == "no_grasp" and ((x.get("near_hold") or 0) >= 3 or (x.get("near_push_dz_mm") or 0) >= 4.0):
            return "approach"
        return "other"
    for row in rows:
        for arm in ("C1aN", "S2"):
            row["arms"][arm]["group"] = group(row, arm)
    N = {"n_scenes": len(rows)}
    for arm in ARMS:
        N[f"{arm}_primary"] = int(sum(r["arms"][arm]["primary"] for r in rows))
        N[f"{arm}_strict_no_neighbour_contact"] = int(sum(r["arms"][arm]["strict"] for r in rows))
    for name, sel in (("near_le_8.2cm", lambda r: r["nearest_fruit_m"] <= NEAR_FRUIT), ("far_gt_8.2cm", lambda r: r["nearest_fruit_m"] > NEAR_FRUIT)):
        sub = [r for r in rows if sel(r)]
        N[name] = {"n": len(sub), **{f"{arm}_primary": int(sum(r["arms"][arm]["primary"] for r in sub)) for arm in ARMS},
                   **{f"{arm}_strict": int(sum(r["arms"][arm]["strict"] for r in sub)) for arm in ARMS}}
    for arm in ("C1aN", "S2"):
        N[f"{arm}_failure_groups"] = {g: sorted(r["episode"] for r in rows if r["arms"][arm]["group"] == g) for g in ("vla_also", "carry", "approach", "other")}
    ok = [r["arms"]["C1aN"] for r in rows if r["arms"]["C1aN"]["primary"]]
    N["C1aN_success_near_push_dz_mm_range"] = [min(x["near_push_dz_mm"] for x in ok if x["near_push_dz_mm"] is not None),
                                               max(x["near_push_dz_mm"] for x in ok if x["near_push_dz_mm"] is not None)]
    N["close_dz_cm_median"] = {f"{arm}_{'ok' if s else 'fail'}": float(np.median([r["arms"][arm]["close"]["dz_m"] * 100 for r in rows
                                                                                    if r["arms"][arm]["close"] and r["arms"][arm]["primary"] == s]))
                               for arm in ("E0", "C1aN") for s in (True, False)}
    N["definitions"] = __doc__
    (a.out / "rows.json").write_text(json.dumps(rows, indent=1, default=float))
    (a.out / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False, default=float))

    # --------------------------------------------------------------- figure 1: 24 scenes x arms, sorted by neighbour distance
    rows_s = sorted(rows, key=lambda r: r["nearest_fruit_m"])
    gcol = {"vla_also": FS.GRID_INK, "carry": FS.CATEGORICAL[1], "approach": FS.CATEGORICAL[7], "other": FS.CATEGORICAL[3]}
    glab = {"vla_also": T("VLA 단독도 실패", "VLA alone also fails"), "carry": T("운반 중 attach 해제 → 낙하", "attach revoked while carrying -> drop"),
            "approach": T("접근: 이웃 과일 회피로 막힘/위로 밀림", "approach blocked / pushed up by the neighbour"),
            "other": T("뚜렷한 서버 결정 없음 (궤적 갈림)", "no single server decision (trajectory divergence)")}
    fig = plt.figure(figsize=(17, 9.6))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.25, 1.1, 1.0, 1.0, 1.0], wspace=0.08)
    axs = [fig.add_subplot(gs[0, i]) for i in range(5)]
    FS.style_axes(fig, axs)
    y = np.arange(len(rows_s))
    ax = axs[0]
    for i, r in enumerate(rows_s):
        for jx, arm in enumerate(ARMS):
            x = r["arms"][arm]; g = x.get("group")
            col = FS.CATEGORICAL[2] if x["primary"] else (gcol[g] if g else FS.CATEGORICAL[7])
            ax.scatter(jx, i, s=150, color=col, edgecolor=FS.INK_2, lw=0.5, zorder=3)
            ax.text(jx, i, "✓" if x["primary"] else "×", ha="center", va="center", fontsize=7.5, color=FS.SURFACE, zorder=4)
            if x["neighbour_rows"]:
                ax.scatter(jx + 0.3, i, s=14, marker="s", color=FS.CATEGORICAL[4], zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels([T("VLA 단독", "VLA alone"), T("기존 방식", "current"), T("새 구성", "new config")], fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels([f"ep{r['episode']}  {r['nearest_fruit']} {r['nearest_fruit_m'] * 100:.1f} cm" for r in rows_s], fontsize=7.5)
    ax.set_xlim(-0.6, 2.7); ax.invert_yaxis()
    k = sum(r["nearest_fruit_m"] <= NEAR_FRUIT for r in rows_s)
    for axx in axs:
        axx.axhline(k - 0.5, color=FS.INK, ls="--", lw=0.9)
    ax.text(-0.55, k - 0.6, T("↑ 이웃 과일 ≤ 8.2 cm", "↑ neighbour <= 8.2 cm"), fontsize=7.5, ha="left", va="bottom", color=FS.INK)
    ax.set_title(T("a  결과 (✓ 접촉 없는 성공, 색 = 실패 유형)\n   ■ = 손·로봇이 이웃 과일에 닿음 (주 지표에는 미포함)",
                   "a  outcome (✓ contact-free success; colour = failure group)\n   ■ = hand/robot touched a neighbour fruit (not in primary)"),
                 fontsize=9, loc="left", color=FS.INK)
    panels = [("extent_mm_max", T("b  target 덩어리 최대 길이 (mm)\n   사과 하나 ≈ 70", "b  target cluster max extent (mm)\n   one apple ~ 70"), (0, 160)),
              ("near_push_dz_mm", T("c  사과 12 cm 안에서 TrajOpt가\n   손끝을 위로 민 평균 (mm)", "c  mean upward TCP push by the TO\n   within 12 cm of the apple (mm)"), (-4, 9)),
              ("near_hold", T("d  사과 12 cm 안 HOLD chunk 수", "d  HOLD chunks within 12 cm of apple"), (0, 36)),
              ("close_dz", T("e  닫을 때 손끝 중점의 사과 중심 위 높이 (cm)\n   ● 기존 방식  ○ VLA 단독", "e  fingertip height above apple centre at close (cm)\n   ● current  ○ VLA alone"), (2.5, 7.5))]
    for axx, (key, title, lim) in zip(axs[1:], panels):
        for i, r in enumerate(rows_s):
            x = r["arms"]["C1aN"]; g = x.get("group")
            col = FS.CATEGORICAL[2] if x["primary"] else gcol.get(g, FS.CATEGORICAL[7])
            if key == "close_dz":
                if x["close"]:
                    axx.scatter(x["close"]["dz_m"] * 100, i, s=36, color=col, zorder=3)
                e0 = r["arms"]["E0"]["close"]
                if e0:
                    axx.scatter(min(e0["dz_m"] * 100, lim[1] - 0.1), i, s=36, facecolor="none", edgecolor=FS.INK_2, zorder=3)
            else:
                val = x.get(key)
                if val is not None:
                    axx.barh(i, val, color=col, height=0.7)
        axx.set_xlim(*lim); axx.set_yticks(y); axx.set_yticklabels([]); axx.invert_yaxis()
        axx.set_title(title, fontsize=9, loc="left", color=FS.INK); axx.grid(axis="x", color=FS.GRID_INK, lw=0.5); axx.set_axisbelow(True)
    handles = [matplotlib.patches.Patch(color=FS.CATEGORICAL[2], label=T("성공", "success"))] + [matplotlib.patches.Patch(color=c, label=glab[g]) for g, c in gcol.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    nn, ff = N["near_le_8.2cm"], N["far_gt_8.2cm"]
    fig.suptitle(T(f"사과 24 장면 실패 분석 (b–e는 기존 방식 서버 기록): 이웃 과일 ≤ 8.2 cm 장면에서 성공 VLA 단독 {nn['E0_primary']}/{nn['n']} · 기존 방식 {nn['C1aN_primary']}/{nn['n']} · 새 구성 {nn['S2_primary']}/{nn['n']},"
                   f"  그 밖 {ff['E0_primary']}/{ff['n']} · {ff['C1aN_primary']}/{ff['n']} · {ff['S2_primary']}/{ff['n']} (구분은 결과를 본 뒤 정함)",
                   f"24 apple scenes, failure analysis (b-e: current-serving server records): neighbour <= 8.2 cm success VLA {nn['E0_primary']}/{nn['n']} · current {nn['C1aN_primary']}/{nn['n']} · new {nn['S2_primary']}/{nn['n']};"
                   f" others {ff['E0_primary']}/{ff['n']} · {ff['C1aN_primary']}/{ff['n']} · {ff['S2_primary']}/{ff['n']} (split chosen after seeing the results)"),
                 fontsize=10, color=FS.INK, x=0.01, ha="left", y=0.995)
    p1 = a.fig / "ifg-apple-fail-overview.png"
    fig.savefig(p1, dpi=190, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    side = {"numbers": {k: v for k, v in N.items() if k != "definitions"},
            "rows": [{"episode": r["episode"], "nearest_fruit": r["nearest_fruit"], "nearest_fruit_m": r["nearest_fruit_m"],
                      **{arm: {kk: r["arms"][arm].get(kk) for kk in ("primary", "strict", "group", "failure_mode", "neighbour_rows", "extent_mm_max",
                                                                     "near_push_dz_mm", "near_push_dmax_mm", "near_hold", "close")} for arm in ARMS}} for r in rows_s],
            "sources": {"run": str(a.run), "script": "AG3S/ag3s/experiments/studies/ifg_apple_fail.py"}}
    p1.with_suffix(".json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=float))

    # --------------------------------------------------------------- figure 2: real scenes (front client video, crop around the apple)
    R = {r["episode"]: r for r in rows}
    picks = [(211, T("ep211 바나나 7.2 cm: 접근 중 HOLD 34번 → 사과에서 7 cm 떨어져 닫음", "ep211 banana 7.2 cm: 34 HOLDs on approach -> closes 7 cm off"), "close"),
             (954, T("ep954 바나나 8.2 cm: 바나나와 5 mm 여유 유지 → 손이 위로 밀려 높게 닫음", "ep954 banana 8.2 cm: 5 mm margin to the banana -> pushed up, closes high"), "close"),
             (250, T("ep250 바나나 8.2 cm: 쥐고 들어 올린 뒤 attach 해제 → 공중 HOLD → 낙하", "ep250 banana 8.2 cm: lifted, attach revoked -> HOLD in the air -> drop"), "drop")]
    fig, axes = plt.subplots(len(picks), 3, figsize=(13.5, 3.4 * len(picks)))
    sside = {"frames": []}
    with tempfile.TemporaryDirectory() as td:
        for i, (ep, title, mode) in enumerate(picks):
            r = R[ep]; apy = bodies[str(ep)]["apple"][1]
            box = (120, 175, 300, 305) if apy < 0 else (335, 175, 515, 305)
            if mode == "drop":
                box = (70, 15, 350, 305) if apy < 0 else (290, 15, 570, 305)
            e0, c1 = r["arms"]["E0"], r["arms"]["C1aN"]
            if mode == "close":
                shots = [("E0", e0["close"]["t"], T("VLA 단독: 닫는 순간", "VLA alone: at close")),
                         ("C1aN", e0["close"]["t"], T("기존 방식: 같은 시각", "current: same time")),
                         ("C1aN", c1["close"]["t"], T("기존 방식: 닫는 순간", "current: at close"))]
            else:
                rk = c1["revokes_lifted"][0]["k"]
                _, zc = replay_of(v, "C1aN", ep); t_rv = int(np.flatnonzero(zc["plan_index"] == rk)[0])
                lift = zc["target_pos"][:, 2] - zc["target_pos"][0, 2]
                t_fall = int(np.argmax((np.arange(len(lift)) > t_rv) & (lift < 0.0)))   # first row below the start height
                t_last = int(np.flatnonzero((np.arange(len(lift)) < t_fall) & (lift > 0.3))[-1])   # last row still lifted > 0.3 m
                shots = [("E0", t_rv, T("VLA 단독: 같은 시각 (운반 중)", "VLA alone: same time (carrying)")),
                         ("C1aN", t_rv, T(f"기존 방식: attach 해제 (k={rk})", f"current: attach revoked (k={rk})")),
                         ("C1aN", t_last, T("기존 방식: 떨어지기 직전", "current: just before the drop"))]
            for jx, (arm, t, lab) in enumerate(shots):
                im = video_frame(pathlib.Path(r["arms"][arm]["run"]) / "third_person_front.mp4", t / STEPS_PER_S, pathlib.Path(td), box)
                axx = axes[i, jx]; axx.axis("off")
                if im is not None:
                    axx.imshow(im)
                axx.set_title(f"{lab}  t={t / STEPS_PER_S:.1f}s", fontsize=8.5, loc="left", color=FS.INK)
                sside["frames"].append({"episode": ep, "arm": arm, "t_row": t, "video_t_s": t / STEPS_PER_S, "crop_px": box, "frame_found": im is not None})
            axes[i, 0].text(0, 1.22, title, transform=axes[i, 0].transAxes, fontsize=9.5, color=FS.INK, ha="left", va="bottom")
    fig.suptitle(T("실패 장면의 실제 모습 (정면 client 영상, 사과 주변 확대): 빨간 사과 앞의 노란 과일이 바나나",
                   "failure scenes (front client video, zoom on the apple): the yellow fruit in front of the red apple is the banana"),
                 fontsize=10, color=FS.INK, x=0.01, ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p2 = a.fig / "ifg-apple-fail-scenes.png"
    fig.savefig(p2, dpi=170, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p2.with_suffix(".json").write_text(json.dumps(sside, indent=1, ensure_ascii=False, default=float))
    print(p1, p2, a.out / "numbers.json")


if __name__ == "__main__":
    main()
