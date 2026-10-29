"""IFG-SPLIT closed-loop results on the 24 IFG-APPLE scenes: E0 (VLA alone) / C1aN (current serving) / S2 (+ alarm + 3D held rule)
/ SPL (current serving + attention partition + split-supersedes-anchor). CPU only, read only.

  MUJOCO_GL=osmesa PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_split_closedloop \
      --run <IFG-APPLE run> --out <run>/verify/IFG-APPLE-4ARM --fig AG3S/ag3s/docs/figures/ifg-split

Inputs: verify/IFG-APPLE-4ARM/agg_*_vs_*.json (agg3.py, primary = locked contact-free success), the MuJoCo replays and the server
records of C1aN and SPL, the client traj.npz (executed actions), IFG-APPLE-FAIL/scene_bodies.json (neighbour distance at reset).
Per run (C1aN, SPL), before the first gripper close (grip command < 0.5): target cluster largest extent (median), HOLD chunks with the
fingertip <= 0.12 m from the apple, accepted attention partitions, split-superseded anchors; over the run: attach revokes while the
apple is >= 0.05 m up. First executed-row divergence C1aN vs SPL from traj.npz executed_actions.
Strict success = primary AND no hand / robot_other - neighbour-fruit contact rows (reported, not the locked primary).
"""
import argparse, glob, json, pathlib, shutil, subprocess, sys, tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)
ARMS = ("E0", "C1aN", "S2", "SPL")
SUB = {"E0": "IFG-APPLE", "C1aN": "IFG-APPLE", "S2": "IFG-APPLE-S2", "SPL": "IFG-APPLE-SPLIT"}
LAB = {"E0": T("VLA 단독", "VLA alone"), "C1aN": T("기존 방식", "current"), "S2": T("+경보·3D쥠", "+alarm·3D"), "SPL": T("수정 (SPL)", "fixed (SPL)")}
NEAR = 0.082
STEPS_PER_S = 15.0


def server_metrics(v, arm, ep):
    jf = glob.glob(str(v / SUB[arm] / "replay" / f"runs_{arm}_{arm}_ep{ep}_*.json"))[0]
    j = json.load(open(jf)); z = np.load(jf[:-5] + ".npz"); al = j["server_chunk_alignment"]["block"]
    cd, off = pathlib.Path(al["chunk_dir"]), int(al["offset"])
    g = z["grip_cmd"]; c = np.flatnonzero(g < 0.5); tc = int(c[0]) if len(c) else 600
    pi, tgt = z["plan_index"], z["target_pos"]; z0 = tgt[0, 2]
    ext, rev, hold_near, sup, parts = [], [], 0, 0, 0
    for k in range(int(pi.max()) + 1):
        idx = np.flatnonzero(pi == k)
        if not len(idx):
            continue
        s = json.loads(str(np.load(cd / f"chunk_{k + off:05d}.npz", allow_pickle=True)["summary_json"]))
        m = s.get("manipulated") or {}; gg = s.get("grasp") or {}
        if gg.get("attach_revoked") and tgt[idx[0], 2] - z0 > 0.05:
            rev.append(k)
        if idx[0] < tc:
            e = (m.get("admissibility") or {}).get("extents_mm")
            if e:
                ext.append(max(e))
            if np.linalg.norm(z["finger_mid"][idx[0]] - tgt[idx[0]]) <= 0.12 and ((s.get("exec_feedback") or {}).get("n_hold") or 0) > 0:
                hold_near += 1
            sup += (m.get("anchor") or {}).get("source") == "split_superseded"
            parts += sum(bool(d.get("method") == "attention_partition" and d.get("accepted"))
                         for d in ((m.get("split") or {}).get("decisions") or []))
    return {"t_close": tc, "ext_mm_median": float(np.median(ext)) if ext else None, "hold_near": hold_near,
            "revokes_lifted_k": rev, "split_superseded_chunks": sup, "partitions_accepted": parts, "run": j["run_dir"]}


def video_frame(mp4, t_s, tmp, box):
    if shutil.which("ffmpeg") is None or not mp4.exists():
        return None
    png = tmp / f"{mp4.parent.name}_{t_s:.3f}.png"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t_s:.3f}", "-i", str(mp4), "-frames:v", "1", str(png)], capture_output=True)
    if r.returncode != 0 or not png.exists():
        return None
    x0, y0, x1, y1 = box
    return plt.imread(png)[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--fig", type=pathlib.Path, required=True)
    a = ap.parse_args()
    v = a.run / "verify"; M = v / "IFG-APPLE-4ARM"; a.fig.mkdir(parents=True, exist_ok=True)
    P = {}
    for f, arms in (("agg_E0_vs_SPL.json", ("E0", "SPL")), ("agg_C1aN_vs_SPL.json", ("C1aN",)), ("agg_S2_vs_SPL.json", ("S2",))):
        for r in json.load(open(M / f))["rows"]:
            for arm in arms:
                P.setdefault(r["episode"], {})[arm] = r["arms"][arm]
    paired = {p: json.load(open(M / f"agg_{p}.json"))["paired_primary"] for p in ("C1aN_vs_SPL", "E0_vs_SPL", "S2_vs_SPL", "E0_vs_C1aN")}
    bodies = json.load(open(v / "IFG-APPLE-FAIL" / "scene_bodies.json"))

    def nd(ep):
        b = bodies[str(ep)]; ap_ = np.array(b["apple"][:2])
        return min((float(np.linalg.norm(np.array(b[f][:2]) - ap_)), f) for f in ("banana", "orange", "pear"))
    eps = sorted(P, key=lambda e: nd(e)[0])
    rows = []
    for ep in eps:
        trC = np.load(glob.glob(str(v / "IFG-APPLE" / "runs" / "C1aN" / f"C1aN_ep{ep}_s*" / "traj.npz"))[0])["executed_actions"]
        trS = np.load(glob.glob(str(v / "IFG-APPLE-SPLIT" / "runs" / "SPL" / f"SPL_ep{ep}_s*" / "traj.npz"))[0])["executed_actions"]
        n = min(len(trC), len(trS)); dd = np.abs(trC[:n] - trS[:n]).max(axis=1)
        row = {"episode": ep, "nearest_fruit": nd(ep)[1], "nearest_fruit_m": nd(ep)[0],
               "first_divergence_row_C1aN_vs_SPL": int(np.argmax(dd > 0)) if (dd > 0).any() else None, "arms": {}}
        for arm in ARMS:
            x = P[ep][arm]; nb = sum(x["contacts"]["neighbour_rows"].values())
            row["arms"][arm] = {"primary": bool(x["primary"]), "strict": bool(x["primary"]) and nb == 0, "failure_mode": x["failure_mode"],
                                "hold_chunks": x["hold_chunks"], "neighbour_rows": nb}
        for arm in ("C1aN", "SPL"):
            row["arms"][arm].update(server_metrics(v, arm, ep))
        rows.append(row)
    N = {"paired_primary": paired, "n_scenes": len(rows),
         "n_identical_C1aN_SPL": int(sum(r["first_divergence_row_C1aN_vs_SPL"] is None for r in rows)),
         "diverged_C1aN_SPL": [r["episode"] for r in rows if r["first_divergence_row_C1aN_vs_SPL"] is not None]}
    for name, sel in (("near_le_8.2cm", lambda r: r["nearest_fruit_m"] <= NEAR), ("far_gt_8.2cm", lambda r: r["nearest_fruit_m"] > NEAR), ("all", lambda r: True)):
        sub = [r for r in rows if sel(r)]
        N[name] = {"n": len(sub), **{f"{arm}_primary": int(sum(r["arms"][arm]["primary"] for r in sub)) for arm in ARMS},
                   **{f"{arm}_strict": int(sum(r["arms"][arm]["strict"] for r in sub)) for arm in ARMS}}
    N["flips_C1aN_to_SPL"] = {"fail_to_success": [r["episode"] for r in rows if not r["arms"]["C1aN"]["primary"] and r["arms"]["SPL"]["primary"]],
                              "success_to_fail": [r["episode"] for r in rows if r["arms"]["C1aN"]["primary"] and not r["arms"]["SPL"]["primary"]]}
    N["revokes_lifted_total"] = {arm: int(sum(len(r["arms"][arm]["revokes_lifted_k"]) for r in rows)) for arm in ("C1aN", "SPL")}
    N["merged_scenes"] = {str(r["episode"]): {arm: {k: r["arms"][arm][k] for k in ("primary", "failure_mode", "ext_mm_median", "hold_near",
                                                                                  "revokes_lifted_k", "t_close", "neighbour_rows")}
                                              for arm in ("C1aN", "SPL")} for r in rows if r["episode"] in (211, 250, 572, 1345)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "split_closedloop_rows.json").write_text(json.dumps(rows, indent=1, default=float))
    (a.out / "split_closedloop_numbers.json").write_text(json.dumps(N, indent=1, default=float))

    # ------------------------------------------------------------- figure 1: outcome grid + strata bars + merged-scene mechanisms
    fig = plt.figure(figsize=(17, 9.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1.25], height_ratios=[1, 1], hspace=0.45, wspace=0.32)
    ax = fig.add_subplot(gs[:, 0]); FS.style_axes(fig, [ax])
    for i, r in enumerate(rows):
        for jx, arm in enumerate(ARMS):
            x = r["arms"][arm]
            ax.scatter(jx, i, s=140, color=FS.CATEGORICAL[2] if x["primary"] else FS.CATEGORICAL[7], edgecolor=FS.INK_2, lw=0.5, zorder=3)
            ax.text(jx, i, "✓" if x["primary"] else "×", ha="center", va="center", fontsize=7.5, color=FS.SURFACE, zorder=4)
            if x["neighbour_rows"]:
                ax.scatter(jx + 0.3, i, s=13, marker="s", color=FS.CATEGORICAL[4], zorder=4)
        if r["first_divergence_row_C1aN_vs_SPL"] is None:
            ax.plot([1, 3], [i, i], color=FS.GRID_INK, lw=1.0, zorder=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([LAB[x] for x in ARMS], fontsize=8.2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"ep{r['episode']}  {r['nearest_fruit']} {r['nearest_fruit_m'] * 100:.1f} cm" for r in rows], fontsize=7.4)
    ax.invert_yaxis(); ax.set_xlim(-0.6, 3.7)
    k = sum(r["nearest_fruit_m"] <= NEAR for r in rows); ax.axhline(k - 0.5, color=FS.INK, ls="--", lw=0.9)
    ax.text(-0.55, k - 0.6, T("↑ 이웃 과일 ≤ 8.2 cm", "↑ neighbour <= 8.2 cm"), fontsize=7.5, va="bottom")
    ax.set_title(T("a  24 장면 결과 (✓ 접촉 없는 성공)\n   ■ 이웃 과일 접촉, 회색 선 = 기존 방식과 SPL 실행 동작이 bit 단위로 같음",
                   "a  24 scenes (✓ contact-free success)\n   ■ neighbour contact; grey line = current and SPL executed actions bit-identical"),
                 fontsize=9, loc="left", color=FS.INK)
    bx = fig.add_subplot(gs[0, 1]); FS.style_axes(fig, [bx])
    groups = (("near_le_8.2cm", T("이웃 ≤ 8.2 cm (7)", "neighbour <= 8.2 cm (7)")), ("far_gt_8.2cm", T("그 밖 (17)", "others (17)")), ("all", T("전체 (24)", "all (24)")))
    w = 0.2
    for jx, arm in enumerate(ARMS):
        vals = [N[g][f"{arm}_primary"] / N[g]["n"] for g, _ in groups]; strict = [N[g][f"{arm}_strict"] / N[g]["n"] for g, _ in groups]
        xs = np.arange(3) + (jx - 1.5) * w
        bx.bar(xs, vals, w, color=FS.CATEGORICAL[[3, 7, 4, 0][jx]], label=LAB[arm])
        bx.scatter(xs, strict, s=14, color=FS.INK, marker="_", zorder=4)
        for xx, g in zip(xs, groups):
            bx.text(xx, N[g[0]][f"{arm}_primary"] / N[g[0]]["n"] + 0.02, f"{N[g[0]][f'{arm}_primary']}", ha="center", fontsize=6.8)
    bx.set_xticks(range(3)); bx.set_xticklabels([l for _, l in groups], fontsize=8); bx.set_ylim(0, 1.12)
    bx.legend(fontsize=7.2, frameon=False, ncol=4, loc="upper center"); bx.grid(axis="y", color=FS.GRID_INK, lw=0.5); bx.set_axisbelow(True)
    bx.set_title(T("b  접촉 없는 성공률 (막대 위 = 성공 수, ─ = 이웃 과일 접촉도 실패로 셀 때)", "b  contact-free success (count on bar; ─ = also failing neighbour contact)"),
                 fontsize=9, loc="left", color=FS.INK)
    tx = fig.add_subplot(gs[1, 1]); tx.axis("off")
    pp = paired
    lines = [(T("짝 비교 (24 장면)", "paired (24 scenes)"), T("둘다", "both"), T("앞만", "1st only"), T("뒤만", "2nd only"), T("둘다 실패", "neither"), T("차 [95% CI]", "diff [95% CI]"))]
    for key, (A_, B_) in (("C1aN_vs_SPL", ("C1aN", "SPL")), ("E0_vs_SPL", ("E0", "SPL")), ("S2_vs_SPL", ("S2", "SPL")), ("E0_vs_C1aN", ("E0", "C1aN"))):
        q = pp[key]; ci = q["ci95_bootstrap"]
        lines.append((f"{LAB[A_]} → {LAB[B_]}", str(q["both"]), str(q[f"only_{A_}"]), str(q[f"only_{B_}"]), str(q["neither"]),
                      f"{q[f'diff_{B_}_minus_{A_}']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"))
    tab = tx.table(cellText=[list(l) for l in lines[1:]], colLabels=list(lines[0]), loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(7.6); tab.scale(1, 1.6)
    for (ri, ci), cell in tab.get_celld().items():
        cell.set_width([0.30, 0.09, 0.09, 0.09, 0.11, 0.32][ci])
    tx.set_title(T("c  같은 장면 짝 비교 (McNemar 대신 episode bootstrap 10,000)", "c  paired comparison (episode bootstrap 10,000)"), fontsize=9, loc="left", color=FS.INK)
    cx = fig.add_subplot(gs[:, 2]); FS.style_axes(fig, [cx])
    ms = (211, 250, 572, 1345)
    labels, ext_c, ext_s = [], [], []
    for ep in ms:
        r = next(r for r in rows if r["episode"] == ep); C, S = r["arms"]["C1aN"], r["arms"]["SPL"]
        labels.append(f"ep{ep}\n{T('기존', 'cur')} {'✓' if C['primary'] else '×'} → SPL {'✓' if S['primary'] else '×'}")
        ext_c.append(C["ext_mm_median"]); ext_s.append(S["ext_mm_median"])
    y = np.arange(len(ms))
    cx.barh(y - 0.18, ext_c, 0.34, color=FS.CATEGORICAL[7], label=T("기존 방식", "current"))
    cx.barh(y + 0.18, ext_s, 0.34, color=FS.CATEGORICAL[0], label="SPL")
    for i, ep in enumerate(ms):
        r = next(r for r in rows if r["episode"] == ep); C, S = r["arms"]["C1aN"], r["arms"]["SPL"]
        cx.text(max(ext_c[i], ext_s[i]) + 3, i, T(f"사과 근처 HOLD {C['hold_near']}→{S['hold_near']}\n운반 중 attach 해제 {len(C['revokes_lifted_k'])}→{len(S['revokes_lifted_k'])}\n닫는 시각 t={C['t_close']}→{S['t_close']}",
                                                  f"HOLD near apple {C['hold_near']}→{S['hold_near']}\nattach revoked carrying {len(C['revokes_lifted_k'])}→{len(S['revokes_lifted_k'])}\nclose t={C['t_close']}→{S['t_close']}"),
                fontsize=7.4, va="center", color=FS.INK_2)
    cx.axvline(70, color=FS.INK, ls=":", lw=0.8); cx.text(71, len(ms) - 0.45, T("사과 하나 ≈ 70 mm", "one apple ~ 70 mm"), fontsize=7.2)
    cx.set_yticks(y); cx.set_yticklabels(labels, fontsize=8); cx.invert_yaxis(); cx.set_xlim(0, 260)
    cx.set_xlabel(T("첫 닫기 전 target 덩어리 최대 길이 median (mm)", "target cluster largest extent before first close, median (mm)"))
    cx.legend(fontsize=8, frameon=False, loc="lower right")
    cx.set_title(T("d  병합되던 4 장면: 덩어리 길이와 서버 판단", "d  the 4 merged scenes: cluster extent and server decisions"), fontsize=9, loc="left", color=FS.INK)
    q = paired["C1aN_vs_SPL"]
    fig.suptitle(T(f"IFG-SPLIT closed-loop (사과 24 장면, seed 1): 기존 방식 {q['both'] + q['only_C1aN']}/24 → SPL {q['both'] + q['only_SPL']}/24, "
                   f"뒤집힌 장면 실패→성공 {len(N['flips_C1aN_to_SPL']['fail_to_success'])} · 성공→실패 {len(N['flips_C1aN_to_SPL']['success_to_fail'])}, "
                   f"{N['n_identical_C1aN_SPL']} 장면은 실행 동작이 기존과 같음",
                   f"IFG-SPLIT closed loop (24 apple scenes, seed 1): current {q['both'] + q['only_C1aN']}/24 → SPL {q['both'] + q['only_SPL']}/24; "
                   f"flips fail→success {len(N['flips_C1aN_to_SPL']['fail_to_success'])}, success→fail {len(N['flips_C1aN_to_SPL']['success_to_fail'])}; "
                   f"{N['n_identical_C1aN_SPL']} scenes bit-identical to current"), fontsize=10.5, color=FS.INK, x=0.01, ha="left", y=0.995)
    p1 = a.fig / "ifg-split-closedloop.png"
    fig.savefig(p1, dpi=190, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p1.with_suffix(".json").write_text(json.dumps({"numbers": N, "rows": rows, "sources": {"run": str(a.run), "script": "AG3S/ag3s/experiments/studies/ifg_split_closedloop.py"}},
                                                  indent=1, ensure_ascii=False, default=float))

    # ------------------------------------------------------------- figure 2: real scenes (front client video), the two flipped scenes
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.2))
    sside = {"frames": []}
    rv250 = next(r for r in rows if r["episode"] == 250)["arms"]["C1aN"]["revokes_lifted_k"][0]
    shots = {250: [("C1aN", ("k", rv250), T(f"기존 방식: attach 해제 순간 (k={rv250})", f"current: attach revoked (k={rv250})")),
                   ("C1aN", ("drop", None), T("기존 방식: 공중 HOLD 뒤 낙하 직전", "current: HOLD in the air, before the drop")),
                   ("SPL", ("k", rv250), T(f"SPL: 같은 시각 (k={rv250}), 쥔 채 운반", f"SPL: same time (k={rv250}), carrying")),
                   ("SPL", ("place", None), T("SPL: 바구니에 놓음", "SPL: placed in the basket"))],
             1345: [("C1aN", ("close", None), T("기존 방식: 닫는 순간 (높게, 일찍)", "current: at close (high, early)")),
                    ("C1aN", ("t", 300), T("기존 방식: 잡기 실패 뒤", "current: after the missed grasp")),
                    ("SPL", ("close", None), T("SPL: 닫는 순간", "SPL: at close")),
                    ("SPL", ("lift", None), T("SPL: 들어 올림", "SPL: lifted"))]}
    with tempfile.TemporaryDirectory() as td:
        for i, ep in enumerate((250, 1345)):
            r = next(r for r in rows if r["episode"] == ep); apy = bodies[str(ep)]["apple"][1]
            for jx, (arm, (kind, val), lab) in enumerate(shots[ep]):
                jf = glob.glob(str(v / SUB[arm] / "replay" / f"runs_{arm}_{arm}_ep{ep}_*.json"))[0]; z = np.load(jf[:-5] + ".npz")
                lift = z["target_pos"][:, 2] - z["target_pos"][0, 2]
                if kind == "place":
                    pl = np.flatnonzero(z["cnt_apple-crate"] > 0); t = int(pl[0]) + 15 if len(pl) else 590
                elif kind == "close":
                    t = r["arms"][arm]["t_close"]
                elif kind == "lift":
                    t = int(np.argmax(lift > 0.15))
                elif kind == "k":
                    t = int(np.flatnonzero(z["plan_index"] == val)[0])
                elif kind == "drop":
                    fall = int(np.argmax((np.arange(len(lift)) > 150) & (lift < 0.0)))
                    t = int(np.flatnonzero((np.arange(len(lift)) < fall) & (lift > 0.3))[-1])
                else:
                    t = int(val)
                box = (40, 15, 360, 305) if apy < 0 else (280, 15, 600, 305)
                im = video_frame(pathlib.Path(r["arms"][arm]["run"]) / "third_person_front.mp4", t / STEPS_PER_S, pathlib.Path(td), box)
                axx = axes[i, jx]; axx.axis("off")
                if im is not None:
                    axx.imshow(im)
                axx.set_title(f"ep{ep} · {lab}\nt={t / STEPS_PER_S:.1f}s", fontsize=8.4, loc="left", color=FS.INK)
                sside["frames"].append({"episode": ep, "arm": arm, "t_row": int(t), "crop_px": box, "frame_found": im is not None})
    fig.suptitle(T("SPL로 결과가 바뀐 두 장면의 실제 모습 (정면 client 영상): ep250 바나나 8.2 cm, ep1345 오렌지 7.2 cm",
                   "the two scenes SPL turned into successes (front client video): ep250 banana 8.2 cm, ep1345 orange 7.2 cm"), fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p2 = a.fig / "ifg-split-closedloop-scenes.png"
    fig.savefig(p2, dpi=170, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p2.with_suffix(".json").write_text(json.dumps(sside, indent=1, ensure_ascii=False))
    print(p1, p2)


if __name__ == "__main__":
    main()
