"""IFG-SPLIT on the 40 IFG-CL3 scenes (independent of the IFG-APPLE failure analysis): C1aN (IFG-CL3) vs SPL (IFG-CL3-SPLIT), with
SURF (IFG-CL3) for reference. CPU only, read only.

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_split_cl3 --agg <run>/verify/IFG-CL3-SPLIT-AGG --cl3 <IFG-CL3 dir> --spl <IFG-CL3-SPLIT dir> \
      --snapshot <run>/snapshot_split --fig AG3S/ag3s/docs/figures/ifg-split

Inputs: agg_C1aN_vs_SPL.json (agg3.py copy with strata clear / adjacent / place, primary = locked contact-free success), scene_bodies.json
(target's nearest other fruit at reset), traj.npz executed_actions (first divergence), the server records (grasp latch state, attach checks,
reach) and the MuJoCo replays (GT target height). ep1034's C1aN run directory is held by an NFS-stuck process and is not opened.
Latch gap = chunks with the GT target >= 0.05 m up while the server latch is not attached (closing / latched). Runs whose replay
server-chunk alignment fails its own check (plan-0 recorded target centroid > 5 cm from the recorded target) are not counted (None).
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
NEAR = 0.082
SKIP_READ = {1034}
STEPS_PER_S = 15.0


def latch_gap(base, arm, ep):
    jf = glob.glob(str(base / "replay" / f"runs_{arm}_{arm}_ep{ep}_*.json"))[0]
    j = json.load(open(jf)); z = np.load(jf[:-5] + ".npz"); al = j["server_chunk_alignment"]["block"]
    if not al.get("found") or al.get("offset") is None:
        return None, None   # no server block aligned (IFG-CL3 ep194 SURF, NFS-quarantined chunk) — not counted
    d0 = [c.get("dist_m") for c in (j["server_chunk_alignment"].get("checks") or []) if c.get("plan_index") == 0]
    if not d0 or d0[0] is None or d0[0] > 0.05:
        # the best-effort alignment points at another run's chunks (IFG-CL3 server_8293 block after the re-run ep1034:
        # C1aN ep1096 / 1211 / 1529 / 1863 — plan-0 target centroid 7.6–63 cm from the recorded target) — not counted
        return None, None
    pi = z["plan_index"]; lift = z["target_pos"][:, 2] - z["target_pos"][0, 2]; rows = []
    for k in range(int(pi.max()) + 1):
        idx = np.flatnonzero(pi == k)
        if not len(idx):
            continue
        s = json.loads(str(np.load(f"{al['chunk_dir']}/chunk_{k + al['offset']:05d}.npz", allow_pickle=True)["summary_json"]))
        g = s.get("grasp") or {}; ev = g.get("evidence") or {}
        rows.append({"k": k, "t": int(idx[0]), "lift": float(lift[idx[0]]), "state": g.get("state"), "attached": g.get("attached") is not None,
                     "reach_mm": ev.get("reach_mm"), "failed": [c for c, v in (ev.get("checks") or {}).items() if not v]})
    n = sum(r["lift"] >= 0.05 and not r["attached"] and r["state"] in ("closing", "latched") for r in rows)
    return n, rows


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
    for k in ("--agg", "--cl3", "--spl", "--snapshot", "--fig"):
        ap.add_argument(k, type=pathlib.Path, required=True)
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    import mujoco
    from integration.rby1.bringup.avoidance_arrow import TcpKinematics, arm_targets
    m = mujoco.MjModel.from_xml_path(str(a.snapshot / "robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
    base_q = np.array(m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")]); kin = TcpKinematics(m)
    AG = json.load(open(a.agg / "agg_C1aN_vs_SPL.json")); SU = json.load(open(a.agg / "agg_C1aN_vs_SURF.json"))
    B = json.load(open(a.agg / "scene_bodies.json")); S = {x["episode"]: x for x in json.load(open(a.agg / "seeds.json"))["items"]}
    surf = {r["episode"]: r["arms"]["SURF"] for r in SU["rows"]}

    def nd(ep):
        b = B[str(ep)]; t = S[ep]["target_fruit"]; tp = np.array(b[t][:2])
        return min((float(np.linalg.norm(np.array(b[f][:2]) - tp)), f) for f in ("apple", "banana", "orange", "pear") if f != t)
    rows = []
    for r in sorted(AG["rows"], key=lambda r: nd(r["episode"])[0]):
        ep, sd = r["episode"], r["seed"]
        div = None
        if ep not in SKIP_READ:
            x = np.load(a.cl3 / "runs" / "C1aN" / f"C1aN_ep{ep}_s{sd}" / "traj.npz")["executed_actions"]
            y = np.load(a.spl / "runs" / "SPL" / f"SPL_ep{ep}_s{sd}" / "traj.npz")["executed_actions"]
            n = min(len(x), len(y)); dd = np.abs(x[:n] - y[:n]).max(axis=1); div = int(np.argmax(dd > 0)) if (dd > 0).any() else -1
        row = {"episode": ep, "stratum": r["stratum"], "target": S[ep]["target_fruit"], "nearest_fruit": nd(ep)[1], "nearest_fruit_m": nd(ep)[0],
               "first_divergence_row": div, "arms": {}}
        for arm, x in (("C1aN", r["arms"]["C1aN"]), ("SPL", r["arms"]["SPL"]), ("SURF", surf[ep])):
            nb = sum(x["contacts"]["neighbour_rows"].values())
            row["arms"][arm] = {"primary": bool(x["primary"]), "strict": bool(x["primary"]) and nb == 0, "failure_mode": x["failure_mode"],
                                "hold_chunks": x["hold_chunks"], "neighbour_rows": nb}
        for arm, base in (("C1aN", a.cl3), ("SPL", a.spl)):
            if ep in SKIP_READ and arm == "C1aN":
                continue
            g, _ = latch_gap(base, arm, ep); row["arms"][arm]["latch_gap_chunks"] = g
        rows.append(row)
    q = AG["paired_primary"]
    N = {"paired_primary_C1aN_vs_SPL": q, "by_stratum": {s: v["paired_primary"] for s, v in AG["by_stratum"].items()},
         "paired_primary_C1aN_vs_SURF": SU["paired_primary"], "n": len(rows),
         "n_identical_executions": int(sum(r["first_divergence_row"] == -1 for r in rows)), "not_read": sorted(SKIP_READ),
         "diverged": [r["episode"] for r in rows if r["first_divergence_row"] not in (-1, None)],
         "flips_fail_to_success": [r["episode"] for r in rows if not r["arms"]["C1aN"]["primary"] and r["arms"]["SPL"]["primary"]],
         "flips_success_to_fail": [r["episode"] for r in rows if r["arms"]["C1aN"]["primary"] and not r["arms"]["SPL"]["primary"]],
         "latch_gap_episodes": {arm: [r["episode"] for r in rows if (r["arms"][arm].get("latch_gap_chunks") or 0) >= 2] for arm in ("C1aN", "SPL")},
         "server_alignment_not_usable": {arm: [r["episode"] for r in rows if r["arms"][arm].get("latch_gap_chunks", "nr") is None] for arm in ("C1aN", "SPL")}}
    for name, sel in (("near_le_8.2cm", lambda r: r["nearest_fruit_m"] <= NEAR), ("far_gt_8.2cm", lambda r: r["nearest_fruit_m"] > NEAR), ("all", lambda r: True)):
        sub = [r for r in rows if sel(r)]
        N[name] = {"n": len(sub), **{f"{arm}_{k}": int(sum(r["arms"][arm][k] for r in sub)) for arm in ("C1aN", "SPL", "SURF") for k in ("primary", "strict")}}
    # ep936 latch timeline (SPL): reach, lift, TO push
    _, tl = latch_gap(a.spl, "SPL", 936)
    jf = glob.glob(str(a.spl / "replay" / "runs_SPL_SPL_ep936_*.json"))[0]; j = json.load(open(jf)); al = j["server_chunk_alignment"]["block"]; side = j["side"]
    for rr in tl:
        c = np.load(f"{al['chunk_dir']}/chunk_{rr['k'] + al['offset']:05d}.npz", allow_pickle=True)
        ref, rf = c["reference_chunk"][:8], c["refined_chunk"][:8]
        rr["to_push_mm"] = float(max(np.linalg.norm(kin.tcp(base_q, side, arm_targets("rby1_16d", y_)[side]) - kin.tcp(base_q, side, arm_targets("rby1_16d", x_)[side]))
                                     for x_, y_ in zip(ref, rf)) * 1000)
    N["ep936_SPL_timeline"] = [rr for rr in tl if 18 <= rr["k"] <= 32]
    out = a.agg / "split_cl3_numbers.json"
    out.write_text(json.dumps({"numbers": N, "rows": rows}, indent=1, default=float))

    # ---------------------------------------------------------------- figure 1
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.3], height_ratios=[1, 1], hspace=0.38, wspace=0.25)
    ax = fig.add_subplot(gs[:, 0]); FS.style_axes(fig, [ax])
    arms = ("C1aN", "SPL", "SURF"); lab = {"C1aN": T("기존 방식", "current"), "SPL": T("수정 (SPL)", "fixed (SPL)"), "SURF": T("표면 영역 (참고)", "surface (ref.)")}
    for i, r in enumerate(rows):
        for jx, arm in enumerate(arms):
            x = r["arms"][arm]
            ax.scatter(jx, i, s=110, color=FS.CATEGORICAL[2] if x["primary"] else FS.CATEGORICAL[7], edgecolor=FS.INK_2, lw=0.5, zorder=3)
            ax.text(jx, i, "✓" if x["primary"] else "×", ha="center", va="center", fontsize=6.5, color=FS.SURFACE, zorder=4)
            if (x.get("latch_gap_chunks") or 0) >= 2:
                ax.scatter(jx + 0.28, i, s=16, marker="v", color=FS.CATEGORICAL[1], zorder=4)
        if r["first_divergence_row"] == -1:
            ax.plot([0, 1], [i, i], color=FS.GRID_INK, lw=1.0, zorder=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([lab[x] for x in arms], fontsize=8.2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"ep{r['episode']} {r['target'][:3]} {r['stratum'][:3]} · {r['nearest_fruit'][:3]} {r['nearest_fruit_m'] * 100:.1f}" for r in rows], fontsize=6.6)
    ax.invert_yaxis(); ax.set_xlim(-0.5, 2.6)
    k = sum(r["nearest_fruit_m"] <= NEAR for r in rows); ax.axhline(k - 0.5, color=FS.INK, ls="--", lw=0.9)
    ax.set_title(T("a  40 장면 (target · 장면 유형 · 가장 가까운 과일 cm)\n   ✓ 접촉 없는 성공 · 회색 선 = 기존과 SPL 실행 bit 동일 · ▼ 들린 채 attach 안 됨 ≥ 2 chunk",
                   "a  40 scenes (target · stratum · nearest fruit cm)\n   ✓ contact-free success · grey = bit-identical · ▼ lifted but not attached >= 2 chunks"),
                 fontsize=8.8, loc="left", color=FS.INK)
    tx = fig.add_subplot(gs[0, 1]); tx.axis("off")
    hdr = [T("구분", "group"), T("기존 방식", "current"), "SPL", T("표면 영역", "surface")]
    cells = [[T(f"전체 {N['all']['n']}", f"all {N['all']['n']}")] + [f"{N['all'][f'{x}_primary']} ({N['all'][f'{x}_strict']})" for x in arms],
             [T(f"이웃 과일 ≤ 8.2 cm ({N['near_le_8.2cm']['n']})", f"neighbour <= 8.2 cm ({N['near_le_8.2cm']['n']})")] + [f"{N['near_le_8.2cm'][f'{x}_primary']} ({N['near_le_8.2cm'][f'{x}_strict']})" for x in arms],
             [T(f"그 밖 ({N['far_gt_8.2cm']['n']})", f"others ({N['far_gt_8.2cm']['n']})")] + [f"{N['far_gt_8.2cm'][f'{x}_primary']} ({N['far_gt_8.2cm'][f'{x}_strict']})" for x in arms]]
    for s_, v in N["by_stratum"].items():
        cells.append([f"{s_} ({v['n_pairs']})", str(v["both"] + v["only_C1aN"]), str(v["both"] + v["only_SPL"]), "—"])
    ci = q["ci95_bootstrap"]
    cells.append([T("짝: 둘다/기존만/SPL만/둘다실패", "pairs: both/cur only/SPL only/neither"), f"{q['both']}/{q['only_C1aN']}/{q['only_SPL']}/{q['neither']}",
                  f"{q['diff_SPL_minus_C1aN']:+.3f}", f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"])
    tab = tx.table(cellText=cells, colLabels=hdr, loc="center", cellLoc="center"); tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.5)
    for (ri, cj), cell in tab.get_celld().items():
        cell.set_width([0.42, 0.2, 0.18, 0.2][cj])
    tx.set_title(T("b  접촉 없는 성공 수 (괄호 = 이웃 과일 접촉도 실패), 마지막 줄 = SPL − 기존 차와 95 % CI", "b  contact-free successes (strict in brackets); last row = SPL − current, 95 % CI"),
                 fontsize=9, loc="left", color=FS.INK)
    bx = fig.add_subplot(gs[1, 1]); FS.style_axes(fig, [bx])
    tl = N["ep936_SPL_timeline"]; ks = [r["k"] for r in tl]
    bx.plot(ks, [r["lift"] * 100 for r in tl], color=FS.CATEGORICAL[1], marker="o", ms=3, lw=1.6, label=T("오렌지 높이 (cm, GT)", "orange height (cm, GT)"))
    bx.plot(ks, [(r["reach_mm"] or np.nan) / 10 for r in tl], color=FS.CATEGORICAL[0], marker="s", ms=3, lw=1.4, label=T("latch reach (cm, 닫기 전 형상 기준)", "latch reach (cm, to the kept geometry)"))
    bx.axhline(12, color=FS.CATEGORICAL[0], ls=":", lw=1); bx.text(ks[0], 12.5, T("reach 문턱 12 cm", "reach limit 12 cm"), fontsize=7, color=FS.CATEGORICAL[0])
    b2 = bx.twinx(); b2.bar(ks, [r["to_push_mm"] for r in tl], color=FS.GRID_INK, width=0.6, alpha=0.9, label=T("TrajOpt 수정 최대 (mm)", "TO push max (mm)"))
    b2.set_ylabel(T("TrajOpt 가 손끝을 민 최대 (mm)", "max TCP push by the TO (mm)"), fontsize=8)
    bx.set_zorder(b2.get_zorder() + 1); bx.patch.set_visible(False)
    for r in tl:
        bx.text(r["k"], -4, (r["state"] or "")[:3] + ("·A" if r["attached"] else ""), fontsize=6.4, ha="center", color=FS.INK_2)
    bx.set_ylim(-6, 32); bx.set_xlabel(T("서버 요청 k (아래 = latch 상태)", "server request k (below: latch state)"))
    h1, l1 = bx.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels(); bx.legend(h1 + h2, l1 + l2, fontsize=7.4, frameon=False, loc="upper left")
    bx.set_title(T("c  ep936 (성공 → 실패, SPL): 닫자마자 들어 올림 → 'settled' 다음 'reach' 실패로 attach 안 됨 → 손에 든 오렌지가 장애물 → TrajOpt 가 최대 155 mm 밀고 낙하",
                   "c  ep936 (success → fail, SPL): lifted right after closing → 'settled' then 'reach' fail, never attached → held orange is an obstacle → TO pushes up to 155 mm, drop"),
                 fontsize=8.6, loc="left", color=FS.INK)
    fig.suptitle(T(f"IFG-SPLIT 독립 장면 확인 (IFG-CL3 40 장면, seed 1): 기존 방식 {q['both'] + q['only_C1aN']}/40 → SPL {q['both'] + q['only_SPL']}/40, "
                   f"실행 bit 동일 {N['n_identical_executions']} 장면 (ep1034 은 읽지 않음)",
                   f"IFG-SPLIT on independent scenes (IFG-CL3 40, seed 1): current {q['both'] + q['only_C1aN']}/40 → SPL {q['both'] + q['only_SPL']}/40; "
                   f"{N['n_identical_executions']} bit-identical (ep1034 not read)"), fontsize=10.5, x=0.01, ha="left", y=0.995)
    p1 = a.fig / "ifg-split-cl3.png"
    fig.savefig(p1, dpi=185, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p1.with_suffix(".json").write_text(json.dumps({"numbers": N, "rows": rows, "sources": {"agg": str(a.agg), "script": "AG3S/ag3s/experiments/studies/ifg_split_cl3.py"}},
                                                  indent=1, ensure_ascii=False, default=float))

    # ---------------------------------------------------------------- figure 2: real scenes ep1582 (fail → success) and ep936 (success → fail)
    def run_of(arm, ep):
        base = a.cl3 / "runs" / "C1aN" if arm == "C1aN" else a.spl / "runs" / "SPL"
        return next(p for p in base.glob(f"{arm}_ep{ep}_s*") if p.is_dir())
    def rep(arm, ep):
        base = a.cl3 if arm == "C1aN" else a.spl
        jf = glob.glob(str(base / "replay" / f"runs_{arm}_{arm}_ep{ep}_*.json"))[0]; return np.load(jf[:-5] + ".npz")
    picks = []
    for ep in (1582, 936):
        zc, zs = rep("C1aN", ep), rep("SPL", ep)
        def tclose(z):
            c = np.flatnonzero(z["grip_cmd"] < 0.5); return int(c[0]) if len(c) else 300
        def tlift(z):
            l = z["target_pos"][:, 2] - z["target_pos"][0, 2]; return int(np.argmax(l > 0.12)) if (l > 0.12).any() else 400
        picks.append((ep, [("C1aN", tclose(zc), T("기존 방식: 닫는 순간", "current: at close")), ("C1aN", min(tlift(zc) + 30, 590), T("기존 방식: 그 뒤", "current: later")),
                           ("SPL", tclose(zs), T("SPL: 닫는 순간", "SPL: at close")), ("SPL", min(tlift(zs) + 30, 590), T("SPL: 그 뒤", "SPL: later"))]))
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.4)); side = {"frames": []}
    with tempfile.TemporaryDirectory() as td:
        for i, (ep, shots) in enumerate(picks):
            tgt = S[ep]["target_fruit"]; y = B[str(ep)][tgt][1]
            box = (40, 15, 360, 305) if y < 0 else (280, 15, 600, 305)
            for jx, (arm, t, lab_) in enumerate(shots):
                im = video_frame(run_of(arm, ep) / "third_person_front.mp4", t / STEPS_PER_S, pathlib.Path(td), box)
                axx = axes[i, jx]; axx.axis("off")
                if im is not None:
                    axx.imshow(im)
                axx.set_title(f"ep{ep} ({tgt}) · {lab_}  t={t / STEPS_PER_S:.1f}s", fontsize=8.4, loc="left")
                side["frames"].append({"episode": ep, "arm": arm, "t_row": t, "crop_px": box, "frame_found": im is not None})
    fig.suptitle(T("IFG-CL3 에서 결과가 갈린 두 장면 (정면 client 영상): ep1582 실패→성공 (바나나 7.6 cm), ep936 성공→실패 (바나나 8.3 cm, latch 미attach)",
                   "the two IFG-CL3 scenes that flipped (front video): ep1582 fail→success (banana 7.6 cm), ep936 success→fail (banana 8.3 cm, never attached)"),
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p2 = a.fig / "ifg-split-cl3-scenes.png"
    fig.savefig(p2, dpi=165, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p2.with_suffix(".json").write_text(json.dumps(side, indent=1))
    print(p1, p2, out)


if __name__ == "__main__":
    main()
