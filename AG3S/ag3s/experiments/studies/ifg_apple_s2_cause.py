"""IFG-APPLE-S2-CAUSE — why the confidence alarm and the 3D held rule behaved differently on the server than offline.
CPU only, read only (server chunk records + MuJoCo replay of the client records + the held-out offline S2FK outputs).

  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_apple_s2_cause \
      --run <IFG-APPLE run> --s2fk <s2fk dir> --out <verify/IFG-APPLE-S2-CAUSE> --fig <figures/ifg-apple>

GT phase of a server request (from the replay at the request's first control row; evaluator only, never fed back):
  pre = before the first lift, fingertip midpoint > 0.08 m from the target · near = before the first lift, ≤ 0.08 m ·
  held = target ≥ 0.02 m above its start height · after_release = after the first lift, fingertip > 0.10 m from the target.
Outputs: rows.jsonl (one per server request), numbers.json, and figures ifg-apple-s2-cause(.png/.json) and
ifg-apple-s2-cause-scene(.png/.json).
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
THR, DHOLD = 0.5904207590672415, 0.0813 + 0.05
LIFT, NEAR, AWAY, ON = 0.02, 0.08, 0.10, 0.06
PHASES = ("pre", "near", "held", "after_release")
STEPS_PER_S = 15.0  # 600 control rows ↔ 40 s client video


def _summary(path):
    return json.loads(str(np.load(path, allow_pickle=True)["summary_json"]))


def _replays(pattern):
    for jf in sorted(glob.glob(pattern)):
        j = json.load(open(jf))
        al = j["server_chunk_alignment"]["block"]
        if al.get("found"):
            yield j, np.load(jf[:-5] + ".npz"), pathlib.Path(al["chunk_dir"]), int(al["offset"])


def server_rows(v):
    rows = []
    for j, z, cd, off in _replays(str(v / "IFG-APPLE-S2" / "replay" / "runs_S2_*.json")):
        pi, tgt, fm = z["plan_index"], z["target_pos"], z["finger_mid"]
        z0 = tgt[0, 2]
        lifted = released = False
        for k in range(int(pi.max()) + 1):
            idx = np.flatnonzero(pi == k)
            if not len(idx):
                continue
            t0 = int(idx[0]); gt = tgt[t0]
            c = np.load(cd / f"chunk_{k + off:05d}.npz", allow_pickle=True)
            s = json.loads(str(c["summary_json"]))
            g = s.get("grasp") or {}; h = g.get("held_3d") or {}; cg = s.get("confidence_gate") or {}
            surf = cg.get("surface") or {}; ex = s.get("exclusion") or {}
            lift = float(gt[2] - z0); tip = float(np.linalg.norm(fm[t0] - gt))
            lifted |= lift > LIFT
            released |= lifted and tip > AWAY
            ph = "after_release" if released else "held" if lift > LIFT else "near" if tip <= NEAR else "pre"
            d = (lambda p: None if p is None else float(np.linalg.norm(np.asarray(p, float) - gt)))
            prox = h.get("proxies") or {}
            near_pts = int((np.linalg.norm(np.asarray(c["filtered_cloud"])[:, :3] - gt, axis=1) < 0.04).sum()) if k % 2 == 0 else None
            checks = (g.get("evidence") or {}).get("checks") or {}
            rows.append(dict(
                ep=int(j["episode"]), k=k, t=t0, phase=ph, state=g.get("state"), state_before=g.get("state_before"),
                attached=g.get("attached") is not None, lift=lift, tip=tip,
                hand_d=h.get("hand_d"), held_gripper=h.get("held_gripper"), held_3d=h.get("held_3d"),
                gt_hand_d=min((float(np.linalg.norm(np.asarray(p) - gt)) for p in prox.values()), default=None),
                err_tc=d(h.get("target_centre")), err_surf=d(surf.get("selected_centroid")),
                gate_score=cg.get("gate_score"), gate_applied=bool(cg.get("applied")), gate_skip=cg.get("skip"),
                switched=bool(cg.get("switched")), excl_active=ex.get("active"),
                checks_3d=checks.get("held_3d"), checks_other=(all(v for kk, v in checks.items() if kk != "held_3d")
                                                               if "held_3d" in checks else None),
                pts_near_target=near_pts))
    return rows


def offline_scores(s2fk):
    F = {}
    for line in open(s2fk / "features_fk.jsonl"):
        r = json.loads(line); F[r["frame_id"]] = r
    by = collections.defaultdict(list)
    for line in open(s2fk / "run/rounds/R11b/conf.jsonl"):
        c = json.loads(line)
        if c["kind"] == "success" and c["frame_id"] in F and c.get("q_mem_chain:prefix_L8") is not None:
            by[F[c["frame_id"]]["grasp_state"]].append(float(c["q_mem_chain:prefix_L8"]))
    return by


def pct(x):
    x = np.asarray([v for v in x if v is not None], float)
    return {"n": int(len(x)), "p10_50_90": [float(np.percentile(x, p)) for p in (10, 50, 90)] if len(x) else None,
            "frac_below_thr": float(np.mean(x < THR)) if len(x) else None}


def ep572(v):
    out = {}
    for arm, pat in (("C1aN", "IFG-APPLE/replay/runs_C1aN_C1aN_ep572_*.json"), ("S2", "IFG-APPLE-S2/replay/runs_S2_S2_ep572_*.json")):
        j, z, cd, off = next(_replays(str(v / pat)))
        out[arm] = dict(z=z, cd=cd, off=off, run=j["run_dir"])
    tz = out["S2"]["z"]; z0 = tz["target_pos"][0, 2]
    rec = []
    for k in range(10, 30):
        idx = np.flatnonzero(tz["plan_index"] == k)
        c = {a: np.load(o["cd"] / f"chunk_{k + o['off']:05d}.npz", allow_pickle=True) for a, o in out.items()}
        s = {a: json.loads(str(c[a]["summary_json"])) for a in c}
        lift = {a: float(o["z"]["target_pos"][np.flatnonzero(o["z"]["plan_index"] == k)[0], 2] - z0) for a, o in out.items()}
        rec.append(dict(
            k=k, t=int(idx[0]), lift_C1aN=lift["C1aN"], lift_S2=lift["S2"],
            ref_diff=float(np.abs(c["C1aN"]["reference_chunk"] - c["S2"]["reference_chunk"]).max()),
            refined_diff=float(np.abs(c["C1aN"]["refined_chunk"] - c["S2"]["refined_chunk"]).max()),
            state_C1aN=s["C1aN"]["grasp"]["state"], state_S2=s["S2"]["grasp"]["state"],
            revoked=(s["S2"]["grasp"].get("attach_revoked") or {}).get("reasons"),
            excl_C1aN=(s["C1aN"].get("exclusion") or {}).get("active"), excl_S2=(s["S2"].get("exclusion") or {}).get("active"),
            gate_score=(s["S2"].get("confidence_gate") or {}).get("gate_score"),
            gate_applied=bool((s["S2"].get("confidence_gate") or {}).get("applied"))))
    return rec, {a: o["run"] for a, o in out.items()}


def video_frame(mp4, t_s, tmp):
    if shutil.which("ffmpeg") is None or not mp4.exists():
        return None
    png = tmp / f"{mp4.parent.name}_{t_s:.3f}.png"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t_s:.3f}", "-i", str(mp4), "-frames:v", "1", str(png)], capture_output=True)
    return plt.imread(png) if r.returncode == 0 and png.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--s2fk", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True); ap.add_argument("--fig", type=pathlib.Path, required=True)
    a = ap.parse_args()
    v = a.run / "verify"; a.out.mkdir(parents=True, exist_ok=True); a.fig.mkdir(parents=True, exist_ok=True)
    R = server_rows(v)
    with open(a.out / "rows.jsonl", "w") as fo:
        for r in R:
            fo.write(json.dumps(r) + "\n")
    off = offline_scores(a.s2fk)
    eps = collections.defaultdict(list)
    for r in R:
        eps[r["ep"]].append(r)
    for e in eps:
        eps[e].sort(key=lambda r: r["k"])

    N = {"n_requests": len(R), "n_episodes": len(eps)}
    N["gate_score_by_gt_phase_server"] = {p: pct([r["gate_score"] for r in R if r["phase"] == p]) for p in PHASES}
    N["gate_score_by_gt_state_offline"] = {p: pct(off[p]) for p in ("pre", "held", "placed")}
    N["region_err_m_by_gt_phase_server"] = {p: pct([r["err_surf"] for r in R if r["phase"] == p]) for p in PHASES}
    for p in PHASES:
        N["region_err_m_by_gt_phase_server"][p].pop("frac_below_thr")
    lock = 0
    for rs in eps.values():
        k0 = next((r["k"] for r in rs if r["switched"]), None)
        after = [r for r in rs if k0 is not None and r["k"] > k0]
        lock += bool(after) and np.mean([r["switched"] for r in after]) > 0.5
    N["memory_lock_in_episodes"] = {"n": int(lock), "of": len(eps),
                                    "definition": "episodes where more than half of the requests after the first memory switch are switches"}
    wrong = [r for r in R if r["err_surf"] is not None and r["err_surf"] > 0.10]
    N["region_off_target_gt_10cm"] = {"n": len(wrong), "gate_score_at_or_above_thr": int(sum(r["gate_score"] is not None and r["gate_score"] >= THR for r in wrong))}
    N["gate_applied_by_gt_phase"] = dict(collections.Counter(r["phase"] for r in R if r["gate_applied"]))
    pts = collections.defaultdict(list)
    for r in R:
        if r["pts_near_target"] is not None:
            pts[f"attached={r['attached']},lifted={r['lift'] > LIFT}"].append(r["pts_near_target"])
    N["filtered_cloud_points_within_4cm_of_gt_target"] = {k: {"n": len(x), "median": float(np.median(x)), "frac_zero": float(np.mean(np.array(x) == 0))}
                                                          for k, x in sorted(pts.items())}
    att = [r for r in R if r["attached"] and r["lift"] > LIFT]
    N["held_3d_target_centre_err_m_attached_lifted"] = pct([r["err_tc"] for r in att]); N["held_3d_target_centre_err_m_attached_lifted"].pop("frac_below_thr")
    N["gt_hand_d_m_attached_lifted"] = pct([r["gt_hand_d"] for r in att]); N["gt_hand_d_m_attached_lifted"].pop("frac_below_thr")
    xt = collections.Counter()
    for r in R:
        if r["state_before"] == "closing" and r["checks_3d"] is not None:
            xt[f"{'on' if r['tip'] <= ON else 'away'}|others_{'pass' if r['checks_other'] else 'fail'}|3d_{'pass' if r['checks_3d'] else 'fail'}"] += 1
    N["attach_decision_crosstab"] = dict(sorted(xt.items()))
    N["attach_decision_crosstab_definition"] = "server requests decided from CLOSING; on = fingertip midpoint ≤ 0.06 m from GT target; others = the existing attach checks"
    N["attach_flipped_by_3d_rule"] = int(xt.get("on|others_pass|3d_fail", 0) + xt.get("away|others_pass|3d_fail", 0))
    first_held = {}
    for arm, pat in (("C1aN", "IFG-APPLE/replay/runs_C1aN_*.json"), ("S2", "IFG-APPLE-S2/replay/runs_S2_*.json")):
        for j, z, cd, o in _replays(str(v / pat)):
            fh = None
            for k in range(int(z["plan_index"].max()) + 1):
                f = cd / f"chunk_{k + o:05d}.npz"
                if f.exists() and (_summary(f).get("grasp") or {}).get("state") == "held":
                    fh = k; break
            first_held.setdefault(int(j["episode"]), {})[arm] = fh
    N["first_attach_request_same"] = {"n_same": int(sum(d.get("C1aN") == d.get("S2") for d in first_held.values())), "of": len(first_held),
                                      "n_attached_C1aN": int(sum(d.get("C1aN") is not None for d in first_held.values()))}
    e572, runs572 = ep572(v)
    first_div = next((r["k"] for r in e572 if r["refined_diff"] > 0), None)
    N["ep572"] = {"rows": e572, "first_refined_divergence_k": first_div,
                  "first_gate_applied_k": next((r["k"] for r in e572 if r["gate_applied"]), None), "runs": runs572}
    (a.out / "numbers.json").write_text(json.dumps(N, indent=1, ensure_ascii=False, default=float))

    # ------------------------------------------------------------------ figure 1: graphs
    fig = plt.figure(figsize=(18.5, 9.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1], height_ratios=[1, 1], hspace=0.42, wspace=0.62)
    ax = fig.add_subplot(gs[:, 0]); bx = fig.add_subplot(gs[0, 1]); cx = fig.add_subplot(gs[0, 2]); dx = fig.add_subplot(gs[1, 1:])
    FS.style_axes(fig, [ax, bx, cx, dx])
    order = sorted(eps)
    K = max(len(rs) for rs in eps.values())
    img = np.full((len(order), K), np.nan)
    for i, e in enumerate(order):
        for r in eps[e]:
            if r["gate_score"] is not None:
                img[i, r["k"]] = r["gate_score"]
    cmap = matplotlib.colors.ListedColormap([FS.CATEGORICAL[7], "#f3b5a8", FS.SEQ_BLUE[2], FS.SEQ_BLUE[7]])
    norm = matplotlib.colors.BoundaryNorm([-1, 0.3, THR, 0.9, 1.01], cmap.N)
    ax.imshow(img, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    for i, e in enumerate(order):
        ks = [r["k"] for r in eps[e] if r["phase"] == "held"]
        if ks:
            ax.plot([min(ks) - 0.5, max(ks) + 0.5], [i + 0.42, i + 0.42], color=FS.INK, lw=2.2, solid_capstyle="butt")
        sw = [r["k"] for r in eps[e] if r["switched"]]
        ax.scatter(sw, [i] * len(sw), s=3, color=FS.INK, marker="o", lw=0)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"ep{e}" for e in order], fontsize=7)
    ax.set_xlabel(T("서버 요청 순서 k (약 0.53 s 간격)", "server request k (~0.53 s apart)"))
    ax.set_title(T("a  경보 점수 시간축 (24 장면). 검은 선 = 사과가 실제로 들린 구간(GT), 점 = 기억이 다른 영역으로 전환",
                   "a  alarm score over time (24 scenes). bar = apple actually lifted (GT), dot = memory switched"), fontsize=9, loc="left", color=FS.INK)
    handles = [matplotlib.patches.Patch(color=cmap(i), label=l) for i, l in enumerate(("< 0.3 (" + T("경보", "alarm") + ")", f"0.3–{THR:.2f} (" + T("경보", "alarm") + ")", f"{THR:.2f}–0.9", "≥ 0.9"))]
    ax.legend(handles=handles, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=4)

    data = [off["pre"], off["held"], off["placed"]] + [[r["gate_score"] for r in R if r["phase"] == p and r["gate_score"] is not None] for p in PHASES]
    names = [T("offline\n잡기 전", "offline\npre"), T("offline\n쥠", "offline\nheld"), T("offline\n놓은 뒤", "offline\nplaced"),
             T("서버\n접근", "server\npre"), T("서버\n손 근처", "server\nnear"), T("서버\n들림", "server\nheld"), T("서버\n놓은 뒤", "server\nreleased")]
    bp = bx.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(FS.GRID_INK if i < 3 else FS.CATEGORICAL[0]); b.set_alpha(0.85); b.set_edgecolor(FS.INK_2)
    for m in bp["medians"]:
        m.set_color(FS.INK)
    bx.axhline(THR, color=FS.CATEGORICAL[7], ls="--", lw=1)
    for i, d in enumerate(data):
        bx.text(i + 1, -0.17, f"{np.mean(np.array(d) < THR):.0%}", ha="center", fontsize=7.5, color=FS.CATEGORICAL[7])
    bx.set_xticklabels(names, fontsize=7.2, color=FS.INK); bx.set_ylim(-0.25, 1.05)
    bx.set_title(T("b  같은 GT 단계 기준 경보 점수: offline 대 서버\n(아래 = 기준 0.590 미만 비율)", "b  alarm score by GT phase: offline vs server\n(below: fraction under 0.590)"),
                 fontsize=9, loc="left", color=FS.INK)
    bx.grid(axis="y", color=FS.GRID_INK, lw=0.5); bx.set_axisbelow(True)

    cats = [("on|others_pass|3d_pass", T("사과 위·기존 통과·3D 통과", "on · others pass · 3D pass")),
            ("on|others_fail|3d_pass", T("사과 위·기존 거절·3D 통과", "on · others fail · 3D pass")),
            ("on|others_fail|3d_fail", T("사과 위·기존 거절·3D 거절", "on · others fail · 3D fail")),
            ("on|others_pass|3d_fail", T("사과 위·기존 통과·3D 거절 (뒤집힘)", "on · others pass · 3D fail (flipped)")),
            ("away|others_fail|3d_pass", T("떨어짐·기존 거절·3D 통과", "away · others fail · 3D pass")),
            ("away|others_fail|3d_fail", T("떨어짐·기존 거절·3D 거절", "away · others fail · 3D fail")),
            ("away|others_pass|3d_fail", T("떨어짐·기존 통과·3D 거절 (뒤집힘)", "away · others pass · 3D fail (flipped)"))]
    vals = [xt.get(c, 0) for c, _ in cats]
    cols = [FS.CATEGORICAL[2] if "flipped" in l or "뒤집힘" in l else FS.CATEGORICAL[0] for _, l in cats]
    cx.barh(range(len(cats)), vals, color=cols)
    for i, vv in enumerate(vals):
        cx.text(vv + 2, i, str(vv), va="center", fontsize=7.5, color=FS.INK_2)
    cx.set_yticks(range(len(cats))); cx.set_yticklabels([l for _, l in cats], fontsize=7.2); cx.invert_yaxis()
    cx.set_xlim(0, max(vals) * 1.18)
    cx.set_title(T("c  attach 판단 요청(CLOSING) 교차표: 3D 규칙이 결정을\n뒤집은 요청 = 0 (24 장면 attach 시점 모두 기존과 같음)",
                   "c  attach decisions (from CLOSING): requests the 3D rule\nflipped = 0 (attach request identical in all 24 scenes)"), fontsize=9, loc="left", color=FS.INK)
    cx.set_xlabel(T("요청 수", "requests"))

    ks = [r["k"] for r in e572]
    dx.plot(ks, [r["lift_C1aN"] for r in e572], color=FS.CATEGORICAL[0], lw=1.8, marker="o", ms=3, label=T("사과 높이 — 기존 방식", "apple height — current serving"))
    dx.plot(ks, [r["lift_S2"] for r in e572], color=FS.CATEGORICAL[1], lw=1.8, marker="o", ms=3, label=T("사과 높이 — 새 구성", "apple height — new config"))
    dx.set_ylabel(T("사과 높이 변화 (m)", "apple lift (m)"))
    d2 = dx.twinx()
    d2.bar(ks, [r["refined_diff"] for r in e572], color=FS.GRID_INK, width=0.6, alpha=0.9, label=T("두 arm 실행 action 차 (rad)", "refined action difference (rad)"))
    d2.set_ylabel(T("두 arm의 실행 action 최대 차 (rad)", "max refined action diff (rad)"), color=FS.INK_2)
    dx.set_zorder(d2.get_zorder() + 1); dx.patch.set_visible(False)
    for r in e572:
        if r["gate_applied"]:
            dx.axvspan(r["k"] - 0.5, r["k"] + 0.5, color=FS.CATEGORICAL[7], alpha=0.10, lw=0)
    rv = next((r["k"] for r in e572 if r["revoked"]), None)
    if rv is not None:
        dx.axvline(rv - 0.5, color=FS.INK, ls=":", lw=1)
        dx.text(rv - 0.4, 0.26, T("latch attach 해제\n(not_following, 두 arm 공통)", "latch revokes attach\n(not_following, both arms)"), fontsize=7.2, color=FS.INK)
    kp = next((r["k"] for r in e572 if r["k"] > 20 and r["lift_C1aN"] < 0.02), None)
    if kp is not None:
        dx.annotate(T("기존 방식: 바구니에 놓음", "current serving: placed in basket"), (kp, 0.01), (kp - 3.2, 0.06), fontsize=7.2, color=FS.CATEGORICAL[0],
                    arrowprops=dict(arrowstyle="->", color=FS.CATEGORICAL[0], lw=0.8))
    dx.annotate(T("새 구성: 바구니 밖으로 낙하", "new config: dropped outside the basket"), (21, 0.002), (21.6, 0.10), fontsize=7.2, color=FS.CATEGORICAL[1],
                arrowprops=dict(arrowstyle="->", color=FS.CATEGORICAL[1], lw=0.8))
    dx.set_xlabel(T("서버 요청 k", "server request k")); dx.set_xlim(min(ks) - 0.5, max(ks) + 0.5)
    h1, l1 = dx.get_legend_handles_labels(); h2, l2 = d2.get_legend_handles_labels()
    dx.legend(h1 + h2 + [matplotlib.patches.Patch(color=FS.CATEGORICAL[7], alpha=0.25)], l1 + l2 + [T("경보가 target 예외를 보류한 요청", "alarm withheld the target exemption")],
              fontsize=7.2, frameon=False, loc="upper right")
    dx.set_title(T("d  ep572 (두 arm 중 결과가 다른 유일한 장면): k=17까지 두 arm 동일 → k=18 경보 보류와 동시에 처음 갈라짐 → 새 구성만 k=20→21에 사과 낙하",
                   "d  ep572 (only discordant scene): identical to k=17 → first divergence at k=18 with the alarm → apple drops k=20→21 in the new config only"),
                 fontsize=9, loc="left", color=FS.INK)
    fig.suptitle(T("새 구성 진단: 경보는 '쥔 동안 사과가 robot mask에 지워짐 → 기억이 다른 물체로 전환 → 놓은 뒤에도 고착', 3D 쥠 규칙은 attach 결정을 바꾸지 않음",
                   "new-config diagnosis: alarm = held apple erased by the robot mask → memory switches to another object → stays locked after release; 3D rule never changed an attach decision"),
                 fontsize=10.5, color=FS.INK, x=0.01, ha="left", y=0.995)
    p1 = a.fig / "ifg-apple-s2-cause.png"
    fig.savefig(p1, dpi=200, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    side = {k: N[k] for k in N if k != "ep572"}
    side["ep572_rows"] = e572
    side["sources"] = {"run": str(a.run), "s2fk": str(a.s2fk), "script": "AG3S/ag3s/experiments/studies/ifg_apple_s2_cause.py"}
    p1.with_suffix(".json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=float))

    # ------------------------------------------------------------------ figure 2: real scene (ep572, both arms)
    pick = [17, 18, 20, 22]
    fig, axes = plt.subplots(2, len(pick), figsize=(4.2 * len(pick), 6.6))
    sside = {"frames": []}
    with tempfile.TemporaryDirectory() as td:
        for i, (arm, lab) in enumerate((("C1aN", T("기존 방식", "current serving")), ("S2", T("새 구성 (+경보 +3D 쥠)", "new config (+alarm +3D held)")))):
            for jx, k in enumerate(pick):
                r = next(r for r in e572 if r["k"] == k)
                t_s = r["t"] / STEPS_PER_S
                im = video_frame(pathlib.Path(runs572[arm]) / "third_person_front.mp4", t_s, pathlib.Path(td))
                axx = axes[i, jx]; axx.axis("off")
                if im is not None:
                    h, w = im.shape[:2]
                    axx.imshow(im[int(0.12 * h):int(0.80 * h), int(0.0 * w):int(0.72 * w)])
                if jx == 0:
                    axx.text(-0.06, 0.5, lab, transform=axx.transAxes, rotation=90, ha="right", va="center", fontsize=9.5, color=FS.INK)
                ex = r["excl_C1aN"] if arm == "C1aN" else r["excl_S2"]
                lift = r["lift_C1aN"] if arm == "C1aN" else r["lift_S2"]
                st = r["state_C1aN"] if arm == "C1aN" else r["state_S2"]
                txt = f"k={k}  t={t_s:.1f}s\n" + T(f"latch {st} · target 예외 {'유지' if ex else '보류'} · 사과 {lift * 100:.0f} cm",
                                                   f"latch {st} · exemption {'kept' if ex else 'WITHHELD'} · apple {lift * 100:.0f} cm")
                axx.set_title(txt, fontsize=8.2, loc="left",
                              color=FS.CATEGORICAL[7] if (arm == "S2" and not ex) else FS.INK)
                sside["frames"].append({"arm": arm, "k": k, "t_row": r["t"], "video_t_s": t_s, "exemption_active": ex, "apple_lift_m": lift,
                                        "latch_state": st, "run": runs572[arm], "frame_found": im is not None})
    fig.suptitle(T("ep572 실제 장면 (정면 client 영상): 두 arm은 k=17까지 같다. k=18에 latch가 attach를 풀자 새 구성의 경보가 target 예외를 보류했고, 사과는 k=20→22 사이 새 구성에서만 떨어졌다",
                   "ep572 real scene (front client video): identical to k=17; at k=18 the latch drops the attach, the alarm withholds the exemption, and the apple falls only in the new config"),
                 fontsize=9.5, color=FS.INK, x=0.01, ha="left")
    fig.tight_layout()
    p2 = a.fig / "ifg-apple-s2-cause-scene.png"
    fig.savefig(p2, dpi=170, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p2.with_suffix(".json").write_text(json.dumps(sside, indent=1, ensure_ascii=False, default=float))
    print(p1, p2, a.out / "numbers.json")


if __name__ == "__main__":
    main()
