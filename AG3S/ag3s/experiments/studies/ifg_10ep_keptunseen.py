"""10 diagnostic scenes, four configurations: current serving / + target split (ATS) / + hand-relative latch reach (HSR, T46g) /
+ closing-kept geometry not counted as an observation (HSR, `--latch-reach-kept-unseen`). CPU only, read only.
Run IFG-10EP-HSR-KEPTUNSEEN; the first three configurations are the existing records of IFG-10EP-TARGETSPLIT-HANDREACH (same scenes, seeds).

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_10ep_keptunseen --run <IFG-APPLE run> --cl3 <IFG-CL3 dir> --fig AG3S/ag3s/docs/figures/ifg-split

Inputs: <kit>/agg4arm (copies: analysis, replay, runs/*.done of all four configurations; agg3.py pairs). Per run and server request
(`ifg_10ep_handreach` rules): grasp-latch state, attached, GT target lift (MuJoCo replay), TrajOpt TCP push (FK, teleop keyframe), HOLD.
Held-object rows (requests with an attached object): carried-centroid error = |reach_carry.centroid_carried_m - GT target centre| (the
centre the hand-relative reach and the attach carry use), coarse-ESDF voxels < 0 within R_HELD of the GT target centre (the held object
still in the field), worst-row link and whether its point is within R_HELD of the GT centre. Primary = locked contact-free success (agg3.py).
Panels b/c draw the carried-centre error and the occupied voxels only up to the last request with the apple >= 5 cm up (rows after the put-down are left out).
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
CFG = ("C1aN", "SPL", "SPLR", "SPLK")
LAB = {"C1aN": T("기존 방식", "current"), "SPL": T("+ target 분리", "+ target split"),
       "SPLR": T("+ 손 기준 쥠 거리", "+ hand-relative reach"),
       "SPLK": T("+ 닫는 동안 유지한 형상은 관측 아님", "+ closing-kept shape not an observation")}
STEPS_PER_S = 15.0
R_HELD = 0.05


def coarse_neg_near(c, p, r):
    o = c["esdf_coarse_origin"]; v = float(c["esdf_coarse_voxel_size"]); d = c["esdf_coarse_distance"]
    lo = np.maximum(np.floor((p - r - o) / v).astype(int), 0); hi = np.minimum(np.ceil((p + r - o) / v).astype(int) + 1, d.shape)
    ijk = np.argwhere(d[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] < 0) + lo
    return int((np.linalg.norm(o + (ijk + 0.5) * v - p, axis=1) <= r).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--cl3", type=pathlib.Path, required=True)
    ap.add_argument("--fig", type=pathlib.Path, required=True)
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    import mujoco
    from integration.rby1.bringup.avoidance_arrow import TcpKinematics, arm_targets
    v = a.run / "verify"; K0 = v / "IFG-10EP-TARGETSPLIT-HANDREACH"; K = v / "IFG-10EP-HSR-KEPTUNSEEN"; M = K / "agg4arm"
    m = mujoco.MjModel.from_xml_path(str(a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
    base_q = np.array(m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")]); kin = TcpKinematics(m)
    S = json.load(open(M / "seeds.json"))["items"]
    pairs = {p: json.load(open(M / f"agg_{p}_vs_SPLK.json")) for p in ("C1aN", "SPL", "SPLR")}
    arm_row = {ep: {} for ep in (it["episode"] for it in S)}
    for p, A in pairs.items():
        for r in A["rows"]:
            arm_row[r["episode"]].update(r["arms"])

    def run_dir(cfg, it):
        apple = it["source_set"] == "IFG-APPLE"
        base = {"C1aN": v / "IFG-APPLE" if apple else a.cl3, "SPL": v / ("IFG-APPLE-SPLIT" if apple else "IFG-CL3-SPLIT"),
                "SPLR": K0, "SPLK": K}[cfg]
        return next(p for p in (base / "runs" / cfg).glob(f"{cfg}_ep{it['episode']}_s*") if p.is_dir())

    def timeline(cfg, it):
        jf = glob.glob(str(M / "replay" / f"runs_{cfg}_{cfg}_ep{it['episode']}_*.json"))[0]; j = json.load(open(jf)); z = np.load(jf[:-5] + ".npz")
        al = j["server_chunk_alignment"]["block"]; chk = [c.get("dist_m") for c in j["server_chunk_alignment"].get("checks", []) if c.get("plan_index") == 0]
        if not al.get("found") or al.get("offset") is None or not chk or chk[0] is None or chk[0] > 0.05:
            return None
        side = j["side"]; pi = z["plan_index"]; rows = []
        for k in range(int(pi.max()) + 1):
            idx = np.flatnonzero(pi == k)
            if not len(idx):
                continue
            c = np.load(f"{al['chunk_dir']}/chunk_{k + al['offset']:05d}.npz", allow_pickle=True); s = json.loads(str(c["summary_json"]))
            g = s.get("grasp") or {}; ef = s.get("exec_feedback") or {}; ev = g.get("evidence") or {}
            gt = z["target_pos"][idx[0]]
            d = max(float(np.linalg.norm(kin.tcp(base_q, side, arm_targets("rby1_16d", y_)[side]) - kin.tcp(base_q, side, arm_targets("rby1_16d", x_)[side])))
                    for x_, y_ in zip(c["reference_chunk"][:8], c["refined_chunk"][:8]))
            row = {"k": k, "t": int(idx[0]), "lift": float(gt[2] - z["target_pos"][0, 2]), "state": g.get("state"),
                   "attached": g.get("attached") is not None, "revoked": bool(g.get("attach_revoked")),
                   "hold": int((ef.get("n_hold") or 0) > 0), "to_push_mm": d * 1000, "reach_source": ev.get("reach_source"),
                   "reach_mm": ev.get("reach_mm")}
            if row["attached"]:
                rc = (ev.get("reach_carry") or {})
                mv = ((s.get("to") or {}).get("metrics") or {}).get("max_violation_pair")
                mv = eval(mv) if isinstance(mv, str) else (mv or {})   # recorded as a Python repr
                p = mv.get("point_m")
                row.update(carried_err_mm=None if rc.get("centroid_carried_m") is None
                           else float(np.linalg.norm(np.asarray(rc["centroid_carried_m"]) - gt) * 1000),
                           carry_seen_frame=rc.get("seen_frame"), occupied_on_held=coarse_neg_near(c, gt, R_HELD),
                           worst_link=mv.get("link"), worst_clearance_mm=None if mv.get("clearance_m") is None else float(mv["clearance_m"]) * 1000,
                           worst_on_held=None if p is None else bool(np.linalg.norm(np.asarray(p) - gt) <= R_HELD))
            rows.append(row)
        return rows

    out = []
    for it in S:
        ep = it["episode"]; row = {"episode": ep, "source": it["source_set"], "target": it["target_fruit"], "cfg": {}}
        for cfg in CFG:
            x = arm_row[ep][cfg]; tl = timeline(cfg, it)
            att = [] if tl is None else [r for r in tl if r["attached"]]
            row["cfg"][cfg] = {
                "primary": bool(x["primary"]), "failure_mode": x["failure_mode"], "hold_chunks": x["hold_chunks"],
                "neighbour_rows": sum(x["contacts"]["neighbour_rows"].values()),
                "lifted_not_attached_chunks": None if tl is None else sum(r["lift"] >= 0.05 and not r["attached"] and r["state"] in ("closing", "latched") for r in tl),
                "first_attach": None if tl is None else next(({"k": r["k"], "lift_m": r["lift"]} for r in tl if r["attached"]), None),
                "attached_requests": None if tl is None else len(att),
                "revokes": None if tl is None else sum(r["revoked"] for r in tl),
                "carried_err_mm_first": next((r["carried_err_mm"] for r in att if r.get("carried_err_mm") is not None), None),
                "held_occupied_after_2": None if tl is None else sum(r["occupied_on_held"] > 0 for r in att[2:]),
                "worst_on_held_violating": None if tl is None else sum(bool(r["worst_on_held"]) and (r["worst_clearance_mm"] or 0) < 0 for r in att),
                "max_to_push_mm_closing_lift": None if tl is None else max([r["to_push_mm"] for r in tl if r["state"] in ("closing", "held")] or [0.0]),
                "timeline": tl}
        out.append(row)
    N = {cfg: int(sum(r["cfg"][cfg]["primary"] for r in out)) for cfg in CFG}
    for p, A in pairs.items():
        N[f"paired_{p}_vs_SPLK"] = A["paired_primary"]
    (M / "ifg_10ep_keptunseen_numbers.json").write_text(json.dumps({"numbers": N, "rows": out}, indent=1, default=float))
    by = {r["episode"]: r for r in out}

    # ---------------------------------------------------------------- figure 1: outcome table + 423 / 1404 timelines
    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.3], hspace=0.42, wspace=0.2)
    ax = fig.add_subplot(gs[:, 0]); FS.style_axes(fig, [ax])
    tag = {250: "ATS", 211: "ATS", 572: "ATS", 1345: "ATS", 1582: "ATS", 423: "HSR", 831: "HSR", 936: "HSR", 126: T("대조", "control"), 1404: T("대조", "control")}
    for i, r in enumerate(out):
        for jx, cfg in enumerate(CFG):
            x = r["cfg"][cfg]
            ax.scatter(jx, i, s=260, color=FS.CATEGORICAL[2] if x["primary"] else FS.CATEGORICAL[7], edgecolor=FS.INK_2, lw=0.5, zorder=3)
            ax.text(jx, i, "✓" if x["primary"] else "×", ha="center", va="center", fontsize=9, color=FS.SURFACE, zorder=4)
            g = x["lifted_not_attached_chunks"]
            if g:
                ax.text(jx + 0.2, i - 0.18, f"{g}", fontsize=7, color=FS.CATEGORICAL[1])
            if x["revokes"]:
                ax.text(jx + 0.2, i + 0.05, T(f"철회{x['revokes']}", f"rev{x['revokes']}"), fontsize=6.3, color=FS.CATEGORICAL[6])
            if not x["primary"]:
                ax.text(jx + 0.2, i + 0.3, x["failure_mode"][:13], fontsize=6.0, color=FS.INK_2)
    ax.set_xticks(range(4)); ax.set_xticklabels([T("+ 닫는 동안 유지한\n형상은 관측 아님", "+ closing-kept shape\nnot an observation") if c == "SPLK" else LAB[c].replace(" + ", "\n+ ") for c in CFG], fontsize=7.4)
    ax.set_yticks(range(len(out))); ax.set_yticklabels([f"{r['source'][4:]} ep{r['episode']} {r['target']}  [{tag[r['episode']]}]" for r in out], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(-0.5, 3.9)
    ax.set_title(T(f"a  진단용 10 장면 (괄호 = 고른 이유, 각 열은 왼쪽 열에 설정 하나를 더함)\n   성공 {N['C1aN']} → {N['SPL']} → {N['SPLR']} → {N['SPLK']} / 10 · 주황 = 들렸는데 쥠 미인정 chunk · 보라 = 쥠 철회",
                   f"a  10 diagnostic scenes (each column adds one setting)\n   success {N['C1aN']} → {N['SPL']} → {N['SPLR']} → {N['SPLK']} / 10 · orange = lifted-not-attached · purple = attach revoked"),
                 fontsize=9, loc="left", color=FS.INK)
    panels = ((423, "b", T("423 (HSR → LTE): 쥔 사과가 거리장에 남아 공중 HOLD → 해결", "423: held apple left in the field → HOLD in the air → fixed")),
              (1404, "c", T("1404 (대조): 손을 따라 옮기는 시작 시점이 접근 중으로 당겨져 쥠 철회 → 퇴보", "1404 (control): carry starts during the approach → attach revoked → regression")))
    for gi, (ep, letter, ttl) in enumerate(panels):
        bx = fig.add_subplot(gs[gi, 1]); FS.style_axes(fig, [bx]); b2 = bx.twinx()
        for cfg, col, ls in (("SPLR", FS.CATEGORICAL[0], "--"), ("SPLK", FS.CATEGORICAL[1], "-")):
            tl = by[ep]["cfg"][cfg]["timeline"]
            ks = [r["k"] for r in tl if r["state"] in ("closing", "held")]
            k0 = (max(k for k in ks if k <= min(r["k"] for r in tl if r["attached"])) if any(r["attached"] for r in tl) else min(ks)) if ks else 0
            sel = [r for r in tl if k0 - 3 <= r["k"] <= k0 + 14]
            bx.plot([r["k"] for r in sel], [r["lift"] * 100 for r in sel], color=col, ls=ls, lw=1.6, marker="o", ms=3,
                    label=T(f"사과 높이 · {LAB[cfg]}", f"apple height · {LAB[cfg]}"))
            for r in sel:
                if r["attached"]:
                    bx.scatter(r["k"], -3 if cfg == "SPLK" else -5, marker="s", s=14, color=col)
                if r["revoked"]:
                    bx.scatter(r["k"], -3 if cfg == "SPLK" else -5, marker="x", s=30, color=FS.CATEGORICAL[6])
                if r["hold"]:
                    bx.scatter(r["k"], 30 if cfg == "SPLK" else 28, marker="v", s=16, color=col)
            k_up = max([r["k"] for r in tl if r["lift"] >= 0.05] or [-1])   # last request with the apple up: drop rows after the put-down
            ys = [(r["k"], r["carried_err_mm"]) for r in sel if r.get("carried_err_mm") is not None and r["k"] <= k_up]
            if ys:
                b2.plot(*zip(*ys), color=col, ls=":", lw=1.3, marker="d", ms=3.5,
                        label=T(f"손이 옮긴 사과 중심 오차 · {LAB[cfg]}", f"carried-centre error · {LAB[cfg]}"))
            if ep == 423:
                occ = [(r["k"], r["occupied_on_held"] * 10) for r in sel if r["attached"] and r["k"] <= k_up]
                if occ:
                    b2.bar([q + (0.18 if cfg == "SPLK" else -0.18) for q, _ in occ], [o for _, o in occ], width=0.34, color=col, alpha=0.35)
        b2.set_ylim(0, 110); b2.set_ylabel(T("중심 오차 (mm) · 막대 = 쥔 사과 위 점유 voxel × 10", "centre error (mm) · bars = occupied voxels on held apple × 10") if ep == 423
                                            else T("중심 오차 (mm)", "centre error (mm)"), fontsize=7.2)
        bx.set_zorder(b2.get_zorder() + 1); bx.patch.set_visible(False)
        bx.set_ylim(-7, 33); bx.set_ylabel(T("사과 높이 (cm)", "apple height (cm)"), fontsize=8)
        x = by[ep]["cfg"]
        bx.set_title(f"{letter}  {ttl}\n   {LAB['SPLR']} {'✓' if x['SPLR']['primary'] else '×'} → {LAB['SPLK']} {'✓' if x['SPLK']['primary'] else '×'}"
                     + T("   (■ 쥠 인정, × 철회, ▼ HOLD)", "   (■ attached, × revoked, ▼ HOLD)"), fontsize=8.6, loc="left", color=FS.INK)
        h1, l1 = bx.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
        bx.legend(h1 + h2, l1 + l2, fontsize=6.6, frameon=False, loc="upper left", ncol=2)
        bx.set_xlabel(T("서버 요청 k", "server request k"), fontsize=8)
    fig.suptitle(T(f"닫는 동안 유지한 형상을 관측으로 세지 않기 (HSR): 성공 {N['SPLR']} → {N['SPLK']} / 10. 423 은 쥔 사과가 거리장에서 지워져 해결, "
                   "1404 는 손을 따라 옮기기 시작하는 시점이 너무 일러져 쥠 철회 (성공은 유지)",
                   f"closing-kept shape not an observation (HSR): {N['SPLR']} → {N['SPLK']} / 10; 423 fixed, 1404 attach revoked (still succeeds)"),
                 fontsize=10, x=0.01, ha="left", y=0.995)
    p1 = a.fig / "ifg-10ep-keptunseen.png"
    fig.savefig(p1, dpi=180, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    side = {"numbers": N, "rows": [{k: v_ for k, v_ in r.items() if k != "cfg"} | {"cfg": {c: {kk: vv for kk, vv in d.items() if kk != "timeline"} for c, d in r["cfg"].items()}} for r in out],
            "panels": {str(ep): {cfg: [{kk: r_.get(kk) for kk in ("k", "lift", "attached", "revoked", "hold", "carried_err_mm", "occupied_on_held", "to_push_mm")}
                                       for r_ in by[ep]["cfg"][cfg]["timeline"]] for cfg in ("SPLR", "SPLK")} for ep in (423, 1404)},
            "definitions": __doc__, "sources": {"kit": str(K), "agg": str(M), "script": "AG3S/ag3s/experiments/studies/ifg_10ep_keptunseen.py"}}
    p1.with_suffix(".json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=float))

    # ---------------------------------------------------------------- figure 2: real scenes 423 and 1404, SPLR vs SPLK at the same times
    def frame(rd, t, box, tmp):
        mp4 = rd / "third_person_front.mp4"
        if shutil.which("ffmpeg") is None or not mp4.exists():
            return None
        png = tmp / f"{rd.name}_{t}.png"
        r_ = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t / STEPS_PER_S:.3f}", "-i", str(mp4), "-frames:v", "1", str(png)], capture_output=True)
        if r_.returncode or not png.exists():
            return None
        x0, y0, x1, y1 = box
        return plt.imread(png)[y0:y1, x0:x1]
    bodies = {}
    for src in ("IFG-APPLE-FAIL/scene_bodies.json", "IFG-CL3-SPLIT/pre/agg_prep/scene_bodies.json"):
        bodies.update(json.load(open(v / src)))
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.4)); sside = {"frames": []}
    with tempfile.TemporaryDirectory() as td:
        for i, ep in enumerate((423, 1404)):
            it = next(x for x in S if x["episode"] == ep); y = bodies[str(ep)][it["target_fruit"]][1]
            box = (40, 15, 360, 305) if y < 0 else (280, 15, 600, 305)
            tl = by[ep]["cfg"]["SPLK"]["timeline"]
            ta = next(r["t"] for r in tl if r["attached"])
            times = (min(ta + 45, 590), min(ta + 105, 590))
            for jx, (cfg, t) in enumerate([(c_, t_) for c_ in ("SPLR", "SPLK") for t_ in times]):
                rd = run_dir(cfg, it); im = frame(rd, t, box, pathlib.Path(td)); axx = axes[i, jx]; axx.axis("off")
                if im is not None:
                    axx.imshow(im)
                axx.set_title(f"ep{ep} · {LAB[cfg]}  t={t / STEPS_PER_S:.1f}s", fontsize=8.2, loc="left")
                sside["frames"].append({"episode": ep, "cfg": cfg, "t_row": t, "crop_px": box, "frame_found": im is not None, "run": str(rd)})
    fig.suptitle(T("실제 장면 (정면 client 영상, 같은 시각): ep423 은 직전 구성에서 사과를 든 채 공중에서 멈췄다가 떨어뜨리고, 새 구성에서는 운반한다. "
                   "ep1404 는 새 구성에서 쥠이 철회된 채 들어 올린다",
                   "real scenes (same times): ep423 stops in the air and drops before, carries now; ep1404 lifts with the attach revoked in the new configuration"),
                 fontsize=9.6, x=0.01, ha="left")
    fig.tight_layout()
    p2 = a.fig / "ifg-10ep-keptunseen-scenes.png"
    fig.savefig(p2, dpi=165, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p2.with_suffix(".json").write_text(json.dumps(sside, indent=1))
    print(p1, p2)


if __name__ == "__main__":
    main()
