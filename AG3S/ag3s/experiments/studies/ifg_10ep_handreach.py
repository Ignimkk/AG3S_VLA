"""10 diagnostic scenes: current serving / + target split (ATS) / + target split + hand-relative latch reach (HSR, T46g).
CPU only, read only. Scenes and configurations as fixed in IFG-SPLIT.task.md / LEAD_AMENDMENTS (IFG-10EP-TARGETSPLIT-HANDREACH).

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_10ep_handreach --run <IFG-APPLE run> --cl3 <IFG-CL3 dir> --fig AG3S/ag3s/docs/figures/ifg-split

Per run and server request: grasp-latch state (attached or not), GT target height (MuJoCo replay), the largest TCP displacement the
trajectory optimiser applied to the executed rows (FK, teleop keyframe), HOLD. "Lifted, not attached" = target >= 0.05 m up while the latch
is closing / latched. Primary = locked contact-free success (agg3.py over copies in <kit>/agg3arm).
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
CFG = ("C1aN", "SPL", "SPLR")
LAB = {"C1aN": T("기존 방식", "current"), "SPL": T("+ target 분리 (ATS)", "+ target split (ATS)"),
       "SPLR": T("+ target 분리 + 손 기준 reach (HSR)", "+ target split + hand-relative reach (HSR)")}
STEPS_PER_S = 15.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--cl3", type=pathlib.Path, required=True)
    ap.add_argument("--fig", type=pathlib.Path, required=True)
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    import mujoco
    from integration.rby1.bringup.avoidance_arrow import TcpKinematics, arm_targets
    v = a.run / "verify"; K = v / "IFG-10EP-TARGETSPLIT-HANDREACH"; M = K / "agg3arm"
    m = mujoco.MjModel.from_xml_path(str(a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"))
    base_q = np.array(m.key_qpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")]); kin = TcpKinematics(m)
    S = json.load(open(M / "seeds.json"))["items"]
    R = {r["episode"]: r for r in json.load(open(M / "agg_C1aN_vs_SPLR.json"))["rows"]}
    R2 = {r["episode"]: r for r in json.load(open(M / "agg_SPL_vs_SPLR.json"))["rows"]}

    def base_of(cfg, it):
        apple = it["source_set"] == "IFG-APPLE"
        return {"C1aN": v / "IFG-APPLE" if apple else a.cl3, "SPL": v / ("IFG-APPLE-SPLIT" if apple else "IFG-CL3-SPLIT"), "SPLR": K}[cfg]

    def timeline(cfg, it):
        b = base_of(cfg, it)
        jf = glob.glob(str(b / "replay" / f"runs_{cfg}_{cfg}_ep{it['episode']}_*.json"))[0]; j = json.load(open(jf)); z = np.load(jf[:-5] + ".npz")
        al = j["server_chunk_alignment"]["block"]; chk = [c.get("dist_m") for c in j["server_chunk_alignment"].get("checks", []) if c.get("plan_index") == 0]
        if not al.get("found") or al.get("offset") is None or not chk or chk[0] is None or chk[0] > 0.05:
            return None
        side = j["side"]; pi = z["plan_index"]; lift = z["target_pos"][:, 2] - z["target_pos"][0, 2]; rows = []
        for k in range(int(pi.max()) + 1):
            idx = np.flatnonzero(pi == k)
            if not len(idx):
                continue
            c = np.load(f"{al['chunk_dir']}/chunk_{k + al['offset']:05d}.npz", allow_pickle=True); s = json.loads(str(c["summary_json"]))
            g = s.get("grasp") or {}; ef = s.get("exec_feedback") or {}
            d = max(float(np.linalg.norm(kin.tcp(base_q, side, arm_targets("rby1_16d", y_)[side]) - kin.tcp(base_q, side, arm_targets("rby1_16d", x_)[side])))
                    for x_, y_ in zip(c["reference_chunk"][:8], c["refined_chunk"][:8]))
            rows.append({"k": k, "t": int(idx[0]), "lift": float(lift[idx[0]]), "state": g.get("state"), "attached": g.get("attached") is not None,
                         "hold": int((ef.get("n_hold") or 0) > 0), "to_push_mm": d * 1000,
                         "reach_source": (g.get("evidence") or {}).get("reach_source")})
        return rows
    out = []
    for it in S:
        ep = it["episode"]; row = {"episode": ep, "source": it["source_set"], "target": it["target_fruit"], "cfg": {}}
        for cfg in CFG:
            x = (R[ep]["arms"].get(cfg) or R2[ep]["arms"].get(cfg))
            tl = timeline(cfg, it)
            gap = None if tl is None else sum(r["lift"] >= 0.05 and not r["attached"] and r["state"] in ("closing", "latched") for r in tl)
            first_att = None if tl is None else next(({"k": r["k"], "lift_m": r["lift"]} for r in tl if r["attached"]), None)
            row["cfg"][cfg] = {"primary": bool(x["primary"]), "failure_mode": x["failure_mode"], "hold_chunks": x["hold_chunks"],
                               "neighbour_rows": sum(x["contacts"]["neighbour_rows"].values()), "lifted_not_attached_chunks": gap,
                               "first_attach": first_att, "timeline": tl}
        out.append(row)
    N = {cfg: int(sum(r["cfg"][cfg]["primary"] for r in out)) for cfg in CFG}
    N["paired_current_vs_split_handreach"] = json.load(open(M / "agg_C1aN_vs_SPLR.json"))["paired_primary"]
    N["paired_split_vs_split_handreach"] = json.load(open(M / "agg_SPL_vs_SPLR.json"))["paired_primary"]
    (M / "ifg_10ep_numbers.json").write_text(json.dumps({"numbers": N, "rows": out}, indent=1, default=float))

    # ---------------------------------------------------------------- figure 1: outcome table + three timelines
    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.35], hspace=0.55, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0]); FS.style_axes(fig, [ax])
    tag = {250: "ATS", 211: "ATS", 572: "ATS", 1345: "ATS", 1582: "ATS", 423: "HSR", 831: "HSR", 936: "HSR", 126: T("대조", "control"), 1404: T("대조", "control")}
    for i, r in enumerate(out):
        for jx, cfg in enumerate(CFG):
            x = r["cfg"][cfg]
            ax.scatter(jx, i, s=260, color=FS.CATEGORICAL[2] if x["primary"] else FS.CATEGORICAL[7], edgecolor=FS.INK_2, lw=0.5, zorder=3)
            ax.text(jx, i, "✓" if x["primary"] else "×", ha="center", va="center", fontsize=9, color=FS.SURFACE, zorder=4)
            g = x["lifted_not_attached_chunks"]
            if g:
                ax.text(jx + 0.22, i - 0.18, f"{g}", fontsize=7, color=FS.CATEGORICAL[1])
            if not x["primary"]:
                ax.text(jx + 0.22, i + 0.25, x["failure_mode"][:13], fontsize=6.3, color=FS.INK_2)
    ax.set_xticks(range(3)); ax.set_xticklabels([LAB[c].replace(" + ", "\n+ ") for c in CFG], fontsize=7.8)
    ax.set_yticks(range(len(out))); ax.set_yticklabels([f"{r['source'][4:]} ep{r['episode']} {r['target']}  [{tag[r['episode']]}]" for r in out], fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(-0.5, 2.9)
    ax.set_title(T(f"a  진단용 10 장면 (괄호 = 장면을 고른 이유)\n   성공 {N['C1aN']} → {N['SPL']} → {N['SPLR']} / 10 · 주황 숫자 = 들렸는데 쥠 인정 안 된 chunk 수",
                   f"a  10 diagnostic scenes [reason picked]\n   success {N['C1aN']} → {N['SPL']} → {N['SPLR']} / 10 · orange = lifted-not-attached chunks"),
                 fontsize=9, loc="left", color=FS.INK)
    by = {r["episode"]: r for r in out}
    for gi, ep in enumerate((831, 936, 423)):
        bx = fig.add_subplot(gs[gi, 1]); FS.style_axes(fig, [bx])
        for cfg, col, ls in (("SPL", FS.CATEGORICAL[0], "--"), ("SPLR", FS.CATEGORICAL[1], "-")):
            tl = by[ep]["cfg"][cfg]["timeline"]
            k0 = min(r["k"] for r in tl if r["state"] in ("closing", "held")) if any(r["state"] in ("closing", "held") for r in tl) else 0
            sel = [r for r in tl if k0 - 2 <= r["k"] <= k0 + 12]
            bx.plot([r["k"] for r in sel], [r["lift"] * 100 for r in sel], color=col, ls=ls, lw=1.6, marker="o", ms=3,
                    label=T(f"물체 높이 · {LAB[cfg]}", f"object height · {LAB[cfg]}"))
            for r in sel:
                if r["attached"]:
                    bx.scatter(r["k"], -3 if cfg == "SPLR" else -5, marker="s", s=14, color=col)
                if r["hold"]:
                    bx.scatter(r["k"], 30 if cfg == "SPLR" else 28, marker="v", s=16, color=col)
            if cfg == "SPLR":
                b2 = bx.twinx(); b2.bar([r["k"] for r in sel], [r["to_push_mm"] for r in sel], color=FS.GRID_INK, width=0.55, alpha=0.9)
                b2.set_ylim(0, 260); b2.set_ylabel(T("TrajOpt 손끝 밀기 (mm)", "TO TCP push (mm)"), fontsize=7.5)
                bx.set_zorder(b2.get_zorder() + 1); bx.patch.set_visible(False)
        x = by[ep]["cfg"]
        bx.set_ylim(-7, 33); bx.set_ylabel(T("높이 (cm)", "height (cm)"), fontsize=8)
        bx.set_title(T(f"ep{ep}: target 분리만 {'✓' if x['SPL']['primary'] else '×'} → + 손 기준 reach {'✓' if x['SPLR']['primary'] else '×'}  (■ 쥠 인정, ▼ HOLD, 회색 막대 = 손 기준 reach 구성의 TrajOpt 밀기)",
                       f"ep{ep}: split only {'✓' if x['SPL']['primary'] else '×'} → + hand reach {'✓' if x['SPLR']['primary'] else '×'}  (■ attached, ▼ HOLD, grey = TO push, hand-reach config)"),
                     fontsize=8.6, loc="left", color=FS.INK)
        if gi == 0:
            bx.legend(fontsize=7, frameon=False, loc="upper left")
        if gi == 2:
            bx.set_xlabel(T("서버 요청 k", "server request k"))
    fig.suptitle(T("손 기준 reach (HSR) 진단 실행: 쥠 인정은 회복 (423 · 831 · 936), 831 성공. 남은 문제 = 들린 첫 chunk 의 쥠 지연 동안 손에 든 물체가 장애물 (LTE) → TrajOpt 밀기 (CCR)",
                   "hand-relative reach (HSR) run: attach recovered (423 · 831 · 936), 831 succeeds; left: during the one-chunk delay the held object is an obstacle (LTE) → TO push (CCR)"),
                 fontsize=10.2, x=0.01, ha="left", y=0.995)
    p1 = a.fig / "ifg-10ep-handreach.png"
    fig.savefig(p1, dpi=185, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    side = {"numbers": N, "rows": [{k: v_ for k, v_ in r.items() if k != "cfg"} | {"cfg": {c: {kk: vv for kk, vv in d.items() if kk != "timeline"} for c, d in r["cfg"].items()}} for r in out],
            "sources": {"kit": str(K), "script": "AG3S/ag3s/experiments/studies/ifg_10ep_handreach.py"}}
    p1.with_suffix(".json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=float))

    # ---------------------------------------------------------------- figure 2: real scenes 831 (fixed) and 423 (HOLD in the air)
    def frame(run_dir, t, box, tmp):
        mp4 = run_dir / "third_person_front.mp4"
        if shutil.which("ffmpeg") is None or not mp4.exists():
            return None
        png = tmp / f"{run_dir.name}_{t}.png"
        r_ = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t / STEPS_PER_S:.3f}", "-i", str(mp4), "-frames:v", "1", str(png)], capture_output=True)
        if r_.returncode or not png.exists():
            return None
        x0, y0, x1, y1 = box
        return plt.imread(png)[y0:y1, x0:x1]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.4)); sside = {"frames": []}
    bodies = {}
    for src in ("IFG-APPLE-FAIL/scene_bodies.json", "IFG-CL3-SPLIT/pre/agg_prep/scene_bodies.json"):
        bodies.update(json.load(open(v / src)))
    with tempfile.TemporaryDirectory() as td:
        for i, ep in enumerate((831, 423)):
            it = next(x for x in S if x["episode"] == ep); y = bodies[str(ep)][it["target_fruit"]][1]
            box = (40, 15, 360, 305) if y < 0 else (280, 15, 600, 305)
            tl_s, tl_r = by[ep]["cfg"]["SPL"]["timeline"], by[ep]["cfg"]["SPLR"]["timeline"]
            ta = next(r["t"] for r in tl_r if r["attached"])
            shots = [("SPL", ta, T("target 분리만: 같은 시각 (쥠 미인정)", "split only: same time (not attached)")),
                     ("SPL", min(ta + 90, 590), T("target 분리만: 6 초 뒤", "split only: 6 s later")),
                     ("SPLR", ta, T("+ 손 기준 reach: 쥠 인정 순간", "+ hand reach: attached")),
                     ("SPLR", min(ta + 90, 590), T("+ 손 기준 reach: 6 초 뒤", "+ hand reach: 6 s later"))]
            for jx, (cfg, t, lab) in enumerate(shots):
                rd = next(p for p in (base_of(cfg, it) / "runs" / cfg).glob(f"{cfg}_ep{ep}_s*") if p.is_dir())
                im = frame(rd, t, box, pathlib.Path(td)); axx = axes[i, jx]; axx.axis("off")
                if im is not None:
                    axx.imshow(im)
                axx.set_title(f"ep{ep} · {lab}  t={t / STEPS_PER_S:.1f}s", fontsize=8.3, loc="left")
                sside["frames"].append({"episode": ep, "cfg": cfg, "t_row": t, "crop_px": box, "frame_found": im is not None})
    fig.suptitle(T("실제 장면 (정면 client 영상): ep831 은 손 기준 reach 로 쥠이 인정되어 성공, ep423 은 늦게 인정된 뒤 손에 든 사과가 거리장에 남아 공중에서 멈춤",
                   "real scenes: ep831 attached with the hand reach and succeeds; ep423 attached late, the held apple stays in the field, HOLD in the air"),
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    p2 = a.fig / "ifg-10ep-handreach-scenes.png"
    fig.savefig(p2, dpi=165, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    p2.with_suffix(".json").write_text(json.dumps(sside, indent=1))
    print(p1, p2)


if __name__ == "__main__":
    main()
