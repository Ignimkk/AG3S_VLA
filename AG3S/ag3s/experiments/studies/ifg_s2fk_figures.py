"""IFG-S2FK figures (rule A): graph, table and real-scene views of the held-out re-measurement without the render mask.

  MUJOCO_GL=osmesa PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_s2fk_figures \
      --s2fk <run>/s2fk --out AG3S/ag3s/docs/figures/ifg-s2fk

Figures (+ same-name .json sidecar with every number drawn):
  ifg-s2fk-memory-gate   episode-mean IoU without / with memory switching and neighbour FP, render vs FK robot mask;
                         confidence gate at the locked threshold (wrong choices caught, false alarms, AUROC)
  ifg-s2fk-phase-held    phase / held accuracy of serving rules, the task-fixed no-learning rules and the locked learned model,
                         render vs FK; paired episode differences against the serving rule
  ifg-s2fk-scenes        two real held-out frames (FK condition): head-camera image + side view of the back-projected depth
                         with the selected target region, both hand proxies and the D_hold radius; GT state and every decision
"""
import argparse, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_stage2_fk as S2  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)
COND = {"render": T("렌더 robot mask (보류)", "render robot mask (on hold)"), "fk": T("FK 구 robot mask (serving)", "FK-sphere robot mask (serving)")}
CC = {"render": FS.GRID_INK, "fk": FS.CATEGORICAL[0]}


def save(fig, out, name, side):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    (out / f"{name}.json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=float))
    print(out / f"{name}.png")


def fig_memory_gate(s2, out):
    fk = json.loads((s2 / "summary_fk.json").read_text())
    rd = json.loads((S2.LOCKED / "summary.json").read_text())
    src = {"render": rd, "fk": fk}
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1.1, 1, 1.3]})
    FS.style_axes(fig, axes)
    ax, bx, cx = axes
    side = {"source": {"render": str(S2.LOCKED / "summary.json"), "fk": str(s2 / "summary_fk.json")}, "conditions": {}}
    x = np.arange(2); w = 0.36
    for i, c in enumerate(("render", "fk")):
        m = src[c]["main_success"]
        iou = [m["A"]["iou"]["episode_mean"], m["B"]["iou"]["episode_mean"]]
        nfp = [m["A"]["neighbor_fp_sum"], m["B"]["neighbor_fp_sum"]]
        p = m["paired_iou"]["B - A"]
        ax.bar(x + (i - 0.5) * w, iou, w, color=CC[c], edgecolor=FS.SURFACE, label=COND[c])
        for j, v in enumerate(iou):
            ax.text(x[j] + (i - 0.5) * w, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5, color=FS.INK_2)
        bx.bar(x + (i - 0.5) * w, np.array(nfp) / 1000, w, color=CC[c], edgecolor=FS.SURFACE)
        for j, v in enumerate(nfp):
            bx.text(x[j] + (i - 0.5) * w, v / 1000 + 1.5, f"{v / 1000:.1f}k", ha="center", fontsize=7.5, color=FS.INK_2)
        g = src[c]["gate_success"]
        side["conditions"][c] = {"iou_A": iou[0], "iou_B": iou[1], "paired_B_minus_A": p, "neighbor_fp_A": nfp[0], "neighbor_fp_B": nfp[1],
                                 "switch_frames": m.get("B_switch_frames"), "gate": g}
        ax.text(0.5 + (i - 0.5) * 0.9, 0.79 - 0.0, "", fontsize=1)
    ax.set_ylim(0.75, 0.92); ax.set_xticks(x)
    ax.set_xticklabels([T("전환 없음 (A)", "no switch (A)"), T("기억 전환 (B)", "memory switch (B)")], fontsize=9, color=FS.INK)
    ax.set_title(T("target 표면 IoU (held-out 34 episode)", "target surface IoU (held-out, 34 episodes)"), fontsize=10, loc="left", color=FS.INK)
    lines = [f"{COND[c].split(' (')[0]}: B−A {side['conditions'][c]['paired_B_minus_A']['diff']:+.3f} "
             f"[{side['conditions'][c]['paired_B_minus_A']['ci95'][0]:+.3f}, {side['conditions'][c]['paired_B_minus_A']['ci95'][1]:+.3f}]" for c in ("render", "fk")]
    ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=7.5, color=FS.INK_2)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right"); ax.grid(axis="y", color=FS.GRID_INK, lw=0.5); ax.set_axisbelow(True)
    bx.set_xticks(x); bx.set_xticklabels([T("전환 없음 (A)", "no switch (A)"), T("기억 전환 (B)", "memory switch (B)")], fontsize=9, color=FS.INK)
    bx.set_title(T("이웃 물체 오포함 voxel 합 (천)", "neighbour false-positive voxels (thousands)"), fontsize=10, loc="left", color=FS.INK)
    bx.grid(axis="y", color=FS.GRID_INK, lw=0.5); bx.set_axisbelow(True)
    keys = (("wrong_choices", T("잘못 고른 frame", "wrong choices")), ("alerts_on_wrong", T("그중 경보", "caught by alarm")), ("false_alarms", T("오경보", "false alarms")))
    y = np.arange(len(keys))[::-1]
    for i, c in enumerate(("render", "fk")):
        g = src[c]["gate_success"]
        vals = [g[k] for k, _ in keys]
        cx.barh(y + (0.5 - i) * 0.36, vals, 0.34, color=CC[c], edgecolor=FS.SURFACE)
        for yy, v in zip(y + (0.5 - i) * 0.36, vals):
            cx.text(v + 2, yy, str(v), va="center", fontsize=7.5, color=FS.INK_2)
    cx.set_yticks(y); cx.set_yticklabels([lab for _, lab in keys], fontsize=9, color=FS.INK)
    gt = [f"{COND[c].split(' (')[0]}: AUROC {src[c]['gate_success']['auroc_wrong']:.3f}, FP {T('제거', 'removed')} "
          f"{src[c]['gate_success']['fp_removed_ratio']:.0%}, TP {T('손실', 'lost')} {src[c]['gate_success']['tp_lost_ratio']:.1%}" for c in ("render", "fk")]
    cx.set_title(T("안전 경보 (잠근 기준값 0.590, 다시 정하지 않음)", "confidence alarm (locked threshold 0.590, not re-tuned)"), fontsize=10, loc="left", color=FS.INK)
    cx.text(0.02, -0.28, "\n".join(gt), transform=cx.transAxes, fontsize=7.5, color=FS.INK_2, va="top")
    cx.grid(axis="x", color=FS.GRID_INK, lw=0.5); cx.set_axisbelow(True)
    fig.suptitle(T("기억 feature 영역 전환과 안전 경보: 렌더 mask 대 FK 구 mask (held-out offline, 성공 rollout)",
                   "memory switching and confidence alarm: render vs FK robot mask (held-out offline, successful rollouts)"),
                 fontsize=10, color=FS.INK, x=0.01, ha="left", y=1.04)
    save(fig, out, "ifg-s2fk-memory-gate", side)


def fig_phase(s2, out):
    d = json.loads((s2 / "phase_summary.json").read_text())
    methods = {"phase": ["HSR argmax (serving)", "rule phase_rule (no learning)", "LR locked (dev render weights)"],
               "held": ["grip<0.9 (serving)", "rule held_rule (no learning)", "LR locked (dev render weights)"]}
    lab = {"HSR argmax (serving)": T("HSR (현재 serving)", "HSR (serving)"), "grip<0.9 (serving)": T("gripper < 0.9 (현재 serving)", "gripper < 0.9 (serving)"),
           "rule phase_rule (no learning)": T("3D 규칙 (학습 없음)", "3D rule (no learning)"), "rule held_rule (no learning)": T("3D 규칙 (학습 없음)", "3D rule (no learning)"),
           "LR locked (dev render weights)": T("학습형 (dev weight 고정)", "learned (dev weights locked)")}
    col = {0: FS.INK_2, 1: FS.CATEGORICAL[2], 2: FS.CATEGORICAL[6]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    FS.style_axes(fig, axes)
    side = {"definitions": {"D_hold_m": d["D_hold_m"], "grip_closed": d["grip_closed"]}, "blocks": {}}
    for ax, task in zip(axes, ("phase", "held")):
        x = np.arange(2)
        for j, mth in enumerate(methods[task]):
            vals = [d[f"{c}|success"][task][mth]["accuracy"] for c in ("render", "fk")]
            ax.bar(x + (j - 1) * 0.26, vals, 0.24, color=col[j], edgecolor=FS.SURFACE, label=lab[mth])
            for k, v in enumerate(vals):
                ax.text(x[k] + (j - 1) * 0.26, v + 0.003, f"{v:.3f}", ha="center", fontsize=7, color=FS.INK_2)
        ax.set_ylim(0.85, 1.0); ax.set_xticks(x); ax.set_xticklabels([COND["render"], COND["fk"]], fontsize=8.5, color=FS.INK)
        ttl = T("작업 단계 판단 (pick / place / home) 정확도", "phase (pick / place / home) accuracy") if task == "phase" else T("파지 판단 (쥠 / 아님) 정확도", "held judgment accuracy")
        ax.set_title(ttl, fontsize=10, loc="left", color=FS.INK)
        pr = d["fk|success"]["paired"]
        ref = methods[task][0]
        txt = [f"FK: {lab[m]} − {lab[ref]} {pr[f'{task}: {m} - {ref}']['diff']:+.3f} [{pr[f'{task}: {m} - {ref}']['ci95'][0]:+.3f}, "
               f"{pr[f'{task}: {m} - {ref}']['ci95'][1]:+.3f}]" for m in methods[task][1:]]
        ax.text(0.02, 0.995, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=7.3, color=FS.INK_2,
                bbox=dict(facecolor=FS.SURFACE, edgecolor="none", pad=1))
        ax.set_ylim(0.85, 1.02)
        ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
        ax.grid(axis="y", color=FS.GRID_INK, lw=0.5); ax.set_axisbelow(True)
    for c in ("render", "fk"):
        for sub in ("success", "all"):
            b = d[f"{c}|{sub}"]
            side["blocks"][f"{c}|{sub}"] = {"n_frames": b["n_frames"], "n_episodes": b["n_episodes"], "region_missing_frames": b["region_missing_frames"],
                                            "class_counts": b["class_counts"],
                                            "accuracy": {t: {m: b[t][m]["accuracy"] for m in b[t]} for t in ("phase", "held")},
                                            "macro_f1": {t: {m: b[t][m]["macro_f1"] for m in b[t]} for t in ("phase", "held")},
                                            "recall": {t: {m: b[t][m]["recall"] for m in b[t]} for t in ("phase", "held")}, "paired": b["paired"]}
    fig.suptitle(T("3D 관계로 파지 · 단계 판단: 현재 serving 규칙, 학습 없는 3D 규칙, 학습형 (held-out offline, 성공 rollout 2,528 frame)",
                   "held / phase from 3D relations: serving rules, no-learning 3D rules, learned model (held-out offline, 2,528 frames)")
                 + "\n" + T("3D 쥠 규칙: gripper < 0.9 그리고 target 중심이 손 대리점에서 0.131 m 안 (0.081 m 손–파지점 + 0.05 m 과일 반지름, 결과 전 고정)",
                            "3D held rule: gripper < 0.9 and target centre within 0.131 m of the hand proxy (0.081 m proxy-to-grasp point + 0.05 m fruit radius, fixed before results)"),
                 fontsize=9.5, color=FS.INK, x=0.01, ha="left", y=1.06)
    save(fig, out, "ifg-s2fk-phase-held", side)


def pick_frames(s2):
    rows = [json.loads(l) for l in open(s2 / "features_fk.jsonl")]
    rows = [r for r in rows if r["kind"] == "success"]
    rows.sort(key=lambda r: (r["episode_id"], r["frame"]))
    by = {}
    for r in rows:
        by.setdefault(r["episode_id"], []).append(r)
    good = bad = None
    for e, rs in sorted(by.items()):
        h, p = S2.rules(rs)
        for r, hh, pp in zip(rs, h, p):
            fin = np.isfinite(r["hand_d"])
            if good is None and r["grasp_state"] != "held" and r["grip"] < S2.GRIP_CLOSED and fin and r["hand_d"] > S2.D_HOLD and hh == 0:
                good = (r, hh, pp)
            if bad is None and r["grasp_state"] == "pre" and pp == 1:
                bad = (r, hh, pp)
        if good and bad:
            break
    return [("correct", good), ("wrong", bad)]


def fig_scenes(s2, out):
    import mujoco
    from AG3S.ag3s.experiments.studies import ifg_graph as G
    from AG3S.ag3s.experiments.studies import ifg_memory as M
    from AG3S.ag3s.experiments.studies.t47_replay import GRIPPER_ROOT_LINKS
    S2.HE.install()
    refs, _ = S2.HE.heldout_refs(S2.DUMP, S2.PREP)
    rmap = {r.frame_id: r for r in refs}
    m = mujoco.MjModel.from_xml_path(str(S2.SNAP_XML)); dd = mujoco.MjData(m)
    roots = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in GRIPPER_ROOT_LINKS}
    below = {n: [] for n in roots}
    for b in range(m.nbody):
        j = b
        while j > 0:
            j = int(m.body_parentid[j])
            for n, r in roots.items():
                if j == r:
                    below[n].append(b)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    cases = pick_frames(s2)
    fig, axes = plt.subplots(len(cases), 2, figsize=(12, 4.6 * len(cases)), gridspec_kw={"width_ratios": [1, 1.25]})
    FS.style_axes(fig, axes)
    side = {"frames": []}
    for i, (kind, (row, hh, pp)) in enumerate(cases):
        ref = rmap[row["frame_id"]]
        with np.load(S2.PREP / "replay" / f"{ref.frame_id}.npz") as z:
            depth, K, Tm, qpos = z["depth"], z["K"], z["T_base_cam"], z["qpos"]
        with np.load(s2 / "robot_mask_fk" / f"{ref.frame_id}.npz") as z:
            rmask = z["robot_mask"]
        with np.load(S2.PREP / "labels" / f"{ref.frame_id}.npz") as z:
            class_map, classes = z["class_map"], [str(c) for c in z["classes"]]
        dd.qpos[:] = qpos; mujoco.mj_kinematics(m, dd)
        Rb, tb = dd.xmat[bid].reshape(3, 3), dd.xpos[bid]
        tcp = [Rb.T @ (dd.xpos[below[n]].mean(0) - tb) for n in GRIPPER_ROOT_LINKS]
        fin = S2.R.load_frame(ref, features=())
        cfg = G.GraphConfig()
        geo = G.build_geometry(depth, K, Tm, rmask, class_map, classes, cfg)
        c = M.r6_choice(geo, cfg, fin.maps["N"])
        ch = c["chosen"] if c is not None else np.zeros(len(geo.pos), bool)
        img = fin.images[0]; img = np.clip((img + 1) / 2, 0, 1)
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        ax = axes[i, 0]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(FS.CATEGORICAL[2] if kind == "correct" else FS.CATEGORICAL[7]); sp.set_linewidth(2)
        gt = row["grasp_state"]
        serv_h = int(row["grip"] < S2.GRIP_CLOSED); hsr_p = ("pick", "place", "home")[int(np.argmax(row["hsr"]))]
        ax.set_title(f"{ref.frame_id}  ·  head camera\n" + T("정답 상태: ", "GT state: ") + gt
                     + f"  ·  gripper {row['grip']:.2f}  ·  hand_d {row['hand_d']:.3f} m", fontsize=8.5, loc="left", color=FS.INK)
        bx = axes[i, 1]
        P = geo.pos
        sub = np.random.default_rng(0).choice(len(P), size=min(len(P), 25000), replace=False)
        bx.scatter(P[sub, 0], P[sub, 2], s=0.3, c=FS.GRID_INK, linewidths=0, rasterized=True)
        bx.scatter(P[ch, 0], P[ch, 2], s=2.5, c=FS.CATEGORICAL[1], linewidths=0, label=T("선택된 target 영역", "selected target region"))
        cen = P[ch].mean(0) if ch.any() else None
        for k, t in enumerate(tcp):
            bx.scatter([t[0]], [t[2]], s=40, marker="x", c=FS.CATEGORICAL[0], label=T("손 대리점", "hand proxy") if k == 0 else None)
            bx.add_patch(plt.Circle((t[0], t[2]), S2.D_HOLD, fill=False, ls="--", lw=0.8, color=FS.CATEGORICAL[0]))
        if cen is not None:
            near = min(tcp, key=lambda t: np.linalg.norm(cen - t))
            bx.plot([cen[0], near[0]], [cen[2], near[2]], color=FS.INK, lw=1)
            bx.scatter([cen[0]], [cen[2]], s=30, c=FS.INK, label=T("영역 중심", "region centre"))
        bx.set_aspect("equal"); bx.set_xlabel(T("x (base, m)", "x (base, m)")); bx.set_ylabel(T("z (base, m)", "z (base, m)"))
        cz = cen if cen is not None else np.mean(tcp, 0)
        bx.set_xlim(cz[0] - 0.35, cz[0] + 0.35); bx.set_ylim(cz[2] - 0.2, cz[2] + 0.3)
        dec = (T("결정 — serving gripper 규칙: ", "decisions — serving gripper rule: ") + ("held" if serv_h else "not held")
               + T(" · 3D 쥠 규칙: ", " · 3D held rule: ") + ("held" if hh else "not held")
               + T(" · HSR 단계: ", " · HSR phase: ") + hsr_p + T(" · 3D 단계 규칙: ", " · 3D phase rule: ") + ("pick", "place", "home")[pp])
        bx.set_title(T("옆에서 본 3D (깊이 역투영, 점선 원 = 0.131 m)", "side view of back-projected depth (dashed circle = 0.131 m)") + "\n" + dec, fontsize=8, loc="left", color=FS.INK)
        bx.legend(fontsize=7, frameon=False, loc="upper right"); bx.grid(color=FS.GRID_INK, lw=0.4)
        side["frames"].append({"case": kind, "frame_id": ref.frame_id, "gt_state": gt, "grip": row["grip"], "hand_d_m": row["hand_d"], "height_m": row["height"],
                               "serving_held": serv_h, "rule_held": hh, "hsr_phase": hsr_p, "rule_phase": ("pick", "place", "home")[pp],
                               "region_nodes": int(ch.sum()), "tcp_base_m": [t.tolist() for t in tcp], "region_centre_base_m": None if cen is None else cen.tolist()})
    fig.suptitle(T("실제 held-out 장면 (FK 구 mask). 위: 3D 쥠 규칙이 맞고 gripper 규칙이 틀린 frame. 아래: 3D 단계 규칙이 틀린 frame (닫히는 중)",
                   "real held-out frames (FK mask). top: 3D held rule right, gripper rule wrong. bottom: 3D phase rule wrong (gripper closing)"),
                 fontsize=9.5, color=FS.INK, x=0.01, ha="left", y=1.0)
    fig.subplots_adjust(top=1 - 0.7 / fig.get_figheight(), hspace=0.32)
    save(fig, out, "ifg-s2fk-scenes", side)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s2fk", required=True, type=pathlib.Path); ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--only", nargs="*", choices=["memory", "phase", "scenes"])
    a = ap.parse_args()
    want = set(a.only or ["memory", "phase", "scenes"])
    if "memory" in want:
        fig_memory_gate(a.s2fk, a.out)
    if "phase" in want:
        fig_phase(a.s2fk, a.out)
    if "scenes" in want:
        fig_scenes(a.s2fk, a.out)


if __name__ == "__main__":
    main()
