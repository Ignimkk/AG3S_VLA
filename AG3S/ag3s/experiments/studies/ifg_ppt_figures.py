"""Presentation figures for the three contributions (2026-10-15 deck): plain Korean annotations, no episode / step codes.

New names under AG3S/ag3s/docs/figures/ppt-1015/ (the ifg/ evidence figures are kept unchanged):
  c1-attention-to-surface.png   VLA attention -> seed patches -> serving ball exemption vs depth-continuous surface
  c1-held-object-kept.png       sphere-volume robot removal vs rendered-robot-surface removal on a held fruit
  c2-feature-identity.png       memory of the target region; raw vs centered VLA feature similarity
  c2-identity-alert.png         low similarity to the remembered target -> alert
  c3-contact-rules.png          link x object clearance rules (only the grasping hand may touch the target)
  c3-target-free-map.png        distance map with / without the target surface (what the grasping hand queries)

  MUJOCO_GL=osmesa PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_ppt_figures \
      --run $RUN --out AG3S/ag3s/docs/figures/ppt-1015 [--only c1 ...]
"""
import argparse, json, os, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_experiment_figures as X  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_graph as G  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_memory as M  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_reader as R  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_round as RD  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_round_mass as RM  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_round_ns as NS  # noqa: E402
from AG3S.ag3s.experiments.studies.ifg_baseline import inside_primitive  # noqa: E402
from AG3S.ag3s.experiments.studies.ifg_figures import _project  # noqa: E402

CAT = figstyle.CATEGORICAL
OK_C, BAD_C, MISS_C = CAT[2], CAT[1], CAT[0]
INK, NOTE, SURFACE = figstyle.INK, "#8f8e89", figstyle.SURFACE
CAM_KO = ("머리 카메라", "왼손 손목 카메라", "오른손 손목 카메라")
CLASS_KO = {"target": "target", "destination": "바구니", "other_object": "다른 물체"}
LEGEND3 = [("target 표면 (맞게 찾음)", OK_C), ("target이 아닌 표면 (잘못 포함)", BAD_C), ("놓친 target 표면", MISS_C)]


def legend(fig, items, y=0.01, fs=12):
    fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=9, color=c, label=l) for l, c in items], loc="lower center",
               ncol=len(items), frameon=False, fontsize=fs, bbox_to_anchor=(0.5, y))


def save(fig, out, name, side, **kw):
    fig.savefig(out / f"{name}.png", dpi=200, facecolor=SURFACE, **kw); plt.close(fig)
    (out / f"{name}.json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=float))
    print(name)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c8c7c2")


def zoom_box(u, v, pad=14, minsz=70):
    u0, u1, v0, v1 = u.min() - pad, u.max() + pad, v.min() - pad, v.max() + pad
    s = max(u1 - u0, v1 - v0, minsz); cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    cu = min(max(cu, s / 2 - 0.5), 223.5 - s / 2); cv = min(max(cv, s / 2 - 0.5), 223.5 - s / 2)
    return cu - s / 2, cu + s / 2, cv - s / 2, cv + s / 2


def set_zoom(ax, box):
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[3], box[2])


def pts(ax, u, v, sel, col, s):
    ax.scatter(u[sel], v[sel], s=s, c=[col], lw=0)


# --------------------------------------------------------------------------------------------- contribution 1
def rows(run, sub, arm):
    return X.rows_of(run, sub, arm)


def pick_c1(run):
    """Pre-fixed rule: approach / pre-grasp frames where the serving ball keeps the right object (identity correct,
    recall >= 0.8) and the candidate reaches IoU >= 0.9; take the largest serving-ball false-positive volume."""
    b, c = rows(run, "baseline_rm", "T54ball+rm:N"), rows(run, "rounds/R6", "top1-only-ns+rm:N")
    L = [(b[f]["FP"], f) for f in c if f in b and b[f]["moment"][:2] in ("F1", "M1", "M2") and b[f].get("identity_correct")
         and (b[f]["recall"] or 0) >= 0.8 and (c[f]["iou"] or 0) >= 0.9]
    L.sort(reverse=True)
    return [f for _, f in L[:5]], b, c


def c1_attention_to_surface(run, out, fid=None):
    top, b, c = pick_c1(run)
    fid = fid or top[0]
    fr = X.Frame(run, fid, mask="render")
    s = X.seed_and_g(fr, True); n = len(s["sub"].pos)
    cand = X.to_full(fr, s["idx"], RD.region_gate(s["m_sub"], s["sub"], np.ones(n), np.zeros(n), 0.5, s["sl_sub"], top1_only=True)[0]) >= 0.5
    meta = json.loads((run / "baseline_rm/raw" / f"{fid}.json").read_text())
    prim = meta["arms"]["N"]["prim"]
    ev_gt = fr.geo.eval_target >= 0.5
    ball = inside_primitive(fr.geo.eval_pos, prim)
    cam = int(np.bincount(np.asarray(s["six"]) // 256, minlength=3).argmax())  # camera holding most seed patches
    K, T = fr.K[cam], fr.T[cam]
    u, v, ok = _project(fr.geo.pos, K, T)
    ue, ve, oke = _project(fr.geo.eval_pos, K, T)
    # ball outline (ellipse in the 224 x 224 policy image, which is a non-uniform resize of 480 x 640)
    cc = np.asarray(prim["center"]); rad = float(prim["dimensions"][0])
    pc = (cc - T[:3, 3]) @ T[:3, :3]
    uc, vc, _ = _project(cc[None], K, T)
    rx, ry = K[0, 0] * rad / pc[2] * 224 / 640, K[1, 1] * rad / pc[2] * 224 / 480
    sel_all = (oke & (ball | ev_gt))
    box = zoom_box(np.r_[ue[sel_all], uc - rx, uc + rx], np.r_[ve[sel_all], vc - ry, vc + ry], pad=8)

    fig, ax = plt.subplots(1, 4, figsize=(18, 5.4)); figstyle.style_axes(fig, ax)
    ax[0].imshow(fr.imgs[cam]); ax[0].add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec=CAT[7], lw=2))
    ax[0].set_title(f"① 입력 장면 ({CAM_KO[cam]}, 빨간 상자 = 확대)", fontsize=13)
    ax[0].text(112, 232, f"명령: \"{fr.fin.prompt.strip()}\"", ha="center", va="top", fontsize=11, color=INK, wrap=True)

    A = fr.fin.maps["N"][cam * 256:(cam + 1) * 256].reshape(16, 16)
    a_ = A / A.max(); rgba = np.zeros((16, 16, 4)); rgba[..., :3] = matplotlib.colors.to_rgb(CAT[7]); rgba[..., 3] = 0.75 * a_ ** 0.7
    ax[1].imshow(fr.imgs[cam]); ax[1].imshow(rgba, extent=(-.5, 223.5, 223.5, -.5), interpolation="nearest")
    X.seed_patch_boxes(ax[1], s["six"], cam, CAT[3])
    for b_ in ax[1].patches:
        b_.set_linewidth(2.4)
    noun = next((w for w in ("apple", "banana", "orange", "pear", "lemon", "peach", "lime") if w in fr.fin.prompt.lower()), "target")
    set_zoom(ax[1], box); ax[1].set_title(f"② VLA attention: '{noun}' 단어가 보는 곳\n(진한 빨강 = 높음, 노란 상자 = 시작점 칸)", fontsize=13)

    ax[2].imshow(fr.imgs[cam])
    pts(ax[2], ue, ve, oke & ball & ev_gt, OK_C, 14); pts(ax[2], ue, ve, oke & ball & ~ev_gt, BAD_C, 14)
    ax[2].add_patch(Ellipse((uc[0], vc[0]), 2 * rx, 2 * ry, fill=False, ec=INK, lw=1.6, ls="--"))
    set_zoom(ax[2], box)
    pb = b[fid]["precision"]
    ax[2].set_title(f"③ 기존: attention 주변 공(ball) 영역\n영역의 {100 * (1 - pb):.0f}%가 target이 아닌 표면", fontsize=13)

    ax[3].imshow(fr.imgs[cam])
    pts(ax[3], u, v, ok & cand & fr.gt, OK_C, 14); pts(ax[3], u, v, ok & cand & ~fr.gt, BAD_C, 14); pts(ax[3], u, v, ok & ~cand & fr.gt, MISS_C, 14)
    set_zoom(ax[3], box)
    pc_ = c[fid]["precision"]
    ax[3].set_title(f"④ 제안: depth로 이어진 표면만 따라감\n영역의 {100 * (1 - pc_):.0f}%가 target이 아닌 표면", fontsize=13)
    for a in ax:
        clean(a)
    legend(fig, LEGEND3)
    fig.suptitle("Attention은 '무엇을', depth 표면은 '어디까지를' 정한다 — 외부 segmentation 모델 없이", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save(fig, out, "c1-attention-to-surface",
         dict(frame=fid, camera=X.CAMS[cam], rule="approach/pre-grasp, serving ball identity correct, ball recall>=0.8, candidate IoU>=0.9; "
              "largest serving-ball FP voxels", candidates_top5=top,
              serving_ball=dict(arm="T54ball+rm:N", **{k: b[fid][k] for k in ("iou", "precision", "recall", "FP", "neighbor_fp", "support_fp", "robot_fp")}),
              candidate=dict(arm="top1-only-ns+rm:N", **{k: c[fid][k] for k in ("iou", "precision", "recall", "FP", "neighbor_fp", "support_fp", "robot_fp")}),
              ball=prim, seed_patches=[int(p) for p in s["six"]]))


def c1_held_object_kept(run, out, fid="T54c_ep1817_f020"):
    ref = X.refs()[fid]; cfg = G.GraphConfig()
    fin = R.load_frame(ref, features=())
    with np.load(run / "s0/replay" / f"{fid}.npz") as z:
        depth, fk, K, T = z["depth"], z["robot_mask"], z["K"], z["T_base_cam"]
    with np.load(run / "rounds/R6/robot_mask" / f"{fid}.npz") as z:
        rm = z["robot_mask"]
    with np.load(run / "s0/labels" / f"{fid}.npz") as z:
        cm, classes = z["class_map"], [str(x) for x in z["classes"]]
    with np.load(ref.path / "inputs.npz") as z:
        img = np.clip((z["image_right_wrist" if "image_right_wrist" in z.files else f"image_{R.DUMP_CAMERAS[2]}"] + 1) / 2, 0, 1)
    c = 2
    rr = {"fk": rows(run, "rounds/R5", "top1-only-ns:N")[fid], "rm": rows(run, "rounds/R6", "top1-only-ns+rm:N")[fid]}
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 10.5)); figstyle.style_axes(fig, ax.ravel())
    side = dict(frame=fid, camera="right wrist", panels={})
    for j, (key, mk, head) in enumerate((("fk", fk, "기존: 로봇 주변을 공 모양 부피로 지움"), ("rm", rm, "제안: 로봇 표면만 정확히 지움"))):
        a = ax[0, j]; a.imshow(img)
        a.imshow(np.ma.masked_where(~mk[c], mk[c]), extent=(-.5, 223.5, 223.5, -.5),
                 cmap=matplotlib.colors.ListedColormap([CAT[6]]), alpha=0.55, interpolation="nearest")
        removed = int(((cm[c] == 0) & mk[c]).sum())
        a.set_title(f"{head}\n지워진 과일 pixel {removed:,}개", fontsize=13)
        geo = G.build_geometry(depth, K, T, mk, cm, classes, cfg)
        sub, _, _, mt, _ = NS.g_and_top1(geo, cfg, fin.maps["N"], NS.eligible(geo, cfg, True))
        pred = mt >= 0.5; gt = geo.eval_target[sub.node_eval] >= 0.5
        u, v, ok = _project(sub.pos, K[c], T[c])
        b_ = ax[1, j]; b_.imshow(img)
        pts(b_, u, v, ok & pred & gt, OK_C, 3); pts(b_, u, v, ok & pred & ~gt, BAD_C, 3); pts(b_, u, v, ok & ~pred & gt, MISS_C, 3)
        b_.set_title(f"복원된 target 표면\n정답과 겹침(IoU) {rr[key]['iou']:.2f}", fontsize=13)
        side["panels"][key] = dict(target_px_removed=removed, iou=rr[key]["iou"], recall=rr[key]["recall"], precision=rr[key]["precision"])
    for a in ax.ravel():
        clean(a)
    fig.text(0.5, 0.075, "위: 보라 = 로봇으로 판단해 지운 영역", ha="center", fontsize=12, color=NOTE)
    legend(fig, LEGEND3, y=0.02, fs=11)
    fig.suptitle("손에 든 물체를 로봇과 함께 지우지 않는다 (오른손 손목 카메라)", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    save(fig, out, "c1-held-object-kept", side)


# --------------------------------------------------------------------------------------------- contribution 2
def c2_feature_identity(run, out, fid_m="T53_ep1822_f005", fid="T53_ep1822_f019"):
    fm, fc = X.Frame(run, fid_m, mask="render", feats=("prefix_L8",)), X.Frame(run, fid, mask="render", feats=("prefix_L8",))
    cm = M.r6_choice(fm.geo, fm.cfg, fm.fin.maps["N"])
    comp = RM.regions(fc.geo, ~fc.geo.support); okc = comp >= 0; px = fc.geo.patch_px.tocsr()
    cam = fc.best_cam()
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.2)); figstyle.style_axes(fig, ax)
    cm_cam = fm.best_cam(cm["chosen"])
    fm.draw(ax[0], cm_cam, [(cm["chosen"], OK_C, 3)], "")
    ax[0].set_title("① 접근할 때 찾은 target 영역을 기억\n(초록 = 기억한 target)", fontsize=13)
    side = dict(memory_frame=fid_m, frame=fid, panels={})
    for a, name, head in ((ax[1], "raw", "② 나중 장면 (잡은 뒤): 그대로 비교하면\n모든 물체가 비슷하다"),
                          (ax[2], "centered", "③ 모든 칸의 공통 성분을 빼고(centering) 비교하면\ntarget만 높다")):
        Um, Uc = R.normalize(fm.fin.features["prefix_L8"]), R.normalize(fc.fin.features["prefix_L8"])
        if name == "centered":
            Um, Uc = R.normalize(Um - Um.mean(0)), R.normalize(Uc - Uc.mean(0))
        mem = M.unit_rows(np.asarray(fm.geo.patch_px.tocsr()[cm["chosen"]].sum(0)).ravel(), Um)
        val = np.full(len(fc.geo.pos), np.nan); labs = []
        for r in np.unique(comp[okc]):
            k = comp == r
            if k.sum() < 20 or (fc.geo.pos[k].max(0) - fc.geo.pos[k].min(0)).max() > 0.4:
                continue
            cs = float(M.unit_rows(np.asarray(px[k].sum(0)).ravel(), Uc) @ mem); val[k] = cs
            cl = fc.classes[int(fc.geo.eval_class_frac[fc.geo.node_eval][k].sum(0).argmax())]
            labs.append((cs, cl, fc.geo.pos[k].mean(0)))
        u, v, okp = _project(fc.geo.pos, fc.K[cam], fc.T[cam]); sel = okp & np.isfinite(val)
        a.imshow(fc.imgs[cam]); sc = a.scatter(u[sel], v[sel], s=3, c=val[sel], cmap=figstyle.sequential_cmap(), lw=0)
        cb = fig.colorbar(sc, ax=a, fraction=0.04); cb.set_label("기억한 target과의 유사도", fontsize=11)
        best = {}
        for cs, cl, p in labs:  # one label per class: the highest region
            if cl not in best or cs > best[cl][0]:
                best[cl] = (cs, p)
        for cl, (cs, p) in best.items():
            if cl not in CLASS_KO:
                continue
            uu, vv, oo = _project(p[None], fc.K[cam], fc.T[cam])
            if oo[0]:
                a.annotate(f"{CLASS_KO.get(cl, cl)} {cs:.2f}", (uu[0], vv[0]), xytext=(0, -16), textcoords="offset points", ha="center",
                           fontsize=12, weight="bold", color=INK if cl != "target" else CAT[5],
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        a.set_title(head, fontsize=13)
        side["panels"][name] = {CLASS_KO[cl]: v_[0] for cl, v_ in best.items() if cl in CLASS_KO}
    for a in ax:
        clean(a)
    fig.suptitle("VLA feature로 '아까 그 target인가'를 확인한다", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out, "c2-feature-identity", side)


def c2_identity_alert(run, out):
    C = [json.loads(l) for l in open(run / "rounds/R11b/conf.jsonl")]
    C = [c for c in C if c.get("q_mem_chain:prefix_L8") is not None and c["target_voxels"] > 0]
    thr = json.loads((run / "LOCK_heldout.json").read_text())["arms"]["confidence_gate"]["abstain_if_below"]
    C.sort(key=lambda c: c["q_mem_chain:prefix_L8"])
    wrong_alerts = [c for c in C if c["q_mem_chain:prefix_L8"] < thr and (c["r6_iou"] or 0) < 0.1][:4]
    show = wrong_alerts + C[-2:]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.5)); figstyle.style_axes(fig, axes.ravel())
    side = dict(threshold=thr, rule="4 lowest-score alerts on wrong choices + 2 highest scores; display only",
                note="all 147 frames: 15 alerts, 6 of them on the 6 wrong choices, 9 false alarms (not shown)", frames=[])
    for a, c in zip(axes.ravel(), show):
        fr = X.Frame(run, c["frame_id"], mask="render")
        ch = M.r6_choice(fr.geo, fr.cfg, fr.fin.maps["N"])["chosen"]
        cam = fr.best_cam(ch if ch.any() else None)
        sc = c["q_mem_chain:prefix_L8"]; alert = sc < thr; wrong = (c["r6_iou"] or 0) < 0.1
        fr.draw(a, cam, [(fr.gt & ~ch, MISS_C, 3), (ch, BAD_C if wrong else OK_C, 3)], "")
        a.set_title(("경보: 기억한 target과 다름" if alert else "통과: 기억한 target과 같음") + f"\n유사도 {sc:.2f}",
                    fontsize=13, color=CAT[7] if alert else INK)
        clean(a)
        side["frames"].append(dict(frame=c["frame_id"], score=sc, alert=bool(alert), wrong=bool(wrong)))
    legend(fig, [("attention이 고른 영역 (맞음)", OK_C), ("attention이 고른 영역 (틀림)", BAD_C), ("실제 target", MISS_C)])
    fig.suptitle(f"기억한 target과 유사도가 낮으면(< {thr:.2f}) 경보 → 그 영역을 target으로 확정하지 않고 장애물로 둔다", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=3.0)
    save(fig, out, "c2-identity-alert", side)


# --------------------------------------------------------------------------------------------- contribution 3
def c3_contact_rules(run, out):
    AUTH, OTHER, HELD = CAT[1], CAT[0], CAT[3]
    fig = plt.figure(figsize=(12.0, 5.8), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 120); ax.set_ylim(0, 58); ax.axis("off")
    cols = [("일반 장애물", "테이블 · 다른 과일"), ("Target (잡을 과일)", "VLA attention으로 찾은 표면"), ("목적지 (바구니)", "")]
    rws = [("잡는 손", "손바닥 · 손가락", AUTH), ("그 밖의 로봇", "손목 · 팔 · 몸통 · 반대손", OTHER), ("손에 든 과일", "잡은 뒤 로봇과 함께 움직임", HELD)]
    X0, CW, Y0, RH, HW = 22, 32, 46.5, 11.2, 20
    for j, (h, s) in enumerate(cols):
        x = X0 + j * CW
        ax.text(x + CW / 2, Y0 + 6.4, h, ha="center", fontsize=14, weight="bold", color=INK)
        ax.text(x + CW / 2, Y0 + 3.4, s, ha="center", fontsize=10, color=NOTE)
    for i, (h, s, c) in enumerate(rws):
        y = Y0 - (i + 1) * RH
        ax.add_patch(FancyBboxPatch((1, y + 0.6), HW - 1.5, RH - 1.2, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc=matplotlib.colors.to_rgba(c, 0.12), ec=c, lw=1.2))
        ax.text(2.5, y + RH / 2 + 1.4, h, fontsize=14, weight="bold", color=INK, va="center")
        ax.text(2.5, y + RH / 2 - 1.9, s, fontsize=10, color=NOTE, va="center")

    def cell(i, j, big, small=None, color=INK, fc="#ffffff", ec="#d9d8d3"):
        x = X0 + j * CW; y = Y0 - (i + 1) * RH
        ax.add_patch(FancyBboxPatch((x + 0.6, y + 0.6), CW - 1.2, RH - 1.2, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=1.0))
        ax.text(x + CW / 2, y + RH / 2 + (1.6 if small else 0), big, ha="center", va="center", fontsize=18, weight="bold", color=color)
        if small:
            ax.text(x + CW / 2, y + RH / 2 - 2.6, small, ha="center", va="center", fontsize=10, color=NOTE, linespacing=1.25)

    cell(0, 0, "50 mm")
    x = X0 + CW; y = Y0 - RH
    ax.add_patch(FancyBboxPatch((x + 0.6, y + 0.6), CW - 1.2, RH - 1.2, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=matplotlib.colors.to_rgba(AUTH, 0.07), ec=AUTH, lw=1.6))
    steps = [("이동", 50), ("접근", 20), ("파지 직전", 5), ("파지", 0)]
    bw = (CW - 5) / 4
    for k, (p, val) in enumerate(steps):
        bx = x + 2.5 + k * bw; h = 3.2 * val / 50
        ax.add_patch(plt.Rectangle((bx + 0.6, y + 2.7), bw - 1.2, max(h, 0.15), fc=AUTH, ec="none", alpha=0.85))
        ax.text(bx + bw / 2, y + 2.7 + max(h, 0.15) + 0.3, f"{val}", ha="center", va="bottom", fontsize=11, weight="bold", color=INK)
        ax.text(bx + bw / 2, y + 1.6, p, ha="center", fontsize=9, color=NOTE)
    ax.text(x + CW / 2, y + RH - 2.0, "다가갈수록 허용 거리를 줄인다 [mm]", ha="center", va="center", fontsize=10.5, color=INK)
    cell(0, 2, "50 mm")
    cell(1, 0, "50 mm")
    cell(1, 1, "50 mm", "팔·손목이 과일에 닿으면\n파지가 아니라 충돌", color=OTHER)
    cell(1, 2, "50 mm")
    cell(2, 0, "50 mm")
    cell(2, 1, "—", "자기 자신")
    cell(2, 2, "20 mm", "바구니 안에 넣어야 하므로\n여유를 줄인다", color=HELD, fc=matplotlib.colors.to_rgba(HELD, 0.08))
    notes = ["• 접촉 예외는 '잡는 손'에만 준다. 그 밖의 로봇에게 target은 여전히 장애물이다.",
             "• 0 mm는 제약을 끄는 것이 아니다. 닿는 것만 허용하고 파고드는 것은 막는다.",
             "• 판단할 수 없는 link · 단계 · 물체는 기본값 50 mm를 유지한다 (잘못 느슨하면 부딪힌다)."]
    for k, t in enumerate(notes):
        ax.text(1.5, 8.6 - k * 2.4, t, fontsize=11, color=INK if k == 0 else NOTE, va="center")
    fig.savefig(out / "c3-contact-rules.png", dpi=220, facecolor=SURFACE); plt.close(fig)
    (out / "c3-contact-rules.json").write_text(json.dumps(dict(
        values_mm=dict(obstacle=50, target_other_links=50, target_grasping_hand_by_phase=dict(steps), held_vs_destination=20, other_vs_destination=50),
        source="redrawn from AG3S/ag3s/docs/figures/ppt-0930/make_adapter_margin_figs.py (same values; sources listed in its sidecar)"),
        indent=1, ensure_ascii=False))
    print("c3-contact-rules")


def c3_target_free_map(run, out):
    src = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", "/mnt/dev/AG3S_VLA-storage")) / \
        "results/workspace/live_test/20260928_t17/server_constraints/run_0000/chunk_00000.npz"
    ch = np.load(src, allow_pickle=True)
    tc = np.asarray(ch["target_centroid"], np.float64); vs = float(ch["esdf_voxel_size"])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("esdf_light", [figstyle.SEQ_BLUE[5], figstyle.SEQ_BLUE[1], "#ffffff"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), facecolor=SURFACE)
    stats = {}
    for a, key, okey, title in ((axes[0], "esdf_distance", "esdf_origin", "그 밖의 로봇이 보는 지도\n(Target 포함)"),
                                (axes[1], "esdf_free_distance", "esdf_free_origin", "잡는 손이 보는 지도\n(Target 표면만 제외)")):
        o = np.asarray(ch[okey], np.float64); k = int(round((tc[2] - o[2]) / vs))
        img = np.asarray(ch[key][:, :, k], np.float64); nx, ny = img.shape
        ext = (o[1] - vs / 2, o[1] + (ny - 0.5) * vs, o[0] - vs / 2, o[0] + (nx - 0.5) * vs)
        a.imshow(np.ma.masked_where(img <= 0, np.clip(img, 0, 0.06)), origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=0.06, interpolation="nearest")
        a.imshow(np.ma.masked_where(img > 0, np.ones_like(img)), origin="lower", extent=ext, cmap=matplotlib.colors.ListedColormap([INK]), interpolation="nearest")
        a.add_patch(plt.Circle((tc[1], tc[0]), 0.055, fill=False, ec=CAT[7], lw=2.0, ls=(0, (4, 2))))
        a.set_xlim(tc[1] + 0.22, tc[1] - 0.22); a.set_ylim(tc[0] - 0.20, tc[0] + 0.20); a.set_aspect("equal"); clean(a)
        a.set_title(title, fontsize=13, color=INK, pad=6)
        stats[key] = {"k_index": k, "occupied_voxels_in_slice": int((img <= 0).sum())}
    axes[0].annotate("Target (사과)", xy=(tc[1] - 0.055, tc[0]), xytext=(tc[1] - 0.12, tc[0] + 0.14), fontsize=12, color=CAT[7],
                     arrowprops=dict(arrowstyle="->", color=CAT[7]))
    axes[1].annotate("Target만 사라지고\n다른 물체는 그대로", xy=(tc[1] - 0.055, tc[0]), xytext=(tc[1] - 0.07, tc[0] + 0.15), fontsize=12,
                     color=CAT[7], arrowprops=dict(arrowstyle="->", color=CAT[7]))
    fig.text(0.5, 0.03, "위에서 본 단면 (Target 높이) · 검정 = 물체 · 진한 파랑 = 물체 표면에 가까움", ha="center", fontsize=11, color=NOTE)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    save(fig, out, "c3-target-free-map", dict(src=str(src), target_centroid_m=tc.round(4).tolist(), voxel_m=vs, **stats,
                                             note="existing serving target-free layer (ball-based target); illustrates the interface only"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--c1-frame", default=None)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if not figstyle.use_korean():
        raise SystemExit("CJK font missing")
    figs = dict(c1=lambda r, o: c1_attention_to_surface(r, o, a.c1_frame), c1held=c1_held_object_kept, c2id=c2_feature_identity,
                c2alert=c2_identity_alert, c3rules=c3_contact_rules, c3map=c3_target_free_map)
    for k, f in figs.items():
        if a.only and k not in a.only:
            continue
        f(a.run, a.out)


if __name__ == "__main__":
    main()
