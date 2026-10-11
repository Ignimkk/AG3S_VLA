"""T45 figures: t45-scene / t45-graph / t45-table (+ .json sidecars). Numbers only from outputs/verify/T45/*.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, numpy as np, pathlib
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T45/p'))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from AG3S.ag3s.experiments.common import figstyle as FS
KO = FS.use_korean()
T = lambda ko, en: ko if KO else en
O = pathlib.Path(_ag3s_workspace_path('outputs/verify/T45')); F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t45')); F.mkdir(parents=True, exist_ok=True)
cov = json.load(open(O / "coverage.json")); ev = json.load(open(O / "events_cands.json")); cand = json.load(open(O / "candidates.json"))
import common as C
CAT = FS.CATEGORICAL
MODEL_LABEL = {"server": T("서버 (T44d)", "server (T44d)"), "palm_scale1": "palm scale 1.0", "palm_body_o2.5": "palm body cover 2.5 mm", "palm_body_o5": "palm body cover 5 mm"}
MODELS = ["server", "palm_scale1", "palm_body_o2.5", "palm_body_o5"]

def palm_surface_body():
    pts = []
    for g in C.coll_geoms(C.BODY("EE_BODY_R")):
        mesh = int(C.M.geom_dataid[g]); v0, nv = int(C.M.mesh_vertadr[mesh]), int(C.M.mesh_vertnum[mesh])
        V = np.asarray(C.M.mesh_vert[v0:v0 + nv], float).reshape(-1, 3)
        R = np.zeros(9); __import__("mujoco").mju_quat2Mat(R, C.M.geom_quat[g]); V = V @ R.reshape(3, 3).T + C.M.geom_pos[g]
        pts.append(C.hull_sample(V, n_per_area=3e5)[0])
    return np.vstack(pts)

# ---------------- scene ----------------
def scene():
    import candidates as K
    side = {}
    img = np.load(O / "render_C1a_ep54_s542_t157.npy")
    P = palm_surface_body() * 1e3
    palm_ev = [e for e in ev if "EE_BODY_R" in e.get("bodies", [])]
    CP = np.array([p["point_body_mm"] for e in palm_ev for p in e["points"]])
    qms = {"server": C.model("gripper"), "palm_body_o2.5": K.models()["palm_body_o2.5"], "palm_scale1": K.models()["palm_scale1"]}
    fig = plt.figure(figsize=(17, 8.6))
    ax0 = fig.add_axes([0.01, 0.30, 0.30, 0.62]); ax0.imshow(img); ax0.axis("off")
    ax0.set_title(T("MuJoCo: C1a_ep54_s542 t=157 손바닥(EE_BODY_R) – orange 접촉\n(target 사과를 쥐는 중, closing, 18.1 N)",
                    "MuJoCo: C1a_ep54_s542 t=157 palm (EE_BODY_R) - orange contact\n(closing on target apple, 18.1 N)"), fontsize=9, color=FS.INK)
    views = [(0, 2, "x", "z"), (1, 2, "y", "z")]
    axes = []
    for j, (name, lab) in enumerate([("server", T("서버 구 (scale 0.05, 23 구 · r 5.7–6.2 mm)", "server spheres (scale 0.05, 23 · r 5.7-6.2 mm)")),
                                      ("palm_scale1", T("후보 palm scale 1.0", "candidate palm scale 1.0")),
                                      ("palm_body_o2.5", T("후보 palm body cover 2.5 mm", "candidate palm body cover 2.5 mm"))]):
        c, r = qms[name].local_spheres("ee_right"); c = c * 1e3; r = r * 1e3
        for k, (a, b, an, bn) in enumerate(views):
            ax = fig.add_axes([0.33 + 0.225 * j, 0.53 - 0.47 * k, 0.21, 0.42]); axes.append(ax)
            ax.scatter(P[:, a], P[:, b], s=0.3, color=FS.GRID_INK, rasterized=True, label=T("MJCF collision hull 표면", "MJCF collision hull surface"))
            for cc, rr in zip(c, r):
                ax.add_patch(Circle((cc[a], cc[b]), rr, fill=False, lw=0.6, color=CAT[0], alpha=0.8))
            ax.scatter(CP[:, a], CP[:, b], s=14, color=CAT[7], zorder=5, label=T("palm 접촉점 (8 event)", "palm contact points (8 events)"))
            ax.set_aspect("equal"); ax.set_xlim(-75, 75); ax.set_ylim(-95, 25)
            ax.set_xlabel(f"{an} (mm, EE_BODY_R frame)", fontsize=8); ax.set_ylabel(f"{bn} (mm)", fontsize=8)
            if k == 0: ax.set_title(lab + f"\n{len(r)} " + T("구", "spheres"), fontsize=9, color=FS.INK)
            side[f"{name}_{an}{bn}"] = {"n_spheres": int(len(r)), "r_mm": [float(r.min()), float(r.max())]}
    FS.style_axes(fig, axes)
    axes[0].legend(loc="upper left", fontsize=7, frameon=False, markerscale=3)
    fig.text(0.01, 0.12, T("투영: palm body frame 의 x–z (위) · y–z (아래).\n파란 원 = 구의 투영 (손바닥 구만).\n손가락 구는 FC (mesh 를 다 담음, 넘침 ≤ 2.5 mm) 라 넣지 않았다.\n"
                           "링크 짝: URDF ee_right ↔ MJCF EE_BODY_R\n(구 위치로 확인, frame 차 0.09 mm · 0.0004°).",
                           "Projection: palm body frame x-z (top), y-z (bottom). Blue circles = palm spheres only. Finger spheres (FC, contain the mesh, overflow <= 2.5 mm) omitted.\n"
                           "Pairing URDF ee_right <-> MJCF EE_BODY_R by sphere position (frame offset 0.09 mm, 0.0004 deg)."), fontsize=8, color=FS.INK_2)
    fig.savefig(F / "t45-scene.png", dpi=150); plt.close(fig)
    json.dump({"render": "outputs/verify/T45/render_C1a_ep54_s542_t157.npy", "render_meta": json.load(open(O / "render_meta.json")),
               "palm_contact_points_body_mm": CP.tolist(), "spheres": side}, open(F / "t45-scene.json", "w"), indent=1)

# ---------------- graph ----------------
def graph():
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    m = cov["models"]["gripper"]["per_link"]
    links = ["ee_right", "ee_finger_r1", "ee_finger_r2", "ee_left", "ee_finger_l1", "ee_finger_l2"]
    lab = [f"{L}\n({m[L]['open']['mjcf_body']})" for L in links]
    x = np.arange(len(links))
    own = [m[L]["open"]["own_spheres_vs_mjcf"] for L in links]; caps = [m[L]["open"]["source_capsules_vs_mjcf"] for L in links]
    ax[0].bar(x - 0.2, [o["frac_out"] for o in own], 0.4, color=CAT[0], label=T("서버 구 (자기 link)", "server spheres (own link)"))
    ax[0].bar(x + 0.2, [c["frac_out"] for c in caps], 0.4, color=CAT[1], label=T("원 capsule (배율 1.0)", "source capsules (scale 1.0)"))
    ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=7); ax[0].set_ylabel(T("MJCF 겉면 표본 중 구 밖 비율", "fraction of MJCF surface samples outside"))
    ax[0].set_title(T("① coverage: 밖 비율 (열림 자세)", "1 coverage: fraction outside (open)"), fontsize=10, color=FS.INK); ax[0].legend(fontsize=7, frameon=False)
    for k, o in enumerate(own): ax[0].text(x[k] - 0.2, o["frac_out"] + 0.01, f"{o['frac_out']:.3f}", ha="center", fontsize=7, color=FS.INK_2)
    # p95 / max
    for j, (key, mk) in enumerate([("out_p95_mm", "o"), ("out_max_mm", "^")]):
        ax[1].plot(x, [o[key] for o in own], mk + "-", color=CAT[0], label=T(f"서버 구 {key[4:7]}", f"server {key[4:7]}"), alpha=1 - 0.4 * j)
        ax[1].plot(x, [c[key] for c in caps], mk + "--", color=CAT[1], label=T(f"원 capsule {key[4:7]}", f"source capsule {key[4:7]}"), alpha=1 - 0.4 * j)
    for nm, ls in (("half", ":"), ("closed", "-.")):
        ax[1].plot(x, [m[L][nm]["hand_union_vs_mjcf"]["out_max_mm"] for L in links], "s" + ls, color=CAT[2], ms=4,
                   label=T(f"손 전체 구 max ({nm})", f"whole-hand spheres max ({nm})"))
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=7); ax[1].set_ylabel(T("밖 거리 (mm)", "outside distance (mm)"))
    ax[1].set_title(T("② 밖 거리 p95 / max", "2 outside distance p95 / max"), fontsize=10, color=FS.INK); ax[1].legend(fontsize=6.5, frameon=False, ncol=2)
    # events
    rows = sorted(ev, key=lambda e: ("EE_BODY_R" not in e.get("bodies", []), e["arm"], e["name"], e["t_first"]))
    y = np.arange(len(rows))
    for k, mname in enumerate(MODELS):
        v = [e.get(f"{mname}_gap_hand_mm_min_med_max", [np.nan] * 3)[1] for e in rows]
        ax[2].scatter(v, y + (k - 1.5) * 0.18, s=14, color=CAT[k], label=MODEL_LABEL[mname], zorder=3)
    ax[2].axvline(0, color=FS.INK_2, lw=0.8)
    ax[2].set_yticks(y); ax[2].set_yticklabels([f"{e['arm']} {e['name'][len(e['arm'])+1:]} t{e['t_first']} {('palm' if 'EE_BODY_R' in e.get('bodies', []) else 'finger')}" for e in rows], fontsize=6)
    ax[2].invert_yaxis(); ax[2].set_xlabel(T("접촉점 → 손 전체 구 합집합 거리 (mm, >0 = 밖) — event 중앙값", "contact point to whole-hand sphere union (mm, >0 outside), event median"))
    ax[2].set_title(T("③ T44d 접촉 24 event: 접촉점이 구 밖인가", "3 T44d 24 contact events: is the contact point outside the spheres"), fontsize=10, color=FS.INK)
    ax[2].legend(fontsize=7, frameon=False, loc="lower right")
    FS.style_axes(fig, ax); fig.tight_layout(); fig.savefig(F / "t45-graph.png", dpi=150); plt.close(fig)
    json.dump({"links": links, "own_open": own, "source_capsules_open": caps,
               "hand_union_max_mm": {nm: [m[L][nm]["hand_union_vs_mjcf"]["out_max_mm"] for L in links] for nm in ("open", "half", "closed")},
               "events": [{"arm": e["arm"], "name": e["name"], "t_first": e["t_first"], "bodies": e.get("bodies"),
                           **{f"{mn}_gap_hand_mm_min_med_max": e.get(f"{mn}_gap_hand_mm_min_med_max") for mn in MODELS},
                           **{f"{mn}_gap_own_mm_min_med_max": e.get(f"{mn}_gap_own_mm_min_med_max") for mn in MODELS}} for e in rows]},
              open(F / "t45-graph.json", "w"), indent=1)

# ---------------- table ----------------
def table():
    m = cov["models"]["gripper"]["per_link"]
    fig = plt.figure(figsize=(17, 9.5)); fig.patch.set_facecolor(FS.SURFACE)
    def tbl(ax, header, rows, title, widths=None):
        ax.axis("off"); ax.set_title(title, fontsize=10, color=FS.INK, loc="left")
        t = ax.table(cellText=rows, colLabels=header, loc="upper left", cellLoc="center", colWidths=widths)
        t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.35)
        for (i, j), c in t.get_celld().items():
            c.set_edgecolor(FS.GRID_INK)
            if i == 0: c.set_facecolor(FS.SEQ_BLUE[0])
    a1 = fig.add_axes([0.02, 0.62, 0.96, 0.33])
    rows = []
    for L in ["ee_right", "ee_finger_r1", "ee_finger_r2", "ee_left", "ee_finger_l1", "ee_finger_l2"]:
        o = m[L]["open"]; s = o["own_spheres_vs_mjcf"]; c = o["source_capsules_vs_mjcf"]; h = o["hand_union_vs_mjcf"]
        rows.append([L, o["mjcf_body"], f"{o['spheres_own']['n']}", f"{o['spheres_own']['r_mm_min']:.1f}–{o['spheres_own']['r_mm_max']:.1f}",
                     f"{s['frac_out']:.3f}", f"{s['out_median_mm']:.1f} / {s['out_p95_mm']:.1f} / {s['out_max_mm']:.1f}",
                     f"{h['frac_out']:.3f} · {h['out_max_mm']:.1f}", f"{o['excess']['union_outside_geom_cm3']:.1f} · {o['excess']['protrusion_max_mm']:.1f}",
                     " / ".join(f"{x['r_mm']:.1f}" for x in o["source_capsules"]), f"{c['frac_out']:.3f} · {c['out_max_mm']:.1f}"])
    tbl(a1, [T("URDF link", "URDF link"), T("MJCF body (위치로 짝)", "MJCF body (by position)"), T("구 수", "n"), "r (mm)", T("밖 비율", "frac out"),
             T("밖 거리 med / p95 / max (mm)", "out med / p95 / max (mm)"), T("손 전체 구: 밖 비율 · max", "whole hand: frac · max"),
             T("구 과잉 cm³ · 최대 돌출 mm", "excess cm3 · max protrusion mm"), T("원 capsule r (mm, 배율 1.0)", "source capsule r (mm)"), T("원 capsule: 밖 · max", "source caps: frac · max")],
        rows, T("① 서버 constraint model (--links gripper = arms 의 손 구와 동일) vs MJCF collision hull — 열림 자세 (손가락 ±0.05)",
                "1 server constraint model (gripper; arms hand spheres identical) vs MJCF collision hulls, open fingers"))
    a2 = fig.add_axes([0.02, 0.33, 0.96, 0.25]); rows = []
    for k in MODELS:
        r = cand[k]; o = r["ee_right_open"]; ch = o["channel"]
        pe = [e for e in ev if "EE_BODY_R" in e.get("bodies", [])]
        nin = sum(1 for e in pe for p in e["points"] if p[f"{k}_gap_hand_mm"] <= 0); ntot = sum(len(e["points"]) for e in pe)
        rows.append([MODEL_LABEL[k], f"{r['n_spheres_total']}", f"{r['per_link_n'].get('ee_right')}", f"{o['spheres_own']['r_mm_min']:.1f}–{o['spheres_own']['r_mm_max']:.1f}",
                     f"{o['own_spheres_vs_mjcf']['frac_out']:.3f} · {o['own_spheres_vs_mjcf']['out_max_mm']:.1f}",
                     f"{o['excess']['protrusion_max_mm']:.1f}",
                     f"{ch['palm_spheres_tipmost_in_channel_mm']:.1f} vs {ch['palm_mesh_tipmost_in_channel_mm']:.1f}" if ch["palm_mesh_tipmost_in_channel_mm"] is not None else "-",
                     f"{ch['palm_spheres_lateral_reach_mm']:.1f} vs {ch['palm_mesh_lateral_reach_mm']:.1f}",
                     f"{nin} / {ntot}", f"{16 * r['n_spheres_total']}"])
    tbl(a2, [T("모델 (손바닥만 다름)", "model (palm only differs)"), T("구 전체", "n total"), "ee_right n", "r (mm)", T("palm 밖 비율 · max (mm)", "palm frac out · max"),
             T("palm 최대 돌출 (mm)", "palm max protrusion"), T("통로 안 손끝 쪽 끝: 구 vs mesh (mm)", "tip-ward end in channel: spheres vs mesh"),
             T("손가락 축 방향 폭: 구 vs mesh (mm)", "reach along finger axis: spheres vs mesh"), T("palm 접촉점 중 구 안", "palm contact pts inside"), T("ESDF 질의 행/chunk (16×구)", "ESDF rows/chunk")],
        rows, T("② 최소 수정 후보 (오프라인 계산만, 구현 안 함) — 열림 자세, ee_right", "2 minimal fix candidates (offline only, not implemented), open, ee_right"))
    a3 = fig.add_axes([0.02, 0.02, 0.96, 0.27])
    hist = [["T6f", T("capsule_radius_scale 손잡이 도입 (기본 1.0, 비트 동일)", "knob added (default 1.0)"), "AG3S_T0T6_LOG.md:2825-2830"],
            ["T28 (2026-09-28)", T("사용자 Q5 '구가 너무 커서인가 — 최대한 작게'; live 0.2 (손가락 4.7 · 손바닥 8.9 mm) → 0.05", "user Q5: as small as possible; live 0.2 -> 0.05"), "T28.task.md:15,63,90"],
            ["T28 P1-3a", T("1.0: 왼손 손가락 구 사이 −28.5 mm (겹침) · palm 정점 100 % 안 / 0.05: +2.35 mm · palm 정점 10.6 % 안", "1.0: finger sphere gap -28.5 mm, palm verts 100% in / 0.05: +2.35 mm, 10.6%"), "T28.verify.partial.json P1-3a"],
            ["T8e", T("손바닥을 제약에 넣으면 (ee_left, crate) 18 행 −6.07 mm 가 남음 (palm r_crit 29.76 mm)", "palm in constraints: (ee_left, crate) 18 rows -6.07 mm"), "T8e.verify.json:12323"],
            ["T43 R / FC", T("손가락만 FC 로 교체 (구 122→270, TO CPU 중앙 ×1.4–1.7); 손바닥은 0.05 축선 사슬 그대로. --gripper-cover 는 손바닥 567 구", "fingers replaced by FC (122->270, TO x1.4-1.7); palm unchanged; gripper-cover 567 palm spheres"), "AG3S_T0T6_LOG.md:5907-5921 · T43R.impl.md:53-56"]]
    tbl(a3, [T("단계", "step"), T("무엇", "what"), T("근거", "source")], hist, T("③ capsule-radius-scale 0.05 의 이력 (로그 인용)", "3 history of capsule-radius-scale 0.05"), widths=[0.1, 0.62, 0.25])
    fig.savefig(F / "t45-table.png", dpi=150); plt.close(fig)
    json.dump({"per_link_open": {L: m[L]["open"] for L in m}, "candidates": cand, "history": hist}, open(F / "t45-table.json", "w"), indent=1, default=str)

if __name__ == "__main__":
    for f in sys.argv[1:] or ["scene", "graph", "table"]:
        globals()[f](); print("ok", f)
