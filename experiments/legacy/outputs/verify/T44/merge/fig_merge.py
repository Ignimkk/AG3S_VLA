"""T44b D-merge figures (verifier): t44m-scene / t44m-graph / t44m-table (+ .json sidecars).
Reads merge_gt.json, merge_vla.json, agg_merge.json, wrongpick.json, att_2d.json, raw_gt/, raw_vla/."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle as FS

M = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/merge'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t44')); FIG.mkdir(parents=True, exist_ok=True)
for n in ():
    assert not (FIG / f"{n}.png").exists(), f"{n}.png exists — pick a new name"
KO = FS.use_korean()
L = (lambda ko, en: ko) if KO else (lambda ko, en: en)
A = json.load(open(M / "agg_merge.json"))
W = json.load(open(M / "wrongpick.json"))
G = {r["key"]: r for r in json.load(open(M / "merge_gt.json"))}
V = {r["key"]: r for r in json.load(open(M / "merge_vla.json"))}
C_T, C_O, C_X = FS.CATEGORICAL[0], FS.CATEGORICAL[1], "#b9b8b2"

EXAMPLES = [("demo_ep0012_f068", L("VLA split 로 해결", "resolved by split (VLA)")),
            ("demo_ep0022_f104", L("VLA attention 이 이웃에", "VLA attention on neighbour")),
            ("demo_ep0027_f068", L("기하: 틈 < 15 mm", "geometry: gap < 15 mm"))]


def crop_box(z, tid, oid, cam="zed_left", pad=40):
    ids = z[f"bodyids_{cam}"]
    m = (ids == tid) | (ids == oid)
    v, u = np.nonzero(m)
    return max(0, v.min() - pad), min(ids.shape[0], v.max() + pad), max(0, u.min() - pad), min(ids.shape[1], u.max() + pad)


def fig_scene():
    fig, axes = plt.subplots(len(EXAMPLES), 4, figsize=(17, 4.2 * len(EXAMPLES)))
    FS.style_axes(fig, axes)
    side = []
    V_npz = np.load(M / "vla_attention.npz", allow_pickle=True)
    vk = {str(k): i for i, k in enumerate(V_npz["keys"])}
    for row, (key, tag) in enumerate(EXAMPLES):
        zg, zv = np.load(M / "raw_gt" / f"{key}.npz"), np.load(M / "raw_vla" / f"{key}.npz")
        rg, rv = G[key], V[key]
        names = {int(k): v for k, v in json.loads(str(zg["body_names"])).items()}
        inv = {v: k for k, v in names.items()}
        t, o = rg["target"], rg["merge_with"]
        tid, oid = inv[t], inv[o]
        v0, v1, u0, u1 = crop_box(zg, tid, oid)
        rgb = zg["rgb_zed_left"]
        ids = zg["bodyids_zed_left"]
        ax = axes[row, 0]
        ax.imshow(rgb[v0:v1, u0:u1])
        for bid, c in ((tid, C_T), (oid, C_O)):
            ax.contour((ids[v0:v1, u0:u1] == bid).astype(float), levels=[0.5], colors=[c], linewidths=1.5)
        ax.set_title(f"{key}\n{L('target', 'target')} {t} / {L('이웃', 'neighbour')} {o} · {tag}", fontsize=9)
        ax.axis("off")
        for j, (src, att) in enumerate((("GT Gaussian", zg["attmap_zed_left"] if "attmap_zed_left" in zg.files else None),
                                        ("VLA (L8 H2, step 0, last)", V_npz["attention"][vk[key], 0]))):
            ax = axes[row, 1 + j]
            ax.imshow(rgb[v0:v1, u0:u1])
            if att is not None:
                up = np.kron(att / att.max(), np.ones((30, 40)))[v0:v1, u0:u1]
                ax.imshow(up, cmap=FS.sequential_cmap(), alpha=0.55, vmin=0, vmax=1)
            for bid, c in ((tid, C_T), (oid, C_O)):
                ax.contour((ids[v0:v1, u0:u1] == bid).astype(float), levels=[0.5], colors=[c], linewidths=1.2)
            r = rg if j == 0 else rv
            ratio = r["split_recorded"]["ratio"]
            ax.set_title(f"zed_left {src}\nsplit: {r['split_recorded']['reason']} · ratio {ratio}", fontsize=9)
            ax.axis("off")
        # top-down parent cluster
        ax = axes[row, 3]
        P = zg["points"]; nm = np.array([names.get(int(b), "none") for b in zg["body_id"]])
        cl = zg["cluster_label"]
        par = np.zeros(len(P), bool)
        is_t = (nm == t) & ~zg["support"]
        cands = sorted(((int((is_t & (cl == c)).sum()), c) for c in range(cl.max() + 1)), key=lambda x: -x[0])
        par = cl == cands[0][1]
        q = P[par]; lab = nm[par]
        for nmx, c, mk in ((t, C_T, "o"), (o, C_O, "s")):
            mm = lab == nmx
            ax.scatter(q[mm, 0] * 1000, q[mm, 1] * 1000, s=10, c=c, marker=mk, label=f"{nmx} ({mm.sum()})")
        mm = ~np.isin(lab, [t, o])
        if mm.any():
            ax.scatter(q[mm, 0] * 1000, q[mm, 1] * 1000, s=8, c=C_X, label=f"other ({mm.sum()})")
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.legend(fontsize=7, loc="best")
        ax.set_title(L(f"GT attention 의 parent cluster (위에서 본 xy)\nmesh 틈 {rg['mesh_gap_merged_mm']} mm · 점 틈 {rg['cloud_gap_mm']} mm",
                       f"parent cluster, top view (xy)\nmesh gap {rg['mesh_gap_merged_mm']} mm · cloud gap {rg['cloud_gap_mm']} mm"), fontsize=9)
        side.append({"key": key, "tag": tag, "target": t, "neighbour": o, "mesh_gap_mm": rg["mesh_gap_merged_mm"],
                     "cloud_gap_mm": rg["cloud_gap_mm"], "parent_n": rg["parent"]["n"], "parent_target_fraction": rg["parent"]["target_fraction"],
                     "gt_split": rg["split_recorded"], "vla_split": rv["split_recorded"],
                     "vla_pipeline_cluster": rv["pipeline_cluster"]})
    fig.suptitle(L("T44b D-merge — 합쳐진 cluster: 실제 장면 (head 카메라), GT 합성 vs 실제 VLA attention, parent cluster",
                   "T44b D-merge — merged cluster: real scene (head camera), GT Gaussian vs real VLA attention, parent cluster"), fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "t44m-scene.png", dpi=130)
    (FIG / "t44m-scene.json").write_text(json.dumps({"examples": side, "source": str(M)}, indent=1, ensure_ascii=False))


CANDS = [("current: recluster15 + ratio2 (norm att)", L("현재 (15 mm · ratio 2 norm)", "current (15 mm, ratio 2 norm)")),
         ("A: recluster12 + ratio2", L("(a) 12 mm · ratio 2 norm [config]", "(a) 12 mm, ratio 2 norm [config]")),
         ("A: recluster15 + ratio2 (raw att)", L("(a) 15 mm · ratio 2 raw", "(a) 15 mm, ratio 2 raw")),
         ("A: recluster12 + ratio2 (raw att)", L("(a) 12 mm · ratio 2 raw", "(a) 12 mm, ratio 2 raw")),
         ("A: recluster10 + ratio2", L("(a) 10 mm · ratio 2 norm", "(a) 10 mm, ratio 2 norm")),
         ("A: dbscan eps20", L("(a) DBSCAN eps 20 mm", "(a) DBSCAN eps 20 mm")),
         ("B: convexity + raw ratio2 gate", L("(b) convexity (LCCP 류) + gate", "(b) convexity (LCCP-like) + gate")),
         ("B: sphere + raw ratio2 gate", L("(b) sphere RANSAC + gate", "(b) sphere RANSAC + gate")),
         ("B: colour + raw ratio2 gate", L("(b) 색 region + gate", "(b) colour region + gate")),
         ("B: attention min-cut (raw)", L("(b) attention min-cut", "(b) attention min-cut")),
         ("A+B: recluster12 | colour (union of cuts) + raw ratio2 gate", L("(a+b) 12 mm ∪ 색 + gate", "(a+b) 12 mm ∪ colour + gate"))]


def fig_graph():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    FS.style_axes(fig, axes)
    side = {}
    # (a) gaps
    ax = axes[0]
    keys = sorted(G)
    for merged, c, mk, lab in ((True, C_O, "s", L("merged (34)", "merged (34)")), (False, C_T, "o", L("not merged (30)", "not merged (30)"))):
        ks = [k for k in keys if G[k]["e2_merged"] == merged]
        xs, ys = [], []
        for k in ks:
            r = G[k]
            if merged:
                xs.append(r["mesh_gap_merged_mm"]); ys.append(r["cloud_gap_mm"])
            else:
                z = np.load(M / "raw_gt" / f"{k}.npz")
                names = {int(a): b for a, b in json.loads(str(z["body_names"])).items()}
                nm = np.array([names.get(int(b), "none") for b in z["body_id"]])
                o = min(r["mesh_gap_mm"], key=r["mesh_gap_mm"].get)
                T, O = (nm == r["target"]) & ~z["support"], (nm == o) & ~z["support"]
                from scipy.spatial import cKDTree
                d = cKDTree(z["points"][O]).query(z["points"][T])[0].min() * 1000 if O.any() and T.any() else np.nan
                xs.append(r["mesh_gap_mm"][o]); ys.append(round(float(d), 2))
        ax.scatter(xs, ys, c=c, marker=mk, s=28, label=lab, alpha=0.85)
        side[f"gaps_{'merged' if merged else 'not_merged'}"] = [{"key": k, "mesh_gap_mm": x, "cloud_gap_mm": y} for k, x, y in zip(ks, xs, ys)]
    for y, s in ((10, L("fusion voxel 10", "fusion voxel 10")), (15, L("split radius 15", "split radius 15")), (30, L("DBSCAN eps 30", "DBSCAN eps 30"))):
        ax.axhline(y, color=FS.GRID_INK, lw=1, ls="--"); ax.text(1, y + 0.6, s, fontsize=7, color=FS.INK_2)
    ax.set_xlabel(L("GT mesh 최소 틈 target–최근접 과일 [mm]", "GT mesh min gap target–nearest fruit [mm]"))
    ax.set_ylabel(L("관측 점군 최소 틈 [mm]", "observed cloud min gap [mm]"))
    ax.set_title(L("(a) 합쳐짐 = 관측 틈 < 30 mm (connectivity)", "(a) merged = observed gap < 30 mm (connectivity)"), fontsize=10)
    ax.legend(fontsize=8)
    # (b) split ratio GT vs VLA
    ax = axes[1]
    mk = [k for k in keys if G[k]["e2_merged"]]
    order = sorted(mk, key=lambda k: G[k]["cloud_gap_mm"])
    def rv(r):
        x = r["split_recorded"]["ratio"]
        return np.nan if x is None else (1e4 if x == "inf" else float(x))
    xg = [rv(G[k]) for k in order]; xv = [rv(V[k]) for k in order]
    idx = np.arange(len(order))
    ax.scatter(idx, xg, c=C_O, marker="s", s=22, label="GT Gaussian")
    ax.scatter(idx, xv, c=C_T, marker="o", s=22, label="VLA")
    ax.axhline(2.0, color=FS.INK_2, lw=1, ls="--"); ax.text(0, 2.15, "attention_split_ratio 2.0", fontsize=7, color=FS.INK_2)
    for i, k in enumerate(order):
        if G[k]["split_recorded"]["reason"] == "one_component":
            ax.axvspan(i - 0.4, i + 0.4, color=FS.SEQ_BLUE[0], alpha=0.6, lw=0)
    ax.set_yscale("log"); ax.set_ylim(0.3, 2e4)
    ax.set_xlabel(L("merged frame (관측 틈 오름차순) · 음영 = 15 mm 재군집이 1 성분", "merged frame (by observed gap) · shaded = one 15 mm component"))
    ax.set_ylabel(L("split ratio (target 성분 / 다른 성분 평균 attention)", "split ratio (mean attention, target comp / others)"))
    ax.set_title(L("(b) ratio ≥ 2: GT 합성 0/26, VLA 20/26 (음영 8 = 1 성분)", "(b) ratio >= 2: GT Gaussian 0/26, VLA 20/26 (8 shaded = one component)"), fontsize=10)
    ax.legend(fontsize=8)
    side["ratio"] = [{"key": k, "gt": G[k]["split_recorded"]["ratio"], "vla": V[k]["split_recorded"]["ratio"],
                      "gt_reason": G[k]["split_recorded"]["reason"], "vla_reason": V[k]["split_recorded"]["reason"]} for k in order]
    # (c) candidates, VLA attention, att selection
    ax = axes[2]
    cv = A["vla"]["candidates"]
    y = np.arange(len(CANDS))[::-1]
    ok = [cv[c]["att"]["merged_pure_and_complete"] for c, _ in CANDS]
    orc = [cv[c]["oracle"]["merged_pure_and_complete"] for c, _ in CANDS]
    wr = [W["vla"][c]["att"]["merged_wrong_pick"] for c, _ in CANDS]
    nm_ok = [cv[c]["att"]["nonmerged_complete"] for c, _ in CANDS]
    ax.barh(y, orc, color=FS.SEQ_BLUE[1], label=L("성분 oracle 선택 (분할 상한)", "oracle component (segmentation bound)"))
    ax.barh(y, ok, color=C_T, height=0.5, label=L("attention mass 선택 (pipeline 규칙)", "attention-mass pick (pipeline rule)"))
    for yi, a, w, n in zip(y, ok, wr, nm_ok):
        ax.text(a + 0.3, yi - 0.1, f"{a}/34" + (f" · {L('오선택', 'wrong')} {w}" if w else "") + (f" · {L('비병합 손상', 'non-merged broken')} {30 - n}" if n < 30 else ""), fontsize=7, color=FS.INK)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in CANDS], fontsize=8)
    ax.axvline(cv[CANDS[0][0]]["att"]["merged_pure_and_complete"], color=FS.INK_2, lw=0.8, ls=":")
    ax.set_xlim(0, 37)
    ax.set_xlabel(L("merged 34 frame 중 target 성분 precision ≥ 0.95 · recall ≥ 0.90", "of 34 merged: target segment precision >= 0.95 and recall >= 0.90"))
    ax.set_title(L("(c) 후보 offline 시험 (VLA attention)", "(c) offline candidates (VLA attention)"), fontsize=10)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    side["candidates_vla"] = {c: {"att_pure_complete": a, "oracle_pure_complete": o, "wrong_pick": w, "nonmerged_complete": n}
                              for (c, _), a, o, w, n in zip(CANDS, ok, orc, wr, nm_ok)}
    fig.tight_layout()
    fig.savefig(FIG / "t44m-graph.png", dpi=130)
    (FIG / "t44m-graph.json").write_text(json.dumps(side, indent=1, ensure_ascii=False))


def fig_table():
    dg, dv = A["gt"]["decomposition"], A["vla"]["decomposition"]
    rows1 = [
        [L("merged frame (V-E2 정의)", "merged frames (V-E2 def.)"), "34/64", "34/64"],
        [L("관측 점 틈 < DBSCAN eps 30 mm", "cloud gap < DBSCAN eps 30 mm"), f"{dg['cloud_gap_mm']['n_lt_30']}/34 (8.9–29.0 mm)", "same geometry"],
        [L("GT mesh 틈 < fusion voxel 10 mm (닿음)", "mesh gap < fusion voxel 10 mm (touching)"), f"{dg['mesh_gap_mm']['n_lt_10']}/34", "–"],
        [L("관측 점 틈 < split radius 15 mm", "cloud gap < split radius 15 mm"), f"{dg['cloud_gap_mm']['n_lt_15']}/34", "–"],
        [L("split trigger (narrowest ≤ 71.5 < largest)", "split trigger (narrowest <= 71.5 < largest)"), f"{dg['trigger']['n_triggered_now']}/34", f"{dv['trigger']['n_triggered_now']}/34"],
        [L("15 mm 재군집 ≥ 2 성분 (모두 순수)", "15 mm recluster >= 2 comps (all pure)"), f"{dg['recluster15']['n_ge2_components']}/34", f"{dv['recluster15']['n_ge2_components']}/34"],
        [L("ratio ≥ 2 (norm attention)", "ratio >= 2 (norm attention)"), f"{dg['recorded_ratio_norm']['n_ge_2']}/{dg['recorded_ratio_norm']['n']} (med {dg['recorded_ratio_norm']['median']})",
         f"{dv['recorded_ratio_norm']['n_ge_2']}/{dv['recorded_ratio_norm']['n']} (med {dv['recorded_ratio_norm']['median']})"],
        [L("기존 split 로 해결", "resolved by existing split"), "0/34", f"{dv['cause_classes'].get('resolved by the existing split', 0)}/34"],
        [L("남은 원인: 기하 (15 mm 1 성분)", "left: geometry (one 15 mm comp)"), f"{dg['cause_classes'].get('geometry: no clean 15 mm component (gap < split radius)', 0)}",
         f"{dv['cause_classes'].get('geometry: no clean 15 mm component (gap < split radius)', 0)}"],
        [L("남은 원인: attention ratio < 2", "left: attention ratio < 2"), f"{dg['cause_classes'].get('attention: components separate, ratio < 2', 0)}",
         f"{dv['cause_classes'].get('attention: components separate, ratio < 2', 0)}"],
        [L("chosen cluster = target 쪽 cluster (64)", "chosen cluster = target-holding cluster (64)"), f"{dg['chosen_all']['n_is_target_cluster']}/64", f"{dv['chosen_all']['n_is_target_cluster']}/64"],
    ]
    cg, cv = A["gt"]["candidates"], A["vla"]["candidates"]
    rows2 = []
    for c, lab in CANDS:
        rows2.append([lab, f"{cg[c]['att']['merged_pure_and_complete']}", f"{cv[c]['att']['merged_pure_and_complete']}",
                      f"{cv[c]['oracle']['merged_pure_and_complete']}", f"{W['vla'][c]['att']['merged_wrong_pick']}",
                      f"{cv[c]['att']['nonmerged_complete']}/30"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.5), gridspec_kw={"height_ratios": [1.05, 1.0]})
    FS.style_axes(fig, axes)
    for ax in axes:
        ax.axis("off")
    t1 = axes[0].table(cellText=rows1, colLabels=[L("단계 / 값", "stage / value"), "GT Gaussian attention", "VLA attention"],
                       loc="center", cellLoc="left", colWidths=[0.5, 0.25, 0.25])
    t2 = axes[1].table(cellText=rows2, colLabels=[L("후보", "candidate"), "GT: att", "VLA: att", "VLA: oracle", L("VLA 오선택", "VLA wrong pick"),
                                                  L("VLA 비병합 유지", "VLA non-merged kept")],
                       loc="center", cellLoc="left", colWidths=[0.38, 0.1, 0.1, 0.12, 0.13, 0.17])
    for t in (t1, t2):
        t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.35)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor(FS.GRID_INK)
            if r == 0:
                cell.set_facecolor(FS.SEQ_BLUE[0])
    axes[0].set_title(L("merge 원인 분해 (E2-demo 64 frame, 같은 장면 · 같은 pipeline, attention 출처만 다름)",
                        "merge cause decomposition (E2-demo 64 frames, same scene and pipeline, attention source only differs)"), fontsize=10)
    axes[1].set_title(L("후보 offline 시험: merged 34 중 target 성분 precision ≥ 0.95 · recall ≥ 0.90 (att = attention mass 선택)",
                        "offline candidates: of 34 merged, target segment precision >= 0.95 and recall >= 0.90 (att = attention-mass pick)"), fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "t44m-table.png", dpi=130)
    (FIG / "t44m-table.json").write_text(json.dumps({"decomposition": rows1, "candidates": rows2}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    fig_scene(); fig_graph(); fig_table()
    print("ok")
