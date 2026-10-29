"""T38 Phase B verifier figures -> AG3S/ag3s/docs/figures/t38/t38b-*.png (+ .json sidecar).
(1) t38b-stage-chain     stacked per-stage medians, five codes, chain C (GPU alone) + chains A/B totals
(2) t38b-chunk-total     per-seq total (AG3S+TO), five codes, chain C, 533 ms line
(3) t38b-equivalence     V1 table (items x recordings) + V5 column
(4) t38b-scene-mask      E3b_ep1800_r2 seq 18 (latched, one chunk before closing): depth + robot mask + kept voxel pixels,
                         5c72d37 vs b4f06ec, and the XOR difference maps
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle

B = pathlib.Path(_ag3s_workspace_path('outputs/verify/T38/B'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t38'))
CAT = figstyle.CATEGORICAL
BUDGET = 533.0
figstyle.use_korean()
CODES = [("tag", "b4f06ec (tag)"), ("b1", "1a35423 B1+B5"), ("b2", "305dca1 B2"), ("b3", "044ecc4 B3"), ("b4", "5c72d37 B4")]


def save(fig, name, data):
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight", facecolor=figstyle.SURFACE)
    (FIG / f"{name}.json").write_text(json.dumps(data, indent=1, default=float))
    plt.close(fig)
    print("wrote", p)


def stage_chain():
    S = {ch: json.loads((B / "cmp" / f"v2{ch}_summary.json").read_text()) for ch in "abc"}
    segs = [("scene_reconstruction", "scene_reconstruction"), ("support_surface", "support_surface"),
            ("target_grounding", "target_grounding"), ("esdf", "esdf (cuRobo)"),
            ("constraint_generation_minus_esdf", "constraint_generation - esdf"), ("timing.trajopt", "TO (trajopt)"),
            ("ag3s_minus_stages", "other AG3S (attention_lifting, glue)")]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    figstyle.style_axes(fig, ax)
    x = np.arange(len(CODES))
    data = {"source": "outputs/verify/T38/B/cmp/v2{a,b,c}_summary.json", "bars": "chain c (GPU alone), median per stage, seq 2-75",
            "codes": {}}
    for i, (k, lab) in enumerate(CODES):
        o = S["c"][f"v2c_{k}"]
        bottom = 0.0
        d = {}
        for j, (key, name) in enumerate(segs):
            v = max(o[key]["median"], 0.0)
            ax.bar(x[i], v, 0.55, bottom=bottom, color=CAT[j], label=name if i == 0 else None, edgecolor=figstyle.SURFACE, lw=0.5)
            bottom += v; d[key] = round(o[key]["median"], 1)
        tm, tp = o["timing.total"]["median"], o["timing.total"]["p90"]
        ax.plot([x[i] + 0.33], [tm], marker="D", color=figstyle.INK, ms=5, label="total median (chain c)" if i == 0 else None)
        ax.plot([x[i] + 0.33], [tp], marker="_", color=figstyle.INK, ms=12, mew=1.5, label="total p90 (chain c)" if i == 0 else None)
        for ch, mk in (("a", "o"), ("b", "s")):
            ax.plot([x[i] + 0.42], [S[ch][f"v2{ch}_{k}"]["timing.total"]["median"]], marker=mk, mfc="none", color=figstyle.INK_2, ms=5,
                    label=f"total median chain {ch} (GPU shared)" if i == 0 else None)
        ax.text(x[i], max(bottom, tp) + 60, f"{tm:,.0f} ms\n<=533: {o['chunks_total_le_533ms']}", ha="center", fontsize=8, color=figstyle.INK)
        d.update({"total_median": round(tm, 1), "total_p90": round(tp, 1), "le_533": o["chunks_total_le_533ms"],
                  "total_median_chain_a": round(S["a"][f"v2a_{k}"]["timing.total"]["median"], 1),
                  "total_median_chain_b": round(S["b"][f"v2b_{k}"]["timing.total"]["median"], 1)})
        data["codes"][lab] = d
    ax.axhline(BUDGET, color=CAT[7], lw=1, ls="--")
    ax.text(0.5, BUDGET + 30, "chunk period 533 ms", color=CAT[7], fontsize=8, ha="center", bbox=dict(fc=figstyle.SURFACE, ec="none", pad=1))
    ax.set_xticks(x); ax.set_xticklabels([c[1] for c in CODES], fontsize=8)
    ax.set_ylabel("ms per chunk (stacked stage medians)")
    ax.set_ylim(0, 3700)
    ax.set_title("T38 Phase B replay E3b_ep1807_r2 seq 2-75: per-stage median by commit (chain c, GPU alone)", fontsize=10, color=figstyle.INK)
    ax.legend(fontsize=7, loc="upper right", ncol=2, frameon=False)
    save(fig, "t38b-stage-chain", data)


def chunk_total():
    S = json.loads((B / "cmp" / "v2c_summary.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 4.8))
    figstyle.style_axes(fig, ax)
    data = {"source": "outputs/verify/T38/B/cmp/v2c_summary.json (total_by_seq = SafePolicy timing total, AG3S+TO)", "series": {}}
    for i, (k, lab) in enumerate(CODES):
        tb = S[f"v2c_{k}"]["total_by_seq"]
        seqs = sorted(int(s) for s in tb); vals = [tb[str(s)] for s in seqs]
        ax.plot(seqs, vals, color=CAT[i], lw=1.3, label=f"{lab}  median {np.median(vals):,.0f} ms")
        data["series"][lab] = {str(s): round(v, 1) for s, v in zip(seqs, vals)}
    ax.axhline(BUDGET, color=CAT[7], lw=1, ls="--"); ax.text(75, BUDGET * 1.06, "533 ms", color=CAT[7], fontsize=8, ha="right")
    ax.set_yscale("log"); ax.set_ylim(150, 6000)
    ax.set_yticks([200, 300, 533, 1000, 2000, 4000]); ax.set_yticklabels(["200", "300", "533", "1000", "2000", "4000"])
    ax.set_xlabel("planning chunk seq (E3b_ep1807_r2)"); ax.set_ylabel("AG3S + TO ms (log)")
    for s, lab, yy in ((3, "latched", 4500), (15, "closing", 4500), (16, "held", 3900), (29, "placed", 4500)):
        ax.axvline(s, color=figstyle.GRID_INK, lw=0.8); ax.text(s + 0.3, yy, lab, fontsize=7, color=figstyle.INK_2)
    ax.set_title("per-chunk total time, five commits, chain c (GPU alone), seq 2-75", fontsize=10, color=figstyle.INK)
    ax.legend(fontsize=7, loc="center right", frameon=False)
    save(fig, "t38b-chunk-total", data)


def equivalence():
    recs = ["ep1800_r1", "ep1808_r1", "ep1800_r2", "ep1807_r1", "ep1807_r2"]
    C = {r: json.loads((B / "cmp" / f"v1_{r}.json").read_text()) for r in recs}
    V5 = json.loads((B / "cmp" / "v5_ep1800_r2_tag_vs_b4cpu.json").read_text())
    cov = json.loads((B / "cmp" / "coverage_v1.json").read_text())
    items = [("actions_bytes", "refined actions (float32 bytes)"), ("verdict", "verdict (safe, status, certified, reasons)"),
             ("worst_row", "worst row (max_violation_pair)"), ("max_violation_m", "max_violation_m"), ("grasp", "grasp record (non-time)"),
             ("has_target", "has_target"), ("ag3s_status", "AG3S status"), ("manipulated", "manipulated object (id, state, centroid, geometry)"),
             ("target_points", "target points"), ("esdf_tiers", "ESDF tiers (distance + label sha256, grid)"),
             ("fused_cloud", "fused cloud (7 CSR arrays + attention)"), ("per_camera_cloud", "per-camera voxel cloud (points, uv, attention)"),
             ("depth_masks", "per-camera depth robot mask"), ("response_non_time", "full response, non-time leaves")]
    cols = [f"E3b_{r}" for r in recs] + ["V5: 5c72d37 cpu\nvs tag, ep1800_r2"]
    cells, data = [], {"source": "outputs/verify/T38/B/cmp/v1_*.json, v5_ep1800_r2_tag_vs_b4cpu.json", "rows": {}}
    for key, name in items:
        row = [f"{C[r]['same_count'][key]}/{C[r]['n_chunks']}" for r in recs] + [f"{V5['same_count'][key]}/{V5['n_chunks']}"]
        cells.append(row); data["rows"][name] = dict(zip(cols, row))
    extra = [("mask pixels XOR (sum over camera pairs)", lambda c: f"{c['depth_mask_pixels']['xor_px_total']} px / {c['depth_mask_pixels']['pairs']} pairs"),
             ("record npz chunks, all arrays byte-equal", lambda c: f"{c['record_npz']['chunks_arrays_identical']}/{c['record_npz']['n_b']}"),
             ("record arrays compared", lambda c: f"{c['record_npz']['n_arrays_compared']}"),
             ("record summary_json, non-time leaves", lambda c: f"{c['record_npz']['summary_non_time_identical']}/{c['record_npz']['n_b']}")]
    for name, f in extra:
        row = [f(C[r]) for r in recs] + [f(V5)]
        cells.append(row); data["rows"][name] = dict(zip(cols, row))
    row = [", ".join(f"{k[:4]} {v}" for k, v in cov[r]["replay_grasp_states"].items()) for r in recs] + ["seq 1-40"]
    cells.append(row); data["rows"]["grasp states replayed"] = dict(zip(cols, row))
    rows = [n for _, n in items] + [n for n, _ in extra] + ["grasp states replayed"]
    fig, ax = plt.subplots(figsize=(17, 7.6))
    fig.patch.set_facecolor(figstyle.SURFACE); ax.axis("off")
    t = ax.table(cellText=cells, rowLabels=rows, colLabels=cols, loc="center", cellLoc="center", rowLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 1.55)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        if r == 0:
            cell.set_facecolor(SEQ0); cell.set_text_props(color=figstyle.INK)
        elif c == -1:
            cell.set_facecolor(figstyle.SURFACE)
        else:
            cell.set_facecolor(figstyle.SURFACE)
    ax.set_title("T38 V1 / V5: b4f06ec (tag) vs 5c72d37, replay pass T (seq 1-75 after warm 4), verifier's compare.py", fontsize=10, color=figstyle.INK)
    save(fig, "t38b-equivalence", data)


SEQ0 = figstyle.SEQ_BLUE[0]


def scene_mask(run="ep1800_r2", seq=18):
    A = np.load(B / "runs" / f"v1_b4_{run}" / "keep_T.npz"); T = np.load(B / "runs" / f"v1_tag_{run}" / "keep_T.npz")
    cams = ["head", "left_wrist", "right_wrist"]
    fig, axes = plt.subplots(3, 4, figsize=(17, 10.2))
    figstyle.style_axes(fig, axes)
    k = f"seq{seq:03d}"
    data = {"run": f"E3b_{run}", "seq": seq, "state": "latched (seq 19 = closing)", "source": [
        f"outputs/verify/T38/B/runs/v1_b4_{run}/keep_T.npz", f"outputs/verify/T38/B/runs/v1_tag_{run}/keep_T.npz"], "cameras": {}}
    cmap = figstyle.sequential_cmap()
    for i, cam in enumerate(cams):
        d = A[f"{k}/depth/{cam}"]; dt = T[f"{k}/depth/{cam}"]
        mg, mt = A[f"{k}/mask/{cam}"], T[f"{k}/mask/{cam}"]
        ug, ut = A[f"{k}/uv/{cam}"], T[f"{k}/uv/{cam}"]
        H, W = d.shape
        kg = np.zeros((H, W), bool); kt = np.zeros((H, W), bool)
        kg[ug[:, 1], ug[:, 0]] = True; kt[ut[:, 1], ut[:, 0]] = True
        for j, (dd, mm, kk, name) in enumerate(((d, mg, kg, "5c72d37 GPU path"), (dt, mt, kt, "b4f06ec tag path"))):
            ax = axes[i, j]
            ax.imshow(np.where(dd > 0, dd, np.nan), cmap=cmap, vmin=np.nanpercentile(np.where(dd > 0, dd, np.nan), 1), vmax=np.nanpercentile(np.where(dd > 0, dd, np.nan), 99))
            ov = np.zeros((H, W, 4)); ov[mm] = matplotlib.colors.to_rgba(CAT[1], 0.55)
            ax.imshow(ov)
            yy, xx = np.nonzero(kk); ax.scatter(xx[::7], yy[::7], s=0.15, color=CAT[2], alpha=0.8)
            ax.set_title(f"{cam} - {name}\nrobot mask {int(mm.sum()):,} px (orange), kept voxels {int(kk.sum()):,} (green)", fontsize=8, color=figstyle.INK)
            ax.set_xticks([]); ax.set_yticks([])
        xm = mg ^ mt; xk = kg ^ kt; xd = int((d != dt).sum())
        ax = axes[i, 2]; ax.imshow(xm, cmap=matplotlib.colors.ListedColormap([figstyle.SURFACE, CAT[7]]), vmin=0, vmax=1)
        ax.set_title(f"{cam}: robot mask XOR = {int(xm.sum())} px\n(depth input differing px = {xd})", fontsize=8, color=figstyle.INK)
        ax.set_xticks([]); ax.set_yticks([])
        ax = axes[i, 3]; ax.imshow(xk, cmap=matplotlib.colors.ListedColormap([figstyle.SURFACE, CAT[7]]), vmin=0, vmax=1)
        pd = "equal" if (ug.shape == ut.shape and np.array_equal(ug, ut) and np.array_equal(A[f"{k}/pts/{cam}"], T[f"{k}/pts/{cam}"])) else "DIFFER"
        ax.set_title(f"{cam}: kept-voxel pixel XOR = {int(xk.sum())} px\n(uv + points arrays {pd})", fontsize=8, color=figstyle.INK)
        ax.set_xticks([]); ax.set_yticks([])
        for a_ in (axes[i, 2], axes[i, 3]):
            for sp_ in a_.spines.values():
                sp_.set_visible(True); sp_.set_color(figstyle.INK_2)
        data["cameras"][cam] = {"robot_mask_px_gpu": int(mg.sum()), "robot_mask_px_tag": int(mt.sum()), "mask_xor_px": int(xm.sum()),
                                "kept_px_gpu": int(kg.sum()), "kept_px_tag": int(kt.sum()), "kept_xor_px": int(xk.sum()),
                                "depth_input_differing_px": xd, "uv_points_arrays": pd, "n_kept_points": int(len(ug))}
    fig.suptitle(f"E3b_{run} seq {seq} (latched, one chunk before closing): per-camera depth, self-filter robot mask, kept voxel pixels; "
                 "5c72d37 vs b4f06ec and difference maps", fontsize=10, color=figstyle.INK)
    save(fig, "t38b-scene-mask", data)


if __name__ == "__main__":
    for name in ("t38b-stage-chain", "t38b-chunk-total", "t38b-equivalence", "t38b-scene-mask"):
        assert not (FIG / f"{name}.png").exists() or "--overwrite-own" in sys.argv, f"{name} exists"
    stage_chain(); chunk_total(); equivalence(); scene_mask()
