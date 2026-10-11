"""T38 figures from summary_<run>.json: stage bars (graph), per-stage function table, input-size table, pipeline map.
writes AG3S/ag3s/docs/figures/t38/t38-{stage-bars,function-table,input-sizes,pipeline-map}.png + .json
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-o4")
from AG3S.ag3s.experiments.common import figstyle

run = sys.argv[1] if len(sys.argv) > 1 else "E3b_ep1807_r2"
label = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T38'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t38')); FIG.mkdir(parents=True, exist_ok=True)
S = json.loads((OUT / f"summary_{run}{label}.json").read_text())
figstyle.use_korean()
CAT = figstyle.CATEGORICAL
BUDGET = 533.0


def save(fig, name, data):
    fig.savefig(FIG / f"{name}.png", dpi=130, bbox_inches="tight", facecolor=figstyle.SURFACE)
    (FIG / f"{name}.json").write_text(json.dumps(data, indent=1, default=float))
    plt.close(fig)


# ------------------------------------------------------------------ (2) stage bars
def stage_bars(passname="pass_T"):
    P = S[passname]
    st, tm = P["stages_profile_ms_seq2_75"], P["timing_seq2_75"]
    rec = S["recorded_all_449"]["profile_ms"]; rto = S["recorded_all_449"]["to_timing_ms"]
    rows = [  # label, replay stat, recorded stat
        ("scene_reconstruction", st["scene_reconstruction"], rec["scene_reconstruction"]),
        ("support_surface", st["support_surface"], rec["support_surface"]),
        ("target_grounding", st["target_grounding"], rec["target_grounding"]),
        ("constraint_generation\n minus esdf", st["constraint_generation_minus_esdf"], rec.get("constraint_generation_minus_esdf")),
        ("esdf (cuRobo)", st["esdf"], rec["esdf"]),
        ("TO linearize", tm["to_linearize"], rto.get("linearize")),
        ("TO check", tm["to_check"], rto.get("check")),
        ("TO assemble", tm["to_assemble"], rto.get("assemble")),
        ("TO qp", tm["to_qp"], rto.get("qp")),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1, 1.5]})
    y = np.arange(len(rows))[::-1]
    for i, (lab, r, rr) in enumerate(rows):
        a.barh(y[i] + 0.2, r["median"], 0.38, color=CAT[0], label="replay (worktree, pass T) median" if i == 0 else None)
        a.errorbar(r["p90"], y[i] + 0.2, fmt="|", color=figstyle.INK_2, ms=8)
        a.text(max(r["median"], r["p90"]) + 15, y[i] + 0.2, f"{r['median']:.0f} (p90 {r['p90']:.0f})", va="center", fontsize=7, color=figstyle.INK)
        if rr is not None:
            a.barh(y[i] - 0.2, rr["median"], 0.38, color=CAT[1], label="recorded T34 E3b server (450 chunks) median" if i == 0 else None)
            a.text(rr["median"] + 15, y[i] - 0.2, f"{rr['median']:.0f}", va="center", fontsize=7, color=figstyle.INK)
    a.axvline(BUDGET, color=CAT[7], lw=1, ls="--"); a.text(BUDGET + 10, y[0] + 0.55, "chunk period 533 ms", color=CAT[7], fontsize=7)
    a.set_yticks(y); a.set_yticklabels([r[0] for r in rows], fontsize=8)
    a.set_xlabel("ms per chunk (bar = median, tick = p90)")
    a.legend(fontsize=7, loc="lower right")
    tot = tm["ag3s_plus_trajopt"]["median"]
    a.set_title(f"per-stage wall time, seq 2-75 (seq 1 = new mapper, separate)\nAG3S + TO median {tot:.0f} ms vs 533 ms period", fontsize=9)
    # per chunk stacked
    pc = [r for r in P["per_chunk"]]
    seqs = np.array([r["seq"] for r in pc])
    comp = [("scene_reconstruction", lambda r: r["profile_ms"]["scene_reconstruction"]),
            ("support_surface", lambda r: r["profile_ms"]["support_surface"]),
            ("target_grounding", lambda r: r["profile_ms"]["target_grounding"]),
            ("constraint_generation - esdf", lambda r: r["profile_ms"]["constraint_generation"] - r["profile_ms"]["esdf"]),
            ("esdf", lambda r: r["profile_ms"]["esdf"]),
            ("TO (trajopt)", lambda r: r["timing_ms"]["trajopt"])]
    bottom = np.zeros(len(pc)); data = {}
    for i, (lab, fn) in enumerate(comp):
        v = np.array([fn(r) for r in pc]); data[lab] = v.tolist()
        b.bar(seqs, v, bottom=bottom, color=CAT[i], width=0.85, label=lab); bottom += v
    b.axhline(BUDGET, color=CAT[7], lw=1, ls="--")
    phases = [r["phase_label"] for r in pc]; grasp = [r["grasp_state"] for r in pc]
    for k in range(1, len(pc)):
        if grasp[k] != grasp[k - 1]:
            b.axvline(seqs[k] - 0.5, color=figstyle.GRID_INK, lw=0.8)
            b.text(seqs[k] - 0.3, bottom.max() * 1.02, grasp[k], fontsize=6, rotation=90, va="bottom", color=figstyle.INK_2)
    b.text(1.2, bottom.max() * 1.02, grasp[0], fontsize=6, rotation=90, va="bottom", color=figstyle.INK_2)
    b.set_xlabel("planning seq (E3b_ep1807_r2, 75 chunks; labels = grasp latch state)"); b.set_ylabel("ms")
    b.set_title("per-chunk stacked stage time (replay pass T, no extra sync)", fontsize=9)
    b.legend(fontsize=7, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    b.set_ylim(0, bottom.max() * 1.18)
    figstyle.style_axes(fig, [a, b])
    save(fig, "t38-stage-bars", {"pass": passname, "rows": [{"label": r[0], "replay": r[1], "recorded": r[2]} for r in rows],
                                 "ag3s_plus_trajopt": tm["ag3s_plus_trajopt"], "per_chunk_seq": seqs.tolist(), "per_chunk_ms": data,
                                 "budget_ms": BUDGET, "source": f"outputs/verify/T38/summary_{run}{label}.json"})


# ------------------------------------------------------------------ (3a) function table
def function_table():
    cp = S.get("cprofile") or {}
    order = [k for k in ("scene_reconstruction", "support_surface", "target_grounding", "constraint_generation", "whole_infer") if k in cp]
    if not order:
        return
    n_top = 12
    fig, axes = plt.subplots(len(order), 1, figsize=(15, 3.6 * len(order)), gridspec_kw={"hspace": 0.45})
    axes = np.atleast_1d(axes); data = {}
    for ax, k in zip(axes, order):
        t = cp[k]
        src = t["top_cumulative"] if k != "whole_infer" else t["top_tottime"]
        rows = [r for r in src if r["device"] != "harness wrapper"][:n_top]
        cell = [[r["function"][-95:], f"{r['cumtime_ms']:.1f}", f"{r['tottime_ms']:.1f}", f"{r['ncalls_per_chunk']:.0f}", r["device"]] for r in rows]
        ax.axis("off")
        tb = ax.table(cellText=cell, colLabels=["function (file:line)", "cum ms/chunk", "self ms/chunk", "calls/chunk", "where"],
                      colWidths=[0.62, 0.08, 0.08, 0.08, 0.14], loc="center", cellLoc="left")
        tb.auto_set_font_size(False); tb.set_fontsize(6.5); tb.scale(1, 1.05)
        for (i, j), c in tb.get_celld().items():
            c.set_edgecolor(figstyle.GRID_INK)
            if i == 0:
                c.set_facecolor("#eeeeea")
            elif j == 4 and "GPU" in c.get_text().get_text():
                c.set_facecolor(figstyle.SEQ_BLUE[0])
        dev = t["self_time_by_device_ms_per_chunk"]
        devs = ", ".join(f"{a} {b:.0f}" for a, b in list(dev.items())[:5])
        ax.set_title(f"{k}: top {n_top} by {'cumulative' if k != 'whole_infer' else 'self'} time, mean of {t['n_profiles']} selected chunks (cProfile)\n"
                     f"self time by where (ms/chunk): {devs}", fontsize=8, loc="left", pad=14)
        data[k] = {"rows": rows, "self_time_by_device_ms_per_chunk": dev, "files": t["files"]}
    fig.text(0.01, 0.005, "cProfile, pass P (per top-level stage) and pass W (whole SafePolicy.infer), quiet run _q2, chunks seq 2,10,15,16,22,28,40. "
             "Cython cKDTree methods do not appear in cProfile; their workers=-1 thread fan-out appears as threading.* / _thread.lock rows.",
             fontsize=7, color=figstyle.INK_2)
    fig.patch.set_facecolor(figstyle.SURFACE)
    save(fig, "t38-function-table", data)


# ------------------------------------------------------------------ (3b) input sizes
def input_sizes(selected=(2, 10, 15, 16, 22, 28, 40)):
    C = {c["seq"]: c for c in S.get("census") or []}
    if not C:
        return
    hdr = ["seq", "grasp", "depth px\n(3 cams)", "back-proj.\npoints", "voxelised\n(5 mm)", "self-filt.\nremoved",
           "fused\n(fusion vox.)", "support\npoints", "seeds", "grow in ->\ngrown", "DBSCAN\ncalls / in", "clusters",
           "KD builds\n(points)", "KD queries", "ESDF tiers\n(voxels)", "TSDF\nblocks", "H2D / D2H\nMB"]
    rows = []; data = []
    for q in selected:
        c = C.get(q)
        if c is None:
            continue
        sz = c["sizes"]; fs = c["fn_sizes"]
        pcr = sz["per_camera_recon"]
        raw = sum(v["n_raw"] for v in pcr.values()); vox = sum(v["n_voxel"] for v in pcr.values())
        gr = (fs.get("target_grounding.grow_region") or [{}])[0]
        db = fs.get("target_grounding.dbscan") or []
        gt = (fs.get("target_grounding.ground_target") or [{}])[0]
        tiers = len(sz["esdf_tiers"] or []) + (1 if any("_build_tiers" in k for k in []) else 0)
        nq = sum(v for k, v in c["kd_queries_by_site"].items())
        tf = c["fn_calls"].get("CuroboFieldBuilder._build_tiers", 0)
        r = [q, c["grasp_state"], f"{3 * 640 * 480:,}", f"{raw:,}", f"{vox:,}", f"{sz['n_self_filtered']:,}",
             f"{sz['n_points_fused']:,}", f"{sz['n_support_points']:,}", f"{gt.get('n_seeds', '')}",
             f"{gr.get('n_in', '')} -> {gr.get('n_grown', '')}", f"{len(db)} / {db[0]['n_in'] if db else ''}",
             f"{gt.get('n_clusters', '')}", f"{c['kd_builds']} ({c['kd_build_points_total']:,})", f"{nq:,}",
             f"{1 + tf} ({(1 + tf) * 128 ** 3:,})", f"{sz['esdf_n_active_blocks']:,}",
             f"{c['h2d_bytes'] / 1e6:.1f} / {c['d2h_bytes'] / 1e6:.1f}"]
        rows.append([str(x) for x in r])
        data.append({"seq": q, "grasp": c["grasp_state"], "depth_px": 3 * 640 * 480, "backprojected": raw, "voxelised": vox,
                     "per_camera": pcr, "self_filtered": sz["n_self_filtered"], "fused": sz["n_points_fused"],
                     "support_points": sz["n_support_points"], "seeds": gt.get("n_seeds"), "grow_region": gr,
                     "dbscan": db, "clusters": gt.get("n_clusters"), "kd_builds": c["kd_builds"],
                     "kd_build_points": c["kd_build_points_total"], "kd_builds_by_site": c["kd_builds_by_site"],
                     "kd_build_sizes_by_site": c["kd_build_sizes_by_site"], "kd_queries_by_site": c["kd_queries_by_site"],
                     "build_tiers_calls": tf, "esdf_tiers": sz["esdf_tiers"], "tsdf_active_blocks": sz["esdf_n_active_blocks"],
                     "h2d_bytes": c["h2d_bytes"], "d2h_bytes": c["d2h_bytes"], "copies": c["copies"],
                     "kd_same_input_built_more_than_once": c["kd_same_input_built_more_than_once"]})
    fig, ax = plt.subplots(figsize=(17, 0.5 + 0.42 * (len(rows) + 2)))
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 2.0)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(figstyle.GRID_INK)
        if i == 0:
            c.set_facecolor("#eeeeea")
    ax.set_title(f"T38 input sizes per stage, selected chunks of {run} (census pass C). 3 cameras every chunk. "
                 "ESDF: coarse 20 mm + fine 5 mm + target-free fine 5 mm (until attach), each 128^3 = 2,097,152 voxels",
                 fontsize=8, loc="left")
    fig.patch.set_facecolor(figstyle.SURFACE)
    save(fig, "t38-input-sizes", {"rows": data, "header": hdr})


if __name__ == "__main__":
    stage_bars(); function_table(); input_sizes()
    print("ok")
