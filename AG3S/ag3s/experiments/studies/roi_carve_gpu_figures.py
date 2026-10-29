"""AG3S-ROI-CARVE-GPU figures (rule A): where the field stage's time went and what the GPU judgement saves (graph),
the pre-fixed verdicts (table), and the episode-1345 neighbour contact against the recorded sphere model (scene).

  MUJOCO_GL=osmesa .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.studies.roi_carve_gpu_figures \
      --verify <run>/verify/AG3S-ROI-CARVE-GPU --tdm-run <AG3S-TASK-DYNAMIC-MAP run> --smoke <smoke.json> \
      --out AG3S/ag3s/docs/figures/roi-carve-gpu
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import pathlib

import numpy as np

KITS = (("AG3S-TASK-DYNAMIC-MAP", None), ("AG3S-TASK-DYNAMIC-MAP-E3", "B_ROI_FP"), ("AG3S-TASK-DYNAMIC-MAP-E4", "B_ROI_FPR"))


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def server_costs(tdm_run: pathlib.Path) -> dict:
    """Per configuration: per-chunk `profile_ms.esdf` and `esdf_stats.roi_carve.ms` (target-layer chunks left out)."""
    out: dict = {}
    for kit, conf in KITS:
        for f in sorted(glob.glob(str(tdm_run / "verify" / kit / "runs" / "server_*" / "server_constraints" / "run_*" / "chunk_*.npz"))):
            with np.load(f, allow_pickle=True) as z:
                s = json.loads(str(z["summary_json"]))
            es = s.get("esdf_stats") or {}
            if _lit(es.get("target_layer")) is not None:
                continue
            roi = _lit(es.get("roi_carve"))
            c = conf or ("B" if roi is None else "B_ROI")
            d = out.setdefault(c, {"esdf": [], "roi": []})
            d["esdf"].append(float((s.get("profile_ms") or {}).get("esdf", np.nan)))
            d["roi"].append(float(roi["ms"]) if roi else 0.0)
    return out


def _med(v):
    return round(float(np.median(v)), 2)


def _mean(v):
    return round(float(np.mean(v)), 2)


def _p95(v):
    return round(float(np.percentile(v, 95)), 2)


def fig_cost(out_dir, costs, cpu_rows, gpu_rows, ko):
    import matplotlib.pyplot as plt
    from AG3S.ag3s.experiments.common import figstyle

    C = figstyle.CATEGORICAL
    names = {"B": ("기준 구성", "baseline"), "B_ROI": ("관심 영역 지우기", "ROI carve"),
             "B_ROI_FP": ("가장자리 보존", "footprint"), "B_ROI_FPR": ("가장자리 보존 완화\n(serving 기본)", "relaxed footprint\n(serving default)")}
    order = ["B", "B_ROI", "B_ROI_FP", "B_ROI_FPR"]
    # stacks are means (they add up); each bar is labelled with its mean and median
    rest = [_mean(np.array(costs[c]["esdf"]) - np.array(costs[c]["roi"])) for c in order]
    roi = [_mean(costs[c]["roi"]) for c in order]
    total = [_med(costs[c]["esdf"]) for c in order]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    figstyle.style_axes(fig, axes)
    ax = axes[0]
    x = np.arange(len(order))
    ax.bar(x, rest, color=C[0], label="거리장 단계 − 관심 영역 지우기" if ko else "field stage − ROI carve")
    ax.bar(x, roi, bottom=rest, color=C[1], label="관심 영역 지우기" if ko else "ROI carve")
    for i, t in enumerate(total):
        ax.text(i, rest[i] + roi[i] + 6, (f"평균 {rest[i] + roi[i]:.0f} · 중앙값 {t:.0f} ms" if ko else
                                          f"mean {rest[i] + roi[i]:.0f} · median {t:.0f} ms"), ha="center", fontsize=8)
    ax.set_xticks(x, [names[c][0 if ko else 1] for c in order], fontsize=8)
    ax.set_ylabel("ms (평균, 부분별)" if ko else "ms (mean of each part)")
    ax.set_title("서버 기록: 거리장 단계 시간 (18 장면, closed-loop)" if ko else "server records: field stage (18 scenes)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    parts_cpu = {"filter": _mean([r.get("footprint.filter", 0) for r in cpu_rows]),
                 "judge": _mean([r.get("seen_through", 0) - r.get("footprint.filter", 0) for r in cpu_rows]),
                 "host": _mean([r.get("occupied.host", 0) for r in cpu_rows]),
                 "other": _mean([r["roi_carve_total"] - r.get("seen_through", 0) - r.get("occupied.host", 0) for r in cpu_rows])}
    cpu_tot = _med([r["roi_carve_total"] for r in cpu_rows])
    gpu_tot = _med([r["roi_carve_total"] + r.get("gpu_cameras", 0) for r in gpu_rows])
    gpu_mean = _mean([r["roi_carve_total"] + r.get("gpu_cameras", 0) for r in gpu_rows])
    lab = {"filter": ("영역 최솟값 filter (scipy)", "footprint min filter (scipy)"),
           "judge": ("투영 · 비교 (numpy)", "projection · compare (numpy)"),
           "host": ("voxel 중심 · 상자 (numpy)", "voxel centres · boxes (numpy)"),
           "other": ("나머지 (복사 · 쓰기 등)", "rest (copies, write)")}
    bottom = 0.0
    for k, col in zip(("filter", "judge", "host", "other"), (C[1], C[3], C[4], C[6])):
        ax.bar(0, parts_cpu[k], bottom=bottom, color=col, label=lab[k][0 if ko else 1])
        bottom += parts_cpu[k]
    ax.bar(1, gpu_mean, color=C[2], label="GPU 판정 전체 (depth 업로드 포함)" if ko else "GPU judgement (incl. depth upload)")
    ax.text(0, bottom + 5, (f"평균 {bottom:.1f} · 중앙값 {cpu_tot:.1f} ms" if ko else f"mean {bottom:.1f} · median {cpu_tot:.1f} ms"), ha="center", fontsize=8)
    ax.text(1, gpu_mean + 5, (f"평균 {gpu_mean:.1f} · 중앙값 {gpu_tot:.1f} ms" if ko else f"mean {gpu_mean:.1f} · median {gpu_tot:.1f} ms"), ha="center", fontsize=8)
    ax.set_xticks([0, 1], ["CPU (numpy · scipy)", "GPU (torch)"])
    ax.set_ylabel("ms (평균, 부분별)" if ko else "ms (mean of each part)")
    ax.set_title("offline replay: 관심 영역 지우기 한 번 (두 TSDF, 683 요청)" if ko else "offline replay: one ROI carve (2 TSDFs, 683 requests)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "roi-carve-gpu-cost.png", dpi=150)
    plt.close(fig)
    side = {"server": {c: {"n_chunks": len(costs[c]["esdf"]), "esdf_ms_median": total[i], "esdf_ms_p95": _p95(costs[c]["esdf"]),
                           "esdf_ms_mean": _mean(costs[c]["esdf"]), "roi_ms_median": _med(costs[c]["roi"]), "roi_ms_mean": roi[i],
                           "esdf_minus_roi_ms_median": _med(np.array(costs[c]["esdf"]) - np.array(costs[c]["roi"])),
                           "esdf_minus_roi_ms_mean": rest[i]} for i, c in enumerate(order)},
            "replay_cpu_parts_ms_mean": parts_cpu, "replay_gpu_total_ms_mean": gpu_mean, "replay_cpu_total_ms": {"median": cpu_tot, "p95": _p95([r["roi_carve_total"] for r in cpu_rows])},
            "replay_gpu_total_ms": {"median": gpu_tot, "p95": _p95([r["roi_carve_total"] + r.get("gpu_cameras", 0) for r in gpu_rows])},
            "note": "stacked parts are means (additive); totals also given as median / p95; server = closed-loop records of AG3S-TASK-DYNAMIC-MAP E2/E3/E4"}
    (out_dir / "roi-carve-gpu-cost.json").write_text(json.dumps(side, indent=1))
    return side


def fig_table(out_dir, cpu_sum, cost_side, smoke, ko):
    import matplotlib.pyplot as plt
    from AG3S.ag3s.experiments.common import figstyle

    g = cpu_sum["gpu_compare"]
    c, gp = cost_side["replay_cpu_total_ms"], cost_side["replay_gpu_total_ms"]
    red_med = 1 - gp["median"] / c["median"]
    red_p95 = 1 - gp["p95"] / c["p95"]
    rows = [
        ["G1 동등성" if ko else "G1 equivalence", "CPU · GPU 지울 voxel 집합 같음" if ko else "same voxels to free",
         f"판정 {g['n_judged_cpu']:,} voxel · {g['n_calls']} 번, 지울 voxel {g['n_hit_cpu']:,} vs {g['n_hit_gpu']:,}, 차이 {g['cpu_only'] + g['gpu_only']}",
         "통과" if g["cpu_only"] + g["gpu_only"] == 0 else "미달"],
        ["G2 속도" if ko else "G2 speed", "중앙값 −50 % 이상, p95 감소" if ko else "median −50 %, p95 down",
         f"중앙값 {c['median']} → {gp['median']} ms ({-red_med * 100:.1f} %), p95 {c['p95']} → {gp['p95']} ms ({-red_p95 * 100:.1f} %)",
         "통과" if red_med >= 0.5 and red_p95 > 0 else "미달"],
        ["G3 서버 경로" if ko else "G3 server path", "1 episode 정상 종료 · device = cuda" if ko else "1 episode ends · device = cuda",
         smoke.get("text", "—"), smoke.get("verdict", "—")],
    ]
    import textwrap

    cells = [[r[0], r[1], textwrap.fill(r[2], 80), r[3]] for r in rows]
    fig, ax = plt.subplots(figsize=(13, 3.4))
    figstyle.style_axes(fig, ax)
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=(["기준", "결과 전 고정 내용", "측정", "판정"] if ko else ["criterion", "pre-fixed", "measured", "verdict"]),
                 loc="center", cellLoc="left", colWidths=[0.12, 0.22, 0.56, 0.08])
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.8)
    for (r, c), cell in t.get_celld().items():   # rows with a wrapped measurement get the height of their lines
        if r > 0:
            cell.set_height(cell.get_height() * max(1, cells[r - 1][2].count("\n") + 1))
    ax.set_title("AG3S-ROI-CARVE-GPU 판정 (E4 18 장면 replay + 서버 1 episode)" if ko else "AG3S-ROI-CARVE-GPU verdicts", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "roi-carve-gpu-table.png", dpi=150)
    plt.close(fig)
    (out_dir / "roi-carve-gpu-table.json").write_text(json.dumps({"rows": rows, "gpu_compare": g, "replay_cpu_total_ms": c,
                                                                  "replay_gpu_total_ms": gp, "smoke": smoke}, indent=1, ensure_ascii=False))


def fig_scene(out_dir, tdm_run, request, ko):
    """Episode 1345 (E4 kit), one request: the touching hand geom, the orange, the recorded spheres, a field slice."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from scipy.spatial import cKDTree
    from AG3S.ag3s.experiments.common import figstyle
    from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as tde

    kit = tdm_run / "verify" / "AG3S-TASK-DYNAMIC-MAP-E4"
    pdir, chunks = tde.episodes_of(kit, "B_ROI_FPR")[1345]
    gt = tde.scene_truth(str(tdm_run / "snapshot_tdm3/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"),
                         tde.obstacle_profile(pdir.parent.parent))
    j, f, _ = next(c for c in chunks if c[0] == request)
    step = np.load(pdir / f"step_{j:05d}.npz")
    samples, _ = gt.at(np.asarray(step["qpos"], np.float64))
    R, o = gt.base_pose
    m, d = gt.m, gt.d
    nb = np.vstack([samples[i] for i, (k, n, b) in enumerate(gt.kind) if n == "orange"])
    g = 1052
    gp = ((gt._sample_local(g) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) - o) @ R
    z = np.load(f, allow_pickle=True)
    s = json.loads(str(z["summary_json"]))
    names = list(s["sphere_link_names"])
    cen = np.asarray(z["sphere_centres"], np.float64)
    rad = np.asarray(z["sphere_radii"], np.float64)
    dn = cKDTree(nb).query(gp)[0]
    close = dn <= dn.min() + 0.002
    cp = gp[close].mean(axis=0)
    v, org, vs = tde._grid(z, "esdf")
    C = figstyle.CATEGORICAL
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    figstyle.style_axes(fig, axes)
    win = 0.08
    links = ("ee_finger_l1", "ee_finger_l2", "ee_left")
    for ax, (a, b), title in ((axes[0], (0, 2), ("옆에서 (x–z)", "side (x–z)")), (axes[1], (0, 1), ("위에서 (x–y)", "top (x–y)"))):
        sel_nb = np.all(np.abs(nb - cp) < win, axis=1)
        ax.scatter(nb[sel_nb, a], nb[sel_nb, b], s=3, color=C[1], label="오렌지 GT 표면" if ko else "orange GT surface")
        ax.scatter(gp[:, a], gp[:, b], s=2, color=C[6], alpha=0.5, label="왼손 몸체 mesh (geom 1052)" if ko else "left hand body mesh (geom 1052)")
        ax.scatter(gp[close, a], gp[close, b], s=8, color=C[7], label="접촉 부위 (가장 가까운 2 mm)" if ko else "contact region (closest 2 mm)")
        for k, (cc, rr) in enumerate(zip(cen, rad)):
            if names[k] in links and np.all(np.abs(cc - cp) < win):
                ax.add_patch(Circle((cc[a], cc[b]), rr, fill=False, lw=0.8,
                                    color={"ee_finger_l1": C[0], "ee_finger_l2": C[2], "ee_left": C[3]}[names[k]]))
        for L, col in zip(links, (C[0], C[2], C[3])):
            ax.plot([], [], color=col, lw=1, label=f"AG3S 구 {L}" if ko else f"AG3S spheres {L}")
        if a == 0 and b == 2:
            ax.axhline(cp[2], color=figstyle.INK_2 if hasattr(figstyle, "INK_2") else "0.4", ls="--", lw=0.8)
            ax.text(cp[0] - win + 0.003, cp[2] + 0.002, "단면 높이" if ko else "slice height", fontsize=8)
        ax.set_xlim(cp[a] - win, cp[a] + win)
        ax.set_ylim(cp[b] - win, cp[b] + win)
        ax.set_aspect("equal")
        ax.set_xlabel(f"base {'xyz'[a]} (m)")
        ax.set_ylabel(f"base {'xyz'[b]} (m)")
        ax.set_title(title[0 if ko else 1], fontsize=10)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    ax = axes[2]
    kz = int(np.rint((cp[2] - org[2]) / vs))
    xs = org[0] + np.arange(v.shape[0]) * vs
    ys = org[1] + np.arange(v.shape[1]) * vs
    sl = v[:, :, kz].astype(np.float64) * 1e3
    im = ax.imshow(np.clip(sl.T, -20, 60), origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap=figstyle.sequential_cmap())
    ax.contour(xs, ys, sl.T, levels=[0.0], colors=[figstyle.INK], linewidths=0.8)
    band = np.abs(nb[:, 2] - cp[2]) < 0.004
    ax.scatter(nb[band, 0], nb[band, 1], s=4, color=C[1])
    gb = np.abs(gp[:, 2] - cp[2]) < 0.004
    ax.scatter(gp[gb, 0], gp[gb, 1], s=3, color=C[6], alpha=0.6)
    ax.scatter(gp[close, 0], gp[close, 1], s=8, color=C[7])
    ax.set_xlim(cp[0] - win, cp[0] + win)
    ax.set_ylim(cp[1] - win, cp[1] + win)
    ax.set_aspect("equal")
    ax.set_xlabel("base x (m)")
    ax.set_ylabel("base y (m)")
    ax.set_title((f"main fine 거리장 단면 z = {cp[2]:.3f} m (검은 선 = 0 mm)" if ko else f"main fine field slice z = {cp[2]:.3f} m (black = 0 mm)"), fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="mm")
    fig.suptitle((f"episode 1345 요청 {request} (가장자리 보존 완화 구성): 왼손 몸체가 오렌지에 닿는 자리는 AG3S 구 밖이다" if ko else
                  f"episode 1345 request {request}: the hand body touches the orange outside the AG3S spheres"), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "roi-carve-gpu-scene-ep1345.png", dpi=150)
    plt.close(fig)
    out_all = np.linalg.norm(gp[close][:, None] - cen[None], axis=2) - rad[None]
    side = {"episode": 1345, "request": request, "chunk": pathlib.Path(f).name, "geom": g,
            "geom_to_orange_min_mm": round(float(dn.min()) * 1e3, 3), "contact_region_n": int(close.sum()),
            "contact_region_centre_base_m": [round(float(x), 4) for x in cp],
            "contact_region_outside_nearest_sphere_mm": [round(float(out_all.min(axis=1).min()) * 1e3, 3),
                                                         round(float(out_all.min(axis=1).max()) * 1e3, 3)],
            "sphere_radius_mm": {L: [round(float(rad[[k for k, n in enumerate(names) if n == L]].min()) * 1e3, 2),
                                     round(float(rad[[k for k, n in enumerate(names) if n == L]].max()) * 1e3, 2)] for L in links},
            "slice_z_m": round(float(org[2] + kz * vs), 4), "field": "esdf_distance (main fine layer, recorded)"}
    (out_dir / "roi-carve-gpu-scene-ep1345.json").write_text(json.dumps(side, indent=1))
    return side


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", required=True)
    ap.add_argument("--tdm-run", required=True)
    ap.add_argument("--smoke", default=None, help="json {text, verdict} for the G3 row")
    ap.add_argument("--request", type=int, default=13)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    from AG3S.ag3s.experiments.common import figstyle

    ko = figstyle.use_korean()
    V = pathlib.Path(a.verify)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    costs = server_costs(pathlib.Path(a.tdm_run))
    cpu_rows = json.loads((V / "cpu_18" / "rows.json").read_text())
    gpu_rows = json.loads((V / "gpu_18" / "rows.json").read_text())
    side = fig_cost(out, costs, cpu_rows, gpu_rows, ko)
    smoke = json.loads(pathlib.Path(a.smoke).read_text()) if a.smoke else {}
    fig_table(out, json.loads((V / "cpu_18" / "summary.json").read_text()), side, smoke, ko)
    print(json.dumps(fig_scene(out, pathlib.Path(a.tdm_run), a.request, ko)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
