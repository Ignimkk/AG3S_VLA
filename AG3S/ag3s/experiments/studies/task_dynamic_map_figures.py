"""AG3S-TASK-DYNAMIC-MAP figures (rule A): the scene, the field metrics, the judgement table. Each PNG gets a JSON sidecar.

  scene  — episode `--scene-episode` (default 211), the chunk where the base configuration's main tier holds the most target-old
           residue: top view of the served main fine tier's surface voxels (|d| ≤ half a voxel) in a band above the table, base vs
           target layer at the same chunk index; GT apple now (blue) and the apple's earlier poses (dashed)
  bars   — per configuration: target-old residue (main / hand), other-body old residue (main), missed obstacles (pp vs base),
           served esdf ms (median)
  table  — the pre-registered criteria 1–4 per configuration with the values

  MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \\
      AG3S.ag3s.experiments.studies.task_dynamic_map_figures --kit <V> --model-xml <xml> --out AG3S/ag3s/docs/figures/task-dynamic-map
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from AG3S.ag3s.experiments.common import figstyle as fs
from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as E

CONFS = ("B", "B_TL", "B_ROI", "B_TL_ROI")


def labels(ko: bool) -> dict:
    if ko:
        return {"B": "기준 (NBCTS+추종)", "B_TL": "+ target 전용 TSDF", "B_ROI": "+ 관심 영역 지우기", "B_TL_ROI": "+ 둘 다"}
    return {"B": "base (NBCTS+follow)", "B_TL": "+ target-only TSDF", "B_ROI": "+ ROI carve", "B_TL_ROI": "+ both"}


def field_summary(kit: pathlib.Path) -> dict:
    out = {}
    for c in CONFS:
        for sub in ("field_eval_A1", "field_eval_A2", "field_eval"):
            p = kit / sub / c / "summary.json"
            if p.exists():
                out[c] = json.loads(p.read_text())["configurations"][c]
                break
    return out


def scene(kit, xml, ep, out, ko):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows_b = json.loads((kit / "field_eval_A1" / "B" / f"rows_B_ep{ep}.json").read_text())[1:]
    k = int(np.argmax([(r.get("main") or {}).get("target_old", 0) for r in rows_b]))
    j = rows_b[k]["request"]
    data = {"episode": ep, "request": j, "panels": {}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=150)
    L = labels(ko)
    for ax, c in zip(axes, ("B", "B_TL")):
        pdir, chunks = E.episodes_of(kit, c)[ep]
        chunk = next(f for jj, f, st in chunks if jj == j)
        gt = E.scene_truth(xml, E.obstacle_profile(pdir.parent.parent))
        t_idx = [i for i, (kk, n, b) in enumerate(gt.kind) if n == E.TARGET]
        olds = []
        for jj, f, st in chunks:
            if jj > j:
                break
            s, _ = gt.at(np.load(pdir / f"step_{jj:05d}.npz")["qpos"])
            olds.append(np.vstack([s[i] for i in t_idx]))
        now = olds[-1]
        z_tab = float(olds[0][:, 2].min())   # the table under the apple's first pose (the apple may have fallen since)
        with np.load(chunk) as z:
            surf = E._surface(E._grid(z, "esdf"))
        band = surf[(surf[:, 2] > z_tab + 0.015) & (surf[:, 2] < z_tab + 0.09)]
        lo = now.min(0) - 0.12
        hi = now.max(0) + 0.12
        for p in olds[:-1]:
            lo, hi = np.minimum(lo, p.min(0) - 0.06), np.maximum(hi, p.max(0) + 0.06)
        band = band[np.all((band[:, :2] >= lo[:2]) & (band[:, :2] <= hi[:2]), axis=1)]
        from scipy.spatial import cKDTree
        moved = [p for p in olds[:-1] if np.linalg.norm(p.mean(0) - now.mean(0)) > E.MOVED_MM / 1000.0]
        old_hit = np.zeros(len(band), bool)
        if moved and len(band):
            d_old = cKDTree(np.vstack(moved)).query(band)[0]
            d_now = cKDTree(now).query(band)[0]
            old_hit = (d_old <= E.STALE_MM / 1000.0) & (d_now > E.STALE_MM / 1000.0)
        ax.scatter(band[~old_hit, 0], band[~old_hit, 1], s=2, c="#b5b4ae", label=("거리장 표면" if ko else "field surface"))
        ax.scatter(band[old_hit, 0], band[old_hit, 1], s=3, c=fs.CATEGORICAL[1],
                   label=("사과 옛 자리 표면" if ko else "surface at the apple's old place"))
        ax.scatter(now[:, 0], now[:, 1], s=1, c=fs.CATEGORICAL[0], label=("사과 지금 (GT)" if ko else "apple now (GT)"))
        if moved:
            first = moved[0]
            ax.scatter(first[:, 0], first[:, 1], s=0.5, c="none", edgecolors=fs.CATEGORICAL[0], linewidths=0.3,
                       label=("사과 처음 자리 (GT)" if ko else "apple first pose (GT)"))
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{L[c]} · ep{ep} req {j} · " + (f"옛 자리 표면 {int(old_hit.sum())}" if ko else f"old-place surface {int(old_hit.sum())}"))
        ax.legend(loc="upper right", fontsize=7, markerscale=4)
        data["panels"][c] = {"n_band_surface": int(len(band)), "n_old_place_surface": int(old_hit.sum()),
                             "apple_shift_from_first_mm": round(1000 * float(np.linalg.norm(now.mean(0) - olds[0].mean(0))), 1),
                             "z_band_m": [round(z_tab + 0.015, 4), round(z_tab + 0.09, 4)], "layer": "esdf_distance (main fine)"}
    fs.style_axes(fig, axes)
    fig.suptitle(("위에서 본 서버 거리장 표면 (탁자 위 15–90 mm) — 사과가 밀린 뒤" if ko else
                  "Served field surface from above (15–90 mm above the table) — after the apple was pushed"), fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"task-dynamic-map-scene-ep{ep}.png")
    (out / f"task-dynamic-map-scene-ep{ep}.json").write_text(json.dumps(data, indent=1))
    return data


def bars(fsum, cl, out, ko):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = labels(ko)
    cs = [c for c in CONFS if c in fsum]
    base = fsum["B"]
    vals = {
        "target_old_main": [fsum[c]["main"]["target_old"] for c in cs],
        "target_old_hand": [fsum[c]["hand"]["target_old"] for c in cs],
        "moved_old_main": [fsum[c]["main"]["moved_old"] for c in cs],
        "missed_occluded_main_pp": [100 * (fsum[c]["main"]["missed_frac"]["occluded_seen_before"]
                                           - base["main"]["missed_frac"]["occluded_seen_before"]) for c in cs],
        "missed_visible_main_pp": [100 * (fsum[c]["main"]["missed_frac"]["visible"] - base["main"]["missed_frac"]["visible"]) for c in cs],
        "esdf_ms_median": [cl["configurations"][c]["esdf_ms"]["median"] for c in cs],
    }
    titles = {"target_old_main": ("target 옛 자리 잔상 · 손 외 몸 층", "target-old residue · main tier"),
              "target_old_hand": ("target 옛 자리 잔상 · 손 층", "target-old residue · hand tier"),
              "moved_old_main": ("다른 물체 옛 자리 잔상 · 손 외 몸 층", "other-body old residue · main tier"),
              "missed_occluded_main_pp": ("잊은 장애물 (가려졌던 점) · 기준 대비 pp", "missed obstacles (occluded) · pp vs base"),
              "missed_visible_main_pp": ("잊은 장애물 (보이는 점) · 기준 대비 pp", "missed obstacles (visible) · pp vs base"),
              "esdf_ms_median": ("서버 거리장 단계 ms (중앙값)", "served esdf stage ms (median)")}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=150)
    for ax, key in zip(axes.ravel(), vals):
        ax.bar(range(len(cs)), vals[key], color=[fs.CATEGORICAL[i] for i in range(len(cs))])
        ax.set_xticks(range(len(cs)), [L[c] for c in cs], rotation=20, fontsize=7)
        ax.set_title(titles[key][0 if ko else 1], fontsize=9)
        for i, v in enumerate(vals[key]):
            ax.text(i, v, f"{v:.2f}" if isinstance(v, float) and abs(v) < 100 else f"{v:,.0f}", ha="center", va="bottom", fontsize=7)
        if key.endswith("_pp"):
            ax.axhline(1.0, color=fs.INK, lw=0.8, ls="--")
    fs.style_axes(fig, axes.ravel())
    fig.suptitle(("10 장면 합 · 평가용 관심 영역 안 (사과 처음 자리 ∪ 바구니 ± 0.10 m, 사과 지금 ± 0.10 m)" if ko else
                  "10 scenes summed · inside the evaluation region"), fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "task-dynamic-map-bars.png")
    (out / "task-dynamic-map-bars.json").write_text(json.dumps({"configurations": cs, **vals}, indent=1))
    return vals


def judge(fsum, cl) -> dict:
    base = fsum["B"]
    clb = cl["configurations"]["B"]
    res = {}
    for c in CONFS:
        if c == "B" or c not in fsum:
            continue
        f = fsum[c]
        j = {}
        if "TL" in c:
            j["1_target_old_main"] = _red(base["main"]["target_old"], f["main"]["target_old"], 0.90)
            j["1_target_old_hand"] = _red(base["hand"]["target_old"], f["hand"]["target_old"], 0.90)
        if "ROI" in c:
            j["2_moved_old_main"] = _red(base["main"]["moved_old"], f["main"]["moved_old"], 0.50)
        for L in ("main", "hand"):
            for k in ("visible", "occluded_seen_before"):
                d = 100 * (f[L]["missed_frac"][k] - base[L]["missed_frac"][k])
                j[f"3_missed_{L}_{k}_pp"] = {"value": round(d, 3), "pass": d <= 1.0}
        x = cl["configurations"][c]
        j["4_success"] = {"value": x["n_success"], "base": clb["n_success"], "pass": x["n_success"] >= clb["n_success"]}
        j["4_forbidden_episodes"] = {"value": x["n_forbidden"], "base": clb["n_forbidden"], "pass": x["n_forbidden"] <= clb["n_forbidden"]}
        undecided = [k for k, v in j.items() if v.get("pass") is None]
        j["adopt"] = (not undecided) and all(v["pass"] for v in j.values() if isinstance(v, dict) and "pass" in v)
        res[c] = j
    return res


def _red(b, v, need):
    if b == 0:
        return {"value": v, "base": b, "reduction": None, "pass": None, "note": "base 0 — not decidable"}
    r = 1.0 - v / b
    return {"value": v, "base": b, "reduction": round(r, 4), "pass": r >= need}


def table(res, cl, out, ko):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = labels(ko)
    keys = sorted({k for j in res.values() for k in j if k != "adopt"})
    cols = ["B"] + list(res)
    cell = []
    for k in keys + ["adopt"]:
        row = [k]
        for c in cols:
            if c == "B":
                x = cl["configurations"]["B"]
                row.append({"4_success": str(x["n_success"]), "4_forbidden_episodes": str(x["n_forbidden"])}.get(k, "—"))
                continue
            v = res[c].get(k)
            if k == "adopt":
                row.append(("채택" if ko else "adopt") if v else ("미채택" if ko else "not adopted"))
            elif v is None:
                row.append("—")
            else:
                val = v.get("reduction", v.get("value"))
                val = f"{100 * val:.1f}% ↓" if "reduction" in v and val is not None else str(val)
                row.append(f"{val} {'✓' if v['pass'] else ('?' if v['pass'] is None else '✗')}")
        cell.append(row)
    fig, ax = plt.subplots(figsize=(13, 0.42 * len(cell) + 1.2), dpi=150)
    ax.axis("off")
    t = ax.table(cellText=cell, colLabels=[("기준" if ko else "criterion")] + [L[c] for c in cols], loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(7)
    t.scale(1, 1.3)
    ax.set_title(("미리 정한 판정 (task §5 · §5a)" if ko else "Pre-registered judgement (task §5 · §5a)"), fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "task-dynamic-map-table.png")
    (out / "task-dynamic-map-table.json").write_text(json.dumps({"judgement": res, "rows": cell}, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", required=True, type=pathlib.Path)
    ap.add_argument("--model-xml", required=True)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--scene-episode", type=int, default=954)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("task-dynamic-map-bars.png", "task-dynamic-map-table.png", f"task-dynamic-map-scene-ep{args.scene_episode}.png"):
        if (args.out / name).exists():
            raise SystemExit(f"{args.out / name} exists — figures are not overwritten (rule A); choose a new --out")
    ko = fs.use_korean()
    fsum = field_summary(args.kit)
    cl = json.loads((args.kit / "agg" / "tdm_closed_loop.json").read_text())
    scene(args.kit, args.model_xml, args.scene_episode, args.out, ko)
    bars(fsum, cl, args.out, ko)
    res = judge(fsum, cl)
    table(res, cl, args.out, ko)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
