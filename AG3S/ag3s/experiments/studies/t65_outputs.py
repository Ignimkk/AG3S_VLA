"""T65 — human-facing outputs folder for the fixed-input attention-source comparison (`handoff/T65.task.md`).

    scene_<label>_r<j>.png   per source: head image + 16x16 attention, head depth + target points + visible GT apple
    graph_scene_diff.png     per scene, (source − noun) on the three primary metrics, signed so that up = better
    graph_metric_means.png   scene-mean of every metric per source
    table_*.md / tables.json scene means, decision, parity
    (every figure has a same-name .json sidecar with the numbers it draws)

usage: MUJOCO_GL=osmesa t65_outputs.py RUN65 OUTPUTS_DIR      (.venv-openpi-live; PYTHONPATH = snapshot_core + manipulation src)
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve()


def _load(name):
    sp = importlib.util.spec_from_file_location(name, HERE.with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


M = _load("t65_metrics")       # puts the snapshot on sys.path first
sys.path.insert(0, str(HERE.parents[1] / "common"))
import figstyle  # noqa: E402

NAMES = {"noun_target_L10H7": "noun token (current)", "action_last_L8H2": "action head L8H2 (common)",
         "action_mean_L10H7": "action head L10H7 (T61 procedure)"}
MLAB = {"identity": "identity (majority = apple)", "precision_5mm": "precision 5 mm", "other_object": "other object chosen",
        "recall_5mm": "visible-surface recall 5 mm", "vis_centroid_mm": "centroid error (mm)", "none_object": "points on no object",
        "no_target": "no target", "apple_left_frac": "target left in hand field", "wrong_removed_frac": "wrongly removed voxels"}


def save(fig, path: pathlib.Path, nums: dict) -> None:
    fig.savefig(path, dpi=150)
    plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps(nums, ensure_ascii=False, indent=1, default=float))


def project(p, st, cam="zed_left"):
    K, T = np.asarray(st[f"K_{cam}"], float), np.asarray(st[f"T_base_cam_{cam}"], float)
    pc = (p - T[:3, 3]) @ T[:3, :3]
    ok = pc[:, 2] > 1e-6
    return K[0, 0] * pc[ok, 0] / pc[ok, 2] + K[0, 2], K[1, 1] * pc[ok, 1] / pc[ok, 2] + K[1, 2]


def scene_figure(run65, label, j, rows, o: pathlib.Path) -> None:
    """One request j of run `label`, the three sources side by side."""
    spec = {s["label"]: s for s in json.loads((M.V / "module/spec_core.json").read_text())}[label]
    group = ("OCC_k" if label.startswith("가림") else "REG_k") + label[-1]
    name = pathlib.Path(spec["client_dir"]).name
    steps = sorted(pathlib.Path(spec["client_dir"]).glob("ag3s/run_0000/step_*.npz"), key=lambda p: int(np.load(p)["t_step"]))
    st = np.load(steps[j])
    gt = M.tde.scene_truth(M.XML, tuple(spec["obstacle"]))
    samples, _ = gt.at(np.asarray(st["qpos"], float))
    ap = np.vstack([samples[i] for i, k in enumerate(gt.kind) if k[1] == "apple"])
    vis = M.visible(ap, st, head_only=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    nums = {"label": label, "request": j, "t_step": int(st["t_step"]), "sources": {}}
    for c, src in enumerate(M.SOURCES):
        seg = M.replay_segments(run65 / "verify/T65" / src / group)[name]
        z = np.load(seg[j], allow_pickle=True)
        att = np.asarray(z["attention_head"], float)
        tp = np.asarray(z["target_points"], float) if "target_points" in z.files else np.zeros((0, 3))
        r = next((x for x in rows[src][label] if x["request"] == j), {})
        a = axes[0, c]
        a.imshow(st["image_cam_high"])
        a.imshow(att / max(att.max(), 1e-9), extent=(0, 224, 224, 0), cmap=figstyle.sequential_cmap(), alpha=0.55)
        a.set_title(f"{NAMES[src]}\nattention on head image", fontsize=9); a.axis("off")
        b = axes[1, c]
        b.imshow(np.asarray(st["depth_zed_left"], float) / 1000.0, cmap="gray")
        u, v = project(ap[vis], st)
        b.scatter(u, v, s=1, c=figstyle.CATEGORICAL[3], label="visible GT apple")
        if len(tp):
            u, v = project(tp, st)
            b.scatter(u, v, s=2, c=figstyle.CATEGORICAL[2], label="target points")
        b.set_title(f"majority {r.get('majority_object', 'no target')} · precision {r.get('precision_5mm', float('nan')):.2f}"
                    f"\nrecall {r.get('recall_5mm', float('nan')):.2f} · left in hand field {r.get('apple_left_frac') or float('nan'):.2f}",
                    fontsize=8)
        b.axis("off")
        nums["sources"][src] = {k: r.get(k) for k in ("majority_object", "precision_5mm", "recall_5mm", "vis_centroid_mm",
                                                       "apple_left_frac", "removed_by_class")} | {"n_target_points": int(len(tp))}
    axes[1, 0].legend(loc="lower left", fontsize=7, markerscale=6)
    eng = label.replace("가림", "occlusion ").replace("회귀", "regression ")
    fig.suptitle(f"{eng} request {j} (same recorded observation, attention source changed)", fontsize=10)
    fig.tight_layout()
    save(fig, o / f"scene_{label}_r{j}.png", nums)


def main(run65: pathlib.Path, o: pathlib.Path) -> None:
    o.mkdir(parents=True, exist_ok=True)
    a = run65 / "verify/T65/analysis"
    res = json.loads((a / "t65_metrics.json").read_text())
    rows = json.loads((a / "t65_rows.json").read_text())
    scenes, dec = res["scenes"], res["decision_ats"]
    sc = sorted(scenes[M.BASE])
    cols = figstyle.CATEGORICAL

    # graph 1: per-scene signed difference on the primary metrics
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    nums = {}
    for ax, k in zip(axes, M.PRIMARY):
        for i, src in enumerate(M.SOURCES[1:]):
            d = dec[src]["diff"][k]["per_scene"]
            y = [M.DIRECTION[k] * d.get(s, np.nan) for s in sc]
            ax.bar(np.arange(len(sc)) + (i - 0.5) * 0.4, y, 0.4, color=cols[i + 1], label=NAMES[src])
            nums.setdefault(k, {})[src] = dict(zip(sc, y))
        ax.axhline(0, color=figstyle.INK, lw=0.8)
        ax.axhline(M.IMPROVE_MIN, color=figstyle.GRID_INK, lw=0.8, ls="--")
        ax.set_xticks(range(len(sc)), [s.replace("가림", "occl").replace("회귀", "reg") for s in sc], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{MLAB[k]}: (source − noun), up = better", fontsize=9)
    axes[0].legend(fontsize=7)
    figstyle.style_axes(fig, list(axes)); fig.tight_layout()
    save(fig, o / "graph_scene_diff.png", nums)

    # graph 2: scene-mean of every metric per source
    ks = list(M.DIRECTION)
    fig, axes = plt.subplots(1, len(ks), figsize=(2.0 * len(ks), 3.4))
    nums = {}
    for ax, k in zip(axes, ks):
        vals = [np.nanmean([scenes[s][x][k] if scenes[s][x][k] is not None else np.nan for x in sc]) for s in M.SOURCES]
        ax.bar(range(3), vals, color=cols[:3])
        ax.set_xticks(range(3), ["noun", "L8H2", "L10H7"], fontsize=7)
        ax.set_title(MLAB[k] + (" ↑" if M.DIRECTION[k] > 0 else " ↓"), fontsize=7)
        nums[k] = dict(zip(M.SOURCES, vals))
    figstyle.style_axes(fig, list(axes)); fig.tight_layout()
    save(fig, o / "graph_metric_means.png", nums)

    # tables
    lines = ["| 지표 (좋은 방향) | " + " | ".join(NAMES[s] for s in M.SOURCES) + " |", "|---|---|---|---|"]
    for k in ks:
        lines.append(f"| {MLAB[k]} ({'↑' if M.DIRECTION[k] > 0 else '↓'}) | "
                     + " | ".join(f"{nums[k][s]:.3f}" for s in M.SOURCES) + " |")
    lines += ["", "| 비교 구성 | 주 지표 3 개 각 ≥ 2 %p 개선 | 나머지 악화 없음 | 9 장면 중 ≥ 5 개선 | ATS 판정 | 악화 지표 |",
              "|---|---|---|---|---|---|"]
    for s in M.SOURCES[1:]:
        x = dec[s]
        lines.append(f"| {NAMES[s]} | {x['c1_primary_improve_2pp']} | {x['c2_no_other_worse']} | {x['c3_5of9_scenes']} | "
                     f"{'통과' if x['ats_pass'] else '불통과 → 명사 유지'} | {', '.join(x['worse_others']) or '-'} |")
    lines += ["", "| 지표 | 비교 구성 | 장면 평균 차 (좋은 방향 +) | 개선 장면 | 악화 장면 | 장면 수 |", "|---|---|---|---|---|---|"]
    for k in ks:
        for s in M.SOURCES[1:]:
            d = dec[s]["diff"][k]
            g = d["good_direction_mean"]
            lines.append(f"| {MLAB[k]} | {NAMES[s]} | {g:+.4f} | {d['scenes_improved']} | {d['scenes_worse']} | {d['n_scenes']} |"
                         if g is not None else f"| {MLAB[k]} | {NAMES[s]} | - | - | - | 0 |")
    par = {g: json.loads((a / f"parity_{g}.json").read_text()) for g in M.GROUPS if (a / f"parity_{g}.json").exists()}
    lines += ["", "| 명사 재생 parity | 요청 | target 선택 (id · grounding) 일치 | + visible/occluded 상태까지 | Chamfer median (mm) | attention argmax 일치 | gate |",
              "|---|---|---|---|---|---|---|"]
    for g, p in par.items():
        lines.append(f"| {g} | {p['n']} | {p['selection_agreement']:.3f} | {p.get('selection_with_state_agreement', float('nan')):.3f} | "
                     f"{p['chamfer_median_m'] * 1e3:.2f} | {p['att_argmax_equal']} | {'PASS' if p['gate_pass'] else 'FAIL'} |")
    (o / "table_t65.md").write_text("\n".join(lines) + "\n")
    (o / "tables.json").write_text(json.dumps({"scene_mean": nums, "decision": {s: {k: v for k, v in dec[s].items() if k != "diff"}
                                                                                for s in dec},
                                               "parity": {g: {k: v for k, v in p.items() if k != "rows"} for g, p in par.items()}},
                                              ensure_ascii=False, indent=1, default=float))

    # scene figures: per run, the pre-grasp request with the largest precision spread across sources; the two
    # regression runs with the largest spread + the first occlusion run (all sources identical there)
    spread = []
    for lab in rows[M.BASE]:
        by = {s: {r["request"]: r for r in rows[s].get(lab, [])} for s in M.SOURCES}
        best = None
        for j in sorted(set.intersection(*(set(b) for b in by.values()))):
            v = [by[s][j].get("precision_5mm", 0.0) for s in M.SOURCES]
            if best is None or max(v) - min(v) > best[0]:
                best = (max(v) - min(v), lab, j)
        if best:
            spread.append(best)
    reg = sorted((x for x in spread if x[1].startswith("회귀")), reverse=True)
    seen, picks = set(), []
    for _, lab, j in reg:
        if lab[:-3] not in seen:
            seen.add(lab[:-3]); picks.append((lab, j))
        if len(picks) == 2:
            break
    picks.append(next((lab, j) for _, lab, j in spread if lab.startswith("가림")))
    for lab, j in picks:
        scene_figure(run65, lab, j, rows, o)
    print("->", o, picks)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
