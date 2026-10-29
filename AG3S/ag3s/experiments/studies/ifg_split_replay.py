"""IFG-SPLIT offline — re-ground the IFG-APPLE C1aN server chunks (T43 `ground_replay` method: fresh TargetConfirm, the recorded
attention cloud / support mask / hand spheres / latch phase) with the serving clustering keys, then with the IFG-SPLIT keys on top.
CPU only, read only.

  OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_split_replay \
      --run <IFG-APPLE run> --out <run>/verify/IFG-SPLIT-OFFLINE --fig AG3S/ag3s/docs/figures/ifg-split

serving = the C1aN server's clustering keys (start_server.sh T2 + T44A): split radius 12 mm, raw ratio, closing_keep_geometry,
latched_identity_hold, supported_max_bottom 60 mm. SPL = serving + attention_split_partition + split_supersedes_anchor.
Per chunk: the manipulated object's state, points, largest extent, anchor; vs GT (replay target_pos; neighbour fruits at reset from
IFG-APPLE-FAIL/scene_bodies.json): frac_neighbour = share of its points nearer a neighbour fruit centre than the apple centre.
Parity: serving vs the record (state, largest extent, centroid) on every chunk.
"""
import argparse, copy, glob, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)
HAND = ("ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")
SERVING = {"clustering": {"attention_split_radius": 0.012, "attention_split_ratio_source": "raw",
                          "closing_keep_geometry": True, "latched_identity_hold": True, "supported_max_bottom": 0.06}}
SPL = copy.deepcopy(SERVING)
SPL["clustering"].update({"attention_split_partition": True, "split_supersedes_anchor": True})
MERGED_SHOW = (211, 250, 572, 1345)
SHOW_K = 5


def replay_episode(v, bodies, ep, overrides, keep_points_k=()):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.stages.admissibility import HandSpheres
    from AG3S.ag3s.stages.support_surface import fit_support_surfaces
    from AG3S.ag3s.stages.target_grounding import TargetConfirm, ground_target
    from AG3S.ag3s.types import AttentionPointCloud, PointCloud
    jf = glob.glob(str(v / "IFG-APPLE" / "replay" / f"runs_C1aN_C1aN_ep{ep}_*.json"))[0]
    j = json.load(open(jf)); z = np.load(jf[:-5] + ".npz")
    al = j["server_chunk_alignment"]["block"]; cd, off = pathlib.Path(al["chunk_dir"]), int(al["offset"])
    pi = z["plan_index"]; tgt = z["target_pos"]
    others = [np.array(bodies[str(ep)][f]) for f in ("banana", "orange", "pear")]
    cfg = AG3SConfig().with_overrides(overrides)
    conf = TargetConfirm(cfg.clustering, max_opening=0.07146, overlap_distance=float(cfg.pointcloud.voxel_size) or cfg.clustering.eps)
    rows, pts_out = [], {}
    for k in range(int(pi.max()) + 1):
        idx = np.flatnonzero(pi == k)
        if not len(idx):
            continue
        gt = tgt[int(idx[0])]
        with np.load(cd / f"chunk_{k + off:05d}.npz", allow_pickle=False) as c:
            s = json.loads(str(c["summary_json"])); excl = s.get("exclusion") or {}; g = s.get("grasp") or {}
            conf.freeze(bool(excl.get("grasp_active")) or bool(g.get("attached")))
            conf.note_latched(g.get("state") in ("latched", "closing"))
            conf.note_pre_attach_grasp(bool(excl.get("grasp_active")) and not bool(g.get("attached")))
            conf.note_subtask((s.get("subtask") or {}).get("p")); conf.note_placed(g.get("state") == "placed")
            names = s.get("sphere_link_names") or []; keep = np.array([n in HAND for n in names], bool)
            if keep.any():
                conf.note_hand(HandSpheres(np.asarray(c["sphere_centres"], float)[keep], np.asarray(c["sphere_radii"], float)[keep],
                                           tuple(n for n, kk in zip(names, keep) if kk)))
            if getattr(conf, "supported_max_bottom", None) is not None:
                surf, _ = fit_support_surfaces(PointCloud(points=np.asarray(c["filtered_cloud"], np.float64)), cfg.support_surface, timestamp=0.0)
                conf.note_support(np.asarray([[*np.asarray(f.normal, float), float(f.offset)] for f in surf], float).reshape(-1, 4) if surf else None)
            apc = AttentionPointCloud(PointCloud(c["attention_cloud"].astype(np.float64)), c["attention_values"], raw_attention=c["attention_raw"])
            ground_target(apc, cfg.clustering, seed_percentile=cfg.attention.seed_percentile, seed_threshold=cfg.attention.seed_threshold,
                          exclude_mask=c["support_mask"], min_radius=cfg.geometry.min_radius, timestamp=float(c["t_step"]), confirm=conf)
            if k in keep_points_k:
                cl = np.asarray(c["attention_cloud"])[:, :3]; near = np.linalg.norm(cl - gt, axis=1) < 0.13
                pts_out[k] = {"cloud": cl[near], "raw": np.asarray(c["attention_raw"])[near]}
        man = conf.manipulated; rec_m = s.get("manipulated") or {}
        r = {"k": k, "state_srv": g.get("state"), "rec_state": rec_m.get("state"),
             "rec_ext": max((rec_m.get("admissibility") or {}).get("extents_mm") or [0]) or None, "rec_centroid": rec_m.get("centroid")}
        if man is not None:
            P = np.asarray(man.points).reshape(-1, 3)
            da = np.linalg.norm(P - gt, axis=1); do = np.min(np.stack([np.linalg.norm(P - o, axis=1) for o in others]), axis=0)
            ext = man.admissibility.extents_m if getattr(man, "admissibility", None) is not None else None
            r.update({"state": man.state, "n": int(len(P)), "ext_mm": None if ext is None else float(max(ext) * 1000),
                      "centroid": [float(x) for x in np.asarray(man.centroid).reshape(3)],
                      "frac_neighbour": float(np.mean(do < da)) if len(P) else None,
                      "anchor_src": (man.anchor or {}).get("source"), "anchor_ext": (man.anchor or {}).get("extents_mm"),
                      "reason": getattr(conf.last, "reason", None) if conf.last is not None else None})
            if k in keep_points_k:
                pts_out[k]["target"] = P
        rows.append(r)
    return rows, pts_out, tgt, others


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--fig", type=pathlib.Path, required=True)
    a = ap.parse_args()
    v = a.run / "verify"; a.out.mkdir(parents=True, exist_ok=True); a.fig.mkdir(parents=True, exist_ok=True)
    bodies = json.load(open(v / "IFG-APPLE-FAIL" / "scene_bodies.json"))
    eps = [it["episode"] for it in json.load(open(v / "IFG-APPLE-3ARM" / "seeds.json"))["items"]]
    R, PTS = {"serving": {}, "SPL": {}}, {}
    for ep in eps:
        for name, ov in (("serving", SERVING), ("SPL", SPL)):
            rows, pts, tgt, others = replay_episode(v, bodies, ep, ov, keep_points_k=(SHOW_K,) if ep in MERGED_SHOW else ())
            R[name][ep] = rows
            if pts:
                PTS[(ep, name)] = (pts[SHOW_K], tgt, others)
        print(ep, flush=True)
    (a.out / "rows.json").write_text(json.dumps({n: {str(e): r for e, r in d.items()} for n, d in R.items()}, default=float))
    N = {"parity_serving_vs_record": {}, "per_scene": {}}
    for ep in eps:
        S, F = R["serving"][ep], R["SPL"][ep]
        same = [r.get("state") == r["rec_state"] and (r.get("ext_mm") is None or r["rec_ext"] is None or abs(r["ext_mm"] - r["rec_ext"]) < 0.05)
                for r in S]
        N["parity_serving_vs_record"][str(ep)] = {"n_chunks": len(S), "n_identical": int(sum(same))}
        fc = next((r["k"] for r in S if r["state_srv"] in ("closing", "held")), len(S))
        s = [r for r in S[:fc] if r.get("n")]; f = [r for r in F[:fc] if r.get("n")]
        nb = bodies[str(ep)]; apx = np.array(nb["apple"][:2])
        near = min(float(np.linalg.norm(np.array(nb[x][:2]) - apx)) for x in ("banana", "orange", "pear"))
        N["per_scene"][str(ep)] = {
            "nearest_fruit_m": near, "pre_close_chunks": fc,
            "neighbour_frac_median": {"serving": float(np.median([r["frac_neighbour"] for r in s])), "SPL": float(np.median([r["frac_neighbour"] for r in f]))},
            "ext_mm_median": {"serving": float(np.median([r["ext_mm"] for r in s])), "SPL": float(np.median([r["ext_mm"] for r in f]))},
            "anchor_ext_mm_last": {"serving": max(s[-1]["anchor_ext"]) if s[-1].get("anchor_ext") else None,
                                   "SPL": max(f[-1]["anchor_ext"]) if f[-1].get("anchor_ext") else None},
            "n_supersede": int(sum(r.get("reason") == "split_superseded_anchor" for r in F[:fc])),
            "pre_close_identical": bool(all(x.get("n") == y.get("n") and x.get("ext_mm") == y.get("ext_mm") for x, y in zip(S[:fc], F[:fc])))}
    N["n_scenes_pre_close_identical"] = int(sum(d["pre_close_identical"] for d in N["per_scene"].values()))
    N["n_scenes_parity_all_chunks"] = int(sum(d["n_identical"] == d["n_chunks"] for d in N["parity_serving_vs_record"].values()))
    held = {}
    for ep in (250, 572):
        held[str(ep)] = [{"k": r["k"], "state_srv": r["state_srv"], "serving": R["serving"][ep][i].get("state"), "SPL": r.get("state")}
                         for i, r in enumerate(R["SPL"][ep]) if r["state_srv"] == "held"]
    N["held_phase_states"] = held
    (a.out / "numbers.json").write_text(json.dumps(N, indent=1, default=float))

    # ---------------------------------------------------------------- figure: top view of the target points + 24-scene bars
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.9], hspace=0.45, wspace=0.18)
    side = {"scenes": {}}
    for j, ep in enumerate(MERGED_SHOW):
        for i, name in enumerate(("serving", "SPL")):
            ax = fig.add_subplot(gs[i, j]); FS.style_axes(fig, [ax])
            pts, tgt, others = PTS[(ep, name)]
            k_row = next(r for r in R[name][ep] if r["k"] == SHOW_K)
            gt = tgt[0]
            cl, raw = pts["cloud"], pts["raw"]
            sc = ax.scatter(cl[:, 1] * 100, cl[:, 0] * 100, c=np.log10(np.maximum(raw, 1e-6)), cmap=FS.sequential_cmap(), s=4, vmin=-6, vmax=0, lw=0)
            P = pts.get("target")
            if P is not None:
                da = np.linalg.norm(P - gt, axis=1); do = np.min(np.stack([np.linalg.norm(P - o, axis=1) for o in others]), axis=0)
                isn = do < da
                ax.scatter(P[~isn, 1] * 100, P[~isn, 0] * 100, s=16, facecolor="none", edgecolor=FS.CATEGORICAL[7], lw=0.8,
                           label=T("target 점 · 사과", "target pts · apple"))
                ax.scatter(P[isn, 1] * 100, P[isn, 0] * 100, s=16, facecolor="none", edgecolor=FS.CATEGORICAL[3], lw=0.8,
                           label=T("target 점 · 이웃 과일", "target pts · neighbour"))
            nbn = min(("banana", "orange", "pear"), key=lambda f: np.linalg.norm(np.array(bodies[str(ep)][f][:2]) - gt[:2]))
            nbp = np.array(bodies[str(ep)][nbn])
            ax.plot(gt[1] * 100, gt[0] * 100, "+", color=FS.INK, ms=9); ax.plot(nbp[1] * 100, nbp[0] * 100, "x", color=FS.INK, ms=7)
            ax.set_aspect("equal"); ax.set_xlim(gt[1] * 100 - 11, gt[1] * 100 + 11); ax.set_ylim(gt[0] * 100 - 9, gt[0] * 100 + 13)
            lab = T("기존 방식", "current") if name == "serving" else T("수정 (SPL)", "fixed (SPL)")
            ax.set_title(f"ep{ep} · {lab}\n" + T(f"{nbn} 비율 {k_row['frac_neighbour']:.2f} · 길이 {k_row['ext_mm']:.0f} mm",
                                                  f"{nbn} share {k_row['frac_neighbour']:.2f} · extent {k_row['ext_mm']:.0f} mm"),
                         fontsize=8.6, loc="left", color=FS.INK)
            ax.set_xlabel("y (cm)", fontsize=7.5); ax.set_ylabel("x (cm)", fontsize=7.5)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, frameon=False, loc="upper left")
            side["scenes"].setdefault(str(ep), {})[name] = {"k": SHOW_K, "frac_neighbour": k_row["frac_neighbour"], "ext_mm": k_row["ext_mm"],
                                                            "n_target": None if P is None else int(len(P)), "neighbour": nbn}
    cax = fig.add_axes([0.92, 0.42, 0.008, 0.4]); cb = fig.colorbar(sc, cax=cax); cb.set_label(T("log10 raw attention", "log10 raw attention"), fontsize=8)
    bx = fig.add_subplot(gs[2, :]); FS.style_axes(fig, [bx])
    order = sorted(eps, key=lambda e: N["per_scene"][str(e)]["nearest_fruit_m"])
    x = np.arange(len(order)); w = 0.38
    s_v = [N["per_scene"][str(e)]["neighbour_frac_median"]["serving"] for e in order]
    f_v = [N["per_scene"][str(e)]["neighbour_frac_median"]["SPL"] for e in order]
    bx.bar(x - w / 2, s_v, w, color=FS.CATEGORICAL[3], label=T("기존 방식", "current"))
    bx.bar(x + w / 2, f_v, w, color=FS.CATEGORICAL[0], label=T("수정 (SPL)", "fixed (SPL)"))
    bx.set_xticks(x); bx.set_xticklabels([f"ep{e}\n{N['per_scene'][str(e)]['nearest_fruit_m'] * 100:.1f}" for e in order], fontsize=6.6)
    bx.set_ylabel(T("target 점 중 이웃 과일 비율\n(첫 닫기 전 median)", "neighbour share of target pts\n(median before first close)"), fontsize=8)
    bx.legend(fontsize=8, frameon=False); bx.grid(axis="y", color=FS.GRID_INK, lw=0.5); bx.set_axisbelow(True)
    bx.set_title(T(f"24 장면 (아래 숫자 = 가장 가까운 과일까지 cm). 첫 닫기 전 수정 전후가 같은 장면 {N['n_scenes_pre_close_identical']}/24, "
                   f"기존 방식 replay가 기록과 모든 chunk 일치 {N['n_scenes_parity_all_chunks']}/24",
                   f"24 scenes (cm to the nearest fruit). Identical before the first close: {N['n_scenes_pre_close_identical']}/24; "
                   f"serving replay = record on every chunk: {N['n_scenes_parity_all_chunks']}/24"), fontsize=9, loc="left", color=FS.INK)
    fig.suptitle(T(f"IFG-SPLIT offline (기록 replay, k={SHOW_K}, 위에서 본 모습): 붙어 있는 이웃 과일을 attention 으로 떼어 낸다 (+ = 사과 중심, × = 이웃 과일 중심)",
                   f"IFG-SPLIT offline (record replay, k={SHOW_K}, top view): the touching neighbour is cut off by attention (+ apple centre, x neighbour centre)"),
                 fontsize=10.5, color=FS.INK, x=0.01, ha="left", y=0.995)
    p = a.fig / "ifg-split-offline.png"
    fig.savefig(p, dpi=180, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    side.update({k: N[k] for k in ("n_scenes_pre_close_identical", "n_scenes_parity_all_chunks")})
    side["per_scene"] = N["per_scene"]; side["sources"] = {"run": str(a.run), "script": "AG3S/ag3s/experiments/studies/ifg_split_replay.py"}
    p.with_suffix(".json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=float))
    print(p, a.out / "numbers.json")


if __name__ == "__main__":
    main()
