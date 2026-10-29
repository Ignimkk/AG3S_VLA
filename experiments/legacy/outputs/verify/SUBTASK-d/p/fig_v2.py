"""SUBTASK-d figures (rule A) from p/analyze_v2.py output. Every png gets a .json sidecar with the numbers it draws.
  subtask-d-scene-<ep>_s<seed>.png : one (episode, seed) pair, gate off (rows 1-2) vs gate on (rows 3-4), three chunks
       (last chunk before latch PLACED, first chunk where gate-on released the target, +N chunks). Row a: third-person front
       frame (mp4 frame = control step (t-1)*8, clamped). Row b: top view in the robot base frame of the server record of that
       chunk: ESDF main tier (5 mm) xy slice at the z of the carve ball (or of the true apple), filtered cloud (grey),
       attention-weighted points (top 3 % attention, orange), target points (red), manipulated centroid (x), target-free
       ball (dashed circle) and the voxels the target-free layer frees at that slice (green), true apple centre (star, from the
       client planning row object_poses converted world -> base with FK), destination AABB (rectangle).
  subtask-d-timeline.png  : chunk timelines of the pair (latch / confirmed label / exclusion active / gate blocked), off vs on;
       plus all paired runs: after-PLACED exclusion-active fraction per run (off vs on) and PLACED -> release delay.
  subtask-d-table.png     : V0 identity, paired 2x2 + McNemar, V2 summary numbers.
usage: fig_v2.py ANALYSIS_JSON OUT_DIR EP SEED [V0_SUMMARY_JSON]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-sd")
from AG3S.ag3s.experiments.common import figstyle as fs  # noqa: E402
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P  # noqa: E402
import imageio.v2 as iio  # noqa: E402

KO = fs.use_korean()
R = json.loads(pathlib.Path(sys.argv[1]).read_text()); F = pathlib.Path(sys.argv[2]); F.mkdir(parents=True, exist_ok=True)
EP, SEED = int(sys.argv[3]), int(sys.argv[4]); V0 = json.loads(pathlib.Path(sys.argv[5]).read_text()) if len(sys.argv) > 5 else None
C = fs.CATEGORICAL
SIDEC = {"off": C[0], "on": C[1]}
LATCH = ["searching", "latched", "closing", "held", "placed"]
LATCHC = {s: c for s, c in zip(LATCH, [C[6], C[0], C[3], C[1], C[2]])}
LABELC = {"pick": C[0], "place": C[1], "home": C[2], "None": "#e6e5e0"}
PAIR = next(p for p in R["pairs"] if p["episode"] == EP and p["seed"] == SEED)
ROFF, RON = R["runs"][PAIR["off_run"]], R["runs"][PAIR["on_run"]]
MJ = P.build(); M, D = MJ[0], MJ[1]


def save(fig, name, side, dpi=130):
    p = F / name; fig.savefig(p, dpi=dpi, facecolor=fs.SURFACE); plt.close(fig)
    p.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(p)


def L(ko, en):
    return ko if KO else en


def planning(run):
    rows = [json.loads(l) for l in (pathlib.Path(run["run_dir"]) / "frames.jsonl").read_text().splitlines() if l.strip()]
    return {r["seq"]: r for r in rows if r["kind"] == "planning"}


def base_pose(qpos):
    P.set_q(M, D, qpos); x0 = P.base_to_world(M, D, [0, 0, 0])
    Rm = np.stack([P.base_to_world(M, D, e) - x0 for e in np.eye(3)], axis=1)
    return x0, Rm


# ---------------------------------------------------------------- scene
def scene():
    t_p = RON["t_placed"] or ROFF["t_placed"]
    t_rel = RON["after_placed"]["first_released_t"] or (t_p + 3)
    ts = [("before PLACED", t_p - 1), ("gate-on release", t_rel), ("release +10", min(t_rel + 10, 75))]
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 16.5), gridspec_kw=dict(height_ratios=[0.72, 1, 0.72, 1]))
    side = {"pair": {"episode": EP, "seed": SEED, "off_run": ROFF["run_dir"], "on_run": RON["run_dir"]}, "cells": []}
    for si, (sd, run) in enumerate((("off", ROFF), ("on", RON))):
        plan = planning(run); mp4 = pathlib.Path(run["run_dir"]) / "third_person_front.mp4"; rd = iio.get_reader(mp4); nfr = rd.count_frames()
        bych = {c["t"]: c for c in run["chunks"]}
        for j, (lab, t) in enumerate(ts):
            c = bych[t]; z = np.load(c["file"]); s = json.loads(str(z["summary_json"])); p = plan[t]
            k = min(max((t - 1) * 8, 0), nfr - 1)
            ax = axes[2 * si, j]; ax.imshow(rd.get_data(k)[40:420, 60:600]); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{lab}   t_step(chunk) = {t}   control step {k}", fontsize=9, color=fs.INK)
            if j == 0:
                ax.set_ylabel(f"gate {sd}\n3rd person", fontsize=10, color=SIDEC[sd])
            # top view (base frame)
            ax = axes[2 * si + 1, j]
            x0, Rm = base_pose(p["qpos"])
            apple_b = Rm.T @ (np.asarray(p["object_poses"]["apple"]["pos"]) - x0)
            tf = (s.get("esdf_stats") or {}).get("target_free")
            tf = eval(tf) if isinstance(tf, str) else (tf or {})  # str(dict) of literals written by the server record
            ball = tf.get("ball") if isinstance(tf, dict) else None
            zc = ball["centre_m"][2] if ball else float(apple_b[2])
            org, vs, shp = np.asarray(z["esdf_origin"]), float(z["esdf_voxel_size"]), np.asarray(z["esdf_shape"])
            kz = int(np.clip(round((zc - org[2]) / vs), 0, shp[2] - 1))
            sl = z["esdf_distance"][:, :, kz].astype(np.float32)
            ext = [org[0], org[0] + vs * shp[0], org[1], org[1] + vs * shp[1]]
            im = ax.imshow(np.clip(sl, 0, 0.1).T, origin="lower", extent=ext, cmap=fs.sequential_cmap().reversed(), vmin=0, vmax=0.1, alpha=0.85)
            n_freed = 0
            if "esdf_free_distance" in z.files:
                fo = np.asarray(z["esdf_free_origin"])
                if np.allclose(fo, org) and tuple(z["esdf_free_shape"]) == tuple(shp):
                    fr = z["esdf_free_distance"][:, :, kz].astype(np.float32)
                    freed = (fr - sl) > 1e-3; n_freed = int(freed.sum())
                    if n_freed:
                        ax.contourf(np.linspace(ext[0], ext[1], shp[0]), np.linspace(ext[2], ext[3], shp[1]), freed.T.astype(float), levels=[0.5, 1.5], colors=[C[2]], alpha=0.55)
            cl = z["filtered_cloud"]; av = z["attention_values"]
            near = np.abs(cl[:, 2] - zc) < 0.15
            sub = cl[near][:: max(1, near.sum() // 4000)]
            ax.scatter(sub[:, 0], sub[:, 1], s=0.6, c="#9a9a95", lw=0)
            if av.size == cl.shape[0] and av.size:
                top = av >= np.quantile(av, 0.97)
                ax.scatter(cl[top, 0], cl[top, 1], s=1.5, c=C[3], lw=0)
            tp = z["target_points"] if "target_points" in z.files else np.zeros((0, 3))
            if len(tp):
                ax.scatter(tp[:, 0], tp[:, 1], s=4, c=C[7], lw=0)
            m = s.get("manipulated") or {}
            if m.get("centroid"):
                ax.plot(m["centroid"][0], m["centroid"][1], "x", color=fs.INK, ms=10, mew=2)
            if ball:
                ax.add_patch(Circle(ball["centre_m"][:2], ball["radius_mm"] / 1000, fill=False, ls="--", color=C[7], lw=1.5))
            ax.plot(apple_b[0], apple_b[1], "*", color=C[4], ms=14, mec=fs.INK, mew=0.6)
            dst = s.get("destination") or {}
            if dst.get("aabb_min"):
                a0, a1 = dst["aabb_min"], dst["aabb_max"]
                ax.add_patch(Rectangle(a0[:2], a1[0] - a0[0], a1[1] - a0[1], fill=False, color=C[5], lw=1.2))
            ax.set_xlim(0.2, 0.8); ax.set_ylim(-0.45, 0.5); ax.set_aspect("equal")
            ex = s.get("exclusion") or {}; st = s.get("subtask") or {}
            lab_s = st.get("label") if "subtask" in s else "n/a (T39 record has no subtask key)"
            ax.set_title(f"latch {c['latch']} | label {lab_s} | manipulated {m.get('id')}\n"
                         f"exclusion {'ON' if ex.get('active') else 'off'} ({ex.get('source')}) | blocked {st.get('blocked')} | released {st.get('released')}\n"
                         f"xy slice at z = {org[2] + kz * vs:.3f} m (base), main tier 5 mm", fontsize=7.5, color=fs.INK)
            ax.set_xlabel("x base [m]", fontsize=8); ax.set_ylabel("y base [m]", fontsize=8) if j == 0 else None
            fs.style_axes(fig, ax)
            side["cells"].append({"gate": sd, "label": lab, "t": t, "mp4_frame": k, "server_chunk": c["file"], "latch": c["latch"],
                                  "subtask_label": st.get("label"), "blocked": st.get("blocked"), "released": st.get("released"),
                                  "manipulated_id": m.get("id"), "manipulated_centroid_base": m.get("centroid"),
                                  "exclusion": {k2: ex.get(k2) for k2 in ("active", "source", "mechanism")},
                                  "target_free_ball": ball, "esdf_slice_z_m": float(org[2] + kz * vs), "n_freed_voxels_in_slice": n_freed,
                                  "apple_true_base": apple_b.tolist(), "n_target_points": int(len(tp))})
        rd.close()
    h = [Patch(color=C[2], alpha=0.55, label=L("target-free layer 가 비운 voxel (이 단면)", "voxels freed by the target-free layer (this slice)")),
         plt.Line2D([], [], ls="--", color=C[7], label="target-free ball"), plt.Line2D([], [], marker="x", ls="", color=fs.INK, label="manipulated centroid"),
         plt.Line2D([], [], marker="*", ls="", color=C[4], mec=fs.INK, ms=12, label=L("실제 사과 중심", "true apple centre")),
         plt.Line2D([], [], marker="o", ls="", color=C[3], label=L("attention 상위 3 %", "top 3 % attention")),
         plt.Line2D([], [], marker="o", ls="", color=C[7], label="target points"), Patch(fill=False, color=C[5], label="destination AABB")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    cax = fig.add_axes([0.905, 0.35, 0.012, 0.3]); cb = fig.colorbar(im, cax=cax); cb.set_label("ESDF main tier distance [m] (clipped 0.1)", fontsize=8)
    fig.suptitle(L(f"ep{EP} seed {SEED}: PLACED 전후 — gate off (위) vs gate on (아래), 위에서 본 단면은 robot base frame",
                   f"ep{EP} seed {SEED}: around latch PLACED - gate off (top) vs gate on (bottom); top views in robot base frame"), fontsize=12, color=fs.INK)
    fig.subplots_adjust(top=0.95, bottom=0.06, hspace=0.32, wspace=0.12, right=0.88)
    save(fig, f"subtask-d-scene-{EP}_s{SEED}.png", side, dpi=110)


# ---------------------------------------------------------------- timeline + distribution
def band(ax, y, ts, vals, cmap, h=0.8):
    for t, v in zip(ts, vals):
        ax.add_patch(Rectangle((t - 0.5, y - h / 2), 1, h, color=cmap.get(str(v), "#e6e5e0"), lw=0))


def timeline():
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.35, wspace=0.22)
    ax = fig.add_subplot(gs[0, :])
    rows = []; side = {"pair": {"episode": EP, "seed": SEED}, "rows": {}}
    for sd, run in (("off", ROFF), ("on", RON)):
        ch = run["chunks"]; ts = [c["t"] for c in ch]
        rows += [(f"{sd}: latch", ts, [c["latch"] for c in ch], LATCHC),
                 (f"{sd}: confirmed label", ts, [str(c["label"]) for c in ch], LABELC),
                 (f"{sd}: exclusion active", ts, [str(c["excl_active"]) for c in ch], {"True": C[7], "False": "#e6e5e0"}),
                 (f"{sd}: gate blocked", ts, [str(c["blocked"]) for c in ch], {"first": C[6], "switch": C[4], "None": "#f2f1ed"})]
        hold = set(run.get("hold_t") or [])
        rows += [(f"{sd}: HOLD chunk", ts, [str(t in hold) for t in ts], {"True": fs.INK, "False": "#f2f1ed"})]
    for i, (lab, ts, vals, cm) in enumerate(rows):
        band(ax, len(rows) - 1 - i, ts, vals, cm); side["rows"][lab] = dict(zip(map(str, ts), vals))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
    ax.set_xlim(0.5, 75.5); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xlabel("chunk t_step (server) = client planning seq")
    dv = PAIR["divergence"]
    if dv.get("first_step_gt0") is not None:
        tch = dv["first_step_gt0"] // 8 + 1; ax.axvline(tch - 0.5, color=fs.INK, lw=1.2, ls=":")
        ax.annotate(L(f"실행 action 첫 차이: chunk {tch}", f"first executed-action difference: chunk {tch}"), xy=(tch - 0.5, len(rows) - 0.4), xytext=(tch + 4, len(rows) + 0.3),
                    fontsize=8, annotation_clip=False, arrowprops=dict(arrowstyle="-", color=fs.INK, lw=0.6))
    for tp, sd in ((ROFF["t_placed"], "off"), (RON["t_placed"], "on")):
        if tp:
            ax.axvline(tp - 0.5, color=SIDEC[sd], lw=1, ls="--")
    hdl = [Patch(color=v, label=f"latch {k}") for k, v in LATCHC.items()] + [Patch(color=v, label=f"label {k}") for k, v in LABELC.items()] + \
          [Patch(color=C[7], label="exclusion active"), Patch(color=C[6], label="blocked first"), Patch(color=C[4], label="blocked switch"), Patch(color=fs.INK, label="HOLD")]
    ax.legend(handles=hdl, ncol=7, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_title(L(f"ep{EP} seed {SEED}: chunk 별 latch · 확정 label · exclusion (gate off 위 5 줄, gate on 아래 5 줄; 점선 = latch PLACED)",
                   f"ep{EP} seed {SEED}: per-chunk latch, confirmed label, exclusion (off: top 5 rows, on: bottom 5; dashed = PLACED)"), fontsize=10, loc="left")
    fs.style_axes(fig, ax)
    # distribution: after-PLACED exclusion-active fraction per run, paired
    ax2 = fig.add_subplot(gs[1, 0]); pts = []
    for p in R["pairs"]:
        ro, rn = R["runs"][p["off_run"]], R["runs"][p["on_run"]]
        fo, fn = ro["after_placed"]["excl_active_frac"], rn["after_placed"]["excl_active_frac"]
        pts.append({"episode": p["episode"], "seed": p["seed"], "off": fo, "on": fn, "off_n_after": ro["after_placed"]["n_chunks"], "on_n_after": rn["after_placed"]["n_chunks"]})
        if fo is not None and fn is not None:
            ax2.plot([0, 1], [fo, fn], color=fs.GRID_INK, lw=0.8, zorder=1)
    rng = np.random.default_rng(0)
    for x, sd in ((0, "off"), (1, "on")):
        v = [q[sd] for q in pts if q[sd] is not None]
        ax2.scatter(x + rng.uniform(-0.06, 0.06, len(v)), v, s=22, color=SIDEC[sd], zorder=2, label=f"gate {sd} (n={len(v)} runs reaching PLACED)")
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["gate off (T39 E3b)", "gate on"]); ax2.set_xlim(-0.4, 1.4)
    ax2.set_ylabel(L("PLACED 뒤 exclusion active chunk 비율", "share of after-PLACED chunks with exclusion active")); ax2.legend(fontsize=8, loc="center")
    ax2.set_title(L("run 별 (같은 episode·seed 짝은 회색 선)", "per run (grey line = same episode, seed)"), fontsize=9, loc="left")
    fs.style_axes(fig, ax2)
    ax3 = fig.add_subplot(gs[1, 1])
    d_rel = [R["runs"][p["on_run"]]["after_placed"]["placed_to_release_chunks"] for p in R["pairs"] if R["runs"][p["on_run"]]["placed"]]
    d_sw = [R["runs"][p["off_run"]]["after_placed"]["placed_to_first_switch_chunks"] for p in R["pairs"] if R["runs"][p["off_run"]]["placed"]]
    bins = np.arange(-0.5, max([x for x in d_rel + d_sw if x is not None] + [10]) + 1.5, 1)
    ax3.hist([x for x in d_rel if x is not None], bins=bins, color=SIDEC["on"], alpha=0.85, label=L("gate on: PLACED → 해제 (released)", "gate on: PLACED -> release"))
    ax3.hist([x for x in d_sw if x is not None], bins=bins, color=SIDEC["off"], alpha=0.55, label=L("gate off: PLACED → 첫 switch", "gate off: PLACED -> first switch"))
    ax3.set_xlabel("chunks after latch PLACED"); ax3.set_ylabel("runs"); ax3.legend(fontsize=8)
    ax3.set_title(L(f"해제 없음 (gate on, PLACED 도달) = {sum(x is None for x in d_rel)} · switch 없음 (gate off) = {sum(x is None for x in d_sw)}",
                    f"no release (on) = {sum(x is None for x in d_rel)}, no switch (off) = {sum(x is None for x in d_sw)}"), fontsize=9, loc="left")
    fs.style_axes(fig, ax3)
    side["after_placed_excl_frac"] = pts; side["placed_to_release_chunks_on"] = d_rel; side["placed_to_first_switch_chunks_off"] = d_sw
    save(fig, "subtask-d-timeline.png", side)


# ---------------------------------------------------------------- table
def table():
    S = R["summary"]; T = R["paired_2x2"]; so, sn = S["off"], S["on"]
    def st(x, k="median"):
        return "-" if not x else f"{x[k]:.1f}"
    rows = [
        ("runs (pairs)", so["n_runs"], sn["n_runs"]),
        ("success / grasp / place", f'{so["n_success"]} / {so["n_grasp"]} / {so["n_place"]}', f'{sn["n_success"]} / {sn["n_grasp"]} / {sn["n_place"]}'),
        ("runs reaching latch PLACED", so["n_placed_latch"], sn["n_placed_latch"]),
        ("  of those, runs with switch after PLACED", so["n_placed_runs_with_switch_after"], sn["n_placed_runs_with_switch_after"]),
        ("  switch decisions after PLACED", so["n_switch_after_placed_total"], sn["n_switch_after_placed_total"]),
        ("  exclusion active / chunks after PLACED", f'{so["after_placed_excl_active_total"]} / {so["after_placed_chunks_total"]}', f'{sn["after_placed_excl_active_total"]} / {sn["after_placed_chunks_total"]}'),
        ("  carve centre: nearest object (counts)", json.dumps(so["after_placed_carved_nearest_counts"]), json.dumps(sn["after_placed_carved_nearest_counts"])),
        ("  carve centre to apple, median mm", st(so["after_placed_carved_to_apple_mm"]), st(sn["after_placed_carved_to_apple_mm"])),
        ("runs with released / released chunks", f'{so["n_runs_released"]} / {so["n_released_chunks_total"]}', f'{sn["n_runs_released"]} / {sn["n_released_chunks_total"]}'),
        ("switch decisions before PLACED", so["n_switch_before_placed_total"], sn["n_switch_before_placed_total"]),
        ("gate blocked before PLACED (chunks)", len(so["blocked_before_placed"]), len(sn["blocked_before_placed"])),
        ("gate blocked all chunks", json.dumps(so["blocked_counts_all"]), json.dumps(sn["blocked_counts_all"])),
        ("HOLD chunks after PLACED / from first release", f'{so["n_hold_after_placed_total"]} / {so["n_hold_from_release_total"]}', f'{sn["n_hold_after_placed_total"]} / {sn["n_hold_from_release_total"]} (of {sn["n_chunks_from_release_total"]})'),
        ("HOLD kinds from first release", json.dumps(so["hold_kind_from_release"]), json.dumps(sn["hold_kind_from_release"])),
        ("HOLD kinds all chunks", json.dumps(so["hold_kind_counts"]), json.dumps(sn["hold_kind_counts"])),
        ("release chunk - place chunk (runs)", "-", json.dumps(sorted(sn["release_chunk_minus_place_chunk"]))),
        ("HOLD chunks / runs with HOLD", f'{so["n_hold_chunks_total"]} / {so["n_runs_with_hold"]}', f'{sn["n_hold_chunks_total"]} / {sn["n_runs_with_hold"]}'),
        ("gate action (execute / hold)", f'{so["gate_action_counts"].get("execute", 0)} / {so["gate_action_counts"].get("hold", 0)}', f'{sn["gate_action_counts"].get("execute", 0)} / {sn["gate_action_counts"].get("hold", 0)}'),
        ("clearance min over runs, mm", f'{so["clearance_min_mm_overall"]:.1f}', f'{sn["clearance_min_mm_overall"]:.1f}'),
        ("chunks with clearance < 0", so["n_chunks_clearance_lt0"], sn["n_chunks_clearance_lt0"]),
        ("SQP time-budget hits (chunks)", so["n_budget_hit"], sn["n_budget_hit"]),
        ("round trip median / p95 ms (excl. 1st)", f'{st(so["time_ms_pooled_excl_first"]["round_trip"])} / {st(so["time_ms_pooled_excl_first"]["round_trip"], "p95")}', f'{st(sn["time_ms_pooled_excl_first"]["round_trip"])} / {st(sn["time_ms_pooled_excl_first"]["round_trip"], "p95")}'),
        ("server ag3s / trajopt / infer median ms", "/".join(st(so["time_ms_pooled_excl_first"][f"server_{k}"]) for k in ("ag3s", "trajopt", "infer")), "/".join(st(sn["time_ms_pooled_excl_first"][f"server_{k}"]) for k in ("ag3s", "trajopt", "infer"))),
    ]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 12), gridspec_kw=dict(height_ratios=[0.33, 1]))
    for a in (a1, a2):
        a.axis("off")
    hdr = ["", "both", "off only", "on only", "neither", "n pairs", "exact McNemar p"]
    cell = [[f] + [T[f][k] for k in ("both", "off_only", "on_only", "neither", "n_pairs")] + [f'{T[f]["mcnemar_exact_p"]:.3f}'] for f in ("success", "grasp", "place")]
    t1 = a1.table(cellText=cell, colLabels=hdr, loc="center", cellLoc="center"); a1.set_title(L("짝 2×2 (gate off = T39 E3b, gate on = SUBTASK-d), (episode, seed) 짝", "paired 2x2 (off = T39 E3b, on = SUBTASK-d), paired by (episode, seed)"), fontsize=10, loc="left")
    t2 = a2.table(cellText=[[a, str(b), str(c)] for a, b, c in rows], colLabels=["", "gate off (T39 E3b)", "gate on"], loc="upper center", cellLoc="left", colWidths=[0.42, 0.29, 0.29])
    v0txt = ""
    if V0:
        v0txt = "V0: " + "; ".join(f'{n}: executed actions equal {v["executed_actions_equal"]}, latch/manip/exclusion equal {v["named_per_chunk"]["grasp_state_equal"] and v["named_per_chunk"]["manipulated_id_equal"] and v["named_per_chunk"]["exclusion_equal"]}' for n, v in V0["runs"].items())
    a2.set_title(L("V2 숫자 (전부 analysis json 에서)\n", "V2 numbers (all from the analysis json)\n") + v0txt, fontsize=9, loc="left")
    for tb in (t1, t2):
        tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.75)
        for (r, c), cl in tb.get_celld().items():
            cl.set_edgecolor(fs.GRID_INK)
            if r == 0:
                cl.set_facecolor("#efeeea")
    fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
    save(fig, "subtask-d-table.png", {"paired_2x2": T, "rows": [[a, b, c] for a, b, c in rows], "v0": V0 and {n: {k: v[k] for k in ("executed_actions_equal", "named_per_chunk", "other_values")} for n, v in V0["runs"].items()}})


if __name__ == "__main__":
    which = [a for a in sys.argv if a.startswith("--only=")]
    todo = which[0][7:].split(",") if which else ["scene", "timeline", "table"]
    for f in todo:
        globals()[f]()
