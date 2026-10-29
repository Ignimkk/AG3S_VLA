"""SUBTASK-e figures (rule A) from p/analyze_e.py output. Every png gets a .json sidecar with the numbers it draws.
  subtask-e-scene-<ep>_s<seed>.png : one (episode, seed), gate on before the change (SUBTASK-d, rows 1-2) vs after (SUBTASK-e, rows 3-4),
       four chunks (t_rel - 1, release chunk t_rel, t_rel + 10, last chunk 75). Row a: third-person front frame (mp4 frame =
       control step (t-1)*8 + 7, i.e. the end of that chunk, clamped). Row b: top view in the robot base frame of the server record
       of that chunk: ESDF main tier (5 mm) xy slice at the z of the true apple centre, filtered cloud (grey), top 3 % attention
       (orange), target points (red), manipulated centroid (x), true apple (star), destination AABB; inset: side view (x-z) of the
       filtered cloud with the slice plane (line).
  subtask-e-timeline.png : (a) per-chunk latch / confirmed label / gate blocked / released / HOLD / subtask_no_target reason of the
       pair, before vs after; (b) left-arm joint distance to the ready pose per control step (off, on_d, on_e) of the pair;
       (c) HOLD chunks from release per run, before vs after; (d) left-arm joint distance to the ready pose at the last step per run.
  subtask-e-table.png : regression baseline, V0, paired 2x2 + McNemar (off vs after, before vs after), summary numbers.
usage: fig_e.py ANALYSIS_JSON OUT_DIR EP SEED [V0_SUMMARY_JSON] [BASELINE_JUDGE_JSON] [--only=scene,timeline,table]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-se")
from AG3S.ag3s.experiments.common import figstyle as fs  # noqa: E402
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P  # noqa: E402
import imageio.v2 as iio  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
KO = fs.use_korean()
R = json.loads(pathlib.Path(ARGS[0]).read_text()); F = pathlib.Path(ARGS[1]); F.mkdir(parents=True, exist_ok=True)
EP, SEED = int(ARGS[2]), int(ARGS[3])
V0 = json.loads(pathlib.Path(ARGS[4]).read_text()) if len(ARGS) > 4 else None
BL = json.loads(pathlib.Path(ARGS[5]).read_text()) if len(ARGS) > 5 else None
C = fs.CATEGORICAL
SIDES = ("off", "on_d", "on_e")
SIDEC = {"off": C[0], "on_d": C[1], "on_e": C[2]}
SIDEN = {"off": "gate off (T39 E3b)", "on_d": "gate on, before (SUBTASK-d)", "on_e": "gate on, after (SUBTASK-e)"}
LATCH = ["searching", "latched", "closing", "held", "placed"]
LATCHC = {s: c for s, c in zip(LATCH, [C[6], C[0], C[3], C[1], C[2]])}
LABELC = {"pick": C[0], "place": C[1], "home": C[2], "None": "#e6e5e0"}
PR = R["per_run"]
KEY = f"{EP}_s{SEED}"
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
    rd_, re_ = PR[f"on_d/{KEY}"], PR[f"on_e/{KEY}"]
    t_rel = re_["t_release"] or rd_["t_release"]
    ts = [(L("해제 직전", "before release"), t_rel - 1), (L("해제 chunk", "release chunk"), t_rel), ("release +10", min(t_rel + 10, 75)), (L("마지막 chunk", "last chunk"), 75)]
    opt = [a for a in sys.argv if a.startswith("--ts=")]
    if opt:
        tt = [int(v) for v in opt[0][5:].split(",")]
        ts = [(L(f"해제 {t - t_rel:+d} chunk", f"release {t - t_rel:+d}") if t != t_rel else L("해제 chunk", "release chunk"), t) for t in tt]
    fig, axes = plt.subplots(4, 4, figsize=(17.5, 17.5), gridspec_kw=dict(height_ratios=[0.72, 1, 0.72, 1]))
    side = {"pair": {"episode": EP, "seed": SEED, "on_d_run": rd_["run_dir"], "on_e_run": re_["run_dir"]}, "cells": []}
    im = None
    for si, (sd, run) in enumerate((("on_d", rd_), ("on_e", re_))):
        plan = planning(run); mp4 = pathlib.Path(run["run_dir"]) / "third_person_front.mp4"; rd = iio.get_reader(mp4); nfr = rd.count_frames()
        bych = {c["t"]: c for c in run["rows"]}
        for j, (lab, t) in enumerate(ts):
            c = bych[t]; z = np.load(c["file"]); s = json.loads(str(z["summary_json"])); p = plan[t]
            k = min(max((t - 1) * 8 + 7, 0), nfr - 1)
            ax = axes[2 * si, j]; ax.imshow(rd.get_data(k)[40:420, 60:600]); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{lab}   chunk t = {t}   control step {k}\nHOLD {c['hold']} {c['hold_kinds'] or ''}  gate {c['gate_action']}", fontsize=8.5, color=fs.INK)
            if j == 0:
                ax.set_ylabel(f"{SIDEN[sd]}\n3rd person", fontsize=10, color=SIDEC[sd])
            ax = axes[2 * si + 1, j]
            x0, Rm = base_pose(p["qpos"])
            apple_b = Rm.T @ (np.asarray(p["object_poses"]["apple"]["pos"]) - x0)
            zc = float(apple_b[2])
            org, vs, shp = np.asarray(z["esdf_origin"]), float(z["esdf_voxel_size"]), np.asarray(z["esdf_shape"])
            kz = int(np.clip(round((zc - org[2]) / vs), 0, shp[2] - 1)); zs = float(org[2] + kz * vs)
            sl = z["esdf_distance"][:, :, kz].astype(np.float32)
            ext = [org[0], org[0] + vs * shp[0], org[1], org[1] + vs * shp[1]]
            im = ax.imshow(np.clip(sl, 0, 0.1).T, origin="lower", extent=ext, cmap=fs.sequential_cmap().reversed(), vmin=0, vmax=0.1, alpha=0.85)
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
            ax.plot(apple_b[0], apple_b[1], "*", color=C[4], ms=14, mec=fs.INK, mew=0.6)
            dst = s.get("destination") or {}
            if dst.get("aabb_min"):
                a0, a1 = dst["aabb_min"], dst["aabb_max"]
                ax.add_patch(Rectangle(a0[:2], a1[0] - a0[0], a1[1] - a0[1], fill=False, color=C[5], lw=1.2))
            ax.set_xlim(0.2, 0.8); ax.set_ylim(-0.45, 0.5); ax.set_aspect("equal")
            ex = s.get("exclusion") or {}; st = s.get("subtask") or {}
            vr = [x.get("kind") for x in ((s.get("verdict") or {}).get("reasons") or []) if isinstance(x, dict)]
            ax.set_title(f"latch {c['latch']} | label {st.get('label')} | released {st.get('released')}\n"
                         f"status {s.get('status', '')} grounding {s.get('grounding_status')} | exclusion {'ON' if ex.get('active') else 'off'}\n"
                         f"reasons {c['reason_kinds']} | clearance {c['clearance_min_mm'] if c['clearance_min_mm'] is None else round(c['clearance_min_mm'], 1)} mm\n"
                         f"xy slice z = {zs:.3f} m (base), main tier 5 mm", fontsize=7.2, color=fs.INK)
            ax.set_xlabel("x base [m]", fontsize=8)
            if j == 0:
                ax.set_ylabel("y base [m]", fontsize=8)
            fs.style_axes(fig, ax)
            ins = ax.inset_axes([0.56, 0.01, 0.43, 0.33])
            sv = cl[(np.abs(cl[:, 1] - apple_b[1]) < 0.1)]; sv = sv[:: max(1, len(sv) // 3000)]
            ins.scatter(sv[:, 0], sv[:, 2], s=1.2, c="#7a7a75", lw=0); ins.axhline(zs, color=C[7], lw=1.2)
            ins.plot(apple_b[0], apple_b[2], "*", color=C[4], ms=7, mec=fs.INK, mew=0.4)
            ins.set_xlim(0.2, 0.8); ins.set_ylim(zs - 0.35, zs + 0.35); ins.set_xticks([]); ins.set_yticks([])
            ins.set_title(L("옆에서 (x-z): 단면 위치", "side view (x-z): slice plane"), fontsize=6, pad=1.5); ins.set_facecolor("#ffffffd0")
            side["cells"].append({"side": sd, "label": lab, "t": t, "mp4_frame": k, "server_chunk": c["file"], "latch": c["latch"],
                                  "subtask_label": st.get("label"), "blocked": st.get("blocked"), "released": st.get("released"),
                                  "grounding_status": s.get("grounding_status"), "server_verdict_reason_kinds": vr,
                                  "client_reason_kinds": c["reason_kinds"], "hold": c["hold"], "hold_kinds": c["hold_kinds"], "gate_action": c["gate_action"],
                                  "clearance_min_mm": c["clearance_min_mm"], "manipulated_id": m.get("id"),
                                  "exclusion": {k2: ex.get(k2) for k2 in ("active", "source", "mechanism")},
                                  "esdf_slice_z_m": zs, "apple_true_base": apple_b.tolist(), "n_target_points": int(len(tp))})
        rd.close()
    h = [plt.Line2D([], [], marker="x", ls="", color=fs.INK, label="manipulated centroid"),
         plt.Line2D([], [], marker="*", ls="", color=C[4], mec=fs.INK, ms=12, label=L("실제 사과 중심", "true apple centre")),
         plt.Line2D([], [], marker="o", ls="", color=C[3], label=L("attention 상위 3 %", "top 3 % attention")),
         plt.Line2D([], [], marker="o", ls="", color=C[7], label="target points"), Patch(fill=False, color=C[5], label="destination AABB"),
         plt.Line2D([], [], color=C[7], label=L("단면 높이 (inset)", "slice height (inset)"))]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=8.5, frameon=False)
    cax = fig.add_axes([0.915, 0.35, 0.01, 0.3]); cb = fig.colorbar(im, cax=cax); cb.set_label("ESDF main tier distance [m] (clipped 0.1)", fontsize=8)
    fig.suptitle(L(f"ep{EP} seed {SEED}: B3 해제 전후 — gate on 수정 전 (위, SUBTASK-d) vs 수정 후 (아래, SUBTASK-e); 위에서 본 단면은 robot base frame",
                   f"ep{EP} seed {SEED}: around the B3 release - gate on before the change (top, SUBTASK-d) vs after (bottom, SUBTASK-e)"), fontsize=12, color=fs.INK)
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.36, wspace=0.12, right=0.9, left=0.05)
    save(fig, f"subtask-e-scene-{EP}_s{SEED}.png", side, dpi=105)


# ---------------------------------------------------------------- timeline
def band(ax, y, ts, vals, cmap, h=0.8):
    for t, v in zip(ts, vals):
        ax.add_patch(Rectangle((t - 0.5, y - h / 2), 1, h, color=cmap.get(str(v), "#e6e5e0"), lw=0))


def timeline():
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 0.8, 0.9], hspace=0.42, wspace=0.2)
    ax = fig.add_subplot(gs[0, :])
    rows = []; side = {"pair": {"episode": EP, "seed": SEED}, "rows": {}}
    for sd in ("on_d", "on_e"):
        run = PR[f"{sd}/{KEY}"]; ch = run["rows"]; ts = [c["t"] for c in ch]
        tag = L("수정 전", "before") if sd == "on_d" else L("수정 후", "after")
        rows += [(f"{tag}: latch", ts, [c["latch"] for c in ch], LATCHC),
                 (f"{tag}: confirmed label", ts, [str(c["label"]) for c in ch], LABELC),
                 (f"{tag}: gate blocked", ts, [str(c["blocked"]) for c in ch], {"first": C[6], "switch": C[4], "None": "#f2f1ed"}),
                 (f"{tag}: released", ts, [str(c["released"]) for c in ch], {"True": C[5], "False": "#f2f1ed"}),
                 (f"{tag}: HOLD", ts, [str(c["hold"]) for c in ch], {"True": fs.INK, "False": "#f2f1ed"}),
                 (f"{tag}: reason subtask_no_target", ts, [str("subtask_no_target" in c["reason_kinds"]) for c in ch], {"True": C[7], "False": "#f2f1ed"})]
    for i, (lab, ts, vals, cm) in enumerate(rows):
        band(ax, len(rows) - 1 - i, ts, vals, cm); side["rows"][lab] = dict(zip(map(str, ts), vals))
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
    ax.set_xlim(0.5, 75.5); ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xlabel("chunk t_step (server) = client planning seq")
    idn = R["identity_on_e_vs_on_d"].get(KEY, {})
    if idn.get("earliest_any_t") is not None:
        t0 = idn["earliest_any_t"]; ax.axvline(t0 - 0.5, color=fs.INK, lw=1.2, ls=":")
    hdl = [Patch(color=v, label=f"latch {k}") for k, v in LATCHC.items()] + [Patch(color=v, label=f"label {k}") for k, v in LABELC.items()] + \
          [Patch(color=C[6], label="blocked first"), Patch(color=C[4], label="blocked switch"), Patch(color=C[5], label="released"),
           Patch(color=fs.INK, label="HOLD"), Patch(color=C[7], label="reason subtask_no_target")]
    ax.legend(handles=hdl, ncol=8, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    t0s = idn.get("earliest_any_t")
    ax.set_title(L(f"(a) ep{EP} seed {SEED}: chunk 별 latch · label · gate · HOLD (gate on 수정 전 위 6 줄, 수정 후 아래 6 줄); 점선 = 수정 전/후 첫 차이 chunk {t0s}",
                   f"(a) ep{EP} seed {SEED}: per-chunk latch, label, gate, HOLD (before: top 6, after: bottom 6); dotted = first before/after difference, chunk {t0s}"), fontsize=10, loc="left")
    fs.style_axes(fig, ax)
    side["identity"] = idn
    # (b) joint distance to ready pose
    ax = fig.add_subplot(gs[1, :]); side["dL"] = {}
    for sd in SIDES:
        run = PR.get(f"{sd}/{KEY}")
        if not run:
            continue
        y = run["series"]["dL"]; ax.plot(range(len(y)), y, color=SIDEC[sd], lw={"off": 3.2, "on_d": 1.6, "on_e": 1.6}[sd], ls={"off": "-", "on_d": "-", "on_e": "--"}[sd], label=SIDEN[sd]); side["dL"][sd] = y
        if run["s_ref"] is not None:
            ax.axvline(run["s_ref"], color=SIDEC[sd], lw=1, ls="--")
    ax.set_xlabel("control step"); ax.set_ylabel(L("왼팔 관절 거리 ||q - q_ready|| [rad]", "left-arm ||q - q_ready|| [rad]"))
    ax.set_title(L("(b) 준비 자세 (control step 0 의 관절값) 까지의 왼팔 관절 거리; 점선 = 해제 chunk 시작 (off: PLACED chunk 시작)",
                   "(b) left-arm joint distance to the ready pose (step-0 joints); dashed = release chunk start (off: PLACED chunk start)"), fontsize=9.5, loc="left")
    ax.legend(fontsize=8); fs.style_axes(fig, ax)
    # (c) HOLD from release per run
    ax = fig.add_subplot(gs[2, 0]); keys = sorted({k.split("/", 1)[1] for k in PR}, key=lambda s: (int(s.split("_s")[0]), int(s.split("_s")[1])))
    rel = [k for k in keys if PR.get(f"on_d/{k}", {}).get("t_release") is not None or PR.get(f"on_e/{k}", {}).get("t_release") is not None]
    x = np.arange(len(rel)); w = 0.4; side["hold_from_release"] = {}
    for i, sd in enumerate(("on_d", "on_e")):
        v = [PR[f"{sd}/{k}"]["regions"]["from_release"]["n_hold"] if f"{sd}/{k}" in PR else 0 for k in rel]
        n = [PR[f"{sd}/{k}"]["regions"]["from_release"]["n_chunks"] if f"{sd}/{k}" in PR else 0 for k in rel]
        ax.bar(x + (i - 0.5) * w, n, w, color=SIDEC[sd], alpha=0.25)
        ax.bar(x + (i - 0.5) * w, v, w, color=SIDEC[sd], label=f"{SIDEN[sd]}: HOLD {sum(v)} / {sum(n)} chunks")
        side["hold_from_release"][sd] = {"runs": rel, "n_hold": v, "n_chunks": n}
    ax.set_xticks(x); ax.set_xticklabels(rel, rotation=70, fontsize=7); ax.set_ylabel(L("해제 뒤 chunk 수", "chunks from release"))
    ax.set_title(L("(c) run 별 해제 뒤 HOLD (진한 막대) / 해제 뒤 chunk (옅은 막대)", "(c) per run: HOLD from release (solid) / chunks from release (light)"), fontsize=9.5, loc="left")
    ax.set_ylim(0, 1.45 * max(1, max(max(side["hold_from_release"][sd]["n_chunks"] or [1]) for sd in ("on_d", "on_e"))))
    ax.legend(fontsize=7.5, loc="upper center", ncol=1); fs.style_axes(fig, ax)
    # (d) dL at last step, paired
    ax = fig.add_subplot(gs[2, 1]); side["dL_last"] = {}
    for k in rel:
        v = [PR.get(f"{sd}/{k}", {}).get("pose", {}).get("dL_rad_last") for sd in SIDES]
        side["dL_last"][k] = dict(zip(SIDES, v))
        if all(a is not None for a in v):
            ax.plot(range(3), v, color=fs.GRID_INK, lw=0.8, zorder=1)
    rng = np.random.default_rng(0)
    for i, sd in enumerate(SIDES):
        v = [side["dL_last"][k][sd] for k in rel if side["dL_last"][k][sd] is not None]
        ax.scatter(i + rng.uniform(-0.05, 0.05, len(v)), v, s=22, color=SIDEC[sd], zorder=2)
        if v:
            ax.plot([i - 0.2, i + 0.2], [np.median(v)] * 2, color=fs.INK, lw=1.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([SIDEN[s].replace(", ", ",\n") for s in SIDES], fontsize=8)
    ax.set_ylabel(L("마지막 step 의 왼팔 ||q - q_ready|| [rad]", "left-arm ||q - q_ready|| at step 599 [rad]"))
    ax.set_title(L(f"(d) 해제가 있었던 run ({len(rel)} 짝) 의 마지막 자세 — 가로선 = median, 회색 선 = 같은 (episode, seed)",
                   f"(d) final pose of the {len(rel)} pairs with a release - bar = median, grey = same (episode, seed)"), fontsize=9.5, loc="left")
    fs.style_axes(fig, ax)
    save(fig, "subtask-e-timeline.png", side)


# ---------------------------------------------------------------- table
def table():
    S = R["summary"]; T = R["paired_2x2"]; I = R["identity_summary"]
    def g(sd, *ks, default="-"):
        x = S[sd]
        for k in ks:
            if not isinstance(x, dict) or k not in x:
                return default
            x = x[k]
        return x
    def j(x):
        return json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
    def fmt1(x):
        return "-" if x is None else f"{x:.1f}"
    def st(x, k="median", f="{:.2f}"):
        return "-" if not x else f.format(x[k])
    rows = [("runs", *[g(s, "n_runs") for s in SIDES]),
            ("success / grasp / place", *[f'{g(s, "n_success")} / {g(s, "n_grasp")} / {g(s, "n_place")}' for s in SIDES]),
            ("runs latch PLACED / runs released", *[f'{g(s, "n_placed_latch")} / {g(s, "n_runs_released")}' for s in SIDES]),
            ("HOLD chunks / runs with HOLD", *[f'{g(s, "n_hold_chunks_total")} / {g(s, "n_runs_with_hold")}' for s in SIDES]),
            ("HOLD kinds (all chunks)", *[j(g(s, "hold_kind_counts")) for s in SIDES]),
            ("from release: HOLD / chunks", *[f'{g(s, "regions", "from_release", "n_hold")} / {g(s, "regions", "from_release", "n_chunks")}' for s in SIDES]),
            ("from release: HOLD kinds", *[j(g(s, "regions", "from_release", "hold_kinds")) for s in SIDES]),
            ("from release: grounding_status", *[j(g(s, "regions", "from_release", "grounding_status")) for s in SIDES]),
            ("from release: TO status", *[j(g(s, "regions", "from_release", "to_status")) for s in SIDES]),
            ("from release: clearance min mm / chunks < 0", *[f'{fmt1(g(s, "regions", "from_release", "clearance_min_mm"))} / {g(s, "regions", "from_release", "n_clearance_lt0")}' for s in SIDES]),
            ("before release: HOLD / chunks", *[f'{sum(g(s, "regions", r, "n_hold") for r in ("before_first_held", "held_before_placed", "placed_before_release"))} / {sum(g(s, "regions", r, "n_chunks") for r in ("before_first_held", "held_before_placed", "placed_before_release"))}' for s in SIDES]),
            ("subtask_no_target chunks (client / server) / runs", *[f'{g(s, "snt", "n_chunks")} / {g(s, "snt", "n_chunks_server")} / {g(s, "snt", "n_runs")}' for s in SIDES]),
            ("subtask_no_target by region", *[j(g(s, "snt", "by_region")) for s in SIDES]),
            ("subtask_no_target by blocked", *[j(g(s, "snt", "by_blocked")) for s in SIDES]),
            ("subtask_no_target chunks that are HOLD", *[g(s, "snt", "n_hold_among") for s in SIDES]),
            ("gate blocked = first chunks (all)", *[len(g(s, "first_blocked", default=[])) for s in SIDES]),
            ("pose: ||q_L - q_ready|| at ref, median rad", *[st(g(s, "pose", "dL_rad_at_ref", default=None)) for s in SIDES]),
            ("pose: ||q_L - q_ready|| last step, median rad", *[st(g(s, "pose", "dL_rad_last", default=None)) for s in SIDES]),
            ("pose: EE distance to step 0, last step, median mm", *[st(g(s, "pose", "dEE_mm_last", default=None), f="{:.0f}") for s in SIDES]),
            ("pose: left-arm path from ref, median rad", *[st(g(s, "pose", "L_path_rad_from_ref", default=None)) for s in SIDES]),
            ("pose: ref kind (runs)", *[j(g(s, "pose", "ref_kind")) for s in SIDES]),
            ("MuJoCo contact steps from ref (robot vs object)", *[j({k: v["n_steps"] for k, v in g(s, "contacts_from_ref").items() if v["n_steps"]}) for s in SIDES]),
            ("clearance min over runs mm / chunks < 0", *[f'{g(s, "clearance_min_mm_overall"):.1f} / {g(s, "n_chunks_clearance_lt0")}' for s in SIDES]),
            ("SQP time-budget hits (chunks)", *[g(s, "n_budget_hit") for s in SIDES])]
    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(16, 19), gridspec_kw=dict(height_ratios=[0.33, 0.42, 1.6]))
    for a in (a0, a1, a2):
        a.axis("off")
    hdr = ["", "both", "A only", "B only", "neither", "n pairs", "exact McNemar p"]
    cell = []
    for A_, B_ in (("off", "on_e"), ("on_d", "on_e"), ("off", "on_d")):
        for f in ("success", "grasp", "place"):
            t = T[f"{A_}_vs_{B_}:{f}"]
            cell.append([f"A = {A_}, B = {B_}: {f}", t["both"], t[f"{A_}_only"], t[f"{B_}_only"], t["neither"], t["n_pairs"], f'{t["mcnemar_exact_p"]:.3f}'])
    t1 = a1.table(cellText=cell, colLabels=hdr, cellLoc="center", colWidths=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.14], bbox=[0, 0, 1, 0.9])
    a1.set_title(L("짝 2×2 · exact McNemar (off = T39 E3b, on_d = SUBTASK-d 수정 전, on_e = SUBTASK-e 수정 후), (episode, seed) 짝",
                   "paired 2x2, exact McNemar (off = T39 E3b, on_d = SUBTASK-d before, on_e = SUBTASK-e after)"), fontsize=10, loc="left")
    top = []
    if BL:
        for b in ("legacy", "curobo"):
            top.append([f"baseline {b}", f'{BL[b]["violated_start"]} · {BL[b]["has_target"]} · {BL[b]["frame0_clearance_before_mm"]!r}', str(BL[b]["matched"])])
    if V0:
        for n, v in V0["runs"].items():
            top.append([f"V0 gate off {n} vs T39", f'executed {v["executed_actions_equal"]} · planning {v["planning_actions_equal"]} · npz other {v["traj_npz_differ_other"]} · frames other {v["frames_other_leaf_diff"]}',
                        str(v["named_per_chunk"]["grasp_state_equal"] and v["named_per_chunk"]["manipulated_id_equal"] and v["named_per_chunk"]["exclusion_equal"])])
    top.append(["on_e vs on_d identity", f'fully identical {I["n_fully_identical"]}/{I["n_pairs"]} · identical before first gated HOLD (t_gh_d) {I["n_identical_before_t_gh_d"]}/{I["n_pairs"]} · before release (t_rel_d) {I["n_identical_before_t_rel_d"]}/{I["n_pairs"]}',
                f'first diff - t_gh_d: {j(I["earliest_minus_t_gh_d"])}'])
    t0 = a0.table(cellText=top, colLabels=["", "value", "flag / note"], cellLoc="left", colWidths=[0.2, 0.6, 0.2], bbox=[0, 0, 1, 0.88])
    a0.set_title(L("기준선 · V0 · 수정 전/후 동일성", "baseline, V0, before/after identity"), fontsize=10, loc="left")
    t2 = a2.table(cellText=[[a, str(b), str(c), str(d)] for a, b, c, d in rows], colLabels=["", SIDEN["off"], SIDEN["on_d"], SIDEN["on_e"]],
                  cellLoc="left", colWidths=[0.28, 0.24, 0.24, 0.24], bbox=[0, 0, 1, 0.97])
    a2.set_title(L("요약 숫자 (전부 analysis json 에서)", "summary numbers (all from the analysis json)"), fontsize=10, loc="left")
    for tb in (t0, t1, t2):
        tb.auto_set_font_size(False); tb.set_fontsize(7.6)
        for (r, c), cl in tb.get_celld().items():
            cl.set_edgecolor(fs.GRID_INK)
            if r == 0:
                cl.set_facecolor("#efeeea")
    fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
    save(fig, "subtask-e-table.png", {"paired_2x2": T, "identity_summary": I, "top": top, "rows": [list(map(str, r)) for r in rows]})


if __name__ == "__main__":
    which = [a for a in sys.argv if a.startswith("--only=")]
    todo = which[0][7:].split(",") if which else ["scene", "timeline", "table"]
    for f in todo:
        globals()[f]()
