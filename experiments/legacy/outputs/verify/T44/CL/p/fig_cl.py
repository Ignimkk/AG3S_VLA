"""T44d CL figures (verifier). usage: fig_cl.py  -> AG3S/ag3s/docs/figures/t44/t44d-{scene,graph,table}.png + .json, and 2 videos (C0 | C1b side by side)
Reads CL/agg.json. English labels only (technical terms)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, subprocess
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from AG3S.ag3s.experiments.common import figstyle

CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t44'))
ARMS = ["C0", "C1a", "C1b", "C2"]
COL = {a: figstyle.CATEGORICAL[i] for i, a in enumerate(ARMS)}
D = json.load(open(CL / "agg.json"))
figstyle.use_korean()
for n in ("t44d-scene", "t44d-graph", "t44d-table"):
    pass


def run(a, e, s):
    return D["runs"][f"{a}_ep{e}_s{s}"]


def frame_at(mp4, idx):
    import cv2
    cap = cv2.VideoCapture(str(mp4)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, fr = cap.read(); cap.release()
    return fr[:, :, ::-1] if ok else None


# ---------------------------------------------------------------- graph
def graph():
    side = {}
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5)); figstyle.style_axes(fig, ax)
    A = D["arms"]; x = np.arange(len(ARMS))
    # (a) outcomes
    a0 = ax[0, 0]; mets = [("task_success", "t1_success", "task success"), ("success_ct", "t1_success", "CT success"), ("robust_grasp", "t2_grasp", "robust grasp"), ("first_grasp_ok", "t2_grasp", "first grasp ok"), ("grasp_final", "t2_grasp", "grasp (final)")]
    w = 0.8 / len(ARMS)
    for i, a in enumerate(ARMS):
        vals = [A[a][t][k] for k, t, _ in mets]
        a0.bar(np.arange(len(mets)) + (i - 1.5) * w, vals, w, color=COL[a], label=a)
        for j, v in enumerate(vals):
            a0.text(j + (i - 1.5) * w, v + 0.2, str(v), ha="center", fontsize=7, color=figstyle.INK_2)
        side.setdefault("outcomes", {})[a] = dict(zip([m[2] for m in mets], vals))
    a0.set_xticks(range(len(mets))); a0.set_xticklabels([m[2] for m in mets], fontsize=8); a0.set_ylim(0, 20); a0.set_ylabel("runs out of 18"); a0.set_title("(a) outcomes per arm (18 runs)", fontsize=10, color=figstyle.INK); a0.legend(fontsize=8, frameon=False, ncol=4)
    # (b) contacts
    a1 = ax[0, 1]; cls = ["hand-fruit_other", "robot_other-target", "robot_other-fruit_other"]
    for i, a in enumerate(ARMS):
        vals = [A[a]["t3_contacts"][c]["runs_any"] for c in cls]
        a1.bar(np.arange(len(cls)) + (i - 1.5) * w, vals, w, color=COL[a], label=a)
        for j, v in enumerate(vals):
            a1.text(j + (i - 1.5) * w, v + 0.05, str(v), ha="center", fontsize=7, color=figstyle.INK_2)
        side.setdefault("contact_runs", {})[a] = dict(zip(cls, vals))
        side.setdefault("contact_rows", {})[a] = {c: A[a]["t3_contacts"][c]["contact_rows"] for c in cls}
    a1.set_xticks(range(len(cls))); a1.set_xticklabels(cls, fontsize=8); a1.set_ylabel("runs with contact (of 18)"); a1.set_title("(b) new-class contact runs (replay, any row)", fontsize=10, color=figstyle.INK)
    # (c) HOLD
    a2 = ax[0, 2]
    for i, a in enumerate(ARMS):
        v = [D["runs"][k]["n_hold_chunks"] for k in D["runs"] if D["runs"][k]["arm"] == a]
        jit = np.linspace(-0.18, 0.18, len(v))
        a2.scatter(np.full(len(v), i) + jit, v, color=COL[a], s=26, zorder=3)
        a2.text(i, max(v) + 2.5, f"sum {sum(v)}\n{sum(1 for z in v if z)} runs", ha="center", fontsize=8, color=figstyle.INK_2)
        side.setdefault("hold_chunks_per_run", {})[a] = v
    a2.set_xticks(range(len(ARMS))); a2.set_xticklabels(ARMS); a2.set_ylabel("HOLD chunks per run"); a2.set_ylim(-2, 80); a2.set_title("(c) HOLD chunks per run (75 chunks/run)", fontsize=10, color=figstyle.INK)
    # (d) timing
    a3 = ax[1, 0]
    data = {}
    for key, lab in (("to_ms_server_trajopt_pooled", "TO ms"), ("ag3s_ms_server_ag3s_pooled", "AG3S ms")):
        data[lab] = {a: A[a]["t5_timing"][key] for a in ARMS}
    for j, lab in enumerate(data):
        for i, a in enumerate(ARMS):
            st = data[lab][a]
            xx = j * 5 + i
            a3.bar(xx, st["median"], 0.8, color=COL[a], label=a if j == 0 else None)
            a3.plot([xx, xx], [st["median"], st["p90"]], color=figstyle.INK, lw=1.2)
            a3.plot([xx - 0.2, xx + 0.2], [st["p90"]] * 2, color=figstyle.INK, lw=1.2)
            a3.text(xx, st["p90"] + 12, f"{st['median']:.0f}\n{st['p90']:.0f}", ha="center", fontsize=7, color=figstyle.INK_2)
        side.setdefault("timing_ms_median_p90", {})[lab] = {a: [data[lab][a]["median"], data[lab][a]["p90"]] for a in ARMS}
    a3.set_xticks([1.5, 6.5]); a3.set_xticklabels(["TO (server trajopt)", "AG3S (server ag3s)"]); a3.set_ylabel("ms (bar = median, whisker to p90)"); a3.set_title("(d) per-chunk time, pooled over runs (first chunk excluded)", fontsize=10, color=figstyle.INK); a3.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.88))
    # (e) pair matrix
    a4 = ax[1, 1]; keys = [(e, s) for e in D["episodes"] for s in D["episodes_info"][str(e)]["seeds"]]
    M = np.zeros((len(keys), len(ARMS)))
    for r, (e, s) in enumerate(keys):
        for c, a in enumerate(ARMS):
            o = run(a, e, s); M[r, c] = 2 if o["success_ct"] else (1 if o["success"] else 0)
    cm = matplotlib.colors.ListedColormap([figstyle.GRID_INK, figstyle.SEQ_BLUE[5], figstyle.SEQ_BLUE[11]])
    a4.imshow(M, cmap=cm, vmin=0, vmax=2, aspect="auto")
    a4.set_xticks(range(len(ARMS))); a4.set_xticklabels(ARMS); a4.set_yticks(range(len(keys))); a4.set_yticklabels([f"ep{e} s{s} {D['episodes_info'][str(e)]['target_fruit'][:3]}" for e, s in keys], fontsize=7)
    a4.set_title("(e) per (episode, seed): dark = CT success, mid = success with pre-release contact, light = fail", fontsize=9, color=figstyle.INK)
    side["pair_matrix_0fail_1success_noCT_2successCT"] = {f"ep{e}_s{s}": [int(v) for v in M[r]] for r, (e, s) in enumerate(keys)}
    # (f) trace
    a5 = ax[1, 2]
    ks = [("n_removed_sum", "removed (ball)"), ("n_kept_nontarget_sum", "kept non-target"), ("n_ambiguous_sum", "ambiguous (kept)"), ("n_unowned_removed_sum", "unowned removed"), ("coarse_n_shared_kept_sum", "coarse shared kept")]
    w2 = 0.8 / len(ARMS)
    for i, a in enumerate(ARMS):
        vals = [A[a]["t6_trace"][k] for k, _ in ks]
        a5.bar(np.arange(len(ks)) + (i - 1.5) * w2, [max(v, 0.8) for v in vals], w2, color=COL[a], label=a)
        side.setdefault("trace_sums", {})[a] = dict(zip([k for k, _ in ks], vals))
    a5.set_yscale("log"); a5.set_xticks(range(len(ks))); a5.set_xticklabels([l for _, l in ks], fontsize=7, rotation=12); a5.set_ylabel("seed voxels, summed over chunks (0 drawn at 0.8)"); a5.set_title("(f) server trace: ball seeds by treatment", fontsize=10, color=figstyle.INK)
    fig.suptitle("T44d closed-loop, 18 (episode, seed) runs per arm, train-split crowded episodes (nn < 0.10 m)", fontsize=12, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "t44d-graph.png", dpi=130); plt.close(fig)
    (FIG / "t44d-graph.json").write_text(json.dumps({"source": str(CL / "agg.json"), "numbers": side}, indent=1))


# ---------------------------------------------------------------- table
def table():
    A = D["arms"]
    rows = [("task success (n/18)", lambda a: A[a]["t1_success"]["task_success"]), ("CT success", lambda a: A[a]["t1_success"]["success_ct"]),
            ("success_rel (Z4 release)", lambda a: A[a]["t1_success"]["success_rel"]), ("grasp final", lambda a: A[a]["t2_grasp"]["grasp_final"]),
            ("first grasp ok (no regrasp)", lambda a: A[a]["t2_grasp"]["first_grasp_ok"]), ("regrasp runs", lambda a: A[a]["t2_grasp"]["regrasp_runs"]),
            ("slip runs (rule B)", lambda a: A[a]["t2_grasp"]["slip_runs"]), ("robust grasp", lambda a: A[a]["t2_grasp"]["robust_grasp"]),
            ("hand-fruit_other runs / rows", lambda a: f"{A[a]['t3_contacts']['hand-fruit_other']['runs_any']} / {A[a]['t3_contacts']['hand-fruit_other']['contact_rows']}"),
            ("robot_other-target runs / rows", lambda a: f"{A[a]['t3_contacts']['robot_other-target']['runs_any']} / {A[a]['t3_contacts']['robot_other-target']['contact_rows']}"),
            ("robot_other-fruit_other runs / rows", lambda a: f"{A[a]['t3_contacts']['robot_other-fruit_other']['runs_any']} / {A[a]['t3_contacts']['robot_other-fruit_other']['contact_rows']}"),
            ("hand-target runs / rows", lambda a: f"{A[a]['t3_contacts']['hand-target']['runs_any']} / {A[a]['t3_contacts']['hand-target']['contact_rows']}"),
            ("HOLD chunks / runs", lambda a: f"{A[a]['t4_hold']['hold_chunks_total']} / {A[a]['t4_hold']['hold_runs']}"),
            ("comms timeout/stale/error chunks", lambda a: A[a]["t4_hold"]["comms_fail_chunks"]),
            ("TO ms median / p90", lambda a: f"{A[a]['t5_timing']['to_ms_server_trajopt_pooled']['median']:.0f} / {A[a]['t5_timing']['to_ms_server_trajopt_pooled']['p90']:.0f}"),
            ("AG3S ms median / p90", lambda a: f"{A[a]['t5_timing']['ag3s_ms_server_ag3s_pooled']['median']:.0f} / {A[a]['t5_timing']['ag3s_ms_server_ag3s_pooled']['p90']:.0f}"),
            ("round trip ms median / p90", lambda a: f"{A[a]['t5_timing']['client_round_trip_ms_pooled']['median']:.0f} / {A[a]['t5_timing']['client_round_trip_ms_pooled']['p90']:.0f}"),
            ("split fired chunks / components", lambda a: f"{A[a]['t6_trace']['split_fired_chunks']} / {A[a]['t6_trace']['split_components']}"),
            ("n_kept_nontarget / unowned_removed", lambda a: f"{A[a]['t6_trace']['n_kept_nontarget_sum']} / {A[a]['t6_trace']['n_unowned_removed_sum']}"),
            ("coarse n_shared_kept (chunks)", lambda a: f"{A[a]['t6_trace']['coarse_n_shared_kept_sum']} ({A[a]['t6_trace']['coarse_chunks']})"),
            ("target_rows_unauthorized_kept", lambda a: "absent" if not A[a]["t6_trace"]["path_target_rows_unauthorized_kept_present_in_runs"] else A[a]["t6_trace"]["path_target_rows_unauthorized_kept_sum"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
    for ax in (ax1, ax2):
        ax.axis("off")
    fig.patch.set_facecolor(figstyle.SURFACE)
    cells = [[lab] + [str(f(a)) for a in ARMS] for lab, f in rows]
    t = ax1.table(cellText=cells, colLabels=["metric (18 runs / arm)"] + ARMS, loc="center", cellLoc="center", colLoc="center", colWidths=[0.4, 0.15, 0.15, 0.15, 0.15])
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.45)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        if r == 0:
            cell.set_facecolor(figstyle.SEQ_BLUE[3] if c else figstyle.SEQ_BLUE[1]); cell.set_text_props(color=figstyle.INK)
        else:
            cell.set_facecolor(figstyle.SURFACE)
        if c == 0:
            cell.set_text_props(ha="left"); cell._loc = "left"
    ax1.set_title("per-arm summary (tables 1-6)", fontsize=10, color=figstyle.INK)
    prow = []
    P = D["pairs"]
    for pk in ("C0_vs_C1a", "C0_vs_C1b", "C0_vs_C2", "C1a_vs_C1b", "C1b_vs_C2"):
        for f, lab in (("success_ct", "CT success"), ("success", "task success"), ("robust_grasp", "robust grasp")):
            x = P[pk][f]
            prow.append([pk.replace("_vs_", " vs "), lab, f"{x['A_total']} / {x['B_total']}", f"{x['A_only']} / {x['B_only']}", f"{x['mcnemar_exact_p']:.4f}"])
    t2 = ax2.table(cellText=prow, colLabels=["pair (A vs B)", "metric", "A / B total", "A-only / B-only", "McNemar exact p"], loc="center", cellLoc="center")
    t2.auto_set_font_size(False); t2.set_fontsize(8.5); t2.scale(1, 1.45)
    for (r, c), cell in t2.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK); cell.set_facecolor(figstyle.SEQ_BLUE[3] if r == 0 else figstyle.SURFACE)
    ax2.set_title("paired comparisons on (episode, seed), n = 18 pairs", fontsize=10, color=figstyle.INK)
    fig.suptitle("T44d closed-loop: 4 arms x 18 runs (9 train episodes x 2 seeds). Small n: p values are descriptive.", fontsize=11, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "t44d-table.png", dpi=130); plt.close(fig)
    (FIG / "t44d-table.json").write_text(json.dumps({"source": str(CL / "agg.json"), "rows": {lab: {a: str(f(a)) for a in ARMS} for lab, f in rows}, "pairs": {pk: {f: {k: v for k, v in P[pk][f].items()} for f in ("success_ct", "success", "robust_grasp")} for pk in P}}, indent=1, default=float))


# ---------------------------------------------------------------- scene
def scene():
    blocks = [(0, 1, "C0", "C1b"), (12, 122, "C0", "C1b"), (30, 302, "C0", "C1b")]
    meta = {}
    fig, axs = plt.subplots(6, 3, figsize=(14, 24)); figstyle.style_axes(fig, axs)
    for b, (e, s, A, B) in enumerate(blocks):
        rj = {a: json.load(open(run(a, e, s)["replay_json"])) for a in (A, B)}
        tz = {a: np.load(run(a, e, s)["replay_json"].replace(".json", ".npz"))["t"] for a in (A, B)}
        # times (control step): first hand-target contact of the later arm; C0 release or 300; end
        ht = {a: (min(ev["t_first"] for ev in rj[a]["contact_events"]["hand-target"]) if rj[a]["contact_events"]["hand-target"] else None) for a in (A, B)}
        if e == 30:
            fo = [ev["t_first"] for ev in rj[B]["contact_events"]["hand-fruit_other"]]
            tt = [fo[0] - 20, fo[0], fo[0] + 30]; lab = ["20 steps before first hand-fruit_other contact (C1b)", "first hand-fruit_other contact (C1b)", "30 steps after"]
        else:
            t0 = min(v for v in ht.values() if v is not None)
            relA = run(A, e, s)["release_t"]
            tt = [t0, int(relA) if relA else 300, 600]; lab = ["first hand-target contact", "C0 release step" if relA else "t = 300", "end of run"]
        meta[f"ep{e}_s{s}"] = {"times": tt, "labels": lab}
        for r, a in enumerate((A, B)):
            o = run(a, e, s)
            for c, t in enumerate(tt):
                ax = axs[2 * b + r, c]; ax.axis("off")
                idx = int(np.searchsorted(tz[a], t)); idx = min(idx, 599)
                fr = frame_at(pathlib.Path(o["run_dir"]) / "third_person_front.mp4", idx)
                if fr is not None:
                    ax.imshow(fr)
                ax.set_title(f"{a}  ep{e} s{s} ({D['episodes_info'][str(e)]['target_fruit']}, {D['episodes_info'][str(e)]['used_arm']})  t={t}\n{lab[c]}", fontsize=8, color=figstyle.INK)
                if c == 0:
                    ax.text(0.01, 0.02, f"success={o['success']}  CT={o['success_ct']}  {o['failure_mode']}\nHOLD chunks={o['n_hold_chunks']}  contacts rows (hand-fruit_other)={o['contacts_rows']['hand-fruit_other']}", transform=ax.transAxes, fontsize=7.5, color="white",
                            bbox=dict(facecolor="black", alpha=0.55, pad=2))
                meta[f"ep{e}_s{s}"].setdefault(a, {})[f"frame_idx_t{t}"] = idx
    fig.suptitle("T44d scene: front camera frames of runs where C0 and C1b differ (video frame = control step; row pairs = C0 / C1b)", fontsize=11, color=figstyle.INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(FIG / "t44d-scene.png", dpi=100); plt.close(fig)
    (FIG / "t44d-scene.json").write_text(json.dumps({"source": str(CL / "agg.json"), "blocks": meta, "run_dirs": {f"ep{e}_s{s}": {a: run(a, e, s)["run_dir"] for a in (A, B)} for e, s, A, B in blocks}}, indent=1))


def videos():
    import cv2
    out = []
    for e, s in ((0, 1), (12, 122)):
        a, b = run("C0", e, s), run("C1b", e, s)
        ca = cv2.VideoCapture(str(pathlib.Path(a["run_dir"]) / "third_person_front.mp4")); cb = cv2.VideoCapture(str(pathlib.Path(b["run_dir"]) / "third_person_front.mp4"))
        tmp = CL / f"tmp_ep{e}_s{s}.mp4"; w = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 480)); i = 0
        while True:
            ok1, f1 = ca.read(); ok2, f2 = cb.read()
            if not (ok1 and ok2):
                break
            for f, lab in ((f1, f"C0  success={a['success']}  HOLD={a['n_hold_chunks']}"), (f2, f"C1b  success={b['success']} ({b['failure_mode']})  HOLD={b['n_hold_chunks']}")):
                cv2.rectangle(f, (0, 0), (640, 26), (0, 0, 0), -1); cv2.putText(f, f"ep{e} s{s}  {lab}  step {i}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            w.write(np.hstack([f1, f2])); i += 1
        w.release()
        dst = FIG / f"t44d-ep{e}-s{s}-C0-vs-C1b.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", str(dst)], check=True)
        tmp.unlink(); out.append(str(dst))
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["graph", "table", "scene", "videos"]
    r = {}
    for w in which:
        r[w] = globals()[w]()
    print(r)
