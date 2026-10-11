"""T42-2 invisible-obstacle diagnostic figure + T42-2.verify.json key 'invisible_diag' (verifier).
usage: fig_invis.py   (reads invis_diag.json, analysis_place1.json, raw/invis/probe_<EP>.json)
left  : first-chunk difference max |predicted_chunks[0] - twin| (rad, log) per episode, E3b (E0 identical first chunk), for
        visible obstacle / invisible obstacle (rgba alpha 0) / no obstacle on snapshot work-t42c (ep1834 only); twin = T42-1 run.
middle: policy-image pixel values that differ from the clear model at the first observation (offline render_probe), for the visible
        obstacle, the invisible obstacle and the parked obstacle model (no active slot), summed over the 3 policy cameras.
right : AG3S raw_cloud points within 15 mm of the obstacle volume (max over the run's E3b chunks): no-obstacle twin (virtual
        placement, = other surfaces near that volume), invisible run, visible run.
table : per run first divergent control step (|diff| > 1e-3) vs the twin, outcome, obstacle stop phase / t."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/2')); H = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t42'))
D = json.loads((T / "invis_diag.json").read_text()); A = json.loads((T / "analysis_place1.json").read_text())
figstyle.use_korean()
EPS = [1830, 983, 1215, 1834]
C = figstyle.CATEGORICAL


def vis_run(cond, ep):
    return next(v for k, v in A["runs"].items() if v["cond"] == cond and v["episode"] == ep)


def inv_run(cond, ep):
    return next(v for k, v in D["runs"].items() if v["cond"] == cond and v["episode"] == ep and v["kind"] == "invis")


def chunk0_vis(cond, ep):
    r = vis_run(cond, ep)
    a = np.load(pathlib.Path(r["obst"]["run_dir"]) / "traj.npz"); b = np.load(pathlib.Path(r["divergence"]["reference_run"]) / "traj.npz")
    return float(np.abs(a["predicted_chunks"][0] - b["predicted_chunks"][0]).max())


rows = []; data = {"episodes": {}}
fig = plt.figure(figsize=(16, 7.2)); gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.9])
ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]); axt = fig.add_subplot(gs[1, :])
figstyle.style_axes(fig, [ax1, ax2, ax3])
x = np.arange(len(EPS)); w = 0.26
c0v = [chunk0_vis("E3b", e) for e in EPS]; c0i = [inv_run("E3b", e)["vs_T42_1_no_obstacle"]["first_chunk_diff_max_abs"] for e in EPS]
c0c = [next((v["vs_T42_1_no_obstacle"]["first_chunk_diff_max_abs"] for v in D["runs"].values() if v["kind"] == "clear42c" and v["cond"] == "E3b" and v["episode"] == e), None) for e in EPS]
ax1.bar(x - w, c0v, w, color=C[1], label="visible obstacle"); ax1.bar(x, c0i, w, color=C[0], label="invisible (alpha 0)")
for i, v in enumerate(c0c):
    if v is not None:
        ax1.bar(x[i] + w, max(v, 1e-6), w, color=C[2], label="no obstacle, work-t42c")
        ax1.text(x[i] + w, 1.3e-6, "0 (bit-equal)", ha="center", fontsize=7, rotation=90, color=figstyle.INK)
ax1.set_yscale("log"); ax1.set_ylim(1e-6, 1); ax1.set_xticks(x); ax1.set_xticklabels([f"ep{e}" for e in EPS])
ax1.set_ylabel("max |first chunk - twin| (rad)"); ax1.set_title("policy first chunk vs T42-1 twin (E3b = E0)", color=figstyle.INK, fontsize=10)
ax1.legend(fontsize=7, frameon=False)
pv, pi, pp = [], [], []
for e in EPS:
    pr = D["offline_probe"][str(e)]; P = pr["profiles"]; ks = list(P)
    pv.append(sum(v["n_values_diff"] for v in P[ks[1]]["policy_images_vs_clear"].values()))
    pi.append(sum(v["n_values_diff"] for v in P[ks[2]]["policy_images_vs_clear"].values()))
    pp.append(sum(pr["parked_profile"]["policy_images_vs_clear"].values()))
ax2.bar(x - w, pv, w, color=C[1], label="visible"); ax2.bar(x, pi, w, color=C[0], label="invisible"); ax2.bar(x + w, pp, w, color=C[3], label="parked model, no slot")
ax2.set_yscale("log"); ax2.set_xticks(x); ax2.set_xticklabels([f"ep{e}" for e in EPS]); ax2.set_ylabel("uint8 values differing from clear model")
ax2.set_title("policy images at t0 vs clear model (offline)", color=figstyle.INK, fontsize=10); ax2.legend(fontsize=7, frameon=False)
tw = [inv_run("E3b", e)["ag3s_raw_cloud_on_obstacle_no_obstacle_twin_virtual"]["max_over_chunks"] for e in EPS]
ci = [inv_run("E3b", e)["ag3s_raw_cloud_on_obstacle_invisible"]["max_over_chunks"] for e in EPS]
cv = [inv_run("E3b", e)["ag3s_raw_cloud_on_obstacle_visible_place_run"]["max_over_chunks"] for e in EPS]
ax3.bar(x - w, tw, w, color=C[2], label="no obstacle (virtual)"); ax3.bar(x, ci, w, color=C[0], label="invisible"); ax3.bar(x + w, cv, w, color=C[1], label="visible")
ax3.set_xticks(x); ax3.set_xticklabels([f"ep{e}" for e in EPS]); ax3.set_ylabel("cloud points within 15 mm (max over chunks)")
ax3.set_title("AG3S raw cloud on the obstacle volume (E3b)", color=figstyle.INK, fontsize=10); ax3.legend(fontsize=7, frameon=False)
cols = ["ep", "cond", "obstacle", "first div step vs twin", "success", "grasp", "place", "stop phase / t", "lift t", "contact"]
for e in EPS:
    for cond in ("E0", "E3b"):
        v = vis_run(cond, e); i = inv_run(cond, e)
        for lab, r, div, ok, gr, pl, sp, st, lt, cp in (
                ("visible", v, v["divergence"]["first_step_diff_gt_1e-3"], v["success"], v["grasp"], v["place"], v["obst"]["stop_phase"], v["obst"]["stop_t"], v["obst"]["lift_t"],
                 (v["obst"].get("client_obstacle_safety") or {}).get("contact_pairs")),
                ("invisible", i, i["vs_T42_1_no_obstacle"]["first_step_diff_gt_1e-3"], i["success"], i["grasp"], i["place"], i.get("stop_phase"), i.get("stop_t"), i.get("lift_t"),
                 (i.get("client_contact") or {}).get("contact_pairs"))):
            row = [e, cond, lab, div, "Y" if ok else "N", "Y" if gr else "N", "Y" if pl else "N", f"{sp} / {st}" if sp else "-", lt, "; ".join("-".join(p) for p in (cp or [])) or "none"]
            rows.append([str(z) for z in row])
    for cond in ("E0", "E3b"):
        cl = next((v for v in D["runs"].values() if v["kind"] == "clear42c" and v["cond"] == cond and v["episode"] == e), None)
        if cl:
            rows.append([str(z) for z in [e, cond, "none (work-t42c)", cl["vs_T42_1_no_obstacle"]["first_step_diff_gt_1e-3"], "Y" if cl["success"] else "N",
                                          "Y" if cl["grasp"] else "N", "Y" if cl["place"] else "N", "-", "-", "-"]])
axt.set_axis_off(); tb = axt.table(cellText=rows, colLabels=cols, cellLoc="center", bbox=[0, 0, 1, 1]); tb.auto_set_font_size(False); tb.set_fontsize(7)
for (i_, j), c in tb.get_celld().items():
    c.set_edgecolor(figstyle.GRID_INK)
    if i_ == 0:
        c.set_text_props(color=figstyle.INK, weight="bold")
fig.suptitle("T42-2 invisible-obstacle diagnostic (seed rep 1; twin = T42-1 no-obstacle run, snapshot work-t42; 'None' = no divergence)", color=figstyle.INK, fontsize=11)
fig.tight_layout()
p = OUT / "t42-2-invisible-diag.png"; fig.savefig(p, dpi=150, facecolor=figstyle.SURFACE); plt.close(fig)
side = {"definitions": __doc__, "episodes": EPS, "first_chunk_diff_E3b": {"visible": c0v, "invisible": c0i, "clear_work_t42c": c0c},
        "policy_image_values_diff_vs_clear": {"visible": pv, "invisible": pi, "parked": pp},
        "ag3s_cloud_points_within_15mm": {"no_obstacle_virtual": tw, "invisible": ci, "visible": cv}, "table_columns": cols, "table_rows": rows}
(OUT / "t42-2-invisible-diag.json").write_text(json.dumps(side, indent=1, default=float)); print(p)
# verify key
V = json.loads((H / "T42-2.verify.json").read_text())
probe = {}
for e in EPS:
    pr = D["offline_probe"][str(e)]; P = pr["profiles"]; ks = list(P)
    probe[e] = {"model_stat_extent": {"clear": P[ks[0]]["stat_extent"], "obstacle_model": P[ks[1]]["stat_extent"]},
                "model_stat_center": {"clear": P[ks[0]]["stat_center"], "obstacle_model": P[ks[1]]["stat_center"]},
                "qpos_qvel_t0_equal_clear": {"visible": [P[ks[1]]["qpos_equal_clear"], P[ks[1]]["qvel_equal_clear"]], "invisible": [P[ks[2]]["qpos_equal_clear"], P[ks[2]]["qvel_equal_clear"]],
                                             "parked": pr["parked_profile"]["qpos_equal_clear"]},
                "contacts_with_obstacle_at_t0": {"visible": P[ks[1]]["n_contacts_with_obstacle"], "invisible": P[ks[2]]["n_contacts_with_obstacle"]},
                "policy_images_vs_clear": {"visible": P[ks[1]]["policy_images_vs_clear"], "invisible": P[ks[2]]["policy_images_vs_clear"],
                                           "parked": pr["parked_profile"]["policy_images_vs_clear"]},
                "policy_images_invisible_vs_parked": pr["parked_profile"]["policy_images_vs_invisible"],
                "depth_invisible_vs_parked_n_pixels_diff": pr["parked_profile"]["depth_vs_invisible_n_gt_1e-9"],
                "depth_invisible_vs_visible_n_pixels_diff": pr["invisible_vs_visible"]["depth"],
                "depth_vs_clear_max_abs_m": {"visible": {k: v["max_abs_m"] for k, v in P[ks[1]]["depth_vs_clear"].items()},
                                             "invisible": {k: v["max_abs_m"] for k, v in P[ks[2]]["depth_vs_clear"].items()}},
                "collision_geoms_active_invisible": pr["active_check_invisible"]}
runs = {}
for k, r in D["runs"].items():
    runs[k] = {kk: r.get(kk) for kk in ("cond", "kind", "episode", "seed", "success", "grasp", "place", "grasp_t", "place_t", "obstacle_stop", "stop_t", "stop_phase",
                                        "lift_t", "client_contact", "holds", "vs_T42_1_no_obstacle", "vs_visible_place_run", "vs_clear42c",
                                        "ag3s_raw_cloud_on_obstacle_no_obstacle_twin_virtual", "ag3s_raw_cloud_on_obstacle_invisible",
                                        "ag3s_raw_cloud_on_obstacle_visible_place_run")}
V["invisible_diag"] = {"_comment": "numbers and paths only. Profile t42_2_ep<EP>_place_invis = the place-phase placement with visual geom rgba [0,0,0,0] (configs/t42_2_ep<EP>_invis.json, also holds t42_2_ep<EP>_parked = obstacle model, no slot). Seed rep 1. Twin = T42-1 no-obstacle run (snapshot work-t42).",
                       "commands": ["p/render_probe.py (offline, CPU)", "p/launch_t42_2.sh invis (E3b 8251, E0 8250)", "p/invis_diag.py", "p/fig_invis.py"],
                       "raw": ["outputs/verify/T42/2/raw/invis/probe_<EP>.json", "outputs/verify/T42/2/runs/{E0,E3b}/{invis,clear42c}/", "outputs/verify/T42/2/invis_diag.json"],
                       "definitions": D["definitions"], "offline_probe": probe, "runs": runs,
                       "figure": {"path": "AG3S/ag3s/docs/figures/t42/t42-2-invisible-diag.png", "kind": "graph+table", "sidecar": "AG3S/ag3s/docs/figures/t42/t42-2-invisible-diag.json"}}
V["figures"] = [f for f in V["figures"] if "invisible-diag" not in f["path"]] + [V["invisible_diag"]["figure"]]
(H / "T42-2.verify.json").write_text(json.dumps(V, indent=1, default=float)); print("verify key written")
