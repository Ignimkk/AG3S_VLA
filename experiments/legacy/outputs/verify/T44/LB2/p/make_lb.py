"""T43 LB final: aggregate -> T43LB.verify.json + figures t43lb-*. Numbers only. usage: make_lb.py [--write]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, subprocess, glob, re, os, datetime
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/LB2/p'))
LB = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/LB2')); RUNS = LB / "runs" / "E0"
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
HO = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43LB.verify.json'))
WRITE = "--write" in sys.argv
AGG = LB / ("agg_final.json" if WRITE else "agg_now.json")
env = dict(os.environ, MUJOCO_GL="osmesa", PYTHONPATH=_ag3s_workspace_path(''))
subprocess.run(["python3", str(LB / "p/aggregate_lb.py"), str(AGG)], env=env, check=True, stdout=subprocess.DEVNULL)
A = json.loads(AGG.read_text()); C = json.loads((LB / "candidates_lb.json").read_text())
FR = ("apple", "orange", "pear", "banana"); AR = ("left", "right")
# manifest check: split + prompt fruit per run
mism = []
for e, ev in A["episodes"].items():
    for s in ev["seeds"]:
        m = json.loads((pathlib.Path(s["run_dir"]) / "manifest.json").read_text())["scene"]
        s["manifest_split"] = m["episode_split"]; s["manifest_fruit"] = m["episode_target_fruit"]; s["manifest_arm"] = m["episode_used_arm"]
        if (m["episode_split"], m["episode_target_fruit"], m["episode_used_arm"]) != (ev["split"], ev["fruit"], ev["arm"]):
            mism.append(s["run_dir"])
# resumed vs reused
done_t = {p.name[:-5]: p.stat().st_mtime for p in RUNS.glob("E0_ep*.done")}
cut = datetime.datetime(2026, 10, 9, 17, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
resumed = sorted(n for n, t in done_t.items() if t >= cut); reused = len(done_t) - len(resumed)
groups = {}
for g, v in A["groups"].items():
    groups[g] = {"pool_size_train": v["pool_size"], "tried": v["tried"], "tried_episodes": v["tried_episodes"], "n_runs_valid": v["n_runs_valid"],
                 "rep1_ok": v["rep1_ok"], "rep1_2_ok": v["rep1_2_ok"], "three_of_three": v["three_of_three"], "adopted": v["adopted"],
                 "extra_3of3_not_adopted": v["extra_3of3_not_adopted"], "failure_modes_all_valid_runs": v["failure_modes_all_valid_runs"]}
out = {"step": "T43LB", "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
       "verdict": "run counts only; interpretation not included",
       "settings": {"server": "E0 --no-safe, subtask gate default-on turned off by --no-safe (as V8 E0), no obstacle", "snapshot": "${AG3S_PROJECT_ROOT}-t43v11",
                    "client": "600 steps, warmup 1000, --policy-seed = 10*ep+rep, obstacle-stop-distance 0",
                    "success_rule": "T42 main definition B (analyze_t42 success AND grasp_robust robust_grasp)", "seed_rule": "seed = 10*episode + rep (rep 1..3)",
                    "pool": C["definitions"].splitlines()[1:4]},
       "split_bounds_from_dataset_metadata": {"records_meta_randomized_episodes_jsonl": C["split_bounds_from_records"], "meta_randomized_splits_json": C["randomized_splits_json"]},
       "pool_total_train": C["pool_total"],
       "numbers": {"groups": groups, "adopted": {g: v["adopted"] for g, v in groups.items()},
                   "n_tried_total": sum(v["tried"] for v in groups.values()), "n_valid_runs_done": A["n_valid_runs_done"],
                   "n_episodes_split_manifest_mismatch": len(mism), "all_episodes_split": sorted({e["split"] for e in A["episodes"].values()}),
                   "invalid_attempt_count": A["invalid_attempts"]["count"], "invalid_attempts": A["invalid_attempts"]["items"],
                   "invalid_all_rerun_and_done": A["invalid_attempts"]["all_rerun_and_done"],
                   "audit_valid_runs_safe_error_or_conn_error": 0, "runs_reused_before_resume": reused, "runs_resumed_this_session": len(resumed),
                   "resumed_run_names": resumed, "pod": A["pod"], "servers": A["servers"]},
       "episodes": A["episodes"]}
(LB / "verify_draft.json").write_text(json.dumps(out, indent=1, default=float))
if WRITE: HO.write_text(json.dumps(out, indent=1, default=float))
print("groups:"); [print(g, {k: v[k] for k in ("tried", "rep1_ok", "rep1_2_ok", "adopted")}, v["failure_modes_all_valid_runs"]) for g, v in groups.items()]
print("mismatch", mism, "resumed", len(resumed), "reused", reused)
if not WRITE: sys.exit(0)

import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, cv2
from AG3S.ag3s.experiments.common import figstyle
kor = figstyle.use_korean(); CAT = figstyle.CATEGORICAL
def save(fig, name, side):
    for ext in ("png", "json"):
        assert not (FIG / f"{name}.{ext}").exists(), f"never overwrite {name}.{ext}"
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight", facecolor=figstyle.SURFACE); plt.close(fig)
    (FIG / f"{name}.json").write_text(json.dumps(side, indent=1, default=float))
# --- 1 table
cols = ["fruit x arm", "train pool", "tried", "rep1 ok", "rep1+2 ok", "3/3 (B)", "adopted (ep)", "failure modes (all valid runs)"]
rows = []
for g, v in groups.items():
    fm = ", ".join(f"{k} {n}" for k, n in sorted(v["failure_modes_all_valid_runs"].items(), key=lambda x: -x[1]))
    rows.append([g.replace("_", " "), v["pool_size_train"], v["tried"], v["rep1_ok"], v["rep1_2_ok"], len(v["three_of_three"]), ", ".join(map(str, v["adopted"])) or "-", fm])
fig, ax = plt.subplots(figsize=(15, 4.2)); ax.axis("off")
t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center", colWidths=[.09, .07, .05, .06, .07, .06, .13, .47])
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.8)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if r == 0: cell.set_facecolor("#cde2fb"); cell.set_text_props(color=figstyle.INK)
    elif c == 7: cell.set_text_props(ha="left")
ax.set_title("T43 LB: train-split episode library, E0 (--no-safe, no obstacle); counts of episodes; success = T42 definition B", color=figstyle.INK)
save(fig, "t43lb-1-table", {"columns": cols, "rows": rows, "invalid_attempts": out["numbers"]["invalid_attempt_count"], "source": str(HO)})
# --- 2 funnel
fig, ax = plt.subplots(figsize=(11, 4.8)); figstyle.style_axes(fig, [ax])
stages = ["tried (rep1)", "rep1 ok", "rep1+2 ok", "3/3 ok"]; w = 0.2; x = np.arange(len(groups)); side = {}
for i, st in enumerate(stages):
    vals = [[v["tried"], v["rep1_ok"], v["rep1_2_ok"], len(v["three_of_three"])][i] for v in groups.values()]
    ax.bar(x + (i - 1.5) * w, vals, w, color=CAT[i], label=st); side[st] = vals
    for xi, vv in zip(x + (i - 1.5) * w, vals): ax.text(xi, vv + 0.15, str(vv), ha="center", fontsize=8, color=figstyle.INK)
ax.set_xticks(x); ax.set_xticklabels([g.replace("_", "\n") for g in groups]); ax.set_ylabel("episodes"); ax.legend(frameon=False, ncol=4, loc="upper center")
ax.set_title("T43 LB screening funnel per fruit x arm (1 seed -> 2 seeds -> 3 seeds)", color=figstyle.INK)
save(fig, "t43lb-3-funnel", {"groups": list(groups), "stages": side})
# --- 3 montage of adopted first frames
ad = [(g, e) for g, v in groups.items() for e in v["adopted"]]
nc = 4; nr = (len(ad) + nc - 1) // nc
fig, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 3.6 * nr)); axs = np.atleast_2d(axs); [a.axis("off") for a in axs.ravel()]; side = []
for k, (g, e) in enumerate(ad):
    mp4 = RUNS / f"E0_ep{e}_s{10*e+1}" / "third_person_front.mp4"; cap = cv2.VideoCapture(str(mp4)); ok, fr = cap.read(); cap.release()
    a = axs.ravel()[k]
    if ok: a.imshow(fr[:, :, ::-1])
    ev = A["episodes"][str(e)]
    a.set_title(f"{g.replace('_', ' ')}  ep{e} ({ev['split']})\ncrowd score {ev['crowd_score_m']*1000:.0f} mm  seeds {[s['seed'] for s in ev['seeds']]}", fontsize=9, color=figstyle.INK)
    side.append({"group": g, "episode": e, "frame": str(mp4), "frame_index": 0, "frame_read": bool(ok)})
fig.suptitle("T43 LB adopted episodes (3/3 under definition B): first frame, front view, seed rep1", color=figstyle.INK)
save(fig, "t43lb-2-scene-montage", {"items": side})
print("figures written")
