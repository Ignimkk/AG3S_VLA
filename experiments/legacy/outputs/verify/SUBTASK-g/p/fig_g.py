"""SUBTASK-g figure (rule A, table): subtask-g-table.png + .json sidecar. Inputs: baseline/judge.json, baseline/leafdiff_vs_sf.json,
flags/flag_matrix.json, analysis/summary.json, SUBTASK-f baseline/judge.json (sha1 reference). usage: fig_g.py OUT_DIR"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-sg")
from AG3S.ag3s.experiments.common import figstyle as fs  # noqa: E402
G = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-g')); OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
KO = fs.use_korean()
L = lambda ko, en: ko if KO else en
BL = json.load(open(G / "baseline/judge.json")); BLF = json.load(open(_ag3s_workspace_path('outputs/verify/SUBTASK-f/baseline/judge.json')))
LD = json.load(open(G / "baseline/leafdiff_vs_sf.json")); FM = json.load(open(G / "flags/flag_matrix.json")); S = json.load(open(G / "analysis/summary.json"))
EXP = {"legacy": "14/15 · 15/15 · -29.031048280806342", "curobo": "10/15 · 15/15 · -9.171877401271193"}

t0 = []
for b in ("legacy", "curobo"):
    x = BL[b]
    t0.append([f"baseline {b}", f'{x["violated_start"]} · {x["has_target"]} · {x["frame0_clearance_before_mm"]!r}', EXP[b], str(x["matched"]),
               str(x["sha1"]["repr_json_list"] == BLF[b]["sha1"]["repr_json_list"]),
               ", ".join(f'{k} {v["n_frames"]}' for k, v in LD[b]["frame_leaf_diff"].items()) or "-"])

t1 = []
for name in ("no_flag", "--no-subtask-gate", "--subtask-gate", "--no-attention", "--no-safe", "--no-perception",
             "--subtask-gate --no-attention", "--no-subtask-gate --no-attention", "--subtask-gate --no-subtask-gate"):
    c = FM["cases"][name]
    if c["rejected"]:
        t1.append([name, "rejected (argparse error)", "-", "-", "-"])
    else:
        note = c["startup_note"]
        t1.append([name, "started", str(c["resolved_subtask_gate"]), json.dumps(c["config_section"]["clustering"]),
                   (note[:62] + "...") if note else "-"])

st = S["server_startup"]
def gl(c):
    lines = [l for v in st[c].values() for l in v]
    return " / ".join(l.replace("INFO:root:", "")[:70] for l in lines)
t2 = []
for c, lab in (("E3bD", "E3bD (no gate flag)"), ("E3bN", "E3bN (E3b + --no-subtask-gate)")):
    pk = [p for p in S["pairs"] if p.startswith("D_" if c == "E3bD" else "N_vs_SF")]
    vals = sorted({v for p in pk for v in S["pairs"][p]["new"]["subtask_gate_values"]})
    n = sum(S["pairs"][p]["new"]["n_chunks"] for p in pk)
    t2.append([lab, gl(c), f'{",".join(vals)} ({n} chunks)'])

REF = {"D_vs_SFon": "SUBTASK-f gate on (E3bG)", "N_vs_T39": "T39 E3b (gate off)", "N_vs_SFoff": "SUBTASK-f V0 E3b (gate off)"}
t3 = []
for p, v in S["pairs"].items():
    k = p.rsplit("_ep", 1)[0]; run = "ep" + p.rsplit("_ep", 1)[1]
    fd = {f: x["n_chunks_differ"] for f, x in v["decision_field_diff"].items() if x["n_chunks_differ"]}
    nonsub = {f: n for f, n in fd.items() if not f.startswith("subtask.")}
    sgp = lambda s: "/".join("Y" if s[q] else "N" for q in ("success", "grasp", "place"))
    other = v["manifest_other_leaf_diff"] + v["server_array_diff_other"] + v["frames_other_leaf_diff"] + v["server_summary_other_leaf_diff"]
    t3.append([("E3bD" if k.startswith("D") else "E3bN") + " " + run, REF[k], str(v["executed_actions_equal"]), str(v["planning_actions_equal"]),
               f'{len(nonsub)} / {len(fd)}', f'{sgp(v["ref"])} | {sgp(v["new"])}',
               f'{v["ref"]["n_hold_chunks"]} | {v["new"]["n_hold_chunks"]}',
               f'{v["ref"]["grounding_status_counts"].get("subtask_gated", 0)} | {v["new"]["grounding_status_counts"].get("subtask_gated", 0)}',
               f'{",".join(v["ref"]["subtask_gate_values"])} | {",".join(v["new"]["subtask_gate_values"])}',
               ", ".join(sorted(set(other))) or "-"])

fig, axs = plt.subplots(4, 1, figsize=(17, 13.5), gridspec_kw=dict(height_ratios=[0.33, 0.95, 0.3, 0.95]))
for a in axs:
    a.axis("off")
tabs = []
tabs.append(axs[0].table(cellText=t0, colLabels=["", "violated start · has_target · frame0 clearance_before mm", "expected", "matched",
                                                  "sha1 == SUBTASK-f", "leaf diff vs SUBTASK-f (frames)"],
                         cellLoc="left", colWidths=[0.1, 0.28, 0.2, 0.07, 0.1, 0.25], bbox=[0, 0, 1, 0.85]))
axs[0].set_title(L("(1) 회귀 기준선 (offline, label 없음) — snapshot ${AG3S_PROJECT_ROOT}-sg", "(1) regression baseline (offline, no label) on snapshot ${AG3S_PROJECT_ROOT}-sg"), fontsize=10, loc="left")
tabs.append(axs[1].table(cellText=t1, colLabels=["serve_safe flags (+ --config/--checkpoint/--model-xml)", "parse", "resolved args.subtask_gate",
                                                  "build_ag3s config section", "startup log note"],
                         cellLoc="left", colWidths=[0.24, 0.13, 0.12, 0.2, 0.31], bbox=[0, 0, 1, 0.92]))
axs[1].set_title(L("(2) flag 해석 (모델 없이, snapshot 의 build_parser · reject_bad_flag_combinations · resolve_subtask_gate · subtask_config_section)",
                   "(2) flag resolution without a model (snapshot functions)") + f'  ·  config default clustering.subtask_gate = {FM["ClusteringConfig_default_subtask_gate"]}  ·  manifest: {FM["main_manifest_line"][0]}',
                 fontsize=9.5, loc="left")
tabs.append(axs[2].table(cellText=t2, colLabels=["live server", "startup log (gate lines)", "chunk summary subtask.gate values"],
                         cellLoc="left", colWidths=[0.17, 0.6, 0.23], bbox=[0, 0, 1, 0.8]))
axs[2].set_title(L("(3) 띄운 서버의 기동 로그와 기록 (meta.json 은 SIGTERM 종료에서 쓰이지 않는다)", "(3) live servers: startup log and per-chunk record (meta.json not written on SIGTERM)"), fontsize=10, loc="left")
tabs.append(axs[3].table(cellText=t3, colLabels=["new run", "reference", "executed\nactions equal", "planning\nactions equal",
                                                  "decision fields\ndiffering\n(non-subtask / 14)", "S/G/P ref | new", "HOLD chunks ref | new",
                                                  "subtask_gated\nchunks ref | new", "subtask.gate ref | new", "other leaf diffs"],
                         cellLoc="center", colWidths=[0.1, 0.14, 0.07, 0.07, 0.11, 0.09, 0.08, 0.09, 0.09, 0.16], bbox=[0, 0, 1, 0.88]))
axs[3].set_title(L("(4) 폐루프 2 run × 2 조건 vs 기준 run (같은 episode · seed)", "(4) closed loop, 2 runs x 2 conditions vs reference runs (same episode, seed)"), fontsize=10, loc="left")
DD = lambda x: x.replace("--", "-\u200b-")  # keep "--" from being drawn as a dash ligature
for tb in tabs:
    for cl in tb.get_celld().values():
        cl.get_text().set_text(DD(cl.get_text().get_text()))
    tb.auto_set_font_size(False); tb.set_fontsize(7.3)
    for (r, c), cl in tb.get_celld().items():
        cl.set_edgecolor(fs.GRID_INK)
        if r == 0:
            cl.set_facecolor("#efeeea"); cl.get_text().set_wrap(True)
fig.patch.set_facecolor(fs.SURFACE); fig.tight_layout()
p = OUT / "subtask-g-table.png"; fig.savefig(p, dpi=130, facecolor=fs.SURFACE); plt.close(fig)
p.with_suffix(".json").write_text(json.dumps({"sources": {"baseline": str(G / "baseline/judge.json"), "flags": str(G / "flags/flag_matrix.json"),
    "analysis": str(G / "analysis/summary.json")}, "baseline": t0, "flags": t1, "startup": t2, "closed_loop": t3,
    "decision_fields": list(next(iter(S["pairs"].values()))["decision_field_diff"].keys())}, indent=1, ensure_ascii=False))
print(p)
