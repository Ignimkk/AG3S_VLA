"""T59 presentation slide 2: key-part timeline of ep1809 (frame 0 .. W2 end). No new measurement; only t59_frames_W2.jsonl / t59_w2_window.json.
Rows: physical phase, HSR confirmed label, U (phase-unsplit, current-frame attention), Sg (phase-split, current system).
N=1 variant (illustration, lead 2026-10-13): HSR label = raw per-frame argmax (no debounce); Sg release re-coloured at the first frame with
argmax == home and placed_seen and not frozen. Serving actually ran N=3. Usage: python <this> <verify/T59 dir> <FIG dir>"""
import hashlib, json, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory as btf
sys.path.insert(0, sys.argv[3])
from t59_fig_common import CLS_COLOR, PHASE_COLOR, HSR_COLOR, KO, T  # noqa
from AG3S.ag3s.experiments.common import figstyle as fs

V, FIG = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
KEY = "ep1809"
NAME = "t59-ppt-timeline-ep1809-key-N1"
for ext in (".png", ".json"):
    if (FIG / (NAME + ext)).exists():
        raise SystemExit(f"{FIG / (NAME + ext)} exists")
sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
fr = [json.loads(l) for l in open(V / "t59_frames_W2.jsonl")]
win = json.loads((V / "t59_w2_window.json").read_text())
ep = win["episodes"][KEY]
w = sorted([f for f in fr if f["key"] == KEY and f["in_window_W2"]], key=lambda f: f["frame"])
x = [f["frame"] for f in w]
assert x[0] == 0 and x[-1] == ep["W2_end"] == 39 and len(w) == 40

# --- N=1 counterfactual (illustration) ---
for f in w:
    f["hsr_N1"] = str(f["S_hsr_argmax"]) if f["S_hsr_argmax"] is not None else "None"
rel_N1 = next(f["frame"] for f in w if f["S_hsr_argmax"] == "home" and f["S_subtask_placed_seen"] and not f["S_decision_frozen"])
rel_N3 = next(f["frame"] for f in w if f["S_class_at_geometry_frame"] == "no_target" and f["S_hsr_label"] == "home")
for f in w:
    f["Sg_N1"] = "no_target" if f["frame"] >= rel_N1 else f["S_class_at_geometry_frame"]
changed_hsr = [f["frame"] for f in w if f["hsr_N1"] != str(f["S_hsr_label"]) and f["S_hsr_label"] is not None]  # frames 0-1: no label yet under N=3 (start-up), not outlined
changed_sg = [f["frame"] for f in w if f["Sg_N1"] != f["S_class_at_geometry_frame"]]
held = [f for f in w if f["phase"] == "held"]
N = sum(f["U_class"] != "target" for f in held)
rel = [f for f in w if f["Sg_N1"] == "no_target"]
M = sum(f["U_class"] == "target" for f in rel)
held_rng = (held[0]["frame"], held[-1]["frame"]); rel_rng = (rel[0]["frame"], rel[-1]["frame"])
assert [f["frame"] for f in held] == list(range(held_rng[0], held_rng[1] + 1))
assert [f["frame"] for f in rel] == list(range(rel_rng[0], rel_rng[1] + 1))

FS = 18
plt.rcParams.update({"font.size": FS})
NONE_LAB = T("없음", "none")
fig = plt.figure(figsize=(16, 9), facecolor=fs.SURFACE)
L, R = 0.225, 0.985
rows = [("phase", T("물리 phase", "physical phase"), [f["phase"] for f in w], PHASE_COLOR),
        ("hsr", T("HSR label\n(N=1: 바로 확정)", "HSR label\n(N=1)"), [f["hsr_N1"] for f in w], HSR_COLOR),
        ("U", T("phase 미분할:\n현재 frame attention", "phase-unsplit:\ncurrent-frame attention"), [f["U_class"] for f in w], CLS_COLOR),
        ("Sg", T("phase 분할:\n현재 시스템 (N=1)", "phase-split:\ncurrent system (N=1)"), [f["Sg_N1"] for f in w], CLS_COLOR)]
RH, GAP, TOP = 0.105, 0.032, 0.675   # figure fractions
axs = {}
for i, (k, lab, vals, cmap) in enumerate(rows):
    y0 = TOP - i * (RH + GAP) - RH
    ax = fig.add_axes([L, y0, R - L, RH]); axs[k] = ax
    for xi, v in zip(x, vals):
        ax.add_patch(plt.Rectangle((xi - 0.5, 0), 1, 1, facecolor=cmap.get(v, "#ffffff"), edgecolor=fs.SURFACE, lw=1.0,
                                   hatch="///" if v in ("no_target", "None") else None))
    i0 = 0
    while i0 < len(vals):
        j = i0
        while j + 1 < len(vals) and vals[j + 1] == vals[i0]:
            j += 1
        v = str(vals[i0]); txt = NONE_LAB if v == "no_target" else ("-" if v == "None" else v)
        run_in = (j - i0 + 1) * 16 * (R - L) / 40
        if (j - i0 >= 1 or v == "no_target") and len(txt) * 0.17 <= run_in and not (v == "no_target" and j == i0):
            col = matplotlib.colors.to_rgb(cmap.get(vals[i0], "#ffffff"))
            ax.text((x[i0] + x[j]) / 2, 0.5, txt, ha="center", va="center", fontsize=FS - 1,
                    color="white" if sum(col) / 3 < 0.55 else fs.INK,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85) if v == "no_target" else None)
        i0 = j + 1
    ax.set_xlim(-0.5, 39.5); ax.set_ylim(0, 1); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=FS, color=fs.INK, labelpad=12)
    ax.tick_params(labelbottom=False, length=0)
# (no outline on changed cells — user 2026-10-13)
axb = axs["Sg"]
axb.tick_params(labelbottom=True, length=5, labelsize=FS, colors=fs.INK_2)
axb.set_xticks(range(0, 40, 5)); axb.spines["bottom"].set_visible(False)
axb.set_xlabel(T("frame (= t_step / 8)", "frame (= t_step / 8)"), fontsize=FS, color=fs.INK)

def bracket(ax, a, b, y, tick=0.12, lw=2.4):
    tr = btf(ax.transData, ax.transAxes)
    ax.plot([a, a, b, b], [y - tick, y, y, y - tick], color=fs.INK, lw=lw, transform=tr, clip_on=False, solid_capstyle="butt")
    return tr

atop = axs["phase"]
# a) held span
tr = bracket(atop, held_rng[0] - 0.5, held_rng[1] + 0.5, 1.20)
atop.text((held_rng[0] + held_rng[1]) / 2 - 3.0, 1.30, T(f"들고 가는 중: {N}/{len(held)} frame\n다른 과일 또는 없음", f"carrying: {N}/{len(held)} frames\nother fruit or none"),
          transform=tr, ha="center", va="bottom", fontsize=FS, color=fs.INK, linespacing=1.25)
# b) released span (leader up to a higher tier)
tr = bracket(atop, rel_rng[0] - 0.5, rel_rng[1] + 0.5, 1.20)
mid = (rel_rng[0] + rel_rng[1]) / 2
atop.plot([mid, mid], [1.20, 1.78], color=fs.INK, lw=2.0, transform=tr, clip_on=False)
atop.text(39.5, 1.84, T(f"해제 뒤: 미분할은 놓인 banana 를 다시\ntarget 으로 ({M} frame)", f"after release: unsplit picks the placed\nbanana as target again ({M} frames)"),
          transform=tr, ha="right", va="bottom", fontsize=FS, color=fs.INK, linespacing=1.25)

fig.text(0.02, 0.965, T("ep1809 (banana) — 들고 가는 중 · 놓은 뒤, attention 만으로 고른 target 대 현재 시스템",
                        "ep1809 (banana) - carrying / after placing: attention-only target vs current system"),
         fontsize=21, color=fs.INK, ha="left", va="top")
fig.text(0.02, 0.915, T("16D 실제 rollout, frame 0–39 · HSR label 은 매 frame 판단을 바로 확정 (N=1) 한 것으로 표시",
     "16D real rollout, frame 0-39 · HSR label shown with N=1 (per-frame decision)"),
         fontsize=FS, color=fs.INK_2, ha="left", va="top")
h = [Patch(facecolor=CLS_COLOR["target"], label="target"), Patch(facecolor=CLS_COLOR["other_object"], label="other_object"),
     Patch(facecolor="white", edgecolor=fs.GRID_INK, hatch="///", label=NONE_LAB)]
h += [Patch(facecolor=PHASE_COLOR[p], label=p) for p in ("pre", "held", "placed")]
fig.legend(handles=h, loc="lower center", ncol=6, fontsize=FS - 1, frameon=False, bbox_to_anchor=(0.5, 0.005),
           handlelength=1.6, columnspacing=1.6, handletextpad=0.5)
out = FIG / (NAME + ".png")
fig.savefig(out, dpi=150, facecolor=fs.SURFACE); plt.close(fig)

def runs(vals):
    r, i = [], 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        r.append({"value": vals[i], "frames": [x[i], x[j]], "n": j - i + 1}); i = j + 1
    return r
inputs = {str(p): sha(p) for p in (V / "t59_frames_W2.jsonl", V / "t59_w2_window.json")}
side = {"figure": out.name, "episode": KEY, "target": ep["target"], "used_arm": ep["used_arm"], "figsize_in": [16, 9], "dpi": 150,
        "inputs_sha256": inputs, "script": "AG3S/ag3s/docs/figures/ppt-1013/t59_key_N1.py (derived from verify/T59/p/t59_w2_key_figs_ep1809.py)",
        "window_frames": [x[0], x[-1]], "W2_end_reason": ep["W2_end_reason"], "ready_dist_at_W2_end_rad": ep["ready_dist_at_W2_end_rad"],
        "colour_source": "t59_fig_common.py (CLS_COLOR, PHASE_COLOR, HSR_COLOR), same as t59-ppt-timeline-ep1809-W2.png",
        "rows_top_to_bottom": {"physical phase": "phase", "HSR confirmed label": "S_hsr_label", "phase-unsplit (U)": "U_class", "phase-split, current system (Sg)": "S_class_at_geometry_frame"},
        "row_runs": {"phase": runs([f["phase"] for f in w]), "S_hsr_label": runs([str(f["S_hsr_label"]) for f in w]),
                     "U_class": runs([f["U_class"] for f in w]), "S_class_at_geometry_frame": runs([f["S_class_at_geometry_frame"] for f in w])},
        "annotation_a": {"definition": "held frames with U_class != target", "held_frame_range": list(held_rng), "n_held": len(held), "N": N,
                         "frames_U_not_target": [f["frame"] for f in held if f["U_class"] != "target"],
                         "U_by_frame": {f["frame"]: f["U_class"] for f in held}},
        "annotation_b": {"definition": "N=1 released frames (Sg_N1 == no_target); count U_class == target", "frame_range": list(rel_rng),
                         "n_frames": len(rel), "M": M, "U_by_frame": {f["frame"]: f["U_class"] for f in rel}},
        "N1_counterfactual": {"note": "illustration only, not a measurement of an N=1 server; serving ran subtask_confirm_frames=3", "hsr_row": "S_hsr_argmax per frame",
                        "release_rule": "first frame with argmax==home and S_subtask_placed_seen and not S_decision_frozen", "release_frame_N1": rel_N1, "release_frame_N3_actual": rel_N3,
                        "changed_hsr_frames": changed_hsr, "changed_Sg_frames": changed_sg},
        "dropped_from_old_figure": ["serving latch row", "HSR probability curves", "selection-rule sentence", "dotted W1-end line", "triangle markers"],
        "label_note": "no_target shown as 'none' in Korean ('없음'); ASCII fallback 'none' if no CJK font"}
out.with_suffix(".json").write_text(json.dumps(side, indent=1, ensure_ascii=False))
print(out, "N", N, "M", M, held_rng, rel_rng)
