"""T43 P figure 3 (table): per episode, T40 F round-1 placement vs T43 P main (a', b, c, d, e) vs T43 P relaxed c, all with the T43 P
metrics (bd6a002 certified distance).  Columns: crate-frame u, v, dyaw, bar height h; a' = min robot clearance over reset + warmup + t0
(all robot geoms); client reset (arms, client's own check); b = crate min; d = approach+grasp min (robot, apple); c = min(GRIP, APPLE)
over transport per run (E0 / E3b, negative = penetration depth); c_apple per run; free height over bar (e).  Numbers in mm.
-> AG3S/ag3s/docs/figures/t43/t43p-table.png + .json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle
figstyle.use_korean()
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P/raw')); OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
TAGS = ["1807_18071", "1968_19681", "1982_19822", "1995_19952", "1967_19672"]
mm = lambda v: "-" if v is None else f"{v * 1000:.1f}"
HEAD = ["ep", "placement", "u (mm)", "v (mm)", "dyaw", "h (m)", "a' reset+warmup+t0", "client reset arms", "b crate", "d approach+grasp",
        "c E0", "c E3b", "c_apple E0", "c_apple E3b", "free height (e)", "passes"]
rows, data = [], {"definitions": __doc__, "rows": []}
for tag in TAGS:
    sel = json.loads((RAW / f"select_p_{tag}.json").read_text()); old = json.loads((RAW / f"old_f_{tag}.json").read_text())
    old["dyaw_deg"] = math.degrees(old["yaw"])  # crate yaw within 0.001 deg of 0 in all 5 episodes
    old["free_height_over_bar_m"] = None
    for name, e in (("T40 F r1", old), ("T43 P", sel.get("chosen")), ("T43 P c-relaxed", sel.get("chosen_relaxed"))):
        if e is None:
            rows.append([tag[:4], name, "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "none"]); continue
        ok = "".join(k for k, f in (("a", "pass_a"), ("b", "pass_b"), ("c", "pass_c"), ("r", "pass_c_relaxed"), ("d", "pass_d"), ("e", "pass_e")) if e.get(f))
        r = [tag[:4], name, f"{e['u'] * 1000:.0f}", f"{e['v'] * 1000:.0f}", f"{e['dyaw_deg']:.0f}", f"{e['h']:.2f}", mm(e["a_min"]),
             mm(e["client_reset_arms"]), mm(e["b_crate_min"]), mm(e["d_min"]), mm(e["c_min_per_run"]["E0"]), mm(e["c_min_per_run"]["E3b"]),
             mm(e["c_apple_per_run"]["E0"]), mm(e["c_apple_per_run"]["E3b"]), mm(e.get("free_height_over_bar_m")), ok]
        rows.append(r); data["rows"].append(dict(zip(HEAD, r)))
fig, ax = plt.subplots(figsize=(22, 0.42 * len(rows) + 1.4)); fig.patch.set_facecolor(figstyle.SURFACE); ax.set_axis_off()
t = ax.table(cellText=rows, colLabels=HEAD, loc="center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.35)
for (i, j), cell in t.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if i == 0:
        cell.set_facecolor(figstyle.SEQ_BLUE[0]); cell.set_text_props(color=figstyle.INK)
    elif rows[i - 1][1] == "T40 F r1":
        cell.set_facecolor("#f3f2ee")
ax.set_title("T43 P hurdle placements (mm; c < 0 = recorded path penetrates the bar; passes: a=a' >= 30, b, c, r=c relaxed, d >= 30, e >= 50)",
             color=figstyle.INK, fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "t43p-table.png", dpi=130, facecolor=figstyle.SURFACE)
(OUT / "t43p-table.json").write_text(json.dumps(data, indent=1)); print(OUT / "t43p-table.png")
