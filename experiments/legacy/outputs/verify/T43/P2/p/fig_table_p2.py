"""T43 P2 figure (table): per episode, the chosen overbar_0 (crate frame u, v; dyaw; bar-centre height h; leg spacing L; bar radius r)
and the criteria: a' = min robot clearance over reset + 1000-step warmup + t0 (all robot geoms; uncapped, check_seeds.py);
client reset (arms / objects, the client's own check, dryrun_p2.py); b = crate min (legs outside the crate: yes/no); d = approach + grasp
min (robot, apple; uncapped); c = min(GRIP, APPLE) over transport per run (E0 / E3b; negative = recorded path penetrates the bar);
other seeds = c over every other recorded E0/E3b run of the episode that lifted the apple (n hit / n); e = free height over the bar
(kinematic upper bound) and drop gap (bar edge to far interior wall; apple width 64-67 mm).  Numbers in mm.
-> AG3S/ag3s/docs/figures/t43/t43p2-table.png + .json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P2/p'))
from AG3S.ag3s.experiments.common import figstyle
import p2lib as F
figstyle.use_korean()
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
mm = lambda v: "-" if v is None else f"{v * 1000:.1f}"
HEAD = ["ep (arm)", "u", "v", "dyaw", "h (m)", "L (m)", "r", "a' reset+warmup+t0", "client reset arms / obj", "b crate", "legs out",
        "d approach+grasp", "c E0", "c E3b", "other seeds hit", "free height (e)", "drop gap", "passes"]
rows, data = [], {"definitions": __doc__, "rows": []}
for tag in F.TAGS:
    s = json.loads((F.RAW / f"select_p2_{tag}.json").read_text()); e = s["chosen"]; k = json.loads((F.RAW / f"check_seeds_{tag}.json").read_text())
    u = k["uncapped_search_seed"]; seed = tag.split("_")[1]
    oth = [r for n, r in k["runs"].items() if not n.endswith(seed) and r.get("car_min") is not None]
    hit = sum(r["car_min"] <= 0 for r in oth)
    arm = "R" if s["side"] < 0 else "L"
    ok = "".join(q for q, f in (("a", "pass_a"), ("b", "pass_b"), ("c", "pass_c"), ("d", "pass_d"), ("e", "pass_e")) if e.get(f))
    dr = s["dryrun"]
    r = [f"{tag[:4]} ({arm})", f"{e['u'] * 1000:.0f}", f"{e['v'] * 1000:+.0f}", f"{e['dyaw_deg']:.0f}", f"{e['h']:.2f}", f"{e['L']:.2f}",
         f"{e['r'] * 1000:.0f}", mm(u["a_min"]), f"{mm(dr['robot_clearance_m'])} / {mm(dr['object_clearance_m'])}", (">=60" if e["b_crate_min"] >= 0.06 else mm(e["b_crate_min"])),
         "yes" if e["legs_outside_crate"] else "no", mm(u["d_min"]), mm(e["c_min_per_run"]["E0"]), mm(e["c_min_per_run"]["E3b"]),
         f"{hit}/{len(oth)}", mm(e["free_height_over_bar_m"]), mm(e["drop_gap_m"]), ok]
    rows.append(r); data["rows"].append(dict(zip(HEAD, r)) | {"tag": tag, "config": s["config"]})
fig, ax = plt.subplots(figsize=(24, 0.45 * len(rows) + 1.5)); fig.patch.set_facecolor(figstyle.SURFACE); ax.set_axis_off()
t = ax.table(cellText=rows, colLabels=HEAD, loc="center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 1.4)
for (i, j), cell in t.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if i == 0:
        cell.set_facecolor(figstyle.SEQ_BLUE[0]); cell.set_text_props(color=figstyle.INK)
ax.set_title("T43 P2 overbar_0 across the crate inside the handles (mm unless noted; u, v = bar centre in the crate frame, +v = left-handle side; "
             "c < 0 = recorded path penetrates the bar; passes: a=a' >= 30, b, c, d >= 30, e >= 50)", color=figstyle.INK, fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "t43p2-table.png", dpi=130, facecolor=figstyle.SURFACE)
(OUT / "t43p2-table.json").write_text(json.dumps(data, indent=1)); print(OUT / "t43p2-table.png")
