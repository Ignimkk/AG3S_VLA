"""T40 F figure: chosen hurdle placements (round 1 and round 2) at the client's reset scene with the recorded T39 apple-centre
paths [lift_t, place_t] (blue E0 = CATEGORICAL[0], orange E3b = CATEGORICAL[1]); oblique renders from render_f.py.
-> AG3S/ag3s/docs/figures/t40/t40fg-placement.png + .json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle
figstyle.use_korean()
R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/F/render')); RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/F/raw'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t40'))
tags = ["1807_18071", "1968_19681", "1982_19822", "1995_19952", "1967_19672"]
fig, axes = plt.subplots(2, 5, figsize=(20, 6.4)); fig.patch.set_facecolor(figstyle.SURFACE)
data = {"definitions": __doc__, "panels": {}}
for j, t in enumerate(tags):
    for i, (rt, sel) in enumerate((("r1", "select"), ("r2", "dselect"))):
        ax = axes[i, j]; ax.set_axis_off()
        p = R / f"{t}_{rt}_oblique.png"; f = RAW / f"{sel}_{t}.json"
        if not p.exists() or not f.exists():
            ax.text(0.5, 0.5, "no round 2" if rt == "r2" else "-", ha="center", va="center", color=figstyle.INK_2); continue
        e = json.loads(f.read_text())["chosen"]
        ax.imshow(Image.open(p))
        ax.set_title(f"ep{t[:4]} {rt}: ({e['x']:.3f}, {e['y']:.3f}) yaw {e['yaw'] * 57.2958:.1f} deg  bar {e['h']:.2f} m", color=figstyle.INK, fontsize=8)
        data["panels"][f"{t}_{rt}"] = {"image": str(p), "x": e["x"], "y": e["y"], "yaw_rad": e["yaw"], "bar_height_m": e["h"]}
fig.suptitle("T40 F/D hurdle placements at the client reset scene; lines = recorded T39 apple paths (blue E0, orange E3b). ep1828: no placement",
             color=figstyle.INK, fontsize=11)
fig.tight_layout(h_pad=1.6); fig.savefig(OUT / "t40fg-placement.png", dpi=130, facecolor=figstyle.SURFACE)
(OUT / "t40fg-placement.json").write_text(json.dumps(data, indent=1)); print(OUT / "t40fg-placement.png")
