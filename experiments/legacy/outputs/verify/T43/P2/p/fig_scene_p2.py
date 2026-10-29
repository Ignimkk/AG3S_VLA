"""T43 P2 figure (scene): rendered t=0 scene (recorded planning row 0 of E0, after the 1000-step warmup) per episode with the T43 P2
overbar_0 (red rgba, config activated by manager.activate; render_p2.py), front preset (top) and oblique close view (bottom);
lines = recorded apple-centre paths [lift_t, place_t] (blue E0 = CATEGORICAL[0], orange E3b = CATEGORICAL[1]).
-> AG3S/ag3s/docs/figures/t43/t43p2-scene.png + .json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P2/p'))
from AG3S.ag3s.experiments.common import figstyle
import p2lib as F
figstyle.use_korean()
R = F.ROOT / "render"; OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
tags = F.TAGS
fig, axes = plt.subplots(2, len(tags), figsize=(3.6 * len(tags), 6.6)); fig.patch.set_facecolor(figstyle.SURFACE)
data = {"definitions": __doc__, "panels": {}}
for j, t in enumerate(tags):
    s = json.loads((F.RAW / f"select_p2_{t}.json").read_text()); e = s["chosen"]
    for i, view in enumerate(("front", "oblique")):
        ax = axes[i, j]; ax.set_axis_off(); p = R / f"{t}_{view}.png"; ax.imshow(Image.open(p))
        if i == 0:
            ax.set_title(f"ep{t[:4]} s{t[5:]}\nv {e['v'] * 1000:+.0f} mm, h {e['h']:.2f} m, L {e['L']:.2f} m\n"
                         f"a' {e['a_min'] * 1000:.0f}  d {e['d_min'] * 1000:.0f}  crate {e['b_crate_min'] * 1000:.0f} mm",
                         color=figstyle.INK, fontsize=8.5)
        data["panels"][f"{t}_{view}"] = {"image": str(p), "config": s["config"], **{k: e[k] for k in ("x", "y", "yaw", "h", "L", "r", "u", "v", "dyaw_deg", "a_min", "d_min", "b_crate_min")}}
fig.suptitle("T43 P2 overbar_0 (red) across the crate inside the handles, t=0 after warmup; lines = recorded apple paths (blue E0, orange E3b); "
             "a', d, crate in mm (60 = the 60 mm search cap; uncapped values in the table)", color=figstyle.INK, fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "t43p2-scene.png", dpi=110, facecolor=figstyle.SURFACE)
(OUT / "t43p2-scene.json").write_text(json.dumps(data, indent=1)); print(OUT / "t43p2-scene.png")
