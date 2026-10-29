"""T43 P figure 2 (scene): rendered t=0 scene (recorded planning row 0 of E0, after the 1000-step warmup) per episode with the T43 P
hurdle (render_p.py), front preset (top) and oblique close view (bottom); lines = recorded T39 apple-centre paths [lift_t, place_t]
(blue E0 = CATEGORICAL[0], orange E3b = CATEGORICAL[1]).  ep1995: no placement passes all criteria; the panel shows the trade-off
candidate d_best_given_abc (passes a', b, c; d below 30 mm).  -> AG3S/ag3s/docs/figures/t43/t43p-scene.png + .json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle
figstyle.use_korean()
R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P/render')); RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P/raw'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
tags = ["1807_18071", "1968_19681", "1982_19822", "1995_19952", "1967_19672"]
fig, axes = plt.subplots(2, 5, figsize=(22, 7)); fig.patch.set_facecolor(figstyle.SURFACE)
data = {"definitions": __doc__, "panels": {}}
for j, t in enumerate(tags):
    s = json.loads((RAW / f"select_p_{t}.json").read_text())
    kind, e = ("main", s["chosen"]) if s.get("chosen") else ("tradeoff", s["tradeoff"]["d_best_given_abc"])
    for i, view in enumerate(("front", "oblique")):
        ax = axes[i, j]; ax.set_axis_off(); p = R / f"{t}_{kind}_{view}.png"
        ax.imshow(Image.open(p))
        if i == 0:
            lab = "T43 P" if kind == "main" else "no pass; trade-off (d fails)"
            ax.set_title(f"ep{t[:4]} {lab}\nu {e['u'] * 1000:+.0f} v {e['v'] * 1000:.0f} mm, dyaw {e['dyaw_deg']:+d} deg, h {e['h']:.2f} m\n"
                         f"a' {e['a_min'] * 1000:.1f}  d {e['d_min'] * 1000:.1f}  crate {e['b_crate_min'] * 1000:.1f} mm",
                         color=figstyle.INK, fontsize=8.5)
        data["panels"][f"{t}_{view}"] = {"image": str(p), "kind": kind, **{k: e[k] for k in ("x", "y", "yaw", "h", "u", "v", "dyaw_deg", "a_min", "d_min", "b_crate_min")}}
fig.suptitle("T43 P hurdle at t=0 (after warmup); lines = recorded T39 apple paths (blue E0, orange E3b); a', d, crate in mm (values >= 60 are the 60 mm cap)",
             color=figstyle.INK, fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "t43p-scene.png", dpi=120, facecolor=figstyle.SURFACE)
(OUT / "t43p-scene.json").write_text(json.dumps(data, indent=1)); print(OUT / "t43p-scene.png")
