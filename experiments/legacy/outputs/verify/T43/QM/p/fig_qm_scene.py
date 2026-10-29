"""T43 QM real crossing scene: third_person_front.mp4 frames (recorded run, post state of row t) of V8 and V11AB of one ep:seed at their measured apple
crossing rows, with the apple-centre paths of BOTH arms projected into the recording camera (free camera of pi05_infer.configure_view_camera 'front':
lookat (0.45, 0, 0.85), distance 1.7, azimuth 180, elevation -18, fovy 45 deg (model vis.global fovy), 640 x 480).
Measured path = qm_rows apple_M (post state). Commanded path = xm_cross s_C / z_C of the apple centre (commanded state, apple rigid with the palm) placed at the
measured along-bar coordinate of the same row (the camera looks along the bar axis, so that coordinate is depth only). In-hand rows up to Z4 release.
usage: fig_qm_scene.py OUT_PNG EP_SEED (e.g. 1807_s18073)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, subprocess, tempfile
sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out = pathlib.Path(sys.argv[1]); assert not out.exists() and not out.with_suffix(".json").exists(), f"{out} exists"
ES = sys.argv[2]; T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); Q = T / "QM"
CRF = {"V8": T / "XM/raw/cross_V8.json", "V11AB": T / "V11/raw/cross_V11AB.json"}
D = json.load(open(Q / "qm_metrics.json")); P = D["per_run"]
figstyle.use_korean(); C = figstyle.CATEGORICAL; COL = {"V8": C[0], "V11AB": C[1]}
W, H, FOVY = 640, 480, 45.0
az, el = np.deg2rad(180.0), np.deg2rad(-18.0); look = np.array([0.45, 0.0, 0.85]); dist = 1.7
fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]); up = np.array([-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)])
right = np.cross(fwd, up); cam = look - dist * fwd; fpx = (H / 2) / np.tan(np.deg2rad(FOVY) / 2)


def proj(p):
    v = np.asarray(p) - cam; zc = v @ fwd
    return np.array([W / 2 + fpx * (v @ right) / zc, H / 2 - fpx * (v @ up) / zc])


paths = {}; frames = {}
for arm in ("V8", "V11AB"):
    key = f"{arm}/H/{arm}_ep{ES}"; r = json.load(open(CRF[arm]))["runs"][key]; o = P[f"H:{ES}"][arm]
    rw = np.load(Q / "raw/rows" / f"{arm}_H_{arm}_ep{ES}.npz")
    bc = np.array(r["bar"]["centre"]); ax_ = np.array(r["bar"]["axis_h"]); n = np.array(r["bar"]["n_travel"]); top = r["bar"]["top_z"]
    rel = o["release_t"]; rows = [x for x in r["rows"] if x["inhand"] and (rel is None or x["t"] <= rel)]
    pm, pc = [], []
    for x in rows:
        aM = rw["apple_M"][x["t"]]; a_along = float((aM - bc) @ ax_)
        pm.append(aM); pc.append(bc + 1e-3 * x["s_C"]["apple"] * n + a_along * ax_ + np.array([0, 0, top - bc[2] + 1e-3 * x["z_C"]["apple"]]))
    paths[arm] = {"t": [x["t"] for x in rows], "M_world": np.array(pm), "C_world": np.array(pc), "crossing_t": o["crossing_t"], "bar": r["bar"], "mp4": r["mp4"]}
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["ffmpeg", "-v", "error", "-i", r["mp4"], "-vf", f"select='eq(n\\,{o['crossing_t']})'", "-vsync", "0", f"{td}/f.png"], check=True)
        frames[arm] = mpimg.imread(f"{td}/f.png")
fig, axs = plt.subplots(2, 2, figsize=(17, 12.5), gridspec_kw={"height_ratios": [1, 1.05]}); side = {"script": __file__, "ep_seed": ES, "camera": {"lookat": look.tolist(), "distance": dist,
        "azimuth_deg": 180.0, "elevation_deg": -18.0, "fovy_deg": FOVY, "size": [W, H]}, "panels": {}}
bar = paths["V11AB"]["bar"]; bc = np.array(bar["centre"]); ax_ = np.array(bar["axis_h"]); hl = 1e-3 * bar["half_length_mm"]
bar_px = [proj(bc - hl * ax_).tolist(), proj(bc + hl * ax_).tolist(), proj(bc).tolist()]
for j, arm in enumerate(("V8", "V11AB")):
    for row, zoom in ((0, False), (1, True)):
        a = axs[row, j]; a.imshow(frames[arm]); a.axis("off")
        for arm2 in ("V8", "V11AB"):
            pm = np.array([proj(p) for p in paths[arm2]["M_world"]]); pc = np.array([proj(p) for p in paths[arm2]["C_world"]])
            a.plot(pm[:, 0], pm[:, 1], color=COL[arm2], lw=2.4 if zoom else 1.8, label=f"{arm2} apple centre, measured M")
            a.plot(pc[:, 0], pc[:, 1], color=COL[arm2], lw=1.2, ls="--", label=f"{arm2} apple centre, commanded C")
            tx = paths[arm2]["crossing_t"]
            if tx in paths[arm2]["t"]:
                q = pm[paths[arm2]["t"].index(tx)]; a.scatter([q[0]], [q[1]], color=COL[arm2], s=50, edgecolor="black", zorder=5)
            side["panels"].setdefault(arm2, {"t": paths[arm2]["t"], "M_px": pm.tolist(), "C_px": pc.tolist(), "M_world": paths[arm2]["M_world"].tolist(),
                                             "C_world": paths[arm2]["C_world"].tolist(), "crossing_t": tx,
                                             "h_at_crossing_M_mm": P[f"H:{ES}"][arm2]["apple_bottom_minus_bar_top_at_crossing_M_mm"],
                                             "h_at_crossing_C_mm": P[f"H:{ES}"][arm2]["apple_bottom_minus_bar_top_at_crossing_C_mm"],
                                             "contact_apple_carry": P[f"H:{ES}"][arm2]["contact_apple_carry"], "video_frame": tx})
        a.scatter([bar_px[2][0]], [bar_px[2][1]], marker="+", color="white", s=60, zorder=6)
        if zoom:
            cx, cy = bar_px[2]; a.set_xlim(cx - 150, cx + 150); a.set_ylim(cy + 90, cy - 150)
        o = P[f"H:{ES}"][arm]
        a.set_title(f"{arm} ep{ES} video frame t={o['crossing_t']} ({o['crossing_t'] / 15:.1f} s) = its measured apple crossing row{' (zoom)' if zoom else ''}\n"
                    f"apple bottom - bar top at crossing: M {o['apple_bottom_minus_bar_top_at_crossing_M_mm']:.1f} mm, C {o['apple_bottom_minus_bar_top_at_crossing_C_mm']:.1f} mm; "
                    f"apple-hurdle contact in carry {o['contact_apple_carry']}", fontsize=9)
axs[0, 0].legend(fontsize=8, loc="lower left")
side["bar_px"] = bar_px
fig.suptitle("T43 QM real crossing scene: both arms' apple-centre paths (in hand, to release) projected into the recording camera; + = bar centre; dot = crossing row", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=100, facecolor=figstyle.SURFACE); out.with_suffix(".json").write_text(json.dumps(side, default=float)); print(out)
