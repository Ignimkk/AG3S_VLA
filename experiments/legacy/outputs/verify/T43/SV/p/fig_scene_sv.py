"""T43 SV real-scene figure (verifier, CPU only): the closing moment of one pair, no-servo V8 vs servo V9, with the binding rows.
usage: T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8 fig_scene_sv.py OUT_PNG STAGE EPSEED     (e.g. H ep1968_s19681)
cols 1-2: MuJoCo render of the measured state at the planning row of each arm's first closing chunk (frames.jsonl planning qpos, full
          state incl. apple), free camera at the apple (azimuth / elevation / distance in the sidecar). Markers: the binding row of the
          chunk before closing and of the closing chunk (sphere centre at the recorded worst point, base -> world with the planning-row FK,
          radius = that query sphere); colour = binding kind (waypoint / path).
col 3:   finger height over the table top (min over finger spheres of z - table_top_z - r, MuJoCo truth plane) per control row around
          the first closing, rows relative to it: plan P, servo prediction S, measured M (raw/rows_{V8,V9}.json, sv_rows.py).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p')); sys.path.insert(0, _ag3s_workspace_path(''))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
import hold_w as HW  # noqa: E402
import dx_common as C  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

out_png = pathlib.Path(sys.argv[1]); stage, epseed = sys.argv[2], sys.argv[3]
assert not out_png.exists(), f"{out_png} exists"
SV = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/SV/raw'))
G = json.load(open(SV / "grasp_V9V8.json"))["runs"][f"V9/{stage}/V9_{epseed}"]
ROWS = {a: json.load(open(SV / f"rows_{a}.json"))["runs"][f"{a}/{stage}/{a}_{epseed}"] for a in ("V8", "V9")}
AN = {"V8": _ag3s_workspace_path('outputs/verify/T43/V8/analysis/analysis.json'), "V9": _ag3s_workspace_path('outputs/verify/T43/V9/analysis/analysis.json')}
cl = HW.load_client(); kor = figstyle.use_korean()
CAT = figstyle.CATEGORICAL
COL = {"P": CAT[1], "S": CAT[2], "M": CAT[0]}
KIND = {"waypoint": CAT[3], "path": CAT[4]}


def hexrgba(h, a=1.0):
    h = h.lstrip("#"); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a], np.float32)


fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0]})
side = {"script": __file__, "stage": stage, "episode_seed": epseed, "snapshot": str(HW.HY), "panels": []}
for col, arm in enumerate(("V8", "V9")):
    a = json.load(open(AN[arm]))["runs"][f"{arm}/{stage}/{arm}_{epseed}"]; rd = pathlib.Path(a["run_dir"])
    man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = man["scene"]["episode_index"]
    z = np.load(rd / "traj.npz", allow_pickle=True); seed = int(z["policy_seed"])
    stt = rd.parent / f"{rd.name}.start"
    cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
    L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]
    fr = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]; plan = {r["t_step"]: r for r in fr if r["kind"] == "planning"}
    ga = G["arms"][arm]; fc = ga["first_closing_t"]; ch = {c["t"]: c for c in ga["chunks"]}
    q = np.asarray(plan[fc]["qpos"], float); d.qpos[:] = q; mujoco.mj_forward(m, d)
    ab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple"); look = d.xpos[ab].copy()
    Tb = C.T_world_base(m, d)
    marks = []; binfo = []
    for t in (fc - 8, fc):
        c = ch.get(t)
        if c is None:
            continue
        b = c["binding"]; rc = c.get("recon") or {}; x = (rc.get("path") if b["rows"] == "path" else rc.get("wp")) or {}
        # recorded worst point (base frame) -> world with this planning row's base pose
        seg = C.lit(a["server_segment"]); chunks = C.segment_chunks(seg)
        rec_pt = None
        p_seq = plan[t]["seq"]
        with np.load(chunks[p_seq], allow_pickle=True) as zz:
            s = json.loads(str(zz["summary_json"])); met = (s.get("to") or {}).get("metrics") or {}
            src = C.lit(met.get("path_check")).get("worst") if b["rows"] == "path" else C.lit(met.get("max_violation_pair"))
            rec_pt = (src or {}).get("point_m")
        if rec_pt is not None:
            d.qpos[:] = np.asarray(plan[t]["qpos"], float); mujoco.mj_kinematics(m, d); Tb_t = C.T_world_base(m, d)
            pw = Tb_t[:3, :3] @ np.asarray(rec_pt, float) + Tb_t[:3, 3]
            rad = 1e-3 * (x.get("radius_mm") or 6.0)
            marks.append((pw, rad, hexrgba(KIND[b["rows"]], 0.85 if t == fc else 0.45)))
        binfo.append({"t": t, "rows": b["rows"], "link": b["link"], "clearance_mm": b["clearance_mm"], "class": x.get("recorded_class") or x.get("class"),
                      "recon_class": x.get("class"), "label": x.get("truth_nearest"), "reads_target_free": x.get("reads_target_free"),
                      "point_world": None if rec_pt is None else pw.tolist(), "outward_mm": c["outward_mm"], "tip_alignment_now": c.get("tip_alignment_now"),
                      "tip_apple_mm": c["tip_apple_mm"], "initial_source": c.get("initial_source")})
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    rr = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = look + np.array([0.0, 0.0, 0.02]); cam.azimuth = 200.0; cam.elevation = -8.0; cam.distance = 0.42
    rr.update_scene(d, camera=cam)
    for cc, rad, rgba in marks:
        s_ = rr.scene
        if s_.ngeom < s_.maxgeom:
            mujoco.mjv_initGeom(s_.geoms[s_.ngeom], int(mujoco.mjtGeom.mjGEOM_SPHERE), np.array([rad, 0, 0]), np.asarray(cc, float), np.eye(3).reshape(-1), rgba); s_.ngeom += 1
    img = rr.render(); rr.close()
    ax = axes[col]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    bc = binfo[-1] if binfo else {}
    tz = (bc.get("tip_alignment_now") or {})
    lines = [f"{arm} ({'servo model' if arm == 'V9' else 'no servo'}) {stage} {epseed}  first closing t={fc}",
             f"fingertip-apple {bc.get('tip_apple_mm', float('nan')):.0f} mm  tip dz {tz.get('dz_mm')} mm  dxy {tz.get('dxy_mm')} mm"]
    for bi in binfo:
        lines.append(f"t={bi['t']} binding {bi['rows']} {bi['link']} {bi['clearance_mm']:.1f} mm  class {bi['class']} near {bi['label']}  outward {bi['outward_mm']:+.1f}")
    ax.set_title("\n".join(lines), fontsize=8.2)
    side["panels"].append({"arm": arm, "first_closing_t": fc, "camera": {"lookat": (look + np.array([0, 0, 0.02])).tolist(), "azimuth": 200.0, "elevation": -8.0, "distance": 0.42},
                           "binding": binfo, "marker_colour": {"waypoint": KIND["waypoint"], "path": KIND["path"]}})
# col 3: finger height over the table
ax = axes[2]; ts_json = {}
for arm, ls in (("V8", "--"), ("V9", "-")):
    r = ROWS[arm]; fc = r["first_closing_t"]; xs = {"P": [], "S": [], "M": []}
    for c in r["chunks"]:
        if c["phase"] != "approach" or not (fc - 24 <= c["t0"] <= fc):
            continue
        for k, (P, S, M) in enumerate(zip(c["P"], c["S"], c["M"])):
            t = c["t0"] + k - fc
            xs["P"].append((t, P[1]["finger_table_mm"])); xs["S"].append((t, S[1]["finger_table_mm"]))
            if M is not None:
                xs["M"].append((t + 1, M[1]["finger_table_mm"]))
    for st in ("P", "S", "M"):
        v = np.asarray(sorted(xs[st]), float)
        if len(v):
            ax.plot(v[:, 0], v[:, 1], ls, color=COL[st], lw=2.2 if st == "M" else 1.3, label=f"{arm} {st}")
    ts_json[arm] = {k: sorted(v) for k, v in xs.items()}
ax.axhline(0, color=figstyle.INK_2, lw=0.8); ax.axvline(0, color=figstyle.INK_2, lw=0.8, ls=":")
ax.set_xlabel("control row relative to the first closing chunk"); ax.set_ylabel("finger sphere height over table top (mm)")
ax.set_title("finger height over the table: P plan, S servo prediction, M measured\n(dashed V8 no servo, solid V9 servo)", fontsize=8.5)
ax.set_ylim(-5, 60); ax.legend(fontsize=7, ncol=2); figstyle.style_axes(fig, [ax])
side["finger_table_series"] = ts_json
fig.suptitle("T43 SV - closing moment, V8 (no servo) vs V9 (servo model): binding rows and finger height (marker spheres = binding rows; waypoint / path colours in the sidecar)", fontsize=10.5)
fig.tight_layout(); fig.savefig(out_png, dpi=125)
json.dump(side, open(out_png.with_suffix(".json"), "w"), indent=1, default=float)
print("wrote", out_png)
