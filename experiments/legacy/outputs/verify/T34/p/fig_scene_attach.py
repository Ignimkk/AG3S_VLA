"""T34 attach-moment scene: vertical slice of the recorded field through the held primary sphere, at the attach chunk.
usage: fig_scene_attach.py OUT_PNG CASE [CASE ...]   CASE = COND:RUN:T:SOURCE   SOURCE = T33 (T33 record + offline rebuilt
T34 spheres from outputs/verify/T34/off/offline_j1j2.py rebuild) | T34 (T34 closed-loop record, held.spheres_parent_frame
= lifted query spheres, held.filter_spheres_parent_frame = fit spheres).
Per case, 3 panels:
  (1) layout: top view (base x-y) of the recorded cloud near the hand (support points grey, target points), the slice line (x-z
      plane at the primary sphere centre's y) and the GT apple centre.
  (2) slice: combined field d (fine+coarse min, fixed_input_ablation.build_field) on the x-z plane y = primary centre y at the
      attach chunk; d <= 0 region filled; contour d = 0; TLS table plane (support_mask points z > 0.5) line; the GT table top
      (MuJoCo geom probe) line; spheres at the attach-chunk pose (q_now = planning row qpos): fit+pad (T33 query / T34 filter
      copy, dashed) and lifted (T34 query, solid); GT apple (sphere of radius 34 mm drawn at its GT centre, dotted).
  (3) profile: d(h) along the plane normal at 6 table columns 60-120 mm from the primary centre, fine layer / coarse layer / combined.
Sidecar .json: sphere centres/radii (base), plane, band values, bottom-above-plane per sphere, GT apple centre."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T34/off'))
from AG3S.ag3s.experiments.common import figstyle as fs
from AG3S.ag3s.fields.curobo_field import layer_from_arrays
import AG3S.trajopt.experiments.fixed_input_ablation as T
import p12_records as P
fs.use_korean()
C = fs.CATEGORICAL
OUTP = pathlib.Path(sys.argv[1]); CASES = sys.argv[2:]
PORT = {"E3a": 8204, "E3b": 8206}
m, d, probe, obj, _ = P.build()


def world_to_base(pw):
    import mujoco
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    return d.xmat[b].reshape(3, 3).T @ (np.asarray(pw, float) - d.xpos[b])


def load(tag, cond, run, t):
    V = pathlib.Path(f"{_ag3s_workspace_path()}/outputs/verify/{tag}")
    A = json.loads((V / "p" / f"{cond}_analysis.json").read_text())["runs"][run]
    rows = {int(x["seq"]): x for x in (json.loads(l) for l in (V / f"{cond}_rec" / run / "frames.jsonl").read_text().splitlines() if l.strip())
            if x.get("kind") == "planning"}
    c = next(c for c in A["chunks"] if c.get("server_chunk") and c["t"] == t)
    f = V / f"{cond}_rec" / c["server_chunk"]
    z = np.load(f, allow_pickle=False)
    rec = V / f"{cond}_rec"
    args = T.build_parser().parse_args(["--constraints", str(rec / f"server_{PORT[cond]}/server_constraints/run_0000"),
                                        "--frames", str(rec / run / "frames.jsonl"), "--server-log", str(rec / f"serve_safe_{PORT[cond]}.log"), "--out", "/dev/null"])
    settings = T.resolve_settings(args, T.parse_server_log(args.server_log), json.loads((rec / run / "manifest.json").read_text()))
    rb, qadr, _ = T.build_robot(settings)
    return z, json.loads(str(z["summary_json"])), rows[int(c["seq"])], rb, qadr, str(f)


fig, axes = plt.subplots(len(CASES), 3, figsize=(18, 5.6 * len(CASES)), squeeze=False,
                         gridspec_kw={"width_ratios": [1, 1.5, 1.1]})
side = {"doc": __doc__, "cases": {}}
for i, case in enumerate(CASES):
    cond, run, t, src = case.split(":"); t = int(t)
    z, s, row, rb, qadr, fpath = load(src, cond, run, t)
    held = s["grasp"]["attached"]["held"]
    link = held["frame_link"]
    q_now = T.q_now_from_qpos(row["qpos"], qadr)
    Tq = rb.link_pose(q_now, link)
    if src == "T34":
        lifted = [(np.asarray(x["centre_m"]), x["radius_mm"] / 1000) for x in held["spheres_parent_frame"]]
        fit = [(np.asarray(x["centre_m"]), x["radius_mm"] / 1000) for x in held["filter_spheres_parent_frame"]]
        sup = held.get("support") or {}
    else:
        import offline_j1j2 as OJ
        ch = OJ.chunks(cond, run); a = next(k for k, c in enumerate(ch) if c["t"] == t)
        ag, _k = OJ.rebuild(cond, run, ch, a)
        lifted = [(np.asarray(x["centre_m"]), x["radius_mm"] / 1000) for x in ag._held_record["spheres_parent_frame"]]
        fit = [(np.asarray(x["centre_m"]), x["radius_mm"] / 1000) for x in held["spheres_parent_frame"]]
        sup = ag._held_record.get("support") or {}
    L = [(Tq[:3, :3] @ c + Tq[:3, 3], r) for c, r in lifted]
    Fz = [(Tq[:3, :3] @ c + Tq[:3, 3], r) for c, r in fit]
    pts = np.asarray(z["raw_cloud"], float); sm = np.asarray(z["support_mask"], bool)
    tab = pts[sm & (pts[:, 2] > 0.5)]
    c0 = tab.mean(0); n = np.linalg.svd(tab - c0, full_matrices=False)[2][-1]; n = n if n[2] > 0 else -n; off = float(n @ c0)
    P.set_q(m, d, row["qpos"])
    apple = world_to_base(row["object_poses"]["apple"]["pos"])
    cy = L[0][0][1]
    # GT table top z (base) under the primary centre: probe distance to table body at a point above the table
    probe_pt = np.array([L[0][0][0], cy, off + 0.05])
    _, _, per = P.nearest(m, d, probe, obj, P.base_to_world(m, d, probe_pt))
    gt_table_z = probe_pt[2] - per["table"]
    field = T.build_field(z, outside_distance=1.0)
    fine = layer_from_arrays(z["esdf_distance"], z["esdf_origin"], float(z["esdf_voxel_size"]))
    coarse = layer_from_arrays(z["esdf_coarse_distance"], z["esdf_coarse_origin"], float(z["esdf_coarse_voxel_size"]))
    # (1) layout
    ax = axes[i, 0]
    near = np.linalg.norm(pts[:, :2] - L[0][0][:2], axis=1) < 0.25
    ax.scatter(pts[near & sm, 0], pts[near & sm, 1], s=0.3, color=fs.GRID_INK, label="support pts")
    ax.scatter(pts[near & ~sm, 0], pts[near & ~sm, 1], s=0.3, color=C[0], label="other pts")
    tp = np.asarray(z["target_points"], float)
    if len(tp):
        ax.scatter(tp[:, 0], tp[:, 1], s=1, color=C[1], label="target pts (chunk)")
    ax.axhline(cy, color=C[7], lw=1.2, label="slice y = primary centre y")
    ax.plot(apple[0], apple[1], "*", color=C[2], ms=10, label="GT apple centre")
    ax.plot(L[0][0][0], L[0][0][1], "o", mfc="none", color=C[3], ms=9, label="held primary centre")
    ax.set_aspect("equal"); ax.set_xlabel("base x (m)"); ax.set_ylabel("base y (m)")
    ax.set_title(f"{src} {run} t={t}: layout (top view, 250 mm around the hand)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left", markerscale=1, framealpha=0.8)
    # (2) slice
    ax = axes[i, 1]
    xs = np.linspace(L[0][0][0] - 0.16, L[0][0][0] + 0.16, 321); zs = np.linspace(off - 0.05, off + 0.16, 211)
    X, Z = np.meshgrid(xs, zs)
    D = np.asarray(field.distance(np.column_stack([X.ravel(), np.full(X.size, cy), Z.ravel()])), float).reshape(X.shape)
    im = ax.pcolormesh(X, (Z - off) * 1000, np.clip(D, 0, 0.08) * 1000, cmap=fs.sequential_cmap(), shading="auto")
    ax.contourf(X, (Z - off) * 1000, (D <= 1e-4).astype(float), levels=[0.5, 1.5], colors=[C[7]], alpha=0.35)
    ax.contour(X, (Z - off) * 1000, D, levels=[1e-4], colors=[C[7]], linewidths=0.8)
    plt.colorbar(im, ax=ax, label="combined field d (mm, clipped 0-80)")
    ax.axhline(0, color=fs.INK, lw=1.0, label="TLS table plane")
    ax.axhline((gt_table_z - off) * 1000, color=fs.INK, lw=1.0, ls=":", label="GT table top")
    band = (sup.get("band") or {}).get("band_q10_q50_q90_mm")
    if band:
        ax.axhline(band[2], color=C[3], lw=0.9, ls="-.", label=f"measured band q90 {band[2]:.2f} mm")
    th = np.linspace(0, 2 * np.pi, 200)
    rec_s = []
    for lab_, S_, ls_, col_ in (("fit + pad (T33 query = filter copy)", Fz, "--", C[5]), ("lifted (T34 query)", L, "-", C[3])):
        for k, (c_, r_) in enumerate(S_):
            ax.plot(c_[0] + r_ * np.cos(th), (c_[2] - off + r_ * np.sin(th)) * 1000, ls_, color=col_, lw=1.3, label=lab_ if k == 0 else None)
            rec_s.append({"set": lab_, "k": k, "centre_base_m": [round(float(v), 5) for v in c_], "radius_mm": round(1000 * r_, 3),
                          "bottom_above_plane_mm": round(1000 * float(n @ c_ - off - r_), 2), "slice_offset_y_mm": round(1000 * float(c_[1] - cy), 2)})
    ax.plot(apple[0] + 0.034 * np.cos(th), (apple[2] - off + 0.034 * np.sin(th)) * 1000, ":", color=C[2], lw=1.2, label="GT apple (r 34 mm at GT centre)")
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(-50, 160)
    ax.set_xlabel("base x (m)  [slice at base y = %.3f m]" % cy); ax.set_ylabel("height above TLS table plane (mm)")
    ax.set_title("x-z slice at attach chunk: d <= 0 filled (red), spheres at attach pose", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper right")
    # (3) profile
    ax = axes[i, 2]
    H = np.arange(-0.03, 0.045, 0.0005)
    prof = {}
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    for j, a_ in enumerate(ang):
        xy = L[0][0][:2] + 0.09 * np.array([np.cos(a_), np.sin(a_)])
        z0 = (off - n[0] * xy[0] - n[1] * xy[1]) / n[2]
        Pq = np.array([xy[0], xy[1], z0])[None] + H[:, None] * n[None]
        for nm, fl, col_ in (("fine", fine, C[0]), ("coarse", coarse, C[4]), ("combined", field, C[7])):
            v = np.asarray(fl.distance(Pq), float) * 1000
            prof.setdefault(nm, []).append([round(float(x), 3) for x in v])
            ax.plot(H * 1000, v, color=col_, lw=0.9, alpha=0.8, label=nm if j == 0 else None)
    ax.plot(H * 1000, H * 1000, ":", color=fs.INK_2, lw=0.8, label="d = h")
    ax.axvline(0, color=fs.INK, lw=0.8)
    ax.set_xlabel("h above TLS plane (mm)"); ax.set_ylabel("d (mm)"); ax.set_ylim(-5, 45)
    ax.set_title("d(h) at 6 table columns 90 mm around the held centre", fontsize=9); ax.legend(fontsize=7)
    fs.style_axes(fig, axes[i].tolist())
    side["cases"][case] = {"chunk": fpath, "frame_link": link, "plane_tls": {"normal": n.tolist(), "offset_m": off},
                           "gt_table_top_minus_plane_mm": round(1000 * float(gt_table_z - off), 2), "support_record": sup,
                           "spheres": rec_s, "gt_apple_centre_base_m": apple.tolist(), "slice_y_m": float(cy),
                           "profile_h_mm": (H * 1000).round(2).tolist(), "profile_d_mm": prof}
fig.suptitle("T34 attach moment: held query spheres (T33 fit+pad vs T34 lifted), table plane and the field's zero band", fontsize=11)
fig.tight_layout()
assert not OUTP.exists(), OUTP
fig.savefig(OUTP, dpi=120); plt.close(fig)
OUTP.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(OUTP)
