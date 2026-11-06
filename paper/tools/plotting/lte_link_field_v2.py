"""LTE figure v2: panels (a)-(c) of paper-lte-link-field.png kept, panel (d) redrawn with the per-voxel diagnosis,
plus the diagnosis views for outputs/ (real scene, graphs, table).

Inputs (read only):
  - per-voxel diagnosis written by paper/tools/analysis/lte_removed_voxels.py (same input and class rule as
    AG3S-CORE-EVAL/module_eval.py; class counts reproduce the core evaluation exactly)
  - AG3S/ag3s/docs/figures/core-eval/paper/paper-lte-link-field.png (panels a-c)
  - AG3S-CORE-EVALUATION.verify.json (apple_left_frac_median)

usage: MUJOCO_GL=osmesa .venv-ag3s/bin/python paper/tools/plotting/lte_link_field_v2.py DIAG_DIR OUTPUTS_DIR
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "paper/tools/analysis"))
import lte_removed_voxels as LRV  # noqa: E402  (also puts the core snapshot on sys.path)

FIGDIR = REPO / "AG3S/ag3s/docs/figures/core-eval/paper"
OLD = FIGDIR / "paper-lte-link-field.png"
VERIFY = REPO / "AG3S/ag3s/docs/handoff/AG3S-CORE-EVALUATION.verify.json"
NAME = "paper-lte-link-field-v2"
EXAMPLE = ("회귀954_k1", 23)   # the request shown in panels (a)-(c)

COL = {"apple": "#E8711A", "apple_inside": "#F4B183", "apple_stem_leaf": "#8FBF5A", "none": "#17BECF",
       "robot": "#F2C80F", "fruit": "#2E75B6", "obstacle": "#C00000", "table": "#C00000", "crate": "#C00000"}


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def pct(n, tot):
    return 100.0 * n / tot


def example_voxels():
    """Recompute the removed voxels of the example request with the same rule (for the real-scene view)."""
    spec = {r["label"]: r for r in json.loads((LRV.CE / "module/spec_core.json").read_text())}
    r = spec[EXAMPLE[0]]
    gt = LRV.tde.scene_truth(LRV.XML, (r["obstacle"][0], r["obstacle"][1]))
    rob = LRV.tde.RobotSurface(gt)
    m = gt.m
    ab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    ai = [i for i, (k, n, b) in enumerate(gt.kind) if n == "apple"]
    apple_r = float(max(m.geom_size[g].max() for g in range(m.ngeom) if int(m.geom_bodyid[g]) == ab and m.geom_contype[g] != 0))
    apple_vis = [(g, gt._sample_local(g)) for g in range(m.ngeom) if int(m.geom_bodyid[g]) == ab
                 and m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and float(m.geom_size[g].max()) < 0.02]
    steps = sorted(pathlib.Path(r["client_dir"]).glob("ag3s/run_0000/step_*.npz"), key=lambda p: int(np.load(p)["t_step"]))
    chunks = r["chunks"] if isinstance(r["chunks"], list) else eval(r["chunks"])  # noqa: S307
    j = EXAMPLE[1]
    z = np.load(chunks[j], allow_pickle=True)
    qpos = np.asarray(np.load(steps[j])["qpos"], float)
    samples, _ = gt.at(qpos)
    R, o = gt.base_pose
    ac = (gt.d.xpos[ab] - o) @ R
    vs = float(z["target_free_trace_voxel_size"])
    P = np.asarray(z["target_free_trace_origin"], float) + (np.asarray(z["target_free_removed_ijk"], float) + .5) * vs
    vis = [((loc @ gt.d.geom_xmat[g].reshape(3, 3).T + gt.d.geom_xpos[g]) - o) @ R for g, loc in apple_vis]
    rp = rob.at()
    allp = np.vstack([p for p in samples if len(p)] + [rp] + vis)
    cls = np.concatenate([[("apple" if gt.kind[i][1] == "apple" else gt.kind[i][0])] * len(p) for i, p in enumerate(samples) if len(p)]
                         + [["robot"] * len(rp)] + [["apple_stem_leaf"] * len(v) for v in vis])
    dd, ii = cKDTree(allp).query(P)
    c = np.where(dd * 1e3 <= LRV.CLASS_MM, cls[ii], "none")
    c = np.where((c == "none") & (np.linalg.norm(P - ac, axis=1) <= apple_r), "apple_inside", c)
    Pw = P @ R.T + o   # base frame -> world frame for rendering
    acw = ac @ R.T + o
    return gt, qpos, Pw, c, vs, acw, [steps[j], chunks[j]]


def render_example(out: pathlib.Path):
    gt, qpos, Pw, c, vs, acw, src = example_voxels()
    m, d = gt.m, mujoco.MjData(gt.m)
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    rgba = m.geom_rgba.copy()
    ab = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    m.geom_rgba[np.where(m.geom_bodyid == ab)[0], 3] = 0.25
    m.vis.global_.offwidth, m.vis.global_.offheight = 1600, 1200
    rr = mujoco.Renderer(m, 1200, 1600, max_geom=20000)
    imgs = []
    for az in (150, 240):
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = acw, 0.30, az, -25
        rr.update_scene(d, cam)
        sc = rr.scene
        for p, k in zip(Pw, c):
            if sc.ngeom >= sc.maxgeom:
                break
            col = np.array(matplotlib.colors.to_rgba(COL.get(str(k), "#C00000")), np.float32)
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([vs / 2] * 3) * 0.92, p, np.eye(3).ravel(), col)
            sc.ngeom += 1
        imgs.append(rr.render().copy())
    rr.close()
    m.geom_rgba[:] = rgba
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    for ax, im, t in zip(axs, imgs, ("front view", "side view")):
        ax.imshow(im); ax.set_axis_off(); ax.set_title(t, fontsize=10)
    names = {"apple": "apple surface (within 8 mm)", "apple_inside": "inside apple radius", "apple_stem_leaf": "stem / leaf",
             "none": "no surface within 8 mm", "robot": "hand surface (within 8 mm)", "fruit": "neighbour fruit"}
    cnt = collections.Counter(c.tolist())
    h = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in names if cnt.get(k)]
    fig.legend(h, [f"{names[k]}: {cnt[k]}" for k in names if cnt.get(k)], ncol=3, fontsize=8, loc="lower center", frameon=False)
    fig.suptitle(f"Removed voxels of regression scene 954 (draw 1), request {EXAMPLE[1]}, by ground-truth class (apple drawn transparent)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(out / "1_실제씬_지운voxel_class별_954k1_요청23.png")
    plt.close(fig)
    return {k: int(v) for k, v in cnt.items()}, [str(s) for s in src]


def main(diag: pathlib.Path, outdir: pathlib.Path):
    outdir.mkdir(parents=True, exist_ok=True)
    V = json.loads((diag / "lte_removed_voxels.json").read_text())
    tot = len(V)
    cnt = collections.Counter(v["cls"] for v in V)
    none = [v for v in V if v["cls"] == "none"]
    rob = [v for v in V if v["cls"] == "robot"]
    a = lambda vs, k: np.array([v[k] for v in vs], float)  # noqa: E731
    left_med = json.loads(VERIFY.read_text())["numbers"]["LTE"]["apple_left_frac_median"]
    S = {
        "n_removed_total": tot, "class_counts": dict(cnt), "class_percent": {k: pct(v, tot) for k, v in cnt.items()},
        "target_apple_percent": pct(cnt["apple"] + cnt["apple_inside"] + cnt["apple_stem_leaf"], tot),
        "apple_surface_removed_percent_median": 100 * (1 - left_med),
        "none": {"d_apple_surface_mm_pct5_25_50_75_95": np.percentile(a(none, "d_apple_surf_mm"), [5, 25, 50, 75, 95]).tolist(),
                 "frac_d_apple_le_12": float(np.mean(a(none, "d_apple_surf_mm") <= 12)),
                 "frac_d_apple_le_20": float(np.mean(a(none, "d_apple_surf_mm") <= 20)),
                 "nearest_class_beyond_8mm": dict(collections.Counter(v["nearest_any"] for v in none)),
                 "in_served_target_ball": float(np.mean([v["in_ball"] for v in none])),
                 "per_run": dict(collections.Counter(v["run"] for v in none))},
        "robot": {"bodies": dict(collections.Counter(v["robot_body"] for v in rob)),
                  "d_robot_mm_pct5_50_95": np.percentile(a(rob, "d_robot_mm"), [5, 50, 95]).tolist(),
                  "frac_also_within_8mm_of_apple": float(np.mean(a(rob, "d_apple_surf_mm") <= 8)),
                  "frac_within_12mm_of_apple": float(np.mean(a(rob, "d_apple_surf_mm") <= 12)),
                  "per_run": dict(collections.Counter(v["run"] for v in rob))},
    }

    # ---- outputs/: graphs
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), dpi=150)
    for vs, lab, colr in ((none, "no surface within 8 mm", COL["none"]), (rob, "hand surface", COL["robot"])):
        axs[0].hist(np.clip(a(vs, "d_apple_surf_mm"), 0, 50), bins=50, range=(0, 50), alpha=0.7, color=colr, label=lab)
    axs[0].axvline(8, color="k", ls="--", lw=0.8); axs[0].set_xlabel("distance to the nearest apple surface sample (mm, clipped at 50)")
    axs[0].set_ylabel("removed voxels"); axs[0].legend(fontsize=8); axs[0].set_title("Where the two classes lie relative to the apple")
    nc = collections.Counter(v["nearest_any"] for v in none).most_common()
    axs[1].barh([k for k, _ in nc][::-1], [pct(x, len(none)) for _, x in nc][::-1], color=COL["none"])
    axs[1].set_xlabel("% of 'no surface within 8 mm' voxels"); axs[1].set_title("Nearest object beyond 8 mm")
    rb = collections.Counter(v["robot_body"] for v in rob).most_common()
    axs[2].barh([k for k, _ in rb][::-1], [pct(x, len(rob)) for _, x in rb][::-1], color=COL["robot"])
    axs[2].set_xlabel("% of 'hand surface' voxels"); axs[2].set_title("Nearest robot body (within 8 mm)")
    fig.tight_layout(); fig.savefig(outdir / "2_그래프_표면없음_손_위치분포.png"); plt.close(fig)

    ex_counts, ex_src = render_example(outdir)
    S["example"] = {"run": EXAMPLE[0], "request": EXAMPLE[1], "class_counts": ex_counts, "sources": ex_src}

    # ---- paper figure v2: (a)-(c) kept, (d) redrawn
    old = Image.open(OLD).convert("RGB")
    left = old.crop((0, 0, 1180, old.height))
    dpi = 300
    W, H = 1030, old.height
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0.42, 0.27, 0.54, 0.56])
    rows = [("Target apple", [("apple", "surface"), ("apple_inside", "interior band"), ("apple_stem_leaf", "stem / leaf")]),
            ("No surface within 8 mm", [("none", None)]),
            ("Grasping-hand surface", [("robot", None)]),
            ("Adjacent fruit", [("fruit", None)]),
            ("Obstacle / table / basket", [("obstacle", None)])]
    for yi, (lab, parts) in enumerate(rows[::-1]):
        x0 = 0.0
        for k, sub in parts:
            v = pct(cnt.get(k, 0), tot)
            ax.barh(yi, v, left=x0, color=COL[k], height=0.62, edgecolor="white", lw=0.4)
            x0 += v
        ax.text(x0 + 1.2, yi, f"{x0:.1f}%", va="center", fontsize=6.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6.5)
    ax.set_xlim(0, 100); ax.set_xlabel("% of removed voxels (18 runs)", fontsize=6.5, labelpad=1)
    ax.tick_params(axis="x", labelsize=6, length=2, pad=1); ax.tick_params(axis="y", length=0)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    hd = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("apple", "apple_inside", "apple_stem_leaf")]
    ax.legend(hd, ["surface", "interior band", "stem / leaf"], fontsize=5.5, frameon=False, loc="lower right", handlelength=0.9)
    fig.text(0.5, 0.94, "(d) Removed voxels by ground-truth class", ha="center", fontsize=8)
    fig.text(0.66, 0.035, "Observed apple surface removed from the\ncontact-link field: "
             f"{S['apple_surface_removed_percent_median']:.1f}% (median, 18 runs)", fontsize=5.8, ha="center")
    fig.canvas.draw()
    dimg = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]); plt.close(fig)
    comp = Image.new("RGB", (left.width + dimg.width, H), "white")
    comp.paste(left, (0, 0)); comp.paste(dimg, (left.width, 0))
    png, pdf = FIGDIR / f"{NAME}.png", FIGDIR / f"{NAME}.pdf"
    for p in (png, pdf):
        if p.exists():
            raise SystemExit(f"{p} exists; pick a new name instead of overwriting")
    comp.save(png, dpi=(dpi, dpi)); comp.save(pdf, resolution=dpi)
    comp.save(outdir / "3_논문그림_LTE_v2.png")
    side = {"asset": str(png.relative_to(REPO)), "script": "paper/tools/plotting/lte_link_field_v2.py",
            "panels_a_c": {"source": str(OLD.relative_to(REPO)), "sha256": sha(OLD), "crop_px": [0, 0, 1180, old.height]},
            "panel_d": "redrawn from the per-voxel diagnosis; class rule unchanged (nearest GT sample within 8 mm)",
            "diagnosis": {"dir": str(diag), "sha256_voxels": sha(diag / "lte_removed_voxels.json")},
            "printed_numbers": {r[0]: sum(pct(cnt.get(k, 0), tot) for k, _ in r[1]) for r in rows}
            | {"apple_surface_removed_median": S["apple_surface_removed_percent_median"]},
            "summary": S, "is_measured_result": True}
    (FIGDIR / f"{NAME}.json").write_text(json.dumps(side, ensure_ascii=False, indent=1, default=float))
    (outdir / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1, default=float))
    print(json.dumps({k: S[k] for k in ("class_percent", "target_apple_percent", "apple_surface_removed_percent_median")}, default=float))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
