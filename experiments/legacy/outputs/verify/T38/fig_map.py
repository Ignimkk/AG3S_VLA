"""T38 figure (3) diagram: the server path per chunk as measured — stage boxes with median ms (pass T, quiet run _q2),
where the time runs (CPU / GPU), input sizes (census pass C) and the repeated work found by the census.
writes AG3S/ag3s/docs/figures/t38/t38-pipeline-map.png + .json
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-o4")
from AG3S.ag3s.experiments.common import figstyle

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T38'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t38'))
S = json.loads((OUT / "summary_E3b_ep1807_r2_q2.json").read_text())
T = S["pass_T"]; st = T["stages_profile_ms_seq2_75"]; tm = T["timing_seq2_75"]; fn = T["functions_seq2_75"]
C = {c["seq"]: c for c in S["census"]}
mi = {k: json.loads((OUT / "micro" / f"micro_kdtree_{k}.json").read_text()) for k in ("quiet_seq2", "quiet_seq22")}
med = lambda d: d["median"]
f = lambda k: fn.get(k, {}).get("median_ms", float("nan"))
c2, c22 = C[2], C[22]
CP = {k: next((r["cumtime_ms"] for r in S["cprofile"][k]["top_cumulative"] if "robot_sphere_mask" in r["function"]), float("nan"))
      for k in ("scene_reconstruction", "constraint_generation")}
figstyle.use_korean()
fig, ax = plt.subplots(figsize=(16, 11)); ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")
CPU, GPU = figstyle.SEQ_BLUE[1], figstyle.CATEGORICAL[2]
rows = [
    ("request", "3 cameras x 640x480 depth (uint16 mm) + 3 x 16x16 attention + q per camera", None, None,
     "921,600 depth px per chunk"),
    ("scene_reconstruction", f"per camera (python loop x3): backproject -> voxel 5 mm\n-> self-filter -> lift attention; then fuse",
     med(st["scene_reconstruction"]), CPU,
     f"backproject {c22['sizes']['n_points_raw']:,} pts -> voxel {c22['sizes']['n_points_voxel']:,} -> fused {c22['sizes']['n_points_fused']:,} (seq 22)\n"
     f"self-filter: 587 spheres, python loop, 1,761 single-point KD queries x3 cams; robot_sphere_mask {CP['scene_reconstruction']:.0f} ms cum (cProfile)"),
    ("support_surface", "RANSAC plane fit x2 on the fused cloud (numpy)", med(st["support_surface"]), CPU,
     f"fit_plane_ransac n_in {c22['fn_sizes']['support_surface.fit_plane_ransac'][0]['n_in']:,} / {c22['fn_sizes']['support_surface.fit_plane_ransac'][1]['n_in']:,}"),
    ("target_grounding", "seeds -> grow_region (KD waves) -> dbscan\n-> split -> score -> confirm",
     med(st["target_grounding"]), CPU,
     f"grow_region {f('target_grounding.grow_region'):.0f} ms on {c22['fn_sizes']['target_grounding.grow_region'][0]['n_in']:,} pts, "
     f"dbscan {f('target_grounding.dbscan'):.0f} ms on {c22['fn_sizes']['target_grounding.dbscan'][0]['n_in']:,} pts\n"
     f"cKDTree workers=-1 on 256 cores: about 2,138 threads started per chunk (cProfile). micro: grow_region "
     f"{mi['quiet_seq22']['grow_region_code_workers-1_ms']:.0f} ms (workers=-1) vs {mi['quiet_seq22']['grow_copy_workers1_ms']:.0f} ms (workers=1), same result"),
    ("constraint_generation\n(minus esdf)", "_depth_cameras_from: _robot_mask_for x3 =\nbackproject again + self-filter again (full-res)",
     med(st["constraint_generation_minus_esdf"]), CPU,
     f"_robot_mask_for {f('AG3S._robot_mask_for'):.0f} ms: same depth back-projected a 2nd time (6 backproject calls / chunk),\n"
     f"self-filter on {c22['sizes']['n_points_raw']:,} un-voxelised pts (x2 per camera while held: held-sphere mask on the same points);"
     f" ConstraintBuilder.build {f('ConstraintBuilder.build'):.1f} ms"),
    ("esdf (cuRobo)", "integrate depth (coarse + fine TSDF), ESDF tiers,\nlabels, target-free fine tier",
     med(st["esdf"]), GPU,
     f"GPU kernels ~6-7 ms + memcpy ~3-4 ms per chunk (torch.profiler); rest is CPU: _labels_from_sites {f('CuroboFieldBuilder._labels_from_sites'):.0f}, "
     f"_build_tiers {f('CuroboFieldBuilder._build_tiers'):.0f}, _integrate {f('CuroboFieldBuilder._integrate'):.0f} ms\n"
     f"H2D {c2['h2d_bytes'] / 1e6:.1f} MB (each depth uploaded twice: coarse + fine mapper), D2H {c2['d2h_bytes'] / 1e6:.1f} MB (3 tiers x (values + site index), 128^3 each)"),
    ("TO (trajopt)", "SQP 3 iterations: linearize / assemble / qp (osqp) / check",
     med(tm["trajopt"]), CPU,
     f"linearize {med(tm['to_linearize']):.0f}, check {med(tm['to_check']):.0f}, assemble {med(tm['to_assemble']):.0f}, qp {med(tm['to_qp']):.0f} ms;"
     f" cKDTree on manipulated points rebuilt 9x / chunk (same points) before attach"),
    ("recorder (_record)", "--record-constraints npz writer, after timing_ms.total", f("SafePolicy._record"), figstyle.GRID_INK,
     "not in profile_ms / timing_ms; in wall time"),
]
y = 10.3; data = []
for name, what, ms, col, note in rows:
    h = 1.05
    if col is not None:
        ax.add_patch(FancyBboxPatch((0.2, y - h), 3.4, h - 0.12, boxstyle="round,pad=0.02", fc=col, ec=figstyle.INK_2, lw=0.6))
        w = min(ms / 2600.0 * 3.0, 3.0)
        ax.add_patch(FancyBboxPatch((3.8, y - h + 0.3), w, 0.36, boxstyle="square,pad=0", fc=figstyle.CATEGORICAL[0], ec="none"))
        ax.text(3.85 + w, y - h + 0.48, f"{ms:.0f} ms", fontsize=8, va="center", color=figstyle.INK)
    else:
        ax.add_patch(FancyBboxPatch((0.2, y - h), 3.4, h - 0.12, boxstyle="round,pad=0.02", fc="#ffffff", ec=figstyle.INK_2, lw=0.6))
    ax.text(0.3, y - 0.3, name, fontsize=9, va="center", color=figstyle.INK)
    ax.text(0.3, y - 0.68, what, fontsize=6, va="center", color=figstyle.INK_2, wrap=True)
    ax.text(7.6, y - 0.45, note, fontsize=7, va="center", color=figstyle.INK)
    data.append({"stage": name, "what": what, "median_ms": ms, "note": note})
    y -= h + 0.08
ax.text(0.2, 10.75, f"T38 AG3S server path per chunk (replay E3b_ep1807_r2 seq 2-75, worktree pre-gpu-parallel). AG3S + TO median "
        f"{med(tm['ag3s_plus_trajopt']):.0f} ms; chunk period 533 ms. light blue = CPU, green = GPU stage (time mostly CPU, see note)",
        fontsize=9, color=figstyle.INK)
fig.patch.set_facecolor(figstyle.SURFACE)
fig.savefig(FIG / "t38-pipeline-map.png", dpi=130, bbox_inches="tight", facecolor=figstyle.SURFACE)
(FIG / "t38-pipeline-map.json").write_text(json.dumps({"rows": data, "micro": mi, "source": "outputs/verify/T38/summary_E3b_ep1807_r2_q2.json"}, indent=1, default=float))
print("ok")
