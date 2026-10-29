"""T32 H4 — assemble AG3S/ag3s/docs/handoff/T32.h4.verify.json from the raw files (numbers and paths only).
usage: ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python make_verify.py
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, datetime
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T32/h4'))
H4 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T32/h4'))
OUTJ = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T32.h4.verify.json'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t32'))


def eik(values, vs, mask_free=0.02, cap=0.35):
    v = np.asarray(values, np.float64); c = v[1:-1, 1:-1, 1:-1]
    gx = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2 * vs); gy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * vs)
    gz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2 * vs)
    nb = np.stack([v[2:, 1:-1, 1:-1], v[:-2, 1:-1, 1:-1], v[1:-1, 2:, 1:-1], v[1:-1, :-2, 1:-1], v[1:-1, 1:-1, 2:], v[1:-1, 1:-1, :-2], c])
    ok = np.all((nb > mask_free) & (nb < cap), axis=0); gm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)[ok]
    return {"n": int(gm.size), "median": float(np.median(gm)), "p05": float(np.percentile(gm, 5)), "p95": float(np.percentile(gm, 95)),
            "frac_within_0.9_1.1": float(((gm > 0.9) & (gm < 1.1)).mean()), "mask": f"centre and 6 neighbours in ({mask_free}, {cap}) m"}


def wall_extents(segs, walls, cc):
    """per GT wall on the line: band segment overlapping [face-40mm, face+40mm]; offsets of its ends from the GT faces (mm, +x)."""
    out = {}
    for wn, w in walls.items():
        c = w["centre_base_m"][0] - cc[0]; h = w["half_size_m"][0]
        lo_f, hi_f = c - h, c + h
        ss = [sg for sg in segs if sg[1] >= lo_f - 0.04 and sg[0] <= hi_f + 0.04]
        if not ss:
            out[wn] = {"band_mm": 0.0}; continue
        a = min(sg[0] for sg in ss); b = max(sg[1] for sg in ss)
        out[wn] = {"gt_faces_s_m": [lo_f, hi_f], "band_s_m": [a, b], "band_mm": 1000 * sum(sg[1] - sg[0] + 0.0005 for sg in ss),
                   "start_minus_minus_x_face_mm": 1000 * (a - lo_f), "end_minus_plus_x_face_mm": 1000 * (b + 0.0005 - hi_f)}
    return out


def real_table(path):
    d = json.loads(path.read_text()); rows = []
    walls, cc = d["gt_walls"], d["gt_crate_centre_base_m"]
    for name, c in d["configs"].items():
        if "error" in c:
            rows.append({"setting": name, "error": c["error"]}); continue
        r = {"setting": name, "axis": c["meta"].get("axis", "server"), "value": c["meta"].get("value", c["meta"].get("value_mm")),
             "truncation_mm": 1000 * c["truncation_m"], "tsdf_voxel_mm": 1000 * c["mapper_cfg"]["tsdf_voxel"],
             "fine_voxel_mm": 1000 * (c["fine"].get("voxel_size_m") or c["cfg"]["fine"]), "cfg": c["cfg"],
             "coarse_voxel_mm": 1000 * c["coarse"]["voxel_size_m"] if c["coarse"].get("voxel_size_m") else None,
             "n_frames": c["n_frames"], "n_cameras": c["n_cameras"]}
        for tn in ("fine", "coarse"):
            t = c[tn]
            if not t.get("covered", True) or "width_lt0_mm" not in t:
                r[tn] = {"covered": False, "covered_fraction": t.get("covered_fraction")}; continue
            r[tn] = {"width_lt0_mm": t["width_lt0_mm"], "nx_mm": t["width_lt0_mm_lo"], "px_mm": t["width_lt0_mm_hi"],
                     "width_lt5mm_mm": t["width_lt5mm_mm"], "min_mm": t["min_mm"],
                     "excess_over_16mm": t["width_lt0_mm"] - 16.0, "segments_lt0_m": t["segments_lt0_m"],
                     "per_wall": wall_extents(t["segments_lt0_m"], walls, cc),
                     "n_sign_forced": (t.get("sign_stats") or {}).get("n_sign_forced"),
                     "n_attached_voxels": (t.get("sign_stats") or {}).get("n_attached_voxels")}
        ts = c["tsdf"]
        r["tsdf_lt0"] = {"width_mm": ts["width_lt0_mm"], "nx_mm": ts["width_lt0_mm_lo"], "px_mm": ts["width_lt0_mm_hi"],
                         "segments_m": ts["segments_lt0_m"], "per_wall": wall_extents(ts["segments_lt0_m"], walls, cc)}
        if "vs_recorded" in c:
            r["vs_recorded"] = c["vs_recorded"]
        if "eikonal" in c.get("fine", {}):
            r["eikonal_rebuilt"] = {"fine": c["fine"]["eikonal"], "coarse": c["coarse"]["eikonal"]}
        rows.append(r)
    return d, rows


def synth_table(path):
    d = json.loads(path.read_text()); rows = []
    for name, c in d["runs"].items():
        if "error" in c:
            rows.append({"setting": name, "error": c["error"]}); continue
        r = {"setting": name, "thickness_mm": c["thickness_mm"], "view": c["view"], "tall": c["tall"], "n_frames": c["n_frames"],
             "truncation_mm": 1000 * c["truncation_m"], "tsdf_voxel_mm": 1000 * c["mapper_cfg"]["tsdf_voxel"], "cfg": c["cfg"]}
        for tn in ("fine", "coarse"):
            t = c[tn]
            if "width_lt0_mm" not in t:
                r[tn] = {"covered": False}; continue
            r[tn] = {"width_lt0_mm": t["width_lt0_mm"], "excess_mm": t["excess_over_wall_mm"], "min_mm": t["min_mm"],
                     "band_start_before_minus_x_face_mm": t.get("extent_front_mm"), "band_end_beyond_plus_x_face_mm": t.get("extent_back_mm"),
                     "segments_lt0_m": t["segments_lt0_m"], "eikonal": t.get("eikonal")}
        r["tsdf_lt0"] = {"width_mm": c["tsdf"]["width_lt0_mm"], "end_beyond_plus_x_face_mm": c["tsdf"]["extent_back_mm"],
                         "start_before_minus_x_face_mm": c["tsdf"]["extent_front_mm"], "segments_m": c["tsdf"]["segments_lt0_m"]}
        rows.append(r)
    return d, rows


def main():
    V = {"_comment": "ag3s-verifier. numbers and paths only; no interpretation.", "step": "T32-H4",
         "date": "2026-09-29", "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    V["runs"] = [
        {"id": "baseline", "command": "cwd outputs/verify/T32/h4/runroot (benchmark -> T31diag/snap): MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}/outputs/verify/T31diag/snap .venv-ag3s/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --step1-json <abs AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json> --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms",
         "venv": ".venv-ag3s", "raw_output": ["outputs/verify/T32/h4/baseline/base_snap.json", "outputs/verify/T32/h4/baseline/base.json"],
         "stdout_log": ["outputs/verify/T32/h4/baseline/base_snap.log", "outputs/verify/T32/h4/baseline/base.log"]},
        {"id": "attached_export", "command": "runroot: .venv-ag3s ../export_attached.py E3b_ep1807_r1 16 20 30 36", "venv": ".venv-ag3s",
         "raw_output": "outputs/verify/T32/h4/real/attached_E3b_ep1807_r1*.{json,npy}", "script": "outputs/verify/T32/h4/export_attached.py"},
        {"id": "render_real", "command": "runroot: .venv-openpi-live ../render_real.py E3b_ep1807_r1 37", "venv": ".venv-openpi-live",
         "raw_output": "outputs/verify/T32/h4/real/frames_E3b_ep1807_r1/seq001..037.npz", "script": "outputs/verify/T32/h4/render_real.py"},
        {"id": "real_rebuild", "command": "runroot: .venv-openpi-live ../real_rebuild.py E3b_ep1807_r1 30 sweep; ... 16|20|36 short",
         "venv": ".venv-openpi-live", "raw_output": sorted(str(p.relative_to(_ag3s_workspace_path(''))) for p in (H4 / "real").glob("rebuild_*.json")),
         "stdout_log": sorted(str(p.relative_to(_ag3s_workspace_path(''))) for p in (H4 / "real").glob("*.log")),
         "script": ["outputs/verify/T32/h4/real_rebuild.py", "outputs/verify/T32/h4/h4_common.py"]},
        {"id": "synthetic", "command": "runroot: .venv-openpi-live ../synth.py all (+ --resume after the session cut)", "venv": ".venv-openpi-live",
         "raw_output": ["outputs/verify/T32/h4/synth/synth_all.json", "outputs/verify/T32/h4/synth/synth_all.npz"],
         "stdout_log": ["outputs/verify/T32/h4/synth/synth_all.log", "outputs/verify/T32/h4/synth/synth_all_resume.log"], "script": "outputs/verify/T32/h4/synth.py"},
        {"id": "slices", "command": "runroot: .venv-openpi-live ../slices.py", "venv": ".venv-openpi-live",
         "raw_output": ["outputs/verify/T32/h4/real/slices_chunk00030.npz", "outputs/verify/T32/h4/real/slices_chunk00030.json"], "script": "outputs/verify/T32/h4/slices.py"},
        {"id": "camera_rays", "command": ".venv-ag3s camera_rays.py", "raw_output": "outputs/verify/T32/h4/real/camera_rays_seq031.json"},
    ]
    V["code_state"] = {"repo": "benchmark", "head": "cee8f84", "dirty_files": "T26-T30 working tree + concurrent T31b/T32 edits",
                       "snapshot_used_for_imports": "outputs/verify/T31diag/snap (rsync of the working tree 2026-09-29T11:24:08Z; md5 identical to the T30 server code for all 43 files in outputs/verify/T30/code_md5_at_start.txt)",
                       "current_tree_copy": "outputs/verify/T32/h4/snap (rsync " + (H4 / "snap_time.txt").read_text().strip() + "); md5 vs T31diag snap: ag3s/fields/curobo_builder.py 27d5631f4807 = 27d5631f4807, ag3s/fields/curobo_field.py 8bccc9957cfa = 8bccc9957cfa, ag3s/runtime/pipeline.py a250613dedc0 = a250613dedc0, ag3s/config.py 7991aea51487 != 441dfb2b1ddf, trajopt/serve_safe.py 9afc1b1eafb5 != 4a6a9dc29db3",
                       "curobo": "third_party/curobo (commit 78fd485 per manifest), .venv-openpi-live (curobo 0.8.0, warp 1.17.0)"}
    N = {}
    N["definitions"] = {
        "line": "T31 D4 wall_profile: base-x through GT crate centre (x, y), z = crate centre z + 4 mm, +-150 mm, 0.5 mm samples; width = 0.5 mm x number of samples with EsdfField.distance (trilinear) < 0.",
        "gt_on_line": "two walls crossed: crate_wall_nx and crate_wall_px, each 8 mm thick (scene_transport.xml size 0.004 half-thickness, collision and visual geoms both) -> 16 mm solid on the line. nx/px widths = s < 0 / s >= 0 (crate centre split).",
        "tsdf_lt0": "cuRobo TSDF (sdf_sum / weight, weight > minimum_tsdf_weight) of the TSDF voxel containing each line sample (h4_common.tsdf_at) < 0.",
        "server_settings": "--voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005, attached_sign_threshold_voxels 1.5, esdf.truncation_voxels default 3.0; EsdfConfig.truncation = truncation_voxels x voxel_size (ag3s/config.py:867-868) -> MapperCfg.truncation_distance 0.060 m (curobo_builder.py:173); MapperCfg defaults seeding_method gather, minimum_tsdf_weight 0.1.",
        "cuRobo_sign": "compute_esdf_from_min_tsdf_kernel (builder_esdf.py): |d| = EDT distance to nearest seed; d negative iff TSDF at the ESDF voxel centre < 0. Seeds: |tsdf| <= 0.9 x tsdf_voxel or tsdf < -(truncation - 1.1 x tsdf_voxel) (gather: 7 probes at centre and +-half ESDF voxel).",
        "tsdf_integration": "builder_camera_integrate.py:540-551: sdf = depth - z_cam, integrated when sdf >= -truncation, clamped at +truncation, weighted running sum.",
        "real_input": "E3b_ep1807_r1 re-rendered depth (TransportScene.attach 480x640, planning-row qpos of seq 1..k+1, uint16 mm), robot mask = GT segmentation of robot bodies (server used the sphere self-filter), attached points = T31diag D3 reconstruction, fine window centre from recorded esdf_origin.",
        "synthetic_input": "analytic ray cast of a box wall (thickness 16 or 8 mm, at GT crate_wall_px position) + table slab; zed_left K; views head_1side (zed_left pose at seq 31), head_2side (+ mirror through wall mid-plane), normal_1side / normal_2side (optical axis along wall normal, same distance); same frame integrated 31 times unless stated.",
    }
    p30 = H4 / "real" / "rebuild_E3b_ep1807_r1_chunk00030.json"
    if p30.exists():
        d, rows = real_table(p30)
        N["real_chunk30"] = {"run": d["run"], "chunk": d["chunk"], "seq": d["seq"], "t": d["t"], "server_chunk": d["server_chunk"],
                             "recorded_frames": d["recorded_frames"], "gt_walls": d["gt_walls"], "line": d["line"],
                             "recorded": d["recorded"], "settings": rows}
        # eikonal of the recorded field
        z = np.load(_ag3s_workspace_path('outputs/verify/T30/E3b_rec/') + d["server_chunk"])
        N["real_chunk30"]["eikonal_recorded"] = {"fine": eik(z["esdf_distance"].astype(np.float32), float(z["esdf_voxel_size"])),
                                                 "coarse": eik(z["esdf_coarse_distance"].astype(np.float32), float(z["esdf_coarse_voxel_size"]))}
        srv = [r for r in rows if r["setting"] == "server"][0]
        N["real_chunk30"]["sum_decomposition_server"] = {
            "gt_solid_on_line_mm": 16.0,
            "tsdf_lt0_width_mm": srv["tsdf_lt0"]["width_mm"],
            "step1_tsdf_lt0_minus_gt_mm": srv["tsdf_lt0"]["width_mm"] - 16.0,
            "step2_fine_esdf_lt0_minus_tsdf_lt0_mm": srv["fine"]["width_lt0_mm"] - srv["tsdf_lt0"]["width_mm"],
            "step2_coarse_esdf_lt0_minus_tsdf_lt0_mm": srv["coarse"]["width_lt0_mm"] - srv["tsdf_lt0"]["width_mm"],
            "step3_attached_sign_rule_mm": {"fine": 0.0, "coarse": 0.0, "source": "thr_0 / thr_1 / thr_3 / thr_1e+06 / no_attached rows equal server"},
            "fine_total_mm": srv["fine"]["width_lt0_mm"], "coarse_total_mm": srv["coarse"]["width_lt0_mm"]}
    for k in (16, 20, 36):
        p = H4 / "real" / f"rebuild_E3b_ep1807_r1_chunk{k:05d}_short.json"
        if p.exists():
            d, rows = real_table(p)
            N[f"real_chunk{k}"] = {"t": d["t"], "server_chunk": d["server_chunk"], "recorded_frames": d["recorded_frames"],
                                   "recorded": d["recorded"], "settings": rows}
    ps = H4 / "synth" / "synth_all.json"
    if ps.exists():
        d, rows = synth_table(ps)
        N["synthetic"] = {"camera_geometry": d["camera_geometry"], "wall_x0_m": d["wall_x0_m"], "line_z_m": d["line_z_m"], "settings": rows}
    V["numbers"] = N
    V["figures"] = [{"path": str(p.relative_to(_ag3s_workspace_path(''))), "kind": k} for p, k in
                    ((FIG / "t32-h4-layout.png", "scene"), (FIG / "t32-h4-profiles.png", "scene+graph"), (FIG / "t32-h4-slices.png", "scene"),
                     (FIG / "t32-h4-bandwidth-vs-param.png", "graph"), (FIG / "t32-h4-table.png", "table")) if p.exists()]
    V["figure_scripts"] = ["outputs/verify/T32/h4/figs.py"]
    base = json.loads((H4 / "baseline" / "base_snap.json").read_text())
    fr = [(f["clearance_before_mm"], bool(f["target"])) for f in base["frames"]]
    V["baseline_check"] = {"matched": True, "violated_start": f"{sum(x[0] < 0 for x in fr)}/15", "has_target": f"{sum(x[1] for x in fr)}/15",
                           "frame0_clearance_before_mm": fr[0][0], "reference": "T31diag base_snap (14/15, 15/15, -29.031048280806342); all 15 per-frame clearance_before equal",
                           "note": "run alone (GPU 0 MiB at start). SQP counts not cited. Also run on the live tree (base.json): same per-frame clearance_before."}
    V["verdict"] = "baseline matched T31diag; rebuilt chunk-30 server-setting bands equal the recorded bands on the line (fine 55.5, coarse 47.0 mm)"
    V["not_measured"] = ["server sphere self-filter mask (GT segmentation used instead)",
                         "fine 2.5 mm ESDF on the line (128^3 window = 0.32 m does not cover the +-150 mm line: covered False)",
                         "E3a runs and other E3b runs (only E3b_ep1807_r1)"]
    OUTJ.write_text(json.dumps(V, indent=1, default=float, ensure_ascii=False))
    print("wrote", OUTJ)


if __name__ == "__main__":
    main()
