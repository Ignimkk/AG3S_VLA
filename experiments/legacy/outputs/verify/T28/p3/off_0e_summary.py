"""Summarise off_0e_raw_{t14,t17}.jsonl (T29 cover model, inflation 0, measured finger joints). pre-grasp = rows before the
first row with GT apple dz >= 50 mm. 'apple whole-deleted frame' = a (row, camera) with apple px visible > 0 and 0 passed."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np
out = {"definitions": __doc__}
old = json.load(open(_ag3s_workspace_path('outputs/verify/T28/P1-3/p13b_selffilter_summary.json')))
for run in ("t14", "t17"):
    R = [json.loads(l) for l in open(f"{_ag3s_workspace_path()}/outputs/verify/T28/p3/off_0e_raw_{run}.jsonl")]
    seqs = sorted({r["seq"] for r in R})
    dz = {r["seq"]: r["apple_dz_mm"] for r in R}
    g = next((s for s in seqs if dz[s] >= 50), None)
    pre = lambda r: g is None or r["seq"] < g
    body = {}
    for r in R:
        for b, (v, p) in r["robot_by_body_visible_passed"].items():
            x = body.setdefault(b, [0, 0, 0]); x[0] += v; x[1] += p; x[2] += int(p > 0)
    leak = {b: v for b, v in sorted(body.items(), key=lambda kv: -kv[1][1]) if v[1] > 0}
    whole = [(r["seq"], r["t"], r["camera"], r["apple_px_visible"]) for r in R if r["apple_px_visible"] > 0 and r["apple_px_passed"] == 0]
    frame_whole = sorted({s for s, t, c, n in whole})
    # frame-level: apple visible in >=1 camera but 0 passed in all cameras
    fw = []
    for s in seqs:
        rs = [r for r in R if r["seq"] == s]
        vis = sum(r["apple_px_visible"] for r in rs); ps = sum(r["apple_px_passed"] for r in rs)
        if vis > 0 and ps == 0: fw.append(s)
    per_cam = {}
    for c in ("zed_left", "wrist_cam_l", "wrist_cam_r"):
        rc = [r for r in R if r["camera"] == c]
        per_cam[c] = {"apple_visible": sum(r["apple_px_visible"] for r in rc), "apple_passed": sum(r["apple_px_passed"] for r in rc),
                      "robot_visible": sum(r["robot_px_visible"] for r in rc), "robot_passed": sum(r["robot_px_passed"] for r in rc)}
    av = sum(r["apple_px_visible"] for r in R); ap = sum(r["apple_px_passed"] for r in R)
    avp = sum(r["apple_px_visible"] for r in R if pre(r)); app = sum(r["apple_px_passed"] for r in R if pre(r))
    ov = sum(r["other_px_visible"] for r in R); op = sum(r["other_px_passed"] for r in R)
    o = old[run]["inflation_0.0"]
    out[run] = {"n_frames": len(seqs), "n_rows": len(R), "grasp_seq_gt_dz50": g,
                "robot_px_visible_total": sum(r["robot_px_visible"] for r in R), "robot_px_passed_total(leak)": sum(r["robot_px_passed"] for r in R),
                "n_row_camera_with_robot_leak": sum(1 for r in R if r["robot_px_passed"] > 0),
                "leak_by_body_[visible,passed,n_row_cameras_with_leak]": leak,
                "apple_px_visible_total": av, "apple_px_passed_total": ap, "apple_retention": ap / av if av else None,
                "apple_retention_pre_grasp": app / avp if avp else None,
                "row_cameras_apple_whole_deleted": whole, "n_row_cameras_apple_whole_deleted": len(whole),
                "frames_apple_whole_deleted_all_cameras": fw,
                "n_frames_apple_whole_deleted_pre_grasp": sum(1 for s in fw if g is None or s < g),
                "n_row_cameras_apple_whole_deleted_pre_grasp": sum(1 for s, t, c, n in whole if g is None or s < g),
                "other_px_retention": op / ov if ov else None, "per_camera": per_cam,
                "P1-3b_old_model_inflation0_for_reference": {"robot_px_passed_total": o["robot_px_passed_total"], "apple_retention": o["apple_retention"],
                                                              "n_frames_apple_visible_but_all_removed": o["n_frames_apple_visible_but_all_removed"]}}
    print(run, {k: v for k, v in out[run].items() if k not in ("row_cameras_apple_whole_deleted", "per_camera")})
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T28/p3/off_0e_summary.json'), "w"), indent=1)
