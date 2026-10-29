"""Summarize p13b_raw.jsonl -> p13b_selffilter_summary.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, collections
D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/P1-3'))
rows = [json.loads(l) for l in (D / "p13b_raw.jsonl").read_text().splitlines()]
out = {"definitions": {
    "source": "outputs/verify/T28/P1-3/p13b_selffilter.py (header docstring); raw per (run, seq, camera, inflation) in p13b_raw.jsonl",
    "leak": "robot GT pixels (MuJoCo body under the robot root) whose back-projected point passes the self-filter (stays in the cloud)",
    "apple_retention": "apple GT pixels passed / apple GT pixels visible",
    "frames": "all planning rows: T14 seq 1-30 (30), T17 seq 1-75 (75); cameras zed_left, wrist_cam_l, wrist_cam_r"}}
for run in ("t14", "t17"):
    R = {}
    for s in (0.0, 0.05):
        rs = [r for r in rows if r["run"] == run and r["inflation"] == s]
        seqs = sorted({r["seq"] for r in rs})
        per_body = collections.defaultdict(lambda: [0, 0, 0])   # visible, passed, n_frames_with_leak
        per_cam = {}
        for cam in ("zed_left", "wrist_cam_l", "wrist_cam_r"):
            rc = [r for r in rs if r["camera"] == cam]
            per_cam[cam] = {"apple_visible": sum(r["apple_px_visible"] for r in rc), "apple_passed": sum(r["apple_px_passed"] for r in rc),
                            "robot_visible": sum(r["robot_px_visible"] for r in rc), "robot_passed": sum(r["robot_px_passed"] for r in rc),
                            "n_frames_robot_leak_gt0": sum(r["robot_px_passed"] > 0 for r in rc)}
        for r in rs:
            for b, (v, p) in r["robot_by_body_visible_passed"].items():
                per_body[b][0] += v; per_body[b][1] += p; per_body[b][2] += int(p > 0)
        per_seq = []
        for q in seqs:
            rq = [r for r in rs if r["seq"] == q]
            per_seq.append({"seq": q, "t": rq[0]["t"], "apple_visible": sum(r["apple_px_visible"] for r in rq),
                            "apple_passed": sum(r["apple_px_passed"] for r in rq),
                            "apple_passed_wrist_l": sum(r["apple_px_passed"] for r in rq if r["camera"] == "wrist_cam_l"),
                            "apple_visible_wrist_l": sum(r["apple_px_visible"] for r in rq if r["camera"] == "wrist_cam_l"),
                            "robot_passed": sum(r["robot_px_passed"] for r in rq),
                            "robot_passed_by_body": {b: v for b, v in collections.Counter({b: vp[1] for r in rq for b, vp in r["robot_by_body_visible_passed"].items() if vp[1] > 0}).items()}})
        av = sum(r["apple_px_visible"] for r in rs); ap = sum(r["apple_px_passed"] for r in rs)
        R[f"inflation_{s}"] = {
            "n_frames": len(seqs), "n_rows": len(rs),
            "apple_px_visible_total": av, "apple_px_passed_total": ap, "apple_retention": ap / av if av else None,
            "n_frames_apple_visible_but_all_removed": sum(1 for p in per_seq if p["apple_visible"] > 0 and p["apple_passed"] == 0),
            "robot_px_visible_total": sum(r["robot_px_visible"] for r in rs), "robot_px_passed_total": sum(r["robot_px_passed"] for r in rs),
            "n_frames_with_robot_leak": sum(1 for p in per_seq if p["robot_passed"] > 0),
            "leak_by_body_[visible_px, passed_px, n_camera_frames_with_leak]": {b: v for b, v in sorted(per_body.items(), key=lambda x: -x[1][1]) if v[1] > 0},
            "per_camera": per_cam, "per_seq": per_seq}
    out[run] = R
    for s in ("inflation_0.0", "inflation_0.05"):
        x = R[s]; print(run, s, "apple", x["apple_px_passed_total"], "/", x["apple_px_visible_total"], "frames all-removed", x["n_frames_apple_visible_but_all_removed"],
                        "| robot leak", x["robot_px_passed_total"], "/", x["robot_px_visible_total"], "frames", x["n_frames_with_robot_leak"])
        print("   ", x["leak_by_body_[visible_px, passed_px, n_camera_frames_with_leak]"])
(D / "p13b_selffilter_summary.json").write_text(json.dumps(out, indent=1))
