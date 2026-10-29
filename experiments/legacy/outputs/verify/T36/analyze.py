"""T36-baseline: collect judgment numbers and per-frame clearance_before from legacy + cuRobo runs.
usage: python analyze.py  -> writes outputs/verify/T36/summary.json (numbers only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, hashlib, struct

T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T36'))


def load(p):
    d = json.loads(pathlib.Path(p).read_text())
    f = d["frames"]
    before = [r["clearance_before_mm"] for r in f]
    return {
        "path": str(pathlib.Path(p).relative_to(_ag3s_workspace_path(''))),
        "violated_start": sum(b < 0 for b in before),
        "violated_start_set": [r["i"] for r in f if r["clearance_before_mm"] < 0],
        "has_target": sum(bool(r["target"]) for r in f),
        "frame0_clearance_before_mm": before[0],
        "clearance_before_mm": before,
        "clearance_before_sha1": hashlib.sha1(struct.pack(f"{len(before)}d", *before)).hexdigest(),
        "arm_before_mm": [r["arm_before_mm"] for r in f],
        "esdf_backend": sorted({r["esdf_backend"] for r in f}),
        "esdf_occupied": [r["esdf_occupied"] for r in f],
        "target_voxels_carved": [r["target_voxels_carved"] for r in f],
        "esdf_observed_frustum": [r.get("esdf_observed_frustum") for r in f],
        "n": len(f),
        # SQP-side values: not baseline (wall-clock budget); kept for completeness only
        "status_counts": {s: sum(1 for r in f if r["status"] == s) for s in sorted({r["status"] for r in f})},
        "iterations": [r["iterations"] for r in f],
        "ag3s_ms": [r["ag3s_ms"] for r in f],
    }


out = {"legacy": {}, "curobo": {}}
for p in sorted((T / "legacy").glob("*.json")):
    out["legacy"][p.stem] = load(p)
for p in sorted((T / "curobo").glob("*.json")):
    out["curobo"][p.stem] = load(p)

clean = [k for k in out["curobo"] if "overlap" not in k]
keys = ("violated_start", "has_target", "frame0_clearance_before_mm")
if clean:
    ref = out["curobo"][clean[0]]
    cmp = {}
    for k in clean:
        r = out["curobo"][k]
        diff = [i for i in range(r["n"]) if r["clearance_before_mm"][i] != ref["clearance_before_mm"][i]]
        cmp[k] = {"judgment_equal_to_" + clean[0]: all(r[x] == ref[x] for x in keys),
                  "per_frame_clearance_before_bitwise_equal": not diff,
                  "frames_differing": diff,
                  "max_abs_diff_mm": max([abs(r["clearance_before_mm"][i] - ref["clearance_before_mm"][i]) for i in diff], default=0.0),
                  "occupied_equal": r["esdf_occupied"] == ref["esdf_occupied"]}
    out["curobo_repro"] = cmp
    leg = out["legacy"].get("legacy_shim")
    if leg:
        out["legacy_vs_curobo"] = {
            "frames": list(range(ref["n"])),
            "legacy_before_mm": leg["clearance_before_mm"],
            "curobo_before_mm": ref["clearance_before_mm"],
            "curobo_minus_legacy_mm": [c - l for c, l in zip(ref["clearance_before_mm"], leg["clearance_before_mm"])],
            "legacy_violated_set": leg["violated_start_set"],
            "curobo_violated_set": ref["violated_start_set"],
            "violated_only_legacy": sorted(set(leg["violated_start_set"]) - set(ref["violated_start_set"])),
            "violated_only_curobo": sorted(set(ref["violated_start_set"]) - set(leg["violated_start_set"])),
        }
(T / "summary.json").write_text(json.dumps(out, indent=1))
for grp in ("legacy", "curobo"):
    for k, r in out[grp].items():
        print(f"{grp:6s} {k:22s} {r['violated_start']}/{r['n']} has_target {r['has_target']}/{r['n']} "
              f"f0 {r['frame0_clearance_before_mm']!r} sha1 {r['clearance_before_sha1'][:12]} status {r['status_counts']}")
print(json.dumps(out.get("curobo_repro"), indent=1))
