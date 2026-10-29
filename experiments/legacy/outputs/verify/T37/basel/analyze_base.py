"""T37 baselines: judgment numbers (violated_start, has_target, frame0 clearance_before) + per-frame clearance_before vs T36 reference runs.
usage: python3 analyze_base.py -> basel/base_summary.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, hashlib, struct
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T37/basel'))
REF = {"legacy": (_ag3s_workspace_path('outputs/verify/T36/legacy/legacy_shim.json'), 14, 15, -29.031048280806342),
       "curobo": (_ag3s_workspace_path('outputs/verify/T36/curobo/curobo_r1.json'), 10, 15, -9.171877401271193)}
def load(p):
    f = json.loads(pathlib.Path(p).read_text())["frames"]
    b = [r["clearance_before_mm"] for r in f]
    return {"violated_start": sum(x < 0 for x in b), "has_target": sum(bool(r["target"]) for r in f), "n": len(f),
            "frame0_clearance_before_mm": b[0], "clearance_before_mm": b,
            "clearance_before_sha1": hashlib.sha1(struct.pack(f"{len(b)}d", *b)).hexdigest(),
            "esdf_backend": sorted({r["esdf_backend"] for r in f}),
            "status_counts_not_baseline": {s: sum(1 for r in f if r["status"] == s) for s in sorted({r["status"] for r in f})}}
out = {}
for mode, tag in (("legacy", "legacy_shim"), ("curobo", "curobo_r1")):
    r = load(T / mode / f"{tag}.json"); refp, vs, ht, f0 = REF[mode]; ref = load(refp)
    r["expected"] = {"violated_start": vs, "has_target": ht, "frame0_clearance_before_mm": f0}
    r["matches_expected"] = (r["violated_start"] == vs and r["has_target"] == ht and r["frame0_clearance_before_mm"] == f0)
    r["reference_run"] = refp
    r["per_frame_clearance_before_bitwise_equal_to_reference"] = r["clearance_before_mm"] == ref["clearance_before_mm"]
    r["raw"] = str(T / mode / f"{tag}.json"); r["log"] = str(T / mode / f"{tag}.log"); r["pre"] = str(T / mode / f"{tag}.pre.txt")
    out[mode] = r
    print(mode, f"{r['violated_start']}/{r['n']} has_target {r['has_target']}/{r['n']} f0 {r['frame0_clearance_before_mm']!r}",
          "match", r["matches_expected"], "per-frame eq", r["per_frame_clearance_before_bitwise_equal_to_reference"], r["esdf_backend"], r["status_counts_not_baseline"])
(T / "base_summary.json").write_text(json.dumps(out, indent=1))
