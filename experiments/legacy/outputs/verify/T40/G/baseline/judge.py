"""T40 G baseline judge (copy of SUBTASK-e judge): violated_start (clearance_before < 0), has_target, frame0 clearance_before, sha1 of the 15
clearance_before values (repr) — vs regression-baseline skill / T36.baseline.verify.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, hashlib
REF = {"legacy": (14, 15, -29.031048280806342, "fd943e3e7aa9fc6dd80f3582c67d2d87c0f6caa6"),
       "curobo": (10, 15, -9.171877401271193, "247da8a22076ebbabb466b95d70031d4e768be82")}
out = {}
for b in ("legacy", "curobo"):
    fr = json.load(open(f"{_ag3s_workspace_path()}/outputs/verify/T40/G/baseline/{b}.json"))["frames"]
    cb = [f["clearance_before_mm"] for f in fr]
    v = sum(c < 0 for c in cb); ht = sum(bool(f["target"]) for f in fr)
    sha_variants = {"repr_json_list": hashlib.sha1(json.dumps(cb).encode()).hexdigest(),
                    "repr_join": hashlib.sha1(",".join(repr(c) for c in cb).encode()).hexdigest()}
    r = REF[b]
    out[b] = {"violated_start": f"{v}/{len(fr)}", "has_target": f"{ht}/{len(fr)}", "frame0_clearance_before_mm": cb[0],
              "clearance_before_mm": cb, "sha1": sha_variants,
              "matched": v == r[0] and ht == r[1] and cb[0] == r[2]}
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T40/G/baseline/judge.json'), "w"), indent=1)
print(json.dumps({b: {k: out[b][k] for k in ("violated_start", "has_target", "frame0_clearance_before_mm", "matched", "sha1")} for b in out}, indent=1))
