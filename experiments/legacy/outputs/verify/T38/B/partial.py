"""update T38.verify.partial.json from the cmp/*.json files present (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, time
B = pathlib.Path(_ag3s_workspace_path('outputs/verify/T38/B'))
P = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T38.verify.partial.json'))
d = json.loads(P.read_text()) if P.exists() else {"step": "T38-phaseB-verify", "date": "2026-10-02", "done": [], "numbers": {}}
d["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
v1 = {}
for f in sorted((B / "cmp").glob("v1_*.json")):
    c = json.loads(f.read_text())
    v1[f.stem[3:]] = {"n_chunks": c["n_chunks"], "same_count": c["same_count"], "depth_mask_pixels": c["depth_mask_pixels"],
                      "record_chunks_arrays_identical": f'{c["record_npz"]["chunks_arrays_identical"]}/{c["record_npz"]["n_a"]}',
                      "record_arrays_compared": c["record_npz"]["n_arrays_compared"],
                      "record_summary_non_time_identical": f'{c["record_npz"]["summary_non_time_identical"]}/{c["record_npz"]["n_a"]}',
                      "raw": str(f)}
d["numbers"]["V1"] = v1
for extra in sorted(B.glob("partial_*.json")):
    d["numbers"][extra.stem[len("partial_"):]] = json.loads(extra.read_text())
P.write_text(json.dumps(d, indent=1, ensure_ascii=False))
print(list(d["numbers"]), list(v1))
