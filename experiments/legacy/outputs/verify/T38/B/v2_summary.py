"""V2/V4 reduction: per run, seq 2..75 of pass T. median and p90 (numpy.percentile, linear) per stage.
stage times are the code's own StageProfiler values (ConstraintSet.profile = summary_json.profile_ms) and
SafePolicy timing (ag3s / trajopt / total); wall = harness perf_counter around infer with cuda sync."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
B = pathlib.Path(_ag3s_workspace_path('outputs/verify/T38/B'))
STAGES = ("scene_reconstruction", "support_surface", "attention_lifting", "target_grounding", "esdf", "constraint_generation")


def stats(v):
    v = np.asarray(v, float)
    return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "mean": float(v.mean()),
            "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def summarise(label):
    rows = [json.loads(l) for l in open(B / "runs" / label / "pass_T.jsonl")]
    rows = [r for r in rows if r["seq"] >= 2]
    out = {"label": label, "seqs": [rows[0]["seq"], rows[-1]["seq"]]}
    for s in STAGES:
        out[s] = stats([r["profile_ms"].get(s, 0.0) for r in rows])
    out["constraint_generation_minus_esdf"] = stats([r["profile_ms"]["constraint_generation"] - r["profile_ms"]["esdf"] for r in rows])
    for k in ("ag3s", "trajopt", "total", "infer"):
        out[f"timing.{k}"] = stats([r["timing_ms"][k] for r in rows])
    out["wall"] = stats([r["wall_ms"] for r in rows])
    out["ag3s_minus_stages"] = stats([r["timing_ms"]["ag3s"] - sum(r["profile_ms"].get(s, 0.0) for s in STAGES if s != "esdf") for r in rows])
    tot = np.asarray([r["timing_ms"]["total"] for r in rows])
    out["chunks_total_le_533ms"] = f"{int((tot <= 533.0).sum())}/{tot.size}"
    out["total_by_seq"] = {int(r["seq"]): r["timing_ms"]["total"] for r in rows}
    out["torch_max_alloc_mib"] = stats([r["torch_max_alloc_mib"] for r in rows])
    out["torch_max_reserved_mib"] = stats([r["torch_max_reserved_mib"] for r in rows])
    out["torch_alloc_after_mib"] = stats([r["torch_alloc_after_mib"] for r in rows])
    out["actions_sha_by_seq"] = {int(r["seq"]): r["actions_sha"] for r in rows}
    meta = json.loads((B / "runs" / label / "meta.json").read_text())
    out["nvsmi_max_used_mib"] = meta["nvsmi_max_used_mib"]
    samples = [json.loads(l) for l in open(B / "runs" / label / "gpu_samples.jsonl") if l.startswith('{"t"')]
    out["nvsmi_used_min_mib"] = min(s["used"] for s in samples) if samples else None
    out["nvsmi_n_samples"] = len(samples)
    pre = (B / "runs" / label / "pre_check.txt").read_text()
    out["pre_check"] = pre.strip().splitlines()
    return out


if __name__ == "__main__":
    res = {l: summarise(l) for l in sys.argv[2:]}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    for l, o in res.items():
        print(f"{l:12s} recon {o['scene_reconstruction']['median']:7.1f} sup {o['support_surface']['median']:6.1f} "
              f"ground {o['target_grounding']['median']:7.1f} esdf {o['esdf']['median']:6.1f} cgen-esdf {o['constraint_generation_minus_esdf']['median']:6.1f} "
              f"TO {o['timing.trajopt']['median']:6.1f} total {o['timing.total']['median']:7.1f} (p90 {o['timing.total']['p90']:7.1f}) "
              f"<=533 {o['chunks_total_le_533ms']} torch {o['torch_max_alloc_mib']['max']:.0f} nvsmi {o['nvsmi_max_used_mib']}")
