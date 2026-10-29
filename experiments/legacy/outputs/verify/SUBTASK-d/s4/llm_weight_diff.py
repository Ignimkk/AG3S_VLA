"""SUBTASK-d §4 — recompute "16D checkpoint LLM base weight vs pi05_base" (audit: rel. error 1.7e-3; SigLIP up to 17 %).
CPU only (JAX_PLATFORMS=cpu), numpy restore, float64 arithmetic. No model forward.
Per leaf present in both checkpoints with the same shape (LoRA leaves exist only in 16D): rel = ||a - b|| / ||b||.
Groups by path: llm (PaliGemma.llm.*, excluding lora_*), img (PaliGemma.img.* = SigLIP), other.
Reports per group: n leaves, pooled rel = sqrt(sum ||a-b||^2) / sqrt(sum ||b||^2), median / max per-leaf rel, max-leaf path,
and leaves only in one checkpoint. usage: python llm_weight_diff.py -> s4/llm_weight_diff.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('third_party/openpi/src'))
from openpi.models import model as _model  # noqa: E402
import flax.traverse_util as tu  # noqa: E402
CK16 = _ag3s_workspace_path('legacy/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999/params')
BASE = _ag3s_workspace_path('legacy/pi05_TO_hybrid/cache/openpi/openpi-assets/checkpoints/pi05_base/params')
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-d/s4/llm_weight_diff.json'))
a = tu.flatten_dict(_model.restore_params(CK16, restore_type=np.ndarray), sep="/")
b = tu.flatten_dict(_model.restore_params(BASE, restore_type=np.ndarray), sep="/")


def group(k):
    if "lora" in k:
        return "lora"
    if k.startswith("PaliGemma/llm"):
        return "llm"
    if k.startswith("PaliGemma/img"):
        return "img"
    return "other"


res = {"ck16": CK16, "base": BASE, "only_in_16d": sorted(set(a) - set(b)), "only_in_base": sorted(set(b) - set(a)),
       "dtypes_16d": sorted({str(v.dtype) for v in a.values()}), "dtypes_base": sorted({str(v.dtype) for v in b.values()}), "groups": {}}
per = {}
for k in sorted(set(a) & set(b)):
    x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
    if x.shape != y.shape:
        per[k] = {"group": group(k), "shape_mismatch": [list(x.shape), list(y.shape)]}; continue
    dn, bn = float(np.linalg.norm(x - y)), float(np.linalg.norm(y))
    per[k] = {"group": group(k), "diff_l2": dn, "base_l2": bn, "rel": dn / bn if bn else None, "max_abs": float(np.abs(x - y).max()),
              "equal": bool(np.array_equal(x, y)), "n": int(x.size)}
for g in sorted({v["group"] for v in per.values()}):
    L = [(k, v) for k, v in per.items() if v["group"] == g and "rel" in v and v["rel"] is not None]
    if not L:
        continue
    rel = np.array([v["rel"] for _, v in L])
    kmax = max(L, key=lambda kv: kv[1]["rel"])
    res["groups"][g] = {"n_leaves": len(L), "n_equal": sum(v["equal"] for _, v in L),
                        "pooled_rel": float(np.sqrt(sum(v["diff_l2"] ** 2 for _, v in L)) / np.sqrt(sum(v["base_l2"] ** 2 for _, v in L))),
                        "median_rel": float(np.median(rel)), "max_rel": float(rel.max()), "max_rel_leaf": kmax[0], "min_rel": float(rel.min())}
res["per_leaf"] = per
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("groups", "dtypes_16d", "dtypes_base")}, indent=1)); print("only_in_16d", len(res["only_in_16d"]), "only_in_base", len(res["only_in_base"]))
