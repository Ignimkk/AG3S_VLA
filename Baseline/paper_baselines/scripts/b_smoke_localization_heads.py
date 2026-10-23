"""Wiring smoke (Implementer B): frozen localization heads on one recorded dev call (no numbers judged).

Run (from the checkout root): CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv-ag3s/bin/python Baseline/paper_baselines/scripts/b_smoke_localization_heads.py
"""
import json, sys
from pathlib import Path
import numpy as np

from Baseline.paper_baselines.providers import localization_heads as lh
from Baseline.paper_baselines.interface import TargetQuery
S = Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
audit = S / "20261012T202459_309e4c6da477_sl3-attention-audit-dev"
src = S / "20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0"
row = json.loads((audit / "attention_index.jsonl").read_text().splitlines()[0])
z = np.load(audit / row["attn_npz"])
k = 0
obs = int(z["obs_index"][k])
with np.load(src / row["npz"]) as ep:
    raw_head = ep["obs_agentview_image"][obs]; raw_wrist = ep["obs_robot0_eye_in_hand_image"][obs]
rgb = {"head": raw_head[::-1], "left_wrist": raw_wrist[::-1]}  # adapter OpenCV frame = raw[::-1] (attention.py doc)
q = TargetQuery(instruction=row["task_description"], target_phrase="black bowl")
out = {}
for proto in ("official_code", "paper"):
    prov = lh.LocalizationHeadsProvider(Path("Baseline/paper_baselines/providers/frozen") /
                                        f"localization_heads_pi05_libero_spatial_dev_init0-4_{proto}.json")
    res = prov.predict(rgb, q, aux=lh.aux_from_attention_npz(z, k))
    out[proto] = {c: {"found": r.found, "mask_px": int(r.mask.sum()), "shape": list(r.mask.shape),
                      "boxes": None if r.boxes is None else r.boxes.tolist(), "heads": r.extras["heads"],
                      "peak_xy": r.extras.get("peak_xy"), "latency_ms": round(r.latency_ms, 2)} for c, r in res.items()}
out["source"] = {"audit_run": audit.name, "attn_npz": row["attn_npz"], "call": k, "obs_index": obs}
dst = Path("outputs/paper_baselines_smoke/B_localization_heads_smoke.json"); dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
