"""Wiring smoke (Implementer B) of the DINO path with a RANDOM-INIT ViT-S/16 state dict (no download): official
dino/vision_transformer.py loads, last-block qkv hook, key reshape, ncut, mask in RGB frame. Numbers meaningless.

Run (from the checkout root): source Baseline/paper_baselines/setup/paper_baselines_env.sh;
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD .venv-baselines/bin/python Baseline/paper_baselines/scripts/b_smoke_tokencut_randinit.py
"""
import json, sys, resource
from pathlib import Path
import numpy as np

import torch
from Baseline.paper_baselines.providers import tokencut as tc
from Baseline.paper_baselines.interface import TargetQuery
scratch = Path("outputs/paper_baselines_smoke")
vits = tc.load_official_module("dino/vision_transformer.py", "_tmp_vits")
torch.manual_seed(0)
m = vits.vit_small(patch_size=16, num_classes=0)
scratch.mkdir(parents=True, exist_ok=True); wp = scratch / "vits16_randinit.pth"; torch.save(m.state_dict(), wp)
ex = tc.DinoKeyExtractor(weights_path=str(wp))
rgb = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
rgb2 = (np.random.default_rng(1).random((200, 250, 3)) * 255).astype(np.uint8)
feats, dims = ex(rgb)
prov = tc.TokenCutProvider(extractor=ex)
res = prov.predict({"head": rgb, "left_wrist": rgb2}, TargetQuery("pick the bowl", "bowl"))
out = {c: {"found": r.found, "mask_shape": list(r.mask.shape), "mask_px": int(r.mask.sum()),
           "feature_grid": r.extras.get("feature_grid"), "latency_ms": round(r.latency_ms, 1)} for c, r in res.items()}
out["feats_shape"] = list(feats.shape); out["dims"] = list(dims); out["weights"] = "RANDOM INIT (wiring only)"
out["load_msg"] = ex.weights["load_msg"]; out["maxrss_MB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(json.dumps(out, indent=1))
dst = Path("outputs/paper_baselines_smoke/B_tokencut_randinit_smoke.json"); dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, indent=1))
wp.unlink()  # random-init weights are only a wiring device
