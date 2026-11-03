"""T82 §1 — is the serving HSR5 feature the T77 feature? (GPU, run inside `gpu_lock.py shared`.)

For recorded closed-loop frames (an AG3S client run, `ag3s/run_0000/step_*.npz`) whose T77 summary npz exists
(`pi05_hidden --pool summary`), this builds the request the way the serving client sends it (16D state, CHW images,
the recorded prompt), runs it through the **serving** code — `AttentionPolicy._infer_extended` with the noun attention
source (`noun_target:L10H7`, the T78 configuration) and the HSR5 probe, i.e. `JaxPrefixBackend.run(want_hsr5=True)` on
the attention copy (`return_attn_probs=True`, gemma `return_hidden` on for that pass only) — and compares:

* the 8192-D feature with the T77 npz rows (layer 4, rows 0, 1, 2, 4): relative Frobenius error and max |diff|;
* the probe argmax on both (`SubtaskProbe.predict`);
* hsr5 on vs off on the same request: the attention cell and the kv_L4 feature (max |diff|), so the flag does not
  move what the server already used;
* prefix-pass time (ms) of `JaxPrefixBackend.run` with and without `want_hsr5` (after warm-up, `--repeat` each,
  interleaved), wall-clock including the host copies.

usage (from the checkout root, env.sh sourced):
  python3 scripts/setup/gpu_lock.py shared --est-gib 30 --tag t82 -- .venv-openpi-live/bin/python -m \\
      AG3S.ag3s.experiments.studies.t82_feature_check --t77-npz <T77 closed npz> --out <json> [--frames 0 10 ...]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import time

import numpy as np

T77_CLOSED = ("/mnt/dev/AG3S_VLA-storage/results/development/20261030T004052+0900_8ab3c7ef7757_1c61eaabd79c/verify/T77/"
              "smoke/hidden/closed/T71_REG_R0_k0_runs_B_ROI_FPR_B_ROI_FPR_ep211_s2113.npz")
SOURCE = "noun_target:L10H7"


def relerr(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--t77-npz", default=T77_CLOSED)
    ap.add_argument("--frames", nargs="+", type=int, default=[0, 10, 19, 20, 24, 40, 60, 74])
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    import jax

    from AG3S.ag3s.experiments.sources import pi05_hidden as ph
    from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler, load_policy
    from AG3S.ag3s.experiments.sources.policy_record import POLICY_CAMERA_NAMES
    from AG3S.trajopt.attention_policy import AttentionPolicy
    from AG3S.trajopt.attention_source import JaxPrefixBackend, NounTokens, noun_lookup, parse_attention_source
    from AG3S.trajopt.hsr5 import FEATURE_ROWS, HIDDEN_LAYER, load_hsr5_probe

    if jax.default_backend() != "gpu":
        raise SystemExit(f"expected a GPU backend, got {jax.devices()}")
    z = np.load(args.t77_npz)
    layers = [int(x) for x in z["layers"]]
    li = layers.index(HIDDEN_LAYER)
    files = [str(f) for f in z["source_files"]]
    config = args.config or str(z["config"])
    ckpt = args.checkpoint or str(z["checkpoint"])

    t0 = time.perf_counter()
    attn_policy = load_policy(config, ckpt)          # the serving attention copy (return_attn_probs=True)
    load_s = time.perf_counter() - t0
    sampler = AttentionSampler(attn_policy._model)  # noqa: SLF001
    backend = JaxPrefixBackend(sampler)
    source = parse_attention_source(SOURCE)
    noun_tokens = NounTokens.from_config(config)
    probe = load_hsr5_probe()

    class _Served:  # the action policy is not needed for the feature; its input transform is the same pipeline
        def infer(self, obs, **_):
            return {"actions": np.zeros((50, 16), np.float32)}

        _input_transform = staticmethod(attn_policy._input_transform)  # noqa: SLF001

    on = AttentionPolicy(_Served(), None, sampler=sampler, backend=backend, attention_source=source,
                         noun_tokens=noun_tokens, subtask_probe="default", hsr5_probe=probe)
    off = AttentionPolicy(_Served(), None, sampler=sampler, backend=backend, attention_source=source,
                          noun_tokens=noun_tokens, subtask_probe="default")

    rows = []
    for i in args.frames:
        fr = ph.read_frame("ag3s", pathlib.Path(files[i]), None)
        prompt = str(z["prompts"][i])
        assert int(z["t_step"][i]) == fr.t_step, (i, fr.t_step)
        obs = {"state": fr.state, "prompt": prompt,
               "images": {k: np.transpose(fr.images[k], (2, 0, 1)) for k in POLICY_CAMERA_NAMES}}
        r_on = on.infer(dict(obs))
        r_off = off.infer(dict(obs))
        want = np.asarray(z["hidden_summary"][i, li][list(FEATURE_ROWS)], np.float32).reshape(-1)
        # the serving feature itself (the same call AttentionPolicy made)
        inputs = attn_policy._input_transform(dict(obs))  # noqa: SLF001
        look = noun_lookup(inputs, prompt, noun_tokens)
        out = backend.run(inputs, source=source, rows=look.rows if look.ok else None, want_feature=True,
                          v_layer=None, want_hsr5=True)
        got = out["hsr5_feature"]
        offline_pred = probe.predict(want)
        cam_off = np.concatenate([np.asarray(r_off["attention"][k]).ravel() for k in sorted(r_off["attention"])])
        cam_on = np.concatenate([np.asarray(r_on["attention"][k]).ravel() for k in sorted(r_on["attention"])])
        rows.append({
            "frame": i, "t_step": fr.t_step, "noun_ok": bool(look.ok), "path": out["path"],
            "relerr": relerr(got, want), "max_abs": float(np.abs(got - want).max()),
            "relerr_by_row": [relerr(got[j * 2048:(j + 1) * 2048], want[j * 2048:(j + 1) * 2048]) for j in range(4)],
            "serving_argmax": r_on["hsr5"]["argmax"], "offline_argmax": offline_pred["argmax"],
            "p_maxdiff": float(max(abs(r_on["hsr5"]["p"][c] - offline_pred["p"][c]) for c in offline_pred["p"])),
            "attention_maxdiff_on_off": float(np.abs(cam_on - cam_off).max()),
            "subtask_p_maxdiff_on_off": float(max(abs(r_on["subtask"]["p"][c] - r_off["subtask"]["p"][c])
                                                   for c in r_off["subtask"]["p"])),
            "result_keys_extra": sorted(set(r_on) - set(r_off)),
        })
        print(json.dumps(rows[-1]), flush=True)

    # timing: backend.run with / without want_hsr5, interleaved, after one warm-up each (compile excluded)
    fr = ph.read_frame("ag3s", pathlib.Path(files[args.frames[0]]), None)
    obs = {"state": fr.state, "prompt": str(z["prompts"][args.frames[0]]),
           "images": {k: np.transpose(fr.images[k], (2, 0, 1)) for k in POLICY_CAMERA_NAMES}}
    inputs = attn_policy._input_transform(dict(obs))  # noqa: SLF001
    look = noun_lookup(inputs, obs["prompt"], noun_tokens)
    kw = dict(source=source, rows=look.rows if look.ok else None, want_feature=True, v_layer=None)
    for flag in (False, True):
        backend.run(inputs, **kw, want_hsr5=flag)
    ms = {"off": [], "on": [], "on_inner": []}
    for _ in range(int(args.repeat)):
        for flag, key in ((False, "off"), (True, "on")):
            t = time.perf_counter()
            o = backend.run(inputs, **kw, want_hsr5=flag)
            ms[key].append((time.perf_counter() - t) * 1000.0)
            if flag:
                ms["on_inner"].append(o["hsr5_ms"])

    def summ(v):
        return {"median": round(statistics.median(v), 2), "p95": round(float(np.percentile(v, 95)), 2),
                "min": round(min(v), 2), "n": len(v)}

    res = {
        "t77_npz": args.t77_npz, "config": config, "checkpoint": ckpt, "source": SOURCE, "probe": probe.name,
        "device": str(jax.devices()[0]), "model_load_s": round(load_s, 1),
        "frames": rows,
        "relerr_max": max(r["relerr"] for r in rows), "relerr_median": statistics.median(r["relerr"] for r in rows),
        "argmax_agree": sum(r["serving_argmax"] == r["offline_argmax"] for r in rows), "n": len(rows),
        "pass_1e-3": all(r["relerr"] <= 1e-3 for r in rows),
        "prefix_ms": {k: summ(v) for k, v in ms.items()},
        "timing_note": "JaxPrefixBackend.run wall-clock incl. host copies; on = want_hsr5 (the same noun prefix pass "
                       "with return_hidden + on-device pooling); on_inner = the hsr5_pass call alone",
        "env": {"XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                "XLA_FLAGS": os.environ.get("XLA_FLAGS")},
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=1))


if __name__ == "__main__":
    main()
