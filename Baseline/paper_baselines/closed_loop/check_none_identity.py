"""``--method none`` vs the VLA-only server, same process, same policy, recorded SL5c requests (V3 GPU check).

Both served objects wrap the **same** loaded policy; every episode starts with its ``ag3s/policy_seed`` request, which
re-keys the policy RNG, so the two sequences are independent of order. Reported per call:

* ``bit_identical``: ``MethodServer(none)`` chunk == ``PandaVLAOnly(SeededPolicy(policy))`` chunk (``tobytes``);
* ``max_abs_vs_recorded``: against the chunk the SL5c VLA-only server returned for the same request (another
  process: equal only with ``XLA_FLAGS=--xla_gpu_autotune_level=0`` in both, else ~1e-3, ``policy_seed.py``).

    source scripts/setup/env.sh
    export OPENPI_DATA_HOME=$PWD/.cache/openpi XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_command_buffer=
    flock -w 3600 /tmp/ag3s-vla-gpu-validation.lock env PYTHONPATH=$PWD .venv-openpi-live/bin/python -m \
        Baseline.paper_baselines.closed_loop.check_none_identity --stems safelibero_spatial_t0_e0 --calls 0-9 --out <dir>

``--fake-policy recorded`` runs the same check on CPU without the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "third_party/openpi/src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
SL5C_LI_VLA = "20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only"


def main(argv=None) -> int:
    from AG3S.adapters.panda.server import PandaVLAOnly
    from AG3S.trajopt.policy_seed import SeededPolicy

    from Baseline.paper_baselines.closed_loop import arms
    from Baseline.paper_baselines.closed_loop.replay import RecordedEpisode, parse_calls, resolve_run
    from Baseline.paper_baselines.closed_loop.serve_method import MethodServer, default_checkpoint, load_norm

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=SL5C_LI_VLA)
    ap.add_argument("--stems", nargs="+", default=["safelibero_spatial_t0_e0"])
    ap.add_argument("--calls", default="0-9")
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--fake-policy", choices=("none", "recorded"), default="none")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    run = resolve_run(a.run)
    ckpt = pathlib.Path(a.checkpoint or default_checkpoint())
    if a.fake_policy == "recorded":
        from Baseline.paper_baselines.closed_loop.fake_policy import FakePolicy

        policy = FakePolicy(load_norm(ckpt), source="recorded", run_dir=str(run))
    else:
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        policy = _policy_config.create_trained_policy(_config.get_config(a.config), str(ckpt))
    ref = PandaVLAOnly(SeededPolicy(policy))
    srv = MethodServer(policy, arms.resolve("none"))
    rows = []
    for stem in a.stems:
        ep = RecordedEpisode(run, stem)
        calls = parse_calls(a.calls, ep.n_calls)
        calls = calls if calls[:1] == [0] else [0] + calls
        outs = {}
        for name, served in (("vla_only", ref), ("method_none", srv)):
            outs[name] = []
            for k in calls:
                if hasattr(policy, "set_context"):  # fake policy: chunk lookup by (seed, seq)
                    policy.set_context(policy_seed=ep.policy_seed, seq=k)
                outs[name].append(np.asarray(served.infer(ep.request(k))["actions"]))
        for i, k in enumerate(calls):
            a_ref, a_m = outs["vla_only"][i], outs["method_none"][i]
            rec = np.asarray(ep.calls["actions_response"][k])
            rows.append({"stem": stem, "call": k, "bit_identical": bool(a_ref.dtype == a_m.dtype and
                                                                        a_ref.tobytes() == a_m.tobytes()),
                         "max_abs_vs_vla_only": float(np.abs(a_ref.astype(np.float64) - a_m).max()),
                         "max_abs_vs_recorded": float(np.abs(a_m.astype(np.float64) - rec).max())})
            print(json.dumps(rows[-1]), flush=True)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"run": str(run), "checkpoint": str(ckpt), "fake_policy": a.fake_policy, "rows": rows,
               "all_bit_identical": all(r["bit_identical"] for r in rows),
               "max_abs_vs_recorded": max(r["max_abs_vs_recorded"] for r in rows)}
    (out / "none_identity.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("all_bit_identical", "max_abs_vs_recorded")}))
    return 0 if summary["all_bit_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
