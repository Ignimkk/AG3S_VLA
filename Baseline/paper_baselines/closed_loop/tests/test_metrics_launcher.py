"""Closed-loop metrics against the recorded SL5c runs, and the launcher's client argv against SL5c's.

``.venv-ag3s`` (no JAX / torch needed)::

    PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests/test_metrics_launcher.py
"""
from __future__ import annotations

import json
import os
import pathlib

import pytest

from Baseline.paper_baselines.closed_loop import metrics as M

ROOT = pathlib.Path(__file__).resolve().parents[4]
STORAGE = pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", ROOT.parent / "AG3S_VLA-storage"))
RB = STORAGE / "results/benchmark"
SL5C = {"LI_vla_only": RB / "20261014T180105_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_only",
        "LI_vla_v11ab_B": RB / "20261014T182539_61f44671805b_sl5c-safelibero-spatial-dev-LI-vla_v11ab-B",
        "LII_vla_only": RB / "20261014T190837_61f44671805b_sl5c-safelibero-spatial-dev-LII-vla_only"}
VERIFY = ROOT / "AG3S/ag3s/docs/handoff/SL5c-safelibero-dev.verify.json"
need_runs = pytest.mark.skipif(not all(p.is_dir() for p in SL5C.values()) or not VERIFY.exists(),
                               reason="SL5c runs / verify.json not available")


def row(success, steps, disp, budget=280):
    return {"official_success": success, "env_steps_excluding_wait": steps, "max_steps": budget, "task_id": 0,
            "episode_idx": 0, "safelibero": {"max_displacement_m": disp, "collided": disp > 0.001,
                                             "safe_success": success and disp <= 0.001}}


def test_official_definitions_on_synthetic_rows():
    rows = [row(True, 120, 0.0), row(True, 90, 0.005), row(False, 280, 0.02), row(False, 280, 0.0)]
    o = M.safelibero_official(rows)
    assert o["TSR"] == 0.5 and o["CAR"] == 0.5 and o["SS"] == 0.25
    assert o["t_per_episode"] == [119, 89, 280, 280]  # main_aegis: success breaks before t += 1
    assert o["ETS"] == (119 + 89 + 280 + 280) / 4
    k = M.knows(rows)
    assert (k["SR"], k["CR"], k["SSR"]) == (0.5, 0.5, 0.25)
    ml = M.multilink_11mm(rows)
    assert ml["collision_rate"] == 0.25 and ml["safe_success_rate"] == 0.5  # 5 mm is no collision at 11 mm
    assert M.cfm(rows)["ETS"] == o["ETS"]


@need_runs
@pytest.mark.parametrize("name", ["LI_vla_only", "LI_vla_v11ab_B"])
def test_sl5c_level1_numbers_match_the_verified_record(name):
    ref = json.loads(VERIFY.read_text())["numbers"]["LI_paired"]
    key = "vla_only" if name == "LI_vla_only" else "vla_v11ab"
    s = M.summarize(SL5C[name])
    assert s["sl5c"]["n"] == ref["n"] == 20
    assert s["sl5c"]["official_success"] == ref[f"{key}_success"]
    assert s["sl5c"]["collided"] == ref[f"{key}_collided"]
    assert s["sl5c"]["safe_success"] == ref[f"{key}_safe_success"]
    o = s["safelibero_official"]
    assert o["n_success"] == ref[f"{key}_success"] and o["n_collision"] == ref[f"{key}_collided"]
    assert o["n_safe_success"] == ref[f"{key}_safe_success"] and o["collision_consistent_with_client"]
    assert o["budget"] == [{"policy_steps": 280, "step_budget": "total_incl_wait"}]


@need_runs
def test_sl5c_level2_vla_only_and_pairing():
    s = M.summarize(SL5C["LII_vla_only"], SL5C["LI_vla_only"])
    assert (s["sl5c"]["official_success"], s["sl5c"]["collided"], s["sl5c"]["safe_success"]) == (11, 18, 2)  # SL5c log
    p = M.summarize(SL5C["LI_vla_v11ab_B"], SL5C["LI_vla_only"])["paired_vs_ref"]
    assert p["n_common"] == 20 and p["official_success"]["here"] == 3 and p["official_success"]["ref"] == 11
    assert p["safe_success"]["here_only"] == []  # SL5c: no "VLA-only fails -> V11AB safe success" case


@need_runs
def test_launcher_client_argv_equals_the_sl5c_client_argv():
    from Baseline.paper_baselines.closed_loop import launch_method as L
    from benchmark.runners import launch_libero_paired as lp

    man = json.loads((SL5C["LI_vla_only"] / "run_manifest.json").read_text())
    rec = next(e["settings"]["argv"] for e in man["executions"] if e["settings"].get("role") == "client")
    args = L.build_parser().parse_args(["--method", "aegis", "--safety-level", "I", "--task-ids", "0,1,2,3",
                                        "--episode-ids", "0,1,2,3,4", "--run-id", "x",
                                        "--data-split", "dev (SafeLIBERO Spatial ep 0-4 per task x level, user 2026-10-14)"])
    run_dir, port = rec[rec.index("--run-dir") + 1], int(rec[rec.index("--port") + 1])
    mine = lp.client_argv(run_dir=run_dir, port=port, args=L.paired_namespace(args))
    assert mine == rec


def test_launcher_server_argv_and_arm_record():
    from Baseline.paper_baselines.closed_loop import arms
    from Baseline.paper_baselines.closed_loop import launch_method as L

    args = L.build_parser().parse_args(["--method", "nudge", "--safety-level", "I", "--task-ids", "0",
                                        "--episode-ids", "0", "--run-id", "x"])
    argv = L.server_argv(args, 1234, "/ckpt")
    assert argv[1:3] == ["-m", "Baseline.paper_baselines.closed_loop.serve_method"] and "--fake-policy" not in argv
    assert argv[argv.index("--checkpoint") + 1] == "/ckpt"
    for name in arms.ARMS:
        a = arms.resolve(name)
        assert a["privileged"] is False and a["privileged_inputs"] == [] and a["deviation_from_paper"]
    with pytest.raises(ValueError):
        arms.resolve("nudge", obstacles="detector")
    assert arms.resolve("aegis", obstacles="depth")["target"] == "detector"
