"""``closed_loop/jobs.py``: suite x level x task x episode x arm -> ``run_parallel_gpu.sh`` lines that follow the
benchmark registry (``suite_registry`` split rules), and the per suite x level x arm summary.

    PYTHONPATH=$PWD .venv-ag3s/bin/python -m pytest -q Baseline/paper_baselines/closed_loop/tests/test_jobs.py
"""
from __future__ import annotations

import json
import shlex

import pytest

from Baseline.paper_baselines.closed_loop import jobs as J
from Baseline.paper_baselines.closed_loop import launch_method as lm
from benchmark.runners import eval_matrix as em
from benchmark.runners import suite_registry as reg


def plan(arms="none,aegis", suites="all", split="dev", **kw):
    return J.build_jobs(arms=J.parse_arms(arms), suites=J.parse_suites(suites), split=split, run_prefix="t", **kw)


def test_every_in_scope_suite_and_level_and_the_split_rules():
    jobs, skipped = plan()
    pairs = {(j["suite"], j["level"]) for j in jobs}
    want = {(n, lv) for n in reg.suite_names(in_scope_only=True) for lv in reg.levels(n)}
    assert pairs == want and len(want) == 12  # LIBERO 4 suites + SafeLIBERO 4 suites x L I / L II
    assert not skipped and len(jobs) == 2 * 12
    for j in jobs:
        assert j["episode_ids"] == reg.split_episodes("dev") == [0, 1, 2, 3, 4]
        assert j["task_ids"] == list(range(reg.suite(j["suite"])["n_tasks"]))
        assert j["data_split"] == reg.split_label(j["suite"], j["level"], "dev")
        a = j["argv"]
        assert a[a.index("--data-split") + 1] == j["data_split"]
        assert ("--safety-level" in a) == reg.is_safelibero(j["suite"])
        if reg.is_safelibero(j["suite"]):
            assert a[a.index("--safety-level") + 1] == j["level"]
        ns = lm.build_parser().parse_args(a)  # launch_method's own parser and selection check accept the line
        lm.check_selection(ns)
        assert j["run_suffix"] == f"{j['run_id']}-{lm.run_tag(ns)}"
    assert len({j["run_suffix"] for j in jobs}) == len(jobs)  # no two jobs write the same run name
    test, _ = plan(split="test", suites="safelibero_goal:II,libero_10", tasks_per_job=2)
    assert all(j["episode_ids"] == list(range(5, 50)) and len(j["task_ids"]) <= 2 for j in test)
    assert {j["data_split"] for j in test} == {reg.split_label("safelibero_goal", "II", "test"),
                                               reg.split_label("libero_10", None, "test")}


def test_selection_errors_follow_the_registry():
    with pytest.raises(ValueError, match="not in the dev split"):
        plan(suites="safelibero_spatial:I", episode_ids=[5])
    with pytest.raises(ValueError):
        plan(suites="safelibero_spatial:I", task_ids=[4])  # SafeLIBERO suites have 4 tasks
    with pytest.raises(ValueError, match="out of scope"):
        plan(suites="libero_90")
    with pytest.raises(NotImplementedError, match="LIBERO-Safety"):
        J.parse_suites("libero_safety")
    assert J.FAMILY_HOOKS["libero_safety"] is None


def test_privileged_arms_only_on_safelibero_and_their_client_flags():
    jobs, skipped = plan(arms="none,knows:gt,constrained_flow_matching:gt,aegis:gt-names")
    assert {s["suite"] for s in skipped} == set(reg.suite_names(family="libero"))
    assert len(skipped) == 3 * 4
    gt = [j for j in jobs if j["privileged"]]
    assert gt and all(reg.is_safelibero(j["suite"]) for j in gt)
    for j in gt:
        ns = lm.build_parser().parse_args(j["argv"])
        pargs = lm.paired_namespace(ns)
        c = lm.method_client_argv(ns, "<run>", 0, pargs)
        assert "--emit-privileged-obstacles" in c
        assert ("--capture-segmentation" in c) == j["variant"].startswith("knows:gt-seg")
        assert lm.resolve_arm(ns)["privileged"] is True
    none = next(j for j in jobs if j["variant"] == "none" and j["suite"] == "safelibero_spatial")
    ns = lm.build_parser().parse_args(none["argv"])
    pargs = lm.paired_namespace(ns)
    assert lm.method_client_argv(ns, "<run>", 0, pargs) == lm.lp.client_argv(run_dir="<run>", port=0, args=pargs)


def test_none_is_always_rerun_and_presets():
    with pytest.raises(ValueError, match="no 'none' arm"):
        J.main(["plan", "--arms", "aegis", "--split", "dev", "--run-prefix", "t", "--out-dir", "/nonexistent/x"])
    jobs, _ = plan(arms="none,aegis,aegis@v5,constrained_flow_matching@v5", suites="safelibero_spatial:I")
    v5 = {j["method"]: j for j in jobs if j["variant"].endswith("@v5")}
    a = v5["aegis"]["argv"]
    assert json.loads(a[a.index("--method-kwargs") + 1]) == {"crop_mode": "box"} and "--no-suite-extras" in a
    c = v5["constrained_flow_matching"]["argv"]
    assert json.loads(c[c.index("--method-kwargs") + 1]) == {"action_box": False}
    ns = lm.build_parser().parse_args(a)
    assert "--suite" not in lm.server_argv(ns, 0, None) and v5["aegis"]["run_suffix"].endswith("-LI-aegis-v5")
    ns = lm.build_parser().parse_args(next(j for j in jobs if j["variant"] == "aegis:obst-detector:target-detector")["argv"])
    s = lm.server_argv(ns, 0, None)
    assert s[s.index("--suite") + 1] == "safelibero_spatial"
    with pytest.raises(ValueError):
        J.parse_arms("aegis:gt@v5")


def test_jobs_txt_is_run_parallel_gpu_input(tmp_path):
    J.main(["smoke", "--suites", "safelibero_spatial,libero_goal", "--out-dir", str(tmp_path)])
    lines = [l for l in (tmp_path / "jobs.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    p = json.loads((tmp_path / "plan.json").read_text())
    assert len(lines) == len(p["jobs"]) == 3 * 3  # (L I, L II, LIBERO goal) x (none, aegis, CFM)
    assert sorted(shlex.split(l) == j["argv"] for l, j in zip(lines, p["jobs"])) == [True] * len(lines)
    assert all(j["task_ids"] == [0] and j["episode_ids"] == [0] for j in p["jobs"])
    assert "AG3S_PARALLEL_TEST_LAUNCHER" in p["meta"]["run"] and J.LAUNCHER_REL in p["meta"]["run"]


def _row(t, e, ok, col, steps=100, suite="safelibero_spatial"):
    return {"suite": suite, "task_id": t, "episode_idx": e, "official_success": ok, "status": "success" if ok else
            "max_steps", "env_steps_excluding_wait": steps, "safety_level": "I", "max_steps": 280,
            "safelibero": {"collided": col, "safe_success": ok and not col, "max_displacement_m": 0.02 if col else 0.0}}


def test_aggregate_per_suite_level_arm_and_paired_vs_none(tmp_path):
    jobs, _ = plan(arms="none,aegis", suites="safelibero_spatial:I", task_ids=[0, 1], episode_ids=[0, 1])
    results = tmp_path / "results"
    outcomes = {"none": [(True, True), (True, False), (False, False), (True, True)],
                "aegis": [(True, False), (True, False), (False, False), (False, False)]}
    for j in jobs:
        run = results / f"20261025T000000_abc_{j['run_suffix']}"
        run.mkdir(parents=True)
        arm = "none" if j["variant"] == "none" else "aegis"
        keys = [(t, e) for t in j["task_ids"] for e in j["episode_ids"]]
        rows = [_row(t, e, *o) for (t, e), o in zip(keys, outcomes[arm])]
        (run / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    agg = J.aggregate({"meta": {}, "jobs": jobs}, results)
    assert not agg["missing"] and len(agg["groups"]) == 2
    g = {x["variant"]: x for x in agg["groups"]}
    a = g["aegis:obst-detector:target-detector"]
    o = a["metrics"]["safelibero_official"]
    assert (o["n"], o["TSR"], o["CAR"], o["SS"]) == (4, 0.5, 1.0, 0.5)
    assert g["none"]["metrics"]["safelibero_official"]["CAR"] == 0.5
    p = a["paired_vs_none"]["safe_success"]
    assert (p["both"], p["vla_only_only"], p["version_only"]) == (1, 0, 1)
    assert p == em.paired_metrics({k: em.episode_view(_row(*k, *o)) for k, o in zip([(0, 0), (0, 1), (1, 0), (1, 1)], outcomes["none"])},
                                  {k: em.episode_view(_row(*k, *o)) for k, o in zip([(0, 0), (0, 1), (1, 0), (1, 1)], outcomes["aegis"])},
                                  "safe_success")
    assert "safelibero" in agg["families"] and "| safelibero_spatial | I | none |" in J.markdown(agg)
    agg2 = J.aggregate({"meta": {}, "jobs": jobs + plan(arms="none,knows", suites="safelibero_spatial:II")[0]}, results)
    assert len(agg2["missing"]) == 2
