"""BM-SETUP-A: evaluation recipe ``benchmark/runners/eval_matrix.py`` (CPU; no server, synthetic run directories)."""
from __future__ import annotations

import json
import math
import pathlib
import shlex
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture
def em():
    from benchmark.runners import eval_matrix as em

    return em


def test_parse_suites(em):
    allp = em.parse_suites("all")
    assert len(allp) == 12 and ("libero_10", None) in allp and ("safelibero_long", "II") in allp
    assert em.parse_suites("safelibero_goal:II,libero_object") == [("safelibero_goal", "II"), ("libero_object", None)]
    assert len(em.parse_suites("safelibero")) == 8
    with pytest.raises(ValueError):
        em.parse_suites("libero_90")
    with pytest.raises(ValueError):
        em.parse_suites("safelibero_goal:III")


def test_build_jobs_lines_run_through_the_launcher(monkeypatch, capsys, em):
    from benchmark.runners import launch_libero_paired as lp

    jobs = em.build_jobs(suites=em.parse_suites("libero_goal,safelibero_long:I"), version="V11AB-pb7", conversion="C",
                         split="test", tasks_per_job=2, run_prefix="t", step="S")
    assert [j["task_ids"] for j in jobs] == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 1], [2, 3]]
    assert all(j["episode_ids"] == list(range(5, 50)) for j in jobs)
    assert jobs[5]["run_suffix"]["vla_v11ab"] == "t-safelibero_long-test-t0-1-LI-vla_v11ab-V11AB-pb7-C"
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(ROOT))
    monkeypatch.setattr(lp.l0, "checkpoint_identity", lambda path, require=True: {"path": "/ckpt"})
    monkeypatch.setattr(lp.l0, "libero_python", lambda: "/fake/libero/python")
    for j in jobs:
        assert lp.main(shlex.split(shlex.join(j["argv"])) + ["--dry-run"]) == 0  # the jobs.txt line round trip
        out = capsys.readouterr().out
        assert '"--safety-level"' in out if j["level"] else '"--safety-level"' not in out
    with pytest.raises(ValueError, match="not in the dev split"):
        em.build_jobs(suites=[("libero_goal", None)], version="V11AB", conversion="B", split="dev", episode_ids=[7],
                      run_prefix="t", step="S")
    with pytest.raises(ValueError):
        em.build_jobs(suites=[("libero_goal", None)], version="V99", conversion="B", split="dev", run_prefix="t", step="S")


def test_smoke_plan_and_estimate(tmp_path, em, monkeypatch):
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(ROOT))
    assert em.main(["smoke", "--out-dir", str(tmp_path)]) == 0
    plan = json.loads((tmp_path / "plan.json").read_text())
    lines = [l for l in (tmp_path / "jobs.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == len(plan["jobs"]) == 12
    assert all("--modes vla_only,vla_v11ab" in l and "--ag3s-version V11AB --conversion B" in l for l in lines)
    j = next(x for x in plan["jobs"] if x["suite"] == "safelibero_long")
    t = em.TIMING
    expect = sum(t["server_ready_s"][m] + 550 * t["s_per_env_step"][f"safelibero/{m}"] for m in em.MODES)
    assert j["estimate_s"]["total_s"] == pytest.approx(expect)
    assert plan["estimate"]["wall_upper_bound_s"] <= plan["estimate"]["serial_upper_bound_s"]


def test_statistics(em):
    lo, hi = em.wilson(11, 20)
    assert (round(lo, 4), round(hi, 4)) == (0.3421, 0.7418)
    assert em.wilson(0, 0) is None
    assert em.mcnemar_exact_p(0, 0) == 1.0
    assert em.mcnemar_exact_p(0, 6) == pytest.approx(2 / 64)
    assert em.mcnemar_exact_p(3, 3) == 1.0
    ci = em.paired_bootstrap_ci([0] * 10, [1] * 10)
    assert ci == [1.0, 1.0]
    assert em.official_ets({"env_steps_excluding_wait": 120, "official_success": True}) == 119
    assert em.official_ets({"env_steps_excluding_wait": 300, "official_success": False}) == 300


def _write_run(root, suffix, rows):
    d = root / f"20261023T000000_abc_{suffix}"
    d.mkdir(parents=True)
    (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def _row(t, e, ok, steps, collided=None, seed=7):
    r = {"task_id": t, "episode_idx": e, "init_state_index": e, "official_success": ok, "status": "success" if ok else "max_steps",
         "env_steps_excluding_wait": steps, "safe_client": {"policy_seed": seed * 100000 + t * 1000 + e, "hold_calls": 0}}
    if collided is not None:
        r["safelibero"] = {"collided": collided, "safe_success": ok and not collided, "obstacle": "o"}
    return r


def test_aggregate_paired(tmp_path, em):
    jobs = em.build_jobs(suites=[("safelibero_spatial", "I"), ("libero_goal", None)], version="V11AB-pb7",
                         conversion="C", split="dev", task_ids=[0], episode_ids=[0, 1, 2, 3], run_prefix="a", step="S")
    sl, lg = jobs
    # SafeLIBERO: VLA-only 3 success (1 collided), version 2 success (0 collided)
    _write_run(tmp_path, sl["run_suffix"]["vla_only"], [_row(0, 0, True, 100, False), _row(0, 1, True, 120, True),
                                                        _row(0, 2, True, 90, False), _row(0, 3, False, 280, False)])
    _write_run(tmp_path, sl["run_suffix"]["vla_v11ab"], [_row(0, 0, True, 110, False), _row(0, 1, True, 130, False),
                                                         _row(0, 2, False, 280, False), _row(0, 3, False, 280, False)])
    _write_run(tmp_path, lg["run_suffix"]["vla_only"], [_row(0, e, e != 3, 100) for e in range(4)])
    agg = em.aggregate({"jobs": jobs, "meta": {}}, tmp_path)
    assert agg["missing"] == [lg["run_suffix"]["vla_v11ab"]]
    g = next(x for x in agg["groups"] if x["suite"] == "safelibero_spatial")
    a, b = g["vla_only"], g["version"]
    assert (a["official_success"], a["collided"], a["safe_success"]) == (3, 1, 2)
    assert (b["official_success"], b["collided"], b["safe_success"]) == (2, 0, 2)
    assert a["CAR"] == 0.75 and a["TSR"] == 0.75 and a["ETS"] == pytest.approx((99 + 119 + 89 + 280) / 4)
    p = g["paired"]
    assert (p["official_success"]["both"], p["official_success"]["vla_only_only"], p["official_success"]["version_only"],
            p["official_success"]["neither"]) == (2, 1, 0, 1)
    assert p["safe_success"]["gained"] == [[0, 1]] and p["safe_success"]["lost"] == [[0, 2]]
    assert p["no_collision"]["version_only"] == 1
    assert "safelibero" in agg["families"] and agg["families"]["safelibero"]["version"]["n"] == 4
    assert "| safelibero_spatial | I | 4 |" in em.markdown(agg)


def test_episode_reuse_rule(em):
    job = {"suite": "safelibero_spatial", "level": "I", "seed": 7, "step_budget": "total_incl_wait",
           "episode_ids": [0, 1, 2, 3, 4]}
    rows = {(t, e): _row(t, e, True, 100, False) for t in range(4) for e in range(5)}
    basis = {"problems": [], "suite": "safelibero_spatial", "safety_level": "I", "seed": 7,
             "step_budget": "total_incl_wait", "task_ids": [0, 1, 2, 3], "episode_ids": [0, 1, 2, 3, 4], "episodes": rows}
    assert em.episode_reusable(basis, job, 2, 4) == (True, "ok")
    assert em.episode_reusable({**basis, "problems": ["code_changed"]}, job, 2, 4)[0] is False
    assert em.episode_reusable({**basis, "safety_level": "II"}, job, 0, 0)[0] is False
    assert em.episode_reusable({**basis, "step_budget": "main_aegis"}, job, 0, 0)[0] is False
    # reset history: a run that ran only episode 3 shows the fixtures of the task's first reset, not its fourth
    ok, why = em.episode_reusable({**basis, "episode_ids": [3]}, {**job, "episode_ids": [0, 1, 2, 3]}, 0, 3)
    assert not ok and "reset history" in why
    # ...but a prefix of the planned order is fine
    assert em.episode_reusable({**basis, "episode_ids": [0, 1]}, job, 1, 1) == (True, "ok")
    bad = dict(rows)
    bad[(1, 1)] = {**rows[(1, 1)], "status": "execution_error"}
    assert em.episode_reusable({**basis, "episodes": bad}, job, 1, 1) == (False, "execution_error")
    bad[(1, 2)] = {**rows[(1, 2)], "safe_client": {"policy_seed": 1}}
    assert em.episode_reusable({**basis, "episodes": bad}, job, 1, 2) == (False, "policy_seed differs")


def _manifest_run(tmp_path, em, *, commit, protocol_patch=None, dirty_patch=""):
    from benchmark.runners import launch_libero_paired as lp
    from types import SimpleNamespace

    a = SimpleNamespace(task_suite_name="safelibero_spatial", safety_level="I", task_ids=[0], episode_ids=[0, 1],
                        seed=7, resize_size=224, replan_steps=5, num_steps_wait=10)
    proto = em._jsonable(lp.protocol_settings(a))
    proto.update(protocol_patch or {})
    d = tmp_path / f"run-{len(list(tmp_path.iterdir()))}"
    d.mkdir()
    deps = json.loads((ROOT / "doc/migration/source-manifest.json").read_text())["dependencies"]
    settings = {"condition": "vla_only", "protocol": proto, "runtime_schedule": em._jsonable(lp.runtime_schedule(a)),
                "safety_level": "I", "task_ids": [0], "episode_ids": [0, 1], "seed": 7,
                "checkpoint": {"aggregate_sha256": "A", "norm_stats_sha256": "N"},
                "versions": {"libero_python": {"v": 1}, "openpi_python": {"v": 2}}}
    (d / "run_manifest.json").write_text(json.dumps({"commit": commit, "dirty": bool(dirty_patch),
                                                     "dependencies": deps, "settings": settings}))
    (d / "source.patch").write_text(dirty_patch)
    (d / "episodes.jsonl").write_text("".join(json.dumps(_row(0, e, True, 100, False)) + "\n" for e in (0, 1)))
    return d


def test_run_basis_and_apply_reuse(tmp_path, em):
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    deps = json.loads((ROOT / "doc/migration/source-manifest.json").read_text())["dependencies"]
    now = {"checkpoint": {"aggregate_sha256": "A", "norm_stats_sha256": "N"},
           "versions": {"libero_python": {"v": 1}, "openpi_python": {"v": 2}}, "dependencies": deps, "head": head}
    good = _manifest_run(tmp_path, em, commit=head)
    b = em.run_basis(good, now=now)
    assert b["problems"] == [], b["diffs"]
    other = _manifest_run(tmp_path, em, commit=head, protocol_patch={"max_steps": 300})
    assert em.run_basis(other, now=now)["problems"] == ["protocol"]
    dirty = _manifest_run(tmp_path, em, commit=head,
                          dirty_patch="diff --git a/benchmark/runners/libero_safe_client.py b/benchmark/runners/libero_safe_client.py\n")
    assert "uncommitted_at_run" in em.run_basis(dirty, now=now)["problems"]
    assert "checkpoint" in em.run_basis(good, now={**now, "checkpoint": {"aggregate_sha256": "B"}})["problems"]
    jobs = em.build_jobs(suites=[("safelibero_spatial", "I")], version="V11AB-pb7", conversion="C", split="dev",
                         task_ids=[0], episode_ids=[0, 1], run_prefix="r", step="S")
    rep = em.apply_reuse(jobs, [other, good], now=now)
    assert jobs[0]["modes"] == ["vla_v11ab"] and "--modes vla_v11ab " in shlex.join(jobs[0]["argv"]) + " "
    assert jobs[0]["vla_only_source"] == {"0:0": str(good), "0:1": str(good)}
    assert rep["jobs"][0]["reused"] is True
    jobs2 = em.build_jobs(suites=[("safelibero_spatial", "I")], version="V11AB-pb7", conversion="C", split="dev",
                          task_ids=[0], episode_ids=[0, 1, 2], run_prefix="r", step="S")
    rep2 = em.apply_reuse(jobs2, [good], now=now)
    assert jobs2[0]["modes"] == ["vla_only", "vla_v11ab"] and jobs2[0]["vla_only_source"] is None
    assert list(rep2["jobs"][0]["not_reusable"]) == ["0:2"]
