"""Evaluation recipe (BM-SETUP-A): suite x level x task x episode matrix -> run_parallel_gpu.sh jobs -> aggregate.

Runs under ``.venv-ag3s`` (CPU only; it never starts a server). Four sub-commands:

``plan``       a version (``AG3S_VERSIONS`` key) + conversion + suites / levels / tasks + split (or explicit episodes)
               -> ``<out>/jobs.txt`` (one ``launch_libero_paired.py`` argument line per job, the input of
               ``benchmark/runners/run_parallel_gpu.sh``) and ``<out>/plan.json`` (jobs, expected run-id suffixes,
               time estimate). ``--vla-only reuse`` checks earlier VLA-only runs (pairing basis below) and drops the
               ``vla_only`` mode from every job whose episodes are all covered; ``rerun`` (default) pairs fresh.
``smoke``      the GPU smoke list: every in-scope suite x level, task 0, episode 0 (dev), VLA-only + V11AB (B).
``aggregate``  after the jobs ran: per suite x level the official success, SafeLIBERO CAR / TSR / ETS / safe success
               (definitions of ``benchmark/configs/suites.json``), the paired transitions against VLA-only, Wilson 95 %
               intervals, McNemar exact p and a paired bootstrap interval of the rate difference.
``basis``      only the VLA-only pairing basis of given runs for a plan (what ``plan --vla-only reuse`` writes).

VLA-only reuse rule (SL5g ``sl5g_pairing_basis``, user 2026-10-18: "reuse VLA-only only if path, env, checkpoint,
settings, task/init, policy seed and reset history are identical; else rerun that baseline"), checked per episode:

1. the run is a ``vla_only`` run of the same suite / level / seed;
2. its manifest ``settings.protocol`` and ``runtime_schedule`` equal what the launcher computes now for that run's
   own task / episode ids (step budget included);
3. checkpoint aggregate SHA-256 and norm-stats SHA-256 equal the checkpoint of the plan;
4. ``settings.versions.libero_python`` / ``openpi_python`` (interpreter + package versions) equal now;
5. ``git diff <run commit> HEAD`` touches none of :data:`VLA_ONLY_CODE_PATHS` and the run was not started with
   uncommitted changes in them (``source.patch``); a path may be accepted only with ``--accept-diff PATH`` and is then
   listed in the basis with the commits that touched it;
6. dependencies (``run_manifest.dependencies``) equal ``doc/active/migration/source-manifest.json`` now;
7. the episode exists, ``status != execution_error``, ``init_state_index == episode_idx`` and
   ``policy_seed == seed*100000 + task*1000 + episode``;
8. reset history: the run's episode list for that task is identical up to and including the episode (the fixture
   placement of episode k depends on how many resets the task's env saw before, SL4b). Task grouping does not
   matter: each task gets a fresh env re-seeded with ``env.seed(seed)``.

    source scripts/setup/env.sh
    .venv-ag3s/bin/python benchmark/runners/eval_matrix.py plan --version V11AB-pb7 --conversion C \\
        --suites safelibero --split dev --run-prefix bm-pb7 --out-dir outputs/eval/bm-pb7-dev
    benchmark/runners/run_parallel_gpu.sh --jobs outputs/eval/bm-pb7-dev/jobs.txt
    .venv-ag3s/bin/python benchmark/runners/eval_matrix.py aggregate --plan outputs/eval/bm-pb7-dev/plan.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.runners import suite_registry as reg  # noqa: E402

LAUNCHER = "benchmark/runners/launch_libero_paired.py"
MODES = ("vla_only", "vla_v11ab")
#: Code the VLA-only condition runs (client + ``serve_safe --no-safe`` -> ``AG3S/adapters/panda/server.PandaVLAOnly``
#: around ``policy_seed.SeededPolicy``) and the simulator / policy sources it loads.
VLA_ONLY_CODE_PATHS = (
    "benchmark/runners/libero_safe_client.py", "benchmark/runners/libero_pi05_client.py",
    "benchmark/adapters/libero_panda/__init__.py", "benchmark/adapters/libero_panda/camera.py",
    "benchmark/adapters/libero_panda/wire_pack.py", "benchmark/adapters/libero_panda/client_env.py",
    "benchmark/runners/launch_libero_paired.py", "benchmark/runners/launch_libero_pi05_only.py",
    "AG3S/trajopt/serve_safe.py", "AG3S/trajopt/policy_seed.py",
    "AG3S/adapters/panda/server.py", "AG3S/adapters/panda/action.py", "AG3S/adapters/panda/model.py",
    "third_party/openpi", "benchmark/LIBERO", "benchmark/SafeLIBERO", "scripts/setup")
#: Time model (s): medians of the SL5d-SL5i paired runs on libero_spatial / safelibero_spatial
#: (``outputs/verify/BM-SETUP-A/timing_basis.json``). Other suites have not been timed.
TIMING = {"s_per_env_step": {"libero/vla_only": 0.263, "libero/vla_v11ab": 0.361,
                             "safelibero/vla_only": 0.312, "safelibero/vla_v11ab": 0.440},
          "server_ready_s": {"vla_only": 62.0, "vla_v11ab": 56.0},
          "source": "outputs/verify/BM-SETUP-A/timing_basis.json (median over runs; spatial suites only)"}
SMOKE = {"version": "V11AB", "conversion": "B", "task_ids": [0], "episode_ids": [0], "split": "dev",
         "run_prefix": "bm-smoke", "step": "BM-SMOKE"}


# --- selection --------------------------------------------------------------------------------------------------
def parse_suites(text: str) -> list[tuple[str, str | None]]:
    """``all`` | ``libero`` | ``safelibero`` | comma list of ``suite`` or ``suite:I`` / ``suite:II``."""
    out = []
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        if item in ("all", "libero", "safelibero"):
            fam = None if item == "all" else item
            for name in reg.suite_names(in_scope_only=True, family=fam):
                out += [(name, lv) for lv in reg.levels(name)]
            continue
        name, _, level = item.partition(":")
        reg.suite(name)
        if not reg.suite(name)["in_scope"]:
            raise ValueError(f"{name} is out of scope (benchmark/configs/suites.json)")
        lvls = reg.levels(name)
        if level:
            if level not in lvls:
                raise ValueError(f"{name}: level {level!r} not in {lvls}")
            out.append((name, level))
        else:
            out += [(name, lv) for lv in lvls]
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def _ints(text):
    return [int(x) for x in str(text).replace(",", " ").split()] if text not in (None, "") else None


def _launcher():
    from benchmark.runners import launch_libero_paired as lp

    return lp


def job_run_id(prefix: str, suite: str, split: str, tasks: list[int]) -> str:
    """``<prefix>-<suite>-<split>-t<first>[-<last>]``; the launcher appends ``-L<level>-<mode tag>``."""
    t = f"t{tasks[0]}" if len(tasks) == 1 else f"t{tasks[0]}-{tasks[-1]}"
    return f"{prefix}-{suite}-{split}-{t}"


def build_jobs(*, suites, version, conversion, split, task_ids=None, episode_ids=None, tasks_per_job=0, seed=7,
               run_prefix, step, modes=MODES, step_budget="total_incl_wait", launcher_extra=()) -> list[dict]:
    lp = _launcher()
    from AG3S.adapters.panda import profile

    profile.version_server_flags(version)  # unknown version -> ValueError
    if conversion not in ("A", "B", "C", "D", "E"):
        raise ValueError("conversion must be A, B, C, D or E")
    jobs = []
    for suite, level in suites:
        tasks = list(task_ids) if task_ids is not None else list(range(int(reg.suite(suite)["n_tasks"])))
        eps = list(episode_ids) if episode_ids is not None else reg.split_episodes(split)
        outside = sorted(set(eps) - set(reg.split_episodes(split)))
        if outside:
            raise ValueError(f"episodes {outside} are not in the {split} split")
        label = reg.split_label(suite, level, split)
        reg.check_selection(suite, level, tasks, eps, label)
        size = tasks_per_job or len(tasks)
        for i in range(0, len(tasks), size):
            group = tasks[i:i + size]
            run_id = job_run_id(run_prefix, suite, split, group)
            argv = ["--task-suite-name", suite] + (["--safety-level", level] if level else []) + [
                "--task-ids", ",".join(map(str, group)), "--episode-ids", ",".join(map(str, eps)),
                "--seed", str(seed), "--run-id", run_id, "--modes", ",".join(modes), "--step", step,
                "--data-split", label, "--ag3s-version", version, "--conversion", conversion]
            if step_budget != "total_incl_wait":
                argv += ["--safelibero-step-budget", step_budget]
            argv += list(launcher_extra)
            ns = lp.build_parser().parse_args(argv)  # the launcher's own parser must accept the line
            jobs.append({"suite": suite, "level": level, "task_ids": group, "episode_ids": eps, "split": split,
                         "data_split": label, "seed": seed, "run_id": run_id, "modes": list(modes),
                         "version": version, "conversion": conversion, "step_budget": step_budget, "argv": argv,
                         "run_suffix": {m: f"{run_id}-{lp.run_tag(m, ns)}" for m in MODES},
                         "estimate_s": estimate_job(suite, step_budget, len(group) * len(eps), modes)})
    return jobs


def estimate_job(suite: str, step_budget: str, n_episodes: int, modes) -> dict:
    """Upper bound: every episode runs to the step budget (+ wait); plus one server start per mode."""
    fam = reg.suite(suite)["family"]
    steps = reg.total_env_steps(suite, step_budget) * n_episodes
    per = {m: TIMING["server_ready_s"][m] + steps * TIMING["s_per_env_step"][f"{fam}/{m}"] for m in modes}
    return {"env_steps_upper_bound": steps, "per_mode_s": per, "total_s": sum(per.values())}


def schedule_estimate(jobs, max_parallel=2) -> dict:
    """Greedy list schedule of the job upper bounds on ``max_parallel`` lanes (run_parallel_gpu.sh order)."""
    lanes = [0.0] * max_parallel
    for j in jobs:
        k = lanes.index(min(lanes))
        lanes[k] += j["estimate_s"]["total_s"]
    serial = sum(j["estimate_s"]["total_s"] for j in jobs)
    return {"jobs": len(jobs), "serial_upper_bound_s": serial, "max_parallel": max_parallel,
            "wall_upper_bound_s": max(lanes) if jobs else 0.0, "timing": TIMING}


def write_plan(out_dir: Path, jobs: list[dict], meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {"schema": 1, "meta": meta, "jobs": jobs, "estimate": schedule_estimate(jobs)}
    lines = [f"# eval_matrix {meta.get('command')} {meta.get('version')} {meta.get('conversion')} split={meta.get('split')}",
             f"# {len(jobs)} jobs; upper-bound wall time at 2 parallel: {plan['estimate']['wall_upper_bound_s'] / 3600:.2f} h",
             "# run: source scripts/setup/env.sh && benchmark/runners/run_parallel_gpu.sh --jobs <this file>"]
    lines += [shlex.join(j["argv"]) for j in jobs]
    (out_dir / "jobs.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    return plan


# --- VLA-only pairing basis ---------------------------------------------------------------------------------------
def _git(*a, check=True):
    return subprocess.run(["git", "-C", str(PROJECT_ROOT), *a], capture_output=True, text=True, check=check).stdout


def _diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}")
    elif a != b:
        out.append({"key": path, "run": a, "now": b})
    return out


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


def _patch_paths(run_dir: Path) -> list[str]:
    p = run_dir / "source.patch"
    if not p.is_file():
        return []
    return sorted({line.split(" b/", 1)[1] for line in p.read_text(errors="replace").splitlines()
                   if line.startswith("diff --git ") and " b/" in line})


def _under(path: str, roots) -> bool:
    return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)


def run_basis(run_dir: Path, *, now: dict, accept_diff=()) -> dict:
    """Run-level checks 1-6 of the module docstring for VLA-only run ``run_dir``."""
    lp = _launcher()
    m = json.loads((run_dir / "run_manifest.json").read_text())
    s = m["settings"]
    out = {"run": str(run_dir), "commit": m["commit"], "dirty": m["dirty"], "problems": []}
    if s.get("condition") != "vla_only":
        out["problems"].append(f"condition {s.get('condition')!r} is not vla_only")
        return out
    proto = s["protocol"]
    args = SimpleNamespace(task_suite_name=proto["suite"], safety_level=s.get("safety_level"),
                           task_ids=s["task_ids"], episode_ids=s["episode_ids"], seed=s["seed"], resize_size=224,
                           replan_steps=5, num_steps_wait=10,
                           safelibero_step_budget=(proto.get("safelibero_step_budget") or {}).get("budget",
                                                                                             "total_incl_wait"))
    out.update({"suite": proto["suite"], "safety_level": s.get("safety_level"), "seed": s["seed"],
                "task_ids": s["task_ids"], "episode_ids": s["episode_ids"], "step_budget": args.safelibero_step_budget})
    d = {"protocol": _diff(s["protocol"], _jsonable(lp.protocol_settings(args))),
         "runtime_schedule": _diff(s["runtime_schedule"], _jsonable(lp.runtime_schedule(args)))}
    ck_run, ck_now = s.get("checkpoint") or {}, now["checkpoint"]
    d["checkpoint"] = [k for k in ("aggregate_sha256", "norm_stats_sha256") if ck_run.get(k) != ck_now.get(k)]
    v_run = (s.get("versions") or {})
    d["interpreters"] = {k: _diff(v_run.get(k), now["versions"].get(k)) for k in ("libero_python", "openpi_python")}
    d["interpreters"] = {k: v for k, v in d["interpreters"].items() if v}
    d["dependencies"] = _diff(m.get("dependencies"), now["dependencies"])
    changed = [p for p in _git("diff", "--name-only", m["commit"], "HEAD", "--", *VLA_ONLY_CODE_PATHS).split() if p]
    dirty_in_path = [p for p in _patch_paths(run_dir) if _under(p, VLA_ONLY_CODE_PATHS)]
    accepted = {p: _git("log", "--oneline", f"{m['commit']}..HEAD", "--", p).splitlines()
                for p in changed if p in accept_diff}
    d["code_changed"] = [p for p in changed if p not in accept_diff]
    d["code_accepted"] = accepted
    d["uncommitted_at_run"] = dirty_in_path
    out["diffs"] = d
    for key in ("protocol", "runtime_schedule", "checkpoint", "interpreters", "dependencies", "code_changed",
                "uncommitted_at_run"):
        if d[key]:
            out["problems"].append(key)
    rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines() if line.strip()]
    out["episodes"] = {(int(r["task_id"]), int(r["episode_idx"])): r for r in rows}
    return out


def episode_reusable(basis: dict, job: dict, task: int, episode: int) -> tuple[bool, str]:
    """Checks 7-8 for one episode of ``job`` against a run whose run-level ``basis`` passed."""
    if basis["problems"]:
        return False, "run: " + ",".join(basis["problems"])
    if (basis["suite"], basis["safety_level"], basis["seed"], basis["step_budget"]) != (
            job["suite"], job["level"], job["seed"], job["step_budget"]):
        return False, "suite/level/seed/step budget differ"
    if task not in basis["task_ids"]:
        return False, "task not in run"
    row = basis["episodes"].get((task, episode))
    if row is None:
        return False, "episode not in run"
    if row.get("status") == "execution_error":
        return False, "execution_error"
    if int(row.get("init_state_index", -1)) != episode:
        return False, "init_state_index differs"
    seed_rule = job["seed"] * 100000 + task * 1000 + episode
    if int((row.get("safe_client") or {}).get("policy_seed", -1)) != seed_rule:
        return False, "policy_seed differs"
    i = job["episode_ids"].index(episode)
    run_eps = list(basis["episode_ids"])
    if run_eps[:i + 1] != job["episode_ids"][:i + 1]:
        return False, f"reset history differs (run order {run_eps[:i + 1]} vs job {job['episode_ids'][:i + 1]})"
    return True, "ok"


def now_reference(checkpoint_dir: str | None = None) -> dict:
    lp = _launcher()
    l0 = lp.l0
    ck = l0.checkpoint_identity(Path(checkpoint_dir or l0.default_checkpoint()), require=False)
    v = _jsonable(lp.versions())
    deps = json.loads((PROJECT_ROOT / "doc/active/migration/source-manifest.json").read_text())["dependencies"]
    return {"checkpoint": {k: ck.get(k) for k in ("path", "aggregate_sha256", "norm_stats_sha256")},
            "versions": {k: v.get(k) for k in ("libero_python", "openpi_python")}, "dependencies": deps,
            "head": _git("rev-parse", "HEAD").strip()}


def apply_reuse(jobs: list[dict], candidates: list[Path], *, now: dict, accept_diff=()) -> dict:
    """Per job: VLA-only source of every episode, or keep ``vla_only`` in the job's modes. Returns the basis."""
    bases = [run_basis(Path(c), now=now, accept_diff=accept_diff) for c in candidates]
    report = {"now": now, "accept_diff": list(accept_diff), "rule": __doc__.split("VLA-only reuse rule", 1)[1]
              .split("    source scripts", 1)[0].strip(), "runs": [], "jobs": []}
    for b in bases:
        report["runs"].append({k: v for k, v in b.items() if k != "episodes"})
    for job in jobs:
        sources, reasons = {}, {}
        for t in job["task_ids"]:
            for e in job["episode_ids"]:
                why = []
                for b in bases:
                    ok, reason = episode_reusable(b, job, t, e)
                    if ok:
                        sources[f"{t}:{e}"] = b["run"]
                        break
                    why.append(f"{Path(b['run']).name}: {reason}")
                else:
                    reasons[f"{t}:{e}"] = why or ["no candidate run"]
        covered = not reasons
        if covered and "vla_only" in job["modes"]:
            job["modes"] = [m for m in job["modes"] if m != "vla_only"]
            k = job["argv"].index("--modes")
            job["argv"][k + 1] = ",".join(job["modes"])
            job["estimate_s"] = estimate_job(job["suite"], job["step_budget"],
                                             len(job["task_ids"]) * len(job["episode_ids"]), job["modes"])
        job["vla_only_source"] = sources if covered else None
        report["jobs"].append({"run_id": job["run_id"], "reused": covered, "sources": sources,
                               "not_reusable": reasons})
    return report


# --- aggregation ----------------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p on the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(x: list[int], y: list[int], n_boot: int = 10000, seed: int = 0) -> list[float] | None:
    """95 % percentile interval of mean(y - x) over episodes resampled with replacement."""
    import numpy as np

    if not x:
        return None
    d = np.asarray(y, float) - np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def official_ets(row: dict) -> int:
    """main_aegis ``t`` at episode end: policy steps - 1 on success (break before ``t += 1``), else policy steps."""
    steps = int(row["env_steps_excluding_wait"])
    return steps - 1 if row.get("official_success") else steps


def episode_view(row: dict) -> dict:
    sl = row.get("safelibero") or {}
    out = {"official_success": bool(row.get("official_success")), "status": row.get("status"),
           "steps": row.get("env_steps_excluding_wait"), "hold_calls": (row.get("safe_client") or {}).get("hold_calls")}
    if sl:
        out.update({"collided": bool(sl.get("collided")), "safe_success": bool(sl.get("safe_success")),
                    "ets": official_ets(row), "obstacle": sl.get("obstacle")})
    return out


def condition_metrics(eps: list[dict], safelibero: bool) -> dict:
    n = len(eps)
    k = sum(e["official_success"] for e in eps)
    out = {"n": n, "official_success": k, "official_success_rate": k / n if n else None,
           "official_success_wilson95": wilson(k, n),
           "execution_error": sum(e["status"] == "execution_error" for e in eps)}
    if safelibero and n:
        col = sum(e["collided"] for e in eps)
        ss = sum(e["safe_success"] for e in eps)
        out.update({"collided": col, "safe_success": ss, "safe_success_rate": ss / n,
                    "safe_success_wilson95": wilson(ss, n), "CAR": 1 - col / n, "CAR_wilson95": wilson(n - col, n),
                    "TSR": k / n, "ETS": sum(e["ets"] for e in eps) / n})
    return out


def paired_metrics(base: dict, new: dict, key: str) -> dict:
    keys = sorted(set(base) & set(new))
    x = [int(base[k][key]) for k in keys]
    y = [int(new[k][key]) for k in keys]
    n11 = sum(a and b for a, b in zip(x, y))
    n10 = sum(a and not b for a, b in zip(x, y))  # VLA-only true, version false
    n01 = sum(b and not a for a, b in zip(x, y))  # VLA-only false, version true
    n00 = sum(not a and not b for a, b in zip(x, y))
    return {"n_pairs": len(keys), "both": n11, "vla_only_only": n10, "version_only": n01, "neither": n00,
            "diff_rate": (sum(y) - sum(x)) / len(keys) if keys else None,
            "diff_bootstrap95": paired_bootstrap_ci(x, y), "mcnemar_exact_p": mcnemar_exact_p(n10, n01),
            "gained": [list(k) for k in keys if base[k][key] is False and new[k][key] is True],
            "lost": [list(k) for k in keys if base[k][key] is True and new[k][key] is False]}


def find_run(results_root: Path, suffix: str) -> Path | None:
    hits = sorted(p for p in results_root.glob(f"*_{suffix}") if (p / "episodes.jsonl").is_file())
    return hits[-1] if hits else None


def load_episodes(run: Path) -> dict:
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines() if line.strip()]
    return {(int(r["task_id"]), int(r["episode_idx"])): episode_view(r) for r in rows}


def aggregate(plan: dict, results_root: Path) -> dict:
    groups = {}
    missing = []
    for job in plan["jobs"]:
        g = groups.setdefault((job["suite"], job["level"]), {"vla_only": {}, "vla_v11ab": {}, "runs": {"vla_only": [], "vla_v11ab": []}})
        wanted = {(t, e) for t in job["task_ids"] for e in job["episode_ids"]}
        for mode in MODES:
            if mode == "vla_only" and job.get("vla_only_source"):
                for key, run in job["vla_only_source"].items():
                    t, e = (int(v) for v in key.split(":"))
                    g["vla_only"][(t, e)] = load_episodes(Path(run))[(t, e)]
                    if run not in g["runs"]["vla_only"]:
                        g["runs"]["vla_only"].append(run)
                continue
            if mode not in job["modes"]:
                continue
            run = find_run(results_root, job["run_suffix"][mode])
            if run is None:
                missing.append(job["run_suffix"][mode])
                continue
            g["runs"][mode].append(str(run))
            eps = load_episodes(run)
            g[mode].update({k: v for k, v in eps.items() if k in wanted})
            absent = sorted(wanted - set(eps))
            if absent:
                missing.append(f"{run.name}: episodes {absent}")
    out = {"plan_meta": plan.get("meta"), "missing": missing, "groups": [], "families": {}}
    fam_acc = {}
    for (suite, level), g in groups.items():
        sl = reg.is_safelibero(suite)
        row = {"suite": suite, "level": level, "runs": g["runs"],
               "vla_only": condition_metrics(list(g["vla_only"].values()), sl),
               "version": condition_metrics(list(g["vla_v11ab"].values()), sl), "paired": {}}
        if g["vla_only"] and g["vla_v11ab"]:
            row["paired"]["official_success"] = paired_metrics(g["vla_only"], g["vla_v11ab"], "official_success")
            if sl:
                row["paired"]["safe_success"] = paired_metrics(g["vla_only"], g["vla_v11ab"], "safe_success")
                inv = {k: {"no_collision": not v["collided"]} for k, v in g["vla_only"].items()}
                inv2 = {k: {"no_collision": not v["collided"]} for k, v in g["vla_v11ab"].items()}
                row["paired"]["no_collision"] = paired_metrics(inv, inv2, "no_collision")
        row["episodes"] = [{"task_id": k[0], "episode_idx": k[1], "vla_only": g["vla_only"].get(k),
                            "version": g["vla_v11ab"].get(k)} for k in sorted(set(g["vla_only"]) | set(g["vla_v11ab"]))]
        out["groups"].append(row)
        fam = reg.suite(suite)["family"]
        acc = fam_acc.setdefault(fam, {"vla_only": [], "version": []})
        acc["vla_only"] += list(g["vla_only"].values())
        acc["version"] += list(g["vla_v11ab"].values())
    for fam, acc in fam_acc.items():
        out["families"][fam] = {c: condition_metrics(v, fam == "safelibero") for c, v in acc.items()}
    budgets = {j["step_budget"] for j in plan["jobs"]}
    out["notes"] = {"ets_budget": sorted(budgets),
                    "ets_comparability": "ETS / TSR equal main_aegis only with --safelibero-step-budget main_aegis; "
                                         "with total_incl_wait a failure ends at total - 20 policy steps"}
    return out


def markdown(agg: dict) -> str:
    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}"

    def ci(x):
        return "-" if not x else f"[{100 * x[0]:.1f}, {100 * x[1]:.1f}]"

    lines = ["| suite | level | n | VLA-only success % | version success % | Δ (paired 95 % CI) | McNemar p |"
             " VLA-only safe % | version safe % | CAR VLA / ver | ETS VLA / ver |", "|" + "---|" * 11]
    for g in agg["groups"]:
        a, b, p = g["vla_only"], g["version"], g["paired"].get("official_success", {})
        lines.append("| {} | {} | {} | {} {} | {} {} | {} {} | {} | {} | {} | {} | {} |".format(
            g["suite"], g["level"] or "-", b["n"], pct(a["official_success_rate"]), ci(a["official_success_wilson95"]),
            pct(b["official_success_rate"]), ci(b["official_success_wilson95"]), pct(p.get("diff_rate")),
            ci(p.get("diff_bootstrap95")), "-" if not p else f"{p['mcnemar_exact_p']:.3g}",
            pct(a.get("safe_success_rate")), pct(b.get("safe_success_rate")),
            f"{pct(a.get('CAR'))} / {pct(b.get('CAR'))}" if "CAR" in b else "-",
            f"{a.get('ETS', 0):.1f} / {b.get('ETS', 0):.1f}" if "ETS" in b else "-"))
    return "\n".join(lines) + "\n"


# --- CLI ----------------------------------------------------------------------------------------------------------
def _common(ap):
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-dir", default=None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--version", required=True, help="AG3S_VERSIONS key (AG3S/adapters/panda/profile.py)")
    p.add_argument("--conversion", required=True, choices=("A", "B", "C", "D", "E"))
    p.add_argument("--suites", default="all", help="all | libero | safelibero | suite[:I|II],...")
    p.add_argument("--split", required=True, choices=reg.SPLITS)
    p.add_argument("--tasks", default=None, help="task ids (default: every task of each suite)")
    p.add_argument("--episodes", default=None, help="subset of the split's episodes (default: the whole split)")
    p.add_argument("--tasks-per-job", type=int, default=0, help="0 = one job per suite x level")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--step", default="BM-EVAL")
    p.add_argument("--safelibero-step-budget", default="total_incl_wait", choices=("total_incl_wait", "main_aegis"))
    p.add_argument("--vla-only", default="rerun", choices=("rerun", "reuse", "skip"),
                   help="rerun: pair fresh; reuse: --vla-only-runs (basis per episode); skip: version only, no pairing")
    p.add_argument("--vla-only-runs", nargs="*", default=[])
    p.add_argument("--accept-diff", nargs="*", default=[], help="VLA-only code paths whose diff is accepted (recorded)")
    p.add_argument("--launcher-extra", default="", help="extra launcher arguments for every job (shell quoting)")
    _common(p)
    s = sub.add_parser("smoke")
    s.add_argument("--suites", default="all")
    s.add_argument("--safelibero-step-budget", default="total_incl_wait", choices=("total_incl_wait", "main_aegis"))
    _common(s)
    a = sub.add_parser("aggregate")
    a.add_argument("--plan", required=True)
    a.add_argument("--results-root", default=None)
    a.add_argument("--out", default=None, help="default: <plan dir>/aggregate.json (+ .md)")
    args = ap.parse_args(argv)

    if args.command == "aggregate":
        plan = json.loads(Path(args.plan).read_text())
        from benchmark.runners import libero_pi05_client as base

        root = Path(args.results_root) if args.results_root else base.results_root()
        agg = aggregate(plan, root)
        out = Path(args.out) if args.out else Path(args.plan).with_name("aggregate.json")
        out.write_text(json.dumps(agg, indent=1, ensure_ascii=False, default=str))
        out.with_suffix(".md").write_text(markdown(agg))
        print(markdown(agg))
        if agg["missing"]:
            print("missing:", json.dumps(agg["missing"], indent=1), file=sys.stderr)
        return 0 if not agg["missing"] else 1

    if os.environ.get("AG3S_PROJECT_ROOT") != str(PROJECT_ROOT):
        raise RuntimeError("source scripts/setup/env.sh of this checkout first")
    out_dir = Path(args.out_dir)
    if args.command == "smoke":
        jobs = build_jobs(suites=parse_suites(args.suites), version=SMOKE["version"], conversion=SMOKE["conversion"],
                          split=SMOKE["split"], task_ids=SMOKE["task_ids"], episode_ids=SMOKE["episode_ids"],
                          run_prefix=SMOKE["run_prefix"] + ("" if args.safelibero_step_budget == "total_incl_wait"
                                                            else "-aegis"),
                          step=SMOKE["step"], step_budget=args.safelibero_step_budget)
        meta = {"command": "smoke", **SMOKE, "step_budget": args.safelibero_step_budget,
                "head": _git("rev-parse", "HEAD").strip()}
        plan = write_plan(out_dir, jobs, meta)
    else:
        modes = ("vla_v11ab",) if args.vla_only == "skip" else MODES
        jobs = build_jobs(suites=parse_suites(args.suites), version=args.version, conversion=args.conversion,
                          split=args.split, task_ids=_ints(args.tasks), episode_ids=_ints(args.episodes),
                          tasks_per_job=args.tasks_per_job, seed=args.seed, run_prefix=args.run_prefix, step=args.step,
                          modes=modes, step_budget=args.safelibero_step_budget,
                          launcher_extra=shlex.split(args.launcher_extra))
        meta = {"command": "plan", "version": args.version, "conversion": args.conversion, "split": args.split,
                "vla_only": args.vla_only, "head": _git("rev-parse", "HEAD").strip()}
        if args.vla_only == "reuse":
            if not args.vla_only_runs:
                raise ValueError("--vla-only reuse needs --vla-only-runs RUN_DIR ...")
            basis = apply_reuse(jobs, [Path(r) for r in args.vla_only_runs], now=now_reference(args.checkpoint_dir),
                                accept_diff=tuple(args.accept_diff))
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "pairing_basis.json").write_text(json.dumps(basis, indent=1, ensure_ascii=False, default=str))
            meta["pairing_basis"] = str(out_dir / "pairing_basis.json")
        plan = write_plan(out_dir, jobs, meta)
    est = plan["estimate"]
    print(f"{out_dir / 'jobs.txt'}: {est['jobs']} jobs, upper bound {est['serial_upper_bound_s'] / 3600:.2f} h serial, "
          f"{est['wall_upper_bound_s'] / 3600:.2f} h at {est['max_parallel']} parallel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
