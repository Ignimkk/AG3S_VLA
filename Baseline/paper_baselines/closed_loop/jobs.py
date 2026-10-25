"""Benchmark jobs for the comparison methods (BASELINE-BENCH-READY): suite x level x task x episode x arm ->
``run_parallel_gpu.sh`` job lines of ``launch_method.py`` -> per suite x level x arm summary.

Runs under ``.venv-ag3s`` (CPU only; never starts a server). The suite / level / split rules are the benchmark's own
(``benchmark/configs/suites.json`` through ``benchmark/runners/suite_registry`` and ``eval_matrix.parse_suites``);
nothing is copied, and every job line is parsed and checked by ``launch_method``'s own parser and
``check_selection`` before it is written.

``plan``       arms (``none`` | ``<arm>`` | ``<arm>:gt`` | ``<arm>:gt-<source>`` | ``<arm>@v5``; ``arms.arm_spec``,
               :data:`PRESETS`) x suites / levels x
               tasks x split (or explicit episodes) -> ``<out>/jobs.txt`` + ``<out>/plan.json``. One job = one arm on one
               suite x level, the split's episodes of a group of tasks (``--tasks-per-job``).
``smoke``      every in-scope suite x level, task 0, episode 0 (dev), the given arms (default ``none`` + one filter +
               one guidance arm).
``aggregate``  after the jobs ran: per suite x level x arm the official SafeLIBERO TSR / CAR / ETS / SS, LIBERO SR, KNOWS
               SR / CR / SSR, multi-link 11 mm (``closed_loop.metrics``, one definition per block), Wilson 95 % intervals
               (``eval_matrix.condition_metrics``), the paired comparison against the ``none`` arm of the same suite x
               level x episodes (``eval_matrix.paired_metrics``: McNemar exact p, paired bootstrap), the per-call method
               statistics, and per-family totals.

``none`` is **re-run** in every plan (no reuse of earlier VLA-only runs; the benchmark session's recommendation for the
final evaluation, 2026-10-25), so every arm is paired with a VLA-only run of the same code, checkpoint and episodes.
Privileged (``:gt``) arms need SafeLIBERO; on LIBERO suites they are skipped and listed in ``plan.json["skipped"]``.
LIBERO-Safety: :data:`FAMILY_HOOKS` has its slot; it raises until that family's client / registry hook exists.

    source scripts/setup/env.sh
    .venv-ag3s/bin/python -m Baseline.paper_baselines.closed_loop.jobs plan --arms none,aegis,aegis:gt \\
        --suites safelibero --split dev --run-prefix blb --out-dir outputs/eval/baselines-dev
    AG3S_PARALLEL_TEST_LAUNCHER="$PWD/.venv-ag3s/bin/python $PWD/Baseline/paper_baselines/closed_loop/launch_method.py" \\
        benchmark/runners/run_parallel_gpu.sh --jobs outputs/eval/baselines-dev/jobs.txt --est-gib-per-run 16
    .venv-ag3s/bin/python -m Baseline.paper_baselines.closed_loop.jobs aggregate --plan outputs/eval/baselines-dev/plan.json
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Baseline.paper_baselines.closed_loop import arms as A  # noqa: E402
from benchmark.runners import eval_matrix as em  # noqa: E402
from benchmark.runners import suite_registry as reg  # noqa: E402

LAUNCHER_REL = "Baseline/paper_baselines/closed_loop/launch_method.py"
#: ``run_parallel_gpu.sh`` runs ``launch_libero_paired.py`` unless overridden; until it has a ``--launcher`` option the
#: override variable selects this launcher (lock / preflight / inherited-fd checks unchanged).
LAUNCHER_ENV = "AG3S_PARALLEL_TEST_LAUNCHER"
DEFAULT_STEP = "BASELINE-BENCH"
SMOKE = {"task_ids": [0], "episode_ids": [0], "split": "dev", "run_prefix": "blb-smoke", "step": "BASELINE-BENCH-SMOKE",
         "arms": "none,aegis,constrained_flow_matching"}
#: Suite families and how a job reaches them. ``libero`` / ``safelibero``: ``suite_registry`` + ``launch_method``
#: (``libero_safe_client.py``). ``libero_safety``: reserved -- fill in once the family's client / registry hook exists.
FAMILY_HOOKS = {"libero": "launch_method", "safelibero": "launch_method", "libero_safety": None}
#: Time model, s per env step (wait included) per arm: V5 pilot job wall time (SafeLIBERO spatial L I dev, 20 episodes,
#: one job = one arm) minus ~45 s server start, / (20 x 300 env steps). Episodes that succeed end early, so this is an
#: average over successes and failures, not an upper bound; other suites and the ``:gt`` arms are not timed.
TIMING = {"s_per_env_step": {"none": 0.2255, "aegis": 0.3015, "multilink_cbf": 0.3117, "knows": 0.3760,
                             "omniguide": 0.3672, "lan_o3dp_guidance_adapted": 0.4408, "nudge": 0.4813,
                             "constrained_flow_matching": 0.7717},
          "server_ready_s": 60.0,
          "source": "AG3S/ag3s/docs/handoff/CLOSED-LOOP-FIX-V5.verify.json numbers.gpu_time.pilot_job_wall_s"}


# --- selection ----------------------------------------------------------------------------------------------------
def parse_suites(text: str) -> list[tuple[str, str | None]]:
    """``eval_matrix.parse_suites`` plus the reserved ``libero_safety`` family (raises until its hook exists)."""
    items = [x.strip() for x in str(text).split(",") if x.strip()]
    for item in items:
        if item.startswith("libero_safety"):
            if FAMILY_HOOKS["libero_safety"] is None:
                raise NotImplementedError("LIBERO-Safety has no client / registry hook yet (FAMILY_HOOKS['libero_safety'])")
    return em.parse_suites(text)


#: ``<arm>@<preset>``: earlier behaviour of the methods, selected through ``launch_method`` arguments only.
#: ``v5`` = the CLOSED-LOOP-FIX V5 pilot (``3412861c``): box crop (``crop_mode="box"``), CFM without the action box,
#: no ``extras["suite"]`` (AEGIS workspace box off). Run tag ``<arm>-v5``.
PRESETS = {"v5": {"common": ["--no-suite-extras"],
                  "method_kwargs": {"aegis": {"crop_mode": "box"}, "knows": {"crop_mode": "box"},
                                    "multilink_cbf": {"crop_mode": "box"},
                                    "constrained_flow_matching": {"action_box": False}}}}


def preset_args(method: str, preset: str | None) -> list[str]:
    if preset is None:
        return []
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; available: {sorted(PRESETS)}")
    p = PRESETS[preset]
    out = list(p["common"]) + ["--tag", preset]
    kw = p["method_kwargs"].get(method)
    if kw:
        out += ["--method-kwargs", json.dumps(kw, sort_keys=True)]
    return out


def parse_arms(text: str) -> list[dict]:
    """``<arm>[:gt[-<source>]][@<preset>]`` comma list (``arms.arm_spec`` + :data:`PRESETS`)."""
    out, seen = [], set()
    for spec in [x.strip() for x in str(text).split(",") if x.strip()]:
        base, _, preset = spec.partition("@")
        a = A.arm_spec(base)
        if preset and a["gt"] is not None:
            raise ValueError(f"{spec!r}: presets reproduce earlier non-privileged runs; no --gt arm existed then")
        a["extra"] = preset_args(a["method"], preset or None)
        key = a["rec"]["variant"] + (f"@{preset}" if preset else "")
        if key not in seen:
            seen.add(key)
            out.append({"spec": spec, "key": key, **a})
    if not out:
        raise ValueError("no arms given")
    return out


def _ints(text):
    return [int(x) for x in str(text).replace(",", " ").split()] if text not in (None, "") else None


def _launcher():
    from Baseline.paper_baselines.closed_loop import launch_method as lm

    return lm


def job_run_id(prefix: str, suite: str, split: str, tasks: list[int]) -> str:
    """``<prefix>-<suite>-<split>-t<first>[-<last>]``; ``launch_method`` appends ``-L<level>-<arm tag>``."""
    t = f"t{tasks[0]}" if len(tasks) == 1 else f"t{tasks[0]}-{tasks[-1]}"
    return f"{prefix}-{suite}-{split}-{t}"


def estimate_job(suite: str, step_budget: str, n_episodes: int, method: str) -> dict:
    steps = reg.total_env_steps(suite, step_budget) * n_episodes
    s = TIMING["server_ready_s"] + steps * TIMING["s_per_env_step"][method]
    return {"env_steps_upper_bound": steps, "total_s": s}


def build_jobs(*, arms, suites, split, task_ids=None, episode_ids=None, tasks_per_job=0, seed=7, run_prefix,
               step=DEFAULT_STEP, step_budget="total_incl_wait", launcher_extra=()) -> tuple[list[dict], list[dict]]:
    """Job records + skipped (arm, suite) pairs. Every line goes through ``launch_method``'s parser and checks."""
    lm = _launcher()
    jobs, skipped = [], []
    for suite, level in suites:
        safe = reg.is_safelibero(suite)
        tasks = list(task_ids) if task_ids is not None else list(range(int(reg.suite(suite)["n_tasks"])))
        eps = list(episode_ids) if episode_ids is not None else reg.split_episodes(split)
        outside = sorted(set(eps) - set(reg.split_episodes(split)))
        if outside:
            raise ValueError(f"episodes {outside} are not in the {split} split")
        label = reg.split_label(suite, level, split)
        reg.check_selection(suite, level, tasks, eps, label)
        size = tasks_per_job or len(tasks)
        for arm in arms:
            if arm["rec"].get("privileged") and not safe:
                skipped.append({"arm": arm["spec"], "suite": suite, "level": level,
                                "reason": "privileged (--gt) arms need SafeLIBERO obstacle ground truth"})
                continue
            for i in range(0, len(tasks), size):
                group = tasks[i:i + size]
                run_id = job_run_id(run_prefix, suite, split, group)
                gt = [] if arm["gt"] is None else (["--gt"] if arm["gt"] == "default" else ["--gt", arm["gt"]])
                argv = ["--method", arm["method"]] + gt + ["--task-suite-name", suite] + (["--safety-level", level] if level else []) + [
                    "--task-ids", ",".join(map(str, group)), "--episode-ids", ",".join(map(str, eps)),
                    "--seed", str(seed), "--run-id", run_id, "--step", step, "--data-split", label]
                if step_budget != "total_incl_wait":
                    argv += ["--safelibero-step-budget", step_budget]
                argv += list(arm["extra"]) + list(launcher_extra)
                ns = lm.build_parser().parse_args(argv)  # the launcher's own parser and checks accept the line
                lm.check_selection(ns)
                rec = lm.resolve_arm(ns)
                jobs.append({"suite": suite, "level": level, "family": reg.suite(suite)["family"], "task_ids": group,
                             "episode_ids": eps, "split": split, "data_split": label, "seed": seed, "run_id": run_id,
                             "arm": arm["spec"], "method": arm["method"], "variant": arm["key"],
                             "privileged": bool(rec["privileged"]), "client_flags": rec.get("client_flags") or [],
                             "step_budget": step_budget, "argv": argv, "run_suffix": f"{run_id}-{lm.run_tag(ns)}",
                             "estimate_s": estimate_job(suite, step_budget, len(group) * len(eps), arm["method"])})
    # longest first: run_parallel_gpu.sh starts jobs in file order on <= 2 lanes
    jobs.sort(key=lambda j: -j["estimate_s"]["total_s"])
    return jobs, skipped


def schedule_estimate(jobs, max_parallel=2) -> dict:
    lanes = [0.0] * max_parallel
    for j in jobs:
        k = lanes.index(min(lanes))
        lanes[k] += j["estimate_s"]["total_s"]
    return {"jobs": len(jobs), "serial_s": sum(j["estimate_s"]["total_s"] for j in jobs), "max_parallel": max_parallel,
            "wall_s": max(lanes) if jobs else 0.0, "timing": TIMING}


def run_command(jobs_txt: str) -> str:
    return (f'{LAUNCHER_ENV}="$PWD/.venv-ag3s/bin/python $PWD/{LAUNCHER_REL}" '
            f"benchmark/runners/run_parallel_gpu.sh --jobs {jobs_txt} --max-parallel 2 --est-gib-per-run 16")


def _git(*a) -> str:
    return subprocess.run(["git", "-C", str(PROJECT_ROOT), *a], capture_output=True, text=True).stdout.strip()


def write_plan(out_dir: Path, jobs: list[dict], skipped: list[dict], meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = {"schema": 1, "meta": {**meta, "launcher": LAUNCHER_REL, "run": run_command(str(out_dir / "jobs.txt"))},
            "jobs": jobs, "skipped": skipped, "estimate": schedule_estimate(jobs)}
    est = plan["estimate"]
    lines = [f"# closed_loop.jobs {meta.get('command')} arms={meta.get('arms')} split={meta.get('split')} "
             f"head={meta.get('head')}",
             f"# {len(jobs)} jobs (launch_method.py argument lines); estimate {est['serial_s'] / 3600:.2f} h serial, "
             f"{est['wall_s'] / 3600:.2f} h at 2 parallel (V5 pilot time model)",
             f"# run: source scripts/setup/env.sh && {plan['meta']['run']}"]
    lines += [shlex.join(j["argv"]) for j in jobs]
    (out_dir / "jobs.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    return plan


# --- aggregation ----------------------------------------------------------------------------------------------------
def _rows(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines() if line.strip()]


def group_metrics(rows: list[dict], safelibero: bool) -> dict:
    """The ``closed_loop.metrics`` blocks over the rows of one suite x level x arm (several runs merged)."""
    from Baseline.paper_baselines.closed_loop import metrics as M

    out = {"n": len(rows), "libero": M.libero(rows),
           "wilson": em.condition_metrics([em.episode_view(r) for r in rows], safelibero)}
    if safelibero and rows:
        out.update(safelibero_official={k: v for k, v in M.safelibero_official(rows).items() if k != "t_per_episode"},
                   knows=M.knows(rows), multilink_11mm=M.multilink_11mm(rows), sl5c=M.sl5c(rows))
    return out


def aggregate(plan: dict, results_root: Path) -> dict:
    from Baseline.paper_baselines.closed_loop import metrics as M

    groups: dict = {}
    missing = []
    for job in plan["jobs"]:
        key = (job["suite"], job["level"], job["variant"])
        g = groups.setdefault(key, {"arm": job["arm"], "method": job["method"], "privileged": job["privileged"],
                                    "family": job["family"], "rows": {}, "runs": [], "method_calls": []})
        wanted = {(t, e) for t in job["task_ids"] for e in job["episode_ids"]}
        run = em.find_run(results_root, job["run_suffix"])
        if run is None:
            missing.append(job["run_suffix"])
            continue
        g["runs"].append(str(run))
        rows = {(int(r["task_id"]), int(r["episode_idx"])): r for r in _rows(run)}
        g["rows"].update({k: v for k, v in rows.items() if k in wanted})
        absent = sorted(wanted - set(rows))
        if absent:
            missing.append(f"{run.name}: episodes {absent}")
        mc = M.method_calls(run)
        if mc is not None:
            g["method_calls"].append({"run": run.name, **mc})
    out = {"plan_meta": plan.get("meta"), "missing": missing, "groups": [], "families": {}}
    fam_acc: dict = {}
    for (suite, level, variant), g in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        sl = reg.is_safelibero(suite)
        rows = [g["rows"][k] for k in sorted(g["rows"])]
        row = {"suite": suite, "level": level, "arm": g["arm"], "variant": variant, "method": g["method"],
               "privileged": g["privileged"], "runs": g["runs"], "metrics": group_metrics(rows, sl),
               "method_calls": g["method_calls"], "paired_vs_none": None}
        ref = groups.get((suite, level, "none"))
        if variant != "none" and ref and ref["rows"]:
            base = {k: em.episode_view(v) for k, v in ref["rows"].items()}
            new = {k: em.episode_view(v) for k, v in g["rows"].items()}
            p = {"official_success": em.paired_metrics(base, new, "official_success")}
            if sl:
                p["safe_success"] = em.paired_metrics(base, new, "safe_success")
                p["no_collision"] = em.paired_metrics({k: {"nc": not v["collided"]} for k, v in base.items()},
                                                      {k: {"nc": not v["collided"]} for k, v in new.items()}, "nc")
            row["paired_vs_none"] = p
        out["groups"].append(row)
        acc = fam_acc.setdefault((g["family"], variant), [])
        acc += rows
    for (fam, variant), rows in sorted(fam_acc.items()):
        out["families"].setdefault(fam, {})[variant] = group_metrics(rows, fam == "safelibero")
    budgets = sorted({j["step_budget"] for j in plan["jobs"]})
    out["notes"] = {"ets_budget": budgets,
                    "ets_comparability": "ETS / TSR equal main_aegis only with --safelibero-step-budget main_aegis; "
                                         "with total_incl_wait a failure ends at total - 20 policy steps",
                    "paired_vs_none": "same suite x level x (task, episode) of the plan's own none arm (re-run)"}
    return out


def markdown(agg: dict) -> str:
    def pct(x):
        return "-" if x is None else f"{100 * x:.1f}"

    def ci(x):
        return "" if not x else f" [{100 * x[0]:.1f}, {100 * x[1]:.1f}]"

    lines = ["| suite | level | arm | priv. | n | TSR / SR % | CAR % | SS % | ETS | Δ SS vs none (95 % CI) | McNemar p |"
             " fallbacks | server ms p50 |", "|" + "---|" * 13]
    for g in agg["groups"]:
        m = g["metrics"]
        o = m.get("safelibero_official") or {}
        w = m["wilson"]
        p = (g["paired_vs_none"] or {}).get("safe_success" if o else "official_success") or {}
        mc = g["method_calls"]
        fb = sum(int(x.get("fallbacks") or 0) for x in mc) if mc else None
        p50 = [x.get("server_ms_p50") for x in mc if x.get("server_ms_p50") is not None]
        lines.append("| {} | {} | {} | {} | {} | {}{} | {} | {}{} | {} | {}{} | {} | {} | {} |".format(
            g["suite"], g["level"] or "-", g["variant"], "yes" if g["privileged"] else "no", m["n"],
            pct(w["official_success_rate"]), ci(w["official_success_wilson95"]), pct(o.get("CAR")),
            pct(o.get("SS")), ci(w.get("safe_success_wilson95")) if o else "",
            "-" if o.get("ETS") is None else f"{o['ETS']:.1f}", pct(p.get("diff_rate")),
            ci(p.get("diff_bootstrap95")), "-" if not p else f"{p['mcnemar_exact_p']:.3g}",
            "-" if fb is None else fb, "-" if not p50 else f"{max(p50):.0f}"))
    return "\n".join(lines) + "\n"


# --- CLI ----------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--arms", required=True, help="comma list: none, <arm>, <arm>:gt, <arm>:gt-<source>")
    p.add_argument("--suites", default="all", help="all | libero | safelibero | suite[:I|II],...")
    p.add_argument("--split", required=True, choices=reg.SPLITS)
    p.add_argument("--tasks", default=None)
    p.add_argument("--episodes", default=None, help="subset of the split's episodes")
    p.add_argument("--tasks-per-job", type=int, default=0, help="0 = one job per arm x suite x level")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--step", default=DEFAULT_STEP)
    p.add_argument("--safelibero-step-budget", default="total_incl_wait", choices=("total_incl_wait", "main_aegis"))
    p.add_argument("--launcher-extra", default="", help="extra launch_method.py arguments for every job")
    p.add_argument("--allow-no-none", action="store_true", help="plan without the none arm (no pairing)")
    p.add_argument("--out-dir", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--arms", default=SMOKE["arms"])
    s.add_argument("--suites", default="all")
    s.add_argument("--safelibero-step-budget", default="total_incl_wait", choices=("total_incl_wait", "main_aegis"))
    s.add_argument("--run-prefix", default=SMOKE["run_prefix"])
    s.add_argument("--launcher-extra", default="")
    s.add_argument("--out-dir", required=True)
    a = sub.add_parser("aggregate")
    a.add_argument("--plan", required=True)
    a.add_argument("--results-root", default=None)
    a.add_argument("--out", default=None, help="default: <plan dir>/aggregate.json (+ .md)")
    args = ap.parse_args(argv)

    if args.command == "aggregate":
        plan = json.loads(Path(args.plan).read_text())
        from common.run_artifacts import results_root

        root = Path(args.results_root) if args.results_root else results_root()
        agg = aggregate(plan, root)
        out = Path(args.out) if args.out else Path(args.plan).with_name("aggregate.json")
        out.write_text(json.dumps(agg, indent=1, ensure_ascii=False, default=str))
        out.with_suffix(".md").write_text(markdown(agg))
        print(markdown(agg))
        if agg["missing"]:
            print("missing:", json.dumps(agg["missing"], indent=1), file=sys.stderr)
        return 0 if not agg["missing"] else 1

    arms = parse_arms(args.arms)
    if args.command == "plan" and "none" not in [x["rec"]["variant"] for x in arms] and not args.allow_no_none:
        raise ValueError("the plan has no 'none' arm (VLA-only is re-run in every plan); add it or --allow-no-none")
    if args.command == "smoke":
        jobs, skipped = build_jobs(arms=arms, suites=parse_suites(args.suites), split=SMOKE["split"],
                                   task_ids=SMOKE["task_ids"], episode_ids=SMOKE["episode_ids"],
                                   run_prefix=args.run_prefix + ("" if args.safelibero_step_budget == "total_incl_wait"
                                                                 else "-aegis"),
                                   step=SMOKE["step"], step_budget=args.safelibero_step_budget,
                                   launcher_extra=shlex.split(args.launcher_extra))
        meta = {"command": "smoke", **SMOKE, "arms": args.arms, "step_budget": args.safelibero_step_budget}
    else:
        jobs, skipped = build_jobs(arms=arms, suites=parse_suites(args.suites), split=args.split,
                                   task_ids=_ints(args.tasks), episode_ids=_ints(args.episodes),
                                   tasks_per_job=args.tasks_per_job, seed=args.seed, run_prefix=args.run_prefix,
                                   step=args.step, step_budget=args.safelibero_step_budget,
                                   launcher_extra=shlex.split(args.launcher_extra))
        meta = {"command": "plan", "arms": args.arms, "split": args.split, "step_budget": args.safelibero_step_budget}
    meta["head"] = _git("rev-parse", "HEAD")
    plan = write_plan(Path(args.out_dir), jobs, skipped, meta)
    est = plan["estimate"]
    print(f"{Path(args.out_dir) / 'jobs.txt'}: {est['jobs']} jobs ({len(skipped)} arm x suite skipped), "
          f"{est['serial_s'] / 3600:.2f} h serial, {est['wall_s'] / 3600:.2f} h at 2 parallel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
