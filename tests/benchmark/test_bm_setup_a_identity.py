"""BM-SETUP-A: the shared runner opened to every suite must leave the existing (spatial) behaviour byte-identical.

Reference = the launcher / client at ``2c58cee1`` (HEAD before BM-SETUP-A). Compared with every default:
server argv, client argv, protocol / runtime-schedule settings, the ``create_run`` kwargs, the started server argv and
the execution records of ``run_mode``, the dry-run output (launcher, Python 3.11) and the client's wiring
(``max_steps``, extra jsonl fields, the parsed arguments) for LIBERO Spatial and SafeLIBERO Spatial (Python 3.8).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PRE = "2c58cee11e57d0969f8909eae4a939e866718dad"
LAUNCHER = "benchmark/runners/launch_libero_paired.py"
PY38 = ROOT / ".venv-libero/bin/python"

CASES = {
    "libero_spatial": ["--task-ids", "0,1", "--episode-ids", "0,1,2", "--run-id", "x", "--conversion", "C"],
    "libero_spatial_seg": ["--task-ids", "3", "--episode-ids", "4", "--run-id", "x", "--conversion", "B",
                           "--capture-segmentation", "--ag3s-version", "V11AB-pb7", "--seed", "11"],
    "safelibero_spatial_I": ["--task-suite-name", "safelibero_spatial", "--safety-level", "I", "--task-ids", "0,1,2,3",
                             "--episode-ids", "0,1,2,3,4", "--run-id", "x", "--conversion", "C", "--data-split",
                             "dev_SafeLIBERO_Spatial_ep0-4_user_2026-10-14", "--ag3s-version", "V11AB-pb5"],
    "safelibero_spatial_II": ["--task-suite-name", "safelibero_spatial", "--safety-level", "II", "--task-ids", "2",
                              "--episode-ids", "1", "--run-id", "x", "--conversion", "A", "--data-split", "d"],
}


def _module_at(commit: str, name: str):
    try:
        src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{LAUNCHER}"], text=True,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {commit} not available in this checkout")
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(ROOT / LAUNCHER)
    sys.modules[name] = mod
    try:
        exec(compile(src, str(ROOT / LAUNCHER), "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


@pytest.fixture
def lp():
    from benchmark.runners import launch_libero_paired as lp

    return lp


@pytest.fixture
def lp_old():
    return _module_at(PRE, "_launch_libero_paired_pre_bm_setup_a")


def _args(mod, argv):
    a = mod.build_parser().parse_args(argv)
    for name, value in (("resize_size", 224), ("replan_steps", 5), ("num_steps_wait", 10)):
        setattr(a, name, getattr(a, name, value))
    return a


@pytest.mark.parametrize("case", sorted(CASES))
def test_argv_and_settings_identical(lp, lp_old, case):
    new, old = _args(lp, CASES[case]), _args(lp_old, CASES[case])
    assert {k: v for k, v in vars(new).items() if k != "safelibero_step_budget"} == vars(old)
    assert new.safelibero_step_budget == "total_incl_wait"
    for mode in ("vla_only", "vla_v11ab"):
        assert lp.server_argv(mode, "/ckpt", 7, new.ag3s_version) == lp_old.server_argv(mode, "/ckpt", 7, old.ag3s_version)
        assert lp.run_tag(mode, new) == _old_tag(mode, old)
    assert lp.client_argv("/run", 7, new) == lp_old.client_argv("/run", 7, old)
    dump = lambda x: json.dumps(x, default=str).encode()  # noqa: E731
    assert dump(lp.protocol_settings(new)) == dump(lp_old.protocol_settings(old))
    assert dump(lp.runtime_schedule(new)) == dump(lp_old.runtime_schedule(old))


def _old_tag(mode, args):  # the inline tag code of run_mode at 2c58cee1
    tag = mode
    if mode == "vla_v11ab":
        tag = f"{mode}-{args.conversion}" if args.ag3s_version == "V11AB" else f"{mode}-{args.ag3s_version}-{args.conversion}"
    if args.safety_level:
        tag = f"L{args.safety_level}-{tag}"
    return tag


def _run_mode(monkeypatch, mod, tmp_path, mode, args):
    """run_mode with every side effect faked (protocol / runtime settings are the real ones)."""
    import common.run_artifacts as ra

    run_dir = tmp_path / f"run-{mod.__name__}-{mode}"
    (run_dir / "logs").mkdir(parents=True)
    seen = {"create": None, "executions": [], "popen": None}

    def create_run(**kw):
        seen["create"] = kw
        return run_dir

    def popen(argv, **kw):
        seen["popen"] = list(argv)
        return SimpleNamespace(pid=4242)

    monkeypatch.setattr(ra, "create_run", create_run)
    monkeypatch.setattr(ra, "record_execution",
                        lambda run, *, settings, checkpoint=None: seen["executions"].append(json.loads(json.dumps(settings))))
    monkeypatch.setattr(mod.l0, "choose_port", lambda requested=None: 23456)
    monkeypatch.setattr(mod.l0, "openpi_python", lambda: "/fake/openpi/python")
    monkeypatch.setattr(mod.l0, "libero_python", lambda: "/fake/libero/python")
    monkeypatch.setattr(mod.l0, "wait_ready", lambda server, port, timeout: 1.0)
    monkeypatch.setattr(mod.l0, "client_environment", lambda: {})
    monkeypatch.setattr(mod.l0, "stop_own_process", lambda server: 0)
    monkeypatch.setattr(mod, "gpu_compute_apps", lambda: [])
    monkeypatch.setattr(mod, "GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda s: None,
                                                     gmtime=lambda: None, strftime=lambda fmt, t=None: "<utc>"))
    monkeypatch.delenv("AG3S_GPU_LOCK_HELD", raising=False)
    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(STDOUT=subprocess.STDOUT, Popen=popen,
                                                           run=lambda *a, **k: SimpleNamespace(returncode=0)))
    mod.run_mode(mode, args, {"path": "/ckpt"}, {"versions": {"v": 1}})
    return seen, run_dir


@pytest.mark.parametrize("mode", ["vla_only", "vla_v11ab"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_run_mode_records_identical(monkeypatch, tmp_path, lp, lp_old, case, mode):
    new, new_dir = _run_mode(monkeypatch, lp, tmp_path, mode, _args(lp, CASES[case]))
    old, old_dir = _run_mode(monkeypatch, lp_old, tmp_path, mode, _args(lp_old, CASES[case]))

    def dump(x, run_dir):
        return json.dumps(x).replace(str(run_dir), "<R>").encode()

    assert dump(new["popen"], new_dir) == dump(old["popen"], old_dir)
    assert dump(new["create"], new_dir) == dump(old["create"], old_dir)
    assert dump(new["executions"], new_dir) == dump(old["executions"], old_dir)


@pytest.mark.parametrize("case", sorted(CASES))
def test_dry_run_output_identical(monkeypatch, capsys, lp, lp_old, case):
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(ROOT))
    outs = []
    for mod in (lp, lp_old):
        monkeypatch.setattr(mod.l0, "checkpoint_identity", lambda path, require=True: {"path": "/ckpt"})
        monkeypatch.setattr(mod.l0, "libero_python", lambda: "/fake/libero/python")
        assert mod.main(CASES[case] + ["--dry-run"]) == 0
        outs.append(capsys.readouterr().out.encode())
    assert outs[0] == outs[1]


def test_main_aegis_budget_is_opt_in(lp):
    sl = _args(lp, CASES["safelibero_spatial_I"] + ["--safelibero-step-budget", "main_aegis"])
    argv = lp.client_argv("/run", 7, sl)
    assert argv[-2:] == ["--safelibero-step-budget", "main_aegis"]
    p = lp.protocol_settings(sl)
    assert p["max_steps"] == 300 and p["num_steps_wait"] == 20 and p["safelibero_step_budget"]["budget"] == "main_aegis"
    long_ = _args(lp, ["--task-suite-name", "safelibero_long", "--safety-level", "I", "--task-ids", "0", "--episode-ids",
                       "0", "--run-id", "x", "--conversion", "C", "--data-split", "d",
                       "--safelibero-step-budget", "main_aegis"])
    assert lp.protocol_settings(long_)["max_steps"] == 550


@pytest.mark.parametrize("argv, match", [
    (["--task-suite-name", "libero_goal", "--safelibero-step-budget", "main_aegis"], "SafeLIBERO suites only"),
    (["--task-suite-name", "safelibero_object", "--safety-level", "I", "--task-ids", "4", "--data-split", "d"], "task ids"),
    (["--task-suite-name", "libero_10", "--episode-ids", "50", "--data-split", "d"], "episode ids"),
    (["--task-suite-name", "libero_object", "--data-split", "dev (LIBERO Goal init 0-4, user 2026-10-23)"], "belongs to"),
    (["--task-suite-name", "safelibero_goal", "--safety-level", "II", "--episode-ids", "5",
      "--data-split", "dev_SafeLIBERO_Goal_ep0-4_user_2026-10-23"], "not in the dev split"),
])
def test_bad_selections_are_refused_before_the_checkpoint(monkeypatch, lp, argv, match):
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(ROOT))
    monkeypatch.setattr(lp.l0, "checkpoint_identity", lambda *a, **k: pytest.fail("refusal must come first"))
    base = {"--task-ids": "0", "--episode-ids": "0"}
    full = list(argv)
    for k, v in base.items():
        if k not in full:
            full += [k, v]
    with pytest.raises(ValueError, match=match):
        lp.main(full + ["--run-id", "x", "--conversion", "C"])


# --- client (Python 3.8): wiring of main() before / after, every default --------------------------------------------
_CLIENT_PROBE = r'''
import importlib.util, json, pathlib, subprocess, sys, types
root = pathlib.Path.cwd()
sys.path.insert(0, str(root / "benchmark/runners"))
which, argv = sys.argv[1], json.loads(sys.argv[2])
import libero_pi05_client as base
if which == "old":
    src = subprocess.check_output(["git", "show", "%s:benchmark/runners/libero_safe_client.py"], text=True)
    mod = types.ModuleType("libero_safe_client_old"); mod.__file__ = str(root / "benchmark/runners/libero_safe_client.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
else:
    import libero_safe_client as mod
seen = {}
base.validate_run_dir = lambda p: pathlib.Path(sys.argv[3])
base.make_policy_client = lambda h, p: (object(), "fake")
def run_tasks(args, run_dir, jsonl, task_suite, max_steps, policy_for, observer_for=None, after_episode=None):
    client, connector, extra = policy_for(args.task_ids[0], args.episode_ids[0])
    seen.update({"max_steps": max_steps, "extra": extra, "n_tasks": task_suite.n_tasks,
                 "args": {k: v for k, v in sorted(vars(args).items()) if k not in ("safelibero_step_budget", "emit_privileged_obstacles")},
                 "collision": None if client.collision is None else client.collision.wait,
                 "policy_seed": client.seed})
    return 0
base.run_tasks = run_tasks
rc = mod.main(argv)
print(json.dumps({"rc": rc, **seen}, default=str))
''' % PRE


def _client(which: str, argv: list[str], tmp_path) -> dict:
    if not PY38.exists():
        pytest.skip(".venv-libero not installed")
    from benchmark.runners import launch_libero_pi05_only as l0

    run = tmp_path / which
    run.mkdir()
    r = subprocess.run([str(PY38), "-c", _CLIENT_PROBE, which, json.dumps(argv), str(run)], cwd=str(ROOT),
                       capture_output=True, text=True, env=l0.client_environment(), timeout=600)
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    assert r.returncode == 0 and lines, r.stdout[-2000:] + r.stderr[-3000:]
    return json.loads(lines[-1])


@pytest.mark.parametrize("case", ["libero_spatial", "safelibero_spatial_II"])
def test_client_wiring_identical(tmp_path, lp, case):
    argv = lp.client_argv("/run", 7, _args(lp, CASES[case]))[2:]  # drop python + script
    new, old = _client("new", argv, tmp_path), _client("old", argv, tmp_path)
    assert json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)
    assert new["max_steps"] == (220 if case == "libero_spatial" else 280)


def test_client_main_aegis_budget(tmp_path, lp):
    argv = lp.client_argv("/run", 7, _args(lp, CASES["safelibero_spatial_II"] +
                                           ["--safelibero-step-budget", "main_aegis"]))[2:]
    out = _client("new", argv, tmp_path)
    assert out["max_steps"] == 300 and out["collision"] == 20
    assert out["extra"]["safelibero_step_budget"]["policy_steps"] == 300
