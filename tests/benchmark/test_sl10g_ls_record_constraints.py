"""SL10g: ``launch_libero_safety_paired.py --record-constraints {none,occupancy,full}`` (CPU, no GPU, no server).

The LIBERO-Safety wrapper hard-coded ``record_constraints = "none"``. It now exposes the shared launcher's option
(``launch_libero_paired.record_constraints_argv`` / ``record_constraints_summary`` / ``run_mode``). With ``none`` (default)
the dry-run output and the server argv must stay byte-identical to the wrapper at the base commit ``541d0bc8``.
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
BASE_COMMIT = "541d0bc8"  # wrapper before --record-constraints
WRAPPER = "benchmark/runners/launch_libero_safety_paired.py"
PC = "AG3S-final-panda-v3-e0to-pc"
BASE_ARGV = ["--task-suite-name", "human_safety", "--level", "0", "--task-ids", "0", "--episode-ids", "0",
             "--run-id", "x", "--policy-checkpoint", "pi05_libero", "--conversion", "B", "--ag3s-version", PC]


@pytest.fixture
def lsp(monkeypatch):
    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_pi05_only as l0
    from benchmark.runners import launch_libero_safety_paired as mod

    # install_hooks() swaps module attributes of lp / l0 for the whole process: restore them after each test
    for obj, names in ((lp, ("CLIENT_SCRIPT", "client_argv", "protocol_settings", "runtime_schedule", "server_argv")),
                       (l0, ("libero_python",))):
        for n in names:
            monkeypatch.setattr(obj, n, getattr(obj, n))
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(ROOT))
    # the vendored LIBERO-Safety env_wrapper.py (control frequency) is not in every worktree: the value is not under test
    monkeypatch.setattr(l0, "libero_control_freq", lambda path=None: 20)
    monkeypatch.setattr(mod, "checkpoint_identity", lambda name, require=True: {"path": "/ckpt", "policy_checkpoint": name})
    return mod


def _wrapper_at_base(lsp_mod):
    """The wrapper source at the base commit as its own module object (same shared ``lp`` / ``l0``)."""
    try:
        src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:{WRAPPER}"], text=True,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {BASE_COMMIT} not available in this checkout")
    spec = importlib.util.spec_from_loader("_ls_wrapper_base", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(ROOT / WRAPPER)
    exec(compile(src, str(ROOT / WRAPPER), "exec"), mod.__dict__)
    mod.checkpoint_identity = lsp_mod.checkpoint_identity
    return mod


def _dry(mod, capsys, *extra):
    assert mod.main(BASE_ARGV + list(extra) + ["--dry-run"]) == 0
    return capsys.readouterr().out


# ---- parser ----------------------------------------------------------------------------------------------------

def test_parser_accepts_the_option_with_the_launcher_choices(lsp):
    from benchmark.runners import launch_libero_paired as lp

    p = lsp.build_parser()
    assert p.parse_args(BASE_ARGV).record_constraints == "none"
    for choice in lp.RECORD_CONSTRAINTS_CHOICES:
        assert p.parse_args(BASE_ARGV + ["--record-constraints", choice]).record_constraints == choice
    with pytest.raises(SystemExit):
        p.parse_args(BASE_ARGV + ["--record-constraints", "bogus"])
    action = next(a for a in p._actions if a.dest == "record_constraints")      # noqa: SLF001
    assert tuple(action.choices) == lp.RECORD_CONSTRAINTS_CHOICES and action.default == "none"


# ---- none (default): identical to the wrapper at the base commit -----------------------------------------------

@pytest.mark.parametrize("extra", [[], ["--modes", "vla_v11ab"], ["--modes", "vla_only"]])
def test_default_dry_run_byte_identical_to_base(lsp, capsys, extra):
    old = _wrapper_at_base(lsp)
    out_old = _dry(old, capsys, *extra)
    out_new = _dry(lsp, capsys, *extra)
    assert out_new == out_old
    assert "--record-constraints" not in out_new


def test_default_none_adds_no_flag_and_no_settings_key(lsp, monkeypatch, tmp_path):
    seen = _run_mode(lsp, monkeypatch, tmp_path, [])
    assert seen["popen"] == seen["settings_server_argv"]
    assert "--record-constraints" not in seen["popen"] and "record_constraints" not in seen["settings"]
    assert all("record_constraints" not in e for e in seen["executions"])


# ---- full / occupancy: the LIBERO launcher's record argv ---------------------------------------------------------

@pytest.mark.parametrize("record", ["occupancy", "full"])
def test_dry_run_appends_the_libero_launcher_record_argv(lsp, capsys, record):
    from benchmark.runners import launch_libero_paired as lp

    old = _wrapper_at_base(lsp)
    base = json.loads(_dry(old, capsys, "--modes", "vla_v11ab").split("\n[", 1)[0].replace("\n  ", ""))
    out = _dry(lsp, capsys, "--modes", "vla_v11ab", "--record-constraints", record)
    got = json.loads(out.split("\n[", 1)[0].replace("\n  ", ""))
    assert got["vla_v11ab"] == base["vla_v11ab"] + lp.record_constraints_argv("vla_v11ab", lp.RUN_DIR_TOKEN, record)
    assert got["vla_v11ab"][-4:] == ["--record-constraints", "<run_dir>/server_constraints",
                                     "--record-constraints-esdf", record]


def test_run_mode_full_appends_resolved_argv_settings_and_chunk_summary(lsp, monkeypatch, tmp_path):
    seen = _run_mode(lsp, monkeypatch, tmp_path, ["--record-constraints", "full"], chunks=3)
    run_dir = seen["run_dir"]
    tail = ["--record-constraints", f"{run_dir}/server_constraints", "--record-constraints-esdf", "full"]
    assert seen["popen"][-4:] == tail and seen["popen"][:-4] == seen["settings_server_argv"][:-4]
    assert seen["settings_server_argv"][-4:] == ["--record-constraints", "<run_dir>/server_constraints",
                                                 "--record-constraints-esdf", "full"]
    assert seen["settings"]["record_constraints"]["esdf"] == "full"
    server = next(e for e in seen["executions"] if e.get("role") == "server")
    assert server["argv"] == seen["popen"]
    result = next(e for e in seen["executions"] if e.get("role") == "launcher_result")
    assert result["record_constraints"]["n_chunk_files"] == 3 and result["record_constraints"]["esdf"] == "full"
    assert result["record_constraints"]["writer_runs"] == ["run_0000"]
    assert (pathlib.Path(run_dir) / "server_constraints" / "run_0000" / "chunk_00000.npz").exists()


# ---- refusals: whatever the LIBERO launcher refuses is refused here ----------------------------------------------

@pytest.mark.parametrize("record", ["occupancy", "full"])
def test_vla_only_mode_is_refused_with_recording(lsp, record):
    with pytest.raises(ValueError, match="vla_only"):
        lsp.main(BASE_ARGV + ["--record-constraints", record, "--dry-run"])                 # default: both modes
    with pytest.raises(ValueError, match="vla_only"):
        lsp.main(BASE_ARGV + ["--record-constraints", record, "--modes", "vla_only", "--dry-run"])


def test_launcher_helper_refuses_non_server_mode_and_unknown_value(lsp):
    from benchmark.runners import launch_libero_paired as lp

    with pytest.raises(ValueError):
        lp.record_constraints_argv("vla_only", "/r", "full")
    with pytest.raises(ValueError):
        lp.record_constraints_argv("vla_v11ab", "/r", "bogus")


# ---- run_mode harness (every side effect faked) ------------------------------------------------------------------

def _run_mode(lsp, monkeypatch, tmp_path, extra, *, chunks=0):
    import common.run_artifacts as ra
    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_pi05_only as l0
    from benchmark.runners import libero_safety_client as lsc

    args = lsp.build_parser().parse_args(BASE_ARGV + list(extra))
    for name, value in (("resize_size", 224), ("replan_steps", 5), ("num_steps_wait", lsc.NUM_STEPS_WAIT)):
        setattr(args, name, value)
    args.safety_level = None
    args.data_split = lsp.SPLIT_TEXT[args.data_split]
    lsp.install_hooks(args)
    run_dir = tmp_path / "run-ls"
    (run_dir / "logs").mkdir(parents=True)
    seen = {"executions": [], "run_dir": str(run_dir)}

    def create_run(**kw):
        seen["settings"] = json.loads(json.dumps(kw["settings"], default=str))
        seen["settings_server_argv"] = seen["settings"]["server_argv"]
        return run_dir

    def popen(argv, **kw):
        seen["popen"] = list(argv)
        for i in range(chunks):   # plays a server whose ConstraintRecordWriter wrote `chunks` files
            d = run_dir / lp.RECORD_CONSTRAINTS_DIRNAME / "run_0000"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"chunk_{i:05d}.npz").write_bytes(b"x" * 10)
        return SimpleNamespace(pid=4242)

    monkeypatch.setattr(ra, "create_run", create_run)
    monkeypatch.setattr(ra, "record_execution",
                        lambda run, *, settings, checkpoint=None: seen["executions"].append(json.loads(json.dumps(settings))))
    monkeypatch.setattr(l0, "choose_port", lambda requested=None: 23456)
    monkeypatch.setattr(l0, "openpi_python", lambda: "/fake/openpi/python")
    monkeypatch.setattr(l0, "wait_ready", lambda server, port, timeout: 1.0)
    monkeypatch.setattr(l0, "client_environment", lambda: {})
    monkeypatch.setattr(l0, "stop_own_process", lambda server: 0)
    monkeypatch.setattr(lp, "protocol_settings", lambda a: {"p": 1})
    monkeypatch.setattr(lp, "runtime_schedule", lambda a: {"r": 1})
    monkeypatch.setattr(lp, "gpu_compute_apps", lambda: [])
    monkeypatch.setattr(lp, "GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.delenv("AG3S_GPU_LOCK_HELD", raising=False)
    monkeypatch.setattr(lp.subprocess, "Popen", popen)
    monkeypatch.setattr(lp.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    lp.run_mode("vla_v11ab", args, {"path": "/ckpt"}, {"versions": {"v": 1}})
    return seen
