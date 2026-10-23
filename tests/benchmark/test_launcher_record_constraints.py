"""SL5j-launcher: ``launch_libero_paired.py --record-constraints {none,occupancy,full}`` (CPU, no GPU, no server).

The launcher only orchestrates: with ``none`` (default) the server argv, the manifest settings and the dry-run
output must stay byte-identical to the launcher before SL5j (commit ``32a4ca8f``); otherwise two serve_safe flags
are appended to the VLA+V11AB server argv under the run directory, and the ``vla_only`` mode is refused.
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PRE_SL5J = "32a4ca8f0e9c0bcefbf9c61e8073c48af8b73b46"  # launcher before --record-constraints
LAUNCHER = "benchmark/runners/launch_libero_paired.py"
BASE_ARGV = ["--task-ids", "0", "--episode-ids", "0", "--run-id", "x", "--conversion", "C"]


def _module_at(commit: str, name: str):
    """The launcher module as it was at ``commit`` (its own module object; ``l0`` etc. are the current ones)."""
    try:
        src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{LAUNCHER}"], text=True,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"commit {commit} not available in this checkout")
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(ROOT / LAUNCHER)  # PROJECT_ROOT / CLIENT_SCRIPT resolve as for the real file
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
    return _module_at(PRE_SL5J, "_launch_libero_paired_pre_sl5j")


def _args(lp, *extra):
    a = lp.build_parser().parse_args(BASE_ARGV + list(extra))
    for name, value in (("resize_size", 224), ("replan_steps", 5), ("num_steps_wait", 10)):
        setattr(a, name, getattr(a, name, value))
    return a


def _run_mode(monkeypatch, mod, tmp_path, mode, args, *, chunks=0):
    """run_mode with every side effect faked; returns (create_run settings, executions, Popen argv)."""
    import common.run_artifacts as ra

    run_dir = tmp_path / f"run-{mod.__name__}-{mode}"
    (run_dir / "logs").mkdir(parents=True)
    seen = {"create": None, "executions": [], "popen": None}

    def create_run(**kw):
        seen["create"] = kw
        return run_dir

    def popen(argv, **kw):
        seen["popen"] = list(argv)
        for i in range(chunks):  # plays a server whose ConstraintRecordWriter wrote `chunks` files
            d = run_dir / mod.RECORD_CONSTRAINTS_DIRNAME / "run_0000"
            d.mkdir(parents=True, exist_ok=True)
            (d / f"chunk_{i:05d}.npz").write_bytes(b"x" * 10)
        if chunks:
            (run_dir / mod.RECORD_CONSTRAINTS_DIRNAME / "run_0000" / "meta.json").write_text("{}")
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
    monkeypatch.setattr(mod, "protocol_settings", lambda a: {"p": 1})
    monkeypatch.setattr(mod, "runtime_schedule", lambda a: {"r": 1})
    monkeypatch.setattr(mod, "gpu_compute_apps", lambda: [])
    monkeypatch.setattr(mod, "GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=lambda: 100.0, sleep=lambda s: None,  # fixed clock
                                                     gmtime=lambda: None, strftime=lambda fmt, t=None: "<utc>"))
    monkeypatch.delenv("AG3S_GPU_LOCK_HELD", raising=False)
    monkeypatch.setattr(mod, "subprocess", SimpleNamespace(STDOUT=subprocess.STDOUT, Popen=popen,
                                                           run=lambda *a, **k: SimpleNamespace(returncode=0)))
    mod.run_mode(mode, args, {"path": "/ckpt"}, {"versions": {"v": 1}})
    return seen, run_dir


# ---- none (default): byte-identical to the launcher before SL5j ---------------------------------------------

def test_default_is_none(lp):
    assert _args(lp).record_constraints == "none"
    assert lp.record_constraints_argv("vla_v11ab", "/r", "none") == []
    assert lp.record_constraints_argv("vla_only", "/r", "none") == []


@pytest.mark.parametrize("mode", ["vla_only", "vla_v11ab"])
@pytest.mark.parametrize("version", ["V11AB", "V11AB-pb5", "V11AB-pb7"])
def test_default_server_argv_and_settings_byte_identical_to_pre_sl5j(monkeypatch, tmp_path, lp, lp_old, mode, version):
    new, new_dir = _run_mode(monkeypatch, lp, tmp_path, mode, _args(lp, "--ag3s-version", version))
    old, old_dir = _run_mode(monkeypatch, lp_old, tmp_path, mode, _args(lp_old, "--ag3s-version", version))

    def dump(x, run_dir):  # the two fake run directories differ only in name; nothing else may differ
        return json.dumps(x, sort_keys=False).replace(str(run_dir), "<R>").encode()

    assert dump(new["popen"], new_dir) == dump(old["popen"], old_dir)       # what the server is started with
    assert dump(new["create"], new_dir) == dump(old["create"], old_dir)     # manifest settings (incl. server_argv)
    assert len(new["executions"]) == len(old["executions"]) == 3            # server, client, launcher_result
    assert dump(new["executions"], new_dir) == dump(old["executions"], old_dir)
    assert "--record-constraints" not in new["popen"] and "record_constraints" not in new["create"]["settings"]


def test_default_dry_run_output_byte_identical_to_pre_sl5j(monkeypatch, capsys, lp, lp_old):
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(ROOT))
    outs = []
    for mod in (lp, lp_old):
        monkeypatch.setattr(mod.l0, "checkpoint_identity", lambda path, require=True: {"path": "/ckpt"})
        monkeypatch.setattr(mod.l0, "libero_python", lambda: "/fake/libero/python")
        assert mod.main(BASE_ARGV + ["--dry-run", "--ag3s-version", "V11AB-pb7"]) == 0
        outs.append(capsys.readouterr().out.encode())
    assert outs[0] == outs[1]


# ---- occupancy / full: two flags under the run directory ------------------------------------------------------

@pytest.mark.parametrize("record", ["occupancy", "full"])
def test_record_flags_are_appended_under_the_run_dir(monkeypatch, tmp_path, lp, record):
    args = _args(lp, "--record-constraints", record, "--modes", "vla_v11ab", "--ag3s-version", "V11AB-pb7")
    seen, run_dir = _run_mode(monkeypatch, lp, tmp_path, "vla_v11ab", args, chunks=3)
    base = ["/fake/openpi/python", "-m", "AG3S.trajopt.serve_safe"] + lp.server_argv("vla_v11ab", "/ckpt", 23456,
                                                                                     "V11AB-pb7")
    tail = ["--record-constraints", f"{run_dir}/server_constraints", "--record-constraints-esdf", record]
    assert seen["popen"] == base + tail                                       # appended at the very end, resolved
    assert seen["executions"][0]["argv"] == base + tail                       # role=server entry = real argv
    settings = seen["create"]["settings"]
    assert settings["server_argv"] == base + ["--record-constraints", "<run_dir>/server_constraints",
                                              "--record-constraints-esdf", record]
    assert settings["record_constraints"]["esdf"] == record
    assert settings["record_constraints"]["dir"] == "<run_dir>/server_constraints"
    result = seen["executions"][-1]
    assert result["role"] == "launcher_result"
    rc = result["record_constraints"]
    assert (rc["esdf"], rc["n_chunk_files"], rc["bytes"], rc["writer_runs"], rc["meta_json"]) == \
        (record, 3, 30, ["run_0000"], ["run_0000"])
    assert rc["dir"] == str(run_dir / "server_constraints")


def test_record_flags_parse_in_serve_safe(lp):
    from AG3S.adapters.panda import profile
    from AG3S.trajopt import serve_safe as ss

    argv = lp.server_argv("vla_v11ab", "/ckpt", 1, "V11AB-pb7") + lp.record_constraints_argv(
        "vla_v11ab", "/r", "occupancy")
    args = ss.build_parser().parse_args(argv)
    ss.reject_bad_flag_combinations(ss.build_parser(), args)
    assert (args.record_constraints, args.record_constraints_esdf) == ("/r/server_constraints", "occupancy")
    # every V11AB-pb7 flag is still there and unchanged
    assert profile.v11ab_build_kwargs(args) == profile.v11ab_build_kwargs(
        ss.build_parser().parse_args(lp.server_argv("vla_v11ab", "/ckpt", 1, "V11AB-pb7")))


def test_no_chunk_file_is_reported(monkeypatch, tmp_path, capsys, lp):
    args = _args(lp, "--record-constraints", "full", "--modes", "vla_v11ab")
    seen, _ = _run_mode(monkeypatch, lp, tmp_path, "vla_v11ab", args, chunks=0)
    assert seen["executions"][-1]["record_constraints"]["n_chunk_files"] == 0
    assert "no chunk file" in capsys.readouterr().out


def test_dry_run_shows_the_record_flags(monkeypatch, capsys, lp):
    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(ROOT))
    monkeypatch.setattr(lp.l0, "checkpoint_identity", lambda path, require=True: {"path": "/ckpt"})
    monkeypatch.setattr(lp.l0, "libero_python", lambda: "/fake/libero/python")
    assert lp.main(BASE_ARGV + ["--dry-run", "--modes", "vla_v11ab", "--record-constraints", "full"]) == 0
    first = capsys.readouterr().out.split("\n}\n")[0] + "\n}"
    argv = json.loads(first)["vla_v11ab"]
    assert argv[-4:] == ["--record-constraints", "<run_dir>/server_constraints", "--record-constraints-esdf", "full"]


# ---- vla_only refused -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("modes", [None, "vla_only", "vla_only,vla_v11ab", "vla_v11ab,vla_only"])
def test_vla_only_mode_is_refused_before_any_run(monkeypatch, lp, modes):
    import common.run_artifacts as ra

    monkeypatch.setenv("AG3S_PROJECT_ROOT", str(ROOT))
    monkeypatch.setattr(lp.l0, "checkpoint_identity",
                        lambda path, require=True: pytest.fail("refusal must come before the checkpoint check"))
    monkeypatch.setattr(ra, "create_run", lambda **kw: pytest.fail("no run may be allocated after a refusal"))
    extra = ["--record-constraints", "occupancy"] + (["--modes", modes] if modes else [])
    with pytest.raises(ValueError, match="vla_only"):
        lp.main(BASE_ARGV + extra)


def test_vla_only_argv_is_refused_at_the_helper(lp):
    with pytest.raises(ValueError, match="vla_v11ab server only"):
        lp.record_constraints_argv("vla_only", "/r", "full")
    with pytest.raises(ValueError):
        lp.record_constraints_argv("vla_v11ab", "/r", "dense")


def test_bad_choice_is_rejected_by_the_parser(lp):
    with pytest.raises(SystemExit):
        lp.build_parser().parse_args(BASE_ARGV + ["--record-constraints", "dense"])


# ---- run_parallel_gpu.sh passes the flag through unchanged ----------------------------------------------------

def test_parallel_script_passes_the_flag_through(tmp_path):
    fake = tmp_path / "fake_launcher.py"
    fake.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\nexit 0\n")
    smi.chmod(0o755)
    cg = tmp_path / "cgroup"
    cg.mkdir()
    (cg / "memory.current").write_text(str(1 << 30) + "\n")
    (cg / "memory.stat").write_text("inactive_file 0\n")
    line = "--task-ids 0 --episode-ids 0 --run-id j --conversion C --modes vla_v11ab --record-constraints occupancy"
    (tmp_path / "jobs.txt").write_text(line + "\n")
    import os
    env = {**os.environ, "AG3S_PROJECT_ROOT": str(ROOT), "PYTHONPATH": str(ROOT),
           "AG3S_PARALLEL_TEST_LOCK": str(tmp_path / "gpu.lock"),
           "AG3S_PARALLEL_TEST_LAUNCHER": f"{ROOT / '.venv-ag3s/bin/python'} {fake}",
           "AG3S_PARALLEL_TEST_NVIDIA_SMI": str(smi), "AG3S_PARALLEL_TEST_CGROUP_DIR": str(cg)}
    env.pop("AG3S_GPU_LOCK_HELD", None)
    log = tmp_path / "batch.log"
    p = subprocess.run(["bash", str(ROOT / "benchmark/runners/run_parallel_gpu.sh"), "--jobs", str(tmp_path / "jobs.txt"),
                        "--stagger-s", "0", "--log", str(log)], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    out = json.loads((tmp_path / "batch.job0.out").read_text().strip().splitlines()[-1])
    assert out == line.split()
    # fcntl import kept for parity with test_parallel_gpu_launcher (lock file must be free afterwards)
    with open(tmp_path / "gpu.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)
