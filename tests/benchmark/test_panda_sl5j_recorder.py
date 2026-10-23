"""SL5j-recorder: the Panda server's ``--record-constraints`` (``AG3S/adapters/panda/server.py``), CPU only.

* without the flag the server build is the one before SL5j (commit ``32a4ca8f``): the ``SafePolicy`` arguments
  (``recorder=None``) and the ``info`` record are compared with the old ``server.py`` run side by side;
* with the flag a background ``ConstraintRecordWriter`` is made under the given directory, handed to
  ``SafePolicy(recorder=...)`` and carries the RB-Y1 meta keys + the Panda ones;
* ``main`` closes it when ``serve_forever`` ends (return, KeyboardInterrupt, SystemExit) -- with mocks, and with a
  real process stopped by SIGINT / SIGTERM, where ``meta.json`` must exist afterwards.

The legacy ESDF backend is used for building (CPU); ``--target-field-policy`` is dropped as serve_safe refuses it
there (``benchmark/runners/ccr_replay.panda_server_args``, as the SL5a golden does).
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import signal
import subprocess
import sys
import textwrap
import types
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PRE_SL5J = "32a4ca8f0e9c0bcefbf9c61e8073c48af8b73b46"  # server.py before the recorder
SERVER = "AG3S/adapters/panda/server.py"


def _old_server():
    try:
        src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{PRE_SL5J}:{SERVER}"], text=True,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git object {PRE_SL5J} not available")
    mod = types.ModuleType("_panda_server_pre_sl5j")
    mod.__file__ = str(ROOT / SERVER)
    exec(compile(src, f"{PRE_SL5J}:{SERVER}", "exec"), mod.__dict__)  # noqa: S102
    return mod


def _args(version="V11AB", extra=()):
    from AG3S.adapters.panda import profile as pf
    from benchmark.runners import ccr_replay as cr

    _, argv = cr.panda_server_args("legacy")
    return pf.parse_v11ab_args(argv=argv + pf.version_server_flags(version) + list(extra))


class _Stub:
    def infer(self, obs, **kw):  # pragma: no cover - not called here
        raise AssertionError


def _describe(value):
    """Comparable description of a SafePolicy argument (objects by type + their record / config)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (dict, tuple, list)):
        return json.loads(json.dumps(value, default=repr))
    module = type(value).__module__
    # classes defined in server.py itself: the old file is loaded under another module name
    module = "panda.server" if module in ("AG3S.adapters.panda.server", "_panda_server_pre_sl5j") else module
    out = {"type": f"{module}.{type(value).__qualname__}"}
    for attr in ("to_dict", "record"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                out[attr] = json.loads(json.dumps(fn(), default=repr))
            except TypeError:
                pass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        out["fields"] = json.loads(json.dumps(dataclasses.asdict(value), default=repr))
    config = getattr(value, "config", None)
    if config is not None and callable(getattr(config, "to_dict", None)):
        out["config"] = json.loads(json.dumps(config.to_dict(), default=repr))
    return out


@pytest.fixture
def capture_safe_policy(monkeypatch):
    """``SafePolicy`` that records its constructor arguments (both server versions import it at call time)."""
    from AG3S.trajopt import safe_policy

    seen = []

    class Capturing(safe_policy.SafePolicy):
        def __init__(self, *a, **kw):
            seen.append((a, kw))
            super().__init__(*a, **kw)

    monkeypatch.setattr(safe_policy, "SafePolicy", Capturing)
    return seen


def _build(module, args, seen):
    served, info = module.build_safe_server(_Stub(), args)
    a, kw = seen[-1]
    return served, info, {"args": [_describe(x) for x in a], "kwargs": {k: _describe(v) for k, v in sorted(kw.items())}}


@pytest.mark.parametrize("version", ["V11AB", "V11AB-pb5", "V11AB-pb7"])
def test_without_flag_build_matches_pre_sl5j(version, capture_safe_policy, monkeypatch):
    from AG3S.adapters.panda import server as ps

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    old = _old_server()
    args = _args(version)
    assert args.record_constraints is None
    _, info_old, call_old = _build(old, args, capture_safe_policy)
    served, info_new, call_new = _build(ps, args, capture_safe_policy)
    assert call_new["kwargs"]["recorder"] is None and call_old["kwargs"]["recorder"] is None
    assert json.dumps(call_new, sort_keys=True) == json.dumps(call_old, sort_keys=True)
    assert json.dumps(info_new, sort_keys=True, default=str) == json.dumps(info_old, sort_keys=True, default=str)
    assert "record_constraints" not in info_new
    assert served.recorder is None and ps.close_recorder(served) is None


@pytest.mark.parametrize("esdf", ["full", "occupancy"])
def test_flag_makes_a_background_writer_and_passes_it(esdf, tmp_path, capture_safe_policy, monkeypatch):
    from AG3S.adapters.panda import server as ps
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    root = tmp_path / "server_constraints"
    _, info_plain, call_plain = _build(ps, _args("V11AB-pb7"), capture_safe_policy)
    args = _args("V11AB-pb7", ["--record-constraints", str(root), "--record-constraints-esdf", esdf])
    served, info, call = _build(ps, args, capture_safe_policy)
    rec = capture_safe_policy[-1][1]["recorder"]
    assert isinstance(rec, ConstraintRecordWriter) and served.recorder is rec
    assert rec.background and rec.esdf_mode == esdf and rec.run_dir == root / "run_0000"
    # only the recorder differs in the SafePolicy arguments; info only gains `record_constraints`
    assert {k: v for k, v in call["kwargs"].items() if k != "recorder"} == \
           {k: v for k, v in call_plain["kwargs"].items() if k != "recorder"}
    assert {k: v for k, v in info.items() if k != "record_constraints"} == info_plain
    assert info["record_constraints"] == {"dir": str(root), "run_dir": str(root / "run_0000"), "esdf_mode": esdf,
                                          "background": True, "ag3s_version": "V11AB-pb7"}
    meta = rec.meta
    json.dumps(meta)  # close() writes it with plain json
    # RB-Y1 meta keys (serve_safe RB-Y1 branch) present, with the Panda's effective values
    for key in ("config", "checkpoint", "model_xml", "links", "joint_limits", "sqp", "plan_horizon", "subtask"):
        assert key in meta, key
    assert meta["model_xml"] is None and meta["plan_horizon"] == 10 and meta["joint_limits"]["source"] == "urdf"
    assert meta["self_collision"] is False and meta["links"] == "arms"
    assert meta["robot"] == "panda" and meta["ag3s_version"] == "V11AB-pb7"
    assert meta["conversion"] == {"mode": "C", "name": __import__("AG3S.adapters.panda.action",
                                                                 fromlist=["x"]).CONVERSION_MODES["C"]}
    assert meta["version_flags"]["closing_hand_exempt"] is True and meta["version_flags"]["links"] == "arms"
    assert meta["grasp_contact"] == info["grasp_contact"]
    assert meta["esdf_backend"] == "legacy" and meta["horizon"] == ps.PANDA_HORIZON
    closed = ps.close_recorder(served)
    assert closed == {"closed": True, "run_dir": str(root / "run_0000"), "n_chunks": 0, "n_failed": 0}
    on_disk = json.loads((root / "run_0000" / "meta.json").read_text())
    assert on_disk["writer"]["mode"] == "background" and on_disk["robot"] == "panda" and on_disk["n_chunks"] == 0


def test_every_version_is_read_back():
    from AG3S.adapters.panda import profile as pf

    for version in pf.AG3S_VERSIONS:
        assert pf.match_ag3s_version(_args(version)) == version
    assert pf.match_ag3s_version(_args("V11AB", ["--grasp-zone-mm", "7"])) is None


def test_record_flag_checks(caplog, tmp_path):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss

    ps.check_record_flags(SimpleNamespace(record_constraints=None, no_safe=True, esdf_backend="nope"))  # no flag: no-op
    with pytest.raises(SystemExit, match="no-safe"):
        ps.check_record_flags(SimpleNamespace(record_constraints="d", no_safe=True, esdf_backend="curobo"))
    with pytest.raises(SystemExit, match="빈 디렉토리"):
        ps.check_record_flags(SimpleNamespace(record_constraints="d", no_safe=False, esdf_backend="nope"))
    # the cuRobo + occupancy warning, as the RB-Y1 server gives it (no cuRobo import needed for the writer)
    args = pf.parse_v11ab_args(argv=pf.v11ab_server_argv() + ["--robot", "panda", "--record-constraints",
                                                               str(tmp_path / "occ"), "--record-constraints-esdf",
                                                               "occupancy"])
    assert args.esdf_backend == "curobo"
    to_config = ps._replace_horizon_limits(ss.trajopt_config_from_args(args))
    with caplog.at_level("WARNING"):
        rec = ps.constraint_recorder(args, to_config, conversion="A", ag3s_info={})
    try:
        assert "점유 배열을 남기지 않습니다" in caplog.text
        assert rec.meta["esdf_backend"] == "curobo" and rec.meta["ag3s_version"] == "V11AB"
    finally:
        rec.close()


# --- main: close on every way out -----------------------------------------------------------------------------
def _fake_openpi(monkeypatch, serve_forever):
    pc = types.ModuleType("openpi.policies.policy_config")
    pc.create_trained_policy = lambda *a, **k: object()
    ws = types.ModuleType("openpi.serving.websocket_policy_server")

    class Server:
        def __init__(self, **kw):
            pass

        def serve_forever(self):
            serve_forever()

    ws.WebsocketPolicyServer = Server
    cfg = types.ModuleType("openpi.training.config")
    cfg.get_config = lambda name: name
    pkgs = {"openpi": types.ModuleType("openpi"), "openpi.policies": types.ModuleType("openpi.policies"),
            "openpi.serving": types.ModuleType("openpi.serving"), "openpi.training": types.ModuleType("openpi.training")}
    pkgs["openpi.policies"].policy_config = pc
    pkgs["openpi.serving"].websocket_policy_server = ws
    pkgs["openpi.training"].config = cfg
    for name, mod in {**pkgs, pc.__name__: pc, ws.__name__: ws, cfg.__name__: cfg}.items():
        monkeypatch.setitem(sys.modules, name, mod)


class _Rec:
    def __init__(self):
        self.closed = 0
        self.run_dir = "fake"
        self.meta = {}
        self.n_failed = 0

    def close(self):
        self.closed += 1
        self.meta["n_chunks"] = 0


def _raise(exc):
    def fn():
        raise exc
    return fn


@pytest.mark.parametrize("ending", ["return", "keyboard_interrupt", "system_exit"])
@pytest.mark.parametrize("recording", [False, True])
def test_main_closes_the_recorder(ending, recording, monkeypatch):
    from AG3S.adapters.panda import server as ps

    serve = {"return": lambda: None, "keyboard_interrupt": _raise(KeyboardInterrupt()),
             "system_exit": _raise(SystemExit(128 + signal.SIGTERM))}[ending]
    _fake_openpi(monkeypatch, serve)
    rec = _Rec() if recording else None
    served = SimpleNamespace(recorder=rec, metadata={})
    monkeypatch.setattr(ps, "build_safe_server", lambda policy, args, attention_model=None: (served, {}))
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    before = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    args = SimpleNamespace(config="c", checkpoint="k", default_prompt=None, no_safe=False, no_attention=True, port=0,
                           record_constraints="d" if recording else None, esdf_backend="curobo")
    try:
        if ending == "return":
            ps.main(args, None)
        else:
            with pytest.raises((KeyboardInterrupt, SystemExit)):
                ps.main(args, None)
        installed = signal.getsignal(signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, before)
    if recording:
        assert rec.closed == 1
        assert installed is not signal.SIG_DFL  # serve_safe.exit_cleanly_on_sigterm
    else:
        assert installed is signal.SIG_DFL  # nothing installed without a recorder (as before)


_PROC = textwrap.dedent("""
    import asyncio, sys, types
    import numpy as np
    sys.path.insert(0, {root!r})
    from AG3S.adapters.panda import server as ps
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    async def forever():
        print("READY", flush=True)
        await asyncio.sleep(3600)

    pc = types.ModuleType("openpi.policies.policy_config"); pc.create_trained_policy = lambda *a, **k: object()
    ws = types.ModuleType("openpi.serving.websocket_policy_server")
    class Server:
        def __init__(self, **kw): pass
        def serve_forever(self): asyncio.run(forever())   # the openpi server's shape
    ws.WebsocketPolicyServer = Server
    cfg = types.ModuleType("openpi.training.config"); cfg.get_config = lambda n: n
    for name, mod in (("openpi", types.ModuleType("openpi")), ("openpi.policies", types.ModuleType("p")),
                      ("openpi.serving", types.ModuleType("s")), ("openpi.training", types.ModuleType("t"))):
        sys.modules[name] = mod
    sys.modules["openpi.policies"].policy_config = pc; sys.modules[pc.__name__] = pc
    sys.modules["openpi.serving"].websocket_policy_server = ws; sys.modules[ws.__name__] = ws
    sys.modules["openpi.training"].config = cfg; sys.modules[cfg.__name__] = cfg

    rec = ConstraintRecordWriter({out!r}, esdf_mode="full", background=True, meta={{"robot": "panda"}})
    for i in range(3):
        rec.record(t_step=i, chunk_index=i + 1, constraint_set=None, debug={{}}, clearance=np.full((2, 3), 0.1))
    served = types.SimpleNamespace(recorder=rec, metadata={{}})
    ps.build_safe_server = lambda policy, args, attention_model=None: (served, {{}})
    args = types.SimpleNamespace(config="c", checkpoint="k", default_prompt=None, no_safe=False, no_attention=True,
                                 port=0, record_constraints={out!r}, esdf_backend="curobo")
    ps.main(args, None)
""")


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_stopped_process_leaves_meta_json(sig, tmp_path):
    out = tmp_path / "server_constraints"
    script = tmp_path / "proc.py"
    script.write_text(_PROC.format(root=str(ROOT), out=str(out)))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)}
    env.pop("AG3S_RUN_DIR", None)
    proc = subprocess.Popen([sys.executable, str(script)], cwd=str(tmp_path), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        assert line.strip() == "READY", line + proc.stderr.read()
        proc.send_signal(sig)
        stdout, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    meta_path = out / "run_0000" / "meta.json"
    assert meta_path.exists(), stdout + stderr
    meta = json.loads(meta_path.read_text())
    assert meta["n_chunks"] == 3 and meta["writer"] == {"mode": "background", "n_failed": 0, "failed_paths": []}
    assert sorted(p.name for p in (out / "run_0000").glob("chunk_*.npz")) == [f"chunk_{i:05d}.npz" for i in range(3)]
    with np.load(out / "run_0000" / "chunk_00002.npz") as z:
        assert z["clearance"].shape == (2, 3)
    assert proc.returncode != 0  # the signal still ends the process (KeyboardInterrupt / SystemExit 143)
