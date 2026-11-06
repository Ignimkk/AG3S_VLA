"""SL10f -- "미인증 HOLD 풀기 진단": `AG3S-final-panda-v3-e0to-pc-uncwaive` (task `AG3S/ag3s/docs/handoff/SL10f.task.md` §2). CPU.

The version is registration only: the pc version's flags plus the existing serve_safe flag ``--allow-uncertified``
(``safety.require_certified_geometry=False``). Checked here:

(a) new version == pc version + exactly that flag; it is a final version that takes the HSR6 probe like the pc version;
(b) every pre-existing version is unchanged (the registry minus the new key hashes to the value measured before this change),
    as are their ``PC_LATCH_VERSIONS`` membership and the record keys (``allow_uncertified`` joins a record only when set);
(c) the parsed server args give ``allow_uncertified=True`` (only that dest differs from the pc version), the Panda server
    accepts them, the resolved TrajOpt config has ``safety.require_certified_geometry`` False (pc: True), and the version name
    is read back from the flags;
(d) the paired LIBERO / LIBERO-Safety launchers (and the eval matrix's name check) accept the version name.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from tests.benchmark.test_sl6a_panda_final import _Stub, gpu_ok  # noqa: F401 (fixture)

PC = "AG3S-final-panda-v3-e0to-pc"
NEW = f"{PC}-uncwaive"
FLAG = "--allow-uncertified"

#: sha256 of ``json.dumps({name: list(flags)}, sort_keys=True)`` over AG3S_VERSIONS before SL10f (50 versions), measured on
#: main 62b4a888 with the registry code of that commit.
OLD_REGISTRY_SHA256 = "9cb986d5e0408f5db0e4a90dc25d1d98117828265d4b4fda7075c3d342539d64"
OLD_REGISTRY_SIZE = 50


def _probe_or_skip():
    from AG3S.adapters.panda import profile as pf

    if not pathlib.Path(pf.PANDA_HSR6_PROBE).exists():
        pytest.skip("Panda HSR6 probe not in this storage root")


def _args(v, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(v)
    argv[argv.index("--config") + 1] = "pi05_libero"
    return pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(v))


# ------------------------------------------------------------------------------------------------ (a) + (b) registry
def test_new_version_is_the_pc_version_plus_the_one_flag():
    from AG3S.adapters.panda import profile as pf

    assert pf.SL10F_VERSIONS == (NEW,) and pf.UNCERTIFIED_WAIVE_FLAGS == (FLAG,)
    assert list(pf.AG3S_VERSIONS[NEW]) == list(pf.AG3S_VERSIONS[PC]) + [FLAG]
    assert FLAG not in pf.AG3S_VERSIONS[PC]
    assert pf.is_final_version(NEW) and NEW in pf.PC_LATCH_VERSIONS
    assert pf.version_server_argv(NEW) == pf.version_server_argv(PC)
    assert pf.version_server_flags(NEW)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]
    assert pf.version_server_flags(NEW)[:-3] == pf.version_server_flags(PC)[:-2]          # same, plus the one flag
    assert pf.version_server_flags(NEW)[-3] == FLAG
    assert not any(FLAG in f for n, f in pf.AG3S_VERSIONS.items() if n != NEW)


def test_every_other_version_is_unchanged():
    from AG3S.adapters.panda import profile as pf

    old = {k: list(v) for k, v in pf.AG3S_VERSIONS.items() if k != NEW and k not in pf.SL11_VERSIONS and not k.endswith("-lsfork") and k not in pf.SL11_GAP_VERSIONS and k not in pf.SL12_VERSIONS and k not in pf.SL13_VERSIONS and k not in pf.SL14_VERSIONS and k not in pf.SL15_NTM_VERSIONS and k not in pf.SL15_SFHAND_VERSIONS and k not in pf.SL16_HVC_VERSIONS}   # SL11 comes later (§5: the -gap pair; SL12 the -oraclegt pair)
    assert len(old) == OLD_REGISTRY_SIZE
    assert hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest() == OLD_REGISTRY_SHA256
    assert pf.PC_LATCH_VERSIONS.count(NEW) == 1 and NEW in pf.PC_LATCH_VERSIONS[:-len(pf.SL11_VERSIONS)]         # the others keep their place


# ------------------------------------------------------------------------------------------------ (c) server side
def test_flag_parses_and_only_it_differs(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps

    _probe_or_skip()
    a, b = _args(NEW, monkeypatch), _args(PC, monkeypatch)
    assert a.allow_uncertified is True and b.allow_uncertified is False
    assert {k for k in vars(a) if vars(a)[k] != vars(b)[k]} == {"allow_uncertified"}
    assert "allow_uncertified" not in ps.UNPORTED_FINAL_CONFIG_DESTS
    ps.check_unported_flags(a, None)                                                   # the Panda server does not refuse it
    from AG3S.trajopt import serve_safe as ss

    ss.reject_hsr6_flags(ss.build_parser(), a)
    ss.reject_panda_only_flags(ss.build_parser(), a)
    # the resolved TrajOpt config: certification not required (pc: required)
    assert ss.trajopt_config_from_args(a).safety.require_certified_geometry is False
    assert ss.trajopt_config_from_args(b).safety.require_certified_geometry is True
    # record keys: on adds its own, off keeps the old ones; the name is read back
    assert pf.match_ag3s_version(a) == NEW and pf.match_ag3s_version(b) == PC
    va, vb = pf.version_flag_values(a), pf.version_flag_values(b)
    assert va["allow_uncertified"] is True and "allow_uncertified" not in vb
    assert set(vb) <= set(va) and all(va[k] == vb[k] for k in vb)


def test_old_versions_record_keys_do_not_gain_the_dest(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    _probe_or_skip()
    for v in (PC, f"{PC}-surface", f"{PC}-plift", f"{PC}-hfq", "AG3S-final-panda-v3-e0to", "V11AB-pb7"):
        assert "allow_uncertified" not in pf.version_flag_values(_args(v, monkeypatch)), v


def test_the_server_builds_with_the_waive(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps

    _probe_or_skip()
    served, _info = ps.build_safe_server(_Stub(), _args(NEW, monkeypatch), gpu_warm_up=False)
    assert served._policy.to_config.safety.require_certified_geometry is False            # noqa: SLF001
    base, _ = ps.build_safe_server(_Stub(), _args(PC, monkeypatch), gpu_warm_up=False)
    assert base._policy.to_config.safety.require_certified_geometry is True               # noqa: SLF001


# ------------------------------------------------------------------------------------------------ (d) launchers
def test_launchers_accept_the_version(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_safety_paired as lsp

    got = lp.server_argv("vla_v11ab", "/ck", 9, NEW, hsr6_probe="/p.npz")
    want = lp.server_argv("vla_v11ab", "/ck", 9, PC, hsr6_probe="/p.npz")
    assert got == want[:-2] + [FLAG, "--hsr6-probe", "/p.npz"]
    a = lp.build_parser().parse_args(["--task-ids", "1", "--episode-ids", "1", "--run-id", "x", "--conversion", "C",
                                      "--ag3s-version", NEW])
    assert a.ag3s_version == NEW and pf.version_server_flags(NEW, **lp.version_options(a))[-3] == FLAG
    # the LIBERO-Safety launcher wraps the same server_argv with its own --config
    wrapped = lsp.make_server_argv("pi05_libero_safety")("vla_v11ab", "/ck", 9, NEW)
    assert FLAG in wrapped and wrapped[wrapped.index("--robot") + 1] == "panda"
    s = lsp.build_parser().parse_args(["--task-suite-name", "libero_safety", "--level", "1", "--policy-checkpoint",
                                       "pi05_libero_safety", "--task-ids", "0", "--episode-ids", "0", "--run-id", "x",
                                       "--conversion", "C", "--ag3s-version", NEW]) \
        if "libero_safety" in lsp.lsc.SUITES else None
    assert s is None or s.ag3s_version == NEW
    # eval_matrix.build_jobs validates the name with exactly this call
    assert pf.version_server_flags(NEW)
