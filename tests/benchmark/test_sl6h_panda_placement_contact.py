"""SL6-H — `AG3S-final-panda-v3-e0to-pc` and the Panda server side of HSR6 ⑤ placement contact (task
`AG3S/ag3s/docs/handoff/SL6.task.md` §11 H1-H4), CPU only.

1. H3: the version = E0판 + `--hsr6-placement-contact`; every version of HEAD d53745f5 keeps its flags and base argv
   (profile loaded from git); the launcher argv; read-back.
2. H4: the Panda server accepts the flag with `--hsr6-probe` (ported list) and refuses it without; the CPU build turns
   it on (SafePolicy `hsr6.placement_contact`, config dump); E0판 / 3판 / 2판 build the same SafePolicy arguments and
   `info` as the HEAD server module.
3. H2: `panda/hsr6` on the response (HSR6 on only) and the client's `_calls.jsonl` row key `hsr6` (py3.8 client, only
   when the server sends it).
"""
from __future__ import annotations

import json
import subprocess
import types

import numpy as np
import pytest

from tests.benchmark.test_sl6a_panda_final import PROBE, ROOT, _final_args, _Stub, gpu_ok  # noqa: F401 (fixture)

HEAD = "d53745f505392ee7364bd75d88fa6a4caa7ebf9d"     # SL6-F, the commit SL6-H starts from
PC = "AG3S-final-panda-v3-e0to-pc"
E0 = "AG3S-final-panda-v3-e0to"


def _head_module(path: str, name: str):
    try:
        src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{HEAD}:{path}"], text=True,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git object {HEAD} not available")
    mod = types.ModuleType(name)
    mod.__file__ = str(ROOT / path)
    exec(compile(src, f"{HEAD}:{path}", "exec"), mod.__dict__)  # noqa: S102
    return mod


# ------------------------------------------------------------------------------------------------ 1. H3 version
def test_pc_version_is_e0to_plus_the_flag(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    assert pf.AG3S_VERSIONS[PC] == pf.AG3S_VERSIONS[E0] + ("--hsr6-placement-contact",)
    assert PC in pf.FINAL_CONFIG_VERSIONS and pf.FINAL_CONFIG_VERSIONS[-1] == PC
    assert pf.version_server_flags(PC)[-3:] == ["--hsr6-placement-contact", "--hsr6-probe", pf.PANDA_HSR6_PROBE]
    args = _final_args(PC)
    assert args.hsr6_placement_contact is True and _final_args(E0).hsr6_placement_contact is False
    assert pf.match_ag3s_version(args) == PC and pf.match_ag3s_version(_final_args(E0)) == E0
    # the E0판 TO settings are kept
    assert args.to_multi_seed in (0, None) and args.minimal_intervention_mm is None
    assert args.continuity_after_grasp and args.deflection_rate_after_grasp


def test_every_head_version_keeps_its_argv(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    old = _head_module("AG3S/adapters/panda/profile.py", "_panda_profile_head")
    assert set(old.AG3S_VERSIONS) == set(pf.AG3S_VERSIONS) - {PC} - set(pf.P0M_VERSIONS) - set(pf.P0M_SL9_VERSIONS) - set(pf.PC_LATCH_VERSIONS) - set(pf.PC_NOPROBE_VERSIONS) - set(pf.LSFORK_VERSIONS) - set(pf.SL12_VERSIONS) - set(pf.SL13_VERSIONS) - set(pf.SL14_VERSIONS) - set(pf.SL15_NTM_VERSIONS) - set(pf.SL15_SFHAND_VERSIONS) - set(pf.SL16_HSC_VERSIONS) - set(pf.SL16_HVC_VERSIONS) \
        - set(pf.GT_VERSIONS) - set(pf.GT2_VERSIONS) - set(pf.GT3_VERSIONS) - set(pf.GT4_VERSIONS) - set(pf.GT5_VERSIONS) \
        - set(pf.GT6_VERSIONS) - set(pf.GT7_VERSIONS) - set(pf.GT8_VERSIONS) - set(pf.GT9_VERSIONS) - set(pf.GT10_VERSIONS) - set(pf.GT11_VERSIONS) - set(getattr(pf, "GT12_VERSIONS", ())) - set(getattr(pf, "GT13_VERSIONS", ())) - set(getattr(pf, "GT14_VERSIONS", ())) - set(getattr(pf, "GT15_VERSIONS", ()))   # SL12 -oraclegt; SL7 P0m, SL7d/e pc variants, SL9 P0m A/B, SL11 -lsfork · SL-GTTO(2-9)
    assert pf.FINAL_CONFIG_VERSIONS == old.FINAL_CONFIG_VERSIONS + (PC,)
    for name, flags in old.AG3S_VERSIONS.items():
        assert pf.AG3S_VERSIONS[name] == flags, name
        assert pf.version_server_argv(name) == old.version_server_argv(name), name
        assert pf.version_server_flags(name) == old.version_server_flags(name), name
        assert "--hsr6-placement-contact" not in pf.version_server_flags(name), name
    for name in ("V11AB", "V11AB-pb7", "AG3S-final-panda-v2", "AG3S-final-panda-v3", E0):
        args = (_final_args(name) if pf.is_final_version(name)
                else pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(pf.AG3S_VERSIONS[name])))
        assert args.hsr6_placement_contact is False and pf.match_ag3s_version(args) == name, name


def test_launcher_argv(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from benchmark.runners import launch_libero_paired as lp

    e0 = lp.server_argv("vla_v11ab", "/ck", 9, E0, hsr6_probe="/p.npz")
    pc = lp.server_argv("vla_v11ab", "/ck", 9, PC, hsr6_probe="/p.npz")
    assert pc == e0[:-2] + ["--hsr6-placement-contact", "--hsr6-probe", "/p.npz"]
    assert "--no-roi-carve" not in pc and pc[pc.index("--robot") + 1] == "panda"
    a = lp.build_parser().parse_args(["--task-ids", "1", "--episode-ids", "1", "--run-id", "x", "--conversion", "C",
                                      "--ag3s-version", PC])
    assert lp.version_options(a) == {} and pf.version_server_flags(PC, **lp.version_options(a))[-1] == \
        pf.PANDA_HSR6_PROBE


# ------------------------------------------------------------------------------------------------ 2. H4 server
def test_server_accepts_the_flag_with_the_probe_only(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps

    assert "hsr6_placement_contact" in ps.PORTED_FINAL_CONFIG_DESTS
    assert "hsr6_placement_contact" not in ps.UNPORTED_FINAL_CONFIG_DESTS
    ps.check_unported_flags(_final_args(PC), None)                                   # with the probe: accepted
    argv = pf.version_server_argv(PC) + ["--robot", "panda"] + list(pf.AG3S_VERSIONS[PC])
    with pytest.raises(ValueError, match="needs --hsr6-probe"):
        ps.check_unported_flags(pf.parse_v11ab_args(argv=argv), None)
    # the flag alone on a V11AB-family argv: an HSR6 use without the Panda probe
    alone = pf.parse_v11ab_args(argv=pf.version_server_argv("V11AB-pb7") + ["--robot", "panda"]
                                + list(pf.AG3S_VERSIONS["V11AB-pb7"]) + ["--hsr6-placement-contact"])
    with pytest.raises(ValueError, match="needs --hsr6-probe"):
        ps.check_unported_flags(alone, None)
    assert "hsr6_placement_contact" in ps.final_options_on(_final_args(PC))
    assert "hsr6_placement_contact" not in ps.final_options_on(_final_args(E0))


def test_pc_builds_on_cpu(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    served, info = ps.build_safe_server(_Stub(), _final_args(PC), gpu_warm_up=False)
    safe = served._policy
    fc = info["final_config"]
    assert fc["version"] == PC
    assert safe.hsr6.placement_contact is True
    assert fc["hsr6"]["flags"] == ["attach_assist", "held_gate", "target_exempt", "objective", "placement_contact"]
    assert "hsr6_placement_contact" in fc["options_on"]
    # E0판 otherwise: robot model, grasp rules, TO settings
    assert info["ag3s"]["n_constraint_spheres"] == 133 + 37
    assert safe.latch_phase and safe.held_hand_exempt and safe.closing_hand_exempt and safe.ag3s.reach_at_capture
    sqp, cost = safe.to_config.sqp, safe.to_config.cost
    assert not (getattr(sqp, "multi_seed", 0) or 0) and getattr(sqp, "minimal_intervention", None) is None
    assert cost.continuity_after_grasp and cost.deflection_rate_after_grasp
    json.dumps(info, default=str)
    e0, _ = ps.build_safe_server(_Stub(), _final_args(E0), gpu_warm_up=False)
    assert e0._policy.hsr6.placement_contact is False and e0._policy._last_placement_contact == {}


@pytest.mark.parametrize("version", ["AG3S-final-panda-v2", "AG3S-final-panda-v3", E0])
def test_existing_final_builds_match_the_head_server(version, gpu_ok, monkeypatch):
    """Flag off: the HEAD server module and this one construct SafePolicy with the same arguments and the same `info`
    for the earlier final versions (the V11AB family is `test_sl6a_panda_final.test_v11ab_family_build_is_the_pre_sl6_build`)."""
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import safe_policy
    from tests.benchmark.test_panda_sl5j_recorder import _describe

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    seen = []

    class Capturing(safe_policy.SafePolicy):
        def __init__(self, *a, **kw):
            seen.append((a, kw))
            super().__init__(*a, **kw)

    monkeypatch.setattr(safe_policy, "SafePolicy", Capturing)
    old = _head_module("AG3S/adapters/panda/server.py", "_panda_server_head")
    args = _final_args(version)

    def build(module):
        served, info = module.build_safe_server(_Stub(), args, gpu_warm_up=False)
        a, kw = seen[-1]
        return info, {"args": [_describe(x) for x in a], "kwargs": {k: _describe(v) for k, v in sorted(kw.items())}}

    info_old, call_old = build(old)
    info_new, call_new = build(ps)
    norm = lambda d: json.loads(json.dumps(d, default=str).replace("_panda_server_head", "panda.server"))  # noqa: E731
    assert json.dumps(norm(call_new), sort_keys=True) == json.dumps(norm(call_old), sort_keys=True)
    assert json.dumps(info_new, sort_keys=True, default=str) == json.dumps(info_old, sort_keys=True, default=str)
    assert seen[-1][1]["hsr6"].placement_contact is False


# ------------------------------------------------------------------------------------------------ 3. H2 records
def test_response_carries_the_hsr6_record_only_with_hsr6_on(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import hsr6 as H6
    from AG3S.trajopt.grasp_latch import GraspPhase, LatchEvent
    from tests.benchmark.test_panda_sl5j_recorder import _args

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    assert ps.hsr6_call_record(types.SimpleNamespace(hsr6=None)) is None
    v11ab, _ = ps.build_safe_server(_Stub(), _args("V11AB-pb7"))
    assert ps.hsr6_call_record(v11ab._policy) is None                                    # no key on V11AB / pb
    served, _ = ps.build_safe_server(_Stub(), _final_args(PC), gpu_warm_up=False)
    safe = served._policy
    p = {c: (0.8 if c == "placement" else 0.05) for c in H6.PHASES6}
    safe._pending = {}
    safe._hsr6_start({"p": p, "argmax": "placement", "probe": "hsr6_panda_kv_L4"})
    safe._latch.phase = GraspPhase.HELD
    safe._latch._last_grasp = LatchEvent(phase=GraspPhase.HELD, evidence={"checks": {}})
    from tests.trajopt.test_sl6h_placement_contact import _snapshot

    safe._hsr6_placement_contact(_snapshot(attached_points=np.zeros((2, 3)), attached_parent_link="panda_hand"))
    safe.refiner.last_result = types.SimpleNamespace(metrics={"placement_contact": {"rows_released": 4,
                                                                                     "held_rows_released": 2}})
    safe._hsr6_placement_contact_result()
    rec = json.loads(json.dumps(ps.hsr6_call_record(safe)))
    assert rec["phase"] == "placement" and rec["p"]["placement"] == pytest.approx(0.8)
    assert rec["flags"][-1] == "placement_contact" and rec["changed"]["placement_contact"] is True
    pc = rec["detail"]["placement_contact"]
    assert pc["on"] is True and pc["applied"] is True and pc["rows_released"] == 4 and pc["held_rows_released"] == 2


def test_client_row_records_the_hsr6_record():
    """The py3.8 client writes `hsr6` into the call row only when the server sends `panda/hsr6`; actions unchanged."""
    code = r'''
import sys, json, numpy as np
sys.path.insert(0, "benchmark/runners")
import libero_safe_client as c

class WS:
    def __init__(self, extra):
        self.extra = extra
    def infer(self, req):
        out = {"actions": np.full((10, 7), 0.25), "safe": True, "panda/mode": "vla_v11ab"}
        out.update(self.extra)
        return out

rows, acts = [], []
for extra in ({}, {"panda/hsr6": {"phase": "placement", "p": {"placement": np.float32(0.8)},
                                  "detail": {"placement_contact": {"on": True, "rows_released": 3}}}}):
    cl = c.SafeClient(WS(extra), 7, 0, 0, 5)
    cl.eef = [None]
    cl._request = lambda element: ({"panda/T_world_base": np.eye(4)}, np.zeros(9))
    out = cl.infer({})
    rows.append(cl.calls[-1])
    acts.append(np.asarray(out["actions"]).tolist())
print(json.dumps({"keys0": sorted(rows[0]), "keys1": sorted(rows[1]), "hsr6": rows[1].get("hsr6"),
                  "same_actions": acts[0] == acts[1]}))
'''
    r = subprocess.run([str(ROOT / ".venv-libero/bin/python"), "-c", code], cwd=str(ROOT), capture_output=True,
                       text=True)
    if r.returncode != 0 and "No module named" in r.stderr:
        pytest.skip(r.stderr.splitlines()[-1])
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert "hsr6" not in out["keys0"] and set(out["keys1"]) - set(out["keys0"]) == {"hsr6"}
    assert out["hsr6"]["phase"] == "placement" and out["hsr6"]["detail"]["placement_contact"]["rows_released"] == 3
    assert out["same_actions"] is True
