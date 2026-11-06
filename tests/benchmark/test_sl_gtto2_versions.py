"""SL-GTTO2 — `AG3S-commonto-gt2` = `AG3S-commonto-gt` + `--oracle-admissible --oracle-held-target` (task SL-GTTO2 §1), CPU.

1. Version: argv = the SL-GTTO argv + the two flags; same base argv / per-run rules / client oracle flags as the GT
   versions; read back by `match_ag3s_version`. Every other version: no new flag, no new recorded flag key, no new
   identity key, and `AG3S-commonto-gt`'s config / per-call / dump records carry no SL-GTTO2 key.
2. Panda server: both dests ported, the GT2 version builds on CPU with both clustering keys on and records them.
3. Recorded GT-arm frames (offline replay, `tests/benchmark/libero_gtto2_replay.py`; skipped when storage is absent):
   F1 — LIBERO t4e0 and SafeLIBERO t3e0 first calls: the run's server gives `no_admissible` (`not_supported`), the GT2
   flags give grounding `ok` with the refusal recorded. F2 — SafeLIBERO t1e1 calls 0–13 (closing 6–11, attach 12): the
   run's server integrates code-1 pixels into the TSDF in every closing / held call, the GT2 flags integrate none.
"""
from __future__ import annotations

import json

import pytest

from tests.benchmark.test_sl6a_panda_final import _Stub, gpu_ok  # noqa: F401 (fixture)
from tests.benchmark.test_sl_gtto_gt_perception import EXPECTED_GT, GT, _args

GT2 = "AG3S-commonto-gt2"
NEW_FLAGS = ("--oracle-admissible", "--oracle-held-target")
NEW_DESTS = ("oracle_admissible", "oracle_held_target")


# ================================================================================================ 1 versions
def test_gt2_argv_is_the_gt_argv_plus_the_two_flags(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    assert pf.GT2_VERSIONS == (GT2,) and pf.GT_VERSIONS == (GT, "AG3S-commonto-gt-pass")
    assert list(pf.AG3S_VERSIONS[GT2]) == EXPECTED_GT + list(NEW_FLAGS)
    assert pf.is_final_version(GT2) and pf.version_requires_oracle_segmentation(GT2)
    assert pf.version_server_argv(GT2) == pf.version_server_argv(GT)
    assert pf.version_server_flags(GT2) == list(pf.AG3S_VERSIONS[GT2])
    for kw in ({"hsr6_probe": "x"}, {"attention_source": "noun_target:L8H6"}):
        with pytest.raises(ValueError):
            pf.version_server_flags(GT2, **kw)
    args = _args(GT2)
    assert pf.match_ag3s_version(args) == GT2
    vals = pf.version_flag_values(args)
    assert vals["oracle_admissible"] is True and vals["oracle_held_target"] is True


def test_no_other_version_gets_a_new_flag_or_key(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    for name in pf.AG3S_VERSIONS:
        if (name == GT2 or name in getattr(pf, "GT3_VERSIONS", ()) or name in getattr(pf, "GT4_VERSIONS", ())
                or name in getattr(pf, "GT5_VERSIONS", ()) or name in getattr(pf, "GT6_VERSIONS", ())
                or name in getattr(pf, "GT7_VERSIONS", ()) or name in getattr(pf, "GT8_VERSIONS", ()) or name in getattr(pf, "GT9_VERSIONS", ())):
            # SL-GTTO3 / SL-GTTO4 variants carry the GT2 flags
            continue
        assert not set(NEW_FLAGS) & set(pf.AG3S_VERSIONS[name]), name
        try:
            parsed = pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(pf.AG3S_VERSIONS[name]))
        except SystemExit:
            continue
        assert not set(NEW_DESTS) & set(pf.version_flag_values(parsed)), name
        assert not set(NEW_DESTS) & set(ss.t43t_identity_section(parsed)), name
    assert pf.match_ag3s_version(_args(GT)) == GT


def test_launcher_hands_the_gt2_client_the_oracle_flags():
    from benchmark.runners import launch_libero_paired as lp
    from tests.benchmark.test_launcher_record_constraints import BASE_ARGV

    args = lp.build_parser().parse_args(BASE_ARGV + ["--ag3s-version", GT2])
    argv = lp.oracle_client_argv("vla_v11ab", args, ["c"])
    assert argv[0] == "c" and set(lp.ORACLE_CLIENT_FLAGS) <= set(argv)
    assert lp.oracle_client_argv("vla_only", args, ["c"]) == ["c"]


# ================================================================================================ 2 server
def test_server_ports_and_builds_gt2_on_cpu(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps

    for d in NEW_DESTS:
        assert d in ps.PORTED_FINAL_CONFIG_DESTS and d not in ps.UNPORTED_FINAL_CONFIG_DESTS
    ps.check_unported_flags(_args(GT2), None)
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    served, info = ps.build_safe_server(_Stub(), _args(GT2), gpu_warm_up=False)
    c = served._policy.ag3s.config.clustering
    assert c.oracle_admissible and c.oracle_held_target and c.grounding_source == "oracle"
    gp = info["final_config"]["gt_perception"]
    assert gp["oracle_admissible"] is True and gp["oracle_held_target"] is True
    assert info["final_config"]["version"] == GT2


def test_gt_version_records_carry_no_gt2_key(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import server as ps

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    served, info = ps.build_safe_server(_Stub(), _args(GT), gpu_warm_up=False)
    safe = served._policy
    assert not set(NEW_DESTS) & set(info["final_config"]["gt_perception"])
    c = safe.ag3s.config.clustering
    assert c.oracle_admissible is False and c.oracle_held_target is False
    safe.ag3s._grounding_frame = {"source": "oracle", "n_code1": 3, "cameras": ["head"]}
    rec = ps.gt_perception_call_record(safe)
    assert set(rec["grounding"]) == {"source", "n_code1", "cameras"}
    json.dumps(rec)


# ================================================================================================ 3 recorded frames
def _episode(run_attr, stem, extra=()):
    from tests.benchmark import libero_gtto2_replay as rp

    run = getattr(rp, run_attr)
    if not rp.available(run, stem):
        pytest.skip(f"recorded GT-arm episode {run}/{stem} not in this storage")
    return rp.Episode(run, stem, extra)


@pytest.mark.parametrize("run_attr, stem", [("GT_LIBERO", "libero_spatial_t4_e0"),
                                            ("GT_SAFE_T3", "safelibero_spatial_t3_e0")])
def test_replay_f1_grounds_the_refused_gt_target(run_attr, stem):
    base = _episode(run_attr, stem)
    m, dbg, _ = base.step(0, grasp=False)
    assert base.rows[0]["gt_perception"]["grounding"]["status"] == "no_admissible"
    assert m["grounding_source"]["status"] == "no_admissible"
    assert m["grounding_source"]["n_code1"] == base.rows[0]["gt_perception"]["grounding"]["n_code1"]
    from tests.benchmark.libero_gtto2_replay import admissibility

    adm = admissibility(base)
    assert [a["reason"] for a in adm] == ["not_supported"] and adm[0]["graspable"] and not adm[0]["overlaps_destination"]
    on = _episode(run_attr, stem, NEW_FLAGS)
    m, dbg, _ = on.step(0, grasp=False)
    rec = m["grounding_source"]
    assert rec["status"] == "ok" and dbg["grounding"].target is not None
    assert [r["reason"] for r in rec["admissibility_would_refuse"]] == ["not_supported"]
    assert rec["exclusion_gate_would_refuse"] is None


def test_replay_f2_keeps_the_held_bowl_out_of_the_tsdf():
    base = _episode("GT_SAFE_T1", "safelibero_spatial_t1_e1")
    on = _episode("GT_SAFE_T1", "safelibero_spatial_t1_e1", NEW_FLAGS)
    for k in range(14):
        state = base.rows[k]["grasp"]["state"]
        _, _, seg = base.step(k)
        n0, kept0 = base.tsdf_code1(seg)
        m1, _, seg1 = on.step(k)
        n1, kept1 = on.tsdf_code1(seg1)
        assert n0 == n1
        if state in ("closing", "held"):
            assert kept0 > 0 and kept1 == 0, (k, state, kept0, kept1)
            assert m1["grounding_source"]["held_target_mask"]["phase"] == ("held" if state == "held" else "closing")
        else:
            assert kept1 == kept0, (k, state)
            assert m1["grounding_source"]["held_target_mask"]["active"] is False
    assert base.ag3s.attached is not None and on.ag3s.attached is not None
