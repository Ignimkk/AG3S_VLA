"""SL8b — the three Panda versions that loosen the lock on the first-chosen object (task
`AG3S/ag3s/docs/handoff/SL8b.task.md` §2.3): pc판 + a window vote (`-idwin`), + the seed-attention release
(`-idattn`), + both (`-idboth`). The rules are default off, so pc판 itself is unchanged."""
from __future__ import annotations

import pathlib

import pytest

from tests.benchmark.test_sl6a_panda_final import _Stub, gpu_ok  # noqa: F401 (fixture)

PC = "AG3S-final-panda-v3-e0to-pc"
WIN = {"target_switch_window": 5, "target_switch_votes": 3}
ATTN = {"identity_attention_release": 4, "identity_attention_off_frac": 0.7, "identity_attention_radius": 0.015}
VERSIONS = {PC + "-idwin": WIN, PC + "-idattn": ATTN, PC + "-idboth": {**WIN, **ATTN}}
ID_KEYS = {"target_switch_window", "target_switch_votes", "identity_attention_release",
           "identity_attention_off_frac", "identity_attention_radius"}


def _identity(version):
    from AG3S.adapters.panda import profile as pf

    argv = pf.version_server_argv(version)
    args = pf.parse_v11ab_args(argv=argv + pf.version_server_flags(version))
    return pf.v11ab_build_kwargs(args)["identity"] or {}


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_versions_parse_and_build_their_clustering_overrides(version, gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.ag3s.config import AG3SConfig

    assert version in pf.AG3S_VERSIONS and version in pf.PC_LATCH_VERSIONS and pf.is_final_version(version)
    ident = _identity(version)
    assert {k: v for k, v in ident.items() if k in ID_KEYS} == VERSIONS[version]
    cfg = AG3SConfig.from_dict({"clustering": ident}).clustering
    cfg.validate()
    assert all(getattr(cfg, k) == v for k, v in VERSIONS[version].items())


def test_the_versions_are_pc_plus_only_their_flags(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    base = list(pf.AG3S_VERSIONS[PC])
    assert list(pf.AG3S_VERSIONS[PC + "-idwin"]) == base + list(pf.ID_WINDOW_FLAGS)
    assert list(pf.AG3S_VERSIONS[PC + "-idattn"]) == base + list(pf.ID_ATTENTION_FLAGS)
    assert list(pf.AG3S_VERSIONS[PC + "-idboth"]) == base + list(pf.ID_WINDOW_FLAGS) + list(pf.ID_ATTENTION_FLAGS)
    assert pf.ID_WINDOW_FLAGS == ("--identity-switch-window", "5", "--identity-switch-votes", "3")
    assert pf.ID_ATTENTION_FLAGS == ("--identity-attention-release", "4", "--identity-attention-off-frac", "0.7",
                                     "--identity-attention-radius-mm", "15")


def test_pc_and_the_other_versions_keep_both_rules_off(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.ag3s.config import ClusteringConfig

    for v in (PC, PC + "-latch", PC + "-surface-fb", "AG3S-final-panda-v3-e0to", "V11AB"):
        assert not set(_identity(v)) & ID_KEYS, v
        assert not any(f.startswith("--identity-") for f in pf.AG3S_VERSIONS[v]), v
    c = ClusteringConfig()
    assert c.target_switch_window is None and c.identity_attention_release is None


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_versions_pass_the_panda_server_checks_and_build_the_served_config(version, gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps

    if not pathlib.Path(pf.PANDA_HSR6_PROBE).exists():
        pytest.skip("Panda HSR6 probe not in this storage root")
    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(version)
    argv[argv.index("--config") + 1] = "pi05_libero"
    args = pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(version))
    ps.check_unported_flags(args, None)
    served, _ = ps.build_safe_server(_Stub(), args, gpu_warm_up=False)
    c = served._policy.ag3s.config.clustering
    assert {k: getattr(c, k) for k in VERSIONS[version]} == VERSIONS[version]
    assert served._policy.ag3s._target_confirm.identity_rules_on
