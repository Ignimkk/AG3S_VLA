"""SOTA clearance candidates (dev-57, 2026-11-11): `-to5` / `-to0` (+ `-tro`) are the base
version's argv plus the named flags only; the evaluated bases are unchanged; the server parser takes the later value."""
from AG3S.adapters.panda import profile as pf

BASE = "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-"


def test_names_and_flags():
    assert len(pf.SOTA_VERSIONS) == 8 + len(pf.SOTA_OM_VERSIONS) + len(pf.SOTA_NTC_VERSIONS)
    assert set(pf.SOTA_VERSIONS) <= set(pf.GT15_VERSIONS)
    for b in (BASE + "pe123458", BASE + "pe12345811"):
        base = tuple(pf.AG3S_VERSIONS[b])
        assert "--ee-action-qp-translation-only" not in base and base.count("--passthrough-obstacle-tolerance-mm") == 1
        for suffix, flags in pf._SOTA_SUFFIXES.items():
            assert tuple(pf.AG3S_VERSIONS[b + suffix]) == base + tuple(flags)
            assert tuple(pf.AG3S_VERSIONS[b + suffix + "-tro"]) == base + tuple(flags) + ("--ee-action-qp-translation-only",)
            assert pf.version_server_flags(b + suffix)   # accepted by the Panda profile (final-base rules)


def test_later_value_wins_in_the_server_parser():
    from AG3S.trajopt import serve_safe as ss
    from tests.trajopt.test_t88_passthrough_when_clear import BASE as ARGV
    a = ss.build_parser().parse_args(ARGV + ["--passthrough-obstacle-tolerance-mm", "10",
                                              "--passthrough-obstacle-tolerance-mm", "0"])
    assert float(a.passthrough_obstacle_tolerance_mm) == 0.0


def test_v11ab_record_keys_unchanged():
    fam = pf.version_flag_values(pf.parse_v11ab_args())
    assert "palm_shell" not in fam and "passthrough_obstacle_tolerance_mm" not in fam


def test_no_target_certify_versions():
    assert len(pf.SOTA_NTC_VERSIONS) == 24 and len(set(pf.SOTA_VERSIONS)) == len(pf.SOTA_VERSIONS)
    for name in pf.SOTA_NTC_VERSIONS:
        parent = name[:-len("-tro")] if name.endswith("-tro") else name[:-len("-ntc")]
        extra = "--ee-action-qp-translation-only" if name.endswith("-tro") else "--no-target-certify"
        assert tuple(pf.AG3S_VERSIONS[name]) == tuple(pf.AG3S_VERSIONS[parent]) + (extra,)
        assert tuple(pf.AG3S_VERSIONS[name]).count("--no-target-certify") == 1
        assert pf.version_server_flags(name)
