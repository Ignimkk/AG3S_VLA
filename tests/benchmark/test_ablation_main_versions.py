"""Main ablation table versions (2026-11-16, plan §9.45): rows 4 (-ab7), 3 (-m3), 2 (-m2) on X-fwd."""
from __future__ import annotations


def test_main_ablation_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss

    xf = pf.SOTA_FINAL_X_FWD
    stem = xf[:-len("-tro")]
    r4, r3, r2 = stem + "-ab7-tro", stem + "-m3-tro", stem + "-m2-tro"
    assert pf.SOTA_ABL_MAIN_VERSIONS == (r3, r2) and r4 in pf.SOTA_TAB3_CODE_VERSIONS
    assert set((r4, r3, r2)) <= set(pf.GT15_VERSIONS)
    base = tuple(pf.AG3S_VERSIONS[xf])
    assert pf.AG3S_VERSIONS[r3] == base + ("--target-field-policy", "exclude-all", "--no-attached-query")
    want = {r4: ("ball-owned", "exclude-authorized"), r3: ("ball-owned", "exclude-all"), r2: ("ball", "none")}
    ap = ss.build_parser()
    for v, (excl, pol) in want.items():
        a = pf.parse_v11ab_args(argv=pf.version_server_argv(v) + list(pf.AG3S_VERSIONS[v]))
        ss.reject_bad_flag_combinations(ap, a)
        srv.check_unported_flags(a, None)
        ss.trajopt_config_from_args(a).validate()
        assert a.target_exclusion == excl and a.target_field_policy == pol
        assert a.no_attached_query is True and a.attention_inpass is True
