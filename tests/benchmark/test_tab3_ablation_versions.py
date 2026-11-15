"""TAB3 ablation versions (2026-11-15, plan §9.44): X-fwd with one component replaced; argv parses and passes the checks."""
from __future__ import annotations


def test_tab3_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss

    xf = pf.SOTA_FINAL_X_FWD
    names = tuple(xf[:-len("-tro")] + s + "-tro" for s in ("-ab2", "-ab5", "-ab6"))
    assert pf.SOTA_TAB3_VERSIONS == names and set(names) <= set(pf.GT15_VERSIONS)
    base = tuple(pf.AG3S_VERSIONS[xf])
    for f in ("--target-exclusion", "--owned-max-dist-voxels", "--nontarget-evidence"):
        assert f in base and f not in pf.AG3S_VERSIONS[names[0]] and f not in pf.AG3S_VERSIONS[names[1]]
    assert pf.AG3S_VERSIONS[names[2]] == base + ("--target-field-policy", "exclude-all")
    ap = ss.build_parser()
    want = {names[0]: ("ball", "exclude-authorized"), names[1]: ("ball", "none"), names[2]: ("ball-owned", "exclude-all")}
    for v, (excl, pol) in want.items():
        a = ap.parse_args(pf.version_server_argv(v) + list(pf.version_server_flags(v)))
        ss.reject_bad_flag_combinations(ap, a)
        srv.check_unported_flags(a, None)
        ss.trajopt_config_from_args(a).validate()
        assert a.target_exclusion == excl and a.target_field_policy == pol and a.attention_inpass is True
