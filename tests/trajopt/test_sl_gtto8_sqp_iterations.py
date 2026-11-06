"""SL-GTTO8 (F6, lead self-ruling G45, 2026-11-05): the zr version with the SQP run to 8 iterations.

Only the profile entry is new (``--sqp-min-iterations`` / ``--sqp-max-iterations`` already exist in serve_safe, T31).
"""
from __future__ import annotations


def test_it8_version_is_zr_plus_sqp_iterations(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss
    from tests.benchmark.test_sl_gtto_gt_perception import _args

    v = "AG3S-commonto-gt2-termsoff-pass-zr-it8"
    assert pf.GT8_VERSIONS == (v,)
    assert list(pf.AG3S_VERSIONS[v]) == list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr"]) + [
        "--sqp-min-iterations", "8", "--sqp-max-iterations", "8"]
    assert pf.is_final_version(v) and pf.version_requires_oracle_segmentation(v)
    args = _args(v)
    assert pf.match_ag3s_version(args) == v
    values = pf.version_flag_values(args)
    assert values["sqp_min_iterations"] == 8 and values["sqp_max_iterations"] == 8
    zr = pf.version_flag_values(_args("AG3S-commonto-gt2-termsoff-pass-zr"))
    assert "sqp_min_iterations" not in zr and "sqp_max_iterations" not in zr
    to = ss.trajopt_config_from_args(args)
    assert to.sqp.min_iterations == 8 and to.sqp.max_iterations == 8
    assert to.sqp.passthrough_when_clear
    base = ss.trajopt_config_from_args(_args("AG3S-commonto-gt2-termsoff-pass-zr"))
    assert base.sqp.min_iterations == 3 and base.sqp.max_iterations == 3
    for name, flags in pf.AG3S_VERSIONS.items():
        if name != v:
            assert "--sqp-min-iterations" not in flags and "--sqp-max-iterations" not in flags, name


from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: E402,F401 (fixture)
