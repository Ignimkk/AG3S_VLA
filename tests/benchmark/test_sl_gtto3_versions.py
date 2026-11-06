"""SL-GTTO3 — TrajOpt-side variants of `AG3S-commonto-gt2` (lead self-ruling G26, 2026-11-05), CPU.

termsoff = gt2 − `--continuity-after-grasp --deflection-rate-after-grasp` + `--w-continuity 0 --w-deflection-rate 0`;
pass = gt2 + `--passthrough-when-clear`; termsoff-pass = both. Same base argv / per-run rules / client oracle flags as
gt2, read back by `match_ag3s_version`, and gt2's argv is unchanged.
"""
from __future__ import annotations

from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: F401 (fixture)
from tests.benchmark.test_sl_gtto_gt_perception import EXPECTED_GT, _args

GT2 = "AG3S-commonto-gt2"
TERMS = ["--continuity-after-grasp", "--deflection-rate-after-grasp"]
OFF = ["--w-continuity", "0", "--w-deflection-rate", "0"]
GT2_FLAGS = ["--oracle-admissible", "--oracle-held-target"]


def test_gt3_argv(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    assert list(pf.AG3S_VERSIONS[GT2]) == EXPECTED_GT + GT2_FLAGS        # gt2 unchanged
    termsoff = [t for t in EXPECTED_GT + GT2_FLAGS if t not in TERMS] + OFF
    assert list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff"]) == termsoff
    assert list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-pass"]) == EXPECTED_GT + GT2_FLAGS + ["--passthrough-when-clear"]
    assert list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass"]) == termsoff + ["--passthrough-when-clear"]
    for v in pf.GT3_VERSIONS:
        assert pf.is_final_version(v) and pf.version_requires_oracle_segmentation(v)
        assert pf.version_server_argv(v) == pf.version_server_argv(GT2)
        assert pf.version_server_flags(v) == list(pf.AG3S_VERSIONS[v])


def test_gt3_config_and_readback(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss

    for v in (GT2,) + pf.GT3_VERSIONS:
        args = _args(v)
        assert pf.match_ag3s_version(args) == v, v
        to = ss.trajopt_config_from_args(args)
        off = "termsoff" in v
        assert bool(to.cost.continuity_after_grasp) is (not off), v
        assert bool(to.cost.deflection_rate_after_grasp) is (not off), v
        if off:
            assert float(to.cost.w_continuity) == 0.0 and float(to.cost.w_deflection_rate) == 0.0, v
        assert bool(to.sqp.passthrough_when_clear) is v.endswith("pass"), v
        vals = pf.version_flag_values(args)
        assert vals["oracle_admissible"] is True and vals["oracle_held_target"] is True
