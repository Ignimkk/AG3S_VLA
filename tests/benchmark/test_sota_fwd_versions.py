"""FWDATT Tab. 1 versions (sdm-f6, 2026-11-13): ``<v>-fwd-tro`` = ``v`` + ``--attention-inpass`` and nothing else."""
from __future__ import annotations

FROZEN = ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-"
          "pg4-hr150-hb-qs-rs-tro")


def test_fwd_versions_are_base_plus_inpass_only():
    from AG3S.adapters.panda import profile as pf

    bases = (FROZEN, FROZEN[:-4] + "-gh156a-tro", FROZEN[:-4] + "-a12h3-tro")
    assert pf.SOTA_FWD_VERSIONS == tuple(b[:-4] + "-fwd-tro" for b in bases)
    assert set(pf.SOTA_FWD_VERSIONS) <= set(pf.SOTA_VERSIONS)
    for b in bases:
        assert "--attention-inpass" not in pf.AG3S_VERSIONS[b]
        assert tuple(pf.AG3S_VERSIONS[b[:-4] + "-fwd-tro"]) == tuple(pf.AG3S_VERSIONS[b]) + ("--attention-inpass",)


def test_fwd_versions_parse_with_the_flag_on():
    from AG3S.adapters.panda import profile as pf

    for v in pf.SOTA_FWD_VERSIONS:
        args = pf.parse_v11ab_args(argv=pf.version_server_argv(v) + list(pf.AG3S_VERSIONS[v]))
        assert args.attention_inpass is True
    a = pf.parse_v11ab_args(argv=pf.version_server_argv(FROZEN[:-4] + "-a12h3-fwd-tro")
                            + list(pf.AG3S_VERSIONS[FROZEN[:-4] + "-a12h3-fwd-tro"]))
    assert a.attention_source == "action:last:L12H3"
