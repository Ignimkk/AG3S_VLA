"""SL-ATT1 (sdm-f6, 2026-11-13): KNOWS attention cell L12H3 and agent-view-only attention on the frozen version."""
from __future__ import annotations

import argparse

import numpy as np
import pytest

FROZEN = ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-"
          "pg4-hr150-hb-qs-rs-tro")


def test_versions_are_frozen_plus_named_flags_only():
    from AG3S.adapters.panda import profile as pf

    a, h = FROZEN[:-len("-tro")] + "-a12h3-tro", FROZEN[:-len("-tro")] + "-a12h3h-tro"
    assert pf.SOTA_ATT_VERSIONS == (a, h) and set(pf.SOTA_ATT_VERSIONS) <= set(pf.SOTA_VERSIONS)
    base = tuple(pf.AG3S_VERSIONS[FROZEN])
    assert "--attention-source" not in base and "--attention-head-only" not in base
    assert tuple(pf.AG3S_VERSIONS[a]) == base + ("--attention-source", "action:last:L12H3")
    assert tuple(pf.AG3S_VERSIONS[h]) == base + ("--attention-source", "action:last:L12H3", "--attention-head-only")
    assert [v for v, f in pf.AG3S_VERSIONS.items() if "--attention-head-only" in f] == [h]


def test_parser_and_port():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv

    assert "attention_head_only" in srv.PORTED_FINAL_CONFIG_DESTS
    h = FROZEN[:-len("-tro")] + "-a12h3h-tro"
    args = pf.parse_v11ab_args(argv=pf.version_server_argv(h) + list(pf.AG3S_VERSIONS[h]))
    assert args.attention_head_only is True and args.attention_source == "action:last:L12H3"
    base = pf.parse_v11ab_args(argv=pf.version_server_argv(FROZEN) + list(pf.AG3S_VERSIONS[FROZEN]))
    assert base.attention_head_only is False and base.attention_source is None


def test_rby1_refuses_head_only():
    from AG3S.trajopt import serve_safe

    ap = serve_safe.build_parser()
    with pytest.raises(SystemExit):
        serve_safe.reject_panda_only_flags(ap, argparse.Namespace(robot="rby1", attention_head_only=True))
    serve_safe.reject_panda_only_flags(ap, argparse.Namespace(robot="panda", attention_head_only=True))
    assert ap.get_default("attention_head_only") is False


def test_extractor_drops_the_wrist_map_only_when_head_only():
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss

    g = {"cam_high": np.full((16, 16), 1e-4, np.float32), "cam_left_wrist": np.full((16, 16), 0.3, np.float32)}
    both = ss.attention_extractor(alias=srv.attention_alias(False), cameras=srv.attention_cameras(False))(None, {"attention": g})
    head = ss.attention_extractor(alias=srv.attention_alias(True), cameras=srv.attention_cameras(True))(None, {"attention": g})
    assert set(both) == {"head", "left_wrist"} and set(head) == {"head"}
    assert srv.attention_alias(False) == srv.ATTENTION_ALIAS and srv.attention_cameras(False) == srv.ATTENTION_CAMERAS
    # the policy may also name the cameras by wire name: still only head passes
    head2 = ss.attention_extractor(alias=srv.attention_alias(True), cameras=srv.attention_cameras(True))(
        None, {"attention": {"head": g["cam_high"], "left_wrist": g["cam_left_wrist"]}})
    assert set(head2) == {"head"}
