"""SL-SOTA18 -- hand-sized clusters are graspable (`--graspable-hand-sized-mm`, `clustering.graspable_hand_sized`; sdm-f6
2026-11-13). CPU.

(a) rule: off = the T26 rule bit for bit; on = narrowest <= opening OR largest <= the hand size (a mug passes; a crate, a
    basket, a book do not); unknown inputs stay unknown;
(b) assess: the mug's reason goes from not_graspable to admissible (the destination-overlap branch is after it, unchanged);
(c) config validation, server option -> clustering key, version, the exclusion gate's twin check.
"""
from __future__ import annotations

import numpy as np
import pytest

OPEN = 0.078
HAND = 0.156
MUG = (0.100, 0.115, 0.135)        # measured in the 1/10 table run (mm: 97-120, 98-131, 130-141)
MOKA = (0.080, 0.136, 0.149)
CRATE = (0.185, 0.190, 0.213)
BOOK = (0.114, 0.145, 0.233)
BOWL = (0.055, 0.140, 0.140)
FROZEN = ("AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-"
          "hr150-hb-qs-rs-tro")


def _box(ext, n=400, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-0.5, 0.5, (n, 3)) * np.asarray(ext)) + np.array([0.5, 0.0, 0.1])


def test_rule():
    from AG3S.ag3s.stages.admissibility import graspable

    for ext in (MUG, MOKA, CRATE, BOOK, BOWL, (0.03, 0.05, 0.3)):
        assert graspable(np.array(ext), OPEN) == (min(ext) <= OPEN)                 # off: the T26 rule
        assert graspable(np.array(ext), OPEN, None) == graspable(np.array(ext), OPEN)
    assert graspable(np.array(MUG), OPEN, HAND) is True and graspable(np.array(MOKA), OPEN, HAND) is True
    assert graspable(np.array(CRATE), OPEN, HAND) is False and graspable(np.array(BOOK), OPEN, HAND) is False
    assert graspable(np.array(BOWL), OPEN, HAND) is True                             # already graspable by its height
    assert graspable(np.array((0.03, 0.05, 0.3)), OPEN, HAND) is True                # long and thin: by the narrowest extent
    assert graspable(None, OPEN, HAND) is None and graspable(np.array(MUG), None, HAND) is None
    assert graspable(np.array([np.nan, 0.1, 0.1]), OPEN, HAND) is None


def test_assess_reason():
    from AG3S.ag3s.stages import admissibility as ad

    mug = _box(MUG)
    # extents given: the rule is on the principal extents the grounding measured (PCA axes of a near-cubic sample can
    # sit on a diagonal, so a synthetic box is not a stand-in for the recorded mug extents)
    kw = dict(max_opening_m=OPEN, destination=None, overlap_distance=0.01, overlap_fraction=0.5)
    off = ad.assess(mug, extents=np.array(MUG), **kw)
    on = ad.assess(mug, extents=np.array(MUG), hand_sized_m=HAND, **kw)
    assert off.reason == ad.NOT_GRASPABLE and not off.admissible
    assert on.reason == ad.ADMISSIBLE and on.graspable is True and on.narrowest_extent_m > OPEN
    assert off.extents_m == on.extents_m and off.record().keys() == on.record().keys()
    assert ad.assess(_box(CRATE), extents=np.array(CRATE), hand_sized_m=HAND, **kw).reason == ad.NOT_GRASPABLE
    bowl = _box(BOWL)
    assert (ad.assess(bowl, extents=np.array(BOWL), **kw).reason
            == ad.assess(bowl, extents=np.array(BOWL), hand_sized_m=HAND, **kw).reason == ad.ADMISSIBLE)


def test_config_option_version_and_gate():
    from AG3S.ag3s.config import AG3SConfigError, ClusteringConfig
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe

    assert ClusteringConfig().graspable_hand_sized is None
    ClusteringConfig(graspable_hand_sized=HAND).validate()
    for bad in (0.0, -0.1, float("nan"), True, "0.1"):
        with pytest.raises(AG3SConfigError):
            ClusteringConfig(graspable_hand_sized=bad).validate()
    name = FROZEN[:-len("-tro")] + "-gh156-tro"
    name_hf = FROZEN[:-len("-tro")] + "-hf30-gh156-tro"                                 # + SL-SOTA17's held band
    assert pf.SOTA_HANDSIZED_VERSIONS == (name, name_hf) and name in pf.SOTA_VERSIONS and name_hf in pf.SOTA_VERSIONS
    assert pf.AG3S_VERSIONS[name] == tuple(pf.AG3S_VERSIONS[FROZEN]) + ("--graspable-hand-sized-mm", "156")
    assert pf.AG3S_VERSIONS[name_hf] == (tuple(pf.AG3S_VERSIONS[FROZEN]) + ("--obstacle-held-beneath-band-mm", "30")
                                         + ("--graspable-hand-sized-mm", "156"))
    assert [v for v, f in pf.AG3S_VERSIONS.items() if "--graspable-hand-sized-mm" in f] == [
        name, name_hf, name[:-len("-gh156-tro")] + "-gh156a-tro", name_hf[:-len("-gh156-tro")] + "-gh156a-tro"]
    assert "graspable_hand_sized_mm" in srv.PORTED_FINAL_CONFIG_DESTS
    args = pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(pf.AG3S_VERSIONS[name]))
    assert args.graspable_hand_sized_mm == 156.0
    assert serve_safe.t43t_identity_section(args)["graspable_hand_sized"] == pytest.approx(0.156)
    base = pf.parse_v11ab_args(argv=pf.version_server_argv(FROZEN) + list(pf.AG3S_VERSIONS[FROZEN]))
    assert base.graspable_hand_sized_mm is None and "graspable_hand_sized" not in serve_safe.t43t_identity_section(base)
    assert pf.version_server_flags(name)

    # the gate's twin of the rule
    from AG3S.ag3s.runtime.pipeline import AG3S as Pipeline
    from AG3S.ag3s.stages import admissibility as ad

    class Stub:
        pass

    for hand, ext, want in ((None, MUG, False), (HAND, MUG, True), (HAND, CRATE, False)):
        s = Stub()
        s.config = Stub()
        s.config.clustering = ClusteringConfig(graspable_hand_sized=hand)
        adm = ad.assess(_box(ext), extents=np.array(ext), max_opening_m=OPEN, destination=None, overlap_distance=0.01,
                        overlap_fraction=0.5, hand_sized_m=hand)
        assert Pipeline._hand_sized(s, adm) is want


def test_attention_gate_sl_sota18b():
    """SL-SOTA18b: a hand-sized-only cluster stays admissible only when it leads on mean attention."""
    import dataclasses as dc
    from AG3S.ag3s.config import AG3SConfigError, ClusteringConfig
    from AG3S.ag3s.stages import admissibility as ad
    from AG3S.ag3s.stages.target_grounding import TargetConfirm
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe

    kw = dict(max_opening_m=OPEN, destination=None, overlap_distance=0.01, overlap_fraction=0.5, hand_sized_m=HAND)
    mug = ad.assess(_box(MUG), extents=np.array(MUG), **kw)
    bowl = ad.assess(_box(BOWL), extents=np.array(BOWL), **kw)
    crate = ad.assess(_box(CRATE), extents=np.array(CRATE), **kw)
    assert mug.admissible and bowl.admissible and not crate.admissible

    @dc.dataclass
    class C:
        mean_attention: float

    tc = TargetConfirm.__new__(TargetConfirm)
    tc.max_opening = OPEN
    # E1a t0 e10: the red-mug obstacle leads the score but not the attention -> outranked; the bowl is untouched
    out = tc._hand_sized_attention_gate([mug, bowl], [C(0.07), C(0.33)])
    assert out[0].reason == ad.HAND_SIZED_OUTRANKED and not out[0].admissible and out[1] is bowl
    # the task's mug leads on attention -> admissible
    out = tc._hand_sized_attention_gate([mug, bowl], [C(0.40), C(0.10)])
    assert out[0] is mug and out[0].admissible
    # an inadmissible rival does not count; two mugs: the less attended one is outranked
    assert tc._hand_sized_attention_gate([mug, crate], [C(0.05), C(0.9)])[0] is mug
    out = tc._hand_sized_attention_gate([mug, mug], [C(0.2), C(0.3)])
    assert out[0].reason == ad.HAND_SIZED_OUTRANKED and out[1] is mug
    # no attention on the cluster record: unchanged
    assert tc._hand_sized_attention_gate([mug, bowl], [object(), C(0.3)])[0] is mug
    # config / option / version plumbing
    ClusteringConfig(graspable_hand_sized=HAND, graspable_hand_sized_attention_top=True).validate()
    with pytest.raises(AG3SConfigError):
        ClusteringConfig(graspable_hand_sized_attention_top=True).validate()
    name = FROZEN[:-len("-tro")] + "-gh156a-tro"
    assert name in pf.SOTA_HANDSIZED_ATT_VERSIONS and name in pf.SOTA_VERSIONS
    args = pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(pf.AG3S_VERSIONS[name]))
    sec = serve_safe.t43t_identity_section(args)
    assert sec["graspable_hand_sized_attention_top"] is True and sec["graspable_hand_sized"] == pytest.approx(HAND)
    base = pf.parse_v11ab_args(argv=pf.version_server_argv(FROZEN[:-len("-tro")] + "-gh156-tro")
                               + list(pf.AG3S_VERSIONS[FROZEN[:-len("-tro")] + "-gh156-tro"]))
    assert "graspable_hand_sized_attention_top" not in serve_safe.t43t_identity_section(base)
