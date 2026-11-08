"""SL-GTTO10 (lead self-ruling G53, 2026-11-07): F8 = F7 + GT destination (F8a) + held / hand voxel freeing (F8b) + opening-rule latch (F8c)."""
from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.adapters.panda import profile as pf
from AG3S.adapters.panda import server as srv
from AG3S.ag3s.config import AG3SConfig, AG3SConfigError
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.trajopt import serve_safe as ss

V = "AG3S-commonto-gt2-termsoff-pass-zr-ee-hdp"


def test_version_is_f7_plus_the_f8_flags():
    assert pf.GT10_VERSIONS == (V,)
    assert list(pf.AG3S_VERSIONS[V]) == list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee"]) + list(pf.GT10_FLAGS)
    assert pf.is_gtto_version(V) and pf.is_final_version(V) and pf.version_requires_oracle_segmentation(V)
    assert {"oracle_destination", "destination_margin_mm"} <= set(srv.PORTED_FINAL_CONFIG_DESTS)
    for name, flags in pf.AG3S_VERSIONS.items():
        if name != V and name not in getattr(pf, "GT11_VERSIONS", ()) + getattr(pf, "GT12_VERSIONS", ()) + getattr(pf, "GT13_VERSIONS", ()) + getattr(pf, "GT14_VERSIONS", ()) + getattr(pf, "GT15_VERSIONS", ()):   # SL-GTTO11 / 12 build on F8
            assert "--oracle-destination" not in flags and "--destination-margin-mm" not in flags, name


def test_flags_reach_the_ag3s_config():
    args = pf.parse_v11ab_args(argv=pf.version_server_argv(V) + list(pf.AG3S_VERSIONS[V]))
    kw = pf.v11ab_build_kwargs(args)
    assert kw["contact"] == {"destination_margin": 0.0}
    assert kw["held_free_query_spheres"] is True and kw["hand_voxel_carve"] == {"mode": "current", "radius_m": 0.02}
    cfg_dict, identity = pf.v11ab_config_dict(kw)
    cfg = AG3SConfig.from_dict(cfg_dict)
    assert cfg.contact.destination_margin == 0.0
    assert identity["oracle_destination"] is True and identity["grounding_source"] == "oracle"   # clustering keys: identity overrides
    vals = pf.version_flag_values(args)
    assert vals["oracle_destination"] is True and vals["destination_margin_mm"] == 0.0 and vals["grasp_opening_rule"] is True
    # F7 (no F8 flag): neither key joins its record, config default kept
    a7 = pf.parse_v11ab_args(argv=pf.version_server_argv(V) + list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-ee"]))
    v7 = pf.version_flag_values(a7)
    assert "oracle_destination" not in v7 and "destination_margin_mm" not in v7
    assert pf.v11ab_build_kwargs(a7).get("contact") is None


def test_destination_needs_the_gt_grounding():
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"clustering": {"oracle_destination": True}})
    assert ss.destination_margin_section(types.SimpleNamespace(destination_margin_mm=None)) == {}
    assert ss.destination_margin_section(types.SimpleNamespace(destination_margin_mm=5.0)) == {"destination_margin": 0.005}


def _fusion(points, obs, n_cams=1):
    """Fake `FusedPointCloud`-like object: obs = per point list of (cam, u, v)."""
    offs = np.cumsum([0] + [len(o) for o in obs])
    flat = [x for o in obs for x in o]
    cloud = types.SimpleNamespace(
        points=np.asarray(points, np.float64), n_observations=len(flat), cameras=[types.SimpleNamespace(value="head")] * n_cams,
        obs_camera=np.asarray([c for c, _, _ in flat], np.int16), obs_uv=np.asarray([[u, v] for _, u, v in flat], np.int32),
        obs_offset=np.asarray(offs, np.int64))
    cloud.__len__ = lambda: len(points)
    return types.SimpleNamespace(cloud=_Len(cloud, len(points)))


class _Len:
    def __init__(self, inner, n):
        self._inner, self._n = inner, n

    def __getattr__(self, k):
        return getattr(self._inner, k)

    def __len__(self):
        return self._n


def _call(meta, labels, fusion):
    stub = types.SimpleNamespace(_oracle_pending={"meta": meta, "labels": labels})
    return AG3S._oracle_destination_points(stub, fusion)


def test_destination_points_are_the_unanimous_code_points():
    lab = np.zeros((4, 4), np.uint8)
    lab[0, 0] = 5          # destination code
    lab[0, 1] = 5
    lab[1, 1] = 1          # target
    pts = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    obs = [[(0, 0, 0)], [(0, 1, 0), (0, 1, 1)], [(0, 1, 1)], [(0, 3, 3)]]   # point 1 is mixed (5 and 1)
    meta = {"destination": "left_hand_1", "codes": {"akita_black_bowl_1": 1, "left_hand_1": 5}}
    got, rec = _call(meta, {"head": lab}, _fusion(pts, obs))
    assert rec["code"] == 5 and rec["n_points"] == 1 and rec["why"] is None
    assert np.allclose(got, [[0, 0, 0]])


@pytest.mark.parametrize("meta,why", [({"codes": {"a": 1}}, "no_destination_named"),
                                      ({"destination": "plate_1", "codes": {"a": 1}}, "destination_not_in_legend")])
def test_no_destination_no_injection(meta, why):
    lab = np.zeros((2, 2), np.uint8)
    got, rec = _call(meta, {"head": lab}, _fusion([[0, 0, 0]], [[(0, 0, 0)]]))
    assert got is None and rec["why"] == why


def test_client_meta_names_the_destination_only_when_the_goal_has_one():
    import pathlib
    import sys
    root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "benchmark" / "adapters" / "libero_panda"))
    import gtto_oracle_codes as oc

    c = oc.OracleCoder("bowl_1", ["bowl_1", "left_hand_1"], source={"rule": "r", "destination": "left_hand_1"})
    assert c.meta()["destination"] == "left_hand_1"
    assert "destination" not in oc.OracleCoder("bowl_1", ["bowl_1"], source={"rule": "r"}).meta()
