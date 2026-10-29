"""T44b A4 — the attention-split ratio on raw (pre-normalisation) attention, the split radius and the
I2b `--held-permission-parent-links` CLI.

T44M (VLA attention, 34 merged demo frames): percentile normalisation saturates the top of the map,
so the target and the neighbour fruit both read ~1.0 and the ratio gate (2.0) fails although the raw
attention separates them. `clustering.attention_split_ratio_source = "raw"` makes the gate read
`AttentionPointCloud.raw_attention`; the target candidate is still the largest normalised mass.

Pinned here:
  - a synthetic rig where normalised and raw disagree, both ways (saturated pair → only raw splits;
    flat raw pair under a stretched normalisation → only normalised splits);
  - the default (`normalized`, `ratio_attention=None`) is the T31b decision and record, key for key;
  - `ground_target` reads `raw_attention` only with `raw`, and falls back (recorded) without one;
  - config round trip / validation, CLI flags → config keys, `main()` refusals, wiring.
"""

from __future__ import annotations

import dataclasses
import inspect
import sys

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.stages.attention_split import (
    ATTENTION_NOT_DISCRIMINATIVE,
    RATIO_NORMALIZED,
    RATIO_RAW,
    RATIO_SOURCES,
    SPLIT,
    split_by_attention,
)
from AG3S.ag3s.stages.target_grounding import TargetConfirm, ground_target
from AG3S.ag3s.types import AttentionPointCloud, PointCloud

CFG = ClusteringConfig()
RAW = dataclasses.replace(CFG, attention_split_ratio_source="raw")
OPENING = 0.07146
APPLE_C = np.array([0.50, 0.00, 0.80])
APPLE_R, ORANGE_R, GAP = 0.031, 0.030, 0.020
ORANGE_C = APPLE_C + np.array([0.0, APPLE_R + GAP + ORANGE_R, 0.0])


def _sphere(centre, radius, n=400, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return np.asarray(centre) + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


APPLE = _sphere(APPLE_C, APPLE_R, seed=0)
ORANGE = _sphere(ORANGE_C, ORANGE_R, seed=1)
PTS = np.concatenate([APPLE, ORANGE])
NA, NO = len(APPLE), len(ORANGE)


def _two(a, o):
    return np.r_[np.full(NA, a), np.full(NO, o)].astype(np.float64)


# Saturated pair (the T44M failure): normalised both ~1.0, raw apple 4.5x the orange.
SAT_NORM = _two(1.0, 0.98)
SAT_RAW = _two(0.009, 0.002)
# The opposite: normalisation stretched a tiny raw difference (raw 1.2x, normalised 3x).
STR_NORM = _two(0.9, 0.3)
STR_RAW = _two(0.0060, 0.0050)


def _split(att, ratio_attention=None, **kw):
    kw.setdefault("max_opening", OPENING)
    kw.setdefault("radius", CFG.attention_split_radius)
    kw.setdefault("min_points", CFG.min_points)
    kw.setdefault("min_ratio", CFG.attention_split_ratio)
    return split_by_attention(PTS, att, np.arange(len(PTS)), ratio_attention=ratio_attention, **kw)


# ------------------------------------------------------------------------------- 1. synthetic rig


def test_saturated_pair_splits_only_on_raw():
    norm = _split(SAT_NORM)
    raw = _split(SAT_NORM, SAT_RAW)
    assert norm.reason == ATTENTION_NOT_DISCRIMINATIVE and norm.ratio < 2.0
    assert norm.ratio_source == RATIO_NORMALIZED
    assert raw.reason == SPLIT and raw.ratio == pytest.approx(4.5)
    assert raw.ratio_source == RATIO_RAW
    np.testing.assert_array_equal(np.sort(raw.target.indices), np.arange(NA))   # the apple
    assert raw.components[0].mean_ratio_attention == pytest.approx(0.009)
    assert raw.components[1].mean_ratio_attention == pytest.approx(0.002)


def test_stretched_pair_splits_only_on_normalized():
    assert _split(STR_NORM).reason == SPLIT
    raw = _split(STR_NORM, STR_RAW)
    assert raw.reason == ATTENTION_NOT_DISCRIMINATIVE and raw.ratio == pytest.approx(1.2)


def test_the_candidate_is_still_the_largest_normalised_mass():
    """Raw says the orange, normalised mass says the apple: the candidate is the apple, and the gate
    (raw apple ÷ raw orange < 1) refuses — raw never picks a different target, it only gates."""
    d = _split(_two(0.9, 0.1), _two(0.001, 0.009))
    np.testing.assert_array_equal(np.sort(d.components[0].indices), np.arange(NA))
    assert d.reason == ATTENTION_NOT_DISCRIMINATIVE and d.ratio < 1.0


def test_raw_record_carries_the_source_and_the_raw_means():
    rec = _split(SAT_NORM, SAT_RAW).record()
    assert rec["ratio_source"] == "raw"
    assert [c["mean_ratio_attention"] for c in rec["components"]] == pytest.approx([0.009, 0.002])


def test_ratio_attention_length_mismatch_raises():
    with pytest.raises(ValueError):
        _split(SAT_NORM, SAT_RAW[:-1])


def test_untriggered_cluster_is_none_in_both_modes():
    assert _split(SAT_NORM, max_opening=0.5) is None
    assert _split(SAT_NORM, SAT_RAW, max_opening=0.5) is None


# --------------------------------------------------------------------- 2. default = T31b, bit for bit

T31B_RECORD_KEYS = {"method", "accepted", "reason", "trigger", "max_opening_mm", "radius_mm",
                    "min_points", "min_ratio", "ratio", "before", "after", "components"}
T31B_COMPONENT_KEYS = {"n_points", "extents_mm", "attention_mass", "mean_attention", "centroid"}


@pytest.mark.parametrize("att", [SAT_NORM, STR_NORM, _two(0.9, 0.1)], ids=["sat", "str", "far"])
def test_default_record_is_the_t31b_record(att):
    d = _split(att)
    rec = d.record()
    assert set(rec) == T31B_RECORD_KEYS
    assert all(set(c) == T31B_COMPONENT_KEYS for c in rec["components"])
    assert all(c.mean_ratio_attention is None for c in d.components)


@pytest.mark.parametrize("seed", range(5))
def test_gate_on_the_same_values_is_the_default_decision(seed):
    """`ratio_attention = attention` changes nothing but the source label: the gate arithmetic is
    the T31b one."""
    rng = np.random.default_rng(seed)
    att = np.r_[rng.uniform(0.2, 1.0, NA), rng.uniform(0.0, 0.6, NO)]
    a, b = _split(att), _split(att, att)
    assert (a.reason, a.ratio) == (b.reason, b.ratio)
    for ca, cb in zip(a.components, b.components):
        np.testing.assert_array_equal(ca.indices, cb.indices)
        assert ca.mean_attention == cb.mean_attention == cb.mean_ratio_attention


def _apc(norm, raw):
    return AttentionPointCloud(PointCloud(PTS), np.asarray(norm, np.float32),
                               None if raw is None else np.asarray(raw, np.float32))


def _ground(apc, cfg, confirm=None):
    return ground_target(apc, cfg, seed_threshold=0.3, confirm=confirm)


def _confirm(cfg):
    return TargetConfirm(cfg, max_opening=OPENING, overlap_distance=0.005)


def test_ground_target_default_ignores_raw_attention():
    """Default config: the raw values on the cloud do not reach the split (normalised gate)."""
    norm = SAT_NORM + np.linspace(0.0, 1e-4, len(PTS))
    with_raw = _ground(_apc(norm, SAT_RAW), dataclasses.replace(CFG, gripper_max_opening=OPENING))
    no_raw = _ground(_apc(norm, None), dataclasses.replace(CFG, gripper_max_opening=OPENING))
    assert [s.reason for s in with_raw.splits] == [s.reason for s in no_raw.splits] == [
        ATTENTION_NOT_DISCRIMINATIVE]
    assert len(with_raw.clusters) == len(no_raw.clusters) == 1
    assert with_raw.clusters[0].split == no_raw.clusters[0].split


def test_ground_target_raw_splits_and_records_the_source():
    norm = SAT_NORM + np.linspace(0.0, 1e-4, len(PTS))
    conf = _confirm(RAW)
    g = _ground(_apc(norm, SAT_RAW), RAW, confirm=conf)
    assert [s.reason for s in g.splits] == [SPLIT]
    roles = sorted(c.split["role"] for c in g.clusters)
    assert roles == ["obstacle", "target"]
    tgt = next(c for c in g.clusters if c.split["role"] == "target")
    np.testing.assert_array_equal(np.sort(tgt.point_indices), np.arange(NA))
    assert conf._split["ratio_source"] == "raw"
    assert conf._split["decisions"][0]["ratio_source"] == "raw"


def test_ground_target_raw_without_raw_attention_falls_back_and_says_so():
    norm = SAT_NORM + np.linspace(0.0, 1e-4, len(PTS))
    conf = _confirm(RAW)
    g = _ground(_apc(norm, None), RAW, confirm=conf)
    assert [s.reason for s in g.splits] == [ATTENTION_NOT_DISCRIMINATIVE]
    assert conf._split["ratio_source"] == "normalized_fallback"
    assert "ratio_source" not in conf._split["decisions"][0]


def test_default_confirm_record_has_no_ratio_source():
    norm = SAT_NORM + np.linspace(0.0, 1e-4, len(PTS))
    conf = _confirm(CFG)
    _ground(_apc(norm, SAT_RAW), CFG, confirm=conf)
    assert "ratio_source" not in conf._split


def test_radius_12mm_reaches_the_split():
    cfg = dataclasses.replace(RAW, attention_split_radius=0.012, gripper_max_opening=OPENING)
    norm = SAT_NORM + np.linspace(0.0, 1e-4, len(PTS))
    g = _ground(_apc(norm, SAT_RAW), cfg)
    assert g.splits[0].radius_m == 0.012 and g.splits[0].reason == SPLIT


# ----------------------------------------------------------------------------------- 3. config


def test_config_default_round_trip_and_validation():
    assert ClusteringConfig().attention_split_ratio_source == "normalized"
    assert ClusteringConfig().attention_split_radius == 0.015
    assert set(RATIO_SOURCES) == {"normalized", "raw"}
    cfg = AG3SConfig.from_dict({"clustering": {"attention_split_ratio_source": "raw",
                                               "attention_split_radius": 0.012}})
    again = AG3SConfig.from_dict(cfg.to_dict())
    assert again.clustering == cfg.clustering
    assert again.clustering.attention_split_ratio_source == "raw"
    for bad in ("Raw", "mean_raw", None, 1):
        with pytest.raises(AG3SConfigError):
            AG3SConfig.from_dict({"clustering": {"attention_split_ratio_source": bad}})


# -------------------------------------------------------------------------------------- 4. CLI

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _parse(*argv):
    from AG3S.trajopt.serve_safe import build_parser

    return build_parser().parse_args([*BASE_ARGV, *argv])


def test_cli_defaults_give_no_keys():
    from AG3S.trajopt import serve_safe

    args = _parse()
    assert serve_safe.split_identity_section(args) == {}
    assert serve_safe.held_permission_overrides(args) == {}
    to = serve_safe.trajopt_config_from_args(args)
    assert to.collision.held_permission_parent_links == 1


def test_cli_split_flags_map_to_clustering_keys():
    from AG3S.trajopt import serve_safe

    args = _parse("--attention-split-radius-mm", "12", "--attention-split-ratio-source", "raw")
    sec = serve_safe.split_identity_section(args)
    assert sec == {"attention_split_radius": 0.012, "attention_split_ratio_source": "raw"}
    cfg = AG3SConfig.from_dict({"clustering": sec})
    assert (cfg.clustering.attention_split_radius,
            cfg.clustering.attention_split_ratio_source) == (0.012, "raw")
    # The keys pass `assert_ag3s_config` once merged with the T43 identity keys.
    full = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": {"backend": "curobo"}})
    serve_safe.assert_ag3s_config(full.with_overrides({"clustering": sec}),
                                  collision_backend="esdf", esdf_backend="curobo", clustering=sec)
    with pytest.raises(RuntimeError):   # a key that did not survive is caught
        serve_safe.assert_ag3s_config(full, collision_backend="esdf", esdf_backend="curobo",
                                      clustering=sec)
    assert serve_safe.split_identity_section(
        _parse("--attention-split-ratio-source", "normalized")) == {
            "attention_split_ratio_source": "normalized"}


def test_cli_held_permission_parent_links_reaches_the_to_config():
    from AG3S.trajopt import serve_safe

    for n in (0, 2):
        args = _parse("--target-permission-strict", "--held-permission-parent-links", str(n))
        assert serve_safe.held_permission_overrides(args) == {"held_permission_parent_links": n}
        to = serve_safe.trajopt_config_from_args(args)
        assert to.collision.held_permission_parent_links == n
        assert to.collision.target_permission_strict is True


def _main_errors(monkeypatch, *argv):
    from AG3S.trajopt import serve_safe

    monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, *argv])
    with pytest.raises(SystemExit) as e:
        serve_safe.main()
    return e.value.code


def test_cli_refusals(monkeypatch, capsys):
    assert _main_errors(monkeypatch, "--held-permission-parent-links", "1") == 2
    assert "--target-permission-strict" in capsys.readouterr().err
    assert _main_errors(monkeypatch, "--target-permission-strict",
                        "--held-permission-parent-links", "-1") == 2
    assert "0 이상" in capsys.readouterr().err
    assert _main_errors(monkeypatch, "--attention-split-radius-mm", "0") == 2
    assert "--attention-split-radius-mm" in capsys.readouterr().err
    assert _main_errors(monkeypatch, "--no-safe", "--attention-split-ratio-source", "raw") == 2
    assert "AG3S" in capsys.readouterr().err


def test_main_passes_the_split_keys_into_build_ag3s():
    from AG3S.trajopt import serve_safe

    src = inspect.getsource(serve_safe.ag3s_from_args)   # `main` builds AG3S through it (AG3S-DYNAMIC-FIELD)
    assert "split_identity_section(args)" in src
    src_to = inspect.getsource(serve_safe.trajopt_config_from_args)
    assert "held_permission_overrides(args)" in src_to
