"""IFG-SPLIT — attention partition of a merged cluster and a split target replacing an unsplit anchor.

IFG-APPLE (24 apple scenes): when a neighbour fruit touches the apple (gap 4–12 mm), the T31b re-cluster
at 12 mm finds one component and the apple stays merged with it (136–151 mm). Inside the merged cluster the
raw attention still separates the fruits (median 0.05–0.18 vs 3e-5–4e-3). Pinned here:
  - `split_by_attention(partition=False)` is the T31b decision (default), `partition=True` cuts a touching
    pair by attention and keeps every point;
  - a lone fruit with an uneven attention side, or a flat map, is not cut (`partition_no_drop` / flat);
  - `TargetConfirm(split_supersedes_anchor)`: a split target replaces an anchor adopted merged, not a
    lone-object anchor cut by a few centimetres (the hand's shadow);
  - config validation, CLI flags → config keys, `--no-safe` refusal.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.stages.attention_split import (
    ONE_COMPONENT,
    PARTITION_FLAT,
    PARTITION_METHOD,
    PARTITION_NO_DROP,
    SPLIT,
    SPLIT_METHOD,
    split_by_attention,
)
from AG3S.ag3s.stages.target_grounding import (
    SPLIT_SUPERSEDED_ANCHOR,
    SUBSET_KEPT_ANCHOR,
    TargetConfirm,
    ground_target,
)
from AG3S.ag3s.types import AttentionPointCloud, PointCloud

OPENING = 0.07146
RADIUS = 0.012
APPLE_C = np.array([0.50, 0.00, 0.85])
APPLE_R = 0.033
#: The neighbour (an elongated "banana") 6 mm from the apple surface: connected at 12 mm.
GAP = 0.006


def _sphere(centre, radius, n, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return np.asarray(centre) + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


def _banana(n, seed):
    rng = np.random.default_rng(seed)
    t = rng.uniform(-0.045, 0.045, n)
    v = rng.normal(size=(n, 2))
    v = 0.014 * v / np.linalg.norm(v, axis=1, keepdims=True)
    x0 = APPLE_C[0] + APPLE_R + GAP + 0.014
    return np.c_[x0 + v[:, 0], APPLE_C[1] + t, APPLE_C[2] + v[:, 1]]


APPLE = _sphere(APPLE_C, APPLE_R, 300, 0)
BANANA = _banana(200, 1)
PTS = np.concatenate([APPLE, BANANA])
NA = len(APPLE)
MEMBERS = np.arange(len(PTS))
RAW = np.r_[np.full(NA, 0.12), np.full(len(BANANA), 0.002)]
NORM = np.r_[np.full(NA, 1.0), np.full(len(BANANA), 0.05)]


def _split(att, ratio_att=None, *, partition, points=PTS, members=MEMBERS):
    return split_by_attention(points, att, members, max_opening=OPENING, radius=RADIUS, min_points=20,
                              min_ratio=2.0, ratio_attention=ratio_att, partition=partition,
                              partition_min_drop=0.03)


def test_touching_pair_is_one_component_without_partition():
    d = _split(NORM, RAW, partition=False)
    assert d.reason == ONE_COMPONENT and d.method == SPLIT_METHOD
    assert d.record()["method"] == SPLIT_METHOD


def test_partition_cuts_the_touching_pair_by_attention_and_keeps_every_point():
    d = _split(NORM, RAW, partition=True)
    assert d.reason == SPLIT and d.method == PARTITION_METHOD
    tgt = set(d.target.indices.tolist())
    apple = set(range(NA))
    assert len(tgt & apple) / NA >= 0.95          # recall of the apple
    assert len(tgt & apple) / len(tgt) >= 0.95    # precision
    allidx = np.sort(np.concatenate(d.groups()))
    np.testing.assert_array_equal(allidx, MEMBERS)  # no point dropped, none twice
    assert d.record()["method"] == PARTITION_METHOD


def test_partition_reads_the_gate_attention():
    """With the ratio on raw attention the cut is the same as on a normalised map that separates."""
    a = _split(NORM, RAW, partition=True)
    b = _split(NORM, None, partition=True)
    assert a.reason == b.reason == SPLIT
    np.testing.assert_array_equal(np.sort(a.target.indices), np.sort(b.target.indices))


def test_flat_attention_is_not_cut():
    d = _split(np.full(len(PTS), 0.5), np.full(len(PTS), 0.1), partition=True)
    assert d.reason == PARTITION_FLAT and not d.accepted


def test_lone_fruit_with_a_dim_side_is_not_cut():
    """A lone elongated fruit (triggered) whose far half has low attention: cutting it would not shorten
    the cluster by 30 mm, so it stays whole."""
    rng = np.random.default_rng(3)
    v = rng.normal(size=(400, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pts = APPLE_C + v * np.array([0.040, 0.030, 0.030])   # 80 x 60 x 60 mm, triggered at 71.46
    raw = np.where(pts[:, 1] > APPLE_C[1] + 0.012, 0.001, 0.15)
    d = _split(np.clip(raw * 8, 0, 1), raw, partition=True, points=pts, members=np.arange(len(pts)))
    assert d is not None and d.reason == PARTITION_NO_DROP and not d.accepted


def test_untriggered_cluster_is_none_with_partition():
    d = _split(NORM[:NA], RAW[:NA], partition=True, points=APPLE, members=np.arange(NA))
    assert d is None


# ---------------------------------------------------------------- 2. TargetConfirm: split supersedes anchor


def _cfg(**kw):
    return dataclasses.replace(ClusteringConfig(), attention_split_radius=RADIUS,
                               attention_split_ratio_source="raw", gripper_max_opening=OPENING, **kw)


def _ground(cfg, conf, norm, raw):
    apc = AttentionPointCloud(PointCloud(PTS), np.asarray(norm, np.float32), np.asarray(raw, np.float32))
    return ground_target(apc, cfg, seed_threshold=0.3, confirm=conf)


def _run_two_frames(cfg):
    """Frame 1: near-flat attention over the pair (partition: flat) → adopted merged. Frame 2: the apple
    stands out → split."""
    conf = TargetConfirm(cfg, max_opening=OPENING, overlap_distance=0.005)
    _ground(cfg, conf, np.linspace(0.85, 0.95, len(PTS)), np.linspace(0.09, 0.11, len(PTS)))
    first_n = len(np.asarray(conf.manipulated.points).reshape(-1, 3))
    _ground(cfg, conf, NORM, RAW)
    return conf, first_n


def test_split_target_replaces_a_merged_anchor_when_on():
    conf, first_n = _run_two_frames(_cfg(attention_split_partition=True, split_supersedes_anchor=True))
    assert first_n == len(PTS)
    assert conf.last.reason == SPLIT_SUPERSEDED_ANCHOR
    man = conf.manipulated
    assert man.anchor["source"] == "split_superseded" and man.anchor["from_split"]
    n = len(np.asarray(man.points).reshape(-1, 3))
    assert n < len(PTS) and max(man.anchor["extents_mm"]) < 0.08 * 1000


def test_merged_anchor_is_kept_when_off():
    conf, _ = _run_two_frames(_cfg(attention_split_partition=True, split_supersedes_anchor=False))
    assert conf.last.reason == SUBSET_KEPT_ANCHOR
    assert conf.manipulated.anchor["source"] == "first"


def test_defaults_are_off():
    c = ClusteringConfig()
    assert c.attention_split_partition is False and c.split_supersedes_anchor is False
    assert c.attention_partition_min_drop == 0.03


@pytest.mark.parametrize("key,val", [("attention_split_partition", 1), ("split_supersedes_anchor", "yes"),
                                     ("attention_partition_min_drop", -0.01),
                                     ("attention_partition_min_drop", float("nan"))])
def test_config_validation(key, val):
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict({"clustering": {key: val}})


# ----------------------------------------------------------------------------------- 3. CLI


BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _parse(*argv):
    from AG3S.trajopt.serve_safe import build_parser

    return build_parser().parse_args([*BASE_ARGV, *argv])


def test_cli_flags_map_to_clustering_keys():
    from AG3S.trajopt import serve_safe

    assert serve_safe.split_identity_section(_parse()) == {}
    sec = serve_safe.split_identity_section(_parse("--attention-split-partition", "--split-supersedes-anchor"))
    assert sec == {"attention_split_partition": True, "split_supersedes_anchor": True}
    cfg = AG3SConfig.from_dict({"clustering": sec})
    assert cfg.clustering.attention_split_partition and cfg.clustering.split_supersedes_anchor


def test_cli_refuses_the_flags_without_ag3s(monkeypatch, capsys):
    from AG3S.trajopt import serve_safe

    for flag in ("--attention-split-partition", "--split-supersedes-anchor"):
        monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, "--no-safe", flag])
        with pytest.raises(SystemExit) as e:
            serve_safe.main()
        assert e.value.code == 2
        assert flag in capsys.readouterr().err
