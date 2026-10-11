"""Config tests: the ablation axes must be reachable, and typos must be loud.

The spec requires nine ablations to be reachable "by config alone". Each is asserted here rather
than being taken on trust, because the failure mode is silent: a run that ignores an override
reports the default's numbers under the variant's name.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, PhaseRule
from AG3S.ag3s.types import Phase, PrimitiveType

DEFAULT_YAML = pathlib.Path("AG3S/ag3s/configs/default.yaml")


def test_defaults_match_the_spec_section_9_table():
    cfg = AG3SConfig()
    assert cfg.attention.normalization == "percentile"
    assert cfg.attention.seed_percentile == 95
    assert cfg.attention.temperature == 1.0
    assert cfg.clustering.method == "dbscan"
    assert (cfg.clustering.eps, cfg.clustering.min_points) == (0.03, 20)
    assert cfg.pointcloud.voxel_size == 0.005
    assert cfg.pointcloud.self_filter is True
    assert cfg.pointcloud.max_points == 200000
    assert cfg.collision_candidate.cluster_method == "dbscan"
    assert cfg.collision_candidate.preserve_unknown is True
    assert cfg.geometry.primitive == "sphere"
    assert cfg.geometry.safety_margin == 0.05
    assert cfg.geometry.max_candidates == 32
    assert cfg.contact.phase_aware is True
    assert cfg.constraint.squared_distance is True
    assert cfg.constraint.horizon == 20


def test_shipped_yaml_loads_and_equals_the_defaults():
    """`configs/default.yaml` must be documentation, not a second source of truth."""
    if not DEFAULT_YAML.exists():
        pytest.skip(f"{DEFAULT_YAML} not found (run from the workspace root)")
    assert AG3SConfig.from_yaml(DEFAULT_YAML) == AG3SConfig()


def test_roundtrip_through_dict_is_lossless():
    cfg = AG3SConfig()
    assert AG3SConfig.from_dict(cfg.to_dict()) == cfg


# ------------------------------------------------------------------- the nine ablation axes


@pytest.mark.parametrize(
    "overrides, check",
    [
        ({"attention": {"normalization": "softmax"}}, lambda c: c.attention.normalization == "softmax"),
        ({"attention": {"layer": 12, "head": 3}}, lambda c: (c.attention.layer, c.attention.head) == (12, 3)),
        ({"attention": {"seed_percentile": 80.0}}, lambda c: c.attention.seed_percentile == 80.0),
        ({"clustering": {"use_3d_connectivity": False}}, lambda c: c.clustering.use_3d_connectivity is False),
        ({"clustering": {"method": "region_growing"}}, lambda c: c.clustering.method == "region_growing"),
        ({"pointcloud": {"self_filter": False}}, lambda c: c.pointcloud.self_filter is False),
        ({"geometry": {"primitive": "capsule"}}, lambda c: c.geometry.primitive_type is PrimitiveType.CAPSULE),
        ({"geometry": {"safety_margin": 0.12}}, lambda c: c.geometry.safety_margin == 0.12),
        ({"contact": {"phase_aware": False}}, lambda c: c.contact.phase_aware is False),
    ],
    ids=[
        "attention-normalization",
        "attention-layer-head",
        "seed-threshold",
        "2d-only-vs-3d-grounding",
        "clustering-method",
        "self-filter-toggle",
        "primitive-type",
        "safety-margin",
        "phase-aware-toggle",
    ],
)
def test_every_required_ablation_is_reachable_by_config_alone(overrides, check):
    assert check(AG3SConfig().with_overrides(overrides))


def test_with_overrides_leaves_untouched_fields_alone():
    """A partial override must not silently reset its section's other fields."""
    cfg = AG3SConfig().with_overrides({"geometry": {"primitive": "box"}})
    assert cfg.geometry.primitive == "box"
    assert cfg.geometry.safety_margin == AG3SConfig().geometry.safety_margin
    assert cfg.clustering == AG3SConfig().clustering


# ----------------------------------------------------------------------------- validation


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(AG3SConfigError, match="unknown top-level"):
        AG3SConfig.from_dict({"attenton": {}})


def test_unknown_section_key_is_rejected():
    """A misplaced `seed_percentile` must fail loudly, not run the default silently."""
    with pytest.raises(AG3SConfigError, match="unknown key"):
        AG3SConfig.from_dict({"clustering": {"seed_percentile": 95}})


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"attention": {"normalization": "zscore"}}, "normalization"),
        ({"attention": {"seed_percentile": 140}}, "seed_percentile"),
        ({"attention": {"temperature": 0.0}}, "temperature"),
        ({"clustering": {"eps": 0.0}}, "eps"),
        ({"clustering": {"min_points": 0}}, "min_points"),
        ({"pointcloud": {"max_points": 0}}, "max_points"),
        ({"pointcloud": {"depth_min": 5.0, "depth_max": 1.0}}, "depth_min"),
        ({"geometry": {"primitive": "torus"}}, "primitive"),
        ({"geometry": {"safety_margin": -0.01}}, "safety_margin"),
        ({"constraint": {"horizon": 0}}, "horizon"),
    ],
)
def test_out_of_range_values_are_rejected(overrides, match):
    with pytest.raises(AG3SConfigError, match=match):
        AG3SConfig().with_overrides(overrides)


# ---------------------------------------------------------------------------- phase rules


def test_default_phase_rules_follow_the_spec_progression():
    """TRANSIT enabled -> APPROACH reduced -> PRE_GRASP controlled -> GRASP contact allowed.

    The scales apply to the *contact-authorized* links only; see `test_clearance.py` for what they
    mean once a link name is in play.
    """
    contact = AG3SConfig().contact
    scales = [contact.rule_for(p).margin_scale for p in
              (Phase.TRANSIT, Phase.APPROACH, Phase.PRE_GRASP, Phase.GRASP)]
    assert scales == sorted(scales, reverse=True), f"margin must not grow through the phases: {scales}"
    assert contact.rule_for(Phase.TRANSIT).collision_enabled is True
    assert contact.rule_for(Phase.GRASP).contact_permission is True


def test_no_phase_may_disable_the_target_constraint():
    """The safety invariant that replaced the old `collision_enabled=False` at GRASP.

    Disabling a candidate removes its rows from the NLP, so the robot may pass through it freely.
    Permitted contact is a margin of zero on named links, which still forbids penetration. A config
    that tries to express contact by disabling is rejected rather than quietly obeyed.
    """
    for phase in Phase:
        assert AG3SConfig().contact.rule_for(phase).collision_enabled is True

    with pytest.raises(AG3SConfigError, match="collision_enabled must be True"):
        AG3SConfig().with_overrides({"contact": {"phase_rules": {
            **AG3SConfig().to_dict()["contact"]["phase_rules"],
            "grasp": {"collision_enabled": False, "contact_permission": True, "margin_scale": 0.0},
        }}})


def test_a_negative_contact_margin_is_rejected():
    """Zero means "may touch". Negative would mean "may penetrate", which is not a clearance."""
    with pytest.raises(AG3SConfigError, match="contact_margin must be >= 0"):
        AG3SConfig().with_overrides({"contact": {"contact_margin": -0.01}})


def test_contact_links_are_per_manipulator_and_reject_unknown_names():
    contact = AG3SConfig().contact
    assert "ee_finger_r1" in contact.links_for("right")
    assert "ee_finger_r1" not in contact.links_for("left")
    # An unknown manipulator authorizes nothing rather than raising into the safety path.
    assert contact.links_for("third_arm") == ()
    with pytest.raises(ValueError):
        AG3SConfig().with_overrides({"contact": {"contact_links": {"tail": ["link_a"]}}})


def test_phase_aware_off_pins_every_phase_to_full_margin():
    contact = dataclasses.replace(AG3SConfig().contact, phase_aware=False)
    for phase in Phase:
        rule = contact.rule_for(phase)
        assert rule == PhaseRule(collision_enabled=True, contact_permission=False, margin_scale=1.0)


def test_phase_rules_can_be_overridden_from_yaml_shaped_dicts():
    cfg = AG3SConfig().with_overrides(
        {"contact": {"phase_rules": {
            "transit": {"collision_enabled": True, "contact_permission": False, "margin_scale": 2.0},
            "approach": {"collision_enabled": True, "contact_permission": False, "margin_scale": 1.0},
            "pre_grasp": {"collision_enabled": True, "contact_permission": False, "margin_scale": 0.5},
            "grasp": {"collision_enabled": True, "contact_permission": True, "margin_scale": 0.0},
        }}}
    )
    assert cfg.contact.rule_for(Phase.TRANSIT).margin_scale == 2.0


def test_missing_phase_rule_is_rejected():
    with pytest.raises(AG3SConfigError, match="missing"):
        AG3SConfig().with_overrides({"contact": {"phase_rules": {"transit": {
            "collision_enabled": True, "contact_permission": False, "margin_scale": 1.0}}}})


def test_unknown_phase_name_is_rejected():
    with pytest.raises(ValueError, match="unknown phase"):
        AG3SConfig().with_overrides({"contact": {"phase_rules": {
            **AG3SConfig().to_dict()["contact"]["phase_rules"],
            "hover": {"collision_enabled": True, "contact_permission": False, "margin_scale": 1.0},
        }}})


def test_phase_parse_accepts_strings_and_enums():
    assert Phase.parse("approach") is Phase.APPROACH
    assert Phase.parse(Phase.GRASP) is Phase.GRASP
    with pytest.raises(ValueError, match="unknown phase"):
        Phase.parse("hovering")
