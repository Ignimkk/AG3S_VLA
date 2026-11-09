"""AEGIS name-wording fidelity fix (2026-11-09, user-approved): the privileged-names arm hands Grounding DINO the official
``prefer_list`` wording (benchmark/SafeLIBERO/main/utils.py:466) -- what a correct GLM-4.5V answer looks like -- instead
of the display name. Other arms / methods keep the display wording (default ``name_wording="display"``)."""
from __future__ import annotations

import pytest

from Baseline.paper_baselines.methods.aegis.adapter import AegisSafetyFilter
from Baseline.paper_baselines.methods.safety_common import (AEGIS_PREFER_LIST_WORDING, ObstacleCloudSource,
                                                            obstacle_query_phrase)

OFFICIAL = ['yellow rectangular book', 'blue moka pot', 'red mug', 'white storage box', 'black wine bottle',
            'red milk carton', 'gray rectangular binder']


@pytest.mark.parametrize("display,official", [
    ("yellow_book_1", "yellow rectangular book"), ("moka_pot_1", "blue moka pot"), ("moka_pot_small_1", "blue moka pot"),
    ("red_coffee_mug_1", "red mug"), ("white_storage_box_1", "white storage box"),
    ("wine_bottle_1", "black wine bottle"), ("wine_bottle_small_1", "black wine bottle"),
    ("milk_1", "red milk carton"), ("milk_small_1", "red milk carton")])
def test_every_safelibero_obstacle_maps_to_the_official_list(display, official):
    assert obstacle_query_phrase(display, "aegis_prefer_list") == official
    assert official in OFFICIAL


def test_display_wording_is_unchanged_and_unknown_names_fall_back():
    assert obstacle_query_phrase("moka_pot_1") == "moka pot"
    assert obstacle_query_phrase("akita_black_bowl_1", "aegis_prefer_list") == "akita black bowl"
    assert set(AEGIS_PREFER_LIST_WORDING.values()) <= set(OFFICIAL)
    with pytest.raises(ValueError):
        obstacle_query_phrase("milk_1", "other")


def test_only_aegis_defaults_to_the_official_wording():
    assert ObstacleCloudSource("detector", detector_prompt="obstacle_names").name_wording == "display"
    f = AegisSafetyFilter("detector", detector_prompt="obstacle_names")
    assert f.source.name_wording == "aegis_prefer_list"
    assert f.source.provenance()["name_wording"] == "aegis_prefer_list"
