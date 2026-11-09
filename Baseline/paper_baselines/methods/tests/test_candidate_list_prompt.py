"""detector_prompt="candidate_list" (2026-11-10): the GLM-4.5V naming step of AEGIS / multi-link CBF replaced by Grounding
DINO over the official prefer_list -> one obstacle per episode, cropped like the privileged-names path."""
import numpy as np
import pytest

from Baseline.paper_baselines.methods import safety_common as SC
from Baseline.paper_baselines.methods.tests.test_depth_trim_crop import BACK, FRONT, HEAD, inside, project_box, scene


class PhraseDetector:
    """Per-phrase (boxes, scores); the target phrase gets its own box."""

    def __init__(self, table: dict):
        self.table, self.calls = table, []

    def provenance(self):
        return {"detector": "phrase table (test)"}

    def detect(self, rgb, text, box_threshold, text_threshold):
        self.calls.append(text)
        bx, sc = self.table.get(text, ([], []))
        b = np.asarray(bx, float).reshape(-1, 4)
        return b, np.asarray(sc, float), [text] * len(b)


def make(det, **kw):
    return SC.ObstacleCloudSource("detector", detector=det, detector_prompt="candidate_list", robot_filter=False,
                                  crop_mode="depth_trim", min_points=20, **kw)


def test_prefer_list_matches_official_code():
    assert SC.aegis_prefer_list("safelibero_spatial") == ['yellow rectangular book', 'blue moka pot', 'red mug',
                                                          'white storage box', 'black wine bottle', 'red milk carton']
    assert SC.aegis_prefer_list("safelibero_long")[-1] == "gray rectangular binder"
    assert "gray rectangular binder" not in SC.aegis_prefer_list("libero_10")  # official: '"long" in task_suite_name'


def test_highest_scoring_candidate_is_the_one_obstacle():
    obs = scene()
    obs.extras["suite"] = "safelibero_spatial"
    fb, bb = project_box(HEAD, *FRONT, pad=0.25), project_box(HEAD, *BACK, pad=0.1)
    det = PhraseDetector({"red mug": ([fb], [0.42]), "black wine bottle": ([bb], [0.61])})
    clouds = make(det)(obs)
    assert list(clouds.points) == ["black wine bottle"]
    p = clouds.points["black wine bottle"]
    assert inside(p, *BACK).mean() > 0.9
    cl = clouds.info["candidate_list"]
    assert cl["chosen"] == "black wine bottle" and cl["scores"]["red mug"] == pytest.approx(0.42)
    assert not clouds.privileged


def test_target_box_is_not_a_candidate_and_no_detection_gives_no_obstacle():
    obs = scene()
    fb = project_box(HEAD, *FRONT, pad=0.25)
    det = PhraseDetector({"target thing": ([fb], [0.9]), "red mug": ([fb], [0.8])})  # the mug box IS the target
    clouds = make(det)(obs)
    assert clouds.info["candidate_list"]["chosen"] is None and clouds.points == {}


def test_other_prompts_unchanged():
    det = PhraseDetector({})
    with pytest.raises(ValueError):
        SC.ObstacleCloudSource("detector", detector=det, detector_prompt="nonsense")
    s = SC.ObstacleCloudSource("detector", detector=det, detector_prompt="all_objects")
    assert s.detector_prompt == "all_objects"


class FakeVLM:
    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def provenance(self):
        return {"vlm": "fake"}

    def name(self, rgb, instruction, prefer_list):
        self.seen.append((instruction, list(prefer_list)))
        return self.answer, {"raw": self.answer, "name": self.answer, "in_prefer_list": self.answer in prefer_list}


def test_vlm_name_is_the_one_grounding_phrase():
    from Baseline.paper_baselines.providers.vlm_namer import clean_answer, official_prompt

    obs = scene()
    obs.extras["suite"] = "safelibero_long"
    bb = project_box(HEAD, *BACK, pad=0.1)
    det = PhraseDetector({"black wine bottle": ([bb], [0.5]), "red mug": ([bb], [0.9])})
    vlm = FakeVLM("black wine bottle")
    clouds = SC.ObstacleCloudSource("detector", detector=det, detector_prompt="vlm_name", vlm=vlm, robot_filter=False,
                                    crop_mode="depth_trim", min_points=20)(obs)
    assert list(clouds.points) == ["black wine bottle"] and not clouds.privileged
    assert vlm.seen[0][1][-1] == "gray rectangular binder" and "red mug" not in det.calls  # only the VLM's phrase
    assert clouds.info["vlm_name"]["name"] == "black wine bottle"
    assert clean_answer('<|begin_of_box|> "Black Wine Bottle." ') == "black wine bottle"
    assert official_prompt("do x", ["a", "b"]).endswith("preferably from this list when applicable: ['a', 'b']. "
                                                         "Output only the object name, with no additional words.")
