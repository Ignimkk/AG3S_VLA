"""Interface contract (synthetic, no torch)."""
import numpy as np
import pytest

from Baseline.paper_baselines import interface as I


def test_query_prompt_modes():
    q = I.TargetQuery("pick the black bowl", "black bowl")
    assert q.prompt == "black bowl"
    assert I.TargetQuery("pick the black bowl", "black bowl", prompt_mode="instruction").prompt == "pick the black bowl"
    with pytest.raises(ValueError):
        I.TargetQuery("x", "y", prompt_mode="noun")


def test_not_found_is_empty_and_valid():
    r = I.TargetMaskResult.not_found("head", (4, 5, 3), 1.5, boxes=np.zeros((0, 4)), reason="no_box")
    assert r.mask.shape == (4, 5) and r.mask.dtype == bool and not r.mask.any() and not r.found
    assert r.extras["not_found_reason"] == "no_box"
    I.validate_results({"head": r}, {"head": np.zeros((4, 5, 3), np.uint8)})
    s = r.summary()
    assert s["mask_pixels"] == 0 and s["found"] is False


def test_validate_rejects_contract_breaks():
    rgb = {"head": np.zeros((4, 5, 3), np.uint8)}
    bad_shape = I.TargetMaskResult("head", np.zeros((5, 4), bool), True, 1.0, None, [], 0.0)
    with pytest.raises(ValueError):
        I.validate_results({"head": bad_shape}, rgb)
    invented = I.TargetMaskResult("head", np.ones((4, 5), bool), False, None, None, [], 0.0)
    with pytest.raises(ValueError):
        I.validate_results({"head": invented}, rgb)
    with pytest.raises(ValueError):
        I.validate_results({}, rgb)
    wrong_dtype = I.TargetMaskResult("head", np.ones((4, 5), np.uint8), True, 1.0, None, [], 0.0)
    with pytest.raises(ValueError):
        I.validate_results({"head": wrong_dtype}, rgb)


def test_select_top1_highest_score_lowest_index_on_tie():
    assert I.select_top1([]) is None
    assert I.select_top1([0.2, 0.9, 0.5]) == 1
    assert I.select_top1([0.7, 0.3, 0.7]) == 0


def test_check_rgb():
    with pytest.raises(ValueError):
        I.check_rgb(np.zeros((4, 5), np.uint8))
    with pytest.raises(ValueError):
        I.check_rgb(np.zeros((4, 5, 3), np.float32))


def test_registry_lazy_and_custom():
    assert {"gt_oracle", "grounded_sam", "grounded_sam2", "yoloe", "tokencut", "localization_heads"} <= set(I.available_providers())
    with pytest.raises(KeyError):
        I.get_provider("nope")

    @I.register_provider("_dummy")
    class Dummy:
        name = "_dummy"

        def __init__(self, k=1):
            self.k = k

        def provenance(self):
            return {"k": self.k}

        def reset(self):
            pass

        def predict(self, rgb, query, *, aux=None):
            return {c: I.TargetMaskResult.not_found(c, im.shape, 0.0) for c, im in rgb.items()}

    p = I.get_provider("_dummy", k=3)
    assert isinstance(p, I.TargetMaskProvider) and p.provenance() == {"k": 3}


def test_gt_oracle_contract():
    p = I.get_provider("gt_oracle")
    seg = np.full((6, 7), -1, np.int16)
    seg[1:3, 2:4] = 84
    seg[4, 4] = 125
    rgb = {"head": np.zeros((6, 7, 3), np.uint8), "left_wrist": np.zeros((6, 7, 3), np.uint8)}
    res = p.predict(rgb, I.TargetQuery("x", "bowl"), aux={"segmentation": {"head": seg, "left_wrist": np.full((6, 7), -1)},
                                                          "target_geom_ids": [84, 85]})
    I.validate_results(res, rgb)
    assert res["head"].found and res["head"].mask.sum() == 4
    assert not res["left_wrist"].found and res["left_wrist"].extras["not_found_reason"] == "target_not_visible"
    with pytest.raises(ValueError):
        p.predict(rgb, I.TargetQuery("x", "bowl"))


def test_phrase_from_object():
    from Baseline.paper_baselines.providers.gt_oracle import phrase_from_object

    assert phrase_from_object("akita_black_bowl_1") == "akita black bowl"
    assert phrase_from_object("plate") == "plate"
