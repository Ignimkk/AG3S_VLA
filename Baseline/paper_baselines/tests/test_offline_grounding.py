"""offline_grounding wiring: label LUT, ATS slicing, arm parsing (synthetic); one real recorded frame if present."""
import json
import pathlib

import numpy as np
import pytest

from Baseline.paper_baselines.eval import metrics as M
from Baseline.paper_baselines.eval import offline_grounding as OG

SL3 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark/20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0")
SCENE = pathlib.Path(__file__).resolve().parents[3] / "outputs/verify/SL4/raw/scene"


def test_label_lut_roles():
    bodyid = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    names = ["world", "table", "robot0_link0", "mount0_base", "bowl_1_main", "plate_1_main", "cup_1_main", "cab_1_main"]
    target = {"target": "bowl_1", "destination": "plate_1", "fixtures": ["main_table", "cab_1"],
              "instances_geom_ids": {"bowl_1": [4], "plate_1": [5], "cup_1": [6], "main_table": [1], "cab_1": [7]}}
    lut = OG.label_lut(target, bodyid, names)
    C = M.CLASSES
    assert [C[i] for i in lut[:8]] == ["background", "support", "robot", "mount", "target", "destination", "other_object", "fixture"]
    seg = np.array([[-1, 4], [2, 7]])
    assert [C[i] for i in OG.seg_labels(seg, lut).ravel()] == ["background", "target", "robot", "fixture"]


def test_parse_arm():
    assert OG.parse_arm("grounded_sam2:track") == ("grounded_sam2", {"mode": "track"})
    assert OG.parse_arm("yoloe") == ("yoloe", {})


def test_ats_points_slice(tmp_path):
    run = tmp_path / "run"
    (run / "perception").mkdir(parents=True)
    np.savez(run / "perception" / "s_perception.npz", obs_index=np.array([10, 15]), has_target=np.array([True, False]),
             target_points=np.arange(15, dtype=np.float32).reshape(5, 3), target_points_offsets=np.array([0, 3, 5]),
             t_process_s=np.array([0.1, 0.2]))
    a = OG.ATSRun(run)
    p = a.points("s", 10)
    assert p["found"] and p["points"].shape == (3, 3) and p["latency_ms"] == pytest.approx(100.0)
    assert a.points("s", 15)["points"].shape == (2, 3) and not a.points("s", 15)["found"]
    assert a.points("s", 99) is None and a.points("missing", 10) is None


@pytest.mark.skipif(not (SL3 / "episodes/libero_spatial_t0_e0_geom.npz").exists() or not SCENE.exists(),
                    reason="recorded SL3 dev episode / SL4 scene GT not present")
def test_real_recorded_frame_wiring():
    ep_meta = next(e for e in OG.load_episodes(SL3) if e["stem"] == "libero_spatial_t0_e0")
    ep = OG.Episode(ep_meta, OG.CAMERAS, SCENE)
    assert ep.target["target"] == "akita_black_bowl_1" and ep.eval_frames()[:2] == [10, 15]
    fr = ep.frame(10)
    for c in OG.CAMERAS:
        assert fr["rgb"][c].shape == fr["depth_m"][c].shape + (3,) == fr["seg"][c].shape + (3,)
    q = OG.query_for(ep, "phrase", {})
    assert q.target_phrase == "akita black bowl" and "black bowl" in q.instruction
    from Baseline.paper_baselines.interface import get_provider

    res = get_provider("gt_oracle").predict(fr["rgb"], q, aux={"segmentation": fr["seg"], "target_geom_ids": ep.target_geoms.tolist()})
    grid = M.build_eval_grid([{"depth_m": fr["depth_m"][c], "K": fr["K"][c], "T_base_cam": fr["T"][c], "labels": fr["labels"][c]}
                              for c in OG.CAMERAS])
    row = OG.evaluate_masks(res, fr, grid, ep, "none", OG.CAMERAS)
    assert row["cameras"]["head"]["2d_iou"] == 1.0  # oracle mask == GT label mask by construction
    assert row["3d_recall"] == pytest.approx(1.0)
    assert 0 < row["lift_within_10mm"] <= 1 and row["found_any"]
    json.dumps(row, default=float)
