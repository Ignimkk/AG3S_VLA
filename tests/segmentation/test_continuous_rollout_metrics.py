import numpy as np
from experiments.segmentation.eval_continuous_rollout import boundary_f1, box_iou


def test_bbox_iou_uses_visible_target_extent_and_handles_missing_box():
    gt = np.zeros((12, 12), bool)
    gt[2:6, 3:8] = True
    assert box_iou([3, 2, 8, 6], gt) == 1.0
    assert box_iou([8, 2, 10, 6], gt) == 0.0
    assert box_iou(None, gt) is None
    assert box_iou([0, 0, 1, 1], np.zeros_like(gt)) is None


def test_boundary_f1_handles_empty_and_pixel_tolerance():
    gt = np.zeros((20, 20), bool)
    gt[5:12, 5:12] = True
    assert boundary_f1(gt, gt) == 1.0
    assert boundary_f1(np.zeros_like(gt), gt) == 0.0
    assert boundary_f1(gt, np.zeros_like(gt)) is None
    shifted = np.zeros_like(gt)
    shifted[5:12, 7:14] = True
    assert boundary_f1(shifted, gt) == 1.0
    far = np.zeros_like(gt)
    far[12:19, 12:19] = True
    assert boundary_f1(far, gt) < 1.0
